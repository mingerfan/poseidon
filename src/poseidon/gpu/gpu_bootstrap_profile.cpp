#include "poseidon/gpu/gpu_bootstrap_profile.h"

#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/advance/homomorphic_mod.h"
#include "poseidon/basics/util/ntt.h"
#include "poseidon/basics/util/rlwe.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"
#include "poseidon/keygenerator.h"
#include "poseidon/poseidon_context.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace poseidon
{
namespace gpu
{
namespace
{

std::uint32_t log2_power_of_two(std::uint32_t value, const char *name)
{
    if (value == 0 || (value & (value - 1)) != 0)
    {
        throw std::invalid_argument(std::string(name) + " must be a power of two");
    }
    std::uint32_t result = 0;
    while (value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

LinearMatrixGroup make_coeff_to_slot_matrix_group(
    const PoseidonContext &context,
    CKKSEncoder &encoder,
    double scaling,
    const GpuBootstrapProfileConfig &config,
    double input_scale,
    double min_scale)
{
    const std::size_t matrix_depth = config.c2s_layer_groups.empty()
        ? 3
        : config.c2s_layer_groups.size();
    HomomorphicDFTMatrixLiteral literal(
        encode,
        context.parameters_literal()->log_n(),
        context.parameters_literal()->log_slots(),
        static_cast<std::uint32_t>(
            context.parameters_literal()->q().size() - 1),
        std::vector<std::uint32_t>(matrix_depth, 1),
        /*repack_imag_to_real=*/true,
        scaling,
        /*bit_reversed=*/false,
        config.c2s_log_bsgs_ratio,
        config.c2s_layer_groups,
        config.c2s_direct_layer_threshold);

    LinearMatrixGroup result;
    literal.create_dynamic(
        result,
        encoder,
        input_scale,
        min_scale,
        min_scale,
        /*value_normalization=*/1.0);
    return result;
}

LinearMatrixGroup make_slot_to_coeff_matrix_group(
    const PoseidonContext &context,
    CKKSEncoder &encoder,
    std::uint32_t level_start,
    double scaling,
    const GpuBootstrapProfileConfig &config,
    double input_scale,
    double min_scale,
    double value_normalization = 1.0)
{
    HomomorphicDFTMatrixLiteral literal(
        decode,
        context.parameters_literal()->log_n(),
        context.parameters_literal()->log_slots(),
        level_start,
        std::vector<std::uint32_t>(3, 1),
        /*repack_imag_to_real=*/true,
        scaling,
        /*bit_reversed=*/false,
        config.s2c_log_bsgs_ratio);

    LinearMatrixGroup result;
    literal.create_dynamic(
        result,
        encoder,
        input_scale,
        min_scale,
        min_scale,
        value_normalization);
    return result;
}

std::vector<std::uint32_t> dft_stage_rescale_counts(
    const LinearMatrixGroup &matrix_group)
{
    if (!matrix_group.rescale_counts().empty())
    {
        return matrix_group.rescale_counts();
    }
    return std::vector<std::uint32_t>(
        matrix_group.data().size(),
        std::max(matrix_group.step(), std::uint32_t{1}));
}

std::size_t consumed_q_count(const LinearMatrixGroup &matrix_group)
{
    const auto counts = dft_stage_rescale_counts(matrix_group);
    return std::accumulate(counts.begin(), counts.end(), std::size_t{0});
}

double planned_dft_output_scale(
    const PoseidonContext &context,
    double input_scale,
    const LinearMatrixGroup &matrix_group)
{
    if (matrix_group.data().empty())
    {
        throw std::invalid_argument("GPU bootstrap DFT matrix group is empty");
    }
    const auto first = context.crt_context()->first_context_data();
    if (!first)
    {
        throw std::invalid_argument("GPU bootstrap DFT scale plan has no context");
    }

    const auto counts = dft_stage_rescale_counts(matrix_group);
    std::size_t q_count =
        static_cast<std::size_t>(matrix_group.data().front().level) + 1;
    if (q_count == 0 || q_count > first->coeff_modulus().size())
    {
        throw std::invalid_argument(
            "GPU bootstrap DFT scale plan has an invalid start level");
    }

    double scale = input_scale;
    for (std::size_t stage = 0; stage < matrix_group.data().size(); ++stage)
    {
        scale *= matrix_group.data()[stage].scale;
        if (counts[stage] >= q_count)
        {
            throw std::invalid_argument(
                "GPU bootstrap DFT scale plan consumes its modulus chain");
        }
        for (std::uint32_t drop = 0; drop < counts[stage]; ++drop)
        {
            scale /= static_cast<double>(
                first->coeff_modulus().at(q_count - 1).value());
            --q_count;
        }
    }
    if (!(scale > 0.0) || !std::isfinite(scale))
    {
        throw std::invalid_argument("GPU bootstrap DFT output scale is invalid");
    }
    return scale;
}

GaloisKeys make_galois_keys(
    const PoseidonContext &context,
    KeyGenerator &key_generator,
    const LinearMatrixGroup &coeff_to_slot,
    const LinearMatrixGroup &slot_to_coeff)
{
    std::set<int> steps{0};
    steps.insert(
        coeff_to_slot.rot_index().begin(), coeff_to_slot.rot_index().end());
    steps.insert(
        slot_to_coeff.rot_index().begin(), slot_to_coeff.rot_index().end());

    const auto galois_tool = context.crt_context()->galois_tool();
    std::vector<std::uint32_t> elements;
    elements.reserve(steps.size());
    for (const int step : steps)
    {
        elements.push_back(galois_tool->get_elt_from_step(step));
    }

    GaloisKeys result;
    key_generator.create_galois_keys(elements, result);
    return result;
}

std::vector<std::size_t> required_dft_key_q_counts(
    std::size_t input_q_count,
    const LinearMatrixGroup &matrix_group,
    bool include_post_dft_conjugation)
{
    const auto stage_counts = dft_stage_rescale_counts(matrix_group);
    std::vector<std::size_t> result;
    result.reserve(
        matrix_group.data().size() +
        (include_post_dft_conjugation ? 1 : 0));

    std::size_t q_count = input_q_count;
    for (const auto stage_drop : stage_counts)
    {
        result.push_back(q_count);
        if (stage_drop == 0 || stage_drop >= q_count)
        {
            throw std::invalid_argument(
                "GPU bootstrap DFT key plan consumes its modulus chain");
        }
        q_count -= stage_drop;
    }
    if (include_post_dft_conjugation)
    {
        result.push_back(q_count);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::size_t> merge_q_counts(
    std::vector<std::size_t> lhs,
    const std::vector<std::size_t> &rhs)
{
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    std::sort(lhs.begin(), lhs.end());
    lhs.erase(std::unique(lhs.begin(), lhs.end()), lhs.end());
    return lhs;
}

// The ephemeral secret of sparse-secret encapsulation, in the key context.
SecretKey make_sparse_secret(const PoseidonContext &context, std::size_t hamming_weight)
{
    const auto global_context_data = context.crt_context();
    const auto &key_context = *global_context_data->key_context_data();
    const auto parms_id = global_context_data->key_parms_id();
    const std::size_t degree = key_context.parms().degree();
    const std::size_t modulus_count = key_context.coeff_modulus().size();
    SecretKey secret;
    secret.data().resize(context, parms_id, degree * modulus_count);
    util::sample_poly_ternary_with_hamming(context.random_generator()->create(), context,
                                           parms_id, hamming_weight, secret.data().data());
    util::RNSIter coefficients(secret.data().data(), degree);
    util::ntt_negacyclic_harvey(coefficients, modulus_count,
                                global_context_data->small_ntt_tables());
    secret.parms_id() = parms_id;
    return secret;
}

// A single KSwitch key in the relinearization-key layout (key index 0).
std::shared_ptr<GpuEvaluationKeyData> upload_switch_key(
    const KSwitchKeys &key, int cuda_device_id)
{
    RelinKeys layout;
    static_cast<KSwitchKeys &>(layout) = key;
    return std::make_shared<GpuEvaluationKeyData>(
        GpuUploader::upload_relin_keys(layout, cuda_device_id));
}

// Context, GPU tables and evaluator of the small q0 * P_low modulus on which
// the dense -> ephemeral switch runs.
struct SparseLowRuntime
{
    explicit SparseLowRuntime(const ParametersLiteral &literal)
        : parameters(literal), context(parameters) {}
    ParametersLiteral parameters;
    PoseidonContext context;
    std::unique_ptr<GpuParameterData> gpu_parameters;
    std::unique_ptr<GpuEvaluator> evaluator;
};

// The first q_keep Q limbs and first p_keep P limbs of a key-context secret
// (NTT form; the per-prime NTT tables are the same in both contexts).
SecretKey project_secret(const SecretKey &source, std::size_t degree,
                         std::size_t global_q, std::size_t global_p,
                         const PoseidonContext &low, std::size_t q_keep, std::size_t p_keep)
{
    if (source.data().coeff_count() != degree * (global_q + global_p))
        throw std::logic_error("GPU bootstrap secret storage does not match Q/P parameters");
    SecretKey result;
    const auto key_id = low.crt_context()->key_parms_id();
    result.data().resize(low, key_id, degree * (q_keep + p_keep));
    std::copy_n(source.data().data(), degree * q_keep, result.data().data());
    std::copy_n(source.data().data() + degree * global_q, degree * p_keep,
                result.data().data() + degree * q_keep);
    result.parms_id() = key_id;
    return result;
}

// POSEIDON_GPU_SSE_LOW_P: special primes of the dense -> ephemeral key's own
// modulus (default 2, i.e. q0 * p0 * p1 with the two q0 primes, about 126
// bits); 0 keeps the key on the full Q*P modulus (the previous behaviour).
std::size_t sparse_low_p_count()
{
    const char *raw = std::getenv("POSEIDON_GPU_SSE_LOW_P");
    if (raw == nullptr || *raw == '\0')
        return 2;
    char *end = nullptr;
    const auto value = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || value > 64)
        throw std::invalid_argument("POSEIDON_GPU_SSE_LOW_P must be an integer in [0, 64]");
    return static_cast<std::size_t>(value);
}

void install_sparse_encapsulation(
    const PoseidonContext &context,
    KeyGenerator &key_generator,
    int cuda_device_id,
    std::uint32_t hamming_weight,
    GpuBootstrapData &data,
    std::size_t raised_p_count = 0)
{
    if (hamming_weight == 0)
        return;
    const auto degree = context.crt_context()->key_context_data()->parms().degree();
    if (hamming_weight > degree)
        throw std::invalid_argument("GPU bootstrap sparse secret weight exceeds the ring degree");
    const SecretKey sparse = make_sparse_secret(context, hamming_weight);
    const SecretKey &dense = key_generator.secret_key();
    // The ephemeral -> dense key encrypts the sparse secret under the dense
    // one. It runs at the full Q level, on that level's special modulus: the
    // context's P, or a P prefix when the Boot uses a level-aware P there.
    {
        const auto &global_literal = *context.parameters_literal();
        const std::size_t global_q_count = global_literal.q().size();
        const std::size_t global_p_count = global_literal.p().size();
        if (raised_p_count == 0 || raised_p_count == global_p_count)
        {
            data.sparse_switch_to_dense = upload_switch_key(
                key_generator.create_switch_key(sparse, dense), cuda_device_id);
        }
        else
        {
            if (raised_p_count > global_p_count)
                throw std::invalid_argument("GPU bootstrap raised-level P exceeds the global P");
            ParametersLiteral literal = global_literal;
            literal.set_modulus(global_literal.q(),
                std::vector<Modulus>(global_literal.p().begin(),
                    global_literal.p().begin() + static_cast<std::ptrdiff_t>(raised_p_count)));
            const PoseidonContext top(literal);
            const SecretKey dense_top = project_secret(
                dense, degree, global_q_count, global_p_count, top, global_q_count, raised_p_count);
            const SecretKey sparse_top = project_secret(
                sparse, degree, global_q_count, global_p_count, top, global_q_count, raised_p_count);
            KeyGenerator top_keygen(top, dense_top);
            data.sparse_switch_to_dense = upload_switch_key(
                top_keygen.create_switch_key(sparse_top, dense_top), cuda_device_id);
        }
    }

    const std::size_t low_p = sparse_low_p_count();
    const auto &global = *context.parameters_literal();
    const std::size_t global_q = global.q().size();
    const std::size_t global_p = global.p().size();
    const std::size_t low_q = std::size_t{global.q0_level()} + 1;
    if (low_p == 0)
    {
        data.sparse_switch_to_ephemeral = upload_switch_key(
            key_generator.create_switch_key(dense, sparse), cuda_device_id);
        std::cout << "GPU_BOOT_SSE dense_to_ephemeral_modulus=full q=" << global_q
                  << " p=" << global_p << '\n';
        return;
    }
    if (low_p > global_p || low_q >= global_q)
        throw std::invalid_argument("GPU bootstrap sparse low modulus exceeds the global Q/P");

    ParametersLiteral literal = global;
    literal.set_modulus(
        std::vector<Modulus>(global.q().begin(),
                             global.q().begin() + static_cast<std::ptrdiff_t>(low_q)),
        std::vector<Modulus>(global.p().begin(),
                             global.p().begin() + static_cast<std::ptrdiff_t>(low_p)));
    auto runtime = std::make_shared<SparseLowRuntime>(literal);
    const SecretKey dense_low =
        project_secret(dense, degree, global_q, global_p, runtime->context, low_q, low_p);
    const SecretKey sparse_low =
        project_secret(sparse, degree, global_q, global_p, runtime->context, low_q, low_p);
    KeyGenerator low_keygen(runtime->context, dense_low);
    data.sparse_switch_to_ephemeral = upload_switch_key(
        low_keygen.create_switch_key(dense_low, sparse_low), cuda_device_id);
    runtime->gpu_parameters = std::make_unique<GpuParameterData>(
        runtime->context, cuda_device_id, std::vector<std::size_t>{low_q});
    runtime->evaluator = std::make_unique<GpuEvaluator>(*runtime->gpu_parameters);
    if (runtime->context.crt_context()->parms_id_map().at(low_q - 1) !=
        context.crt_context()->parms_id_map().at(low_q - 1))
        throw std::logic_error("GPU bootstrap sparse low context is incompatible with q0");
    double bits = 0;
    for (const auto &q : runtime->parameters.q()) bits += std::log2(double(q.value()));
    for (const auto &p : runtime->parameters.p()) bits += std::log2(double(p.value()));
    std::cout << "GPU_BOOT_SSE dense_to_ephemeral_modulus=low q=" << low_q << " p=" << low_p
              << " bits=" << bits << " ephemeral_h=" << hamming_weight << '\n';
    data.sparse_low_evaluator = runtime->evaluator.get();
    data.sparse_low_owner = std::move(runtime);
}

// Special-modulus size of the manual Boot's key switches at Q level q (see
// GpuResNet20ManualOptions::level_special_modulus).
std::size_t level_special_p_count(
    const ParametersLiteral &global,
    std::size_t q,
    const GpuResNet20ManualOptions &options)
{
    using Mode = GpuResNet20ManualOptions::LevelSpecialModulus;
    const std::size_t global_p = global.p().size();
    if (options.level_special_modulus == Mode::Global)
        return global_p;
    std::size_t p = q <= options.dnum1_max_q ? q : (q + 1) / 2;
    p = std::clamp<std::size_t>(p, 1, global_p);
    if (options.level_special_modulus != Mode::Secure)
        return p;
    double q_bits = 0;
    for (std::size_t i = 0; i < q; ++i)
        q_bits += std::log2(static_cast<double>(global.q()[i].value()));
    for (std::size_t dnum = (q + p - 1) / p; dnum <= q; ++dnum)
    {
        const std::size_t candidate = (q + dnum - 1) / dnum;
        if (candidate > global_p)
            continue;
        double p_bits = 0;
        for (std::size_t j = 0; j < candidate; ++j)
            p_bits += std::log2(static_cast<double>(global.p()[j].value()));
        if (q_bits + p_bits <= options.secure_modulus_bits)
            return candidate;
    }
    throw std::invalid_argument("secure Boot special modulus: Q" + std::to_string(q) +
                                " exceeds the modulus budget");
}

// Per-level keys and parameter levels for a level-aware special modulus:
// levels are grouped by their P; each group gets one Q-prefix + P-prefix
// sub-context covering its highest level, its Galois keys (the group's
// elements) and/or relinearization key, and the parameter levels to adopt.
// A group whose P is the context's own keeps the main keys and levels, and
// the main Galois key set is regenerated with only the elements those levels
// use (or just the conjugation key when every level has its own P).
void install_level_special_modulus(
    const PoseidonContext &context,
    KeyGenerator &key_generator,
    int cuda_device_id,
    const GpuResNet20ManualOptions &options,
    const std::map<std::size_t, std::set<std::uint32_t>> &galois_by_level,
    const std::vector<std::size_t> &relin_levels,
    GpuGaloisKeysData &galois_keys,
    GpuRelinKeysData &relin_keys,
    std::vector<GpuLevelInfo> &level_parameters)
{
    const auto &global = *context.parameters_literal();
    const std::size_t degree = context.crt_context()->key_context_data()->parms().degree();
    const std::size_t global_q = global.q().size();
    const std::size_t global_p = global.p().size();
    struct Group
    {
        std::size_t top_q = 0;
        std::set<std::size_t> galois_levels, relin_levels, levels;
        std::set<std::uint32_t> elements;
    };
    std::map<std::size_t, Group> groups;
    const auto group_of = [&](std::size_t q) -> Group & {
        if (q == 0 || q > global_q)
            throw std::invalid_argument("level-aware Boot key level is outside the Q chain");
        auto &group = groups[level_special_p_count(global, q, options)];
        group.top_q = std::max(group.top_q, q);
        group.levels.insert(q);
        return group;
    };
    for (const auto &[q, elements] : galois_by_level)
    {
        auto &group = group_of(q);
        group.galois_levels.insert(q);
        group.elements.insert(elements.begin(), elements.end());
    }
    for (const auto q : relin_levels)
        group_of(q).relin_levels.insert(q);

    std::map<std::size_t, std::shared_ptr<const GpuEvaluationKeyData>> galois_overrides;
    std::map<std::size_t, std::shared_ptr<const GpuEvaluationKeyData>> relin_overrides;
    std::set<std::uint32_t> main_elements;
    std::vector<std::size_t> main_galois_levels;
    double max_bits = 0;
    for (const auto &[p, group] : groups)
    {
        double bits = 0;
        for (std::size_t i = 0; i < group.top_q; ++i) bits += std::log2(double(global.q()[i].value()));
        for (std::size_t j = 0; j < p; ++j) bits += std::log2(double(global.p()[j].value()));
        max_bits = std::max(max_bits, bits);
        std::cout << "GPU_BOOT_LEVEL_P p=" << p << " dnum=" << (group.top_q + p - 1) / p
                  << " top_q=" << group.top_q << " levels=" << group.levels.size()
                  << " galois_levels=" << group.galois_levels.size()
                  << " elements=" << group.elements.size()
                  << " relin_levels=" << group.relin_levels.size()
                  << " key_modulus_bits=" << bits
                  << (p == global_p ? " shared_with_main=true" : "") << '\n';
        if (p == global_p)
        {
            main_elements.insert(group.elements.begin(), group.elements.end());
            main_galois_levels.insert(main_galois_levels.end(),
                group.galois_levels.begin(), group.galois_levels.end());
            continue;
        }
        ParametersLiteral literal = global;
        literal.set_modulus(
            std::vector<Modulus>(global.q().begin(),
                global.q().begin() + static_cast<std::ptrdiff_t>(group.top_q)),
            std::vector<Modulus>(global.p().begin(),
                global.p().begin() + static_cast<std::ptrdiff_t>(p)));
        const PoseidonContext sub(literal);
        const SecretKey dense = project_secret(
            key_generator.secret_key(), degree, global_q, global_p, sub, group.top_q, p);
        KeyGenerator sub_keygen(sub, dense);
        if (!group.galois_levels.empty())
        {
            GaloisKeys cpu;
            sub_keygen.create_galois_keys(
                std::vector<std::uint32_t>(group.elements.begin(), group.elements.end()), cpu);
            auto uploaded = std::make_shared<GpuGaloisKeysData>(
                GpuUploader::upload_double_hoist_galois_keys(cpu, cuda_device_id));
            GpuUploader::prepare_key_views_for_q_counts(*uploaded,
                std::vector<std::size_t>(group.galois_levels.begin(), group.galois_levels.end()));
            for (const auto q : group.galois_levels) galois_overrides[q] = uploaded;
        }
        if (!group.relin_levels.empty())
        {
            RelinKeys cpu;
            sub_keygen.create_relin_keys(cpu);
            auto uploaded = std::make_shared<GpuRelinKeysData>(
                GpuUploader::upload_relin_keys(cpu, cuda_device_id));
            GpuUploader::prepare_key_views_for_q_counts(*uploaded,
                std::vector<std::size_t>(group.relin_levels.begin(), group.relin_levels.end()));
            for (const auto q : group.relin_levels) relin_overrides[q] = uploaded;
        }
        GpuParameterData donor(sub, cuda_device_id,
            std::vector<std::size_t>(group.levels.begin(), group.levels.end()));
        for (const auto q : group.levels)
        {
            if (sub.crt_context()->parms_id_map().at(q - 1) !=
                context.crt_context()->parms_id_map().at(q - 1))
                throw std::logic_error("level-aware Boot sub-context is incompatible with the Q chain");
            level_parameters.push_back(donor.take_level_by_q_count(q));
        }
    }
    // The main Galois set keeps only what the shared-P levels use.
    main_elements.insert(context.crt_context()->galois_tool()->get_elt_from_step(0));
    {
        GaloisKeys cpu;
        key_generator.create_galois_keys(
            std::vector<std::uint32_t>(main_elements.begin(), main_elements.end()), cpu);
        galois_keys = GpuUploader::upload_double_hoist_galois_keys(cpu, cuda_device_id);
        if (!main_galois_levels.empty())
            GpuUploader::prepare_key_views_for_q_counts(galois_keys, main_galois_levels);
    }
    galois_keys.level_keys = std::make_shared<const std::map<std::size_t,
        std::shared_ptr<const GpuEvaluationKeyData>>>(std::move(galois_overrides));
    relin_keys.level_keys = std::make_shared<const std::map<std::size_t,
        std::shared_ptr<const GpuEvaluationKeyData>>>(std::move(relin_overrides));
    std::cout << "GPU_BOOT_LEVEL_P mode="
              << (options.level_special_modulus ==
                          GpuResNet20ManualOptions::LevelSpecialModulus::Secure
                      ? "secure" : "performance")
              << " groups=" << groups.size() << " adopted_levels=" << level_parameters.size()
              << " main_galois_elements=" << main_elements.size()
              << " max_key_modulus_bits=" << max_bits << '\n';
}

void validate_config(
    const PoseidonContext &context,
    int cuda_device_id,
    const GpuBootstrapProfileConfig &config)
{
    if (config.profile_id.empty())
    {
        throw std::invalid_argument("GPU bootstrap profile id is empty");
    }
    if (cuda_device_id < 0)
    {
        throw std::invalid_argument("GPU bootstrap CUDA device id is negative");
    }
    if (config.eval_mod_log_scale == 0 || config.eval_mod_log_scale >= 63 ||
        config.eval_mod_sine_degree == 0 ||
        config.eval_mod_generation_degree < config.eval_mod_sine_degree ||
        config.eval_mod_log_split == 0 || config.eval_mod_log_split >= 31 ||
        config.logical_rescale_count == 0 ||
        config.linear_transform_mode == GpuLinearTransformMode::SingleHoistBsgs)
    {
        throw std::invalid_argument("GPU bootstrap profile configuration is invalid");
    }
    if (config.eval_mod_fixed_degree_refit &&
        config.eval_mod_truncate_degree.has_value())
    {
        throw std::invalid_argument(
            "GPU bootstrap fixed-degree refit and truncation are mutually exclusive");
    }
    if (config.eval_mod_truncate_degree.has_value() &&
        *config.eval_mod_truncate_degree != config.eval_mod_sine_degree)
    {
        throw std::invalid_argument(
            "GPU bootstrap truncate degree must equal sine degree");
    }

    const auto parameters = context.parameters_literal();
    if (!parameters || parameters->q().empty() || parameters->p().empty() ||
        parameters->q0_level() + std::size_t{1} >= parameters->q().size())
    {
        throw std::invalid_argument(
            "GPU bootstrap context has an invalid Q/P modulus chain");
    }
    if (config.eval_mod_log_scale < parameters->log_scale() ||
        config.eval_mod_log_scale - parameters->log_scale() >= 32 ||
        (config.project_real && config.eval_mod_log_scale == parameters->log_scale()))
    {
        throw std::invalid_argument("GPU bootstrap output normalization is outside uint32 range");
    }
    log2_power_of_two(config.bootstrap_ratio, "GPU bootstrap ratio");
}

} // namespace

GpuBootstrapProfile GpuBootstrapProfileBuilder::build(
    const PoseidonContext &context,
    KeyGenerator &key_generator,
    int cuda_device_id,
    const GpuBootstrapProfileConfig &config)
{
    validate_config(context, cuda_device_id, config);
    gpu_check_cuda(cudaSetDevice(cuda_device_id), "cudaSetDevice");

    const auto parameters = context.parameters_literal();
    const std::size_t full_q_count = parameters->q().size();
    const std::uint32_t q0_level = parameters->q0_level();
    const double eval_mod_scale =
        std::exp2(static_cast<double>(config.eval_mod_log_scale));
    const std::uint32_t log_message_ratio =
        log2_power_of_two(config.bootstrap_ratio, "GPU bootstrap ratio");

    CKKSEncoder encoder(context);
    EvalModPoly eval_mod_poly(
        context,
        CosDiscrete,
        eval_mod_scale,
        /*level_start=*/0,
        log_message_ratio,
        config.eval_mod_double_angle,
        config.eval_mod_k,
        config.eval_mod_arcsine_degree,
        config.eval_mod_generation_degree);
    if (config.eval_mod_fixed_degree_refit)
    {
        eval_mod_poly.refit_discrete_cosine_fixed_degree(
            config.eval_mod_sine_degree);
    }
    else if (config.eval_mod_truncate_degree.has_value())
    {
        eval_mod_poly.truncate_sine_polynomial(
            *config.eval_mod_truncate_degree);
    }
    if (config.eval_mod_full_coefficients)
    {
        eval_mod_poly.enable_full_cosine_coefficients();
    }
    if (eval_mod_poly.sine_poly().degree() != config.eval_mod_sine_degree)
    {
        throw std::runtime_error(
            "GPU bootstrap EvalMod polynomial degree does not match profile");
    }

    const double c2s_scaling =
        eval_mod_poly.q_div() /
        (eval_mod_poly.k() * eval_mod_poly.sc_fac() * eval_mod_poly.q_diff());
    const double s2c_scaling =
        parameters->scale() /
        (eval_mod_poly.scaling_factor() / eval_mod_poly.message_ratio());
    const double raised_scale = eval_mod_scale;

    auto coeff_to_slot = make_coeff_to_slot_matrix_group(
        context,
        encoder,
        c2s_scaling,
        config,
        raised_scale,
        eval_mod_scale);
    const std::size_t c2s_consumed = consumed_q_count(coeff_to_slot);
    if (c2s_consumed >= full_q_count)
    {
        throw std::invalid_argument(
            "GPU bootstrap CoeffToSlot consumes the complete modulus chain");
    }
    const std::size_t c2s_output_q_count = full_q_count - c2s_consumed;
    const auto c2s_output_parms_id = context.crt_context()->parms_id_map().at(
        static_cast<std::uint32_t>(c2s_output_q_count - 1));
    const double c2s_output_scale = planned_dft_output_scale(
        context, raised_scale, coeff_to_slot);
    eval_mod_poly.set_level_start(
        static_cast<std::uint32_t>(c2s_output_q_count - 1));

    RelinKeys cpu_relin_keys;
    key_generator.create_relin_keys(cpu_relin_keys);
    auto relin_keys = std::make_shared<GpuRelinKeysData>(
        GpuUploader::upload_relin_keys(cpu_relin_keys, cuda_device_id));

    GpuEvalModUploadOptions eval_mod_options;
    eval_mod_options.dynamic_rescale = true;
    eval_mod_options.polynomial_log_split = config.eval_mod_log_split;
    eval_mod_options.flat_bsgs_b8 = false;
    eval_mod_options.virtual_degree_bound =
        config.eval_mod_virtual_degree_bound;
    eval_mod_options.lead_leaf_resplit = false;
    auto eval_mod = GpuUploader::upload_eval_mod_high_precision(
        eval_mod_poly,
        encoder,
        c2s_output_parms_id,
        cuda_device_id,
        relin_keys.get(),
        parms_id_zero,
        config.logical_rescale_count,
        nullptr,
        /*include_input_offset=*/true,
        std::numeric_limits<std::uint32_t>::max(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        /*fuse_leaf_terms_before_rescale=*/true,
        c2s_output_scale,
        /*metadata_only=*/false,
        eval_mod_options);
    if (config.eval_mod_sine_degree == 22 &&
        config.eval_mod_log_split == 2 &&
        config.eval_mod_virtual_degree_bound &&
        !config.eval_mod_full_coefficients)
    {
        std::vector<std::uint32_t> basis_degrees;
        basis_degrees.reserve(eval_mod.basis_steps.size());
        bool materialized_degree_bound = false;
        for (const auto &step : eval_mod.basis_steps)
        {
            basis_degrees.push_back(step.output_degree);
            materialized_degree_bound |= step.output_degree == 32;
        }
        const std::vector<std::uint32_t> expected_basis{2, 3, 4, 8, 16};
        if (eval_mod.polynomial_degree != 22 ||
            eval_mod.polynomial_log_split != 2 ||
            basis_degrees != expected_basis ||
            eval_mod.polynomial_blocks.size() != 6 ||
            eval_mod.polynomial_combine_steps.size() != 5 ||
            materialized_degree_bound)
        {
            throw std::runtime_error(
                "GPU bootstrap degree-22 baby-4 plan invariant failed");
        }
    }
    if (eval_mod.output_q_count == 0 ||
        eval_mod.output_q_count >= c2s_output_q_count)
    {
        throw std::runtime_error(
            "GPU bootstrap EvalMod produced an invalid output level");
    }
    const auto eval_mod_context = context.crt_context()->get_context_data(
        eval_mod.output_parms_id);
    if (!eval_mod_context ||
        eval_mod_context->coeff_modulus().size() != eval_mod.output_q_count)
    {
        throw std::runtime_error(
            "GPU bootstrap EvalMod output is absent from the context");
    }

    const double s2c_input_scale = eval_mod.output_scale;
    const auto build_slot_to_coeff = [&](std::size_t input_q_count) {
        const auto level = static_cast<std::uint32_t>(input_q_count - 1);
        auto group = make_slot_to_coeff_matrix_group(
            context, encoder, level, s2c_scaling, config, s2c_input_scale, eval_mod_scale);
        const double raw_output_scale = planned_dft_output_scale(
            context, s2c_input_scale, group);
        group = make_slot_to_coeff_matrix_group(
            context, encoder, level, s2c_scaling, config, s2c_input_scale, eval_mod_scale,
            eval_mod_scale / raw_output_scale);
        if (consumed_q_count(group) >= input_q_count)
        {
            throw std::invalid_argument(
                "GPU bootstrap SlotToCoeff consumes the remaining modulus chain");
        }
        return group;
    };
    std::size_t s2c_input_q_count = eval_mod.output_q_count;
    auto slot_to_coeff = build_slot_to_coeff(s2c_input_q_count);
    std::size_t output_q_count = s2c_input_q_count - consumed_q_count(slot_to_coeff);
    if (config.output_level.has_value())
    {
        const auto declared_q_count = static_cast<std::size_t>(*config.output_level) + 1;
        if (*config.output_level < 0 || output_q_count < declared_q_count)
        {
            throw std::invalid_argument(
                "GPU bootstrap cannot reach its declared output level");
        }
        while (output_q_count > declared_q_count)
        {
            s2c_input_q_count -= output_q_count - declared_q_count;
            slot_to_coeff = build_slot_to_coeff(s2c_input_q_count);
            output_q_count = s2c_input_q_count - consumed_q_count(slot_to_coeff);
        }
        if (output_q_count != declared_q_count)
        {
            throw std::runtime_error(
                "GPU bootstrap SlotToCoeff misses the declared output level");
        }
    }
    const auto s2c_input_parms_id = context.crt_context()->parms_id_map().at(
        static_cast<std::uint32_t>(s2c_input_q_count - 1));
    std::cerr << "GPU_BOOT_PROFILE id=" << config.profile_id
              << " c2s_output_q=" << c2s_output_q_count
              << " evalmod_output_q=" << eval_mod.output_q_count
              << " s2c_input_q=" << s2c_input_q_count
              << " output_level=" << output_q_count - 1
              << " sparse_hw=" << config.sparse_secret_hamming_weight << '\n';

    auto cpu_galois_keys = make_galois_keys(
        context, key_generator, coeff_to_slot, slot_to_coeff);
    auto uploaded_galois_keys = config.linear_transform_mode ==
            GpuLinearTransformMode::DoubleHoistBsgs
        ? GpuUploader::upload_double_hoist_galois_keys(
              cpu_galois_keys, cuda_device_id)
        : GpuUploader::upload_galois_keys(
              cpu_galois_keys, cuda_device_id);
    const auto c2s_key_q_counts = required_dft_key_q_counts(
        full_q_count, coeff_to_slot, /*include_post_dft_conjugation=*/true);
    const auto s2c_key_q_counts = required_dft_key_q_counts(
        s2c_input_q_count,
        slot_to_coeff,
        /*include_post_dft_conjugation=*/false);
    GpuUploader::prepare_key_views_for_q_counts(
        uploaded_galois_keys,
        merge_q_counts(c2s_key_q_counts, s2c_key_q_counts));
    auto galois_keys = std::make_shared<GpuGaloisKeysData>(
        std::move(uploaded_galois_keys));

    Plaintext minus_i;
    encoder.encode(
        std::complex<double>(0.0, -1.0),
        c2s_output_parms_id,
        1.0,
        minus_i);
    Plaintext plus_i;
    encoder.encode(
        std::complex<double>(0.0, 1.0),
        s2c_input_parms_id,
        1.0,
        plus_i);

    GpuBootstrapData bootstrap_data;
    if (s2c_input_q_count != eval_mod.output_q_count)
    {
        bootstrap_data.slot_to_coeff_input_q_count = s2c_input_q_count;
        bootstrap_data.slot_to_coeff_input_parms_id = s2c_input_parms_id;
    }
    install_sparse_encapsulation(context, key_generator, cuda_device_id,
                                 config.sparse_secret_hamming_weight, bootstrap_data);
    bootstrap_data.linear_transform_mode = config.linear_transform_mode;
    bootstrap_data.allow_environment_linear_transform_override = false;
    bootstrap_data.q0_parms_id = context.crt_context()->parms_id_map().at(q0_level);
    bootstrap_data.q0_over_message_ratio = std::exp2(std::round(std::log2(
        context.crt_context()->q0() /
        static_cast<double>(config.bootstrap_ratio))));
    bootstrap_data.raised_scale_override = raised_scale;
    bootstrap_data.slot_to_coeff_input_scale = s2c_input_scale;
    bootstrap_data.project_real = config.project_real;
    // S2C already includes context_scale * message_ratio / evalmod_scale.
    // Compensate its actual value factor, independently of message_ratio.
    bootstrap_data.output_ratio = std::uint32_t{1} <<
        (config.eval_mod_log_scale - parameters->log_scale());
    bootstrap_data.slot_to_coeff_output_scale = eval_mod_scale;
    if (config.linear_transform_mode ==
        GpuLinearTransformMode::DoubleHoistBsgs)
    {
        bootstrap_data.coeff_to_slot_matrix_qp =
            GpuUploader::upload_linear_matrix_group_qp(
                coeff_to_slot,
                context,
                cuda_device_id,
                std::max(coeff_to_slot.step(), std::uint32_t{1}),
                config.compress_linear_plaintexts);
        bootstrap_data.slot_to_coeff_matrix_qp =
            GpuUploader::upload_linear_matrix_group_qp(
                slot_to_coeff,
                context,
                cuda_device_id,
                std::max(slot_to_coeff.step(), std::uint32_t{1}),
                config.compress_linear_plaintexts);
        // Pointer tables only: what an expanded Boot's bsgs_group steps run.
        for (const auto &matrix : bootstrap_data.coeff_to_slot_matrix_qp.data())
            bootstrap_data.coeff_to_slot_groups_qp.push_back(
                GpuUploader::upload_double_hoist_group_views(matrix, cuda_device_id));
        for (const auto &matrix : bootstrap_data.slot_to_coeff_matrix_qp.data())
            bootstrap_data.slot_to_coeff_groups_qp.push_back(
                GpuUploader::upload_double_hoist_group_views(matrix, cuda_device_id));
    }
    else
    {
        bootstrap_data.coeff_to_slot_matrix =
            GpuUploader::upload_linear_matrix_group(
                coeff_to_slot, cuda_device_id);
        bootstrap_data.slot_to_coeff_matrix =
            GpuUploader::upload_linear_matrix_group(
                slot_to_coeff, cuda_device_id);
    }
    bootstrap_data.minus_i_plaintext =
        GpuUploader::upload_plaintext(minus_i, cuda_device_id);
    bootstrap_data.plus_i_plaintext =
        GpuUploader::upload_plaintext(plus_i, cuda_device_id);
    bootstrap_data.eval_mod = std::move(eval_mod);

    GpuBootstrapProfile result;
    result.profile_id = config.profile_id;
    result.cuda_device_id = cuda_device_id;
    result.input_level_min = static_cast<int>(q0_level + 1);
    result.input_level_max = static_cast<int>(full_q_count - 1);
    result.output_level = static_cast<int>(output_q_count - 1);
    result.output_scale_log2 = static_cast<int>(config.eval_mod_log_scale);
    result.bootstrap_data = std::move(bootstrap_data);
    result.relin_keys = std::move(relin_keys);
    result.galois_keys = std::move(galois_keys);
    return result;
}

GpuBootstrapProfile GpuBootstrapProfileBuilder::build_resnet20_manual(
    const PoseidonContext &context, KeyGenerator &key_generator, int cuda_device_id,
    const GpuResNet20ManualOptions &options)
{
    const auto parameters = context.parameters_literal();
    // The saved Q50 chain, or Q49 when the application leaves out one of the
    // convolution primes below the Boot (its merged-mask convolution uses two
    // levels). The Boot's own primes are the same physical primes either way:
    // ModRaise to Q(full), CtS down to Q(full-4), EvalMod output Q(full-19).
    const std::size_t full_q = parameters ? parameters->q().size() : 0;
    if (!parameters || parameters->log_n() != 16 || parameters->log_slots() != 15 ||
        (full_q != 50 && full_q != 49) || parameters->p().size() != 25 ||
        parameters->q0_level() != 1)
        throw std::invalid_argument("manual ResNet20 Boot requires N65536, Q50/P25 (or Q49/P25), q0 level 1 and 32768 slots");
    const std::size_t c2s_output_q = full_q - 4;
    const std::size_t output_q = full_q - 19;
    gpu_check_cuda(cudaSetDevice(cuda_device_id), "cudaSetDevice");
    CKKSEncoder encoder(context);
    constexpr double working_scale = double(std::uint64_t{1} << 45);
    constexpr double application_scale = double(std::uint64_t{1} << 40);
    if (options.double_angle == 0 || options.k == 0 || options.sine_degree == 0 ||
        options.log_split == 0)
        throw std::invalid_argument("manual ResNet20 Boot EvalMod options are invalid");
    EvalModPoly polynomial(context, CosDiscrete, working_scale, 0, 5, options.double_angle,
                           options.k, 0, options.sine_degree);
    if (options.full_coefficients)
        polynomial.enable_full_cosine_coefficients();
    GpuBootstrapProfileConfig config;
    const double c2s_scaling = polynomial.q_div() /
        (polynomial.k() * polynomial.sc_fac() * polynomial.q_diff());
    if (std::abs(options.k * context.crt_context()->q0() * c2s_scaling / working_scale - 1) > 1e-12)
        throw std::invalid_argument("manual ResNet20 Boot integer-period contract failed");
    auto c2s = make_coeff_to_slot_matrix_group(context, encoder, c2s_scaling,
        config, working_scale, working_scale);
    auto s2c = make_slot_to_coeff_matrix_group(context, encoder, 5, 1.0,
        config, application_scale, working_scale);
    const std::vector<std::uint32_t> expected_drops{1, 1, 2};
    if (dft_stage_rescale_counts(c2s) != expected_drops ||
        dft_stage_rescale_counts(s2c) != expected_drops)
        throw std::invalid_argument("manual ResNet20 Boot requires S2C/C2S drop pattern 1,1,2");
    const double c2s_scale = planned_dft_output_scale(context, working_scale, c2s);
    const auto c2s_id = context.crt_context()->parms_id_map().at(c2s_output_q - 1);
    polynomial.set_level_start(static_cast<std::uint32_t>(c2s_output_q - 1));
    GpuEvalModUploadOptions upload_options;
    upload_options.dynamic_rescale = true;
    upload_options.polynomial_log_split = options.log_split;
    upload_options.flat_bsgs_b8 = false;
    upload_options.virtual_degree_bound = false;
    upload_options.lead_leaf_resplit = false;
    const auto native = GpuUploader::upload_eval_mod_high_precision(
        polynomial, encoder, c2s_id, cuda_device_id, nullptr, parms_id_zero, 1,
        nullptr, true, std::numeric_limits<std::uint32_t>::max(), NAN, NAN,
        true, c2s_scale, true, upload_options);
    // Output Q31 (Q30 on the Q49 chain): a cheaper EvalMod may end higher;
    // the surplus is dropped (zero-copy) before the real/imag merge.
    if (native.output_q_count < output_q)
        throw std::invalid_argument("manual ResNet20 Boot EvalMod must end at or above Q(full-19)");
    const double alpha = application_scale / native.output_scale;
    if (!(alpha > 0 && alpha < 1))
        throw std::invalid_argument("manual ResNet20 Boot output fold is invalid");
    // The fold passes through every double-angle squaring.
    const double seed = std::pow(alpha, 1.0 / std::ldexp(1.0, static_cast<int>(options.double_angle)));
    Polynomial coefficients = polynomial.sine_poly();
    for (auto &coefficient : coefficients.data()) coefficient *= seed;

    RelinKeys cpu_relin;
    key_generator.create_relin_keys(cpu_relin);
    auto relin = std::make_shared<GpuRelinKeysData>(
        GpuUploader::upload_relin_keys(cpu_relin, cuda_device_id));
    auto eval_mod = GpuUploader::upload_eval_mod_high_precision(
        polynomial, encoder, c2s_id, cuda_device_id, relin.get(), parms_id_zero, 1,
        &coefficients, true, std::numeric_limits<std::uint32_t>::max(),
        polynomial.sqrt_2pi() * seed, NAN, true, c2s_scale, false, upload_options);
    if (eval_mod.output_q_count != native.output_q_count ||
        std::abs(eval_mod.output_scale / native.output_scale - 1) > 1e-12)
        throw std::runtime_error("manual ResNet20 output fold changed the Q/scale schedule");
    auto cpu_rotations = make_galois_keys(context, key_generator, c2s, s2c);
    auto rotations = std::make_shared<GpuGaloisKeysData>(
        GpuUploader::upload_double_hoist_galois_keys(cpu_rotations, cuda_device_id));
    GpuUploader::prepare_key_views_for_q_counts(*rotations,
        merge_q_counts(required_dft_key_q_counts(full_q, c2s, true),
            merge_q_counts(required_dft_key_q_counts(6, s2c, false), {output_q})));
    GpuBootstrapData data;
    data.linear_transform_mode = GpuLinearTransformMode::DoubleHoistBsgs;
    data.allow_environment_linear_transform_override = false;
    data.slot_to_coeff_first = true;
    data.input_q_count = 6;
    data.input_scale = application_scale;
    data.output_scale_multiplier = alpha;
    data.slot_to_coeff_output_scale = application_scale;
    data.q0_parms_id = context.crt_context()->parms_id_map().at(1);
    data.q0_over_message_ratio = std::exp2(std::round(std::log2(
        context.crt_context()->q0() / 32)));
    data.raised_scale_override = working_scale;
    data.project_real = true;
    data.output_ratio = 32;
    data.eval_mod = std::move(eval_mod);
    data.coeff_to_slot_matrix_qp = GpuUploader::upload_linear_matrix_group_qp(
        c2s, context, cuda_device_id, 1, true);
    data.slot_to_coeff_matrix_qp = GpuUploader::upload_linear_matrix_group_qp(
        s2c, context, cuda_device_id, 1, true);
    // Pointer tables only, over the compressed diagonals: what an expanded
    // Boot's bsgs_group steps run.
    for (const auto &matrix : data.coeff_to_slot_matrix_qp.data())
        data.coeff_to_slot_groups_qp.push_back(
            GpuUploader::upload_double_hoist_group_views(matrix, cuda_device_id));
    for (const auto &matrix : data.slot_to_coeff_matrix_qp.data())
        data.slot_to_coeff_groups_qp.push_back(
            GpuUploader::upload_double_hoist_group_views(matrix, cuda_device_id));
    Plaintext minus_i, plus_i;
    encoder.encode(std::complex<double>(0, -1), c2s_id, 1.0, minus_i);
    if (data.eval_mod.output_q_count > output_q)
    {
        data.slot_to_coeff_input_q_count = output_q;
        data.slot_to_coeff_input_parms_id =
            context.crt_context()->parms_id_map().at(output_q - 1);
    }
    encoder.encode(std::complex<double>(0, 1),
                   data.slot_to_coeff_input_q_count ? data.slot_to_coeff_input_parms_id
                                                    : data.eval_mod.output_parms_id,
                   1.0, plus_i);
    install_sparse_encapsulation(context, key_generator, cuda_device_id,
                                 options.sparse_secret_hamming_weight, data,
                                 level_special_p_count(*parameters, full_q, options));
    data.minus_i_plaintext = GpuUploader::upload_plaintext(minus_i, cuda_device_id);
    data.plus_i_plaintext = GpuUploader::upload_plaintext(plus_i, cuda_device_id);
    GpuBootstrapProfile result;
    if (options.level_special_modulus != GpuResNet20ManualOptions::LevelSpecialModulus::Global)
    {
        // Galois elements per Q level, from the uploaded double-hoist plans
        // (exactly the steps the GPU runs); every Galois level also gets the
        // conjugation key. Relinearization levels come from the EvalMod plan.
        const auto galois_tool = context.crt_context()->galois_tool();
        std::map<std::size_t, std::set<std::uint32_t>> galois_by_level;
        // A matrix runs at the level its plaintexts were encoded for
        // (parms_id_map index `level`, i.e. Q count level + 1); the layer
        // order of a group is not the execution order.
        const auto add_group = [&](const GpuLinearMatrixGroupQP &group) {
            for (std::size_t i = 0; i < group.data().size(); ++i)
            {
                auto &elements = galois_by_level[std::size_t{group.data()[i].level} + 1];
                for (const int step : group.data()[i].plan.baby_steps)
                    if (step != 0) elements.insert(galois_tool->get_elt_from_step(step));
                for (const int step : group.data()[i].plan.giant_steps)
                    if (step != 0) elements.insert(galois_tool->get_elt_from_step(step));
            }
        };
        add_group(data.coeff_to_slot_matrix_qp);
        add_group(data.slot_to_coeff_matrix_qp);
        // Conjugations: after CtS (real/imag split) and the real projection.
        // (required_dft_key_q_counts returns a sorted set, so the post-CtS
        // level is computed from the stage drops.)
        std::size_t post_cts_q = full_q;
        for (const auto drop : dft_stage_rescale_counts(c2s))
            post_cts_q -= drop;
        galois_by_level[post_cts_q];
        galois_by_level[output_q];
        for (auto &[level, elements] : galois_by_level)
        {
            (void)level;
            elements.insert(galois_tool->get_elt_from_step(0));
        }
        for (const auto &[level, elements] : galois_by_level)
            std::cout << "GPU_BOOT_LEVEL_P galois_level q=" << level
                      << " elements=" << elements.size() << '\n';
        install_level_special_modulus(context, key_generator, cuda_device_id, options,
            galois_by_level, data.eval_mod.required_relin_q_counts, *rotations, *relin,
            result.level_parameters);
    }
    result.profile_id = "poseidon-resnet20-manual-q" + std::to_string(full_q) + "-scale40";
    result.cuda_device_id = cuda_device_id;
    result.input_level_min = result.input_level_max = 5;
    result.input_scale_log2 = 40;
    result.output_level = static_cast<int>(output_q - 1);
    result.output_scale_log2 = 40;
    result.bootstrap_data = std::move(data);
    result.relin_keys = std::move(relin);
    result.galois_keys = std::move(rotations);
    return result;
}

std::size_t GpuBootstrapData::allocated_bytes() const
{
    std::size_t bytes = 0;
    const auto add = [&](std::size_t value) {
        if (value > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("native Boot persistent bytes overflow");
        bytes += value;
    };
    const auto plain = [&](const GpuPlaintextData &value) {
        for (const auto &field : value.fields_) add(field.buffer.allocated_bytes());
    };
    const auto plains = [&](const auto &values) { for (const auto &value : values) plain(value); };
    const auto classic = [&](const GpuLinearMatrixGroup &group) {
        for (const auto &matrix : group.data())
            for (const auto &entry : matrix.plain_vec) plain(entry.second);
    };
    const auto plan_tables = [&](const GpuDoubleHoistMatrixPlan &plan) {
        add(plan.diagonal_q_ptrs.allocated_bytes()); add(plan.diagonal_p_ptrs.allocated_bytes());
        add(plan.diagonal_periods.allocated_bytes()); add(plan.term_baby_indices.allocated_bytes());
        add(plan.group_term_offsets_device.allocated_bytes());
    };
    const auto qp = [&](const GpuLinearMatrixGroupQP &group) {
        for (const auto &matrix : group.data()) {
            for (const auto &entry : matrix.plain_vec_qp) plain(entry.second);
            for (const auto &entry : matrix.compressed_plain_vec_qp)
                add(entry.second.residues.allocated_bytes());
            plan_tables(matrix.plan);
        }
    };
    const auto group_views = [&](const std::vector<std::vector<GpuMatrixPlainQP>> &layers) {
        for (const auto &layer : layers)
            for (const auto &view : layer) plan_tables(view.plan);
    };
    plain(post_raise_plaintext); plain(minus_i_plaintext); plain(plus_i_plaintext);
    plain(eval_mod_input_offset_plaintext); plains(eval_mod_polynomial_plaintexts);
    plains(double_angle_plaintexts); plain(eval_mod.input_offset_plaintext);
    plains(eval_mod.chebyshev_one_plaintexts); plains(eval_mod.polynomial_coefficients);
    plains(eval_mod.double_angle_constants);
    for (const auto &term : eval_mod.polynomial_terms) plain(term.coefficient_plaintext);
    for (const auto &block : eval_mod.polynomial_blocks)
        for (const auto &term : block.terms) plain(term.coefficient_plaintext);
    for (const auto &step : eval_mod.basis_steps) {
        plain(step.operand_alignment_plaintext); plain(step.correction_alignment_plaintext);
        plain(step.correction_plaintext);
    }
    for (const auto &step : eval_mod.polynomial_combine_steps) {
        plain(step.product_scale_plaintext); plain(step.remainder_scale_plaintext);
    }
    classic(coeff_to_slot_matrix); classic(slot_to_coeff_matrix);
    qp(coeff_to_slot_matrix_qp); qp(slot_to_coeff_matrix_qp);
    group_views(coeff_to_slot_groups_qp); group_views(slot_to_coeff_groups_qp);
    return bytes;
}

} // namespace gpu
} // namespace poseidon
