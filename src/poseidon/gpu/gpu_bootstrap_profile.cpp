#include "poseidon/gpu/gpu_bootstrap_profile.h"

#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/advance/homomorphic_mod.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"
#include "poseidon/keygenerator.h"
#include "poseidon/poseidon_context.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
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
        parameters->q0_level() >= parameters->q().size())
    {
        throw std::invalid_argument(
            "GPU bootstrap context has an invalid Q/P modulus chain");
    }
    if (config.schedule != GpuBootstrapSchedule::Standard &&
        config.schedule != GpuBootstrapSchedule::StCFirst)
    {
        throw std::invalid_argument("GPU bootstrap schedule is invalid");
    }
    if (config.schedule == GpuBootstrapSchedule::StCFirst &&
        (config.stc_input_q_count <= parameters->q0_level() ||
         config.stc_input_q_count > parameters->q().size() ||
         !(config.stc_scaling > 0.0) || !std::isfinite(config.stc_scaling) ||
         config.project_real))
    {
        throw std::invalid_argument(
            "GPU StC-first bootstrap profile configuration is invalid");
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
    const double q0_over_message_ratio = std::exp2(std::round(std::log2(
        context.crt_context()->q0() /
        static_cast<double>(config.bootstrap_ratio))));

    if (config.schedule == GpuBootstrapSchedule::StCFirst)
    {
        auto slot_to_coeff = make_slot_to_coeff_matrix_group(
            context,
            encoder,
            config.stc_input_q_count - 1,
            config.stc_scaling,
            config,
            parameters->scale(),
            eval_mod_scale);
        const std::size_t stc_consumed = consumed_q_count(slot_to_coeff);
        if (stc_consumed >= config.stc_input_q_count ||
            config.stc_input_q_count - stc_consumed !=
                static_cast<std::size_t>(q0_level) + 1)
        {
            throw std::invalid_argument(
                "GPU StC-first matrix must finish at the ModRaise base level");
        }

        const double ideal_input_factor =
            1.0 /
            (eval_mod_poly.k() * eval_mod_poly.sc_fac() *
             eval_mod_poly.q_diff() *
             static_cast<double>(config.bootstrap_ratio));
        const double raised_c2s_scale =
            q0_over_message_ratio * c2s_scaling / ideal_input_factor;
        if (!(raised_c2s_scale > 0.0) || !std::isfinite(raised_c2s_scale))
        {
            throw std::runtime_error(
                "GPU StC-first raised C2S scale is invalid");
        }

        auto coeff_to_slot = make_coeff_to_slot_matrix_group(
            context,
            encoder,
            c2s_scaling,
            config,
            eval_mod_scale,
            eval_mod_scale);
        const std::size_t c2s_consumed = consumed_q_count(coeff_to_slot);
        if (c2s_consumed >= full_q_count)
        {
            throw std::invalid_argument(
                "GPU StC-first CoeffToSlot consumes the complete modulus chain");
        }
        const std::size_t c2s_output_q_count =
            full_q_count - c2s_consumed;
        const auto c2s_output_parms_id =
            context.crt_context()->parms_id_map().at(
                static_cast<std::uint32_t>(c2s_output_q_count - 1));
        const double c2s_output_scale = planned_dft_output_scale(
            context, raised_c2s_scale, coeff_to_slot);
        eval_mod_poly.set_level_start(
            static_cast<std::uint32_t>(c2s_output_q_count - 1));

        RelinKeys cpu_relin_keys;
        key_generator.create_relin_keys(cpu_relin_keys);
        auto relin_keys = std::make_shared<GpuRelinKeysData>(
            GpuUploader::upload_relin_keys(
                cpu_relin_keys, cuda_device_id));

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
            config.eval_mod_virtual_degree_bound)
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
                    "GPU StC-first degree-22 baby-4 plan invariant failed");
            }
        }
        if (eval_mod.output_q_count <= 1 ||
            eval_mod.output_q_count >= c2s_output_q_count)
        {
            throw std::runtime_error(
                "GPU StC-first EvalMod produced an invalid output level");
        }
        const auto eval_mod_context =
            context.crt_context()->get_context_data(
                eval_mod.output_parms_id);
        if (!eval_mod_context ||
            eval_mod_context->coeff_modulus().size() !=
                eval_mod.output_q_count)
        {
            throw std::runtime_error(
                "GPU StC-first EvalMod output is absent from the context");
        }

        auto cpu_galois_keys = make_galois_keys(
            context, key_generator, coeff_to_slot, slot_to_coeff);
        auto uploaded_galois_keys = config.linear_transform_mode ==
                GpuLinearTransformMode::DoubleHoistBsgs
            ? GpuUploader::upload_double_hoist_galois_keys(
                  cpu_galois_keys, cuda_device_id)
            : GpuUploader::upload_galois_keys(
                  cpu_galois_keys, cuda_device_id);
        const auto stc_key_q_counts = required_dft_key_q_counts(
            config.stc_input_q_count,
            slot_to_coeff,
            /*include_post_dft_conjugation=*/false);
        const auto c2s_key_q_counts = required_dft_key_q_counts(
            full_q_count,
            coeff_to_slot,
            /*include_post_dft_conjugation=*/true);
        GpuUploader::prepare_key_views_for_q_counts(
            uploaded_galois_keys,
            merge_q_counts(stc_key_q_counts, c2s_key_q_counts));
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
            eval_mod.output_parms_id,
            1.0,
            plus_i);

        const double output_normalization_value =
            static_cast<double>(
                eval_mod_context->coeff_modulus().back().value()) /
            eval_mod.output_scale;
        Plaintext output_scale_normalization;
        encoder.encode(
            output_normalization_value,
            eval_mod.output_parms_id,
            eval_mod_scale,
            output_scale_normalization);

        GpuBootstrapData bootstrap_data;
        bootstrap_data.schedule = GpuBootstrapSchedule::StCFirst;
        bootstrap_data.linear_transform_mode = config.linear_transform_mode;
        bootstrap_data.allow_environment_linear_transform_override = false;
        bootstrap_data.q0_parms_id =
            context.crt_context()->parms_id_map().at(q0_level);
        bootstrap_data.q0_over_message_ratio = q0_over_message_ratio;
        bootstrap_data.post_raise_c2s_input_scale = raised_c2s_scale;
        bootstrap_data.project_real = false;
        bootstrap_data.output_ratio = config.bootstrap_ratio;
        bootstrap_data.output_scale_override = eval_mod_scale;
        if (config.linear_transform_mode ==
            GpuLinearTransformMode::DoubleHoistBsgs)
        {
            bootstrap_data.coeff_to_slot_matrix_qp =
                GpuUploader::upload_linear_matrix_group_qp(
                    coeff_to_slot,
                    context,
                    cuda_device_id,
                    std::max(coeff_to_slot.step(), std::uint32_t{1}));
            bootstrap_data.slot_to_coeff_matrix_qp =
                GpuUploader::upload_linear_matrix_group_qp(
                    slot_to_coeff,
                    context,
                    cuda_device_id,
                    std::max(slot_to_coeff.step(), std::uint32_t{1}));
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
        bootstrap_data.output_scale_normalization_plaintext =
            GpuUploader::upload_plaintext(
                output_scale_normalization, cuda_device_id);
        bootstrap_data.eval_mod = std::move(eval_mod);

        GpuBootstrapProfile result;
        result.profile_id = config.profile_id;
        result.cuda_device_id = cuda_device_id;
        result.input_level_min =
            static_cast<int>(config.stc_input_q_count - 1);
        result.input_level_max = result.input_level_min;
        result.output_level = static_cast<int>(
            bootstrap_data.eval_mod.output_q_count - 2);
        result.output_scale_log2 =
            static_cast<int>(config.eval_mod_log_scale);
        result.bootstrap_data = std::move(bootstrap_data);
        result.relin_keys = std::move(relin_keys);
        result.galois_keys = std::move(galois_keys);
        return result;
    }

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
        config.eval_mod_virtual_degree_bound)
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
    auto slot_to_coeff = make_slot_to_coeff_matrix_group(
        context,
        encoder,
        static_cast<std::uint32_t>(eval_mod_context->level()),
        s2c_scaling,
        config,
        s2c_input_scale,
        eval_mod_scale);
    const double raw_s2c_output_scale = planned_dft_output_scale(
        context, s2c_input_scale, slot_to_coeff);
    slot_to_coeff = make_slot_to_coeff_matrix_group(
        context,
        encoder,
        static_cast<std::uint32_t>(eval_mod_context->level()),
        s2c_scaling,
        config,
        s2c_input_scale,
        eval_mod_scale,
        eval_mod_scale / raw_s2c_output_scale);
    const std::size_t s2c_consumed = consumed_q_count(slot_to_coeff);
    if (s2c_consumed >= eval_mod.output_q_count)
    {
        throw std::invalid_argument(
            "GPU bootstrap SlotToCoeff consumes the remaining modulus chain");
    }
    const std::size_t output_q_count =
        eval_mod.output_q_count - s2c_consumed;

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
        eval_mod.output_q_count,
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
        eval_mod.output_parms_id,
        1.0,
        plus_i);

    GpuBootstrapData bootstrap_data;
    bootstrap_data.linear_transform_mode = config.linear_transform_mode;
    bootstrap_data.allow_environment_linear_transform_override = false;
    bootstrap_data.q0_parms_id = context.crt_context()->parms_id_map().at(q0_level);
    bootstrap_data.q0_over_message_ratio = q0_over_message_ratio;
    bootstrap_data.raised_scale_override = raised_scale;
    bootstrap_data.slot_to_coeff_input_scale = s2c_input_scale;
    bootstrap_data.project_real = config.project_real;
    bootstrap_data.output_ratio = config.bootstrap_ratio;
    bootstrap_data.slot_to_coeff_output_scale = eval_mod_scale;
    if (config.linear_transform_mode ==
        GpuLinearTransformMode::DoubleHoistBsgs)
    {
        bootstrap_data.coeff_to_slot_matrix_qp =
            GpuUploader::upload_linear_matrix_group_qp(
                coeff_to_slot,
                context,
                cuda_device_id,
                std::max(coeff_to_slot.step(), std::uint32_t{1}));
        bootstrap_data.slot_to_coeff_matrix_qp =
            GpuUploader::upload_linear_matrix_group_qp(
                slot_to_coeff,
                context,
                cuda_device_id,
                std::max(slot_to_coeff.step(), std::uint32_t{1}));
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
    result.input_level_min = static_cast<int>(q0_level);
    result.input_level_max = static_cast<int>(full_q_count - 1);
    result.output_level = static_cast<int>(output_q_count - 1);
    result.output_scale_log2 = static_cast<int>(config.eval_mod_log_scale);
    result.bootstrap_data = std::move(bootstrap_data);
    result.relin_keys = std::move(relin_keys);
    result.galois_keys = std::move(galois_keys);
    return result;
}

} // namespace gpu
} // namespace poseidon
