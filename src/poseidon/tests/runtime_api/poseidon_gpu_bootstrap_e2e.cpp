#include "poseidon/ckks_encoder.h"
#include "poseidon/advance/homomorphic_mod.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_bootstrap_profile.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/boot_recipe.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "runtime/runtime.hpp"
#include "manual_bootstrap_reference.h"
#include "nlohmann/json.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using poseidon::runtime_api::PoseidonGpuApi;
using poseidon::runtime_api::PoseidonGpuValue;

constexpr int kSkip = 77;
constexpr int kCudaDevice = 0;
constexpr int kLogScale = 40;
constexpr std::uint32_t kQ0Level = 1;
constexpr double kTolerance = 1.0e-3;
// The ResNet20 profiles below were validated at this bound, before scale45.
constexpr double kLegacyTolerance = 3.0e-3;
const std::string kContextId = "poseidon-runtime-native-bootstrap";
const std::string kOperatorSpecSha =
    "sha256:2222222222222222222222222222222222222222222222222222222222222222";
const std::string kPlanSha =
    "sha256:3333333333333333333333333333333333333333333333333333333333333333";

std::size_t env_size_or(const char *name, std::size_t fallback)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
    {
        return fallback;
    }
    return static_cast<std::size_t>(std::stoull(raw));
}

int log2_degree(std::size_t degree)
{
    if (degree < 2 || (degree & (degree - 1)) != 0)
    {
        throw std::invalid_argument("bootstrap degree must be a power of two");
    }
    int result = 0;
    while (degree > 1)
    {
        degree >>= 1;
        ++result;
    }
    return result;
}

// The ResNet20 profiles (degree-58 da3 full cosine, scale52, real) predate the
// validated scale45 contract and keep the mixed Q chain they were measured with.
bool legacy_profile()
{
    return std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_MODEL_PROFILE") != nullptr ||
           std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_FULL_COSINE") != nullptr ||
           std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_REAL_PROFILE") != nullptr;
}

std::vector<std::uint32_t> mixed_bootstrap_q_chain()
{
    return {
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        28, 28, 31, 31, 32, 32, 30, 31, 32, 31, 32,
        32, 31, 31, 31, 32, 32, 31, 32, 32, 32, 30};
}

poseidon::PoseidonContext make_context(std::size_t degree)
{
    const int log_n = log2_degree(degree);
    const bool model_profile = std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_MODEL_PROFILE") != nullptr;
    const bool real_profile = std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_REAL_PROFILE") != nullptr;
    poseidon::ParametersLiteral parameters(
        CKKS,
        log_n,
        log_n - 1,
        model_profile ? (real_profile ? 51 : 52) : kLogScale,
        /*hamming_weight=*/model_profile || std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_MANUAL_Q50") ? 192 : 0,
        kQ0Level,
        poseidon::Modulus(0),
        {},
        {},
        poseidon::sec_level_type::none);
    if (const char *fixture = std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_MANUAL_Q50")) {
        std::ifstream input(fixture);
        const auto json = nlohmann::json::parse(input);
        if (degree != json.at("poly_degree").get<std::size_t>())
            throw std::runtime_error("manual Q50 fixture degree mismatch");
        std::vector<poseidon::Modulus> q, p;
        for (const auto &value : json.at("rns_moduli")) q.emplace_back(value.get<std::uint64_t>());
        for (const auto &value : json.at("p_moduli")) p.emplace_back(value.get<std::uint64_t>());
        parameters.set_modulus(q, p);
    }
    else if (legacy_profile())
    {
        auto q = mixed_bootstrap_q_chain();
        if (model_profile) q.insert(q.begin(), 16, 32);
        parameters.set_log_modulus(q, std::vector<std::uint32_t>(model_profile ? 25 : 9, 32));
    }
    else
    {
        parameters.set_log_modulus(
            std::vector<std::uint32_t>(50, 30),
            std::vector<std::uint32_t>(8, 30));
    }
    return poseidon::PoseidonContext(parameters);
}

fhegpu::Place host_place()
{
    return {fhegpu::PlaceKind::Host, 0, 0};
}

fhegpu::Place device_place()
{
    return {fhegpu::PlaceKind::Device, 0, 0};
}

fhegpu::LoadedOperatorSpec make_operator_spec(
    const poseidon::PoseidonContext &context,
    fhegpu::BootProfile boot_profile)
{
    fhegpu::OperatorSpec spec;
    spec.id = "poseidon-gpu-native-bootstrap-e2e";
    spec.version = 1;
    spec.status = "test";
    spec.target_id = "poseidon-ckks-gpu";
    spec.rescale_mode = fhegpu::RescaleMode::Lazy;
    spec.context_id = kContextId;

    const auto parameters = context.parameters_literal();
    spec.poly_degree = parameters->degree();
    for (const auto &modulus : parameters->q())
    {
        spec.rns_moduli_log2.push_back(modulus.bit_count());
        spec.rns_moduli.push_back(modulus.value());
    }
    spec.max_modulus_log2 = *std::max_element(
        spec.rns_moduli_log2.begin(), spec.rns_moduli_log2.end());
    spec.default_scale_log2 = static_cast<int>(parameters->log_scale());
    spec.level_lower_bound = 0;
    spec.level_upper_bound = static_cast<int>(parameters->q().size() - 1);

    fhegpu::OperatorSupport boot;
    boot.supported = true;
    spec.operators.emplace(fhegpu::ComputeKind::Boot, std::move(boot));
    fhegpu::OperatorSupport rescale;
    rescale.supported = true;
    rescale.max_levels_per_op = 4;
    spec.operators.emplace(fhegpu::ComputeKind::Rescale, std::move(rescale));
    spec.boot_profiles = {std::move(boot_profile)};
    return {std::move(spec), kOperatorSpecSha};
}

fhegpu::TargetConfig make_target(const fhegpu::LoadedOperatorSpec &loaded_spec)
{
    fhegpu::TargetConfig target;
    target.target_id = loaded_spec.spec.target_id;
    target.world_size = 1;
    target.device_counts = {1};
    target.capability_version = 1;
    target.operator_spec = {
        loaded_spec.spec.id,
        loaded_spec.spec.version,
        loaded_spec.source_sha256};
    return target;
}

fhegpu::CommAction transfer(
    fhegpu::TransferId id,
    fhegpu::ValueId input,
    fhegpu::ValueId output,
    const fhegpu::Place &source,
    const fhegpu::Place &destination)
{
    fhegpu::CommAction action;
    action.id = id;
    action.kind = fhegpu::CommKind::Transfer;
    action.hint = fhegpu::CommHint::PointToPoint;
    action.inputs = {input};
    action.outputs = {output};
    action.sources = {source};
    action.destinations = {destination};
    action.output_types = {fhegpu::ValueKind::Ciphertext};
    return action;
}

fhegpu::LoadedRuntimePlan make_plan(
    const fhegpu::LoadedOperatorSpec &loaded_spec,
    const fhegpu::BootProfile &profile)
{
    const auto host = host_place();
    const auto device = device_place();
    const int input_level = profile.input_scale_log2
        ? profile.input_level_max : loaded_spec.spec.level_upper_bound;

    fhegpu::RuntimePlan plan;
    plan.plan_id = 1;
    plan.target = make_target(loaded_spec);
    plan.values = {
        {0, fhegpu::ValueKind::Ciphertext, host, kContextId,
         input_level, kLogScale, true, 2},
        {1, fhegpu::ValueKind::Ciphertext, device, kContextId,
         input_level, kLogScale, true, 2},
        {2, fhegpu::ValueKind::Ciphertext, device, kContextId,
         profile.output_level, profile.output_scale_log2, true,
         profile.output_components},
        {3, fhegpu::ValueKind::Ciphertext, host, kContextId,
         profile.output_level, profile.output_scale_log2, true,
         profile.output_components},
    };
    plan.external_inputs = {0};
    plan.initialization = {
        {0, transfer(0, 0, 1, host, device)},
    };

    fhegpu::ComputeOp boot;
    boot.kind = fhegpu::ComputeKind::Boot;
    boot.inputs = {1};
    boot.output = 2;
    boot.place = device;
    boot.attrs = fhegpu::BootAttrs{
        profile.output_level,
        profile.output_scale_log2,
        profile.output_components,
        profile.profile_id,
        fhegpu::BootImplementation::Native};
    plan.execution = {{1, std::move(boot)}};
    plan.finalization = {
        {2, transfer(1, 2, 3, device, host)},
    };
    plan.final_outputs = {3};
    return {std::move(plan), kPlanSha};
}

/*
 * A Boot cut into the steps of `granularity` (stage, group or op), driven by
 * the profile's own recipe in the order expand-ckks-bootstrap emits them
 * (compiler/lib/Dialect/CKKS/Transforms/ExpandBootstrap.cpp), each step being
 * what PoseidonGpuApi runs for a RuntimePlan V6 boot_step.
 */
poseidon::gpu::GpuCiphertextData run_boot_steps(
    const poseidon::gpu::GpuEvaluator &evaluator,
    const poseidon::gpu::GpuBootstrapProfile &profile,
    const nlohmann::json &recipe,
    const std::string &granularity,
    const poseidon::gpu::GpuCiphertextData &input,
    poseidon::gpu::GpuBootstrapWorkspace &workspace)
{
    using poseidon::gpu::GpuCiphertextData;
    using poseidon::runtime_api::boot_plaintext;
    const auto &data = profile.bootstrap_data;
    const auto &keys = *profile.galois_keys;
    const auto add = [&](const GpuCiphertextData &a, const GpuCiphertextData &b) {
        GpuCiphertextData out; evaluator.add(a, b, out); return out;
    };
    const auto times = [&](const GpuCiphertextData &a, const std::string &key) {
        GpuCiphertextData out; evaluator.multiply_plain_q_view(a, boot_plaintext(data, key), out); return out;
    };
    const auto relabel = [&](const GpuCiphertextData &a, double scale_log2) {
        GpuCiphertextData out; evaluator.bootstrap_set_scale(a, std::exp2(scale_log2), out); return out;
    };
    const auto transform = [&](bool c2s, GpuCiphertextData x) {
        const auto &layers = recipe.at(c2s ? "coeff_to_slot" : "slot_to_coeff").at("layers");
        for (std::size_t index = 0; index < layers.size(); ++index)
        {
            const auto &layer = layers[index];
            GpuCiphertextData y;
            if (granularity == "stage")
            {
                evaluator.bootstrap_linear_layer(x, data, c2s, index, keys, workspace, y);
                x = std::move(y);
                continue;
            }
            std::map<int, GpuCiphertextData> rotated;
            if (granularity == "op")
                for (const auto &step : layer.at("baby_steps"))
                    if (const int baby = step.get<int>(); baby != 0)
                        evaluator.rotate(x, baby, keys, rotated[baby]);
            std::vector<GpuCiphertextData> partials;
            const auto &groups = layer.at("groups");
            for (std::size_t group = 0; group < groups.size(); ++group)
            {
                GpuCiphertextData partial;
                if (granularity == "group")
                {
                    evaluator.bootstrap_bsgs_group(x, data, c2s, index, group, keys, workspace, partial);
                    partials.push_back(std::move(partial));
                    continue;
                }
                bool first = true;
                for (const auto &term : groups[group].at("terms"))
                {
                    const int baby = term.at(0).get<int>();
                    auto product = times(baby == 0 ? x : rotated.at(baby),
                        std::string(c2s ? "c2s/" : "s2c/") + std::to_string(index) + "/" +
                        std::to_string(term.at(1).get<int>()));
                    partial = first ? std::move(product) : add(partial, product);
                    first = false;
                }
                if (const int giant = groups[group].at("giant").get<int>(); giant != 0)
                {
                    GpuCiphertextData turned;
                    evaluator.rotate(partial, giant, keys, turned);
                    partial = std::move(turned);
                }
                partials.push_back(std::move(partial));
            }
            GpuCiphertextData total = std::move(partials.front());
            for (std::size_t i = 1; i < partials.size(); ++i) total = add(total, partials[i]);
            evaluator.rescale_many(total, y, layer.at("rescale_count").get<std::uint32_t>());
            y.meta.scale = std::exp2(layer.at("output_scale_log2").get<double>());
            x = std::move(y);
        }
        return x;
    };

    const bool s2c_first = recipe.value("order", "") == "slot_to_coeff_first";
    GpuCiphertextData x;
    if (s2c_first)
    {
        // SlotToCoeff on the input, the rounded product to ModRaise's scale.
        GpuCiphertextData copy;
        evaluator.bootstrap_set_scale(input, input.meta.scale, copy);
        x = transform(false, std::move(copy));
        const auto &prepare = recipe.at("prepare");
        GpuCiphertextData y;
        evaluator.multiply_scalar(x, prepare.at("multiplier").get<std::uint64_t>(), y);
        y.meta.scale = std::exp2(prepare.at("scale_log2").get<double>());
        evaluator.bootstrap_raise(y, data, workspace, x);
    }
    else
        evaluator.bootstrap_raise(input, data, workspace, x);
    if (const auto &post = recipe.at("post_raise"); !post.is_null())
    {
        GpuCiphertextData y;
        if (post.at("kind") == "scalar")
        {
            evaluator.multiply_scalar(x, post.at("multiplier").get<std::uint64_t>(), y);
            y.meta.scale = std::exp2(post.at("scale_log2").get<double>());
        }
        else
            y = times(x, "post_raise");
        x = std::move(y);
    }
    x = transform(true, std::move(x));
    GpuCiphertextData conjugated;
    evaluator.bootstrap_conjugate(x, keys, workspace, conjugated);
    const auto real = add(x, conjugated);
    GpuCiphertextData difference;
    evaluator.sub(x, conjugated, difference);
    const auto imaginary = times(difference, "minus_i");
    GpuCiphertextData eval_real, eval_imaginary;
    evaluator.eval_mod_high_precision(real, data, *profile.relin_keys, workspace, eval_real);
    evaluator.eval_mod_high_precision(imaginary, data, *profile.relin_keys, workspace, eval_imaginary);
    if (s2c_first)
    {
        // real + i * imag, the fold, plus its conjugate, then the output label.
        GpuCiphertextData folded, conjugate;
        evaluator.multiply_scalar(add(times(eval_imaginary, "plus_i"), eval_real),
            recipe.at("merge").at("fold_multiplier").get<std::uint64_t>(), folded);
        evaluator.bootstrap_conjugate(folded, keys, workspace, conjugate);
        return relabel(add(folded, conjugate), recipe.at("output_scale_log2").get<double>());
    }
    if (const auto &scale = recipe.at("slot_to_coeff_input_scale_log2"); !scale.is_null())
    {
        eval_real = relabel(eval_real, scale.get<double>());
        eval_imaginary = relabel(eval_imaginary, scale.get<double>());
    }
    x = transform(false, add(times(eval_imaginary, "plus_i"), eval_real));
    if (const auto &scale = recipe.at("slot_to_coeff_output_scale_log2"); !scale.is_null())
        x = relabel(x, scale.get<double>());
    if (recipe.at("project_real").get<bool>())
    {
        GpuCiphertextData conjugate;
        evaluator.bootstrap_conjugate(x, keys, workspace, conjugate);
        x = add(x, conjugate);
    }
    if (const auto multiplier = recipe.at("output_multiplier").get<std::uint64_t>(); multiplier > 1)
    {
        GpuCiphertextData y;
        evaluator.multiply_scalar(x, multiplier, y);
        x = std::move(y);
    }
    return x;
}

std::vector<std::complex<double>> make_message(std::size_t slot_count)
{
    std::vector<std::complex<double>> result(slot_count);
    for (std::size_t index = 0; index < slot_count; ++index)
    {
        result[index] = {
            static_cast<double>((index % 17) + 1) / 32.0,
            std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_REAL_PROFILE") != nullptr ||
                    std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_MANUAL_Q50") != nullptr
                ? 0.0 : static_cast<double>((index % 11) + 1) / 64.0};
    }
    return result;
}

} // namespace

int main()
{
    int device_count = 0;
    const cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count <= kCudaDevice)
    {
        std::cerr << "[SKIP] Runtime native bootstrap requires CUDA device 0: "
                  << cudaGetErrorString(cuda_status) << '\n';
        return kSkip;
    }

    try
    {
        const std::size_t degree = env_size_or(
            "POSEIDON_RUNTIME_BOOTSTRAP_DEGREE",
            std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_MANUAL_Q50") ? 65536 : 16384);
        auto context = make_context(degree);
        poseidon::KeyGenerator key_generator(context);
        poseidon::PublicKey public_key;
        key_generator.create_public_key(public_key);
        poseidon::Encryptor encryptor(context, public_key);
        poseidon::Decryptor decryptor(context, key_generator.secret_key());
        poseidon::CKKSEncoder encoder(context);

        PoseidonGpuApi api(kContextId, context, kCudaDevice);
        poseidon::gpu::GpuBootstrapProfileConfig profile_config;
        if (!legacy_profile())
        {
            profile_config.profile_id = "poseidon-gpu-degree58-da2-scale45";
            profile_config.eval_mod_log_scale = 45;
            profile_config.eval_mod_sine_degree = 58;
            profile_config.eval_mod_generation_degree = 59;
            profile_config.eval_mod_truncate_degree.reset();
            profile_config.eval_mod_double_angle = 2;
        }
        if (std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_MODEL_PROFILE") != nullptr)
        {
            profile_config.eval_mod_log_scale = 52;
        }
        if (std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_FULL_COSINE") != nullptr)
        {
            // The generated degree-59 discrete fit has effective even degree 58.
            // Truncating it to degree 22 changes the refresh function itself.
            profile_config.profile_id = "poseidon-gpu-degree58-da3-full-cosine";
            if (profile_config.eval_mod_log_scale == 52)
                profile_config.profile_id += "-scale52";
            profile_config.eval_mod_sine_degree = 58;
            profile_config.eval_mod_truncate_degree.reset();
        }
        if (std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_REAL_PROFILE") != nullptr)
        {
            profile_config.project_real = true;
            profile_config.profile_id += "-real";
        }
        const char *linear_mode =
            std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_LINEAR_MODE");
        if (linear_mode != nullptr && std::string(linear_mode) == "classic")
        {
            profile_config.linear_transform_mode =
                poseidon::gpu::GpuLinearTransformMode::ClassicBsgs;
        }
        else if (linear_mode != nullptr && std::string(linear_mode) == "no_hoist")
        {
            profile_config.linear_transform_mode =
                poseidon::gpu::GpuLinearTransformMode::NoHoistBsgs;
        }
        // Cerium-style Boot: degree-22 full-coefficient fit at K=12 with
        // three double angles, ModRaise under a sparse ephemeral secret.
        poseidon::gpu::GpuResNet20ManualOptions manual_options;
        if (std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_D22K12") != nullptr)
        {
            profile_config.profile_id += "-d22k12";
            profile_config.eval_mod_sine_degree = 22;
            profile_config.eval_mod_generation_degree = 22;
            profile_config.eval_mod_truncate_degree.reset();
            profile_config.eval_mod_k = 12;
            profile_config.eval_mod_double_angle = 3;
            profile_config.eval_mod_log_split = 2;
            profile_config.eval_mod_full_coefficients = true;
            manual_options.sine_degree = 22;
            manual_options.double_angle = 3;
            manual_options.k = 12;
            manual_options.log_split = 2;
            manual_options.full_coefficients = true;
        }
        if (const char *raw = std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_SPARSE_HW"))
        {
            profile_config.sparse_secret_hamming_weight =
                static_cast<std::uint32_t>(std::stoul(raw));
            manual_options.sparse_secret_hamming_weight =
                profile_config.sparse_secret_hamming_weight;
        }
        if (std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_COMPRESS") != nullptr)
        {
            profile_config.compress_linear_plaintexts = true;
        }
        if (const char *raw = std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_DECLARED_OUTPUT_LEVEL"))
        {
            profile_config.output_level = std::stoi(raw);
        }
        const bool manual = std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_MANUAL_Q50");
        auto profile = manual
            ? poseidon::gpu::GpuBootstrapProfileBuilder::build_resnet20_manual(
                  context, key_generator, kCudaDevice, manual_options)
            : poseidon::gpu::GpuBootstrapProfileBuilder::build(context, key_generator, kCudaDevice, profile_config);
        if (!manual && !legacy_profile() &&
            (profile.input_level_min != static_cast<int>(kQ0Level + 1) ||
             profile.output_level != 22 || profile.output_scale_log2 != 45))
            throw std::runtime_error("validated scale45 Boot contract changed");
        if (manual && (profile.input_level_min != 5 || profile.input_level_max != 5 ||
            profile.input_scale_log2 != 40 || profile.output_level != 30 || profile.output_scale_log2 != 40))
            throw std::runtime_error("manual Q6/scale40 -> Q31/scale40 contract changed");
        if (const char *raw =
                std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_OUTPUT_RATIO"))
        {
            profile.bootstrap_data.output_ratio =
                static_cast<std::uint32_t>(std::stoul(raw));
        }
        if (std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_SKIP_OUTPUT_SCALE") !=
            nullptr)
        {
            profile.bootstrap_data.slot_to_coeff_output_scale = 0.0;
        }
        if (const char *raw =
                std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_OUTPUT_SCALE"))
        {
            profile.bootstrap_data.slot_to_coeff_output_scale =
                std::stod(raw);
        }
        if (const char *raw =
                std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_OUTPUT_SCALE_LOG2"))
        {
            profile.output_scale_log2 = std::stoi(raw);
        }
        const auto boot_profile =
            poseidon::runtime_api::make_native_boot_profile(profile);
        std::cout << "NATIVE_BOOT_PROFILE id=" << profile.profile_id
                  << " linear_mode=" << static_cast<int>(profile_config.linear_transform_mode)
                  << " evalmod_degree=" << profile_config.eval_mod_sine_degree
                  << " evalmod_log_scale=" << profile_config.eval_mod_log_scale
                  << " project_real=" << profile_config.project_real
                  << " input_level_min=" << profile.input_level_min
                  << " input_level_max=" << profile.input_level_max
                  << " output_level=" << profile.output_level
                  << " output_scale_log2=" << profile.output_scale_log2 << '\n';
        std::cout << "NATIVE_BOOT_CONTEXT degree=" << context.parameters_literal()->degree()
                  << " hamming_weight=" << context.parameters_literal()->hamming_weight()
                  << " default_scale_log2=" << context.parameters_literal()->log_scale()
                  << " q_bits=";
        for (const auto &q : context.parameters_literal()->q())
            std::cout << q.bit_count() << ',';
        std::cout << " p_bits=";
        for (const auto &p : context.parameters_literal()->p())
            std::cout << p.bit_count() << ',';
        std::cout << '\n';
        const auto loaded_spec = make_operator_spec(context, boot_profile);
        const auto loaded_plan = make_plan(loaded_spec, boot_profile);

        const auto message = make_message(
            std::size_t{1} << context.parameters_literal()->log_slots());
        poseidon::Plaintext plaintext;
        encoder.encode(message, context.crt_context()->parms_id_map().at(
            loaded_plan.plan.values.front().level), std::ldexp(1.0, kLogScale), plaintext);
        poseidon::Ciphertext ciphertext;
        encryptor.encrypt(plaintext, ciphertext);

        if (std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_DIAGNOSTICS") != nullptr)
        {
            // Offline diagnostics only: decrypt intermediates to separate
            // approximation error from GPU polynomial execution error.
            poseidon::gpu::GpuParameterData parameters(context, 0);
            poseidon::gpu::GpuEvaluator evaluator(parameters);
            auto input = poseidon::gpu::GpuUploader::upload_ciphertext(ciphertext, 0);
            poseidon::gpu::GpuBootstrapWorkspace workspace;
            poseidon::gpu::GpuCiphertextData result;
            evaluator.bootstrap(input, profile.bootstrap_data, *profile.relin_keys,
                                *profile.galois_keys, workspace, result);
            poseidon::EvalModPoly polynomial(context, poseidon::CosDiscrete,
                std::exp2(profile_config.eval_mod_log_scale), 0, 5, 3, 25, 0, 59);
            if (profile_config.eval_mod_truncate_degree)
                polynomial.truncate_sine_polynomial(*profile_config.eval_mod_truncate_degree);
            const auto decode = [&](const auto &value) {
                poseidon::Ciphertext host;
                poseidon::gpu::GpuUploader::download_ciphertext(value, host, context);
                poseidon::Plaintext plain;
                decryptor.decrypt(host, plain);
                std::vector<std::complex<double>> slots;
                encoder.decode(plain, slots);
                return slots;
            };
            const auto check = [&](const char *name, const auto &source, const auto &output) {
                const auto x = decode(source), y = decode(output);
                double arithmetic = 0, approximation = 0, max_input = 0;
                for (std::size_t slot = 0; slot < x.size(); ++slot)
                {
                    const auto z = x[slot]-0.5/(polynomial.sc_fac()*
                        (polynomial.sine_poly_b()-polynomial.sine_poly_a()));
                    std::complex<double> previous=1, current=z;
                    auto expected=polynomial.sine_poly().data()[0];
                    for (std::size_t i=1; i<polynomial.sine_poly().data().size(); ++i)
                    {
                        expected += polynomial.sine_poly().data()[i]*current;
                        auto next=2.0*z*current-previous;
                        previous=current; current=next;
                    }
                    double constant=polynomial.sqrt_2pi();
                    for (unsigned i=0; i<polynomial.double_angle(); ++i)
                    { constant*=constant; expected=2.0*expected*expected-constant; }
                    const double pi=std::acos(-1.0);
                    const auto ideal=polynomial.q_diff()/(2*pi)*
                        std::sin(2*pi*polynomial.k()*polynomial.sc_fac()*x[slot]);
                    arithmetic=std::max(arithmetic,std::abs(y[slot]-expected));
                    approximation=std::max(approximation,std::abs(expected-ideal));
                    max_input=std::max(max_input,std::abs(x[slot]));
                    if (slot<4) std::cout << "NATIVE_STAGE_SAMPLE branch=" << name
                        << " slot=" << slot << " input=" << x[slot] << " output=" << y[slot]
                        << " polynomial=" << expected << " ideal=" << ideal << std::endl;
                }
                std::cout << "NATIVE_STAGE_DIAGNOSTIC branch=" << name
                    << " max_input=" << max_input << " gpu_vs_plain_polynomial=" << arithmetic
                    << " polynomial_vs_ideal_sine=" << approximation << std::endl;
            };
            check("real",workspace.coeff_to_slot_real,workspace.eval_mod_real);
            check("imag",workspace.coeff_to_slot_imag,workspace.eval_mod_imag);
        }
        if (std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_STEPS") != nullptr)
        {
            // An expanded Boot (RuntimePlan V6) at every granularity must
            // refresh as well as the fused bootstrap() does.
            poseidon::gpu::GpuParameterData parameters(context, kCudaDevice);
            poseidon::gpu::GpuEvaluator evaluator(parameters);
            const auto input = poseidon::gpu::GpuUploader::upload_ciphertext(ciphertext, kCudaDevice);
            poseidon::gpu::GpuBootstrapWorkspace workspace;
            poseidon::gpu::GpuCiphertextData fused;
            evaluator.bootstrap(input, profile.bootstrap_data, *profile.relin_keys,
                                *profile.galois_keys, workspace, fused);
            const auto recipe = poseidon::runtime_api::make_boot_recipe(profile, context);
            const auto decode = [&](const poseidon::gpu::GpuCiphertextData &value) {
                poseidon::Ciphertext host;
                poseidon::gpu::GpuUploader::download_ciphertext(value, host, context);
                poseidon::Plaintext plain;
                decryptor.decrypt(host, plain);
                std::vector<std::complex<double>> slots;
                encoder.decode(plain, slots);
                return slots;
            };
            const auto reference = decode(fused);
            const bool double_hoist = recipe.at("linear_transform_mode") == "double_hoist";
            for (const std::string granularity : {"stage", "group", "op"})
            {
                if (granularity == "group" && !double_hoist) continue;
                if (granularity == "op" && !recipe.value("diagonal_plaintexts", true))
                {
                    std::cout << "BOOT_STEPS granularity=op skipped: the profile's diagonals are compressed\n";
                    continue;
                }
                const auto stepped = run_boot_steps(evaluator, profile, recipe, granularity, input, workspace);
                if (stepped.meta.q_count != fused.meta.q_count ||
                    std::abs(std::log2(stepped.meta.scale) - std::log2(fused.meta.scale)) > 1e-9)
                    throw std::runtime_error(granularity + " steps end at another level or scale than bootstrap()");
                const auto values = decode(stepped);
                double to_fused = 0, to_message = 0;
                for (std::size_t slot = 0; slot < message.size(); ++slot)
                {
                    // The manual Boot has no arcsine stage: it is checked against
                    // its own refresh function, as the fused Boot is below.
                    const auto expected = manual ? std::complex<double>(
                        poseidon::test::manual_bootstrap_reference(context, message[slot].real()), 0)
                        : message[slot];
                    to_fused = std::max(to_fused, std::abs(values[slot] - reference[slot]));
                    to_message = std::max(to_message, std::abs(values[slot] - expected));
                }
                std::cout << "BOOT_STEPS granularity=" << granularity << " max_abs_vs_fused=" << to_fused
                          << " max_abs_vs_message=" << to_message << '\n';
                if (to_message > (manual ? 1e-4 : legacy_profile() ? kLegacyTolerance : kTolerance))
                    throw std::runtime_error(granularity + " Boot steps error exceeds tolerance: " +
                                             std::to_string(to_message));
            }
        }
        api.configure_native_bootstrap(0, std::move(profile));

        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(0, 1, 1, api);
        const fhegpu::RuntimeResources resources{
            loaded_spec, std::nullopt, false};
        std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
        inputs.emplace(
            0, PoseidonGpuValue::from_host_ciphertext(std::move(ciphertext)));
        const auto artifact = runtime.run(loaded_plan, resources, inputs);

        const auto &output = artifact.values.at(3).value.host_ciphertext();
        poseidon::Plaintext output_plaintext;
        decryptor.decrypt(output, output_plaintext);
        std::vector<std::complex<double>> decoded;
        encoder.decode(output_plaintext, decoded);

        double max_abs_error = 0.0;
        double max_identity_error = 0.0;
        for (std::size_t index = 0; index < message.size(); ++index)
        {
            const auto reference = manual ? std::complex<double>(
                poseidon::test::manual_bootstrap_reference(context, message[index].real()), 0) : message[index];
            max_abs_error = std::max(max_abs_error, std::abs(decoded.at(index) - reference));
            max_identity_error = std::max(max_identity_error, std::abs(decoded.at(index) - message[index]));
        }
        if (max_abs_error > (manual ? 1e-4 : legacy_profile() ? kLegacyTolerance : kTolerance))
        {
            std::cerr << "[INFO] output size=" << output.size()
                      << " scale=" << output.scale()
                      << " q_count=" << output.coeff_modulus_size() << '\n';
            const std::size_t sample_count =
                std::min<std::size_t>(4, message.size());
            for (std::size_t index = 0; index < sample_count; ++index)
            {
                std::cerr << "[INFO] sample[" << index << "] expected="
                          << message[index] << " actual=" << decoded[index]
                          << " error="
                          << std::abs(decoded[index] - message[index]) << '\n';
            }
            throw std::runtime_error(
                "Runtime native bootstrap error exceeds tolerance: " +
                std::to_string(max_abs_error));
        }

        std::cout << "[PASS] RuntimePlan native GPU bootstrap"
                  << " degree=" << degree
                  << " output_level=" << boot_profile.output_level
                  << " output_scale_log2=" << boot_profile.output_scale_log2
                  << " max_abs_error=" << max_abs_error << '\n';
        if (manual) std::cout << "[INFO] manual arcsine_degree=0 max_identity_error=" << max_identity_error << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
