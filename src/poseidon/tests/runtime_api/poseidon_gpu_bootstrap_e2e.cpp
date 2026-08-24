#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/evaluator/evaluator_ckks_base.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/gpu/gpu_bootstrap_profile.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "runtime/runtime.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
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
constexpr double kSemanticRegressionLimit = 10.0;
constexpr double kRuntimeDirectTolerance = 1.0e-7;
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

std::vector<std::uint32_t> mixed_bootstrap_q_chain()
{
    return {
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        28, 30, 31, 31, 32, 32, 30, 31, 32, 31, 32,
        32, 31, 31, 31, 32, 22, 31, 32, 32, 32, 30};
}

poseidon::PoseidonContext make_context(std::size_t degree)
{
    const int log_n = log2_degree(degree);
    poseidon::ParametersLiteral parameters(
        CKKS,
        log_n,
        log_n - 1,
        kLogScale,
        /*hamming_weight=*/0,
        kQ0Level,
        poseidon::Modulus(0),
        {},
        {},
        poseidon::sec_level_type::none);
    parameters.set_log_modulus(
        mixed_bootstrap_q_chain(),
        std::vector<std::uint32_t>(9, 32));
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

fhegpu::TargetConfig make_target(
    const fhegpu::LoadedOperatorSpec &loaded_spec,
    int local_device_count)
{
    fhegpu::TargetConfig target;
    target.target_id = loaded_spec.spec.target_id;
    target.world_size = 1;
    target.device_counts = {local_device_count};
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
    const fhegpu::BootProfile &profile,
    int local_device_count,
    bool return_to_host)
{
    const auto host = host_place();
    const auto device = device_place();
    if (profile.input_level_min != profile.input_level_max)
    {
        throw std::invalid_argument(
            "StC-first bootstrap E2E requires one exact input level");
    }
    const int input_level = profile.input_level_min;

    fhegpu::RuntimePlan plan;
    plan.plan_id = return_to_host ? 1 : 2;
    plan.target = make_target(loaded_spec, local_device_count);
    plan.values = {
        {0, fhegpu::ValueKind::Ciphertext, host, kContextId,
         input_level, kLogScale, true, 2},
        {1, fhegpu::ValueKind::Ciphertext, device, kContextId,
         input_level, kLogScale, true, 2},
        {2, fhegpu::ValueKind::Ciphertext, device, kContextId,
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
    if (return_to_host)
    {
        plan.values.push_back(
            {3, fhegpu::ValueKind::Ciphertext, host, kContextId,
             profile.output_level, profile.output_scale_log2, true,
             profile.output_components});
        plan.finalization = {
            {2, transfer(1, 2, 3, device, host)},
        };
        plan.final_outputs = {3};
    }
    else
    {
        plan.final_outputs = {2};
    }
    return {std::move(plan), kPlanSha};
}

double milliseconds(std::uint64_t nanoseconds)
{
    return static_cast<double>(nanoseconds) / 1.0e6;
}

double elapsed_milliseconds(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::string precise_double(double value)
{
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

void accumulate_multi_gpu_timing(
    poseidon::runtime_api::MultiGpuBootstrapTiming &sum,
    const poseidon::runtime_api::MultiGpuBootstrapTiming &sample)
{
    if (sum.gpu_count == 0)
    {
        sum.gpu_count = sample.gpu_count;
        sum.eval_mod_device_count = sample.eval_mod_device_count;
        sum.s2c_layers.resize(sample.s2c_layers.size());
        for (std::size_t layer = 0;
             layer < sample.s2c_layers.size();
             ++layer)
        {
            sum.s2c_layers[layer].active_device_count =
                sample.s2c_layers[layer].active_device_count;
        }
        sum.c2s_layers.resize(sample.c2s_layers.size());
        for (std::size_t layer = 0;
             layer < sample.c2s_layers.size();
             ++layer)
        {
            sum.c2s_layers[layer].active_device_count =
                sample.c2s_layers[layer].active_device_count;
        }
    }
    if (sum.gpu_count != sample.gpu_count ||
        sum.eval_mod_device_count != sample.eval_mod_device_count ||
        sum.s2c_layers.size() != sample.s2c_layers.size() ||
        sum.c2s_layers.size() != sample.c2s_layers.size())
    {
        throw std::runtime_error(
            "multi-GPU bootstrap timing shape changed between iterations");
    }
    sum.prepare_c2s_ms += sample.prepare_c2s_ms;
    sum.modraise_ms += sample.modraise_ms;
    sum.eval_mod_branches_ms += sample.eval_mod_branches_ms;
    sum.imag_result_copy_ms += sample.imag_result_copy_ms;
    sum.finalize_ms += sample.finalize_ms;
    sum.total_ms += sample.total_ms;
    const auto accumulate_layers = [](
        std::vector<poseidon::runtime_api::MultiGpuBootstrapLayerTiming> &destination_layers,
        const std::vector<poseidon::runtime_api::MultiGpuBootstrapLayerTiming> &source_layers) {
        for (std::size_t layer = 0; layer < source_layers.size(); ++layer)
        {
            auto &destination = destination_layers[layer];
            const auto &source = source_layers[layer];
            destination.fanout_and_partial_compute_ms +=
                source.fanout_and_partial_compute_ms;
            destination.qp_reduction_ms += source.qp_reduction_ms;
            destination.shared_moddown_rescale_ms +=
                source.shared_moddown_rescale_ms;
            destination.total_ms += source.total_ms;
        }
    };
    accumulate_layers(sum.s2c_layers, sample.s2c_layers);
    accumulate_layers(sum.c2s_layers, sample.c2s_layers);
}

std::vector<std::complex<double>> make_message(std::size_t slot_count)
{
    std::vector<std::complex<double>> result(slot_count);
    for (std::size_t index = 0; index < slot_count; ++index)
    {
        result[index] = {
            static_cast<double>((index % 17) + 1) / 32.0,
            static_cast<double>((index % 11) + 1) / 64.0};
    }
    return result;
}

} // namespace

int main()
{
    const std::size_t requested_gpu_count = env_size_or(
        "POSEIDON_RUNTIME_BOOTSTRAP_GPU_COUNT", 1);
    const std::size_t c2s_device_limit = env_size_or(
        "POSEIDON_RUNTIME_BOOTSTRAP_C2S_DEVICE_LIMIT",
        requested_gpu_count);
    const std::size_t s2c_device_limit = env_size_or(
        "POSEIDON_RUNTIME_BOOTSTRAP_S2C_DEVICE_LIMIT",
        requested_gpu_count);
    const std::size_t eval_mod_device_limit = env_size_or(
        "POSEIDON_RUNTIME_BOOTSTRAP_EVALMOD_DEVICE_LIMIT",
        requested_gpu_count);
    if (requested_gpu_count != 1 && requested_gpu_count != 2 &&
        requested_gpu_count != 4)
    {
        std::cerr
            << "[FAIL] POSEIDON_RUNTIME_BOOTSTRAP_GPU_COUNT must be 1, 2, or 4\n";
        return 1;
    }
    if (c2s_device_limit == 0 || c2s_device_limit > requested_gpu_count)
    {
        std::cerr
            << "[FAIL] POSEIDON_RUNTIME_BOOTSTRAP_C2S_DEVICE_LIMIT must be in [1, gpu_count]\n";
        return 1;
    }
    if (s2c_device_limit == 0 || s2c_device_limit > requested_gpu_count)
    {
        std::cerr
            << "[FAIL] POSEIDON_RUNTIME_BOOTSTRAP_S2C_DEVICE_LIMIT must be in [1, gpu_count]\n";
        return 1;
    }
    if (requested_gpu_count > 1 &&
        ((eval_mod_device_limit != 2 && eval_mod_device_limit != 4) ||
         eval_mod_device_limit > requested_gpu_count))
    {
        std::cerr
            << "[FAIL] POSEIDON_RUNTIME_BOOTSTRAP_EVALMOD_DEVICE_LIMIT must be 2 or 4 and not exceed gpu_count\n";
        return 1;
    }
    int device_count = 0;
    const cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess ||
        device_count < static_cast<int>(requested_gpu_count))
    {
        std::cerr << "[SKIP] Runtime native bootstrap requires "
                  << requested_gpu_count << " CUDA device(s): "
                  << cudaGetErrorString(cuda_status) << '\n';
        return kSkip;
    }

    try
    {
        const std::size_t degree = env_size_or(
            "POSEIDON_RUNTIME_BOOTSTRAP_DEGREE", 65536);
        const std::size_t warmup = env_size_or(
            "POSEIDON_RUNTIME_BOOTSTRAP_WARMUP", 2);
        const std::size_t iterations = env_size_or(
            "POSEIDON_RUNTIME_BOOTSTRAP_ITERATIONS", 5);
        if (iterations == 0)
        {
            throw std::invalid_argument(
                "POSEIDON_RUNTIME_BOOTSTRAP_ITERATIONS must be nonzero");
        }
        auto context = make_context(degree);
        poseidon::KeyGenerator key_generator(context);
        poseidon::PublicKey public_key;
        key_generator.create_public_key(public_key);
        poseidon::Encryptor encryptor(context, public_key);
        poseidon::Decryptor decryptor(context, key_generator.secret_key());
        poseidon::CKKSEncoder encoder(context);
        auto cpu_evaluator = poseidon::PoseidonFactory::get_instance()
            ->create_ckks_evaluator(context);

        std::vector<int> cuda_device_ids;
        cuda_device_ids.reserve(requested_gpu_count);
        for (std::size_t index = 0; index < requested_gpu_count; ++index)
        {
            cuda_device_ids.push_back(static_cast<int>(index));
        }
        PoseidonGpuApi api(kContextId, context, cuda_device_ids);
        poseidon::gpu::GpuBootstrapProfileConfig profile_config;
        if (requested_gpu_count > 1)
        {
            /*
             * Hydra Table V reduces BS as compute nodes grow.  Eight/four
             * giant groups keep each two/four-GPU worker at no more than four
             * local groups, preserving the fused QP path while exposing all
             * DFT layers to the scheduler.
             */
            if (requested_gpu_count == 2)
            {
                profile_config.c2s_bsgs_n1_overrides = {4096, 256, 16, 2};
                profile_config.s2c_bsgs_n1_overrides = {8, 256, 4096};
            }
            else
            {
                profile_config.c2s_bsgs_n1_overrides = {2048, 128, 8, 1};
                profile_config.s2c_bsgs_n1_overrides = {4, 128, 2048};
            }
        }
        poseidon::gpu::GpuBootstrapProfileCpuKeys shared_cpu_keys;
        auto profile = poseidon::gpu::GpuBootstrapProfileBuilder::build(
            context,
            key_generator,
            kCudaDevice,
            profile_config,
            requested_gpu_count > 1 ? &shared_cpu_keys : nullptr);
        std::vector<poseidon::gpu::GpuBootstrapProfile> secondary_profiles;
        secondary_profiles.reserve(
            requested_gpu_count > 0 ? requested_gpu_count - 1 : 0);
        for (std::size_t device_index = 1;
             device_index < requested_gpu_count;
             ++device_index)
        {
            secondary_profiles.push_back(
                poseidon::gpu::GpuBootstrapProfileBuilder::build(
                    context,
                    key_generator,
                    static_cast<int>(device_index),
                    profile_config,
                    nullptr,
                    &shared_cpu_keys));
        }
        std::vector<std::size_t> c2s_giant_groups;
        std::vector<std::size_t> s2c_giant_groups;
        for (const auto &matrix :
             profile.bootstrap_data.slot_to_coeff_matrix_qp.data())
        {
            s2c_giant_groups.push_back(matrix.plan.giant_steps.size());
        }
        for (const auto &matrix :
             profile.bootstrap_data.coeff_to_slot_matrix_qp.data())
        {
            c2s_giant_groups.push_back(matrix.plan.giant_steps.size());
        }
        const auto boot_profile =
            poseidon::runtime_api::make_native_boot_profile(profile);
        const auto loaded_spec = make_operator_spec(context, boot_profile);
        const auto device_plan = make_plan(
            loaded_spec,
            boot_profile,
            static_cast<int>(requested_gpu_count),
            /*return_to_host=*/false);
        const auto host_plan = make_plan(
            loaded_spec,
            boot_profile,
            static_cast<int>(requested_gpu_count),
            /*return_to_host=*/true);

        const auto message = make_message(
            std::size_t{1} << context.parameters_literal()->log_slots());
        poseidon::Plaintext plaintext;
        encoder.encode(message, std::ldexp(1.0, kLogScale), plaintext);
        poseidon::Ciphertext ciphertext;
        encryptor.encrypt(plaintext, ciphertext);
        cpu_evaluator->drop_modulus(
            ciphertext,
            ciphertext,
            context.crt_context()->parms_id_map().at(
                static_cast<std::uint32_t>(boot_profile.input_level_min)));
        ciphertext.scale() = std::ldexp(1.0, kLogScale);

        std::vector<std::complex<double>> direct_decoded;
        std::vector<std::complex<double>> staged_decoded;
        std::vector<std::complex<double>> partitioned_decoded;
        int staged_raw_level = 0;
        double staged_raw_scale_log2 = 0.0;
        int staged_branch_level = 0;
        double staged_branch_scale_log2 = 0.0;
        int staged_eval_mod_level = 0;
        double staged_eval_mod_scale_log2 = 0.0;
        double direct_gpu_boot_ms = 0.0;
        double direct_prepare_c2s_ms = 0.0;
        std::vector<double> direct_c2s_layer_ms;
        poseidon::gpu::GpuBootstrapWorkspace::EvalModStageTiming
            direct_eval_mod_timing;
        std::vector<poseidon::gpu::GpuBootstrapWorkspace::EvalModMultiplyTiming>
            direct_eval_mod_multiply_timings;
        {
            poseidon::gpu::GpuParameterData gpu_parameters(
                context, kCudaDevice);
            poseidon::gpu::GpuEvaluator gpu_evaluator(gpu_parameters);
            auto gpu_input = poseidon::gpu::GpuUploader::upload_ciphertext(
                ciphertext, kCudaDevice);
            poseidon::gpu::GpuBootstrapWorkspace workspace;
            poseidon::gpu::GpuCiphertextData gpu_output;
            auto run_direct_boot = [&]() {
                gpu_evaluator.bootstrap(
                    gpu_input,
                    profile.bootstrap_data,
                    *profile.relin_keys,
                    *profile.galois_keys,
                    workspace,
                    gpu_output);
            };
            for (std::size_t index = 0; index < warmup; ++index)
            {
                run_direct_boot();
            }
            poseidon::gpu::gpu_check_cuda(
                cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t index = 0; index < iterations; ++index)
            {
                run_direct_boot();
            }
            poseidon::gpu::gpu_check_cuda(
                cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start);
            direct_gpu_boot_ms =
                static_cast<double>(elapsed.count()) /
                1000.0 / static_cast<double>(iterations);
            poseidon::Ciphertext direct_output;
            poseidon::gpu::GpuUploader::download_ciphertext(
                gpu_output, direct_output, context);
            poseidon::Plaintext direct_plaintext;
            decryptor.decrypt(direct_output, direct_plaintext);
            encoder.decode(direct_plaintext, direct_decoded);

            poseidon::gpu::GpuBootstrapWorkspace staged_workspace;
            poseidon::gpu::GpuCiphertextData staged_raw;
            poseidon::gpu::GpuCiphertextData staged_real;
            poseidon::gpu::GpuCiphertextData staged_imag;
            poseidon::gpu::GpuCiphertextData staged_eval_real;
            poseidon::gpu::GpuCiphertextData staged_eval_imag;
            poseidon::gpu::GpuCiphertextData staged_output;
            poseidon::gpu::GpuCiphertextData partitioned_quotient;
            poseidon::gpu::GpuCiphertextData partitioned_product;
            poseidon::gpu::GpuCiphertextData partitioned_remainder;
            poseidon::gpu::GpuCiphertextData partitioned_eval_real;
            poseidon::gpu::GpuCiphertextData partitioned_output;
            gpu_evaluator.bootstrap_stc_first_transform(
                gpu_input,
                profile.bootstrap_data,
                *profile.galois_keys,
                staged_workspace,
                staged_raw);
            staged_raw_level = static_cast<int>(
                context.crt_context()
                    ->get_context_data(staged_raw.meta.parms_id)
                    ->level());
            staged_raw_scale_log2 = std::log2(staged_raw.meta.scale);
            gpu_evaluator.bootstrap_extract_real(
                staged_raw,
                profile.bootstrap_data,
                *profile.galois_keys,
                staged_workspace,
                staged_real);
            gpu_evaluator.bootstrap_extract_imag(
                staged_raw,
                profile.bootstrap_data,
                *profile.galois_keys,
                staged_workspace,
                staged_imag);
            staged_branch_level = static_cast<int>(
                context.crt_context()
                    ->get_context_data(staged_real.meta.parms_id)
                    ->level());
            staged_branch_scale_log2 = std::log2(staged_real.meta.scale);
            staged_workspace.capture_eval_mod_stage_timing = true;
            gpu_evaluator.eval_mod_high_precision(
                staged_real,
                profile.bootstrap_data,
                *profile.relin_keys,
                staged_workspace,
                staged_eval_real);
            direct_eval_mod_timing = staged_workspace.eval_mod_stage_timing;
            direct_eval_mod_multiply_timings =
                staged_workspace.eval_mod_multiply_timings;
            staged_workspace.capture_eval_mod_stage_timing = false;
            gpu_evaluator.eval_mod_high_precision(
                staged_imag,
                profile.bootstrap_data,
                *profile.relin_keys,
                staged_workspace,
                staged_eval_imag);
            const auto &eval_mod_combines =
                profile.bootstrap_data.eval_mod.polynomial_combine_steps;
            if (eval_mod_combines.size() != 5)
            {
                throw std::runtime_error(
                    "degree-22 partition test requires five combine nodes");
            }
            poseidon::gpu::GpuBootstrapWorkspace quotient_workspace;
            poseidon::gpu::GpuBootstrapWorkspace remainder_workspace;
            const poseidon::gpu::GpuEvalModPolynomialPartition
                quotient_partition{
                    0, 1, eval_mod_combines.front().output_node};
            const poseidon::gpu::GpuEvalModPolynomialPartition
                remainder_partition{
                    1, 4, eval_mod_combines[3].output_node};
            gpu_evaluator.eval_mod_high_precision(
                staged_real,
                profile.bootstrap_data,
                *profile.relin_keys,
                quotient_workspace,
                partitioned_quotient,
                &quotient_partition);
            gpu_evaluator.eval_mod_degree22_root_product(
                partitioned_quotient,
                profile.bootstrap_data,
                *profile.relin_keys,
                quotient_workspace,
                partitioned_product);
            gpu_evaluator.eval_mod_high_precision(
                staged_real,
                profile.bootstrap_data,
                *profile.relin_keys,
                remainder_workspace,
                partitioned_remainder,
                &remainder_partition);
            gpu_evaluator.eval_mod_degree22_finish_partials(
                partitioned_product,
                partitioned_remainder,
                profile.bootstrap_data,
                *profile.relin_keys,
                remainder_workspace,
                partitioned_eval_real);
            staged_eval_mod_level = static_cast<int>(
                context.crt_context()
                    ->get_context_data(staged_eval_real.meta.parms_id)
                    ->level());
            staged_eval_mod_scale_log2 =
                std::log2(staged_eval_real.meta.scale);
            gpu_evaluator.bootstrap_stc_first_finalize(
                staged_eval_real,
                staged_eval_imag,
                profile.bootstrap_data,
                staged_workspace,
                staged_output);
            gpu_evaluator.bootstrap_stc_first_finalize(
                partitioned_eval_real,
                staged_eval_imag,
                profile.bootstrap_data,
                remainder_workspace,
                partitioned_output);
            poseidon::Ciphertext staged_host_output;
            poseidon::gpu::GpuUploader::download_ciphertext(
                staged_output, staged_host_output, context);
            poseidon::Plaintext staged_plaintext;
            decryptor.decrypt(staged_host_output, staged_plaintext);
            encoder.decode(staged_plaintext, staged_decoded);
            poseidon::Ciphertext partitioned_host_output;
            poseidon::gpu::GpuUploader::download_ciphertext(
                partitioned_output, partitioned_host_output, context);
            poseidon::Plaintext partitioned_plaintext;
            decryptor.decrypt(
                partitioned_host_output, partitioned_plaintext);
            encoder.decode(partitioned_plaintext, partitioned_decoded);

            direct_c2s_layer_ms.assign(
                profile.bootstrap_data.coeff_to_slot_matrix_qp.data().size(),
                0.0);
            poseidon::gpu::GpuBootstrapWorkspace timing_workspace;
            auto run_direct_c2s_stages = [&](bool record) {
                poseidon::gpu::GpuCiphertextData current;
                auto stage_start = std::chrono::steady_clock::now();
                gpu_evaluator.bootstrap_stc_first_prepare_c2s(
                    gpu_input,
                    profile.bootstrap_data,
                    *profile.galois_keys,
                    timing_workspace,
                    current);
                poseidon::gpu::gpu_check_cuda(
                    cudaStreamSynchronize(
                        poseidon::gpu::gpu_execution_stream()),
                    "cudaStreamSynchronize direct C2S prepare timing");
                if (record)
                {
                    direct_prepare_c2s_ms += elapsed_milliseconds(
                        stage_start, std::chrono::steady_clock::now());
                }
                for (std::size_t layer = 0;
                     layer < direct_c2s_layer_ms.size();
                     ++layer)
                {
                    poseidon::gpu::GpuCiphertextData next;
                    stage_start = std::chrono::steady_clock::now();
                    gpu_evaluator.dft_double_hoist_layer(
                        current,
                        profile.bootstrap_data.coeff_to_slot_matrix_qp,
                        layer,
                        *profile.galois_keys,
                        timing_workspace.coeff_to_slot_double_hoist,
                        next);
                    poseidon::gpu::gpu_check_cuda(
                        cudaStreamSynchronize(
                            poseidon::gpu::gpu_execution_stream()),
                        "cudaStreamSynchronize direct C2S layer timing");
                    if (record)
                    {
                        direct_c2s_layer_ms[layer] += elapsed_milliseconds(
                            stage_start, std::chrono::steady_clock::now());
                    }
                    current = std::move(next);
                }
            };
            for (std::size_t index = 0; index < warmup; ++index)
            {
                run_direct_c2s_stages(false);
            }
            for (std::size_t index = 0; index < iterations; ++index)
            {
                run_direct_c2s_stages(true);
            }
            direct_prepare_c2s_ms /= static_cast<double>(iterations);
            for (auto &layer_ms : direct_c2s_layer_ms)
            {
                layer_ms /= static_cast<double>(iterations);
            }
        }

        api.configure_native_bootstrap(0, std::move(profile));
        for (std::size_t device_index = 1;
             device_index < requested_gpu_count;
             ++device_index)
        {
            api.configure_native_bootstrap(
                static_cast<int>(device_index),
                std::move(secondary_profiles[device_index - 1]));
        }
        if (requested_gpu_count > 1)
        {
            api.configure_multi_gpu_bootstrap(
                boot_profile.profile_id,
                cuda_device_ids,
                c2s_device_limit,
                eval_mod_device_limit,
                s2c_device_limit);
        }

        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(
            0, 1, static_cast<int>(requested_gpu_count), api);
        const fhegpu::RuntimeResources resources{
            loaded_spec, std::nullopt, false};
        std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
        inputs.emplace(
            0, PoseidonGpuValue::from_host_ciphertext(std::move(ciphertext)));
        std::vector<std::complex<double>> decoded;

        for (std::size_t index = 0; index < warmup; ++index)
        {
            static_cast<void>(
                runtime.run(device_plan, resources, inputs));
        }
        double runtime_device_boot_ms = 0.0;
        poseidon::runtime_api::MultiGpuBootstrapTiming multi_gpu_timing_sum;
        for (std::size_t index = 0; index < iterations; ++index)
        {
            const auto artifact = runtime.run(device_plan, resources, inputs);
            runtime_device_boot_ms += milliseconds(
                artifact.timing.online_execution_nanoseconds);
            if (requested_gpu_count > 1)
            {
                const auto timing = api.last_multi_gpu_bootstrap_timing(
                    boot_profile.profile_id);
                if (!timing)
                {
                    throw std::runtime_error(
                        "multi-GPU bootstrap timing was not recorded");
                }
                accumulate_multi_gpu_timing(
                    multi_gpu_timing_sum, *timing);
            }
        }
        runtime_device_boot_ms /= static_cast<double>(iterations);

        for (std::size_t index = 0; index < warmup; ++index)
        {
            static_cast<void>(runtime.run(host_plan, resources, inputs));
        }
        double host_setup_ms = 0.0;
        double host_h2d_ms = 0.0;
        double runtime_boot_d2h_ms = 0.0;
        poseidon::runtime_api::MultiGpuBootstrapTiming
            host_multi_gpu_timing_sum;
        for (std::size_t index = 0; index < iterations; ++index)
        {
            const auto artifact = runtime.run(host_plan, resources, inputs);
            host_setup_ms += milliseconds(
                artifact.timing.setup_nanoseconds);
            host_h2d_ms += milliseconds(
                artifact.timing.initialization_nanoseconds);
            runtime_boot_d2h_ms += milliseconds(
                artifact.timing.online_execution_nanoseconds);
            if (requested_gpu_count > 1)
            {
                const auto timing = api.last_multi_gpu_bootstrap_timing(
                    boot_profile.profile_id);
                if (!timing)
                {
                    throw std::runtime_error(
                        "host-plan multi-GPU bootstrap timing was not recorded");
                }
                accumulate_multi_gpu_timing(
                    host_multi_gpu_timing_sum, *timing);
            }
            if (index + 1 == iterations)
            {
                const auto &output =
                    artifact.values.at(3).value.host_ciphertext();
                poseidon::Plaintext output_plaintext;
                decryptor.decrypt(output, output_plaintext);
                encoder.decode(output_plaintext, decoded);
            }
        }
        host_setup_ms /= static_cast<double>(iterations);
        host_h2d_ms /= static_cast<double>(iterations);
        runtime_boot_d2h_ms /= static_cast<double>(iterations);
        const double runtime_plan_ms =
            host_setup_ms + host_h2d_ms + runtime_boot_d2h_ms;
        const double runtime_vs_direct_delta_ms =
            runtime_device_boot_ms - direct_gpu_boot_ms;
        const double runtime_d2h_increment_ms =
            runtime_boot_d2h_ms - runtime_device_boot_ms;

        double max_abs_error = 0.0;
        double runtime_direct_max_abs_error = 0.0;
        double staged_direct_max_abs_error = 0.0;
        double partitioned_direct_max_abs_error = 0.0;
        for (std::size_t index = 0; index < message.size(); ++index)
        {
            max_abs_error = std::max(
                max_abs_error, std::abs(decoded.at(index) - message[index]));
            runtime_direct_max_abs_error = std::max(
                runtime_direct_max_abs_error,
                std::abs(decoded.at(index) - direct_decoded.at(index)));
            staged_direct_max_abs_error = std::max(
                staged_direct_max_abs_error,
                std::abs(staged_decoded.at(index) - direct_decoded.at(index)));
            partitioned_direct_max_abs_error = std::max(
                partitioned_direct_max_abs_error,
                std::abs(
                    partitioned_decoded.at(index) -
                    direct_decoded.at(index)));
        }
        if (!std::isfinite(max_abs_error) ||
            max_abs_error > kSemanticRegressionLimit)
        {
            throw std::runtime_error(
                "Runtime StC-first bootstrap semantic error exceeds the "
                "degree-22 regression bound: " +
                std::to_string(max_abs_error));
        }
        if (!std::isfinite(runtime_direct_max_abs_error) ||
            runtime_direct_max_abs_error > kRuntimeDirectTolerance)
        {
            throw std::runtime_error(
                "Runtime StC-first bootstrap differs from direct GPU "
                "execution: " +
                std::to_string(runtime_direct_max_abs_error) +
                " (high_precision=" +
                precise_double(runtime_direct_max_abs_error) + ")");
        }
        if (!std::isfinite(staged_direct_max_abs_error) ||
            staged_direct_max_abs_error > kRuntimeDirectTolerance)
        {
            throw std::runtime_error(
                "Staged StC-first bootstrap differs from monolithic GPU "
                "execution: " +
                std::to_string(staged_direct_max_abs_error));
        }
        if (!std::isfinite(partitioned_direct_max_abs_error) ||
            partitioned_direct_max_abs_error > kRuntimeDirectTolerance)
        {
            throw std::runtime_error(
                "Partitioned degree-22 EvalMod differs from monolithic GPU "
                "execution: " +
                std::to_string(partitioned_direct_max_abs_error));
        }

        std::cout << "[PASS] RuntimePlan native GPU StC-first bootstrap"
                  << " gpu_count=" << requested_gpu_count
                  << " degree=" << degree
                  << " input_level=" << boot_profile.input_level_min
                  << " output_level=" << boot_profile.output_level
                  << " output_scale_log2=" << boot_profile.output_scale_log2
                  << " runtime_direct_max_abs_error="
                  << runtime_direct_max_abs_error
                  << " staged_direct_max_abs_error="
                  << staged_direct_max_abs_error
                  << " partitioned_direct_max_abs_error="
                  << partitioned_direct_max_abs_error
                  << " source_semantic_max_abs_error=" << max_abs_error
                  << " regression_limit=" << kSemanticRegressionLimit
                  << '\n';
        std::cout
            << "[BENCH] warmup=" << warmup
            << " iterations=" << iterations
            << " gpu_count=" << requested_gpu_count
            << " s2c_device_limit=" << s2c_device_limit
            << " c2s_device_limit=" << c2s_device_limit
            << " eval_mod_device_limit=" << eval_mod_device_limit
            << " direct_gpu_boot_ms=" << direct_gpu_boot_ms
            << " runtime_device_boot_ms=" << runtime_device_boot_ms
            << " runtime_vs_direct_delta_ms="
            << runtime_vs_direct_delta_ms
            << " runtime_d2h_increment_ms=" << runtime_d2h_increment_ms
            << " runtime_boot_d2h_ms=" << runtime_boot_d2h_ms
            << " runtime_h2d_ms=" << host_h2d_ms
            << " runtime_setup_ms=" << host_setup_ms
            << " runtime_plan_ms=" << runtime_plan_ms
            << '\n';
        std::cout
            << "[STAGES] raw_level=" << staged_raw_level
            << " raw_scale_log2=" << staged_raw_scale_log2
            << " branch_level=" << staged_branch_level
            << " branch_scale_log2=" << staged_branch_scale_log2
            << " eval_mod_level=" << staged_eval_mod_level
            << " eval_mod_scale_log2=" << staged_eval_mod_scale_log2
            << '\n';
        std::cout << "[STAGES] c2s_giant_groups=";
        for (std::size_t index = 0; index < c2s_giant_groups.size(); ++index)
        {
            if (index != 0)
            {
                std::cout << ',';
            }
            std::cout << c2s_giant_groups[index];
        }
        std::cout << '\n';
        std::cout << "[STAGES] s2c_giant_groups=";
        for (std::size_t index = 0; index < s2c_giant_groups.size(); ++index)
        {
            if (index != 0)
            {
                std::cout << ',';
            }
            std::cout << s2c_giant_groups[index];
        }
        std::cout << '\n';
        std::cout
            << "[SINGLE_GPU_C2S] prepare_c2s_ms="
            << direct_prepare_c2s_ms;
        for (std::size_t layer = 0;
             layer < direct_c2s_layer_ms.size();
             ++layer)
        {
            std::cout << " layer" << layer << "_ms="
                      << direct_c2s_layer_ms[layer];
        }
        std::cout << '\n';
        std::cout
            << "[SINGLE_GPU_EVALMOD] input_preparation_ms="
            << direct_eval_mod_timing.input_preparation_ms
            << " basis_generation_ms="
            << direct_eval_mod_timing.basis_generation_ms
            << " leaf_evaluation_ms="
            << direct_eval_mod_timing.leaf_evaluation_ms
            << " bsgs_combine_ms="
            << direct_eval_mod_timing.bsgs_combine_ms
            << " double_angle_ms="
            << direct_eval_mod_timing.double_angle_ms
            << " output_alignment_ms="
            << direct_eval_mod_timing.output_alignment_ms
            << " total_ms=" << direct_eval_mod_timing.total_ms
            << '\n';
        for (const auto &timing : direct_eval_mod_multiply_timings)
        {
            std::cout
                << "[SINGLE_GPU_EVALMOD_MUL] label=" << timing.label
                << " q_count=" << timing.q_count
                << " decomposition_count=" << timing.decomposition_count
                << " is_square=" << (timing.is_square ? 1 : 0)
                << " gpu_ms=" << timing.gpu_ms
                << '\n';
        }
        if (requested_gpu_count > 1)
        {
            const double divisor = static_cast<double>(iterations);
            std::cout
                << "[MULTI_GPU_STAGES] prepare_c2s_ms="
                << multi_gpu_timing_sum.prepare_c2s_ms / divisor
                << " modraise_ms="
                << multi_gpu_timing_sum.modraise_ms / divisor
                << " eval_mod_branches_ms="
                << multi_gpu_timing_sum.eval_mod_branches_ms / divisor
                << " imag_result_copy_ms="
                << multi_gpu_timing_sum.imag_result_copy_ms / divisor
                << " finalize_ms="
                << multi_gpu_timing_sum.finalize_ms / divisor
                << " internal_total_ms="
                << multi_gpu_timing_sum.total_ms / divisor
                << " eval_mod_devices="
                << multi_gpu_timing_sum.eval_mod_device_count
                << '\n';
            for (std::size_t layer = 0;
                 layer < multi_gpu_timing_sum.s2c_layers.size();
                 ++layer)
            {
                const auto &timing = multi_gpu_timing_sum.s2c_layers[layer];
                std::cout
                    << "[MULTI_GPU_S2C_LAYER] layer=" << layer
                    << " active_devices=" << timing.active_device_count
                    << " fanout_partial_ms="
                    << timing.fanout_and_partial_compute_ms / divisor
                    << " qp_reduce_ms="
                    << timing.qp_reduction_ms / divisor
                    << " shared_moddown_rescale_ms="
                    << timing.shared_moddown_rescale_ms / divisor
                    << " total_ms=" << timing.total_ms / divisor
                    << '\n';
            }
            for (std::size_t layer = 0;
                 layer < multi_gpu_timing_sum.c2s_layers.size();
                 ++layer)
            {
                const auto &timing = multi_gpu_timing_sum.c2s_layers[layer];
                std::cout
                    << "[MULTI_GPU_C2S_LAYER] layer=" << layer
                    << " active_devices=" << timing.active_device_count
                    << " fanout_partial_ms="
                    << timing.fanout_and_partial_compute_ms / divisor
                    << " qp_reduce_ms="
                    << timing.qp_reduction_ms / divisor
                    << " shared_moddown_rescale_ms="
                    << timing.shared_moddown_rescale_ms / divisor
                    << " total_ms=" << timing.total_ms / divisor
                    << '\n';
            }
            std::cout
                << "[MULTI_GPU_HOST_PHASE] prepare_c2s_ms="
                << host_multi_gpu_timing_sum.prepare_c2s_ms / divisor
                << " eval_mod_branches_ms="
                << host_multi_gpu_timing_sum.eval_mod_branches_ms / divisor
                << " imag_result_copy_ms="
                << host_multi_gpu_timing_sum.imag_result_copy_ms / divisor
                << " finalize_ms="
                << host_multi_gpu_timing_sum.finalize_ms / divisor
                << " internal_total_ms="
                << host_multi_gpu_timing_sum.total_ms / divisor
                << '\n';
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
