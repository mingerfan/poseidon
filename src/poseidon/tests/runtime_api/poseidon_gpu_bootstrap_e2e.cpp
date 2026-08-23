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
#include <iostream>
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
    const fhegpu::BootProfile &profile,
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
    plan.target = make_target(loaded_spec);
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

        PoseidonGpuApi api(kContextId, context, kCudaDevice);
        auto profile = poseidon::gpu::GpuBootstrapProfileBuilder::build(
            context, key_generator, kCudaDevice);
        const auto boot_profile =
            poseidon::runtime_api::make_native_boot_profile(profile);
        const auto loaded_spec = make_operator_spec(context, boot_profile);
        const auto device_plan = make_plan(
            loaded_spec, boot_profile, /*return_to_host=*/false);
        const auto host_plan = make_plan(
            loaded_spec, boot_profile, /*return_to_host=*/true);

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
        double direct_gpu_boot_ms = 0.0;
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
        }

        api.configure_native_bootstrap(0, std::move(profile));

        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(0, 1, 1, api);
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
        for (std::size_t index = 0; index < iterations; ++index)
        {
            const auto artifact = runtime.run(device_plan, resources, inputs);
            runtime_device_boot_ms += milliseconds(
                artifact.timing.online_execution_nanoseconds);
        }
        runtime_device_boot_ms /= static_cast<double>(iterations);

        for (std::size_t index = 0; index < warmup; ++index)
        {
            static_cast<void>(runtime.run(host_plan, resources, inputs));
        }
        double host_setup_ms = 0.0;
        double host_h2d_ms = 0.0;
        double runtime_boot_d2h_ms = 0.0;
        for (std::size_t index = 0; index < iterations; ++index)
        {
            const auto artifact = runtime.run(host_plan, resources, inputs);
            host_setup_ms += milliseconds(
                artifact.timing.setup_nanoseconds);
            host_h2d_ms += milliseconds(
                artifact.timing.initialization_nanoseconds);
            runtime_boot_d2h_ms += milliseconds(
                artifact.timing.online_execution_nanoseconds);
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
        for (std::size_t index = 0; index < message.size(); ++index)
        {
            max_abs_error = std::max(
                max_abs_error, std::abs(decoded.at(index) - message[index]));
            runtime_direct_max_abs_error = std::max(
                runtime_direct_max_abs_error,
                std::abs(decoded.at(index) - direct_decoded.at(index)));
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
                std::to_string(runtime_direct_max_abs_error));
        }

        std::cout << "[PASS] RuntimePlan native GPU StC-first bootstrap"
                  << " degree=" << degree
                  << " input_level=" << boot_profile.input_level_min
                  << " output_level=" << boot_profile.output_level
                  << " output_scale_log2=" << boot_profile.output_scale_log2
                  << " runtime_direct_max_abs_error="
                  << runtime_direct_max_abs_error
                  << " source_semantic_max_abs_error=" << max_abs_error
                  << " regression_limit=" << kSemanticRegressionLimit
                  << '\n';
        std::cout
            << "[BENCH] warmup=" << warmup
            << " iterations=" << iterations
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
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
