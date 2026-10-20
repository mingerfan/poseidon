#include "poseidon/ckks_encoder.h"
#include "poseidon/advance/homomorphic_mod.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_bootstrap_profile.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "runtime/runtime.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
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
constexpr double kTolerance = 3.0e-3;
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
        /*hamming_weight=*/model_profile ? 192 : 0,
        kQ0Level,
        poseidon::Modulus(0),
        {},
        {},
        poseidon::sec_level_type::none);
    auto q = mixed_bootstrap_q_chain();
    if (model_profile) q.insert(q.begin(), 16, 32);
    parameters.set_log_modulus(q, std::vector<std::uint32_t>(model_profile ? 25 : 9, 32));
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
    const fhegpu::BootProfile &profile)
{
    const auto host = host_place();
    const auto device = device_place();
    const int input_level = loaded_spec.spec.level_upper_bound;

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

std::vector<std::complex<double>> make_message(std::size_t slot_count)
{
    std::vector<std::complex<double>> result(slot_count);
    for (std::size_t index = 0; index < slot_count; ++index)
    {
        result[index] = {
            static_cast<double>((index % 17) + 1) / 32.0,
            std::getenv("POSEIDON_RUNTIME_BOOTSTRAP_REAL_PROFILE") != nullptr
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
            "POSEIDON_RUNTIME_BOOTSTRAP_DEGREE", 16384);
        auto context = make_context(degree);
        poseidon::KeyGenerator key_generator(context);
        poseidon::PublicKey public_key;
        key_generator.create_public_key(public_key);
        poseidon::Encryptor encryptor(context, public_key);
        poseidon::Decryptor decryptor(context, key_generator.secret_key());
        poseidon::CKKSEncoder encoder(context);

        PoseidonGpuApi api(kContextId, context, kCudaDevice);
        poseidon::gpu::GpuBootstrapProfileConfig profile_config;
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
        profile_config.eval_mod_log_scale = 60;
        profile_config.bootstrap_ratio = 128;
        profile_config.profile_id = "poseidon-gpu-degree58-da3-full-cosine-scale60-ratio128-real";
        auto profile = poseidon::gpu::GpuBootstrapProfileBuilder::build(
            context, key_generator, kCudaDevice, profile_config);
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
        encoder.encode(message, std::ldexp(1.0, kLogScale), plaintext);
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
                std::exp2(profile_config.eval_mod_log_scale), 0, 7, 3, 25, 0, 59);
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

        std::ofstream samples(std::getenv("PRECISION_PROBE_OUTPUT"));
        samples << std::setprecision(17) << "{\"expected\":[";
        for(std::size_t i=0;i<message.size();++i){if(i)samples<<",";samples<<message[i].real();}
        samples << "],\"actual\":[";
        for(std::size_t i=0;i<decoded.size();++i){if(i)samples<<",";samples<<decoded[i].real();}
        samples << "]}\n"; samples.close();
        double max_abs_error = 0.0;
        for (std::size_t index = 0; index < message.size(); ++index)
        {
            max_abs_error = std::max(
                max_abs_error, std::abs(decoded.at(index) - message[index]));
        }
        if (max_abs_error > kTolerance)
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
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
