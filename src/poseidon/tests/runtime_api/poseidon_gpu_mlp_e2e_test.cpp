#include "poseidon/basics/modulus.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/runtime.hpp"
#include "runtime/verifier.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

using Json = nlohmann::json;
using poseidon::runtime_api::PoseidonGpuApi;
using poseidon::runtime_api::PoseidonGpuValue;

constexpr int kCudaDevice = 0;
constexpr double kAbsoluteTolerance = 0.1;
constexpr double kRelativeTolerance = 6e-3;
constexpr double kImaginaryTolerance = 1e-2;
constexpr const char *kAllowNoBootEnv =
    "POSEIDON_GPU_MLP_ALLOW_NO_BOOT";

bool env_flag_enabled(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" &&
           text != "off" && text != "OFF";
}

struct Comparison
{
    double max_abs = 0.0;
    double rms = 0.0;
    double max_relative = 0.0;
    bool within_tolerance = true;
};

Json read_json(const std::filesystem::path &path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("cannot open JSON file: " + path.string());
    }
    Json value;
    input >> value;
    return value;
}

std::vector<double> read_numbers(const Json &root, const char *member,
                                 std::size_t expected_size)
{
    const auto &array = root.at(member);
    if (!array.is_array() || array.size() != expected_size)
    {
        throw std::runtime_error(std::string(member) + " has the wrong length");
    }
    std::vector<double> result;
    result.reserve(array.size());
    for (const auto &value : array)
    {
        const double number = value.get<double>();
        if (!std::isfinite(number))
        {
            throw std::runtime_error(std::string(member) + " contains a non-finite number");
        }
        result.push_back(number);
    }
    return result;
}

std::vector<double> pack_mlp_input(const std::vector<double> &logical)
{
    if (logical.size() != 784)
    {
        throw std::runtime_error("MLP logical input must contain 784 values");
    }
    std::vector<double> packed;
    packed.reserve(1600);
    for (std::size_t block_index = 0; block_index < 8; ++block_index)
    {
        std::vector<double> block(100, 0.0);
        const std::size_t begin = block_index * 100;
        const std::size_t count =
            std::min<std::size_t>(100, logical.size() - begin);
        std::copy_n(logical.begin() + begin, count, block.begin());
        packed.insert(packed.end(), block.begin(), block.end());
        packed.insert(packed.end(), block.begin(), block.end());
    }
    return packed;
}

std::uint32_t exact_log2(std::uint64_t value)
{
    if (value < 2 || (value & (value - 1)) != 0)
    {
        throw std::runtime_error("poly_degree must be a power of two");
    }
    std::uint32_t result = 0;
    while (value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

poseidon::PoseidonContext make_context(const fhegpu::OperatorSpec &spec)
{
    std::vector<int> modulus_bits = spec.rns_moduli_log2;
    modulus_bits.insert(modulus_bits.end(), 2, spec.max_modulus_log2);
    const auto moduli = poseidon::CoeffModulus::Create(spec.poly_degree, modulus_bits);
    const std::size_t q_count = spec.rns_moduli_log2.size();
    std::vector<poseidon::Modulus> q(moduli.begin(), moduli.begin() + q_count);
    std::vector<poseidon::Modulus> p(moduli.begin() + q_count, moduli.end());
    const std::uint32_t log_n = exact_log2(spec.poly_degree);
    poseidon::ParametersLiteral parameters(
        CKKS, log_n, log_n - 1,
        static_cast<std::uint32_t>(spec.default_scale_log2), 0, 0,
        poseidon::Modulus(0), q, p, poseidon::sec_level_type::none);
    return poseidon::PoseidonContext(parameters);
}

const fhegpu::ValueDesc &find_value(const fhegpu::RuntimePlan &plan,
                                    fhegpu::ValueId id)
{
    const auto found = std::find_if(
        plan.values.begin(), plan.values.end(),
        [id](const fhegpu::ValueDesc &value) { return value.id == id; });
    if (found == plan.values.end())
    {
        throw std::runtime_error("missing ValueDesc " + std::to_string(id));
    }
    return *found;
}

std::size_t count_transfers(const fhegpu::RuntimePlan &plan)
{
    std::size_t count = 0;
    const auto add_phase = [&count](const std::vector<fhegpu::Instruction> &phase) {
        count += static_cast<std::size_t>(std::count_if(
            phase.begin(), phase.end(), [](const fhegpu::Instruction &instruction) {
                return std::holds_alternative<fhegpu::CommAction>(instruction.body);
            }));
    };
    add_phase(plan.initialization);
    add_phase(plan.execution);
    add_phase(plan.finalization);
    return count;
}

std::size_t count_boots(const fhegpu::RuntimePlan &plan)
{
    return static_cast<std::size_t>(std::count_if(
        plan.execution.begin(), plan.execution.end(),
        [](const fhegpu::Instruction &instruction) {
            const auto *op = std::get_if<fhegpu::ComputeOp>(&instruction.body);
            return op != nullptr && op->kind == fhegpu::ComputeKind::Boot;
        }));
}

PoseidonGpuValue download(PoseidonGpuApi &api, fhegpu::ValueId id,
                          const fhegpu::Place &source,
                          const PoseidonGpuValue &value)
{
    if (source.kind == fhegpu::PlaceKind::Host)
    {
        return value;
    }
    fhegpu::CommAction action;
    action.id = 0;
    action.kind = fhegpu::CommKind::Transfer;
    action.hint = fhegpu::CommHint::PointToPoint;
    action.inputs = {id};
    action.outputs = {id + 1};
    action.sources = {source};
    action.destinations = {{fhegpu::PlaceKind::Host, 0, 0}};
    action.output_types = {fhegpu::ValueKind::Ciphertext};
    auto handle = api.communicate_async(action, {value});
    auto outputs = api.wait(handle);
    if (outputs.size() != 1)
    {
        throw std::runtime_error("GPU result download produced the wrong output count");
    }
    return std::move(outputs.front());
}

Comparison compare(const std::vector<double> &expected,
                   const std::vector<double> &actual)
{
    Comparison result;
    double squared_error = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        const double difference = std::abs(expected[index] - actual[index]);
        result.max_abs = std::max(result.max_abs, difference);
        squared_error += difference * difference;
        const double relative =
            difference / std::max(std::abs(expected[index]), kAbsoluteTolerance);
        result.max_relative = std::max(result.max_relative, relative);
        if (difference > kAbsoluteTolerance +
                             kRelativeTolerance * std::abs(expected[index]))
        {
            result.within_tolerance = false;
        }
    }
    result.rms = std::sqrt(squared_error / expected.size());
    return result;
}

Json comparison_json(const Comparison &comparison)
{
    return {{"max_abs", comparison.max_abs},
            {"rms", comparison.rms},
            {"max_relative", comparison.max_relative},
            {"within_tolerance", comparison.within_tolerance}};
}

void write_json(const std::filesystem::path &path, const Json &value)
{
    if (!path.parent_path().empty())
    {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output)
    {
        throw std::runtime_error("cannot write JSON file: " + path.string());
    }
    output << value.dump(2) << '\n';
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 7)
    {
        std::cerr << "usage: poseidon_runtime_gpu_mlp_e2e PLAN OPERATOR_SPEC "
                     "BUNDLE_DIR FIXTURE MOCK_RESULT REPORT_JSON\n";
        return 2;
    }

    try
    {
        const auto loaded_plan = fhegpu::RuntimePlanJsonReader::read_file(argv[1]);
        const auto loaded_spec = fhegpu::OperatorSpecReader::read_file(argv[2]);
        const Json fixture = read_json(argv[4]);
        const Json mock_result = read_json(argv[5]);
        const auto &plan = loaded_plan.plan;
        if (plan.target.world_size != 1 || plan.target.device_counts != std::vector<int>{1} ||
            plan.external_inputs.size() != 1 || plan.final_outputs.size() != 1)
        {
            throw std::runtime_error(
                "GPU MLP test requires one rank, one device, one input, and one output");
        }
        if (fixture.at("format_version") != 1 ||
            mock_result.at("format_version") != 1 ||
            mock_result.at("passed") != true ||
            fixture.at("seed") != mock_result.at("seed") ||
            fixture.at("model_sha256") != mock_result.at("model_sha256"))
        {
            throw std::runtime_error("fixture and MockVecApi result do not match");
        }
        const std::size_t plan_boots = count_boots(plan);
        const bool allow_no_boot = env_flag_enabled(kAllowNoBootEnv);
        if (plan_boots == 0 && !allow_no_boot)
        {
            throw std::runtime_error(
                "GPU MLP RuntimePlan contains no Host Boot; set " +
                std::string(kAllowNoBootEnv) +
                "=1 to run a no-Boot calibration plan");
        }

        const auto requirements = fhegpu::PlanVerifier::verify(
            plan, loaded_spec, false);
        poseidon::PoseidonContext context = make_context(loaded_spec.spec);
        poseidon::KeyGenerator key_generator(context);
        auto secret_key =
            std::make_shared<poseidon::SecretKey>(key_generator.secret_key());
        auto public_key = std::make_shared<poseidon::PublicKey>();
        auto relin_keys = std::make_shared<poseidon::RelinKeys>();
        auto galois_keys = std::make_shared<poseidon::GaloisKeys>();
        bool needs_relin = false;
        std::set<int> rotation_steps;
        for (const auto &key : requirements.keys)
        {
            if (key.kind == fhegpu::KeyKind::Relin)
            {
                needs_relin = true;
            }
            else if (key.kind == fhegpu::KeyKind::Galois)
            {
                if (!key.rotation_step)
                {
                    throw std::runtime_error("Galois key requirement has no rotation step");
                }
                rotation_steps.insert(*key.rotation_step);
            }
        }

        const auto key_start = std::chrono::steady_clock::now();
        key_generator.create_public_key(*public_key);
        if (needs_relin)
        {
            key_generator.create_relin_keys(*relin_keys);
        }
        if (!rotation_steps.empty())
        {
            key_generator.create_galois_keys(
                std::vector<int>(rotation_steps.begin(), rotation_steps.end()),
                *galois_keys);
        }
        const auto key_finish = std::chrono::steady_clock::now();

        const auto logical_input = read_numbers(fixture, "input", 784);
        const auto python_output = read_numbers(fixture, "python_output", 10);
        const auto mock_output = read_numbers(mock_result, "output", 10);
        const fhegpu::ValueId input_id = plan.external_inputs.front();
        const auto &input_desc = find_value(plan, input_id);
        poseidon::CKKSEncoder encoder(context);
        poseidon::Plaintext input_plain;
        encoder.encode(
            pack_mlp_input(logical_input),
            context.crt_context()->parms_id_map().at(
                static_cast<std::uint32_t>(input_desc.level)),
            std::ldexp(1.0, input_desc.scale_log2), input_plain);
        poseidon::Encryptor encryptor(context, *public_key);
        poseidon::Ciphertext input_cipher;
        encryptor.encrypt(input_plain, input_cipher);

        PoseidonGpuApi api(loaded_spec.spec.context_id, context, kCudaDevice,
                           relin_keys, galois_keys, public_key, secret_key);
        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(0, 1, 1, api);
        const fhegpu::RuntimeResources resources{
            loaded_spec, std::filesystem::path(argv[3]), false};
        std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
        inputs.emplace(input_id,
                       PoseidonGpuValue::from_host_ciphertext(std::move(input_cipher)));
        const auto run_start = std::chrono::steady_clock::now();
        const auto artifact = runtime.run(loaded_plan, resources, inputs);
        const auto run_finish = std::chrono::steady_clock::now();

        const fhegpu::ValueId output_id = plan.final_outputs.front();
        const auto &output_desc = find_value(plan, output_id);
        auto host_output = download(
            api, output_id, output_desc.place, artifact.values.at(output_id).value);
        poseidon::Decryptor decryptor(context, *secret_key);
        poseidon::Plaintext output_plain;
        decryptor.decrypt(host_output.host_ciphertext(), output_plain);
        std::vector<std::complex<double>> decoded;
        encoder.decode(output_plain, decoded);
        std::vector<double> poseidon_output;
        poseidon_output.reserve(10);
        double max_imaginary = 0.0;
        for (std::size_t index = 0; index < 10; ++index)
        {
            poseidon_output.push_back(decoded.at(index).real());
            max_imaginary = std::max(max_imaginary,
                                     std::abs(decoded.at(index).imag()));
        }

        const Comparison against_python = compare(python_output, poseidon_output);
        const Comparison against_mock = compare(mock_output, poseidon_output);
        const bool passed = against_python.within_tolerance &&
                            against_mock.within_tolerance &&
                            max_imaginary <= kImaginaryTolerance &&
                            artifact.timing.boot_calls == plan_boots;
        const double key_seconds =
            std::chrono::duration<double>(key_finish - key_start).count();
        const double runtime_seconds =
            std::chrono::duration<double>(run_finish - run_start).count();
        const double compute_seconds =
            artifact.timing.compute_including_boot_nanoseconds * 1e-9;
        const double boot_seconds = artifact.timing.boot_nanoseconds * 1e-9;
        const double setup_seconds = artifact.timing.setup_nanoseconds * 1e-9;
        const double initialization_seconds =
            artifact.timing.initialization_nanoseconds * 1e-9;
        const double online_execution_seconds =
            artifact.timing.online_execution_nanoseconds * 1e-9;
        const Json report{
            {"format_version", 1},
            {"passed", passed},
            {"cuda_device", kCudaDevice},
            {"seed", fixture.at("seed")},
            {"model_sha256", fixture.at("model_sha256")},
            {"plan_sha256", loaded_plan.source_sha256},
            {"operator_spec_sha256", loaded_spec.source_sha256},
            {"poly_degree", loaded_spec.spec.poly_degree},
            {"q_modulus_count", loaded_spec.spec.rns_moduli_log2.size()},
            {"rotation_key_count", rotation_steps.size()},
            {"allow_no_boot", allow_no_boot},
            {"transfer_count", count_transfers(plan)},
            {"key_generation_seconds", key_seconds},
            {"runtime_seconds", runtime_seconds},
            {"runtime_timing",
             {{"setup_seconds", setup_seconds},
              {"initialization_seconds", initialization_seconds},
              {"online_execution_seconds", online_execution_seconds},
              {"compute_calls", artifact.timing.compute_calls},
              {"boot_calls", artifact.timing.boot_calls},
              {"compute_including_boot_seconds", compute_seconds},
              {"boot_seconds", boot_seconds},
              {"compute_excluding_boot_seconds", compute_seconds - boot_seconds}}},
            {"input_level", input_desc.level},
            {"input_scale_log2", input_desc.scale_log2},
            {"output_level", output_desc.level},
            {"output_scale_log2", output_desc.scale_log2},
            {"tolerances",
             {{"absolute", kAbsoluteTolerance},
              {"relative", kRelativeTolerance},
              {"imaginary", kImaginaryTolerance}}},
            {"python_comparison", comparison_json(against_python)},
            {"mock_comparison", comparison_json(against_mock)},
            {"max_imaginary", max_imaginary},
            {"python_output", python_output},
            {"mock_output", mock_output},
            {"poseidon_output", poseidon_output},
        };
        write_json(argv[6], report);
        std::cout << (passed ? "PASS" : "FAIL")
                  << " boots=" << artifact.timing.boot_calls
                  << " transfers=" << count_transfers(plan)
                  << " rotations=" << rotation_steps.size()
                  << " key_seconds=" << key_seconds
                  << " runtime_seconds=" << runtime_seconds
                  << " initialization_seconds=" << initialization_seconds
                  << " online_execution_seconds=" << online_execution_seconds
                  << " boot_seconds=" << boot_seconds
                  << " python_max_abs=" << against_python.max_abs
                  << " max_imaginary=" << max_imaginary << '\n'
                  << "report=" << argv[6] << '\n';
        return passed ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
