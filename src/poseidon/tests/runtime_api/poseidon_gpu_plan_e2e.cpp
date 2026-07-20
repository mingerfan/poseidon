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

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
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
constexpr double kTolerance = 5e-3;

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

PoseidonGpuValue download(PoseidonGpuApi &api, fhegpu::ValueId id,
                          const fhegpu::Place &source,
                          const PoseidonGpuValue &value)
{
    fhegpu::CommAction action;
    action.id = 1;
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

void write_report(const std::filesystem::path &path, const Json &report)
{
    if (!path.parent_path().empty())
    {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output)
    {
        throw std::runtime_error("cannot write report: " + path.string());
    }
    output << report.dump(2) << '\n';
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: poseidon_runtime_gpu_plan_e2e "
                     "RUNTIME_PLAN OPERATOR_SPEC REPORT\n";
        return 2;
    }

    try
    {
        const auto loaded_plan = fhegpu::RuntimePlanJsonReader::read_file(argv[1]);
        const auto loaded_spec = fhegpu::OperatorSpecReader::read_file(argv[2]);
        const auto &plan = loaded_plan.plan;
        if (plan.target.world_size != 1 || plan.target.device_counts != std::vector<int>{1} ||
            plan.external_inputs.size() != 1 || plan.final_outputs.size() != 1)
        {
            throw std::runtime_error(
                "GPU end-to-end test requires one rank, one device, one input, and one output");
        }

        poseidon::PoseidonContext context = make_context(loaded_spec.spec);
        poseidon::KeyGenerator key_generator(context);
        auto relin_keys = std::make_shared<poseidon::RelinKeys>();
        key_generator.create_relin_keys(*relin_keys);
        poseidon::PublicKey public_key;
        key_generator.create_public_key(public_key);
        poseidon::Encryptor encryptor(context, public_key);
        poseidon::Decryptor decryptor(context, key_generator.secret_key());
        poseidon::CKKSEncoder encoder(context);

        const std::vector<double> input{-0.75, -0.5, -0.25, 0.125,
                                        0.25, 0.5, 0.75, 0.875};
        std::vector<double> expected;
        expected.reserve(input.size());
        for (double value : input)
        {
            expected.push_back(value * value * value * value + value);
        }

        const fhegpu::ValueId input_id = plan.external_inputs.front();
        const auto &input_desc = find_value(plan, input_id);
        poseidon::Plaintext plaintext;
        encoder.encode(input, std::ldexp(1.0, input_desc.scale_log2), plaintext);
        poseidon::Ciphertext ciphertext;
        encryptor.encrypt(plaintext, ciphertext);

        PoseidonGpuApi api(loaded_spec.spec.context_id, context, kCudaDevice, relin_keys);
        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(0, 1, 1, api);
        const fhegpu::RuntimeResources resources{loaded_spec, std::nullopt, false};
        std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
        inputs.emplace(input_id,
                       PoseidonGpuValue::from_host_ciphertext(std::move(ciphertext)));
        const auto artifact = runtime.run(loaded_plan, resources, inputs);

        const fhegpu::ValueId output_id = plan.final_outputs.front();
        const auto &output_desc = find_value(plan, output_id);
        auto host_output = download(
            api, output_id, output_desc.place, artifact.values.at(output_id).value);
        poseidon::Plaintext result_plaintext;
        decryptor.decrypt(host_output.host_ciphertext(), result_plaintext);
        std::vector<std::complex<double>> decoded;
        encoder.decode(result_plaintext, decoded);

        double max_abs_error = 0.0;
        double max_imaginary = 0.0;
        std::vector<double> actual;
        actual.reserve(expected.size());
        for (std::size_t index = 0; index < expected.size(); ++index)
        {
            actual.push_back(decoded.at(index).real());
            max_abs_error = std::max(
                max_abs_error, std::abs(decoded.at(index).real() - expected[index]));
            max_imaginary = std::max(max_imaginary,
                                     std::abs(decoded.at(index).imag()));
        }
        const bool passed = max_abs_error <= kTolerance &&
                            max_imaginary <= kTolerance;
        const Json report{
            {"format_version", 1},
            {"passed", passed},
            {"expression", "x^4 + x"},
            {"cuda_device", kCudaDevice},
            {"plan_sha256", loaded_plan.source_sha256},
            {"operator_spec_sha256", loaded_spec.source_sha256},
            {"input_level", input_desc.level},
            {"input_scale_log2", input_desc.scale_log2},
            {"output_level", output_desc.level},
            {"output_scale_log2", output_desc.scale_log2},
            {"tolerance", kTolerance},
            {"max_abs_error", max_abs_error},
            {"max_imaginary", max_imaginary},
            {"input", input},
            {"expected", expected},
            {"actual", actual},
        };
        write_report(argv[3], report);
        std::cout << (passed ? "PASS" : "FAIL")
                  << " max_abs_error=" << max_abs_error
                  << " max_imaginary=" << max_imaginary << '\n'
                  << "report=" << argv[3] << '\n';
        return passed ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
