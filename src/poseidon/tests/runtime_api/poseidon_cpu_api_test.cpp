#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/runtime_api/poseidon_cpu_api.h"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

using poseidon::runtime_api::PoseidonCpuApi;
using poseidon::runtime_api::PoseidonCpuValue;

int tests_run = 0;

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void run_test(const char *name, const std::function<void()> &test)
{
    test();
    ++tests_run;
    std::cout << "[PASS] " << name << '\n';
}

fhegpu::LoadedOperatorSpec make_operator_spec(const poseidon::PoseidonContext &context,
                                              const std::string &context_id)
{
    fhegpu::OperatorSpec spec;
    spec.id = "poseidon-cpu-api-test";
    spec.version = 1;
    spec.status = "test";
    spec.target_id = "poseidon-ckks-cpu";
    spec.rescale_mode = fhegpu::RescaleMode::Eager;
    spec.context_id = context_id;

    const auto parameters = context.parameters_literal();
    spec.poly_degree = parameters->degree();
    for (const auto &modulus : parameters->q())
    {
        spec.rns_moduli_log2.push_back(modulus.bit_count());
    }
    spec.max_modulus_log2 =
        *std::max_element(spec.rns_moduli_log2.begin(), spec.rns_moduli_log2.end());
    spec.default_scale_log2 = static_cast<int>(parameters->log_scale());
    spec.level_lower_bound = 0;
    spec.level_upper_bound = static_cast<int>(parameters->q().size() - 1);

    fhegpu::OperatorSupport add_plain;
    add_plain.supported = true;
    spec.operators.emplace(fhegpu::ComputeKind::AddCP, std::move(add_plain));

    return {std::move(spec), "sha256:"
                             "0000000000000000000000000000000000000000000000000000000000000000"};
}

fhegpu::RuntimePlan make_add_plain_plan(const fhegpu::LoadedOperatorSpec &loaded_spec, int level,
                                        int scale_log2)
{
    const fhegpu::Place host{fhegpu::PlaceKind::Host, 0, 0};
    fhegpu::RuntimePlan plan;
    plan.plan_id = 1;
    plan.target.target_id = "poseidon-ckks-cpu";
    plan.target.world_size = 1;
    plan.target.device_counts = {0};
    plan.target.capability_version = 1;
    plan.target.operator_spec = {loaded_spec.spec.id, loaded_spec.spec.version,
                                 loaded_spec.source_sha256};
    plan.values = {
        {0, fhegpu::ValueKind::Ciphertext, host, loaded_spec.spec.context_id, level, scale_log2,
         true, 2},
        {1, fhegpu::ValueKind::Plaintext, host, loaded_spec.spec.context_id, level, scale_log2,
         true, 1},
        {2, fhegpu::ValueKind::Ciphertext, host, loaded_spec.spec.context_id, level, scale_log2,
         true, 2},
    };
    plan.external_inputs = {0};
    plan.initialization = {
        {0, fhegpu::EncodeOp{fhegpu::InlineEncodePayload{{0.5, -1.0, 2.0, 3.0}}, 1}},
    };
    plan.execution = {
        {1, fhegpu::ComputeOp{fhegpu::ComputeKind::AddCP, {0, 1}, 2, host, {}}},
    };
    plan.final_outputs = {2};
    return plan;
}

void test_single_process_host_add_plain()
{
    constexpr std::uint32_t degree = 4096;
    const std::string context_id = "poseidon-cpu-test-context";
    poseidon::ParametersLiteralDefault parameters(CKKS, degree, poseidon::sec_level_type::tc128);
    poseidon::PoseidonContext context(parameters);
    const int level = static_cast<int>(parameters.q().size() - 1);
    const int scale_log2 = static_cast<int>(parameters.log_scale());
    const double scale = std::ldexp(1.0, scale_log2);

    poseidon::KeyGenerator key_generator(context);
    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::Decryptor decryptor(context, key_generator.secret_key());
    poseidon::CKKSEncoder encoder(context);

    const std::vector<double> input{1.0, 2.0, 3.0, 4.0};
    poseidon::Plaintext input_plain;
    encoder.encode(input, scale, input_plain);
    poseidon::Ciphertext input_cipher;
    encryptor.encrypt(input_plain, input_cipher);

    const auto operator_spec = make_operator_spec(context, context_id);
    const auto plan = make_add_plain_plan(operator_spec, level, scale_log2);
    const fhegpu::LoadedRuntimePlan loaded_plan{
        plan, "sha256:"
              "1111111111111111111111111111111111111111111111111111111111111111"};
    const fhegpu::RuntimeResources resources{operator_spec, std::nullopt, false};

    PoseidonCpuApi api(context_id, context);
    fhegpu::SequentialRuntime<PoseidonCpuApi> runtime(0, 1, 0, api);
    std::unordered_map<fhegpu::ValueId, PoseidonCpuValue> inputs;
    inputs.emplace(0, PoseidonCpuValue::from_ciphertext(std::move(input_cipher)));

    const auto artifact = runtime.run(loaded_plan, resources, inputs);
    const auto &result_cipher = artifact.values.at(2).value.ciphertext();
    poseidon::Plaintext result_plain;
    decryptor.decrypt(result_cipher, result_plain);
    std::vector<std::complex<double>> result;
    encoder.decode(result_plain, result);

    const std::vector<double> expected{1.5, 1.0, 5.0, 7.0};
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        require(std::abs(result[i].real() - expected[i]) < 1e-5,
                "unexpected AddCP result at slot " + std::to_string(i));
        require(std::abs(result[i].imag()) < 1e-5,
                "unexpected imaginary component at slot " + std::to_string(i));
    }
}

void test_rejects_multi_process_target()
{
    poseidon::ParametersLiteralDefault parameters(CKKS, 4096, poseidon::sec_level_type::tc128);
    poseidon::PoseidonContext context(parameters);
    const std::string context_id = "poseidon-cpu-test-context";
    auto operator_spec = make_operator_spec(context, context_id);
    PoseidonCpuApi api(context_id, context);

    fhegpu::TargetConfig target;
    target.target_id = "poseidon-ckks-cpu";
    target.world_size = 2;
    target.device_counts = {0, 0};
    target.capability_version = 1;

    bool rejected = false;
    try
    {
        api.preflight("sha256:"
                      "1111111111111111111111111111111111111111111111111111111111111111",
                      false, target, operator_spec.spec, {});
    }
    catch (const std::exception &error)
    {
        rejected = std::string(error.what()).find("one process") != std::string::npos;
    }
    require(rejected, "multi-process target was not rejected clearly");
}

} // namespace

int main()
{
    try
    {
        run_test("single-process Host AddCP", test_single_process_host_add_plain);
        run_test("reject multi-process target", test_rejects_multi_process_target);
        std::cout << tests_run << " Poseidon CPU Runtime Api tests passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
