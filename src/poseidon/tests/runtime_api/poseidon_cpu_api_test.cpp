#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/runtime_api/poseidon_cpu_api.h"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
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
    fhegpu::OperatorSupport add_cipher;
    add_cipher.supported = true;
    spec.operators.emplace(fhegpu::ComputeKind::AddCC, std::move(add_cipher));

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

void test_decrypt_reencrypt_boot()
{
    constexpr std::uint32_t degree = 4096;
    const std::string context_id = "poseidon-cpu-boot-test-context";
    poseidon::ParametersLiteralDefault parameters(CKKS, degree,
                                                  poseidon::sec_level_type::tc128);
    poseidon::PoseidonContext context(parameters);
    poseidon::KeyGenerator key_generator(context);
    auto public_key = std::make_shared<poseidon::PublicKey>();
    auto secret_key = std::make_shared<poseidon::SecretKey>(key_generator.secret_key());
    key_generator.create_public_key(*public_key);

    const std::vector<double> input{0.25, -0.5, 1.5, 2.0};
    poseidon::CKKSEncoder encoder(context);
    poseidon::Plaintext input_plain;
    encoder.encode(input, context.crt_context()->parms_id_map().at(0),
                   parameters.scale(), input_plain);
    poseidon::Encryptor encryptor(context, *public_key);
    poseidon::Ciphertext input_cipher;
    encryptor.encrypt(input_plain, input_cipher);

    const int target_level = static_cast<int>(parameters.q().size() - 1);
    const int scale_log2 = static_cast<int>(parameters.log_scale());
    PoseidonCpuApi api(context_id, context, {}, {}, public_key, secret_key);
    fhegpu::ComputeOp op{
        fhegpu::ComputeKind::Boot,
        {0},
        1,
        {fhegpu::PlaceKind::Host, 0, 0},
        fhegpu::BootAttrs{target_level, scale_log2, 2, "test-boot",
                         fhegpu::BootImplementation::DecryptReencrypt},
    };
    const auto output = api.compute(
        op, {PoseidonCpuValue::from_ciphertext(std::move(input_cipher))});
    api.validate_value(
        output,
        {1, fhegpu::ValueKind::Ciphertext,
         {fhegpu::PlaceKind::Host, 0, 0}, context_id, target_level,
         scale_log2, true, 2});

    poseidon::Decryptor decryptor(context, *secret_key);
    poseidon::Plaintext output_plain;
    decryptor.decrypt(output.ciphertext(), output_plain);
    std::vector<std::complex<double>> decoded;
    encoder.decode(output_plain, decoded);
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        require(std::abs(decoded[i].real() - input[i]) < 1e-5,
                "unexpected decrypt_reencrypt Boot result");
        require(std::abs(decoded[i].imag()) < 1e-5,
                "unexpected decrypt_reencrypt Boot imaginary component");
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

#if defined(POSEIDON_RUNTIME_CPU_MPI)
fhegpu::RuntimePlan make_mpi_plan(const fhegpu::LoadedOperatorSpec &loaded_spec, int world_size,
                                  int level, int scale_log2)
{
    if (world_size != 2 && world_size != 4)
    {
        throw std::invalid_argument("Poseidon CPU MPI test requires 2 or 4 ranks");
    }

    fhegpu::RuntimePlan plan;
    plan.plan_id = 2;
    plan.target.target_id = "poseidon-ckks-cpu";
    plan.target.world_size = world_size;
    plan.target.device_counts.assign(static_cast<std::size_t>(world_size), 0);
    plan.target.capability_version = 1;
    plan.target.operator_spec = {loaded_spec.spec.id, loaded_spec.spec.version,
                                 loaded_spec.source_sha256};

    fhegpu::ValueId next_value_id = 0;
    std::size_t next_ordinal = 0;
    fhegpu::TransferId next_transfer_id = 1;
    const auto add_value = [&](fhegpu::ValueKind kind, int rank, int components) {
        const fhegpu::ValueId id = next_value_id++;
        plan.values.push_back(
            {id, kind, {fhegpu::PlaceKind::Host, rank, 0}, loaded_spec.spec.context_id, level,
             scale_log2, true, components});
        return id;
    };

    const fhegpu::ValueId source_cipher = add_value(fhegpu::ValueKind::Ciphertext, 0, 2);
    const fhegpu::ValueId source_plain = add_value(fhegpu::ValueKind::Plaintext, 0, 1);
    plan.external_inputs = {source_cipher};
    plan.initialization.push_back(
        {next_ordinal++,
         fhegpu::EncodeOp{fhegpu::InlineEncodePayload{{0.5, -1.0, 2.0, 3.0}},
                            source_plain}});

    std::vector<fhegpu::ValueId> ciphers(static_cast<std::size_t>(world_size));
    std::vector<fhegpu::ValueId> plains(static_cast<std::size_t>(world_size));
    ciphers[0] = source_cipher;
    plains[0] = source_plain;
    for (int rank = 1; rank < world_size; ++rank)
    {
        ciphers[static_cast<std::size_t>(rank)] =
            add_value(fhegpu::ValueKind::Ciphertext, rank, 2);
        plains[static_cast<std::size_t>(rank)] =
            add_value(fhegpu::ValueKind::Plaintext, rank, 1);
    }

    const auto add_distribution = [&](fhegpu::ValueId input,
                                      const std::vector<fhegpu::ValueId> &outputs,
                                      fhegpu::ValueKind kind) {
        fhegpu::CommAction action;
        action.id = next_transfer_id++;
        action.kind = world_size == 2 ? fhegpu::CommKind::Transfer
                                      : fhegpu::CommKind::Replicate;
        action.hint = world_size == 2 ? fhegpu::CommHint::PointToPoint
                                      : fhegpu::CommHint::Broadcast;
        action.inputs = {input};
        action.sources = {{fhegpu::PlaceKind::Host, 0, 0}};
        for (int rank = 1; rank < world_size; ++rank)
        {
            action.outputs.push_back(outputs[static_cast<std::size_t>(rank)]);
            action.destinations.push_back({fhegpu::PlaceKind::Host, rank, 0});
            action.output_types.push_back(kind);
        }
        plan.initialization.push_back({next_ordinal++, std::move(action)});
    };
    add_distribution(source_cipher, ciphers, fhegpu::ValueKind::Ciphertext);
    add_distribution(source_plain, plains, fhegpu::ValueKind::Plaintext);

    std::vector<fhegpu::ValueId> branches(static_cast<std::size_t>(world_size));
    for (int rank = 0; rank < world_size; ++rank)
    {
        branches[static_cast<std::size_t>(rank)] =
            add_value(fhegpu::ValueKind::Ciphertext, rank, 2);
        plan.execution.push_back(
            {next_ordinal++,
             fhegpu::ComputeOp{fhegpu::ComputeKind::AddCP,
                               {ciphers[static_cast<std::size_t>(rank)],
                                plains[static_cast<std::size_t>(rank)]},
                               branches[static_cast<std::size_t>(rank)],
                               {fhegpu::PlaceKind::Host, rank, 0}, {}}});
    }

    std::vector<fhegpu::ValueId> remote_branches;
    for (int rank = 1; rank < world_size; ++rank)
    {
        const fhegpu::ValueId received = add_value(fhegpu::ValueKind::Ciphertext, 0, 2);
        remote_branches.push_back(received);
        fhegpu::CommAction action;
        action.id = next_transfer_id++;
        action.kind = fhegpu::CommKind::Transfer;
        action.hint = fhegpu::CommHint::PointToPoint;
        action.inputs = {branches[static_cast<std::size_t>(rank)]};
        action.outputs = {received};
        action.sources = {{fhegpu::PlaceKind::Host, rank, 0}};
        action.destinations = {{fhegpu::PlaceKind::Host, 0, 0}};
        action.output_types = {fhegpu::ValueKind::Ciphertext};
        plan.execution.push_back({next_ordinal++, std::move(action)});
    }

    fhegpu::ValueId accumulated = branches[0];
    for (fhegpu::ValueId remote : remote_branches)
    {
        const fhegpu::ValueId output = add_value(fhegpu::ValueKind::Ciphertext, 0, 2);
        plan.execution.push_back(
            {next_ordinal++,
             fhegpu::ComputeOp{fhegpu::ComputeKind::AddCC, {accumulated, remote}, output,
                               {fhegpu::PlaceKind::Host, 0, 0}, {}}});
        accumulated = output;
    }
    plan.final_outputs = {accumulated};
    return plan;
}

void test_mpi_rejects_plan_digest_mismatch(int rank, int world_size)
{
    const std::string context_id = "poseidon-cpu-mpi-test-context";
    poseidon::ParametersLiteralDefault parameters(CKKS, 4096,
                                                  poseidon::sec_level_type::tc128);
    poseidon::PoseidonContext context(parameters);
    const auto operator_spec = make_operator_spec(context, context_id);

    fhegpu::TargetConfig target;
    target.target_id = "poseidon-ckks-cpu";
    target.world_size = world_size;
    target.device_counts.assign(static_cast<std::size_t>(world_size), 0);
    target.capability_version = 1;

    const std::string digest =
        rank == world_size - 1
            ? "sha256:3333333333333333333333333333333333333333333333333333333333333333"
            : "sha256:2222222222222222222222222222222222222222222222222222222222222222";
    PoseidonCpuApi api(context_id, context, MPI_COMM_WORLD);
    bool rejected = false;
    try
    {
        api.preflight(digest, false, target, operator_spec.spec, {});
    }
    catch (const std::exception &error)
    {
        rejected = std::string(error.what()).find("RuntimePlan source SHA-256 mismatch") !=
                   std::string::npos;
    }
    int local_rejected = rejected ? 1 : 0;
    int all_rejected = 0;
    MPI_Allreduce(&local_rejected, &all_rejected, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    require(all_rejected == 1, "MPI plan digest mismatch was not rejected by every rank");
}

void test_mpi_runtime(int rank, int world_size)
{
    constexpr std::uint32_t degree = 4096;
    const std::string context_id = "poseidon-cpu-mpi-test-context";
    poseidon::ParametersLiteralDefault parameters(CKKS, degree, poseidon::sec_level_type::tc128);
    poseidon::PoseidonContext context(parameters);
    const int level = static_cast<int>(parameters.q().size() - 1);
    const int scale_log2 = static_cast<int>(parameters.log_scale());
    const double scale = std::ldexp(1.0, scale_log2);

    poseidon::KeyGenerator key_generator(context);
    poseidon::PublicKey public_key;
    key_generator.create_public_key(public_key);
    poseidon::Encryptor encryptor(context, public_key);
    poseidon::CKKSEncoder encoder(context);

    const std::vector<double> input{1.0, 2.0, 3.0, 4.0};
    poseidon::Plaintext input_plain;
    encoder.encode(input, scale, input_plain);
    poseidon::Ciphertext input_cipher;
    encryptor.encrypt(input_plain, input_cipher);

    const auto operator_spec = make_operator_spec(context, context_id);
    const auto plan = make_mpi_plan(operator_spec, world_size, level, scale_log2);
    const fhegpu::LoadedRuntimePlan loaded_plan{
        plan, "sha256:2222222222222222222222222222222222222222222222222222222222222222"};
    const fhegpu::RuntimeResources resources{operator_spec, std::nullopt, false};

    PoseidonCpuApi api(context_id, context, MPI_COMM_WORLD);
    require(api.rank() == rank && api.world_size() == world_size,
            "Poseidon CPU Api MPI identity mismatch");
    fhegpu::SequentialRuntime<PoseidonCpuApi> runtime(api.rank(), api.world_size(), 0, api);
    std::unordered_map<fhegpu::ValueId, PoseidonCpuValue> inputs;
    if (rank == 0)
    {
        inputs.emplace(0, PoseidonCpuValue::from_ciphertext(std::move(input_cipher)));
    }
    const auto artifact = runtime.run(loaded_plan, resources, inputs);

    if (rank != 0)
    {
        require(artifact.values.empty(), "non-root MPI rank published a final output");
        return;
    }

    poseidon::Decryptor decryptor(context, key_generator.secret_key());
    poseidon::Plaintext result_plain;
    decryptor.decrypt(artifact.values.at(plan.final_outputs.front()).value.ciphertext(),
                      result_plain);
    std::vector<std::complex<double>> result;
    encoder.decode(result_plain, result);
    const std::vector<double> constant{0.5, -1.0, 2.0, 3.0};
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        const double expected = static_cast<double>(world_size) * (input[i] + constant[i]);
        require(std::abs(result[i].real() - expected) < 1e-4,
                "unexpected MPI result at slot " + std::to_string(i));
        require(std::abs(result[i].imag()) < 1e-4,
                "unexpected MPI imaginary component at slot " + std::to_string(i));
    }
}
#endif

} // namespace

int main(int argc, char **argv)
{
#if defined(POSEIDON_RUNTIME_CPU_MPI)
    if (argc == 2 && std::string(argv[1]) == "--mpi")
    {
        int provided = MPI_THREAD_SINGLE;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        int rank = 0;
        int world_size = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        int local_ok = provided >= MPI_THREAD_FUNNELED && (world_size == 2 || world_size == 4);
        if (local_ok)
        {
            try
            {
                test_mpi_rejects_plan_digest_mismatch(rank, world_size);
                test_mpi_runtime(rank, world_size);
            }
            catch (const std::exception &error)
            {
                std::fprintf(stderr, "[rank %d] [FAIL] %s\n", rank, error.what());
                local_ok = 0;
            }
        }
        int global_ok = 0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0)
        {
            std::printf("Poseidon CPU Runtime Api MPI (%d ranks): %s\n", world_size,
                        global_ok ? "PASS" : "FAIL");
        }
        MPI_Finalize();
        return global_ok ? 0 : 1;
    }
#endif

    try
    {
        run_test("single-process Host AddCP", test_single_process_host_add_plain);
        run_test("decrypt_reencrypt Boot", test_decrypt_reencrypt_boot);
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
