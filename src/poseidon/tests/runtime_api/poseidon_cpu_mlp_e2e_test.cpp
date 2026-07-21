#include "poseidon/basics/modulus.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/poseidon_cpu_api.h"
#include "poseidon/runtime_api/rotation_key_basis.h"
#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/runtime.hpp"
#include "runtime/verifier.hpp"

#include <nlohmann/json.hpp>

#if defined(POSEIDON_RUNTIME_CPU_MPI)
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

using Json = nlohmann::json;
using poseidon::runtime_api::PoseidonCpuApi;
using poseidon::runtime_api::PoseidonCpuValue;

constexpr double kAbsoluteTolerance = 1e-4;
constexpr double kRelativeTolerance = 1e-6;
constexpr double kImaginaryTolerance = 1e-5;

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
        if (!value.is_number())
        {
            throw std::runtime_error(std::string(member) + " contains a non-number");
        }
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

std::vector<double> pack_resnet20_input(const std::vector<double> &logical)
{
    constexpr std::size_t image_values = 3 * 32 * 32;
    constexpr std::size_t packed_block = 4096;
    constexpr std::size_t repetitions = 4;
    constexpr double activation_scale = 32.0;
    if (logical.size() != image_values)
    {
        throw std::runtime_error(
            "ResNet-20 logical input must contain 3072 values");
    }
    std::vector<double> block(packed_block, 0.0);
    for (std::size_t index = 0; index < logical.size(); ++index)
    {
        block[index] = logical[index] / activation_scale;
    }
    std::vector<double> packed;
    packed.reserve(packed_block * repetitions);
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
    {
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
    modulus_bits.insert(modulus_bits.end(), spec.rns_moduli_log2.begin(),
                        spec.rns_moduli_log2.end());
    const auto moduli = poseidon::CoeffModulus::Create(spec.poly_degree, modulus_bits);
    const std::size_t q_count = spec.rns_moduli_log2.size();
    std::vector<poseidon::Modulus> q(moduli.begin(), moduli.begin() + q_count);
    std::vector<poseidon::Modulus> p(moduli.begin() + q_count, moduli.end());
    const std::uint32_t log_n = exact_log2(spec.poly_degree);
    poseidon::ParametersLiteral parameters(
        CKKS, log_n, log_n - 1,
        static_cast<std::uint32_t>(spec.default_scale_log2), 5, 0,
        poseidon::Modulus(0), q, p, poseidon::sec_level_type::none);
    return poseidon::PoseidonContext(parameters);
}

const fhegpu::ValueDesc &find_value(const fhegpu::RuntimePlan &plan,
                                    fhegpu::ValueId id)
{
    for (const auto &value : plan.values)
    {
        if (value.id == id)
        {
            return value;
        }
    }
    throw std::runtime_error("missing ValueDesc " + std::to_string(id));
}

Comparison compare(const std::vector<double> &expected,
                   const std::vector<double> &actual,
                   double absolute_tolerance = kAbsoluteTolerance,
                   double relative_tolerance = kRelativeTolerance)
{
    if (expected.size() != actual.size())
    {
        throw std::runtime_error("cannot compare outputs with different lengths");
    }
    Comparison result;
    double squared_error = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const double difference = std::abs(expected[i] - actual[i]);
        result.max_abs = std::max(result.max_abs, difference);
        squared_error += difference * difference;
        const double relative =
            difference / std::max(std::abs(expected[i]), absolute_tolerance);
        result.max_relative = std::max(result.max_relative, relative);
        if (difference >
            absolute_tolerance + relative_tolerance * std::abs(expected[i]))
        {
            result.within_tolerance = false;
        }
    }
    result.rms = std::sqrt(squared_error / expected.size());
    return result;
}

Json comparison_json(const Comparison &comparison)
{
    return {
        {"max_abs", comparison.max_abs},
        {"rms", comparison.rms},
        {"max_relative", comparison.max_relative},
        {"within_tolerance", comparison.within_tolerance},
    };
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

#if defined(POSEIDON_RUNTIME_CPU_MPI)
void check_mpi(int code, const char *operation)
{
    if (code == MPI_SUCCESS)
    {
        return;
    }
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(code, message, &length);
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::string(message, static_cast<std::size_t>(length)));
}

void broadcast_secret_key(const poseidon::PoseidonContext &context,
                          poseidon::SecretKey &secret_key, int rank)
{
    const auto mode = poseidon::compr_mode_type::none;
    std::uint64_t byte_count = 0;
    std::vector<poseidon::poseidon_byte> bytes;
    if (rank == 0)
    {
        const std::streamoff size = secret_key.save_size(mode);
        if (size <= 0 || static_cast<std::uint64_t>(size) >
                             static_cast<std::uint64_t>(
                                 std::numeric_limits<int>::max()))
        {
            throw std::runtime_error("serialized secret key exceeds MPI count range");
        }
        byte_count = static_cast<std::uint64_t>(size);
        bytes.resize(static_cast<std::size_t>(byte_count));
        if (secret_key.save(bytes.data(), bytes.size(), mode) != size)
        {
            throw std::runtime_error("secret key serialization size mismatch");
        }
    }
    check_mpi(MPI_Bcast(&byte_count, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD),
              "MPI_Bcast(secret key size)");
    if (byte_count == 0 || byte_count >
                               static_cast<std::uint64_t>(
                                   std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("invalid broadcast secret key size");
    }
    if (rank != 0)
    {
        bytes.resize(static_cast<std::size_t>(byte_count));
    }
    check_mpi(MPI_Bcast(bytes.data(), static_cast<int>(bytes.size()), MPI_BYTE, 0,
                        MPI_COMM_WORLD),
              "MPI_Bcast(secret key)");
    if (rank != 0)
    {
        if (secret_key.load(context, bytes.data(), bytes.size()) !=
            static_cast<std::streamoff>(bytes.size()))
        {
            throw std::runtime_error("secret key load size mismatch");
        }
        secret_key.data().resize(context, secret_key.parms_id(),
                                 secret_key.data().coeff_count());
    }
}
#endif

std::size_t count_transfers(const fhegpu::RuntimePlan &plan)
{
    std::size_t result = 0;
    const auto count_phase = [&](const std::vector<fhegpu::Instruction> &phase) {
        result += static_cast<std::size_t>(std::count_if(
            phase.begin(), phase.end(), [](const fhegpu::Instruction &instruction) {
                return std::holds_alternative<fhegpu::CommAction>(instruction.body);
            }));
    };
    count_phase(plan.initialization);
    count_phase(plan.execution);
    count_phase(plan.finalization);
    return result;
}

int run_e2e(char **paths, bool mpi_mode, int rank, int world_size)
{
    const auto loaded_plan = fhegpu::RuntimePlanJsonReader::read_file(paths[0]);
    const auto loaded_spec = fhegpu::OperatorSpecReader::read_file(paths[1]);
    const Json fixture = read_json(paths[3]);
    const Json mock_result = read_json(paths[4]);
    if (fixture.at("format_version") != 1 ||
        mock_result.at("format_version") != 1 ||
        mock_result.at("passed") != true ||
        fixture.at("seed") != mock_result.at("seed") ||
        fixture.at("model_sha256") != mock_result.at("model_sha256"))
    {
        throw std::runtime_error("fixture and MockVecApi result do not match");
    }
    if (loaded_plan.plan.target.world_size != world_size)
    {
        throw std::runtime_error("RuntimePlan world size does not match execution mode");
    }
    const std::string model = fixture.value("model", "mlp");
    if (model != "mlp" && model != "resnet20")
    {
        throw std::runtime_error("unsupported model fixture");
    }
    const std::size_t logical_input_size =
        model == "resnet20" ? 3 * 32 * 32 : 784;
    const std::vector<double> input =
        read_numbers(fixture, "input", logical_input_size);
    const std::vector<double> python_output =
        read_numbers(fixture, "python_output", 10);
    const std::vector<double> mock_output =
        read_numbers(mock_result, "output", 10);

    const auto requirements = fhegpu::PlanVerifier::verify(
        loaded_plan.plan, loaded_spec, false);
    poseidon::PoseidonContext context = make_context(loaded_spec.spec);
    auto secret_key = std::make_shared<poseidon::SecretKey>();
    if (rank == 0)
    {
        poseidon::KeyGenerator owner(context);
        *secret_key = owner.secret_key();
    }
#if defined(POSEIDON_RUNTIME_CPU_MPI)
    if (mpi_mode)
    {
        broadcast_secret_key(context, *secret_key, rank);
    }
#else
    static_cast<void>(mpi_mode);
#endif
    poseidon::KeyGenerator key_generator(context, *secret_key);
    auto public_key = std::make_shared<poseidon::PublicKey>();
    auto relin_keys = std::make_shared<poseidon::RelinKeys>();
    auto galois_keys = std::make_shared<poseidon::GaloisKeys>();
    bool needs_relin = false;
    std::set<int> logical_rotation_steps;
    for (const auto &key : requirements.keys)
    {
        if (key.place.rank != rank)
        {
            continue;
        }
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
            logical_rotation_steps.insert(*key.rotation_step);
        }
    }
    const std::set<int> rotation_key_steps =
        poseidon::runtime_api::binary_rotation_key_basis(
            logical_rotation_steps, context.parameters_literal()->slot());
    const auto key_start = std::chrono::steady_clock::now();
    key_generator.create_public_key(*public_key);
    if (needs_relin)
    {
        key_generator.create_relin_keys(*relin_keys);
    }
    if (!rotation_key_steps.empty())
    {
        key_generator.create_galois_keys(
            std::vector<int>(rotation_key_steps.begin(), rotation_key_steps.end()),
            *galois_keys);
    }
    const auto key_finish = std::chrono::steady_clock::now();

    if (loaded_plan.plan.external_inputs.size() != 1)
    {
        throw std::runtime_error("model plan must have one external input");
    }
    const fhegpu::ValueId input_id = loaded_plan.plan.external_inputs.front();
    const auto &input_desc = find_value(loaded_plan.plan, input_id);
    std::unordered_map<fhegpu::ValueId, PoseidonCpuValue> inputs;
    poseidon::CKKSEncoder encoder(context);
    if (input_desc.place.rank == rank)
    {
        poseidon::Plaintext input_plain;
        encoder.encode(
            model == "resnet20" ? pack_resnet20_input(input)
                                 : pack_mlp_input(input),
            context.crt_context()->parms_id_map().at(
                static_cast<std::uint32_t>(input_desc.level)),
            std::ldexp(1.0, input_desc.scale_log2), input_plain);
        poseidon::Encryptor encryptor(context, *public_key);
        poseidon::Ciphertext input_cipher;
        encryptor.encrypt(input_plain, input_cipher);
        inputs.emplace(input_id,
                       PoseidonCpuValue::from_ciphertext(std::move(input_cipher)));
    }

    std::unique_ptr<PoseidonCpuApi> api;
#if defined(POSEIDON_RUNTIME_CPU_MPI)
    if (mpi_mode)
    {
        api = std::make_unique<PoseidonCpuApi>(
            loaded_spec.spec.context_id, context, MPI_COMM_WORLD, relin_keys,
            galois_keys, public_key, secret_key);
    }
    else
#endif
    {
        api = std::make_unique<PoseidonCpuApi>(
            loaded_spec.spec.context_id, context, relin_keys, galois_keys,
            public_key, secret_key);
    }
    fhegpu::SequentialRuntime<PoseidonCpuApi> runtime(rank, world_size, 0, *api);
    const fhegpu::RuntimeResources resources{
        loaded_spec, std::filesystem::path(paths[2]), false};
    const auto run_start = std::chrono::steady_clock::now();
    const auto artifact = runtime.run(loaded_plan, resources, inputs);
    const auto run_finish = std::chrono::steady_clock::now();

    if (loaded_plan.plan.final_outputs.size() != 1)
    {
        throw std::runtime_error("model plan must have one final output");
    }
    const auto final_id = loaded_plan.plan.final_outputs.front();
    const int final_rank = find_value(loaded_plan.plan, final_id).place.rank;
    std::vector<double> poseidon_output(10, 0.0);
    double max_imaginary = 0.0;
    if (rank == final_rank)
    {
        poseidon::Decryptor decryptor(context, *secret_key);
        poseidon::Plaintext output_plain;
        decryptor.decrypt(artifact.values.at(final_id).value.ciphertext(),
                          output_plain);
        std::vector<std::complex<double>> decoded;
        encoder.decode(output_plain, decoded);
        if (decoded.size() < 10)
        {
            throw std::runtime_error("decoded model output has fewer than 10 slots");
        }
        for (std::size_t i = 0; i < 10; ++i)
        {
            poseidon_output[i] = decoded[i].real();
            max_imaginary = std::max(max_imaginary, std::abs(decoded[i].imag()));
        }
    }
#if defined(POSEIDON_RUNTIME_CPU_MPI)
    if (mpi_mode)
    {
        check_mpi(MPI_Bcast(poseidon_output.data(),
                            static_cast<int>(poseidon_output.size()), MPI_DOUBLE,
                            final_rank, MPI_COMM_WORLD),
                  "MPI_Bcast(MLP output)");
        check_mpi(MPI_Bcast(&max_imaginary, 1, MPI_DOUBLE, final_rank,
                            MPI_COMM_WORLD),
                  "MPI_Bcast(MLP imaginary error)");
    }
#endif

    const double python_absolute_tolerance =
        model == "resnet20" ? 0.1 : kAbsoluteTolerance;
    const Comparison against_python = compare(
        python_output, poseidon_output, python_absolute_tolerance,
        kRelativeTolerance);
    const Comparison against_mock = compare(mock_output, poseidon_output);
    const bool require_python_match = model != "resnet20";
    bool passed = (!require_python_match || against_python.within_tolerance) &&
                  against_mock.within_tolerance &&
                  max_imaginary <= kImaginaryTolerance;
#if defined(POSEIDON_RUNTIME_CPU_MPI)
    if (mpi_mode)
    {
        int local_passed = passed ? 1 : 0;
        int all_passed = 0;
        check_mpi(MPI_Allreduce(&local_passed, &all_passed, 1, MPI_INT, MPI_MIN,
                                MPI_COMM_WORLD),
                  "MPI_Allreduce(MLP result)");
        passed = all_passed != 0;
    }
#endif

    const double key_seconds =
        std::chrono::duration<double>(key_finish - key_start).count();
    const double run_seconds =
        std::chrono::duration<double>(run_finish - run_start).count();
    const double compute_including_boot_seconds =
        static_cast<double>(artifact.timing.compute_including_boot_nanoseconds) *
        1e-9;
    const double boot_seconds =
        static_cast<double>(artifact.timing.boot_nanoseconds) * 1e-9;
    const double compute_excluding_boot_seconds =
        static_cast<double>(
            artifact.timing.compute_excluding_boot_nanoseconds()) *
        1e-9;
    const double setup_seconds =
        static_cast<double>(artifact.timing.setup_nanoseconds) * 1e-9;
    const double initialization_seconds =
        static_cast<double>(artifact.timing.initialization_nanoseconds) * 1e-9;
    const double online_execution_seconds =
        static_cast<double>(artifact.timing.online_execution_nanoseconds) * 1e-9;
    const std::array<double, 8> local_seconds{
        key_seconds, run_seconds, compute_including_boot_seconds, boot_seconds,
        compute_excluding_boot_seconds, setup_seconds, initialization_seconds,
        online_execution_seconds};
    const std::array<std::uint64_t, 4> local_counts{
        static_cast<std::uint64_t>(artifact.timing.compute_calls),
        static_cast<std::uint64_t>(artifact.timing.boot_calls),
        static_cast<std::uint64_t>(rotation_key_steps.size()),
        static_cast<std::uint64_t>(needs_relin ? 1 : 0)};
    std::vector<double> gathered_seconds;
    std::vector<std::uint64_t> gathered_counts;
    if (rank == 0)
    {
        gathered_seconds.resize(static_cast<std::size_t>(world_size) *
                                local_seconds.size());
        gathered_counts.resize(static_cast<std::size_t>(world_size) *
                               local_counts.size());
    }
#if defined(POSEIDON_RUNTIME_CPU_MPI)
    if (mpi_mode)
    {
        check_mpi(MPI_Gather(local_seconds.data(),
                             static_cast<int>(local_seconds.size()), MPI_DOUBLE,
                             rank == 0 ? gathered_seconds.data() : nullptr,
                             static_cast<int>(local_seconds.size()), MPI_DOUBLE, 0,
                             MPI_COMM_WORLD),
                  "MPI_Gather(MLP seconds)");
        check_mpi(MPI_Gather(local_counts.data(),
                             static_cast<int>(local_counts.size()), MPI_UINT64_T,
                             rank == 0 ? gathered_counts.data() : nullptr,
                             static_cast<int>(local_counts.size()), MPI_UINT64_T, 0,
                             MPI_COMM_WORLD),
                  "MPI_Gather(MLP counts)");
    }
    else
#endif
    {
        gathered_seconds.assign(local_seconds.begin(), local_seconds.end());
        gathered_counts.assign(local_counts.begin(), local_counts.end());
    }

    if (rank == 0)
    {
        Json rank_timings = Json::array();
        double critical_key_seconds = 0.0;
        double critical_run_seconds = 0.0;
        double critical_compute_seconds = 0.0;
        double critical_boot_seconds = 0.0;
        double critical_non_boot_seconds = 0.0;
        double critical_setup_seconds = 0.0;
        double critical_initialization_seconds = 0.0;
        double critical_online_execution_seconds = 0.0;
        std::uint64_t total_compute_calls = 0;
        std::uint64_t total_boot_calls = 0;
        std::uint64_t total_rotation_keys = 0;
        for (int measured_rank = 0; measured_rank < world_size; ++measured_rank)
        {
            const std::size_t seconds_offset =
                static_cast<std::size_t>(measured_rank) * local_seconds.size();
            const std::size_t counts_offset =
                static_cast<std::size_t>(measured_rank) * local_counts.size();
            critical_key_seconds =
                std::max(critical_key_seconds, gathered_seconds[seconds_offset]);
            critical_run_seconds = std::max(
                critical_run_seconds, gathered_seconds[seconds_offset + 1]);
            critical_compute_seconds = std::max(
                critical_compute_seconds, gathered_seconds[seconds_offset + 2]);
            critical_boot_seconds = std::max(
                critical_boot_seconds, gathered_seconds[seconds_offset + 3]);
            critical_non_boot_seconds = std::max(
                critical_non_boot_seconds, gathered_seconds[seconds_offset + 4]);
            critical_setup_seconds = std::max(
                critical_setup_seconds, gathered_seconds[seconds_offset + 5]);
            critical_initialization_seconds = std::max(
                critical_initialization_seconds,
                gathered_seconds[seconds_offset + 6]);
            critical_online_execution_seconds = std::max(
                critical_online_execution_seconds,
                gathered_seconds[seconds_offset + 7]);
            total_compute_calls += gathered_counts[counts_offset];
            total_boot_calls += gathered_counts[counts_offset + 1];
            total_rotation_keys += gathered_counts[counts_offset + 2];
            rank_timings.push_back(
                {{"rank", measured_rank},
                 {"key_generation_seconds", gathered_seconds[seconds_offset]},
                 {"runtime_seconds", gathered_seconds[seconds_offset + 1]},
                 {"compute_including_boot_seconds",
                  gathered_seconds[seconds_offset + 2]},
                 {"boot_seconds", gathered_seconds[seconds_offset + 3]},
                 {"compute_excluding_boot_seconds",
                  gathered_seconds[seconds_offset + 4]},
                 {"setup_seconds", gathered_seconds[seconds_offset + 5]},
                 {"initialization_seconds",
                  gathered_seconds[seconds_offset + 6]},
                 {"online_execution_seconds",
                  gathered_seconds[seconds_offset + 7]},
                 {"compute_calls", gathered_counts[counts_offset]},
                 {"boot_calls", gathered_counts[counts_offset + 1]},
                 {"rotation_key_count", gathered_counts[counts_offset + 2]},
                 {"has_relin_key", gathered_counts[counts_offset + 3] != 0}});
        }
        Json report{
            {"format_version", 1},
            {"passed", passed},
            {"model", model},
            {"require_python_match", require_python_match},
            {"world_size", world_size},
            {"final_output_rank", final_rank},
            {"transfer_count", count_transfers(loaded_plan.plan)},
            {"seed", fixture.at("seed")},
            {"model_sha256", fixture.at("model_sha256")},
            {"plan_sha256", loaded_plan.source_sha256},
            {"rotation_key_count", total_rotation_keys},
            {"key_generation_seconds", critical_key_seconds},
            {"runtime_seconds", critical_run_seconds},
            {"runtime_timing",
             {{"seconds_aggregation", "maximum_rank"},
              {"setup_seconds", critical_setup_seconds},
              {"initialization_seconds", critical_initialization_seconds},
              {"online_execution_seconds", critical_online_execution_seconds},
              {"compute_calls", total_compute_calls},
              {"boot_calls", total_boot_calls},
              {"compute_including_boot_seconds", critical_compute_seconds},
              {"boot_seconds", critical_boot_seconds},
              {"compute_excluding_boot_seconds", critical_non_boot_seconds}}},
            {"rank_timings", std::move(rank_timings)},
            {"tolerances",
             {{"absolute", kAbsoluteTolerance},
              {"relative", kRelativeTolerance},
              {"imaginary", kImaginaryTolerance},
              {"python_absolute", python_absolute_tolerance}}},
            {"python_comparison", comparison_json(against_python)},
            {"mock_comparison", comparison_json(against_mock)},
            {"max_imaginary", max_imaginary},
            {"python_output", python_output},
            {"mock_output", mock_output},
            {"poseidon_output", poseidon_output},
        };
        write_json(paths[5], report);

        std::cout << (passed ? "PASS" : "FAIL")
                  << " world_size=" << world_size
                  << " transfers=" << count_transfers(loaded_plan.plan)
                  << " final_rank=" << final_rank
                  << " rotations=" << total_rotation_keys
                  << " key_seconds=" << critical_key_seconds
                  << " runtime_seconds=" << critical_run_seconds
                  << " initialization_seconds="
                  << critical_initialization_seconds
                  << " online_execution_seconds="
                  << critical_online_execution_seconds
                  << " compute_including_boot_seconds="
                  << critical_compute_seconds
                  << " boot_seconds=" << critical_boot_seconds
                  << " compute_excluding_boot_seconds="
                  << critical_non_boot_seconds
                  << " python_max_abs=" << against_python.max_abs
                  << " mock_max_abs=" << against_mock.max_abs
                  << " max_imaginary=" << max_imaginary << '\n'
                  << "report_json=" << paths[5] << '\n';
    }
    return passed ? 0 : 1;
}

} // namespace

int main(int argc, char **argv)
{
    const bool mpi_mode = argc > 1 && std::string(argv[1]) == "--mpi";
    int rank = 0;
    int world_size = 1;
#if defined(POSEIDON_RUNTIME_CPU_MPI)
    if (mpi_mode)
    {
        int provided = MPI_THREAD_SINGLE;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (provided < MPI_THREAD_FUNNELED)
        {
            std::fprintf(stderr, "[rank %d] MPI_THREAD_FUNNELED is unavailable\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
#else
    if (mpi_mode)
    {
        std::cerr << "FAIL Poseidon CPU MPI support is not enabled\n";
        return 2;
    }
#endif

    const int expected_argc = mpi_mode ? 8 : 7;
    if (argc != expected_argc)
    {
        if (rank == 0)
        {
            std::cerr << "usage: poseidon_runtime_cpu_mlp_e2e [--mpi] PLAN "
                         "OPERATOR_SPEC BUNDLE_DIR FIXTURE MOCK_RESULT REPORT_JSON\n";
        }
#if defined(POSEIDON_RUNTIME_CPU_MPI)
        if (mpi_mode)
        {
            MPI_Finalize();
        }
#endif
        return 2;
    }

    int result = 1;
    try
    {
        result = run_e2e(argv + (mpi_mode ? 2 : 1), mpi_mode, rank, world_size);
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "[rank %d] FAIL %s\n", rank, error.what());
#if defined(POSEIDON_RUNTIME_CPU_MPI)
        if (mpi_mode)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
#endif
    }
#if defined(POSEIDON_RUNTIME_CPU_MPI)
    if (mpi_mode)
    {
        MPI_Finalize();
    }
#endif
    return result;
}
