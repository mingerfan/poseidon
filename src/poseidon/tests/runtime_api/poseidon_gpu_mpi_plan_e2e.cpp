#include "poseidon/basics/modulus.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "poseidon/runtime_api/rotation_key_basis.h"
#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/runtime.hpp"

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace
{

using Json = nlohmann::json;
using poseidon::runtime_api::PoseidonGpuApi;
using poseidon::runtime_api::PoseidonGpuValue;

constexpr std::size_t kInputSize = 8;
constexpr double kExpectedFactor = -10.0;
constexpr double kTolerance = 1e-4;

void check_mpi(int status, const char *what)
{
    if (status == MPI_SUCCESS)
    {
        return;
    }
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    (void)MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(what) + ": " +
                             std::string(message, static_cast<std::size_t>(length)));
}

std::vector<int> parse_rank_to_node(const std::string &encoded)
{
    std::vector<int> result;
    std::size_t begin = 0;
    while (begin <= encoded.size())
    {
        const std::size_t end = encoded.find('x', begin);
        const std::string token = encoded.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (token.empty())
        {
            throw std::invalid_argument("rank-to-node contains an empty entry");
        }
        std::size_t consumed = 0;
        const int value = std::stoi(token, &consumed);
        if (consumed != token.size() || value < 0)
        {
            throw std::invalid_argument(
                "rank-to-node entries must be nonnegative integers");
        }
        result.push_back(value);
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return result;
}

std::string encode_x_list(const std::vector<int> &values)
{
    std::ostringstream encoded;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0)
        {
            encoded << 'x';
        }
        encoded << values[index];
    }
    return encoded.str();
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

std::vector<int> local_cuda_devices(MPI_Comm world_comm, int local_device_count)
{
    MPI_Comm local_comm = MPI_COMM_NULL;
    check_mpi(MPI_Comm_split_type(world_comm, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &local_comm),
              "MPI_Comm_split_type");
    int local_rank = 0;
    check_mpi(MPI_Comm_rank(local_comm, &local_rank), "MPI_Comm_rank local");
    int device_offset = 0;
    check_mpi(MPI_Exscan(&local_device_count, &device_offset, 1, MPI_INT,
                         MPI_SUM, local_comm),
              "MPI_Exscan local device counts");
    if (local_rank == 0)
    {
        device_offset = 0;
    }
    check_mpi(MPI_Comm_free(&local_comm), "MPI_Comm_free local");

    std::vector<int> result(static_cast<std::size_t>(local_device_count));
    for (int index = 0; index < local_device_count; ++index)
    {
        result[static_cast<std::size_t>(index)] = device_offset + index;
    }
    return result;
}

void require_same_topology(const std::vector<int> &rank_to_node, int world_size)
{
    std::vector<int> gathered(static_cast<std::size_t>(world_size) *
                              rank_to_node.size());
    check_mpi(MPI_Allgather(rank_to_node.data(), world_size, MPI_INT,
                            gathered.data(), world_size, MPI_INT,
                            MPI_COMM_WORLD),
              "MPI_Allgather rank-to-node");
    for (int rank = 0; rank < world_size; ++rank)
    {
        const auto begin = gathered.begin() +
                           static_cast<std::ptrdiff_t>(rank) * world_size;
        if (!std::equal(rank_to_node.begin(), rank_to_node.end(), begin))
        {
            throw std::runtime_error("rank-to-node differs across MPI ranks");
        }
    }
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
              "MPI_Bcast secret key size");
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
    check_mpi(MPI_Bcast(bytes.data(), static_cast<int>(bytes.size()), MPI_BYTE,
                        0, MPI_COMM_WORLD),
              "MPI_Bcast secret key");
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

PoseidonGpuValue download(PoseidonGpuApi &api, fhegpu::ValueId id,
                          const fhegpu::Place &source,
                          const PoseidonGpuValue &value)
{
    fhegpu::CommAction action;
    action.id = std::numeric_limits<fhegpu::TransferId>::max();
    action.kind = fhegpu::CommKind::Transfer;
    action.hint = fhegpu::CommHint::PointToPoint;
    action.inputs = {id};
    action.outputs = {id + 1};
    action.sources = {source};
    action.destinations = {{fhegpu::PlaceKind::Host, source.rank, 0}};
    action.output_types = {fhegpu::ValueKind::Ciphertext};
    auto handle = api.communicate_async(action, {value});
    auto outputs = api.wait(handle);
    if (outputs.size() != 1)
    {
        throw std::runtime_error("GPU result download produced the wrong output count");
    }
    return std::move(outputs.front());
}

struct PlanStats
{
    std::size_t negates = 0;
    std::size_t adds = 0;
    std::size_t cross_rank_transfers = 0;
    std::size_t cross_rank_device_transfers = 0;
    std::size_t cross_rank_host_uploads = 0;
    std::set<std::pair<int, int>> compute_places;
};

PlanStats validate_smoke_plan(const fhegpu::RuntimePlan &plan)
{
    PlanStats stats;
    const auto inspect_phase = [&](const std::vector<fhegpu::Instruction> &phase) {
        for (const auto &instruction : phase)
        {
            if (const auto *compute =
                    std::get_if<fhegpu::ComputeOp>(&instruction.body))
            {
                stats.compute_places.emplace(compute->place.rank,
                                             compute->place.index);
                if (compute->kind == fhegpu::ComputeKind::Negate)
                {
                    ++stats.negates;
                }
                else if (compute->kind == fhegpu::ComputeKind::AddCC)
                {
                    ++stats.adds;
                }
                else
                {
                    throw std::runtime_error(
                        "MPI GPU smoke plan contains an unexpected compute op");
                }
                continue;
            }
            const auto *communication =
                std::get_if<fhegpu::CommAction>(&instruction.body);
            if (communication == nullptr || communication->sources.size() != 1)
            {
                continue;
            }
            for (const auto &destination : communication->destinations)
            {
                const auto &source = communication->sources.front();
                if (source.rank == destination.rank)
                {
                    continue;
                }
                ++stats.cross_rank_transfers;
                if (source.kind == fhegpu::PlaceKind::Host &&
                    destination.kind == fhegpu::PlaceKind::Device)
                {
                    ++stats.cross_rank_host_uploads;
                    continue;
                }
                if (source.kind != fhegpu::PlaceKind::Device ||
                    destination.kind != fhegpu::PlaceKind::Device)
                {
                    throw std::runtime_error(
                        "MPI GPU smoke plan requires an unsupported remote transfer");
                }
                ++stats.cross_rank_device_transfers;
            }
        }
    };
    inspect_phase(plan.initialization);
    inspect_phase(plan.execution);
    inspect_phase(plan.finalization);

    std::set<std::pair<int, int>> expected_places;
    for (std::size_t rank = 0; rank < plan.target.device_counts.size(); ++rank)
    {
        for (int device = 0; device < plan.target.device_counts[rank]; ++device)
        {
            expected_places.emplace(static_cast<int>(rank), device);
        }
    }
    if (stats.negates != 10 || stats.adds != 9 ||
        stats.cross_rank_device_transfers == 0 ||
        stats.compute_places != expected_places)
    {
        throw std::runtime_error(
            "MPI GPU smoke plan does not exercise the expected distributed graph");
    }
    return stats;
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
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) !=
        MPI_SUCCESS)
    {
        return 2;
    }

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int exit_code = 0;

    try
    {
        if (argc != 6 || std::string(argv[4]) != "--rank-to-node")
        {
            throw std::invalid_argument(
                "usage: poseidon_gpu_mpi_plan_e2e PLAN OPERATOR_SPEC REPORT "
                "--rank-to-node 0x1");
        }
        const std::vector<int> rank_to_node = parse_rank_to_node(argv[5]);
        if (static_cast<int>(rank_to_node.size()) != world_size)
        {
            throw std::invalid_argument(
                "rank-to-node length must match MPI world size");
        }
        require_same_topology(rank_to_node, world_size);

        const auto loaded_plan = fhegpu::RuntimePlanJsonReader::read_file(argv[1]);
        const auto loaded_spec = fhegpu::OperatorSpecReader::read_file(argv[2]);
        const auto &plan = loaded_plan.plan;
        if (plan.target.world_size != world_size ||
            plan.target.device_counts.size() !=
                static_cast<std::size_t>(world_size) ||
            std::any_of(plan.target.device_counts.begin(),
                        plan.target.device_counts.end(),
                        [](int count) { return count <= 0; }) ||
            plan.external_inputs.size() != 1 || plan.final_outputs.size() != 1)
        {
            throw std::runtime_error(
                "MPI GPU smoke plan topology or input/output arity is invalid");
        }
        const PlanStats stats = validate_smoke_plan(plan);
        const auto requirements =
            fhegpu::PlanVerifier::verify(plan, loaded_spec, false);
        poseidon::PoseidonContext context = make_context(loaded_spec.spec);

        auto secret_key = std::make_shared<poseidon::SecretKey>();
        if (rank == 0)
        {
            poseidon::KeyGenerator owner(context);
            *secret_key = owner.secret_key();
        }
        broadcast_secret_key(context, *secret_key, rank);
        poseidon::KeyGenerator key_generator(context, *secret_key);
        auto public_key = std::make_shared<poseidon::PublicKey>();
        auto relin_keys = std::make_shared<poseidon::RelinKeys>();
        auto galois_keys = std::make_shared<poseidon::GaloisKeys>();
        key_generator.create_public_key(*public_key);

        bool needs_relin = false;
        std::set<int> rotation_steps;
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
                    throw std::runtime_error(
                        "Galois key requirement has no rotation step");
                }
                rotation_steps.insert(*key.rotation_step);
            }
        }
        if (needs_relin)
        {
            key_generator.create_relin_keys(*relin_keys);
        }
        const auto key_basis = poseidon::runtime_api::binary_rotation_key_basis(
            rotation_steps, context.parameters_literal()->slot());
        if (!key_basis.empty())
        {
            key_generator.create_galois_keys(
                std::vector<int>(key_basis.begin(), key_basis.end()),
                *galois_keys);
        }

        const int local_device_count =
            plan.target.device_counts[static_cast<std::size_t>(rank)];
        const auto cuda_devices =
            local_cuda_devices(MPI_COMM_WORLD, local_device_count);
        poseidon::runtime_api::GpuProcessTopology topology;
        topology.device_counts = plan.target.device_counts;
        topology.rank_to_node = rank_to_node;
        PoseidonGpuApi api(loaded_spec.spec.context_id, context, MPI_COMM_WORLD,
                           cuda_devices, std::move(topology), relin_keys,
                           galois_keys, public_key, secret_key);
        if (api.mpi_rank() != rank || api.mpi_world_size() != world_size)
        {
            throw std::runtime_error("Poseidon GPU MPI identity mismatch");
        }

        const std::array<double, kInputSize> input{
            -0.75, -0.5, -0.25, 0.125, 0.25, 0.5, 0.75, 0.875};
        const fhegpu::ValueId input_id = plan.external_inputs.front();
        const auto &input_desc = find_value(plan, input_id);
        if (input_desc.kind != fhegpu::ValueKind::Ciphertext ||
            input_desc.place.kind != fhegpu::PlaceKind::Host)
        {
            throw std::runtime_error(
                "MPI GPU smoke input must be a Host ciphertext");
        }
        std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
        if (input_desc.place.rank == rank)
        {
            poseidon::CKKSEncoder encoder(context);
            poseidon::Plaintext plaintext;
            encoder.encode(
                std::vector<double>(input.begin(), input.end()),
                context.crt_context()->parms_id_map().at(
                    static_cast<std::uint32_t>(input_desc.level)),
                std::ldexp(1.0, input_desc.scale_log2), plaintext);
            poseidon::Encryptor encryptor(context, *public_key);
            poseidon::Ciphertext ciphertext;
            encryptor.encrypt(plaintext, ciphertext);
            inputs.emplace(
                input_id,
                PoseidonGpuValue::from_host_ciphertext(std::move(ciphertext)));
        }

        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(
            rank, world_size, local_device_count, api);
        const fhegpu::RuntimeResources resources{loaded_spec, std::nullopt, false};
        const auto artifact = runtime.run(loaded_plan, resources, inputs);

        const fhegpu::ValueId final_id = plan.final_outputs.front();
        const auto &final_desc = find_value(plan, final_id);
        const int final_rank = final_desc.place.rank;
        std::array<double, kInputSize> actual{};
        double max_imaginary = 0.0;
        if (rank == final_rank)
        {
            const auto &artifact_value = artifact.values.at(final_id).value;
            std::optional<PoseidonGpuValue> downloaded;
            const PoseidonGpuValue *host_value = &artifact_value;
            if (final_desc.place.kind == fhegpu::PlaceKind::Device)
            {
                downloaded.emplace(
                    download(api, final_id, final_desc.place, artifact_value));
                host_value = &*downloaded;
            }
            else if (final_desc.place.kind != fhegpu::PlaceKind::Host)
            {
                throw std::runtime_error("MPI GPU smoke output has invalid place");
            }
            poseidon::Decryptor decryptor(context, *secret_key);
            poseidon::Plaintext plaintext;
            decryptor.decrypt(host_value->host_ciphertext(), plaintext);
            poseidon::CKKSEncoder encoder(context);
            std::vector<std::complex<double>> decoded;
            encoder.decode(plaintext, decoded);
            for (std::size_t index = 0; index < actual.size(); ++index)
            {
                actual[index] = decoded.at(index).real();
                max_imaginary = std::max(
                    max_imaginary, std::abs(decoded.at(index).imag()));
            }
        }
        check_mpi(MPI_Bcast(actual.data(), static_cast<int>(actual.size()),
                            MPI_DOUBLE, final_rank, MPI_COMM_WORLD),
                  "MPI_Bcast decoded output");
        check_mpi(MPI_Bcast(&max_imaginary, 1, MPI_DOUBLE, final_rank,
                            MPI_COMM_WORLD),
                  "MPI_Bcast imaginary error");

        double max_abs_error = 0.0;
        for (std::size_t index = 0; index < actual.size(); ++index)
        {
            max_abs_error = std::max(
                max_abs_error,
                std::abs(actual[index] - kExpectedFactor * input[index]));
        }
        const int local_passed =
            max_abs_error <= kTolerance && max_imaginary <= kTolerance ? 1 : 0;
        int all_passed = 0;
        check_mpi(MPI_Allreduce(&local_passed, &all_passed, 1, MPI_INT,
                                MPI_MIN, MPI_COMM_WORLD),
                  "MPI_Allreduce result");

        if (rank == 0)
        {
            const Json report{
                {"format_version", 1},
                {"passed", all_passed != 0},
                {"expression", "-10*x"},
                {"world_size", world_size},
                {"device_counts", plan.target.device_counts},
                {"rank_to_node", rank_to_node},
                {"plan_sha256", loaded_plan.source_sha256},
                {"operator_spec_sha256", loaded_spec.source_sha256},
                {"negate_calls", stats.negates},
                {"add_calls", stats.adds},
                {"cross_rank_transfers", stats.cross_rank_transfers},
                {"cross_rank_device_transfers",
                 stats.cross_rank_device_transfers},
                {"cross_rank_host_uploads", stats.cross_rank_host_uploads},
                {"max_abs_error", max_abs_error},
                {"max_imaginary", max_imaginary},
                {"tolerance", kTolerance},
                {"actual", actual},
            };
            write_report(argv[3], report);
            std::cout << (all_passed != 0 ? "PASS" : "FAIL")
                      << " MPI GPU RuntimePlan: world=" << world_size
                      << " device_counts="
                      << encode_x_list(plan.target.device_counts)
                      << " cross_rank_transfers="
                      << stats.cross_rank_transfers
                      << " max_abs_error=" << max_abs_error << '\n';
        }
        exit_code = all_passed != 0 ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[rank " << rank << "] MPI GPU RuntimePlan failed: "
                  << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit_code = 1;
    }

    MPI_Finalize();
    return exit_code;
}
