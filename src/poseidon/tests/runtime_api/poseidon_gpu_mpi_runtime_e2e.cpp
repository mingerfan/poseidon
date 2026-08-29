#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "poseidon/runtime_api/rotation_key_basis.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/runtime.hpp"
#include "runtime/verifier.hpp"
#include "mpi_gpu_runtime_common.hpp"

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
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
using poseidon::runtime_api::test::broadcast_secret_key;
using poseidon::runtime_api::test::check_mpi;
using poseidon::runtime_api::test::encode_x_list;
using poseidon::runtime_api::test::find_value;
using poseidon::runtime_api::test::local_cuda_devices;
using poseidon::runtime_api::test::make_context;
using poseidon::runtime_api::test::parse_rank_to_node;
using poseidon::runtime_api::test::require_same_topology;

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

struct GenericPlanStats
{
    std::size_t compute_instructions = 0;
    std::size_t initialization_compute = 0;
    std::size_t execution_compute = 0;
    std::size_t finalization_compute = 0;
    std::size_t communication_instructions = 0;
    std::size_t initialization_communication = 0;
    std::size_t execution_communication = 0;
    std::size_t finalization_communication = 0;
    std::size_t cross_rank_communication = 0;
    std::size_t cross_rank_device_communication = 0;
};

void count_phase(const std::vector<fhegpu::Instruction> &phase,
                 std::size_t &compute_count,
                 std::size_t &communication_count,
                 GenericPlanStats &stats)
{
    for (const auto &instruction : phase)
    {
        if (std::holds_alternative<fhegpu::ComputeOp>(instruction.body))
        {
            ++compute_count;
            ++stats.compute_instructions;
            continue;
        }
        if (std::holds_alternative<fhegpu::EncodeOp>(instruction.body))
        {
            continue;
        }
        const auto *communication =
            std::get_if<fhegpu::CommAction>(&instruction.body);
        if (communication == nullptr)
        {
            throw std::runtime_error("RuntimePlan contains an unknown instruction");
        }
        ++communication_count;
        ++stats.communication_instructions;
        for (std::size_t index = 0; index < communication->destinations.size();
             ++index)
        {
            if (communication->sources.front().rank ==
                communication->destinations.at(index).rank)
            {
                continue;
            }
            ++stats.cross_rank_communication;
            if (communication->sources.front().kind ==
                    fhegpu::PlaceKind::Device &&
                communication->destinations.at(index).kind ==
                    fhegpu::PlaceKind::Device)
            {
                ++stats.cross_rank_device_communication;
            }
        }
    }
}

GenericPlanStats summarize_plan(const fhegpu::RuntimePlan &plan)
{
    GenericPlanStats stats;
    count_phase(plan.initialization, stats.initialization_compute,
                stats.initialization_communication, stats);
    count_phase(plan.execution, stats.execution_compute,
                stats.execution_communication, stats);
    count_phase(plan.finalization, stats.finalization_compute,
                stats.finalization_communication, stats);
    return stats;
}

std::size_t parse_positive(const char *text, const char *name,
                           bool allow_zero = false)
{
    std::size_t consumed = 0;
    const std::string value(text);
    const unsigned long parsed = std::stoul(value, &consumed);
    if (consumed != value.size() || (!allow_zero && parsed == 0))
    {
        throw std::invalid_argument(std::string(name) +
                                    " must be a positive integer");
    }
    if (parsed > static_cast<unsigned long>(INT_MAX))
    {
        throw std::invalid_argument(std::string(name) + " is too large");
    }
    return static_cast<std::size_t>(parsed);
}

Json timing_json(const std::vector<double> &critical_online_seconds)
{
    if (critical_online_seconds.empty())
    {
        throw std::invalid_argument("no measured iterations");
    }
    const auto [minimum, maximum] = std::minmax_element(
        critical_online_seconds.begin(), critical_online_seconds.end());
    const double sum = std::accumulate(
        critical_online_seconds.begin(), critical_online_seconds.end(), 0.0);
    return {
        {"per_iteration_seconds", critical_online_seconds},
        {"average_seconds", sum / critical_online_seconds.size()},
        {"min_seconds", *minimum},
        {"max_seconds", *maximum},
    };
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
        if (argc < 7)
        {
            throw std::invalid_argument(
                "usage: poseidon_gpu_mpi_runtime_e2e PLAN OPERATOR_SPEC "
                "BUNDLE_DIR REPORT --rank-to-node 0x0 "
                "[--warmups N] [--iterations N]");
        }
        const std::filesystem::path plan_path = argv[1];
        const std::filesystem::path spec_path = argv[2];
        const std::string bundle_text = argv[3];
        const std::filesystem::path report_path = argv[4];
        if (std::string(argv[5]) != "--rank-to-node")
        {
            throw std::invalid_argument(
                "the --rank-to-node option is required");
        }
        const std::vector<int> rank_to_node = parse_rank_to_node(argv[6]);
        if (static_cast<int>(rank_to_node.size()) != world_size)
        {
            throw std::invalid_argument(
                "rank-to-node length must match MPI world size");
        }
        require_same_topology(rank_to_node, world_size);

        std::size_t warmups = 0;
        std::size_t iterations = 1;
        for (int index = 7; index < argc; ++index)
        {
            const std::string option(argv[index]);
            if ((option == "--warmups" || option == "--iterations") &&
                index + 1 < argc)
            {
                const std::size_t parsed =
                    parse_positive(argv[++index], option.c_str(), true);
                if (option == "--warmups")
                {
                    warmups = parsed;
                }
                else
                {
                    iterations = parsed;
                }
                continue;
            }
            throw std::invalid_argument("unknown option: " + option);
        }
        if (iterations == 0)
        {
            throw std::invalid_argument("--iterations must be positive");
        }
        if (iterations > static_cast<std::size_t>(INT_MAX))
        {
            throw std::invalid_argument("--iterations is too large");
        }

        const auto loaded_plan =
            fhegpu::RuntimePlanJsonReader::read_file(plan_path.string());
        const auto loaded_spec =
            fhegpu::OperatorSpecReader::read_file(spec_path.string());
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
                "RuntimePlan topology and input/output arity do not match the MPI run");
        }
        const auto requirements =
            fhegpu::PlanVerifier::verify(plan, loaded_spec, false);
        const GenericPlanStats stats = summarize_plan(plan);
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

        const fhegpu::ValueId input_id = plan.external_inputs.front();
        const auto &input_desc = find_value(plan, input_id);
        if (input_desc.kind != fhegpu::ValueKind::Ciphertext ||
            input_desc.place.kind != fhegpu::PlaceKind::Host)
        {
            throw std::runtime_error(
                "generic MPI runner requires one Host ciphertext input");
        }
        std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
        if (input_desc.place.rank == rank)
        {
            poseidon::CKKSEncoder encoder(context);
            poseidon::Plaintext plaintext;
            const std::vector<double> input{
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            encoder.encode(
                input,
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

        std::optional<std::filesystem::path> bundle_dir;
        if (bundle_text != "-")
        {
            bundle_dir = std::filesystem::path(bundle_text);
        }
        else if (plan.plaintext_bundle)
        {
            throw std::invalid_argument(
                "BUNDLE_DIR must name the plaintext bundle for this plan");
        }
        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(
            rank, world_size, local_device_count, api,
            fhegpu::DeviceExecutionMode::PerDeviceWorkers,
            static_cast<std::size_t>(local_device_count));
        const fhegpu::RuntimeResources resources{
            loaded_spec, std::move(bundle_dir), false};

        for (std::size_t index = 0; index < warmups; ++index)
        {
            check_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier warmup");
            (void)runtime.run(loaded_plan, resources, inputs);
        }

        std::vector<double> local_online_seconds;
        local_online_seconds.reserve(iterations);
        for (std::size_t index = 0; index < iterations; ++index)
        {
            check_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier iteration");
            const auto artifact = runtime.run(loaded_plan, resources, inputs);
            local_online_seconds.push_back(
                static_cast<double>(artifact.timing.online_execution_nanoseconds) *
                1e-9);
        }

        std::vector<double> gathered;
        if (rank == 0)
        {
            gathered.resize(static_cast<std::size_t>(world_size) * iterations);
        }
        check_mpi(MPI_Gather(
                      local_online_seconds.data(), static_cast<int>(iterations),
                      MPI_DOUBLE, rank == 0 ? gathered.data() : nullptr,
                      static_cast<int>(iterations), MPI_DOUBLE, 0,
                      MPI_COMM_WORLD),
                  "MPI_Gather online timings");

        if (rank == 0)
        {
            std::vector<double> critical_online_seconds(iterations, 0.0);
            for (std::size_t iteration = 0; iteration < iterations; ++iteration)
            {
                for (int remote_rank = 0; remote_rank < world_size;
                     ++remote_rank)
                {
                    critical_online_seconds[iteration] = std::max(
                        critical_online_seconds[iteration],
                        gathered[static_cast<std::size_t>(remote_rank) * iterations +
                                 iteration]);
                }
            }
            const auto report = Json{
                {"format_version", 1},
                {"passed", true},
                {"runner", "poseidon_gpu_mpi_runtime_e2e"},
                {"runtime_scope", "online_execution_only"},
                {"plan_sha256", loaded_plan.source_sha256},
                {"operator_spec_sha256", loaded_spec.source_sha256},
                {"world_size", world_size},
                {"device_counts", plan.target.device_counts},
                {"rank_to_node", rank_to_node},
                {"warmups", warmups},
                {"iterations", iterations},
                {"plan_stats",
                 {{"compute_instructions", stats.compute_instructions},
                  {"initialization_compute", stats.initialization_compute},
                  {"execution_compute", stats.execution_compute},
                  {"finalization_compute", stats.finalization_compute},
                  {"communication_instructions",
                   stats.communication_instructions},
                  {"initialization_communication",
                   stats.initialization_communication},
                  {"execution_communication", stats.execution_communication},
                  {"finalization_communication",
                   stats.finalization_communication},
                  {"cross_rank_communication", stats.cross_rank_communication},
                  {"cross_rank_device_communication",
                   stats.cross_rank_device_communication}}},
                {"online_timing", timing_json(critical_online_seconds)},
            };
            write_report(report_path, report);
            const double average =
                report.at("online_timing").at("average_seconds").get<double>();
            std::cout << "PASS runner=poseidon_gpu_mpi_runtime_e2e"
                      << " world=" << world_size
                      << " device_counts="
                      << encode_x_list(plan.target.device_counts)
                      << " online_average_seconds=" << average
                      << " cross_rank_communication="
                      << stats.cross_rank_communication << '\n'
                      << "report=" << report_path.string() << '\n';
        }
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
