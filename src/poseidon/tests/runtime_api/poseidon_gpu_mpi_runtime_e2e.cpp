#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "poseidon/runtime_api/rotation_key_basis.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/decryptor.h"
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include "runtime/plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/runtime.hpp"
#include "poseidon/runtime_api/communication/cuda_local_transfer.h"
#include "runtime/verifier.hpp"
#include "mpi_gpu_runtime_common.hpp"

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
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

constexpr const char *kUsage =
    "usage: poseidon_gpu_mpi_runtime_e2e PLAN OPERATOR_SPEC BUNDLE_DIR REPORT "
    "(--local | --rank-to-node 0x1) [--warmups N] [--iterations N] [--measure-memory] [--expected-output JSON] [--execution-mode sequential|per_device_workers]";

class NvtxRange
{
public:
    explicit NvtxRange(const char *name)
    {
        nvtxRangePushA(name);
    }

    ~NvtxRange()
    {
        end();
    }

    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;

    void end() noexcept
    {
        if (active_)
        {
            nvtxRangePop();
            active_ = false;
        }
    }

private:
    bool active_ = true;
};

// Install a measured pool before the API, and keep it alive until API caches are destroyed.
class MemoryCounter {
public:
    explicit MemoryCounter(int device) : device_(device) {
        if (cudaSetDevice(device) != cudaSuccess) throw std::runtime_error("memory counter device selection failed");
        previous_ = rmm::mr::get_current_device_resource();
        pool_ = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(&upstream_, 64ULL << 20);
        counter_ = std::make_unique<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>(pool_.get());
        rmm::mr::set_current_device_resource(counter_.get());
    }
    ~MemoryCounter() { rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_}, previous_); }
    int64_t active() const { return counter_->get_bytes_counter().value; }
    void begin() { counter_->push_counters(); }
    Json end(int rank, int index) {
        auto bytes = counter_->pop_counters().first;
        if (cudaSetDevice(device_) != cudaSuccess) throw std::runtime_error("memory sample device selection failed");
        std::size_t free = 0, total = 0;
        if (cudaMemGetInfo(&free, &total) != cudaSuccess) throw std::runtime_error("CUDA memory sample failed");
        return {{"rank", rank}, {"index", index}, {"peak_increment_bytes", bytes.peak},
                {"allocation_bytes", bytes.total}, {"net_change_bytes", bytes.value}, {"active_bytes", active()},
                {"pool_reserved_bytes", pool_->pool_size()}, {"cuda_free_bytes", free}, {"cuda_total_bytes", total}};
    }
private:
    int device_;
    rmm::mr::device_memory_resource *previous_;
    rmm::mr::cuda_memory_resource upstream_;
    std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool_;
    std::unique_ptr<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>> counter_;
};

struct RunnerOptions
{
    std::filesystem::path plan_path;
    std::filesystem::path spec_path;
    std::string bundle_text;
    std::filesystem::path report_path;
    bool local = false;
    std::vector<int> rank_to_node;
    std::size_t warmups = 0;
    std::size_t iterations = 1;
    bool measure_memory = false;
    fhegpu::DeviceExecutionMode execution_mode = fhegpu::DeviceExecutionMode::PerDeviceWorkers;
    std::optional<std::filesystem::path> expected_output;
};

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
    std::size_t release_instructions = 0;
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
        if (std::holds_alternative<fhegpu::ReleaseOp>(instruction.body))
        {
            ++stats.release_instructions;
            continue;
        }
        if (std::holds_alternative<fhegpu::EncodeOp>(instruction.body) ||
            std::holds_alternative<fhegpu::FenceOp>(instruction.body))
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

bool requests_local_mode(int argc, char **argv)
{
    return std::any_of(argv + std::min(argc, 5), argv + argc,
                       [](const char *argument) {
                           return std::string(argument) == "--local";
                       });
}

RunnerOptions parse_options(int argc, char **argv)
{
    if (argc < 5)
    {
        throw std::invalid_argument(kUsage);
    }

    RunnerOptions options;
    options.plan_path = argv[1];
    options.spec_path = argv[2];
    options.bundle_text = argv[3];
    options.report_path = argv[4];
    bool saw_local = false;
    bool saw_rank_to_node = false;
    for (int index = 5; index < argc; ++index)
    {
        const std::string option(argv[index]);
        if (option == "--local")
        {
            if (saw_local)
            {
                throw std::invalid_argument("--local may be specified only once");
            }
            saw_local = true;
            options.local = true;
            continue;
        }
        if (option == "--rank-to-node")
        {
            if (saw_rank_to_node)
            {
                throw std::invalid_argument(
                    "--rank-to-node may be specified only once");
            }
            if (++index >= argc)
            {
                throw std::invalid_argument(
                    "--rank-to-node requires an argument");
            }
            saw_rank_to_node = true;
            options.rank_to_node = parse_rank_to_node(argv[index]);
            continue;
        }
        if (option == "--execution-mode") {
            if (++index >= argc) throw std::invalid_argument("--execution-mode requires an argument");
            const std::string mode = argv[index];
            if (mode == "sequential") options.execution_mode = fhegpu::DeviceExecutionMode::Sequential;
            else if (mode == "per_device_workers") options.execution_mode = fhegpu::DeviceExecutionMode::PerDeviceWorkers;
            else throw std::invalid_argument("invalid execution mode");
            continue;
        }
        if (option == "--measure-memory") { options.measure_memory = true; continue; }
        if (option == "--expected-output") {
            if (++index >= argc) throw std::invalid_argument("--expected-output requires a JSON file");
            options.expected_output = argv[index]; continue;
        }
        if (option == "--warmups" || option == "--iterations")
        {
            if (++index >= argc)
            {
                throw std::invalid_argument(option + " requires an argument");
            }
            const std::size_t parsed =
                parse_positive(argv[index], option.c_str(), true);
            if (option == "--warmups")
            {
                options.warmups = parsed;
            }
            else
            {
                options.iterations = parsed;
            }
            continue;
        }
        throw std::invalid_argument("unknown option: " + option);
    }

    if (options.local == saw_rank_to_node)
    {
        throw std::invalid_argument(
            "specify exactly one of --local and --rank-to-node");
    }
    if (options.iterations == 0)
    {
        throw std::invalid_argument("--iterations must be positive");
    }
    return options;
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
    const bool local_mode = requests_local_mode(argc, argv);
    bool mpi_initialized = false;
    int rank = 0;
    int world_size = 1;
    if (!local_mode)
    {
        int provided = MPI_THREAD_SINGLE;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) !=
            MPI_SUCCESS)
        {
            return 2;
        }
        mpi_initialized = true;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    int exit_code = 0;

    try
    {
        const RunnerOptions options = parse_options(argc, argv);
        std::vector<int> rank_to_node =
            options.local ? std::vector<int>{0} : options.rank_to_node;
        if (static_cast<int>(rank_to_node.size()) != world_size)
        {
            throw std::invalid_argument(
                "rank-to-node length must match the runtime world size");
        }
        if (world_size > 1)
        {
            require_same_topology(rank_to_node, world_size);
        }

        NvtxRange setup_range("setup");
        const auto loaded_plan =
            fhegpu::RuntimePlanReader::read_file(options.plan_path.string());
        const auto loaded_spec =
            fhegpu::OperatorSpecReader::read_file(options.spec_path.string());
        const auto &plan = loaded_plan.plan;
        if (plan.target.world_size != world_size ||
            plan.target.device_counts.size() !=
                static_cast<std::size_t>(world_size) ||
            std::any_of(plan.target.device_counts.begin(),
                        plan.target.device_counts.end(),
                        [](int count) { return count <= 0; }) ||
            plan.external_inputs.empty() || plan.final_outputs.empty())
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
        if (world_size > 1)
        {
            broadcast_secret_key(context, *secret_key, rank);
        }
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
        std::vector<int> cuda_devices(
            static_cast<std::size_t>(local_device_count));
        if (world_size == 1)
        {
            std::iota(cuda_devices.begin(), cuda_devices.end(), 0);
        }
        else
        {
            cuda_devices =
                local_cuda_devices(MPI_COMM_WORLD, local_device_count);
        }

        std::vector<std::unique_ptr<MemoryCounter>> memory;
        if (options.measure_memory)
            for (int device : cuda_devices) memory.push_back(std::make_unique<MemoryCounter>(device));
        std::unique_ptr<PoseidonGpuApi> api;
        if (world_size == 1)
        {
            api = std::make_unique<PoseidonGpuApi>(
                loaded_spec.spec.context_id, context, cuda_devices, relin_keys,
                galois_keys, public_key, secret_key);
        }
        else
        {
            poseidon::runtime_api::GpuProcessTopology topology;
            topology.device_counts = plan.target.device_counts;
            topology.rank_to_node = rank_to_node;
            api = std::make_unique<PoseidonGpuApi>(
                loaded_spec.spec.context_id, context, MPI_COMM_WORLD,
                cuda_devices, std::move(topology), relin_keys, galois_keys,
                public_key, secret_key);
        }
        if (api->mpi_rank() != rank || api->mpi_world_size() != world_size)
        {
            throw std::runtime_error("Poseidon GPU runtime identity mismatch");
        }

        Json expected;
        if (options.expected_output) {
            std::ifstream expected_file(*options.expected_output);
            if (!expected_file) throw std::runtime_error("cannot open expected output");
            expected_file >> expected;
        }
        if (options.expected_output && plan.final_outputs.size() != 1)
            throw std::runtime_error("a single-output oracle cannot validate multiple outputs");
        std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
        for (const auto input_id : plan.external_inputs) {
            const auto &input_desc = find_value(plan, input_id);
            if (input_desc.kind != fhegpu::ValueKind::Ciphertext ||
                input_desc.place.kind != fhegpu::PlaceKind::Host)
            {
                throw std::runtime_error(
                    "generic runner requires Host ciphertext inputs");
            }
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
        }
        std::optional<std::filesystem::path> bundle_dir;
        if (options.bundle_text != "-")
        {
            bundle_dir = std::filesystem::path(options.bundle_text);
        }
        else if (plan.plaintext_bundle)
        {
            throw std::invalid_argument(
                "BUNDLE_DIR must name the plaintext bundle for this plan");
        }
        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(
            rank, world_size, local_device_count, *api,
            options.execution_mode,
            static_cast<std::size_t>(local_device_count));
        const fhegpu::RuntimeResources resources{
            loaded_spec, std::move(bundle_dir), false};
        setup_range.end();

        if (options.warmups > 0)
        {
            NvtxRange warmup_range("warmup");
            for (std::size_t index = 0; index < options.warmups; ++index)
            {
                if (world_size > 1)
                {
                    check_mpi(MPI_Barrier(MPI_COMM_WORLD),
                              "MPI_Barrier warmup");
                }
                (void)runtime.run(loaded_plan, resources, inputs);
            }
        }

        Json iteration_memory = Json::array();
        Json streaming_timing = Json::array();
        Json numerical_checks = Json::array();
        std::map<fhegpu::ValueId, std::vector<std::complex<double>>> first_outputs;
        std::vector<int64_t> memory_baseline;
        for (const auto &counter : memory) memory_baseline.push_back(counter->active());
        std::vector<double> local_online_seconds;
        local_online_seconds.reserve(options.iterations);
        for (std::size_t index = 0; index < options.iterations; ++index)
        {
            if (world_size > 1)
            {
                check_mpi(MPI_Barrier(MPI_COMM_WORLD),
                          "MPI_Barrier iteration");
            }
            const std::string range_name =
                "online.iteration." + std::to_string(index + 1);
            NvtxRange iteration_range(range_name.c_str());
            for (const auto &counter : memory) counter->begin();
            {
            const auto artifact = runtime.run(loaded_plan, resources, inputs);
            streaming_timing.push_back({{"encode_calls", artifact.timing.encode_calls},
                {"encode_seconds", artifact.timing.encode_nanoseconds * 1e-9},
                {"bundle_read_calls", artifact.timing.bundle_read_calls},
                {"bundle_read_seconds", artifact.timing.bundle_read_nanoseconds * 1e-9},
                {"bundle_read_bytes", artifact.timing.bundle_read_bytes},
                {"raw_peak_bytes", artifact.timing.raw_peak_bytes},
                {"fences", artifact.timing.fence_calls},
                {"fence_seconds", artifact.timing.fence_nanoseconds * 1e-9}});
            for (const auto output_id : plan.final_outputs) {
                auto output_desc = find_value(plan, output_id);
                if (output_desc.place.rank == rank) {
                    auto output = artifact.values.at(output_id).value;
                    if (output_desc.place.kind == fhegpu::PlaceKind::Device) {
                        fhegpu::CommAction transfer;
                        transfer.id = 1000000000ULL + index;
                        transfer.kind = fhegpu::CommKind::Transfer;
                        transfer.inputs = {output_id}; transfer.outputs = {output_id};
                        transfer.sources = {output_desc.place};
                        transfer.destinations = {{fhegpu::PlaceKind::Host, rank, 0}};
                        transfer.output_types = {fhegpu::ValueKind::Ciphertext};
                        output_desc.place = transfer.destinations.front();
                        auto handle = api->communicate_async(transfer, {output}, {output_desc});
                        auto downloaded = api->wait(handle);
                        if (downloaded.size() != 1 || !downloaded.front()) throw std::runtime_error("numerical check download failed");
                        output = std::move(*downloaded.front());
                    }
                    poseidon::Decryptor decryptor(context, *secret_key);
                    poseidon::CKKSEncoder encoder(context);
                    poseidon::Plaintext plaintext;
                    decryptor.decrypt(output.host_ciphertext(), plaintext);
                    std::vector<std::complex<double>> slots; encoder.decode(plaintext, slots);
                    for (const auto &slot : slots)
                        if (!std::isfinite(slot.real()) || !std::isfinite(slot.imag()))
                            throw std::runtime_error("non-finite numerical output");
                    auto &first_output = first_outputs[output_id];
                    if (first_output.empty()) first_output = slots;
                    double repeat_error = 0, expected_error = 0;
                    for (size_t i = 0; i < slots.size(); ++i) repeat_error = std::max(repeat_error, std::abs(slots[i] - first_output.at(i)));
                    if (repeat_error > 1e-4) throw std::runtime_error("repeated output changed");
                    if (!expected.is_null()) {
                        const auto &indices = expected.at("indices"), &values = expected.at("values");
                        if (indices.size() != values.size() || indices.empty()) throw std::runtime_error("invalid numerical oracle");
                        for (size_t i = 0; i < indices.size(); ++i)
                            expected_error = std::max(expected_error, std::abs(slots.at(indices[i].get<size_t>()) - values[i].get<double>()));
                        if (expected_error > expected.at("absolute_tolerance").get<double>()) throw std::runtime_error("output differs from numerical oracle");
                    }
                    numerical_checks.push_back({{"output_id", std::to_string(output_id)}, {"iteration", index + 1}, {"repeat_max_error", repeat_error}, {"expected_max_error", expected_error}});
                }
            }
            local_online_seconds.push_back(
                static_cast<double>(artifact.timing.online_execution_nanoseconds) *
                1e-9);
            }
            api->drain();
            Json devices = Json::array();
            for (size_t device = 0; device < memory.size(); ++device) {
                devices.push_back(memory[device]->end(rank, device));
                if (memory[device]->active() != memory_baseline[device])
                    throw std::runtime_error("GPU active memory did not return to the warmup baseline");
            }
            iteration_memory.push_back({{"iteration", index + 1}, {"devices", std::move(devices)}});
        }
        Json decoded_output = Json::array();
        // Preserve the old single-output report. Multi-output smoke runs report
        // finite/repeat checks per output, without implying a model oracle.
        if (first_outputs.size() == 1)
            for (const auto &slot : first_outputs.begin()->second)
                decoded_output.push_back({slot.real(), slot.imag()});
        write_report(options.report_path.string() + ".rank" + std::to_string(rank) + ".checks.json",
                     {{"rank", rank}, {"memory_measurement", "Active RMM allocations, including cached keys/parameters in the warmup baseline; pool reservation is excluded."},
                      {"streaming_timing", streaming_timing},
                      {"pinned_live_bytes", poseidon::runtime_api::communication::PinnedHostBuffer::live_bytes()},
                      {"pinned_process_peak_bytes", poseidon::runtime_api::communication::PinnedHostBuffer::peak_bytes()},
                      {"memory_baseline_bytes", memory_baseline}, {"iterations", iteration_memory}, {"numerical_checks", numerical_checks}, {"decoded_output", decoded_output}, {"oracle_provided", !expected.is_null()}});

        std::vector<double> gathered;
        if (world_size == 1)
        {
            gathered = local_online_seconds;
        }
        else
        {
            if (rank == 0)
            {
                gathered.resize(static_cast<std::size_t>(world_size) *
                                options.iterations);
            }
            check_mpi(MPI_Gather(
                          local_online_seconds.data(),
                          static_cast<int>(options.iterations), MPI_DOUBLE,
                          rank == 0 ? gathered.data() : nullptr,
                          static_cast<int>(options.iterations), MPI_DOUBLE, 0,
                          MPI_COMM_WORLD),
                      "MPI_Gather online timings");
        }

        if (rank == 0)
        {
            std::vector<double> critical_online_seconds(
                options.iterations, 0.0);
            for (std::size_t iteration = 0;
                 iteration < options.iterations; ++iteration)
            {
                for (int remote_rank = 0; remote_rank < world_size;
                     ++remote_rank)
                {
                    critical_online_seconds[iteration] = std::max(
                        critical_online_seconds[iteration],
                        gathered[static_cast<std::size_t>(remote_rank) *
                                     options.iterations +
                                 iteration]);
                }
            }
            const auto report = Json{
                {"format_version", 1},
                {"passed", true},
                {"runner", "poseidon_gpu_mpi_runtime_e2e"},
                {"execution_mode", options.local ? "local" : "mpi"},
                {"device_execution_mode", options.execution_mode == fhegpu::DeviceExecutionMode::Sequential ? "sequential" : "per_device_workers"},
                {"mpi_initialized", mpi_initialized},
                {"nccl_transport_enabled", world_size > 1},
                {"runtime_scope", "online_execution_only"},
                {"plan_sha256", loaded_plan.source_sha256},
                {"operator_spec_sha256", loaded_spec.source_sha256},
                {"world_size", world_size},
                {"device_counts", plan.target.device_counts},
                {"rank_to_node", rank_to_node},
                {"warmups", options.warmups},
                {"iterations", options.iterations},
                {"plan_stats",
                 {{"release_instructions", stats.release_instructions},
                  {"compute_instructions", stats.compute_instructions},
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
            write_report(options.report_path, report);
            const double average =
                report.at("online_timing").at("average_seconds").get<double>();
            std::cout << "PASS runner=poseidon_gpu_mpi_runtime_e2e"
                      << " world=" << world_size
                      << " mode=" << (options.local ? "local" : "mpi")
                      << " device_counts="
                      << encode_x_list(plan.target.device_counts)
                      << " online_average_seconds=" << average
                      << " cross_rank_communication="
                      << stats.cross_rank_communication << '\n'
                      << "report=" << options.report_path.string() << '\n';
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << (local_mode ? "[local]" :
                         "[rank " + std::to_string(rank) + "]")
                  << " GPU RuntimePlan failed: " << error.what() << '\n';
        if (mpi_initialized && world_size > 1)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        exit_code = 1;
    }

    if (mpi_initialized)
    {
        MPI_Finalize();
    }
    return exit_code;
}
