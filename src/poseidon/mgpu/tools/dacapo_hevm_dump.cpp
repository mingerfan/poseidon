#include "poseidon/mgpu/compiler/dacapo_artifacts.h"
#include "poseidon/mgpu/comm/execution_preflight.h"
#include "poseidon/mgpu/comm/topology.h"
#include "poseidon/mgpu/ir/schedule_summary.h"
#include "poseidon/mgpu/runtime/hevm_io_binding.h"
#include "poseidon/mgpu/runtime/poseidon_gpu_schedule_preflight.h"
#include "poseidon/util/json.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace poseidon::mgpu;

namespace
{

struct ToolOptions
{
    std::string hevm_path;
    std::string constants_path;
    int device_count = 1;
    int default_device = 0;
    std::vector<int> compute_devices;
    std::optional<int> upload_device;
    std::optional<int> download_device;
    bool round_robin_compute = false;
    bool dump_schedule = true;
    bool summary_json = false;
    bool poseidon_gpu_preflight = false;
    bool preflight_comm_available = false;
    bool preflight_relin_keys_available = false;
    bool preflight_galois_keys_available = false;
    bool communication_plan = false;
    bool communication_execution_preflight = false;
    bool execution_cuda_peer_available = false;
    bool execution_inter_node_available = false;
    int node_count = 1;
    int devices_per_node = 0;
    bool opcode_summary = false;
};

void print_usage(std::ostream &stream)
{
    stream
        << "usage: poseidon_mgpu_dacapo_hevm_dump --hevm <file> --constants <file> "
           "[--devices N] [--default-device N] [--round-robin-compute] "
           "[--compute-devices a,b,c] "
           "[--upload-device N] "
           "[--download-device N] "
           "[--poseidon-gpu-preflight] "
           "[--preflight-comm-available] "
           "[--preflight-relin-keys] "
           "[--preflight-galois-keys] "
           "[--communication-plan] "
           "[--communication-execution-preflight] "
           "[--execution-cuda-peer-available] "
           "[--execution-inter-node-available] "
           "[--opcode-summary] "
           "[--nodes N] "
           "[--devices-per-node N] "
           "[--summary-json] "
           "[--no-schedule]\n";
}

int parse_int_arg(const std::string &name, const char *value)
{
    if (value == nullptr)
    {
        throw std::invalid_argument("missing value for " + name);
    }

    std::size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    if (consumed != std::string(value).size())
    {
        throw std::invalid_argument("invalid integer for " + name + ": " + value);
    }
    return parsed;
}

std::vector<int> parse_compute_devices(const char *value)
{
    if (value == nullptr || std::string(value).empty())
    {
        throw std::invalid_argument("missing value for --compute-devices");
    }

    std::vector<int> devices;
    const std::string text = value;
    std::size_t begin = 0;
    while (begin <= text.size())
    {
        const std::size_t end = text.find(',', begin);
        const std::string item = text.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (item.empty())
        {
            throw std::invalid_argument("empty device id in --compute-devices");
        }
        devices.push_back(parse_int_arg("--compute-devices", item.c_str()));
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return devices;
}

ToolOptions parse_args(int argc, char **argv)
{
    ToolOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_usage(std::cout);
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--hevm")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --hevm");
            }
            options.hevm_path = argv[i];
            continue;
        }
        if (arg == "--constants")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --constants");
            }
            options.constants_path = argv[i];
            continue;
        }
        if (arg == "--devices")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --devices");
            }
            options.device_count = parse_int_arg("--devices", argv[i]);
            continue;
        }
        if (arg == "--default-device")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --default-device");
            }
            options.default_device = parse_int_arg("--default-device", argv[i]);
            continue;
        }
        if (arg == "--round-robin-compute")
        {
            options.round_robin_compute = true;
            continue;
        }
        if (arg == "--compute-devices")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --compute-devices");
            }
            options.compute_devices = parse_compute_devices(argv[i]);
            options.round_robin_compute = true;
            continue;
        }
        if (arg == "--upload-device")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --upload-device");
            }
            options.upload_device = parse_int_arg("--upload-device", argv[i]);
            continue;
        }
        if (arg == "--download-device")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --download-device");
            }
            options.download_device = parse_int_arg("--download-device", argv[i]);
            continue;
        }
        if (arg == "--no-schedule")
        {
            options.dump_schedule = false;
            continue;
        }
        if (arg == "--summary-json")
        {
            options.summary_json = true;
            continue;
        }
        if (arg == "--poseidon-gpu-preflight")
        {
            options.poseidon_gpu_preflight = true;
            continue;
        }
        if (arg == "--preflight-comm-available")
        {
            options.preflight_comm_available = true;
            options.poseidon_gpu_preflight = true;
            continue;
        }
        if (arg == "--preflight-relin-keys")
        {
            options.preflight_relin_keys_available = true;
            options.poseidon_gpu_preflight = true;
            continue;
        }
        if (arg == "--preflight-galois-keys")
        {
            options.preflight_galois_keys_available = true;
            options.poseidon_gpu_preflight = true;
            continue;
        }
        if (arg == "--communication-plan")
        {
            options.communication_plan = true;
            continue;
        }
        if (arg == "--communication-execution-preflight")
        {
            options.communication_plan = true;
            options.communication_execution_preflight = true;
            continue;
        }
        if (arg == "--execution-cuda-peer-available")
        {
            options.communication_plan = true;
            options.communication_execution_preflight = true;
            options.execution_cuda_peer_available = true;
            continue;
        }
        if (arg == "--execution-inter-node-available")
        {
            options.communication_plan = true;
            options.communication_execution_preflight = true;
            options.execution_inter_node_available = true;
            continue;
        }
        if (arg == "--opcode-summary")
        {
            options.opcode_summary = true;
            continue;
        }
        if (arg == "--nodes")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --nodes");
            }
            options.node_count = parse_int_arg("--nodes", argv[i]);
            options.communication_plan = true;
            continue;
        }
        if (arg == "--devices-per-node")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --devices-per-node");
            }
            options.devices_per_node = parse_int_arg("--devices-per-node", argv[i]);
            options.communication_plan = true;
            continue;
        }

        throw std::invalid_argument("unknown argument: " + arg);
    }

    if (options.hevm_path.empty())
    {
        throw std::invalid_argument("--hevm is required");
    }
    if (options.constants_path.empty())
    {
        throw std::invalid_argument("--constants is required");
    }
    if (options.device_count <= 0)
    {
        throw std::invalid_argument("--devices must be positive");
    }
    if (options.default_device < 0 || options.default_device >= options.device_count)
    {
        throw std::invalid_argument("--default-device must be in [0, devices)");
    }
    if (options.upload_device.has_value() &&
        (*options.upload_device < 0 || *options.upload_device >= options.device_count))
    {
        throw std::invalid_argument("--upload-device must be in [0, devices)");
    }
    if (options.download_device.has_value() &&
        (*options.download_device < 0 || *options.download_device >= options.device_count))
    {
        throw std::invalid_argument("--download-device must be in [0, devices)");
    }
    if (options.node_count <= 0)
    {
        throw std::invalid_argument("--nodes must be positive");
    }
    if (options.devices_per_node < 0)
    {
        throw std::invalid_argument("--devices-per-node must be non-negative");
    }
    if (options.communication_plan)
    {
        const int devices_per_node =
            options.devices_per_node == 0 ? options.device_count : options.devices_per_node;
        if (devices_per_node <= 0)
        {
            throw std::invalid_argument("communication topology devices per node must be positive");
        }
        if (options.node_count * devices_per_node < options.device_count)
        {
            throw std::invalid_argument(
                "communication topology has fewer logical devices than --devices");
        }
    }
    return options;
}

std::string read_binary_file(const std::string &path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("failed to open file: " + path);
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (stream.bad())
    {
        throw std::runtime_error("failed to read file: " + path);
    }
    return buffer.str();
}

StaticSchedulePipelineOptions make_pipeline_options(const ToolOptions &tool_options)
{
    StaticSchedulePipelineOptions options;
    options.device_count = tool_options.device_count;
    options.emit_debug_dump = tool_options.dump_schedule;
    options.placement.default_device = tool_options.default_device;
    if (tool_options.round_robin_compute)
    {
        options.placement.policy = StaticPlacementPolicy::RoundRobinCompute;
    }
    options.placement.compute_devices = tool_options.compute_devices;
    options.placement.upload_device = tool_options.upload_device;
    options.placement.download_device = tool_options.download_device;
    return options;
}

nlohmann::json parse_json_object(const std::string &text)
{
    return nlohmann::json::parse(text);
}

MgpuTopology make_tool_topology(const ToolOptions &tool_options)
{
    if (tool_options.node_count == 1 && tool_options.devices_per_node == 0)
    {
        return make_single_node_topology(tool_options.device_count);
    }

    const int devices_per_node =
        tool_options.devices_per_node == 0 ? tool_options.device_count
                                           : tool_options.devices_per_node;
    return make_uniform_cluster_topology(tool_options.node_count, devices_per_node);
}

nlohmann::json optional_preflight_json(
    const ToolOptions &tool_options,
    const MgpuSchedule &schedule)
{
    if (!tool_options.poseidon_gpu_preflight)
    {
        return nullptr;
    }

    const PoseidonGpuSchedulePreflightResult result =
        preflight_poseidon_gpu_schedule(
            schedule,
            PoseidonGpuSchedulePreflightOptions{
                tool_options.device_count,
                tool_options.preflight_comm_available,
                tool_options.preflight_relin_keys_available,
                tool_options.preflight_galois_keys_available,
            });
    return parse_json_object(poseidon_gpu_schedule_preflight_to_json(result, -1));
}

nlohmann::json optional_communication_plan_json(
    const ToolOptions &tool_options,
    const MgpuSchedule &schedule)
{
    if (!tool_options.communication_plan)
    {
        return nullptr;
    }

    const MgpuCommunicationPlan plan =
        plan_schedule_communication(schedule, make_tool_topology(tool_options));
    return parse_json_object(communication_plan_to_json(plan, -1));
}

nlohmann::json optional_communication_execution_preflight_json(
    const ToolOptions &tool_options,
    const MgpuSchedule &schedule)
{
    if (!tool_options.communication_execution_preflight)
    {
        return nullptr;
    }

    const MgpuCommunicationPlan plan =
        plan_schedule_communication(schedule, make_tool_topology(tool_options));
    if (!plan.ok())
    {
        return nullptr;
    }

    const MgpuCommunicationExecutionPreflight preflight =
        preflight_communication_execution(
            plan,
            MgpuCommunicationExecutionOptions{
                true,
                tool_options.execution_cuda_peer_available,
                tool_options.execution_inter_node_available,
            });
    return parse_json_object(communication_execution_preflight_to_json(preflight, -1));
}

nlohmann::json hevm_opcode_summary_json(const DacapoHevmOpcodeSummary &summary)
{
    nlohmann::json root;
    root["ok"] = summary.ok();
    root["operation_count"] = summary.operation_count;
    root["alloc_count"] = summary.alloc_count;
    root["opcode_counts"] = nlohmann::json::array();
    for (const DacapoHevmOpcodeCount &count : summary.opcode_counts)
    {
        root["opcode_counts"].push_back(nlohmann::json{
            { "opcode", count.opcode },
            { "name", count.name },
            { "count", count.count },
            { "supported", count.supported },
        });
    }
    root["diagnostics"] = nlohmann::json::array();
    for (const DacapoAdapterDiagnostic &diagnostic : summary.diagnostics)
    {
        root["diagnostics"].push_back(nlohmann::json{
            { "offset", diagnostic.offset },
            { "message", diagnostic.message },
        });
    }
    return root;
}

void print_opcode_summary_text(const DacapoHevmOpcodeSummary &summary)
{
    std::cout << "hevm_opcode_summary:\n";
    std::cout << "  operations: " << summary.operation_count << '\n';
    std::cout << "  allocs: " << summary.alloc_count << '\n';
    for (const DacapoHevmOpcodeCount &count : summary.opcode_counts)
    {
        std::cout << "  opcode " << count.opcode << " " << count.name << ": "
                  << count.count << " supported="
                  << (count.supported ? "true" : "false") << '\n';
    }
    if (!summary.diagnostics.empty())
    {
        std::cout << "  diagnostics:\n";
        for (const DacapoAdapterDiagnostic &diagnostic : summary.diagnostics)
        {
            std::cout << "    offset " << diagnostic.offset << ": "
                      << diagnostic.message << '\n';
        }
    }
}

void print_optional_preflight(
    const ToolOptions &tool_options,
    const MgpuSchedule &schedule)
{
    if (!tool_options.poseidon_gpu_preflight)
    {
        return;
    }

    const PoseidonGpuSchedulePreflightResult result =
        preflight_poseidon_gpu_schedule(
            schedule,
            PoseidonGpuSchedulePreflightOptions{
                tool_options.device_count,
                tool_options.preflight_comm_available,
                tool_options.preflight_relin_keys_available,
                tool_options.preflight_galois_keys_available,
            });
    dump_poseidon_gpu_schedule_preflight(std::cout, result);
}

void print_optional_communication_plan(
    const ToolOptions &tool_options,
    const MgpuSchedule &schedule)
{
    if (!tool_options.communication_plan)
    {
        return;
    }

    const MgpuCommunicationPlan plan =
        plan_schedule_communication(schedule, make_tool_topology(tool_options));
    dump_communication_plan(std::cout, plan);
}

void print_optional_communication_execution_preflight(
    const ToolOptions &tool_options,
    const MgpuSchedule &schedule)
{
    if (!tool_options.communication_execution_preflight)
    {
        return;
    }

    const MgpuCommunicationPlan plan =
        plan_schedule_communication(schedule, make_tool_topology(tool_options));
    if (!plan.ok())
    {
        return;
    }

    const MgpuCommunicationExecutionPreflight preflight =
        preflight_communication_execution(
            plan,
            MgpuCommunicationExecutionOptions{
                true,
                tool_options.execution_cuda_peer_available,
                tool_options.execution_inter_node_available,
            });
    dump_communication_execution_preflight(std::cout, preflight);
}

void print_io_summary(const HevmIoBindingPlan &plan)
{
    std::cout << "hevm_io:\n";
    std::cout << "  cipher_inputs: " << plan.cipher_inputs.size() << '\n';
    std::cout << "  plaintext_constants: " << plan.plain_inputs.size() << '\n';
    std::cout << "  results: " << plan.results.size() << '\n';
}

void print_text_summary(
    const MgpuScheduleSummary &summary, std::size_t constant_count,
    const HevmIoBindingPlan &io_plan)
{
    std::cout << "schedule_ops: " << summary.total_ops << '\n';
    std::cout << "constants: " << constant_count << '\n';
    print_io_summary(io_plan);
    std::cout << "op_counts:\n";
    for (const MgpuOpKindCount &count : summary.op_counts)
    {
        if (count.count > 0)
        {
            std::cout << "  " << to_string(count.kind) << ": " << count.count << '\n';
        }
    }
    std::cout << "device_op_counts:\n";
    for (const MgpuDeviceOpCount &count : summary.device_op_counts)
    {
        std::cout << "  device " << count.device_id << ": " << count.count << '\n';
    }
    if (summary.unassigned_device_ops > 0)
    {
        std::cout << "  unassigned: " << summary.unassigned_device_ops << '\n';
    }
    if (summary.invalid_device_ops > 0)
    {
        std::cout << "  invalid: " << summary.invalid_device_ops << '\n';
    }
}

nlohmann::json make_tool_summary_json(
    const MgpuScheduleSummary &summary, std::size_t constant_count,
    const HevmIoBindingPlan &io_plan, nlohmann::json preflight,
    nlohmann::json communication_plan,
    nlohmann::json communication_execution_preflight,
    nlohmann::json opcode_summary,
    std::optional<std::string> debug_dump)
{
    nlohmann::json root;
    root["version"] = 1;
    root["schedule"] = parse_json_object(schedule_summary_to_json(summary, -1));
    root["constants"] = nlohmann::json{
        { "vectors", constant_count },
    };
    root["hevm_io"] = nlohmann::json{
        { "cipher_inputs", io_plan.cipher_inputs.size() },
        { "plaintext_constants", io_plan.plain_inputs.size() },
        { "results", io_plan.results.size() },
    };
    if (!preflight.is_null())
    {
        root["poseidon_gpu_preflight"] = std::move(preflight);
    }
    if (!communication_plan.is_null())
    {
        root["communication_plan"] = std::move(communication_plan);
    }
    if (!communication_execution_preflight.is_null())
    {
        root["communication_execution_preflight"] =
            std::move(communication_execution_preflight);
    }
    if (!opcode_summary.is_null())
    {
        root["hevm_opcode_summary"] = std::move(opcode_summary);
    }
    if (debug_dump.has_value())
    {
        root["debug_dump"] = *debug_dump;
    }
    return root;
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        const ToolOptions tool_options = parse_args(argc, argv);
        std::optional<DacapoHevmOpcodeSummary> opcode_summary;
        if (tool_options.opcode_summary)
        {
            opcode_summary = summarize_hevm_opcodes(read_binary_file(tool_options.hevm_path));
            if (!opcode_summary->ok())
            {
                std::cerr << opcode_summary->format_diagnostics() << '\n';
                return EXIT_FAILURE;
            }
        }

        const DacapoHevmArtifactResult artifacts =
            prepare_dacapo_hevm_artifacts_from_files(
                DacapoHevmArtifactPaths{
                    tool_options.hevm_path,
                    tool_options.constants_path,
                },
                make_pipeline_options(tool_options));
        if (!artifacts.ok())
        {
            if (opcode_summary.has_value())
            {
                print_opcode_summary_text(*opcode_summary);
            }
            std::cerr << artifacts.format_diagnostics() << '\n';
            return EXIT_FAILURE;
        }

        const HevmIoBindingPlanResult io_plan =
            build_hevm_io_binding_plan(artifacts.schedule);
        if (!io_plan.ok())
        {
            std::cerr << io_plan.format_diagnostics() << '\n';
            return EXIT_FAILURE;
        }

        const MgpuScheduleSummary summary =
            summarize_schedule(artifacts.schedule, tool_options.device_count);

        if (tool_options.summary_json)
        {
            const std::optional<std::string> debug_dump =
                tool_options.dump_schedule ? std::optional<std::string>(artifacts.debug_dump)
                                           : std::nullopt;
            nlohmann::json preflight =
                optional_preflight_json(tool_options, artifacts.schedule);
            nlohmann::json communication_plan =
                optional_communication_plan_json(tool_options, artifacts.schedule);
            nlohmann::json communication_execution_preflight =
                optional_communication_execution_preflight_json(
                    tool_options, artifacts.schedule);
            nlohmann::json opcode_summary_json_value =
                opcode_summary.has_value() ? hevm_opcode_summary_json(*opcode_summary)
                                           : nullptr;
            std::cout << make_tool_summary_json(
                             summary, artifacts.constants.values.size(), io_plan.plan,
                             std::move(preflight), std::move(communication_plan),
                             std::move(communication_execution_preflight),
                             std::move(opcode_summary_json_value), debug_dump)
                             .dump(2)
                      << '\n';
        }
        else
        {
            if (opcode_summary.has_value())
            {
                print_opcode_summary_text(*opcode_summary);
            }
            print_text_summary(summary, artifacts.constants.values.size(), io_plan.plan);
            print_optional_preflight(tool_options, artifacts.schedule);
            print_optional_communication_plan(tool_options, artifacts.schedule);
            print_optional_communication_execution_preflight(
                tool_options, artifacts.schedule);

            if (tool_options.dump_schedule)
            {
                std::cout << "\n" << artifacts.debug_dump;
            }
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "poseidon_mgpu_dacapo_hevm_dump: " << ex.what() << '\n';
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
