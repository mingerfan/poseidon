#include "poseidon/frontends/dacapo/dacapo_artifacts.h"
#include "poseidon/mgpu/compiler/static_schedule_config.h"
#include "poseidon/mgpu/comm/execution_preflight.h"
#include "poseidon/mgpu/comm/topology.h"
#include "poseidon/mgpu/ir/schedule_summary.h"
#include "poseidon/mgpu/runtime/preflight/poseidon_gpu_execution_preflight.h"
#include "poseidon/frontends/dacapo/hevm_artifact_report.h"
#include "poseidon/frontends/dacapo/hevm_io_binding.h"
#include "poseidon/mgpu/runtime/preflight/poseidon_gpu_schedule_preflight.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
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
    StaticScheduleExecutionConfig config;
    bool dump_schedule = true;
    bool summary_json = false;
    std::string summary_json_path;
    std::string schedule_path;
};

std::string read_binary_file(const std::string &path);

void print_usage(std::ostream &stream)
{
    stream
        << "usage: poseidon_mgpu_dacapo_hevm_dump --hevm <file> --constants <file> "
           "[--config <file>] "
           "[--devices N] [--default-device N] "
           "[--scheduler single_device|greedy_ready|value_aware_heft|value_aware_peft] "
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
           "[--require-ready] "
           "[--nodes N] "
           "[--devices-per-node N] "
           "[--summary-json] "
           "[--write-summary-json <file>] "
           "[--write-schedule <file>] "
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

std::int64_t topology_device_count(int node_count, int devices_per_node)
{
    return static_cast<std::int64_t>(node_count) *
           static_cast<std::int64_t>(devices_per_node);
}

bool argument_requires_value(const std::string &arg)
{
    return arg == "--config" || arg == "--hevm" || arg == "--constants" ||
           arg == "--devices" || arg == "--default-device" ||
           arg == "--scheduler" ||
           arg == "--compute-devices" || arg == "--upload-device" ||
           arg == "--download-device" || arg == "--write-summary-json" ||
           arg == "--write-schedule" || arg == "--nodes" ||
           arg == "--devices-per-node";
}

void load_config_file(ToolOptions &options, const char *path)
{
    if (path == nullptr)
    {
        throw std::invalid_argument("missing value for --config");
    }

    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(read_binary_file(path));
    if (!result.ok())
    {
        throw std::invalid_argument(
            "invalid --config " + std::string(path) + ":\n" +
            result.format_diagnostics());
    }
    options.config = result.config;
    options.dump_schedule = result.config.pipeline.emit_debug_dump;
}

void load_config_files_first(ToolOptions &options, int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_usage(std::cout);
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--config")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --config");
            }
            load_config_file(options, argv[i]);
            continue;
        }
        if (argument_requires_value(arg) && i + 1 < argc)
        {
            ++i;
        }
    }
}

void apply_require_ready(ToolOptions &options)
{
    if (!options.config.require_ready)
    {
        return;
    }
    options.config.opcode_summary = true;
    options.config.poseidon_gpu_preflight = true;
    options.config.communication_plan = true;
    options.config.communication_execution_preflight = true;
}

ToolOptions parse_args(int argc, char **argv)
{
    ToolOptions options;
    load_config_files_first(options, argc, argv);
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_usage(std::cout);
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--config")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --config");
            }
            continue;
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
            options.config.pipeline.device_count = parse_int_arg("--devices", argv[i]);
            continue;
        }
        if (arg == "--default-device")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --default-device");
            }
            options.config.pipeline.scheduler.default_device =
                parse_int_arg("--default-device", argv[i]);
            continue;
        }
        if (arg == "--scheduler")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --scheduler");
            }
            const std::optional<StaticSchedulerKind> kind =
                static_scheduler_kind_from_string(argv[i]);
            if (!kind.has_value())
            {
                throw std::invalid_argument(
                    "unknown scheduler kind for --scheduler: " + std::string(argv[i]));
            }
            options.config.pipeline.scheduler.kind = *kind;
            continue;
        }
        if (arg == "--round-robin-compute")
        {
            throw std::invalid_argument(
                "--round-robin-compute is no longer supported; use --scheduler");
        }
        if (arg == "--compute-devices")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --compute-devices");
            }
            options.config.pipeline.scheduler.compute_devices =
                parse_compute_devices(argv[i]);
            continue;
        }
        if (arg == "--upload-device")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --upload-device");
            }
            options.config.pipeline.scheduler.upload_device =
                parse_int_arg("--upload-device", argv[i]);
            continue;
        }
        if (arg == "--download-device")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --download-device");
            }
            options.config.pipeline.scheduler.download_device =
                parse_int_arg("--download-device", argv[i]);
            continue;
        }
        if (arg == "--no-schedule")
        {
            options.dump_schedule = false;
            options.config.pipeline.emit_debug_dump = false;
            continue;
        }
        if (arg == "--summary-json")
        {
            options.summary_json = true;
            continue;
        }
        if (arg == "--write-summary-json")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --write-summary-json");
            }
            options.summary_json_path = argv[i];
            continue;
        }
        if (arg == "--write-schedule")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --write-schedule");
            }
            options.schedule_path = argv[i];
            continue;
        }
        if (arg == "--poseidon-gpu-preflight")
        {
            options.config.poseidon_gpu_preflight = true;
            continue;
        }
        if (arg == "--preflight-comm-available")
        {
            options.config.preflight_comm_available = true;
            options.config.poseidon_gpu_preflight = true;
            continue;
        }
        if (arg == "--preflight-relin-keys")
        {
            options.config.preflight_relin_keys_available = true;
            options.config.poseidon_gpu_preflight = true;
            continue;
        }
        if (arg == "--preflight-galois-keys")
        {
            options.config.preflight_galois_keys_available = true;
            options.config.poseidon_gpu_preflight = true;
            continue;
        }
        if (arg == "--communication-plan")
        {
            options.config.communication_plan = true;
            continue;
        }
        if (arg == "--communication-execution-preflight")
        {
            options.config.communication_plan = true;
            options.config.communication_execution_preflight = true;
            continue;
        }
        if (arg == "--execution-cuda-peer-available")
        {
            options.config.communication_plan = true;
            options.config.communication_execution_preflight = true;
            options.config.communication_execution.cuda_peer_available = true;
            continue;
        }
        if (arg == "--execution-inter-node-available")
        {
            options.config.communication_plan = true;
            options.config.communication_execution_preflight = true;
            options.config.communication_execution.inter_node_available = true;
            continue;
        }
        if (arg == "--opcode-summary")
        {
            options.config.opcode_summary = true;
            continue;
        }
        if (arg == "--require-ready")
        {
            options.config.require_ready = true;
            apply_require_ready(options);
            continue;
        }
        if (arg == "--nodes")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --nodes");
            }
            options.config.node_count = parse_int_arg("--nodes", argv[i]);
            options.config.communication_plan = true;
            continue;
        }
        if (arg == "--devices-per-node")
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for --devices-per-node");
            }
            options.config.devices_per_node =
                parse_int_arg("--devices-per-node", argv[i]);
            options.config.communication_plan = true;
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
    apply_require_ready(options);

    if (options.config.pipeline.device_count <= 0)
    {
        throw std::invalid_argument("--devices must be positive");
    }
    if (options.config.pipeline.scheduler.default_device < 0 ||
        options.config.pipeline.scheduler.default_device >=
            options.config.pipeline.device_count)
    {
        throw std::invalid_argument("--default-device must be in [0, devices)");
    }
    if (options.config.pipeline.scheduler.upload_device.has_value() &&
        (*options.config.pipeline.scheduler.upload_device < 0 ||
         *options.config.pipeline.scheduler.upload_device >=
             options.config.pipeline.device_count))
    {
        throw std::invalid_argument("--upload-device must be in [0, devices)");
    }
    if (options.config.pipeline.scheduler.download_device.has_value() &&
        (*options.config.pipeline.scheduler.download_device < 0 ||
         *options.config.pipeline.scheduler.download_device >=
             options.config.pipeline.device_count))
    {
        throw std::invalid_argument("--download-device must be in [0, devices)");
    }
    for (std::size_t i = 0;
         i < options.config.pipeline.scheduler.compute_devices.size(); ++i)
    {
        const int compute_device =
            options.config.pipeline.scheduler.compute_devices[i];
        if (compute_device < 0 ||
            compute_device >= options.config.pipeline.device_count)
        {
            throw std::invalid_argument(
                "--compute-devices entries must be in [0, devices)");
        }
        for (std::size_t j = 0; j < i; ++j)
        {
            if (options.config.pipeline.scheduler.compute_devices[j] ==
                compute_device)
            {
                throw std::invalid_argument(
                    "--compute-devices must not contain duplicate device ids");
            }
        }
    }
    if (options.config.node_count <= 0)
    {
        throw std::invalid_argument("--nodes must be positive");
    }
    if (options.config.devices_per_node < 0)
    {
        throw std::invalid_argument("--devices-per-node must be non-negative");
    }
    if (options.config.communication_plan)
    {
        const int devices_per_node =
            options.config.devices_per_node == 0 ? options.config.pipeline.device_count
                                                 : options.config.devices_per_node;
        if (devices_per_node <= 0)
        {
            throw std::invalid_argument("communication topology devices per node must be positive");
        }
        const std::int64_t total_devices =
            topology_device_count(options.config.node_count, devices_per_node);
        if (total_devices > std::numeric_limits<int>::max())
        {
            throw std::invalid_argument(
                "communication topology device count exceeds logical device id range");
        }
        if (total_devices < options.config.pipeline.device_count)
        {
            throw std::invalid_argument(
                "communication topology has fewer logical devices than --devices");
        }
    }
    return options;
}

StaticScheduleExecutionConfig effective_config(const ToolOptions &tool_options)
{
    StaticScheduleExecutionConfig config = tool_options.config;
    config.pipeline.emit_debug_dump =
        tool_options.dump_schedule || !tool_options.schedule_path.empty();
    return config;
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

void write_text_file(const std::string &path, const std::string &contents)
{
    std::ofstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to create file: " + path);
    }
    stream << contents;
    if (!stream)
    {
        throw std::runtime_error("failed to write file: " + path);
    }
}

StaticSchedulePipelineOptions make_pipeline_options(const ToolOptions &tool_options)
{
    return effective_config(tool_options).pipeline;
}

MgpuTopology make_tool_topology(const ToolOptions &tool_options)
{
    return make_static_schedule_execution_topology(tool_options.config);
}

PoseidonGpuExecutionPreflightOptions make_execution_preflight_options(
    const ToolOptions &tool_options)
{
    PoseidonGpuExecutionPreflightOptions options;
    options.device_count = tool_options.config.pipeline.device_count;
    options.copy_ops_have_comm = tool_options.config.preflight_comm_available;
    options.relin_keys_available =
        tool_options.config.preflight_relin_keys_available;
    options.galois_keys_available =
        tool_options.config.preflight_galois_keys_available;
    options.check_communication_plan = tool_options.config.communication_plan;
    if (options.check_communication_plan)
    {
        options.topology = make_tool_topology(tool_options);
    }
    options.check_communication_execution =
        tool_options.config.communication_execution_preflight;
    options.communication_execution = tool_options.config.communication_execution;
    return options;
}

bool needs_execution_preflight(const ToolOptions &tool_options)
{
    return tool_options.config.poseidon_gpu_preflight ||
           tool_options.config.require_ready;
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
    const std::optional<PoseidonGpuSchedulePreflightResult> &preflight)
{
    if (!preflight.has_value())
    {
        return;
    }

    dump_poseidon_gpu_schedule_preflight(std::cout, *preflight);
}

void print_optional_communication_plan(
    const std::optional<MgpuCommunicationPlan> &plan)
{
    if (!plan.has_value())
    {
        return;
    }

    dump_communication_plan(std::cout, *plan);
}

void print_optional_communication_execution_preflight(
    const std::optional<MgpuCommunicationExecutionPreflight> &preflight)
{
    if (!preflight.has_value())
    {
        return;
    }

    dump_communication_execution_preflight(std::cout, *preflight);
}

void print_optional_execution_preflight(
    const std::optional<PoseidonGpuExecutionPreflightResult> &preflight)
{
    if (!preflight.has_value())
    {
        return;
    }

    dump_poseidon_gpu_execution_preflight(std::cout, *preflight);
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

std::string make_summary_json(
    const ToolOptions &tool_options, const MgpuScheduleSummary &summary,
    std::size_t constant_count, const HevmIoBindingPlan &io_plan,
    const std::optional<PoseidonGpuSchedulePreflightResult> &poseidon_preflight,
    const std::optional<MgpuCommunicationPlan> &communication_plan,
    const std::optional<MgpuCommunicationExecutionPreflight>
        &communication_execution_preflight,
    const std::optional<PoseidonGpuExecutionPreflightResult>
        &execution_preflight,
    const std::optional<DacapoHevmOpcodeSummary> &opcode_summary,
    const std::optional<HevmArtifactReadinessResult> &readiness,
    const std::optional<std::string> &debug_dump)
{
    const StaticScheduleExecutionConfig config = effective_config(tool_options);
    HevmArtifactReportInput report;
    report.hevm_path = tool_options.hevm_path;
    report.constants_path = tool_options.constants_path;
    report.execution_config = &config;
    report.schedule_summary = &summary;
    report.constant_count = constant_count;
    report.io_plan = &io_plan;
    report.poseidon_gpu_preflight =
        poseidon_preflight.has_value() ? &*poseidon_preflight : nullptr;
    report.communication_plan =
        communication_plan.has_value() ? &*communication_plan : nullptr;
    report.communication_execution_preflight =
        communication_execution_preflight.has_value()
            ? &*communication_execution_preflight
            : nullptr;
    report.poseidon_gpu_execution_preflight =
        execution_preflight.has_value() ? &*execution_preflight : nullptr;
    report.hevm_opcode_summary =
        opcode_summary.has_value() ? &*opcode_summary : nullptr;
    report.hevm_artifact_readiness =
        readiness.has_value() ? &*readiness : nullptr;
    report.debug_dump = debug_dump.has_value() ? &*debug_dump : nullptr;
    return hevm_artifact_report_to_json(report, 2);
}

std::string make_artifact_failure_summary_json(
    const ToolOptions &tool_options, const DacapoHevmArtifactResult &artifacts,
    const std::optional<DacapoHevmOpcodeSummary> &opcode_summary,
    const std::optional<HevmArtifactReadinessResult> &readiness)
{
    const StaticScheduleExecutionConfig config = effective_config(tool_options);
    HevmArtifactFailureReportInput input;
    input.hevm_path = tool_options.hevm_path;
    input.constants_path = tool_options.constants_path;
    input.execution_config = &config;
    input.artifacts = &artifacts;
    input.hevm_opcode_summary =
        opcode_summary.has_value() ? &*opcode_summary : nullptr;
    input.hevm_artifact_readiness =
        readiness.has_value() ? &*readiness : nullptr;
    return hevm_artifact_failure_report_to_json(input, 2);
}

DacapoHevmArtifactResult make_hevm_read_failure_result(
    const ToolOptions &tool_options, const std::string &message)
{
    DacapoHevmArtifactResult result;
    result.diagnostics.push_back(DacapoHevmArtifactDiagnostic{
        "read_hevm",
        tool_options.hevm_path,
        0,
        message,
    });
    return result;
}

DacapoHevmArtifactResult make_opcode_summary_failure_result(
    const ToolOptions &tool_options, const DacapoHevmOpcodeSummary &summary)
{
    DacapoHevmArtifactResult result;
    for (const DacapoAdapterDiagnostic &diagnostic : summary.diagnostics)
    {
        result.diagnostics.push_back(DacapoHevmArtifactDiagnostic{
            "dacapo_opcode_summary",
            tool_options.hevm_path,
            diagnostic.offset,
            diagnostic.message,
        });
    }
    return result;
}

void write_optional_failure_summary(
    const ToolOptions &tool_options, const DacapoHevmArtifactResult &artifacts,
    const std::optional<DacapoHevmOpcodeSummary> &opcode_summary,
    const std::optional<HevmArtifactReadinessResult> &readiness)
{
    if (!tool_options.summary_json && tool_options.summary_json_path.empty())
    {
        return;
    }

    const std::string failure_summary_json =
        make_artifact_failure_summary_json(
            tool_options, artifacts, opcode_summary, readiness);
    if (!tool_options.summary_json_path.empty())
    {
        write_text_file(
            tool_options.summary_json_path, failure_summary_json + "\n");
    }
    if (tool_options.summary_json)
    {
        std::cout << failure_summary_json << '\n';
    }
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        const ToolOptions tool_options = parse_args(argc, argv);
        std::optional<DacapoHevmOpcodeSummary> opcode_summary;
        if (tool_options.config.opcode_summary)
        {
            std::string hevm_input;
            try
            {
                hevm_input = read_binary_file(tool_options.hevm_path);
            }
            catch (const std::exception &ex)
            {
                const DacapoHevmArtifactResult artifacts =
                    make_hevm_read_failure_result(tool_options, ex.what());
                write_optional_failure_summary(
                    tool_options, artifacts, std::nullopt, std::nullopt);
                std::cerr << artifacts.format_diagnostics() << '\n';
                return EXIT_FAILURE;
            }
            opcode_summary = summarize_hevm_opcodes(hevm_input);
            if (!opcode_summary->ok())
            {
                const DacapoHevmArtifactResult artifacts =
                    make_opcode_summary_failure_result(
                        tool_options, *opcode_summary);
                write_optional_failure_summary(
                    tool_options, artifacts, opcode_summary, std::nullopt);
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
            if (opcode_summary.has_value() && !tool_options.summary_json)
            {
                print_opcode_summary_text(*opcode_summary);
            }
            std::optional<HevmArtifactReadinessResult> readiness;
            if (tool_options.config.require_ready && opcode_summary.has_value())
            {
                readiness =
                    check_hevm_artifact_readiness(
                        HevmArtifactReadinessInput{ &*opcode_summary });
                dump_hevm_artifact_readiness(std::cerr, *readiness);
            }
            if (tool_options.summary_json || !tool_options.summary_json_path.empty())
            {
                write_optional_failure_summary(
                    tool_options, artifacts, opcode_summary, readiness);
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
            summarize_schedule(
                artifacts.schedule, tool_options.config.pipeline.device_count);

        std::optional<PoseidonGpuExecutionPreflightResult> execution_preflight;
        if (needs_execution_preflight(tool_options))
        {
            execution_preflight = preflight_poseidon_gpu_execution_plan(
                artifacts.schedule, make_execution_preflight_options(tool_options));
        }

        std::optional<PoseidonGpuSchedulePreflightResult> poseidon_preflight;
        if (tool_options.config.poseidon_gpu_preflight)
        {
            poseidon_preflight = execution_preflight.has_value()
                                     ? execution_preflight->poseidon_gpu_preflight
                                     : preflight_poseidon_gpu_schedule(
                                           artifacts.schedule,
                                           PoseidonGpuSchedulePreflightOptions{
                                               tool_options.config.pipeline.device_count,
                                               tool_options.config
                                                   .preflight_comm_available,
                                               tool_options
                                                   .config
                                                   .preflight_relin_keys_available,
                                               tool_options
                                                   .config
                                                   .preflight_galois_keys_available,
                                           });
        }

        std::optional<MgpuCommunicationPlan> communication_plan;
        if (tool_options.config.communication_plan)
        {
            communication_plan =
                execution_preflight.has_value() &&
                        execution_preflight->communication_plan_evaluated
                    ? execution_preflight->communication_plan
                    : plan_schedule_communication(
                          artifacts.schedule, make_tool_topology(tool_options));
        }

        std::optional<MgpuCommunicationExecutionPreflight>
            communication_execution_preflight;
        if (tool_options.config.communication_execution_preflight &&
            communication_plan.has_value() && communication_plan->ok())
        {
            communication_execution_preflight =
                execution_preflight.has_value() &&
                        execution_preflight
                            ->communication_execution_preflight_evaluated
                    ? execution_preflight->communication_execution_preflight
                    : preflight_communication_execution(
                          *communication_plan,
                          tool_options.config.communication_execution);
        }

        std::optional<HevmArtifactReadinessResult> readiness;
        if (tool_options.config.require_ready)
        {
            HevmArtifactReadinessInput readiness_input;
            readiness_input.opcode_summary =
                opcode_summary.has_value() ? &*opcode_summary : nullptr;
            readiness_input.poseidon_gpu_execution_preflight =
                execution_preflight.has_value() ? &*execution_preflight : nullptr;
            readiness = check_hevm_artifact_readiness(readiness_input);
        }

        const std::optional<std::string> debug_dump =
            tool_options.dump_schedule ? std::optional<std::string>(artifacts.debug_dump)
                                       : std::nullopt;
        std::optional<std::string> summary_json;
        if (tool_options.summary_json || !tool_options.summary_json_path.empty())
        {
            summary_json = make_summary_json(
                tool_options, summary, artifacts.constants.values.size(), io_plan.plan,
                poseidon_preflight, communication_plan,
                communication_execution_preflight, execution_preflight,
                opcode_summary, readiness, debug_dump);
        }

        if (!tool_options.summary_json_path.empty())
        {
            write_text_file(tool_options.summary_json_path, *summary_json + "\n");
        }
        if (!tool_options.schedule_path.empty())
        {
            write_text_file(tool_options.schedule_path, artifacts.debug_dump);
        }

        if (tool_options.summary_json)
        {
            std::cout << *summary_json << '\n';
        }
        else
        {
            if (opcode_summary.has_value())
            {
                print_opcode_summary_text(*opcode_summary);
            }
            print_text_summary(summary, artifacts.constants.values.size(), io_plan.plan);
            print_optional_preflight(poseidon_preflight);
            print_optional_communication_plan(communication_plan);
            print_optional_communication_execution_preflight(
                communication_execution_preflight);
            print_optional_execution_preflight(execution_preflight);
            if (readiness.has_value())
            {
                dump_hevm_artifact_readiness(std::cout, *readiness);
            }

            if (tool_options.dump_schedule)
            {
                std::cout << "\n" << artifacts.debug_dump;
            }
        }

        if (readiness.has_value() && !readiness->ok())
        {
            return EXIT_FAILURE;
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
