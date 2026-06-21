#include "poseidon/mgpu/compiler/dacapo_artifacts.h"
#include "poseidon/mgpu/ir/schedule_summary.h"
#include "poseidon/mgpu/runtime/hevm_io_binding.h"
#include "poseidon/mgpu/runtime/poseidon_gpu_schedule_preflight.h"
#include "poseidon/util/json.h"

#include <cstdlib>
#include <iostream>
#include <optional>
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
    return options;
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
        const DacapoHevmArtifactResult artifacts =
            prepare_dacapo_hevm_artifacts_from_files(
                DacapoHevmArtifactPaths{
                    tool_options.hevm_path,
                    tool_options.constants_path,
                },
                make_pipeline_options(tool_options));
        if (!artifacts.ok())
        {
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
            std::cout << make_tool_summary_json(
                             summary, artifacts.constants.values.size(), io_plan.plan,
                             std::move(preflight), debug_dump)
                             .dump(2)
                      << '\n';
        }
        else
        {
            print_text_summary(summary, artifacts.constants.values.size(), io_plan.plan);
            print_optional_preflight(tool_options, artifacts.schedule);

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
