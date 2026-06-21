#include "poseidon/mgpu/compiler/dacapo_artifacts.h"
#include "poseidon/mgpu/comm/topology.h"
#include "poseidon/mgpu/ir/schedule_summary.h"
#include "poseidon/mgpu/runtime/hevm_io_binding.h"
#include "poseidon/mgpu/runtime/poseidon_gpu_schedule_preflight.h"

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

constexpr int kSkip = 77;

const char *get_env(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr || std::string(value).empty())
    {
        return nullptr;
    }
    return value;
}

int parse_int(const char *name, const char *value)
{
    if (value == nullptr)
    {
        throw std::invalid_argument(std::string("missing value for ") + name);
    }

    std::size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    if (consumed != std::string(value).size())
    {
        throw std::invalid_argument(std::string("invalid integer for ") + name + ": " + value);
    }
    return parsed;
}

bool parse_bool(const char *value)
{
    if (value == nullptr)
    {
        return false;
    }

    const std::string text = value;
    return text == "1" || text == "ON" || text == "on" || text == "true" || text == "TRUE";
}

std::optional<int> parse_optional_device(const char *name)
{
    const char *value = get_env(name);
    if (value == nullptr)
    {
        return std::nullopt;
    }
    return parse_int(name, value);
}

std::vector<int> parse_device_list(const char *name)
{
    const char *value = get_env(name);
    if (value == nullptr)
    {
        return {};
    }

    const std::string text = value;
    std::vector<int> devices;
    std::size_t begin = 0;
    while (begin <= text.size())
    {
        const std::size_t end = text.find(',', begin);
        const std::string item = text.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (item.empty())
        {
            throw std::invalid_argument(std::string("empty device id in ") + name);
        }
        devices.push_back(parse_int(name, item.c_str()));
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return devices;
}

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

std::string read_binary_file(const std::string &path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("failed to open external HEVM file: " + path);
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (stream.bad())
    {
        throw std::runtime_error("failed to read external HEVM file: " + path);
    }
    return buffer.str();
}

void validate_constant_indices(
    const HevmIoBindingPlan &plan, const DacapoConstantTable &constants)
{
    for (const HevmPlainInputSlot &slot : plan.plain_inputs)
    {
        if (slot.constant_index >= constants.values.size())
        {
            throw std::runtime_error(
                "HEVM plaintext constant index " + std::to_string(slot.constant_index) +
                " is out of range for " + std::to_string(constants.values.size()) +
                " parsed constants");
        }
    }
}

void print_opcode_summary(const DacapoHevmOpcodeSummary &summary)
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
}

StaticSchedulePipelineOptions make_pipeline_options()
{
    StaticSchedulePipelineOptions options;
    if (const char *device_count = get_env("POSEIDON_MGPU_EXTERNAL_DEVICE_COUNT"))
    {
        options.device_count = parse_int("POSEIDON_MGPU_EXTERNAL_DEVICE_COUNT", device_count);
    }
    if (const char *default_device = get_env("POSEIDON_MGPU_EXTERNAL_DEFAULT_DEVICE"))
    {
        options.placement.default_device =
            parse_int("POSEIDON_MGPU_EXTERNAL_DEFAULT_DEVICE", default_device);
    }

    options.placement.upload_device =
        parse_optional_device("POSEIDON_MGPU_EXTERNAL_UPLOAD_DEVICE");
    options.placement.download_device =
        parse_optional_device("POSEIDON_MGPU_EXTERNAL_DOWNLOAD_DEVICE");
    options.placement.compute_devices =
        parse_device_list("POSEIDON_MGPU_EXTERNAL_COMPUTE_DEVICES");
    if (!options.placement.compute_devices.empty() ||
        parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_ROUND_ROBIN_COMPUTE")))
    {
        options.placement.policy = StaticPlacementPolicy::RoundRobinCompute;
    }
    options.emit_debug_dump = parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_DEBUG_DUMP"));
    return options;
}

void print_io_summary(const HevmIoBindingPlan &plan)
{
    std::cout << "hevm_io:\n";
    std::cout << "  cipher_inputs: " << plan.cipher_inputs.size() << '\n';
    std::cout << "  plaintext_constants: " << plan.plain_inputs.size() << '\n';
    std::cout << "  results: " << plan.results.size() << '\n';
}

}  // namespace

int main()
{
    try
    {
        const char *hevm_path = get_env("POSEIDON_MGPU_EXTERNAL_HEVM");
        const char *constants_path = get_env("POSEIDON_MGPU_EXTERNAL_CST");
        if (hevm_path == nullptr && constants_path == nullptr)
        {
            std::cout << "skipping external HEVM artifact test; set "
                      << "POSEIDON_MGPU_EXTERNAL_HEVM and POSEIDON_MGPU_EXTERNAL_CST\n";
            return kSkip;
        }
        if (hevm_path == nullptr || constants_path == nullptr)
        {
            throw std::invalid_argument(
                "POSEIDON_MGPU_EXTERNAL_HEVM and POSEIDON_MGPU_EXTERNAL_CST must be set together");
        }

        const StaticSchedulePipelineOptions options = make_pipeline_options();
        require(options.device_count > 0, "POSEIDON_MGPU_EXTERNAL_DEVICE_COUNT must be positive");

        const DacapoHevmOpcodeSummary opcode_summary =
            summarize_hevm_opcodes(read_binary_file(hevm_path));
        require(
            opcode_summary.ok(),
            "external HEVM opcode summary failed:\n" + opcode_summary.format_diagnostics());
        print_opcode_summary(opcode_summary);

        const DacapoHevmArtifactResult artifacts =
            prepare_dacapo_hevm_artifacts_from_files(
                DacapoHevmArtifactPaths{ hevm_path, constants_path }, options);
        require(artifacts.ok(), "external HEVM artifact load failed:\n" +
                                    artifacts.format_diagnostics());
        require(!artifacts.schedule.ops.empty(), "external HEVM schedule must not be empty");

        const HevmIoBindingPlanResult io_plan =
            build_hevm_io_binding_plan(artifacts.schedule);
        require(io_plan.ok(), "external HEVM IO plan failed:\n" +
                                  io_plan.format_diagnostics());
        require(!io_plan.plan.cipher_inputs.empty(), "external HEVM must have ciphertext inputs");
        require(!io_plan.plan.results.empty(), "external HEVM must have results");
        validate_constant_indices(io_plan.plan, artifacts.constants);

        const MgpuScheduleSummary summary =
            summarize_schedule(artifacts.schedule, options.device_count);
        require(summary.invalid_device_ops == 0, "external HEVM schedule has invalid devices");
        require(
            summary.unassigned_device_ops == 0,
            "external HEVM schedule has unassigned devices after placement");

        std::cout << "external_hevm: " << hevm_path << '\n';
        std::cout << "external_constants: " << constants_path << '\n';
        std::cout << "constants: " << artifacts.constants.values.size() << '\n';
        print_io_summary(io_plan.plan);
        std::cout << dump_schedule_summary(summary);
        const PoseidonGpuSchedulePreflightResult preflight =
            preflight_poseidon_gpu_schedule(
                artifacts.schedule,
                PoseidonGpuSchedulePreflightOptions{
                    options.device_count,
                    /*copy_ops_have_comm=*/true,
                    /*relin_keys_available=*/true,
                    /*galois_keys_available=*/true,
        });
        std::cout << dump_poseidon_gpu_schedule_preflight(preflight);

        const MgpuCommunicationPlan communication_plan =
            plan_schedule_communication(
                artifacts.schedule, make_single_node_topology(options.device_count));
        require(
            communication_plan.ok(),
            "external HEVM communication plan failed:\n" +
                communication_plan.format_diagnostics());
        std::cout << dump_communication_plan(communication_plan);
        if (!artifacts.debug_dump.empty())
        {
            std::cout << '\n' << artifacts.debug_dump;
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu external HEVM artifact test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
