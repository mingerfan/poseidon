#include "poseidon/mgpu/compiler/dacapo_artifacts.h"
#include "poseidon/mgpu/compiler/static_schedule_config.h"
#include "poseidon/mgpu/comm/execution_preflight.h"
#include "poseidon/mgpu/comm/topology.h"
#include "poseidon/mgpu/ir/schedule_summary.h"
#include "poseidon/mgpu/runtime/gpu_execution_preflight.h"
#include "poseidon/mgpu/runtime/hevm_artifact_report.h"
#include "poseidon/mgpu/runtime/hevm_artifact_readiness.h"
#include "poseidon/mgpu/runtime/hevm_io_binding.h"
#include "poseidon/mgpu/runtime/poseidon_gpu_schedule_preflight.h"
#include "poseidon/tests/mgpu/hevm_test_utils.h"
#include "poseidon/util/json.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using namespace poseidon::mgpu;

namespace
{

constexpr int kSkip = 77;
using Json = nlohmann::json;

MgpuTopology make_external_topology(const StaticScheduleExecutionConfig &config);

class TempDir
{
public:
    TempDir()
    {
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("poseidon_mgpu_external_hevm_artifact_test_" + std::to_string(tick));
        std::filesystem::create_directories(path_);
    }

    TempDir(const TempDir &) = delete;
    TempDir &operator=(const TempDir &) = delete;

    ~TempDir()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    std::string path(const std::string &name) const
    {
        return (path_ / name).string();
    }

private:
    std::filesystem::path path_;
};

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

bool should_expect_inter_node_missing()
{
    return parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_EXPECT_INTER_NODE_MISSING"));
}

bool should_expect_artifact_failure_report()
{
    return parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_EXPECT_ARTIFACT_FAILURE_REPORT"));
}

bool should_expect_missing_hevm_failure_report()
{
    return parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_EXPECT_MISSING_HEVM_REPORT"));
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

void append_i64(std::string &output, std::int64_t value)
{
    const auto bits = static_cast<std::uint64_t>(value);
    for (int i = 0; i < 8; ++i)
    {
        output.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
    }
}

void append_double(std::string &output, double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 8; ++i)
    {
        output.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
    }
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

void write_binary_file(const std::string &path, const std::string &contents)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("failed to create mock external HEVM artifact: " + path);
    }
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!stream)
    {
        throw std::runtime_error("failed to write mock external HEVM artifact: " + path);
    }
}

void write_text_file(const std::string &path, const std::string &contents)
{
    std::ofstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to create external HEVM report: " + path);
    }
    stream << contents;
    if (!stream)
    {
        throw std::runtime_error("failed to write external HEVM report: " + path);
    }
}

std::string read_text_file(const std::string &path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open external HEVM output: " + path);
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (stream.bad())
    {
        throw std::runtime_error("failed to read external HEVM output: " + path);
    }
    return buffer.str();
}

std::string make_mock_hevm_binary()
{
    return test::make_hevm_binary(
        1, 1, 2, 1, { 1 },
        {
            test::HevmOpRecord{ 0, 0, 0, test::make_hevm_encode_attr(2, 20) },
            test::HevmOpRecord{ 9, 1, 0, 0 },
        },
        test::HevmConfigMetadata{
            { 20 },
            { 2 },
            { 40 },
            { 2 },
            2,
        });
}

std::string make_rich_mock_hevm_binary()
{
    return test::make_hevm_binary(
        2, 1, 6, 2, { 5 },
        {
            test::HevmOpRecord{ 0, 0, 0, test::make_hevm_encode_attr(3, 20) },
            test::HevmOpRecord{ 0, 1, 1, test::make_hevm_encode_attr(3, 20) },
            test::HevmOpRecord{ 1, 2, 0, 1 },
            test::HevmOpRecord{ 6, 3, 2, 1 },
            test::HevmOpRecord{ 8, 4, 3, 1 },
            test::HevmOpRecord{ 3, 4, 4, 0 },
            test::HevmOpRecord{ 7, 5, 4, 0 },
        },
        test::HevmConfigMetadata{
            { 20, 20 },
            { 3, 3 },
            { 20 },
            { 2 },
            3,
        });
}

std::string make_unsupported_mock_hevm_binary()
{
    return test::make_hevm_binary(
        1, 1, 2, 1, { 1 },
        {
            test::HevmOpRecord{ 0, 0, 0, test::make_hevm_encode_attr(2, 20) },
            test::HevmOpRecord{ 4, 1, 0, 1 },
        },
        test::HevmConfigMetadata{
            { 20 },
            { 2 },
            { 40 },
            { 2 },
            2,
        });
}

std::string make_mock_constant_file()
{
    std::string output;
    append_i64(output, 1);
    append_i64(output, 4);
    append_double(output, 0.5);
    append_double(output, -1.0);
    append_double(output, 2.0);
    append_double(output, 3.5);
    return output;
}

std::string make_rich_mock_constant_file()
{
    std::string output;
    append_i64(output, 2);
    append_i64(output, 4);
    append_double(output, 0.5);
    append_double(output, -1.0);
    append_double(output, 2.0);
    append_double(output, 3.5);
    append_i64(output, 4);
    append_double(output, 1.25);
    append_double(output, 0.25);
    append_double(output, -0.75);
    append_double(output, 4.0);
    return output;
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

void validate_report_json_file(
    const std::string &path, const StaticScheduleExecutionConfig &config,
    const MgpuScheduleSummary &summary, const HevmIoBindingPlan &io_plan,
    const HevmArtifactReadinessResult &readiness)
{
    const Json report = Json::parse(read_text_file(path));
    require(report.at("version").get<int>() == 1, "external report version mismatch");
    require(
        report.at("execution_gate").at("ok").get<bool>() == readiness.ok(),
        "external report execution gate does not match readiness");
    require(
        report.at("execution_gate").at("status").get<std::string>() ==
            (readiness.ok() ? "ready" : "not_ready"),
        "external report execution gate status mismatch");
    if (readiness.ok())
    {
        require(
            report.at("execution_gate").at("diagnostics").empty(),
            "ready external report should not include gate diagnostics");
    }
    else
    {
        require(
            !report.at("execution_gate").at("diagnostics").empty(),
            "not-ready external report should include gate diagnostics");
        require(
            report.at("execution_gate")
                    .at("diagnostics")
                    .at(0)
                    .at("message")
                    .get<std::string>()
                    .find(readiness.diagnostics.at(0).message) != std::string::npos,
            "external report gate diagnostic should mirror readiness diagnostics");
    }
    require(
        report.at("execution_config").at("device_count").get<int>() ==
            config.pipeline.device_count,
        "external report device_count mismatch");
    require(
        report.at("schedule").at("total_ops").get<std::size_t>() == summary.total_ops,
        "external report schedule total_ops mismatch");
    require(
        report.at("hevm_io").at("cipher_inputs").get<std::size_t>() ==
            io_plan.cipher_inputs.size(),
        "external report cipher input count mismatch");
    require(
        report.at("hevm_io").at("results").get<std::size_t>() ==
            io_plan.results.size(),
        "external report result count mismatch");
    require(
        report.at("poseidon_gpu_execution_preflight").at("ok").is_boolean(),
        "external report missing aggregate execution preflight");
    require(
        report.at("hevm_artifact_readiness").at("ok").get<bool>() == readiness.ok(),
        "external report readiness mismatch");
}

void validate_failure_report_json_file(
    const std::string &path, const StaticScheduleExecutionConfig &config,
    const DacapoHevmArtifactResult &artifacts,
    const DacapoHevmOpcodeSummary &opcode_summary,
    const std::optional<HevmArtifactReadinessResult> &readiness)
{
    const Json report = Json::parse(read_text_file(path));
    require(report.at("version").get<int>() == 1, "failure report version mismatch");
    require(
        !report.at("execution_gate").at("ok").get<bool>(),
        "failure report execution gate should be not-ready");
    require(
        report.at("execution_gate").at("status").get<std::string>() ==
            "not_ready",
        "failure report execution gate status mismatch");
    require(
        !report.at("execution_gate")
             .at("checks")
             .at("schedule_built")
             .get<bool>(),
        "failure report should mark schedule_built false");
    require(
        report.at("execution_gate").at("diagnostics").size() >=
            artifacts.diagnostics.size(),
        "failure report should include gate diagnostics");
    require(
        report.at("execution_config").at("device_count").get<int>() ==
            config.pipeline.device_count,
        "failure report device_count mismatch");
    require(
        report.at("artifact_diagnostics").size() == artifacts.diagnostics.size(),
        "failure report artifact diagnostic count mismatch");
    require(
        !artifacts.diagnostics.empty(),
        "test failure artifact should include diagnostics");
    require(
        report.at("artifact_diagnostics")
                .at(0)
                .at("message")
                .get<std::string>()
                .find(artifacts.diagnostics.at(0).message) != std::string::npos,
        "failure report artifact diagnostic message mismatch");
    require(
        report.at("hevm_opcode_summary").at("operation_count").get<std::uint64_t>() ==
            opcode_summary.operation_count,
        "failure report opcode operation count mismatch");
    if (readiness.has_value())
    {
        require(
            report.at("hevm_artifact_readiness").at("ok").get<bool>() ==
                readiness->ok(),
            "failure report readiness mismatch");
    }
}

void write_and_validate_failure_report(
    const std::string &path, const std::string &hevm_path,
    const std::string &constants_path, const StaticScheduleExecutionConfig &config,
    const DacapoHevmArtifactResult &artifacts,
    const DacapoHevmOpcodeSummary *opcode_summary,
    const std::optional<HevmArtifactReadinessResult> &readiness)
{
    HevmArtifactFailureReportInput report;
    report.hevm_path = hevm_path;
    report.constants_path = constants_path;
    report.execution_config = &config;
    report.artifacts = &artifacts;
    report.hevm_opcode_summary = opcode_summary;
    report.hevm_artifact_readiness =
        readiness.has_value() ? &*readiness : nullptr;
    write_text_file(
        path, hevm_artifact_failure_report_to_json(report, 2) + "\n");
    if (opcode_summary != nullptr)
    {
        validate_failure_report_json_file(
            path, config, artifacts, *opcode_summary, readiness);
    }
    else
    {
        const Json report_json = Json::parse(read_text_file(path));
        require(
            !report_json.at("execution_gate").at("ok").get<bool>(),
            "missing HEVM report execution gate should be not-ready");
        require(
            !report_json.at("execution_gate")
                 .at("checks")
                 .at("artifacts_loaded")
                 .get<bool>(),
            "missing HEVM report should mark artifacts_loaded false");
        require(
            report_json.at("artifact_diagnostics")
                    .at(0)
                    .at("stage")
                    .get<std::string>() == "read_hevm",
            "missing HEVM report stage mismatch");
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
    if (get_env("POSEIDON_MGPU_EXTERNAL_SCHEDULE_DUMP") != nullptr)
    {
        options.emit_debug_dump = true;
    }
    return options;
}

void apply_require_ready(StaticScheduleExecutionConfig &config)
{
    if (!config.require_ready)
    {
        return;
    }
    config.opcode_summary = true;
    config.poseidon_gpu_preflight = true;
    config.communication_plan = true;
    config.communication_execution_preflight = true;
}

StaticScheduleExecutionConfig parse_external_config_text(
    const std::string &text, const std::string &source)
{
    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(text);
    if (!result.ok())
    {
        throw std::runtime_error(
            "external HEVM config failed to parse from " + source + ":\n" +
            result.format_diagnostics());
    }
    return result.config;
}

StaticScheduleExecutionConfig make_external_config()
{
    const char *config_path = get_env("POSEIDON_MGPU_EXTERNAL_CONFIG");
    const char *config_json = get_env("POSEIDON_MGPU_EXTERNAL_CONFIG_JSON");
    if (config_path != nullptr && config_json != nullptr)
    {
        throw std::invalid_argument(
            "POSEIDON_MGPU_EXTERNAL_CONFIG and POSEIDON_MGPU_EXTERNAL_CONFIG_JSON cannot both be set");
    }
    if (config_path != nullptr)
    {
        return parse_external_config_text(read_binary_file(config_path), config_path);
    }
    if (config_json != nullptr)
    {
        return parse_external_config_text(config_json, "POSEIDON_MGPU_EXTERNAL_CONFIG_JSON");
    }

    StaticScheduleExecutionConfig config;
    config.pipeline = make_pipeline_options();
    config.preflight_comm_available =
        parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_PREFLIGHT_COMM_AVAILABLE"));
    config.preflight_relin_keys_available =
        parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_PREFLIGHT_RELIN_KEYS"));
    config.preflight_galois_keys_available =
        parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_PREFLIGHT_GALOIS_KEYS"));
    config.communication_plan = true;
    config.communication_execution_preflight = true;
    config.communication_execution.same_device_available = true;
    config.communication_execution.cuda_peer_available =
        parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_EXECUTION_CUDA_PEER_AVAILABLE"));
    config.communication_execution.inter_node_available =
        parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_EXECUTION_INTER_NODE_AVAILABLE"));
    config.require_ready =
        parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_REQUIRE_READY"));
    apply_require_ready(config);

    if (const char *nodes = get_env("POSEIDON_MGPU_EXTERNAL_NODES"))
    {
        config.node_count = parse_int("POSEIDON_MGPU_EXTERNAL_NODES", nodes);
    }
    if (const char *devices = get_env("POSEIDON_MGPU_EXTERNAL_DEVICES_PER_NODE"))
    {
        config.devices_per_node =
            parse_int("POSEIDON_MGPU_EXTERNAL_DEVICES_PER_NODE", devices);
    }
    return config;
}

void print_io_summary(const HevmIoBindingPlan &plan)
{
    std::cout << "hevm_io:\n";
    std::cout << "  cipher_inputs: " << plan.cipher_inputs.size() << '\n';
    std::cout << "  plaintext_constants: " << plan.plain_inputs.size() << '\n';
    std::cout << "  results: " << plan.results.size() << '\n';
}

std::size_t op_count_for_kind(const MgpuScheduleSummary &summary, MgpuOpKind kind)
{
    for (const MgpuOpKindCount &count : summary.op_counts)
    {
        if (count.kind == kind)
        {
            return count.count;
        }
    }
    return 0;
}

PoseidonGpuSchedulePreflightOptions make_preflight_options(
    const StaticScheduleExecutionConfig &config)
{
    return PoseidonGpuSchedulePreflightOptions{
        config.pipeline.device_count,
        config.preflight_comm_available,
        config.preflight_relin_keys_available,
        config.preflight_galois_keys_available,
    };
}

PoseidonGpuExecutionPreflightOptions make_execution_preflight_options(
    const StaticScheduleExecutionConfig &config)
{
    PoseidonGpuExecutionPreflightOptions preflight_options;
    const PoseidonGpuSchedulePreflightOptions gpu_options =
        make_preflight_options(config);
    preflight_options.device_count = gpu_options.device_count;
    preflight_options.copy_ops_have_comm = gpu_options.copy_ops_have_comm;
    preflight_options.relin_keys_available = gpu_options.relin_keys_available;
    preflight_options.galois_keys_available = gpu_options.galois_keys_available;
    preflight_options.check_communication_plan = true;
    preflight_options.topology = make_external_topology(config);
    preflight_options.check_communication_execution = true;
    preflight_options.communication_execution = config.communication_execution;
    return preflight_options;
}

MgpuTopology make_external_topology(const StaticScheduleExecutionConfig &config)
{
    int devices_per_node = config.devices_per_node;
    require(config.node_count > 0, "POSEIDON_MGPU_EXTERNAL_NODES must be positive");
    require(
        devices_per_node >= 0,
        "POSEIDON_MGPU_EXTERNAL_DEVICES_PER_NODE must be non-negative");

    if (config.node_count == 1 && devices_per_node == 0)
    {
        return make_single_node_topology(config.pipeline.device_count);
    }

    if (devices_per_node == 0)
    {
        devices_per_node = config.pipeline.device_count;
    }
    require(
        config.node_count * devices_per_node >= config.pipeline.device_count,
        "external HEVM topology has fewer logical devices than device_count");
    return make_uniform_cluster_topology(config.node_count, devices_per_node);
}

}  // namespace

int main()
{
    try
    {
        const bool use_mock_artifact =
            parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_MOCK_ARTIFACT"));
        const bool use_rich_mock_artifact =
            parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_RICH_MOCK_ARTIFACT"));
        const bool use_unsupported_mock_artifact =
            parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_UNSUPPORTED_MOCK_ARTIFACT"));
        const bool use_missing_hevm_mock_artifact =
            parse_bool(get_env("POSEIDON_MGPU_EXTERNAL_MISSING_HEVM_MOCK_ARTIFACT"));
        std::optional<TempDir> mock_temp;
        std::string hevm_path = get_env("POSEIDON_MGPU_EXTERNAL_HEVM") == nullptr
                                    ? ""
                                    : get_env("POSEIDON_MGPU_EXTERNAL_HEVM");
        std::string constants_path = get_env("POSEIDON_MGPU_EXTERNAL_CST") == nullptr
                                         ? ""
                                         : get_env("POSEIDON_MGPU_EXTERNAL_CST");

        if (use_mock_artifact || use_rich_mock_artifact ||
            use_unsupported_mock_artifact || use_missing_hevm_mock_artifact)
        {
            mock_temp.emplace();
            hevm_path = mock_temp->path("mock_resnet20.hevm");
            constants_path = mock_temp->path("mock_resnet20.cst");
            if (!use_missing_hevm_mock_artifact)
            {
                write_binary_file(
                    hevm_path,
                    use_unsupported_mock_artifact
                        ? make_unsupported_mock_hevm_binary()
                        : (use_rich_mock_artifact ? make_rich_mock_hevm_binary()
                                                  : make_mock_hevm_binary()));
            }
            write_binary_file(
                constants_path,
                use_rich_mock_artifact ? make_rich_mock_constant_file()
                                       : make_mock_constant_file());
        }

        if (hevm_path.empty() && constants_path.empty())
        {
            std::cout << "skipping external HEVM artifact test; set "
                      << "POSEIDON_MGPU_EXTERNAL_HEVM and POSEIDON_MGPU_EXTERNAL_CST "
                      << "or POSEIDON_MGPU_EXTERNAL_MOCK_ARTIFACT=1 or "
                      << "POSEIDON_MGPU_EXTERNAL_RICH_MOCK_ARTIFACT=1 or "
                      << "POSEIDON_MGPU_EXTERNAL_UNSUPPORTED_MOCK_ARTIFACT=1 or "
                      << "POSEIDON_MGPU_EXTERNAL_MISSING_HEVM_MOCK_ARTIFACT=1\n";
            return kSkip;
        }
        if (hevm_path.empty() || constants_path.empty())
        {
            throw std::invalid_argument(
                "POSEIDON_MGPU_EXTERNAL_HEVM and POSEIDON_MGPU_EXTERNAL_CST must be set together");
        }

        const StaticScheduleExecutionConfig config = make_external_config();
        require(
            config.pipeline.device_count > 0,
            "POSEIDON_MGPU_EXTERNAL_DEVICE_COUNT must be positive");

        std::string hevm_input;
        try
        {
            hevm_input = read_binary_file(hevm_path);
        }
        catch (const std::exception &ex)
        {
            DacapoHevmArtifactResult artifacts;
            artifacts.diagnostics.push_back(DacapoHevmArtifactDiagnostic{
                "read_hevm",
                hevm_path,
                0,
                ex.what(),
            });
            if (const char *report_path = get_env("POSEIDON_MGPU_EXTERNAL_REPORT_JSON"))
            {
                write_and_validate_failure_report(
                    report_path, hevm_path, constants_path, config, artifacts,
                    nullptr, std::nullopt);
                std::cout << "external_failure_report_json: " << report_path
                          << '\n';
            }
            if (should_expect_missing_hevm_failure_report())
            {
                return EXIT_SUCCESS;
            }
            throw;
        }

        const DacapoHevmOpcodeSummary opcode_summary =
            summarize_hevm_opcodes(hevm_input);
        require(
            opcode_summary.ok(),
            "external HEVM opcode summary failed:\n" + opcode_summary.format_diagnostics());
        print_opcode_summary(opcode_summary);

        const DacapoHevmArtifactResult artifacts =
            prepare_dacapo_hevm_artifacts_from_files(
                DacapoHevmArtifactPaths{ hevm_path, constants_path },
                config.pipeline);
        if (!artifacts.ok())
        {
            std::optional<HevmArtifactReadinessResult> failure_readiness;
            if (config.require_ready)
            {
                failure_readiness = check_hevm_artifact_readiness(
                    HevmArtifactReadinessInput{ &opcode_summary });
                std::cout << dump_hevm_artifact_readiness(*failure_readiness);
            }
            if (const char *report_path = get_env("POSEIDON_MGPU_EXTERNAL_REPORT_JSON"))
            {
                write_and_validate_failure_report(
                    report_path, hevm_path, constants_path, config, artifacts,
                    &opcode_summary, failure_readiness);
                std::cout << "external_failure_report_json: " << report_path
                          << '\n';
            }
            if (should_expect_artifact_failure_report())
            {
                return EXIT_SUCCESS;
            }
        }
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
            summarize_schedule(artifacts.schedule, config.pipeline.device_count);
        require(summary.invalid_device_ops == 0, "external HEVM schedule has invalid devices");
        require(
            summary.unassigned_device_ops == 0,
            "external HEVM schedule has unassigned devices after placement");
        if (use_rich_mock_artifact)
        {
            require(
                op_count_for_kind(summary, MgpuOpKind::Rotate) == 1,
                "rich mock artifact should include one rotate");
            require(
                op_count_for_kind(summary, MgpuOpKind::Multiply) == 1,
                "rich mock artifact should include one ciphertext multiply");
            require(
                op_count_for_kind(summary, MgpuOpKind::Add) == 1,
                "rich mock artifact should include one ciphertext add");
            require(
                op_count_for_kind(summary, MgpuOpKind::Rescale) == 1,
                "rich mock artifact should include one rescale");
            require(
                summary.copy_ops > 0,
                "rich mock artifact placement should exercise explicit copy insertion");
        }

        std::cout << "external_hevm: " << hevm_path << '\n';
        std::cout << "external_constants: " << constants_path << '\n';
        std::cout << "constants: " << artifacts.constants.values.size() << '\n';
        print_io_summary(io_plan.plan);
        std::cout << dump_schedule_summary(summary);
        const PoseidonGpuExecutionPreflightResult execution_preflight =
            preflight_poseidon_gpu_execution_plan(
                artifacts.schedule, make_execution_preflight_options(config));
        require(
            execution_preflight.schedule_verification.ok(),
            "external HEVM execution schedule verification failed:\n" +
                execution_preflight.schedule_verification.format_errors());

        const PoseidonGpuSchedulePreflightResult &preflight =
            execution_preflight.poseidon_gpu_preflight;
        std::cout << dump_poseidon_gpu_schedule_preflight(preflight);
        if (use_rich_mock_artifact)
        {
            require(
                preflight.requires_comm,
                "rich mock artifact should require the communication layer");
            require(
                preflight.requires_galois_keys,
                "rich mock artifact should require GaloisKeys");
        }

        require(
            execution_preflight.communication_plan_evaluated,
            "external HEVM execution preflight should include a communication plan");
        const MgpuCommunicationPlan &communication_plan =
            execution_preflight.communication_plan;
        require(
            communication_plan.ok(),
            "external HEVM communication plan failed:\n" +
                communication_plan.format_diagnostics());
        std::cout << dump_communication_plan(communication_plan);
        require(
            execution_preflight.communication_execution_preflight_evaluated,
            "external HEVM execution preflight should include communication execution");
        const MgpuCommunicationExecutionPreflight &communication_preflight =
            execution_preflight.communication_execution_preflight;
        std::cout << dump_communication_execution_preflight(communication_preflight);
        std::cout << dump_poseidon_gpu_execution_preflight(execution_preflight);

        HevmArtifactReadinessInput readiness_input;
        readiness_input.opcode_summary = &opcode_summary;
        readiness_input.poseidon_gpu_execution_preflight = &execution_preflight;
        const HevmArtifactReadinessResult readiness =
            check_hevm_artifact_readiness(readiness_input);
        std::cout << dump_hevm_artifact_readiness(readiness);

        if (should_expect_inter_node_missing())
        {
            require(
                communication_plan.inter_node_copies > 0,
                "expected cluster preview to produce inter-node copy routes");
            require(
                communication_preflight.inter_node_routes > 0,
                "expected cluster preview to preflight inter-node routes");
            require(
                !communication_preflight.ok(),
                "expected cluster preview to report missing inter-node backend");
            require(
                !readiness.ok(),
                "expected cluster preview readiness to remain not-ready");
            require(
                readiness.format_diagnostics().find(
                    "inter-node communication backend is not available") !=
                    std::string::npos,
                "expected readiness diagnostics to mention missing inter-node backend");
        }

        if (const char *report_path = get_env("POSEIDON_MGPU_EXTERNAL_REPORT_JSON"))
        {
            const std::string *debug_dump =
                artifacts.debug_dump.empty() ? nullptr : &artifacts.debug_dump;
            HevmArtifactReportInput report;
            report.hevm_path = hevm_path;
            report.constants_path = constants_path;
            report.execution_config = &config;
            report.schedule_summary = &summary;
            report.constant_count = artifacts.constants.values.size();
            report.io_plan = &io_plan.plan;
            report.poseidon_gpu_preflight = &preflight;
            report.communication_plan = &communication_plan;
            report.communication_execution_preflight = &communication_preflight;
            report.poseidon_gpu_execution_preflight = &execution_preflight;
            report.hevm_opcode_summary = &opcode_summary;
            report.hevm_artifact_readiness = &readiness;
            report.debug_dump = debug_dump;
            write_text_file(
                report_path, hevm_artifact_report_to_json(report, 2) + "\n");
            validate_report_json_file(
                report_path, config, summary, io_plan.plan, readiness);
            std::cout << "external_report_json: " << report_path << '\n';
        }

        if (const char *schedule_path = get_env("POSEIDON_MGPU_EXTERNAL_SCHEDULE_DUMP"))
        {
            write_text_file(schedule_path, artifacts.debug_dump);
            require(
                read_text_file(schedule_path).find("mgpu.schedule") !=
                    std::string::npos,
                "external HEVM schedule dump should contain mgpu.schedule");
            std::cout << "external_schedule_dump: " << schedule_path << '\n';
        }

        if (config.require_ready)
        {
            require(
                readiness.ok(),
                "external HEVM artifact is not ready for execution:\n" +
                    readiness.format_diagnostics());
        }

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
