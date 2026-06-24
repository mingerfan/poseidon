#include "poseidon/tests/frontends/dacapo/hevm_test_utils.h"

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

namespace
{

namespace fs = std::filesystem;
constexpr int kSkip = 77;

class TempDir
{
public:
    TempDir()
    {
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() /
                ("poseidon_mgpu_resnet20_dump_preflight_test_" +
                 std::to_string(tick));
        fs::create_directories(path_);
    }

    TempDir(const TempDir &) = delete;
    TempDir &operator=(const TempDir &) = delete;

    ~TempDir()
    {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    fs::path root() const
    {
        return path_;
    }

    std::string path(const std::string &name) const
    {
        return (path_ / name).string();
    }

private:
    fs::path path_;
};

void require_contains(const std::string &text, const std::string &needle);

const char *get_env(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr || std::string(value).empty())
    {
        return nullptr;
    }
    return value;
}

bool parse_bool(const char *value)
{
    if (value == nullptr)
    {
        return false;
    }

    const std::string text = value;
    return text == "1" || text == "ON" || text == "on" || text == "true" ||
           text == "TRUE";
}

bool should_expect_missing_hevm_failure_report()
{
    return parse_bool(
        get_env("POSEIDON_MGPU_RESNET20_EXPECT_MISSING_HEVM_REPORT"));
}

bool should_expect_missing_constants_failure_report()
{
    return parse_bool(
        get_env("POSEIDON_MGPU_RESNET20_EXPECT_MISSING_CONSTANTS_REPORT"));
}

bool should_expect_invalid_hevm_failure_report()
{
    return parse_bool(
        get_env("POSEIDON_MGPU_RESNET20_EXPECT_INVALID_HEVM_REPORT"));
}

bool should_expect_inter_node_route_metadata()
{
    return parse_bool(
        get_env("POSEIDON_MGPU_RESNET20_EXPECT_INTER_NODE_ROUTE_METADATA"));
}

std::string shell_quote(const std::string &text)
{
    std::string output = "'";
    for (const char ch : text)
    {
        if (ch == '\'')
        {
            output += "'\\''";
        }
        else
        {
            output.push_back(ch);
        }
    }
    output.push_back('\'');
    return output;
}

void append_optional_flag(
    std::string &command, const char *env_name, const char *flag_name)
{
    if (const char *value = get_env(env_name))
    {
        command += " ";
        command += flag_name;
        command += " ";
        command += shell_quote(value);
    }
}

void append_optional_bool_flag(
    std::string &command, const char *env_name, const char *flag_name)
{
    if (parse_bool(get_env(env_name)))
    {
        command += " ";
        command += flag_name;
    }
}

void append_dump_tool_overrides(std::string &command)
{
    append_optional_flag(
        command, "POSEIDON_MGPU_RESNET20_DEVICE_COUNT", "--devices");
    append_optional_flag(
        command, "POSEIDON_MGPU_RESNET20_DEFAULT_DEVICE", "--default-device");
    append_optional_flag(
        command, "POSEIDON_MGPU_RESNET20_UPLOAD_DEVICE", "--upload-device");
    append_optional_flag(
        command, "POSEIDON_MGPU_RESNET20_COMPUTE_DEVICES", "--compute-devices");
    append_optional_flag(
        command, "POSEIDON_MGPU_RESNET20_DOWNLOAD_DEVICE", "--download-device");
    append_optional_flag(command, "POSEIDON_MGPU_RESNET20_NODES", "--nodes");
    append_optional_flag(
        command, "POSEIDON_MGPU_RESNET20_DEVICES_PER_NODE",
        "--devices-per-node");
    append_optional_flag(
        command, "POSEIDON_MGPU_RESNET20_SCHEDULER", "--scheduler");
    append_optional_bool_flag(
        command, "POSEIDON_MGPU_RESNET20_EXECUTION_CUDA_PEER_AVAILABLE",
        "--execution-cuda-peer-available");
    append_optional_bool_flag(
        command, "POSEIDON_MGPU_RESNET20_EXECUTION_INTER_NODE_AVAILABLE",
        "--execution-inter-node-available");
    append_optional_bool_flag(
        command, "POSEIDON_MGPU_RESNET20_PREFLIGHT_COMM_AVAILABLE",
        "--preflight-comm-available");
    append_optional_bool_flag(
        command, "POSEIDON_MGPU_RESNET20_PREFLIGHT_RELIN_KEYS",
        "--preflight-relin-keys");
    append_optional_bool_flag(
        command, "POSEIDON_MGPU_RESNET20_PREFLIGHT_GALOIS_KEYS",
        "--preflight-galois-keys");
    append_optional_bool_flag(
        command, "POSEIDON_MGPU_RESNET20_REQUIRE_READY", "--require-ready");
}

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void require_summary_expectations(const std::string &summary)
{
    if (const char *device_count = get_env("POSEIDON_MGPU_RESNET20_EXPECT_DEVICE_COUNT"))
    {
        require_contains(
            summary,
            "\"device_count\": " + std::string(device_count));
    }
    if (const char *nodes = get_env("POSEIDON_MGPU_RESNET20_EXPECT_NODES"))
    {
        require_contains(summary, "\"nodes\": " + std::string(nodes));
    }
    if (const char *devices_per_node =
            get_env("POSEIDON_MGPU_RESNET20_EXPECT_DEVICES_PER_NODE"))
    {
        require_contains(
            summary,
            "\"devices_per_node\": " + std::string(devices_per_node));
    }
    if (const char *compute_devices =
            get_env("POSEIDON_MGPU_RESNET20_EXPECT_COMPUTE_DEVICES"))
    {
        std::string expected = "\"compute_devices\": [";
        const std::string text = compute_devices;
        std::size_t begin = 0;
        bool first = true;
        while (begin <= text.size())
        {
            const std::size_t end = text.find(',', begin);
            const std::string item = text.substr(
                begin, end == std::string::npos ? std::string::npos : end - begin);
            if (item.empty())
            {
                throw std::invalid_argument(
                    "empty device id in POSEIDON_MGPU_RESNET20_EXPECT_COMPUTE_DEVICES");
            }
            expected += first ? "\n        " : ",\n        ";
            expected += item;
            first = false;
            if (end == std::string::npos)
            {
                break;
            }
            begin = end + 1;
        }
        expected += "\n      ]";
        require_contains(summary, expected);
    }
    if (should_expect_inter_node_route_metadata())
    {
        require_contains(summary, "\"stage\": \"communication_execution_preflight\"");
        require_contains(summary, "\"route_index\":");
        require_contains(summary, "\"transport\": \"inter_node\"");
        require_contains(summary, "\"source_device\":");
        require_contains(summary, "\"destination_device\":");
        require_contains(
            summary, "inter-node communication backend is not available");
    }
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

void write_binary_file(const fs::path &path, const std::string &contents)
{
    fs::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("failed to create artifact: " + path.string());
    }
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!stream)
    {
        throw std::runtime_error("failed to write artifact: " + path.string());
    }
}

std::string read_text_file(const std::string &path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open output: " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (stream.bad())
    {
        throw std::runtime_error("failed to read output: " + path);
    }
    return buffer.str();
}

void require_contains(const std::string &text, const std::string &needle)
{
    require(
        text.find(needle) != std::string::npos,
        "expected text to contain '" + needle + "'\ntext:\n" + text);
}

fs::path hevm_path(const fs::path &dacapo_root)
{
    return dacapo_root / "examples" / "optimized" / "dacapo" /
           "ResNet.40._hecate_ResNet.hevm";
}

fs::path constants_path(const fs::path &dacapo_root)
{
    return dacapo_root / "examples" / "traced" / "_hecate_ResNet.cst";
}

std::string make_mock_hevm_binary(bool unsupported_opcode)
{
    const std::uint16_t final_opcode = unsupported_opcode ? 4 : 9;
    return poseidon::mgpu::test::make_hevm_binary(
        1, 1, 3, 1, { 2 },
        {
            poseidon::mgpu::test::HevmOpRecord{
                0, 0, 0, poseidon::mgpu::test::make_hevm_encode_attr(2, 20) },
            poseidon::mgpu::test::HevmOpRecord{ final_opcode, 1, 0, 0 },
            poseidon::mgpu::test::HevmOpRecord{ final_opcode, 2, 1, 0 },
        },
        poseidon::mgpu::test::HevmConfigMetadata{
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

void create_mock_resnet20_artifacts(
    const fs::path &dacapo_root, bool unsupported_opcode)
{
    write_binary_file(
        hevm_path(dacapo_root), make_mock_hevm_binary(unsupported_opcode));
    write_binary_file(constants_path(dacapo_root), make_mock_constant_file());
}

void create_mock_resnet20_artifacts_without_hevm(const fs::path &dacapo_root)
{
    write_binary_file(constants_path(dacapo_root), make_mock_constant_file());
}

void create_mock_resnet20_artifacts_without_constants(const fs::path &dacapo_root)
{
    write_binary_file(hevm_path(dacapo_root), make_mock_hevm_binary(false));
}

void create_invalid_mock_resnet20_hevm_artifacts(const fs::path &dacapo_root)
{
    write_binary_file(hevm_path(dacapo_root), std::string(24, 'x'));
    write_binary_file(constants_path(dacapo_root), make_mock_constant_file());
}

std::string build_command(
    const std::string &dump_tool_path, const fs::path &dacapo_root,
    const std::string &summary_path, const std::string &schedule_path)
{
    std::string command =
        shell_quote(dump_tool_path) +
        " --hevm " + shell_quote(hevm_path(dacapo_root).string()) +
        " --constants " + shell_quote(constants_path(dacapo_root).string()) +
        " --write-summary-json " + shell_quote(summary_path) +
        " --write-schedule " + shell_quote(schedule_path) +
        " --no-schedule";

    if (const char *config = get_env("POSEIDON_MGPU_RESNET20_CONFIG"))
    {
        command += " --config ";
        command += shell_quote(config);
    }
    else
    {
        command +=
            " --devices 8"
            " --upload-device 0"
            " --compute-devices 0,1,2,3,4,5,6,7"
            " --download-device 0"
            " --opcode-summary"
            " --communication-plan"
            " --communication-execution-preflight"
            " --execution-cuda-peer-available"
            " --poseidon-gpu-preflight"
            " --preflight-comm-available"
            " --preflight-relin-keys"
            " --preflight-galois-keys"
            " --require-ready";
    }
    append_dump_tool_overrides(command);
    return command;
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        require(argc == 2, "expected Dacapo HEVM dump tool path");
        const bool use_mock =
            parse_bool(get_env("POSEIDON_MGPU_RESNET20_MOCK_ARTIFACTS"));
        const bool use_unsupported_mock =
            parse_bool(get_env("POSEIDON_MGPU_RESNET20_UNSUPPORTED_MOCK_ARTIFACTS"));
        const bool use_missing_hevm_mock =
            parse_bool(get_env("POSEIDON_MGPU_RESNET20_MISSING_HEVM_MOCK_ARTIFACTS"));
        const bool use_missing_constants_mock =
            parse_bool(get_env(
                "POSEIDON_MGPU_RESNET20_MISSING_CONSTANTS_MOCK_ARTIFACTS"));
        const bool use_invalid_hevm_mock =
            parse_bool(get_env("POSEIDON_MGPU_RESNET20_INVALID_HEVM_MOCK_ARTIFACTS"));
        const bool allow_not_ready =
            parse_bool(get_env("POSEIDON_MGPU_RESNET20_ALLOW_NOT_READY"));
        std::optional<TempDir> mock_root;
        fs::path dacapo_root;

        if (use_mock || use_unsupported_mock || use_missing_hevm_mock ||
            use_missing_constants_mock || use_invalid_hevm_mock)
        {
            mock_root.emplace();
            dacapo_root = mock_root->root();
            if (use_missing_hevm_mock)
            {
                create_mock_resnet20_artifacts_without_hevm(dacapo_root);
            }
            else if (use_missing_constants_mock)
            {
                create_mock_resnet20_artifacts_without_constants(dacapo_root);
            }
            else if (use_invalid_hevm_mock)
            {
                create_invalid_mock_resnet20_hevm_artifacts(dacapo_root);
            }
            else
            {
                create_mock_resnet20_artifacts(dacapo_root, use_unsupported_mock);
            }
        }
        else if (const char *root = get_env("POSEIDON_MGPU_RESNET20_DACAPO_ROOT"))
        {
            dacapo_root = root;
        }
        else if (const char *root = get_env("DACAPO_ROOT"))
        {
            dacapo_root = root;
        }
        else
        {
            std::cout << "skipping ResNet20 dump preflight test; set "
                      << "POSEIDON_MGPU_RESNET20_DACAPO_ROOT or DACAPO_ROOT, "
                      << "or POSEIDON_MGPU_RESNET20_MOCK_ARTIFACTS=1\n";
            return kSkip;
        }

        TempDir output;
        const std::string summary_path =
            get_env("POSEIDON_MGPU_RESNET20_PREFLIGHT_JSON") == nullptr
                ? output.path("resnet20-mgpu-preflight.json")
                : get_env("POSEIDON_MGPU_RESNET20_PREFLIGHT_JSON");
        const std::string schedule_path =
            get_env("POSEIDON_MGPU_RESNET20_SCHEDULE_DUMP") == nullptr
                ? output.path("resnet20-mgpu-schedule.mlir")
                : get_env("POSEIDON_MGPU_RESNET20_SCHEDULE_DUMP");

        const std::string command =
            build_command(argv[1], dacapo_root, summary_path, schedule_path);
        const int exit_code = std::system(command.c_str());

        const std::string summary = read_text_file(summary_path);
        require_contains(summary, "\"execution_gate\"");
        require_summary_expectations(summary);
        if (summary.find("\"status\": \"not_ready\"") != std::string::npos)
        {
            require(
                allow_not_ready,
                "ResNet20 dump preflight failed: " + command +
                    "\nsummary:\n" + summary);
            require_contains(summary, "\"status\": \"not_ready\"");
            if (should_expect_missing_hevm_failure_report())
            {
                require_contains(summary, "\"artifacts_loaded\": false");
                require_contains(summary, "\"stage\": \"read_hevm\"");
                require_contains(summary, "failed to open file");
            }
            else if (should_expect_missing_constants_failure_report())
            {
                require_contains(summary, "\"artifacts_loaded\": false");
                require_contains(summary, "\"stage\": \"read_constants\"");
                require_contains(summary, "\"hevm_opcode_summary\"");
                require_contains(summary, "failed to open Dacapo artifact file");
            }
            else if (should_expect_invalid_hevm_failure_report())
            {
                require_contains(summary, "\"artifacts_loaded\": true");
                require_contains(summary, "\"stage\": \"dacapo_opcode_summary\"");
                require_contains(summary, "\"hevm_opcode_summary\"");
                require_contains(summary, "invalid HEVM magic number");
            }
            else
            {
                require_contains(summary, "\"hevm_opcode_summary\"");
            }
            std::cout << "resnet20_dump_preflight_not_ready: " << summary_path
                      << '\n';
            return EXIT_SUCCESS;
        }
        require(exit_code == 0, "ResNet20 dump preflight failed: " + command);

        const std::string schedule = read_text_file(schedule_path);
        require_contains(summary, "\"status\": \"ready\"");
        require_contains(summary, "\"hevm_opcode_summary\"");
        require_contains(summary, "\"poseidon_gpu_execution_preflight\"");
        require_contains(schedule, "mgpu.schedule");
        std::cout << "resnet20_dump_preflight_json: " << summary_path << '\n';
        std::cout << "resnet20_dump_schedule: " << schedule_path << '\n';
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu ResNet20 dump preflight test failed: " << ex.what()
                  << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
