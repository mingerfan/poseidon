#include <chrono>
#include <cstdlib>
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
                ("poseidon_mgpu_resnet20_artifact_path_test_" +
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

void append_dump_hint_overrides(std::string &command)
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

void write_text_file(const fs::path &path, const std::string &contents)
{
    fs::create_directories(path.parent_path());
    std::ofstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to create mock artifact: " + path.string());
    }
    stream << contents;
    if (!stream)
    {
        throw std::runtime_error("failed to write mock artifact: " + path.string());
    }
}

std::string read_text_file(const std::string &path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open report: " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (stream.bad())
    {
        throw std::runtime_error("failed to read report: " + path);
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

void create_mock_resnet20_artifacts(const fs::path &dacapo_root)
{
    write_text_file(hevm_path(dacapo_root), "mock hevm\n");
    write_text_file(constants_path(dacapo_root), "mock constants\n");
}

std::string build_command(
    const std::string &tool_path, const fs::path &dacapo_root,
    const std::string &report_path)
{
    std::string command =
        shell_quote(tool_path) +
        " --dacapo-root " + shell_quote(dacapo_root.string()) +
        " --write-summary-json " + shell_quote(report_path) +
        " --no-command";

    if (const char *config = get_env("POSEIDON_MGPU_RESNET20_CONFIG"))
    {
        command += " --config ";
        command += shell_quote(config);
    }
    if (const char *dump_tool = get_env("POSEIDON_MGPU_RESNET20_DUMP_TOOL"))
    {
        command += " --dump-tool ";
        command += shell_quote(dump_tool);
    }
    if (const char *summary_path = get_env("POSEIDON_MGPU_RESNET20_SUMMARY_PATH"))
    {
        command += " --summary-path ";
        command += shell_quote(summary_path);
    }
    if (const char *schedule_path = get_env("POSEIDON_MGPU_RESNET20_SCHEDULE_PATH"))
    {
        command += " --schedule-path ";
        command += shell_quote(schedule_path);
    }
    append_dump_hint_overrides(command);
    return command;
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        require(argc == 2, "expected resnet20 artifact check tool path");
        const std::string tool_path = argv[1];
        const bool use_mock =
            parse_bool(get_env("POSEIDON_MGPU_RESNET20_MOCK_ARTIFACTS"));
        std::optional<TempDir> mock_root;
        fs::path dacapo_root;

        if (use_mock)
        {
            mock_root.emplace();
            dacapo_root = mock_root->root();
            create_mock_resnet20_artifacts(dacapo_root);
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
            std::cout << "skipping ResNet20 artifact path test; set "
                      << "POSEIDON_MGPU_RESNET20_DACAPO_ROOT or DACAPO_ROOT, "
                      << "or POSEIDON_MGPU_RESNET20_MOCK_ARTIFACTS=1\n";
            return kSkip;
        }

        TempDir output;
        const std::string report_path =
            get_env("POSEIDON_MGPU_RESNET20_REPORT_JSON") == nullptr
                ? output.path("resnet20-artifact-paths.json")
                : get_env("POSEIDON_MGPU_RESNET20_REPORT_JSON");
        const std::string command = build_command(tool_path, dacapo_root, report_path);
        const int exit_code = std::system(command.c_str());
        require(exit_code == 0, "ResNet20 artifact path check failed: " + command);

        const std::string report = read_text_file(report_path);
        require_contains(report, "\"ready\": true");
        require_contains(report, "\"status\": \"ready\"");
        require_contains(report, "ResNet.40._hecate_ResNet.hevm");
        require_contains(report, "_hecate_ResNet.cst");
        if (const char *device_count = get_env("POSEIDON_MGPU_RESNET20_DEVICE_COUNT"))
        {
            require_contains(report, "--devices '" + std::string(device_count) + "'");
        }
        if (const char *compute_devices =
                get_env("POSEIDON_MGPU_RESNET20_COMPUTE_DEVICES"))
        {
            require_contains(
                report, "--compute-devices '" + std::string(compute_devices) + "'");
        }
        if (parse_bool(get_env("POSEIDON_MGPU_RESNET20_EXECUTION_CUDA_PEER_AVAILABLE")))
        {
            require_contains(report, "--execution-cuda-peer-available");
        }
        std::cout << "resnet20_artifact_path_report: " << report_path << '\n';
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu ResNet20 artifact path test failed: " << ex.what()
                  << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
