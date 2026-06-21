#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace
{

namespace fs = std::filesystem;

class TempDir
{
public:
    TempDir()
    {
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() /
                ("poseidon_mgpu_resnet20_artifact_check_test_" +
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

    std::string path(const std::string &name) const
    {
        return (path_ / name).string();
    }

    fs::path root() const
    {
        return path_;
    }

private:
    fs::path path_;
};

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
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

void write_text_file(const std::string &path, const std::string &contents)
{
    std::ofstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to create test file: " + path);
    }
    stream << contents;
    if (!stream)
    {
        throw std::runtime_error("failed to write test file: " + path);
    }
}

std::string read_text_file(const std::string &path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open test output: " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (stream.bad())
    {
        throw std::runtime_error("failed to read test output: " + path);
    }
    return buffer.str();
}

void require_contains(const std::string &text, const std::string &needle)
{
    require(
        text.find(needle) != std::string::npos,
        "expected output to contain '" + needle + "'\noutput:\n" + text);
}

void require_not_contains(const std::string &text, const std::string &needle)
{
    require(
        text.find(needle) == std::string::npos,
        "expected output not to contain '" + needle + "'\noutput:\n" + text);
}

fs::path hevm_path(const fs::path &root)
{
    return root / "examples" / "optimized" / "dacapo" /
           "ResNet.40._hecate_ResNet.hevm";
}

fs::path constants_path(const fs::path &root)
{
    return root / "examples" / "traced" / "_hecate_ResNet.cst";
}

void create_mock_resnet20_artifacts(const fs::path &root)
{
    fs::create_directories(hevm_path(root).parent_path());
    fs::create_directories(constants_path(root).parent_path());
    write_text_file(hevm_path(root).string(), "mock hevm");
    write_text_file(constants_path(root).string(), "mock constants");
}

std::string run_tool(
    const std::string &tool_path, const std::string &arguments, int &exit_code)
{
    TempDir output;
    const std::string stdout_path = output.path("stdout.txt");
    const std::string stderr_path = output.path("stderr.txt");
    const std::string command =
        shell_quote(tool_path) + " " + arguments + " > " +
        shell_quote(stdout_path) + " 2> " + shell_quote(stderr_path);

    exit_code = std::system(command.c_str());
    return read_text_file(stdout_path) + read_text_file(stderr_path);
}

void test_missing_artifacts_report(const std::string &tool_path)
{
    TempDir dacapo_root;
    int exit_code = 0;
    const std::string output = run_tool(
        tool_path,
        "--dacapo-root " + shell_quote(dacapo_root.root().string()),
        exit_code);

    require(exit_code != 0, "missing artifact check should fail");
    require_contains(output, "resnet20_artifact_check:");
    require_contains(output, "status: missing_artifacts");
    require_contains(output, "missing hevm");
    require_contains(output, "missing constants");
    require_contains(output, "hc-trace ResNet");
    require_contains(output, "hbt dacapo 40 ResNet HEAAN GPU");
    require_contains(output, "dump_command:");
    require_contains(output, "ResNet.40._hecate_ResNet.hevm");
    require_contains(output, "_hecate_ResNet.cst");
}

void test_ready_artifacts_command_hint(const std::string &tool_path)
{
    TempDir dacapo_root;
    create_mock_resnet20_artifacts(dacapo_root.root());

    int exit_code = 0;
    const std::string output = run_tool(
        tool_path,
        "--dacapo-root " + shell_quote(dacapo_root.root().string()) +
            " --dump-tool /tmp/poseidon_mgpu_dacapo_hevm_dump"
            " --summary-path /tmp/custom-summary.json"
            " --schedule-path /tmp/custom-schedule.mlir",
        exit_code);

    require(exit_code == 0, "ready artifact check should pass");
    require_contains(output, "status: ready");
    require_contains(output, "present: true");
    require_contains(output, "bytes: ");
    require_contains(output, "dump_command:");
    require_contains(output, "--hevm");
    require_contains(output, hevm_path(dacapo_root.root()).string());
    require_contains(output, "--constants");
    require_contains(output, constants_path(dacapo_root.root()).string());
    require_contains(output, "--devices 8");
    require_contains(output, "--compute-devices 0,1,2,3,4,5,6,7");
    require_contains(output, "--require-ready");
    require_contains(output, "--write-summary-json '/tmp/custom-summary.json'");
    require_contains(output, "--write-schedule '/tmp/custom-schedule.mlir'");
}

void test_config_command_hint(const std::string &tool_path)
{
    TempDir dacapo_root;
    create_mock_resnet20_artifacts(dacapo_root.root());
    const std::string config_path = dacapo_root.path("single_node_8gpu.json");
    write_text_file(config_path, "{\"version\":1}\n");

    int exit_code = 0;
    const std::string output = run_tool(
        tool_path,
        "--dacapo-root " + shell_quote(dacapo_root.root().string()) +
            " --config " + shell_quote(config_path),
        exit_code);

    require(exit_code == 0, "config artifact check should pass");
    require_contains(output, "status: ready");
    require_contains(output, "config: ");
    require_contains(output, "--config " + shell_quote(config_path));
    require_contains(output, "--summary-json");
    require_not_contains(output, "--devices 8");
    require_not_contains(output, "--execution-cuda-peer-available");
}

void test_config_command_hint_with_overrides(const std::string &tool_path)
{
    TempDir dacapo_root;
    create_mock_resnet20_artifacts(dacapo_root.root());
    const std::string config_path = dacapo_root.path("single_node_8gpu.json");
    const std::string report_path = dacapo_root.path("artifact-check.json");
    write_text_file(config_path, "{\"version\":1}\n");

    int exit_code = 0;
    const std::string output = run_tool(
        tool_path,
        "--dacapo-root " + shell_quote(dacapo_root.root().string()) +
            " --config " + shell_quote(config_path) +
            " --devices 2"
            " --compute-devices 0,1"
            " --devices-per-node 2"
            " --execution-cuda-peer-available"
            " --write-summary-json " + shell_quote(report_path),
        exit_code);

    require(exit_code == 0, "config override artifact check should pass");
    require_contains(output, "status: ready");
    require_contains(output, "--config " + shell_quote(config_path));
    require_contains(output, "--devices '2'");
    require_contains(output, "--compute-devices '0,1'");
    require_contains(output, "--devices-per-node '2'");
    require_contains(output, "--execution-cuda-peer-available");
    const std::string report = read_text_file(report_path);
    require_contains(report, "\"status\": \"ready\"");
    require_contains(report, "--devices '2'");
    require_contains(report, "--compute-devices '0,1'");
    require_contains(report, "--execution-cuda-peer-available");
}

void test_no_command_mode(const std::string &tool_path)
{
    TempDir dacapo_root;
    create_mock_resnet20_artifacts(dacapo_root.root());

    int exit_code = 0;
    const std::string output = run_tool(
        tool_path,
        "--dacapo-root " + shell_quote(dacapo_root.root().string()) +
            " --no-command",
        exit_code);

    require(exit_code == 0, "no-command artifact check should pass");
    require_contains(output, "status: ready");
    require_not_contains(output, "dump_command:");
}

void test_summary_json_for_ready_artifacts(const std::string &tool_path)
{
    TempDir dacapo_root;
    create_mock_resnet20_artifacts(dacapo_root.root());
    const std::string config_path = dacapo_root.path("single_node_8gpu.json");
    const std::string report_path = dacapo_root.path("artifact-check.json");
    write_text_file(config_path, "{\"version\":1}\n");

    int exit_code = 0;
    const std::string output = run_tool(
        tool_path,
        "--dacapo-root " + shell_quote(dacapo_root.root().string()) +
            " --config " + shell_quote(config_path) +
            " --write-summary-json " + shell_quote(report_path) +
            " --summary-json"
            " --no-command",
        exit_code);

    require(exit_code == 0, "ready JSON artifact check should pass");
    const std::string report = read_text_file(report_path);
    require_contains(output, "\"version\": 1");
    require_contains(output, "\"ready\": true");
    require_contains(report, "\"status\": \"ready\"");
    require_contains(report, "\"artifacts\"");
    require_contains(report, "\"hevm\"");
    require_contains(report, "\"constants\"");
    require_contains(report, "\"present\": true");
    require_contains(report, "\"config\"");
    require_contains(report, config_path);
    require_contains(report, "\"dump_command\"");
    require_contains(report, "ResNet.40._hecate_ResNet.hevm");
    require_contains(report, "_hecate_ResNet.cst");
}

void test_missing_config_report(const std::string &tool_path)
{
    TempDir dacapo_root;
    create_mock_resnet20_artifacts(dacapo_root.root());
    const std::string config_path = dacapo_root.path("missing-config.json");
    const std::string report_path = dacapo_root.path("missing-config-report.json");

    int exit_code = 0;
    const std::string output = run_tool(
        tool_path,
        "--dacapo-root " + shell_quote(dacapo_root.root().string()) +
            " --config " + shell_quote(config_path) +
            " --write-summary-json " + shell_quote(report_path),
        exit_code);

    require(exit_code != 0, "missing config check should fail");
    require_contains(output, "status: missing_config");
    require_contains(output, "missing config");
    const std::string report = read_text_file(report_path);
    require_contains(report, "\"ready\": false");
    require_contains(report, "\"status\": \"missing_config\"");
    require_contains(report, "\"config\"");
    require_contains(report, "\"present\": false");
    require_contains(report, "\"diagnostic\": \"missing\"");
}

void test_summary_json_for_missing_artifacts(const std::string &tool_path)
{
    TempDir dacapo_root;
    const std::string report_path = dacapo_root.path("missing-artifact-check.json");

    int exit_code = 0;
    const std::string output = run_tool(
        tool_path,
        "--dacapo-root " + shell_quote(dacapo_root.root().string()) +
            " --write-summary-json " + shell_quote(report_path) +
            " --no-command",
        exit_code);

    require(exit_code != 0, "missing artifact JSON check should fail");
    require_not_contains(output, "\"version\": 1");
    const std::string report = read_text_file(report_path);
    require_contains(report, "\"ready\": false");
    require_contains(report, "\"status\": \"missing_artifacts\"");
    require_contains(report, "\"present\": false");
    require_contains(report, "\"diagnostic\": \"missing\"");
    require_contains(report, "\"generation_hint\"");
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        require(argc == 2, "expected resnet20 artifact check tool path");
        test_missing_artifacts_report(argv[1]);
        test_ready_artifacts_command_hint(argv[1]);
        test_config_command_hint(argv[1]);
        test_config_command_hint_with_overrides(argv[1]);
        test_no_command_mode(argv[1]);
        test_summary_json_for_ready_artifacts(argv[1]);
        test_missing_config_report(argv[1]);
        test_summary_json_for_missing_artifacts(argv[1]);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu ResNet20 artifact check tool test failed: "
                  << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu ResNet20 artifact check tool tests passed\n";
    return EXIT_SUCCESS;
}
