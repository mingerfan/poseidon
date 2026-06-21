#include "poseidon/tests/mgpu/hevm_test_utils.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace
{

class TempDir
{
public:
    TempDir()
    {
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("poseidon_mgpu_dacapo_hevm_dump_tool_test_" +
                 std::to_string(tick));
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

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
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

void write_binary_file(const std::string &path, const std::string &contents)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("failed to create test artifact: " + path);
    }
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!stream)
    {
        throw std::runtime_error("failed to write test artifact: " + path);
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

std::string make_hevm_binary()
{
    return poseidon::mgpu::test::make_hevm_binary(
        1, 1, 2, 1, { 1 },
        {
            poseidon::mgpu::test::HevmOpRecord{
                0, 0, 0, poseidon::mgpu::test::make_hevm_encode_attr(2, 20) },
            poseidon::mgpu::test::HevmOpRecord{ 9, 1, 0, 0 },
        },
        poseidon::mgpu::test::HevmConfigMetadata{
            { 20 },
            { 2 },
            { 40 },
            { 2 },
            2,
        });
}

std::string make_two_compute_hevm_binary()
{
    return poseidon::mgpu::test::make_hevm_binary(
        1, 1, 3, 1, { 2 },
        {
            poseidon::mgpu::test::HevmOpRecord{
                0, 0, 0, poseidon::mgpu::test::make_hevm_encode_attr(2, 20) },
            poseidon::mgpu::test::HevmOpRecord{ 9, 1, 0, 0 },
            poseidon::mgpu::test::HevmOpRecord{ 9, 2, 1, 0 },
        },
        poseidon::mgpu::test::HevmConfigMetadata{
            { 20 },
            { 2 },
            { 40 },
            { 2 },
            2,
        });
}

std::string make_unsupported_opcode_hevm_binary()
{
    return poseidon::mgpu::test::make_hevm_binary(
        1, 1, 2, 1, { 1 },
        {
            poseidon::mgpu::test::HevmOpRecord{
                0, 0, 0, poseidon::mgpu::test::make_hevm_encode_attr(2, 20) },
            poseidon::mgpu::test::HevmOpRecord{ 4, 1, 0, 1 },
        },
        poseidon::mgpu::test::HevmConfigMetadata{
            { 20 },
            { 2 },
            { 40 },
            { 2 },
            2,
        });
}

std::string make_constant_file()
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

void test_write_schedule_and_report(const std::string &tool_path)
{
    TempDir temp;
    const std::string hevm_path = temp.path("mock.hevm");
    const std::string constants_path = temp.path("mock.cst");
    const std::string summary_path = temp.path("summary.json");
    const std::string schedule_path = temp.path("schedule.mgpu");
    const std::string stdout_path = temp.path("stdout.txt");
    write_binary_file(hevm_path, make_hevm_binary());
    write_binary_file(constants_path, make_constant_file());

    const std::string command =
        shell_quote(tool_path) +
        " --hevm " + shell_quote(hevm_path) +
        " --constants " + shell_quote(constants_path) +
        " --devices 2"
        " --upload-device 0"
        " --compute-devices 1"
        " --download-device 0"
        " --opcode-summary"
        " --communication-plan"
        " --communication-execution-preflight"
        " --execution-cuda-peer-available"
        " --poseidon-gpu-preflight"
        " --preflight-comm-available"
        " --preflight-relin-keys"
        " --preflight-galois-keys"
        " --write-summary-json " + shell_quote(summary_path) +
        " --write-schedule " + shell_quote(schedule_path) +
        " --no-schedule > " + shell_quote(stdout_path);

    const int exit_code = std::system(command.c_str());
    require(exit_code == 0, "dump tool command failed: " + command);

    const std::string stdout_text = read_text_file(stdout_path);
    const std::string summary_text = read_text_file(summary_path);
    const std::string schedule_text = read_text_file(schedule_path);

    require_not_contains(stdout_text, "mgpu.schedule");
    require_not_contains(summary_text, "\"debug_dump\"");
    require_contains(summary_text, "\"poseidon_gpu_execution_preflight\"");
    require_contains(summary_text, "\"communication_plan\"");
    require_contains(schedule_text, "mgpu.schedule");
    require_contains(schedule_text, "mgpu.copy_cipher");
    require_contains(schedule_text, "mgpu.copy_plain");
}

void test_config_file_template_report(
    const std::string &tool_path, const std::string &config_dir)
{
    TempDir temp;
    const std::string hevm_path = temp.path("mock.hevm");
    const std::string constants_path = temp.path("mock.cst");
    const std::string summary_path = temp.path("summary.json");
    const std::string stdout_path = temp.path("stdout.txt");
    const std::string config_path =
        (std::filesystem::path(config_dir) / "single_node_8gpu.json").string();
    write_binary_file(hevm_path, make_two_compute_hevm_binary());
    write_binary_file(constants_path, make_constant_file());

    const std::string command =
        shell_quote(tool_path) +
        " --config " + shell_quote(config_path) +
        " --hevm " + shell_quote(hevm_path) +
        " --constants " + shell_quote(constants_path) +
        " --write-summary-json " + shell_quote(summary_path) +
        " --no-schedule > " + shell_quote(stdout_path);

    const int exit_code = std::system(command.c_str());
    require(exit_code == 0, "dump tool config command failed: " + command);

    const std::string stdout_text = read_text_file(stdout_path);
    const std::string summary_text = read_text_file(summary_path);

    require_not_contains(stdout_text, "mgpu.schedule");
    require_contains(summary_text, "\"device_count\": 8");
    require_contains(summary_text, "\"cuda_peer\": true");
    require_contains(summary_text, "\"require_ready\": true");
    require_contains(summary_text, "\"execution_gate\"");
    require_contains(summary_text, "\"status\": \"ready\"");
    require_contains(summary_text, "\"copy_plain\": 1");
    require_contains(summary_text, "\"copy_cipher\": 2");
}

void test_failure_report_for_unsupported_opcode(const std::string &tool_path)
{
    TempDir temp;
    const std::string hevm_path = temp.path("unsupported.hevm");
    const std::string constants_path = temp.path("unsupported.cst");
    const std::string summary_path = temp.path("failure_summary.json");
    const std::string stdout_path = temp.path("stdout.txt");
    const std::string stderr_path = temp.path("stderr.txt");
    write_binary_file(hevm_path, make_unsupported_opcode_hevm_binary());
    write_binary_file(constants_path, make_constant_file());

    const std::string command =
        shell_quote(tool_path) +
        " --hevm " + shell_quote(hevm_path) +
        " --constants " + shell_quote(constants_path) +
        " --opcode-summary"
        " --require-ready"
        " --write-summary-json " + shell_quote(summary_path) +
        " --no-schedule > " + shell_quote(stdout_path) +
        " 2> " + shell_quote(stderr_path);

    const int exit_code = std::system(command.c_str());
    require(exit_code != 0, "unsupported opcode command should fail: " + command);

    const std::string stdout_text = read_text_file(stdout_path);
    const std::string stderr_text = read_text_file(stderr_path);
    const std::string summary_text = read_text_file(summary_path);

    require_not_contains(stdout_text, "mgpu.schedule");
    require_contains(stderr_text, "unsupported HEVM opcode 4");
    require_contains(summary_text, "\"status\": \"not_ready\"");
    require_contains(summary_text, "\"schedule_built\": false");
    require_contains(summary_text, "\"artifact_diagnostics\"");
    require_contains(summary_text, "\"hevm_opcode_summary\"");
    require_contains(summary_text, "\"name\": \"ModswitchC\"");
    require_contains(summary_text, "\"supported\": false");
    require_contains(summary_text, "unsupported HEVM opcode 4");
}

void test_failure_report_for_missing_hevm_file(const std::string &tool_path)
{
    TempDir temp;
    const std::string hevm_path = temp.path("missing.hevm");
    const std::string constants_path = temp.path("mock.cst");
    const std::string summary_path = temp.path("missing_hevm_summary.json");
    const std::string stdout_path = temp.path("stdout.txt");
    const std::string stderr_path = temp.path("stderr.txt");
    write_binary_file(constants_path, make_constant_file());

    const std::string command =
        shell_quote(tool_path) +
        " --hevm " + shell_quote(hevm_path) +
        " --constants " + shell_quote(constants_path) +
        " --opcode-summary"
        " --write-summary-json " + shell_quote(summary_path) +
        " --no-schedule > " + shell_quote(stdout_path) +
        " 2> " + shell_quote(stderr_path);

    const int exit_code = std::system(command.c_str());
    require(exit_code != 0, "missing HEVM command should fail: " + command);

    const std::string stdout_text = read_text_file(stdout_path);
    const std::string stderr_text = read_text_file(stderr_path);
    const std::string summary_text = read_text_file(summary_path);

    require_not_contains(stdout_text, "mgpu.schedule");
    require_contains(stderr_text, "read_hevm");
    require_contains(summary_text, "\"status\": \"not_ready\"");
    require_contains(summary_text, "\"artifacts_loaded\": false");
    require_contains(summary_text, "\"schedule_built\": false");
    require_contains(summary_text, "\"stage\": \"read_hevm\"");
    require_contains(summary_text, "failed to open file");
}

void test_failure_report_for_missing_constants_file(const std::string &tool_path)
{
    TempDir temp;
    const std::string hevm_path = temp.path("mock.hevm");
    const std::string constants_path = temp.path("missing.cst");
    const std::string summary_path = temp.path("missing_constants_summary.json");
    const std::string stdout_path = temp.path("stdout.txt");
    const std::string stderr_path = temp.path("stderr.txt");
    write_binary_file(hevm_path, make_hevm_binary());

    const std::string command =
        shell_quote(tool_path) +
        " --hevm " + shell_quote(hevm_path) +
        " --constants " + shell_quote(constants_path) +
        " --opcode-summary"
        " --write-summary-json " + shell_quote(summary_path) +
        " --no-schedule > " + shell_quote(stdout_path) +
        " 2> " + shell_quote(stderr_path);

    const int exit_code = std::system(command.c_str());
    require(exit_code != 0, "missing constants command should fail: " + command);

    const std::string stdout_text = read_text_file(stdout_path);
    const std::string stderr_text = read_text_file(stderr_path);
    const std::string summary_text = read_text_file(summary_path);

    require_not_contains(stdout_text, "mgpu.schedule");
    require_contains(stderr_text, "read_constants");
    require_contains(summary_text, "\"status\": \"not_ready\"");
    require_contains(summary_text, "\"artifacts_loaded\": false");
    require_contains(summary_text, "\"schedule_built\": false");
    require_contains(summary_text, "\"stage\": \"read_constants\"");
    require_contains(summary_text, "\"hevm_opcode_summary\"");
    require_contains(summary_text, "failed to open Dacapo artifact file");
}

void test_failure_report_for_invalid_hevm_file(const std::string &tool_path)
{
    TempDir temp;
    const std::string hevm_path = temp.path("invalid.hevm");
    const std::string constants_path = temp.path("mock.cst");
    const std::string summary_path = temp.path("invalid_hevm_summary.json");
    const std::string stdout_path = temp.path("stdout.txt");
    const std::string stderr_path = temp.path("stderr.txt");
    write_binary_file(hevm_path, std::string(24, 'x'));
    write_binary_file(constants_path, make_constant_file());

    const std::string command =
        shell_quote(tool_path) +
        " --hevm " + shell_quote(hevm_path) +
        " --constants " + shell_quote(constants_path) +
        " --opcode-summary"
        " --write-summary-json " + shell_quote(summary_path) +
        " --no-schedule > " + shell_quote(stdout_path) +
        " 2> " + shell_quote(stderr_path);

    const int exit_code = std::system(command.c_str());
    require(exit_code != 0, "invalid HEVM command should fail: " + command);

    const std::string stdout_text = read_text_file(stdout_path);
    const std::string stderr_text = read_text_file(stderr_path);
    const std::string summary_text = read_text_file(summary_path);

    require_not_contains(stdout_text, "mgpu.schedule");
    require_contains(stderr_text, "invalid HEVM magic number");
    require_contains(summary_text, "\"status\": \"not_ready\"");
    require_contains(summary_text, "\"artifacts_loaded\": true");
    require_contains(summary_text, "\"schedule_built\": false");
    require_contains(summary_text, "\"stage\": \"dacapo_opcode_summary\"");
    require_contains(summary_text, "\"hevm_opcode_summary\"");
    require_contains(summary_text, "invalid HEVM magic number");
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        require(argc == 3, "expected dump tool path and mgpu config directory");
        test_write_schedule_and_report(argv[1]);
        test_config_file_template_report(argv[1], argv[2]);
        test_failure_report_for_unsupported_opcode(argv[1]);
        test_failure_report_for_missing_hevm_file(argv[1]);
        test_failure_report_for_missing_constants_file(argv[1]);
        test_failure_report_for_invalid_hevm_file(argv[1]);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu Dacapo HEVM dump tool test failed: " << ex.what()
                  << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu Dacapo HEVM dump tool tests passed\n";
    return EXIT_SUCCESS;
}
