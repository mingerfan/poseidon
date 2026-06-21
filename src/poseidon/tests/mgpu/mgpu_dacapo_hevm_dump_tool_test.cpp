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

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        require(argc == 2, "expected path to poseidon_mgpu_dacapo_hevm_dump");
        test_write_schedule_and_report(argv[1]);
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
