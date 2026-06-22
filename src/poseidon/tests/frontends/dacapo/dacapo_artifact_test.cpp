#include "poseidon/frontends/dacapo/dacapo_artifacts.h"
#include "poseidon/tests/frontends/dacapo/hevm_test_utils.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

using namespace poseidon::mgpu;

namespace
{

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void require_contains(const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos)
    {
        throw std::runtime_error("expected text to contain: " + needle + "\ntext:\n" + text);
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

class TempDir
{
public:
    TempDir()
    {
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("poseidon_mgpu_dacapo_artifact_test_" + std::to_string(tick));
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

std::string make_hevm_binary()
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

std::string make_truncated_constant_file()
{
    std::string output;
    append_i64(output, 1);
    append_i64(output, 2);
    append_double(output, 1.0);
    return output;
}

void test_loads_hevm_schedule_and_constants_from_files()
{
    TempDir temp;
    const std::string hevm_path = temp.path("resnet20_mock.hevm");
    const std::string constants_path = temp.path("resnet20_mock.cst");
    write_binary_file(hevm_path, make_hevm_binary());
    write_binary_file(constants_path, make_constant_file());

    StaticSchedulePipelineOptions options;
    options.device_count = 1;
    options.emit_debug_dump = true;

    const DacapoHevmArtifactResult result = prepare_dacapo_hevm_artifacts_from_files(
        DacapoHevmArtifactPaths{ hevm_path, constants_path }, options);

    require(result.ok(), "artifact load failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 4, "expected HEVM schedule without inserted copies");
    require(
        result.schedule.ops[0].kind == MgpuOpKind::UploadCipher,
        "first op should upload HEVM ciphertext argument");
    require(
        result.schedule.ops[1].kind == MgpuOpKind::UploadPlain,
        "second op should upload HEVM plaintext constant");
    require(
        result.schedule.ops[2].kind == MgpuOpKind::MultiplyPlain,
        "third op should multiply by plaintext");
    require(
        result.schedule.ops[3].kind == MgpuOpKind::Download,
        "last op should download HEVM result");
    require(result.constants.values.size() == 1, "expected one parsed constant vector");
    require(result.constants.values[0].size() == 4, "constant vector length mismatch");
    require(result.constants.values[0][1] == -1.0, "constant vector value mismatch");
    require_contains(result.debug_dump, "mgpu.multiply_plain");
    require_contains(result.debug_dump, "hevm_constant_index=0");
}

void test_reports_missing_hevm_file()
{
    TempDir temp;
    const std::string constants_path = temp.path("resnet20_mock.cst");
    write_binary_file(constants_path, make_constant_file());

    const DacapoHevmArtifactResult result = prepare_dacapo_hevm_artifacts_from_files(
        DacapoHevmArtifactPaths{ temp.path("missing.hevm"), constants_path });

    require(!result.ok(), "missing HEVM file should fail");
    require(result.schedule.ops.empty(), "failed artifact load should not return a schedule");
    require(result.constants.values.empty(), "failed artifact load should not return constants");
    require_contains(result.format_diagnostics(), "read_hevm");
    require_contains(result.format_diagnostics(), "failed to open Dacapo artifact file");
}

void test_reports_constant_parse_errors()
{
    TempDir temp;
    const std::string hevm_path = temp.path("resnet20_mock.hevm");
    const std::string constants_path = temp.path("bad.cst");
    write_binary_file(hevm_path, make_hevm_binary());
    write_binary_file(constants_path, make_truncated_constant_file());

    const DacapoHevmArtifactResult result = prepare_dacapo_hevm_artifacts_from_files(
        DacapoHevmArtifactPaths{ hevm_path, constants_path });

    require(!result.ok(), "bad constants file should fail");
    require(result.schedule.ops.empty(), "failed artifact load should not return a schedule");
    require(result.constants.values.empty(), "failed artifact load should not return constants");
    require_contains(result.format_diagnostics(), "constants");
    require_contains(result.format_diagnostics(), "truncated Dacapo constant file");
}

}  // namespace

int main()
{
    try
    {
        test_loads_hevm_schedule_and_constants_from_files();
        test_reports_missing_hevm_file();
        test_reports_constant_parse_errors();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu Dacapo artifact test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu Dacapo artifact tests passed\n";
    return EXIT_SUCCESS;
}
