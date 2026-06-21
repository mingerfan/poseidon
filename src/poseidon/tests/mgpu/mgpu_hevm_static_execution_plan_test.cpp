#include "poseidon/mgpu/runtime/hevm_static_execution_plan.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"
#include "poseidon/tests/mgpu/hevm_test_utils.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using namespace poseidon;
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
                ("poseidon_mgpu_hevm_static_execution_plan_test_" +
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

ParametersLiteral make_ckks_test_parameters()
{
    ParametersLiteral parms(
        CKKS,
        /*log_n=*/12,
        /*log_slots=*/11,
        /*log_scale=*/20,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        Modulus(0),
        std::vector<Modulus>{},
        std::vector<Modulus>{},
        sec_level_type::none);
    parms.set_log_modulus(std::vector<std::uint32_t>(3, 30), {});
    return parms;
}

std::string make_hevm_binary(std::uint16_t encode_level = 2)
{
    return test::make_hevm_binary(
        1, 1, 2, 1, { 1 },
        {
            test::HevmOpRecord{
                0, 0, 0, test::make_hevm_encode_attr(encode_level, 20) },
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

void write_artifacts(
    const TempDir &temp, const std::string &hevm, const std::string &constants,
    DacapoHevmArtifactPaths &paths)
{
    paths.hevm_path = temp.path("resnet20_mock.hevm");
    paths.constants_path = temp.path("resnet20_mock.cst");
    write_binary_file(paths.hevm_path, hevm);
    write_binary_file(paths.constants_path, constants);
}

void test_prepares_static_execution_plan_from_files()
{
    TempDir temp;
    DacapoHevmArtifactPaths paths;
    write_artifacts(temp, make_hevm_binary(), make_constant_file(), paths);

    StaticSchedulePipelineOptions options;
    options.device_count = 1;
    options.emit_debug_dump = true;

    const PoseidonContext context(make_ckks_test_parameters());
    const HevmStaticExecutionPlanResult result =
        prepare_hevm_static_execution_plan_from_files(context, paths, options);

    require(result.ok(), "execution plan prepare failed:\n" + result.format_diagnostics());
    require(result.plan.schedule.ops.size() == 4, "expected four scheduled HEVM ops");
    require(result.plan.constants.values.size() == 1, "expected one constant vector");
    require(result.plan.io_plan.cipher_inputs.size() == 1, "expected one cipher input");
    require(result.plan.io_plan.plain_inputs.size() == 1, "expected one plaintext input");
    require(result.plan.io_plan.results.size() == 1, "expected one result binding");
    require(
        result.plan.encoded_plaintexts.size() == 1,
        "expected one encoded plaintext upload");

    const auto &parms_id_map = context.crt_context()->parms_id_map();
    const Plaintext &plaintext = *result.plan.encoded_plaintexts[0].plaintext;
    require(plaintext.parms_id() == parms_id_map.at(2), "encoded plaintext level mismatch");
    require(plaintext.scale() == std::ldexp(1.0, 20), "encoded plaintext scale mismatch");
    require_contains(result.plan.debug_dump, "mgpu.multiply_plain");
    require_contains(result.plan.debug_dump, "hevm_result_level=2");
}

void test_propagates_artifact_load_errors()
{
    TempDir temp;
    DacapoHevmArtifactPaths paths;
    paths.hevm_path = temp.path("resnet20_mock.hevm");
    paths.constants_path = temp.path("missing.cst");
    write_binary_file(paths.hevm_path, make_hevm_binary());

    const PoseidonContext context(make_ckks_test_parameters());
    const HevmStaticExecutionPlanResult result =
        prepare_hevm_static_execution_plan_from_files(context, paths);

    require(!result.ok(), "missing constants artifact should fail");
    require(result.plan.schedule.ops.empty(), "failed plan should not return a schedule");
    require(
        result.plan.encoded_plaintexts.empty(),
        "failed plan should not return encoded plaintexts");
    require_contains(result.format_diagnostics(), "artifact.read_constants");
    require_contains(result.format_diagnostics(), "failed to open Dacapo artifact file");
}

void test_propagates_plaintext_encoding_errors()
{
    TempDir temp;
    DacapoHevmArtifactPaths paths;
    write_artifacts(temp, make_hevm_binary(/*encode_level=*/63), make_constant_file(), paths);

    const PoseidonContext context(make_ckks_test_parameters());
    const HevmStaticExecutionPlanResult result =
        prepare_hevm_static_execution_plan_from_files(context, paths);

    require(!result.ok(), "missing encoding level should fail");
    require(result.plan.schedule.ops.empty(), "failed plan should not return a schedule");
    require_contains(result.format_diagnostics(), "hevm_plaintext_encoding");
    require_contains(result.format_diagnostics(), "missing Poseidon parms_id");
}

}  // namespace

int main()
{
    try
    {
        test_prepares_static_execution_plan_from_files();
        test_propagates_artifact_load_errors();
        test_propagates_plaintext_encoding_errors();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu HEVM static execution plan test failed: " << ex.what()
                  << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu HEVM static execution plan tests passed\n";
    return EXIT_SUCCESS;
}
