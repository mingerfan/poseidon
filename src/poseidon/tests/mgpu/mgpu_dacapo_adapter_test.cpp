#include "poseidon/mgpu/compiler/dacapo_adapter.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

struct HevmOpRecord
{
    std::uint16_t opcode = 0;
    std::uint16_t dst = 0;
    std::uint16_t lhs = 0;
    std::uint16_t rhs = 0;
};

void append_u16(std::string &output, std::uint16_t value)
{
    output.push_back(static_cast<char>(value & 0xFF));
    output.push_back(static_cast<char>((value >> 8) & 0xFF));
}

void append_u32(std::string &output, std::uint32_t value)
{
    for (int i = 0; i < 4; ++i)
    {
        output.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
    }
}

void append_u64(std::string &output, std::uint64_t value)
{
    for (int i = 0; i < 8; ++i)
    {
        output.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
    }
}

std::string make_hevm_binary(
    std::uint64_t arg_length, std::uint64_t res_length,
    std::uint64_t num_ctxt_buffer, std::uint64_t num_ptxt_buffer,
    const std::vector<std::uint64_t> &result_registers,
    const std::vector<HevmOpRecord> &ops)
{
    const std::uint64_t config_array_count = arg_length * 2 + res_length * 3;
    const std::uint64_t config_body_length = 40 + config_array_count * 8;

    std::string output;
    append_u32(output, 0x4845564D);
    append_u32(output, 24);
    append_u64(output, arg_length);
    append_u64(output, res_length);

    append_u64(output, config_body_length);
    append_u64(output, ops.size());
    append_u64(output, num_ctxt_buffer);
    append_u64(output, num_ptxt_buffer);
    append_u64(output, 0);

    for (std::uint64_t i = 0; i < arg_length; ++i)
    {
        append_u64(output, 0);
    }
    for (std::uint64_t i = 0; i < arg_length; ++i)
    {
        append_u64(output, 0);
    }
    for (std::uint64_t i = 0; i < res_length; ++i)
    {
        append_u64(output, 0);
    }
    for (std::uint64_t i = 0; i < res_length; ++i)
    {
        append_u64(output, 0);
    }
    for (std::uint64_t i = 0; i < res_length; ++i)
    {
        append_u64(output, result_registers[i]);
    }

    for (const HevmOpRecord &op : ops)
    {
        append_u16(output, op.opcode);
        append_u16(output, op.dst);
        append_u16(output, op.lhs);
        append_u16(output, op.rhs);
    }

    return output;
}

void test_format_strings()
{
    require(std::string(to_string(DacapoInputFormat::Unknown)) == "unknown",
            "unknown format string mismatch");
    require(std::string(to_string(DacapoInputFormat::EarthMlirText)) == "earth_mlir_text",
            "MLIR format string mismatch");
    require(std::string(to_string(DacapoInputFormat::HevmBinary)) == "hevm_binary",
            "HEVM format string mismatch");
    require(std::string(to_string(DacapoInputFormat::Json)) == "json",
            "JSON format string mismatch");
}

void test_empty_input_fails_clearly()
{
    const DacapoAdapterResult result = translate_dacapo_schedule(
        "", DacapoAdapterOptions{ DacapoInputFormat::EarthMlirText });
    require(!result.ok(), "empty input should fail");
    require_contains(result.format_diagnostics(), "input is empty");
}

void test_json_format_translates_internal_schedule()
{
    const char *json = R"json(
{
  "version": 1,
  "ops": [
    {"kind": "upload_cipher", "device": 0, "outputs": [1], "name": "input"},
    {"kind": "download", "device": 0, "inputs": [1]}
  ]
}
)json";

    const DacapoAdapterResult result =
        translate_dacapo_schedule(json, DacapoAdapterOptions{ DacapoInputFormat::Json });
    require(result.ok(), "JSON adapter input should translate:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 2, "translated op count mismatch");
    require(result.schedule.ops[0].kind == MgpuOpKind::UploadCipher, "translated op kind mismatch");
    require(result.schedule.ops[0].debug_name == "input", "translated debug name mismatch");
    require(result.schedule.ops[1].inputs[0].id == 1, "translated input id mismatch");
}

void test_json_format_reports_parse_diagnostics()
{
    const DacapoAdapterResult result = translate_dacapo_schedule(
        R"json({"version": 1, "ops": [{"kind": "not_an_op", "device": 0}]})json",
        DacapoAdapterOptions{ DacapoInputFormat::Json });

    require(!result.ok(), "invalid JSON adapter input should fail");
    require(result.schedule.ops.empty(), "failed JSON adapter should not produce a partial schedule");
    require_contains(result.format_diagnostics(), "JSON schedule /ops/0/kind");
    require_contains(result.format_diagnostics(), "unknown op kind");
}

void test_hevm_binary_translates_supported_ops()
{
    const std::string hevm = make_hevm_binary(
        1, 1, 4, 1, { 3 },
        {
            HevmOpRecord{ 0, 0, 0, 0 },
            HevmOpRecord{ 9, 1, 0, 0 },
            HevmOpRecord{ 7, 2, 1, 0 },
            HevmOpRecord{ 3, 3, 2, 0 },
        });

    const DacapoAdapterResult result = translate_dacapo_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary });

    require(result.ok(), "HEVM adapter input should translate:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 6, "translated HEVM op count mismatch");
    require(result.schedule.ops[0].kind == MgpuOpKind::UploadCipher, "HEVM arg op mismatch");
    require(result.schedule.ops[0].outputs[0].id == 1, "HEVM arg value id mismatch");
    require(result.schedule.ops[0].device_id == -1, "HEVM ops should be unassigned");
    require(result.schedule.ops[1].kind == MgpuOpKind::UploadPlain, "HEVM encode op mismatch");
    require(result.schedule.ops[1].outputs[0].id == 5, "HEVM plain value id mismatch");
    require(result.schedule.ops[2].kind == MgpuOpKind::MultiplyPlain, "HEVM mulcp mismatch");
    require(result.schedule.ops[2].inputs[0].id == 1, "HEVM mulcp cipher input mismatch");
    require(result.schedule.ops[2].inputs[1].id == 5, "HEVM mulcp plain input mismatch");
    require(result.schedule.ops[2].outputs[0].id == 2, "HEVM mulcp output mismatch");
    require(result.schedule.ops[3].kind == MgpuOpKind::AddPlain, "HEVM addcp mismatch");
    require(result.schedule.ops[3].inputs[0].id == 2, "HEVM addcp cipher input mismatch");
    require(result.schedule.ops[3].inputs[1].id == 5, "HEVM addcp plain input mismatch");
    require(result.schedule.ops[4].kind == MgpuOpKind::Rescale, "HEVM rescale mismatch");
    require(result.schedule.ops[5].kind == MgpuOpKind::Download, "HEVM result op mismatch");
    require(result.schedule.ops[5].inputs[0].id == 4, "HEVM result input mismatch");
}

void test_hevm_binary_reports_unsupported_opcode()
{
    const std::string hevm = make_hevm_binary(
        1, 1, 2, 0, { 1 },
        {
            HevmOpRecord{ 2, 1, 0, 0 },
        });

    const DacapoAdapterResult result = translate_dacapo_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary });

    require(!result.ok(), "unsupported HEVM opcode should fail");
    require(result.schedule.ops.empty(), "failed HEVM adapter should not produce a schedule");
    require_contains(result.format_diagnostics(), "unsupported HEVM opcode 2");
}

void test_hevm_binary_reports_bad_magic()
{
    std::string hevm = make_hevm_binary(1, 1, 1, 0, { 0 }, {});
    hevm[0] = '\0';

    const DacapoAdapterResult result = translate_dacapo_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary });

    require(!result.ok(), "bad HEVM magic should fail");
    require_contains(result.format_diagnostics(), "invalid HEVM magic number");
}

void test_unknown_dacapo_format_does_not_guess()
{
    const DacapoAdapterResult result = translate_dacapo_schedule(
        "module { func.func @main() }",
        DacapoAdapterOptions{ DacapoInputFormat::EarthMlirText });
    require(!result.ok(), "unimplemented adapter should fail");
    require_contains(result.format_diagnostics(), "translation is not implemented");
    require(result.schedule.ops.empty(), "failed adapter should not produce a schedule");
}

}  // namespace

int main()
{
    try
    {
        test_format_strings();
        test_empty_input_fails_clearly();
        test_json_format_translates_internal_schedule();
        test_json_format_reports_parse_diagnostics();
        test_hevm_binary_translates_supported_ops();
        test_hevm_binary_reports_unsupported_opcode();
        test_hevm_binary_reports_bad_magic();
        test_unknown_dacapo_format_does_not_guess();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu Dacapo adapter test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu Dacapo adapter tests passed\n";
    return EXIT_SUCCESS;
}
