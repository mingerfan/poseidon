#include "poseidon/mgpu/compiler/dacapo_adapter.h"
#include "poseidon/tests/mgpu/hevm_test_utils.h"

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
    const std::string hevm = test::make_hevm_binary(
        1, 1, 1, 1, { 0 },
        {
            test::HevmOpRecord{ 0, 0, 0, test::make_hevm_encode_attr(4, 40) },
            test::HevmOpRecord{ 9, 0, 0, 0 },
            test::HevmOpRecord{ 7, 0, 0, 0 },
            test::HevmOpRecord{ 3, 0, 0, 0 },
        },
        test::HevmConfigMetadata{
            { 45 },
            { 12 },
            { 40 },
            { 8 },
            16,
        });

    const DacapoAdapterResult result = translate_dacapo_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary });

    require(result.ok(), "HEVM adapter input should translate:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 6, "translated HEVM op count mismatch");
    require(result.schedule.ops[0].kind == MgpuOpKind::UploadCipher, "HEVM arg op mismatch");
    require(result.schedule.ops[0].outputs[0].id == 1, "HEVM arg value id mismatch");
    require(result.schedule.ops[0].device_id == -1, "HEVM ops should be unassigned");
    require(
        result.schedule.ops[0].integer_attributes.at("hevm_arg_index") == 0,
        "HEVM arg index metadata mismatch");
    require(
        result.schedule.ops[0].integer_attributes.at("hevm_arg_scale") == 45,
        "HEVM arg scale metadata mismatch");
    require(
        result.schedule.ops[0].integer_attributes.at("hevm_arg_level") == 12,
        "HEVM arg level metadata mismatch");
    require(
        result.schedule.ops[0].integer_attributes.at("hevm_init_level") == 16,
        "HEVM init level metadata mismatch");
    require(result.schedule.ops[1].kind == MgpuOpKind::UploadPlain, "HEVM encode op mismatch");
    require(result.schedule.ops[1].outputs[0].id == 2, "HEVM plain value id mismatch");
    require(
        result.schedule.ops[1].integer_attributes.at("encode_level") == 4,
        "HEVM encode level mismatch");
    require(
        result.schedule.ops[1].integer_attributes.at("encode_scale") == 40,
        "HEVM encode scale mismatch");
    require(
        result.schedule.ops[1].integer_attributes.at("hevm_plain_register") == 0,
        "HEVM plain register metadata mismatch");
    require(
        result.schedule.ops[1].integer_attributes.at("hevm_constant_index") == 0,
        "HEVM constant index metadata mismatch");
    require(result.schedule.ops[2].kind == MgpuOpKind::MultiplyPlain, "HEVM mulcp mismatch");
    require(result.schedule.ops[2].inputs[0].id == 1, "HEVM mulcp cipher input mismatch");
    require(result.schedule.ops[2].inputs[1].id == 2, "HEVM mulcp plain input mismatch");
    require(result.schedule.ops[2].outputs[0].id == 3, "HEVM mulcp output mismatch");
    require(result.schedule.ops[3].kind == MgpuOpKind::AddPlain, "HEVM addcp mismatch");
    require(result.schedule.ops[3].inputs[0].id == 3, "HEVM addcp cipher input mismatch");
    require(result.schedule.ops[3].inputs[1].id == 2, "HEVM addcp plain input mismatch");
    require(result.schedule.ops[3].outputs[0].id == 4, "HEVM addcp output mismatch");
    require(result.schedule.ops[4].kind == MgpuOpKind::Rescale, "HEVM rescale mismatch");
    require(result.schedule.ops[4].inputs[0].id == 4, "HEVM rescale input mismatch");
    require(result.schedule.ops[4].outputs[0].id == 5, "HEVM rescale output mismatch");
    require(result.schedule.ops[5].kind == MgpuOpKind::Download, "HEVM result op mismatch");
    require(result.schedule.ops[5].inputs[0].id == 5, "HEVM result input mismatch");
    require(
        result.schedule.ops[5].integer_attributes.at("hevm_result_index") == 0,
        "HEVM result index metadata mismatch");
    require(
        result.schedule.ops[5].integer_attributes.at("hevm_result_register") == 0,
        "HEVM result register metadata mismatch");
    require(
        result.schedule.ops[5].integer_attributes.at("hevm_result_scale") == 40,
        "HEVM result scale metadata mismatch");
    require(
        result.schedule.ops[5].integer_attributes.at("hevm_result_level") == 8,
        "HEVM result level metadata mismatch");
}

void test_hevm_binary_translates_rotate_attributes()
{
    const std::string hevm = test::make_hevm_binary(
        1, 1, 2, 0, { 1 },
        {
            test::HevmOpRecord{ 1, 1, 0, static_cast<std::uint16_t>(-3) },
        });

    const DacapoAdapterResult result = translate_dacapo_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary });

    require(result.ok(), "HEVM rotate should translate:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 3, "HEVM rotate op count mismatch");
    require(result.schedule.ops[1].kind == MgpuOpKind::Rotate, "HEVM rotate kind mismatch");
    require(result.schedule.ops[1].inputs[0].id == 1, "HEVM rotate input mismatch");
    require(result.schedule.ops[1].outputs[0].id == 2, "HEVM rotate output mismatch");
    require(
        result.schedule.ops[1].integer_attributes.at("rotate_step") == -3,
        "HEVM rotate_step mismatch");
}

void test_hevm_binary_translates_bootstrap_target_level()
{
    const std::string hevm = test::make_hevm_binary(
        1, 1, 2, 0, { 1 },
        {
            test::HevmOpRecord{ 10, 1, 0, 6 },
        });

    const DacapoAdapterResult result = translate_dacapo_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary });

    require(result.ok(), "HEVM bootstrap should translate:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 3, "HEVM bootstrap op count mismatch");
    require(
        result.schedule.ops[1].kind == MgpuOpKind::BootstrapFallback,
        "HEVM bootstrap kind mismatch");
    require(
        result.schedule.ops[1].integer_attributes.at("target_level") == 6,
        "HEVM bootstrap target level mismatch");
}

void test_hevm_binary_translates_negate()
{
    const std::string hevm = test::make_hevm_binary(
        1, 1, 2, 0, { 1 },
        {
            test::HevmOpRecord{ 2, 1, 0, 0 },
        });

    const DacapoAdapterResult result = translate_dacapo_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary });

    require(result.ok(), "HEVM negate should translate:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 3, "HEVM negate op count mismatch");
    require(result.schedule.ops[1].kind == MgpuOpKind::Negate, "HEVM negate kind mismatch");
    require(result.schedule.ops[1].inputs[0].id == 1, "HEVM negate input mismatch");
    require(result.schedule.ops[1].outputs[0].id == 2, "HEVM negate output mismatch");
}

void test_hevm_opcode_summary_counts_supported_and_unsupported_ops()
{
    const std::string hevm = test::make_hevm_binary(
        1, 1, 2, 1, { 1 },
        {
            test::HevmOpRecord{ 0, 0, 0, test::make_hevm_encode_attr(4, 40) },
            test::HevmOpRecord{ 4, 1, 0, 0 },
            test::HevmOpRecord{ 5, 1, 0, 0 },
            test::HevmOpRecord{ 9, 1, 0, 0 },
        });

    const DacapoHevmOpcodeSummary summary = summarize_hevm_opcodes(hevm);
    require(summary.ok(), "HEVM opcode summary should parse:\n" + summary.format_diagnostics());
    require(summary.operation_count == 4, "HEVM opcode operation count mismatch");
    require(summary.alloc_count == 0, "HEVM opcode alloc count mismatch");

    bool saw_encode = false;
    bool saw_modswitch = false;
    bool saw_upscale = false;
    bool saw_mulcp = false;
    for (const DacapoHevmOpcodeCount &count : summary.opcode_counts)
    {
        if (count.opcode == 0)
        {
            saw_encode = true;
            require(count.name == "Encode", "Encode opcode name mismatch");
            require(count.supported, "Encode should be marked supported");
        }
        if (count.opcode == 4)
        {
            saw_modswitch = true;
            require(count.name == "ModswitchC", "Modswitch opcode name mismatch");
            require(!count.supported, "Modswitch should be marked unsupported");
        }
        if (count.opcode == 5)
        {
            saw_upscale = true;
            require(count.name == "UpscaleC", "Upscale opcode name mismatch");
            require(!count.supported, "Upscale should be marked unsupported");
        }
        if (count.opcode == 9)
        {
            saw_mulcp = true;
            require(count.name == "MulCP", "MulCP opcode name mismatch");
            require(count.supported, "MulCP should be marked supported");
        }
    }

    require(saw_encode, "expected Encode opcode");
    require(saw_modswitch, "expected Modswitch opcode");
    require(saw_upscale, "expected Upscale opcode");
    require(saw_mulcp, "expected MulCP opcode");
}

void test_hevm_opcode_summary_reports_bad_magic()
{
    std::string hevm = test::make_hevm_binary(1, 1, 1, 0, { 0 }, {});
    hevm[0] = '\0';

    const DacapoHevmOpcodeSummary summary = summarize_hevm_opcodes(hevm);
    require(!summary.ok(), "bad HEVM magic should fail opcode summary");
    require_contains(summary.format_diagnostics(), "invalid HEVM magic number");
}

void test_hevm_binary_reports_unsupported_opcode()
{
    const std::string hevm = test::make_hevm_binary(
        1, 1, 2, 0, { 1 },
        {
            test::HevmOpRecord{ 4, 1, 0, 0 },
        });

    const DacapoAdapterResult result = translate_dacapo_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary });

    require(!result.ok(), "unsupported HEVM opcode should fail");
    require(result.schedule.ops.empty(), "failed HEVM adapter should not produce a schedule");
    require_contains(result.format_diagnostics(), "unsupported HEVM opcode 4");
}

void test_hevm_binary_reports_bad_magic()
{
    std::string hevm = test::make_hevm_binary(1, 1, 1, 0, { 0 }, {});
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
        test_hevm_binary_translates_rotate_attributes();
        test_hevm_binary_translates_bootstrap_target_level();
        test_hevm_binary_translates_negate();
        test_hevm_opcode_summary_counts_supported_and_unsupported_ops();
        test_hevm_opcode_summary_reports_bad_magic();
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
