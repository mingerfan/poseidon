#include "poseidon/mgpu/compiler/static_schedule_pipeline.h"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace poseidon::mgpu;

namespace
{

MgpuValueRef value(ValueId id)
{
    return MgpuValueRef{ id };
}

MgpuOp op(
    MgpuOpKind kind, int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs)
{
    return MgpuOp{ kind, device_id, std::move(inputs), std::move(outputs), {} };
}

MgpuOp op_with_attrs(
    MgpuOpKind kind, int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs, std::unordered_map<std::string, std::int64_t> attrs)
{
    MgpuOp result{ kind, device_id, std::move(inputs), std::move(outputs), {} };
    result.integer_attributes = std::move(attrs);
    return result;
}

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

MgpuSchedule make_resnet_like_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 1, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::MultiplyPlain, 1, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(4) }));
    schedule.ops.push_back(op(MgpuOpKind::Add, 1, { value(3), value(4) }, { value(5) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(5) }, {}));
    return schedule;
}

void test_pipeline_inserts_copies_and_dumps()
{
    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.emit_debug_dump = true;

    const StaticSchedulePipelineResult result =
        prepare_static_schedule(make_resnet_like_schedule(), options);

    require(result.ok(), "pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 9, "expected three inserted copies");
    require(result.schedule.ops[2].kind == MgpuOpKind::CopyCipher,
            "multiply input should be copied to device 1");
    require(result.schedule.ops[5].kind == MgpuOpKind::CopyCipher,
            "add input should be copied to device 1");
    require(result.schedule.ops[8].kind == MgpuOpKind::Download,
            "last op should remain download");
    require_contains(result.debug_dump, "mgpu.schedule");
    require_contains(result.debug_dump, "name=\"auto_copy\"");
    require_contains(result.debug_dump, "mgpu.download device=0");
}

void test_pipeline_reports_copy_insertion_errors()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 0, { value(99) }, { value(1) }));

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule);
    require(!result.ok(), "pipeline should fail for unknown input");
    require_contains(result.format_diagnostics(), "copy_insertion op #0");
    require_contains(result.format_diagnostics(), "unknown input value %99");
}

void test_pipeline_reports_verifier_errors()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Rotate, kUnassignedDevice, { value(1) }, { value(2) }));

    StaticSchedulePipelineOptions options;
    options.device_count = 1;

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(!result.ok(), "pipeline should fail when required static attributes are missing");
    require_contains(result.format_diagnostics(), "verify op #1");
    require_contains(result.format_diagnostics(), "missing integer attribute 'rotate_step'");
}

void test_pipeline_places_unassigned_ops_before_copy_insertion()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, kUnassignedDevice, {}, { value(2) }));
    schedule.ops.push_back(
        op(MgpuOpKind::MultiplyPlain, kUnassignedDevice, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(3) }, {}));

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.default_device = 1;

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(result.ok(), "pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 4, "no copies should be required after single-device placement");
    for (const MgpuOp &placed_op : result.schedule.ops)
    {
        require(placed_op.device_id == 1, "pipeline should place all ops on default device");
    }
}

void test_pipeline_preserves_integer_attributes()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op_with_attrs(
        MgpuOpKind::Rotate, kUnassignedDevice, { value(1) }, { value(2) },
        { { "rotate_step", 5 } }));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(2) }, {}));

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.default_device = 1;
    options.emit_debug_dump = true;

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(result.ok(), "pipeline failed:\n" + result.format_diagnostics());
    require(
        result.schedule.ops[1].integer_attributes.at("rotate_step") == 5,
        "pipeline should preserve rotate_step");
    require_contains(result.debug_dump, "attrs={rotate_step=5}");
}

void test_pipeline_prepares_dacapo_json_input()
{
    const char *json = R"json(
{
  "version": 1,
  "ops": [
    {"kind": "upload_cipher", "device": 0, "outputs": [1]},
    {"kind": "upload_plain", "device": 1, "outputs": [2]},
    {"kind": "multiply_plain", "device": 1, "inputs": [1, 2], "outputs": [3]},
    {"kind": "download", "device": 0, "inputs": [3]}
  ]
}
)json";

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.emit_debug_dump = true;

    const StaticSchedulePipelineResult result = prepare_dacapo_static_schedule(
        json, DacapoAdapterOptions{ DacapoInputFormat::Json }, options);

    require(result.ok(), "Dacapo JSON pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 6, "expected two inserted copies");
    require(result.schedule.ops[2].kind == MgpuOpKind::CopyCipher,
            "multiply input should be copied to device 1");
    require(result.schedule.ops[5].kind == MgpuOpKind::Download,
            "last op should remain download");
    require_contains(result.debug_dump, "mgpu.copy_cipher");
    require_contains(result.debug_dump, "mgpu.download device=0");
}

void test_pipeline_prepares_hevm_binary_input()
{
    const std::string hevm = make_hevm_binary(
        1, 1, 3, 1, { 2 },
        {
            HevmOpRecord{ 0, 0, 0, static_cast<std::uint16_t>((4 << 10) + 40) },
            HevmOpRecord{ 9, 1, 0, 0 },
            HevmOpRecord{ 1, 2, 1, static_cast<std::uint16_t>(-1) },
        });

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.policy = StaticPlacementPolicy::RoundRobinCompute;
    options.emit_debug_dump = true;

    const StaticSchedulePipelineResult result = prepare_dacapo_static_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary }, options);

    require(result.ok(), "Dacapo HEVM pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 6, "expected one inserted copy");
    require(result.schedule.ops[1].kind == MgpuOpKind::UploadPlain, "expected HEVM encode op");
    require(
        result.schedule.ops[1].integer_attributes.at("encode_level") == 4,
        "HEVM encode level should survive pipeline");
    require(
        result.schedule.ops[1].integer_attributes.at("encode_scale") == 40,
        "HEVM encode scale should survive pipeline");
    require(result.schedule.ops[3].kind == MgpuOpKind::CopyCipher,
            "rotate should receive copied ciphertext");
    require(
        result.schedule.ops[4].integer_attributes.at("rotate_step") == -1,
        "HEVM rotate_step should survive copy insertion");
    require(result.schedule.ops[5].kind == MgpuOpKind::Download,
            "last HEVM op should remain download");
    require(
        result.schedule.ops[4].inputs[0].id == result.schedule.ops[3].outputs[0].id,
        "rotate should consume the copied mulcp result");
    require_contains(result.debug_dump, "mgpu.multiply_plain");
    require_contains(result.debug_dump, "attrs={rotate_step=-1}");
    require_contains(result.debug_dump, "name=\"auto_copy\"");
}

void test_pipeline_reports_dacapo_adapter_errors()
{
    const StaticSchedulePipelineResult result = prepare_dacapo_static_schedule(
        R"json({"version": 1, "ops": [{"kind": "bad", "device": 0}]})json",
        DacapoAdapterOptions{ DacapoInputFormat::Json });

    require(!result.ok(), "invalid Dacapo JSON pipeline input should fail");
    require_contains(result.format_diagnostics(), "dacapo_adapter");
    require_contains(result.format_diagnostics(), "unknown op kind");
    require(result.schedule.ops.empty(), "failed Dacapo adapter should not produce a schedule");
}

}  // namespace

int main()
{
    try
    {
        test_pipeline_inserts_copies_and_dumps();
        test_pipeline_reports_copy_insertion_errors();
        test_pipeline_reports_verifier_errors();
        test_pipeline_places_unassigned_ops_before_copy_insertion();
        test_pipeline_preserves_integer_attributes();
        test_pipeline_prepares_dacapo_json_input();
        test_pipeline_prepares_hevm_binary_input();
        test_pipeline_reports_dacapo_adapter_errors();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu static pipeline test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu static pipeline tests passed\n";
    return EXIT_SUCCESS;
}
