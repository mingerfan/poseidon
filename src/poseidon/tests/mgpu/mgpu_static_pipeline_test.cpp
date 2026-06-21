#include "poseidon/mgpu/compiler/static_schedule_pipeline.h"

#include <cstdlib>
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
        test_pipeline_places_unassigned_ops_before_copy_insertion();
        test_pipeline_preserves_integer_attributes();
        test_pipeline_prepares_dacapo_json_input();
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
