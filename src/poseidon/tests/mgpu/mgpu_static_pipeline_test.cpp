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

void test_pipeline_round_robin_compute_starts_at_default_device()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, kUnassignedDevice, {}, { value(2) }));
    schedule.ops.push_back(
        op(MgpuOpKind::MultiplyPlain, kUnassignedDevice, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, kUnassignedDevice, { value(3) }, { value(4) }));

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.default_device = 1;
    options.placement.policy = StaticPlacementPolicy::RoundRobinCompute;

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(result.ok(), "pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops[0].device_id == 1, "upload cipher should use default device");
    require(result.schedule.ops[1].device_id == 1, "upload plain should use default device");
    require(result.schedule.ops[2].kind == MgpuOpKind::MultiplyPlain,
            "first compute should not need copies on default device");
    require(result.schedule.ops[2].device_id == 1,
            "first round-robin compute should start at default device");
    require(result.schedule.ops[3].kind == MgpuOpKind::CopyCipher,
            "second compute should receive copied input");
    require(result.schedule.ops[4].kind == MgpuOpKind::Rescale,
            "second compute should follow inserted copy");
    require(result.schedule.ops[4].device_id == 0,
            "second round-robin compute should wrap to device 0");
}

void test_pipeline_round_robin_uses_explicit_compute_device_order()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, kUnassignedDevice, {}, { value(2) }));
    schedule.ops.push_back(
        op(MgpuOpKind::MultiplyPlain, kUnassignedDevice, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, kUnassignedDevice, { value(3) }, { value(4) }));

    StaticSchedulePipelineOptions options;
    options.device_count = 3;
    options.placement.default_device = 1;
    options.placement.policy = StaticPlacementPolicy::RoundRobinCompute;
    options.placement.compute_devices = { 2, 0 };

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(result.ok(), "pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 7, "expected copies into explicit compute devices");
    require(result.schedule.ops[0].device_id == 1, "upload cipher should use default device");
    require(result.schedule.ops[1].device_id == 1, "upload plain should use default device");
    require(result.schedule.ops[2].kind == MgpuOpKind::CopyCipher,
            "first compute should receive copied ciphertext");
    require(result.schedule.ops[2].device_id == 2,
            "cipher copy should target first explicit compute device");
    require(result.schedule.ops[3].kind == MgpuOpKind::CopyPlain,
            "first compute should receive copied plaintext");
    require(result.schedule.ops[3].device_id == 2,
            "plain copy should target first explicit compute device");
    require(result.schedule.ops[4].kind == MgpuOpKind::MultiplyPlain,
            "first compute should follow inserted copies");
    require(result.schedule.ops[4].device_id == 2,
            "first compute should use first explicit compute device");
    require(result.schedule.ops[5].kind == MgpuOpKind::CopyCipher,
            "second compute should receive copied ciphertext");
    require(result.schedule.ops[5].device_id == 0,
            "second compute copy should target second explicit compute device");
    require(result.schedule.ops[6].kind == MgpuOpKind::Rescale,
            "second compute should follow inserted copy");
    require(result.schedule.ops[6].device_id == 0,
            "second compute should use second explicit compute device");
}

void test_pipeline_inserts_copy_to_explicit_download_device()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, kUnassignedDevice, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(2) }, {}));

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.default_device = 0;
    options.placement.policy = StaticPlacementPolicy::RoundRobinCompute;
    options.placement.compute_devices = { 1 };
    options.placement.download_device = 0;

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(result.ok(), "pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 5, "expected compute and download copies");
    require(result.schedule.ops[0].kind == MgpuOpKind::UploadCipher,
            "first op should upload input");
    require(result.schedule.ops[0].device_id == 0, "upload should use default device");
    require(result.schedule.ops[1].kind == MgpuOpKind::CopyCipher,
            "compute should receive copied input");
    require(result.schedule.ops[1].device_id == 1, "compute input copy should target device 1");
    require(result.schedule.ops[2].kind == MgpuOpKind::Rescale,
            "compute should follow its input copy");
    require(result.schedule.ops[2].device_id == 1, "compute should run on device 1");
    require(result.schedule.ops[3].kind == MgpuOpKind::CopyCipher,
            "download should receive copied result");
    require(result.schedule.ops[3].device_id == 0, "download copy should target device 0");
    require(result.schedule.ops[4].kind == MgpuOpKind::Download,
            "download should follow result copy");
    require(result.schedule.ops[4].device_id == 0, "download should run on device 0");
}

void test_pipeline_inserts_copies_from_explicit_upload_device()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, kUnassignedDevice, {}, { value(2) }));
    schedule.ops.push_back(
        op(MgpuOpKind::MultiplyPlain, kUnassignedDevice, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(3) }, {}));

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.default_device = 0;
    options.placement.upload_device = 1;
    options.placement.download_device = 1;

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(result.ok(), "pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 7, "expected copies from upload and to download devices");
    require(result.schedule.ops[0].kind == MgpuOpKind::UploadCipher,
            "first op should upload cipher");
    require(result.schedule.ops[0].device_id == 1, "cipher upload should use upload device");
    require(result.schedule.ops[1].kind == MgpuOpKind::UploadPlain,
            "second op should upload plain");
    require(result.schedule.ops[1].device_id == 1, "plain upload should use upload device");
    require(result.schedule.ops[2].kind == MgpuOpKind::CopyCipher,
            "compute should receive copied cipher input");
    require(result.schedule.ops[2].device_id == 0, "cipher input copy should target compute device");
    require(result.schedule.ops[3].kind == MgpuOpKind::CopyPlain,
            "compute should receive copied plain input");
    require(result.schedule.ops[3].device_id == 0, "plain input copy should target compute device");
    require(result.schedule.ops[4].kind == MgpuOpKind::MultiplyPlain,
            "compute should follow input copies");
    require(result.schedule.ops[4].device_id == 0, "compute should use default device");
    require(result.schedule.ops[5].kind == MgpuOpKind::CopyCipher,
            "download should receive copied result");
    require(result.schedule.ops[5].device_id == 1, "result copy should target download device");
    require(result.schedule.ops[6].kind == MgpuOpKind::Download,
            "download should follow result copy");
    require(result.schedule.ops[6].device_id == 1, "download should use download device");
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

}  // namespace

int main()
{
    try
    {
        test_pipeline_inserts_copies_and_dumps();
        test_pipeline_reports_copy_insertion_errors();
        test_pipeline_reports_verifier_errors();
        test_pipeline_places_unassigned_ops_before_copy_insertion();
        test_pipeline_round_robin_compute_starts_at_default_device();
        test_pipeline_round_robin_uses_explicit_compute_device_order();
        test_pipeline_inserts_copy_to_explicit_download_device();
        test_pipeline_inserts_copies_from_explicit_upload_device();
        test_pipeline_preserves_integer_attributes();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu static pipeline test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu static pipeline tests passed\n";
    return EXIT_SUCCESS;
}
