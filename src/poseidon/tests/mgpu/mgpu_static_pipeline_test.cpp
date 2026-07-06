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

void test_pipeline_single_device_places_unassigned_ops()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, kUnassignedDevice, {}, { value(2) }));
    schedule.ops.push_back(
        op(MgpuOpKind::MultiplyPlain, kUnassignedDevice, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(3) }, {}));

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.scheduler.default_device = 1;

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(result.ok(), "pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 4, "single-device path should not add copies");
    for (const MgpuOp &placed_op : result.schedule.ops)
    {
        require(placed_op.device_id == 1, "pipeline should place all ops on default device");
    }
}

void test_pipeline_reports_scheduler_errors()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 0, { value(99) }, { value(1) }));

    StaticSchedulePipelineOptions options;
    options.scheduler.kind = StaticSchedulerKind::GreedyReady;

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(!result.ok(), "pipeline should fail for unknown input");
    require_contains(result.format_diagnostics(), "scheduler op #0");
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

void test_pipeline_greedy_ready_splits_branches_and_dumps()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op_with_attrs(
        MgpuOpKind::Rotate, kUnassignedDevice, { value(1) }, { value(2) },
        { { "rotate_step", 5 } }));
    schedule.ops.push_back(op(MgpuOpKind::Multiply, kUnassignedDevice, { value(1), value(1) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(2) }, {}));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(3) }, {}));

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.scheduler.kind = StaticSchedulerKind::GreedyReady;
    options.scheduler.compute_devices = { 0, 1 };
    options.emit_debug_dump = true;

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(result.ok(), "pipeline failed:\n" + result.format_diagnostics());
    require(result.preflight.parallelism_found, "greedy-ready preflight should report parallelism");
    require_contains(result.debug_dump, "mgpu.schedule");
    require_contains(result.debug_dump, "greedy_ready_copy");
    require_contains(result.debug_dump, "attrs={rotate_step=5}");
}

void test_pipeline_unimplemented_scheduler_is_diagnostic()
{
    StaticSchedulePipelineOptions options;
    options.scheduler.kind = StaticSchedulerKind::ValueAwarePeft;

    const StaticSchedulePipelineResult result = prepare_static_schedule(MgpuSchedule{}, options);
    require(!result.ok(), "unimplemented scheduler should fail");
    require_contains(result.format_diagnostics(), "not implemented yet");
}

}  // namespace

int main()
{
    try
    {
        test_pipeline_single_device_places_unassigned_ops();
        test_pipeline_reports_scheduler_errors();
        test_pipeline_reports_verifier_errors();
        test_pipeline_greedy_ready_splits_branches_and_dumps();
        test_pipeline_unimplemented_scheduler_is_diagnostic();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu static pipeline test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu static pipeline tests passed\n";
    return EXIT_SUCCESS;
}
