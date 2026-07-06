#include "poseidon/mgpu/comm/topology.h"
#include "poseidon/mgpu/compiler/schedule_verifier.h"
#include "poseidon/mgpu/compiler/scheduler/static_scheduler.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
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
    std::vector<MgpuValueRef> outputs, std::string name = {})
{
    return MgpuOp{ kind, device_id, std::move(inputs), std::move(outputs), std::move(name) };
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

std::size_t count_kind(const MgpuSchedule &schedule, MgpuOpKind kind)
{
    std::size_t count = 0;
    for (const MgpuOp &op : schedule.ops)
    {
        if (op.kind == kind)
        {
            ++count;
        }
    }
    return count;
}

void require_verifies(const MgpuSchedule &schedule, int device_count)
{
    const ScheduleVerificationResult verification =
        verify_schedule(schedule, ScheduleVerifierOptions{ device_count });
    require(verification.ok(), "schedule failed verification:\n" + verification.format_errors());
}

StaticSchedulerOptions greedy_options()
{
    StaticSchedulerOptions options;
    options.kind = StaticSchedulerKind::GreedyReady;
    options.device_count = 2;
    options.compute_devices = { 0, 1 };
    options.upload_device = 0;
    options.download_device = 0;
    return options;
}

void test_single_device_scheduler_uses_existing_runtime_shape()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, kUnassignedDevice, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(2) }, {}));

    StaticSchedulerOptions options;
    options.kind = StaticSchedulerKind::SingleDevice;
    options.device_count = 2;
    options.default_device = 1;

    const StaticSchedulingResult result = schedule_static(schedule, options);
    require(result.ok(), "single-device scheduler failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 3, "single-device scheduler should not add copies");
    require(result.schedule.ops[1].device_id == 1, "compute should use default device");
    require_verifies(result.schedule, 2);
}

void test_greedy_ready_keeps_linear_chain_on_one_gpu()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, kUnassignedDevice, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Negate, kUnassignedDevice, { value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(3) }, {}));

    const StaticSchedulingResult result = schedule_static(schedule, greedy_options());
    require(result.ok(), "greedy scheduler failed:\n" + result.format_diagnostics());
    require(count_kind(result.schedule, MgpuOpKind::CopyCipher) == 0,
            "linear chain should not require cross-GPU copies");
    require_verifies(result.schedule, 2);
}

void test_greedy_ready_splits_fanout_branches()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Rotate, kUnassignedDevice, { value(1) }, { value(2) }));
    schedule.ops.back().integer_attributes.emplace("rotate_step", 1);
    schedule.ops.push_back(op(MgpuOpKind::Multiply, kUnassignedDevice, { value(1), value(1) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(2) }, {}));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(3) }, {}));

    const StaticSchedulingResult result = schedule_static(schedule, greedy_options());
    require(result.ok(), "greedy scheduler failed:\n" + result.format_diagnostics());
    require(count_kind(result.schedule, MgpuOpKind::CopyCipher) >= 1,
            "fanout split should copy the shared ciphertext");
    require(result.preflight.parallelism_found, "fanout schedule should report parallelism");
    require(result.preflight.estimated_speedup > 1.0, "fanout should estimate speedup");
    require_verifies(result.schedule, 2);

    const MgpuCommunicationPlan plan =
        plan_schedule_communication(result.schedule, make_single_node_topology(2));
    require(plan.ok(), "communication plan should verify:\n" + plan.format_diagnostics());
}

void test_greedy_ready_plaintext_preload_uses_upload_not_copy_plain()
{
    MgpuSchedule schedule;
    MgpuOp plain = op(MgpuOpKind::UploadPlain, kUnassignedDevice, {}, { value(1) }, "plain");
    plain.integer_attributes.emplace("hevm_constant_index", 7);
    plain.integer_attributes.emplace("hevm_plain_register", 0);
    plain.integer_attributes.emplace("encode_level", 1);
    plain.integer_attributes.emplace("encode_scale", 40);
    schedule.ops.push_back(std::move(plain));
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::MultiplyPlain, kUnassignedDevice, { value(2), value(1) }, { value(4) }));
    schedule.ops.push_back(op(MgpuOpKind::MultiplyPlain, kUnassignedDevice, { value(3), value(1) }, { value(5) }));

    const StaticSchedulingResult result = schedule_static(schedule, greedy_options());
    require(result.ok(), "greedy scheduler failed:\n" + result.format_diagnostics());
    require(count_kind(result.schedule, MgpuOpKind::CopyPlain) == 0,
            "plaintext should be preloaded, not copied GPU-to-GPU");
    require(count_kind(result.schedule, MgpuOpKind::UploadPlain) >= 2,
            "plaintext fanout should create multiple preload uploads");
    require_verifies(result.schedule, 2);
}

void test_latency_table_parser()
{
    const char json[] =
        R"({"latencyTable":{"add_single":[2],"mul_double":[697],"rotate_single":[685]}})";
    const LatencyTableParseResult parsed = parse_latency_table_json(json);
    require(parsed.ok(), "latency table should parse:\n" + parsed.format_diagnostics());
    require(parsed.table.latency_for(MgpuOpKind::Multiply, 0) == 697.0e-4,
            "latency unit conversion mismatch");
}

void test_unimplemented_scheduler_reports_diagnostic()
{
    StaticSchedulerOptions options;
    options.kind = StaticSchedulerKind::ValueAwareHeft;

    const StaticSchedulingResult result = schedule_static(MgpuSchedule{}, options);
    require(!result.ok(), "HEFT placeholder should fail until implemented");
    require_contains(result.format_diagnostics(), "not implemented yet");
}

}  // namespace

int main()
{
    try
    {
        test_single_device_scheduler_uses_existing_runtime_shape();
        test_greedy_ready_keeps_linear_chain_on_one_gpu();
        test_greedy_ready_splits_fanout_branches();
        test_greedy_ready_plaintext_preload_uses_upload_not_copy_plain();
        test_latency_table_parser();
        test_unimplemented_scheduler_reports_diagnostic();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu static scheduler test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu static scheduler tests passed\n";
    return EXIT_SUCCESS;
}
