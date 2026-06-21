#include "poseidon/mgpu/compiler/static_placement.h"
#include "poseidon/mgpu/compiler/schedule_verifier.h"

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
    std::vector<MgpuValueRef> outputs)
{
    return MgpuOp{ kind, device_id, std::move(inputs), std::move(outputs), {} };
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

void test_single_device_places_unassigned_ops()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, kUnassignedDevice, {}, { value(2) }));
    schedule.ops.push_back(
        op(MgpuOpKind::MultiplyPlain, kUnassignedDevice, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(3) }, {}));

    StaticPlacementOptions options;
    options.device_count = 2;
    options.default_device = 1;

    const StaticPlacementResult result = place_static_schedule(schedule, options);
    require(result.ok(), "single-device placement failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 4, "placed op count mismatch");
    for (const MgpuOp &placed_op : result.schedule.ops)
    {
        require(placed_op.device_id == 1, "all ops should be placed on default device");
    }

    const ScheduleVerificationResult verification =
        verify_schedule(result.schedule, ScheduleVerifierOptions{ 2 });
    require(verification.ok(), "placed schedule should verify:\n" + verification.format_errors());
}

void test_preserves_existing_devices()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, kUnassignedDevice, { value(1) }, { value(2) }));

    StaticPlacementOptions options;
    options.device_count = 2;
    options.default_device = 1;

    const StaticPlacementResult result = place_static_schedule(schedule, options);
    require(result.ok(), "placement failed:\n" + result.format_diagnostics());
    require(result.schedule.ops[0].device_id == 0, "assigned upload should stay on device 0");
    require(result.schedule.ops[1].device_id == 1, "unassigned compute should use default device");
}

void test_round_robin_compute_policy()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Relinearize, kUnassignedDevice, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, kUnassignedDevice, { value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Rotate, kUnassignedDevice, { value(3) }, { value(4) }));

    StaticPlacementOptions options;
    options.device_count = 2;
    options.policy = StaticPlacementPolicy::RoundRobinCompute;

    const StaticPlacementResult result = place_static_schedule(schedule, options);
    require(result.ok(), "round-robin placement failed:\n" + result.format_diagnostics());
    require(result.schedule.ops[1].device_id == 0, "first compute should use device 0");
    require(result.schedule.ops[2].device_id == 1, "second compute should use device 1");
    require(result.schedule.ops[3].device_id == 0, "third compute should wrap to device 0");
}

void test_round_robin_compute_starts_at_default_device()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Relinearize, kUnassignedDevice, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, kUnassignedDevice, { value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Rotate, kUnassignedDevice, { value(3) }, { value(4) }));

    StaticPlacementOptions options;
    options.device_count = 2;
    options.default_device = 1;
    options.policy = StaticPlacementPolicy::RoundRobinCompute;

    const StaticPlacementResult result = place_static_schedule(schedule, options);
    require(result.ok(), "round-robin placement failed:\n" + result.format_diagnostics());
    require(result.schedule.ops[0].device_id == 1, "upload should use default device");
    require(result.schedule.ops[1].device_id == 1, "first compute should start at default device");
    require(result.schedule.ops[2].device_id == 0, "second compute should wrap to device 0");
    require(result.schedule.ops[3].device_id == 1, "third compute should return to device 1");
}

void test_unassigned_copy_is_diagnostic()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, kUnassignedDevice, { value(1) }, { value(2) }));

    StaticPlacementOptions options;
    options.device_count = 2;

    const StaticPlacementResult result = place_static_schedule(schedule, options);
    require(!result.ok(), "unassigned copy destination should fail placement");
    require_contains(
        result.format_diagnostics(),
        "copy operation destination device must be assigned before placement");
}

void test_invalid_existing_device_is_diagnostic()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 3, {}, { value(1) }));

    StaticPlacementOptions options;
    options.device_count = 2;

    const StaticPlacementResult result = place_static_schedule(schedule, options);
    require(!result.ok(), "invalid existing device should fail placement");
    require_contains(result.format_diagnostics(), "invalid assigned device 3");
}

}  // namespace

int main()
{
    try
    {
        test_single_device_places_unassigned_ops();
        test_preserves_existing_devices();
        test_round_robin_compute_policy();
        test_round_robin_compute_starts_at_default_device();
        test_unassigned_copy_is_diagnostic();
        test_invalid_existing_device_is_diagnostic();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu static placement test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu static placement tests passed\n";
    return EXIT_SUCCESS;
}
