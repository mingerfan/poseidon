#include "poseidon/mgpu/runtime/poseidon_gpu_schedule_preflight.h"

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

MgpuOp op_with_attr(
    MgpuOpKind kind, int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs, const std::string &attr_name,
    std::int64_t attr_value)
{
    MgpuOp result{ kind, device_id, std::move(inputs), std::move(outputs), {} };
    result.integer_attributes.emplace(attr_name, attr_value);
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

MgpuSchedule make_supported_single_device_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::AddPlain, 0, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 0, { value(3) }, { value(4) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(4) }, {}));
    return schedule;
}

void test_supported_schedule_ok()
{
    PoseidonGpuSchedulePreflightOptions options;
    options.device_count = 1;
    const PoseidonGpuSchedulePreflightResult result =
        preflight_poseidon_gpu_schedule(make_supported_single_device_schedule(), options);

    require(result.ok(), "supported schedule should pass preflight:\n" +
                             result.format_diagnostics());
    require(!result.requires_comm, "single-device schedule should not require comm");
    require(!result.requires_relin_keys, "schedule should not require relin keys");
    require(!result.requires_galois_keys, "schedule should not require galois keys");
    require(result.devices.size() == 1 && result.devices[0] == 0, "device summary mismatch");

    const std::string text = dump_poseidon_gpu_schedule_preflight(result);
    require_contains(text, "status: ok");
    require_contains(text, "requires_comm: false");
}

void test_copy_requires_comm_unless_provided()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 1, { value(2) }, {}));

    PoseidonGpuSchedulePreflightOptions options;
    options.device_count = 2;
    PoseidonGpuSchedulePreflightResult result =
        preflight_poseidon_gpu_schedule(schedule, options);
    require(!result.ok(), "copy schedule without comm should fail preflight");
    require(result.requires_comm, "copy schedule should require comm");
    require_contains(result.format_diagnostics(), "communication layer");

    options.copy_ops_have_comm = true;
    result = preflight_poseidon_gpu_schedule(schedule, options);
    require(result.ok(), "copy schedule with comm should pass:\n" + result.format_diagnostics());
    require(result.requires_comm, "copy schedule should still report comm requirement");
}

void test_keys_are_reported()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Multiply, 0, { value(1), value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Relinearize, 0, { value(2) }, { value(3) }));
    schedule.ops.push_back(op_with_attr(
        MgpuOpKind::Rotate, 0, { value(3) }, { value(4) }, "rotate_step", 1));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(4) }, {}));

    PoseidonGpuSchedulePreflightOptions options;
    options.device_count = 1;
    PoseidonGpuSchedulePreflightResult result =
        preflight_poseidon_gpu_schedule(schedule, options);
    require(!result.ok(), "missing keys should fail preflight");
    require(result.requires_relin_keys, "schedule should require relin keys");
    require(result.requires_galois_keys, "schedule should require galois keys");
    require_contains(result.format_diagnostics(), "RelinKeys");
    require_contains(result.format_diagnostics(), "GaloisKeys");

    options.relin_keys_available = true;
    options.galois_keys_available = true;
    result = preflight_poseidon_gpu_schedule(schedule, options);
    require(result.ok(), "schedule with keys should pass:\n" + result.format_diagnostics());
}

void test_unsupported_bootstrap_and_bad_devices()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, -1, {}, { value(1) }));
    schedule.ops.push_back(
        op(MgpuOpKind::BootstrapFallback, 3, { value(1) }, { value(2) }));

    PoseidonGpuSchedulePreflightOptions options;
    options.device_count = 2;
    const PoseidonGpuSchedulePreflightResult result =
        preflight_poseidon_gpu_schedule(schedule, options);
    require(!result.ok(), "bad schedule should fail preflight");
    require_contains(result.format_diagnostics(), "unassigned device");
    require_contains(result.format_diagnostics(), "outside device_count");
    require_contains(result.format_diagnostics(), "bootstrap fallback is not implemented");
}

void test_json_dump()
{
    MgpuSchedule schedule = make_supported_single_device_schedule();
    schedule.ops.push_back(op_with_attr(
        MgpuOpKind::Rotate, 0, { value(4) }, { value(5) }, "rotate_step", -1));

    PoseidonGpuSchedulePreflightOptions options;
    options.device_count = 1;
    const PoseidonGpuSchedulePreflightResult result =
        preflight_poseidon_gpu_schedule(schedule, options);

    const std::string json = poseidon_gpu_schedule_preflight_to_json(result);
    require_contains(json, "\"ok\": false");
    require_contains(json, "\"requires_galois_keys\": true");
    require_contains(json, "GaloisKeys");
}

}  // namespace

int main()
{
    try
    {
        test_supported_schedule_ok();
        test_copy_requires_comm_unless_provided();
        test_keys_are_reported();
        test_unsupported_bootstrap_and_bad_devices();
        test_json_dump();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu Poseidon GPU preflight test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu Poseidon GPU preflight tests passed\n";
    return EXIT_SUCCESS;
}
