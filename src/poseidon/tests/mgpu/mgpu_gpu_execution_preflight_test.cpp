#include "poseidon/mgpu/runtime/gpu_execution_preflight.h"

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

MgpuOp rotate_op(
    int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs, std::int64_t step)
{
    MgpuOp result{ MgpuOpKind::Rotate, device_id, std::move(inputs), std::move(outputs), {} };
    result.integer_attributes.emplace("rotate_step", step);
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

MgpuSchedule make_single_device_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::MultiplyPlain, 0, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(3) }, {}));
    return schedule;
}

MgpuSchedule make_copy_and_rotate_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));
    schedule.ops.push_back(rotate_op(1, { value(2) }, { value(3) }, 1));
    schedule.ops.push_back(op(MgpuOpKind::Download, 1, { value(3) }, {}));
    return schedule;
}

void test_supported_single_device_schedule_passes()
{
    PoseidonGpuExecutionPreflightOptions options;
    options.device_count = 1;
    const PoseidonGpuExecutionPreflightResult result =
        preflight_poseidon_gpu_execution_plan(make_single_device_schedule(), options);

    require(result.ok(), "single-device schedule should pass:\n" + result.format_diagnostics());
    require(result.schedule_verification.ok(), "schedule verifier should pass");
    require(result.poseidon_gpu_preflight.ok(), "Poseidon GPU preflight should pass");
    require(!result.communication_plan_evaluated, "communication plan should not run by default");

    const std::string text = dump_poseidon_gpu_execution_preflight(result);
    require_contains(text, "status: ok");
    require_contains(text, "communication_plan: not_run");

    const std::string json = poseidon_gpu_execution_preflight_to_json(result);
    require_contains(json, "\"ok\": true");
    require_contains(json, "\"schedule_verification\"");
}

void test_verifier_errors_are_reported()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 1, { value(1) }, { value(2) }));

    PoseidonGpuExecutionPreflightOptions options;
    options.device_count = 2;
    const PoseidonGpuExecutionPreflightResult result =
        preflight_poseidon_gpu_execution_plan(schedule, options);

    require(!result.ok(), "missing copy should fail execution preflight");
    require_contains(result.format_diagnostics(), "schedule_verification");
    require_contains(result.format_diagnostics(), "input value %1 is on device 0");
}

void test_keys_and_communication_execution_are_reported()
{
    PoseidonGpuExecutionPreflightOptions options;
    options.device_count = 2;
    options.copy_ops_have_comm = true;
    options.check_communication_plan = true;
    options.topology = make_single_node_topology(2);
    options.check_communication_execution = true;
    options.communication_execution = MgpuCommunicationExecutionOptions{
        true,
        false,
        false,
    };

    const PoseidonGpuExecutionPreflightResult result =
        preflight_poseidon_gpu_execution_plan(make_copy_and_rotate_schedule(), options);

    require(!result.ok(), "missing GaloisKeys and CUDA peer backend should fail");
    require(result.schedule_verification.ok(), "schedule verifier should pass");
    require(!result.poseidon_gpu_preflight.ok(), "missing GaloisKeys should fail");
    require(result.communication_plan_evaluated, "communication plan should run");
    require(
        result.communication_execution_preflight_evaluated,
        "communication execution preflight should run");
    require_contains(result.format_diagnostics(), "GaloisKeys");
    require_contains(result.format_diagnostics(), "CUDA peer");

    options.galois_keys_available = true;
    options.communication_execution.cuda_peer_available = true;
    const PoseidonGpuExecutionPreflightResult ready =
        preflight_poseidon_gpu_execution_plan(make_copy_and_rotate_schedule(), options);
    require(ready.ok(), "declared keys and CUDA peer backend should pass:\n" +
                            ready.format_diagnostics());
}

}  // namespace

int main()
{
    try
    {
        test_supported_single_device_schedule_passes();
        test_verifier_errors_are_reported();
        test_keys_and_communication_execution_are_reported();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu GPU execution preflight test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu GPU execution preflight tests passed\n";
    return EXIT_SUCCESS;
}
