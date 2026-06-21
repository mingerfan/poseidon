#include "poseidon/mgpu/runtime/poseidon_gpu_hevm_executor.h"

#include "poseidon/mgpu/runtime/poseidon_gpu_schedule_handler.h"
#include "poseidon/mgpu/runtime/static_schedule_executor.h"

#include <algorithm>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

void add_error(
    PoseidonGpuHevmExecutionResult &result, std::size_t op_index, std::string message)
{
    result.execution.errors.push_back(
        ScheduleExecutionError{ op_index, std::move(message) });
}

std::vector<int> schedule_devices(const MgpuSchedule &schedule)
{
    std::vector<int> devices;
    for (const MgpuOp &op : schedule.ops)
    {
        if (op.device_id < 0)
        {
            continue;
        }
        if (std::find(devices.begin(), devices.end(), op.device_id) == devices.end())
        {
            devices.push_back(op.device_id);
        }
    }
    return devices;
}

void upload_keys_for_schedule_devices(
    PoseidonGpuScheduleHandler &handler, const MgpuSchedule &schedule,
    const PoseidonGpuHevmExecutionOptions &options)
{
    if (options.relin_keys == nullptr && options.galois_keys == nullptr)
    {
        return;
    }

    for (const int device_id : schedule_devices(schedule))
    {
        handler.upload_keys_for_device(device_id, options.relin_keys, options.galois_keys);
    }
}

void bind_hevm_plan_inputs(
    PoseidonGpuScheduleHandler &handler, const HevmStaticExecutionPlan &plan,
    const std::vector<std::shared_ptr<const Ciphertext>> &cipher_inputs,
    const PoseidonGpuHevmExecutionOptions &options)
{
    bind_hevm_cipher_inputs(handler, plan.io_plan, cipher_inputs);
    bind_hevm_encoded_plain_inputs(handler, plan.encoded_plaintexts);
    upload_keys_for_schedule_devices(handler, plan.schedule, options);
}

}  // namespace

std::string PoseidonGpuHevmExecutionResult::format_errors() const
{
    return execution.format_errors();
}

PoseidonGpuHevmExecutionResult execute_hevm_static_plan_with_poseidon_gpu(
    const PoseidonContext &context, const HevmStaticExecutionPlan &plan,
    const std::vector<std::shared_ptr<const Ciphertext>> &cipher_inputs,
    const PoseidonGpuHevmExecutionOptions &options)
{
    PoseidonGpuHevmExecutionResult result;
    if (options.device_count <= 0)
    {
        add_error(result, 0, "device_count must be positive");
        return result;
    }

    try
    {
        PoseidonGpuScheduleHandler handler(context);
        bind_hevm_plan_inputs(handler, plan, cipher_inputs, options);
        ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ options.device_count });
        result.execution = interpreter.run(plan.schedule, handler);
        if (!result.execution.ok())
        {
            return result;
        }

        result.results = collect_hevm_results(handler, plan.io_plan);
    }
    catch (const std::exception &ex)
    {
        add_error(result, plan.schedule.ops.size(), ex.what());
    }
    catch (...)
    {
        add_error(result, plan.schedule.ops.size(), "unknown HEVM GPU execution error");
    }

    return result;
}

PoseidonGpuHevmExecutionResult execute_hevm_static_plan_with_poseidon_gpu(
    const PoseidonContext &context, const HevmStaticExecutionPlan &plan,
    const std::vector<std::shared_ptr<const Ciphertext>> &cipher_inputs, GpuComm &comm,
    const PoseidonGpuHevmExecutionOptions &options)
{
    PoseidonGpuHevmExecutionResult result;
    if (options.device_count <= 0)
    {
        add_error(result, 0, "device_count must be positive");
        return result;
    }

    try
    {
        PoseidonGpuScheduleHandler handler(context);
        bind_hevm_plan_inputs(handler, plan, cipher_inputs, options);
        StaticScheduleExecutor executor(
            comm, handler, StaticScheduleExecutorOptions{ options.device_count });
        result.execution = executor.run(plan.schedule);
        if (!result.execution.ok())
        {
            return result;
        }

        result.results = collect_hevm_results(handler, plan.io_plan);
    }
    catch (const std::exception &ex)
    {
        add_error(result, plan.schedule.ops.size(), ex.what());
    }
    catch (...)
    {
        add_error(result, plan.schedule.ops.size(), "unknown HEVM GPU execution error");
    }

    return result;
}

}  // namespace poseidon::mgpu
