#include "poseidon/mgpu/runtime/static_schedule_executor.h"

#include "poseidon/mgpu/runtime/comm_schedule_handler.h"

namespace poseidon::mgpu
{

StaticScheduleExecutor::StaticScheduleExecutor(
    GpuComm &comm, ScheduleOpHandler &non_copy_handler,
    StaticScheduleExecutorOptions options)
    : comm_(comm), non_copy_handler_(non_copy_handler), options_(options)
{
}

ScheduleExecutionResult StaticScheduleExecutor::run(const MgpuSchedule &schedule) const
{
    CopyDispatchingScheduleHandler copy_handler(comm_, &non_copy_handler_);
    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ options_.device_count });
    return interpreter.run(schedule, copy_handler);
}

}  // namespace poseidon::mgpu
