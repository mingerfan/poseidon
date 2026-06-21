#pragma once

#include "poseidon/mgpu/comm/gpu_comm.h"
#include "poseidon/mgpu/runtime/schedule_interpreter.h"

namespace poseidon::mgpu
{

struct StaticScheduleExecutorOptions
{
    int device_count = 1;
};

class StaticScheduleExecutor
{
public:
    StaticScheduleExecutor(
        GpuComm &comm, ScheduleOpHandler &non_copy_handler,
        StaticScheduleExecutorOptions options = {});

    ScheduleExecutionResult run(const MgpuSchedule &schedule) const;

private:
    GpuComm &comm_;
    ScheduleOpHandler &non_copy_handler_;
    StaticScheduleExecutorOptions options_;
};

}  // namespace poseidon::mgpu
