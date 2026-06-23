#pragma once

#include "poseidon/mgpu/runtime/backend/schedule_execution_backend.h"
#include "poseidon/mgpu/runtime/type_def/schedule_execution_result.h"

namespace poseidon::mgpu
{

struct SequentialScheduleExecutorOptions
{
    int device_count = 1;
};

class SequentialScheduleExecutor
{
public:
    explicit SequentialScheduleExecutor(SequentialScheduleExecutorOptions options = {});

    ScheduleExecutionResult run(const MgpuSchedule &schedule, ScheduleExecutionBackend &backend) const;

private:
    SequentialScheduleExecutorOptions options_;
};

}  // namespace poseidon::mgpu
