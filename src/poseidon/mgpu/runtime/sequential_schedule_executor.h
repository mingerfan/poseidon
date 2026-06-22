#pragma once

#include "poseidon/mgpu/compiler/schedule_verifier.h"
#include "poseidon/mgpu/runtime/object_store.h"

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct SequentialScheduleExecutorOptions
{
    int device_count = 1;
};

struct ScheduleExecutionError
{
    std::size_t op_index = 0;
    std::string message;
};

struct ScheduleExecutionResult
{
    MgpuObjectStore object_store;
    std::vector<ScheduleExecutionError> errors;

    bool ok() const noexcept
    {
        return errors.empty();
    }

    std::string format_errors() const;
};

class ScheduleExecutionBackend
{
public:
    virtual ~ScheduleExecutionBackend() = default;

    virtual void execute(const MgpuOp &op, MgpuObjectStore &object_store) = 0;
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
