#pragma once

#include "poseidon/mgpu/ir/schedule.h"
#include "poseidon/mgpu/runtime/type_def/object_store.h"

namespace poseidon::mgpu
{

class ScheduleExecutionBackend
{
public:
    virtual ~ScheduleExecutionBackend() = default;

    virtual void execute(const MgpuOp &op, MgpuObjectStore &object_store) = 0;
};

}  // namespace poseidon::mgpu
