#pragma once

#include "poseidon/mgpu/comm/gpu_comm.h"
#include "poseidon/mgpu/runtime/backend/schedule_execution_backend.h"

namespace poseidon::mgpu
{

class CopyDispatchingExecutionBackend final : public ScheduleExecutionBackend
{
public:
    explicit CopyDispatchingExecutionBackend(GpuComm &comm, ScheduleExecutionBackend *fallback = nullptr);

    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override;

private:
    GpuComm &comm_;
    ScheduleExecutionBackend *fallback_ = nullptr;
};

}  // namespace poseidon::mgpu
