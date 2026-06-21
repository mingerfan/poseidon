#pragma once

#include "poseidon/mgpu/comm/gpu_comm.h"
#include "poseidon/mgpu/runtime/schedule_interpreter.h"

namespace poseidon::mgpu
{

class CopyDispatchingScheduleHandler final : public ScheduleOpHandler
{
public:
    explicit CopyDispatchingScheduleHandler(GpuComm &comm, ScheduleOpHandler *fallback = nullptr);

    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override;

private:
    GpuComm &comm_;
    ScheduleOpHandler *fallback_ = nullptr;
};

}  // namespace poseidon::mgpu
