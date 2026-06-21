#pragma once

#include "poseidon/mgpu/comm/inter_node_transport.h"
#include "poseidon/mgpu/comm/materialized_gpu_comm.h"
#include "poseidon/mgpu/comm/topology.h"
#include "poseidon/mgpu/runtime/schedule_interpreter.h"
#include "poseidon/mgpu/runtime/static_schedule_executor.h"

namespace poseidon::mgpu
{

class PlannedCommunicationStaticScheduleExecutor
{
public:
    PlannedCommunicationStaticScheduleExecutor(
        MgpuTopology topology, GpuObjectCopyMaterializer &materializer,
        GpuObjectCopyBackend &local_backend,
        InterNodeTransportBackend &inter_node_backend,
        ScheduleOpHandler &non_copy_handler,
        StaticScheduleExecutorOptions options = {});

    ScheduleExecutionResult run(const MgpuSchedule &schedule) const;

private:
    MgpuTopology topology_;
    GpuObjectCopyMaterializer &materializer_;
    GpuObjectCopyBackend &local_backend_;
    InterNodeTransportBackend &inter_node_backend_;
    ScheduleOpHandler &non_copy_handler_;
    StaticScheduleExecutorOptions options_;
};

}  // namespace poseidon::mgpu
