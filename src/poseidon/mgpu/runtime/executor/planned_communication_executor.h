#pragma once

#include "poseidon/mgpu/comm/execution_preflight.h"
#include "poseidon/mgpu/comm/inter_node_transport.h"
#include "poseidon/mgpu/comm/materialized_gpu_comm.h"
#include "poseidon/mgpu/comm/topology.h"
#include "poseidon/mgpu/runtime/executor/sequential_schedule_executor.h"

namespace poseidon::mgpu
{

struct StaticScheduleExecutionConfig;

class PlannedCommunicationScheduleExecutor
{
public:
    PlannedCommunicationScheduleExecutor(
        MgpuTopology topology, GpuObjectCopyMaterializer &materializer,
        GpuObjectCopyBackend &local_backend,
        InterNodeTransportBackend &inter_node_backend,
        ScheduleExecutionBackend &non_copy_backend,
        SequentialScheduleExecutorOptions options = {},
        MgpuCommunicationExecutionOptions communication_options = {});

    static PlannedCommunicationScheduleExecutor from_config(
        const StaticScheduleExecutionConfig &config,
        GpuObjectCopyMaterializer &materializer,
        GpuObjectCopyBackend &local_backend,
        InterNodeTransportBackend &inter_node_backend,
        ScheduleExecutionBackend &non_copy_backend);

    ScheduleExecutionResult run(const MgpuSchedule &schedule) const;

private:
    MgpuTopology topology_;
    GpuObjectCopyMaterializer &materializer_;
    GpuObjectCopyBackend &local_backend_;
    InterNodeTransportBackend &inter_node_backend_;
    ScheduleExecutionBackend &non_copy_backend_;
    SequentialScheduleExecutorOptions options_;
    MgpuCommunicationExecutionOptions communication_options_;
};

}  // namespace poseidon::mgpu
