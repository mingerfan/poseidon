#include "poseidon/mgpu/runtime/planned_communication_executor.h"

#include "poseidon/mgpu/comm/planned_materialized_gpu_comm.h"
#include "poseidon/mgpu/comm/routed_object_copy.h"

#include <exception>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

void copy_communication_plan_errors(
    ScheduleExecutionResult &result, const MgpuCommunicationPlan &plan)
{
    for (const MgpuCommunicationPlanDiagnostic &diagnostic : plan.diagnostics)
    {
        result.errors.push_back(
            ScheduleExecutionError{ diagnostic.op_index, diagnostic.message });
    }
}

}  // namespace

PlannedCommunicationStaticScheduleExecutor::
    PlannedCommunicationStaticScheduleExecutor(
        MgpuTopology topology, GpuObjectCopyMaterializer &materializer,
        GpuObjectCopyBackend &local_backend,
        InterNodeTransportBackend &inter_node_backend,
        ScheduleOpHandler &non_copy_handler,
        StaticScheduleExecutorOptions options)
    : topology_(std::move(topology)), materializer_(materializer),
      local_backend_(local_backend), inter_node_backend_(inter_node_backend),
      non_copy_handler_(non_copy_handler), options_(options)
{
}

ScheduleExecutionResult PlannedCommunicationStaticScheduleExecutor::run(
    const MgpuSchedule &schedule) const
{
    ScheduleExecutionResult result;
    const MgpuCommunicationPlan communication_plan =
        plan_schedule_communication(schedule, topology_);
    if (!communication_plan.ok())
    {
        copy_communication_plan_errors(result, communication_plan);
        return result;
    }

    try
    {
        RoutedGpuObjectCopyBackend routed_backend(
            topology_, local_backend_, inter_node_backend_);
        PlannedMaterializedGpuComm comm(
            communication_plan, materializer_, routed_backend);
        StaticScheduleExecutor executor(comm, non_copy_handler_, options_);
        return executor.run(schedule);
    }
    catch (const std::exception &ex)
    {
        result.errors.push_back(ScheduleExecutionError{ 0, ex.what() });
        return result;
    }
    catch (...)
    {
        result.errors.push_back(ScheduleExecutionError{
            0,
            "unknown planned communication setup error",
        });
        return result;
    }
}

}  // namespace poseidon::mgpu
