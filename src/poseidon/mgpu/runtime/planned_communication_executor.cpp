#include "poseidon/mgpu/runtime/planned_communication_executor.h"

#include "poseidon/mgpu/comm/planned_materialized_gpu_comm.h"
#include "poseidon/mgpu/comm/routed_object_copy.h"

#include <cstddef>
#include <exception>
#include <sstream>
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

std::size_t route_op_index(
    const MgpuSchedule &schedule, const MgpuCopyRoute &route)
{
    for (std::size_t op_index = 0; op_index < schedule.ops.size(); ++op_index)
    {
        const MgpuOp &op = schedule.ops[op_index];
        if (!is_copy_op(op.kind) || op.inputs.size() != 1 ||
            op.outputs.size() != 1)
        {
            continue;
        }
        if (op.inputs[0].id == route.source_id &&
            op.outputs[0].id == route.destination_id)
        {
            return op_index;
        }
    }
    return 0;
}

std::string format_execution_diagnostic_message(
    const MgpuCommunicationExecutionDiagnostic &diagnostic)
{
    std::ostringstream stream;
    stream << "communication route #" << diagnostic.route_index << " "
           << diagnostic.message;
    return stream.str();
}

void copy_communication_execution_errors(
    ScheduleExecutionResult &result, const MgpuSchedule &schedule,
    const MgpuCommunicationPlan &plan,
    const MgpuCommunicationExecutionPreflight &preflight)
{
    for (const MgpuCommunicationExecutionDiagnostic &diagnostic :
         preflight.diagnostics)
    {
        std::size_t op_index = 0;
        if (diagnostic.route_index < plan.routes.size())
        {
            op_index =
                route_op_index(schedule, plan.routes[diagnostic.route_index]);
        }
        result.errors.push_back(ScheduleExecutionError{
            op_index,
            format_execution_diagnostic_message(diagnostic),
        });
    }
}

}  // namespace

PlannedCommunicationStaticScheduleExecutor::
    PlannedCommunicationStaticScheduleExecutor(
        MgpuTopology topology, GpuObjectCopyMaterializer &materializer,
        GpuObjectCopyBackend &local_backend,
        InterNodeTransportBackend &inter_node_backend,
        ScheduleOpHandler &non_copy_handler,
        StaticScheduleExecutorOptions options,
        MgpuCommunicationExecutionOptions communication_options)
    : topology_(std::move(topology)), materializer_(materializer),
      local_backend_(local_backend), inter_node_backend_(inter_node_backend),
      non_copy_handler_(non_copy_handler), options_(options),
      communication_options_(communication_options)
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

    const MgpuCommunicationExecutionPreflight execution_preflight =
        preflight_communication_execution(
            communication_plan, communication_options_);
    if (!execution_preflight.ok())
    {
        copy_communication_execution_errors(
            result, schedule, communication_plan, execution_preflight);
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
