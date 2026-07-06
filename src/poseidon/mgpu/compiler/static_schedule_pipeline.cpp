#include "poseidon/mgpu/compiler/static_schedule_pipeline.h"

#include "poseidon/mgpu/compiler/schedule_verifier.h"

#include <sstream>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

void add_diagnostic(
    StaticSchedulePipelineResult &result, std::string stage, std::size_t op_index,
    std::string message)
{
    result.diagnostics.push_back(
        StaticSchedulePipelineDiagnostic{ std::move(stage), op_index, std::move(message) });
}

}  // namespace

std::string StaticSchedulePipelineResult::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << diagnostics[i].stage << " op #" << diagnostics[i].op_index << ": "
               << diagnostics[i].message;
    }
    return stream.str();
}

StaticSchedulePipelineResult prepare_static_schedule(
    const MgpuSchedule &schedule, const StaticSchedulePipelineOptions &options)
{
    StaticSchedulePipelineResult result;

    StaticSchedulerOptions scheduler_options = options.scheduler;
    scheduler_options.device_count = options.device_count;
    const StaticSchedulingResult scheduling_result =
        schedule_static(schedule, scheduler_options);
    result.schedule = scheduling_result.schedule;
    result.preflight = scheduling_result.preflight;
    for (const StaticSchedulingDiagnostic &diagnostic : scheduling_result.diagnostics)
    {
        add_diagnostic(result, "scheduler", diagnostic.op_index, diagnostic.message);
    }
    if (!scheduling_result.ok())
    {
        return result;
    }

    const ScheduleVerificationResult verification =
        verify_schedule(result.schedule, ScheduleVerifierOptions{ options.device_count });
    for (const ScheduleVerificationError &error : verification.errors)
    {
        add_diagnostic(result, "verify", error.op_index, error.message);
    }
    if (!verification.ok())
    {
        return result;
    }

    if (options.emit_debug_dump)
    {
        result.debug_dump = dump_schedule(result.schedule);
    }

    return result;
}

}  // namespace poseidon::mgpu
