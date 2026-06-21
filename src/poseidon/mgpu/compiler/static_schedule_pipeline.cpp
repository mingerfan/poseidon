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

    StaticPlacementOptions placement_options = options.placement;
    placement_options.device_count = options.device_count;
    const StaticPlacementResult placement_result =
        place_static_schedule(schedule, placement_options);
    result.schedule = placement_result.schedule;
    for (const StaticPlacementDiagnostic &diagnostic : placement_result.diagnostics)
    {
        add_diagnostic(result, "placement", diagnostic.op_index, diagnostic.message);
    }
    if (!placement_result.ok())
    {
        return result;
    }

    const CopyInsertionResult copy_result =
        insert_required_copies(result.schedule, options.copy_insertion);
    result.schedule = copy_result.schedule;
    for (const CopyInsertionDiagnostic &diagnostic : copy_result.diagnostics)
    {
        add_diagnostic(result, "copy_insertion", diagnostic.op_index, diagnostic.message);
    }
    if (!copy_result.ok())
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

StaticSchedulePipelineResult prepare_dacapo_static_schedule(
    std::string_view input, const DacapoAdapterOptions &adapter_options,
    const StaticSchedulePipelineOptions &pipeline_options)
{
    StaticSchedulePipelineResult result;

    const DacapoAdapterResult adapter_result =
        translate_dacapo_schedule(input, adapter_options);
    for (const DacapoAdapterDiagnostic &diagnostic : adapter_result.diagnostics)
    {
        add_diagnostic(
            result, "dacapo_adapter", diagnostic.offset, diagnostic.message);
    }
    if (!adapter_result.ok())
    {
        return result;
    }

    return prepare_static_schedule(adapter_result.schedule, pipeline_options);
}

}  // namespace poseidon::mgpu
