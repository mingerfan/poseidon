#include "poseidon/frontends/dacapo/dacapo_schedule_pipeline.h"

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
