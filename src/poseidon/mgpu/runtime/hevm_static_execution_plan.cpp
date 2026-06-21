#include "poseidon/mgpu/runtime/hevm_static_execution_plan.h"

#include <sstream>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

void add_diagnostic(
    HevmStaticExecutionPlanResult &result, std::string stage, std::string path,
    std::size_t location, std::string message)
{
    result.diagnostics.push_back(HevmStaticExecutionPlanDiagnostic{
        std::move(stage),
        std::move(path),
        location,
        std::move(message),
    });
}

void add_artifact_diagnostics(
    HevmStaticExecutionPlanResult &result, const DacapoHevmArtifactResult &artifacts)
{
    for (const DacapoHevmArtifactDiagnostic &diagnostic : artifacts.diagnostics)
    {
        add_diagnostic(
            result, "artifact." + diagnostic.stage, diagnostic.path, diagnostic.location,
            diagnostic.message);
    }
}

void add_io_plan_diagnostics(
    HevmStaticExecutionPlanResult &result, const HevmIoBindingPlanResult &io_plan)
{
    for (const HevmIoBindingDiagnostic &diagnostic : io_plan.diagnostics)
    {
        add_diagnostic(
            result, "hevm_io_binding", {}, diagnostic.op_index, diagnostic.message);
    }
}

void add_plaintext_diagnostics(
    HevmStaticExecutionPlanResult &result,
    const HevmPlaintextEncodingResult &plaintexts)
{
    for (const HevmPlaintextEncodingDiagnostic &diagnostic : plaintexts.diagnostics)
    {
        add_diagnostic(
            result, "hevm_plaintext_encoding", {}, diagnostic.value_id,
            diagnostic.message);
    }
}

}  // namespace

std::string HevmStaticExecutionPlanResult::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }

        const HevmStaticExecutionPlanDiagnostic &diagnostic = diagnostics[i];
        stream << diagnostic.stage;
        if (!diagnostic.path.empty())
        {
            stream << " " << diagnostic.path;
        }
        stream << " @" << diagnostic.location << ": " << diagnostic.message;
    }
    return stream.str();
}

HevmStaticExecutionPlanResult prepare_hevm_static_execution_plan(
    const PoseidonContext &context, const DacapoHevmArtifactResult &artifacts)
{
    HevmStaticExecutionPlanResult result;
    add_artifact_diagnostics(result, artifacts);
    if (!artifacts.ok())
    {
        return result;
    }

    const HevmIoBindingPlanResult io_plan =
        build_hevm_io_binding_plan(artifacts.schedule);
    add_io_plan_diagnostics(result, io_plan);
    if (!io_plan.ok())
    {
        return result;
    }

    const HevmPlaintextEncodingResult plaintexts =
        encode_hevm_plain_inputs(context, io_plan.plan, artifacts.constants);
    add_plaintext_diagnostics(result, plaintexts);
    if (!plaintexts.ok())
    {
        return result;
    }

    result.plan.schedule = artifacts.schedule;
    result.plan.constants = artifacts.constants;
    result.plan.io_plan = io_plan.plan;
    result.plan.encoded_plaintexts = plaintexts.plaintexts;
    result.plan.debug_dump = artifacts.debug_dump;
    return result;
}

HevmStaticExecutionPlanResult prepare_hevm_static_execution_plan_from_files(
    const PoseidonContext &context, const DacapoHevmArtifactPaths &paths,
    const StaticSchedulePipelineOptions &pipeline_options)
{
    const DacapoHevmArtifactResult artifacts =
        prepare_dacapo_hevm_artifacts_from_files(paths, pipeline_options);
    return prepare_hevm_static_execution_plan(context, artifacts);
}

}  // namespace poseidon::mgpu
