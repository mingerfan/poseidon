#include "poseidon/mgpu/runtime/hevm_artifact_readiness.h"

#include "poseidon/util/json.h"

#include <ostream>
#include <sstream>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

using Json = nlohmann::json;

void add_diagnostic(
    HevmArtifactReadinessResult &result, std::string stage,
    std::size_t location, std::string message)
{
    result.diagnostics.push_back(HevmArtifactReadinessDiagnostic{
        std::move(stage),
        location,
        std::move(message),
    });
}

const char *status_label(bool evaluated, bool ok) noexcept
{
    if (!evaluated)
    {
        return "not_run";
    }
    return ok ? "ok" : "error";
}

Json diagnostics_to_json(
    const std::vector<HevmArtifactReadinessDiagnostic> &diagnostics)
{
    Json result = Json::array();
    for (const HevmArtifactReadinessDiagnostic &diagnostic : diagnostics)
    {
        result.push_back(Json{
            { "stage", diagnostic.stage },
            { "location", diagnostic.location },
            { "message", diagnostic.message },
        });
    }
    return result;
}

Json check_to_json(bool evaluated, bool ok)
{
    return Json{
        { "evaluated", evaluated },
        { "ok", evaluated ? ok : false },
        { "status", status_label(evaluated, ok) },
    };
}

}  // namespace

std::string HevmArtifactReadinessResult::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        const HevmArtifactReadinessDiagnostic &diagnostic = diagnostics[i];
        stream << diagnostic.stage << " @" << diagnostic.location << ": "
               << diagnostic.message;
    }
    return stream.str();
}

HevmArtifactReadinessResult check_hevm_artifact_readiness(
    const HevmArtifactReadinessInput &input)
{
    HevmArtifactReadinessResult result;

    if (input.opcode_summary != nullptr)
    {
        result.hevm_opcode_summary_evaluated = true;
        result.hevm_opcodes_supported = input.opcode_summary->ok();
        for (const DacapoAdapterDiagnostic &diagnostic :
             input.opcode_summary->diagnostics)
        {
            add_diagnostic(
                result, "hevm_opcode_summary", diagnostic.offset,
                diagnostic.message);
        }
        for (const DacapoHevmOpcodeCount &count :
             input.opcode_summary->opcode_counts)
        {
            if (count.supported)
            {
                continue;
            }
            result.hevm_opcodes_supported = false;
            std::ostringstream stream;
            stream << "unsupported HEVM opcode " << count.opcode << " "
                   << count.name << " count " << count.count;
            add_diagnostic(
                result, "hevm_opcode_summary",
                static_cast<std::size_t>(count.opcode), stream.str());
        }
    }

    if (input.poseidon_gpu_preflight != nullptr)
    {
        result.poseidon_gpu_preflight_evaluated = true;
        result.poseidon_gpu_preflight_ok = input.poseidon_gpu_preflight->ok();
        for (const PoseidonGpuSchedulePreflightDiagnostic &diagnostic :
             input.poseidon_gpu_preflight->diagnostics)
        {
            add_diagnostic(
                result, "poseidon_gpu_preflight", diagnostic.op_index,
                diagnostic.message);
        }
    }

    if (input.communication_plan != nullptr)
    {
        result.communication_plan_evaluated = true;
        result.communication_plan_ok = input.communication_plan->ok();
        for (const MgpuCommunicationPlanDiagnostic &diagnostic :
             input.communication_plan->diagnostics)
        {
            add_diagnostic(
                result, "communication_plan", diagnostic.op_index,
                diagnostic.message);
        }
    }

    if (input.communication_execution_preflight != nullptr)
    {
        result.communication_execution_preflight_evaluated = true;
        result.communication_execution_preflight_ok =
            input.communication_execution_preflight->ok();
        for (const MgpuCommunicationExecutionDiagnostic &diagnostic :
             input.communication_execution_preflight->diagnostics)
        {
            add_diagnostic(
                result, "communication_execution_preflight",
                diagnostic.route_index, diagnostic.message);
        }
    }

    return result;
}

std::string dump_hevm_artifact_readiness(
    const HevmArtifactReadinessResult &result)
{
    std::ostringstream stream;
    dump_hevm_artifact_readiness(stream, result);
    return stream.str();
}

void dump_hevm_artifact_readiness(
    std::ostream &stream, const HevmArtifactReadinessResult &result)
{
    stream << "hevm_artifact_readiness:\n";
    stream << "  status: " << (result.ok() ? "ok" : "error") << '\n';
    stream << "  hevm_opcodes: "
           << status_label(
                  result.hevm_opcode_summary_evaluated,
                  result.hevm_opcodes_supported)
           << '\n';
    stream << "  poseidon_gpu_preflight: "
           << status_label(
                  result.poseidon_gpu_preflight_evaluated,
                  result.poseidon_gpu_preflight_ok)
           << '\n';
    stream << "  communication_plan: "
           << status_label(
                  result.communication_plan_evaluated,
                  result.communication_plan_ok)
           << '\n';
    stream << "  communication_execution_preflight: "
           << status_label(
                  result.communication_execution_preflight_evaluated,
                  result.communication_execution_preflight_ok)
           << '\n';
    if (!result.diagnostics.empty())
    {
        stream << "  diagnostics:\n";
        for (const HevmArtifactReadinessDiagnostic &diagnostic :
             result.diagnostics)
        {
            stream << "    " << diagnostic.stage << " @"
                   << diagnostic.location << ": " << diagnostic.message << '\n';
        }
    }
}

std::string hevm_artifact_readiness_to_json(
    const HevmArtifactReadinessResult &result, int indent)
{
    Json root;
    root["version"] = 1;
    root["ok"] = result.ok();
    root["checks"] = Json{
        { "hevm_opcodes",
          check_to_json(
              result.hevm_opcode_summary_evaluated,
              result.hevm_opcodes_supported) },
        { "poseidon_gpu_preflight",
          check_to_json(
              result.poseidon_gpu_preflight_evaluated,
              result.poseidon_gpu_preflight_ok) },
        { "communication_plan",
          check_to_json(
              result.communication_plan_evaluated,
              result.communication_plan_ok) },
        { "communication_execution_preflight",
          check_to_json(
              result.communication_execution_preflight_evaluated,
              result.communication_execution_preflight_ok) },
    };
    root["diagnostics"] = diagnostics_to_json(result.diagnostics);
    return root.dump(indent);
}

}  // namespace poseidon::mgpu
