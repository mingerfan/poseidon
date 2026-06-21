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

void add_communication_execution_diagnostic(
    HevmArtifactReadinessResult &result,
    const MgpuCommunicationExecutionDiagnostic &diagnostic)
{
    HevmArtifactReadinessDiagnostic readiness_diagnostic;
    readiness_diagnostic.stage = "communication_execution_preflight";
    readiness_diagnostic.location = diagnostic.route_index;
    readiness_diagnostic.message = diagnostic.message;
    readiness_diagnostic.has_route = diagnostic.has_route;
    readiness_diagnostic.route_index = diagnostic.route_index;
    readiness_diagnostic.transport = diagnostic.transport;
    readiness_diagnostic.source_device = diagnostic.source_device;
    readiness_diagnostic.destination_device = diagnostic.destination_device;
    result.diagnostics.push_back(std::move(readiness_diagnostic));
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
        Json entry{
            { "stage", diagnostic.stage },
            { "location", diagnostic.location },
            { "message", diagnostic.message },
            { "has_route", diagnostic.has_route },
        };
        if (diagnostic.has_route)
        {
            entry["route_index"] = diagnostic.route_index;
            entry["transport"] = to_string(diagnostic.transport);
            entry["source_device"] = diagnostic.source_device;
            entry["destination_device"] = diagnostic.destination_device;
        }
        result.push_back(std::move(entry));
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

void add_schedule_verification_diagnostics(
    HevmArtifactReadinessResult &result,
    const ScheduleVerificationResult &verification)
{
    for (const ScheduleVerificationError &error : verification.errors)
    {
        add_diagnostic(
            result, "schedule_verification", error.op_index, error.message);
    }
}

void add_poseidon_gpu_preflight_diagnostics(
    HevmArtifactReadinessResult &result,
    const PoseidonGpuSchedulePreflightResult &preflight)
{
    for (const PoseidonGpuSchedulePreflightDiagnostic &diagnostic :
         preflight.diagnostics)
    {
        add_diagnostic(
            result, "poseidon_gpu_preflight", diagnostic.op_index,
            diagnostic.message);
    }
}

void add_communication_plan_diagnostics(
    HevmArtifactReadinessResult &result,
    const MgpuCommunicationPlan &plan)
{
    for (const MgpuCommunicationPlanDiagnostic &diagnostic : plan.diagnostics)
    {
        add_diagnostic(
            result, "communication_plan", diagnostic.op_index,
            diagnostic.message);
    }
}

void add_communication_execution_preflight_diagnostics(
    HevmArtifactReadinessResult &result,
    const MgpuCommunicationExecutionPreflight &preflight)
{
    for (const MgpuCommunicationExecutionDiagnostic &diagnostic :
         preflight.diagnostics)
    {
        add_communication_execution_diagnostic(result, diagnostic);
    }
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

    if (input.poseidon_gpu_execution_preflight != nullptr)
    {
        const PoseidonGpuExecutionPreflightResult &preflight =
            *input.poseidon_gpu_execution_preflight;
        result.poseidon_gpu_execution_preflight_evaluated = true;
        result.poseidon_gpu_execution_preflight_ok = preflight.ok();

        result.poseidon_gpu_preflight_evaluated = true;
        result.poseidon_gpu_preflight_ok = preflight.poseidon_gpu_preflight.ok();
        result.communication_plan_evaluated = preflight.communication_plan_evaluated;
        result.communication_plan_ok = preflight.communication_plan_evaluated
                                           ? preflight.communication_plan.ok()
                                           : true;
        result.communication_execution_preflight_evaluated =
            preflight.communication_execution_preflight_evaluated;
        result.communication_execution_preflight_ok =
            preflight.communication_execution_preflight_evaluated
                ? preflight.communication_execution_preflight.ok()
                : true;

        add_schedule_verification_diagnostics(
            result, preflight.schedule_verification);
        add_poseidon_gpu_preflight_diagnostics(
            result, preflight.poseidon_gpu_preflight);
        if (preflight.communication_plan_evaluated)
        {
            add_communication_plan_diagnostics(result, preflight.communication_plan);
        }
        if (preflight.communication_execution_preflight_evaluated)
        {
            add_communication_execution_preflight_diagnostics(
                result, preflight.communication_execution_preflight);
        }
    }
    else if (input.poseidon_gpu_preflight != nullptr)
    {
        result.poseidon_gpu_preflight_evaluated = true;
        result.poseidon_gpu_preflight_ok = input.poseidon_gpu_preflight->ok();
        add_poseidon_gpu_preflight_diagnostics(
            result, *input.poseidon_gpu_preflight);
    }

    if (input.poseidon_gpu_execution_preflight == nullptr &&
        input.communication_plan != nullptr)
    {
        result.communication_plan_evaluated = true;
        result.communication_plan_ok = input.communication_plan->ok();
        add_communication_plan_diagnostics(result, *input.communication_plan);
    }

    if (input.poseidon_gpu_execution_preflight == nullptr &&
        input.communication_execution_preflight != nullptr)
    {
        result.communication_execution_preflight_evaluated = true;
        result.communication_execution_preflight_ok =
            input.communication_execution_preflight->ok();
        add_communication_execution_preflight_diagnostics(
            result, *input.communication_execution_preflight);
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
    stream << "  poseidon_gpu_execution_preflight: "
           << status_label(
                  result.poseidon_gpu_execution_preflight_evaluated,
                  result.poseidon_gpu_execution_preflight_ok)
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
        { "poseidon_gpu_execution_preflight",
          check_to_json(
              result.poseidon_gpu_execution_preflight_evaluated,
              result.poseidon_gpu_execution_preflight_ok) },
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
