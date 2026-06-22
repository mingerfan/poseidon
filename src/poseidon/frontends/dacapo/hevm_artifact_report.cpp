#include "poseidon/frontends/dacapo/hevm_artifact_report.h"

#include "poseidon/util/json.h"

#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

using Json = nlohmann::json;

Json parse_json_object(const std::string &text)
{
    return Json::parse(text);
}

Json hevm_opcode_summary_to_json(const DacapoHevmOpcodeSummary &summary)
{
    Json root;
    root["ok"] = summary.ok();
    root["operation_count"] = summary.operation_count;
    root["alloc_count"] = summary.alloc_count;
    root["opcode_counts"] = Json::array();
    for (const DacapoHevmOpcodeCount &count : summary.opcode_counts)
    {
        root["opcode_counts"].push_back(Json{
            { "opcode", count.opcode },
            { "name", count.name },
            { "count", count.count },
            { "supported", count.supported },
        });
    }
    root["diagnostics"] = Json::array();
    for (const DacapoAdapterDiagnostic &diagnostic : summary.diagnostics)
    {
        root["diagnostics"].push_back(Json{
            { "offset", diagnostic.offset },
            { "message", diagnostic.message },
        });
    }
    return root;
}

void add_readiness_diagnostic(
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

Json readiness_diagnostics_to_json(
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
        add_readiness_diagnostic(
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
        add_readiness_diagnostic(
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
        add_readiness_diagnostic(
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

void validate_required_input(const HevmArtifactReportInput &input)
{
    if (input.execution_config == nullptr)
    {
        throw std::invalid_argument(
            "HEVM artifact report requires execution_config");
    }
    if (input.schedule_summary == nullptr)
    {
        throw std::invalid_argument(
            "HEVM artifact report requires schedule_summary");
    }
    if (input.io_plan == nullptr)
    {
        throw std::invalid_argument("HEVM artifact report requires io_plan");
    }
}

void validate_required_failure_input(const HevmArtifactFailureReportInput &input)
{
    if (input.execution_config == nullptr)
    {
        throw std::invalid_argument(
            "HEVM artifact failure report requires execution_config");
    }
    if (input.artifacts == nullptr)
    {
        throw std::invalid_argument(
            "HEVM artifact failure report requires artifacts");
    }
}

Json artifact_diagnostics_to_json(
    const std::vector<DacapoHevmArtifactDiagnostic> &diagnostics)
{
    Json root = Json::array();
    for (const DacapoHevmArtifactDiagnostic &diagnostic : diagnostics)
    {
        root.push_back(Json{
            { "stage", diagnostic.stage },
            { "path", diagnostic.path },
            { "location", diagnostic.location },
            { "message", diagnostic.message },
        });
    }
    return root;
}

Json readiness_diagnostic_to_json(
    const HevmArtifactReadinessDiagnostic &diagnostic)
{
    Json root{
        { "stage", diagnostic.stage },
        { "location", diagnostic.location },
        { "message", diagnostic.message },
        { "has_route", diagnostic.has_route },
    };
    if (diagnostic.has_route)
    {
        root["route_index"] = diagnostic.route_index;
        root["transport"] = to_string(diagnostic.transport);
        root["source_device"] = diagnostic.source_device;
        root["destination_device"] = diagnostic.destination_device;
    }
    return root;
}

Json execution_preflight_diagnostics_to_json(
    const PoseidonGpuExecutionPreflightResult &preflight)
{
    Json diagnostics = Json::array();

    for (const ScheduleVerificationError &error :
         preflight.schedule_verification.errors)
    {
        diagnostics.push_back(Json{
            { "stage", "schedule_verification" },
            { "location", error.op_index },
            { "message", error.message },
            { "has_route", false },
        });
    }

    for (const PoseidonGpuSchedulePreflightDiagnostic &diagnostic :
         preflight.poseidon_gpu_preflight.diagnostics)
    {
        diagnostics.push_back(Json{
            { "stage", "poseidon_gpu_preflight" },
            { "location", diagnostic.op_index },
            { "message", diagnostic.message },
            { "has_route", false },
        });
    }

    if (preflight.communication_plan_evaluated)
    {
        for (const MgpuCommunicationPlanDiagnostic &diagnostic :
             preflight.communication_plan.diagnostics)
        {
            diagnostics.push_back(Json{
                { "stage", "communication_plan" },
                { "location", diagnostic.op_index },
                { "message", diagnostic.message },
                { "has_route", false },
            });
        }
    }

    if (preflight.communication_execution_preflight_evaluated)
    {
        for (const MgpuCommunicationExecutionDiagnostic &diagnostic :
             preflight.communication_execution_preflight.diagnostics)
        {
            Json entry{
                { "stage", "communication_execution_preflight" },
                { "location", diagnostic.route_index },
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
            diagnostics.push_back(std::move(entry));
        }
    }

    return diagnostics;
}

bool artifact_read_succeeded(const DacapoHevmArtifactResult &artifacts)
{
    for (const DacapoHevmArtifactDiagnostic &diagnostic : artifacts.diagnostics)
    {
        if (diagnostic.stage == "read_hevm" ||
            diagnostic.stage == "read_constants")
        {
            return false;
        }
    }
    return true;
}

Json artifact_failure_gate_diagnostics_to_json(
    const HevmArtifactFailureReportInput &input)
{
    Json root = Json::array();
    if (input.hevm_artifact_readiness != nullptr)
    {
        for (const HevmArtifactReadinessDiagnostic &diagnostic :
             input.hevm_artifact_readiness->diagnostics)
        {
            root.push_back(readiness_diagnostic_to_json(diagnostic));
        }
    }
    for (const DacapoHevmArtifactDiagnostic &diagnostic :
         input.artifacts->diagnostics)
    {
        root.push_back(Json{
            { "stage", diagnostic.stage },
            { "path", diagnostic.path },
            { "location", diagnostic.location },
            { "message", diagnostic.message },
            { "has_route", false },
        });
    }
    return root;
}

Json execution_gate_to_json(const HevmArtifactReportInput &input)
{
    const bool readiness_evaluated = input.hevm_artifact_readiness != nullptr;
    const bool readiness_ok =
        readiness_evaluated && input.hevm_artifact_readiness->ok();
    const bool execution_preflight_evaluated =
        input.poseidon_gpu_execution_preflight != nullptr;
    const bool execution_preflight_ok =
        execution_preflight_evaluated &&
        input.poseidon_gpu_execution_preflight->ok();

    Json root;
    root["ok"] = readiness_evaluated ? readiness_ok
                                      : execution_preflight_ok;
    root["status"] = root["ok"].get<bool>() ? "ready" : "not_ready";
    root["checks"] = Json{
        { "artifacts_loaded", true },
        { "schedule_built", true },
        { "hevm_io_bound", true },
        { "readiness_evaluated", readiness_evaluated },
        { "readiness_ok", readiness_ok },
        { "poseidon_gpu_execution_preflight_evaluated",
          execution_preflight_evaluated },
        { "poseidon_gpu_execution_preflight_ok", execution_preflight_ok },
    };
    root["diagnostics"] = Json::array();
    if (readiness_evaluated)
    {
        for (const HevmArtifactReadinessDiagnostic &diagnostic :
             input.hevm_artifact_readiness->diagnostics)
        {
            root["diagnostics"].push_back(readiness_diagnostic_to_json(diagnostic));
        }
    }
    else if (execution_preflight_evaluated)
    {
        root["diagnostics"] = execution_preflight_diagnostics_to_json(
            *input.poseidon_gpu_execution_preflight);
    }
    else
    {
        root["diagnostics"].push_back(Json{
            { "stage", "execution_gate" },
            { "location", 0 },
            { "has_route", false },
            { "message",
              "HEVM artifact report has no readiness or execution preflight result" },
        });
    }
    return root;
}

Json artifact_failure_execution_gate_to_json(
    const HevmArtifactFailureReportInput &input)
{
    const bool readiness_evaluated = input.hevm_artifact_readiness != nullptr;
    const bool readiness_ok =
        readiness_evaluated && input.hevm_artifact_readiness->ok();

    Json root;
    root["ok"] = false;
    root["status"] = "not_ready";
    root["checks"] = Json{
        { "artifacts_loaded", artifact_read_succeeded(*input.artifacts) },
        { "schedule_built", false },
        { "hevm_io_bound", false },
        { "readiness_evaluated", readiness_evaluated },
        { "readiness_ok", readiness_ok },
    };
    root["diagnostics"] = artifact_failure_gate_diagnostics_to_json(input);
    return root;
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
            add_readiness_diagnostic(
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
            add_readiness_diagnostic(
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
            add_communication_plan_diagnostics(
                result, preflight.communication_plan);
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
    root["diagnostics"] = readiness_diagnostics_to_json(result.diagnostics);
    return root.dump(indent);
}

std::string hevm_artifact_report_to_json(
    const HevmArtifactReportInput &input, int indent)
{
    validate_required_input(input);

    Json root;
    root["version"] = 1;
    root["execution_gate"] = execution_gate_to_json(input);
    if (!input.hevm_path.empty() || !input.constants_path.empty())
    {
        root["artifacts"] = Json{
            { "hevm", input.hevm_path },
            { "constants", input.constants_path },
        };
    }
    root["execution_config"] = parse_json_object(
        static_schedule_execution_config_to_json(*input.execution_config, -1));
    root["schedule"] = parse_json_object(
        schedule_summary_to_json(*input.schedule_summary, -1));
    root["constants"] = Json{
        { "vectors", input.constant_count },
    };
    root["hevm_io"] = Json{
        { "cipher_inputs", input.io_plan->cipher_inputs.size() },
        { "plaintext_constants", input.io_plan->plain_inputs.size() },
        { "results", input.io_plan->results.size() },
    };

    if (input.poseidon_gpu_preflight != nullptr)
    {
        root["poseidon_gpu_preflight"] = parse_json_object(
            poseidon_gpu_schedule_preflight_to_json(
                *input.poseidon_gpu_preflight, -1));
    }
    if (input.communication_plan != nullptr)
    {
        root["communication_plan"] = parse_json_object(
            communication_plan_to_json(*input.communication_plan, -1));
    }
    if (input.communication_execution_preflight != nullptr)
    {
        root["communication_execution_preflight"] = parse_json_object(
            communication_execution_preflight_to_json(
                *input.communication_execution_preflight, -1));
    }
    if (input.poseidon_gpu_execution_preflight != nullptr)
    {
        root["poseidon_gpu_execution_preflight"] = parse_json_object(
            poseidon_gpu_execution_preflight_to_json(
                *input.poseidon_gpu_execution_preflight, -1));
    }
    if (input.hevm_opcode_summary != nullptr)
    {
        root["hevm_opcode_summary"] =
            hevm_opcode_summary_to_json(*input.hevm_opcode_summary);
    }
    if (input.hevm_artifact_readiness != nullptr)
    {
        root["hevm_artifact_readiness"] = parse_json_object(
            hevm_artifact_readiness_to_json(*input.hevm_artifact_readiness, -1));
    }
    if (input.debug_dump != nullptr)
    {
        root["debug_dump"] = *input.debug_dump;
    }
    return root.dump(indent);
}

std::string hevm_artifact_failure_report_to_json(
    const HevmArtifactFailureReportInput &input, int indent)
{
    validate_required_failure_input(input);

    Json root;
    root["version"] = 1;
    root["execution_gate"] = artifact_failure_execution_gate_to_json(input);
    root["artifacts"] = Json{
        { "hevm", input.hevm_path },
        { "constants", input.constants_path },
    };
    root["execution_config"] = parse_json_object(
        static_schedule_execution_config_to_json(*input.execution_config, -1));
    root["artifact_diagnostics"] =
        artifact_diagnostics_to_json(input.artifacts->diagnostics);
    if (input.hevm_opcode_summary != nullptr)
    {
        root["hevm_opcode_summary"] =
            hevm_opcode_summary_to_json(*input.hevm_opcode_summary);
    }
    if (input.hevm_artifact_readiness != nullptr)
    {
        root["hevm_artifact_readiness"] = parse_json_object(
            hevm_artifact_readiness_to_json(
                *input.hevm_artifact_readiness, -1));
    }
    return root.dump(indent);
}

}  // namespace poseidon::mgpu
