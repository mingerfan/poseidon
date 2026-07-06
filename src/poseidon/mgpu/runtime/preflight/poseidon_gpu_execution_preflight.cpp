#include "poseidon/mgpu/runtime/preflight/poseidon_gpu_execution_preflight.h"

#include "poseidon/util/json.h"

#include <ostream>
#include <sstream>

namespace poseidon::mgpu
{
namespace
{

using Json = nlohmann::json;

Json parse_json_object(const std::string &text)
{
    return Json::parse(text);
}

void append_stage_diagnostics(
    std::ostringstream &stream, bool &wrote_any, const char *stage,
    const std::string &diagnostics)
{
    if (diagnostics.empty())
    {
        return;
    }
    if (wrote_any)
    {
        stream << '\n';
    }
    wrote_any = true;
    stream << stage << ":\n" << diagnostics;
}

const char *status_label(bool evaluated, bool ok) noexcept
{
    if (!evaluated)
    {
        return "not_run";
    }
    return ok ? "ok" : "error";
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

std::string PoseidonGpuExecutionPreflightResult::format_diagnostics() const
{
    std::ostringstream stream;
    bool wrote_any = false;
    append_stage_diagnostics(
        stream, wrote_any, "schedule_verification",
        schedule_verification.format_errors());
    append_stage_diagnostics(
        stream, wrote_any, "poseidon_gpu_preflight",
        poseidon_gpu_preflight.format_diagnostics());
    if (communication_plan_evaluated)
    {
        append_stage_diagnostics(
            stream, wrote_any, "communication_plan",
            communication_plan.format_diagnostics());
    }
    if (communication_execution_preflight_evaluated)
    {
        append_stage_diagnostics(
            stream, wrote_any, "communication_execution_preflight",
            communication_execution_preflight.format_diagnostics());
    }
    return stream.str();
}

PoseidonGpuExecutionPreflightResult preflight_poseidon_gpu_execution_plan(
    const MgpuSchedule &schedule,
    const PoseidonGpuExecutionPreflightOptions &options)
{
    PoseidonGpuExecutionPreflightResult result;
    result.schedule_verification =
        verify_schedule(schedule, ScheduleVerifierOptions{ options.device_count });
    result.poseidon_gpu_preflight =
        preflight_poseidon_gpu_schedule(
            schedule,
            PoseidonGpuSchedulePreflightOptions{
                options.device_count,
                options.copy_ops_have_comm,
                options.relin_keys_available,
                options.galois_keys_available,
            });

    if (options.check_communication_plan || options.check_communication_execution)
    {
        result.communication_plan_evaluated = true;
        result.communication_plan = plan_schedule_communication(schedule, options.topology);
    }

    if (options.check_communication_execution && result.communication_plan.ok())
    {
        result.communication_execution_preflight_evaluated = true;
        result.communication_execution_preflight =
            preflight_communication_execution(
                result.communication_plan, options.communication_execution);
    }

    return result;
}

std::string dump_poseidon_gpu_execution_preflight(
    const PoseidonGpuExecutionPreflightResult &result)
{
    std::ostringstream stream;
    dump_poseidon_gpu_execution_preflight(stream, result);
    return stream.str();
}

void dump_poseidon_gpu_execution_preflight(
    std::ostream &stream, const PoseidonGpuExecutionPreflightResult &result)
{
    stream << "poseidon_gpu_execution_preflight:\n";
    stream << "  status: " << (result.ok() ? "ok" : "error") << '\n';
    stream << "  schedule_verification: "
           << (result.schedule_verification.ok() ? "ok" : "error") << '\n';
    stream << "  poseidon_gpu_preflight: "
           << (result.poseidon_gpu_preflight.ok() ? "ok" : "error") << '\n';
    stream << "  communication_plan: "
           << (result.communication_plan_evaluated
                   ? (result.communication_plan.ok() ? "ok" : "error")
                   : "not_run")
           << '\n';
    stream << "  communication_execution_preflight: "
           << (result.communication_execution_preflight_evaluated
                   ? (result.communication_execution_preflight.ok() ? "ok" : "error")
                   : "not_run")
           << '\n';
    const std::string diagnostics = result.format_diagnostics();
    if (!diagnostics.empty())
    {
        stream << "  diagnostics:\n";
        std::istringstream diagnostic_stream(diagnostics);
        std::string line;
        while (std::getline(diagnostic_stream, line))
        {
            stream << "    " << line << '\n';
        }
    }
}

std::string poseidon_gpu_execution_preflight_to_json(
    const PoseidonGpuExecutionPreflightResult &result, int indent)
{
    Json root;
    root["version"] = 1;
    root["ok"] = result.ok();
    root["checks"] = Json{
        { "schedule_verification",
          check_to_json(true, result.schedule_verification.ok()) },
        { "poseidon_gpu_preflight",
          check_to_json(true, result.poseidon_gpu_preflight.ok()) },
        { "communication_plan",
          check_to_json(
              result.communication_plan_evaluated,
              result.communication_plan.ok()) },
        { "communication_execution_preflight",
          check_to_json(
              result.communication_execution_preflight_evaluated,
              result.communication_execution_preflight.ok()) },
    };
    root["schedule_verification"] = Json{
        { "ok", result.schedule_verification.ok() },
        { "diagnostics", Json::array() },
    };
    for (const ScheduleVerificationError &error : result.schedule_verification.errors)
    {
        root["schedule_verification"]["diagnostics"].push_back(Json{
            { "op_index", error.op_index },
            { "message", error.message },
        });
    }
    root["poseidon_gpu_preflight"] = parse_json_object(
        poseidon_gpu_schedule_preflight_to_json(result.poseidon_gpu_preflight, -1));
    root["communication_plan"] =
        result.communication_plan_evaluated
            ? parse_json_object(communication_plan_to_json(result.communication_plan, -1))
            : Json(nullptr);
    root["communication_execution_preflight"] =
        result.communication_execution_preflight_evaluated
            ? parse_json_object(communication_execution_preflight_to_json(
                  result.communication_execution_preflight, -1))
            : Json(nullptr);
    return root.dump(indent);
}

}  // namespace poseidon::mgpu
