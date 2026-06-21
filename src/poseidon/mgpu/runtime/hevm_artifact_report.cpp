#include "poseidon/mgpu/runtime/hevm_artifact_report.h"

#include "poseidon/util/json.h"

#include <stdexcept>

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

}  // namespace

std::string hevm_artifact_report_to_json(
    const HevmArtifactReportInput &input, int indent)
{
    validate_required_input(input);

    Json root;
    root["version"] = 1;
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

}  // namespace poseidon::mgpu
