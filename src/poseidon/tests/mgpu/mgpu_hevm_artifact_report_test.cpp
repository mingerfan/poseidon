#include "poseidon/mgpu/runtime/hevm_artifact_report.h"

#include "poseidon/util/json.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace poseidon::mgpu;

namespace
{

MgpuValueRef value(ValueId id)
{
    return MgpuValueRef{ id };
}

MgpuOp op(
    MgpuOpKind kind, int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs)
{
    return MgpuOp{ kind, device_id, std::move(inputs), std::move(outputs), {} };
}

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

MgpuSchedule make_report_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 1, { value(2) }, {}));
    return schedule;
}

void test_report_includes_execution_evidence()
{
    StaticScheduleExecutionConfig config;
    config.pipeline.device_count = 2;
    config.pipeline.placement.upload_device = 0;
    config.pipeline.placement.compute_devices = { 1 };
    config.pipeline.placement.download_device = 1;
    config.preflight_comm_available = true;
    config.communication_plan = true;
    config.communication_execution_preflight = true;
    config.communication_execution.cuda_peer_available = true;
    config.require_ready = true;

    const MgpuSchedule schedule = make_report_schedule();
    const MgpuScheduleSummary summary =
        summarize_schedule(schedule, config.pipeline.device_count);

    HevmIoBindingPlan io_plan;
    io_plan.cipher_inputs.push_back(HevmCipherInputSlot{ 0, 1, 0, 40, 2, 2 });
    io_plan.results.push_back(HevmResultSlot{ 0, 2, 2, 1, 40, 2 });

    DacapoHevmOpcodeSummary opcode_summary;
    opcode_summary.operation_count = 1;
    opcode_summary.opcode_counts.push_back(
        DacapoHevmOpcodeCount{ 9, 1, "OutputC", true });

    PoseidonGpuExecutionPreflightOptions execution_options;
    execution_options.device_count = config.pipeline.device_count;
    execution_options.copy_ops_have_comm = true;
    execution_options.check_communication_plan = true;
    execution_options.topology = make_single_node_topology(2);
    execution_options.check_communication_execution = true;
    execution_options.communication_execution = config.communication_execution;
    const PoseidonGpuExecutionPreflightResult execution_preflight =
        preflight_poseidon_gpu_execution_plan(schedule, execution_options);
    require(
        execution_preflight.ok(),
        "execution preflight should pass:\n" +
            execution_preflight.format_diagnostics());

    HevmArtifactReadinessInput readiness_input;
    readiness_input.opcode_summary = &opcode_summary;
    readiness_input.poseidon_gpu_execution_preflight = &execution_preflight;
    const HevmArtifactReadinessResult readiness =
        check_hevm_artifact_readiness(readiness_input);
    require(readiness.ok(), "readiness should pass");

    const std::string debug_dump = "mgpu.schedule {\n}\n";
    HevmArtifactReportInput input;
    input.hevm_path = "/tmp/model.hevm";
    input.constants_path = "/tmp/model.cst";
    input.execution_config = &config;
    input.schedule_summary = &summary;
    input.constant_count = 3;
    input.io_plan = &io_plan;
    input.poseidon_gpu_preflight =
        &execution_preflight.poseidon_gpu_preflight;
    input.communication_plan = &execution_preflight.communication_plan;
    input.communication_execution_preflight =
        &execution_preflight.communication_execution_preflight;
    input.poseidon_gpu_execution_preflight = &execution_preflight;
    input.hevm_opcode_summary = &opcode_summary;
    input.hevm_artifact_readiness = &readiness;
    input.debug_dump = &debug_dump;

    const nlohmann::json report =
        nlohmann::json::parse(hevm_artifact_report_to_json(input));
    require(report.at("version").get<int>() == 1, "version mismatch");
    require(
        report.at("execution_gate").at("ok").get<bool>(),
        "execution gate should be ok");
    require(
        report.at("execution_gate").at("status").get<std::string>() == "ready",
        "execution gate status mismatch");
    require(
        report.at("execution_gate")
            .at("checks")
            .at("readiness_evaluated")
            .get<bool>(),
        "execution gate should record readiness evaluation");
    require(
        report.at("execution_gate").at("diagnostics").empty(),
        "ready execution gate should not include diagnostics");
    require(
        report.at("artifacts").at("hevm").get<std::string>() == "/tmp/model.hevm",
        "HEVM path missing");
    require(
        report.at("execution_config").at("device_count").get<int>() == 2,
        "device count missing");
    require(
        report.at("schedule").at("category_counts").at("copy").get<int>() == 1,
        "copy count missing");
    require(
        report.at("constants").at("vectors").get<int>() == 3,
        "constant count missing");
    require(
        report.at("hevm_io").at("cipher_inputs").get<int>() == 1,
        "cipher input count missing");
    require(
        report.at("poseidon_gpu_execution_preflight").at("ok").get<bool>(),
        "execution preflight should be ok");
    require(
        report.at("hevm_opcode_summary").at("ok").get<bool>(),
        "opcode summary should be ok");
    require(
        report.at("hevm_artifact_readiness").at("ok").get<bool>(),
        "readiness should be ok");
    require(
        report.at("debug_dump").get<std::string>().find("mgpu.schedule") !=
            std::string::npos,
        "debug dump missing");
}

void test_report_execution_gate_reports_not_ready()
{
    StaticScheduleExecutionConfig config;
    config.pipeline.device_count = 2;
    config.pipeline.placement.upload_device = 0;
    config.pipeline.placement.compute_devices = { 1 };
    config.pipeline.placement.download_device = 1;
    config.preflight_comm_available = true;
    config.communication_plan = true;
    config.communication_execution_preflight = true;
    config.require_ready = true;

    const MgpuSchedule schedule = make_report_schedule();
    const MgpuScheduleSummary summary =
        summarize_schedule(schedule, config.pipeline.device_count);

    HevmIoBindingPlan io_plan;
    io_plan.cipher_inputs.push_back(HevmCipherInputSlot{ 0, 1, 0, 40, 2, 2 });
    io_plan.results.push_back(HevmResultSlot{ 0, 2, 2, 1, 40, 2 });

    DacapoHevmOpcodeSummary opcode_summary;
    opcode_summary.operation_count = 1;
    opcode_summary.opcode_counts.push_back(
        DacapoHevmOpcodeCount{ 9, 1, "OutputC", true });

    PoseidonGpuExecutionPreflightOptions execution_options;
    execution_options.device_count = config.pipeline.device_count;
    execution_options.copy_ops_have_comm = true;
    execution_options.check_communication_plan = true;
    execution_options.topology = make_single_node_topology(2);
    execution_options.check_communication_execution = true;
    execution_options.communication_execution = config.communication_execution;
    const PoseidonGpuExecutionPreflightResult execution_preflight =
        preflight_poseidon_gpu_execution_plan(schedule, execution_options);
    require(!execution_preflight.ok(), "execution preflight should fail");

    HevmArtifactReadinessInput readiness_input;
    readiness_input.opcode_summary = &opcode_summary;
    readiness_input.poseidon_gpu_execution_preflight = &execution_preflight;
    const HevmArtifactReadinessResult readiness =
        check_hevm_artifact_readiness(readiness_input);
    require(!readiness.ok(), "readiness should fail");

    HevmArtifactReportInput input;
    input.execution_config = &config;
    input.schedule_summary = &summary;
    input.io_plan = &io_plan;
    input.poseidon_gpu_execution_preflight = &execution_preflight;
    input.hevm_opcode_summary = &opcode_summary;
    input.hevm_artifact_readiness = &readiness;

    const nlohmann::json report =
        nlohmann::json::parse(hevm_artifact_report_to_json(input));
    require(
        !report.at("execution_gate").at("ok").get<bool>(),
        "execution gate should fail");
    require(
        report.at("execution_gate").at("status").get<std::string>() ==
            "not_ready",
        "execution gate status mismatch");
    require(
        !report.at("execution_gate")
             .at("checks")
             .at("readiness_ok")
             .get<bool>(),
        "execution gate should record failed readiness");
    require(
        !report.at("execution_gate").at("diagnostics").empty(),
        "not-ready execution gate should include diagnostics");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("stage")
                .get<std::string>() == "communication_execution_preflight",
        "execution gate diagnostic stage mismatch");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("message")
                .get<std::string>()
                .find("CUDA peer or host-staged copy backend is not available") !=
            std::string::npos,
        "execution gate diagnostic message mismatch");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("route_index")
                .get<int>() == 0,
        "execution gate route index missing");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("transport")
                .get<std::string>() == "cuda_peer",
        "execution gate transport missing");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("source_device")
                .get<int>() == 0,
        "execution gate source device missing");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("destination_device")
                .get<int>() == 1,
        "execution gate destination device missing");
    require(
        !report.at("hevm_artifact_readiness").at("ok").get<bool>(),
        "readiness JSON should fail");
}

void test_report_execution_gate_without_readiness_keeps_structured_diagnostics()
{
    StaticScheduleExecutionConfig config;
    config.pipeline.device_count = 2;
    config.communication_plan = true;
    config.communication_execution_preflight = true;

    const MgpuSchedule schedule = make_report_schedule();
    const MgpuScheduleSummary summary =
        summarize_schedule(schedule, config.pipeline.device_count);

    HevmIoBindingPlan io_plan;
    io_plan.cipher_inputs.push_back(HevmCipherInputSlot{ 0, 1, 0, 40, 2, 2 });
    io_plan.results.push_back(HevmResultSlot{ 0, 2, 2, 1, 40, 2 });

    PoseidonGpuExecutionPreflightOptions execution_options;
    execution_options.device_count = config.pipeline.device_count;
    execution_options.copy_ops_have_comm = true;
    execution_options.check_communication_plan = true;
    execution_options.topology = make_single_node_topology(2);
    execution_options.check_communication_execution = true;
    execution_options.communication_execution = config.communication_execution;
    const PoseidonGpuExecutionPreflightResult execution_preflight =
        preflight_poseidon_gpu_execution_plan(schedule, execution_options);
    require(!execution_preflight.ok(), "execution preflight should fail");

    HevmArtifactReportInput input;
    input.execution_config = &config;
    input.schedule_summary = &summary;
    input.io_plan = &io_plan;
    input.poseidon_gpu_execution_preflight = &execution_preflight;

    const nlohmann::json report =
        nlohmann::json::parse(hevm_artifact_report_to_json(input));
    require(
        !report.at("execution_gate").at("ok").get<bool>(),
        "execution gate without readiness should fail");
    require(
        !report.at("execution_gate")
             .at("checks")
             .at("readiness_evaluated")
             .get<bool>(),
        "execution gate should record missing readiness evaluation");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("stage")
                .get<std::string>() == "communication_execution_preflight",
        "execution gate fallback diagnostic stage mismatch");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("route_index")
                .get<int>() == 0,
        "execution gate fallback route index missing");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("transport")
                .get<std::string>() == "cuda_peer",
        "execution gate fallback transport missing");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("source_device")
                .get<int>() == 0,
        "execution gate fallback source device missing");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("destination_device")
                .get<int>() == 1,
        "execution gate fallback destination device missing");
}

void test_report_execution_gate_requires_gate_input()
{
    StaticScheduleExecutionConfig config;
    config.pipeline.device_count = 1;

    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(1) }, {}));
    const MgpuScheduleSummary summary =
        summarize_schedule(schedule, config.pipeline.device_count);

    HevmIoBindingPlan io_plan;
    io_plan.cipher_inputs.push_back(HevmCipherInputSlot{ 0, 1, 0, 40, 2, 2 });
    io_plan.results.push_back(HevmResultSlot{ 0, 1, 1, 0, 40, 2 });

    HevmArtifactReportInput input;
    input.execution_config = &config;
    input.schedule_summary = &summary;
    input.io_plan = &io_plan;

    const nlohmann::json report =
        nlohmann::json::parse(hevm_artifact_report_to_json(input));
    require(
        !report.at("execution_gate").at("ok").get<bool>(),
        "report without gate input should be not-ready");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("stage")
                .get<std::string>() == "execution_gate",
        "missing gate input diagnostic stage mismatch");
    require(
        report.at("execution_gate")
                .at("diagnostics")
                .at(0)
                .at("message")
                .get<std::string>()
                .find("no readiness or execution preflight result") !=
            std::string::npos,
        "missing gate input diagnostic message mismatch");
}

void test_failure_report_includes_artifact_diagnostics()
{
    StaticScheduleExecutionConfig config;
    config.pipeline.device_count = 2;
    config.require_ready = true;

    DacapoHevmArtifactResult artifacts;
    artifacts.diagnostics.push_back(DacapoHevmArtifactDiagnostic{
        "dacapo_adapter",
        "/tmp/model.hevm",
        16,
        "unsupported HEVM opcode 4",
    });

    DacapoHevmOpcodeSummary opcode_summary;
    opcode_summary.operation_count = 1;
    opcode_summary.opcode_counts.push_back(
        DacapoHevmOpcodeCount{ 4, 1, "ModswitchC", false });

    HevmArtifactReadinessInput readiness_input;
    readiness_input.opcode_summary = &opcode_summary;
    const HevmArtifactReadinessResult readiness =
        check_hevm_artifact_readiness(readiness_input);
    require(!readiness.ok(), "unsupported opcode readiness should fail");

    HevmArtifactFailureReportInput input;
    input.hevm_path = "/tmp/model.hevm";
    input.constants_path = "/tmp/model.cst";
    input.execution_config = &config;
    input.artifacts = &artifacts;
    input.hevm_opcode_summary = &opcode_summary;
    input.hevm_artifact_readiness = &readiness;

    const nlohmann::json report =
        nlohmann::json::parse(hevm_artifact_failure_report_to_json(input));
    require(
        !report.at("execution_gate").at("ok").get<bool>(),
        "failure report gate should fail");
    require(
        report.at("execution_gate").at("status").get<std::string>() ==
            "not_ready",
        "failure report gate status mismatch");
    require(
        !report.at("execution_gate")
             .at("checks")
             .at("schedule_built")
             .get<bool>(),
        "failure report should mark schedule_built false");
    require(
        report.at("artifact_diagnostics").at(0).at("message").get<std::string>() ==
            "unsupported HEVM opcode 4",
        "failure report artifact diagnostic missing");
    require(
        report.at("hevm_opcode_summary")
                .at("opcode_counts")
                .at(0)
                .at("supported")
                .get<bool>() == false,
        "failure report opcode support flag mismatch");
    require(
        report.at("hevm_artifact_readiness").at("ok").get<bool>() == false,
        "failure report readiness should fail");
}

void test_failure_report_execution_gate_preserves_route_metadata()
{
    StaticScheduleExecutionConfig config;
    config.pipeline.device_count = 32;
    config.node_count = 4;
    config.devices_per_node = 8;
    config.communication_execution_preflight = true;

    DacapoHevmArtifactResult artifacts;

    MgpuCommunicationExecutionPreflight communication_preflight;
    communication_preflight.inter_node_routes = 1;
    communication_preflight.diagnostics.push_back(
        MgpuCommunicationExecutionDiagnostic{
            3,
            MgpuTransportKind::InterNode,
            7,
            8,
            "inter-node communication backend is not available",
            true });

    HevmArtifactReadinessInput readiness_input;
    readiness_input.communication_execution_preflight = &communication_preflight;
    const HevmArtifactReadinessResult readiness =
        check_hevm_artifact_readiness(readiness_input);
    require(!readiness.ok(), "missing inter-node backend readiness should fail");

    HevmArtifactFailureReportInput input;
    input.hevm_path = "/tmp/model.hevm";
    input.constants_path = "/tmp/model.cst";
    input.execution_config = &config;
    input.artifacts = &artifacts;
    input.hevm_artifact_readiness = &readiness;

    const nlohmann::json report =
        nlohmann::json::parse(hevm_artifact_failure_report_to_json(input));
    const nlohmann::json &diagnostic =
        report.at("execution_gate").at("diagnostics").at(0);
    require(
        diagnostic.at("stage").get<std::string>() ==
            "communication_execution_preflight",
        "failure report route diagnostic stage mismatch");
    require(
        diagnostic.at("route_index").get<int>() == 3,
        "failure report route index missing");
    require(
        diagnostic.at("transport").get<std::string>() == "inter_node",
        "failure report route transport missing");
    require(
        diagnostic.at("source_device").get<int>() == 7,
        "failure report route source device missing");
    require(
        diagnostic.at("destination_device").get<int>() == 8,
        "failure report route destination device missing");
}

void test_report_rejects_missing_required_sections()
{
    HevmArtifactReportInput input;
    bool threw = false;
    try
    {
        (void)hevm_artifact_report_to_json(input);
    }
    catch (const std::invalid_argument &ex)
    {
        threw = std::string(ex.what()).find("execution_config") != std::string::npos;
    }
    require(threw, "missing required input should throw");
}

}  // namespace

int main()
{
    try
    {
        test_report_includes_execution_evidence();
        test_report_execution_gate_reports_not_ready();
        test_report_execution_gate_without_readiness_keeps_structured_diagnostics();
        test_report_execution_gate_requires_gate_input();
        test_failure_report_includes_artifact_diagnostics();
        test_failure_report_execution_gate_preserves_route_metadata();
        test_report_rejects_missing_required_sections();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu HEVM artifact report test failed: " << ex.what()
                  << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu HEVM artifact report tests passed\n";
    return EXIT_SUCCESS;
}
