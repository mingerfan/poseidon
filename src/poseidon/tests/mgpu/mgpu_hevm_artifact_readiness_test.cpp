#include "poseidon/mgpu/runtime/hevm_artifact_readiness.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace poseidon::mgpu;

namespace
{

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void require_contains(const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos)
    {
        throw std::runtime_error("expected text to contain: " + needle + "\ntext:\n" + text);
    }
}

DacapoHevmOpcodeSummary make_supported_opcode_summary()
{
    DacapoHevmOpcodeSummary summary;
    summary.operation_count = 1;
    summary.opcode_counts.push_back(DacapoHevmOpcodeCount{ 9, 1, "MulCP", true });
    return summary;
}

void test_all_checks_pass()
{
    const DacapoHevmOpcodeSummary opcode_summary = make_supported_opcode_summary();
    const PoseidonGpuSchedulePreflightResult gpu_preflight;
    const MgpuCommunicationPlan communication_plan;
    const MgpuCommunicationExecutionPreflight communication_preflight;

    const HevmArtifactReadinessResult result = check_hevm_artifact_readiness(
        HevmArtifactReadinessInput{
            &opcode_summary,
            &gpu_preflight,
            &communication_plan,
            &communication_preflight,
        });

    require(result.ok(), "readiness should pass");
    require(result.hevm_opcode_summary_evaluated, "opcode summary should be evaluated");
    require(result.poseidon_gpu_preflight_evaluated, "GPU preflight should be evaluated");
    require(
        result.communication_execution_preflight_evaluated,
        "communication execution preflight should be evaluated");

    const std::string text = dump_hevm_artifact_readiness(result);
    require_contains(text, "status: ok");
    require_contains(text, "hevm_opcodes: ok");

    const std::string json = hevm_artifact_readiness_to_json(result);
    require_contains(json, "\"ok\": true");
    require_contains(json, "\"communication_execution_preflight\"");
}

void test_unsupported_opcode_fails()
{
    DacapoHevmOpcodeSummary opcode_summary = make_supported_opcode_summary();
    opcode_summary.opcode_counts.push_back(
        DacapoHevmOpcodeCount{ 4, 2, "ModswitchC", false });

    const HevmArtifactReadinessResult result = check_hevm_artifact_readiness(
        HevmArtifactReadinessInput{ &opcode_summary });

    require(!result.ok(), "unsupported opcode should fail readiness");
    require(!result.hevm_opcodes_supported, "opcode support flag should fail");
    require_contains(result.format_diagnostics(), "unsupported HEVM opcode 4");
    require_contains(result.format_diagnostics(), "ModswitchC");
}

void test_preflight_diagnostics_are_propagated()
{
    const DacapoHevmOpcodeSummary opcode_summary = make_supported_opcode_summary();

    PoseidonGpuSchedulePreflightResult gpu_preflight;
    gpu_preflight.diagnostics.push_back(
        PoseidonGpuSchedulePreflightDiagnostic{ 7, "bootstrap fallback is not implemented" });

    MgpuCommunicationPlan communication_plan;
    communication_plan.diagnostics.push_back(
        MgpuCommunicationPlanDiagnostic{ 3, "copy destination device 9 is not present in topology" });

    MgpuCommunicationExecutionPreflight communication_preflight;
    communication_preflight.diagnostics.push_back(
        MgpuCommunicationExecutionDiagnostic{ 1, "inter-node communication backend is not available" });

    const HevmArtifactReadinessResult result = check_hevm_artifact_readiness(
        HevmArtifactReadinessInput{
            &opcode_summary,
            &gpu_preflight,
            &communication_plan,
            &communication_preflight,
        });

    require(!result.ok(), "preflight diagnostics should fail readiness");
    require(!result.poseidon_gpu_preflight_ok, "GPU preflight flag should fail");
    require(!result.communication_plan_ok, "communication plan flag should fail");
    require(
        !result.communication_execution_preflight_ok,
        "communication execution preflight flag should fail");
    require_contains(result.format_diagnostics(), "bootstrap fallback");
    require_contains(result.format_diagnostics(), "not present in topology");
    require_contains(result.format_diagnostics(), "inter-node communication backend");
}

}  // namespace

int main()
{
    try
    {
        test_all_checks_pass();
        test_unsupported_opcode_fails();
        test_preflight_diagnostics_are_propagated();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu HEVM artifact readiness test failed: "
                  << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu HEVM artifact readiness tests passed\n";
    return EXIT_SUCCESS;
}
