#pragma once

#include "poseidon/mgpu/comm/execution_preflight.h"
#include "poseidon/mgpu/comm/topology.h"
#include "poseidon/frontends/dacapo/dacapo_adapter.h"
#include "poseidon/frontends/dacapo/dacapo_artifacts.h"
#include "poseidon/mgpu/compiler/static_schedule_config.h"
#include "poseidon/mgpu/ir/schedule_summary.h"
#include "poseidon/mgpu/runtime/preflight/poseidon_gpu_execution_preflight.h"
#include "poseidon/frontends/dacapo/hevm_io_binding.h"
#include "poseidon/mgpu/runtime/preflight/poseidon_gpu_schedule_preflight.h"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct HevmArtifactReadinessInput
{
    const DacapoHevmOpcodeSummary *opcode_summary = nullptr;
    const PoseidonGpuSchedulePreflightResult *poseidon_gpu_preflight = nullptr;
    const MgpuCommunicationPlan *communication_plan = nullptr;
    const MgpuCommunicationExecutionPreflight *communication_execution_preflight =
        nullptr;
    const PoseidonGpuExecutionPreflightResult *poseidon_gpu_execution_preflight =
        nullptr;
};

struct HevmArtifactReadinessDiagnostic
{
    std::string stage;
    std::size_t location = 0;
    std::string message;
    bool has_route = false;
    std::size_t route_index = 0;
    MgpuTransportKind transport = MgpuTransportKind::SameDevice;
    int source_device = 0;
    int destination_device = 0;
};

struct HevmArtifactReadinessResult
{
    bool hevm_opcode_summary_evaluated = false;
    bool hevm_opcodes_supported = true;
    bool poseidon_gpu_execution_preflight_evaluated = false;
    bool poseidon_gpu_execution_preflight_ok = true;
    bool poseidon_gpu_preflight_evaluated = false;
    bool poseidon_gpu_preflight_ok = true;
    bool communication_plan_evaluated = false;
    bool communication_plan_ok = true;
    bool communication_execution_preflight_evaluated = false;
    bool communication_execution_preflight_ok = true;
    std::vector<HevmArtifactReadinessDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

HevmArtifactReadinessResult check_hevm_artifact_readiness(
    const HevmArtifactReadinessInput &input);

std::string dump_hevm_artifact_readiness(
    const HevmArtifactReadinessResult &result);
void dump_hevm_artifact_readiness(
    std::ostream &stream, const HevmArtifactReadinessResult &result);

std::string hevm_artifact_readiness_to_json(
    const HevmArtifactReadinessResult &result, int indent = 2);

struct HevmArtifactReportInput
{
    std::string hevm_path;
    std::string constants_path;
    const StaticScheduleExecutionConfig *execution_config = nullptr;
    const MgpuScheduleSummary *schedule_summary = nullptr;
    std::size_t constant_count = 0;
    const HevmIoBindingPlan *io_plan = nullptr;
    const PoseidonGpuSchedulePreflightResult *poseidon_gpu_preflight = nullptr;
    const MgpuCommunicationPlan *communication_plan = nullptr;
    const MgpuCommunicationExecutionPreflight
        *communication_execution_preflight = nullptr;
    const PoseidonGpuExecutionPreflightResult
        *poseidon_gpu_execution_preflight = nullptr;
    const DacapoHevmOpcodeSummary *hevm_opcode_summary = nullptr;
    const HevmArtifactReadinessResult *hevm_artifact_readiness = nullptr;
    const std::string *debug_dump = nullptr;
};

struct HevmArtifactFailureReportInput
{
    std::string hevm_path;
    std::string constants_path;
    const StaticScheduleExecutionConfig *execution_config = nullptr;
    const DacapoHevmArtifactResult *artifacts = nullptr;
    const DacapoHevmOpcodeSummary *hevm_opcode_summary = nullptr;
    const HevmArtifactReadinessResult *hevm_artifact_readiness = nullptr;
};

std::string hevm_artifact_report_to_json(
    const HevmArtifactReportInput &input, int indent = 2);

std::string hevm_artifact_failure_report_to_json(
    const HevmArtifactFailureReportInput &input, int indent = 2);

}  // namespace poseidon::mgpu
