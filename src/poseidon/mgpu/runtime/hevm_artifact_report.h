#pragma once

#include "poseidon/mgpu/comm/execution_preflight.h"
#include "poseidon/mgpu/comm/topology.h"
#include "poseidon/mgpu/compiler/dacapo_adapter.h"
#include "poseidon/mgpu/compiler/static_schedule_config.h"
#include "poseidon/mgpu/ir/schedule_summary.h"
#include "poseidon/mgpu/runtime/gpu_execution_preflight.h"
#include "poseidon/mgpu/runtime/hevm_artifact_readiness.h"
#include "poseidon/mgpu/runtime/hevm_io_binding.h"
#include "poseidon/mgpu/runtime/poseidon_gpu_schedule_preflight.h"

#include <cstddef>
#include <string>

namespace poseidon::mgpu
{

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

std::string hevm_artifact_report_to_json(
    const HevmArtifactReportInput &input, int indent = 2);

}  // namespace poseidon::mgpu
