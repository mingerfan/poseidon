#pragma once

#include "poseidon/mgpu/comm/execution_preflight.h"
#include "poseidon/mgpu/comm/topology.h"
#include "poseidon/mgpu/compiler/schedule_verifier.h"
#include "poseidon/mgpu/runtime/preflight/poseidon_gpu_schedule_preflight.h"

#include <iosfwd>
#include <string>

namespace poseidon::mgpu
{

struct PoseidonGpuExecutionPreflightOptions
{
    int device_count = 1;
    bool copy_ops_have_comm = false;
    bool relin_keys_available = false;
    bool galois_keys_available = false;
    bool check_communication_plan = false;
    MgpuTopology topology;
    bool check_communication_execution = false;
    MgpuCommunicationExecutionOptions communication_execution;
};

struct PoseidonGpuExecutionPreflightResult
{
    ScheduleVerificationResult schedule_verification;
    PoseidonGpuSchedulePreflightResult poseidon_gpu_preflight;
    bool communication_plan_evaluated = false;
    MgpuCommunicationPlan communication_plan;
    bool communication_execution_preflight_evaluated = false;
    MgpuCommunicationExecutionPreflight communication_execution_preflight;

    bool ok() const noexcept
    {
        return schedule_verification.ok() && poseidon_gpu_preflight.ok() &&
               (!communication_plan_evaluated || communication_plan.ok()) &&
               (!communication_execution_preflight_evaluated ||
                communication_execution_preflight.ok());
    }

    std::string format_diagnostics() const;
};

PoseidonGpuExecutionPreflightResult preflight_poseidon_gpu_execution_plan(
    const MgpuSchedule &schedule,
    const PoseidonGpuExecutionPreflightOptions &options = {});

std::string dump_poseidon_gpu_execution_preflight(
    const PoseidonGpuExecutionPreflightResult &result);
void dump_poseidon_gpu_execution_preflight(
    std::ostream &stream, const PoseidonGpuExecutionPreflightResult &result);

std::string poseidon_gpu_execution_preflight_to_json(
    const PoseidonGpuExecutionPreflightResult &result, int indent = 2);

}  // namespace poseidon::mgpu
