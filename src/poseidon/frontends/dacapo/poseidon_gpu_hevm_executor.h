#pragma once

#include "poseidon/ciphertext.h"
#include "poseidon/mgpu/comm/gpu_comm.h"
#include "poseidon/frontends/dacapo/hevm_static_execution_plan.h"
#include "poseidon/mgpu/runtime/sequential_schedule_executor.h"
#include "poseidon/poseidon_context.h"

#include <memory>
#include <string>
#include <vector>

namespace poseidon
{

class GaloisKeys;
class RelinKeys;

namespace mgpu
{

struct PoseidonGpuHevmExecutionOptions
{
    int device_count = 1;
    const RelinKeys *relin_keys = nullptr;
    const GaloisKeys *galois_keys = nullptr;
};

struct PoseidonGpuHevmExecutionResult
{
    ScheduleExecutionResult execution;
    std::vector<std::shared_ptr<Ciphertext>> results;

    bool ok() const noexcept
    {
        return execution.ok();
    }

    std::string format_errors() const;
};

PoseidonGpuHevmExecutionResult execute_hevm_static_plan_with_poseidon_gpu(
    const PoseidonContext &context, const HevmStaticExecutionPlan &plan,
    const std::vector<std::shared_ptr<const Ciphertext>> &cipher_inputs,
    const PoseidonGpuHevmExecutionOptions &options = {});

PoseidonGpuHevmExecutionResult execute_hevm_static_plan_with_poseidon_gpu(
    const PoseidonContext &context, const HevmStaticExecutionPlan &plan,
    const std::vector<std::shared_ptr<const Ciphertext>> &cipher_inputs, GpuComm &comm,
    const PoseidonGpuHevmExecutionOptions &options = {});

}  // namespace mgpu
}  // namespace poseidon
