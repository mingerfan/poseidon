#pragma once

#include "poseidon/mgpu/ir/schedule.h"
#include "poseidon/mgpu/runtime/backend/io_binding_backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace poseidon::mgpu
{

struct HevmCipherInputSlot
{
    std::uint64_t index = 0;
    ValueId value_id = 0;
    int device_id = 0;
    std::uint64_t scale = 0;
    std::uint64_t level = 0;
    std::uint64_t init_level = 0;
};

struct HevmResultSlot
{
    std::uint64_t index = 0;
    std::uint64_t register_id = 0;
    ValueId value_id = 0;
    int device_id = 0;
    std::uint64_t scale = 0;
    std::uint64_t level = 0;
};

struct HevmPlainInputSlot
{
    std::uint64_t register_id = 0;
    std::uint64_t constant_index = 0;
    ValueId value_id = 0;
    int device_id = 0;
    std::uint64_t scale = 0;
    std::uint64_t level = 0;
};

struct HevmIoBindingPlan
{
    std::vector<HevmCipherInputSlot> cipher_inputs;
    std::vector<HevmPlainInputSlot> plain_inputs;
    std::vector<HevmResultSlot> results;
};

struct HevmIoBindingDiagnostic
{
    std::size_t op_index = 0;
    std::string message;
};

struct HevmIoBindingPlanResult
{
    HevmIoBindingPlan plan;
    std::vector<HevmIoBindingDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

HevmIoBindingPlanResult build_hevm_io_binding_plan(const MgpuSchedule &schedule);

void bind_hevm_cipher_inputs(
    IoBindingExecutionBackend &io, const HevmIoBindingPlan &plan,
    const std::vector<std::shared_ptr<void>> &cipher_inputs);

void bind_hevm_plain_inputs_by_constant_index(
    IoBindingExecutionBackend &io, const HevmIoBindingPlan &plan,
    const std::unordered_map<std::uint64_t, std::shared_ptr<void>> &plain_inputs);

std::vector<std::shared_ptr<void>> collect_hevm_results(
    const IoBindingExecutionBackend &io, const HevmIoBindingPlan &plan);

}  // namespace poseidon::mgpu
