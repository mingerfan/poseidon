#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct PoseidonGpuSchedulePreflightOptions
{
    int device_count = 1;
    bool copy_ops_have_comm = false;
    bool relin_keys_available = false;
    bool galois_keys_available = false;
};

struct PoseidonGpuSchedulePreflightDiagnostic
{
    std::size_t op_index = 0;
    std::string message;
};

struct PoseidonGpuSchedulePreflightResult
{
    bool requires_comm = false;
    bool requires_relin_keys = false;
    bool requires_galois_keys = false;
    std::vector<int> devices;
    std::vector<PoseidonGpuSchedulePreflightDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

PoseidonGpuSchedulePreflightResult preflight_poseidon_gpu_schedule(
    const MgpuSchedule &schedule,
    const PoseidonGpuSchedulePreflightOptions &options = {});

std::string dump_poseidon_gpu_schedule_preflight(
    const PoseidonGpuSchedulePreflightResult &result);
void dump_poseidon_gpu_schedule_preflight(
    std::ostream &stream, const PoseidonGpuSchedulePreflightResult &result);

std::string poseidon_gpu_schedule_preflight_to_json(
    const PoseidonGpuSchedulePreflightResult &result, int indent = 2);

}  // namespace poseidon::mgpu
