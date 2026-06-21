#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

constexpr int kUnassignedDevice = -1;

enum class StaticPlacementPolicy
{
    SingleDevice,
    RoundRobinCompute
};

struct StaticPlacementOptions
{
    int device_count = 1;
    int default_device = 0;
    StaticPlacementPolicy policy = StaticPlacementPolicy::SingleDevice;
    bool preserve_existing_devices = true;
    std::vector<int> compute_devices;
};

struct StaticPlacementDiagnostic
{
    std::size_t op_index = 0;
    std::string message;
};

struct StaticPlacementResult
{
    MgpuSchedule schedule;
    std::vector<StaticPlacementDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

const char *to_string(StaticPlacementPolicy policy) noexcept;

StaticPlacementResult place_static_schedule(
    const MgpuSchedule &schedule, const StaticPlacementOptions &options = {});

}  // namespace poseidon::mgpu
