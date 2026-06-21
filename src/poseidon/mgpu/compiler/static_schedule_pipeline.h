#pragma once

#include "poseidon/mgpu/compiler/copy_insertion.h"
#include "poseidon/mgpu/compiler/static_placement.h"
#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct StaticSchedulePipelineOptions
{
    int device_count = 1;
    StaticPlacementOptions placement;
    CopyInsertionOptions copy_insertion;
    bool emit_debug_dump = false;
};

struct StaticSchedulePipelineDiagnostic
{
    std::string stage;
    std::size_t op_index = 0;
    std::string message;
};

struct StaticSchedulePipelineResult
{
    MgpuSchedule schedule;
    std::string debug_dump;
    std::vector<StaticSchedulePipelineDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

StaticSchedulePipelineResult prepare_static_schedule(
    const MgpuSchedule &schedule, const StaticSchedulePipelineOptions &options = {});

}  // namespace poseidon::mgpu
