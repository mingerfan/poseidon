#pragma once

#include "poseidon/mgpu/compiler/scheduler/static_scheduler.h"
#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct StaticSchedulePipelineOptions
{
    int device_count = 1;
    StaticSchedulerOptions scheduler;
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
    StaticSchedulePreflight preflight;
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
