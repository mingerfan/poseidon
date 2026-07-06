#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace poseidon::mgpu
{

struct ScheduleJsonDiagnostic
{
    std::string path;
    std::string message;
};

struct ScheduleJsonParseResult
{
    MgpuSchedule schedule;
    std::vector<ScheduleJsonDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

ScheduleJsonParseResult parse_schedule_json(std::string_view text);
std::string schedule_to_json(const MgpuSchedule &schedule, int indent = 2);

}  // namespace poseidon::mgpu
