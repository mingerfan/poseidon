#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct ScheduleVerifierOptions
{
    int device_count = 1;
};

struct ScheduleVerificationError
{
    std::size_t op_index = 0;
    std::string message;
};

struct ScheduleVerificationResult
{
    std::vector<ScheduleVerificationError> errors;

    bool ok() const noexcept
    {
        return errors.empty();
    }

    std::string format_errors() const;
};

ScheduleVerificationResult verify_schedule(
    const MgpuSchedule &schedule, const ScheduleVerifierOptions &options = {});

}  // namespace poseidon::mgpu
