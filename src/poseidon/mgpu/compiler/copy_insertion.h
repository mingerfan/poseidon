#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct CopyInsertionOptions
{
    ValueId next_value_id = 1;
};

struct CopyInsertionDiagnostic
{
    std::size_t op_index = 0;
    std::string message;
};

struct CopyInsertionResult
{
    MgpuSchedule schedule;
    std::vector<CopyInsertionDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

CopyInsertionResult insert_required_copies(
    const MgpuSchedule &schedule, const CopyInsertionOptions &options = {});

}  // namespace poseidon::mgpu
