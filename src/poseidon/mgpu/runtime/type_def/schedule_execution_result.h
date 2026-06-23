#pragma once

#include "poseidon/mgpu/runtime/type_def/object_store.h"

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct ScheduleExecutionError
{
    std::size_t op_index = 0;
    std::string message;
};

struct ScheduleExecutionResult
{
    MgpuObjectStore object_store;
    std::vector<ScheduleExecutionError> errors;

    bool ok() const noexcept
    {
        return errors.empty();
    }

    std::string format_errors() const;
};

}  // namespace poseidon::mgpu
