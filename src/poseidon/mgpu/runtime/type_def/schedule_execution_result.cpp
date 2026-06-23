#include "poseidon/mgpu/runtime/type_def/schedule_execution_result.h"

#include <sstream>

namespace poseidon::mgpu
{

std::string ScheduleExecutionResult::format_errors() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < errors.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << "op #" << errors[i].op_index << ": " << errors[i].message;
    }
    return stream.str();
}

}  // namespace poseidon::mgpu
