#include "poseidon/mgpu/comm/gpu_object_copy.h"

#include <sstream>

namespace poseidon::mgpu
{
namespace
{

void add_error(GpuObjectCopyValidationResult &result, const std::string &message)
{
    result.errors.push_back(message);
}

}  // namespace

std::string GpuObjectCopyValidationResult::format_errors() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < errors.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << errors[i];
    }
    return stream.str();
}

GpuObjectCopyValidationResult validate_full_object_copy_request(
    const GpuObjectCopyRequest &request)
{
    GpuObjectCopyValidationResult result;

    if (request.source_id == 0)
    {
        add_error(result, "source value id 0 is reserved");
    }
    if (request.destination_id == 0)
    {
        add_error(result, "destination value id 0 is reserved");
    }

    if (request.buffers.size() != 1)
    {
        std::ostringstream stream;
        stream << "V1 full-object copy requires exactly one buffer, got "
               << request.buffers.size();
        add_error(result, stream.str());
        return result;
    }

    const GpuObjectBufferCopy &buffer = request.buffers[0];
    if (buffer.bytes == 0)
    {
        add_error(result, "object copy buffer must be non-empty");
    }
    if (buffer.source == nullptr)
    {
        add_error(result, "object copy source pointer is null");
    }
    if (buffer.destination == nullptr)
    {
        add_error(result, "object copy destination pointer is null");
    }
    if (buffer.source_device < 0)
    {
        add_error(result, "object copy source device must be non-negative");
    }
    if (buffer.destination_device < 0)
    {
        add_error(result, "object copy destination device must be non-negative");
    }

    return result;
}

}  // namespace poseidon::mgpu
