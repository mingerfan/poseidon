#include "poseidon/mgpu/comm/materialized_gpu_comm.h"

#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{

MaterializedGpuComm::MaterializedGpuComm(
    GpuObjectCopyMaterializer &materializer, GpuObjectCopyBackend &backend)
    : materializer_(materializer), backend_(backend)
{
}

std::shared_ptr<void> MaterializedGpuComm::copy(const GpuCommCopyRequest &request)
{
    if (request.source_object == nullptr)
    {
        throw std::invalid_argument("materialized GPU copy source object is null");
    }

    MaterializedGpuObjectCopy materialized = materializer_.materialize_copy(request);
    if (materialized.destination_object == nullptr)
    {
        throw std::invalid_argument("materialized GPU copy destination object is null");
    }

    const GpuObjectCopyValidationResult validation =
        validate_full_object_copy_request(materialized.object_copy);
    if (!validation.ok())
    {
        throw std::invalid_argument(validation.format_errors());
    }

    backend_.copy_object(materialized.object_copy);
    return std::move(materialized.destination_object);
}

}  // namespace poseidon::mgpu
