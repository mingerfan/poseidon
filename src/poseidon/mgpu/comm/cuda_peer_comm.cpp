#include "poseidon/mgpu/comm/cuda_peer_comm.h"
#include "poseidon/runtime_api/communication/cuda_local_transfer.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace poseidon::mgpu
{
namespace
{

runtime_api::communication::DeviceBufferCopy adapt_buffer_copy(
    const CudaPeerCopyRequest &request)
{
    return runtime_api::communication::DeviceBufferCopy{
        request.source,
        request.destination,
        request.bytes,
        request.source_device,
        request.destination_device,
    };
}

runtime_api::communication::DeviceBufferCopy adapt_buffer_copy(
    const GpuObjectBufferCopy &request)
{
    return runtime_api::communication::DeviceBufferCopy{
        request.source,
        request.destination,
        request.bytes,
        request.source_device,
        request.destination_device,
    };
}

}  // namespace

int CudaPeerComm::visible_device_count()
{
    return runtime_api::communication::CudaLocalTransfer::visible_device_count();
}

bool CudaPeerComm::can_access_peer(int destination_device, int source_device)
{
    return runtime_api::communication::CudaLocalTransfer::can_access_peer(
        destination_device, source_device);
}

void CudaPeerComm::copy_buffer(const CudaPeerCopyRequest &request) const
{
    runtime_api::communication::CudaLocalTransfer{}.copy_sync(adapt_buffer_copy(request));
}

void CudaPeerComm::copy_object(const GpuObjectCopyRequest &request)
{
    const GpuObjectCopyValidationResult validation =
        validate_full_object_copy_request(request);
    if (!validation.ok())
    {
        throw std::invalid_argument(validation.format_errors());
    }

    runtime_api::communication::CudaLocalTransfer{}.copy_sync(
        adapt_buffer_copy(request.buffers.front()));
}

void CudaPeerComm::copy_objects(const std::vector<GpuObjectCopyRequest> &requests)
{
    std::vector<runtime_api::communication::DeviceBufferCopy> buffers;
    buffers.reserve(requests.size());
    for (const auto &request : requests)
    {
        const auto validation = validate_full_object_copy_request(request);
        if (!validation.ok())
        {
            throw std::invalid_argument(validation.format_errors());
        }
        buffers.push_back(adapt_buffer_copy(request.buffers.front()));
    }
    runtime_api::communication::CudaLocalTransfer{}.copy_sync(buffers);
}

}  // namespace poseidon::mgpu
