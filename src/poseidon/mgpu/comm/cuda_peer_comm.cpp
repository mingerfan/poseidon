#include "poseidon/mgpu/comm/cuda_peer_comm.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace poseidon::mgpu
{
namespace
{

void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void validate_request(const CudaPeerCopyRequest &request)
{
    if (request.bytes == 0)
    {
        return;
    }
    if (request.source == nullptr)
    {
        throw std::invalid_argument("CudaPeerComm source pointer is null");
    }
    if (request.destination == nullptr)
    {
        throw std::invalid_argument("CudaPeerComm destination pointer is null");
    }
    if (request.source_device < 0 || request.destination_device < 0)
    {
        throw std::invalid_argument("CudaPeerComm device ids must be non-negative");
    }
}

void copy_with_host_staging(const CudaPeerCopyRequest &request)
{
    std::vector<unsigned char> host(request.bytes);

    check_cuda(cudaSetDevice(request.source_device), "CudaPeerComm cudaSetDevice source");
    check_cuda(
        cudaMemcpy(
            host.data(),
            request.source,
            request.bytes,
            cudaMemcpyDeviceToHost),
        "CudaPeerComm cudaMemcpy device to host");

    check_cuda(cudaSetDevice(request.destination_device), "CudaPeerComm cudaSetDevice destination");
    check_cuda(
        cudaMemcpy(
            request.destination,
            host.data(),
            request.bytes,
            cudaMemcpyHostToDevice),
            "CudaPeerComm cudaMemcpy host to device");
}

}  // namespace

int CudaPeerComm::visible_device_count()
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice)
    {
        (void)cudaGetLastError();
        return 0;
    }
    check_cuda(status, "CudaPeerComm cudaGetDeviceCount");
    return device_count;
}

bool CudaPeerComm::can_access_peer(int destination_device, int source_device)
{
    if (destination_device == source_device)
    {
        return true;
    }

    int can_access = 0;
    check_cuda(
        cudaDeviceCanAccessPeer(&can_access, destination_device, source_device),
        "CudaPeerComm cudaDeviceCanAccessPeer");
    return can_access != 0;
}

CudaPeerAccessSnapshot CudaPeerComm::peer_access_snapshot(
    int destination_device, int source_device) const
{
    if (source_device < 0 || destination_device < 0)
    {
        throw std::invalid_argument("CudaPeerComm device ids must be non-negative");
    }

    std::lock_guard<std::mutex> lock(peer_access_mutex_);
    const auto iter =
        peer_access_cache_.find(PeerAccessKey{ destination_device, source_device });
    if (iter == peer_access_cache_.end())
    {
        return {};
    }

    return CudaPeerAccessSnapshot{
        true,
        iter->second.supported,
        iter->second.enabled,
        iter->second.enable_call_count,
    };
}

bool CudaPeerComm::ensure_peer_access(
    int destination_device, int source_device) const
{
    if (destination_device == source_device)
    {
        return true;
    }

    std::lock_guard<std::mutex> lock(peer_access_mutex_);
    PeerAccessCacheEntry &entry =
        peer_access_cache_[PeerAccessKey{ destination_device, source_device }];

    if (!entry.support_checked)
    {
        int can_access = 0;
        check_cuda(
            cudaDeviceCanAccessPeer(
                &can_access, destination_device, source_device),
            "CudaPeerComm cudaDeviceCanAccessPeer");
        entry.supported = can_access != 0;
        entry.support_checked = true;
    }

    if (!entry.supported)
    {
        return false;
    }
    if (entry.enabled)
    {
        return true;
    }

    check_cuda(cudaSetDevice(destination_device), "CudaPeerComm cudaSetDevice");
    const cudaError_t status = cudaDeviceEnablePeerAccess(source_device, 0);
    ++entry.enable_call_count;
    if (status == cudaSuccess || status == cudaErrorPeerAccessAlreadyEnabled)
    {
        if (status == cudaErrorPeerAccessAlreadyEnabled)
        {
            (void)cudaGetLastError();
        }
        entry.enabled = true;
        return true;
    }

    check_cuda(status, "CudaPeerComm cudaDeviceEnablePeerAccess");
    return false;
}

void CudaPeerComm::copy_buffer(const CudaPeerCopyRequest &request) const
{
    validate_request(request);
    if (request.bytes == 0)
    {
        return;
    }

    if (request.source_device == request.destination_device)
    {
        check_cuda(cudaSetDevice(request.destination_device), "CudaPeerComm cudaSetDevice");
        check_cuda(
            cudaMemcpy(
                request.destination,
                request.source,
                request.bytes,
                cudaMemcpyDeviceToDevice),
            "CudaPeerComm cudaMemcpy device to device");
        return;
    }

    if (ensure_peer_access(request.destination_device, request.source_device))
    {
        check_cuda(
            cudaMemcpyPeer(
                request.destination,
                request.destination_device,
                request.source,
                request.source_device,
                request.bytes),
            "CudaPeerComm cudaMemcpyPeer");
        return;
    }

    copy_with_host_staging(request);
}

void CudaPeerComm::copy_buffer_peer_async(
    const CudaPeerCopyRequest &request, cudaStream_t stream) const
{
    validate_request(request);
    if (request.bytes == 0)
    {
        return;
    }

    check_cuda(cudaSetDevice(request.destination_device), "CudaPeerComm cudaSetDevice");
    if (request.source_device == request.destination_device)
    {
        check_cuda(
            cudaMemcpyAsync(
                request.destination,
                request.source,
                request.bytes,
                cudaMemcpyDeviceToDevice,
                stream),
            "CudaPeerComm cudaMemcpyAsync device to device");
        return;
    }

    if (!ensure_peer_access(request.destination_device, request.source_device))
    {
        std::ostringstream stream_message;
        stream_message
            << "CudaPeerComm async peer copy from device "
            << request.source_device << " to device "
            << request.destination_device
            << " requires CUDA peer access; synchronous copy_buffer provides "
               "host-staging fallback";
        throw std::runtime_error(stream_message.str());
    }

    check_cuda(
        cudaMemcpyPeerAsync(
            request.destination,
            request.destination_device,
            request.source,
            request.source_device,
            request.bytes,
            stream),
        "CudaPeerComm cudaMemcpyPeerAsync");
}

void CudaPeerComm::copy_object_peer_async(
    const GpuObjectCopyRequest &request, cudaStream_t stream) const
{
    const GpuObjectCopyValidationResult validation =
        validate_full_object_copy_request(request);
    if (!validation.ok())
    {
        throw std::invalid_argument(validation.format_errors());
    }

    const GpuObjectBufferCopy &buffer = request.buffers[0];
    copy_buffer_peer_async(CudaPeerCopyRequest{
        buffer.source,
        buffer.destination,
        buffer.bytes,
        buffer.source_device,
        buffer.destination_device,
    }, stream);
}

void CudaPeerComm::copy_object(const GpuObjectCopyRequest &request)
{
    const GpuObjectCopyValidationResult validation =
        validate_full_object_copy_request(request);
    if (!validation.ok())
    {
        throw std::invalid_argument(validation.format_errors());
    }

    const GpuObjectBufferCopy &buffer = request.buffers[0];
    copy_buffer(CudaPeerCopyRequest{
        buffer.source,
        buffer.destination,
        buffer.bytes,
        buffer.source_device,
        buffer.destination_device,
    });
}

}  // namespace poseidon::mgpu
