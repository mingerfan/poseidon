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

std::size_t checked_add(std::size_t a, std::size_t b, const char *what)
{
    if (b > static_cast<std::size_t>(-1) - a)
    {
        throw std::overflow_error(what);
    }
    return a + b;
}

bool can_batch_copy_objects(
    const std::vector<GpuObjectCopyRequest> &requests,
    int &source_device,
    int &destination_device,
    std::size_t &total_bytes)
{
    total_bytes = 0;
    if (requests.empty())
    {
        return false;
    }

    const GpuObjectCopyValidationResult first_validation =
        validate_full_object_copy_request(requests[0]);
    if (!first_validation.ok())
    {
        throw std::invalid_argument(first_validation.format_errors());
    }

    source_device = requests[0].buffers[0].source_device;
    destination_device = requests[0].buffers[0].destination_device;
    for (const GpuObjectCopyRequest &request : requests)
    {
        const GpuObjectCopyValidationResult validation =
            validate_full_object_copy_request(request);
        if (!validation.ok())
        {
            throw std::invalid_argument(validation.format_errors());
        }

        const GpuObjectBufferCopy &buffer = request.buffers[0];
        if (buffer.source_device != source_device ||
            buffer.destination_device != destination_device)
        {
            return false;
        }
        total_bytes =
            checked_add(total_bytes, buffer.bytes, "CudaPeerComm batch bytes overflow");
    }
    return true;
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

CudaPeerPackScratch::~CudaPeerPackScratch()
{
    release();
}

void CudaPeerPackScratch::reserve(
    int source_device, int destination_device, std::size_t bytes)
{
    if (source_device < 0 || destination_device < 0)
    {
        throw std::invalid_argument("CudaPeerPackScratch device ids must be non-negative");
    }
    if (bytes == 0)
    {
        return;
    }

    reserve_buffer(
        source_pack_,
        source_capacity_bytes_,
        source_device_,
        source_allocation_count_,
        source_device,
        bytes,
        "CudaPeerPackScratch cudaSetDevice source",
        "CudaPeerPackScratch cudaMalloc source pack",
        "CudaPeerPackScratch cudaFree source pack");
    reserve_buffer(
        destination_pack_,
        destination_capacity_bytes_,
        destination_device_,
        destination_allocation_count_,
        destination_device,
        bytes,
        "CudaPeerPackScratch cudaSetDevice destination",
        "CudaPeerPackScratch cudaMalloc destination pack",
        "CudaPeerPackScratch cudaFree destination pack");
}

void CudaPeerPackScratch::release() noexcept
{
    release_buffer(destination_pack_, destination_capacity_bytes_, destination_device_);
    release_buffer(source_pack_, source_capacity_bytes_, source_device_);
}

CudaPeerPackScratchSnapshot CudaPeerPackScratch::snapshot() const
{
    return CudaPeerPackScratchSnapshot{
        source_capacity_bytes_,
        destination_capacity_bytes_,
        source_allocation_count_,
        destination_allocation_count_,
        source_device_,
        destination_device_,
    };
}

void CudaPeerPackScratch::reserve_buffer(
    void *&buffer,
    std::size_t &capacity_bytes,
    int &device,
    std::size_t &allocation_count,
    int requested_device,
    std::size_t requested_bytes,
    const char *set_device_what,
    const char *malloc_what,
    const char *free_what)
{
    if (buffer != nullptr && device == requested_device &&
        capacity_bytes >= requested_bytes)
    {
        return;
    }

    if (buffer != nullptr)
    {
        check_cuda(cudaSetDevice(device), set_device_what);
        check_cuda(cudaFree(buffer), free_what);
        buffer = nullptr;
        capacity_bytes = 0;
        device = -1;
    }

    check_cuda(cudaSetDevice(requested_device), set_device_what);
    check_cuda(cudaMalloc(&buffer, requested_bytes), malloc_what);
    capacity_bytes = requested_bytes;
    device = requested_device;
    ++allocation_count;
}

void CudaPeerPackScratch::release_buffer(
    void *&buffer,
    std::size_t &capacity_bytes,
    int &device) noexcept
{
    if (buffer == nullptr)
    {
        return;
    }

    (void)cudaSetDevice(device);
    (void)cudaFree(buffer);
    buffer = nullptr;
    capacity_bytes = 0;
    device = -1;
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

void CudaPeerComm::copy_objects(const std::vector<GpuObjectCopyRequest> &requests)
{
    GpuObjectCopyBackend::copy_objects(requests);
}

void CudaPeerComm::copy_objects_pack_unpack(
    const std::vector<GpuObjectCopyRequest> &requests)
{
    CudaPeerPackScratch scratch;
    copy_objects_pack_unpack(requests, scratch);
}

void CudaPeerComm::copy_objects_pack_unpack(
    const std::vector<GpuObjectCopyRequest> &requests,
    CudaPeerPackScratch &scratch)
{
    int source_device = 0;
    int destination_device = 0;
    std::size_t total_bytes = 0;
    if (!can_batch_copy_objects(requests, source_device, destination_device, total_bytes))
    {
        GpuObjectCopyBackend::copy_objects(requests);
        return;
    }
    if (total_bytes == 0)
    {
        return;
    }

    scratch.reserve(source_device, destination_device, total_bytes);

    std::size_t offset = 0;
    check_cuda(cudaSetDevice(source_device), "CudaPeerComm cudaSetDevice batch source");
    for (const GpuObjectCopyRequest &request : requests)
    {
        const GpuObjectBufferCopy &buffer = request.buffers[0];
        check_cuda(
            cudaMemcpy(
                static_cast<unsigned char *>(scratch.source_pack_) + offset,
                buffer.source,
                buffer.bytes,
                cudaMemcpyDeviceToDevice),
            "CudaPeerComm cudaMemcpy batch pack source");
        offset += buffer.bytes;
    }

    copy_buffer(CudaPeerCopyRequest{
        scratch.source_pack_,
        scratch.destination_pack_,
        total_bytes,
        source_device,
        destination_device,
    });

    offset = 0;
    check_cuda(
        cudaSetDevice(destination_device),
        "CudaPeerComm cudaSetDevice batch unpack destination");
    for (const GpuObjectCopyRequest &request : requests)
    {
        const GpuObjectBufferCopy &buffer = request.buffers[0];
        check_cuda(
            cudaMemcpy(
                buffer.destination,
                static_cast<unsigned char *>(scratch.destination_pack_) + offset,
                buffer.bytes,
                cudaMemcpyDeviceToDevice),
            "CudaPeerComm cudaMemcpy batch unpack destination");
        offset += buffer.bytes;
    }
}

}  // namespace poseidon::mgpu
