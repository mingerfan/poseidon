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

void enable_peer_access_if_possible(int destination_device, int source_device)
{
    check_cuda(cudaSetDevice(destination_device), "CudaPeerComm cudaSetDevice");

    const cudaError_t status = cudaDeviceEnablePeerAccess(source_device, 0);
    if (status == cudaSuccess || status == cudaErrorPeerAccessAlreadyEnabled)
    {
        if (status == cudaErrorPeerAccessAlreadyEnabled)
        {
            (void)cudaGetLastError();
        }
        return;
    }

    check_cuda(status, "CudaPeerComm cudaDeviceEnablePeerAccess");
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

    if (can_access_peer(request.destination_device, request.source_device))
    {
        enable_peer_access_if_possible(request.destination_device, request.source_device);
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

    void *source_pack = nullptr;
    void *destination_pack = nullptr;
    check_cuda(cudaSetDevice(source_device), "CudaPeerComm cudaSetDevice batch source");
    check_cuda(
        cudaMalloc(&source_pack, total_bytes),
        "CudaPeerComm cudaMalloc batch source pack");
    try
    {
        std::size_t offset = 0;
        for (const GpuObjectCopyRequest &request : requests)
        {
            const GpuObjectBufferCopy &buffer = request.buffers[0];
            check_cuda(
                cudaMemcpy(
                    static_cast<unsigned char *>(source_pack) + offset,
                    buffer.source,
                    buffer.bytes,
                    cudaMemcpyDeviceToDevice),
                "CudaPeerComm cudaMemcpy batch pack source");
            offset += buffer.bytes;
        }

        check_cuda(
            cudaSetDevice(destination_device),
            "CudaPeerComm cudaSetDevice batch destination");
        check_cuda(
            cudaMalloc(&destination_pack, total_bytes),
            "CudaPeerComm cudaMalloc batch destination pack");

        copy_buffer(CudaPeerCopyRequest{
            source_pack,
            destination_pack,
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
                    static_cast<unsigned char *>(destination_pack) + offset,
                    buffer.bytes,
                    cudaMemcpyDeviceToDevice),
                "CudaPeerComm cudaMemcpy batch unpack destination");
            offset += buffer.bytes;
        }
    }
    catch (...)
    {
        if (destination_pack != nullptr)
        {
            (void)cudaSetDevice(destination_device);
            (void)cudaFree(destination_pack);
        }
        if (source_pack != nullptr)
        {
            (void)cudaSetDevice(source_device);
            (void)cudaFree(source_pack);
        }
        throw;
    }

    check_cuda(cudaSetDevice(destination_device), "CudaPeerComm cudaSetDevice batch free destination");
    check_cuda(cudaFree(destination_pack), "CudaPeerComm cudaFree batch destination pack");
    check_cuda(cudaSetDevice(source_device), "CudaPeerComm cudaSetDevice batch free source");
    check_cuda(cudaFree(source_pack), "CudaPeerComm cudaFree batch source pack");
}

}  // namespace poseidon::mgpu
