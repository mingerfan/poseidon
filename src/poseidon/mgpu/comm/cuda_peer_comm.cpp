#include "poseidon/mgpu/comm/cuda_peer_comm.h"

#include <cuda_runtime_api.h>

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

void CudaPeerComm::copy_object(const GpuObjectCopyRequest &request) const
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
