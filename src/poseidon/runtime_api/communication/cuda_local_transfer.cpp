#include "poseidon/runtime_api/communication/cuda_local_transfer.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace poseidon::runtime_api::communication
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

void validate_request(const DeviceBufferCopy &request)
{
    if (request.bytes == 0)
    {
        throw std::invalid_argument("CUDA transfer buffer must be non-empty");
    }
    if (request.source == nullptr || request.destination == nullptr)
    {
        throw std::invalid_argument("CUDA transfer buffer pointer is null");
    }
    if (request.source_device < 0 || request.destination_device < 0)
    {
        throw std::invalid_argument("CUDA transfer device id is negative");
    }
}

void enable_peer_access(int destination_device, int source_device)
{
    check_cuda(cudaSetDevice(destination_device), "CUDA transfer cudaSetDevice");
    const cudaError_t status = cudaDeviceEnablePeerAccess(source_device, 0);
    if (status == cudaSuccess)
    {
        return;
    }
    if (status == cudaErrorPeerAccessAlreadyEnabled)
    {
        (void)cudaGetLastError();
        return;
    }
    check_cuda(status, "CUDA transfer cudaDeviceEnablePeerAccess");
}

void copy_host_staged(const DeviceBufferCopy &request)
{
    void *host = nullptr;
    check_cuda(cudaMallocHost(&host, request.bytes), "CUDA transfer cudaMallocHost");
    try
    {
        check_cuda(cudaSetDevice(request.source_device),
                   "CUDA transfer cudaSetDevice source");
        check_cuda(cudaMemcpy(host, request.source, request.bytes, cudaMemcpyDeviceToHost),
                   "CUDA transfer device to host");
        check_cuda(cudaSetDevice(request.destination_device),
                   "CUDA transfer cudaSetDevice destination");
        check_cuda(cudaMemcpy(request.destination, host, request.bytes, cudaMemcpyHostToDevice),
                   "CUDA transfer host to device");
    }
    catch (...)
    {
        (void)cudaFreeHost(host);
        throw;
    }
    check_cuda(cudaFreeHost(host), "CUDA transfer cudaFreeHost");
}

} // namespace

int CudaLocalTransfer::visible_device_count()
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice)
    {
        (void)cudaGetLastError();
        return 0;
    }
    check_cuda(status, "CUDA transfer cudaGetDeviceCount");
    return device_count;
}

bool CudaLocalTransfer::can_access_peer(int destination_device, int source_device)
{
    if (destination_device == source_device)
    {
        return true;
    }
    int can_access = 0;
    check_cuda(cudaDeviceCanAccessPeer(&can_access, destination_device, source_device),
               "CUDA transfer cudaDeviceCanAccessPeer");
    return can_access != 0;
}

CudaTransferRoute CudaLocalTransfer::select_route(int destination_device, int source_device,
                                                  CudaTransferRoute requested)
{
    if (destination_device < 0 || source_device < 0)
    {
        throw std::invalid_argument("CUDA transfer device id is negative");
    }

    if (requested == CudaTransferRoute::Auto)
    {
        if (destination_device == source_device)
        {
            return CudaTransferRoute::SameDevice;
        }
        return can_access_peer(destination_device, source_device)
                   ? CudaTransferRoute::PeerToPeer
                   : CudaTransferRoute::HostStaged;
    }
    if (requested == CudaTransferRoute::SameDevice &&
        destination_device != source_device)
    {
        throw std::invalid_argument("same-device CUDA route has different endpoints");
    }
    if (requested == CudaTransferRoute::PeerToPeer)
    {
        if (destination_device == source_device)
        {
            throw std::invalid_argument("peer CUDA route has identical endpoints");
        }
        if (!can_access_peer(destination_device, source_device))
        {
            throw std::runtime_error("requested CUDA peer route is unavailable");
        }
    }
    if (requested == CudaTransferRoute::HostStaged &&
        destination_device == source_device)
    {
        throw std::invalid_argument("host-staged CUDA route has identical endpoints");
    }
    if (requested != CudaTransferRoute::SameDevice &&
        requested != CudaTransferRoute::PeerToPeer &&
        requested != CudaTransferRoute::HostStaged)
    {
        throw std::invalid_argument("unknown CUDA transfer route");
    }
    return requested;
}

CudaTransferRoute CudaLocalTransfer::copy_sync(const DeviceBufferCopy &request,
                                               CudaTransferRoute requested) const
{
    validate_request(request);
    const CudaTransferRoute route = select_route(
        request.destination_device, request.source_device, requested);
    switch (route)
    {
    case CudaTransferRoute::SameDevice:
        check_cuda(cudaSetDevice(request.destination_device),
                   "CUDA transfer cudaSetDevice");
        check_cuda(cudaMemcpy(request.destination, request.source, request.bytes,
                              cudaMemcpyDeviceToDevice),
                   "CUDA transfer device to device");
        break;
    case CudaTransferRoute::PeerToPeer:
        enable_peer_access(request.destination_device, request.source_device);
        check_cuda(cudaMemcpyPeer(request.destination, request.destination_device,
                                  request.source, request.source_device, request.bytes),
                   "CUDA transfer cudaMemcpyPeer");
        break;
    case CudaTransferRoute::HostStaged:
        copy_host_staged(request);
        break;
    case CudaTransferRoute::Auto:
        throw std::logic_error("CUDA transfer route was not resolved");
    }
    return route;
}

void CudaLocalTransfer::copy_sync(const std::vector<DeviceBufferCopy> &requests,
                                  CudaTransferRoute requested) const
{
    for (const auto &request : requests)
    {
        copy_sync(request, requested);
    }
}

} // namespace poseidon::runtime_api::communication
