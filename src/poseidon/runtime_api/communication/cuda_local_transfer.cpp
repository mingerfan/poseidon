#include "poseidon/runtime_api/communication/cuda_local_transfer.h"

#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_stream_wait_trace.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

cudaStream_t create_stream(int device)
{
    check_cuda(cudaSetDevice(device), "CUDA transfer cudaSetDevice");
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "CUDA transfer cudaStreamCreateWithFlags");
    return stream;
}

cudaEvent_t create_event(int device)
{
    check_cuda(cudaSetDevice(device), "CUDA transfer cudaSetDevice");
    cudaEvent_t event = nullptr;
    check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
               "CUDA transfer cudaEventCreateWithFlags");
    return event;
}

cudaEvent_t record_execution_ready(int device)
{
    const cudaEvent_t event = create_event(device);
    try
    {
        check_cuda(
            cudaEventRecord(event, gpu::gpu_execution_stream()),
            "CUDA transfer cudaEventRecord destination ready");
    }
    catch (...)
    {
        (void)cudaEventDestroy(event);
        throw;
    }
    return event;
}

} // namespace

PinnedHostBuffer::PinnedHostBuffer(std::size_t bytes) : bytes_(bytes)
{
    if (bytes_ == 0)
    {
        throw std::invalid_argument("CUDA pinned Host buffer must be non-empty");
    }
    check_cuda(cudaMallocHost(&data_, bytes_), "CUDA transfer cudaMallocHost");
}

PinnedHostBuffer::~PinnedHostBuffer()
{
    if (data_ != nullptr)
    {
        (void)cudaFreeHost(data_);
    }
}

void *PinnedHostBuffer::data() noexcept
{
    return data_;
}

const void *PinnedHostBuffer::data() const noexcept
{
    return data_;
}

std::size_t PinnedHostBuffer::size() const noexcept
{
    return bytes_;
}

struct CudaTransferRequest::State
{
    struct Stream
    {
        int device = 0;
        cudaStream_t value = nullptr;
    };

    struct Event
    {
        int device = 0;
        cudaEvent_t value = nullptr;
    };

    ~State()
    {
        if (!waited && completion_recorded)
        {
            (void)cudaSetDevice(completion.device);
            (void)cudaEventSynchronize(completion.value);
        }
        else if (!waited)
        {
            for (const auto &stream : submitted_streams)
            {
                (void)cudaSetDevice(stream.device);
                (void)cudaStreamSynchronize(stream.value);
            }
        }
        for (auto &event : events)
        {
            if (event.value != nullptr)
            {
                (void)cudaSetDevice(event.device);
                (void)cudaEventDestroy(event.value);
            }
        }
        if (completion.value != nullptr)
        {
            (void)cudaSetDevice(completion.device);
            (void)cudaEventDestroy(completion.value);
        }
    }

    void retain_stream(int device, cudaStream_t stream)
    {
        const auto found = std::find_if(
            submitted_streams.begin(), submitted_streams.end(),
            [device](const Stream &submitted) {
                return submitted.device == device;
            });
        if (found == submitted_streams.end())
        {
            submitted_streams.push_back({device, stream});
        }
    }

    std::shared_ptr<void> stream_owner;
    std::vector<Stream> submitted_streams;
    std::vector<Event> events;
    Event completion;
    std::shared_ptr<PinnedHostBuffer> staging;
    bool completion_recorded = false;
    bool waited = false;
};

struct CudaLocalTransfer::State
{
    struct Stream
    {
        int device = 0;
        cudaStream_t value = nullptr;
    };

    explicit State(const std::vector<int> &cuda_device_ids)
    {
        if (cuda_device_ids.empty())
        {
            throw std::invalid_argument(
                "CUDA transfer requires at least one device");
        }
        streams.reserve(cuda_device_ids.size());
        try
        {
            for (std::size_t index = 0; index < cuda_device_ids.size(); ++index)
            {
                const int device = cuda_device_ids[index];
                if (device < 0 ||
                    std::find(cuda_device_ids.begin(),
                              cuda_device_ids.begin() + index,
                              device) != cuda_device_ids.begin() + index)
                {
                    throw std::invalid_argument(
                        "CUDA transfer device ids must be nonnegative and unique");
                }
                streams.push_back({device, create_stream(device)});
            }
        }
        catch (...)
        {
            destroy_streams();
            throw;
        }
    }

    ~State()
    {
        for (const auto &stream : streams)
        {
            if (stream.value != nullptr)
            {
                (void)cudaSetDevice(stream.device);
                (void)cudaStreamSynchronize(stream.value);
            }
        }
        destroy_streams();
    }

    cudaStream_t stream(int device) const
    {
        const auto found = std::find_if(
            streams.begin(), streams.end(),
            [device](const Stream &stream) { return stream.device == device; });
        if (found == streams.end())
        {
            throw std::invalid_argument(
                "CUDA transfer device is not configured");
        }
        return found->value;
    }

private:
    void destroy_streams() noexcept
    {
        for (auto &stream : streams)
        {
            if (stream.value != nullptr)
            {
                (void)cudaSetDevice(stream.device);
                (void)cudaStreamDestroy(stream.value);
                stream.value = nullptr;
            }
        }
    }

    std::vector<Stream> streams;
};

CudaLocalTransfer::CudaLocalTransfer(std::vector<int> cuda_device_ids)
    : state_(std::make_shared<State>(cuda_device_ids))
{}

CudaLocalTransfer::~CudaLocalTransfer() = default;

CudaTransferRequest::CudaTransferRequest() = default;
CudaTransferRequest::~CudaTransferRequest() = default;
CudaTransferRequest::CudaTransferRequest(CudaTransferRequest &&) noexcept = default;
CudaTransferRequest &CudaTransferRequest::operator=(CudaTransferRequest &&) noexcept = default;

cudaEvent_t CudaTransferRequest::completion_event() const
{
    if (!state_ || !state_->completion_recorded ||
        state_->completion.value == nullptr)
    {
        throw std::logic_error("CUDA transfer request has no completion event");
    }
    return state_->completion.value;
}

int CudaTransferRequest::completion_device() const
{
    if (!state_ || !state_->completion_recorded ||
        state_->completion.value == nullptr)
    {
        throw std::logic_error("CUDA transfer request has no completion event");
    }
    return state_->completion.device;
}

void CudaTransferRequest::wait()
{
    if (!state_ || !state_->completion_recorded ||
        state_->completion.value == nullptr)
    {
        throw std::logic_error("CUDA transfer request has no completion event");
    }
    if (state_->waited)
    {
        return;
    }
    check_cuda(cudaSetDevice(state_->completion.device),
               "CUDA transfer cudaSetDevice");
    check_cuda(cudaEventSynchronize(state_->completion.value),
               "CUDA transfer cudaEventSynchronize");
    state_->waited = true;
}

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

void CudaLocalTransfer::enable_peer_access(int destination_device, int source_device)
{
    if (destination_device < 0 || source_device < 0 ||
        destination_device == source_device)
    {
        throw std::invalid_argument("CUDA peer access endpoints are invalid");
    }
    if (!can_access_peer(destination_device, source_device))
    {
        throw std::runtime_error("CUDA peer access is unavailable");
    }

    check_cuda(cudaSetDevice(destination_device),
               "CUDA peer access cudaSetDevice");
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
    check_cuda(status, "CUDA peer access cudaDeviceEnablePeerAccess");
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

CudaTransferRequest CudaLocalTransfer::copy_async(
    const DeviceBufferCopy &request, CudaTransferRoute requested,
    cudaEvent_t source_ready) const
{
    validate_request(request);
    const CudaTransferRoute route = select_route(
        request.destination_device, request.source_device, requested);

    CudaTransferRequest result;
    result.state_ = std::make_unique<CudaTransferRequest::State>();
    auto &state = *result.state_;
    state.stream_owner = state_;

    if (route == CudaTransferRoute::HostStaged)
    {
        state.staging = std::make_shared<PinnedHostBuffer>(request.bytes);
        const cudaStream_t source_stream = state_->stream(request.source_device);
        state.retain_stream(request.source_device, source_stream);
        if (source_ready != nullptr)
        {
            gpu::gpu_stream_wait_event(
                source_stream, source_ready, request.source_device,
                "gpu.stream_wait.transfer_source_ready",
                "CUDA transfer cudaStreamWaitEvent source");
        }
        check_cuda(cudaMemcpyAsync(state.staging->data(), request.source, request.bytes,
                                   cudaMemcpyDeviceToHost, source_stream),
                   "CUDA transfer cudaMemcpyAsync device to Host staging");
        const cudaEvent_t staged = create_event(request.source_device);
        state.events.push_back({request.source_device, staged});
        check_cuda(cudaEventRecord(staged, source_stream),
                   "CUDA transfer cudaEventRecord Host staging");

        const cudaStream_t destination_stream =
            state_->stream(request.destination_device);
        state.retain_stream(request.destination_device, destination_stream);
        const cudaEvent_t destination_ready =
            record_execution_ready(request.destination_device);
        state.events.push_back({request.destination_device, destination_ready});
        check_cuda(cudaSetDevice(request.destination_device),
                   "CUDA transfer cudaSetDevice destination");
        gpu::gpu_stream_wait_event(
            destination_stream, destination_ready, request.destination_device,
            "gpu.stream_wait.transfer_destination_ready",
            "CUDA transfer cudaStreamWaitEvent destination ready");
        gpu::gpu_stream_wait_event(
            destination_stream, staged, request.destination_device,
            "gpu.stream_wait.transfer_host_staging",
            "CUDA transfer cudaStreamWaitEvent destination");
        check_cuda(cudaMemcpyAsync(request.destination, state.staging->data(),
                                   request.bytes, cudaMemcpyHostToDevice,
                                   destination_stream),
                   "CUDA transfer cudaMemcpyAsync Host staging to device");
        state.completion = {
            request.destination_device, create_event(request.destination_device)};
        check_cuda(cudaEventRecord(state.completion.value, destination_stream),
                   "CUDA transfer cudaEventRecord destination");
        state.completion_recorded = true;
        return result;
    }

    const cudaStream_t stream = state_->stream(request.destination_device);
    state.retain_stream(request.destination_device, stream);
    const cudaEvent_t destination_ready =
        record_execution_ready(request.destination_device);
    state.events.push_back({request.destination_device, destination_ready});
    gpu::gpu_stream_wait_event(
        stream, destination_ready, request.destination_device,
        "gpu.stream_wait.transfer_destination_ready",
        "CUDA transfer cudaStreamWaitEvent destination ready");
    if (source_ready != nullptr)
    {
        gpu::gpu_stream_wait_event(
            stream, source_ready, request.destination_device,
            "gpu.stream_wait.transfer_source_ready",
            "CUDA transfer cudaStreamWaitEvent");
    }
    if (route == CudaTransferRoute::SameDevice)
    {
        check_cuda(cudaMemcpyAsync(request.destination, request.source, request.bytes,
                                   cudaMemcpyDeviceToDevice, stream),
                   "CUDA transfer cudaMemcpyAsync device to device");
    }
    else if (route == CudaTransferRoute::PeerToPeer)
    {
        check_cuda(cudaSetDevice(request.destination_device),
                   "CUDA transfer cudaSetDevice destination");
        check_cuda(cudaMemcpyPeerAsync(request.destination, request.destination_device,
                                       request.source, request.source_device,
                                       request.bytes, stream),
                   "CUDA transfer cudaMemcpyPeerAsync");
    }
    else
    {
        throw std::logic_error("CUDA asynchronous transfer route was not resolved");
    }
    state.completion = {
        request.destination_device, create_event(request.destination_device)};
    check_cuda(cudaEventRecord(state.completion.value, stream),
               "CUDA transfer cudaEventRecord");
    state.completion_recorded = true;
    return result;
}

CudaTransferRequest CudaLocalTransfer::copy_host_to_device_async(
    const std::shared_ptr<PinnedHostBuffer> &source, void *destination,
    std::size_t bytes, int destination_device) const
{
    if (!source || source->data() == nullptr || destination == nullptr || bytes == 0 ||
        bytes > source->size() || destination_device < 0)
    {
        throw std::invalid_argument("CUDA Host-to-device transfer is invalid");
    }
    CudaTransferRequest result;
    result.state_ = std::make_unique<CudaTransferRequest::State>();
    auto &state = *result.state_;
    state.stream_owner = state_;
    state.staging = source;
    const cudaStream_t stream = state_->stream(destination_device);
    state.retain_stream(destination_device, stream);
    const cudaEvent_t destination_ready = record_execution_ready(destination_device);
    state.events.push_back({destination_device, destination_ready});
    gpu::gpu_stream_wait_event(
        stream, destination_ready, destination_device,
        "gpu.stream_wait.transfer_destination_ready",
        "CUDA transfer cudaStreamWaitEvent destination ready");
    check_cuda(cudaMemcpyAsync(destination, source->data(), bytes,
                               cudaMemcpyHostToDevice, stream),
               "CUDA transfer cudaMemcpyAsync Host to device");
    state.completion = {destination_device, create_event(destination_device)};
    check_cuda(cudaEventRecord(state.completion.value, stream),
               "CUDA transfer cudaEventRecord Host to device");
    state.completion_recorded = true;
    return result;
}

CudaTransferRequest CudaLocalTransfer::copy_device_to_host_async(
    const void *source, int source_device,
    const std::shared_ptr<PinnedHostBuffer> &destination, std::size_t bytes,
    cudaEvent_t source_ready) const
{
    if (!destination || destination->data() == nullptr || source == nullptr || bytes == 0 ||
        bytes > destination->size() || source_device < 0)
    {
        throw std::invalid_argument("CUDA device-to-Host transfer is invalid");
    }
    CudaTransferRequest result;
    result.state_ = std::make_unique<CudaTransferRequest::State>();
    auto &state = *result.state_;
    state.stream_owner = state_;
    state.staging = destination;
    const cudaStream_t stream = state_->stream(source_device);
    state.retain_stream(source_device, stream);
    if (source_ready != nullptr)
    {
        gpu::gpu_stream_wait_event(
            stream, source_ready, source_device,
            "gpu.stream_wait.transfer_source_ready",
            "CUDA transfer cudaStreamWaitEvent device to Host");
    }
    check_cuda(cudaMemcpyAsync(destination->data(), source, bytes,
                               cudaMemcpyDeviceToHost, stream),
               "CUDA transfer cudaMemcpyAsync device to Host");
    state.completion = {source_device, create_event(source_device)};
    check_cuda(cudaEventRecord(state.completion.value, stream),
               "CUDA transfer cudaEventRecord device to Host");
    state.completion_recorded = true;
    return result;
}

} // namespace poseidon::runtime_api::communication
