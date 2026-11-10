#include "poseidon/runtime_api/communication/cuda_local_transfer.h"

#include "poseidon/gpu/gpu_memory.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poseidon::runtime_api::communication
{
namespace
{

std::atomic<std::size_t> pinned_live{0}, pinned_peak{0};
std::atomic<std::size_t> pinned_allocations{0},pinned_frees{0};
std::atomic<std::size_t> pinned_cuda_allocations{0}, pinned_reuses{0};
std::atomic<std::uint64_t> h2d_completed_bytes{0}, h2d_elapsed_nanoseconds{0},
    h2d_completed_count{0};

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

// Pinned Host memory is kept for reuse instead of being returned to CUDA.
// cudaMallocHost and cudaFreeHost take the driver's lock, so every other
// thread's launches and event calls wait behind them, and cudaFreeHost can
// also wait for queued GPU work.  On a host without peer access every
// card-to-card copy is staged through a buffer of its own size: on the 2 x
// RTX 4090 host a fresh buffer cost more than the copy (2.5 GB/s against
// 12.9 GB/s with a preallocated one), and the frees stalled the other card's
// submission thread for seconds (nsys, 2026-11-01/02).
//
// A returned block is cached up to POSEIDON_GPU_RUNTIME_PINNED_CACHE_MB
// (default 4096, 0 disables the cache) and handed to a later request of at
// most its size and at least two thirds of it.  Every user already keeps its
// buffer until the copy that reads or writes it has completed, which is all
// reuse needs.  The cache is never torn down: blocks still cached at exit go
// back with the process, after CUDA itself may already have been unloaded.
class PinnedHostCache
{
public:
    static PinnedHostCache &instance()
    {
        static PinnedHostCache *cache = new PinnedHostCache();
        return *cache;
    }

    void *acquire(std::size_t bytes, std::size_t &capacity)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto found = free_.lower_bound(bytes);
            if (found != free_.end() && found->first - bytes <= found->first / 3)
            {
                capacity = found->first;
                void *data = found->second;
                cached_bytes_ -= capacity;
                free_.erase(found);
                pinned_reuses.fetch_add(1, std::memory_order_relaxed);
                return data;
            }
        }
        capacity = round_up(bytes);
        void *data = nullptr;
        check_cuda(cudaMallocHost(&data, capacity), "CUDA transfer cudaMallocHost");
        pinned_cuda_allocations.fetch_add(1, std::memory_order_relaxed);
        return data;
    }

    void release(void *data, std::size_t capacity) noexcept
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (cached_bytes_ + capacity <= limit_)
            {
                free_.emplace(capacity, data);
                cached_bytes_ += capacity;
                return;
            }
        }
        (void)cudaFreeHost(data);
    }

    std::size_t cached_bytes()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cached_bytes_;
    }

private:
    PinnedHostCache()
    {
        std::size_t megabytes = 4096;
        if (const char *text = std::getenv("POSEIDON_GPU_RUNTIME_PINNED_CACHE_MB"))
        {
            char *end = nullptr;
            const unsigned long long value = std::strtoull(text, &end, 10);
            if (end == text || *end != '\0')
            {
                throw std::invalid_argument(
                    "POSEIDON_GPU_RUNTIME_PINNED_CACHE_MB must be a nonnegative integer");
            }
            megabytes = static_cast<std::size_t>(value);
        }
        limit_ = megabytes << 20;
    }

    // Sizes come from a modulus chain, so many requests differ by a few
    // limbs; rounding lets one cached block serve its neighbours.
    static std::size_t round_up(std::size_t bytes)
    {
        constexpr std::size_t granule = 64 * 1024;
        return bytes > ~std::size_t{0} - granule ? bytes
                                                  : (bytes + granule - 1) / granule * granule;
    }

    std::mutex mutex_;
    std::multimap<std::size_t, void *> free_;
    std::size_t cached_bytes_ = 0;
    std::size_t limit_ = 0;
};

void copy_host_staged(const DeviceBufferCopy &request)
{
    PinnedHostBuffer host(request.bytes);
    check_cuda(cudaSetDevice(request.source_device),
               "CUDA transfer cudaSetDevice source");
    check_cuda(cudaMemcpy(host.data(), request.source, request.bytes, cudaMemcpyDeviceToHost),
               "CUDA transfer device to host");
    check_cuda(cudaSetDevice(request.destination_device),
               "CUDA transfer cudaSetDevice destination");
    check_cuda(cudaMemcpy(request.destination, host.data(), request.bytes, cudaMemcpyHostToDevice),
               "CUDA transfer host to device");
}

cudaStream_t create_stream(int device)
{
    check_cuda(cudaSetDevice(device), "CUDA transfer cudaSetDevice");
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "CUDA transfer cudaStreamCreateWithFlags");
    return stream;
}

cudaEvent_t create_event(int device, unsigned int flags = cudaEventDisableTiming)
{
    check_cuda(cudaSetDevice(device), "CUDA transfer cudaSetDevice");
    cudaEvent_t event = nullptr;
    check_cuda(cudaEventCreateWithFlags(&event, flags),
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
    data_ = PinnedHostCache::instance().acquire(bytes_, capacity_);
    pinned_allocations.fetch_add(1,std::memory_order_relaxed);
    const auto live = pinned_live.fetch_add(bytes_, std::memory_order_relaxed) + bytes_;
    auto peak = pinned_peak.load(std::memory_order_relaxed);
    while (peak < live && !pinned_peak.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {}

}

PinnedHostBuffer::~PinnedHostBuffer()
{
    if (data_ != nullptr)
    {
        PinnedHostCache::instance().release(data_, capacity_);
        pinned_frees.fetch_add(1,std::memory_order_relaxed);
        pinned_live.fetch_sub(bytes_, std::memory_order_relaxed);
    }
}

std::size_t PinnedHostBuffer::live_bytes() noexcept { return pinned_live.load(std::memory_order_relaxed); }
std::size_t PinnedHostBuffer::peak_bytes() noexcept { return pinned_peak.load(std::memory_order_relaxed); }
std::size_t PinnedHostBuffer::allocation_calls() noexcept { return pinned_allocations.load(std::memory_order_relaxed); }
std::size_t PinnedHostBuffer::free_calls() noexcept { return pinned_frees.load(std::memory_order_relaxed); }
std::size_t PinnedHostBuffer::cuda_allocation_calls() noexcept { return pinned_cuda_allocations.load(std::memory_order_relaxed); }
std::size_t PinnedHostBuffer::reuse_calls() noexcept { return pinned_reuses.load(std::memory_order_relaxed); }
std::size_t PinnedHostBuffer::cached_bytes() noexcept
{
    try { return PinnedHostCache::instance().cached_bytes(); }
    catch (...) { return 0; }
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
            if (cudaEventSynchronize(completion.value) == cudaSuccess)
            {
                try { record_h2d_completion(); } catch (...) {}
            }
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
        if (h2d_start.value != nullptr)
        {
            (void)cudaSetDevice(h2d_start.device);
            (void)cudaEventDestroy(h2d_start.value);
        }
    }

    // Caller holds wait_mutex, or owns the State exclusively during destruction.
    void record_h2d_completion()
    {
        if (h2d_start.value == nullptr || h2d_counted) return;
        float milliseconds = 0;
        check_cuda(cudaEventElapsedTime(&milliseconds, h2d_start.value, completion.value),
                   "CUDA transfer cudaEventElapsedTime Host to device");
        if (!std::isfinite(milliseconds) || milliseconds < 0)
            throw std::runtime_error("CUDA Host-to-device event duration is invalid");
        const auto nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(milliseconds) * 1000000.0));
        h2d_completed_bytes.fetch_add(h2d_transfer_bytes, std::memory_order_relaxed);
        h2d_elapsed_nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
        h2d_completed_count.fetch_add(1, std::memory_order_relaxed);
        h2d_counted = true;
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
    Event h2d_start;
    std::uint64_t h2d_transfer_bytes = 0;
    std::shared_ptr<PinnedHostBuffer> staging;
    std::mutex wait_mutex;
    bool completion_recorded = false;
    bool waited = false;
    bool h2d_counted = false;
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
                submission_mutexes.try_emplace(device);
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

    std::mutex &submission_mutex(int device)
    {
        const auto found = submission_mutexes.find(device);
        if (found == submission_mutexes.end())
            throw std::invalid_argument("CUDA transfer device is not configured");
        return found->second;
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
    std::map<int, std::mutex> submission_mutexes;
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

bool CudaTransferRequest::holds_staging() const
{
    return state_ && state_->staging != nullptr;
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

bool CudaTransferRequest::collect_completed()
{
    const auto event = completion_event();
    std::unique_lock<std::mutex> lock(state_->wait_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return false;
    if (state_->waited)
    {
        return true;
    }
    check_cuda(cudaSetDevice(state_->completion.device),
               "CUDA transfer cudaSetDevice");
    const auto status = cudaEventQuery(event);
    if (status == cudaErrorNotReady)
    {
        return false;
    }
    check_cuda(status, "CUDA transfer cudaEventQuery");
    state_->record_h2d_completion();
    state_->waited = true;
    state_->staging.reset();
    return true;
}

void CudaTransferRequest::wait()
{
    if (!state_ || !state_->completion_recorded ||
        state_->completion.value == nullptr)
    {
        throw std::logic_error("CUDA transfer request has no completion event");
    }
    std::lock_guard<std::mutex> lock(state_->wait_mutex);
    if (state_->waited)
    {
        return;
    }
    check_cuda(cudaSetDevice(state_->completion.device),
               "CUDA transfer cudaSetDevice");
    check_cuda(cudaEventSynchronize(state_->completion.value),
               "CUDA transfer cudaEventSynchronize");
    state_->record_h2d_completion();
    state_->waited = true;
    // The staging buffer is only needed until the copy that reads or writes it
    // has finished, which is exactly now.  Holding it until the request itself
    // is destroyed charged its cudaFreeHost to whoever dropped the last
    // reference to the request: for the initialisation uploads that was the end
    // of the online phase, where 122 buffers cost ~200 ms -- 28% of a 733 ms
    // latency measurement, none of it computation.  A download keeps its own
    // shared_ptr in CommHandle::State::deferred_outputs, so the buffer it still
    // has to read survives this reset.
    state_->staging.reset();
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

std::uint64_t CudaLocalTransfer::h2d_bytes() noexcept
{
    return h2d_completed_bytes.load(std::memory_order_relaxed);
}

std::uint64_t CudaLocalTransfer::h2d_nanoseconds() noexcept
{
    return h2d_elapsed_nanoseconds.load(std::memory_order_relaxed);
}

std::uint64_t CudaLocalTransfer::h2d_completed_calls() noexcept
{
    return h2d_completed_count.load(std::memory_order_relaxed);
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

    // Keep each stream's dependency/copy/event sequence indivisible. Otherwise
    // another submitter's readiness wait can enter an H2D timing interval.
    std::unique_lock<std::mutex> destination_lock(
        state_->submission_mutex(request.destination_device), std::defer_lock);
    std::unique_lock<std::mutex> source_lock;
    if (route == CudaTransferRoute::HostStaged)
    {
        source_lock = std::unique_lock<std::mutex>(
            state_->submission_mutex(request.source_device), std::defer_lock);
        std::lock(destination_lock, source_lock);
    }
    else destination_lock.lock();

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
            check_cuda(cudaStreamWaitEvent(source_stream, source_ready, 0),
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
        check_cuda(cudaStreamWaitEvent(destination_stream, destination_ready, 0),
                   "CUDA transfer cudaStreamWaitEvent destination ready");
        check_cuda(cudaStreamWaitEvent(destination_stream, staged, 0),
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
    check_cuda(cudaStreamWaitEvent(stream, destination_ready, 0),
               "CUDA transfer cudaStreamWaitEvent destination ready");
    if (source_ready != nullptr)
    {
        check_cuda(cudaStreamWaitEvent(stream, source_ready, 0),
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
    std::lock_guard<std::mutex> submission_lock(state_->submission_mutex(destination_device));
    CudaTransferRequest result;
    result.state_ = std::make_unique<CudaTransferRequest::State>();
    auto &state = *result.state_;
    state.stream_owner = state_;
    state.staging = source;
    const cudaStream_t stream = state_->stream(destination_device);
    state.retain_stream(destination_device, stream);
    const cudaEvent_t destination_ready = record_execution_ready(destination_device);
    state.events.push_back({destination_device, destination_ready});
    check_cuda(cudaStreamWaitEvent(stream, destination_ready, 0),
               "CUDA transfer cudaStreamWaitEvent destination ready");
    state.h2d_start = {destination_device, create_event(destination_device, cudaEventDefault)};
    state.completion = {destination_device, create_event(destination_device, cudaEventDefault)};
    state.h2d_transfer_bytes = bytes;
    check_cuda(cudaEventRecord(state.h2d_start.value, stream),
               "CUDA transfer cudaEventRecord Host to device start");
    check_cuda(cudaMemcpyAsync(destination, source->data(), bytes,
                               cudaMemcpyHostToDevice, stream),
               "CUDA transfer cudaMemcpyAsync Host to device");
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
    std::lock_guard<std::mutex> submission_lock(state_->submission_mutex(source_device));
    CudaTransferRequest result;
    result.state_ = std::make_unique<CudaTransferRequest::State>();
    auto &state = *result.state_;
    state.stream_owner = state_;
    state.staging = destination;
    const cudaStream_t stream = state_->stream(source_device);
    state.retain_stream(source_device, stream);
    if (source_ready != nullptr)
    {
        check_cuda(cudaStreamWaitEvent(stream, source_ready, 0),
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
