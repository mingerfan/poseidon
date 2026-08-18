#include "poseidon/gpu/gpu_stream_wait_trace.h"

#include "poseidon/gpu/gpu_memory.h"
#include "runtime/thread_trace.hpp"

#include <cmath>
#include <cstdint>
#include <memory>

namespace poseidon::gpu
{
namespace
{

class StreamWaitTiming
{
public:
    explicit StreamWaitTiming(int device) : device_(device)
    {
        gpu_check_cuda(cudaSetDevice(device_), "stream wait trace cudaSetDevice");
        gpu_check_cuda(
            cudaEventCreateWithFlags(&begin_, cudaEventDefault),
            "stream wait trace cudaEventCreate begin");
        try
        {
            gpu_check_cuda(
                cudaEventCreateWithFlags(&end_, cudaEventDefault),
                "stream wait trace cudaEventCreate end");
        }
        catch (...)
        {
            (void)cudaEventDestroy(begin_);
            begin_ = nullptr;
            throw;
        }
    }

    StreamWaitTiming(const StreamWaitTiming &) = delete;
    StreamWaitTiming &operator=(const StreamWaitTiming &) = delete;

    ~StreamWaitTiming()
    {
        int previous_device = -1;
        const bool restore_device =
            cudaGetDevice(&previous_device) == cudaSuccess &&
            previous_device != device_;
        (void)cudaSetDevice(device_);
        if (begin_ != nullptr)
        {
            (void)cudaEventDestroy(begin_);
        }
        if (end_ != nullptr)
        {
            (void)cudaEventDestroy(end_);
        }
        if (restore_device)
        {
            (void)cudaSetDevice(previous_device);
        }
    }

    cudaEvent_t begin() const noexcept
    {
        return begin_;
    }

    cudaEvent_t end() const noexcept
    {
        return end_;
    }

    std::uint64_t elapsed_ns() const
    {
        int previous_device = -1;
        gpu_check_cuda(
            cudaGetDevice(&previous_device),
            "stream wait trace cudaGetDevice");
        gpu_check_cuda(cudaSetDevice(device_), "stream wait trace cudaSetDevice");
        try
        {
            gpu_check_cuda(
                cudaEventSynchronize(end_),
                "stream wait trace cudaEventSynchronize");
            float elapsed_ms = 0.0F;
            gpu_check_cuda(
                cudaEventElapsedTime(&elapsed_ms, begin_, end_),
                "stream wait trace cudaEventElapsedTime");
            gpu_check_cuda(
                cudaSetDevice(previous_device),
                "stream wait trace restore cudaSetDevice");
            return static_cast<std::uint64_t>(
                std::llround(static_cast<double>(elapsed_ms) * 1000000.0));
        }
        catch (...)
        {
            (void)cudaSetDevice(previous_device);
            throw;
        }
    }

private:
    int device_ = 0;
    cudaEvent_t begin_ = nullptr;
    cudaEvent_t end_ = nullptr;
};

} // namespace

void gpu_stream_wait_event(
    cudaStream_t consumer_stream, cudaEvent_t dependency_event,
    int consumer_device, const char *trace_name, const char *error_context)
{
    if (!fhegpu::ThreadTrace::enabled())
    {
        gpu_check_cuda(
            cudaStreamWaitEvent(consumer_stream, dependency_event, 0),
            error_context);
        return;
    }

    auto timing = std::make_shared<StreamWaitTiming>(consumer_device);
    const std::uint64_t submit_begin_ns = fhegpu::ThreadTrace::timestamp_ns();
    gpu_check_cuda(
        cudaEventRecord(timing->begin(), consumer_stream),
        "stream wait trace cudaEventRecord begin");
    gpu_check_cuda(
        cudaStreamWaitEvent(consumer_stream, dependency_event, 0),
        error_context);
    gpu_check_cuda(
        cudaEventRecord(timing->end(), consumer_stream),
        "stream wait trace cudaEventRecord end");
    const std::uint64_t submit_end_ns = fhegpu::ThreadTrace::timestamp_ns();
    fhegpu::ThreadTrace::record_deferred_duration(
        trace_name, reinterpret_cast<const void *>(dependency_event),
        consumer_device, submit_begin_ns, submit_end_ns,
        [timing = std::move(timing)] { return timing->elapsed_ns(); });
}

} // namespace poseidon::gpu
