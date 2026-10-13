#pragma once
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_parameter.h"
#include <cufft.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
namespace poseidon::runtime_api {
// One complete real-slot FP64 embedding + RNS + NTT, on a fixed device/stream.
// The runtime permits only one active submission per instance.
class GpuPlaintextEncoder {
public:
    struct CompletedTiming {
        std::uint64_t calls=0, raw_h2d_bytes=0;
        std::uint64_t raw_h2d_nanoseconds=0, encode_nanoseconds=0;
    };
    GpuPlaintextEncoder(std::size_t degree, int device);
    ~GpuPlaintextEncoder();
    GpuPlaintextEncoder(const GpuPlaintextEncoder &) = delete;
    GpuPlaintextEncoder &operator=(const GpuPlaintextEncoder &) = delete;
    cudaStream_t stream() const {return stream_;}
    std::size_t workspace_bytes(std::size_t limbs) const {return degree_*20 + degree_*limbs*4 + fft_work_bytes_;}
    std::size_t persistent_bytes() const {return degree_*20;}
    std::shared_ptr<void> submit(const double *pinned_slots, double scale,
        const gpu::GpuParameterShard &parameters, gpu::GpuPlaintextData &output);
    // Cumulative completed GPU event durations; wait=false only queries completion.
    // wait=true is for reporting after the caller has drained submissions.
    CompletedTiming completed_timing(bool wait=false);
private:
    struct TimingEvents;
    std::size_t degree_, fft_work_bytes_=0;
    int device_;
    cudaStream_t stream_=nullptr;
    cufftHandle plan_=0;
    gpu::DeviceVector<std::uint32_t> indices_;
    gpu::DeviceVector<double2> twists_;
    std::mutex timing_mutex_;
    std::vector<std::unique_ptr<TimingEvents>> timing_events_;
    CompletedTiming timing_;
};
}
