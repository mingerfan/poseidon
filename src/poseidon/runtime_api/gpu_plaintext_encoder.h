#pragma once
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_parameter.h"
#include <cufft.h>
#include <memory>
namespace poseidon::runtime_api {
// One complete real-slot FP64 embedding + RNS + NTT, on a fixed device/stream.
// The runtime permits only one active submission per instance.
class GpuPlaintextEncoder {
public:
    GpuPlaintextEncoder(std::size_t degree, int device);
    ~GpuPlaintextEncoder();
    GpuPlaintextEncoder(const GpuPlaintextEncoder &) = delete;
    GpuPlaintextEncoder &operator=(const GpuPlaintextEncoder &) = delete;
    cudaStream_t stream() const {return stream_;}
    std::size_t workspace_bytes(std::size_t limbs) const {return degree_*20 + degree_*limbs*4 + fft_work_bytes_;}
    std::size_t persistent_bytes() const {return degree_*20;}
    std::shared_ptr<void> submit(const double *pinned_slots, double scale,
        const gpu::GpuParameterShard &parameters, gpu::GpuPlaintextData &output);
private:
    std::size_t degree_, fft_work_bytes_=0;
    int device_;
    cudaStream_t stream_=nullptr;
    cufftHandle plan_=0;
    gpu::DeviceVector<std::uint32_t> indices_;
    gpu::DeviceVector<double2> twists_;
};
}
