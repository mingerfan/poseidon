#pragma once

#include "poseidon/gpu/gpu_parameter.h"
#include <cufft.h>

// Benchmark prototype, not a runtime API. Real full-slot inputs, one device,
// one Q-only level, and rounded coefficient magnitudes strictly below 2^64.
// The caller must call check_result() before using the output.
class GpuEncodeExperiment
{
public:
    enum class Backend { cuda, tensor, tilelang_cuda, tilelang_tensor };
    GpuEncodeExperiment(std::size_t degree, std::size_t limbs, std::size_t batch,
                        const std::vector<std::uint32_t> &indices,
                        const std::vector<double2> &twists);
    ~GpuEncodeExperiment();
    GpuEncodeExperiment(const GpuEncodeExperiment &) = delete;
    GpuEncodeExperiment &operator=(const GpuEncodeExperiment &) = delete;

    // Input and output are contiguous [batch][slots] and [batch][limb][N].
    // prepare includes permutation, double-precision FFT and RNS expansion.
    void prepare(const double *input, double scale, const poseidon::gpu::GpuParameterShard &parameters,
                 bool batched);
    // Input is [batch][N] real coefficients after CPU inverse embedding,
    // including normalization/twist but before scaling or rounding. No FFT.
    void prepare_coefficients(const double *input, double scale,
                              const poseidon::gpu::GpuParameterShard &parameters, bool batched);
    void initialize_tilelang(const poseidon::gpu::GpuParameterShard &parameters);
    void validate_tilelang_ntt(const poseidon::gpu::GpuParameterShard &parameters);
    void ntt(const poseidon::gpu::GpuParameterShard &parameters, Backend backend, bool batched);
    void check_result() const;
    poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> output;

private:
    void prepare_one(const double *input, double scale,
                     const poseidon::gpu::GpuParameterShard &parameters,
                     std::size_t first, std::size_t count, cufftHandle plan);
    std::size_t degree_, limbs_, batch_;
    poseidon::gpu::DeviceVector<std::uint32_t> indices_;
    poseidon::gpu::DeviceVector<double2> twists_, fft_;
    poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> rns_;
    poseidon::gpu::DeviceVector<int> error_;
    poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> tilelang_weights_;
    cufftHandle single_plan_ = 0, batch_plan_ = 0;
};
