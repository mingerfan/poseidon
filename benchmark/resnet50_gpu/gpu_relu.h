#pragma once

#include "gpu_ckks_runtime.h"
#include "gpu_multiplexed_tensor.h"

namespace poseidon::benchmark::resnet50_gpu
{

struct GpuReluConfig
{
    int alpha = 13;
    double scaled_value = 1.7;
};

GpuCkksRuntime::DeviceCiphertext polynomial_relu(
    const GpuCkksRuntime::DeviceCiphertext &input,
    const GpuCkksRuntime &runtime,
    const GpuReluConfig &config = {});

GpuMultiplexedTensor polynomial_relu(
    const GpuMultiplexedTensor &input,
    const GpuCkksRuntime &runtime,
    const GpuReluConfig &config = {});

double polynomial_relu_reference(
    double input,
    const GpuReluConfig &config = {});


#if defined(POSEIDON_GPU_NATIVE50_PRECISION_BOOTSTRAP)
GpuCkksRuntime::DeviceCiphertext polynomial_relu_maxpool(const GpuCkksRuntime::DeviceCiphertext &,const GpuCkksRuntime &);
double polynomial_relu_maxpool_reference(double);
#endif

}  // namespace poseidon::benchmark::resnet50_gpu
