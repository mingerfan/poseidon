#pragma once

#include "resnet50_config.h"

namespace poseidon::benchmark::resnet18_gpu
{

using ResNet18GpuConfig = resnet50_gpu::ResNet50GpuConfig;

ResNet18GpuConfig make_resnet18_gpu_config();

}  // namespace poseidon::benchmark::resnet18_gpu
