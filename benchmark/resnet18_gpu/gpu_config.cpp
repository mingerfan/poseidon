#include "gpu_config.h"

namespace poseidon::benchmark::resnet18_gpu
{

ResNet18GpuConfig make_resnet18_gpu_config()
{
    // ResNet18 and ResNet50 now share the exact S2C-first Q50/P25 profile.
    return resnet50_gpu::make_resnet50_gpu_config();
}

}  // namespace poseidon::benchmark::resnet18_gpu
