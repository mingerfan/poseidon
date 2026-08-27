#pragma once

#include <cstddef>
#include <vector>

#include "resnet20_topology.h"
#include "resnet20_weights.h"
#include "gpu_config.h"

namespace poseidon::benchmark::resnet20_gpu
{

using ResNet20GpuConfig = core::GpuConfig;

struct GpuResNet20Result
{
    std::size_t image_id = 0;
    std::size_t completed_blocks = 0;
    int true_label = -1;
    int predicted_label = -1;
    double max_logit_error = 0.0;
    double gpu_only_elapsed_seconds = 0.0;
    std::vector<double> logits;
};

GpuResNet20Result run_gpu_resnet20(std::size_t image_id,
                                   const ResNet20GpuConfig &config,
                                   const ResNet20Topology &topology,
                                   const ResNet20Weights &weights,
                                   std::size_t max_blocks = 9);

// Generates every required direct Galois key before any network operation,
// then performs one untimed cache warmup. The measured pass runs from the stem
// through the encrypted FC output; final decryption is outside the interval.
GpuResNet20Result run_gpu_resnet20_preloaded(
    std::size_t image_id,
    const ResNet20GpuConfig &config,
    const ResNet20Topology &topology,
    const ResNet20Weights &weights);

GpuResNet20Result run_gpu_resnet20_head_check(std::size_t image_id,
                                              const ResNet20GpuConfig &config,
                                              const ResNet20Topology &topology,
                                              const ResNet20Weights &weights);

}  // namespace poseidon::benchmark::resnet20_gpu
