#pragma once

#include "resnet18_topology.h"
#include "resnet18_weights.h"
#include "gpu_config.h"

#include <cstddef>
#include <vector>

namespace poseidon::benchmark::resnet18_gpu
{

struct GpuResNet18Result
{
    std::size_t image_id = 0;
    std::size_t completed_blocks = 0;
    int true_label = -1;
    int predicted_label = -1;
    double max_logit_error = 0.0;
    double gpu_only_elapsed_seconds = 0.0;
    std::vector<double> logits;
};

GpuResNet18Result run_gpu_resnet18(
    std::size_t image_id,
    const ResNet18GpuConfig &config,
    const ResNet18Topology &topology,
    const ResNet18Weights &weights,
    std::size_t max_blocks = 8);

GpuResNet18Result run_gpu_resnet18_preloaded(
    std::size_t image_id,
    const ResNet18GpuConfig &config,
    const ResNet18Topology &topology,
    const ResNet18Weights &weights,
    std::size_t max_blocks = 8);

// For models whose expanded CKKS plaintexts do not all fit device memory:
// prepare one operator's device working set, measure its transfer-free replay,
// release that operator's plaintexts, and accumulate GPU-only stage times.
GpuResNet18Result run_gpu_resnet18_staged_gpu_only(
    std::size_t image_id,
    const ResNet18GpuConfig &config,
    const ResNet18Topology &topology,
    const ResNet18Weights &weights,
    std::size_t max_blocks = 8);

GpuResNet18Result run_gpu_resnet18_head_check(
    std::size_t image_id,
    const ResNet18GpuConfig &config,
    const ResNet18Topology &topology,
    const ResNet18Weights &weights);

}  // namespace poseidon::benchmark::resnet18_gpu
