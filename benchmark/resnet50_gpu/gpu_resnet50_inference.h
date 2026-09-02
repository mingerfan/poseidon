#pragma once

#include "gpu_ckks_runtime.h"
#include "resnet50_topology.h"
#include "resnet50_weights.h"

#include <cstddef>
#include <vector>

namespace poseidon::benchmark::resnet50_gpu
{

struct GpuResNet50Result
{
    std::size_t image_id = 0;
    std::size_t completed_blocks = 0;
    int true_label = -1;
    int predicted_label = -1;
    double max_logit_error = 0.0;
    double gpu_only_elapsed_seconds = 0.0;
    std::vector<double> logits;
};

// Runs the Trident ResNet50 polynomial network. CPU work is restricted to
// loading/packing/encoding inputs and weights and decrypting the final logits;
// convolution, BN, pooling, residual addition, bootstrapping, polynomial ReLU
// and fully connected evaluation all use GpuCkksRuntime operations.
GpuResNet50Result run_gpu_resnet50(
    std::size_t image_id,
    const ResNet50GpuConfig &config,
    const ResNet50Topology &topology,
    const ResNet50Weights &weights,
    std::size_t max_blocks = 16);

// Runs an untimed preparation pass and then times an identical replay whose
// input, keys, bootstrap matrices and encoded operands are already on device.
// Use only for a prefix whose expanded operands fit device memory.
GpuResNet50Result run_gpu_resnet50_preloaded(
    std::size_t image_id,
    const ResNet50GpuConfig &config,
    const ResNet50Topology &topology,
    const ResNet50Weights &weights,
    std::size_t max_blocks = 16);

// Prepares and measures one operator at a time, keeping the live activation,
// evaluation keys and bootstrap matrices on GPU while bounding plaintext
// cache memory for the full network.
GpuResNet50Result run_gpu_resnet50_staged_gpu_only(
    std::size_t image_id,
    const ResNet50GpuConfig &config,
    const ResNet50Topology &topology,
    const ResNet50Weights &weights,
    std::size_t max_blocks = 16);

// Computes the feature extractor in plaintext, then encrypts the real
// 7x7x2048 tensor and validates the production k=16 GPU pooling/FC head.
GpuResNet50Result run_gpu_resnet50_head_check(
    std::size_t image_id,
    const ResNet50GpuConfig &config,
    const ResNet50Topology &topology,
    const ResNet50Weights &weights);

}  // namespace poseidon::benchmark::resnet50_gpu
