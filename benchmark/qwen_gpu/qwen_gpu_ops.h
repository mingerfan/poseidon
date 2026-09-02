#pragma once

#include "qwen_gpu_tensor.h"

#include "poseidon/advance/polynomial_evaluation.h"

#include <cstddef>
#include <vector>

namespace poseidon::benchmark::qwen_gpu
{

using Runtime = resnet50_gpu::GpuCkksRuntime;

GpuEncryptedTensor add(
    const GpuEncryptedTensor &lhs, const GpuEncryptedTensor &rhs,
    const Runtime &runtime);
GpuEncryptedTensor subtract(
    const GpuEncryptedTensor &lhs, const GpuEncryptedTensor &rhs,
    const Runtime &runtime);
GpuEncryptedTensor add_plain(
    const GpuEncryptedTensor &input, const qwen::Tensor &plain,
    const Runtime &runtime);
GpuEncryptedTensor multiply_plain(
    const GpuEncryptedTensor &input, const qwen::Tensor &plain,
    const Runtime &runtime);
GpuEncryptedTensor multiply(
    const GpuEncryptedTensor &lhs, const GpuEncryptedTensor &rhs,
    const Runtime &runtime);
GpuEncryptedTensor bootstrap(
    const GpuEncryptedTensor &input, const Runtime &runtime,
    double value_scale = 1.0);

// Dense X*W^T+b over each independently encrypted token. The implementation
// uses a 32-way baby-step/giant-step diagonal schedule and shares baby
// rotations across all output chunks.
GpuEncryptedTensor linear(
    const GpuEncryptedTensor &input, const qwen::Tensor &weight,
    const qwen::Tensor *bias, const Runtime &runtime);

GpuEncryptedTensor rope(
    const GpuEncryptedTensor &input, std::size_t head_count,
    std::size_t head_dim, std::size_t position_offset, double theta,
    const Runtime &runtime);

GpuEncryptedTensor token_view(
    const GpuEncryptedTensor &input, std::size_t token,
    const Runtime &runtime);
GpuEncryptedTensor concatenate_tokens(
    std::vector<GpuEncryptedTensor> tokens, const Runtime &runtime);

// Evaluates one Chebyshev polynomial, or per-slot polynomials of a common
// degree, using a balanced 2^k-1 split tree.
GpuEncryptedTensor evaluate_chebyshev(
    const GpuEncryptedTensor &input,
    const poseidon::Polynomial &polynomial,
    const Runtime &runtime);
GpuEncryptedTensor evaluate_chebyshev(
    const GpuEncryptedTensor &input,
    const std::vector<poseidon::Polynomial> &polynomials,
    const std::vector<std::vector<int>> &slot_indexes,
    const Runtime &runtime);

GpuEncryptedTensor affine_to_chebyshev_domain(
    const GpuEncryptedTensor &input,
    const std::vector<double> &minimum_by_feature,
    const std::vector<double> &maximum_by_feature,
    const Runtime &runtime);

GpuEncryptedTensor rms_norm(
    const GpuEncryptedTensor &input, const qwen::Tensor &weight,
    double epsilon, double minimum_variance, double maximum_variance,
    int polynomial_samples, const Runtime &runtime);

GpuEncryptedTensor silu(
    const GpuEncryptedTensor &input,
    const std::vector<double> &minimum_by_feature,
    const std::vector<double> &maximum_by_feature,
    int polynomial_samples, const Runtime &runtime);

GpuEncryptedTensor sigmoid(
    const GpuEncryptedTensor &input, double minimum, double maximum,
    int polynomial_samples, const Runtime &runtime);
GpuEncryptedTensor softplus(
    const GpuEncryptedTensor &input, double minimum, double maximum,
    int polynomial_samples, const Runtime &runtime);

}  // namespace poseidon::benchmark::qwen_gpu
