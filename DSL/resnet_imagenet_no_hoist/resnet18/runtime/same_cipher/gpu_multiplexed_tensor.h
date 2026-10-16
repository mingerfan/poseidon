#pragma once

#include <cstddef>
#include <vector>

#include "gpu_ckks_runtime.h"

namespace poseidon::benchmark::resnet50_gpu
{

struct GpuMultiplexedTensor
{
    int h = 0;
    int w = 0;
    int c = 0;
    int k = 1;
    int pages_per_cipher = 2;
    std::size_t page_size = 0;
    std::size_t slot_count = 0;
    std::vector<GpuCkksRuntime::DeviceCiphertext> packs;

    std::size_t slot_index(int channel, int row, int col) const;
    std::size_t pack_index(int channel) const;
    void validate() const;
};

GpuMultiplexedTensor encrypt_multiplexed_chw(const std::vector<double> &values, int h,
                                             int w, int c, int k,
                                             const GpuCkksRuntime &runtime);

// Encrypts the 224x224x3 input as stride-2 im2col patches, evaluates the
// 7x7 stem convolution and folded batch normalization on the GPU, and packs
// the 64 output channels into the same multiplexed representation used by the
// residual stages. The input values must already be divided by the network
// boundary (120 for the Trident parameters).
GpuMultiplexedTensor encrypted_stem_conv2d_bn(
    const std::vector<double> &image_chw, int input_h, int input_w, int input_channels,
    int out_channels, int stride, int kernel_h, int kernel_w,
    const std::vector<double> &weights, const std::vector<double> &bn_scale,
    const std::vector<double> &bn_bias, const GpuCkksRuntime &runtime);

std::vector<double> decrypt_multiplexed_chw(const GpuMultiplexedTensor &tensor,
                                            const GpuCkksRuntime &runtime);

GpuMultiplexedTensor batch_norm(const GpuMultiplexedTensor &input,
                                const std::vector<double> &channel_scale,
                                const std::vector<double> &channel_bias,
                                const GpuCkksRuntime &runtime);

// Weight order matches Trident: [out_channel][in_channel][kh][kw]. BN scale
// and bias are applied to the convolution result inside the same packed layer.
GpuMultiplexedTensor conv2d_bn(const GpuMultiplexedTensor &input, int out_channels,
                               int stride, int kernel_h, int kernel_w,
                               const std::vector<double> &weights,
                               const std::vector<double> &bn_scale,
                               const std::vector<double> &bn_bias,
                               const GpuCkksRuntime &runtime);

GpuMultiplexedTensor residual_add(const GpuMultiplexedTensor &left,
                                  const GpuMultiplexedTensor &right,
                                  const GpuCkksRuntime &runtime);

GpuMultiplexedTensor average_pool2d_stride2(const GpuMultiplexedTensor &input,
                                            const GpuCkksRuntime &runtime);

// CIFAR ResNet Option-A shortcut: selects the even spatial coordinates and
// doubles the channel count by padding equally with zero channels.
GpuMultiplexedTensor downsample_shortcut(const GpuMultiplexedTensor &input,
                                         const GpuCkksRuntime &runtime);

// Returns one packed ciphertext with channel averages in slots [0, c).
GpuCkksRuntime::DeviceCiphertext global_average_pool(const GpuMultiplexedTensor &input,
                                                     double output_scale_factor,
                                                     const GpuCkksRuntime &runtime);

// Matrix order is [class][feature]. Each output ciphertext stores its logit
// in slot zero, matching the Trident head representation.
std::vector<GpuCkksRuntime::DeviceCiphertext> fully_connected(
    const GpuCkksRuntime::DeviceCiphertext &features, int feature_count,
    const std::vector<double> &matrix, const std::vector<double> &bias, int class_count,
    const GpuCkksRuntime &runtime);

}  // namespace poseidon::benchmark::resnet50_gpu
