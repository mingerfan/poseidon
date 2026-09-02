#pragma once

#include "gpu_ckks_runtime.h"

#include "core/tensor.h"

#include <cstddef>
#include <vector>

namespace poseidon::benchmark::qwen_gpu
{

struct GpuTensorLayout
{
    std::size_t tokens = 0;
    std::size_t features = 0;
    std::size_t token_stride = 1024;
    std::size_t slot_count = 0;

    std::size_t feature_chunks() const;
    std::size_t cipher_count() const;
    void validate() const;
};

class GpuEncryptedTensor
{
public:
    using Ciphertext = resnet50_gpu::GpuCkksRuntime::DeviceCiphertext;

    GpuEncryptedTensor() = default;
    GpuEncryptedTensor(GpuTensorLayout layout,
                       std::vector<Ciphertext> ciphertexts);

    const GpuTensorLayout &layout() const noexcept;
    const std::vector<Ciphertext> &ciphertexts() const noexcept;
    std::vector<Ciphertext> &ciphertexts() noexcept;
    const Ciphertext &cipher(std::size_t token, std::size_t chunk) const;
    Ciphertext &cipher(std::size_t token, std::size_t chunk);

private:
    std::size_t cipher_index(std::size_t token, std::size_t chunk) const;

    GpuTensorLayout layout_;
    std::vector<Ciphertext> ciphertexts_;
};

GpuEncryptedTensor encrypt_tensor(
    const qwen::Tensor &tensor,
    const resnet50_gpu::GpuCkksRuntime &runtime,
    std::size_t token_stride = 1024);

qwen::Tensor decrypt_tensor(
    const GpuEncryptedTensor &tensor,
    const resnet50_gpu::GpuCkksRuntime &runtime);

}  // namespace poseidon::benchmark::qwen_gpu
