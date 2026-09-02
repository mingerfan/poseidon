#pragma once

#include "qwen_gpu_ops.h"

#include "he/encrypted_attention.h"
#include "model/qwen_config.h"

#include <cstddef>

namespace poseidon::benchmark::qwen_gpu
{

class GpuKVCache
{
public:
    bool empty() const noexcept;
    std::size_t size() const noexcept;
    void clear();
    void append(
        const GpuEncryptedTensor &key,
        const GpuEncryptedTensor &value,
        const Runtime &runtime);

    const GpuEncryptedTensor &key() const;
    const GpuEncryptedTensor &value() const;

private:
    GpuEncryptedTensor key_;
    GpuEncryptedTensor value_;
};

GpuEncryptedTensor stable_causal_gqa_attention(
    const GpuEncryptedTensor &query,
    const GpuEncryptedTensor &key,
    const GpuEncryptedTensor &value,
    const qwen::QwenConfig &model_config,
    const qwen::he::StableAttentionApproximationConfig &approximation,
    const Runtime &runtime, GpuKVCache *cache = nullptr);

}  // namespace poseidon::benchmark::qwen_gpu
