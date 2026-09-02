#pragma once

#include "qwen_gpu_ops.h"
#include "qwen_gpu_attention.h"

#include "he/encrypted_decoder.h"
#include "model/plain_decoder.h"
#include "model/qwen_config.h"

#include <cstddef>
#include <functional>
#include <string>

namespace poseidon::benchmark::qwen_gpu
{

using DecoderTrace = std::function<void(
    const std::string &, const GpuEncryptedTensor &)>;

// Executes one mathematically complete decoder layer for a one-token prompt.
// Causal attention has exactly one visible key in this case, so softmax is 1
// and the encrypted attention output is the GQA-repeated V projection. This
// is the same single-token shortcut used by Trident's HE attention path.
GpuEncryptedTensor single_token_decoder_layer(
    const GpuEncryptedTensor &input,
    const qwen::DecoderLayerWeights &weights,
    const qwen::QwenConfig &model_config,
    const qwen::he::EncryptedDecoderApproximationConfig &approximation,
    std::size_t layer_index, const Runtime &runtime,
    const DecoderTrace &trace = {});

GpuEncryptedTensor decoder_layer(
    const GpuEncryptedTensor &input,
    const qwen::DecoderLayerWeights &weights,
    const qwen::QwenConfig &model_config,
    const qwen::he::EncryptedDecoderApproximationConfig &approximation,
    std::size_t layer_index, std::size_t position_offset,
    const Runtime &runtime, const DecoderTrace &trace = {},
    GpuKVCache *cache = nullptr);

}  // namespace poseidon::benchmark::qwen_gpu
