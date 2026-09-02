#include "qwen_gpu_tensor.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace poseidon::benchmark::qwen_gpu
{

namespace
{

std::size_t divide_round_up(std::size_t value, std::size_t divisor)
{
    return (value + divisor - 1) / divisor;
}

}  // namespace

std::size_t GpuTensorLayout::feature_chunks() const
{
    return divide_round_up(features, token_stride);
}

std::size_t GpuTensorLayout::cipher_count() const
{
    return tokens * feature_chunks();
}

void GpuTensorLayout::validate() const
{
    if (tokens == 0 || features == 0 || token_stride == 0 ||
        slot_count == 0 || token_stride > slot_count ||
        slot_count % token_stride != 0)
    {
        throw std::invalid_argument("invalid GPU Qwen tensor layout");
    }
}

GpuEncryptedTensor::GpuEncryptedTensor(
    GpuTensorLayout layout, std::vector<Ciphertext> ciphertexts)
    : layout_(std::move(layout)), ciphertexts_(std::move(ciphertexts))
{
    layout_.validate();
    if (ciphertexts_.size() != layout_.cipher_count())
    {
        throw std::invalid_argument(
            "GPU Qwen ciphertext count does not match its layout");
    }
}

const GpuTensorLayout &GpuEncryptedTensor::layout() const noexcept
{
    return layout_;
}

const std::vector<GpuEncryptedTensor::Ciphertext> &
GpuEncryptedTensor::ciphertexts() const noexcept
{
    return ciphertexts_;
}

std::vector<GpuEncryptedTensor::Ciphertext> &
GpuEncryptedTensor::ciphertexts() noexcept
{
    return ciphertexts_;
}

const GpuEncryptedTensor::Ciphertext &GpuEncryptedTensor::cipher(
    std::size_t token, std::size_t chunk) const
{
    return ciphertexts_.at(cipher_index(token, chunk));
}

GpuEncryptedTensor::Ciphertext &GpuEncryptedTensor::cipher(
    std::size_t token, std::size_t chunk)
{
    return ciphertexts_.at(cipher_index(token, chunk));
}

std::size_t GpuEncryptedTensor::cipher_index(
    std::size_t token, std::size_t chunk) const
{
    if (token >= layout_.tokens || chunk >= layout_.feature_chunks())
    {
        throw std::out_of_range("GPU Qwen tensor index is out of range");
    }
    return token * layout_.feature_chunks() + chunk;
}

GpuEncryptedTensor encrypt_tensor(
    const qwen::Tensor &tensor,
    const resnet50_gpu::GpuCkksRuntime &runtime,
    std::size_t token_stride)
{
    if (tensor.rank() != 2)
    {
        throw std::invalid_argument(
            "GPU Qwen encrypt_tensor expects a rank-2 tensor");
    }
    GpuTensorLayout layout{
        tensor.dim(0), tensor.dim(1), token_stride, runtime.slot_count()};
    layout.validate();

    std::vector<GpuEncryptedTensor::Ciphertext> ciphertexts;
    ciphertexts.reserve(layout.cipher_count());
    for (std::size_t token = 0; token < layout.tokens; ++token)
    {
        for (std::size_t chunk = 0; chunk < layout.feature_chunks(); ++chunk)
        {
            std::vector<double> slots(layout.slot_count, 0.0);
            const std::size_t begin = chunk * layout.token_stride;
            const std::size_t end = std::min(
                begin + layout.token_stride, layout.features);
            for (std::size_t feature = begin; feature < end; ++feature)
            {
                slots[feature - begin] = tensor.at(token, feature);
            }
            ciphertexts.push_back(runtime.encrypt(slots));
        }
    }
    return GpuEncryptedTensor(layout, std::move(ciphertexts));
}

qwen::Tensor decrypt_tensor(
    const GpuEncryptedTensor &tensor,
    const resnet50_gpu::GpuCkksRuntime &runtime)
{
    tensor.layout().validate();
    qwen::Tensor result(
        {tensor.layout().tokens, tensor.layout().features});
    for (std::size_t token = 0; token < tensor.layout().tokens; ++token)
    {
        for (std::size_t chunk = 0;
             chunk < tensor.layout().feature_chunks(); ++chunk)
        {
            const auto slots = runtime.decrypt(tensor.cipher(token, chunk));
            const std::size_t begin = chunk * tensor.layout().token_stride;
            const std::size_t end = std::min(
                begin + tensor.layout().token_stride,
                tensor.layout().features);
            for (std::size_t feature = begin; feature < end; ++feature)
            {
                result.at(token, feature) = slots[feature - begin].real();
            }
        }
    }
    return result;
}

}  // namespace poseidon::benchmark::qwen_gpu
