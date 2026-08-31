#include "gpu_multiplexed_tensor.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace poseidon::benchmark::resnet20_gpu::core
{
namespace
{

std::size_t ceil_div(std::size_t value, std::size_t divisor)
{
    return (value + divisor - 1) / divisor;
}

int channels_per_page(int k)
{
    if (k <= 0)
    {
        throw std::invalid_argument("multiplexed k must be positive");
    }
    return k * k;
}

GpuMultiplexedTensor make_shape(int h, int w, int c, int k, std::size_t slot_count)
{
    if (h <= 0 || w <= 0 || c <= 0)
    {
        throw std::invalid_argument("multiplexed tensor dimensions must be positive");
    }
    GpuMultiplexedTensor result;
    result.h = h;
    result.w = w;
    result.c = c;
    result.k = k;
    result.slot_count = slot_count;
    result.page_size =
        static_cast<std::size_t>(h * k) * static_cast<std::size_t>(w * k);
    const std::size_t maximum_pages = slot_count / result.page_size;
    if (maximum_pages == 0)
    {
        throw std::invalid_argument("multiplexed tensor does not fit CKKS slots");
    }
    // Use the largest power-of-two page count that fits the ciphertext. This
    // retains the existing two-page ImageNet layout while allowing CIFAR
    // tensors (page_size=1024) to use all 32 pages of a 32768-slot ciphertext.
    result.pages_per_cipher = 1;
    while (static_cast<std::size_t>(result.pages_per_cipher) * 2 <= maximum_pages)
    {
        result.pages_per_cipher *= 2;
    }
    const std::size_t pages = ceil_div(c, channels_per_page(k));
    result.packs.resize(ceil_div(pages, result.pages_per_cipher));
    return result;
}

void require_same_layout(const GpuMultiplexedTensor &left,
                         const GpuMultiplexedTensor &right)
{
    if (left.h != right.h || left.w != right.w || left.c != right.c ||
        left.k != right.k || left.pages_per_cipher != right.pages_per_cipher ||
        left.page_size != right.page_size || left.slot_count != right.slot_count ||
        left.packs.size() != right.packs.size())
    {
        throw std::invalid_argument("multiplexed residual layout mismatch");
    }
}

std::vector<int> channels_for_pack(const GpuMultiplexedTensor &tensor, std::size_t pack)
{
    std::vector<int> result;
    for (int channel = 0; channel < tensor.c; ++channel)
    {
        if (tensor.pack_index(channel) == pack)
        {
            result.push_back(channel);
        }
    }
    return result;
}

GpuCkksRuntime::DeviceCiphertext sum_local_channels_to_base(
    const GpuCkksRuntime::DeviceCiphertext &source, const GpuMultiplexedTensor &shape,
    std::size_t pack_index, const GpuCkksRuntime &runtime)
{
    auto sum = runtime.drop_to_q_count(source, source.meta.q_count);
    for (int step = 1; step < shape.k; step <<= 1)
    {
        auto rotated = runtime.rotate_composed(sum, step);
        sum = runtime.add(sum, rotated);
    }
    for (int step = 1; step < shape.k; step <<= 1)
    {
        auto rotated = runtime.rotate_composed(
            sum, static_cast<long long>(step) * shape.k * shape.w);
        sum = runtime.add(sum, rotated);
    }
    const std::size_t active_pages =
        ceil_div(channels_for_pack(shape, pack_index).size(),
                 static_cast<std::size_t>(channels_per_page(shape.k)));
    if (active_pages <= 1)
    {
        return sum;
    }
    if ((active_pages & (active_pages - 1)) == 0)
    {
        auto page_sum = runtime.drop_to_q_count(sum, sum.meta.q_count);
        for (std::size_t step = 1; step < active_pages; step <<= 1)
        {
            auto rotated = runtime.rotate_composed(
                page_sum, static_cast<long long>(step) * shape.page_size);
            page_sum = runtime.add(page_sum, rotated);
        }
        return page_sum;
    }
    auto page_sum = runtime.drop_to_q_count(sum, sum.meta.q_count);
    for (std::size_t page = 1; page < active_pages; ++page)
    {
        auto rotated = runtime.rotate_composed(
            sum, static_cast<long long>(page) * shape.page_size);
        page_sum = runtime.add(page_sum, rotated);
    }
    return page_sum;
}

long long output_channel_shift(const GpuMultiplexedTensor &output, int output_channel)
{
    const int page_channels = channels_per_page(output.k);
    const int page = output_channel / page_channels;
    const int local_page = page % output.pages_per_cipher;
    const int channel_in_page = output_channel % page_channels;
    const int row_offset = channel_in_page / output.k;
    const int col_offset = channel_in_page % output.k;
    return -static_cast<long long>(local_page) * output.page_size -
           static_cast<long long>(row_offset) * output.w * output.k - col_offset;
}

}  // namespace

std::size_t GpuMultiplexedTensor::slot_index(int channel, int row, int col) const
{
    if (channel < 0 || channel >= c || row < 0 || row >= h || col < 0 || col >= w)
    {
        throw std::out_of_range("multiplexed tensor index is out of range");
    }
    const int page_channels = channels_per_page(k);
    const int page = channel / page_channels;
    const int channel_in_page = channel % page_channels;
    const int local_page = page % pages_per_cipher;
    const int row_offset = channel_in_page / k;
    const int col_offset = channel_in_page % k;
    return static_cast<std::size_t>(local_page) * page_size +
           static_cast<std::size_t>(row * k + row_offset) *
               static_cast<std::size_t>(w * k) +
           static_cast<std::size_t>(col * k + col_offset);
}

std::size_t GpuMultiplexedTensor::pack_index(int channel) const
{
    if (channel < 0 || channel >= c)
    {
        throw std::out_of_range("multiplexed channel is out of range");
    }
    const int page = channel / channels_per_page(k);
    return static_cast<std::size_t>(page / pages_per_cipher);
}

void GpuMultiplexedTensor::validate() const
{
    if (h <= 0 || w <= 0 || c <= 0 || k <= 0 || packs.empty() ||
        page_size * static_cast<std::size_t>(pages_per_cipher) > slot_count)
    {
        throw std::invalid_argument("invalid GPU multiplexed tensor");
    }
}

GpuMultiplexedTensor encrypt_multiplexed_chw(const std::vector<double> &values, int h,
                                             int w, int c, int k,
                                             const GpuCkksRuntime &runtime)
{
    if (values.size() != static_cast<std::size_t>(h * w * c))
    {
        throw std::invalid_argument("CHW input value count mismatch");
    }
    auto result = make_shape(h, w, c, k, runtime.slot_count());
    std::vector<std::vector<double>> packed(
        result.packs.size(), std::vector<double>(result.slot_count, 0.0));
    for (int channel = 0; channel < c; ++channel)
    {
        for (int row = 0; row < h; ++row)
        {
            for (int col = 0; col < w; ++col)
            {
                packed[result.pack_index(channel)][result.slot_index(channel, row,
                                                                     col)] =
                    values[(static_cast<std::size_t>(channel) * h + row) * w + col];
            }
        }
    }
    for (std::size_t pack = 0; pack < result.packs.size(); ++pack)
    {
        result.packs[pack] = runtime.encrypt(packed[pack]);
    }
    return result;
}

GpuMultiplexedTensor encrypted_stem_conv2d_bn(
    const std::vector<double> &image_chw, int input_h, int input_w, int input_channels,
    int out_channels, int stride, int kernel_h, int kernel_w,
    const std::vector<double> &weights, const std::vector<double> &bn_scale,
    const std::vector<double> &bn_bias, const GpuCkksRuntime &runtime)
{
    if ((stride != 1 && stride != 2) || kernel_h <= 0 || kernel_w <= 0 ||
        kernel_h % 2 == 0 || kernel_w % 2 == 0 || input_h <= 0 || input_w <= 0 ||
        input_channels <= 0 || out_channels <= 0)
    {
        throw std::invalid_argument("GPU stem convolution shape is unsupported");
    }
    if (image_chw.size() !=
            static_cast<std::size_t>(input_h * input_w * input_channels) ||
        weights.size() != static_cast<std::size_t>(out_channels * input_channels *
                                                   kernel_h * kernel_w) ||
        bn_scale.size() != static_cast<std::size_t>(out_channels) ||
        bn_bias.size() != static_cast<std::size_t>(out_channels))
    {
        throw std::invalid_argument("GPU stem convolution parameter mismatch");
    }

    const int output_h = input_h / stride;
    const int output_w = input_w / stride;
    const std::size_t spatial_count = static_cast<std::size_t>(output_h) * output_w;
    if (spatial_count > runtime.slot_count())
    {
        throw std::invalid_argument("GPU stem im2col patch does not fit CKKS slots");
    }

    const int pad_h = kernel_h / 2;
    const int pad_w = kernel_w / 2;
    std::vector<GpuCkksRuntime::DeviceCiphertext> patches;
    patches.reserve(static_cast<std::size_t>(input_channels * kernel_h * kernel_w));
    for (int input_channel = 0; input_channel < input_channels; ++input_channel)
    {
        for (int kh = 0; kh < kernel_h; ++kh)
        {
            for (int kw = 0; kw < kernel_w; ++kw)
            {
                std::vector<double> slots(runtime.slot_count(), 0.0);
                for (int output_row = 0; output_row < output_h; ++output_row)
                {
                    for (int output_col = 0; output_col < output_w; ++output_col)
                    {
                        const int input_row = output_row * stride + kh - pad_h;
                        const int input_col = output_col * stride + kw - pad_w;
                        if (input_row < 0 || input_row >= input_h || input_col < 0 ||
                            input_col >= input_w)
                        {
                            continue;
                        }
                        const auto input_index =
                            (static_cast<std::size_t>(input_channel) * input_h +
                             input_row) *
                                input_w +
                            input_col;
                        slots[static_cast<std::size_t>(output_row) * output_w +
                              output_col] = image_chw[input_index];
                    }
                }
                patches.push_back(runtime.encrypt(slots));
            }
        }
    }

    auto output =
        make_shape(output_h, output_w, out_channels, /*k=*/1, runtime.slot_count());
    for (std::size_t output_pack = 0; output_pack < output.packs.size(); ++output_pack)
    {
        std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> pack_sum;
        for (const int output_channel : channels_for_pack(output, output_pack))
        {
            std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> channel_sum;
            for (int input_channel = 0; input_channel < input_channels; ++input_channel)
            {
                for (int kh = 0; kh < kernel_h; ++kh)
                {
                    for (int kw = 0; kw < kernel_w; ++kw)
                    {
                        const auto patch_index =
                            (static_cast<std::size_t>(input_channel) * kernel_h + kh) *
                                kernel_w +
                            kw;
                        const auto weight_index =
                            ((static_cast<std::size_t>(output_channel) *
                                  input_channels +
                              input_channel) *
                                 kernel_h +
                             kh) *
                                kernel_w +
                            kw;
                        const double coefficient =
                            weights[weight_index] * bn_scale[output_channel];
                        if (coefficient == 0.0)
                        {
                            continue;
                        }
                        auto term = runtime.multiply_plain_scalar_rescale(
                            patches[patch_index], coefficient);
                        if (channel_sum)
                        {
                            *channel_sum = runtime.add(*channel_sum, term);
                        }
                        else
                        {
                            channel_sum =
                                std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                                    std::move(term));
                        }
                    }
                }
            }
            if (!channel_sum)
            {
                throw std::runtime_error("GPU stem produced an empty output channel");
            }
            const long long shift = output_channel_shift(output, output_channel);
            if (shift != 0)
            {
                *channel_sum = runtime.rotate_composed(*channel_sum, shift);
            }
            if (pack_sum)
            {
                *pack_sum = runtime.add(*pack_sum, *channel_sum);
            }
            else
            {
                pack_sum = std::move(channel_sum);
            }
        }

        std::vector<double> bias_slots(output.slot_count, 0.0);
        for (const int output_channel : channels_for_pack(output, output_pack))
        {
            for (int row = 0; row < output.h; ++row)
            {
                for (int col = 0; col < output.w; ++col)
                {
                    bias_slots[output.slot_index(output_channel, row, col)] =
                        bn_bias[output_channel];
                }
            }
        }
        output.packs[output_pack] = runtime.add_plain(*pack_sum, bias_slots);
    }
    return output;
}

std::vector<double> decrypt_multiplexed_chw(const GpuMultiplexedTensor &tensor,
                                            const GpuCkksRuntime &runtime)
{
    tensor.validate();
    std::vector<std::vector<std::complex<double>>> decoded;
    decoded.reserve(tensor.packs.size());
    for (const auto &pack : tensor.packs)
    {
        decoded.push_back(runtime.decrypt(pack));
    }
    std::vector<double> result(
        static_cast<std::size_t>(tensor.h * tensor.w * tensor.c));
    for (int channel = 0; channel < tensor.c; ++channel)
    {
        for (int row = 0; row < tensor.h; ++row)
        {
            for (int col = 0; col < tensor.w; ++col)
            {
                result[(static_cast<std::size_t>(channel) * tensor.h + row) * tensor.w +
                       col] = decoded[tensor.pack_index(channel)]
                                     [tensor.slot_index(channel, row, col)]
                                         .real();
            }
        }
    }
    return result;
}

GpuMultiplexedTensor batch_norm(const GpuMultiplexedTensor &input,
                                const std::vector<double> &channel_scale,
                                const std::vector<double> &channel_bias,
                                const GpuCkksRuntime &runtime)
{
    input.validate();
    if (channel_scale.size() != static_cast<std::size_t>(input.c) ||
        channel_bias.size() != static_cast<std::size_t>(input.c))
    {
        throw std::invalid_argument("batch-norm channel parameter mismatch");
    }
    auto output = make_shape(input.h, input.w, input.c, input.k, input.slot_count);
    for (std::size_t pack = 0; pack < input.packs.size(); ++pack)
    {
        std::vector<double> scales(input.slot_count, 0.0);
        std::vector<double> biases(input.slot_count, 0.0);
        for (int channel = 0; channel < input.c; ++channel)
        {
            if (input.pack_index(channel) != pack)
            {
                continue;
            }
            for (int row = 0; row < input.h; ++row)
            {
                for (int col = 0; col < input.w; ++col)
                {
                    const auto slot = input.slot_index(channel, row, col);
                    scales[slot] = channel_scale[channel];
                    biases[slot] = channel_bias[channel];
                }
            }
        }
        auto scaled = runtime.multiply_plain_rescale(input.packs[pack], scales);
        output.packs[pack] = runtime.add_plain(scaled, biases);
    }
    return output;
}

GpuMultiplexedTensor conv2d_bn(const GpuMultiplexedTensor &input, int out_channels,
                               int stride, int kernel_h, int kernel_w,
                               const std::vector<double> &weights,
                               const std::vector<double> &bn_scale,
                               const std::vector<double> &bn_bias,
                               const GpuCkksRuntime &runtime)
{
    input.validate();
    if ((stride != 1 && stride != 2) || kernel_h <= 0 || kernel_w <= 0 ||
        kernel_h % 2 == 0 || kernel_w % 2 == 0)
    {
        throw std::invalid_argument("GPU multiplexed convolution shape is unsupported");
    }
    const std::size_t expected_weights =
        static_cast<std::size_t>(out_channels) * input.c * kernel_h * kernel_w;
    if (weights.size() != expected_weights ||
        bn_scale.size() != static_cast<std::size_t>(out_channels) ||
        bn_bias.size() != static_cast<std::size_t>(out_channels))
    {
        throw std::invalid_argument("GPU multiplexed convolution parameter mismatch");
    }

    const int output_k = stride == 1 ? input.k : input.k * 2;
    auto output = make_shape(input.h / stride, input.w / stride, out_channels, output_k,
                             input.slot_count);
    const int pad_h = kernel_h / 2;
    const int pad_w = kernel_w / 2;

    // Trident's CIFAR layout uses the unused pages as replicas of the input
    // tensor. A single compact plaintext can then carry a different output
    // channel's weights in every replica. This reduces the expensive 3x3
    // plaintext products from out_channels*9 to ceil(out_channels/replicas)*9.
    // Keep the original implementation below as the general multi-pack path.
    const std::size_t active_input_pages = ceil_div(
        static_cast<std::size_t>(input.c),
        static_cast<std::size_t>(channels_per_page(input.k)));
    const std::size_t input_replicas =
        static_cast<std::size_t>(input.pages_per_cipher) / active_input_pages;
    if (input.packs.size() == 1 && output.packs.size() == 1 &&
        input_replicas > 1 &&
        active_input_pages * input_replicas ==
            static_cast<std::size_t>(input.pages_per_cipher))
    {
        const std::size_t input_span = active_input_pages * input.page_size;
        auto replicated = runtime.drop_to_q_count(
            input.packs.front(), input.packs.front().meta.q_count);
        for (std::size_t replica = 1; replica < input_replicas; ++replica)
        {
            auto copy = runtime.rotate_composed(
                input.packs.front(),
                -static_cast<long long>(replica * input_span));
            replicated = runtime.add(replicated, copy);
        }

        const std::size_t kernel_count =
            static_cast<std::size_t>(kernel_h * kernel_w);
        std::vector<std::unique_ptr<GpuCkksRuntime::DeviceCiphertext>>
            rotated_inputs(kernel_count);
        for (int kh = 0; kh < kernel_h; ++kh)
        {
            for (int kw = 0; kw < kernel_w; ++kw)
            {
                const long long spatial_step =
                    static_cast<long long>(input.k) * input.k * input.w *
                        (kh - pad_h) +
                    static_cast<long long>(input.k) * (kw - pad_w);
                if (spatial_step != 0)
                {
                    rotated_inputs[static_cast<std::size_t>(
                        kh * kernel_w + kw)] =
                        std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                            runtime.rotate_composed(replicated, spatial_step));
                }
            }
        }

        const std::size_t output_groups = ceil_div(
            static_cast<std::size_t>(out_channels), input_replicas);
        std::cout << "[GPU multiplexed conv] input_replicas=" << input_replicas
                  << " output_groups=" << output_groups
                  << " lazy_plain_accumulate=on\n";
        std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> pack_sum;
        for (std::size_t group = 0; group < output_groups; ++group)
        {
            std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> group_sum;
            for (int kh = 0; kh < kernel_h; ++kh)
            {
                for (int kw = 0; kw < kernel_w; ++kw)
                {
                    const auto kernel_index = static_cast<std::size_t>(
                        kh * kernel_w + kw);
                    const auto *spatial = &replicated;
                    if (rotated_inputs[kernel_index])
                    {
                        spatial = rotated_inputs[kernel_index].get();
                    }

                    std::vector<double> compact_weight(input.slot_count, 0.0);
                    bool nonzero = false;
                    for (std::size_t replica = 0; replica < input_replicas;
                         ++replica)
                    {
                        const std::size_t output_channel =
                            group * input_replicas + replica;
                        if (output_channel >= static_cast<std::size_t>(out_channels))
                        {
                            continue;
                        }
                        for (int input_channel = 0; input_channel < input.c;
                             ++input_channel)
                        {
                            const std::size_t weight_index =
                                ((output_channel * input.c + input_channel) *
                                     kernel_h +
                                 kh) *
                                    kernel_w +
                                kw;
                            const double coefficient =
                                weights[weight_index] * bn_scale[output_channel];
                            if (coefficient == 0.0)
                            {
                                continue;
                            }
                            for (int oh = 0; oh < output.h; ++oh)
                            {
                                for (int ow = 0; ow < output.w; ++ow)
                                {
                                    const int ih = oh * stride + kh - pad_h;
                                    const int iw = ow * stride + kw - pad_w;
                                    if (ih < 0 || ih >= input.h || iw < 0 ||
                                        iw >= input.w)
                                    {
                                        continue;
                                    }
                                    const auto slot = input.slot_index(
                                                          input_channel,
                                                          oh * stride,
                                                          ow * stride) +
                                                      replica * input_span;
                                    compact_weight[slot] = coefficient;
                                    nonzero = true;
                                }
                            }
                        }
                    }
                    if (!nonzero)
                    {
                        continue;
                    }
                    const double plain_scale =
                        runtime.last_modulus_value(*spatial);
                    if (group_sum)
                    {
                        runtime.multiply_plain_accumulate(
                            *spatial, compact_weight, plain_scale, *group_sum);
                    }
                    else
                    {
                        group_sum =
                            std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                                runtime.multiply_plain(
                                    *spatial, compact_weight, plain_scale));
                    }
                }
            }
            if (!group_sum)
            {
                throw std::runtime_error(
                    "GPU replicated convolution produced an empty group");
            }

            auto folded = sum_local_channels_to_base(
                *group_sum, input, 0, runtime);
            // Keep all kernel products and channel folds at the product scale,
            // then consume one q prime for the completed output group.
            folded = runtime.rescale(folded, 1);
            for (std::size_t replica = 0; replica < input_replicas; ++replica)
            {
                const std::size_t output_channel =
                    group * input_replicas + replica;
                if (output_channel >= static_cast<std::size_t>(out_channels))
                {
                    continue;
                }
                auto selected = runtime.drop_to_q_count(
                    folded, folded.meta.q_count);
                const long long shift =
                    static_cast<long long>(replica * input_span) +
                    output_channel_shift(
                        output, static_cast<int>(output_channel));
                if (shift != 0)
                {
                    selected = runtime.rotate_composed(selected, shift);
                }
                std::vector<double> selector(output.slot_count, 0.0);
                for (int oh = 0; oh < output.h; ++oh)
                {
                    for (int ow = 0; ow < output.w; ++ow)
                    {
                        selector[output.slot_index(
                            static_cast<int>(output_channel), oh, ow)] =
                            1.0;
                    }
                }
                const double selector_scale =
                    runtime.last_modulus_value(selected);
                if (pack_sum)
                {
                    runtime.multiply_plain_accumulate(
                        selected, selector, selector_scale, *pack_sum);
                }
                else
                {
                    pack_sum =
                        std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                            runtime.multiply_plain(
                                selected, selector, selector_scale));
                }
            }
        }
        if (!pack_sum)
        {
            throw std::runtime_error(
                "GPU replicated convolution produced an empty output pack");
        }
        *pack_sum = runtime.rescale(*pack_sum, 1);
        std::vector<double> bias_slots(output.slot_count, 0.0);
        for (int output_channel = 0; output_channel < out_channels;
             ++output_channel)
        {
            for (int oh = 0; oh < output.h; ++oh)
            {
                for (int ow = 0; ow < output.w; ++ow)
                {
                    bias_slots[output.slot_index(output_channel, oh, ow)] =
                        bn_bias[output_channel];
                }
            }
        }
        output.packs.front() = runtime.add_plain(*pack_sum, bias_slots);
        return output;
    }

    // Spatial rotations depend only on the input pack and kernel position,
    // not on the output channel. Cache them once per convolution instead of
    // repeating the same key switch for every output channel.
    const std::size_t kernel_count =
        static_cast<std::size_t>(kernel_h * kernel_w);
    std::vector<std::vector<std::unique_ptr<GpuCkksRuntime::DeviceCiphertext>>>
        rotated_inputs(input.packs.size());
    for (std::size_t input_pack = 0; input_pack < input.packs.size(); ++input_pack)
    {
        rotated_inputs[input_pack].resize(kernel_count);
        for (int kh = 0; kh < kernel_h; ++kh)
        {
            for (int kw = 0; kw < kernel_w; ++kw)
            {
                const long long spatial_step =
                    static_cast<long long>(input.k) * input.k * input.w *
                        (kh - pad_h) +
                    static_cast<long long>(input.k) * (kw - pad_w);
                if (spatial_step != 0)
                {
                    const auto kernel_index =
                        static_cast<std::size_t>(kh * kernel_w + kw);
                    rotated_inputs[input_pack][kernel_index] =
                        std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                            runtime.rotate_composed(input.packs[input_pack],
                                                    spatial_step));
                }
            }
        }
    }

    for (std::size_t output_pack = 0; output_pack < output.packs.size(); ++output_pack)
    {
        std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> pack_sum;
        for (const int output_channel : channels_for_pack(output, output_pack))
        {
            std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> folded_channel_sum;
            for (std::size_t input_pack = 0; input_pack < input.packs.size();
                 ++input_pack)
            {
                std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> input_pack_sum;
                const auto input_channels = channels_for_pack(input, input_pack);
                for (int kh = 0; kh < kernel_h; ++kh)
                {
                    for (int kw = 0; kw < kernel_w; ++kw)
                    {
                        const auto kernel_index =
                            static_cast<std::size_t>(kh * kernel_w + kw);
                        const auto *spatial = &input.packs[input_pack];
                        if (rotated_inputs[input_pack][kernel_index])
                        {
                            spatial = rotated_inputs[input_pack][kernel_index].get();
                        }

                        std::vector<double> compact_weight(input.slot_count, 0.0);
                        bool nonzero = false;
                        for (const int input_channel : input_channels)
                        {
                            const std::size_t weight_index =
                                ((static_cast<std::size_t>(output_channel) * input.c +
                                  input_channel) *
                                     kernel_h +
                                 kh) *
                                    kernel_w +
                                kw;
                            const double coefficient =
                                weights[weight_index] * bn_scale[output_channel];
                            if (coefficient == 0.0)
                            {
                                continue;
                            }
                            for (int oh = 0; oh < output.h; ++oh)
                            {
                                for (int ow = 0; ow < output.w; ++ow)
                                {
                                    const int ih = oh * stride + kh - pad_h;
                                    const int iw = ow * stride + kw - pad_w;
                                    if (ih < 0 || ih >= input.h || iw < 0 ||
                                        iw >= input.w)
                                    {
                                        continue;
                                    }
                                    compact_weight[input.slot_index(
                                        input_channel, oh * stride, ow * stride)] =
                                        coefficient;
                                    nonzero = true;
                                }
                            }
                        }
                        if (!nonzero)
                        {
                            continue;
                        }
                        const double plain_scale =
                            runtime.last_modulus_value(*spatial);
                        if (input_pack_sum)
                        {
                            runtime.multiply_plain_accumulate(
                                *spatial, compact_weight, plain_scale,
                                *input_pack_sum);
                        }
                        else
                        {
                            input_pack_sum =
                                std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                                    runtime.multiply_plain(
                                        *spatial, compact_weight, plain_scale));
                        }
                    }
                }
                if (!input_pack_sum)
                {
                    continue;
                }
                auto folded = sum_local_channels_to_base(*input_pack_sum, input,
                                                         input_pack, runtime);
                const long long shift = output_channel_shift(output, output_channel);
                if (shift != 0)
                {
                    folded = runtime.rotate_composed(folded, shift);
                }
                if (folded_channel_sum)
                {
                    *folded_channel_sum = runtime.add(
                        *folded_channel_sum, folded);
                }
                else
                {
                    folded_channel_sum =
                        std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                            std::move(folded));
                }
            }
            if (!folded_channel_sum)
            {
                throw std::runtime_error(
                    "GPU convolution produced an empty output channel");
            }
            // All kernel and input-pack terms now share one product-scale
            // accumulator. Rescale once before output-channel placement.
            *folded_channel_sum = runtime.rescale(*folded_channel_sum, 1);
            std::vector<double> selector(output.slot_count, 0.0);
            for (int oh = 0; oh < output.h; ++oh)
            {
                for (int ow = 0; ow < output.w; ++ow)
                {
                    selector[output.slot_index(output_channel, oh, ow)] = 1.0;
                }
            }
            const double selector_scale =
                runtime.last_modulus_value(*folded_channel_sum);
            if (pack_sum)
            {
                runtime.multiply_plain_accumulate(
                    *folded_channel_sum, selector, selector_scale, *pack_sum);
            }
            else
            {
                pack_sum = std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                    runtime.multiply_plain(
                        *folded_channel_sum, selector, selector_scale));
            }
        }
        if (!pack_sum)
        {
            throw std::runtime_error("GPU convolution produced an empty output pack");
        }
        *pack_sum = runtime.rescale(*pack_sum, 1);
        std::vector<double> bias_slots(output.slot_count, 0.0);
        for (const int output_channel : channels_for_pack(output, output_pack))
        {
            for (int oh = 0; oh < output.h; ++oh)
            {
                for (int ow = 0; ow < output.w; ++ow)
                {
                    bias_slots[output.slot_index(output_channel, oh, ow)] =
                        bn_bias[output_channel];
                }
            }
        }
        output.packs[output_pack] = runtime.add_plain(*pack_sum, bias_slots);
    }
    return output;
}

GpuMultiplexedTensor residual_add(const GpuMultiplexedTensor &left,
                                  const GpuMultiplexedTensor &right,
                                  const GpuCkksRuntime &runtime)
{
    left.validate();
    right.validate();
    require_same_layout(left, right);
    auto output = make_shape(left.h, left.w, left.c, left.k, left.slot_count);
    for (std::size_t pack = 0; pack < left.packs.size(); ++pack)
    {
        output.packs[pack] = runtime.add_aligned(left.packs[pack], right.packs[pack]);
    }
    return output;
}

GpuMultiplexedTensor average_pool2d_stride2(const GpuMultiplexedTensor &input,
                                            const GpuCkksRuntime &runtime)
{
    input.validate();
    if (input.h % 2 != 0 || input.w % 2 != 0)
    {
        throw std::invalid_argument(
            "stride-2 average pool requires even spatial dimensions");
    }
    auto output =
        make_shape(input.h / 2, input.w / 2, input.c, input.k * 2, input.slot_count);
    if (output.page_size != input.page_size)
    {
        throw std::logic_error(
            "stride-2 average pool must preserve multiplexed page size");
    }

    for (std::size_t output_pack = 0; output_pack < output.packs.size(); ++output_pack)
    {
        std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> sum;
        for (const int channel : channels_for_pack(output, output_pack))
        {
            const std::size_t input_pack = input.pack_index(channel);
            for (int kh = 0; kh < 3; ++kh)
            {
                for (int kw = 0; kw < 3; ++kw)
                {
                    std::vector<double> mask(output.slot_count, 0.0);
                    bool populated = false;
                    long long rotation_step = 0;
                    bool has_rotation = false;
                    for (int oh = 0; oh < output.h; ++oh)
                    {
                        for (int ow = 0; ow < output.w; ++ow)
                        {
                            const int ih = oh * 2 + kh - 1;
                            const int iw = ow * 2 + kw - 1;
                            if (ih < 0 || ih >= input.h || iw < 0 || iw >= input.w)
                            {
                                continue;
                            }
                            const auto target = output.slot_index(channel, oh, ow);
                            const auto source = input.slot_index(channel, ih, iw);
                            mask[target] = 1.0;
                            if (!has_rotation)
                            {
                                rotation_step = static_cast<long long>(source) -
                                                static_cast<long long>(target);
                                has_rotation = true;
                            }
                            populated = true;
                        }
                    }
                    if (!populated)
                    {
                        continue;
                    }
                    const auto *source = &input.packs[input_pack];
                    std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> rotated;
                    if (rotation_step != 0)
                    {
                        rotated = std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                            runtime.rotate_composed(*source, rotation_step));
                        source = rotated.get();
                    }
                    const double plain_scale =
                        runtime.last_modulus_value(*source);
                    if (sum)
                    {
                        runtime.multiply_plain_accumulate(
                            *source, mask, plain_scale, *sum);
                    }
                    else
                    {
                        sum = std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                            runtime.multiply_plain(
                                *source, mask, plain_scale));
                    }
                }
            }
        }
        if (!sum)
        {
            throw std::runtime_error("GPU average pool produced an empty output pack");
        }
        *sum = runtime.rescale(*sum, 1);
        std::vector<double> average(output.slot_count, 1.0 / 9.0);
        output.packs[output_pack] = runtime.multiply_plain_rescale(*sum, average);
    }
    return output;
}

GpuMultiplexedTensor downsample_shortcut(const GpuMultiplexedTensor &input,
                                         const GpuCkksRuntime &runtime)
{
    input.validate();
    if (input.h % 2 != 0 || input.w % 2 != 0 || input.c % 2 != 0)
    {
        throw std::invalid_argument(
            "Option-A shortcut requires even spatial and channel dimensions");
    }

    auto output = make_shape(input.h / 2, input.w / 2, input.c * 2, input.k * 2,
                             input.slot_count);
    if (output.page_size != input.page_size)
    {
        throw std::logic_error("Option-A shortcut must preserve multiplexed page size");
    }

    const int channel_offset = input.c / 2;
    for (std::size_t output_pack = 0; output_pack < output.packs.size(); ++output_pack)
    {
        std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> sum;
        for (int input_channel = 0; input_channel < input.c; ++input_channel)
        {
            const int output_channel = input_channel + channel_offset;
            if (output.pack_index(output_channel) != output_pack)
            {
                continue;
            }

            std::vector<double> mask(output.slot_count, 0.0);
            long long rotation_step = 0;
            bool has_rotation = false;
            for (int row = 0; row < output.h; ++row)
            {
                for (int col = 0; col < output.w; ++col)
                {
                    const auto source =
                        input.slot_index(input_channel, row * 2, col * 2);
                    const auto target = output.slot_index(output_channel, row, col);
                    const auto current_step =
                        static_cast<long long>(source) - static_cast<long long>(target);
                    if (!has_rotation)
                    {
                        rotation_step = current_step;
                        has_rotation = true;
                    }
                    else if (rotation_step != current_step)
                    {
                        throw std::logic_error(
                            "Option-A shortcut mapping is not a single rotation");
                    }
                    mask[target] = 1.0;
                }
            }

            const auto &source_pack = input.packs[input.pack_index(input_channel)];
            const auto *source = &source_pack;
            std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> rotated;
            if (rotation_step != 0)
            {
                rotated = std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                    runtime.rotate_composed(source_pack, rotation_step));
                source = rotated.get();
            }
            const double plain_scale =
                runtime.last_modulus_value(*source);
            if (sum)
            {
                runtime.multiply_plain_accumulate(
                    *source, mask, plain_scale, *sum);
            }
            else
            {
                sum =
                    std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                        runtime.multiply_plain(*source, mask, plain_scale));
            }
        }
        if (!sum)
        {
            // Option-A pads both sides of the channel dimension with zeros.
            // With multiple output packs, the leading and trailing packs can
            // therefore be entirely zero while still being part of the valid
            // residual layout. Materialize those packs at the same level and
            // scale as the populated packs.
            std::vector<double> zero_mask(output.slot_count, 0.0);
            const double plain_scale =
                runtime.last_modulus_value(input.packs.front());
            output.packs[output_pack] = runtime.rescale(
                runtime.multiply_plain(
                    input.packs.front(), zero_mask, plain_scale),
                1);
            continue;
        }
        output.packs[output_pack] = runtime.rescale(*sum, 1);
    }
    return output;
}

GpuCkksRuntime::DeviceCiphertext global_average_pool(const GpuMultiplexedTensor &input,
                                                     double output_scale_factor,
                                                     const GpuCkksRuntime &runtime)
{
    input.validate();
    if (!std::isfinite(output_scale_factor))
    {
        throw std::invalid_argument("global average pool scale factor must be finite");
    }
    std::vector<double> column_mask(input.slot_count, 0.0);
    std::vector<double> row_mask(input.slot_count, 0.0);
    for (int channel = 0; channel < input.c; ++channel)
    {
        for (int row = 0; row < input.h; ++row)
        {
            column_mask[input.slot_index(channel, row, 0)] = 1.0;
        }
        row_mask[input.slot_index(channel, 0, 0)] = 1.0;
    }

    std::vector<GpuCkksRuntime::DeviceCiphertext> spatial_sums;
    spatial_sums.reserve(input.packs.size());
    for (const auto &pack : input.packs)
    {
        std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> column_sum;
        for (int col = 0; col < input.w; ++col)
        {
            auto rotated =
                runtime.rotate_composed(pack, static_cast<long long>(col) * input.k);
            const double plain_scale =
                runtime.last_modulus_value(rotated);
            if (column_sum)
            {
                runtime.multiply_plain_accumulate(
                    rotated, column_mask, plain_scale, *column_sum);
            }
            else
            {
                column_sum =
                    std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                        runtime.multiply_plain(
                            rotated, column_mask, plain_scale));
            }
        }
        *column_sum = runtime.rescale(*column_sum, 1);

        std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> spatial_sum;
        const int packed_width = input.w * input.k;
        for (int row = 0; row < input.h; ++row)
        {
            auto rotated = runtime.rotate_composed(
                *column_sum, static_cast<long long>(row) * input.k * packed_width);
            const double plain_scale =
                runtime.last_modulus_value(rotated);
            if (spatial_sum)
            {
                runtime.multiply_plain_accumulate(
                    rotated, row_mask, plain_scale, *spatial_sum);
            }
            else
            {
                spatial_sum =
                    std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                        runtime.multiply_plain(
                            rotated, row_mask, plain_scale));
            }
        }
        spatial_sums.push_back(runtime.rescale(*spatial_sum, 1));
    }

    std::unique_ptr<GpuCkksRuntime::DeviceCiphertext> compacted;
    for (int channel = 0; channel < input.c; ++channel)
    {
        const auto pack = input.pack_index(channel);
        const auto source_slot = input.slot_index(channel, 0, 0);
        const auto target_slot = static_cast<std::size_t>(channel);
        auto rotated = runtime.rotate_composed(
            spatial_sums[pack],
            static_cast<long long>(source_slot) - static_cast<long long>(target_slot));
        std::vector<double> target_mask(input.slot_count, 0.0);
        target_mask[target_slot] = 1.0;
        const double plain_scale =
            runtime.last_modulus_value(rotated);
        if (compacted)
        {
            runtime.multiply_plain_accumulate(
                rotated, target_mask, plain_scale, *compacted);
        }
        else
        {
            compacted =
                std::make_unique<GpuCkksRuntime::DeviceCiphertext>(
                    runtime.multiply_plain(
                        rotated, target_mask, plain_scale));
        }
    }

    *compacted = runtime.rescale(*compacted, 1);

    std::vector<double> factor(
        input.slot_count, output_scale_factor / static_cast<double>(input.h * input.w));
    return runtime.multiply_plain_rescale(*compacted, factor);
}

std::vector<GpuCkksRuntime::DeviceCiphertext> fully_connected(
    const GpuCkksRuntime::DeviceCiphertext &features, int feature_count,
    const std::vector<double> &matrix, const std::vector<double> &bias, int class_count,
    const GpuCkksRuntime &runtime)
{
    if (feature_count <= 0 || class_count <= 0 ||
        feature_count > static_cast<int>(runtime.slot_count()) ||
        matrix.size() != static_cast<std::size_t>(feature_count * class_count) ||
        bias.size() != static_cast<std::size_t>(class_count))
    {
        throw std::invalid_argument("GPU fully-connected parameter mismatch");
    }
    std::vector<GpuCkksRuntime::DeviceCiphertext> result;
    result.reserve(class_count);
    for (int output = 0; output < class_count; ++output)
    {
        std::vector<double> weight_slots(runtime.slot_count(), 0.0);
        for (int feature = 0; feature < feature_count; ++feature)
        {
            weight_slots[feature] =
                matrix[static_cast<std::size_t>(output) * feature_count + feature];
        }
        auto sum = runtime.multiply_plain_rescale(features, weight_slots);
        for (int step = 1; step < feature_count; step <<= 1)
        {
            auto rotated = runtime.rotate_composed(sum, step);
            sum = runtime.add(sum, rotated);
        }
        std::vector<double> bias_slots(runtime.slot_count(), 0.0);
        bias_slots[0] = bias[output];
        result.push_back(runtime.add_plain(sum, bias_slots));
    }
    return result;
}

}  // namespace poseidon::benchmark::resnet20_gpu::core
