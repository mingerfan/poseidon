#include "gpu_resnet18_inference.h"

#include "gpu_ckks_runtime.h"
#include "gpu_multiplexed_tensor.h"
#include "gpu_relu.h"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace poseidon::benchmark::resnet18_gpu
{
namespace
{

namespace shared_gpu = poseidon::benchmark::resnet50_gpu;

constexpr double kBoundary = 20.0;
constexpr double kBatchNormEpsilon = 1.0e-5;
constexpr int kImageHeight = 224;
constexpr int kImageWidth = 224;
constexpr int kImageChannels = 3;
constexpr int kClassCount = 1000;
constexpr int kFinalChannels = 512;

using GpuCkksRuntime = shared_gpu::GpuCkksRuntime;
using GpuMultiplexedTensor = shared_gpu::GpuMultiplexedTensor;

struct BatchNormAffine
{
    std::vector<double> scale;
    std::vector<double> bias;
};

struct PlainTensor
{
    int h = 0;
    int w = 0;
    int c = 0;
    std::vector<double> values;
};

BatchNormAffine folded_batch_norm(
    const std::vector<double> &bias,
    const std::vector<double> &mean,
    const std::vector<double> &variance,
    const std::vector<double> &weight)
{
    if (bias.size() != mean.size() || bias.size() != variance.size() ||
        bias.size() != weight.size())
    {
        throw std::invalid_argument("ResNet18 batch-normalization shape mismatch");
    }
    BatchNormAffine result;
    result.scale.resize(bias.size());
    result.bias.resize(bias.size());
    for (std::size_t channel = 0; channel < bias.size(); ++channel)
    {
        result.scale[channel] =
            weight[channel] / std::sqrt(variance[channel] + kBatchNormEpsilon);
        result.bias[channel] =
            (bias[channel] - mean[channel] * result.scale[channel]) / kBoundary;
    }
    return result;
}

void log_tensor_state(const std::string &label, const GpuMultiplexedTensor &tensor)
{
    if (tensor.packs.empty())
    {
        std::cout << "[GPU ResNet18] " << label << " empty\n";
        return;
    }
    std::size_t minimum_q = tensor.packs.front().meta.q_count;
    std::size_t maximum_q = minimum_q;
    double minimum_log_scale = std::log2(tensor.packs.front().meta.scale);
    double maximum_log_scale = minimum_log_scale;
    for (const auto &pack : tensor.packs)
    {
        minimum_q = std::min(minimum_q, pack.meta.q_count);
        maximum_q = std::max(maximum_q, pack.meta.q_count);
        const double log_scale = std::log2(pack.meta.scale);
        minimum_log_scale = std::min(minimum_log_scale, log_scale);
        maximum_log_scale = std::max(maximum_log_scale, log_scale);
    }
    std::cout << "[GPU ResNet18] " << label
              << " shape=" << tensor.h << 'x' << tensor.w << 'x' << tensor.c
              << " k=" << tensor.k << " packs=" << tensor.packs.size()
              << " q=" << minimum_q << ".." << maximum_q
              << " log2_scale=" << minimum_log_scale << ".."
              << maximum_log_scale << '\n';
}

GpuMultiplexedTensor empty_like(const GpuMultiplexedTensor &input)
{
    GpuMultiplexedTensor output;
    output.h = input.h;
    output.w = input.w;
    output.c = input.c;
    output.k = input.k;
    output.pages_per_cipher = input.pages_per_cipher;
    output.page_size = input.page_size;
    output.slot_count = input.slot_count;
    output.packs.resize(input.packs.size());
    return output;
}

GpuMultiplexedTensor clone_tensor(
    const GpuMultiplexedTensor &input,
    const GpuCkksRuntime &runtime)
{
    auto output = empty_like(input);
    for (std::size_t pack = 0; pack < input.packs.size(); ++pack)
    {
        output.packs[pack] = runtime.drop_to_q_count(
            input.packs[pack], input.packs[pack].meta.q_count);
    }
    return output;
}

template <typename Function>
auto timed(const std::string &label, Function &&function)
{
    const auto begin = std::chrono::steady_clock::now();
    auto result = std::forward<Function>(function)();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - begin).count();
    std::cout << "[GPU ResNet18] " << label << " elapsed_ms=" << elapsed << '\n';
    return result;
}

class StagewiseGpuTimer
{
public:
    StagewiseGpuTimer(GpuCkksRuntime &runtime, bool enabled)
        : runtime_(runtime), enabled_(enabled)
    {}

    template <typename Function>
    auto run(const std::string &label, Function &&function)
    {
        if (!enabled_)
        {
            return timed(label, std::forward<Function>(function));
        }
        const auto preparation_begin = std::chrono::steady_clock::now();
        {
            auto prepared = function();
            runtime_.synchronize();
        }
        const auto preparation_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - preparation_begin).count();
        runtime_.synchronize();
        const auto measured_begin = std::chrono::steady_clock::now();
        auto result = function();
        runtime_.synchronize();
        const double measured_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - measured_begin).count();
        elapsed_seconds_ += measured_seconds;
        std::cout << "[GPU ResNet18] " << label
                  << " preparation_ms=" << preparation_ms
                  << " staged_gpu_seconds=" << measured_seconds << '\n';
        runtime_.clear_plaintext_cache();
        return result;
    }

    double elapsed_seconds() const noexcept
    {
        return elapsed_seconds_;
    }

private:
    GpuCkksRuntime &runtime_;
    bool enabled_ = false;
    double elapsed_seconds_ = 0.0;
};

GpuMultiplexedTensor apply_conv_bn(
    const GpuMultiplexedTensor &input,
    int out_channels,
    int stride,
    int kernel,
    const std::vector<double> &conv_weight,
    const std::vector<double> &bn_bias,
    const std::vector<double> &bn_mean,
    const std::vector<double> &bn_variance,
    const std::vector<double> &bn_weight,
    const GpuCkksRuntime &runtime)
{
    const auto affine = folded_batch_norm(
        bn_bias, bn_mean, bn_variance, bn_weight);
    return shared_gpu::conv2d_bn(
        input, out_channels, stride, kernel, kernel, conv_weight,
        affine.scale, affine.bias, runtime);
}

GpuMultiplexedTensor bootstrap_tensor(
    const GpuMultiplexedTensor &input,
    const GpuCkksRuntime &runtime)
{
    auto output = empty_like(input);
    for (std::size_t pack = 0; pack < input.packs.size(); ++pack)
    {
        output.packs[pack] = runtime.bootstrap(input.packs[pack]);
        std::cout << "[GPU ResNet18] bootstrap pack " << pack + 1 << '/'
                  << input.packs.size() << '\n';
    }
    return output;
}

GpuMultiplexedTensor relu_tensor(
    const GpuMultiplexedTensor &input,
    const GpuCkksRuntime &runtime)
{
    auto output = empty_like(input);
    for (std::size_t pack = 0; pack < input.packs.size(); ++pack)
    {
        output.packs[pack] = shared_gpu::polynomial_relu(input.packs[pack], runtime);
        std::cout << "[GPU ResNet18] ReLU pack " << pack + 1 << '/'
                  << input.packs.size() << '\n';
    }
    return output;
}

void verify_values(
    const std::string &label,
    const std::vector<double> &expected,
    const std::vector<double> &actual,
    double tolerance)
{
    if (expected.size() != actual.size())
    {
        throw std::runtime_error(label + " verification shape mismatch");
    }
    double maximum_error = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        maximum_error = std::max(
            maximum_error, std::abs(expected[index] - actual[index]));
    }
    std::cout << "[GPU ResNet18] " << label
              << " max_abs_error=" << maximum_error << '\n';
    if (!std::isfinite(maximum_error) || maximum_error > tolerance)
    {
        throw std::runtime_error(label + " verification failed");
    }
}

GpuMultiplexedTensor bootstrap_relu(
    const std::string &label,
    const GpuMultiplexedTensor &input,
    const GpuCkksRuntime &runtime,
    StagewiseGpuTimer &stage_timer,
    const std::vector<double> *expected_bootstrap = nullptr)
{
    auto refreshed = stage_timer.run(label + ".bootstrap", [&]() {
        return bootstrap_tensor(input, runtime);
    });
    log_tensor_state(label + ".bootstrap", refreshed);
    if (expected_bootstrap)
    {
        verify_values(
            label + ".bootstrap", *expected_bootstrap,
            shared_gpu::decrypt_multiplexed_chw(refreshed, runtime), 1.0e-1);
    }
    auto activated = stage_timer.run(label + ".relu", [&]() {
        return relu_tensor(refreshed, runtime);
    });
    log_tensor_state(label + ".relu", activated);
    return activated;
}

int argmax(const std::vector<double> &values)
{
    if (values.empty())
    {
        return -1;
    }
    return static_cast<int>(std::distance(
        values.begin(), std::max_element(values.begin(), values.end())));
}

std::vector<double> plain_stem_reference(
    const std::vector<double> &image,
    const std::vector<double> &weights,
    const BatchNormAffine &affine)
{
    constexpr int output_h = kImageHeight / 2;
    constexpr int output_w = kImageWidth / 2;
    std::vector<double> output(static_cast<std::size_t>(64 * output_h * output_w));
    for (int oc = 0; oc < 64; ++oc)
    {
        for (int oh = 0; oh < output_h; ++oh)
        {
            for (int ow = 0; ow < output_w; ++ow)
            {
                double sum = 0.0;
                for (int ic = 0; ic < kImageChannels; ++ic)
                {
                    for (int kh = 0; kh < 7; ++kh)
                    {
                        for (int kw = 0; kw < 7; ++kw)
                        {
                            const int ih = oh * 2 + kh - 3;
                            const int iw = ow * 2 + kw - 3;
                            if (ih < 0 || ih >= kImageHeight ||
                                iw < 0 || iw >= kImageWidth)
                            {
                                continue;
                            }
                            const auto image_index =
                                (static_cast<std::size_t>(ic) * kImageHeight + ih) *
                                    kImageWidth + iw;
                            const auto weight_index =
                                ((static_cast<std::size_t>(oc) * kImageChannels + ic) * 7 +
                                 kh) * 7 + kw;
                            sum += image[image_index] * weights[weight_index];
                        }
                    }
                }
                output[(static_cast<std::size_t>(oc) * output_h + oh) * output_w + ow] =
                    sum * affine.scale[oc] + affine.bias[oc];
            }
        }
    }
    return output;
}

std::vector<double> plain_stem_relu_pool(std::vector<double> stem)
{
    constexpr int input_h = kImageHeight / 2;
    constexpr int input_w = kImageWidth / 2;
    constexpr int output_h = input_h / 2;
    constexpr int output_w = input_w / 2;
    for (double &value : stem)
    {
        value = shared_gpu::polynomial_relu_reference(value);
    }
    std::vector<double> output(static_cast<std::size_t>(64 * output_h * output_w));
    for (int channel = 0; channel < 64; ++channel)
    {
        for (int oh = 0; oh < output_h; ++oh)
        {
            for (int ow = 0; ow < output_w; ++ow)
            {
                double sum = 0.0;
                for (int kh = 0; kh < 3; ++kh)
                {
                    for (int kw = 0; kw < 3; ++kw)
                    {
                        const int ih = oh * 2 + kh - 1;
                        const int iw = ow * 2 + kw - 1;
                        if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w)
                        {
                            sum += stem[(static_cast<std::size_t>(channel) * input_h + ih) *
                                        input_w + iw];
                        }
                    }
                }
                output[(static_cast<std::size_t>(channel) * output_h + oh) * output_w + ow] =
                    sum / 9.0;
            }
        }
    }
    return output;
}

PlainTensor plain_conv_bn(
    const PlainTensor &input,
    int out_channels,
    int stride,
    int kernel,
    const std::vector<double> &weights,
    const BatchNormAffine &affine)
{
    const int output_h = input.h / stride;
    const int output_w = input.w / stride;
    PlainTensor output{
        output_h,
        output_w,
        out_channels,
        std::vector<double>(static_cast<std::size_t>(output_h * output_w * out_channels))};
    const int padding = kernel / 2;
    for (int oc = 0; oc < out_channels; ++oc)
    {
        for (int oh = 0; oh < output_h; ++oh)
        {
            for (int ow = 0; ow < output_w; ++ow)
            {
                double sum = 0.0;
                for (int ic = 0; ic < input.c; ++ic)
                {
                    for (int kh = 0; kh < kernel; ++kh)
                    {
                        for (int kw = 0; kw < kernel; ++kw)
                        {
                            const int ih = oh * stride + kh - padding;
                            const int iw = ow * stride + kw - padding;
                            if (ih < 0 || ih >= input.h || iw < 0 || iw >= input.w)
                            {
                                continue;
                            }
                            const auto input_index =
                                (static_cast<std::size_t>(ic) * input.h + ih) * input.w + iw;
                            const auto weight_index =
                                ((static_cast<std::size_t>(oc) * input.c + ic) * kernel + kh) *
                                    kernel + kw;
                            sum += input.values[input_index] * weights[weight_index];
                        }
                    }
                }
                output.values[(static_cast<std::size_t>(oc) * output_h + oh) *
                              output_w + ow] =
                    sum * affine.scale[oc] + affine.bias[oc];
            }
        }
    }
    return output;
}

void plain_relu_inplace(PlainTensor &tensor)
{
    for (double &value : tensor.values)
    {
        value = shared_gpu::polynomial_relu_reference(value);
    }
}

PlainTensor plain_add(PlainTensor left, const PlainTensor &right)
{
    if (left.h != right.h || left.w != right.w || left.c != right.c ||
        left.values.size() != right.values.size())
    {
        throw std::invalid_argument("plain ResNet18 residual shape mismatch");
    }
    for (std::size_t index = 0; index < left.values.size(); ++index)
    {
        left.values[index] += right.values[index];
    }
    return left;
}

PlainTensor plain_resnet18_features(
    const std::vector<double> &image,
    const ResNet18Topology &topology,
    const ResNet18Weights &weights)
{
    std::size_t conv_index = 0;
    std::size_t bn_index = 0;
    auto affine = folded_batch_norm(
        weights.bn_bias[bn_index], weights.bn_running_mean[bn_index],
        weights.bn_running_var[bn_index], weights.bn_weight[bn_index]);
    auto stem = plain_stem_reference(image, weights.conv_weight[conv_index], affine);
    ++conv_index;
    ++bn_index;
    PlainTensor tensor{56, 56, 64, plain_stem_relu_pool(std::move(stem))};

    for (const auto &spec : topology.blocks)
    {
        auto shortcut = tensor;
        affine = folded_batch_norm(
            weights.bn_bias[bn_index], weights.bn_running_mean[bn_index],
            weights.bn_running_var[bn_index], weights.bn_weight[bn_index]);
        auto branch = plain_conv_bn(
            tensor, spec.output_channels, spec.stride, 3,
            weights.conv_weight[conv_index], affine);
        ++conv_index;
        ++bn_index;
        plain_relu_inplace(branch);

        affine = folded_batch_norm(
            weights.bn_bias[bn_index], weights.bn_running_mean[bn_index],
            weights.bn_running_var[bn_index], weights.bn_weight[bn_index]);
        branch = plain_conv_bn(
            branch, spec.output_channels, 1, 3,
            weights.conv_weight[conv_index], affine);
        ++conv_index;
        ++bn_index;

        if (spec.projection)
        {
            const auto downsample = static_cast<std::size_t>(spec.stage - 2);
            affine = folded_batch_norm(
                weights.downsample_bn_bias[downsample],
                weights.downsample_bn_running_mean[downsample],
                weights.downsample_bn_running_var[downsample],
                weights.downsample_bn_weight[downsample]);
            shortcut = plain_conv_bn(
                shortcut, spec.output_channels, spec.stride, 1,
                weights.downsample_weight[downsample], affine);
        }
        tensor = plain_add(std::move(branch), shortcut);
        plain_relu_inplace(tensor);
    }
    return tensor;
}

std::vector<double> plain_head_logits(
    const PlainTensor &features,
    const ResNet18Weights &weights)
{
    if (features.h != 7 || features.w != 7 || features.c != kFinalChannels)
    {
        throw std::invalid_argument("ResNet18 head expects 7x7x512 features");
    }
    std::vector<double> averages(kFinalChannels, 0.0);
    for (int channel = 0; channel < kFinalChannels; ++channel)
    {
        for (int row = 0; row < features.h; ++row)
        {
            for (int col = 0; col < features.w; ++col)
            {
                averages[channel] += features.values[
                    (static_cast<std::size_t>(channel) * features.h + row) *
                        features.w + col] *
                    kBoundary / static_cast<double>(features.h * features.w);
            }
        }
    }
    std::vector<double> logits(kClassCount, 0.0);
    for (int output = 0; output < kClassCount; ++output)
    {
        double value = weights.linear_bias[output];
        for (int feature = 0; feature < kFinalChannels; ++feature)
        {
            value += weights.linear_weight[
                         static_cast<std::size_t>(output) * kFinalChannels + feature] *
                     averages[feature];
        }
        logits[output] = value;
    }
    return logits;
}

}  // namespace

GpuResNet18Result run_gpu_resnet18_impl(
    std::size_t image_id,
    const ResNet18Topology &topology,
    const ResNet18Weights &weights,
    std::size_t max_blocks,
    GpuCkksRuntime &runtime,
    bool enable_validation,
    bool measure_gpu_only,
    bool measure_staged_gpu_only)
{
    topology.validate();
    if (max_blocks > topology.blocks.size())
    {
        throw std::invalid_argument("ResNet18 max_blocks exceeds the topology");
    }
    GpuResNet18Result result;
    result.image_id = image_id;
    result.true_label = load_imagenet_label(image_id);
    const auto image = load_imagenet_image_chw(image_id, kBoundary);
    const bool validate_intermediates = enable_validation &&
        (max_blocks <= 1 ||
         std::getenv("POSEIDON_GPU_RESNET18_VALIDATE_BLOCKS") != nullptr);
    StagewiseGpuTimer stage_timer(runtime, measure_staged_gpu_only);
    std::chrono::steady_clock::time_point gpu_only_start;
    bool profiler_started = false;
    if (measure_gpu_only)
    {
        runtime.synchronize();
        if (cudaProfilerStart() != cudaSuccess)
        {
            throw std::runtime_error("cudaProfilerStart failed");
        }
        profiler_started = true;
        gpu_only_start = std::chrono::steady_clock::now();
    }
    const auto finish_gpu_only = [&]() {
        if (!measure_gpu_only || !profiler_started)
        {
            return;
        }
        runtime.synchronize();
        result.gpu_only_elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - gpu_only_start).count();
        if (cudaProfilerStop() != cudaSuccess)
        {
            throw std::runtime_error("cudaProfilerStop failed");
        }
        profiler_started = false;
        std::cout << "[GPU ResNet18] gpu_only_preloaded_elapsed_seconds="
                  << result.gpu_only_elapsed_seconds << '\n';
    };

    std::size_t conv_index = 0;
    std::size_t bn_index = 0;
    const auto stem_affine = folded_batch_norm(
        weights.bn_bias[bn_index], weights.bn_running_mean[bn_index],
        weights.bn_running_var[bn_index], weights.bn_weight[bn_index]);
    auto tensor = stage_timer.run("stem.conv1_bn1", [&]() {
        return shared_gpu::encrypted_stem_conv2d_bn(
            image, kImageHeight, kImageWidth, kImageChannels, 64,
            /*stride=*/2, /*kernel_h=*/7, /*kernel_w=*/7,
            weights.conv_weight[conv_index], stem_affine.scale,
            stem_affine.bias, runtime);
    });
    ++conv_index;
    ++bn_index;
    log_tensor_state("stem.conv1_bn1", tensor);

    std::vector<double> plain_stem;
    std::optional<PlainTensor> plain_tensor;
    if (validate_intermediates)
    {
        plain_stem = plain_stem_reference(
            image, weights.conv_weight[0], stem_affine);
        verify_values(
            "stem.conv1_bn1", plain_stem,
            shared_gpu::decrypt_multiplexed_chw(tensor, runtime), 2.0e-4);
    }
    tensor = stage_timer.run(
        "stem.relu", [&]() { return relu_tensor(tensor, runtime); });
    tensor = stage_timer.run("stem.avgpool", [&]() {
        return shared_gpu::average_pool2d_stride2(tensor, runtime);
    });
    if (measure_staged_gpu_only)
    {
        runtime.clear_ciphertext_cache();
    }
    log_tensor_state("stem.avgpool", tensor);
    if (validate_intermediates)
    {
        plain_tensor = PlainTensor{
            56, 56, 64, plain_stem_relu_pool(std::move(plain_stem))};
        verify_values(
            "stem.relu_avgpool", plain_tensor->values,
            shared_gpu::decrypt_multiplexed_chw(tensor, runtime), 2.0e-3);
    }

    for (std::size_t block_index = 0; block_index < max_blocks; ++block_index)
    {
        const auto &spec = topology.blocks[block_index];
        const std::string prefix = basic_block_name(spec);
        auto shortcut = clone_tensor(tensor, runtime);
        std::optional<PlainTensor> plain_shortcut;
        std::optional<PlainTensor> plain_branch;
        if (plain_tensor)
        {
            plain_shortcut = *plain_tensor;
            const auto affine = folded_batch_norm(
                weights.bn_bias[bn_index], weights.bn_running_mean[bn_index],
                weights.bn_running_var[bn_index], weights.bn_weight[bn_index]);
            plain_branch = plain_conv_bn(
                *plain_tensor, spec.output_channels, spec.stride, 3,
                weights.conv_weight[conv_index], affine);
        }

        auto branch = stage_timer.run(prefix + ".conv1_bn1", [&]() {
            return apply_conv_bn(
                tensor, spec.output_channels, spec.stride, 3,
                weights.conv_weight[conv_index], weights.bn_bias[bn_index],
                weights.bn_running_mean[bn_index], weights.bn_running_var[bn_index],
                weights.bn_weight[bn_index], runtime);
        });
        if (plain_branch)
        {
            verify_values(
                prefix + ".conv1_bn1", plain_branch->values,
                shared_gpu::decrypt_multiplexed_chw(branch, runtime), 2.0e-2);
        }
        ++conv_index;
        ++bn_index;
        branch = bootstrap_relu(
            prefix + ".act1", branch, runtime, stage_timer,
            plain_branch ? &plain_branch->values : nullptr);

        if (plain_branch)
        {
            plain_relu_inplace(*plain_branch);
            verify_values(
                prefix + ".act1", plain_branch->values,
                shared_gpu::decrypt_multiplexed_chw(branch, runtime), 2.0e-1);
            const auto affine = folded_batch_norm(
                weights.bn_bias[bn_index], weights.bn_running_mean[bn_index],
                weights.bn_running_var[bn_index], weights.bn_weight[bn_index]);
            *plain_branch = plain_conv_bn(
                *plain_branch, spec.output_channels, 1, 3,
                weights.conv_weight[conv_index], affine);
        }

        branch = stage_timer.run(prefix + ".conv2_bn2", [&]() {
            return apply_conv_bn(
                branch, spec.output_channels, 1, 3,
                weights.conv_weight[conv_index], weights.bn_bias[bn_index],
                weights.bn_running_mean[bn_index], weights.bn_running_var[bn_index],
                weights.bn_weight[bn_index], runtime);
        });
        if (plain_branch)
        {
            verify_values(
                prefix + ".conv2_bn2", plain_branch->values,
                shared_gpu::decrypt_multiplexed_chw(branch, runtime), 5.0e-1);
        }
        ++conv_index;
        ++bn_index;

        if (spec.projection)
        {
            const auto downsample = static_cast<std::size_t>(spec.stage - 2);
            if (plain_shortcut)
            {
                const auto affine = folded_batch_norm(
                    weights.downsample_bn_bias[downsample],
                    weights.downsample_bn_running_mean[downsample],
                    weights.downsample_bn_running_var[downsample],
                    weights.downsample_bn_weight[downsample]);
                *plain_shortcut = plain_conv_bn(
                    *plain_shortcut, spec.output_channels, spec.stride, 1,
                    weights.downsample_weight[downsample], affine);
            }
            shortcut = stage_timer.run(prefix + ".shortcut", [&]() {
                return apply_conv_bn(
                    shortcut, spec.output_channels, spec.stride, 1,
                    weights.downsample_weight[downsample],
                    weights.downsample_bn_bias[downsample],
                    weights.downsample_bn_running_mean[downsample],
                    weights.downsample_bn_running_var[downsample],
                    weights.downsample_bn_weight[downsample], runtime);
            });
            if (plain_shortcut)
            {
                verify_values(
                    prefix + ".shortcut", plain_shortcut->values,
                    shared_gpu::decrypt_multiplexed_chw(shortcut, runtime), 2.0e-2);
            }
        }

        tensor = stage_timer.run(prefix + ".add", [&]() {
            return shared_gpu::residual_add(branch, shortcut, runtime);
        });
        if (plain_branch && plain_shortcut)
        {
            plain_tensor = plain_add(std::move(*plain_branch), *plain_shortcut);
            verify_values(
                prefix + ".add", plain_tensor->values,
                shared_gpu::decrypt_multiplexed_chw(tensor, runtime), 5.0e-1);
        }
        tensor = bootstrap_relu(
            prefix + ".act2", tensor, runtime, stage_timer,
            plain_tensor ? &plain_tensor->values : nullptr);
        if (plain_tensor)
        {
            plain_relu_inplace(*plain_tensor);
            verify_values(
                prefix + ".act2", plain_tensor->values,
                shared_gpu::decrypt_multiplexed_chw(tensor, runtime), 1.0);
        }
        result.completed_blocks = block_index + 1;
        log_tensor_state(prefix + ".output", tensor);
    }

    if (max_blocks != topology.blocks.size())
    {
        finish_gpu_only();
        if (measure_staged_gpu_only)
        {
            result.gpu_only_elapsed_seconds = stage_timer.elapsed_seconds();
            std::cout << "[GPU ResNet18] staged_gpu_only_elapsed_seconds="
                      << result.gpu_only_elapsed_seconds << '\n';
        }
        std::cout << "[GPU ResNet18] staged run stopped after " << max_blocks
                  << " basic block(s)\n";
        return result;
    }

    auto features = stage_timer.run("head.global_avgpool", [&]() {
        return shared_gpu::global_average_pool(tensor, kBoundary, runtime);
    });
    auto encrypted_logits = stage_timer.run("head.fc", [&]() {
        return shared_gpu::fully_connected(
            features, kFinalChannels, weights.linear_weight, weights.linear_bias,
            kClassCount, runtime);
    });
    finish_gpu_only();
    if (measure_staged_gpu_only)
    {
        result.gpu_only_elapsed_seconds = stage_timer.elapsed_seconds();
        std::cout << "[GPU ResNet18] staged_gpu_only_elapsed_seconds="
                  << result.gpu_only_elapsed_seconds << '\n';
    }
    result.logits.resize(encrypted_logits.size());
    for (std::size_t index = 0; index < encrypted_logits.size(); ++index)
    {
        result.logits[index] = runtime.decrypt(encrypted_logits[index])[0].real();
    }
    result.predicted_label = argmax(result.logits);
    if (plain_tensor)
    {
        const auto reference_logits = plain_head_logits(*plain_tensor, weights);
        for (std::size_t output = 0; output < reference_logits.size(); ++output)
        {
            result.max_logit_error = std::max(
                result.max_logit_error,
                std::abs(result.logits[output] - reference_logits[output]));
        }
        const int plain_prediction = argmax(reference_logits);
        std::cout << "[GPU ResNet18] full_validation plain_pred="
                  << plain_prediction << " gpu_pred=" << result.predicted_label
                  << " max_logit_error=" << result.max_logit_error << '\n';
        if (plain_prediction != result.predicted_label ||
            !std::isfinite(result.max_logit_error) || result.max_logit_error > 0.1)
        {
            throw std::runtime_error("GPU ResNet18 full validation failed");
        }
    }
    std::cout << "[GPU ResNet18] image=" << image_id
              << " true_label=" << result.true_label
              << " predicted_label=" << result.predicted_label << '\n';
    return result;
}

GpuResNet18Result run_gpu_resnet18(
    std::size_t image_id,
    const ResNet18GpuConfig &config,
    const ResNet18Topology &topology,
    const ResNet18Weights &weights,
    std::size_t max_blocks)
{
    GpuCkksRuntime runtime(config);
    if (max_blocks == 0)
    {
        std::cout << "[GPU ResNet18] generating inference evaluation keys\n";
        runtime.initialize_inference_evaluation_keys();
    }
    else
    {
        std::cout << "[GPU ResNet18] initializing shared inference/bootstrap keys\n";
        runtime.initialize_bootstrap();
    }
    return run_gpu_resnet18_impl(
        image_id, topology, weights, max_blocks, runtime,
        /*enable_validation=*/true, /*measure_gpu_only=*/false,
        /*measure_staged_gpu_only=*/false);
}

GpuResNet18Result run_gpu_resnet18_preloaded(
    std::size_t image_id,
    const ResNet18GpuConfig &config,
    const ResNet18Topology &topology,
    const ResNet18Weights &weights,
    std::size_t max_blocks)
{
    if (max_blocks > topology.blocks.size())
    {
        throw std::invalid_argument("ResNet18 preloaded max_blocks exceeds topology");
    }
    GpuCkksRuntime runtime(config);
    if (max_blocks == 0)
    {
        runtime.initialize_inference_evaluation_keys();
    }
    else
    {
        runtime.initialize_bootstrap();
    }
    runtime.enable_full_device_cache();
    std::cout << "[GPU ResNet18] preparing device-resident input/model cache\n";
    const auto prepared_result = run_gpu_resnet18_impl(
        image_id, topology, weights, max_blocks, runtime,
        /*enable_validation=*/false, /*measure_gpu_only=*/false,
        /*measure_staged_gpu_only=*/false);
    runtime.synchronize();
    std::cout << "[GPU ResNet18] preparation complete; starting GPU-only pass\n";
    auto measured_result = run_gpu_resnet18_impl(
        image_id, topology, weights, max_blocks, runtime,
        /*enable_validation=*/false, /*measure_gpu_only=*/true,
        /*measure_staged_gpu_only=*/false);
    if (prepared_result.logits.size() != measured_result.logits.size())
    {
        throw std::runtime_error("preloaded replay logit shape mismatch");
    }
    double replay_logit_error = 0.0;
    for (std::size_t index = 0; index < prepared_result.logits.size(); ++index)
    {
        replay_logit_error = std::max(
            replay_logit_error,
            std::abs(prepared_result.logits[index] - measured_result.logits[index]));
    }
    std::cout << "[GPU ResNet18] preloaded_replay_max_logit_error="
              << replay_logit_error << '\n';
    if (prepared_result.predicted_label != measured_result.predicted_label ||
        !std::isfinite(replay_logit_error) || replay_logit_error > 1.0e-8)
    {
        throw std::runtime_error("preloaded GPU replay verification failed");
    }
    return measured_result;
}

GpuResNet18Result run_gpu_resnet18_staged_gpu_only(
    std::size_t image_id,
    const ResNet18GpuConfig &config,
    const ResNet18Topology &topology,
    const ResNet18Weights &weights,
    std::size_t max_blocks)
{
    if (max_blocks > topology.blocks.size())
    {
        throw std::invalid_argument("ResNet18 staged max_blocks exceeds topology");
    }
    GpuCkksRuntime runtime(config);
    if (max_blocks == 0)
    {
        runtime.initialize_inference_evaluation_keys();
    }
    else
    {
        runtime.initialize_bootstrap();
    }
    runtime.enable_full_device_cache();
    std::cout << "[GPU ResNet18] starting layer-wise prepare/replay timing\n";
    return run_gpu_resnet18_impl(
        image_id, topology, weights, max_blocks, runtime,
        /*enable_validation=*/false, /*measure_gpu_only=*/false,
        /*measure_staged_gpu_only=*/true);
}

GpuResNet18Result run_gpu_resnet18_head_check(
    std::size_t image_id,
    const ResNet18GpuConfig &config,
    const ResNet18Topology &topology,
    const ResNet18Weights &weights)
{
    topology.validate();
    GpuResNet18Result result;
    result.image_id = image_id;
    result.true_label = load_imagenet_label(image_id);
    const auto image = load_imagenet_image_chw(image_id, kBoundary);
    const auto plain_features = plain_resnet18_features(image, topology, weights);
    const auto reference_logits = plain_head_logits(plain_features, weights);

    GpuCkksRuntime runtime(config);
    runtime.initialize_inference_evaluation_keys();
    auto encrypted_features = shared_gpu::encrypt_multiplexed_chw(
        plain_features.values, plain_features.h, plain_features.w,
        plain_features.c, 16, runtime);
    const std::size_t post_relu_q_count =
        config.application_q_count() -
        14 * config.physical_primes_per_application_level;
    for (auto &pack : encrypted_features.packs)
    {
        pack = runtime.drop_to_q_count(pack, post_relu_q_count);
    }
    auto pooled = timed("head_check.global_avgpool", [&]() {
        return shared_gpu::global_average_pool(encrypted_features, kBoundary, runtime);
    });
    auto encrypted_logits = timed("head_check.fc", [&]() {
        return shared_gpu::fully_connected(
            pooled, kFinalChannels, weights.linear_weight, weights.linear_bias,
            kClassCount, runtime);
    });
    result.logits.resize(encrypted_logits.size());
    for (std::size_t output = 0; output < encrypted_logits.size(); ++output)
    {
        result.logits[output] = runtime.decrypt(encrypted_logits[output])[0].real();
        result.max_logit_error = std::max(
            result.max_logit_error,
            std::abs(result.logits[output] - reference_logits[output]));
    }
    result.predicted_label = argmax(result.logits);
    const int plain_prediction = argmax(reference_logits);
    std::cout << "[GPU ResNet18] head_check plain_pred=" << plain_prediction
              << " gpu_pred=" << result.predicted_label
              << " max_logit_error=" << result.max_logit_error << '\n';
    if (plain_prediction != result.predicted_label || result.max_logit_error > 1.0e-3)
    {
        throw std::runtime_error("GPU ResNet18 head check failed");
    }
    return result;
}

}  // namespace poseidon::benchmark::resnet18_gpu
