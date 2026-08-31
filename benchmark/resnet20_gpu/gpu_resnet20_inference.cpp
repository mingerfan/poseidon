#include "gpu_resnet20_inference.h"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "gpu_ckks_runtime.h"
#include "gpu_multiplexed_tensor.h"
#include "gpu_relu.h"

namespace poseidon::benchmark::resnet20_gpu
{
namespace
{

namespace shared_gpu = poseidon::benchmark::resnet20_gpu::core;

constexpr double kBoundary = 40.0;
constexpr double kBatchNormEpsilon = 1.0e-5;
constexpr int kImageHeight = 32;
constexpr int kImageWidth = 32;
constexpr int kImageChannels = 3;
constexpr int kClassCount = 10;
constexpr int kFinalChannels = 64;

const std::vector<int> &resnet20_direct_rotation_steps()
{
    // Exhaustive logical rotations recorded from the complete fixed ResNet20
    // topology. Keep these keys ready before the first network operation.
    static const std::vector<int> steps{
        1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 28, 31, 32, 33, 40, 48,
        56, 62, 64, 66, 84, 124, 128, 132, 256, 512, 960, 990, 991,
        1008, 1024, 1952, 1982, 2016, 2047, 2048, 2078, 3024, 3040,
        3070, 4032, 4062, 4063, 4095, 4096, 5024, 5054, 5119, 6112,
        7135, 7168, 8190, 8191, 8192, 9184, 10207, 12285, 12288,
        15360, 16352, 16382, 16383, 16384, 17408, 18432, 19456,
        20447, 20480, 21504, 22528, 23552, 24542, 24543, 24573,
        24576, 25568, 25600, 26592, 26624, 27616, 27648, 28637,
        28640, 28672, 29600, 29632, 29664, 29696, 30624, 30656,
        30688, 30720, 31648, 31680, 31712, 31743, 31744, 31774,
        32636, 32640, 32644, 32672, 32702, 32704, 32706, 32735,
        32736, 32737, 32752, 32760, 32764, 32766, 32767};
    return steps;
}

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

BatchNormAffine folded_batch_norm(const std::vector<double> &bias,
                                  const std::vector<double> &mean,
                                  const std::vector<double> &variance,
                                  const std::vector<double> &weight)
{
    if (bias.size() != mean.size() || bias.size() != variance.size() ||
        bias.size() != weight.size())
    {
        throw std::invalid_argument("ResNet20 batch-normalization shape mismatch");
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

GpuMultiplexedTensor clone_tensor(const GpuMultiplexedTensor &input,
                                  const GpuCkksRuntime &runtime)
{
    auto output = empty_like(input);
    for (std::size_t pack = 0; pack < input.packs.size(); ++pack)
    {
        output.packs[pack] =
            runtime.drop_to_q_count(input.packs[pack], input.packs[pack].meta.q_count);
    }
    return output;
}

template <typename Function>
auto timed(const std::string &label, Function &&function)
{
    const auto begin = std::chrono::steady_clock::now();
    auto result = std::forward<Function>(function)();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
    std::cout << "[GPU ResNet20] " << label << " elapsed_ms=" << elapsed << '\n';
    return result;
}

void log_tensor_state(const std::string &label, const GpuMultiplexedTensor &tensor)
{
    if (tensor.packs.empty())
    {
        std::cout << "[GPU ResNet20] " << label << " empty\n";
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
    std::cout << "[GPU ResNet20] " << label << " shape=" << tensor.h << 'x' << tensor.w
              << 'x' << tensor.c << " k=" << tensor.k
              << " pages_per_cipher=" << tensor.pages_per_cipher
              << " packs=" << tensor.packs.size() << " q=" << minimum_q << ".."
              << maximum_q << " log2_scale=" << minimum_log_scale << ".."
              << maximum_log_scale << '\n';
}

GpuMultiplexedTensor apply_conv_bn(const GpuMultiplexedTensor &input, int out_channels,
                                   int stride, const std::vector<double> &conv_weight,
                                   const std::vector<double> &bn_bias,
                                   const std::vector<double> &bn_mean,
                                   const std::vector<double> &bn_variance,
                                   const std::vector<double> &bn_weight,
                                   const GpuCkksRuntime &runtime)
{
    const auto affine = folded_batch_norm(bn_bias, bn_mean, bn_variance, bn_weight);
    return shared_gpu::conv2d_bn(input, out_channels, stride, 3, 3, conv_weight,
                                 affine.scale, affine.bias, runtime);
}

GpuMultiplexedTensor bootstrap_relu(const std::string &label,
                                    const GpuMultiplexedTensor &input,
                                    const GpuCkksRuntime &runtime)
{
    auto refreshed = empty_like(input);
    for (std::size_t pack = 0; pack < input.packs.size(); ++pack)
    {
        refreshed.packs[pack] =
            timed(label + ".bootstrap.pack" + std::to_string(pack),
                  [&]() { return runtime.bootstrap(input.packs[pack]); });
    }
    log_tensor_state(label + ".bootstrap", refreshed);
    auto activated = empty_like(refreshed);
    for (std::size_t pack = 0; pack < refreshed.packs.size(); ++pack)
    {
        activated.packs[pack] = timed(
            label + ".relu.pack" + std::to_string(pack), [&]()
            { return shared_gpu::polynomial_relu(refreshed.packs[pack], runtime); });
    }
    log_tensor_state(label, activated);
    return activated;
}

PlainTensor plain_conv_bn(const PlainTensor &input, int out_channels, int stride,
                          const std::vector<double> &weights,
                          const BatchNormAffine &affine)
{
    const int output_h = input.h / stride;
    const int output_w = input.w / stride;
    PlainTensor output{output_h, output_w, out_channels,
                       std::vector<double>(static_cast<std::size_t>(
                           output_h * output_w * out_channels))};
    for (int oc = 0; oc < out_channels; ++oc)
    {
        for (int oh = 0; oh < output_h; ++oh)
        {
            for (int ow = 0; ow < output_w; ++ow)
            {
                double sum = 0.0;
                for (int ic = 0; ic < input.c; ++ic)
                {
                    for (int kh = 0; kh < 3; ++kh)
                    {
                        for (int kw = 0; kw < 3; ++kw)
                        {
                            const int ih = oh * stride + kh - 1;
                            const int iw = ow * stride + kw - 1;
                            if (ih < 0 || ih >= input.h || iw < 0 || iw >= input.w)
                            {
                                continue;
                            }
                            const auto input_index =
                                (static_cast<std::size_t>(ic) * input.h + ih) *
                                    input.w +
                                iw;
                            const auto weight_index =
                                ((static_cast<std::size_t>(oc) * input.c + ic) * 3 +
                                 kh) *
                                    3 +
                                kw;
                            sum += input.values[input_index] * weights[weight_index];
                        }
                    }
                }
                output
                    .values[(static_cast<std::size_t>(oc) * output_h + oh) * output_w +
                            ow] = sum * affine.scale[oc] + affine.bias[oc];
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
        throw std::invalid_argument("plain ResNet20 residual shape mismatch");
    }
    for (std::size_t index = 0; index < left.values.size(); ++index)
    {
        left.values[index] += right.values[index];
    }
    return left;
}

PlainTensor plain_downsample_shortcut(const PlainTensor &input)
{
    PlainTensor output{
        input.h / 2, input.w / 2, input.c * 2,
        std::vector<double>(
            static_cast<std::size_t>(input.h / 2 * input.w / 2 * input.c * 2), 0.0)};
    const int channel_offset = input.c / 2;
    for (int channel = 0; channel < input.c; ++channel)
    {
        for (int row = 0; row < output.h; ++row)
        {
            for (int col = 0; col < output.w; ++col)
            {
                output.values[(static_cast<std::size_t>(channel + channel_offset) *
                                   output.h +
                               row) *
                                  output.w +
                              col] =
                    input.values[(static_cast<std::size_t>(channel) * input.h +
                                  row * 2) *
                                     input.w +
                                 col * 2];
            }
        }
    }
    return output;
}

PlainTensor plain_resnet20_features(const std::vector<double> &image,
                                    const ResNet20Topology &topology,
                                    const ResNet20Weights &weights)
{
    std::size_t conv = 0;
    std::size_t bn = 0;
    PlainTensor tensor{kImageHeight, kImageWidth, kImageChannels, image};
    auto affine = folded_batch_norm(weights.bn_bias[bn], weights.bn_running_mean[bn],
                                    weights.bn_running_var[bn], weights.bn_weight[bn]);
    tensor = plain_conv_bn(tensor, 16, 1, weights.conv_weight[conv], affine);
    ++conv;
    ++bn;
    plain_relu_inplace(tensor);

    for (const auto &spec : topology.blocks)
    {
        auto shortcut = tensor;
        affine = folded_batch_norm(weights.bn_bias[bn], weights.bn_running_mean[bn],
                                   weights.bn_running_var[bn], weights.bn_weight[bn]);
        auto branch = plain_conv_bn(tensor, spec.output_channels, spec.stride,
                                    weights.conv_weight[conv], affine);
        ++conv;
        ++bn;
        plain_relu_inplace(branch);
        affine = folded_batch_norm(weights.bn_bias[bn], weights.bn_running_mean[bn],
                                   weights.bn_running_var[bn], weights.bn_weight[bn]);
        branch = plain_conv_bn(branch, spec.output_channels, 1,
                               weights.conv_weight[conv], affine);
        ++conv;
        ++bn;
        if (spec.option_a_shortcut)
        {
            shortcut = plain_downsample_shortcut(shortcut);
        }
        tensor = plain_add(std::move(branch), shortcut);
        plain_relu_inplace(tensor);
    }
    return tensor;
}

std::vector<double> plain_head_logits(const PlainTensor &features,
                                      const ResNet20Weights &weights)
{
    if (features.h != 8 || features.w != 8 || features.c != kFinalChannels)
    {
        throw std::invalid_argument("ResNet20 head expects 8x8x64 features");
    }
    std::vector<double> averages(kFinalChannels, 0.0);
    for (int channel = 0; channel < kFinalChannels; ++channel)
    {
        for (int row = 0; row < features.h; ++row)
        {
            for (int col = 0; col < features.w; ++col)
            {
                averages[channel] +=
                    features
                        .values[(static_cast<std::size_t>(channel) * features.h + row) *
                                    features.w +
                                col] *
                    kBoundary / static_cast<double>(features.h * features.w);
            }
        }
    }
    std::vector<double> logits(kClassCount, 0.0);
    for (int output = 0; output < kClassCount; ++output)
    {
        logits[output] = weights.linear_bias[output];
        for (int feature = 0; feature < kFinalChannels; ++feature)
        {
            logits[output] +=
                weights
                    .linear_weight[static_cast<std::size_t>(output) * kFinalChannels +
                                   feature] *
                averages[feature];
        }
    }
    return logits;
}

int argmax(const std::vector<double> &values)
{
    return values.empty()
               ? -1
               : static_cast<int>(std::distance(
                     values.begin(), std::max_element(values.begin(), values.end())));
}

void verify_values(const std::string &label, const std::vector<double> &expected,
                   const std::vector<double> &actual, double tolerance)
{
    if (expected.size() != actual.size())
    {
        throw std::runtime_error(label + " verification shape mismatch");
    }
    double maximum_error = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        maximum_error =
            std::max(maximum_error, std::abs(expected[index] - actual[index]));
    }
    std::cout << "[GPU ResNet20] " << label << " max_abs_error=" << maximum_error
              << '\n';
    if (!std::isfinite(maximum_error) || maximum_error > tolerance)
    {
        throw std::runtime_error(label + " verification failed");
    }
}

}  // namespace

GpuResNet20Result run_gpu_resnet20_impl(
    std::size_t image_id,
    const ResNet20Topology &topology,
    const ResNet20Weights &weights,
    std::size_t max_blocks,
    GpuCkksRuntime &runtime,
    bool enable_validation,
    bool measure_gpu_only)
{
    topology.validate();
    if (max_blocks > topology.blocks.size())
    {
        throw std::invalid_argument("ResNet20 max_blocks exceeds the topology");
    }
    GpuResNet20Result result;
    result.image_id = image_id;
    result.true_label = load_cifar10_label(image_id);
    const auto image = load_cifar10_image_chw(image_id, kBoundary);
    const bool validate = enable_validation &&
        (max_blocks <= 1 ||
         std::getenv("POSEIDON_GPU_RESNET20_VALIDATE_BLOCKS") != nullptr);
    std::chrono::steady_clock::time_point gpu_only_start;
    if (measure_gpu_only)
    {
        runtime.synchronize();
        if (cudaProfilerStart() != cudaSuccess)
        {
            throw std::runtime_error("cudaProfilerStart failed");
        }
        gpu_only_start = std::chrono::steady_clock::now();
    }

    std::size_t conv = 0;
    std::size_t bn = 0;
    auto affine = folded_batch_norm(weights.bn_bias[bn], weights.bn_running_mean[bn],
                                    weights.bn_running_var[bn], weights.bn_weight[bn]);
    auto tensor =
        timed("stem.conv1_bn1",
              [&]()
              {
                  return shared_gpu::encrypted_stem_conv2d_bn(
                      image, kImageHeight, kImageWidth, kImageChannels, 16, 1, 3, 3,
                      weights.conv_weight[conv], affine.scale, affine.bias, runtime);
              });
    ++conv;
    ++bn;
    log_tensor_state("stem.conv1_bn1", tensor);

    std::optional<PlainTensor> plain_tensor;
    if (validate)
    {
        PlainTensor plain_input{kImageHeight, kImageWidth, kImageChannels, image};
        plain_tensor =
            plain_conv_bn(plain_input, 16, 1, weights.conv_weight[0], affine);
        verify_values("stem.conv1_bn1", plain_tensor->values,
                      shared_gpu::decrypt_multiplexed_chw(tensor, runtime), 2.0e-4);
    }
    for (std::size_t pack = 0; pack < tensor.packs.size(); ++pack)
    {
        tensor.packs[pack] = shared_gpu::polynomial_relu(tensor.packs[pack], runtime);
    }
    if (plain_tensor)
    {
        plain_relu_inplace(*plain_tensor);
        verify_values("stem.relu", plain_tensor->values,
                      shared_gpu::decrypt_multiplexed_chw(tensor, runtime), 2.0e-3);
    }
    log_tensor_state("stem.relu", tensor);

    for (std::size_t block_index = 0; block_index < max_blocks; ++block_index)
    {
        const auto &spec = topology.blocks[block_index];
        const auto prefix = basic_block_name(spec);
        auto shortcut = clone_tensor(tensor, runtime);
        std::optional<PlainTensor> plain_shortcut;
        std::optional<PlainTensor> plain_branch;
        if (plain_tensor)
        {
            plain_shortcut = *plain_tensor;
            affine =
                folded_batch_norm(weights.bn_bias[bn], weights.bn_running_mean[bn],
                                  weights.bn_running_var[bn], weights.bn_weight[bn]);
            plain_branch =
                plain_conv_bn(*plain_tensor, spec.output_channels, spec.stride,
                              weights.conv_weight[conv], affine);
        }
        auto branch =
            timed(prefix + ".conv1_bn1",
                  [&]()
                  {
                      return apply_conv_bn(
                          tensor, spec.output_channels, spec.stride,
                          weights.conv_weight[conv], weights.bn_bias[bn],
                          weights.bn_running_mean[bn], weights.bn_running_var[bn],
                          weights.bn_weight[bn], runtime);
                  });
        ++conv;
        ++bn;
        if (plain_branch)
        {
            verify_values(prefix + ".conv1_bn1", plain_branch->values,
                          shared_gpu::decrypt_multiplexed_chw(branch, runtime),
                          2.0e-2);
        }
        branch = bootstrap_relu(prefix + ".act1", branch, runtime);
        if (plain_branch)
        {
            plain_relu_inplace(*plain_branch);
            affine =
                folded_batch_norm(weights.bn_bias[bn], weights.bn_running_mean[bn],
                                  weights.bn_running_var[bn], weights.bn_weight[bn]);
            *plain_branch = plain_conv_bn(*plain_branch, spec.output_channels, 1,
                                          weights.conv_weight[conv], affine);
        }
        branch =
            timed(prefix + ".conv2_bn2",
                  [&]()
                  {
                      return apply_conv_bn(
                          branch, spec.output_channels, 1, weights.conv_weight[conv],
                          weights.bn_bias[bn], weights.bn_running_mean[bn],
                          weights.bn_running_var[bn], weights.bn_weight[bn], runtime);
                  });
        ++conv;
        ++bn;
        if (plain_branch)
        {
            verify_values(prefix + ".conv2_bn2", plain_branch->values,
                          shared_gpu::decrypt_multiplexed_chw(branch, runtime),
                          2.0e-2);
        }

        if (spec.option_a_shortcut)
        {
            shortcut =
                timed(prefix + ".shortcut", [&]()
                      { return shared_gpu::downsample_shortcut(shortcut, runtime); });
            if (plain_shortcut)
            {
                *plain_shortcut = plain_downsample_shortcut(*plain_shortcut);
                verify_values(prefix + ".shortcut", plain_shortcut->values,
                              shared_gpu::decrypt_multiplexed_chw(shortcut, runtime),
                              2.0e-2);
            }
        }
        tensor = timed(prefix + ".add", [&]()
                       { return shared_gpu::residual_add(branch, shortcut, runtime); });
        tensor = bootstrap_relu(prefix + ".act2", tensor, runtime);
        if (plain_branch && plain_shortcut)
        {
            plain_tensor = plain_add(std::move(*plain_branch), *plain_shortcut);
            plain_relu_inplace(*plain_tensor);
            verify_values(prefix + ".output", plain_tensor->values,
                          shared_gpu::decrypt_multiplexed_chw(tensor, runtime), 1.0);
        }
        result.completed_blocks = block_index + 1;
    }

    if (max_blocks != topology.blocks.size())
    {
        std::cout << "[GPU ResNet20] staged run stopped after " << max_blocks
                  << " basic block(s)\n";
        return result;
    }

    auto features =
        timed("head.global_avgpool", [&]()
              { return shared_gpu::global_average_pool(tensor, kBoundary, runtime); });
    auto encrypted_logits =
        timed("head.fc",
              [&]()
              {
                  return shared_gpu::fully_connected(
                      features, kFinalChannels, weights.linear_weight,
                      weights.linear_bias, kClassCount, runtime);
              });
    if (measure_gpu_only)
    {
        runtime.synchronize();
        result.gpu_only_elapsed_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - gpu_only_start).count();
        if (cudaProfilerStop() != cudaSuccess)
        {
            throw std::runtime_error("cudaProfilerStop failed");
        }
        std::cout << "[GPU ResNet20] gpu_only_preloaded_elapsed_seconds="
                  << result.gpu_only_elapsed_seconds << '\n';
    }
    const auto decoded_logits = runtime.decrypt(encrypted_logits);
    result.logits.resize(kClassCount);
    for (std::size_t index = 0; index < result.logits.size(); ++index)
    {
        result.logits[index] = decoded_logits[index].real();
    }
    result.predicted_label = argmax(result.logits);
    if (plain_tensor)
    {
        const auto reference_logits = plain_head_logits(*plain_tensor, weights);
        for (std::size_t output = 0; output < reference_logits.size(); ++output)
        {
            result.max_logit_error =
                std::max(result.max_logit_error,
                         std::abs(result.logits[output] - reference_logits[output]));
        }
        const int plain_prediction = argmax(reference_logits);
        std::cout << "[GPU ResNet20] full_validation plain_pred=" << plain_prediction
                  << " gpu_pred=" << result.predicted_label
                  << " max_logit_error=" << result.max_logit_error << '\n';
        if (plain_prediction != result.predicted_label ||
            !std::isfinite(result.max_logit_error) || result.max_logit_error > 0.1)
        {
            throw std::runtime_error("GPU ResNet20 full validation failed");
        }
    }
    std::cout << "[GPU ResNet20] image=" << image_id
              << " true_label=" << result.true_label
              << " predicted_label=" << result.predicted_label << '\n';
    return result;
}

GpuResNet20Result run_gpu_resnet20(std::size_t image_id,
                                   const ResNet20GpuConfig &config,
                                   const ResNet20Topology &topology,
                                   const ResNet20Weights &weights,
                                   std::size_t max_blocks)
{
    GpuCkksRuntime runtime(config);
    if (max_blocks == 0)
    {
        runtime.initialize_inference_evaluation_keys();
    }
    else
    {
        runtime.initialize_bootstrap();
    }
    return run_gpu_resnet20_impl(
        image_id, topology, weights, max_blocks, runtime,
        /*enable_validation=*/true, /*measure_gpu_only=*/false);
}

GpuResNet20Result run_gpu_resnet20_preloaded(
    std::size_t image_id,
    const ResNet20GpuConfig &config,
    const ResNet20Topology &topology,
    const ResNet20Weights &weights)
{
    GpuCkksRuntime runtime(config);
    runtime.initialize_bootstrap();
    runtime.initialize_direct_rotation_keys(
        resnet20_direct_rotation_steps());
    runtime.synchronize();
    runtime.enable_full_device_cache();
    std::cout << "[GPU ResNet20] all rotation keys ready; preparing "
                 "device-resident input/model cache\n";
    const auto direct_prepared_result = run_gpu_resnet20_impl(
        image_id, topology, weights, topology.blocks.size(), runtime,
        /*enable_validation=*/false, /*measure_gpu_only=*/false);
    runtime.synchronize();

    std::cout << "[GPU ResNet20] preparation complete; starting GPU-only pass\n";
    auto measured_result = run_gpu_resnet20_impl(
        image_id, topology, weights, topology.blocks.size(), runtime,
        /*enable_validation=*/false, /*measure_gpu_only=*/true);
    double replay_logit_error = 0.0;
    if (direct_prepared_result.logits.size() != measured_result.logits.size())
    {
        throw std::runtime_error("preloaded replay logit shape mismatch");
    }
    for (std::size_t index = 0;
         index < direct_prepared_result.logits.size(); ++index)
    {
        replay_logit_error = std::max(
            replay_logit_error,
            std::abs(direct_prepared_result.logits[index] -
                     measured_result.logits[index]));
    }
    std::cout << "[GPU ResNet20] preloaded_replay_max_logit_error="
              << replay_logit_error << '\n';
    if (direct_prepared_result.predicted_label !=
            measured_result.predicted_label ||
        !std::isfinite(replay_logit_error) || replay_logit_error > 1.0e-8)
    {
        throw std::runtime_error("preloaded GPU replay verification failed");
    }
    return measured_result;
}

GpuResNet20Result run_gpu_resnet20_head_check(std::size_t image_id,
                                              const ResNet20GpuConfig &config,
                                              const ResNet20Topology &topology,
                                              const ResNet20Weights &weights)
{
    topology.validate();
    GpuResNet20Result result;
    result.image_id = image_id;
    result.true_label = load_cifar10_label(image_id);
    const auto image = load_cifar10_image_chw(image_id, kBoundary);
    const auto plain_features = plain_resnet20_features(image, topology, weights);
    const auto reference_logits = plain_head_logits(plain_features, weights);

    GpuCkksRuntime runtime(config);
    runtime.initialize_inference_evaluation_keys();
    auto encrypted_features = shared_gpu::encrypt_multiplexed_chw(
        plain_features.values, plain_features.h, plain_features.w, plain_features.c, 4,
        runtime);
    for (auto &pack : encrypted_features.packs)
    {
        pack = runtime.drop_to_q_count(pack, 11);
    }
    auto pooled =
        shared_gpu::global_average_pool(encrypted_features, kBoundary, runtime);
    auto encrypted_logits =
        shared_gpu::fully_connected(pooled, kFinalChannels, weights.linear_weight,
                                    weights.linear_bias, kClassCount, runtime);
    const auto decoded_logits = runtime.decrypt(encrypted_logits);
    result.logits.resize(kClassCount);
    for (std::size_t output = 0; output < result.logits.size(); ++output)
    {
        result.logits[output] = decoded_logits[output].real();
        result.max_logit_error =
            std::max(result.max_logit_error,
                     std::abs(result.logits[output] - reference_logits[output]));
    }
    result.predicted_label = argmax(result.logits);
    const int plain_prediction = argmax(reference_logits);
    std::cout << "[GPU ResNet20] head_check plain_pred=" << plain_prediction
              << " gpu_pred=" << result.predicted_label
              << " max_logit_error=" << result.max_logit_error << '\n';
    if (plain_prediction != result.predicted_label || result.max_logit_error > 1.0e-3)
    {
        throw std::runtime_error("GPU ResNet20 head check failed");
    }
    return result;
}

}  // namespace poseidon::benchmark::resnet20_gpu
