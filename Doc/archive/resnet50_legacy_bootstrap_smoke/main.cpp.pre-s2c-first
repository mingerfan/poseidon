#include "gpu_ckks_runtime.h"
#include "gpu_multiplexed_tensor.h"
#include "gpu_resnet50_inference.h"
#include "gpu_relu.h"
#include "resnet50_config.h"
#include "resnet50_topology.h"
#include "resnet50_weights.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void run_gpu_smoke(
    const poseidon::benchmark::resnet50_gpu::ResNet50GpuConfig &config)
{
    using poseidon::benchmark::resnet50_gpu::GpuCkksRuntime;
    GpuCkksRuntime runtime(config);

    const std::vector<double> input{ 0.125, -0.25, 0.5, 1.0, -1.5, 2.0 };
    const std::vector<double> weights{ 1.25, -0.5, 2.0, 0.125, -0.75, 1.5 };
    auto encrypted = runtime.encrypt(input);
    auto product = runtime.multiply_plain_rescale(encrypted, weights);
    const auto decoded = runtime.decrypt(product);

    double max_error = 0.0;
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        max_error = std::max(
            max_error,
            std::abs(decoded[index].real() - input[index] * weights[index]));
    }
    std::cout << "GPU multiply_plain/rescale smoke max_error=" << max_error << '\n';
    if (max_error > 1.0e-5)
    {
        throw std::runtime_error("GPU CKKS smoke error exceeds 1e-5");
    }

    // The tensor smoke below exercises pooling/convolution rotations beyond
    // the first four powers of two, so prepare the same reusable rotation set
    // used by inference.
    runtime.initialize_resnet50_evaluation_keys();
    auto squared = runtime.square_relinearize_rescale(encrypted);
    const auto decoded_square = runtime.decrypt(squared);
    max_error = 0.0;
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        max_error = std::max(
            max_error,
            std::abs(decoded_square[index].real() - input[index] * input[index]));
    }
    std::cout << "GPU square/relinearize/logical-rescale max_error="
              << max_error << '\n';
    if (max_error > 2.0e-4)
    {
        throw std::runtime_error("GPU CKKS square smoke error exceeds 2e-4");
    }

    auto rotated = runtime.rotate(encrypted, 1);
    const auto decoded_rotation = runtime.decrypt(rotated);
    max_error = 0.0;
    for (std::size_t index = 0; index + 1 < input.size(); ++index)
    {
        max_error = std::max(
            max_error,
            std::abs(decoded_rotation[index].real() - input[index + 1]));
    }
    std::cout << "GPU rotate(1) max_error=" << max_error << '\n';
    if (max_error > 1.0e-4)
    {
        throw std::runtime_error("GPU CKKS rotation smoke error exceeds 1e-4");
    }

    const std::vector<double> relu_input{
        -1.0, -0.5, -0.125, 0.0, 0.125, 0.5, 1.0,
    };
    auto encrypted_relu_input = runtime.encrypt(relu_input);
    auto encrypted_relu = poseidon::benchmark::resnet50_gpu::polynomial_relu(
        encrypted_relu_input, runtime);
    const auto decoded_relu = runtime.decrypt(encrypted_relu);
    max_error = 0.0;
    for (std::size_t index = 0; index < relu_input.size(); ++index)
    {
        const double expected =
            poseidon::benchmark::resnet50_gpu::polynomial_relu_reference(
                relu_input[index]);
        max_error = std::max(
            max_error, std::abs(decoded_relu[index].real() - expected));
        if (!std::isfinite(decoded_relu[index].real()) ||
            std::abs(decoded_relu[index].real() - expected) > 1.0)
        {
            std::cerr << "ReLU diagnostic slot=" << index
                      << " input=" << relu_input[index]
                      << " expected=" << expected
                      << " actual=" << decoded_relu[index].real() << '\n';
        }
    }
    std::cout << "GPU polynomial ReLU [15,15,27] max_error="
              << max_error
              << " q=" << encrypted_relu.meta.q_count
              << " log2_scale=" << std::log2(encrypted_relu.meta.scale) << '\n';
    if (max_error > 2.0e-2)
    {
        throw std::runtime_error("GPU polynomial ReLU error exceeds 2e-2");
    }
    auto raised_relu = runtime.bootstrap_modraise(encrypted_relu);
    const auto decoded_raised_relu = runtime.decrypt(raised_relu);
    max_error = 0.0;
    for (std::size_t index = 0; index < relu_input.size(); ++index)
    {
        max_error = std::max(
            max_error,
            std::abs(decoded_raised_relu[index].real() - decoded_relu[index].real()));
    }
    std::cout << "GPU bootstrap prepare/ModRaise max_error=" << max_error
              << " output_q=" << raised_relu.meta.q_count << '\n';
    if (max_error > 2.0e-5 || raised_relu.meta.q_count != config.log_q.size())
    {
        throw std::runtime_error("GPU extended-chain ModRaise validation failed");
    }

    std::vector<double> chw(2 * 4 * 4);
    for (std::size_t index = 0; index < chw.size(); ++index)
    {
        chw[index] = (static_cast<double>(index) - 12.0) / 16.0;
    }
    const auto tensor =
        poseidon::benchmark::resnet50_gpu::encrypt_multiplexed_chw(
            chw, 4, 4, 2, 1, runtime);
    const std::vector<double> bn_scale{ 1.5, -0.5 };
    const std::vector<double> bn_bias{ 0.25, -0.125 };
    const auto normalized = poseidon::benchmark::resnet50_gpu::batch_norm(
        tensor, bn_scale, bn_bias, runtime);
    const auto residual = poseidon::benchmark::resnet50_gpu::residual_add(
        tensor, normalized, runtime);
    const auto decoded_tensor =
        poseidon::benchmark::resnet50_gpu::decrypt_multiplexed_chw(
            residual, runtime);
    max_error = 0.0;
    for (std::size_t index = 0; index < chw.size(); ++index)
    {
        const std::size_t channel = index / 16;
        const double expected = chw[index] +
                                chw[index] * bn_scale[channel] +
                                bn_bias[channel];
        max_error = std::max(max_error, std::abs(decoded_tensor[index] - expected));
    }
    std::cout << "GPU multiplexed BN/residual max_error=" << max_error << '\n';
    if (max_error > 2.0e-5)
    {
        throw std::runtime_error("GPU multiplexed tensor smoke error exceeds 2e-5");
    }

    const std::vector<double> single_channel(chw.begin(), chw.begin() + 16);
    const auto conv_input =
        poseidon::benchmark::resnet50_gpu::encrypt_multiplexed_chw(
            single_channel, 4, 4, 1, 1, runtime);
    const auto conv_output = poseidon::benchmark::resnet50_gpu::conv2d_bn(
        conv_input,
        /*out_channels=*/1,
        /*stride=*/1,
        /*kernel_h=*/1,
        /*kernel_w=*/1,
        /*weights=*/{ 1.75 },
        /*bn_scale=*/{ -0.5 },
        /*bn_bias=*/{ 0.2 },
        runtime);
    const auto decoded_conv =
        poseidon::benchmark::resnet50_gpu::decrypt_multiplexed_chw(
            conv_output, runtime);
    max_error = 0.0;
    for (std::size_t index = 0; index < single_channel.size(); ++index)
    {
        const double expected = single_channel[index] * 1.75 * -0.5 + 0.2;
        max_error = std::max(max_error, std::abs(decoded_conv[index] - expected));
    }
    std::cout << "GPU multiplexed Conv1x1+BN max_error=" << max_error << '\n';
    // The Q36/2^40 profile trades a small amount of SIMD convolution
    // precision for a much shorter chain; real image-0 Bottleneck validation
    // remains near 1e-4, so keep this synthetic stress case below 1e-3.
    if (max_error > 1.0e-3)
    {
        throw std::runtime_error("GPU multiplexed convolution smoke error exceeds 1e-3");
    }

    auto pooled = poseidon::benchmark::resnet50_gpu::global_average_pool(
        tensor, 1.0, runtime);
    const std::vector<double> fc_matrix{
        1.25, -0.75,
        -0.5, 2.0,
    };
    const std::vector<double> fc_bias{ 0.125, -0.25 };
    auto logits = poseidon::benchmark::resnet50_gpu::fully_connected(
        pooled, 2, fc_matrix, fc_bias, 2, runtime);
    std::vector<double> channel_average(2, 0.0);
    for (std::size_t index = 0; index < chw.size(); ++index)
    {
        channel_average[index / 16] += chw[index] / 16.0;
    }
    max_error = 0.0;
    for (std::size_t output = 0; output < logits.size(); ++output)
    {
        const auto decoded_logit = runtime.decrypt(logits[output]);
        const double expected =
            channel_average[0] * fc_matrix[output * 2] +
            channel_average[1] * fc_matrix[output * 2 + 1] +
            fc_bias[output];
        max_error = std::max(
            max_error, std::abs(decoded_logit[0].real() - expected));
    }
    std::cout << "GPU global-average-pool/FC max_error=" << max_error << '\n';
    if (max_error > 5.0e-4)
    {
        throw std::runtime_error("GPU pooling/FC smoke error exceeds 5e-4");
    }

    const std::vector<double> pool_values{ 1.0, 2.0, 3.0, 4.0 };
    const auto pool_input =
        poseidon::benchmark::resnet50_gpu::encrypt_multiplexed_chw(
            pool_values, 2, 2, 1, 1, runtime);
    const auto pool_output =
        poseidon::benchmark::resnet50_gpu::average_pool2d_stride2(
            pool_input, runtime);
    const auto decoded_pool =
        poseidon::benchmark::resnet50_gpu::decrypt_multiplexed_chw(
            pool_output, runtime);
    const double pool_error = std::abs(
        decoded_pool[0] - (1.0 + 2.0 + 3.0 + 4.0) / 9.0);
    std::cout << "GPU average-pool stride2 max_error=" << pool_error << '\n';
    if (pool_error > 2.0e-5)
    {
        throw std::runtime_error("GPU average-pool smoke error exceeds 2e-5");
    }
}

void run_bootstrap_smoke(
    const poseidon::benchmark::resnet50_gpu::ResNet50GpuConfig &config)
{
    using poseidon::benchmark::resnet50_gpu::GpuCkksRuntime;
    GpuCkksRuntime runtime(config);
    const std::vector<double> pattern{
        -0.125, -0.0625, -0.01, 0.0, 0.01, 0.0625, 0.125,
    };
    std::vector<double> input(runtime.slot_count());
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        input[index] = pattern[index % pattern.size()];
    }
    auto encrypted = runtime.encrypt(input);
    std::cout << "Initializing reusable GPU bootstrap data..." << std::endl;
    runtime.initialize_bootstrap();
    std::cout << "Executing full GPU bootstrap..." << std::endl;
    auto refreshed = runtime.bootstrap(encrypted);
    const auto decoded = runtime.decrypt(refreshed);
    double max_error = 0.0;
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        max_error = std::max(
            max_error, std::abs(decoded[index].real() - input[index]));
    }
    std::cout << "GPU full bootstrap max_error=" << max_error
              << " output_q=" << refreshed.meta.q_count
              << " log2_scale=" << std::log2(refreshed.meta.scale) << '\n';
    // The S2C 2^60 matrix-scale profile is expected to stay well below 1e-6
    // over all 32768 slots (about 4.3e-8 on the validation V100).
    if (max_error > 1.0e-6)
    {
        throw std::runtime_error("GPU full bootstrap error exceeds 1e-6");
    }
}

}  // namespace

int main(int argc, char **argv)
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    try
    {
        const auto config =
            poseidon::benchmark::resnet50_gpu::make_resnet50_gpu_config();
        const auto topology =
            poseidon::benchmark::resnet50_gpu::make_resnet50_topology();
        std::cout << "Poseidon GPU ResNet50\n"
                  << "N=" << config.degree()
                  << " slots=" << config.slot_count()
                  << " Q=" << config.log_q.size()
                  << " P=" << config.log_p.size()
                  << " dnum=" << config.dnum << '\n'
                  << "application_scale=2^" << config.application_log_scale
                  << " evalmod_scale=2^" << config.evalmod_log_scale
                  << " bootstrap_output_scale=2^"
                  << config.bootstrap_output_log_scale
                  << " bootstrap_Q=" << config.bootstrap_q_count << '\n'
                  << "application_q=" << config.application_q_count()
                  << " logical_application_levels=" << config.application_levels << '\n'
                  << "blocks=" << topology.blocks.size()
                  << " bootstrap_points=" << topology.bootstrap_point_count()
                  << '\n';
        if (argc == 2 && std::string(argv[1]) == "--smoke")
        {
            run_gpu_smoke(config);
        }
        else if (argc == 2 && std::string(argv[1]) == "--relu-reference")
        {
            for (double value : { -1.5, -1.0, -0.5, -0.125, 0.0, 0.125, 0.5, 1.0, 1.5 })
            {
                std::cout << "relu_reference(" << value << ")="
                          << poseidon::benchmark::resnet50_gpu::polynomial_relu_reference(value)
                          << '\n';
            }
        }
        else if (argc == 2 && std::string(argv[1]) == "--bootstrap-smoke")
        {
            run_bootstrap_smoke(config);
        }
        else if (argc == 3 && std::string(argv[1]) == "--head-check")
        {
            const auto image_id = static_cast<std::size_t>(std::stoull(argv[2]));
            const auto weights =
                poseidon::benchmark::resnet50_gpu::load_resnet50_weights();
            const auto result =
                poseidon::benchmark::resnet50_gpu::run_gpu_resnet50_head_check(
                    image_id, config, topology, weights);
            std::cout << "GPU ResNet50 head check image=" << result.image_id
                      << " predicted_label=" << result.predicted_label
                      << " max_logit_error=" << result.max_logit_error << '\n';
        }
        else if ((argc == 3 || argc == 4) &&
                 std::string(argv[1]) == "--infer")
        {
            const auto inference_start = std::chrono::steady_clock::now();
            const auto image_id = static_cast<std::size_t>(std::stoull(argv[2]));
            const auto max_blocks = argc == 4
                ? static_cast<std::size_t>(std::stoull(argv[3]))
                : topology.blocks.size();
            const auto weights =
                poseidon::benchmark::resnet50_gpu::load_resnet50_weights();
            const auto result =
                poseidon::benchmark::resnet50_gpu::run_gpu_resnet50(
                    image_id, config, topology, weights, max_blocks);
            std::cout << "GPU ResNet50 result image=" << result.image_id
                      << " completed_blocks=" << result.completed_blocks
                      << " true_label=" << result.true_label
                      << " predicted_label=" << result.predicted_label << '\n';
            const auto inference_elapsed =
                std::chrono::steady_clock::now() - inference_start;
            const auto inference_elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    inference_elapsed).count();
            const auto elapsed_hours = inference_elapsed_ms / 3600000;
            const auto elapsed_minutes =
                (inference_elapsed_ms % 3600000) / 60000;
            const auto elapsed_seconds =
                static_cast<double>(inference_elapsed_ms % 60000) / 1000.0;
            std::cout << "[GPU ResNet50] inference_total_elapsed_ms="
                      << inference_elapsed_ms
                      << " inference_total_elapsed_seconds="
                      << static_cast<double>(inference_elapsed_ms) / 1000.0
                      << " inference_total_elapsed_hours="
                      << static_cast<double>(inference_elapsed_ms) / 3600000.0
                      << " inference_total_elapsed_hms="
                      << elapsed_hours << 'h'
                      << elapsed_minutes << 'm'
                      << elapsed_seconds << 's'
                      << '\n';
        }
        else if ((argc == 3 || argc == 4) &&
                 std::string(argv[1]) == "--gpu-only")
        {
            const auto image_id = static_cast<std::size_t>(std::stoull(argv[2]));
            const auto max_blocks = argc == 4
                ? static_cast<std::size_t>(std::stoull(argv[3]))
                : topology.blocks.size();
            const auto weights =
                poseidon::benchmark::resnet50_gpu::load_resnet50_weights();
            const auto result =
                poseidon::benchmark::resnet50_gpu::run_gpu_resnet50_preloaded(
                    image_id, config, topology, weights, max_blocks);
            std::cout << "GPU ResNet50 preloaded result image=" << result.image_id
                      << " completed_blocks=" << result.completed_blocks
                      << " true_label=" << result.true_label
                      << " predicted_label=" << result.predicted_label
                      << " gpu_only_elapsed_seconds="
                      << result.gpu_only_elapsed_seconds << '\n';
        }
        else if ((argc == 3 || argc == 4) &&
                 std::string(argv[1]) == "--gpu-staged")
        {
            const auto image_id = static_cast<std::size_t>(std::stoull(argv[2]));
            const auto max_blocks = argc == 4
                ? static_cast<std::size_t>(std::stoull(argv[3]))
                : topology.blocks.size();
            const auto weights =
                poseidon::benchmark::resnet50_gpu::load_resnet50_weights();
            const auto result =
                poseidon::benchmark::resnet50_gpu::run_gpu_resnet50_staged_gpu_only(
                    image_id, config, topology, weights, max_blocks);
            std::cout << "GPU ResNet50 staged result image=" << result.image_id
                      << " completed_blocks=" << result.completed_blocks
                      << " true_label=" << result.true_label
                      << " predicted_label=" << result.predicted_label
                      << " staged_gpu_only_elapsed_seconds="
                      << result.gpu_only_elapsed_seconds << '\n';
        }
        else if (argc == 2 && std::string(argv[1]) == "--weights-check")
        {
            const auto weights =
                poseidon::benchmark::resnet50_gpu::load_resnet50_weights();
            const auto image =
                poseidon::benchmark::resnet50_gpu::load_imagenet_image_chw(
                    0, 120.0);
            const int label =
                poseidon::benchmark::resnet50_gpu::load_imagenet_label(0);
            std::cout << "weights_check conv=" << weights.conv_weight.size()
                      << " bn=" << weights.bn_weight.size()
                      << " downsample=" << weights.downsample_weight.size()
                      << " fc=" << weights.linear_weight.size()
                      << " image_values=" << image.size()
                      << " label0=" << label << '\n';
        }
        else if (argc != 1)
        {
            throw std::invalid_argument(
                "usage: poseidon_gpu_resnet50 "
                "[--smoke|--bootstrap-smoke|--relu-reference|--weights-check|"
                "--head-check IMAGE_ID|"
                "--infer IMAGE_ID [MAX_BLOCKS]|"
                "--gpu-only IMAGE_ID [MAX_BLOCKS]|"
                "--gpu-staged IMAGE_ID [MAX_BLOCKS]]");
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "poseidon_gpu_resnet50: " << error.what() << '\n';
        return 1;
    }
}
