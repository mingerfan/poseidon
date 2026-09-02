#include "gpu_resnet18_inference.h"
#include "gpu_config.h"
#include "resnet18_topology.h"
#include "resnet18_weights.h"

#include "gpu_ckks_runtime.h"

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

namespace resnet18 = poseidon::benchmark::resnet18_gpu;
namespace shared_gpu = poseidon::benchmark::resnet50_gpu;

void run_gpu_smoke(const shared_gpu::ResNet50GpuConfig &config)
{
    shared_gpu::GpuCkksRuntime runtime(config);
    const std::vector<double> input{0.125, -0.25, 0.5, 1.0};
    const std::vector<double> weights{1.25, -0.5, 2.0, 0.125};
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
    std::cout << "GPU ResNet18 multiply_plain smoke max_error=" << max_error << '\n';
    if (max_error > 1.0e-5)
    {
        throw std::runtime_error("GPU ResNet18 smoke error exceeds 1e-5");
    }

    // Torchvision ResNet18 contains nonzero stem weights below the current
    // 32-bit plaintext quantization unit. They must round to zero cleanly.
    auto quantized_zero = runtime.multiply_plain_scalar_rescale(
        encrypted, 2.22518393e-11);
    const auto decoded_zero = runtime.decrypt(quantized_zero);
    max_error = 0.0;
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        max_error = std::max(max_error, std::abs(decoded_zero[index].real()));
    }
    std::cout << "GPU ResNet18 sub-resolution scalar max_error="
              << max_error << '\n';
    if (max_error > 1.0e-8 ||
        quantized_zero.meta.q_count + 1 != encrypted.meta.q_count)
    {
        throw std::runtime_error(
            "GPU ResNet18 sub-resolution scalar quantization metadata is invalid");
    }

    auto scalar_product = runtime.multiply_plain_scalar_rescale(encrypted, -0.125);
    const auto decoded_scalar = runtime.decrypt(scalar_product);
    max_error = 0.0;
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        max_error = std::max(
            max_error, std::abs(decoded_scalar[index].real() + input[index] * 0.125));
    }
    std::cout << "GPU ResNet18 device scalar max_error=" << max_error << '\n';
    if (max_error > 1.0e-5)
    {
        throw std::runtime_error("GPU ResNet18 device scalar smoke failed");
    }
}

void print_elapsed(std::chrono::steady_clock::duration elapsed)
{
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    const auto hours = elapsed_ms / 3600000;
    const auto minutes = (elapsed_ms % 3600000) / 60000;
    const auto seconds = static_cast<double>(elapsed_ms % 60000) / 1000.0;
    std::cout << "[GPU ResNet18] inference_total_elapsed_ms=" << elapsed_ms
              << " inference_total_elapsed_seconds="
              << static_cast<double>(elapsed_ms) / 1000.0
              << " inference_total_elapsed_hours="
              << static_cast<double>(elapsed_ms) / 3600000.0
              << " inference_total_elapsed_hms="
              << hours << 'h' << minutes << 'm' << seconds << 's' << '\n';
}

}  // namespace

int main(int argc, char **argv)
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    try
    {
        const auto config = resnet18::make_resnet18_gpu_config();
        const auto topology = resnet18::make_resnet18_topology();
        std::cout << "Poseidon GPU ResNet18\n"
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
                  << " bootstrap_points=" << topology.bootstrap_point_count() << '\n';

        if (argc == 2 && std::string(argv[1]) == "--smoke")
        {
            run_gpu_smoke(config);
        }
        else if (argc == 2 && std::string(argv[1]) == "--topology-check")
        {
            topology.validate();
            for (const auto &block : topology.blocks)
            {
                std::cout << resnet18::basic_block_name(block)
                          << " input_channels=" << block.input_channels
                          << " output_channels=" << block.output_channels
                          << " stride=" << block.stride
                          << " projection=" << (block.projection ? 1 : 0) << '\n';
            }
        }
        else if (argc == 2 && std::string(argv[1]) == "--weights-check")
        {
            const auto weights = resnet18::load_resnet18_weights();
            const auto image = resnet18::load_imagenet_image_chw(0, 20.0);
            const int label = resnet18::load_imagenet_label(0);
            std::cout << "weights_check conv=" << weights.conv_weight.size()
                      << " bn=" << weights.bn_weight.size()
                      << " downsample=" << weights.downsample_weight.size()
                      << " fc=" << weights.linear_weight.size()
                      << " image_values=" << image.size()
                      << " label0=" << label << '\n';
        }
        else if (argc == 3 && std::string(argv[1]) == "--head-check")
        {
            const auto start = std::chrono::steady_clock::now();
            const auto image_id = static_cast<std::size_t>(std::stoull(argv[2]));
            const auto weights = resnet18::load_resnet18_weights();
            const auto result = resnet18::run_gpu_resnet18_head_check(
                image_id, config, topology, weights);
            std::cout << "GPU ResNet18 head check image=" << result.image_id
                      << " predicted_label=" << result.predicted_label
                      << " max_logit_error=" << result.max_logit_error << '\n';
            print_elapsed(std::chrono::steady_clock::now() - start);
        }
        else if ((argc == 3 || argc == 4) && std::string(argv[1]) == "--infer")
        {
            const auto start = std::chrono::steady_clock::now();
            const auto image_id = static_cast<std::size_t>(std::stoull(argv[2]));
            const auto max_blocks = argc == 4
                ? static_cast<std::size_t>(std::stoull(argv[3]))
                : topology.blocks.size();
            const auto weights = resnet18::load_resnet18_weights();
            const auto result = resnet18::run_gpu_resnet18(
                image_id, config, topology, weights, max_blocks);
            std::cout << "GPU ResNet18 result image=" << result.image_id
                      << " completed_blocks=" << result.completed_blocks
                      << " true_label=" << result.true_label
                      << " predicted_label=" << result.predicted_label << '\n';
            print_elapsed(std::chrono::steady_clock::now() - start);
        }
        else if ((argc == 3 || argc == 4) &&
                 std::string(argv[1]) == "--gpu-only")
        {
            const auto image_id = static_cast<std::size_t>(std::stoull(argv[2]));
            const auto max_blocks = argc == 4
                ? static_cast<std::size_t>(std::stoull(argv[3]))
                : topology.blocks.size();
            const auto weights = resnet18::load_resnet18_weights();
            const auto result = resnet18::run_gpu_resnet18_preloaded(
                image_id, config, topology, weights, max_blocks);
            std::cout << "GPU ResNet18 preloaded result image=" << result.image_id
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
            const auto weights = resnet18::load_resnet18_weights();
            const auto result = resnet18::run_gpu_resnet18_staged_gpu_only(
                image_id, config, topology, weights, max_blocks);
            std::cout << "GPU ResNet18 staged result image=" << result.image_id
                      << " completed_blocks=" << result.completed_blocks
                      << " true_label=" << result.true_label
                      << " predicted_label=" << result.predicted_label
                      << " staged_gpu_only_elapsed_seconds="
                      << result.gpu_only_elapsed_seconds << '\n';
        }
        else
        {
            throw std::invalid_argument(
                "usage: poseidon_gpu_resnet18 "
                "[--smoke|--topology-check|--weights-check|"
                "--head-check IMAGE_ID|--infer IMAGE_ID [MAX_BLOCKS]|"
                "--gpu-only IMAGE_ID [MAX_BLOCKS]|"
                "--gpu-staged IMAGE_ID [MAX_BLOCKS]]");
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "poseidon_gpu_resnet18: " << error.what() << '\n';
        return 1;
    }
}
