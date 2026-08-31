#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_ckks_runtime.h"
#include "gpu_multiplexed_tensor.h"
#include "gpu_resnet20_inference.h"
#include "resnet20_topology.h"
#include "resnet20_weights.h"
#include "gpu_config.h"

namespace
{

namespace resnet20 = poseidon::benchmark::resnet20_gpu;
namespace shared_gpu = poseidon::benchmark::resnet20_gpu::core;

void run_gpu_smoke(const shared_gpu::GpuConfig &config)
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
            max_error, std::abs(decoded[index].real() - input[index] * weights[index]));
    }
    std::cout << "GPU ResNet20 smoke max_error=" << max_error << '\n';
    if (max_error > 1.0e-5)
    {
        throw std::runtime_error("GPU ResNet20 smoke error exceeds 1e-5");
    }
}

void run_shortcut_check(const shared_gpu::GpuConfig &config)
{
    shared_gpu::GpuCkksRuntime runtime(config);
    runtime.initialize_inference_evaluation_keys();
    constexpr int input_height = 32;
    constexpr int input_width = 32;
    constexpr int output_height = input_height / 2;
    constexpr int output_width = input_width / 2;
    constexpr int input_channels = 16;
    constexpr int output_channels = 32;
    std::vector<double> input(input_height * input_width * input_channels);
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        input[index] =
            static_cast<double>(static_cast<int>(index % 101) - 50) / 1000.0;
    }
    auto encrypted = shared_gpu::encrypt_multiplexed_chw(
        input, input_height, input_width, input_channels, 1, runtime);
    auto downsampled = shared_gpu::downsample_shortcut(encrypted, runtime);
    const auto actual = shared_gpu::decrypt_multiplexed_chw(downsampled, runtime);
    std::vector<double> expected(
        output_height * output_width * output_channels, 0.0);
    for (int channel = 0; channel < input_channels; ++channel)
    {
        for (int row = 0; row < output_height; ++row)
        {
            for (int col = 0; col < output_width; ++col)
            {
                expected[(static_cast<std::size_t>(channel + 8) * output_height + row) *
                             output_width +
                         col] = input[(static_cast<std::size_t>(channel) * input_height +
                                      row * 2) *
                                         input_width +
                                     col * 2];
            }
        }
    }
    double max_error = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        max_error = std::max(max_error, std::abs(expected[index] - actual[index]));
    }
    std::cout << "GPU ResNet20 Option-A shortcut max_error=" << max_error << '\n';
    if (max_error > 1.0e-5 || downsampled.h != output_height ||
        downsampled.w != output_width || downsampled.c != output_channels ||
        downsampled.k != 2 ||
        downsampled.packs.size() != 1 || downsampled.pages_per_cipher != 32)
    {
        throw std::runtime_error("GPU ResNet20 Option-A shortcut check failed");
    }
}

void run_hoist_check(const shared_gpu::GpuConfig &config)
{
    shared_gpu::GpuCkksRuntime runtime(config);
    const std::vector<int> steps{1, 3, 7, 31};
    runtime.initialize_direct_rotation_keys(steps);
    std::vector<double> input(runtime.slot_count());
    for (std::size_t slot = 0; slot < input.size(); ++slot)
    {
        input[slot] =
            static_cast<double>(static_cast<int>(slot % 257) - 128) / 4096.0;
    }
    auto encrypted = runtime.encrypt(input);
    encrypted = runtime.drop_to_q_count(encrypted, 8);
    std::vector<long long> batch_steps(steps.begin(), steps.end());
    auto batched = runtime.rotate_many_composed(encrypted, batch_steps);
    double max_error = 0.0;
    for (std::size_t index = 0; index < steps.size(); ++index)
    {
        const auto reference = runtime.decrypt(
            runtime.rotate(encrypted, steps[index]));
        const auto actual = runtime.decrypt(batched[index]);
        for (std::size_t slot = 0; slot < actual.size(); ++slot)
        {
            max_error = std::max(
                max_error,
                std::abs(actual[slot].real() - reference[slot].real()));
        }
    }
    std::cout << "GPU ResNet20 hoisted rotate max_error=" << max_error << '\n';
    if (max_error > 1.0e-5)
    {
        throw std::runtime_error("GPU ResNet20 hoisted rotate check failed");
    }
}

void print_elapsed(std::chrono::steady_clock::duration elapsed)
{
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    std::cout << "[GPU ResNet20] inference_total_elapsed_ms=" << elapsed_ms
              << " inference_total_elapsed_seconds="
              << static_cast<double>(elapsed_ms) / 1000.0 << '\n';
}

}  // namespace

int main(int argc, char **argv)
{
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    try
    {
        const auto config = shared_gpu::make_gpu_config();
        const auto topology = resnet20::make_resnet20_topology();
        std::cout << "Poseidon GPU ResNet20\n"
                  << "N=" << config.degree() << " slots=" << config.slot_count()
                  << " Q=" << config.log_q.size() << " P=" << config.log_p.size()
                  << " dnum=" << config.dnum << '\n'
                  << "application_scale=2^" << config.application_log_scale
                  << " evalmod_scale=2^" << config.evalmod_log_scale
                  << " bootstrap_output_scale=2^"
                  << config.bootstrap_output_log_scale
                  << " bootstrap_Q=" << config.bootstrap_q_count << '\n'
                  << "blocks=" << topology.blocks.size()
                  << " bootstrap_points=" << topology.bootstrap_point_count() << '\n';

        if (argc == 2 && std::string(argv[1]) == "--smoke")
        {
            run_gpu_smoke(config);
        }
        else if (argc == 2 && std::string(argv[1]) == "--shortcut-check")
        {
            run_shortcut_check(config);
        }
        else if (argc == 2 && std::string(argv[1]) == "--hoist-check")
        {
            run_hoist_check(config);
        }
        else if (argc == 2 && std::string(argv[1]) == "--topology-check")
        {
            topology.validate();
            for (const auto &block : topology.blocks)
            {
                std::cout << resnet20::basic_block_name(block)
                          << " input_channels=" << block.input_channels
                          << " output_channels=" << block.output_channels
                          << " stride=" << block.stride
                          << " option_a=" << (block.option_a_shortcut ? 1 : 0) << '\n';
            }
        }
        else if (argc == 2 && std::string(argv[1]) == "--weights-check")
        {
            const auto weights = resnet20::load_resnet20_weights();
            const auto image = resnet20::load_cifar10_image_chw(0, 40.0);
            const int label = resnet20::load_cifar10_label(0);
            std::cout << "weights_check conv=" << weights.conv_weight.size()
                      << " bn=" << weights.bn_weight.size()
                      << " fc=" << weights.linear_weight.size()
                      << " image_values=" << image.size() << " label0=" << label
                      << '\n';
        }
        else if (argc == 3 && std::string(argv[1]) == "--head-check")
        {
            const auto start = std::chrono::steady_clock::now();
            const auto image_id = static_cast<std::size_t>(std::stoull(argv[2]));
            const auto weights = resnet20::load_resnet20_weights();
            const auto result = resnet20::run_gpu_resnet20_head_check(
                image_id, config, topology, weights);
            std::cout << "GPU ResNet20 head check image=" << result.image_id
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
            const auto weights = resnet20::load_resnet20_weights();
            const auto result = resnet20::run_gpu_resnet20(image_id, config, topology,
                                                           weights, max_blocks);
            std::cout << "GPU ResNet20 result image=" << result.image_id
                      << " completed_blocks=" << result.completed_blocks
                      << " true_label=" << result.true_label
                      << " predicted_label=" << result.predicted_label << '\n';
            print_elapsed(std::chrono::steady_clock::now() - start);
        }
        else if (argc == 3 && std::string(argv[1]) == "--gpu-only")
        {
            const auto image_id = static_cast<std::size_t>(std::stoull(argv[2]));
            const auto weights = resnet20::load_resnet20_weights();
            const auto result = resnet20::run_gpu_resnet20_preloaded(
                image_id, config, topology, weights);
            std::cout << "GPU ResNet20 preloaded result image=" << result.image_id
                      << " true_label=" << result.true_label
                      << " predicted_label=" << result.predicted_label
                      << " gpu_only_elapsed_seconds="
                      << result.gpu_only_elapsed_seconds << '\n';
        }
        else
        {
            throw std::invalid_argument(
                "usage: poseidon_gpu_resnet20 "
                "[--smoke|--shortcut-check|--hoist-check|--topology-check|--weights-check|"
                "--head-check IMAGE_ID|--infer IMAGE_ID [MAX_BLOCKS]|"
                "--gpu-only IMAGE_ID]");
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "poseidon_gpu_resnet20: " << error.what() << '\n';
        return 1;
    }
}
