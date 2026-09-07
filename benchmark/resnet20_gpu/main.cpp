#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

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

void print_gpu_memory_snapshot(const char *phase)
{
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("cudaMemGetInfo failed: ") +
            cudaGetErrorString(status));
    }
    constexpr double mib = 1024.0 * 1024.0;
    std::cout << "[GPU memory] phase=" << phase
              << " used_mib="
              << static_cast<double>(total_bytes - free_bytes) / mib
              << " free_mib=" << static_cast<double>(free_bytes) / mib
              << " total_mib=" << static_cast<double>(total_bytes) / mib
              << '\n';
}

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
    runtime.initialize_direct_rotation_keys(steps, {7, 8});
    std::vector<double> input(runtime.slot_count());
    for (std::size_t slot = 0; slot < input.size(); ++slot)
    {
        input[slot] =
            static_cast<double>(static_cast<int>(slot % 257) - 128) / 4096.0;
    }
    const auto encrypted = runtime.encrypt(input);
    std::vector<long long> batch_steps(steps.begin(), steps.end());
    double max_error = 0.0;
    double max_relinearize_error = 0.0;
    for (const std::size_t q_count : {std::size_t{8}, std::size_t{7}})
    {
        const auto shape = runtime.application_keyswitch_shape(q_count);
        std::cout << "GPU ResNet20 application keyswitch route q=" << q_count
                  << " p=" << shape.p_count
                  << " effective_dnum=" << shape.effective_dnum
                  << " level_aware=" << (shape.level_aware ? 1 : 0) << '\n';
        const auto at_level = runtime.drop_to_q_count(encrypted, q_count);
        auto batched = runtime.rotate_many_composed(at_level, batch_steps);
        double level_max_error = 0.0;
        for (std::size_t index = 0; index < steps.size(); ++index)
        {
            const auto direct = runtime.decrypt(
                runtime.rotate(at_level, steps[index]));
            const auto actual = runtime.decrypt(batched[index]);
            for (std::size_t slot = 0; slot < actual.size(); ++slot)
            {
                const auto expected_slot =
                    (slot + static_cast<std::size_t>(steps[index])) % input.size();
                level_max_error = std::max(
                    level_max_error,
                    std::abs(actual[slot].real() - input[expected_slot]));
                level_max_error = std::max(
                    level_max_error,
                    std::abs(direct[slot].real() - input[expected_slot]));
            }
        }
        max_error = std::max(max_error, level_max_error);
        std::cout << "GPU ResNet20 level-aware hoisted rotate q=" << q_count
                  << " max_error=" << level_max_error << '\n';

        const auto squared = runtime.decrypt(
            runtime.square_relinearize_rescale(at_level));
        double level_relinearize_error = 0.0;
        for (std::size_t slot = 0; slot < squared.size(); ++slot)
        {
            const double expected = input[slot] * input[slot];
            level_relinearize_error = std::max(
                level_relinearize_error,
                std::abs(squared[slot].real() - expected));
        }
        max_relinearize_error = std::max(
            max_relinearize_error, level_relinearize_error);
        std::cout << "GPU ResNet20 level-aware relinearize q=" << q_count
                  << " max_error=" << level_relinearize_error << '\n';
    }
    if (max_error > 5.0e-4 || max_relinearize_error > 5.0e-4)
    {
        throw std::runtime_error("GPU ResNet20 hoisted rotate check failed");
    }
}

void run_fixed_dnum_check(const shared_gpu::GpuConfig &config)
{
    if (::setenv(
            "POSEIDON_APPLICATION_KEYSWITCH_P_MODE",
            "fixed_dnum",
            1) != 0)
    {
        throw std::runtime_error(
            "failed to enable fixed-dnum application KeySwitch mode");
    }

    shared_gpu::GpuCkksRuntime runtime(config);
    const std::vector<int> steps{1, 3, 7, 31};
    runtime.initialize_direct_rotation_keys(steps, {7, 8});

    const auto expected_p_count = [&](std::size_t q_count)
    {
        return std::max<std::size_t>(
            2, (q_count + config.dnum - 1) / config.dnum);
    };
    for (const std::size_t q_count : {std::size_t{7}, std::size_t{8}})
    {
        const auto shape = runtime.application_keyswitch_shape(q_count);
        const auto expected_p = expected_p_count(q_count);
        if (!shape.level_aware || shape.p_count != expected_p ||
            shape.effective_dnum !=
                (q_count + expected_p - 1) / expected_p)
        {
            throw std::runtime_error(
                "fixed-dnum convolution KeySwitch route has the wrong shape");
        }
        std::cout << "GPU ResNet20 fixed-dnum route q=" << q_count
                  << " p=" << shape.p_count
                  << " target_dnum=" << config.dnum
                  << " effective_dnum=" << shape.effective_dnum << '\n';
    }

    std::vector<double> input(runtime.slot_count());
    for (std::size_t slot = 0; slot < input.size(); ++slot)
    {
        input[slot] =
            static_cast<double>(static_cast<int>(slot % 257) - 128) / 4096.0;
    }
    const auto encrypted = runtime.encrypt(input);

    double rotation_max_error = 0.0;
    std::vector<long long> batch_steps(steps.begin(), steps.end());
    for (const std::size_t q_count : {std::size_t{8}, std::size_t{7}})
    {
        const auto at_level = runtime.drop_to_q_count(encrypted, q_count);
        const auto rotated = runtime.rotate_many_composed(at_level, batch_steps);
        for (std::size_t index = 0; index < rotated.size(); ++index)
        {
            const auto actual = runtime.decrypt(rotated[index]);
            for (std::size_t slot = 0; slot < actual.size(); ++slot)
            {
                const auto expected_slot =
                    (slot + static_cast<std::size_t>(steps[index])) % input.size();
                rotation_max_error = std::max(
                    rotation_max_error,
                    std::abs(actual[slot].real() - input[expected_slot]));
            }
        }
    }

    // Exercise the lazy high-Q context used by application ReLU. Q32 maps to
    // P16/P11/P8 for configured dnum 2/3/4 respectively.
    const std::size_t high_q_count = config.application_q_count() - 4;
    const auto high_shape = runtime.application_keyswitch_shape(high_q_count);
    const auto expected_high_p = expected_p_count(high_q_count);
    if (!high_shape.level_aware || high_shape.p_count != expected_high_p ||
        high_shape.effective_dnum != config.dnum)
    {
        throw std::runtime_error(
            "fixed-dnum high-Q relinearization route has the wrong shape");
    }
    const auto high_level = runtime.drop_to_q_count(encrypted, high_q_count);
    const auto squared = runtime.decrypt(
        runtime.square_relinearize_rescale(high_level));
    double relinearize_max_error = 0.0;
    for (std::size_t slot = 0; slot < squared.size(); ++slot)
    {
        relinearize_max_error = std::max(
            relinearize_max_error,
            std::abs(squared[slot].real() - input[slot] * input[slot]));
    }
    std::cout << "GPU ResNet20 fixed-dnum high route q=" << high_q_count
              << " p=" << high_shape.p_count
              << " target_dnum=" << config.dnum
              << " effective_dnum=" << high_shape.effective_dnum
              << " rotation_max_error=" << rotation_max_error
              << " relinearize_max_error=" << relinearize_max_error << '\n';
    if (rotation_max_error > 5.0e-4 || relinearize_max_error > 5.0e-4)
    {
        throw std::runtime_error(
            "GPU ResNet20 fixed-dnum application KeySwitch check failed");
    }
}

void run_bootstrap_check(const shared_gpu::GpuConfig &config)
{
    if (::setenv("POSEIDON_LEVEL_AWARE_BOOTSTRAP_P", "1", 1) != 0)
    {
        throw std::runtime_error(
            "failed to enable level-aware Bootstrap P for correctness check");
    }
    shared_gpu::GpuCkksRuntime runtime(config);
    runtime.initialize_bootstrap();

    std::vector<double> input(runtime.slot_count());
    for (std::size_t slot = 0; slot < input.size(); ++slot)
    {
        input[slot] =
            static_cast<double>(static_cast<int>(slot % 257) - 128) /
            2048.0;
    }
    const auto encrypted = runtime.encrypt(input);

    const auto optimized = runtime.bootstrap(encrypted);
    runtime.synchronize();
    if (::setenv("POSEIDON_LEVEL_AWARE_BOOTSTRAP_P", "0", 1) != 0)
    {
        throw std::runtime_error(
            "failed to disable level-aware Bootstrap P for reference check");
    }
    const auto global_basis = runtime.bootstrap(encrypted);
    runtime.synchronize();
    (void)::unsetenv("POSEIDON_LEVEL_AWARE_BOOTSTRAP_P");

    const auto optimized_slots = runtime.decrypt(optimized);
    const auto global_slots = runtime.decrypt(global_basis);
    double optimized_max_error = 0.0;
    double global_max_error = 0.0;
    double optimized_vs_global_max_error = 0.0;
    for (std::size_t slot = 0; slot < input.size(); ++slot)
    {
        optimized_max_error = std::max(
            optimized_max_error,
            std::abs(optimized_slots[slot].real() - input[slot]));
        global_max_error = std::max(
            global_max_error,
            std::abs(global_slots[slot].real() - input[slot]));
        optimized_vs_global_max_error = std::max(
            optimized_vs_global_max_error,
            std::abs(
                optimized_slots[slot].real() -
                global_slots[slot].real()));
    }
    std::cout << "GPU ResNet20 Bootstrap level-aware max_error="
              << optimized_max_error
              << " global_basis_max_error=" << global_max_error
              << " optimized_vs_global_max_error="
              << optimized_vs_global_max_error << '\n';
    if (!std::isfinite(optimized_max_error) ||
        !std::isfinite(global_max_error) ||
        !std::isfinite(optimized_vs_global_max_error) ||
        optimized_vs_global_max_error > 5.0e-3 ||
        optimized_max_error > 5.0e-2)
    {
        throw std::runtime_error(
            "GPU ResNet20 level-aware Bootstrap correctness check failed");
    }
}

void run_bootstrap_single_check(const shared_gpu::GpuConfig &config)
{
    print_gpu_memory_snapshot("before_runtime");
    shared_gpu::GpuCkksRuntime runtime(config);
    runtime.synchronize();
    print_gpu_memory_snapshot("after_runtime");

    runtime.initialize_bootstrap();
    runtime.synchronize();
    print_gpu_memory_snapshot("after_bootstrap_init");

    std::vector<double> input(runtime.slot_count());
    for (std::size_t slot = 0; slot < input.size(); ++slot)
    {
        input[slot] =
            static_cast<double>(static_cast<int>(slot % 257) - 128) /
            2048.0;
    }
    const auto encrypted = runtime.encrypt(input);
    runtime.synchronize();

    const auto start = std::chrono::steady_clock::now();
    const auto result = runtime.bootstrap(encrypted);
    runtime.synchronize();
    const auto elapsed = std::chrono::steady_clock::now() - start;
    print_gpu_memory_snapshot("after_single_bootstrap");

    const auto actual = runtime.decrypt(result);
    double max_error = 0.0;
    for (std::size_t slot = 0; slot < input.size(); ++slot)
    {
        max_error = std::max(
            max_error,
            std::abs(actual[slot] - std::complex<double>(input[slot], 0.0)));
    }
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    std::cout << "GPU ResNet20 Bootstrap single elapsed_ms="
              << static_cast<double>(elapsed_us) / 1000.0
              << " max_error=" << max_error << '\n';
    if (!std::isfinite(max_error) || max_error > 5.0e-2)
    {
        throw std::runtime_error(
            "GPU ResNet20 single Bootstrap correctness check failed");
    }
}

void run_bootstrap_digit_batch_check(
    const shared_gpu::GpuConfig &config,
    std::size_t iteration_count)
{
    if (iteration_count == 0 || config.degree() != 65536 ||
        config.log_p.size() != 9 || config.dnum != 4)
    {
        throw std::invalid_argument(
            "bootstrap digit-batch check requires iterations > 0 and N65536/P9/dnum4");
    }

    shared_gpu::GpuCkksRuntime runtime(config);
    runtime.initialize_bootstrap();
    std::vector<double> input(runtime.slot_count());
    for (std::size_t slot = 0; slot < input.size(); ++slot)
    {
        input[slot] =
            static_cast<double>(static_cast<int>(slot % 257) - 128) /
            2048.0;
    }
    const auto encrypted = runtime.encrypt(input);
    runtime.synchronize();

    const auto execute =
        [&](bool digit_batched)
        {
            setenv(
                "POSEIDON_DOUBLE_HOIST_P9_DIGIT_BATCHED",
                digit_batched ? "1" : "0",
                1);
            const auto start = std::chrono::steady_clock::now();
            const auto output = runtime.bootstrap(encrypted);
            runtime.synchronize();
            const auto elapsed = std::chrono::steady_clock::now() - start;
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(elapsed).count();
            return std::make_pair(elapsed_ms, runtime.decrypt(output));
        };

    // Populate both workspace variants before entering the measured loop.
    (void)execute(false);
    (void)execute(true);

    double baseline_total_ms = 0.0;
    double batched_total_ms = 0.0;
    std::vector<std::complex<double>> baseline_slots;
    std::vector<std::complex<double>> batched_slots;
    for (std::size_t iteration = 0; iteration < iteration_count; ++iteration)
    {
        auto baseline = execute(false);
        auto batched = execute(true);
        baseline_total_ms += baseline.first;
        batched_total_ms += batched.first;
        baseline_slots = std::move(baseline.second);
        batched_slots = std::move(batched.second);
    }

    double baseline_max_error = 0.0;
    double batched_max_error = 0.0;
    double cross_max_error = 0.0;
    for (std::size_t slot = 0; slot < input.size(); ++slot)
    {
        const std::complex<double> expected(input[slot], 0.0);
        baseline_max_error = std::max(
            baseline_max_error,
            std::abs(baseline_slots[slot] - expected));
        batched_max_error = std::max(
            batched_max_error,
            std::abs(batched_slots[slot] - expected));
        cross_max_error = std::max(
            cross_max_error,
            std::abs(batched_slots[slot] - baseline_slots[slot]));
    }
    const double baseline_average_ms =
        baseline_total_ms / static_cast<double>(iteration_count);
    const double batched_average_ms =
        batched_total_ms / static_cast<double>(iteration_count);
    const double speedup = baseline_average_ms / batched_average_ms;
    std::cout << "GPU ResNet20 Bootstrap digit-batch A/B iterations="
              << iteration_count
              << " baseline_avg_ms=" << baseline_average_ms
              << " batched_avg_ms=" << batched_average_ms
              << " speedup=" << speedup
              << " baseline_max_error=" << baseline_max_error
              << " batched_max_error=" << batched_max_error << '\n';
    std::cout << "GPU ResNet20 Bootstrap digit-batch cross_max_error="
              << cross_max_error << '\n';
    if (!std::isfinite(baseline_max_error) ||
        !std::isfinite(batched_max_error) ||
        !std::isfinite(cross_max_error) ||
        baseline_max_error > 5.0e-2 || batched_max_error > 5.0e-2 ||
        cross_max_error > 5.0e-3)
    {
        throw std::runtime_error(
            "GPU ResNet20 Bootstrap digit-batch A/B correctness check failed");
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
        else if (argc == 2 && std::string(argv[1]) == "--fixed-dnum-check")
        {
            run_fixed_dnum_check(config);
        }
        else if (argc == 2 && std::string(argv[1]) == "--bootstrap-check")
        {
            run_bootstrap_check(config);
        }
        else if (argc == 2 &&
                 std::string(argv[1]) == "--bootstrap-single-check")
        {
            run_bootstrap_single_check(config);
        }
        else if (argc == 3 &&
                 std::string(argv[1]) == "--bootstrap-digit-batch-check")
        {
            run_bootstrap_digit_batch_check(
                config,
                static_cast<std::size_t>(std::stoull(argv[2])));
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
        else if ((argc == 3 || argc == 4) &&
                 std::string(argv[1]) == "--direct-infer")
        {
            const auto start = std::chrono::steady_clock::now();
            const auto image_id = static_cast<std::size_t>(std::stoull(argv[2]));
            const auto max_blocks = argc == 4
                                        ? static_cast<std::size_t>(std::stoull(argv[3]))
                                        : topology.blocks.size();
            const auto weights = resnet20::load_resnet20_weights();
            const auto result = resnet20::run_gpu_resnet20_direct(
                image_id, config, topology, weights, max_blocks);
            std::cout << "GPU ResNet20 direct result image=" << result.image_id
                      << " completed_blocks=" << result.completed_blocks
                      << " true_label=" << result.true_label
                      << " predicted_label=" << result.predicted_label;
            if (result.completed_blocks == topology.blocks.size())
            {
                std::cout << " max_logit_error=" << result.max_logit_error;
            }
            std::cout << '\n';
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
                "[--smoke|--shortcut-check|--hoist-check|--fixed-dnum-check|"
                "--bootstrap-check|"
                "--bootstrap-single-check|"
                "--bootstrap-digit-batch-check ITERATIONS|"
                "--topology-check|--weights-check|"
                "--head-check IMAGE_ID|--infer IMAGE_ID [MAX_BLOCKS]|"
                "--direct-infer IMAGE_ID [MAX_BLOCKS]|"
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
