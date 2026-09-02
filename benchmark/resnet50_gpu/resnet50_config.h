#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace poseidon::benchmark::resnet50_gpu
{

struct ResNet50GpuConfig
{
    std::uint32_t log_n = 16;
    std::uint32_t log_slots = 15;
    // Two physical uint32 GPU primes implement one logical application level.
    // Application arithmetic uses 2^40 while EvalMod uses the higher 2^45
    // bootstrap scale and returns directly to the 2^40 application scale.
    std::uint32_t log_scale = 40;
    std::uint32_t application_log_scale = 40;
    std::uint32_t evalmod_log_scale = 45;
    std::uint32_t bootstrap_output_log_scale = 40;
    std::uint32_t bootstrap_q_count = 34;
    std::uint32_t q0_level = 1;
    std::uint32_t message_ratio = 32;
    std::uint32_t boundary_k = 25;
    std::uint32_t evalmod_degree = 59;
    std::uint32_t double_angle = 2;
    std::uint32_t dnum = 2;
    // Seventeen 2-prime physical groups cover the longest pre-bootstrap path:
    // stem multiply (1), degree-[15,15,27] ReLU (28), average pool (2) and
    // the next convolution (2), with one physical Q prime above q0.
    std::uint32_t application_levels = 17;
    std::uint32_t application_prime_bits = 32;
    std::uint32_t cipher_product_rescale_primes = 1;
    std::uint32_t physical_primes_per_application_level = 2;
    std::vector<std::uint32_t> log_q;
    std::vector<std::uint32_t> log_p;

    std::size_t degree() const noexcept;
    std::size_t slot_count() const noexcept;
    std::size_t application_q_count() const noexcept;
    std::uint32_t application_rescale_bits() const noexcept;
    std::uint32_t post_multiply_scale_adjustment_bits() const noexcept;
    void validate() const;
};

ResNet50GpuConfig make_resnet50_gpu_config();

}  // namespace poseidon::benchmark::resnet50_gpu
