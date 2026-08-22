#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace poseidon::benchmark::resnet20_gpu::core
{

struct GpuConfig
{
    std::uint32_t log_n = 16;
    std::uint32_t log_slots = 15;
    // The GPU backend stores RNS residues in uint32, so the CPU 40/45-bit
    // logical profile is represented by several <=32-bit physical primes.
    std::uint32_t log_scale = 40;
    std::uint32_t application_log_scale = 40;
    std::uint32_t evalmod_log_scale = 45;
    // Bootstrap works on the verified Q34 prefix and returns directly to the
    // application scale. These are parameters rather than runtime constants.
    std::uint32_t bootstrap_output_log_scale = 40;
    std::uint32_t bootstrap_q_count = 34;
    std::uint32_t q0_level = 1;
    std::uint32_t message_ratio = 32;
    std::uint32_t boundary_k = 25;
    std::uint32_t evalmod_degree = 59;
    std::uint32_t double_angle = 2;
    std::uint32_t dnum = 2;
    // Seventeen logical application levels cover convolution plus the
    // 14-level degree-[15,15,27] ReLU. At scale 2^40, one 32-bit rescale
    // leaves about 48 scale bits and a second prime normalizes back to 40.
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

GpuConfig make_gpu_config();

}  // namespace poseidon::benchmark::resnet20_gpu::core
