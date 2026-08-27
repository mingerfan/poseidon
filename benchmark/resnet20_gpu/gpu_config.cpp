#include "gpu_config.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace poseidon::benchmark::resnet20_gpu::core
{
namespace
{

std::vector<std::uint32_t> verified_bootstrap_tail()
{
    return {
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        28, 28, 31, 31, 32, 32, 30, 31, 32, 31, 32,
        32, 31, 31, 31, 32, 32, 31, 32, 32, 32, 30};
}

std::uint32_t configured_dnum()
{
    const char *raw = std::getenv("POSEIDON_GPU_RESNET20_DNUM");
    if (raw == nullptr)
    {
        return 2;
    }

    try
    {
        const std::string text(raw);
        std::size_t parsed = 0;
        const auto value = std::stoul(text, &parsed, 10);
        if (parsed != text.size() || value == 0 || value > 36)
        {
            throw std::invalid_argument("out of range");
        }
        return static_cast<std::uint32_t>(value);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument(
            "POSEIDON_GPU_RESNET20_DNUM must be an integer in [1, 36]");
    }
}

}  // namespace

std::size_t GpuConfig::degree() const noexcept
{
    return std::size_t{1} << log_n;
}

std::size_t GpuConfig::slot_count() const noexcept
{
    return std::size_t{1} << log_slots;
}

std::size_t GpuConfig::application_q_count() const noexcept
{
    return static_cast<std::size_t>(q0_level + 1) +
           static_cast<std::size_t>(application_levels) *
               physical_primes_per_application_level;
}

std::uint32_t GpuConfig::application_rescale_bits() const noexcept
{
    return application_prime_bits * cipher_product_rescale_primes;
}

std::uint32_t GpuConfig::post_multiply_scale_adjustment_bits() const noexcept
{
    // After multiplying two scale-2^s ciphertexts and removing r bits, the
    // scale is 2^(2s-r). Multiplying by an encoding of 1 and removing one
    // more physical prime b returns to 2^s when its scale is 2^(r+b-s).
    return application_rescale_bits() + application_prime_bits - log_scale;
}

void GpuConfig::validate() const
{
    if (log_n != 16 || log_slots + 1 != log_n)
    {
        throw std::invalid_argument("GPU ResNet20 requires N=65536 and 32768 slots");
    }
    if (log_scale == 0 || log_scale >= 63 ||
        application_log_scale != log_scale)
    {
        throw std::invalid_argument("GPU ResNet20 application scale profile is invalid");
    }
    if (evalmod_log_scale == 0 || evalmod_log_scale >= 63 ||
        bootstrap_output_log_scale == 0 || bootstrap_output_log_scale >= 63 ||
        bootstrap_output_log_scale != application_log_scale)
    {
        throw std::invalid_argument(
            "GPU ResNet20 bootstrap must return to the application scale");
    }
    if (q0_level != 1 || log_q.size() <= q0_level)
    {
        throw std::invalid_argument("GPU ResNet20 requires a two-prime q0 base");
    }
    if (physical_primes_per_application_level !=
            cipher_product_rescale_primes + 1 ||
        application_rescale_bits() >= 2 * application_log_scale ||
        post_multiply_scale_adjustment_bits() == 0 ||
        post_multiply_scale_adjustment_bits() >= 63)
    {
        throw std::invalid_argument(
            "GPU ResNet20 physical application level cannot normalize to its logical scale");
    }
    if (application_q_count() > log_q.size())
    {
        throw std::invalid_argument(
            "GPU ResNet20 Q chain is too short for application plus bootstrap");
    }
    if (bootstrap_q_count <= q0_level + 1 || bootstrap_q_count > log_q.size())
    {
        throw std::invalid_argument("GPU ResNet20 bootstrap Q prefix is invalid");
    }
    if (log_p.empty() || (log_q.size() + log_p.size() - 1) / log_p.size() != dnum)
    {
        throw std::invalid_argument("GPU ResNet20 Q/P chains do not realize configured dnum");
    }
    const auto invalid_q = std::find_if(log_q.begin(), log_q.end(), [](std::uint32_t bits) {
        return bits == 0 || bits > 32;
    });
    const auto invalid_p = std::find_if(log_p.begin(), log_p.end(), [](std::uint32_t bits) {
        return bits == 0 || bits > 32;
    });
    if (invalid_q != log_q.end() || invalid_p != log_p.end())
    {
        throw std::invalid_argument("GPU ResNet20 physical primes must fit uint32 residues");
    }
}

GpuConfig make_gpu_config()
{
    GpuConfig config;
    config.dnum = configured_dnum();

    // q[0..1] is the centered q0 base. The next 32 entries reproduce the
    // verified Q34 high-precision bootstrap chain. Application-only 32-bit
    // primes extend that prefix to Q36. Bootstrap raises only to Q34 and then
    // basis-extends its refreshed result back to Q36.
    config.log_q = {32, 32};
    const auto tail = verified_bootstrap_tail();
    config.log_q.insert(config.log_q.end(), tail.begin(), tail.end());
    config.log_q.insert(
        config.log_q.end(),
        config.application_q_count() - config.log_q.size(),
        config.application_prime_bits);

    const std::size_t p_count =
        (config.log_q.size() + config.dnum - 1) / config.dnum;
    // With the shorter Q36 chain, dnum=2 retains an 18-prime P basis so the
    // many rotations in packed convolution do not lose precision at 2^40.
    config.log_p.assign(p_count, 32);
    config.validate();
    return config;
}

}  // namespace poseidon::benchmark::resnet20_gpu::core
