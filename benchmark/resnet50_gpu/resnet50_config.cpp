#include "resnet50_config.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace poseidon::benchmark::resnet50_gpu
{
namespace
{

std::vector<std::uint64_t> s2c_first_q_moduli()
{
    return {
        4255252481ULL, 4255645697ULL, 4258922497ULL, 4260364289ULL, 4261675009ULL,
        4264427521ULL, 4265476097ULL, 4266393601ULL, 4267442177ULL, 1040842753ULL,
        1042415617ULL, 1043464193ULL, 1045430273ULL, 1048707073ULL, 1049100289ULL,
        1051721729ULL, 1052508161ULL, 1053818881ULL, 1054212097ULL, 1055260673ULL,
        1056178177ULL, 1056440321ULL, 1060765697ULL, 1062469633ULL, 1062862849ULL,
        1064697857ULL, 1065484289ULL, 1068236801ULL, 1070727169ULL, 1071513601ULL,
        1073479681ULL, 2135162881ULL, 4272291841ULL, 4272685057ULL, 1040056321ULL,
        2135818241ULL, 4273340417ULL, 2142502913ULL, 4274126849ULL, 4276092929ULL,
        2144468993ULL, 2146041857ULL, 2146959361ULL, 4277403649ULL, 4279369729ULL,
        2147352577ULL, 4280025089ULL, 4280156161ULL, 4281204737ULL, 1038745601ULL};
}

std::vector<std::uint64_t> s2c_first_p_moduli()
{
    return {
        4293918721ULL, 4291952641ULL, 4289462273ULL, 4288806913ULL, 4286709761ULL,
        4286054401ULL, 4284874753ULL, 4284088321ULL, 4283301889ULL, 4271505409ULL,
        4271374337ULL, 4269015041ULL, 4253024257ULL, 4252631041ULL, 4252106753ULL,
        4251844609ULL, 4247388161ULL, 4245553153ULL, 4245422081ULL, 4243980289ULL,
        4241883137ULL, 4241620993ULL, 4241489921ULL, 4240310273ULL, 4240048129ULL};
}

std::vector<std::uint32_t> bits_of(const std::vector<std::uint64_t> &moduli)
{
    std::vector<std::uint32_t> result;
    result.reserve(moduli.size());
    for (const auto modulus : moduli)
    {
        std::uint32_t bits = 0;
        while ((std::uint64_t{1} << bits) <= modulus && bits < 63)
        {
            ++bits;
        }
        result.push_back(bits);
    }
    return result;
}

std::uint32_t configured_dnum()
{
    const char *text = std::getenv("POSEIDON_GPU_DNUM");
    if (text == nullptr || *text == '\0')
    {
        return 2;
    }
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 2 || value > 5)
    {
        throw std::invalid_argument(
            "POSEIDON_GPU_DNUM must be one of 2, 3, 4, or 5");
    }
    return static_cast<std::uint32_t>(value);
}

}  // namespace

std::size_t ResNet50GpuConfig::degree() const noexcept
{
    return std::size_t{1} << log_n;
}

std::size_t ResNet50GpuConfig::slot_count() const noexcept
{
    return std::size_t{1} << log_slots;
}

std::size_t ResNet50GpuConfig::application_q_count() const noexcept
{
    return static_cast<std::size_t>(q0_level + 1) +
           static_cast<std::size_t>(application_levels) *
               physical_primes_per_application_level;
}

std::uint32_t ResNet50GpuConfig::application_rescale_bits() const noexcept
{
    return application_prime_bits * cipher_product_rescale_primes;
}

std::uint32_t ResNet50GpuConfig::post_multiply_scale_adjustment_bits() const noexcept
{
    return application_rescale_bits() + application_prime_bits - log_scale;
}

void ResNet50GpuConfig::validate() const
{
    if (log_n != 16 || log_slots + 1 != log_n)
    {
        throw std::invalid_argument("GPU ResNet50 requires N=65536 and 32768 slots");
    }
    if (log_scale == 0 || log_scale >= 63 ||
        application_log_scale != log_scale)
    {
        throw std::invalid_argument("GPU application scale profile is invalid");
    }
    if (evalmod_log_scale == 0 || evalmod_log_scale >= 63 ||
        bootstrap_output_log_scale == 0 || bootstrap_output_log_scale >= 63)
    {
        throw std::invalid_argument("GPU bootstrap scale profile is invalid");
    }
    if (q0_level != 1 || log_q.size() <= q0_level)
    {
        throw std::invalid_argument("GPU ResNet50 requires a two-prime q0 base");
    }
    if (physical_primes_per_application_level !=
            cipher_product_rescale_primes + 1 ||
        application_rescale_bits() >= 2 * application_log_scale ||
        post_multiply_scale_adjustment_bits() == 0 ||
        post_multiply_scale_adjustment_bits() >= 63)
    {
        throw std::invalid_argument(
            "GPU ResNet50 physical application level must preserve its natural scale");
    }
    if (application_q_count() > log_q.size())
    {
        throw std::invalid_argument(
            "GPU ResNet50 Q chain is too short for application plus bootstrap");
    }
    if (bootstrap_q_count <= q0_level + 1 || bootstrap_q_count > log_q.size())
    {
        throw std::invalid_argument("GPU bootstrap Q prefix is invalid");
    }
    if (log_p.empty() || (log_q.size() + log_p.size() - 1) / log_p.size() != dnum)
    {
        throw std::invalid_argument("GPU ResNet50 Q/P chains do not realize configured dnum");
    }
    const auto invalid_q = std::find_if(log_q.begin(), log_q.end(), [](std::uint32_t bits) {
        return bits == 0 || bits > 32;
    });
    const auto invalid_p = std::find_if(log_p.begin(), log_p.end(), [](std::uint32_t bits) {
        return bits == 0 || bits > 32;
    });
    if (invalid_q != log_q.end() || invalid_p != log_p.end())
    {
        throw std::invalid_argument("GPU ResNet50 physical primes must fit uint32 residues");
    }
    if (!q_moduli.empty() || !p_moduli.empty())
    {
        if (q_moduli.size() != log_q.size() || p_moduli.size() != log_p.size())
        {
            throw std::invalid_argument(
                "GPU ResNet50 exact moduli must match the configured Q/P counts");
        }
        const auto invalid_exact_q = std::find_if(
            q_moduli.begin(), q_moduli.end(), [](std::uint64_t value) {
                return value == 0 || value > std::numeric_limits<std::uint32_t>::max();
            });
        const auto invalid_exact_p = std::find_if(
            p_moduli.begin(), p_moduli.end(), [](std::uint64_t value) {
                return value == 0 || value > std::numeric_limits<std::uint32_t>::max();
            });
        if (invalid_exact_q != q_moduli.end() || invalid_exact_p != p_moduli.end())
        {
            throw std::invalid_argument(
                "GPU ResNet50 exact moduli must fit uint32 residues");
        }
    }
}

ResNet50GpuConfig make_resnet50_gpu_config()
{
    ResNet50GpuConfig config;

    config.q_moduli = s2c_first_q_moduli();
    config.dnum = configured_dnum();
    auto p_moduli = s2c_first_p_moduli();
    const std::size_t p_count =
        (config.q_moduli.size() + config.dnum - 1) / config.dnum;
    if (p_count == 0 || p_count > p_moduli.size())
    {
        throw std::invalid_argument(
            "configured dnum exceeds the available P modulus chain");
    }
    config.p_moduli.assign(p_moduli.begin(), p_moduli.begin() + p_count);
    config.log_q = bits_of(config.q_moduli);
    config.log_p = bits_of(config.p_moduli);
    config.bootstrap_q_count = static_cast<std::uint32_t>(config.q_moduli.size());
    config.validate();
    return config;
}

}  // namespace poseidon::benchmark::resnet50_gpu
