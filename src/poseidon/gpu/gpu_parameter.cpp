#include "poseidon/gpu/gpu_parameter.h"

#include "poseidon/poseidon_context.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace poseidon
{
namespace gpu
{
namespace
{

GpuWord checked_gpu_word(std::uint64_t value, const char *what)
{
    if (value > std::numeric_limits<GpuWord>::max())
    {
        throw std::invalid_argument(what);
    }
    return static_cast<GpuWord>(value);
}

std::vector<GpuWord> copy_moduli_to_gpu_words(
    const std::vector<Modulus> &moduli,
    const char *what)
{
    std::vector<GpuWord> result;
    result.reserve(moduli.size());
    for (const auto &modulus : moduli)
    {
        result.push_back(checked_gpu_word(modulus.value(), what));
    }
    return result;
}

GpuWide barrett_ratio_64(const Modulus &modulus)
{
    const auto value = modulus.value();
    if (value == 0)
    {
        throw std::invalid_argument("GpuParameterData cannot build Barrett constant for zero modulus");
    }

    const auto numerator = static_cast<unsigned __int128>(1) << 64;
    return static_cast<GpuWide>(numerator / value);
}

std::vector<GpuWide> copy_barrett_ratios(const std::vector<Modulus> &moduli)
{
    std::vector<GpuWide> result;
    result.reserve(moduli.size());
    for (const auto &modulus : moduli)
    {
        result.push_back(barrett_ratio_64(modulus));
    }
    return result;
}

}  // namespace

GpuParameterData::GpuParameterData(const PoseidonContext &context, int device_id)
{
    build_from_poseidon_context(context, device_id);
}

void GpuParameterData::build_from_poseidon_context(
    const PoseidonContext &context,
    int device_id)
{
    levels_.clear();

    auto crt_context = context.crt_context();
    if (!crt_context)
    {
        throw std::invalid_argument("GpuParameterData requires a valid PoseidonContext");
    }

    auto context_data = crt_context->key_context_data();
    if (!context_data)
    {
        context_data = crt_context->first_context_data();
    }
    while (context_data)
    {
        const auto &parms = context_data->parms();
        const auto &q = parms.q();
        const auto &p = parms.p();

        GpuLevelInfo level;
        level.parms_id = context_data->parms_id();
        level.degree = parms.degree();
        level.q_count = q.size();
        level.p_count = p.size();

        GpuParameterShard shard;
        shard.device_id = device_id;
        shard.limb_begin = 0;
        shard.limb_count = level.q_count + level.p_count;

        auto q_words = copy_moduli_to_gpu_words(
            q,
            "GpuParameterData only supports q primes that fit in GpuWord");
        auto p_words = copy_moduli_to_gpu_words(
            p,
            "GpuParameterData only supports p primes that fit in GpuWord");
        auto q_barrett_ratios = copy_barrett_ratios(q);
        auto p_barrett_ratios = copy_barrett_ratios(p);

        shard.q_primes = DeviceVector<GpuWord>(q_words.size(), device_id);
        if (!q_words.empty())
        {
            shard.q_primes.copy_from_host(q_words.data(), q_words.size());
        }

        shard.p_primes = DeviceVector<GpuWord>(p_words.size(), device_id);
        if (!p_words.empty())
        {
            shard.p_primes.copy_from_host(p_words.data(), p_words.size());
        }

        shard.q_modulus_constants =
            DeviceVector<GpuWide>(q_barrett_ratios.size(), device_id);
        if (!q_barrett_ratios.empty())
        {
            shard.q_modulus_constants.copy_from_host(
                q_barrett_ratios.data(),
                q_barrett_ratios.size());
        }

        shard.p_modulus_constants =
            DeviceVector<GpuWide>(p_barrett_ratios.size(), device_id);
        if (!p_barrett_ratios.empty())
        {
            shard.p_modulus_constants.copy_from_host(
                p_barrett_ratios.data(),
                p_barrett_ratios.size());
        }

        level.shards.push_back(std::move(shard));
        levels_.push_back(std::move(level));

        context_data = context_data->next_context_data();
    }
}

const GpuLevelInfo &GpuParameterData::get_level(const parms_id_type &parms_id) const
{
    for (const auto &level : levels_)
    {
        if (level.parms_id == parms_id)
        {
            return level;
        }
    }

    throw std::out_of_range("GpuParameterData level not found for parms_id");
}

bool GpuParameterData::empty() const
{
    return levels_.empty();
}

}  // namespace gpu
}  // namespace poseidon
