#include "poseidon/gpu/gpu_parameter.h"

#include "poseidon/basics/util/ntt.h"
#include "poseidon/poseidon_context.h"

#include <cstdint>
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

template <typename T>
std::vector<T> concatenate_vectors(
    const std::vector<T> &first,
    const std::vector<T> &second)
{
    std::vector<T> result;
    result.reserve(first.size() + second.size());
    result.insert(result.end(), first.begin(), first.end());
    result.insert(result.end(), second.begin(), second.end());
    return result;
}

std::vector<GpuWord> copy_ntt_root_operands(
    const util::NTTTables *ntt_tables,
    std::size_t limb_count,
    std::size_t degree,
    bool inverse)
{
    if (limb_count != 0 && ntt_tables == nullptr)
    {
        throw std::invalid_argument("GpuParameterData requires CPU NTT tables");
    }

    std::vector<GpuWord> result(limb_count * degree);
    for (std::size_t limb = 0; limb < limb_count; ++limb)
    {
        const auto &table = ntt_tables[limb];
        if (table.coeff_count() != degree)
        {
            throw std::invalid_argument("GpuParameterData NTT table degree mismatch");
        }

        const auto *roots = inverse
            ? table.get_from_inv_root_powers()
            : table.get_from_root_powers();

        for (std::size_t i = 0; i < degree; ++i)
        {
            result[limb * degree + i] = checked_gpu_word(
                roots[i].operand,
                inverse
                    ? "GpuParameterData only supports inverse NTT roots that fit in GpuWord"
                    : "GpuParameterData only supports NTT roots that fit in GpuWord");
        }
    }
    return result;
}

std::vector<GpuWord> copy_inv_degree_operands(
    const util::NTTTables *ntt_tables,
    std::size_t limb_count)
{
    if (limb_count != 0 && ntt_tables == nullptr)
    {
        throw std::invalid_argument("GpuParameterData requires CPU NTT tables");
    }

    std::vector<GpuWord> result(limb_count);
    for (std::size_t limb = 0; limb < limb_count; ++limb)
    {
        result[limb] = checked_gpu_word(
            ntt_tables[limb].inv_degree_modulo().operand,
            "GpuParameterData only supports inverse degree constants that fit in GpuWord");
    }
    return result;
}

std::vector<GpuWord> copy_inv_q_last_mod_q_operands(
    const util::RNSTool *rns_tool,
    std::size_t q_count)
{
    if (q_count < 2)
    {
        return {};
    }
    if (rns_tool == nullptr || rns_tool->inv_q_last_mod_q() == nullptr)
    {
        throw std::invalid_argument("GpuParameterData requires RNSTool rescale constants");
    }

    std::vector<GpuWord> result(q_count - 1);
    const auto *inv_q_last_mod_q = rns_tool->inv_q_last_mod_q();
    for (std::size_t i = 0; i < q_count - 1; ++i)
    {
        result[i] = checked_gpu_word(
            inv_q_last_mod_q[i].operand,
            "GpuParameterData only supports inv_q_last_mod_q constants that fit in GpuWord");
    }
    return result;
}

std::vector<GpuWord> compute_half_q_last_mod_q(
    const std::vector<Modulus> &q,
    GpuWord half_q_last)
{
    if (q.size() < 2)
    {
        return {};
    }

    std::vector<GpuWord> result(q.size() - 1);
    for (std::size_t i = 0; i < q.size() - 1; ++i)
    {
        result[i] = checked_gpu_word(
            static_cast<std::uint64_t>(half_q_last) % q[i].value(),
            "GpuParameterData only supports half_q_last_mod_q constants that fit in GpuWord");
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
    const auto *small_ntt_tables = crt_context->small_ntt_tables();
    const std::size_t p_ntt_table_offset =
        context.parameters_literal()->q().size();

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
        if (!q.empty())
        {
            shard.q_last = checked_gpu_word(
                q.back().value(),
                "GpuParameterData only supports q_last that fits in GpuWord");
            shard.half_q_last = static_cast<GpuWord>(shard.q_last >> 1);
        }

        auto q_words = copy_moduli_to_gpu_words(
            q,
            "GpuParameterData only supports q primes that fit in GpuWord");
        auto p_words = copy_moduli_to_gpu_words(
            p,
            "GpuParameterData only supports p primes that fit in GpuWord");
        auto q_barrett_ratios = copy_barrett_ratios(q);
        auto p_barrett_ratios = copy_barrett_ratios(p);
        auto rns_words = concatenate_vectors(q_words, p_words);
        auto rns_barrett_ratios =
            concatenate_vectors(q_barrett_ratios, p_barrett_ratios);

        const auto *p_ntt_tables = small_ntt_tables + p_ntt_table_offset;
        auto q_ntt_roots = copy_ntt_root_operands(
            small_ntt_tables,
            q.size(),
            level.degree,
            false);
        auto p_ntt_roots = copy_ntt_root_operands(
            p_ntt_tables,
            p.size(),
            level.degree,
            false);
        auto q_intt_roots = copy_ntt_root_operands(
            small_ntt_tables,
            q.size(),
            level.degree,
            true);
        auto p_intt_roots = copy_ntt_root_operands(
            p_ntt_tables,
            p.size(),
            level.degree,
            true);
        auto q_inv_degree = copy_inv_degree_operands(
            small_ntt_tables,
            q.size());
        auto p_inv_degree = copy_inv_degree_operands(
            p_ntt_tables,
            p.size());
        auto rns_ntt_roots =
            concatenate_vectors(q_ntt_roots, p_ntt_roots);
        auto rns_intt_roots =
            concatenate_vectors(q_intt_roots, p_intt_roots);
        auto rns_inv_degree =
            concatenate_vectors(q_inv_degree, p_inv_degree);
        auto half_q_last_mod_q =
            compute_half_q_last_mod_q(q, shard.half_q_last);
        auto inv_q_last_mod_q = copy_inv_q_last_mod_q_operands(
            context_data->rns_tool(),
            q.size());

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

        shard.rns_primes = DeviceVector<GpuWord>(rns_words.size(), device_id);
        if (!rns_words.empty())
        {
            shard.rns_primes.copy_from_host(
                rns_words.data(),
                rns_words.size());
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

        shard.rns_modulus_constants =
            DeviceVector<GpuWide>(rns_barrett_ratios.size(), device_id);
        if (!rns_barrett_ratios.empty())
        {
            shard.rns_modulus_constants.copy_from_host(
                rns_barrett_ratios.data(),
                rns_barrett_ratios.size());
        }

        shard.half_q_last_mod_q =
            DeviceVector<GpuWord>(half_q_last_mod_q.size(), device_id);
        if (!half_q_last_mod_q.empty())
        {
            shard.half_q_last_mod_q.copy_from_host(
                half_q_last_mod_q.data(),
                half_q_last_mod_q.size());
        }

        shard.inv_q_last_mod_q =
            DeviceVector<GpuWord>(inv_q_last_mod_q.size(), device_id);
        if (!inv_q_last_mod_q.empty())
        {
            shard.inv_q_last_mod_q.copy_from_host(
                inv_q_last_mod_q.data(),
                inv_q_last_mod_q.size());
        }

        shard.ntt_tables =
            DeviceVector<GpuWord>(rns_ntt_roots.size(), device_id);
        if (!rns_ntt_roots.empty())
        {
            shard.ntt_tables.copy_from_host(
                rns_ntt_roots.data(),
                rns_ntt_roots.size());
        }

        shard.intt_tables =
            DeviceVector<GpuWord>(rns_intt_roots.size(), device_id);
        if (!rns_intt_roots.empty())
        {
            shard.intt_tables.copy_from_host(
                rns_intt_roots.data(),
                rns_intt_roots.size());
        }

        shard.inv_degree_modulo =
            DeviceVector<GpuWord>(rns_inv_degree.size(), device_id);
        if (!rns_inv_degree.empty())
        {
            shard.inv_degree_modulo.copy_from_host(
                rns_inv_degree.data(),
                rns_inv_degree.size());
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

const GpuLevelInfo &GpuParameterData::get_next_level(const parms_id_type &parms_id) const
{
    for (std::size_t i = 0; i < levels_.size(); ++i)
    {
        if (levels_[i].parms_id == parms_id)
        {
            if (i + 1 >= levels_.size())
            {
                throw std::out_of_range("GpuParameterData next level not found for parms_id");
            }
            return levels_[i + 1];
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
