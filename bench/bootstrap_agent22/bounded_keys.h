#pragma once
#include "q43_p22.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace poseidon::agent22 {
inline constexpr double pq_budget_bits = 1782.0;
template <class Profile, class Parameters>
void audit_bounded_keys(const Profile &profile, const Parameters &literal)
{
    double max_bits = 0;
    const auto audit = [&](const auto &key, std::size_t active_q, const char *kind) {
        const auto &meta = key.meta;
        if (key.empty() || meta.q_count < active_q || meta.q_count > literal.q().size() ||
            meta.p_count == 0 || meta.p_count > literal.p().size())
            throw std::runtime_error("invalid GPU evaluation key base");
        const auto decomposition = (meta.q_count + meta.p_count - 1) / meta.p_count;
        if (meta.decomposition_count != decomposition)
            throw std::runtime_error("invalid GPU evaluation key decomposition");
        double bits = 0;
        for (std::size_t i = 0; i < meta.q_count; ++i)
            bits += std::log2(double(literal.q()[i].value()));
        for (std::size_t i = 0; i < meta.p_count; ++i)
            bits += std::log2(double(literal.p()[i].value()));
        if (!(bits < pq_budget_bits))
            throw std::runtime_error("GPU evaluation key exceeds the modulus budget");
        max_bits = std::max(max_bits, bits);
        std::cout << "KEY_BUDGET kind=" << kind << " active_q=" << active_q
                  << " storage_q=" << meta.q_count << " P=" << meta.p_count
                  << " dnum=" << decomposition << " log2_pq=" << bits << '\n';
    };
    const auto audit_table = [&](const auto &keys, const char *kind) {
        if (!keys.empty() || !keys.fields_.empty() || !keys.level_keys || keys.level_keys->empty())
            throw std::runtime_error("bounded Agent22 requires keys at each level, with no global key");
        for (const auto &[q, key] : *keys.level_keys)
        {
            if (!key)
                throw std::runtime_error("missing GPU key for a scheduled level");
            audit(*key, q, kind);
        }
    };
    audit_table(*profile.galois_keys, "rotation");
    audit_table(*profile.relin_keys, "relin");

    const auto &peak = profile.galois_keys->for_level(q_primes.size()).meta;
    const auto &low = profile.bootstrap_data.sparse_switch_to_ephemeral;
    const auto &raised = profile.bootstrap_data.sparse_switch_to_dense;
    if (peak.q_count != q_primes.size() || peak.p_count != 11 || peak.decomposition_count != 4 ||
        !low || low->meta.q_count != 2 || low->meta.p_count != 2 ||
        !raised || raised->meta.q_count != peak.q_count || raised->meta.p_count != peak.p_count)
        throw std::runtime_error("unexpected Agent22 peak or sparse-secret switching modulus");
    audit(*low, 2, "main_to_ephemeral");
    audit(*raised, peak.q_count, "ephemeral_to_main");
    std::cout << "KEY_BUDGET max_log2_pq=" << max_bits
              << " budget_bits=" << pq_budget_bits << " global_key_fallback=absent" << std::endl;
}
} // namespace poseidon::agent22
