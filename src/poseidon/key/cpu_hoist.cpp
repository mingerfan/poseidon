#include "keyswitch_hybrid.h"

#include <algorithm>
#include <stdexcept>

namespace poseidon
{
namespace
{
using namespace util;

void hoist_decompose_modup_ntt(const PoseidonContext &context, const Ciphertext &source,
                               CpuHoistWorkspace &workspace, MemoryPoolHandle pool)
{
    const auto data = context.crt_context()->get_context_data(source.parms_id());
    const auto *rns = data->qp_rns_tool();
    const std::size_t n = data->parms().degree();
    const auto q = rns->base_q()->size();
    const auto p = rns->base_p()->size();
    const auto p_offset = context.crt_context()->key_context_data()->parms().q().size();
    const auto tables = context.crt_context()->small_ntt_tables();
    auto &hoisted = workspace.source_hoist;
    hoisted.parms_id = source.parms_id();
    hoisted.degree = n;
    hoisted.q_count = q;
    hoisted.p_count = p;
    hoisted.dnum = (q + p - 1) / p;
    hoisted.digits_q_ntt.resize(mul_safe(hoisted.dnum, q, n));
    hoisted.digits_p_ntt.resize(mul_safe(hoisted.dnum, p, n));
    workspace.source_coeff_q.assign(source.data(1), source.data(1) + q * n);
    for (std::size_t limb = 0; limb < q; ++limb)
    {
        inverse_ntt_negacyclic_harvey(workspace.source_coeff_q.data() + limb * n, tables[limb]);
    }

    for (std::size_t digit = 0; digit < hoisted.dnum; ++digit)
    {
        RNSIter digit_q(hoisted.digits_q_ntt.data() + digit * q * n, n);
        RNSIter digit_p(hoisted.digits_p_ntt.data() + digit * p * n, n);
        const auto begin = digit * p;
        const auto end = std::min(begin + p, q);
        rns->mod_up_copy_q(ConstRNSIter(workspace.source_coeff_q.data(), n), digit, digit_q, pool);
        if (end - begin == 1)
        {
            rns->mod_up_from_one_base_q(digit_q, digit, pool);
            rns->mod_up_from_one_base_p(digit_q, digit, digit_p, pool);
        }
        else
        {
            rns->mod_up_base_q(digit_q, digit, pool);
            rns->mod_up_base_p(digit_q, digit, digit_p, pool);
        }
        for (std::size_t limb = 0; limb < q; ++limb)
        {
            if (limb >= begin && limb < end)
            {
                // These residues already exist in source NTT form.
                set_uint(source.data(1) + limb * n, n, digit_q[limb]);
            }
            else
            {
                ntt_negacyclic_harvey(digit_q[limb], tables[limb]);
            }
        }
        for (std::size_t limb = 0; limb < p; ++limb)
        {
            ntt_negacyclic_harvey(digit_p[limb], tables[p_offset + limb]);
        }
    }
    ++workspace.last_counts.source_decompose_count;
}

void validate_direct_key(const GaloisKeys &keys, std::uint32_t elt, std::size_t n, std::size_t q,
                         std::size_t p, std::size_t key_q_count)
{
    const auto &digits = keys.key(elt);
    const auto dnum = (q + p - 1) / p;
    if (digits.size() < dnum)
    {
        throw std::invalid_argument("hoisted rotation: too few key digits");
    }
    for (std::size_t digit = 0; digit < dnum; ++digit)
    {
        const auto &key = digits[digit].data();
        if (key.size() != 2 || key.poly_modulus_degree() != n ||
            key.coeff_modulus_size() < key_q_count + p || !key.is_ntt_form())
        {
            throw std::invalid_argument("hoisted rotation: incompatible key shape");
        }
    }
}

void keyswitch_multsum_no_moddown(const PoseidonContext &context, const GaloisKeys &keys,
                                  std::uint32_t elt, CpuHoistWorkspace &workspace)
{
    const auto &hoisted = workspace.source_hoist;
    const auto n = hoisted.degree;
    const auto q = hoisted.q_count;
    const auto p = hoisted.p_count;
    const auto data = context.crt_context()->get_context_data(hoisted.parms_id);
    const auto &moduli_q = data->parms().q();
    const auto &moduli_p = context.crt_context()->key_context_data()->parms().p();
    // Evaluation keys retain their full Q prefix even at a lower source level.
    const auto p_offset = context.crt_context()->key_context_data()->parms().q().size();
    const auto galois = context.crt_context()->galois_tool();
    const auto &key_digits = keys.key(elt);
    workspace.permuted_q.resize(q * n);
    workspace.permuted_p.resize(p * n);
    workspace.accumulator_q.resize(2 * q * n);
    workspace.accumulator_p.resize(2 * p * n);
    workspace.product.resize(n);

    for (std::size_t digit = 0; digit < hoisted.dnum; ++digit)
    {
        galois->apply_galois_ntt(ConstRNSIter(hoisted.digits_q_ntt.data() + digit * q * n, n), q,
                                 elt, RNSIter(workspace.permuted_q.data(), n));
        galois->apply_galois_ntt(ConstRNSIter(hoisted.digits_p_ntt.data() + digit * p * n, n), p,
                                 elt, RNSIter(workspace.permuted_p.data(), n));
        for (std::size_t component = 0; component < 2; ++component)
        {
            const auto *key = key_digits[digit].data().data(component);
            for (std::size_t limb = 0; limb < q; ++limb)
            {
                auto *acc = workspace.accumulator_q.data() + (component * q + limb) * n;
                auto *product = digit == 0 ? acc : workspace.product.data();
                dyadic_product_coeffmod(workspace.permuted_q.data() + limb * n, key + limb * n, n,
                                        moduli_q[limb], product);
                if (digit != 0)
                {
                    add_poly_coeffmod(acc, product, n, moduli_q[limb], acc);
                }
            }
            for (std::size_t limb = 0; limb < p; ++limb)
            {
                auto *acc = workspace.accumulator_p.data() + (component * p + limb) * n;
                auto *product = digit == 0 ? acc : workspace.product.data();
                dyadic_product_coeffmod(workspace.permuted_p.data() + limb * n,
                                        key + (p_offset + limb) * n, n, moduli_p[limb], product);
                if (digit != 0)
                {
                    add_poly_coeffmod(acc, product, n, moduli_p[limb], acc);
                }
            }
        }
    }
    ++workspace.last_counts.key_switch_count;
}

void moddown_qp_ciphertext_to_q(const PoseidonContext &context, CpuHoistWorkspace &workspace,
                                Ciphertext &destination, MemoryPoolHandle pool)
{
    const auto &hoisted = workspace.source_hoist;
    const auto n = hoisted.degree;
    const auto q = hoisted.q_count;
    const auto p = hoisted.p_count;
    const auto data = context.crt_context()->get_context_data(hoisted.parms_id);
    const auto p_offset = context.crt_context()->key_context_data()->parms().q().size();
    const auto tables = context.crt_context()->small_ntt_tables();
    destination.resize(context, hoisted.parms_id, 2);
    for (std::size_t component = 0; component < 2; ++component)
    {
        auto *acc_q = workspace.accumulator_q.data() + component * q * n;
        auto *acc_p = workspace.accumulator_p.data() + component * p * n;
        for (std::size_t limb = 0; limb < q; ++limb)
        {
            inverse_ntt_negacyclic_harvey(acc_q + limb * n, tables[limb]);
        }
        for (std::size_t limb = 0; limb < p; ++limb)
        {
            inverse_ntt_negacyclic_harvey(acc_p + limb * n, tables[p_offset + limb]);
        }
        data->qp_rns_tool()->mod_down(ConstRNSIter(acc_q, n), ConstRNSIter(acc_p, n),
                                      RNSIter(destination.data(component), n), pool);
        for (std::size_t limb = 0; limb < q; ++limb)
        {
            ntt_negacyclic_harvey(destination.data(component) + limb * n, tables[limb]);
        }
    }
    destination.is_ntt_form() = true;
    ++workspace.last_counts.moddown_count;
}
} // namespace

void KSwitchBase::rotate_many_hoisted(const Ciphertext &encrypted, const std::vector<int> &steps,
                                      const GaloisKeys &galois_keys, CpuHoistWorkspace &workspace,
                                      std::vector<Ciphertext> &destination) const
{
    workspace.last_counts = {};
    std::vector<Ciphertext> result(steps.size());
    for (std::size_t i = 0; i < steps.size(); ++i)
    {
        if (steps[i] == 0)
        {
            result[i] = encrypted;
        }
        else
        {
            rotate(encrypted, result[i], steps[i], galois_keys);
            ++workspace.last_counts.fallback_rotation_count;
        }
    }
    // Commit only after the batch finishes; encrypted may refer into destination.
    destination = std::move(result);
}

void KSwitchHybrid::rotate_many_hoisted(const Ciphertext &encrypted, const std::vector<int> &steps,
                                        const GaloisKeys &galois_keys, CpuHoistWorkspace &workspace,
                                        std::vector<Ciphertext> &destination) const
{
    workspace.last_counts = {};
    if (steps.empty())
    {
        destination.clear();
        return;
    }
    const auto data = context_.crt_context()->get_context_data(encrypted.parms_id());
    if (!data || data->parms().scheme() != CKKS || !encrypted.is_ntt_form() ||
        encrypted.size() != 2 || encrypted.poly_modulus_degree() != data->parms().degree() ||
        encrypted.coeff_modulus_size() != data->parms().q().size())
    {
        throw std::invalid_argument(
            "hoisted rotation requires a size-2 Q-basis CKKS NTT ciphertext");
    }
    const auto n = data->parms().degree();
    const auto q = data->parms().q().size();
    const auto p = data->qp_rns_tool()->base_p()->size();
    const auto key_q_count = context_.crt_context()->key_context_data()->parms().q().size();
    const auto galois = context_.crt_context()->galois_tool();
    std::vector<std::uint32_t> direct_elts(steps.size(), 0);
    bool need_hoist = false;
    for (std::size_t i = 0; i < steps.size(); ++i)
    {
        if (steps[i] == 0)
            continue;
        const auto slots = static_cast<std::int64_t>(n / 2);
        if (static_cast<std::int64_t>(steps[i]) <= -slots ||
            static_cast<std::int64_t>(steps[i]) >= slots)
        {
            throw std::invalid_argument("hoisted rotation: step count too large");
        }
        if (galois_keys.parms_id() != context_.crt_context()->key_parms_id())
        {
            throw std::invalid_argument("hoisted rotation: incompatible Galois keys");
        }
        const auto elt = galois->get_elt_from_step(steps[i]);
        if (galois_keys.has_key(elt))
        {
            validate_direct_key(galois_keys, elt, n, q, p, key_q_count);
            direct_elts[i] = elt;
            need_hoist = true;
        }
    }
    if (need_hoist)
    {
        hoist_decompose_modup_ntt(context_, encrypted, workspace, pool_);
    }
    std::vector<Ciphertext> result(steps.size());
    for (std::size_t i = 0; i < steps.size(); ++i)
    {
        if (steps[i] == 0)
        {
            result[i] = encrypted;
        }
        else if (direct_elts[i] == 0)
        {
            // A composed NAF rotation changes c1 at each step and cannot reuse
            // the original source decomposition across its intermediate values.
            rotate(encrypted, result[i], steps[i], galois_keys);
            ++workspace.last_counts.fallback_rotation_count;
        }
        else
        {
            keyswitch_multsum_no_moddown(context_, galois_keys, direct_elts[i], workspace);
            moddown_qp_ciphertext_to_q(context_, workspace, result[i], pool_);
            galois->apply_galois_ntt(ConstRNSIter(encrypted.data(0), n), q, direct_elts[i],
                                     RNSIter(workspace.permuted_q.data(), n));
            for (std::size_t limb = 0; limb < q; ++limb)
            {
                auto *c0 = result[i].data(0) + limb * n;
                add_poly_coeffmod(c0, workspace.permuted_q.data() + limb * n, n,
                                  data->parms().q()[limb], c0);
            }
            result[i].scale() = encrypted.scale();
            result[i].correction_factor() = encrypted.correction_factor();
        }
    }
    destination = std::move(result);
}
} // namespace poseidon
