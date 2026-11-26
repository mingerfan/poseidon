#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace poseidon::agent22
{
// Check coefficient-form data across the full RNS base before NTT/key generation.
template <typename Moduli>
void audit_secret(
    const char *role, const std::uint64_t *data, std::size_t degree,
    const Moduli &moduli, std::size_t weight, bool balanced)
{
    if (moduli.empty() || weight == 0 || weight > degree)
        throw std::invalid_argument("secret audit requires an RNS base and 1 <= h <= N");

    std::size_t positive = 0;
    std::size_t negative = 0;
    for (std::size_t i = 0; i < degree; ++i)
    {
        const auto first = data[i];
        const bool minus_one = first == moduli.front().value() - 1;
        if (first > 1 && !minus_one)
            throw std::runtime_error("secret audit detected a non-ternary coefficient");
        positive += first == 1;
        negative += minus_one;
        for (std::size_t j = 1; j < moduli.size(); ++j)
        {
            const auto expected = minus_one ? moduli[j].value() - 1 : first;
            if (data[j * degree + i] != expected)
                throw std::runtime_error("secret audit detected inconsistent RNS limbs");
        }
    }
    if (positive + negative != weight || (balanced && positive != negative))
        throw std::runtime_error("secret audit detected an unexpected weight or distribution");

    std::cout << "SECRET_AUDIT role=" << role << " actual_h=" << positive + negative
              << " rns_limbs=" << moduli.size()
              << " nonzero=PASS ternary=PASS RNS=PASS distribution="
              << (balanced ? "balanced_fixed_weight" : "fixed_weight") << std::endl;
}
} // namespace poseidon::agent22
