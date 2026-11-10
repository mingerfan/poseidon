#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace poseidon::runtime_api
{

inline int normalize_rotation_step(int step, std::size_t slot_count)
{
    if (slot_count < 2 || slot_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument("rotation slot count is unsupported");
    }

    const auto slots = static_cast<std::int64_t>(slot_count);
    std::int64_t normalized = static_cast<std::int64_t>(step) % slots;
    const std::int64_t half = slots / 2;
    if (normalized > half)
    {
        normalized -= slots;
    }
    else if (normalized <= -half)
    {
        normalized += slots;
    }
    return static_cast<int>(normalized);
}

/*
 * How a rotation without a direct key is split into keyed power-of-two
 * rotations. The default is the non-adjacent form (31 = 32 - 1: two key
 * switches instead of five); POSEIDON_RUNTIME_ROTATION_DECOMPOSITION=binary
 * restores the plain binary split. The key basis below follows the same
 * choice, so keys and rotations always agree.
 */
inline bool rotation_decomposition_is_naf()
{
    const char *value = std::getenv("POSEIDON_RUNTIME_ROTATION_DECOMPOSITION");
    if (value == nullptr || value[0] == '\0')
        return true;
    const std::string mode(value);
    if (mode == "naf")
        return true;
    if (mode == "binary")
        return false;
    throw std::invalid_argument("POSEIDON_RUNTIME_ROTATION_DECOMPOSITION must be naf or binary");
}

inline std::vector<int> decompose_rotation_step(int step, std::size_t slot_count, bool naf)
{
    const int normalized = normalize_rotation_step(step, slot_count);
    const int sign = normalized < 0 ? -1 : 1;
    std::uint64_t remaining = static_cast<std::uint64_t>(
        normalized < 0 ? -static_cast<std::int64_t>(normalized) : normalized);

    std::vector<int> result;
    if (naf)
    {
        std::int64_t n = static_cast<std::int64_t>(remaining);
        std::int64_t power = 1;
        while (n != 0)
        {
            if ((n & 1) != 0)
            {
                const std::int64_t digit = (n & 3) == 1 ? 1 : -1;
                const int term = normalize_rotation_step(
                    static_cast<int>(sign * digit * power), slot_count);
                if (term != 0)
                    result.push_back(term);
                n -= digit;
            }
            n >>= 1;
            power <<= 1;
        }
        return result;
    }
    for (std::uint64_t power = 1; remaining != 0; power <<= 1)
    {
        if ((remaining & power) != 0)
        {
            result.push_back(sign * static_cast<int>(power));
            remaining -= power;
        }
    }
    return result;
}

inline std::vector<int> decompose_rotation_step(int step, std::size_t slot_count)
{
    return decompose_rotation_step(step, slot_count, rotation_decomposition_is_naf());
}

/*
 * The keyed rotations a step without a direct key is applied as: the split
 * the key basis uses (NAF by default) when every term has a key, else the
 * plain binary split, which keys made for the binary basis -- e.g. by a
 * caller that generates its own -- still cover. Empty when neither is keyed.
 */
template <class HasKey>
std::vector<int> keyed_rotation_terms(int step, std::size_t slot_count, HasKey &&has_key)
{
    const auto keyed = [&](const std::vector<int> &terms) {
        for (int term : terms)
        {
            if (!has_key(term))
                return false;
        }
        return true;
    };
    auto terms = decompose_rotation_step(step, slot_count);
    if (keyed(terms))
        return terms;
    if (rotation_decomposition_is_naf())
    {
        terms = decompose_rotation_step(step, slot_count, false);
        if (keyed(terms))
            return terms;
    }
    return {};
}

inline std::set<int> binary_rotation_key_basis(const std::set<int> &rotation_steps,
                                               std::size_t slot_count)
{
    std::set<int> result;
    for (int step : rotation_steps)
    {
        const auto terms = decompose_rotation_step(step, slot_count);
        result.insert(terms.begin(), terms.end());
    }
    return result;
}

} // namespace poseidon::runtime_api
