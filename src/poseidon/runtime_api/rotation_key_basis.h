#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
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

inline std::vector<int> decompose_rotation_step(int step, std::size_t slot_count)
{
    const int normalized = normalize_rotation_step(step, slot_count);
    const int sign = normalized < 0 ? -1 : 1;
    std::uint64_t remaining = static_cast<std::uint64_t>(
        normalized < 0 ? -static_cast<std::int64_t>(normalized) : normalized);

    std::vector<int> result;
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
