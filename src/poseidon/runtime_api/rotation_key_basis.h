#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace poseidon::runtime_api
{

// This controls key generation in plan runners, not the GPU/CPU API's
// fallback when an externally supplied key set lacks a direct key.
enum class RotationKeyPolicy { Direct, Binary };

inline RotationKeyPolicy parse_rotation_key_policy(const char *value)
{
    if (value == nullptr || value[0] == '\0' || std::string_view(value) == "direct")
    {
        return RotationKeyPolicy::Direct;
    }
    if (std::string_view(value) == "binary")
    {
        return RotationKeyPolicy::Binary;
    }
    throw std::invalid_argument(
        "POSEIDON_RUNTIME_ROTATION_KEY_POLICY must be direct or binary");
}

inline RotationKeyPolicy rotation_key_policy_from_environment()
{
    return parse_rotation_key_policy(std::getenv("POSEIDON_RUNTIME_ROTATION_KEY_POLICY"));
}

inline const char *rotation_key_policy_name(RotationKeyPolicy policy)
{
    switch (policy)
    {
    case RotationKeyPolicy::Direct: return "direct";
    case RotationKeyPolicy::Binary: return "binary";
    }
    throw std::invalid_argument("unknown rotation key policy");
}

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

inline std::set<int> rotation_key_steps(const std::set<int> &rotation_steps,
                                       std::size_t slot_count,
                                       RotationKeyPolicy policy = RotationKeyPolicy::Direct)
{
    switch (policy)
    {
    case RotationKeyPolicy::Binary:
        return binary_rotation_key_basis(rotation_steps, slot_count);
    case RotationKeyPolicy::Direct:
    {
        std::set<int> result;
        for (int step : rotation_steps)
        {
            const int normalized = normalize_rotation_step(step, slot_count);
            if (normalized != 0)
            {
                result.insert(normalized);
            }
        }
        return result;
    }
    }
    throw std::invalid_argument("unknown rotation key policy");
}

} // namespace poseidon::runtime_api
