#pragma once
#include "poseidon/poseidon_context.h"
#include "poseidon/key/secretkey.h"
#include <array>
#include <string_view>

namespace poseidon::runtime_api
{
inline constexpr std::array<std::string_view, 4> s2c_bootstrap_profiles{
    "s2c-first-degree59-da2-q50p25", "s2c-first-degree30-da3-q34p9", "s2c-first-degree22-da3-q43-dnum4",
    "s2c-first-degree22-da3-q43-dnum4-real-only"};

// Use the same context for encryption, key generation, and the runtime.
PoseidonContext make_s2c_bootstrap_context(std::string_view profile);
SecretKey make_s2c_bootstrap_secret(const PoseidonContext &context, std::string_view profile);
} // namespace poseidon::runtime_api
