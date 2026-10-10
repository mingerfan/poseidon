#pragma once

#include "runtime/plaintext_bundle.hpp"

#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

inline fhegpu::BundleReadOptions runtime_bundle_options_from_env()
{
    constexpr const char *name = "POSEIDON_RUNTIME_BUNDLE_RESIDENT_BYTES";
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return {};
    }
    const std::string_view text(value);
    fhegpu::BundleReadOptions options;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), options.resident_byte_limit);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    {
        throw std::runtime_error(
            "POSEIDON_RUNTIME_BUNDLE_RESIDENT_BYTES must be a nonnegative integer");
    }
    return options;
}
