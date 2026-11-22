#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace llama_manual {

// One real CKKS slot holds one feature of one token. No implicit repetition.
// slot(feature, token) = feature * token_capacity + token.
struct Layout {
    std::size_t token_capacity = 128;
    std::size_t feature_tile = 256;
    std::size_t slots = 32768;
    std::size_t baby_steps = 16;

    void validate() const {
        if (!token_capacity || !feature_tile || !baby_steps ||
            feature_tile > std::numeric_limits<std::size_t>::max() / token_capacity ||
            token_capacity * feature_tile != slots ||
            feature_tile % baby_steps != 0 ||
            slots > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("invalid full-slot feature-major BSGS layout");
        }
    }
    std::size_t tiles(std::size_t width) const {
        validate();
        return width / feature_tile + (width % feature_tile != 0);
    }
};

inline std::vector<std::vector<double>> pack(
    const Layout &layout, const std::vector<double> &token_major,
    std::size_t tokens, std::size_t width) {
    layout.validate();
    if (!width || !tokens || tokens > layout.token_capacity ||
        width > std::numeric_limits<std::size_t>::max() / tokens ||
        token_major.size() != tokens * width) {
        throw std::invalid_argument("input shape does not match packing layout");
    }
    std::vector<std::vector<double>> result(layout.tiles(width),
                                            std::vector<double>(layout.slots, 0.0));
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t f = 0; f < width; ++f) {
            const double value = token_major[t * width + f];
            if (!std::isfinite(value)) throw std::invalid_argument("non-finite input");
            result[f / layout.feature_tile][(f % layout.feature_tile) *
                                           layout.token_capacity + t] = value;
        }
    }
    return result;
}

inline std::vector<double> unpack(
    const Layout &layout, const std::vector<std::vector<double>> &packed,
    std::size_t tokens, std::size_t width) {
    layout.validate();
    if (!width || !tokens || tokens > layout.token_capacity ||
        width > std::numeric_limits<std::size_t>::max() / tokens ||
        packed.size() != layout.tiles(width)) {
        throw std::invalid_argument("invalid unpack shape");
    }
    for (const auto &tile : packed) {
        if (tile.size() != layout.slots) throw std::invalid_argument("wrong slot count");
    }
    std::vector<double> result(tokens * width);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t f = 0; f < width; ++f) {
            result[t * width + f] = packed[f / layout.feature_tile]
                [(f % layout.feature_tile) * layout.token_capacity + t];
        }
    }
    return result;
}

} // namespace llama_manual
