#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bert::compact {

inline bool power_of_two(std::size_t n) { return n && !(n & (n - 1)); }
inline unsigned log2_exact(std::size_t n) {
    if (!power_of_two(n)) throw std::invalid_argument("expected power of two");
    unsigned result = 0;
    for (; n > 1; n >>= 1) ++result;
    return result;
}
inline std::uint32_t reverse_bits(std::uint32_t n, unsigned bits) {
    std::uint32_t out = 0;
    for (unsigned i = 0; i < bits; ++i) { out = (out << 1) | (n & 1); n >>= 1; }
    return out;
}

struct DiagonalIndex { int diagonal; int giant; std::size_t baby; };
inline DiagonalIndex split_diagonal(int diagonal, std::size_t baby) {
    if (!baby || baby > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid baby stride");
    int rem = diagonal % static_cast<int>(baby);
    if (rem < 0) rem += static_cast<int>(baby);
    return {diagonal, diagonal - rem, static_cast<std::size_t>(rem)};
}

// W is the original, independent float32 [out,in] dense matrix. The mask
// excludes all wraparound feature pairs, not token rows. The inverse giant
// rotation is applied to the plaintext BEFORE encoding, never to weights.
inline std::vector<double> diagonal_period(const float *weights,
        std::size_t input_width, std::size_t output_width,
        std::size_t input_tile, std::size_t output_tile,
        std::size_t tile, int diagonal, int giant) {
    if (!weights || !tile || input_width % tile || output_width % tile ||
        input_tile >= input_width / tile || output_tile >= output_width / tile)
        throw std::invalid_argument("invalid dense diagonal shape");
    std::vector<double> result(tile);
    for (std::size_t i = 0; i < tile; ++i) {
        const int col = static_cast<int>(i) + diagonal;
        if (col < 0 || col >= static_cast<int>(tile)) continue;
        // rot(D,-g)[j]=D[j-g], hence D[i] goes to j=i+g (mod tile).
        const auto j = static_cast<std::size_t>(
            (static_cast<int>(i) + giant + static_cast<int>(tile) * 2) %
            static_cast<int>(tile));
        result[j] = weights[(output_tile * tile + i) * input_width +
                            input_tile * tile + static_cast<std::size_t>(col)];
    }
    return result;
}

inline const std::vector<std::uint32_t> &bit_reversed_indices(std::size_t degree) {
    // Encoding a matrix repeatedly uses the same N. Cache this immutable
    // index map per CPU thread instead of recomputing N*logN bit operations
    // for every one of tens of thousands of independently encoded terms.
    const auto bits = log2_exact(degree);
    if (degree > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("RNS degree does not fit compact indexing");
    thread_local std::size_t cached_degree = 0;
    thread_local std::vector<std::uint32_t> indices;
    if (cached_degree != degree) {
        indices.resize(degree);
        for (std::size_t j = 0; j < degree; ++j)
            indices[j] = reverse_bits(static_cast<std::uint32_t>(j), bits);
        cached_degree = degree;
    }
    return indices;
}

struct ExactPeriodicRns {
    std::size_t degree = 0;
    std::size_t limbs = 0;
    std::size_t period = 0;
    std::vector<std::uint32_t> values; // [limb][period], no quantization
    bool fallback() const { return period == degree; }
};

// Exact representation change of an already encoded full-N NTT plaintext.
// The full-period fallback is ORDINARY NTT order, not bit-reversed order.
// A candidate is accepted only after checking EVERY residue in EVERY limb.
inline ExactPeriodicRns compress_exact(const std::uint64_t *full,
        std::size_t degree, std::size_t limbs, std::size_t expected_period = 512) {
    if (!full || !power_of_two(degree) || !limbs ||
        degree > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("invalid exact RNS compression shape");
    const auto &reversed = bit_reversed_indices(degree);
    const auto has_period = [&](std::size_t period) {
        if (period == degree) return true;
        for (std::size_t l = 0; l < limbs; ++l) {
            const auto *src = full + l * degree;
            for (std::size_t j = 0; j < degree; ++j) {
                const auto representative = reversed[reversed[j] & (period - 1)];
                if (src[j] != src[representative]) return false;
            }
        }
        return true;
    };
    std::size_t period = power_of_two(expected_period)
        ? std::min(degree, expected_period) : 1;
    while (period < degree && !has_period(period)) period <<= 1;
    while (period > 1 && has_period(period / 2)) period >>= 1;
    ExactPeriodicRns result{degree, limbs, period, {}};
    result.values.resize(limbs * period);
    for (std::size_t l = 0; l < limbs; ++l) {
        for (std::size_t k = 0; k < period; ++k) {
            const auto value = full[l * degree + (period == degree ? k : reversed[k])];
            if (value > std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("RNS residue does not fit GPU uint32");
            result.values[l * period + k] = static_cast<std::uint32_t>(value);
        }
    }
    // Deliberate independent reconstruction pass, including the NTT-order
    // fallback. This is setup-time validation, not a timed GPU approximation.
    for (std::size_t l = 0; l < limbs; ++l)
        for (std::size_t j = 0; j < degree; ++j) {
            const auto k = period == degree ? j : reversed[j] & (period - 1);
            if (result.values[l * period + k] != full[l * degree + j])
                throw std::logic_error("full-N exact RNS reconstruction failed");
        }
    return result;
}

} // namespace bert::compact
