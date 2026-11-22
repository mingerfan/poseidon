#pragma once

#include "llama_manual/layout.hpp"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

namespace llama_manual {

struct LinearStats {
    std::uint64_t plain_products = 0;
    std::uint64_t additions = 0;
    std::uint64_t baby_rotations = 0;
    std::uint64_t giant_rotations = 0;
    std::uint64_t rescale_calls = 0;
    std::size_t output_batch_tiles = 0;
    // Logical temporary ciphertext owners only; excludes inputs, final outputs,
    // keys, hoist workspace, allocator pools and transient kernel allocations.
    std::size_t temporary_owner_bound = 0;
};

// WeightAt(out_feature, in_feature) follows HF Linear [out, in] convention.
// Diagonals are generated on demand; no full encoded-weight expansion is kept.
template<class WeightAt>
std::vector<double> bsgs_diagonal(const Layout &l, WeightAt &weight,
    std::size_t input_width, std::size_t output_width,
    std::size_t input_tile, std::size_t output_tile,
    std::size_t giant, std::size_t baby) {
    const auto shift = giant * l.baby_steps;
    const auto diagonal = shift + baby;
    std::vector<double> result(l.slots, 0.0);
    // Pre-rotate the diagonal RIGHT by the giant step; ciphertext rotations
    // are LEFT. This is essential for Y = sum_g rotate(sum_b rot(X,b)*D',g).
    for (std::size_t f = 0; f < l.feature_tile; ++f) {
        const auto j = (f + l.feature_tile - shift) % l.feature_tile;
        const auto out = output_tile * l.feature_tile + j;
        const auto in = input_tile * l.feature_tile + (j + diagonal) % l.feature_tile;
        if (out >= output_width || in >= input_width) continue;
        const double value = weight(out, in);
        if (!std::isfinite(value)) throw std::invalid_argument("non-finite weight");
        std::fill_n(result.begin() + f * l.token_capacity, l.token_capacity, value);
    }
    return result;
}

// Immediate C++ calls, not a DSL, graph tracer or compiler-generated plan.
// Backend owns arithmetic; FloatSlots is ONLY a server validation backend.
// The GPU backend uses public evaluation keys and never receives a secret key.
template<class Backend, class WeightAt>
std::vector<typename Backend::Cipher> linear(
    Backend &ops, const Layout &l,
    const std::vector<typename Backend::Cipher> &input,
    std::size_t input_width, std::size_t output_width, WeightAt weight,
    std::size_t output_batch_tiles = 2, LinearStats *report = nullptr) {
    using Cipher = typename Backend::Cipher;
    l.validate();
    if (!input_width || !output_width || !output_batch_tiles ||
        input.size() != l.tiles(input_width)) {
        throw std::invalid_argument("invalid Linear dimensions");
    }
    ops.validate_linear_inputs(input, l.slots);
    const auto outputs = l.tiles(output_width);
    const auto batch_size = std::min(output_batch_tiles, outputs);
    const auto giants = l.feature_tile / l.baby_steps;
    LinearStats stats;
    stats.output_batch_tiles = batch_size;
    stats.temporary_owner_bound = batch_size * giants + l.baby_steps + 3;
    std::vector<int> baby_steps;
    for (std::size_t b = 1; b < l.baby_steps; ++b)
        baby_steps.push_back(static_cast<int>(b * l.token_capacity));
    std::vector<Cipher> output;
    output.reserve(outputs);
    for (std::size_t start = 0; start < outputs; start += batch_size) {
        const auto count = std::min(batch_size, outputs - start);
        std::vector<std::optional<Cipher>> groups(count * giants);
        for (std::size_t i = 0; i < input.size(); ++i) {
            auto babies = ops.rotate_many(input[i], baby_steps);
            if (babies.size() != baby_steps.size())
                throw std::runtime_error("backend returned wrong rotation count");
            stats.baby_rotations += baby_steps.size();
            for (std::size_t o = 0; o < count; ++o) {
                for (std::size_t g = 0; g < giants; ++g) {
                    auto &acc = groups[o * giants + g];
                    for (std::size_t b = 0; b < l.baby_steps; ++b) {
                        auto diagonal = bsgs_diagonal(l, weight, input_width,
                            output_width, i, start + o, g, b);
                        const Cipher &x = b == 0 ? input[i] : babies[b - 1];
                        auto product = ops.multiply_plain_raw(x, diagonal);
                        ++stats.plain_products;
                        if (acc) {
                            acc = ops.add(*acc, product);
                            ++stats.additions;
                        } else acc.emplace(std::move(product));
                    }
                }
            }
        }
        for (std::size_t o = 0; o < count; ++o) {
            std::optional<Cipher> sum;
            for (std::size_t g = 0; g < giants; ++g) {
                auto &group = groups[o * giants + g];
                Cipher term = g == 0 ? std::move(*group) : ops.rotate(*group,
                    static_cast<int>(g * l.baby_steps * l.token_capacity));
                if (g != 0) ++stats.giant_rotations;
                group.reset();
                if (sum) { sum = ops.add(*sum, term); ++stats.additions; }
                else sum.emplace(std::move(term));
            }
            // All products have the same level/scale; one final multi-prime
            // rescale per output tile. Requires adequate pre-rescale Q budget.
            output.push_back(ops.finish_linear(std::move(*sum)));
            ++stats.rescale_calls;
        }
    }
    if (report) *report = stats;
    return output;
}

} // namespace llama_manual
