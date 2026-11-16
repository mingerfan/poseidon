#pragma once

#include "prepared_qk.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bert::attention::reference {

namespace detail {

inline std::size_t checked_slots(std::size_t tokens, std::size_t head_dim) {
    if (tokens == 0 || head_dim == 0 || head_dim > tokens) {
        throw std::invalid_argument("QK requires tokens > 0 and 0 < head_dim <= tokens");
    }
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (tokens > maximum / 2 || tokens > maximum / (2 * tokens)) {
        throw std::invalid_argument("QK tile/slot dimensions overflow");
    }
    return tokens * (2 * tokens);
}

inline std::size_t validate_inputs(const std::vector<double> &query,
                                   const std::vector<double> &key,
                                   std::size_t tokens, std::size_t head_dim) {
    const auto slots = checked_slots(tokens, head_dim);
    if (query.size() != slots || key.size() != slots) {
        throw std::invalid_argument("QK inputs must each have tokens * (2 * tokens) slots");
    }
    // Each token contains exactly two T-lane head groups. The D meaningful
    // lanes are followed by genuine zero padding, not a third/fourth head.
    const auto tile = 2 * tokens;
    for (std::size_t i = 0; i < slots; ++i) {
        if (!std::isfinite(query[i]) || !std::isfinite(key[i])) {
            throw std::invalid_argument("QK inputs must be finite, including gap lanes");
        }
        const auto lane = (i % tile) % tokens;
        if (lane >= head_dim && (query[i] != 0.0 || key[i] != 0.0)) {
            throw std::invalid_argument("QK input head-gap lanes must be zero");
        }
    }
    return slots;
}

// Full-vector cyclic LEFT rotation, never a rotation local to a head or row.
inline std::vector<double> rotate_left(const std::vector<double> &input,
                                       std::size_t step) {
    step %= input.size();
    std::vector<double> output(input.size());
    std::rotate_copy(input.begin(), input.begin() + step, input.end(), output.begin());
    return output;
}

inline void add_in_place(std::vector<double> &target, const std::vector<double> &source) {
    for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] += source[i];
    }
}

inline bool representable_scale(double log2_scale, double minimum = 0.0) {
    if (!std::isfinite(log2_scale)) {
        return false;
    }
    const double scale = std::exp2(log2_scale);
    return std::isfinite(scale) && scale > minimum;
}

} // namespace detail

// Scalar translation of cerium_packing.py::_Arithmetic.cyclic_qk. Production
// uses T128/D64/baby8, but this reference also supports smaller gapped heads.
// Non-power-of-two D works when its padded reduction width fits in T. The
// requested baby is capped at T (as in Python's min(8, T)) and must divide T.
inline std::vector<double> cyclic_qk(const std::vector<double> &query,
                                      const std::vector<double> &key,
                                      std::size_t tokens = 128,
                                      std::size_t head_dim = 64,
                                      std::size_t baby = 8) {
    const auto slots = detail::validate_inputs(query, key, tokens, head_dim);
    const auto tile = 2 * tokens;
    if (baby == 0) {
        throw std::invalid_argument("QK baby step must be positive");
    }
    baby = std::min(baby, tokens);
    if (tokens % baby != 0) {
        throw std::invalid_argument("QK baby step must divide tokens");
    }
    std::size_t reduction_width = 1;
    while (reduction_width < head_dim) {
        if (reduction_width > tokens / 2) {
            throw std::invalid_argument("QK padded reduction must fit within a head group");
        }
        reduction_width *= 2;
    }

    std::vector<std::vector<double>> keys;
    keys.reserve(baby);
    for (std::size_t b = 0; b < baby; ++b) {
        keys.push_back(detail::rotate_left(key, tile * b));
    }
    const double normalization = 1.0 / std::sqrt(static_cast<double>(head_dim));
    std::vector<double> output(slots, 0.0);
    const auto giants = tokens / baby;
    for (std::size_t g = 0; g < giants; ++g) {
        const auto query_step = tile * baby * g;
        const auto shifted_query = detail::rotate_left(query, (slots - query_step) % slots);
        std::vector<double> subtotal(slots, 0.0);
        for (std::size_t b = 0; b < baby; ++b) {
            std::vector<double> product(slots);
            for (std::size_t i = 0; i < slots; ++i) {
                product[i] = shifted_query[i] * keys[b][i];
            }
            // Rotate the CURRENT product at every stage, not the initial one.
            for (std::size_t shift = 1; shift < head_dim; shift *= 2) {
                detail::add_in_place(product, detail::rotate_left(product, shift));
            }
            for (std::size_t i = 0; i < slots; ++i) {
                product[i] *= i % tokens == 0 ? normalization : 0.0;
            }
            if (b != 0) {
                product = detail::rotate_left(product, slots - b);
            }
            detail::add_in_place(subtotal, product);
        }
        // This nonzero giant compensation is essential to cyclic score order.
        const auto giant_step = (tile * baby - baby) * g;
        if (giant_step != 0) {
            subtotal = detail::rotate_left(subtotal, giant_step);
        }
        detail::add_in_place(output, subtotal);
    }
    return output;
}

// Scalar vector-operation counts, recorded at the paired call sites below.
// Rescale2 and shared-source batches are only abstract crypto schedule counts:
// scalar doubles execute neither CKKS rescales nor real hoisted rotations.
struct PairedScheduleCounts {
    std::uint64_t cc_multiply_calls = 0;
    std::uint64_t cp_multiply_calls = 0;
    std::uint64_t rotation_calls = 0;
    std::uint64_t add_calls = 0;
    std::uint64_t abstract_rescale2_calls = 0;
    std::uint64_t logical_shared_source_batches = 0;
};

// Pair adjacent baby products in the genuine zero message gaps. This is NOT
// an equivalence claim for dirty-gap messages or bitwise-zero encrypted gaps.
// D must be T/2 and a power of two; the baseline's wider D domain is unchanged.
inline std::vector<double> paired_cyclic_qk(const std::vector<double> &query,
                                           const std::vector<double> &key,
                                           std::size_t tokens = 128,
                                           std::size_t head_dim = 64,
                                           std::size_t baby = 8,
                                           PairedScheduleCounts *counts = nullptr) {
    const auto slots = detail::validate_inputs(query, key, tokens, head_dim);
    if (tokens % 2 != 0 || head_dim != tokens / 2 ||
        (head_dim & (head_dim - 1)) != 0) {
        throw std::invalid_argument("paired QK requires power-of-two D = T/2");
    }
    if (baby == 0 || baby % 2 != 0) {
        throw std::invalid_argument("paired QK baby step must be positive and even");
    }
    baby = std::min(baby, tokens);
    if (tokens % baby != 0) {
        throw std::invalid_argument("paired QK baby step must divide tokens");
    }
    const auto tile = 2 * tokens;
    const auto giants = tokens / baby;
    PairedScheduleCounts measured;
    const auto rotate = [&](const std::vector<double> &input, std::size_t shift) {
        shift %= slots;
        if (shift == 0) return input;
        ++measured.rotation_calls;
        return detail::rotate_left(input, shift);
    };
    const auto add = [&](std::vector<double> &target, const std::vector<double> &source) {
        ++measured.add_calls;
        detail::add_in_place(target, source);
    };
    const auto multiply = [&](const std::vector<double> &left, const std::vector<double> &right) {
        ++measured.cc_multiply_calls;
        ++measured.abstract_rescale2_calls; // each true CC needs its own R2
        std::vector<double> product(slots);
        for (std::size_t i = 0; i < slots; ++i) product[i] = left[i] * right[i];
        return product;
    };
    const double normalization = 1.0 / std::sqrt(static_cast<double>(head_dim));
    const auto mask = [&](const std::vector<double> &input, std::size_t start) {
        ++measured.cp_multiply_calls;
        std::vector<double> masked(slots);
        for (std::size_t i = 0; i < slots; ++i) {
            masked[i] = input[i] * (i % tokens == start ? normalization : 0.0);
        }
        return masked;
    };
    // These are two logical batches, not a scalar/CPU hoist measurement.
    ++measured.logical_shared_source_batches;
    std::vector<std::vector<double>> keys;
    keys.reserve(baby);
    for (std::size_t b = 0; b < baby; ++b) keys.push_back(rotate(key, tile * b));
    ++measured.logical_shared_source_batches;
    std::vector<std::vector<double>> queries;
    queries.reserve(giants);
    for (std::size_t g = 0; g < giants; ++g) {
        queries.push_back(rotate(query, (slots - tile * baby * g) % slots));
    }
    std::vector<double> output;
    for (std::size_t g = 0; g < giants; ++g) {
        std::vector<double> subtotal;
        for (std::size_t b_even = 0; b_even < baby; b_even += 2) {
            auto p0 = multiply(queries[g], keys[b_even]);
            const auto p1 = multiply(queries[g], keys[b_even + 1]);
            // LEFT -D moves P1's live D lanes right into P0's zero gap.
            add(p0, rotate(p1, slots - head_dim));
            // Each rotate reads the CURRENT combined source, not the original.
            for (std::size_t shift = 1; shift < head_dim; shift *= 2) {
                add(p0, rotate(p0, shift));
            }
            auto masked_a = mask(p0, 0);
            const auto masked_b = mask(p0, head_dim);
            // LEFT +(D-1) moves the second head-start result to output lane 1.
            add(masked_a, rotate(masked_b, head_dim - 1));
            if (b_even != 0) masked_a = rotate(masked_a, slots - b_even);
            if (b_even == 0) subtotal = std::move(masked_a);
            else add(subtotal, masked_a);
        }
        subtotal = rotate(subtotal, (tile * baby - baby) * g);
        if (g == 0) output = std::move(subtotal);
        else add(output, subtotal);
    }
    ++measured.abstract_rescale2_calls; // one final R2 after all mask CP sums
    if (counts) *counts = measured;
    return output;
}

// Actual scalar vector-operation calls for pre-CC input packing. ABSTRACT R2
// and LOGICAL source batches describe a crypto schedule only: doubles execute
// neither CKKS rescaling nor real shared-decomposition/hoisted rotations.
struct PackedInputScheduleCounts {
    std::uint64_t cc_multiply_calls = 0;
    std::uint64_t cp_multiply_calls = 0;
    std::uint64_t rotation_calls = 0;
    std::uint64_t add_calls = 0;
    std::uint64_t abstract_rescale2_calls = 0;
    std::uint64_t logical_shared_source_batches = 0;
};

// Pack the SAME query and adjacent-token keys BEFORE CC. Genuine zero ideal-
// message gaps are essential: Q+Q.rotate(-D), K+K.rotate(tile-D) occupy the two
// halves of each T-lane head group without cross terms. This is not a claim
// about bitwise-zero CKKS gaps, ciphertext equivalence, or GPU correctness.
inline std::vector<double> packed_input_cyclic_qk(
    const std::vector<double> &query, const std::vector<double> &key,
    std::size_t tokens = 128, std::size_t head_dim = 64, std::size_t baby = 8,
    PackedInputScheduleCounts *counts = nullptr) {
    const auto slots = detail::validate_inputs(query, key, tokens, head_dim);
    if (tokens % 2 != 0 || head_dim != tokens / 2 ||
        (head_dim & (head_dim - 1)) != 0) {
        throw std::invalid_argument("packed-input QK requires power-of-two D = T/2");
    }
    if (baby == 0 || baby % 2 != 0) {
        throw std::invalid_argument("packed-input QK baby step must be positive and even");
    }
    baby = std::min(baby, tokens);
    if (tokens % baby != 0) {
        throw std::invalid_argument("packed-input QK baby step must divide tokens");
    }
    const auto tile = 2 * tokens;
    const auto giants = tokens / baby;
    PackedInputScheduleCounts measured;
    const auto rotate = [&](const std::vector<double> &input, std::size_t shift) {
        shift %= slots;
        if (shift == 0) return input;
        ++measured.rotation_calls;
        return detail::rotate_left(input, shift);
    };
    const auto add = [&](std::vector<double> &target, const std::vector<double> &source) {
        ++measured.add_calls;
        detail::add_in_place(target, source);
    };
    const auto multiply = [&](const std::vector<double> &left, const std::vector<double> &right) {
        ++measured.cc_multiply_calls;
        ++measured.abstract_rescale2_calls;
        std::vector<double> product(slots);
        for (std::size_t i = 0; i < slots; ++i) product[i] = left[i] * right[i];
        return product;
    };
    const double normalization = 1.0 / std::sqrt(static_cast<double>(head_dim));
    const auto mask = [&](const std::vector<double> &input, std::size_t start) {
        ++measured.cp_multiply_calls;
        std::vector<double> masked(slots);
        for (std::size_t i = 0; i < slots; ++i) {
            masked[i] = input[i] * (i % tokens == start ? normalization : 0.0);
        }
        return masked;
    };
    auto packed_query = query;
    add(packed_query, rotate(query, slots - head_dim));
    auto packed_key = key;
    add(packed_key, rotate(key, tile - head_dim));

    // Two LOGICAL shared-source batches; g=0 and bEven=0 are copies, not
    // rotations. At T128/baby8, only g=1..15 and bEven=2,4,6 rotate.
    ++measured.logical_shared_source_batches;
    std::vector<std::vector<double>> queries;
    queries.reserve(giants);
    for (std::size_t g = 0; g < giants; ++g) {
        queries.push_back(rotate(packed_query, (slots - tile * baby * g) % slots));
    }
    ++measured.logical_shared_source_batches;
    std::vector<std::vector<double>> key_pairs;
    key_pairs.reserve(baby / 2);
    for (std::size_t b_even = 0; b_even < baby; b_even += 2) {
        key_pairs.push_back(rotate(packed_key, tile * b_even));
    }
    std::vector<double> output;
    for (std::size_t g = 0; g < giants; ++g) {
        std::vector<double> subtotal;
        for (std::size_t e = 0; e < baby / 2; ++e) {
            const auto b_even = 2 * e;
            // ONE actual scalar CC contains both adjacent baby products.
            auto product = multiply(queries[g], key_pairs[e]);
            for (std::size_t shift = 1; shift < head_dim; shift *= 2) {
                add(product, rotate(product, shift)); // CURRENT source
            }
            auto masked_a = mask(product, 0);
            const auto masked_b = mask(product, head_dim);
            add(masked_a, rotate(masked_b, head_dim - 1));
            if (b_even != 0) masked_a = rotate(masked_a, slots - b_even);
            if (e == 0) subtotal = std::move(masked_a);
            else add(subtotal, masked_a);
        }
        subtotal = rotate(subtotal, (tile * baby - baby) * g);
        if (g == 0) output = std::move(subtotal);
        else add(output, subtotal);
    }
    ++measured.abstract_rescale2_calls; // ONE final logical R2 after CP sums
    if (counts) *counts = measured;
    return output;
}

// Independent indexed dot-product oracle: no rotation, mask, BSGS, or reduction
// helper is shared with cyclic_qk. d indexes cyclic key distance, NOT key row.
inline std::vector<double> direct_oracle(const std::vector<double> &query,
                                         const std::vector<double> &key,
                                         std::size_t tokens = 128,
                                         std::size_t head_dim = 64) {
    const auto slots = detail::validate_inputs(query, key, tokens, head_dim);
    const auto tile = 2 * tokens;
    const long double normalization = 1.0L / std::sqrt(static_cast<long double>(head_dim));
    std::vector<double> output(slots);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t h = 0; h < 2; ++h) {
            for (std::size_t d = 0; d < tokens; ++d) {
                long double dot = 0.0L;
                for (std::size_t j = 0; j < head_dim; ++j) {
                    dot += static_cast<long double>(query[t * tile + h * tokens + j]) *
                           static_cast<long double>(key[((t + d) % tokens) * tile + h * tokens + j]);
                }
                output[t * tile + h * tokens + d] = static_cast<double>(dot * normalization);
            }
        }
    }
    return output;
}

struct ScalePlan {
    double intermediate_scale_log2;
    double mask_scale_log2;
    double output_scale_log2;
};

// Poseidon physical level L has L+1 ordered Q primes. Two REAL primes are
// dropped after CC and after CP. The declared metadata is checked, never fixed
// up or overwritten to hide a mismatch with the actual ordered modulus chain.
inline ScalePlan derive_scale_plan(const QkSpec &spec,
                                    const std::vector<std::uint64_t> &ordered_q) {
    if (spec.input_level < 4 || spec.input_level >= ordered_q.size() ||
        spec.output_level != spec.input_level - 4) {
        throw std::invalid_argument("QK requires available physical levels Lin -> Lin-2 -> Lin-4");
    }
    if (!detail::representable_scale(spec.input_scale_log2) ||
        !detail::representable_scale(spec.intermediate_scale_log2) ||
        !detail::representable_scale(spec.output_scale_log2)) {
        throw std::invalid_argument("QK scales must exponentiate to finite positive doubles");
    }
    for (const auto q : ordered_q) {
        if (q <= 1) {
            throw std::invalid_argument("QK ordered Q moduli must be greater than one");
        }
    }
    const auto level = spec.input_level;
    const double intermediate = 2.0 * spec.input_scale_log2 -
                                std::log2(static_cast<double>(ordered_q[level])) -
                                std::log2(static_cast<double>(ordered_q[level - 1]));
    if (!detail::representable_scale(intermediate) ||
        std::abs(intermediate - spec.intermediate_scale_log2) > 1e-8) {
        throw std::invalid_argument("QK intermediate scale disagrees with the real two-prime rescale");
    }
    const double mask = spec.output_scale_log2 - intermediate +
                        std::log2(static_cast<double>(ordered_q[level - 2])) +
                        std::log2(static_cast<double>(ordered_q[level - 3]));
    // Reject unusable encoding scales before any GPU/context setup. A finite
    // logarithm alone does not prevent exp2 overflow, underflow, or scale <= 1.
    if (!detail::representable_scale(mask, 1.0)) {
        throw std::invalid_argument("QK mask encoding scale must be finite and greater than one");
    }
    return {intermediate, mask, spec.output_scale_log2};
}

} // namespace bert::attention::reference
