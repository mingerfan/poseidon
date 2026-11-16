#pragma once

// Standalone scalar AV and physical-scale references. No Poseidon/CUDA headers.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bert::attention::av_reference {

struct Layout {
    std::size_t tokens = 128;
    std::size_t tile = 256;
    std::size_t head_dim = 64;
    std::size_t baby = 8;
};

namespace detail {
inline std::size_t checked_slots(const Layout &layout) {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (layout.tokens == 0 || layout.tokens > maximum / 2 ||
        layout.tile != 2 * layout.tokens || layout.tokens > maximum / layout.tile)
        throw std::invalid_argument("AV requires tile=2*tokens and a nonoverflowing full slot block");
    if (layout.head_dim == 0 || layout.head_dim > layout.tokens)
        throw std::invalid_argument("AV requires 0 < head_dim <= tokens");
    if (layout.baby == 0 || layout.tokens % std::min(layout.baby, layout.tokens) != 0)
        throw std::invalid_argument("AV capped baby step must be positive and divide tokens");
    return layout.tokens * layout.tile;
}

// Full ciphertext LEFT rotation: output[i] = input[(i+step)%slots].
inline std::vector<double> rotate_left(const std::vector<double> &input, std::size_t step) {
    step %= input.size();
    std::vector<double> output(input.size());
    std::rotate_copy(input.begin(), input.begin() + step, input.end(), output.begin());
    return output;
}
inline void add(std::vector<double> &target, const std::vector<double> &source) {
    for (std::size_t i = 0; i < target.size(); ++i) target[i] += source[i];
}
inline bool representable_scale(double log2_scale, double minimum = 0.0) {
    const auto scale = std::exp2(log2_scale);
    return std::isfinite(log2_scale) && std::isfinite(scale) && scale > minimum;
}
} // namespace detail

// V contains two T-lane heads per token, each with D live lanes and T-D
// genuine zero-gap lanes. Scores, unlike V, use ALL T cyclic-offset lanes.
inline void validate_values(const std::vector<double> &values, const Layout &layout = {}) {
    const auto slots = detail::checked_slots(layout);
    if (values.size() != slots)
        throw std::invalid_argument("AV values must contain tokens*tile full ciphertext slots");
    for (std::size_t i = 0; i < slots; ++i) {
        if (!std::isfinite(values[i]))
            throw std::invalid_argument("AV values contain a nonfinite lane at slot " + std::to_string(i));
        if (i % layout.tokens >= layout.head_dim && values[i] != 0.0)
            throw std::invalid_argument("AV values have a nonzero head-gap lane at slot " +
                                        std::to_string(i) + " (lane " +
                                        std::to_string(i % layout.tokens) + ")");
    }
}

namespace detail {
inline std::size_t validate_inputs(const std::vector<double> &scores,
                                   const std::vector<double> &values, const Layout &layout) {
    const auto slots = checked_slots(layout);
    if (values.size() != slots)
        throw std::invalid_argument("AV values must contain tokens*tile full ciphertext slots");
    for (const auto value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("AV values must be finite in ALL lanes");
    if (scores.size() != values.size())
        throw std::invalid_argument("AV scores must contain tokens*tile full ciphertext slots");
    for (const auto score : scores)
        if (!std::isfinite(score)) throw std::invalid_argument("AV scores must be finite in ALL offset lanes");
    return values.size();
}
} // namespace detail

// Independent indexed oracle; no masks, rotations, or BSGS indexing. This is
// weighted V BEFORE attention normalization, not an ordinary row-major AV.
// General baseline arithmetic includes ALL feature lanes, even dirty gaps;
// callers wanting a strict zero-gap contract must separately validate_values.
inline std::vector<double> direct(const std::vector<double> &scores,
                                  const std::vector<double> &values,
                                  const Layout &layout = {}) {
    const auto slots = detail::validate_inputs(scores, values, layout);
    std::vector<double> output(slots, 0.0);
    for (std::size_t t = 0; t < layout.tokens; ++t)
        for (std::size_t h = 0; h < 2; ++h)
            for (std::size_t j = 0; j < layout.tokens; ++j) {
                long double sum = 0;
                for (std::size_t d = 0; d < layout.tokens; ++d)
                    sum += static_cast<long double>(scores[t * layout.tile + h * layout.tokens + d]) *
                           values[((t + d) % layout.tokens) * layout.tile + h * layout.tokens + j];
                output[t * layout.tile + h * layout.tokens + j] = static_cast<double>(sum);
            }
    return output;
}

// Literal scalar translation of cerium_packing.py::_Arithmetic.cyclic_av and
// _cyclic_av_mask. Each mask is explicitly T-periodic across ALL slots. The
// shared V babies, diagonal selectors, giant compensations, and score left /
// right masks are deliberately independent of the direct oracle above.
inline std::vector<double> bsgs(const std::vector<double> &scores,
                                const std::vector<double> &values,
                                const Layout &layout = {}) {
    const auto slots = detail::validate_inputs(scores, values, layout);
    const auto baby = std::min(layout.baby, layout.tokens);
    const auto giants = layout.tokens / baby;
    std::vector<std::vector<double>> babies;
    for (std::size_t b = 0; b < baby; ++b)
        babies.push_back(detail::rotate_left(values, layout.tile * b));
    std::vector<double> diagonal(slots, 0.0);
    for (std::size_t g = 0; g < giants; ++g) {
        std::vector<double> subtotal(slots, 0.0);
        for (std::size_t b = 0; b < baby; ++b) {
            const auto shift = baby * g + b;
            for (std::size_t i = 0; i < slots; ++i)
                subtotal[i] += babies[b][i] * (i % layout.tokens == shift ? 1.0 : 0.0);
        }
        // T-periodic selectors need NO extra giant mask compensation.
        detail::add(diagonal, detail::rotate_left(subtotal, layout.tile * baby * g));
    }
    babies.clear();
    for (std::size_t b = 0; b < baby; ++b)
        babies.push_back(detail::rotate_left(diagonal, layout.tile * b));
    std::vector<double> output(slots, 0.0);
    for (std::size_t g = 0; g < giants; ++g) {
        const auto negative_step = (layout.tile * baby - baby) * g;
        const auto base = detail::rotate_left(scores, (slots - negative_step) % slots);
        std::vector<double> subtotal(slots, 0.0);
        for (std::size_t b = 0; b < baby; ++b) {
            const auto shift = baby * g + b;
            if (shift == 0) {
                for (std::size_t i = 0; i < slots; ++i) subtotal[i] += base[i] * babies[b][i];
                continue;
            }
            const auto left = detail::rotate_left(base, b);
            const auto right = detail::rotate_left(base, slots - layout.tokens + b);
            for (std::size_t i = 0; i < slots; ++i) {
                const auto lane = i % layout.tokens;
                const auto cyclic = left[i] * (lane < layout.tokens - shift ? 1.0 : 0.0) +
                                    right[i] * (lane >= layout.tokens - shift ? 1.0 : 0.0);
                subtotal[i] += cyclic * babies[b][i];
            }
        }
        detail::add(output, detail::rotate_left(subtotal, layout.tile * baby * g));
    }
    return output;
}

// Actual scalar execution callsites in the packed schedule. A rescale call is
// a batched R2 or R4; physical_drops counts its individual prime drops. Hoists
// count nonempty shared-source rotation families, not individual rotations.
struct PackedInputCounts {
    std::size_t cp = 0, cc = 0, relin = 0;
    std::size_t rescale = 0, r2 = 0, r4 = 0, physical_drops = 0;
    std::size_t rotations = 0, additions = 0, hoists = 0;
};

namespace detail {
inline void validate_partial_layout(const Layout &layout) {
    checked_slots(layout);
    if (layout.tokens != 2 * layout.head_dim ||
        std::min(layout.baby, layout.tokens) % 2 != 0)
        throw std::invalid_argument("packed AV requires tokens=2*head_dim and an even capped baby");
}
struct PackedArithmetic {
    PackedInputCounts &counts;
    std::vector<double> rotate(const std::vector<double> &input, std::size_t step) {
        if (step % input.size() == 0) return input;
        ++counts.rotations;
        return rotate_left(input, step);
    }
    void add_to(std::vector<double> &a, const std::vector<double> &b) {
        ++counts.additions;
        add(a, b);
    }
    std::vector<double> cp(const std::vector<double> &a, const std::vector<double> &mask) {
        ++counts.cp;
        std::vector<double> result(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) result[i] = a[i] * mask[i];
        return result;
    }
    std::vector<double> cc(const std::vector<double> &a, const std::vector<double> &b) {
        ++counts.cc;
        std::vector<double> result(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) result[i] = a[i] * b[i];
        return result;
    }
    void relinearize(std::vector<double> &) { ++counts.relin; }
    void rescale(std::vector<double> &, std::size_t drops) {
        ++counts.rescale;
        counts.r2 += drops == 2;
        counts.r4 += drops == 4;
        counts.physical_drops += drops;
    }
    void hoist() { ++counts.hoists; }
};

// The two mutation switches are deliberately internal: tests must demonstrate
// that global tile-D and -D are not interchangeable with tempting wrong signs.
inline std::vector<double> partial_packed_input_impl(
    const std::vector<double> &scores, const std::vector<double> &values,
    const Layout &layout, PackedInputCounts *out_counts,
    bool wrong_diagonal_rotation = false, bool wrong_score_rotation = false) {
    const auto slots = validate_inputs(scores, values, layout);
    validate_values(values, layout); // Unlike general baseline AV, gaps MUST be zero.
    validate_partial_layout(layout);
    const auto baby = std::min(layout.baby, layout.tokens);
    const auto giants = layout.tokens / baby;
    PackedInputCounts counts;
    PackedArithmetic op{counts};
    std::vector<std::vector<double>> value_babies{values};
    if (baby > 1) op.hoist();
    for (std::size_t b = 1; b < baby; ++b)
        value_babies.push_back(op.rotate(values, layout.tile * b));
    std::vector<double> diagonal;
    for (std::size_t g = 0; g < giants; ++g) {
        std::vector<double> subtotal;
        for (std::size_t b = 0; b < baby; ++b) {
            const auto shift = baby * g + b;
            std::vector<double> selector(slots);
            for (std::size_t i = shift; i < slots; i += layout.tokens) selector[i] = 1;
            auto term = op.cp(value_babies[b], selector);
            if (b == 0) subtotal = std::move(term); else op.add_to(subtotal, term);
        }
        if (g != 0) subtotal = op.rotate(subtotal, layout.tile * baby * g);
        if (g == 0) diagonal = std::move(subtotal); else op.add_to(diagonal, subtotal);
    }
    op.rescale(diagonal, 2); // Exactly one physical R2, L8 -> canonical L6.
    value_babies.clear();
    auto diagonal_gap = op.rotate(diagonal, wrong_diagonal_rotation ?
                                  slots - layout.head_dim : layout.tile - layout.head_dim);
    op.add_to(diagonal, diagonal_gap);
    std::vector<std::vector<double>> even_babies{std::move(diagonal)};
    if (baby > 2) op.hoist();
    for (std::size_t b = 2; b < baby; b += 2)
        even_babies.push_back(op.rotate(even_babies[0], layout.tile * b));
    // Score giant bases are a shared-source rotation family.
    std::vector<std::vector<double>> bases{scores};
    if (giants > 1) op.hoist();
    for (std::size_t g = 1; g < giants; ++g)
        bases.push_back(op.rotate(scores, (slots - (layout.tile * baby - baby) * g) % slots));
    std::vector<double> output;
    for (std::size_t g = 0; g < giants; ++g) {
        const auto &base = bases[g];
        if (baby > 1) op.hoist(); // All left/right source rotations share this base.
        const auto score_term = [&](std::size_t b) {
            const auto shift = baby * g + b;
            std::vector<double> left_mask(slots), right_mask(slots);
            bool has_right = false;
            for (std::size_t i = 0; i < slots; ++i) {
                const auto j = i % layout.tokens;
                left_mask[i] = double(j < layout.head_dim && j < layout.tokens - shift);
                right_mask[i] = double(j < layout.head_dim && j >= layout.tokens - shift);
                has_right = has_right || right_mask[i] != 0;
            }
            auto left_source = b == 0 ? base : op.rotate(base, b);
            // Shift zero is a genuine Mlive CP, never a full-slot unity CP.
            auto term = op.cp(left_source, left_mask);
            if (has_right) {
                auto right_source = op.rotate(base, slots - layout.tokens + b);
                auto right = op.cp(right_source, right_mask);
                op.add_to(term, right); // Add at L10, before a single R4.
            }
            op.rescale(term, 4);
            return term;
        };
        std::vector<double> subtotal;
        for (std::size_t b = 0; b < baby; b += 2) {
            auto even = score_term(b), odd = score_term(b + 1);
            auto odd_gap = op.rotate(odd, wrong_score_rotation ? layout.head_dim : slots - layout.head_dim);
            op.add_to(even, odd_gap);
            auto product = op.cc(even, even_babies[b / 2]);
            op.relinearize(product);
            op.rescale(product, 2);
            if (b == 0) subtotal = std::move(product); else op.add_to(subtotal, product);
        }
        if (g != 0) subtotal = op.rotate(subtotal, layout.tile * baby * g);
        if (g == 0) output = std::move(subtotal); else op.add_to(output, subtotal);
    }
    auto shifted = op.rotate(output, layout.head_dim);
    op.add_to(output, shifted);
    if (out_counts) *out_counts = counts;
    return output;
}
} // namespace detail

// PARTIAL, fully defined all-slot result. P=[A0,B0,A1,B1], output=P+rot(D).
// Live=A+B=full AV. Gap=B_here+A_nexthead, and is generally NOT ZERO.
inline std::vector<double> partial_packed_input_bsgs(
    const std::vector<double> &scores, const std::vector<double> &values,
    const Layout &layout = {}, PackedInputCounts *counts = nullptr) {
    return detail::partial_packed_input_impl(scores, values, layout, counts);
}

// Independent indexed A/B oracle; no BSGS, rotations, selectors or packed
// intermediate arithmetic. Scores retain ALL T offsets, including the high half.
inline std::vector<double> partial_direct_oracle(
    const std::vector<double> &scores, const std::vector<double> &values,
    const Layout &layout = {}) {
    const auto slots = detail::validate_inputs(scores, values, layout);
    validate_values(values, layout);
    detail::validate_partial_layout(layout);
    std::vector<long double> a(slots), b(slots);
    for (std::size_t t = 0; t < layout.tokens; ++t)
        for (std::size_t h = 0; h < 2; ++h)
            for (std::size_t j = 0; j < layout.head_dim; ++j) {
                const auto i = t * layout.tile + h * layout.tokens + j;
                for (std::size_t d = 0; d < layout.tokens; ++d) {
                    const auto term = static_cast<long double>(scores[t * layout.tile + h * layout.tokens + d]) *
                        values[((t + d) % layout.tokens) * layout.tile + h * layout.tokens + j];
                    if (d % 2 == j % 2) a[i] += term; else b[i] += term;
                }
            }
    std::vector<double> output(slots);
    for (std::size_t t = 0; t < layout.tokens; ++t)
        for (std::size_t h = 0; h < 2; ++h)
            for (std::size_t j = 0; j < layout.head_dim; ++j) {
                const auto i = t * layout.tile + h * layout.tokens + j;
                const auto next = (h == 0 ? t * layout.tile + layout.tokens :
                                   ((t + 1) % layout.tokens) * layout.tile) + j;
                output[i] = static_cast<double>(a[i] + b[i]);
                output[i + layout.head_dim] = static_cast<double>(b[i] + a[next]);
            }
    return output;
}

struct ScaleSpec {
    // Physical Poseidon level L has L+1 Q primes, not L primes.
    std::size_t score_level = 10;
    std::size_t value_level = 8;
    std::size_t common_level = 6;
    std::size_t output_level = 4;
    double score_scale_log2 = 55.360150528219684;
    double value_scale_log2 = 55.35255783895382;
    double common_scale_log2 = 55.3727119134628;
    double output_scale_log2 = 55.4442212582195;
};

struct ScalePlan {
    double score_mask_scale_log2; // one genuine CP followed by four real drops
    double value_mask_scale_log2; // value selector CP followed by two real drops
    double legacy_mid_scale_log2; // old score mask CP(S10), Rescale2 -> S8
    double common_scale_log2;
    double output_scale_log2;
};

// The old score path is CPmask(Sscore), Rescale2, CPunity(Svalue), Rescale2.
// Its replacement derives a wide mask scale from the SAME four actual primes.
// Reject impossible/inconsistent declarations; never repair scale metadata.
inline ScalePlan derive_scale_plan(const std::vector<std::uint64_t> &ordered_q,
                                    const ScaleSpec &spec = {}) {
    if (spec.score_level < 6 || spec.score_level >= ordered_q.size() ||
        spec.value_level != spec.score_level - 2 ||
        spec.common_level != spec.score_level - 4 ||
        spec.output_level != spec.score_level - 6)
        throw std::invalid_argument("AV requires available physical levels L -> L-2 -> L-4 -> L-6");
    for (const auto q : ordered_q)
        if (q <= 1) throw std::invalid_argument("AV ordered Q moduli must be greater than one");
    for (const auto scale : {spec.score_scale_log2, spec.value_scale_log2,
                             spec.common_scale_log2, spec.output_scale_log2})
        if (!detail::representable_scale(scale))
            throw std::invalid_argument("AV scales must exponentiate to finite positive doubles");
    const auto l = spec.score_level;
    const auto drop2 = [&](std::size_t level) {
        return std::log2(static_cast<double>(ordered_q[level])) +
               std::log2(static_cast<double>(ordered_q[level - 1]));
    };
    const double mid = 2 * spec.score_scale_log2 - drop2(l);
    const double common = mid + spec.value_scale_log2 - drop2(l - 2);
    const double output = 2 * spec.common_scale_log2 - drop2(l - 4);
    if (!detail::representable_scale(mid) || !detail::representable_scale(common) ||
        !detail::representable_scale(output) ||
        std::abs(mid - spec.value_scale_log2) > 1e-8 ||
        std::abs(common - spec.common_scale_log2) > 1e-8 ||
        std::abs(output - spec.output_scale_log2) > 1e-8)
        throw std::invalid_argument("AV declared scales disagree with actual ordered-prime rescales");
    const double score_mask = spec.common_scale_log2 + drop2(l) + drop2(l - 2) - spec.score_scale_log2;
    const double value_mask = spec.common_scale_log2 + drop2(l - 2) - spec.value_scale_log2;
    if (!detail::representable_scale(score_mask, 1.0) ||
        !detail::representable_scale(value_mask, 1.0) ||
        !detail::representable_scale(spec.score_scale_log2 + score_mask) ||
        !detail::representable_scale(spec.value_scale_log2 + value_mask) ||
        !detail::representable_scale(2 * spec.score_scale_log2) ||
        !detail::representable_scale(2 * spec.value_scale_log2) ||
        !detail::representable_scale(2 * spec.common_scale_log2))
        throw std::invalid_argument("AV CP/CC encoding and product scales must be finite and usable");
    return {score_mask, value_mask, mid, spec.common_scale_log2, spec.output_scale_log2};
}

} // namespace bert::attention::av_reference
