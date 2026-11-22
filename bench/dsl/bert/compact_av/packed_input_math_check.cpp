// CPU scalar validation of PackedInput2Partial, including its DEFINED dirty
// gaps. No claim of CKKS, GPU, complete attention or BERT equivalence.
#include "av_reference.h"
#include <iomanip>
#include <iostream>
#include <random>

namespace {
namespace ref = bert::attention::av_reference;
std::size_t checks = 0, compared_slots = 0, onehot_cases = 0;
double max_error = 0, max_normalized_error = 0;
ref::PackedInputCounts production_counts;
void check(bool ok, const std::string &message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
void near(double actual, double expected, const std::string &message,
          double tolerance = 2e-11) {
    ++checks;
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance * (1 + std::abs(expected)))
        throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    max_error = std::max(max_error, std::abs(actual - expected));
}
void compare(const std::vector<double> &a, const std::vector<double> &b,
             const std::string &label) {
    check(a.size() == b.size(), label + " size");
    for (std::size_t i = 0; i < a.size(); ++i)
        near(a[i], b[i], label + " slot " + std::to_string(i));
    compared_slots += a.size();
}
template <typename F> void rejects(F f, const std::string &label) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, label);
}
struct Fixture { ref::Layout layout; std::vector<double> scores, values; };
Fixture fixture(std::size_t tokens) {
    Fixture f{{tokens, 2 * tokens, tokens / 2, 8},
              std::vector<double>(2 * tokens * tokens), std::vector<double>(2 * tokens * tokens)};
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t h = 0; h < 2; ++h)
            for (std::size_t j = 0; j < tokens; ++j) {
                const auto i = t * f.layout.tile + h * tokens + j;
                // ALL offsets, including 64..127, deliberately nonzero.
                f.scores[i] = (h ? -0.43 : 0.27) + 0.003 * t - 0.002 * j +
                              0.013 * ((3 * t + 7 * j + h) % 17);
                if (j < tokens / 2)
                    f.values[i] = (h ? 0.81 : -0.17) - 0.004 * t + 0.005 * j +
                                  0.019 * ((11 * t + 5 * j + h) % 23);
            }
    return f;
}
void check_counts(const ref::PackedInputCounts &c, std::size_t tokens) {
    const bool full = tokens == 128;
    check(c.cp == (full ? 319u : 39u), "actual CP callsites");
    check(c.cc == tokens / 2 && c.relin == tokens / 2, "actual paired CC/relin callsites");
    check(c.rescale == (full ? 193u : 25u) && c.r2 == tokens / 2 + 1 &&
          c.r4 == tokens, "actual batched RSC/R2/R4 callsites");
    check(c.physical_drops == (full ? 642u : 82u), "actual physical prime drops");
    check(c.rotations == (full ? 296u : 44u), "actual rotation callsites");
    check(c.additions == (full ? 319u : 39u), "actual add callsites");
    check(c.hoists == (full ? 19u : 5u), "actual shared-source hoist families");
}
void test_layout(std::size_t tokens) {
    const auto f = fixture(tokens);
    const auto &l = f.layout;
    ref::PackedInputCounts c;
    const auto partial = ref::partial_packed_input_bsgs(f.scores, f.values, l, &c);
    const auto oracle = ref::partial_direct_oracle(f.scores, f.values, l);
    const auto full = ref::direct(f.scores, f.values, l);
    compare(partial, oracle, "all-slot independent A/B partial oracle T" + std::to_string(tokens));
    compare(ref::bsgs(f.scores, f.values, l), full, "unchanged clean baseline");
    check_counts(c, tokens);
    if (tokens == 128) production_counts = c;
    std::size_t dirty = 0, high_half_nonzero = 0;
    for (std::size_t i = 0; i < partial.size(); ++i) {
        if (i % tokens < l.head_dim) near(partial[i], full[i], "partial live equals full AV");
        else {
            dirty += std::abs(partial[i]) > 1e-8;
            high_half_nonzero += f.scores[i] != 0;
            check(full[i] == 0, "baseline clean gaps remain zero");
        }
    }
    check(dirty > partial.size() / 3, "partial genuinely has defined dirty gaps");
    check(high_half_nonzero == partial.size() / 2, "all high-half score offsets retained");
    for (const bool mutate_diagonal : {false, true}) {
        const auto wrong = ref::detail::partial_packed_input_impl(
            f.scores, f.values, l, nullptr, mutate_diagonal, !mutate_diagonal);
        double error = 0;
        for (std::size_t i = 0; i < wrong.size(); ++i)
            error = std::max(error, std::abs(wrong[i] - oracle[i]));
        check(error > 1e-3, mutate_diagonal ? "wrong Dpack -D must fail" : "wrong Cpack +D must fail");
    }
    // Partial gaps intentionally couple adjacent heads; LIVE lanes must not.
    auto one_head_values = f.values;
    for (std::size_t i = 0; i < one_head_values.size(); ++i)
        if ((i / tokens) % 2 == 1) one_head_values[i] = 0;
    const auto isolated = ref::partial_packed_input_bsgs(f.scores, one_head_values, l);
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t j = 0; j < l.head_dim; ++j) {
            near(isolated[t * l.tile + j], partial[t * l.tile + j], "head0 live isolation");
            check(isolated[t * l.tile + tokens + j] == 0, "head1 live isolation");
        }
}
void test_onehot(std::size_t tokens) {
    const ref::Layout l{tokens, 2 * tokens, tokens / 2, 8};
    const std::size_t slots = tokens * l.tile;
    for (const auto d : {std::size_t{0}, tokens / 2, tokens / 2 + 1, tokens - 1})
        for (const auto j : {std::size_t{0}, l.head_dim - 1})
            for (const auto t : {std::size_t{0}, tokens - 1})
                for (std::size_t h = 0; h < 2; ++h) {
                    std::vector<double> s(slots), v(slots);
                    s[t * l.tile + h * tokens + d] = 3;
                    v[((t + d) % tokens) * l.tile + h * tokens + j] = 5;
                    const auto partial = ref::partial_packed_input_bsgs(s, v, l);
                    compare(partial, ref::partial_direct_oracle(s, v, l), "onehot all slots");
                    check(partial[t * l.tile + h * tokens + j] == 15,
                          "onehot key0/D/D+1/T-1 live including final-token wrap");
                    ++onehot_cases;
                }
}
std::vector<double> merge_pair(const std::vector<double> &a, const std::vector<double> &b,
                               const ref::Layout &l) {
    auto result = a;
    ref::detail::add(result, ref::detail::rotate_left(b, b.size() - l.head_dim));
    return result;
}
void test_normalization(std::size_t tokens, bool constant) {
    auto f = fixture(tokens);
    const auto &l = f.layout;
    std::mt19937_64 rng(0x3a11 + tokens);
    std::uniform_real_distribution<double> score_distribution(0.7, 1.3), value_distribution(-0.8, 0.9);
    for (std::size_t i = 0; i < f.scores.size(); ++i) {
        f.scores[i] = constant ? 1 : score_distribution(rng);
        f.values[i] = i % tokens < l.head_dim ? (constant ? 1 : value_distribution(rng)) : 0;
    }
    const auto partial = ref::partial_packed_input_bsgs(f.scores, f.values, l);
    const auto full = ref::direct(f.scores, f.values, l);
    std::vector<double> ordinary(f.scores.size()), masked(f.scores.size()), bad(f.scores.size());
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t h = 0; h < 2; ++h) {
            double denominator = 0, half_denominator = 0;
            for (std::size_t d = 0; d < tokens; ++d) {
                denominator += f.scores[t * l.tile + h * tokens + d];
                if (d < l.head_dim) half_denominator += f.scores[t * l.tile + h * tokens + d];
            }
            check(denominator > half_denominator + tokens / 4.0, "denominator uses ALL keys, never just D");
            for (std::size_t j = 0; j < tokens; ++j) {
                const auto i = t * l.tile + h * tokens + j;
                const double m = double(j < l.head_dim);
                const double n = double(tokens);
                double r = denominator * (-1 / (n * n)) + 2 / n;
                // Genuine MASKED SEED multiplication, not post-hoc zeroing.
                double mr = denominator * (-m / (n * n)) + 2 * m / n;
                for (int k = 0; k < 2; ++k) {
                    const double correction = 2 - denominator * mr;
                    if (m == 0) check(correction == 2 && mr == 0,
                                      "gap correction is TWO; OUTER r preserves zero");
                    r = r * (2 - denominator * r);
                    mr = mr * correction;
                }
                if (m == 0) check(mr == 0, "three-iteration maskedRecip3 has exact ideal gap zero");
                else near(mr, r, "maskedRecip3 live matches ordinary reciprocal");
                ordinary[i] = full[i] * r;
                masked[i] = partial[i] * mr; // Existing unique consumer CC clears gaps.
                bad[i] = partial[i] * r;
                max_normalized_error = std::max(max_normalized_error, std::abs(masked[i] - ordinary[i]));
            }
        }
    compare(masked, ordinary, "full-slot normalized partial equals ordinary full AV");
    compare(merge_pair(masked, masked, l), merge_pair(ordinary, ordinary, l),
            "full-slot normalized context after existing merge_heads");
    if (constant) {
        const auto wrong_merge = merge_pair(bad, bad, l);
        const auto good_merge = merge_pair(masked, masked, l);
        for (std::size_t i = 0; i < good_merge.size(); ++i) {
            near(good_merge[i], 1, "constant masked context merges to one", 0);
            near(wrong_merge[i], 2, "unmasked constant negative control doubles old merge", 0);
        }
    }
}
void test_invalid() {
    auto f = fixture(16);
    for (const auto i : {8u, 31u, 511u}) {
        auto dirty = f.values;
        dirty[i] = 0.01;
        rejects([&] { ref::partial_packed_input_bsgs(f.scores, dirty, f.layout); }, "packed rejects dirty V gaps");
        rejects([&] { ref::partial_direct_oracle(f.scores, dirty, f.layout); }, "oracle rejects dirty V gaps");
    }
    for (const auto l : {ref::Layout{16, 32, 7, 8}, ref::Layout{16, 32, 8, 1},
                         ref::Layout{16, 31, 8, 8}, ref::Layout{16, 32, 8, 3}}) {
        rejects([&] { ref::partial_packed_input_bsgs(f.scores, f.values, l); }, "packed invalid layout");
        rejects([&] { ref::partial_direct_oracle(f.scores, f.values, l); }, "oracle invalid layout");
    }
    f.scores[511] = std::numeric_limits<double>::infinity();
    rejects([&] { ref::partial_packed_input_bsgs(f.scores, f.values, f.layout); }, "high-half score must be finite");
    rejects([&] { ref::partial_direct_oracle(f.scores, f.values, f.layout); }, "oracle high-half score must be finite");
}
} // namespace
int main() {
    try {
        for (const std::size_t t : {16u, 128u}) {
            test_layout(t);
            test_onehot(t);
            test_normalization(t, false);
            test_normalization(t, true);
        }
        test_invalid();
        const auto &c = production_counts;
        std::cout << std::setprecision(17)
                  << "{\"cpu_only\":true,\"scope\":\"scalar_defined_partial_not_ckks_gpu_or_bert\","
                  << "\"full_slots\":32768,\"independent_parity_oracle\":true,\"dirty_partial_gaps\":true,"
                  << "\"all128_keys\":true,\"rotation_sign_mutations_rejected\":true,\"maskedRecip3_fullslot_equivalence\":true,"
                  << "\"unmasked_constant_merge_doubles\":true,\"onehot_cases\":" << onehot_cases
                  << ",\"cp\":" << c.cp << ",\"cc\":" << c.cc << ",\"relin\":" << c.relin
                  << ",\"rsc\":" << c.rescale << ",\"r2\":" << c.r2 << ",\"r4\":" << c.r4
                  << ",\"physical_drops\":" << c.physical_drops << ",\"rot\":" << c.rotations
                  << ",\"add\":" << c.additions << ",\"hoist\":" << c.hoists
                  << ",\"checks\":" << checks << ",\"compared_slots\":" << compared_slots
                  << ",\"max_abs_error\":" << max_error << ",\"normalized_max_abs_error\":" << max_normalized_error
                  << ",\"ckks_caveat\":\"masked seed PT approximately 55.43 bits minus 14-bit coefficient gives approximately 41.43-bit amplitude budget; encrypted gaps are not exact zero and noise can differ\"}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "PackedInput2Partial scalar: " << e.what() << '\n';
        return 1;
    }
}
