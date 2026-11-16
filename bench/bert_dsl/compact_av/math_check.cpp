// Standalone scalar checks: no Poseidon, CUDA, GPU context, or external library.
#include "av_reference.h"
#include "diagnostic_profile.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
namespace ref = bert::attention::av_reference;
namespace profile = bert::attention::av_diagnostic;
std::size_t checks = 0, compared_slots = 0;
double maximum_error = 0;

void check(bool condition, const std::string &message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, const std::string &message,
          double absolute = 2e-11, double relative = 2e-12) {
    const auto error = std::abs(actual - expected);
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        error > absolute + relative * std::abs(expected)) {
        std::ostringstream text;
        text << std::setprecision(17) << message << ": actual=" << actual
             << ", expected=" << expected << ", error=" << error;
        check(false, text.str());
    }
    ++checks;
    maximum_error = std::max(maximum_error, error);
}
template <class Function>
void invalid(Function function, const std::string &message) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, message + " must throw std::invalid_argument");
}
void compare(const std::vector<double> &actual, const std::vector<double> &expected,
             const ref::Layout &layout, const std::string &label) {
    check(actual.size() == expected.size() && actual.size() == layout.tokens * layout.tile,
          label + " full ciphertext output size");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        near(actual[i], expected[i], label + " slot " + std::to_string(i));
        if (i % layout.tokens >= layout.head_dim)
            check(actual[i] == 0.0 && expected[i] == 0.0, label + " exact zero output gap");
    }
    compared_slots += actual.size();
}
struct Fixture { ref::Layout layout; std::vector<double> scores, values; };
Fixture fixture(std::size_t tokens, std::size_t dim, std::size_t baby = 8) {
    Fixture input{{tokens, 2 * tokens, dim, baby}, std::vector<double>(2 * tokens * tokens),
                  std::vector<double>(2 * tokens * tokens, 0.0)};
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t h = 0; h < 2; ++h)
            for (std::size_t j = 0; j < tokens; ++j) {
                const auto i = t * input.layout.tile + h * tokens + j;
                // Scores are deliberately nonzero in j>=D, too: those are
                // live cyclic offsets, NOT value-feature gap lanes.
                input.scores[i] = (h ? -0.17 : 0.23) + 0.003 * t - 0.007 * j +
                                  0.013 * ((5 * t + 3 * j + 7 * h) % 19);
                if (j < dim)
                    input.values[i] = (h ? 0.71 : -0.43) - 0.011 * t + 0.017 * j +
                                      0.019 * ((11 * t + 7 * j + 3 * h) % 23);
            }
    ref::validate_values(input.values, input.layout);
    return input;
}
void zero_head(std::vector<double> &values, const ref::Layout &l, std::size_t h) {
    for (std::size_t t = 0; t < l.tokens; ++t)
        for (std::size_t j = 0; j < l.tokens; ++j) values[t * l.tile + h * l.tokens + j] = 0;
}

void test_giants_and_offset_lanes() {
    const ref::Layout l{16, 32, 8, 8};
    std::vector<double> scores(512, 0.0), values(512, 0.0);
    scores[3 * l.tile + 11] = 2.5; // d11 = nonzero giant1/baby3, beyond D8
    values[14 * l.tile + 7] = -4;
    scores[15 * l.tile + l.tokens + 9] = -1.25; // giant1/baby1, token wrap
    values[8 * l.tile + l.tokens + 2] = 6;
    const auto output = ref::bsgs(scores, values, l);
    compare(output, ref::direct(scores, values, l), l, "T16 nonzero giants and wrapped keys");
    near(output[3 * l.tile + 7], -10, "known giant1/baby3 weighted value; NO normalization", 0, 0);
    near(output[15 * l.tile + l.tokens + 2], -7.5, "known wrapped second-head value", 0, 0);
    std::size_t nonzero = 0;
    for (const auto x : output) nonzero += x != 0;
    check(nonzero == 2, "giant fixture has exactly two nonzero output lanes");

    const ref::Layout full;
    scores.assign(32768, 0); values.assign(32768, 0);
    scores[127 * full.tile + 127] = 3; // d127 > D64, giant15/baby7
    values[126 * full.tile + 63] = 5;
    const auto all = ref::bsgs(scores, values, full);
    compare(all, ref::direct(scores, values, full), full, "T128 final giant/final score offset");
    near(all[127 * full.tile + 63], 15, "T128 d127 must not be dropped as a value gap", 0, 0);
}

void test_layouts() {
    for (const auto dim : {1u, 3u, 8u, 15u, 16u}) {
        const auto f = fixture(16, dim);
        compare(ref::bsgs(f.scores, f.values, f.layout), ref::direct(f.scores, f.values, f.layout),
                f.layout, "T16 D" + std::to_string(dim));
    }
    for (const auto baby : {1u, 2u, 4u, 8u, 16u}) {
        const auto f = fixture(16, 8, baby);
        compare(ref::bsgs(f.scores, f.values, f.layout), ref::direct(f.scores, f.values, f.layout),
                f.layout, "T16 baby" + std::to_string(baby));
    }
    for (const auto tokens : {1u, 4u}) {
        const auto f = fixture(tokens, 1);
        compare(ref::bsgs(f.scores, f.values, f.layout), ref::direct(f.scores, f.values, f.layout),
                f.layout, "Python min(baby,T) cap T" + std::to_string(tokens));
    }
}

void test_full_layout() {
    const auto f = fixture(128, 64);
    const auto &l = f.layout;
    check(f.scores.size() == 32768 && l.tile == 256 && l.tokens - l.head_dim == 64,
          "production layout has 32768 slots and two 64-live/64-gap heads per row");
    const auto expected = ref::direct(f.scores, f.values);
    const auto baseline = ref::bsgs(f.scores, f.values);
    compare(baseline, expected, l, "T128 all 32768 output slots");
    check(std::abs(baseline[0] - baseline[128]) > 0.1, "fixture heads are distinguishable");

    std::size_t rowmajor_differences = 0;
    for (std::size_t t = 0; t < l.tokens; ++t)
        for (std::size_t h = 0; h < 2; ++h)
            for (std::size_t j = 0; j < l.head_dim; ++j) {
                long double rowmajor = 0;
                for (std::size_t d = 0; d < l.tokens; ++d)
                    rowmajor += static_cast<long double>(f.scores[t * l.tile + h * l.tokens + d]) *
                                f.values[d * l.tile + h * l.tokens + j];
                const auto i = t * l.tile + h * l.tokens + j;
                if (t == 0) near(baseline[i], static_cast<double>(rowmajor), "row0 agrees with row-major AV");
                else rowmajor_differences += std::abs(baseline[i] - static_cast<double>(rowmajor)) > 1e-6;
            }
    check(rowmajor_differences > 3 * (2 * (l.tokens - 1) * l.head_dim) / 4,
          "cyclic AV differs from ordinary row-major AV at most nonzero token rows");

    for (std::size_t removed = 0; removed < 2; ++removed) {
        auto s = f.scores, v = f.values;
        if (removed == 0) zero_head(v, l, removed); else zero_head(s, l, removed);
        const auto isolated = ref::bsgs(s, v, l);
        compare(isolated, ref::direct(s, v, l), l, "T128 isolated head " + std::to_string(removed));
        for (std::size_t i = 0; i < isolated.size(); ++i)
            near(isolated[i], (i % l.tile) / l.tokens == removed ? 0 : baseline[i], "head isolation");
    }
    auto crossed = fixture(16, 8);
    zero_head(crossed.scores, crossed.layout, 1);
    zero_head(crossed.values, crossed.layout, 0);
    for (const auto x : ref::bsgs(crossed.scores, crossed.values, crossed.layout))
        check(x == 0, "opposite score/value heads must never cross-multiply");
}

void test_rejections() {
    const auto f = fixture(16, 8);
    const auto both = [&](const std::vector<double> &s, const std::vector<double> &v,
                          const ref::Layout &l, const std::string &label) {
        invalid([&] { (void)ref::direct(s, v, l); }, "direct " + label);
        invalid([&] { (void)ref::bsgs(s, v, l); }, "BSGS " + label);
    };
    for (const ref::Layout l : {ref::Layout{0, 0, 1, 8}, {16, 31, 8, 8}, {16, 32, 0, 8},
                                {16, 32, 17, 8}, {16, 32, 8, 0}, {16, 32, 8, 3},
                                {std::numeric_limits<std::size_t>::max(), 0, 1, 8},
                                {std::numeric_limits<std::size_t>::max() / 2,
                                 std::numeric_limits<std::size_t>::max() - 1, 1, 1}})
        both(f.scores, f.values, l, "invalid/overflowing layout");
    for (const bool alter_scores : {false, true}) {
        for (const int size_change : {-1, 1}) {
            auto s = f.scores, v = f.values;
            auto &bad = alter_scores ? s : v;
            bad.resize(size_change == -1 ? bad.size() - 1 : bad.size() + 1);
            both(s, v, f.layout, "wrong input slot count");
        }
        for (const auto x : {std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()})
            for (const auto i : {0u, 15u, 16u, 511u}) {
                auto s = f.scores, v = f.values;
                (alter_scores ? s : v)[i] = x;
                both(s, v, f.layout, "nonfinite live/offset/gap lane");
            }
    }
    for (const auto i : {8u, 31u, 511u}) {
        auto dirty = f.values;
        dirty[i] = 0.25;
        // General baseline AV is full cyclic arithmetic even for dirty gaps.
        // Only the separately requested strict value validator rejects them.
        const auto actual = ref::bsgs(f.scores, dirty, f.layout);
        const auto expected = ref::direct(f.scores, dirty, f.layout);
        bool nonzero_gap = false;
        for (std::size_t slot = 0; slot < actual.size(); ++slot) {
            near(actual[slot], expected[slot], "dirty-gap full baseline slot " + std::to_string(slot));
            if (slot % f.layout.tokens >= f.layout.head_dim && actual[slot] != 0)
                nonzero_gap = true;
        }
        compared_slots += actual.size();
        check(nonzero_gap, "dirty-gap baseline must retain nonzero full cyclic output gaps");
        bool diagnosed = false;
        try { ref::validate_values(dirty, f.layout); }
        catch (const std::invalid_argument &e) {
            const std::string text(e.what());
            diagnosed = text.find("gap") != std::string::npos && text.find(std::to_string(i)) != std::string::npos;
        }
        check(diagnosed, "dirty gap diagnosis identifies both gap and physical slot");
    }
    // A score in that same physical lane is legitimate and must be accepted.
    auto s = f.scores;
    s[511] = 7;
    compare(ref::bsgs(s, f.values, f.layout), ref::direct(s, f.values, f.layout),
            f.layout, "live score offset in a physical value-gap lane");
}

long double drop(std::size_t level, std::size_t count, const std::vector<std::uint64_t> &q) {
    long double product = 1;
    for (std::size_t i = 0; i < count; ++i) product *= q.at(level - i);
    return std::log2(product); // independent log of product, not sum of double logs
}
void test_scale_plans() {
    const auto q = profile::q52();
    check(profile::prime_pool().size() == 64 && q.size() == 52 && profile::p11().size() == 11,
          "exact q52-p11 pool sizes");
    check(q.back() == 256376833 && profile::p11().front() == 257949697 &&
          profile::p11().back() == 268042241, "Q and P preserve exact pool order and skip index52");
    const ref::ScaleSpec canonical;
    for (const auto level : {6u, 10u, 12u, 51u}) {
        ref::ScaleSpec spec;
        spec.score_level = level; spec.value_level = level - 2;
        spec.common_level = level - 4; spec.output_level = level - 6;
        spec.score_scale_log2 = profile::canonical_scale_log2(level);
        spec.value_scale_log2 = profile::canonical_scale_log2(level - 2);
        spec.common_scale_log2 = profile::canonical_scale_log2(level - 4);
        spec.output_scale_log2 = profile::canonical_scale_log2(level - 6);
        const auto before = spec;
        const auto plan = ref::derive_scale_plan(q, spec);
        const auto mask = spec.common_scale_log2 + drop(level, 4, q) - spec.score_scale_log2;
        const auto vmask = spec.common_scale_log2 + drop(level - 2, 2, q) - spec.value_scale_log2;
        near(plan.score_mask_scale_log2, static_cast<double>(mask), "wide mask uses all FOUR actual primes", 1e-12, 0);
        near(plan.value_mask_scale_log2, static_cast<double>(vmask), "value mask actual-prime formula", 1e-12, 0);
        near(plan.legacy_mid_scale_log2, spec.value_scale_log2, "old score mid S8", 1e-12, 0);
        near(static_cast<double>(spec.score_scale_log2 + mask - drop(level, 4, q)),
             spec.common_scale_log2, "new score CP + physical Rescale4", 1e-12, 0);
        near(static_cast<double>(2 * static_cast<long double>(spec.common_scale_log2) - drop(level - 4, 2, q)),
             spec.output_scale_log2, "CC/relin + physical Rescale2", 1e-12, 0);
        check(before.score_level == spec.score_level && before.value_level == spec.value_level &&
              before.common_level == spec.common_level && before.output_level == spec.output_level &&
              before.score_scale_log2 == spec.score_scale_log2 && before.value_scale_log2 == spec.value_scale_log2 &&
              before.common_scale_log2 == spec.common_scale_log2 && before.output_scale_log2 == spec.output_scale_log2,
              "scale helper must not change declarations");
    }
    const auto plan = ref::derive_scale_plan(q);
    check(plan.score_mask_scale_log2 > 110 && plan.score_mask_scale_log2 < 111,
          "canonical wide mask is about 110.7 bits, not 55/56 bits");
    near(plan.value_mask_scale_log2, canonical.value_scale_log2, "canonical value-mask scale S8", 1e-12, 0);
    near(profile::canonical_scale_log2(10), canonical.score_scale_log2, "canonical S10", 1e-12, 0);
    near(profile::canonical_scale_log2(8), canonical.value_scale_log2, "canonical S8", 1e-12, 0);
    near(profile::canonical_scale_log2(6), canonical.common_scale_log2, "canonical S6", 1e-12, 0);
    near(profile::canonical_scale_log2(4), canonical.output_scale_log2, "canonical S4", 1e-12, 0);
    invalid([&] { (void)profile::canonical_scale_log2(52); }, "absent canonical level");
    auto bad = canonical;
    bad.score_scale_log2 = bad.value_scale_log2 = bad.common_scale_log2 = bad.output_scale_log2 = 56;
    invalid([&] { (void)ref::derive_scale_plan(q, bad); }, "nominal 56-bit scale declarations");
    for (const auto field : {0u, 1u, 2u, 3u}) {
        bad = canonical;
        auto *scale = field == 0 ? &bad.score_scale_log2 : field == 1 ? &bad.value_scale_log2 :
                      field == 2 ? &bad.common_scale_log2 : &bad.output_scale_log2;
        *scale += 0.01;
        invalid([&] { (void)ref::derive_scale_plan(q, bad); }, "inconsistent declared scale");
        for (const auto x : {std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity(), 1024.0, -2000.0}) {
            *scale = x;
            invalid([&] { (void)ref::derive_scale_plan(q, bad); }, "nonrepresentable declared scale");
        }
    }
    for (const auto field : {0u, 1u, 2u, 3u}) {
        bad = canonical;
        auto *level = field == 0 ? &bad.score_level : field == 1 ? &bad.value_level :
                      field == 2 ? &bad.common_level : &bad.output_level;
        *level = std::numeric_limits<std::size_t>::max();
        invalid([&] { (void)ref::derive_scale_plan(q, bad); }, "invalid physical level");
    }
    invalid([&] { (void)ref::derive_scale_plan({}); }, "empty Q chain");
    auto changed_q = q;
    changed_q.resize(10);
    invalid([&] { (void)ref::derive_scale_plan(changed_q); }, "missing q10");
    for (const auto bad_prime : {0u, 1u}) {
        changed_q = q; changed_q[0] = bad_prime;
        invalid([&] { (void)ref::derive_scale_plan(changed_q); }, "invalid actual modulus");
    }
    changed_q = q; std::swap(changed_q[10], changed_q[8]);
    invalid([&] { (void)ref::derive_scale_plan(changed_q); }, "wrong ordered-prime drop groups");
    // Each stage scale is finite here, but the CC product's exp2 is not.
    changed_q.assign(11, (std::uint64_t{1} << 63) - 1);
    bad = canonical; bad.score_scale_log2 = 230;
    bad.value_scale_log2 = 2 * bad.score_scale_log2 - static_cast<double>(drop(10, 2, changed_q));
    bad.common_scale_log2 = 2 * bad.value_scale_log2 - static_cast<double>(drop(8, 2, changed_q));
    bad.output_scale_log2 = 2 * bad.common_scale_log2 - static_cast<double>(drop(6, 2, changed_q));
    invalid([&] { (void)ref::derive_scale_plan(changed_q, bad); }, "overflowing intermediate product scale");
}
} // namespace

int main() {
    try {
        test_giants_and_offset_lanes(); test_layouts(); test_full_layout();
        test_rejections(); test_scale_plans();
        std::cout << std::setprecision(17)
                  << "{\"cpu_only\":true,\"independent_direct_vs_original_bsgs\":true,"
                  << "\"full_slots\":32768,\"tokens\":128,\"head_dim\":64,"
                  << "\"head_isolation\":true,\"anti_rowmajor\":true,\"dirty_gap_diagnosis\":true,"
                  << "\"checks\":" << checks << ",\"compared_slots\":" << compared_slots
                  << ",\"max_abs_error\":" << maximum_error << "}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "AV scalar/math: " << e.what() << '\n'; return 1;
    }
}
