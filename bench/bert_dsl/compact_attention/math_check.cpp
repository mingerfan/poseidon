// Pure scalar/math metadata checks: no CUDA, GPU context, or Poseidon headers.
#include "qk_reference.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using bert::attention::QkSpec;
namespace ref = bert::attention::reference;

std::size_t checks = 0;
std::size_t compared_slots = 0;
double max_absolute_error = 0.0;
double paired_oracle_max_error = 0.0;
double paired_baseline_max_error = 0.0;
ref::PairedScheduleCounts production_paired_counts;
double packed_input_oracle_max_error = 0.0;
double packed_input_baseline_max_error = 0.0;
double packed_input_paired_max_error = 0.0;
ref::PackedInputScheduleCounts production_packed_input_counts;

void check(bool condition, const std::string &message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, const std::string &message,
          double absolute = 2e-11, double relative = 2e-12) {
    const double error = std::abs(actual - expected);
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        error > absolute + relative * std::abs(expected)) {
        std::ostringstream detail;
        detail << std::setprecision(17) << message << ": actual=" << actual
               << ", expected=" << expected << ", error=" << error;
        check(false, detail.str());
    }
    ++checks;
    max_absolute_error = std::max(max_absolute_error, error);
}

template <class Function>
void invalid(Function function, const std::string &message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, message + " must throw std::invalid_argument");
}

void compare_all(const std::vector<double> &actual, const std::vector<double> &expected,
                 std::size_t tokens, const std::string &label) {
    check(actual.size() == expected.size(), label + " output size");
    check(actual.size() == 2 * tokens * tokens, label + " full score block size");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto t = i / (2 * tokens);
        const auto h = (i % (2 * tokens)) / tokens;
        const auto d = i % tokens;
        near(actual[i], expected[i], label + " [t=" + std::to_string(t) +
             ",h=" + std::to_string(h) + ",d=" + std::to_string(d) + "]");
    }
    compared_slots += actual.size();
}

struct Fixture {
    std::size_t tokens;
    std::size_t head_dim;
    std::vector<double> query;
    std::vector<double> key;
};

Fixture fixture(std::size_t tokens, std::size_t head_dim) {
    Fixture input{tokens, head_dim, std::vector<double>(2 * tokens * tokens, 0.0),
                  std::vector<double>(2 * tokens * tokens, 0.0)};
    const auto tile = 2 * tokens;
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t j = 0; j < head_dim; ++j) {
            const auto first = t * tile + j;
            const auto second = first + tokens;
            // Different, nonsymmetric functions in both heads and both inputs.
            // In particular, key-token identity cannot be replaced by query t.
            input.query[first] = 0.15 + 0.013 * t - 0.017 * j + 0.021 * ((3 * t + 5 * j) % 19);
            input.key[first] = -0.4 + 0.009 * t + 0.023 * j + 0.033 * ((7 * t + 11 * j) % 23);
            input.query[second] = 1.1 - 0.011 * t + 0.019 * j - 0.027 * ((11 * t + 3 * j) % 17);
            input.key[second] = -0.7 + 0.021 * t - 0.015 * j + 0.029 * ((5 * t + 13 * j) % 29);
        }
    }
    return input;
}

void check_gap_fixture(const Fixture &input) {
    std::size_t gaps = 0;
    for (std::size_t t = 0; t < input.tokens; ++t) {
        for (std::size_t h = 0; h < 2; ++h) {
            for (std::size_t j = input.head_dim; j < input.tokens; ++j) {
                const auto slot = t * 2 * input.tokens + h * input.tokens + j;
                check(input.query[slot] == 0.0 && input.key[slot] == 0.0,
                      "fixture must contain genuine zero head gaps");
                ++gaps;
            }
        }
    }
    check(gaps == 2 * input.tokens * (input.tokens - input.head_dim), "gap count");
}

void test_giant_compensation() {
    constexpr std::size_t tokens = 16, dimension = 8, tile = 32;
    std::vector<double> query(tokens * tile, 0.0), key(query.size(), 0.0);
    query[3 * tile + 7] = 2.5;
    key[14 * tile + 7] = -4.0; // cyclic distance 11 = giant 1, baby 3
    query[15 * tile + tokens + 2] = -1.25;
    key[8 * tile + tokens + 2] = 6.0; // distance 9, wraps token boundary
    const auto expected = ref::direct_oracle(query, key, tokens, dimension);
    const auto actual = ref::cyclic_qk(query, key, tokens, dimension, 8);
    compare_all(actual, expected, tokens, "T16 nonzero giant/wrap compensation");
    near(actual[3 * tile + 11], -10.0 / std::sqrt(8.0), "giant1/baby3 known score");
    near(actual[15 * tile + tokens + 9], -7.5 / std::sqrt(8.0), "giant1/baby1 wrapped score");
    std::size_t nonzero = 0;
    for (const auto value : actual) {
        nonzero += value != 0.0;
    }
    check(nonzero == 2, "spike fixture must have exactly two nonzero scores");
}

void test_small_layouts() {
    // Also test non-power-of-two D: real zero gaps pad the doubling reduction.
    for (const std::size_t dimension : {1, 3, 5, 8, 15, 16}) {
        const auto input = fixture(16, dimension);
        check_gap_fixture(input);
        const auto expected = ref::direct_oracle(input.query, input.key, 16, dimension);
        compare_all(ref::cyclic_qk(input.query, input.key, 16, dimension), expected, 16,
                    "T16 D" + std::to_string(dimension));
    }
    const auto input = fixture(16, 8);
    const auto expected = ref::direct_oracle(input.query, input.key, 16, 8);
    for (const std::size_t baby : {1, 2, 4, 8, 16}) {
        compare_all(ref::cyclic_qk(input.query, input.key, 16, 8, baby), expected, 16,
                    "T16 baby" + std::to_string(baby));
    }
    const auto tiny = fixture(4, 2);
    compare_all(ref::cyclic_qk(tiny.query, tiny.key, 4, 2),
                ref::direct_oracle(tiny.query, tiny.key, 4, 2), 4, "Python min(8,T) baby cap");
}

void zero_head(std::vector<double> &values, std::size_t tokens, std::size_t head) {
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t j = 0; j < tokens; ++j) {
            values[t * 2 * tokens + head * tokens + j] = 0.0;
        }
    }
}

void test_full_layout_and_isolation() {
    constexpr std::size_t tokens = 128, dimension = 64, tile = 256;
    const auto input = fixture(tokens, dimension);
    check_gap_fixture(input);
    const auto expected = ref::direct_oracle(input.query, input.key);
    const auto baseline = ref::cyclic_qk(input.query, input.key);
    compare_all(baseline, expected, tokens, "T128 all 32768 cyclic scores");
    check(std::abs(baseline[0] - baseline[tokens]) > 0.1, "fixture heads are observably distinct");

    // Compare against a truly conventional [query,key] row-major QK matrix.
    // The same output slot d represents key (t+d)%T, NOT conventional key d.
    std::size_t rowmajor_differences = 0;
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t h = 0; h < 2; ++h) {
            for (std::size_t d = 0; d < tokens; ++d) {
                long double rowmajor = 0.0L;
                for (std::size_t j = 0; j < dimension; ++j) {
                    rowmajor += static_cast<long double>(input.query[t * tile + h * tokens + j]) *
                                static_cast<long double>(input.key[d * tile + h * tokens + j]);
                }
                rowmajor /= std::sqrt(static_cast<long double>(dimension));
                const auto slot = t * tile + h * tokens + d;
                if (t == 0) {
                    near(baseline[slot], static_cast<double>(rowmajor), "row0 cyclic/rowmajor agreement");
                } else if (std::abs(baseline[slot] - static_cast<double>(rowmajor)) > 1e-6) {
                    ++rowmajor_differences;
                }
            }
        }
    }
    check(rowmajor_differences > 3 * (2 * tokens * (tokens - 1)) / 4,
          "cyclic scores must differ from row-major QK at most nonzero query rows");

    // Zeroing either head in one operand must remove only that score head.
    for (std::size_t removed_head = 0; removed_head < 2; ++removed_head) {
        auto query = input.query;
        auto key = input.key;
        if (removed_head == 0) {
            zero_head(key, tokens, removed_head);
        } else {
            zero_head(query, tokens, removed_head);
        }
        const auto isolated = ref::cyclic_qk(query, key);
        compare_all(isolated, ref::direct_oracle(query, key), tokens,
                    "T128 isolated head " + std::to_string(removed_head));
        for (std::size_t t = 0; t < tokens; ++t) {
            for (std::size_t h = 0; h < 2; ++h) {
                for (std::size_t d = 0; d < tokens; ++d) {
                    const auto slot = t * tile + h * tokens + d;
                    near(isolated[slot], h == removed_head ? 0.0 : baseline[slot], "head isolation");
                }
            }
        }
    }

    auto crossed = fixture(16, 8);
    zero_head(crossed.query, 16, 1);
    zero_head(crossed.key, 16, 0);
    const auto scores = ref::cyclic_qk(crossed.query, crossed.key, 16, 8);
    for (const auto score : scores) {
        check(score == 0.0, "opposite Q/K heads must never cross-multiply");
    }
}

void compare_paired(const Fixture &input, std::size_t baby, const std::string &label,
                    ref::PairedScheduleCounts *counts = nullptr) {
    const auto paired = ref::paired_cyclic_qk(input.query, input.key, input.tokens,
                                              input.head_dim, baby, counts);
    const auto baseline = ref::cyclic_qk(input.query, input.key, input.tokens, input.head_dim, baby);
    const auto oracle = ref::direct_oracle(input.query, input.key, input.tokens, input.head_dim);
    compare_all(paired, oracle, input.tokens, label + " independent oracle");
    compare_all(paired, baseline, input.tokens, label + " unchanged baseline");
    for (std::size_t i = 0; i < paired.size(); ++i) {
        paired_oracle_max_error = std::max(paired_oracle_max_error, std::abs(paired[i] - oracle[i]));
        paired_baseline_max_error = std::max(paired_baseline_max_error, std::abs(paired[i] - baseline[i]));
    }
}

void test_paired_schedule() {
    const auto full = fixture(128, 64);
    check_gap_fixture(full);
    compare_paired(full, 8, "paired T128 all 32768 slots", &production_paired_counts);
    const auto &counts = production_paired_counts;
    check(counts.cc_multiply_calls == 128, "paired actual scalar CC sites");
    check(counts.cp_multiply_calls == 128, "paired actual scalar CP mask sites");
    check(counts.rotation_calls == 597, "paired actual nonidentity rotation sites");
    check(counts.add_calls == 575, "paired actual addition sites, no zero-initializer adds");
    check(counts.abstract_rescale2_calls == 129, "paired abstract CC plus final R2 schedule");
    check(counts.logical_shared_source_batches == 2, "paired logical source batches, not real hoists");
    const auto defaults = ref::paired_cyclic_qk(full.query, full.key);
    compare_all(defaults, ref::direct_oracle(full.query, full.key), 128, "paired default API");

    const auto small = fixture(16, 8);
    for (const std::size_t baby : {2, 4, 8, 16}) {
        compare_paired(small, baby, "paired T16 baby" + std::to_string(baby));
    }
    const auto tiny = fixture(4, 2);
    compare_paired(tiny, 8, "paired min(8,T) baby cap");

    // Sparse known answers catch odd P1 packing, g=1 compensation, full-vector
    // wraparound, the final production offset b=127, and distinct head signs.
    for (const std::size_t tokens : {16, 128}) {
        const auto dimension = tokens / 2, tile = 2 * tokens;
        Fixture spikes{tokens, dimension, std::vector<double>(tokens * tile, 0.0),
                       std::vector<double>(tokens * tile, 0.0)};
        spikes.query[dimension - 1] = 2.5;
        spikes.key[(tokens - 1) * tile + dimension - 1] = -4.0;
        spikes.query[(tokens - 1) * tile + tokens] = -1.25;
        spikes.key[(tokens - 2) * tile + tokens] = -6.0;
        compare_paired(spikes, 8, "paired final odd baby/wrapped heads T" + std::to_string(tokens));
        const auto scores = ref::paired_cyclic_qk(spikes.query, spikes.key, tokens, dimension);
        near(scores[tokens - 1], -10.0 / std::sqrt(double(dimension)), "paired last offset head0");
        near(scores[(tokens - 1) * tile + tokens + tokens - 1],
             7.5 / std::sqrt(double(dimension)), "paired last offset wrapped head1");
        std::size_t nonzero = 0;
        for (const auto value : scores) nonzero += value != 0.0;
        check(nonzero == 2, "paired sparse fixture keeps heads isolated");
    }
    for (std::size_t head = 0; head < 2; ++head) {
        auto isolated = small;
        zero_head(head == 0 ? isolated.query : isolated.key, 16, head);
        compare_paired(isolated, 8, "paired isolated head " + std::to_string(head));
    }
}

void test_paired_input_rejection() {
    const auto input = fixture(16, 8);
    const auto reject = [](const std::vector<double> &q, const std::vector<double> &k,
                           std::size_t tokens, std::size_t dim, const std::string &label) {
        invalid([&] { (void)ref::paired_cyclic_qk(q, k, tokens, dim); }, "paired " + label);
    };
    reject(input.query, input.key, 0, 8, "zero T");
    reject(input.query, input.key, 16, 0, "zero D");
    reject(input.query, input.key, 16, 17, "D greater than T");
    reject({}, {}, std::numeric_limits<std::size_t>::max(), 1, "tile overflow");
    reject({}, {}, std::numeric_limits<std::size_t>::max() / 2, 1, "slot overflow");
    reject({}, {}, 16, 8, "empty input");
    const auto illegal = fixture(16, 7);
    reject(illegal.query, illegal.key, 16, 7, "D not T/2");
    const auto nonpower = fixture(12, 6);
    reject(nonpower.query, nonpower.key, 12, 6, "non-power-of-two reduction");
    const auto odd = fixture(15, 7);
    reject(odd.query, odd.key, 15, 7, "odd T");
    for (const std::size_t baby : {0, 1, 3, 6, 17}) {
        invalid([&] { (void)ref::paired_cyclic_qk(input.query, input.key, 16, 8, baby); },
                "paired zero/odd/nondividing baby");
    }
    for (const bool alter_query : {false, true}) {
        for (const bool shorten : {false, true}) {
            auto q = input.query, k = input.key;
            auto &bad = alter_query ? q : k;
            bad.resize(shorten ? bad.size() - 1 : bad.size() + 1);
            reject(q, k, 16, 8, "short/long operand");
        }
        for (const std::size_t slot : {std::size_t{0}, std::size_t{16}, std::size_t{511}}) {
            for (const double value : {std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::infinity(),
                                       -std::numeric_limits<double>::infinity()}) {
                auto q = input.query, k = input.key;
                (alter_query ? q : k)[slot] = value;
                reject(q, k, 16, 8, "nonfinite live/gap lane");
            }
        }
        // Explicitly reject dirty message gaps; paired/baseline equality is
        // intentionally NOT tested or promised outside the zero-gap domain.
        for (const std::size_t slot : {std::size_t{8}, std::size_t{24}, std::size_t{511}}) {
            for (const double value : {-0.25, 0.25}) {
                auto q = input.query, k = input.key;
                (alter_query ? q : k)[slot] = value;
                reject(q, k, 16, 8, "nonzero dirty head gap");
            }
        }
    }
}

// Independent loop-cardinality formulas, not totals supplied by the schedule.
void check_packed_input_counts(const ref::PackedInputScheduleCounts &counts,
                               std::size_t tokens, std::size_t dimension, std::size_t baby) {
    baby = std::min(baby, tokens);
    const auto giants = tokens / baby, pairs = baby / 2, terms = giants * pairs;
    std::size_t reductions = 0;
    for (auto width = dimension; width > 1; width /= 2) ++reductions;
    const auto rotations = 2 + (giants - 1) + (pairs - 1) + terms * reductions +
                           (dimension > 1 ? terms : 0) + giants * (pairs - 1) + (giants - 1);
    const auto additions = 2 + terms * reductions + terms + giants * (pairs - 1) + (giants - 1);
    check(counts.cc_multiply_calls == terms, "packed-input counted actual scalar CC sites");
    check(counts.cp_multiply_calls == 2 * terms, "packed-input counted actual scalar CP sites");
    check(counts.rotation_calls == rotations, "packed-input counted nonidentity full-vector rotations");
    check(counts.add_calls == additions, "packed-input counted actual adds, not zero-initializer adds");
    check(counts.abstract_rescale2_calls == terms + 1, "packed-input ABSTRACT R2 schedule only");
    check(counts.logical_shared_source_batches == 2, "packed-input LOGICAL batches, not real hoists");
}

void compare_packed_input(const Fixture &input, std::size_t baby, const std::string &label,
                          ref::PackedInputScheduleCounts *production = nullptr) {
    ref::PackedInputScheduleCounts counts;
    const auto packed = ref::packed_input_cyclic_qk(input.query, input.key, input.tokens,
                                                    input.head_dim, baby, &counts);
    const auto baseline = ref::cyclic_qk(input.query, input.key, input.tokens, input.head_dim, baby);
    const auto paired = ref::paired_cyclic_qk(input.query, input.key, input.tokens, input.head_dim, baby);
    const auto oracle = ref::direct_oracle(input.query, input.key, input.tokens, input.head_dim);
    compare_all(packed, oracle, input.tokens, label + " independent oracle");
    compare_all(packed, baseline, input.tokens, label + " unchanged baseline");
    compare_all(packed, paired, input.tokens, label + " unchanged paired2");
    check_packed_input_counts(counts, input.tokens, input.head_dim, baby);
    if (production) *production = counts;
    for (std::size_t i = 0; i < packed.size(); ++i) {
        packed_input_oracle_max_error = std::max(packed_input_oracle_max_error, std::abs(packed[i] - oracle[i]));
        packed_input_baseline_max_error = std::max(packed_input_baseline_max_error, std::abs(packed[i] - baseline[i]));
        packed_input_paired_max_error = std::max(packed_input_paired_max_error, std::abs(packed[i] - paired[i]));
    }
}

void test_packed_input_schedule() {
    const auto full = fixture(128, 64);
    check_gap_fixture(full);
    compare_packed_input(full, 8, "packed-input T128 all 32768 slots", &production_packed_input_counts);
    const auto &counts = production_packed_input_counts;
    // Fixed production expectations independently check the general formulas.
    check(counts.cc_multiply_calls == 64, "packed-input production 64 actual scalar CCs");
    check(counts.cp_multiply_calls == 128, "packed-input production 128 actual scalar CPs");
    check(counts.rotation_calls == 531, "packed-input production 531 actual scalar rotations");
    check(counts.add_calls == 513, "packed-input production 513 actual scalar additions");
    check(counts.abstract_rescale2_calls == 65, "packed-input production 65 ABSTRACT R2s");
    check(counts.logical_shared_source_batches == 2, "packed-input production 2 LOGICAL batches");
    compare_all(ref::packed_input_cyclic_qk(full.query, full.key),
                ref::direct_oracle(full.query, full.key), 128, "packed-input default API");
    check(std::abs(ref::direct_oracle(full.query, full.key)[0] -
                   ref::direct_oracle(full.query, full.key)[128]) > 0.1,
          "packed-input fixture has observably distinct heads");

    const auto small = fixture(16, 8);
    check_gap_fixture(small);
    for (const std::size_t baby : {2, 4, 8, 16}) {
        compare_packed_input(small, baby, "packed-input T16 baby" + std::to_string(baby));
    }
    compare_packed_input(fixture(4, 2), 8, "packed-input min(8,T) baby cap");
    compare_packed_input(fixture(2, 1), 8, "packed-input D1 identity second-mask rotation");

    // Explicit nonzero giant, odd baby, full-vector wrap, final offset127,
    // both feature extremes, and distinct signed heads (not replicated heads).
    for (const std::size_t tokens : {16, 128}) {
        const auto dimension = tokens / 2, tile = 2 * tokens;
        Fixture spikes{tokens, dimension, std::vector<double>(tokens * tile, 0.0),
                       std::vector<double>(tokens * tile, 0.0)};
        spikes.query[3 * tile + dimension - 1] = 2.5;
        spikes.key[14 * tile + dimension - 1] = -4.0; // d11, g1 at baby8
        spikes.query[(tokens - 1) * tile + tokens] = -1.25;
        spikes.key[(tokens - 2) * tile + tokens] = -6.0; // final d=T-1, wraps
        compare_packed_input(spikes, 8, "packed-input giant/last-offset/wrap T" + std::to_string(tokens));
        const auto scores = ref::packed_input_cyclic_qk(spikes.query, spikes.key, tokens, dimension);
        near(scores[3 * tile + 11], -10.0 / std::sqrt(double(dimension)), "packed-input giant1/baby3");
        near(scores[(tokens - 1) * tile + tokens + tokens - 1],
             7.5 / std::sqrt(double(dimension)), "packed-input final offset wrapped head1");
        std::size_t nonzero = 0;
        for (const auto value : scores) nonzero += value != 0.0;
        check(nonzero == 2, "packed-input sparse signed heads stay isolated");
    }
    for (std::size_t head = 0; head < 2; ++head) {
        auto isolated = small;
        zero_head(head == 0 ? isolated.query : isolated.key, 16, head);
        compare_packed_input(isolated, 8, "packed-input isolated head " + std::to_string(head));
    }
    auto crossed = small;
    zero_head(crossed.query, 16, 1);
    zero_head(crossed.key, 16, 0);
    const auto crossed_scores = ref::packed_input_cyclic_qk(crossed.query, crossed.key, 16, 8);
    for (const auto value : crossed_scores) check(value == 0.0, "packed-input cannot cross opposite heads");
}

void test_packed_input_rejection() {
    const auto input = fixture(16, 8);
    const auto reject = [](const std::vector<double> &q, const std::vector<double> &k,
                           std::size_t tokens, std::size_t dim, const std::string &label) {
        invalid([&] { (void)ref::packed_input_cyclic_qk(q, k, tokens, dim); }, "packed-input " + label);
    };
    reject(input.query, input.key, 0, 8, "zero T");
    reject(input.query, input.key, 16, 0, "zero D");
    reject(input.query, input.key, 16, 17, "D greater than T");
    reject({}, {}, std::numeric_limits<std::size_t>::max(), 1, "tile overflow");
    reject({}, {}, std::numeric_limits<std::size_t>::max() / 2, 1, "slot overflow");
    reject({}, {}, 16, 8, "empty operands");
    for (const auto &illegal : {fixture(16, 7), fixture(12, 6), fixture(15, 7)}) {
        reject(illegal.query, illegal.key, illegal.tokens, illegal.head_dim, "illegal half/power-of-two layout");
    }
    for (const std::size_t baby : {0, 1, 3, 6, 17}) {
        invalid([&] { (void)ref::packed_input_cyclic_qk(input.query, input.key, 16, 8, baby); },
                "packed-input zero/odd/nondividing baby");
    }
    for (const bool alter_query : {false, true}) {
        for (const bool shorten : {false, true}) {
            auto q = input.query, k = input.key;
            auto &bad = alter_query ? q : k;
            bad.resize(shorten ? bad.size() - 1 : bad.size() + 1);
            reject(q, k, 16, 8, "short/long operand");
        }
        for (const std::size_t slot : {std::size_t{0}, std::size_t{16}, std::size_t{511}}) {
            for (const double value : {std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::infinity(),
                                       -std::numeric_limits<double>::infinity()}) {
                auto q = input.query, k = input.key;
                (alter_query ? q : k)[slot] = value;
                reject(q, k, 16, 8, "nonfinite live/gap lane");
            }
        }
        for (const std::size_t slot : {std::size_t{8}, std::size_t{24}, std::size_t{511}}) {
            for (const double value : {-0.25, 0.25}) {
                auto q = input.query, k = input.key;
                (alter_query ? q : k)[slot] = value;
                reject(q, k, 16, 8, "dirty ideal-message gap");
            }
        }
    }
}

void test_input_rejection() {
    const auto input = fixture(16, 8);
    // Both algorithms must reject malformed dimensions, sizes, and layouts.
    const auto reject_both = [](const std::vector<double> &q, const std::vector<double> &k,
                                std::size_t tokens, std::size_t dim, const std::string &label) {
        invalid([&] { (void)ref::cyclic_qk(q, k, tokens, dim); }, "schedule " + label);
        invalid([&] { (void)ref::direct_oracle(q, k, tokens, dim); }, "oracle " + label);
    };
    reject_both(input.query, input.key, 0, 8, "zero tokens");
    reject_both(input.query, input.key, 16, 0, "zero head dimension");
    reject_both(input.query, input.key, 16, 17, "head dimension exceeds T");
    reject_both({}, {}, std::numeric_limits<std::size_t>::max(), 1, "tile overflow");
    reject_both({}, {}, std::numeric_limits<std::size_t>::max() / 2, 1, "slot overflow");
    reject_both({}, {}, 16, 8, "empty inputs");
    for (const bool alter_query : {false, true}) {
        for (const bool shorten : {false, true}) {
            auto q = input.query;
            auto k = input.key;
            auto &bad = alter_query ? q : k;
            bad.resize(shorten ? bad.size() - 1 : bad.size() + 1);
            reject_both(q, k, 16, 8, "wrong input length");
        }
        for (const double bad_value : {std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::infinity(),
                                       -std::numeric_limits<double>::infinity()}) {
            for (const std::size_t slot : {std::size_t{0}, std::size_t{16}, std::size_t{511}}) {
                auto q = input.query;
                auto k = input.key;
                (alter_query ? q : k)[slot] = bad_value;
                reject_both(q, k, 16, 8, "nonfinite active/gap lane");
            }
        }
        for (const std::size_t slot : {std::size_t{8}, std::size_t{31}, std::size_t{511}}) {
            auto q = input.query;
            auto k = input.key;
            (alter_query ? q : k)[slot] = 0.25;
            reject_both(q, k, 16, 8, "nonzero gap lane");
        }
    }
    invalid([&] { (void)ref::cyclic_qk(input.query, input.key, 16, 8, 0); }, "zero baby");
    invalid([&] { (void)ref::cyclic_qk(input.query, input.key, 16, 8, 3); }, "nondividing baby");
    const auto too_wide = fixture(24, 17);
    invalid([&] { (void)ref::cyclic_qk(too_wide.query, too_wide.key, 24, 17); },
            "padded reduction crossing a head boundary");
}

// Exactly the first 52 Q primes of generate_aligned.py::CERIUM_PRIMES for
// candidate('q52-p11'). These are ordered actual integers, NOT 28-bit proxies.
const std::vector<std::uint64_t> ordered_q = {
    204865537, 205651969, 206307329, 207880193, 209059841, 210370561, 211025921, 211812353,
    214171649, 215482369, 215744513, 216137729, 216924161, 217317377, 218628097, 219676673,
    220594177, 221249537, 222035969, 222167041, 222953473, 223215617, 224002049, 224133121,
    225574913, 228065281, 228458497, 228720641, 230424577, 230686721, 230817793, 231473153,
    232390657, 232652801, 234356737, 235798529, 236584961, 236716033, 239337473, 239861761,
    240648193, 241827841, 244842497, 244973569, 245235713, 245760001, 246415361, 249561089,
    253100033, 253493249, 254279681, 256376833
};

// Independent high-precision physical-scale oracle: log of the actual PRODUCT,
// rather than the implementation's sum of two double-precision logarithms.
long double physical_drop(std::size_t level, const std::vector<std::uint64_t> &q) {
    return std::log2(static_cast<long double>(q.at(level)) * q.at(level - 1));
}

QkSpec physical_spec(std::size_t level = 8) {
    QkSpec spec;
    spec.id = "cpu-physical-" + std::to_string(level);
    spec.input_level = level;
    spec.output_level = level - 4;
    spec.input_scale_log2 = 55.0;
    spec.intermediate_scale_log2 = static_cast<double>(2.0L * spec.input_scale_log2 -
                                                      physical_drop(level, ordered_q));
    spec.output_scale_log2 = 55.0;
    return spec;
}

void unchanged(const QkSpec &actual, const QkSpec &before) {
    check(actual.id == before.id && actual.tokens == before.tokens && actual.tile == before.tile &&
          actual.head_dim == before.head_dim && actual.baby == before.baby &&
          actual.input_level == before.input_level && actual.output_level == before.output_level &&
          actual.input_scale_log2 == before.input_scale_log2 &&
          actual.intermediate_scale_log2 == before.intermediate_scale_log2 &&
          actual.output_scale_log2 == before.output_scale_log2, "scale helper must not rewrite QkSpec");
}

void check_physical_plan(const QkSpec &spec, const std::vector<std::uint64_t> &q) {
    const auto before = spec;
    const auto plan = ref::derive_scale_plan(spec, q);
    unchanged(spec, before);
    const auto level = spec.input_level;
    const long double mid = 2.0L * spec.input_scale_log2 - physical_drop(level, q);
    const long double mask = spec.output_scale_log2 - mid + physical_drop(level - 2, q);
    near(plan.intermediate_scale_log2, static_cast<double>(mid), "actual CC two-prime scale", 1e-12, 0);
    near(plan.mask_scale_log2, static_cast<double>(mask), "actual CP mask scale", 1e-12, 0);
    near(plan.output_scale_log2, spec.output_scale_log2, "declared output target", 0, 0);
    const long double final_scale = static_cast<long double>(plan.intermediate_scale_log2) +
                                   plan.mask_scale_log2 - physical_drop(level - 2, q);
    near(static_cast<double>(final_scale), spec.output_scale_log2,
         "physical CC -> reduction -> mask CP -> output formula", 1e-12, 0);
    // Scales themselves, not just level counts, account for both real primes.
    const long double cc_ratio = std::exp2(2.0L * spec.input_scale_log2 - plan.intermediate_scale_log2);
    const long double cc_primes = static_cast<long double>(q[level]) * q[level - 1];
    check(std::abs(cc_ratio / cc_primes - 1.0L) < 1e-13L, "CC rescale divisor equals real Q product");
    const long double cp_ratio = std::exp2(static_cast<long double>(plan.intermediate_scale_log2) +
                                         plan.mask_scale_log2 - plan.output_scale_log2);
    const long double cp_primes = static_cast<long double>(q[level - 2]) * q[level - 3];
    check(std::abs(cp_ratio / cp_primes - 1.0L) < 1e-13L, "CP rescale divisor equals real Q product");
}

void test_scale_plan() {
    check(ordered_q.size() == 52, "q52 actual prime count");
    const auto spec = physical_spec();
    check(spec.input_level == 8 && spec.input_level - 2 == 6 && spec.output_level == 4,
          "physical 8 -> 6 -> 4, not q_count 8 -> 6 -> 4");
    check(std::abs(spec.intermediate_scale_log2 - 55.0) > 1e-3,
          "nominal intermediate is not an actual-prime result");
    auto nominal = spec;
    nominal.input_scale_log2 = 55.0;
    nominal.intermediate_scale_log2 = 55.0;
    nominal.output_scale_log2 = 55.0;
    invalid([&] { (void)ref::derive_scale_plan(nominal, ordered_q); }, "unphysical nominal scale metadata");
    check_physical_plan(QkSpec{}, ordered_q);
    for (const std::size_t level : {4, 8, 12, 51}) {
        check_physical_plan(physical_spec(level), ordered_q);
    }
    const auto plan = ref::derive_scale_plan(spec, ordered_q);
    check(std::abs(plan.mask_scale_log2 - 56.0) > 1e-3, "mask scale is not nominal two 28-bit primes");

    auto changed = spec;
    changed.input_scale_log2 += 0.125;
    changed.intermediate_scale_log2 += 0.25;
    check_physical_plan(changed, ordered_q);
    near(ref::derive_scale_plan(changed, ordered_q).mask_scale_log2, plan.mask_scale_log2 - 0.25,
         "input-scale change propagates through physical mask");
    changed = spec;
    changed.output_scale_log2 += 0.375;
    check_physical_plan(changed, ordered_q);
    near(ref::derive_scale_plan(changed, ordered_q).mask_scale_log2, plan.mask_scale_log2 + 0.375,
         "output target change propagates through mask");

    // Tolerance permits rounding but the returned intermediate stays physical,
    // not the slightly different declared number.
    changed = spec;
    changed.intermediate_scale_log2 += 5e-9;
    const auto before = changed;
    const auto tolerant = ref::derive_scale_plan(changed, ordered_q);
    unchanged(changed, before);
    near(tolerant.intermediate_scale_log2, plan.intermediate_scale_log2, "never override physical intermediate", 0, 0);
    check(tolerant.intermediate_scale_log2 != changed.intermediate_scale_log2,
          "accepted tolerance must not turn into a metadata override");
    for (const double difference : {-2e-8, 2e-8, -1.0, 1.0}) {
        changed = spec;
        changed.intermediate_scale_log2 += difference;
        invalid([&] { (void)ref::derive_scale_plan(changed, ordered_q); }, "intermediate mismatch above 1e-8");
    }

    auto reordered = ordered_q;
    std::swap(reordered[8], reordered[6]);
    invalid([&] { (void)ref::derive_scale_plan(spec, reordered); }, "actual Q order matters");
    changed = spec;
    changed.intermediate_scale_log2 = static_cast<double>(2.0L * changed.input_scale_log2 -
                                                         physical_drop(8, reordered));
    check_physical_plan(changed, reordered);
    const auto reordered_plan = ref::derive_scale_plan(changed, reordered);
    check(std::abs(reordered_plan.intermediate_scale_log2 - plan.intermediate_scale_log2) > 1e-3,
          "real prime reordering changes the CC intermediate");
    near(reordered_plan.mask_scale_log2, plan.mask_scale_log2,
         "swapping within the four rescale primes preserves their total product", 1e-12, 0);
    reordered = ordered_q;
    std::swap(reordered[8], reordered[4]); // bring a previously retained prime into the rescale
    changed.intermediate_scale_log2 = static_cast<double>(2.0L * changed.input_scale_log2 -
                                                         physical_drop(8, reordered));
    check_physical_plan(changed, reordered);
    check(std::abs(ref::derive_scale_plan(changed, reordered).mask_scale_log2 - plan.mask_scale_log2) > 1e-3,
          "changing the four rescale primes changes the physical mask scale");
    auto prefix = std::vector<std::uint64_t>(ordered_q.begin(), ordered_q.begin() + 9);
    near(ref::derive_scale_plan(spec, prefix).mask_scale_log2, plan.mask_scale_log2,
         "physical level8 requires q_count9 and ignores inactive higher levels", 0, 0);
    prefix.pop_back();
    invalid([&] { (void)ref::derive_scale_plan(spec, prefix); }, "physical level8 with only q_count8");
    invalid([&] { (void)ref::derive_scale_plan(spec, {}); }, "empty modulus chain");

    for (const std::size_t level : {std::size_t{0}, std::size_t{3}, std::size_t{52},
                                   std::numeric_limits<std::size_t>::max()}) {
        changed = spec;
        changed.input_level = level;
        invalid([&] { (void)ref::derive_scale_plan(changed, ordered_q); }, "invalid physical input level");
    }
    for (const std::size_t output : {0, 3, 5, 8}) {
        changed = spec;
        changed.output_level = output;
        invalid([&] { (void)ref::derive_scale_plan(changed, ordered_q); }, "wrong physical output level");
    }
    for (const double nonfinite : {std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity()}) {
        for (int field = 0; field < 3; ++field) {
            changed = spec;
            if (field == 0) changed.input_scale_log2 = nonfinite;
            if (field == 1) changed.intermediate_scale_log2 = nonfinite;
            if (field == 2) changed.output_scale_log2 = nonfinite;
            invalid([&] { (void)ref::derive_scale_plan(changed, ordered_q); }, "nonfinite scale metadata");
        }
    }
    for (const std::uint64_t modulus : {0, 1}) {
        auto bad_q = ordered_q;
        bad_q[8] = modulus;
        invalid([&] { (void)ref::derive_scale_plan(spec, bad_q); }, "invalid modulus");
    }
    changed = spec;
    changed.input_scale_log2 = std::numeric_limits<double>::max();
    invalid([&] { (void)ref::derive_scale_plan(changed, ordered_q); }, "nonfinite derived intermediate");
}

void test_scale_representability() {
    const auto spec = physical_spec();
    for (const double impossible : {-2000.0, 2000.0}) {
        for (int field = 0; field < 3; ++field) {
            auto changed = spec;
            if (field == 0) changed.input_scale_log2 = impossible;
            if (field == 1) changed.intermediate_scale_log2 = impossible;
            if (field == 2) changed.output_scale_log2 = impossible;
            invalid([&] { (void)ref::derive_scale_plan(changed, ordered_q); },
                    "finite log scale whose exp2 underflows/overflows");
        }
    }

    const auto plan = ref::derive_scale_plan(spec, ordered_q);
    const double cc_drop = static_cast<double>(physical_drop(8, ordered_q));
    const double cp_first = std::log2(static_cast<double>(ordered_q[6]));
    const double cp_second = std::log2(static_cast<double>(ordered_q[5]));
    // Isolate unrepresentable intermediate scales while input, output, and
    // the would-be mask all exponentiate to finite positive doubles.
    for (const double intermediate : {-2000.0, 1040.0}) {
        auto changed = spec;
        changed.intermediate_scale_log2 = intermediate;
        changed.input_scale_log2 = (intermediate + cc_drop) / 2.0;
        changed.output_scale_log2 = intermediate < 0.0 ? -1050.0 : 1000.0;
        const double mask = changed.output_scale_log2 - intermediate + cp_first + cp_second;
        check(std::isfinite(std::exp2(changed.input_scale_log2)) &&
              std::exp2(changed.input_scale_log2) > 0.0, "representable input in intermediate rejection");
        check(std::isfinite(std::exp2(changed.output_scale_log2)) &&
              std::exp2(changed.output_scale_log2) > 0.0, "representable output in intermediate rejection");
        check(std::isfinite(std::exp2(mask)) && std::exp2(mask) > 1.0,
              "useful would-be mask in intermediate rejection");
        invalid([&] { (void)ref::derive_scale_plan(changed, ordered_q); },
                "physically matched intermediate exp2 underflow/overflow");
    }

    // All declared scales are representable, but the derived mask is not.
    auto changed = spec;
    changed.output_scale_log2 = 1023.75;
    check(std::isfinite(std::exp2(changed.output_scale_log2)), "finite near-limit output scale");
    invalid([&] { (void)ref::derive_scale_plan(changed, ordered_q); }, "derived mask exp2 overflow");
    changed = spec;
    changed.input_scale_log2 = 100.0;
    changed.intermediate_scale_log2 = static_cast<double>(200.0L - physical_drop(8, ordered_q));
    changed.output_scale_log2 = -1070.0; // positive representable subnormal, not exp2(-2000) == 0
    check(std::exp2(changed.output_scale_log2) > 0.0, "representable subnormal output scale");
    invalid([&] { (void)ref::derive_scale_plan(changed, ordered_q); }, "derived mask exp2 underflow");

    changed = spec;
    changed.output_scale_log2 = plan.intermediate_scale_log2 - cp_first - cp_second;
    const double unit_mask_log = changed.output_scale_log2 - plan.intermediate_scale_log2 +
                                 cp_first + cp_second;
    check(unit_mask_log == 0.0, "unit mask fixture has exp2(mask_log2) exactly one");
    invalid([&] { (void)ref::derive_scale_plan(changed, ordered_q); }, "unit mask encoding scale");
    changed.output_scale_log2 -= 1.0;
    invalid([&] { (void)ref::derive_scale_plan(changed, ordered_q); }, "positive mask encoding scale below one");

    // A declaration within the 1e-8 tolerance must not conceal an overflowing
    // REAL intermediate at the exp2 boundary.
    changed = spec;
    changed.input_scale_log2 = (1024.0 + cc_drop) / 2.0;
    const auto real_intermediate = [&] {
        return 2.0 * changed.input_scale_log2 - std::log2(static_cast<double>(ordered_q[8])) -
               std::log2(static_cast<double>(ordered_q[7]));
    };
    if (real_intermediate() < 1024.0) {
        changed.input_scale_log2 = std::nextafter(changed.input_scale_log2,
                                                 std::numeric_limits<double>::infinity());
    }
    changed.intermediate_scale_log2 = std::nextafter(1024.0, 0.0);
    changed.output_scale_log2 = 1000.0;
    check(std::isfinite(std::exp2(changed.intermediate_scale_log2)), "finite declared boundary intermediate");
    check(!std::isfinite(std::exp2(real_intermediate())) &&
          std::abs(real_intermediate() - changed.intermediate_scale_log2) < 1e-8,
          "real boundary intermediate overflows despite allowed metadata tolerance");
    invalid([&] { (void)ref::derive_scale_plan(changed, ordered_q); }, "physical intermediate exp2 overflow");
}

} // namespace

int main() {
    try {
        test_giant_compensation();
        test_small_layouts();
        test_full_layout_and_isolation();
        test_input_rejection();
        test_paired_schedule();
        test_paired_input_rejection();
        test_packed_input_schedule();
        test_packed_input_rejection();
        test_scale_plan();
        test_scale_representability();
        std::cout << std::setprecision(10)
                  << "PASS CPU-only QK math/metadata: " << checks << " always-on checks; "
                  << compared_slots << " compared slots; max_abs_error=" << max_absolute_error
                  << "; real ordered q52 primes; no CUDA/context\n"
                  << "PAIRED actual scalar callsites: CC=" << production_paired_counts.cc_multiply_calls
                  << " CP=" << production_paired_counts.cp_multiply_calls
                  << " rot=" << production_paired_counts.rotation_calls
                  << " add=" << production_paired_counts.add_calls
                  << "; abstract_R2=" << production_paired_counts.abstract_rescale2_calls
                  << " logical_source_batches=" << production_paired_counts.logical_shared_source_batches
                  << "; oracle_max_abs=" << paired_oracle_max_error
                  << " baseline_max_abs=" << paired_baseline_max_error << '\n'
                  << "PACKED_INPUT actual scalar callsites: CC=" << production_packed_input_counts.cc_multiply_calls
                  << " CP=" << production_packed_input_counts.cp_multiply_calls
                  << " rot=" << production_packed_input_counts.rotation_calls
                  << " add=" << production_packed_input_counts.add_calls
                  << "; ABSTRACT_R2=" << production_packed_input_counts.abstract_rescale2_calls
                  << " LOGICAL_source_batches=" << production_packed_input_counts.logical_shared_source_batches
                  << " (scalar doubles: no CKKS rescale or real hoist)"
                  << "; oracle_max_abs=" << packed_input_oracle_max_error
                  << " baseline_max_abs=" << packed_input_baseline_max_error
                  << " paired2_max_abs=" << packed_input_paired_max_error << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL CPU-only QK math/metadata: " << error.what() << '\n';
        return 1;
    }
}
