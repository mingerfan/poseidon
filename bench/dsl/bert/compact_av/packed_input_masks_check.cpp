// Production-ring CPU CKKS encode/decode inventory for packed AV input masks.
// One mask Plaintext at a time; no keys, ciphertexts, evaluator, or GPU APIs.
// Capacity checks concern SCALE headroom, not arbitrary-amplitude overflow.
#include "av_reference.h"
#include "diagnostic_profile.h"
#include "poseidon/ckks_encoder.h"

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace {
namespace ref = bert::attention::av_reference;
namespace profile = bert::attention::av_diagnostic;
using Clock = std::chrono::steady_clock;
constexpr std::size_t degree = 65536, slots = degree / 2;
constexpr std::size_t tokens = 128, dim = 64, tile = 256, baby = 8;
constexpr double tolerance = 1e-12, required_headroom_bits = 32;
std::size_t checks = 0;

void check(bool condition, const std::string &message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::string json_string(const std::string &text) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : text) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
            else out << c;
        }
    }
    out << '"';
    return out.str();
}
template <class T> void print_array(const T &values) {
    std::cout << '[';
    bool first = true;
    for (const auto value : values) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << value;
    }
    std::cout << ']';
}

enum class Kind { diagonal, left_live, right_live, live };
// Independent indexed oracle used for EVERY decoded physical complex slot.
double expected_slot(Kind kind, std::size_t shift, std::size_t slot) {
    const auto lane = slot % tokens;
    switch (kind) {
    case Kind::diagonal: return double(lane == shift);
    case Kind::left_live: return double(lane < dim && lane + shift < tokens);
    case Kind::right_live: return double(lane < dim && lane + shift >= tokens);
    case Kind::live: return double(lane < dim);
    }
    throw std::logic_error("unknown mask kind");
}
// Construct a dense T-periodic mask, not a short row with a zero-filled tail.
std::size_t fill_mask(std::vector<double> &mask, Kind kind, std::size_t shift) {
    check(mask.size() == slots && shift < tokens, "mask shape/shift mismatch");
    std::size_t row_ones = 0;
    for (std::size_t lane = 0; lane < tokens; ++lane) {
        bool selected = false;
        switch (kind) {
        case Kind::diagonal: selected = lane == shift; break;
        case Kind::left_live: selected = lane < tokens - shift && lane < dim; break;
        case Kind::right_live: selected = lane >= tokens - shift && lane < dim; break;
        case Kind::live: selected = lane < dim; break;
        }
        mask[lane] = double(selected);
        row_ones += selected;
    }
    for (std::size_t start = tokens; start < slots; start += tokens)
        std::copy_n(mask.begin(), tokens, mask.begin() + start);
    // Also check the dense inventory for omitted masks, which are NOT encoded.
    for (std::size_t i = 0; i < slots; ++i)
        check(mask[i] == expected_slot(kind, shift, i), "dense mask differs from indexed oracle");
    return row_ones * (slots / tokens);
}

struct Group {
    const char *name;
    std::size_t level;
    double requested_scale;
    std::size_t encodes = 0, decodes = 0, decoded_slots = 0, ones = 0;
    double max_abs = 0, max_real_abs = 0, max_imag_abs = 0;
    double encode_seconds = 0, decode_seconds = 0;
    double actual_scale = 0, scale_log2_max_error = 0;
    poseidon::parms_id_type actual_id = poseidon::parms_id_zero;
};
struct Envelope {
    std::string stage;
    std::size_t level;
    double scale, sum_log2_q, headroom;
};

void encode_decode(const poseidon::PoseidonContext &context,
                   const poseidon::CKKSEncoder &encoder,
                   const std::vector<double> &mask, Kind kind, std::size_t shift,
                   std::size_t ones, Group &group) {
    const std::string label = std::string(group.name) + " shift " + std::to_string(shift);
    check(ones > 0 && ones < slots, label + ": only nontrivial nonzero masks are encoded");
    const auto ids = context.crt_context()->parms_id_map();
    const auto id = ids.at(static_cast<std::uint32_t>(group.level));
    // This is the ONLY mask Plaintext. It is destroyed before the next call.
    poseidon::Plaintext plain;
    auto start = Clock::now();
    encoder.encode(mask, id, group.requested_scale, plain);
    group.encode_seconds += seconds(start);
    ++group.encodes;
    check(plain.parms_id() == id && plain.poly().parms_id() == id,
          label + ": exact physical parms_id mismatch");
    const auto data = context.crt_context()->get_context_data(plain.parms_id());
    check(data && data->level() == group.level && plain.is_ntt_form() && plain.is_valid(context),
          label + ": physical level/NTT/validity mismatch");
    check(plain.poly().poly_degree() == degree && plain.poly().rns_num_q() == group.level + 1 &&
          plain.poly().rns_num_p() == 0 && plain.poly().rns_num_total() == group.level + 1 &&
          plain.coeff_count() == degree * (group.level + 1),
          label + ": actual CPU plaintext RNS dimensions mismatch");
    check(std::isfinite(plain.scale()) && plain.scale() == group.requested_scale,
          label + ": genuine encoding must preserve the exact requested scale");
    group.actual_id = plain.parms_id();
    group.actual_scale = plain.scale();
    group.scale_log2_max_error = std::max(group.scale_log2_max_error,
        std::abs(std::log2(plain.scale()) - std::log2(group.requested_scale)));
    std::vector<std::complex<double>> decoded;
    start = Clock::now();
    encoder.decode(plain, decoded);
    group.decode_seconds += seconds(start);
    ++group.decodes;
    check(decoded.size() == slots, label + ": decode must return ALL 32768 physical slots");
    for (std::size_t i = 0; i < slots; ++i) {
        const auto expected = expected_slot(kind, shift, i);
        const auto real_error = std::abs(decoded[i].real() - expected);
        const auto imag_error = std::abs(decoded[i].imag());
        const auto error = std::abs(decoded[i] - std::complex<double>(expected, 0));
        ++checks;
        if (!std::isfinite(error) || error > tolerance) {
            std::ostringstream message;
            message << std::setprecision(17) << label << ": slot " << i
                    << " decoded " << decoded[i] << ", expected " << expected
                    << ", absolute error " << error << " exceeds " << tolerance;
            throw std::runtime_error(message.str());
        }
        group.max_abs = std::max(group.max_abs, error);
        group.max_real_abs = std::max(group.max_real_abs, real_error);
        group.max_imag_abs = std::max(group.max_imag_abs, imag_error);
        ++group.decoded_slots;
    }
    group.ones += ones;
    check(plain.parms_id() == id && plain.scale() == group.requested_scale,
          label + ": decode changed physical metadata");
}
void print_group(const Group &g) {
    std::cout << "{\"name\":" << json_string(g.name) << ",\"encodes\":" << g.encodes
              << ",\"decodes\":" << g.decodes << ",\"decoded_slots\":" << g.decoded_slots
              << ",\"expected_one_slots\":" << g.ones << ",\"physical_level\":" << g.level
              << ",\"physical_parms_id\":";
    print_array(g.actual_id);
    std::cout << ",\"requested_scale\":" << g.requested_scale << ",\"actual_scale\":" << g.actual_scale
              << ",\"actual_scale_log2\":" << std::log2(g.actual_scale)
              << ",\"scale_log2_max_error\":" << g.scale_log2_max_error
              << ",\"max_abs\":" << g.max_abs << ",\"max_real_abs\":" << g.max_real_abs
              << ",\"max_imag_abs\":" << g.max_imag_abs
              << ",\"encode_seconds\":" << g.encode_seconds << ",\"decode_seconds\":" << g.decode_seconds << '}';
}
} // namespace

int main() {
    const auto begin = Clock::now();
    try {
        using namespace poseidon;
        check(omp_get_max_threads() == 2, "run with OMP_NUM_THREADS=2");
        check(tokens * tile == slots && tile == 2 * tokens && dim * 2 == tokens && tokens % baby == 0,
              "production T128/D64/tile256/full-slot layout mismatch");
        const ref::ScaleSpec spec;
        auto ordered_q = profile::q52();
        const auto plan = ref::derive_scale_plan(ordered_q, spec);
        check(spec.score_level == 10 && spec.value_level == 8 && spec.common_level == 6 && spec.output_level == 4,
              "genuine profile physical levels changed");
        check(std::abs(spec.score_scale_log2 - profile::canonical_scale_log2(10)) < 1e-12 &&
              std::abs(spec.value_scale_log2 - profile::canonical_scale_log2(8)) < 1e-12 &&
              std::abs(spec.common_scale_log2 - profile::canonical_scale_log2(6)) < 1e-12 &&
              std::abs(spec.output_scale_log2 - profile::canonical_scale_log2(4)) < 1e-12,
              "scale declarations must agree with the genuine ordered-prime profile");
        check(plan.score_mask_scale_log2 > 110.7 && plan.score_mask_scale_log2 < 110.8 &&
              plan.value_mask_scale_log2 > 55.3 && plan.value_mask_scale_log2 < 55.4,
              "wide score/value selector scales must be genuinely derived, not substituted");
        ordered_q.resize(11);
        const auto ordered_p = profile::p11();
        std::vector<Modulus> q, p;
        for (const auto prime : ordered_q) q.emplace_back(prime);
        for (const auto prime : ordered_p) p.emplace_back(prime);
        ParametersLiteral literal(CKKS, 16, 15, 28, 0, 1, Modulus(0), q, p, sec_level_type::none);
        PoseidonContext context(literal, false);
        CKKSEncoder encoder(context);
        check(!context.using_hardware() && encoder.slot_count() == slots &&
              context.parameters_literal()->degree() == degree && context.parameters_literal()->slot() == slots,
              "must use the production-size CPU CKKS context");
        const auto actual_parameters = context.parameters_literal();
        check(actual_parameters->q().size() == 11 && actual_parameters->p().size() == 11 && ordered_p.size() == 11,
              "context must contain exactly the first eleven Q primes and exact p11");
        for (std::size_t i = 0; i < 11; ++i) {
            check(actual_parameters->q()[i].value() == ordered_q[i], "exact ordered Q11 mismatch");
            check(actual_parameters->p()[i].value() == ordered_p[i], "exact ordered P11 mismatch");
            check(ordered_q[i] % (2 * degree) == 1 && ordered_p[i] % (2 * degree) == 1,
                  "exact primes must support the production N65536 NTT");
        }
        const auto crt = context.crt_context();
        const auto ids = crt->parms_id_map();
        check(crt->first_context_data()->level() == 10 && crt->first_parms_id() == ids.at(10),
              "eleven Q primes must give physical L10, not L11");
        // Every capacity bound is measured from the actual physical context's
        // ordered Q product, NOT sum of rounded bit widths, and NEVER Q*P.
        std::vector<double> actual_log2_q(11);
        for (std::size_t level = 0; level < 11; ++level) {
            const auto data = crt->get_context_data(ids.at(static_cast<std::uint32_t>(level)));
            check(data && data->level() == level && data->coeff_modulus().size() == level + 1,
                  "physical Q-chain shape mismatch");
            for (std::size_t i = 0; i <= level; ++i) {
                check(data->coeff_modulus()[i].value() == ordered_q[i], "physical context lost ordered Q prefix");
                actual_log2_q[level] += std::log2(double(data->coeff_modulus()[i].value()));
            }
        }
        const double setup_seconds = seconds(begin);
        std::vector<Envelope> envelopes;
        double min_headroom = std::numeric_limits<double>::infinity(), max_canonical_error = 0;
        const auto capacity = [&](const std::string &stage, std::size_t level, double scale) {
            check(level < actual_log2_q.size() && std::isfinite(scale) && scale > 0,
                  stage + ": capacity stage has invalid level/scale");
            const auto headroom = actual_log2_q[level] - std::log2(scale);
            check(headroom >= required_headroom_bits, stage + ": actual sum(log2 Q) < log2(scale)+32");
            min_headroom = std::min(min_headroom, headroom);
            envelopes.push_back({stage, level, scale, actual_log2_q[level], headroom});
        };
        const auto canonical = [&](double scale, double declared, const std::string &label) {
            const auto error = std::abs(std::log2(scale) - declared);
            check(std::isfinite(error) && error < 1e-12, label + ": genuine rescale arithmetic disagrees with profile");
            max_canonical_error = std::max(max_canonical_error, error);
        };
        const auto drop_scales = [&](double scale, std::size_t from, std::size_t count, const std::string &label) {
            for (std::size_t k = 0; k < count; ++k) {
                const auto level = from - k;
                scale /= double(ordered_q.at(level));
                capacity(label + "_after_drop_q" + std::to_string(level), level - 1, scale);
            }
            return scale;
        };
        const auto score_scale = std::exp2(spec.score_scale_log2);
        const auto value_scale = std::exp2(spec.value_scale_log2);
        const auto score_mask_scale = std::exp2(plan.score_mask_scale_log2);
        const auto value_mask_scale = std::exp2(plan.value_mask_scale_log2);
        capacity("score_input_encode", 10, score_scale);
        capacity("value_input_encode", 8, value_scale);
        capacity("diagonal_selector_encode", 8, value_mask_scale);
        capacity("packed_score_mask_encode", 10, score_mask_scale);
        capacity("value_preCP", 8, value_scale * value_mask_scale);
        const auto value6_scale = drop_scales(value_scale * value_mask_scale, 8, 2, "value");
        canonical(value6_scale, spec.common_scale_log2, "value6");
        capacity("wide_score_preCP", 10, score_scale * score_mask_scale);
        const auto score6_scale = drop_scales(score_scale * score_mask_scale, 10, 4, "wide_score");
        canonical(score6_scale, spec.common_scale_log2, "wide_score6");
        // Legacy profile envelope only; these are NOT additional PT encodes.
        capacity("legacy_score_preCP", 10, score_scale * score_scale);
        const auto legacy8_scale = drop_scales(score_scale * score_scale, 10, 2, "legacy_score");
        canonical(legacy8_scale, spec.value_scale_log2, "legacy_score8");
        capacity("legacy_unity_preCP", 8, legacy8_scale * value_scale);
        const auto legacy6_scale = drop_scales(legacy8_scale * value_scale, 8, 2, "legacy_unity");
        canonical(legacy6_scale, spec.common_scale_log2, "legacy_score6");
        capacity("common_preCC", 6, score6_scale * value6_scale);
        const auto output4_scale = drop_scales(score6_scale * value6_scale, 6, 2, "common_CC");
        canonical(output4_scale, spec.output_scale_log2, "output4");

        Group diagonal{"periodic_diagonal_selectors", 8, value_mask_scale};
        Group left{"left_fused_Mlive_shifts1_127", 10, score_mask_scale};
        Group right{"right_fused_Mlive_shifts65_127", 10, score_mask_scale};
        Group live{"shift0_Mlive", 10, score_mask_scale};
        std::vector<double> mask(slots);
        const auto mask_begin = Clock::now();
        for (std::size_t shift = 0; shift < tokens; ++shift) {
            const auto ones = fill_mask(mask, Kind::diagonal, shift);
            check(ones == slots / tokens, "each diagonal selector must select one lane in EVERY period");
            encode_decode(context, encoder, mask, Kind::diagonal, shift, ones, diagonal);
        }
        const auto live_ones = fill_mask(mask, Kind::live, 0);
        check(live_ones == slots / 2, "shift0 mask must be Mlive, NOT full unity");
        encode_decode(context, encoder, mask, Kind::live, 0, live_ones, live);
        std::size_t baseline_score_masks = 1, dense_right_masks = 0, omitted_zero_right_masks = 0;
        std::size_t omitted_zero_slot_checks = 0;
        for (std::size_t shift = 1; shift < tokens; ++shift) {
            // Both original (unfused) split masks are nonempty at EVERY shift.
            check((tokens - shift) * (slots / tokens) > 0 && shift * (slots / tokens) > 0,
                  "baseline split-mask inventory must be dense");
            baseline_score_masks += 2;
            auto ones = fill_mask(mask, Kind::left_live, shift);
            check(ones == std::min(dim, tokens - shift) * (slots / tokens), "left-live cardinality mismatch");
            encode_decode(context, encoder, mask, Kind::left_live, shift, ones, left);
            ones = fill_mask(mask, Kind::right_live, shift);
            ++dense_right_masks;
            const auto expected_lanes = shift > tokens - dim ? shift - (tokens - dim) : 0;
            check(ones == expected_lanes * (slots / tokens), "right-live cardinality mismatch");
            if (shift <= 64) {
                check(ones == 0 && std::all_of(mask.begin(), mask.end(), [](double x) { return x == 0; }),
                      "ALL right-live shifts 1..64 must be identically zero over ALL slots");
                ++omitted_zero_right_masks;
                omitted_zero_slot_checks += slots;
                continue; // Deliberately no all-zero Plaintext allocation/encode.
            }
            check(ones > 0, "right-live shifts 65..127 must all be present");
            encode_decode(context, encoder, mask, Kind::right_live, shift, ones, right);
        }
        const double masks_seconds = seconds(mask_begin);
        const Group *groups[]{&diagonal, &left, &right, &live};
        std::size_t total_encodes = 0, total_decodes = 0, decoded_slots = 0;
        double max_abs = 0, max_real_abs = 0, max_imag_abs = 0, encode_seconds = 0, decode_seconds = 0;
        for (const auto *g : groups) {
            total_encodes += g->encodes;
            total_decodes += g->decodes;
            decoded_slots += g->decoded_slots;
            max_abs = std::max(max_abs, g->max_abs);
            max_real_abs = std::max(max_real_abs, g->max_real_abs);
            max_imag_abs = std::max(max_imag_abs, g->max_imag_abs);
            encode_seconds += g->encode_seconds;
            decode_seconds += g->decode_seconds;
        }
        const auto packed_score_masks = left.encodes + right.encodes + live.encodes;
        check(diagonal.encodes == 128 && left.encodes == 127 && right.encodes == 63 && live.encodes == 1,
              "actual real encode categories must be 128 + 127 + 63 + 1");
        check(dense_right_masks == 127 && omitted_zero_right_masks == 64 &&
              omitted_zero_slot_checks == 64 * slots && baseline_score_masks == 255 && packed_score_masks == 191,
              "dense baseline255/packed191/omitted64 score inventory mismatch");
        check(total_encodes == 319 && total_decodes == 319 && decoded_slots == 319 * slots,
              "must perform exactly 319 real PT encodes/decodes and check EVERY physical slot");
        const std::uint64_t value_pt_bytes = std::uint64_t(degree) * 9 * sizeof(std::uint64_t);
        const std::uint64_t score_pt_bytes = std::uint64_t(degree) * 11 * sizeof(std::uint64_t);
        std::cout << std::setprecision(17)
                  << "{\"passed\":true,\"source\":\"bench/dsl/bert/compact_av/packed_input_masks_check.cpp\","
                  << "\"cpu_only\":true,\"no_gpu\":true,\"no_full_encrypted_t128\":true,"
                  << "\"scope\":\"production_ring_plaintext_masks_only\",\"keys_generated\":0,\"encryption_calls\":0,"
                  << "\"ciphertexts_allocated\":0,\"evaluator_calls\":0,\"metadata_overrides\":false,"
                  << "\"omp_max_threads\":" << omp_get_max_threads()
                  << ",\"degree\":" << degree << ",\"slots\":" << slots << ",\"tokens\":" << tokens
                  << ",\"head_dim\":" << dim << ",\"tile\":" << tile << ",\"baby\":" << baby
                  << ",\"ordered_q\":";
        print_array(ordered_q);
        std::cout << ",\"ordered_p\":";
        print_array(ordered_p);
        std::cout << ",\"score_mask_scale_log2\":" << plan.score_mask_scale_log2
                  << ",\"value_mask_scale_log2\":" << plan.value_mask_scale_log2
                  << ",\"envelope_score6_scale_log2\":" << std::log2(score6_scale)
                  << ",\"envelope_value6_scale_log2\":" << std::log2(value6_scale)
                  << ",\"envelope_output4_scale_log2\":" << std::log2(output4_scale)
                  << ",\"envelope_canonical_scale_log2_max_error\":" << max_canonical_error
                  << ",\"real_pt_encodes\":" << total_encodes << ",\"real_pt_decodes\":" << total_decodes
                  << ",\"decoded_slot_checks\":" << decoded_slots
                  << ",\"wide_score_mask_encodes\":" << packed_score_masks
                  << ",\"baseline_score_masks\":" << baseline_score_masks
                  << ",\"packed_score_masks\":" << packed_score_masks
                  << ",\"dense_right_masks_checked\":" << dense_right_masks
                  << ",\"omitted_zero_right_masks\":" << omitted_zero_right_masks
                  << ",\"omitted_right_shift_first\":1,\"omitted_right_shift_last\":64"
                  << ",\"omitted_zero_right_full_slot_checks\":" << omitted_zero_slot_checks
                  << ",\"tolerance\":" << tolerance << ",\"max_abs\":" << max_abs
                  << ",\"max_real_abs\":" << max_real_abs << ",\"max_imag_abs\":" << max_imag_abs
                  << ",\"elapsed_seconds\":" << seconds(begin) << ",\"context_setup_seconds\":" << setup_seconds
                  << ",\"masks_seconds\":" << masks_seconds << ",\"encode_seconds\":" << encode_seconds
                  << ",\"decode_seconds\":" << decode_seconds
                  << ",\"streaming\":{\"max_live_mask_plaintexts\":1,\"aggregate_plaintext_allocation\":false,"
                  << "\"max_one_cpu_plaintext_rns_payload_bytes\":" << score_pt_bytes
                  << ",\"allocated_gpu_bytes\":0,\"note\":\"CPU encoder/decoder workspaces and memory-pool caching are not a PT inventory allocation or an RSS bound\"}"
                  << ",\"inventory_cpu_rns_payload_bytes_theoretical\":{\"one_value_plaintext\":" << value_pt_bytes
                  << ",\"one_score_plaintext\":" << score_pt_bytes
                  << ",\"all128_value_selectors\":" << diagonal.encodes * value_pt_bytes
                  << ",\"baseline255_score_masks\":" << baseline_score_masks * score_pt_bytes
                  << ",\"packed191_score_masks\":" << packed_score_masks * score_pt_bytes
                  << ",\"baseline_total\":" << diagonal.encodes * value_pt_bytes + baseline_score_masks * score_pt_bytes
                  << ",\"packed_total\":" << diagonal.encodes * value_pt_bytes + packed_score_masks * score_pt_bytes
                  << ",\"note\":\"Theoretical CPU RNS payload inventory only; NOT simultaneously allocated and NOT GPU allocation\"}"
                  << ",\"capacity\":{\"kind\":\"analytical_scale_envelope_not_homomorphic_execution\","
                  << "\"uses_actual_Q_only_sum_log2\":true,\"required_headroom_bits\":" << required_headroom_bits
                  << ",\"minimum_headroom_bits\":" << min_headroom
                  << ",\"arbitrary_amplitude_overflow_proof\":false,"
                  << "\"caveat\":\"Scale headroom plus real 0/1 mask roundtrips is not an overflow proof for arbitrary amplitudes, accumulated sums, or encrypted AV\",\"stages\":[";
        for (std::size_t i = 0; i < envelopes.size(); ++i) {
            const auto &e = envelopes[i];
            if (i) std::cout << ',';
            std::cout << "{\"stage\":" << json_string(e.stage) << ",\"physical_level\":" << e.level
                      << ",\"scale\":" << e.scale << ",\"scale_log2\":" << std::log2(e.scale)
                      << ",\"actual_sum_log2_q\":" << e.sum_log2_q << ",\"headroom_bits\":" << e.headroom << '}';
        }
        std::cout << "]},\"groups\":[";
        for (std::size_t i = 0; i < 4; ++i) {
            if (i) std::cout << ',';
            print_group(*groups[i]);
        }
        std::cout << "],\"checks\":" << checks << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cout << std::setprecision(17) << "{\"passed\":false,\"cpu_only\":true,\"no_gpu\":true,"
                  << "\"no_full_encrypted_t128\":true,\"elapsed_seconds\":" << seconds(begin)
                  << ",\"checks\":" << checks << ",\"error\":" << json_string(error.what()) << "}\n";
        return 1;
    }
}
