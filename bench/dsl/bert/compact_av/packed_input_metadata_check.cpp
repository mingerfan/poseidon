// Genuine CPU CKKS diagnostic of ONE giant0/baby0,1 packed AV pair.
// N1024 is an insecure diagnostic ring, NOT encrypted T128 AV/BERT or GPU
// evidence. The original V diagonal is genuinely built by encrypted BSGS.
// Live output equals two original CC terms; gaps are DEFINED DIRTY partials.
#include "av_reference.h"
#include "diagnostic_profile.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/decryptor.h"
#include "poseidon/keygenerator.h"

#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <utility>

namespace {
namespace ref = bert::attention::av_reference;
namespace profile = bert::attention::av_diagnostic;
constexpr std::size_t degree = 1024, slots = degree / 2;
constexpr std::size_t tokens = 16, tile = 32, dim = 8, baby = 8;
constexpr double tolerance = 1e-8;
std::size_t checks = 0;
double maximum_scale_log2_error = 0;

void check(bool condition, const std::string &label) {
    ++checks;
    if (!condition) throw std::runtime_error(label);
}
void metadata_is(const poseidon::Ciphertext &ct, std::size_t level, double scale,
                 const std::string &label, std::size_t size = 2) {
    check(ct.level() == level && ct.size() == size && ct.is_ntt_form(),
          label + ": real level/size/NTT mismatch");
    check(std::isfinite(scale) && scale > 0 && std::isfinite(ct.scale()) && ct.scale() > 0,
          label + ": nonfinite/nonpositive real scale");
    check(ct.scale() == scale, label + ": exact physical scale arithmetic mismatch");
}
void canonical_scale_is(const poseidon::Ciphertext &ct, double log2_scale,
                        const std::string &label) {
    const auto error = std::abs(std::log2(ct.scale()) - log2_scale);
    check(std::isfinite(error) && error < 1e-12, label + ": canonical log2 scale mismatch");
    maximum_scale_log2_error = std::max(maximum_scale_log2_error, error);
}
double error_between(const std::vector<double> &actual, const std::vector<double> &expected,
                     const std::string &label, double limit = tolerance) {
    check(actual.size() == slots && expected.size() == slots, label + ": compare ALL 512 slots");
    double error = 0;
    for (std::size_t i = 0; i < slots; ++i) {
        check(std::isfinite(actual[i]) && std::isfinite(expected[i]), label + ": nonfinite slot");
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    check(error < limit, label + ": max absolute error " + std::to_string(error));
    return error;
}
struct Counts {
    std::size_t cp = 0, cc = 0, relin = 0, rotations = 0, adds = 0, drops = 0;
};
Counts difference(const Counts &a, const Counts &b) {
    return {a.cp - b.cp, a.cc - b.cc, a.relin - b.relin,
            a.rotations - b.rotations, a.adds - b.adds, a.drops - b.drops};
}
void counts_are(const Counts &actual, const Counts &expected, const std::string &label) {
    check(actual.cp == expected.cp && actual.cc == expected.cc && actual.relin == expected.relin &&
          actual.rotations == expected.rotations && actual.adds == expected.adds && actual.drops == expected.drops,
          label + ": real evaluator call counts mismatch");
}
void print_counts(const Counts &c) {
    std::cout << "{\"cp\":" << c.cp << ",\"cc\":" << c.cc << ",\"relin\":" << c.relin
              << ",\"rotations\":" << c.rotations << ",\"adds\":" << c.adds
              << ",\"one_prime_rescales\":" << c.drops << '}';
}
struct Drop {
    std::string label;
    std::size_t from;
    std::uint64_t prime;
    double before, after;
};
std::size_t index(std::size_t t, std::size_t h, std::size_t j) {
    return (t % tokens) * tile + h * tokens + j;
}
} // namespace

int main() {
    try {
        using namespace poseidon;
        const ref::Layout layout{tokens, tile, dim, baby};
        const ref::ScaleSpec spec;
        auto ordered_q = profile::q52();
        const auto plan = ref::derive_scale_plan(ordered_q, spec);
        ordered_q.resize(11);
        const auto exact_p = profile::p11();
        std::vector<Modulus> q, p;
        for (const auto prime : ordered_q) q.emplace_back(prime);
        for (const auto prime : exact_p) p.emplace_back(prime);
        ParametersLiteral literal(CKKS, 10, 9, 28, 0, 1, Modulus(0), q, p, sec_level_type::none);
        PoseidonContext context(literal);
        auto *factory = PoseidonFactory::get_instance();
        factory->set_device_type(DEVICE_SOFTWARE);
        auto evaluator = factory->create_ckks_evaluator(context);
        check(!context.using_hardware() && factory->get_device_type() == DEVICE_SOFTWARE,
              "must use the real software CPU evaluator");
        check(context.crt_context()->first_context_data()->level() == 10,
              "actual context must start at physical L10");
        check(context.parameters_literal()->q().size() == 11 && context.parameters_literal()->p().size() == 11,
              "actual diagnostic context must have exactly Q11/P11");
        for (std::size_t i = 0; i < 11; ++i) {
            check(context.parameters_literal()->q()[i].value() == ordered_q[i], "exact ordered Q prime mismatch");
            check(context.parameters_literal()->p()[i].value() == exact_p[i], "exact ordered P prime mismatch");
        }
        check(plan.score_mask_scale_log2 > 110.7 && plan.score_mask_scale_log2 < 110.8,
              "must genuinely encode the ~110.712708-bit wide score mask");
        CKKSEncoder encoder(context);
        KeyGenerator keygen(context);
        PublicKey public_key;
        RelinKeys relin_keys;
        GaloisKeys galois;
        keygen.create_public_key(public_key);
        keygen.create_relin_keys(relin_keys);
        std::set<int> step_set{-int(dim), int(dim), int(tile - dim), 1};
        for (std::size_t b = 1; b < baby; ++b) step_set.insert(int(tile * b));
        step_set.insert(int(tile * baby));
        keygen.create_galois_keys(std::vector<int>(step_set.begin(), step_set.end()), galois);
        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, keygen.secret_key());
        const auto ids = context.crt_context()->parms_id_map();
        Counts calls;
        std::vector<Drop> drops;
        std::size_t requested_drops = 0, wide_mask_encodes = 0;
        double mask_encoding_error = 0;
        const auto decode = [&](const Ciphertext &ct) {
            Plaintext plain;
            decryptor.decrypt(ct, plain);
            std::vector<double> decoded;
            encoder.decode(plain, decoded);
            check(decoded.size() == slots, "decrypt must return all 512 physical slots");
            for (const auto lane : decoded) check(std::isfinite(lane), "nonfinite decrypted lane");
            return decoded;
        };
        const auto multiply_mask = [&](const Ciphertext &input, const std::vector<double> &mask,
                                       double log2_scale, const std::string &label) {
            check(mask.size() == slots, label + ": encode a genuine full-slot mask");
            std::size_t ones = 0;
            for (const auto lane : mask) {
                check(lane == 0 || lane == 1, label + ": mask must be exactly 0/1");
                ones += lane == 1;
            }
            check(ones > 0 && ones < slots, label + ": fixture requires nontrivial masks");
            Plaintext plain;
            const auto requested_scale = std::exp2(log2_scale);
            // A real wide CKKS encoding, NEVER plain.scale()/ct.scale() assignment.
            encoder.encode(mask, input.parms_id(), requested_scale, plain);
            check(plain.parms_id() == input.parms_id() && plain.scale() == requested_scale,
                  label + ": real encoded mask level/scale mismatch");
            std::vector<double> decoded;
            encoder.decode(plain, decoded);
            mask_encoding_error = std::max(mask_encoding_error,
                error_between(decoded, mask, label + " encoded mask", 1e-12));
            if (log2_scale > 100) ++wide_mask_encodes;
            Ciphertext output;
            evaluator->multiply_plain(input, plain, output);
            ++calls.cp;
            metadata_is(output, input.level(), input.scale() * plain.scale(), label + " CP");
            check(output.parms_id() == input.parms_id(), label + ": CP changed real parameter ID");
            return output;
        };
        const auto rescale = [&](Ciphertext current, std::size_t count, const std::string &label) {
            requested_drops += count;
            for (std::size_t k = 0; k < count; ++k) {
                const auto from = current.level();
                check(from > 0 && from < ordered_q.size(), label + ": invalid physical drop level");
                const auto before = current.scale();
                const auto prime = ordered_q.at(from);
                const auto expected = before / double(prime);
                Ciphertext next;
                evaluator->rescale(current, next);
                ++calls.drops;
                metadata_is(next, from - 1, expected, label + " drop q" + std::to_string(from));
                check(next.parms_id() == ids.at(from - 1), label + ": real post-drop parameter ID mismatch");
                drops.push_back({label, from, prime, before, next.scale()});
                current = std::move(next);
            }
            return current;
        };
        const auto rotate = [&](const Ciphertext &input, int step, const std::string &label) {
            check(step != 0, label + ": count only genuine nonidentity rotations");
            Ciphertext output;
            evaluator->rotate(input, output, step, galois);
            ++calls.rotations;
            metadata_is(output, input.level(), input.scale(), label + " rotate");
            check(output.parms_id() == input.parms_id(), label + ": rotation changed real parameter ID");
            return output;
        };
        const auto add = [&](const Ciphertext &a, const Ciphertext &b, const std::string &label) {
            check(a.parms_id() == b.parms_id() && a.scale() == b.scale(), label + ": actual add inputs must align");
            Ciphertext output;
            evaluator->add(a, b, output);
            ++calls.adds;
            metadata_is(output, a.level(), a.scale(), label + " add");
            check(output.parms_id() == a.parms_id(), label + ": addition changed real parameter ID");
            return output;
        };
        const auto cc_relin_r2 = [&](const Ciphertext &a, const Ciphertext &b, const std::string &label) {
            check(a.level() == 6 && a.parms_id() == b.parms_id(), label + ": CC requires actual L6 operands");
            Ciphertext product, relinearized;
            const auto product_scale = a.scale() * b.scale();
            evaluator->multiply(a, b, product);
            ++calls.cc;
            metadata_is(product, 6, product_scale, label + " CC", 3);
            evaluator->relinearize(product, relinearized, relin_keys);
            ++calls.relin;
            metadata_is(relinearized, 6, product_scale, label + " relin");
            auto output = rescale(std::move(relinearized), 2, label + " R2");
            metadata_is(output, 4, (product_scale / double(ordered_q[6])) / double(ordered_q[5]), label + " output");
            canonical_scale_is(output, spec.output_scale_log2, label + " output");
            return output;
        };

        std::vector<double> scores(slots), values(slots, 0), diagonal_expected(slots, 0), live_mask(slots);
        for (std::size_t t = 0; t < tokens; ++t)
            for (std::size_t h = 0; h < 2; ++h)
                for (std::size_t j = 0; j < tokens; ++j) {
                    const auto i = index(t, h, j);
                    scores[i] = (h ? -0.31 : 0.17) + 0.002 * t - 0.003 * j +
                                0.011 * ((7 * t + 5 * j + h) % 13);
                    live_mask[i] = double(j < dim);
                    if (j < dim)
                        values[i] = (h ? 0.41 : -0.23) - 0.004 * t + 0.007 * j +
                                    0.013 * ((3 * t + 11 * j + h) % 17);
                }
        ref::validate_values(values, layout); // Genuine finite clean client V gaps.
        for (std::size_t t = 0; t < tokens; ++t)
            for (std::size_t h = 0; h < 2; ++h)
                for (std::size_t j = 0; j < dim; ++j)
                    diagonal_expected[index(t, h, j)] = values[index(t + j, h, j)];
        Plaintext score_plain, value_plain;
        encoder.encode(scores, ids.at(10), std::exp2(spec.score_scale_log2), score_plain);
        encoder.encode(values, ids.at(8), std::exp2(spec.value_scale_log2), value_plain);
        Ciphertext score10, value8;
        encryptor.encrypt(score_plain, score10);
        encryptor.encrypt(value_plain, value8);
        metadata_is(score10, 10, std::exp2(spec.score_scale_log2), "score input");
        metadata_is(value8, 8, std::exp2(spec.value_scale_log2), "value input");
        const auto score_input_error = error_between(decode(score10), scores, "score input");
        const auto value_input_error = error_between(decode(value8), values, "clean-gapped value input");

        // Original BSGS V diagonal, not a plaintext refresh or an indexed
        // encrypted fixture: L8 babies, all16 periodic selectors, nonzero
        // giant1 (+256), followed by ONE physical R2 (two evaluator drops).
        const auto diagonal_start = calls;
        std::vector<Ciphertext> value_babies{value8};
        for (std::size_t b = 1; b < baby; ++b)
            value_babies.push_back(rotate(value8, int(tile * b), "value baby"));
        Ciphertext diagonal_masked;
        for (std::size_t g = 0; g < tokens / baby; ++g) {
            Ciphertext subtotal;
            for (std::size_t b = 0; b < baby; ++b) {
                const auto shift = baby * g + b;
                std::vector<double> selector(slots, 0);
                for (std::size_t i = shift; i < slots; i += tokens) selector[i] = 1;
                auto term = multiply_mask(value_babies[b], selector, plan.value_mask_scale_log2, "value selector");
                if (b == 0) subtotal = std::move(term);
                else subtotal = add(subtotal, term, "value baby sum");
            }
            if (g != 0) subtotal = rotate(subtotal, int(tile * baby * g), "value giant");
            if (g == 0) diagonal_masked = std::move(subtotal);
            else diagonal_masked = add(diagonal_masked, subtotal, "value giant sum");
        }
        auto diagonal6 = rescale(std::move(diagonal_masked), 2, "value diagonal normalization");
        canonical_scale_is(diagonal6, spec.common_scale_log2, "BSGS diagonal6");
        const auto diagonal_calls = difference(calls, diagonal_start);
        counts_are(diagonal_calls, {16, 0, 0, 8, 15, 2}, "genuine original BSGS diagonal");
        const auto diagonal_error = error_between(decode(diagonal6), diagonal_expected, "encrypted BSGS diagonal");

        // Giant0 only: C_b[t,h,j]=S[t,h,(j+b)%T] for j<D, else0.
        // Since b=0,1 and D=8<T=16, the live range never wraps for b1;
        // rotate(+1)*Mlive is EXACTLY the cyclic live score, with no right
        // branch. Even b0 must perform a genuine wide Mlive CP and R4.
        const auto score_start = calls;
        auto even6 = rescale(multiply_mask(score10, live_mask, plan.score_mask_scale_log2, "even b0 wide Mlive"),
                             4, "even score R4");
        const auto shifted_score10 = rotate(score10, 1, "odd b1 score source");
        auto odd6 = rescale(multiply_mask(shifted_score10, live_mask, plan.score_mask_scale_log2, "odd b1 wide Mlive"),
                            4, "odd score R4");
        canonical_scale_is(even6, spec.common_scale_log2, "even score6");
        canonical_scale_is(odd6, spec.common_scale_log2, "odd score6");
        const auto score_calls = difference(calls, score_start);
        counts_are(score_calls, {2, 0, 0, 1, 0, 8}, "two real wide-mask score normalizations");
        check(wide_mask_encodes == 2, "both b0 and b1 must genuinely encode a wide Mlive mask");
        std::vector<double> even_score_expected(slots, 0), odd_score_expected(slots, 0);
        for (std::size_t t = 0; t < tokens; ++t)
            for (std::size_t h = 0; h < 2; ++h)
                for (std::size_t j = 0; j < dim; ++j) {
                    even_score_expected[index(t, h, j)] = scores[index(t, h, j)];
                    odd_score_expected[index(t, h, j)] = scores[index(t, h, (j + 1) % tokens)];
                }
        const auto even_score_error = error_between(decode(even6), even_score_expected, "even live cyclic score");
        const auto odd_score_error = error_between(decode(odd6), odd_score_expected, "odd live cyclic score");

        // INDEPENDENT defined-pair oracle: only baby0/1, not a complete AV.
        // Original BSGS has d=(j+b)%T, NOT d=b. Thus A uses d=j (even OR
        // odd depending on j), B uses d=j+1, and parity is that of d-j.
        // No rotations, masks, ciphertext diagonals, or packed intermediates
        // are used in this oracle. next(h1,t)=(h0,t+1 modT), including wrap.
        const auto indexed_term = [&](std::size_t t, std::size_t h, std::size_t j, std::size_t b) {
            const auto d = (j + b) % tokens;
            return scores[index(t, h, d)] * values[index(t + d, h, j)];
        };
        std::vector<double> original_even_expected(slots, 0), original_odd_expected(slots, 0);
        std::vector<double> original_pair_expected(slots, 0), packed_expected(slots), one_cc_expected(slots);
        std::vector<double> cpack_expected(slots), dpack_expected(slots);
        std::size_t defined_dirty_gaps = 0, a_odd_d_witnesses = 0, b_even_d_witnesses = 0;
        double expected_gap_max_abs = 0;
        for (std::size_t t = 0; t < tokens; ++t)
            for (std::size_t h = 0; h < 2; ++h)
                for (std::size_t j = 0; j < dim; ++j) {
                    const auto live = index(t, h, j), gap = index(t, h, dim + j);
                    const auto a = indexed_term(t, h, j, 0), b = indexed_term(t, h, j, 1);
                    const auto next_h = 1 - h;
                    const auto next_t = (t + (h == 1 ? 1 : 0)) % tokens;
                    const auto a_next_head = indexed_term(next_t, next_h, j, 0);
                    original_even_expected[live] = a;
                    original_odd_expected[live] = b;
                    original_pair_expected[live] = a + b;
                    one_cc_expected[live] = a;
                    one_cc_expected[gap] = b;
                    packed_expected[live] = a + b;
                    packed_expected[gap] = b + a_next_head;
                    cpack_expected[live] = scores[index(t, h, j)];
                    cpack_expected[gap] = scores[index(t, h, (j + 1) % tokens)];
                    dpack_expected[live] = values[index(t + j, h, j)];
                    dpack_expected[gap] = values[index(t + (j + 1) % tokens, h, j)];
                    defined_dirty_gaps += std::abs(packed_expected[gap]) > 1e-5;
                    expected_gap_max_abs = std::max(expected_gap_max_abs, std::abs(packed_expected[gap]));
                    a_odd_d_witnesses += j % 2 == 1;
                    b_even_d_witnesses += (j + 1) % 2 == 0;
                }
        check(defined_dirty_gaps == slots / 2, "fixture must expose every one of the256 DIRTY partial gaps");
        check(a_odd_d_witnesses == slots / 4 && b_even_d_witnesses == slots / 4,
              "pair oracle must distinguish baby parity from score-offset parity");

        const auto packed_start = calls;
        const auto cpack = add(even6, rotate(odd6, -int(dim), "Codd into gaps -8"), "Cpack");
        const auto dpack = add(diagonal6, rotate(diagonal6, int(tile - dim), "odd diagonal into gaps +24"), "Dpack");
        const auto cpack_error = error_between(decode(cpack), cpack_expected, "Cpack even/odd interleave");
        const auto dpack_error = error_between(decode(dpack), dpack_expected, "Dpack even/odd interleave");
        const auto product4 = cc_relin_r2(cpack, dpack, "packed ONE CC");
        const auto one_cc_error = error_between(decode(product4), one_cc_expected, "packed ONE CC before rotate/add");
        const auto packed4 = add(product4, rotate(product4, int(dim), "packed output +8"), "packed output");
        const auto packed_calls = difference(calls, packed_start);
        counts_are(packed_calls, {0, 1, 1, 3, 3, 2}, "one genuine packed CC/relin/R2 and rotate+8/add");

        // Original two-term CC comparison. Reusing the same normalized live
        // scores is valid because the original V diagonal has genuine gaps0.
        // It does NOT turn their gaps into the packed partial-gap contract.
        const auto original_start = calls;
        const auto original_even4 = cc_relin_r2(even6, diagonal6, "original b0");
        const auto odd_diagonal6 = rotate(diagonal6, int(tile), "original b1 value diagonal +32");
        const auto original_odd4 = cc_relin_r2(odd6, odd_diagonal6, "original b1");
        const auto original_pair4 = add(original_even4, original_odd4, "original two CC terms");
        const auto original_calls = difference(calls, original_start);
        counts_are(original_calls, {0, 2, 2, 1, 1, 4}, "original two genuine CC/relin/R2 terms");
        check(packed4.parms_id() == original_pair4.parms_id() && packed4.scale() == original_pair4.scale(),
              "packed and original pair must have EXACTLY the same actual output ID/scale");
        const auto original_even_error = error_between(decode(original_even4), original_even_expected, "original even term indexed oracle");
        const auto original_odd_error = error_between(decode(original_odd4), original_odd_expected, "original odd term indexed oracle");
        const auto original_decoded = decode(original_pair4), packed_decoded = decode(packed4);
        const auto original_pair_error = error_between(original_decoded, original_pair_expected, "original pair ALL slots");
        const auto packed_error = error_between(packed_decoded, packed_expected, "packed pair independently indexed ALL slots");
        double packed_live_error = 0, packed_gap_error = 0, live_vs_original_error = 0;
        double gap_vs_original_difference = 0, original_gap_max_abs = 0;
        std::size_t compared_live_slots = 0, compared_gap_slots = 0;
        for (std::size_t i = 0; i < slots; ++i) {
            if (i % tokens < dim) {
                ++compared_live_slots;
                packed_live_error = std::max(packed_live_error, std::abs(packed_decoded[i] - packed_expected[i]));
                live_vs_original_error = std::max(live_vs_original_error, std::abs(packed_decoded[i] - original_decoded[i]));
            } else {
                ++compared_gap_slots;
                packed_gap_error = std::max(packed_gap_error, std::abs(packed_decoded[i] - packed_expected[i]));
                gap_vs_original_difference = std::max(gap_vs_original_difference, std::abs(packed_decoded[i] - original_decoded[i]));
                original_gap_max_abs = std::max(original_gap_max_abs, std::abs(original_decoded[i]));
            }
        }
        check(compared_live_slots == 256 && compared_gap_slots == 256, "must compare all live AND all partial-gap lanes");
        check(live_vs_original_error < tolerance && original_gap_max_abs < tolerance,
              "packed live must match the two original terms, whose gaps remain zero");
        check(gap_vs_original_difference > 1e-3 && expected_gap_max_abs > 1e-3,
              "must NOT claim packed gaps equal original zero gaps");
        for (std::size_t j = 0; j < dim; ++j) {
            const auto wrap_gap = index(tokens - 1, 1, dim + j);
            const auto independent_wrap = indexed_term(tokens - 1, 1, j, 1) + indexed_term(0, 0, j, 0);
            check(std::abs(packed_decoded[wrap_gap] - independent_wrap) < tolerance,
                  "last head's gap must wrap to the NEXT token0/head0, not stay in token15");
        }
        counts_are(calls, {18, 3, 3, 13, 19, 16}, "whole diagnostic");
        check(calls.drops == requested_drops && drops.size() == requested_drops,
              "every requested drop must call and validate the physical evaluator");
        metadata_is(score10, 10, std::exp2(spec.score_scale_log2), "immutable encrypted score input");
        metadata_is(value8, 8, std::exp2(spec.value_scale_log2), "immutable encrypted value input");
        std::cout << std::setprecision(17)
                  << "{\"cpu_only\":true,\"scope\":\"one_giant0_babies0_1_pair_not_full_encrypted_av\","
                  << "\"diagnostic_insecure_ring\":true,\"degree\":1024,\"slots\":512,\"tokens\":16,\"head_dim\":8,\"tile\":32,\"baby\":8,"
                  << "\"output_semantics\":\"live_A_plus_B_gap_B_here_plus_A_nexthead_with_token_wrap\","
                  << "\"encrypted_original_bsgs_value_diagonal\":true,\"finite_client_value_gaps_exact_zero\":true,"
                  << "\"genuine_wide_score_mask_cp_r4\":true,\"b0_also_genuine_mlive_cp\":true,\"metadata_overrides\":false,"
                  << "\"score_input_level\":10,\"value_input_level\":8,\"common_level\":6,\"output_level\":4,"
                  << "\"score_mask_scale_log2\":" << plan.score_mask_scale_log2
                  << ",\"value_mask_scale_log2\":" << plan.value_mask_scale_log2
                  << ",\"actual_score6_scale_log2\":" << std::log2(even6.scale())
                  << ",\"actual_diagonal6_scale_log2\":" << std::log2(diagonal6.scale())
                  << ",\"actual_output4_scale_log2\":" << std::log2(packed4.scale())
                  << ",\"maximum_canonical_scale_log2_error\":" << maximum_scale_log2_error
                  << ",\"wide_mask_encodes\":" << wide_mask_encodes << ",\"ordered_q\":[";
        for (std::size_t i = 0; i < ordered_q.size(); ++i) std::cout << (i ? "," : "") << ordered_q[i];
        std::cout << "],\"ordered_p\":[";
        for (std::size_t i = 0; i < exact_p.size(); ++i) std::cout << (i ? "," : "") << exact_p[i];
        std::cout << "],\"diagonal_calls\":"; print_counts(diagonal_calls);
        std::cout << ",\"score_calls\":"; print_counts(score_calls);
        std::cout << ",\"packed_pair_calls\":"; print_counts(packed_calls);
        std::cout << ",\"original_pair_calls\":"; print_counts(original_calls);
        std::cout << ",\"total_calls\":"; print_counts(calls);
        std::cout << ",\"requested_physical_drops\":" << requested_drops
                  << ",\"mask_encoding_max_abs\":" << mask_encoding_error
                  << ",\"score_input_max_abs\":" << score_input_error << ",\"value_input_max_abs\":" << value_input_error
                  << ",\"diagonal_max_abs\":" << diagonal_error
                  << ",\"even_score_max_abs\":" << even_score_error << ",\"odd_score_max_abs\":" << odd_score_error
                  << ",\"cpack_max_abs\":" << cpack_error << ",\"dpack_max_abs\":" << dpack_error
                  << ",\"one_cc_interleaved_max_abs\":" << one_cc_error
                  << ",\"original_even_max_abs\":" << original_even_error << ",\"original_odd_max_abs\":" << original_odd_error
                  << ",\"original_pair_all512_max_abs\":" << original_pair_error
                  << ",\"packed_pair_all512_max_abs\":" << packed_error
                  << ",\"packed_live256_max_abs\":" << packed_live_error << ",\"packed_dirty_gap256_max_abs\":" << packed_gap_error
                  << ",\"packed_live_vs_two_original_cc_max_abs\":" << live_vs_original_error
                  << ",\"original_zero_gap_max_abs\":" << original_gap_max_abs
                  << ",\"dirty_gap_vs_original_max_difference\":" << gap_vs_original_difference
                  << ",\"expected_dirty_gaps\":" << defined_dirty_gaps
                  << ",\"a_odd_score_offset_witnesses\":" << a_odd_d_witnesses
                  << ",\"b_even_score_offset_witnesses\":" << b_even_d_witnesses
                  << ",\"last_token_head_wrap_checked\":true,\"physical_drop_trace\":[";
        for (std::size_t i = 0; i < drops.size(); ++i) {
            const auto &d = drops[i];
            std::cout << (i ? "," : "") << "{\"label\":\"" << d.label << "\",\"from\":" << d.from
                      << ",\"to\":" << d.from - 1 << ",\"prime\":" << d.prime
                      << ",\"scale_before\":" << d.before << ",\"scale_after\":" << d.after << '}';
        }
        std::cout << "],\"tolerance\":" << tolerance << ",\"checks\":" << checks << "}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "AV CPU encrypted packed-input pair metadata: " << e.what() << '\n';
        return 1;
    }
}
