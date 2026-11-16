// Real CPU CKKS test of the AV normalization/term pipeline. In particular,
// ENCODE a genuine 0/1 mask at ~2^110.7, CP, and really rescale four primes.
// This is not a GPU test or a claim of full encrypted T128 AV correctness.
#include "av_reference.h"
#include "diagnostic_profile.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/decryptor.h"
#include "poseidon/keygenerator.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

namespace {
namespace ref = bert::attention::av_reference;
namespace profile = bert::attention::av_diagnostic;
std::size_t checks = 0;
double maximum_scale_error = 0;

void check(bool condition, const std::string &message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void scale_is(const poseidon::Ciphertext &ct, std::size_t level, double scale,
              const std::string &label, std::size_t size = 2) {
    check(ct.level() == level && ct.size() == size, label + ": real level/size mismatch");
    check(std::isfinite(ct.scale()) && ct.scale() > 0 && std::isfinite(scale) && scale > 0,
          label + ": nonfinite/nonpositive real scale");
    check(ct.scale() == scale, label + ": scale must follow the exact real arithmetic, not a metadata edit");
}
void canonical_scale_is(const poseidon::Ciphertext &ct, double expected_log2,
                        const std::string &label) {
    const auto error = std::abs(std::log2(ct.scale()) - expected_log2);
    check(std::isfinite(error) && error < 1e-12, label + ": actual scale differs from canonical declaration");
    maximum_scale_error = std::max(maximum_scale_error, error);
}
double error_between(const std::vector<double> &actual, const std::vector<double> &expected,
                     const std::string &label, double tolerance = 1e-8) {
    check(actual.size() == expected.size(), label + ": decoded full-slot size mismatch");
    double error = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        check(std::isfinite(actual[i]) && std::isfinite(expected[i]), label + ": nonfinite decoded/reference lane");
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    check(error < tolerance, label + ": decoded math error " + std::to_string(error));
    return error;
}
} // namespace

int main() {
    try {
        using namespace poseidon;
        constexpr std::size_t degree = 1024, slots = degree / 2;
        const ref::Layout layout{16, 32, 8, 8};
        const ref::ScaleSpec spec;
        auto ordered_q = profile::q52();
        const auto plan = ref::derive_scale_plan(ordered_q, spec);
        ordered_q.resize(spec.score_level + 1);
        std::vector<Modulus> q, p;
        for (const auto prime : ordered_q) q.emplace_back(prime);
        for (const auto prime : profile::p11()) p.emplace_back(prime);
        // Diagnostic ring only (NOT a security parameter set); actual Cerium
        // primes retain their precise order. No core library is rebuilt.
        ParametersLiteral literal(CKKS, 10, 9, 28, 0, 1, Modulus(0), q, p, sec_level_type::none);
        PoseidonContext context(literal);
        auto evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);
        CKKSEncoder encoder(context);
        KeyGenerator keygen(context);
        PublicKey public_key;
        RelinKeys relin;
        GaloisKeys galois;
        keygen.create_public_key(public_key);
        keygen.create_relin_keys(relin);
        std::vector<int> steps;
        for (std::size_t b = 1; b < layout.baby; ++b) steps.push_back(int(layout.tile * b));
        steps.push_back(int(layout.tile * layout.baby)); // nonzero giant at T16
        constexpr std::size_t combined_shift = 11, combined_baby = 3;
        const int score_giant_step = int(layout.baby) - int(layout.tile * layout.baby);
        steps.push_back(score_giant_step); // exact source base for giant1
        steps.push_back(int(combined_baby));
        steps.push_back(int(combined_baby) - int(layout.tokens)); // right source: -13
        keygen.create_galois_keys(steps, galois);
        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, keygen.secret_key());
        const auto ids = context.crt_context()->parms_id_map();
        check(context.crt_context()->first_context_data()->level() == 10, "actual initial level must be L10 (11 Q primes)");
        check(plan.score_mask_scale_log2 > 110 && plan.score_mask_scale_log2 < 111,
              "wide mask must not silently fall back to an ordinary ~55-bit scale");

        std::vector<double> scores(slots), values(slots, 0), diagonal_expected(slots, 0), unity(slots, 1);
        for (std::size_t t = 0; t < layout.tokens; ++t)
            for (std::size_t h = 0; h < 2; ++h)
                for (std::size_t j = 0; j < layout.tokens; ++j) {
                    const auto i = t * layout.tile + h * layout.tokens + j;
                    scores[i] = (h ? -0.31 : 0.17) + 0.002 * t - 0.003 * j +
                                0.011 * ((7 * t + 5 * j + h) % 13);
                    if (j < layout.head_dim)
                        values[i] = (h ? 0.41 : -0.23) - 0.004 * t + 0.007 * j +
                                    0.013 * ((3 * t + 11 * j + h) % 17);
                }
        ref::validate_values(values, layout);
        for (std::size_t t = 0; t < layout.tokens; ++t)
            for (std::size_t h = 0; h < 2; ++h)
                for (std::size_t j = 0; j < layout.tokens; ++j)
                    diagonal_expected[t * layout.tile + h * layout.tokens + j] =
                        values[((t + j) % layout.tokens) * layout.tile + h * layout.tokens + j];
        const auto decode = [&](const Ciphertext &ct) {
            Plaintext plain;
            decryptor.decrypt(ct, plain);
            std::vector<double> result;
            encoder.decode(plain, result);
            check(result.size() == slots, "decrypt must return all 512 physical slots");
            for (const auto value : result) check(std::isfinite(value), "nonfinite decrypted lane");
            return result;
        };
        Plaintext score_plain, value_plain;
        encoder.encode(scores, ids.at(10), std::exp2(spec.score_scale_log2), score_plain);
        encoder.encode(values, ids.at(8), std::exp2(spec.value_scale_log2), value_plain);
        Ciphertext score10, value8;
        encryptor.encrypt(score_plain, score10);
        encryptor.encrypt(value_plain, value8);
        scale_is(score10, 10, std::exp2(spec.score_scale_log2), "encrypted scores");
        scale_is(value8, 8, std::exp2(spec.value_scale_log2), "encrypted values");
        const auto score_input_error = error_between(decode(score10), scores, "score input");
        const auto value_input_error = error_between(decode(value8), values, "value input");

        double mask_encoding_error = 0;
        std::size_t wide_mask_encodes = 0, actual_rescale_calls = 0, requested_rescale_calls = 0;
        const auto multiply_mask = [&](const Ciphertext &input, const std::vector<double> &mask,
                                       double log2_scale, const std::string &label) {
            check(mask.size() == slots, label + ": use an explicitly FULL-slot mask, not a short zero-tailed row");
            for (const auto m : mask) check(m == 0 || m == 1, label + ": mask must contain genuine 0/1 coefficients");
            Plaintext encoded;
            const auto requested_scale = std::exp2(log2_scale);
            // This call is essential: a metadata scale assignment is NOT an
            // implementation of the fused wide-mask normalization.
            encoder.encode(mask, input.parms_id(), requested_scale, encoded);
            check(encoded.parms_id() == input.parms_id() && encoded.scale() == requested_scale,
                  label + ": real mask encode level/scale mismatch");
            std::vector<double> decoded_mask;
            encoder.decode(encoded, decoded_mask);
            mask_encoding_error = std::max(mask_encoding_error,
                error_between(decoded_mask, mask, label + " encoded mask", 1e-12));
            if (log2_scale > 100) ++wide_mask_encodes;
            Ciphertext result;
            evaluator->multiply_plain(input, encoded, result);
            scale_is(result, input.level(), input.scale() * encoded.scale(), label + " real CP");
            return result;
        };
        const auto rescale = [&](Ciphertext current, std::size_t count, const std::string &label) {
            requested_rescale_calls += count;
            for (std::size_t k = 0; k < count; ++k) {
                const auto old_level = current.level();
                const auto expected_scale = current.scale() / double(ordered_q.at(old_level));
                Ciphertext next;
                evaluator->rescale(current, next);
                ++actual_rescale_calls;
                scale_is(next, old_level - 1, expected_scale,
                         label + " drop actual q" + std::to_string(old_level));
                current = std::move(next);
            }
            return current;
        };
        const auto rotate = [&](const Ciphertext &input, int step, const std::string &label) {
            Ciphertext output;
            evaluator->rotate(input, output, step, galois);
            scale_is(output, input.level(), input.scale(), label + " rotate");
            return output;
        };
        const auto add = [&](const Ciphertext &a, const Ciphertext &b, const std::string &label) {
            check(a.parms_id() == b.parms_id() && a.scale() == b.scale(), label + ": add inputs must really align");
            Ciphertext output;
            evaluator->add(a, b, output);
            scale_is(output, a.level(), a.scale(), label + " add");
            return output;
        };

        // Genuine encrypted original BSGS diagonal: baby rotations at L8,
        // periodic CP selectors, nonzero giant rotation, then real Rescale2.
        // Summing before rescale is linear and preserves the exact metadata.
        std::vector<Ciphertext> value_babies;
        value_babies.push_back(value8);
        for (std::size_t b = 1; b < layout.baby; ++b)
            value_babies.push_back(rotate(value8, int(layout.tile * b), "value baby"));
        Ciphertext diagonal_masked;
        for (std::size_t g = 0; g < layout.tokens / layout.baby; ++g) {
            Ciphertext subtotal;
            for (std::size_t b = 0; b < layout.baby; ++b) {
                const auto shift = layout.baby * g + b;
                std::vector<double> selector(slots, 0);
                for (std::size_t i = shift; i < slots; i += layout.tokens) selector[i] = 1;
                auto term = multiply_mask(value_babies[b], selector, plan.value_mask_scale_log2, "value selector");
                if (b == 0) subtotal = std::move(term); else subtotal = add(subtotal, term, "value baby sum");
            }
            if (g != 0) subtotal = rotate(subtotal, int(layout.tile * layout.baby * g), "value giant");
            if (g == 0) diagonal_masked = std::move(subtotal);
            else diagonal_masked = add(diagonal_masked, subtotal, "value giant sum");
        }
        auto diagonal6 = rescale(std::move(diagonal_masked), 2, "value normalization");
        check(diagonal6.level() == 6, "actual normalized value diagonal must be L6");
        canonical_scale_is(diagonal6, spec.common_scale_log2, "value diagonal6");
        const auto diagonal_error = error_between(decode(diagonal6), diagonal_expected, "encrypted BSGS value diagonal6");

        struct MaskCase { const char *name; int kind; std::size_t shift; };
        const std::vector<MaskCase> cases{{"left11", 0, 11}, {"right11", 1, 11},
                                         {"diagonal9", 2, 9}, {"full_unity", 3, 0}};
        double old_score_error = 0, new_score_error = 0, normalized_equivalence_error = 0;
        double old_product_error = 0, new_product_error = 0, product_equivalence_error = 0;
        std::size_t nontrivial_masks = 0;
        const auto cc_relin_rescale2 = [&](const Ciphertext &normalized, const std::string &label) {
            Ciphertext product, relinearized;
            const auto product_scale = normalized.scale() * diagonal6.scale();
            evaluator->multiply(normalized, diagonal6, product);
            scale_is(product, 6, product_scale, label + " CC", 3);
            evaluator->relinearize(product, relinearized, relin);
            scale_is(relinearized, 6, product_scale, label + " relin");
            auto output = rescale(std::move(relinearized), 2, label + " CC normalization");
            check(output.level() == 4, label + ": actual CC output must be L4");
            canonical_scale_is(output, spec.output_scale_log2, label + " output4");
            return output;
        };
        for (const auto &test : cases) {
            const std::string label(test.name);
            std::vector<double> mask(slots), masked_expected(slots), product_expected(slots);
            std::size_t ones = 0;
            for (std::size_t i = 0; i < slots; ++i) {
                const auto lane = i % layout.tokens;
                mask[i] = test.kind == 0 ? double(lane < layout.tokens - test.shift) :
                          test.kind == 1 ? double(lane >= layout.tokens - test.shift) :
                          test.kind == 2 ? double(lane == test.shift) : 1.0;
                ones += mask[i] == 1;
                masked_expected[i] = scores[i] * mask[i];
                product_expected[i] = masked_expected[i] * diagonal_expected[i];
            }
            if (test.kind != 3) {
                check(ones > 0 && ones < slots, label + ": test must exercise a nontrivial mask");
                ++nontrivial_masks;
            }
            auto new6 = rescale(multiply_mask(score10, mask, plan.score_mask_scale_log2, label + " new wide"),
                                4, label + " new Rescale4");
            auto old8 = rescale(multiply_mask(score10, mask, spec.score_scale_log2, label + " old mask"),
                                2, label + " old Rescale2 first");
            canonical_scale_is(old8, spec.value_scale_log2, label + " old score8");
            auto old6 = rescale(multiply_mask(old8, unity, spec.value_scale_log2, label + " old full unity"),
                                2, label + " old Rescale2 second");
            check(new6.level() == 6 && old6.level() == 6, label + ": both paths must REALLY reach L6");
            canonical_scale_is(new6, spec.common_scale_log2, label + " new score6");
            canonical_scale_is(old6, spec.common_scale_log2, label + " old score6");
            const auto new_decoded = decode(new6), old_decoded = decode(old6);
            new_score_error = std::max(new_score_error, error_between(new_decoded, masked_expected, label + " new score math"));
            old_score_error = std::max(old_score_error, error_between(old_decoded, masked_expected, label + " old score math"));
            normalized_equivalence_error = std::max(normalized_equivalence_error,
                error_between(new_decoded, old_decoded, label + " decoded old/new normalization equivalence"));
            const auto new4 = cc_relin_rescale2(new6, label + " new");
            const auto old4 = cc_relin_rescale2(old6, label + " old");
            const auto new_output = decode(new4), old_output = decode(old4);
            new_product_error = std::max(new_product_error, error_between(new_output, product_expected, label + " new AV term"));
            old_product_error = std::max(old_product_error, error_between(old_output, product_expected, label + " old AV term"));
            product_equivalence_error = std::max(product_equivalence_error,
                error_between(new_output, old_output, label + " decoded old/new AV term equivalence"));
        }
        check(nontrivial_masks > 0 && wide_mask_encodes == cases.size(), "wide-scale path must encode real nontrivial masks");

        // Exercise the ACTUAL two-source score path, not just its independent
        // masks. For giant1/baby3, shift11, the exact Cerium base is rotate(-248),
        // followed by LEFT baby rotation +3 and RIGHT rotation 3-T=-13. The
        // two nontrivial masked CPs must be added at L10 BEFORE one Rescale4.
        // Restoring the giant rotation +256 leaves a true cyclic-within-T
        // score rotation by 11, which has an independent indexed oracle below.
        check(combined_shift == layout.baby + combined_baby, "combined fixture must exercise giant1/baby3");
        const auto score_base = rotate(score10, score_giant_step, "combined score giant base");
        const auto left_source = rotate(score_base, int(combined_baby), "combined left source");
        const auto right_source = rotate(score_base, int(combined_baby) - int(layout.tokens), "combined right source");
        std::vector<double> left_mask(slots), right_mask(slots), cyclic_expected(slots), cyclic_product_expected(slots);
        std::size_t left_ones = 0, right_ones = 0, cyclic_vs_global_differences = 0;
        for (std::size_t t = 0; t < layout.tokens; ++t)
            for (std::size_t h = 0; h < 2; ++h)
                for (std::size_t j = 0; j < layout.tokens; ++j) {
                    const auto i = t * layout.tile + h * layout.tokens + j;
                    left_mask[i] = double(j < layout.tokens - combined_shift);
                    right_mask[i] = double(j >= layout.tokens - combined_shift);
                    left_ones += left_mask[i] == 1;
                    right_ones += right_mask[i] == 1;
                    check(left_mask[i] + right_mask[i] == 1, "combined masks must partition every physical slot");
                    cyclic_expected[i] = scores[t * layout.tile + h * layout.tokens + (j + combined_shift) % layout.tokens];
                    cyclic_product_expected[i] = cyclic_expected[i] * diagonal_expected[i];
                    cyclic_vs_global_differences += std::abs(cyclic_expected[i] - scores[(i + combined_shift) % slots]) > 1e-6;
                }
        check(left_ones > 0 && left_ones < slots && right_ones > 0 && right_ones < slots,
              "both combined CP masks must be genuinely nontrivial");
        check(cyclic_vs_global_differences > slots / 2,
              "combined fixture must distinguish within-head cyclic rotation from a global slot rotation");
        nontrivial_masks += 2;
        const auto combined_new_start = actual_rescale_calls;
        const auto wide_left = multiply_mask(left_source, left_mask, plan.score_mask_scale_log2, "combined new left wide");
        const auto wide_right = multiply_mask(right_source, right_mask, plan.score_mask_scale_log2, "combined new right wide");
        auto wide_sum = add(wide_left, wide_right, "combined new masked CP sum at L10");
        scale_is(wide_sum, 10, score10.scale() * std::exp2(plan.score_mask_scale_log2), "combined new pre-rescale sum");
        check(actual_rescale_calls == combined_new_start, "combined new branches must NOT rescale before addition");
        auto combined_new6 = rescale(std::move(wide_sum), 4, "combined new single Rescale4");
        check(actual_rescale_calls - combined_new_start == 4, "combined new sum must drop exactly four real primes");
        combined_new6 = rotate(combined_new6, int(layout.tile * layout.baby), "combined new restore giant");
        canonical_scale_is(combined_new6, spec.common_scale_log2, "combined new score6");

        const auto combined_old_start = actual_rescale_calls;
        const auto old_left = multiply_mask(left_source, left_mask, spec.score_scale_log2, "combined old left mask");
        const auto old_right = multiply_mask(right_source, right_mask, spec.score_scale_log2, "combined old right mask");
        auto old_sum = add(old_left, old_right, "combined old masked CP sum at L10");
        scale_is(old_sum, 10, score10.scale() * std::exp2(spec.score_scale_log2), "combined old pre-rescale sum");
        check(actual_rescale_calls == combined_old_start, "combined old branches must NOT rescale before addition");
        auto combined_old8 = rescale(std::move(old_sum), 2, "combined old first Rescale2");
        canonical_scale_is(combined_old8, spec.value_scale_log2, "combined old score8");
        auto combined_old6 = rescale(multiply_mask(combined_old8, unity, spec.value_scale_log2, "combined old full unity"),
                                    2, "combined old second Rescale2");
        check(actual_rescale_calls - combined_old_start == 4, "combined old path must drop exactly four real primes");
        combined_old6 = rotate(combined_old6, int(layout.tile * layout.baby), "combined old restore giant");
        canonical_scale_is(combined_old6, spec.common_scale_log2, "combined old score6");
        check(combined_new6.level() == 6 && combined_old6.level() == 6, "combined old/new paths must REALLY reach L6");
        const auto combined_new_decoded = decode(combined_new6), combined_old_decoded = decode(combined_old6);
        const auto combined_new_score_error = error_between(combined_new_decoded, cyclic_expected, "combined new cyclic-within-T score math");
        const auto combined_old_score_error = error_between(combined_old_decoded, cyclic_expected, "combined old cyclic-within-T score math");
        const auto combined_score_equivalence_error = error_between(combined_new_decoded, combined_old_decoded, "combined old/new normalization equivalence");
        const auto combined_new4 = cc_relin_rescale2(combined_new6, "combined new");
        const auto combined_old4 = cc_relin_rescale2(combined_old6, "combined old");
        const auto combined_new_output = decode(combined_new4), combined_old_output = decode(combined_old4);
        const auto combined_new_product_error = error_between(combined_new_output, cyclic_product_expected, "combined new AV term math");
        const auto combined_old_product_error = error_between(combined_old_output, cyclic_product_expected, "combined old AV term math");
        const auto combined_product_equivalence_error = error_between(combined_new_output, combined_old_output, "combined old/new AV term equivalence");
        check(wide_mask_encodes == cases.size() + 2, "combined path must genuinely encode both wide masks");
        check(actual_rescale_calls == requested_rescale_calls, "every requested physical drop must invoke the real evaluator");
        scale_is(score10, 10, std::exp2(spec.score_scale_log2), "immutable score input");
        scale_is(value8, 8, std::exp2(spec.value_scale_log2), "immutable value input");
        std::cout << std::setprecision(17)
                  << "{\"cpu_only\":true,\"scope\":\"normalization_and_av_term_not_full_encrypted_av\","
                  << "\"genuine_wide_mask_encode_cp_rescale4\":true,\"old_cp_rescale2_cp_unity_rescale2\":true,"
                  << "\"encrypted_bsgs_diagonal\":true,\"genuine_cc_relin_rescale2\":true,"
                  << "\"two_source_masked_add_before_rescale4\":true,\"cyclic_within_tokens_shift\":" << combined_shift
                  << ",\"degree\":1024,\"physical_score_level\":10,\"physical_value_level\":8,"
                  << "\"physical_common_level\":6,\"physical_output_level\":4,"
                  << "\"score_mask_scale_log2\":" << plan.score_mask_scale_log2
                  << ",\"value_mask_scale_log2\":" << plan.value_mask_scale_log2
                  << ",\"wide_mask_encodes\":" << wide_mask_encodes << ",\"nontrivial_masks\":" << nontrivial_masks
                  << ",\"actual_rescale_calls\":" << actual_rescale_calls
                  << ",\"requested_rescale_calls\":" << requested_rescale_calls
                  << ",\"scale_log2_max_error\":" << maximum_scale_error
                  << ",\"mask_encoding_max_abs\":" << mask_encoding_error
                  << ",\"score_input_max_abs\":" << score_input_error << ",\"value_input_max_abs\":" << value_input_error
                  << ",\"diagonal_max_abs\":" << diagonal_error
                  << ",\"old_score_max_abs\":" << old_score_error << ",\"new_score_max_abs\":" << new_score_error
                  << ",\"normalization_old_new_max_abs\":" << normalized_equivalence_error
                  << ",\"old_product_max_abs\":" << old_product_error << ",\"new_product_max_abs\":" << new_product_error
                  << ",\"product_old_new_max_abs\":" << product_equivalence_error
                  << ",\"combined_new_score_max_abs\":" << combined_new_score_error
                  << ",\"combined_old_score_max_abs\":" << combined_old_score_error
                  << ",\"combined_score_old_new_max_abs\":" << combined_score_equivalence_error
                  << ",\"combined_new_product_max_abs\":" << combined_new_product_error
                  << ",\"combined_old_product_max_abs\":" << combined_old_product_error
                  << ",\"combined_product_old_new_max_abs\":" << combined_product_equivalence_error
                  << ",\"checks\":" << checks << "}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "AV CPU encrypted metadata: " << e.what() << '\n'; return 1;
    }
}
