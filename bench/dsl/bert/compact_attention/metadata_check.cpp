// Genuine CPU encrypt -> CC -> relin -> two real rescales -> headreduce ->
// full CKKS mask -> two real rescales -> decrypt. No CUDA headers or context.
#include "prepared_qk.h"
#include "qk_reference.h"
#include "diagnostic_profile.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/decryptor.h"
#include "poseidon/keygenerator.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
}
int main() {
    try {
        using namespace poseidon;
        using namespace bert::attention;
        constexpr std::size_t degree = 1024, tokens = 16, tile = 32, dim = 8;
        auto ordered_q = diagnostic::q52();
        ordered_q.resize(9);
        std::vector<Modulus> q, p;
        for (auto prime : ordered_q) q.emplace_back(prime);
        for (auto prime : diagnostic::p11()) p.emplace_back(prime);
        QkSpec spec;
        const auto plan = reference::derive_scale_plan(spec, ordered_q);
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
        keygen.create_galois_keys(std::vector<int>{1,2,4,-8,7,24,32,-1}, galois);
        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, keygen.secret_key());
        std::vector<double> a(degree / 2, 0), b(a.size(), 0), mask(a.size(), 0), expected(a.size(), 0);
        for (std::size_t t = 0; t < tokens; ++t)
            for (std::size_t h = 0; h < 2; ++h) {
                mask[t * tile + h * tokens] = 1.0 / std::sqrt(double(dim));
                for (std::size_t f = 0; f < dim; ++f) {
                    const auto i = t * tile + h * tokens + f;
                    a[i] = (h ? -0.21 : 0.13) + 0.0007 * double(t) + 0.003 * double(f);
                    b[i] = (h ? 0.37 : -0.08) - 0.001 * double(t) + 0.002 * double(f);
                    expected[t * tile + h * tokens] += a[i] * b[i] / std::sqrt(double(dim));
                }
            }
        Plaintext pa, pb;
        encoder.encode(a, std::exp2(spec.input_scale_log2), pa);
        encoder.encode(b, pa.parms_id(), pa.scale(), pb);
        Ciphertext ca, cb, product, reduced, rel;
        encryptor.encrypt(pa, ca); encryptor.encrypt(pb, cb);
        check(ca.level() == spec.input_level, "CPU input physical level mismatch");
        evaluator->multiply(ca, cb, product);
        const auto product_scale = ca.scale() * cb.scale();
        check(product.scale() == product_scale && product.size() == 3, "CC physical scale/size mismatch");
        evaluator->relinearize(product, rel, relin);
        check(rel.scale() == product_scale && rel.size() == 2, "relin changed real scale");
        evaluator->rescale(rel, reduced);
        check(reduced.scale() == product_scale / double(q[8].value()), "first rescale ignored actual Q");
        Ciphertext reduced2;
        evaluator->rescale(reduced, reduced2);
        double expected_scale = product_scale / double(q[8].value()) / double(q[7].value());
        check(reduced2.scale() == expected_scale && reduced2.level() == 6, "Rescale2 physical metadata mismatch");
        check(std::abs(std::log2(reduced2.scale()) - plan.intermediate_scale_log2) < 1e-12,
              "derived intermediate scale mismatch");
        for (int shift : {1,2,4}) {
            Ciphertext rotated, sum;
            evaluator->rotate(reduced2, rotated, shift, galois);
            check(rotated.scale() == expected_scale, "rotate changed scale");
            evaluator->add(reduced2, rotated, sum);
            check(sum.scale() == expected_scale, "headreduce add changed scale");
            reduced2 = std::move(sum);
        }
        Plaintext encoded_mask;
        encoder.encode(mask, reduced2.parms_id(), std::exp2(plan.mask_scale_log2), encoded_mask);
        Ciphertext masked, last1, result;
        evaluator->multiply_plain(reduced2, encoded_mask, masked);
        expected_scale *= encoded_mask.scale();
        check(masked.scale() == expected_scale, "mask physical CP scale mismatch");
        evaluator->rescale(masked, last1);
        expected_scale /= double(q[6].value());
        check(last1.scale() == expected_scale, "mask first drop scale mismatch");
        evaluator->rescale(last1, result);
        expected_scale /= double(q[5].value());
        check(result.scale() == expected_scale && result.level() == spec.output_level,
              "final real scale/physical level mismatch");
        const auto scale_error = std::abs(std::log2(result.scale()) - spec.output_scale_log2);
        check(scale_error < 1e-12, "final real scale differs from declaration");
        Plaintext decrypted;
        decryptor.decrypt(result, decrypted);
        std::vector<double> decoded;
        encoder.decode(decrypted, decoded);
        double error = 0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            check(std::isfinite(decoded[i]), "nonfinite CPU decrypted result");
            error = std::max(error, std::abs(decoded[i] - expected[i]));
        }
        check(error < 1e-8, "CPU encrypted QK-term correctness error");

        // Genuine paired D8 experiment, using two independently encrypted
        // products and the SAME actual Q chain as the untouched baseline above.
        // Ideal input gaps are zero; CKKS encoding/noise gaps are not bitwise 0.
        std::vector<double> c(a.size(), 0.0), d(a.size(), 0.0), mask_b(a.size(), 0.0);
        std::vector<double> expected_second(a.size(), 0.0), expected_paired = expected;
        for (std::size_t t = 0; t < tokens; ++t) {
            for (std::size_t h = 0; h < 2; ++h) {
                const auto start = t * tile + h * tokens;
                mask_b[start + dim] = 1.0 / std::sqrt(double(dim));
                for (std::size_t f = 0; f < dim; ++f) {
                    const auto i = start + f;
                    c[i] = (h ? -0.24 : 0.17) + 0.0013 * double(t) - 0.002 * double(f);
                    d[i] = (h ? 0.29 : 0.23) - 0.0009 * double(t) + 0.004 * double(f);
                    expected_second[start] += c[i] * d[i] / std::sqrt(double(dim));
                }
                expected_paired[start + 1] = expected_second[start];
            }
        }
        bool has_positive = false, has_negative = false;
        for (const auto value : expected_paired) {
            has_positive = has_positive || value > 0.0;
            has_negative = has_negative || value < 0.0;
        }
        check(has_positive && has_negative, "paired fixture must include positive and negative scores");
        Plaintext pc, pd;
        encoder.encode(c, pa.parms_id(), pa.scale(), pc);
        encoder.encode(d, pa.parms_id(), pa.scale(), pd);
        Ciphertext cc, cd;
        encryptor.encrypt(pc, cc);
        encryptor.encrypt(pd, cd);
        const auto cc_relin_r2 = [&](const Ciphertext &left, const Ciphertext &right) {
            check(left.level() == 8 && right.level() == 8 && left.size() == 2 && right.size() == 2,
                  "paired CC input physical level/size mismatch");
            Ciphertext raw, relinearized, first, second;
            evaluator->multiply(left, right, raw);
            const auto scale = left.scale() * right.scale();
            check(raw.scale() == scale && raw.size() == 3 && raw.level() == 8,
                  "paired true CC scale/level/size mismatch");
            evaluator->relinearize(raw, relinearized, relin);
            check(relinearized.scale() == scale && relinearized.size() == 2 && relinearized.level() == 8,
                  "paired true relin changed physical metadata");
            evaluator->rescale(relinearized, first);
            check(first.scale() == scale / double(q[8].value()) && first.level() == 7,
                  "paired CC first actual-prime rescale mismatch");
            evaluator->rescale(first, second);
            check(second.scale() == scale / double(q[8].value()) / double(q[7].value()) &&
                  second.level() == 6 && second.size() == 2,
                  "paired CC second actual-prime rescale mismatch");
            check(std::abs(std::log2(second.scale()) - plan.intermediate_scale_log2) < 1e-12,
                  "paired CC derived intermediate scale mismatch");
            return second;
        };
        const auto rotate_checked = [&](const Ciphertext &input, int shift) {
            Ciphertext rotated;
            evaluator->rotate(input, rotated, shift, galois);
            check(rotated.scale() == input.scale() && rotated.level() == input.level() &&
                  rotated.size() == input.size(), "paired rotation changed physical metadata");
            return rotated;
        };
        const auto add_checked = [&](const Ciphertext &left, const Ciphertext &right) {
            check(left.scale() == right.scale() && left.parms_id() == right.parms_id(),
                  "paired add operands have incompatible physical metadata");
            Ciphertext sum;
            evaluator->add(left, right, sum);
            check(sum.scale() == left.scale() && sum.level() == left.level() && sum.size() == 2,
                  "paired add changed physical metadata");
            return sum;
        };
        const auto reduce_head = [&](Ciphertext input) {
            for (int shift : {1,2,4}) {
                // Deliberately rotate the current accumulated source.
                input = add_checked(input, rotate_checked(input, shift));
            }
            return input;
        };
        const auto mask_checked = [&](const Ciphertext &input, const Plaintext &plain) {
            check(input.level() == 6 && input.parms_id() == plain.parms_id() &&
                  plain.scale() == std::exp2(plan.mask_scale_log2),
                  "paired full CKKS mask was not encoded at derived scale/physical level");
            Ciphertext masked_term;
            evaluator->multiply_plain(input, plain, masked_term);
            check(masked_term.scale() == input.scale() * plain.scale() && masked_term.level() == 6,
                  "paired genuine mask CP physical scale/level mismatch");
            return masked_term;
        };
        const auto final_r2 = [&](const Ciphertext &input) {
            check(input.level() == 6, "paired final R2 must start at physical level6");
            Ciphertext first, second;
            evaluator->rescale(input, first);
            check(first.scale() == input.scale() / double(q[6].value()) && first.level() == 5,
                  "paired mask first actual-prime rescale mismatch");
            evaluator->rescale(first, second);
            check(second.scale() == input.scale() / double(q[6].value()) / double(q[5].value()) &&
                  second.level() == 4 && second.size() == 2,
                  "paired mask second actual-prime rescale mismatch");
            check(std::abs(std::log2(second.scale()) - spec.output_scale_log2) < 1e-12,
                  "paired final physical scale differs from declaration");
            return second;
        };
        const auto decode_checked = [&](const Ciphertext &input) {
            Plaintext plain;
            std::vector<double> values;
            decryptor.decrypt(input, plain);
            encoder.decode(plain, values);
            check(values.size() == a.size(), "paired decoded slot count mismatch");
            for (const auto value : values) check(std::isfinite(value), "nonfinite paired CPU decoded slot");
            return values;
        };
        const auto term0 = cc_relin_r2(ca, cb);
        const auto term1 = cc_relin_r2(cc, cd);
        // Old schedule: the baseline result already supplies separately reduced,
        // masked, and R2'd term0. Independently do exactly that for term1 too.
        const auto separately_reduced1 = reduce_head(term1);
        const auto separate1 = final_r2(mask_checked(separately_reduced1, encoded_mask));
        const auto decoded_separate1 = decode_checked(separate1);
        double separate_error = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            separate_error = std::max(separate_error, std::abs(decoded_separate1[i] - expected_second[i]));
        }
        check(separate_error < 1e-8, "independently reduced second encrypted term error");

        // New schedule: real P1.rotate(-D), one shared reduction, two genuine
        // full CKKS masks, real B.rotate(D-1), add, then ONE final real R2.
        const auto packed = add_checked(term0, rotate_checked(term1, -int(dim)));
        const auto shared_reduced = reduce_head(packed);
        Plaintext encoded_mask_b;
        encoder.encode(mask_b, shared_reduced.parms_id(), std::exp2(plan.mask_scale_log2), encoded_mask_b);
        const auto paired_a = mask_checked(shared_reduced, encoded_mask);
        const auto paired_b = mask_checked(shared_reduced, encoded_mask_b);
        const auto paired_result = final_r2(add_checked(paired_a, rotate_checked(paired_b, int(dim - 1))));
        const auto decoded_paired = decode_checked(paired_result);
        const auto paired_scale_error = std::abs(std::log2(paired_result.scale()) - spec.output_scale_log2);
        check(paired_result.level() == 4 && paired_result.scale() == result.scale(),
              "paired and baseline output physical metadata differ");
        double paired_error = 0.0, paired_vs_separate = 0.0;
        std::size_t paired_starts = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            paired_error = std::max(paired_error, std::abs(decoded_paired[i] - expected_paired[i]));
        }
        for (std::size_t t = 0; t < tokens; ++t) {
            for (std::size_t h = 0; h < 2; ++h) {
                const auto start = t * tile + h * tokens;
                const auto error0 = std::abs(decoded_paired[start] - decoded[start]);
                const auto error1 = std::abs(decoded_paired[start + 1] - decoded_separate1[start]);
                check(error0 < 1e-8 && error1 < 1e-8, "paired starts0/1 differ from old separately reduced terms");
                paired_vs_separate = std::max({paired_vs_separate, error0, error1});
                paired_starts += 2;
            }
        }
        check(paired_error < 1e-8, "genuine CPU paired full-slot decode error");

        // A DIFFERENT experiment from unrelated c*d above: SAME encrypted Q
        // for P0=Q*K and P1=Q*K.rotate(tile), including adjacent-token wrap.
        // The independent indexed oracle supplies only desired d0/d1 scores;
        // every other slot must decode to zero (within CKKS noise tolerance).
        const auto oracle = reference::direct_oracle(a, b, tokens, dim);
        std::vector<double> expected_input_packed(a.size(), 0.0);
        for (std::size_t t = 0; t < tokens; ++t) {
            for (std::size_t h = 0; h < 2; ++h) {
                const auto start = t * tile + h * tokens;
                expected_input_packed[start] = oracle[start];
                expected_input_packed[start + 1] = oracle[start + 1];
                check(std::abs(oracle[start] - expected[start]) < 1e-15,
                      "same-Q oracle d0 disagrees with original scalar target");
            }
        }
        // Original TWO-CC schedule, genuinely encrypted and decoded. term0 is
        // the unchanged Q*K CC/relin/R2 above, NOT the unrelated c*d term1.
        const auto adjacent_key = rotate_checked(cb, int(tile));
        const auto adjacent_term = cc_relin_r2(ca, adjacent_key);
        const auto old_same_q_a = mask_checked(reduce_head(term0), encoded_mask);
        const auto old_same_q_b = mask_checked(reduce_head(adjacent_term), encoded_mask);
        const auto old_same_q_result = final_r2(
            add_checked(old_same_q_a, rotate_checked(old_same_q_b, -1)));
        const auto decoded_old_same_q = decode_checked(old_same_q_result);
        double old_same_q_error = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            old_same_q_error = std::max(old_same_q_error,
                std::abs(decoded_old_same_q[i] - expected_input_packed[i]));
        }
        check(old_same_q_error < 1e-8, "original same-Q adjacent-key TWO-CC decode error");

        // New pre-CC schedule: true Q.rotate(-8), K.rotate(24), and additions
        // at input level8. Do not decrypt-check gaps or edit scale/level fields.
        // Count only this branch's actual arithmetic calls, not the old pair.
        std::size_t input_cc = 0, input_relin = 0, input_cp = 0;
        std::size_t input_rot = 0, input_add = 0, input_rescales = 0;
        const auto input_rotate = [&](const Ciphertext &input, int shift) {
            auto rotated = rotate_checked(input, shift);
            ++input_rot;
            return rotated;
        };
        const auto input_addition = [&](const Ciphertext &left, const Ciphertext &right) {
            auto sum = add_checked(left, right);
            ++input_add;
            return sum;
        };
        const auto input_mask = [&](const Ciphertext &input, const Plaintext &plain) {
            auto masked_term = mask_checked(input, plain);
            ++input_cp;
            return masked_term;
        };
        const auto input_rescale = [&](const Ciphertext &input) {
            check(input.level() > 0 && input.level() < q.size(), "packed-input rescale level invalid");
            Ciphertext rescaled;
            evaluator->rescale(input, rescaled);
            ++input_rescales;
            check(rescaled.level() + 1 == input.level() && rescaled.size() == input.size() &&
                  rescaled.scale() == input.scale() / double(q[input.level()].value()),
                  "packed-input real one-prime rescale changed physical metadata incorrectly");
            return rescaled;
        };
        const auto input_query = input_addition(ca, input_rotate(ca, -int(dim)));
        const auto input_key = input_addition(cb, input_rotate(cb, int(tile - dim)));
        check(input_query.level() == 8 && input_key.level() == 8 &&
              input_query.scale() == ca.scale() && input_key.scale() == cb.scale(),
              "packed-input pre-CC packing changed input physical metadata");
        Ciphertext input_raw, input_relinearized;
        evaluator->multiply(input_query, input_key, input_raw);
        ++input_cc;
        check(input_raw.level() == 8 && input_raw.size() == 3 && input_raw.scale() == product_scale,
              "packed-input ONE true CC physical metadata mismatch");
        evaluator->relinearize(input_raw, input_relinearized, relin);
        ++input_relin;
        check(input_relinearized.level() == 8 && input_relinearized.size() == 2 &&
              input_relinearized.scale() == product_scale, "packed-input relin changed real scale/level");
        auto input_reduced = input_rescale(input_rescale(input_relinearized));
        const auto input_intermediate_log2 = std::log2(input_reduced.scale());
        check(input_reduced.level() == 6 &&
              std::abs(input_intermediate_log2 - plan.intermediate_scale_log2) < 1e-12,
              "packed-input CC R2 intermediate physical metadata mismatch");
        for (int shift : {1,2,4}) {
            // Deliberately rotate the current accumulated packed CC source.
            input_reduced = input_addition(input_reduced, input_rotate(input_reduced, shift));
        }
        const auto input_a = input_mask(input_reduced, encoded_mask);
        const auto input_b = input_mask(input_reduced, encoded_mask_b);
        const auto input_sum = input_addition(input_a, input_rotate(input_b, int(dim - 1)));
        const auto input_result = input_rescale(input_rescale(input_sum));
        check(input_cc == 1 && input_relin == 1 && input_cp == 2 && input_rot == 6 &&
              input_add == 6 && input_rescales == 4, "packed-input actual CPU arithmetic call counts");
        check(input_result.level() == 4 && input_result.size() == 2 &&
              input_result.scale() == old_same_q_result.scale() && input_result.scale() == result.scale(),
              "packed-input and original same-Q pair output physical metadata differ");
        check(ca.level() == 8 && cb.level() == 8 && ca.scale() == pa.scale() && cb.scale() == pb.scale(),
              "packed-input arithmetic mutated original encrypted Q/K metadata");
        const auto input_scale_error = std::abs(std::log2(input_result.scale()) - spec.output_scale_log2);
        check(input_scale_error < 1e-12, "packed-input final real scale differs from declaration");
        const auto decoded_input = decode_checked(input_result);
        double input_error = 0.0, input_vs_old = 0.0, input_starts_vs_old = 0.0;
        std::size_t input_starts = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            input_error = std::max(input_error, std::abs(decoded_input[i] - expected_input_packed[i]));
            input_vs_old = std::max(input_vs_old, std::abs(decoded_input[i] - decoded_old_same_q[i]));
        }
        for (std::size_t t = 0; t < tokens; ++t) {
            for (std::size_t h = 0; h < 2; ++h) {
                const auto start = t * tile + h * tokens;
                for (std::size_t offset = 0; offset < 2; ++offset) {
                    const auto i = start + offset;
                    const auto difference = std::abs(decoded_input[i] - decoded_old_same_q[i]);
                    check(difference < 1e-8, "packed-input starts differ from original SAME-Q adjacent-key pair");
                    input_starts_vs_old = std::max(input_starts_vs_old, difference);
                    ++input_starts;
                }
            }
        }
        check(input_error < 1e-8 && input_vs_old < 1e-8, "packed-input full-slot CPU decode comparison error");
        std::cout.precision(16);
        std::cout << "{\"cpu_only\":true,\"genuine_cc_relin_rescale2_mask_rescale2\":true,"
            << "\"physical_input_level\":8,\"physical_intermediate_level\":6,\"physical_output_level\":4,"
            << "\"mask_scale_log2\":" << plan.mask_scale_log2 << ",\"scale_log2_error\":" << scale_error
            << ",\"decode_max_abs\":" << error
            << ",\"genuine_paired_two_products\":true,\"paired_compared_head_starts\":" << paired_starts
            << ",\"paired_scale_log2_error\":" << paired_scale_error
            << ",\"separate_second_decode_max_abs\":" << separate_error
            << ",\"paired_decode_max_abs\":" << paired_error
            << ",\"paired_vs_separate_max_abs\":" << paired_vs_separate
            << ",\"genuine_packed_input_same_query_adjacent_keys\":true"
            << ",\"packed_input_q_rotation\":-8,\"packed_input_k_rotation\":24,\"old_pair_key_rotation\":32"
            << ",\"packed_input_actual_cc_calls\":" << input_cc
            << ",\"packed_input_actual_relin_calls\":" << input_relin
            << ",\"packed_input_actual_cp_calls\":" << input_cp
            << ",\"packed_input_actual_rotation_calls\":" << input_rot
            << ",\"packed_input_actual_add_calls\":" << input_add
            << ",\"packed_input_actual_one_prime_rescale_calls\":" << input_rescales
            << ",\"packed_input_compared_slots\":" << a.size()
            << ",\"packed_input_compared_head_starts\":" << input_starts
            << ",\"packed_input_intermediate_scale_log2\":" << input_intermediate_log2
            << ",\"packed_input_output_scale_log2\":" << std::log2(input_result.scale())
            << ",\"packed_input_scale_log2_error\":" << input_scale_error
            << ",\"old_same_query_pair_decode_max_abs\":" << old_same_q_error
            << ",\"packed_input_decode_max_abs\":" << input_error
            << ",\"packed_input_vs_old_pair_all_slots_max_abs\":" << input_vs_old
            << ",\"packed_input_vs_old_pair_starts_max_abs\":" << input_starts_vs_old << "}\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << "QK CPU metadata: " << e.what() << '\n'; return 1; }
}
