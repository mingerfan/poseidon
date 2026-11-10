#include "poseidon/decryptor.h"
#include "poseidon/basics/util/rlwe.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/keygenerator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace poseidon;

namespace
{
using Complex = std::complex<double>;
constexpr double tolerance = 2e-5;
double largest_error = 0;

void require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <typename F> void require_rejected(F &&action, const std::string &label)
{
    try
    {
        action();
    }
    catch (const std::exception &)
    {
        return;
    }
    throw std::runtime_error(label + " was accepted");
}

ParametersLiteral parameters(std::size_t q, std::size_t p, std::uint32_t log_n = 12)
{
    ParametersLiteral result{CKKS, log_n, log_n - 1, 40, 64, 0, 0, {}, {}, sec_level_type::none};
    result.set_log_modulus(std::vector<std::uint32_t>(q, 45), std::vector<std::uint32_t>(p, 45));
    return result;
}

std::vector<Complex> message(std::size_t slots, double phase = 0)
{
    std::vector<Complex> result(slots);
    for (std::size_t i = 0; i < slots; ++i)
    {
        result[i] = {0.2 * std::sin(0.13 * i + phase), 0.1 * std::cos(0.17 * i - phase)};
    }
    return result;
}

std::vector<Complex> rotated(const std::vector<Complex> &input, int step)
{
    auto result = input;
    const auto shift = (step % static_cast<int>(input.size()) + input.size()) % input.size();
    std::rotate(result.begin(), result.begin() + shift, result.end());
    return result;
}

std::vector<Complex> decode(Decryptor &decryptor, CKKSEncoder &encoder, const Ciphertext &cipher)
{
    Plaintext plain;
    std::vector<Complex> result;
    decryptor.decrypt(cipher, plain);
    encoder.decode(plain, result);
    return result;
}

void check_close(const std::vector<Complex> &expected, const std::vector<Complex> &actual,
                 const std::string &label)
{
    require(expected.size() == actual.size(), label + ": slot count changed");
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        const double error = std::abs(actual[i] - expected[i]);
        largest_error = std::max(largest_error, error);
        if (!std::isfinite(error) || error > tolerance)
        {
            throw std::runtime_error(label + ": slot " + std::to_string(i) +
                                     " error=" + std::to_string(error));
        }
    }
}

void check_equal(const Ciphertext &left, const Ciphertext &right, const std::string &label)
{
    require(left.parms_id() == right.parms_id() && left.size() == right.size() &&
                left.scale() == right.scale() && left.is_ntt_form() == right.is_ntt_form() &&
                left.coeff_modulus_size() == right.coeff_modulus_size(),
            label + ": metadata changed");
    const auto words = left.poly_modulus_degree() * left.coeff_modulus_size() * left.size();
    require(std::equal(left.data(), left.data() + words, right.data()), label + ": data changed");
}

void check_sampler_bounds()
{
    // Seed 5 previously sampled coefficient N and overwrote the trailing guard.
    ParametersLiteral params{CKKS, 6, 5, 40, 16, 0, 0, {}, {}, sec_level_type::none};
    params.set_log_modulus({45, 45, 45, 45, 45}, {45, 45});
    PoseidonContext context(params);
    const auto words = 64 * context.crt_context()->key_context_data()->coeff_modulus().size();
    constexpr std::uint64_t guard = 0xabcdef1234567890ULL;
    std::vector<std::uint64_t> buffer(words + 2, 0);
    buffer.front() = buffer.back() = guard;
    prng_seed_type seed{};
    seed[0] = 5;
    util::sample_poly_ternary_with_hamming(context.random_generator()->create(seed), context,
                                           context.crt_context()->key_parms_id(),
                                           buffer.data() + 1);
    require(buffer.front() == guard && buffer.back() == guard,
            "secret sampler wrote outside polynomial");
}

void check_single_prime_modup()
{
    // P straddles the source Q prime. The old Q[i] comparison both skipped
    // necessary reduction into P=13 and read beyond the single-Q array.
    constexpr std::size_t n = 8;
    auto pool = MemoryPoolHandle::New();
    const std::vector<Modulus> q_moduli{Modulus(17)};
    const std::vector<Modulus> p_moduli{Modulus(13), Modulus(19), Modulus(23)};
    util::RNSBase q(q_moduli, pool), p(p_moduli, pool);
    util::RNSToolQP rns(n, q, p, Modulus(0), pool);
    const std::vector<std::uint64_t> source{0, 1, 12, 13, 16, 2, 7, 15};
    std::vector<std::uint64_t> result(n * p_moduli.size());
    rns.mod_up_from_one_base_p(util::ConstRNSIter(source.data(), n), 0,
                               util::RNSIter(result.data(), n), pool);
    for (std::size_t limb = 0; limb < p_moduli.size(); ++limb)
        for (std::size_t i = 0; i < n; ++i)
            require(result[limb * n + i] == source[i] % p_moduli[limb].value(),
                    "single-prime ModUp produced an incorrect P residue");
}

Ciphertext encrypt(CKKSEncoder &encoder, Encryptor &encryptor, const std::vector<Complex> &values)
{
    Plaintext plain;
    Ciphertext cipher;
    encoder.encode(values, std::ldexp(1.0, 40), plain);
    encryptor.encrypt(plain, cipher);
    return cipher;
}

void check_batch(EvaluatorCkksBase &evaluator, Decryptor &decryptor, CKKSEncoder &encoder,
                 const Ciphertext &source, const std::vector<Complex> &values,
                 const std::vector<int> &steps, const GaloisKeys &keys,
                 CpuHoistWorkspace &workspace, std::size_t expected_fallbacks)
{
    Ciphertext original = source;
    check_close(values, decode(decryptor, encoder, source), "source encryption");
    std::vector<Ciphertext> results;
    evaluator.rotate_many_hoisted(source, steps, keys, workspace, results);
    require(results.size() == steps.size(), "batch output count changed");
    const auto nonzero = static_cast<std::size_t>(
        std::count_if(steps.begin(), steps.end(), [](int step) { return step != 0; }));
    const auto direct = nonzero - expected_fallbacks;
    require(workspace.last_counts.source_decompose_count == (direct ? 1U : 0U),
            "batch did not share one decomposition");
    require(workspace.last_counts.key_switch_count == direct &&
                workspace.last_counts.moddown_count == direct &&
                workspace.last_counts.fallback_rotation_count == expected_fallbacks,
            "unexpected staged/fallback counts");
    check_equal(original, source, "batch mutated source");
    for (std::size_t i = 0; i < steps.size(); ++i)
    {
        require(results[i].parms_id() == source.parms_id() &&
                    results[i].scale() == source.scale() && results[i].size() == 2 &&
                    results[i].is_ntt_form(),
                "rotation metadata changed");
        Ciphertext ordinary;
        evaluator.rotate(source, ordinary, steps[i], keys);
        const auto label = "Q=" + std::to_string(source.coeff_modulus_size()) +
                           " step=" + std::to_string(steps[i]);
        check_close(rotated(values, steps[i]), decode(decryptor, encoder, ordinary),
                    label + " ordinary oracle");
        check_close(rotated(values, steps[i]), decode(decryptor, encoder, results[i]),
                    label + " hoisted oracle");
        check_close(decode(decryptor, encoder, ordinary), decode(decryptor, encoder, results[i]),
                    "ordinary rotation comparison");
        if (steps[i] == 0)
            check_equal(source, results[i], "identity rotation");
    }
}

void check_bsgs(PoseidonContext &context, EvaluatorCkksBase &evaluator, KeyGenerator &keygen,
                Decryptor &decryptor, CKKSEncoder &encoder, const Ciphertext &source,
                const std::vector<Complex> &values)
{
    // Independent diagonal formula: y[k] = sum_d diagonal[d][k] * x[k+d].
    // Nonconstant complex diagonals also exercise the BSGS pre-rotation layout.
    std::map<int, std::vector<Complex>> diagonals;
    const std::vector<int> offsets{0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 15};
    std::vector<Complex> expected(values.size(), Complex{});
    for (int offset : offsets)
    {
        auto &diagonal = diagonals[offset];
        diagonal.resize(values.size());
        for (std::size_t k = 0; k < values.size(); ++k)
        {
            diagonal[k] = {0.05 + 0.01 * std::sin(0.07 * k + offset),
                           0.02 * std::cos(0.11 * k - offset)};
            expected[k] += diagonal[k] * values[(k + offset) % values.size()];
        }
    }
    const auto data = context.crt_context()->get_context_data(source.parms_id());
    const auto scale = static_cast<double>(data->parms().q().back().value());
    MatrixPlain matrix;
    gen_linear_transform_bsgs(matrix, matrix.rot_index, encoder, diagonals,
                              static_cast<std::uint32_t>(data->level()), scale, 1,
                              context.parameters_literal()->log_slots(), 4);
    GaloisKeys keys;
    keygen.create_galois_keys(matrix.rot_index, keys);
    Ciphertext actual;
    evaluator.multiply_by_diag_matrix_bsgs(source, matrix, actual, keys);
    check_close(expected, decode(decryptor, encoder, actual), "BSGS slot oracle");

    // Reference uses direct diagonals and individual ordinary rotations, rather
    // than reproducing the optimized BSGS grouping or its hoist calls.
    std::vector<int> nonzero_offsets(offsets.begin() + 1, offsets.end());
    GaloisKeys reference_keys;
    keygen.create_galois_keys(nonzero_offsets, reference_keys);
    Ciphertext sum;
    for (int offset : offsets)
    {
        Plaintext plain;
        encoder.encode(diagonals.at(offset), source.parms_id(), scale, plain);
        Ciphertext rotation;
        if (offset != 0)
            evaluator.rotate(source, rotation, offset, reference_keys);
        evaluator.multiply_plain_accumulate(offset == 0 ? source : rotation, plain, sum);
    }
    Ciphertext reference;
    evaluator.rescale_dynamic(sum, reference, source.scale());
    require(actual.parms_id() == reference.parms_id() && actual.scale() == reference.scale(),
            "BSGS changed rescale semantics");
    check_close(decode(decryptor, encoder, reference), decode(decryptor, encoder, actual),
                "ordinary diagonal reference");

    // Pre-fill three ring-buffer entries to exercise the streamed BSGS entry
    // point without a producer thread or changing its synchronization protocol.
    for (int offset : {12, 13, 15})
        diagonals.erase(offset);
    MatrixPlain streamed;
    gen_linear_transform_bsgs(streamed, streamed.rot_index, encoder, diagonals,
                              static_cast<std::uint32_t>(data->level()), scale, 1,
                              context.parameters_literal()->log_slots(), 4);
    auto [groups, giants, babies] = bsgs_index(streamed.plain_vec, values.size(), streamed.n1);
    require(groups.size() == 3, "streamed fixture needs three groups");
    for (std::size_t i = 0; i < groups.size(); ++i)
        streamed.plain_vec_pool[i] = streamed.plain_vec;
    streamed.write_idx = 3;
    evaluator.multiply_by_diag_matrix_bsgs(source, streamed, reference, keys);
    evaluator.multiply_by_diag_matrix_bsgs_with_mutex(source, streamed, actual, keys, groups,
                                                      giants, babies);
    check_close(decode(decryptor, encoder, reference), decode(decryptor, encoder, actual),
                "streamed BSGS comparison");
}

void run_case(std::size_t q_count, std::size_t p_count)
{
    auto context =
        PoseidonFactory::get_instance()->create_poseidon_context(parameters(q_count, p_count));
    auto evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);
    KeyGenerator keygen(context);
    require(!keygen.secret_key().data().is_zero(), "regression requires a nonzero secret key");
    PublicKey public_key;
    keygen.create_public_key(public_key);
    const std::vector<int> steps{1, -1, 2, -2, 3, -3, 0, 1};
    GaloisKeys keys;
    keygen.create_galois_keys(std::vector<int>{1, -1, 2, -2, 3, -3}, keys);
    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, keygen.secret_key());
    const auto values = message(context.parameters_literal()->slot());
    auto source = encrypt(encoder, encryptor, values);
    CpuHoistWorkspace workspace;
    const bool hybrid = context.key_switch_variant() == HYBRID;
    for (std::size_t q = q_count; q > 0; --q)
    {
        check_batch(*evaluator, decryptor, encoder, source, values, steps, keys, workspace,
                    hybrid ? 0 : 7);
        if (q > 1)
            evaluator->drop_modulus_to_next(source, source);
    }

    source = encrypt(encoder, encryptor, values);
    check_batch(*evaluator, decryptor, encoder, source, values, steps, keys, workspace,
                hybrid ? 0 : 7);
    const auto retained = workspace.allocated_bytes();
    const auto other_values = message(values.size(), 0.9);
    auto other = encrypt(encoder, encryptor, other_values);
    check_batch(*evaluator, decryptor, encoder, other, other_values, steps, keys, workspace,
                hybrid ? 0 : 7);
    require(retained == workspace.allocated_bytes(), "same-shape workspace grew");

    // Alias-safe output commit and zero-only/empty batches with no keys.
    std::vector<Ciphertext> aliased{source};
    evaluator->rotate_many_hoisted(aliased.front(), {1, 0, -1}, keys, workspace, aliased);
    check_close(rotated(values, -1), decode(decryptor, encoder, aliased[2]), "aliased batch");
    check_equal(source, aliased[1], "aliased identity");
    evaluator->rotate_many_hoisted(source, {0, 0}, GaloisKeys{}, workspace, aliased);
    require(workspace.last_counts.source_decompose_count == 0, "identity batch decomposed");
    check_equal(source, aliased[0], "identity without keys");
    evaluator->rotate_many_hoisted(source, {}, GaloisKeys{}, workspace, aliased);
    require(aliased.empty(), "empty batch has outputs");

    if (hybrid)
    {
        GaloisKeys sparse_keys;
        keygen.create_galois_keys(std::vector<int>{1, -1, 2, -2, 4, -4}, sparse_keys);
        check_batch(*evaluator, decryptor, encoder, source, values, {1, 3, -3, 0, 2, 1},
                    sparse_keys, workspace, 2);
        check_batch(*evaluator, decryptor, encoder, source, values, {3, -3}, sparse_keys, workspace,
                    2);

        aliased = {source};
        require_rejected(
            [&] {
                evaluator->rotate_many_hoisted(source, {1, 16}, sparse_keys, workspace, aliased);
            },
            "missing key");
        require(aliased.size() == 1, "failed batch changed output count");
        check_equal(source, aliased.front(), "failed batch changed output");
        auto invalid = source;
        invalid.is_ntt_form() = false;
        require_rejected(
            [&] { evaluator->rotate_many_hoisted(invalid, {1}, keys, workspace, aliased); },
            "coefficient-form input");
        invalid = source;
        invalid.resize(context, source.parms_id(), 3);
        require_rejected(
            [&] { evaluator->rotate_many_hoisted(invalid, {1}, keys, workspace, aliased); },
            "size-3 input");
        require_rejected(
            [&] { evaluator->rotate_many_hoisted(Ciphertext{}, {1}, keys, workspace, aliased); },
            "empty input");
        require_rejected(
            [&] {
                evaluator->rotate_many_hoisted(source, {static_cast<int>(values.size())}, keys,
                                               workspace, aliased);
            },
            "large step");
        require_rejected(
            [&] {
                evaluator->rotate_many_hoisted(source, {std::numeric_limits<int>::min()}, keys,
                                               workspace, aliased);
            },
            "INT_MIN step");
        GaloisKeys malformed = keys;
        const auto elt = context.crt_context()->galois_tool()->get_elt_from_step(1);
        malformed.data()[GaloisKeys::get_index(elt)].resize(1);
        if ((q_count + p_count - 1) / p_count > 1)
        {
            require_rejected(
                [&] { evaluator->rotate_many_hoisted(source, {1}, malformed, workspace, aliased); },
                "short key digit vector");
        }
        malformed = keys;
        malformed.data()[GaloisKeys::get_index(elt)][0] = PublicKey{};
        require_rejected(
            [&] { evaluator->rotate_many_hoisted(source, {1}, malformed, workspace, aliased); },
            "empty key component");
        malformed = keys;
        malformed.parms_id() = parms_id_zero;
        require_rejected(
            [&] { evaluator->rotate_many_hoisted(source, {1}, malformed, workspace, aliased); },
            "wrong key parameters");
    }
    check_bsgs(context, *evaluator, keygen, decryptor, encoder, source, values);
    std::cout << "PASS Q=" << q_count << " P=" << p_count
              << " variant=" << static_cast<int>(context.key_switch_variant()) << '\n';
}

void benchmark()
{
    auto context = PoseidonFactory::get_instance()->create_poseidon_context(parameters(10, 3, 13));
    auto evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);
    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    std::vector<int> steps;
    for (int step = 1; step <= 16; ++step)
        steps.push_back(step);
    GaloisKeys keys;
    keygen.create_galois_keys(steps, keys);
    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, keygen.secret_key());
    const auto values = message(context.parameters_literal()->slot());
    const auto source = encrypt(encoder, encryptor, values);
    CpuHoistWorkspace workspace;
    std::vector<Ciphertext> outputs(steps.size());
    auto ordinary = [&] {
        for (std::size_t i = 0; i < steps.size(); ++i)
            evaluator->rotate(source, outputs[i], steps[i], keys);
    };
    auto hoisted = [&] { evaluator->rotate_many_hoisted(source, steps, keys, workspace, outputs); };
    ordinary();
    hoisted();
    std::vector<double> ordinary_ms, hoisted_ms;
    auto measure = [](auto &&action) {
        const auto start = std::chrono::steady_clock::now();
        action();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();
    };
    for (int repeat = 0; repeat < 5; ++repeat)
    {
        // Alternate order so one backend does not always run first.
        if (repeat % 2)
        {
            hoisted_ms.push_back(measure(hoisted));
            ordinary_ms.push_back(measure(ordinary));
        }
        else
        {
            ordinary_ms.push_back(measure(ordinary));
            hoisted_ms.push_back(measure(hoisted));
        }
    }
    hoisted();
    for (std::size_t i = 0; i < steps.size(); ++i)
        check_close(rotated(values, steps[i]), decode(decryptor, encoder, outputs[i]),
                    "benchmark oracle");
    std::sort(ordinary_ms.begin(), ordinary_ms.end());
    std::sort(hoisted_ms.begin(), hoisted_ms.end());
    std::cout << "BENCH N=8192 Q=10 P=3 rotations=16 median_of=5 ordinary_ms=" << ordinary_ms[2]
              << " hoisted_ms=" << hoisted_ms[2] << " ratio=" << ordinary_ms[2] / hoisted_ms[2]
              << " retained_workspace_bytes=" << workspace.allocated_bytes() << '\n';
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        PoseidonFactory::get_instance()->set_device_type(DEVICE_SOFTWARE);
        check_single_prime_modup();
        check_sampler_bounds();
        for (const auto [q, p] : std::vector<std::pair<std::size_t, std::size_t>>{
                 {5, 2}, {8, 3}, {3, 5}, {4, 1}, {3, 3}})
            run_case(q, p);
        if (argc == 2 && std::string(argv[1]) == "--benchmark")
            benchmark();
        std::cout << "CPU hoist regressions passed; maximum slot error=" << largest_error << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "CPU hoist regression failed: " << error.what() << '\n';
        return 1;
    }
}
