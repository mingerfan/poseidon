#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/advance/bootstrapper.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/evaluator/evaluator_ckks_base.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/keygenerator.h"
#include "poseidon/util/debug.h"
#include "poseidon/util/random_sample.h"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace poseidon;

namespace
{

std::vector<uint32_t> bootstrap_modulus_chain()
{
    std::vector<uint32_t> chain(16, 51);
    chain[1] = 46;
    // chain[2] = 46;
    return chain;
}

std::vector<uint32_t> bootstrap_modulus_chain_32()
{
    // The 32-bit EvalMod schedule consumes one more level than the 51-bit
    // reference schedule, so retain one additional 32-bit working prime.
    std::vector<uint32_t> chain(17, 32);
    chain[1] = 27;
    return chain;
}

std::vector<uint32_t> bootstrap_modulus_chain_dual_30()
{
    // One logical bootstrap level is represented by two physical 30-bit
    // primes. Two leading primes form the centered ModRaise q0 base.
    return std::vector<uint32_t>(34, 30);
}

void print_error(const std::vector<std::complex<double>> &actual,
                 const std::vector<std::complex<double>> &expected)
{
    double max_error = 0.0;
    double squared_error_sum = 0.0;
    std::size_t max_error_slot = 0;

    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const double error = std::abs(actual[i] - expected[i]);
        squared_error_sum += error * error;
        if (error > max_error)
        {
            max_error = error;
            max_error_slot = i;
        }
    }

    const double rmse =
        std::sqrt(squared_error_sum / static_cast<double>(expected.size()));
    std::cout << "max abs error : " << max_error << " at slot " << max_error_slot << '\n';
    std::cout << "rmse          : " << rmse << '\n';
}

std::vector<std::complex<double>> decode_ciphertext(
    const Ciphertext &cipher, Decryptor &decryptor, CKKSEncoder &encoder)
{
    Plaintext plain;
    std::vector<std::complex<double>> decoded;
    decryptor.decrypt(cipher, plain);
    encoder.decode(plain, decoded);
    return decoded;
}

std::vector<std::complex<double>> scale_values(
    const std::vector<std::complex<double>> &values, double factor)
{
    auto scaled = values;
    for (auto &value : scaled)
    {
        value *= factor;
    }
    return scaled;
}

void print_trace_row(const std::string &name, const Ciphertext &cipher,
                     const std::vector<std::complex<double>> &decoded,
                     const std::vector<std::complex<double>> *expected = nullptr)
{
    double max_abs = 0.0;
    double max_error = 0.0;
    for (std::size_t index = 0; index < decoded.size(); ++index)
    {
        max_abs = std::max(max_abs, std::abs(decoded[index]));
        if (expected)
        {
            max_error = std::max(max_error,
                                 std::abs(decoded[index] - (*expected)[index]));
        }
    }

    std::cout << "| " << std::left << std::setw(30) << name
              << " | " << std::right << std::setw(3) << cipher.coeff_modulus_size()
              << " | " << std::setw(5) << cipher.level()
              << " | " << std::setw(11) << std::fixed << std::setprecision(4)
              << std::log2(cipher.scale())
              << " | " << std::setw(13) << std::scientific << std::setprecision(5)
              << max_abs << " | ";
    if (expected)
    {
        std::cout << std::setw(13) << max_error;
    }
    else
    {
        std::cout << std::setw(13) << "-";
    }
    std::cout << " |\n";
}

void print_bootstrap_trace(const BootstrapTrace &trace,
                           const std::vector<std::complex<double>> &source,
                           uint32_t output_ratio, Decryptor &decryptor,
                           CKKSEncoder &encoder,
                           const Bootstrapper &plain_eval_mod,
                           uint32_t double_angle, double inverse_coeff)
{
    const auto prepared = decode_ciphertext(trace.prepared_q0, decryptor, encoder);
    const auto raised = decode_ciphertext(trace.raised, decryptor, encoder);
    const auto c2s_real_raw =
        decode_ciphertext(trace.coeff_to_slot_real_raw, decryptor, encoder);
    const auto c2s_imag_raw =
        decode_ciphertext(trace.coeff_to_slot_imag_raw, decryptor, encoder);
    const auto c2s_real_aligned =
        decode_ciphertext(trace.coeff_to_slot_real_aligned, decryptor, encoder);
    const auto c2s_imag_aligned =
        decode_ciphertext(trace.coeff_to_slot_imag_aligned, decryptor, encoder);

    std::cout << "\n[CPU dual-30 bootstrap stage trace]\n"
              << "| " << std::left << std::setw(30) << "stage"
              << " | " << std::right << std::setw(3) << "q"
              << " | " << std::setw(5) << "level"
              << " | " << std::setw(11) << "log2(scale)"
              << " | " << std::setw(13) << "max |value|"
              << " | " << std::setw(13) << "max error" << " |\n"
              << "|--------------------------------|-----|-------|-------------|"
                 "---------------|---------------|\n";

    print_trace_row("prepared_q0 / source", trace.prepared_q0, prepared, &source);
    print_trace_row("raised (q0 aliases expected)", trace.raised, raised);
    print_trace_row("c2s_real_raw", trace.coeff_to_slot_real_raw, c2s_real_raw);
    print_trace_row("c2s_real_aligned / raw", trace.coeff_to_slot_real_aligned,
                    c2s_real_aligned, &c2s_real_raw);
    print_trace_row("c2s_imag_raw", trace.coeff_to_slot_imag_raw, c2s_imag_raw);
    print_trace_row("c2s_imag_aligned / raw", trace.coeff_to_slot_imag_aligned,
                    c2s_imag_aligned, &c2s_imag_raw);

    const auto real_poly = decode_ciphertext(
        trace.eval_mod_real.polynomial_output, decryptor, encoder);
    const auto imag_poly = decode_ciphertext(
        trace.eval_mod_imag.polynomial_output, decryptor, encoder);
    std::vector<std::vector<std::complex<double>>> real_eval_expected(
        static_cast<std::size_t>(double_angle) + 1,
        std::vector<std::complex<double>>(c2s_real_aligned.size()));
    std::vector<std::vector<std::complex<double>>> imag_eval_expected(
        static_cast<std::size_t>(double_angle) + 1,
        std::vector<std::complex<double>>(c2s_imag_aligned.size()));
    for (std::size_t slot = 0; slot < c2s_real_aligned.size(); ++slot)
    {
        const auto real_stages = plain_eval_mod.eval_mod_plain_trace(
            c2s_real_aligned[slot], double_angle, inverse_coeff);
        const auto imag_stages = plain_eval_mod.eval_mod_plain_trace(
            c2s_imag_aligned[slot], double_angle, inverse_coeff);
        for (std::size_t stage = 0; stage < real_stages.size(); ++stage)
        {
            real_eval_expected[stage][slot] = real_stages[stage];
            imag_eval_expected[stage][slot] = imag_stages[stage];
        }
    }
    print_trace_row("evalmod_real_polynomial",
                    trace.eval_mod_real.polynomial_output, real_poly,
                    &real_eval_expected[0]);
    print_trace_row("evalmod_imag_polynomial",
                    trace.eval_mod_imag.polynomial_output, imag_poly,
                    &imag_eval_expected[0]);

    for (std::size_t index = 0;
         index < trace.eval_mod_real.double_angle_outputs.size(); ++index)
    {
        const auto decoded = decode_ciphertext(
            trace.eval_mod_real.double_angle_outputs[index], decryptor, encoder);
        print_trace_row("evalmod_real_da_" + std::to_string(index + 1),
                        trace.eval_mod_real.double_angle_outputs[index], decoded,
                        &real_eval_expected[index + 1]);
    }
    for (std::size_t index = 0;
         index < trace.eval_mod_imag.double_angle_outputs.size(); ++index)
    {
        const auto decoded = decode_ciphertext(
            trace.eval_mod_imag.double_angle_outputs[index], decryptor, encoder);
        print_trace_row("evalmod_imag_da_" + std::to_string(index + 1),
                        trace.eval_mod_imag.double_angle_outputs[index], decoded,
                        &imag_eval_expected[index + 1]);
    }

    const auto s2c = decode_ciphertext(trace.slot_to_coeff, decryptor, encoder);
    const auto s2c_expected = scale_values(source, 1.0 / output_ratio);
    print_trace_row("slot_to_coeff / source/ratio", trace.slot_to_coeff, s2c,
                    &s2c_expected);

    const auto projected = decode_ciphertext(trace.projected, decryptor, encoder);
    const double projected_ratio = output_ratio / 2.0;
    const auto projected_expected = scale_values(source, 1.0 / projected_ratio);
    print_trace_row("projected / source/(ratio/2)", trace.projected, projected,
                    &projected_expected);

    const auto final_output =
        decode_ciphertext(trace.final_output, decryptor, encoder);
    print_trace_row("final / source", trace.final_output, final_output, &source);
    std::cout.unsetf(std::ios::floatfield);
}

int run_legacy_bootstrap()
{
    std::cout << "\nLegacy bootstrap test\n";

    ParametersLiteral parameters{CKKS, 15, 14, 32, 1, 1, 0, {}, {}};
    std::vector<uint32_t> log_q(30, 32);
    parameters.set_log_modulus(log_q, {32});

    PoseidonFactory::get_instance()->set_device_type(DEVICE_SOFTWARE);
    auto context = PoseidonFactory::get_instance()->create_poseidon_context(parameters);
    auto evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);

    const int slot_count = 1 << parameters.log_slots();
    std::vector<std::complex<double>> source;
    sample_random_complex_vector(source, slot_count);
    for (auto &value : source)
    {
        value = std::sin(value);
    }

    PublicKey public_key;
    RelinKeys relin_keys;
    GaloisKeys galois_keys;
    CKKSEncoder encoder(context);
    KeyGenerator keygen(context);
    keygen.create_public_key(public_key);
    keygen.create_relin_keys(relin_keys);
    keygen.create_galois_keys(galois_keys);
    Encryptor encryptor(context, public_key, keygen.secret_key());
    Decryptor decryptor(context, keygen.secret_key());

    Plaintext plain;
    Ciphertext cipher;
    encoder.encode(source, static_cast<int64_t>(1) << 40, plain);
    encryptor.encrypt(plain, cipher);

    const auto start = std::chrono::high_resolution_clock::now();
    evaluator->multiply_relin(cipher, cipher, cipher, relin_keys);
    evaluator->rescale_dynamic(cipher, cipher, static_cast<int64_t>(1) << 45);

    EvalModPoly eval_mod_poly(context, CosDiscrete, static_cast<uint64_t>(1) << 40,
                              1, 9, 3, 16, 0, 30);
    evaluator->bootstrap(cipher, cipher, relin_keys, galois_keys, encoder, eval_mod_poly);
    const auto stop = std::chrono::high_resolution_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << "Bootstrap TIME: " << elapsed << " microseconds\n";

    Plaintext result_plain;
    std::vector<std::complex<double>> result;
    decryptor.decrypt(cipher, result_plain);
    encoder.decode(result_plain, result);
    for (int i = 0; i < 10; ++i)
    {
        source[static_cast<std::size_t>(i)] *= source[static_cast<std::size_t>(i)];
        std::printf("source vec[%d] : %0.10f + %0.10f I \n", i,
                    std::real(source[static_cast<std::size_t>(i)]),
                    std::imag(source[static_cast<std::size_t>(i)]));
        std::printf("result vec[%d] : %0.10f + %0.10f I \n", i,
                    std::real(result[static_cast<std::size_t>(i)]),
                    std::imag(result[static_cast<std::size_t>(i)]));
    }
    GetPrecisionStats(result, source);
    return 0;
}

int run_new_bootstrap()
{
    std::cout << "\nNew 14-level bootstrap test\n";

    constexpr uint32_t log_n = 16;
    constexpr uint32_t log_slots = log_n - 1;
    ParametersLiteral parameters{CKKS, log_n, log_slots, 46, 5, 0, 0, {}, {}};
    const auto q_chain = bootstrap_modulus_chain();
    parameters.set_log_modulus(q_chain, {51});

    PoseidonFactory::get_instance()->set_device_type(DEVICE_SOFTWARE);
    auto context = PoseidonFactory::get_instance()->create_poseidon_context(parameters);
    auto evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);

    KeyGenerator keygen(context);
    PublicKey public_key;
    RelinKeys relin_keys;
    GaloisKeys galois_keys;
    keygen.create_public_key(public_key);
    keygen.create_relin_keys(relin_keys);
    keygen.create_galois_keys(galois_keys);

    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key, keygen.secret_key());
    Decryptor decryptor(context, keygen.secret_key());

    std::vector<std::complex<double>> source(encoder.slot_count());
    for (std::size_t i = 0; i < source.size(); ++i)
    {
        source[i] = std::sin(static_cast<double>(i) / 32.0);
    }

    Plaintext plain;
    encoder.encode(source, parameters.scale(), plain);
    Ciphertext input;
    encryptor.encrypt(plain, input);

    BootstrapConfig config;
    config.boundary_k = 25;
    config.log_message_ratio = 5;
    config.double_angle = 2;
    config.scaling_log = 51;
    config.output_ratio = 32;
    config.project_real = true;

    std::cout << "bootstrap config: boundary_k=" << config.boundary_k
              << " log_message_ratio=" << config.log_message_ratio
              << " double_angle=" << config.double_angle
              << " scaling_log=" << config.scaling_log << '\n';

    Ciphertext output;
    const auto start = std::chrono::high_resolution_clock::now();
    evaluator->bootstrap(input, output, relin_keys, galois_keys, encoder, config);
    const auto stop = std::chrono::high_resolution_clock::now();

    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    const auto raised_level = context.crt_context()->first_context_data()->level();
    std::cout << "bootstrap levels : " << raised_level - output.level() << " ("
              << raised_level << " -> " << output.level() << ")\n";
    std::cout << "bootstrap time   : " << elapsed << " ms\n";

    Plaintext result_plain;
    std::vector<std::complex<double>> result;
    decryptor.decrypt(output, result_plain);
    encoder.decode(result_plain, result);

    std::cout << "source preview   :";
    for (std::size_t i = 0; i < 8; ++i)
    {
        std::cout << ' ' << source[i].real();
    }
    std::cout << '\n';
    std::cout << "result preview   :";
    for (std::size_t i = 0; i < 8; ++i)
    {
        std::cout << ' ' << result[i].real();
    }
    std::cout << '\n';
    print_error(result, source);

    return 0;
}

int run_new_bootstrap_32()
{
    std::cout << "\nNew 32/27-bit bootstrap test\n";

    constexpr uint32_t log_n = 16;
    constexpr uint32_t log_slots = log_n - 1;
    constexpr uint32_t log_scale = 27;
    constexpr uint32_t scaling_log = 32;
    ParametersLiteral parameters{
        CKKS, log_n, log_slots, log_scale, 5, 0, 0, {}, {}};
    const auto q_chain = bootstrap_modulus_chain_32();
    parameters.set_log_modulus(q_chain, {32, 32, 32, 32, 32});

    PoseidonFactory::get_instance()->set_device_type(DEVICE_SOFTWARE);
    auto context = PoseidonFactory::get_instance()->create_poseidon_context(parameters);
    auto evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);

    KeyGenerator keygen(context);
    PublicKey public_key;
    RelinKeys relin_keys;
    GaloisKeys galois_keys;
    keygen.create_public_key(public_key);
    keygen.create_relin_keys(relin_keys);
    keygen.create_galois_keys(galois_keys);

    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key, keygen.secret_key());
    Decryptor decryptor(context, keygen.secret_key());

    std::vector<std::complex<double>> source(encoder.slot_count());
    for (std::size_t i = 0; i < source.size(); ++i)
    {
        source[i] = std::sin(static_cast<double>(i) / 32.0);
    }

    Plaintext plain;
    encoder.encode(source, parameters.scale(), plain);
    Ciphertext input;
    encryptor.encrypt(plain, input);

    Plaintext input_plain;
    std::vector<std::complex<double>> input_decoded;
    decryptor.decrypt(input, input_plain);
    encoder.decode(input_plain, input_decoded);
    std::cout << "\n[input encryption correctness]\n";
    print_error(input_decoded, source);

    BootstrapConfig config;
    config.boundary_k = 25;
    config.log_message_ratio = 5;
    config.double_angle = 2;
    config.scaling_log = scaling_log;
    config.output_ratio = 32;
    config.project_real = true;

    std::cout << "q chain bits    :";
    for (const auto bits : q_chain)
    {
        std::cout << ' ' << bits;
    }
    std::cout << "\np chain bits    : 32 32 32 32 32"
              << "\ninput log scale : " << log_scale
              << "\nEvalMod log scale: " << scaling_log
              << "\nbootstrap config: boundary_k=" << config.boundary_k
              << " log_message_ratio=" << config.log_message_ratio
              << " double_angle=" << config.double_angle
              << " scaling_log=" << config.scaling_log << '\n';

    Ciphertext output;
    const auto start = std::chrono::high_resolution_clock::now();
    evaluator->bootstrap(input, output, relin_keys, galois_keys, encoder, config);
    const auto stop = std::chrono::high_resolution_clock::now();

    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    const auto raised_level = context.crt_context()->first_context_data()->level();
    std::cout << "bootstrap levels : " << raised_level - output.level() << " ("
              << raised_level << " -> " << output.level() << ")\n";
    std::cout << "output log scale : " << std::log2(output.scale()) << '\n';
    std::cout << "bootstrap time   : " << elapsed << " ms\n";

    Plaintext result_plain;
    std::vector<std::complex<double>> result;
    decryptor.decrypt(output, result_plain);
    encoder.decode(result_plain, result);

    std::cout << "source preview   :";
    for (std::size_t i = 0; i < 8; ++i)
    {
        std::cout << ' ' << source[i].real();
    }
    std::cout << '\n';
    std::cout << "result preview   :";
    for (std::size_t i = 0; i < 8; ++i)
    {
        std::cout << ' ' << result[i].real();
    }
    std::cout << '\n';
    print_error(result, source);

    double bootstrap_max_error = 0.0;
    for (std::size_t i = 0; i < source.size(); ++i)
    {
        bootstrap_max_error =
            std::max(bootstrap_max_error, std::abs(result[i] - source[i]));
    }
    constexpr double correctness_tolerance = 1.0e-3;
    const bool correct = bootstrap_max_error <= correctness_tolerance;
    std::cout << "correctness tol  : " << correctness_tolerance
              << "\ncorrect          : " << (correct ? "YES" : "NO") << '\n';

    return correct ? 0 : 2;
}

int run_new_bootstrap_dual_30()
{
    std::cout << "\nCPU dual-30-bit logical-rescale bootstrap test\n";

    constexpr uint32_t log_n = 14;
    constexpr uint32_t log_slots = log_n - 1;
    constexpr uint32_t log_scale = 30;
    constexpr uint32_t scaling_log = 60;
    constexpr uint32_t q0_level = 1;
    ParametersLiteral parameters{
        CKKS, log_n, log_slots, log_scale, 5, q0_level, 0, {}, {}};
    const auto q_chain = bootstrap_modulus_chain_dual_30();
    parameters.set_log_modulus(q_chain, {30, 30, 30, 30, 30});

    PoseidonFactory::get_instance()->set_device_type(DEVICE_SOFTWARE);
    auto context = PoseidonFactory::get_instance()->create_poseidon_context(parameters);
    auto evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);

    KeyGenerator keygen(context);
    PublicKey public_key;
    RelinKeys relin_keys;
    GaloisKeys galois_keys;
    keygen.create_public_key(public_key);
    keygen.create_relin_keys(relin_keys);
    keygen.create_galois_keys(galois_keys);

    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key, keygen.secret_key());
    Decryptor decryptor(context, keygen.secret_key());

    std::vector<std::complex<double>> source(encoder.slot_count());
    for (std::size_t i = 0; i < source.size(); ++i)
    {
        source[i] = std::sin(static_cast<double>(i) / 32.0);
    }

    Plaintext plain;
    encoder.encode(source, parameters.scale(), plain);
    Ciphertext input;
    encryptor.encrypt(plain, input);

    Plaintext input_plain;
    std::vector<std::complex<double>> input_decoded;
    decryptor.decrypt(input, input_plain);
    encoder.decode(input_plain, input_decoded);
    std::cout << "\n[input encryption correctness]\n";
    print_error(input_decoded, source);

    BootstrapConfig config;
    config.boundary_k = 25;
    config.log_message_ratio = 5;
    config.double_angle = 2;
    config.scaling_log = scaling_log;
    config.output_ratio = 32;
    config.project_real = true;
    config.logical_rescale_count = 2;
    config.q0_modulus_count = 2;
    BootstrapTrace trace;
    config.trace = &trace;

    std::cout << "degree            : " << (1ULL << log_n)
              << "\nq prime count      : " << q_chain.size()
              << "\np prime count      : 5"
              << "\nphysical prime bits: 30"
              << "\ninput log scale    : " << log_scale
              << "\nEvalMod log scale  : " << scaling_log
              << "\nq0 physical primes : " << config.q0_modulus_count
              << "\nprimes per rescale : " << config.logical_rescale_count
              << "\nS2C normalization : "
              << context.crt_context()->q0() / std::ldexp(1.0, scaling_log)
              << '\n';

    Ciphertext output;
    const auto start = std::chrono::high_resolution_clock::now();
    evaluator->bootstrap(input, output, relin_keys, galois_keys, encoder, config);
    const auto stop = std::chrono::high_resolution_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    Plaintext result_plain;
    std::vector<std::complex<double>> result;
    decryptor.decrypt(output, result_plain);
    encoder.decode(result_plain, result);

    const auto raised_level = context.crt_context()->first_context_data()->level();
    std::cout << "bootstrap levels   : " << raised_level - output.level() << " ("
              << raised_level << " -> " << output.level() << ")\n"
              << "output log scale   : " << std::log2(output.scale()) << '\n'
              << "bootstrap time     : " << elapsed << " ms\n"
              << "\n[decrypted output versus source plaintext]\n";
    print_error(result, source);
    Bootstrapper plain_eval_mod(
        context, *evaluator, encoder, parameters.log_slots(), config.boundary_k,
        input.scale(), parameters.scale(), config.cosine_heap_path,
        config.logical_rescale_count, config.q0_modulus_count,
        context.crt_context()->q0());
    const double inverse_coeff =
        config.inverse_coeff > 0.0
            ? config.inverse_coeff
            : plain_eval_mod.inverse_coefficient(config.double_angle);
    print_bootstrap_trace(trace, source, config.output_ratio, decryptor, encoder,
                          plain_eval_mod, config.double_angle, inverse_coeff);

    double bootstrap_max_error = 0.0;
    for (std::size_t i = 0; i < source.size(); ++i)
    {
        bootstrap_max_error =
            std::max(bootstrap_max_error, std::abs(result[i] - source[i]));
    }
    constexpr double correctness_tolerance = 1.0e-3;
    const bool correct = bootstrap_max_error <= correctness_tolerance;
    std::cout << "correctness tol    : " << correctness_tolerance
              << "\ncorrect            : " << (correct ? "YES" : "NO") << '\n';
    return correct ? 0 : 2;
}

} // namespace

int main(int argc, char **argv)
{
    std::cout << BANNER << '\n';
    std::cout << "POSEIDON SOFTWARE VERSION: " << POSEIDON_VERSION << "\n";

    if (argc == 1)
    {
        return run_legacy_bootstrap();
    }

    const std::string mode = argv[1];
    if (mode == "--new")
    {
        return run_new_bootstrap();
    }
    if (mode == "--new-32")
    {
        return run_new_bootstrap_32();
    }
    if (mode == "--new-dual-30")
    {
        return run_new_bootstrap_dual_30();
    }
    if (mode == "--all")
    {
        const int legacy_status = run_legacy_bootstrap();
        return legacy_status == 0 ? run_new_bootstrap() : legacy_status;
    }

    std::cerr << "usage: " << argv[0]
              << " [--new|--new-32|--new-dual-30|--all]\n";
    return 1;
}
