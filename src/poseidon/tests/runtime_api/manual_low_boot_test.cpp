// Standalone native-Boot regression (no RuntimePlan/boot_step dependency).
// Usage: manual_low_boot_test relu_precision_fixture.txt [--host|--gpu-low|--gpu-high]
// --host (default) does not initialize CUDA. GPU modes need N65536-capable memory.
#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/advance/homomorphic_mod.h"
#include "poseidon/basics/util/rlwe.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_bootstrap_profile.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace poseidon;
using namespace poseidon::gpu;
namespace
{
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

PoseidonContext make_context(const std::string &fixture, unsigned log_n)
{
    std::ifstream input(fixture);
    std::string magic;
    std::size_t nq = 0, np = 0;
    input >> magic >> nq >> np;
    require(magic == "RELU_PRECISION_V1" && nq == 50 && np == 25, "Q50/P25 fixture required");
    std::vector<Modulus> q, p;
    for (std::size_t i = 0; i < nq + np; ++i)
    {
        std::uint64_t value = 0;
        require(bool(input >> value), "truncated fixture primes");
        (i < nq ? q : p).emplace_back(value);
    }
    return PoseidonContext(ParametersLiteral(CKKS, log_n, log_n - 1, 40, 192, 1,
                                            Modulus(0), q, p, sec_level_type::none));
}

void sampler_test(const PoseidonContext &context)
{
    const auto id = context.crt_context()->key_parms_id();
    const auto level = context.crt_context()->key_context_data();
    const std::size_t n = level->parms().degree();
    const auto &moduli = level->coeff_modulus();
    std::vector<std::uint64_t> coefficients(n * moduli.size(), 42);
    const auto verify = [&](std::size_t weight) {
        std::size_t nonzero = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto first = coefficients[i];
            require(first == 0 || first == 1 || first == moduli.front().value() - 1,
                    "sampler produced a non-ternary coefficient");
            nonzero += first != 0;
            for (std::size_t limb = 1; limb < moduli.size(); ++limb)
                require(coefficients[limb * n + i] ==
                            (first == moduli.front().value() - 1 ? moduli[limb].value() - 1 : first),
                        "sampler signs differ between RNS limbs");
        }
        require(nonzero == weight, "sampler did not produce the exact Hamming weight");
    };
    for (const auto weight : {std::size_t{0}, std::size_t{32}, n})
    {
        std::fill(coefficients.begin(), coefficients.end(), 42);
        util::sample_poly_ternary_with_hamming(context.random_generator()->create(), context,
                                               id, weight, coefficients.data());
        verify(weight);
    }
    util::sample_poly_ternary_with_hamming(context.random_generator()->create(), context,
                                           id, coefficients.data());
    verify(192); // Existing callers keep the context-selected distribution.
    bool rejected = false;
    try
    {
        util::sample_poly_ternary_with_hamming(context.random_generator()->create(), context,
                                               id, n + 1, coefficients.data());
    }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "out-of-range sparse weight was accepted");
    std::cout << "PASS explicit/default sparse sampler, zero initialization and RNS signs\n";
}

std::complex<double> chebyshev(const Polynomial &polynomial, double x, double fold)
{
    std::complex<double> previous = 0, current = 0;
    const auto &coefficients = polynomial.data();
    for (std::size_t i = coefficients.size() - 1; i > 0; --i)
    {
        const auto next = coefficients[i] * fold + 2.0 * x * current - previous;
        previous = current;
        current = next;
    }
    return coefficients.front() * fold + x * current - previous;
}

void plan_test(const PoseidonContext &context, const GpuResNet20ManualOptions &options)
{
    CKKSEncoder encoder(context);
    constexpr double working_scale = double(std::uint64_t{1} << 45);
    EvalModPoly polynomial(context, CosDiscrete, working_scale, 45, 5,
                           options.double_angle, options.k, 0, options.sine_degree);
    if (options.full_coefficients) polynomial.enable_full_cosine_coefficients();
    // ApproximateCos may omit the final zero odd term of the legacy degree59 fit.
    require(polynomial.sine_poly().degree() <= options.sine_degree &&
                polynomial.sine_poly().degree() + 1 >= options.sine_degree,
            "polynomial degree changed");
    const double c2s_scaling = polynomial.q_div() /
        (polynomial.k() * polynomial.sc_fac() * polynomial.q_diff());
    require(std::abs(options.k * context.crt_context()->q0() * c2s_scaling /
                         working_scale - 1) < 1e-12, "integer-period contract failed");
    const auto parameters = context.parameters_literal();
    HomomorphicDFTMatrixLiteral literal(encode, parameters->log_n(), parameters->log_slots(),
        49, {1, 1, 1}, true, c2s_scaling, false, 1);
    LinearMatrixGroup c2s;
    literal.create_dynamic(c2s, encoder, working_scale, working_scale, working_scale, 1.0);
    double c2s_scale = working_scale;
    std::size_t q_count = 50;
    const auto &q = parameters->q();
    for (std::size_t i = 0; i < c2s.data().size(); ++i)
    {
        c2s_scale *= c2s.data()[i].scale;
        for (unsigned j = 0; j < c2s.rescale_counts()[i]; ++j)
            c2s_scale /= double(q[--q_count].value());
    }
    polynomial.set_level_start(static_cast<std::uint32_t>(q_count - 1));
    GpuEvalModUploadOptions upload;
    upload.dynamic_rescale = true;
    upload.polynomial_log_split = options.log_split;
    upload.flat_bsgs_b8 = false;
    upload.virtual_degree_bound = false;
    upload.lead_leaf_resplit = false;
    const auto id = context.crt_context()->parms_id_map().at(q_count - 1);
    const auto native = GpuUploader::upload_eval_mod_high_precision(
        polynomial, encoder, id, 0, nullptr, parms_id_zero, 1, nullptr, true,
        std::numeric_limits<std::uint32_t>::max(), NAN, NAN, true, c2s_scale, true, upload);
    require(native.output_q_count >= 31, "EvalMod cannot retain the manual Q31 contract");
    require(native.polynomial_degree == polynomial.sine_poly().degree(),
            "uploader changed polynomial degree");
    const double alpha = std::ldexp(1.0, 40) / native.output_scale;
    const double seed = std::pow(alpha, std::ldexp(1.0, -int(options.double_angle)));
    require(alpha > 0 && alpha < 1, "output fold is invalid");
    Polynomial coefficients = polynomial.sine_poly();
    for (auto &coefficient : coefficients.data()) coefficient *= seed;
    const auto folded_plan = GpuUploader::upload_eval_mod_high_precision(
        polynomial, encoder, id, 0, nullptr, parms_id_zero, 1, &coefficients, true,
        std::numeric_limits<std::uint32_t>::max(), polynomial.sqrt_2pi() * seed,
        NAN, true, c2s_scale, true, upload);
    require(folded_plan.output_q_count == native.output_q_count &&
                folded_plan.output_scale == native.output_scale &&
                folded_plan.required_relin_q_counts == native.required_relin_q_counts,
            "coefficient fold changed the EvalMod Q/scale/key-view schedule");
    if (options.full_coefficients)
        require(std::any_of(native.basis_steps.begin(), native.basis_steps.end(),
                            [](const auto &step) { return step.output_degree == 3; }),
                "full-coefficient low plan omitted its odd T3 basis");
    // Pure arithmetic oracle for the DA-dependent fold; no ciphertext shortcut.
    for (double x : {-0.9, -0.3, 0.0, 0.4, 0.95})
    {
        auto plain = chebyshev(polynomial.sine_poly(), x, 1);
        auto folded = chebyshev(polynomial.sine_poly(), x, seed);
        double base = polynomial.sqrt_2pi(), folded_base = base * seed;
        for (unsigned i = 0; i < options.double_angle; ++i)
        {
            base *= base;
            folded_base *= folded_base;
            plain = 2.0 * plain * plain - base;
            folded = 2.0 * folded * folded - folded_base;
        }
        require(std::abs(folded - alpha * plain) < 1e-10, "DA-dependent output fold failed");
    }
    std::cout << "PASS degree=" << options.sine_degree << " DA=" << options.double_angle
              << " K=" << options.k << " native_output_q=" << native.output_q_count
              << " output_log_scale=" << std::log2(native.output_scale) << '\n';
}

void gpu_test(const PoseidonContext &context, const GpuResNet20ManualOptions &options)
{
    KeyGenerator generator(context);
    PublicKey public_key;
    generator.create_public_key(public_key);
    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, generator.secret_key());
    auto profile = GpuBootstrapProfileBuilder::build_resnet20_manual(context, generator, 0, options);
    require(profile.input_level_min == 5 && profile.input_level_max == 5 &&
                profile.output_level == 30 && profile.output_scale_log2 == 40,
            "manual Boot metadata changed");
    require(bool(profile.bootstrap_data.sparse_switch_to_ephemeral) ==
                (options.sparse_secret_hamming_weight != 0), "sparse keys missing/unexpected");
    std::vector<std::complex<double>> values(encoder.slot_count());
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = 0.1 * std::sin(0.013 * i);
    Plaintext plain;
    encoder.encode(values, context.crt_context()->parms_id_map().at(5), std::ldexp(1.0, 40), plain);
    Ciphertext source;
    encryptor.encrypt(plain, source);
    auto input = GpuUploader::upload_ciphertext(source, 0);
    GpuParameterData parameters(context, 0);
    GpuEvaluator evaluator(parameters);
    GpuBootstrapWorkspace workspace;
    GpuCiphertextData output;
    evaluator.bootstrap(input, profile.bootstrap_data, *profile.relin_keys,
                        *profile.galois_keys, workspace, output);
    require(output.meta.q_count == 31 && output.meta.scale == std::ldexp(1.0, 40),
            "native Boot changed the declared output Q/scale");
    Ciphertext host;
    GpuUploader::download_ciphertext(output, host, context);
    decryptor.decrypt(host, plain);
    std::vector<std::complex<double>> actual;
    encoder.decode(plain, actual);
    double worst = 0;
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        const double error = std::abs(actual[i] - values[i]);
        require(std::isfinite(error), "native Boot produced a non-finite result");
        worst = std::max(worst, error);
    }
    std::cout << "NATIVE_BOOT degree=" << options.sine_degree << " max_error=" << worst << '\n';
    require(worst < 2e-3, "native Boot precision regression");
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        require(argc >= 2 && argc <= 3, "usage: manual_low_boot_test FIXTURE [--host|--gpu-low|--gpu-high]");
        const std::string mode = argc == 3 ? argv[2] : "--host";
        require(mode == "--host" || mode == "--gpu-low" || mode == "--gpu-high", "unknown mode");
        const GpuResNet20ManualOptions low{22, 3, 12, 2, true, 32};
        if (mode == "--host")
        {
            // Same Q/P chain and scale plan, smaller ring to keep the CPU check light.
            auto context = make_context(argv[1], 12);
            sampler_test(context);
            plan_test(context, {});
            plan_test(context, low);
        }
        else
        {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            {
                std::cout << "SKIP: CUDA device unavailable\n";
                return 77;
            }
            auto context = make_context(argv[1], 16);
            gpu_test(context, mode == "--gpu-low" ? low : GpuResNet20ManualOptions{});
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "manual low Boot test failed: " << error.what() << '\n';
        return 1;
    }
}
