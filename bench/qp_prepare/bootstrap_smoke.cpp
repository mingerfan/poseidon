// Native GPU Boot numerical smoke test. This is an insecure, small test profile,
// not a production parameter recommendation. Run each QP mode in a new process.
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_bootstrap_profile.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace poseidon;
using namespace poseidon::gpu;

int main() {
    std::cout << std::unitbuf << std::setprecision(12);
    auto start = std::chrono::steady_clock::now();
    auto stage = [&](const char *name) {
        std::cout << "stage=" << name << " elapsed_s="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << '\n';
    };
    try {
        const char *mode = std::getenv("POSEIDON_GPU_QP_PREPARE");
        if (!mode || (std::string(mode) != "reference" && std::string(mode) != "optimized"))
            throw std::runtime_error("set POSEIDON_GPU_QP_PREPARE=reference or optimized");
        std::cout << "mode=" << mode << " N=4096 log_slots=11 Q=50x30 P=8x30"
                  << " q0_level=1 log_input_scale=40 secret_hamming_weight=5"
                  << " evalmod_scale=45 sine_degree=58 generation_degree=59 double_angle=2"
                  << " tolerance=0.001 security=none\n";
        ParametersLiteral literal(CKKS, 12, 11, 40, 5, 1, Modulus(0), {}, {}, sec_level_type::none);
        literal.set_log_modulus(std::vector<std::uint32_t>(50, 30), std::vector<std::uint32_t>(8, 30));
        PoseidonContext context(literal);
        KeyGenerator generator(context);
        if (generator.secret_key().data().is_zero()) throw std::runtime_error("zero secret key");
        PublicKey public_key;
        generator.create_public_key(public_key);
        CKKSEncoder encoder(context);
        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, generator.secret_key());
        stage("profile_begin");
        GpuBootstrapProfileConfig config;
        config.profile_id = "smoke-degree58-da2-scale45";
        config.eval_mod_log_scale = 45;
        config.eval_mod_sine_degree = 58;
        config.eval_mod_generation_degree = 59;
        config.eval_mod_truncate_degree.reset();
        config.eval_mod_double_angle = 2;
        auto profile = GpuBootstrapProfileBuilder::build(context, generator, 0, config);
        gpu_check_cuda(cudaDeviceSynchronize(), "profile synchronize");
        stage("profile_complete");
        if (!profile.relin_keys || !profile.galois_keys) throw std::runtime_error("profile missing keys");
        std::size_t matrix_count = 0, diagonal_count = 0;
        for (const auto *group : {&profile.bootstrap_data.coeff_to_slot_matrix_qp,
                                  &profile.bootstrap_data.slot_to_coeff_matrix_qp}) {
            for (const auto &matrix : group->data()) {
                ++matrix_count;
                diagonal_count += matrix.plain_vec_qp.size();
            }
        }
        std::cout << "qp_matrices=" << matrix_count << " qp_diagonals=" << diagonal_count << '\n';
        if (!matrix_count || !diagonal_count) throw std::runtime_error("no prepared QP diagonals");
        std::vector<std::complex<double>> message(encoder.slot_count());
        for (std::size_t i = 0; i < message.size(); ++i)
            message[i] = {double(i % 17 + 1) / 32., double(i % 11 + 1) / 64.};
        Plaintext plain;
        encoder.encode(message, context.crt_context()->parms_id_map().at(profile.input_level_max),
                       std::ldexp(1., 40), plain);
        Ciphertext host_input;
        encryptor.encrypt(plain, host_input);
        GpuParameterData parameters(context, 0);
        for (auto &level : profile.level_parameters) parameters.adopt_level(std::move(level));
        GpuEvaluator evaluator(parameters);
        GpuBootstrapWorkspace workspace;
        auto input = GpuUploader::upload_ciphertext(host_input, 0);
        GpuCiphertextData output;
        gpu_check_cuda(cudaDeviceSynchronize(), "input synchronize");
        std::cout << "input_q=" << host_input.coeff_modulus_size() << " input_scale=" << host_input.scale()
                  << " input_log2_scale=" << std::log2(host_input.scale()) << '\n';
        stage("bootstrap_begin");
        evaluator.bootstrap(input, profile.bootstrap_data, *profile.relin_keys, *profile.galois_keys,
                            workspace, output);
        gpu_check_cuda(cudaDeviceSynchronize(), "bootstrap synchronize");
        stage("bootstrap_complete");
        Ciphertext host_output;
        GpuUploader::download_ciphertext(output, host_output, context);
        gpu_check_cuda(cudaDeviceSynchronize(), "download synchronize");
        Plaintext decrypted;
        decryptor.decrypt(host_output, decrypted);
        std::vector<std::complex<double>> decoded;
        encoder.decode(decrypted, decoded);
        if (decoded.size() < message.size()) throw std::runtime_error("short decode");
        double max_error = 0, sum_error = 0, sum_sq = 0;
        for (std::size_t i = 0; i < message.size(); ++i) {
            const double e = std::abs(decoded[i] - message[i]);
            if (!std::isfinite(e)) throw std::runtime_error("nonfinite error");
            max_error = std::max(max_error, e); sum_error += e; sum_sq += e * e;
        }
        std::cout << "output_q=" << host_output.coeff_modulus_size() << " output_scale=" << host_output.scale()
                  << " output_log2_scale=" << std::log2(host_output.scale()) << " slots=" << message.size()
                  << " max_abs_error=" << max_error << " mean_abs_error=" << sum_error/message.size()
                  << " rms_error=" << std::sqrt(sum_sq/message.size()) << '\n';
        for (std::size_t i = 0; i < 4; ++i)
            std::cout << "sample=" << i << " expected=" << message[i] << " actual=" << decoded[i] << '\n';
        if (const char *path = std::getenv("POSEIDON_BOOTSTRAP_SMOKE_OUTPUT")) {
            std::ofstream file(path);
            if (!file) throw std::runtime_error("cannot open decoded output file");
            file << std::setprecision(17);
            for (std::size_t i = 0; i < message.size(); ++i)
                file << decoded[i].real() << ' ' << decoded[i].imag() << '\n';
            file.close();
            if (!file) throw std::runtime_error("cannot write decoded output file");
        }
        stage("decrypt_complete");
        if (max_error > 1e-3) throw std::runtime_error("bootstrap error exceeds tolerance");
        std::cout << "PASS native_bootstrap_decrypt\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
