// Check the BV/P1 key layout used by both Trident functional profiles.
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace poseidon;
using namespace poseidon::gpu;

int main(int argc, char **argv)
try
{
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--low-word-control"))
        throw std::invalid_argument("usage: functional_keyswitch_probe 15|31 [--low-word-control]");
    const std::string degree = argv[1];
    if (degree != "15" && degree != "31")
        throw std::invalid_argument("harmonic degree must be 15 or 31");

    const bool low_word_control = argc == 3;
    const std::uint32_t log_scale = low_word_control ? 27 : 40;
    const std::uint32_t log_p = low_word_control ? 30 : 51;
    std::vector<std::uint32_t> q_bits{45, 40, 40, 40, 40};
    q_bits.resize(degree == "15" ? 19 : 20, 45);
    if (low_word_control)
        std::fill(q_bits.begin(), q_bits.end(), 29);
    ParametersLiteral literal(CKKS, 15, 14, log_scale, 32, 0, Modulus(0),
                              {}, {}, sec_level_type::none);
    literal.set_log_modulus(q_bits, {log_p});
    auto factory = PoseidonFactory::get_instance();
    factory->set_device_type(DEVICE_SOFTWARE);
    auto context = factory->create_poseidon_context(literal);
    if (context.key_switch_variant() != BV)
        throw std::runtime_error("expected BV key switching for one special prime");
    auto cpu = factory->create_ckks_evaluator(context);
    CKKSEncoder encoder(context);
    KeyGenerator keys(context);
    PublicKey pk;
    GaloisKeys rotations;
    RelinKeys relin;
    keys.create_public_key(pk);
    const std::vector<int> rotation_steps{1, -3};
    keys.create_galois_keys(std::vector<int>{1, -3, 0}, rotations);
    keys.create_relin_keys(relin);
    Encryptor encryptor(context, pk);
    Decryptor decryptor(context, keys.secret_key());

    std::vector<std::complex<double>> values(encoder.slot_count());
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = {std::sin(i * 0.031) / 2, std::cos(i * 0.017) / 4};
    Plaintext pt;
    encoder.encode(values, std::exp2(log_scale), pt);
    Ciphertext input;
    encryptor.encrypt(pt, input);

    std::cout << std::unitbuf << "D" << degree << " N=32768 Q=" << q_bits.size()
              << " P=1 p_bits=" << log_p << " variant=BV"
              << " low_word_control=" << low_word_control << '\n';
    GpuParameterData params(context, 0);
    GpuEvaluator gpu(params);
    auto gpu_rotations = GpuUploader::upload_galois_keys(rotations, 0);
    auto hoisted_rotations = GpuUploader::upload_double_hoist_galois_keys(rotations, 0);
    GpuDoubleHoistWorkspace hoist_workspace;
    auto gpu_relin = GpuUploader::upload_relin_keys(relin, 0);

    auto decode = [&](const Ciphertext &ciphertext) {
        Plaintext plain;
        std::vector<std::complex<double>> slots;
        decryptor.decrypt(ciphertext, plain);
        encoder.decode(plain, slots);
        return slots;
    };
    auto check = [&](const char *name, const Ciphertext &ref, const GpuCiphertextData &result) {
        Ciphertext actual;
        GpuUploader::download_ciphertext(result, actual, context);
        if (actual.parms_id() != ref.parms_id() || actual.size() != 2 ||
            !std::isfinite(actual.scale()) || std::abs(actual.scale() / ref.scale() - 1) > 1e-8)
            throw std::runtime_error(std::string(name) + ": level, size or scale mismatch");
        const auto expected = decode(ref);
        const auto got = decode(actual);
        double error = 0;
        for (std::size_t i = 0; i < got.size(); ++i)
        {
            const double diff = std::abs(got[i] - expected[i]);
            if (!std::isfinite(diff))
                throw std::runtime_error(std::string(name) + ": nonfinite result");
            error = std::max(error, diff);
        }
        std::cout << name << " level=" << actual.level() << " max_error=" << error << '\n';
        if (error > 1e-6)
            throw std::runtime_error(std::string(name) + ": GPU vs CPU exceeded 1e-6");
    };

    for (const auto level : {input.level(), decltype(input.level()){2}})
    {
        Ciphertext at_level;
        cpu->drop_modulus(input, at_level, level);
        const auto device_input = GpuUploader::upload_ciphertext(at_level, 0);
        Ciphertext reference;
        GpuCiphertextData device_output;
        cpu->rotate(at_level, reference, 1, rotations);
        gpu.rotate(device_input, 1, gpu_rotations, device_output);
        check("rotate", reference, device_output);
        std::vector<GpuCiphertextData> hoisted_output;
        try
        {
            gpu.rotate_many_hoisted(device_input, rotation_steps, hoisted_rotations,
                                    hoist_workspace, hoisted_output);
            throw std::runtime_error("BV hoisting unexpectedly accepted");
        }
        catch (const std::invalid_argument &error)
        {
            if (std::string(error.what()).find("BV hoisting is unsupported") == std::string::npos)
                throw;
        }
        cpu->conjugate(at_level, rotations, reference);
        gpu.conjugate(device_input, gpu_rotations, device_output);
        check("conjugate", reference, device_output);
        cpu->multiply_relin(at_level, at_level, reference, relin);
        GpuCiphertextData squared;
        gpu.square(device_input, squared);
        gpu.relinearize(squared, gpu_relin, device_output);
        check("square_relin", reference, device_output);
    }
    std::cout << "PASS\n";
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}

