#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/keygenerator.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// CPU primitive baseline, not a DSL/compiler or bootstrap test.
int main()
{
    using namespace poseidon;
    try {
        ParametersLiteralDefault params(CKKS, 16384, sec_level_type::tc128);
        auto factory = PoseidonFactory::get_instance();
        factory->set_device_type(DEVICE_SOFTWARE);
        auto context = factory->create_poseidon_context(params);
        auto evaluator = factory->create_ckks_evaluator(context);
        KeyGenerator keys(context);
        PublicKey public_key;
        keys.create_public_key(public_key);
        CKKSEncoder encoder(context);
        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, keys.secret_key());
        const double scale = std::pow(2.0, 48);
        const double tolerance = 1e-6;
        std::vector<std::complex<double>> x(params.slot()), w(params.slot());
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = (static_cast<int>(i % 17) - 8) / 16.0;
            w[i] = (static_cast<int>(i % 7) + 1) / 8.0;
        }
        Plaintext px, pw, decoded;
        encoder.encode(x, scale, px);
        encoder.encode(w, scale, pw);
        Ciphertext cx, result;
        encryptor.encrypt(px, cx);
        std::cout << std::setprecision(17)
                  << "backend=software N=16384 security_request=tc128 scale=" << scale
                  << " slots=" << x.size() << " max_abs_tolerance=" << tolerance << '\n';
        std::cout << "Q=";
        for (const auto &prime : params.q()) std::cout << prime.value() << ',';
        std::cout << "\nP=";
        for (const auto &prime : params.p()) std::cout << prime.value() << ',';
        std::cout << '\n';
        auto check = [&](const char *name, const Ciphertext &cipher,
                         const std::vector<std::complex<double>> &expected) {
            decryptor.decrypt(cipher, decoded);
            std::vector<std::complex<double>> actual;
            encoder.decode(decoded, actual);
            if (actual.size() != expected.size()) throw std::runtime_error("slot count mismatch");
            double total = 0, maximum = 0, relative = 0, relative_nonzero = 0;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const double error = std::abs(actual[i] - expected[i]);
                if (!std::isfinite(error)) throw std::runtime_error("non-finite output");
                total += error;
                maximum = std::max(maximum, error);
                relative = std::max(relative, error / std::max(1e-12, std::abs(expected[i])));
                if (std::abs(expected[i]) > 1e-12)
                    relative_nonzero = std::max(relative_nonzero, error / std::abs(expected[i]));
                if (i < 8) std::cout << name << " slot=" << i << " expected=" << expected[i]
                                      << " actual=" << actual[i] << '\n';
            }
            std::cout << name << " MAE=" << total / actual.size() << " max_abs=" << maximum
                      << " max_relative_floor_1e-12=" << relative
                      << " max_relative_nonzero=" << relative_nonzero << '\n';
            if (maximum > tolerance) throw std::runtime_error("absolute error threshold exceeded");
        };
        check("roundtrip", cx, x);
        evaluator->add(cx, cx, result);
        auto expected = x;
        for (auto &value : expected) value *= 2;
        check("add", result, expected);
        evaluator->multiply_plain(cx, pw, result);
        evaluator->rescale(result, result);
        for (std::size_t i = 0; i < x.size(); ++i) expected[i] = x[i] * w[i];
        check("multiply_plain_rescale", result, expected);
        std::cout << "PASS\n";
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
