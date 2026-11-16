// CPU-only genuine encrypt -> MulCP -> real Rescale(n) -> AddCP check of the
// exact scale contract used by compact_linear.cpp. No GPU dependency.
#include "compact_layout.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/decryptor.h"
#include "poseidon/keygenerator.h"
#include <iostream>

int main() {
    try {
        using namespace poseidon;
        using namespace bert::compact;
        constexpr std::size_t n = 1024, nq = 6, np = 2;
        std::vector<int> bits(nq + np, 28);
        bits[0] = 31;
        auto primes = CoeffModulus::Create(n, bits);
        std::vector<Modulus> q(primes.begin(), primes.begin()+nq), p(primes.begin()+nq, primes.end());
        ParametersLiteral literal(CKKS, 10, 9, 28, 0, 1, Modulus(0), q, p, sec_level_type::none);
        PoseidonContext context(literal);
        auto evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);
        CKKSEncoder encoder(context);
        KeyGenerator keys(context);
        PublicKey public_key;
        keys.create_public_key(public_key);
        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, keys.secret_key());
        double max_error = 0, max_scale_error = 0;
        for (unsigned count : {1U, 2U}) {
            const double input_log2 = count == 1 ? 30.125 : 56.125;
            const double output_log2 = count == 1 ? 30.25 : 55.875;
            double plain_log2 = output_log2 - input_log2;
            for (std::size_t i = nq-count; i < nq; ++i) plain_log2 += std::log2(double(q[i].value()));
            std::vector<double> x(n/2), w(n/2), b(n/2);
            for (std::size_t i = 0; i < n/2; ++i) {
                x[i] = 0.1 + 0.0001 * i;
                w[i] = 0.02 * std::sin(double(i));
                b[i] = 0.003 * std::cos(double(i));
            }
            Plaintext input, weight;
            encoder.encode(x, std::exp2(input_log2), input);
            encoder.encode(w, std::exp2(plain_log2), weight);
            Ciphertext ct, product, result;
            encryptor.encrypt(input, ct);
            evaluator->multiply_plain(ct, weight, product);
            const double actual_product = ct.scale() * weight.scale();
            if (product.scale() != actual_product) throw std::runtime_error("CPU MulCP product scale is not physical");
            double expected_scale = actual_product;
            result = std::move(product);
            for (std::size_t drop = 0; drop < count; ++drop) {
                evaluator->rescale(result, result);
                expected_scale /= double(q[nq-1-drop].value());
            }
            if (result.scale() != expected_scale) throw std::runtime_error("CPU rescale scale was not divided by actual dropped Q");
            max_scale_error = std::max(max_scale_error, std::abs(std::log2(result.scale()) - output_log2));
            Plaintext bias;
            encoder.encode(b, result.parms_id(), result.scale(), bias);
            evaluator->add_plain(result, bias, result);
            Plaintext decrypted;
            decryptor.decrypt(result, decrypted);
            std::vector<double> decoded;
            encoder.decode(decrypted, decoded);
            for (std::size_t i = 0; i < n/2; ++i)
                max_error = std::max(max_error, std::abs(decoded[i] - (x[i]*w[i]+b[i])));
        }
        if (max_scale_error > 1e-12 || max_error > 1e-5) throw std::runtime_error("physical scale/dense coefficient check failed");
        std::cout.precision(16);
        std::cout << "{\"cpu_only\":true,\"real_mulcp_rescale_bias\":true,\"rescale_counts\":[1,2],\"max_scale_log2_error\":"
            << max_scale_error << ",\"decode_max_abs\":" << max_error << "}\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
