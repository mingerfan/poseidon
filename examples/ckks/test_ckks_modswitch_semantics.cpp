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

// Actual Poseidon software primitive oracle. Not a HEVM backend or GPU test.
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
        // Explicit primitive-test scale, matching the default CKKS log-scale.
        const double scale = std::ldexp(1.0, 40);
        std::vector<std::complex<double>> input(params.slot());
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = (static_cast<int>(i % 17) - 8) / 32.0;
        Plaintext encoded;
        encoder.encode(input, scale, encoded);
        Ciphertext source;
        encryptor.encrypt(encoded, source);
        const auto q_count = source.coeff_modulus_size();
        const auto degree = source.poly_modulus_degree();
        if (q_count < 3 || !source.is_ntt_form())
            throw std::runtime_error("Unexpected default chain or representation");
        auto equal_prefix = [&](const Ciphertext &a, const Ciphertext &b) {
            if (a.size() != b.size() || a.coeff_modulus_size() > b.coeff_modulus_size())
                return false;
            for (std::size_t c = 0; c < a.size(); ++c)
                if (!std::equal(a.data(c), a.data(c) + a.coeff_modulus_size()*degree, b.data(c)))
                    return false;
            return true;
        };
        const Ciphertext original = source;
        const auto &mapping = context.crt_context()->parms_id_map();
        std::cout << std::setprecision(17)
                  << "{\"backend\":\"poseidon_software_primitive\",\"gpu_executed\":false,"
                  << "\"security_request\":\"tc128\",\"degree\":16384,\"slots\":" << input.size()
                  << ",\"scale\":" << scale << ",\"source_q_count\":" << q_count
                  << ",\"source_poseidon_level\":" << source.level() << ",\"q_moduli\":[";
        for (std::size_t i = 0; i < params.q().size(); ++i)
            std::cout << (i ? "," : "") << params.q()[i].value();
        std::cout << "],\"p_moduli\":[";
        for (std::size_t i = 0; i < params.p().size(); ++i)
            std::cout << (i ? "," : "") << params.p()[i].value();
        std::cout << "],\"cases\":[";
        Ciphertext sequential = source;
        for (std::size_t target = q_count; target >= 1; --target) {
            const auto target_id = mapping.at(static_cast<std::uint32_t>(target-1));
            const auto data = context.crt_context()->get_context_data(target_id);
            if (!data || !data->parms().p().empty() || data->parms().q().size() != target)
                throw std::runtime_error("Target is not exact Q-only prefix");
            Ciphertext direct, inplace = source;
            evaluator->drop_modulus(source, direct, target_id);
            evaluator->drop_modulus(inplace, inplace, target_id);
            evaluator->drop_modulus(sequential, sequential, target_id);
            if (direct.parms_id() != target_id || inplace.parms_id() != target_id ||
                sequential.parms_id() != target_id || source.parms_id() != original.parms_id() ||
                source.scale() != original.scale() || source.is_ntt_form() != original.is_ntt_form() ||
                direct.coeff_modulus_size() != target ||
                direct.scale() != source.scale() || !direct.is_ntt_form() ||
                inplace.scale() != source.scale() || sequential.scale() != source.scale() ||
                !equal_prefix(direct, source) || !equal_prefix(direct, inplace) ||
                !equal_prefix(inplace, direct) || !equal_prefix(direct, sequential) ||
                !equal_prefix(sequential, direct) || !equal_prefix(source, original))
                throw std::runtime_error("Drop did not preserve exact prefix/metadata/alias semantics");
            Plaintext plaintext;
            decryptor.decrypt(direct, plaintext);
            std::vector<std::complex<double>> actual;
            encoder.decode(plaintext, actual);
            if (actual.size() != input.size())
                throw std::runtime_error("Slot count changed");
            double total = 0.0, maximum = 0.0;
            for (std::size_t i = 0; i < input.size(); ++i) {
                const double error = std::abs(actual[i]-input[i]);
                if (!std::isfinite(error) || error > 1e-5 + 1e-4*std::abs(input[i]))
                    throw std::runtime_error("Plaintext changed beyond fixed threshold");
                maximum = std::max(maximum, error);
                total += error;
            }
            std::cout << (target == q_count ? "" : ",")
                      << "{\"drop_count\":" << q_count-target
                      << ",\"target_q_count\":" << target << ",\"poseidon_level\":" << target-1
                      << ",\"prefix_exact\":true,\"alias_matches\":true,\"sequential_matches\":true,"
                      << "\"scale_unchanged\":true,\"ntt_unchanged\":true,\"mae\":"
                      << total/input.size() << ",\"max_absolute_error\":" << maximum << "}";
        }
        Ciphertext rescaled;
        evaluator->rescale(source, rescaled);
        if (rescaled.coeff_modulus_size() != q_count-1 || rescaled.scale() == source.scale())
            throw std::runtime_error("Rescale counterexample did not distinguish scale semantics");
        bool empty_rejected = false;
        try {
            Ciphertext empty, result;
            evaluator->drop_modulus(empty, result, source.parms_id());
        } catch (const std::exception &) {
            empty_rejected = true;
        }
        if (!empty_rejected)
            throw std::runtime_error("Empty input unexpectedly accepted");
        std::cout << "],\"rescale_changes_scale\":true,\"rescaled_scale\":" << rescaled.scale()
                  << ",\"empty_rejected\":true,\"atol\":1e-5,\"rtol\":1e-4,\"status\":\"passed\"}\n";
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
