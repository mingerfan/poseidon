#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_wide_bv.h"
#include "poseidon/factory/poseidon_factory.h"
#include "functional_wide_probe_keys.h"
#include "poseidon/keygenerator.h"
#include "poseidon/decryptor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using namespace poseidon;
using namespace poseidon::gpu;

namespace
{
using Word = std::uint64_t;

void check_fused_qp(const ParametersLiteral &literal)
{
    auto moduli = literal.q();
    moduli.insert(moduli.end(), literal.p().begin(), literal.p().end());
    const std::size_t degree = literal.degree(), words = degree * moduli.size();
    std::vector<poseidon::gpu::detail::WideBvLimb> tables;
    for (const auto &q : moduli)
    {
        const auto ratio = q.const_ratio();
        tables.push_back({q.value(), ratio[0], ratio[1], 0, 0});
    }
    std::vector<Word> input(words), key(2 * words), initial(2 * words), expected(2 * words), transformed(words), actual(2 * words);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    auto next = [&]() { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; };
    for (std::size_t i = 0; i < words; ++i)
    {
        const auto q = moduli[i / degree].value();
        input[i] = next() % q;
        for (int c = 0; c < 2; ++c)
        {
            key[c * words + i] = next() % q;
            initial[c * words + i] = next() % q;
        }
        if (i % degree == 0) { input[i] = q - 1; key[i] = q - 1; initial[i] = q - 1; }
    }
    GpuWideNtt ntt(literal.log_n(), moduli, 0);
    DeviceVector<Word> values(words, 0), keys(2 * words, 0), sums(2 * words, 0);
    DeviceVector<poseidon::gpu::detail::WideBvLimb> limbs(tables.size(), 0);
    values.copy_from_host(input.data(), words);
    ntt.forward(values);
    values.copy_to_host(transformed.data(), words);
    expected = initial;
    for (std::size_t i = 0; i < words; ++i)
    {
        const auto q = moduli[i / degree].value();
        for (int c = 0; c < 2; ++c)
        {
            const auto at = c * words + i;
            const Word product = static_cast<unsigned __int128>(transformed[i]) * key[at] % q;
            const Word sum = expected[at] + product;
            expected[at] = sum >= q ? sum - q : sum;
        }
    }
    values.copy_from_host(input.data(), words);
    keys.copy_from_host(key.data(), key.size());
    sums.copy_from_host(initial.data(), initial.size());
    limbs.copy_from_host(tables.data(), tables.size());
    ntt.forward_accumulate(values, keys.data(), sums, limbs);
    sums.copy_to_host(actual.data(), actual.size());
    if (actual != expected) throw std::runtime_error("fused NTT keyacc Q/P residue mismatch");
    std::cout << "fused NTT keyacc Q/P limbs=" << moduli.size() << " residues=" << actual.size() << " match=EXACT\n";
}

class BvProbe
{
public:
    BvProbe(const ParametersLiteral &literal, std::vector<Modulus> q)
        : q_(std::move(q)), degree_(literal.degree()), words_(degree_ * q_.size()),
          bv_(literal.log_n(), q_, literal.p().front(), 0),
          input_(2 * words_, 0), output_(2 * words_, 0)
    {}

    void check(const char *operation, const Ciphertext &input, std::uint32_t galois_element,
               const std::vector<PublicKey> &digits, const Ciphertext &reference,
               CKKSEncoder &encoder, Decryptor &decryptor)
    {
        const std::size_t qp_words = words_ + degree_;
        std::vector<Word> packed_key(2 * qp_words * q_.size());
        DeviceVector<Word> key(packed_key.size(), 0);
        probe::upload_bv_key(digits, degree_, q_.size(), packed_key, key);
        input_.copy_from_host(input.data(0), input_.size());
        if (galois_element != 0)
            bv_.apply_galois(input_, key, galois_element, output_);
        else
            bv_.square_relinearize(input_, key, output_);

        std::vector<Word> actual(output_.size());
        output_.copy_to_host(actual.data(), actual.size());
        ::unsetenv("POSEIDON_WIDE_NTT_KEYACC");
        if (galois_element) bv_.apply_galois(input_, key, galois_element, output_);
        else bv_.square_relinearize(input_, key, output_);
        std::vector<Word> unfused(output_.size());
        output_.copy_to_host(unfused.data(), unfused.size());
        ::setenv("POSEIDON_WIDE_NTT_KEYACC", "1", 1);
        if (galois_element) bv_.apply_galois(input_, key, galois_element, output_);
        else bv_.square_relinearize(input_, key, output_);
        std::vector<Word> fused(output_.size());
        output_.copy_to_host(fused.data(), fused.size());
        if (unfused != fused) throw std::runtime_error("NTT key accumulation residue mismatch");
        std::cout << operation << " fused keyacc Q=" << q_.size() << " residues=" << fused.size() << " match=EXACT\n";
        for (std::size_t component = 0; component < 2; ++component)
        {
            for (std::size_t i = 0; i < words_; ++i)
            {
                const Word value = actual[component * words_ + i];
                if (value != reference.data(component)[i])
                    throw std::runtime_error(std::string(operation) + " mismatch at component " +
                        std::to_string(component) + " word " + std::to_string(i));
            }
        }
        if (galois_element != 0)
        {
            if (!prepared_)
            {
                bv_.prepare_galois(input_);
                prepared_ = true;
            }
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                bv_.apply_galois_prepared(input_, key, galois_element, output_);
                std::vector<Word> hoisted(output_.size());
                output_.copy_to_host(hoisted.data(), hoisted.size());
                if (hoisted != actual)
                    throw std::runtime_error(std::string(operation) + " hoisted mismatch");
            }
            std::cout << operation << " hoisted Q=" << q_.size() << " match=EXACT\n";
            if (!digits_prepared_)
            {
                bv_.prepare_galois_digits(input_);
                digits_prepared_ = true;
                DeviceVector<Word> wrong_input(input_.size(), 0);
                bool rejected = false;
                try { bv_.apply_galois_digits(wrong_input, key, galois_element, output_); }
                catch (const std::invalid_argument &) { rejected = true; }
                if (!rejected) throw std::runtime_error("shared digits accepted different input");
            }
            bv_.apply_galois_digits(input_, key, galois_element, output_);
            std::vector<Word> shared(output_.size());
            output_.copy_to_host(shared.data(), shared.size());
            DeviceVector<Word> rotated_key(key.size(), 0);
            bv_.inverse_rotate_key(key, galois_element, rotated_key);
            bv_.apply_galois_digits(input_, rotated_key, galois_element, output_, true);
            std::vector<Word> cached(output_.size());
            output_.copy_to_host(cached.data(), cached.size());
            if (cached != shared)
                throw std::runtime_error(std::string(operation) + " cached transformed key mismatch");
            Ciphertext candidate(reference);
            std::copy(shared.begin(), shared.end(), candidate.data(0));
            Plaintext expected_plain, actual_plain;
            decryptor.decrypt(reference, expected_plain);
            decryptor.decrypt(candidate, actual_plain);
            std::vector<std::complex<double>> expected_slots, actual_slots;
            encoder.decode(expected_plain, expected_slots);
            encoder.decode(actual_plain, actual_slots);
            double error = 0;
            for (std::size_t i = 0; i < expected_slots.size(); ++i)
                error = std::max(error, std::abs(actual_slots[i] - expected_slots[i]));
            std::cout << operation << " shared_digits Q=" << q_.size() << " max_error=" << error
                      << " residue_exact=" << (shared == actual) << '\n';
            if (!std::isfinite(error) || error > 1e-3)
                throw std::runtime_error(std::string(operation) + " shared digit decoded mismatch");
        }
        std::cout << operation << " Q=" << q_.size() << " residues="
                  << actual.size() << " match=EXACT\n";
    }

private:
    std::vector<Modulus> q_;
    std::size_t degree_;
    std::size_t words_;
    GpuWideBv bv_;
    DeviceVector<Word> input_;
    DeviceVector<Word> output_;
    bool prepared_ = false;
    bool digits_prepared_ = false;
};
} // namespace

int main(int argc, char **argv)
try
{
    if (argc != 2 || (std::string(argv[1]) != "15" && std::string(argv[1]) != "31"))
        throw std::invalid_argument("usage: functional_wide_bv_probe 15|31");
    const bool d15 = std::string(argv[1]) == "15";
    {
        const std::vector<std::uint64_t> original{0, 1, (std::uint64_t{1} << 40) - 1,
            (std::uint64_t{1} << 51) - 1, (std::uint64_t{1} << 56) - 1};
        DeviceVector<std::uint64_t> input(original.size(), 0), output(original.size(), 0);
        DeviceVector<std::uint8_t> packed(original.size() * 7, 0);
        input.copy_from_host(original.data(), original.size());
        wide_pack_56(input, packed);
        wide_unpack_56(packed, output);
        std::vector<std::uint64_t> actual(original.size());
        output.copy_to_host(actual.data(), actual.size());
        if (actual != original) throw std::runtime_error("56-bit key packing round trip failed");
        std::cout << "56-bit key packing round trip PASS\n";
    }
    std::vector<std::uint32_t> q_bits{45, 40, 40, 40, 40};
    q_bits.resize(d15 ? 19 : 20, 45);
    ParametersLiteral literal(CKKS, 15, 14, 40, 32, 0, Modulus(0),
                              {}, {}, sec_level_type::none);
    literal.set_log_modulus(q_bits, {51});
    check_fused_qp(literal);
    {
        std::vector<Modulus> primes = literal.q();
        primes.insert(primes.end(), literal.p().begin(), literal.p().end());
        std::uint64_t state = 0x9e3779b97f4a7c15ULL;
        for (const auto &prime : primes)
        {
            const auto q = prime.value(), ratio = prime.const_ratio()[1];
            const auto check = [=](std::uint64_t x) {
                const auto estimate = std::uint64_t((static_cast<unsigned __int128>(x) * ratio) >> 64);
                auto remainder = x - estimate * q;
                if (remainder >= q) remainder -= q;
                if (remainder != x % q) throw std::runtime_error("Barrett reduction mismatch");
            };
            for (auto x : {std::uint64_t{0}, std::uint64_t{1}, q - 1, q, q + 1,
                           UINT64_MAX}) check(x);
            for (int i = 0; i < 10000; ++i)
            {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                check(state);
            }
        }
        std::cout << "Barrett word reduction boundaries and random PASS\n";
    }
    auto factory = PoseidonFactory::get_instance();
    factory->set_device_type(DEVICE_SOFTWARE);
    auto context = factory->create_poseidon_context(literal);
    if (context.key_switch_variant() != BV)
        throw std::runtime_error("expected BV context");

    auto cpu = factory->create_ckks_evaluator(context);
    CKKSEncoder encoder(context);
    KeyGenerator keys(context);
    Decryptor decryptor(context, keys.secret_key());
    PublicKey pk;
    GaloisKeys rotations;
    RelinKeys relin;
    const std::vector<int> steps{1, -3, 16, -127, 512, 4096, 0};
    keys.create_public_key(pk);
    keys.create_galois_keys(steps, rotations);
    keys.create_relin_keys(relin);

    std::vector<std::complex<double>> values(encoder.slot_count());
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = {std::sin(i * 0.031) / 2, std::cos(i * 0.017) / 4};
    Plaintext plaintext;
    encoder.encode(values, std::exp2(40), plaintext);
    Ciphertext input;
    Encryptor encryptor(context, pk);
    encryptor.encrypt(plaintext, input);

    std::cout << std::unitbuf << "D" << argv[1] << " N=" << literal.degree()
              << " Q=" << q_bits.size() << " P=1 bits=51\n";
    const auto galois = context.crt_context()->galois_tool();
    for (const auto level : {input.level(), decltype(input.level()){10}, decltype(input.level()){4}, decltype(input.level()){2}, decltype(input.level()){0}})
    {
        Ciphertext at_level, reference;
        cpu->drop_modulus(input, at_level, level);
        const std::size_t q_count = at_level.coeff_modulus_size();
        BvProbe probe(literal, std::vector<Modulus>(literal.q().begin(), literal.q().begin() + q_count));
        for (int step : steps)
        {
            const auto element = galois->get_elt_from_step(step);
            if (step == 0)
                cpu->conjugate(at_level, rotations, reference);
            else
                cpu->rotate(at_level, reference, step, rotations);
            const auto label = step == 0 ? std::string("conjugation") : "rotate(" + std::to_string(step) + ")";
            probe.check(label.c_str(),
                        at_level, element, rotations.key(element), reference, encoder, decryptor);
        }
        if (level > 0)
        {
            Ciphertext squared;
            cpu->multiply(at_level, at_level, squared);
            cpu->relinearize(squared, reference, relin);
            probe.check("square+relinearization", at_level, 0, relin.key(2), reference, encoder, decryptor);
        }
    }
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
