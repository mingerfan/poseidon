#include "poseidon/batchencoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/keygenerator.h"
#include <iostream>
#include <vector>

using namespace poseidon;

bool check(const char *name, const Ciphertext &cipher, Decryptor &decryptor,
           BatchEncoder &encoder, const std::vector<uint64_t> &expected)
{
    Plaintext plain;
    std::vector<uint64_t> actual;
    decryptor.decrypt(cipher, plain);
    encoder.decode(plain, actual);
    size_t mismatches = actual.size() != expected.size();
    for (size_t i = 0; i < std::min(actual.size(), expected.size()); ++i)
        mismatches += actual[i] != expected[i];
    std::cout << name << " mismatches=" << mismatches
              << " expected0=" << expected[0] << " actual0=" << actual[0] << '\n';
    return mismatches == 0;
}

int main()
{
    bool passed = true;
    PoseidonFactory::get_instance()->set_device_type(DEVICE_SOFTWARE);
    for (size_t p_count : {size_t(1), size_t(2)})
    for (int repetition = 0; repetition < 3; ++repetition)
    {
        ParametersLiteral parameters{BGV, 13, 12, 0, 5, 0, 65537, {}, {}, sec_level_type::none};
        parameters.set_log_modulus(std::vector<uint32_t>(3, 45), std::vector<uint32_t>(p_count, 45));
        auto context = PoseidonFactory::get_instance()->create_poseidon_context(parameters);
        auto evaluator = PoseidonFactory::get_instance()->create_bgv_evaluator(context);
        BatchEncoder encoder(context);
        KeyGenerator keygen(context);
        PublicKey public_key;
        RelinKeys relin;
        GaloisKeys galois;
        keygen.create_public_key(public_key);
        keygen.create_relin_keys(relin);
        keygen.create_galois_keys(std::vector<int>{1, -1}, galois);
        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, keygen.secret_key());
        std::vector<uint64_t> message(parameters.degree());
        for (size_t i = 0; i < message.size(); ++i) message[i] = (i * 7 + 3) % 97;
        auto product = message;
        for (auto &value : product) value = value * value % 65537;
        Plaintext plain;
        encoder.encode(message, plain);
        Ciphertext encrypted, multiplied, relinearized, rotated, restored;
        encryptor.encrypt(plain, encrypted);
        std::cout << "CASE q=3 p=" << p_count << " key_switch_variant="
                  << context.key_switch_variant() << " repetition=" << repetition << '\n';
        passed &= check("fresh", encrypted, decryptor, encoder, message);
        evaluator->multiply(encrypted, encrypted, multiplied);
        passed &= check("multiply_without_relin", multiplied, decryptor, encoder, product);
        evaluator->multiply_relin(encrypted, encrypted, relinearized, relin);
        passed &= check("multiply_relin", relinearized, decryptor, encoder, product);
        evaluator->rotate_row(encrypted, rotated, 1, galois);
        evaluator->rotate_row(rotated, restored, -1, galois);
        passed &= check("rotate_roundtrip", restored, decryptor, encoder, message);
    }
    return passed ? 0 : 1;
}
