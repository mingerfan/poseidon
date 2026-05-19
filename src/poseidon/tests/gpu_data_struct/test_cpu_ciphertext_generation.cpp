#include "poseidon/parameters_literal.h"
#include "poseidon/poseidon_context.h"
#include "poseidon/keygenerator.h"
#include "poseidon/encryptor.h"
#include "poseidon/decryptor.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/plaintext.h"
#include "poseidon/ciphertext.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_TRUE(cond)                                                          \
    do                                                                            \
    {                                                                             \
        if (!(cond))                                                              \
        {                                                                         \
            std::cerr << "[FAILED] " << #cond << " at line " << __LINE__ << "\n"; \
            return EXIT_FAILURE;                                                  \
        }                                                                         \
    } while (0)

#define CHECK_EQ(a, b)                                                             \
    do                                                                             \
    {                                                                              \
        if (!((a) == (b)))                                                         \
        {                                                                          \
            std::cerr << "[FAILED] " << #a << " == " << #b                        \
                      << " at line " << __LINE__ << "\n";                         \
            std::cerr << "  left  = " << (a) << "\n";                             \
            std::cerr << "  right = " << (b) << "\n";                             \
            return EXIT_FAILURE;                                                   \
        }                                                                          \
    } while (0)

int main()
{
    using namespace poseidon;

    try
    {
        std::cout << "===== CPU Ciphertext Generation Test =====\n";

        // 1. 创建 CKKS 参数
        // 第一版测试建议用 4096 或 8192。
        // 4096 更快，8192 更接近后续真实测试。
        const std::size_t degree = 4096;

        ParametersLiteralDefault parms(CKKS, degree, sec_level_type::tc128);
        PoseidonContext context(parms);

        std::cout << "[OK] Context created\n";
        std::cout << "degree          = " << parms.degree() << "\n";
        std::cout << "slots           = " << parms.slot() << "\n";
        std::cout << "q size          = " << parms.q().size() << "\n";
        std::cout << "p size          = " << parms.p().size() << "\n";
        std::cout << "default scale   = " << parms.scale() << "\n";

        // 2. 生成密钥
        KeyGenerator keygen(context);

        PublicKey public_key;
        keygen.create_public_key(public_key);

        const SecretKey &secret_key = keygen.secret_key();

        std::cout << "[OK] Keys generated\n";

        // 3. 创建 encoder / encryptor / decryptor
        CKKSEncoder encoder(context);
        Encryptor encryptor(context, public_key, secret_key);
        Decryptor decryptor(context, secret_key);

        std::cout << "[OK] Encoder / Encryptor / Decryptor created\n";

        // 4. 编码明文
        std::vector<double> input{1.25, -2.5, 3.75, 4.125};

        Plaintext plain;
        encoder.encode(input, parms.scale(), plain);

        std::cout << "[OK] Plaintext encoded\n";

        // 5. 加密生成 CPU Ciphertext
        Ciphertext ct;
        encryptor.encrypt(plain, ct);

        std::cout << "[OK] Ciphertext encrypted\n";

        // 6. 检查 Ciphertext 元信息
        CHECK_TRUE(ct.is_valid());
        CHECK_EQ(ct.size(), static_cast<std::size_t>(2));
        CHECK_EQ(ct.poly_modulus_degree(), degree);
        CHECK_EQ(ct.coeff_modulus_size(), parms.q().size());
        CHECK_TRUE(ct.is_ntt_form());
        CHECK_TRUE(ct.data() != nullptr);

        const std::size_t raw_uint64_count =
            ct.size() * ct.poly_modulus_degree() * ct.coeff_modulus_size();

        CHECK_TRUE(raw_uint64_count > 0);

        std::cout << "ciphertext size              = " << ct.size() << "\n";
        std::cout << "poly_modulus_degree          = " << ct.poly_modulus_degree() << "\n";
        std::cout << "coeff_modulus_size           = " << ct.coeff_modulus_size() << "\n";
        std::cout << "is_ntt_form                  = " << ct.is_ntt_form() << "\n";
        std::cout << "scale                        = " << ct.scale() << "\n";
        std::cout << "raw uint64 count             = " << raw_uint64_count << "\n";
        std::cout << "raw byte count               = " << raw_uint64_count * sizeof(std::uint64_t) << "\n";

        // 7. 检查前几个 raw 数据是否能访问
        std::cout << "first 8 raw coefficients:\n";
        for (std::size_t i = 0; i < 8 && i < raw_uint64_count; i++)
        {
            std::cout << "ct.data()[" << i << "] = " << ct.data()[i] << "\n";
        }

        // 8. 解密并 decode，验证 CPU 端密文确实可用
        Plaintext decrypted_plain;
        decryptor.decrypt(ct, decrypted_plain);

        std::vector<double> output;
        encoder.decode(decrypted_plain, output);

        const double eps = 1e-3;

        for (std::size_t i = 0; i < input.size(); i++)
        {
            double err = std::abs(input[i] - output[i]);
            if (err > eps)
            {
                std::cerr << "[FAILED] decode mismatch at index " << i << "\n";
                std::cerr << "input  = " << input[i] << "\n";
                std::cerr << "output = " << output[i] << "\n";
                std::cerr << "error  = " << err << "\n";
                return EXIT_FAILURE;
            }
        }

        std::cout << "[OK] Decrypt and decode passed\n";
        std::cout << "===== TEST PASSED =====\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[EXCEPTION] " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}