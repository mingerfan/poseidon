#include "poseidon/batchencoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/key/kswitchkeys.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

#include <cuda_runtime_api.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int kSkip = 77;

void fail(const std::string &message, int line)
{
    std::ostringstream stream;
    stream << "[FAILED] " << message << " at line " << line;
    throw std::runtime_error(stream.str());
}

#define CHECK_TRUE(cond)                                                          \
    do                                                                            \
    {                                                                             \
        if (!(cond))                                                              \
        {                                                                         \
            fail(#cond, __LINE__);                                                \
        }                                                                         \
    } while (0)

#define CHECK_EQ(a, b)                                                            \
    do                                                                            \
    {                                                                             \
        const auto left_value = (a);                                               \
        const auto right_value = (b);                                              \
        if (!(left_value == right_value))                                         \
        {                                                                         \
            std::ostringstream stream;                                            \
            stream << #a << " == " << #b << " (left=" << left_value              \
                   << ", right=" << right_value << ")";                         \
            fail(stream.str(), __LINE__);                                         \
        }                                                                         \
    } while (0)

class RmmPoolScope
{
public:
    explicit RmmPoolScope(int device_id)
        : device_id_(device_id), pool_(&upstream_, 1 << 20, std::nullopt)
    {
        auto status = cudaSetDevice(device_id_);
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("cudaSetDevice failed: ") + cudaGetErrorString(status));
        }

        previous_ = rmm::mr::get_current_device_resource();
        rmm::mr::set_current_device_resource(&pool_);
    }

    RmmPoolScope(const RmmPoolScope &) = delete;
    RmmPoolScope &operator=(const RmmPoolScope &) = delete;

    ~RmmPoolScope()
    {
        try
        {
            cudaSetDevice(device_id_);
            rmm::mr::set_current_device_resource(previous_);
        }
        catch (...)
        {}
    }

private:
    int device_id_ = 0;
    rmm::mr::cuda_memory_resource upstream_;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_;
    rmm::mr::device_memory_resource *previous_ = nullptr;
};

poseidon::ParametersLiteral make_gpu_test_parameters()
{
    return poseidon::ParametersLiteral(
        BFV,
        /*log_n=*/12,
        /*log_slots=*/12,
        /*log_scale=*/0,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        poseidon::Modulus(65537),
        std::vector<poseidon::Modulus>{
            poseidon::Modulus(786433),
            poseidon::Modulus(1032193),
        },
        std::vector<poseidon::Modulus>{
            poseidon::Modulus(1179649),
        },
        poseidon::sec_level_type::none);
}

void expect_device_words_equal_moduli(
    const poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> &device_words,
    const std::vector<poseidon::Modulus> &moduli)
{
    CHECK_EQ(device_words.size(), moduli.size());

    std::vector<poseidon::gpu::GpuWord> host_words(device_words.size());
    if (!host_words.empty())
    {
        device_words.copy_to_host(host_words.data(), host_words.size());
    }

    for (std::size_t i = 0; i < moduli.size(); ++i)
    {
        CHECK_EQ(static_cast<std::uint64_t>(host_words[i]), moduli[i].value());
    }
}

void expect_parameter_level_matches(
    const poseidon::gpu::GpuLevelInfo &gpu_level,
    const poseidon::ParametersLiteral &cpu_parms,
    int device_id)
{
    CHECK_TRUE(gpu_level.parms_id == cpu_parms.parms_id());
    CHECK_EQ(gpu_level.degree, cpu_parms.degree());
    CHECK_EQ(gpu_level.q_count, cpu_parms.q().size());
    CHECK_EQ(gpu_level.p_count, cpu_parms.p().size());
    CHECK_EQ(gpu_level.shards.size(), std::size_t{1});

    const auto &shard = gpu_level.shards[0];
    CHECK_EQ(shard.device_id, device_id);
    CHECK_EQ(shard.limb_begin, std::size_t{0});
    CHECK_EQ(shard.limb_count, cpu_parms.q().size() + cpu_parms.p().size());

    expect_device_words_equal_moduli(shard.q_primes, cpu_parms.q());
    expect_device_words_equal_moduli(shard.p_primes, cpu_parms.p());
}

void expect_plaintexts_equal(const poseidon::Plaintext &expected,
                             const poseidon::Plaintext &actual)
{
    CHECK_EQ(actual.coeff_count(), expected.coeff_count());
    CHECK_TRUE(actual.parms_id() == expected.parms_id());
    CHECK_EQ(actual.scale(), expected.scale());

    for (std::size_t i = 0; i < expected.coeff_count(); ++i)
    {
        CHECK_EQ(actual.data()[i], expected.data()[i]);
    }
}

void expect_ciphertexts_equal(const poseidon::Ciphertext &expected,
                              const poseidon::Ciphertext &actual)
{
    CHECK_TRUE(actual.parms_id() == expected.parms_id());
    CHECK_EQ(actual.size(), expected.size());
    CHECK_EQ(actual.poly_modulus_degree(), expected.poly_modulus_degree());
    CHECK_EQ(actual.coeff_modulus_size(), expected.coeff_modulus_size());
    CHECK_EQ(actual.is_ntt_form(), expected.is_ntt_form());
    CHECK_EQ(actual.scale(), expected.scale());
    CHECK_EQ(actual.correction_factor(), expected.correction_factor());

    const auto word_count =
        expected.size() * expected.poly_modulus_degree() * expected.coeff_modulus_size();
    for (std::size_t i = 0; i < word_count; ++i)
    {
        CHECK_EQ(actual.data()[i], expected.data()[i]);
    }
}

void expect_gpu_plaintext_shape(const poseidon::gpu::GpuPlaintextData &gpu_plain,
                                const poseidon::Plaintext &cpu_plain,
                                int device_id)
{
    CHECK_TRUE(!gpu_plain.empty());
    CHECK_TRUE(gpu_plain.meta.parms_id == cpu_plain.parms_id());
    CHECK_EQ(gpu_plain.meta.scale, cpu_plain.scale());
    CHECK_EQ(gpu_plain.meta.is_ntt_form, cpu_plain.is_ntt_form());
    CHECK_EQ(gpu_plain.meta.degree, cpu_plain.coeff_count());
    CHECK_EQ(gpu_plain.meta.q_count, std::size_t{1});
    CHECK_EQ(gpu_plain.meta.p_count, std::size_t{0});
    CHECK_EQ(gpu_plain.fields_.size(), std::size_t{1});
    CHECK_EQ(gpu_plain.fields_[0].device_id, device_id);
    CHECK_EQ(gpu_plain.fields_[0].size(), cpu_plain.coeff_count());

    const auto view = gpu_plain.make_const_view();
    CHECK_EQ(view.poly.poly_id, std::size_t{0});
    CHECK_EQ(view.poly.shards.size(), std::size_t{1});
    CHECK_TRUE(view.poly.shards[0].ptr != nullptr);
    CHECK_EQ(view.poly.shards[0].device_id, device_id);
    CHECK_EQ(view.poly.shards[0].limb_count, std::size_t{1});
    CHECK_EQ(view.poly.shards[0].coeff_count, cpu_plain.coeff_count());
}

void expect_gpu_ciphertext_shape(const poseidon::gpu::GpuCiphertextData &gpu_cipher,
                                 const poseidon::Ciphertext &cpu_cipher,
                                 std::size_t q_count,
                                 std::size_t p_count,
                                 int device_id)
{
    CHECK_TRUE(!gpu_cipher.empty());
    CHECK_TRUE(gpu_cipher.meta.parms_id == cpu_cipher.parms_id());
    CHECK_EQ(gpu_cipher.meta.scale, cpu_cipher.scale());
    CHECK_EQ(gpu_cipher.meta.correction_factor, cpu_cipher.correction_factor());
    CHECK_EQ(gpu_cipher.meta.is_ntt_form, cpu_cipher.is_ntt_form());
    CHECK_EQ(gpu_cipher.meta.degree, cpu_cipher.poly_modulus_degree());
    CHECK_EQ(gpu_cipher.meta.q_count, q_count);
    CHECK_EQ(gpu_cipher.meta.p_count, p_count);
    CHECK_EQ(gpu_cipher.meta.component_count, cpu_cipher.size());
    CHECK_EQ(gpu_cipher.fields_.size(), std::size_t{1});
    CHECK_EQ(gpu_cipher.polys_.size(), cpu_cipher.size());

    const auto limb_count = q_count + p_count;
    const auto field_size = cpu_cipher.poly_modulus_degree() * limb_count;
    CHECK_EQ(gpu_cipher.fields_[0].device_id, device_id);
    CHECK_EQ(gpu_cipher.fields_[0].size(), field_size * cpu_cipher.size());

    const auto view = gpu_cipher.make_const_view();
    CHECK_EQ(view.polys.size(), cpu_cipher.size());

    for (std::size_t component = 0; component < cpu_cipher.size(); ++component)
    {
        CHECK_EQ(gpu_cipher.polys_[component].poly_id, component);
        CHECK_EQ(gpu_cipher.polys_[component].degree, cpu_cipher.poly_modulus_degree());
        CHECK_EQ(gpu_cipher.polys_[component].q_count, q_count);
        CHECK_EQ(gpu_cipher.polys_[component].p_count, p_count);
        CHECK_EQ(gpu_cipher.polys_[component].shards.size(), std::size_t{1});

        const auto &shard = gpu_cipher.polys_[component].shards[0];
        CHECK_EQ(shard.field_index, std::size_t{0});
        CHECK_EQ(shard.field_offset, component * field_size);
        CHECK_EQ(shard.limb_begin, std::size_t{0});
        CHECK_EQ(shard.limb_count, limb_count);
        CHECK_EQ(shard.coeff_begin, std::size_t{0});
        CHECK_EQ(shard.coeff_count, cpu_cipher.poly_modulus_degree());

        CHECK_EQ(view.polys[component].poly_id, component);
        CHECK_EQ(view.polys[component].shards.size(), std::size_t{1});
        CHECK_TRUE(view.polys[component].shards[0].ptr != nullptr);
        CHECK_EQ(view.polys[component].shards[0].device_id, device_id);
    }
}

void expect_gpu_evaluation_key_matches_cpu(
    const poseidon::KSwitchKeys &cpu_keys,
    const poseidon::gpu::GpuEvaluationKeyData &gpu_keys,
    int device_id)
{
    CHECK_TRUE(!gpu_keys.empty());
    CHECK_TRUE(gpu_keys.meta.key_parms_id == cpu_keys.parms_id());
    CHECK_EQ(gpu_keys.meta.key_count, cpu_keys.data().size());
    CHECK_TRUE(gpu_keys.meta.decomposition_count > 0);
    CHECK_EQ(gpu_keys.meta.component_count, std::size_t{2});
    CHECK_EQ(gpu_keys.polys_.size(), gpu_keys.poly_metadata_.size());
    CHECK_TRUE(!gpu_keys.fields_.empty());

    const auto view = gpu_keys.make_const_view();
    CHECK_EQ(view.polys.size(), gpu_keys.polys_.size());

    for (std::size_t poly_index = 0; poly_index < gpu_keys.polys_.size(); ++poly_index)
    {
        const auto &poly = gpu_keys.polys_[poly_index];
        const auto &mapping = gpu_keys.poly_metadata_[poly_index];

        CHECK_EQ(mapping.poly_id, poly_index);
        CHECK_EQ(poly.poly_id, poly_index);
        CHECK_EQ(poly.degree, gpu_keys.meta.degree);
        CHECK_EQ(poly.q_count, gpu_keys.meta.q_count);
        CHECK_EQ(poly.p_count, gpu_keys.meta.p_count);
        CHECK_EQ(poly.shards.size(), std::size_t{1});

        const auto &shard = poly.shards[0];
        CHECK_TRUE(shard.field_index < gpu_keys.fields_.size());
        CHECK_EQ(shard.limb_begin, std::size_t{0});
        CHECK_EQ(shard.limb_count, gpu_keys.meta.q_count + gpu_keys.meta.p_count);
        CHECK_EQ(shard.coeff_begin, std::size_t{0});
        CHECK_EQ(shard.coeff_count, gpu_keys.meta.degree);

        CHECK_EQ(view.polys[poly_index].poly_id, poly_index);
        CHECK_EQ(view.polys[poly_index].shards.size(), std::size_t{1});
        CHECK_TRUE(view.polys[poly_index].shards[0].ptr != nullptr);
        CHECK_EQ(view.polys[poly_index].shards[0].device_id, device_id);

        const auto &cpu_cipher =
            cpu_keys.data()[mapping.key_index][mapping.decomposition_index].data();
        const auto component = mapping.component_index;
        const auto word_count = cpu_cipher.poly_modulus_degree() * cpu_cipher.coeff_modulus_size();
        const auto &field = gpu_keys.fields_[shard.field_index];
        CHECK_TRUE(shard.field_offset + word_count <= field.size());
        CHECK_EQ(field.device_id, device_id);

        std::vector<poseidon::gpu::GpuWord> host_words(field.size());
        field.buffer.copy_to_host(
            host_words.data(),
            host_words.size());
        for (std::size_t i = 0; i < word_count; ++i)
        {
            CHECK_EQ(
                static_cast<std::uint64_t>(host_words[shard.field_offset + i]),
                cpu_cipher.data(component)[i]);
        }
    }
}

int check_cuda_runtime()
{
    int device_count = 0;
    auto status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess)
    {
        std::cerr << "[SKIP] CUDA runtime is unavailable: "
                  << cudaGetErrorString(status) << "\n";
        return kSkip;
    }
    if (device_count == 0)
    {
        std::cerr << "[SKIP] No CUDA device is visible\n";
        return kSkip;
    }
    return EXIT_SUCCESS;
}

int run_test()
{
    using namespace poseidon;
    using namespace poseidon::gpu;

    const int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const auto parms = make_gpu_test_parameters();
    PoseidonContext context(parms);

    CHECK_EQ(parms.degree(), std::size_t{4096});
    CHECK_EQ(context.key_switch_variant(), BV);
    CHECK_TRUE(context.crt_context()->using_keyswitch());

    GpuParameterData gpu_parameters(context, device_id);
    CHECK_TRUE(!gpu_parameters.empty());

    const auto key_context_data = context.crt_context()->key_context_data();
    const auto first_context_data = context.crt_context()->first_context_data();
    CHECK_TRUE(key_context_data != nullptr);
    CHECK_TRUE(first_context_data != nullptr);

    expect_parameter_level_matches(
        gpu_parameters.get_level(key_context_data->parms_id()),
        key_context_data->parms(),
        device_id);
    expect_parameter_level_matches(
        gpu_parameters.get_level(first_context_data->parms_id()),
        first_context_data->parms(),
        device_id);
    std::cout << "[OK] GPU parameter data matches CPU context\n";

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);

    BatchEncoder encoder(context);
    Plaintext plain;
    encoder.encode(std::vector<std::uint64_t>{1, 2, 3, 4, 5, 6, 7, 8}, plain);

    auto gpu_plain = GpuUploader::upload_plaintext(plain, device_id);
    expect_gpu_plaintext_shape(gpu_plain, plain, device_id);

    Plaintext plain_roundtrip;
    GpuUploader::download_plaintext(gpu_plain, plain_roundtrip, context);
    expect_plaintexts_equal(plain, plain_roundtrip);
    std::cout << "[OK] GPU plaintext upload/download roundtrip passed\n";

    Encryptor encryptor(context, public_key, keygen.secret_key());
    Ciphertext cipher;
    encryptor.encrypt(plain, cipher);

    auto gpu_cipher = GpuUploader::upload_ciphertext(cipher, device_id);
    expect_gpu_ciphertext_shape(
        gpu_cipher,
        cipher,
        parms.q().size(),
        std::size_t{0},
        device_id);

    Ciphertext cipher_roundtrip;
    GpuUploader::download_ciphertext(gpu_cipher, cipher_roundtrip, context);
    expect_ciphertexts_equal(cipher, cipher_roundtrip);
    std::cout << "[OK] GPU ciphertext upload/download roundtrip passed\n";

    const auto &public_key_cipher = public_key.data();
    auto gpu_public_key = GpuUploader::upload_ciphertext(public_key_cipher, device_id);
    expect_gpu_ciphertext_shape(
        gpu_public_key,
        public_key_cipher,
        parms.q().size(),
        parms.p().size(),
        device_id);

    Ciphertext public_key_roundtrip;
    GpuUploader::download_ciphertext(gpu_public_key, public_key_roundtrip, context);
    expect_ciphertexts_equal(public_key_cipher, public_key_roundtrip);
    std::cout << "[OK] GPU public key ciphertext layout passed\n";

    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    auto gpu_relin_keys = GpuUploader::upload_relin_keys(relin_keys, device_id);
    expect_gpu_evaluation_key_matches_cpu(relin_keys, gpu_relin_keys, device_id);
    std::cout << "[OK] GPU relin key layout and data passed\n";

    GaloisKeys galois_keys;
    keygen.create_galois_keys(std::vector<int>{1}, galois_keys);
    auto gpu_galois_keys = GpuUploader::upload_galois_keys(galois_keys, device_id);
    expect_gpu_evaluation_key_matches_cpu(galois_keys, gpu_galois_keys, device_id);
    std::cout << "[OK] GPU galois key layout and data passed\n";

    std::cout << "===== GPU DATA STRUCTURE TEST PASSED =====\n";
    return EXIT_SUCCESS;
}

}  // namespace

int main()
{
    const int cuda_status = check_cuda_runtime();
    if (cuda_status != EXIT_SUCCESS)
    {
        return cuda_status;
    }

    try
    {
        return run_test();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[EXCEPTION] " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
