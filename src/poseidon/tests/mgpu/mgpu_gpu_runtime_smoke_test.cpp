#include "poseidon/batchencoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/keygenerator.h"
#include "poseidon/mgpu/runtime/poseidon_gpu_schedule_handler.h"
#include "poseidon/mgpu/runtime/schedule_interpreter.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

#include <cuda_runtime_api.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace poseidon;
using namespace poseidon::mgpu;

namespace
{

constexpr int kSkip = 77;

MgpuValueRef value(ValueId id)
{
    return MgpuValueRef{ id };
}

MgpuOp op(
    MgpuOpKind kind, int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs)
{
    return MgpuOp{ kind, device_id, std::move(inputs), std::move(outputs), {} };
}

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void require_plaintexts_equal(const Plaintext &expected, const Plaintext &actual)
{
    require(actual.coeff_count() == expected.coeff_count(), "plaintext coeff_count mismatch");
    require(actual.parms_id() == expected.parms_id(), "plaintext parms_id mismatch");
    require(actual.scale() == expected.scale(), "plaintext scale mismatch");
    for (std::size_t i = 0; i < expected.coeff_count(); ++i)
    {
        if (actual.data()[i] != expected.data()[i])
        {
            std::ostringstream stream;
            stream << "plaintext coefficient mismatch at " << i;
            throw std::runtime_error(stream.str());
        }
    }
}

void require_ciphertexts_equal(const Ciphertext &expected, const Ciphertext &actual)
{
    require(actual.parms_id() == expected.parms_id(), "ciphertext parms_id mismatch");
    require(actual.size() == expected.size(), "ciphertext size mismatch");
    require(
        actual.poly_modulus_degree() == expected.poly_modulus_degree(),
        "ciphertext degree mismatch");
    require(
        actual.coeff_modulus_size() == expected.coeff_modulus_size(),
        "ciphertext coeff modulus size mismatch");
    require(actual.is_ntt_form() == expected.is_ntt_form(), "ciphertext NTT form mismatch");
    require(actual.scale() == expected.scale(), "ciphertext scale mismatch");
    require(
        actual.correction_factor() == expected.correction_factor(),
        "ciphertext correction factor mismatch");

    const auto word_count =
        expected.size() * expected.poly_modulus_degree() * expected.coeff_modulus_size();
    for (std::size_t i = 0; i < word_count; ++i)
    {
        if (actual.data()[i] != expected.data()[i])
        {
            std::ostringstream stream;
            stream << "ciphertext word mismatch at " << i;
            throw std::runtime_error(stream.str());
        }
    }
}

class RmmPoolScope
{
public:
    explicit RmmPoolScope(int device_id)
        : device_id_(device_id), pool_(&upstream_, 1 << 20, std::nullopt)
    {
        const cudaError_t status = cudaSetDevice(device_id_);
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
            (void)cudaSetDevice(device_id_);
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

int check_cuda_runtime()
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0)
    {
        std::cout << "Skipping GPU runtime smoke test: no CUDA device is available\n";
        return kSkip;
    }
    return EXIT_SUCCESS;
}

ParametersLiteral make_gpu_test_parameters()
{
    return ParametersLiteral(
        BFV,
        /*log_n=*/12,
        /*log_slots=*/12,
        /*log_scale=*/0,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        Modulus(65537),
        std::vector<Modulus>{
            Modulus(786433),
            Modulus(1032193),
        },
        std::vector<Modulus>{
            Modulus(1179649),
        },
        sec_level_type::none);
}

ParametersLiteral make_gpu_ckks_test_parameters()
{
    ParametersLiteral parms(
        CKKS,
        /*log_n=*/12,
        /*log_slots=*/11,
        /*log_scale=*/20,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        Modulus(0),
        std::vector<Modulus>{},
        std::vector<Modulus>{},
        sec_level_type::none);
    parms.set_log_modulus(std::vector<std::uint32_t>(3, 30), {});
    return parms;
}

void run_gpu_runtime_smoke()
{
    constexpr int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const ParametersLiteral parms = make_gpu_test_parameters();
    PoseidonContext context(parms);
    auto cpu_evaluator = PoseidonFactory::get_instance()->create_bfv_evaluator(context);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);

    BatchEncoder encoder(context);
    Plaintext plain0;
    Plaintext plain1;
    encoder.encode(std::vector<std::uint64_t>{ 1, 2, 3, 4, 5, 6, 7, 8 }, plain0);
    encoder.encode(std::vector<std::uint64_t>{ 8, 7, 6, 5, 4, 3, 2, 1 }, plain1);

    Encryptor encryptor(context, public_key, keygen.secret_key());
    Ciphertext cipher0;
    Ciphertext cipher1;
    encryptor.encrypt(plain0, cipher0);
    encryptor.encrypt(plain1, cipher1);

    Ciphertext expected_sum;
    cpu_evaluator->add(cipher0, cipher1, expected_sum);

    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, device_id, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, device_id, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, device_id, {}, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Add, device_id, { value(1), value(2) }, { value(4) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(1) }, {}));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(3) }, {}));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(4) }, {}));

    PoseidonGpuScheduleHandler handler(context);
    handler.bind_cipher_upload(1, std::make_shared<Ciphertext>(cipher0));
    handler.bind_cipher_upload(2, std::make_shared<Ciphertext>(cipher1));
    handler.bind_plain_upload(3, std::make_shared<Plaintext>(plain0));

    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 1 });
    const ScheduleExecutionResult result = interpreter.run(schedule, handler);
    require(result.ok(), "GPU runtime schedule failed:\n" + result.format_errors());
    require(handler.has_cipher_download(1), "missing ciphertext download");
    require(handler.has_plain_download(3), "missing plaintext download");
    require(handler.has_cipher_download(4), "missing add output download");
    require_ciphertexts_equal(cipher0, *handler.cipher_download(1));
    require_plaintexts_equal(plain0, *handler.plain_download(3));
    require_ciphertexts_equal(expected_sum, *handler.cipher_download(4));
}

void run_gpu_runtime_multiply_plain_smoke()
{
    constexpr int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const ParametersLiteral parms = make_gpu_ckks_test_parameters();
    PoseidonContext context(parms);
    auto cpu_evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);

    CKKSEncoder encoder(context);
    Plaintext plain0;
    Plaintext plain1;
    encoder.encode(std::vector<double>{ 1.0, 2.0, 3.0, 4.0 }, parms.scale(), plain0);
    encoder.encode(std::vector<double>{ 0.5, -1.0, 2.0, 3.5 }, parms.scale(), plain1);

    Encryptor encryptor(context, public_key, keygen.secret_key());
    Ciphertext cipher0;
    encryptor.encrypt(plain0, cipher0);

    Ciphertext expected_product;
    cpu_evaluator->multiply_plain(cipher0, plain1, expected_product);

    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, device_id, {}, { value(10) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, device_id, {}, { value(11) }));
    schedule.ops.push_back(
        op(MgpuOpKind::MultiplyPlain, device_id, { value(10), value(11) }, { value(12) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(12) }, {}));

    PoseidonGpuScheduleHandler handler(context);
    handler.bind_cipher_upload(10, std::make_shared<Ciphertext>(cipher0));
    handler.bind_plain_upload(11, std::make_shared<Plaintext>(plain1));

    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 1 });
    const ScheduleExecutionResult result = interpreter.run(schedule, handler);
    require(result.ok(), "GPU runtime multiply_plain schedule failed:\n" + result.format_errors());
    require(handler.has_cipher_download(12), "missing multiply_plain output download");
    require_ciphertexts_equal(expected_product, *handler.cipher_download(12));
}

void run_gpu_runtime_add_plain_rescale_smoke()
{
    constexpr int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const ParametersLiteral parms = make_gpu_ckks_test_parameters();
    PoseidonContext context(parms);
    auto cpu_evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);

    CKKSEncoder encoder(context);
    Plaintext input_plain;
    Plaintext bias_plain;
    encoder.encode(std::vector<double>{ 1.25, -2.0, 3.5, 4.0 }, parms.scale(), input_plain);
    encoder.encode(std::vector<double>{ 0.25, 0.5, -1.5, 2.0 }, parms.scale(), bias_plain);

    Encryptor encryptor(context, public_key, keygen.secret_key());
    Ciphertext input_cipher;
    encryptor.encrypt(input_plain, input_cipher);

    Ciphertext expected_add_plain;
    Ciphertext expected_rescale;
    cpu_evaluator->add_plain(input_cipher, bias_plain, expected_add_plain);
    cpu_evaluator->rescale(expected_add_plain, expected_rescale);

    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, device_id, {}, { value(20) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, device_id, {}, { value(21) }));
    schedule.ops.push_back(
        op(MgpuOpKind::AddPlain, device_id, { value(20), value(21) }, { value(22) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, device_id, { value(22) }, { value(23) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(22) }, {}));
    schedule.ops.push_back(op(MgpuOpKind::Download, device_id, { value(23) }, {}));

    PoseidonGpuScheduleHandler handler(context);
    handler.bind_cipher_upload(20, std::make_shared<Ciphertext>(input_cipher));
    handler.bind_plain_upload(21, std::make_shared<Plaintext>(bias_plain));

    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 1 });
    const ScheduleExecutionResult result = interpreter.run(schedule, handler);
    require(
        result.ok(),
        "GPU runtime add_plain/rescale schedule failed:\n" + result.format_errors());
    require(handler.has_cipher_download(22), "missing add_plain output download");
    require(handler.has_cipher_download(23), "missing rescale output download");
    require_ciphertexts_equal(expected_add_plain, *handler.cipher_download(22));
    require_ciphertexts_equal(expected_rescale, *handler.cipher_download(23));
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
        run_gpu_runtime_smoke();
        run_gpu_runtime_multiply_plain_smoke();
        run_gpu_runtime_add_plain_rescale_smoke();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu GPU runtime smoke test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu GPU runtime smoke tests passed\n";
    return EXIT_SUCCESS;
}
