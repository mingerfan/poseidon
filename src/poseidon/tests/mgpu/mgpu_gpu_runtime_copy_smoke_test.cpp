#include "poseidon/batchencoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/keygenerator.h"
#include "poseidon/mgpu/comm/cuda_peer_comm.h"
#include "poseidon/mgpu/comm/gpu_object_materializer.h"
#include "poseidon/mgpu/comm/gpu_comm.h"
#include "poseidon/mgpu/runtime/backend/copy_dispatching_backend.h"
#include "poseidon/mgpu/runtime/backend/poseidon_gpu_execution_backend.h"
#include "poseidon/mgpu/runtime/executor/sequential_schedule_executor.h"
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
    explicit RmmPoolScope(const std::vector<int> &device_ids)
    {
        if (device_ids.empty())
        {
            throw std::invalid_argument("RmmPoolScope requires at least one device");
        }

        scopes_.reserve(device_ids.size());
        for (const int device_id : device_ids)
        {
            cudaError_t status = cudaSetDevice(device_id);
            if (status != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("cudaSetDevice failed: ") + cudaGetErrorString(status));
            }

            auto scope = std::make_unique<DeviceScope>();
            scope->device_id = device_id;
            scope->upstream = std::make_unique<rmm::mr::cuda_memory_resource>();
            scope->pool =
                std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
                    scope->upstream.get(), 1 << 20, std::nullopt);
            scope->previous = rmm::mr::get_current_device_resource();
            rmm::mr::set_current_device_resource(scope->pool.get());
            scopes_.push_back(std::move(scope));
        }
    }

    RmmPoolScope(const RmmPoolScope &) = delete;
    RmmPoolScope &operator=(const RmmPoolScope &) = delete;

    ~RmmPoolScope()
    {
        for (auto iter = scopes_.rbegin(); iter != scopes_.rend(); ++iter)
        {
            try
            {
                (void)cudaSetDevice((*iter)->device_id);
                rmm::mr::set_current_device_resource((*iter)->previous);
            }
            catch (...)
            {}
        }
    }

private:
    struct DeviceScope
    {
        int device_id = 0;
        std::unique_ptr<rmm::mr::cuda_memory_resource> upstream;
        std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool;
        rmm::mr::device_memory_resource *previous = nullptr;
    };

    std::vector<std::unique_ptr<DeviceScope>> scopes_;
};

int check_cuda_runtime()
{
    const int device_count = CudaPeerComm::visible_device_count();
    if (device_count < 2)
    {
        std::cout << "Skipping GPU runtime copy smoke test: requires at least 2 CUDA devices\n";
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

void run_cross_device_copy_smoke()
{
    constexpr int source_device = 0;
    constexpr int destination_device = 1;
    RmmPoolScope rmm_scope({ source_device, destination_device });

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
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, source_device, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, source_device, {}, { value(2) }));
    schedule.ops.push_back(
        op(MgpuOpKind::CopyCipher, destination_device, { value(1) }, { value(3) }));
    schedule.ops.push_back(
        op(MgpuOpKind::CopyCipher, destination_device, { value(2) }, { value(4) }));
    schedule.ops.push_back(
        op(MgpuOpKind::Add, destination_device, { value(3), value(4) }, { value(5) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, destination_device, { value(5) }, {}));

    PoseidonGpuExecutionBackend gpu_backend(context);
    gpu_backend.bind_cipher_upload(1, std::make_shared<Ciphertext>(cipher0));
    gpu_backend.bind_cipher_upload(2, std::make_shared<Ciphertext>(cipher1));

    PoseidonGpuObjectCopyMaterializer materializer;
    CudaPeerComm peer_backend;
    MaterializedGpuComm comm(materializer, peer_backend);
    CopyDispatchingExecutionBackend copy_backend(comm, &gpu_backend);
    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 2 });

    const ScheduleExecutionResult result = executor.run(schedule, copy_backend);
    require(result.ok(), "GPU runtime cross-device schedule failed:\n" + result.format_errors());
    require(gpu_backend.has_cipher_download(5), "missing add output download");
    require_ciphertexts_equal(expected_sum, *gpu_backend.cipher_download(5));
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
        run_cross_device_copy_smoke();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu GPU runtime copy smoke test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu GPU runtime copy smoke tests passed\n";
    return EXIT_SUCCESS;
}
