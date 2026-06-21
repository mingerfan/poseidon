#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/mgpu/comm/gpu_object_materializer.h"

#include <cuda_runtime_api.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace poseidon::mgpu;

namespace
{

constexpr int kSkip = 77;

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void require_contains(const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos)
    {
        throw std::runtime_error("expected text to contain: " + needle + "\ntext:\n" + text);
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
        std::cout << "Skipping GPU object materializer test: no CUDA device is available\n";
        return kSkip;
    }
    return EXIT_SUCCESS;
}

poseidon::gpu::GpuCiphertextData make_single_shard_ciphertext(int device_id)
{
    auto object = poseidon::gpu::GpuCiphertextData::allocate_single_device(
        /*degree=*/8,
        /*q_count=*/2,
        /*component_count=*/2,
        device_id,
        /*p_count=*/0);
    object.meta.component_count = 2;
    return object;
}

poseidon::gpu::GpuCiphertextData make_multi_shard_ciphertext(int device_id)
{
    poseidon::gpu::GpuPolyShard first;
    first.limb_begin = 0;
    first.limb_count = 1;
    first.coeff_begin = 0;
    first.coeff_count = 8;

    poseidon::gpu::GpuPolyShard second;
    second.limb_begin = 1;
    second.limb_count = 1;
    second.coeff_begin = 0;
    second.coeff_count = 8;

    auto object = poseidon::gpu::GpuCiphertextData::allocate_single_device_sharded(
        /*degree=*/8,
        /*q_count=*/2,
        /*component_count=*/2,
        device_id,
        std::vector<poseidon::gpu::GpuPolyShard>{ first, second },
        /*p_count=*/0);
    object.meta.component_count = 2;
    return object;
}

void test_materializer_accepts_single_full_shard_ciphertext()
{
    constexpr int device_id = 0;
    auto source = std::make_shared<poseidon::gpu::GpuCiphertextData>(
        make_single_shard_ciphertext(device_id));

    PoseidonGpuObjectCopyMaterializer materializer;
    MaterializedGpuObjectCopy materialized = materializer.materialize_copy(GpuCommCopyRequest{
        1,
        2,
        MgpuValueKind::Ciphertext,
        device_id,
        device_id,
        source,
    });

    require(materialized.destination_object != nullptr, "destination object should be set");
    require(materialized.object_copy.buffers.size() == 1, "copy should have one buffer");
    const std::size_t expected_bytes =
        source->fields_[0].size() * sizeof(poseidon::gpu::GpuWord);
    require(materialized.object_copy.buffers[0].bytes == expected_bytes, "copy byte count mismatch");

    auto destination =
        std::static_pointer_cast<poseidon::gpu::GpuCiphertextData>(materialized.destination_object);
    require(destination->polys_[0].shards.size() == 1, "destination should keep one shard");
    require(
        destination->polys_[0].shards[0].limb_count ==
            source->polys_[0].shards[0].limb_count,
        "destination shard limb_count mismatch");
}

void test_materializer_rejects_multi_shard_ciphertext()
{
    constexpr int device_id = 0;
    auto source = std::make_shared<poseidon::gpu::GpuCiphertextData>(
        make_multi_shard_ciphertext(device_id));

    PoseidonGpuObjectCopyMaterializer materializer;
    bool failed = false;
    try
    {
        (void)materializer.materialize_copy(GpuCommCopyRequest{
            1,
            2,
            MgpuValueKind::Ciphertext,
            device_id,
            device_id,
            source,
        });
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(ex.what(), "one full shard");
    }

    require(failed, "multi-shard ciphertext copy should fail");
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
        RmmPoolScope rmm_scope(0);
        test_materializer_accepts_single_full_shard_ciphertext();
        test_materializer_rejects_multi_shard_ciphertext();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu GPU object materializer test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu GPU object materializer tests passed\n";
    return EXIT_SUCCESS;
}
