#include "poseidon/runtime_api/poseidon_gpu_api.h"

#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_bootstrap_profile.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"
#include "poseidon/runtime_api/communication/cuda_local_transfer.h"
#include "poseidon/runtime_api/communication/gpu_object_copy.h"
#include "poseidon/runtime_api/rotation_key_basis.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#include <cuda_runtime_api.h>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace poseidon::runtime_api
{

fhegpu::BootProfile make_native_boot_profile(
    const gpu::GpuBootstrapProfile &profile)
{
    if (profile.profile_id.empty() ||
        profile.input_level_min < 0 ||
        profile.input_level_min > profile.input_level_max ||
        profile.input_components <= 0 || profile.output_level < 0 ||
        profile.output_scale_log2 <= 0 || profile.output_components <= 0)
    {
        throw std::invalid_argument(
            "Poseidon GPU native Boot profile metadata is invalid");
    }
    fhegpu::BootProfile result;
    result.profile_id = profile.profile_id;
    result.implementation = fhegpu::BootImplementation::Native;
    result.input_level_min = profile.input_level_min;
    result.input_level_max = profile.input_level_max;
    result.input_components = profile.input_components;
    result.output_level = profile.output_level;
    result.output_scale_log2 = profile.output_scale_log2;
    result.output_components = profile.output_components;
    result.needs_secret_key = false;
    result.needs_host_compute = false;
    return result;
}

namespace
{

constexpr std::size_t kMaximumInitialDevicePoolSize = 24ULL << 30;
constexpr std::size_t kMinimumInitialDevicePoolSize = 64ULL << 20;

std::size_t initial_device_pool_size()
{
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    gpu::gpu_check_cuda(
        cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    (void)total_bytes;
    if (free_bytes < kMinimumInitialDevicePoolSize)
    {
        throw std::runtime_error(
            "Poseidon GPU Runtime has insufficient free device memory");
    }

    const char *configured =
        std::getenv("POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB");
    if (configured != nullptr && *configured != '\0')
    {
        std::size_t consumed = 0;
        const std::string text(configured);
        unsigned long long megabytes = 0;
        try
        {
            megabytes = std::stoull(text, &consumed, 10);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument(
                "POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB must be an integer");
        }
        if (consumed != text.size() || megabytes == 0 ||
            megabytes >
                std::numeric_limits<std::size_t>::max() / (1ULL << 20))
        {
            throw std::invalid_argument(
                "POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB is out of range");
        }
        const std::size_t requested =
            static_cast<std::size_t>(megabytes) << 20;
        if (requested > free_bytes - free_bytes / 10)
        {
            throw std::invalid_argument(
                "POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB leaves insufficient "
                "device-memory headroom");
        }
        return requested;
    }

    return std::min(
        kMaximumInitialDevicePoolSize,
        free_bytes / 4);
}

class DeviceMemoryPool
{
public:
    explicit DeviceMemoryPool(int cuda_device_id)
        : cuda_device_id_(cuda_device_id)
    {
        gpu::gpu_check_cuda(cudaSetDevice(cuda_device_id_), "cudaSetDevice");
        previous_resource_ = rmm::mr::get_current_device_resource();
        upstream_ = std::make_unique<rmm::mr::cuda_memory_resource>();
        pool_ = std::make_unique<
            rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
            upstream_.get(), initial_device_pool_size());
        rmm::mr::set_per_device_resource(
            rmm::cuda_device_id{cuda_device_id_}, pool_.get());
    }

    DeviceMemoryPool(const DeviceMemoryPool &) = delete;
    DeviceMemoryPool &operator=(const DeviceMemoryPool &) = delete;

    ~DeviceMemoryPool()
    {
        gpu::unregister_device_memory_resource_owner(pool_.get(), this);
        (void)cudaSetDevice(cuda_device_id_);
        if (rmm::mr::get_per_device_resource(
                rmm::cuda_device_id{cuda_device_id_}) == pool_.get())
        {
            rmm::mr::set_per_device_resource(
                rmm::cuda_device_id{cuda_device_id_}, previous_resource_);
        }
    }

    rmm::mr::device_memory_resource *resource() const noexcept
    {
        return pool_.get();
    }

private:
    int cuda_device_id_ = 0;
    rmm::mr::device_memory_resource *previous_resource_ = nullptr;
    std::unique_ptr<rmm::mr::cuda_memory_resource> upstream_;
    std::unique_ptr<
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool_;
};

std::shared_ptr<DeviceMemoryPool> acquire_device_memory_pool(int cuda_device_id)
{
    static std::mutex mutex;
    static std::unordered_map<int, std::weak_ptr<DeviceMemoryPool>> pools;

    gpu::gpu_check_cuda(cudaSetDevice(cuda_device_id), "cudaSetDevice");
    std::lock_guard<std::mutex> lock(mutex);
    const auto existing = pools.find(cuda_device_id);
    if (existing != pools.end())
    {
        if (auto pool = existing->second.lock())
        {
            return pool;
        }
    }

    auto *current = rmm::mr::get_current_device_resource();
    if (dynamic_cast<rmm::mr::cuda_memory_resource *>(current) == nullptr)
    {
        return {};
    }

    auto pool = std::make_shared<DeviceMemoryPool>(cuda_device_id);
    gpu::register_device_memory_resource_owner(pool->resource(), pool);
    pools[cuda_device_id] = pool;
    return pool;
}

double exact_scale(int scale_log2)
{
    const double result = std::ldexp(1.0, scale_log2);
    if (!(result > 0.0) || !std::isfinite(result))
    {
        throw std::invalid_argument("scale_log2 is outside the supported double range");
    }
    return result;
}

void require_host_place(const fhegpu::Place &place, const char *where)
{
    if (place.kind != fhegpu::PlaceKind::Host || place.rank != 0 || place.index != 0)
    {
        throw std::invalid_argument(std::string(where) + " requires Host(rank=0,index=0)");
    }
}

void require_device_place(const fhegpu::Place &place, const char *where)
{
    if (place.kind != fhegpu::PlaceKind::Device || place.rank != 0 || place.index < 0)
    {
        throw std::invalid_argument(std::string(where) + " requires a local Device Place");
    }
}

int value_cuda_device_id(const PoseidonGpuValue &value, const char *where)
{
    if (value.place_kind() != fhegpu::PlaceKind::Device)
    {
        throw std::invalid_argument(std::string(where) + " requires a Device value");
    }

    if (value.kind() == fhegpu::ValueKind::Plaintext)
    {
        const auto &plain = value.device_plaintext();
        if (plain.fields_.size() != 1 ||
            plain.fields_.front().device_id != plain.fields_.front().buffer.device_id())
        {
            throw std::invalid_argument(std::string(where) +
                                        " has invalid CUDA device metadata");
        }
        return plain.fields_.front().device_id;
    }

    const auto &cipher = value.device_ciphertext();
    if (cipher.fields_.size() != 1 ||
        cipher.fields_.front().device_id != cipher.fields_.front().buffer.device_id())
    {
        throw std::invalid_argument(std::string(where) +
                                    " has invalid CUDA device metadata");
    }
    return cipher.fields_.front().device_id;
}

const gpu::GpuCiphertextData &require_ciphertext(const std::vector<PoseidonGpuValue> &inputs,
                                                 std::size_t index)
{
    if (index >= inputs.size())
    {
        throw std::invalid_argument("missing GPU ciphertext input");
    }
    return inputs[index].device_ciphertext();
}

const gpu::GpuPlaintextData &require_plaintext(const std::vector<PoseidonGpuValue> &inputs,
                                               std::size_t index)
{
    if (index >= inputs.size())
    {
        throw std::invalid_argument("missing GPU plaintext input");
    }
    return inputs[index].device_plaintext();
}

const Ciphertext &require_host_ciphertext(const std::vector<PoseidonGpuValue> &inputs,
                                          std::size_t index)
{
    if (index >= inputs.size())
    {
        throw std::invalid_argument("missing Host ciphertext input");
    }
    return inputs[index].host_ciphertext();
}

std::vector<int> available_rotation_steps(const PoseidonContext &context,
                                          const GaloisKeys &keys, int requested_step)
{
    const std::size_t slot_count = context.parameters_literal()->slot();
    const int normalized = normalize_rotation_step(requested_step, slot_count);
    if (normalized == 0)
    {
        return {};
    }

    const auto galois_tool = context.crt_context()->galois_tool();
    if (keys.has_key(galois_tool->get_elt_from_step(normalized)))
    {
        return {normalized};
    }

    auto steps = decompose_rotation_step(normalized, slot_count);
    for (int step : steps)
    {
        if (!keys.has_key(galois_tool->get_elt_from_step(step)))
        {
            throw std::runtime_error("Poseidon GPU Api lacks a binary rotation key");
        }
    }
    return steps;
}

communication::CudaTransferRoute cuda_transfer_route(fhegpu::CommHint hint)
{
    switch (hint)
    {
    case fhegpu::CommHint::HostStaged:
        return communication::CudaTransferRoute::HostStaged;
    case fhegpu::CommHint::Auto:
    case fhegpu::CommHint::PointToPoint:
    case fhegpu::CommHint::Broadcast:
    case fhegpu::CommHint::Tree:
    case fhegpu::CommHint::Ring:
        return communication::CudaTransferRoute::Auto;
    }
    throw std::invalid_argument("Poseidon GPU communication hint is unknown");
}

std::size_t checked_mul_size(std::size_t left, std::size_t right,
                             const char *what)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error(what);
    }
    return left * right;
}

double elapsed_milliseconds(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::vector<communication::DeviceBufferCopy> prepare_qp_partial_copy(
    const gpu::GpuQPCiphertextBuffer &source,
    gpu::GpuQPCiphertextBuffer &destination,
    int destination_device)
{
    if (source.device_id < 0 || destination_device < 0 ||
        source.batch_count != 1 || source.degree == 0 ||
        source.q_count == 0 || source.p_count == 0)
    {
        throw std::invalid_argument(
            "GPU QP partial copy source shape is invalid");
    }
    destination.ensure_capacity(
        destination_device,
        source.degree,
        source.q_count,
        source.p_count,
        1);
    const std::size_t q_words = checked_mul_size(
        checked_mul_size(2, source.q_count, "QP Q copy size overflow"),
        source.degree,
        "QP Q copy size overflow");
    const std::size_t p_words = checked_mul_size(
        checked_mul_size(2, source.p_count, "QP P copy size overflow"),
        source.degree,
        "QP P copy size overflow");
    return {
        communication::DeviceBufferCopy{
            source.q.data(),
            destination.q.data(),
            checked_mul_size(
                q_words, sizeof(gpu::GpuWord), "QP Q copy bytes overflow"),
            source.device_id,
            destination_device},
        communication::DeviceBufferCopy{
            source.p.data(),
            destination.p.data(),
            checked_mul_size(
                p_words, sizeof(gpu::GpuWord), "QP P copy bytes overflow"),
            source.device_id,
            destination_device}};
}

gpu::GpuWord checked_gpu_word(std::uint64_t value, const char *what)
{
    if (value > std::numeric_limits<gpu::GpuWord>::max())
    {
        throw std::invalid_argument(what);
    }
    return static_cast<gpu::GpuWord>(value);
}

void ciphertext_limb_shape(const Ciphertext &source, std::size_t &q_count,
                           std::size_t &p_count)
{
    q_count = source.coeff_modulus_size();
    p_count = 0;
    if (!source.polys().empty() && source.polys().front().poly_degree() != 0)
    {
        q_count = source.polys().front().rns_num_q();
        p_count = source.polys().front().rns_num_p();
    }
}

void plaintext_limb_shape(const Plaintext &source, std::size_t &degree,
                          std::size_t &q_count, std::size_t &p_count)
{
    if (source.is_ntt_form())
    {
        degree = source.poly().poly_degree();
        q_count = source.poly().rns_num_q();
        p_count = source.poly().rns_num_p();
        if (degree == 0 || q_count + p_count == 0)
        {
            throw std::invalid_argument(
                "GPU asynchronous plaintext upload has invalid RNS shape");
        }
        return;
    }
    degree = source.coeff_count();
    q_count = source.coeff_count() == 0 ? 0 : 1;
    p_count = 0;
}

gpu::GpuPlaintextData prepare_plaintext_upload(
    const Plaintext &source, int destination_device,
    std::shared_ptr<communication::PinnedHostBuffer> &staging)
{
    if (!source.is_valid())
    {
        throw std::invalid_argument(
            "GPU asynchronous plaintext upload requires a valid plaintext");
    }
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;
    plaintext_limb_shape(source, degree, q_count, p_count);
    const std::size_t word_count = checked_mul_size(
        degree, q_count + p_count, "GPU plaintext upload size overflow");
    if (word_count == 0 || word_count != source.coeff_count())
    {
        throw std::invalid_argument(
            "GPU asynchronous plaintext upload shape mismatch");
    }

    auto destination = gpu::GpuPlaintextData::allocate_single_device(
        degree, q_count, destination_device, p_count);
    destination.meta.parms_id = source.parms_id();
    destination.meta.scale = source.scale();
    destination.meta.is_ntt_form = source.is_ntt_form();

    staging = std::make_shared<communication::PinnedHostBuffer>(
        checked_mul_size(word_count, sizeof(gpu::GpuWord),
                         "GPU plaintext upload byte size overflow"));
    auto *packed = static_cast<gpu::GpuWord *>(staging->data());
    for (std::size_t index = 0; index < word_count; ++index)
    {
        packed[index] = checked_gpu_word(
            source.data()[index],
            "GPU asynchronous plaintext upload residue does not fit in GpuWord");
    }
    return destination;
}

gpu::GpuCiphertextData prepare_ciphertext_upload(
    const Ciphertext &source, int destination_device,
    std::shared_ptr<communication::PinnedHostBuffer> &staging)
{
    if (!source.is_valid() || source.size() == 0)
    {
        throw std::invalid_argument(
            "GPU asynchronous ciphertext upload requires a valid ciphertext");
    }
    std::size_t q_count = 0;
    std::size_t p_count = 0;
    ciphertext_limb_shape(source, q_count, p_count);
    const std::size_t component_words = checked_mul_size(
        source.poly_modulus_degree(), q_count + p_count,
        "GPU ciphertext upload component size overflow");
    const std::size_t word_count = checked_mul_size(
        component_words, source.size(), "GPU ciphertext upload size overflow");

    auto destination = gpu::GpuCiphertextData::allocate_single_device(
        source.poly_modulus_degree(), q_count, source.size(), destination_device,
        p_count);
    destination.meta.parms_id = source.parms_id();
    destination.meta.scale = source.scale();
    destination.meta.correction_factor = source.correction_factor();
    destination.meta.is_ntt_form = source.is_ntt_form();

    staging = std::make_shared<communication::PinnedHostBuffer>(
        checked_mul_size(word_count, sizeof(gpu::GpuWord),
                         "GPU ciphertext upload byte size overflow"));
    auto *packed = static_cast<gpu::GpuWord *>(staging->data());
    for (std::size_t component = 0; component < source.size(); ++component)
    {
        const auto *component_data = source.data(component);
        for (std::size_t index = 0; index < component_words; ++index)
        {
            packed[component * component_words + index] = checked_gpu_word(
                component_data[index],
                "GPU asynchronous ciphertext upload residue does not fit in GpuWord");
        }
    }
    return destination;
}

Plaintext finish_plaintext_download(
    const gpu::GpuPlaintextMeta &metadata,
    const communication::PinnedHostBuffer &staging,
    const PoseidonContext &context)
{
    const std::size_t word_count = checked_mul_size(
        metadata.degree, metadata.q_count + metadata.p_count,
        "GPU plaintext download size overflow");
    if (staging.size() != checked_mul_size(
            word_count, sizeof(gpu::GpuWord),
            "GPU plaintext download byte size overflow"))
    {
        throw std::invalid_argument("GPU plaintext download staging size mismatch");
    }

    Plaintext destination;
    if (metadata.is_ntt_form || metadata.parms_id != parms_id_zero)
    {
        destination.resize(context, metadata.parms_id, word_count);
    }
    else
    {
        destination.resize(word_count);
        destination.parms_id() = parms_id_zero;
    }
    destination.scale() = metadata.scale;
    const auto *packed = static_cast<const gpu::GpuWord *>(staging.data());
    for (std::size_t index = 0; index < word_count; ++index)
    {
        destination.data()[index] = static_cast<std::uint64_t>(packed[index]);
    }
    return destination;
}

Ciphertext finish_ciphertext_download(
    const gpu::GpuCiphertextMeta &metadata,
    const communication::PinnedHostBuffer &staging,
    const PoseidonContext &context)
{
    const std::size_t component_words = checked_mul_size(
        metadata.degree, metadata.q_count + metadata.p_count,
        "GPU ciphertext download component size overflow");
    const std::size_t word_count = checked_mul_size(
        component_words, metadata.component_count,
        "GPU ciphertext download size overflow");
    if (staging.size() != checked_mul_size(
            word_count, sizeof(gpu::GpuWord),
            "GPU ciphertext download byte size overflow"))
    {
        throw std::invalid_argument("GPU ciphertext download staging size mismatch");
    }

    Ciphertext destination;
    destination.resize(context, metadata.parms_id, metadata.component_count);
    if (destination.poly_modulus_degree() != metadata.degree ||
        destination.coeff_modulus_size() != metadata.q_count + metadata.p_count)
    {
        throw std::invalid_argument(
            "GPU ciphertext download shape does not match PoseidonContext");
    }
    destination.scale() = metadata.scale;
    destination.correction_factor() = metadata.correction_factor;
    destination.is_ntt_form() = metadata.is_ntt_form;
    const auto *packed = static_cast<const gpu::GpuWord *>(staging.data());
    for (std::size_t index = 0; index < word_count; ++index)
    {
        destination.data()[index] = static_cast<std::uint64_t>(packed[index]);
    }
    return destination;
}

template <class GpuValue>
void require_single_full_shard(const GpuValue &value, std::size_t component_count,
                               int cuda_device_id, const char *where)
{
    if (value.fields_.size() != 1 || value.fields_.front().device_id != cuda_device_id ||
        value.fields_.front().buffer.device_id() != cuda_device_id)
    {
        throw std::runtime_error(std::string(where) +
                                 " must have one field on the configured CUDA device");
    }

    const std::size_t limb_count = value.meta.q_count + value.meta.p_count;
    const std::size_t component_words = value.meta.degree * limb_count;
    if (value.fields_.front().size() != component_words * component_count)
    {
        throw std::runtime_error(std::string(where) + " field size does not match metadata");
    }
}

void require_full_poly(const gpu::GpuRNSPoly &poly, std::size_t component,
                       std::size_t degree, std::size_t q_count, std::size_t p_count,
                       const char *where)
{
    const std::size_t limb_count = q_count + p_count;
    const std::size_t component_words = degree * limb_count;
    if (poly.poly_id != component || poly.degree != degree || poly.q_count != q_count ||
        poly.p_count != p_count || poly.shards.size() != 1)
    {
        throw std::runtime_error(std::string(where) + " polynomial metadata is invalid");
    }

    const auto &shard = poly.shards.front();
    if (shard.field_index != 0 || shard.field_offset != component * component_words ||
        shard.limb_begin != 0 || shard.limb_count != limb_count || shard.coeff_begin != 0 ||
        shard.coeff_count != degree)
    {
        throw std::runtime_error(std::string(where) + " must use one complete shard");
    }
}

} // namespace

struct PoseidonGpuApi::DeviceState
{
    struct NativeBootstrapState
    {
        NativeBootstrapState(
            gpu::GpuBootstrapData bootstrap_data,
            std::shared_ptr<const gpu::GpuRelinKeysData> relin_keys,
            std::shared_ptr<const gpu::GpuGaloisKeysData> galois_keys)
            : bootstrap_data(std::move(bootstrap_data)),
              relin_keys(std::move(relin_keys)),
              galois_keys(std::move(galois_keys))
        {}

        gpu::GpuBootstrapData bootstrap_data;
        std::shared_ptr<const gpu::GpuRelinKeysData> relin_keys;
        std::shared_ptr<const gpu::GpuGaloisKeysData> galois_keys;
        gpu::GpuBootstrapWorkspace workspace;
    };

    int cuda_device_id = 0;
    std::shared_ptr<DeviceMemoryPool> memory_pool;
    std::unique_ptr<gpu::GpuParameterData> gpu_parameters;
    std::unique_ptr<gpu::GpuEvaluator> evaluator;
    std::unordered_map<std::size_t, std::unique_ptr<gpu::GpuRelinKeysData>>
        relin_keys_by_q_count;
    std::unordered_map<std::size_t, std::unique_ptr<gpu::GpuGaloisKeysData>>
        galois_keys_by_q_count;
    std::unordered_map<std::size_t, std::set<std::uint32_t>>
        galois_elements_by_q_count;
    std::unordered_map<std::string, std::unique_ptr<NativeBootstrapState>>
        native_bootstrap_by_profile;
};

class PoseidonGpuValue::ReadyEvent
{
public:
    static std::shared_ptr<ReadyEvent> record(int cuda_device_id)
    {
        auto ready = std::shared_ptr<ReadyEvent>(new ReadyEvent(cuda_device_id));
        try
        {
            gpu::gpu_check_cuda(cudaSetDevice(cuda_device_id), "cudaSetDevice");
            gpu::gpu_check_cuda(
                cudaEventCreateWithFlags(&ready->event_, cudaEventDisableTiming),
                "cudaEventCreateWithFlags");
            gpu::gpu_check_cuda(
                cudaEventRecord(ready->event_, gpu::gpu_execution_stream()),
                "cudaEventRecord");
            ready->recorded_ = true;
        }
        catch (...)
        {
            (void)cudaSetDevice(cuda_device_id);
            (void)cudaDeviceSynchronize();
            throw;
        }
        return ready;
    }

    ReadyEvent(const ReadyEvent &) = delete;
    ReadyEvent &operator=(const ReadyEvent &) = delete;

    ~ReadyEvent()
    {
        if (event_ != nullptr)
        {
            (void)cudaSetDevice(cuda_device_id_);
            (void)cudaEventDestroy(event_);
        }
    }

    void wait()
    {
        if (waited_)
        {
            return;
        }
        if (!recorded_)
        {
            throw std::logic_error("Poseidon GPU ready event was not recorded");
        }
        else
        {
            gpu::gpu_check_cuda(cudaSetDevice(cuda_device_id_), "cudaSetDevice");
            gpu::gpu_check_cuda(cudaEventSynchronize(event_), "cudaEventSynchronize");
        }
        waited_ = true;
    }

    cudaEvent_t event() const
    {
        if (!recorded_ || event_ == nullptr)
        {
            throw std::logic_error("Poseidon GPU ready event was not recorded");
        }
        return event_;
    }

    void wait_on_execution_stream(int cuda_device_id) const
    {
        if (cuda_device_id_ == cuda_device_id &&
            producer_thread_ == std::this_thread::get_id())
        {
            return;
        }
        gpu::gpu_check_cuda(
            cudaStreamWaitEvent(gpu::gpu_execution_stream(), event(), 0),
            "cudaStreamWaitEvent compute input");
    }

private:
    explicit ReadyEvent(int cuda_device_id)
        : cuda_device_id_(cuda_device_id),
          producer_thread_(std::this_thread::get_id())
    {}

    int cuda_device_id_ = 0;
    cudaEvent_t event_ = nullptr;
    bool recorded_ = false;
    bool waited_ = false;
    std::thread::id producer_thread_;
};

struct PoseidonGpuApi::CommHandle::State
{
    struct DeferredPlaintext
    {
        std::size_t output_slot = 0;
        gpu::GpuPlaintextMeta metadata;
        std::shared_ptr<communication::PinnedHostBuffer> staging;
    };

    struct DeferredCiphertext
    {
        std::size_t output_slot = 0;
        gpu::GpuCiphertextMeta metadata;
        std::shared_ptr<communication::PinnedHostBuffer> staging;
    };

    using DeferredOutput = std::variant<DeferredPlaintext, DeferredCiphertext>;

    // Requests must be destroyed before the buffers they read or write.
    std::vector<std::optional<Value>> outputs;
    std::vector<DeferredOutput> deferred_outputs;
    std::vector<std::optional<communication::CudaTransferRequest>> requests;
    bool waited = false;
};

PoseidonGpuApi::CommHandle::CommHandle() = default;
PoseidonGpuApi::CommHandle::~CommHandle() = default;
PoseidonGpuApi::CommHandle::CommHandle(CommHandle &&) noexcept = default;
PoseidonGpuApi::CommHandle &PoseidonGpuApi::CommHandle::operator=(
    CommHandle &&) noexcept = default;

PoseidonGpuValue::PoseidonGpuValue(Storage storage) : storage_(std::move(storage)) {}

PoseidonGpuValue &PoseidonGpuValue::operator=(const PoseidonGpuValue &other)
{
    if (this != &other)
    {
        ready_ = other.ready_;
        storage_ = other.storage_;
    }
    return *this;
}

PoseidonGpuValue &PoseidonGpuValue::operator=(PoseidonGpuValue &&other) noexcept
{
    if (this != &other)
    {
        ready_ = std::move(other.ready_);
        storage_ = std::move(other.storage_);
    }
    return *this;
}

PoseidonGpuValue PoseidonGpuValue::from_host_plaintext(Plaintext value)
{
    return PoseidonGpuValue(std::make_shared<Plaintext>(std::move(value)));
}

PoseidonGpuValue PoseidonGpuValue::from_host_ciphertext(Ciphertext value)
{
    return PoseidonGpuValue(std::make_shared<Ciphertext>(std::move(value)));
}

PoseidonGpuValue PoseidonGpuValue::from_device_plaintext(gpu::GpuPlaintextData value)
{
    return PoseidonGpuValue(std::make_shared<gpu::GpuPlaintextData>(std::move(value)));
}

PoseidonGpuValue PoseidonGpuValue::from_device_ciphertext(gpu::GpuCiphertextData value)
{
    return PoseidonGpuValue(std::make_shared<gpu::GpuCiphertextData>(std::move(value)));
}

fhegpu::ValueKind PoseidonGpuValue::kind() const
{
    return std::holds_alternative<std::shared_ptr<Plaintext>>(storage_) ||
                   std::holds_alternative<std::shared_ptr<gpu::GpuPlaintextData>>(storage_)
               ? fhegpu::ValueKind::Plaintext
               : fhegpu::ValueKind::Ciphertext;
}

fhegpu::PlaceKind PoseidonGpuValue::place_kind() const
{
    return std::holds_alternative<std::shared_ptr<Plaintext>>(storage_) ||
                   std::holds_alternative<std::shared_ptr<Ciphertext>>(storage_)
               ? fhegpu::PlaceKind::Host
               : fhegpu::PlaceKind::Device;
}

const Plaintext &PoseidonGpuValue::host_plaintext() const
{
    const auto *value = std::get_if<std::shared_ptr<Plaintext>>(&storage_);
    if (value == nullptr || *value == nullptr)
    {
        throw std::invalid_argument("Poseidon GPU Api value is not a Host plaintext");
    }
    return **value;
}

const Ciphertext &PoseidonGpuValue::host_ciphertext() const
{
    const auto *value = std::get_if<std::shared_ptr<Ciphertext>>(&storage_);
    if (value == nullptr || *value == nullptr)
    {
        throw std::invalid_argument("Poseidon GPU Api value is not a Host ciphertext");
    }
    return **value;
}

const gpu::GpuPlaintextData &PoseidonGpuValue::device_plaintext() const
{
    const auto *value = std::get_if<std::shared_ptr<gpu::GpuPlaintextData>>(&storage_);
    if (value == nullptr || *value == nullptr)
    {
        throw std::invalid_argument("Poseidon GPU Api value is not a Device plaintext");
    }
    return **value;
}

const gpu::GpuCiphertextData &PoseidonGpuValue::device_ciphertext() const
{
    const auto *value = std::get_if<std::shared_ptr<gpu::GpuCiphertextData>>(&storage_);
    if (value == nullptr || *value == nullptr)
    {
        throw std::invalid_argument("Poseidon GPU Api value is not a Device ciphertext");
    }
    return **value;
}

PoseidonGpuApi::PoseidonGpuApi(std::string context_id, PoseidonContext context,
                               int cuda_device_id,
                               std::shared_ptr<const RelinKeys> relin_keys,
                               std::shared_ptr<const GaloisKeys> galois_keys,
                               std::shared_ptr<const PublicKey> boot_public_key,
                               std::shared_ptr<const SecretKey> boot_secret_key)
    : PoseidonGpuApi(std::move(context_id), std::move(context),
                     std::vector<int>{cuda_device_id}, std::move(relin_keys),
                     std::move(galois_keys), std::move(boot_public_key),
                     std::move(boot_secret_key))
{}

PoseidonGpuApi::PoseidonGpuApi(std::string context_id, PoseidonContext context,
                               std::vector<int> cuda_device_ids,
                               std::shared_ptr<const RelinKeys> relin_keys,
                               std::shared_ptr<const GaloisKeys> galois_keys,
                               std::shared_ptr<const PublicKey> boot_public_key,
                               std::shared_ptr<const SecretKey> boot_secret_key)
    : context_id_(std::move(context_id)), context_(std::move(context)),
      relin_keys_(std::move(relin_keys)), galois_keys_(std::move(galois_keys))
{
    if (context_id_.empty())
    {
        throw std::invalid_argument("Poseidon GPU Api context id is empty");
    }
    if (context_.parameters_literal()->scheme() != CKKS)
    {
        throw std::invalid_argument("Poseidon GPU Api requires a CKKS context");
    }
    if (static_cast<bool>(boot_public_key) != static_cast<bool>(boot_secret_key))
    {
        throw std::invalid_argument(
            "Poseidon GPU decrypt_reencrypt Boot requires public and secret keys");
    }
    if (boot_public_key)
    {
        boot_encryptor_ = std::make_unique<Encryptor>(context_, *boot_public_key);
        boot_decryptor_ = std::make_unique<Decryptor>(context_, *boot_secret_key);
    }

    if (cuda_device_ids.empty())
    {
        throw std::invalid_argument("Poseidon GPU Api requires at least one CUDA device");
    }

    int visible_device_count = 0;
    gpu::gpu_check_cuda(cudaGetDeviceCount(&visible_device_count), "cudaGetDeviceCount");
    for (std::size_t i = 0; i < cuda_device_ids.size(); ++i)
    {
        const int cuda_device_id = cuda_device_ids[i];
        if (cuda_device_id < 0 || cuda_device_id >= visible_device_count)
        {
            throw std::invalid_argument("Poseidon GPU Api CUDA device id is unavailable");
        }
        if (std::find(cuda_device_ids.begin(), cuda_device_ids.begin() + i,
                      cuda_device_id) != cuda_device_ids.begin() + i)
        {
            throw std::invalid_argument("Poseidon GPU Api CUDA device ids must be unique");
        }
    }

    for (const int destination_device : cuda_device_ids)
    {
        for (const int source_device : cuda_device_ids)
        {
            if (destination_device != source_device &&
                communication::CudaLocalTransfer::can_access_peer(
                    destination_device, source_device))
            {
                communication::CudaLocalTransfer::enable_peer_access(
                    destination_device, source_device);
            }
        }
    }

    const auto parameters = context_.parameters_literal();
    for (const auto &modulus : parameters->q())
    {
        if (modulus.bit_count() > std::numeric_limits<gpu::GpuWord>::digits)
        {
            throw std::invalid_argument("Poseidon GPU Api q modulus does not fit in GpuWord");
        }
    }
    for (const auto &modulus : parameters->p())
    {
        if (modulus.bit_count() > std::numeric_limits<gpu::GpuWord>::digits)
        {
            throw std::invalid_argument("Poseidon GPU Api p modulus does not fit in GpuWord");
        }
    }

    encoder_ = std::make_unique<CKKSEncoder>(context_);
    devices_.reserve(cuda_device_ids.size());
    for (const int cuda_device_id : cuda_device_ids)
    {
        gpu::gpu_check_cuda(cudaSetDevice(cuda_device_id), "cudaSetDevice");
        auto device = std::make_shared<DeviceState>();
        device->cuda_device_id = cuda_device_id;
        device->memory_pool = acquire_device_memory_pool(cuda_device_id);
        device->gpu_parameters =
            std::make_unique<gpu::GpuParameterData>(context_, cuda_device_id);
        device->evaluator = std::make_unique<gpu::GpuEvaluator>(*device->gpu_parameters);
        devices_.push_back(std::move(device));
    }
    synchronize_all_devices();
    cuda_transfer_ =
        std::make_unique<communication::CudaLocalTransfer>(cuda_device_ids);
}

PoseidonGpuApi::~PoseidonGpuApi()
{
    for (const auto &device : devices_)
    {
        (void)cudaSetDevice(device->cuda_device_id);
        (void)cudaDeviceSynchronize();
    }
}

void PoseidonGpuApi::configure_native_bootstrap(
    std::string operator_profile,
    int logical_device_index,
    gpu::GpuBootstrapData bootstrap_data,
    gpu::GpuRelinKeysData relin_keys,
    gpu::GpuGaloisKeysData galois_keys)
{
    if (relin_keys.empty() || galois_keys.empty())
    {
        throw std::invalid_argument(
            "Poseidon GPU native Boot requires uploaded evaluation keys");
    }

    configure_native_bootstrap(
        std::move(operator_profile),
        logical_device_index,
        std::move(bootstrap_data),
        std::make_shared<const gpu::GpuRelinKeysData>(std::move(relin_keys)),
        std::make_shared<const gpu::GpuGaloisKeysData>(std::move(galois_keys)));
}

void PoseidonGpuApi::configure_native_bootstrap(
    std::string operator_profile,
    int logical_device_index,
    gpu::GpuBootstrapData bootstrap_data,
    std::shared_ptr<const gpu::GpuRelinKeysData> relin_keys,
    std::shared_ptr<const gpu::GpuGaloisKeysData> galois_keys)
{
    if (operator_profile.empty())
    {
        throw std::invalid_argument(
            "Poseidon GPU native Boot profile id is empty");
    }
    if (relin_keys == nullptr || galois_keys == nullptr ||
        relin_keys->empty() || galois_keys->empty())
    {
        throw std::invalid_argument(
            "Poseidon GPU native Boot requires uploaded evaluation keys");
    }

    auto &device = device_state(logical_device_index);
    gpu::gpu_check_cuda(cudaSetDevice(device.cuda_device_id), "cudaSetDevice");
    auto state = std::make_unique<DeviceState::NativeBootstrapState>(
        std::move(bootstrap_data),
        std::move(relin_keys),
        std::move(galois_keys));
    const bool inserted = device.native_bootstrap_by_profile.emplace(
        std::move(operator_profile), std::move(state)).second;
    if (!inserted)
    {
        throw std::invalid_argument(
            "Poseidon GPU native Boot profile is already configured");
    }
}

void PoseidonGpuApi::configure_native_bootstrap(
    int logical_device_index,
    gpu::GpuBootstrapProfile profile)
{
    const auto &device = device_state(logical_device_index);
    if (profile.cuda_device_id != device.cuda_device_id)
    {
        throw std::invalid_argument(
            "Poseidon GPU native Boot profile belongs to a different CUDA device");
    }
    configure_native_bootstrap(
        std::move(profile.profile_id),
        logical_device_index,
        std::move(profile.bootstrap_data),
        std::move(profile.relin_keys),
        std::move(profile.galois_keys));
}

void PoseidonGpuApi::configure_multi_gpu_bootstrap(
    std::string operator_profile,
    std::vector<int> logical_device_indices,
    std::size_t c2s_device_limit,
    std::size_t eval_mod_device_limit)
{
    if (operator_profile.empty())
    {
        throw std::invalid_argument(
            "Poseidon multi-GPU Boot profile id is empty");
    }
    if (logical_device_indices.size() != 2 &&
        logical_device_indices.size() != 4)
    {
        throw std::invalid_argument(
            "Poseidon multi-GPU Boot requires exactly two or four devices");
    }
    const std::set<int> unique_devices(
        logical_device_indices.begin(), logical_device_indices.end());
    if (unique_devices.size() != logical_device_indices.size())
    {
        throw std::invalid_argument(
            "Poseidon multi-GPU Boot devices must be distinct");
    }
    if (c2s_device_limit == 0)
    {
        c2s_device_limit = 1;
    }
    if (c2s_device_limit > logical_device_indices.size())
    {
        throw std::invalid_argument(
            "Poseidon multi-GPU Boot C2S device limit exceeds device count");
    }
    if (eval_mod_device_limit == 0)
    {
        eval_mod_device_limit = std::min(
            logical_device_indices.size(), std::size_t{2});
    }
    if ((eval_mod_device_limit != 2 && eval_mod_device_limit != 4) ||
        eval_mod_device_limit > logical_device_indices.size())
    {
        throw std::invalid_argument(
            "Poseidon multi-GPU Boot EvalMod device limit must be 2 or 4 and not exceed device count");
    }
    for (const int logical_device_index : logical_device_indices)
    {
        const auto &device = device_state(logical_device_index);
        if (device.native_bootstrap_by_profile.find(operator_profile) ==
            device.native_bootstrap_by_profile.end())
        {
            throw std::invalid_argument(
                "Poseidon multi-GPU Boot profile is missing on logical device " +
                std::to_string(logical_device_index));
        }
    }

    MultiGpuBootstrapPlan plan;
    plan.logical_device_indices = logical_device_indices;
    plan.eval_mod_device_count = eval_mod_device_limit;
    if (logical_device_indices.size() == 4)
    {
        auto &coordinator = device_state(logical_device_indices.front());
        auto &coordinator_native =
            *coordinator.native_bootstrap_by_profile.at(operator_profile);
        const auto &coordinator_data = coordinator_native.bootstrap_data;
        if (coordinator_data.schedule != gpu::GpuBootstrapSchedule::StCFirst ||
            coordinator_data.linear_transform_mode !=
                gpu::GpuLinearTransformMode::DoubleHoistBsgs ||
            coordinator_data.allow_environment_linear_transform_override)
        {
            throw std::invalid_argument(
                "Poseidon four-GPU Boot requires a fixed StC-first double-hoist profile");
        }
        if (eval_mod_device_limit == 4 &&
            (coordinator_data.eval_mod.polynomial_degree != 22 ||
             coordinator_data.eval_mod.polynomial_basis !=
                 gpu::GpuEvalModPolynomialBasis::Chebyshev ||
             coordinator_data.eval_mod.polynomial_blocks.size() != 6 ||
             coordinator_data.eval_mod.polynomial_combine_steps.size() != 5 ||
             coordinator_data.eval_mod.basis_steps.size() != 5))
        {
            throw std::invalid_argument(
                "Poseidon four-GPU EvalMod root partition requires the degree-22 baby-4 Chebyshev plan");
        }
        const std::size_t layer_count =
            coordinator_data.coeff_to_slot_matrix_qp.data().size();
        if (layer_count == 0)
        {
            throw std::invalid_argument(
                "Poseidon four-GPU Boot has no C2S layers");
        }
        plan.c2s_active_device_counts.reserve(layer_count);
        for (std::size_t layer = 0; layer < layer_count; ++layer)
        {
            const std::size_t group_count = coordinator_data
                .coeff_to_slot_matrix_qp.data()[layer].plan.giant_steps.size();
            if (group_count == 0)
            {
                throw std::invalid_argument(
                    "Poseidon four-GPU Boot has an empty C2S giant-step layer");
            }
            const std::size_t active_count =
                std::min(c2s_device_limit, group_count);
            plan.c2s_active_device_counts.push_back(active_count);

            for (std::size_t device_offset = 0;
                 device_offset < logical_device_indices.size();
                 ++device_offset)
            {
                auto &current = device_state(
                    logical_device_indices[device_offset]);
                auto &native =
                    *current.native_bootstrap_by_profile.at(operator_profile);
                auto &matrices =
                    native.bootstrap_data.coeff_to_slot_matrix_qp.data();
                if (matrices.size() != layer_count ||
                    matrices[layer].plan.giant_steps.size() != group_count)
                {
                    throw std::invalid_argument(
                        "Poseidon four-GPU Boot profiles have different C2S plans");
                }
                if (device_offset >= active_count)
                {
                    continue;
                }
                const std::size_t groups_per_device =
                    group_count / active_count;
                const std::size_t remainder = group_count % active_count;
                const std::size_t group_begin =
                    device_offset * groups_per_device +
                    std::min(device_offset, remainder);
                const std::size_t group_end =
                    group_begin + groups_per_device +
                    (device_offset < remainder ? 1 : 0);
                gpu::GpuUploader::restrict_double_hoist_giant_groups(
                    matrices[layer], group_begin, group_end);
            }
        }
    }
    const bool inserted = multi_gpu_bootstrap_by_profile_.emplace(
        std::move(operator_profile),
        std::move(plan)).second;
    if (!inserted)
    {
        throw std::invalid_argument(
            "Poseidon multi-GPU Boot profile is already configured");
    }
}

std::optional<MultiGpuBootstrapTiming>
PoseidonGpuApi::last_multi_gpu_bootstrap_timing(
    const std::string &operator_profile) const
{
    std::lock_guard<std::mutex> lock(multi_gpu_bootstrap_timing_mutex_);
    const auto found =
        multi_gpu_bootstrap_timing_by_profile_.find(operator_profile);
    if (found == multi_gpu_bootstrap_timing_by_profile_.end())
    {
        return std::nullopt;
    }
    return found->second;
}

std::string PoseidonGpuApi::name() const
{
    return "PoseidonGpuApi";
}

int PoseidonGpuApi::local_device_count() const noexcept
{
    return static_cast<int>(devices_.size());
}

int PoseidonGpuApi::cuda_device_id(int logical_device_index) const
{
    return device_state(logical_device_index).cuda_device_id;
}

PoseidonGpuApi::Value PoseidonGpuApi::encode_plaintext(
    const fhegpu::ValueDesc &output_desc, const std::vector<double> &slots)
{
    require_host_place(output_desc.place, "Poseidon GPU Encode");
    if (output_desc.kind != fhegpu::ValueKind::Plaintext || output_desc.components != 1)
    {
        throw std::invalid_argument("Poseidon GPU Encode output must be a Host plaintext");
    }
    if (output_desc.context != context_id_ || !output_desc.ntt)
    {
        throw std::invalid_argument("Poseidon GPU Encode metadata does not match context");
    }

    const auto parms_id =
        context_.crt_context()->parms_id_map().at(static_cast<std::uint32_t>(output_desc.level));
    Plaintext output;
    encoder_->encode(slots, parms_id, exact_scale(output_desc.scale_log2), output);
    return Value::from_host_plaintext(std::move(output));
}

PoseidonGpuApi::Value PoseidonGpuApi::compute(const fhegpu::ComputeOp &op,
                                              const std::vector<Value> &inputs)
{
    if (op.kind == fhegpu::ComputeKind::Boot)
    {
        const auto attrs = std::get<fhegpu::BootAttrs>(op.attrs);
        if (attrs.implementation == fhegpu::BootImplementation::Native)
        {
            auto &device = device_state(op.place, "Poseidon GPU native Boot");
            if (inputs.size() != 1)
            {
                throw std::invalid_argument(
                    "Poseidon GPU native Boot requires one ciphertext input");
            }
            if (value_cuda_device_id(inputs.front(), "Poseidon GPU native Boot input") !=
                device.cuda_device_id)
            {
                throw std::invalid_argument(
                    "Poseidon GPU native Boot input is not on the operation CUDA device");
            }
            const auto resources =
                device.native_bootstrap_by_profile.find(attrs.operator_profile);
            if (resources == device.native_bootstrap_by_profile.end())
            {
                throw std::runtime_error(
                    "Poseidon GPU native Boot profile is not configured: " +
                    attrs.operator_profile);
            }
            if (inputs.front().ready_ != nullptr)
            {
                inputs.front().ready_->wait_on_execution_stream(
                    device.cuda_device_id);
            }

            gpu::GpuCiphertextData output;
            auto &native = *resources->second;
            const auto multi_plan =
                multi_gpu_bootstrap_by_profile_.find(attrs.operator_profile);
            if (multi_plan == multi_gpu_bootstrap_by_profile_.end())
            {
                device.evaluator->bootstrap(
                    require_ciphertext(inputs, 0),
                    native.bootstrap_data,
                    *native.relin_keys,
                    *native.galois_keys,
                    native.workspace,
                    output);
            }
            else
            {
                const auto &logical_devices =
                    multi_plan->second.logical_device_indices;
                if ((logical_devices.size() != 2 &&
                     logical_devices.size() != 4) ||
                    logical_devices.front() != op.place.index)
                {
                    throw std::runtime_error(
                        "Poseidon multi-GPU Boot must execute on its coordinator device");
                }
                auto &secondary = device_state(logical_devices[1]);
                const auto secondary_resources =
                    secondary.native_bootstrap_by_profile.find(
                        attrs.operator_profile);
                if (secondary_resources ==
                    secondary.native_bootstrap_by_profile.end())
                {
                    throw std::runtime_error(
                        "Poseidon multi-GPU Boot secondary profile is unavailable");
                }
                auto &secondary_native = *secondary_resources->second;
                MultiGpuBootstrapTiming bootstrap_timing;
                bootstrap_timing.gpu_count = logical_devices.size();
                bootstrap_timing.eval_mod_device_count =
                    multi_plan->second.eval_mod_device_count;
                const auto bootstrap_timing_start =
                    std::chrono::steady_clock::now();

                gpu::GpuCiphertextData c2s_dft_result;
                if (logical_devices.size() == 2)
                {
                    const auto prefix_start =
                        std::chrono::steady_clock::now();
                    device.evaluator->bootstrap_stc_first_transform(
                        require_ciphertext(inputs, 0),
                        native.bootstrap_data,
                        *native.galois_keys,
                        native.workspace,
                        c2s_dft_result);
                    gpu::gpu_check_cuda(
                        cudaStreamSynchronize(gpu::gpu_execution_stream()),
                        "cudaStreamSynchronize two-GPU Boot C2S prefix timing");
                    bootstrap_timing.prepare_c2s_ms = elapsed_milliseconds(
                        prefix_start, std::chrono::steady_clock::now());
                }
                else
                {
                    const auto prepare_start =
                        std::chrono::steady_clock::now();
                    gpu::GpuCiphertextData current_layer_input;
                    device.evaluator->bootstrap_stc_first_prepare_c2s(
                        require_ciphertext(inputs, 0),
                        native.bootstrap_data,
                        *native.galois_keys,
                        native.workspace,
                        current_layer_input);
                    gpu::gpu_check_cuda(
                        cudaStreamSynchronize(gpu::gpu_execution_stream()),
                        "cudaStreamSynchronize four-GPU Boot C2S prepare");
                    bootstrap_timing.prepare_c2s_ms = elapsed_milliseconds(
                        prepare_start, std::chrono::steady_clock::now());

                    const auto &active_counts =
                        multi_plan->second.c2s_active_device_counts;
                    const auto &coordinator_matrices =
                        native.bootstrap_data.coeff_to_slot_matrix_qp;
                    if (active_counts.size() !=
                        coordinator_matrices.data().size())
                    {
                        throw std::logic_error(
                            "Poseidon four-GPU Boot C2S plan is incomplete");
                    }

                    for (std::size_t layer = 0;
                         layer < active_counts.size();
                         ++layer)
                    {
                        const std::size_t active_count = active_counts[layer];
                        if (active_count == 0 ||
                            active_count > logical_devices.size())
                        {
                            throw std::logic_error(
                                "Poseidon four-GPU Boot C2S active-device count is invalid");
                        }
                        MultiGpuBootstrapLayerTiming layer_timing;
                        layer_timing.active_device_count = active_count;
                        const auto layer_start =
                            std::chrono::steady_clock::now();
                        const auto partial_start = layer_start;

                        std::vector<gpu::GpuCiphertextData> remote_inputs(
                            active_count > 0 ? active_count - 1 : 0);
                        std::vector<communication::CudaTransferRequest>
                            input_transfers;
                        input_transfers.reserve(remote_inputs.size());
                        for (std::size_t offset = 1;
                             offset < active_count;
                             ++offset)
                        {
                            auto &remote =
                                device_state(logical_devices[offset]);
                            const auto copies =
                                communication::prepare_full_object_copy(
                                    current_layer_input,
                                    remote_inputs[offset - 1],
                                    remote.cuda_device_id);
                            if (copies.size() != 1)
                            {
                                throw std::logic_error(
                                    "Poseidon four-GPU Boot C2S input copy is not contiguous");
                            }
                            input_transfers.push_back(
                                cuda_transfer_->copy_async(
                                    copies.front(),
                                    communication::CudaTransferRoute::Auto));
                        }

                        using PartialQpResult = std::pair<
                            gpu::GpuQPCiphertextBuffer,
                            gpu::GpuCiphertextMeta>;
                        std::vector<std::future<PartialQpResult>>
                            compute_futures;
                        compute_futures.reserve(remote_inputs.size());
                        for (std::size_t offset = 1;
                             offset < active_count;
                             ++offset)
                        {
                            auto &remote =
                                device_state(logical_devices[offset]);
                            auto &remote_native =
                                *remote.native_bootstrap_by_profile.at(
                                    attrs.operator_profile);
                            auto *remote_ptr = &remote;
                            auto *remote_native_ptr = &remote_native;
                            compute_futures.push_back(std::async(
                                std::launch::async,
                                [remote_ptr,
                                 remote_native_ptr,
                                 layer,
                                 remote_input = std::move(
                                     remote_inputs[offset - 1]),
                                 transfer = std::move(
                                     input_transfers[offset - 1])]() mutable {
                                    transfer.wait();
                                    gpu::gpu_check_cuda(
                                        cudaSetDevice(
                                            remote_ptr->cuda_device_id),
                                        "cudaSetDevice four-GPU Boot C2S worker");
                                    gpu::GpuQPCiphertextBuffer partial;
                                    gpu::GpuCiphertextMeta partial_meta;
                                    remote_ptr->evaluator
                                        ->dft_double_hoist_layer_partial_qp(
                                            remote_input,
                                            remote_native_ptr->bootstrap_data
                                                .coeff_to_slot_matrix_qp,
                                            layer,
                                            *remote_native_ptr->galois_keys,
                                            remote_native_ptr->workspace
                                                .coeff_to_slot_double_hoist,
                                            partial,
                                            partial_meta);
                                    gpu::gpu_check_cuda(
                                        cudaStreamSynchronize(
                                            gpu::gpu_execution_stream()),
                                        "cudaStreamSynchronize four-GPU Boot C2S worker");
                                    return PartialQpResult{
                                        std::move(partial), partial_meta};
                                }));
                        }

                        gpu::gpu_check_cuda(
                            cudaSetDevice(device.cuda_device_id),
                            "cudaSetDevice four-GPU Boot C2S coordinator");
                        std::vector<gpu::GpuQPCiphertextBuffer> partials(
                            active_count);
                        std::vector<gpu::GpuCiphertextMeta> partial_metas(
                            active_count);
                        device.evaluator->dft_double_hoist_layer_partial_qp(
                            current_layer_input,
                            coordinator_matrices,
                            layer,
                            *native.galois_keys,
                            native.workspace.coeff_to_slot_double_hoist,
                            partials.front(),
                            partial_metas.front());
                        gpu::gpu_check_cuda(
                            cudaStreamSynchronize(
                                gpu::gpu_execution_stream()),
                            "cudaStreamSynchronize four-GPU Boot C2S coordinator");
                        for (std::size_t offset = 1;
                             offset < active_count;
                             ++offset)
                        {
                            auto partial =
                                compute_futures[offset - 1].get();
                            partials[offset] = std::move(partial.first);
                            partial_metas[offset] = partial.second;
                            if (partial_metas[offset].parms_id !=
                                    partial_metas.front().parms_id ||
                                partial_metas[offset].scale !=
                                    partial_metas.front().scale)
                            {
                                throw std::logic_error(
                                    "Poseidon four-GPU Boot QP partial metadata differs");
                            }
                        }
                        layer_timing.fanout_and_partial_compute_ms =
                            elapsed_milliseconds(
                                partial_start,
                                std::chrono::steady_clock::now());

                        const auto reduction_start =
                            std::chrono::steady_clock::now();
                        for (std::size_t stride = 1;
                             stride < active_count;
                             stride *= 2)
                        {
                            std::vector<std::future<void>> reduce_futures;
                            for (std::size_t destination_offset = 0;
                                 destination_offset < active_count;
                                 destination_offset += 2 * stride)
                            {
                                const std::size_t source_offset =
                                    destination_offset + stride;
                                if (source_offset >= active_count)
                                {
                                    continue;
                                }
                                reduce_futures.push_back(std::async(
                                    std::launch::async,
                                    [&, destination_offset, source_offset]() {
                                        auto &destination_device =
                                            device_state(logical_devices[
                                                destination_offset]);
                                        gpu::GpuQPCiphertextBuffer copied_source;
                                        const auto copies =
                                            prepare_qp_partial_copy(
                                                partials[source_offset],
                                                copied_source,
                                                destination_device
                                                    .cuda_device_id);
                                        if (copies.size() != 2)
                                        {
                                            throw std::logic_error(
                                                "Poseidon four-GPU Boot QP tree copy is incomplete");
                                        }
                                        cuda_transfer_->copy_sync(
                                            copies,
                                            communication::CudaTransferRoute::Auto);
                                        gpu::gpu_check_cuda(
                                            cudaSetDevice(
                                                destination_device
                                                    .cuda_device_id),
                                            "cudaSetDevice four-GPU Boot tree reduction");
                                        destination_device.evaluator
                                            ->add_double_hoist_partial_qp_inplace(
                                            partials[destination_offset],
                                            copied_source,
                                            partial_metas[
                                                destination_offset]);
                                        gpu::gpu_check_cuda(
                                            cudaStreamSynchronize(
                                                gpu::gpu_execution_stream()),
                                            "cudaStreamSynchronize four-GPU Boot tree reduction");
                                    }));
                            }
                            for (auto &future : reduce_futures)
                            {
                                future.get();
                            }
                        }
                        layer_timing.qp_reduction_ms =
                            elapsed_milliseconds(
                                reduction_start,
                                std::chrono::steady_clock::now());
                        const auto shared_moddown_start =
                            std::chrono::steady_clock::now();
                        gpu::gpu_check_cuda(
                            cudaSetDevice(device.cuda_device_id),
                            "cudaSetDevice four-GPU Boot shared ModDown");
                        device.evaluator
                            ->finalize_dft_double_hoist_layer_partial_qp(
                                partials.front(),
                                partial_metas.front(),
                                coordinator_matrices,
                                layer,
                                native.workspace
                                    .coeff_to_slot_double_hoist,
                                current_layer_input);
                        gpu::gpu_check_cuda(
                            cudaStreamSynchronize(
                                gpu::gpu_execution_stream()),
                            "cudaStreamSynchronize four-GPU Boot shared ModDown");
                        const auto layer_end =
                            std::chrono::steady_clock::now();
                        layer_timing.shared_moddown_rescale_ms =
                            elapsed_milliseconds(
                                shared_moddown_start, layer_end);
                        layer_timing.total_ms =
                            elapsed_milliseconds(layer_start, layer_end);
                        bootstrap_timing.c2s_layers.push_back(layer_timing);
                    }
                    c2s_dft_result = std::move(current_layer_input);
                }
                gpu::gpu_check_cuda(
                    cudaStreamSynchronize(gpu::gpu_execution_stream()),
                    "cudaStreamSynchronize multi-GPU Boot prefix");

                const auto eval_mod_branches_start =
                    std::chrono::steady_clock::now();
                gpu::GpuCiphertextData secondary_c2s_dft_result;
                const auto prefix_copies =
                    communication::prepare_full_object_copy(
                        c2s_dft_result,
                        secondary_c2s_dft_result,
                        secondary.cuda_device_id);
                if (prefix_copies.size() != 1)
                {
                    throw std::logic_error(
                        "Poseidon multi-GPU Boot prefix copy is not contiguous");
                }
                auto prefix_transfer = cuda_transfer_->copy_async(
                    prefix_copies.front(),
                    communication::CudaTransferRoute::Auto);

                auto run_partitioned_eval_mod =
                    [&](DeviceState &coordinator,
                        auto &coordinator_native,
                        DeviceState &helper,
                        auto &helper_native,
                        const gpu::GpuCiphertextData &branch_input,
                        const char *branch_name) {
                        const auto &coordinator_combines =
                            coordinator_native.bootstrap_data.eval_mod
                                .polynomial_combine_steps;
                        const auto &helper_combines =
                            helper_native.bootstrap_data.eval_mod
                                .polynomial_combine_steps;
                        if (coordinator_combines.size() != 5 ||
                            helper_combines.size() != 5)
                        {
                            throw std::logic_error(
                                "Poseidon four-GPU EvalMod requires the degree-22 five-combine plan");
                        }

                        /* Record, rather than Host-wait for, the extraction. */
                        gpu::gpu_check_cuda(
                            cudaSetDevice(coordinator.cuda_device_id),
                            "cudaSetDevice four-GPU EvalMod input ready");
                        cudaEvent_t branch_input_ready = nullptr;
                        gpu::gpu_check_cuda(
                            cudaEventCreateWithFlags(
                                &branch_input_ready,
                                cudaEventDisableTiming),
                            "cudaEventCreate four-GPU EvalMod input ready");
                        gpu::gpu_check_cuda(
                            cudaEventRecord(
                                branch_input_ready,
                                gpu::gpu_execution_stream()),
                            "cudaEventRecord four-GPU EvalMod input ready");

                        gpu::GpuCiphertextData helper_input;
                        const auto helper_input_copies =
                            communication::prepare_full_object_copy(
                                branch_input,
                                helper_input,
                                helper.cuda_device_id);
                        if (helper_input_copies.size() != 1)
                        {
                            throw std::logic_error(
                                "Poseidon four-GPU EvalMod helper input is not contiguous");
                        }
                        auto helper_input_transfer =
                            cuda_transfer_->copy_async(
                                helper_input_copies.front(),
                                communication::CudaTransferRoute::Auto,
                                branch_input_ready);

                        auto helper_future = std::async(
                            std::launch::async,
                            [&, helper_input = std::move(helper_input),
                             transfer = std::move(helper_input_transfer),
                             source_ready = branch_input_ready,
                             source_device = coordinator.cuda_device_id,
                             helper_partition =
                                 gpu::GpuEvalModPolynomialPartition{
                                     1,
                                     4,
                                     helper_combines[3].output_node,
                                     4,
                                     2,
                                     6}]()
                                mutable {
                                transfer.wait();
                                gpu::gpu_check_cuda(
                                    cudaSetDevice(source_device),
                                    "cudaSetDevice four-GPU EvalMod source event destroy");
                                gpu::gpu_check_cuda(
                                    cudaEventDestroy(source_ready),
                                    "cudaEventDestroy four-GPU EvalMod input ready");
                                gpu::gpu_check_cuda(
                                    cudaSetDevice(helper.cuda_device_id),
                                    "cudaSetDevice four-GPU EvalMod helper");
                                gpu::GpuCiphertextData remainder;
                                helper.evaluator->eval_mod_high_precision(
                                    helper_input,
                                    helper_native.bootstrap_data,
                                    *helper_native.relin_keys,
                                    helper_native.workspace,
                                    remainder,
                                    &helper_partition);
                                gpu::gpu_check_cuda(
                                    cudaStreamSynchronize(
                                    gpu::gpu_execution_stream()),
                                    "cudaStreamSynchronize four-GPU EvalMod helper");
                                return remainder;
                            });

                        gpu::gpu_check_cuda(
                            cudaSetDevice(coordinator.cuda_device_id),
                            "cudaSetDevice four-GPU EvalMod coordinator");
                        const gpu::GpuEvalModPolynomialPartition
                            quotient_partition{
                                0,
                                1,
                                coordinator_combines.front().output_node,
                                5,
                                0,
                                2};
                        gpu::GpuCiphertextData quotient;
                        gpu::GpuCiphertextData product;
                        coordinator.evaluator->eval_mod_high_precision(
                            branch_input,
                            coordinator_native.bootstrap_data,
                            *coordinator_native.relin_keys,
                            coordinator_native.workspace,
                            quotient,
                            &quotient_partition);
                        coordinator.evaluator
                            ->eval_mod_degree22_root_product(
                                quotient,
                                coordinator_native.bootstrap_data,
                                *coordinator_native.relin_keys,
                                coordinator_native.workspace,
                                product);

                        auto remote_remainder = helper_future.get();
                        gpu::GpuCiphertextData local_remainder;
                        const auto remainder_copies =
                            communication::prepare_full_object_copy(
                                remote_remainder,
                                local_remainder,
                                coordinator.cuda_device_id);
                        if (remainder_copies.size() != 1)
                        {
                            throw std::logic_error(
                                "Poseidon four-GPU EvalMod root remainder is not contiguous");
                        }
                        cuda_transfer_->copy_sync(
                            remainder_copies.front(),
                            communication::CudaTransferRoute::Auto);
                        gpu::gpu_check_cuda(
                            cudaSetDevice(coordinator.cuda_device_id),
                            "cudaSetDevice four-GPU EvalMod partial merge");
                        gpu::GpuCiphertextData result;
                        coordinator.evaluator
                            ->eval_mod_degree22_finish_partials(
                                product,
                                local_remainder,
                                coordinator_native.bootstrap_data,
                                *coordinator_native.relin_keys,
                                coordinator_native.workspace,
                                result);
                        gpu::gpu_check_cuda(
                            cudaStreamSynchronize(
                                gpu::gpu_execution_stream()),
                            branch_name);
                        return result;
                    };

                auto imag_future = std::async(
                    std::launch::async,
                    [&, remote_input = std::move(secondary_c2s_dft_result),
                     transfer = std::move(prefix_transfer)]() mutable {
                        transfer.wait();
                        gpu::gpu_check_cuda(
                            cudaSetDevice(secondary.cuda_device_id),
                            "cudaSetDevice multi-GPU Boot imaginary branch");
                        gpu::GpuCiphertextData imag_input;
                        gpu::GpuCiphertextData imag_output;
                        secondary.evaluator->bootstrap_extract_imag(
                            remote_input,
                            secondary_native.bootstrap_data,
                            *secondary_native.galois_keys,
                            secondary_native.workspace,
                            imag_input);
                        if (multi_plan->second.eval_mod_device_count == 4)
                        {
                            auto &helper =
                                device_state(logical_devices[3]);
                            auto &helper_native =
                                *helper.native_bootstrap_by_profile.at(
                                    attrs.operator_profile);
                            imag_output = run_partitioned_eval_mod(
                                secondary,
                                secondary_native,
                                helper,
                                helper_native,
                                imag_input,
                                "cudaStreamSynchronize four-GPU Boot imaginary branch");
                        }
                        else
                        {
                            secondary.evaluator->eval_mod_high_precision(
                                imag_input,
                                secondary_native.bootstrap_data,
                                *secondary_native.relin_keys,
                                secondary_native.workspace,
                                imag_output);
                        }
                        gpu::gpu_check_cuda(
                            cudaStreamSynchronize(gpu::gpu_execution_stream()),
                            "cudaStreamSynchronize multi-GPU Boot imaginary branch");
                        return imag_output;
                    });

                gpu::gpu_check_cuda(
                    cudaSetDevice(device.cuda_device_id),
                    "cudaSetDevice multi-GPU Boot real branch");
                gpu::GpuCiphertextData real_input;
                gpu::GpuCiphertextData real_output;
                device.evaluator->bootstrap_extract_real(
                    c2s_dft_result,
                    native.bootstrap_data,
                    *native.galois_keys,
                    native.workspace,
                    real_input);
                if (multi_plan->second.eval_mod_device_count == 4)
                {
                    auto &helper = device_state(logical_devices[2]);
                    auto &helper_native =
                        *helper.native_bootstrap_by_profile.at(
                            attrs.operator_profile);
                    real_output = run_partitioned_eval_mod(
                        device,
                        native,
                        helper,
                        helper_native,
                        real_input,
                        "cudaStreamSynchronize four-GPU Boot real branch");
                }
                else
                {
                    device.evaluator->eval_mod_high_precision(
                        real_input,
                        native.bootstrap_data,
                        *native.relin_keys,
                        native.workspace,
                        real_output);
                }
                gpu::gpu_check_cuda(
                    cudaStreamSynchronize(gpu::gpu_execution_stream()),
                    "cudaStreamSynchronize multi-GPU Boot real branch");

                auto secondary_imag_output = imag_future.get();
                bootstrap_timing.eval_mod_branches_ms =
                    elapsed_milliseconds(
                        eval_mod_branches_start,
                        std::chrono::steady_clock::now());
                const auto result_copy_start =
                    std::chrono::steady_clock::now();
                gpu::GpuCiphertextData local_imag_output;
                const auto result_copies =
                    communication::prepare_full_object_copy(
                        secondary_imag_output,
                        local_imag_output,
                        device.cuda_device_id);
                if (result_copies.size() != 1)
                {
                    throw std::logic_error(
                        "Poseidon multi-GPU Boot result copy is not contiguous");
                }
                cuda_transfer_->copy_sync(
                    result_copies.front(),
                    communication::CudaTransferRoute::Auto);
                bootstrap_timing.imag_result_copy_ms =
                    elapsed_milliseconds(
                        result_copy_start,
                        std::chrono::steady_clock::now());
                gpu::gpu_check_cuda(
                    cudaSetDevice(device.cuda_device_id),
                    "cudaSetDevice multi-GPU Boot finalize");
                const auto finalize_start =
                    std::chrono::steady_clock::now();
                device.evaluator->bootstrap_stc_first_finalize(
                    real_output,
                    local_imag_output,
                    native.bootstrap_data,
                    native.workspace,
                    output);
                gpu::gpu_check_cuda(
                    cudaStreamSynchronize(gpu::gpu_execution_stream()),
                    "cudaStreamSynchronize multi-GPU Boot finalize timing");
                const auto bootstrap_timing_end =
                    std::chrono::steady_clock::now();
                bootstrap_timing.finalize_ms = elapsed_milliseconds(
                    finalize_start, bootstrap_timing_end);
                bootstrap_timing.total_ms = elapsed_milliseconds(
                    bootstrap_timing_start, bootstrap_timing_end);
                {
                    std::lock_guard<std::mutex> lock(
                        multi_gpu_bootstrap_timing_mutex_);
                    multi_gpu_bootstrap_timing_by_profile_[
                        attrs.operator_profile] = std::move(bootstrap_timing);
                }
            }
            if (output.meta.q_count != q_count_for_level(attrs.target_level) ||
                output.meta.component_count !=
                    static_cast<std::size_t>(attrs.target_components) ||
                output.meta.scale != exact_scale(attrs.target_scale_log2))
            {
                throw std::runtime_error(
                    "Poseidon GPU native Boot output does not match RuntimePlan metadata");
            }

            auto result = Value::from_device_ciphertext(std::move(output));
            result.ready_ = PoseidonGpuValue::ReadyEvent::record(
                device.cuda_device_id);
            retain_in_flight(inputs);
            retain_in_flight({result});
            return result;
        }

        require_host_place(op.place, "Poseidon GPU decrypt_reencrypt Boot");
        if (attrs.implementation != fhegpu::BootImplementation::DecryptReencrypt)
        {
            throw std::runtime_error("Poseidon GPU Boot implementation is unsupported");
        }
        if (!boot_encryptor_ || !boot_decryptor_)
        {
            throw std::runtime_error("Poseidon GPU decrypt_reencrypt Boot has no keys");
        }

        Plaintext decrypted;
        boot_decryptor_->decrypt(require_host_ciphertext(inputs, 0), decrypted);
        std::vector<std::complex<double>> slots;
        encoder_->decode(decrypted, slots);
        Plaintext refreshed;
        const auto parms_id = context_.crt_context()->parms_id_map().at(
            static_cast<std::uint32_t>(attrs.target_level));
        encoder_->encode(slots, parms_id, exact_scale(attrs.target_scale_log2), refreshed);
        Ciphertext output;
        boot_encryptor_->encrypt(refreshed, output);
        return Value::from_host_ciphertext(std::move(output));
    }

    auto &device = device_state(op.place, "Poseidon GPU compute");
    gpu::gpu_check_cuda(cudaSetDevice(device.cuda_device_id), "cudaSetDevice");
    for (const auto &input : inputs)
    {
        if (value_cuda_device_id(input, "Poseidon GPU compute input") !=
            device.cuda_device_id)
        {
            throw std::invalid_argument(
                "Poseidon GPU compute input is not on the operation CUDA device");
        }
    }
    for (const auto &input : inputs)
    {
        if (input.ready_ != nullptr)
        {
            input.ready_->wait_on_execution_stream(device.cuda_device_id);
        }
    }
    gpu::GpuCiphertextData output;
    std::vector<std::shared_ptr<void>> temporaries;

    switch (op.kind)
    {
    case fhegpu::ComputeKind::AddCC:
        device.evaluator->add(require_ciphertext(inputs, 0), require_ciphertext(inputs, 1),
                              output);
        break;
    case fhegpu::ComputeKind::AddCP:
        device.evaluator->add_plain(require_ciphertext(inputs, 0),
                                    require_plaintext(inputs, 1), output);
        break;
    case fhegpu::ComputeKind::SubCC:
        device.evaluator->sub(require_ciphertext(inputs, 0), require_ciphertext(inputs, 1),
                              output);
        break;
    case fhegpu::ComputeKind::SubCP:
        device.evaluator->sub_plain(require_ciphertext(inputs, 0),
                                    require_plaintext(inputs, 1), output);
        break;
    case fhegpu::ComputeKind::MulCC:
        device.evaluator->multiply(require_ciphertext(inputs, 0),
                                   require_ciphertext(inputs, 1), output);
        break;
    case fhegpu::ComputeKind::MulCP:
        device.evaluator->multiply_plain(require_ciphertext(inputs, 0),
                                         require_plaintext(inputs, 1), output);
        break;
    case fhegpu::ComputeKind::Negate:
        device.evaluator->negate(require_ciphertext(inputs, 0), output);
        break;
    case fhegpu::ComputeKind::Rotate:
    {
        const auto &input = require_ciphertext(inputs, 0);
        if (galois_keys_ == nullptr)
        {
            throw std::runtime_error("Poseidon GPU Rotate requires GaloisKeys");
        }
        const auto steps = available_rotation_steps(
            context_, *galois_keys_, std::get<fhegpu::RotateAttrs>(op.attrs).steps);
        if (steps.empty())
        {
            device.evaluator->rotate(
                input, 0, galois_keys_for(device, input.meta.q_count), output);
            break;
        }

        std::vector<std::shared_ptr<gpu::GpuCiphertextData>> intermediates;
        intermediates.reserve(steps.size());
        const gpu::GpuCiphertextData *source = &input;
        for (int step : steps)
        {
            auto next = std::make_shared<gpu::GpuCiphertextData>();
            device.evaluator->rotate(
                *source, step, galois_keys_for(device, input.meta.q_count), *next);
            source = next.get();
            intermediates.push_back(std::move(next));
        }
        output = std::move(*intermediates.back());
        intermediates.pop_back();
        for (auto &intermediate : intermediates)
        {
            temporaries.push_back(std::move(intermediate));
        }
        break;
    }
    case fhegpu::ComputeKind::Rescale:
    {
        if (!max_rescale_levels_per_op_)
        {
            throw std::runtime_error("Poseidon GPU Api preflight was not completed");
        }
        const auto attrs = std::get<fhegpu::RescaleAttrs>(op.attrs);
        const auto &input = require_ciphertext(inputs, 0);
        const auto context_data = context_.crt_context()->get_context_data(input.meta.parms_id);
        if (context_data == nullptr)
        {
            throw std::invalid_argument("Poseidon GPU Rescale input has an unknown parms_id");
        }
        const int input_level = static_cast<int>(context_data->level());
        const int drop_count = input_level - attrs.target_level;
        if (drop_count <= 0 || drop_count > *max_rescale_levels_per_op_)
        {
            throw std::invalid_argument("Poseidon GPU Rescale target level is unsupported");
        }

        std::vector<std::shared_ptr<gpu::GpuCiphertextData>> intermediates;
        intermediates.reserve(static_cast<std::size_t>(drop_count));
        const gpu::GpuCiphertextData *source = &input;
        for (int dropped = 0; dropped < drop_count; ++dropped)
        {
            auto next = std::make_shared<gpu::GpuCiphertextData>();
            device.evaluator->rescale(*source, *next);
            source = next.get();
            intermediates.push_back(std::move(next));
        }
        intermediates.back()->meta.scale = exact_scale(attrs.target_scale_log2);
        output = std::move(*intermediates.back());
        intermediates.pop_back();
        for (auto &intermediate : intermediates)
        {
            temporaries.push_back(std::move(intermediate));
        }
        break;
    }
    case fhegpu::ComputeKind::Relinearize:
    {
        const auto &input = require_ciphertext(inputs, 0);
        if (relin_keys_ == nullptr)
        {
            throw std::runtime_error("Poseidon GPU Relinearize requires RelinKeys");
        }
        device.evaluator->relinearize(input, relin_keys_for(device, input.meta.q_count),
                                      output);
        break;
    }
    case fhegpu::ComputeKind::ModSwitch:
    {
        const auto attrs = std::get<fhegpu::ModSwitchAttrs>(op.attrs);
        if (attrs.target_level < 0)
        {
            throw std::invalid_argument("Poseidon GPU ModSwitch target level is negative");
        }
        const auto parms_id =
            context_.crt_context()->parms_id_map().at(
                static_cast<std::uint32_t>(attrs.target_level));
        device.evaluator->drop_modulus(
            require_ciphertext(inputs, 0), output, parms_id);
        break;
    }
    case fhegpu::ComputeKind::Boot:
        throw std::logic_error("Poseidon GPU Boot reached Device dispatch");
    }

    auto result = Value::from_device_ciphertext(std::move(output));
    result.ready_ = PoseidonGpuValue::ReadyEvent::record(device.cuda_device_id);
    retain_in_flight(inputs, std::move(temporaries));
    retain_in_flight({result});
    return result;
}

PoseidonGpuApi::CommHandle PoseidonGpuApi::communicate_async(
    const fhegpu::CommAction &action, const std::vector<Value> &local_inputs)
{
    if (action.inputs.size() != 1 || action.sources.size() != 1 ||
        action.outputs.size() != action.destinations.size() ||
        action.outputs.size() != action.output_types.size() || local_inputs.size() != 1)
    {
        throw std::invalid_argument("Poseidon GPU communication mapping is invalid");
    }
    if (action.kind == fhegpu::CommKind::Transfer && action.outputs.size() != 1)
    {
        throw std::invalid_argument("Poseidon GPU Transfer requires one output");
    }
    if (action.kind == fhegpu::CommKind::Replicate && action.outputs.size() < 2)
    {
        throw std::invalid_argument("Poseidon GPU Replicate requires at least two outputs");
    }
    if (action.kind != fhegpu::CommKind::Transfer &&
        action.kind != fhegpu::CommKind::Replicate)
    {
        throw std::invalid_argument("Poseidon GPU communication kind is unknown");
    }

    const auto &source_place = action.sources.front();
    const auto &input = local_inputs.front();
    retain_in_flight(local_inputs);
    if (source_place.kind == fhegpu::PlaceKind::Host)
    {
        require_host_place(source_place, "Poseidon GPU communication source");
        if (input.place_kind() != fhegpu::PlaceKind::Host)
        {
            throw std::invalid_argument("Poseidon GPU communication expected a Host input");
        }
    }
    else
    {
        const auto &source_device =
            device_state(source_place, "Poseidon GPU communication source");
        if (value_cuda_device_id(input, "Poseidon GPU communication input") !=
            source_device.cuda_device_id)
        {
            throw std::invalid_argument(
                "Poseidon GPU communication input is not on the source CUDA device");
        }
    }

    for (std::size_t slot = 0; slot < action.destinations.size(); ++slot)
    {
        const auto &destination = action.destinations[slot];
        if (input.kind() != action.output_types[slot])
        {
            throw std::invalid_argument("Poseidon GPU communication value kind mismatch");
        }
        if (destination == source_place)
        {
            throw std::invalid_argument(
                "Poseidon GPU communication destination equals source Place");
        }
        if (std::find(action.destinations.begin(), action.destinations.begin() + slot,
                      destination) != action.destinations.begin() + slot)
        {
            throw std::invalid_argument(
                "Poseidon GPU communication destination is duplicated");
        }
        if (destination.kind == fhegpu::PlaceKind::Host)
        {
            require_host_place(destination, "Poseidon GPU communication destination");
        }
        else
        {
            static_cast<void>(
                device_state(destination, "Poseidon GPU communication destination"));
        }
    }

    const auto requested_route = cuda_transfer_route(action.hint);
    auto &cuda_transfer = *cuda_transfer_;
    CommHandle handle;
    handle.state_ = std::make_unique<CommHandle::State>();
    auto &state = *handle.state_;
    state.outputs.resize(action.outputs.size());
    state.requests.resize(action.outputs.size());
    state.deferred_outputs.reserve(action.outputs.size());
    const cudaEvent_t source_ready =
        input.ready_ != nullptr ? input.ready_->event() : nullptr;

    for (std::size_t slot = 0; slot < action.destinations.size(); ++slot)
    {
        const auto &destination = action.destinations[slot];
        if (source_place.kind == fhegpu::PlaceKind::Host)
        {
            const int destination_cuda_device =
                device_state(destination, "Poseidon GPU communication destination")
                    .cuda_device_id;
            if (input.kind() == fhegpu::ValueKind::Plaintext)
            {
                std::shared_ptr<communication::PinnedHostBuffer> staging;
                auto output = prepare_plaintext_upload(
                    input.host_plaintext(), destination_cuda_device, staging);
                state.requests[slot].emplace(
                    cuda_transfer.copy_host_to_device_async(
                        staging, output.fields_.front().data(), staging->size(),
                        destination_cuda_device));
                state.outputs[slot].emplace(
                    Value::from_device_plaintext(std::move(output)));
            }
            else
            {
                std::shared_ptr<communication::PinnedHostBuffer> staging;
                auto output = prepare_ciphertext_upload(
                    input.host_ciphertext(), destination_cuda_device, staging);
                state.requests[slot].emplace(
                    cuda_transfer.copy_host_to_device_async(
                        staging, output.fields_.front().data(), staging->size(),
                        destination_cuda_device));
                state.outputs[slot].emplace(
                    Value::from_device_ciphertext(std::move(output)));
            }
            continue;
        }

        if (destination.kind == fhegpu::PlaceKind::Host)
        {
            if (input.kind() == fhegpu::ValueKind::Plaintext)
            {
                const auto &source = input.device_plaintext();
                const std::size_t bytes = checked_mul_size(
                    source.fields_.front().size(), sizeof(gpu::GpuWord),
                    "GPU plaintext download byte size overflow");
                auto staging =
                    std::make_shared<communication::PinnedHostBuffer>(bytes);
                state.requests[slot].emplace(
                    cuda_transfer.copy_device_to_host_async(
                        source.fields_.front().data(),
                        source.fields_.front().device_id, staging, bytes,
                        source_ready));
                state.deferred_outputs.emplace_back(
                    CommHandle::State::DeferredPlaintext{
                        slot, source.meta, std::move(staging)});
            }
            else
            {
                const auto &source = input.device_ciphertext();
                const std::size_t bytes = checked_mul_size(
                    source.fields_.front().size(), sizeof(gpu::GpuWord),
                    "GPU ciphertext download byte size overflow");
                auto staging =
                    std::make_shared<communication::PinnedHostBuffer>(bytes);
                state.requests[slot].emplace(
                    cuda_transfer.copy_device_to_host_async(
                        source.fields_.front().data(),
                        source.fields_.front().device_id, staging, bytes,
                        source_ready));
                state.deferred_outputs.emplace_back(
                    CommHandle::State::DeferredCiphertext{
                        slot, source.meta, std::move(staging)});
            }
            continue;
        }

        const int destination_cuda_device =
            device_state(destination, "Poseidon GPU communication destination")
                .cuda_device_id;
        if (input.kind() == fhegpu::ValueKind::Plaintext)
        {
            gpu::GpuPlaintextData output;
            const auto copies = communication::prepare_full_object_copy(
                input.device_plaintext(), output, destination_cuda_device);
            if (copies.size() != 1)
            {
                throw std::logic_error(
                    "GPU plaintext Transfer produced multiple buffer copies");
            }
            state.requests[slot].emplace(cuda_transfer.copy_async(
                copies.front(), requested_route, source_ready));
            state.outputs[slot].emplace(
                Value::from_device_plaintext(std::move(output)));
        }
        else
        {
            gpu::GpuCiphertextData output;
            const auto copies = communication::prepare_full_object_copy(
                input.device_ciphertext(), output, destination_cuda_device);
            if (copies.size() != 1)
            {
                throw std::logic_error(
                    "GPU ciphertext Transfer produced multiple buffer copies");
            }
            state.requests[slot].emplace(cuda_transfer.copy_async(
                copies.front(), requested_route, source_ready));
            state.outputs[slot].emplace(
                Value::from_device_ciphertext(std::move(output)));
        }
    }
    return handle;
}

std::vector<PoseidonGpuApi::Value> PoseidonGpuApi::wait(CommHandle &handle)
{
    if (!handle.state_)
    {
        throw std::runtime_error("Poseidon GPU communication handle is empty");
    }
    auto &state = *handle.state_;
    if (state.waited)
    {
        throw std::runtime_error("Poseidon GPU communication handle was already waited");
    }
    for (auto &request : state.requests)
    {
        if (!request)
        {
            throw std::logic_error(
                "Poseidon GPU communication has no transfer request");
        }
        request->wait();
    }
    for (const auto &deferred : state.deferred_outputs)
    {
        std::visit(
            [&](const auto &output) {
                auto &request = state.requests.at(output.output_slot);
                if (!request)
                {
                    throw std::logic_error(
                        "Poseidon GPU deferred output has no transfer request");
                }
                request.reset();
                using Output = std::decay_t<decltype(output)>;
                if constexpr (std::is_same_v<Output,
                                             CommHandle::State::DeferredPlaintext>)
                {
                    state.outputs[output.output_slot].emplace(
                        Value::from_host_plaintext(finish_plaintext_download(
                            output.metadata, *output.staging, context_)));
                }
                else
                {
                    state.outputs[output.output_slot].emplace(
                        Value::from_host_ciphertext(finish_ciphertext_download(
                            output.metadata, *output.staging, context_)));
                }
            },
            deferred);
    }

    std::vector<Value> outputs;
    outputs.reserve(state.outputs.size());
    for (auto &output : state.outputs)
    {
        if (!output)
        {
            throw std::logic_error("Poseidon GPU communication output is unavailable");
        }
        outputs.push_back(std::move(*output));
        output.reset();
    }
    retain_in_flight(outputs);
    for (auto &request : state.requests)
    {
        request.reset();
    }
    state.waited = true;
    state.requests.clear();
    state.deferred_outputs.clear();
    return outputs;
}

void PoseidonGpuApi::synchronize(Value &value)
{
    if (value.place_kind() == fhegpu::PlaceKind::Device)
    {
        if (value.ready_ != nullptr)
        {
            value.ready_->wait();
        }
        else
        {
            const int value_device =
                value_cuda_device_id(value, "Poseidon GPU synchronize");
            const auto configured =
                std::find_if(
                    devices_.begin(),
                    devices_.end(),
                    [value_device](const auto &device) {
                        return device->cuda_device_id == value_device;
                    });
            if (configured == devices_.end())
            {
                throw std::invalid_argument(
                    "Poseidon GPU synchronize value is not on a configured CUDA device");
            }
            synchronize_device(value_device);
        }
    }
    synchronize_all_devices();
    release_completed_in_flight();
}

void PoseidonGpuApi::preflight(std::string_view plan_source_sha256,
                               bool skip_artifact_digest_checks,
                               const fhegpu::TargetConfig &target,
                               const fhegpu::OperatorSpec &operator_spec,
                               const fhegpu::PlanRequirements &requirements)
{
    static_cast<void>(skip_artifact_digest_checks);
    if (plan_source_sha256.size() != 71 || plan_source_sha256.substr(0, 7) != "sha256:")
    {
        throw std::invalid_argument("invalid RuntimePlan source SHA-256");
    }
    if (target.target_id != "poseidon-ckks-gpu" || target.capability_version != 1)
    {
        throw std::invalid_argument("Poseidon GPU Api target is unsupported");
    }
    if (target.world_size != 1)
    {
        throw std::invalid_argument(
            "Poseidon GPU Api currently supports one local process");
    }
    if (target.device_counts.size() != 1 ||
        target.device_counts.front() != local_device_count())
    {
        throw std::invalid_argument(
            "Poseidon GPU Api target device count does not match configured CUDA devices");
    }

    const auto parameters = context_.parameters_literal();
    std::vector<int> modulus_bits;
    modulus_bits.reserve(parameters->q().size());
    for (const auto &modulus : parameters->q())
    {
        modulus_bits.push_back(modulus.bit_count());
    }

    if (operator_spec.status == "placeholder")
    {
        throw std::invalid_argument("Poseidon GPU Api rejects placeholder OperatorSpec");
    }
    if (operator_spec.target_id != target.target_id ||
        operator_spec.context_id != context_id_ ||
        operator_spec.poly_degree != parameters->degree() ||
        operator_spec.rns_moduli_log2 != modulus_bits ||
        operator_spec.max_modulus_log2 !=
            *std::max_element(modulus_bits.begin(), modulus_bits.end()) ||
        operator_spec.default_scale_log2 != static_cast<int>(parameters->log_scale()) ||
        operator_spec.rescale_mode != fhegpu::RescaleMode::Lazy ||
        operator_spec.level_lower_bound < 0 ||
        operator_spec.level_upper_bound >= static_cast<int>(modulus_bits.size()) ||
        operator_spec.level_lower_bound > operator_spec.level_upper_bound)
    {
        throw std::invalid_argument("OperatorSpec parameters do not match Poseidon GPU context");
    }

    const auto boot_support = operator_spec.operators.find(fhegpu::ComputeKind::Boot);
    const bool boot_supported = boot_support != operator_spec.operators.end() &&
                                boot_support->second.supported;
    if (boot_supported != !operator_spec.boot_profiles.empty())
    {
        throw std::invalid_argument(
            "Poseidon GPU OperatorSpec Boot support does not match Boot profiles");
    }
    bool has_configured_native_bootstrap = false;
    for (const auto &device : devices_)
    {
        has_configured_native_bootstrap =
            has_configured_native_bootstrap ||
            !device->native_bootstrap_by_profile.empty();
    }
    const auto rescale = operator_spec.operators.find(fhegpu::ComputeKind::Rescale);
    if (rescale == operator_spec.operators.end() || !rescale->second.supported ||
        !rescale->second.max_levels_per_op || *rescale->second.max_levels_per_op < 4)
    {
        throw std::invalid_argument(
            "Poseidon GPU OperatorSpec must support at least four rescale levels per op");
    }
    max_rescale_levels_per_op_ = *rescale->second.max_levels_per_op;

    for (const auto capability : requirements.capabilities)
    {
        const bool local = capability == fhegpu::RequiredCapability::Encode ||
                           capability == fhegpu::RequiredCapability::Transfer ||
                           capability == fhegpu::RequiredCapability::Replicate;
        const bool native_boot = has_configured_native_bootstrap &&
                                 capability ==
                                     fhegpu::RequiredCapability::BootNative;
        const bool boot = boot_encryptor_ != nullptr && boot_decryptor_ != nullptr &&
                          (capability == fhegpu::RequiredCapability::HostCompute ||
                           capability ==
                               fhegpu::RequiredCapability::BootDecryptReencrypt);
        if (!local && !native_boot && !boot)
        {
            throw std::runtime_error("Poseidon GPU Api lacks required capability: " +
                                     fhegpu::to_string(capability));
        }
    }

    std::map<std::pair<fhegpu::Place, std::size_t>, std::set<std::uint32_t>>
        required_galois_elements;
    const auto galois_tool = context_.crt_context()->galois_tool();

    for (const auto &key : requirements.keys)
    {
        if (key.kind == fhegpu::KeyKind::Secret)
        {
            require_host_place(key.place, "Poseidon GPU SecretKey");
            if (!boot_decryptor_)
            {
                throw std::runtime_error("Poseidon GPU Api lacks decrypt_reencrypt Boot keys");
            }
            continue;
        }

        auto &device = device_state(key.place, "Poseidon GPU key");
        if (!key.level)
        {
            throw std::runtime_error("Poseidon GPU evaluation-key requirement has no level");
        }
        const std::size_t q_count = q_count_for_level(*key.level);
        if (key.kind == fhegpu::KeyKind::Relin)
        {
            if (relin_keys_ == nullptr || !relin_keys_->has_key(2))
            {
                throw std::runtime_error("Poseidon GPU Api lacks RelinKeys");
            }
            materialize_relin_keys(device, q_count);
        }
        else if (key.kind == fhegpu::KeyKind::Galois)
        {
            if (galois_keys_ == nullptr || !key.rotation_step)
            {
                throw std::runtime_error("Poseidon GPU Api lacks GaloisKeys");
            }
            const auto steps = available_rotation_steps(
                context_, *galois_keys_, *key.rotation_step);
            auto &galois_elts = required_galois_elements[{key.place, q_count}];
            for (int step : steps)
            {
                galois_elts.insert(galois_tool->get_elt_from_step(step));
            }
        }
        else
        {
            throw std::runtime_error("Poseidon GPU Api does not support secret-key operations");
        }
    }

    for (const auto &[place_and_q_count, galois_elts] : required_galois_elements)
    {
        auto &device = device_state(
            place_and_q_count.first, "Poseidon GPU GaloisKeys");
        materialize_galois_keys(
            device, place_and_q_count.second, galois_elts);
    }

    synchronize_all_devices();
}

[[noreturn]] void PoseidonGpuApi::abort_all(int, const std::string &reason)
{
    throw std::runtime_error(reason);
}

void PoseidonGpuApi::validate_value(const Value &value,
                                    const fhegpu::ValueDesc &expected) const
{
    if (expected.context != context_id_ || value.kind() != expected.kind ||
        value.place_kind() != expected.place.kind)
    {
        throw std::runtime_error(
            "Poseidon GPU value kind, place, or context does not match ValueDesc");
    }

    int actual_level = 0;
    double actual_scale = 0.0;
    bool actual_ntt = false;
    int actual_components = 0;

    if (expected.place.kind == fhegpu::PlaceKind::Host)
    {
        require_host_place(expected.place, "Poseidon GPU Host value");
        if (expected.kind == fhegpu::ValueKind::Plaintext)
        {
            const auto &plain = value.host_plaintext();
            const auto context_data = context_.crt_context()->get_context_data(plain.parms_id());
            if (context_data == nullptr)
            {
                throw std::runtime_error("Poseidon Host plaintext has an unknown parms_id");
            }
            actual_level = static_cast<int>(context_data->level());
            actual_scale = plain.scale();
            actual_ntt = plain.is_ntt_form();
            actual_components = 1;
        }
        else
        {
            const auto &cipher = value.host_ciphertext();
            if (cipher.poly_modulus_degree() != context_.parameters_literal()->degree())
            {
                throw std::runtime_error(
                    "Poseidon Host ciphertext polynomial degree does not match context");
            }
            actual_level = static_cast<int>(cipher.level());
            actual_scale = cipher.scale();
            actual_ntt = cipher.is_ntt_form();
            actual_components = static_cast<int>(cipher.size());
        }
    }
    else
    {
        const int expected_cuda_device =
            device_state(expected.place, "Poseidon GPU Device value").cuda_device_id;
        if (expected.kind == fhegpu::ValueKind::Plaintext)
        {
            const auto &plain = value.device_plaintext();
            const auto context_data = context_.crt_context()->get_context_data(plain.meta.parms_id);
            if (context_data == nullptr)
            {
                throw std::runtime_error("Poseidon GPU plaintext has an unknown parms_id");
            }
            if (plain.meta.degree != context_.parameters_literal()->degree() ||
                plain.meta.q_count != context_data->parms().q().size() ||
                plain.meta.p_count != 0 || plain.poly_.degree != plain.meta.degree ||
                plain.poly_.q_count != plain.meta.q_count || plain.poly_.p_count != 0)
            {
                throw std::runtime_error("Poseidon GPU plaintext shape does not match context");
            }
            require_single_full_shard(plain, 1, expected_cuda_device,
                                      "Poseidon GPU plaintext");
            require_full_poly(plain.poly_, 0, plain.meta.degree, plain.meta.q_count, 0,
                              "Poseidon GPU plaintext");
            actual_level = static_cast<int>(context_data->level());
            actual_scale = plain.meta.scale;
            actual_ntt = plain.meta.is_ntt_form;
            actual_components = 1;
        }
        else
        {
            const auto &cipher = value.device_ciphertext();
            const auto context_data =
                context_.crt_context()->get_context_data(cipher.meta.parms_id);
            if (context_data == nullptr)
            {
                throw std::runtime_error("Poseidon GPU ciphertext has an unknown parms_id");
            }
            if (cipher.meta.degree != context_.parameters_literal()->degree() ||
                cipher.meta.q_count != context_data->parms().q().size() ||
                cipher.meta.p_count != 0 ||
                cipher.meta.component_count != cipher.polys_.size())
            {
                throw std::runtime_error("Poseidon GPU ciphertext shape does not match context");
            }
            require_single_full_shard(cipher, cipher.meta.component_count,
                                      expected_cuda_device, "Poseidon GPU ciphertext");
            for (std::size_t component = 0; component < cipher.polys_.size(); ++component)
            {
                require_full_poly(cipher.polys_[component], component, cipher.meta.degree,
                                  cipher.meta.q_count, 0, "Poseidon GPU ciphertext");
            }
            actual_level = static_cast<int>(context_data->level());
            actual_scale = cipher.meta.scale;
            actual_ntt = cipher.meta.is_ntt_form;
            actual_components = static_cast<int>(cipher.meta.component_count);
        }
    }

    if (!(actual_scale > 0.0) || !std::isfinite(actual_scale))
    {
        throw std::runtime_error("Poseidon GPU Api value scale is invalid");
    }
    if (actual_level != expected.level ||
        std::abs(std::log2(actual_scale) - expected.scale_log2) > 1e-6 ||
        actual_ntt != expected.ntt || actual_components != expected.components)
    {
        throw std::runtime_error("Poseidon GPU Api value metadata does not match ValueDesc " +
                                 std::to_string(expected.id));
    }
}

PoseidonGpuApi::DeviceState &PoseidonGpuApi::device_state(
    const fhegpu::Place &place, const char *where)
{
    require_device_place(place, where);
    return device_state(place.index);
}

const PoseidonGpuApi::DeviceState &PoseidonGpuApi::device_state(
    const fhegpu::Place &place, const char *where) const
{
    require_device_place(place, where);
    return device_state(place.index);
}

PoseidonGpuApi::DeviceState &PoseidonGpuApi::device_state(int logical_device_index)
{
    return const_cast<DeviceState &>(
        static_cast<const PoseidonGpuApi &>(*this).device_state(logical_device_index));
}

const PoseidonGpuApi::DeviceState &PoseidonGpuApi::device_state(
    int logical_device_index) const
{
    if (logical_device_index < 0 ||
        logical_device_index >= static_cast<int>(devices_.size()))
    {
        throw std::invalid_argument("Poseidon GPU logical device index is unavailable");
    }
    return *devices_[static_cast<std::size_t>(logical_device_index)];
}

void PoseidonGpuApi::retain_in_flight(
    const std::vector<Value> &values,
    std::vector<std::shared_ptr<void>> resources)
{
    resources.reserve(resources.size() + values.size());
    for (const auto &value : values)
    {
        std::visit(
            [&](const auto &storage) {
                resources.push_back(storage);
            },
            value.storage_);
    }
    std::lock_guard<std::mutex> lock(in_flight_mutex_);
    for (auto &resource : resources)
    {
        in_flight_resources_.push_back(std::move(resource));
    }
}

void PoseidonGpuApi::release_completed_in_flight()
{
    std::lock_guard<std::mutex> lock(in_flight_mutex_);
    in_flight_resources_.clear();
}

void PoseidonGpuApi::synchronize_device(int cuda_device_id) const
{
    gpu::gpu_check_cuda(cudaSetDevice(cuda_device_id), "cudaSetDevice");
    gpu::gpu_check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void PoseidonGpuApi::synchronize_all_devices() const
{
    for (const auto &device : devices_)
    {
        synchronize_device(device->cuda_device_id);
    }
}

std::size_t PoseidonGpuApi::q_count_for_level(int level) const
{
    if (level < 0)
    {
        throw std::invalid_argument("Poseidon GPU key level is negative");
    }
    const auto parms_id = context_.crt_context()->parms_id_map().at(
        static_cast<std::uint32_t>(level));
    const auto context_data = context_.crt_context()->get_context_data(parms_id);
    if (context_data == nullptr)
    {
        throw std::invalid_argument("Poseidon GPU key level is unknown");
    }
    return context_data->parms().q().size();
}

void PoseidonGpuApi::materialize_relin_keys(DeviceState &device,
                                            std::size_t q_count)
{
    if (device.relin_keys_by_q_count.count(q_count) != 0)
    {
        return;
    }
    if (relin_keys_ == nullptr)
    {
        throw std::runtime_error("Poseidon GPU Relinearize requires RelinKeys");
    }

    auto keys = std::make_unique<gpu::GpuRelinKeysData>(
        gpu::GpuUploader::upload_relin_keys(*relin_keys_, device.cuda_device_id, q_count));
    const auto [inserted, ok] =
        device.relin_keys_by_q_count.emplace(q_count, std::move(keys));
    if (!ok)
    {
        throw std::logic_error("Poseidon GPU RelinKeys cache insertion failed");
    }
}

void PoseidonGpuApi::materialize_galois_keys(
    DeviceState &device, std::size_t q_count,
    const std::set<std::uint32_t> &galois_elts)
{
    if (galois_elts.empty())
    {
        return;
    }
    if (galois_keys_ == nullptr)
    {
        throw std::runtime_error("Poseidon GPU Rotate requires GaloisKeys");
    }

    auto combined_galois_elts = galois_elts;
    const auto existing_elements = device.galois_elements_by_q_count.find(q_count);
    if (existing_elements != device.galois_elements_by_q_count.end())
    {
        combined_galois_elts.insert(
            existing_elements->second.begin(), existing_elements->second.end());
        if (combined_galois_elts == existing_elements->second)
        {
            return;
        }
    }

    const std::vector<std::uint32_t> selected_galois_elts(
        combined_galois_elts.begin(), combined_galois_elts.end());
    auto keys = std::make_unique<gpu::GpuGaloisKeysData>(
        gpu::GpuUploader::upload_galois_keys(*galois_keys_, device.cuda_device_id,
                                             q_count, selected_galois_elts));
    device.galois_keys_by_q_count.insert_or_assign(q_count, std::move(keys));
    device.galois_elements_by_q_count.insert_or_assign(
        q_count, std::move(combined_galois_elts));
}

const gpu::GpuRelinKeysData &PoseidonGpuApi::relin_keys_for(
    DeviceState &device, std::size_t q_count)
{
    const auto existing = device.relin_keys_by_q_count.find(q_count);
    if (existing == device.relin_keys_by_q_count.end())
    {
        throw std::runtime_error(
            "Poseidon GPU RelinKeys were not preloaded for the input level");
    }
    return *existing->second;
}

const gpu::GpuGaloisKeysData &PoseidonGpuApi::galois_keys_for(
    DeviceState &device, std::size_t q_count)
{
    const auto existing = device.galois_keys_by_q_count.find(q_count);
    if (existing == device.galois_keys_by_q_count.end())
    {
        throw std::runtime_error(
            "Poseidon GPU GaloisKeys were not preloaded for the input level");
    }
    return *existing->second;
}

} // namespace poseidon::runtime_api
