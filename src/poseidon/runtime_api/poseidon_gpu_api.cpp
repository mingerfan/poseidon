#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "runtime/thread_trace.hpp"

#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_stream_wait_trace.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"
#include "poseidon/runtime_api/communication/cuda_local_transfer.h"
#include "poseidon/runtime_api/communication/gpu_object_copy.h"
#ifdef POSEIDON_RUNTIME_GPU_NCCL
#include "poseidon/runtime_api/communication/nccl_mpi_transport.h"
#endif
#include "poseidon/runtime_api/rotation_key_basis.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
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
namespace
{

constexpr std::size_t kInitialDevicePoolSize = 24ULL << 30;

#ifdef POSEIDON_RUNTIME_GPU_NCCL
void check_mpi_status(int status, const char *what)
{
    if (status != MPI_SUCCESS)
    {
        char message[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        (void)MPI_Error_string(status, message, &length);
        throw std::runtime_error(std::string(what) + ": " +
                                 std::string(message, static_cast<std::size_t>(length)));
    }
}

void require_same_mpi_string(MPI_Comm communicator, int world_size,
                             std::string_view value, const char *what)
{
    if (value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(std::string(what) + " is too large for MPI_Allgather");
    }
    const int length = static_cast<int>(value.size());
    std::vector<int> lengths(static_cast<std::size_t>(world_size));
    check_mpi_status(MPI_Allgather(&length, 1, MPI_INT, lengths.data(), 1, MPI_INT,
                                   communicator),
                     "MPI_Allgather string length");
    for (int remote_length : lengths)
    {
        if (remote_length != length)
        {
            throw std::runtime_error(std::string(what) + " length differs across MPI ranks");
        }
    }
    if (length == 0)
    {
        return;
    }
    std::vector<char> gathered(static_cast<std::size_t>(world_size) *
                               static_cast<std::size_t>(length));
    check_mpi_status(MPI_Allgather(value.data(), length, MPI_CHAR, gathered.data(),
                                   length, MPI_CHAR, communicator),
                     "MPI_Allgather string");
    for (int rank = 0; rank < world_size; ++rank)
    {
        const char *remote = gathered.data() +
                             static_cast<std::size_t>(rank) *
                                 static_cast<std::size_t>(length);
        if (!std::equal(value.begin(), value.end(), remote))
        {
            throw std::runtime_error(std::string(what) + " differs across MPI ranks");
        }
    }
}
#endif

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
            upstream_.get(), kInitialDevicePoolSize);
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
    fhegpu::ThreadTraceLockGuard lock(
        mutex, "gpu.device_memory_pool_registry");
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

void require_host_place(const fhegpu::Place &place, const char *where,
                        int local_rank = 0)
{
    if (place.kind != fhegpu::PlaceKind::Host || place.rank != local_rank ||
        place.index != 0)
    {
        throw std::invalid_argument(std::string(where) + " requires Host(rank=" +
                                    std::to_string(local_rank) + ",index=0)");
    }
}

void require_device_place(const fhegpu::Place &place, const char *where,
                          int local_rank = 0)
{
    if (place.kind != fhegpu::PlaceKind::Device || place.rank != local_rank ||
        place.index < 0)
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

#ifdef POSEIDON_RUNTIME_GPU_NCCL
struct GpuWireHeader
{
    std::uint32_t magic = 0x50474431U; // PGD1
    std::uint32_t version = 1;
    std::uint32_t kind = 0; // 1 plaintext, 2 ciphertext
    std::uint32_t reserved = 0;
    std::uint64_t degree = 0;
    std::uint64_t q_count = 0;
    std::uint64_t p_count = 0;
    std::uint64_t component_count = 0;
    std::uint64_t bytes = 0;
    parms_id_type parms_id{};
    double scale = 1.0;
    std::uint64_t correction_factor = 1;
    std::uint8_t is_ntt_form = 0;
    std::uint8_t padding[7]{};
};

static_assert(std::is_trivially_copyable_v<GpuWireHeader>);

GpuWireHeader make_wire_header(const gpu::GpuPlaintextData &source)
{
    if (source.fields_.empty() || source.fields_.front().device_id < 0)
    {
        throw std::invalid_argument("NCCL plaintext source has no GPU field");
    }
    require_single_full_shard(source, 1, source.fields_.front().device_id,
                               "NCCL plaintext source");
    require_full_poly(source.poly_, 0, source.meta.degree, source.meta.q_count,
                      source.meta.p_count, "NCCL plaintext source");
    GpuWireHeader header;
    header.kind = 1;
    header.degree = source.meta.degree;
    header.q_count = source.meta.q_count;
    header.p_count = source.meta.p_count;
    header.component_count = 1;
    header.bytes = checked_mul_size(source.fields_.front().size(),
                                    sizeof(gpu::GpuWord),
                                    "NCCL plaintext wire size overflow");
    header.parms_id = source.meta.parms_id;
    header.scale = source.meta.scale;
    header.is_ntt_form = source.meta.is_ntt_form ? 1 : 0;
    return header;
}

GpuWireHeader make_wire_header(const gpu::GpuCiphertextData &source)
{
    if (source.fields_.empty() || source.fields_.front().device_id < 0)
    {
        throw std::invalid_argument("NCCL ciphertext source has no GPU field");
    }
    require_single_full_shard(source, source.meta.component_count,
                               source.fields_.front().device_id,
                               "NCCL ciphertext source");
    for (std::size_t component = 0; component < source.polys_.size(); ++component)
    {
        require_full_poly(source.polys_[component], component, source.meta.degree,
                          source.meta.q_count, source.meta.p_count,
                          "NCCL ciphertext source");
    }
    GpuWireHeader header;
    header.kind = 2;
    header.degree = source.meta.degree;
    header.q_count = source.meta.q_count;
    header.p_count = source.meta.p_count;
    header.component_count = source.meta.component_count;
    header.bytes = checked_mul_size(source.fields_.front().size(),
                                    sizeof(gpu::GpuWord),
                                    "NCCL ciphertext wire size overflow");
    header.parms_id = source.meta.parms_id;
    header.scale = source.meta.scale;
    header.correction_factor = source.meta.correction_factor;
    header.is_ntt_form = source.meta.is_ntt_form ? 1 : 0;
    return header;
}

void validate_wire_header(const GpuWireHeader &header,
                          const PoseidonContext &context)
{
    if (header.magic != 0x50474431U || header.version != 1 ||
        (header.kind != 1 && header.kind != 2) || header.degree == 0 ||
        header.q_count == 0 || header.component_count == 0 ||
        header.bytes == 0)
    {
        throw std::runtime_error("NCCL wire header is invalid");
    }
    if (!std::isfinite(header.scale) || header.scale <= 0.0)
    {
        throw std::runtime_error("NCCL wire header scale is invalid");
    }
    const auto context_data = context.crt_context()->get_context_data(header.parms_id);
    if (context_data == nullptr ||
        context.parameters_literal()->degree() != header.degree ||
        context_data->parms().q().size() != header.q_count || header.p_count != 0)
    {
        throw std::runtime_error("NCCL wire header does not match PoseidonContext");
    }
    const std::size_t words = checked_mul_size(
        checked_mul_size(static_cast<std::size_t>(header.degree),
                         static_cast<std::size_t>(header.q_count),
                         "NCCL wire word count overflow"),
        static_cast<std::size_t>(header.component_count),
        "NCCL wire word count overflow");
    if (header.bytes != checked_mul_size(words, sizeof(gpu::GpuWord),
                                         "NCCL wire byte count overflow"))
    {
        throw std::runtime_error("NCCL wire header byte count mismatch");
    }
}

gpu::GpuPlaintextData allocate_wire_plaintext(const GpuWireHeader &header,
                                               int device,
                                               const PoseidonContext &context)
{
    validate_wire_header(header, context);
    if (header.kind != 1)
    {
        throw std::invalid_argument("NCCL wire header is not plaintext");
    }
    auto result = gpu::GpuPlaintextData::allocate_single_device(
        header.degree, header.q_count, device, header.p_count);
    result.meta.parms_id = header.parms_id;
    result.meta.scale = header.scale;
    result.meta.is_ntt_form = header.is_ntt_form != 0;
    return result;
}

gpu::GpuCiphertextData allocate_wire_ciphertext(const GpuWireHeader &header,
                                                 int device,
                                                 const PoseidonContext &context)
{
    validate_wire_header(header, context);
    if (header.kind != 2)
    {
        throw std::invalid_argument("NCCL wire header is not ciphertext");
    }
    auto result = gpu::GpuCiphertextData::allocate_single_device(
        header.degree, header.q_count, header.component_count, device,
        header.p_count);
    result.meta.parms_id = header.parms_id;
    result.meta.scale = header.scale;
    result.meta.correction_factor = header.correction_factor;
    result.meta.is_ntt_form = header.is_ntt_form != 0;
    return result;
}

int mpi_tag_for(MPI_Comm communicator, fhegpu::TransferId transfer_id,
                std::size_t slot)
{
    int *tag_upper_bound = nullptr;
    int attribute_present = 0;
    if (MPI_Comm_get_attr(communicator, MPI_TAG_UB, &tag_upper_bound,
                          &attribute_present) != MPI_SUCCESS ||
        !attribute_present || tag_upper_bound == nullptr || *tag_upper_bound < 4)
    {
        throw std::runtime_error("MPI communicator has no usable MPI_TAG_UB");
    }
    const std::uint64_t bound = static_cast<std::uint64_t>(*tag_upper_bound - 1);
    const std::uint64_t mixed = transfer_id * 1315423911ULL +
                                static_cast<std::uint64_t>(slot) * 2654435761ULL;
    return static_cast<int>(mixed % bound) + 1;
}
#endif

} // namespace

struct PoseidonGpuApi::DeviceState
{
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
        gpu::gpu_stream_wait_event(
            gpu::gpu_execution_stream(), event(), cuda_device_id,
            "gpu.stream_wait.compute_input",
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
#ifdef POSEIDON_RUNTIME_GPU_NCCL
    std::vector<communication::NcclMpiTransport::Request> nccl_requests;
    std::vector<std::shared_ptr<GpuWireHeader>> wire_headers;
    std::vector<MPI_Request> mpi_requests;
#endif
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

gpu::GpuPlaintextData &PoseidonGpuValue::device_plaintext()
{
    auto *value = std::get_if<std::shared_ptr<gpu::GpuPlaintextData>>(&storage_);
    if (value == nullptr || *value == nullptr)
    {
        throw std::invalid_argument("Poseidon GPU Api value is not a Device plaintext");
    }
    return **value;
}

gpu::GpuCiphertextData &PoseidonGpuValue::device_ciphertext()
{
    auto *value = std::get_if<std::shared_ptr<gpu::GpuCiphertextData>>(&storage_);
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
    device_counts_ = {static_cast<int>(cuda_device_ids.size())};
    rank_to_node_ = {0};
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

#ifdef POSEIDON_RUNTIME_GPU_NCCL
PoseidonGpuApi::PoseidonGpuApi(
    std::string context_id, PoseidonContext context, MPI_Comm control_comm,
    std::vector<int> cuda_device_ids, GpuProcessTopology topology,
    std::shared_ptr<const RelinKeys> relin_keys,
    std::shared_ptr<const GaloisKeys> galois_keys,
    std::shared_ptr<const PublicKey> boot_public_key,
    std::shared_ptr<const SecretKey> boot_secret_key)
    : PoseidonGpuApi(std::move(context_id), std::move(context),
                     std::move(cuda_device_ids), std::move(relin_keys),
                     std::move(galois_keys), std::move(boot_public_key),
                     std::move(boot_secret_key))
{
    if (topology.device_counts.empty())
    {
        throw std::invalid_argument(
            "Poseidon GPU distributed topology has no device counts");
    }
    if (!topology.rank_to_node.empty() &&
        topology.rank_to_node.size() != topology.device_counts.size())
    {
        throw std::invalid_argument(
            "Poseidon GPU distributed topology rank_to_node length mismatch");
    }

    std::vector<int> local_cuda_device_ids;
    local_cuda_device_ids.reserve(devices_.size());
    for (const auto &device : devices_)
    {
        local_cuda_device_ids.push_back(device->cuda_device_id);
    }
    nccl_transport_ = std::make_unique<communication::NcclMpiTransport>(
        communication::NcclProcessTopology{
            control_comm, std::move(local_cuda_device_ids),
            topology.device_counts});
    mpi_rank_ = nccl_transport_->mpi_rank();
    mpi_world_size_ = nccl_transport_->mpi_world_size();
    device_counts_ = std::move(topology.device_counts);
    rank_to_node_ = std::move(topology.rank_to_node);
    if (rank_to_node_.empty())
    {
        rank_to_node_.resize(device_counts_.size());
        for (std::size_t index = 0; index < rank_to_node_.size(); ++index)
        {
            rank_to_node_[index] = static_cast<int>(index);
        }
    }
}
#endif

PoseidonGpuApi::~PoseidonGpuApi()
{
    for (const auto &device : devices_)
    {
        (void)cudaSetDevice(device->cuda_device_id);
        (void)cudaDeviceSynchronize();
    }
}

std::string PoseidonGpuApi::name() const
{
    return "PoseidonGpuApi";
}

int PoseidonGpuApi::mpi_rank() const noexcept
{
    return mpi_rank_;
}

int PoseidonGpuApi::mpi_world_size() const noexcept
{
    return mpi_world_size_;
}

int PoseidonGpuApi::local_device_count() const noexcept
{
    return static_cast<int>(devices_.size());
}

int PoseidonGpuApi::cuda_device_id(int logical_device_index) const
{
    return device_state(logical_device_index).cuda_device_id;
}

int PoseidonGpuApi::nccl_rank(int logical_device_index) const
{
#ifdef POSEIDON_RUNTIME_GPU_NCCL
    if (nccl_transport_ != nullptr)
    {
        return nccl_transport_->nccl_rank(logical_device_index);
    }
#endif
    if (logical_device_index < 0 ||
        logical_device_index >= local_device_count())
    {
        throw std::out_of_range("Poseidon GPU logical device index is out of range");
    }
    return logical_device_index;
}

#ifdef POSEIDON_RUNTIME_GPU_NCCL
int PoseidonGpuApi::nccl_rank_for_place(const fhegpu::Place &place) const
{
    if (place.kind != fhegpu::PlaceKind::Device || place.rank < 0 ||
        place.rank >= mpi_world_size_ || place.index < 0 ||
        static_cast<std::size_t>(place.rank) >= device_counts_.size() ||
        place.index >= device_counts_[static_cast<std::size_t>(place.rank)])
    {
        throw std::invalid_argument("Poseidon GPU Place has no NCCL rank mapping");
    }
    if (place.rank == mpi_rank_)
    {
        return nccl_transport_->nccl_rank(place.index);
    }
    int offset = 0;
    for (int rank = 0; rank < place.rank; ++rank)
    {
        if (device_counts_[static_cast<std::size_t>(rank)] >
            std::numeric_limits<int>::max() - offset)
        {
            throw std::overflow_error("Poseidon GPU NCCL rank offset overflow");
        }
        offset += device_counts_[static_cast<std::size_t>(rank)];
    }
    return offset + place.index;
}
#endif

PoseidonGpuApi::Value PoseidonGpuApi::encode_plaintext(
    const fhegpu::ValueDesc &output_desc, const std::vector<double> &slots)
{
    require_host_place(output_desc.place, "Poseidon GPU Encode", mpi_rank_);
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
        require_host_place(op.place, "Poseidon GPU decrypt_reencrypt Boot", mpi_rank_);
        const auto attrs = std::get<fhegpu::BootAttrs>(op.attrs);
        if (attrs.implementation != fhegpu::BootImplementation::DecryptReencrypt)
        {
            throw std::runtime_error("Poseidon GPU native Boot is not implemented");
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
#ifdef POSEIDON_RUNTIME_GPU_NCCL
    const bool remote_source = !action.sources.empty() &&
                               action.sources.front().rank != mpi_rank_;
    const bool remote_destination = std::any_of(
        action.destinations.begin(), action.destinations.end(),
        [this](const fhegpu::Place &place) { return place.rank != mpi_rank_; });
    if (nccl_transport_ != nullptr && (remote_source || remote_destination))
    {
        return communicate_distributed(action, local_inputs);
    }
#endif
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
            require_host_place(source_place, "Poseidon GPU communication source",
                               mpi_rank_);
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
            require_host_place(destination,
                               "Poseidon GPU communication destination", mpi_rank_);
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

#ifdef POSEIDON_RUNTIME_GPU_NCCL
PoseidonGpuApi::CommHandle PoseidonGpuApi::communicate_distributed(
    const fhegpu::CommAction &action, const std::vector<Value> &local_inputs)
{
    if (nccl_transport_ == nullptr)
    {
        throw std::logic_error("distributed GPU communication has no NCCL transport");
    }
    if (action.inputs.size() != 1 || action.sources.size() != 1 ||
        action.outputs.size() != action.destinations.size() ||
        action.outputs.size() != action.output_types.size())
    {
        throw std::invalid_argument("distributed GPU communication mapping is invalid");
    }
    if (action.kind == fhegpu::CommKind::Transfer && action.outputs.size() != 1)
    {
        throw std::invalid_argument("distributed GPU Transfer requires one output");
    }
    if (action.kind == fhegpu::CommKind::Replicate && action.outputs.size() < 2)
    {
        throw std::invalid_argument("distributed GPU Replicate requires at least two outputs");
    }
    if (action.kind != fhegpu::CommKind::Transfer &&
        action.kind != fhegpu::CommKind::Replicate)
    {
        throw std::invalid_argument("distributed GPU communication kind is unknown");
    }

    const auto &source_place = action.sources.front();
    const bool source_local = source_place.rank == mpi_rank_;
    if (local_inputs.size() != (source_local ? 1U : 0U))
    {
        throw std::invalid_argument(
            "distributed GPU communication local input role is invalid");
    }
    if (source_place.rank < 0 || source_place.rank >= mpi_world_size_)
    {
        throw std::invalid_argument("distributed GPU source rank is out of range");
    }
    if (source_place.kind != fhegpu::PlaceKind::Device)
    {
        throw std::invalid_argument(
            "distributed GPU communication currently requires a Device source");
    }
    if (source_place.index < 0 ||
        source_place.index >= device_counts_[static_cast<std::size_t>(source_place.rank)])
    {
        throw std::invalid_argument("distributed GPU source device index is out of range");
    }

    for (std::size_t slot = 0; slot < action.destinations.size(); ++slot)
    {
        const auto &destination = action.destinations[slot];
        if (destination.kind != fhegpu::PlaceKind::Device ||
            destination.rank < 0 || destination.rank >= mpi_world_size_ ||
            destination.index < 0 ||
            destination.index >=
                device_counts_[static_cast<std::size_t>(destination.rank)])
        {
            throw std::invalid_argument(
                "distributed GPU destination must be a valid Device Place");
        }
        if (destination == source_place)
        {
            throw std::invalid_argument(
                "distributed GPU communication destination equals source Place");
        }
        if (std::find(action.destinations.begin(), action.destinations.begin() +
                          static_cast<std::ptrdiff_t>(slot),
                      destination) !=
            action.destinations.begin() + static_cast<std::ptrdiff_t>(slot))
        {
            throw std::invalid_argument(
                "distributed GPU communication destination is duplicated");
        }
    }

    const Value *input = source_local ? &local_inputs.front() : nullptr;
    if (source_local)
    {
        if (input->place_kind() != fhegpu::PlaceKind::Device ||
            std::any_of(action.output_types.begin(), action.output_types.end(),
                        [input](fhegpu::ValueKind kind) {
                            return kind != input->kind();
                        }))
        {
            throw std::invalid_argument(
                "distributed GPU source value kind/place is invalid");
        }
        const auto &source_device =
            device_state(source_place, "distributed GPU source");
        if (value_cuda_device_id(*input, "distributed GPU source value") !=
            source_device.cuda_device_id)
        {
            throw std::invalid_argument(
                "distributed GPU source value is on the wrong CUDA device");
        }
        retain_in_flight(local_inputs);
    }

    CommHandle handle;
    handle.state_ = std::make_unique<CommHandle::State>();
    auto &state = *handle.state_;
    state.outputs.resize(action.outputs.size());
    state.requests.resize(action.outputs.size());
    state.wire_headers.reserve(action.outputs.size());
    state.mpi_requests.reserve(action.outputs.size());

    bool participates_in_nccl = false;
    if (source_local)
    {
        participates_in_nccl = std::any_of(
            action.destinations.begin(), action.destinations.end(),
            [this](const fhegpu::Place &place) { return place.rank != mpi_rank_; });
    }
    else
    {
        participates_in_nccl = std::any_of(
            action.destinations.begin(), action.destinations.end(),
            [this](const fhegpu::Place &place) { return place.rank == mpi_rank_; });
    }
    if (participates_in_nccl)
    {
        nccl_transport_->group_start();
    }

    try
    {
        for (std::size_t slot = 0; slot < action.destinations.size(); ++slot)
        {
            const auto &destination = action.destinations[slot];
            if (source_local && destination.rank == mpi_rank_)
            {
                const auto requested_route = cuda_transfer_route(action.hint);
                const auto &source_device =
                    device_state(source_place, "distributed GPU source");
                const auto &destination_device =
                    device_state(destination, "distributed GPU destination");
                const cudaEvent_t source_ready =
                    input->ready_ != nullptr ? input->ready_->event() : nullptr;
                if (input->kind() == fhegpu::ValueKind::Plaintext)
                {
                    gpu::GpuPlaintextData output;
                    const auto copies = communication::prepare_full_object_copy(
                        input->device_plaintext(), output,
                        destination_device.cuda_device_id);
                    state.requests[slot].emplace(cuda_transfer_->copy_async(
                        copies.front(), requested_route, source_ready));
                    state.outputs[slot].emplace(
                        Value::from_device_plaintext(std::move(output)));
                }
                else
                {
                    gpu::GpuCiphertextData output;
                    const auto copies = communication::prepare_full_object_copy(
                        input->device_ciphertext(), output,
                        destination_device.cuda_device_id);
                    state.requests[slot].emplace(cuda_transfer_->copy_async(
                        copies.front(), requested_route, source_ready));
                    state.outputs[slot].emplace(
                        Value::from_device_ciphertext(std::move(output)));
                }
                static_cast<void>(source_device);
                continue;
            }
            if (!source_local && destination.rank != mpi_rank_)
            {
                continue;
            }

            const int source_nccl_rank = nccl_rank_for_place(source_place);
            const int destination_nccl_rank = nccl_rank_for_place(destination);
            const int tag = mpi_tag_for(
                nccl_transport_->control_comm(), action.id, slot);

            if (source_local)
            {
                GpuWireHeader header = input->kind() == fhegpu::ValueKind::Plaintext
                                           ? make_wire_header(input->device_plaintext())
                                           : make_wire_header(input->device_ciphertext());
                if (header.kind !=
                    (action.output_types[slot] == fhegpu::ValueKind::Plaintext ? 1U : 2U))
                {
                    throw std::invalid_argument(
                        "distributed GPU output kind does not match source value");
                }
                auto header_storage = std::make_shared<GpuWireHeader>(header);
                MPI_Request header_request = MPI_REQUEST_NULL;
                check_mpi_status(
                    MPI_Isend(header_storage.get(), static_cast<int>(sizeof(*header_storage)),
                              MPI_BYTE, destination.rank, tag,
                              nccl_transport_->control_comm(), &header_request),
                    "MPI_Isend GPU wire header");
                state.wire_headers.push_back(std::move(header_storage));
                state.mpi_requests.push_back(header_request);

                const void *source_buffer = input->kind() == fhegpu::ValueKind::Plaintext
                                                 ? static_cast<const void *>(
                                                       input->device_plaintext().fields_.front().data())
                                                 : static_cast<const void *>(
                                                       input->device_ciphertext().fields_.front().data());
                const std::size_t bytes = header.bytes;
                state.nccl_requests.push_back(nccl_transport_->send_async(
                    source_place.index, destination_nccl_rank, source_buffer, bytes,
                    input->ready_ != nullptr ? input->ready_->event() : nullptr));
            }
            else
            {
                GpuWireHeader header{};
                check_mpi_status(
                    MPI_Recv(&header, static_cast<int>(sizeof(header)), MPI_BYTE,
                             source_place.rank, tag, nccl_transport_->control_comm(),
                             MPI_STATUS_IGNORE),
                    "MPI_Recv GPU wire header");
                const std::uint32_t expected_kind =
                    action.output_types[slot] == fhegpu::ValueKind::Plaintext ? 1U : 2U;
                if (header.kind != expected_kind)
                {
                    throw std::runtime_error(
                        "distributed GPU wire header kind does not match output descriptor");
                }
                const int destination_cuda_device =
                    device_state(destination, "distributed GPU destination")
                        .cuda_device_id;
                if (header.kind == 1)
                {
                    auto output = allocate_wire_plaintext(
                        header, destination_cuda_device, context_);
                    state.outputs[slot].emplace(
                        Value::from_device_plaintext(std::move(output)));
                    auto &device_output = state.outputs[slot]->device_plaintext();
                    state.nccl_requests.push_back(nccl_transport_->recv_async(
                        destination.index, source_nccl_rank,
                        device_output.fields_.front().data(), header.bytes));
                }
                else
                {
                    auto output = allocate_wire_ciphertext(
                        header, destination_cuda_device, context_);
                    state.outputs[slot].emplace(
                        Value::from_device_ciphertext(std::move(output)));
                    auto &device_output = state.outputs[slot]->device_ciphertext();
                    state.nccl_requests.push_back(nccl_transport_->recv_async(
                        destination.index, source_nccl_rank,
                        device_output.fields_.front().data(), header.bytes));
                }
            }
        }
        if (participates_in_nccl)
        {
            nccl_transport_->group_end();
            for (auto &request : state.nccl_requests)
            {
                nccl_transport_->record_event(request);
            }
        }
    }
    catch (...)
    {
        if (participates_in_nccl)
        {
            try
            {
                nccl_transport_->group_end();
            }
            catch (...)
            {}
        }
        throw;
    }
    return handle;
}
#endif

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
            continue;
        }
        request->wait();
    }
#ifdef POSEIDON_RUNTIME_GPU_NCCL
    if (nccl_transport_ != nullptr)
    {
        for (auto &request : state.nccl_requests)
        {
            nccl_transport_->wait(request);
        }
        if (!state.mpi_requests.empty())
        {
            check_mpi_status(
                MPI_Waitall(static_cast<int>(state.mpi_requests.size()),
                            state.mpi_requests.data(), MPI_STATUSES_IGNORE),
                "MPI_Waitall GPU wire headers");
        }
    }
#endif
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
            continue;
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
#ifdef POSEIDON_RUNTIME_GPU_NCCL
    state.nccl_requests.clear();
    state.mpi_requests.clear();
    state.wire_headers.clear();
#endif
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
            return;
        }
        const int value_device = value_cuda_device_id(value, "Poseidon GPU synchronize");
        const auto configured =
            std::find_if(devices_.begin(), devices_.end(), [value_device](const auto &device) {
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

void PoseidonGpuApi::preflight(std::string_view plan_source_sha256,
                               bool skip_artifact_digest_checks,
                               const fhegpu::TargetConfig &target,
                               const fhegpu::OperatorSpec &operator_spec,
                               const fhegpu::PlanRequirements &requirements)
{
    if (plan_source_sha256.size() != 71 || plan_source_sha256.substr(0, 7) != "sha256:")
    {
        throw std::invalid_argument("invalid RuntimePlan source SHA-256");
    }
    if (target.target_id != "poseidon-ckks-gpu" || target.capability_version != 1)
    {
        throw std::invalid_argument("Poseidon GPU Api target is unsupported");
    }
    if (target.world_size != mpi_world_size_)
    {
        throw std::invalid_argument(
            "Poseidon GPU Api target world size does not match MPI world size");
    }
    if (target.device_counts.size() !=
            static_cast<std::size_t>(mpi_world_size_) ||
        target.device_counts[static_cast<std::size_t>(mpi_rank_)] !=
            local_device_count())
    {
        throw std::invalid_argument(
            "Poseidon GPU Api target device count does not match this MPI rank");
    }

#ifdef POSEIDON_RUNTIME_GPU_NCCL
    if (nccl_transport_ != nullptr)
    {
        const int local_skip = skip_artifact_digest_checks ? 1 : 0;
        std::vector<int> skip_values(static_cast<std::size_t>(mpi_world_size_));
        check_mpi_status(MPI_Allgather(&local_skip, 1, MPI_INT, skip_values.data(),
                                       1, MPI_INT, nccl_transport_->control_comm()),
                         "MPI_Allgather skip_artifact_digest_checks");
        if (!std::all_of(skip_values.begin(), skip_values.end(),
                         [local_skip](int value) { return value == local_skip; }))
        {
            throw std::runtime_error(
                "Poseidon GPU skip_artifact_digest_checks differs across MPI ranks");
        }
        if (!skip_artifact_digest_checks)
        {
            require_same_mpi_string(nccl_transport_->control_comm(), mpi_world_size_,
                                    plan_source_sha256, "RuntimePlan source digest");
        }
        require_same_mpi_string(nccl_transport_->control_comm(), mpi_world_size_,
                                context_id_, "Poseidon GPU context id");
    }
#else
    static_cast<void>(skip_artifact_digest_checks);
#endif

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
    for (const auto &profile : operator_spec.boot_profiles)
    {
        if (profile.implementation != fhegpu::BootImplementation::DecryptReencrypt)
        {
            throw std::invalid_argument(
                "Poseidon GPU Api supports only decrypt_reencrypt Boot profiles");
        }
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
        const bool boot = boot_encryptor_ != nullptr && boot_decryptor_ != nullptr &&
                          (capability == fhegpu::RequiredCapability::HostCompute ||
                           capability ==
                               fhegpu::RequiredCapability::BootDecryptReencrypt);
        if (!local && !boot)
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
        require_host_place(key.place, "Poseidon GPU SecretKey", mpi_rank_);
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

[[noreturn]] void PoseidonGpuApi::abort_all(int exit_code, const std::string &reason)
{
#ifdef POSEIDON_RUNTIME_GPU_NCCL
    if (nccl_transport_ != nullptr)
    {
        nccl_transport_->abort();
        (void)MPI_Abort(nccl_transport_->control_comm(), exit_code);
    }
#else
    static_cast<void>(exit_code);
#endif
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
        require_host_place(expected.place, "Poseidon GPU Host value", mpi_rank_);
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
    require_device_place(place, where, mpi_rank_);
    return device_state(place.index);
}

const PoseidonGpuApi::DeviceState &PoseidonGpuApi::device_state(
    const fhegpu::Place &place, const char *where) const
{
    require_device_place(place, where, mpi_rank_);
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
    fhegpu::ThreadTraceLockGuard lock(
        in_flight_mutex_, "gpu.in_flight_resources");
    for (auto &resource : resources)
    {
        in_flight_resources_.push_back(std::move(resource));
    }
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
