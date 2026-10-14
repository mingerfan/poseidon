#include "poseidon/runtime_api/communication/gpu_object_copy.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace poseidon::runtime_api::communication
{
namespace
{

void validate_single_field(std::size_t field_count, const char *type_name)
{
    if (field_count != 1)
    {
        throw std::invalid_argument(std::string(type_name) +
                                    " copy requires exactly one full-object field");
    }
}

std::size_t checked_mul(std::size_t left, std::size_t right, const char *what)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error(what);
    }
    return left * right;
}

void validate_full_shard(const gpu::GpuRNSPoly &poly, const gpu::GpuPolyShard &shard,
                         std::size_t expected_offset, const char *type_name)
{
    if (shard.field_index != 0 || shard.field_offset != expected_offset)
    {
        throw std::invalid_argument(std::string(type_name) + " copy field layout is invalid");
    }
    if (shard.limb_begin != 0 || shard.limb_count != poly.q_count + poly.p_count)
    {
        throw std::invalid_argument(std::string(type_name) +
                                    " copy requires a full limb shard");
    }
    if (shard.coeff_begin != 0 || shard.coeff_count != poly.degree)
    {
        throw std::invalid_argument(std::string(type_name) +
                                    " copy requires a full coefficient shard");
    }
}

void validate_full_object(const gpu::GpuCiphertextData &source)
{
    validate_single_field(source.fields_.size(), "ciphertext");
    if (source.empty())
    {
        throw std::invalid_argument("ciphertext copy requires a non-empty object");
    }
    if (source.meta.degree == 0 || source.meta.q_count + source.meta.p_count == 0 ||
        source.meta.component_count != source.polys_.size())
    {
        throw std::invalid_argument("ciphertext copy component metadata mismatch");
    }
    const std::size_t component_words = checked_mul(
        source.meta.degree, source.meta.q_count + source.meta.p_count,
        "ciphertext copy component size overflow");
    if (source.fields_.front().size() !=
        checked_mul(component_words, source.meta.component_count,
                    "ciphertext copy field size overflow"))
    {
        throw std::invalid_argument("ciphertext copy field size does not match metadata");
    }
    for (std::size_t component = 0; component < source.polys_.size(); ++component)
    {
        const auto &poly = source.polys_[component];
        if (poly.poly_id != component || poly.degree != source.meta.degree ||
            poly.q_count != source.meta.q_count || poly.p_count != source.meta.p_count)
        {
            throw std::invalid_argument("ciphertext copy polynomial metadata mismatch");
        }
        if (poly.shards.size() != 1)
        {
            throw std::invalid_argument(
                "ciphertext copy requires exactly one full shard per component");
        }
        validate_full_shard(poly, poly.shards.front(),
                            checked_mul(component, component_words,
                                        "ciphertext copy field offset overflow"),
                            "ciphertext");
    }
}

void validate_full_object(const gpu::GpuPlaintextData &source)
{
    validate_single_field(source.fields_.size(), "plaintext");
    if (source.empty())
    {
        throw std::invalid_argument("plaintext copy requires a non-empty object");
    }
    if (source.meta.degree == 0 || source.meta.q_count + source.meta.p_count == 0 ||
        source.poly_.degree != source.meta.degree ||
        source.poly_.q_count != source.meta.q_count ||
        source.poly_.p_count != source.meta.p_count || source.poly_.poly_id != 0)
    {
        throw std::invalid_argument("plaintext copy polynomial metadata mismatch");
    }
    if (source.fields_.front().size() !=
        checked_mul(source.meta.degree, source.meta.q_count + source.meta.p_count,
                    "plaintext copy field size overflow"))
    {
        throw std::invalid_argument("plaintext copy field size does not match metadata");
    }
    if (source.poly_.shards.size() != 1)
    {
        throw std::invalid_argument("plaintext copy requires exactly one full shard");
    }
    validate_full_shard(source.poly_, source.poly_.shards.front(), 0, "plaintext");
}

DeviceBufferCopy make_buffer_copy(const gpu::GpuFieldData &source,
                                  gpu::GpuFieldData &destination)
{
    if (source.device_id < 0 || destination.device_id < 0 ||
        source.buffer.device_id() != source.device_id ||
        destination.buffer.device_id() != destination.device_id)
    {
        throw std::invalid_argument("GPU object copy field device metadata is invalid");
    }
    if (source.size() != destination.size() || source.size() == 0)
    {
        throw std::invalid_argument("GPU object copy field sizes do not match");
    }
    return DeviceBufferCopy{
        source.data(),
        destination.data(),
        checked_mul(source.size(), sizeof(gpu::GpuWord),
                    "GPU object copy byte size overflow"),
        source.device_id,
        destination.device_id,
    };
}

} // namespace

std::vector<DeviceBufferCopy> prepare_full_object_copy(
    const gpu::GpuCiphertextData &source, gpu::GpuCiphertextData &destination,
    int destination_device)
{
    validate_full_object(source);
    if (destination_device < 0)
    {
        throw std::invalid_argument("GPU object copy destination device is negative");
    }

    const std::vector<gpu::GpuPolyShard> shard_template{source.polys_.front().shards.front()};
    destination = gpu::GpuCiphertextData::allocate_single_device_sharded(
        source.meta.degree, source.meta.q_count, source.meta.component_count,
        destination_device, shard_template, source.meta.p_count);
    destination.meta = source.meta;

    return {make_buffer_copy(source.fields_.front(), destination.fields_.front())};
}

std::vector<DeviceBufferCopy> prepare_full_object_copy(
    const gpu::GpuPlaintextData &source, gpu::GpuPlaintextData &destination,
    int destination_device)
{
    validate_full_object(source);
    if (destination_device < 0)
    {
        throw std::invalid_argument("GPU object copy destination device is negative");
    }

    destination = gpu::GpuPlaintextData::allocate_single_device(
        source.meta.degree, source.meta.q_count, destination_device, source.meta.p_count);
    destination.meta = source.meta;

    return {make_buffer_copy(source.fields_.front(), destination.fields_.front())};
}

} // namespace poseidon::runtime_api::communication
