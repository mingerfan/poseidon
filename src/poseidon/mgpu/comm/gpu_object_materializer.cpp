#include "poseidon/mgpu/comm/gpu_object_materializer.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace poseidon::mgpu
{
namespace
{

void validate_single_field(
    const std::size_t field_count, const char *type_name)
{
    if (field_count != 1)
    {
        throw std::invalid_argument(
            std::string("V1 ") + type_name +
            " object copy requires exactly one full-object field");
    }
}

void validate_full_shard(
    const gpu::GpuRNSPoly &poly, const gpu::GpuPolyShard &shard, const char *type_name)
{
    if (shard.field_index != 0)
    {
        throw std::invalid_argument(
            std::string("V1 ") + type_name + " object copy requires field_index 0");
    }
    if (shard.limb_begin != 0 || shard.limb_count != poly.q_count + poly.p_count)
    {
        throw std::invalid_argument(
            std::string("V1 ") + type_name + " object copy requires a full limb shard");
    }
    if (shard.coeff_begin != 0 || shard.coeff_count != poly.degree)
    {
        throw std::invalid_argument(
            std::string("V1 ") + type_name +
            " object copy requires a full coefficient shard");
    }
}

void validate_ciphertext_full_object(const gpu::GpuCiphertextData &source)
{
    validate_single_field(source.fields_.size(), "ciphertext");
    if (source.empty())
    {
        throw std::invalid_argument("V1 ciphertext object copy requires a non-empty object");
    }
    if (source.meta.component_count != source.polys_.size())
    {
        throw std::invalid_argument("V1 ciphertext object copy component metadata mismatch");
    }

    for (const auto &poly : source.polys_)
    {
        if (poly.shards.size() != 1)
        {
            throw std::invalid_argument(
                "V1 ciphertext object copy requires exactly one full shard per component");
        }
        validate_full_shard(poly, poly.shards[0], "ciphertext");
    }
}

void validate_plaintext_full_object(const gpu::GpuPlaintextData &source)
{
    validate_single_field(source.fields_.size(), "plaintext");
    if (source.empty())
    {
        throw std::invalid_argument("V1 plaintext object copy requires a non-empty object");
    }
    if (source.poly_.shards.size() != 1)
    {
        throw std::invalid_argument(
            "V1 plaintext object copy requires exactly one full shard");
    }
    validate_full_shard(source.poly_, source.poly_.shards[0], "plaintext");
}

void append_buffer_copy(
    GpuObjectCopyRequest &request, const gpu::GpuFieldData &source,
    gpu::GpuFieldData &destination)
{
    request.buffers.push_back(GpuObjectBufferCopy{
        source.data(),
        destination.data(),
        source.size() * sizeof(gpu::GpuWord),
        source.device_id,
        destination.device_id,
    });
}

}  // namespace

GpuObjectCopyRequest make_full_object_copy_request(
    ValueId source_id, ValueId destination_id, const gpu::GpuCiphertextData &source,
    gpu::GpuCiphertextData &destination, int destination_device)
{
    validate_ciphertext_full_object(source);

    const std::vector<gpu::GpuPolyShard> shard_template{ source.polys_[0].shards[0] };
    destination = gpu::GpuCiphertextData::allocate_single_device_sharded(
        source.meta.degree, source.meta.q_count, source.meta.component_count,
        destination_device, shard_template, source.meta.p_count);
    destination.meta = source.meta;

    GpuObjectCopyRequest request;
    request.source_id = source_id;
    request.destination_id = destination_id;
    request.kind = MgpuValueKind::Ciphertext;
    append_buffer_copy(request, source.fields_[0], destination.fields_[0]);
    return request;
}

GpuObjectCopyRequest make_full_object_copy_request(
    ValueId source_id, ValueId destination_id, const gpu::GpuPlaintextData &source,
    gpu::GpuPlaintextData &destination, int destination_device)
{
    validate_plaintext_full_object(source);

    destination = gpu::GpuPlaintextData::allocate_single_device(
        source.meta.degree, source.meta.q_count, destination_device, source.meta.p_count);
    destination.meta = source.meta;

    GpuObjectCopyRequest request;
    request.source_id = source_id;
    request.destination_id = destination_id;
    request.kind = MgpuValueKind::Plaintext;
    append_buffer_copy(request, source.fields_[0], destination.fields_[0]);
    return request;
}

MaterializedGpuObjectCopy PoseidonGpuObjectCopyMaterializer::materialize_copy(
    const GpuCommCopyRequest &request)
{
    if (request.source_object == nullptr)
    {
        throw std::invalid_argument("GPU object copy source object is null");
    }

    MaterializedGpuObjectCopy result;
    if (request.kind == MgpuValueKind::Ciphertext)
    {
        auto source = std::static_pointer_cast<gpu::GpuCiphertextData>(request.source_object);
        auto destination = std::make_shared<gpu::GpuCiphertextData>();
        result.object_copy = make_full_object_copy_request(
            request.source_id, request.destination_id, *source, *destination,
            request.destination_device);
        result.destination_object = destination;
        return result;
    }

    auto source = std::static_pointer_cast<gpu::GpuPlaintextData>(request.source_object);
    auto destination = std::make_shared<gpu::GpuPlaintextData>();
    result.object_copy = make_full_object_copy_request(
        request.source_id, request.destination_id, *source, *destination,
        request.destination_device);
    result.destination_object = destination;
    return result;
}

}  // namespace poseidon::mgpu
