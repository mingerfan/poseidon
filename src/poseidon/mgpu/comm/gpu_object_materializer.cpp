#include "poseidon/mgpu/comm/gpu_object_materializer.h"

#include <memory>
#include <stdexcept>
#include <string>

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
    validate_single_field(source.fields_.size(), "ciphertext");

    destination = gpu::GpuCiphertextData::allocate_single_device(
        source.meta.degree, source.meta.q_count, source.meta.component_count,
        destination_device, source.meta.p_count);
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
    validate_single_field(source.fields_.size(), "plaintext");

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
