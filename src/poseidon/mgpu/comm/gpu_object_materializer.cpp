#include "poseidon/mgpu/comm/gpu_object_materializer.h"
#include "poseidon/runtime_api/communication/gpu_object_copy.h"

#include <memory>
#include <stdexcept>
#include <vector>

namespace poseidon::mgpu
{
namespace
{

GpuObjectBufferCopy adapt_buffer_copy(
    const runtime_api::communication::DeviceBufferCopy &copy)
{
    return GpuObjectBufferCopy{
        copy.source,
        copy.destination,
        copy.bytes,
        copy.source_device,
        copy.destination_device,
    };
}

}  // namespace

GpuObjectCopyRequest make_full_object_copy_request(
    ValueId source_id, ValueId destination_id, const gpu::GpuCiphertextData &source,
    gpu::GpuCiphertextData &destination, int destination_device)
{
    GpuObjectCopyRequest request;
    request.source_id = source_id;
    request.destination_id = destination_id;
    request.kind = MgpuValueKind::Ciphertext;
    for (const auto &copy : runtime_api::communication::prepare_full_object_copy(
             source, destination, destination_device))
    {
        request.buffers.push_back(adapt_buffer_copy(copy));
    }
    return request;
}

GpuObjectCopyRequest make_full_object_copy_request(
    ValueId source_id, ValueId destination_id, const gpu::GpuPlaintextData &source,
    gpu::GpuPlaintextData &destination, int destination_device)
{
    GpuObjectCopyRequest request;
    request.source_id = source_id;
    request.destination_id = destination_id;
    request.kind = MgpuValueKind::Plaintext;
    for (const auto &copy : runtime_api::communication::prepare_full_object_copy(
             source, destination, destination_device))
    {
        request.buffers.push_back(adapt_buffer_copy(copy));
    }
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

MaterializedGpuObjectBatchCopy PoseidonGpuObjectCopyMaterializer::materialize_copy_batch(
    const std::vector<GpuCommCopyRequest> &requests)
{
    MaterializedGpuObjectBatchCopy result;
    result.destination_objects.reserve(requests.size());
    result.object_copies.reserve(requests.size());

    for (const GpuCommCopyRequest &request : requests)
    {
        MaterializedGpuObjectCopy materialized = materialize_copy(request);
        result.destination_objects.push_back(std::move(materialized.destination_object));
        result.object_copies.push_back(std::move(materialized.object_copy));
    }
    return result;
}

}  // namespace poseidon::mgpu
