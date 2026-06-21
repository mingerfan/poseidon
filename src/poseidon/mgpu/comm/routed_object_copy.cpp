#include "poseidon/mgpu/comm/routed_object_copy.h"

#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

void validate_route_matches_request(
    const MgpuCopyRoute &route, const GpuObjectCopyRequest &request)
{
    if (route.source_id != request.source_id ||
        route.destination_id != request.destination_id)
    {
        throw std::invalid_argument(
            "copy route value ids do not match object-copy request");
    }
    if (route.kind != request.kind)
    {
        throw std::invalid_argument(
            "copy route value kind does not match object-copy request");
    }

    const GpuObjectCopyValidationResult validation =
        validate_full_object_copy_request(request);
    if (!validation.ok())
    {
        throw std::invalid_argument(validation.format_errors());
    }

    for (const GpuObjectBufferCopy &buffer : request.buffers)
    {
        if (buffer.source_device != route.source_device ||
            buffer.destination_device != route.destination_device)
        {
            throw std::invalid_argument(
                "object-copy buffer devices do not match copy route devices");
        }
    }
}

void validate_same_device_route(const MgpuCopyRoute &route)
{
    if (route.transport == MgpuTransportKind::SameDevice &&
        route.source_device != route.destination_device)
    {
        throw std::invalid_argument(
            "same-device copy route has different source and destination devices");
    }
}

}  // namespace

RoutedGpuObjectCopyBackend::RoutedGpuObjectCopyBackend(
    MgpuTopology topology, GpuObjectCopyBackend &local_backend,
    InterNodeTransportBackend &inter_node_backend)
    : topology_(std::move(topology)), local_backend_(local_backend),
      inter_node_backend_(inter_node_backend)
{
}

void RoutedGpuObjectCopyBackend::copy_object(
    const MgpuCopyRoute &route, const GpuObjectCopyRequest &request)
{
    validate_route_matches_request(route, request);
    validate_same_device_route(route);

    if (route.transport == MgpuTransportKind::InterNode)
    {
        inter_node_backend_.copy_object(
            make_inter_node_object_copy_request(route, request, topology_));
        return;
    }

    local_backend_.copy_object(request);
}

}  // namespace poseidon::mgpu
