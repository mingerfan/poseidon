#include "poseidon/mgpu/comm/inter_node_transport.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace poseidon::mgpu
{
namespace
{

const MgpuLogicalDevice &lookup_device(
    const MgpuTopology &topology, int logical_device)
{
    const auto iter = std::find_if(
        topology.devices.begin(), topology.devices.end(),
        [logical_device](const MgpuLogicalDevice &device) {
            return device.logical_device == logical_device;
        });
    if (iter == topology.devices.end())
    {
        std::ostringstream stream;
        stream << "logical device " << logical_device
               << " is not present in topology";
        throw std::invalid_argument(stream.str());
    }
    return *iter;
}

InterNodeCopyEndpoint endpoint_from_device(const MgpuLogicalDevice &device)
{
    return InterNodeCopyEndpoint{
        device.logical_device,
        device.node_id,
        device.local_device,
    };
}

void validate_route_matches_object_copy(
    const MgpuCopyRoute &route, const GpuObjectCopyRequest &object_copy)
{
    if (route.transport != MgpuTransportKind::InterNode)
    {
        throw std::invalid_argument(
            "inter-node transport requires an inter-node copy route");
    }
    if (route.source_id != object_copy.source_id ||
        route.destination_id != object_copy.destination_id)
    {
        throw std::invalid_argument(
            "inter-node route value ids do not match object-copy request");
    }
    if (route.kind != object_copy.kind)
    {
        throw std::invalid_argument(
            "inter-node route value kind does not match object-copy request");
    }
}

void validate_buffer_devices(
    const GpuObjectCopyRequest &object_copy, int source_device,
    int destination_device)
{
    for (const GpuObjectBufferCopy &buffer : object_copy.buffers)
    {
        if (buffer.source_device != source_device ||
            buffer.destination_device != destination_device)
        {
            throw std::invalid_argument(
                "inter-node object-copy buffer devices do not match route devices");
        }
    }
}

}  // namespace

void MissingInterNodeTransportBackend::copy_object(
    const InterNodeObjectCopyRequest &request)
{
    std::ostringstream stream;
    stream << "inter-node communication backend is not configured for device "
           << request.source.logical_device << " -> "
           << request.destination.logical_device;
    throw std::runtime_error(stream.str());
}

InterNodeObjectCopyRequest make_inter_node_object_copy_request(
    const MgpuCopyRoute &route, const GpuObjectCopyRequest &object_copy,
    const MgpuTopology &topology)
{
    validate_route_matches_object_copy(route, object_copy);

    const GpuObjectCopyValidationResult validation =
        validate_full_object_copy_request(object_copy);
    if (!validation.ok())
    {
        throw std::invalid_argument(validation.format_errors());
    }

    const MgpuLogicalDevice &source =
        lookup_device(topology, route.source_device);
    const MgpuLogicalDevice &destination =
        lookup_device(topology, route.destination_device);
    if (source.node_id == destination.node_id)
    {
        throw std::invalid_argument(
            "inter-node transport route endpoints are on the same node");
    }
    validate_buffer_devices(
        object_copy, route.source_device, route.destination_device);

    InterNodeObjectCopyRequest request;
    request.source_id = route.source_id;
    request.destination_id = route.destination_id;
    request.kind = route.kind;
    request.source = endpoint_from_device(source);
    request.destination = endpoint_from_device(destination);
    request.object_copy = object_copy;
    return request;
}

}  // namespace poseidon::mgpu
