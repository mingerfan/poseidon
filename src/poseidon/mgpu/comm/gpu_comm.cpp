#include "poseidon/mgpu/comm/gpu_comm.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

void add_error(GpuObjectCopyValidationResult &result, const std::string &message)
{
    result.errors.push_back(message);
}

std::pair<ValueId, ValueId> route_key(ValueId source_id, ValueId destination_id)
{
    return std::make_pair(source_id, destination_id);
}

std::string format_route_key(ValueId source_id, ValueId destination_id)
{
    std::ostringstream stream;
    stream << "%" << source_id << " -> %" << destination_id;
    return stream.str();
}

const MgpuLogicalDevice &lookup_device(
    const MgpuTopology &topology, int logical_device, const char *role)
{
    const auto iter = std::find_if(
        topology.devices.begin(), topology.devices.end(),
        [logical_device](const MgpuLogicalDevice &device) {
            return device.logical_device == logical_device;
        });
    if (iter == topology.devices.end())
    {
        std::ostringstream stream;
        stream << "copy route " << role << " device " << logical_device
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

void validate_request_matches_route(
    const MgpuCopyRoute &route, const GpuCommCopyRequest &request)
{
    if (route.kind != request.kind)
    {
        throw std::invalid_argument(
            "planned communication route kind does not match copy request");
    }
    if (route.source_device != request.source_device ||
        route.destination_device != request.destination_device)
    {
        throw std::invalid_argument(
            "planned communication route devices do not match copy request");
    }
}

void validate_request_has_source_object(const GpuCommCopyRequest &request)
{
    if (request.source_object != nullptr)
    {
        return;
    }

    throw std::invalid_argument(
        "planned materialized GPU copy source object is null");
}

void validate_materialized_destination_object(
    const std::shared_ptr<void> &destination_object)
{
    if (destination_object != nullptr)
    {
        return;
    }

    throw std::invalid_argument("materialized GPU copy destination object is null");
}

void validate_route_matches_object_copy(
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

void validate_route_matches_topology(
    const MgpuTopology &topology, const MgpuCopyRoute &route)
{
    const MgpuLogicalDevice &source =
        lookup_device(topology, route.source_device, "source");
    const MgpuLogicalDevice &destination =
        lookup_device(topology, route.destination_device, "destination");

    switch (route.transport)
    {
    case MgpuTransportKind::SameDevice:
        if (source.logical_device != destination.logical_device)
        {
            throw std::invalid_argument(
                "same-device copy route endpoints are different logical devices");
        }
        return;
    case MgpuTransportKind::CudaPeer:
        if (source.logical_device == destination.logical_device)
        {
            throw std::invalid_argument(
                "CUDA peer copy route uses the same source and destination device");
        }
        if (source.node_id != destination.node_id)
        {
            throw std::invalid_argument(
                "CUDA peer copy route endpoints are on different nodes");
        }
        return;
    case MgpuTransportKind::InterNode:
        if (source.node_id == destination.node_id)
        {
            throw std::invalid_argument(
                "inter-node copy route endpoints are on the same node");
        }
        return;
    }
}

InterNodeObjectCopyRequest make_inter_node_request(
    const MgpuCopyRoute &route, const GpuObjectCopyRequest &object_copy,
    const MgpuTopology &topology)
{
    const MgpuLogicalDevice &source =
        lookup_device(topology, route.source_device, "source");
    const MgpuLogicalDevice &destination =
        lookup_device(topology, route.destination_device, "destination");

    InterNodeObjectCopyRequest request;
    request.source_id = route.source_id;
    request.destination_id = route.destination_id;
    request.kind = route.kind;
    request.source = endpoint_from_device(source);
    request.destination = endpoint_from_device(destination);
    request.object_copy = object_copy;
    return request;
}

void copy_object_for_route(
    const MgpuTopology &topology,
    GpuObjectCopyBackend &local_backend,
    InterNodeTransportBackend &inter_node_backend,
    const MgpuCopyRoute &route,
    const GpuObjectCopyRequest &request)
{
    validate_route_matches_object_copy(route, request);
    validate_route_matches_topology(topology, route);

    if (route.transport == MgpuTransportKind::InterNode)
    {
        inter_node_backend.copy_object(
            make_inter_node_request(route, request, topology));
        return;
    }

    local_backend.copy_object(request);
}

}  // namespace

std::string GpuObjectCopyValidationResult::format_errors() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < errors.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << errors[i];
    }
    return stream.str();
}

GpuObjectCopyValidationResult validate_full_object_copy_request(
    const GpuObjectCopyRequest &request)
{
    GpuObjectCopyValidationResult result;

    if (request.source_id == 0)
    {
        add_error(result, "source value id 0 is reserved");
    }
    if (request.destination_id == 0)
    {
        add_error(result, "destination value id 0 is reserved");
    }

    if (request.buffers.size() != 1)
    {
        std::ostringstream stream;
        stream << "V1 full-object copy requires exactly one buffer, got "
               << request.buffers.size();
        add_error(result, stream.str());
        return result;
    }

    const GpuObjectBufferCopy &buffer = request.buffers[0];
    if (buffer.bytes == 0)
    {
        add_error(result, "object copy buffer must be non-empty");
    }
    if (buffer.source == nullptr)
    {
        add_error(result, "object copy source pointer is null");
    }
    if (buffer.destination == nullptr)
    {
        add_error(result, "object copy destination pointer is null");
    }
    if (buffer.source_device < 0)
    {
        add_error(result, "object copy source device must be non-negative");
    }
    if (buffer.destination_device < 0)
    {
        add_error(result, "object copy destination device must be non-negative");
    }

    return result;
}

std::vector<std::shared_ptr<void>> GpuComm::copy_batch(
    const std::vector<GpuCommCopyRequest> &requests)
{
    std::vector<std::shared_ptr<void>> results;
    results.reserve(requests.size());
    for (const GpuCommCopyRequest &request : requests)
    {
        results.push_back(copy(request));
    }
    return results;
}

std::shared_ptr<void> SameDeviceGpuComm::copy(const GpuCommCopyRequest &request)
{
    if (request.source_device != request.destination_device)
    {
        std::ostringstream stream;
        stream << "cross-device copy %" << request.source_id << " from device "
               << request.source_device << " to device " << request.destination_device
               << " requires a multi-GPU communication backend";
        throw std::runtime_error(stream.str());
    }
    return request.source_object;
}

MaterializedGpuObjectBatchCopy GpuObjectCopyMaterializer::materialize_copy_batch(
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

void GpuObjectCopyBackend::copy_objects(
    const std::vector<GpuObjectCopyRequest> &requests)
{
    for (const GpuObjectCopyRequest &request : requests)
    {
        copy_object(request);
    }
}

MaterializedGpuComm::MaterializedGpuComm(
    GpuObjectCopyMaterializer &materializer, GpuObjectCopyBackend &backend)
    : materializer_(materializer), backend_(backend)
{
}

std::shared_ptr<void> MaterializedGpuComm::copy(const GpuCommCopyRequest &request)
{
    if (request.source_object == nullptr)
    {
        throw std::invalid_argument("materialized GPU copy source object is null");
    }

    MaterializedGpuObjectCopy materialized = materializer_.materialize_copy(request);
    validate_materialized_destination_object(materialized.destination_object);

    const GpuObjectCopyValidationResult validation =
        validate_full_object_copy_request(materialized.object_copy);
    if (!validation.ok())
    {
        throw std::invalid_argument(validation.format_errors());
    }

    backend_.copy_object(materialized.object_copy);
    return std::move(materialized.destination_object);
}

std::vector<std::shared_ptr<void>> MaterializedGpuComm::copy_batch(
    const std::vector<GpuCommCopyRequest> &requests)
{
    for (const GpuCommCopyRequest &request : requests)
    {
        if (request.source_object == nullptr)
        {
            throw std::invalid_argument("materialized GPU copy source object is null");
        }
    }

    MaterializedGpuObjectBatchCopy materialized =
        materializer_.materialize_copy_batch(requests);
    if (materialized.destination_objects.size() != requests.size() ||
        materialized.object_copies.size() != requests.size())
    {
        throw std::invalid_argument(
            "materialized GPU batch copy result size mismatch");
    }

    for (std::size_t index = 0; index < materialized.object_copies.size(); ++index)
    {
        validate_materialized_destination_object(
            materialized.destination_objects[index]);
        const GpuObjectCopyValidationResult validation =
            validate_full_object_copy_request(materialized.object_copies[index]);
        if (!validation.ok())
        {
            throw std::invalid_argument(validation.format_errors());
        }
    }

    backend_.copy_objects(materialized.object_copies);
    return std::move(materialized.destination_objects);
}

void MissingInterNodeTransportBackend::copy_object(
    const InterNodeObjectCopyRequest &request)
{
    std::ostringstream stream;
    stream << "inter-node communication backend is not configured for device "
           << request.source.logical_device << " -> "
           << request.destination.logical_device;
    throw std::runtime_error(stream.str());
}

PlannedMaterializedGpuComm::PlannedMaterializedGpuComm(
    const MgpuCommunicationPlan &plan, MgpuTopology topology,
    GpuObjectCopyMaterializer &materializer,
    GpuObjectCopyBackend &local_backend,
    InterNodeTransportBackend &inter_node_backend)
    : routes_(plan.routes), topology_(std::move(topology)),
      materializer_(materializer), local_backend_(local_backend),
      inter_node_backend_(inter_node_backend)
{
    if (!plan.ok())
    {
        throw std::invalid_argument(
            "planned materialized GPU communication requires a valid "
            "communication plan:\n" +
            plan.format_diagnostics());
    }

    for (std::size_t route_index = 0; route_index < routes_.size(); ++route_index)
    {
        const MgpuCopyRoute &route = routes_[route_index];
        if (route.source_id == 0 || route.destination_id == 0)
        {
            throw std::invalid_argument(
                "planned communication route uses reserved value id 0");
        }

        const auto [_, inserted] = route_indices_.emplace(
            route_key(route.source_id, route.destination_id), route_index);
        if (!inserted)
        {
            throw std::invalid_argument(
                "duplicate planned communication route for object copy " +
                format_route_key(route.source_id, route.destination_id));
        }
    }
}

std::shared_ptr<void> PlannedMaterializedGpuComm::copy(
    const GpuCommCopyRequest &request)
{
    const auto route_iter =
        route_indices_.find(route_key(request.source_id, request.destination_id));
    if (route_iter == route_indices_.end())
    {
        throw std::invalid_argument(
            "no planned communication route for object copy " +
            format_route_key(request.source_id, request.destination_id));
    }

    const MgpuCopyRoute &route = routes_[route_iter->second];
    validate_request_matches_route(route, request);
    validate_request_has_source_object(request);

    MaterializedGpuObjectCopy materialized =
        materializer_.materialize_copy(request);
    validate_materialized_destination_object(materialized.destination_object);

    copy_object_for_route(
        topology_, local_backend_, inter_node_backend_, route,
        materialized.object_copy);
    return std::move(materialized.destination_object);
}

std::vector<std::shared_ptr<void>> PlannedMaterializedGpuComm::copy_batch(
    const std::vector<GpuCommCopyRequest> &requests)
{
    if (requests.empty())
    {
        return {};
    }

    std::vector<const MgpuCopyRoute *> request_routes;
    request_routes.reserve(requests.size());
    bool has_inter_node_route = false;
    for (const GpuCommCopyRequest &request : requests)
    {
        const auto route_iter =
            route_indices_.find(route_key(request.source_id, request.destination_id));
        if (route_iter == route_indices_.end())
        {
            throw std::invalid_argument(
                "no planned communication route for object copy " +
                format_route_key(request.source_id, request.destination_id));
        }

        const MgpuCopyRoute &route = routes_[route_iter->second];
        validate_request_matches_route(route, request);
        validate_request_has_source_object(request);
        validate_route_matches_topology(topology_, route);
        if (route.transport == MgpuTransportKind::InterNode)
        {
            has_inter_node_route = true;
        }
        request_routes.push_back(&route);
    }

    MaterializedGpuObjectBatchCopy materialized =
        materializer_.materialize_copy_batch(requests);
    if (materialized.destination_objects.size() != requests.size() ||
        materialized.object_copies.size() != requests.size())
    {
        throw std::invalid_argument(
            "materialized GPU batch copy result size mismatch");
    }

    for (std::size_t index = 0; index < materialized.object_copies.size(); ++index)
    {
        validate_materialized_destination_object(
            materialized.destination_objects[index]);
        validate_route_matches_object_copy(
            *request_routes[index], materialized.object_copies[index]);
    }

    if (has_inter_node_route)
    {
        for (std::size_t index = 0; index < materialized.object_copies.size(); ++index)
        {
            copy_object_for_route(
                topology_, local_backend_, inter_node_backend_,
                *request_routes[index], materialized.object_copies[index]);
        }
    }
    else
    {
        local_backend_.copy_objects(materialized.object_copies);
    }

    return std::move(materialized.destination_objects);
}

}  // namespace poseidon::mgpu
