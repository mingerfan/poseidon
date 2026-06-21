#include "poseidon/mgpu/comm/planned_materialized_gpu_comm.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

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

}  // namespace

PlannedMaterializedGpuComm::PlannedMaterializedGpuComm(
    const MgpuCommunicationPlan &plan, GpuObjectCopyMaterializer &materializer,
    RoutedGpuObjectCopyBackend &backend)
    : routes_(plan.routes), materializer_(materializer), backend_(backend)
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

    MaterializedGpuObjectCopy materialized =
        materializer_.materialize_copy(request);
    if (materialized.destination_object == nullptr)
    {
        throw std::invalid_argument(
            "materialized GPU copy destination object is null");
    }

    backend_.copy_object(route, materialized.object_copy);
    return std::move(materialized.destination_object);
}

}  // namespace poseidon::mgpu
