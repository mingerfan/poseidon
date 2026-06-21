#include "poseidon/mgpu/comm/execution_preflight.h"

#include "poseidon/util/json.h"

#include <ostream>
#include <sstream>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

using Json = nlohmann::json;

void add_diagnostic(
    MgpuCommunicationExecutionPreflight &preflight,
    std::size_t route_index,
    const MgpuCopyRoute &route,
    std::string message)
{
    preflight.diagnostics.push_back(
        MgpuCommunicationExecutionDiagnostic{
            route_index,
            route.transport,
            route.source_device,
            route.destination_device,
            std::move(message),
        });
}

bool route_supported(
    const MgpuCopyRoute &route,
    const MgpuCommunicationExecutionOptions &options)
{
    switch (route.transport)
    {
    case MgpuTransportKind::SameDevice:
        return options.same_device_available;
    case MgpuTransportKind::CudaPeer:
        return options.cuda_peer_available;
    case MgpuTransportKind::InterNode:
        return options.inter_node_available;
    }
    return false;
}

const char *missing_transport_message(MgpuTransportKind transport) noexcept
{
    switch (transport)
    {
    case MgpuTransportKind::SameDevice:
        return "same-device copy support is not available";
    case MgpuTransportKind::CudaPeer:
        return "CUDA peer or host-staged copy backend is not available";
    case MgpuTransportKind::InterNode:
        return "inter-node communication backend is not available";
    }
    return "unknown communication backend is not available";
}

void increment_count(
    MgpuCommunicationExecutionPreflight &preflight,
    MgpuTransportKind transport)
{
    switch (transport)
    {
    case MgpuTransportKind::SameDevice:
        ++preflight.same_device_routes;
        return;
    case MgpuTransportKind::CudaPeer:
        ++preflight.cuda_peer_routes;
        return;
    case MgpuTransportKind::InterNode:
        ++preflight.inter_node_routes;
        return;
    }
}

Json diagnostics_to_json(
    const std::vector<MgpuCommunicationExecutionDiagnostic> &diagnostics)
{
    Json result = Json::array();
    for (const MgpuCommunicationExecutionDiagnostic &diagnostic : diagnostics)
    {
        result.push_back(Json{
            { "route_index", diagnostic.route_index },
            { "transport", to_string(diagnostic.transport) },
            { "source_device", diagnostic.source_device },
            { "destination_device", diagnostic.destination_device },
            { "message", diagnostic.message },
        });
    }
    return result;
}

}  // namespace

std::string MgpuCommunicationExecutionPreflight::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << "route #" << diagnostics[i].route_index << " "
               << to_string(diagnostics[i].transport) << " device "
               << diagnostics[i].source_device << " -> "
               << diagnostics[i].destination_device << ": "
               << diagnostics[i].message;
    }
    return stream.str();
}

MgpuCommunicationExecutionPreflight preflight_communication_execution(
    const MgpuCommunicationPlan &plan,
    const MgpuCommunicationExecutionOptions &options)
{
    MgpuCommunicationExecutionPreflight preflight;
    for (std::size_t route_index = 0; route_index < plan.routes.size(); ++route_index)
    {
        const MgpuCopyRoute &route = plan.routes[route_index];
        increment_count(preflight, route.transport);
        if (!route_supported(route, options))
        {
            std::ostringstream stream;
            stream << missing_transport_message(route.transport) << " for device "
                   << route.source_device << " -> " << route.destination_device;
            add_diagnostic(preflight, route_index, route, stream.str());
        }
    }
    return preflight;
}

std::string dump_communication_execution_preflight(
    const MgpuCommunicationExecutionPreflight &preflight)
{
    std::ostringstream stream;
    dump_communication_execution_preflight(stream, preflight);
    return stream.str();
}

void dump_communication_execution_preflight(
    std::ostream &stream,
    const MgpuCommunicationExecutionPreflight &preflight)
{
    stream << "mgpu.communication_execution_preflight:\n";
    stream << "  status: " << (preflight.ok() ? "ok" : "error") << '\n';
    stream << "  same_device_routes: " << preflight.same_device_routes << '\n';
    stream << "  cuda_peer_routes: " << preflight.cuda_peer_routes << '\n';
    stream << "  inter_node_routes: " << preflight.inter_node_routes << '\n';
    if (!preflight.diagnostics.empty())
    {
        stream << "  diagnostics:\n";
        for (const MgpuCommunicationExecutionDiagnostic &diagnostic :
             preflight.diagnostics)
        {
            stream << "    route #" << diagnostic.route_index << " "
                   << to_string(diagnostic.transport) << " device "
                   << diagnostic.source_device << " -> "
                   << diagnostic.destination_device << ": "
                   << diagnostic.message << '\n';
        }
    }
}

std::string communication_execution_preflight_to_json(
    const MgpuCommunicationExecutionPreflight &preflight,
    int indent)
{
    Json root;
    root["version"] = 1;
    root["ok"] = preflight.ok();
    root["counts"] = Json{
        { "same_device", preflight.same_device_routes },
        { "cuda_peer", preflight.cuda_peer_routes },
        { "inter_node", preflight.inter_node_routes },
    };
    root["diagnostics"] = diagnostics_to_json(preflight.diagnostics);
    return root.dump(indent);
}

}  // namespace poseidon::mgpu
