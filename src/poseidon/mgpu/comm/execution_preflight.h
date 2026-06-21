#pragma once

#include "poseidon/mgpu/comm/topology.h"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct MgpuCommunicationExecutionOptions
{
    bool same_device_available = true;
    bool cuda_peer_available = false;
    bool inter_node_available = false;
};

struct MgpuCommunicationExecutionDiagnostic
{
    std::size_t route_index = 0;
    std::string message;
};

struct MgpuCommunicationExecutionPreflight
{
    std::size_t same_device_routes = 0;
    std::size_t cuda_peer_routes = 0;
    std::size_t inter_node_routes = 0;
    std::vector<MgpuCommunicationExecutionDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

MgpuCommunicationExecutionPreflight preflight_communication_execution(
    const MgpuCommunicationPlan &plan,
    const MgpuCommunicationExecutionOptions &options = {});

std::string dump_communication_execution_preflight(
    const MgpuCommunicationExecutionPreflight &preflight);
void dump_communication_execution_preflight(
    std::ostream &stream,
    const MgpuCommunicationExecutionPreflight &preflight);

std::string communication_execution_preflight_to_json(
    const MgpuCommunicationExecutionPreflight &preflight, int indent = 2);

}  // namespace poseidon::mgpu
