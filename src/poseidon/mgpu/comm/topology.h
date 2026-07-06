#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

enum class MgpuTransportKind
{
    SameDevice,
    CudaPeer,
    InterNode
};

struct MgpuLogicalDevice
{
    int logical_device = 0;
    int node_id = 0;
    int local_device = 0;
};

struct MgpuTopology
{
    std::vector<MgpuLogicalDevice> devices;
};

struct MgpuCopyRoute
{
    ValueId source_id = 0;
    ValueId destination_id = 0;
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    int source_device = 0;
    int destination_device = 0;
    MgpuTransportKind transport = MgpuTransportKind::SameDevice;
};

struct MgpuCommunicationPlanDiagnostic
{
    std::size_t op_index = 0;
    std::string message;
};

struct MgpuCommunicationPlan
{
    std::vector<MgpuCopyRoute> routes;
    std::size_t same_device_copies = 0;
    std::size_t cuda_peer_copies = 0;
    std::size_t inter_node_copies = 0;
    std::vector<MgpuCommunicationPlanDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

const char *to_string(MgpuTransportKind kind) noexcept;

MgpuTopology make_single_node_topology(int device_count);
MgpuTopology make_uniform_cluster_topology(int node_count, int devices_per_node);

MgpuCommunicationPlan plan_schedule_communication(
    const MgpuSchedule &schedule, const MgpuTopology &topology);

std::string dump_communication_plan(const MgpuCommunicationPlan &plan);
void dump_communication_plan(std::ostream &stream, const MgpuCommunicationPlan &plan);

std::string communication_plan_to_json(
    const MgpuCommunicationPlan &plan, int indent = 2);

}  // namespace poseidon::mgpu
