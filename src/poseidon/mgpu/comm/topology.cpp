#include "poseidon/mgpu/comm/topology.h"

#include "poseidon/util/json.h"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

using Json = nlohmann::json;

struct ValueState
{
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    int device_id = 0;
};

void add_diagnostic(
    MgpuCommunicationPlan &plan, std::size_t op_index, std::string message)
{
    plan.diagnostics.push_back(
        MgpuCommunicationPlanDiagnostic{ op_index, std::move(message) });
}

void add_topology_diagnostic(MgpuCommunicationPlan &plan, std::string message)
{
    add_diagnostic(plan, 0, std::move(message));
}

bool validate_topology(MgpuCommunicationPlan &plan, const MgpuTopology &topology)
{
    bool valid = true;
    std::unordered_map<int, std::size_t> logical_devices;
    std::set<std::pair<int, int>> local_devices;

    for (std::size_t index = 0; index < topology.devices.size(); ++index)
    {
        const MgpuLogicalDevice &device = topology.devices[index];
        if (device.logical_device < 0)
        {
            std::ostringstream stream;
            stream << "topology device entry #" << index
                   << " has negative logical device "
                   << device.logical_device;
            add_topology_diagnostic(plan, stream.str());
            valid = false;
        }
        if (device.node_id < 0)
        {
            std::ostringstream stream;
            stream << "topology device entry #" << index
                   << " has negative node id " << device.node_id;
            add_topology_diagnostic(plan, stream.str());
            valid = false;
        }
        if (device.local_device < 0)
        {
            std::ostringstream stream;
            stream << "topology device entry #" << index
                   << " has negative local device " << device.local_device;
            add_topology_diagnostic(plan, stream.str());
            valid = false;
        }

        const auto [logical_iter, logical_inserted] =
            logical_devices.emplace(device.logical_device, index);
        if (!logical_inserted)
        {
            std::ostringstream stream;
            stream << "duplicate logical device " << device.logical_device
                   << " in topology entries #" << logical_iter->second
                   << " and #" << index;
            add_topology_diagnostic(plan, stream.str());
            valid = false;
        }

        const auto [_, local_inserted] =
            local_devices.emplace(device.node_id, device.local_device);
        if (!local_inserted)
        {
            std::ostringstream stream;
            stream << "duplicate local device " << device.local_device
                   << " on node " << device.node_id << " in topology";
            add_topology_diagnostic(plan, stream.str());
            valid = false;
        }
    }

    return valid;
}

const MgpuLogicalDevice *find_device(const MgpuTopology &topology, int logical_device)
{
    const auto iter = std::find_if(
        topology.devices.begin(), topology.devices.end(),
        [logical_device](const MgpuLogicalDevice &device) {
            return device.logical_device == logical_device;
        });
    if (iter == topology.devices.end())
    {
        return nullptr;
    }
    return &(*iter);
}

MgpuTransportKind route_transport(
    const MgpuLogicalDevice &source, const MgpuLogicalDevice &destination)
{
    if (source.logical_device == destination.logical_device)
    {
        return MgpuTransportKind::SameDevice;
    }
    if (source.node_id == destination.node_id)
    {
        return MgpuTransportKind::CudaPeer;
    }
    return MgpuTransportKind::InterNode;
}

MgpuValueKind copy_kind(MgpuOpKind kind)
{
    return kind == MgpuOpKind::CopyPlain ? MgpuValueKind::Plaintext
                                         : MgpuValueKind::Ciphertext;
}

bool output_kind(MgpuOpKind kind, MgpuValueKind &output)
{
    switch (kind)
    {
    case MgpuOpKind::UploadPlain:
    case MgpuOpKind::CopyPlain:
        output = MgpuValueKind::Plaintext;
        return true;
    case MgpuOpKind::UploadCipher:
    case MgpuOpKind::CopyCipher:
    case MgpuOpKind::Add:
    case MgpuOpKind::AddPlain:
    case MgpuOpKind::Sub:
    case MgpuOpKind::MultiplyPlain:
    case MgpuOpKind::Multiply:
    case MgpuOpKind::Negate:
    case MgpuOpKind::Relinearize:
    case MgpuOpKind::Rescale:
    case MgpuOpKind::Rotate:
    case MgpuOpKind::BootstrapFallback:
        output = MgpuValueKind::Ciphertext;
        return true;
    case MgpuOpKind::Download:
        return false;
    }
    return false;
}

void define_output(
    MgpuCommunicationPlan &plan, std::unordered_map<ValueId, ValueState> &values,
    std::size_t op_index, ValueId id, MgpuValueKind kind, int device_id)
{
    if (id == 0)
    {
        add_diagnostic(plan, op_index, "output value id 0 is reserved");
        return;
    }

    const auto [_, inserted] = values.emplace(id, ValueState{ kind, device_id });
    if (!inserted)
    {
        std::ostringstream stream;
        stream << "duplicate output value %" << id;
        add_diagnostic(plan, op_index, stream.str());
    }
}

void maybe_define_output(
    MgpuCommunicationPlan &plan, std::unordered_map<ValueId, ValueState> &values,
    std::size_t op_index, const MgpuOp &op)
{
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    if (!output_kind(op.kind, kind))
    {
        return;
    }
    if (op.outputs.size() != 1)
    {
        add_diagnostic(plan, op_index, "stateful op must have exactly one output");
        return;
    }
    define_output(plan, values, op_index, op.outputs[0].id, kind, op.device_id);
}

void increment_transport_count(MgpuCommunicationPlan &plan, MgpuTransportKind transport)
{
    switch (transport)
    {
    case MgpuTransportKind::SameDevice:
        ++plan.same_device_copies;
        return;
    case MgpuTransportKind::CudaPeer:
        ++plan.cuda_peer_copies;
        return;
    case MgpuTransportKind::InterNode:
        ++plan.inter_node_copies;
        return;
    }
}

Json routes_to_json(const std::vector<MgpuCopyRoute> &routes)
{
    Json result = Json::array();
    for (const MgpuCopyRoute &route : routes)
    {
        result.push_back(Json{
            { "source_id", route.source_id },
            { "destination_id", route.destination_id },
            { "kind", to_string(route.kind) },
            { "source_device", route.source_device },
            { "destination_device", route.destination_device },
            { "transport", to_string(route.transport) },
        });
    }
    return result;
}

Json diagnostics_to_json(const std::vector<MgpuCommunicationPlanDiagnostic> &diagnostics)
{
    Json result = Json::array();
    for (const MgpuCommunicationPlanDiagnostic &diagnostic : diagnostics)
    {
        result.push_back(Json{
            { "op_index", diagnostic.op_index },
            { "message", diagnostic.message },
        });
    }
    return result;
}

}  // namespace

const char *to_string(MgpuTransportKind kind) noexcept
{
    switch (kind)
    {
    case MgpuTransportKind::SameDevice:
        return "same_device";
    case MgpuTransportKind::CudaPeer:
        return "cuda_peer";
    case MgpuTransportKind::InterNode:
        return "inter_node";
    }
    return "unknown";
}

MgpuTopology make_single_node_topology(int device_count)
{
    if (device_count < 0)
    {
        throw std::invalid_argument("device_count must be non-negative");
    }

    MgpuTopology topology;
    topology.devices.reserve(static_cast<std::size_t>(device_count));
    for (int device = 0; device < device_count; ++device)
    {
        topology.devices.push_back(MgpuLogicalDevice{ device, 0, device });
    }
    return topology;
}

MgpuTopology make_uniform_cluster_topology(int node_count, int devices_per_node)
{
    if (node_count < 0)
    {
        throw std::invalid_argument("node_count must be non-negative");
    }
    if (devices_per_node < 0)
    {
        throw std::invalid_argument("devices_per_node must be non-negative");
    }
    const std::size_t node_size = static_cast<std::size_t>(node_count);
    const std::size_t devices_per_node_size =
        static_cast<std::size_t>(devices_per_node);
    if (node_size != 0 &&
        devices_per_node_size >
            std::numeric_limits<std::size_t>::max() / node_size)
    {
        throw std::invalid_argument(
            "uniform cluster topology device count exceeds addressable size");
    }
    const std::size_t total_devices = node_size * devices_per_node_size;
    if (total_devices >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument(
            "uniform cluster topology device count exceeds logical device id range");
    }

    MgpuTopology topology;
    topology.devices.reserve(total_devices);
    int logical_device = 0;
    for (int node = 0; node < node_count; ++node)
    {
        for (int local = 0; local < devices_per_node; ++local)
        {
            topology.devices.push_back(MgpuLogicalDevice{ logical_device, node, local });
            ++logical_device;
        }
    }
    return topology;
}

std::string MgpuCommunicationPlan::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << "op #" << diagnostics[i].op_index << ": "
               << diagnostics[i].message;
    }
    return stream.str();
}

MgpuCommunicationPlan plan_schedule_communication(
    const MgpuSchedule &schedule, const MgpuTopology &topology)
{
    MgpuCommunicationPlan plan;
    if (!validate_topology(plan, topology))
    {
        return plan;
    }

    std::unordered_map<ValueId, ValueState> values;
    for (std::size_t op_index = 0; op_index < schedule.ops.size(); ++op_index)
    {
        const MgpuOp &op = schedule.ops[op_index];
        if (!is_copy_op(op.kind))
        {
            maybe_define_output(plan, values, op_index, op);
            continue;
        }
        if (op.inputs.size() != 1 || op.outputs.size() != 1)
        {
            add_diagnostic(
                plan, op_index, "copy op must have exactly one input and one output");
            continue;
        }

        const auto value_iter = values.find(op.inputs[0].id);
        if (value_iter == values.end())
        {
            std::ostringstream stream;
            stream << "unknown copy input value %" << op.inputs[0].id;
            add_diagnostic(plan, op_index, stream.str());
            continue;
        }

        const MgpuValueKind expected_kind = copy_kind(op.kind);
        if (value_iter->second.kind != expected_kind)
        {
            std::ostringstream stream;
            stream << "copy input value %" << op.inputs[0].id << " expected "
                   << to_string(expected_kind) << ", got "
                   << to_string(value_iter->second.kind);
            add_diagnostic(plan, op_index, stream.str());
            continue;
        }

        const int source_device = value_iter->second.device_id;
        const MgpuLogicalDevice *source = find_device(topology, source_device);
        const MgpuLogicalDevice *destination = find_device(topology, op.device_id);
        if (destination == nullptr)
        {
            std::ostringstream stream;
            stream << "copy destination device " << op.device_id
                   << " is not present in topology";
            add_diagnostic(plan, op_index, stream.str());
            continue;
        }

        if (source == nullptr)
        {
            std::ostringstream stream;
            stream << "copy source device " << source_device
                   << " is not present in topology";
            add_diagnostic(plan, op_index, stream.str());
            continue;
        }

        const MgpuTransportKind transport = route_transport(*source, *destination);
        MgpuCopyRoute route;
        route.source_id = op.inputs[0].id;
        route.destination_id = op.outputs[0].id;
        route.kind = copy_kind(op.kind);
        route.source_device = source->logical_device;
        route.destination_device = destination->logical_device;
        route.transport = transport;
        increment_transport_count(plan, transport);
        plan.routes.push_back(route);
        maybe_define_output(plan, values, op_index, op);
    }
    return plan;
}

std::string dump_communication_plan(const MgpuCommunicationPlan &plan)
{
    std::ostringstream stream;
    dump_communication_plan(stream, plan);
    return stream.str();
}

void dump_communication_plan(std::ostream &stream, const MgpuCommunicationPlan &plan)
{
    stream << "mgpu.communication_plan:\n";
    stream << "  same_device: " << plan.same_device_copies << '\n';
    stream << "  cuda_peer: " << plan.cuda_peer_copies << '\n';
    stream << "  inter_node: " << plan.inter_node_copies << '\n';
    for (const MgpuCopyRoute &route : plan.routes)
    {
        stream << "  copy %" << route.source_id << " -> %" << route.destination_id
               << " " << to_string(route.kind) << " device "
               << route.source_device << " -> " << route.destination_device
               << " via " << to_string(route.transport) << '\n';
    }
    if (!plan.diagnostics.empty())
    {
        stream << "  diagnostics:\n";
        for (const MgpuCommunicationPlanDiagnostic &diagnostic : plan.diagnostics)
        {
            stream << "    op #" << diagnostic.op_index << ": "
                   << diagnostic.message << '\n';
        }
    }
}

std::string communication_plan_to_json(const MgpuCommunicationPlan &plan, int indent)
{
    Json root;
    root["version"] = 1;
    root["ok"] = plan.ok();
    root["counts"] = Json{
        { "same_device", plan.same_device_copies },
        { "cuda_peer", plan.cuda_peer_copies },
        { "inter_node", plan.inter_node_copies },
    };
    root["routes"] = routes_to_json(plan.routes);
    root["diagnostics"] = diagnostics_to_json(plan.diagnostics);
    return root.dump(indent);
}

}  // namespace poseidon::mgpu
