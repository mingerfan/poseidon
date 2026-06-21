#include "poseidon/mgpu/comm/topology.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace poseidon::mgpu;

namespace
{

MgpuValueRef value(ValueId id)
{
    return MgpuValueRef{ id };
}

MgpuOp op(
    MgpuOpKind kind, int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs)
{
    return MgpuOp{ kind, device_id, std::move(inputs), std::move(outputs), {} };
}

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void require_contains(const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos)
    {
        throw std::runtime_error("expected text to contain: " + needle + "\ntext:\n" + text);
    }
}

void test_single_node_topology()
{
    const MgpuTopology topology = make_single_node_topology(8);
    require(topology.devices.size() == 8, "single-node topology device count mismatch");
    require(topology.devices[7].logical_device == 7, "logical device mismatch");
    require(topology.devices[7].node_id == 0, "single-node node id mismatch");
    require(topology.devices[7].local_device == 7, "single-node local device mismatch");
}

void test_uniform_cluster_topology()
{
    const MgpuTopology topology = make_uniform_cluster_topology(4, 8);
    require(topology.devices.size() == 32, "cluster topology device count mismatch");
    require(topology.devices[0].node_id == 0, "cluster first node mismatch");
    require(topology.devices[7].local_device == 7, "cluster local device mismatch");
    require(topology.devices[8].node_id == 1, "cluster second node mismatch");
    require(topology.devices[31].node_id == 3, "cluster last node mismatch");
    require(topology.devices[31].local_device == 7, "cluster last local device mismatch");
}

MgpuSchedule make_copy_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 1, { value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 2, { value(3) }, { value(4) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 2, { value(4) }, {}));
    return schedule;
}

void test_communication_plan_classifies_routes()
{
    const MgpuTopology topology = make_uniform_cluster_topology(2, 2);
    const MgpuCommunicationPlan plan =
        plan_schedule_communication(make_copy_schedule(), topology);

    require(plan.ok(), "communication plan should pass:\n" + plan.format_diagnostics());
    require(plan.routes.size() == 3, "route count mismatch");
    require(plan.cuda_peer_copies == 1, "cuda peer copy count mismatch");
    require(plan.same_device_copies == 1, "same-device copy count mismatch");
    require(plan.inter_node_copies == 1, "inter-node copy count mismatch");
    require(plan.routes[0].transport == MgpuTransportKind::CudaPeer, "first route mismatch");
    require(plan.routes[0].source_device == 0, "first route source mismatch");
    require(plan.routes[0].destination_device == 1, "first route destination mismatch");
    require(plan.routes[1].transport == MgpuTransportKind::SameDevice, "second route mismatch");
    require(plan.routes[2].transport == MgpuTransportKind::InterNode, "third route mismatch");

    const std::string text = dump_communication_plan(plan);
    require_contains(text, "cuda_peer: 1");
    require_contains(text, "inter_node: 1");
    require_contains(text, "device 1 -> 2 via inter_node");

    const std::string json = communication_plan_to_json(plan);
    require_contains(json, "\"cuda_peer\": 1");
    require_contains(json, "\"transport\": \"inter_node\"");
}

void test_communication_plan_reports_bad_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 1, { value(99) }, { value(100) }));

    const MgpuCommunicationPlan plan =
        plan_schedule_communication(schedule, make_single_node_topology(2));
    require(!plan.ok(), "unknown copy input should fail planning");
    require_contains(plan.format_diagnostics(), "unknown copy input value %99");
}

}  // namespace

int main()
{
    try
    {
        test_single_node_topology();
        test_uniform_cluster_topology();
        test_communication_plan_classifies_routes();
        test_communication_plan_reports_bad_schedule();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu topology test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu topology tests passed\n";
    return EXIT_SUCCESS;
}
