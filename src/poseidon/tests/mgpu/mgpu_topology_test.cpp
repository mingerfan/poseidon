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

void test_communication_plan_routes_plaintext_copies()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(10) }));
    schedule.ops.push_back(
        op(MgpuOpKind::CopyPlain, 1, { value(10) }, { value(11) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 1, { value(11) }, {}));

    const MgpuCommunicationPlan plan =
        plan_schedule_communication(schedule, make_single_node_topology(2));

    require(plan.ok(), "plaintext copy plan should pass:\n" + plan.format_diagnostics());
    require(plan.routes.size() == 1, "plaintext copy route count mismatch");
    require(
        plan.routes[0].kind == MgpuValueKind::Plaintext,
        "plaintext copy route kind mismatch");
    require(
        plan.routes[0].transport == MgpuTransportKind::CudaPeer,
        "plaintext copy route transport mismatch");

    const std::string text = dump_communication_plan(plan);
    require_contains(text, "copy %10 -> %11 plaintext device 0 -> 1");

    const std::string json = communication_plan_to_json(plan);
    require_contains(json, "\"kind\": \"plaintext\"");
    require_contains(json, "\"source_id\": 10");
    require_contains(json, "\"destination_id\": 11");
}

void test_communication_plan_reports_copy_kind_mismatch()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(1) }));
    schedule.ops.push_back(
        op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));

    const MgpuCommunicationPlan plan =
        plan_schedule_communication(schedule, make_single_node_topology(2));

    require(!plan.ok(), "copy kind mismatch should fail planning");
    require(plan.routes.empty(), "copy kind mismatch should not produce a route");
    require_contains(
        plan.format_diagnostics(),
        "copy input value %1 expected ciphertext, got plaintext");
}

void test_communication_plan_reports_copy_arity_errors()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(
        op(MgpuOpKind::CopyCipher, 1, { value(1), value(1) }, { value(2) }));

    const MgpuCommunicationPlan plan =
        plan_schedule_communication(schedule, make_single_node_topology(2));

    require(!plan.ok(), "copy arity mismatch should fail planning");
    require(plan.routes.empty(), "copy arity mismatch should not produce a route");
    require_contains(
        plan.format_diagnostics(),
        "copy op must have exactly one input and one output");
}

void test_communication_plan_reports_missing_route_devices()
{
    MgpuSchedule missing_source;
    missing_source.ops.push_back(
        op(MgpuOpKind::UploadCipher, 3, {}, { value(1) }));
    missing_source.ops.push_back(
        op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));

    const MgpuCommunicationPlan missing_source_plan =
        plan_schedule_communication(missing_source, make_single_node_topology(2));

    require(!missing_source_plan.ok(), "missing source device should fail planning");
    require(
        missing_source_plan.routes.empty(),
        "missing source device should not produce a route");
    require_contains(
        missing_source_plan.format_diagnostics(),
        "copy source device 3 is not present in topology");

    MgpuSchedule missing_destination;
    missing_destination.ops.push_back(
        op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    missing_destination.ops.push_back(
        op(MgpuOpKind::CopyCipher, 3, { value(1) }, { value(2) }));

    const MgpuCommunicationPlan missing_destination_plan =
        plan_schedule_communication(
            missing_destination, make_single_node_topology(2));

    require(
        !missing_destination_plan.ok(),
        "missing destination device should fail planning");
    require(
        missing_destination_plan.routes.empty(),
        "missing destination device should not produce a route");
    require_contains(
        missing_destination_plan.format_diagnostics(),
        "copy destination device 3 is not present in topology");
}

void test_communication_plan_rejects_duplicate_logical_devices()
{
    MgpuTopology topology;
    topology.devices.push_back(MgpuLogicalDevice{ 0, 0, 0 });
    topology.devices.push_back(MgpuLogicalDevice{ 0, 1, 0 });

    const MgpuCommunicationPlan plan =
        plan_schedule_communication(make_copy_schedule(), topology);

    require(!plan.ok(), "duplicate logical devices should fail planning");
    require(plan.routes.empty(), "invalid topology should not produce routes");
    require_contains(plan.format_diagnostics(), "duplicate logical device 0");
}

void test_communication_plan_rejects_duplicate_local_devices()
{
    MgpuTopology topology;
    topology.devices.push_back(MgpuLogicalDevice{ 0, 0, 0 });
    topology.devices.push_back(MgpuLogicalDevice{ 1, 0, 0 });

    const MgpuCommunicationPlan plan =
        plan_schedule_communication(make_copy_schedule(), topology);

    require(!plan.ok(), "duplicate local devices should fail planning");
    require(plan.routes.empty(), "invalid topology should not produce routes");
    require_contains(plan.format_diagnostics(), "duplicate local device 0 on node 0");
}

void test_communication_plan_rejects_negative_topology_ids()
{
    MgpuTopology topology;
    topology.devices.push_back(MgpuLogicalDevice{ -1, -2, -3 });

    MgpuSchedule schedule;
    const MgpuCommunicationPlan plan =
        plan_schedule_communication(schedule, topology);

    require(!plan.ok(), "negative topology ids should fail planning");
    require_contains(
        plan.format_diagnostics(), "has negative logical device -1");
    require_contains(plan.format_diagnostics(), "has negative node id -2");
    require_contains(plan.format_diagnostics(), "has negative local device -3");
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
        test_communication_plan_routes_plaintext_copies();
        test_communication_plan_reports_copy_kind_mismatch();
        test_communication_plan_reports_copy_arity_errors();
        test_communication_plan_reports_missing_route_devices();
        test_communication_plan_rejects_duplicate_logical_devices();
        test_communication_plan_rejects_duplicate_local_devices();
        test_communication_plan_rejects_negative_topology_ids();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu topology test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu topology tests passed\n";
    return EXIT_SUCCESS;
}
