#include "poseidon/mgpu/runtime/planned_communication_executor.h"

#include "poseidon/mgpu/compiler/static_schedule_config.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
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
        throw std::runtime_error(
            "expected text to contain: " + needle + "\ntext:\n" + text);
    }
}

class UploadVectorHandler final : public ScheduleOpHandler
{
public:
    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override
    {
        executed_ops.push_back(op.kind);
        if (op.kind != MgpuOpKind::UploadCipher)
        {
            return;
        }

        object_store.define(
            op.outputs[0].id, MgpuValueKind::Ciphertext, op.device_id,
            std::make_shared<std::vector<int>>(
                std::initializer_list<int>{ 1, 2, 3, 4 }));
    }

    std::vector<MgpuOpKind> executed_ops;
};

class VectorCopyMaterializer final : public GpuObjectCopyMaterializer
{
public:
    MaterializedGpuObjectCopy materialize_copy(
        const GpuCommCopyRequest &request) override
    {
        requests.push_back(request);
        auto source =
            std::static_pointer_cast<std::vector<int>>(request.source_object);
        auto destination = std::make_shared<std::vector<int>>(source->size(), 0);

        MaterializedGpuObjectCopy result;
        result.destination_object = destination;
        result.object_copy.source_id = request.source_id;
        result.object_copy.destination_id = request.destination_id;
        result.object_copy.kind = request.kind;
        result.object_copy.buffers.push_back(GpuObjectBufferCopy{
            source->data(),
            destination->data(),
            source->size() * sizeof(int),
            request.source_device,
            request.destination_device,
        });
        return result;
    }

    std::vector<GpuCommCopyRequest> requests;
};

class CopyingLocalBackend final : public GpuObjectCopyBackend
{
public:
    void copy_object(const GpuObjectCopyRequest &request) override
    {
        requests.push_back(request);
        const GpuObjectBufferCopy &buffer = request.buffers[0];
        std::memcpy(buffer.destination, buffer.source, buffer.bytes);
    }

    std::vector<GpuObjectCopyRequest> requests;
};

class CopyingInterNodeBackend final : public InterNodeTransportBackend
{
public:
    void copy_object(const InterNodeObjectCopyRequest &request) override
    {
        requests.push_back(request);
        const GpuObjectBufferCopy &buffer = request.object_copy.buffers[0];
        std::memcpy(buffer.destination, buffer.source, buffer.bytes);
    }

    std::vector<InterNodeObjectCopyRequest> requests;
};

MgpuCommunicationExecutionOptions all_routes_available()
{
    MgpuCommunicationExecutionOptions options;
    options.cuda_peer_available = true;
    options.inter_node_available = true;
    return options;
}

MgpuSchedule make_two_route_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(
        op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));
    schedule.ops.push_back(
        op(MgpuOpKind::CopyCipher, 4, { value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 4, { value(3) }, {}));
    return schedule;
}

MgpuSchedule make_same_device_copy_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(
        op(MgpuOpKind::CopyCipher, 0, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(2) }, {}));
    return schedule;
}

void test_executor_uses_static_plan_for_copy_routes()
{
    UploadVectorHandler handler;
    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;
    PlannedCommunicationStaticScheduleExecutor executor(
        make_uniform_cluster_topology(2, 4), materializer, local_backend,
        inter_node_backend, handler, StaticScheduleExecutorOptions{ 5 },
        all_routes_available());

    const ScheduleExecutionResult result = executor.run(make_two_route_schedule());

    require(result.ok(), "planned executor should run:\n" + result.format_errors());
    require(
        handler.executed_ops.size() == 2,
        "handler should see upload and download only");
    require(
        materializer.requests.size() == 2,
        "materializer should receive both planned copies");
    require(
        local_backend.requests.size() == 1,
        "local backend should execute intra-node copy");
    require(
        inter_node_backend.requests.size() == 1,
        "inter-node backend should execute inter-node copy");
    require(
        inter_node_backend.requests[0].source.node_id == 0,
        "inter-node source node mismatch");
    require(
        inter_node_backend.requests[0].destination.node_id == 1,
        "inter-node destination node mismatch");
    require(
        result.object_store.at(3).device_id == 4,
        "final copied value should be on destination device");
    require(
        result.object_store.has_object(3),
        "final copied value should keep object");

    const auto copied = result.object_store.object_as<std::vector<int>>(3);
    require(
        *copied == std::vector<int>({ 1, 2, 3, 4 }),
        "final copied vector mismatch");
}

void test_executor_routes_same_device_copy_through_comm_layer()
{
    UploadVectorHandler handler;
    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;
    PlannedCommunicationStaticScheduleExecutor executor(
        make_single_node_topology(1), materializer, local_backend,
        inter_node_backend, handler, StaticScheduleExecutorOptions{ 1 });

    const ScheduleExecutionResult result =
        executor.run(make_same_device_copy_schedule());

    require(
        result.ok(),
        "planned executor should run same-device copy:\n" +
            result.format_errors());
    require(
        handler.executed_ops.size() == 2,
        "handler should see upload and download only");
    require(
        materializer.requests.size() == 1,
        "same-device copy should still be materialized explicitly");
    require(
        local_backend.requests.size() == 1,
        "same-device copy should use local object-copy backend");
    require(
        inter_node_backend.requests.empty(),
        "same-device copy should not use inter-node backend");
    require(
        materializer.requests[0].source_device == 0 &&
            materializer.requests[0].destination_device == 0,
        "same-device materializer route mismatch");
    require(
        local_backend.requests[0].buffers[0].source_device == 0 &&
            local_backend.requests[0].buffers[0].destination_device == 0,
        "same-device backend buffer route mismatch");
    require(result.object_store.at(2).device_id == 0, "copy destination device mismatch");
    require(result.object_store.has_object(2), "same-device copy should keep object");

    const auto copied = result.object_store.object_as<std::vector<int>>(2);
    require(
        *copied == std::vector<int>({ 1, 2, 3, 4 }),
        "same-device copied vector mismatch");
}

void test_executor_reports_plan_diagnostics_before_execution()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(
        op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));

    UploadVectorHandler handler;
    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;
    PlannedCommunicationStaticScheduleExecutor executor(
        make_single_node_topology(1), materializer, local_backend,
        inter_node_backend, handler, StaticScheduleExecutorOptions{ 2 });

    const ScheduleExecutionResult result = executor.run(schedule);

    require(!result.ok(), "missing topology route should fail");
    require_contains(
        result.format_errors(), "copy destination device 1 is not present");
    require(
        handler.executed_ops.empty(),
        "plan diagnostics should stop execution before upload");
    require(
        materializer.requests.empty(),
        "plan diagnostics should stop before materialization");
    require(local_backend.requests.empty(), "local backend should not run");
    require(
        inter_node_backend.requests.empty(),
        "inter-node backend should not run");
}

void test_executor_reports_missing_backend_before_execution()
{
    UploadVectorHandler handler;
    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;

    MgpuCommunicationExecutionOptions options;
    options.cuda_peer_available = true;
    options.inter_node_available = false;

    PlannedCommunicationStaticScheduleExecutor executor(
        make_uniform_cluster_topology(2, 4), materializer, local_backend,
        inter_node_backend, handler, StaticScheduleExecutorOptions{ 5 },
        options);

    const ScheduleExecutionResult result = executor.run(make_two_route_schedule());

    require(!result.ok(), "missing inter-node backend should fail");
    require(result.errors.size() == 1, "expected one missing backend error");
    require(
        result.errors[0].op_index == 2,
        "missing inter-node backend should be reported at second copy op");
    require_contains(
        result.format_errors(),
        "op #2: communication route #1");
    require_contains(
        result.format_errors(),
        "inter-node communication backend is not available");
    require(
        handler.executed_ops.empty(),
        "backend preflight should stop execution before upload");
    require(
        materializer.requests.empty(),
        "backend preflight should stop before materialization");
    require(local_backend.requests.empty(), "local backend should not run");
    require(
        inter_node_backend.requests.empty(),
        "inter-node backend should not run");
}

void test_executor_reports_missing_cuda_peer_backend_at_copy_op()
{
    UploadVectorHandler handler;
    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;

    MgpuCommunicationExecutionOptions options;
    options.cuda_peer_available = false;
    options.inter_node_available = true;

    PlannedCommunicationStaticScheduleExecutor executor(
        make_uniform_cluster_topology(2, 4), materializer, local_backend,
        inter_node_backend, handler, StaticScheduleExecutorOptions{ 5 },
        options);

    const ScheduleExecutionResult result = executor.run(make_two_route_schedule());

    require(!result.ok(), "missing CUDA peer backend should fail");
    require(result.errors.size() == 1, "expected one missing CUDA peer error");
    require(
        result.errors[0].op_index == 1,
        "missing CUDA peer backend should be reported at first copy op");
    require_contains(
        result.format_errors(),
        "op #1: communication route #0");
    require_contains(
        result.format_errors(),
        "CUDA peer or host-staged copy backend is not available");
    require(
        handler.executed_ops.empty(),
        "CUDA backend preflight should stop execution before upload");
    require(
        materializer.requests.empty(),
        "CUDA backend preflight should stop before materialization");
    require(local_backend.requests.empty(), "local backend should not run");
    require(
        inter_node_backend.requests.empty(),
        "inter-node backend should not run");
}

void test_executor_from_config_uses_topology_and_backend_declarations()
{
    StaticScheduleExecutionConfig config;
    config.pipeline.device_count = 5;
    config.node_count = 2;
    config.devices_per_node = 4;
    config.communication_execution.cuda_peer_available = true;
    config.communication_execution.inter_node_available = false;

    UploadVectorHandler handler;
    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;

    PlannedCommunicationStaticScheduleExecutor executor =
        PlannedCommunicationStaticScheduleExecutor::from_config(
            config, materializer, local_backend, inter_node_backend, handler);

    const ScheduleExecutionResult result = executor.run(make_two_route_schedule());

    require(
        !result.ok(),
        "executor built from config should reject undeclared inter-node backend");
    require_contains(
        result.format_errors(),
        "inter-node communication backend is not available");
    require(handler.executed_ops.empty(), "config backend gate should stop upload");
    require(
        materializer.requests.empty(),
        "config backend gate should stop before materialization");
    require(local_backend.requests.empty(), "local backend should not run");
    require(
        inter_node_backend.requests.empty(),
        "inter-node backend should not run");

    config.communication_execution.inter_node_available = true;
    PlannedCommunicationStaticScheduleExecutor cluster_executor =
        PlannedCommunicationStaticScheduleExecutor::from_config(
            config, materializer, local_backend, inter_node_backend, handler);

    const ScheduleExecutionResult cluster_result =
        cluster_executor.run(make_two_route_schedule());

    require(
        cluster_result.ok(),
        "executor built from cluster-capable config should run:\n" +
            cluster_result.format_errors());
    require(
        local_backend.requests.size() == 1,
        "config executor local copy count mismatch");
    require(
        inter_node_backend.requests.size() == 1,
        "config executor inter-node copy count mismatch");
    require(
        cluster_result.object_store.at(3).device_id == 4,
        "config executor final device mismatch");
}

}  // namespace

int main()
{
    try
    {
        test_executor_uses_static_plan_for_copy_routes();
        test_executor_routes_same_device_copy_through_comm_layer();
        test_executor_reports_plan_diagnostics_before_execution();
        test_executor_reports_missing_backend_before_execution();
        test_executor_reports_missing_cuda_peer_backend_at_copy_op();
        test_executor_from_config_uses_topology_and_backend_declarations();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu planned communication executor test failed: "
                  << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu planned communication executor tests passed\n";
    return EXIT_SUCCESS;
}
