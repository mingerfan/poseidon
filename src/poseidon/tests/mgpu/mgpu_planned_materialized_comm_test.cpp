#include "poseidon/mgpu/comm/planned_materialized_gpu_comm.h"

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

MgpuCopyRoute route(
    ValueId source_id, ValueId destination_id, int source_device,
    int destination_device, MgpuTransportKind transport)
{
    MgpuCopyRoute copy_route;
    copy_route.source_id = source_id;
    copy_route.destination_id = destination_id;
    copy_route.kind = MgpuValueKind::Ciphertext;
    copy_route.source_device = source_device;
    copy_route.destination_device = destination_device;
    copy_route.transport = transport;
    return copy_route;
}

GpuCommCopyRequest request_from_route(
    const MgpuCopyRoute &copy_route, std::shared_ptr<void> source_object)
{
    GpuCommCopyRequest request;
    request.source_id = copy_route.source_id;
    request.destination_id = copy_route.destination_id;
    request.kind = copy_route.kind;
    request.source_device = copy_route.source_device;
    request.destination_device = copy_route.destination_device;
    request.source_object = std::move(source_object);
    return request;
}

MgpuCommunicationPlan plan_with_route(const MgpuCopyRoute &copy_route)
{
    MgpuCommunicationPlan plan;
    plan.routes.push_back(copy_route);
    return plan;
}

void test_local_planned_route_uses_local_backend()
{
    const MgpuCopyRoute cuda_peer =
        route(10, 11, 1, 2, MgpuTransportKind::CudaPeer);
    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;
    RoutedGpuObjectCopyBackend routed_backend(
        make_uniform_cluster_topology(2, 4), local_backend, inter_node_backend);
    PlannedMaterializedGpuComm comm(
        plan_with_route(cuda_peer), materializer, routed_backend);

    auto source = std::make_shared<std::vector<int>>(
        std::initializer_list<int>{ 1, 2, 3 });
    const std::shared_ptr<void> copied =
        comm.copy(request_from_route(cuda_peer, source));

    require(materializer.requests.size() == 1, "materializer request mismatch");
    require(local_backend.requests.size() == 1, "local backend request mismatch");
    require(
        inter_node_backend.requests.empty(),
        "inter-node backend should not receive CUDA peer route");
    require(
        *std::static_pointer_cast<std::vector<int>>(copied) == *source,
        "local routed copy result mismatch");
}

void test_inter_node_planned_route_uses_inter_node_backend()
{
    const MgpuCopyRoute inter_node =
        route(20, 21, 1, 4, MgpuTransportKind::InterNode);
    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;
    RoutedGpuObjectCopyBackend routed_backend(
        make_uniform_cluster_topology(2, 4), local_backend, inter_node_backend);
    PlannedMaterializedGpuComm comm(
        plan_with_route(inter_node), materializer, routed_backend);

    auto source =
        std::make_shared<std::vector<int>>(std::initializer_list<int>{ 4, 5 });
    const std::shared_ptr<void> copied =
        comm.copy(request_from_route(inter_node, source));

    require(local_backend.requests.empty(), "local backend should not run");
    require(
        inter_node_backend.requests.size() == 1,
        "inter-node backend request mismatch");
    require(
        inter_node_backend.requests[0].source.node_id == 0,
        "inter-node source node mismatch");
    require(
        inter_node_backend.requests[0].destination.node_id == 1,
        "inter-node destination node mismatch");
    require(
        *std::static_pointer_cast<std::vector<int>>(copied) == *source,
        "inter-node routed copy result mismatch");
}

void test_missing_planned_route_fails_before_materialization()
{
    const MgpuCopyRoute cuda_peer =
        route(30, 31, 1, 2, MgpuTransportKind::CudaPeer);
    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;
    RoutedGpuObjectCopyBackend routed_backend(
        make_uniform_cluster_topology(2, 4), local_backend, inter_node_backend);
    PlannedMaterializedGpuComm comm(
        plan_with_route(cuda_peer), materializer, routed_backend);

    bool failed = false;
    try
    {
        (void)comm.copy(GpuCommCopyRequest{
            99,
            100,
            MgpuValueKind::Ciphertext,
            1,
            2,
            std::make_shared<std::vector<int>>(1, 7),
        });
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(ex.what(), "no planned communication route");
    }

    require(failed, "missing route should fail");
    require(
        materializer.requests.empty(),
        "missing route should fail before materialization");
    require(local_backend.requests.empty(), "local backend should not run");
    require(
        inter_node_backend.requests.empty(),
        "inter-node backend should not run");
}

void test_request_route_device_mismatch_fails_before_materialization()
{
    const MgpuCopyRoute cuda_peer =
        route(40, 41, 1, 2, MgpuTransportKind::CudaPeer);
    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;
    RoutedGpuObjectCopyBackend routed_backend(
        make_uniform_cluster_topology(2, 4), local_backend, inter_node_backend);
    PlannedMaterializedGpuComm comm(
        plan_with_route(cuda_peer), materializer, routed_backend);

    GpuCommCopyRequest request =
        request_from_route(cuda_peer, std::make_shared<std::vector<int>>(1, 7));
    request.destination_device = 3;

    bool failed = false;
    try
    {
        (void)comm.copy(request);
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(ex.what(), "route devices do not match copy request");
    }

    require(failed, "device mismatch should fail");
    require(
        materializer.requests.empty(),
        "device mismatch should fail before materialization");
    require(local_backend.requests.empty(), "local backend should not run");
    require(
        inter_node_backend.requests.empty(),
        "inter-node backend should not run");
}

void test_null_source_object_fails_before_materialization()
{
    const MgpuCopyRoute cuda_peer =
        route(45, 46, 1, 2, MgpuTransportKind::CudaPeer);
    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;
    RoutedGpuObjectCopyBackend routed_backend(
        make_uniform_cluster_topology(2, 4), local_backend, inter_node_backend);
    PlannedMaterializedGpuComm comm(
        plan_with_route(cuda_peer), materializer, routed_backend);

    GpuCommCopyRequest request;
    request.source_id = cuda_peer.source_id;
    request.destination_id = cuda_peer.destination_id;
    request.kind = cuda_peer.kind;
    request.source_device = cuda_peer.source_device;
    request.destination_device = cuda_peer.destination_device;

    bool failed = false;
    try
    {
        (void)comm.copy(request);
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(ex.what(), "source object is null");
    }

    require(failed, "null source object should fail");
    require(
        materializer.requests.empty(),
        "null source object should fail before materialization");
    require(local_backend.requests.empty(), "local backend should not run");
    require(
        inter_node_backend.requests.empty(),
        "inter-node backend should not run");
}

void test_invalid_plan_is_rejected()
{
    MgpuCommunicationPlan plan;
    plan.diagnostics.push_back(
        MgpuCommunicationPlanDiagnostic{ 7, "mock communication failure" });

    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;
    RoutedGpuObjectCopyBackend routed_backend(
        make_uniform_cluster_topology(2, 4), local_backend, inter_node_backend);

    bool failed = false;
    try
    {
        PlannedMaterializedGpuComm comm(plan, materializer, routed_backend);
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(ex.what(), "requires a valid communication plan");
        require_contains(ex.what(), "mock communication failure");
    }
    require(failed, "invalid plan should fail");
}

void test_duplicate_planned_route_is_rejected()
{
    const MgpuCopyRoute cuda_peer =
        route(50, 51, 1, 2, MgpuTransportKind::CudaPeer);
    MgpuCommunicationPlan plan;
    plan.routes.push_back(cuda_peer);
    plan.routes.push_back(cuda_peer);

    VectorCopyMaterializer materializer;
    CopyingLocalBackend local_backend;
    CopyingInterNodeBackend inter_node_backend;
    RoutedGpuObjectCopyBackend routed_backend(
        make_uniform_cluster_topology(2, 4), local_backend, inter_node_backend);

    bool failed = false;
    try
    {
        PlannedMaterializedGpuComm comm(plan, materializer, routed_backend);
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(ex.what(), "duplicate planned communication route");
    }
    require(failed, "duplicate planned route should fail");
}

}  // namespace

int main()
{
    try
    {
        test_local_planned_route_uses_local_backend();
        test_inter_node_planned_route_uses_inter_node_backend();
        test_missing_planned_route_fails_before_materialization();
        test_request_route_device_mismatch_fails_before_materialization();
        test_null_source_object_fails_before_materialization();
        test_invalid_plan_is_rejected();
        test_duplicate_planned_route_is_rejected();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu planned materialized comm test failed: "
                  << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu planned materialized comm tests passed\n";
    return EXIT_SUCCESS;
}
