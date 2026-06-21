#include "poseidon/mgpu/comm/routed_object_copy.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
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

class RecordingLocalBackend final : public GpuObjectCopyBackend
{
public:
    void copy_object(const GpuObjectCopyRequest &request) override
    {
        requests.push_back(request);
    }

    std::vector<GpuObjectCopyRequest> requests;
};

class RecordingInterNodeBackend final : public InterNodeTransportBackend
{
public:
    void copy_object(const InterNodeObjectCopyRequest &request) override
    {
        requests.push_back(request);
    }

    std::vector<InterNodeObjectCopyRequest> requests;
};

MgpuCopyRoute route(MgpuTransportKind transport)
{
    MgpuCopyRoute copy_route;
    copy_route.source_id = 10;
    copy_route.destination_id = 11;
    copy_route.kind = MgpuValueKind::Ciphertext;
    copy_route.source_device = 1;
    copy_route.destination_device =
        transport == MgpuTransportKind::InterNode ? 4 : 2;
    copy_route.transport = transport;
    if (transport == MgpuTransportKind::SameDevice)
    {
        copy_route.destination_device = copy_route.source_device;
    }
    return copy_route;
}

GpuObjectCopyRequest object_copy(const MgpuCopyRoute &copy_route)
{
    static int source[] = { 1, 2, 3, 4 };
    static int destination[] = { 0, 0, 0, 0 };

    GpuObjectCopyRequest request;
    request.source_id = copy_route.source_id;
    request.destination_id = copy_route.destination_id;
    request.kind = copy_route.kind;
    request.buffers.push_back(GpuObjectBufferCopy{
        source,
        destination,
        sizeof(source),
        copy_route.source_device,
        copy_route.destination_device,
    });
    return request;
}

void test_local_routes_use_local_backend()
{
    RecordingLocalBackend local_backend;
    RecordingInterNodeBackend inter_node_backend;
    RoutedGpuObjectCopyBackend router(
        make_uniform_cluster_topology(2, 4), local_backend, inter_node_backend);

    const MgpuCopyRoute same_device = route(MgpuTransportKind::SameDevice);
    router.copy_object(same_device, object_copy(same_device));

    const MgpuCopyRoute cuda_peer = route(MgpuTransportKind::CudaPeer);
    router.copy_object(cuda_peer, object_copy(cuda_peer));

    require(
        local_backend.requests.size() == 2,
        "local backend request count mismatch");
    require(
        inter_node_backend.requests.empty(),
        "inter-node backend should not receive local routes");
    require(
        local_backend.requests[0].buffers[0].source_device == 1,
        "same-device source mismatch");
    require(
        local_backend.requests[1].buffers[0].destination_device == 2,
        "cuda-peer destination mismatch");
}

void test_inter_node_route_uses_inter_node_backend()
{
    RecordingLocalBackend local_backend;
    RecordingInterNodeBackend inter_node_backend;
    RoutedGpuObjectCopyBackend router(
        make_uniform_cluster_topology(2, 4), local_backend, inter_node_backend);

    const MgpuCopyRoute inter_node = route(MgpuTransportKind::InterNode);
    router.copy_object(inter_node, object_copy(inter_node));

    require(
        local_backend.requests.empty(),
        "local backend should not receive inter-node route");
    require(
        inter_node_backend.requests.size() == 1,
        "inter-node backend request count mismatch");
    require(
        inter_node_backend.requests[0].source.node_id == 0,
        "inter-node source node mismatch");
    require(
        inter_node_backend.requests[0].destination.node_id == 1,
        "inter-node destination node mismatch");
    require(
        inter_node_backend.requests[0].object_copy.buffers.size() == 1,
        "inter-node backend should receive the full-object buffer");
}

void test_mismatched_route_and_request_fails_before_backend()
{
    RecordingLocalBackend local_backend;
    RecordingInterNodeBackend inter_node_backend;
    RoutedGpuObjectCopyBackend router(
        make_uniform_cluster_topology(2, 4), local_backend, inter_node_backend);

    const MgpuCopyRoute cuda_peer = route(MgpuTransportKind::CudaPeer);
    GpuObjectCopyRequest request = object_copy(cuda_peer);
    request.buffers[0].destination_device = 3;

    bool failed = false;
    try
    {
        router.copy_object(cuda_peer, request);
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(ex.what(), "buffer devices do not match copy route");
    }
    require(failed, "mismatched route and request should fail");
    require(
        local_backend.requests.empty(),
        "local backend should not run after validation failure");
    require(
        inter_node_backend.requests.empty(),
        "inter-node backend should not run after validation failure");
}

void test_cuda_peer_route_across_nodes_fails_before_backend()
{
    RecordingLocalBackend local_backend;
    RecordingInterNodeBackend inter_node_backend;
    RoutedGpuObjectCopyBackend router(
        make_uniform_cluster_topology(2, 4), local_backend, inter_node_backend);

    MgpuCopyRoute cuda_peer = route(MgpuTransportKind::CudaPeer);
    cuda_peer.destination_device = 4;

    bool failed = false;
    try
    {
        router.copy_object(cuda_peer, object_copy(cuda_peer));
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(
            ex.what(),
            "CUDA peer copy route endpoints are on different nodes");
    }

    require(failed, "cross-node CUDA peer route should fail");
    require(local_backend.requests.empty(), "local backend should not run");
    require(
        inter_node_backend.requests.empty(),
        "inter-node backend should not run");
}

void test_inter_node_route_inside_node_fails_before_backend()
{
    RecordingLocalBackend local_backend;
    RecordingInterNodeBackend inter_node_backend;
    RoutedGpuObjectCopyBackend router(
        make_uniform_cluster_topology(2, 4), local_backend, inter_node_backend);

    MgpuCopyRoute inter_node = route(MgpuTransportKind::InterNode);
    inter_node.destination_device = 2;

    bool failed = false;
    try
    {
        router.copy_object(inter_node, object_copy(inter_node));
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(
            ex.what(),
            "inter-node copy route endpoints are on the same node");
    }

    require(failed, "same-node inter-node route should fail");
    require(local_backend.requests.empty(), "local backend should not run");
    require(
        inter_node_backend.requests.empty(),
        "inter-node backend should not run");
}

}  // namespace

int main()
{
    try
    {
        test_local_routes_use_local_backend();
        test_inter_node_route_uses_inter_node_backend();
        test_mismatched_route_and_request_fails_before_backend();
        test_cuda_peer_route_across_nodes_fails_before_backend();
        test_inter_node_route_inside_node_fails_before_backend();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu routed object-copy test failed: " << ex.what()
                  << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu routed object-copy tests passed\n";
    return EXIT_SUCCESS;
}
