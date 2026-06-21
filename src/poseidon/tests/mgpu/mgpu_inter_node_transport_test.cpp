#include "poseidon/mgpu/comm/inter_node_transport.h"

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

MgpuCopyRoute route(MgpuTransportKind transport)
{
    MgpuCopyRoute copy_route;
    copy_route.source_id = 10;
    copy_route.destination_id = 11;
    copy_route.kind = MgpuValueKind::Ciphertext;
    copy_route.source_device = 1;
    copy_route.destination_device = 4;
    copy_route.transport = transport;
    return copy_route;
}

GpuObjectCopyRequest object_copy()
{
    static int source[] = { 1, 2, 3, 4 };
    static int destination[] = { 0, 0, 0, 0 };

    GpuObjectCopyRequest request;
    request.source_id = 10;
    request.destination_id = 11;
    request.kind = MgpuValueKind::Ciphertext;
    request.buffers.push_back(GpuObjectBufferCopy{
        source,
        destination,
        sizeof(source),
        1,
        4,
    });
    return request;
}

void test_builds_inter_node_request_from_route_and_topology()
{
    const MgpuTopology topology = make_uniform_cluster_topology(2, 4);
    const InterNodeObjectCopyRequest request =
        make_inter_node_object_copy_request(
            route(MgpuTransportKind::InterNode), object_copy(), topology);

    require(request.source_id == 10, "source id mismatch");
    require(request.destination_id == 11, "destination id mismatch");
    require(request.kind == MgpuValueKind::Ciphertext, "kind mismatch");
    require(request.source.logical_device == 1, "source logical device mismatch");
    require(request.source.node_id == 0, "source node mismatch");
    require(request.source.local_device == 1, "source local device mismatch");
    require(
        request.destination.logical_device == 4,
        "destination logical device mismatch");
    require(request.destination.node_id == 1, "destination node mismatch");
    require(
        request.destination.local_device == 0,
        "destination local device mismatch");
    require(
        request.object_copy.buffers.size() == 1,
        "inter-node copy should retain full-object buffer");
}

void test_rejects_non_inter_node_route()
{
    bool failed = false;
    try
    {
        (void)make_inter_node_object_copy_request(
            route(MgpuTransportKind::CudaPeer), object_copy(),
            make_uniform_cluster_topology(2, 4));
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(ex.what(), "requires an inter-node copy route");
    }
    require(failed, "non-inter-node route should fail");
}

void test_rejects_mismatched_object_copy()
{
    GpuObjectCopyRequest copy = object_copy();
    copy.buffers[0].destination_device = 5;

    bool failed = false;
    try
    {
        (void)make_inter_node_object_copy_request(
            route(MgpuTransportKind::InterNode), copy,
            make_uniform_cluster_topology(2, 4));
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(ex.what(), "buffer devices do not match route devices");
    }
    require(failed, "mismatched object copy should fail");
}

void test_missing_backend_fails_clearly()
{
    const InterNodeObjectCopyRequest request =
        make_inter_node_object_copy_request(
            route(MgpuTransportKind::InterNode), object_copy(),
            make_uniform_cluster_topology(2, 4));

    MissingInterNodeTransportBackend backend;
    bool failed = false;
    try
    {
        backend.copy_object(request);
    }
    catch (const std::runtime_error &ex)
    {
        failed = true;
        require_contains(ex.what(), "inter-node communication backend is not configured");
        require_contains(ex.what(), "device 1 -> 4");
    }
    require(failed, "missing inter-node backend should fail");
}

}  // namespace

int main()
{
    try
    {
        test_builds_inter_node_request_from_route_and_topology();
        test_rejects_non_inter_node_route();
        test_rejects_mismatched_object_copy();
        test_missing_backend_fails_clearly();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu inter-node transport test failed: " << ex.what()
                  << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu inter-node transport tests passed\n";
    return EXIT_SUCCESS;
}
