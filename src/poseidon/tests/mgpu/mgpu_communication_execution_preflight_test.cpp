#include "poseidon/mgpu/comm/execution_preflight.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

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
        throw std::runtime_error("expected text to contain: " + needle + "\ntext:\n" + text);
    }
}

MgpuCopyRoute route(int source_device, int destination_device, MgpuTransportKind transport)
{
    return MgpuCopyRoute{
        1,
        2,
        MgpuValueKind::Ciphertext,
        source_device,
        destination_device,
        transport,
    };
}

MgpuCommunicationPlan make_mixed_plan()
{
    MgpuCommunicationPlan plan;
    plan.routes.push_back(route(0, 0, MgpuTransportKind::SameDevice));
    plan.routes.push_back(route(0, 1, MgpuTransportKind::CudaPeer));
    plan.routes.push_back(route(1, 8, MgpuTransportKind::InterNode));
    plan.same_device_copies = 1;
    plan.cuda_peer_copies = 1;
    plan.inter_node_copies = 1;
    return plan;
}

void test_same_device_only_defaults()
{
    const MgpuCommunicationExecutionPreflight preflight =
        preflight_communication_execution(make_mixed_plan());

    require(!preflight.ok(), "mixed plan should fail with default execution options");
    require(preflight.same_device_routes == 1, "same-device route count mismatch");
    require(preflight.cuda_peer_routes == 1, "cuda peer route count mismatch");
    require(preflight.inter_node_routes == 1, "inter-node route count mismatch");
    require(preflight.diagnostics.size() == 2, "expected cuda and inter-node diagnostics");
    require_contains(preflight.format_diagnostics(), "CUDA peer");
    require_contains(preflight.format_diagnostics(), "inter-node communication backend");
}

void test_single_node_backend_accepts_cuda_peer()
{
    MgpuCommunicationExecutionOptions options;
    options.cuda_peer_available = true;
    const MgpuCommunicationExecutionPreflight preflight =
        preflight_communication_execution(make_mixed_plan(), options);

    require(!preflight.ok(), "single-node backend should still reject inter-node routes");
    require(preflight.diagnostics.size() == 1, "expected one inter-node diagnostic");
    require_contains(preflight.format_diagnostics(), "inter-node communication backend");
}

void test_cluster_backend_accepts_all_routes()
{
    MgpuCommunicationExecutionOptions options;
    options.cuda_peer_available = true;
    options.inter_node_available = true;
    const MgpuCommunicationExecutionPreflight preflight =
        preflight_communication_execution(make_mixed_plan(), options);

    require(preflight.ok(), "cluster-capable backend should accept all routes");

    const std::string text = dump_communication_execution_preflight(preflight);
    require_contains(text, "status: ok");
    require_contains(text, "inter_node_routes: 1");

    const std::string json = communication_execution_preflight_to_json(preflight);
    require_contains(json, "\"ok\": true");
    require_contains(json, "\"inter_node\": 1");
}

}  // namespace

int main()
{
    try
    {
        test_same_device_only_defaults();
        test_single_node_backend_accepts_cuda_peer();
        test_cluster_backend_accepts_all_routes();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu communication execution preflight test failed: "
                  << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu communication execution preflight tests passed\n";
    return EXIT_SUCCESS;
}
