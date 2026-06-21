#include "poseidon/mgpu/compiler/static_schedule_config.h"

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

void test_parse_single_node_eight_gpu_config()
{
    const char *json = R"json(
{
  "version": 1,
  "device_count": 8,
  "placement": {
    "policy": "round_robin_compute",
    "default_device": 0,
    "upload_device": 0,
    "compute_devices": [0, 1, 2, 3, 4, 5, 6, 7],
    "download_device": 0
  },
  "topology": {
    "nodes": 1,
    "devices_per_node": 8
  },
  "preflight": {
    "opcode_summary": true,
    "poseidon_gpu_preflight": true,
    "comm_available": true,
    "relin_keys": true,
    "galois_keys": true,
    "communication_plan": true,
    "communication_execution": true,
    "require_ready": true
  },
  "execution_backends": {
    "same_device": true,
    "cuda_peer": true,
    "inter_node": false
  }
}
)json";

    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(json);
    require(result.ok(), "8-GPU config should parse:\n" + result.format_diagnostics());

    const StaticScheduleExecutionConfig &config = result.config;
    require(config.pipeline.device_count == 8, "device_count mismatch");
    require(
        config.pipeline.placement.policy == StaticPlacementPolicy::RoundRobinCompute,
        "placement policy mismatch");
    require(
        config.pipeline.placement.compute_devices.size() == 8,
        "compute_devices size mismatch");
    require(config.node_count == 1, "node_count mismatch");
    require(config.devices_per_node == 8, "devices_per_node mismatch");
    require(config.opcode_summary, "opcode summary should be enabled");
    require(config.poseidon_gpu_preflight, "GPU preflight should be enabled");
    require(config.communication_plan, "communication plan should be enabled");
    require(
        config.communication_execution_preflight,
        "communication execution preflight should be enabled");
    require(config.communication_execution.cuda_peer_available, "CUDA peer backend mismatch");

    const MgpuTopology topology = make_static_schedule_execution_topology(config);
    require(topology.devices.size() == 8, "topology device count mismatch");
    require(topology.devices[7].local_device == 7, "topology local device mismatch");
}

void test_parse_cluster_preview_config()
{
    const char *json = R"json(
{
  "version": 1,
  "device_count": 32,
  "placement": {
    "default_device": 0,
    "upload_device": 0,
    "compute_devices": [0, 8, 16, 24],
    "download_device": 0
  },
  "topology": {
    "nodes": 4,
    "devices_per_node": 8
  },
  "preflight": {
    "communication_plan": true,
    "communication_execution": true
  },
  "execution_backends": {
    "same_device": true,
    "cuda_peer": true,
    "inter_node": false
  }
}
)json";

    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(json);
    require(result.ok(), "cluster config should parse:\n" + result.format_diagnostics());
    require(
        result.config.pipeline.placement.policy ==
            StaticPlacementPolicy::RoundRobinCompute,
        "compute_devices should imply round-robin placement");

    const MgpuTopology topology = make_static_schedule_execution_topology(result.config);
    require(topology.devices.size() == 32, "cluster topology device count mismatch");
    require(topology.devices[8].node_id == 1, "second node id mismatch");
    require(topology.devices[31].local_device == 7, "last local device mismatch");
}

void test_require_ready_enables_hard_gate_checks()
{
    const char *json = R"json(
{
  "version": 1,
  "device_count": 2,
  "placement": {
    "compute_devices": [1]
  },
  "preflight": {
    "require_ready": true
  }
}
)json";

    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(json);
    require(result.ok(), "require-ready config should parse:\n" + result.format_diagnostics());
    require(result.config.require_ready, "require_ready should be set");
    require(result.config.opcode_summary, "require_ready should enable opcode summary");
    require(
        result.config.poseidon_gpu_preflight,
        "require_ready should enable Poseidon GPU preflight");
    require(
        result.config.communication_plan,
        "require_ready should enable communication planning");
    require(
        result.config.communication_execution_preflight,
        "require_ready should enable communication execution preflight");
}

void test_config_json_round_trip()
{
    StaticScheduleExecutionConfig config;
    config.pipeline.device_count = 4;
    config.pipeline.emit_debug_dump = true;
    config.pipeline.placement.policy = StaticPlacementPolicy::RoundRobinCompute;
    config.pipeline.placement.upload_device = 0;
    config.pipeline.placement.compute_devices = { 1, 2 };
    config.pipeline.placement.download_device = 0;
    config.node_count = 2;
    config.devices_per_node = 2;
    config.poseidon_gpu_preflight = true;
    config.preflight_comm_available = true;
    config.communication_plan = true;
    config.communication_execution_preflight = true;
    config.communication_execution.cuda_peer_available = true;

    const std::string json = static_schedule_execution_config_to_json(config);
    require_contains(json, "\"device_count\": 4");
    require_contains(json, "\"compute_devices\"");
    require_contains(json, "\"cuda_peer\": true");

    const StaticScheduleExecutionConfigParseResult parsed =
        parse_static_schedule_execution_config_json(json);
    require(parsed.ok(), "round-trip config should parse:\n" + parsed.format_diagnostics());
    require(parsed.config.pipeline.device_count == 4, "round-trip device count mismatch");
    require(parsed.config.pipeline.emit_debug_dump, "round-trip debug dump mismatch");
    require(parsed.config.node_count == 2, "round-trip node count mismatch");
    require(
        parsed.config.pipeline.placement.compute_devices.size() == 2,
        "round-trip compute_devices mismatch");
}

void test_config_diagnostics()
{
    const char *json = R"json(
{
  "version": 2,
  "device_count": 2,
  "placement": {
    "policy": "dynamic",
    "default_device": 9,
    "compute_devices": [1, 1, 4]
  },
  "topology": {
    "nodes": 1,
    "devices_per_node": 1
  },
  "preflight": {
    "communication_plan": true,
    "communication_execution": "yes"
  },
  "execution_backends": {
    "cuda_peer": "yes"
  }
}
)json";

    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(json);
    require(!result.ok(), "invalid config should fail");
    require_contains(result.format_diagnostics(), "/version");
    require_contains(result.format_diagnostics(), "unknown placement policy");
    require_contains(result.format_diagnostics(), "/placement/default_device");
    require_contains(result.format_diagnostics(), "duplicate compute device");
    require_contains(result.format_diagnostics(), "/placement/compute_devices/2");
    require_contains(result.format_diagnostics(), "/preflight/communication_execution");
    require_contains(result.format_diagnostics(), "/execution_backends/cuda_peer");
    require_contains(result.format_diagnostics(), "fewer logical devices");
}

}  // namespace

int main()
{
    try
    {
        test_parse_single_node_eight_gpu_config();
        test_parse_cluster_preview_config();
        test_require_ready_enables_hard_gate_checks();
        test_config_json_round_trip();
        test_config_diagnostics();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu static schedule config test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu static schedule config tests passed\n";
    return EXIT_SUCCESS;
}
