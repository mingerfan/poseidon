#include "poseidon/mgpu/compiler/static_schedule_config.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

std::string read_text_file(const std::filesystem::path &path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open config template: " + path.string());
    }

    std::string result(
        (std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    if (stream.bad())
    {
        throw std::runtime_error("failed to read config template: " + path.string());
    }
    return result;
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

void test_config_template(
    const std::filesystem::path &config_dir, const std::string &filename,
    int expected_device_count, int expected_node_count, int expected_devices_per_node,
    bool expected_cuda_peer, bool expected_inter_node, bool expected_require_ready)
{
    const std::filesystem::path path = config_dir / filename;
    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(read_text_file(path));
    require(result.ok(), filename + " should parse:\n" + result.format_diagnostics());

    const StaticScheduleExecutionConfig &config = result.config;
    require(
        config.pipeline.device_count == expected_device_count,
        filename + " device_count mismatch");
    require(config.node_count == expected_node_count, filename + " node_count mismatch");
    require(
        config.devices_per_node == expected_devices_per_node,
        filename + " devices_per_node mismatch");
    require(config.opcode_summary, filename + " should enable opcode summary");
    require(
        config.poseidon_gpu_preflight,
        filename + " should enable Poseidon GPU preflight");
    require(config.communication_plan, filename + " should enable communication plan");
    require(
        config.communication_execution_preflight,
        filename + " should enable communication execution preflight");
    require(
        config.communication_execution.same_device_available,
        filename + " should declare same-device execution");
    require(
        config.communication_execution.cuda_peer_available == expected_cuda_peer,
        filename + " CUDA peer backend mismatch");
    require(
        config.communication_execution.inter_node_available == expected_inter_node,
        filename + " inter-node backend mismatch");
    require(
        config.require_ready == expected_require_ready,
        filename + " require_ready mismatch");

    const MgpuTopology topology = make_static_schedule_execution_topology(config);
    require(
        topology.devices.size() == static_cast<std::size_t>(expected_device_count),
        filename + " topology device count mismatch");
    require(topology.devices.front().node_id == 0, filename + " first node mismatch");
    require(
        topology.devices.back().node_id == expected_node_count - 1,
        filename + " last node mismatch");
    require(
        topology.devices.back().local_device == expected_devices_per_node - 1,
        filename + " last local device mismatch");
}

void test_bundled_config_templates(const std::filesystem::path &config_dir)
{
    require(
        std::filesystem::is_directory(config_dir),
        "expected config template directory: " + config_dir.string());

    test_config_template(
        config_dir, "single_gpu.json", 1, 1, 1, false, false, true);
    test_config_template(
        config_dir, "single_node_8gpu.json", 8, 1, 8, true, false, true);
    test_config_template(
        config_dir, "cluster_4x8_preview.json", 32, 4, 8, true, false, false);
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        test_parse_single_node_eight_gpu_config();
        test_parse_cluster_preview_config();
        test_require_ready_enables_hard_gate_checks();
        test_config_json_round_trip();
        test_config_diagnostics();
        if (argc > 1)
        {
            test_bundled_config_templates(argv[1]);
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu static schedule config test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu static schedule config tests passed\n";
    return EXIT_SUCCESS;
}
