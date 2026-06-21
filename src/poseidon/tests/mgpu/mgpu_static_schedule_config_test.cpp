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

void test_single_node_zero_devices_per_node_defaults_to_device_count()
{
    const char *json = R"json(
{
  "version": 1,
  "device_count": 4,
  "topology": {
    "nodes": 1,
    "devices_per_node": 0
  },
  "preflight": {
    "communication_plan": true,
    "communication_execution": true
  }
}
)json";

    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(json);
    require(
        result.ok(),
        "single-node config with implicit devices_per_node should parse:\n" +
            result.format_diagnostics());

    const MgpuTopology topology = make_static_schedule_execution_topology(result.config);
    require(topology.devices.size() == 4, "implicit single-node topology size mismatch");
    require(topology.devices[0].node_id == 0, "implicit topology first node mismatch");
    require(topology.devices[3].logical_device == 3, "implicit topology logical id mismatch");
    require(topology.devices[3].local_device == 3, "implicit topology local id mismatch");
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

void test_config_json_round_trip_preserves_unset_optional_devices()
{
    StaticScheduleExecutionConfig config;
    config.pipeline.device_count = 3;
    config.pipeline.placement.default_device = 1;
    config.pipeline.placement.policy = StaticPlacementPolicy::RoundRobinCompute;
    config.pipeline.placement.compute_devices = { 1, 2 };

    const std::string json = static_schedule_execution_config_to_json(config);
    require_contains(json, "\"upload_device\": null");
    require_contains(json, "\"download_device\": null");

    const StaticScheduleExecutionConfigParseResult parsed =
        parse_static_schedule_execution_config_json(json);
    require(
        parsed.ok(),
        "round-trip config with null optional devices should parse:\n" +
            parsed.format_diagnostics());
    require(
        !parsed.config.pipeline.placement.upload_device.has_value(),
        "round-trip upload_device should remain unset");
    require(
        !parsed.config.pipeline.placement.download_device.has_value(),
        "round-trip download_device should remain unset");
    require(
        parsed.config.pipeline.placement.policy ==
            StaticPlacementPolicy::RoundRobinCompute,
        "round-trip compute_devices should imply round-robin placement");
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

void test_config_reports_json_shape_errors()
{
    const StaticScheduleExecutionConfigParseResult non_object =
        parse_static_schedule_execution_config_json("[1, 2]");
    require(!non_object.ok(), "non-object config root should fail");
    require_contains(non_object.format_diagnostics(), "/: expected an object");

    const StaticScheduleExecutionConfigParseResult malformed =
        parse_static_schedule_execution_config_json("{");
    require(!malformed.ok(), "malformed config JSON should fail");
    require_contains(malformed.format_diagnostics(), "/:");

    const char *json = R"json(
{
  "version": 1,
  "device_count": 2,
  "placement": [],
  "topology": [],
  "preflight": [],
  "execution_backends": []
}
)json";

    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(json);
    require(!result.ok(), "object-valued config sections should reject arrays");
    require_contains(result.format_diagnostics(), "/placement: expected an object");
    require_contains(result.format_diagnostics(), "/topology: expected an object");
    require_contains(result.format_diagnostics(), "/preflight: expected an object");
    require_contains(
        result.format_diagnostics(),
        "/execution_backends: expected an object");
}

void test_config_reports_scalar_type_errors()
{
    const char *json = R"json(
{
  "version": true,
  "device_count": "8",
  "emit_debug_dump": 1,
  "placement": {
    "policy": false,
    "default_device": null,
    "upload_device": "0",
    "download_device": false,
    "compute_devices": [0, "1"]
  },
  "topology": {
    "nodes": "1",
    "devices_per_node": false
  },
  "preflight": {
    "opcode_summary": 1
  },
  "execution_backends": {
    "same_device": 1
  }
}
)json";

    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(json);
    require(!result.ok(), "scalar type mismatches should fail");
    require_contains(result.format_diagnostics(), "/version: expected an integer");
    require_contains(result.format_diagnostics(), "/device_count: expected an integer");
    require_contains(result.format_diagnostics(), "/emit_debug_dump: expected a boolean");
    require_contains(result.format_diagnostics(), "/placement/policy: expected a string");
    require_contains(result.format_diagnostics(), "/placement/default_device: expected an integer");
    require_contains(
        result.format_diagnostics(),
        "/placement/upload_device: expected an integer or null");
    require_contains(
        result.format_diagnostics(),
        "/placement/download_device: expected an integer or null");
    require_contains(
        result.format_diagnostics(),
        "/placement/compute_devices/1: expected an integer");
    require_contains(result.format_diagnostics(), "/topology/nodes: expected an integer");
    require_contains(
        result.format_diagnostics(),
        "/topology/devices_per_node: expected an integer");
    require_contains(
        result.format_diagnostics(),
        "/preflight/opcode_summary: expected a boolean");
    require_contains(
        result.format_diagnostics(),
        "/execution_backends/same_device: expected a boolean");
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
    test_config_template(
        config_dir, "cluster_4x8_node_spread_preview.json", 32, 4, 8, true,
        false, false);
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        test_parse_single_node_eight_gpu_config();
        test_parse_cluster_preview_config();
        test_require_ready_enables_hard_gate_checks();
        test_single_node_zero_devices_per_node_defaults_to_device_count();
        test_config_json_round_trip();
        test_config_json_round_trip_preserves_unset_optional_devices();
        test_config_diagnostics();
        test_config_reports_json_shape_errors();
        test_config_reports_scalar_type_errors();
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
