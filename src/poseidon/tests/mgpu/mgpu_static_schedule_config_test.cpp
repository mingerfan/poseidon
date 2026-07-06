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

void test_parse_scheduler_config()
{
    const char *json = R"json(
{
  "version": 1,
  "device_count": 8,
  "scheduler": {
    "kind": "greedy_ready",
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
    require(result.ok(), "scheduler config should parse:\n" + result.format_diagnostics());

    const StaticScheduleExecutionConfig &config = result.config;
    require(config.pipeline.device_count == 8, "device_count mismatch");
    require(config.pipeline.scheduler.kind == StaticSchedulerKind::GreedyReady,
            "scheduler kind mismatch");
    require(config.pipeline.scheduler.compute_devices.size() == 8,
            "compute_devices size mismatch");
    require(config.require_ready, "require_ready should be set");
    require(config.communication_execution.cuda_peer_available, "CUDA peer backend mismatch");

    const MgpuTopology topology = make_static_schedule_execution_topology(config);
    require(topology.devices.size() == 8, "topology device count mismatch");
}

void test_legacy_single_device_placement_fields_parse()
{
    const char *json = R"json(
{
  "version": 1,
  "device_count": 2,
  "placement": {
    "policy": "single_device",
    "default_device": 1,
    "upload_device": 0,
    "download_device": 1
  }
}
)json";

    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(json);
    require(result.ok(), "legacy single-device placement should parse:\n" + result.format_diagnostics());
    require(result.config.pipeline.scheduler.kind == StaticSchedulerKind::SingleDevice,
            "legacy placement should keep single-device scheduler");
    require(result.config.pipeline.scheduler.default_device == 1,
            "legacy default device mismatch");
}

void test_round_robin_config_is_rejected()
{
    const char *json = R"json(
{
  "version": 1,
  "device_count": 2,
  "placement": {
    "policy": "round_robin_compute",
    "compute_devices": [0, 1]
  }
}
)json";

    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(json);
    require(!result.ok(), "round-robin placement should be rejected");
    require_contains(result.format_diagnostics(), "no longer supported");
}

void test_require_ready_enables_hard_gate_checks()
{
    const char *json = R"json(
{
  "version": 1,
  "device_count": 2,
  "scheduler": {
    "kind": "greedy_ready",
    "compute_devices": [0, 1]
  },
  "preflight": {
    "require_ready": true
  }
}
)json";

    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(json);
    require(result.ok(), "require-ready config should parse:\n" + result.format_diagnostics());
    require(result.config.opcode_summary, "require_ready should enable opcode summary");
    require(result.config.poseidon_gpu_preflight, "require_ready should enable GPU preflight");
    require(result.config.communication_plan, "require_ready should enable communication plan");
    require(result.config.communication_execution_preflight,
            "require_ready should enable communication execution preflight");
}

void test_config_json_round_trip()
{
    StaticScheduleExecutionConfig config;
    config.pipeline.device_count = 4;
    config.pipeline.emit_debug_dump = true;
    config.pipeline.scheduler.kind = StaticSchedulerKind::GreedyReady;
    config.pipeline.scheduler.upload_device = 0;
    config.pipeline.scheduler.compute_devices = { 1, 2 };
    config.pipeline.scheduler.download_device = 0;
    config.node_count = 2;
    config.devices_per_node = 2;
    config.poseidon_gpu_preflight = true;
    config.preflight_comm_available = true;
    config.communication_plan = true;
    config.communication_execution_preflight = true;
    config.communication_execution.cuda_peer_available = true;

    const std::string json = static_schedule_execution_config_to_json(config);
    require_contains(json, "\"scheduler\"");
    require_contains(json, "\"kind\": \"greedy_ready\"");
    require_contains(json, "\"compute_devices\"");

    const StaticScheduleExecutionConfigParseResult parsed =
        parse_static_schedule_execution_config_json(json);
    require(parsed.ok(), "round-trip config should parse:\n" + parsed.format_diagnostics());
    require(parsed.config.pipeline.scheduler.kind == StaticSchedulerKind::GreedyReady,
            "round-trip scheduler kind mismatch");
    require(parsed.config.pipeline.scheduler.compute_devices.size() == 2,
            "round-trip compute_devices mismatch");
}

void test_config_diagnostics()
{
    const char *json = R"json(
{
  "version": 2,
  "device_count": 2,
  "scheduler": {
    "kind": "dynamic",
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
    require_contains(result.format_diagnostics(), "unknown scheduler kind");
    require_contains(result.format_diagnostics(), "/scheduler/default_device");
    require_contains(result.format_diagnostics(), "duplicate compute device");
    require_contains(result.format_diagnostics(), "/scheduler/compute_devices/2");
    require_contains(result.format_diagnostics(), "/preflight/communication_execution");
    require_contains(result.format_diagnostics(), "/execution_backends/cuda_peer");
    require_contains(result.format_diagnostics(), "fewer logical devices");
}

void test_config_reports_shape_and_type_errors()
{
    const StaticScheduleExecutionConfigParseResult non_object =
        parse_static_schedule_execution_config_json("[1, 2]");
    require(!non_object.ok(), "non-object config root should fail");
    require_contains(non_object.format_diagnostics(), "/: expected an object");

    const char *json = R"json(
{
  "version": true,
  "device_count": "8",
  "emit_debug_dump": 1,
  "scheduler": {
    "kind": false,
    "default_device": null,
    "upload_device": "0",
    "download_device": false,
    "compute_devices": [0, "1"]
  },
  "topology": [],
  "preflight": [],
  "execution_backends": []
}
)json";

    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(json);
    require(!result.ok(), "shape/type mismatches should fail");
    require_contains(result.format_diagnostics(), "/version: expected an integer");
    require_contains(result.format_diagnostics(), "/device_count: expected an integer");
    require_contains(result.format_diagnostics(), "/emit_debug_dump: expected a boolean");
    require_contains(result.format_diagnostics(), "/scheduler/kind: expected a string");
    require_contains(result.format_diagnostics(), "/scheduler/default_device: expected an integer");
    require_contains(result.format_diagnostics(), "/scheduler/upload_device: expected an integer or null");
    require_contains(result.format_diagnostics(), "/scheduler/download_device: expected an integer or null");
    require_contains(result.format_diagnostics(), "/scheduler/compute_devices/1: expected an integer");
    require_contains(result.format_diagnostics(), "/topology: expected an object");
    require_contains(result.format_diagnostics(), "/preflight: expected an object");
    require_contains(result.format_diagnostics(), "/execution_backends: expected an object");
}

void test_config_template(
    const std::filesystem::path &config_dir, const std::string &filename,
    StaticSchedulerKind expected_scheduler, int expected_device_count,
    int expected_node_count, int expected_devices_per_node,
    bool expected_cuda_peer, bool expected_inter_node, bool expected_require_ready)
{
    const std::filesystem::path path = config_dir / filename;
    const StaticScheduleExecutionConfigParseResult result =
        parse_static_schedule_execution_config_json(read_text_file(path));
    require(result.ok(), filename + " should parse:\n" + result.format_diagnostics());

    const StaticScheduleExecutionConfig &config = result.config;
    require(config.pipeline.scheduler.kind == expected_scheduler,
            filename + " scheduler kind mismatch");
    require(config.pipeline.device_count == expected_device_count,
            filename + " device_count mismatch");
    require(config.node_count == expected_node_count, filename + " node_count mismatch");
    require(config.devices_per_node == expected_devices_per_node,
            filename + " devices_per_node mismatch");
    require(config.communication_execution.cuda_peer_available == expected_cuda_peer,
            filename + " CUDA peer backend mismatch");
    require(config.communication_execution.inter_node_available == expected_inter_node,
            filename + " inter-node backend mismatch");
    require(config.require_ready == expected_require_ready,
            filename + " require_ready mismatch");

    const MgpuTopology topology = make_static_schedule_execution_topology(config);
    require(topology.devices.size() == static_cast<std::size_t>(expected_device_count),
            filename + " topology device count mismatch");
}

void test_bundled_config_templates(const std::filesystem::path &config_dir)
{
    require(std::filesystem::is_directory(config_dir),
            "expected config template directory: " + config_dir.string());

    test_config_template(
        config_dir, "single_gpu.json", StaticSchedulerKind::SingleDevice,
        1, 1, 1, false, false, true);
    test_config_template(
        config_dir, "single_node_8gpu.json", StaticSchedulerKind::GreedyReady,
        8, 1, 8, true, false, true);
    test_config_template(
        config_dir, "cluster_4x8_preview.json", StaticSchedulerKind::GreedyReady,
        32, 4, 8, true, false, false);
    test_config_template(
        config_dir, "cluster_4x8_node_spread_preview.json", StaticSchedulerKind::GreedyReady,
        32, 4, 8, true, false, false);
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        test_parse_scheduler_config();
        test_legacy_single_device_placement_fields_parse();
        test_round_robin_config_is_rejected();
        test_require_ready_enables_hard_gate_checks();
        test_config_json_round_trip();
        test_config_diagnostics();
        test_config_reports_shape_and_type_errors();
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
