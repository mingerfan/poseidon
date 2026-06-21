#pragma once

#include "poseidon/mgpu/comm/execution_preflight.h"
#include "poseidon/mgpu/comm/topology.h"
#include "poseidon/mgpu/compiler/static_schedule_pipeline.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace poseidon::mgpu
{

struct StaticScheduleExecutionConfig
{
    StaticSchedulePipelineOptions pipeline;
    int node_count = 1;
    int devices_per_node = 0;
    bool opcode_summary = false;
    bool poseidon_gpu_preflight = false;
    bool preflight_comm_available = false;
    bool preflight_relin_keys_available = false;
    bool preflight_galois_keys_available = false;
    bool communication_plan = false;
    bool communication_execution_preflight = false;
    MgpuCommunicationExecutionOptions communication_execution;
    bool require_ready = false;
};

struct StaticScheduleExecutionConfigDiagnostic
{
    std::string path;
    std::string message;
};

struct StaticScheduleExecutionConfigParseResult
{
    StaticScheduleExecutionConfig config;
    std::vector<StaticScheduleExecutionConfigDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

StaticScheduleExecutionConfigParseResult parse_static_schedule_execution_config_json(
    std::string_view text);

std::string static_schedule_execution_config_to_json(
    const StaticScheduleExecutionConfig &config, int indent = 2);

MgpuTopology make_static_schedule_execution_topology(
    const StaticScheduleExecutionConfig &config);

}  // namespace poseidon::mgpu
