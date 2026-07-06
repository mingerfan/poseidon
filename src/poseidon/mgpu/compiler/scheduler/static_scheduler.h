#pragma once

#include "poseidon/mgpu/compiler/copy_insertion.h"
#include "poseidon/mgpu/compiler/scheduler/latency_table.h"
#include "poseidon/mgpu/compiler/static_placement.h"
#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

enum class StaticSchedulerKind
{
    SingleDevice,
    GreedyReady,
    ValueAwareHeft,
    ValueAwarePeft
};

struct GreedyReadySchedulerOptions
{
    double default_copy_latency = 1.0e-4;
};

struct StaticSchedulerOptions
{
    StaticSchedulerKind kind = StaticSchedulerKind::SingleDevice;
    int device_count = 1;
    int default_device = 0;
    std::vector<int> compute_devices;
    std::optional<int> upload_device;
    std::optional<int> download_device;
    bool preserve_existing_devices = true;
    CopyInsertionOptions copy_insertion;
    LatencyTable latency_table = make_default_latency_table();
    GreedyReadySchedulerOptions greedy_ready;
};

struct StaticSchedulePreflight
{
    double estimated_makespan = 0.0;
    double single_device_baseline = 0.0;
    double estimated_speedup = 0.0;
    double critical_path_lower_bound = 0.0;
    double work_lower_bound = 0.0;
    bool parallelism_found = false;
    std::size_t ciphertext_copy_count = 0;
    std::size_t plaintext_preload_count = 0;
    std::vector<double> compute_time_per_gpu;
};

struct StaticSchedulingDiagnostic
{
    std::size_t op_index = 0;
    std::string message;
};

struct StaticSchedulingResult
{
    MgpuSchedule schedule;
    StaticSchedulePreflight preflight;
    std::vector<StaticSchedulingDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

const char *to_string(StaticSchedulerKind kind) noexcept;
std::optional<StaticSchedulerKind> static_scheduler_kind_from_string(
    std::string_view name) noexcept;

StaticSchedulingResult schedule_static(
    const MgpuSchedule &schedule, const StaticSchedulerOptions &options = {});

}  // namespace poseidon::mgpu
