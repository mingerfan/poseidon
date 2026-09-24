#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace poseidon::benchmark::s2c_first
{
using GpuTimeInterval = std::pair<std::uint64_t, std::uint64_t>;

inline std::uint64_t gpu_interval_union_ns(std::vector<GpuTimeInterval> intervals)
{
    for (const auto &[start, end] : intervals)
        if (!start || end <= start)
            throw std::invalid_argument("invalid GPU activity timestamps");
    std::sort(intervals.begin(), intervals.end());
    std::uint64_t total = 0, covered_end = 0;
    for (const auto &[start, end] : intervals)
    {
        if (end > covered_end)
            total += end - std::max(start, covered_end);
        covered_end = std::max(covered_end, end);
    }
    return total;
}

enum class GpuActivityType
{
    kernel,
    device_copy,
    memset
};

struct GpuTimedActivity
{
    std::uint64_t start, end, stage;
    std::uint32_t device;
    GpuActivityType type;
};

struct GpuActivityUnion
{
    std::uint64_t total_ns = 0;
    // Stage unions are attribution diagnostics only. They are not additive
    // when stages overlap.
    std::map<std::uint64_t, std::uint64_t> stage_ns;
    std::map<std::uint64_t, std::uint64_t> stage_kernel_ns;
    std::map<std::uint64_t, std::uint64_t> stage_device_copy_ns;
    std::map<std::uint64_t, std::uint64_t> stage_memset_ns;
};

inline GpuActivityUnion summarize_gpu_activity(
    const std::vector<GpuTimedActivity> &activities)
{
    if (activities.empty())
        throw std::invalid_argument(
            "CUPTI recorded no GPU activities; refusing timing fallback");
    std::vector<GpuTimeInterval> all;
    std::map<std::uint64_t, std::vector<GpuTimeInterval>> by_stage;
    std::map<std::uint64_t, std::map<GpuActivityType, std::vector<GpuTimeInterval>>>
        by_stage_type;
    all.reserve(activities.size());
    for (const auto &activity : activities)
    {
        if (activity.device != activities.front().device)
            throw std::invalid_argument(
                "strict single-GPU timing recorded multiple devices");
        if (!activity.stage)
            throw std::invalid_argument(
                "GPU activity has no measured-stage attribution");
        switch (activity.type)
        {
        case GpuActivityType::kernel:
        case GpuActivityType::device_copy:
        case GpuActivityType::memset:
            break;
        default:
            throw std::invalid_argument("unsupported GPU activity type");
        }
        all.emplace_back(activity.start, activity.end);
        by_stage[activity.stage].emplace_back(activity.start, activity.end);
        by_stage_type[activity.stage][activity.type].emplace_back(
            activity.start, activity.end);
    }
    GpuActivityUnion result;
    result.total_ns = gpu_interval_union_ns(std::move(all));
    for (auto &[stage, intervals] : by_stage)
        result.stage_ns[stage] = gpu_interval_union_ns(std::move(intervals));
    for (auto &[stage, by_type] : by_stage_type)
    {
        result.stage_kernel_ns[stage] =
            gpu_interval_union_ns(std::move(by_type[GpuActivityType::kernel]));
        result.stage_device_copy_ns[stage] =
            gpu_interval_union_ns(std::move(by_type[GpuActivityType::device_copy]));
        result.stage_memset_ns[stage] =
            gpu_interval_union_ns(std::move(by_type[GpuActivityType::memset]));
    }
    return result;
}
} // namespace poseidon::benchmark::s2c_first
