#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace poseidon::benchmark::s2c_first
{
struct GpuActivityTimingResult
{
    struct Stage
    {
        double gpu_ms = 0;
        double kernel_ms = 0;
        double device_copy_ms = 0;
        double memset_ms = 0;
    };
    std::map<std::string, Stage> stages;
    // Per-stage kernel-name union times.
    std::map<std::string, std::map<std::string, double>> stage_kernels_ms;
    // Union(kernel, memcpy, memset) across all streams and measured stages.
    double gpu_total_ms = 0;
    std::uint64_t gpu_total_ns = 0, activities = 0;
    std::uint64_t host_transfer_bytes = 0, host_transfers = 0;
    void print() const;
};

// Strict CUPTI activity collector. By default, activity collection is enabled
// only between stage(label, true) and stage(label, false), so offline work can
// run between measured stages. begin_continuous() instead keeps collection
// enabled while stage() changes only the external-correlation label; this gives
// an asynchronous breakdown without inserting stage-local synchronization.
// Every captured activity must have an explicit stage, all activity must belong
// to one GPU, no records may be dropped, and H2D/D2H transfers are rejected.
class GpuActivityTiming
{
  public:
    GpuActivityTiming();
    ~GpuActivityTiming();
    GpuActivityTiming(const GpuActivityTiming &) = delete;
    GpuActivityTiming &operator=(const GpuActivityTiming &) = delete;
    void begin_continuous();
    void stage(const std::string &label, bool begin);
    GpuActivityTimingResult finish();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace poseidon::benchmark::s2c_first
