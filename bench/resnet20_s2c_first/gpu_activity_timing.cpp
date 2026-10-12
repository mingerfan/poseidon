#include "gpu_activity_timing.h"
#include "gpu_activity_intervals.h"

#include <cupti.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace poseidon::benchmark::s2c_first
{
namespace
{
void check(CUptiResult status)
{
    if (status == CUPTI_SUCCESS)
        return;
    const char *message = nullptr;
    cuptiGetResultString(status, &message);
    throw std::runtime_error(
        std::string("strict GPU timing CUPTI: ") +
        (message ? message : "unknown error"));
}

constexpr CUpti_ActivityKind kinds[] = {
    CUPTI_ACTIVITY_KIND_RUNTIME,
    CUPTI_ACTIVITY_KIND_DRIVER,
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
    CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
    CUPTI_ACTIVITY_KIND_MEMCPY,
    CUPTI_ACTIVITY_KIND_MEMSET,
};
} // namespace

class GpuActivityTiming::Impl
{
  public:
    struct Activity
    {
        std::uint64_t start, end, bytes;
        std::uint32_t correlation, device;
        int kind, copy_kind;
    };

    inline static Impl *active = nullptr;
    std::mutex mutex;
    std::vector<Activity> activities;
    std::unordered_map<std::uint32_t, std::uint64_t> correlations;
    std::map<std::string, std::uint64_t> labels;
    std::uint64_t dropped = 0;
    std::string error;
    unsigned enabled = 0;
    bool finished = false;
    bool stage_open = false;
    bool continuous = false;
    std::uint64_t stage_id = 0;

    static void CUPTIAPI request(
        std::uint8_t **buffer, std::size_t *size, std::size_t *records)
    {
        *size = 8 * 1024 * 1024;
        *buffer = static_cast<std::uint8_t *>(std::malloc(*size));
        *records = 0;
        if (!*buffer)
        {
            *size = 0;
            if (active)
            {
                std::lock_guard<std::mutex> lock(active->mutex);
                active->error = "CUPTI activity buffer allocation failed";
            }
        }
    }

    static void CUPTIAPI complete(
        CUcontext context, std::uint32_t stream, std::uint8_t *buffer,
        std::size_t, std::size_t valid)
    {
        if (!active)
        {
            std::free(buffer);
            return;
        }
        std::lock_guard<std::mutex> lock(active->mutex);
        try
        {
            CUpti_Activity *record = nullptr;
            CUptiResult status;
            while ((status = cuptiActivityGetNextRecord(
                        buffer, valid, &record)) == CUPTI_SUCCESS)
            {
                if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION)
                {
                    const auto &activity =
                        *reinterpret_cast<CUpti_ActivityExternalCorrelation *>(record);
                    if (activity.externalKind ==
                        CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0)
                        active->correlations[activity.correlationId] =
                            activity.externalId;
                }
                else if (record->kind ==
                             CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
                         record->kind == CUPTI_ACTIVITY_KIND_KERNEL)
                {
                    // The fields used here are in the stable Kernel4 prefix.
                    const auto &activity =
                        *reinterpret_cast<CUpti_ActivityKernel4 *>(record);
                    active->activities.push_back(
                        {activity.start, activity.end, 0,
                         activity.correlationId, activity.deviceId, 0, 0});
                }
                else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY)
                {
                    const auto &activity =
                        *reinterpret_cast<CUpti_ActivityMemcpy *>(record);
                    active->activities.push_back(
                        {activity.start, activity.end, activity.bytes,
                         activity.correlationId, activity.deviceId, 1,
                         activity.copyKind});
                }
                else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET)
                {
                    const auto &activity =
                        *reinterpret_cast<CUpti_ActivityMemset *>(record);
                    active->activities.push_back(
                        {activity.start, activity.end, activity.bytes,
                         activity.correlationId, activity.deviceId, 2, 0});
                }
            }
            if (status != CUPTI_ERROR_MAX_LIMIT_REACHED)
                check(status);
            std::size_t lost = 0;
            check(cuptiActivityGetNumDroppedRecords(context, stream, &lost));
            active->dropped += lost;
        }
        catch (const std::exception &exception)
        {
            active->error = exception.what();
        }
        catch (...)
        {
            active->error = "activity callback failed";
        }
        std::free(buffer);
    }

    CUptiResult disable() noexcept
    {
        CUptiResult status = CUPTI_SUCCESS;
        while (enabled)
        {
            const auto current = cuptiActivityDisable(kinds[--enabled]);
            if (current != CUPTI_SUCCESS)
                status = current;
        }
        return status;
    }

    CUptiResult stop() noexcept
    {
        CUptiResult status = disable();
        const auto flush = cuptiActivityFlushAll(0);
        if (status == CUPTI_SUCCESS && flush != CUPTI_SUCCESS)
            status = flush;
        active = nullptr;
        return status;
    }
};

GpuActivityTiming::GpuActivityTiming() : impl_(std::make_unique<Impl>())
{
    if (Impl::active)
        throw std::logic_error("nested GPU activity collectors are not supported");
    Impl::active = impl_.get();
    try
    {
        check(cuptiActivityRegisterCallbacks(Impl::request, Impl::complete));
    }
    catch (...)
    {
        impl_->stop();
        throw;
    }
}

GpuActivityTiming::~GpuActivityTiming()
{
    if (!impl_->finished)
        impl_->stop();
}

void GpuActivityTiming::begin_continuous()
{
    if (impl_->finished)
        throw std::logic_error("GPU activity timing already finished");
    if (impl_->continuous || impl_->stage_open || impl_->enabled)
        throw std::logic_error("GPU continuous activity timing already started");
    try
    {
        for (auto kind : kinds)
        {
            check(cuptiActivityEnable(kind));
            ++impl_->enabled;
        }
        impl_->continuous = true;
    }
    catch (...)
    {
        impl_->disable();
        throw;
    }
}

void GpuActivityTiming::stage(const std::string &label, bool begin)
{
    if (impl_->finished)
        throw std::logic_error("GPU activity timing already finished");
    if (begin)
    {
        if (impl_->stage_open)
            throw std::logic_error("nested GPU activity timing stages are not supported");
        auto [iterator, inserted] =
            impl_->labels.try_emplace(label, impl_->labels.size() + 1);
        (void)inserted;
        try
        {
            if (!impl_->continuous)
            {
                for (auto kind : kinds)
                {
                    check(cuptiActivityEnable(kind));
                    ++impl_->enabled;
                }
            }
            check(cuptiActivityPushExternalCorrelationId(
                CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, iterator->second));
            impl_->stage_id = iterator->second;
            impl_->stage_open = true;
        }
        catch (...)
        {
            if (!impl_->continuous)
                impl_->disable();
            throw;
        }
    }
    else
    {
        if (!impl_->stage_open)
            throw std::logic_error("GPU activity timing stage is not open");
        std::uint64_t id = 0;
        check(cuptiActivityPopExternalCorrelationId(
            CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &id));
        if (id != impl_->stage_id)
            throw std::logic_error("GPU activity timing correlation stack mismatch");
        impl_->stage_open = false;
        impl_->stage_id = 0;
        if (!impl_->continuous)
        {
            check(cuptiActivityFlushAll(0));
            check(impl_->disable());
        }
    }
}

GpuActivityTimingResult GpuActivityTiming::finish()
{
    if (impl_->finished)
        throw std::logic_error("GPU activity timing already finished");
    if (impl_->stage_open)
        throw std::logic_error("cannot finish GPU activity timing with an open stage");
    check(cuptiActivityFlushAll(0));
    const auto stop_status = impl_->stop();
    impl_->finished = true;
    check(stop_status);
    if (!impl_->error.empty())
        throw std::runtime_error(impl_->error);
    if (impl_->dropped)
        throw std::runtime_error(
            "CUPTI dropped records; GPU activity timing is incomplete");

    GpuActivityTimingResult result;
    std::unordered_map<std::uint64_t, std::string> names;
    for (const auto &[name, id] : impl_->labels)
        names[id] = name;

    for (const auto &activity : impl_->activities)
    {
        if (activity.kind == 1 &&
            activity.copy_kind != CUPTI_ACTIVITY_MEMCPY_KIND_DTOD)
        {
            result.host_transfer_bytes += activity.bytes;
            ++result.host_transfers;
        }
    }
    if (result.host_transfers)
        throw std::runtime_error(
            "strict measured replay included host/device transfers: " +
            std::to_string(result.host_transfers) + " copies, " +
            std::to_string(result.host_transfer_bytes) + " bytes");

    std::vector<GpuTimedActivity> attributed;
    attributed.reserve(impl_->activities.size());
    for (const auto &activity : impl_->activities)
    {
        const auto correlation =
            impl_->correlations.find(activity.correlation);
        if (correlation == impl_->correlations.end() ||
            !names.count(correlation->second))
            throw std::runtime_error(
                "GPU activity has no measured-stage attribution");
        attributed.push_back(
            {activity.start, activity.end, correlation->second,
             activity.device, static_cast<GpuActivityType>(activity.kind)});
    }
    const auto summary = summarize_gpu_activity(attributed);
    result.gpu_total_ns = summary.total_ns;
    result.gpu_total_ms = static_cast<double>(summary.total_ns) / 1.0e6;
    result.activities = attributed.size();
    for (const auto &[stage, nanoseconds] : summary.stage_ns)
        result.stages[names.at(stage)].gpu_ms =
            static_cast<double>(nanoseconds) / 1.0e6;
    return result;
}

void GpuActivityTimingResult::print() const
{
    const auto flags = std::cout.flags();
    const auto precision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(6)
              << "GPU_ACTIVITY_TIMING gpu_total_ms=" << gpu_total_ms
              << " activities=" << activities
              << " host_transfers=" << host_transfers
              << " host_transfer_bytes=" << host_transfer_bytes << '\n';
    std::cout.flags(flags);
    std::cout.precision(precision);
}
} // namespace poseidon::benchmark::s2c_first
