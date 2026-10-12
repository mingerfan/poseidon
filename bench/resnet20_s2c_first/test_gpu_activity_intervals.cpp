#include "gpu_activity_intervals.h"

#include <iostream>
#include <limits>
#include <random>

namespace timing = poseidon::benchmark::s2c_first;
using Activity = timing::GpuTimedActivity;
using Type = timing::GpuActivityType;

void require(bool value, const char *message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <class Function>
void must_reject(Function function)
{
    bool rejected = false;
    try
    {
        function();
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    require(rejected, "invalid activity trace was accepted");
}

int main()
{
    try
    {
        require(timing::gpu_interval_union_ns({}) == 0, "empty union");
        std::vector<Activity> mixed{
            {20, 30, 2, 0, Type::device_copy},
            {40, 45, 3, 0, Type::memset},
            {10, 25, 1, 0, Type::kernel},
            {24, 35, 2, 0, Type::memset},
            {10, 25, 1, 0, Type::kernel},
            {12, 16, 1, 0, Type::device_copy},
        };
        const auto combined = timing::summarize_gpu_activity(mixed);
        require(combined.total_ns == 30, "mixed overlap/gap accounting");
        require(combined.stage_ns.at(1) == 15 &&
                    combined.stage_ns.at(2) == 15 &&
                    combined.stage_ns.at(3) == 5,
                "stage union accounting");
        require(combined.total_ns != 35,
                "global total incorrectly sums stage unions");
        require(timing::summarize_gpu_activity(
                    {{1, 6, 1, 0, Type::device_copy},
                     {6, 10, 1, 0, Type::memset}})
                    .total_ns == 9,
                "memory-only activity must be included");
        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        require(timing::gpu_interval_union_ns(
                    {{maximum - 10, maximum}, {1, 2}}) == 11,
                "large integer timestamp precision");
        require(timing::gpu_interval_union_ns({{1, maximum}}) == maximum - 1,
                "integer duration overflow");
        must_reject([] { timing::summarize_gpu_activity({}); });
        must_reject([] { timing::gpu_interval_union_ns({{0, 10}}); });
        must_reject([] { timing::gpu_interval_union_ns({{10, 10}}); });
        must_reject([] { timing::gpu_interval_union_ns({{11, 10}}); });
        must_reject([&] {
            auto trace = mixed;
            trace[0].stage = 0;
            timing::summarize_gpu_activity(trace);
        });
        must_reject([&] {
            auto trace = mixed;
            trace[0].device = 1;
            timing::summarize_gpu_activity(trace);
        });

        std::mt19937 random(20260914);
        for (int trial = 0; trial < 1000; ++trial)
        {
            std::vector<Activity> trace;
            bool occupied[128]{};
            bool stage_occupied[4][128]{};
            for (unsigned index = 0, count = 1 + random() % 100;
                 index < count; ++index)
            {
                const unsigned start = 1 + random() % 100;
                const unsigned end = start + 1 + random() % 20;
                const unsigned stage = 1 + random() % 3;
                trace.push_back(
                    {start, end, stage, 0,
                     static_cast<Type>(random() % 3)});
                for (auto nanosecond = start; nanosecond < end; ++nanosecond)
                    occupied[nanosecond] =
                        stage_occupied[stage][nanosecond] = true;
            }
            const auto summary = timing::summarize_gpu_activity(trace);
            require(summary.total_ns ==
                        std::count(std::begin(occupied), std::end(occupied), true),
                    "randomized global union mismatch");
            for (const auto &[stage, nanoseconds] : summary.stage_ns)
                require(nanoseconds ==
                            std::count(std::begin(stage_occupied[stage]),
                                       std::end(stage_occupied[stage]), true),
                        "randomized stage union mismatch");
        }
        std::cout << "GPU activity interval union tests PASS\n";
        return 0;
    }
    catch (const std::exception &exception)
    {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
