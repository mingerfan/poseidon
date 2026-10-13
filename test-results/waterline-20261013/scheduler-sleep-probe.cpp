// Local host timer measurement only: no executor, GPU, or model execution.
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sys/resource.h>
#include <thread>
#include <vector>

double cpu_us(const rusage &usage) {
    return (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec)*1e6 +
           usage.ru_utime.tv_usec + usage.ru_stime.tv_usec;
}
int main() {
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t samples = 1000;
    std::cout << "scope=local-host-timer-only no-executor no-GPU no-model\n";
    for (const int requested_us : {1000, 100}) {
        std::vector<double> durations;
        durations.reserve(samples);
        rusage before{}, after{};
        getrusage(RUSAGE_SELF, &before);
        const auto wall_begin = Clock::now();
        for (std::size_t index=0; index<samples; ++index) {
            const auto begin = Clock::now();
            std::this_thread::sleep_for(std::chrono::microseconds(requested_us));
            durations.push_back(std::chrono::duration<double,std::micro>(Clock::now()-begin).count());
        }
        const double wall_ms = std::chrono::duration<double,std::milli>(Clock::now()-wall_begin).count();
        getrusage(RUSAGE_SELF, &after);
        std::sort(durations.begin(), durations.end());
        std::cout << "requested_us=" << requested_us << " samples=" << samples
                  << " mean_us=" << std::accumulate(durations.begin(),durations.end(),0.)/samples
                  << " median_us=" << durations[samples/2] << " p95_us=" << durations[samples*95/100]
                  << " wall_ms=" << wall_ms << " process_cpu_ms=" << (cpu_us(after)-cpu_us(before))/1000
                  << " voluntary_context_switches=" << after.ru_nvcsw-before.ru_nvcsw << '\n';
    }
}
