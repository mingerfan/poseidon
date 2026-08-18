#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void check(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

double percentile(const std::vector<double> &sorted, double fraction)
{
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

struct Stats
{
    double mean;
    double minimum;
    double median;
    double p95;
    double maximum;
};

Stats summarize(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return {
        std::accumulate(values.begin(), values.end(), 0.0) /
            static_cast<double>(values.size()),
        values.front(),
        percentile(values, 0.50),
        percentile(values, 0.95),
        values.back(),
    };
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const int warmup = argc > 1 ? std::atoi(argv[1]) : 20;
        const int iterations = argc > 2 ? std::atoi(argv[2]) : 100;
        if (warmup < 1 || iterations < 2)
        {
            throw std::invalid_argument("warmup must be positive and iterations >= 2");
        }

        int device_count = 0;
        check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count < 2)
        {
            throw std::runtime_error("at least two CUDA devices are required");
        }

        for (int destination = 0; destination < device_count; ++destination)
        {
            for (int source = 0; source < device_count; ++source)
            {
                if (source == destination)
                {
                    continue;
                }
                int can_access = 0;
                check(cudaDeviceCanAccessPeer(&can_access, destination, source),
                      "cudaDeviceCanAccessPeer");
                if (!can_access)
                {
                    throw std::runtime_error("a directed GPU pair lacks peer access");
                }
                check(cudaSetDevice(destination), "cudaSetDevice enable peer");
                const cudaError_t status = cudaDeviceEnablePeerAccess(source, 0);
                if (status == cudaErrorPeerAccessAlreadyEnabled)
                {
                    (void)cudaGetLastError();
                }
                else
                {
                    check(status, "cudaDeviceEnablePeerAccess");
                }
            }
        }

        std::printf("source,destination,q_count,bytes,event_mean_us,event_min_us,"
                    "event_p50_us,event_p95_us,event_max_us,wall_mean_us,wall_min_us,"
                    "wall_p50_us,wall_p95_us,wall_max_us,event_GBps,batched_event_us,"
                    "batched_GBps\n");

        constexpr std::size_t degree = 8192;
        constexpr std::size_t components = 2;
        constexpr std::size_t word_bytes = sizeof(std::uint32_t);
        constexpr int minimum_q_count = 5;
        constexpr int maximum_q_count = 17;
        const std::size_t maximum_bytes = degree * components * maximum_q_count * word_bytes;

        for (int source = 0; source < device_count; ++source)
        {
            for (int destination = 0; destination < device_count; ++destination)
            {
                if (source == destination)
                {
                    continue;
                }

                void *source_buffer = nullptr;
                void *destination_buffer = nullptr;
                cudaStream_t stream = nullptr;
                cudaEvent_t start = nullptr;
                cudaEvent_t stop = nullptr;

                check(cudaSetDevice(source), "cudaSetDevice source allocation");
                check(cudaMalloc(&source_buffer, maximum_bytes), "cudaMalloc source");
                check(cudaMemset(source_buffer, 0x5a, maximum_bytes), "cudaMemset source");
                check(cudaSetDevice(destination), "cudaSetDevice destination allocation");
                check(cudaMalloc(&destination_buffer, maximum_bytes), "cudaMalloc destination");
                check(cudaMemset(destination_buffer, 0, maximum_bytes),
                      "cudaMemset destination");
                check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                      "cudaStreamCreateWithFlags");
                check(cudaEventCreate(&start), "cudaEventCreate start");
                check(cudaEventCreate(&stop), "cudaEventCreate stop");

                for (int q_count = minimum_q_count; q_count <= maximum_q_count; ++q_count)
                {
                    const std::size_t bytes = degree * components *
                                              static_cast<std::size_t>(q_count) * word_bytes;
                    for (int i = 0; i < warmup; ++i)
                    {
                        check(cudaMemcpyPeerAsync(destination_buffer, destination,
                                                  source_buffer, source, bytes, stream),
                              "cudaMemcpyPeerAsync warmup");
                    }
                    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");

                    std::vector<double> event_us;
                    std::vector<double> wall_us;
                    event_us.reserve(iterations);
                    wall_us.reserve(iterations);
                    for (int i = 0; i < iterations; ++i)
                    {
                        const auto wall_start = std::chrono::steady_clock::now();
                        check(cudaEventRecord(start, stream), "cudaEventRecord start");
                        check(cudaMemcpyPeerAsync(destination_buffer, destination,
                                                  source_buffer, source, bytes, stream),
                              "cudaMemcpyPeerAsync measured");
                        check(cudaEventRecord(stop, stream), "cudaEventRecord stop");
                        check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
                        const auto wall_stop = std::chrono::steady_clock::now();
                        float milliseconds = 0.0F;
                        check(cudaEventElapsedTime(&milliseconds, start, stop),
                              "cudaEventElapsedTime");
                        event_us.push_back(static_cast<double>(milliseconds) * 1000.0);
                        wall_us.push_back(std::chrono::duration<double, std::micro>(
                                              wall_stop - wall_start)
                                              .count());
                    }

                    check(cudaEventRecord(start, stream), "cudaEventRecord batch start");
                    for (int i = 0; i < iterations; ++i)
                    {
                        check(cudaMemcpyPeerAsync(destination_buffer, destination,
                                                  source_buffer, source, bytes, stream),
                              "cudaMemcpyPeerAsync batch");
                    }
                    check(cudaEventRecord(stop, stream), "cudaEventRecord batch stop");
                    check(cudaEventSynchronize(stop), "cudaEventSynchronize batch stop");
                    float batch_milliseconds = 0.0F;
                    check(cudaEventElapsedTime(&batch_milliseconds, start, stop),
                          "cudaEventElapsedTime batch");

                    const Stats event = summarize(event_us);
                    const Stats wall = summarize(wall_us);
                    const double batch_us = static_cast<double>(batch_milliseconds) * 1000.0 /
                                            static_cast<double>(iterations);
                    const double event_gbps = static_cast<double>(bytes) /
                                              event.mean / 1000.0;
                    const double batch_gbps = static_cast<double>(bytes) /
                                              batch_us / 1000.0;
                    std::printf("%d,%d,%d,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,"
                                "%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.3f,%.4f\n",
                                source, destination, q_count, bytes,
                                event.mean, event.minimum, event.median, event.p95,
                                event.maximum, wall.mean, wall.minimum, wall.median,
                                wall.p95, wall.maximum, event_gbps, batch_us, batch_gbps);
                    std::fflush(stdout);
                }

                check(cudaEventDestroy(stop), "cudaEventDestroy stop");
                check(cudaEventDestroy(start), "cudaEventDestroy start");
                check(cudaStreamDestroy(stream), "cudaStreamDestroy");
                check(cudaFree(destination_buffer), "cudaFree destination");
                check(cudaSetDevice(source), "cudaSetDevice source free");
                check(cudaFree(source_buffer), "cudaFree source");
            }
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
