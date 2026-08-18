#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_stream_wait_trace.h"
#include "runtime/thread_trace.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

void require(bool condition, const char *message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    try
    {
        const auto nonce = std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count();
        const auto output_path = std::filesystem::temp_directory_path() /
                                 ("poseidon-gpu-stream-wait-trace-" +
                                  std::to_string(nonce) + ".json");
        if (::setenv(
                "POSEIDON_THREAD_TRACE", output_path.string().c_str(), 1) != 0)
        {
            throw std::runtime_error("setenv failed");
        }

        int device_count = 0;
        poseidon::gpu::gpu_check_cuda(
            cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        require(device_count > 0, "no CUDA device is available");
        poseidon::gpu::gpu_check_cuda(cudaSetDevice(0), "cudaSetDevice");

        cudaStream_t producer_stream = nullptr;
        cudaStream_t consumer_stream = nullptr;
        cudaEvent_t dependency_event = nullptr;
        poseidon::gpu::gpu_check_cuda(
            cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking),
            "cudaStreamCreate producer");
        poseidon::gpu::gpu_check_cuda(
            cudaStreamCreateWithFlags(&consumer_stream, cudaStreamNonBlocking),
            "cudaStreamCreate consumer");
        poseidon::gpu::gpu_check_cuda(
            cudaEventCreateWithFlags(
                &dependency_event, cudaEventDisableTiming),
            "cudaEventCreate dependency");
        poseidon::gpu::gpu_check_cuda(
            cudaEventRecord(dependency_event, producer_stream),
            "cudaEventRecord dependency");

        fhegpu::ThreadTrace::set_thread_name("gpu-stream-wait-trace-test");
        poseidon::gpu::gpu_stream_wait_event(
            consumer_stream, dependency_event, 0,
            "gpu.stream_wait.test_dependency",
            "cudaStreamWaitEvent test dependency");
        poseidon::gpu::gpu_check_cuda(
            cudaStreamSynchronize(consumer_stream),
            "cudaStreamSynchronize consumer");
        fhegpu::ThreadTrace::write_json(0);

        poseidon::gpu::gpu_check_cuda(
            cudaEventDestroy(dependency_event), "cudaEventDestroy dependency");
        poseidon::gpu::gpu_check_cuda(
            cudaStreamDestroy(consumer_stream), "cudaStreamDestroy consumer");
        poseidon::gpu::gpu_check_cuda(
            cudaStreamDestroy(producer_stream), "cudaStreamDestroy producer");

        std::ifstream input(output_path);
        require(static_cast<bool>(input), "stream wait trace JSON was not written");
        const nlohmann::json trace = nlohmann::json::parse(input);
        bool found_wait = false;
        for (const auto &thread : trace.at("threads"))
        {
            if (thread.at("name") != "gpu-stream-wait-trace-test")
            {
                continue;
            }
            require(
                thread.at("metrics")
                        .at("gpu.stream_wait.test_dependency")
                        .at("count") == 1,
                "stream wait metric is missing");
            const auto &events = thread.at("deferred_duration_events");
            require(events.size() == 1, "stream wait event is missing");
            require(events.at(0).at("device") == 0, "wrong stream wait device");
            found_wait = true;
        }
        require(found_wait, "stream wait thread trace is missing");
        std::filesystem::remove(output_path);
        std::cout << "GPU stream wait trace test passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "GPU stream wait trace test failed: " << error.what() << '\n';
        return 1;
    }
}
