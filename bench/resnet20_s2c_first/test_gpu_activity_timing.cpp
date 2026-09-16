#include "gpu_activity_timing.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace timing = poseidon::benchmark::s2c_first;

void cuda_check(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
}

int main()
{
    void *source = nullptr, *destination = nullptr;
    try
    {
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        cuda_check(cudaFree(nullptr), "initialize CUDA context");
        cuda_check(cudaMalloc(&source, 8 * 1024 * 1024), "cudaMalloc source");
        cuda_check(cudaMalloc(&destination, 8 * 1024 * 1024),
                   "cudaMalloc destination");
        cuda_check(cudaDeviceSynchronize(), "synchronize setup");

        timing::GpuActivityTiming collector;
        collector.stage("memory_contract", true);
        cuda_check(cudaMemset(source, 0x5a, 8 * 1024 * 1024), "cudaMemset");
        cuda_check(cudaMemcpy(destination, source, 8 * 1024 * 1024,
                              cudaMemcpyDeviceToDevice),
                   "cudaMemcpy D2D");
        cuda_check(cudaDeviceSynchronize(), "synchronize measured work");
        collector.stage("memory_contract", false);

        // A transfer outside a measured stage is offline preparation and must
        // not leak into the strict online result.
        std::vector<unsigned char> host(8 * 1024 * 1024, 0x3c);
        cuda_check(cudaMemcpy(source, host.data(), host.size(),
                              cudaMemcpyHostToDevice),
                   "offline cudaMemcpy H2D");
        collector.stage("second_stage", true);
        cuda_check(cudaMemset(destination, 0xa5, 8 * 1024 * 1024),
                   "second cudaMemset");
        cuda_check(cudaDeviceSynchronize(), "synchronize second measured work");
        collector.stage("second_stage", false);
        const auto result = collector.finish();
        if (!(result.gpu_total_ns > 0 && result.activities >= 3 &&
              result.host_transfers == 0 &&
              result.stages.count("memory_contract") == 1 &&
              result.stages.count("second_stage") == 1))
            throw std::runtime_error("invalid strict CUPTI timing result");
        result.print();

        bool rejected_host_transfer = false;
        try
        {
            timing::GpuActivityTiming strict;
            strict.stage("invalid_online_transfer", true);
            cuda_check(cudaMemcpy(source, host.data(), host.size(),
                                  cudaMemcpyHostToDevice),
                       "measured cudaMemcpy H2D");
            cuda_check(cudaDeviceSynchronize(), "synchronize invalid transfer");
            strict.stage("invalid_online_transfer", false);
            (void)strict.finish();
        }
        catch (const std::runtime_error &error)
        {
            rejected_host_transfer =
                std::string(error.what()).find("included host/device transfers") !=
                std::string::npos;
        }
        if (!rejected_host_transfer)
            throw std::runtime_error("measured H2D transfer was not rejected");

        // Continuous mode changes correlation labels without synchronizing or
        // enabling/disabling CUPTI at each boundary.
        timing::GpuActivityTiming continuous;
        continuous.begin_continuous();
        continuous.stage("first_async_category", true);
        cuda_check(cudaMemsetAsync(source, 0x11, 8 * 1024 * 1024),
                   "first asynchronous cudaMemset");
        continuous.stage("first_async_category", false);
        continuous.stage("second_async_category", true);
        cuda_check(cudaMemsetAsync(destination, 0x22, 8 * 1024 * 1024),
                   "second asynchronous cudaMemset");
        continuous.stage("second_async_category", false);
        cuda_check(cudaDeviceSynchronize(), "synchronize continuous work once");
        const auto continuous_result = continuous.finish();
        if (!(continuous_result.activities >= 2 &&
              continuous_result.stages.count("first_async_category") == 1 &&
              continuous_result.stages.count("second_async_category") == 1))
            throw std::runtime_error("continuous CUPTI attribution failed");

        cuda_check(cudaFree(destination), "cudaFree destination");
        destination = nullptr;
        cuda_check(cudaFree(source), "cudaFree source");
        source = nullptr;
        std::cout << "GPU activity timing runtime contract PASS\n";
        return 0;
    }
    catch (const std::exception &exception)
    {
        if (destination)
            cudaFree(destination);
        if (source)
            cudaFree(source);
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
