// Small allocator-mechanism tests; these do not replay the full-model OOM.
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr std::size_t MiB = 1ULL << 20;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Attempt {
    std::size_t bytes;
    bool succeeded;
};

class RecordingUpstream final : public rmm::mr::device_memory_resource {
public:
    explicit RecordingUpstream(rmm::mr::device_memory_resource* upstream) : upstream_(upstream) {}
    std::vector<Attempt> attempts;

private:
    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
    {
        try {
            void* pointer = upstream_->allocate(bytes, stream);
            attempts.push_back({bytes, true});
            return pointer;
        } catch (...) {
            attempts.push_back({bytes, false});
            throw;
        }
    }
    void do_deallocate(void* pointer, std::size_t bytes, rmm::cuda_stream_view stream) override
    {
        upstream_->deallocate(pointer, bytes, stream);
    }
    bool do_is_equal(const rmm::mr::device_memory_resource& other) const noexcept override
    {
        return this == &other;
    }
    rmm::mr::device_memory_resource* upstream_;
};

using Pool = rmm::mr::pool_memory_resource<RecordingUpstream>;

void fragmented_pool_negative()
{
    rmm::mr::cuda_memory_resource cuda;
    RecordingUpstream upstream(&cuda);
    Pool pool(&upstream, 4 * MiB, 16 * MiB);
    rmm::mr::statistics_resource_adaptor<Pool> measured(&pool);
    void* a = measured.allocate(4 * MiB);
    void* b = measured.allocate(6 * MiB);
    void* c = measured.allocate(4 * MiB);
    require(pool.pool_size() == 14 * MiB, "unexpected geometric growth sizes");
    measured.deallocate(a, 4 * MiB);
    measured.deallocate(b, 6 * MiB);
    require(measured.get_bytes_counter().value == 4 * MiB, "unexpected live bytes before negative control");
    require(pool.pool_size() - measured.get_bytes_counter().value == 10 * MiB,
            "negative control must have ten MiB total free");
    const auto attempts_before = upstream.attempts.size();
    bool rejected = false;
    try {
        void* unexpected = measured.allocate(8 * MiB);
        measured.deallocate(unexpected, 8 * MiB);
    } catch (const rmm::out_of_memory& error) {
        require(std::string(error.what()).find("Maximum pool size exceeded") != std::string::npos,
                "negative control changed the expected RMM error");
        rejected = true;
        std::cout << "EXPECTED_ERROR fragmented_pool " << error.what() << '\n';
    }
    require(rejected, "fragmentation negative control unexpectedly succeeded");
    require(upstream.attempts.size() == attempts_before,
            "cap remainder smaller than request must reject without an upstream allocation attempt");
    measured.deallocate(c, 4 * MiB);
    require(measured.get_bytes_counter().value == 0, "negative control leaked live bytes");
    std::cout << "PASS fragmented_pool_negative initial_bytes=" << 4 * MiB
              << " cap_bytes=" << 16 * MiB << " reserved_bytes=" << pool.pool_size()
              << " live_before_request_bytes=" << 4 * MiB << " total_free_bytes=" << 10 * MiB
              << " request_bytes=" << 8 * MiB << " upstream_attempts=" << upstream.attempts.size() << '\n';
}

void preallocated_pool_positive()
{
    rmm::mr::cuda_memory_resource cuda;
    RecordingUpstream upstream(&cuda);
    Pool pool(&upstream, 16 * MiB, 16 * MiB);
    rmm::mr::statistics_resource_adaptor<Pool> measured(&pool);
    void* a = measured.allocate(4 * MiB);
    void* b = measured.allocate(6 * MiB);
    void* c = measured.allocate(4 * MiB);
    measured.deallocate(a, 4 * MiB);
    measured.deallocate(b, 6 * MiB);
    void* d = measured.allocate(8 * MiB);
    require(upstream.attempts.size() == 1 && upstream.attempts[0].bytes == 16 * MiB &&
            upstream.attempts[0].succeeded, "preallocated pool unexpectedly grew upstream");
    require(measured.get_bytes_counter().value == 12 * MiB, "preallocated pool live count mismatch");
    measured.deallocate(d, 8 * MiB);
    measured.deallocate(c, 4 * MiB);
    require(measured.get_bytes_counter().value == 0, "positive control leaked live bytes");
    std::cout << "PASS preallocated_pool_positive initial_bytes=" << 16 * MiB
              << " cap_bytes=" << 16 * MiB << " reserved_bytes=" << pool.pool_size()
              << " request_bytes=" << 8 * MiB << " upstream_attempts=" << upstream.attempts.size() << '\n';
}

void geometric_backoff_positive()
{
    rmm::mr::cuda_memory_resource cuda;
    rmm::mr::limiting_resource_adaptor<rmm::mr::cuda_memory_resource> limiter(&cuda, 8 * MiB);
    RecordingUpstream upstream(&limiter);
    {
        Pool pool(&upstream, 4 * MiB, 16 * MiB);
        void* a = pool.allocate(4 * MiB);
        void* b = pool.allocate(4 * MiB);
        require(upstream.attempts.size() == 3 &&
                upstream.attempts[0].bytes == 4 * MiB && upstream.attempts[0].succeeded &&
                upstream.attempts[1].bytes == 6 * MiB && !upstream.attempts[1].succeeded &&
                upstream.attempts[2].bytes == 4 * MiB && upstream.attempts[2].succeeded,
                "expected upstream attempts: initial four, failed six, successful four MiB");
        require(pool.pool_size() == 8 * MiB && limiter.get_allocated_bytes() == 8 * MiB,
                "backoff reserved bytes mismatch");
        pool.deallocate(b, 4 * MiB);
        pool.deallocate(a, 4 * MiB);
        std::cout << "PASS geometric_backoff_positive initial_bytes=" << 4 * MiB
                  << " cap_bytes=" << 16 * MiB << " upstream_limit_bytes=" << 8 * MiB
                  << " failed_growth_bytes=" << 6 * MiB << " successful_retry_bytes=" << 4 * MiB
                  << " reserved_bytes=" << pool.pool_size() << '\n';
    }
    require(limiter.get_allocated_bytes() == 0, "pool destruction did not return upstream allocations");
}

int main()
{
    try {
        require(cudaSetDevice(0) == cudaSuccess && cudaFree(nullptr) == cudaSuccess,
                "cannot initialize visible CUDA device zero");
        fragmented_pool_negative();
        preallocated_pool_positive();
        geometric_backoff_positive();
        require(cudaDeviceSynchronize() == cudaSuccess, "final CUDA synchronization failed");
        std::cout << "PASS ALL_RMM_MECHANISM_TESTS count=3 full_model_oom_replayed=false\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
