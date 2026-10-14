// Multi-thread single-GPU LAUNCH-RATE benchmark (self-timed, no nsys needed).
//
// Measures the pure CPU-side kernel-launch path under multi-thread driver
// contention. Each thread launches `rounds` copies of a REAL poseidon kernel
// (apply_galois_ntt_poly_shard_kernel, verbatim, N=8192 limb=17 grid=544
// block=256) back-to-back with NO sync between launches. The per-stream queue
// depth stays well below the driver limit (~1024), so cudaLaunchKernel never
// blocks on the GPU: elapsed/rounds is the driver launch cost per kernel.
//
// Contention: N threads on ONE device -> N threads inside libcuda
// simultaneously (same driver locks the real 4-GPU runtime contends on).
//
// Usage: keyswitch_launchrate [rounds] [threads] [device]
//   rounds  launches per thread (default 500, queue stays < 1024)
//   threads number of launch threads (default 1)
//   device  GPU index (default 0)

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>

using GpuWord = std::uint32_t;

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// REAL poseidon kernel (verbatim from gpu_keyswitch_kernels.cu)
// ---------------------------------------------------------------------------

__device__ __forceinline__ std::uint32_t reverse_bits_limited(
    std::uint32_t value, unsigned int bit_count)
{
    return __brev(value) >> (32U - bit_count);
}

__global__ void apply_galois_ntt_poly_shard_kernel(
    GpuWord *destination, const GpuWord *source,
    std::uint32_t galois_elt, std::size_t limb_count,
    std::size_t degree, unsigned int degree_power)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * degree;
    if (tid >= total) return;
    const std::uint32_t degree_u32 = static_cast<std::uint32_t>(degree);
    const std::uint32_t degree_minus_one = degree_u32 - 1;
    const std::size_t limb = tid >> degree_power;
    const std::uint32_t coeff = static_cast<std::uint32_t>(tid & degree_minus_one);
    const std::uint32_t reversed =
        reverse_bits_limited(degree_u32 + coeff, degree_power + 1);
    const std::uint64_t index_raw =
        (static_cast<std::uint64_t>(galois_elt) *
         static_cast<std::uint64_t>(reversed)) >> 1;
    const std::uint32_t source_coeff =
        reverse_bits_limited(
            static_cast<std::uint32_t>(index_raw & degree_minus_one),
            degree_power);
    destination[limb * degree + coeff] = source[limb * degree + source_coeff];
}

// ---------------------------------------------------------------------------

struct BenchState {
    std::atomic<bool> stop{false};
    std::atomic<long> done{0};
    int gpu;
    long rounds;
    std::size_t degree, limb_count;
    unsigned int degree_power;
    int grid, block;
    GpuWord *src, *dst;
    cudaStream_t stream;
};

std::atomic<int> go{0};

void barrier() {
    go.fetch_sub(1);
    while (go.load() != 0) {}
}

void warmup_thread_fn(BenchState &st, int warmup_rounds) {
    cudaSetDevice(st.gpu);
    barrier();
    for (int i = 0; i < warmup_rounds; ++i) {
        apply_galois_ntt_poly_shard_kernel<<<st.grid, st.block, 0, st.stream>>>(
            st.dst, st.src, 3, st.limb_count, st.degree, st.degree_power);
    }
}

void measure_thread_fn(BenchState &st, long &total_us, long &n) {
    cudaSetDevice(st.gpu);
    barrier();
    const auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < st.rounds && !st.stop.load(); ++i) {
        apply_galois_ntt_poly_shard_kernel<<<st.grid, st.block, 0, st.stream>>>(
            st.dst, st.src, 3, st.limb_count, st.degree, st.degree_power);
    }
    const auto t1 = std::chrono::steady_clock::now();
    CHECK(cudaGetLastError());
    total_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    n = st.rounds;
}

int main(int argc, char **argv) {
    const std::size_t degree = 8192;
    const std::size_t limb_count = 17;
    const unsigned int degree_power = 13;
    constexpr int block = 256;
    const int grid = (int)((limb_count * degree + block - 1) / block);
    long rounds = 500;
    int threads = 1;
    int gpu = 0;
    if (argc > 1) rounds = atol(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);
    if (argc > 3) gpu = atoi(argv[3]);

    int dev_count = 0;
    CHECK(cudaGetDeviceCount(&dev_count));
    if (gpu >= dev_count) { fprintf(stderr, "need GPU %d (have %d)\n", gpu, dev_count); return 1; }
    cudaSetDevice(gpu);

    const std::size_t poly_bytes = degree * limb_count * sizeof(GpuWord);
    std::vector<GpuWord> hsrc(poly_bytes / sizeof(GpuWord), 0x12345678u);

    std::vector<BenchState> st(threads);
    for (int t = 0; t < threads; ++t) {
        st[t].gpu = gpu;
        st[t].rounds = rounds;
        st[t].degree = degree;
        st[t].limb_count = limb_count;
        st[t].degree_power = degree_power;
        st[t].grid = grid;
        st[t].block = block;
        CHECK(cudaMalloc(&st[t].src, poly_bytes));
        CHECK(cudaMalloc(&st[t].dst, poly_bytes));
        CHECK(cudaMemcpy(st[t].src, hsrc.data(), poly_bytes, cudaMemcpyHostToDevice));
        CHECK(cudaStreamCreate(&st[t].stream));
    }

    printf("degree=%zu limb=%zu grid=%d block=%d rounds=%ld threads=%d gpu=%d launchrate\n",
           degree, limb_count, grid, block, rounds, threads, gpu);
    fflush(stdout);

    // warmup: 3 rounds per thread (JIT), then measure
    {
        go.store(threads);
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back(warmup_thread_fn, std::ref(st[t]), 3);
        for (auto &t : ts) t.join();
        cudaDeviceSynchronize();
    }

    long long tot_us = 0;
    long long tot_n = 0;
    {
        for (int t = 0; t < threads; ++t) st[t].done.store(0);
        go.store(threads);
        std::vector<long> us(threads), n(threads);
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back(measure_thread_fn, std::ref(st[t]),
                            std::ref(us[t]), std::ref(n[t]));
        for (auto &t : ts) t.join();
        printf("%-8s %10s %12s\n", "thread", "total_us", "per_launch_us");
        for (int t = 0; t < threads; ++t) {
            printf("%-8d %10ld %12.4f\n", t, us[t],
                   (double)us[t] / n[t]);
            tot_us += us[t];
            tot_n += n[t];
        }
        printf("ALL-THREADS total=%lld us launches=%lld avg_per_launch=%.4f us\n",
               tot_us, tot_n, (double)tot_us / tot_n);
        fflush(stdout);
    }
    cudaDeviceSynchronize();

    for (int t = 0; t < threads; ++t) {
        CHECK(cudaFree(st[t].src));
        CHECK(cudaFree(st[t].dst));
        CHECK(cudaStreamDestroy(st[t].stream));
    }
    printf("done\n");
    return 0;
}
