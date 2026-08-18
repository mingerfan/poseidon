// Dual-thread microbenchmark: per-GPU launch thread + comm thread.
//
// Mirrors the REAL 4-GPU runtime model (direct-4gpu.nsys-rep):
//   - 4 LAUNCH threads (one per GPU): only cudaLaunchKernel
//   - 4 COMM threads (one per GPU):   only cudaMemcpyPeerAsync / event sync
//
// Here with 2 GPUs: GPU0 gets launch0 + comm0, GPU1 gets launch1 + comm1.
//
// Modes (both in one process, timeline-ordered):
//   no-comm: launch threads run alone, each doing exactly `rounds` launches
//   comm:    launch threads AND comm threads run concurrently
//            -> measures whether concurrent P2P copy affects the CPU-side
//               cudaLaunchKernel duration of the launch threads.
//
// Real kernel: apply_galois_ntt_poly_shard_kernel (verbatim from poseidon),
// N=8192, limb=17, grid=544, block=256 (identical to MLP).
//
// Launch threads use one stream per GPU; comm threads use one copy stream
// per GPU, on an unrelated 64MB buffer (2MB x 16 per comm round).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>

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
// REAL kernel (verbatim from poseidon gpu_keyswitch_kernels.cu)
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
// Per-GPU state
// ---------------------------------------------------------------------------

struct BenchState {
    std::atomic<int> go{0};      // barrier: wait until all participants set
    std::atomic<bool> stop{false};
    std::atomic<long> done{0};   // launch count done by the launch thread

    int gpu;
    std::size_t degree;
    std::size_t limb_count;
    unsigned int degree_power;
    int grid, block;
    long rounds;                 // exact number of launches this mode

    GpuWord *src, *dst;          // galois buffers (this GPU)
    cudaStream_t stream;

    // comm thread state
    GpuWord *p2p_src, *p2p_dst;  // 64MB buffers, unrelated to galois
    cudaStream_t copy_stream;
    std::size_t p2p_chunk;
    int p2p_copies;
};

// Launch thread: launch `rounds` kernels on the GPU's compute stream.
void launch_thread_fn(BenchState &st) {
    cudaSetDevice(st.gpu);
    st.go.fetch_sub(1);
    while (st.go.load() != 0) {}   // barrier: wait for all participants
    for (long i = 0; i < st.rounds && !st.stop.load(); ++i) {
        apply_galois_ntt_poly_shard_kernel<<<st.grid, st.block, 0, st.stream>>>(
            st.dst, st.src, 3, st.limb_count, st.degree, st.degree_power);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr, "launch err: %s\n", cudaGetErrorString(e)); exit(1); }
        st.done.fetch_add(1);
    }
}

// Comm thread: issue P2P copies continuously until stop.
void comm_thread_fn(BenchState &st) {
    cudaSetDevice(st.gpu);
    st.go.fetch_sub(1);
    while (st.go.load() != 0) {}   // barrier
    const int other = 1 - st.gpu;
    while (!st.stop.load()) {
        for (int c = 0; c < st.p2p_copies; ++c) {
            cudaError_t e = cudaMemcpyPeerAsync(
                (char*)st.p2p_dst + (std::size_t)c * st.p2p_chunk, other,
                (char*)st.p2p_src + (std::size_t)c * st.p2p_chunk, st.gpu,
                st.p2p_chunk, st.copy_stream);
            if (e != cudaSuccess) { fprintf(stderr, "p2p err: %s\n", cudaGetErrorString(e)); exit(1); }
        }
    }
}

int main(int argc, char **argv) {
    const std::size_t degree = 8192;
    const std::size_t limb_count = 17;
    const unsigned int degree_power = 13;
    constexpr int block = 256;
    const int grid = (int)((limb_count * degree + block - 1) / block);
    const std::size_t p2p_chunk = 2u << 20;  // 2MB per copy
    const int p2p_copies = 16;
    long rounds = 500;  // launches per launch thread per mode
    if (argc > 1) rounds = atol(argv[1]);

    int dev_count = 0;
    CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count < 2) { fprintf(stderr, "need 2 GPUs\n"); return 1; }
    cudaSetDevice(0);
    int can = 0; CHECK(cudaDeviceCanAccessPeer(&can, 0, 1));
    if (!can) { fprintf(stderr, "P2P 0<->1 not supported\n"); return 1; }
    cudaSetDevice(0); CHECK(cudaDeviceEnablePeerAccess(1, 0));
    cudaSetDevice(1); CHECK(cudaDeviceEnablePeerAccess(0, 0));

    const std::size_t poly_bytes = degree * limb_count * sizeof(GpuWord);
    const std::size_t p2p_bytes = 64u << 20;

    std::vector<GpuWord> hsrc(poly_bytes / sizeof(GpuWord), 0x12345678u);
    std::vector<GpuWord> hp2p(p2p_bytes / sizeof(GpuWord), 0xabcdef01u);

    BenchState st[2];
    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        st[g].gpu = g;
        st[g].degree = degree;
        st[g].limb_count = limb_count;
        st[g].degree_power = degree_power;
        st[g].grid = grid;
        st[g].block = block;
        st[g].rounds = rounds;
        st[g].p2p_chunk = p2p_chunk;
        st[g].p2p_copies = p2p_copies;
        CHECK(cudaMalloc(&st[g].src, poly_bytes));
        CHECK(cudaMalloc(&st[g].dst, poly_bytes));
        CHECK(cudaMalloc(&st[g].p2p_src, p2p_bytes));
        CHECK(cudaMalloc(&st[g].p2p_dst, p2p_bytes));
        CHECK(cudaMemcpy(st[g].src, hsrc.data(), poly_bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(st[g].p2p_src, hp2p.data(), p2p_bytes, cudaMemcpyHostToDevice));
        CHECK(cudaStreamCreate(&st[g].stream));
        CHECK(cudaStreamCreate(&st[g].copy_stream));
    }

    printf("degree=%zu limb=%zu grid=%d block=%d P2P=2MBx%d rounds=%ld\n",
           degree, limb_count, grid, block, p2p_copies, rounds);
    fflush(stdout);

    // ---- MODE 0: no-comm (launch threads only, comm threads idle) ----
    {
        printf("[mode=no-comm] start\n"); fflush(stdout);
        // barrier participants: 1 per GPU (launch thread only)
        for (int g = 0; g < 2; ++g) { st[g].go.store(1); st[g].stop.store(false); st[g].done.store(0); }
        std::thread l0(launch_thread_fn, std::ref(st[0]));
        std::thread l1(launch_thread_fn, std::ref(st[1]));
        while (st[0].done.load() < rounds || st[1].done.load() < rounds) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        st[0].stop.store(true); st[1].stop.store(true);
        l0.join(); l1.join();
        printf("[mode=no-comm] done launches=%ld/%ld\n", st[0].done.load(), st[1].done.load());
        fflush(stdout);
        // sync GPU so next mode starts clean
        for (int g = 0; g < 2; ++g) { cudaSetDevice(g); CHECK(cudaDeviceSynchronize()); }
    }

    // ---- MODE 1: comm (launch threads + comm threads concurrently) ----
    {
        printf("[mode=comm] start\n"); fflush(stdout);
        // barrier participants: 2 per GPU (launch + comm)
        for (int g = 0; g < 2; ++g) { st[g].go.store(2); st[g].stop.store(false); st[g].done.store(0); }
        std::thread l0(launch_thread_fn, std::ref(st[0]));
        std::thread l1(launch_thread_fn, std::ref(st[1]));
        std::thread c0(comm_thread_fn, std::ref(st[0]));
        std::thread c1(comm_thread_fn, std::ref(st[1]));
        while (st[0].done.load() < rounds || st[1].done.load() < rounds) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        st[0].stop.store(true); st[1].stop.store(true);
        l0.join(); l1.join(); c0.join(); c1.join();
        printf("[mode=comm] done launches=%ld/%ld\n", st[0].done.load(), st[1].done.load());
        fflush(stdout);
        for (int g = 0; g < 2; ++g) { cudaSetDevice(g); CHECK(cudaDeviceSynchronize()); }
    }

    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        CHECK(cudaFree(st[g].src)); CHECK(cudaFree(st[g].dst));
        CHECK(cudaFree(st[g].p2p_src)); CHECK(cudaFree(st[g].p2p_dst));
        CHECK(cudaStreamDestroy(st[g].stream));
        CHECK(cudaStreamDestroy(st[g].copy_stream));
    }
    printf("done\n");
    return 0;
}
