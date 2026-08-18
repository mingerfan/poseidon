// Two-GPU P2P overlap microbenchmark.
//
// Purpose: measure whether concurrent P2P (Peer-to-Peer) copies inflate the
// wall duration of memory-bound / compute-bound kernels on the same GPU.
//
// Design:
//   * Two GPUs (0 and 1). Each iteration launches a set of kernels on both
//     GPUs. In "comm" mode we also issue cudaMemcpyPeerAsync 0->1 on a
//     dedicated copy stream so it overlaps with the kernels.
//   * Every (mode, kernel) combination is wrapped in an NVTX range so the
//     nsys timeline / SQLite can classify each kernel by its NVTX parent.
//   * The kernel set includes:
//       - dyadic_product (memory bound, few elements per thread)
//       - galois_apply   (memory bound, small reads)
//       - nt4_fused      (a fused forward-NTT-ish stage, compute+memory)
//       - long_kernel    (compute-heavy dummy loop) -> should be insensitive
//   * We run `iterations` repetitions per mode.
//
// Compile:
//   nvcc -arch=sm_70 -o p2p_kernel_bench p2p_kernel_bench.cu
//
// Run:
//   nsys profile -t cuda,nvtx -o bench ./p2p_kernel_bench

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>

#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Memory-bound dyadic product: one uint32 per thread.
__global__ void dyadic_product_kernel(uint32_t *out, const uint32_t *a,
                                      const uint32_t *b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (a[i] * b[i]) & 0xffffffffu;
}

// Galois-style apply: read a permuted index.
__global__ void galois_apply_kernel(uint32_t *out, const uint32_t *in,
                                    uint32_t n, uint32_t step) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[(i * step) & (n - 1)];
}

// Fused forward-NTT-ish: several dependent memory ops + multiply.
__global__ void fused_ntt_kernel(uint32_t *out, const uint32_t *in,
                                 uint32_t n, uint32_t w) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t t = in[i];
        t = (t + in[(i + n/2) & (n - 1)]) * w;
        t = (t * w + in[(i * 3) & (n - 1)]) ^ 0x5f3759dfu;
        out[i] = t;
    }
}

// Compute-heavy dummy: insensitive to memory bandwidth.
__global__ void long_compute_kernel(uint32_t *out, uint32_t n,
                                    uint32_t iters) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = i;
    for (uint32_t k = 0; k < iters; ++k) {
        acc = acc * 1664525u + 1013904223u;
        acc ^= acc >> 13;
    }
    if (i < n) out[i] = acc;
}

// ---------------------------------------------------------------------------
// Hybrid keyswitch-style kernels (from poseidon::gpu::kernel)
// ---------------------------------------------------------------------------

// hybrid_forward_ntt_modup_qp_fused_stage_kernel:
// ModUp + forward NTT fused stage. Reads q-poly, writes qp-poly, multiple
// buffers, medium memory footprint, grid 20-40 blocks x 256 threads.
__global__ void hybrid_ntt_modup_fused_stage_kernel(
    uint32_t *qp_out, const uint32_t *q_in, const uint32_t *root,
    uint32_t n, uint32_t w) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t t = q_in[i];
        t = (t + q_in[i ^ (n >> 1)]) * w;
        t = (t * root[i & 255u]) + q_in[(i * 5) & (n - 1)];
        qp_out[i] = t;
    }
}

// hybrid_forward_ntt_q_two_components_fused_stage_kernel:
// Two-component fused stage, grid 40-72, moderate.
__global__ void hybrid_ntt_q_two_comp_fused_kernel(
    uint32_t *out0, uint32_t *out1, const uint32_t *in0,
    const uint32_t *in1, uint32_t n, uint32_t w) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t a = in0[i];
        uint32_t b = in1[i];
        a = (a + in0[(i * 7) & (n - 1)]) * w;
        b = (b ^ (a >> 16)) * w;
        out0[i] = a;
        out1[i] = b;
    }
}

// hybrid_apply_moddown_ntt_add_back_two_components_kernel:
// ModDown + NTT + add-back, two components, grid 320-576 (large).
__global__ void hybrid_moddown_ntt_add_back_kernel(
    uint32_t *dst, const uint32_t *src0, const uint32_t *src1,
    const uint32_t *mod0, uint32_t n, uint32_t w) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t t = src0[i];
        t = (t + src1[i]) * w;
        t = (t - (mod0[i & 255u] * t >> 24)) ^ 0xa5a5a5a5u;
        dst[i] = t;
    }
}

// ---------------------------------------------------------------------------
// NVTX helper
// ---------------------------------------------------------------------------

static void nvtx_push(const char *name) {
#ifdef USE_NVTX
    nvtxRangePushA(name);
#endif
}
static void nvtx_pop() {
#ifdef USE_NVTX
    nvtxRangePop();
#endif
}

// ---------------------------------------------------------------------------

static void launch_set(cudaStream_t stream, uint32_t *bufs[8],
                       uint32_t n, uint32_t iters) {
    // ONE compute stream per GPU (mirrors the real runtime: kernels serialize
    // on a single stream; only the P2P copy stream runs in parallel).
    // slots: 0/1/2 = dyadic out/a/b, 3 = fused misc, 4 = hntt qp_out,
    //        5 = hntt q_in, 6 = q_two out, 7 = moddown misc
    const uint32_t n_threads = 256;
    const uint32_t grid = (n + n_threads - 1) / n_threads;

    dyadic_product_kernel<<<grid, n_threads, 0, stream>>>(
        bufs[0], bufs[1], bufs[2], n);
    galois_apply_kernel<<<grid, n_threads, 0, stream>>>(
        bufs[0], bufs[1], n, 3);
    fused_ntt_kernel<<<grid, n_threads, 0, stream>>>(
        bufs[3], bufs[1], n, 0x1234u);

    // keyswitch-style kernels (mirror poseidon::gpu::kernel launch configs)
    const uint32_t hntt_grid = (n / 8 + n_threads - 1) / n_threads;   // grid ~64 for n=4M
    hybrid_ntt_modup_fused_stage_kernel<<<hntt_grid, n_threads, 0, stream>>>(
        bufs[4], bufs[5], bufs[1], n, 0x2345u);
    hybrid_ntt_q_two_comp_fused_kernel<<<hntt_grid, n_threads, 0, stream>>>(
        bufs[6], bufs[3], bufs[5], bufs[1], n, 0x3456u);
    hybrid_moddown_ntt_add_back_kernel<<<hntt_grid * 4, n_threads, 0, stream>>>(
        bufs[3], bufs[4], bufs[6], bufs[1], n, 0x4567u);

    long_compute_kernel<<<grid, n_threads, 0, stream>>>(
        bufs[3], n, iters);
}

int main(int argc, char **argv) {
    uint32_t n = 1u << 22;           // 4M elements (16MB buffers)
    uint32_t iters = 3000;           // compute iterations inside long kernel
    int iterations = 50;             // per-mode repeats
    if (argc > 1) iterations = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    const std::size_t bytes = (std::size_t)n * sizeof(uint32_t);

    int dev_count = 0;
    CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count < 2) {
        fprintf(stderr, "need at least 2 GPUs\n");
        return 1;
    }

    cudaSetDevice(0);
    int can = 0;
    CHECK(cudaDeviceCanAccessPeer(&can, 0, 1));
    if (!can) { fprintf(stderr, "P2P 0<->1 not supported\n"); return 1; }
    cudaSetDevice(0);
    CHECK(cudaDeviceEnablePeerAccess(1, 0));
    cudaSetDevice(1);
    CHECK(cudaDeviceEnablePeerAccess(0, 0));
    cudaSetDevice(0);

    // Buffers on each GPU
    constexpr int kBufCount = 8;
    std::vector<uint32_t*> hbuf(kBufCount);
    for (int s = 0; s < kBufCount; ++s) CHECK(cudaMallocHost(&hbuf[s], bytes));
    for (int s = 0; s < kBufCount; ++s)
        for (uint32_t i = 0; i < n; ++i) hbuf[s][i] = (i * 2654435761u) & 0xffffffffu;

    uint32_t *bufs[2][kBufCount];
    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        for (int s = 0; s < kBufCount; ++s) CHECK(cudaMalloc(&bufs[g][s], bytes));
        for (int s = 0; s < kBufCount; ++s)
            CHECK(cudaMemcpy(bufs[g][s], hbuf[s], bytes, cudaMemcpyHostToDevice));
    }

    cudaStream_t streams[2];
    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        CHECK(cudaStreamCreate(&streams[g]));
    }
    cudaStream_t copy_stream;
    cudaSetDevice(0);
    CHECK(cudaStreamCreate(&copy_stream));

    printf("n=%u (%zu MB per buf), iterations=%d, long_iters=%u\n",
           n, bytes >> 20, iterations, iters);
    printf("mode: no-comm (baseline) then comm (P2P overlap)\n");
    fflush(stdout);

    // P2P chunk size for the interleaved copy (contiguous traffic)
    // Match the real MLP profile: P2P ~0.59MB @ ~68us there, so use 2MB here
    // (≈220us at 9GB/s) so each P2P overlaps several kernels on the single
    // compute stream, like the real 4-GPU run.
    const std::size_t p2p_chunk = 2u * 1024u * 1024u;   // 2 MB per copy
    const int p2p_copies = 16;                          // 16 * 2MB = 32MB per iter

    for (int mode = 0; mode < 2; ++mode) {
        const bool with_comm = (mode == 1);
        const char *mode_name = with_comm ? "comm" : "no-comm";
        for (int it = 0; it < iterations; ++it) {
            char range[64];
            snprintf(range, sizeof(range), "mode=%s iter=%d", mode_name, it);
            nvtx_push(range);

            // BOTH modes run the exact same kernel workload:
            //   p2p_copies rounds of launch_set on both GPUs.
            // The ONLY difference: comm mode additionally issues a P2P copy
            // (on the dedicated copy_stream) right before each launch round,
            // so the copy engine overlaps with the kernels.
            for (int c = 0; c < p2p_copies; ++c) {
                if (with_comm) {
                    CHECK(cudaMemcpyPeerAsync(
                        (char*)bufs[1][2] + (std::size_t)c * p2p_chunk, 1,
                        (char*)bufs[0][2] + (std::size_t)c * p2p_chunk, 0,
                        p2p_chunk, copy_stream));
                }
                cudaSetDevice(0);
                launch_set(streams[0], bufs[0], n, iters);
                cudaSetDevice(1);
                launch_set(streams[1], bufs[1], n, iters);
            }

            for (int g = 0; g < 2; ++g) {
                cudaSetDevice(g);
                CHECK(cudaStreamSynchronize(streams[g]));
            }
            if (with_comm) CHECK(cudaStreamSynchronize(copy_stream));

            nvtx_pop();
        }
    }

    // cleanup
    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        for (int s = 0; s < kBufCount; ++s) CHECK(cudaFree(bufs[g][s]));
        CHECK(cudaStreamDestroy(streams[g]));
    }
    cudaStreamDestroy(copy_stream);
    for (int s = 0; s < kBufCount; ++s) CHECK(cudaFreeHost(hbuf[s]));

    printf("done\n");
    return 0;
}
