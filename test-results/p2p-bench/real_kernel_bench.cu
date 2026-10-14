// Real-kernel microbenchmark: apply_galois_ntt_poly_shard_kernel
// (THE actual MLP kernel, copied verbatim from
//  poseidon-gpu-worker-experiment/src/poseidon/gpu/kernels/gpu_keyswitch_kernels.cu)
//
// N = 8192 (MLP poly degree), limb_count = 17 (MLP q_modulus_count).
// grid/block identical to the real launch (block=256, grid=(limb*degree)/256).
//
// Measures via nsys, per mode (no-comm / comm):
//   1. kernel LAUNCH time   (CPU-side cudaLaunchKernel duration)
//   2. kernel COMPUTE time  (GPU-side execution duration)
// comm mode additionally overlaps P2P copies on a dedicated copy stream,
// on a buffer unrelated to the galois kernel (isolates bandwidth, not data).
//
// Single compute stream per GPU (matches real runtime).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
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
    std::uint32_t value,
    unsigned int bit_count)
{
    return __brev(value) >> (32U - bit_count);
}

__global__ void apply_galois_ntt_poly_shard_kernel(
    GpuWord *destination,
    const GpuWord *source,
    std::uint32_t galois_elt,
    std::size_t limb_count,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * degree;
    if (tid >= total)
    {
        return;
    }

    const std::uint32_t degree_u32 = static_cast<std::uint32_t>(degree);
    const std::uint32_t degree_minus_one = degree_u32 - 1;
    const std::size_t limb = tid >> degree_power;
    const std::uint32_t coeff =
        static_cast<std::uint32_t>(tid & degree_minus_one);

    const std::uint32_t reversed =
        reverse_bits_limited(degree_u32 + coeff, degree_power + 1);
    const std::uint64_t index_raw =
        (static_cast<std::uint64_t>(galois_elt) *
         static_cast<std::uint64_t>(reversed)) >> 1;
    const std::uint32_t source_coeff =
        reverse_bits_limited(
            static_cast<std::uint32_t>(index_raw & degree_minus_one),
            degree_power);

    destination[limb * degree + coeff] =
        source[limb * degree + source_coeff];
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    const std::size_t degree = 8192;      // MLP N
    const std::size_t limb_count = 17;    // MLP q count
    const int iterations = 30;
    const std::uint32_t galois_elt = 3;

    int dev_count = 0;
    CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count < 2) { fprintf(stderr, "need 2 GPUs\n"); return 1; }

    cudaSetDevice(0);
    int can = 0;
    CHECK(cudaDeviceCanAccessPeer(&can, 0, 1));
    if (!can) { fprintf(stderr, "P2P 0<->1 not supported\n"); return 1; }
    cudaSetDevice(0); CHECK(cudaDeviceEnablePeerAccess(1, 0));
    cudaSetDevice(1); CHECK(cudaDeviceEnablePeerAccess(0, 0));
    cudaSetDevice(0);

    const std::size_t poly_bytes = degree * limb_count * sizeof(GpuWord);
    const std::size_t p2p_bytes = 64u << 20;  // 64MB P2P buffer (16*2MB+slack)

    // host init
    std::vector<GpuWord> hsrc(poly_bytes / sizeof(GpuWord), 0x12345678u);

    // buffers: [gpu][0]=src poly, [gpu][1]=dst poly, [gpu][2]=P2P buf
    GpuWord *bufs[2][3];
    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        CHECK(cudaMalloc(&bufs[g][0], poly_bytes));
        CHECK(cudaMalloc(&bufs[g][1], poly_bytes));
        CHECK(cudaMalloc(&bufs[g][2], p2p_bytes));
        CHECK(cudaMemcpy(bufs[g][0], hsrc.data(), poly_bytes, cudaMemcpyHostToDevice));
    }

    cudaStream_t stream[2];
    for (int g = 0; g < 2; ++g) { cudaSetDevice(g); CHECK(cudaStreamCreate(&stream[g])); }
    cudaStream_t copy_stream;
    cudaSetDevice(0); CHECK(cudaStreamCreate(&copy_stream));

    const std::size_t p2p_chunk = 2u << 20;  // 2MB per P2P (matches MLP-scale traffic)
    const int p2p_copies = 16;

    // real launch config: block=256, grid=(limb*degree)/256
    constexpr int block = 256;
    const unsigned int degree_power = 13;  // log2(8192)
    const int grid = static_cast<int>((limb_count * degree + block - 1) / block);

    printf("degree=%zu limb=%zu grid=%d block=%d P2P=2MBx%d iter=%d\n",
           degree, limb_count, grid, block, p2p_copies, iterations);
    fflush(stdout);

    for (int mode = 0; mode < 2; ++mode) {
        const bool with_comm = (mode == 1);
        const char *mode_name = with_comm ? "comm" : "no-comm";
        for (int it = 0; it < iterations; ++it) {
            for (int c = 0; c < p2p_copies; ++c) {
                if (with_comm) {
                    cudaSetDevice(0);
                    CHECK(cudaMemcpyPeerAsync(
                        (char*)bufs[1][2] + (std::size_t)c * p2p_chunk, 1,
                        (char*)bufs[0][2] + (std::size_t)c * p2p_chunk, 0,
                        p2p_chunk, copy_stream));
                }
                for (int g = 0; g < 2; ++g) {
                    cudaSetDevice(g);
                    apply_galois_ntt_poly_shard_kernel<<<grid, block, 0, stream[g]>>>(
                        bufs[g][1], bufs[g][0], galois_elt, limb_count, degree, degree_power);
                    cudaError_t e = cudaGetLastError();
                    if (e != cudaSuccess) {
                        fprintf(stderr, "galois launch err: %s\n", cudaGetErrorString(e));
                        return 1;
                    }
                }
            }
            for (int g = 0; g < 2; ++g) { cudaSetDevice(g); CHECK(cudaStreamSynchronize(stream[g])); }
            if (with_comm) CHECK(cudaStreamSynchronize(copy_stream));
        }
    }

    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        for (int s = 0; s < 3; ++s) CHECK(cudaFree(bufs[g][s]));
        CHECK(cudaStreamDestroy(stream[g]));
    }
    cudaStreamDestroy(copy_stream);
    printf("done\n");
    return 0;
}
