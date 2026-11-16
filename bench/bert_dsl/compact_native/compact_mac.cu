#include "compact_mac.h"
#include "compact_layout.h"
#include <cuda_runtime.h>

namespace bert::compact::detail {
namespace {
using poseidon::gpu::GpuWord;
using poseidon::gpu::GpuWide;

// Identical exact Barrett arithmetic to Poseidon's GPU elementwise and
// double-hoist kernels (their device helpers have anonymous-TU scope).
// In particular, no uint64 division/remainder appears in the MAC loop.
__device__ __forceinline__ GpuWord reduce_u64(
        GpuWide value, GpuWord q, GpuWide ratio) {
    const GpuWide quotient = __umul64hi(value, ratio);
    GpuWide r = value - quotient * static_cast<GpuWide>(q);
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    return static_cast<GpuWord>(r);
}
__device__ __forceinline__ GpuWord add_mod(GpuWord a, GpuWord b, GpuWord q) {
    const GpuWide sum = static_cast<GpuWide>(a) + b;
    return static_cast<GpuWord>(sum >= q ? sum - q : sum);
}

__global__ void compact_group_mac(
        GpuWord *output0, GpuWord *output1,
        const MacTerm *terms, std::size_t term_count,
        const GpuWord *primes, const GpuWide *barrett,
        std::size_t degree, std::size_t q_count, unsigned logn) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= degree * q_count) return;
    const std::size_t limb = tid >> logn;
    const auto j = static_cast<std::uint32_t>(tid & (degree - 1));
    const auto reversed = __brev(j) >> (32U - logn);
    const GpuWord q = primes[limb];
    const GpuWide ratio = barrett[limb];
    GpuWord sum0 = 0, sum1 = 0;
    for (std::size_t t = 0; t < term_count; ++t) {
        const MacTerm term = terms[t];
        // Full fallback stays in ordinary NTT order, as in core uploader.
        const auto index = term.period == degree ? j : reversed & (term.period - 1);
        const GpuWord p = term.plaintext[limb * term.period + index];
        sum0 = add_mod(sum0, reduce_u64(
            static_cast<GpuWide>(term.cipher0[tid]) * p, q, ratio), q);
        sum1 = add_mod(sum1, reduce_u64(
            static_cast<GpuWide>(term.cipher1[tid]) * p, q, ratio), q);
    }
    output0[tid] = sum0;
    output1[tid] = sum1;
}
} // namespace

void launch_compact_group_mac(
        GpuWord *output0, GpuWord *output1,
        const MacTerm *terms, std::size_t term_count,
        const GpuWord *q_primes, const GpuWide *q_barrett,
        std::size_t degree, std::size_t q_count) {
    if (!output0 || !output1 || !terms || !term_count || !q_primes || !q_barrett ||
        !power_of_two(degree) || !q_count)
        throw std::invalid_argument("invalid compressed Q MAC arguments");
    constexpr unsigned threads = 256;
    const auto blocks = static_cast<unsigned>((degree * q_count + threads - 1) / threads);
    compact_group_mac<<<blocks, threads, 0, poseidon::gpu::gpu_execution_stream()>>>(
        output0, output1, terms, term_count, q_primes, q_barrett,
        degree, q_count, log2_exact(degree));
    poseidon::gpu::gpu_check_cuda(cudaGetLastError(), "compact_group_mac launch");
}
} // namespace bert::compact::detail
