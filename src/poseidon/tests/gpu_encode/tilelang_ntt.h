#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Experimental N=65536, q<2^31, contiguous [batch][limb][N]. Buffers are
// out-of-place on entry; subsequent stages update destination in-place.
cudaError_t launch_encode_tilelang_ntt(
    const std::uint32_t *source, std::uint32_t *values,
    const std::uint32_t *roots, const std::uint32_t *matrices,
    const std::uint32_t *primes, const std::uint64_t *ratios,
    const std::uint32_t *weights, int limbs, int batch,
    bool tensor, cudaStream_t stream);
