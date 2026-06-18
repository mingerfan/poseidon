#include "poseidon/gpu/gpu_tensor_core_gemm.h"

#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace poseidon
{
namespace gpu
{
namespace
{

using nvcuda::wmma::accumulator;
using nvcuda::wmma::col_major;
using nvcuda::wmma::fill_fragment;
using nvcuda::wmma::fragment;
using nvcuda::wmma::load_matrix_sync;
using nvcuda::wmma::matrix_a;
using nvcuda::wmma::matrix_b;
using nvcuda::wmma::mem_row_major;
using nvcuda::wmma::mma_sync;
using nvcuda::wmma::row_major;
using nvcuda::wmma::store_matrix_sync;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

constexpr int kSimtTileM = 16;
constexpr int kSimtTileN = 16;
constexpr int kSimtTileK = 64;

void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(status));
    }
}

int ceil_div(int x, int y)
{
    return (x + y - 1) / y;
}

int grid_1d(std::size_t total, int block)
{
    const std::size_t blocks =
        (total + static_cast<std::size_t>(block) - 1) /
        static_cast<std::size_t>(block);
    return static_cast<int>(
        std::max<std::size_t>(1, std::min<std::size_t>(blocks, 65535)));
}

std::size_t checked_mul(
    std::size_t left,
    std::size_t right,
    const char *what)
{
    if (left != 0 &&
        right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error(what);
    }
    return left * right;
}

void validate_positive_shape(GpuGemmShape shape, const char *name)
{
    if (shape.m <= 0 || shape.n <= 0 || shape.k <= 0)
    {
        throw std::invalid_argument(
            std::string(name) + ": matrix dimensions must be positive");
    }
}

void validate_wmma_shape(GpuGemmShape shape, const char *name)
{
    validate_positive_shape(shape, name);
    if (shape.m % kWmmaM != 0 ||
        shape.n % kWmmaN != 0 ||
        shape.k % kWmmaK != 0)
    {
        throw std::invalid_argument(
            std::string(name) +
            ": tensor-core path requires M, N, and K divisible by 16");
    }
}

void validate_u8_accumulator_limit(GpuGemmShape shape, const char *name)
{
    if (shape.k > kTensorCoreU8MaxKWithoutAccumulatorOverflow)
    {
        throw std::invalid_argument(
            std::string(name) +
            ": u8 WMMA partial sums can overflow int32 for this K");
    }
}

void validate_batch_count(int batch_count, const char *name)
{
    if (batch_count <= 0)
    {
        throw std::invalid_argument(
            std::string(name) + ": batch count must be positive");
    }
    if (batch_count > 65535)
    {
        throw std::invalid_argument(
            std::string(name) + ": batch count exceeds CUDA grid z limit");
    }
}

void require_tensor_core_integer_support(const char *name)
{
    if (!supports_tensor_core_integer_gemm())
    {
        throw std::runtime_error(
            std::string(name) + ": integer WMMA requires SM 7.5 or newer");
    }
}

void validate_not_null(const void *ptr, const char *name)
{
    if (ptr == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null pointer");
    }
}

std::size_t matrix_element_count(int rows, int cols)
{
    return checked_mul(
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols),
        "matrix element count overflow");
}

__device__ __forceinline__ int pack_s8x4(
    std::int8_t x0,
    std::int8_t x1,
    std::int8_t x2,
    std::int8_t x3)
{
    return static_cast<int>(
        static_cast<std::uint32_t>(static_cast<std::uint8_t>(x0)) |
        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(x1)) << 8) |
        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(x2)) << 16) |
        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(x3)) << 24));
}

__global__ void s8_wmma_gemm_kernel(
    const std::int8_t *a,
    const std::int8_t *b_col_major,
    std::int32_t *c,
    int m,
    int n,
    int k)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    const int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warp_n = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = warp_m * kWmmaM;
    const int col = warp_n * kWmmaN;

    if (row >= m || col >= n)
    {
        return;
    }

    fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, std::int8_t, row_major> a_frag;
    fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, std::int8_t, col_major> b_frag;
    fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, int> acc_frag;

    fill_fragment(acc_frag, 0);

    for (int kk = 0; kk < k; kk += kWmmaK)
    {
        load_matrix_sync(a_frag, a + static_cast<std::size_t>(row) * k + kk, k);
        load_matrix_sync(
            b_frag,
            b_col_major + static_cast<std::size_t>(col) * k + kk,
            k);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    store_matrix_sync(
        c + static_cast<std::size_t>(row) * n + col,
        acc_frag,
        n,
        mem_row_major);
#else
    (void)a;
    (void)b_col_major;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
    asm("trap;");
#endif
}

	__global__ void u8_wmma_gemm_kernel(
	    const std::uint8_t *a,
	    const std::uint8_t *b_col_major,
	    std::int32_t *c,
    int m,
    int n,
    int k)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    const int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warp_n = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = warp_m * kWmmaM;
    const int col = warp_n * kWmmaN;

    if (row >= m || col >= n)
    {
        return;
    }

    fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, std::uint8_t, row_major> a_frag;
    fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, std::uint8_t, col_major> b_frag;
    fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, int> acc_frag;

    fill_fragment(acc_frag, 0);

    for (int kk = 0; kk < k; kk += kWmmaK)
    {
        load_matrix_sync(a_frag, a + static_cast<std::size_t>(row) * k + kk, k);
        load_matrix_sync(
            b_frag,
            b_col_major + static_cast<std::size_t>(col) * k + kk,
            k);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    store_matrix_sync(
        c + static_cast<std::size_t>(row) * n + col,
        acc_frag,
        n,
        mem_row_major);
#else
    (void)a;
    (void)b_col_major;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
    asm("trap;");
	#endif
	}

	__device__ std::uint32_t pow2_mod_device(int shift, std::uint32_t modulus);

	__global__ void u8_wmma_mod_gemm_kernel(
	    const std::uint8_t *a_segments,
	    const std::uint8_t *b_segments_col_major,
	    std::uint32_t *c,
	    int m,
	    int n,
	    int k,
	    const std::uint32_t *modulus_ptr)
	{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
	    constexpr int kWarpTilesPerBlock =
	        (128 / 32) * 4;
	    __shared__ int tile_storage[kWarpTilesPerBlock][kWmmaM * kWmmaN];

	    const int warp_x = threadIdx.x / warpSize;
	    const int lane = threadIdx.x % warpSize;
	    const int warp_in_block = threadIdx.y * (blockDim.x / warpSize) + warp_x;
	    const int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	    const int warp_n = blockIdx.y * blockDim.y + threadIdx.y;
	    const int row = warp_m * kWmmaM;
	    const int col = warp_n * kWmmaN;

	    if (row >= m || col >= n)
	    {
	        return;
	    }

	    const std::uint32_t modulus = *modulus_ptr;
	    const std::size_t total_a =
	        static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
	    const std::size_t total_b =
	        static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
	    int *tile = tile_storage[warp_in_block];

	    fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, std::uint8_t, row_major> a_frag;
	    fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, std::uint8_t, col_major> b_frag;
	    fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, int> acc_frag;

	    bool first = true;
	    for (int a_segment = 0; a_segment < kTensorCoreU32SegmentCount;
	         ++a_segment)
	    {
	        for (int b_segment = 0; b_segment < kTensorCoreU32SegmentCount;
	             ++b_segment)
	        {
	            const std::uint8_t *a_base =
	                a_segments + static_cast<std::size_t>(a_segment) * total_a;
	            const std::uint8_t *b_base =
	                b_segments_col_major +
	                static_cast<std::size_t>(b_segment) * total_b;

	            fill_fragment(acc_frag, 0);
	            for (int kk = 0; kk < k; kk += kWmmaK)
	            {
	                load_matrix_sync(
	                    a_frag,
	                    a_base + static_cast<std::size_t>(row) * k + kk,
	                    k);
	                load_matrix_sync(
	                    b_frag,
	                    b_base + static_cast<std::size_t>(col) * k + kk,
	                    k);
	                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	            }

	            store_matrix_sync(tile, acc_frag, kWmmaN, mem_row_major);
	            __syncwarp();

	            const std::uint32_t factor =
	                pow2_mod_device((a_segment + b_segment) * 8, modulus);
	            for (int elem = lane; elem < kWmmaM * kWmmaN; elem += warpSize)
	            {
	                const int local_row = elem / kWmmaN;
	                const int local_col = elem - local_row * kWmmaN;
	                const int global_row = row + local_row;
	                const int global_col = col + local_col;
	                if (global_row < m && global_col < n)
	                {
	                    const std::uint32_t term =
	                        static_cast<std::uint32_t>(
	                            (static_cast<std::uint64_t>(
	                                 static_cast<std::uint32_t>(tile[elem])) *
	                             factor) %
	                            modulus);
	                    const std::size_t c_index =
	                        static_cast<std::size_t>(global_row) * n +
	                        static_cast<std::size_t>(global_col);
	                    if (first)
	                    {
	                        c[c_index] = term;
	                    }
	                    else
	                    {
	                        const std::uint64_t sum =
	                            static_cast<std::uint64_t>(c[c_index]) + term;
	                        c[c_index] = static_cast<std::uint32_t>(
	                            sum >= modulus ? sum - modulus : sum);
	                    }
	                }
	            }
	            __syncwarp();
	            first = false;
	        }
	    }
#else
	    (void)a_segments;
	    (void)b_segments_col_major;
	    (void)c;
	    (void)m;
	    (void)n;
	    (void)k;
	    (void)modulus_ptr;
	    asm("trap;");
	#endif
		}

    __global__ void u8_wmma_mod_batched_gemm_kernel(
        const std::uint8_t *a_segments,
        const std::uint8_t *b_segments_col_major,
        std::uint32_t *c,
        int m,
        int n,
        int k,
        int batch_count,
        const std::uint32_t *modulus_ptr)
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
        constexpr int kWarpTilesPerBlock =
            (128 / 32) * 4;
        __shared__ int tile_storage[kWarpTilesPerBlock][kWmmaM * kWmmaN];

        const int batch = static_cast<int>(blockIdx.z);
        if (batch >= batch_count)
        {
            return;
        }

        const int warp_x = threadIdx.x / warpSize;
        const int lane = threadIdx.x % warpSize;
        const int warp_in_block =
            threadIdx.y * (blockDim.x / warpSize) + warp_x;
        const int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
        const int warp_n = blockIdx.y * blockDim.y + threadIdx.y;
        const int row = warp_m * kWmmaM;
        const int col = warp_n * kWmmaN;

        if (row >= m || col >= n)
        {
            return;
        }

        const std::uint32_t modulus = *modulus_ptr;
        const std::size_t per_batch_a =
            static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
        const std::size_t per_batch_b =
            static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
        const std::size_t per_batch_c =
            static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
        const std::size_t total_a =
            static_cast<std::size_t>(batch_count) * per_batch_a;
        const std::size_t total_b =
            static_cast<std::size_t>(batch_count) * per_batch_b;
        int *tile = tile_storage[warp_in_block];

        fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, std::uint8_t, row_major> a_frag;
        fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, std::uint8_t, col_major> b_frag;
        fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, int> acc_frag;

        bool first = true;
        for (int a_segment = 0; a_segment < kTensorCoreU32SegmentCount;
             ++a_segment)
        {
            for (int b_segment = 0; b_segment < kTensorCoreU32SegmentCount;
                 ++b_segment)
            {
                const std::uint8_t *a_base =
                    a_segments + static_cast<std::size_t>(a_segment) * total_a +
                    static_cast<std::size_t>(batch) * per_batch_a;
                const std::uint8_t *b_base =
                    b_segments_col_major +
                    static_cast<std::size_t>(b_segment) * total_b +
                    static_cast<std::size_t>(batch) * per_batch_b;

                fill_fragment(acc_frag, 0);
                for (int kk = 0; kk < k; kk += kWmmaK)
                {
                    load_matrix_sync(
                        a_frag,
                        a_base + static_cast<std::size_t>(row) * k + kk,
                        k);
                    load_matrix_sync(
                        b_frag,
                        b_base + static_cast<std::size_t>(col) * k + kk,
                        k);
                    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }

                store_matrix_sync(tile, acc_frag, kWmmaN, mem_row_major);
                __syncwarp();

                const std::uint32_t factor =
                    pow2_mod_device((a_segment + b_segment) * 8, modulus);
                for (int elem = lane; elem < kWmmaM * kWmmaN; elem += warpSize)
                {
                    const int local_row = elem / kWmmaN;
                    const int local_col = elem - local_row * kWmmaN;
                    const int global_row = row + local_row;
                    const int global_col = col + local_col;
                    if (global_row < m && global_col < n)
                    {
                        const std::uint32_t term =
                            static_cast<std::uint32_t>(
                                (static_cast<std::uint64_t>(
                                     static_cast<std::uint32_t>(tile[elem])) *
                                 factor) %
                                modulus);
                        const std::size_t c_index =
                            static_cast<std::size_t>(batch) * per_batch_c +
                            static_cast<std::size_t>(global_row) * n +
                            static_cast<std::size_t>(global_col);
                        if (first)
                        {
                            c[c_index] = term;
                        }
                        else
                        {
                            const std::uint64_t sum =
                                static_cast<std::uint64_t>(c[c_index]) + term;
                            c[c_index] = static_cast<std::uint32_t>(
                                sum >= modulus ? sum - modulus : sum);
                        }
                    }
                }
                __syncwarp();
                first = false;
            }
        }
#else
        (void)a_segments;
        (void)b_segments_col_major;
        (void)c;
        (void)m;
        (void)n;
        (void)k;
        (void)batch_count;
        (void)modulus_ptr;
        asm("trap;");
#endif
    }

__global__ void s8_simt_dp4a_gemm_kernel(
    const std::int8_t *a,
    const std::int8_t *b_col_major,
    std::int32_t *c,
    int m,
    int n,
    int k)
{
    __shared__ std::int8_t a_tile[kSimtTileM][kSimtTileK];
    __shared__ std::int8_t b_tile[kSimtTileN][kSimtTileK];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int row = blockIdx.y * kSimtTileM + ty;
    const int col = blockIdx.x * kSimtTileN + tx;

    int acc = 0;

    for (int k0 = 0; k0 < k; k0 += kSimtTileK)
    {
        for (int idx = tid; idx < kSimtTileM * kSimtTileK;
             idx += blockDim.x * blockDim.y)
        {
            const int tile_row = idx / kSimtTileK;
            const int tile_k = idx - tile_row * kSimtTileK;
            const int global_row = blockIdx.y * kSimtTileM + tile_row;
            const int global_k = k0 + tile_k;
            a_tile[tile_row][tile_k] =
                (global_row < m && global_k < k)
                    ? a[static_cast<std::size_t>(global_row) * k + global_k]
                    : 0;
        }

        for (int idx = tid; idx < kSimtTileN * kSimtTileK;
             idx += blockDim.x * blockDim.y)
        {
            const int tile_col = idx / kSimtTileK;
            const int tile_k = idx - tile_col * kSimtTileK;
            const int global_col = blockIdx.x * kSimtTileN + tile_col;
            const int global_k = k0 + tile_k;
            b_tile[tile_col][tile_k] =
                (global_col < n && global_k < k)
                    ? b_col_major[static_cast<std::size_t>(global_col) * k +
                                  global_k]
                    : 0;
        }

        __syncthreads();

        if (row < m && col < n)
        {
#pragma unroll
            for (int kk = 0; kk < kSimtTileK; kk += 4)
            {
                const int a4 = pack_s8x4(
                    a_tile[ty][kk],
                    a_tile[ty][kk + 1],
                    a_tile[ty][kk + 2],
                    a_tile[ty][kk + 3]);
                const int b4 = pack_s8x4(
                    b_tile[tx][kk],
                    b_tile[tx][kk + 1],
                    b_tile[tx][kk + 2],
                    b_tile[tx][kk + 3]);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
                acc = __dp4a(a4, b4, acc);
#else
                acc += static_cast<int>(a_tile[ty][kk]) *
                       static_cast<int>(b_tile[tx][kk]);
                acc += static_cast<int>(a_tile[ty][kk + 1]) *
                       static_cast<int>(b_tile[tx][kk + 1]);
                acc += static_cast<int>(a_tile[ty][kk + 2]) *
                       static_cast<int>(b_tile[tx][kk + 2]);
                acc += static_cast<int>(a_tile[ty][kk + 3]) *
                       static_cast<int>(b_tile[tx][kk + 3]);
#endif
            }
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        c[static_cast<std::size_t>(row) * n + col] = acc;
    }
}

__global__ void u32_cuda_core_gemm_kernel(
    const std::uint32_t *a,
    const std::uint32_t *b_col_major,
    std::uint32_t *c,
    int m,
    int n,
    int k)
{
    __shared__ std::uint32_t a_tile[kSimtTileM][kSimtTileK];
    __shared__ std::uint32_t b_tile[kSimtTileN][kSimtTileK];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int row = blockIdx.y * kSimtTileM + ty;
    const int col = blockIdx.x * kSimtTileN + tx;

    std::uint32_t acc = 0;

    for (int k0 = 0; k0 < k; k0 += kSimtTileK)
    {
        for (int idx = tid; idx < kSimtTileM * kSimtTileK;
             idx += blockDim.x * blockDim.y)
        {
            const int tile_row = idx / kSimtTileK;
            const int tile_k = idx - tile_row * kSimtTileK;
            const int global_row = blockIdx.y * kSimtTileM + tile_row;
            const int global_k = k0 + tile_k;
            a_tile[tile_row][tile_k] =
                (global_row < m && global_k < k)
                    ? a[static_cast<std::size_t>(global_row) * k + global_k]
                    : 0;
        }

        for (int idx = tid; idx < kSimtTileN * kSimtTileK;
             idx += blockDim.x * blockDim.y)
        {
            const int tile_col = idx / kSimtTileK;
            const int tile_k = idx - tile_col * kSimtTileK;
            const int global_col = blockIdx.x * kSimtTileN + tile_col;
            const int global_k = k0 + tile_k;
            b_tile[tile_col][tile_k] =
                (global_col < n && global_k < k)
                    ? b_col_major[static_cast<std::size_t>(global_col) * k +
                                  global_k]
                    : 0;
        }

        __syncthreads();

        if (row < m && col < n)
        {
#pragma unroll
            for (int kk = 0; kk < kSimtTileK; ++kk)
            {
                acc += a_tile[ty][kk] * b_tile[tx][kk];
            }
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        c[static_cast<std::size_t>(row) * n + col] = acc;
    }
}

__global__ void split_u32_to_u8_segments_kernel(
    const std::uint32_t *a,
    const std::uint32_t *b_col_major,
    std::uint8_t *a_segments,
    std::uint8_t *b_segments_col_major,
    int m,
    int n,
    int k)
{
    const std::size_t total_a =
        static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
    const std::size_t total_b =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
    const std::size_t total = total_a > total_b ? total_a : total_b;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) *
        static_cast<std::size_t>(gridDim.x);

    for (std::size_t idx =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride)
    {
        if (idx < total_a)
        {
            const std::uint32_t value = a[idx];
            for (int segment = 0; segment < kTensorCoreU32SegmentCount;
                 ++segment)
            {
                a_segments[static_cast<std::size_t>(segment) * total_a + idx] =
                    static_cast<std::uint8_t>((value >> (segment * 8)) & 0xffu);
            }
        }

        if (idx < total_b)
        {
            const std::uint32_t value = b_col_major[idx];
            for (int segment = 0; segment < kTensorCoreU32SegmentCount;
                 ++segment)
            {
                b_segments_col_major
                    [static_cast<std::size_t>(segment) * total_b + idx] =
                        static_cast<std::uint8_t>(
                            (value >> (segment * 8)) & 0xffu);
            }
        }
    }
}

__global__ void split_u32_to_u8_segments_batched_kernel(
    const std::uint32_t *a,
    const std::uint32_t *b_col_major,
    std::uint8_t *a_segments,
    std::uint8_t *b_segments_col_major,
    int m,
    int n,
    int k,
    int batch_count)
{
    const std::size_t total_a =
        static_cast<std::size_t>(batch_count) *
        static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
    const std::size_t total_b =
        static_cast<std::size_t>(batch_count) *
        static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
    const std::size_t total = total_a > total_b ? total_a : total_b;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) *
        static_cast<std::size_t>(gridDim.x);

    for (std::size_t idx =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride)
    {
        if (idx < total_a)
        {
            const std::uint32_t value = a[idx];
            for (int segment = 0; segment < kTensorCoreU32SegmentCount;
                 ++segment)
            {
                a_segments[static_cast<std::size_t>(segment) * total_a + idx] =
                    static_cast<std::uint8_t>((value >> (segment * 8)) & 0xffu);
            }
        }

        if (idx < total_b)
        {
            const std::uint32_t value = b_col_major[idx];
            for (int segment = 0; segment < kTensorCoreU32SegmentCount;
                 ++segment)
            {
                b_segments_col_major
                    [static_cast<std::size_t>(segment) * total_b + idx] =
                        static_cast<std::uint8_t>(
                            (value >> (segment * 8)) & 0xffu);
            }
        }
    }
}

__global__ void fuse_low32_partial_kernel(
    const std::int32_t *partial,
    std::uint32_t *c,
    std::size_t total,
    int shift,
    bool first)
{
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) *
        static_cast<std::size_t>(gridDim.x);

    for (std::size_t idx =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride)
    {
        const std::uint32_t shifted =
            static_cast<std::uint32_t>(partial[idx]) << shift;
        c[idx] = first ? shifted : c[idx] + shifted;
    }
}

__device__ std::uint32_t pow2_mod_device(int shift, std::uint32_t modulus)
{
    std::uint32_t result = 1 % modulus;
    for (int i = 0; i < shift; ++i)
    {
        const std::uint64_t doubled =
            static_cast<std::uint64_t>(result) << 1;
        result = static_cast<std::uint32_t>(doubled % modulus);
    }
    return result;
}

void launch_s8_wmma_unchecked(
    const std::int8_t *a,
    const std::int8_t *b_col_major,
    std::int32_t *c,
    GpuGemmShape shape,
    cudaStream_t stream)
{
    dim3 block(128, 4, 1);
    dim3 grid(
        ceil_div(shape.m, kWmmaM * (static_cast<int>(block.x) / 32)),
        ceil_div(shape.n, kWmmaN * static_cast<int>(block.y)),
        1);
    s8_wmma_gemm_kernel<<<grid, block, 0, stream>>>(
        a,
        b_col_major,
        c,
        shape.m,
        shape.n,
        shape.k);
    check_cuda(cudaGetLastError(), "launch_tensor_core_s8_gemm kernel");
}

void launch_u8_wmma_unchecked(
    const std::uint8_t *a,
    const std::uint8_t *b_col_major,
    std::int32_t *c,
    GpuGemmShape shape,
    cudaStream_t stream)
{
    dim3 block(128, 4, 1);
    dim3 grid(
        ceil_div(shape.m, kWmmaM * (static_cast<int>(block.x) / 32)),
        ceil_div(shape.n, kWmmaN * static_cast<int>(block.y)),
        1);
    u8_wmma_gemm_kernel<<<grid, block, 0, stream>>>(
        a,
        b_col_major,
        c,
        shape.m,
        shape.n,
        shape.k);
    check_cuda(cudaGetLastError(), "launch_tensor_core_u8_gemm kernel");
}

void launch_u8_wmma_mod_unchecked(
    const std::uint8_t *a_segments,
    const std::uint8_t *b_segments_col_major,
    std::uint32_t *c,
    GpuGemmShape shape,
    const std::uint32_t *modulus,
    cudaStream_t stream)
{
    dim3 block(128, 4, 1);
    dim3 grid(
        ceil_div(shape.m, kWmmaM * (static_cast<int>(block.x) / 32)),
        ceil_div(shape.n, kWmmaN * static_cast<int>(block.y)),
        1);
    u8_wmma_mod_gemm_kernel<<<grid, block, 0, stream>>>(
        a_segments,
        b_segments_col_major,
        c,
        shape.m,
        shape.n,
        shape.k,
        modulus);
    check_cuda(cudaGetLastError(), "launch_tensor_core_u32_mod_gemm kernel");
}

void launch_u8_wmma_mod_batched_unchecked(
    const std::uint8_t *a_segments,
    const std::uint8_t *b_segments_col_major,
    std::uint32_t *c,
    GpuGemmShape shape,
    int batch_count,
    const std::uint32_t *modulus,
    cudaStream_t stream)
{
    dim3 block(128, 4, 1);
    dim3 grid(
        ceil_div(shape.m, kWmmaM * (static_cast<int>(block.x) / 32)),
        ceil_div(shape.n, kWmmaN * static_cast<int>(block.y)),
        batch_count);
    u8_wmma_mod_batched_gemm_kernel<<<grid, block, 0, stream>>>(
        a_segments,
        b_segments_col_major,
        c,
        shape.m,
        shape.n,
        shape.k,
        batch_count,
        modulus);
    check_cuda(
        cudaGetLastError(),
        "launch_tensor_core_u32_mod_batched_gemm kernel");
}

void launch_fuse_low32_partial_unchecked(
    const std::int32_t *partial,
    std::uint32_t *c,
    GpuGemmShape shape,
    int shift,
    bool first,
    cudaStream_t stream)
{
    constexpr int block = 256;
    const std::size_t total = matrix_element_count(shape.m, shape.n);
    fuse_low32_partial_kernel<<<grid_1d(total, block), block, 0, stream>>>(
        partial,
        c,
        total,
        shift,
        first);
    check_cuda(
        cudaGetLastError(),
        "launch_tensor_core_u32_low32_gemm fuse kernel");
}

void validate_u32_workspace(
    const TensorCoreU32GemmWorkspace &workspace,
    const char *name)
{
    validate_not_null(workspace.a_segments, name);
    validate_not_null(workspace.b_segments_col_major, name);
    validate_not_null(workspace.partial, name);
}

}  // namespace

bool supports_tensor_core_integer_gemm(int device_id)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        (void)cudaGetLastError();
        return false;
    }

    if (device_id < 0)
    {
        if (cudaGetDevice(&device_id) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }
    }

    if (device_id < 0 || device_id >= device_count)
    {
        return false;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
    {
        (void)cudaGetLastError();
        return false;
    }

    return prop.major * 10 + prop.minor >= 75;
}

TensorCoreU32GemmWorkspaceSizes tensor_core_u32_workspace_sizes(
    GpuGemmShape shape)
{
    validate_wmma_shape(shape, "tensor_core_u32_workspace_sizes");
    validate_u8_accumulator_limit(shape, "tensor_core_u32_workspace_sizes");

    const std::size_t total_a = matrix_element_count(shape.m, shape.k);
    const std::size_t total_b = matrix_element_count(shape.n, shape.k);
    const std::size_t total_c = matrix_element_count(shape.m, shape.n);

    TensorCoreU32GemmWorkspaceSizes sizes;
    sizes.a_segments_bytes = checked_mul(
        total_a,
        kTensorCoreU32SegmentCount * sizeof(std::uint8_t),
        "u32 tensor-core A segment workspace overflow");
    sizes.b_segments_col_major_bytes = checked_mul(
        total_b,
        kTensorCoreU32SegmentCount * sizeof(std::uint8_t),
        "u32 tensor-core B segment workspace overflow");
    sizes.partial_bytes = checked_mul(
        total_c,
        sizeof(std::int32_t),
        "u32 tensor-core partial workspace overflow");
    return sizes;
}

void launch_tensor_core_u8_gemm(
    const std::uint8_t *a_row_major,
    const std::uint8_t *b_col_major,
    std::int32_t *c_row_major,
    GpuGemmShape shape,
    cudaStream_t stream)
{
    validate_not_null(a_row_major, "launch_tensor_core_u8_gemm A");
    validate_not_null(b_col_major, "launch_tensor_core_u8_gemm B");
    validate_not_null(c_row_major, "launch_tensor_core_u8_gemm C");
    validate_wmma_shape(shape, "launch_tensor_core_u8_gemm");
    validate_u8_accumulator_limit(shape, "launch_tensor_core_u8_gemm");
    require_tensor_core_integer_support("launch_tensor_core_u8_gemm");

    launch_u8_wmma_unchecked(
        a_row_major,
        b_col_major,
        c_row_major,
        shape,
        stream);
}

void launch_tensor_core_s8_gemm(
    const std::int8_t *a_row_major,
    const std::int8_t *b_col_major,
    std::int32_t *c_row_major,
    GpuGemmShape shape,
    cudaStream_t stream)
{
    validate_not_null(a_row_major, "launch_tensor_core_s8_gemm A");
    validate_not_null(b_col_major, "launch_tensor_core_s8_gemm B");
    validate_not_null(c_row_major, "launch_tensor_core_s8_gemm C");
    validate_wmma_shape(shape, "launch_tensor_core_s8_gemm");
    require_tensor_core_integer_support("launch_tensor_core_s8_gemm");

    launch_s8_wmma_unchecked(
        a_row_major,
        b_col_major,
        c_row_major,
        shape,
        stream);
}

void launch_cuda_core_s8_gemm(
    const std::int8_t *a_row_major,
    const std::int8_t *b_col_major,
    std::int32_t *c_row_major,
    GpuGemmShape shape,
    cudaStream_t stream)
{
    validate_not_null(a_row_major, "launch_cuda_core_s8_gemm A");
    validate_not_null(b_col_major, "launch_cuda_core_s8_gemm B");
    validate_not_null(c_row_major, "launch_cuda_core_s8_gemm C");
    validate_positive_shape(shape, "launch_cuda_core_s8_gemm");

    dim3 block(kSimtTileN, kSimtTileM, 1);
    dim3 grid(
        ceil_div(shape.n, kSimtTileN),
        ceil_div(shape.m, kSimtTileM),
        1);
    s8_simt_dp4a_gemm_kernel<<<grid, block, 0, stream>>>(
        a_row_major,
        b_col_major,
        c_row_major,
        shape.m,
        shape.n,
        shape.k);
    check_cuda(cudaGetLastError(), "launch_cuda_core_s8_gemm kernel");
}

void launch_cuda_core_u32_gemm(
    const std::uint32_t *a_row_major,
    const std::uint32_t *b_col_major,
    std::uint32_t *c_row_major,
    GpuGemmShape shape,
    cudaStream_t stream)
{
    validate_not_null(a_row_major, "launch_cuda_core_u32_gemm A");
    validate_not_null(b_col_major, "launch_cuda_core_u32_gemm B");
    validate_not_null(c_row_major, "launch_cuda_core_u32_gemm C");
    validate_positive_shape(shape, "launch_cuda_core_u32_gemm");

    dim3 block(kSimtTileN, kSimtTileM, 1);
    dim3 grid(
        ceil_div(shape.n, kSimtTileN),
        ceil_div(shape.m, kSimtTileM),
        1);
    u32_cuda_core_gemm_kernel<<<grid, block, 0, stream>>>(
        a_row_major,
        b_col_major,
        c_row_major,
        shape.m,
        shape.n,
        shape.k);
    check_cuda(cudaGetLastError(), "launch_cuda_core_u32_gemm kernel");
}

void launch_split_u32_to_u8_segments(
    const std::uint32_t *a_row_major,
    const std::uint32_t *b_col_major,
    std::uint8_t *a_segments,
    std::uint8_t *b_segments_col_major,
    GpuGemmShape shape,
    cudaStream_t stream)
{
    validate_not_null(a_row_major, "launch_split_u32_to_u8_segments A");
    validate_not_null(b_col_major, "launch_split_u32_to_u8_segments B");
    validate_not_null(a_segments, "launch_split_u32_to_u8_segments A segments");
    validate_not_null(
        b_segments_col_major,
        "launch_split_u32_to_u8_segments B segments");
    validate_positive_shape(shape, "launch_split_u32_to_u8_segments");

    constexpr int block = 256;
    const std::size_t total_a = matrix_element_count(shape.m, shape.k);
    const std::size_t total_b = matrix_element_count(shape.n, shape.k);
    const std::size_t total = std::max(total_a, total_b);

    split_u32_to_u8_segments_kernel<<<grid_1d(total, block), block, 0, stream>>>(
        a_row_major,
        b_col_major,
        a_segments,
        b_segments_col_major,
        shape.m,
        shape.n,
        shape.k);
    check_cuda(cudaGetLastError(), "launch_split_u32_to_u8_segments kernel");
}

void launch_split_u32_to_u8_segments_batched(
    const std::uint32_t *a_row_major,
    const std::uint32_t *b_col_major,
    std::uint8_t *a_segments,
    std::uint8_t *b_segments_col_major,
    GpuGemmShape shape,
    int batch_count,
    cudaStream_t stream)
{
    validate_not_null(
        a_row_major,
        "launch_split_u32_to_u8_segments_batched A");
    validate_not_null(
        b_col_major,
        "launch_split_u32_to_u8_segments_batched B");
    validate_not_null(
        a_segments,
        "launch_split_u32_to_u8_segments_batched A segments");
    validate_not_null(
        b_segments_col_major,
        "launch_split_u32_to_u8_segments_batched B segments");
    validate_positive_shape(shape, "launch_split_u32_to_u8_segments_batched");
    validate_batch_count(
        batch_count,
        "launch_split_u32_to_u8_segments_batched");

    constexpr int block = 256;
    const std::size_t total_a =
        static_cast<std::size_t>(batch_count) *
        matrix_element_count(shape.m, shape.k);
    const std::size_t total_b =
        static_cast<std::size_t>(batch_count) *
        matrix_element_count(shape.n, shape.k);
    const std::size_t total = std::max(total_a, total_b);

    split_u32_to_u8_segments_batched_kernel<<<
        grid_1d(total, block), block, 0, stream>>>(
        a_row_major,
        b_col_major,
        a_segments,
        b_segments_col_major,
        shape.m,
        shape.n,
        shape.k,
        batch_count);
    check_cuda(
        cudaGetLastError(),
        "launch_split_u32_to_u8_segments_batched kernel");
}

void launch_tensor_core_u32_low32_gemm_from_segments(
    const std::uint8_t *a_segments,
    const std::uint8_t *b_segments_col_major,
    std::int32_t *partial,
    std::uint32_t *c_row_major,
    GpuGemmShape shape,
    cudaStream_t stream)
{
    validate_not_null(
        a_segments,
        "launch_tensor_core_u32_low32_gemm_from_segments A segments");
    validate_not_null(
        b_segments_col_major,
        "launch_tensor_core_u32_low32_gemm_from_segments B segments");
    validate_not_null(
        partial,
        "launch_tensor_core_u32_low32_gemm_from_segments partial");
    validate_not_null(
        c_row_major,
        "launch_tensor_core_u32_low32_gemm_from_segments C");
    validate_wmma_shape(
        shape,
        "launch_tensor_core_u32_low32_gemm_from_segments");
    validate_u8_accumulator_limit(
        shape,
        "launch_tensor_core_u32_low32_gemm_from_segments");
    require_tensor_core_integer_support(
        "launch_tensor_core_u32_low32_gemm_from_segments");

    const std::size_t total_a = matrix_element_count(shape.m, shape.k);
    const std::size_t total_b = matrix_element_count(shape.n, shape.k);
    bool first = true;

    for (int segment_sum = 0; segment_sum <= 3; ++segment_sum)
    {
        for (int a_segment = 0; a_segment <= segment_sum; ++a_segment)
        {
            const int b_segment = segment_sum - a_segment;
            const std::uint8_t *a_ptr =
                a_segments + static_cast<std::size_t>(a_segment) * total_a;
            const std::uint8_t *b_ptr =
                b_segments_col_major +
                static_cast<std::size_t>(b_segment) * total_b;

            launch_u8_wmma_unchecked(a_ptr, b_ptr, partial, shape, stream);
            launch_fuse_low32_partial_unchecked(
                partial,
                c_row_major,
                shape,
                segment_sum * 8,
                first,
                stream);
            first = false;
        }
    }
}

void launch_tensor_core_u32_low32_gemm(
    const std::uint32_t *a_row_major,
    const std::uint32_t *b_col_major,
    std::uint32_t *c_row_major,
    GpuGemmShape shape,
    const TensorCoreU32GemmWorkspace &workspace,
    cudaStream_t stream)
{
    validate_not_null(a_row_major, "launch_tensor_core_u32_low32_gemm A");
    validate_not_null(b_col_major, "launch_tensor_core_u32_low32_gemm B");
    validate_not_null(c_row_major, "launch_tensor_core_u32_low32_gemm C");
    validate_u32_workspace(workspace, "launch_tensor_core_u32_low32_gemm");
    validate_wmma_shape(shape, "launch_tensor_core_u32_low32_gemm");
    validate_u8_accumulator_limit(
        shape,
        "launch_tensor_core_u32_low32_gemm");
    require_tensor_core_integer_support(
        "launch_tensor_core_u32_low32_gemm");

    launch_split_u32_to_u8_segments(
        a_row_major,
        b_col_major,
        workspace.a_segments,
        workspace.b_segments_col_major,
        shape,
        stream);
    launch_tensor_core_u32_low32_gemm_from_segments(
        workspace.a_segments,
        workspace.b_segments_col_major,
        workspace.partial,
        c_row_major,
        shape,
        stream);
}

void launch_tensor_core_u32_mod_gemm_device_modulus(
    const std::uint32_t *a_row_major,
    const std::uint32_t *b_col_major,
    std::uint32_t *c_row_major,
    GpuGemmShape shape,
    const std::uint32_t *modulus,
    const TensorCoreU32GemmWorkspace &workspace,
    cudaStream_t stream)
{
    validate_not_null(
        a_row_major,
        "launch_tensor_core_u32_mod_gemm_device_modulus A");
    validate_not_null(
        b_col_major,
        "launch_tensor_core_u32_mod_gemm_device_modulus B");
    validate_not_null(
        c_row_major,
        "launch_tensor_core_u32_mod_gemm_device_modulus C");
    validate_not_null(
        modulus,
        "launch_tensor_core_u32_mod_gemm_device_modulus modulus");
    validate_u32_workspace(
        workspace,
        "launch_tensor_core_u32_mod_gemm_device_modulus");
    validate_wmma_shape(
        shape,
        "launch_tensor_core_u32_mod_gemm_device_modulus");
    validate_u8_accumulator_limit(
        shape,
        "launch_tensor_core_u32_mod_gemm_device_modulus");
    require_tensor_core_integer_support(
        "launch_tensor_core_u32_mod_gemm_device_modulus");

    launch_split_u32_to_u8_segments(
        a_row_major,
        b_col_major,
        workspace.a_segments,
        workspace.b_segments_col_major,
        shape,
        stream);

    launch_u8_wmma_mod_unchecked(
        workspace.a_segments,
        workspace.b_segments_col_major,
        c_row_major,
        shape,
        modulus,
        stream);
}

void launch_tensor_core_u32_mod_batched_gemm_from_segments(
    const std::uint8_t *a_segments,
    const std::uint8_t *b_segments_col_major,
    std::uint32_t *c_row_major,
    GpuGemmShape per_batch_shape,
    int batch_count,
    const std::uint32_t *modulus,
    cudaStream_t stream)
{
    validate_not_null(
        a_segments,
        "launch_tensor_core_u32_mod_batched_gemm_from_segments A segments");
    validate_not_null(
        b_segments_col_major,
        "launch_tensor_core_u32_mod_batched_gemm_from_segments B segments");
    validate_not_null(
        c_row_major,
        "launch_tensor_core_u32_mod_batched_gemm_from_segments C");
    validate_not_null(
        modulus,
        "launch_tensor_core_u32_mod_batched_gemm_from_segments modulus");
    validate_wmma_shape(
        per_batch_shape,
        "launch_tensor_core_u32_mod_batched_gemm_from_segments");
    validate_u8_accumulator_limit(
        per_batch_shape,
        "launch_tensor_core_u32_mod_batched_gemm_from_segments");
    validate_batch_count(
        batch_count,
        "launch_tensor_core_u32_mod_batched_gemm_from_segments");
    require_tensor_core_integer_support(
        "launch_tensor_core_u32_mod_batched_gemm_from_segments");

    launch_u8_wmma_mod_batched_unchecked(
        a_segments,
        b_segments_col_major,
        c_row_major,
        per_batch_shape,
        batch_count,
        modulus,
        stream);
}

void launch_tensor_core_u32_mod_batched_gemm_device_modulus(
    const std::uint32_t *a_row_major,
    const std::uint32_t *b_col_major,
    std::uint32_t *c_row_major,
    GpuGemmShape per_batch_shape,
    int batch_count,
    const std::uint32_t *modulus,
    const TensorCoreU32GemmWorkspace &workspace,
    cudaStream_t stream)
{
    validate_not_null(
        a_row_major,
        "launch_tensor_core_u32_mod_batched_gemm_device_modulus A");
    validate_not_null(
        b_col_major,
        "launch_tensor_core_u32_mod_batched_gemm_device_modulus B");
    validate_not_null(
        c_row_major,
        "launch_tensor_core_u32_mod_batched_gemm_device_modulus C");
    validate_not_null(
        modulus,
        "launch_tensor_core_u32_mod_batched_gemm_device_modulus modulus");
    validate_u32_workspace(
        workspace,
        "launch_tensor_core_u32_mod_batched_gemm_device_modulus");
    validate_wmma_shape(
        per_batch_shape,
        "launch_tensor_core_u32_mod_batched_gemm_device_modulus");
    validate_u8_accumulator_limit(
        per_batch_shape,
        "launch_tensor_core_u32_mod_batched_gemm_device_modulus");
    validate_batch_count(
        batch_count,
        "launch_tensor_core_u32_mod_batched_gemm_device_modulus");
    require_tensor_core_integer_support(
        "launch_tensor_core_u32_mod_batched_gemm_device_modulus");

    launch_split_u32_to_u8_segments_batched(
        a_row_major,
        b_col_major,
        workspace.a_segments,
        workspace.b_segments_col_major,
        per_batch_shape,
        batch_count,
        stream);

    launch_tensor_core_u32_mod_batched_gemm_from_segments(
        workspace.a_segments,
        workspace.b_segments_col_major,
        c_row_major,
        per_batch_shape,
        batch_count,
        modulus,
        stream);
}

}  // namespace gpu
}  // namespace poseidon
