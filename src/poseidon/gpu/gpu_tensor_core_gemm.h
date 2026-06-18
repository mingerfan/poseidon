#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace poseidon
{
namespace gpu
{

struct GpuGemmShape
{
    int m = 0;
    int n = 0;
    int k = 0;
};

struct TensorCoreU32GemmWorkspace
{
    std::uint8_t *a_segments = nullptr;
    std::uint8_t *b_segments_col_major = nullptr;
    std::int32_t *partial = nullptr;
};

struct TensorCoreU32GemmWorkspaceSizes
{
    std::size_t a_segments_bytes = 0;
    std::size_t b_segments_col_major_bytes = 0;
    std::size_t partial_bytes = 0;

    std::size_t total_bytes() const
    {
        return a_segments_bytes + b_segments_col_major_bytes + partial_bytes;
    }
};

inline constexpr int kTensorCoreIntegerTile = 16;
inline constexpr int kTensorCoreU32SegmentCount = 4;
inline constexpr int kTensorCoreU32Low32U8GemmCount = 10;
inline constexpr int kTensorCoreU8MaxKWithoutAccumulatorOverflow = 33025;

bool supports_tensor_core_integer_gemm(int device_id = -1);

TensorCoreU32GemmWorkspaceSizes tensor_core_u32_workspace_sizes(
    GpuGemmShape shape);

void launch_tensor_core_u8_gemm(
    const std::uint8_t *a_row_major,
    const std::uint8_t *b_col_major,
    std::int32_t *c_row_major,
    GpuGemmShape shape,
    cudaStream_t stream = nullptr);

void launch_tensor_core_s8_gemm(
    const std::int8_t *a_row_major,
    const std::int8_t *b_col_major,
    std::int32_t *c_row_major,
    GpuGemmShape shape,
    cudaStream_t stream = nullptr);

inline void launch_tensor_core_int8_gemm(
    const std::int8_t *a_row_major,
    const std::int8_t *b_col_major,
    std::int32_t *c_row_major,
    GpuGemmShape shape,
    cudaStream_t stream = nullptr)
{
    launch_tensor_core_s8_gemm(
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
    cudaStream_t stream = nullptr);

inline void launch_cuda_core_int8_gemm(
    const std::int8_t *a_row_major,
    const std::int8_t *b_col_major,
    std::int32_t *c_row_major,
    GpuGemmShape shape,
    cudaStream_t stream = nullptr)
{
    launch_cuda_core_s8_gemm(
        a_row_major,
        b_col_major,
        c_row_major,
        shape,
        stream);
}

void launch_cuda_core_u32_gemm(
    const std::uint32_t *a_row_major,
    const std::uint32_t *b_col_major,
    std::uint32_t *c_row_major,
    GpuGemmShape shape,
    cudaStream_t stream = nullptr);

void launch_split_u32_to_u8_segments(
    const std::uint32_t *a_row_major,
    const std::uint32_t *b_col_major,
    std::uint8_t *a_segments,
    std::uint8_t *b_segments_col_major,
    GpuGemmShape shape,
    cudaStream_t stream = nullptr);

void launch_tensor_core_u32_low32_gemm_from_segments(
    const std::uint8_t *a_segments,
    const std::uint8_t *b_segments_col_major,
    std::int32_t *partial,
    std::uint32_t *c_row_major,
    GpuGemmShape shape,
    cudaStream_t stream = nullptr);

void launch_tensor_core_u32_low32_gemm(
    const std::uint32_t *a_row_major,
    const std::uint32_t *b_col_major,
    std::uint32_t *c_row_major,
    GpuGemmShape shape,
    const TensorCoreU32GemmWorkspace &workspace,
    cudaStream_t stream = nullptr);

void launch_tensor_core_u32_mod_gemm_device_modulus(
    const std::uint32_t *a_row_major,
    const std::uint32_t *b_col_major,
    std::uint32_t *c_row_major,
    GpuGemmShape shape,
    const std::uint32_t *modulus,
    const TensorCoreU32GemmWorkspace &workspace,
    cudaStream_t stream = nullptr);

void launch_tensor_core_u32_mod_batched_gemm_from_segments(
    const std::uint8_t *a_segments,
    const std::uint8_t *b_segments_col_major,
    std::uint32_t *c_row_major,
    GpuGemmShape per_batch_shape,
    int batch_count,
    const std::uint32_t *modulus,
    cudaStream_t stream = nullptr);

void launch_tensor_core_u32_mod_batched_gemm_device_modulus(
    const std::uint32_t *a_row_major,
    const std::uint32_t *b_col_major,
    std::uint32_t *c_row_major,
    GpuGemmShape per_batch_shape,
    int batch_count,
    const std::uint32_t *modulus,
    const TensorCoreU32GemmWorkspace &workspace,
    cudaStream_t stream = nullptr);

}  // namespace gpu
}  // namespace poseidon
