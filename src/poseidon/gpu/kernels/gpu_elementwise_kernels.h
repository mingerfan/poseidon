#pragma once

#include "poseidon/gpu/gpu_rns_poly.h"
#include "poseidon/gpu/gpu_parameter.h"

#include <cstddef>

namespace poseidon
{
namespace gpu
{
namespace kernel
{

/**
 * @brief Launch modular addition kernel for one aligned RNS-poly shard.
 *
 * This is a host-side launcher function.
 *
 * Future responsibility:
 * - select CUDA device;
 * - compute grid/block configuration;
 * - extract raw device pointers from shard views;
 * - extract modulus tables from parameter_shard;
 * - launch __global__ CUDA kernel.
 */
void launch_add_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &left_shard,
    const GpuConstPolyShardView &right_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Launch one kernel that adds two aligned ciphertext component shards.
 *
 * This is a small ciphertext-level fast path for common CKKS ciphertexts:
 * - component 0 and component 1 have the same shard shape;
 * - both components live on the same device;
 * - one kernel processes both component shards.
 */
void launch_add_two_poly_shards(
    const GpuPolyShardView &destination_shard0,
    const GpuPolyShardView &destination_shard1,
    const GpuConstPolyShardView &left_shard0,
    const GpuConstPolyShardView &left_shard1,
    const GpuConstPolyShardView &right_shard0,
    const GpuConstPolyShardView &right_shard1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Launch modular subtraction kernel for one aligned RNS-poly shard.
 */
void launch_sub_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &left_shard,
    const GpuConstPolyShardView &right_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Launch modular negation kernel for one aligned RNS-poly shard.
 */
void launch_negate_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Launch device-to-device copy kernel for one aligned RNS-poly shard.
 *
 * This operation does not need modulus parameters.
 */
void launch_copy_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    std::size_t degree);

/**
 * @brief Launch dyadic modular product kernel for one aligned RNS-poly shard.
 *
 * Used by multiply_plain.
 */
void launch_dyadic_product_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &left_shard,
    const GpuConstPolyShardView &right_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Launch dyadic modular multiply-accumulate kernel for one aligned shard.
 *
 * Used by ciphertext-ciphertext multiplication:
 * destination += left * right mod q.
 */
void launch_multiply_accumulate_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &left_shard,
    const GpuConstPolyShardView &right_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
