#pragma once

#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_rns_poly.h"

#include <cstddef>

namespace poseidon
{
namespace gpu
{
namespace kernel
{

struct GpuNttKernelOccupancyInfo
{
    int device_id = 0;
    int block_size = 0;
    int sm_count = 0;
    int warp_size = 0;
    int max_threads_per_sm = 0;
    int max_threads_per_block = 0;
    int active_blocks_per_sm = 0;
    int active_threads_per_sm = 0;
    int active_warps_per_sm = 0;
    int theoretical_active_blocks = 0;
    double occupancy = 0.0;
};

/**
 * @brief Launch a simple Barrett-based forward NTT for one aligned RNS-poly shard.
 *
 * First implementation constraints:
 * - source and destination must cover full coefficient limbs;
 * - q limbs only;
 * - roots are read from GpuParameterShard::ntt_tables in
 *   [local limb][degree roots] layout.
 */
void launch_forward_ntt_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Launch a diagnostic forward NTT with all stages inside one kernel.
 *
 * This variant assigns one CUDA block to one RNS limb and uses __syncthreads()
 * between stages. It is intended for measuring launch overhead and stage
 * synchronization behavior, not as the default high-throughput implementation.
 */
void launch_forward_ntt_poly_shard_single_kernel(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_forward_ntt_poly_shard_single_kernel_with_block_size(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    int block_size);

GpuNttKernelOccupancyInfo query_forward_ntt_single_kernel_occupancy(
    int device_id,
    int block_size);

/**
 * @brief Launch exactly one forward NTT stage in-place for one RNS-poly shard.
 *
 * This is mainly useful for micro-benchmarks and stage-level experiments.
 * The caller provides the same m/gap pair used by the full forward NTT loop:
 * for (m = 1, gap = degree / 2; m < degree; m <<= 1, gap >>= 1).
 */
void launch_forward_ntt_stage_poly_shard(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t m,
    std::size_t gap);

/**
 * @brief Launch a simple Barrett-based inverse NTT for one aligned RNS-poly shard.
 *
 * The inverse transform performs the final multiplication by N^{-1} with
 * GpuParameterShard::inv_degree_modulo.
 */
void launch_inverse_ntt_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
