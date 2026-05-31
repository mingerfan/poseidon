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
