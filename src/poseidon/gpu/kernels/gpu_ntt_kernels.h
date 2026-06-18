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

struct NttStageProfileSnapshot
{
    static constexpr std::size_t kMaxStageCount = 16;

    std::size_t stage_count = 0;
    double stage_total_ms[kMaxStageCount] = {};
    std::size_t stage_event_count[kMaxStageCount] = {};
};

void reset_ntt_stage_profile();

void set_ntt_stage_profile_enabled(bool enabled);

NttStageProfileSnapshot get_ntt_stage_profile_snapshot();

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

void launch_forward_ntt_components_shard_tensor(
    const GpuPolyShardView &first_destination_shard,
    const GpuConstPolyShardView &first_source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t component_count,
    std::size_t component_stride);

void launch_inverse_ntt_components_shard_tensor(
    const GpuPolyShardView &first_destination_shard,
    const GpuConstPolyShardView &first_source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t component_count,
    std::size_t component_stride);

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
