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

void launch_forward_ntt_poly_shard_fourstep_65536(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_forward_ntt_qp_active_fourstep_65536(
    GpuWord *destination_q,
    GpuWord *destination_p,
    const GpuWord *source_q,
    const GpuWord *source_p,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool phase1_ready = false);

void launch_hybrid_modup_p9_forward_ntt_qp_active_phase1_fourstep_65536(
    GpuWord *destination_q,
    GpuWord *destination_p,
    const GpuWord *c2_coeff,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool source_preweighted = false,
    unsigned int target_limb_tile = 1);

void launch_forward_ntt_qp_active_fourstep_mul_accumulate_two_components_65536(
    GpuWord *partial_q,
    GpuWord *partial_p,
    const GpuWord *source_q,
    const GpuWord *source_p,
    GpuWord *accum_q0,
    GpuWord *accum_p0,
    GpuWord *accum_q1,
    GpuWord *accum_p1,
    const GpuWord *c2_ntt,
    const GpuWord *key_q0,
    const GpuWord *key_p0,
    const GpuWord *key_q1,
    const GpuWord *key_p1,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    bool overwrite_accum,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool phase1_ready = false);

void launch_hybrid_convert_p_to_q_forward_ntt_two_components_fourstep_65536(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    const GpuWord *source_p0,
    const GpuWord *source_p1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_convert_p9_to_q_forward_ntt_two_components_fourstep_65536(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    const GpuWord *source_p0,
    const GpuWord *source_p1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool source_preweighted = false);

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

/**
 * @brief Launch the optimized out-of-place two-phase inverse NTT for N=65536.
 */
void launch_inverse_ntt_poly_shard_fourstep_65536(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * Apply the N=65536 two-phase inverse NTT to equally shaped, strided RNS
 * polynomials. The batch dimension is folded into grid.z so all inputs share
 * one phase-1 launch and one phase-2 launch.
 */
void launch_inverse_ntt_poly_shard_batch_fourstep_65536(
    const GpuPolyShardView &first_destination_shard,
    const GpuConstPolyShardView &first_source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t batch_count,
    std::size_t destination_batch_stride,
    std::size_t source_batch_stride);

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
