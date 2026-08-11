#pragma once

#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_rns_poly.h"

#include <cstddef>
#include <cstdint>

namespace poseidon
{
namespace gpu
{
namespace kernel
{

/** Fixed device-side indirection used by cached rotation CUDA Graphs. */
struct GpuRotationPointerBindings
{
    GpuWord *destination0 = nullptr;
    GpuWord *destination1 = nullptr;
    const GpuWord *source0 = nullptr;
    const GpuWord *source1 = nullptr;
};

void launch_update_rotation_pointer_bindings(
    GpuRotationPointerBindings *bindings,
    GpuWord *destination0,
    GpuWord *destination1,
    const GpuWord *source0,
    const GpuWord *source1,
    int device_id);

void launch_hybrid_modup_decomposition(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    const GpuWord *c2_ntt,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_modup_decomposition_forward_ntt_first_stage(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled8(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_modup_decomposition_row_tiled8(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_forward_ntt_qp_mul_accumulate_two_components(
    GpuWord *accum_q0,
    GpuWord *accum_p0,
    GpuWord *accum_q1,
    GpuWord *accum_p1,
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_ntt,
    const GpuWord *key_qp0,
    const GpuWord *key_qp1,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool overwrite_accum,
    bool fuse_decomp_q,
    bool skip_first_ntt_stage);

void launch_hybrid_multiply_accumulate(
    GpuWord *accum_q,
    GpuWord *accum_p,
    const GpuWord *modup_q,
    const GpuWord *modup_p,
    const GpuWord *key_qp,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_multiply_accumulate_two_components(
    GpuWord *accum_q0,
    GpuWord *accum_p0,
    GpuWord *accum_q1,
    GpuWord *accum_p1,
    const GpuWord *modup_q,
    const GpuWord *modup_p,
    const GpuWord *c2_ntt,
    const GpuWord *key_qp0,
    const GpuWord *key_qp1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    bool overwrite_accum);

void launch_hybrid_moddown(
    GpuWord *accum_q,
    const GpuWord *accum_p,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_convert_p_to_q(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_convert_p_to_q_forward_ntt(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_forward_ntt_q_two_components(
    GpuWord *values0,
    GpuWord *values1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_apply_moddown_ntt(
    GpuWord *accum_q0,
    GpuWord *accum_q1,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_apply_moddown_ntt_add_back(
    const GpuPolyShardView &destination_shard0,
    const GpuPolyShardView &destination_shard1,
    const GpuWord *accum_q0,
    const GpuWord *accum_q1,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool overwrite_destination1);

void launch_hybrid_apply_moddown_ntt_add_back_bound(
    const GpuRotationPointerBindings *bindings,
    const GpuPolyShardView &destination_shard0,
    const GpuPolyShardView &destination_shard1,
    const GpuWord *accum_q0,
    const GpuWord *accum_q1,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool overwrite_destination1);

void launch_apply_galois_ntt_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    std::uint32_t galois_elt,
    std::size_t degree);

void launch_apply_galois_ntt_poly_shard_bound(
    const GpuRotationPointerBindings *bindings,
    bool second_component,
    bool bind_destination,
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    std::uint32_t galois_elt,
    std::size_t degree);

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
