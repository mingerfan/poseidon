#pragma once

#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_parameter.h"

#include <cstddef>
#include <cstdint>

namespace poseidon
{
namespace gpu
{
namespace kernel
{

void launch_double_hoist_pre_rotated_keymul_digit(
    GpuWord *destination_q0,
    GpuWord *destination_p0,
    GpuWord *destination_q1,
    GpuWord *destination_p1,
    const GpuWord *digit_q,
    const GpuWord *digit_p,
    const GpuWord *key_q0,
    const GpuWord *key_p0,
    const GpuWord *key_q1,
    const GpuWord *key_p1,
    std::uint32_t galois_elt,
    bool overwrite,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_double_hoist_pre_rotated_keymul_batch(
    GpuWord *destination_q0,
    GpuWord *destination_p0,
    GpuWord *destination_q1,
    GpuWord *destination_p1,
    const GpuWord *digits_q,
    const GpuWord *digits_p,
    const GpuWord *const *key_q0_ptrs,
    const GpuWord *const *key_p0_ptrs,
    const GpuWord *const *key_q1_ptrs,
    const GpuWord *const *key_p1_ptrs,
    std::size_t dnum,
    std::uint32_t galois_elt,
    bool overwrite,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_double_hoist_lift_identity(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    GpuWord *destination_p0,
    GpuWord *destination_p1,
    const GpuWord *source_q0,
    const GpuWord *source_q1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool accumulate);

void launch_double_hoist_add_lifted_galois_c0(
    GpuWord *destination_q0,
    const GpuWord *source_q0,
    std::uint32_t galois_elt,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * Fuse all non-identity giant rotations into one Q launch and one P launch.
 * The grid spans group x limb x coefficient and writes the existing group QP
 * workspace. Reduction is deliberately deferred to the fused outer ModDown.
 */
void launch_double_hoist_pre_rotated_giant_group_reduce(
    GpuWord *scratch_group_q,
    GpuWord *scratch_group_p,
    const GpuWord *inner_q_batch,
    const GpuWord *const *host_group_digit_q_ptrs,
    const GpuWord *const *host_group_digit_p_ptrs,
    const std::uint32_t *host_group_indices,
    const std::uint32_t *host_galois_elts,
    const std::uint32_t *host_key_indices,
    const GpuWord *const *key_q0_ptrs,
    const GpuWord *const *key_p0_ptrs,
    const GpuWord *const *key_q1_ptrs,
    const GpuWord *const *key_p1_ptrs,
    std::size_t active_group_count,
    std::size_t dnum,
    std::size_t storage_dnum,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_double_hoist_reduce_p_groups(
    GpuWord *destination_p0,
    GpuWord *destination_p1,
    const GpuWord *group_p,
    std::size_t group_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_double_hoist_qp_plain_mul_accumulate_groups(
    GpuWord *group_q,
    GpuWord *group_p,
    const GpuWord *baby_q,
    const GpuWord *baby_p,
    const GpuWord *const *diagonal_q_ptrs,
    const GpuWord *const *diagonal_p_ptrs,
    const std::uint32_t *term_baby_indices,
    const std::uint32_t *group_term_offsets,
    std::size_t group_count,
    std::size_t term_count,
    std::size_t tile_begin,
    std::size_t tile_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
