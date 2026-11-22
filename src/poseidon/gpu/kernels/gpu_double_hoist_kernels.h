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

// The largest baby-step batch and giant-group count carried by one fused
// KeySwitch/plaintext-MAC launch. Keep these shared with host-side argument
// packing so complete direct stages issue in a single Q/P launch pair.
inline constexpr std::size_t kMaxDoubleHoistFusedBabySteps = 16;
inline constexpr std::size_t kMaxDoubleHoistFusedBabyGroups = 8;

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
    const GpuWord *lifted_c0_source_q,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

// The largest rotation batch carried by one multi-step InversePreRotated keymul
// launch. Callers with more steps chunk cleanly across batches of this size.
inline constexpr std::size_t kMaxDoubleHoistKeymulManySteps = 32;

struct DoubleHoistKeymulManyArgs
{
    std::uint32_t galois_elts[kMaxDoubleHoistKeymulManySteps];
    std::uint32_t key_indices[kMaxDoubleHoistKeymulManySteps];
};

/**
 * Compute InversePreRotated HYBRID key-multiplies (and optional lifted c0)
 * for up to kMaxDoubleHoistKeymulManySteps rotation steps against one
 * hoisted source decomposition. Steps with galois_elt == 0 emit the lifted
 * identity ciphertext.
 */
void launch_double_hoist_pre_rotated_keymul_many(
    GpuWord *destination_q,
    GpuWord *destination_p,
    const GpuWord *digits_q,
    const GpuWord *digits_p,
    const GpuWord *lifted_c0_source_q,
    const GpuWord *lifted_c1_source_q,
    const GpuWord *const *key_q0_ptrs,
    const GpuWord *const *key_p0_ptrs,
    const GpuWord *const *key_q1_ptrs,
    const GpuWord *const *key_p1_ptrs,
    const DoubleHoistKeymulManyArgs &args,
    std::size_t step_count,
    std::size_t dnum,
    std::size_t storage_dnum,
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

/**
 * Produce one QP accumulator for all giant groups. Each thread owns one
 * (limb, coefficient) output, walks the non-identity groups in registers,
 * folds in the identity group, and performs a single global store per
 * ciphertext component. This avoids materializing one QP ciphertext per
 * giant group before the shared outer ModDown.
 */
void launch_double_hoist_pre_rotated_giant_group_accumulate(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    GpuWord *destination_p0,
    GpuWord *destination_p1,
    const GpuWord *inner_q_batch,
    std::size_t identity_group_index,
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

inline constexpr std::size_t kMaxGiantReduceBatch = 8;
inline constexpr std::size_t kMaxGiantReduceBatchGroups = 16;

void launch_double_hoist_pre_rotated_giant_group_accumulate_batch(
    GpuWord *destination_q,
    GpuWord *destination_p,
    std::size_t batch_count,
    const GpuWord *inner_q_batch,
    std::size_t inner_batch_stride_words,
    std::size_t identity_group_index,
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

/**
 * Like launch_double_hoist_pre_rotated_giant_group_accumulate, but c0 of each
 * giant group and the identity group are read from the QP group accumulators
 * (group_q / group_p, [group][component] layout), so only c1 of the
 * non-identity groups needs the inner ModDown.
 */
void launch_double_hoist_pre_rotated_giant_group_accumulate_qp_c0(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    GpuWord *destination_p0,
    GpuWord *destination_p1,
    const GpuWord *group_q,
    const GpuWord *group_p,
    std::size_t identity_group_index,
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
    const std::uint32_t *diagonal_periods,
    const std::uint32_t *term_baby_indices,
    const std::uint32_t *group_term_offsets,
    std::size_t group_count,
    std::size_t term_count,
    std::size_t tile_begin,
    std::size_t tile_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool compressed_plaintexts,
    bool initialize_accumulators);

/**
 * Fuse inverse-pre-rotated baby KeySwitch with QP plaintext MAC. One thread
 * computes a baby result in registers, applies every giant-group diagonal
 * that consumes it, and writes only the final group accumulators.
 */
void launch_double_hoist_fused_baby_keyswitch_plain_accumulate(
    GpuWord *group_q,
    GpuWord *group_p,
    const GpuWord *digits_q,
    const GpuWord *digits_p,
    const GpuWord *source_q0,
    const GpuWord *source_q1,
    const std::uint32_t *host_galois_elts,
    const std::uint32_t *host_key_indices,
    const std::uint32_t *host_term_indices,
    const GpuWord *const *key_q0_ptrs,
    const GpuWord *const *key_p0_ptrs,
    const GpuWord *const *key_q1_ptrs,
    const GpuWord *const *key_p1_ptrs,
    const GpuWord *const *diagonal_q_ptrs,
    const GpuWord *const *diagonal_p_ptrs,
    const std::uint32_t *diagonal_periods,
    std::size_t group_count,
    std::size_t term_count,
    std::size_t tile_count,
    std::size_t dnum,
    std::size_t storage_dnum,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool compressed_plaintexts,
    bool initialize_accumulators);

/**
 * Compute InversePreRotated HYBRID key-multiplies (and optional lifted c0)
 * for two ciphertexts sharing one Galois automorphism in one key read pass.
 */
void launch_double_hoist_pre_rotated_keymul_pair(
    GpuWord *destination_a_q0,
    GpuWord *destination_a_p0,
    GpuWord *destination_a_q1,
    GpuWord *destination_a_p1,
    GpuWord *destination_b_q0,
    GpuWord *destination_b_p0,
    GpuWord *destination_b_q1,
    GpuWord *destination_b_p1,
    const GpuWord *digits_a_q,
    const GpuWord *digits_a_p,
    const GpuWord *digits_b_q,
    const GpuWord *digits_b_p,
    const GpuWord *const *key_q0_ptrs,
    const GpuWord *const *key_p0_ptrs,
    const GpuWord *const *key_q1_ptrs,
    const GpuWord *const *key_p1_ptrs,
    std::size_t dnum,
    std::uint32_t galois_elt,
    const GpuWord *lifted_c0_a_q,
    const GpuWord *lifted_c0_b_q,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

inline constexpr std::size_t kMaxDoubleHoistKeymulBatchSources = 8;

struct DoubleHoistKeymulManyBatchArgs
{
    std::uint32_t galois_elts[kMaxDoubleHoistKeymulManySteps];
    std::uint32_t key_indices[kMaxDoubleHoistKeymulManySteps];
    const GpuWord *digits_q[kMaxDoubleHoistKeymulBatchSources];
    const GpuWord *digits_p[kMaxDoubleHoistKeymulBatchSources];
    const GpuWord *lifted_c0_q[kMaxDoubleHoistKeymulBatchSources];
    GpuWord *destination_q[kMaxDoubleHoistKeymulBatchSources];
    GpuWord *destination_p[kMaxDoubleHoistKeymulBatchSources];
};

/**
 * Compute hoisted QP key-switching accumulators (with fused lifted c0 on Q)
 * for up to kMaxDoubleHoistKeymulBatchSources inputs across shared rotation steps.
 */
void launch_double_hoist_pre_rotated_keymul_many_batch(
    const DoubleHoistKeymulManyBatchArgs &args,
    std::size_t source_count,
    std::size_t step_count,
    std::size_t step_stride_words_q,
    std::size_t step_stride_words_p,
    const GpuWord *const *key_q0_ptrs,
    const GpuWord *const *key_p0_ptrs,
    const GpuWord *const *key_q1_ptrs,
    const GpuWord *const *key_p1_ptrs,
    std::size_t dnum,
    std::size_t storage_dnum,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
