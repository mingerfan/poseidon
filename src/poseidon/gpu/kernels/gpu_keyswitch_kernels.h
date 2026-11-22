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

void launch_hybrid_modup_decomposition_pair(
    GpuWord *modup_q0,
    GpuWord *modup_p0,
    const GpuWord *c2_coeff0,
    const GpuWord *c2_ntt0,
    GpuWord *modup_q1,
    GpuWord *modup_p1,
    const GpuWord *c2_coeff1,
    const GpuWord *c2_ntt1,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

inline constexpr std::size_t kMaxHybridModUpBatch = 32;

struct CopyDecompQBatchArgs
{
    GpuWord *digit0_q[kMaxHybridModUpBatch];
    const GpuWord *source_q[kMaxHybridModUpBatch];
};

struct HybridModUpBatchArgs
{
    GpuWord *modup_q[kMaxHybridModUpBatch];
    GpuWord *modup_p[kMaxHybridModUpBatch];
    const GpuWord *c2_coeff[kMaxHybridModUpBatch];
    const GpuWord *c2_ntt[kMaxHybridModUpBatch];
};

/**
 * Copy all Q decomposition intervals from a batch of NTT source polynomials
 * into each hoisted decomposition's per-digit Q buffers in a single launch.
 */
void launch_copy_decomp_q_limbs_batch(
    const CopyDecompQBatchArgs &args,
    std::size_t batch_count,
    std::size_t q_count,
    std::size_t p_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * Execute lazy HYBRID ModUp decomposition across a batch of ciphertexts that
 * share the same decomposition index and modulus level.
 */
void launch_hybrid_modup_decomposition_batch(
    const HybridModUpBatchArgs &batch_args,
    std::size_t batch_count,
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

/**
 * Materialize only the source digit after multiplication by the punctured-Q
 * inverse. The compact [decomp_limb_count, degree] buffer is consumed by the
 * split P=9 ModUp/Four-step phase1 path and is substantially smaller than a
 * full active-QP ModUp result.
 */
void launch_hybrid_preweight_modup_source_p9(
    GpuWord *weighted_source,
    const GpuWord *c2_coeff,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * Forward NTT only the Q limbs outside the decomposition block and all P
 * limbs. Q limbs inside the block are already copied from the NTT source.
 */
void launch_hybrid_forward_ntt_qp_active(
    GpuWord *modup_q,
    GpuWord *modup_p,
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
    const GpuWord *key_q0,
    const GpuWord *key_p0,
    const GpuWord *key_q1,
    const GpuWord *key_p1,
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
    const GpuWord *key_q0,
    const GpuWord *key_p0,
    const GpuWord *key_q1,
    const GpuWord *key_p1,
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
    std::size_t degree,
    bool source_preweighted = false);

void launch_hybrid_preweight_p_two_components(
    GpuWord *accum_p0,
    GpuWord *accum_p1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool source_preweighted = false);

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

void launch_hybrid_apply_moddown_ntt_out_of_place(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    const GpuWord *source_q0,
    const GpuWord *source_q1,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_apply_moddown_ntt_out_of_place_batch(
    GpuWord *destination_q,
    const GpuWord *source_q,
    const GpuWord *converted_q,
    std::size_t batch_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * P->Q basis conversion of both components into coefficient form (lazy
 * 96-bit accumulation, any P); the caller runs the Q forward NTT.
 */
void launch_hybrid_convert_p_to_q_coeff_lazy(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    const GpuWord *p_coeff0,
    const GpuWord *p_coeff1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * Contiguous batched P->Q basis conversion into coefficient form for
 * `batch_count` ciphertexts ([batch][2][q_count|p_count][degree]).
 */
void launch_hybrid_convert_p_to_q_coeff_lazy_batch(
    GpuWord *converted_q,
    const GpuWord *p_coeff,
    std::size_t batch_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/** Most components launch_hybrid_apply_moddown_ntt_selected accepts. */
inline constexpr std::size_t kMaxSelectedModDownComponents = 64;

/**
 * ModDown apply (source - converted) * P^{-1} for scattered components:
 * converted slot k is component component_indices[k] of source/destination
 * (consecutive Q components of hybrid_base_q_count * degree words).
 */
void launch_hybrid_apply_moddown_ntt_selected(
    GpuWord *destination_q,
    const GpuWord *source_q,
    const GpuWord *converted_q,
    const std::uint32_t *component_indices,
    std::size_t component_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

void launch_hybrid_apply_moddown_ntt_from_q_groups(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    const GpuWord *group_q,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    std::size_t group_count,
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
    const GpuConstPolyShardView *add_source_shard0 = nullptr,
    const GpuConstPolyShardView *add_source_shard1 = nullptr);

void launch_hybrid_apply_moddown_ntt_add_back_rescale_x2(
    const GpuPolyShardView &destination_shard0,
    const GpuPolyShardView &destination_shard1,
    const GpuConstPolyShardView &correction_ntt_shard0,
    const GpuConstPolyShardView &correction_ntt_shard1,
    const GpuWord *accum_q0,
    const GpuWord *accum_q1,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    const GpuConstPolyShardView *add_source_shard0 = nullptr,
    const GpuConstPolyShardView *add_source_shard1 = nullptr);

void launch_apply_galois_ntt_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    std::uint32_t galois_elt,
    std::size_t degree);

inline constexpr std::size_t kMaxRelinBatchSources = 8;
inline constexpr std::size_t kMaxRelinDnum = 8;

struct RelinKeymulBatchArgs
{
    const GpuWord *digits_q[kMaxRelinBatchSources];
    const GpuWord *digits_p[kMaxRelinBatchSources];
    const GpuWord *c0_q[kMaxRelinBatchSources];
    const GpuWord *c1_q[kMaxRelinBatchSources];
    const GpuWord *key_q0[kMaxRelinDnum];
    const GpuWord *key_p0[kMaxRelinDnum];
    const GpuWord *key_q1[kMaxRelinDnum];
    const GpuWord *key_p1[kMaxRelinDnum];
};

void launch_relin_keymul_batch(
    GpuWord *destination_q,
    GpuWord *destination_p,
    const RelinKeymulBatchArgs &args,
    std::size_t source_count,
    std::size_t dnum,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
