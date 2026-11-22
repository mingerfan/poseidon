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
 * @brief Launch modular multiplication by one host scalar for one aligned shard.
 *
 * Used by bootstrap scale matching: destination = source * scalar mod q_i.
 * The uint64 scalar is reduced before multiplying the 32-bit coefficient.
 */
void launch_multiply_scalar_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    std::uint64_t scalar,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Launch CKKS bootstrap ModRaise base conversion for one shard.
 *
 * Source is a q-only coefficient-domain prefix Q_l. Destination is the full
 * q-only coefficient-domain Q_L. Existing source limbs are copied and missing
 * suffix limbs are produced by the BaseConverter-compatible fast conversion.
 */
void launch_bootstrap_modraise_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &source_parameter_shard,
    const GpuParameterShard &target_parameter_shard,
    std::size_t source_q_count,
    std::size_t target_q_count,
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

/**
 * @brief Accumulate one plaintext product into both ciphertext components.
 *
 * destination[c] += ciphertext[c] * plaintext for c in {0,1}. A 2-D grid
 * keeps coefficient accesses contiguous while removing one component launch.
 */
void launch_multiply_plain_accumulate_two_components(
    const GpuPolyShardView &destination0,
    const GpuPolyShardView &destination1,
    const GpuConstPolyShardView &ciphertext0,
    const GpuConstPolyShardView &ciphertext1,
    const GpuConstPolyShardView &plaintext,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Fused CAccum-style leaf kernel for up to four plaintext products.
 *
 * destination[c] = sum_j ciphertext_j[c] * plaintext_j when accumulate=false;
 * destination[c] += sum_j ciphertext_j[c] * plaintext_j when accumulate=true.
 * Only c0/c1 are processed. Higher ciphertext components, if any, are left
 * untouched by the caller.
 */
void launch_multiply_plain_caccumulate_two_components_4(
    const GpuPolyShardView &destination0,
    const GpuPolyShardView &destination1,
    const GpuConstPolyShardView *ciphertexts0,
    const GpuConstPolyShardView *ciphertexts1,
    const GpuConstPolyShardView *plaintexts,
    std::size_t term_count,
    const GpuConstPolyShardView *constant_plaintext,
    bool accumulate,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/** Exact modular 2*source - plaintext or 2*source - correction*plaintext. */
void launch_double_sub_plain_two_components(
    const GpuPolyShardView &destination0,
    const GpuPolyShardView &destination1,
    const GpuConstPolyShardView &source0,
    const GpuConstPolyShardView &source1,
    const GpuConstPolyShardView &plaintext,
    const GpuConstPolyShardView *correction0,
    const GpuConstPolyShardView *correction1,
    const GpuParameterShard &parameters);

/**
 * @brief Compute all three output components of a size-2 ciphertext product.
 *
 * Two launches preserve low per-thread register pressure: one 2-D grid writes
 * d0/d2 and one grid writes d1 without an intermediate read-modify-write.
 */
void launch_multiply_two_component_ciphertexts(
    const GpuPolyShardView &destination_shard0,
    const GpuPolyShardView &destination_shard1,
    const GpuPolyShardView &destination_shard2,
    const GpuConstPolyShardView &left_shard0,
    const GpuConstPolyShardView &left_shard1,
    const GpuConstPolyShardView &right_shard0,
    const GpuConstPolyShardView &right_shard1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Specialized size-2 ciphertext square.
 *
 * Uses symmetry to write d0=a0^2, d1=2*a0*a1, d2=a1^2 in one kernel.
 */
void launch_square_two_component_ciphertext(
    const GpuPolyShardView &destination_shard0,
    const GpuPolyShardView &destination_shard1,
    const GpuPolyShardView &destination_shard2,
    const GpuConstPolyShardView &source_shard0,
    const GpuConstPolyShardView &source_shard1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/** Expand compact 512-slot per-limb periodic NTT residues on GPU (degree=65536). */
void launch_expand_periodic_plaintext(
    GpuWord *destination,
    const GpuWord *compact_ntt,
    std::size_t q_count,
    std::size_t degree,
    int device_id);

inline constexpr std::size_t kMaxElementwisePairPtrs = 24;
inline constexpr std::size_t kMaxElementwiseCiphertextBatch = 8;

struct MultiplyPlainPtrsArgs
{
    GpuWord *dst[kMaxElementwisePairPtrs];
    const GpuWord *src[kMaxElementwisePairPtrs];
};

struct BinaryPolyPtrsArgs
{
    GpuWord *dst[kMaxElementwisePairPtrs];
    const GpuWord *lhs[kMaxElementwisePairPtrs];
    const GpuWord *rhs[kMaxElementwisePairPtrs];
};

struct MultiplyTwoComponentsBatchArgs
{
    GpuWord *dst0[kMaxElementwiseCiphertextBatch];
    GpuWord *dst1[kMaxElementwiseCiphertextBatch];
    GpuWord *dst2[kMaxElementwiseCiphertextBatch];
    const GpuWord *lhs0[kMaxElementwiseCiphertextBatch];
    const GpuWord *lhs1[kMaxElementwiseCiphertextBatch];
    const GpuWord *rhs0[kMaxElementwiseCiphertextBatch];
    const GpuWord *rhs1[kMaxElementwiseCiphertextBatch];
};

enum class DoublePlainPairOp
{
    SubCorr = 0,
    SubPlainC0 = 1,
    AddPlainC0 = 2
};

/**
 * @brief Multiply up to 8 ciphertext component shards by one shared plaintext shard.
 */
void launch_multiply_plain_ptrs(
    const MultiplyPlainPtrsArgs &args,
    std::size_t count,
    const GpuConstPolyShardView &plain_shard,
    const GpuPolyShardView &reference_dst_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Add or subtract up to 8 ciphertext component shard pairs in one launch.
 */
void launch_add_or_sub_ptrs(
    const BinaryPolyPtrsArgs &args,
    std::size_t count,
    bool subtract,
    const GpuPolyShardView &reference_dst_shard,
    const GpuParameterShard &parameter_shard);

/**
 * @brief Fused multiply (or square) for up to kMaxElementwiseCiphertextBatch two-component ciphertexts.
 */
void launch_multiply_batch_two_components(
    const MultiplyTwoComponentsBatchArgs &args,
    std::size_t count,
    bool is_square,
    const GpuPolyShardView &reference_dst_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Fused multiply (or square) for 1 or 2 two-component ciphertexts.
 */
void launch_multiply_pair_two_components(
    const GpuPolyShardView &dst_a0,
    const GpuPolyShardView &dst_a1,
    const GpuPolyShardView &dst_a2,
    const GpuConstPolyShardView &lhs_a0,
    const GpuConstPolyShardView &lhs_a1,
    const GpuConstPolyShardView &rhs_a0,
    const GpuConstPolyShardView &rhs_a1,
    const GpuPolyShardView *dst_b0,
    const GpuPolyShardView *dst_b1,
    const GpuPolyShardView *dst_b2,
    const GpuConstPolyShardView *lhs_b0,
    const GpuConstPolyShardView *lhs_b1,
    const GpuConstPolyShardView *rhs_b0,
    const GpuConstPolyShardView *rhs_b1,
    bool is_square,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Fused 2*src +/- plain (or plain * corr) across one or two lanes.
 */
void launch_double_sub_or_add_plain_pair(
    DoublePlainPairOp op,
    const GpuPolyShardView &dst_a0,
    const GpuPolyShardView &dst_a1,
    const GpuConstPolyShardView &src_a0,
    const GpuConstPolyShardView &src_a1,
    const GpuPolyShardView *dst_b0,
    const GpuPolyShardView *dst_b1,
    const GpuConstPolyShardView *src_b0,
    const GpuConstPolyShardView *src_b1,
    const GpuConstPolyShardView &plain_shard,
    const GpuConstPolyShardView *corr_a0,
    const GpuConstPolyShardView *corr_a1,
    const GpuConstPolyShardView *corr_b0,
    const GpuConstPolyShardView *corr_b1,
    const GpuParameterShard &parameter_shard);

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
