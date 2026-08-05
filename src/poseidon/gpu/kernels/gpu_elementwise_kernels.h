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
 */
void launch_multiply_scalar_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    GpuWord scalar,
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
    bool accumulate,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

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

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
