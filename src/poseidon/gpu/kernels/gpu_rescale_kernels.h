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
 * @brief Build the q_last correction terms used by CKKS rescale.
 *
 * Input q_last_coeff_shard must contain one full coefficient-domain q_last
 * limb, typically produced by INTT on the last q limb. The destination
 * correction_shard receives one correction limb for every remaining q_i:
 *
 * correction_i = ((q_last_coeff + floor(q_last / 2)) mod q_last mod q_i
 *                 - floor(q_last / 2) mod q_i) mod q_i.
 *
 * The correction is written in coefficient form. Call forward NTT on the
 * correction shard before launch_apply_q_last_rescale_correction_poly_shard.
 */
void launch_build_q_last_rescale_correction_poly_shard(
    const GpuPolyShardView &correction_shard,
    const GpuConstPolyShardView &q_last_coeff_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Apply the q_last rescale correction to NTT-form source limbs.
 *
 * source_shard and correction_ntt_shard must cover the same remaining q_i
 * limb range in NTT form. The destination receives:
 *
 * destination_i = (source_i - correction_i) * q_last^{-1} mod q_i.
 */
void launch_apply_q_last_rescale_correction_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuConstPolyShardView &correction_ntt_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Reconstruct centered residues modulo q_last_minus_one*q_last.
 *
 * dropped_coefficients stores two coefficient-domain limbs per ciphertext
 * component in [component][second-last,last][coefficient] order. The output
 * stores the canonical residue in [0,M); values above floor(M/2) represent
 * negative centered remainders.
 */
void launch_reconstruct_q_last_two_centered_remainders(
    GpuWide *centered_remainders,
    const GpuWord *dropped_coefficients,
    std::size_t component_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Convert centered M-residues to every retained q_i in one batch.
 *
 * correction is packed as [component][retained q limb][coefficient].
 */
void launch_build_q_last_two_rescale_correction_batch(
    GpuWord *correction,
    const GpuWide *centered_remainders,
    std::size_t component_count,
    std::size_t destination_q_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Reconstruct two dropped limbs and build retained-q corrections.
 *
 * This is equivalent to launch_reconstruct_q_last_two_centered_remainders
 * followed by launch_build_q_last_two_rescale_correction_batch, but avoids
 * the intermediate centered-remainder global buffer and one kernel launch.
 */
void launch_build_q_last_two_rescale_correction_batch_fused(
    GpuWord *correction,
    const GpuWord *dropped_coefficients,
    std::size_t component_count,
    std::size_t destination_q_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Apply one two-prime rescale to two ciphertext components.
 *
 * This batches the final (source-correction)*M^{-1} operation for c0 and c1.
 */
void launch_apply_q_last_two_rescale_correction_batch2(
    const GpuPolyShardView &destination0,
    const GpuPolyShardView &destination1,
    const GpuConstPolyShardView &source0,
    const GpuConstPolyShardView &source1,
    const GpuWord *correction_ntt,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Convert an arbitrary dropped q suffix to exact mixed-radix digits.
 *
 * dropped_coefficients is both input and output and is packed as
 * [component][dropped limb][coefficient]. The conversion is exact and avoids
 * reconstructing the (potentially wider than 64-bit) dropped-base integer.
 */
void launch_convert_rescale_dropped_to_mixed_radix(
    GpuWord *dropped_coefficients,
    std::size_t component_count,
    std::size_t source_q_count,
    std::size_t rescale_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief BConv exact centered mixed-radix remainders to retained q limbs.
 */
void launch_build_rescale_many_correction_batch(
    GpuWord *correction,
    const GpuWord *mixed_radix_digits,
    std::size_t component_count,
    std::size_t source_q_count,
    std::size_t rescale_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

/**
 * @brief Apply (source-correction)*prod(Q_drop)^{-1} to retained limbs.
 */
void launch_apply_rescale_many_correction_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuConstPolyShardView &correction_ntt_shard,
    std::size_t source_q_count,
    std::size_t rescale_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree);

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
