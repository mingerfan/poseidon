#include "poseidon/gpu/kernels/gpu_keyswitch_kernels.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace poseidon
{
namespace gpu
{
namespace kernel
{
namespace
{

__device__ __forceinline__ GpuWord barrett_reduce_u64_u32(
    GpuWide value,
    GpuWord modulus,
    GpuWide barrett_ratio)
{
    const GpuWide quotient = __umul64hi(value, barrett_ratio);
    GpuWide reduced =
        value - quotient * static_cast<GpuWide>(modulus);

    if (reduced >= modulus)
    {
        reduced -= modulus;
    }
    if (reduced >= modulus)
    {
        reduced -= modulus;
    }

    return static_cast<GpuWord>(reduced);
}

__device__ __forceinline__ GpuWord add_mod(
    GpuWord left,
    GpuWord right,
    GpuWord modulus)
{
    const GpuWide sum =
        static_cast<GpuWide>(left) + static_cast<GpuWide>(right);
    return static_cast<GpuWord>(sum >= modulus ? sum - modulus : sum);
}

__device__ __forceinline__ GpuWord sub_mod(
    GpuWord left,
    GpuWord right,
    GpuWord modulus)
{
    return left >= right
        ? static_cast<GpuWord>(left - right)
        : static_cast<GpuWord>(
              static_cast<GpuWide>(left) + modulus - right);
}

__device__ __forceinline__ GpuWord mul_mod(
    GpuWord left,
    GpuWord right,
    GpuWord modulus,
    GpuWide barrett_ratio)
{
    return barrett_reduce_u64_u32(
        static_cast<GpuWide>(left) * static_cast<GpuWide>(right),
        modulus,
        barrett_ratio);
}

__device__ __forceinline__ GpuWord reduce_to_modulus(
    GpuWord value,
    GpuWord modulus,
    GpuWide barrett_ratio)
{
    return value < modulus
        ? value
        : barrett_reduce_u64_u32(value, modulus, barrett_ratio);
}

__device__ __forceinline__ std::uint32_t reverse_bits_limited(
    std::uint32_t value,
    unsigned int bit_count)
{
    return __brev(value) >> (32U - bit_count);
}

__device__ __forceinline__ GpuWord hybrid_modup_contribution(
    const GpuWord *c2_coeff,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *conversion_matrix,
    const GpuWord *qi_inv_punctured,
    std::size_t matrix_row_offset,
    std::size_t inv_offset,
    std::size_t decomp_limb_begin,
    std::size_t col,
    std::size_t coeff,
    std::size_t degree,
    GpuWord target_modulus,
    GpuWide target_barrett)
{
    const std::size_t source_q_limb = decomp_limb_begin + col;
    const GpuWord value = c2_coeff[source_q_limb * degree + coeff];
    const GpuWord source_modulus = rns_primes[source_q_limb];
    const GpuWide source_barrett = rns_modulus_constants[source_q_limb];
    const GpuWord inv_punctured = qi_inv_punctured[inv_offset + col];
    const GpuWord weighted = inv_punctured == 1
        ? value
        : mul_mod(value, inv_punctured, source_modulus, source_barrett);
    const GpuWord matrix_value = conversion_matrix[matrix_row_offset + col];
    return mul_mod(weighted, matrix_value, target_modulus, target_barrett);
}

__device__ __forceinline__ void hybrid_convert_p_to_q_contribution(
    GpuWord &sum0,
    GpuWord &sum1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *moddown_p_to_q_matrix,
    const GpuWord *p_inv_punctured,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t q_limb,
    std::size_t p_limb,
    std::size_t coeff,
    std::size_t degree,
    GpuWord q_modulus,
    GpuWide q_barrett)
{
    const std::size_t p_table_limb = base_q_size + p_limb;
    const GpuWord p_modulus = rns_primes[p_table_limb];
    const GpuWide p_barrett = rns_modulus_constants[p_table_limb];
    const GpuWord matrix_value =
        moddown_p_to_q_matrix[q_limb * base_p_size + p_limb];
    const GpuWord inv_punctured = p_inv_punctured[p_limb];

    const GpuWord p_value0 = accum_p0[p_limb * degree + coeff];
    const GpuWord weighted0 = inv_punctured == 1
        ? p_value0
        : mul_mod(p_value0, inv_punctured, p_modulus, p_barrett);
    sum0 = add_mod(
        sum0,
        mul_mod(weighted0, matrix_value, q_modulus, q_barrett),
        q_modulus);

    const GpuWord p_value1 = accum_p1[p_limb * degree + coeff];
    const GpuWord weighted1 = inv_punctured == 1
        ? p_value1
        : mul_mod(p_value1, inv_punctured, p_modulus, p_barrett);
    sum1 = add_mod(
        sum1,
        mul_mod(weighted1, matrix_value, q_modulus, q_barrett),
        q_modulus);
}

template <bool SourcePreweighted>
__device__ __forceinline__ void hybrid_convert_p_to_q_pair_contribution(
    GpuWord &left0,
    GpuWord &right0,
    GpuWord &left1,
    GpuWord &right1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *moddown_p_to_q_matrix,
    const GpuWord *p_inv_punctured,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t q_limb,
    std::size_t p_limb,
    std::size_t left_coeff,
    std::size_t right_coeff,
    std::size_t degree,
    GpuWord q_modulus,
    GpuWide q_barrett)
{
    const std::size_t p_table_limb = base_q_size + p_limb;
    const std::size_t p_base = p_limb * degree;
    const GpuWord matrix_value =
        moddown_p_to_q_matrix[q_limb * base_p_size + p_limb];

    const GpuWord p0_left = accum_p0[p_base + left_coeff];
    GpuWord p0_left_weighted = p0_left;
    GpuWord p0_right_weighted = accum_p0[p_base + right_coeff];
    GpuWord p1_left_weighted = accum_p1[p_base + left_coeff];
    GpuWord p1_right_weighted = accum_p1[p_base + right_coeff];
    if constexpr (!SourcePreweighted)
    {
        const GpuWord p_modulus = rns_primes[p_table_limb];
        const GpuWide p_barrett = rns_modulus_constants[p_table_limb];
        const GpuWord inv_punctured = p_inv_punctured[p_limb];
        if (inv_punctured != 1)
        {
            p0_left_weighted = mul_mod(
                p0_left_weighted, inv_punctured, p_modulus, p_barrett);
            p0_right_weighted = mul_mod(
                p0_right_weighted, inv_punctured, p_modulus, p_barrett);
            p1_left_weighted = mul_mod(
                p1_left_weighted, inv_punctured, p_modulus, p_barrett);
            p1_right_weighted = mul_mod(
                p1_right_weighted, inv_punctured, p_modulus, p_barrett);
        }
    }
    left0 = add_mod(
        left0,
        mul_mod(p0_left_weighted, matrix_value, q_modulus, q_barrett),
        q_modulus);

    right0 = add_mod(
        right0,
        mul_mod(p0_right_weighted, matrix_value, q_modulus, q_barrett),
        q_modulus);

    left1 = add_mod(
        left1,
        mul_mod(p1_left_weighted, matrix_value, q_modulus, q_barrett),
        q_modulus);

    right1 = add_mod(
        right1,
        mul_mod(p1_right_weighted, matrix_value, q_modulus, q_barrett),
        q_modulus);
}

template <int FixedPCount, bool SourcePreweighted>
__device__ __forceinline__ void hybrid_convert_p_to_q_pair(
    GpuWord &left0,
    GpuWord &right0,
    GpuWord &left1,
    GpuWord &right1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *moddown_p_to_q_matrix,
    const GpuWord *p_inv_punctured,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t q_limb,
    std::size_t left_coeff,
    std::size_t right_coeff,
    std::size_t degree,
    GpuWord q_modulus,
    GpuWide q_barrett)
{
    left0 = 0;
    right0 = 0;
    left1 = 0;
    right1 = 0;

    if constexpr (FixedPCount > 0)
    {
#pragma unroll
        for (std::size_t p_limb = 0; p_limb < FixedPCount; ++p_limb)
        {
            hybrid_convert_p_to_q_pair_contribution<SourcePreweighted>(
                left0,
                right0,
                left1,
                right1,
                accum_p0,
                accum_p1,
                rns_primes,
                rns_modulus_constants,
                moddown_p_to_q_matrix,
                p_inv_punctured,
                base_q_size,
                base_p_size,
                q_limb,
                p_limb,
                left_coeff,
                right_coeff,
                degree,
                q_modulus,
                q_barrett);
        }
    }
    else
    {
        for (std::size_t p_limb = 0; p_limb < base_p_size; ++p_limb)
        {
            hybrid_convert_p_to_q_pair_contribution<SourcePreweighted>(
                left0,
                right0,
                left1,
                right1,
                accum_p0,
                accum_p1,
                rns_primes,
                rns_modulus_constants,
                moddown_p_to_q_matrix,
                p_inv_punctured,
                base_q_size,
                base_p_size,
                q_limb,
                p_limb,
                left_coeff,
                right_coeff,
                degree,
                q_modulus,
                q_barrett);
        }
    }
}

__global__ void apply_galois_ntt_poly_shard_kernel(
    GpuWord *destination,
    const GpuWord *source,
    std::uint32_t galois_elt,
    std::size_t limb_count,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * degree;
    if (tid >= total)
    {
        return;
    }

    const std::uint32_t degree_u32 = static_cast<std::uint32_t>(degree);
    const std::uint32_t degree_minus_one = degree_u32 - 1;
    const std::size_t limb = tid >> degree_power;
    const std::uint32_t coeff =
        static_cast<std::uint32_t>(tid & degree_minus_one);

    const std::uint32_t reversed =
        reverse_bits_limited(degree_u32 + coeff, degree_power + 1);
    const std::uint64_t index_raw =
        (static_cast<std::uint64_t>(galois_elt) *
         static_cast<std::uint64_t>(reversed)) >> 1;
    const std::uint32_t source_coeff =
        reverse_bits_limited(
            static_cast<std::uint32_t>(index_raw & degree_minus_one),
            degree_power);

    destination[limb * degree + coeff] =
        source[limb * degree + source_coeff];
}

__global__ void hybrid_modup_qp_kernel(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    const GpuWord *c2_ntt,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *q_matrix_offsets,
    const GpuWord *q_matrices,
    const GpuWord *p_matrix_offsets,
    const GpuWord *p_matrices,
    const GpuWord *qi_inv_punctured,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t q_total = base_q_size * degree;
    const std::size_t p_total = base_p_size * degree;
    const std::size_t total = q_total + p_total;
    if (tid >= total)
    {
        return;
    }

    const std::size_t inv_offset = decomp_index * base_p_size;
    const std::size_t degree_mask = degree - 1;
    const std::size_t coeff = tid & degree_mask;

    /* 整体线程分为两部分，一部分用来计算模升的模数q,另一部分用来计算模升的模数p */
    if (tid < q_total)
    {
        const std::size_t target_q_limb = tid >> degree_power;

        /* 当前 dnum 块内的 Q limb 已经是 NTT 形态，后续乘密钥时直接读 c2_ntt。 */
        if (target_q_limb >= decomp_limb_begin &&
            target_q_limb < decomp_limb_begin + decomp_limb_count)
        {
            return;
        }

        const std::size_t target_offset = tid;
        const GpuWord target_modulus = rns_primes[target_q_limb];
        const GpuWide target_barrett =
            rns_modulus_constants[target_q_limb];

        /* 当前分快只有一个模数，那就直接对应取模 */
        if (decomp_limb_count == 1)
        {
            const GpuWord value =
                c2_coeff[decomp_limb_begin * degree + coeff];
            /*如果小于模数，则返回，否则做巴雷特约减*/
            modup_q[target_offset] =
                reduce_to_modulus(value, target_modulus, target_barrett);
            return;
        }

        const std::size_t matrix_offset = q_matrix_offsets[decomp_index];
        const std::size_t matrix_row_offset =
            matrix_offset + target_q_limb * base_p_size;
        GpuWord sum = 0;
        /* 一般情况下的基转换 */
        std::size_t col = 0;
        for (; col + 1 < decomp_limb_count; col += 2)
        {
            sum = add_mod(
                sum,
                hybrid_modup_contribution(
                    c2_coeff,
                    rns_primes,
                    rns_modulus_constants,
                    q_matrices,
                    qi_inv_punctured,
                    matrix_row_offset,
                    inv_offset,
                    decomp_limb_begin,
                    col,
                    coeff,
                    degree,
                    target_modulus,
                    target_barrett),
                target_modulus);
            sum = add_mod(
                sum,
                hybrid_modup_contribution(
                    c2_coeff,
                    rns_primes,
                    rns_modulus_constants,
                    q_matrices,
                    qi_inv_punctured,
                    matrix_row_offset,
                    inv_offset,
                    decomp_limb_begin,
                    col + 1,
                    coeff,
                    degree,
                    target_modulus,
                    target_barrett),
                target_modulus);
        }
        for (; col < decomp_limb_count; ++col)
        {
            sum = add_mod(
                sum,
                hybrid_modup_contribution(
                    c2_coeff,
                    rns_primes,
                    rns_modulus_constants,
                    q_matrices,
                    qi_inv_punctured,
                    matrix_row_offset,
                    inv_offset,
                    decomp_limb_begin,
                    col,
                    coeff,
                    degree,
                    target_modulus,
                    target_barrett),
                target_modulus);
        }

        modup_q[target_offset] = sum;
        return;
    }

    /* 此处开始计算p模数模升*/
    const std::size_t local_tid = tid - q_total;
    const std::size_t target_p_limb = local_tid >> degree_power;
    const std::size_t target_offset = target_p_limb * degree + coeff;
    const std::size_t target_table_limb = base_q_size + target_p_limb;
    const GpuWord target_modulus = rns_primes[target_table_limb];
    const GpuWide target_barrett =
        rns_modulus_constants[target_table_limb];

    if (decomp_limb_count == 1)
    {
        const GpuWord value =
            c2_coeff[decomp_limb_begin * degree + coeff];
        modup_p[target_offset] =
            reduce_to_modulus(value, target_modulus, target_barrett);
        return;
    }

    const std::size_t matrix_offset = p_matrix_offsets[decomp_index];
    const std::size_t matrix_row_offset =
        matrix_offset + target_p_limb * base_p_size;
    GpuWord sum = 0;

    std::size_t col = 0;
    for (; col + 1 < decomp_limb_count; col += 2)
    {
        sum = add_mod(
            sum,
            hybrid_modup_contribution(
                c2_coeff,
                rns_primes,
                rns_modulus_constants,
                p_matrices,
                qi_inv_punctured,
                matrix_row_offset,
                inv_offset,
                decomp_limb_begin,
                col,
                coeff,
                degree,
                target_modulus,
                target_barrett),
            target_modulus);
        sum = add_mod(
            sum,
            hybrid_modup_contribution(
                c2_coeff,
                rns_primes,
                rns_modulus_constants,
                p_matrices,
                qi_inv_punctured,
                matrix_row_offset,
                inv_offset,
                decomp_limb_begin,
                col + 1,
                coeff,
                degree,
                target_modulus,
                target_barrett),
            target_modulus);
    }
    for (; col < decomp_limb_count; ++col)
    {
        sum = add_mod(
            sum,
            hybrid_modup_contribution(
                c2_coeff,
                rns_primes,
                rns_modulus_constants,
                p_matrices,
                qi_inv_punctured,
                matrix_row_offset,
                inv_offset,
                decomp_limb_begin,
                col,
                coeff,
                degree,
                target_modulus,
                target_barrett),
            target_modulus);
    }

    modup_p[target_offset] = sum;
}

__device__ __forceinline__ void hybrid_modup_pair_contribution(
    GpuWord &left_sum,
    GpuWord &right_sum,
    const GpuWord *c2_coeff,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *conversion_matrix,
    const GpuWord *qi_inv_punctured,
    std::size_t matrix_row_offset,
    std::size_t inv_offset,
    std::size_t decomp_limb_begin,
    std::size_t col,
    std::size_t left_coeff,
    std::size_t right_coeff,
    std::size_t degree,
    GpuWord target_modulus,
    GpuWide target_barrett)
{
    const std::size_t source_q_limb = decomp_limb_begin + col;
    const std::size_t source_base = source_q_limb * degree;
    const GpuWord source_modulus = rns_primes[source_q_limb];
    const GpuWide source_barrett = rns_modulus_constants[source_q_limb];
    const GpuWord inv_punctured = qi_inv_punctured[inv_offset + col];
    const GpuWord matrix_value = conversion_matrix[matrix_row_offset + col];

    const GpuWord left_value = c2_coeff[source_base + left_coeff];
    const GpuWord left_weighted = inv_punctured == 1
        ? left_value
        : mul_mod(
              left_value,
              inv_punctured,
              source_modulus,
              source_barrett);
    left_sum = add_mod(
        left_sum,
        mul_mod(
            left_weighted,
            matrix_value,
            target_modulus,
            target_barrett),
        target_modulus);

    const GpuWord right_value = c2_coeff[source_base + right_coeff];
    const GpuWord right_weighted = inv_punctured == 1
        ? right_value
        : mul_mod(
              right_value,
              inv_punctured,
              source_modulus,
              source_barrett);
    right_sum = add_mod(
        right_sum,
        mul_mod(
            right_weighted,
            matrix_value,
            target_modulus,
            target_barrett),
        target_modulus);
}

template <int FixedDecompLimbCount>
__device__ __forceinline__ void hybrid_modup_pair(
    GpuWord &left,
    GpuWord &right,
    const GpuWord *c2_coeff,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *conversion_matrix,
    const GpuWord *qi_inv_punctured,
    std::size_t matrix_row_offset,
    std::size_t inv_offset,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    std::size_t left_coeff,
    std::size_t right_coeff,
    std::size_t degree,
    GpuWord target_modulus,
    GpuWide target_barrett)
{
    if constexpr (FixedDecompLimbCount == 1)
    {
        const std::size_t source_base = decomp_limb_begin * degree;
        left = reduce_to_modulus(
            c2_coeff[source_base + left_coeff],
            target_modulus,
            target_barrett);
        right = reduce_to_modulus(
            c2_coeff[source_base + right_coeff],
            target_modulus,
            target_barrett);
        return;
    }

    left = 0;
    right = 0;
    if constexpr (FixedDecompLimbCount > 0)
    {
#pragma unroll
        for (std::size_t col = 0;
             col < FixedDecompLimbCount;
             ++col)
        {
            hybrid_modup_pair_contribution(
                left,
                right,
                c2_coeff,
                rns_primes,
                rns_modulus_constants,
                conversion_matrix,
                qi_inv_punctured,
                matrix_row_offset,
                inv_offset,
                decomp_limb_begin,
                col,
                left_coeff,
                right_coeff,
                degree,
                target_modulus,
                target_barrett);
        }
    }
    else
    {
        for (std::size_t col = 0; col < decomp_limb_count; ++col)
        {
            hybrid_modup_pair_contribution(
                left,
                right,
                c2_coeff,
                rns_primes,
                rns_modulus_constants,
                conversion_matrix,
                qi_inv_punctured,
                matrix_row_offset,
                inv_offset,
                decomp_limb_begin,
                col,
                left_coeff,
                right_coeff,
                degree,
                target_modulus,
                target_barrett);
        }
    }
}

template <int FixedDecompLimbCount>
__global__ void hybrid_modup_qp_forward_ntt_first_stage_kernel(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    const GpuWord *q_matrix_offsets,
    const GpuWord *q_matrices,
    const GpuWord *p_matrix_offsets,
    const GpuWord *p_matrices,
    const GpuWord *qi_inv_punctured,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t butterflies_per_limb = degree >> 1;
    const std::size_t active_q_count = base_q_size - decomp_limb_count;
    const std::size_t active_limb_count = active_q_count + base_p_size;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = active_limb_count * butterflies_per_limb;
    if (tid >= total)
    {
        return;
    }

    const std::size_t active_limb = tid >> (degree_power - 1);
    const std::size_t left_coeff = tid & (butterflies_per_limb - 1);
    const std::size_t right_coeff = left_coeff + butterflies_per_limb;

    GpuWord *values = modup_q;
    const GpuWord *conversion_matrix = q_matrices;
    std::size_t value_limb = active_limb;
    std::size_t table_limb = active_limb;
    std::size_t matrix_row_limb = active_limb;
    std::size_t matrix_offset = q_matrix_offsets[decomp_index];

    if (active_limb < active_q_count)
    {
        if (active_limb >= decomp_limb_begin)
        {
            value_limb = active_limb + decomp_limb_count;
            table_limb = value_limb;
            matrix_row_limb = value_limb;
        }
    }
    else
    {
        const std::size_t p_limb = active_limb - active_q_count;
        values = modup_p;
        conversion_matrix = p_matrices;
        value_limb = p_limb;
        table_limb = base_q_size + p_limb;
        matrix_row_limb = p_limb;
        matrix_offset = p_matrix_offsets[decomp_index];
    }

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett = rns_modulus_constants[table_limb];
    const std::size_t matrix_row_offset =
        matrix_offset + matrix_row_limb * base_p_size;
    const std::size_t inv_offset = decomp_index * base_p_size;

    GpuWord left;
    GpuWord right;
    hybrid_modup_pair<FixedDecompLimbCount>(
        left,
        right,
        c2_coeff,
        rns_primes,
        rns_modulus_constants,
        conversion_matrix,
        qi_inv_punctured,
        matrix_row_offset,
        inv_offset,
        decomp_limb_begin,
        decomp_limb_count,
        left_coeff,
        right_coeff,
        degree,
        modulus,
        barrett);

    const GpuWord root = roots[table_limb * degree + 1];
    const GpuWord twisted_right = mul_mod(right, root, modulus, barrett);
    const std::size_t value_base = value_limb * degree;
    values[value_base + left_coeff] = add_mod(left, twisted_right, modulus);
    values[value_base + right_coeff] = sub_mod(left, twisted_right, modulus);
}

__global__ void hybrid_modup_qp_forward_ntt_first_stage_row_tiled2_kernel(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    const GpuWord *q_matrix_offsets,
    const GpuWord *q_matrices,
    const GpuWord *p_matrix_offsets,
    const GpuWord *p_matrices,
    const GpuWord *qi_inv_punctured,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree)
{
    constexpr std::size_t kSourceLimbCount = 2;
    constexpr std::size_t kCoefficientTile = 32;
    constexpr std::size_t kTargetRows = 4;
    __shared__ GpuWord weighted_source[4][kCoefficientTile];

    const std::size_t lane = threadIdx.x;
    const std::size_t row = threadIdx.y;
    const std::size_t butterflies_per_limb = degree >> 1;
    const std::size_t coefficient_pair =
        blockIdx.x * kCoefficientTile + lane;
    const bool valid_coefficient =
        coefficient_pair < butterflies_per_limb;

    const std::size_t source_col = row >> 1;
    const std::size_t coefficient_side = row & 1;
    GpuWord weighted = 0;
    if (valid_coefficient)
    {
        const std::size_t source_q_limb =
            decomp_limb_begin + source_col;
        const std::size_t source_coeff =
            coefficient_pair + coefficient_side * butterflies_per_limb;
        const GpuWord source_modulus = rns_primes[source_q_limb];
        const GpuWide source_barrett =
            rns_modulus_constants[source_q_limb];
        const GpuWord inv_punctured =
            qi_inv_punctured[
                decomp_index * base_p_size + source_col];
        const GpuWord source_value =
            c2_coeff[source_q_limb * degree + source_coeff];
        weighted = inv_punctured == 1
            ? source_value
            : mul_mod(
                  source_value,
                  inv_punctured,
                  source_modulus,
                  source_barrett);
    }
    weighted_source[row][lane] = weighted;
    __syncthreads();

    const std::size_t active_q_count =
        base_q_size - kSourceLimbCount;
    const std::size_t active_limb_count =
        active_q_count + base_p_size;
    const std::size_t active_limb =
        blockIdx.y * kTargetRows + row;
    if (!valid_coefficient || active_limb >= active_limb_count)
    {
        return;
    }

    GpuWord *values = modup_q;
    const GpuWord *conversion_matrix = q_matrices;
    std::size_t value_limb = active_limb;
    std::size_t table_limb = active_limb;
    std::size_t matrix_row_limb = active_limb;
    std::size_t matrix_offset = q_matrix_offsets[decomp_index];

    if (active_limb < active_q_count)
    {
        if (active_limb >= decomp_limb_begin)
        {
            value_limb = active_limb + kSourceLimbCount;
            table_limb = value_limb;
            matrix_row_limb = value_limb;
        }
    }
    else
    {
        const std::size_t p_limb = active_limb - active_q_count;
        values = modup_p;
        conversion_matrix = p_matrices;
        value_limb = p_limb;
        table_limb = base_q_size + p_limb;
        matrix_row_limb = p_limb;
        matrix_offset = p_matrix_offsets[decomp_index];
    }

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett = rns_modulus_constants[table_limb];
    const std::size_t matrix_row_offset =
        matrix_offset + matrix_row_limb * base_p_size;
    const GpuWord matrix0 = conversion_matrix[matrix_row_offset];
    const GpuWord matrix1 = conversion_matrix[matrix_row_offset + 1];

    const GpuWord left = add_mod(
        mul_mod(
            weighted_source[0][lane],
            matrix0,
            modulus,
            barrett),
        mul_mod(
            weighted_source[2][lane],
            matrix1,
            modulus,
            barrett),
        modulus);
    const GpuWord right = add_mod(
        mul_mod(
            weighted_source[1][lane],
            matrix0,
            modulus,
            barrett),
        mul_mod(
            weighted_source[3][lane],
            matrix1,
            modulus,
            barrett),
        modulus);

    const GpuWord root = roots[table_limb * degree + 1];
    const GpuWord twisted_right = mul_mod(right, root, modulus, barrett);
    const std::size_t value_base = value_limb * degree;
    values[value_base + coefficient_pair] =
        add_mod(left, twisted_right, modulus);
    values[value_base + coefficient_pair + butterflies_per_limb] =
        sub_mod(left, twisted_right, modulus);
}

__global__ void hybrid_modup_qp_forward_ntt_first_stage_row_tiled8_kernel(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    const GpuWord *q_matrix_offsets,
    const GpuWord *q_matrices,
    const GpuWord *p_matrix_offsets,
    const GpuWord *p_matrices,
    const GpuWord *qi_inv_punctured,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree)
{
    constexpr std::size_t kSourceLimbCount = 2;
    constexpr std::size_t kCoefficientTile = 32;
    constexpr std::size_t kSourcePlanes = 4;
    constexpr std::size_t kTargetRows = 8;
    __shared__ GpuWord weighted_source[kSourcePlanes][kCoefficientTile];

    const std::size_t lane = threadIdx.x;
    const std::size_t row = threadIdx.y;
    const std::size_t butterflies_per_limb = degree >> 1;
    const std::size_t coefficient_pair =
        blockIdx.x * kCoefficientTile + lane;
    const bool valid_coefficient =
        coefficient_pair < butterflies_per_limb;

    if (row < kSourcePlanes)
    {
        const std::size_t source_col = row >> 1;
        const std::size_t coefficient_side = row & 1;
        GpuWord weighted = 0;
        if (valid_coefficient)
        {
            const std::size_t source_q_limb =
                decomp_limb_begin + source_col;
            const std::size_t source_coeff =
                coefficient_pair + coefficient_side * butterflies_per_limb;
            const GpuWord source_modulus = rns_primes[source_q_limb];
            const GpuWide source_barrett =
                rns_modulus_constants[source_q_limb];
            const GpuWord inv_punctured =
                qi_inv_punctured[
                    decomp_index * base_p_size + source_col];
            const GpuWord source_value =
                c2_coeff[source_q_limb * degree + source_coeff];
            weighted = inv_punctured == 1
                ? source_value
                : mul_mod(
                      source_value,
                      inv_punctured,
                      source_modulus,
                      source_barrett);
        }
        weighted_source[row][lane] = weighted;
    }
    __syncthreads();

    const std::size_t active_q_count =
        base_q_size - kSourceLimbCount;
    const std::size_t active_limb_count =
        active_q_count + base_p_size;
    const std::size_t active_limb =
        blockIdx.y * kTargetRows + row;
    if (!valid_coefficient || active_limb >= active_limb_count)
    {
        return;
    }

    GpuWord *values = modup_q;
    const GpuWord *conversion_matrix = q_matrices;
    std::size_t value_limb = active_limb;
    std::size_t table_limb = active_limb;
    std::size_t matrix_row_limb = active_limb;
    std::size_t matrix_offset = q_matrix_offsets[decomp_index];

    if (active_limb < active_q_count)
    {
        if (active_limb >= decomp_limb_begin)
        {
            value_limb = active_limb + kSourceLimbCount;
            table_limb = value_limb;
            matrix_row_limb = value_limb;
        }
    }
    else
    {
        const std::size_t p_limb = active_limb - active_q_count;
        values = modup_p;
        conversion_matrix = p_matrices;
        value_limb = p_limb;
        table_limb = base_q_size + p_limb;
        matrix_row_limb = p_limb;
        matrix_offset = p_matrix_offsets[decomp_index];
    }

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett = rns_modulus_constants[table_limb];
    const std::size_t matrix_row_offset =
        matrix_offset + matrix_row_limb * base_p_size;
    const GpuWord matrix0 = conversion_matrix[matrix_row_offset];
    const GpuWord matrix1 = conversion_matrix[matrix_row_offset + 1];

    const GpuWord left = add_mod(
        mul_mod(
            weighted_source[0][lane],
            matrix0,
            modulus,
            barrett),
        mul_mod(
            weighted_source[2][lane],
            matrix1,
            modulus,
            barrett),
        modulus);
    const GpuWord right = add_mod(
        mul_mod(
            weighted_source[1][lane],
            matrix0,
            modulus,
            barrett),
        mul_mod(
            weighted_source[3][lane],
            matrix1,
            modulus,
            barrett),
        modulus);

    const GpuWord root = roots[table_limb * degree + 1];
    const GpuWord twisted_right = mul_mod(right, root, modulus, barrett);
    const std::size_t value_base = value_limb * degree;
    values[value_base + coefficient_pair] =
        add_mod(left, twisted_right, modulus);
    values[value_base + coefficient_pair + butterflies_per_limb] =
        sub_mod(left, twisted_right, modulus);
}

__global__ void hybrid_modup_qp_row_tiled8_kernel(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *q_matrix_offsets,
    const GpuWord *q_matrices,
    const GpuWord *p_matrix_offsets,
    const GpuWord *p_matrices,
    const GpuWord *qi_inv_punctured,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree)
{
    constexpr std::size_t kSourceLimbCount = 2;
    constexpr std::size_t kCoefficientTile = 32;
    constexpr std::size_t kSourcePlanes = 4;
    constexpr std::size_t kTargetRows = 8;
    __shared__ GpuWord weighted_source[kSourcePlanes][kCoefficientTile];

    const std::size_t lane = threadIdx.x;
    const std::size_t row = threadIdx.y;
    const std::size_t coefficients_per_half = degree >> 1;
    const std::size_t coefficient_pair =
        blockIdx.x * kCoefficientTile + lane;
    const bool valid_coefficient =
        coefficient_pair < coefficients_per_half;

    if (row < kSourcePlanes)
    {
        const std::size_t source_col = row >> 1;
        const std::size_t coefficient_side = row & 1;
        GpuWord weighted = 0;
        if (valid_coefficient)
        {
            const std::size_t source_q_limb =
                decomp_limb_begin + source_col;
            const std::size_t source_coeff =
                coefficient_pair + coefficient_side * coefficients_per_half;
            const GpuWord source_modulus = rns_primes[source_q_limb];
            const GpuWide source_barrett =
                rns_modulus_constants[source_q_limb];
            const GpuWord inv_punctured =
                qi_inv_punctured[
                    decomp_index * base_p_size + source_col];
            const GpuWord source_value =
                c2_coeff[source_q_limb * degree + source_coeff];
            weighted = inv_punctured == 1
                ? source_value
                : mul_mod(
                      source_value,
                      inv_punctured,
                      source_modulus,
                      source_barrett);
        }
        weighted_source[row][lane] = weighted;
    }
    __syncthreads();

    const std::size_t active_q_count =
        base_q_size - kSourceLimbCount;
    const std::size_t active_limb_count =
        active_q_count + base_p_size;
    const std::size_t active_limb =
        blockIdx.y * kTargetRows + row;
    if (!valid_coefficient || active_limb >= active_limb_count)
    {
        return;
    }

    GpuWord *values = modup_q;
    const GpuWord *conversion_matrix = q_matrices;
    std::size_t value_limb = active_limb;
    std::size_t table_limb = active_limb;
    std::size_t matrix_row_limb = active_limb;
    std::size_t matrix_offset = q_matrix_offsets[decomp_index];

    if (active_limb < active_q_count)
    {
        if (active_limb >= decomp_limb_begin)
        {
            value_limb = active_limb + kSourceLimbCount;
            table_limb = value_limb;
            matrix_row_limb = value_limb;
        }
    }
    else
    {
        const std::size_t p_limb = active_limb - active_q_count;
        values = modup_p;
        conversion_matrix = p_matrices;
        value_limb = p_limb;
        table_limb = base_q_size + p_limb;
        matrix_row_limb = p_limb;
        matrix_offset = p_matrix_offsets[decomp_index];
    }

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett = rns_modulus_constants[table_limb];
    const std::size_t matrix_row_offset =
        matrix_offset + matrix_row_limb * base_p_size;
    const GpuWord matrix0 = conversion_matrix[matrix_row_offset];
    const GpuWord matrix1 = conversion_matrix[matrix_row_offset + 1];

    const GpuWord left = add_mod(
        mul_mod(
            weighted_source[0][lane],
            matrix0,
            modulus,
            barrett),
        mul_mod(
            weighted_source[2][lane],
            matrix1,
            modulus,
            barrett),
        modulus);
    const GpuWord right = add_mod(
        mul_mod(
            weighted_source[1][lane],
            matrix0,
            modulus,
            barrett),
        mul_mod(
            weighted_source[3][lane],
            matrix1,
            modulus,
            barrett),
        modulus);

    const std::size_t value_base = value_limb * degree;
    values[value_base + coefficient_pair] = left;
    values[value_base + coefficient_pair + coefficients_per_half] = right;
}

template <int FixedSourceLimbCount>
__global__ void hybrid_modup_qp_p9_row_tiled8_kernel(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *q_matrix_offsets,
    const GpuWord *q_matrices,
    const GpuWord *p_matrix_offsets,
    const GpuWord *p_matrices,
    const GpuWord *qi_inv_punctured,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree)
{
    static_assert(
        FixedSourceLimbCount >= 1 && FixedSourceLimbCount <= 9,
        "P=9 row-tiled ModUp supports one through nine source limbs");
    constexpr std::size_t kCoefficientTile = 32;
    constexpr std::size_t kSourcePlanes = FixedSourceLimbCount * 2;
    constexpr std::size_t kTargetRows = 8;
    constexpr std::size_t kBlockThreads =
        kCoefficientTile * kTargetRows;
    __shared__ GpuWord weighted_source[kSourcePlanes][kCoefficientTile];

    const std::size_t lane = threadIdx.x;
    const std::size_t row = threadIdx.y;
    const std::size_t linear_thread =
        row * kCoefficientTile + lane;
    const std::size_t coefficients_per_half = degree >> 1;

    for (std::size_t source_index = linear_thread;
         source_index < kSourcePlanes * kCoefficientTile;
         source_index += kBlockThreads)
    {
        const std::size_t source_plane =
            source_index / kCoefficientTile;
        const std::size_t source_lane =
            source_index % kCoefficientTile;
        const std::size_t source_col = source_plane >> 1;
        const std::size_t coefficient_side = source_plane & 1;
        const std::size_t coefficient_pair =
            blockIdx.x * kCoefficientTile + source_lane;
        GpuWord weighted = 0;
        if (coefficient_pair < coefficients_per_half)
        {
            const std::size_t source_q_limb =
                decomp_limb_begin + source_col;
            const std::size_t source_coeff = coefficient_pair +
                coefficient_side * coefficients_per_half;
            const GpuWord source_modulus = rns_primes[source_q_limb];
            const GpuWide source_barrett =
                rns_modulus_constants[source_q_limb];
            const GpuWord inv_punctured = qi_inv_punctured[
                decomp_index * base_p_size + source_col];
            weighted = c2_coeff[
                source_q_limb * degree + source_coeff];
            if constexpr (FixedSourceLimbCount > 1)
            {
                if (inv_punctured != 1)
                {
                    weighted = mul_mod(
                        weighted,
                        inv_punctured,
                        source_modulus,
                        source_barrett);
                }
            }
        }
        weighted_source[source_plane][source_lane] = weighted;
    }
    __syncthreads();

    const std::size_t coefficient_pair =
        blockIdx.x * kCoefficientTile + lane;
    const std::size_t active_q_count =
        base_q_size - FixedSourceLimbCount;
    const std::size_t active_limb_count =
        active_q_count + base_p_size;
    const std::size_t active_limb =
        blockIdx.y * kTargetRows + row;
    if (coefficient_pair >= coefficients_per_half ||
        active_limb >= active_limb_count)
    {
        return;
    }

    GpuWord *values = modup_q;
    const GpuWord *conversion_matrix = q_matrices;
    std::size_t value_limb = active_limb;
    std::size_t table_limb = active_limb;
    std::size_t matrix_row_limb = active_limb;
    std::size_t matrix_offset = q_matrix_offsets[decomp_index];
    if (active_limb < active_q_count)
    {
        if (active_limb >= decomp_limb_begin)
        {
            value_limb = active_limb + FixedSourceLimbCount;
            table_limb = value_limb;
            matrix_row_limb = value_limb;
        }
    }
    else
    {
        const std::size_t p_limb = active_limb - active_q_count;
        values = modup_p;
        conversion_matrix = p_matrices;
        value_limb = p_limb;
        table_limb = base_q_size + p_limb;
        matrix_row_limb = p_limb;
        matrix_offset = p_matrix_offsets[decomp_index];
    }

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett = rns_modulus_constants[table_limb];
    const std::size_t matrix_row_offset =
        matrix_offset + matrix_row_limb * base_p_size;
    GpuWord left = 0;
    GpuWord right = 0;
    if constexpr (FixedSourceLimbCount == 1)
    {
        left = reduce_to_modulus(
            weighted_source[0][lane],
            modulus,
            barrett);
        right = reduce_to_modulus(
            weighted_source[1][lane],
            modulus,
            barrett);
    }
    else
    {
#pragma unroll
        for (std::size_t source_col = 0;
             source_col < FixedSourceLimbCount;
             ++source_col)
        {
            const GpuWord matrix_value =
                conversion_matrix[matrix_row_offset + source_col];
            left = add_mod(
                left,
                mul_mod(
                    weighted_source[source_col * 2][lane],
                    matrix_value,
                    modulus,
                    barrett),
                modulus);
            right = add_mod(
                right,
                mul_mod(
                    weighted_source[source_col * 2 + 1][lane],
                    matrix_value,
                    modulus,
                    barrett),
                modulus);
        }
    }

    const std::size_t value_base = value_limb * degree;
    values[value_base + coefficient_pair] = left;
    values[value_base + coefficient_pair + coefficients_per_half] = right;
}

__global__ void hybrid_forward_ntt_modup_qp_stage_kernel(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    std::size_t m,
    std::size_t gap)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t butterflies_per_limb = degree >> 1;
    const std::size_t active_q_count = base_q_size - decomp_limb_count;
    const std::size_t active_limb_count = active_q_count + base_p_size;
    const std::size_t total = active_limb_count * butterflies_per_limb;
    if (tid >= total)
    {
        return;
    }

    const std::size_t active_limb = tid / butterflies_per_limb;
    const std::size_t local_butterfly = tid % butterflies_per_limb;
    const std::size_t group = local_butterfly / gap;
    const std::size_t j = local_butterfly % gap;

    GpuWord *values = modup_q;
    std::size_t value_limb = active_limb;
    std::size_t table_limb = active_limb;

    if (active_limb < active_q_count)
    {
        if (active_limb >= decomp_limb_begin)
        {
            value_limb = active_limb + decomp_limb_count;
            table_limb = value_limb;
        }
    }
    else
    {
        const std::size_t p_limb = active_limb - active_q_count;
        values = modup_p;
        value_limb = p_limb;
        table_limb = base_q_size + p_limb;
    }

    const std::size_t x_index =
        value_limb * degree + group * (gap << 1) + j;
    const std::size_t y_index = x_index + gap;

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord root = roots[table_limb * degree + m + group];

    const GpuWord u = values[x_index];
    const GpuWord v = mul_mod(values[y_index], root, modulus, barrett_ratio);

    values[x_index] = add_mod(u, v, modulus);
    values[y_index] = sub_mod(u, v, modulus);
}

template <int FusionStages>
__global__ void hybrid_forward_ntt_modup_qp_fused_stage_kernel(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    std::size_t m,
    std::size_t gap)
{
    static_assert(
        FusionStages >= 2 && FusionStages <= 3,
        "HYBRID QP NTT fusion supports 2-3 stages");

    constexpr std::size_t kLocalSize =
        static_cast<std::size_t>(1) << FusionStages;

    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t tiles_per_limb = degree >> FusionStages;
    const std::size_t active_q_count = base_q_size - decomp_limb_count;
    const std::size_t active_limb_count = active_q_count + base_p_size;
    const std::size_t total = active_limb_count * tiles_per_limb;
    if (tid >= total)
    {
        return;
    }

    const std::size_t active_limb = tid / tiles_per_limb;
    const std::size_t local_tile = tid % tiles_per_limb;

    GpuWord *values = modup_q;
    std::size_t value_limb = active_limb;
    std::size_t table_limb = active_limb;

    if (active_limb < active_q_count)
    {
        if (active_limb >= decomp_limb_begin)
        {
            value_limb = active_limb + decomp_limb_count;
            table_limb = value_limb;
        }
    }
    else
    {
        const std::size_t p_limb = active_limb - active_q_count;
        values = modup_p;
        value_limb = p_limb;
        table_limb = base_q_size + p_limb;
    }

    const std::size_t final_gap = gap >> (FusionStages - 1);
    const std::size_t outer_group = local_tile / final_gap;
    const std::size_t j = local_tile % final_gap;
    const std::size_t base_index =
        value_limb * degree + outer_group * (gap << 1) + j;

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord *limb_roots = roots + table_limb * degree;

    GpuWord local[kLocalSize];
#pragma unroll
    for (std::size_t i = 0; i < kLocalSize; ++i)
    {
        local[i] = values[base_index + i * final_gap];
    }

#pragma unroll
    for (int stage = 0; stage < FusionStages; ++stage)
    {
        const std::size_t local_stride =
            static_cast<std::size_t>(1) << (FusionStages - 1 - stage);
        const std::size_t stage_m = m << stage;
        const std::size_t stage_group_base = outer_group << stage;

#pragma unroll
        for (std::size_t block = 0; block < kLocalSize;
             block += (local_stride << 1))
        {
            const std::size_t block_group =
                block / (local_stride << 1);
            const GpuWord root =
                limb_roots[stage_m + stage_group_base + block_group];

#pragma unroll
            for (std::size_t offset = 0; offset < local_stride; ++offset)
            {
                const GpuWord u = local[block + offset];
                const GpuWord v = mul_mod(
                    local[block + offset + local_stride],
                    root,
                    modulus,
                    barrett_ratio);
                local[block + offset] = add_mod(u, v, modulus);
                local[block + offset + local_stride] =
                    sub_mod(u, v, modulus);
            }
        }
    }

#pragma unroll
    for (std::size_t i = 0; i < kLocalSize; ++i)
    {
        values[base_index + i * final_gap] = local[i];
    }
}

__device__ __forceinline__ void hybrid_accumulate_two_products(
    GpuWord *accum0,
    GpuWord *accum1,
    GpuWord value,
    GpuWord key0,
    GpuWord key1,
    GpuWord modulus,
    GpuWide barrett_ratio,
    bool overwrite_accum)
{
    const GpuWord product0 =
        mul_mod(value, key0, modulus, barrett_ratio);
    const GpuWord product1 =
        mul_mod(value, key1, modulus, barrett_ratio);

    if (overwrite_accum)
    {
        *accum0 = product0;
        *accum1 = product1;
        return;
    }

    *accum0 = add_mod(*accum0, product0, modulus);
    *accum1 = add_mod(*accum1, product1, modulus);
}

template <int FusionStages>
__global__ void
hybrid_forward_ntt_modup_qp_final_mul_accumulate_kernel(
    GpuWord *accum_q0,
    GpuWord *accum_p0,
    GpuWord *accum_q1,
    GpuWord *accum_p1,
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    const GpuWord *key_q0,
    const GpuWord *key_p0,
    const GpuWord *key_q1,
    const GpuWord *key_p1,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    std::size_t m,
    std::size_t gap,
    bool overwrite_accum)
{
    static_assert(
        FusionStages >= 1 && FusionStages <= 3,
        "HYBRID final QP NTT/IP fusion supports 1-3 stages");

    constexpr std::size_t kLocalSize =
        static_cast<std::size_t>(1) << FusionStages;

    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t tiles_per_limb = degree >> FusionStages;
    const std::size_t active_q_count = base_q_size - decomp_limb_count;
    const std::size_t active_limb_count = active_q_count + base_p_size;
    const std::size_t total = active_limb_count * tiles_per_limb;
    if (tid >= total)
    {
        return;
    }

    const std::size_t active_limb = tid / tiles_per_limb;
    const std::size_t local_tile = tid % tiles_per_limb;

    GpuWord *values = modup_q;
    GpuWord *accum0 = accum_q0;
    GpuWord *accum1 = accum_q1;
    std::size_t value_limb = active_limb;
    std::size_t table_limb = active_limb;
    const GpuWord *key0 = key_q0;
    const GpuWord *key1 = key_q1;

    if (active_limb < active_q_count)
    {
        if (active_limb >= decomp_limb_begin)
        {
            value_limb = active_limb + decomp_limb_count;
            table_limb = value_limb;
        }
    }
    else
    {
        const std::size_t p_limb = active_limb - active_q_count;
        values = modup_p;
        accum0 = accum_p0;
        accum1 = accum_p1;
        value_limb = p_limb;
        table_limb = base_q_size + p_limb;
        key0 = key_p0;
        key1 = key_p1;
    }

    const std::size_t final_gap = gap >> (FusionStages - 1);
    const std::size_t outer_group = local_tile / final_gap;
    const std::size_t j = local_tile % final_gap;
    const std::size_t base_index =
        value_limb * degree + outer_group * (gap << 1) + j;

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord *limb_roots = roots + table_limb * degree;

    GpuWord local[kLocalSize];
#pragma unroll
    for (std::size_t i = 0; i < kLocalSize; ++i)
    {
        local[i] = values[base_index + i * final_gap];
    }

#pragma unroll
    for (int stage = 0; stage < FusionStages; ++stage)
    {
        const std::size_t local_stride =
            static_cast<std::size_t>(1) << (FusionStages - 1 - stage);
        const std::size_t stage_m = m << stage;
        const std::size_t stage_group_base = outer_group << stage;

#pragma unroll
        for (std::size_t block = 0; block < kLocalSize;
             block += (local_stride << 1))
        {
            const std::size_t block_group =
                block / (local_stride << 1);
            const GpuWord root =
                limb_roots[stage_m + stage_group_base + block_group];

#pragma unroll
            for (std::size_t offset = 0; offset < local_stride; ++offset)
            {
                const GpuWord u = local[block + offset];
                const GpuWord v = mul_mod(
                    local[block + offset + local_stride],
                    root,
                    modulus,
                    barrett_ratio);
                local[block + offset] = add_mod(u, v, modulus);
                local[block + offset + local_stride] =
                    sub_mod(u, v, modulus);
            }
        }
    }

#pragma unroll
    for (std::size_t i = 0; i < kLocalSize; ++i)
    {
        const std::size_t local_offset = base_index + i * final_gap;
        hybrid_accumulate_two_products(
            accum0 + local_offset,
            accum1 + local_offset,
            local[i],
            key0[local_offset],
            key1[local_offset],
            modulus,
            barrett_ratio,
            overwrite_accum);
    }
}

template <int FusionStages>
__global__ void
hybrid_forward_ntt_modup_qp_final_mul_accumulate_fused_decomp_q_kernel(
    GpuWord *accum_q0,
    GpuWord *accum_p0,
    GpuWord *accum_q1,
    GpuWord *accum_p1,
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_ntt,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    const GpuWord *key_q0,
    const GpuWord *key_p0,
    const GpuWord *key_q1,
    const GpuWord *key_p1,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    unsigned int degree_power,
    std::size_t m,
    std::size_t gap,
    bool overwrite_accum)
{
    static_assert(
        FusionStages >= 1 && FusionStages <= 3,
        "HYBRID final QP NTT/IP fusion supports 1-3 stages");

    constexpr std::size_t kLocalSize =
        static_cast<std::size_t>(1) << FusionStages;

    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t tiles_per_limb = degree >> FusionStages;
    const std::size_t active_q_count = base_q_size - decomp_limb_count;
    const std::size_t active_limb_count = active_q_count + base_p_size;
    const std::size_t total = active_limb_count * tiles_per_limb;
    const std::size_t decomp_total = decomp_limb_count * degree;
    if (tid >= total + decomp_total)
    {
        return;
    }

    /* Append the skipped Q coefficients as scalar tasks in the same launch. */
    if (tid >= total)
    {
        const std::size_t decomp_tid = tid - total;
        const std::size_t offset =
            decomp_limb_begin * degree + decomp_tid;
        const std::size_t q_limb = offset >> degree_power;
        const GpuWord decomp_modulus = rns_primes[q_limb];
        const GpuWide decomp_barrett = rns_modulus_constants[q_limb];

        hybrid_accumulate_two_products(
            accum_q0 + offset,
            accum_q1 + offset,
            c2_ntt[offset],
            key_q0[offset],
            key_q1[offset],
            decomp_modulus,
            decomp_barrett,
            overwrite_accum);
        return;
    }

    {
        const std::size_t active_limb = tid / tiles_per_limb;
        const std::size_t local_tile = tid % tiles_per_limb;

        GpuWord *values = modup_q;
        GpuWord *accum0 = accum_q0;
        GpuWord *accum1 = accum_q1;
        std::size_t value_limb = active_limb;
        std::size_t table_limb = active_limb;
        const GpuWord *key0 = key_q0;
        const GpuWord *key1 = key_q1;

        if (active_limb < active_q_count)
        {
            if (active_limb >= decomp_limb_begin)
            {
                value_limb = active_limb + decomp_limb_count;
                table_limb = value_limb;
            }
        }
        else
        {
            const std::size_t p_limb = active_limb - active_q_count;
            values = modup_p;
            accum0 = accum_p0;
            accum1 = accum_p1;
            value_limb = p_limb;
            table_limb = base_q_size + p_limb;
            key0 = key_p0;
            key1 = key_p1;
        }

        const std::size_t final_gap = gap >> (FusionStages - 1);
        const std::size_t outer_group = local_tile / final_gap;
        const std::size_t j = local_tile % final_gap;
        const std::size_t base_index =
            value_limb * degree + outer_group * (gap << 1) + j;

        const GpuWord modulus = rns_primes[table_limb];
        const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
        const GpuWord *limb_roots = roots + table_limb * degree;

        GpuWord local[kLocalSize];
#pragma unroll
        for (std::size_t i = 0; i < kLocalSize; ++i)
        {
            local[i] = values[base_index + i * final_gap];
        }

#pragma unroll
        for (int stage = 0; stage < FusionStages; ++stage)
        {
            const std::size_t local_stride =
                static_cast<std::size_t>(1) << (FusionStages - 1 - stage);
            const std::size_t stage_m = m << stage;
            const std::size_t stage_group_base = outer_group << stage;

#pragma unroll
            for (std::size_t block = 0; block < kLocalSize;
                 block += (local_stride << 1))
            {
                const std::size_t block_group =
                    block / (local_stride << 1);
                const GpuWord root =
                    limb_roots[stage_m + stage_group_base + block_group];

#pragma unroll
                for (std::size_t offset = 0; offset < local_stride; ++offset)
                {
                    const GpuWord u = local[block + offset];
                    const GpuWord v = mul_mod(
                        local[block + offset + local_stride],
                        root,
                        modulus,
                        barrett_ratio);
                    local[block + offset] = add_mod(u, v, modulus);
                    local[block + offset + local_stride] =
                        sub_mod(u, v, modulus);
                }
            }
        }

#pragma unroll
        for (std::size_t i = 0; i < kLocalSize; ++i)
        {
            const std::size_t local_offset = base_index + i * final_gap;
            hybrid_accumulate_two_products(
                accum0 + local_offset,
                accum1 + local_offset,
                local[i],
                key0[local_offset],
                key1[local_offset],
                modulus,
                barrett_ratio,
                overwrite_accum);
        }
    }

}

__global__ void hybrid_decomp_q_multiply_accumulate_two_components_kernel(
    GpuWord *accum_q0,
    GpuWord *accum_q1,
    const GpuWord *c2_ntt,
    const GpuWord *key_qp0,
    const GpuWord *key_qp1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    std::size_t degree,
    unsigned int degree_power,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    bool overwrite_accum)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = decomp_limb_count * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid >> degree_power;
    const std::size_t coeff = tid & (degree - 1);
    const std::size_t q_limb = decomp_limb_begin + local_limb;
    const std::size_t offset = q_limb * degree + coeff;
    const GpuWord modulus = rns_primes[q_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[q_limb];

    hybrid_accumulate_two_products(
        accum_q0 + offset,
        accum_q1 + offset,
        c2_ntt[offset],
        key_qp0[offset],
        key_qp1[offset],
        modulus,
        barrett_ratio,
        overwrite_accum);
}

#if 0
/* Retired all-dnum PAccum experiments. */
__global__ void hybrid_paccum_all_dnum_two_components_kernel(
    GpuWord *accum_q0,
    GpuWord *accum_p0,
    GpuWord *accum_q1,
    GpuWord *accum_p1,
    const GpuWord *all_modup_q,
    const GpuWord *all_modup_p,
    const GpuWord *c2_ntt,
    const GpuWord *const *key_qp0_by_dnum,
    const GpuWord *const *key_qp1_by_dnum,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    std::size_t decomp_count,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t q_word_count = base_q_size * degree;
    const std::size_t p_word_count = base_p_size * degree;
    const std::size_t total = q_word_count + p_word_count;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total)
    {
        return;
    }

    const bool p_limb = tid >= q_word_count;
    const std::size_t local_offset = p_limb ? tid - q_word_count : tid;
    const std::size_t local_limb = local_offset >> degree_power;
    const std::size_t table_limb = p_limb
        ? base_q_size + local_limb
        : local_limb;
    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];

    GpuWord acc0 = 0;
    GpuWord acc1 = 0;

    for (std::size_t decomp_index = 0;
         decomp_index < decomp_count;
         ++decomp_index)
    {
        const GpuWord *key_qp0 = key_qp0_by_dnum[decomp_index];
        const GpuWord *key_qp1 = key_qp1_by_dnum[decomp_index];
        GpuWord value = 0;
        std::size_t key_offset = 0;

        if (p_limb)
        {
            value =
                all_modup_p[decomp_index * p_word_count + local_offset];
            key_offset = q_word_count + local_offset;
        }
        else
        {
            const std::size_t decomp_limb_begin =
                decomp_index * base_p_size;
            const std::size_t remaining_q =
                base_q_size - decomp_limb_begin;
            const std::size_t decomp_limb_count =
                remaining_q < base_p_size ? remaining_q : base_p_size;
            const bool reuse_c2_ntt =
                local_limb >= decomp_limb_begin &&
                local_limb < decomp_limb_begin + decomp_limb_count;
            value = reuse_c2_ntt
                ? c2_ntt[local_offset]
                : all_modup_q[decomp_index * q_word_count + local_offset];
            key_offset = local_offset;
        }

        const GpuWord product0 =
            mul_mod(value, key_qp0[key_offset], modulus, barrett_ratio);
        const GpuWord product1 =
            mul_mod(value, key_qp1[key_offset], modulus, barrett_ratio);
        acc0 = add_mod(acc0, product0, modulus);
        acc1 = add_mod(acc1, product1, modulus);
    }

    if (p_limb)
    {
        accum_p0[local_offset] = acc0;
        accum_p1[local_offset] = acc1;
    }
    else
    {
        accum_q0[local_offset] = acc0;
        accum_q1[local_offset] = acc1;
    }
}

__global__ __launch_bounds__(256, 4)
void hybrid_final_ntt_paccum_all_dnum_two_components_kernel(
    GpuWord *accum_q0,
    GpuWord *accum_p0,
    GpuWord *accum_q1,
    GpuWord *accum_p1,
    const GpuWord *all_modup_q,
    const GpuWord *all_modup_p,
    const GpuWord *c2_ntt,
    const GpuWord *const *key_qp0_by_dnum,
    const GpuWord *const *key_qp1_by_dnum,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    std::size_t decomp_count,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree)
{
    constexpr std::size_t kFinalStages = 3;
    constexpr std::size_t kLocalSize = 1U << kFinalStages;

    const std::size_t tiles_per_limb = degree >> kFinalStages;
    const std::size_t total_limb_count = base_q_size + base_p_size;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_limb_count * tiles_per_limb)
    {
        return;
    }

    const std::size_t table_limb = tid / tiles_per_limb;
    const std::size_t local_limb = table_limb < base_q_size
        ? table_limb
        : table_limb - base_q_size;
    const std::size_t local_tile = tid % tiles_per_limb;
    const std::size_t local_base =
        local_limb * degree + local_tile * kLocalSize;
    const std::size_t q_word_count = base_q_size * degree;
    const std::size_t p_word_count = base_p_size * degree;
    const bool p_limb = table_limb >= base_q_size;

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord *limb_roots = roots + table_limb * degree;

    GpuWord acc0[kLocalSize] = {0};
    GpuWord acc1[kLocalSize] = {0};

    for (std::size_t decomp_index = 0;
         decomp_index < decomp_count;
         ++decomp_index)
    {
        const std::size_t decomp_limb_begin =
            decomp_index * base_p_size;
        const std::size_t remaining_q =
            base_q_size - decomp_limb_begin;
        const std::size_t decomp_limb_count =
            remaining_q < base_p_size ? remaining_q : base_p_size;
        const bool reuse_c2_ntt =
            !p_limb &&
            local_limb >= decomp_limb_begin &&
            local_limb < decomp_limb_begin + decomp_limb_count;

        GpuWord local[kLocalSize];
        const GpuWord *partial = p_limb
            ? all_modup_p + decomp_index * p_word_count
            : all_modup_q + decomp_index * q_word_count;
#pragma unroll
        for (std::size_t i = 0; i < kLocalSize; ++i)
        {
            local[i] = reuse_c2_ntt
                ? c2_ntt[local_base + i]
                : partial[local_base + i];
        }

        if (!reuse_c2_ntt)
        {
#pragma unroll
            for (std::size_t stage = 0; stage < kFinalStages; ++stage)
            {
                const std::size_t local_stride =
                    static_cast<std::size_t>(1)
                    << (kFinalStages - 1 - stage);
                const std::size_t stage_m =
                    (degree >> kFinalStages) << stage;
                const std::size_t stage_group_base =
                    local_tile << stage;

#pragma unroll
                for (std::size_t block = 0; block < kLocalSize;
                     block += (local_stride << 1))
                {
                    const std::size_t block_group =
                        block / (local_stride << 1);
                    const GpuWord root =
                        limb_roots[
                            stage_m + stage_group_base + block_group];

#pragma unroll
                    for (std::size_t offset = 0;
                         offset < local_stride;
                         ++offset)
                    {
                        const GpuWord u = local[block + offset];
                        const GpuWord v = mul_mod(
                            local[block + offset + local_stride],
                            root,
                            modulus,
                            barrett_ratio);
                        local[block + offset] = add_mod(u, v, modulus);
                        local[block + offset + local_stride] =
                            sub_mod(u, v, modulus);
                    }
                }
            }
        }

        const GpuWord *key_qp0 = key_qp0_by_dnum[decomp_index];
        const GpuWord *key_qp1 = key_qp1_by_dnum[decomp_index];
        const std::size_t key_base = p_limb
            ? q_word_count + local_base
            : local_base;
#pragma unroll
        for (std::size_t i = 0; i < kLocalSize; ++i)
        {
            acc0[i] = add_mod(
                acc0[i],
                mul_mod(
                    local[i],
                    key_qp0[key_base + i],
                    modulus,
                    barrett_ratio),
                modulus);
            acc1[i] = add_mod(
                acc1[i],
                mul_mod(
                    local[i],
                    key_qp1[key_base + i],
                    modulus,
                    barrett_ratio),
                modulus);
        }
    }

    GpuWord *output0 = p_limb ? accum_p0 : accum_q0;
    GpuWord *output1 = p_limb ? accum_p1 : accum_q1;
#pragma unroll
    for (std::size_t i = 0; i < kLocalSize; ++i)
    {
        output0[local_base + i] = acc0[i];
        output1[local_base + i] = acc1[i];
    }
}
#endif

__global__ void hybrid_multiply_accumulate_kernel(
    GpuWord *accum_q,
    GpuWord *accum_p,
    const GpuWord *modup_q,
    const GpuWord *modup_p,
    const GpuWord *key_qp,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t q_word_count = base_q_size * degree;
    const std::size_t p_word_count = base_p_size * degree;
    const std::size_t total = q_word_count + p_word_count;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total)
    {
        return;
    }

    GpuWord *accum = accum_q;
    const GpuWord *modup = modup_q;
    std::size_t local_offset = tid;
    std::size_t table_limb = tid >> degree_power;
    std::size_t key_offset = tid;

    if (tid >= q_word_count)
    {
        local_offset = tid - q_word_count;
        accum = accum_p;
        modup = modup_p;
        table_limb = base_q_size + (local_offset >> degree_power);
        key_offset = q_word_count + local_offset;
    }

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord product = mul_mod(
        modup[local_offset],
        key_qp[key_offset],
        modulus,
        barrett_ratio);
    accum[local_offset] =
        add_mod(accum[local_offset], product, modulus);
}

__global__ void hybrid_multiply_accumulate_two_components_kernel(
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
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    unsigned int degree_power,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    bool overwrite_accum)
{
    const std::size_t q_word_count = base_q_size * degree;
    const std::size_t p_word_count = base_p_size * degree;
    const std::size_t total = q_word_count + p_word_count;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total)
    {
        return;
    }

    GpuWord *accum0 = accum_q0;
    GpuWord *accum1 = accum_q1;
    const GpuWord *modup = modup_q;
    std::size_t local_offset = tid;
    std::size_t table_limb = tid >> degree_power;
    const GpuWord *key0 = key_q0;
    const GpuWord *key1 = key_q1;

    if (tid >= q_word_count)
    {
        local_offset = tid - q_word_count;
        accum0 = accum_p0;
        accum1 = accum_p1;
        modup = modup_p;
        table_limb = base_q_size + (local_offset >> degree_power);
        key0 = key_p0;
        key1 = key_p1;
    }

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const bool reuse_c2_ntt =
        tid < q_word_count &&
        table_limb >= decomp_limb_begin &&
        table_limb < decomp_limb_begin + decomp_limb_count;
    const GpuWord value =
        reuse_c2_ntt ? c2_ntt[local_offset] : modup[local_offset];
    const GpuWord product0 = mul_mod(
        value,
        key0[local_offset],
        modulus,
        barrett_ratio);
    const GpuWord product1 = mul_mod(
        value,
        key1[local_offset],
        modulus,
        barrett_ratio);

    if (overwrite_accum)
    {
        accum0[local_offset] = product0;
        accum1[local_offset] = product1;
        return;
    }

    accum0[local_offset] =
        add_mod(accum0[local_offset], product0, modulus);
    accum1[local_offset] =
        add_mod(accum1[local_offset], product1, modulus);
}

__global__ void hybrid_moddown_kernel(
    GpuWord *accum_q,
    const GpuWord *accum_p,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *moddown_p_to_q_matrix,
    const GpuWord *p_inv_punctured,
    const GpuWord *inv_p_mod_q,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = base_q_size * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t q_limb = tid >> degree_power;
    const std::size_t coeff = tid & (degree - 1);
    const GpuWord q_modulus = rns_primes[q_limb];
    const GpuWide q_barrett = rns_modulus_constants[q_limb];
    GpuWord converted_p = 0;

    for (std::size_t p_limb = 0; p_limb < base_p_size; ++p_limb)
    {
        const std::size_t p_table_limb = base_q_size + p_limb;
        const GpuWord p_value =
            accum_p[p_limb * degree + coeff];
        const GpuWord p_modulus = rns_primes[p_table_limb];
        const GpuWide p_barrett =
            rns_modulus_constants[p_table_limb];
        const GpuWord weighted = p_inv_punctured[p_limb] == 1
            ? p_value
            : mul_mod(
                  p_value,
                  p_inv_punctured[p_limb],
                  p_modulus,
                  p_barrett);
        const GpuWord matrix_value =
            moddown_p_to_q_matrix[q_limb * base_p_size + p_limb];
        const GpuWord product =
            mul_mod(weighted, matrix_value, q_modulus, q_barrett);
        converted_p = add_mod(converted_p, product, q_modulus);
    }

    const GpuWord difference =
        sub_mod(accum_q[tid], converted_p, q_modulus);
    accum_q[tid] = mul_mod(
        difference,
        inv_p_mod_q[q_limb],
        q_modulus,
        q_barrett);
}

__global__ void hybrid_convert_p_to_q_two_components_kernel(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *moddown_p_to_q_matrix,
    const GpuWord *p_inv_punctured,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = base_q_size * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t q_limb = tid >> degree_power;
    const std::size_t coeff = tid & (degree - 1);
    const GpuWord q_modulus = rns_primes[q_limb];
    const GpuWide q_barrett = rns_modulus_constants[q_limb];
    GpuWord sum0 = 0;
    GpuWord sum1 = 0;

    std::size_t p_limb = 0;
    for (; p_limb + 1 < base_p_size; p_limb += 2)
    {
        hybrid_convert_p_to_q_contribution(
            sum0,
            sum1,
            accum_p0,
            accum_p1,
            rns_primes,
            rns_modulus_constants,
            moddown_p_to_q_matrix,
            p_inv_punctured,
            base_q_size,
            base_p_size,
            q_limb,
            p_limb,
            coeff,
            degree,
            q_modulus,
            q_barrett);
        hybrid_convert_p_to_q_contribution(
            sum0,
            sum1,
            accum_p0,
            accum_p1,
            rns_primes,
            rns_modulus_constants,
            moddown_p_to_q_matrix,
            p_inv_punctured,
            base_q_size,
            base_p_size,
            q_limb,
            p_limb + 1,
            coeff,
            degree,
            q_modulus,
            q_barrett);
    }
    for (; p_limb < base_p_size; ++p_limb)
    {
        hybrid_convert_p_to_q_contribution(
            sum0,
            sum1,
            accum_p0,
            accum_p1,
            rns_primes,
            rns_modulus_constants,
            moddown_p_to_q_matrix,
            p_inv_punctured,
            base_q_size,
            base_p_size,
            q_limb,
            p_limb,
            coeff,
            degree,
            q_modulus,
            q_barrett);
    }

    converted_q0[tid] = sum0;
    converted_q1[tid] = sum1;
}

template <int FixedPCount, bool SourcePreweighted>
__global__ void hybrid_convert_p_to_q_forward_ntt_first_stage_kernel(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    const GpuWord *moddown_p_to_q_matrix,
    const GpuWord *p_inv_punctured,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t butterflies_per_limb = degree >> 1;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = base_q_size * butterflies_per_limb;
    if (tid >= total)
    {
        return;
    }

    const std::size_t q_limb = tid >> (degree_power - 1);
    const std::size_t left_coeff = tid & (butterflies_per_limb - 1);
    const std::size_t right_coeff = left_coeff + butterflies_per_limb;
    const GpuWord q_modulus = rns_primes[q_limb];
    const GpuWide q_barrett = rns_modulus_constants[q_limb];

    GpuWord left0;
    GpuWord right0;
    GpuWord left1;
    GpuWord right1;
    hybrid_convert_p_to_q_pair<FixedPCount, SourcePreweighted>(
        left0,
        right0,
        left1,
        right1,
        accum_p0,
        accum_p1,
        rns_primes,
        rns_modulus_constants,
        moddown_p_to_q_matrix,
        p_inv_punctured,
        base_q_size,
        base_p_size,
        q_limb,
        left_coeff,
        right_coeff,
        degree,
        q_modulus,
        q_barrett);

    const GpuWord root = roots[q_limb * degree + 1];
    const std::size_t q_base = q_limb * degree;

    const GpuWord twisted_right0 =
        mul_mod(right0, root, q_modulus, q_barrett);
    converted_q0[q_base + left_coeff] =
        add_mod(left0, twisted_right0, q_modulus);
    converted_q0[q_base + right_coeff] =
        sub_mod(left0, twisted_right0, q_modulus);

    const GpuWord twisted_right1 =
        mul_mod(right1, root, q_modulus, q_barrett);
    converted_q1[q_base + left_coeff] =
        add_mod(left1, twisted_right1, q_modulus);
    converted_q1[q_base + right_coeff] =
        sub_mod(left1, twisted_right1, q_modulus);
}

__global__ void hybrid_preweight_p_two_components_kernel(
    GpuWord *accum_p0,
    GpuWord *accum_p1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *p_inv_punctured,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = base_p_size * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t p_limb = tid >> degree_power;
    const GpuWord inv_punctured = p_inv_punctured[p_limb];
    if (inv_punctured == 1)
    {
        return;
    }

    const std::size_t table_limb = base_q_size + p_limb;
    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett = rns_modulus_constants[table_limb];
    accum_p0[tid] = mul_mod(
        accum_p0[tid], inv_punctured, modulus, barrett);
    accum_p1[tid] = mul_mod(
        accum_p1[tid], inv_punctured, modulus, barrett);
}

__global__ void
hybrid_convert_p_to_q_forward_ntt_first_stage_row_tiled8_kernel(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    const GpuWord *moddown_p_to_q_matrix,
    const GpuWord *p_inv_punctured,
    std::size_t base_q_size,
    std::size_t degree)
{
    constexpr std::size_t kPCount = 2;
    constexpr std::size_t kCoefficientTile = 32;
    constexpr std::size_t kTargetRows = 8;
    constexpr std::size_t kPlanesPerComponent = kPCount * 2;
    constexpr std::size_t kSourcePlanes = 2 * kPlanesPerComponent;
    __shared__ GpuWord weighted_p[kSourcePlanes][kCoefficientTile];

    const std::size_t lane = threadIdx.x;
    const std::size_t row = threadIdx.y;
    const std::size_t butterflies_per_limb = degree >> 1;
    const std::size_t coefficient_pair =
        blockIdx.x * kCoefficientTile + lane;
    const bool valid_coefficient =
        coefficient_pair < butterflies_per_limb;

    const std::size_t component = row / kPlanesPerComponent;
    const std::size_t component_plane = row % kPlanesPerComponent;
    const std::size_t p_limb = component_plane >> 1;
    const std::size_t coefficient_side = component_plane & 1;
    GpuWord weighted = 0;
    if (valid_coefficient)
    {
        const std::size_t coefficient =
            coefficient_pair + coefficient_side * butterflies_per_limb;
        const std::size_t p_table_limb = base_q_size + p_limb;
        const GpuWord p_modulus = rns_primes[p_table_limb];
        const GpuWide p_barrett =
            rns_modulus_constants[p_table_limb];
        const GpuWord inv_punctured = p_inv_punctured[p_limb];
        const GpuWord *component_values =
            component == 0 ? accum_p0 : accum_p1;
        const GpuWord value =
            component_values[p_limb * degree + coefficient];
        weighted = inv_punctured == 1
            ? value
            : mul_mod(value, inv_punctured, p_modulus, p_barrett);
    }
    weighted_p[row][lane] = weighted;
    __syncthreads();

    const std::size_t q_limb = blockIdx.y * kTargetRows + row;
    if (!valid_coefficient || q_limb >= base_q_size)
    {
        return;
    }

    const GpuWord q_modulus = rns_primes[q_limb];
    const GpuWide q_barrett = rns_modulus_constants[q_limb];
    const std::size_t matrix_base = q_limb * kPCount;
    const GpuWord matrix0 = moddown_p_to_q_matrix[matrix_base];
    const GpuWord matrix1 = moddown_p_to_q_matrix[matrix_base + 1];

    const GpuWord left0 = add_mod(
        mul_mod(weighted_p[0][lane], matrix0, q_modulus, q_barrett),
        mul_mod(weighted_p[2][lane], matrix1, q_modulus, q_barrett),
        q_modulus);
    const GpuWord right0 = add_mod(
        mul_mod(weighted_p[1][lane], matrix0, q_modulus, q_barrett),
        mul_mod(weighted_p[3][lane], matrix1, q_modulus, q_barrett),
        q_modulus);
    const GpuWord left1 = add_mod(
        mul_mod(weighted_p[4][lane], matrix0, q_modulus, q_barrett),
        mul_mod(weighted_p[6][lane], matrix1, q_modulus, q_barrett),
        q_modulus);
    const GpuWord right1 = add_mod(
        mul_mod(weighted_p[5][lane], matrix0, q_modulus, q_barrett),
        mul_mod(weighted_p[7][lane], matrix1, q_modulus, q_barrett),
        q_modulus);

    const GpuWord root = roots[q_limb * degree + 1];
    const std::size_t q_base = q_limb * degree;
    const std::size_t right_coefficient =
        coefficient_pair + butterflies_per_limb;

    const GpuWord twisted_right0 =
        mul_mod(right0, root, q_modulus, q_barrett);
    converted_q0[q_base + coefficient_pair] =
        add_mod(left0, twisted_right0, q_modulus);
    converted_q0[q_base + right_coefficient] =
        sub_mod(left0, twisted_right0, q_modulus);

    const GpuWord twisted_right1 =
        mul_mod(right1, root, q_modulus, q_barrett);
    converted_q1[q_base + coefficient_pair] =
        add_mod(left1, twisted_right1, q_modulus);
    converted_q1[q_base + right_coefficient] =
        sub_mod(left1, twisted_right1, q_modulus);
}

template <bool SourcePreweighted>
__global__ void
hybrid_convert_p9_to_q_forward_ntt_first_stage_row_tiled8_kernel(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    const GpuWord *moddown_p_to_q_matrix,
    const GpuWord *p_inv_punctured,
    std::size_t base_q_size,
    std::size_t degree)
{
    constexpr std::size_t kPCount = 9;
    constexpr std::size_t kCoefficientTile = 32;
    constexpr std::size_t kTargetRows = 8;
    constexpr std::size_t kPlanesPerComponent = kPCount * 2;
    constexpr std::size_t kSourcePlanes = 2 * kPlanesPerComponent;
    constexpr std::size_t kBlockThreads =
        kCoefficientTile * kTargetRows;
    __shared__ GpuWord weighted_p[kSourcePlanes][kCoefficientTile];

    const std::size_t lane = threadIdx.x;
    const std::size_t row = threadIdx.y;
    const std::size_t linear_thread =
        row * kCoefficientTile + lane;
    const std::size_t butterflies_per_limb = degree >> 1;

    for (std::size_t source_index = linear_thread;
         source_index < kSourcePlanes * kCoefficientTile;
         source_index += kBlockThreads)
    {
        const std::size_t source_plane =
            source_index / kCoefficientTile;
        const std::size_t source_lane =
            source_index % kCoefficientTile;
        const std::size_t component =
            source_plane / kPlanesPerComponent;
        const std::size_t component_plane =
            source_plane % kPlanesPerComponent;
        const std::size_t p_limb = component_plane >> 1;
        const std::size_t coefficient_side = component_plane & 1;
        const std::size_t coefficient_pair =
            blockIdx.x * kCoefficientTile + source_lane;

        GpuWord weighted = 0;
        if (coefficient_pair < butterflies_per_limb)
        {
            const std::size_t coefficient = coefficient_pair +
                coefficient_side * butterflies_per_limb;
            const GpuWord *component_values =
                component == 0 ? accum_p0 : accum_p1;
            weighted = component_values[p_limb * degree + coefficient];
            if constexpr (!SourcePreweighted)
            {
                const std::size_t table_limb = base_q_size + p_limb;
                const GpuWord modulus = rns_primes[table_limb];
                const GpuWide barrett =
                    rns_modulus_constants[table_limb];
                const GpuWord inv_punctured = p_inv_punctured[p_limb];
                if (inv_punctured != 1)
                {
                    weighted = mul_mod(
                        weighted,
                        inv_punctured,
                        modulus,
                        barrett);
                }
            }
        }
        weighted_p[source_plane][source_lane] = weighted;
    }
    __syncthreads();

    const std::size_t coefficient_pair =
        blockIdx.x * kCoefficientTile + lane;
    const std::size_t q_limb = blockIdx.y * kTargetRows + row;
    if (coefficient_pair >= butterflies_per_limb ||
        q_limb >= base_q_size)
    {
        return;
    }

    const GpuWord q_modulus = rns_primes[q_limb];
    const GpuWide q_barrett = rns_modulus_constants[q_limb];
    const std::size_t matrix_base = q_limb * kPCount;
    GpuWord left0 = 0;
    GpuWord right0 = 0;
    GpuWord left1 = 0;
    GpuWord right1 = 0;
#pragma unroll
    for (std::size_t p_limb = 0; p_limb < kPCount; ++p_limb)
    {
        const GpuWord matrix_value =
            moddown_p_to_q_matrix[matrix_base + p_limb];
        const std::size_t component0_plane = p_limb * 2;
        const std::size_t component1_plane =
            kPlanesPerComponent + component0_plane;
        left0 = add_mod(
            left0,
            mul_mod(
                weighted_p[component0_plane][lane],
                matrix_value,
                q_modulus,
                q_barrett),
            q_modulus);
        right0 = add_mod(
            right0,
            mul_mod(
                weighted_p[component0_plane + 1][lane],
                matrix_value,
                q_modulus,
                q_barrett),
            q_modulus);
        left1 = add_mod(
            left1,
            mul_mod(
                weighted_p[component1_plane][lane],
                matrix_value,
                q_modulus,
                q_barrett),
            q_modulus);
        right1 = add_mod(
            right1,
            mul_mod(
                weighted_p[component1_plane + 1][lane],
                matrix_value,
                q_modulus,
                q_barrett),
            q_modulus);
    }

    const GpuWord root = roots[q_limb * degree + 1];
    const std::size_t q_base = q_limb * degree;
    const std::size_t right_coefficient =
        coefficient_pair + butterflies_per_limb;
    const GpuWord twisted_right0 =
        mul_mod(right0, root, q_modulus, q_barrett);
    converted_q0[q_base + coefficient_pair] =
        add_mod(left0, twisted_right0, q_modulus);
    converted_q0[q_base + right_coefficient] =
        sub_mod(left0, twisted_right0, q_modulus);

    const GpuWord twisted_right1 =
        mul_mod(right1, root, q_modulus, q_barrett);
    converted_q1[q_base + coefficient_pair] =
        add_mod(left1, twisted_right1, q_modulus);
    converted_q1[q_base + right_coefficient] =
        sub_mod(left1, twisted_right1, q_modulus);
}

template <int FusionStages>
__global__ void hybrid_forward_ntt_q_two_components_fused_stage_kernel(
    GpuWord *values0,
    GpuWord *values1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    std::size_t base_q_size,
    std::size_t degree,
    std::size_t m,
    std::size_t gap)
{
    static_assert(
        FusionStages >= 1 && FusionStages <= 3,
        "HYBRID two-component Q NTT fusion supports 1-3 stages");

    constexpr std::size_t kLocalSize =
        static_cast<std::size_t>(1) << FusionStages;
    const std::size_t tiles_per_limb = degree >> FusionStages;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = 2 * base_q_size * tiles_per_limb;
    if (tid >= total)
    {
        return;
    }

    const std::size_t component_limb = tid / tiles_per_limb;
    const std::size_t local_tile = tid % tiles_per_limb;
    const bool second_component = component_limb >= base_q_size;
    const std::size_t q_limb = second_component
        ? component_limb - base_q_size
        : component_limb;
    GpuWord *values = second_component ? values1 : values0;

    const std::size_t final_gap = gap >> (FusionStages - 1);
    const std::size_t outer_group = local_tile / final_gap;
    const std::size_t j = local_tile % final_gap;
    const std::size_t base_index =
        q_limb * degree + outer_group * (gap << 1) + j;
    const GpuWord modulus = rns_primes[q_limb];
    const GpuWide barrett = rns_modulus_constants[q_limb];
    const GpuWord *limb_roots = roots + q_limb * degree;

    GpuWord local[kLocalSize];
#pragma unroll
    for (std::size_t i = 0; i < kLocalSize; ++i)
    {
        local[i] = values[base_index + i * final_gap];
    }

#pragma unroll
    for (int stage = 0; stage < FusionStages; ++stage)
    {
        const std::size_t local_stride =
            static_cast<std::size_t>(1) << (FusionStages - 1 - stage);
        const std::size_t stage_m = m << stage;
        const std::size_t stage_group_base = outer_group << stage;

#pragma unroll
        for (std::size_t block = 0; block < kLocalSize;
             block += (local_stride << 1))
        {
            const std::size_t block_group = block / (local_stride << 1);
            const GpuWord root =
                limb_roots[stage_m + stage_group_base + block_group];

#pragma unroll
            for (std::size_t offset = 0; offset < local_stride; ++offset)
            {
                const GpuWord u = local[block + offset];
                const GpuWord v = mul_mod(
                    local[block + offset + local_stride],
                    root,
                    modulus,
                    barrett);
                local[block + offset] = add_mod(u, v, modulus);
                local[block + offset + local_stride] =
                    sub_mod(u, v, modulus);
            }
        }
    }

#pragma unroll
    for (std::size_t i = 0; i < kLocalSize; ++i)
    {
        values[base_index + i * final_gap] = local[i];
    }
}

__global__ void hybrid_apply_moddown_ntt_two_components_kernel(
    GpuWord *accum_q0,
    GpuWord *accum_q1,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *inv_p_mod_q,
    std::size_t base_q_size,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = base_q_size * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t q_limb = tid >> degree_power;
    const GpuWord modulus = rns_primes[q_limb];
    const GpuWide barrett = rns_modulus_constants[q_limb];
    const GpuWord inv_p = inv_p_mod_q[q_limb];

    const GpuWord difference0 =
        sub_mod(accum_q0[tid], converted_q0[tid], modulus);
    accum_q0[tid] = mul_mod(difference0, inv_p, modulus, barrett);

    const GpuWord difference1 =
        sub_mod(accum_q1[tid], converted_q1[tid], modulus);
    accum_q1[tid] = mul_mod(difference1, inv_p, modulus, barrett);
}

__global__ void hybrid_apply_moddown_ntt_out_of_place_two_components_kernel(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    const GpuWord *source_q0,
    const GpuWord *source_q1,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *inv_p_mod_q,
    std::size_t base_q_size,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = base_q_size * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t q_limb = tid >> degree_power;
    const GpuWord modulus = rns_primes[q_limb];
    const GpuWide barrett = rns_modulus_constants[q_limb];
    const GpuWord inv_p = inv_p_mod_q[q_limb];

    destination_q0[tid] = mul_mod(
        sub_mod(source_q0[tid], converted_q0[tid], modulus),
        inv_p,
        modulus,
        barrett);
    destination_q1[tid] = mul_mod(
        sub_mod(source_q1[tid], converted_q1[tid], modulus),
        inv_p,
        modulus,
        barrett);
}

__global__ void hybrid_apply_moddown_ntt_out_of_place_batch_kernel(
    GpuWord *destination_q,
    const GpuWord *source_q,
    const GpuWord *converted_q,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *inv_p_mod_q,
    std::size_t batch_count,
    std::size_t base_q_size,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t words_per_component = base_q_size * degree;
    const std::size_t total = batch_count * 2 * words_per_component;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total)
    {
        return;
    }
    const std::size_t local = tid % words_per_component;
    const std::size_t q_limb = local >> degree_power;
    const GpuWord modulus = rns_primes[q_limb];
    destination_q[tid] = mul_mod(
        sub_mod(source_q[tid], converted_q[tid], modulus),
        inv_p_mod_q[q_limb],
        modulus,
        rns_modulus_constants[q_limb]);
}

__global__ void hybrid_apply_moddown_ntt_from_q_groups_kernel(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    const GpuWord *group_q,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *inv_p_mod_q,
    std::size_t group_count,
    std::size_t base_q_size,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t component_words = base_q_size * degree;
    if (tid >= component_words)
    {
        return;
    }
    const std::size_t q_limb = tid >> degree_power;
    const GpuWord modulus = rns_primes[q_limb];
    GpuWord sum0 = 0;
    GpuWord sum1 = 0;
    for (std::size_t group = 0; group < group_count; ++group)
    {
        sum0 = add_mod(
            sum0,
            group_q[(group * 2) * component_words + tid],
            modulus);
        sum1 = add_mod(
            sum1,
            group_q[(group * 2 + 1) * component_words + tid],
            modulus);
    }
    const GpuWide barrett = rns_modulus_constants[q_limb];
    const GpuWord inv_p = inv_p_mod_q[q_limb];
    destination_q0[tid] = mul_mod(
        sub_mod(sum0, converted_q0[tid], modulus),
        inv_p,
        modulus,
        barrett);
    destination_q1[tid] = mul_mod(
        sub_mod(sum1, converted_q1[tid], modulus),
        inv_p,
        modulus,
        barrett);
}

__global__ void hybrid_apply_moddown_ntt_add_back_two_components_kernel(
    GpuWord *destination0,
    GpuWord *destination1,
    const GpuWord *add_source0,
    const GpuWord *add_source1,
    const GpuWord *accum_q0,
    const GpuWord *accum_q1,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *inv_p_mod_q,
    std::size_t modulus_offset,
    std::size_t limb_count,
    std::size_t coeff_count,
    unsigned int degree_power)
{
    const std::size_t values_per_component = limb_count * coeff_count;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= values_per_component * 2)
    {
        return;
    }

    const bool second_component = tid >= values_per_component;
    const std::size_t local_index =
        second_component ? tid - values_per_component : tid;
    const std::size_t local_limb = local_index >> degree_power;
    const std::size_t local_coeff = local_index & (coeff_count - 1);
    const std::size_t table_limb = modulus_offset + local_limb;
    const std::size_t scratch_index = table_limb * coeff_count + local_coeff;
    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett = rns_modulus_constants[table_limb];
    const GpuWord inv_p = inv_p_mod_q[table_limb];

    GpuWord *destination = second_component ? destination1 : destination0;
    const GpuWord *add_source =
        second_component ? add_source1 : add_source0;
    const GpuWord *accum_q = second_component ? accum_q1 : accum_q0;
    const GpuWord *converted_q = second_component ? converted_q1 : converted_q0;

    const GpuWord difference =
        sub_mod(accum_q[scratch_index], converted_q[scratch_index], modulus);
    const GpuWord moddown =
        mul_mod(difference, inv_p, modulus, barrett);
    destination[local_index] =
        add_mod(add_source[local_index], moddown, modulus);
}

__global__ void hybrid_apply_moddown_ntt_add_back_rescale_x2_two_components_kernel(
    GpuWord *destination0,
    GpuWord *destination1,
    const GpuWord *add_source0,
    const GpuWord *add_source1,
    const GpuWord *correction_ntt0,
    const GpuWord *correction_ntt1,
    const GpuWord *accum_q0,
    const GpuWord *accum_q1,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *inv_p_mod_q,
    const GpuWord *inv_q_last_two_product_mod_q,
    std::size_t limb_count,
    std::size_t coeff_count,
    unsigned int degree_power)
{
    const std::size_t values_per_component = limb_count * coeff_count;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= values_per_component * 2)
    {
        return;
    }

    const bool second_component = tid >= values_per_component;
    const std::size_t local_index =
        second_component ? tid - values_per_component : tid;
    const std::size_t q_limb = local_index >> degree_power;
    const GpuWord modulus = rns_primes[q_limb];
    const GpuWide barrett = rns_modulus_constants[q_limb];

    GpuWord *destination = second_component ? destination1 : destination0;
    const GpuWord *add_source =
        second_component ? add_source1 : add_source0;
    const GpuWord *correction_ntt =
        second_component ? correction_ntt1 : correction_ntt0;
    const GpuWord *accum_q = second_component ? accum_q1 : accum_q0;
    const GpuWord *converted_q = second_component ? converted_q1 : converted_q0;

    const GpuWord switched = mul_mod(
        sub_mod(accum_q[local_index], converted_q[local_index], modulus),
        inv_p_mod_q[q_limb],
        modulus,
        barrett);
    const GpuWord relin_value =
        add_mod(add_source[local_index], switched, modulus);
    const GpuWord corrected =
        sub_mod(relin_value, correction_ntt[local_index], modulus);
    destination[local_index] = mul_mod(
        corrected,
        inv_q_last_two_product_mod_q[q_limb],
        modulus,
        barrett);
}

void validate_hybrid_tables(
    const char *name,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    const std::size_t decomp_count = parameter_shard.hybrid_decomp_count;

    if (degree == 0 || base_q_size == 0 || base_p_size == 0 || decomp_count == 0)
    {
        throw std::invalid_argument(std::string(name) + ": empty HYBRID shape");
    }
    if (parameter_shard.rns_primes.data() == nullptr ||
        parameter_shard.rns_modulus_constants.data() == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null RNS modulus tables");
    }
    if (parameter_shard.rns_primes.size() < base_q_size + base_p_size ||
        parameter_shard.rns_modulus_constants.size() < base_q_size + base_p_size)
    {
        throw std::invalid_argument(std::string(name) + ": RNS modulus tables are too small");
    }
    if (parameter_shard.hybrid_q_conv_matrix_offsets.data() == nullptr ||
        parameter_shard.hybrid_p_conv_matrix_offsets.data() == nullptr ||
        parameter_shard.hybrid_q_conv_matrices.data() == nullptr ||
        parameter_shard.hybrid_p_conv_matrices.data() == nullptr ||
        parameter_shard.hybrid_qi_inv_punctured.data() == nullptr ||
        parameter_shard.hybrid_moddown_p_to_q_matrix.data() == nullptr ||
        parameter_shard.hybrid_p_inv_punctured.data() == nullptr ||
        parameter_shard.hybrid_inv_p_mod_q.data() == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null HYBRID table pointer");
    }
    if (parameter_shard.hybrid_q_conv_matrix_offsets.size() < decomp_count + 1 ||
        parameter_shard.hybrid_p_conv_matrix_offsets.size() < decomp_count + 1 ||
        parameter_shard.hybrid_q_conv_matrices.size() <
            decomp_count * base_q_size * base_p_size ||
        parameter_shard.hybrid_p_conv_matrices.size() <
            decomp_count * base_p_size * base_p_size ||
        parameter_shard.hybrid_qi_inv_punctured.size() < decomp_count * base_p_size ||
        parameter_shard.hybrid_moddown_p_to_q_matrix.size() < base_q_size * base_p_size ||
        parameter_shard.hybrid_p_inv_punctured.size() < base_p_size ||
        parameter_shard.hybrid_inv_p_mod_q.size() < base_q_size)
    {
        throw std::invalid_argument(std::string(name) + ": HYBRID tables are too small");
    }
}

unsigned int checked_log2_degree(std::size_t degree, const char *name)
{
    if (degree < 2 || (degree & (degree - 1)) != 0)
    {
        throw std::invalid_argument(std::string(name) + ": degree must be a power of two");
    }

    unsigned int result = 0;
    while (degree > 1)
    {
        degree >>= 1;
        ++result;
    }
    return result;
}

void launch_hybrid_forward_ntt_q_two_components_stages(
    GpuWord *values0,
    GpuWord *values1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    unsigned int degree_power,
    unsigned int completed_stages,
    const char *name)
{
    constexpr int kFusionStages = 3;
    constexpr int block_size = 256;
    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    std::size_t remaining_stages = degree_power - completed_stages;
    std::size_t m = static_cast<std::size_t>(1) << completed_stages;
    std::size_t gap = degree >> (completed_stages + 1);

    while (remaining_stages > 0)
    {
        std::size_t stage_count = remaining_stages % kFusionStages;
        if (stage_count == 0)
        {
            stage_count = kFusionStages;
        }

        const std::size_t tiles_per_limb = degree >> stage_count;
        const std::size_t total = 2 * base_q_size * tiles_per_limb;
        const int grid_size = static_cast<int>(
            (total + block_size - 1) / block_size);

        if (stage_count == 1)
        {
            hybrid_forward_ntt_q_two_components_fused_stage_kernel<1>
                <<<grid_size, block_size>>>(
                    values0,
                    values1,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    parameter_shard.ntt_tables.data(),
                    base_q_size,
                    degree,
                    m,
                    gap);
        }
        else if (stage_count == 2)
        {
            hybrid_forward_ntt_q_two_components_fused_stage_kernel<2>
                <<<grid_size, block_size>>>(
                    values0,
                    values1,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    parameter_shard.ntt_tables.data(),
                    base_q_size,
                    degree,
                    m,
                    gap);
        }
        else
        {
            hybrid_forward_ntt_q_two_components_fused_stage_kernel<3>
                <<<grid_size, block_size>>>(
                    values0,
                    values1,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    parameter_shard.ntt_tables.data(),
                    base_q_size,
                    degree,
                    m,
                    gap);
        }
        gpu_check_cuda(cudaGetLastError(), name);

        m <<= stage_count;
        gap >>= stage_count;
        remaining_stages -= stage_count;
    }
}

}  // namespace

void launch_hybrid_modup_decomposition(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    const GpuWord *c2_ntt,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_modup_decomposition",
        parameter_shard,
        degree);

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    if (modup_q == nullptr || modup_p == nullptr ||
        c2_coeff == nullptr || c2_ntt == nullptr)
    {
        throw std::invalid_argument("launch_hybrid_modup_decomposition: null data pointer");
    }
    /* Check that the current HYBRID decomposition block stays inside base Q. */
    if (decomp_index >= parameter_shard.hybrid_decomp_count ||
        decomp_limb_count == 0 ||
        decomp_limb_count > base_p_size ||
        decomp_limb_begin + decomp_limb_count > base_q_size)
    {
        throw std::invalid_argument("launch_hybrid_modup_decomposition: invalid decomposition range");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_modup_decomposition cudaSetDevice");

    constexpr int block_size = 256;
    const std::size_t total = (base_q_size + base_p_size) * degree;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_modup_decomposition");
    /* 发射模升kernel */
    hybrid_modup_qp_kernel<<<grid_size, block_size>>>(
        modup_q,
        modup_p,
        c2_coeff,
        c2_ntt,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.hybrid_q_conv_matrix_offsets.data(),
        parameter_shard.hybrid_q_conv_matrices.data(),
        parameter_shard.hybrid_p_conv_matrix_offsets.data(),
        parameter_shard.hybrid_p_conv_matrices.data(),
        parameter_shard.hybrid_qi_inv_punctured.data(),
        decomp_index,
        decomp_limb_begin,
        decomp_limb_count,
        base_q_size,
        base_p_size,
        degree,
        degree_power);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_modup_decomposition kernel launch");
}

void launch_hybrid_modup_decomposition_forward_ntt_first_stage(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_modup_decomposition_forward_ntt_first_stage",
        parameter_shard,
        degree);

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    if (modup_q == nullptr || modup_p == nullptr || c2_coeff == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_modup_decomposition_forward_ntt_first_stage: null data pointer");
    }
    if (parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_modup_decomposition_forward_ntt_first_stage: null NTT table pointer");
    }
    if (degree < 2 ||
        decomp_index >= parameter_shard.hybrid_decomp_count ||
        decomp_limb_count == 0 ||
        decomp_limb_count > base_p_size ||
        decomp_limb_begin + decomp_limb_count > base_q_size)
    {
        throw std::invalid_argument(
            "launch_hybrid_modup_decomposition_forward_ntt_first_stage: invalid decomposition range");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_modup_decomposition_forward_ntt_first_stage cudaSetDevice");

    constexpr int block_size = 256;
    const std::size_t active_limb_count =
        base_q_size - decomp_limb_count + base_p_size;
    const std::size_t total = active_limb_count * (degree >> 1);
    const int grid_size = static_cast<int>(
        (total + block_size - 1) / block_size);
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_modup_decomposition_forward_ntt_first_stage");

#define POSEIDON_LAUNCH_MODUP_NTT_HEAD(FIXED_COUNT)                         \
    hybrid_modup_qp_forward_ntt_first_stage_kernel<FIXED_COUNT>            \
        <<<grid_size, block_size>>>(                                       \
            modup_q,                                                       \
            modup_p,                                                       \
            c2_coeff,                                                      \
            parameter_shard.rns_primes.data(),                             \
            parameter_shard.rns_modulus_constants.data(),                  \
            parameter_shard.ntt_tables.data(),                             \
            parameter_shard.hybrid_q_conv_matrix_offsets.data(),           \
            parameter_shard.hybrid_q_conv_matrices.data(),                 \
            parameter_shard.hybrid_p_conv_matrix_offsets.data(),           \
            parameter_shard.hybrid_p_conv_matrices.data(),                 \
            parameter_shard.hybrid_qi_inv_punctured.data(),                \
            decomp_index,                                                  \
            decomp_limb_begin,                                             \
            decomp_limb_count,                                             \
            base_q_size,                                                   \
            base_p_size,                                                   \
            degree,                                                        \
            degree_power)

    if (decomp_limb_count == 1)
    {
        POSEIDON_LAUNCH_MODUP_NTT_HEAD(1);
    }
    else if (decomp_limb_count == 2)
    {
        POSEIDON_LAUNCH_MODUP_NTT_HEAD(2);
    }
    else if (decomp_limb_count == 3)
    {
        POSEIDON_LAUNCH_MODUP_NTT_HEAD(3);
    }
    else if (decomp_limb_count == 9)
    {
        POSEIDON_LAUNCH_MODUP_NTT_HEAD(9);
    }
    else
    {
        POSEIDON_LAUNCH_MODUP_NTT_HEAD(0);
    }
#undef POSEIDON_LAUNCH_MODUP_NTT_HEAD

    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_modup_decomposition_forward_ntt_first_stage kernel launch");
}

void launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    if (decomp_limb_count != 2)
    {
        launch_hybrid_modup_decomposition_forward_ntt_first_stage(
            modup_q,
            modup_p,
            c2_coeff,
            decomp_index,
            decomp_limb_begin,
            decomp_limb_count,
            parameter_shard,
            degree);
        return;
    }

    validate_hybrid_tables(
        "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled",
        parameter_shard,
        degree);
    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    if (modup_q == nullptr || modup_p == nullptr || c2_coeff == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled: null data pointer");
    }
    if (parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled: null NTT table pointer");
    }
    if (decomp_index >= parameter_shard.hybrid_decomp_count ||
        decomp_limb_count > base_p_size ||
        decomp_limb_begin + decomp_limb_count > base_q_size)
    {
        throw std::invalid_argument(
            "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled: invalid decomposition range");
    }

    checked_log2_degree(
        degree,
        "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled");
    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled cudaSetDevice");

    constexpr unsigned int kCoefficientTile = 32;
    constexpr unsigned int kTargetRows = 4;
    const std::size_t active_limb_count =
        base_q_size - decomp_limb_count + base_p_size;
    const dim3 block_size(kCoefficientTile, kTargetRows);
    const dim3 grid_size(
        static_cast<unsigned int>(
            ((degree >> 1) + kCoefficientTile - 1) / kCoefficientTile),
        static_cast<unsigned int>(
            (active_limb_count + kTargetRows - 1) / kTargetRows));

    hybrid_modup_qp_forward_ntt_first_stage_row_tiled2_kernel
        <<<grid_size, block_size>>>(
            modup_q,
            modup_p,
            c2_coeff,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.ntt_tables.data(),
            parameter_shard.hybrid_q_conv_matrix_offsets.data(),
            parameter_shard.hybrid_q_conv_matrices.data(),
            parameter_shard.hybrid_p_conv_matrix_offsets.data(),
            parameter_shard.hybrid_p_conv_matrices.data(),
            parameter_shard.hybrid_qi_inv_punctured.data(),
            decomp_index,
            decomp_limb_begin,
            base_q_size,
            base_p_size,
            degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled kernel launch");
}

void launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled8(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    if (decomp_limb_count != 2)
    {
        launch_hybrid_modup_decomposition_forward_ntt_first_stage(
            modup_q,
            modup_p,
            c2_coeff,
            decomp_index,
            decomp_limb_begin,
            decomp_limb_count,
            parameter_shard,
            degree);
        return;
    }

    validate_hybrid_tables(
        "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled8",
        parameter_shard,
        degree);
    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    if (modup_q == nullptr || modup_p == nullptr || c2_coeff == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled8: null data pointer");
    }
    if (parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled8: null NTT table pointer");
    }
    if (decomp_index >= parameter_shard.hybrid_decomp_count ||
        decomp_limb_count > base_p_size ||
        decomp_limb_begin + decomp_limb_count > base_q_size)
    {
        throw std::invalid_argument(
            "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled8: invalid decomposition range");
    }

    checked_log2_degree(
        degree,
        "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled8");
    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled8 cudaSetDevice");

    constexpr unsigned int kCoefficientTile = 32;
    constexpr unsigned int kTargetRows = 8;
    const std::size_t active_limb_count =
        base_q_size - decomp_limb_count + base_p_size;
    const dim3 block_size(kCoefficientTile, kTargetRows);
    const dim3 grid_size(
        static_cast<unsigned int>(
            ((degree >> 1) + kCoefficientTile - 1) / kCoefficientTile),
        static_cast<unsigned int>(
            (active_limb_count + kTargetRows - 1) / kTargetRows));

    hybrid_modup_qp_forward_ntt_first_stage_row_tiled8_kernel
        <<<grid_size, block_size>>>(
            modup_q,
            modup_p,
            c2_coeff,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.ntt_tables.data(),
            parameter_shard.hybrid_q_conv_matrix_offsets.data(),
            parameter_shard.hybrid_q_conv_matrices.data(),
            parameter_shard.hybrid_p_conv_matrix_offsets.data(),
            parameter_shard.hybrid_p_conv_matrices.data(),
            parameter_shard.hybrid_qi_inv_punctured.data(),
            decomp_index,
            decomp_limb_begin,
            base_q_size,
            base_p_size,
            degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled8 kernel launch");
}

void launch_hybrid_modup_decomposition_row_tiled8(
    GpuWord *modup_q,
    GpuWord *modup_p,
    const GpuWord *c2_coeff,
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    if (decomp_limb_count == 0 || decomp_limb_count > 9)
    {
        throw std::invalid_argument(
            "launch_hybrid_modup_decomposition_row_tiled8: one through nine decomposition limbs are required");
    }
    validate_hybrid_tables(
        "launch_hybrid_modup_decomposition_row_tiled8",
        parameter_shard,
        degree);
    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    if (modup_q == nullptr || modup_p == nullptr || c2_coeff == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_modup_decomposition_row_tiled8: null data pointer");
    }
    if (decomp_index >= parameter_shard.hybrid_decomp_count ||
        decomp_limb_count > base_p_size ||
        decomp_limb_begin + decomp_limb_count > base_q_size)
    {
        throw std::invalid_argument(
            "launch_hybrid_modup_decomposition_row_tiled8: invalid decomposition range");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_modup_decomposition_row_tiled8 cudaSetDevice");
    constexpr unsigned int kCoefficientTile = 32;
    constexpr unsigned int kTargetRows = 8;
    const std::size_t active_limb_count =
        base_q_size - decomp_limb_count + base_p_size;
    const dim3 block_size(kCoefficientTile, kTargetRows);
    const dim3 grid_size(
        static_cast<unsigned int>(
            ((degree >> 1) + kCoefficientTile - 1) / kCoefficientTile),
        static_cast<unsigned int>(
            (active_limb_count + kTargetRows - 1) / kTargetRows));

    if (decomp_limb_count == 2)
    {
        hybrid_modup_qp_row_tiled8_kernel
            <<<grid_size, block_size>>>(
                modup_q,
                modup_p,
                c2_coeff,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.hybrid_q_conv_matrix_offsets.data(),
                parameter_shard.hybrid_q_conv_matrices.data(),
                parameter_shard.hybrid_p_conv_matrix_offsets.data(),
                parameter_shard.hybrid_p_conv_matrices.data(),
                parameter_shard.hybrid_qi_inv_punctured.data(),
                decomp_index,
                decomp_limb_begin,
                base_q_size,
                base_p_size,
                degree);
    }
    else
    {
#define POSEIDON_LAUNCH_P9_MODUP_ROW_TILED8(FIXED_COUNT)                   \
        hybrid_modup_qp_p9_row_tiled8_kernel<FIXED_COUNT>                 \
            <<<grid_size, block_size>>>(                                  \
                modup_q,                                                  \
                modup_p,                                                  \
                c2_coeff,                                                 \
                parameter_shard.rns_primes.data(),                        \
                parameter_shard.rns_modulus_constants.data(),             \
                parameter_shard.hybrid_q_conv_matrix_offsets.data(),      \
                parameter_shard.hybrid_q_conv_matrices.data(),            \
                parameter_shard.hybrid_p_conv_matrix_offsets.data(),      \
                parameter_shard.hybrid_p_conv_matrices.data(),            \
                parameter_shard.hybrid_qi_inv_punctured.data(),           \
                decomp_index,                                             \
                decomp_limb_begin,                                        \
                base_q_size,                                              \
                base_p_size,                                              \
                degree)

        switch (decomp_limb_count)
        {
        case 1:
            POSEIDON_LAUNCH_P9_MODUP_ROW_TILED8(1);
            break;
        case 3:
            POSEIDON_LAUNCH_P9_MODUP_ROW_TILED8(3);
            break;
        case 4:
            POSEIDON_LAUNCH_P9_MODUP_ROW_TILED8(4);
            break;
        case 5:
            POSEIDON_LAUNCH_P9_MODUP_ROW_TILED8(5);
            break;
        case 6:
            POSEIDON_LAUNCH_P9_MODUP_ROW_TILED8(6);
            break;
        case 7:
            POSEIDON_LAUNCH_P9_MODUP_ROW_TILED8(7);
            break;
        case 8:
            POSEIDON_LAUNCH_P9_MODUP_ROW_TILED8(8);
            break;
        case 9:
            POSEIDON_LAUNCH_P9_MODUP_ROW_TILED8(9);
            break;
        default:
            throw std::logic_error(
                "launch_hybrid_modup_decomposition_row_tiled8: unreachable decomposition width");
        }
#undef POSEIDON_LAUNCH_P9_MODUP_ROW_TILED8
    }
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_modup_decomposition_row_tiled8 kernel launch");
}

#if 0
/* Retired all-dnum preparation launchers. */
void launch_hybrid_forward_ntt_qp(
    GpuWord *modup_q,
    GpuWord *modup_p,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_forward_ntt_qp",
        parameter_shard,
        degree);

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    if (modup_q == nullptr || modup_p == nullptr)
    {
        throw std::invalid_argument("launch_hybrid_forward_ntt_qp: null data pointer");
    }
    if (parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument("launch_hybrid_forward_ntt_qp: null NTT table pointer");
    }
    if (decomp_limb_count == 0 ||
        decomp_limb_count > base_p_size ||
        decomp_limb_begin + decomp_limb_count > base_q_size)
    {
        throw std::invalid_argument("launch_hybrid_forward_ntt_qp: invalid decomposition range");
    }
    if (parameter_shard.limb_begin != 0 ||
        parameter_shard.limb_count < base_q_size + base_p_size)
    {
        throw std::invalid_argument("launch_hybrid_forward_ntt_qp: parameter shard must cover full QP limb range");
    }
    if (parameter_shard.ntt_tables.size() <
        (base_q_size + base_p_size) * degree)
    {
        throw std::invalid_argument("launch_hybrid_forward_ntt_qp: NTT tables are too small");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_forward_ntt_qp cudaSetDevice");

    /* 计算真正需要做NTT的部分，decomp_limb_count在块内，之前有NTT形式不需要重复计算*/
    const std::size_t active_limb_count =
        base_q_size - decomp_limb_count + base_p_size;
    constexpr int kFusionStages = 3;
    constexpr int block_size = 256;
    std::size_t remaining_stages = checked_log2_degree(
        degree,
        "launch_hybrid_forward_ntt_qp");

    for (std::size_t m = 1, gap = degree >> 1;
         remaining_stages > 0;)
    {
        std::size_t stage_count = remaining_stages % kFusionStages;
        if (stage_count == 0)
        {
            stage_count = kFusionStages;
        }

        const std::size_t tiles_per_limb = degree >> stage_count;
        const std::size_t total_tiles = active_limb_count * tiles_per_limb;
        const int grid_size = static_cast<int>(
            (total_tiles + block_size - 1) / block_size);

        if (stage_count == 1)
        {
            hybrid_forward_ntt_modup_qp_stage_kernel<<<grid_size, block_size>>>(
                modup_q,
                modup_p,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.ntt_tables.data(),
                decomp_limb_begin,
                decomp_limb_count,
                base_q_size,
                base_p_size,
                degree,
                m,
                gap);
        }
        else if (stage_count == 2)
        {
            hybrid_forward_ntt_modup_qp_fused_stage_kernel<2>
                <<<grid_size, block_size>>>(
                    modup_q,
                    modup_p,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    parameter_shard.ntt_tables.data(),
                    decomp_limb_begin,
                    decomp_limb_count,
                    base_q_size,
                    base_p_size,
                    degree,
                    m,
                    gap);
        }
        else
        {
            hybrid_forward_ntt_modup_qp_fused_stage_kernel<3>
                <<<grid_size, block_size>>>(
                    modup_q,
                    modup_p,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    parameter_shard.ntt_tables.data(),
                    decomp_limb_begin,
                    decomp_limb_count,
                    base_q_size,
                    base_p_size,
                    degree,
                    m,
                    gap);
        }
        gpu_check_cuda(
            cudaGetLastError(),
            "launch_hybrid_forward_ntt_qp stage kernel launch");

        m <<= stage_count;
        gap >>= stage_count;
        remaining_stages -= stage_count;
    }
}

void launch_hybrid_forward_ntt_qp_prepare_final_tail(
    GpuWord *modup_q,
    GpuWord *modup_p,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_forward_ntt_qp_prepare_final_tail",
        parameter_shard,
        degree);

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    if (modup_q == nullptr || modup_p == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_prepare_final_tail: null data pointer");
    }
    if (parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_prepare_final_tail: null NTT table pointer");
    }
    if (decomp_limb_count == 0 ||
        decomp_limb_count > base_p_size ||
        decomp_limb_begin + decomp_limb_count > base_q_size)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_prepare_final_tail: invalid decomposition range");
    }
    if (parameter_shard.limb_begin != 0 ||
        parameter_shard.limb_count < base_q_size + base_p_size)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_prepare_final_tail: parameter shard must cover full QP limb range");
    }

    const std::size_t degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_forward_ntt_qp_prepare_final_tail");
    constexpr std::size_t kFinalStages = 3;
    if (degree_power <= kFinalStages)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_prepare_final_tail: degree is too small");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_forward_ntt_qp_prepare_final_tail cudaSetDevice");

    const std::size_t active_limb_count =
        base_q_size - decomp_limb_count + base_p_size;
    constexpr std::size_t kFusionStages = 3;
    constexpr int block_size = 256;
    std::size_t remaining_stages = degree_power - 1;
    std::size_t m = 2;
    std::size_t gap = degree >> 2;

    while (remaining_stages > kFinalStages)
    {
        std::size_t stage_count = remaining_stages % kFusionStages;
        if (stage_count == 0)
        {
            stage_count = kFusionStages;
        }

        const std::size_t tiles_per_limb = degree >> stage_count;
        const std::size_t total_tiles = active_limb_count * tiles_per_limb;
        const int grid_size = static_cast<int>(
            (total_tiles + block_size - 1) / block_size);

        if (stage_count == 1)
        {
            hybrid_forward_ntt_modup_qp_stage_kernel<<<grid_size, block_size>>>(
                modup_q,
                modup_p,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.ntt_tables.data(),
                decomp_limb_begin,
                decomp_limb_count,
                base_q_size,
                base_p_size,
                degree,
                m,
                gap);
        }
        else if (stage_count == 2)
        {
            hybrid_forward_ntt_modup_qp_fused_stage_kernel<2>
                <<<grid_size, block_size>>>(
                    modup_q,
                    modup_p,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    parameter_shard.ntt_tables.data(),
                    decomp_limb_begin,
                    decomp_limb_count,
                    base_q_size,
                    base_p_size,
                    degree,
                    m,
                    gap);
        }
        else
        {
            hybrid_forward_ntt_modup_qp_fused_stage_kernel<3>
                <<<grid_size, block_size>>>(
                    modup_q,
                    modup_p,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    parameter_shard.ntt_tables.data(),
                    decomp_limb_begin,
                    decomp_limb_count,
                    base_q_size,
                    base_p_size,
                    degree,
                    m,
                    gap);
        }
        gpu_check_cuda(
            cudaGetLastError(),
            "launch_hybrid_forward_ntt_qp_prepare_final_tail stage kernel launch");

        m <<= stage_count;
        gap >>= stage_count;
        remaining_stages -= stage_count;
    }
}
#endif

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
    bool skip_first_ntt_stage)
{
    validate_hybrid_tables(
        "launch_hybrid_forward_ntt_qp_mul_accumulate_two_components",
        parameter_shard,
        degree);

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    if (accum_q0 == nullptr || accum_p0 == nullptr ||
        accum_q1 == nullptr || accum_p1 == nullptr ||
        modup_q == nullptr || modup_p == nullptr ||
        c2_ntt == nullptr || key_q0 == nullptr || key_p0 == nullptr ||
        key_q1 == nullptr || key_p1 == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_mul_accumulate_two_components: null data pointer");
    }
    if (parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_mul_accumulate_two_components: null NTT table pointer");
    }
    if (decomp_limb_count == 0 ||
        decomp_limb_count > base_p_size ||
        decomp_limb_begin + decomp_limb_count > base_q_size)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_mul_accumulate_two_components: invalid decomposition range");
    }
    if (parameter_shard.limb_begin != 0 ||
        parameter_shard.limb_count < base_q_size + base_p_size)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_mul_accumulate_two_components: parameter shard must cover full QP limb range");
    }
    if (parameter_shard.ntt_tables.size() <
        (base_q_size + base_p_size) * degree)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_mul_accumulate_two_components: NTT tables are too small");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_forward_ntt_qp_mul_accumulate_two_components cudaSetDevice");

    const std::size_t active_limb_count =
        base_q_size - decomp_limb_count + base_p_size;
    constexpr int kFusionStages = 3;
    constexpr int block_size = 256;
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_forward_ntt_qp_mul_accumulate_two_components");
    std::size_t remaining_stages = degree_power;
    std::size_t first_m = 1;
    std::size_t first_gap = degree >> 1;
    if (skip_first_ntt_stage)
    {
        if (degree_power < 2)
        {
            throw std::invalid_argument(
                "launch_hybrid_forward_ntt_qp_mul_accumulate_two_components: cannot skip the first NTT stage for degree below four");
        }
        --remaining_stages;
        first_m <<= 1;
        first_gap >>= 1;
    }

    for (std::size_t m = first_m, gap = first_gap;
         remaining_stages > 0;)
    {
        std::size_t stage_count = remaining_stages % kFusionStages;
        if (stage_count == 0)
        {
            stage_count = kFusionStages;
        }

        const std::size_t tiles_per_limb = degree >> stage_count;
        const std::size_t total_tiles = active_limb_count * tiles_per_limb;
        const bool final_stage = remaining_stages == stage_count;
        const std::size_t total_tasks = final_stage && fuse_decomp_q
            ? total_tiles + decomp_limb_count * degree
            : total_tiles;
        const int grid_size = static_cast<int>(
            (total_tasks + block_size - 1) / block_size);

        if (final_stage)
        {
            if (stage_count == 1)
            {
                if (fuse_decomp_q)
                {
                    hybrid_forward_ntt_modup_qp_final_mul_accumulate_fused_decomp_q_kernel<1>
                        <<<grid_size, block_size>>>(
                            accum_q0,
                            accum_p0,
                            accum_q1,
                            accum_p1,
                            modup_q,
                            modup_p,
                            c2_ntt,
                            parameter_shard.rns_primes.data(),
                            parameter_shard.rns_modulus_constants.data(),
                            parameter_shard.ntt_tables.data(),
                            key_q0,
                            key_p0,
                            key_q1,
                            key_p1,
                            decomp_limb_begin,
                            decomp_limb_count,
                            base_q_size,
                            base_p_size,
                            degree,
                            degree_power,
                            m,
                            gap,
                            overwrite_accum);
                }
                else
                {
                    hybrid_forward_ntt_modup_qp_final_mul_accumulate_kernel<1>
                        <<<grid_size, block_size>>>(
                            accum_q0,
                            accum_p0,
                            accum_q1,
                            accum_p1,
                            modup_q,
                            modup_p,
                            parameter_shard.rns_primes.data(),
                            parameter_shard.rns_modulus_constants.data(),
                            parameter_shard.ntt_tables.data(),
                            key_q0,
                            key_p0,
                            key_q1,
                            key_p1,
                            decomp_limb_begin,
                            decomp_limb_count,
                            base_q_size,
                            base_p_size,
                            degree,
                            m,
                            gap,
                            overwrite_accum);
                }
            }
            else if (stage_count == 2)
            {
                if (fuse_decomp_q)
                {
                    hybrid_forward_ntt_modup_qp_final_mul_accumulate_fused_decomp_q_kernel<2>
                        <<<grid_size, block_size>>>(
                            accum_q0,
                            accum_p0,
                            accum_q1,
                            accum_p1,
                            modup_q,
                            modup_p,
                            c2_ntt,
                            parameter_shard.rns_primes.data(),
                            parameter_shard.rns_modulus_constants.data(),
                            parameter_shard.ntt_tables.data(),
                            key_q0,
                            key_p0,
                            key_q1,
                            key_p1,
                            decomp_limb_begin,
                            decomp_limb_count,
                            base_q_size,
                            base_p_size,
                            degree,
                            degree_power,
                            m,
                            gap,
                            overwrite_accum);
                }
                else
                {
                    hybrid_forward_ntt_modup_qp_final_mul_accumulate_kernel<2>
                        <<<grid_size, block_size>>>(
                            accum_q0,
                            accum_p0,
                            accum_q1,
                            accum_p1,
                            modup_q,
                            modup_p,
                            parameter_shard.rns_primes.data(),
                            parameter_shard.rns_modulus_constants.data(),
                            parameter_shard.ntt_tables.data(),
                            key_q0,
                            key_p0,
                            key_q1,
                            key_p1,
                            decomp_limb_begin,
                            decomp_limb_count,
                            base_q_size,
                            base_p_size,
                            degree,
                            m,
                            gap,
                            overwrite_accum);
                }
            }
            else
            {
                if (fuse_decomp_q)
                {
                    hybrid_forward_ntt_modup_qp_final_mul_accumulate_fused_decomp_q_kernel<3>
                        <<<grid_size, block_size>>>(
                            accum_q0,
                            accum_p0,
                            accum_q1,
                            accum_p1,
                            modup_q,
                            modup_p,
                            c2_ntt,
                            parameter_shard.rns_primes.data(),
                            parameter_shard.rns_modulus_constants.data(),
                            parameter_shard.ntt_tables.data(),
                            key_q0,
                            key_p0,
                            key_q1,
                            key_p1,
                            decomp_limb_begin,
                            decomp_limb_count,
                            base_q_size,
                            base_p_size,
                            degree,
                            degree_power,
                            m,
                            gap,
                            overwrite_accum);
                }
                else
                {
                    hybrid_forward_ntt_modup_qp_final_mul_accumulate_kernel<3>
                        <<<grid_size, block_size>>>(
                            accum_q0,
                            accum_p0,
                            accum_q1,
                            accum_p1,
                            modup_q,
                            modup_p,
                            parameter_shard.rns_primes.data(),
                            parameter_shard.rns_modulus_constants.data(),
                            parameter_shard.ntt_tables.data(),
                            key_q0,
                            key_p0,
                            key_q1,
                            key_p1,
                            decomp_limb_begin,
                            decomp_limb_count,
                            base_q_size,
                            base_p_size,
                            degree,
                            m,
                            gap,
                            overwrite_accum);
                }
            }
        }
        else if (stage_count == 1)
        {
            hybrid_forward_ntt_modup_qp_stage_kernel<<<grid_size, block_size>>>(
                modup_q,
                modup_p,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.ntt_tables.data(),
                decomp_limb_begin,
                decomp_limb_count,
                base_q_size,
                base_p_size,
                degree,
                m,
                gap);
        }
        else if (stage_count == 2)
        {
            hybrid_forward_ntt_modup_qp_fused_stage_kernel<2>
                <<<grid_size, block_size>>>(
                    modup_q,
                    modup_p,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    parameter_shard.ntt_tables.data(),
                    decomp_limb_begin,
                    decomp_limb_count,
                    base_q_size,
                    base_p_size,
                    degree,
                    m,
                    gap);
        }
        else
        {
            hybrid_forward_ntt_modup_qp_fused_stage_kernel<3>
                <<<grid_size, block_size>>>(
                    modup_q,
                    modup_p,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    parameter_shard.ntt_tables.data(),
                    decomp_limb_begin,
                    decomp_limb_count,
                    base_q_size,
                    base_p_size,
                    degree,
                    m,
                    gap);
        }
        gpu_check_cuda(
            cudaGetLastError(),
            "launch_hybrid_forward_ntt_qp_mul_accumulate_two_components stage kernel launch");

        m <<= stage_count;
        gap >>= stage_count;
        remaining_stages -= stage_count;
    }

    if (!fuse_decomp_q)
    {
        const std::size_t decomp_total = decomp_limb_count * degree;
        const int decomp_grid_size = static_cast<int>(
            (decomp_total + block_size - 1) / block_size);
        hybrid_decomp_q_multiply_accumulate_two_components_kernel
            <<<decomp_grid_size, block_size>>>(
                accum_q0,
                accum_q1,
                c2_ntt,
                key_q0,
                key_q1,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                degree,
                degree_power,
                decomp_limb_begin,
                decomp_limb_count,
                overwrite_accum);
        gpu_check_cuda(
            cudaGetLastError(),
            "launch_hybrid_forward_ntt_qp_mul_accumulate_two_components baseline decomp Q kernel launch");
    }
}

#if 0
/* Retired all-dnum PAccum launchers. */
void launch_hybrid_paccum_all_dnum_two_components(
    GpuWord *accum_q0,
    GpuWord *accum_p0,
    GpuWord *accum_q1,
    GpuWord *accum_p1,
    const GpuWord *all_modup_q,
    const GpuWord *all_modup_p,
    const GpuWord *c2_ntt,
    const GpuWord *const *key_qp0_by_dnum,
    const GpuWord *const *key_qp1_by_dnum,
    std::size_t decomp_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_paccum_all_dnum_two_components",
        parameter_shard,
        degree);
    if (accum_q0 == nullptr || accum_p0 == nullptr ||
        accum_q1 == nullptr || accum_p1 == nullptr ||
        all_modup_q == nullptr || all_modup_p == nullptr ||
        c2_ntt == nullptr ||
        key_qp0_by_dnum == nullptr || key_qp1_by_dnum == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_paccum_all_dnum_two_components: null data pointer");
    }
    if (decomp_count == 0 ||
        decomp_count > parameter_shard.hybrid_decomp_count)
    {
        throw std::invalid_argument(
            "launch_hybrid_paccum_all_dnum_two_components: invalid decomposition count");
    }
    if (parameter_shard.limb_begin != 0 ||
        parameter_shard.limb_count <
            parameter_shard.hybrid_base_q_count +
                parameter_shard.hybrid_base_p_count)
    {
        throw std::invalid_argument(
            "launch_hybrid_paccum_all_dnum_two_components: parameter shard must cover full QP limb range");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_paccum_all_dnum_two_components cudaSetDevice");

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    const std::size_t total = (base_q_size + base_p_size) * degree;
    if (total == 0)
    {
        return;
    }

    constexpr int block_size = 256;
    const int grid_size =
        static_cast<int>((total + block_size - 1) / block_size);
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_paccum_all_dnum_two_components");

    hybrid_paccum_all_dnum_two_components_kernel
        <<<grid_size, block_size>>>(
            accum_q0,
            accum_p0,
            accum_q1,
            accum_p1,
            all_modup_q,
            all_modup_p,
            c2_ntt,
            key_qp0_by_dnum,
            key_qp1_by_dnum,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            decomp_count,
            base_q_size,
            base_p_size,
            degree,
            degree_power);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_paccum_all_dnum_two_components kernel launch");
}

void launch_hybrid_final_ntt_paccum_all_dnum_two_components(
    GpuWord *accum_q0,
    GpuWord *accum_p0,
    GpuWord *accum_q1,
    GpuWord *accum_p1,
    const GpuWord *all_modup_q,
    const GpuWord *all_modup_p,
    const GpuWord *c2_ntt,
    const GpuWord *const *key_qp0_by_dnum,
    const GpuWord *const *key_qp1_by_dnum,
    std::size_t decomp_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_final_ntt_paccum_all_dnum_two_components",
        parameter_shard,
        degree);
    if (accum_q0 == nullptr || accum_p0 == nullptr ||
        accum_q1 == nullptr || accum_p1 == nullptr ||
        all_modup_q == nullptr || all_modup_p == nullptr ||
        c2_ntt == nullptr ||
        key_qp0_by_dnum == nullptr || key_qp1_by_dnum == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_final_ntt_paccum_all_dnum_two_components: null data pointer");
    }
    if (decomp_count == 0 ||
        decomp_count > parameter_shard.hybrid_decomp_count)
    {
        throw std::invalid_argument(
            "launch_hybrid_final_ntt_paccum_all_dnum_two_components: invalid decomposition count");
    }
    if (parameter_shard.limb_begin != 0 ||
        parameter_shard.limb_count <
            parameter_shard.hybrid_base_q_count +
                parameter_shard.hybrid_base_p_count)
    {
        throw std::invalid_argument(
            "launch_hybrid_final_ntt_paccum_all_dnum_two_components: parameter shard must cover full QP limb range");
    }

    constexpr std::size_t kFinalStages = 3;
    const std::size_t degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_final_ntt_paccum_all_dnum_two_components");
    if (degree_power <= kFinalStages)
    {
        throw std::invalid_argument(
            "launch_hybrid_final_ntt_paccum_all_dnum_two_components: degree is too small");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_final_ntt_paccum_all_dnum_two_components cudaSetDevice");

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    const std::size_t total_tiles =
        (base_q_size + base_p_size) * (degree >> kFinalStages);
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total_tiles + block_size - 1) / block_size);

    hybrid_final_ntt_paccum_all_dnum_two_components_kernel
        <<<grid_size, block_size>>>(
            accum_q0,
            accum_p0,
            accum_q1,
            accum_p1,
            all_modup_q,
            all_modup_p,
            c2_ntt,
            key_qp0_by_dnum,
            key_qp1_by_dnum,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.ntt_tables.data(),
            decomp_count,
            base_q_size,
            base_p_size,
            degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_final_ntt_paccum_all_dnum_two_components kernel launch");
}
#endif

void launch_hybrid_forward_ntt_qp_active(
    GpuWord *modup_q,
    GpuWord *modup_p,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_forward_ntt_qp_active",
        parameter_shard,
        degree);
    const std::size_t base_q_size =
        parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size =
        parameter_shard.hybrid_base_p_count;
    if (modup_q == nullptr || modup_p == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_active: null data pointer");
    }
    if (decomp_limb_count == 0 ||
        decomp_limb_count > base_p_size ||
        decomp_limb_begin + decomp_limb_count > base_q_size)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_active: invalid decomposition range");
    }
    if (parameter_shard.ntt_tables.size() <
        (base_q_size + base_p_size) * degree)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_qp_active: incomplete NTT tables");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_forward_ntt_qp_active cudaSetDevice");
    const std::size_t active_limb_count =
        base_q_size - decomp_limb_count + base_p_size;
    constexpr int kFusionStages = 3;
    constexpr int block_size = 256;
    std::size_t remaining_stages = checked_log2_degree(
        degree,
        "launch_hybrid_forward_ntt_qp_active");

    for (std::size_t m = 1, gap = degree >> 1;
         remaining_stages > 0;)
    {
        std::size_t stage_count =
            remaining_stages % kFusionStages;
        if (stage_count == 0)
        {
            stage_count = kFusionStages;
        }
        const std::size_t tiles_per_limb =
            degree >> stage_count;
        const std::size_t total_tiles =
            active_limb_count * tiles_per_limb;
        const int grid_size = static_cast<int>(
            (total_tiles + block_size - 1) / block_size);

        if (stage_count == 1)
        {
            hybrid_forward_ntt_modup_qp_stage_kernel
                <<<grid_size, block_size>>>(
                    modup_q,
                    modup_p,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    parameter_shard.ntt_tables.data(),
                    decomp_limb_begin,
                    decomp_limb_count,
                    base_q_size,
                    base_p_size,
                    degree,
                    m,
                    gap);
        }
        else if (stage_count == 2)
        {
            hybrid_forward_ntt_modup_qp_fused_stage_kernel<2>
                <<<grid_size, block_size>>>(
                    modup_q,
                    modup_p,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    parameter_shard.ntt_tables.data(),
                    decomp_limb_begin,
                    decomp_limb_count,
                    base_q_size,
                    base_p_size,
                    degree,
                    m,
                    gap);
        }
        else
        {
            hybrid_forward_ntt_modup_qp_fused_stage_kernel<3>
                <<<grid_size, block_size>>>(
                    modup_q,
                    modup_p,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    parameter_shard.ntt_tables.data(),
                    decomp_limb_begin,
                    decomp_limb_count,
                    base_q_size,
                    base_p_size,
                    degree,
                    m,
                    gap);
        }
        gpu_check_cuda(
            cudaGetLastError(),
            "launch_hybrid_forward_ntt_qp_active kernel launch");
        m <<= stage_count;
        gap >>= stage_count;
        remaining_stages -= stage_count;
    }
}

void launch_hybrid_multiply_accumulate(
    GpuWord *accum_q,
    GpuWord *accum_p,
    const GpuWord *modup_q,
    const GpuWord *modup_p,
    const GpuWord *key_qp,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_multiply_accumulate",
        parameter_shard,
        degree);
    if (accum_q == nullptr || accum_p == nullptr ||
        modup_q == nullptr || modup_p == nullptr || key_qp == nullptr)
    {
        throw std::invalid_argument("launch_hybrid_multiply_accumulate: null data pointer");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_multiply_accumulate cudaSetDevice");

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    const std::size_t total = (base_q_size + base_p_size) * degree;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_multiply_accumulate");

    hybrid_multiply_accumulate_kernel<<<grid_size, block_size>>>(
        accum_q,
        accum_p,
        modup_q,
        modup_p,
        key_qp,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        base_q_size,
        base_p_size,
        degree,
        degree_power);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_multiply_accumulate kernel launch");
}

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
    bool overwrite_accum)
{
    validate_hybrid_tables(
        "launch_hybrid_multiply_accumulate_two_components",
        parameter_shard,
        degree);
    if (accum_q0 == nullptr || accum_p0 == nullptr ||
        accum_q1 == nullptr || accum_p1 == nullptr ||
        modup_q == nullptr || modup_p == nullptr || c2_ntt == nullptr ||
        key_q0 == nullptr || key_p0 == nullptr ||
        key_q1 == nullptr || key_p1 == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_multiply_accumulate_two_components: null data pointer");
    }
    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    if (decomp_limb_count == 0 ||
        decomp_limb_count > base_p_size ||
        decomp_limb_begin + decomp_limb_count > base_q_size)
    {
        throw std::invalid_argument(
            "launch_hybrid_multiply_accumulate_two_components: invalid decomposition range");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_multiply_accumulate_two_components cudaSetDevice");

    const std::size_t total = (base_q_size + base_p_size) * degree;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_multiply_accumulate_two_components");

    hybrid_multiply_accumulate_two_components_kernel<<<grid_size, block_size>>>(
        accum_q0,
        accum_p0,
        accum_q1,
        accum_p1,
        modup_q,
        modup_p,
        c2_ntt,
        key_q0,
        key_p0,
        key_q1,
        key_p1,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        base_q_size,
        base_p_size,
        degree,
        degree_power,
        decomp_limb_begin,
        decomp_limb_count,
        overwrite_accum);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_multiply_accumulate_two_components kernel launch");
}

void launch_hybrid_moddown(
    GpuWord *accum_q,
    const GpuWord *accum_p,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_moddown",
        parameter_shard,
        degree);
    if (accum_q == nullptr || accum_p == nullptr)
    {
        throw std::invalid_argument("launch_hybrid_moddown: null data pointer");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_moddown cudaSetDevice");

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    const std::size_t total = base_q_size * degree;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_moddown");

    hybrid_moddown_kernel<<<grid_size, block_size>>>(
        accum_q,
        accum_p,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.hybrid_moddown_p_to_q_matrix.data(),
        parameter_shard.hybrid_p_inv_punctured.data(),
        parameter_shard.hybrid_inv_p_mod_q.data(),
        base_q_size,
        base_p_size,
        degree,
        degree_power);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_moddown kernel launch");
}

void launch_hybrid_convert_p_to_q(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_convert_p_to_q",
        parameter_shard,
        degree);
    if (converted_q0 == nullptr || converted_q1 == nullptr ||
        accum_p0 == nullptr || accum_p1 == nullptr)
    {
        throw std::invalid_argument("launch_hybrid_convert_p_to_q: null data pointer");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_convert_p_to_q cudaSetDevice");

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    const std::size_t total = base_q_size * degree;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_convert_p_to_q");

    hybrid_convert_p_to_q_two_components_kernel<<<grid_size, block_size>>>(
        converted_q0,
        converted_q1,
        accum_p0,
        accum_p1,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.hybrid_moddown_p_to_q_matrix.data(),
        parameter_shard.hybrid_p_inv_punctured.data(),
        base_q_size,
        base_p_size,
        degree,
        degree_power);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_convert_p_to_q kernel launch");
}

void launch_hybrid_convert_p_to_q_forward_ntt(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool source_preweighted)
{
    validate_hybrid_tables(
        "launch_hybrid_convert_p_to_q_forward_ntt",
        parameter_shard,
        degree);
    if (converted_q0 == nullptr || converted_q1 == nullptr ||
        accum_p0 == nullptr || accum_p1 == nullptr ||
        parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_convert_p_to_q_forward_ntt: null data pointer");
    }
    if (degree < 2)
    {
        throw std::invalid_argument(
            "launch_hybrid_convert_p_to_q_forward_ntt: degree must be at least two");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_convert_p_to_q_forward_ntt cudaSetDevice");

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_convert_p_to_q_forward_ntt");
    constexpr int block_size = 256;
    const std::size_t first_stage_total = base_q_size * (degree >> 1);
    const int first_stage_grid = static_cast<int>(
        (first_stage_total + block_size - 1) / block_size);

    if (base_p_size == 2 && source_preweighted)
    {
        hybrid_convert_p_to_q_forward_ntt_first_stage_kernel<2, true>
            <<<first_stage_grid, block_size>>>(
                converted_q0,
                converted_q1,
                accum_p0,
                accum_p1,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.ntt_tables.data(),
                parameter_shard.hybrid_moddown_p_to_q_matrix.data(),
                parameter_shard.hybrid_p_inv_punctured.data(),
                base_q_size,
                base_p_size,
                degree,
                degree_power);
    }
    else if (base_p_size == 2)
    {
        hybrid_convert_p_to_q_forward_ntt_first_stage_kernel<2, false>
            <<<first_stage_grid, block_size>>>(
                converted_q0,
                converted_q1,
                accum_p0,
                accum_p1,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.ntt_tables.data(),
                parameter_shard.hybrid_moddown_p_to_q_matrix.data(),
                parameter_shard.hybrid_p_inv_punctured.data(),
                base_q_size,
                base_p_size,
                degree,
                degree_power);
    }
    else if (base_p_size == 9 && source_preweighted)
    {
        hybrid_convert_p_to_q_forward_ntt_first_stage_kernel<9, true>
            <<<first_stage_grid, block_size>>>(
                converted_q0,
                converted_q1,
                accum_p0,
                accum_p1,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.ntt_tables.data(),
                parameter_shard.hybrid_moddown_p_to_q_matrix.data(),
                parameter_shard.hybrid_p_inv_punctured.data(),
                base_q_size,
                base_p_size,
                degree,
                degree_power);
    }
    else if (base_p_size == 9)
    {
        hybrid_convert_p_to_q_forward_ntt_first_stage_kernel<9, false>
            <<<first_stage_grid, block_size>>>(
                converted_q0,
                converted_q1,
                accum_p0,
                accum_p1,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.ntt_tables.data(),
                parameter_shard.hybrid_moddown_p_to_q_matrix.data(),
                parameter_shard.hybrid_p_inv_punctured.data(),
                base_q_size,
                base_p_size,
                degree,
                degree_power);
    }
    else if (source_preweighted)
    {
        hybrid_convert_p_to_q_forward_ntt_first_stage_kernel<0, true>
            <<<first_stage_grid, block_size>>>(
                converted_q0,
                converted_q1,
                accum_p0,
                accum_p1,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.ntt_tables.data(),
                parameter_shard.hybrid_moddown_p_to_q_matrix.data(),
                parameter_shard.hybrid_p_inv_punctured.data(),
                base_q_size,
                base_p_size,
                degree,
                degree_power);
    }
    else
    {
        hybrid_convert_p_to_q_forward_ntt_first_stage_kernel<0, false>
            <<<first_stage_grid, block_size>>>(
                converted_q0,
                converted_q1,
                accum_p0,
                accum_p1,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.ntt_tables.data(),
                parameter_shard.hybrid_moddown_p_to_q_matrix.data(),
                parameter_shard.hybrid_p_inv_punctured.data(),
                base_q_size,
                base_p_size,
                degree,
                degree_power);
    }
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_convert_p_to_q_forward_ntt first-stage kernel launch");

    launch_hybrid_forward_ntt_q_two_components_stages(
        converted_q0,
        converted_q1,
        parameter_shard,
        degree,
        degree_power,
        1,
        "launch_hybrid_convert_p_to_q_forward_ntt remaining-stage kernel launch");
}

void launch_hybrid_preweight_p_two_components(
    GpuWord *accum_p0,
    GpuWord *accum_p1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_preweight_p_two_components",
        parameter_shard,
        degree);
    if (accum_p0 == nullptr || accum_p1 == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_preweight_p_two_components: null data pointer");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_preweight_p_two_components cudaSetDevice");

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t base_p_size = parameter_shard.hybrid_base_p_count;
    const std::size_t total = base_p_size * degree;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total + block_size - 1) / block_size);
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_preweight_p_two_components");

    hybrid_preweight_p_two_components_kernel<<<grid_size, block_size>>>(
        accum_p0,
        accum_p1,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.hybrid_p_inv_punctured.data(),
        base_q_size,
        base_p_size,
        degree,
        degree_power);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_preweight_p_two_components kernel launch");
}

void launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    const GpuWord *accum_p0,
    const GpuWord *accum_p1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool source_preweighted)
{
    validate_hybrid_tables(
        "launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8",
        parameter_shard,
        degree);
    if (converted_q0 == nullptr || converted_q1 == nullptr ||
        accum_p0 == nullptr || accum_p1 == nullptr ||
        parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8: null data pointer");
    }
    const std::size_t base_p_size =
        parameter_shard.hybrid_base_p_count;
    if (base_p_size != 2 && base_p_size != 9)
    {
        throw std::invalid_argument(
            "launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8: requires two or nine P limbs");
    }
    if (degree < 2)
    {
        throw std::invalid_argument(
            "launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8: degree must be at least two");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8 cudaSetDevice");

    constexpr std::size_t kCoefficientTile = 32;
    constexpr std::size_t kTargetRows = 8;
    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8");
    const std::size_t butterflies_per_limb = degree >> 1;
    const dim3 block_dim(kCoefficientTile, kTargetRows);
    const dim3 grid_dim(
        static_cast<unsigned int>(
            (butterflies_per_limb + kCoefficientTile - 1) /
            kCoefficientTile),
        static_cast<unsigned int>(
            (base_q_size + kTargetRows - 1) / kTargetRows));

    if (base_p_size == 2)
    {
        if (source_preweighted)
        {
            throw std::invalid_argument(
                "launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8: preweighted P=2 source is unsupported");
        }
        hybrid_convert_p_to_q_forward_ntt_first_stage_row_tiled8_kernel
            <<<grid_dim, block_dim>>>(
                converted_q0,
                converted_q1,
                accum_p0,
                accum_p1,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.ntt_tables.data(),
                parameter_shard.hybrid_moddown_p_to_q_matrix.data(),
                parameter_shard.hybrid_p_inv_punctured.data(),
                base_q_size,
                degree);
    }
    else if (source_preweighted)
    {
        hybrid_convert_p9_to_q_forward_ntt_first_stage_row_tiled8_kernel<true>
            <<<grid_dim, block_dim>>>(
                converted_q0,
                converted_q1,
                accum_p0,
                accum_p1,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.ntt_tables.data(),
                parameter_shard.hybrid_moddown_p_to_q_matrix.data(),
                parameter_shard.hybrid_p_inv_punctured.data(),
                base_q_size,
                degree);
    }
    else
    {
        hybrid_convert_p9_to_q_forward_ntt_first_stage_row_tiled8_kernel<false>
            <<<grid_dim, block_dim>>>(
                converted_q0,
                converted_q1,
                accum_p0,
                accum_p1,
                parameter_shard.rns_primes.data(),
                parameter_shard.rns_modulus_constants.data(),
                parameter_shard.ntt_tables.data(),
                parameter_shard.hybrid_moddown_p_to_q_matrix.data(),
                parameter_shard.hybrid_p_inv_punctured.data(),
                base_q_size,
                degree);
    }
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8 first-stage kernel launch");

    launch_hybrid_forward_ntt_q_two_components_stages(
        converted_q0,
        converted_q1,
        parameter_shard,
        degree,
        degree_power,
        1,
        "launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8 remaining-stage kernel launch");
}

void launch_hybrid_forward_ntt_q_two_components(
    GpuWord *values0,
    GpuWord *values1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_forward_ntt_q_two_components",
        parameter_shard,
        degree);
    if (values0 == nullptr || values1 == nullptr ||
        parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_forward_ntt_q_two_components: null data pointer");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_forward_ntt_q_two_components cudaSetDevice");
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_forward_ntt_q_two_components");
    launch_hybrid_forward_ntt_q_two_components_stages(
        values0,
        values1,
        parameter_shard,
        degree,
        degree_power,
        0,
        "launch_hybrid_forward_ntt_q_two_components stage kernel launch");
}

void launch_hybrid_apply_moddown_ntt(
    GpuWord *accum_q0,
    GpuWord *accum_q1,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_apply_moddown_ntt",
        parameter_shard,
        degree);
    if (accum_q0 == nullptr || accum_q1 == nullptr ||
        converted_q0 == nullptr || converted_q1 == nullptr)
    {
        throw std::invalid_argument("launch_hybrid_apply_moddown_ntt: null data pointer");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_apply_moddown_ntt cudaSetDevice");

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t total = base_q_size * degree;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_apply_moddown_ntt");

    hybrid_apply_moddown_ntt_two_components_kernel<<<grid_size, block_size>>>(
        accum_q0,
        accum_q1,
        converted_q0,
        converted_q1,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.hybrid_inv_p_mod_q.data(),
        base_q_size,
        degree,
        degree_power);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_apply_moddown_ntt kernel launch");
}

void launch_hybrid_apply_moddown_ntt_out_of_place(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    const GpuWord *source_q0,
    const GpuWord *source_q1,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_apply_moddown_ntt_out_of_place",
        parameter_shard,
        degree);
    if (destination_q0 == nullptr || destination_q1 == nullptr ||
        source_q0 == nullptr || source_q1 == nullptr ||
        converted_q0 == nullptr || converted_q1 == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_apply_moddown_ntt_out_of_place: null data pointer");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_apply_moddown_ntt_out_of_place cudaSetDevice");

    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    const std::size_t total = base_q_size * degree;
    constexpr int block_size = 256;
    const int grid_size =
        static_cast<int>((total + block_size - 1) / block_size);
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_apply_moddown_ntt_out_of_place");

    hybrid_apply_moddown_ntt_out_of_place_two_components_kernel
        <<<grid_size, block_size>>>(
            destination_q0,
            destination_q1,
            source_q0,
            source_q1,
            converted_q0,
            converted_q1,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.hybrid_inv_p_mod_q.data(),
            base_q_size,
            degree,
            degree_power);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_apply_moddown_ntt_out_of_place kernel launch");
}

void launch_hybrid_apply_moddown_ntt_out_of_place_batch(
    GpuWord *destination_q,
    const GpuWord *source_q,
    const GpuWord *converted_q,
    std::size_t batch_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_apply_moddown_ntt_out_of_place_batch",
        parameter_shard,
        degree);
    if (destination_q == nullptr || source_q == nullptr ||
        converted_q == nullptr || batch_count == 0)
    {
        throw std::invalid_argument(
            "launch_hybrid_apply_moddown_ntt_out_of_place_batch: invalid argument");
    }
    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_apply_moddown_ntt_out_of_place_batch cudaSetDevice");

    const std::size_t base_q_size =
        parameter_shard.hybrid_base_q_count;
    const std::size_t total =
        batch_count * 2 * base_q_size * degree;
    constexpr int block_size = 256;
    const int grid_size =
        static_cast<int>((total + block_size - 1) / block_size);
    hybrid_apply_moddown_ntt_out_of_place_batch_kernel
        <<<grid_size, block_size>>>(
            destination_q,
            source_q,
            converted_q,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.hybrid_inv_p_mod_q.data(),
            batch_count,
            base_q_size,
            degree,
            checked_log2_degree(
                degree,
                "launch_hybrid_apply_moddown_ntt_out_of_place_batch"));
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_apply_moddown_ntt_out_of_place_batch kernel launch");
}

void launch_hybrid_apply_moddown_ntt_from_q_groups(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    const GpuWord *group_q,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    std::size_t group_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_hybrid_tables(
        "launch_hybrid_apply_moddown_ntt_from_q_groups",
        parameter_shard,
        degree);
    if (destination_q0 == nullptr || destination_q1 == nullptr ||
        group_q == nullptr || converted_q0 == nullptr ||
        converted_q1 == nullptr || group_count == 0)
    {
        throw std::invalid_argument(
            "launch_hybrid_apply_moddown_ntt_from_q_groups: "
            "invalid argument");
    }
    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_hybrid_apply_moddown_ntt_from_q_groups cudaSetDevice");
    constexpr int block_size = 256;
    const std::size_t total =
        parameter_shard.hybrid_base_q_count * degree;
    const int grid_size =
        static_cast<int>((total + block_size - 1) / block_size);
    hybrid_apply_moddown_ntt_from_q_groups_kernel
        <<<grid_size, block_size>>>(
            destination_q0,
            destination_q1,
            group_q,
            converted_q0,
            converted_q1,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.hybrid_inv_p_mod_q.data(),
            group_count,
            parameter_shard.hybrid_base_q_count,
            degree,
            checked_log2_degree(
                degree,
                "launch_hybrid_apply_moddown_ntt_from_q_groups"));
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_apply_moddown_ntt_from_q_groups kernel launch");
}

void launch_hybrid_apply_moddown_ntt_add_back(
    const GpuPolyShardView &destination_shard0,
    const GpuPolyShardView &destination_shard1,
    const GpuWord *accum_q0,
    const GpuWord *accum_q1,
    const GpuWord *converted_q0,
    const GpuWord *converted_q1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    const GpuConstPolyShardView *add_source_shard0,
    const GpuConstPolyShardView *add_source_shard1)
{
    validate_hybrid_tables(
        "launch_hybrid_apply_moddown_ntt_add_back",
        parameter_shard,
        degree);
    if (destination_shard0.ptr == nullptr ||
        destination_shard1.ptr == nullptr ||
        accum_q0 == nullptr || accum_q1 == nullptr ||
        converted_q0 == nullptr || converted_q1 == nullptr)
    {
        throw std::invalid_argument(
            "launch_hybrid_apply_moddown_ntt_add_back: null data pointer");
    }
    if (destination_shard0.device_id != destination_shard1.device_id ||
        destination_shard0.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument(
            "launch_hybrid_apply_moddown_ntt_add_back: device mismatch");
    }
    if (degree == 0 ||
        destination_shard0.limb_count == 0 ||
        destination_shard0.coeff_count != degree)
    {
        throw std::invalid_argument(
            "launch_hybrid_apply_moddown_ntt_add_back: invalid shard shape");
    }
    if (destination_shard0.limb_begin != destination_shard1.limb_begin ||
        destination_shard0.limb_count != destination_shard1.limb_count ||
        destination_shard0.coeff_begin != destination_shard1.coeff_begin ||
        destination_shard0.coeff_count != destination_shard1.coeff_count)
    {
        throw std::invalid_argument(
            "launch_hybrid_apply_moddown_ntt_add_back: shard shape mismatch");
    }
    if ((add_source_shard0 == nullptr) != (add_source_shard1 == nullptr))
    {
        throw std::invalid_argument(
            "launch_hybrid_apply_moddown_ntt_add_back: both add-source shards are required");
    }
    if (add_source_shard0 != nullptr)
    {
        if (add_source_shard0->ptr == nullptr ||
            add_source_shard1->ptr == nullptr)
        {
            throw std::invalid_argument(
                "launch_hybrid_apply_moddown_ntt_add_back: null add-source pointer");
        }
        if (add_source_shard0->device_id != destination_shard0.device_id ||
            add_source_shard1->device_id != destination_shard1.device_id ||
            add_source_shard0->limb_begin != destination_shard0.limb_begin ||
            add_source_shard1->limb_begin != destination_shard1.limb_begin ||
            add_source_shard0->limb_count != destination_shard0.limb_count ||
            add_source_shard1->limb_count != destination_shard1.limb_count ||
            add_source_shard0->coeff_begin != destination_shard0.coeff_begin ||
            add_source_shard1->coeff_begin != destination_shard1.coeff_begin ||
            add_source_shard0->coeff_count != destination_shard0.coeff_count ||
            add_source_shard1->coeff_count != destination_shard1.coeff_count)
        {
            throw std::invalid_argument(
                "launch_hybrid_apply_moddown_ntt_add_back: add-source shard placement mismatch");
        }
    }
    if (destination_shard0.coeff_begin != 0)
    {
        throw std::invalid_argument(
            "launch_hybrid_apply_moddown_ntt_add_back: non-zero coeff offset is not supported");
    }
    if (destination_shard0.limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument(
            "launch_hybrid_apply_moddown_ntt_add_back: parameter shard does not cover limb range");
    }

    const std::size_t modulus_offset =
        destination_shard0.limb_begin - parameter_shard.limb_begin;
    const std::size_t base_q_size = parameter_shard.hybrid_base_q_count;
    if (modulus_offset + destination_shard0.limb_count > base_q_size)
    {
        throw std::invalid_argument(
            "launch_hybrid_apply_moddown_ntt_add_back: Q limb range exceeds HYBRID base");
    }

    const std::size_t values_per_component =
        destination_shard0.limb_count * destination_shard0.coeff_count;
    const std::size_t total = values_per_component * 2;
    if (total == 0)
    {
        return;
    }

    gpu_check_cuda(
        cudaSetDevice(destination_shard0.device_id),
        "launch_hybrid_apply_moddown_ntt_add_back cudaSetDevice");

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    const unsigned int degree_power = checked_log2_degree(
        degree,
        "launch_hybrid_apply_moddown_ntt_add_back");
    hybrid_apply_moddown_ntt_add_back_two_components_kernel
        <<<grid_size, block_size>>>(
            destination_shard0.ptr,
            destination_shard1.ptr,
            add_source_shard0 == nullptr
                ? destination_shard0.ptr
                : add_source_shard0->ptr,
            add_source_shard1 == nullptr
                ? destination_shard1.ptr
                : add_source_shard1->ptr,
            accum_q0,
            accum_q1,
            converted_q0,
            converted_q1,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.hybrid_inv_p_mod_q.data(),
            modulus_offset,
            destination_shard0.limb_count,
            destination_shard0.coeff_count,
            degree_power);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_apply_moddown_ntt_add_back kernel launch");
}

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
    const GpuConstPolyShardView *add_source_shard0,
    const GpuConstPolyShardView *add_source_shard1)
{
    constexpr const char *name =
        "launch_hybrid_apply_moddown_ntt_add_back_rescale_x2";
    validate_hybrid_tables(name, parameter_shard, degree);
    if (destination_shard0.ptr == nullptr ||
        destination_shard1.ptr == nullptr ||
        correction_ntt_shard0.ptr == nullptr ||
        correction_ntt_shard1.ptr == nullptr ||
        accum_q0 == nullptr || accum_q1 == nullptr ||
        converted_q0 == nullptr || converted_q1 == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null data pointer");
    }
    if (destination_shard0.device_id != destination_shard1.device_id ||
        destination_shard0.device_id != correction_ntt_shard0.device_id ||
        destination_shard0.device_id != correction_ntt_shard1.device_id ||
        destination_shard0.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument(std::string(name) + ": device mismatch");
    }
    if (degree == 0 ||
        destination_shard0.limb_begin != 0 ||
        destination_shard1.limb_begin != 0 ||
        correction_ntt_shard0.limb_begin != 0 ||
        correction_ntt_shard1.limb_begin != 0 ||
        destination_shard0.coeff_begin != 0 ||
        destination_shard1.coeff_begin != 0 ||
        correction_ntt_shard0.coeff_begin != 0 ||
        correction_ntt_shard1.coeff_begin != 0 ||
        destination_shard0.coeff_count != degree ||
        destination_shard1.coeff_count != degree ||
        correction_ntt_shard0.coeff_count != degree ||
        correction_ntt_shard1.coeff_count != degree ||
        destination_shard0.limb_count == 0 ||
        destination_shard0.limb_count != destination_shard1.limb_count ||
        destination_shard0.limb_count != correction_ntt_shard0.limb_count ||
        destination_shard0.limb_count != correction_ntt_shard1.limb_count)
    {
        throw std::invalid_argument(std::string(name) + ": invalid shard shape");
    }
    if (add_source_shard0 == nullptr || add_source_shard1 == nullptr ||
        add_source_shard0->ptr == nullptr || add_source_shard1->ptr == nullptr)
    {
        throw std::invalid_argument(
            std::string(name) + ": add-source shards are required");
    }
    if (add_source_shard0->device_id != destination_shard0.device_id ||
        add_source_shard1->device_id != destination_shard1.device_id ||
        add_source_shard0->limb_begin != 0 ||
        add_source_shard1->limb_begin != 0 ||
        add_source_shard0->limb_count < destination_shard0.limb_count ||
        add_source_shard1->limb_count < destination_shard1.limb_count ||
        add_source_shard0->coeff_begin != 0 ||
        add_source_shard1->coeff_begin != 0 ||
        add_source_shard0->coeff_count != degree ||
        add_source_shard1->coeff_count != degree)
    {
        throw std::invalid_argument(
            std::string(name) + ": add-source shard placement mismatch");
    }
    const std::size_t destination_q_count = destination_shard0.limb_count;
    if (parameter_shard.limb_begin != 0 ||
        destination_q_count + 2 != parameter_shard.hybrid_base_q_count ||
        parameter_shard.inv_q_last_two_product_mod_q.size() <
            destination_q_count)
    {
        throw std::invalid_argument(
            std::string(name) + ": rescale_x2 constants mismatch");
    }

    const std::size_t values_per_component =
        destination_q_count * degree;
    const std::size_t total = values_per_component * 2;
    if (total == 0)
    {
        return;
    }

    gpu_check_cuda(
        cudaSetDevice(destination_shard0.device_id),
        "launch_hybrid_apply_moddown_ntt_add_back_rescale_x2 cudaSetDevice");

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total + block_size - 1) / block_size);
    hybrid_apply_moddown_ntt_add_back_rescale_x2_two_components_kernel
        <<<grid_size, block_size>>>(
            destination_shard0.ptr,
            destination_shard1.ptr,
            add_source_shard0->ptr,
            add_source_shard1->ptr,
            correction_ntt_shard0.ptr,
            correction_ntt_shard1.ptr,
            accum_q0,
            accum_q1,
            converted_q0,
            converted_q1,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.hybrid_inv_p_mod_q.data(),
            parameter_shard.inv_q_last_two_product_mod_q.data(),
            destination_q_count,
            degree,
            checked_log2_degree(degree, name));
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_apply_moddown_ntt_add_back_rescale_x2 kernel launch");
}

void launch_apply_galois_ntt_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    std::uint32_t galois_elt,
    std::size_t degree)
{
    if (destination_shard.ptr == nullptr || source_shard.ptr == nullptr)
    {
        throw std::invalid_argument("launch_apply_galois_ntt_poly_shard: null data pointer");
    }
    if (destination_shard.ptr == source_shard.ptr)
    {
        throw std::invalid_argument(
            "launch_apply_galois_ntt_poly_shard: in-place permutation is not supported");
    }
    if (destination_shard.device_id != source_shard.device_id)
    {
        throw std::invalid_argument("launch_apply_galois_ntt_poly_shard: device mismatch");
    }
    if (destination_shard.limb_begin != source_shard.limb_begin ||
        destination_shard.limb_count != source_shard.limb_count ||
        destination_shard.coeff_begin != source_shard.coeff_begin ||
        destination_shard.coeff_count != source_shard.coeff_count)
    {
        throw std::invalid_argument("launch_apply_galois_ntt_poly_shard: shard shape mismatch");
    }
    if (destination_shard.coeff_begin != 0 ||
        destination_shard.coeff_count != degree ||
        destination_shard.limb_count == 0)
    {
        throw std::invalid_argument(
            "launch_apply_galois_ntt_poly_shard: only full coefficient shards are supported");
    }
    if (degree > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max() / 2))
    {
        throw std::overflow_error("launch_apply_galois_ntt_poly_shard: degree is too large");
    }

    const std::uint64_t m = static_cast<std::uint64_t>(degree) << 1;
    if ((galois_elt & 1U) == 0U || static_cast<std::uint64_t>(galois_elt) >= m)
    {
        throw std::invalid_argument("launch_apply_galois_ntt_poly_shard: invalid Galois element");
    }

    const unsigned int degree_power =
        checked_log2_degree(degree, "launch_apply_galois_ntt_poly_shard");

    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_apply_galois_ntt_poly_shard cudaSetDevice");

    const std::size_t total = destination_shard.limb_count * degree;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total + block_size - 1) / block_size);

    apply_galois_ntt_poly_shard_kernel<<<grid_size, block_size>>>(
        destination_shard.ptr,
        source_shard.ptr,
        galois_elt,
        destination_shard.limb_count,
        degree,
        degree_power);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_apply_galois_ntt_poly_shard kernel launch");
}

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
