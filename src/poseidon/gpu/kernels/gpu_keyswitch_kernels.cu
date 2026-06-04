#include "poseidon/gpu/kernels/gpu_keyswitch_kernels.h"

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
    std::size_t degree)
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
    const std::size_t coeff = tid % degree;

    /* 整体线程分为两部分，一部分用来计算模升的模数q,另一部分用来计算模升的模数p */
    if (tid < q_total)
    {
        const std::size_t target_q_limb = tid / degree;
        const std::size_t target_offset = target_q_limb * degree + coeff;
        const GpuWord target_modulus = rns_primes[target_q_limb];
        const GpuWide target_barrett =
            rns_modulus_constants[target_q_limb];

        /* 如果在dnum快内直接复制 */
        if (target_q_limb >= decomp_limb_begin &&
            target_q_limb < decomp_limb_begin + decomp_limb_count)
        {
            modup_q[target_offset] = c2_ntt[target_offset];
            return;
        }

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
        GpuWord sum = 0;
        /* 一般情况下的基转换 */
        for (std::size_t col = 0; col < decomp_limb_count; ++col)
        {
            const std::size_t source_q_limb = decomp_limb_begin + col;
            const GpuWord value =
                c2_coeff[source_q_limb * degree + coeff];
            const GpuWord source_modulus = rns_primes[source_q_limb];
            const GpuWide source_barrett =
                rns_modulus_constants[source_q_limb];
            const GpuWord inv_punctured =
                qi_inv_punctured[inv_offset + col];
            const GpuWord weighted = inv_punctured == 1
                ? value
                : mul_mod(value, inv_punctured, source_modulus, source_barrett);
            const GpuWord matrix_value =
                q_matrices[matrix_offset + target_q_limb * base_p_size + col];
            const GpuWord product =
                mul_mod(weighted, matrix_value, target_modulus, target_barrett);
            sum = add_mod(sum, product, target_modulus);
        }

        modup_q[target_offset] = sum;
        return;
    }

    /* 此处开始计算p模数模升*/
    const std::size_t local_tid = tid - q_total;
    const std::size_t target_p_limb = local_tid / degree;
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
    GpuWord sum = 0;

    for (std::size_t col = 0; col < decomp_limb_count; ++col)
    {
        const std::size_t source_q_limb = decomp_limb_begin + col;
        const GpuWord value =
            c2_coeff[source_q_limb * degree + coeff];
        const GpuWord source_modulus = rns_primes[source_q_limb];
        const GpuWide source_barrett =
            rns_modulus_constants[source_q_limb];
        const GpuWord inv_punctured =
            qi_inv_punctured[inv_offset + col];
        const GpuWord weighted = inv_punctured == 1
            ? value
            : mul_mod(value, inv_punctured, source_modulus, source_barrett);
        const GpuWord matrix_value =
            p_matrices[matrix_offset + target_p_limb * base_p_size + col];
        const GpuWord product =
            mul_mod(weighted, matrix_value, target_modulus, target_barrett);
        sum = add_mod(sum, product, target_modulus);
    }

    modup_p[target_offset] = sum;
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
    std::size_t degree)
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
    std::size_t table_limb = tid / degree;
    std::size_t key_offset = tid;

    if (tid >= q_word_count)
    {
        local_offset = tid - q_word_count;
        accum = accum_p;
        modup = modup_p;
        table_limb = base_q_size + local_offset / degree;
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
    std::size_t degree)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = base_q_size * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t q_limb = tid / degree;
    const std::size_t coeff = tid % degree;
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
    std::size_t degree)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = base_q_size * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t q_limb = tid / degree;
    const std::size_t coeff = tid % degree;
    const GpuWord q_modulus = rns_primes[q_limb];
    const GpuWide q_barrett = rns_modulus_constants[q_limb];
    GpuWord sum0 = 0;
    GpuWord sum1 = 0;

    for (std::size_t p_limb = 0; p_limb < base_p_size; ++p_limb)
    {
        const std::size_t p_table_limb = base_q_size + p_limb;
        const GpuWord p_modulus = rns_primes[p_table_limb];
        const GpuWide p_barrett =
            rns_modulus_constants[p_table_limb];
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

    converted_q0[tid] = sum0;
    converted_q1[tid] = sum1;
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
    std::size_t degree)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = base_q_size * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t q_limb = tid / degree;
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
        degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_modup_decomposition kernel launch");
}

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
    const std::size_t total_butterflies =
        active_limb_count * (degree >> 1);
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total_butterflies + block_size - 1) / block_size);

    for (std::size_t m = 1, gap = degree >> 1;
         m < degree;
         m <<= 1, gap >>= 1)
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
        gpu_check_cuda(
            cudaGetLastError(),
            "launch_hybrid_forward_ntt_qp stage kernel launch");
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
        degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_multiply_accumulate kernel launch");
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
        degree);
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
        degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_convert_p_to_q kernel launch");
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

    hybrid_apply_moddown_ntt_two_components_kernel<<<grid_size, block_size>>>(
        accum_q0,
        accum_q1,
        converted_q0,
        converted_q1,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.hybrid_inv_p_mod_q.data(),
        base_q_size,
        degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_hybrid_apply_moddown_ntt kernel launch");
}

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
