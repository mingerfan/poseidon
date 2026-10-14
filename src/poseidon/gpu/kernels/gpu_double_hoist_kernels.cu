#include "poseidon/gpu/kernels/gpu_double_hoist_kernels.h"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace poseidon
{
namespace gpu
{
namespace kernel
{
namespace
{

constexpr std::size_t kMaxFusedGiantGroups = 64;
constexpr std::size_t kMaxFusedBabyGroups = 4;

struct GiantGroupKernelArguments
{
    const GpuWord *digit_ptrs[kMaxFusedGiantGroups];
    std::uint32_t group_indices[kMaxFusedGiantGroups];
    std::uint32_t galois_elts[kMaxFusedGiantGroups];
    std::uint32_t key_indices[kMaxFusedGiantGroups];
};

struct BabyKeyMacKernelArguments
{
    std::uint32_t galois_elts[kMaxDoubleHoistFusedBabySteps];
    std::uint32_t key_indices[kMaxDoubleHoistFusedBabySteps];
    std::uint32_t
        term_indices[kMaxFusedBabyGroups]
                    [kMaxDoubleHoistFusedBabySteps];
};

__device__ __forceinline__ GpuWord reduce_u64(
    GpuWide value,
    GpuWord modulus,
    GpuWide barrett_ratio)
{
    const GpuWide quotient = __umul64hi(value, barrett_ratio);
    GpuWide reduced = value - quotient * static_cast<GpuWide>(modulus);
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

__device__ __forceinline__ GpuWord multiply_mod(
    GpuWord left,
    GpuWord right,
    GpuWord modulus,
    GpuWide barrett_ratio)
{
    return reduce_u64(
        static_cast<GpuWide>(left) * static_cast<GpuWide>(right),
        modulus,
        barrett_ratio);
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

__device__ __forceinline__ std::uint32_t reverse_bits_limited(
    std::uint32_t value,
    unsigned int bit_count)
{
    return __brev(value) >> (32U - bit_count);
}

__global__ void lift_identity_kernel(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    const GpuWord *source_q0,
    const GpuWord *source_q1,
    const GpuWord *p_mod_q,
    const GpuWord *q_primes,
    const GpuWide *q_barrett,
    std::size_t q_count,
    std::size_t degree,
    unsigned int degree_power,
    bool accumulate)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t words_per_component = q_count * degree;
    const std::size_t total = 2 * words_per_component;
    if (tid >= total)
    {
        return;
    }

    const std::size_t component = tid / words_per_component;
    const std::size_t local = tid - component * words_per_component;
    const std::size_t limb = local >> degree_power;
    const GpuWord modulus = q_primes[limb];
    const GpuWord lifted = multiply_mod(
        component == 0 ? source_q0[local] : source_q1[local],
        p_mod_q[limb],
        modulus,
        q_barrett[limb]);
    GpuWord *destination =
        component == 0 ? destination_q0 : destination_q1;
    destination[local] = accumulate
        ? add_mod(destination[local], lifted, modulus)
        : lifted;
}

__global__ void pre_rotated_keymul_digit_kernel(
    GpuWord *destination0,
    GpuWord *destination1,
    const GpuWord *digit,
    const GpuWord *key0,
    const GpuWord *key1,
    std::uint32_t galois_elt,
    const GpuWord *moduli,
    const GpuWide *barrett,
    std::size_t limb_count,
    std::size_t degree,
    unsigned int degree_power,
    bool overwrite)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * degree;
    if (tid >= total)
    {
        return;
    }

    const std::uint32_t degree_u32 = static_cast<std::uint32_t>(degree);
    const std::uint32_t degree_mask = degree_u32 - 1;
    const std::size_t limb = tid >> degree_power;
    const std::uint32_t coeff =
        static_cast<std::uint32_t>(tid & degree_mask);
    const std::uint32_t reversed =
        reverse_bits_limited(degree_u32 + coeff, degree_power + 1);
    const std::uint64_t index_raw =
        (static_cast<std::uint64_t>(galois_elt) * reversed) >> 1;
    const std::uint32_t source_coeff = reverse_bits_limited(
        static_cast<std::uint32_t>(index_raw & degree_mask),
        degree_power);
    const std::size_t source_index = limb * degree + source_coeff;

    const GpuWord modulus = moduli[limb];
    const GpuWide ratio = barrett[limb];
    const GpuWord source_value = digit[source_index];
    const GpuWord product0 = multiply_mod(
        source_value,
        key0[source_index],
        modulus,
        ratio);
    const GpuWord product1 = multiply_mod(
        source_value,
        key1[source_index],
        modulus,
        ratio);
    destination0[tid] = overwrite
        ? product0
        : add_mod(destination0[tid], product0, modulus);
    destination1[tid] = overwrite
        ? product1
        : add_mod(destination1[tid], product1, modulus);
}

template <bool AddLiftedC0>
__global__ void pre_rotated_keymul_batch_kernel(
    GpuWord *destination0,
    GpuWord *destination1,
    const GpuWord *digits,
    const GpuWord *const *key0_ptrs,
    const GpuWord *const *key1_ptrs,
    const GpuWord *lifted_c0_source,
    const GpuWord *p_mod_q,
    std::uint32_t galois_elt,
    const GpuWord *moduli,
    const GpuWide *barrett,
    std::size_t dnum,
    std::size_t limb_count,
    std::size_t degree,
    unsigned int degree_power,
    bool overwrite)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t words_per_digit = limb_count * degree;
    if (tid >= words_per_digit)
    {
        return;
    }

    const std::uint32_t degree_u32 = static_cast<std::uint32_t>(degree);
    const std::uint32_t degree_mask = degree_u32 - 1;
    const std::size_t limb = tid >> degree_power;
    const std::uint32_t coeff =
        static_cast<std::uint32_t>(tid & degree_mask);
    const std::uint32_t reversed =
        reverse_bits_limited(degree_u32 + coeff, degree_power + 1);
    const std::uint64_t index_raw =
        (static_cast<std::uint64_t>(galois_elt) * reversed) >> 1;
    const std::uint32_t source_coeff = reverse_bits_limited(
        static_cast<std::uint32_t>(index_raw & degree_mask),
        degree_power);
    const std::size_t source_index = limb * degree + source_coeff;
    const GpuWord modulus = moduli[limb];
    const GpuWide ratio = barrett[limb];

    GpuWord accumulator0 = 0;
    GpuWord accumulator1 = 0;
    for (std::size_t digit = 0; digit < dnum; ++digit)
    {
        const GpuWord source_value =
            digits[digit * words_per_digit + source_index];
        accumulator0 = add_mod(
            accumulator0,
            multiply_mod(
                source_value,
                key0_ptrs[digit][source_index],
                modulus,
                ratio),
            modulus);
        accumulator1 = add_mod(
            accumulator1,
            multiply_mod(
                source_value,
                key1_ptrs[digit][source_index],
                modulus,
                ratio),
            modulus);
    }
    if constexpr (AddLiftedC0)
    {
        accumulator0 = add_mod(
            accumulator0,
            multiply_mod(
                lifted_c0_source[source_index],
                p_mod_q[limb],
                modulus,
                ratio),
            modulus);
    }
    destination0[tid] = overwrite
        ? accumulator0
        : add_mod(destination0[tid], accumulator0, modulus);
    destination1[tid] = overwrite
        ? accumulator1
        : add_mod(destination1[tid], accumulator1, modulus);
}

template <bool AddLiftedC0>
__global__ void pre_rotated_keymul_dnum1_kernel(
    GpuWord *destination0,
    GpuWord *destination1,
    const GpuWord *digit,
    const GpuWord *const *key0_ptrs,
    const GpuWord *const *key1_ptrs,
    const GpuWord *lifted_c0_source,
    const GpuWord *p_mod_q,
    std::uint32_t galois_elt,
    const GpuWord *moduli,
    const GpuWide *barrett,
    std::size_t limb_count,
    std::size_t degree,
    unsigned int degree_power,
    bool overwrite)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * degree;
    if (tid >= total)
    {
        return;
    }

    const std::uint32_t degree_u32 =
        static_cast<std::uint32_t>(degree);
    const std::uint32_t degree_mask = degree_u32 - 1;
    const std::size_t limb = tid >> degree_power;
    const std::uint32_t coeff =
        static_cast<std::uint32_t>(tid & degree_mask);
    const std::uint32_t reversed = reverse_bits_limited(
        degree_u32 + coeff,
        degree_power + 1);
    const std::uint64_t index_raw =
        (static_cast<std::uint64_t>(galois_elt) * reversed) >> 1;
    const std::uint32_t source_coeff = reverse_bits_limited(
        static_cast<std::uint32_t>(index_raw & degree_mask),
        degree_power);
    const std::size_t source_index =
        limb * degree + source_coeff;
    const GpuWord modulus = moduli[limb];
    const GpuWide ratio = barrett[limb];
    const GpuWord source_value = digit[source_index];
    GpuWord result0 = multiply_mod(
        source_value,
        key0_ptrs[0][source_index],
        modulus,
        ratio);
    const GpuWord result1 = multiply_mod(
        source_value,
        key1_ptrs[0][source_index],
        modulus,
        ratio);
    if constexpr (AddLiftedC0)
    {
        result0 = add_mod(
            result0,
            multiply_mod(
                lifted_c0_source[source_index],
                p_mod_q[limb],
                modulus,
                ratio),
            modulus);
    }
    destination0[tid] = overwrite
        ? result0
        : add_mod(destination0[tid], result0, modulus);
    destination1[tid] = overwrite
        ? result1
        : add_mod(destination1[tid], result1, modulus);
}

__global__ void pre_rotated_giant_group_reduce_kernel(
    GpuWord *group_destination,
    const GpuWord *inner_q_batch,
    GiantGroupKernelArguments group_args,
    const GpuWord *const *key0_ptrs,
    const GpuWord *const *key1_ptrs,
    const GpuWord *p_mod_q,
    const GpuWord *moduli,
    const GpuWide *barrett,
    std::size_t active_group_count,
    std::size_t dnum,
    std::size_t storage_dnum,
    std::size_t limb_count,
    std::size_t q_count,
    std::size_t degree,
    unsigned int degree_power,
    bool q_side)
{
    const std::size_t flat_tid =
        blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t words_per_digit = limb_count * degree;
    const std::size_t total = active_group_count * words_per_digit;
    if (flat_tid >= total)
    {
        return;
    }
    const std::size_t active_group = flat_tid / words_per_digit;
    const std::size_t tid =
        flat_tid - active_group * words_per_digit;

    const std::uint32_t degree_u32 = static_cast<std::uint32_t>(degree);
    const std::uint32_t degree_mask = degree_u32 - 1;
    const std::size_t limb = tid >> degree_power;
    const std::uint32_t coeff =
        static_cast<std::uint32_t>(tid & degree_mask);
    const std::uint32_t reversed =
        reverse_bits_limited(degree_u32 + coeff, degree_power + 1);
    const GpuWord modulus = moduli[limb];
    const GpuWide ratio = barrett[limb];
    GpuWord accumulator0 = 0;
    GpuWord accumulator1 = 0;
    {
        const std::uint32_t galois_elt =
            group_args.galois_elts[active_group];
        const std::uint64_t index_raw =
            (static_cast<std::uint64_t>(galois_elt) * reversed) >> 1;
        const std::uint32_t source_coeff = reverse_bits_limited(
            static_cast<std::uint32_t>(index_raw & degree_mask),
            degree_power);
        const std::size_t source_index =
            limb * degree + source_coeff;
        const std::size_t key_base =
            static_cast<std::size_t>(
                group_args.key_indices[active_group]) *
            storage_dnum;
        const GpuWord *digits =
            group_args.digit_ptrs[active_group];

        for (std::size_t digit = 0; digit < dnum; ++digit)
        {
            const GpuWord source_value =
                digits[digit * words_per_digit + source_index];
            accumulator0 = add_mod(
                accumulator0,
                multiply_mod(
                    source_value,
                    key0_ptrs[key_base + digit][source_index],
                    modulus,
                    ratio),
                modulus);
            accumulator1 = add_mod(
                accumulator1,
                multiply_mod(
                    source_value,
                    key1_ptrs[key_base + digit][source_index],
                    modulus,
                    ratio),
                modulus);
        }

        if (q_side)
        {
            const std::size_t q_component_words = q_count * degree;
            const std::size_t source_q_offset =
                (static_cast<std::size_t>(
                     group_args.group_indices[active_group]) *
                 2) *
                    q_component_words +
                source_index;
            accumulator0 = add_mod(
                accumulator0,
                multiply_mod(
                    inner_q_batch[source_q_offset],
                    p_mod_q[limb],
                    modulus,
                    ratio),
                modulus);
        }
    }

    const std::size_t group =
        group_args.group_indices[active_group];
    const std::size_t component_words = words_per_digit;
    group_destination[
        (group * 2) * component_words + tid] = accumulator0;
    group_destination[
        (group * 2 + 1) * component_words + tid] = accumulator1;
}

__global__ void pre_rotated_giant_group_accumulate_kernel(
    GpuWord *destination0,
    GpuWord *destination1,
    const GpuWord *inner_q_batch,
    std::size_t identity_group_index,
    GiantGroupKernelArguments group_args,
    const GpuWord *const *key0_ptrs,
    const GpuWord *const *key1_ptrs,
    const GpuWord *p_mod_q,
    const GpuWord *moduli,
    const GpuWide *barrett,
    std::size_t active_group_count,
    std::size_t dnum,
    std::size_t storage_dnum,
    std::size_t limb_count,
    std::size_t q_count,
    std::size_t degree,
    unsigned int degree_power,
    bool q_side)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t component_words = limb_count * degree;
    if (tid >= component_words)
    {
        return;
    }

    const std::uint32_t degree_u32 = static_cast<std::uint32_t>(degree);
    const std::uint32_t degree_mask = degree_u32 - 1;
    const std::size_t limb = tid >> degree_power;
    const std::uint32_t coeff =
        static_cast<std::uint32_t>(tid & degree_mask);
    const std::uint32_t reversed =
        reverse_bits_limited(degree_u32 + coeff, degree_power + 1);
    const GpuWord modulus = moduli[limb];
    const GpuWide ratio = barrett[limb];

    GpuWord accumulator0 = 0;
    GpuWord accumulator1 = 0;
    if (q_side)
    {
        const std::size_t q_component_words = q_count * degree;
        const std::size_t identity_offset =
            identity_group_index * 2 * q_component_words + tid;
        accumulator0 = multiply_mod(
            inner_q_batch[identity_offset],
            p_mod_q[limb],
            modulus,
            ratio);
        accumulator1 = multiply_mod(
            inner_q_batch[identity_offset + q_component_words],
            p_mod_q[limb],
            modulus,
            ratio);
    }

    for (std::size_t active_group = 0;
         active_group < active_group_count;
         ++active_group)
    {
        const std::uint32_t galois_elt =
            group_args.galois_elts[active_group];
        const std::uint64_t index_raw =
            (static_cast<std::uint64_t>(galois_elt) * reversed) >> 1;
        const std::uint32_t source_coeff = reverse_bits_limited(
            static_cast<std::uint32_t>(index_raw & degree_mask),
            degree_power);
        const std::size_t source_index = limb * degree + source_coeff;
        const std::size_t key_base =
            static_cast<std::size_t>(
                group_args.key_indices[active_group]) *
            storage_dnum;
        const GpuWord *digits =
            group_args.digit_ptrs[active_group];

        for (std::size_t digit = 0; digit < dnum; ++digit)
        {
            const GpuWord source_value =
                digits[digit * component_words + source_index];
            accumulator0 = add_mod(
                accumulator0,
                multiply_mod(
                    source_value,
                    key0_ptrs[key_base + digit][source_index],
                    modulus,
                    ratio),
                modulus);
            accumulator1 = add_mod(
                accumulator1,
                multiply_mod(
                    source_value,
                    key1_ptrs[key_base + digit][source_index],
                    modulus,
                    ratio),
                modulus);
        }

        if (q_side)
        {
            const std::size_t q_component_words = q_count * degree;
            const std::size_t source_q_offset =
                (static_cast<std::size_t>(
                     group_args.group_indices[active_group]) *
                 2) *
                    q_component_words +
                source_index;
            accumulator0 = add_mod(
                accumulator0,
                multiply_mod(
                    inner_q_batch[source_q_offset],
                    p_mod_q[limb],
                    modulus,
                    ratio),
                modulus);
        }
    }

    destination0[tid] = accumulator0;
    destination1[tid] = accumulator1;
}

__global__ void reduce_qp_groups_kernel(
    GpuWord *destination,
    const GpuWord *group_values,
    const GpuWord *moduli,
    std::size_t group_count,
    std::size_t limb_count,
    std::size_t degree)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t component_words = limb_count * degree;
    const std::size_t total = 2 * component_words;
    if (tid >= total)
    {
        return;
    }
    const std::size_t component = tid / component_words;
    const std::size_t in_component =
        tid - component * component_words;
    const std::size_t limb = in_component / degree;
    const GpuWord modulus = moduli[limb];
    GpuWord accumulator = 0;
    for (std::size_t group = 0; group < group_count; ++group)
    {
        accumulator = add_mod(
            accumulator,
            group_values[
                (group * 2 + component) * component_words +
                in_component],
            modulus);
    }
    destination[tid] = accumulator;
}

__global__ void add_lifted_galois_c0_kernel(
    GpuWord *destination_q0,
    const GpuWord *source_q0,
    std::uint32_t galois_elt,
    const GpuWord *p_mod_q,
    const GpuWord *q_primes,
    const GpuWide *q_barrett,
    std::size_t q_count,
    std::size_t degree,
    unsigned int degree_power)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = q_count * degree;
    if (tid >= total)
    {
        return;
    }

    const std::uint32_t degree_u32 = static_cast<std::uint32_t>(degree);
    const std::uint32_t degree_mask = degree_u32 - 1;
    const std::size_t limb = tid >> degree_power;
    const std::uint32_t coeff =
        static_cast<std::uint32_t>(tid & degree_mask);
    const std::uint32_t reversed =
        reverse_bits_limited(degree_u32 + coeff, degree_power + 1);
    const std::uint64_t index_raw =
        (static_cast<std::uint64_t>(galois_elt) * reversed) >> 1;
    const std::uint32_t source_coeff = reverse_bits_limited(
        static_cast<std::uint32_t>(index_raw & degree_mask),
        degree_power);
    const std::size_t source_index = limb * degree + source_coeff;

    const GpuWord modulus = q_primes[limb];
    const GpuWord lifted = multiply_mod(
        source_q0[source_index],
        p_mod_q[limb],
        modulus,
        q_barrett[limb]);
    destination_q0[tid] = add_mod(destination_q0[tid], lifted, modulus);
}

template <bool Compressed>
__device__ __forceinline__ GpuWord load_qp_diagonal(
    const GpuWord *const *diagonal_ptrs,
    const std::uint32_t *diagonal_periods,
    std::uint32_t term,
    std::size_t limb,
    std::size_t coefficient,
    std::size_t degree,
    unsigned int degree_power)
{
    if constexpr (!Compressed)
    {
        return diagonal_ptrs[term][limb * degree + coefficient];
    }
    else
    {
        const std::size_t period = diagonal_periods[term];
        // A period equal to N is retained in ordinary NTT order. This avoids
        // turning an uncompressible diagonal into a bit-reversed, uncoalesced
        // read stream.
        if (period >= degree)
        {
            return diagonal_ptrs[term][limb * degree + coefficient];
        }
        const auto reversed = reverse_bits_limited(
            static_cast<std::uint32_t>(coefficient),
            degree_power);
        const std::size_t compact_coefficient =
            static_cast<std::size_t>(reversed) & (period - 1);
        return diagonal_ptrs[term][
            limb * period + compact_coefficient];
    }
}

template <bool Compressed>
__global__ void qp_plain_mul_accumulate_kernel(
    GpuWord *group_values,
    const GpuWord *baby_values,
    const GpuWord *const *diagonal_ptrs,
    const std::uint32_t *diagonal_periods,
    const std::uint32_t *term_baby_indices,
    const std::uint32_t *group_term_offsets,
    const GpuWord *moduli,
    const GpuWide *barrett,
    std::size_t group_count,
    std::size_t term_count,
    std::size_t limb_count,
    std::size_t degree,
    unsigned int degree_power,
    std::size_t tile_begin,
    std::size_t tile_count,
    bool initialize_accumulators)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t words_per_group = 2 * limb_count * degree;
    const std::size_t total = group_count * words_per_group;
    if (tid >= total)
    {
        return;
    }

    const std::size_t group = tid / words_per_group;
    const std::size_t in_group = tid - group * words_per_group;
    const std::size_t component_stride = limb_count * degree;
    const std::size_t component = in_group / component_stride;
    const std::size_t in_component =
        in_group - component * component_stride;
    const std::size_t limb = in_component >> degree_power;
    const std::size_t coeff = in_component & (degree - 1);
    const GpuWord modulus = moduli[limb];
    const GpuWide ratio = barrett[limb];

    GpuWord accumulator =
        initialize_accumulators ? 0 : group_values[tid];
    const std::uint32_t begin = group_term_offsets[group];
    const std::uint32_t end = group_term_offsets[group + 1];
    if (end > term_count || begin > end)
    {
        return;
    }

    for (std::uint32_t term = begin; term < end; ++term)
    {
        const std::size_t baby_index = term_baby_indices[term];
        if (baby_index < tile_begin ||
            baby_index >= tile_begin + tile_count)
        {
            continue;
        }

        const std::size_t local_baby = baby_index - tile_begin;
        const std::size_t baby_offset =
            ((local_baby * 2 + component) * limb_count + limb) *
                degree +
            coeff;
        const GpuWord diagonal = load_qp_diagonal<Compressed>(
            diagonal_ptrs,
            diagonal_periods,
            term,
            limb,
            coeff,
            degree,
            degree_power);
        const GpuWord product = multiply_mod(
            baby_values[baby_offset],
            diagonal,
            modulus,
            ratio);
        accumulator = add_mod(accumulator, product, modulus);
    }

    group_values[tid] = accumulator;
}

template <int GroupTile, bool Compressed>
__global__ void qp_plain_mul_accumulate_group_tiled_kernel(
    GpuWord *group_values,
    const GpuWord *baby_values,
    const GpuWord *const *diagonal_ptrs,
    const std::uint32_t *diagonal_periods,
    const std::uint32_t *term_baby_indices,
    const std::uint32_t *group_term_offsets,
    const GpuWord *moduli,
    const GpuWide *barrett,
    std::size_t group_count,
    std::size_t term_count,
    std::size_t limb_count,
    std::size_t degree,
    unsigned int degree_power,
    std::size_t tile_begin,
    std::size_t tile_count,
    bool initialize_accumulators)
{
    constexpr std::size_t kCoefficientTile = 32;
    constexpr std::size_t kMaxBabyTile = 8;
    static_assert(GroupTile == 4 || GroupTile == 8);
    __shared__ GpuWord shared_baby[kMaxBabyTile][kCoefficientTile];

    const std::size_t lane = threadIdx.x;
    const std::size_t group_row = threadIdx.y;
    const std::size_t coefficient =
        blockIdx.x * kCoefficientTile + lane;
    const std::size_t packed_component_limb = blockIdx.y;
    const std::size_t component =
        packed_component_limb / limb_count;
    const std::size_t limb =
        packed_component_limb - component * limb_count;
    const std::size_t group =
        blockIdx.z * GroupTile + group_row;

    const std::size_t linear_thread =
        group_row * kCoefficientTile + lane;
    const std::size_t shared_word_count =
        tile_count * kCoefficientTile;
    for (std::size_t shared_index = linear_thread;
         shared_index < shared_word_count;
         shared_index += kCoefficientTile * GroupTile)
    {
        const std::size_t local_baby =
            shared_index / kCoefficientTile;
        const std::size_t source_lane =
            shared_index - local_baby * kCoefficientTile;
        const std::size_t source_coefficient =
            blockIdx.x * kCoefficientTile + source_lane;
        GpuWord value = 0;
        if (source_coefficient < degree)
        {
            const std::size_t baby_offset =
                ((local_baby * 2 + component) * limb_count + limb) *
                    degree +
                source_coefficient;
            value = baby_values[baby_offset];
        }
        shared_baby[local_baby][source_lane] = value;
    }
    __syncthreads();

    if (group >= group_count || coefficient >= degree)
    {
        return;
    }

    const std::size_t words_per_group = 2 * limb_count * degree;
    const std::size_t group_offset =
        group * words_per_group +
        (component * limb_count + limb) * degree +
        coefficient;
    const GpuWord modulus = moduli[limb];
    const GpuWide ratio = barrett[limb];
    GpuWord accumulator = initialize_accumulators
        ? 0
        : group_values[group_offset];
    const std::uint32_t begin = group_term_offsets[group];
    const std::uint32_t end = group_term_offsets[group + 1];
    if (end > term_count || begin > end)
    {
        return;
    }

    for (std::uint32_t term = begin; term < end; ++term)
    {
        const std::size_t baby_index = term_baby_indices[term];
        if (baby_index < tile_begin ||
            baby_index >= tile_begin + tile_count)
        {
            continue;
        }

        const std::size_t local_baby = baby_index - tile_begin;
        const GpuWord diagonal = load_qp_diagonal<Compressed>(
            diagonal_ptrs,
            diagonal_periods,
            term,
            limb,
            coefficient,
            degree,
            degree_power);
        accumulator = add_mod(
            accumulator,
            multiply_mod(
                shared_baby[local_baby][lane],
                diagonal,
                modulus,
                ratio),
            modulus);
    }

    group_values[group_offset] = accumulator;
}

template <int GroupTile, bool Compressed>
__global__ void qp_plain_mul_accumulate_group_component_fused_kernel(
    GpuWord *__restrict__ group_values,
    const GpuWord *__restrict__ baby_values,
    const GpuWord *const *__restrict__ diagonal_ptrs,
    const std::uint32_t *__restrict__ diagonal_periods,
    const std::uint32_t *__restrict__ term_baby_indices,
    const std::uint32_t *__restrict__ group_term_offsets,
    const GpuWord *__restrict__ moduli,
    const GpuWide *__restrict__ barrett,
    std::size_t group_count,
    std::size_t term_count,
    std::size_t limb_count,
    std::size_t degree,
    unsigned int degree_power,
    std::size_t tile_begin,
    std::size_t tile_count,
    bool initialize_accumulators)
{
    constexpr std::size_t kCoefficientTile = 32;
    constexpr std::size_t kMaxBabyTile = 8;
    static_assert(GroupTile == 4 || GroupTile == 8);
    __shared__ GpuWord
        shared_baby[2][kMaxBabyTile][kCoefficientTile];

    const std::size_t lane = threadIdx.x;
    const std::size_t group_row = threadIdx.y;
    const std::size_t coefficient =
        blockIdx.x * kCoefficientTile + lane;
    const std::size_t limb = blockIdx.y;
    const std::size_t group =
        blockIdx.z * GroupTile + group_row;

    const std::size_t linear_thread =
        group_row * kCoefficientTile + lane;
    const std::size_t words_per_component =
        tile_count * kCoefficientTile;
    const std::size_t shared_word_count = 2 * words_per_component;
    for (std::size_t shared_index = linear_thread;
         shared_index < shared_word_count;
         shared_index += kCoefficientTile * GroupTile)
    {
        const std::size_t component =
            shared_index / words_per_component;
        const std::size_t in_component =
            shared_index - component * words_per_component;
        const std::size_t local_baby =
            in_component / kCoefficientTile;
        const std::size_t source_lane =
            in_component - local_baby * kCoefficientTile;
        const std::size_t source_coefficient =
            blockIdx.x * kCoefficientTile + source_lane;
        GpuWord value = 0;
        if (source_coefficient < degree)
        {
            const std::size_t baby_offset =
                ((local_baby * 2 + component) * limb_count + limb) *
                    degree +
                source_coefficient;
            value = baby_values[baby_offset];
        }
        shared_baby[component][local_baby][source_lane] = value;
    }
    __syncthreads();

    if (group >= group_count || coefficient >= degree)
    {
        return;
    }

    const std::size_t component_stride = limb_count * degree;
    const std::size_t group_offset =
        group * 2 * component_stride + limb * degree + coefficient;
    const GpuWord modulus = moduli[limb];
    const GpuWide ratio = barrett[limb];
    GpuWord accumulator0 = initialize_accumulators
        ? 0
        : group_values[group_offset];
    GpuWord accumulator1 = initialize_accumulators
        ? 0
        : group_values[group_offset + component_stride];
    const std::uint32_t begin = group_term_offsets[group];
    const std::uint32_t end = group_term_offsets[group + 1];
    if (end > term_count || begin > end)
    {
        return;
    }

    for (std::uint32_t term = begin; term < end; ++term)
    {
        const std::size_t baby_index = term_baby_indices[term];
        if (baby_index < tile_begin ||
            baby_index >= tile_begin + tile_count)
        {
            continue;
        }

        const std::size_t local_baby = baby_index - tile_begin;
        const GpuWord diagonal = load_qp_diagonal<Compressed>(
            diagonal_ptrs,
            diagonal_periods,
            term,
            limb,
            coefficient,
            degree,
            degree_power);
        accumulator0 = add_mod(
            accumulator0,
            multiply_mod(
                shared_baby[0][local_baby][lane],
                diagonal,
                modulus,
                ratio),
            modulus);
        accumulator1 = add_mod(
            accumulator1,
            multiply_mod(
                shared_baby[1][local_baby][lane],
                diagonal,
                modulus,
                ratio),
            modulus);
    }

    group_values[group_offset] = accumulator0;
    group_values[group_offset + component_stride] = accumulator1;
}

template <int GroupTile, bool IsQ, bool Compressed>
__global__ void fused_baby_keyswitch_plain_accumulate_kernel(
    GpuWord *__restrict__ group_values,
    const GpuWord *__restrict__ digits,
    const GpuWord *__restrict__ source_q0,
    const GpuWord *__restrict__ source_q1,
    BabyKeyMacKernelArguments arguments,
    const GpuWord *const *__restrict__ key0_ptrs,
    const GpuWord *const *__restrict__ key1_ptrs,
    const GpuWord *const *__restrict__ diagonal_ptrs,
    const std::uint32_t *__restrict__ diagonal_periods,
    const GpuWord *__restrict__ p_mod_q,
    const GpuWord *__restrict__ moduli,
    const GpuWide *__restrict__ barrett,
    std::size_t group_count,
    std::size_t term_count,
    std::size_t tile_count,
    std::size_t dnum,
    std::size_t storage_dnum,
    std::size_t limb_count,
    std::size_t degree,
    unsigned int degree_power,
    bool initialize_accumulators)
{
    static_assert(GroupTile == 1 || GroupTile == 4);
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t words_per_digit = limb_count * degree;
    if (tid >= words_per_digit)
    {
        return;
    }

    const std::size_t limb = tid >> degree_power;
    const std::uint32_t coefficient = static_cast<std::uint32_t>(
        tid & (degree - 1));
    const GpuWord modulus = moduli[limb];
    const GpuWide ratio = barrett[limb];
    const std::size_t component_stride = limb_count * degree;
    const std::size_t group_stride = 2 * component_stride;

    GpuWord accumulator0[GroupTile];
    GpuWord accumulator1[GroupTile];
#pragma unroll
    for (int group = 0; group < GroupTile; ++group)
    {
        if (static_cast<std::size_t>(group) < group_count)
        {
            const std::size_t offset =
                static_cast<std::size_t>(group) * group_stride + tid;
            accumulator0[group] = initialize_accumulators
                ? 0
                : group_values[offset];
            accumulator1[group] = initialize_accumulators
                ? 0
                : group_values[offset + component_stride];
        }
        else
        {
            accumulator0[group] = 0;
            accumulator1[group] = 0;
        }
    }

    const std::uint32_t degree_u32 = static_cast<std::uint32_t>(degree);
    const std::uint32_t degree_mask = degree_u32 - 1;
    for (std::size_t baby = 0; baby < tile_count; ++baby)
    {
        const std::uint32_t galois_elt = arguments.galois_elts[baby];
        GpuWord baby0 = 0;
        GpuWord baby1 = 0;
        if (galois_elt == 0)
        {
            if constexpr (IsQ)
            {
                baby0 = multiply_mod(
                    source_q0[tid], p_mod_q[limb], modulus, ratio);
                baby1 = multiply_mod(
                    source_q1[tid], p_mod_q[limb], modulus, ratio);
            }
        }
        else
        {
            const std::uint32_t reversed = reverse_bits_limited(
                degree_u32 + coefficient, degree_power + 1);
            const std::uint64_t index_raw =
                (static_cast<std::uint64_t>(galois_elt) * reversed) >> 1;
            const std::uint32_t source_coefficient = reverse_bits_limited(
                static_cast<std::uint32_t>(index_raw & degree_mask),
                degree_power);
            const std::size_t source_index =
                limb * degree + source_coefficient;
            const std::size_t key_base =
                static_cast<std::size_t>(arguments.key_indices[baby]) *
                storage_dnum;
            for (std::size_t digit = 0; digit < dnum; ++digit)
            {
                const GpuWord digit_value =
                    digits[digit * words_per_digit + source_index];
                baby0 = add_mod(
                    baby0,
                    multiply_mod(
                        digit_value,
                        key0_ptrs[key_base + digit][source_index],
                        modulus,
                        ratio),
                    modulus);
                baby1 = add_mod(
                    baby1,
                    multiply_mod(
                        digit_value,
                        key1_ptrs[key_base + digit][source_index],
                        modulus,
                        ratio),
                    modulus);
            }
            if constexpr (IsQ)
            {
                baby0 = add_mod(
                    baby0,
                    multiply_mod(
                        source_q0[source_index],
                        p_mod_q[limb],
                        modulus,
                        ratio),
                    modulus);
            }
        }

#pragma unroll
        for (int group = 0; group < GroupTile; ++group)
        {
            if (static_cast<std::size_t>(group) >= group_count)
            {
                continue;
            }
            const std::uint32_t term =
                arguments.term_indices[group][baby];
            if (term >= term_count)
            {
                continue;
            }
            const GpuWord diagonal = load_qp_diagonal<Compressed>(
                diagonal_ptrs,
                diagonal_periods,
                term,
                limb,
                coefficient,
                degree,
                degree_power);
            accumulator0[group] = add_mod(
                accumulator0[group],
                multiply_mod(baby0, diagonal, modulus, ratio),
                modulus);
            accumulator1[group] = add_mod(
                accumulator1[group],
                multiply_mod(baby1, diagonal, modulus, ratio),
                modulus);
        }
    }

#pragma unroll
    for (int group = 0; group < GroupTile; ++group)
    {
        if (static_cast<std::size_t>(group) < group_count)
        {
            const std::size_t offset =
                static_cast<std::size_t>(group) * group_stride + tid;
            group_values[offset] = accumulator0[group];
            group_values[offset + component_stride] = accumulator1[group];
        }
    }
}

unsigned int log2_degree(std::size_t degree, const char *name)
{
    if (degree == 0 || (degree & (degree - 1)) != 0 ||
        degree > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::invalid_argument(std::string(name) + ": invalid degree");
    }
    unsigned int result = 0;
    while ((std::size_t{1} << result) < degree)
    {
        ++result;
    }
    return result;
}

void validate_parameter_shard(
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    const char *name)
{
    const std::size_t q_count = parameter_shard.hybrid_base_q_count;
    const std::size_t p_count = parameter_shard.hybrid_base_p_count;
    if (q_count == 0 || p_count == 0 ||
        parameter_shard.rns_primes.size() < q_count + p_count ||
        parameter_shard.rns_modulus_constants.size() < q_count + p_count ||
        parameter_shard.hybrid_p_mod_q.size() < q_count)
    {
        throw std::invalid_argument(std::string(name) + ": incomplete HYBRID tables");
    }
    (void)log2_degree(degree, name);
}

}  // namespace

void launch_double_hoist_pre_rotated_keymul_digit(
    GpuWord *destination_q0,
    GpuWord *destination_p0,
    GpuWord *destination_q1,
    GpuWord *destination_p1,
    GpuWord *scratch_group_q,
    GpuWord *scratch_group_p,
    const GpuWord *digit_q,
    const GpuWord *digit_p,
    const GpuWord *key_q0,
    const GpuWord *key_p0,
    const GpuWord *key_q1,
    const GpuWord *key_p1,
    std::uint32_t galois_elt,
    bool overwrite,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    const char *name =
        "launch_double_hoist_pre_rotated_keymul_digit";
    validate_parameter_shard(parameter_shard, degree, name);
    if (destination_q0 == nullptr || destination_p0 == nullptr ||
        destination_q1 == nullptr || destination_p1 == nullptr ||
        digit_q == nullptr || digit_p == nullptr ||
        key_q0 == nullptr || key_p0 == nullptr ||
        key_q1 == nullptr || key_p1 == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null pointer");
    }
    gpu_check_cuda(cudaSetDevice(parameter_shard.device_id), name);

    constexpr int block_size = 256;
    const unsigned int degree_power = log2_degree(degree, name);
    const std::size_t q_count = parameter_shard.hybrid_base_q_count;
    const std::size_t p_count = parameter_shard.hybrid_base_p_count;
    const int q_grid = static_cast<int>(
        (q_count * degree + block_size - 1) / block_size);
    pre_rotated_keymul_digit_kernel<<<q_grid, block_size>>>(
        destination_q0,
        destination_q1,
        digit_q,
        key_q0,
        key_q1,
        galois_elt,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        q_count,
        degree,
        degree_power,
        overwrite);
    gpu_check_cuda(cudaGetLastError(), name);

    const int p_grid = static_cast<int>(
        (p_count * degree + block_size - 1) / block_size);
    pre_rotated_keymul_digit_kernel<<<p_grid, block_size>>>(
        destination_p0,
        destination_p1,
        digit_p,
        key_p0,
        key_p1,
        galois_elt,
        parameter_shard.rns_primes.data() + q_count,
        parameter_shard.rns_modulus_constants.data() + q_count,
        p_count,
        degree,
        degree_power,
        overwrite);
    gpu_check_cuda(cudaGetLastError(), name);
}

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
    std::size_t degree)
{
    const char *name =
        "launch_double_hoist_pre_rotated_keymul_batch";
    validate_parameter_shard(parameter_shard, degree, name);
    if (destination_q0 == nullptr || destination_p0 == nullptr ||
        destination_q1 == nullptr || destination_p1 == nullptr ||
        digits_q == nullptr || digits_p == nullptr ||
        key_q0_ptrs == nullptr || key_p0_ptrs == nullptr ||
        key_q1_ptrs == nullptr || key_p1_ptrs == nullptr ||
        dnum == 0)
    {
        throw std::invalid_argument(std::string(name) + ": invalid argument");
    }
    if (lifted_c0_source_q != nullptr && dnum != 1)
    {
        throw std::invalid_argument(
            std::string(name) + ": fused c0 requires dnum=1");
    }
    gpu_check_cuda(cudaSetDevice(parameter_shard.device_id), name);

    constexpr int block_size = 256;
    const unsigned int degree_power = log2_degree(degree, name);
    const std::size_t q_count = parameter_shard.hybrid_base_q_count;
    const std::size_t p_count = parameter_shard.hybrid_base_p_count;
    const int q_grid = static_cast<int>(
        (q_count * degree + block_size - 1) / block_size);
    if (dnum == 1 && lifted_c0_source_q != nullptr)
    {
        pre_rotated_keymul_dnum1_kernel<true><<<q_grid, block_size>>>(
            destination_q0,
            destination_q1,
            digits_q,
            key_q0_ptrs,
            key_q1_ptrs,
            lifted_c0_source_q,
            parameter_shard.hybrid_p_mod_q.data(),
            galois_elt,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            q_count,
            degree,
            degree_power,
            overwrite);
    }
    else if (dnum == 1)
    {
        pre_rotated_keymul_dnum1_kernel<false><<<q_grid, block_size>>>(
            destination_q0,
            destination_q1,
            digits_q,
            key_q0_ptrs,
            key_q1_ptrs,
            nullptr,
            nullptr,
            galois_elt,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            q_count,
            degree,
            degree_power,
            overwrite);
    }
    else
    {
        pre_rotated_keymul_batch_kernel<false><<<q_grid, block_size>>>(
            destination_q0,
            destination_q1,
            digits_q,
            key_q0_ptrs,
            key_q1_ptrs,
            nullptr,
            nullptr,
            galois_elt,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            dnum,
            q_count,
            degree,
            degree_power,
            overwrite);
    }
    gpu_check_cuda(cudaGetLastError(), name);

    const int p_grid = static_cast<int>(
        (p_count * degree + block_size - 1) / block_size);
    if (dnum == 1)
    {
        pre_rotated_keymul_dnum1_kernel<false><<<p_grid, block_size>>>(
            destination_p0,
            destination_p1,
            digits_p,
            key_p0_ptrs,
            key_p1_ptrs,
            nullptr,
            nullptr,
            galois_elt,
            parameter_shard.rns_primes.data() + q_count,
            parameter_shard.rns_modulus_constants.data() + q_count,
            p_count,
            degree,
            degree_power,
            overwrite);
    }
    else
    {
        pre_rotated_keymul_batch_kernel<false><<<p_grid, block_size>>>(
            destination_p0,
            destination_p1,
            digits_p,
            key_p0_ptrs,
            key_p1_ptrs,
            nullptr,
            nullptr,
            galois_elt,
            parameter_shard.rns_primes.data() + q_count,
            parameter_shard.rns_modulus_constants.data() + q_count,
            dnum,
            p_count,
            degree,
            degree_power,
            overwrite);
    }
    gpu_check_cuda(cudaGetLastError(), name);
}

void launch_double_hoist_lift_identity(
    GpuWord *destination_q0,
    GpuWord *destination_q1,
    GpuWord *destination_p0,
    GpuWord *destination_p1,
    const GpuWord *source_q0,
    const GpuWord *source_q1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool accumulate)
{
    const char *name = "launch_double_hoist_lift_identity";
    validate_parameter_shard(parameter_shard, degree, name);
    if (destination_q0 == nullptr || destination_q1 == nullptr ||
        destination_p0 == nullptr || destination_p1 == nullptr ||
        source_q0 == nullptr || source_q1 == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null pointer");
    }
    gpu_check_cuda(cudaSetDevice(parameter_shard.device_id), name);

    const std::size_t q_count = parameter_shard.hybrid_base_q_count;
    const std::size_t p_count = parameter_shard.hybrid_base_p_count;
    const std::size_t total = 2 * q_count * degree;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    lift_identity_kernel<<<grid_size, block_size>>>(
        destination_q0,
        destination_q1,
        source_q0,
        source_q1,
        parameter_shard.hybrid_p_mod_q.data(),
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        q_count,
        degree,
        log2_degree(degree, name),
        accumulate);
    gpu_check_cuda(cudaGetLastError(), name);

    if (!accumulate)
    {
        gpu_check_cuda(
            cudaMemset(destination_p0, 0, p_count * degree * sizeof(GpuWord)),
            name);
        gpu_check_cuda(
            cudaMemset(destination_p1, 0, p_count * degree * sizeof(GpuWord)),
            name);
    }
}

void launch_double_hoist_add_lifted_galois_c0(
    GpuWord *destination_q0,
    const GpuWord *source_q0,
    std::uint32_t galois_elt,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    const char *name = "launch_double_hoist_add_lifted_galois_c0";
    validate_parameter_shard(parameter_shard, degree, name);
    if (destination_q0 == nullptr || source_q0 == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null pointer");
    }
    gpu_check_cuda(cudaSetDevice(parameter_shard.device_id), name);

    const std::size_t q_count = parameter_shard.hybrid_base_q_count;
    const std::size_t total = q_count * degree;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    add_lifted_galois_c0_kernel<<<grid_size, block_size>>>(
        destination_q0,
        source_q0,
        galois_elt,
        parameter_shard.hybrid_p_mod_q.data(),
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        q_count,
        degree,
        log2_degree(degree, name));
    gpu_check_cuda(cudaGetLastError(), name);
}

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
    std::size_t degree)
{
    const char *name =
        "launch_double_hoist_pre_rotated_giant_group_reduce";
    validate_parameter_shard(parameter_shard, degree, name);
    if (scratch_group_q == nullptr || scratch_group_p == nullptr ||
        inner_q_batch == nullptr ||
        host_group_digit_q_ptrs == nullptr ||
        host_group_digit_p_ptrs == nullptr ||
        host_group_indices == nullptr ||
        host_galois_elts == nullptr ||
        host_key_indices == nullptr ||
        key_q0_ptrs == nullptr || key_p0_ptrs == nullptr ||
        key_q1_ptrs == nullptr || key_p1_ptrs == nullptr ||
        active_group_count == 0 ||
        active_group_count > kMaxFusedGiantGroups ||
        dnum == 0 || storage_dnum < dnum)
    {
        throw std::invalid_argument(std::string(name) + ": invalid argument");
    }
    gpu_check_cuda(cudaSetDevice(parameter_shard.device_id), name);

    constexpr int block_size = 256;
    const unsigned int degree_power = log2_degree(degree, name);
    const std::size_t q_count = parameter_shard.hybrid_base_q_count;
    const std::size_t p_count = parameter_shard.hybrid_base_p_count;
    GiantGroupKernelArguments q_group_args{};
    GiantGroupKernelArguments p_group_args{};
    for (std::size_t group = 0;
         group < active_group_count;
         ++group)
    {
        q_group_args.digit_ptrs[group] =
            host_group_digit_q_ptrs[group];
        p_group_args.digit_ptrs[group] =
            host_group_digit_p_ptrs[group];
        q_group_args.group_indices[group] =
            p_group_args.group_indices[group] =
                host_group_indices[group];
        q_group_args.galois_elts[group] =
            p_group_args.galois_elts[group] =
                host_galois_elts[group];
        q_group_args.key_indices[group] =
            p_group_args.key_indices[group] =
                host_key_indices[group];
    }
    const int q_grid = static_cast<int>(
        (active_group_count * q_count * degree +
         block_size - 1) /
        block_size);
    pre_rotated_giant_group_reduce_kernel<<<q_grid, block_size>>>(
        scratch_group_q,
        inner_q_batch,
        q_group_args,
        key_q0_ptrs,
        key_q1_ptrs,
        parameter_shard.hybrid_p_mod_q.data(),
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        active_group_count,
        dnum,
        storage_dnum,
        q_count,
        q_count,
        degree,
        degree_power,
        true);
    gpu_check_cuda(cudaGetLastError(), name);

    const int p_grid = static_cast<int>(
        (active_group_count * p_count * degree +
         block_size - 1) /
        block_size);
    pre_rotated_giant_group_reduce_kernel<<<p_grid, block_size>>>(
        scratch_group_p,
        inner_q_batch,
        p_group_args,
        key_p0_ptrs,
        key_p1_ptrs,
        nullptr,
        parameter_shard.rns_primes.data() + q_count,
        parameter_shard.rns_modulus_constants.data() + q_count,
        active_group_count,
        dnum,
        storage_dnum,
        p_count,
        q_count,
        degree,
        degree_power,
        false);
    gpu_check_cuda(cudaGetLastError(), name);
}

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
    std::size_t degree)
{
    const char *name =
        "launch_double_hoist_pre_rotated_giant_group_accumulate";
    validate_parameter_shard(parameter_shard, degree, name);
    if (destination_q0 == nullptr || destination_q1 == nullptr ||
        destination_p0 == nullptr || destination_p1 == nullptr ||
        inner_q_batch == nullptr ||
        host_group_digit_q_ptrs == nullptr ||
        host_group_digit_p_ptrs == nullptr ||
        host_group_indices == nullptr ||
        host_galois_elts == nullptr ||
        host_key_indices == nullptr ||
        key_q0_ptrs == nullptr || key_p0_ptrs == nullptr ||
        key_q1_ptrs == nullptr || key_p1_ptrs == nullptr ||
        active_group_count == 0 ||
        active_group_count > kMaxFusedGiantGroups ||
        dnum == 0 || storage_dnum < dnum)
    {
        throw std::invalid_argument(std::string(name) + ": invalid argument");
    }
    gpu_check_cuda(cudaSetDevice(parameter_shard.device_id), name);

    GiantGroupKernelArguments q_group_args{};
    GiantGroupKernelArguments p_group_args{};
    for (std::size_t group = 0;
         group < active_group_count;
         ++group)
    {
        q_group_args.digit_ptrs[group] =
            host_group_digit_q_ptrs[group];
        p_group_args.digit_ptrs[group] =
            host_group_digit_p_ptrs[group];
        q_group_args.group_indices[group] =
            p_group_args.group_indices[group] =
                host_group_indices[group];
        q_group_args.galois_elts[group] =
            p_group_args.galois_elts[group] =
                host_galois_elts[group];
        q_group_args.key_indices[group] =
            p_group_args.key_indices[group] =
                host_key_indices[group];
    }

    constexpr int block_size = 256;
    const unsigned int degree_power = log2_degree(degree, name);
    const std::size_t q_count = parameter_shard.hybrid_base_q_count;
    const std::size_t p_count = parameter_shard.hybrid_base_p_count;
    const int q_grid = static_cast<int>(
        (q_count * degree + block_size - 1) / block_size);
    pre_rotated_giant_group_accumulate_kernel<<<q_grid, block_size>>>(
        destination_q0,
        destination_q1,
        inner_q_batch,
        identity_group_index,
        q_group_args,
        key_q0_ptrs,
        key_q1_ptrs,
        parameter_shard.hybrid_p_mod_q.data(),
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        active_group_count,
        dnum,
        storage_dnum,
        q_count,
        q_count,
        degree,
        degree_power,
        true);
    gpu_check_cuda(cudaGetLastError(), name);

    const int p_grid = static_cast<int>(
        (p_count * degree + block_size - 1) / block_size);
    pre_rotated_giant_group_accumulate_kernel<<<p_grid, block_size>>>(
        destination_p0,
        destination_p1,
        inner_q_batch,
        identity_group_index,
        p_group_args,
        key_p0_ptrs,
        key_p1_ptrs,
        nullptr,
        parameter_shard.rns_primes.data() + q_count,
        parameter_shard.rns_modulus_constants.data() + q_count,
        active_group_count,
        dnum,
        storage_dnum,
        p_count,
        q_count,
        degree,
        degree_power,
        false);
    gpu_check_cuda(cudaGetLastError(), name);
}

void launch_double_hoist_reduce_p_groups(
    GpuWord *destination_p0,
    GpuWord *destination_p1,
    const GpuWord *group_p,
    std::size_t group_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    const char *name = "launch_double_hoist_reduce_p_groups";
    validate_parameter_shard(parameter_shard, degree, name);
    if (destination_p0 == nullptr || destination_p1 == nullptr ||
        group_p == nullptr || group_count == 0)
    {
        throw std::invalid_argument(std::string(name) + ": invalid argument");
    }
    gpu_check_cuda(cudaSetDevice(parameter_shard.device_id), name);
    constexpr int block_size = 256;
    const std::size_t q_count =
        parameter_shard.hybrid_base_q_count;
    const std::size_t p_count =
        parameter_shard.hybrid_base_p_count;
    const int p_reduce_grid = static_cast<int>(
        (2 * p_count * degree + block_size - 1) / block_size);
    reduce_qp_groups_kernel<<<p_reduce_grid, block_size>>>(
        destination_p0,
        group_p,
        parameter_shard.rns_primes.data() + q_count,
        group_count,
        p_count,
        degree);
    gpu_check_cuda(cudaGetLastError(), name);
}

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
    bool initialize_accumulators)
{
    const char *name =
        "launch_double_hoist_qp_plain_mul_accumulate_groups";
    validate_parameter_shard(parameter_shard, degree, name);
    if (group_q == nullptr || group_p == nullptr ||
        baby_q == nullptr || baby_p == nullptr ||
        diagonal_q_ptrs == nullptr || diagonal_p_ptrs == nullptr ||
        (compressed_plaintexts && diagonal_periods == nullptr) ||
        term_baby_indices == nullptr || group_term_offsets == nullptr ||
        group_count == 0 || term_count == 0 || tile_count == 0)
    {
        throw std::invalid_argument(std::string(name) + ": invalid argument");
    }
    gpu_check_cuda(cudaSetDevice(parameter_shard.device_id), name);

    constexpr int block_size = 256;
    const unsigned int degree_power = log2_degree(degree, name);
    const std::size_t q_count = parameter_shard.hybrid_base_q_count;
    const std::size_t p_count = parameter_shard.hybrid_base_p_count;
    bool use_group_tiled8 = true;
    if (const char *raw = std::getenv(
            "POSEIDON_DOUBLE_HOIST_QP_MAC_GROUP_TILED_8"))
    {
        const std::string value(raw);
        use_group_tiled8 = value != "0" &&
            value != "OFF" && value != "off" &&
            value != "false" && value != "FALSE";
    }
    use_group_tiled8 = use_group_tiled8 &&
        group_count > 1 && tile_count <= 8;
    bool use_component_fused = true;
    if (const char *raw = std::getenv(
            "POSEIDON_DOUBLE_HOIST_QP_MAC_COMPONENT_FUSED"))
    {
        const std::string value(raw);
        use_component_fused = value != "0" &&
            value != "OFF" && value != "off" &&
            value != "false" && value != "FALSE";
    }
    use_component_fused = use_component_fused && use_group_tiled8;

    auto launch_for_layout = [&](auto layout_tag) {
    constexpr bool kCompressed = decltype(layout_tag)::value;
    if (use_group_tiled8)
    {
        constexpr unsigned int kCoefficientTile = 32;
        const unsigned int group_tile = group_count <= 4 ? 4 : 8;
        const dim3 block(kCoefficientTile, group_tile);
        const dim3 q_grid(
            static_cast<unsigned int>(
                (degree + kCoefficientTile - 1) / kCoefficientTile),
            static_cast<unsigned int>(
                (use_component_fused ? 1 : 2) * q_count),
            static_cast<unsigned int>(
                (group_count + group_tile - 1) / group_tile));
        if (group_tile == 4)
        {
            if (use_component_fused)
            {
                qp_plain_mul_accumulate_group_component_fused_kernel<
                    4, kCompressed>
                    <<<q_grid, block>>>(
                        group_q,
                        baby_q,
                        diagonal_q_ptrs,
                        diagonal_periods,
                        term_baby_indices,
                        group_term_offsets,
                        parameter_shard.rns_primes.data(),
                        parameter_shard.rns_modulus_constants.data(),
                        group_count,
                        term_count,
                        q_count,
                        degree,
                        degree_power,
                        tile_begin,
                        tile_count,
                        initialize_accumulators);
            }
            else
            {
                qp_plain_mul_accumulate_group_tiled_kernel<4, kCompressed>
                    <<<q_grid, block>>>(
                    group_q,
                    baby_q,
                    diagonal_q_ptrs,
                    diagonal_periods,
                    term_baby_indices,
                    group_term_offsets,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    group_count,
                    term_count,
                    q_count,
                    degree,
                    degree_power,
                    tile_begin,
                    tile_count,
                    initialize_accumulators);
            }
        }
        else
        {
            if (use_component_fused)
            {
                qp_plain_mul_accumulate_group_component_fused_kernel<
                    8, kCompressed>
                    <<<q_grid, block>>>(
                        group_q,
                        baby_q,
                        diagonal_q_ptrs,
                        diagonal_periods,
                        term_baby_indices,
                        group_term_offsets,
                        parameter_shard.rns_primes.data(),
                        parameter_shard.rns_modulus_constants.data(),
                        group_count,
                        term_count,
                        q_count,
                        degree,
                        degree_power,
                        tile_begin,
                        tile_count,
                        initialize_accumulators);
            }
            else
            {
                qp_plain_mul_accumulate_group_tiled_kernel<8, kCompressed>
                    <<<q_grid, block>>>(
                    group_q,
                    baby_q,
                    diagonal_q_ptrs,
                    diagonal_periods,
                    term_baby_indices,
                    group_term_offsets,
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    group_count,
                    term_count,
                    q_count,
                    degree,
                    degree_power,
                    tile_begin,
                    tile_count,
                    initialize_accumulators);
            }
        }
    }
    else
    {
        const std::size_t q_total = group_count * 2 * q_count * degree;
        const int q_grid = static_cast<int>(
            (q_total + block_size - 1) / block_size);
        qp_plain_mul_accumulate_kernel<kCompressed>
            <<<q_grid, block_size>>>(
            group_q,
            baby_q,
            diagonal_q_ptrs,
            diagonal_periods,
            term_baby_indices,
            group_term_offsets,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            group_count,
            term_count,
            q_count,
            degree,
            degree_power,
            tile_begin,
            tile_count,
            initialize_accumulators);
    }
    gpu_check_cuda(cudaGetLastError(), name);

    if (use_group_tiled8)
    {
        constexpr unsigned int kCoefficientTile = 32;
        const unsigned int group_tile = group_count <= 4 ? 4 : 8;
        const dim3 block(kCoefficientTile, group_tile);
        const dim3 p_grid(
            static_cast<unsigned int>(
                (degree + kCoefficientTile - 1) / kCoefficientTile),
            static_cast<unsigned int>(
                (use_component_fused ? 1 : 2) * p_count),
            static_cast<unsigned int>(
                (group_count + group_tile - 1) / group_tile));
        if (group_tile == 4)
        {
            if (use_component_fused)
            {
                qp_plain_mul_accumulate_group_component_fused_kernel<
                    4, kCompressed>
                    <<<p_grid, block>>>(
                        group_p,
                        baby_p,
                        diagonal_p_ptrs,
                        diagonal_periods,
                        term_baby_indices,
                        group_term_offsets,
                        parameter_shard.rns_primes.data() + q_count,
                        parameter_shard.rns_modulus_constants.data() + q_count,
                        group_count,
                        term_count,
                        p_count,
                        degree,
                        degree_power,
                        tile_begin,
                        tile_count,
                        initialize_accumulators);
            }
            else
            {
                qp_plain_mul_accumulate_group_tiled_kernel<4, kCompressed>
                    <<<p_grid, block>>>(
                    group_p,
                    baby_p,
                    diagonal_p_ptrs,
                    diagonal_periods,
                    term_baby_indices,
                    group_term_offsets,
                    parameter_shard.rns_primes.data() + q_count,
                    parameter_shard.rns_modulus_constants.data() + q_count,
                    group_count,
                    term_count,
                    p_count,
                    degree,
                    degree_power,
                    tile_begin,
                    tile_count,
                    initialize_accumulators);
            }
        }
        else
        {
            if (use_component_fused)
            {
                qp_plain_mul_accumulate_group_component_fused_kernel<
                    8, kCompressed>
                    <<<p_grid, block>>>(
                        group_p,
                        baby_p,
                        diagonal_p_ptrs,
                        diagonal_periods,
                        term_baby_indices,
                        group_term_offsets,
                        parameter_shard.rns_primes.data() + q_count,
                        parameter_shard.rns_modulus_constants.data() + q_count,
                        group_count,
                        term_count,
                        p_count,
                        degree,
                        degree_power,
                        tile_begin,
                        tile_count,
                        initialize_accumulators);
            }
            else
            {
                qp_plain_mul_accumulate_group_tiled_kernel<8, kCompressed>
                    <<<p_grid, block>>>(
                    group_p,
                    baby_p,
                    diagonal_p_ptrs,
                    diagonal_periods,
                    term_baby_indices,
                    group_term_offsets,
                    parameter_shard.rns_primes.data() + q_count,
                    parameter_shard.rns_modulus_constants.data() + q_count,
                    group_count,
                    term_count,
                    p_count,
                    degree,
                    degree_power,
                    tile_begin,
                    tile_count,
                    initialize_accumulators);
            }
        }
    }
    else
    {
        const std::size_t p_total = group_count * 2 * p_count * degree;
        const int p_grid = static_cast<int>(
            (p_total + block_size - 1) / block_size);
        qp_plain_mul_accumulate_kernel<kCompressed>
            <<<p_grid, block_size>>>(
            group_p,
            baby_p,
            diagonal_p_ptrs,
            diagonal_periods,
            term_baby_indices,
            group_term_offsets,
            parameter_shard.rns_primes.data() + q_count,
            parameter_shard.rns_modulus_constants.data() + q_count,
            group_count,
            term_count,
            p_count,
            degree,
            degree_power,
            tile_begin,
            tile_count,
            initialize_accumulators);
    }
    gpu_check_cuda(cudaGetLastError(), name);
    };

    if (compressed_plaintexts)
    {
        launch_for_layout(std::true_type{});
    }
    else
    {
        launch_for_layout(std::false_type{});
    }
}

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
    bool initialize_accumulators)
{
    const char *name =
        "launch_double_hoist_fused_baby_keyswitch_plain_accumulate";
    validate_parameter_shard(parameter_shard, degree, name);
    if (group_q == nullptr || group_p == nullptr ||
        digits_q == nullptr || digits_p == nullptr ||
        source_q0 == nullptr || source_q1 == nullptr ||
        host_galois_elts == nullptr || host_key_indices == nullptr ||
        host_term_indices == nullptr ||
        key_q0_ptrs == nullptr || key_p0_ptrs == nullptr ||
        key_q1_ptrs == nullptr || key_p1_ptrs == nullptr ||
        diagonal_q_ptrs == nullptr || diagonal_p_ptrs == nullptr ||
        (compressed_plaintexts && diagonal_periods == nullptr) ||
        group_count == 0 || group_count > kMaxFusedBabyGroups ||
        term_count == 0 || tile_count == 0 ||
        tile_count > kMaxDoubleHoistFusedBabySteps ||
        dnum == 0 || storage_dnum < dnum)
    {
        throw std::invalid_argument(std::string(name) + ": invalid argument");
    }
    gpu_check_cuda(cudaSetDevice(parameter_shard.device_id), name);

    BabyKeyMacKernelArguments arguments{};
    for (std::size_t group = 0; group < kMaxFusedBabyGroups; ++group)
    {
        for (std::size_t baby = 0;
             baby < kMaxDoubleHoistFusedBabySteps;
             ++baby)
        {
            arguments.term_indices[group][baby] =
                std::numeric_limits<std::uint32_t>::max();
        }
    }
    for (std::size_t baby = 0; baby < tile_count; ++baby)
    {
        arguments.galois_elts[baby] = host_galois_elts[baby];
        arguments.key_indices[baby] = host_key_indices[baby];
    }
    for (std::size_t group = 0; group < group_count; ++group)
    {
        for (std::size_t baby = 0; baby < tile_count; ++baby)
        {
            arguments.term_indices[group][baby] =
                host_term_indices[
                    group * kMaxDoubleHoistFusedBabySteps + baby];
        }
    }

    int block_size = 128;
    if (const char *raw = std::getenv(
            "POSEIDON_DOUBLE_HOIST_FUSED_BABY_BLOCK_SIZE"))
    {
        try
        {
            block_size = std::stoi(raw);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument(
                std::string(name) + ": invalid fused baby block size");
        }
        if (block_size != 64 && block_size != 128 && block_size != 256)
        {
            throw std::invalid_argument(
                std::string(name) +
                ": fused baby block size must be 64, 128, or 256");
        }
    }
    const unsigned int degree_power = log2_degree(degree, name);
    const std::size_t q_count = parameter_shard.hybrid_base_q_count;
    const std::size_t p_count = parameter_shard.hybrid_base_p_count;
    const int q_grid = static_cast<int>(
        (q_count * degree + block_size - 1) / block_size);
    const int p_grid = static_cast<int>(
        (p_count * degree + block_size - 1) / block_size);

    auto launch_for_layout = [&](auto layout_tag) {
        constexpr bool kCompressed = decltype(layout_tag)::value;
        if (group_count == 1)
        {
            fused_baby_keyswitch_plain_accumulate_kernel<1, true, kCompressed>
                <<<q_grid, block_size>>>(
                    group_q, digits_q, source_q0, source_q1, arguments,
                    key_q0_ptrs, key_q1_ptrs, diagonal_q_ptrs,
                    diagonal_periods, parameter_shard.hybrid_p_mod_q.data(),
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    group_count, term_count, tile_count, dnum, storage_dnum,
                    q_count, degree, degree_power, initialize_accumulators);
            fused_baby_keyswitch_plain_accumulate_kernel<1, false, kCompressed>
                <<<p_grid, block_size>>>(
                    group_p, digits_p, nullptr, nullptr, arguments,
                    key_p0_ptrs, key_p1_ptrs, diagonal_p_ptrs,
                    diagonal_periods, nullptr,
                    parameter_shard.rns_primes.data() + q_count,
                    parameter_shard.rns_modulus_constants.data() + q_count,
                    group_count, term_count, tile_count, dnum, storage_dnum,
                    p_count, degree, degree_power, initialize_accumulators);
        }
        else
        {
            fused_baby_keyswitch_plain_accumulate_kernel<4, true, kCompressed>
                <<<q_grid, block_size>>>(
                    group_q, digits_q, source_q0, source_q1, arguments,
                    key_q0_ptrs, key_q1_ptrs, diagonal_q_ptrs,
                    diagonal_periods, parameter_shard.hybrid_p_mod_q.data(),
                    parameter_shard.rns_primes.data(),
                    parameter_shard.rns_modulus_constants.data(),
                    group_count, term_count, tile_count, dnum, storage_dnum,
                    q_count, degree, degree_power, initialize_accumulators);
            fused_baby_keyswitch_plain_accumulate_kernel<4, false, kCompressed>
                <<<p_grid, block_size>>>(
                    group_p, digits_p, nullptr, nullptr, arguments,
                    key_p0_ptrs, key_p1_ptrs, diagonal_p_ptrs,
                    diagonal_periods, nullptr,
                    parameter_shard.rns_primes.data() + q_count,
                    parameter_shard.rns_modulus_constants.data() + q_count,
                    group_count, term_count, tile_count, dnum, storage_dnum,
                    p_count, degree, degree_power, initialize_accumulators);
        }
        gpu_check_cuda(cudaGetLastError(), name);
    };
    if (compressed_plaintexts)
    {
        launch_for_layout(std::true_type{});
    }
    else
    {
        launch_for_layout(std::false_type{});
    }
}

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
