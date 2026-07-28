#include "poseidon/gpu/kernels/gpu_double_hoist_kernels.h"

#include <cuda_runtime_api.h>

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

constexpr std::size_t kMaxFusedGiantGroups = 64;

struct GiantGroupKernelArguments
{
    const GpuWord *digit_ptrs[kMaxFusedGiantGroups];
    std::uint32_t group_indices[kMaxFusedGiantGroups];
    std::uint32_t galois_elts[kMaxFusedGiantGroups];
    std::uint32_t key_indices[kMaxFusedGiantGroups];
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

__global__ void pre_rotated_keymul_batch_kernel(
    GpuWord *destination0,
    GpuWord *destination1,
    const GpuWord *digits,
    const GpuWord *const *key0_ptrs,
    const GpuWord *const *key1_ptrs,
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
    destination0[tid] = overwrite
        ? accumulator0
        : add_mod(destination0[tid], accumulator0, modulus);
    destination1[tid] = overwrite
        ? accumulator1
        : add_mod(destination1[tid], accumulator1, modulus);
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
    const GpuWord modulus = moduli[limb];
    const GpuWide ratio = barrett[limb];
    GpuWord accumulator0 = 0;
    GpuWord accumulator1 = 0;
    {
        const std::uint32_t galois_elt =
            group_args.galois_elts[active_group];
        const std::uint32_t reversed =
            reverse_bits_limited(degree_u32 + coeff, degree_power + 1);
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

__global__ void qp_plain_mul_accumulate_kernel(
    GpuWord *group_values,
    const GpuWord *baby_values,
    const GpuWord *const *diagonal_ptrs,
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
    std::size_t tile_count)
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

    GpuWord accumulator = group_values[tid];
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
        const GpuWord diagonal = diagonal_ptrs[term][limb * degree + coeff];
        const GpuWord product = multiply_mod(
            baby_values[baby_offset],
            diagonal,
            modulus,
            ratio);
        accumulator = add_mod(accumulator, product, modulus);
    }

    group_values[tid] = accumulator;
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
    gpu_check_cuda(cudaSetDevice(parameter_shard.device_id), name);

    constexpr int block_size = 256;
    const unsigned int degree_power = log2_degree(degree, name);
    const std::size_t q_count = parameter_shard.hybrid_base_q_count;
    const std::size_t p_count = parameter_shard.hybrid_base_p_count;
    const int q_grid = static_cast<int>(
        (q_count * degree + block_size - 1) / block_size);
    pre_rotated_keymul_batch_kernel<<<q_grid, block_size>>>(
        destination_q0,
        destination_q1,
        digits_q,
        key_q0_ptrs,
        key_q1_ptrs,
        galois_elt,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        dnum,
        q_count,
        degree,
        degree_power,
        overwrite);
    gpu_check_cuda(cudaGetLastError(), name);

    const int p_grid = static_cast<int>(
        (p_count * degree + block_size - 1) / block_size);
    pre_rotated_keymul_batch_kernel<<<p_grid, block_size>>>(
        destination_p0,
        destination_p1,
        digits_p,
        key_p0_ptrs,
        key_p1_ptrs,
        galois_elt,
        parameter_shard.rns_primes.data() + q_count,
        parameter_shard.rns_modulus_constants.data() + q_count,
        dnum,
        p_count,
        degree,
        degree_power,
        overwrite);
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
    const std::uint32_t *term_baby_indices,
    const std::uint32_t *group_term_offsets,
    std::size_t group_count,
    std::size_t term_count,
    std::size_t tile_begin,
    std::size_t tile_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    const char *name =
        "launch_double_hoist_qp_plain_mul_accumulate_groups";
    validate_parameter_shard(parameter_shard, degree, name);
    if (group_q == nullptr || group_p == nullptr ||
        baby_q == nullptr || baby_p == nullptr ||
        diagonal_q_ptrs == nullptr || diagonal_p_ptrs == nullptr ||
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

    const std::size_t q_total = group_count * 2 * q_count * degree;
    const int q_grid = static_cast<int>(
        (q_total + block_size - 1) / block_size);
    qp_plain_mul_accumulate_kernel<<<q_grid, block_size>>>(
        group_q,
        baby_q,
        diagonal_q_ptrs,
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
        tile_count);
    gpu_check_cuda(cudaGetLastError(), name);

    const std::size_t p_total = group_count * 2 * p_count * degree;
    const int p_grid = static_cast<int>(
        (p_total + block_size - 1) / block_size);
    qp_plain_mul_accumulate_kernel<<<p_grid, block_size>>>(
        group_p,
        baby_p,
        diagonal_p_ptrs,
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
        tile_count);
    gpu_check_cuda(cudaGetLastError(), name);
}

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
