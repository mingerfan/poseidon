#include "poseidon/gpu/kernels/gpu_rescale_kernels.h"

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
    return static_cast<GpuWord>(
        sum >= modulus ? sum - modulus : sum);
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

/* rescale = INTT + BConv + NTT ,其中NTT/INTT需要完整的coeff，而BConv可能需要完整的limb，但是由于只消除1个模数所以也无所谓*/
void validate_full_coeff_shard(
    const char *name,
    const GpuConstPolyShardView &shard,
    std::size_t degree)
{
    if (shard.ptr == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null shard pointer");
    }
    if (shard.coeff_begin != 0 || shard.coeff_count != degree)
    {
        throw std::invalid_argument(
            std::string(name) + ": shard must cover the full coefficient range");
    }
    if (shard.limb_count == 0)
    {
        throw std::invalid_argument(std::string(name) + ": empty limb range");
    }
}

void validate_full_coeff_shard(
    const char *name,
    const GpuPolyShardView &shard,
    std::size_t degree)
{
    if (shard.ptr == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null shard pointer");
    }
    if (shard.coeff_begin != 0 || shard.coeff_count != degree)
    {
        throw std::invalid_argument(
            std::string(name) + ": shard must cover the full coefficient range");
    }
    if (shard.limb_count == 0)
    {
        throw std::invalid_argument(std::string(name) + ": empty limb range");
    }
}

void validate_parameter_tables(
    const char *name,
    const GpuParameterShard &parameter_shard)
{
    if (parameter_shard.limb_begin != 0)
    {
        throw std::invalid_argument(
            std::string(name) + ": first implementation requires a full parameter shard");
    }
    if (parameter_shard.q_last == 0)
    {
        throw std::invalid_argument(std::string(name) + ": missing q_last modulus");
    }
    if (parameter_shard.rns_primes.data() == nullptr ||
        parameter_shard.rns_modulus_constants.data() == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null modulus table pointer");
    }
    if (parameter_shard.half_q_last_mod_q.data() == nullptr ||
        parameter_shard.inv_q_last_mod_q.data() == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null rescale table pointer");
    }
}

void validate_limb_range_covered(
    const char *name,
    const GpuParameterShard &parameter_shard,
    std::size_t limb_begin,
    std::size_t limb_count)
{
    if (limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument(std::string(name) + ": parameter shard does not cover limb range");
    }

    const std::size_t parameter_offset =
        limb_begin - parameter_shard.limb_begin;
    if (parameter_offset + limb_count > parameter_shard.rns_primes.size() ||
        parameter_offset + limb_count > parameter_shard.rns_modulus_constants.size())
    {
        throw std::invalid_argument(std::string(name) + ": modulus tables do not cover limb range");
    }
}

void validate_build_q_last_rescale_correction(
    const GpuPolyShardView &correction_shard,
    const GpuConstPolyShardView &q_last_coeff_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    constexpr const char *name =
        "launch_build_q_last_rescale_correction_poly_shard";

    validate_full_coeff_shard(name, correction_shard, degree);
    validate_full_coeff_shard(name, q_last_coeff_shard, degree);
    validate_parameter_tables(name, parameter_shard);

    if (q_last_coeff_shard.limb_count != 1)
    {
        throw std::invalid_argument(std::string(name) + ": q_last shard must contain one limb");
    }
    if (correction_shard.device_id != q_last_coeff_shard.device_id ||
        correction_shard.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument(std::string(name) + ": device mismatch");
    }
    if (correction_shard.limb_begin + correction_shard.limb_count >
        q_last_coeff_shard.limb_begin)
    {
        throw std::invalid_argument(std::string(name) + ": correction must target limbs below q_last");
    }
    if (q_last_coeff_shard.limb_begin >= parameter_shard.rns_primes.size())
    {
        throw std::invalid_argument(std::string(name) + ": q_last limb is out of range");
    }
    if (correction_shard.limb_begin + correction_shard.limb_count >
            parameter_shard.half_q_last_mod_q.size() ||
        correction_shard.limb_begin + correction_shard.limb_count >
            parameter_shard.inv_q_last_mod_q.size())
    {
        throw std::invalid_argument(std::string(name) + ": rescale tables do not cover limb range");
    }

    validate_limb_range_covered(
        name,
        parameter_shard,
        correction_shard.limb_begin,
        correction_shard.limb_count);
    validate_limb_range_covered(
        name,
        parameter_shard,
        q_last_coeff_shard.limb_begin,
        q_last_coeff_shard.limb_count);
}

void validate_apply_q_last_rescale_correction(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuConstPolyShardView &correction_ntt_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    constexpr const char *name =
        "launch_apply_q_last_rescale_correction_poly_shard";

    validate_full_coeff_shard(name, destination_shard, degree);
    validate_full_coeff_shard(name, source_shard, degree);
    validate_full_coeff_shard(name, correction_ntt_shard, degree);
    validate_parameter_tables(name, parameter_shard);

    const bool same_shape =
        destination_shard.limb_begin == source_shard.limb_begin &&
        destination_shard.limb_count == source_shard.limb_count &&
        destination_shard.limb_begin == correction_ntt_shard.limb_begin &&
        destination_shard.limb_count == correction_ntt_shard.limb_count;
    if (!same_shape)
    {
        throw std::invalid_argument(std::string(name) + ": shard shape mismatch");
    }

    if (destination_shard.device_id != source_shard.device_id ||
        destination_shard.device_id != correction_ntt_shard.device_id ||
        destination_shard.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument(std::string(name) + ": device mismatch");
    }
    if (destination_shard.limb_begin + destination_shard.limb_count >
            parameter_shard.inv_q_last_mod_q.size() ||
        destination_shard.limb_begin + destination_shard.limb_count >
            parameter_shard.half_q_last_mod_q.size())
    {
        throw std::invalid_argument(std::string(name) + ": rescale tables do not cover limb range");
    }

    validate_limb_range_covered(
        name,
        parameter_shard,
        destination_shard.limb_begin,
        destination_shard.limb_count);
}

/*rescale的关键操作，将最后需要消除的模数先模升到全部q模数，然后和原模数链做差值*/
__global__ void build_q_last_rescale_correction_kernel(
    GpuWord *correction,/*输出，顺序是[local_limb][coeff]也就是local_limb个poly多项式*/
    const GpuWord *q_last_coeff,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *half_q_last_mod_q,
    GpuWord q_last,
    GpuWord half_q_last,
    std::size_t parameter_limb_begin,
    std::size_t q_limb_begin,
    std::size_t limb_count,
    std::size_t degree)
{
    /*每个线程处理一个coeff*/
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / degree;
    const std::size_t coeff_index = tid % degree;
    const std::size_t global_q_limb = q_limb_begin + local_limb;
    const std::size_t table_limb = global_q_limb - parameter_limb_begin;

    const GpuWord qi = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];

    /* 首先加half_q_last是为了四舍五入保证精度，以往理解中直接source_i mod q_last所得到的只是floor的结果*/
    const GpuWord rounded_q_last = add_mod(
        q_last_coeff[coeff_index],
        half_q_last,
        q_last);
    /* [q_last对应的密文分量 + half_q_last]对q_last取模，这里替换了原本直接的q_last对应的密文分量，可以起到round的作用，这样的结果再分别对qi取模得到模数链对应分量 */
    const GpuWord rounded_mod_qi = barrett_reduce_u64_u32(
        rounded_q_last,
        qi,
        barrett_ratio);/*有必要用巴雷特约减吗？*/
    /* 计算的结果再减去一个half_q_last，因为前面增加了一个所以减掉 */
    correction[tid] = sub_mod(
        rounded_mod_qi,
        half_q_last_mod_q[global_q_limb],
        qi);
}

__global__ void apply_q_last_rescale_correction_kernel(
    GpuWord *destination,
    const GpuWord *source,
    const GpuWord *correction_ntt,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *inv_q_last_mod_q,
    std::size_t parameter_limb_begin,
    std::size_t q_limb_begin,
    std::size_t limb_count,
    std::size_t degree)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / degree;
    const std::size_t global_q_limb = q_limb_begin + local_limb;
    const std::size_t table_limb = global_q_limb - parameter_limb_begin;

    const GpuWord qi = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    /*前面模升后的最后一个模数经过NTT，再与sorce作差*/
    const GpuWord difference = sub_mod(
        source[tid],
        correction_ntt[tid],
        qi);
    /*作差结果乘p逆，在rescale里实际上对应的就是q_last的逆元*/
    destination[tid] = mul_mod(
        difference,
        inv_q_last_mod_q[global_q_limb],
        qi,
        barrett_ratio);
}

__global__ void reconstruct_q_last_two_centered_remainders_kernel(
    GpuWide *centered_remainders,
    const GpuWord *dropped_coefficients,
    GpuWord q_second_last,
    GpuWord q_last,
    GpuWord inv_q_last_mod_q_second_last,
    GpuWide q_second_last_barrett,
    std::size_t component_count,
    std::size_t degree)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = component_count * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t component = tid / degree;
    const std::size_t coeff = tid % degree;
    const std::size_t dropped_offset = component * 2 * degree;
    const GpuWord residue_second_last =
        dropped_coefficients[dropped_offset + coeff];
    const GpuWord residue_last =
        dropped_coefficients[dropped_offset + degree + coeff];

    const GpuWord residue_last_mod_second_last =
        barrett_reduce_u64_u32(
            static_cast<GpuWide>(residue_last),
            q_second_last,
            q_second_last_barrett);
    const GpuWord delta = sub_mod(
        residue_second_last,
        residue_last_mod_second_last,
        q_second_last);
    const GpuWord multiplier = mul_mod(
        delta,
        inv_q_last_mod_q_second_last,
        q_second_last,
        q_second_last_barrett);

    centered_remainders[tid] =
        static_cast<GpuWide>(residue_last) +
        static_cast<GpuWide>(q_last) * multiplier;
}

__global__ void build_q_last_two_rescale_correction_batch_kernel(
    GpuWord *correction,
    const GpuWide *centered_remainders,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *q_last_two_product_mod_q,
    GpuWide half_q_last_two_product,
    std::size_t parameter_limb_begin,
    std::size_t component_count,
    std::size_t destination_q_count,
    std::size_t degree)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t values_per_component = destination_q_count * degree;
    const std::size_t total = component_count * values_per_component;
    if (tid >= total)
    {
        return;
    }

    const std::size_t component = tid / values_per_component;
    const std::size_t local = tid % values_per_component;
    const std::size_t q_limb = local / degree;
    const std::size_t coeff = local % degree;
    const std::size_t table_limb = q_limb - parameter_limb_begin;
    const GpuWord qi = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWide remainder =
        centered_remainders[component * degree + coeff];
    const GpuWord remainder_mod_qi =
        barrett_reduce_u64_u32(remainder, qi, barrett_ratio);

    correction[tid] =
        remainder > half_q_last_two_product
            ? sub_mod(
                  remainder_mod_qi,
                  q_last_two_product_mod_q[q_limb],
                  qi)
            : remainder_mod_qi;
}

__global__ void build_q_last_two_rescale_correction_batch_fused_kernel(
    GpuWord *correction,
    const GpuWord *dropped_coefficients,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *q_last_two_product_mod_q,
    GpuWord q_second_last,
    GpuWord q_last,
    GpuWord inv_q_last_mod_q_second_last,
    GpuWide q_second_last_barrett,
    GpuWide half_q_last_two_product,
    std::size_t parameter_limb_begin,
    std::size_t component_count,
    std::size_t destination_q_count,
    std::size_t degree)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t values_per_component = destination_q_count * degree;
    const std::size_t total = component_count * values_per_component;
    if (tid >= total)
    {
        return;
    }

    const std::size_t component = tid / values_per_component;
    const std::size_t local = tid % values_per_component;
    const std::size_t q_limb = local / degree;
    const std::size_t coeff = local % degree;
    const std::size_t dropped_offset = component * 2 * degree;
    const GpuWord residue_second_last =
        dropped_coefficients[dropped_offset + coeff];
    const GpuWord residue_last =
        dropped_coefficients[dropped_offset + degree + coeff];

    const GpuWord residue_last_mod_second_last =
        barrett_reduce_u64_u32(
            static_cast<GpuWide>(residue_last),
            q_second_last,
            q_second_last_barrett);
    const GpuWord delta = sub_mod(
        residue_second_last,
        residue_last_mod_second_last,
        q_second_last);
    const GpuWord multiplier = mul_mod(
        delta,
        inv_q_last_mod_q_second_last,
        q_second_last,
        q_second_last_barrett);
    const GpuWide remainder =
        static_cast<GpuWide>(residue_last) +
        static_cast<GpuWide>(q_last) * multiplier;

    const std::size_t table_limb = q_limb - parameter_limb_begin;
    const GpuWord qi = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord remainder_mod_qi =
        barrett_reduce_u64_u32(remainder, qi, barrett_ratio);

    correction[tid] =
        remainder > half_q_last_two_product
            ? sub_mod(
                  remainder_mod_qi,
                  q_last_two_product_mod_q[q_limb],
                  qi)
            : remainder_mod_qi;
}

__global__ void apply_q_last_two_rescale_correction_batch2_kernel(
    GpuWord *destination0,
    GpuWord *destination1,
    const GpuWord *source0,
    const GpuWord *source1,
    const GpuWord *correction_ntt,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *inv_q_last_two_product_mod_q,
    std::size_t parameter_limb_begin,
    std::size_t destination_q_count,
    std::size_t degree)
{
    const std::size_t values_per_component = destination_q_count * degree;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= values_per_component * 2)
    {
        return;
    }

    const bool second_component = tid >= values_per_component;
    const std::size_t local =
        second_component ? tid - values_per_component : tid;
    const std::size_t q_limb = local / degree;
    const std::size_t table_limb = q_limb - parameter_limb_begin;
    const GpuWord qi = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord *source = second_component ? source1 : source0;
    GpuWord *destination =
        second_component ? destination1 : destination0;
    const GpuWord difference = sub_mod(
        source[local],
        correction_ntt[tid],
        qi);
    destination[local] = mul_mod(
        difference,
        inv_q_last_two_product_mod_q[q_limb],
        qi,
        barrett_ratio);
}

}  // anonymous namespace

void launch_build_q_last_rescale_correction_poly_shard(
    const GpuPolyShardView &correction_shard,
    const GpuConstPolyShardView &q_last_coeff_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_build_q_last_rescale_correction(
        correction_shard,
        q_last_coeff_shard,
        parameter_shard,
        degree);

    gpu_check_cuda(
        cudaSetDevice(correction_shard.device_id),
        "launch_build_q_last_rescale_correction_poly_shard cudaSetDevice");

    constexpr int block_size = 256;
    const std::size_t total = correction_shard.limb_count * degree;
    const int grid_size = static_cast<int>(
        (total + block_size - 1) / block_size);

    build_q_last_rescale_correction_kernel<<<grid_size, block_size>>>(
        correction_shard.ptr,
        q_last_coeff_shard.ptr,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.half_q_last_mod_q.data(),
        parameter_shard.q_last,
        parameter_shard.half_q_last,
        parameter_shard.limb_begin,
        correction_shard.limb_begin,
        correction_shard.limb_count,
        degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_build_q_last_rescale_correction_poly_shard kernel launch");
}

void launch_apply_q_last_rescale_correction_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuConstPolyShardView &correction_ntt_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_apply_q_last_rescale_correction(
        destination_shard,
        source_shard,
        correction_ntt_shard,
        parameter_shard,
        degree);

    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_apply_q_last_rescale_correction_poly_shard cudaSetDevice");

    constexpr int block_size = 256;
    const std::size_t total = destination_shard.limb_count * degree;
    const int grid_size = static_cast<int>(
        (total + block_size - 1) / block_size);

    apply_q_last_rescale_correction_kernel<<<grid_size, block_size>>>(
        destination_shard.ptr,
        source_shard.ptr,
        correction_ntt_shard.ptr,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.inv_q_last_mod_q.data(),
        parameter_shard.limb_begin,
        destination_shard.limb_begin,
        destination_shard.limb_count,
        degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_apply_q_last_rescale_correction_poly_shard kernel launch");
}

void launch_reconstruct_q_last_two_centered_remainders(
    GpuWide *centered_remainders,
    const GpuWord *dropped_coefficients,
    std::size_t component_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    constexpr const char *name =
        "launch_reconstruct_q_last_two_centered_remainders";
    if (centered_remainders == nullptr || dropped_coefficients == nullptr ||
        component_count == 0 || degree == 0)
    {
        throw std::invalid_argument(std::string(name) + ": invalid shape");
    }
    if (parameter_shard.q_second_last == 0 ||
        parameter_shard.q_last == 0 ||
        parameter_shard.q_last_two_product == 0 ||
        parameter_shard.inv_q_last_mod_q_second_last == 0)
    {
        throw std::invalid_argument(
            std::string(name) + ": missing rescale_x2 constants");
    }
    if (parameter_shard.q_second_last_modulus_constant == 0)
    {
        throw std::invalid_argument(
            std::string(name) + ": second-last modulus is out of range");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_reconstruct_q_last_two_centered_remainders cudaSetDevice");
    constexpr int block_size = 256;
    const std::size_t total = component_count * degree;
    const int grid_size = static_cast<int>(
        (total + block_size - 1) / block_size);
    reconstruct_q_last_two_centered_remainders_kernel<<<
        grid_size,
        block_size>>>(
        centered_remainders,
        dropped_coefficients,
        parameter_shard.q_second_last,
        parameter_shard.q_last,
        parameter_shard.inv_q_last_mod_q_second_last,
        parameter_shard.q_second_last_modulus_constant,
        component_count,
        degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_reconstruct_q_last_two_centered_remainders kernel launch");
}

void launch_build_q_last_two_rescale_correction_batch(
    GpuWord *correction,
    const GpuWide *centered_remainders,
    std::size_t component_count,
    std::size_t destination_q_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    constexpr const char *name =
        "launch_build_q_last_two_rescale_correction_batch";
    if (correction == nullptr || centered_remainders == nullptr ||
        component_count == 0 || destination_q_count == 0 || degree == 0)
    {
        throw std::invalid_argument(std::string(name) + ": invalid shape");
    }
    if (parameter_shard.limb_begin != 0 ||
        parameter_shard.q_last_two_product_mod_q.size() <
            destination_q_count ||
        parameter_shard.rns_primes.size() < destination_q_count ||
        parameter_shard.rns_modulus_constants.size() <
            destination_q_count)
    {
        throw std::invalid_argument(
            std::string(name) + ": parameter tables do not cover output");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_build_q_last_two_rescale_correction_batch cudaSetDevice");
    constexpr int block_size = 256;
    const std::size_t total =
        component_count * destination_q_count * degree;
    const int grid_size = static_cast<int>(
        (total + block_size - 1) / block_size);
    build_q_last_two_rescale_correction_batch_kernel<<<
        grid_size,
        block_size>>>(
        correction,
        centered_remainders,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.q_last_two_product_mod_q.data(),
        parameter_shard.half_q_last_two_product,
        parameter_shard.limb_begin,
        component_count,
        destination_q_count,
        degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_build_q_last_two_rescale_correction_batch kernel launch");
}

void launch_build_q_last_two_rescale_correction_batch_fused(
    GpuWord *correction,
    const GpuWord *dropped_coefficients,
    std::size_t component_count,
    std::size_t destination_q_count,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    constexpr const char *name =
        "launch_build_q_last_two_rescale_correction_batch_fused";
    if (correction == nullptr || dropped_coefficients == nullptr ||
        component_count == 0 || destination_q_count == 0 || degree == 0)
    {
        throw std::invalid_argument(std::string(name) + ": invalid shape");
    }
    if (parameter_shard.limb_begin != 0 ||
        parameter_shard.q_second_last == 0 ||
        parameter_shard.q_last == 0 ||
        parameter_shard.q_last_two_product == 0 ||
        parameter_shard.inv_q_last_mod_q_second_last == 0 ||
        parameter_shard.q_second_last_modulus_constant == 0 ||
        parameter_shard.q_last_two_product_mod_q.size() <
            destination_q_count ||
        parameter_shard.rns_primes.size() < destination_q_count ||
        parameter_shard.rns_modulus_constants.size() <
            destination_q_count)
    {
        throw std::invalid_argument(
            std::string(name) + ": parameter tables do not cover output");
    }

    gpu_check_cuda(
        cudaSetDevice(parameter_shard.device_id),
        "launch_build_q_last_two_rescale_correction_batch_fused cudaSetDevice");
    constexpr int block_size = 256;
    const std::size_t total =
        component_count * destination_q_count * degree;
    const int grid_size = static_cast<int>(
        (total + block_size - 1) / block_size);
    build_q_last_two_rescale_correction_batch_fused_kernel<<<
        grid_size,
        block_size>>>(
        correction,
        dropped_coefficients,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.q_last_two_product_mod_q.data(),
        parameter_shard.q_second_last,
        parameter_shard.q_last,
        parameter_shard.inv_q_last_mod_q_second_last,
        parameter_shard.q_second_last_modulus_constant,
        parameter_shard.half_q_last_two_product,
        parameter_shard.limb_begin,
        component_count,
        destination_q_count,
        degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_build_q_last_two_rescale_correction_batch_fused kernel launch");
}

void launch_apply_q_last_two_rescale_correction_batch2(
    const GpuPolyShardView &destination0,
    const GpuPolyShardView &destination1,
    const GpuConstPolyShardView &source0,
    const GpuConstPolyShardView &source1,
    const GpuWord *correction_ntt,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    constexpr const char *name =
        "launch_apply_q_last_two_rescale_correction_batch2";
    validate_full_coeff_shard(name, destination0, degree);
    validate_full_coeff_shard(name, destination1, degree);
    validate_full_coeff_shard(name, source0, degree);
    validate_full_coeff_shard(name, source1, degree);
    const std::size_t destination_q_count = destination0.limb_count;
    if (correction_ntt == nullptr ||
        destination0.limb_begin != 0 ||
        destination1.limb_begin != 0 ||
        source0.limb_begin != 0 ||
        source1.limb_begin != 0 ||
        destination1.limb_count != destination_q_count ||
        source0.limb_count != destination_q_count ||
        source1.limb_count != destination_q_count ||
        destination0.device_id != destination1.device_id ||
        destination0.device_id != source0.device_id ||
        destination0.device_id != source1.device_id ||
        destination0.device_id != parameter_shard.device_id ||
        parameter_shard.inv_q_last_two_product_mod_q.size() <
            destination_q_count)
    {
        throw std::invalid_argument(std::string(name) + ": shape mismatch");
    }

    gpu_check_cuda(
        cudaSetDevice(destination0.device_id),
        "launch_apply_q_last_two_rescale_correction_batch2 cudaSetDevice");
    constexpr int block_size = 256;
    const std::size_t total = 2 * destination_q_count * degree;
    const int grid_size = static_cast<int>(
        (total + block_size - 1) / block_size);
    apply_q_last_two_rescale_correction_batch2_kernel<<<
        grid_size,
        block_size>>>(
        destination0.ptr,
        destination1.ptr,
        source0.ptr,
        source1.ptr,
        correction_ntt,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.inv_q_last_two_product_mod_q.data(),
        parameter_shard.limb_begin,
        destination_q_count,
        degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_apply_q_last_two_rescale_correction_batch2 kernel launch");
}

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
