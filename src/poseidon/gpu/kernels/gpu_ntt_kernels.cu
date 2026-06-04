#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"

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

bool is_power_of_two(std::size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

void validate_ntt_launch_shape(
    const char *name,
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool inverse)
{
    if (destination_shard.ptr == nullptr ||
        source_shard.ptr == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null data pointer");
    }
    if (parameter_shard.rns_primes.data() == nullptr ||
        parameter_shard.rns_modulus_constants.data() == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null modulus table pointer");
    }
    if (parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null NTT table pointer");
    }
    if (inverse &&
        (parameter_shard.intt_tables.data() == nullptr ||
         parameter_shard.inv_degree_modulo.data() == nullptr))
    {
        throw std::invalid_argument(std::string(name) + ": null inverse NTT table pointer");
    }
    if (!inverse && parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null forward NTT table pointer");
    }
    if (!is_power_of_two(degree) || degree < 2)
    {
        throw std::invalid_argument(std::string(name) + ": degree must be a power of two");
    }
    if (destination_shard.device_id != source_shard.device_id ||
        destination_shard.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument(std::string(name) + ": device mismatch");
    }
    if (destination_shard.limb_begin != source_shard.limb_begin ||
        destination_shard.limb_count != source_shard.limb_count ||
        destination_shard.coeff_begin != source_shard.coeff_begin ||
        destination_shard.coeff_count != source_shard.coeff_count)
    {
        throw std::invalid_argument(std::string(name) + ": shard shape mismatch");
    }
    if (destination_shard.coeff_begin != 0 ||
        destination_shard.coeff_count != degree)
    {
        throw std::invalid_argument(
            std::string(name) + ": first implementation requires full coefficient shards");
    }
    if (destination_shard.limb_count == 0)
    {
        throw std::invalid_argument(std::string(name) + ": empty limb range");
    }
    if (destination_shard.limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument(std::string(name) + ": parameter shard does not cover limb range");
    }

    const std::size_t modulus_offset =
        destination_shard.limb_begin - parameter_shard.limb_begin;
    if (modulus_offset + destination_shard.limb_count >
            parameter_shard.rns_primes.size() ||
        modulus_offset + destination_shard.limb_count >
            parameter_shard.rns_modulus_constants.size())
    {
        throw std::invalid_argument(std::string(name) + ": RNS modulus tables do not cover limb range");
    }

    const std::size_t roots_size =
        (modulus_offset + destination_shard.limb_count) * degree;
    if (parameter_shard.ntt_tables.size() < roots_size)
    {
        throw std::invalid_argument(std::string(name) + ": NTT roots do not cover limb range");
    }
    if (inverse &&
        (parameter_shard.intt_tables.size() < roots_size ||
         parameter_shard.inv_degree_modulo.size() <
             modulus_offset + destination_shard.limb_count))
    {
        throw std::invalid_argument(std::string(name) + ": inverse NTT tables do not cover limb range");
    }
}

// NTT的一次迭代计算，forward NTT循环发射kernel，NTT的计算需要完整的coeff。
__global__ void forward_ntt_stage_kernel(
    GpuWord *values,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    std::size_t modulus_offset,/*？*/
    std::size_t limb_count,
    std::size_t degree, /*NTT需要完整的多项式，所以直接输入多项式阶数而非片段的coeff*/
    std::size_t m,
    std::size_t gap)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 每个线程独立计算一个BU，通过degree >> 1得到所需的蝶形算子
    const std::size_t butterflies_per_limb = degree >> 1;
    // RNS分量*对应的BU个数=所需启动的总线程
    const std::size_t total = limb_count * butterflies_per_limb;
    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / butterflies_per_limb;
    const std::size_t local_butterfly = tid % butterflies_per_limb;
    const std::size_t group = local_butterfly / gap; /*group代表在当前迭代轮次下的每组不同twiddle的个数*/
    const std::size_t j = local_butterfly % gap;/*比如NTT-8三次迭代，gap=4->2->1。第一次j=0,1,2,3*/
    const std::size_t table_limb = modulus_offset + local_limb;
    /*butterfly的两个输入位置,当前的degree+2*local_butterfly*/
    const std::size_t x_index = local_limb * degree + group * (gap << 1) + j;
    const std::size_t y_index = x_index + gap;

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    /*当前 stage、当前 group 要用的 twiddle factor，其中m为group个数；cpu侧root存储逻辑就是：stage1:root1；stage2:root2 root3...*/
    const GpuWord root = roots[table_limb * degree + m + group];

    const GpuWord u = values[x_index];
    const GpuWord v = mul_mod(values[y_index], root, modulus, barrett_ratio);

    values[x_index] = add_mod(u, v, modulus);
    values[y_index] = sub_mod(u, v, modulus);
}

// INTT,NTT的逆操作，最后还需要multiply_inv_degree_kernel整体除以N
__global__ void inverse_ntt_stage_kernel(
    GpuWord *values,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    std::size_t modulus_offset,
    std::size_t limb_count,
    std::size_t degree,
    std::size_t m,
    std::size_t gap)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t butterflies_per_limb = degree >> 1;
    const std::size_t total = limb_count * butterflies_per_limb;
    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / butterflies_per_limb;
    const std::size_t local_butterfly = tid % butterflies_per_limb;
    const std::size_t group = local_butterfly / gap;
    const std::size_t j = local_butterfly % gap;
    const std::size_t table_limb = modulus_offset + local_limb;
    const std::size_t root_base = degree - (m << 1) + 1;
    const std::size_t x_index =
        local_limb * degree + group * (gap << 1) + j;
    const std::size_t y_index = x_index + gap;

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord root = roots[table_limb * degree + root_base + group];

    const GpuWord u = values[x_index];
    const GpuWord v = values[y_index];

    values[x_index] = add_mod(u, v, modulus);
    values[y_index] = mul_mod(
        sub_mod(u, v, modulus),
        root,
        modulus,
        barrett_ratio);
}

__global__ void multiply_inv_degree_kernel(
    GpuWord *values,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *inv_degree_modulo,
    std::size_t modulus_offset,
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
    const std::size_t table_limb = modulus_offset + local_limb;
    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];

    values[tid] = mul_mod(
        values[tid],
        inv_degree_modulo[table_limb],
        modulus,
        barrett_ratio);
}

// 计算逻辑是原位置计算的，也就是说需要输出结果有输入的初始值，然后迭代为中间结果，最终计算完成//？有优势吗
void copy_source_to_destination(
    const char *name,
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard)
{
    if (destination_shard.ptr == source_shard.ptr)
    {
        return;
    }

    const std::size_t count =
        destination_shard.limb_count * destination_shard.coeff_count;
    gpu_check_cuda(
        cudaMemcpy(
            destination_shard.ptr,
            source_shard.ptr,
            count * sizeof(GpuWord),
            cudaMemcpyDeviceToDevice),
        name);
}

void launch_forward_stages(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    const std::size_t modulus_offset =
        shard.limb_begin - parameter_shard.limb_begin;
    const std::size_t total_butterflies =
        shard.limb_count * (degree >> 1);

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total_butterflies + block_size - 1) / block_size);

    for (std::size_t m = 1, gap = degree >> 1;
         m < degree;
         m <<= 1, gap >>= 1)
    {
        forward_ntt_stage_kernel<<<grid_size, block_size>>>(
            shard.ptr,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.ntt_tables.data(),
            modulus_offset,
            shard.limb_count,
            degree,
            m,
            gap);
        gpu_check_cuda(
            cudaGetLastError(),
            "launch_forward_ntt_poly_shard stage kernel launch");
    }
}

void launch_inverse_stages(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    const std::size_t modulus_offset =
        shard.limb_begin - parameter_shard.limb_begin;
    const std::size_t total_butterflies =
        shard.limb_count * (degree >> 1);

    constexpr int block_size = 256;
    int grid_size = static_cast<int>(
        (total_butterflies + block_size - 1) / block_size);

    for (std::size_t m = degree >> 1, gap = 1;; m >>= 1, gap <<= 1)
    {
        inverse_ntt_stage_kernel<<<grid_size, block_size>>>(
            shard.ptr,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.intt_tables.data(),
            modulus_offset,
            shard.limb_count,
            degree,
            m,
            gap);
        gpu_check_cuda(
            cudaGetLastError(),
            "launch_inverse_ntt_poly_shard stage kernel launch");

        if (m == 1)
        {
            break;
        }
    }

    const std::size_t total_values = shard.limb_count * degree;
    grid_size = static_cast<int>(
        (total_values + block_size - 1) / block_size);
    multiply_inv_degree_kernel<<<grid_size, block_size>>>(
        shard.ptr,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.inv_degree_modulo.data(),
        modulus_offset,
        shard.limb_count,
        degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_inverse_ntt_poly_shard inv degree kernel launch");
}

}  // anonymous namespace

void launch_forward_ntt_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_ntt_launch_shape(
        "launch_forward_ntt_poly_shard",
        destination_shard,
        source_shard,
        parameter_shard,
        degree,
        false);

    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_forward_ntt_poly_shard cudaSetDevice");
    copy_source_to_destination(
        "launch_forward_ntt_poly_shard cudaMemcpyDeviceToDevice",
        destination_shard,
        source_shard);

    launch_forward_stages(destination_shard, parameter_shard, degree);
}

void launch_inverse_ntt_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_ntt_launch_shape(
        "launch_inverse_ntt_poly_shard",
        destination_shard,
        source_shard,
        parameter_shard,
        degree,
        true);

    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_inverse_ntt_poly_shard cudaSetDevice");
    copy_source_to_destination(
        "launch_inverse_ntt_poly_shard cudaMemcpyDeviceToDevice",
        destination_shard,
        source_shard);

    launch_inverse_stages(destination_shard, parameter_shard, degree);
}

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
