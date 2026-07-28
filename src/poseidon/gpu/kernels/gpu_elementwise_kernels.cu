#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"

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

/**
 * @brief CUDA kernel skeleton for modular addition.
 *
 * This function is the real GPU-side parallel function.
 *
 * Current stage:
 * - Kernel body is intentionally left as TODO.
 */
__global__ void add_poly_shard_kernel(
    GpuWord *destination_values,
    const GpuWord *left_values,
    const GpuWord *right_values,
    const GpuWord *q_primes,
    std::size_t modulus_offset,
    std::size_t limb_count,
    std::size_t coeff_count)
{
    // destination_values：gpu上输出的数组指针，kernel要将结果写到该指针里
    // left/right_values：指的是相加的密文1和密文2，低位求和
    // block*块内线程计算相同的内容，x来索引具体位置，所以表示并行总运行的线程数？
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 当前计算密文的模数个数和每个小密文的poly数，表示总计算量
    std::size_t total = limb_count * coeff_count;

    // 表示当前计算所需线程小于实际启动的线程，所以将溢出的线程返回，避免闲置
    if (tid >= total)
    {
        return;
    }

    // Shard data is packed in limb-major order:
    // [local limb 0 coeffs][local limb 1 coeffs]...
    std::size_t local_limb = tid / coeff_count;
    std::size_t local_coeff = tid % coeff_count;
    std::size_t offset = local_limb * coeff_count + local_coeff;

    GpuWord modulus = q_primes[modulus_offset + local_limb];

    GpuWord sum = left_values[offset] + right_values[offset];

    if (sum < left_values[offset] || sum >= modulus)
    {
        sum -= modulus;
    }

    destination_values[offset] = static_cast<GpuWord>(sum);
}

__global__ void add_two_poly_shards_kernel(
    GpuWord *destination_values0,
    GpuWord *destination_values1,
    const GpuWord *left_values0,
    const GpuWord *left_values1,
    const GpuWord *right_values0,
    const GpuWord *right_values1,
    const GpuWord *q_primes,
    std::size_t modulus_offset,
    std::size_t limb_count,
    std::size_t coeff_count)
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
    const std::size_t local_limb = local_index / coeff_count;
    const std::size_t local_coeff = local_index % coeff_count;
    const std::size_t offset = local_limb * coeff_count + local_coeff;

    GpuWord *destination_values =
        second_component ? destination_values1 : destination_values0;
    const GpuWord *left_values =
        second_component ? left_values1 : left_values0;
    const GpuWord *right_values =
        second_component ? right_values1 : right_values0;

    const GpuWord modulus = q_primes[modulus_offset + local_limb];
    GpuWord sum = left_values[offset] + right_values[offset];

    if (sum < left_values[offset] || sum >= modulus)
    {
        sum -= modulus;
    }

    destination_values[offset] = static_cast<GpuWord>(sum);
}

/**
 * @brief CUDA kernel skeleton for modular subtraction.
 */
__global__ void sub_poly_shard_kernel(
    GpuWord *destination_values,
    const GpuWord *left_values,
    const GpuWord *right_values,
    const GpuWord *q_primes,
    std::size_t modulus_offset,
    std::size_t limb_count,
    std::size_t coeff_count)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = limb_count * coeff_count;
    if (tid >= total)
    {
        return;
    }

    std::size_t local_limb = tid / coeff_count;
    std::size_t local_coeff = tid % coeff_count;
    std::size_t offset = local_limb * coeff_count + local_coeff;

    GpuWord modulus = q_primes[modulus_offset + local_limb];

    GpuWord left = left_values[offset];
    GpuWord right = right_values[offset];

    GpuWord sub = left - right;
    if (left < right)
    {
        sub += modulus;
    }

    destination_values[offset] = sub;
}

/**
 * @brief CUDA kernel skeleton for modular negation.
 */
__global__ void negate_poly_shard_kernel(
    GpuWord *destination_values,
    const GpuWord *source_values,
    const GpuWord *q_primes,
    std::size_t modulus_offset,
    std::size_t limb_count,
    std::size_t coeff_count)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = limb_count * coeff_count;
    if (tid >= total)
    {
        return;
    }

    std::size_t local_limb = tid / coeff_count;
    std::size_t local_coeff = tid % coeff_count;
    std::size_t offset = local_limb * coeff_count + local_coeff;

    GpuWord modulus = q_primes[modulus_offset + local_limb];

    GpuWord source = source_values[offset];
    GpuWord negate = (source == 0) ? 0 : modulus - source;
    destination_values[offset] = negate;
}

/**
 * @brief CUDA kernel skeleton for dyadic product.
 */
__global__ void dyadic_product_poly_shard_kernel(
    GpuWord *destination_values,
    const GpuWord *left_values,
    const GpuWord *right_values,
    const GpuWord *q_primes,
    const GpuWide *q_modulus_constants,
    std::size_t limb_count,
    std::size_t coeff_count)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * coeff_count;

    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / coeff_count;
    const std::size_t offset = tid;

    const GpuWord modulus = q_primes[local_limb];
    const GpuWide barrett_ratio = q_modulus_constants[local_limb];

    const GpuWide product =
        static_cast<GpuWide>(left_values[offset]) *
        static_cast<GpuWide>(right_values[offset]);

    destination_values[offset] =
        barrett_reduce_u64_u32(product, modulus, barrett_ratio);
}

__global__ void multiply_scalar_poly_shard_kernel(
    GpuWord *destination_values,
    const GpuWord *source_values,
    GpuWord scalar,
    const GpuWord *q_primes,
    const GpuWide *q_modulus_constants,
    std::size_t limb_count,
    std::size_t coeff_count)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * coeff_count;

    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / coeff_count;
    const std::size_t offset = tid;

    const GpuWord modulus = q_primes[local_limb];
    const GpuWide barrett_ratio = q_modulus_constants[local_limb];
    const GpuWord scalar_mod =
        barrett_reduce_u64_u32(static_cast<GpuWide>(scalar), modulus, barrett_ratio);

    const GpuWide product =
        static_cast<GpuWide>(source_values[offset]) *
        static_cast<GpuWide>(scalar_mod);

    destination_values[offset] =
        barrett_reduce_u64_u32(product, modulus, barrett_ratio);
}

__global__ void bootstrap_modraise_poly_shard_kernel(
    GpuWord *destination_values,
    const GpuWord *source_values,
    const GpuWord *q_primes,
    const GpuWide *q_modulus_constants,
    const GpuWord *inv_punctured,
    const GpuWord *conversion_matrix,
    std::size_t source_q_count,
    std::size_t target_q_count,
    std::size_t coeff_count)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = target_q_count * coeff_count;

    if (tid >= total)
    {
        return;
    }

    const std::size_t target_limb = tid / coeff_count;
    const std::size_t coeff = tid % coeff_count;
    const std::size_t destination_offset = target_limb * coeff_count + coeff;

    if (target_limb < source_q_count)
    {
        destination_values[destination_offset] =
            source_values[target_limb * coeff_count + coeff];
        return;
    }

    /*
     * Bootstrap ModRaise must extend the centered representative, not the
     * unsigned representative selected by an ordinary RNS base conversion.
     * For the supported q0 bases (one or two <=30-bit physical primes), the
     * reconstructed value fits in 64 bits and can be centered exactly.
     *
     * Keeping this work coefficient-parallel is important: every CUDA thread
     * reconstructs one coefficient and writes one target limb, with no
     * coefficient-domain staging buffer or host participation.
     */
    if (source_q_count <= 2)
    {
        const GpuWord q0 = q_primes[0];
        GpuWide modulus_product = static_cast<GpuWide>(q0);
        GpuWide reconstructed =
            static_cast<GpuWide>(source_values[coeff]);

        if (source_q_count == 2)
        {
            const GpuWord q1 = q_primes[1];
            const GpuWord a0_mod_q1 = static_cast<GpuWord>(reconstructed % q1);
            const GpuWord a1 = source_values[coeff_count + coeff];
            const GpuWord difference =
                a1 >= a0_mod_q1 ? a1 - a0_mod_q1 : a1 + (q1 - a0_mod_q1);

            // inv_punctured[1] = q0^(-1) mod q1 for the two-prime base.
            const GpuWord crt_digit = barrett_reduce_u64_u32(
                static_cast<GpuWide>(difference) *
                    static_cast<GpuWide>(inv_punctured[1]),
                q1,
                q_modulus_constants[1]);
            reconstructed +=
                static_cast<GpuWide>(q0) * static_cast<GpuWide>(crt_digit);
            modulus_product *= static_cast<GpuWide>(q1);
        }

        const bool negative = reconstructed > (modulus_product >> 1);
        const GpuWide magnitude =
            negative ? modulus_product - reconstructed : reconstructed;
        const GpuWord target_modulus = q_primes[target_limb];
        const GpuWord reduced = barrett_reduce_u64_u32(
            magnitude,
            target_modulus,
            q_modulus_constants[target_limb]);
        destination_values[destination_offset] =
            negative && reduced != 0 ? target_modulus - reduced : reduced;
        return;
    }

    const std::size_t row = target_limb - source_q_count;
    const GpuWord target_modulus = q_primes[target_limb];
    const GpuWide target_barrett = q_modulus_constants[target_limb];
    GpuWord sum = 0;

    for (std::size_t source_limb = 0; source_limb < source_q_count; ++source_limb)
    {
        const GpuWord source_modulus = q_primes[source_limb];
        const GpuWide source_barrett = q_modulus_constants[source_limb];
        const GpuWord source_value =
            source_values[source_limb * coeff_count + coeff];
        const GpuWord weighted_source = barrett_reduce_u64_u32(
            static_cast<GpuWide>(source_value) *
                static_cast<GpuWide>(inv_punctured[source_limb]),
            source_modulus,
            source_barrett);

        const GpuWord matrix_value =
            conversion_matrix[row * source_q_count + source_limb];
        const GpuWord term = barrett_reduce_u64_u32(
            static_cast<GpuWide>(weighted_source) *
                static_cast<GpuWide>(matrix_value),
            target_modulus,
            target_barrett);

        GpuWide next = static_cast<GpuWide>(sum) + static_cast<GpuWide>(term);
        if (next >= target_modulus)
        {
            next -= target_modulus;
        }
        sum = static_cast<GpuWord>(next);
    }

    destination_values[destination_offset] = sum;
}

/**
 * @brief CUDA kernel skeleton for dyadic multiply-accumulate.
 * 简单来说就是进行result.c1 += a1 * b0这一步的操作，其中result.c1是对a0和b0的乘积，也就是构建密文密文乘的d1=a0 * b1 + a1 * b0的kernel实现
 */
__global__ void multiply_accumulate_poly_shard_kernel(
    GpuWord *destination_values,
    const GpuWord *left_values,
    const GpuWord *right_values,
    const GpuWord *q_primes,
    const GpuWide *q_modulus_constants,
    std::size_t limb_count,
    std::size_t coeff_count,
    std::size_t degree)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * coeff_count;

    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / coeff_count;
    const std::size_t offset = tid;

    const GpuWord modulus = q_primes[local_limb];
    const GpuWide barrett_ratio = q_modulus_constants[local_limb];

    const GpuWord product = barrett_reduce_u64_u32(
        static_cast<GpuWide>(left_values[offset]) *
            static_cast<GpuWide>(right_values[offset]),
        modulus,
        barrett_ratio);
    const GpuWide sum =
        static_cast<GpuWide>(destination_values[offset]) +
        static_cast<GpuWide>(product);

    destination_values[offset] = static_cast<GpuWord>(
        sum >= modulus ? sum - modulus : sum);
    (void)degree;
}

__global__ void multiply_plain_accumulate_two_components_kernel(
    GpuWord *destination0,
    GpuWord *destination1,
    const GpuWord *ciphertext0,
    const GpuWord *ciphertext1,
    const GpuWord *plaintext,
    const GpuWord *q_primes,
    const GpuWide *q_modulus_constants,
    std::size_t limb_count,
    std::size_t coeff_count)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * coeff_count;
    if (tid >= total)
    {
        return;
    }

    const bool second_component = blockIdx.y != 0;
    GpuWord *destination =
        second_component ? destination1 : destination0;
    const GpuWord *ciphertext =
        second_component ? ciphertext1 : ciphertext0;
    const std::size_t local_limb = tid / coeff_count;
    const GpuWord modulus = q_primes[local_limb];
    const GpuWide barrett_ratio = q_modulus_constants[local_limb];
    const GpuWord product = barrett_reduce_u64_u32(
        static_cast<GpuWide>(ciphertext[tid]) *
            static_cast<GpuWide>(plaintext[tid]),
        modulus,
        barrett_ratio);
    const GpuWide sum =
        static_cast<GpuWide>(destination[tid]) + product;
    destination[tid] = static_cast<GpuWord>(
        sum >= modulus ? sum - modulus : sum);
}

__global__ void multiply_outer_components_kernel(
    GpuWord *destination0,
    GpuWord *destination2,
    const GpuWord *left0,
    const GpuWord *left1,
    const GpuWord *right0,
    const GpuWord *right1,
    const GpuWord *q_primes,
    const GpuWide *q_modulus_constants,
    std::size_t limb_count,
    std::size_t coeff_count)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * coeff_count;
    if (tid >= total)
    {
        return;
    }

    const bool high_component = blockIdx.y != 0;
    const std::size_t local_limb = tid / coeff_count;
    const GpuWord modulus = q_primes[local_limb];
    const GpuWide barrett = q_modulus_constants[local_limb];
    GpuWord *destination = high_component ? destination2 : destination0;
    const GpuWord *left = high_component ? left1 : left0;
    const GpuWord *right = high_component ? right1 : right0;
    destination[tid] = barrett_reduce_u64_u32(
        static_cast<GpuWide>(left[tid]) *
            static_cast<GpuWide>(right[tid]),
        modulus,
        barrett);
}

__global__ void multiply_cross_component_kernel(
    GpuWord *destination1,
    const GpuWord *left0,
    const GpuWord *left1,
    const GpuWord *right0,
    const GpuWord *right1,
    const GpuWord *q_primes,
    const GpuWide *q_modulus_constants,
    std::size_t limb_count,
    std::size_t coeff_count)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * coeff_count;
    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / coeff_count;
    const GpuWord modulus = q_primes[local_limb];
    const GpuWide barrett = q_modulus_constants[local_limb];
    const GpuWord cross0 = barrett_reduce_u64_u32(
        static_cast<GpuWide>(left0[tid]) *
            static_cast<GpuWide>(right1[tid]),
        modulus,
        barrett);
    const GpuWord cross1 = barrett_reduce_u64_u32(
        static_cast<GpuWide>(left1[tid]) *
            static_cast<GpuWide>(right0[tid]),
        modulus,
        barrett);
    const GpuWide cross_sum =
        static_cast<GpuWide>(cross0) + static_cast<GpuWide>(cross1);
    destination1[tid] = static_cast<GpuWord>(
        cross_sum >= modulus ? cross_sum - modulus : cross_sum);
}

__global__ void square_cross_component_kernel(
    GpuWord *destination1,
    const GpuWord *source0,
    const GpuWord *source1,
    const GpuWord *q_primes,
    const GpuWide *q_modulus_constants,
    std::size_t limb_count,
    std::size_t coeff_count)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * coeff_count;
    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / coeff_count;
    const GpuWord modulus = q_primes[local_limb];
    const GpuWide barrett = q_modulus_constants[local_limb];
    const GpuWord cross = barrett_reduce_u64_u32(
        static_cast<GpuWide>(source0[tid]) *
            static_cast<GpuWide>(source1[tid]),
        modulus,
        barrett);
    const GpuWide doubled =
        static_cast<GpuWide>(cross) + static_cast<GpuWide>(cross);
    destination1[tid] = static_cast<GpuWord>(
        doubled >= modulus ? doubled - modulus : doubled);
}

}  // anonymous namespace

void launch_add_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &left_shard,
    const GpuConstPolyShardView &right_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    // 检查是否存在空指针
    if (destination_shard.ptr == nullptr ||
        left_shard.ptr == nullptr ||
        right_shard.ptr == nullptr)
    {
        throw std::invalid_argument("launch_add_poly_shard: null data pointer");
    }

    if (parameter_shard.q_primes.data() == nullptr)
    {
        throw std::invalid_argument("launch_add_poly_shard: null q_primes pointer");
    }

    if (degree == 0 ||
        destination_shard.limb_count == 0 ||
        destination_shard.coeff_count == 0)
    {
        throw std::invalid_argument("launch_add_poly_shard: empty shard shape");
    }

    if (destination_shard.device_id != left_shard.device_id ||
        destination_shard.device_id != right_shard.device_id ||
        destination_shard.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument("launch_add_poly_shard: device mismatch");
    }

    if (destination_shard.limb_begin != left_shard.limb_begin ||
        destination_shard.limb_begin != right_shard.limb_begin ||
        destination_shard.limb_count != left_shard.limb_count ||
        destination_shard.limb_count != right_shard.limb_count ||
        destination_shard.coeff_begin != left_shard.coeff_begin ||
        destination_shard.coeff_begin != right_shard.coeff_begin ||
        destination_shard.coeff_count != left_shard.coeff_count ||
        destination_shard.coeff_count != right_shard.coeff_count)
    {
        throw std::invalid_argument("launch_add_poly_shard: shard shape mismatch");
    }

    if (destination_shard.coeff_count > degree)
    {
        throw std::invalid_argument("launch_add_poly_shard: coeff_count exceeds degree");
    }

    if (destination_shard.limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument("launch_add_poly_shard: parameter shard does not cover limb range");
    }

    const std::size_t modulus_offset =
        destination_shard.limb_begin - parameter_shard.limb_begin;
    
    if (modulus_offset + destination_shard.limb_count > parameter_shard.q_primes.size())
    {
        throw std::invalid_argument("launch_add_poly_shard: q_primes does not cover limb range");
    }

    const std::size_t total_count =
        destination_shard.limb_count * destination_shard.coeff_count;

    if (total_count == 0)
    {
        return;
    }

    // cudaSetDevice选择当前要 launch 的 GPU
    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_add_poly_shard cudaSetDevice");

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total_count + block_size - 1) / block_size);

    add_poly_shard_kernel<<<grid_size, block_size>>>(
        destination_shard.ptr,
        left_shard.ptr,
        right_shard.ptr,
        parameter_shard.q_primes.data(),
        modulus_offset,
        destination_shard.limb_count,
        destination_shard.coeff_count);
    
    // 判断kernel是否正确发射，不代表计算完成
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_add_poly_shard kernel launch");
    
    // Do not synchronize here. The caller decides when to wait, so multiple
    // shard/component kernels can be queued without a CPU round trip per launch.
}

void launch_add_two_poly_shards(
    const GpuPolyShardView &destination_shard0,
    const GpuPolyShardView &destination_shard1,
    const GpuConstPolyShardView &left_shard0,
    const GpuConstPolyShardView &left_shard1,
    const GpuConstPolyShardView &right_shard0,
    const GpuConstPolyShardView &right_shard1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    if (destination_shard0.ptr == nullptr ||
        destination_shard1.ptr == nullptr ||
        left_shard0.ptr == nullptr ||
        left_shard1.ptr == nullptr ||
        right_shard0.ptr == nullptr ||
        right_shard1.ptr == nullptr)
    {
        throw std::invalid_argument("launch_add_two_poly_shards: null data pointer");
    }

    if (parameter_shard.q_primes.data() == nullptr)
    {
        throw std::invalid_argument("launch_add_two_poly_shards: null q_primes pointer");
    }

    if (degree == 0 ||
        destination_shard0.limb_count == 0 ||
        destination_shard0.coeff_count == 0)
    {
        throw std::invalid_argument("launch_add_two_poly_shards: empty shard shape");
    }

    const auto same_device =
        destination_shard0.device_id == destination_shard1.device_id &&
        destination_shard0.device_id == left_shard0.device_id &&
        destination_shard0.device_id == left_shard1.device_id &&
        destination_shard0.device_id == right_shard0.device_id &&
        destination_shard0.device_id == right_shard1.device_id &&
        destination_shard0.device_id == parameter_shard.device_id;
    if (!same_device)
    {
        throw std::invalid_argument("launch_add_two_poly_shards: device mismatch");
    }

    const auto same_shape =
        destination_shard0.limb_begin == destination_shard1.limb_begin &&
        destination_shard0.limb_begin == left_shard0.limb_begin &&
        destination_shard0.limb_begin == left_shard1.limb_begin &&
        destination_shard0.limb_begin == right_shard0.limb_begin &&
        destination_shard0.limb_begin == right_shard1.limb_begin &&
        destination_shard0.limb_count == destination_shard1.limb_count &&
        destination_shard0.limb_count == left_shard0.limb_count &&
        destination_shard0.limb_count == left_shard1.limb_count &&
        destination_shard0.limb_count == right_shard0.limb_count &&
        destination_shard0.limb_count == right_shard1.limb_count &&
        destination_shard0.coeff_begin == destination_shard1.coeff_begin &&
        destination_shard0.coeff_begin == left_shard0.coeff_begin &&
        destination_shard0.coeff_begin == left_shard1.coeff_begin &&
        destination_shard0.coeff_begin == right_shard0.coeff_begin &&
        destination_shard0.coeff_begin == right_shard1.coeff_begin &&
        destination_shard0.coeff_count == destination_shard1.coeff_count &&
        destination_shard0.coeff_count == left_shard0.coeff_count &&
        destination_shard0.coeff_count == left_shard1.coeff_count &&
        destination_shard0.coeff_count == right_shard0.coeff_count &&
        destination_shard0.coeff_count == right_shard1.coeff_count;
    if (!same_shape)
    {
        throw std::invalid_argument("launch_add_two_poly_shards: shard shape mismatch");
    }

    if (destination_shard0.coeff_count > degree)
    {
        throw std::invalid_argument("launch_add_two_poly_shards: coeff_count exceeds degree");
    }

    if (destination_shard0.limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument(
            "launch_add_two_poly_shards: parameter shard does not cover limb range");
    }

    const std::size_t modulus_offset =
        destination_shard0.limb_begin - parameter_shard.limb_begin;
    
    if (modulus_offset + destination_shard0.limb_count > parameter_shard.q_primes.size())
    {
        throw std::invalid_argument("launch_add_two_poly_shards: q_primes does not cover limb range");
    }

    const std::size_t values_per_component =
        destination_shard0.limb_count * destination_shard0.coeff_count;
    const std::size_t total_count = values_per_component * 2;
    if (total_count == 0)
    {
        return;
    }

    gpu_check_cuda(
        cudaSetDevice(destination_shard0.device_id),
        "launch_add_two_poly_shards cudaSetDevice");

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total_count + block_size - 1) / block_size);

    add_two_poly_shards_kernel<<<grid_size, block_size>>>(
        destination_shard0.ptr,
        destination_shard1.ptr,
        left_shard0.ptr,
        left_shard1.ptr,
        right_shard0.ptr,
        right_shard1.ptr,
        parameter_shard.q_primes.data(),
        modulus_offset,
        destination_shard0.limb_count,
        destination_shard0.coeff_count);

    gpu_check_cuda(
        cudaGetLastError(),
        "launch_add_two_poly_shards kernel launch");
}

void launch_sub_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &left_shard,
    const GpuConstPolyShardView &right_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    // 检查是否存在空指针
    if (destination_shard.ptr == nullptr ||
        left_shard.ptr == nullptr ||
        right_shard.ptr == nullptr)
    {
        throw std::invalid_argument("launch_sub_poly_shard: null data pointer");
    }

    if (parameter_shard.q_primes.data() == nullptr)
    {
        throw std::invalid_argument("launch_sub_poly_shard: null q_primes pointer");
    }

    if (degree == 0 ||
        destination_shard.limb_count == 0 ||
        destination_shard.coeff_count == 0)
    {
        throw std::invalid_argument("launch_sub_poly_shard: empty shard shape");
    }

    if (destination_shard.device_id != left_shard.device_id ||
        destination_shard.device_id != right_shard.device_id ||
        destination_shard.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument("launch_sub_poly_shard: device mismatch");
    }

    if (destination_shard.limb_begin != left_shard.limb_begin ||
        destination_shard.limb_begin != right_shard.limb_begin ||
        destination_shard.limb_count != left_shard.limb_count ||
        destination_shard.limb_count != right_shard.limb_count ||
        destination_shard.coeff_begin != left_shard.coeff_begin ||
        destination_shard.coeff_begin != right_shard.coeff_begin ||
        destination_shard.coeff_count != left_shard.coeff_count ||
        destination_shard.coeff_count != right_shard.coeff_count)
    {
        throw std::invalid_argument("launch_sub_poly_shard: shard shape mismatch");
    }

    if (destination_shard.coeff_count > degree)
    {
        throw std::invalid_argument("launch_sub_poly_shard: coeff_count exceeds degree");
    }

    if (destination_shard.limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument("launch_sub_poly_shard: parameter shard does not cover limb range");
    }

    const std::size_t modulus_offset =
        destination_shard.limb_begin - parameter_shard.limb_begin;
    
    if (modulus_offset + destination_shard.limb_count > parameter_shard.q_primes.size())
    {
        throw std::invalid_argument("launch_sub_poly_shard: q_primes does not cover limb range");
    }

    const std::size_t total_count =
        destination_shard.limb_count * destination_shard.coeff_count;

    if (total_count == 0)
    {
        return;
    }

    // cudaSetDevice选择当前要 launch 的 GPU
    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_sub_poly_shard cudaSetDevice");

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total_count + block_size - 1) / block_size);

    sub_poly_shard_kernel<<<grid_size, block_size>>>(
        destination_shard.ptr,
        left_shard.ptr,
        right_shard.ptr,
        parameter_shard.q_primes.data(),
        modulus_offset,
        destination_shard.limb_count,
        destination_shard.coeff_count);
    
    // 判断kernel是否正确发射，不代表计算完成
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_sub_poly_shard kernel launch");
}

void launch_negate_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    if (destination_shard.ptr == nullptr ||
        source_shard.ptr == nullptr)
    {
        throw std::invalid_argument("launch_negate_poly_shard: null data pointer");
    }

    if (parameter_shard.q_primes.data() == nullptr)
    {
        throw std::invalid_argument("launch_negate_poly_shard: null q_primes pointer");
    }

    if (degree == 0 ||
        destination_shard.limb_count == 0 ||
        destination_shard.coeff_count == 0)
    {
        throw std::invalid_argument("launch_negate_poly_shard: empty shard shape");
    }

    if (destination_shard.device_id != source_shard.device_id ||
        destination_shard.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument("launch_negate_poly_shard: device mismatch");
    }

    if (destination_shard.limb_begin != source_shard.limb_begin ||
        destination_shard.limb_count != source_shard.limb_count ||
        destination_shard.coeff_begin != source_shard.coeff_begin ||
        destination_shard.coeff_count != source_shard.coeff_count)
    {
        throw std::invalid_argument("launch_negate_poly_shard: shard shape mismatch");
    }

    if (destination_shard.coeff_count > degree)
    {
        throw std::invalid_argument("launch_negate_poly_shard: coeff_count exceeds degree");
    }

    if (destination_shard.limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument("launch_negate_poly_shard: parameter shard does not cover limb range");
    }

    const std::size_t modulus_offset =
        destination_shard.limb_begin - parameter_shard.limb_begin;
    
    if (modulus_offset + destination_shard.limb_count > parameter_shard.q_primes.size())
    {
        throw std::invalid_argument("launch_negate_poly_shard: q_primes does not cover limb range");
    }

    const std::size_t total_count =
        destination_shard.limb_count * destination_shard.coeff_count;

    if (total_count == 0)
    {
        return;
    }

    // cudaSetDevice选择当前要 launch 的 GPU
    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_negate_poly_shard cudaSetDevice");

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total_count + block_size - 1) / block_size);

    negate_poly_shard_kernel<<<grid_size, block_size>>>(
        destination_shard.ptr,
        source_shard.ptr,
        parameter_shard.q_primes.data(),
        modulus_offset,
        destination_shard.limb_count,
        destination_shard.coeff_count);
    
    // 判断kernel是否正确发射，不代表计算完成
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_negate_poly_shard kernel launch");
}

void launch_copy_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    std::size_t degree)
{
    if (destination_shard.ptr == nullptr || source_shard.ptr == nullptr)
    {
        throw std::invalid_argument("launch_copy_poly_shard: null data pointer");
    }

    if (destination_shard.coeff_count > degree)
    {
        throw std::invalid_argument("launch_copy_poly_shard: coeff_count exceeds degree");
    }

    if (degree == 0 ||
        destination_shard.limb_count == 0 ||
        destination_shard.coeff_count == 0)
    {
        throw std::invalid_argument("launch_copy_poly_shard: empty shard shape");
    }

    if (destination_shard.device_id != source_shard.device_id)
    {
        throw std::invalid_argument("launch_copy_poly_shard: device mismatch");
    }

    if (destination_shard.limb_begin != source_shard.limb_begin ||
        destination_shard.limb_count != source_shard.limb_count ||
        destination_shard.coeff_begin != source_shard.coeff_begin ||
        destination_shard.coeff_count != source_shard.coeff_count)
    {
        throw std::invalid_argument("launch_copy_poly_shard: shard shape mismatch");
    }

    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_copy_poly_shard cudaSetDevice");

    const std::size_t word_count =
        destination_shard.limb_count * destination_shard.coeff_count;

    gpu_check_cuda(
        cudaMemcpy(
            destination_shard.ptr,
            source_shard.ptr,
            word_count * sizeof(GpuWord),
            cudaMemcpyDeviceToDevice),
        "launch_copy_poly_shard cudaMemcpyDeviceToDevice");

    (void)degree;
}

void launch_multiply_scalar_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    GpuWord scalar,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    if (destination_shard.ptr == nullptr || source_shard.ptr == nullptr)
    {
        throw std::invalid_argument("launch_multiply_scalar_poly_shard: null data pointer");
    }
    if (parameter_shard.q_primes.data() == nullptr)
    {
        throw std::invalid_argument("launch_multiply_scalar_poly_shard: null q_primes pointer");
    }
    if (parameter_shard.q_modulus_constants.data() == nullptr)
    {
        throw std::invalid_argument(
            "launch_multiply_scalar_poly_shard: null q_modulus_constants pointer");
    }
    if (degree == 0 ||
        destination_shard.limb_count == 0 ||
        destination_shard.coeff_count == 0)
    {
        throw std::invalid_argument("launch_multiply_scalar_poly_shard: empty shard shape");
    }
    if (destination_shard.device_id != source_shard.device_id ||
        destination_shard.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument("launch_multiply_scalar_poly_shard: device mismatch");
    }
    if (destination_shard.limb_begin != source_shard.limb_begin ||
        destination_shard.limb_count != source_shard.limb_count ||
        destination_shard.coeff_begin != source_shard.coeff_begin ||
        destination_shard.coeff_count != source_shard.coeff_count)
    {
        throw std::invalid_argument("launch_multiply_scalar_poly_shard: shard shape mismatch");
    }
    if (destination_shard.coeff_count > degree)
    {
        throw std::invalid_argument("launch_multiply_scalar_poly_shard: coeff_count exceeds degree");
    }
    if (destination_shard.limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument(
            "launch_multiply_scalar_poly_shard: parameter shard does not cover limb range");
    }

    const std::size_t modulus_offset =
        destination_shard.limb_begin - parameter_shard.limb_begin;

    if (modulus_offset + destination_shard.limb_count > parameter_shard.q_primes.size())
    {
        throw std::invalid_argument(
            "launch_multiply_scalar_poly_shard: q_primes does not cover limb range");
    }
    if (modulus_offset + destination_shard.limb_count >
        parameter_shard.q_modulus_constants.size())
    {
        throw std::invalid_argument(
            "launch_multiply_scalar_poly_shard: q_modulus_constants does not cover limb range");
    }

    const std::size_t total_count =
        destination_shard.limb_count * destination_shard.coeff_count;
    if (total_count == 0)
    {
        return;
    }

    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_multiply_scalar_poly_shard cudaSetDevice");

    constexpr int block_size = 256;
    const int grid_size =
        static_cast<int>((total_count + block_size - 1) / block_size);

    multiply_scalar_poly_shard_kernel<<<grid_size, block_size>>>(
        destination_shard.ptr,
        source_shard.ptr,
        scalar,
        parameter_shard.q_primes.data() + modulus_offset,
        parameter_shard.q_modulus_constants.data() + modulus_offset,
        destination_shard.limb_count,
        destination_shard.coeff_count);

    gpu_check_cuda(
        cudaGetLastError(),
        "launch_multiply_scalar_poly_shard kernel launch");

    (void)degree;
}

void launch_bootstrap_modraise_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &source_parameter_shard,
    const GpuParameterShard &target_parameter_shard,
    std::size_t source_q_count,
    std::size_t target_q_count,
    std::size_t degree)
{
    if (destination_shard.ptr == nullptr || source_shard.ptr == nullptr)
    {
        throw std::invalid_argument("launch_bootstrap_modraise_poly_shard: null data pointer");
    }
    if (target_parameter_shard.q_primes.data() == nullptr ||
        target_parameter_shard.q_modulus_constants.data() == nullptr)
    {
        throw std::invalid_argument(
            "launch_bootstrap_modraise_poly_shard: null modulus table pointer");
    }
    if (degree == 0 ||
        source_q_count == 0 ||
        target_q_count == 0 ||
        source_q_count > target_q_count)
    {
        throw std::invalid_argument("launch_bootstrap_modraise_poly_shard: invalid shape");
    }
    if (destination_shard.device_id != source_shard.device_id ||
        destination_shard.device_id != source_parameter_shard.device_id ||
        destination_shard.device_id != target_parameter_shard.device_id)
    {
        throw std::invalid_argument("launch_bootstrap_modraise_poly_shard: device mismatch");
    }
    if (source_shard.limb_begin != 0 ||
        destination_shard.limb_begin != 0 ||
        source_shard.limb_count != source_q_count ||
        destination_shard.limb_count != target_q_count ||
        source_shard.coeff_begin != 0 ||
        destination_shard.coeff_begin != 0 ||
        source_shard.coeff_count != degree ||
        destination_shard.coeff_count != degree)
    {
        throw std::invalid_argument(
            "launch_bootstrap_modraise_poly_shard: first implementation requires full prefix shards");
    }
    if (target_q_count > target_parameter_shard.q_primes.size() ||
        target_q_count > target_parameter_shard.q_modulus_constants.size())
    {
        throw std::invalid_argument(
            "launch_bootstrap_modraise_poly_shard: modulus table does not cover target Q");
    }
    if (source_parameter_shard.bootstrap_raise_source_q_count != source_q_count ||
        source_parameter_shard.bootstrap_raise_target_q_count != target_q_count)
    {
        throw std::invalid_argument(
            "launch_bootstrap_modraise_poly_shard: ModRaise table shape mismatch");
    }
    if (source_q_count < target_q_count)
    {
        if (source_parameter_shard.bootstrap_raise_inv_punctured.data() == nullptr ||
            source_parameter_shard.bootstrap_raise_matrix.data() == nullptr)
        {
            throw std::invalid_argument(
                "launch_bootstrap_modraise_poly_shard: null ModRaise table pointer");
        }
        const std::size_t suffix_q_count = target_q_count - source_q_count;
        if (source_parameter_shard.bootstrap_raise_inv_punctured.size() <
                source_q_count ||
            source_parameter_shard.bootstrap_raise_matrix.size() <
                suffix_q_count * source_q_count)
        {
            throw std::invalid_argument(
                "launch_bootstrap_modraise_poly_shard: ModRaise table too small");
        }
    }

    const std::size_t total_count = target_q_count * degree;
    if (total_count == 0)
    {
        return;
    }

    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_bootstrap_modraise_poly_shard cudaSetDevice");

    constexpr int block_size = 256;
    const int grid_size =
        static_cast<int>((total_count + block_size - 1) / block_size);

    bootstrap_modraise_poly_shard_kernel<<<grid_size, block_size>>>(
        destination_shard.ptr,
        source_shard.ptr,
        target_parameter_shard.q_primes.data(),
        target_parameter_shard.q_modulus_constants.data(),
        source_parameter_shard.bootstrap_raise_inv_punctured.data(),
        source_parameter_shard.bootstrap_raise_matrix.data(),
        source_q_count,
        target_q_count,
        degree);

    gpu_check_cuda(
        cudaGetLastError(),
        "launch_bootstrap_modraise_poly_shard kernel launch");
}

void launch_dyadic_product_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &left_shard,
    const GpuConstPolyShardView &right_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    if (destination_shard.ptr == nullptr ||
        left_shard.ptr == nullptr ||
        right_shard.ptr == nullptr)
    {
        throw std::invalid_argument("launch_dyadic_product_poly_shard: null data pointer");
    }

    if (parameter_shard.q_primes.data() == nullptr)
    {
        throw std::invalid_argument("launch_dyadic_product_poly_shard: null q_primes pointer");
    }
    if (parameter_shard.q_modulus_constants.data() == nullptr)
    {
        throw std::invalid_argument(
            "launch_dyadic_product_poly_shard: null q_modulus_constants pointer");
    }

    if (degree == 0 ||
        destination_shard.limb_count == 0 ||
        destination_shard.coeff_count == 0)
    {
        throw std::invalid_argument("launch_dyadic_product_poly_shard: empty shard shape");
    }

    if (destination_shard.device_id != left_shard.device_id ||
        destination_shard.device_id != right_shard.device_id ||
        destination_shard.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument("launch_dyadic_product_poly_shard: device mismatch");
    }

    if (destination_shard.limb_begin != left_shard.limb_begin ||
        destination_shard.limb_begin != right_shard.limb_begin ||
        destination_shard.limb_count != left_shard.limb_count ||
        destination_shard.limb_count != right_shard.limb_count ||
        destination_shard.coeff_begin != left_shard.coeff_begin ||
        destination_shard.coeff_begin != right_shard.coeff_begin ||
        destination_shard.coeff_count != left_shard.coeff_count ||
        destination_shard.coeff_count != right_shard.coeff_count)
    {
        throw std::invalid_argument("launch_dyadic_product_poly_shard: shard shape mismatch");
    }

    if (destination_shard.coeff_count > degree)
    {
        throw std::invalid_argument("launch_dyadic_product_poly_shard: coeff_count exceeds degree");
    }

    if (destination_shard.limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument(
            "launch_dyadic_product_poly_shard: parameter shard does not cover limb range");
    }

    const std::size_t modulus_offset =
        destination_shard.limb_begin - parameter_shard.limb_begin;

    if (modulus_offset + destination_shard.limb_count > parameter_shard.q_primes.size())
    {
        throw std::invalid_argument(
            "launch_dyadic_product_poly_shard: q_primes does not cover limb range");
    }
    if (modulus_offset + destination_shard.limb_count >
        parameter_shard.q_modulus_constants.size())
    {
        throw std::invalid_argument(
            "launch_dyadic_product_poly_shard: q_modulus_constants does not cover limb range");
    }

    const std::size_t total_count =
        destination_shard.limb_count * destination_shard.coeff_count;

    if (total_count == 0)
    {
        return;
    }

    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_dyadic_product_poly_shard cudaSetDevice");

    constexpr int block_size = 256;
    const int grid_size =
        static_cast<int>((total_count + block_size - 1) / block_size);

    dyadic_product_poly_shard_kernel<<<grid_size, block_size>>>(
        destination_shard.ptr,
        left_shard.ptr,
        right_shard.ptr,
        parameter_shard.q_primes.data() + modulus_offset,
        parameter_shard.q_modulus_constants.data() + modulus_offset,
        destination_shard.limb_count,
        destination_shard.coeff_count);

    gpu_check_cuda(
        cudaGetLastError(),
        "launch_dyadic_product_poly_shard kernel launch");
}

/**
 * @brief 用来发射对应乘加kernel。首先进行输入格式检查，
 */
void launch_multiply_accumulate_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &left_shard,
    const GpuConstPolyShardView &right_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    if (destination_shard.ptr == nullptr ||
        left_shard.ptr == nullptr ||
        right_shard.ptr == nullptr)
    {
        throw std::invalid_argument("launch_multiply_accumulate_poly_shard: null data pointer");
    }

    if (parameter_shard.q_primes.data() == nullptr)
    {
        throw std::invalid_argument("launch_multiply_accumulate_poly_shard: null q_primes pointer");
    }
    if (parameter_shard.q_modulus_constants.data() == nullptr)
    {
        throw std::invalid_argument(
            "launch_multiply_accumulate_poly_shard: null q_modulus_constants pointer");
    }

    if (degree == 0 ||
        destination_shard.limb_count == 0 ||
        destination_shard.coeff_count == 0)
    {
        throw std::invalid_argument("launch_multiply_accumulate_poly_shard: empty shard shape");
    }

    if (destination_shard.device_id != left_shard.device_id ||
        destination_shard.device_id != right_shard.device_id ||
        destination_shard.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument("launch_multiply_accumulate_poly_shard: device mismatch");
    }

    if (destination_shard.limb_begin != left_shard.limb_begin ||
        destination_shard.limb_begin != right_shard.limb_begin ||
        destination_shard.limb_count != left_shard.limb_count ||
        destination_shard.limb_count != right_shard.limb_count ||
        destination_shard.coeff_begin != left_shard.coeff_begin ||
        destination_shard.coeff_begin != right_shard.coeff_begin ||
        destination_shard.coeff_count != left_shard.coeff_count ||
        destination_shard.coeff_count != right_shard.coeff_count)
    {
        throw std::invalid_argument(
            "launch_multiply_accumulate_poly_shard: shard shape mismatch");
    }

    if (destination_shard.coeff_count > degree)
    {
        throw std::invalid_argument(
            "launch_multiply_accumulate_poly_shard: coeff_count exceeds degree");
    }

    if (destination_shard.limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument(
            "launch_multiply_accumulate_poly_shard: parameter shard does not cover limb range");
    }

    const std::size_t modulus_offset =
        destination_shard.limb_begin - parameter_shard.limb_begin;

    if (modulus_offset + destination_shard.limb_count >
        parameter_shard.q_primes.size())
    {
        throw std::invalid_argument(
            "launch_multiply_accumulate_poly_shard: q_primes does not cover limb range");
    }
    if (modulus_offset + destination_shard.limb_count >
        parameter_shard.q_modulus_constants.size())
    {
        throw std::invalid_argument(
            "launch_multiply_accumulate_poly_shard: q_modulus_constants does not cover limb range");
    }

    const std::size_t total_count =
        destination_shard.limb_count * destination_shard.coeff_count;
    if (total_count == 0)
    {
        return;
    }

    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_multiply_accumulate_poly_shard cudaSetDevice");

    constexpr int block_size = 256;
    const int grid_size =
        static_cast<int>((total_count + block_size - 1) / block_size);

    multiply_accumulate_poly_shard_kernel<<<grid_size, block_size>>>(
        destination_shard.ptr,
        left_shard.ptr,
        right_shard.ptr,
        parameter_shard.q_primes.data() + modulus_offset,
        parameter_shard.q_modulus_constants.data() + modulus_offset,
        destination_shard.limb_count,
        destination_shard.coeff_count,
        degree);

    gpu_check_cuda(
        cudaGetLastError(),
        "launch_multiply_accumulate_poly_shard kernel launch");
}

void launch_multiply_plain_accumulate_two_components(
    const GpuPolyShardView &destination0,
    const GpuPolyShardView &destination1,
    const GpuConstPolyShardView &ciphertext0,
    const GpuConstPolyShardView &ciphertext1,
    const GpuConstPolyShardView &plaintext,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    constexpr const char *name =
        "launch_multiply_plain_accumulate_two_components";
    const bool same_shape =
        destination0.device_id == destination1.device_id &&
        destination0.device_id == ciphertext0.device_id &&
        destination0.device_id == ciphertext1.device_id &&
        destination0.device_id == plaintext.device_id &&
        destination0.device_id == parameter_shard.device_id &&
        destination0.limb_begin == destination1.limb_begin &&
        destination0.limb_begin == ciphertext0.limb_begin &&
        destination0.limb_begin == ciphertext1.limb_begin &&
        destination0.limb_begin == plaintext.limb_begin &&
        destination0.limb_count == destination1.limb_count &&
        destination0.limb_count == ciphertext0.limb_count &&
        destination0.limb_count == ciphertext1.limb_count &&
        destination0.limb_count == plaintext.limb_count &&
        destination0.coeff_begin == destination1.coeff_begin &&
        destination0.coeff_begin == ciphertext0.coeff_begin &&
        destination0.coeff_begin == ciphertext1.coeff_begin &&
        destination0.coeff_begin == plaintext.coeff_begin &&
        destination0.coeff_count == destination1.coeff_count &&
        destination0.coeff_count == ciphertext0.coeff_count &&
        destination0.coeff_count == ciphertext1.coeff_count &&
        destination0.coeff_count == plaintext.coeff_count;
    if (!same_shape || destination0.ptr == nullptr ||
        destination1.ptr == nullptr || ciphertext0.ptr == nullptr ||
        ciphertext1.ptr == nullptr || plaintext.ptr == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": shape mismatch");
    }
    if (destination0.limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument(
            std::string(name) + ": parameter limb range mismatch");
    }
    const std::size_t modulus_offset =
        destination0.limb_begin - parameter_shard.limb_begin;
    if (modulus_offset + destination0.limb_count >
            parameter_shard.q_primes.size() ||
        modulus_offset + destination0.limb_count >
            parameter_shard.q_modulus_constants.size())
    {
        throw std::invalid_argument(
            std::string(name) + ": modulus table range mismatch");
    }

    gpu_check_cuda(
        cudaSetDevice(destination0.device_id),
        "launch_multiply_plain_accumulate_two_components cudaSetDevice");
    constexpr int block_size = 256;
    const std::size_t total =
        destination0.limb_count * destination0.coeff_count;
    const int grid_size = static_cast<int>(
        (total + block_size - 1) / block_size);
    multiply_plain_accumulate_two_components_kernel<<<
        dim3(grid_size, 2),
        block_size>>>(
        destination0.ptr,
        destination1.ptr,
        ciphertext0.ptr,
        ciphertext1.ptr,
        plaintext.ptr,
        parameter_shard.q_primes.data() + modulus_offset,
        parameter_shard.q_modulus_constants.data() + modulus_offset,
        destination0.limb_count,
        destination0.coeff_count);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_multiply_plain_accumulate_two_components kernel launch");
    (void)degree;
}

std::size_t validate_fused_ciphertext_product_shards(
    const char *name,
    const GpuPolyShardView &destination_shard0,
    const GpuPolyShardView &destination_shard1,
    const GpuPolyShardView &destination_shard2,
    const GpuConstPolyShardView &left_shard0,
    const GpuConstPolyShardView &left_shard1,
    const GpuConstPolyShardView &right_shard0,
    const GpuConstPolyShardView &right_shard1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    if (destination_shard0.ptr == nullptr ||
        destination_shard1.ptr == nullptr ||
        destination_shard2.ptr == nullptr ||
        left_shard0.ptr == nullptr ||
        left_shard1.ptr == nullptr ||
        right_shard0.ptr == nullptr ||
        right_shard1.ptr == nullptr ||
        parameter_shard.q_primes.data() == nullptr ||
        parameter_shard.q_modulus_constants.data() == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null data pointer");
    }
    if (degree == 0 ||
        destination_shard0.limb_count == 0 ||
        destination_shard0.coeff_count == 0 ||
        destination_shard0.coeff_count > degree)
    {
        throw std::invalid_argument(std::string(name) + ": invalid shard shape");
    }

    const auto same_placement =
        [&destination_shard0](const auto &candidate)
        {
            return candidate.device_id == destination_shard0.device_id &&
                   candidate.limb_begin == destination_shard0.limb_begin &&
                   candidate.limb_count == destination_shard0.limb_count &&
                   candidate.coeff_begin == destination_shard0.coeff_begin &&
                   candidate.coeff_count == destination_shard0.coeff_count;
        };
    if (!same_placement(destination_shard1) ||
        !same_placement(destination_shard2) ||
        !same_placement(left_shard0) ||
        !same_placement(left_shard1) ||
        !same_placement(right_shard0) ||
        !same_placement(right_shard1) ||
        destination_shard0.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument(
            std::string(name) + ": shard placement mismatch");
    }
    if (destination_shard0.limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument(
            std::string(name) + ": parameter shard does not cover limb range");
    }

    const std::size_t modulus_offset =
        destination_shard0.limb_begin - parameter_shard.limb_begin;
    if (modulus_offset + destination_shard0.limb_count >
            parameter_shard.q_primes.size() ||
        modulus_offset + destination_shard0.limb_count >
            parameter_shard.q_modulus_constants.size())
    {
        throw std::invalid_argument(
            std::string(name) + ": modulus tables do not cover limb range");
    }
    return modulus_offset;
}

void launch_multiply_two_component_ciphertexts(
    const GpuPolyShardView &destination_shard0,
    const GpuPolyShardView &destination_shard1,
    const GpuPolyShardView &destination_shard2,
    const GpuConstPolyShardView &left_shard0,
    const GpuConstPolyShardView &left_shard1,
    const GpuConstPolyShardView &right_shard0,
    const GpuConstPolyShardView &right_shard1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    constexpr const char *name =
        "launch_multiply_two_component_ciphertexts";
    const std::size_t modulus_offset =
        validate_fused_ciphertext_product_shards(
            name,
            destination_shard0,
            destination_shard1,
            destination_shard2,
            left_shard0,
            left_shard1,
            right_shard0,
            right_shard1,
            parameter_shard,
            degree);
    const std::size_t total =
        destination_shard0.limb_count * destination_shard0.coeff_count;
    if (total == 0)
    {
        return;
    }

    gpu_check_cuda(
        cudaSetDevice(destination_shard0.device_id),
        "launch_multiply_two_component_ciphertexts cudaSetDevice");
    constexpr int block_size = 256;
    const unsigned int grid_size =
        static_cast<int>((total + block_size - 1) / block_size);
    multiply_outer_components_kernel<<<dim3(grid_size, 2), block_size>>>(
        destination_shard0.ptr,
        destination_shard2.ptr,
        left_shard0.ptr,
        left_shard1.ptr,
        right_shard0.ptr,
        right_shard1.ptr,
        parameter_shard.q_primes.data() + modulus_offset,
        parameter_shard.q_modulus_constants.data() + modulus_offset,
        destination_shard0.limb_count,
        destination_shard0.coeff_count);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_multiply_two_component_ciphertexts outer kernel launch");
    multiply_cross_component_kernel<<<grid_size, block_size>>>(
        destination_shard1.ptr,
        left_shard0.ptr,
        left_shard1.ptr,
        right_shard0.ptr,
        right_shard1.ptr,
        parameter_shard.q_primes.data() + modulus_offset,
        parameter_shard.q_modulus_constants.data() + modulus_offset,
        destination_shard0.limb_count,
        destination_shard0.coeff_count);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_multiply_two_component_ciphertexts cross kernel launch");
}

void launch_square_two_component_ciphertext(
    const GpuPolyShardView &destination_shard0,
    const GpuPolyShardView &destination_shard1,
    const GpuPolyShardView &destination_shard2,
    const GpuConstPolyShardView &source_shard0,
    const GpuConstPolyShardView &source_shard1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    constexpr const char *name =
        "launch_square_two_component_ciphertext";
    const std::size_t modulus_offset =
        validate_fused_ciphertext_product_shards(
            name,
            destination_shard0,
            destination_shard1,
            destination_shard2,
            source_shard0,
            source_shard1,
            source_shard0,
            source_shard1,
            parameter_shard,
            degree);
    const std::size_t total =
        destination_shard0.limb_count * destination_shard0.coeff_count;
    if (total == 0)
    {
        return;
    }

    gpu_check_cuda(
        cudaSetDevice(destination_shard0.device_id),
        "launch_square_two_component_ciphertext cudaSetDevice");
    constexpr int block_size = 256;
    const unsigned int grid_size =
        static_cast<int>((total + block_size - 1) / block_size);
    multiply_outer_components_kernel<<<dim3(grid_size, 2), block_size>>>(
        destination_shard0.ptr,
        destination_shard2.ptr,
        source_shard0.ptr,
        source_shard1.ptr,
        source_shard0.ptr,
        source_shard1.ptr,
        parameter_shard.q_primes.data() + modulus_offset,
        parameter_shard.q_modulus_constants.data() + modulus_offset,
        destination_shard0.limb_count,
        destination_shard0.coeff_count);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_square_two_component_ciphertext outer kernel launch");
    square_cross_component_kernel<<<grid_size, block_size>>>(
        destination_shard1.ptr,
        source_shard0.ptr,
        source_shard1.ptr,
        parameter_shard.q_primes.data() + modulus_offset,
        parameter_shard.q_modulus_constants.data() + modulus_offset,
        destination_shard0.limb_count,
        destination_shard0.coeff_count);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_square_two_component_ciphertext cross kernel launch");
}

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
