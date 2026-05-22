#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"

#include <stdexcept>

namespace poseidon
{
namespace gpu
{
namespace kernel
{

namespace
{

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
    std::size_t coeff_count,
    std::size_t degree)
{
    // TODO:
    // Each CUDA thread should process one residue:
    //
    // linear_index -> local_limb, local_coeff
    // modulus      -> q_primes[modulus_offset + local_limb]
    // offset       -> local_limb * coeff_count + local_coeff
    // destination_values[offset] =
    //     left_values[offset] + right_values[offset] mod modulus
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

    GpuWide sum =
        static_cast<GpuWide>(left_values[offset]) +
        static_cast<GpuWide>(right_values[offset]);

    if (sum >= modulus)
    {
        sum -= modulus;
    }

    destination_values[offset] = static_cast<GpuWord>(sum);

    (void)degree;
}

/**
 * @brief CUDA kernel skeleton for modular subtraction.
 */
__global__ void sub_poly_shard_kernel(
    GpuWord *destination_values,
    const GpuWord *left_values,
    const GpuWord *right_values,
    const GpuWord *q_primes,
    std::size_t limb_count,
    std::size_t coeff_count,
    std::size_t degree)
{
    // TODO:
    // destination = left - right mod q

    (void)destination_values;
    (void)left_values;
    (void)right_values;
    (void)q_primes;
    (void)limb_count;
    (void)coeff_count;
    (void)degree;
}

/**
 * @brief CUDA kernel skeleton for modular negation.
 */
__global__ void negate_poly_shard_kernel(
    GpuWord *destination_values,
    const GpuWord *source_values,
    const GpuWord *q_primes,
    std::size_t limb_count,
    std::size_t coeff_count,
    std::size_t degree)
{
    // TODO:
    // destination = -source mod q

    (void)destination_values;
    (void)source_values;
    (void)q_primes;
    (void)limb_count;
    (void)coeff_count;
    (void)degree;
}

/**
 * @brief CUDA kernel skeleton for copying one shard.
 */
__global__ void copy_poly_shard_kernel(
    GpuWord *destination_values,
    const GpuWord *source_values,
    std::size_t limb_count,
    std::size_t coeff_count,
    std::size_t degree)
{
    // TODO:
    // destination = source

    (void)destination_values;
    (void)source_values;
    (void)limb_count;
    (void)coeff_count;
    (void)degree;
}

/**
 * @brief CUDA kernel skeleton for dyadic product.
 */
__global__ void dyadic_product_poly_shard_kernel(
    GpuWord *destination_values,
    const GpuWord *left_values,
    const GpuWord *right_values,
    const GpuWord *q_primes,
    const GpuWord *q_modulus_constants,
    std::size_t limb_count,
    std::size_t coeff_count,
    std::size_t degree)
{
    // TODO:
    // destination = left * right mod q
    //
    // Since residues are 32-bit, multiplication should use GpuWide internally.
    // q_modulus_constants is reserved for Barrett/Montgomery reduction data.

    (void)destination_values;
    (void)left_values;
    (void)right_values;
    (void)q_primes;
    (void)q_modulus_constants;
    (void)limb_count;
    (void)coeff_count;
    (void)degree;
}

/**
 * @brief CUDA kernel skeleton for dyadic multiply-accumulate.
 */
__global__ void multiply_accumulate_poly_shard_kernel(
    GpuWord *destination_values,
    const GpuWord *left_values,
    const GpuWord *right_values,
    const GpuWord *q_primes,
    const GpuWord *q_modulus_constants,
    std::size_t limb_count,
    std::size_t coeff_count,
    std::size_t degree)
{
    // TODO:
    // destination = destination + left * right mod q
    //
    // Used by ciphertext-ciphertext multiplication component convolution.

    (void)destination_values;
    (void)left_values;
    (void)right_values;
    (void)q_primes;
    (void)q_modulus_constants;
    (void)limb_count;
    (void)coeff_count;
    (void)degree;
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
        destination_shard.coeff_count,
        degree);
    
    // 判断kernel是否正确发射，不代表计算完成
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_add_poly_shard kernel launch");
    
    // kernel计算异步，CPU等待发射到GPU的计算任务完成，初版用来测试
    gpu_check_cuda(
        cudaDeviceSynchronize(),
        "launch_add_poly_shard kernel sync");
}

void launch_sub_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &left_shard,
    const GpuConstPolyShardView &right_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    // TODO:
    // Host-side launcher for modular subtraction.
    // It should launch sub_poly_shard_kernel.

    (void)destination_shard;
    (void)left_shard;
    (void)right_shard;
    (void)parameter_shard;
    (void)degree;

    throw std::runtime_error("kernel::launch_sub_poly_shard is not implemented yet");
}

void launch_negate_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    // TODO:
    // Host-side launcher for modular negation.
    // It should launch negate_poly_shard_kernel.

    (void)destination_shard;
    (void)source_shard;
    (void)parameter_shard;
    (void)degree;

    throw std::runtime_error("kernel::launch_negate_poly_shard is not implemented yet");
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

void launch_dyadic_product_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &left_shard,
    const GpuConstPolyShardView &right_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    // TODO:
    // Host-side launcher for dyadic modular product.
    //
    // It should launch dyadic_product_poly_shard_kernel with:
    // - parameter_shard.q_primes.data()
    // - parameter_shard.q_modulus_constants.data()

    (void)destination_shard;
    (void)left_shard;
    (void)right_shard;
    (void)parameter_shard;
    (void)degree;

    throw std::runtime_error("kernel::launch_dyadic_product_poly_shard is not implemented yet");
}

void launch_multiply_accumulate_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &left_shard,
    const GpuConstPolyShardView &right_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    // TODO:
    // Host-side launcher for dyadic modular multiply-accumulate.
    //
    // It should launch multiply_accumulate_poly_shard_kernel with:
    // - destination_shard.ptr
    // - left_shard.ptr
    // - right_shard.ptr
    // - parameter_shard.q_primes.data()
    // - parameter_shard.q_modulus_constants.data()

    (void)destination_shard;
    (void)left_shard;
    (void)right_shard;
    (void)parameter_shard;
    (void)degree;

    throw std::runtime_error("kernel::launch_multiply_accumulate_poly_shard is not implemented yet");
}

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
