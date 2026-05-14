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
    std::size_t limb_count,
    std::size_t coeff_count,
    std::size_t degree)
{
    // TODO:
    // Each CUDA thread should process one residue:
    //
    // linear_index -> local_limb, local_coeff
    // modulus      -> q_primes[local_limb]
    // offset       -> local_limb * degree + local_coeff
    // destination_values[offset] =
    //     left_values[offset] + right_values[offset] mod modulus

    (void)destination_values;
    (void)left_values;
    (void)right_values;
    (void)q_primes;
    (void)limb_count;
    (void)coeff_count;
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
    // TODO:
    // Host-side launcher for modular addition.
    //
    // Expected future behavior:
    // 1. cudaSetDevice(destination_shard.device_id)
    // 2. Validate that all shard pointers are non-null.
    // 3. Compute:
    //      total_count = limb_count * coeff_count
    //      block_size  = e.g. 256
    //      grid_size   = ceil(total_count / block_size)
    // 4. Launch:
    //      add_poly_shard_kernel<<<grid_size, block_size>>>(
    //          destination_shard.ptr,
    //          left_shard.ptr,
    //          right_shard.ptr,
    //          parameter_shard.q_primes.data(),
    //          destination_shard.limb_count,
    //          destination_shard.coeff_count,
    //          degree)
    // 5. Check CUDA errors.

    (void)destination_shard;
    (void)left_shard;
    (void)right_shard;
    (void)parameter_shard;
    (void)degree;

    throw std::runtime_error("kernel::launch_add_poly_shard is not implemented yet");
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
    // TODO:
    // Host-side launcher for shard copy.
    // It should launch copy_poly_shard_kernel or use cudaMemcpyAsync if suitable.

    (void)destination_shard;
    (void)source_shard;
    (void)degree;

    throw std::runtime_error("kernel::launch_copy_poly_shard is not implemented yet");
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