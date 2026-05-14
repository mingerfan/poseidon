#include "poseidon/gpu/gpu_elementwise_handler.h"
#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"

#include <stdexcept>

namespace poseidon
{
namespace gpu
{

GpuElementwiseHandler::GpuElementwiseHandler(const GpuParameterData &params)
    : params_(params)
{}

void GpuElementwiseHandler::add_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &left_view,
    const GpuConstCiphertextView &right_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // 1. Check physical placement compatibility among destination/left/right.
    // 2. Add common Poseidon components through add_poly().
    // 3. Copy extra components from the larger ciphertext through copy_poly().
    // 4. add_poly() should eventually call kernel::launch_add_poly_shard(...).

    (void)destination_view;
    (void)left_view;
    (void)right_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::add_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::sub_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &left_view,
    const GpuConstCiphertextView &right_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // 1. Check physical placement compatibility.
    // 2. Subtract common Poseidon components through sub_poly().
    // 3. Handle extra components:
    //    - extra left components are copied;
    //    - extra right components are negated.
    // 4. sub_poly() should eventually call kernel::launch_sub_poly_shard(...).

    (void)destination_view;
    (void)left_view;
    (void)right_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::sub_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::negate_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // Negate every ciphertext component polynomial.
    // negate_poly() should eventually call kernel::launch_negate_poly_shard(...).

    (void)destination_view;
    (void)source_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::negate_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::add_plain_to_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &ciphertext_view,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // CKKS add_plain:
    // - destination.c0 = ciphertext.c0 + plaintext.poly;
    // - destination.ci = ciphertext.ci for i > 0.
    //
    // c0 computation should eventually call add_poly().
    // other components should eventually call copy_poly().

    (void)destination_view;
    (void)ciphertext_view;
    (void)plaintext_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::add_plain_to_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::sub_plain_from_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &ciphertext_view,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // CKKS sub_plain:
    // - destination.c0 = ciphertext.c0 - plaintext.poly;
    // - destination.ci = ciphertext.ci for i > 0.
    //
    // c0 computation should eventually call sub_poly().
    // other components should eventually call copy_poly().

    (void)destination_view;
    (void)ciphertext_view;
    (void)plaintext_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::sub_plain_from_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::multiply_plain_with_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &ciphertext_view,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // For every ciphertext component:
    // - destination.ci = ciphertext.ci * plaintext.poly.
    //
    // Each component computation should eventually call multiply_plain_poly().
    // multiply_plain_poly() should call kernel::launch_dyadic_product_poly_shard(...).

    (void)destination_view;
    (void)ciphertext_view;
    (void)plaintext_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::multiply_plain_with_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::multiply_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &left_view,
    const GpuConstCiphertextView &right_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // Implement Poseidon component-vector convolution:
    //
    // for each left component index:
    //   for each right component index:
    //     destination component at index sum receives product accumulation.
    //
    // Each product accumulation should eventually call multiply_accumulate_poly().
    // multiply_accumulate_poly() should call
    // kernel::launch_multiply_accumulate_poly_shard(...).
    //
    // This corresponds to Cheddar-style Tensor, but must use polys[index].

    (void)destination_view;
    (void)left_view;
    (void)right_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::multiply_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::square_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // Implement optimized square using component-vector symmetry.
    //
    // Internally this should reuse multiply_accumulate_poly() where possible.

    (void)destination_view;
    (void)source_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::square_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::add_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &left_poly,
    const GpuConstRNSPolyView &right_poly,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // 1. Validate that destination_poly, left_poly, and right_poly have
    //    compatible shard counts and shard ranges.
    // 2. For each shard:
    //    - find the matching GpuParameterShard from level_info;
    //    - call kernel::launch_add_poly_shard(...).
    //
    // Example future call:
    //
    // kernel::launch_add_poly_shard(
    //     destination_shard,
    //     left_shard,
    //     right_shard,
    //     parameter_shard,
    //     destination_poly.degree);

    (void)destination_poly;
    (void)left_poly;
    (void)right_poly;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::add_poly is not implemented yet");
}

void GpuElementwiseHandler::sub_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &left_poly,
    const GpuConstRNSPolyView &right_poly,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // 1. Validate shard alignment.
    // 2. Find matching GpuParameterShard.
    // 3. Call kernel::launch_sub_poly_shard(...).

    (void)destination_poly;
    (void)left_poly;
    (void)right_poly;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::sub_poly is not implemented yet");
}

void GpuElementwiseHandler::negate_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &source_poly,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // 1. Validate shard alignment.
    // 2. Find matching GpuParameterShard.
    // 3. Call kernel::launch_negate_poly_shard(...).

    (void)destination_poly;
    (void)source_poly;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::negate_poly is not implemented yet");
}

void GpuElementwiseHandler::copy_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &source_poly,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // 1. Validate shard alignment.
    // 2. Call kernel::launch_copy_poly_shard(...).
    //
    // Copy does not need modulus parameters, but level_info may still be useful
    // for placement validation.

    (void)destination_poly;
    (void)source_poly;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::copy_poly is not implemented yet");
}

void GpuElementwiseHandler::multiply_plain_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &ciphertext_poly,
    const GpuConstRNSPolyView &plaintext_poly,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // 1. Validate placement.
    // 2. Find matching GpuParameterShard.
    // 3. Call kernel::launch_dyadic_product_poly_shard(...).

    (void)destination_poly;
    (void)ciphertext_poly;
    (void)plaintext_poly;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::multiply_plain_poly is not implemented yet");
}

void GpuElementwiseHandler::multiply_accumulate_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &left_poly,
    const GpuConstRNSPolyView &right_poly,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // 1. Validate shard placement.
    // 2. Find matching GpuParameterShard.
    // 3. Call kernel::launch_multiply_accumulate_poly_shard(...).

    (void)destination_poly;
    (void)left_poly;
    (void)right_poly;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::multiply_accumulate_poly is not implemented yet");
}

}  // namespace gpu
}  // namespace poseidon