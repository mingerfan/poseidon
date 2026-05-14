#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_parameter.h"

namespace poseidon
{
namespace gpu
{

/**
 * @brief Handler for coefficient-wise GPU operations.
 *
 * This class corresponds to Cheddar's ElementWiseHandler layer, but it is
 * adapted to Poseidon's component-vector ciphertext structure.
 *
 * Poseidon-style ciphertext representation:
 * - ciphertext_view.polys[0] represents c0;
 * - ciphertext_view.polys[1] represents c1;
 * - ciphertext_view.polys[2] represents c2, if present.
 *
 * This handler must not assume Cheddar-style fixed ax/bx/rx members.
 *
 * Responsibilities:
 * - validate physical placement compatibility;
 * - translate RNS polynomial views into launch tasks;
 * - call kernel launcher functions in poseidon/gpu/kernels.
 *
 * Current stage:
 * - Only framework-level function interfaces are defined.
 * - Real placement checking, task construction, and CUDA launch logic are TODO.
 */
class GpuElementwiseHandler
{
public:
    explicit GpuElementwiseHandler(const GpuParameterData &params);

    /**
     * @brief destination_ciphertext = left_ciphertext + right_ciphertext.
     *
     * Poseidon component-vector rule:
     * - common components are added component-wise;
     * - extra components from the larger ciphertext are copied.
     */
    void add_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &left_view,
        const GpuConstCiphertextView &right_view,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief destination_ciphertext = left_ciphertext - right_ciphertext.
     *
     * Poseidon component-vector rule:
     * - common components are subtracted component-wise;
     * - extra left components are copied;
     * - extra right components are negated.
     */
    void sub_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &left_view,
        const GpuConstCiphertextView &right_view,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief destination_ciphertext = -source_ciphertext.
     */
    void negate_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &source_view,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief destination_ciphertext = source_ciphertext + source_plaintext.
     *
     * CKKS rule:
     * - plaintext is added only to ciphertext component c0;
     * - other ciphertext components are copied.
     */
    void add_plain_to_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &ciphertext_view,
        const GpuConstPlaintextView &plaintext_view,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief destination_ciphertext = source_ciphertext - source_plaintext.
     *
     * CKKS rule:
     * - plaintext is subtracted only from ciphertext component c0;
     * - other ciphertext components are copied.
     */
    void sub_plain_from_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &ciphertext_view,
        const GpuConstPlaintextView &plaintext_view,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief destination_ciphertext = source_ciphertext * source_plaintext.
     *
     * CKKS rule:
     * - every ciphertext component is multiplied by the plaintext polynomial.
     */
    void multiply_plain_with_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &ciphertext_view,
        const GpuConstPlaintextView &plaintext_view,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief destination_ciphertext = left_ciphertext * right_ciphertext.
     *
     * This corresponds to Cheddar-style Tensor operation, but adapted to
     * Poseidon's component-vector structure.
     *
     * Poseidon component convolution rule:
     * - destination.polys[i + j] accumulates
     *   left.polys[i] * right.polys[j].
     */
    void multiply_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &left_view,
        const GpuConstCiphertextView &right_view,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief destination_ciphertext = source_ciphertext^2.
     *
     * This is an optimized form of component-vector convolution.
     */
    void square_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &source_view,
        const GpuLevelInfo &level_info) const;

private:
    /**
     * @brief Add two logical RNS polynomials.
     *
     * Future implementation:
     * - validate shard alignment;
     * - find matching parameter shard;
     * - call kernel::launch_add_poly_shard(...).
     */
    void add_poly(
        GpuRNSPolyView &destination_poly,
        const GpuConstRNSPolyView &left_poly,
        const GpuConstRNSPolyView &right_poly,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief Subtract two logical RNS polynomials.
     *
     * Future implementation:
     * - validate shard alignment;
     * - find matching parameter shard;
     * - call kernel::launch_sub_poly_shard(...).
     */
    void sub_poly(
        GpuRNSPolyView &destination_poly,
        const GpuConstRNSPolyView &left_poly,
        const GpuConstRNSPolyView &right_poly,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief Negate one logical RNS polynomial.
     *
     * Future implementation:
     * - validate shard alignment;
     * - find matching parameter shard;
     * - call kernel::launch_negate_poly_shard(...).
     */
    void negate_poly(
        GpuRNSPolyView &destination_poly,
        const GpuConstRNSPolyView &source_poly,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief Copy one logical RNS polynomial.
     *
     * Used for extra ciphertext components and plain-add/plain-sub component
     * propagation.
     *
     * Future implementation:
     * - validate shard alignment;
     * - call kernel::launch_copy_poly_shard(...).
     */
    void copy_poly(
        GpuRNSPolyView &destination_poly,
        const GpuConstRNSPolyView &source_poly,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief Multiply one ciphertext component by plaintext polynomial.
     *
     * Future implementation:
     * - validate shard alignment;
     * - find matching parameter shard;
     * - call kernel::launch_dyadic_product_poly_shard(...).
     */
    void multiply_plain_poly(
        GpuRNSPolyView &destination_poly,
        const GpuConstRNSPolyView &ciphertext_poly,
        const GpuConstRNSPolyView &plaintext_poly,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief Accumulate product of two ciphertext component polynomials.
     *
     * Used by multiply_ciphertext and square_ciphertext.
     *
     * Future implementation:
     * - validate shard alignment;
     * - find matching parameter shard;
     * - call kernel::launch_multiply_accumulate_poly_shard(...).
     */
    void multiply_accumulate_poly(
        GpuRNSPolyView &destination_poly,
        const GpuConstRNSPolyView &left_poly,
        const GpuConstRNSPolyView &right_poly,
        const GpuLevelInfo &level_info) const;

private:
    const GpuParameterData &params_;
};

}  // namespace gpu
}  // namespace poseidon