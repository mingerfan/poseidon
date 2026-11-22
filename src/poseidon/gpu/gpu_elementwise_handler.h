#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"

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

    /** Exact modular sum of 1..4 plaintext products, with no intermediate CT. */
    void multiply_plain_sum(
        GpuCiphertextView &destination_view,
        const std::vector<GpuConstCiphertextView> &ciphertexts,
        const std::vector<GpuConstPlaintextView> &plaintexts,
        const GpuLevelInfo &level_info,
        bool accumulate = false) const;

    void double_sub_plain(
        GpuCiphertextView &destination,
        const GpuConstCiphertextView &source,
        const GpuConstPlaintextView &plaintext,
        const GpuConstCiphertextView *correction,
        const GpuLevelInfo &level) const;

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
     * @brief destination_ciphertext = source_ciphertext * scalar mod q.
     *
     * CKKS bootstrap scale matching uses this for integer constant
     * multiplication without constructing a full plaintext. The full uint64
     * scalar is reduced modulo each actual q prime before coefficient multiply.
     */
    void multiply_scalar_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &source_view,
        std::uint64_t scalar,
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

    /** Add/subtract a plaintext directly in c0, preserving c1/c2 in place. */
    void add_plain_to_ciphertext_inplace(
        GpuCiphertextView &ciphertext_view,
        const GpuConstPlaintextView &plaintext_view,
        const GpuLevelInfo &level_info) const;

    void sub_plain_from_ciphertext_inplace(
        GpuCiphertextView &ciphertext_view,
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
     * @brief destination[0..1] += ciphertext[0..1] * plaintext.
     *
     * The source ciphertext must have two components. The destination may have
     * two or more components; only c0/c1 are updated. This lets EvalMod add a
     * linear leaf into a size-3 multiplication accumulator before
     * relinearize-rescale, without touching c2.
     */
    void multiply_plain_accumulate_with_ciphertext(
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

    /** Multiply two ciphertexts by one shared plaintext in a single launch per shard. */
    void multiply_plain_pair(
        GpuCiphertextView &destination_a_view,
        GpuCiphertextView &destination_b_view,
        const GpuConstCiphertextView &ciphertext_a_view,
        const GpuConstCiphertextView &ciphertext_b_view,
        const GpuConstRNSPolyView &plaintext_poly,
        const GpuLevelInfo &level_info) const;

    /** Add or subtract two ciphertext pairs in a single launch per shard. */
    void add_or_sub_ciphertext_pair(
        GpuCiphertextView &destination_a_view,
        GpuCiphertextView &destination_b_view,
        const GpuConstCiphertextView &left_a_view,
        const GpuConstCiphertextView &right_a_view,
        const GpuConstCiphertextView &left_b_view,
        const GpuConstCiphertextView &right_b_view,
        bool subtract,
        const GpuLevelInfo &level_info) const;

    /** Fused multiply for two ciphertexts across two lanes (or single lane when b views omitted). */
    void multiply_ciphertext_pair(
        GpuCiphertextView &destination_a_view,
        GpuCiphertextView &destination_b_view,
        const GpuConstCiphertextView &left_a_view,
        const GpuConstCiphertextView &right_a_view,
        const GpuConstCiphertextView &left_b_view,
        const GpuConstCiphertextView &right_b_view,
        const GpuLevelInfo &level_info) const;

    /** Multiply up to kMaxElementwiseCiphertextBatch ciphertext pairs at the same level. */
    void multiply_ciphertext_batch(
        std::vector<GpuCiphertextView> &destination_views,
        const std::vector<GpuConstCiphertextView> &left_views,
        const std::vector<GpuConstCiphertextView> &right_views,
        bool is_square,
        const GpuLevelInfo &level_info) const;

    /** Multiply up to 8 size-2 ciphertexts by one shared plaintext at the same level. */
    void multiply_plain_batch(
        std::vector<GpuCiphertextView> &destination_views,
        const std::vector<GpuConstCiphertextView> &ciphertext_views,
        const GpuConstPlaintextView &plaintext_view,
        const GpuLevelInfo &level_info) const;

    /** Add or subtract up to 8 size-2 ciphertext pairs at the same level. */
    void add_or_sub_ciphertext_batch(
        std::vector<GpuCiphertextView> &destination_views,
        const std::vector<GpuConstCiphertextView> &left_views,
        const std::vector<GpuConstCiphertextView> &right_views,
        bool subtract,
        const GpuLevelInfo &level_info) const;

    /** Fused square for two ciphertexts across two lanes. */
    void square_ciphertext_pair(
        GpuCiphertextView &destination_a_view,
        GpuCiphertextView &destination_b_view,
        const GpuConstCiphertextView &source_a_view,
        const GpuConstCiphertextView &source_b_view,
        const GpuLevelInfo &level_info) const;

    /** Fused 2*src +/- plain (or plain * corr) across one or two lanes. */
    void double_sub_or_add_plain_pair(
        kernel::DoublePlainPairOp op,
        GpuCiphertextView &destination_a_view,
        GpuCiphertextView *destination_b_view,
        const GpuConstCiphertextView &source_a_view,
        const GpuConstCiphertextView *source_b_view,
        const GpuConstRNSPolyView &plaintext_poly,
        const GpuConstCiphertextView *correction_a_view,
        const GpuConstCiphertextView *correction_b_view,
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
