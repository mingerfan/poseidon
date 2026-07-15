#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_linear_transform.h"
#include "poseidon/gpu/gpu_parameter.h"

#include "poseidon/gpu/gpu_elementwise_handler.h"
#include "poseidon/gpu/gpu_keyswitch_handler.h"
#include "poseidon/gpu/gpu_ntt_handler.h"
#include "poseidon/gpu/gpu_modswitch_handler.h"

#include <cstdint>

namespace poseidon
{
namespace gpu
{

/**
 * @brief Top-level GPU evaluator.
 *
 * This class is the highest-level GPU homomorphic operation interface.
 *
 * It plays a role similar to Cheddar's Context operation entry, but uses
 * Poseidon-style naming and Poseidon-style component vectors.
 *
 * Responsibilities:
 * - check FHE semantic validity;
 * - prepare destination metadata;
 * - prepare destination storage;
 * - create views;
 * - select the proper handler.
 *
 * It does not directly launch CUDA kernels.
 * Kernel launch planning belongs to handler classes.
 */
class GpuEvaluator
{
public:
    explicit GpuEvaluator(const GpuParameterData &params);

    void add(
        const GpuCiphertextData &left_ciphertext,
        const GpuCiphertextData &right_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void sub(
        const GpuCiphertextData &left_ciphertext,
        const GpuCiphertextData &right_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void negate(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void add_plain(
        const GpuCiphertextData &source_ciphertext,
        const GpuPlaintextData &source_plaintext,
        GpuCiphertextData &destination_ciphertext) const;

    void sub_plain(
        const GpuCiphertextData &source_ciphertext,
        const GpuPlaintextData &source_plaintext,
        GpuCiphertextData &destination_ciphertext) const;

    void multiply_plain(
        const GpuCiphertextData &source_ciphertext,
        const GpuPlaintextData &source_plaintext,
        GpuCiphertextData &destination_ciphertext) const;

    void ntt_fwd(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void ntt_inv(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void multiply(
        const GpuCiphertextData &left_ciphertext,
        const GpuCiphertextData &right_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void square(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void rescale(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void rescale_dynamic(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext,
        double min_scale) const;

    void drop_modulus(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext,
        parms_id_type target_parms_id) const;

    /**
     * @brief Multiply every ciphertext component by one integer scalar modulo q.
     *
     * This is primarily used by bootstrap scale matching before ModRaise.
     */
    void multiply_scalar(
        const GpuCiphertextData &source_ciphertext,
        std::uint64_t scalar,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief CKKS bootstrap ModRaise.
     *
     * Mirrors EvaluatorCkksBase::raise_modulus for the bootstrap path:
     * source is transformed to coefficient domain if needed, converted from
     * its current q-only prefix level to the first q-only Q level, then
     * transformed back to NTT form.
     */
    void raise_modulus(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Prepare a CKKS ciphertext for bootstrap ModRaise.
     *
     * Mirrors the CPU bootstrap prefix up to the input of raise_modulus:
     * - repeatedly uses ordinary rescale until scale <= 2^54;
     * - drops to q0+1 if needed;
     * - multiplies by round(q0_over_message_ratio / scale) when > 1;
     * - drops to q0.
     *
     * The result is still q-only and in the same NTT form as the input.
     * This function intentionally does not perform Q0 -> QL basis extension.
     */
    void bootstrap_prepare_modraise_input(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext,
        parms_id_type q0_parms_id,
        double q0_over_message_ratio) const;

    /**
     * @brief Relinearize ciphertext.
     *
     * Dispatches size-3 CKKS ciphertexts to the HYBRID key-switch handler and
     * writes a size-2 ciphertext in NTT form.
     */
    void relinearize(
        const GpuCiphertextData &source_ciphertext,
        const GpuRelinKeysData &relin_keys,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Rotate ciphertext.
     *
     * First CKKS implementation:
     * - apply NTT-domain Galois permutation to c0/c1;
     * - key-switch the permuted c1 with the uploaded Galois key;
     * - currently requires the direct Galois key for the requested step.
     */
    void rotate(
        const GpuCiphertextData &source_ciphertext,
        int step,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Conjugate ciphertext.
     *
     * First CKKS implementation:
     * - apply NTT-domain conjugation Galois permutation to c0/c1;
     * - key-switch the permuted c1 with the uploaded conjugation key.
     */
    void conjugate(
        const GpuCiphertextData &source_ciphertext,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Multiply by one pre-uploaded diagonal plaintext matrix using BSGS.
     *
     * This mirrors the current GPU bootstrap test reference while using
     * GPU rotate/multiply_plain/add/rescale primitives. Delayed/dynamic
     * rescale is intentionally not used in this temporary bootstrapping path.
     */
    void multiply_by_diag_matrix_bsgs(
        const GpuCiphertextData &source_ciphertext,
        const GpuMatrixPlain &matrix,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Apply a pre-uploaded DFT linear matrix group.
     */
    void dft(
        const GpuCiphertextData &source_ciphertext,
        const GpuLinearMatrixGroup &matrix_group,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief CKKS CoeffToSlot with CPU-precomputed matrices already on GPU.
     *
     * minus_i_plaintext must be the CPU-encoded plaintext for complex(0, -1)
     * at the DFT output parms_id and scale 1.0, uploaded to GPU outside the
     * timed GPU path.
     */
    void coeff_to_slot(
        const GpuCiphertextData &source_ciphertext,
        const GpuLinearMatrixGroup &matrix_group,
        const GpuPlaintextData &minus_i_plaintext,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &result_real,
        GpuCiphertextData &result_imag) const;

    /**
     * @brief CKKS SlotToCoeff with CPU-precomputed inverse DFT matrices already on GPU.
     *
     * plus_i_plaintext must be the CPU-encoded plaintext for complex(0, 1)
     * at the input imag ciphertext parms_id and scale 1.0, uploaded to GPU
     * outside the timed GPU path.
     */
    void slot_to_coeff(
        const GpuCiphertextData &source_real,
        const GpuCiphertextData &source_imag,
        const GpuLinearMatrixGroup &matrix_group,
        const GpuPlaintextData &plus_i_plaintext,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &result) const;

private:
    const GpuParameterData &params_;

    GpuElementwiseHandler elementwise_handler_;
    GpuKeySwitchHandler keyswitch_handler_;
    GpuNTTHandler ntt_handler_;
    GpuModSwitchHandler modswitch_handler_;
};

}  // namespace gpu
}  // namespace poseidon
