#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_parameter.h"

#include "poseidon/gpu/gpu_elementwise_handler.h"
#include "poseidon/gpu/gpu_keyswitch_handler.h"
#include "poseidon/gpu/gpu_ntt_handler.h"
#include "poseidon/gpu/gpu_modswitch_handler.h"

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

private:
    const GpuParameterData &params_;

    GpuElementwiseHandler elementwise_handler_;
    GpuKeySwitchHandler keyswitch_handler_;
    GpuNTTHandler ntt_handler_;
    GpuModSwitchHandler modswitch_handler_;
};

}  // namespace gpu
}  // namespace poseidon
