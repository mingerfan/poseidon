#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_parameter.h"

namespace poseidon
{
namespace gpu
{

/**
 * @brief GPU evaluator interface.
 *
 * This class should only depend on GPU-side data objects:
 * - GpuCiphertextData;
 * - GpuPlaintextData;
 * - GpuRelinKeysData;
 * - GpuGaloisKeysData;
 * - GpuParameterData.
 *
 * It should not directly depend on PoseidonContext during operator execution.
 */
class GpuEvaluator
{
public:
    explicit GpuEvaluator(const GpuParameterData &params);

    void add(
        const GpuCiphertextData &a,
        const GpuCiphertextData &b,
        GpuCiphertextData &res) const;

    void sub(
        const GpuCiphertextData &a,
        const GpuCiphertextData &b,
        GpuCiphertextData &res) const;

    void negate(
        const GpuCiphertextData &a,
        GpuCiphertextData &res) const;

    void add_plain(
        const GpuCiphertextData &ct,
        const GpuPlaintextData &pt,
        GpuCiphertextData &res) const;

    void sub_plain(
        const GpuCiphertextData &ct,
        const GpuPlaintextData &pt,
        GpuCiphertextData &res) const;

    void multiply_plain(
        const GpuCiphertextData &ct,
        const GpuPlaintextData &pt,
        GpuCiphertextData &res) const;

    void ntt_fwd(
        const GpuCiphertextData &ct,
        GpuCiphertextData &res) const;

    void ntt_inv(
        const GpuCiphertextData &ct,
        GpuCiphertextData &res) const;

    void multiply(
        const GpuCiphertextData &a,
        const GpuCiphertextData &b,
        GpuCiphertextData &res) const;

    void square(
        const GpuCiphertextData &a,
        GpuCiphertextData &res) const;

    void rescale(
        const GpuCiphertextData &ct,
        GpuCiphertextData &res) const;

    void rescale_dynamic(
        const GpuCiphertextData &ct,
        GpuCiphertextData &res,
        double min_scale) const;

    void drop_modulus(
        const GpuCiphertextData &ct,
        GpuCiphertextData &res,
        parms_id_type target_parms_id) const;

    void relinearize(
        const GpuCiphertextData &ct,
        const GpuRelinKeysData &relin_keys,
        GpuCiphertextData &res) const;

    void rotate(
        const GpuCiphertextData &ct,
        int step,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &res) const;

    void conjugate(
        const GpuCiphertextData &ct,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &res) const;

private:
    const GpuParameterData &params_;
};

}  // namespace gpu
}  // namespace poseidon