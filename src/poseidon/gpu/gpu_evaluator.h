#pragma once

#include "poseidon/poseidon_context.h"
#include "poseidon/gpu/gpu_ciphertext.h"

namespace poseidon
{
namespace gpu
{

/**
 * @brief GPU evaluator interface.
 *
 * This class is not intended to replace EvaluatorCkksBase immediately.
 * It provides a separate GPU path that consumes GpuCiphertextData.
 *
 * First-stage target:
 * - add
 * - sub
 * - negate
 *
 * Later:
 * - ntt_fwd / ntt_inv
 * - add_plain / multiply_plain
 * - multiply
 * - rescale
 * - relinearize
 * - rotate
 */
class GpuEvaluator
{
public:
    explicit GpuEvaluator(const PoseidonContext &context)
        : context_(context)
    {}

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

private:
    const PoseidonContext &context_;

    // TODO:
    // Add GPU-side modulus/NTT table cache here.
    // This should not replace PoseidonContext.
    // It only stores kernel-readable device copies of q primes, p primes,
    // Barrett/Montgomery constants, and NTT roots.
};

}  // namespace gpu
}  // namespace poseidon