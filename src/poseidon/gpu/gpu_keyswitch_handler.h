#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_parameter.h"

namespace poseidon
{
namespace gpu
{

/**
 * @brief GPU key-switching operations.
 *
 * Current scope:
 * - CKKS HYBRID relinearization framework;
 * - single-GPU, full-coefficient, q-only input ciphertexts;
 * - key layout lookup and operation validation.
 *
 * The arithmetic pipeline is intentionally split from GpuEvaluator so the
 * eventual HYBRID ModUp/key-multiply/ModDown kernels have one owner.
 */
class GpuKeySwitchHandler
{
public:
    explicit GpuKeySwitchHandler(const GpuParameterData &params);

    void relinearize_hybrid_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &source_view,
        const GpuConstEvaluationKeyView &relin_keys_view,
        const GpuEvaluationKeyData &relin_keys_data,
        const GpuLevelInfo &level_info) const;

private:
    const GpuParameterData &params_;
};

}  // namespace gpu
}  // namespace poseidon
