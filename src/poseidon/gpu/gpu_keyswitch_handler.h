#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_parameter.h"

#include <cstddef>

namespace poseidon
{
namespace gpu
{

/**
 * @brief GPU key-switching operations.
 *
 * Current scope:
 * - CKKS HYBRID key-switching framework;
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

    void switch_key_hybrid_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstRNSPolyView &switch_poly_ntt,
        const GpuConstEvaluationKeyView &switch_keys_view,
        const GpuEvaluationKeyData &switch_keys_data,
        std::size_t key_index,
        const GpuLevelInfo &level_info,
        const GpuConstRNSPolyView *add_back_source0 = nullptr,
        const GpuConstRNSPolyView *add_back_source1 = nullptr) const;

private:
    const GpuParameterData &params_;
};

}  // namespace gpu
}  // namespace poseidon
