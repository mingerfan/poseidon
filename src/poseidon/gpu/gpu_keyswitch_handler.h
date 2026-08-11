#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_parameter.h"

#include <cstddef>
#include <cstdint>
#include <memory>

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
    ~GpuKeySwitchHandler();

    GpuKeySwitchHandler(const GpuKeySwitchHandler &) = delete;
    GpuKeySwitchHandler &operator=(const GpuKeySwitchHandler &) = delete;

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
        bool overwrite_destination1) const;

    /**
     * @brief Rotate and HYBRID key-switch through one cached CUDA Graph.
     *
     * Graph use is selected by POSEIDON_ROTATE_CUDA_GRAPH. The graph owns all
     * intermediate storage; source and destination pointers are rebound for
     * each launch without copying ciphertext payloads.
     */
    void rotate_hybrid_ciphertext_graph(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &source_view,
        const GpuConstEvaluationKeyView &galois_keys_view,
        const GpuEvaluationKeyData &galois_keys_data,
        std::size_t key_index,
        std::uint32_t galois_elt,
        const GpuLevelInfo &level_info) const;

private:
    struct GraphState;

    const GpuParameterData &params_;
    mutable std::unique_ptr<GraphState> graph_state_;
};

}  // namespace gpu
}  // namespace poseidon
