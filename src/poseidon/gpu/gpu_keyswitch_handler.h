#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_double_hoist.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_parameter.h"

#include <cstddef>
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

    void relinearize_hybrid_ciphertext_rescale_x2(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &source_view,
        const GpuConstEvaluationKeyView &relin_keys_view,
        const GpuEvaluationKeyData &relin_keys_data,
        const GpuLevelInfo &source_level_info,
        const GpuLevelInfo &destination_level_info) const;

    void switch_key_hybrid_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstRNSPolyView &switch_poly_ntt,
        const GpuConstEvaluationKeyView &switch_keys_view,
        const GpuEvaluationKeyData &switch_keys_data,
        std::size_t key_index,
        const GpuLevelInfo &level_info) const;

    /**
     * Decompose one Q/NTT polynomial once and retain every HYBRID digit in
     * contiguous Q/P NTT buffers.
     */
    void hoist_decompose_modup_ntt(
        const GpuConstRNSPolyView &switch_poly_ntt,
        const GpuLevelInfo &level_info,
        GpuHoistedDecomposition &destination,
        GpuHybridKeySwitchWorkspace &workspace) const;

    /**
     * Standard-key staged path. The automorphism is applied to each already
     * hoisted digit, then all key products are accumulated in QP without a
     * ModDown. initialize=true overwrites the selected destination batch.
     */
    void keyswitch_multsum_no_moddown(
        const GpuHoistedDecomposition &hoisted,
        std::uint32_t galois_elt,
        const GpuConstEvaluationKeyView &keys,
        const GpuEvaluationKeyData &key_storage,
        std::size_t key_index,
        GpuQPCiphertextBuffer &destination,
        std::size_t destination_batch,
        bool initialize,
        const GpuLevelInfo &level_info,
        GpuHybridKeySwitchWorkspace &workspace) const;

    /**
     * Apply HYBRID P->Q ModDown to one QP ciphertext batch. P INTT and its
     * conversion scratch are persistent in workspace.
     */
    void moddown_qp_ciphertext_to_q(
        GpuQPCiphertextBuffer &source,
        std::size_t source_batch,
        GpuCiphertextData &destination,
        const GpuCiphertextMeta &destination_meta,
        const GpuLevelInfo &level_info,
        GpuHybridKeySwitchWorkspace &workspace) const;

    /**
     * Apply the same HYBRID P->Q ModDown to a contiguous group batch. The
     * P-side INTT/convert/Q-side NTT are issued over all 2*batch components,
     * so giant groups no longer serialize complete ModDown pipelines.
     */
    void moddown_qp_ciphertext_batch_to_q(
        GpuQPCiphertextBuffer &source,
        std::size_t batch_count,
        GpuQCiphertextBatchBuffer &destination,
        const GpuLevelInfo &level_info,
        DeviceVector<GpuWord> &p_coeff,
        DeviceVector<GpuWord> &converted_q) const;

    /**
     * Final double-hoist ModDown. Only P is reduced explicitly; Q group
     * reduction is fused with subtraction and multiplication by P^{-1}.
     */
    void moddown_qp_groups_to_q(
        GpuQPCiphertextBuffer &source_groups,
        std::size_t group_count,
        DeviceVector<GpuWord> &reduced_p_ntt,
        GpuCiphertextData &destination,
        const GpuCiphertextMeta &destination_meta,
        const GpuLevelInfo &level_info,
        GpuHybridKeySwitchWorkspace &workspace) const;

private:
    void switch_key_hybrid_ciphertext_impl(
        GpuCiphertextView &destination_view,
        const GpuConstRNSPolyView &switch_poly_ntt,
        const GpuConstEvaluationKeyView &switch_keys_view,
        const GpuEvaluationKeyData &switch_keys_data,
        std::size_t key_index,
        const GpuLevelInfo &level_info,
        const GpuConstRNSPolyView *add_source0,
        const GpuConstRNSPolyView *add_source1,
        const GpuLevelInfo *rescale_x2_destination_level = nullptr) const;

    struct PersistentWorkspace;

    const GpuParameterData &params_;
    mutable std::unique_ptr<PersistentWorkspace> persistent_workspace_;
};

}  // namespace gpu
}  // namespace poseidon
