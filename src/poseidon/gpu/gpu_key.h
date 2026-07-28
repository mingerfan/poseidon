#pragma once

#include "poseidon/parameters_literal.h"
#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_rns_poly.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace poseidon
{
namespace gpu
{

enum class GpuGaloisKeyFormat : std::uint8_t
{
    Standard = 0,
    InversePreRotated = 1,
};

/**
 * @brief Generic metadata for GPU evaluation keys.
 *
 * This is intentionally generic at the framework stage.
 * Relin keys and Galois keys can be specialized later if their layouts diverge.
 */
struct GpuKeyMeta
{
    parms_id_type key_parms_id{};

    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;

    std::size_t key_count = 0;
    std::size_t decomposition_count = 0;
    std::size_t component_count = 0;
    GpuGaloisKeyFormat galois_format =
        GpuGaloisKeyFormat::Standard;
};

struct GpuEvaluationKeyPolyMeta
{
    std::size_t poly_id = 0;
    std::size_t key_index = 0;
    std::size_t decomposition_index = 0;
    std::size_t component_index = 0;
};

struct GpuEvaluationKeyView
{
    GpuKeyMeta meta;
    std::vector<GpuRNSPolyView> polys;
};

struct GpuConstEvaluationKeyView
{
    /** Active Q prefix exposed by this non-owning level view. */
    GpuKeyMeta meta;

    /** Q limb count in the single owning [Q_storage | P] allocation. */
    std::size_t storage_q_count = 0;
    std::vector<GpuConstRNSPolyView> polys;
};

/**
 * @brief Generic GPU-side evaluation key container.
 *
 * Intended to represent:
 * - relinearization keys;
 * - Galois/rotation keys;
 * - other key-switching keys.
 *
 * The key data also uses the same field/shard/poly model:
 * - fields_ owns GPU memory;
 * - polys_ describes logical key polynomials;
 * - each key polynomial can be sharded across GPUs.
 */
class GpuEvaluationKeyData
{
public:
    GpuKeyMeta meta;

    /**
     * @brief Real GPU memory blocks.
     */
    std::vector<GpuFieldData> fields_;

    /**
     * @brief Logical key RNS polynomials.
     *
     * Each key polynomial may have one or more shards.
     */
    std::vector<GpuRNSPoly> polys_;

    /**
     * @brief Mapping from a flattened GPU key polynomial back to Poseidon's
     * keys_[key_index][decomposition_index].data(component_index) layout.
     */
    std::vector<GpuEvaluationKeyPolyMeta> poly_metadata_;

    /** Actual automorphism indexed by CPU Galois-key slot. */
    std::vector<std::uint32_t> galois_elts_by_key_index;

    /**
     * Setup-time device pointer tables for inverse-pre-rotated Galois keys.
     * Flattened index = key_index * decomposition_count + digit.
     */
    DeviceVector<const GpuWord *> galois_key_q0_ptrs;
    DeviceVector<const GpuWord *> galois_key_p0_ptrs;
    DeviceVector<const GpuWord *> galois_key_q1_ptrs;
    DeviceVector<const GpuWord *> galois_key_p1_ptrs;

public:
    GpuEvaluationKeyData() = default;

    bool empty() const;

    /**
     * @brief Create mutable view of evaluation key data.
     */
    GpuEvaluationKeyView make_view();

    /**
     * @brief Create const view of evaluation key data.
     */
    GpuConstEvaluationKeyView make_const_view() const;

    /**
     * @brief Create a zero-copy view for a lower Q level.
     *
     * The owning key remains one contiguous [Q_storage | P] allocation. The
     * returned view exposes Q[0, active_q_count) while retaining
     * storage_q_count so KeySwitch can address P at Q_storage * degree.
     */
    GpuConstEvaluationKeyView make_const_view(
        std::size_t active_q_count) const;
};

using GpuRelinKeysData = GpuEvaluationKeyData;
using GpuGaloisKeysData = GpuEvaluationKeyData;

}  // namespace gpu
}  // namespace poseidon
