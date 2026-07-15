#pragma once

#include "poseidon/parameters_literal.h"
#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_rns_poly.h"

#include <cstddef>
#include <map>
#include <memory>
#include <vector>

namespace poseidon
{
namespace gpu
{

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
    GpuKeyMeta meta;
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

    /**
     * @brief Optional setup-time compacted key copies indexed by active Q limb count.
     *
     * HYBRID key-switching needs a physical [Q_current | P] key layout.  CPU
     * keys are uploaded at the full key level, so bootstrapping rotations at
     * lower levels should precompute these compact views once during setup
     * instead of repacking keys inside every timed rotate call.
     */
    std::map<std::size_t, std::shared_ptr<GpuEvaluationKeyData>> compacted_by_q_count_;

public:
    GpuEvaluationKeyData() = default;

    bool empty() const;

    bool has_compacted_key(std::size_t q_count) const;

    void store_compacted_key(
        std::size_t q_count,
        GpuEvaluationKeyData compacted_key);

    const GpuEvaluationKeyData &key_for_q_count(std::size_t q_count) const;

    /**
     * @brief Create mutable view of evaluation key data.
     */
    GpuEvaluationKeyView make_view();

    /**
     * @brief Create const view of evaluation key data.
     */
    GpuConstEvaluationKeyView make_const_view() const;
};

using GpuRelinKeysData = GpuEvaluationKeyData;
using GpuGaloisKeysData = GpuEvaluationKeyData;

}  // namespace gpu
}  // namespace poseidon
