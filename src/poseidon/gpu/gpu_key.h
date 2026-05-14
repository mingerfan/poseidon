#pragma once

#include "poseidon/parameters_literal.h"
#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_rns_poly.h"

#include <cstddef>
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

public:
    GpuEvaluationKeyData() = default;

    bool empty() const;

    /**
     * @brief Create mutable view of evaluation key data.
     *
     * TODO:
     * - Translate field_index into device pointers.
     */
    GpuEvaluationKeyView make_view();

    /**
     * @brief Create const view of evaluation key data.
     *
     * TODO:
     * - Translate field_index into const device pointers.
     */
    GpuConstEvaluationKeyView make_const_view() const;
};

using GpuRelinKeysData = GpuEvaluationKeyData;
using GpuGaloisKeysData = GpuEvaluationKeyData;

}  // namespace gpu
}  // namespace poseidon