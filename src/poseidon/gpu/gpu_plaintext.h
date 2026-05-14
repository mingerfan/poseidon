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
 * @brief Semantic metadata of a GPU plaintext.
 */
struct GpuPlaintextMeta
{
    parms_id_type parms_id{};

    double scale = 1.0;
    bool is_ntt_form = false;

    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;
};

struct GpuPlaintextView
{
    GpuPlaintextMeta meta;
    GpuRNSPolyView poly;
};

struct GpuConstPlaintextView
{
    GpuPlaintextMeta meta;
    GpuConstRNSPolyView poly;
};

/**
 * @brief GPU-side plaintext data.
 *
 * Plaintext normally contains one logical RNS polynomial.
 *
 * It still uses the same field/shard/poly model as ciphertext:
 * - fields_ owns GPU memory;
 * - poly_ describes the logical plaintext RNS polynomial;
 * - poly_.shards can describe single-GPU or multi-GPU placement.
 */
class GpuPlaintextData
{
public:
    GpuPlaintextMeta meta;

    /**
     * @brief Real GPU memory blocks.
     *
     * This can be one field on one GPU, or multiple fields across GPUs.
     */
    std::vector<GpuFieldData> fields_;

    /**
     * @brief Logical plaintext RNS polynomial.
     *
     * The plaintext polynomial can also be sharded across limb/coeff dimensions.
     */
    GpuRNSPoly poly_;

public:
    GpuPlaintextData() = default;

    bool empty() const;

    /**
     * @brief Create mutable temporary plaintext view.
     *
     * TODO:
     * - Translate shard.field_index into device pointer.
     * - Validate shard ranges.
     */
    GpuPlaintextView make_view();

    /**
     * @brief Create const temporary plaintext view.
     *
     * TODO:
     * - Translate shard.field_index into const device pointer.
     * - Validate shard ranges.
     */
    GpuConstPlaintextView make_const_view() const;

    /**
     * @brief Allocate single-device plaintext storage.
     *
     * TODO:
     * - Allocate one field covering all q limbs and all coefficients.
     * - Build default full-range shard layout.
     */
    static GpuPlaintextData allocate_single_device(
        std::size_t degree,
        std::size_t q_count,
        int device_id);
};

}  // namespace gpu
}  // namespace poseidon