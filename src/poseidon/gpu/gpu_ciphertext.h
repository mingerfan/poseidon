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

/**
 * @brief Semantic metadata of a GPU ciphertext.
 *
 * This describes the FHE state of the ciphertext.
 * fields_/polys_ describe the physical GPU storage.
 */
struct GpuCiphertextMeta
{
    parms_id_type parms_id{};

    double scale = 1.0;
    std::uint64_t correction_factor = 1;
    bool is_ntt_form = false;

    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;

    std::size_t component_count = 0;
};

struct GpuCiphertextView
{
    GpuCiphertextMeta meta;
    std::vector<GpuRNSPolyView> polys;
};

struct GpuConstCiphertextView
{
    GpuCiphertextMeta meta;
    std::vector<GpuConstRNSPolyView> polys;
};

/**
 * @brief GPU-side ciphertext data.
 *
 * This is the GPU counterpart of Poseidon Ciphertext.
 *
 * Responsibility:
 * - own GPU memory through fields_;
 * - describe c0/c1/c2 through polys_;
 * - expose temporary views for GPU handlers/kernels.
 */
class GpuCiphertextData
{
public:
    GpuCiphertextMeta meta;

    /**
     * @brief Real GPU memory blocks.
     *
     * A ciphertext may contain multiple fields:
     * - each field represents one selected GPU's storage for this object;
     * - shards describe packed slices inside those fields.
     */
    std::vector<GpuFieldData> fields_;

    /**
     * @brief Logical ciphertext components.
     *
     * polys_[0] represents c0;
     * polys_[1] represents c1;
     * polys_[2] represents c2, if present.
     */
    std::vector<GpuRNSPoly> polys_;

public:
    GpuCiphertextData() = default;

    std::size_t size() const;
    bool empty() const;

    /**
     * @brief Create a mutable temporary view.
     */
    GpuCiphertextView make_view();

    /**
     * @brief Create a const temporary view.
     */
    GpuConstCiphertextView make_const_view() const;

    /**
     * @brief Allocate single-device ciphertext storage.
     *
     * This only allocates GPU storage and builds default layout.
     * It does not copy CPU data.
     *
     * Default single-device layout:
     * - one field for the selected GPU;
     * - each ciphertext component is represented by one packed shard;
     * - component shards are stored consecutively in the field.
     *
     * Device allocation is delegated to DeviceVector.
     */
    static GpuCiphertextData allocate_single_device(
        std::size_t degree,
        std::size_t q_count,
        std::size_t component_count,
        int device_id,
        std::size_t p_count = 0);

    /**
     * @brief Allocate single-device ciphertext storage with a shard template.
     *
     * The shard template describes one logical ciphertext component. The
     * allocator applies it to every component and assigns field_index and
     * field_offset automatically. Template field_index/field_offset values are
     * ignored.
     *
     * Each shard is stored as one contiguous packed region:
     * - limb-major inside the shard;
     * - local offset = local_limb * coeff_count + local_coeff.
     */
    static GpuCiphertextData allocate_single_device_sharded(
        std::size_t degree,
        std::size_t q_count,
        std::size_t component_count,
        int device_id,
        const std::vector<GpuPolyShard> &shard_template,
        std::size_t p_count = 0);
};

}  // namespace gpu
}  // namespace poseidon
