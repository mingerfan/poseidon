#pragma once

#include "poseidon/parameters_literal.h"
#include "poseidon/gpu/gpu_memory.h"

#include <cstddef>
#include <vector>

namespace poseidon
{
namespace gpu
{

/**
 * @brief Semantic metadata of a GPU ciphertext.
 *
 * This describes the FHE state of the ciphertext, while fields_/gpupolys_
 * describe physical GPU storage.
 */
struct GpuCiphertextMeta
{
    parms_id_type parms_id{};

    double scale = 1.0;
    bool is_ntt_form = false;

    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;

    std::size_t component_count = 0;
};

/**
 * @brief Physical slice of one logical RNS polynomial component.
 *
 * A shard says:
 * - which GPU memory block stores this slice;
 * - which RNS limbs it covers;
 * - which coefficient range it covers.
 */
struct GpuPolyShard
{
    std::size_t field_index = 0;
    std::size_t field_offset = 0;

    std::size_t limb_begin = 0;
    std::size_t limb_count = 0;

    std::size_t coeff_begin = 0;
    std::size_t coeff_count = 0;
};

/**
 * @brief Logical GPU-side RNS polynomial.
 *
 * For ciphertext:
 * - component_id = 0 means c0;
 * - component_id = 1 means c1;
 * - component_id = 2 means c2.
 *
 * This structure does not own GPU memory. It only records which shards belong
 * to this component.
 */
struct GpuRNSPoly
{
    std::size_t component_id = 0;

    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;

    std::vector<GpuPolyShard> shards;
};

/**
 * @brief Mutable view of one physical shard.
 *
 * View objects are temporary and non-owning. They are created before launching
 * GPU handlers/kernels.
 */
struct GpuPolyShardView
{
    int device_id = 0;
    GpuWord *ptr = nullptr;

    std::size_t limb_begin = 0;
    std::size_t limb_count = 0;

    std::size_t coeff_begin = 0;
    std::size_t coeff_count = 0;
};

/**
 * @brief Const view of one physical shard.
 */
struct GpuConstPolyShardView
{
    int device_id = 0;
    const GpuWord *ptr = nullptr;

    std::size_t limb_begin = 0;
    std::size_t limb_count = 0;

    std::size_t coeff_begin = 0;
    std::size_t coeff_count = 0;
};

struct GpuRNSPolyView
{
    std::size_t component_id = 0;
    std::vector<GpuPolyShardView> shards;
};

struct GpuConstRNSPolyView
{
    std::size_t component_id = 0;
    std::vector<GpuConstPolyShardView> shards;
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
 * Long-term responsibility:
 * - Own GPU memory through fields_;
 * - Describe c0/c1/c2 through gpupolys_;
 * - Expose temporary views for GPU handlers/kernels.
 */
class GpuCiphertextData
{
public:
    GpuCiphertextMeta meta;

    /**
     * @brief Real GPU memory blocks.
     *
     * A single ciphertext may contain multiple fields:
     * - one field per component on one GPU;
     * - or multiple fields per component across multiple GPUs.
     */
    std::vector<GpuFieldData> fields_;

    /**
     * @brief Logical ciphertext components.
     *
     * gpupolys_[0] represents c0;
     * gpupolys_[1] represents c1;
     * gpupolys_[2] represents c2, if present.
     */
    std::vector<GpuRNSPoly> gpupolys_;

public:
    GpuCiphertextData() = default;

    std::size_t size() const;
    bool empty() const;

    /**
     * @brief Create a mutable temporary view.
     *
     * TODO:
     * - Translate shard.field_index into actual device pointers.
     * - Validate shard ranges.
     */
    GpuCiphertextView make_view();

    /**
     * @brief Create a const temporary view.
     *
     * TODO:
     * - Translate shard.field_index into actual const device pointers.
     * - Validate shard ranges.
     */
    GpuConstCiphertextView make_const_view() const;

    /**
     * @brief Allocate single-device ciphertext storage.
     *
     * This only allocates GPU storage and builds default layout.
     * It does not copy CPU data.
     *
     * First-stage layout:
     * - one field per component;
     * - each field stores [q0 | q1 | ... | q_{q_count-1}];
     * - each q block stores degree coefficients.
     *
     * TODO:
     * - Real GPU allocation depends on DeviceVector implementation.
     */
    static GpuCiphertextData allocate_single_device(
        std::size_t degree,
        std::size_t q_count,
        std::size_t component_count,
        int device_id);
};

}  // namespace gpu
}  // namespace poseidon