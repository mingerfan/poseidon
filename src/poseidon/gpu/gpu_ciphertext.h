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
 * @brief Semantic metadata of a GPU ciphertext.
 *
 * This describes the FHE state of the ciphertext.
 * fields_/polys_ describe the physical GPU storage.
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
     * - one field per component on one GPU;
     * - or multiple fields per component across multiple GPUs.
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
     * - one field per ciphertext component;
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