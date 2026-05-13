#pragma once

#include "poseidon/parameters_literal.h"
#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_memory.h"

#include <cstddef>
#include <vector>

namespace poseidon
{
namespace gpu
{

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
    std::vector<GpuRNSPolyView> polys;
};

struct GpuConstPlaintextView
{
    GpuPlaintextMeta meta;
    std::vector<GpuConstRNSPolyView> polys;
};

/**
 * @brief GPU-side plaintext data.
 *
 * Plaintext normally contains one logical RNS polynomial.
 * It still reuses the same field/shard/view model as ciphertext so that
 * add_plain and multiply_plain can share GPU handlers.
 */
class GpuPlaintextData
{
public:
    GpuPlaintextMeta meta;

    std::vector<GpuFieldData> fields_;
    GpuRNSPoly gpupoly_;

public:
    GpuPlaintextData() = default;

    bool empty() const;

    /**
     * @brief Create mutable temporary plaintext view.
     *
     * TODO:
     * - Translate shard.field_index into device pointer.
     */
    GpuPlaintextView make_view();

    /**
     * @brief Create const temporary plaintext view.
     *
     * TODO:
     * - Translate shard.field_index into const device pointer.
     */
    GpuConstPlaintextView make_const_view() const;

    /**
     * @brief Allocate single-device plaintext storage.
     *
     * TODO:
     * - Allocate one field covering all q limbs and all coefficients.
     * - Build default shard layout.
     */
    static GpuPlaintextData allocate_single_device(
        std::size_t degree,
        std::size_t q_count,
        int device_id);
};

}  // namespace gpu
}  // namespace poseidon