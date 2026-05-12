#pragma once

#include "poseidon/parameters_literal.h"
#include "poseidon/gpu/gpu_memory.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace poseidon
{
namespace gpu
{

/**
 * @brief A physical slice of one GPU RNS polynomial component.
 *
 * A GpuRNSPoly represents a logical component, e.g., c0/c1/c2.
 * A GpuPolyShard describes where one physical slice of this component is stored.
 *
 * Example:
 * - field_index = 0
 * - limb_begin  = 0
 * - limb_count  = 4
 * means this shard stores q0-q3 of this component.
 */
struct GpuPolyShard
{
    // Index into GpuCiphertextData::fields_.
    std::size_t field_index = 0;

    // Offset inside fields_[field_index].buffer.
    // First version usually uses 0.
    std::size_t field_offset = 0;

    // RNS limb range.
    std::size_t limb_begin = 0;
    std::size_t limb_count = 0;

    // Coefficient range.
    // First version usually covers [0, degree).
    std::size_t coeff_begin = 0;
    std::size_t coeff_count = 0;
};

/**
 * @brief Logical GPU-side RNS polynomial.
 *
 * This corresponds to one ciphertext component:
 * - component_id = 0 -> c0
 * - component_id = 1 -> c1
 * - component_id = 2 -> c2
 *
 * It does not own GPU memory. It only describes which field buffers contain
 * its physical data.
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
 * @brief Host-side metadata for a GPU ciphertext.
 *
 * These fields are used for evaluator checks and result metadata update.
 * They should not be treated as device-side authoritative data.
 */
struct GpuCiphertextMeta
{
    parms_id_type parms_id = parms_id_zero;

    double scale = 1.0;
    bool is_ntt_form = false;

    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;

    std::size_t component_count = 0;
};

/**
 * @brief View of one physical shard.
 *
 * This is a temporary non-owning object used by kernel launchers.
 */
struct GpuPolyShardView
{
    int device_id = 0;
    std::uint64_t *ptr = nullptr;

    std::size_t limb_begin = 0;
    std::size_t limb_count = 0;

    std::size_t coeff_begin = 0;
    std::size_t coeff_count = 0;
};

/**
 * @brief View of one logical RNS polynomial component.
 */
struct GpuRNSPolyView
{
    std::size_t component_id = 0;
    std::vector<GpuPolyShardView> shards;
};

/**
 * @brief Temporary view of a GPU ciphertext.
 *
 * This object does not own memory. It should be created right before
 * calling a GPU handler/kernel and then discarded.
 */
struct GpuCiphertextView
{
    GpuCiphertextMeta meta;
    std::vector<GpuRNSPolyView> polys;
};

/**
 * @brief GPU-side ciphertext data.
 *
 * This class is the GPU counterpart of Poseidon Ciphertext.
 *
 * It owns GPU memory through fields_.
 * It describes logical ciphertext components through gpupolys_.
 *
 * Rough correspondence:
 * Poseidon Ciphertext::data_   -> fields_
 * Poseidon Ciphertext::polys_  -> gpupolys_
 */
class GpuCiphertextData
{
public:
    GpuCiphertextMeta meta;

    // Real GPU memory owners.
    std::vector<GpuFieldData> fields_;

    // Logical components c0/c1/c2.
    std::vector<GpuRNSPoly> gpupolys_;

public:
    GpuCiphertextData() = default;

    std::size_t size() const
    {
        return gpupolys_.size();
    }

    bool empty() const
    {
        return gpupolys_.empty();
    }

    /**
     * @brief Create a non-owning view for GPU handlers/kernels.
     */
    GpuCiphertextView make_view()
    {
        GpuCiphertextView view;
        view.meta = meta;

        for (const auto &poly : gpupolys_)
        {
            GpuRNSPolyView poly_view;
            poly_view.component_id = poly.component_id;

            for (const auto &shard : poly.shards)
            {
                if (shard.field_index >= fields_.size())
                {
                    throw std::runtime_error("Invalid field_index in GpuPolyShard");
                }

                auto &field = fields_[shard.field_index];

                GpuPolyShardView shard_view;
                shard_view.device_id = field.device_id;
                shard_view.ptr = field.data() + shard.field_offset;

                shard_view.limb_begin = shard.limb_begin;
                shard_view.limb_count = shard.limb_count;
                shard_view.coeff_begin = shard.coeff_begin;
                shard_view.coeff_count = shard.coeff_count;

                poly_view.shards.push_back(shard_view);
            }

            view.polys.push_back(poly_view);
        }

        return view;
    }

    /**
     * @brief Allocate a simple single-GPU ciphertext.
     *
     * First-stage layout:
     * - one field per component
     * - each field stores [q0 block | q1 block | ... | q_{q_count-1} block]
     * - each q block contains degree coefficients
     */
    static GpuCiphertextData create_single_device(
        std::size_t degree,
        std::size_t q_count,
        std::size_t component_count,
        int device_id)
    {
        GpuCiphertextData ct;

        ct.meta.degree = degree;
        ct.meta.q_count = q_count;
        ct.meta.p_count = 0;
        ct.meta.component_count = component_count;
        ct.meta.is_ntt_form = false;

        const std::size_t elems_per_component = degree * q_count;

        for (std::size_t comp_id = 0; comp_id < component_count; ++comp_id)
        {
            const std::size_t field_index = ct.fields_.size();

            ct.fields_.emplace_back(device_id, elems_per_component);

            GpuRNSPoly poly;
            poly.component_id = comp_id;
            poly.degree = degree;
            poly.q_count = q_count;
            poly.p_count = 0;

            GpuPolyShard shard;
            shard.field_index = field_index;
            shard.field_offset = 0;

            shard.limb_begin = 0;
            shard.limb_count = q_count;

            shard.coeff_begin = 0;
            shard.coeff_count = degree;

            poly.shards.push_back(shard);
            ct.gpupolys_.push_back(std::move(poly));
        }

        return ct;
    }
};

}  // namespace gpu
}  // namespace poseidon