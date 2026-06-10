#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_parameter.h"

namespace poseidon
{
namespace gpu
{

/**
 * @brief Handler for GPU modulus switching and rescale operations.
 *
 * This class corresponds to Cheddar's mod-switch handling layer.
 *
 * It is responsible for physical GPU execution of:
 * - rescale;
 * - dynamic rescale;
 * - drop modulus;
 * - future raise modulus / modup / moddown.
 *
 * GpuEvaluator decides semantic metadata changes.
 * This handler prepares GPU launch tasks and launches kernels later.
 */
class GpuModSwitchHandler
{
public:
    explicit GpuModSwitchHandler(const GpuParameterData &params);
    ~GpuModSwitchHandler();

    /**
     * @brief GPU rescale from source level to destination level.
     */
    void rescale_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &source_view,
        const GpuLevelInfo &source_level_info,
        const GpuLevelInfo &destination_level_info) const;

    /**
     * @brief GPU dynamic rescale.
     *
     * This should support the small-prime physical chain and logical multi-prime
     * dropping policy.
     */
    void rescale_dynamic_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &source_view,
        const GpuLevelInfo &source_level_info,
        const GpuLevelInfo &destination_level_info,
        double min_scale) const;

    /**
     * @brief Drop modulus from source level to destination level.
     */
    void drop_modulus_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &source_view,
        const GpuLevelInfo &source_level_info,
        const GpuLevelInfo &destination_level_info) const;

private:
    struct RescaleScratch
    {
        DeviceVector<GpuWord> q_last;
        DeviceVector<GpuWord> correction;
        std::size_t q_last_capacity = 0;
        std::size_t correction_capacity = 0;
        int device_id = -1;
    };

    void ensure_rescale_scratch(
        std::size_t degree,
        std::size_t destination_q_count,
        int device_id) const;

    const GpuParameterData &params_;
    mutable RescaleScratch rescale_scratch_;
};

}  // namespace gpu
}  // namespace poseidon
