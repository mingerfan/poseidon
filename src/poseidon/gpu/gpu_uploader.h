#pragma once

#include "poseidon/ciphertext.h"
#include "poseidon/gpu/gpu_ciphertext.h"

namespace poseidon
{
namespace gpu
{

/**
 * @brief CPU/GPU conversion helper.
 *
 * Current stage:
 * - Only declares the API.
 * - Actual cudaMemcpy logic should be filled later.
 */
class GpuUploader
{
public:
    /**
     * @brief Upload a Poseidon CPU Ciphertext to GPU.
     *
     * Expected behavior:
     * - Read size/degree/q_count/parms_id/scale/is_ntt_form from src.
     * - Allocate GpuCiphertextData.
     * - Copy each component c0/c1/c2 from CPU to GPU fields.
     */
    static GpuCiphertextData upload_ciphertext(
        const Ciphertext &src,
        int device_id);

    /**
     * @brief Download a GPU ciphertext back to Poseidon CPU Ciphertext.
     *
     * Expected behavior:
     * - Resize dst according to src metadata.
     * - Copy GPU fields back to dst.data(component_id).
     * - Restore parms_id/scale/is_ntt_form.
     */
    static void download_ciphertext(
        const GpuCiphertextData &src,
        Ciphertext &dst);
};

}  // namespace gpu
}  // namespace poseidon