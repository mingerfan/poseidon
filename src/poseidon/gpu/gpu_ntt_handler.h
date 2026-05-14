#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_parameter.h"

namespace poseidon
{
namespace gpu
{

/**
 * @brief Handler for GPU NTT and inverse NTT.
 *
 * This class corresponds to Cheddar's NTT handler layer, but it operates on
 * Poseidon's component-vector model.
 *
 * For ciphertext:
 * - every ciphertext component polynomial should be transformed.
 *
 * For plaintext:
 * - the single plaintext polynomial should be transformed.
 */
class GpuNTTHandler
{
public:
    explicit GpuNTTHandler(const GpuParameterData &params);

    /**
     * @brief Forward NTT for every component in a ciphertext.
     */
    void forward_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &source_view,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief Inverse NTT for every component in a ciphertext.
     */
    void inverse_ciphertext(
        GpuCiphertextView &destination_view,
        const GpuConstCiphertextView &source_view,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief Forward NTT for plaintext polynomial.
     */
    void forward_plaintext(
        GpuPlaintextView &destination_view,
        const GpuConstPlaintextView &source_view,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief Inverse NTT for plaintext polynomial.
     */
    void inverse_plaintext(
        GpuPlaintextView &destination_view,
        const GpuConstPlaintextView &source_view,
        const GpuLevelInfo &level_info) const;

private:
    /**
     * @brief Forward NTT for one logical RNS polynomial.
     */
    void forward_poly(
        GpuRNSPolyView &destination_poly,
        const GpuConstRNSPolyView &source_poly,
        const GpuLevelInfo &level_info) const;

    /**
     * @brief Inverse NTT for one logical RNS polynomial.
     */
    void inverse_poly(
        GpuRNSPolyView &destination_poly,
        const GpuConstRNSPolyView &source_poly,
        const GpuLevelInfo &level_info) const;

private:
    const GpuParameterData &params_;
};

}  // namespace gpu
}  // namespace poseidon