#include "poseidon/gpu/gpu_modswitch_handler.h"

#include <stdexcept>

namespace poseidon
{
namespace gpu
{

GpuModSwitchHandler::GpuModSwitchHandler(const GpuParameterData &params)
    : params_(params)
{}

void GpuModSwitchHandler::rescale_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info) const
{
    // TODO:
    // Launch GPU rescale kernels.
    //
    // Must support:
    // - 32-bit physical primes;
    // - logical multi-prime dropping;
    // - destination level metadata prepared by GpuEvaluator.

    (void)destination_view;
    (void)source_view;
    (void)source_level_info;
    (void)destination_level_info;

    throw std::runtime_error("GpuModSwitchHandler::rescale_ciphertext is not implemented yet");
}

void GpuModSwitchHandler::rescale_dynamic_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info,
    double min_scale) const
{
    // TODO:
    // Launch GPU dynamic rescale kernels.

    (void)destination_view;
    (void)source_view;
    (void)source_level_info;
    (void)destination_level_info;
    (void)min_scale;

    throw std::runtime_error("GpuModSwitchHandler::rescale_dynamic_ciphertext is not implemented yet");
}

void GpuModSwitchHandler::drop_modulus_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info) const
{
    // TODO:
    // Launch GPU drop-modulus logic.
    //
    // Depending on physical layout, this may be:
    // - metadata/view update only;
    // - or actual data compaction/copy.

    (void)destination_view;
    (void)source_view;
    (void)source_level_info;
    (void)destination_level_info;

    throw std::runtime_error("GpuModSwitchHandler::drop_modulus_ciphertext is not implemented yet");
}

}  // namespace gpu
}  // namespace poseidon