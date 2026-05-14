#include "poseidon/gpu/gpu_ntt_handler.h"

#include <stdexcept>

namespace poseidon
{
namespace gpu
{

GpuNTTHandler::GpuNTTHandler(const GpuParameterData &params)
    : params_(params)
{}

void GpuNTTHandler::forward_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // Forward NTT for every Poseidon ciphertext component polynomial.

    (void)destination_view;
    (void)source_view;
    (void)level_info;

    throw std::runtime_error("GpuNTTHandler::forward_ciphertext is not implemented yet");
}

void GpuNTTHandler::inverse_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // Inverse NTT for every Poseidon ciphertext component polynomial.

    (void)destination_view;
    (void)source_view;
    (void)level_info;

    throw std::runtime_error("GpuNTTHandler::inverse_ciphertext is not implemented yet");
}

void GpuNTTHandler::forward_plaintext(
    GpuPlaintextView &destination_view,
    const GpuConstPlaintextView &source_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // Forward NTT for the plaintext polynomial.

    (void)destination_view;
    (void)source_view;
    (void)level_info;

    throw std::runtime_error("GpuNTTHandler::forward_plaintext is not implemented yet");
}

void GpuNTTHandler::inverse_plaintext(
    GpuPlaintextView &destination_view,
    const GpuConstPlaintextView &source_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // Inverse NTT for the plaintext polynomial.

    (void)destination_view;
    (void)source_view;
    (void)level_info;

    throw std::runtime_error("GpuNTTHandler::inverse_plaintext is not implemented yet");
}

void GpuNTTHandler::forward_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &source_poly,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // Validate shard placement and launch forward NTT kernel.

    (void)destination_poly;
    (void)source_poly;
    (void)level_info;

    throw std::runtime_error("GpuNTTHandler::forward_poly is not implemented yet");
}

void GpuNTTHandler::inverse_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &source_poly,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // Validate shard placement and launch inverse NTT kernel.

    (void)destination_poly;
    (void)source_poly;
    (void)level_info;

    throw std::runtime_error("GpuNTTHandler::inverse_poly is not implemented yet");
}

}  // namespace gpu
}  // namespace poseidon