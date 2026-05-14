#include "poseidon/gpu/gpu_plaintext.h"

#include <stdexcept>

namespace poseidon
{
namespace gpu
{

bool GpuPlaintextData::empty() const
{
    return fields_.empty();
}

GpuPlaintextView GpuPlaintextData::make_view()
{
    // TODO:
    // Build mutable plaintext view from fields_ and poly_.

    throw std::runtime_error("GpuPlaintextData::make_view is not implemented yet");
}

GpuConstPlaintextView GpuPlaintextData::make_const_view() const
{
    // TODO:
    // Build const plaintext view from fields_ and poly_.

    throw std::runtime_error("GpuPlaintextData::make_const_view is not implemented yet");
}

GpuPlaintextData GpuPlaintextData::allocate_single_device(
    std::size_t degree,
    std::size_t q_count,
    int device_id)
{
    // TODO:
    // Allocate one field for plaintext polynomial and build default full-range shard.

    (void)degree;
    (void)q_count;
    (void)device_id;

    throw std::runtime_error("GpuPlaintextData::allocate_single_device is not implemented yet");
}

}  // namespace gpu
}  // namespace poseidon