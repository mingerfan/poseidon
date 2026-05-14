#include "poseidon/gpu/gpu_ciphertext.h"

#include <stdexcept>

namespace poseidon
{
namespace gpu
{

std::size_t GpuCiphertextData::size() const
{
    return polys_.size();
}

bool GpuCiphertextData::empty() const
{
    return polys_.empty();
}

GpuCiphertextView GpuCiphertextData::make_view()
{
    // TODO:
    // Build a temporary mutable view from fields_ and polys_.
    // This should not allocate GPU memory.
    // It should only translate field_index into GpuWord* pointers.

    throw std::runtime_error("GpuCiphertextData::make_view is not implemented yet");
}

GpuConstCiphertextView GpuCiphertextData::make_const_view() const
{
    // TODO:
    // Build a temporary const view from fields_ and polys_.
    // This should not allocate GPU memory.
    // It should only translate field_index into const GpuWord* pointers.

    throw std::runtime_error("GpuCiphertextData::make_const_view is not implemented yet");
}

GpuCiphertextData GpuCiphertextData::allocate_single_device(
    std::size_t degree,
    std::size_t q_count,
    std::size_t component_count,
    int device_id)
{
    // TODO:
    // Allocate one GpuFieldData per component and build one full-range shard
    // per GpuRNSPoly.
    //
    // Expected result:
    // - fields_[i] stores component ci;
    // - polys_[i] describes component ci;
    // - each polys_[i] has one shard covering all q limbs and all coefficients.

    (void)degree;
    (void)q_count;
    (void)component_count;
    (void)device_id;

    throw std::runtime_error("GpuCiphertextData::allocate_single_device is not implemented yet");
}

}  // namespace gpu
}  // namespace poseidon