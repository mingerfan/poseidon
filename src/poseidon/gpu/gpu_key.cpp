#include "poseidon/gpu/gpu_key.h"

#include <stdexcept>

namespace poseidon
{
namespace gpu
{

bool GpuEvaluationKeyData::empty() const
{
    return key_polys_.empty();
}

GpuEvaluationKeyView GpuEvaluationKeyData::make_view()
{
    // TODO:
    // Build mutable evaluation-key view from fields_ and key_polys_.

    throw std::runtime_error("GpuEvaluationKeyData::make_view is not implemented yet");
}

GpuConstEvaluationKeyView GpuEvaluationKeyData::make_const_view() const
{
    // TODO:
    // Build const evaluation-key view from fields_ and key_polys_.

    throw std::runtime_error("GpuEvaluationKeyData::make_const_view is not implemented yet");
}

}  // namespace gpu
}  // namespace poseidon