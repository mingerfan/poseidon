#include "poseidon/gpu/gpu_parameter.h"

#include <stdexcept>

namespace poseidon
{
namespace gpu
{

GpuParameterData::GpuParameterData(const PoseidonContext &context, int device_id)
{
    build_from_poseidon_context(context, device_id);
}

void GpuParameterData::build_from_poseidon_context(
    const PoseidonContext &context,
    int device_id)
{
    // TODO:
    // This function should convert Poseidon CPU context information into
    // GPU-resident parameter tables.
    //
    // Expected future behavior:
    // 1. Traverse all CrtContext::ContextData levels.
    // 2. Copy q/p primes to device buffers as GpuWord.
    // 3. Build/copy modular reduction constants.
    // 4. Build/copy NTT/INTT tables.
    // 5. Build/copy rescale/modswitch/key-switch related constants.
    //
    // This function should make GpuEvaluator independent of PoseidonContext
    // during GPU operator execution.

    (void)context;
    (void)device_id;

    throw std::runtime_error("GpuParameterData::build_from_poseidon_context is not implemented yet");
}

const GpuLevelInfo &GpuParameterData::get_level(const parms_id_type &parms_id) const
{
    // TODO:
    // Query GpuLevelInfo by parms_id.

    (void)parms_id;

    throw std::runtime_error("GpuParameterData::get_level is not implemented yet");
}

bool GpuParameterData::empty() const
{
    return levels_.empty();
}

}  // namespace gpu
}  // namespace poseidon