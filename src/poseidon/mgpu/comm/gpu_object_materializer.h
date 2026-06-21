#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/mgpu/comm/materialized_gpu_comm.h"

namespace poseidon::mgpu
{

class PoseidonGpuObjectCopyMaterializer final : public GpuObjectCopyMaterializer
{
public:
    MaterializedGpuObjectCopy materialize_copy(
        const GpuCommCopyRequest &request) override;
};

GpuObjectCopyRequest make_full_object_copy_request(
    ValueId source_id, ValueId destination_id, const gpu::GpuCiphertextData &source,
    gpu::GpuCiphertextData &destination, int destination_device);

GpuObjectCopyRequest make_full_object_copy_request(
    ValueId source_id, ValueId destination_id, const gpu::GpuPlaintextData &source,
    gpu::GpuPlaintextData &destination, int destination_device);

}  // namespace poseidon::mgpu
