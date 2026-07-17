#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/runtime_api/communication/device_buffer.h"

#include <vector>

namespace poseidon::runtime_api::communication
{

std::vector<DeviceBufferCopy> prepare_full_object_copy(
    const gpu::GpuCiphertextData &source, gpu::GpuCiphertextData &destination,
    int destination_device);

std::vector<DeviceBufferCopy> prepare_full_object_copy(
    const gpu::GpuPlaintextData &source, gpu::GpuPlaintextData &destination,
    int destination_device);

} // namespace poseidon::runtime_api::communication
