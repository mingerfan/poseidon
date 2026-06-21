#pragma once

#include "poseidon/mgpu/comm/gpu_object_copy.h"
#include "poseidon/mgpu/comm/materialized_gpu_comm.h"

#include <cstddef>

namespace poseidon::mgpu
{

struct CudaPeerCopyRequest
{
    const void *source = nullptr;
    void *destination = nullptr;
    std::size_t bytes = 0;
    int source_device = 0;
    int destination_device = 0;
};

class CudaPeerComm final : public GpuObjectCopyBackend
{
public:
    static int visible_device_count();
    static bool can_access_peer(int destination_device, int source_device);

    void copy_buffer(const CudaPeerCopyRequest &request) const;
    void copy_object(const GpuObjectCopyRequest &request) override;
};

}  // namespace poseidon::mgpu
