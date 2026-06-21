#pragma once

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

class CudaPeerComm
{
public:
    static int visible_device_count();
    static bool can_access_peer(int destination_device, int source_device);

    void copy_buffer(const CudaPeerCopyRequest &request) const;
};

}  // namespace poseidon::mgpu
