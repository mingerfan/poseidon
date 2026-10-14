#pragma once

#include <cstddef>

namespace poseidon::runtime_api::communication
{

struct DeviceBufferCopy
{
    const void *source = nullptr;
    void *destination = nullptr;
    std::size_t bytes = 0;
    int source_device = 0;
    int destination_device = 0;
};

} // namespace poseidon::runtime_api::communication
