#pragma once

#include "poseidon/runtime_api/communication/device_buffer.h"

#include <vector>

namespace poseidon::runtime_api::communication
{

enum class CudaTransferRoute
{
    Auto,
    SameDevice,
    PeerToPeer,
    HostStaged,
};

class CudaLocalTransfer
{
public:
    static int visible_device_count();
    static bool can_access_peer(int destination_device, int source_device);
    static CudaTransferRoute select_route(int destination_device, int source_device,
                                          CudaTransferRoute requested);

    CudaTransferRoute copy_sync(
        const DeviceBufferCopy &request,
        CudaTransferRoute requested = CudaTransferRoute::Auto) const;
    void copy_sync(
        const std::vector<DeviceBufferCopy> &requests,
        CudaTransferRoute requested = CudaTransferRoute::Auto) const;
};

} // namespace poseidon::runtime_api::communication
