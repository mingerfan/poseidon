#pragma once

#include "poseidon/runtime_api/communication/device_buffer.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
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

class PinnedHostBuffer
{
public:
    explicit PinnedHostBuffer(std::size_t bytes);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer &) = delete;
    PinnedHostBuffer &operator=(const PinnedHostBuffer &) = delete;

    void *data() noexcept;
    const void *data() const noexcept;
    std::size_t size() const noexcept;

private:
    void *data_ = nullptr;
    std::size_t bytes_ = 0;
};

class CudaTransferRequest
{
public:
    CudaTransferRequest();
    ~CudaTransferRequest();

    CudaTransferRequest(const CudaTransferRequest &) = delete;
    CudaTransferRequest &operator=(const CudaTransferRequest &) = delete;
    CudaTransferRequest(CudaTransferRequest &&) noexcept;
    CudaTransferRequest &operator=(CudaTransferRequest &&) noexcept;

    void wait();

private:
    struct State;
    std::unique_ptr<State> state_;

    friend class CudaLocalTransfer;
};

class CudaLocalTransfer
{
public:
    static int visible_device_count();
    static bool can_access_peer(int destination_device, int source_device);
    // Required once per directed device pair before submitting a PeerToPeer copy.
    static void enable_peer_access(int destination_device, int source_device);
    static CudaTransferRoute select_route(int destination_device, int source_device,
                                          CudaTransferRoute requested);

    CudaTransferRoute copy_sync(
        const DeviceBufferCopy &request,
        CudaTransferRoute requested = CudaTransferRoute::Auto) const;
    void copy_sync(
        const std::vector<DeviceBufferCopy> &requests,
        CudaTransferRoute requested = CudaTransferRoute::Auto) const;

    CudaTransferRequest copy_async(
        const DeviceBufferCopy &request,
        CudaTransferRoute requested = CudaTransferRoute::Auto,
        cudaEvent_t source_ready = nullptr) const;
    CudaTransferRequest copy_host_to_device_async(
        const std::shared_ptr<PinnedHostBuffer> &source, void *destination,
        std::size_t bytes, int destination_device) const;
    CudaTransferRequest copy_device_to_host_async(
        const void *source, int source_device,
        const std::shared_ptr<PinnedHostBuffer> &destination,
        std::size_t bytes, cudaEvent_t source_ready = nullptr) const;
};

} // namespace poseidon::runtime_api::communication
