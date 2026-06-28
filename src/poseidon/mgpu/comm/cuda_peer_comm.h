#pragma once

#include "poseidon/mgpu/comm/gpu_comm.h"

#include <cstddef>
#include <map>
#include <mutex>
#include <utility>

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

struct CudaPeerAccessSnapshot
{
    bool cached = false;
    bool peer_access_supported = false;
    bool peer_access_enabled = false;
    std::size_t enable_call_count = 0;
};

class CudaPeerComm final : public GpuObjectCopyBackend
{
public:
    static int visible_device_count();
    static bool can_access_peer(int destination_device, int source_device);

    CudaPeerAccessSnapshot peer_access_snapshot(
        int destination_device, int source_device) const;

    void copy_buffer(const CudaPeerCopyRequest &request) const;
    void copy_object(const GpuObjectCopyRequest &request) override;

private:
    struct PeerAccessCacheEntry
    {
        bool support_checked = false;
        bool supported = false;
        bool enabled = false;
        std::size_t enable_call_count = 0;
    };

    using PeerAccessKey = std::pair<int, int>;

    bool ensure_peer_access(int destination_device, int source_device) const;

    mutable std::mutex peer_access_mutex_;
    mutable std::map<PeerAccessKey, PeerAccessCacheEntry> peer_access_cache_;
};

}  // namespace poseidon::mgpu
