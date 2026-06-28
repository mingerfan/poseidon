#pragma once

#include "poseidon/mgpu/comm/gpu_comm.h"

#include <cstddef>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

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

struct CudaPeerPackScratchSnapshot
{
    std::size_t source_capacity_bytes = 0;
    std::size_t destination_capacity_bytes = 0;
    std::size_t source_allocation_count = 0;
    std::size_t destination_allocation_count = 0;
    int source_device = -1;
    int destination_device = -1;
};

class CudaPeerPackScratch final
{
public:
    CudaPeerPackScratch() = default;
    CudaPeerPackScratch(const CudaPeerPackScratch &) = delete;
    CudaPeerPackScratch &operator=(const CudaPeerPackScratch &) = delete;
    ~CudaPeerPackScratch();

    void reserve(
        int source_device, int destination_device, std::size_t bytes);
    void release() noexcept;
    CudaPeerPackScratchSnapshot snapshot() const;

private:
    friend class CudaPeerComm;

    static void reserve_buffer(
        void *&buffer,
        std::size_t &capacity_bytes,
        int &device,
        std::size_t &allocation_count,
        int requested_device,
        std::size_t requested_bytes,
        const char *set_device_what,
        const char *malloc_what,
        const char *free_what);
    static void release_buffer(
        void *&buffer,
        std::size_t &capacity_bytes,
        int &device) noexcept;

    void *source_pack_ = nullptr;
    void *destination_pack_ = nullptr;
    std::size_t source_capacity_bytes_ = 0;
    std::size_t destination_capacity_bytes_ = 0;
    std::size_t source_allocation_count_ = 0;
    std::size_t destination_allocation_count_ = 0;
    int source_device_ = -1;
    int destination_device_ = -1;
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
    void copy_objects(const std::vector<GpuObjectCopyRequest> &requests) override;
    void copy_objects_pack_unpack(
        const std::vector<GpuObjectCopyRequest> &requests);
    void copy_objects_pack_unpack(
        const std::vector<GpuObjectCopyRequest> &requests,
        CudaPeerPackScratch &scratch);

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
