#pragma once

#include <cuda_runtime_api.h>
#include <nccl.h>

#include <cstddef>
#include <vector>

namespace poseidon::mgpu
{

class NcclComm
{
public:
    explicit NcclComm(std::vector<int> devices);
    ~NcclComm();

    NcclComm(const NcclComm &) = delete;
    NcclComm &operator=(const NcclComm &) = delete;

    const std::vector<int> &devices() const noexcept;
    int size() const noexcept;
    int rank_for_device(int device_id) const;
    bool has_native_gather() const noexcept;

    void broadcast(const std::vector<void *> &buffers, std::size_t bytes, int root_rank);
    void gather(
        const std::vector<const void *> &send_buffers, void *root_recv_buffer,
        std::size_t bytes, int root_rank);
    void send_recv(
        const void *send_buffer, void *recv_buffer, std::size_t bytes,
        int source_rank, int destination_rank);

    void synchronize_streams() const;

private:
    void validate_rank(int rank, const char *name) const;
    void validate_buffer_count(std::size_t count, const char *name) const;
    void destroy() noexcept;

private:
    std::vector<int> devices_;
    std::vector<ncclComm_t> comms_;
    std::vector<cudaStream_t> streams_;
};

}  // namespace poseidon::mgpu
