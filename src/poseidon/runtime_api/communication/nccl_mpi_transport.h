#pragma once

#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace poseidon::runtime_api::communication
{

/**
 * @brief Process/device mapping used to initialize one global NCCL clique.
 *
 * rank and index use RuntimePlan semantics: rank is an MPI process and index
 * is a CUDA device local to that process. NCCL ranks are derived from the
 * prefix sums of device_counts and are intentionally not exposed as Places.
 */
struct NcclProcessTopology
{
    MPI_Comm control_comm = MPI_COMM_NULL;
    std::vector<int> local_cuda_device_ids;
    std::vector<int> device_counts;
};

class NcclMpiTransport
{
public:
    class Request
    {
    public:
        Request();
        ~Request();

        Request(const Request &) = delete;
        Request &operator=(const Request &) = delete;
        Request(Request &&) noexcept;
        Request &operator=(Request &&) noexcept;

        bool valid() const noexcept;

    private:
        struct State;
        std::unique_ptr<State> state_;

        explicit Request(std::unique_ptr<State> state);
        friend class NcclMpiTransport;
    };

    explicit NcclMpiTransport(NcclProcessTopology topology);
    ~NcclMpiTransport();

    NcclMpiTransport(const NcclMpiTransport &) = delete;
    NcclMpiTransport &operator=(const NcclMpiTransport &) = delete;

    int mpi_rank() const noexcept;
    int mpi_world_size() const noexcept;
    MPI_Comm control_comm() const noexcept;
    int total_gpu_count() const noexcept;
    int local_device_count() const noexcept;
    int cuda_device_id(int local_device_index) const;
    int nccl_rank(int local_device_index) const;

    /** Begin a batch of point-to-point NCCL operations. */
    void group_start();

    /** Finish a batch started by group_start(). */
    void group_end();

    /** Record completion events after the enclosing NCCL group has ended. */
    void record_event(Request &request);

    /**
     * Enqueue a raw uint32 device-buffer send on a local NCCL rank.
     * The caller must place sends/receives between group_start/group_end.
     */
    Request send_async(int local_device_index, int peer_nccl_rank,
                       const void *device_buffer, std::size_t bytes,
                       cudaEvent_t source_ready = nullptr);

    /** Enqueue a raw uint32 device-buffer receive. */
    Request recv_async(int local_device_index, int peer_nccl_rank,
                       void *device_buffer, std::size_t bytes);

    /** Wait for a request and surface asynchronous CUDA/NCCL failures. */
    void wait(Request &request);

    /** Abort all NCCL communicators. Safe to call once during failure handling. */
    void abort();

private:
    struct DeviceComm;

    static void check_nccl(ncclResult_t status, const char *what);
    static void check_cuda(cudaError_t status, const char *what);
    static void check_mpi(int status, const char *what);

    void validate_local_device_index(int local_device_index) const;
    void validate_peer_rank(int peer_nccl_rank) const;
    DeviceComm &device_comm(int local_device_index);
    const DeviceComm &device_comm(int local_device_index) const;
    void destroy_comms() noexcept;

    MPI_Comm control_comm_ = MPI_COMM_NULL;
    int mpi_rank_ = 0;
    int mpi_world_size_ = 1;
    int total_gpu_count_ = 0;
    std::vector<int> device_counts_;
    std::vector<int> nccl_rank_offsets_;
    std::vector<std::unique_ptr<DeviceComm>> devices_;
    bool group_active_ = false;
    bool aborted_ = false;
};

} // namespace poseidon::runtime_api::communication
