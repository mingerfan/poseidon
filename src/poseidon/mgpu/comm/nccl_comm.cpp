#include "poseidon/mgpu/comm/nccl_comm.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef POSEIDON_MGPU_HAS_NCCL_GATHER
#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 22800
#define POSEIDON_MGPU_HAS_NCCL_GATHER 1
#else
#define POSEIDON_MGPU_HAS_NCCL_GATHER 0
#endif
#endif

namespace poseidon::mgpu
{
namespace
{

void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_nccl(ncclResult_t status, const char *what)
{
    if (status != ncclSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(status));
    }
}

void check_bytes(std::size_t bytes)
{
    if (bytes == 0)
    {
        throw std::invalid_argument("NCCL transfer byte count must be non-zero");
    }
}

void check_pointer(const void *pointer, const char *name)
{
    if (pointer == nullptr)
    {
        throw std::invalid_argument(std::string(name) + " is null");
    }
}

}  // namespace

NcclComm::NcclComm(std::vector<int> devices)
    : devices_(std::move(devices)),
      comms_(devices_.size(), nullptr),
      streams_(devices_.size(), nullptr)
{
    if (devices_.empty())
    {
        throw std::invalid_argument("NcclComm requires at least one device");
    }
    if (std::any_of(devices_.begin(), devices_.end(), [](int device) { return device < 0; }))
    {
        throw std::invalid_argument("NcclComm device ids must be non-negative");
    }
    std::vector<int> sorted = devices_;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    {
        throw std::invalid_argument("NcclComm device ids must be unique");
    }

    int visible_devices = 0;
    check_cuda(cudaGetDeviceCount(&visible_devices), "NcclComm cudaGetDeviceCount");
    for (const int device : devices_)
    {
        if (device >= visible_devices)
        {
            std::ostringstream stream;
            stream << "NcclComm requested device " << device
                   << " but only " << visible_devices << " CUDA devices are visible";
            throw std::runtime_error(stream.str());
        }
    }

    try
    {
        check_nccl(
            ncclCommInitAll(comms_.data(), static_cast<int>(devices_.size()), devices_.data()),
            "NcclComm ncclCommInitAll");
        for (std::size_t rank = 0; rank < devices_.size(); ++rank)
        {
            check_cuda(cudaSetDevice(devices_[rank]), "NcclComm cudaSetDevice");
            check_cuda(
                cudaStreamCreateWithFlags(&streams_[rank], cudaStreamNonBlocking),
                "NcclComm cudaStreamCreateWithFlags");
        }
    }
    catch (...)
    {
        destroy();
        throw;
    }
}

NcclComm::~NcclComm()
{
    destroy();
}

const std::vector<int> &NcclComm::devices() const noexcept
{
    return devices_;
}

int NcclComm::size() const noexcept
{
    return static_cast<int>(devices_.size());
}

int NcclComm::rank_for_device(int device_id) const
{
    const auto iter = std::find(devices_.begin(), devices_.end(), device_id);
    if (iter == devices_.end())
    {
        throw std::invalid_argument("device id is not part of this NCCL communicator");
    }
    return static_cast<int>(std::distance(devices_.begin(), iter));
}

bool NcclComm::has_native_gather() const noexcept
{
    return POSEIDON_MGPU_HAS_NCCL_GATHER != 0;
}

void NcclComm::broadcast(
    const std::vector<void *> &buffers, std::size_t bytes, int root_rank)
{
    check_bytes(bytes);
    validate_rank(root_rank, "root rank");
    validate_buffer_count(buffers.size(), "broadcast buffers");
    for (const void *buffer : buffers)
    {
        check_pointer(buffer, "broadcast buffer");
    }

    check_nccl(ncclGroupStart(), "NcclComm ncclGroupStart broadcast");
    for (std::size_t rank = 0; rank < devices_.size(); ++rank)
    {
        const void *send_buffer = buffers[rank];
        if (rank == static_cast<std::size_t>(root_rank))
        {
            send_buffer = buffers[static_cast<std::size_t>(root_rank)];
        }
        check_nccl(
            ncclBroadcast(
                send_buffer,
                buffers[rank],
                bytes,
                ncclUint8,
                root_rank,
                comms_[rank],
                streams_[rank]),
            "NcclComm ncclBroadcast");
    }
    check_nccl(ncclGroupEnd(), "NcclComm ncclGroupEnd broadcast");
}

void NcclComm::gather(
    const std::vector<const void *> &send_buffers, void *root_recv_buffer,
    std::size_t bytes, int root_rank)
{
    check_bytes(bytes);
    validate_rank(root_rank, "root rank");
    validate_buffer_count(send_buffers.size(), "gather send buffers");
    check_pointer(root_recv_buffer, "gather root receive buffer");
    for (const void *buffer : send_buffers)
    {
        check_pointer(buffer, "gather send buffer");
    }

#if POSEIDON_MGPU_HAS_NCCL_GATHER
    check_nccl(ncclGroupStart(), "NcclComm ncclGroupStart gather");
    for (std::size_t rank = 0; rank < devices_.size(); ++rank)
    {
        void *recv_buffer = const_cast<void *>(send_buffers[rank]);
        if (rank == static_cast<std::size_t>(root_rank))
        {
            recv_buffer = root_recv_buffer;
        }
        check_nccl(
            ncclGather(
                send_buffers[rank],
                recv_buffer,
                bytes,
                ncclUint8,
                root_rank,
                comms_[rank],
                streams_[rank]),
            "NcclComm ncclGather");
    }
    check_nccl(ncclGroupEnd(), "NcclComm ncclGroupEnd gather");
#else
    const auto root_index = static_cast<std::size_t>(root_rank);
    check_cuda(cudaSetDevice(devices_[root_index]), "NcclComm cudaSetDevice gather root");
    void *root_slot =
        static_cast<unsigned char *>(root_recv_buffer) + root_index * bytes;
    if (send_buffers[root_index] != root_slot)
    {
        check_cuda(
            cudaMemcpyAsync(
                root_slot,
                send_buffers[root_index],
                bytes,
                cudaMemcpyDeviceToDevice,
                streams_[root_index]),
            "NcclComm cudaMemcpyAsync gather root self");
    }

    check_nccl(ncclGroupStart(), "NcclComm ncclGroupStart gather");
    for (std::size_t rank = 0; rank < devices_.size(); ++rank)
    {
        if (rank == root_index)
        {
            continue;
        }
        check_nccl(
            ncclSend(
                send_buffers[rank],
                bytes,
                ncclUint8,
                root_rank,
                comms_[rank],
                streams_[rank]),
            "NcclComm ncclSend gather");
        check_nccl(
            ncclRecv(
                static_cast<unsigned char *>(root_recv_buffer) + rank * bytes,
                bytes,
                ncclUint8,
                static_cast<int>(rank),
                comms_[root_index],
                streams_[root_index]),
            "NcclComm ncclRecv gather");
    }
    check_nccl(ncclGroupEnd(), "NcclComm ncclGroupEnd gather");
#endif
}

void NcclComm::send_recv(
    const void *send_buffer, void *recv_buffer, std::size_t bytes,
    int source_rank, int destination_rank)
{
    check_bytes(bytes);
    validate_rank(source_rank, "source rank");
    validate_rank(destination_rank, "destination rank");
    if (source_rank == destination_rank)
    {
        throw std::invalid_argument("NCCL send/recv source and destination ranks must differ");
    }
    check_pointer(send_buffer, "send buffer");
    check_pointer(recv_buffer, "receive buffer");

    check_nccl(ncclGroupStart(), "NcclComm ncclGroupStart sendrecv");
    check_nccl(
        ncclSend(
            send_buffer,
            bytes,
            ncclUint8,
            destination_rank,
            comms_[static_cast<std::size_t>(source_rank)],
            streams_[static_cast<std::size_t>(source_rank)]),
        "NcclComm ncclSend sendrecv");
    check_nccl(
        ncclRecv(
            recv_buffer,
            bytes,
            ncclUint8,
            source_rank,
            comms_[static_cast<std::size_t>(destination_rank)],
            streams_[static_cast<std::size_t>(destination_rank)]),
        "NcclComm ncclRecv sendrecv");
    check_nccl(ncclGroupEnd(), "NcclComm ncclGroupEnd sendrecv");
}

void NcclComm::synchronize_streams() const
{
    for (std::size_t rank = 0; rank < devices_.size(); ++rank)
    {
        check_cuda(cudaSetDevice(devices_[rank]), "NcclComm cudaSetDevice synchronize");
        check_cuda(cudaStreamSynchronize(streams_[rank]), "NcclComm cudaStreamSynchronize");
    }
}

void NcclComm::validate_rank(int rank, const char *name) const
{
    if (rank < 0 || rank >= static_cast<int>(devices_.size()))
    {
        throw std::out_of_range(std::string("NcclComm ") + name + " is out of range");
    }
}

void NcclComm::validate_buffer_count(std::size_t count, const char *name) const
{
    if (count != devices_.size())
    {
        throw std::invalid_argument(std::string("NcclComm ") + name + " count mismatch");
    }
}

void NcclComm::destroy() noexcept
{
    for (std::size_t rank = 0; rank < streams_.size(); ++rank)
    {
        if (streams_[rank] != nullptr)
        {
            (void)cudaSetDevice(devices_[rank]);
            (void)cudaStreamDestroy(streams_[rank]);
            streams_[rank] = nullptr;
        }
    }
    for (ncclComm_t &comm : comms_)
    {
        if (comm != nullptr)
        {
            (void)ncclCommDestroy(comm);
            comm = nullptr;
        }
    }
}

}  // namespace poseidon::mgpu
