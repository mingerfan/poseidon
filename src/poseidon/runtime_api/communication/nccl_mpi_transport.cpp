#include "poseidon/runtime_api/communication/nccl_mpi_transport.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace poseidon::runtime_api::communication
{

struct NcclMpiTransport::Request::State
{
    cudaEvent_t completion_event = nullptr;
    int cuda_device_id = 0;
    cudaStream_t stream = nullptr;
    ncclComm_t communicator = nullptr;
};

struct NcclMpiTransport::DeviceComm
{
    int cuda_device_id = 0;
    ncclComm_t communicator = nullptr;
    cudaStream_t stream = nullptr;
};

NcclMpiTransport::Request::Request() = default;

NcclMpiTransport::Request::Request(std::unique_ptr<State> state)
    : state_(std::move(state))
{}

NcclMpiTransport::Request::~Request()
{
    if (state_ == nullptr)
    {
        return;
    }
    try
    {
        if (state_->completion_event != nullptr)
        {
            (void)cudaSetDevice(state_->cuda_device_id);
            (void)cudaEventSynchronize(state_->completion_event);
            (void)cudaEventDestroy(state_->completion_event);
        }
    }
    catch (...)
    {}
}

NcclMpiTransport::Request::Request(Request &&other) noexcept = default;

NcclMpiTransport::Request &NcclMpiTransport::Request::operator=(
    Request &&other) noexcept = default;

bool NcclMpiTransport::Request::valid() const noexcept
{
    return state_ != nullptr;
}

void NcclMpiTransport::check_nccl(ncclResult_t status, const char *what)
{
    if (status != ncclSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " +
                                 ncclGetErrorString(status));
    }
}

void NcclMpiTransport::check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(status));
    }
}

void NcclMpiTransport::check_mpi(int status, const char *what)
{
    if (status != MPI_SUCCESS)
    {
        char message[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        (void)MPI_Error_string(status, message, &length);
        throw std::runtime_error(std::string(what) + ": " +
                                 std::string(message, static_cast<std::size_t>(length)));
    }
}

NcclMpiTransport::NcclMpiTransport(NcclProcessTopology topology)
{
    int initialized = 0;
    check_mpi(MPI_Initialized(&initialized), "MPI_Initialized");
    if (!initialized)
    {
        throw std::invalid_argument("NCCL MPI transport requires initialized MPI");
    }
    int finalized = 0;
    check_mpi(MPI_Finalized(&finalized), "MPI_Finalized");
    if (finalized)
    {
        throw std::invalid_argument("NCCL MPI transport cannot start after MPI_Finalize");
    }

    int thread_level = MPI_THREAD_SINGLE;
    check_mpi(MPI_Query_thread(&thread_level), "MPI_Query_thread");
    if (thread_level < MPI_THREAD_FUNNELED)
    {
        throw std::runtime_error(
            "NCCL MPI transport requires MPI_THREAD_FUNNELED or stronger");
    }
    if (topology.control_comm == MPI_COMM_NULL)
    {
        throw std::invalid_argument("NCCL MPI transport control communicator is null");
    }

    check_mpi(MPI_Comm_dup(topology.control_comm, &control_comm_), "MPI_Comm_dup");
    try
    {
        check_mpi(MPI_Comm_rank(control_comm_, &mpi_rank_), "MPI_Comm_rank");
        check_mpi(MPI_Comm_size(control_comm_, &mpi_world_size_), "MPI_Comm_size");

        if (topology.device_counts.size() !=
            static_cast<std::size_t>(mpi_world_size_))
        {
            throw std::invalid_argument(
                "NCCL MPI transport device_counts length does not match MPI world size");
        }
        device_counts_ = std::move(topology.device_counts);
        if (std::any_of(device_counts_.begin(), device_counts_.end(),
                        [](int count) { return count < 0; }))
        {
            throw std::invalid_argument(
                "NCCL MPI transport device counts must be nonnegative");
        }
        if (device_counts_[static_cast<std::size_t>(mpi_rank_)] !=
            static_cast<int>(topology.local_cuda_device_ids.size()))
        {
            throw std::invalid_argument(
                "NCCL MPI transport local CUDA count does not match device_counts");
        }
        if (topology.local_cuda_device_ids.empty())
        {
            throw std::invalid_argument(
                "NCCL MPI transport requires at least one local CUDA device");
        }

        nccl_rank_offsets_.resize(device_counts_.size(), 0);
        for (std::size_t index = 1; index < device_counts_.size(); ++index)
        {
            if (device_counts_[index - 1] >
                std::numeric_limits<int>::max() -
                    nccl_rank_offsets_[index - 1])
            {
                throw std::overflow_error("NCCL rank offset overflow");
            }
            nccl_rank_offsets_[index] =
                nccl_rank_offsets_[index - 1] + device_counts_[index - 1];
        }
        for (int count : device_counts_)
        {
            if (count > std::numeric_limits<int>::max() - total_gpu_count_)
            {
                throw std::overflow_error("NCCL rank count overflow");
            }
            total_gpu_count_ += count;
        }
        if (total_gpu_count_ <= 0)
        {
            throw std::invalid_argument("NCCL MPI transport has no GPU ranks");
        }

        int visible_devices = 0;
        check_cuda(cudaGetDeviceCount(&visible_devices), "cudaGetDeviceCount");
        for (std::size_t index = 0; index < topology.local_cuda_device_ids.size();
             ++index)
        {
            const int device = topology.local_cuda_device_ids[index];
            if (device < 0 || device >= visible_devices)
            {
                throw std::invalid_argument(
                    "NCCL MPI transport local CUDA device is unavailable");
            }
            if (std::find(topology.local_cuda_device_ids.begin(),
                          topology.local_cuda_device_ids.begin() +
                              static_cast<std::ptrdiff_t>(index),
                          device) !=
                topology.local_cuda_device_ids.begin() +
                    static_cast<std::ptrdiff_t>(index))
            {
                throw std::invalid_argument(
                    "NCCL MPI transport local CUDA devices must be unique");
            }
        }

        ncclUniqueId unique_id{};
        if (mpi_rank_ == 0)
        {
            check_nccl(ncclGetUniqueId(&unique_id), "ncclGetUniqueId");
        }
        if (sizeof(unique_id) > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("NCCL unique id is too large for MPI_Bcast");
        }
        check_mpi(MPI_Bcast(&unique_id, static_cast<int>(sizeof(unique_id)), MPI_BYTE,
                            0, control_comm_),
                  "MPI_Bcast ncclUniqueId");

        devices_.reserve(topology.local_cuda_device_ids.size());
        for (int device : topology.local_cuda_device_ids)
        {
            auto state = std::make_unique<DeviceComm>();
            state->cuda_device_id = device;
            check_cuda(cudaSetDevice(device), "cudaSetDevice NCCL communicator");
            check_cuda(cudaStreamCreateWithFlags(&state->stream,
                                                 cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags NCCL stream");
            devices_.push_back(std::move(state));
        }

        check_nccl(ncclGroupStart(), "ncclGroupStart communicator initialization");
        group_active_ = true;
        for (std::size_t index = 0; index < devices_.size(); ++index)
        {
            DeviceComm &state = *devices_[index];
            check_cuda(cudaSetDevice(state.cuda_device_id),
                       "cudaSetDevice NCCL communicator initialization");
            check_nccl(
                ncclCommInitRank(
                    &state.communicator, total_gpu_count_, unique_id,
                    nccl_rank(static_cast<int>(index))),
                "ncclCommInitRank");
        }
        check_nccl(ncclGroupEnd(), "ncclGroupEnd communicator initialization");
        group_active_ = false;
    }
    catch (...)
    {
        if (group_active_)
        {
            (void)ncclGroupEnd();
            group_active_ = false;
        }
        destroy_comms();
        if (control_comm_ != MPI_COMM_NULL)
        {
            (void)MPI_Comm_free(&control_comm_);
        }
        throw;
    }
}

NcclMpiTransport::~NcclMpiTransport()
{
    destroy_comms();
    if (control_comm_ != MPI_COMM_NULL)
    {
        int finalized = 0;
        if (MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized)
        {
            (void)MPI_Comm_free(&control_comm_);
        }
        control_comm_ = MPI_COMM_NULL;
    }
}

int NcclMpiTransport::mpi_rank() const noexcept
{
    return mpi_rank_;
}

int NcclMpiTransport::mpi_world_size() const noexcept
{
    return mpi_world_size_;
}

MPI_Comm NcclMpiTransport::control_comm() const noexcept
{
    return control_comm_;
}

int NcclMpiTransport::total_gpu_count() const noexcept
{
    return total_gpu_count_;
}

int NcclMpiTransport::local_device_count() const noexcept
{
    return static_cast<int>(devices_.size());
}

int NcclMpiTransport::cuda_device_id(int local_device_index) const
{
    return device_comm(local_device_index).cuda_device_id;
}

int NcclMpiTransport::nccl_rank(int local_device_index) const
{
    validate_local_device_index(local_device_index);
    return nccl_rank_offsets_[static_cast<std::size_t>(mpi_rank_)] +
           local_device_index;
}

void NcclMpiTransport::group_start()
{
    if (group_active_)
    {
        throw std::logic_error("NCCL operation group is already active");
    }
    check_nccl(ncclGroupStart(), "ncclGroupStart");
    group_active_ = true;
}

void NcclMpiTransport::group_end()
{
    if (!group_active_)
    {
        throw std::logic_error("NCCL operation group is not active");
    }
    try
    {
        check_nccl(ncclGroupEnd(), "ncclGroupEnd");
    }
    catch (...)
    {
        group_active_ = false;
        throw;
    }
    group_active_ = false;
}

NcclMpiTransport::Request NcclMpiTransport::send_async(
    int local_device_index, int peer_nccl_rank, const void *device_buffer,
    std::size_t bytes, cudaEvent_t source_ready)
{
    validate_local_device_index(local_device_index);
    validate_peer_rank(peer_nccl_rank);
    if (!group_active_)
    {
        throw std::logic_error("NCCL send must be posted inside a group");
    }
    if (device_buffer == nullptr || bytes == 0 ||
        bytes % sizeof(std::uint32_t) != 0)
    {
        throw std::invalid_argument(
            "NCCL send requires a non-empty uint32-aligned device buffer");
    }
    DeviceComm &state = device_comm(local_device_index);
    check_cuda(cudaSetDevice(state.cuda_device_id), "cudaSetDevice NCCL send");
    if (source_ready != nullptr)
    {
        check_cuda(cudaStreamWaitEvent(state.stream, source_ready, 0),
                   "cudaStreamWaitEvent NCCL send");
    }
    check_nccl(ncclSend(device_buffer, bytes / sizeof(std::uint32_t), ncclUint32,
                        peer_nccl_rank, state.communicator, state.stream),
               "ncclSend");
    auto request = std::make_unique<Request::State>();
    request->cuda_device_id = state.cuda_device_id;
    request->stream = state.stream;
    request->communicator = state.communicator;
    return Request(std::move(request));
}

NcclMpiTransport::Request NcclMpiTransport::recv_async(
    int local_device_index, int peer_nccl_rank, void *device_buffer,
    std::size_t bytes)
{
    validate_local_device_index(local_device_index);
    validate_peer_rank(peer_nccl_rank);
    if (!group_active_)
    {
        throw std::logic_error("NCCL receive must be posted inside a group");
    }
    if (device_buffer == nullptr || bytes == 0 ||
        bytes % sizeof(std::uint32_t) != 0)
    {
        throw std::invalid_argument(
            "NCCL receive requires a non-empty uint32-aligned device buffer");
    }
    DeviceComm &state = device_comm(local_device_index);
    check_cuda(cudaSetDevice(state.cuda_device_id), "cudaSetDevice NCCL receive");
    check_nccl(ncclRecv(device_buffer, bytes / sizeof(std::uint32_t), ncclUint32,
                        peer_nccl_rank, state.communicator, state.stream),
               "ncclRecv");
    auto request = std::make_unique<Request::State>();
    request->cuda_device_id = state.cuda_device_id;
    request->stream = state.stream;
    request->communicator = state.communicator;
    return Request(std::move(request));
}

void NcclMpiTransport::record_event(Request &request)
{
    if (!request.state_)
    {
        throw std::invalid_argument("NCCL request is empty");
    }
    Request::State &state = *request.state_;
    if (state.completion_event != nullptr)
    {
        throw std::logic_error("NCCL request completion event was already recorded");
    }
    check_cuda(cudaSetDevice(state.cuda_device_id), "cudaSetDevice NCCL record");
    check_cuda(cudaEventCreateWithFlags(&state.completion_event,
                                        cudaEventDisableTiming),
               "cudaEventCreateWithFlags NCCL request");
    check_cuda(cudaEventRecord(state.completion_event, state.stream),
               "cudaEventRecord NCCL request");
}

void NcclMpiTransport::wait(Request &request)
{
    if (!request.state_)
    {
        throw std::invalid_argument("NCCL request is empty");
    }
    Request::State &state = *request.state_;
    if (state.completion_event == nullptr)
    {
        throw std::logic_error(
            "NCCL request completion event was not recorded after group_end");
    }
    check_cuda(cudaSetDevice(state.cuda_device_id), "cudaSetDevice NCCL wait");
    check_cuda(cudaEventSynchronize(state.completion_event),
               "cudaEventSynchronize NCCL request");
    ncclResult_t asynchronous = ncclSuccess;
    check_nccl(ncclCommGetAsyncError(state.communicator, &asynchronous),
               "ncclCommGetAsyncError");
    check_nccl(asynchronous, "NCCL asynchronous request");
    check_cuda(cudaEventDestroy(state.completion_event),
               "cudaEventDestroy NCCL request");
    state.completion_event = nullptr;
    request.state_.reset();
}

void NcclMpiTransport::abort()
{
    if (aborted_)
    {
        return;
    }
    aborted_ = true;
    for (const auto &state : devices_)
    {
        if (state != nullptr && state->communicator != nullptr)
        {
            (void)ncclCommAbort(state->communicator);
            state->communicator = nullptr;
        }
    }
}

void NcclMpiTransport::validate_local_device_index(int local_device_index) const
{
    if (local_device_index < 0 ||
        local_device_index >= static_cast<int>(devices_.size()))
    {
        throw std::out_of_range("NCCL local device index is out of range");
    }
}

void NcclMpiTransport::validate_peer_rank(int peer_nccl_rank) const
{
    if (peer_nccl_rank < 0 || peer_nccl_rank >= total_gpu_count_)
    {
        throw std::out_of_range("NCCL peer rank is out of range");
    }
}

NcclMpiTransport::DeviceComm &NcclMpiTransport::device_comm(
    int local_device_index)
{
    validate_local_device_index(local_device_index);
    return *devices_[static_cast<std::size_t>(local_device_index)];
}

const NcclMpiTransport::DeviceComm &NcclMpiTransport::device_comm(
    int local_device_index) const
{
    validate_local_device_index(local_device_index);
    return *devices_[static_cast<std::size_t>(local_device_index)];
}

void NcclMpiTransport::destroy_comms() noexcept
{
    if (group_active_)
    {
        (void)ncclGroupEnd();
        group_active_ = false;
    }
    for (auto &state : devices_)
    {
        if (state == nullptr)
        {
            continue;
        }
        if (state->cuda_device_id >= 0)
        {
            (void)cudaSetDevice(state->cuda_device_id);
        }
        if (state->stream != nullptr)
        {
            (void)cudaStreamSynchronize(state->stream);
            (void)cudaStreamDestroy(state->stream);
            state->stream = nullptr;
        }
        if (state->communicator != nullptr)
        {
            if (aborted_)
                (void)ncclCommAbort(state->communicator);
            else
                (void)ncclCommDestroy(state->communicator);
            state->communicator = nullptr;
        }
    }
    devices_.clear();
}

} // namespace poseidon::runtime_api::communication
