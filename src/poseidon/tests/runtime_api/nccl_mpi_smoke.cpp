#include "poseidon/runtime_api/communication/nccl_mpi_transport.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::vector<int> parse_counts(const std::string &encoded)
{
    if (encoded.empty())
    {
        throw std::invalid_argument("--device-counts must not be empty");
    }
    std::vector<int> result;
    std::size_t begin = 0;
    while (begin <= encoded.size())
    {
        const std::size_t end = encoded.find('x', begin);
        const std::string token = encoded.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (token.empty())
        {
            throw std::invalid_argument("--device-counts contains an empty entry");
        }
        std::size_t consumed = 0;
        const int value = std::stoi(token, &consumed);
        if (consumed != token.size() || value <= 0)
        {
            throw std::invalid_argument(
                "--device-counts entries must be positive integers");
        }
        result.push_back(value);
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return result;
}

std::vector<int> parse_local_devices(const std::string &encoded)
{
    if (encoded.empty())
    {
        return {};
    }
    std::vector<int> result;
    std::size_t begin = 0;
    while (begin <= encoded.size())
    {
        const std::size_t end = encoded.find(',', begin);
        const std::string token = encoded.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (token.empty())
        {
            throw std::invalid_argument("--local-devices contains an empty entry");
        }
        std::size_t consumed = 0;
        const int value = std::stoi(token, &consumed);
        if (consumed != token.size() || value < 0)
        {
            throw std::invalid_argument(
                "--local-devices entries must be nonnegative integers");
        }
        result.push_back(value);
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return result;
}

void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(status));
    }
}

} // namespace

int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
    {
        return 2;
    }

    int rank = 0;
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    std::string counts_text;
    std::string local_devices_text;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--device-counts" && index + 1 < argc)
        {
            counts_text = argv[++index];
        }
        else if (argument == "--local-devices" && index + 1 < argc)
        {
            local_devices_text = argv[++index];
        }
        else
        {
            std::cerr << "[rank " << rank << "] unknown or incomplete argument: "
                      << argument << '\n';
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    try
    {
        const std::vector<int> counts = parse_counts(counts_text);
        if (static_cast<int>(counts.size()) != world)
        {
            throw std::invalid_argument(
                "--device-counts length must equal MPI world size");
        }
        std::vector<int> local_devices = parse_local_devices(local_devices_text);
        if (local_devices.empty())
        {
            local_devices.resize(static_cast<std::size_t>(counts[rank]));
            for (int index = 0; index < counts[rank]; ++index)
            {
                local_devices[static_cast<std::size_t>(index)] =
                    local_rank * counts[rank] + index;
            }
        }
        if (static_cast<int>(local_devices.size()) != counts[rank])
        {
            throw std::invalid_argument(
                "--local-devices count must match this rank's device count");
        }

        poseidon::runtime_api::communication::NcclMpiTransport transport({
            MPI_COMM_WORLD, local_devices, counts});

        std::vector<void *> send_buffers(local_devices.size(), nullptr);
        std::vector<void *> receive_buffers(local_devices.size(), nullptr);
        std::vector<std::uint32_t> received(local_devices.size(), 0);
        std::vector<poseidon::runtime_api::communication::NcclMpiTransport::Request>
            receives;
        std::vector<poseidon::runtime_api::communication::NcclMpiTransport::Request>
            sends;
        receives.reserve(local_devices.size());
        sends.reserve(local_devices.size());

        for (std::size_t index = 0; index < local_devices.size(); ++index)
        {
            const int device = local_devices[index];
            check_cuda(cudaSetDevice(device), "cudaSetDevice smoke");
            check_cuda(cudaMalloc(&send_buffers[index], sizeof(std::uint32_t)),
                       "cudaMalloc smoke send");
            check_cuda(cudaMalloc(&receive_buffers[index], sizeof(std::uint32_t)),
                       "cudaMalloc smoke receive");
            const std::uint32_t value = static_cast<std::uint32_t>(
                transport.nccl_rank(static_cast<int>(index)) + 1000);
            check_cuda(cudaMemcpy(send_buffers[index], &value, sizeof(value),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy smoke input");
        }

        transport.group_start();
        for (std::size_t index = 0; index < local_devices.size(); ++index)
        {
            const int global_rank = transport.nccl_rank(static_cast<int>(index));
            const int previous_rank =
                (global_rank + transport.total_gpu_count() - 1) %
                transport.total_gpu_count();
            receives.push_back(transport.recv_async(
                static_cast<int>(index), previous_rank, receive_buffers[index],
                sizeof(std::uint32_t)));
            sends.push_back(transport.send_async(
                static_cast<int>(index),
                (global_rank + 1) % transport.total_gpu_count(),
                send_buffers[index],
                sizeof(std::uint32_t)));
        }
        transport.group_end();
        for (auto &request : receives)
        {
            transport.record_event(request);
        }
        for (auto &request : sends)
        {
            transport.record_event(request);
        }

        for (auto &request : receives)
        {
            transport.wait(request);
        }
        for (auto &request : sends)
        {
            transport.wait(request);
        }

        for (std::size_t index = 0; index < local_devices.size(); ++index)
        {
            const int global_rank = transport.nccl_rank(static_cast<int>(index));
            const int previous_rank =
                (global_rank + transport.total_gpu_count() - 1) %
                transport.total_gpu_count();
            check_cuda(cudaSetDevice(local_devices[index]),
                       "cudaSetDevice smoke output");
            check_cuda(cudaMemcpy(&received[index], receive_buffers[index],
                                  sizeof(received[index]),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy smoke output");
            const std::uint32_t expected =
                static_cast<std::uint32_t>(previous_rank + 1000);
            if (received[index] != expected)
            {
                std::ostringstream message;
                message << "rank " << rank << " local GPU " << index
                        << " received " << received[index] << ", expected "
                        << expected;
                throw std::runtime_error(message.str());
            }
        }

        for (std::size_t index = 0; index < send_buffers.size(); ++index)
        {
            check_cuda(cudaSetDevice(local_devices[index]),
                       "cudaSetDevice smoke cleanup");
            check_cuda(cudaFree(send_buffers[index]), "cudaFree smoke send");
            check_cuda(cudaFree(receive_buffers[index]),
                       "cudaFree smoke receive");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
        {
            std::cout << "NCCL MPI smoke passed: mpi_world=" << world
                      << " total_gpu_ranks=" << transport.total_gpu_count()
                      << '\n';
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "[rank " << rank << "] NCCL MPI smoke failed: "
                  << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
