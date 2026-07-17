#include "poseidon/mgpu/comm/cuda_peer_comm.h"
#include "poseidon/runtime_api/communication/cuda_local_transfer.h"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace poseidon::mgpu;

namespace
{

constexpr int kSkip = 77;

void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

int visible_device_count_or_skip()
{
    try
    {
        const int count = CudaPeerComm::visible_device_count();
        if (count <= 0)
        {
            std::cout << "no CUDA devices visible; skipping CUDA comm test\n";
            std::exit(kSkip);
        }
        return count;
    }
    catch (const std::exception &ex)
    {
        std::cout << "CUDA runtime unavailable: " << ex.what() << "; skipping CUDA comm test\n";
        std::exit(kSkip);
    }
}

void test_same_device_copy()
{
    CudaPeerComm comm;
    int *source = nullptr;
    int *destination = nullptr;
    const int input = 12345;
    int output = 0;

    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&source), sizeof(int)), "cudaMalloc source");
    check_cuda(
        cudaMalloc(reinterpret_cast<void **>(&destination), sizeof(int)),
        "cudaMalloc destination");

    try
    {
        check_cuda(
            cudaMemcpy(source, &input, sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy host to device");
        comm.copy_buffer(CudaPeerCopyRequest{ source, destination, sizeof(int), 0, 0 });
        check_cuda(
            cudaMemcpy(&output, destination, sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy device to host");
        require(output == input, "same-device CUDA copy result mismatch");
    }
    catch (...)
    {
        cudaFree(destination);
        cudaFree(source);
        throw;
    }

    check_cuda(cudaFree(destination), "cudaFree destination");
    check_cuda(cudaFree(source), "cudaFree source");
}

void test_route_selection_without_multiple_devices()
{
    using poseidon::runtime_api::communication::CudaLocalTransfer;
    using poseidon::runtime_api::communication::CudaTransferRoute;

    require(CudaLocalTransfer::select_route(0, 0, CudaTransferRoute::Auto) ==
                CudaTransferRoute::SameDevice,
            "auto route should select same-device copy");

    bool rejected = false;
    try
    {
        (void)CudaLocalTransfer::select_route(0, 0, CudaTransferRoute::PeerToPeer);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    require(rejected, "peer route should reject identical endpoints");
}

void test_same_device_object_copy()
{
    CudaPeerComm comm;
    int *source = nullptr;
    int *destination = nullptr;
    const int input = 24680;
    int output = 0;

    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&source), sizeof(int)), "cudaMalloc source");
    check_cuda(
        cudaMalloc(reinterpret_cast<void **>(&destination), sizeof(int)),
        "cudaMalloc destination");

    try
    {
        check_cuda(
            cudaMemcpy(source, &input, sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy host to device");

        GpuObjectCopyRequest request;
        request.source_id = 1;
        request.destination_id = 2;
        request.kind = MgpuValueKind::Ciphertext;
        request.buffers.push_back(GpuObjectBufferCopy{
            source,
            destination,
            sizeof(int),
            0,
            0,
        });

        comm.copy_object(request);
        check_cuda(
            cudaMemcpy(&output, destination, sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy device to host");
        require(output == input, "same-device CUDA object copy result mismatch");
    }
    catch (...)
    {
        cudaFree(destination);
        cudaFree(source);
        throw;
    }

    check_cuda(cudaFree(destination), "cudaFree destination");
    check_cuda(cudaFree(source), "cudaFree source");
}

void test_cross_device_copy_if_available(int device_count)
{
    if (device_count < 2)
    {
        return;
    }

    CudaPeerComm comm;
    int *source = nullptr;
    int *destination = nullptr;
    const int input = 67890;
    int output = 0;

    check_cuda(cudaSetDevice(0), "cudaSetDevice source");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&source), sizeof(int)), "cudaMalloc source");
    check_cuda(
        cudaMemcpy(source, &input, sizeof(int), cudaMemcpyHostToDevice),
        "cudaMemcpy host to source device");

    check_cuda(cudaSetDevice(1), "cudaSetDevice destination");
    check_cuda(
        cudaMalloc(reinterpret_cast<void **>(&destination), sizeof(int)),
        "cudaMalloc destination");

    try
    {
        comm.copy_buffer(CudaPeerCopyRequest{ source, destination, sizeof(int), 0, 1 });
        check_cuda(cudaSetDevice(1), "cudaSetDevice destination readback");
        check_cuda(
            cudaMemcpy(&output, destination, sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy destination device to host");
        require(output == input, "cross-device CUDA copy result mismatch");
    }
    catch (...)
    {
        cudaSetDevice(1);
        cudaFree(destination);
        cudaSetDevice(0);
        cudaFree(source);
        throw;
    }

    check_cuda(cudaSetDevice(1), "cudaSetDevice destination free");
    check_cuda(cudaFree(destination), "cudaFree destination");
    check_cuda(cudaSetDevice(0), "cudaSetDevice source free");
    check_cuda(cudaFree(source), "cudaFree source");
}

void test_cross_device_batch_copy_if_available(int device_count)
{
    if (device_count < 2)
    {
        return;
    }

    CudaPeerComm comm;
    std::vector<int *> sources(3, nullptr);
    std::vector<int *> destinations(3, nullptr);
    const std::vector<int> inputs{ 101, 202, 303 };
    std::vector<int> outputs(3, 0);
    std::vector<GpuObjectCopyRequest> requests;

    try
    {
        for (std::size_t index = 0; index < inputs.size(); ++index)
        {
            check_cuda(cudaSetDevice(0), "cudaSetDevice source");
            check_cuda(
                cudaMalloc(reinterpret_cast<void **>(&sources[index]), sizeof(int)),
                "cudaMalloc batch source");
            check_cuda(
                cudaMemcpy(
                    sources[index],
                    &inputs[index],
                    sizeof(int),
                    cudaMemcpyHostToDevice),
                "cudaMemcpy host to batch source device");

            check_cuda(cudaSetDevice(1), "cudaSetDevice destination");
            check_cuda(
                cudaMalloc(reinterpret_cast<void **>(&destinations[index]), sizeof(int)),
                "cudaMalloc batch destination");

            GpuObjectCopyRequest request;
            request.source_id = static_cast<ValueId>(10 + index);
            request.destination_id = static_cast<ValueId>(20 + index);
            request.kind = MgpuValueKind::Ciphertext;
            request.buffers.push_back(GpuObjectBufferCopy{
                sources[index],
                destinations[index],
                sizeof(int),
                0,
                1,
            });
            requests.push_back(request);
        }

        comm.copy_objects(requests);
        check_cuda(cudaSetDevice(1), "cudaSetDevice batch destination readback");
        for (std::size_t index = 0; index < outputs.size(); ++index)
        {
            check_cuda(
                cudaMemcpy(
                    &outputs[index],
                    destinations[index],
                    sizeof(int),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy batch destination device to host");
            require(outputs[index] == inputs[index], "cross-device CUDA batch copy result mismatch");
        }
    }
    catch (...)
    {
        for (int *destination : destinations)
        {
            if (destination != nullptr)
            {
                cudaSetDevice(1);
                cudaFree(destination);
            }
        }
        for (int *source : sources)
        {
            if (source != nullptr)
            {
                cudaSetDevice(0);
                cudaFree(source);
            }
        }
        throw;
    }

    for (int *destination : destinations)
    {
        check_cuda(cudaSetDevice(1), "cudaSetDevice batch destination free");
        check_cuda(cudaFree(destination), "cudaFree batch destination");
    }
    for (int *source : sources)
    {
        check_cuda(cudaSetDevice(0), "cudaSetDevice batch source free");
        check_cuda(cudaFree(source), "cudaFree batch source");
    }
}

}  // namespace

int main()
{
    try
    {
        const int device_count = visible_device_count_or_skip();
        test_route_selection_without_multiple_devices();
        test_same_device_copy();
        test_same_device_object_copy();
        test_cross_device_copy_if_available(device_count);
        test_cross_device_batch_copy_if_available(device_count);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu CUDA comm test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu CUDA comm tests passed\n";
    return EXIT_SUCCESS;
}
