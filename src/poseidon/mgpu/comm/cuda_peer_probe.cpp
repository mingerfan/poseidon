#include "poseidon/mgpu/comm/cuda_peer_probe.h"

#include "poseidon/mgpu/comm/cuda_peer_comm.h"
#include "poseidon/util/json.h"

#include <cuda_runtime_api.h>

#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

using Json = nlohmann::json;

void check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

Json devices_to_json(const std::vector<CudaPeerDeviceInfo> &devices)
{
    Json result = Json::array();
    for (const CudaPeerDeviceInfo &device : devices)
    {
        result.push_back(Json{
            { "device_id", device.device_id },
            { "name", device.name },
            { "pci_bus_id", device.pci_bus_id },
            { "pci_device_id", device.pci_device_id },
            { "pci_domain_id", device.pci_domain_id },
            { "multiprocessor_count", device.multiprocessor_count },
            { "compute_capability",
              Json{
                  { "major", device.major },
                  { "minor", device.minor },
              } },
        });
    }
    return result;
}

Json peer_matrix_to_json(const std::vector<std::vector<bool>> &peer_access)
{
    Json matrix = Json::array();
    for (const std::vector<bool> &row : peer_access)
    {
        Json json_row = Json::array();
        for (const bool accessible : row)
        {
            json_row.push_back(accessible);
        }
        matrix.push_back(std::move(json_row));
    }
    return matrix;
}

}  // namespace

CudaPeerProbeResult probe_cuda_peer_access()
{
    CudaPeerProbeResult result;
    result.visible_device_count = CudaPeerComm::visible_device_count();
    result.devices.reserve(static_cast<std::size_t>(result.visible_device_count));
    result.peer_access.assign(
        static_cast<std::size_t>(result.visible_device_count),
        std::vector<bool>(static_cast<std::size_t>(result.visible_device_count), false));

    for (int device = 0; device < result.visible_device_count; ++device)
    {
        cudaDeviceProp properties{};
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "CudaPeerProbe cudaGetDeviceProperties");
        result.devices.push_back(CudaPeerDeviceInfo{
            device,
            properties.name,
            properties.pciBusID,
            properties.pciDeviceID,
            properties.pciDomainID,
            properties.multiProcessorCount,
            properties.major,
            properties.minor,
        });
    }

    for (int destination = 0; destination < result.visible_device_count; ++destination)
    {
        for (int source = 0; source < result.visible_device_count; ++source)
        {
            result.peer_access[static_cast<std::size_t>(destination)]
                              [static_cast<std::size_t>(source)] =
                CudaPeerComm::can_access_peer(destination, source);
        }
    }

    return result;
}

bool cuda_peer_probe_has_full_peer_access(
    const CudaPeerProbeResult &result, int required_devices)
{
    if (required_devices < 0)
    {
        return false;
    }
    if (result.visible_device_count <= 0)
    {
        return false;
    }
    if (required_devices > 0 && result.visible_device_count < required_devices)
    {
        return false;
    }

    const int device_count =
        required_devices > 0 ? required_devices : result.visible_device_count;
    for (int destination = 0; destination < device_count; ++destination)
    {
        for (int source = 0; source < device_count; ++source)
        {
            if (destination == source)
            {
                continue;
            }
            if (!result.peer_access[static_cast<std::size_t>(destination)]
                                   [static_cast<std::size_t>(source)])
            {
                return false;
            }
        }
    }
    return true;
}

std::string dump_cuda_peer_probe(const CudaPeerProbeResult &result)
{
    std::ostringstream stream;
    dump_cuda_peer_probe(stream, result);
    return stream.str();
}

void dump_cuda_peer_probe(std::ostream &stream, const CudaPeerProbeResult &result)
{
    stream << "cuda_peer_probe:\n";
    stream << "  visible_devices: " << result.visible_device_count << '\n';
    stream << "  devices:\n";
    for (const CudaPeerDeviceInfo &device : result.devices)
    {
        stream << "    device " << device.device_id << ": " << device.name
               << " cc=" << device.major << "." << device.minor
               << " sm=" << device.multiprocessor_count
               << " pci=" << device.pci_domain_id << ":" << device.pci_bus_id
               << ":" << device.pci_device_id << '\n';
    }
    stream << "  peer_access_matrix:\n";
    for (int destination = 0; destination < result.visible_device_count; ++destination)
    {
        stream << "    dst " << destination << ":";
        for (int source = 0; source < result.visible_device_count; ++source)
        {
            stream << ' '
                   << (result.peer_access[static_cast<std::size_t>(destination)]
                                        [static_cast<std::size_t>(source)]
                           ? '1'
                           : '0');
        }
        stream << '\n';
    }
}

std::string cuda_peer_probe_to_json(const CudaPeerProbeResult &result, int indent)
{
    Json root;
    root["version"] = 1;
    root["visible_device_count"] = result.visible_device_count;
    root["devices"] = devices_to_json(result.devices);
    root["peer_access_matrix"] = peer_matrix_to_json(result.peer_access);
    root["full_peer_access"] = cuda_peer_probe_has_full_peer_access(result);
    return root.dump(indent);
}

}  // namespace poseidon::mgpu
