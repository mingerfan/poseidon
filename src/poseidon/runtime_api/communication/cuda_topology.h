#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace poseidon::runtime_api::communication
{

struct CudaDeviceInfo
{
    int device_id = 0;
    std::string name;
    int pci_bus_id = 0;
    int pci_device_id = 0;
    int pci_domain_id = 0;
    int multiprocessor_count = 0;
    int major = 0;
    int minor = 0;
};

struct CudaTopology
{
    int visible_device_count = 0;
    std::vector<CudaDeviceInfo> devices;
    std::vector<std::vector<bool>> peer_access;
};

CudaTopology probe_cuda_topology();
bool has_full_peer_access(const CudaTopology &topology, int required_devices = 0);
std::string dump_cuda_topology(const CudaTopology &topology);
void dump_cuda_topology(std::ostream &stream, const CudaTopology &topology);
std::string cuda_topology_to_json(const CudaTopology &topology, int indent = 2);

} // namespace poseidon::runtime_api::communication
