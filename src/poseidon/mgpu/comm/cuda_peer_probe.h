#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct CudaPeerDeviceInfo
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

struct CudaPeerProbeResult
{
    int visible_device_count = 0;
    std::vector<CudaPeerDeviceInfo> devices;
    std::vector<std::vector<bool>> peer_access;
};

CudaPeerProbeResult probe_cuda_peer_access();

bool cuda_peer_probe_has_full_peer_access(
    const CudaPeerProbeResult &result, int required_devices = 0);

std::string dump_cuda_peer_probe(const CudaPeerProbeResult &result);
void dump_cuda_peer_probe(std::ostream &stream, const CudaPeerProbeResult &result);

std::string cuda_peer_probe_to_json(
    const CudaPeerProbeResult &result, int indent = 2);

}  // namespace poseidon::mgpu
