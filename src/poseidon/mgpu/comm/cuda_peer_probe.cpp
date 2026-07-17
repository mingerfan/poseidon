#include "poseidon/mgpu/comm/cuda_peer_probe.h"

#include <ostream>
#include <string>

namespace poseidon::mgpu
{

CudaPeerProbeResult probe_cuda_peer_access()
{
    return runtime_api::communication::probe_cuda_topology();
}

bool cuda_peer_probe_has_full_peer_access(
    const CudaPeerProbeResult &result, int required_devices)
{
    return runtime_api::communication::has_full_peer_access(result, required_devices);
}

std::string dump_cuda_peer_probe(const CudaPeerProbeResult &result)
{
    return runtime_api::communication::dump_cuda_topology(result);
}

void dump_cuda_peer_probe(std::ostream &stream, const CudaPeerProbeResult &result)
{
    runtime_api::communication::dump_cuda_topology(stream, result);
}

std::string cuda_peer_probe_to_json(const CudaPeerProbeResult &result, int indent)
{
    return runtime_api::communication::cuda_topology_to_json(result, indent);
}

}  // namespace poseidon::mgpu
