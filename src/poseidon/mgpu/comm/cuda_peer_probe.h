#pragma once

#include "poseidon/runtime_api/communication/cuda_topology.h"

#include <iosfwd>
#include <string>

namespace poseidon::mgpu
{

using CudaPeerDeviceInfo = runtime_api::communication::CudaDeviceInfo;
using CudaPeerProbeResult = runtime_api::communication::CudaTopology;

CudaPeerProbeResult probe_cuda_peer_access();

bool cuda_peer_probe_has_full_peer_access(
    const CudaPeerProbeResult &result, int required_devices = 0);

std::string dump_cuda_peer_probe(const CudaPeerProbeResult &result);
void dump_cuda_peer_probe(std::ostream &stream, const CudaPeerProbeResult &result);

std::string cuda_peer_probe_to_json(
    const CudaPeerProbeResult &result, int indent = 2);

}  // namespace poseidon::mgpu
