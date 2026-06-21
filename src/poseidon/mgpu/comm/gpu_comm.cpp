#include "poseidon/mgpu/comm/gpu_comm.h"

#include <sstream>
#include <stdexcept>

namespace poseidon::mgpu
{

void SameDeviceGpuComm::copy(const GpuCommCopyRequest &request)
{
    if (request.source_device != request.destination_device)
    {
        std::ostringstream stream;
        stream << "cross-device copy %" << request.source_id << " from device "
               << request.source_device << " to device " << request.destination_device
               << " requires a multi-GPU communication backend";
        throw std::runtime_error(stream.str());
    }
}

}  // namespace poseidon::mgpu
