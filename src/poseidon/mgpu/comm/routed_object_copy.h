#pragma once

#include "poseidon/mgpu/comm/inter_node_transport.h"
#include "poseidon/mgpu/comm/materialized_gpu_comm.h"

namespace poseidon::mgpu
{

class RoutedGpuObjectCopyBackend
{
public:
    RoutedGpuObjectCopyBackend(
        MgpuTopology topology, GpuObjectCopyBackend &local_backend,
        InterNodeTransportBackend &inter_node_backend);

    void copy_object(
        const MgpuCopyRoute &route, const GpuObjectCopyRequest &request);

private:
    MgpuTopology topology_;
    GpuObjectCopyBackend &local_backend_;
    InterNodeTransportBackend &inter_node_backend_;
};

}  // namespace poseidon::mgpu
