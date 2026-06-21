#pragma once

#include "poseidon/mgpu/comm/gpu_object_copy.h"
#include "poseidon/mgpu/comm/topology.h"

namespace poseidon::mgpu
{

struct InterNodeCopyEndpoint
{
    int logical_device = 0;
    int node_id = 0;
    int local_device = 0;
};

struct InterNodeObjectCopyRequest
{
    ValueId source_id = 0;
    ValueId destination_id = 0;
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    InterNodeCopyEndpoint source;
    InterNodeCopyEndpoint destination;
    GpuObjectCopyRequest object_copy;
};

class InterNodeTransportBackend
{
public:
    virtual ~InterNodeTransportBackend() = default;

    virtual void copy_object(const InterNodeObjectCopyRequest &request) = 0;
};

class MissingInterNodeTransportBackend final : public InterNodeTransportBackend
{
public:
    void copy_object(const InterNodeObjectCopyRequest &request) override;
};

InterNodeObjectCopyRequest make_inter_node_object_copy_request(
    const MgpuCopyRoute &route, const GpuObjectCopyRequest &object_copy,
    const MgpuTopology &topology);

}  // namespace poseidon::mgpu
