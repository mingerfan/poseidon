#pragma once

#include "poseidon/mgpu/comm/topology.h"

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace poseidon::mgpu
{

struct GpuCommCopyRequest
{
    ValueId source_id = 0;
    ValueId destination_id = 0;
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    int source_device = 0;
    int destination_device = 0;
    std::shared_ptr<void> source_object;
};

struct GpuObjectBufferCopy
{
    const void *source = nullptr;
    void *destination = nullptr;
    std::size_t bytes = 0;
    int source_device = 0;
    int destination_device = 0;
};

struct GpuObjectCopyRequest
{
    ValueId source_id = 0;
    ValueId destination_id = 0;
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    std::vector<GpuObjectBufferCopy> buffers;
};

struct GpuObjectCopyValidationResult
{
    std::vector<std::string> errors;

    bool ok() const noexcept
    {
        return errors.empty();
    }

    std::string format_errors() const;
};

GpuObjectCopyValidationResult validate_full_object_copy_request(
    const GpuObjectCopyRequest &request);

class GpuComm
{
public:
    virtual ~GpuComm() = default;

    virtual std::shared_ptr<void> copy(const GpuCommCopyRequest &request) = 0;
};

class SameDeviceGpuComm final : public GpuComm
{
public:
    std::shared_ptr<void> copy(const GpuCommCopyRequest &request) override;
};

struct MaterializedGpuObjectCopy
{
    std::shared_ptr<void> destination_object;
    GpuObjectCopyRequest object_copy;
};

class GpuObjectCopyMaterializer
{
public:
    virtual ~GpuObjectCopyMaterializer() = default;

    virtual MaterializedGpuObjectCopy materialize_copy(
        const GpuCommCopyRequest &request) = 0;
};

class GpuObjectCopyBackend
{
public:
    virtual ~GpuObjectCopyBackend() = default;

    virtual void copy_object(const GpuObjectCopyRequest &request) = 0;
};

class MaterializedGpuComm final : public GpuComm
{
public:
    MaterializedGpuComm(
        GpuObjectCopyMaterializer &materializer, GpuObjectCopyBackend &backend);

    std::shared_ptr<void> copy(const GpuCommCopyRequest &request) override;

private:
    GpuObjectCopyMaterializer &materializer_;
    GpuObjectCopyBackend &backend_;
};

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

class PlannedMaterializedGpuComm final : public GpuComm
{
public:
    PlannedMaterializedGpuComm(
        const MgpuCommunicationPlan &plan, MgpuTopology topology,
        GpuObjectCopyMaterializer &materializer,
        GpuObjectCopyBackend &local_backend,
        InterNodeTransportBackend &inter_node_backend);

    std::shared_ptr<void> copy(const GpuCommCopyRequest &request) override;

private:
    std::vector<MgpuCopyRoute> routes_;
    std::map<std::pair<ValueId, ValueId>, std::size_t> route_indices_;
    MgpuTopology topology_;
    GpuObjectCopyMaterializer &materializer_;
    GpuObjectCopyBackend &local_backend_;
    InterNodeTransportBackend &inter_node_backend_;
};

}  // namespace poseidon::mgpu
