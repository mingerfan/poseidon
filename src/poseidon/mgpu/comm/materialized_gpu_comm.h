#pragma once

#include "poseidon/mgpu/comm/gpu_comm.h"
#include "poseidon/mgpu/comm/gpu_object_copy.h"

#include <memory>

namespace poseidon::mgpu
{

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

}  // namespace poseidon::mgpu
