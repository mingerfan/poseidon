#pragma once

#include "poseidon/mgpu/ir/schedule.h"

namespace poseidon::mgpu
{

struct GpuCommCopyRequest
{
    ValueId source_id = 0;
    ValueId destination_id = 0;
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    int source_device = 0;
    int destination_device = 0;
};

class GpuComm
{
public:
    virtual ~GpuComm() = default;

    virtual void copy(const GpuCommCopyRequest &request) = 0;
};

class SameDeviceGpuComm final : public GpuComm
{
public:
    void copy(const GpuCommCopyRequest &request) override;
};

}  // namespace poseidon::mgpu
