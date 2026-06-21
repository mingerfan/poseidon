#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <memory>

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

}  // namespace poseidon::mgpu
