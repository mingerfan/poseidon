#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

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

}  // namespace poseidon::mgpu
