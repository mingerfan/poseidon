#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <unordered_map>

namespace poseidon::mgpu
{

struct MgpuObjectMetadata
{
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    int device_id = 0;
};

class MgpuObjectStore
{
public:
    void clear() noexcept;
    std::size_t size() const noexcept;
    bool empty() const noexcept;
    bool contains(ValueId id) const;

    const MgpuObjectMetadata *find(ValueId id) const;
    const MgpuObjectMetadata &at(ValueId id) const;

    void define(ValueId id, MgpuValueKind kind, int device_id);

private:
    std::unordered_map<ValueId, MgpuObjectMetadata> objects_;
};

}  // namespace poseidon::mgpu
