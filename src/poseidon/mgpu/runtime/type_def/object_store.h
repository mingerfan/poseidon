#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <memory>
#include <unordered_map>

namespace poseidon::mgpu
{

struct MgpuObjectMetadata
{
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    int device_id = 0;
    std::shared_ptr<void> object;
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
    void define(
        ValueId id, MgpuValueKind kind, int device_id, std::shared_ptr<void> object);
    void set_object(ValueId id, std::shared_ptr<void> object);
    bool has_object(ValueId id) const;

    template <typename T>
    std::shared_ptr<T> object_as(ValueId id) const
    {
        return std::static_pointer_cast<T>(at(id).object);
    }

private:
    std::unordered_map<ValueId, MgpuObjectMetadata> objects_;
};

}  // namespace poseidon::mgpu
