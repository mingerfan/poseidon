#include "poseidon/mgpu/runtime/object_store.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{

void MgpuObjectStore::clear() noexcept
{
    objects_.clear();
}

std::size_t MgpuObjectStore::size() const noexcept
{
    return objects_.size();
}

bool MgpuObjectStore::empty() const noexcept
{
    return objects_.empty();
}

bool MgpuObjectStore::contains(ValueId id) const
{
    return objects_.find(id) != objects_.end();
}

const MgpuObjectMetadata *MgpuObjectStore::find(ValueId id) const
{
    const auto iter = objects_.find(id);
    if (iter == objects_.end())
    {
        return nullptr;
    }
    return &iter->second;
}

const MgpuObjectMetadata &MgpuObjectStore::at(ValueId id) const
{
    const MgpuObjectMetadata *metadata = find(id);
    if (metadata == nullptr)
    {
        std::ostringstream stream;
        stream << "unknown object %" << id;
        throw std::out_of_range(stream.str());
    }
    return *metadata;
}

void MgpuObjectStore::define(ValueId id, MgpuValueKind kind, int device_id)
{
    define(id, kind, device_id, {});
}

void MgpuObjectStore::define(
    ValueId id, MgpuValueKind kind, int device_id, std::shared_ptr<void> object)
{
    if (id == 0)
    {
        throw std::invalid_argument("object value id 0 is reserved");
    }

    const auto [iter, inserted] =
        objects_.emplace(id, MgpuObjectMetadata{ kind, device_id, std::move(object) });
    if (!inserted)
    {
        std::ostringstream stream;
        stream << "duplicate object %" << id;
        throw std::invalid_argument(stream.str());
    }
}

void MgpuObjectStore::set_object(ValueId id, std::shared_ptr<void> object)
{
    const auto iter = objects_.find(id);
    if (iter == objects_.end())
    {
        std::ostringstream stream;
        stream << "unknown object %" << id;
        throw std::out_of_range(stream.str());
    }
    iter->second.object = std::move(object);
}

bool MgpuObjectStore::has_object(ValueId id) const
{
    const MgpuObjectMetadata *metadata = find(id);
    return metadata != nullptr && metadata->object != nullptr;
}

}  // namespace poseidon::mgpu
