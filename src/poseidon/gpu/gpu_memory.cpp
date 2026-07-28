#include "poseidon/gpu/gpu_memory.h"

#include <mutex>
#include <unordered_map>

namespace poseidon::gpu
{
namespace
{

struct DeviceMemoryResourceOwners
{
    std::mutex mutex;
    std::unordered_map<rmm::mr::device_memory_resource *, std::weak_ptr<void>> owners;
};

DeviceMemoryResourceOwners &device_memory_resource_owners()
{
    static DeviceMemoryResourceOwners result;
    return result;
}

} // namespace

std::shared_ptr<void> device_memory_resource_owner(
    rmm::mr::device_memory_resource *resource)
{
    if (resource == nullptr)
    {
        return {};
    }

    auto &registry = device_memory_resource_owners();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.owners.find(resource);
    if (found == registry.owners.end())
    {
        return {};
    }

    auto owner = found->second.lock();
    if (!owner)
    {
        registry.owners.erase(found);
    }
    return owner;
}

void register_device_memory_resource_owner(
    rmm::mr::device_memory_resource *resource,
    const std::shared_ptr<void> &owner)
{
    if (resource == nullptr || !owner)
    {
        throw std::invalid_argument("GPU memory resource owner registration is invalid");
    }

    auto &registry = device_memory_resource_owners();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.owners.find(resource);
    if (found != registry.owners.end())
    {
        const auto existing = found->second.lock();
        if (existing && existing.get() != owner.get())
        {
            throw std::logic_error("GPU memory resource already has a different owner");
        }
    }
    registry.owners[resource] = owner;
}

void unregister_device_memory_resource_owner(
    rmm::mr::device_memory_resource *resource, const void *owner) noexcept
{
    if (resource == nullptr || owner == nullptr)
    {
        return;
    }

    auto &registry = device_memory_resource_owners();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.owners.find(resource);
    if (found == registry.owners.end())
    {
        return;
    }

    const auto existing = found->second.lock();
    if (!existing || existing.get() == owner)
    {
        registry.owners.erase(found);
    }
}

} // namespace poseidon::gpu
