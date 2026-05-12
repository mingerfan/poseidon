#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace poseidon
{
namespace gpu
{

/**
 * @brief Lightweight device vector placeholder.
 *
 * This class is intended to own a contiguous memory region on one GPU device.
 * The actual CUDA allocation/free logic should be implemented later with
 * cudaSetDevice / cudaMalloc / cudaFree or by reusing Cheddar's DeviceVector.
 *
 * Current stage:
 * - Only defines the interface.
 * - Allocation is left as TODO.
 */
class DeviceVector
{
public:
    DeviceVector() = default;

    DeviceVector(std::size_t size, int device_id)
    {
        allocate(size, device_id);
    }

    DeviceVector(const DeviceVector &) = delete;
    DeviceVector &operator=(const DeviceVector &) = delete;

    DeviceVector(DeviceVector &&) = default;
    DeviceVector &operator=(DeviceVector &&) = default;

    ~DeviceVector()
    {
        release();
    }

    void allocate(std::size_t size, int device_id)
    {
        // TODO:
        // cudaSetDevice(device_id);
        // cudaMalloc(&ptr_, size * sizeof(std::uint64_t));
        //
        // For now, only record metadata.
        size_ = size;
        device_id_ = device_id;
        ptr_ = nullptr;
    }

    void release()
    {
        // TODO:
        // if (ptr_ != nullptr) {
        //     cudaSetDevice(device_id_);
        //     cudaFree(ptr_);
        // }
        ptr_ = nullptr;
        size_ = 0;
    }

    std::uint64_t *data()
    {
        return ptr_;
    }

    const std::uint64_t *data() const
    {
        return ptr_;
    }

    std::size_t size() const
    {
        return size_;
    }

    int device_id() const
    {
        return device_id_;
    }

    bool empty() const
    {
        return size_ == 0;
    }

private:
    std::uint64_t *ptr_ = nullptr;
    std::size_t size_ = 0;
    int device_id_ = 0;
};

/**
 * @brief One physical GPU memory block.
 *
 * This object does not know whether the memory stores c0, c1, q0-q3,
 * or any FHE semantic region. It only owns a memory buffer on one GPU.
 */
struct GpuFieldData
{
    int device_id = 0;
    DeviceVector buffer;

    GpuFieldData() = default;

    GpuFieldData(int dev, std::size_t elem_count)
        : device_id(dev), buffer(elem_count, dev)
    {}

    std::uint64_t *data()
    {
        return buffer.data();
    }

    const std::uint64_t *data() const
    {
        return buffer.data();
    }

    std::size_t size() const
    {
        return buffer.size();
    }

    bool empty() const
    {
        return buffer.empty();
    }
};

}  // namespace gpu
}  // namespace poseidon