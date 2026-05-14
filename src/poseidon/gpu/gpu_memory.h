#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace poseidon
{
namespace gpu
{

/**
 * @brief GPU residue word type.
 *
 * The GPU backend targets small RNS primes below 32 bits.
 * Therefore GPU-side ciphertext/plaintext/key residues are stored as uint32_t.
 *
 * Multiplication kernels may still use uint64_t as intermediate type.
 */
using GpuWord = std::uint32_t;
using GpuWide = std::uint64_t;

/**
 * @brief Lightweight GPU memory owner.
 *
 * This class is intended to own one contiguous device buffer on one GPU.
 *
 * Current stage:
 * - Only defines the framework-level interface.
 * - Real CUDA allocation/free is TODO.
 */
template <typename T>
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

    DeviceVector(DeviceVector &&other) noexcept
    {
        move_from(std::move(other));
    }

    DeviceVector &operator=(DeviceVector &&other) noexcept
    {
        if (this != &other)
        {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    ~DeviceVector()
    {
        release();
    }

    /**
     * @brief Allocate device memory.
     *
     * TODO:
     * - cudaSetDevice(device_id)
     * - cudaMalloc(&ptr_, size * sizeof(T))
     */
    void allocate(std::size_t size, int device_id)
    {
        release();

        // TODO: real GPU allocation.
        ptr_ = nullptr;
        size_ = size;
        device_id_ = device_id;
    }

    /**
     * @brief Release device memory.
     *
     * TODO:
     * - cudaSetDevice(device_id_)
     * - cudaFree(ptr_)
     */
    void release()
    {
        // TODO: real GPU free.
        ptr_ = nullptr;
        size_ = 0;
        device_id_ = 0;
    }

    T *data()
    {
        return ptr_;
    }

    const T *data() const
    {
        return ptr_;
    }

    std::size_t size() const
    {
        return size_;
    }

    bool empty() const
    {
        return size_ == 0;
    }

    int device_id() const
    {
        return device_id_;
    }

private:
    void move_from(DeviceVector &&other) noexcept
    {
        ptr_ = other.ptr_;
        size_ = other.size_;
        device_id_ = other.device_id_;

        other.ptr_ = nullptr;
        other.size_ = 0;
        other.device_id_ = 0;
    }

private:
    T *ptr_ = nullptr;
    std::size_t size_ = 0;
    int device_id_ = 0;
};

/**
 * @brief Physical GPU memory block.
 *
 * This object only knows:
 * - which GPU device it belongs to;
 * - how large the buffer is;
 * - where the device pointer is.
 *
 * It does not know FHE semantics such as c0/c1, RNS limb range,
 * coefficient range, plaintext/key layout, or parameter level.
 */
struct GpuFieldData
{
    int device_id = 0;
    DeviceVector<GpuWord> buffer;

    GpuFieldData() = default;

    GpuFieldData(int dev, std::size_t elem_count)
        : device_id(dev), buffer(elem_count, dev)
    {}

    GpuWord *data()
    {
        return buffer.data();
    }

    const GpuWord *data() const
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