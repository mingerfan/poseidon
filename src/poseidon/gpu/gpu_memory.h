#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

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

inline void gpu_check_cuda(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(status));
    }
}

/**
 * @brief Lightweight GPU memory owner.
 *
 * This class is intended to own one contiguous device buffer on one GPU.
 *
 * Memory is allocated through RAPIDS Memory Manager (RMM). The process should
 * install/configure the desired RMM device resource once during GPU backend
 * initialization. DeviceVector intentionally uses the current device resource
 * instead of constructing a pool locally.
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
            try
            {
                release();
            }
            catch (...)
            {}
            move_from(std::move(other));
        }
        return *this;
    }

    ~DeviceVector()
    {
        try
        {
            release();
        }
        catch (...)
        {}
    }

    void allocate(std::size_t size, int device_id)
    {
        release();

        if (size != 0 && size > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::overflow_error("DeviceVector allocation size overflow");
        }
        size_ = size;
        bytes_ = size * sizeof(T);
        device_id_ = device_id;

        if (size_ == 0)
        {
            return;
        }

        gpu_check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
        ptr_ = static_cast<T *>(
            rmm::mr::get_current_device_resource()->allocate(
                bytes_,
                rmm::cuda_stream_default));
    }

    void copy_from_host(const T *src, std::size_t count)
    {
        if (count > size_)
        {
            throw std::out_of_range("DeviceVector::copy_from_host count exceeds allocation");
        }
        if (count != 0 && src == nullptr)
        {
            throw std::invalid_argument("DeviceVector::copy_from_host source is null");
        }
        if (count == 0)
        {
            return;
        }

        gpu_check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
        gpu_check_cuda(
            cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpyHostToDevice");
    }

    void copy_to_host(T *dst, std::size_t count) const
    {
        if (count > size_)
        {
            throw std::out_of_range("DeviceVector::copy_to_host count exceeds allocation");
        }
        if (count != 0 && dst == nullptr)
        {
            throw std::invalid_argument("DeviceVector::copy_to_host destination is null");
        }
        if (count == 0)
        {
            return;
        }

        gpu_check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
        gpu_check_cuda(
            cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost),
            "cudaMemcpyDeviceToHost");
    }

    void fill_zero()
    {
        if (size_ == 0)
        {
            return;
        }

        gpu_check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
        gpu_check_cuda(cudaMemset(ptr_, 0, bytes_), "cudaMemset");
    }

    void release()
    {
        if (ptr_ != nullptr)
        {
            gpu_check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
            rmm::mr::get_current_device_resource()->deallocate(
                ptr_,
                bytes_,
                rmm::cuda_stream_default);
        }
        ptr_ = nullptr;
        size_ = 0;
        bytes_ = 0;
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
        bytes_ = other.bytes_;
        device_id_ = other.device_id_;

        other.ptr_ = nullptr;
        other.size_ = 0;
        other.bytes_ = 0;
        other.device_id_ = 0;
    }

private:
    T *ptr_ = nullptr;
    std::size_t size_ = 0;
    std::size_t bytes_ = 0;
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
