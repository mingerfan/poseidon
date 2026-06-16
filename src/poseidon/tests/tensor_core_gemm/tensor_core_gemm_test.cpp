#include "poseidon/gpu/gpu_tensor_core_gemm.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void cuda_check(cudaError_t status, const char *what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::uint32_t mix32(std::uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

std::uint8_t u8_a_value(int row, int k)
{
    return static_cast<std::uint8_t>(
        mix32(0x1234u ^ static_cast<std::uint32_t>(row) * 0x9e3779b1u ^
              static_cast<std::uint32_t>(k) * 0x85ebca6bu) &
        0xffu);
}

std::uint8_t u8_b_value(int k, int col)
{
    return static_cast<std::uint8_t>(
        mix32(0x5678u ^ static_cast<std::uint32_t>(k) * 0xc2b2ae35u ^
              static_cast<std::uint32_t>(col) * 0x27d4eb2fu) &
        0xffu);
}

std::int8_t s8_a_value(int row, int k)
{
    return static_cast<std::int8_t>(static_cast<int>(u8_a_value(row, k)) - 128);
}

std::int8_t s8_b_value(int k, int col)
{
    return static_cast<std::int8_t>(static_cast<int>(u8_b_value(k, col)) - 128);
}

std::uint32_t u32_a_value(int row, int k)
{
    return mix32(0xabcd1234u ^ static_cast<std::uint32_t>(row) * 0x9e3779b1u ^
                 static_cast<std::uint32_t>(k) * 0x85ebca6bu);
}

std::uint32_t u32_b_value(int k, int col)
{
    return mix32(0x517cc1b7u ^ static_cast<std::uint32_t>(k) * 0xc2b2ae35u ^
                 static_cast<std::uint32_t>(col) * 0x27d4eb2fu);
}

template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(std::size_t count) : count_(count)
    {
        cuda_check(cudaMalloc(&ptr_, count_ * sizeof(T)), "cudaMalloc");
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer()
    {
        if (ptr_ != nullptr)
        {
            cudaFree(ptr_);
        }
    }

    T *data()
    {
        return ptr_;
    }

    const T *data() const
    {
        return ptr_;
    }

    void copy_from_host(const std::vector<T> &host)
    {
        if (host.size() != count_)
        {
            throw std::invalid_argument("copy_from_host size mismatch");
        }
        cuda_check(
            cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpyHostToDevice");
    }

    std::vector<T> copy_to_host() const
    {
        std::vector<T> host(count_);
        cuda_check(
            cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
            "cudaMemcpyDeviceToHost");
        return host;
    }

private:
    T *ptr_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
void expect_equal(
    const std::vector<T> &actual,
    const std::vector<T> &expected,
    int rows,
    int cols,
    const char *name)
{
    if (actual.size() != expected.size())
    {
        throw std::runtime_error(std::string(name) + ": size mismatch");
    }

    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        if (actual[i] != expected[i])
        {
            const int row = static_cast<int>(i / cols);
            const int col = static_cast<int>(i % cols);
            std::ostringstream oss;
            oss << name << ": mismatch at (" << row << "," << col
                << "), expected=" << +expected[i]
                << ", actual=" << +actual[i];
            throw std::runtime_error(oss.str());
        }
    }

    (void)rows;
}

void test_tensor_core_u8()
{
    const poseidon::gpu::GpuGemmShape shape{32, 48, 32};
    std::vector<std::uint8_t> a(static_cast<std::size_t>(shape.m) * shape.k);
    std::vector<std::uint8_t> b(static_cast<std::size_t>(shape.n) * shape.k);
    std::vector<std::int32_t> expected(
        static_cast<std::size_t>(shape.m) * shape.n);

    for (int row = 0; row < shape.m; ++row)
    {
        for (int kk = 0; kk < shape.k; ++kk)
        {
            a[static_cast<std::size_t>(row) * shape.k + kk] =
                u8_a_value(row, kk);
        }
    }
    for (int col = 0; col < shape.n; ++col)
    {
        for (int kk = 0; kk < shape.k; ++kk)
        {
            b[static_cast<std::size_t>(col) * shape.k + kk] =
                u8_b_value(kk, col);
        }
    }

    for (int row = 0; row < shape.m; ++row)
    {
        for (int col = 0; col < shape.n; ++col)
        {
            std::int32_t acc = 0;
            for (int kk = 0; kk < shape.k; ++kk)
            {
                acc += static_cast<std::int32_t>(u8_a_value(row, kk)) *
                       static_cast<std::int32_t>(u8_b_value(kk, col));
            }
            expected[static_cast<std::size_t>(row) * shape.n + col] = acc;
        }
    }

    DeviceBuffer<std::uint8_t> d_a(a.size());
    DeviceBuffer<std::uint8_t> d_b(b.size());
    DeviceBuffer<std::int32_t> d_c(expected.size());
    d_a.copy_from_host(a);
    d_b.copy_from_host(b);

    poseidon::gpu::launch_tensor_core_u8_gemm(
        d_a.data(),
        d_b.data(),
        d_c.data(),
        shape);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize u8");

    expect_equal(d_c.copy_to_host(), expected, shape.m, shape.n, "u8 tensor core");
    std::cout << "u8 tensor core GEMM passed\n";
}

void test_tensor_core_s8()
{
    const poseidon::gpu::GpuGemmShape shape{32, 16, 48};
    std::vector<std::int8_t> a(static_cast<std::size_t>(shape.m) * shape.k);
    std::vector<std::int8_t> b(static_cast<std::size_t>(shape.n) * shape.k);
    std::vector<std::int32_t> expected(
        static_cast<std::size_t>(shape.m) * shape.n);

    for (int row = 0; row < shape.m; ++row)
    {
        for (int kk = 0; kk < shape.k; ++kk)
        {
            a[static_cast<std::size_t>(row) * shape.k + kk] =
                s8_a_value(row, kk);
        }
    }
    for (int col = 0; col < shape.n; ++col)
    {
        for (int kk = 0; kk < shape.k; ++kk)
        {
            b[static_cast<std::size_t>(col) * shape.k + kk] =
                s8_b_value(kk, col);
        }
    }

    for (int row = 0; row < shape.m; ++row)
    {
        for (int col = 0; col < shape.n; ++col)
        {
            std::int32_t acc = 0;
            for (int kk = 0; kk < shape.k; ++kk)
            {
                acc += static_cast<std::int32_t>(s8_a_value(row, kk)) *
                       static_cast<std::int32_t>(s8_b_value(kk, col));
            }
            expected[static_cast<std::size_t>(row) * shape.n + col] = acc;
        }
    }

    DeviceBuffer<std::int8_t> d_a(a.size());
    DeviceBuffer<std::int8_t> d_b(b.size());
    DeviceBuffer<std::int32_t> d_c(expected.size());
    d_a.copy_from_host(a);
    d_b.copy_from_host(b);

    poseidon::gpu::launch_tensor_core_s8_gemm(
        d_a.data(),
        d_b.data(),
        d_c.data(),
        shape);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize s8");

    expect_equal(d_c.copy_to_host(), expected, shape.m, shape.n, "s8 tensor core");
    std::cout << "s8 tensor core GEMM passed\n";
}

void test_tensor_core_u32_low32()
{
    const poseidon::gpu::GpuGemmShape shape{16, 32, 16};
    const std::size_t total_a = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t total_b = static_cast<std::size_t>(shape.n) * shape.k;
    const std::size_t total_c = static_cast<std::size_t>(shape.m) * shape.n;

    std::vector<std::uint32_t> a(total_a);
    std::vector<std::uint32_t> b(total_b);
    std::vector<std::uint32_t> expected(total_c);

    for (int row = 0; row < shape.m; ++row)
    {
        for (int kk = 0; kk < shape.k; ++kk)
        {
            a[static_cast<std::size_t>(row) * shape.k + kk] =
                u32_a_value(row, kk);
        }
    }
    for (int col = 0; col < shape.n; ++col)
    {
        for (int kk = 0; kk < shape.k; ++kk)
        {
            b[static_cast<std::size_t>(col) * shape.k + kk] =
                u32_b_value(kk, col);
        }
    }

    for (int row = 0; row < shape.m; ++row)
    {
        for (int col = 0; col < shape.n; ++col)
        {
            std::uint32_t acc = 0;
            for (int kk = 0; kk < shape.k; ++kk)
            {
                acc += u32_a_value(row, kk) * u32_b_value(kk, col);
            }
            expected[static_cast<std::size_t>(row) * shape.n + col] = acc;
        }
    }

    const auto sizes = poseidon::gpu::tensor_core_u32_workspace_sizes(shape);
    DeviceBuffer<std::uint32_t> d_a(total_a);
    DeviceBuffer<std::uint32_t> d_b(total_b);
    DeviceBuffer<std::uint32_t> d_c(total_c);
    DeviceBuffer<std::uint8_t> d_a_segments(
        sizes.a_segments_bytes / sizeof(std::uint8_t));
    DeviceBuffer<std::uint8_t> d_b_segments(
        sizes.b_segments_col_major_bytes / sizeof(std::uint8_t));
    DeviceBuffer<std::int32_t> d_partial(
        sizes.partial_bytes / sizeof(std::int32_t));

    d_a.copy_from_host(a);
    d_b.copy_from_host(b);

    poseidon::gpu::TensorCoreU32GemmWorkspace workspace{
        d_a_segments.data(),
        d_b_segments.data(),
        d_partial.data()};

    poseidon::gpu::launch_tensor_core_u32_low32_gemm(
        d_a.data(),
        d_b.data(),
        d_c.data(),
        shape,
        workspace);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize u32 tensor core");

    expect_equal(
        d_c.copy_to_host(),
        expected,
        shape.m,
        shape.n,
        "u32 tensor core low32");
    std::cout << "u32 tensor core low32 GEMM passed\n";
}

void test_cuda_core_fallbacks()
{
    const poseidon::gpu::GpuGemmShape shape{17, 19, 23};
    const std::size_t total_a = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t total_b = static_cast<std::size_t>(shape.n) * shape.k;
    const std::size_t total_c = static_cast<std::size_t>(shape.m) * shape.n;

    std::vector<std::uint32_t> u32_a(total_a);
    std::vector<std::uint32_t> u32_b(total_b);
    std::vector<std::uint32_t> u32_expected(total_c);
    std::vector<std::int8_t> s8_a(total_a);
    std::vector<std::int8_t> s8_b(total_b);
    std::vector<std::int32_t> s8_expected(total_c);

    for (int row = 0; row < shape.m; ++row)
    {
        for (int kk = 0; kk < shape.k; ++kk)
        {
            u32_a[static_cast<std::size_t>(row) * shape.k + kk] =
                u32_a_value(row, kk);
            s8_a[static_cast<std::size_t>(row) * shape.k + kk] =
                s8_a_value(row, kk);
        }
    }
    for (int col = 0; col < shape.n; ++col)
    {
        for (int kk = 0; kk < shape.k; ++kk)
        {
            u32_b[static_cast<std::size_t>(col) * shape.k + kk] =
                u32_b_value(kk, col);
            s8_b[static_cast<std::size_t>(col) * shape.k + kk] =
                s8_b_value(kk, col);
        }
    }

    for (int row = 0; row < shape.m; ++row)
    {
        for (int col = 0; col < shape.n; ++col)
        {
            std::uint32_t u32_acc = 0;
            std::int32_t s8_acc = 0;
            for (int kk = 0; kk < shape.k; ++kk)
            {
                u32_acc += u32_a_value(row, kk) * u32_b_value(kk, col);
                s8_acc += static_cast<std::int32_t>(s8_a_value(row, kk)) *
                          static_cast<std::int32_t>(s8_b_value(kk, col));
            }
            const auto out_idx = static_cast<std::size_t>(row) * shape.n + col;
            u32_expected[out_idx] = u32_acc;
            s8_expected[out_idx] = s8_acc;
        }
    }

    DeviceBuffer<std::uint32_t> d_u32_a(total_a);
    DeviceBuffer<std::uint32_t> d_u32_b(total_b);
    DeviceBuffer<std::uint32_t> d_u32_c(total_c);
    DeviceBuffer<std::int8_t> d_s8_a(total_a);
    DeviceBuffer<std::int8_t> d_s8_b(total_b);
    DeviceBuffer<std::int32_t> d_s8_c(total_c);

    d_u32_a.copy_from_host(u32_a);
    d_u32_b.copy_from_host(u32_b);
    d_s8_a.copy_from_host(s8_a);
    d_s8_b.copy_from_host(s8_b);

    poseidon::gpu::launch_cuda_core_u32_gemm(
        d_u32_a.data(),
        d_u32_b.data(),
        d_u32_c.data(),
        shape);
    poseidon::gpu::launch_cuda_core_s8_gemm(
        d_s8_a.data(),
        d_s8_b.data(),
        d_s8_c.data(),
        shape);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize fallback");

    expect_equal(
        d_u32_c.copy_to_host(),
        u32_expected,
        shape.m,
        shape.n,
        "u32 cuda core");
    expect_equal(
        d_s8_c.copy_to_host(),
        s8_expected,
        shape.m,
        shape.n,
        "s8 cuda core");
    std::cout << "CUDA-core fallback GEMMs passed\n";
}

}  // namespace

int main()
{
    try
    {
        if (!poseidon::gpu::supports_tensor_core_integer_gemm())
        {
            std::cout << "skip: integer Tensor Core GEMM requires CUDA device SM 7.5+\n";
            return 77;
        }

        test_tensor_core_u8();
        test_tensor_core_s8();
        test_tensor_core_u32_low32();
        test_cuda_core_fallbacks();
        std::cout << "all tensor core GEMM tests passed\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
