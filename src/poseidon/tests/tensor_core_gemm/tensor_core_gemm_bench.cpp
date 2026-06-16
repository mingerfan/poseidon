#include "poseidon/gpu/gpu_tensor_core_gemm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct Options
{
    poseidon::gpu::GpuGemmShape shape{256, 256, 256};
    std::string mode = "all";
    int warmup = 3;
    int repeat = 10;
    int verify_samples = 8;
};

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

    void copy_from_host(const std::vector<T> &host)
    {
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

void print_usage(const char *program)
{
    std::cout << "Usage: " << program << " [options]\n"
              << "\n"
              << "Options:\n"
              << "  --size <n>              Square M=N=K size, default 256\n"
              << "  --m <n> --n <n> --k <n> Custom GEMM shape\n"
              << "  --mode <all|u8|u32-tensor|u32-cuda|u32>\n"
              << "  --warmup <n>            Warmup launches, default 3\n"
              << "  --repeat <n>            Timed launches, default 10\n"
              << "  --verify-samples <n>    Sampled CPU verification count, default 8\n";
}

Options parse_options(int argc, char **argv)
{
    Options opt;
    int custom_m = -1;
    int custom_n = -1;
    int custom_k = -1;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument(std::string(name) + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--size")
        {
            const int size = std::stoi(require_value("--size"));
            opt.shape = {size, size, size};
        }
        else if (arg == "--m")
        {
            custom_m = std::stoi(require_value("--m"));
        }
        else if (arg == "--n")
        {
            custom_n = std::stoi(require_value("--n"));
        }
        else if (arg == "--k")
        {
            custom_k = std::stoi(require_value("--k"));
        }
        else if (arg == "--mode")
        {
            opt.mode = require_value("--mode");
        }
        else if (arg == "--warmup")
        {
            opt.warmup = std::stoi(require_value("--warmup"));
        }
        else if (arg == "--repeat")
        {
            opt.repeat = std::stoi(require_value("--repeat"));
        }
        else if (arg == "--verify-samples")
        {
            opt.verify_samples = std::stoi(require_value("--verify-samples"));
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        else
        {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (custom_m != -1 || custom_n != -1 || custom_k != -1)
    {
        if (custom_m == -1 || custom_n == -1 || custom_k == -1)
        {
            throw std::invalid_argument("--m, --n, and --k must be provided together");
        }
        opt.shape = {custom_m, custom_n, custom_k};
    }

    if (opt.shape.m <= 0 || opt.shape.n <= 0 || opt.shape.k <= 0)
    {
        throw std::invalid_argument("matrix dimensions must be positive");
    }
    if (opt.warmup < 0 || opt.repeat <= 0 || opt.verify_samples < 0)
    {
        throw std::invalid_argument("invalid warmup/repeat/verify-samples");
    }
    if (opt.mode != "all" && opt.mode != "u8" && opt.mode != "u32-tensor" &&
        opt.mode != "u32-cuda" && opt.mode != "u32")
    {
        throw std::invalid_argument("--mode must be one of all, u8, u32-tensor, u32-cuda, u32");
    }

    return opt;
}

template <typename Launcher>
float time_cuda(const Launcher &launcher, int warmup, int repeat)
{
    for (int i = 0; i < warmup; ++i)
    {
        launcher();
    }
    cuda_check(cudaDeviceSynchronize(), "warmup synchronize");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < repeat; ++i)
    {
        launcher();
    }
    cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    return elapsed_ms / static_cast<float>(repeat);
}

std::string verify_status(bool ok, int samples)
{
    if (samples == 0)
    {
        return "not_requested";
    }
    return ok ? "pass" : "fail";
}

bool verify_u8_samples(
    const std::vector<std::int32_t> &c,
    poseidon::gpu::GpuGemmShape shape,
    int samples)
{
    for (int sample = 0; sample < samples; ++sample)
    {
        const std::uint32_t h = mix32(0xa5a5a5a5u + static_cast<std::uint32_t>(sample));
        const int row = sample == 0 ? 0 : static_cast<int>(h % shape.m);
        const int col = sample == 1 ? shape.n - 1 : static_cast<int>(mix32(h) % shape.n);
        std::int32_t expected = 0;
        for (int kk = 0; kk < shape.k; ++kk)
        {
            expected += static_cast<std::int32_t>(u8_a_value(row, kk)) *
                        static_cast<std::int32_t>(u8_b_value(kk, col));
        }
        if (c[static_cast<std::size_t>(row) * shape.n + col] != expected)
        {
            return false;
        }
    }
    return true;
}

bool verify_u32_samples(
    const std::vector<std::uint32_t> &c,
    poseidon::gpu::GpuGemmShape shape,
    int samples)
{
    for (int sample = 0; sample < samples; ++sample)
    {
        const std::uint32_t h = mix32(0x3c6ef372u + static_cast<std::uint32_t>(sample));
        const int row = sample == 0 ? 0 : static_cast<int>(h % shape.m);
        const int col = sample == 1 ? shape.n - 1 : static_cast<int>(mix32(h) % shape.n);
        std::uint32_t expected = 0;
        for (int kk = 0; kk < shape.k; ++kk)
        {
            expected += u32_a_value(row, kk) * u32_b_value(kk, col);
        }
        if (c[static_cast<std::size_t>(row) * shape.n + col] != expected)
        {
            return false;
        }
    }
    return true;
}

void print_result(
    poseidon::gpu::GpuGemmShape shape,
    const std::string &kernel,
    float avg_ms,
    int verify_samples,
    bool verify_ok,
    int u8_gemm_count)
{
    const double ops =
        2.0 * static_cast<double>(shape.m) * shape.n * shape.k;
    const double tops = (ops / (static_cast<double>(avg_ms) / 1000.0)) / 1.0e12;

    std::cout << shape.m << "," << shape.n << "," << shape.k << ","
              << kernel << "," << std::fixed << std::setprecision(6)
              << avg_ms << "," << std::setprecision(3) << tops << ","
              << u8_gemm_count << ","
              << verify_status(verify_ok, verify_samples) << "\n"
              << std::flush;
}

void bench_u8(const Options &opt)
{
    const auto shape = opt.shape;
    const std::size_t total_a = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t total_b = static_cast<std::size_t>(shape.n) * shape.k;
    const std::size_t total_c = static_cast<std::size_t>(shape.m) * shape.n;

    std::vector<std::uint8_t> a(total_a);
    std::vector<std::uint8_t> b(total_b);
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

    DeviceBuffer<std::uint8_t> d_a(total_a);
    DeviceBuffer<std::uint8_t> d_b(total_b);
    DeviceBuffer<std::int32_t> d_c(total_c);
    d_a.copy_from_host(a);
    d_b.copy_from_host(b);

    const float avg_ms = time_cuda(
        [&] {
            poseidon::gpu::launch_tensor_core_u8_gemm(
                d_a.data(),
                d_b.data(),
                d_c.data(),
                shape);
        },
        opt.warmup,
        opt.repeat);
    const auto c = d_c.copy_to_host();
    print_result(
        shape,
        "tensor_core_u8_gemm",
        avg_ms,
        opt.verify_samples,
        verify_u8_samples(c, shape, opt.verify_samples),
        1);
}

void bench_u32_tensor(const Options &opt)
{
    const auto shape = opt.shape;
    const std::size_t total_a = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t total_b = static_cast<std::size_t>(shape.n) * shape.k;
    const std::size_t total_c = static_cast<std::size_t>(shape.m) * shape.n;

    std::vector<std::uint32_t> a(total_a);
    std::vector<std::uint32_t> b(total_b);
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

    poseidon::gpu::launch_split_u32_to_u8_segments(
        d_a.data(),
        d_b.data(),
        d_a_segments.data(),
        d_b_segments.data(),
        shape);
    cuda_check(cudaDeviceSynchronize(), "split u32 inputs");

    const float avg_ms = time_cuda(
        [&] {
            poseidon::gpu::launch_tensor_core_u32_low32_gemm_from_segments(
                d_a_segments.data(),
                d_b_segments.data(),
                d_partial.data(),
                d_c.data(),
                shape);
        },
        opt.warmup,
        opt.repeat);
    const auto c = d_c.copy_to_host();
    print_result(
        shape,
        "tensor_core_u32_low32_gemm",
        avg_ms,
        opt.verify_samples,
        verify_u32_samples(c, shape, opt.verify_samples),
        poseidon::gpu::kTensorCoreU32Low32U8GemmCount);
}

void bench_u32_cuda(const Options &opt)
{
    const auto shape = opt.shape;
    const std::size_t total_a = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t total_b = static_cast<std::size_t>(shape.n) * shape.k;
    const std::size_t total_c = static_cast<std::size_t>(shape.m) * shape.n;

    std::vector<std::uint32_t> a(total_a);
    std::vector<std::uint32_t> b(total_b);
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

    DeviceBuffer<std::uint32_t> d_a(total_a);
    DeviceBuffer<std::uint32_t> d_b(total_b);
    DeviceBuffer<std::uint32_t> d_c(total_c);
    d_a.copy_from_host(a);
    d_b.copy_from_host(b);

    const float avg_ms = time_cuda(
        [&] {
            poseidon::gpu::launch_cuda_core_u32_gemm(
                d_a.data(),
                d_b.data(),
                d_c.data(),
                shape);
        },
        opt.warmup,
        opt.repeat);
    const auto c = d_c.copy_to_host();
    print_result(
        shape,
        "cuda_core_u32_gemm",
        avg_ms,
        opt.verify_samples,
        verify_u32_samples(c, shape, opt.verify_samples),
        0);
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options opt = parse_options(argc, argv);

        if (!poseidon::gpu::supports_tensor_core_integer_gemm())
        {
            std::cout << "skip: integer Tensor Core GEMM requires CUDA device SM 7.5+\n";
            return 77;
        }

        int device = 0;
        cudaDeviceProp prop{};
        cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        std::cerr << "device=" << device << " name=\"" << prop.name
                  << "\" sm=" << prop.major << "." << prop.minor
                  << " warmup=" << opt.warmup
                  << " repeat=" << opt.repeat << "\n";

        std::cout << "m,n,k,kernel,avg_ms,effective_tops,u8_gemm_count,verify\n";

        if (opt.mode == "all" || opt.mode == "u8")
        {
            bench_u8(opt);
        }
        if (opt.mode == "all" || opt.mode == "u32" || opt.mode == "u32-tensor")
        {
            bench_u32_tensor(opt);
        }
        if (opt.mode == "all" || opt.mode == "u32" || opt.mode == "u32-cuda")
        {
            bench_u32_cuda(opt);
        }

        cuda_check(cudaDeviceSynchronize(), "final cudaDeviceSynchronize");
        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
