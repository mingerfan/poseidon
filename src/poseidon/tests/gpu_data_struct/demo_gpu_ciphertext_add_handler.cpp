#include "poseidon/ckks_encoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_elementwise_handler.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_ntt_handler.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_tensor_core_gemm.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/gpu/kernels/gpu_keyswitch_kernels.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"
#include "poseidon/basics/randomgen.h"
#include "poseidon/util/rns_tool_qp.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr int kSkip = 77;
constexpr bool kRunCorrectnessChecks = false;
constexpr bool kRunOperationTimingSummary = true;
constexpr bool kRunNttTimingSummary = false;
constexpr const char *kNttAlgorithmEnv = "POSEIDON_NTT_ALGO";
constexpr const char *kNttFusionStagesEnv = "POSEIDON_NTT_FUSION_STAGES";
constexpr const char *kNttFusedMatrixStagesEnv =
    "POSEIDON_NTT_FUSED_MATRIX_STAGES";
constexpr const char *kNttFusedMatrixCacheDirEnv =
    "POSEIDON_NTT_FUSED_MATRIX_CACHE_DIR";
constexpr const char *kNttFusedMatrixProgressEnv =
    "POSEIDON_NTT_FUSED_MATRIX_PROGRESS";
constexpr const char *kNttFusedMatrixFp64TablesEnv =
    "POSEIDON_NTT_FUSED_MATRIX_FP64_TABLES";
constexpr const char *kNttFusedMatrixMaxLevelsEnv =
    "POSEIDON_NTT_FUSED_MATRIX_MAX_LEVELS";
constexpr const char *kDemoSkipTensorNttEnv =
    "POSEIDON_DEMO_SKIP_TENSOR_NTT";
constexpr const char *kDemoSkipTensorFp64NttEnv =
    "POSEIDON_DEMO_SKIP_TENSOR_FP64_NTT";
constexpr const char *kKeySwitchFuseDecompQEnv =
    "POSEIDON_KEYSWITCH_FUSE_DECOMP_Q";
constexpr const char *kKeySwitchFuseModupNttHeadEnv =
    "POSEIDON_KEYSWITCH_FUSE_MODUP_NTT_HEAD";
constexpr const char *kKeySwitchBconvRowTiledEnv =
    "POSEIDON_KEYSWITCH_BCONV_ROW_TILED";
constexpr const char *kKeySwitchBconvRowTiled8Env =
    "POSEIDON_KEYSWITCH_BCONV_ROW_TILED_8";
constexpr const char *kKeySwitchPAccumAllDnumEnv =
    "POSEIDON_KEYSWITCH_PACCUM_ALL_DNUM";
constexpr const char *kKeySwitchPAccumFinalTailEnv =
    "POSEIDON_KEYSWITCH_PACCUM_FINAL_TAIL";
constexpr const char *kDefaultNttFusedMatrixCacheDir =
    "/tmp/poseidon_ntt_tam_cache";

constexpr poseidon::prng_seed_type kBenchmarkPrngSeed{
    0x2f4a8d3c1b765e90ULL,
    0x6c91e2b47a035fd8ULL,
    0x13579bdf2468ace0ULL,
    0xfedcba9876543210ULL,
    0x0badf00d5eed1234ULL,
    0x89abcdef01234567ULL,
    0x55aa55aa33cc33ccULL,
    0xc001d00dcafebeefULL};

class BenchmarkPrngFactory final : public poseidon::UniformRandomGeneratorFactory
{
public:
    explicit BenchmarkPrngFactory(poseidon::prng_seed_type seed)
        : poseidon::UniformRandomGeneratorFactory(seed), base_seed_(seed)
    {
    }

protected:
    POSEIDON_NODISCARD auto create_impl(poseidon::prng_seed_type seed)
        -> std::shared_ptr<poseidon::UniformRandomGenerator> override
    {
        const std::uint64_t stream = stream_index_++;
        for (std::size_t i = 0; i < seed.size(); ++i)
        {
            seed[i] = base_seed_[i] ^
                      (0x9e3779b97f4a7c15ULL *
                       (stream + static_cast<std::uint64_t>(i) + 1));
        }
        return std::make_shared<poseidon::Blake2xbPRNG>(seed);
    }

private:
    poseidon::prng_seed_type base_seed_;
    std::uint64_t stream_index_ = 0;
};

void use_benchmark_randomness(poseidon::PoseidonContext &context)
{
    context.set_random_generator(
        std::make_shared<BenchmarkPrngFactory>(kBenchmarkPrngSeed));
}

class NvtxRange
{
public:
    explicit NvtxRange(const char *name)
    {
        nvtxRangePushA(name);
    }

    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;

    ~NvtxRange()
    {
        nvtxRangePop();
    }
};

bool env_flag_enabled(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return false;
    }

    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" &&
           text != "off" && text != "OFF";
}

std::size_t env_size_or(const char *name, std::size_t fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return fallback;
    }

    return static_cast<std::size_t>(std::stoull(value));
}

class ScopedEnvironmentValue
{
public:
    ScopedEnvironmentValue(const char *name, const char *value)
        : name_(name)
    {
        const char *previous = std::getenv(name_.c_str());
        if (previous != nullptr)
        {
            previous_ = previous;
        }
        if (::setenv(name_.c_str(), value, 1) != 0)
        {
            throw std::runtime_error("failed to set environment variable " + name_);
        }
    }

    ScopedEnvironmentValue(const ScopedEnvironmentValue &) = delete;
    ScopedEnvironmentValue &operator=(const ScopedEnvironmentValue &) = delete;

    ~ScopedEnvironmentValue()
    {
        if (previous_.has_value())
        {
            (void)::setenv(name_.c_str(), previous_->c_str(), 1);
        }
        else
        {
            (void)::unsetenv(name_.c_str());
        }
    }

private:
    std::string name_;
    std::optional<std::string> previous_;
};

class ScopedDefaultEnvironmentValue
{
public:
    ScopedDefaultEnvironmentValue(const char *name, const char *value)
    {
        const char *previous = std::getenv(name);
        if (previous == nullptr || previous[0] == '\0')
        {
            scoped_.emplace(name, value);
        }
    }

private:
    std::optional<ScopedEnvironmentValue> scoped_;
};

std::string env_string_or(const char *name, const char *fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return fallback;
    }
    return value;
}

void print_progress_bar(
    const std::string &label,
    std::size_t current,
    std::size_t total,
    const std::string &detail)
{
    constexpr std::size_t width = 28;
    const std::size_t clamped = std::min(current, total);
    const std::size_t filled = total == 0 ? width : clamped * width / total;

    std::cout << "\r" << label << " [";
    for (std::size_t i = 0; i < width; ++i)
    {
        std::cout << (i < filled ? '#' : '.');
    }
    std::cout << "] " << clamped << "/" << total << " " << detail
              << std::flush;
    if (clamped == total)
    {
        std::cout << "\n";
    }
}

std::vector<std::size_t> env_size_list_or_empty(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return {};
    }

    std::string text(value);
    std::replace(text.begin(), text.end(), ',', ' ');

    std::vector<std::size_t> result;
    std::istringstream stream(text);
    std::size_t item = 0;
    while (stream >> item)
    {
        if (item == 0)
        {
            throw std::invalid_argument(
                std::string(name) + " contains zero q_count");
        }
        result.push_back(item);
    }

    if (result.empty())
    {
        throw std::invalid_argument(
            std::string(name) + " did not contain any q_count values");
    }
    return result;
}

struct TimingResult
{
    double cpu_avg_ms = 0.0;
    double gpu_wall_avg_ms = 0.0;
    double gpu_event_avg_ms = 0.0;
    double wall_speedup = 0.0;
    double event_speedup = 0.0;
};

struct OperationTimingRow
{
    std::string operation;
    TimingResult timing;
    std::string correct;
    bool has_cpu_gpu_comparison = true;
};

struct GpuSampleStats
{
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double median_ms = 0.0;
    double p90_ms = 0.0;
    double max_ms = 0.0;
};

GpuSampleStats summarize_gpu_samples(const std::vector<double> &samples);

struct NttTimingRow
{
    std::string operation;
    std::string mode;
    GpuSampleStats timing;
    std::string correct;
    bool detail = false;
};

struct SweepBenchmarkRow
{
    std::string experiment;
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::string operation;
    std::size_t gpu_work_items = 0;
    std::size_t gpu_kernel_launches = 0;
    double cpu_avg_ms = 0.0;
    double gpu_wall_avg_ms = 0.0;
    double gpu_event_avg_ms = 0.0;
    double wall_speedup = 0.0;
    double event_speedup = 0.0;
};

class RmmPoolScope
{
public:
    explicit RmmPoolScope(int device_id)
        : device_id_(device_id), pool_(&upstream_, 1 << 20, std::nullopt)
    {
        auto status = cudaSetDevice(device_id_);
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("cudaSetDevice failed: ") + cudaGetErrorString(status));
        }

        previous_ = rmm::mr::get_current_device_resource();
        rmm::mr::set_current_device_resource(&pool_);
    }

    RmmPoolScope(const RmmPoolScope &) = delete;
    RmmPoolScope &operator=(const RmmPoolScope &) = delete;

    ~RmmPoolScope()
    {
        try
        {
            cudaSetDevice(device_id_);
            rmm::mr::set_current_device_resource(previous_);
        }
        catch (...)
        {}
    }

private:
    int device_id_ = 0;
    rmm::mr::cuda_memory_resource upstream_;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_;
    rmm::mr::device_memory_resource *previous_ = nullptr;
};

int check_cuda_runtime()
{
    int device_count = 0;
    auto status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess)
    {
        std::cerr << "[SKIP] CUDA runtime is unavailable: "
                  << cudaGetErrorString(status) << "\n";
        return kSkip;
    }
    if (device_count == 0)
    {
        std::cerr << "[SKIP] No CUDA device is visible\n";
        return kSkip;
    }
    return EXIT_SUCCESS;
}

poseidon::ParametersLiteral make_demo_parameters()
{
    poseidon::ParametersLiteral parms(
        CKKS,
        /*log_n=*/16,
        /*log_slots=*/15,
        /*log_scale=*/25,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        poseidon::Modulus(0),
        std::vector<poseidon::Modulus>{},
        std::vector<poseidon::Modulus>{},
        poseidon::sec_level_type::none);

    parms.set_log_modulus(
        std::vector<std::uint32_t>(8, 30),
        std::vector<std::uint32_t>(2, 30));
    return parms;
}

int log2_degree(std::size_t degree)
{
    if (degree < 2 || (degree & (degree - 1)) != 0)
    {
        throw std::invalid_argument("degree must be a power of two");
    }

    int result = 0;
    while (degree > 1)
    {
        degree >>= 1;
        ++result;
    }
    return result;
}

poseidon::ParametersLiteral make_benchmark_parameters(
    std::size_t degree,
    std::size_t q_count,
    std::size_t p_count = 0)
{
    const int log_n = log2_degree(degree);
    poseidon::ParametersLiteral parms(
        CKKS,
        log_n,
        log_n - 1,
        /*log_scale=*/25,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        poseidon::Modulus(0),
        std::vector<poseidon::Modulus>{},
        std::vector<poseidon::Modulus>{},
        poseidon::sec_level_type::none);

    parms.set_log_modulus(
        std::vector<std::uint32_t>(q_count, 30),
        std::vector<std::uint32_t>(p_count, 30));
    return parms;
}

[[maybe_unused]] void print_cipher_meta(
    const std::string &name,
    const poseidon::Ciphertext &cipher)
{
    std::cout << "\n[" << name << " metadata]\n";
    std::cout << "component_count = " << cipher.size() << "\n";
    std::cout << "degree          = " << cipher.poly_modulus_degree() << "\n";
    std::cout << "q_count         = " << cipher.coeff_modulus_size() << "\n";
    std::cout << "is_ntt_form     = " << cipher.is_ntt_form() << "\n";
    std::cout << "scale           = " << cipher.scale() << "\n";
}

[[maybe_unused]] void print_gpu_cipher_layout(
    const std::string &name,
    const poseidon::gpu::GpuCiphertextData &cipher)
{
    std::cout << "\n[" << name << " GPU layout]\n";
    std::cout << "component_count = " << cipher.meta.component_count << "\n";
    std::cout << "degree          = " << cipher.meta.degree << "\n";
    std::cout << "q_count         = " << cipher.meta.q_count << "\n";
    std::cout << "p_count         = " << cipher.meta.p_count << "\n";
    std::cout << "fields          = " << cipher.fields_.size() << "\n";
    std::cout << "polys           = " << cipher.polys_.size() << "\n";

    for (std::size_t i = 0; i < cipher.polys_.size(); ++i)
    {
        const auto &poly = cipher.polys_[i];
        std::cout << "  poly[" << i << "] shards = " << poly.shards.size() << "\n";
        for (std::size_t j = 0; j < poly.shards.size(); ++j)
        {
            const auto &shard = poly.shards[j];
            const auto &field = cipher.fields_[shard.field_index];
            std::cout << "    shard[" << j << "] field=" << shard.field_index
                      << " device=" << field.device_id
                      << " offset=" << shard.field_offset
                      << " limb=[" << shard.limb_begin << ", "
                      << shard.limb_begin + shard.limb_count << ")"
                      << " coeff=[" << shard.coeff_begin << ", "
                      << shard.coeff_begin + shard.coeff_count << ")"
                      << " field_size=" << field.size() << "\n";
        }
    }
}

[[maybe_unused]] void print_first_words(
    const std::string &name,
    const poseidon::Ciphertext &cipher,
    std::size_t count)
{
    const std::size_t total =
        cipher.size() * cipher.poly_modulus_degree() * cipher.coeff_modulus_size();
    count = std::min(count, total);

    std::cout << "\n[" << name << " first " << count << " raw words]\n";
    for (std::size_t i = 0; i < count; ++i)
    {
        std::cout << name << ".data()[" << i << "] = " << cipher.data()[i] << "\n";
    }
}

[[maybe_unused]] bool print_raw_comparison(
    const poseidon::Ciphertext &cpu_result,
    const poseidon::Ciphertext &gpu_result,
    std::size_t max_mismatches)
{
    const std::size_t cpu_total =
        cpu_result.size() * cpu_result.poly_modulus_degree() * cpu_result.coeff_modulus_size();
    const std::size_t gpu_total =
        gpu_result.size() * gpu_result.poly_modulus_degree() * gpu_result.coeff_modulus_size();
    const std::size_t total = std::min(cpu_total, gpu_total);

    std::size_t mismatch_count = 0;
    std::cout << "\n[CPU/GPU raw residue comparison]\n";
    std::cout << "cpu raw words = " << cpu_total << "\n";
    std::cout << "gpu raw words = " << gpu_total << "\n";

    for (std::size_t i = 0; i < total; ++i)
    {
        if (cpu_result.data()[i] != gpu_result.data()[i])
        {
            if (mismatch_count < max_mismatches)
            {
                std::cout << "mismatch[" << mismatch_count << "] index=" << i
                          << " cpu=" << cpu_result.data()[i]
                          << " gpu=" << gpu_result.data()[i] << "\n";
            }
            ++mismatch_count;
        }
    }

    std::cout << "mismatch_count = " << mismatch_count << "\n";
    std::cout << "raw_equal      = " << (mismatch_count == 0 && cpu_total == gpu_total ? "YES" : "NO")
              << "\n";
    return mismatch_count == 0 && cpu_total == gpu_total;
}

bool ciphertext_raw_equal(
    const poseidon::Ciphertext &expected,
    const poseidon::Ciphertext &actual)
{
    const std::size_t expected_total =
        expected.size() * expected.poly_modulus_degree() *
        expected.coeff_modulus_size();
    const std::size_t actual_total =
        actual.size() * actual.poly_modulus_degree() *
        actual.coeff_modulus_size();

    if (expected_total != actual_total)
    {
        return false;
    }

    for (std::size_t i = 0; i < expected_total; ++i)
    {
        if (expected.data()[i] != actual.data()[i])
        {
            return false;
        }
    }
    return expected.is_ntt_form() == actual.is_ntt_form() &&
           expected.parms_id() == actual.parms_id() &&
           expected.size() == actual.size() &&
           expected.poly_modulus_degree() == actual.poly_modulus_degree() &&
           expected.coeff_modulus_size() == actual.coeff_modulus_size();
}

bool device_vector_matches_u64_segment(
    const poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> &actual_device,
    const std::vector<std::uint64_t> &expected,
    std::size_t expected_offset,
    const char *name)
{
    if (expected_offset > expected.size() ||
        actual_device.size() > expected.size() - expected_offset)
    {
        return false;
    }

    std::vector<poseidon::gpu::GpuWord> actual(actual_device.size());
    poseidon::gpu::gpu_check_cuda(
        cudaMemcpy(
            actual.data(),
            actual_device.data(),
            actual.size() * sizeof(poseidon::gpu::GpuWord),
            cudaMemcpyDeviceToHost),
        name);
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        if (actual[i] !=
            static_cast<poseidon::gpu::GpuWord>(expected[expected_offset + i]))
        {
            return false;
        }
    }
    return true;
}

bool hybrid_modup_q_matches_generated_reference(
    const poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> &actual_device,
    const std::vector<std::uint64_t> &expected_coeff_modup_q,
    std::size_t expected_coeff_offset,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    std::size_t base_q_size,
    std::size_t degree,
    const char *name)
{
    const std::size_t expected_size = base_q_size * degree;
    if (actual_device.size() != expected_size)
    {
        return false;
    }
    if (expected_coeff_offset > expected_coeff_modup_q.size() ||
        expected_size > expected_coeff_modup_q.size() - expected_coeff_offset)
    {
        return false;
    }

    std::vector<poseidon::gpu::GpuWord> actual(actual_device.size());
    poseidon::gpu::gpu_check_cuda(
        cudaMemcpy(
            actual.data(),
            actual_device.data(),
            actual.size() * sizeof(poseidon::gpu::GpuWord),
            cudaMemcpyDeviceToHost),
        name);

    const std::size_t decomp_limb_end = decomp_limb_begin + decomp_limb_count;
    for (std::size_t q_limb = 0; q_limb < base_q_size; ++q_limb)
    {
        if (q_limb >= decomp_limb_begin && q_limb < decomp_limb_end)
        {
            continue;
        }
        for (std::size_t coeff = 0; coeff < degree; ++coeff)
        {
            const std::size_t index = q_limb * degree + coeff;
            const std::uint64_t expected =
                expected_coeff_modup_q[expected_coeff_offset + index];
            if (actual[index] != static_cast<poseidon::gpu::GpuWord>(expected))
            {
                return false;
            }
        }
    }
    return true;
}

[[maybe_unused]] void print_decoded_slots(
    const std::string &name,
    const std::vector<double> &slots,
    std::size_t count)
{
    count = std::min(count, slots.size());
    std::cout << "\n[" << name << " decoded first " << count << " slots]\n";
    std::cout << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < count; ++i)
    {
        std::cout << name << "[" << i << "] = " << slots[i] << "\n";
    }
}

template <typename CpuOnce, typename GpuOnce>
TimingResult benchmark_cpu_gpu_average(
    int device_id,
    int timing_iterations,
    CpuOnce cpu_once,
    GpuOnce gpu_once,
    const char *name,
    std::size_t gpu_warmup_iterations = 1)
{
    cpu_once();

    const auto cpu_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < timing_iterations; ++i)
    {
        cpu_once();
    }
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_total_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();

    poseidon::gpu::gpu_check_cuda(cudaSetDevice(device_id), name);
    for (std::size_t i = 0; i < gpu_warmup_iterations; ++i)
    {
        gpu_once();
    }
    poseidon::gpu::gpu_check_cuda(cudaDeviceSynchronize(), name);

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    poseidon::gpu::gpu_check_cuda(cudaEventCreate(&gpu_start), name);
    poseidon::gpu::gpu_check_cuda(cudaEventCreate(&gpu_stop), name);

    const auto gpu_wall_begin = std::chrono::steady_clock::now();
    poseidon::gpu::gpu_check_cuda(cudaEventRecord(gpu_start), name);
    for (int i = 0; i < timing_iterations; ++i)
    {
        gpu_once();
    }
    poseidon::gpu::gpu_check_cuda(cudaEventRecord(gpu_stop), name);
    poseidon::gpu::gpu_check_cuda(cudaEventSynchronize(gpu_stop), name);
    const auto gpu_wall_end = std::chrono::steady_clock::now();

    float gpu_event_total_ms = 0.0F;
    poseidon::gpu::gpu_check_cuda(
        cudaEventElapsedTime(&gpu_event_total_ms, gpu_start, gpu_stop),
        name);

    poseidon::gpu::gpu_check_cuda(cudaEventDestroy(gpu_start), name);
    poseidon::gpu::gpu_check_cuda(cudaEventDestroy(gpu_stop), name);

    TimingResult result;
    result.cpu_avg_ms = cpu_total_ms / timing_iterations;
    result.gpu_wall_avg_ms =
        std::chrono::duration<double, std::milli>(
            gpu_wall_end - gpu_wall_begin).count() / timing_iterations;
    result.gpu_event_avg_ms = gpu_event_total_ms / timing_iterations;
    result.wall_speedup = result.cpu_avg_ms / result.gpu_wall_avg_ms;
    result.event_speedup = result.cpu_avg_ms / result.gpu_event_avg_ms;
    return result;
}

struct CpuHybridBconvScratch
{
    std::vector<std::uint64_t> modup_q;
    std::vector<std::uint64_t> modup_p;
    poseidon::util::PolyIter modup_q_iter;
    poseidon::util::PolyIter modup_p_iter;

    CpuHybridBconvScratch(
        std::size_t decomp_count,
        std::size_t degree,
        std::size_t base_q_size,
        std::size_t base_p_size)
        : modup_q(decomp_count * degree * base_q_size),
          modup_p(decomp_count * degree * base_p_size),
          modup_q_iter(modup_q.data(), degree, base_q_size),
          modup_p_iter(modup_p.data(), degree, base_p_size)
    {}
};

struct GpuHybridBconvScratch
{
    poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> c2_coeff;
    poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> modup_q;
    poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> modup_p;
};

std::size_t checked_benchmark_mul(
    std::size_t left,
    std::size_t right,
    const char *name)
{
    if (left != 0 &&
        right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error(name);
    }
    return left * right;
}

void run_cpu_hybrid_bconv_modup(
    const poseidon::PoseidonContext &context,
    const poseidon::RNSPoly &c2_coeff,
    CpuHybridBconvScratch &scratch)
{
    const auto context_data =
        context.crt_context()->get_context_data(c2_coeff.parms_id());
    if (!context_data)
    {
        throw std::invalid_argument("BConv CPU benchmark: invalid parms_id");
    }

    const auto rns_qp = context_data->qp_rns_tool();
    const auto base_q_size = rns_qp->base_q()->size();
    const auto base_p_size = rns_qp->base_p()->size();
    const auto decomp_count =
        (base_q_size + base_p_size - 1) / base_p_size;
    const auto pool = poseidon::MemoryManager::GetPool();

    for (std::size_t decomp_index = 0; decomp_index < decomp_count;
         ++decomp_index)
    {
        const auto decomp_limb_begin = decomp_index * base_p_size;
        const auto decomp_limb_count = std::min(
            base_p_size,
            base_q_size - decomp_limb_begin);

        rns_qp->mod_up_copy_q(
            c2_coeff.const_poly_iter()[0],
            decomp_index,
            scratch.modup_q_iter[decomp_index],
            pool);
        if (decomp_limb_count == 1)
        {
            rns_qp->mod_up_from_one_base_q(
                scratch.modup_q_iter[decomp_index],
                decomp_index,
                pool);
            rns_qp->mod_up_from_one_base_p(
                scratch.modup_q_iter[decomp_index],
                decomp_index,
                scratch.modup_p_iter[decomp_index],
                pool);
        }
        else
        {
            rns_qp->mod_up_base_q(
                scratch.modup_q_iter[decomp_index],
                decomp_index,
                pool);
            rns_qp->mod_up_base_p(
                scratch.modup_q_iter[decomp_index],
                decomp_index,
                scratch.modup_p_iter[decomp_index],
                pool);
        }
    }
}

const poseidon::gpu::GpuParameterShard &find_benchmark_parameter_shard(
    const poseidon::gpu::GpuLevelInfo &level_info,
    const poseidon::gpu::GpuConstPolyShardView &source_shard)
{
    for (const auto &candidate : level_info.shards)
    {
        const bool same_device = candidate.device_id == source_shard.device_id;
        const bool covers_limb =
            source_shard.limb_begin >= candidate.limb_begin &&
            source_shard.limb_begin + source_shard.limb_count <=
                candidate.limb_begin + candidate.limb_count;
        if (same_device && covers_limb)
        {
            return candidate;
        }
    }

    throw std::invalid_argument(
        "BConv GPU benchmark: no matching parameter shard");
}

void prepare_gpu_hybrid_bconv_c2_coeff(
    GpuHybridBconvScratch &scratch,
    const poseidon::gpu::GpuConstPolyShardView &c2_ntt_shard,
    const poseidon::gpu::GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t q_count)
{
    poseidon::gpu::GpuPolyShardView c2_coeff_shard;
    c2_coeff_shard.device_id = c2_ntt_shard.device_id;
    c2_coeff_shard.ptr = scratch.c2_coeff.data();
    c2_coeff_shard.limb_begin = 0;
    c2_coeff_shard.limb_count = q_count;
    c2_coeff_shard.coeff_begin = 0;
    c2_coeff_shard.coeff_count = degree;

    poseidon::gpu::kernel::launch_inverse_ntt_poly_shard(
        c2_coeff_shard,
        c2_ntt_shard,
        parameter_shard,
        degree);
}

void run_gpu_hybrid_bconv_modup(
    GpuHybridBconvScratch &scratch,
    const poseidon::gpu::GpuConstPolyShardView &c2_ntt_shard,
    const poseidon::gpu::GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t base_q_size,
    std::size_t base_p_size,
    std::size_t decomp_count)
{
    (void)c2_ntt_shard;
    for (std::size_t decomp_index = 0; decomp_index < decomp_count;
         ++decomp_index)
    {
        const std::size_t decomp_limb_begin = decomp_index * base_p_size;
        const std::size_t decomp_limb_count = std::min(
            base_p_size,
            base_q_size - decomp_limb_begin);
        poseidon::gpu::kernel::launch_hybrid_modup_decomposition(
            scratch.modup_q.data(),
            scratch.modup_p.data(),
            scratch.c2_coeff.data(),
            c2_ntt_shard.ptr,
            decomp_index,
            decomp_limb_begin,
            decomp_limb_count,
            parameter_shard,
            degree);
    }
}

template <typename GpuOnce>
GpuSampleStats benchmark_gpu_event_samples(
    int device_id,
    int timing_iterations,
    GpuOnce gpu_once,
    const char *name,
    std::size_t gpu_warmup_iterations)
{
    poseidon::gpu::gpu_check_cuda(cudaSetDevice(device_id), name);
    for (std::size_t i = 0; i < gpu_warmup_iterations; ++i)
    {
        gpu_once();
    }
    poseidon::gpu::gpu_check_cuda(cudaDeviceSynchronize(), name);

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    poseidon::gpu::gpu_check_cuda(cudaEventCreate(&gpu_start), name);
    poseidon::gpu::gpu_check_cuda(cudaEventCreate(&gpu_stop), name);

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(timing_iterations));
    for (int i = 0; i < timing_iterations; ++i)
    {
        poseidon::gpu::gpu_check_cuda(cudaEventRecord(gpu_start), name);
        gpu_once();
        poseidon::gpu::gpu_check_cuda(cudaEventRecord(gpu_stop), name);
        poseidon::gpu::gpu_check_cuda(cudaEventSynchronize(gpu_stop), name);

        float elapsed_ms = 0.0F;
        poseidon::gpu::gpu_check_cuda(
            cudaEventElapsedTime(&elapsed_ms, gpu_start, gpu_stop),
            name);
        samples.push_back(static_cast<double>(elapsed_ms));
    }

    poseidon::gpu::gpu_check_cuda(cudaEventDestroy(gpu_start), name);
    poseidon::gpu::gpu_check_cuda(cudaEventDestroy(gpu_stop), name);

    return summarize_gpu_samples(samples);
}

GpuSampleStats summarize_gpu_samples(const std::vector<double> &samples)
{
    if (samples.empty())
    {
        return GpuSampleStats{};
    }

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    double total_ms = 0.0;
    for (double sample : samples)
    {
        total_ms += sample;
    }

    const std::size_t count = sorted.size();
    const std::size_t median_index = count / 2;
    const std::size_t p90_index =
        std::min(count - 1, ((count * 9 + 9) / 10) - 1);

    GpuSampleStats result;
    result.avg_ms = total_ms / static_cast<double>(count);
    result.min_ms = sorted.front();
    result.median_ms = count % 2 == 0
        ? (sorted[median_index - 1] + sorted[median_index]) / 2.0
        : sorted[median_index];
    result.p90_ms = sorted[p90_index];
    result.max_ms = sorted.back();
    return result;
}

template <typename GpuOnce>
std::vector<GpuSampleStats> benchmark_ntt_stage_profile_samples(
    int device_id,
    int timing_iterations,
    GpuOnce gpu_once,
    const char *name,
    std::size_t gpu_warmup_iterations,
    std::size_t phase_count)
{
    poseidon::gpu::gpu_check_cuda(cudaSetDevice(device_id), name);
    poseidon::gpu::kernel::set_ntt_stage_profile_enabled(false);
    poseidon::gpu::kernel::reset_ntt_stage_profile();
    for (std::size_t i = 0; i < gpu_warmup_iterations; ++i)
    {
        gpu_once();
    }
    poseidon::gpu::gpu_check_cuda(cudaDeviceSynchronize(), name);

    phase_count = std::min(
        phase_count,
        poseidon::gpu::kernel::NttStageProfileSnapshot::kMaxStageCount);
    std::vector<std::vector<double>> phase_samples(phase_count);
    for (auto &samples : phase_samples)
    {
        samples.reserve(static_cast<std::size_t>(timing_iterations));
    }

    for (int i = 0; i < timing_iterations; ++i)
    {
        poseidon::gpu::kernel::reset_ntt_stage_profile();
        poseidon::gpu::kernel::set_ntt_stage_profile_enabled(true);
        gpu_once();
        poseidon::gpu::gpu_check_cuda(cudaDeviceSynchronize(), name);
        poseidon::gpu::kernel::set_ntt_stage_profile_enabled(false);
        const auto snapshot =
            poseidon::gpu::kernel::get_ntt_stage_profile_snapshot();
        for (std::size_t phase = 0; phase < phase_count; ++phase)
        {
            phase_samples[phase].push_back(snapshot.stage_total_ms[phase]);
        }
    }
    poseidon::gpu::kernel::reset_ntt_stage_profile();

    std::vector<GpuSampleStats> result;
    result.reserve(phase_count);
    for (const auto &samples : phase_samples)
    {
        result.push_back(summarize_gpu_samples(samples));
    }
    return result;
}

std::string format_fixed(double value, int precision)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string format_speedup(double value)
{
    return format_fixed(value, 2) + "x";
}

void print_operation_timing_table(
    const std::vector<OperationTimingRow> &rows,
    std::size_t degree,
    std::size_t q_count,
    std::size_t p_count,
    int timing_iterations,
    std::size_t warmup_iterations)
{
    constexpr int op_width = 31;
    constexpr int cpu_width = 14;
    constexpr int wall_width = 19;
    constexpr int event_width = 20;
    constexpr int wall_speedup_width = 14;
    constexpr int event_speedup_width = 15;
    constexpr int correct_width = 7;

    const auto print_border = [&]()
    {
        std::cout << '+'
                  << std::string(op_width + 2, '-')
                  << '+'
                  << std::string(cpu_width + 2, '-')
                  << '+'
                  << std::string(wall_width + 2, '-')
                  << '+'
                  << std::string(event_width + 2, '-')
                  << '+'
                  << std::string(wall_speedup_width + 2, '-')
                  << '+'
                  << std::string(event_speedup_width + 2, '-')
                  << '+'
                  << std::string(correct_width + 2, '-')
                  << "+\n";
    };

    const auto print_row = [&](
        const std::string &operation,
        const std::string &cpu_avg,
        const std::string &gpu_wall_avg,
        const std::string &gpu_event_avg,
        const std::string &wall_speedup,
        const std::string &event_speedup,
        const std::string &correct)
    {
        std::cout << "| " << std::left << std::setw(op_width) << operation
                  << " | " << std::right << std::setw(cpu_width) << cpu_avg
                  << " | " << std::right << std::setw(wall_width) << gpu_wall_avg
                  << " | " << std::right << std::setw(event_width) << gpu_event_avg
                  << " | " << std::right << std::setw(wall_speedup_width) << wall_speedup
                  << " | " << std::right << std::setw(event_speedup_width) << event_speedup
                  << " | " << std::right << std::setw(correct_width) << correct
                  << " |\n";
    };

    std::cout << "\n[operation timing summary]\n";
    print_border();
    print_row(
        "operation",
        "CPU avg (ms)",
        "GPU wall avg (ms)",
        "GPU event avg (ms)",
        "wall speedup",
        "event speedup",
        "correct");
    print_border();
    for (const auto &row : rows)
    {
        print_row(
            row.operation,
            format_fixed(row.timing.cpu_avg_ms, 6),
            format_fixed(row.timing.gpu_wall_avg_ms, 6),
            format_fixed(row.timing.gpu_event_avg_ms, 6),
            format_speedup(row.timing.wall_speedup),
            format_speedup(row.timing.event_speedup),
            row.correct);
    }
    print_border();

    std::cout << "parameters: CKKS, degree=" << degree
              << ", q=" << q_count
              << ", p=" << p_count
              << ", iterations=" << timing_iterations
              << ", warmup=" << warmup_iterations
              << " average\n";
    std::cout << "benchmark input: deterministic key/encrypt randomness\n";
    std::cout << "excluded from timing: encode/encrypt/upload/download/decrypt/decode\n";
    std::cout << "correct: raw residues and metadata compared with CPU result; BConv compares generated limbs in the last HYBRID ModUp block\n";
    std::cout << "ntt variants: ntt_inv/ntt_fwd use default CUDA fused3 unless POSEIDON_NTT_ALGO or POSEIDON_NTT_FUSION_STAGES overrides it\n";
    std::cout << "keyswitch_bconv_modup: HYBRID ModUp/BConv only; c2 INTT and later NTT/key multiply/ModDown are excluded\n";
}

void print_ntt_timing_table(
    const std::vector<NttTimingRow> &rows,
    std::size_t degree,
    std::size_t q_count,
    std::size_t p_count,
    int timing_iterations,
    std::size_t warmup_iterations)
{
    constexpr int op_width = 12;
    constexpr int mode_width = 8;
    constexpr int value_width = 13;
    constexpr int correct_width = 7;

    const auto print_border = [&]()
    {
        std::cout << '+'
                  << std::string(op_width + 2, '-')
                  << '+'
                  << std::string(mode_width + 2, '-')
                  << '+'
                  << std::string(value_width + 2, '-')
                  << '+'
                  << std::string(value_width + 2, '-')
                  << '+'
                  << std::string(value_width + 2, '-')
                  << '+'
                  << std::string(value_width + 2, '-')
                  << '+'
                  << std::string(value_width + 2, '-')
                  << '+'
                  << std::string(correct_width + 2, '-')
                  << "+\n";
    };

    const auto print_row = [&](
        const std::string &operation,
        const std::string &mode,
        const std::string &avg,
        const std::string &min,
        const std::string &median,
        const std::string &p90,
        const std::string &max,
        const std::string &correct)
    {
        std::cout << "| " << std::left << std::setw(op_width) << operation
                  << " | " << std::right << std::setw(mode_width) << mode
                  << " | " << std::right << std::setw(value_width) << avg
                  << " | " << std::right << std::setw(value_width) << min
                  << " | " << std::right << std::setw(value_width) << median
                  << " | " << std::right << std::setw(value_width) << p90
                  << " | " << std::right << std::setw(value_width) << max
                  << " | " << std::right << std::setw(correct_width) << correct
                  << " |\n";
    };

    std::cout << "\n[NTT mode timing summary]\n";
    print_border();
    print_row(
        "operation",
        "mode",
        "avg (ms)",
        "min (ms)",
        "median (ms)",
        "p90 (ms)",
        "max (ms)",
        "correct");
    print_border();
    for (const auto &row : rows)
    {
        const bool skipped = row.correct == "/";
        print_row(
            row.operation,
            row.mode,
            skipped ? "/" : format_fixed(row.timing.avg_ms, 6),
            skipped ? "/" : format_fixed(row.timing.min_ms, 6),
            skipped ? "/" : format_fixed(row.timing.median_ms, 6),
            skipped ? "/" : format_fixed(row.timing.p90_ms, 6),
            skipped ? "/" : format_fixed(row.timing.max_ms, 6),
            row.correct);
    }
    print_border();

    std::cout << "parameters: CKKS, degree=" << degree
              << ", q=" << q_count
              << ", p=" << p_count
              << ", iterations=" << timing_iterations
              << ", warmup=" << warmup_iterations
              << " per NTT mode\n";
    std::cout << "benchmark scope: preallocated GpuCiphertextData + GpuNTTHandler only\n";
    std::cout << "correct: raw residues and metadata compared with CPU NTT result\n";
    std::cout << "algorithm: stage/fused2/fused3/fused4/fourstep/tensor/tensor_fp64 are selected by POSEIDON_NTT_ALGO\n";
    std::cout << "skip controls: POSEIDON_DEMO_SKIP_TENSOR_NTT=1, POSEIDON_DEMO_SKIP_TENSOR_FP64_NTT=1\n";
    std::cout << "phase rows: extra profiled pass; phase0..phase3 split FD=4 chunks and exclude outer copy/final INTT normalization\n";
}

void append_sweep_row(
    std::vector<SweepBenchmarkRow> &rows,
    const std::string &experiment,
    std::size_t degree,
    std::size_t q_count,
    const std::string &operation,
    std::size_t gpu_work_items,
    std::size_t gpu_kernel_launches,
    const TimingResult &timing)
{
    SweepBenchmarkRow row;
    row.experiment = experiment;
    row.degree = degree;
    row.q_count = q_count;
    row.operation = operation;
    row.gpu_work_items = gpu_work_items;
    row.gpu_kernel_launches = gpu_kernel_launches;
    row.cpu_avg_ms = timing.cpu_avg_ms;
    row.gpu_wall_avg_ms = timing.gpu_wall_avg_ms;
    row.gpu_event_avg_ms = timing.gpu_event_avg_ms;
    row.wall_speedup = timing.wall_speedup;
    row.event_speedup = timing.event_speedup;
    rows.push_back(std::move(row));
}

void print_sweep_benchmark_table(
    const std::vector<SweepBenchmarkRow> &rows,
    int timing_iterations)
{
    const std::vector<std::string> operations{
        "sub_operation",
        "multiply_plain",
        "ntt_fwd"};

    std::cout << "\n[parameter sweep benchmark summary]\n";
    std::cout << "iterations per row = " << timing_iterations << "\n";
    std::cout << "gpu wall avg ms    = host wall-clock time with CUDA event synchronization\n";
    std::cout << "gpu event avg ms   = CUDA event elapsed time for the same stream region\n";

    std::cout << std::fixed << std::setprecision(6);

    for (const auto &operation : operations)
    {
        std::cout << "\n[" << operation << "]\n";
        std::cout << std::left
                  << std::setw(12) << "experiment"
                  << std::right
                  << std::setw(10) << "N"
                  << std::setw(10) << "q_count"
                  << std::setw(12) << "items"
                  << std::setw(10) << "launches"
                  << std::setw(14) << "cpu_avg_ms"
                  << std::setw(16) << "gpu_wall_ms"
                  << std::setw(17) << "gpu_event_ms"
                  << std::setw(13) << "wall_spd"
                  << std::setw(14) << "event_spd"
                  << "\n";

        for (const auto &row : rows)
        {
            if (row.operation != operation)
            {
                continue;
            }

            std::cout << std::left
                      << std::setw(12) << row.experiment
                      << std::right
                      << std::setw(10) << row.degree
                      << std::setw(10) << row.q_count
                      << std::setw(12) << row.gpu_work_items
                      << std::setw(10) << row.gpu_kernel_launches
                      << std::setw(14) << row.cpu_avg_ms
                      << std::setw(16) << row.gpu_wall_avg_ms
                      << std::setw(17) << row.gpu_event_avg_ms
                      << std::setw(12) << row.wall_speedup << "x"
                      << std::setw(13) << row.event_speedup << "x"
                      << "\n";
        }
    }
    std::cout << std::right;
}

void run_sweep_benchmark_case(
    const std::string &experiment,
    std::size_t degree,
    std::size_t q_count,
    int device_id,
    int timing_iterations,
    std::vector<SweepBenchmarkRow> &rows)
{
    using namespace poseidon;
    using namespace poseidon::gpu;

    std::cout << "\n[parameter sweep setup] experiment=" << experiment
              << " N=" << degree
              << " q_count=" << q_count << "\n";

    const auto parms = make_benchmark_parameters(degree, q_count);
    PoseidonContext context(parms);
    use_benchmark_randomness(context);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);

    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key, keygen.secret_key());

    const std::vector<double> input0{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> input1{5.0, 6.0, 7.0, 8.0};

    Plaintext plain0;
    Plaintext plain1;
    encoder.encode(input0, parms.scale(), plain0);
    encoder.encode(input1, parms.scale(), plain1);

    Ciphertext ct0;
    Ciphertext ct1;
    encryptor.encrypt(plain0, ct0);
    encryptor.encrypt(plain1, ct1);

    auto cpu_evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);
    GpuParameterData gpu_params(context, device_id);
    GpuEvaluator gpu_evaluator(gpu_params);
    GpuElementwiseHandler gpu_elementwise_handler(gpu_params);

    auto gpu_ct0 = GpuUploader::upload_ciphertext(ct0, device_id);
    auto gpu_ct1 = GpuUploader::upload_ciphertext(ct1, device_id);
    auto gpu_plain1 = GpuUploader::upload_plaintext(plain1, device_id);
    const auto &level_info = gpu_params.get_level(ct0.parms_id());

    auto gpu_ct0_view = gpu_ct0.make_const_view();
    auto gpu_ct1_view = gpu_ct1.make_const_view();
    auto gpu_plain1_view = gpu_plain1.make_const_view();
    const std::size_t component_count = gpu_ct0.meta.component_count;
    const std::size_t elementwise_work_items =
        component_count * degree * q_count;
    const std::size_t elementwise_kernel_launches = component_count;
    const std::size_t ntt_full_work_items =
        component_count *
        q_count *
        (degree >> 1) *
        static_cast<std::size_t>(log2_degree(degree));
    const std::size_t ntt_full_kernel_launches =
        component_count * static_cast<std::size_t>(log2_degree(degree));

    Ciphertext cpu_sub_result;
    GpuCiphertextData gpu_sub_result =
        GpuCiphertextData::allocate_single_device_sharded(
            gpu_ct0.meta.degree,
            gpu_ct0.meta.q_count,
            gpu_ct0.meta.component_count,
            device_id,
            gpu_ct0.polys_.at(0).shards,
            gpu_ct0.meta.p_count);
    gpu_sub_result.meta = gpu_ct0.meta;
    auto gpu_sub_view = gpu_sub_result.make_view();
    append_sweep_row(
        rows,
        experiment,
        degree,
        q_count,
        "sub_operation",
        elementwise_work_items,
        elementwise_kernel_launches,
        benchmark_cpu_gpu_average(
            device_id,
            timing_iterations,
            [&]() { cpu_evaluator->sub(ct0, ct1, cpu_sub_result); },
            [&]()
            {
                gpu_elementwise_handler.sub_ciphertext(
                    gpu_sub_view,
                    gpu_ct0_view,
                    gpu_ct1_view,
                    level_info);
            },
            "sweep sub_operation"));

    Ciphertext cpu_multiply_plain_result;
    GpuCiphertextData gpu_multiply_plain_result =
        GpuCiphertextData::allocate_single_device_sharded(
            gpu_ct0.meta.degree,
            gpu_ct0.meta.q_count,
            gpu_ct0.meta.component_count,
            device_id,
            gpu_ct0.polys_.at(0).shards,
            gpu_ct0.meta.p_count);
    gpu_multiply_plain_result.meta = gpu_ct0.meta;
    gpu_multiply_plain_result.meta.scale =
        gpu_ct0.meta.scale * gpu_plain1.meta.scale;
    auto gpu_multiply_plain_view = gpu_multiply_plain_result.make_view();
    append_sweep_row(
        rows,
        experiment,
        degree,
        q_count,
        "multiply_plain",
        elementwise_work_items,
        elementwise_kernel_launches,
        benchmark_cpu_gpu_average(
            device_id,
            timing_iterations,
            [&]() { cpu_evaluator->multiply_plain(ct0, plain1, cpu_multiply_plain_result); },
            [&]()
            {
                gpu_elementwise_handler.multiply_plain_with_ciphertext(
                    gpu_multiply_plain_view,
                    gpu_ct0_view,
                    gpu_plain1_view,
                    level_info);
            },
            "sweep multiply_plain"));

    Ciphertext cpu_ntt_stage_source;
    cpu_evaluator->ntt_inv(ct0, cpu_ntt_stage_source);
    auto gpu_ntt_stage_source =
        GpuUploader::upload_ciphertext(cpu_ntt_stage_source, device_id);
    Ciphertext cpu_ntt_result;
    GpuCiphertextData gpu_ntt_result;
    append_sweep_row(
        rows,
        experiment,
        degree,
        q_count,
        "ntt_fwd",
        ntt_full_work_items,
        ntt_full_kernel_launches,
        benchmark_cpu_gpu_average(
            device_id,
            timing_iterations,
            [&]() { cpu_evaluator->ntt_fwd(cpu_ntt_stage_source, cpu_ntt_result); },
            [&]() { gpu_evaluator.ntt_fwd(gpu_ntt_stage_source, gpu_ntt_result); },
            "sweep ntt_fwd"));
}

void run_parameter_sweep_benchmarks(int device_id)
{
    constexpr int sweep_timing_iterations = 200;
    constexpr std::size_t n_sweep_q_count = 8;
    const std::vector<std::size_t> n_sweep_degrees{
        8192,
        16384,
        32768,
        65536};
    const std::vector<std::size_t> q_sweep_counts{
        8,
        12,
        16,
        20,
        24,
        28,
        32};

    std::vector<SweepBenchmarkRow> rows;
    rows.reserve((n_sweep_degrees.size() + q_sweep_counts.size()) * 4);

    std::cout << "\n===== PARAMETER SWEEP BENCHMARKS =====\n";
    std::cout << "N sweep fixed q_count = " << n_sweep_q_count << "\n";
    std::cout << "q_count sweep fixed N = 16384\n";

    for (const auto degree : n_sweep_degrees)
    {
        run_sweep_benchmark_case(
            "N_sweep",
            degree,
            n_sweep_q_count,
            device_id,
            sweep_timing_iterations,
            rows);
    }

    for (const auto q_count : q_sweep_counts)
    {
        run_sweep_benchmark_case(
            "q_sweep",
            16384,
            q_count,
            device_id,
            sweep_timing_iterations,
            rows);
    }

    print_sweep_benchmark_table(rows, sweep_timing_iterations);
}

struct NsysMultiplyPlainCase
{
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t component_count = 0;
    std::size_t words_per_component = 0;
    std::size_t blocks_per_kernel = 0;
    poseidon::parms_id_type parms_id{};
    std::unique_ptr<poseidon::PoseidonContext> context;
    std::unique_ptr<poseidon::gpu::GpuParameterData> gpu_params;
    std::unique_ptr<poseidon::gpu::GpuElementwiseHandler> handler;
    poseidon::gpu::GpuCiphertextData gpu_ct0;
    poseidon::gpu::GpuPlaintextData gpu_plain1;
    poseidon::gpu::GpuCiphertextData gpu_result;

    void run_once()
    {
        auto destination_view = gpu_result.make_view();
        auto ciphertext_view = gpu_ct0.make_const_view();
        auto plaintext_view = gpu_plain1.make_const_view();
        const auto &level_info = gpu_params->get_level(parms_id);
        handler->multiply_plain_with_ciphertext(
            destination_view,
            ciphertext_view,
            plaintext_view,
            level_info);
    }
};

NsysMultiplyPlainCase prepare_nsys_multiply_plain_case(
    std::size_t degree,
    std::size_t q_count,
    int device_id)
{
    using namespace poseidon;
    using namespace poseidon::gpu;

    NsysMultiplyPlainCase result;
    result.degree = degree;
    result.q_count = q_count;

    const auto parms = make_benchmark_parameters(degree, q_count);
    result.context = std::make_unique<PoseidonContext>(parms);
    use_benchmark_randomness(*result.context);

    KeyGenerator keygen(*result.context);
    PublicKey public_key;
    keygen.create_public_key(public_key);

    CKKSEncoder encoder(*result.context);
    Encryptor encryptor(*result.context, public_key, keygen.secret_key());

    Plaintext plain0;
    Plaintext plain1;
    encoder.encode(std::vector<double>{1.0, 2.0, 3.0, 4.0}, parms.scale(), plain0);
    encoder.encode(std::vector<double>{5.0, 6.0, 7.0, 8.0}, parms.scale(), plain1);

    Ciphertext ct0;
    encryptor.encrypt(plain0, ct0);
    result.parms_id = ct0.parms_id();

    result.gpu_params = std::make_unique<GpuParameterData>(*result.context, device_id);
    result.handler = std::make_unique<GpuElementwiseHandler>(*result.gpu_params);
    result.gpu_ct0 = GpuUploader::upload_ciphertext(ct0, device_id);
    result.gpu_plain1 = GpuUploader::upload_plaintext(plain1, device_id);

    result.gpu_result =
        GpuCiphertextData::allocate_single_device_sharded(
            result.gpu_ct0.meta.degree,
            result.gpu_ct0.meta.q_count,
            result.gpu_ct0.meta.component_count,
            device_id,
            result.gpu_ct0.polys_.at(0).shards,
            result.gpu_ct0.meta.p_count);
    result.gpu_result.meta = result.gpu_ct0.meta;
    result.gpu_result.meta.scale =
        result.gpu_ct0.meta.scale * result.gpu_plain1.meta.scale;

    constexpr std::size_t block_size = 256;
    result.component_count = result.gpu_ct0.meta.component_count;
    result.words_per_component = degree * q_count;
    result.blocks_per_kernel =
        (result.words_per_component + block_size - 1) / block_size;

    return result;
}

int run_nsys_multiply_plain_probe()
{
    using poseidon::gpu::gpu_check_cuda;

    const int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const std::size_t degree =
        env_size_or("POSEIDON_NSYS_DEGREE", 16384);
    std::vector<std::size_t> q_counts =
        env_size_list_or_empty("POSEIDON_NSYS_Q_SWEEP");
    if (q_counts.empty())
    {
        q_counts.push_back(env_size_or("POSEIDON_NSYS_Q_COUNT", 16));
    }
    const std::size_t timing_iterations =
        env_size_or("POSEIDON_NSYS_ITERATIONS", 200);
    const std::size_t warmup_iterations =
        env_size_or("POSEIDON_NSYS_WARMUP", 5);

    if (timing_iterations == 0)
    {
        throw std::invalid_argument(
            "POSEIDON_NSYS_ITERATIONS must be greater than zero");
    }

    std::vector<NsysMultiplyPlainCase> cases;
    cases.reserve(q_counts.size());
    for (const auto q_count : q_counts)
    {
        cases.push_back(
            prepare_nsys_multiply_plain_case(degree, q_count, device_id));
    }

    gpu_check_cuda(cudaSetDevice(device_id), "nsys multiply_plain cudaSetDevice");
    for (auto &current_case : cases)
    {
        for (std::size_t i = 0; i < warmup_iterations; ++i)
        {
            current_case.run_once();
        }
    }
    gpu_check_cuda(
        cudaDeviceSynchronize(),
        "nsys multiply_plain warmup synchronize");

    constexpr std::size_t block_size = 256;
    std::size_t total_kernel_launches = 0;
    std::size_t total_work_items = 0;
    for (const auto &current_case : cases)
    {
        total_kernel_launches +=
            current_case.component_count * timing_iterations;
        total_work_items +=
            current_case.component_count *
            current_case.words_per_component *
            timing_iterations;
    }

    std::cout << "\n[nsys multiply_plain probe]\n";
    std::cout << "degree                 = " << degree << "\n";
    std::cout << "q_count cases          = ";
    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        if (i != 0)
        {
            std::cout << ",";
        }
        std::cout << cases[i].q_count;
    }
    std::cout << "\n";
    std::cout << "warmup iterations      = " << warmup_iterations << "\n";
    std::cout << "timing iterations      = " << timing_iterations << "\n";
    std::cout << "block_size             = " << block_size << "\n";
    std::cout << "total kernel launches  = " << total_kernel_launches << "\n";
    std::cout << "total logical items    = " << total_work_items << "\n";
    std::cout << "capture range          = cudaProfilerStart/Stop\n";
    std::cout << "nvtx ranges            = multiply_plain q=<q_count>\n";
    for (const auto &current_case : cases)
    {
        std::cout << "  q=" << current_case.q_count
                  << " component_count=" << current_case.component_count
                  << " blocks/kernel=" << current_case.blocks_per_kernel
                  << " launches/iter=" << current_case.component_count
                  << " logical_items="
                  << current_case.component_count *
                         current_case.words_per_component *
                         timing_iterations
                  << "\n";
    }

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    gpu_check_cuda(cudaEventCreate(&gpu_start), "nsys cudaEventCreate start");
    gpu_check_cuda(cudaEventCreate(&gpu_stop), "nsys cudaEventCreate stop");
    std::vector<cudaEvent_t> case_starts(cases.size(), nullptr);
    std::vector<cudaEvent_t> case_stops(cases.size(), nullptr);
    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        gpu_check_cuda(
            cudaEventCreate(&case_starts[i]),
            "nsys cudaEventCreate case start");
        gpu_check_cuda(
            cudaEventCreate(&case_stops[i]),
            "nsys cudaEventCreate case stop");
    }

    gpu_check_cuda(cudaProfilerStart(), "nsys cudaProfilerStart");
    const auto wall_begin = std::chrono::steady_clock::now();
    gpu_check_cuda(cudaEventRecord(gpu_start), "nsys cudaEventRecord start");
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index)
    {
        auto &current_case = cases[case_index];
        const std::string range_name =
            "multiply_plain q=" + std::to_string(current_case.q_count);
        nvtxRangePushA(range_name.c_str());
        gpu_check_cuda(
            cudaEventRecord(case_starts[case_index]),
            "nsys cudaEventRecord case start");
        for (std::size_t i = 0; i < timing_iterations; ++i)
        {
            current_case.run_once();
        }
        gpu_check_cuda(
            cudaEventRecord(case_stops[case_index]),
            "nsys cudaEventRecord case stop");
        gpu_check_cuda(
            cudaEventSynchronize(case_stops[case_index]),
            "nsys cudaEventSynchronize case stop");
        nvtxRangePop();
    }
    gpu_check_cuda(cudaEventRecord(gpu_stop), "nsys cudaEventRecord stop");
    gpu_check_cuda(
        cudaEventSynchronize(gpu_stop),
        "nsys cudaEventSynchronize stop");
    const auto wall_end = std::chrono::steady_clock::now();
    gpu_check_cuda(cudaProfilerStop(), "nsys cudaProfilerStop");

    float gpu_event_total_ms = 0.0F;
    gpu_check_cuda(
        cudaEventElapsedTime(&gpu_event_total_ms, gpu_start, gpu_stop),
        "nsys cudaEventElapsedTime");
    std::vector<float> case_event_total_ms(cases.size(), 0.0F);
    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        gpu_check_cuda(
            cudaEventElapsedTime(
                &case_event_total_ms[i],
                case_starts[i],
                case_stops[i]),
            "nsys cudaEventElapsedTime case");
    }

    const double wall_total_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    const double event_avg_per_iter =
        static_cast<double>(gpu_event_total_ms) /
        (timing_iterations * cases.size());
    const double event_avg_per_kernel =
        static_cast<double>(gpu_event_total_ms) / total_kernel_launches;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "gpu wall total ms      = " << wall_total_ms << "\n";
    std::cout << "gpu event total ms     = " << gpu_event_total_ms << "\n";
    std::cout << "gpu event avg/case-iter ms = " << event_avg_per_iter << "\n";
    std::cout << "gpu event avg/kernel ms= " << event_avg_per_kernel << "\n";
    std::cout << "nsys kernel name       = dyadic_product_poly_shard_kernel\n";
    std::cout << "\n[nsys multiply_plain per-q event summary]\n";
    std::cout << "q_count    event_total_ms    event_avg_iter_ms    event_avg_kernel_ms\n";
    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        const auto &current_case = cases[i];
        const std::size_t case_launches =
            current_case.component_count * timing_iterations;
        std::cout << std::setw(7) << current_case.q_count
                  << std::setw(18) << case_event_total_ms[i]
                  << std::setw(21)
                  << static_cast<double>(case_event_total_ms[i]) /
                         timing_iterations
                  << std::setw(23)
                  << static_cast<double>(case_event_total_ms[i]) /
                         case_launches
                  << "\n";
    }

    gpu_check_cuda(cudaEventDestroy(gpu_start), "nsys cudaEventDestroy start");
    gpu_check_cuda(cudaEventDestroy(gpu_stop), "nsys cudaEventDestroy stop");
    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        gpu_check_cuda(
            cudaEventDestroy(case_starts[i]),
            "nsys cudaEventDestroy case start");
        gpu_check_cuda(
            cudaEventDestroy(case_stops[i]),
            "nsys cudaEventDestroy case stop");
    }

    return EXIT_SUCCESS;
}

struct NsysMultiplyRelinRescaleCase
{
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;
    std::unique_ptr<poseidon::PoseidonContext> context;
    std::unique_ptr<poseidon::gpu::GpuParameterData> gpu_params;
    std::unique_ptr<poseidon::gpu::GpuEvaluator> evaluator;
    poseidon::gpu::GpuCiphertextData gpu_ct0;
    poseidon::gpu::GpuCiphertextData gpu_ct1;
    poseidon::gpu::GpuRelinKeysData gpu_relin_keys;
    poseidon::gpu::GpuCiphertextData gpu_multiply_result;
    poseidon::gpu::GpuCiphertextData gpu_relinearize_result;
    poseidon::gpu::GpuCiphertextData gpu_result;

    void run_once()
    {
        {
            NvtxRange range("chain.multiply");
            evaluator->multiply(gpu_ct0, gpu_ct1, gpu_multiply_result);
        }
        {
            NvtxRange range("chain.relinearize");
            evaluator->relinearize(
                gpu_multiply_result,
                gpu_relin_keys,
                gpu_relinearize_result);
        }
        {
            NvtxRange range("chain.rescale");
            evaluator->rescale(gpu_relinearize_result, gpu_result);
        }
    }

    void prepare_relinearize_input()
    {
        NvtxRange range("relinearize_setup.multiply");
        evaluator->multiply(gpu_ct0, gpu_ct1, gpu_multiply_result);
    }

    void run_relinearize_once()
    {
        NvtxRange range("relinearize");
        evaluator->relinearize(
            gpu_multiply_result,
            gpu_relin_keys,
            gpu_relinearize_result);
    }
};

NsysMultiplyRelinRescaleCase prepare_nsys_multiply_relin_rescale_case(
    std::size_t degree,
    std::size_t q_count,
    std::size_t p_count,
    int device_id)
{
    using namespace poseidon;
    using namespace poseidon::gpu;

    if (p_count == 0)
    {
        throw std::invalid_argument(
            "POSEIDON_NSYS_P_COUNT must be greater than zero for relinearize");
    }

    NsysMultiplyRelinRescaleCase result;
    result.degree = degree;
    result.q_count = q_count;
    result.p_count = p_count;

    const auto parms = make_benchmark_parameters(degree, q_count, p_count);
    result.context = std::make_unique<PoseidonContext>(parms);
    use_benchmark_randomness(*result.context);

    KeyGenerator keygen(*result.context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    CKKSEncoder encoder(*result.context);
    Encryptor encryptor(*result.context, public_key, keygen.secret_key());

    Plaintext plain0;
    Plaintext plain1;
    encoder.encode(std::vector<double>{1.0, 2.0, 3.0, 4.0}, parms.scale(), plain0);
    encoder.encode(std::vector<double>{5.0, 6.0, 7.0, 8.0}, parms.scale(), plain1);

    Ciphertext ct0;
    Ciphertext ct1;
    encryptor.encrypt(plain0, ct0);
    encryptor.encrypt(plain1, ct1);

    result.gpu_params = std::make_unique<GpuParameterData>(*result.context, device_id);
    result.evaluator = std::make_unique<GpuEvaluator>(*result.gpu_params);
    result.gpu_ct0 = GpuUploader::upload_ciphertext(ct0, device_id);
    result.gpu_ct1 = GpuUploader::upload_ciphertext(ct1, device_id);
    result.gpu_relin_keys = GpuUploader::upload_relin_keys(relin_keys, device_id);

    return result;
}

int run_nsys_multiply_relin_rescale_probe()
{
    using poseidon::gpu::gpu_check_cuda;

    const int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const std::size_t degree =
        env_size_or("POSEIDON_NSYS_DEGREE", 65536);
    const std::size_t q_count =
        env_size_or("POSEIDON_NSYS_Q_COUNT", 32);
    const std::size_t p_count =
        env_size_or("POSEIDON_NSYS_P_COUNT", 6);
    const std::size_t timing_iterations =
        env_size_or("POSEIDON_NSYS_ITERATIONS", 10);
    const std::size_t warmup_iterations =
        env_size_or("POSEIDON_NSYS_WARMUP", 1);

    if (timing_iterations == 0)
    {
        throw std::invalid_argument(
            "POSEIDON_NSYS_ITERATIONS must be greater than zero");
    }

    auto current_case = prepare_nsys_multiply_relin_rescale_case(
        degree,
        q_count,
        p_count,
        device_id);

    gpu_check_cuda(
        cudaSetDevice(device_id),
        "nsys multiply_relinearize_rescale cudaSetDevice");
    for (std::size_t i = 0; i < warmup_iterations; ++i)
    {
        current_case.run_once();
    }
    gpu_check_cuda(
        cudaDeviceSynchronize(),
        "nsys multiply_relinearize_rescale warmup synchronize");

    const std::string range_name =
        "multiply_relinearize_rescale N=" + std::to_string(degree) +
        " q=" + std::to_string(q_count) +
        " p=" + std::to_string(p_count);

    std::cout << "\n[nsys multiply + relinearize + rescale probe]\n";
    std::cout << "degree                 = " << degree << "\n";
    std::cout << "q_count                = " << q_count << "\n";
    std::cout << "p_count                = " << p_count << "\n";
    std::cout << "warmup iterations      = " << warmup_iterations << "\n";
    std::cout << "timing iterations      = " << timing_iterations << "\n";
    std::cout << "included in capture    = GpuEvaluator::multiply + relinearize + rescale\n";
    std::cout << "excluded from capture  = context/keygen/encode/encrypt/upload/warmup/download\n";
    std::cout << "capture range          = cudaProfilerStart/Stop\n";
    std::cout << "nvtx range             = " << range_name << "\n";

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    gpu_check_cuda(cudaEventCreate(&gpu_start), "nsys cudaEventCreate start");
    gpu_check_cuda(cudaEventCreate(&gpu_stop), "nsys cudaEventCreate stop");

    gpu_check_cuda(cudaProfilerStart(), "nsys cudaProfilerStart");
    nvtxRangePushA(range_name.c_str());
    const auto wall_begin = std::chrono::steady_clock::now();
    gpu_check_cuda(cudaEventRecord(gpu_start), "nsys cudaEventRecord start");
    for (std::size_t i = 0; i < timing_iterations; ++i)
    {
        current_case.run_once();
    }
    gpu_check_cuda(cudaEventRecord(gpu_stop), "nsys cudaEventRecord stop");
    gpu_check_cuda(
        cudaEventSynchronize(gpu_stop),
        "nsys cudaEventSynchronize stop");
    const auto wall_end = std::chrono::steady_clock::now();
    nvtxRangePop();
    gpu_check_cuda(cudaProfilerStop(), "nsys cudaProfilerStop");

    float gpu_event_total_ms = 0.0F;
    gpu_check_cuda(
        cudaEventElapsedTime(&gpu_event_total_ms, gpu_start, gpu_stop),
        "nsys cudaEventElapsedTime");

    gpu_check_cuda(cudaEventDestroy(gpu_start), "nsys cudaEventDestroy start");
    gpu_check_cuda(cudaEventDestroy(gpu_stop), "nsys cudaEventDestroy stop");

    const double wall_total_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    const double wall_avg_ms = wall_total_ms / timing_iterations;
    const double event_avg_ms =
        static_cast<double>(gpu_event_total_ms) / timing_iterations;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "gpu wall total ms      = " << wall_total_ms << "\n";
    std::cout << "gpu wall avg ms        = " << wall_avg_ms << "\n";
    std::cout << "gpu event total ms     = " << gpu_event_total_ms << "\n";
    std::cout << "gpu event avg ms       = " << event_avg_ms << "\n";

    return EXIT_SUCCESS;
}

#if 0
/* Legacy three-way A/B probe retained for rollback experiments. */
int run_nsys_relinearize_probe()
{
    using poseidon::gpu::gpu_check_cuda;

    const int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const std::size_t degree =
        env_size_or("POSEIDON_NSYS_DEGREE", 65536);
    const std::size_t q_count =
        env_size_or("POSEIDON_NSYS_Q_COUNT", 8);
    const std::size_t p_count =
        env_size_or("POSEIDON_NSYS_P_COUNT", 2);
    const std::size_t timing_iterations =
        env_size_or("POSEIDON_NSYS_ITERATIONS", 1);
    const std::size_t warmup_iterations =
        env_size_or("POSEIDON_NSYS_WARMUP", 0);

    if (timing_iterations == 0)
    {
        throw std::invalid_argument(
            "POSEIDON_NSYS_ITERATIONS must be greater than zero");
    }

    auto current_case = prepare_nsys_multiply_relin_rescale_case(
        degree,
        q_count,
        p_count,
        device_id);

    gpu_check_cuda(
        cudaSetDevice(device_id),
        "nsys relinearize cudaSetDevice");
    current_case.prepare_relinearize_input();
    gpu_check_cuda(
        cudaDeviceSynchronize(),
        "nsys relinearize input synchronize");

    ScopedEnvironmentValue disable_paccum(
        kKeySwitchPAccumAllDnumEnv,
        "0");
    const auto warmup_variant = [&](
        const char *p_to_q_fuse_value,
        const char *batch_q_ntt_value)
    {
        ScopedEnvironmentValue select_decomp_variant(
            kKeySwitchFuseDecompQEnv,
            "1");
        ScopedEnvironmentValue select_modup_ntt_head(
            kKeySwitchFuseModupNttHeadEnv,
            "1");
        ScopedEnvironmentValue select_p_to_q_ntt_head(
            kKeySwitchFusePToQNttHeadEnv,
            p_to_q_fuse_value);
        ScopedEnvironmentValue select_q_ntt_batching(
            kKeySwitchBatchQNttComponentsEnv,
            batch_q_ntt_value);
        for (std::size_t i = 0; i < warmup_iterations; ++i)
        {
            current_case.run_relinearize_once();
        }
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            "nsys relinearize variant warmup synchronize");
    };
    warmup_variant("0", "0");
    warmup_variant("0", "1");
    warmup_variant("1", "1");

    const std::string previous_range_name =
        "relinearize.previous-modified N=" + std::to_string(degree) +
        " q=" + std::to_string(q_count) +
        " p=" + std::to_string(p_count);
    const std::string q_ntt_batched_range_name =
        "relinearize.q-ntt-batched N=" + std::to_string(degree) +
        " q=" + std::to_string(q_count) +
        " p=" + std::to_string(p_count);
    const std::string p_to_q_range_name =
        "relinearize.p2q-ntt-head-fused N=" + std::to_string(degree) +
        " q=" + std::to_string(q_count) +
        " p=" + std::to_string(p_count);

    std::cout << "\n[nsys relinearize probe]\n";
    std::cout << "degree                 = " << degree << "\n";
    std::cout << "q_count                = " << q_count << "\n";
    std::cout << "p_count                = " << p_count << "\n";
    std::cout << "warmup iterations      = " << warmup_iterations << "\n";
    std::cout << "timing iterations      = " << timing_iterations << "\n";
    std::cout << "included in capture    = GpuEvaluator::relinearize only\n";
    std::cout << "excluded from capture  = context/keygen/encode/encrypt/upload/input multiply/warmup/download\n";
    std::cout << "capture range          = cudaProfilerStart/Stop\n";
    std::cout << "nvtx previous range    = " << previous_range_name << "\n";
    std::cout << "nvtx batched range     = " << q_ntt_batched_range_name << "\n";
    std::cout << "nvtx new range         = " << p_to_q_range_name << "\n";
    std::cout << "previous-modified      = fused ModUp/NTT head + fused decomp-Q MAC\n";
    std::cout << "q-ntt-batched          = previous-modified + batched two-component Q NTT\n";
    std::cout << "p2q-ntt-head-fused     = q-ntt-batched + fused P->Q/NTT head\n";
    std::cout << "inner nvtx ranges      = keyswitch.intt_switch_poly, keyswitch.dnum.*, keyswitch.finalize.*\n";

    struct VariantTiming
    {
        double wall_total_ms = 0.0;
        float event_total_ms = 0.0F;
    };

    const auto measure_variant = [&](
        const char *p_to_q_fuse_value,
        const char *batch_q_ntt_value,
        const std::string &range_name)
    {
        ScopedEnvironmentValue select_decomp_variant(
            kKeySwitchFuseDecompQEnv,
            "1");
        ScopedEnvironmentValue select_modup_ntt_head(
            kKeySwitchFuseModupNttHeadEnv,
            "1");
        ScopedEnvironmentValue select_p_to_q_ntt_head(
            kKeySwitchFusePToQNttHeadEnv,
            p_to_q_fuse_value);
        ScopedEnvironmentValue select_q_ntt_batching(
            kKeySwitchBatchQNttComponentsEnv,
            batch_q_ntt_value);
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        gpu_check_cuda(cudaEventCreate(&start), "nsys cudaEventCreate variant start");
        gpu_check_cuda(cudaEventCreate(&stop), "nsys cudaEventCreate variant stop");

        nvtxRangePushA(range_name.c_str());
        const auto wall_begin = std::chrono::steady_clock::now();
        gpu_check_cuda(cudaEventRecord(start), "nsys cudaEventRecord variant start");
        for (std::size_t i = 0; i < timing_iterations; ++i)
        {
            current_case.run_relinearize_once();
        }
        gpu_check_cuda(cudaEventRecord(stop), "nsys cudaEventRecord variant stop");
        gpu_check_cuda(
            cudaEventSynchronize(stop),
            "nsys cudaEventSynchronize variant stop");
        const auto wall_end = std::chrono::steady_clock::now();
        nvtxRangePop();

        VariantTiming result;
        result.wall_total_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
        gpu_check_cuda(
            cudaEventElapsedTime(&result.event_total_ms, start, stop),
            "nsys cudaEventElapsedTime variant");
        gpu_check_cuda(cudaEventDestroy(start), "nsys cudaEventDestroy variant start");
        gpu_check_cuda(cudaEventDestroy(stop), "nsys cudaEventDestroy variant stop");
        return result;
    };

    gpu_check_cuda(cudaProfilerStart(), "nsys cudaProfilerStart");
    const VariantTiming previous = measure_variant(
        "0",
        "0",
        previous_range_name);
    const VariantTiming q_ntt_batched = measure_variant(
        "0",
        "1",
        q_ntt_batched_range_name);
    const VariantTiming p_to_q_fused = measure_variant(
        "1",
        "1",
        p_to_q_range_name);
    gpu_check_cuda(cudaProfilerStop(), "nsys cudaProfilerStop");

    const double previous_wall_avg = previous.wall_total_ms / timing_iterations;
    const double previous_event_avg =
        static_cast<double>(previous.event_total_ms) / timing_iterations;
    const double q_ntt_batched_wall_avg =
        q_ntt_batched.wall_total_ms / timing_iterations;
    const double q_ntt_batched_event_avg =
        static_cast<double>(q_ntt_batched.event_total_ms) / timing_iterations;
    const double p_to_q_wall_avg =
        p_to_q_fused.wall_total_ms / timing_iterations;
    const double p_to_q_event_avg =
        static_cast<double>(p_to_q_fused.event_total_ms) / timing_iterations;
    const double batching_delta_ms =
        q_ntt_batched_event_avg - previous_event_avg;
    const double p_to_q_delta_ms =
        p_to_q_event_avg - q_ntt_batched_event_avg;
    const double overall_speedup = p_to_q_event_avg == 0.0
        ? 0.0
        : previous_event_avg / p_to_q_event_avg;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n[relinearize Q-NTT batching and P->Q/NTT-head fusion]\n";
    std::cout << "variant             wall_total_ms    wall_avg_ms    event_total_ms    event_avg_ms\n";
    std::cout << std::left << std::setw(20) << "previous-modified"
              << std::right << std::setw(16) << previous.wall_total_ms
              << std::setw(15) << previous_wall_avg
              << std::setw(18) << previous.event_total_ms
              << std::setw(16) << previous_event_avg << "\n";
    std::cout << std::left << std::setw(20) << "q-ntt-batched"
              << std::right << std::setw(16) << q_ntt_batched.wall_total_ms
              << std::setw(15) << q_ntt_batched_wall_avg
              << std::setw(18) << q_ntt_batched.event_total_ms
              << std::setw(16) << q_ntt_batched_event_avg << "\n";
    std::cout << std::left << std::setw(20) << "p2q-ntt-head-fused"
              << std::right << std::setw(16) << p_to_q_fused.wall_total_ms
              << std::setw(15) << p_to_q_wall_avg
              << std::setw(18) << p_to_q_fused.event_total_ms
              << std::setw(16) << p_to_q_event_avg << "\n";
    std::cout << "batched - previous event avg ms   = " << batching_delta_ms << "\n";
    std::cout << "p2q-fused - batched event avg ms  = " << p_to_q_delta_ms << "\n";
    std::cout << "previous / final speedup          = " << overall_speedup << "x\n";

    return EXIT_SUCCESS;
}
#endif

int run_nsys_relinearize_probe()
{
    using poseidon::gpu::gpu_check_cuda;

    const int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const std::size_t degree =
        env_size_or("POSEIDON_NSYS_DEGREE", 65536);
    const std::size_t q_count =
        env_size_or("POSEIDON_NSYS_Q_COUNT", 8);
    const std::size_t p_count =
        env_size_or("POSEIDON_NSYS_P_COUNT", 2);
    const std::size_t timing_iterations =
        env_size_or("POSEIDON_NSYS_ITERATIONS", 1);
    const std::size_t warmup_iterations =
        env_size_or("POSEIDON_NSYS_WARMUP", 0);

    if (timing_iterations == 0)
    {
        throw std::invalid_argument(
            "POSEIDON_NSYS_ITERATIONS must be greater than zero");
    }

    auto current_case = prepare_nsys_multiply_relin_rescale_case(
        degree,
        q_count,
        p_count,
        device_id);

    gpu_check_cuda(
        cudaSetDevice(device_id),
        "nsys relinearize cudaSetDevice");
    current_case.prepare_relinearize_input();
    gpu_check_cuda(
        cudaDeviceSynchronize(),
        "nsys relinearize input synchronize");

    ScopedEnvironmentValue disable_paccum(
        kKeySwitchPAccumAllDnumEnv,
        "0");
    ScopedEnvironmentValue select_decomp_variant(
        kKeySwitchFuseDecompQEnv,
        "1");
    ScopedEnvironmentValue select_modup_ntt_head(
        kKeySwitchFuseModupNttHeadEnv,
        "1");

    poseidon::Ciphertext current_output;
    poseidon::Ciphertext row_tiled8_output;
    poseidon::Ciphertext final_tail_output;
    {
        ScopedEnvironmentValue select_final_tail(
            kKeySwitchPAccumFinalTailEnv,
            "0");
        ScopedEnvironmentValue select_row_tiled(
            kKeySwitchBconvRowTiledEnv,
            "0");
        ScopedEnvironmentValue select_row_tiled8(
            kKeySwitchBconvRowTiled8Env,
            "0");
        current_case.run_relinearize_once();
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            "nsys relinearize current correctness synchronize");
        poseidon::gpu::GpuUploader::download_ciphertext(
            current_case.gpu_relinearize_result,
            current_output,
            *current_case.context);
    }
    {
        ScopedEnvironmentValue select_final_tail(
            kKeySwitchPAccumFinalTailEnv,
            "0");
        ScopedEnvironmentValue select_row_tiled(
            kKeySwitchBconvRowTiledEnv,
            "0");
        ScopedEnvironmentValue select_row_tiled8(
            kKeySwitchBconvRowTiled8Env,
            "1");
        current_case.run_relinearize_once();
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            "nsys relinearize row-tiled8 correctness synchronize");
        poseidon::gpu::GpuUploader::download_ciphertext(
            current_case.gpu_relinearize_result,
            row_tiled8_output,
            *current_case.context);
    }
    {
        ScopedEnvironmentValue select_final_tail(
            kKeySwitchPAccumFinalTailEnv,
            "1");
        ScopedEnvironmentValue select_row_tiled(
            kKeySwitchBconvRowTiledEnv,
            "0");
        ScopedEnvironmentValue select_row_tiled8(
            kKeySwitchBconvRowTiled8Env,
            "1");
        current_case.run_relinearize_once();
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            "nsys relinearize final-tail correctness synchronize");
        poseidon::gpu::GpuUploader::download_ciphertext(
            current_case.gpu_relinearize_result,
            final_tail_output,
            *current_case.context);
    }
    if (!ciphertext_raw_equal(current_output, row_tiled8_output))
    {
        throw std::runtime_error(
            "row-tiled8 BConv relinearize result differs from original implementation");
    }
    if (!ciphertext_raw_equal(current_output, final_tail_output))
    {
        throw std::runtime_error(
            "all-dnum final-tail relinearize result differs from original implementation");
    }
    const auto warmup_variant = [&](const char *final_tail_value)
    {
        ScopedEnvironmentValue select_final_tail(
            kKeySwitchPAccumFinalTailEnv,
            final_tail_value);
        ScopedEnvironmentValue select_row_tiled(
            kKeySwitchBconvRowTiledEnv,
            "0");
        ScopedEnvironmentValue select_row_tiled8(
            kKeySwitchBconvRowTiled8Env,
            "1");
        for (std::size_t i = 0; i < warmup_iterations; ++i)
        {
            current_case.run_relinearize_once();
        }
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            "nsys relinearize variant warmup synchronize");
    };
    warmup_variant("0");
    warmup_variant("1");

    const std::string row_tiled8_range_name =
        "relinearize.bconv-row-tiled8 N=" + std::to_string(degree) +
        " q=" + std::to_string(q_count) +
        " p=" + std::to_string(p_count);
    const std::string final_tail_range_name =
        "relinearize.all-dnum-final-tail N=" + std::to_string(degree) +
        " q=" + std::to_string(q_count) +
        " p=" + std::to_string(p_count);
    std::cout << "\n[nsys relinearize probe]\n";
    std::cout << "degree                 = " << degree << "\n";
    std::cout << "q_count                = " << q_count << "\n";
    std::cout << "p_count                = " << p_count << "\n";
    std::cout << "warmup iterations      = " << warmup_iterations << "\n";
    std::cout << "timing iterations      = " << timing_iterations << "\n";
    std::cout << "included in capture    = GpuEvaluator::relinearize only\n";
    std::cout << "excluded from capture  = context/keygen/encode/encrypt/upload/input multiply/warmup/download\n";
    std::cout << "capture range          = cudaProfilerStart/Stop\n";
    std::cout << "nvtx tiled8 range      = " << row_tiled8_range_name << "\n";
    std::cout << "nvtx final-tail range  = " << final_tail_range_name << "\n";
    std::cout << "active finalize path   = fused P->Q/NTT head + batched two-component Q NTT\n";
    std::cout << "bconv-row-tiled-8      = default baseline; shared source tile across eight target-limb warps\n";
    std::cout << "all-dnum-final-tail    = experimental; final fused3 NTT and two-component IP across all dnum\n";
    std::cout << "measurement passes     = 2 (reported totals are pass averages)\n";
    std::cout << "measurement order      = row8,final-tail,final-tail,row8\n";
    std::cout << "raw residue comparison = equal\n";
    std::cout << "inner nvtx ranges      = keyswitch.intt_switch_poly, keyswitch.dnum.*, keyswitch.finalize.*\n";

    struct VariantTiming
    {
        double wall_total_ms = 0.0;
        float event_total_ms = 0.0F;
    };

    const auto measure_variant = [&](
        const char *final_tail_value,
        const std::string &range_name)
    {
        ScopedEnvironmentValue select_final_tail(
            kKeySwitchPAccumFinalTailEnv,
            final_tail_value);
        ScopedEnvironmentValue select_row_tiled(
            kKeySwitchBconvRowTiledEnv,
            "0");
        ScopedEnvironmentValue select_row_tiled8(
            kKeySwitchBconvRowTiled8Env,
            "1");
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        gpu_check_cuda(cudaEventCreate(&start), "nsys cudaEventCreate variant start");
        gpu_check_cuda(cudaEventCreate(&stop), "nsys cudaEventCreate variant stop");

        nvtxRangePushA(range_name.c_str());
        const auto wall_begin = std::chrono::steady_clock::now();
        gpu_check_cuda(cudaEventRecord(start), "nsys cudaEventRecord variant start");
        for (std::size_t i = 0; i < timing_iterations; ++i)
        {
            current_case.run_relinearize_once();
        }
        gpu_check_cuda(cudaEventRecord(stop), "nsys cudaEventRecord variant stop");
        gpu_check_cuda(
            cudaEventSynchronize(stop),
            "nsys cudaEventSynchronize variant stop");
        const auto wall_end = std::chrono::steady_clock::now();
        nvtxRangePop();

        VariantTiming result;
        result.wall_total_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
        gpu_check_cuda(
            cudaEventElapsedTime(&result.event_total_ms, start, stop),
            "nsys cudaEventElapsedTime variant");
        gpu_check_cuda(cudaEventDestroy(start), "nsys cudaEventDestroy variant start");
        gpu_check_cuda(cudaEventDestroy(stop), "nsys cudaEventDestroy variant stop");
        return result;
    };

    gpu_check_cuda(cudaProfilerStart(), "nsys cudaProfilerStart");
    const VariantTiming row_tiled8_forward =
        measure_variant("0", row_tiled8_range_name);
    const VariantTiming final_tail_forward =
        measure_variant("1", final_tail_range_name);
    const VariantTiming final_tail_reverse =
        measure_variant("1", final_tail_range_name);
    const VariantTiming row_tiled8_reverse =
        measure_variant("0", row_tiled8_range_name);
    gpu_check_cuda(cudaProfilerStop(), "nsys cudaProfilerStop");

    const auto average_timing = [](
        const VariantTiming &first,
        const VariantTiming &second)
    {
        VariantTiming result;
        result.wall_total_ms =
            (first.wall_total_ms + second.wall_total_ms) * 0.5;
        result.event_total_ms =
            (first.event_total_ms + second.event_total_ms) * 0.5F;
        return result;
    };
    const VariantTiming row_tiled8 =
        average_timing(row_tiled8_forward, row_tiled8_reverse);
    const VariantTiming final_tail =
        average_timing(final_tail_forward, final_tail_reverse);

    const double row_tiled8_wall_avg =
        row_tiled8.wall_total_ms / timing_iterations;
    const double row_tiled8_event_avg =
        static_cast<double>(row_tiled8.event_total_ms) / timing_iterations;
    const double final_tail_wall_avg =
        final_tail.wall_total_ms / timing_iterations;
    const double final_tail_event_avg =
        static_cast<double>(final_tail.event_total_ms) / timing_iterations;
    const double final_tail_delta_ms =
        final_tail_event_avg - row_tiled8_event_avg;
    const double final_tail_speedup = final_tail_event_avg == 0.0
        ? 0.0
        : row_tiled8_event_avg / final_tail_event_avg;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n[relinearize BConv implementation comparison]\n";
    std::cout << "variant             wall_total_ms    wall_avg_ms    event_total_ms    event_avg_ms\n";
    std::cout << std::left << std::setw(20) << "bconv-row-tiled-8"
              << std::right << std::setw(16) << row_tiled8.wall_total_ms
              << std::setw(15) << row_tiled8_wall_avg
              << std::setw(18) << row_tiled8.event_total_ms
              << std::setw(16) << row_tiled8_event_avg << "\n";
    std::cout << std::left << std::setw(20) << "all-dnum-final-tail"
              << std::right << std::setw(16) << final_tail.wall_total_ms
              << std::setw(15) << final_tail_wall_avg
              << std::setw(18) << final_tail.event_total_ms
              << std::setw(16) << final_tail_event_avg << "\n";
    std::cout << "final-tail - row8 event avg ms    = "
              << final_tail_delta_ms << "\n";
    std::cout << "row8 / final-tail speedup         = "
              << final_tail_speedup << "x\n";

    return EXIT_SUCCESS;
}

int run_demo()
{
    using namespace poseidon;
    using namespace poseidon::gpu;

    const int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const auto parms = make_demo_parameters();
    PoseidonContext context(parms);
    use_benchmark_randomness(context);

    std::cout << "===== GPU Ciphertext Elementwise Evaluator Demo =====\n";
    std::cout << "scheme        = CKKS\n";
    std::cout << "degree        = " << parms.degree() << "\n";
    std::cout << "slots         = " << parms.slot() << "\n";
    std::cout << "q_count       = " << parms.q().size() << "\n";
    std::cout << "p_count       = " << parms.p().size() << "\n";
    std::cout << "scale         = " << parms.scale() << "\n";
    std::cout << "device_id     = " << device_id << "\n";

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    constexpr int rotate_step = 1;
    GaloisKeys galois_keys;
    const auto galois_tool = context.crt_context()->galois_tool();
    keygen.create_galois_keys(
        std::vector<std::uint32_t>{
            galois_tool->get_elt_from_step(rotate_step),
            galois_tool->get_elt_from_step(0)},
        galois_keys);

    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key, keygen.secret_key());

    const std::vector<double> input0{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> input1{5.0, 6.0, 7.0, 8.0};

    if constexpr (kRunCorrectnessChecks)
    {
        std::cout << "\n[input slots]\n";
        for (std::size_t i = 0; i < input0.size(); ++i)
        {
            std::cout << "slot[" << i << "] left=" << input0[i]
                      << " right=" << input1[i]
                      << " expected_sum=" << input0[i] + input1[i]
                      << " expected_sub=" << input0[i] - input1[i]
                      << " expected_negate_left=" << -input0[i] << "\n";
        }
    }

    Plaintext plain0;
    Plaintext plain1;
    encoder.encode(input0, parms.scale(), plain0);
    encoder.encode(input1, parms.scale(), plain1);

    Ciphertext ct0;
    Ciphertext ct1;
    encryptor.encrypt(plain0, ct0);
    encryptor.encrypt(plain1, ct1);

    auto cpu_evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);
    GpuParameterData gpu_params;
    {
        std::cout << "[NTT setup] building regular GPU parameters without TAM matrices\n";
        ScopedEnvironmentValue ntt_fused_matrix_stages(
            kNttFusedMatrixStagesEnv,
            "0");
        gpu_params.build_from_poseidon_context(context, device_id);
    }
    auto gpu_ct0 = GpuUploader::upload_ciphertext(ct0, device_id);
    auto gpu_ct1 = GpuUploader::upload_ciphertext(ct1, device_id);
    auto gpu_plain1 = GpuUploader::upload_plaintext(plain1, device_id);
    auto gpu_relin_keys = GpuUploader::upload_relin_keys(relin_keys, device_id);
    auto gpu_galois_keys = GpuUploader::upload_galois_keys(galois_keys, device_id);

    GpuEvaluator gpu_evaluator(gpu_params);

    if constexpr (kRunCorrectnessChecks)
    {
        // Re-enable detailed CPU/GPU residue and decode checks here when needed.
    }

    if constexpr (kRunCorrectnessChecks)
    {
        std::cout << "\n[multiply + relinearize + rescale correctness]\n";
        Ciphertext cpu_multiply_result;
        Ciphertext cpu_relinearize_result;
        Ciphertext cpu_multiply_relin_rescale_result;
        cpu_evaluator->multiply(ct0, ct1, cpu_multiply_result);
        cpu_evaluator->relinearize(
            cpu_multiply_result,
            cpu_relinearize_result,
            relin_keys);
        cpu_evaluator->rescale(
            cpu_relinearize_result,
            cpu_multiply_relin_rescale_result);

        GpuCiphertextData gpu_multiply_result;
        GpuCiphertextData gpu_relinearize_result;
        GpuCiphertextData gpu_multiply_relin_rescale_output;
        gpu_evaluator.multiply(gpu_ct0, gpu_ct1, gpu_multiply_result);
        gpu_evaluator.relinearize(
            gpu_multiply_result,
            gpu_relin_keys,
            gpu_relinearize_result);
        gpu_evaluator.rescale(
            gpu_relinearize_result,
            gpu_multiply_relin_rescale_output);
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            "GpuEvaluator::multiply_relinearize_rescale correctness sync");

        Ciphertext gpu_relinearize_download;
        Ciphertext gpu_multiply_relin_rescale_download;
        GpuUploader::download_ciphertext(
            gpu_relinearize_result,
            gpu_relinearize_download,
            context);
        GpuUploader::download_ciphertext(
            gpu_multiply_relin_rescale_output,
            gpu_multiply_relin_rescale_download,
            context);

        std::cout << "\n[multiply + relinearize raw comparison]\n";
        const bool relin_raw_equal = print_raw_comparison(
            cpu_relinearize_result,
            gpu_relinearize_download,
            8);
        std::cout << "\n[multiply + relinearize + rescale raw comparison]\n";
        const bool rescale_raw_equal = print_raw_comparison(
            cpu_multiply_relin_rescale_result,
            gpu_multiply_relin_rescale_download,
            8);
        if (!relin_raw_equal || !rescale_raw_equal)
        {
            throw std::runtime_error(
                "GPU multiply + relinearize + rescale correctness check failed");
        }

        std::cout << "\n[rotate correctness]\n";
        std::cout << "rotate step = " << rotate_step << "\n";
        Ciphertext cpu_rotate_result;
        cpu_evaluator->rotate(
            ct0,
            cpu_rotate_result,
            rotate_step,
            galois_keys);

        GpuCiphertextData gpu_rotate_output;
        gpu_evaluator.rotate(
            gpu_ct0,
            rotate_step,
            gpu_galois_keys,
            gpu_rotate_output);
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            "GpuEvaluator::rotate correctness sync");

        Ciphertext gpu_rotate_download;
        GpuUploader::download_ciphertext(
            gpu_rotate_output,
            gpu_rotate_download,
            context);

        std::cout << "\n[rotate raw comparison]\n";
        const bool rotate_raw_equal = print_raw_comparison(
            cpu_rotate_result,
            gpu_rotate_download,
            8);
        if (!rotate_raw_equal)
        {
            throw std::runtime_error("GPU rotate correctness check failed");
        }

        std::cout << "\n[conjugate correctness]\n";
        Ciphertext cpu_conjugate_result;
        cpu_evaluator->conjugate(
            ct0,
            galois_keys,
            cpu_conjugate_result);

        GpuCiphertextData gpu_conjugate_output;
        gpu_evaluator.conjugate(
            gpu_ct0,
            gpu_galois_keys,
            gpu_conjugate_output);
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            "GpuEvaluator::conjugate correctness sync");

        Ciphertext gpu_conjugate_download;
        GpuUploader::download_ciphertext(
            gpu_conjugate_output,
            gpu_conjugate_download,
            context);

        std::cout << "\n[conjugate raw comparison]\n";
        const bool conjugate_raw_equal = print_raw_comparison(
            cpu_conjugate_result,
            gpu_conjugate_download,
            8);
        if (!conjugate_raw_equal)
        {
            throw std::runtime_error("GPU conjugate correctness check failed");
        }
    }

    Ciphertext cpu_multiply_plain_result;
    cpu_evaluator->multiply_plain(ct0, plain1, cpu_multiply_plain_result);

    GpuCiphertextData gpu_multiply_plain_output;
    gpu_evaluator.multiply_plain(gpu_ct0, gpu_plain1, gpu_multiply_plain_output);
    gpu_check_cuda(
        cudaDeviceSynchronize(),
        "GpuEvaluator::multiply_plain precompute sync");

    Ciphertext cpu_multiply_result;
    cpu_evaluator->multiply(ct0, ct1, cpu_multiply_result);

    GpuCiphertextData gpu_multiply_output;
    gpu_evaluator.multiply(gpu_ct0, gpu_ct1, gpu_multiply_output);
    gpu_check_cuda(
        cudaDeviceSynchronize(),
        "GpuEvaluator::multiply precompute sync");

    Ciphertext cpu_ntt_inv_result;
    cpu_evaluator->ntt_inv(ct0, cpu_ntt_inv_result);
    Ciphertext cpu_ntt_fwd_result;
    cpu_evaluator->ntt_fwd(cpu_ntt_inv_result, cpu_ntt_fwd_result);

    GpuCiphertextData gpu_ntt_inv_output;
    gpu_evaluator.ntt_inv(gpu_ct0, gpu_ntt_inv_output);
    gpu_check_cuda(
        cudaDeviceSynchronize(),
        "GpuEvaluator::ntt_inv precompute sync");

    const int timing_iterations =
        static_cast<int>(env_size_or("POSEIDON_DEMO_ITERATIONS", 20));
    const std::size_t warmup_iterations =
        env_size_or("POSEIDON_DEMO_WARMUP", 20);
    if (timing_iterations <= 0)
    {
        throw std::invalid_argument(
            "POSEIDON_DEMO_ITERATIONS must be greater than zero");
    }
    if constexpr (kRunOperationTimingSummary)
    {
        Ciphertext cpu_timing_result;
        Ciphertext cpu_chain_multiply_result;
        Ciphertext cpu_chain_relinearize_result;
        Ciphertext cpu_relinearize_timing_result;
        GpuCiphertextData gpu_timing_output;
        GpuCiphertextData gpu_chain_multiply_output;
        GpuCiphertextData gpu_chain_relinearize_output;
        GpuCiphertextData gpu_relinearize_timing_output;
        std::vector<OperationTimingRow> timing_rows;

        auto benchmark_operation =
            [&](const std::string &name, auto cpu_once, auto gpu_once,
                auto correctness_once)
        {
            const std::string correct = correctness_once();
            timing_rows.push_back(
                OperationTimingRow{
                    name,
                    benchmark_cpu_gpu_average(
                        device_id,
                        timing_iterations,
                        cpu_once,
                        gpu_once,
                        name.c_str(),
                        warmup_iterations),
                    correct});
        };

        auto check_ciphertext_operation =
            [&](auto cpu_once, auto gpu_once,
                const Ciphertext &cpu_output,
                const GpuCiphertextData &gpu_output,
                const char *sync_name)
        {
            cpu_once();
            gpu_once();
            gpu_check_cuda(cudaDeviceSynchronize(), sync_name);
            Ciphertext gpu_download;
            GpuUploader::download_ciphertext(
                gpu_output,
                gpu_download,
                context);
            return ciphertext_raw_equal(cpu_output, gpu_download)
                ? std::string("OK")
                : std::string("FAIL");
        };

        auto benchmark_ciphertext_operation =
            [&](const std::string &name, auto cpu_once, auto gpu_once,
                const Ciphertext &cpu_output,
                const GpuCiphertextData &gpu_output)
        {
            benchmark_operation(
                name,
                cpu_once,
                gpu_once,
                [&]()
                {
                    return check_ciphertext_operation(
                        cpu_once,
                        gpu_once,
                        cpu_output,
                        gpu_output,
                        (name + "_correctness").c_str());
                });
        };

        auto preallocate_gpu_timing_output_like =
            [&](const GpuCiphertextData &reference, std::size_t component_count)
        {
            if (reference.empty() || reference.fields_.empty())
            {
                throw std::invalid_argument(
                    "preallocate_gpu_timing_output_like: empty reference");
            }
            gpu_timing_output =
                GpuCiphertextData::allocate_single_device_sharded(
                    reference.meta.degree,
                    reference.meta.q_count,
                    component_count,
                    reference.fields_.front().device_id,
                    reference.polys_.front().shards,
                    reference.meta.p_count);
            gpu_timing_output.meta = reference.meta;
            gpu_timing_output.meta.component_count = component_count;
        };

        poseidon::RNSPoly cpu_bconv_c2_coeff(
            context,
            cpu_multiply_result.parms_id());
        cpu_bconv_c2_coeff.copy(cpu_multiply_result[2]);
        cpu_bconv_c2_coeff.dot_to_coeff();

        const auto bconv_context_data =
            context.crt_context()->get_context_data(
                cpu_multiply_result.parms_id());
        if (!bconv_context_data)
        {
            throw std::invalid_argument(
                "BConv benchmark: invalid multiply result parms_id");
        }
        const auto bconv_rns_qp = bconv_context_data->qp_rns_tool();
        const std::size_t bconv_base_q_size =
            bconv_rns_qp->base_q()->size();
        const std::size_t bconv_base_p_size =
            bconv_rns_qp->base_p()->size();
        if (bconv_base_p_size == 0)
        {
            throw std::invalid_argument(
                "BConv benchmark requires HYBRID P limbs");
        }
        const std::size_t bconv_decomp_count =
            (bconv_base_q_size + bconv_base_p_size - 1) /
            bconv_base_p_size;
        CpuHybridBconvScratch cpu_bconv_scratch(
            bconv_decomp_count,
            parms.degree(),
            bconv_base_q_size,
            bconv_base_p_size);

        auto gpu_bconv_source_view = gpu_multiply_output.make_const_view();
        const auto &gpu_bconv_c2_ntt_shard =
            gpu_bconv_source_view.polys[2].shards.front();
        const auto &gpu_bconv_level_info =
            gpu_params.get_level(gpu_multiply_output.meta.parms_id);
        const auto &gpu_bconv_parameter_shard =
            find_benchmark_parameter_shard(
                gpu_bconv_level_info,
                gpu_bconv_c2_ntt_shard);
        if (gpu_bconv_parameter_shard.hybrid_base_q_count !=
                bconv_base_q_size ||
            gpu_bconv_parameter_shard.hybrid_base_p_count !=
                bconv_base_p_size ||
            gpu_bconv_parameter_shard.hybrid_decomp_count <
                bconv_decomp_count)
        {
            throw std::invalid_argument(
                "BConv benchmark: CPU/GPU HYBRID base shape mismatch");
        }

        GpuHybridBconvScratch gpu_bconv_scratch;
        gpu_bconv_scratch.c2_coeff =
            poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord>(
                checked_benchmark_mul(
                    bconv_base_q_size,
                    parms.degree(),
                    "BConv c2 coeff scratch size overflow"),
                device_id);
        gpu_bconv_scratch.modup_q =
            poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord>(
                checked_benchmark_mul(
                    bconv_base_q_size,
                    parms.degree(),
                    "BConv modup_q scratch size overflow"),
                device_id);
        gpu_bconv_scratch.modup_p =
            poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord>(
                checked_benchmark_mul(
                    bconv_base_p_size,
                    parms.degree(),
                    "BConv modup_p scratch size overflow"),
                device_id);
        prepare_gpu_hybrid_bconv_c2_coeff(
            gpu_bconv_scratch,
            gpu_bconv_c2_ntt_shard,
            gpu_bconv_parameter_shard,
            parms.degree(),
            bconv_base_q_size);
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            "BConv benchmark c2 INTT precompute sync");

        benchmark_ciphertext_operation(
            "multiply_relinearize_rescale",
            [&]()
            {
                cpu_evaluator->multiply(ct0, ct1, cpu_chain_multiply_result);
                cpu_evaluator->relinearize(
                    cpu_chain_multiply_result,
                    cpu_chain_relinearize_result,
                    relin_keys);
                cpu_evaluator->rescale(
                    cpu_chain_relinearize_result,
                    cpu_timing_result);
            },
            [&]()
            {
                gpu_evaluator.multiply(gpu_ct0, gpu_ct1, gpu_chain_multiply_output);
                gpu_evaluator.relinearize(
                    gpu_chain_multiply_output,
                    gpu_relin_keys,
                    gpu_chain_relinearize_output);
                gpu_evaluator.rescale(
                    gpu_chain_relinearize_output,
                    gpu_timing_output);
            },
            cpu_timing_result,
            gpu_timing_output);

        benchmark_ciphertext_operation(
            "relinearize",
            [&]()
            {
                cpu_evaluator->relinearize(
                    cpu_multiply_result,
                    cpu_relinearize_timing_result,
                    relin_keys);
            },
            [&]()
            {
                gpu_evaluator.relinearize(
                    gpu_multiply_output,
                    gpu_relin_keys,
                    gpu_relinearize_timing_output);
            },
            cpu_relinearize_timing_result,
            gpu_relinearize_timing_output);

        benchmark_operation(
            "keyswitch_bconv_modup",
            [&]()
            {
                run_cpu_hybrid_bconv_modup(
                    context,
                    cpu_bconv_c2_coeff,
                    cpu_bconv_scratch);
            },
            [&]()
            {
                run_gpu_hybrid_bconv_modup(
                    gpu_bconv_scratch,
                    gpu_bconv_c2_ntt_shard,
                    gpu_bconv_parameter_shard,
                    parms.degree(),
                    bconv_base_q_size,
                    bconv_base_p_size,
                    bconv_decomp_count);
            },
            [&]()
            {
                run_cpu_hybrid_bconv_modup(
                    context,
                    cpu_bconv_c2_coeff,
                    cpu_bconv_scratch);
                run_gpu_hybrid_bconv_modup(
                    gpu_bconv_scratch,
                    gpu_bconv_c2_ntt_shard,
                    gpu_bconv_parameter_shard,
                    parms.degree(),
                    bconv_base_q_size,
                    bconv_base_p_size,
                    bconv_decomp_count);
                gpu_check_cuda(
                    cudaDeviceSynchronize(),
                    "keyswitch_bconv_modup_correctness");
                const std::size_t last_decomp =
                    bconv_decomp_count == 0 ? 0 : bconv_decomp_count - 1;
                const std::size_t last_decomp_limb_begin =
                    last_decomp * bconv_base_p_size;
                const std::size_t last_decomp_limb_count =
                    std::min(
                        bconv_base_p_size,
                        bconv_base_q_size - last_decomp_limb_begin);
                const std::size_t q_offset =
                    last_decomp * bconv_base_q_size * parms.degree();
                const std::size_t p_offset =
                    last_decomp * bconv_base_p_size * parms.degree();
                const bool q_ok = hybrid_modup_q_matches_generated_reference(
                    gpu_bconv_scratch.modup_q,
                    cpu_bconv_scratch.modup_q,
                    q_offset,
                    last_decomp_limb_begin,
                    last_decomp_limb_count,
                    bconv_base_q_size,
                    parms.degree(),
                    "keyswitch_bconv_modup q download");
                const bool p_ok = device_vector_matches_u64_segment(
                    gpu_bconv_scratch.modup_p,
                    cpu_bconv_scratch.modup_p,
                    p_offset,
                    "keyswitch_bconv_modup p download");
                return q_ok && p_ok ? std::string("OK") : std::string("FAIL");
            });

        preallocate_gpu_timing_output_like(
            gpu_ct0,
            std::max(gpu_ct0.size(), gpu_ct1.size()));
        benchmark_ciphertext_operation(
            "add",
            [&]() { cpu_evaluator->add(ct0, ct1, cpu_timing_result); },
            [&]() { gpu_evaluator.add(gpu_ct0, gpu_ct1, gpu_timing_output); },
            cpu_timing_result,
            gpu_timing_output);

        preallocate_gpu_timing_output_like(
            gpu_ct0,
            std::max(gpu_ct0.size(), gpu_ct1.size()));
        benchmark_ciphertext_operation(
            "sub",
            [&]() { cpu_evaluator->sub(ct0, ct1, cpu_timing_result); },
            [&]() { gpu_evaluator.sub(gpu_ct0, gpu_ct1, gpu_timing_output); },
            cpu_timing_result,
            gpu_timing_output);

        preallocate_gpu_timing_output_like(gpu_ct0, gpu_ct0.size());
        benchmark_ciphertext_operation(
            "negate",
            [&]()
            {
                cpu_timing_result = ct0;
                for (std::size_t i = 0; i < cpu_timing_result.size(); ++i)
                {
                    cpu_timing_result[i].negate();
                }
            },
            [&]() { gpu_evaluator.negate(gpu_ct0, gpu_timing_output); },
            cpu_timing_result,
            gpu_timing_output);

        preallocate_gpu_timing_output_like(gpu_ct0, gpu_ct0.size());
        benchmark_ciphertext_operation(
            "add_plain",
            [&]() { cpu_evaluator->add_plain(ct0, plain1, cpu_timing_result); },
            [&]() { gpu_evaluator.add_plain(gpu_ct0, gpu_plain1, gpu_timing_output); },
            cpu_timing_result,
            gpu_timing_output);

        preallocate_gpu_timing_output_like(gpu_ct0, gpu_ct0.size());
        benchmark_ciphertext_operation(
            "sub_plain",
            [&]()
            {
                cpu_timing_result = ct0;
                cpu_timing_result[0].sub(plain1.poly(), cpu_timing_result[0]);
            },
            [&]() { gpu_evaluator.sub_plain(gpu_ct0, gpu_plain1, gpu_timing_output); },
            cpu_timing_result,
            gpu_timing_output);

        benchmark_ciphertext_operation(
            "multiply_plain",
            [&]() { cpu_evaluator->multiply_plain(ct0, plain1, cpu_timing_result); },
            [&]() { gpu_evaluator.multiply_plain(gpu_ct0, gpu_plain1, gpu_timing_output); },
            cpu_timing_result,
            gpu_timing_output);

        benchmark_ciphertext_operation(
            "rotate",
            [&]()
            {
                cpu_evaluator->rotate(
                    ct0,
                    cpu_timing_result,
                    rotate_step,
                    galois_keys);
            },
            [&]()
            {
                gpu_evaluator.rotate(
                    gpu_ct0,
                    rotate_step,
                    gpu_galois_keys,
                    gpu_timing_output);
            },
            cpu_timing_result,
            gpu_timing_output);

        benchmark_ciphertext_operation(
            "conjugate",
            [&]()
            {
                cpu_evaluator->conjugate(
                    ct0,
                    galois_keys,
                    cpu_timing_result);
            },
            [&]()
            {
                gpu_evaluator.conjugate(
                    gpu_ct0,
                    gpu_galois_keys,
                    gpu_timing_output);
            },
            cpu_timing_result,
            gpu_timing_output);

        benchmark_ciphertext_operation(
            "rescale",
            [&]() { cpu_evaluator->rescale(cpu_multiply_plain_result, cpu_timing_result); },
            [&]() { gpu_evaluator.rescale(gpu_multiply_plain_output, gpu_timing_output); },
            cpu_timing_result,
            gpu_timing_output);

        benchmark_ciphertext_operation(
            "ntt_inv",
            [&]() { cpu_evaluator->ntt_inv(ct0, cpu_timing_result); },
            [&]() { gpu_evaluator.ntt_inv(gpu_ct0, gpu_timing_output); },
            cpu_timing_result,
            gpu_timing_output);

        benchmark_ciphertext_operation(
            "ntt_fwd",
            [&]() { cpu_evaluator->ntt_fwd(cpu_ntt_inv_result, cpu_timing_result); },
            [&]() { gpu_evaluator.ntt_fwd(gpu_ntt_inv_output, gpu_timing_output); },
            cpu_timing_result,
            gpu_timing_output);

        print_operation_timing_table(
            timing_rows,
            parms.degree(),
            parms.q().size(),
            parms.p().size(),
            timing_iterations,
            warmup_iterations);
    }

    if constexpr (kRunNttTimingSummary)
    {
    GpuNTTHandler ntt_handler(gpu_params);
    auto gpu_ntt_fwd_source =
        GpuUploader::upload_ciphertext(cpu_ntt_inv_result, device_id);

    auto allocate_ntt_destination =
        [&](const GpuCiphertextData &source_ciphertext, bool is_ntt_form)
    {
        const auto &reference_layout = source_ciphertext.polys_.at(0);
        GpuCiphertextData destination =
            GpuCiphertextData::allocate_single_device_sharded(
                source_ciphertext.meta.degree,
                source_ciphertext.meta.q_count,
                source_ciphertext.size(),
                device_id,
                reference_layout.shards,
                source_ciphertext.meta.p_count);
        destination.meta = source_ciphertext.meta;
        destination.meta.component_count = source_ciphertext.size();
        destination.meta.is_ntt_form = is_ntt_form;
        return destination;
    };

    GpuCiphertextData ntt_inv_destination =
        allocate_ntt_destination(gpu_ct0, false);
    GpuCiphertextData ntt_fwd_destination =
        allocate_ntt_destination(gpu_ntt_fwd_source, true);

    auto ntt_inv_source_view = gpu_ct0.make_const_view();
    auto ntt_inv_destination_view = ntt_inv_destination.make_view();
    auto ntt_fwd_source_view = gpu_ntt_fwd_source.make_const_view();
    auto ntt_fwd_destination_view = ntt_fwd_destination.make_view();
    const auto &ntt_inv_level_info =
        gpu_params.get_level(gpu_ct0.meta.parms_id);
    const auto &ntt_fwd_level_info =
        gpu_params.get_level(gpu_ntt_fwd_source.meta.parms_id);

    std::vector<NttTimingRow> ntt_timing_rows;
    const bool tensor_ntt_requested =
        !env_flag_enabled(kDemoSkipTensorNttEnv);
    const bool tensor_fp64_ntt_requested =
        !env_flag_enabled(kDemoSkipTensorFp64NttEnv);
    const bool tensor_ntt_supported =
        tensor_ntt_requested &&
        poseidon::gpu::supports_tensor_core_integer_gemm(device_id);
    const bool tensor_fp64_ntt_supported =
        tensor_fp64_ntt_requested &&
        poseidon::gpu::supports_tensor_core_fp64_gemm(device_id);
    constexpr std::size_t regular_ntt_row_count = 10;
    constexpr std::size_t tensor_ntt_row_count = 4;
    const std::size_t ntt_progress_total =
        regular_ntt_row_count + tensor_ntt_row_count;
    std::size_t ntt_progress_current = 0;
    auto benchmark_ntt_mode =
        [&](const std::string &operation,
            const std::string &mode,
            const std::string &algorithm,
            const std::string &fusion_stages,
            const Ciphertext &cpu_reference,
            const GpuCiphertextData &gpu_destination,
            auto gpu_once)
    {
        const std::string timing_name = operation + "_" + mode;
        ++ntt_progress_current;
        print_progress_bar(
            "[NTT timing]",
            ntt_progress_current,
            ntt_progress_total,
            timing_name);
        ScopedEnvironmentValue ntt_algorithm(
            kNttAlgorithmEnv,
            algorithm.c_str());
        ScopedEnvironmentValue ntt_fusion(
            kNttFusionStagesEnv,
            fusion_stages.c_str());

        gpu_once();
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            (timing_name + "_correctness").c_str());
        Ciphertext gpu_download;
        GpuUploader::download_ciphertext(
            gpu_destination,
            gpu_download,
            context);
        const bool correct =
            ciphertext_raw_equal(cpu_reference, gpu_download);

        ntt_timing_rows.push_back(
            NttTimingRow{
                operation,
                mode,
                benchmark_gpu_event_samples(
                    device_id,
                    timing_iterations,
                    gpu_once,
                    timing_name.c_str(),
                    warmup_iterations),
                correct ? "OK" : "FAIL"});

        if (mode == "fused4" || mode == "tensor" || mode == "tensor_fp64")
        {
            const std::string profile_name = timing_name + "_phase_profile";
            const auto phase_stats =
                benchmark_ntt_stage_profile_samples(
                    device_id,
                    timing_iterations,
                    gpu_once,
                    profile_name.c_str(),
                    warmup_iterations,
                    4);
            for (std::size_t phase = 0; phase < phase_stats.size(); ++phase)
            {
                ntt_timing_rows.push_back(
                    NttTimingRow{
                        "  phase" + std::to_string(phase),
                        mode,
                        phase_stats[phase],
                        "",
                        true});
            }
        }
    };

    auto append_skipped_ntt_mode =
        [&](const std::string &operation, const std::string &mode)
    {
        ++ntt_progress_current;
        print_progress_bar(
            "[NTT timing]",
            ntt_progress_current,
            ntt_progress_total,
            operation + "_" + mode + " skipped");
        ntt_timing_rows.push_back(
            NttTimingRow{
                operation,
                mode,
                GpuSampleStats{},
                "/"});
    };

    const int ntt_fusion_modes[] = {1, 2, 3, 4};
    for (int fusion_stages : ntt_fusion_modes)
    {
        const std::string fusion_text = std::to_string(fusion_stages);
        const std::string algorithm =
            fusion_stages == 1 ? "stage" : "fused";
        const std::string mode =
            fusion_stages == 1 ? "stage" : "fused" + fusion_text;
        benchmark_ntt_mode(
            "ntt_inv",
            mode,
            algorithm,
            fusion_text,
            cpu_ntt_inv_result,
            ntt_inv_destination,
            [&]()
            {
                ntt_handler.inverse_ciphertext(
                    ntt_inv_destination_view,
                    ntt_inv_source_view,
                    ntt_inv_level_info);
            });
    }
    benchmark_ntt_mode(
        "ntt_inv",
        "fourstep",
        "fourstep",
        "1",
        cpu_ntt_inv_result,
        ntt_inv_destination,
        [&]()
        {
            ntt_handler.inverse_ciphertext(
                ntt_inv_destination_view,
                ntt_inv_source_view,
                ntt_inv_level_info);
        });

    for (int fusion_stages : ntt_fusion_modes)
    {
        const std::string fusion_text = std::to_string(fusion_stages);
        const std::string algorithm =
            fusion_stages == 1 ? "stage" : "fused";
        const std::string mode =
            fusion_stages == 1 ? "stage" : "fused" + fusion_text;
        benchmark_ntt_mode(
            "ntt_fwd",
            mode,
            algorithm,
            fusion_text,
            cpu_ntt_fwd_result,
            ntt_fwd_destination,
            [&]()
            {
                ntt_handler.forward_ciphertext(
                    ntt_fwd_destination_view,
                    ntt_fwd_source_view,
                    ntt_fwd_level_info);
            });
    }
    benchmark_ntt_mode(
        "ntt_fwd",
        "fourstep",
        "fourstep",
        "1",
        cpu_ntt_fwd_result,
        ntt_fwd_destination,
        [&]()
        {
            ntt_handler.forward_ciphertext(
                ntt_fwd_destination_view,
                ntt_fwd_source_view,
                ntt_fwd_level_info);
        });

    const auto tam_cache_dir = env_string_or(
        kNttFusedMatrixCacheDirEnv,
        kDefaultNttFusedMatrixCacheDir);

    if (tensor_ntt_supported)
    {
        std::cout << "\n[NTT setup] building INT8 tensor GPU parameters with TAM matrices\n"
                  << "[NTT setup] TAM cache dir: " << tam_cache_dir << "\n";

        GpuParameterData tensor_gpu_params;
        {
            ScopedEnvironmentValue ntt_algorithm(kNttAlgorithmEnv, "tensor");
            ScopedEnvironmentValue ntt_fused_matrix_stages(
                kNttFusedMatrixStagesEnv,
                "4");
            ScopedDefaultEnvironmentValue ntt_fused_matrix_cache_dir(
                kNttFusedMatrixCacheDirEnv,
                tam_cache_dir.c_str());
            ScopedDefaultEnvironmentValue ntt_fused_matrix_progress(
                kNttFusedMatrixProgressEnv,
                "1");
            ScopedEnvironmentValue ntt_fused_matrix_fp64_tables(
                kNttFusedMatrixFp64TablesEnv,
                "0");
            ScopedDefaultEnvironmentValue ntt_fused_matrix_max_levels(
                kNttFusedMatrixMaxLevelsEnv,
                "2");
            tensor_gpu_params.build_from_poseidon_context(context, device_id);
        }

        GpuNTTHandler tensor_ntt_handler(tensor_gpu_params);
        const auto &tensor_ntt_inv_level_info =
            tensor_gpu_params.get_level(gpu_ct0.meta.parms_id);
        const auto &tensor_ntt_fwd_level_info =
            tensor_gpu_params.get_level(gpu_ntt_fwd_source.meta.parms_id);

        benchmark_ntt_mode(
            "ntt_inv",
            "tensor",
            "tensor",
            "4",
            cpu_ntt_inv_result,
            ntt_inv_destination,
            [&]()
            {
                tensor_ntt_handler.inverse_ciphertext(
                    ntt_inv_destination_view,
                    ntt_inv_source_view,
                    tensor_ntt_inv_level_info);
            });
        benchmark_ntt_mode(
            "ntt_fwd",
            "tensor",
            "tensor",
            "4",
            cpu_ntt_fwd_result,
            ntt_fwd_destination,
            [&]()
            {
                tensor_ntt_handler.forward_ciphertext(
                    ntt_fwd_destination_view,
                    ntt_fwd_source_view,
                    tensor_ntt_fwd_level_info);
            });
    }
    else
    {
        append_skipped_ntt_mode("ntt_inv", "tensor");
        append_skipped_ntt_mode("ntt_fwd", "tensor");
    }

    if (tensor_fp64_ntt_supported)
    {
        std::cout << "\n[NTT setup] building FP64 tensor GPU parameters with TAM matrices\n"
                  << "[NTT setup] TAM cache dir: " << tam_cache_dir << "\n";

        GpuParameterData tensor_fp64_gpu_params;
        {
            ScopedEnvironmentValue ntt_algorithm(
                kNttAlgorithmEnv,
                "tensor_fp64");
            ScopedEnvironmentValue ntt_fused_matrix_stages(
                kNttFusedMatrixStagesEnv,
                "4");
            ScopedDefaultEnvironmentValue ntt_fused_matrix_cache_dir(
                kNttFusedMatrixCacheDirEnv,
                tam_cache_dir.c_str());
            ScopedDefaultEnvironmentValue ntt_fused_matrix_progress(
                kNttFusedMatrixProgressEnv,
                "1");
            ScopedEnvironmentValue ntt_fused_matrix_fp64_tables(
                kNttFusedMatrixFp64TablesEnv,
                "1");
            ScopedDefaultEnvironmentValue ntt_fused_matrix_max_levels(
                kNttFusedMatrixMaxLevelsEnv,
                "2");
            tensor_fp64_gpu_params.build_from_poseidon_context(context, device_id);
        }

        GpuNTTHandler tensor_fp64_ntt_handler(tensor_fp64_gpu_params);
        const auto &tensor_fp64_ntt_inv_level_info =
            tensor_fp64_gpu_params.get_level(gpu_ct0.meta.parms_id);
        const auto &tensor_fp64_ntt_fwd_level_info =
            tensor_fp64_gpu_params.get_level(gpu_ntt_fwd_source.meta.parms_id);

        benchmark_ntt_mode(
            "ntt_inv",
            "tensor_fp64",
            "tensor_fp64",
            "4",
            cpu_ntt_inv_result,
            ntt_inv_destination,
            [&]()
            {
                tensor_fp64_ntt_handler.inverse_ciphertext(
                    ntt_inv_destination_view,
                    ntt_inv_source_view,
                    tensor_fp64_ntt_inv_level_info);
            });
        benchmark_ntt_mode(
            "ntt_fwd",
            "tensor_fp64",
            "tensor_fp64",
            "4",
            cpu_ntt_fwd_result,
            ntt_fwd_destination,
            [&]()
            {
                tensor_fp64_ntt_handler.forward_ciphertext(
                    ntt_fwd_destination_view,
                    ntt_fwd_source_view,
                    tensor_fp64_ntt_fwd_level_info);
            });
    }
    else
    {
        append_skipped_ntt_mode("ntt_inv", "tensor_fp64");
        append_skipped_ntt_mode("ntt_fwd", "tensor_fp64");
    }

    print_ntt_timing_table(
        ntt_timing_rows,
        parms.degree(),
        parms.q().size(),
        parms.p().size(),
        timing_iterations,
        warmup_iterations);
    }

    // run_parameter_sweep_benchmarks(device_id);

    std::cout << "\n===== DEMO FINISHED =====\n";
    return EXIT_SUCCESS;
}

}  // namespace

int main()
{
    const int cuda_status = check_cuda_runtime();
    if (cuda_status != EXIT_SUCCESS)
    {
        return cuda_status;
    }

    try
    {
        if (env_flag_enabled("POSEIDON_NSYS_MUL_RELIN_RESCALE"))
        {
            return run_nsys_multiply_relin_rescale_probe();
        }
        if (env_flag_enabled("POSEIDON_NSYS_RELINEARIZE"))
        {
            return run_nsys_relinearize_probe();
        }
        if (env_flag_enabled("POSEIDON_NSYS_MULTIPLY_PLAIN"))
        {
            return run_nsys_multiply_plain_probe();
        }
        return run_demo();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[EXCEPTION] " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
