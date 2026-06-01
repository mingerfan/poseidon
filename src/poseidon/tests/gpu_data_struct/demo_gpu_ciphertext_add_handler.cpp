#include "poseidon/ckks_encoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/basics/util/ntt.h"
#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_elementwise_handler.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

#include <cuda_runtime_api.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr int kSkip = 77;

struct TimingResult
{
    double cpu_avg_ms = 0.0;
    double gpu_wall_avg_ms = 0.0;
    double gpu_event_avg_ms = 0.0;
    double wall_speedup = 0.0;
    double event_speedup = 0.0;
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
        /*log_n=*/15,
        /*log_slots=*/14,
        /*log_scale=*/25,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        poseidon::Modulus(0),
        std::vector<poseidon::Modulus>{},
        std::vector<poseidon::Modulus>{},
        poseidon::sec_level_type::none);

    parms.set_log_modulus(
        std::vector<std::uint32_t>(8, 30),
        std::vector<std::uint32_t>{});
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

std::size_t ceil_div(std::size_t numerator, std::size_t denominator)
{
    return denominator == 0
        ? 0
        : (numerator + denominator - 1) / denominator;
}

poseidon::ParametersLiteral make_benchmark_parameters(
    std::size_t degree,
    std::size_t q_count)
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
        std::vector<std::uint32_t>{});
    return parms;
}

void print_cipher_meta(const std::string &name, const poseidon::Ciphertext &cipher)
{
    std::cout << "\n[" << name << " metadata]\n";
    std::cout << "component_count = " << cipher.size() << "\n";
    std::cout << "degree          = " << cipher.poly_modulus_degree() << "\n";
    std::cout << "q_count         = " << cipher.coeff_modulus_size() << "\n";
    std::cout << "is_ntt_form     = " << cipher.is_ntt_form() << "\n";
    std::cout << "scale           = " << cipher.scale() << "\n";
}

void print_gpu_cipher_layout(
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

void print_first_words(
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

void print_raw_comparison(
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
}

void print_decoded_slots(
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
    const char *name)
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
    gpu_once();
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

std::uint64_t add_mod_u64(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t modulus)
{
    const auto sum =
        static_cast<unsigned __int128>(left) + static_cast<unsigned __int128>(right);
    return static_cast<std::uint64_t>(
        sum >= modulus ? sum - modulus : sum);
}

std::uint64_t sub_mod_u64(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t modulus)
{
    return left >= right
        ? left - right
        : static_cast<std::uint64_t>(
              static_cast<unsigned __int128>(left) + modulus - right);
}

void cpu_forward_ntt_stage_inplace(
    std::uint64_t *values,
    std::size_t limb_count,
    std::size_t degree,
    std::size_t m,
    std::size_t gap,
    const poseidon::util::NTTTables *ntt_tables)
{
    for (std::size_t limb = 0; limb < limb_count; ++limb)
    {
        const auto &table = ntt_tables[limb];
        const auto &modulus = table.modulus();
        const auto modulus_value = modulus.value();
        const auto *roots = table.get_from_root_powers();
        auto *limb_values = values + limb * degree;

        for (std::size_t group = 0; group < m; ++group)
        {
            const auto root = roots[m + group];
            const std::size_t group_base = group * (gap << 1);

            for (std::size_t j = 0; j < gap; ++j)
            {
                const std::size_t x_index = group_base + j;
                const std::size_t y_index = x_index + gap;

                const std::uint64_t u = limb_values[x_index];
                const std::uint64_t v =
                    poseidon::util::multiply_uint_mod(
                        limb_values[y_index],
                        root,
                        modulus);

                limb_values[x_index] = add_mod_u64(u, v, modulus_value);
                limb_values[y_index] = sub_mod_u64(u, v, modulus_value);
            }
        }
    }
}

void cpu_forward_ntt_stage_ciphertext_inplace(
    std::vector<std::uint64_t> &values,
    std::size_t component_count,
    std::size_t q_count,
    std::size_t degree,
    std::size_t m,
    std::size_t gap,
    const poseidon::util::NTTTables *ntt_tables)
{
    const std::size_t component_words = q_count * degree;
    for (std::size_t component = 0; component < component_count; ++component)
    {
        cpu_forward_ntt_stage_inplace(
            values.data() + component * component_words,
            q_count,
            degree,
            m,
            gap,
            ntt_tables);
    }
}

void cpu_forward_ntt_ciphertext_inplace(
    std::vector<std::uint64_t> &values,
    std::size_t component_count,
    std::size_t q_count,
    std::size_t degree,
    const poseidon::util::NTTTables *ntt_tables)
{
    for (std::size_t m = 1, gap = degree >> 1;
         m < degree;
         m <<= 1, gap >>= 1)
    {
        cpu_forward_ntt_stage_ciphertext_inplace(
            values,
            component_count,
            q_count,
            degree,
            m,
            gap,
            ntt_tables);
    }
}

const poseidon::gpu::GpuParameterShard *find_parameter_shard_for_stage(
    const poseidon::gpu::GpuLevelInfo &level_info,
    const poseidon::gpu::GpuPolyShardView &poly_shard)
{
    for (const auto &parameter_shard : level_info.shards)
    {
        if (parameter_shard.device_id == poly_shard.device_id &&
            poly_shard.limb_begin >= parameter_shard.limb_begin &&
            poly_shard.limb_begin + poly_shard.limb_count <=
                parameter_shard.limb_begin + parameter_shard.limb_count)
        {
            return &parameter_shard;
        }
    }
    return nullptr;
}

void launch_gpu_forward_ntt_stage_ciphertext(
    poseidon::gpu::GpuCiphertextView &view,
    const poseidon::gpu::GpuLevelInfo &level_info,
    std::size_t degree,
    std::size_t m,
    std::size_t gap)
{
    for (auto &poly : view.polys)
    {
        for (auto &shard : poly.shards)
        {
            const auto *parameter_shard =
                find_parameter_shard_for_stage(level_info, shard);
            if (parameter_shard == nullptr)
            {
                throw std::invalid_argument(
                    "single-stage NTT benchmark cannot find a matching GPU parameter shard");
            }

            poseidon::gpu::kernel::launch_forward_ntt_stage_poly_shard(
                shard,
                *parameter_shard,
                degree,
                m,
                gap);
        }
    }
}

void launch_gpu_forward_ntt_single_kernel_ciphertext(
    poseidon::gpu::GpuCiphertextView &destination_view,
    const poseidon::gpu::GpuConstCiphertextView &source_view,
    const poseidon::gpu::GpuLevelInfo &level_info,
    std::size_t degree,
    int block_size = 256)
{
    if (destination_view.polys.size() != source_view.polys.size())
    {
        throw std::invalid_argument(
            "single-kernel NTT benchmark ciphertext component count mismatch");
    }

    for (std::size_t poly_index = 0;
         poly_index < destination_view.polys.size();
         ++poly_index)
    {
        auto &destination_poly = destination_view.polys[poly_index];
        const auto &source_poly = source_view.polys[poly_index];
        if (destination_poly.shards.size() != source_poly.shards.size())
        {
            throw std::invalid_argument(
                "single-kernel NTT benchmark poly shard count mismatch");
        }

        for (std::size_t shard_index = 0;
             shard_index < destination_poly.shards.size();
             ++shard_index)
        {
            auto &destination_shard = destination_poly.shards[shard_index];
            const auto &source_shard = source_poly.shards[shard_index];
            const auto *parameter_shard =
                find_parameter_shard_for_stage(level_info, destination_shard);
            if (parameter_shard == nullptr)
            {
                throw std::invalid_argument(
                    "single-kernel NTT benchmark cannot find a matching GPU parameter shard");
            }

            poseidon::gpu::kernel::launch_forward_ntt_poly_shard_single_kernel_with_block_size(
                destination_shard,
                source_shard,
                *parameter_shard,
                degree,
                block_size);
        }
    }
}

void benchmark_forward_ntt_single_stage(
    const poseidon::PoseidonContext &context,
    const poseidon::Ciphertext &stage_source,
    const poseidon::gpu::GpuParameterData &gpu_params,
    int device_id,
    int timing_iterations)
{
    using namespace poseidon;
    using namespace poseidon::gpu;

    const std::size_t degree = stage_source.poly_modulus_degree();
    const std::size_t q_count = stage_source.coeff_modulus_size();
    const std::size_t component_count = stage_source.size();
    const std::size_t total_words = component_count * q_count * degree;
    const std::size_t m = 1;
    const std::size_t gap = degree >> 1;

    auto crt_context = context.crt_context();
    if (!crt_context || crt_context->small_ntt_tables() == nullptr)
    {
        throw std::invalid_argument("single-stage NTT benchmark requires CPU NTT tables");
    }
    const auto *ntt_tables = crt_context->small_ntt_tables();

    std::vector<std::uint64_t> cpu_stage_values(total_words);
    std::copy_n(stage_source.data(), total_words, cpu_stage_values.data());

    auto gpu_stage_data = GpuUploader::upload_ciphertext(stage_source, device_id);
    auto gpu_stage_view = gpu_stage_data.make_view();
    const auto &level_info = gpu_params.get_level(stage_source.parms_id());
    if (level_info.shards.empty())
    {
        throw std::invalid_argument("single-stage NTT benchmark requires a GPU parameter shard");
    }

    std::cout << "\n[CPU/GPU forward NTT single-stage timing]\n";
    std::cout << "iterations              = " << timing_iterations << "\n";
    std::cout << "stage m                 = " << m << "\n";
    std::cout << "stage gap               = " << gap << "\n";
    std::cout << "component_count          = " << component_count << "\n";
    std::cout << "q_count                  = " << q_count << "\n";
    std::cout << "degree                   = " << degree << "\n";
    std::cout << "included in timing       = one forward NTT stage over all ciphertext components\n";
    std::cout << "excluded from timing     = upload/download/full NTT loop/result comparison\n";

    cpu_forward_ntt_stage_ciphertext_inplace(
        cpu_stage_values,
        component_count,
        q_count,
        degree,
        m,
        gap,
        ntt_tables);

    const auto cpu_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < timing_iterations; ++i)
    {
        cpu_forward_ntt_stage_ciphertext_inplace(
            cpu_stage_values,
            component_count,
            q_count,
            degree,
            m,
            gap,
            ntt_tables);
    }
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_total_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();

    gpu_check_cuda(cudaSetDevice(device_id), "single-stage NTT timing cudaSetDevice");
    launch_gpu_forward_ntt_stage_ciphertext(
        gpu_stage_view,
        level_info,
        degree,
        m,
        gap);
    gpu_check_cuda(cudaDeviceSynchronize(), "single-stage NTT warmup sync");

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    gpu_check_cuda(cudaEventCreate(&gpu_start), "single-stage NTT cudaEventCreate start");
    gpu_check_cuda(cudaEventCreate(&gpu_stop), "single-stage NTT cudaEventCreate stop");

    const auto gpu_wall_begin = std::chrono::steady_clock::now();
    gpu_check_cuda(cudaEventRecord(gpu_start), "single-stage NTT cudaEventRecord start");
    for (int i = 0; i < timing_iterations; ++i)
    {
        launch_gpu_forward_ntt_stage_ciphertext(
            gpu_stage_view,
            level_info,
            degree,
            m,
            gap);
    }
    gpu_check_cuda(cudaEventRecord(gpu_stop), "single-stage NTT cudaEventRecord stop");
    gpu_check_cuda(cudaEventSynchronize(gpu_stop), "single-stage NTT cudaEventSynchronize stop");
    const auto gpu_wall_end = std::chrono::steady_clock::now();

    float gpu_event_total_ms = 0.0F;
    gpu_check_cuda(
        cudaEventElapsedTime(&gpu_event_total_ms, gpu_start, gpu_stop),
        "single-stage NTT cudaEventElapsedTime");

    gpu_check_cuda(cudaEventDestroy(gpu_start), "single-stage NTT cudaEventDestroy start");
    gpu_check_cuda(cudaEventDestroy(gpu_stop), "single-stage NTT cudaEventDestroy stop");

    std::uint32_t gpu_anchor = 0;
    if (!gpu_stage_view.polys.empty() && !gpu_stage_view.polys.front().shards.empty())
    {
        gpu_check_cuda(
            cudaMemcpy(
                &gpu_anchor,
                gpu_stage_view.polys.front().shards.front().ptr,
                sizeof(gpu_anchor),
                cudaMemcpyDeviceToHost),
            "single-stage NTT anchor download");
    }

    const double gpu_wall_total_ms =
        std::chrono::duration<double, std::milli>(gpu_wall_end - gpu_wall_begin).count();
    const double cpu_avg_ms = cpu_total_ms / timing_iterations;
    const double gpu_wall_avg_ms = gpu_wall_total_ms / timing_iterations;
    const double gpu_event_avg_ms = gpu_event_total_ms / timing_iterations;
    const std::uint64_t cpu_anchor =
        cpu_stage_values.empty() ? 0 : cpu_stage_values.front();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "cpu total ms        = " << cpu_total_ms << "\n";
    std::cout << "cpu avg ms          = " << cpu_avg_ms << "\n";
    std::cout << "gpu wall total ms   = " << gpu_wall_total_ms << "\n";
    std::cout << "gpu wall avg ms     = " << gpu_wall_avg_ms << "\n";
    std::cout << "gpu event total ms  = " << gpu_event_total_ms << "\n";
    std::cout << "gpu event avg ms    = " << gpu_event_avg_ms << "\n";
    std::cout << "speedup wall        = " << cpu_avg_ms / gpu_wall_avg_ms << "x\n";
    std::cout << "speedup cuda-event  = " << cpu_avg_ms / gpu_event_avg_ms << "x\n";
    std::cout << "cpu anchor          = " << cpu_anchor << "\n";
    std::cout << "gpu anchor          = " << gpu_anchor << "\n";
}

TimingResult measure_forward_ntt_single_stage_average(
    const poseidon::PoseidonContext &context,
    const poseidon::Ciphertext &stage_source,
    const poseidon::gpu::GpuParameterData &gpu_params,
    int device_id,
    int timing_iterations)
{
    using namespace poseidon;
    using namespace poseidon::gpu;

    const std::size_t degree = stage_source.poly_modulus_degree();
    const std::size_t q_count = stage_source.coeff_modulus_size();
    const std::size_t component_count = stage_source.size();
    const std::size_t total_words = component_count * q_count * degree;
    const std::size_t m = 1;
    const std::size_t gap = degree >> 1;

    auto crt_context = context.crt_context();
    if (!crt_context || crt_context->small_ntt_tables() == nullptr)
    {
        throw std::invalid_argument("single-stage NTT benchmark requires CPU NTT tables");
    }
    const auto *ntt_tables = crt_context->small_ntt_tables();

    std::vector<std::uint64_t> cpu_stage_values(total_words);
    std::copy_n(stage_source.data(), total_words, cpu_stage_values.data());

    auto gpu_stage_data = GpuUploader::upload_ciphertext(stage_source, device_id);
    auto gpu_stage_view = gpu_stage_data.make_view();
    const auto &level_info = gpu_params.get_level(stage_source.parms_id());

    auto timing = benchmark_cpu_gpu_average(
        device_id,
        timing_iterations,
        [&]()
        {
            cpu_forward_ntt_stage_ciphertext_inplace(
                cpu_stage_values,
                component_count,
                q_count,
                degree,
                m,
                gap,
                ntt_tables);
        },
        [&]()
        {
            launch_gpu_forward_ntt_stage_ciphertext(
                gpu_stage_view,
                level_info,
                degree,
                m,
                gap);
        },
        "sweep forward NTT single stage");

    volatile std::uint64_t cpu_anchor =
        cpu_stage_values.empty() ? 0 : cpu_stage_values.front();
    (void)cpu_anchor;
    return timing;
}

TimingResult measure_forward_ntt_single_kernel_average(
    const poseidon::PoseidonContext &context,
    const poseidon::Ciphertext &stage_source,
    const poseidon::gpu::GpuParameterData &gpu_params,
    int device_id,
    int timing_iterations,
    int block_size = 256)
{
    using namespace poseidon;
    using namespace poseidon::gpu;

    const std::size_t degree = stage_source.poly_modulus_degree();
    const std::size_t q_count = stage_source.coeff_modulus_size();
    const std::size_t component_count = stage_source.size();
    const std::size_t total_words = component_count * q_count * degree;

    auto crt_context = context.crt_context();
    if (!crt_context || crt_context->small_ntt_tables() == nullptr)
    {
        throw std::invalid_argument("single-kernel NTT benchmark requires CPU NTT tables");
    }
    const auto *ntt_tables = crt_context->small_ntt_tables();

    std::vector<std::uint64_t> cpu_ntt_values(total_words);
    std::copy_n(stage_source.data(), total_words, cpu_ntt_values.data());

    auto gpu_ntt_data = GpuUploader::upload_ciphertext(stage_source, device_id);
    auto gpu_ntt_view = gpu_ntt_data.make_view();
    auto gpu_ntt_const_view = gpu_ntt_data.make_const_view();
    const auto &level_info = gpu_params.get_level(stage_source.parms_id());

    auto timing = benchmark_cpu_gpu_average(
        device_id,
        timing_iterations,
        [&]()
        {
            cpu_forward_ntt_ciphertext_inplace(
                cpu_ntt_values,
                component_count,
                q_count,
                degree,
                ntt_tables);
        },
        [&]()
        {
            launch_gpu_forward_ntt_single_kernel_ciphertext(
                gpu_ntt_view,
                gpu_ntt_const_view,
                level_info,
                degree,
                block_size);
        },
        "sweep forward NTT single kernel");

    volatile std::uint64_t cpu_anchor =
        cpu_ntt_values.empty() ? 0 : cpu_ntt_values.front();
    (void)cpu_anchor;
    return timing;
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
        "ntt_single_stage",
        "ntt_fused_full"};

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
    const std::size_t ntt_stage_work_items =
        component_count * q_count * (degree >> 1);
    const std::size_t ntt_full_work_items =
        ntt_stage_work_items * static_cast<std::size_t>(log2_degree(degree));
    const std::size_t ntt_single_kernel_launches = component_count;

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
    append_sweep_row(
        rows,
        experiment,
        degree,
        q_count,
        "ntt_single_stage",
        ntt_stage_work_items,
        elementwise_kernel_launches,
        measure_forward_ntt_single_stage_average(
            context,
            cpu_ntt_stage_source,
            gpu_params,
            device_id,
            timing_iterations));

    append_sweep_row(
        rows,
        experiment,
        degree,
        q_count,
        "ntt_fused_full",
        ntt_full_work_items,
        ntt_single_kernel_launches,
        measure_forward_ntt_single_kernel_average(
            context,
            cpu_ntt_stage_source,
            gpu_params,
            device_id,
            timing_iterations));
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

void run_ntt_fused_block_size_diagnostic(int device_id)
{
    using namespace poseidon;
    using namespace poseidon::gpu;

    constexpr std::size_t degree = 16384;
    constexpr int timing_iterations = 50;
    const std::vector<std::size_t> q_counts{16, 32};
    const std::vector<int> block_sizes{128, 256, 512, 1024};

    for (const auto q_count : q_counts)
    {
        const auto parms = make_benchmark_parameters(degree, q_count);
        PoseidonContext context(parms);

        KeyGenerator keygen(context);
        PublicKey public_key;
        keygen.create_public_key(public_key);

        CKKSEncoder encoder(context);
        Encryptor encryptor(context, public_key, keygen.secret_key());

        Plaintext plain;
        encoder.encode(std::vector<double>{1.0, 2.0, 3.0, 4.0}, parms.scale(), plain);

        Ciphertext ct;
        encryptor.encrypt(plain, ct);

        auto cpu_evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);
        Ciphertext ntt_source;
        cpu_evaluator->ntt_inv(ct, ntt_source);

        GpuParameterData gpu_params(context, device_id);

        const std::size_t component_count = ntt_source.size();
        const std::size_t blocks_per_launch = q_count;
        const std::size_t total_blocks_all_components =
            component_count * q_count;
        const std::size_t work_items =
            component_count * q_count * (degree >> 1) *
            static_cast<std::size_t>(log2_degree(degree));

        std::cout << "\n[ntt_fused_full block-size diagnostic]\n";
        std::cout << "N                    = " << degree << "\n";
        std::cout << "q_count              = " << q_count << "\n";
        std::cout << "component_count      = " << component_count << "\n";
        std::cout << "iterations per row   = " << timing_iterations << "\n";
        std::cout << "blocks per launch    = " << blocks_per_launch
                  << " (c0/c1 are separate launches)\n";
        std::cout << "total logical blocks = " << total_blocks_all_components
                  << " across all component launches\n";
        std::cout << "work items           = " << work_items << "\n";

        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::left
                  << std::setw(12) << "block_size"
                  << std::right
                  << std::setw(8) << "SMs"
                  << std::setw(14) << "act_blk/SM"
                  << std::setw(16) << "act_blk_dev"
                  << std::setw(14) << "waves/launch"
                  << std::setw(12) << "occupancy"
                  << std::setw(14) << "cpu_avg_ms"
                  << std::setw(16) << "gpu_event_ms"
                  << std::setw(13) << "event_spd"
                  << "\n";

        for (const int block_size : block_sizes)
        {
            const auto occupancy =
                poseidon::gpu::kernel::query_forward_ntt_single_kernel_occupancy(
                    device_id,
                    block_size);
            const auto timing = measure_forward_ntt_single_kernel_average(
                context,
                ntt_source,
                gpu_params,
                device_id,
                timing_iterations,
                block_size);
            const std::size_t waves_per_launch = ceil_div(
                blocks_per_launch,
                static_cast<std::size_t>(occupancy.theoretical_active_blocks));

            std::cout << std::left
                      << std::setw(12) << block_size
                      << std::right
                      << std::setw(8) << occupancy.sm_count
                      << std::setw(14) << occupancy.active_blocks_per_sm
                      << std::setw(16) << occupancy.theoretical_active_blocks
                      << std::setw(14) << waves_per_launch
                      << std::setw(11) << occupancy.occupancy * 100.0 << "%"
                      << std::setw(14) << timing.cpu_avg_ms
                      << std::setw(16) << timing.gpu_event_avg_ms
                      << std::setw(12) << timing.event_speedup << "x"
                      << "\n";
        }
    }
    std::cout << std::right;
}

int run_demo()
{
    using namespace poseidon;
    using namespace poseidon::gpu;

    const int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const auto parms = make_demo_parameters();
    PoseidonContext context(parms);

    std::cout << "===== GPU Ciphertext Elementwise Evaluator Demo =====\n";
    std::cout << "scheme        = CKKS\n";
    std::cout << "degree        = " << parms.degree() << "\n";
    std::cout << "slots         = " << parms.slot() << "\n";
    std::cout << "q_count       = " << parms.q().size() << "\n";
    std::cout << "scale         = " << parms.scale() << "\n";
    std::cout << "device_id     = " << device_id << "\n";

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);

    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key, keygen.secret_key());
    Decryptor decryptor(context, keygen.secret_key());

    const std::vector<double> input0{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> input1{5.0, 6.0, 7.0, 8.0};

    std::cout << "\n[input slots]\n";
    for (std::size_t i = 0; i < input0.size(); ++i)
    {
        std::cout << "slot[" << i << "] left=" << input0[i]
                  << " right=" << input1[i]
                  << " expected_sum=" << input0[i] + input1[i]
                  << " expected_sub=" << input0[i] - input1[i]
                  << " expected_negate_left=" << -input0[i] << "\n";
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
    Ciphertext cpu_result;
    cpu_evaluator->add(ct0, ct1, cpu_result);

    print_cipher_meta("ct0", ct0);
    print_cipher_meta("ct1", ct1);
    print_cipher_meta("cpu_result", cpu_result);
    print_first_words("ct0", ct0, 8);
    print_first_words("ct1", ct1, 8);
    print_first_words("cpu_result", cpu_result, 8);

    std::cout << "\n[GPU upload]\n";
    GpuParameterData gpu_params(context, device_id);
    auto gpu_ct0 = GpuUploader::upload_ciphertext(ct0, device_id);
    auto gpu_ct1 = GpuUploader::upload_ciphertext(ct1, device_id);
    auto gpu_plain1 = GpuUploader::upload_plaintext(plain1, device_id);

    print_gpu_cipher_layout("gpu_ct0", gpu_ct0);
    print_gpu_cipher_layout("gpu_ct1", gpu_ct1);

    GpuEvaluator gpu_evaluator(gpu_params);
    GpuCiphertextData gpu_output;

    std::cout << "\n[GPU evaluator add]\n";
    gpu_evaluator.add(gpu_ct0, gpu_ct1, gpu_output);
    std::cout << "GpuEvaluator::add finished\n";
    print_gpu_cipher_layout("gpu_output_after_add", gpu_output);

    Ciphertext gpu_result;
    GpuUploader::download_ciphertext(gpu_output, gpu_result, context);

    print_cipher_meta("gpu_result", gpu_result);
    print_first_words("gpu_result", gpu_result, 8);
    print_raw_comparison(cpu_result, gpu_result, 8);

    std::cout << "\n[CPU/GPU evaluator sub]\n";
    Ciphertext cpu_sub_result;
    cpu_evaluator->sub(ct0, ct1, cpu_sub_result);

    GpuCiphertextData gpu_sub_output;
    gpu_evaluator.sub(gpu_ct0, gpu_ct1, gpu_sub_output);

    Ciphertext gpu_sub_result;
    GpuUploader::download_ciphertext(gpu_sub_output, gpu_sub_result, context);

    print_cipher_meta("cpu_sub_result", cpu_sub_result);
    print_cipher_meta("gpu_sub_result", gpu_sub_result);
    print_first_words("cpu_sub_result", cpu_sub_result, 8);
    print_first_words("gpu_sub_result", gpu_sub_result, 8);
    print_raw_comparison(cpu_sub_result, gpu_sub_result, 8);

    std::cout << "\n[CPU/GPU evaluator negate]\n";
    Ciphertext cpu_negate_result = ct0;
    for (std::size_t i = 0; i < cpu_negate_result.size(); ++i)
    {
        cpu_negate_result[i].negate();
    }

    GpuCiphertextData gpu_negate_output;
    gpu_evaluator.negate(gpu_ct0, gpu_negate_output);

    Ciphertext gpu_negate_result;
    GpuUploader::download_ciphertext(gpu_negate_output, gpu_negate_result, context);

    print_cipher_meta("cpu_negate_result", cpu_negate_result);
    print_cipher_meta("gpu_negate_result", gpu_negate_result);
    print_first_words("cpu_negate_result", cpu_negate_result, 8);
    print_first_words("gpu_negate_result", gpu_negate_result, 8);
    print_raw_comparison(cpu_negate_result, gpu_negate_result, 8);

    std::cout << "\n[CPU/GPU evaluator multiply_plain]\n";

    Ciphertext cpu_multiply_plain_result;
    cpu_evaluator->multiply_plain(ct0, plain1, cpu_multiply_plain_result);

    GpuCiphertextData gpu_multiply_plain_output;
    gpu_evaluator.multiply_plain(gpu_ct0, gpu_plain1, gpu_multiply_plain_output);

    Ciphertext gpu_multiply_plain_result;
    GpuUploader::download_ciphertext(
        gpu_multiply_plain_output,
        gpu_multiply_plain_result,
        context);

    print_cipher_meta("cpu_multiply_plain_result", cpu_multiply_plain_result);
    print_cipher_meta("gpu_multiply_plain_result", gpu_multiply_plain_result);
    print_first_words("cpu_multiply_plain_result", cpu_multiply_plain_result, 8);
    print_first_words("gpu_multiply_plain_result", gpu_multiply_plain_result, 8);
    print_raw_comparison(cpu_multiply_plain_result, gpu_multiply_plain_result, 8);

    std::cout << "\n[CPU/GPU evaluator rescale]\n";

    Ciphertext cpu_rescale_result;
    cpu_evaluator->rescale(cpu_multiply_plain_result, cpu_rescale_result);

    GpuCiphertextData gpu_rescale_output;
    gpu_evaluator.rescale(gpu_multiply_plain_output, gpu_rescale_output);
    gpu_check_cuda(cudaDeviceSynchronize(), "GpuEvaluator::rescale sync");

    Ciphertext gpu_rescale_result;
    GpuUploader::download_ciphertext(
        gpu_rescale_output,
        gpu_rescale_result,
        context);

    print_cipher_meta("cpu_rescale_result", cpu_rescale_result);
    print_cipher_meta("gpu_rescale_result", gpu_rescale_result);
    print_first_words("cpu_rescale_result", cpu_rescale_result, 8);
    print_first_words("gpu_rescale_result", gpu_rescale_result, 8);
    print_raw_comparison(cpu_rescale_result, gpu_rescale_result, 8);

    std::cout << "\n[CPU/GPU evaluator add_plain]\n";

    Ciphertext cpu_add_plain_result;
    cpu_evaluator->add_plain(ct0, plain1, cpu_add_plain_result);

    GpuCiphertextData gpu_add_plain_output;
    gpu_evaluator.add_plain(gpu_ct0, gpu_plain1, gpu_add_plain_output);

    Ciphertext gpu_add_plain_result;
    GpuUploader::download_ciphertext(gpu_add_plain_output, gpu_add_plain_result, context);

    print_cipher_meta("cpu_add_plain_result", cpu_add_plain_result);
    print_cipher_meta("gpu_add_plain_result", gpu_add_plain_result);
    print_first_words("cpu_add_plain_result", cpu_add_plain_result, 8);
    print_first_words("gpu_add_plain_result", gpu_add_plain_result, 8);
    print_raw_comparison(cpu_add_plain_result, gpu_add_plain_result, 8);

    std::cout << "\n[CPU/GPU evaluator sub_plain]\n";

    Ciphertext cpu_sub_plain_result = ct0;
    cpu_sub_plain_result[0].sub(plain1.poly(), cpu_sub_plain_result[0]);

    GpuCiphertextData gpu_sub_plain_output;
    gpu_evaluator.sub_plain(gpu_ct0, gpu_plain1, gpu_sub_plain_output);

    Ciphertext gpu_sub_plain_result;
    GpuUploader::download_ciphertext(gpu_sub_plain_output, gpu_sub_plain_result, context);

    print_cipher_meta("cpu_sub_plain_result", cpu_sub_plain_result);
    print_cipher_meta("gpu_sub_plain_result", gpu_sub_plain_result);
    print_first_words("cpu_sub_plain_result", cpu_sub_plain_result, 8);
    print_first_words("gpu_sub_plain_result", gpu_sub_plain_result, 8);
    print_raw_comparison(cpu_sub_plain_result, gpu_sub_plain_result, 8);

    std::cout << "\n[CPU/GPU evaluator ntt_inv]\n";

    Ciphertext cpu_ntt_inv_result;
    cpu_evaluator->ntt_inv(ct0, cpu_ntt_inv_result);

    GpuCiphertextData gpu_ntt_inv_output;
    gpu_evaluator.ntt_inv(gpu_ct0, gpu_ntt_inv_output);
    gpu_check_cuda(cudaDeviceSynchronize(), "GpuEvaluator::ntt_inv sync");

    Ciphertext gpu_ntt_inv_result;
    GpuUploader::download_ciphertext(gpu_ntt_inv_output, gpu_ntt_inv_result, context);

    print_cipher_meta("cpu_ntt_inv_result", cpu_ntt_inv_result);
    print_cipher_meta("gpu_ntt_inv_result", gpu_ntt_inv_result);
    print_first_words("cpu_ntt_inv_result", cpu_ntt_inv_result, 8);
    print_first_words("gpu_ntt_inv_result", gpu_ntt_inv_result, 8);
    print_raw_comparison(cpu_ntt_inv_result, gpu_ntt_inv_result, 8);

    std::cout << "\n[CPU/GPU evaluator ntt_fwd roundtrip]\n";

    Ciphertext cpu_ntt_fwd_result;
    cpu_evaluator->ntt_fwd(cpu_ntt_inv_result, cpu_ntt_fwd_result);

    GpuCiphertextData gpu_ntt_fwd_output;
    gpu_evaluator.ntt_fwd(gpu_ntt_inv_output, gpu_ntt_fwd_output);
    gpu_check_cuda(cudaDeviceSynchronize(), "GpuEvaluator::ntt_fwd sync");

    Ciphertext gpu_ntt_fwd_result;
    GpuUploader::download_ciphertext(gpu_ntt_fwd_output, gpu_ntt_fwd_result, context);

    print_cipher_meta("cpu_ntt_fwd_result", cpu_ntt_fwd_result);
    print_cipher_meta("gpu_ntt_fwd_result", gpu_ntt_fwd_result);
    print_first_words("cpu_ntt_fwd_result", cpu_ntt_fwd_result, 8);
    print_first_words("gpu_ntt_fwd_result", gpu_ntt_fwd_result, 8);
    print_raw_comparison(cpu_ntt_fwd_result, gpu_ntt_fwd_result, 8);

    std::cout << "\n[CPU/GPU single-kernel ntt_fwd roundtrip]\n";

    GpuCiphertextData gpu_ntt_single_kernel_output =
        GpuCiphertextData::allocate_single_device_sharded(
            gpu_ntt_inv_output.meta.degree,
            gpu_ntt_inv_output.meta.q_count,
            gpu_ntt_inv_output.meta.component_count,
            device_id,
            gpu_ntt_inv_output.polys_.at(0).shards,
            gpu_ntt_inv_output.meta.p_count);
    gpu_ntt_single_kernel_output.meta = gpu_ntt_inv_output.meta;
    gpu_ntt_single_kernel_output.meta.is_ntt_form = true;

    auto gpu_ntt_single_kernel_view =
        gpu_ntt_single_kernel_output.make_view();
    auto gpu_ntt_inv_const_view = gpu_ntt_inv_output.make_const_view();
    const auto &ntt_level_info =
        gpu_params.get_level(cpu_ntt_inv_result.parms_id());
    launch_gpu_forward_ntt_single_kernel_ciphertext(
        gpu_ntt_single_kernel_view,
        gpu_ntt_inv_const_view,
        ntt_level_info,
        cpu_ntt_inv_result.poly_modulus_degree());
    gpu_check_cuda(
        cudaDeviceSynchronize(),
        "single-kernel ntt_fwd roundtrip sync");

    Ciphertext gpu_ntt_single_kernel_result;
    GpuUploader::download_ciphertext(
        gpu_ntt_single_kernel_output,
        gpu_ntt_single_kernel_result,
        context);

    print_cipher_meta("gpu_ntt_single_kernel_result", gpu_ntt_single_kernel_result);
    print_first_words("gpu_ntt_single_kernel_result", gpu_ntt_single_kernel_result, 8);
    print_raw_comparison(cpu_ntt_fwd_result, gpu_ntt_single_kernel_result, 8);

    Plaintext cpu_plain_result;
    Plaintext gpu_plain_result;
    decryptor.decrypt(cpu_result, cpu_plain_result);
    decryptor.decrypt(gpu_result, gpu_plain_result);

    std::vector<double> cpu_slots;
    std::vector<double> gpu_slots;
    encoder.decode(cpu_plain_result, cpu_slots);
    encoder.decode(gpu_plain_result, gpu_slots);

    print_decoded_slots("cpu_result", cpu_slots, 8);
    print_decoded_slots("gpu_result", gpu_slots, 8);

    constexpr int timing_iterations = 200;
    Ciphertext cpu_timing_result;
    GpuCiphertextData gpu_timing_output;

    auto benchmark_operation =
        [&](const std::string &name, auto cpu_once, auto gpu_once)
    {
        std::cout << "\n[" << name << " operation timing]\n";
        std::cout << "iterations              = " << timing_iterations << "\n";
        std::cout << "included in timing       = top-level operation call\n";
        std::cout << "excluded from timing     = encode/encrypt/upload/download/decrypt/decode\n";

        cpu_once();

        const auto cpu_begin = std::chrono::steady_clock::now();
        for (int i = 0; i < timing_iterations; ++i)
        {
            cpu_once();
        }
        const auto cpu_end = std::chrono::steady_clock::now();
        const double cpu_total_ms =
            std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();

        gpu_check_cuda(cudaSetDevice(device_id), "timing cudaSetDevice");
        gpu_once();
        gpu_check_cuda(cudaDeviceSynchronize(), "timing warmup sync");

        cudaEvent_t gpu_start = nullptr;
        cudaEvent_t gpu_stop = nullptr;
        gpu_check_cuda(cudaEventCreate(&gpu_start), "timing cudaEventCreate start");
        gpu_check_cuda(cudaEventCreate(&gpu_stop), "timing cudaEventCreate stop");

        const auto gpu_wall_begin = std::chrono::steady_clock::now();
        gpu_check_cuda(cudaEventRecord(gpu_start), "timing cudaEventRecord start");
        for (int i = 0; i < timing_iterations; ++i)
        {
            gpu_once();
        }
        gpu_check_cuda(cudaEventRecord(gpu_stop), "timing cudaEventRecord stop");
        gpu_check_cuda(cudaEventSynchronize(gpu_stop), "timing cudaEventSynchronize stop");
        const auto gpu_wall_end = std::chrono::steady_clock::now();

        float gpu_event_total_ms = 0.0F;
        gpu_check_cuda(
            cudaEventElapsedTime(&gpu_event_total_ms, gpu_start, gpu_stop),
            "timing cudaEventElapsedTime");

        gpu_check_cuda(cudaEventDestroy(gpu_start), "timing cudaEventDestroy start");
        gpu_check_cuda(cudaEventDestroy(gpu_stop), "timing cudaEventDestroy stop");

        const double gpu_wall_total_ms =
            std::chrono::duration<double, std::milli>(gpu_wall_end - gpu_wall_begin).count();
        const double cpu_avg_ms = cpu_total_ms / timing_iterations;
        const double gpu_wall_avg_ms = gpu_wall_total_ms / timing_iterations;
        const double gpu_event_avg_ms = gpu_event_total_ms / timing_iterations;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "cpu total ms        = " << cpu_total_ms << "\n";
        std::cout << "cpu avg ms          = " << cpu_avg_ms << "\n";
        std::cout << "gpu wall total ms   = " << gpu_wall_total_ms << "\n";
        std::cout << "gpu wall avg ms     = " << gpu_wall_avg_ms << "\n";
        std::cout << "gpu event total ms  = " << gpu_event_total_ms << "\n";
        std::cout << "gpu event avg ms    = " << gpu_event_avg_ms << "\n";
        std::cout << "speedup wall        = " << cpu_avg_ms / gpu_wall_avg_ms << "x\n";
        std::cout << "speedup cuda-event  = " << cpu_avg_ms / gpu_event_avg_ms << "x\n";
    };

    benchmark_operation(
        "add",
        [&]() { cpu_evaluator->add(ct0, ct1, cpu_timing_result); },
        [&]() { gpu_evaluator.add(gpu_ct0, gpu_ct1, gpu_timing_output); });

    benchmark_operation(
        "sub",
        [&]() { cpu_evaluator->sub(ct0, ct1, cpu_timing_result); },
        [&]() { gpu_evaluator.sub(gpu_ct0, gpu_ct1, gpu_timing_output); });

    benchmark_operation(
        "negate",
        [&]()
        {
            cpu_timing_result = ct0;
            for (std::size_t i = 0; i < cpu_timing_result.size(); ++i)
            {
                cpu_timing_result[i].negate();
            }
        },
        [&]() { gpu_evaluator.negate(gpu_ct0, gpu_timing_output); });

    benchmark_operation(
        "add_plain",
        [&]() { cpu_evaluator->add_plain(ct0, plain1, cpu_timing_result); },
        [&]() { gpu_evaluator.add_plain(gpu_ct0, gpu_plain1, gpu_timing_output); });

    benchmark_operation(
        "sub_plain",
        [&]()
        {
            cpu_timing_result = ct0;
            cpu_timing_result[0].sub(plain1.poly(), cpu_timing_result[0]);
        },
        [&]() { gpu_evaluator.sub_plain(gpu_ct0, gpu_plain1, gpu_timing_output); });

    benchmark_operation(
        "multiply_plain",
        [&]() { cpu_evaluator->multiply_plain(ct0, plain1, cpu_timing_result); },
        [&]() { gpu_evaluator.multiply_plain(gpu_ct0, gpu_plain1, gpu_timing_output); });

    benchmark_operation(
        "rescale",
        [&]() { cpu_evaluator->rescale(cpu_multiply_plain_result, cpu_timing_result); },
        [&]() { gpu_evaluator.rescale(gpu_multiply_plain_output, gpu_timing_output); });

    benchmark_operation(
        "ntt_inv",
        [&]() { cpu_evaluator->ntt_inv(ct0, cpu_timing_result); },
        [&]() { gpu_evaluator.ntt_inv(gpu_ct0, gpu_timing_output); });

    benchmark_operation(
        "ntt_fwd",
        [&]() { cpu_evaluator->ntt_fwd(cpu_ntt_inv_result, cpu_timing_result); },
        [&]() { gpu_evaluator.ntt_fwd(gpu_ntt_inv_output, gpu_timing_output); });

    benchmark_forward_ntt_single_stage(
        context,
        cpu_ntt_inv_result,
        gpu_params,
        device_id,
        timing_iterations);

    run_parameter_sweep_benchmarks(device_id);
    run_ntt_fused_block_size_diagnostic(device_id);

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
        return run_demo();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[EXCEPTION] " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
