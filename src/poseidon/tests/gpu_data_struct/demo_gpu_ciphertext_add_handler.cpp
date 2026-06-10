#include "poseidon/ckks_encoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_elementwise_handler.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

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
        std::vector<std::uint32_t>(32, 30),
        std::vector<std::uint32_t>(6, 30));
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
    GpuParameterData gpu_params(context, device_id);
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

    Ciphertext cpu_multiply_plain_result;
    cpu_evaluator->multiply_plain(ct0, plain1, cpu_multiply_plain_result);

    GpuCiphertextData gpu_multiply_plain_output;
    gpu_evaluator.multiply_plain(gpu_ct0, gpu_plain1, gpu_multiply_plain_output);
    gpu_check_cuda(
        cudaDeviceSynchronize(),
        "GpuEvaluator::multiply_plain precompute sync");

    Ciphertext cpu_ntt_inv_result;
    cpu_evaluator->ntt_inv(ct0, cpu_ntt_inv_result);

    GpuCiphertextData gpu_ntt_inv_output;
    gpu_evaluator.ntt_inv(gpu_ct0, gpu_ntt_inv_output);
    gpu_check_cuda(
        cudaDeviceSynchronize(),
        "GpuEvaluator::ntt_inv precompute sync");

    constexpr int timing_iterations = 20;
    Ciphertext cpu_timing_result;
    Ciphertext cpu_chain_multiply_result;
    Ciphertext cpu_chain_relinearize_result;
    GpuCiphertextData gpu_timing_output;
    GpuCiphertextData gpu_chain_multiply_output;
    GpuCiphertextData gpu_chain_relinearize_output;

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
        });

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
        });

    benchmark_operation(
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
        });

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
