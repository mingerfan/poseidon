#include "gpu_encode_experiment.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/basics/util/croots.h"
#include "poseidon/gpu/gpu_tensor_core_gemm.h"
#include "json.hpp"
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>

using namespace poseidon;
using namespace poseidon::gpu;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace {
class BenchmarkMemoryPool {
    rmm::mr::device_memory_resource *previous_ = nullptr;
    std::unique_ptr<rmm::mr::cuda_memory_resource> upstream_;
    std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool_;
public:
    explicit BenchmarkMemoryPool(std::size_t initial_bytes) {
        if (!initial_bytes) return;
        previous_ = rmm::mr::get_current_device_resource();
        upstream_ = std::make_unique<rmm::mr::cuda_memory_resource>();
        pool_ = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
            upstream_.get(), initial_bytes);
        rmm::mr::set_per_device_resource(rmm::cuda_device_id{0}, pool_.get());
    }
    ~BenchmarkMemoryPool() {
        if (pool_) rmm::mr::set_per_device_resource(rmm::cuda_device_id{0}, previous_);
    }
};
struct PinnedInput {
    double *data = nullptr;
    explicit PinnedInput(std::size_t count) {
        gpu_check_cuda(cudaMallocHost(&data, count * sizeof(double)), "pinned input allocation");
    }
    ~PinnedInput() { if (data) cudaFreeHost(data); }
};
struct Event {
    cudaEvent_t event{};
    Event() { gpu_check_cuda(cudaEventCreate(&event), "event create"); }
    ~Event() { cudaEventDestroy(event); }
    void record() { gpu_check_cuda(cudaEventRecord(event, gpu_execution_stream()), "event record"); }
};
double milliseconds(const Event &start, const Event &end) {
    float value = 0;
    gpu_check_cuda(cudaEventElapsedTime(&value, start.event, end.event), "event elapsed");
    return value;
}
Json statistics(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();
    return {{"median_ms", (samples[(n - 1) / 2] + samples[n / 2]) / 2},
            {"min_ms", samples.front()}, {"max_ms", samples.back()},
            {"p95_ms", samples[static_cast<std::size_t>(std::ceil(n * .95)) - 1]},
            {"samples_ms", samples}};
}

Json measure(GpuEncodeExperiment &encoder, DeviceVector<double> &input,
    const PinnedInput &host, const GpuParameterShard &parameters, double scale,
    GpuEncodeExperiment::Backend backend, bool batched, bool upload, int warmup, int repeat)
{
    std::vector<double> total, preparation, ntt, transfer, wall;
    Event begin, uploaded, prepared, end;
    for (int i = -warmup; i < repeat; ++i) {
        const auto start = Clock::now();
        begin.record();
        if (upload) gpu_check_cuda(cudaMemcpyAsync(input.data(), host.data,
            input.size() * sizeof(double), cudaMemcpyHostToDevice, gpu_execution_stream()), "input upload");
        uploaded.record();
        encoder.prepare(input.data(), scale, parameters, batched);
        prepared.record();
        encoder.ntt(parameters, backend, batched);
        end.record();
        gpu_check_cuda(cudaEventSynchronize(end.event), "encode completion");
        const auto finish = Clock::now();
        encoder.check_result(); // Error materialization outside the reported timings.
        if (i >= 0) {
            total.push_back(milliseconds(begin, end));
            transfer.push_back(milliseconds(begin, uploaded));
            preparation.push_back(milliseconds(uploaded, prepared));
            ntt.push_back(milliseconds(prepared, end));
            wall.push_back(std::chrono::duration<double, std::milli>(finish - start).count());
        }
    }
    return {{"total", statistics(total)}, {"prepare_fft_rns", statistics(preparation)},
            {"ntt", statistics(ntt)}, {"h2d", statistics(transfer)}, {"wall", statistics(wall)}};
}

Json validate(const std::vector<GpuWord> &actual, const std::vector<GpuWord> &reference,
              const PinnedInput &input, const PoseidonContext &context,
              std::size_t batch, std::size_t n, std::size_t limbs, double scale,
              const Json &reference_validation)
{
    if (!reference.empty() && actual != reference)
        throw std::runtime_error("single/batched or CUDA/Tensor NTT residues differ");
    if (!reference.empty()) {
        Json result = reference_validation;
        result["exact_gpu_reference_match"] = true;
        return result;
    }
    CKKSEncoder cpu(context);
    const auto id = context.crt_context()->first_context_data()->parms_id();
    double max_error = 0, max_cpu_difference = 0;
    std::size_t cpu_residue_differences = 0;
    std::size_t decoded_plaintexts = 0;
    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<double> slots(input.data + b * n / 2, input.data + (b + 1) * n / 2);
        Plaintext expected, gpu_plain;
        cpu.encode(slots, id, scale, expected);
        gpu_plain.resize(context, id, n * limbs);
        gpu_plain.parms_id() = id;
        gpu_plain.scale() = scale;
        std::size_t differences = 0;
        for (std::size_t i = 0; i < n * limbs; ++i) {
            gpu_plain.data()[i] = actual[b * n * limbs + i];
            differences += gpu_plain.data()[i] != expected.data()[i];
        }
        cpu_residue_differences += differences;
        // Full CPU residue comparison already proves equivalence. Decode the
        // endpoints and every plaintext whose FFT rounding differs from CPU.
        if (b != 0 && b + 1 != batch && differences == 0) continue;
        ++decoded_plaintexts;
        std::vector<double> decoded, cpu_decoded;
        cpu.decode(gpu_plain, decoded);
        cpu.decode(expected, cpu_decoded);
        for (std::size_t i = 0; i < n / 2; ++i) {
            if (!std::isfinite(decoded[i]) || !std::isfinite(cpu_decoded[i]))
                throw std::runtime_error("non-finite decoded output");
            max_error = std::max(max_error, std::abs(decoded[i] - slots[i]));
            max_cpu_difference = std::max(max_cpu_difference, std::abs(decoded[i] - cpu_decoded[i]));
        }
    }
    // FFT ordering may change rounding by one coefficient. Test decoded values
    // against slots; exact residues are required across our GPU execution modes.
    const double tolerance = std::max(1e-8, 64.0 * std::sqrt(static_cast<double>(n)) / scale);
    if (max_error > tolerance || max_cpu_difference > tolerance)
        throw std::runtime_error("GPU encoder exceeded decoded-error tolerance");
    return {{"checked_plaintexts", batch}, {"max_input_error", max_error},
            {"decoded_plaintexts", decoded_plaintexts},
            {"max_cpu_decode_difference", max_cpu_difference}, {"tolerance", tolerance},
            {"cpu_ntt_residue_differences", cpu_residue_differences},
            {"exact_gpu_reference_match", reference.empty() ? Json(nullptr) : Json(true)}};
}
} // namespace

int main(int argc, char **argv)
try {
    if (argc != 7) throw std::invalid_argument(
        "usage: poseidon_gpu_encode_bench LIMBS BATCH SCALE_LOG2 WARMUP REPEAT REPORT.json");
    const std::size_t n = 65536, limbs = std::stoul(argv[1]), batch = std::stoul(argv[2]);
    const int scale_log2 = std::stoi(argv[3]), warmup = std::stoi(argv[4]), repeat = std::stoi(argv[5]);
    if (!limbs || limbs > 64 || !batch || batch > 64 || warmup < 0 || repeat < 1 ||
        scale_log2 < 20 || scale_log2 > 60 || scale_log2 + 1 >= static_cast<int>(limbs * 30))
        throw std::invalid_argument("unsupported benchmark size or scale");
    const double scale = std::ldexp(1.0, scale_log2);
    gpu_check_cuda(cudaSetDevice(0), "select GPU");
    const char *pool_env = std::getenv("POSEIDON_GPU_ENCODE_POOL_MB");
    const std::string pool_text = pool_env ? pool_env : "64";
    if (pool_text.empty() || pool_text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("POSEIDON_GPU_ENCODE_POOL_MB must be a nonnegative integer");
    const auto pool_mb = std::stoul(pool_text);
    if (pool_mb > 4096) throw std::invalid_argument("encode benchmark pool initial size exceeds 4096 MiB");
    BenchmarkMemoryPool memory_pool(pool_mb * (1ULL << 20));
    cudaDeviceProp properties{};
    gpu_check_cuda(cudaGetDeviceProperties(&properties, 0), "GPU properties");
    setenv("POSEIDON_NTT_ALGO", "tensor", 1);
    setenv("POSEIDON_NTT_FUSION_STAGES", "4", 1);
    setenv("POSEIDON_NTT_FUSED_MATRIX_STAGES", "4", 1);
    setenv("POSEIDON_NTT_FUSED_MATRIX_MAX_LEVELS", "1", 1);
    setenv("POSEIDON_NTT_FUSED_MATRIX_FP64_TABLES", "0", 1);
    ParametersLiteral parameters(CKKS, 16, 15, scale_log2, 0, 0, Modulus(0), {}, {}, sec_level_type::none);
    parameters.set_log_modulus(std::vector<std::uint32_t>(limbs, 30), {});
    const auto setup_start = Clock::now();
    PoseidonContext context(parameters);
    GpuParameterData gpu_parameters(context, 0);
    const auto &level = gpu_parameters.get_first_q_level();
    if (level.q_count != limbs || level.shards.size() != 1)
        throw std::runtime_error("unexpected benchmark level layout");
    const auto &shard = level.shards.front();
    std::vector<std::uint32_t> indices(n);
    std::vector<double2> twists(n);
    util::ComplexRoots roots(2 * n, MemoryManager::GetPool());
    std::size_t position = 1;
    for (std::size_t i = 0; i < n / 2; ++i) {
        indices[i] = (position - 1) / 2;
        indices[n / 2 + i] = (2 * n - position - 1) / 2;
        position = position * 5 % (2 * n);
    }
    for (std::size_t i = 0; i < n; ++i) {
        const auto root = std::conj(roots.get_root(i));
        twists[i] = make_double2(root.real(), root.imag());
    }
    GpuEncodeExperiment encoder(n, limbs, batch, indices, twists);
#ifdef POSEIDON_ENCODE_TILELANG
    if (properties.major * 10 + properties.minor < POSEIDON_ENCODE_TILELANG_SM)
        throw std::runtime_error("GPU is older than the configured TileLang target architecture");
    encoder.initialize_tilelang(shard);
#endif
    DeviceVector<double> input(n / 2 * batch, 0);
    PinnedInput host(input.size());
    std::mt19937_64 rng(20261011);
    for (std::size_t i = 0; i < input.size(); ++i) {
        // Exactly representable values; every plaintext differs. Include zero,
        // negative, alternating-sign and larger-magnitude values as well.
        host.data[i] = (static_cast<int>(rng() % 2049) - 1024) / 1024.0;
        if (i % 97 == 0) host.data[i] *= 16;
        if (i % 101 == 0) host.data[i] = 0;
    }
    input.copy_from_host(host.data, input.size());
    gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "finish setup");
    const auto setup_seconds = std::chrono::duration<double>(Clock::now() - setup_start).count();
    Json report{{"gpu", properties.name}, {"compute_capability", properties.major * 10 + properties.minor},
        {"degree", n}, {"q_limbs", limbs}, {"modulus_bits", 30}, {"scale_log2", scale_log2},
        {"batch", batch}, {"warmup", warmup}, {"repeat", repeat}, {"setup_seconds", setup_seconds},
        {"raw_bytes", input.size() * sizeof(double)}, {"encoded_gpu_bytes", n * limbs * batch * sizeof(GpuWord)},
        {"fft", "cuFFT double Z2Z, all backends"}, {"tensor_scope", "INT8 Tensor Core NTT, fusion=4"},
        {"allocator", pool_mb ? "RMM pool" : "direct cudaMalloc/cudaFree"}, {"initial_pool_mb", pool_mb},
        {"allocation_scope", "workspace, tables, plans and output allocation excluded; existing NTT scratch included"},
        {"modes", Json::array()}};
    std::vector<GpuWord> reference;
    Json reference_validation;
    using Backend = GpuEncodeExperiment::Backend;
    std::vector<std::pair<Backend, std::string>> backends{
        {Backend::cuda, "cuda_ntt"}, {Backend::tensor, "tensor_ntt"}};
#ifdef POSEIDON_ENCODE_TILELANG
    encoder.validate_tilelang_ntt(shard);
    report["tilelang"] = {{"version", "0.1.14"}, {"target_sm", POSEIDON_ENCODE_TILELANG_SM},
        {"boundary_random_ntt_exact_match", true}, {"reduction", "runtime primes, uint64 Barrett"}};
    backends.emplace_back(Backend::tilelang_cuda, "tilelang_cuda_ntt");
    backends.emplace_back(Backend::tilelang_tensor, "tilelang_tensor_ntt");
#endif
    for (const auto &[backend, name] : backends) {
        const bool tensor = backend == Backend::tensor || backend == Backend::tilelang_tensor;
        if (tensor && !supports_tensor_core_integer_gemm()) {
            report["tensor_skipped"] = "SM 7.5+ required";
            continue;
        }
        for (bool batched : {false, true}) {
            Json mode{{"backend", name},
                {"submission", batched ? "batched" : "individual"}};
            std::cout << "measuring " << mode.dump() << std::endl;
            mode["device_only"] = measure(encoder, input, host, shard, scale, backend, batched, false, warmup, repeat);
            std::vector<GpuWord> actual(encoder.output.size());
            encoder.output.copy_to_host(actual.data(), actual.size());
            mode["correctness"] = validate(actual, reference, host, context, batch, n, limbs, scale,
                                            reference_validation);
            if (reference.empty()) {
                reference = std::move(actual);
                reference_validation = mode["correctness"];
            }
            mode["with_raw_h2d"] = measure(encoder, input, host, shard, scale, backend, batched, true, warmup, repeat);
            mode["device_ms_per_plaintext"] = mode["device_only"]["total"]["median_ms"].get<double>() / batch;
            mode["device_plaintexts_per_second"] = 1000.0 * batch / mode["device_only"]["total"]["median_ms"].get<double>();
            report["modes"].push_back(std::move(mode));
        }
    }
    CKKSEncoder cpu(context);
    Plaintext cpu_output;
    std::vector<std::vector<double>> cpu_slots(batch);
    for (std::size_t b = 0; b < batch; ++b)
        cpu_slots[b].assign(host.data + b * n / 2, host.data + (b + 1) * n / 2);
    std::vector<double> cpu_times;
    for (int i = -warmup; i < repeat; ++i) {
        const auto start = Clock::now();
        for (const auto &slots : cpu_slots) cpu.encode(slots, scale, cpu_output);
        const auto finish = Clock::now();
        if (i >= 0) cpu_times.push_back(std::chrono::duration<double, std::milli>(finish - start).count());
    }
    report["cpu_encode_only"] = statistics(cpu_times);
    // Exercise errors produced asynchronously on the GPU, not just CLI checks.
    const double original = host.data[0];
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::ldexp(1.0, 90)}) {
        host.data[0] = bad;
        input.copy_from_host(host.data, input.size());
        encoder.prepare(input.data(), scale, shard, true);
        bool rejected = false;
        try { encoder.check_result(); }
        catch (const std::runtime_error &error) {
            if (std::string(error.what()).find("GPU Encode:") == std::string::npos) throw;
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("GPU Encode accepted invalid coefficients");
    }
    host.data[0] = original;
    report["nonfinite_and_coefficient_overflow_rejected"] = true;
    std::ofstream output(argv[6]);
    if (!output) throw std::runtime_error("cannot open report");
    output << report.dump(2) << '\n';
    output.close();
    if (!output) throw std::runtime_error("cannot write report");
    std::cout << "report: " << argv[6] << std::endl;
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
}
