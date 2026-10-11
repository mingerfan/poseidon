// Compare actual GPU evaluator operations alone and with background Encode.
// Two host threads use distinct CUDA per-thread streams on the same GPU.
#include "gpu_encode_experiment.h"
#include "poseidon/basics/util/croots.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/evaluator/software/evaluator_ckks_software.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "json.hpp"
#include <nvtx3/nvToolsExt.h>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

using namespace poseidon;
using namespace poseidon::gpu;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Backend = GpuEncodeExperiment::Backend;
namespace {
struct Range {
    explicit Range(const std::string &name) { nvtxRangePushA(name.c_str()); }
    ~Range() { nvtxRangePop(); }
};
class Pool {
    rmm::mr::device_memory_resource *previous_ = rmm::mr::get_current_device_resource();
    rmm::mr::cuda_memory_resource upstream_;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_{&upstream_, 64ULL << 20};
public:
    Pool() { rmm::mr::set_current_device_resource(&pool_); }
    ~Pool() { rmm::mr::set_current_device_resource(previous_); }
};
struct Event {
    cudaEvent_t value{};
    Event() { gpu_check_cuda(cudaEventCreate(&value), "create event"); }
    ~Event() { cudaEventDestroy(value); }
    void record() { gpu_check_cuda(cudaEventRecord(value, gpu_execution_stream()), "record event"); }
    void wait() { gpu_check_cuda(cudaEventSynchronize(value), "wait event"); }
};
double elapsed(const Event &a, const Event &b) {
    float ms = 0;
    gpu_check_cuda(cudaEventElapsedTime(&ms, a.value, b.value), "elapsed event");
    return ms;
}
Json stats(const std::vector<double> &samples) {
    if (samples.empty()) return Json(nullptr);
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto n = sorted.size();
    return {{"median_ms", (sorted[(n-1)/2]+sorted[n/2])/2},
        {"mean_ms",std::accumulate(samples.begin(),samples.end(),0.)/n},
        {"p95_ms", sorted[static_cast<std::size_t>(std::ceil(n*.95))-1]},
        {"min_ms", sorted.front()}, {"max_ms", sorted.back()}, {"samples_ms", samples}};
}
void equal_cipher(const GpuCiphertextData &actual, const Ciphertext &expected, const PoseidonContext &context) {
    Ciphertext downloaded;
    GpuUploader::download_ciphertext(actual, downloaded, context);
    if (downloaded.size() != expected.size() || downloaded.parms_id() != expected.parms_id() ||
        downloaded.is_ntt_form() != expected.is_ntt_form() || downloaded.scale() != expected.scale())
        throw std::runtime_error("operator output metadata differs from CPU");
    const auto count = expected.size() * expected.poly_modulus_degree() * expected.coeff_modulus_size();
    if (!std::equal(expected.data(), expected.data()+count, downloaded.data()))
        throw std::runtime_error("operator output residues differ from CPU");
}

// Same inverse embedding and gen=5 slot map as CKKSEncoder, with scale=1.
// Keep only N real coefficients; roots/workspace are transient and reusable.
std::vector<double> cpu_fft_coefficients(const double *input, std::size_t n, std::size_t batch,
    const std::vector<std::uint32_t> &indices, const std::vector<double2> &twists) {
    const int logn = util::get_power_of_two(n);
    std::vector<std::complex<double>> roots(n), work(n);
    for (std::size_t i = 1; i < n; ++i) {
        const auto root = twists[util::reverse_bits(i-1,logn)+1];
        roots[i] = {root.x,root.y};
    }
    util::FFTHandler fft{util::ComplexArith{}};
    const double normalization = 1. / n;
    std::vector<double> coefficients(batch*n);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t i = 0; i < n/2; ++i) {
            const double value = input[b*n/2+i];
            work[util::reverse_bits(indices[i],logn)] = {value,0};
            work[util::reverse_bits(indices[n/2+i],logn)] = {value,0};
        }
        fft.transform_from_rev(work.data(),logn,roots.data(),&normalization);
        for (std::size_t i = 0; i < n; ++i) coefficients[b*n+i] = work[i].real();
    }
    return coefficients;
}

Json validate_encode(const std::vector<GpuWord> &actual, const double *input,
    const PoseidonContext &context, std::size_t n, std::size_t limbs, std::size_t batch,
    bool require_cpu_exact) {
    CKKSEncoder cpu(context);
    const auto id = context.crt_context()->first_context_data()->parms_id();
    const double scale = std::ldexp(1.,40);
    const double tolerance = std::max(1e-8,64.*std::sqrt(static_cast<double>(n))/scale);
    std::size_t differences = 0;
    double max_error = 0;
    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<double> slots(input+b*n/2,input+(b+1)*n/2), decoded;
        Plaintext expected, downloaded;
        cpu.encode(slots,id,scale,expected);
        downloaded.resize(context,id,n*limbs);
        downloaded.parms_id() = id;
        downloaded.scale() = scale;
        for (std::size_t i = 0; i < n*limbs; ++i) {
            downloaded.data()[i] = actual[b*n*limbs+i];
            differences += downloaded.data()[i] != expected.data()[i];
        }
        cpu.decode(downloaded,decoded);
        for (std::size_t i = 0; i < n/2; ++i) {
            if (!std::isfinite(decoded[i])) throw std::runtime_error("non-finite decoded Encode output");
            max_error = std::max(max_error,std::abs(decoded[i]-slots[i]));
        }
    }
    if (max_error > tolerance || (require_cpu_exact && differences))
        throw std::runtime_error("CPU-precomputed or full Encode validation failed");
    return {{"checked_plaintexts",batch}, {"cpu_ntt_residue_differences",differences},
        {"cpu_exact_match",differences == 0}, {"max_input_error",max_error}, {"tolerance",tolerance}};
}

struct EncodeMode { std::string name; Backend backend; bool precomputed = false; };

class EncodeWorker {
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread thread_;
    std::exception_ptr error_;
    bool ready_ = false, started_ = false, idle_ = true, exit_ = false;
    Backend backend_ = Backend::cuda;
    bool precomputed_ = false;
    double period_ms_ = 0;
    std::atomic<bool> active_{false};
    std::atomic<std::uint64_t> completed_{0};
    std::vector<double> samples_;
    unsigned long long stream_id_ = 0;
    Json preparation_ = Json::object();

    void fail(std::exception_ptr error) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = error;
        condition_.notify_all();
    }
    void check() { if (error_) std::rethrow_exception(error_); }
public:
    EncodeWorker(std::size_t n, std::size_t limbs, std::size_t batch,
                 const GpuParameterShard &params, const std::vector<std::uint32_t> &indices,
                 const std::vector<double2> &twists, bool upload, bool compare_cpu_fft,
                 const PoseidonContext &context, std::uint64_t order_seed) {
        thread_ = std::thread([=, &params, &indices, &twists, &context, this] {
            try {
                gpu_check_cuda(cudaSetDevice(0), "worker select GPU");
                gpu_check_cuda(cudaStreamGetId(gpu_execution_stream(), &stream_id_), "worker stream ID");
                // Create and bind FFT plans in the encoding thread.
                GpuEncodeExperiment encoder(n, limbs, batch, indices, twists);
#ifdef POSEIDON_ENCODE_TILELANG
                encoder.initialize_tilelang(params);
                encoder.validate_tilelang_ntt(params);
#endif
                DeviceVector<double> input(batch * n / 2, 0);
                double *pinned = nullptr;
                gpu_check_cuda(cudaMallocHost(&pinned, input.size()*sizeof(double)), "worker pinned input");
                std::unique_ptr<double, decltype(&cudaFreeHost)> host(pinned, cudaFreeHost);
                std::mt19937_64 rng(20261011);
                for (std::size_t i = 0; i < input.size(); ++i)
                    pinned[i] = (static_cast<int>(rng()%2049)-1024)/1024.0;
                input.copy_from_host(pinned, input.size());
                DeviceVector<double> coefficients;
                double *pinned_coefficients = nullptr;
                std::unique_ptr<double, decltype(&cudaFreeHost)> coefficient_host(nullptr,cudaFreeHost);
                if (compare_cpu_fft) {
                    coefficients.allocate(batch*n,0);
                    gpu_check_cuda(cudaMallocHost(&pinned_coefficients,coefficients.size()*sizeof(double)),
                        "worker pinned CPU FFT coefficients");
                    coefficient_host.reset(pinned_coefficients);
                    const auto start = Clock::now();
                    const auto prepared = cpu_fft_coefficients(pinned,n,batch,indices,twists);
                    const double ms = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                    std::copy(prepared.begin(),prepared.end(),pinned_coefficients);
                    coefficients.copy_from_host(pinned_coefficients,coefficients.size());
                    preparation_["cpu_fft_preprocessing_ms"] = ms;
                    preparation_["cpu_fft_preprocessing_ms_per_plaintext"] = ms/batch;
                    preparation_["raw_cache_bytes"] = input.size()*sizeof(double);
                    preparation_["coefficient_cache_bytes"] = coefficients.size()*sizeof(double);
                    preparation_["coefficient_format"] = "FP64 [batch][N], inverse embedding normalized, unscaled/unrounded";
                }
                auto run = [&](Backend backend, bool precomputed = false) {
                    auto &source = precomputed ? coefficients : input;
                    const auto *host_source = precomputed ? pinned_coefficients : pinned;
                    if (upload) gpu_check_cuda(cudaMemcpyAsync(source.data(),host_source,
                        source.size()*sizeof(double),cudaMemcpyHostToDevice,gpu_execution_stream()), "worker input H2D");
                    if (precomputed) encoder.prepare_coefficients(source.data(),std::ldexp(1.,40),params,true);
                    else encoder.prepare(source.data(), std::ldexp(1.,40), params, true);
                    encoder.ntt(params, backend, true);
                };
                run(Backend::cuda);
                encoder.check_result();
                std::array<std::vector<GpuWord>,2> reference;
                reference[0].resize(encoder.output.size());
                encoder.output.copy_to_host(reference[0].data(),reference[0].size());
                std::vector<GpuWord> actual(reference[0].size());
                Event begin, end;
                if (compare_cpu_fft) {
                    preparation_["full_gpu_validation"] = validate_encode(reference[0],pinned,context,n,limbs,batch,false);
                    run(Backend::cuda,true);
                    encoder.check_result();
                    reference[1].resize(actual.size());
                    encoder.output.copy_to_host(reference[1].data(),reference[1].size());
                    preparation_["cpu_fft_validation"] = validate_encode(reference[1],pinned,context,n,limbs,batch,true);
                    // Invalid cached coefficients must be rejected, too.
                    const double saved = pinned_coefficients[0];
                    for (const double invalid : {std::numeric_limits<double>::quiet_NaN(),std::ldexp(1.,90)}) {
                        pinned_coefficients[0] = invalid;
                        coefficients.copy_from_host(pinned_coefficients,coefficients.size());
                        encoder.prepare_coefficients(coefficients.data(),std::ldexp(1.,40),params,true);
                        bool rejected = false;
                        try { encoder.check_result(); } catch (const std::runtime_error &) { rejected = true; }
                        if (!rejected) throw std::runtime_error("invalid cached coefficient was accepted");
                    }
                    pinned_coefficients[0] = saved;
                    coefficients.copy_from_host(pinned_coefficients,coefficients.size());
                    preparation_["invalid_coefficients_rejected"] = true;
                    std::vector<EncodeMode> standalone{{"cuda_encode",Backend::cuda},
                        {"cpu_fft_cuda_encode",Backend::cuda,true}};
#ifdef POSEIDON_ENCODE_TILELANG
                    standalone.push_back({"tilelang_tensor_encode",Backend::tilelang_tensor});
                    standalone.push_back({"cpu_fft_tilelang_tensor_encode",Backend::tilelang_tensor,true});
#endif
                    std::mt19937_64 rng_order(order_seed);
                    std::shuffle(standalone.begin(),standalone.end(),rng_order);
                    for (const auto &mode : standalone) {
                        double warm = 0;
                        std::vector<double> samples;
                        for (int measured = 0; measured < 30;) {
                            begin.record(); run(mode.backend,mode.precomputed); end.record(); end.wait();
                            encoder.check_result();
                            const auto ms = elapsed(begin,end);
                            if (warm < 100) warm += ms;
                            else { samples.push_back(ms); ++measured; }
                        }
                        encoder.output.copy_to_host(actual.data(),actual.size());
                        if (actual != reference[mode.precomputed])
                            throw std::runtime_error("standalone Encode differs from its canonical CUDA output");
                        preparation_["standalone"][mode.name] = stats(samples);
                        preparation_["standalone"][mode.name]["median_ms_per_plaintext"] =
                            preparation_["standalone"][mode.name]["median_ms"].get<double>()/batch;
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    ready_ = true;
                    condition_.notify_all();
                }
                for (;;) {
                    Backend backend;
                    bool precomputed;
                    double period_ms;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        condition_.wait(lock, [&] { return active_.load() || exit_; });
                        if (exit_) break;
                        backend = backend_;
                        precomputed = precomputed_;
                        period_ms = period_ms_;
                    }
                    std::vector<double> samples;
                    double warm_ms = 0;
                    while (active_.load()) {
                        const auto iteration_start = Clock::now();
                        begin.record();
                        run(backend,precomputed);
                        end.record();
                        end.wait();
                        encoder.check_result();
                        const double ms = elapsed(begin,end);
                        if (warm_ms < 100) {
                            warm_ms += ms;
                            if (warm_ms >= 100) {
                                std::lock_guard<std::mutex> lock(mutex_);
                                started_ = true;
                                condition_.notify_all();
                            }
                        } else {
                            samples.push_back(ms);
                            completed_.fetch_add(batch);
                        }
                        if (warm_ms >= 100 && period_ms > 0)
                            std::this_thread::sleep_until(iteration_start +
                                std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(period_ms)));
                    }
                    encoder.output.copy_to_host(actual.data(), actual.size());
                    if (actual != reference[precomputed])
                        throw std::runtime_error("background Encode residues differ from its canonical CUDA output");
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        samples_ = std::move(samples);
                        idle_ = true;
                        condition_.notify_all();
                    }
                }
            } catch (...) { fail(std::current_exception()); }
        });
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return ready_ || error_; });
        if (error_) {
            lock.unlock();
            thread_.join();
            std::rethrow_exception(error_);
        }
    }
    ~EncodeWorker() {
        active_.store(false);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            exit_ = true;
            condition_.notify_all();
        }
        if (thread_.joinable()) thread_.join();
    }
    void start(Backend backend, double period_ms = 0, bool precomputed = false) {
        std::unique_lock<std::mutex> lock(mutex_);
        check();
        backend_ = backend;
        precomputed_ = precomputed;
        period_ms_ = period_ms;
        started_ = false;
        idle_ = false;
        completed_.store(0);
        active_.store(true);
        condition_.notify_all();
        condition_.wait(lock, [&] { return started_ || error_; });
        check();
    }
    Json stop(std::size_t batch) {
        active_.store(false);
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return idle_ || error_; });
        check();
        auto result = stats(samples_);
        if (!result.is_null()) result["median_ms_per_plaintext"] = result["median_ms"].get<double>() / batch;
        return result;
    }
    std::uint64_t completed() const { return completed_.load(); }
    unsigned long long stream_id() const { return stream_id_; }
    const Json &preparation() const { return preparation_; }
};

Json measure(const std::function<void()> &operation, int repeat, int burst, EncodeWorker &worker) {
    Event begin, end;
    double warm_ms = 0;
    while (warm_ms < 100) {
        begin.record();
        for (int i = 0; i < burst; ++i) operation();
        end.record(); end.wait();
        warm_ms += elapsed(begin,end);
    }
    std::vector<double> event_ms, wall_ms;
    const auto produced_start = worker.completed();
    const auto all_start = Clock::now();
    for (int i = 0; i < repeat; ++i) {
        const auto start = Clock::now();
        begin.record();
        for (int j = 0; j < burst; ++j) operation();
        end.record(); end.wait();
        wall_ms.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count()/burst);
        event_ms.push_back(elapsed(begin,end)/burst);
    }
    const double window_ms = std::chrono::duration<double,std::milli>(Clock::now()-all_start).count();
    const auto produced = worker.completed() - produced_start;
    return {{"operator",stats(event_ms)}, {"wall",stats(wall_ms)}, {"measurement_window_ms",window_ms},
        {"encoded_plaintexts_in_window",produced}, {"operator_calls",repeat*burst},
        {"encoded_plaintexts_per_operator",static_cast<double>(produced)/(repeat*burst)},
        {"encode_plaintexts_per_second_in_window",1000.*produced/window_ms}};
}
} // namespace

int main(int argc, char **argv)
try {
    if (argc != 7) throw std::invalid_argument(
        "usage: poseidon_gpu_encode_contention_bench Q_LIMBS BATCH REPEAT BURST RAW_H2D REPORT.json");
    const std::size_t n = 65536, limbs = std::stoul(argv[1]), batch = std::stoul(argv[2]);
    const int repeat = std::stoi(argv[3]), burst = std::stoi(argv[4]), upload = std::stoi(argv[5]);
    const char *demand_env = std::getenv("POSEIDON_GPU_ENCODE_DEMAND_ONLY");
    const std::string demand_text = demand_env ? demand_env : "0";
    if (demand_text != "0" && demand_text != "1")
        throw std::invalid_argument("POSEIDON_GPU_ENCODE_DEMAND_ONLY must be 0 or 1");
    const bool demand_only = demand_text == "1";
    const char *cpu_fft_env = std::getenv("POSEIDON_GPU_ENCODE_COMPARE_CPU_FFT");
    const std::string cpu_fft_text = cpu_fft_env ? cpu_fft_env : "0";
    if (cpu_fft_text != "0" && cpu_fft_text != "1")
        throw std::invalid_argument("POSEIDON_GPU_ENCODE_COMPARE_CPU_FFT must be 0 or 1");
    const bool compare_cpu_fft = cpu_fft_text == "1";
    const char *order_env = std::getenv("POSEIDON_GPU_ENCODE_ORDER_SEED");
    const std::uint64_t order_seed = order_env ? std::stoull(order_env) : 20261012+limbs*100+batch;
    if (limbs < 3 || limbs > 32 || !batch || batch > 32 || repeat < 1 || burst < 1 || burst > 64 ||
        (upload != 0 && upload != 1)) throw std::invalid_argument("unsupported contention geometry");
    gpu_check_cuda(cudaSetDevice(0), "select GPU");
    Pool pool;
    setenv("POSEIDON_NTT_ALGO","tensor",1);
    setenv("POSEIDON_NTT_FUSION_STAGES","4",1);
    setenv("POSEIDON_NTT_FUSED_MATRIX_STAGES","4",1);
    setenv("POSEIDON_NTT_FUSED_MATRIX_MAX_LEVELS","1",1);
    setenv("POSEIDON_NTT_FUSED_MATRIX_FP64_TABLES","0",1);
    ParametersLiteral parameters(CKKS,16,15,40,0,0,Modulus(0),{},{},sec_level_type::none);
    const std::size_t p_count = 4;
    parameters.set_log_modulus(std::vector<std::uint32_t>(limbs,30),std::vector<std::uint32_t>(p_count,30));
    std::cout << "setting up operands and direct rotation/relinearization keys" << std::endl;
    PoseidonContext context(parameters);
    KeyGenerator keygen(context);
    PublicKey public_key; keygen.create_public_key(public_key);
    RelinKeys relin_keys; keygen.create_relin_keys(relin_keys);
    GaloisKeys galois_keys;
    keygen.create_galois_keys(std::vector<std::uint32_t>{context.crt_context()->galois_tool()->get_elt_from_step(1)},galois_keys);
    CKKSEncoder cpu_encoder(context);
    Plaintext plain0, plain1;
    cpu_encoder.encode(std::vector<double>{1.,2.,3.,4.}, std::ldexp(1.,40), plain0);
    cpu_encoder.encode(std::vector<double>{.5,.25,.125,.0625}, std::ldexp(1.,40), plain1);
    Encryptor encryptor(context,public_key,keygen.secret_key());
    Ciphertext ct0, ct1;
    encryptor.encrypt(plain0,ct0); encryptor.encrypt(plain1,ct1);
    GpuParameterData gpu_params(context,0);
    GpuEvaluator gpu(gpu_params);
    auto gpu0 = GpuUploader::upload_ciphertext(ct0,0), gpu1 = GpuUploader::upload_ciphertext(ct1,0);
    auto gpu_plain = GpuUploader::upload_plaintext(plain1,0);
    auto gpu_relin = GpuUploader::upload_relin_keys(relin_keys,0);
    auto gpu_galois = GpuUploader::upload_galois_keys(galois_keys,0);
    setenv("POSEIDON_NTT_ALGO","fourstep",1); // Main computation uses CUDA NTT.
    EvaluatorCkksSoftware cpu(context);
    Ciphertext multiplied;
    cpu.multiply(ct0,ct1,multiplied);
    GpuCiphertextData gpu_multiplied;
    gpu.multiply(gpu0,gpu1,gpu_multiplied);
    std::vector<std::uint32_t> indices(n);
    std::vector<double2> twists(n);
    util::ComplexRoots roots(2*n,MemoryManager::GetPool());
    std::size_t position = 1;
    for (std::size_t i = 0; i < n/2; ++i) {
        indices[i] = (position-1)/2; indices[n/2+i] = (2*n-position-1)/2;
        position = position*5%(2*n);
    }
    for (std::size_t i = 0; i < n; ++i) {
        const auto root = std::conj(roots.get_root(i));
        twists[i] = make_double2(root.real(),root.imag());
    }
    gpu_check_cuda(cudaDeviceSynchronize(),"finish operand setup");
    // Fusion tables are built at the top key level. Encode reads its Q
    // prefix, while stage offsets include all Q+P limbs.
    const auto &encode_params = gpu_params.get_level(context.crt_context()->key_context_data()->parms_id());
    EncodeWorker worker(n,limbs,batch,encode_params.shards.front(),indices,twists,upload,compare_cpu_fft,context,order_seed);
    struct Operation { std::string name; std::function<void()> gpu_call; std::function<void(Ciphertext&)> cpu_call; };
    GpuCiphertextData output;
    std::vector<Operation> operations{
        {"add", [&] { gpu.add(gpu0,gpu1,output); }, [&](Ciphertext &out) { cpu.add(ct0,ct1,out); }},
        {"multiply_plain", [&] { gpu.multiply_plain(gpu0,gpu_plain,output); }, [&](Ciphertext &out) { cpu.multiply_plain(ct0,plain1,out); }},
        {"multiply", [&] { gpu.multiply(gpu0,gpu1,output); }, [&](Ciphertext &out) { cpu.multiply(ct0,ct1,out); }},
        {"rescale", [&] { gpu.rescale(gpu0,output); }, [&](Ciphertext &out) { cpu.rescale(ct0,out); }},
        {"relinearize", [&] { gpu.relinearize(gpu_multiplied,gpu_relin,output); }, [&](Ciphertext &out) { cpu.relinearize(multiplied,out,relin_keys); }},
        {"rotate", [&] { gpu.rotate(gpu0,1,gpu_galois,output); }, [&](Ciphertext &out) { cpu.rotate(ct0,out,1,galois_keys); }}
    };
    cudaDeviceProp prop{};
    gpu_check_cuda(cudaGetDeviceProperties(&prop,0),"GPU info");
    unsigned long long main_stream_id = 0;
    gpu_check_cuda(cudaStreamGetId(gpu_execution_stream(),&main_stream_id),"main stream ID");
    if (main_stream_id == worker.stream_id()) throw std::runtime_error("operator and Encode streams are identical");
    Json report{{"gpu",prop.name}, {"degree",n}, {"q_limbs",limbs}, {"p_limbs",p_count}, {"batch",batch},
        {"scale_log2",40}, {"raw_h2d",static_cast<bool>(upload)}, {"repeat",repeat}, {"burst",burst},
        {"demand_only",demand_only},
        {"order_seed",order_seed},
        {"compare_cpu_fft",compare_cpu_fft}, {"encode_preparation",worker.preparation()},
        {"input_h2d",static_cast<bool>(upload)},
        {"streams","two host threads, distinct CUDA per-thread streams; same GPU, default priority"},
        {"main_stream_id",main_stream_id}, {"encode_stream_id",worker.stream_id()},
        {"scope","actual GpuEvaluator operations with internal allocation, RMM pool; background Encode with optional precomputed CPU FFT"},
        {"rows",Json::array()}};
    std::vector<EncodeMode> backgrounds{{"cuda_encode",Backend::cuda}};
    if (compare_cpu_fft) backgrounds.push_back({"cpu_fft_cuda_encode",Backend::cuda,true});
#ifdef POSEIDON_ENCODE_TILELANG
    if (prop.major*10+prop.minor < POSEIDON_ENCODE_TILELANG_SM)
        throw std::runtime_error("GPU is older than configured TileLang architecture");
    backgrounds.push_back({"tilelang_tensor_encode",Backend::tilelang_tensor});
    if (compare_cpu_fft) backgrounds.push_back({"cpu_fft_tilelang_tensor_encode",Backend::tilelang_tensor,true});
    if (demand_only) {
        backgrounds = {{"tilelang_tensor_encode_demand",Backend::tilelang_tensor}};
        if (compare_cpu_fft) backgrounds.push_back({"cpu_fft_tilelang_tensor_encode_demand",Backend::tilelang_tensor,true});
    }
#else
    if (demand_only) throw std::runtime_error("Demand test requires the TileLang Tensor backend");
#endif
    std::mt19937_64 order_rng(order_seed+1);
    for (auto &op : operations) {
        if (demand_only && op.name != "relinearize" && op.name != "rotate") continue;
        Ciphertext expected;
        op.cpu_call(expected);
        op.gpu_call();
        equal_cipher(output,expected,context);
        Json row{{"operator",op.name}, {"cpu_exact_match",true}, {"backgrounds",Json::object()}};
        // Paired baselines bracket the shuffled background experiments.
        {
            Range range("contention."+op.name+".alone_before");
            row["alone_before"] = measure(op.gpu_call,repeat,burst,worker);
        }
        std::shuffle(backgrounds.begin(),backgrounds.end(),order_rng);
        for (const auto &mode : backgrounds) {
            std::cout << "measuring " << op.name << " + " << mode.name << std::endl;
            const double period_ms = demand_only ?
                row["alone_before"]["operator"]["mean_ms"].get<double>()*batch : 0;
            worker.start(mode.backend,period_ms,mode.precomputed);
            Json result;
            {
                Range range("contention."+op.name+"."+mode.name);
                result = measure(op.gpu_call,repeat,burst,worker);
            }
            result["encode_batch_latency"] = worker.stop(batch);
            result["target_batch_period_ms"] = period_ms;
            result["precomputed_cpu_fft"] = mode.precomputed;
            result["input_bytes_per_batch"] = batch*n*sizeof(double)/(mode.precomputed ? 1 : 2);
            if (period_ms > 0) result["target_plaintexts_per_second"] = 1000.*batch/period_ms;
            equal_cipher(output,expected,context);
            result["operator_and_encode_exact_match"] = true;
            row["backgrounds"][mode.name] = std::move(result);
        }
        {
            Range range("contention."+op.name+".alone_after");
            row["alone_after"] = measure(op.gpu_call,repeat,burst,worker);
        }
        const double baseline = (row["alone_before"]["operator"]["median_ms"].get<double>() +
            row["alone_after"]["operator"]["median_ms"].get<double>())/2;
        row["baseline_median_ms"] = baseline;
        const double baseline_mean = (row["alone_before"]["operator"]["mean_ms"].get<double>() +
            row["alone_after"]["operator"]["mean_ms"].get<double>())/2;
        row["baseline_mean_ms"] = baseline_mean;
        for (auto &result : row["backgrounds"].items()) {
            result.value()["operator_slowdown"] = result.value()["operator"]["median_ms"].get<double>()/baseline;
            result.value()["operator_mean_slowdown"] = result.value()["operator"]["mean_ms"].get<double>()/baseline_mean;
        }
        report["rows"].push_back(std::move(row));
    }
    std::ofstream out(argv[6]);
    if (!out) throw std::runtime_error("cannot open report");
    out << report.dump(2) << '\n'; out.close();
    if (!out) throw std::runtime_error("cannot write report");
    std::cout << "report: " << argv[6] << std::endl;
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
}
