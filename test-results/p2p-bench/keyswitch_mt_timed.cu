// Multi-thread, single-GPU keyswitch.hybrid launch-overhead benchmark.
// SELF-TIMED variant: no nsys needed (use where nsys CUPTI tracing is
// unavailable, e.g. AutoDL containers).
//
// Same kernel workload as keyswitch_mt_bench (full GpuEvaluator::rotate ->
// all 11 real keyswitch kernels, N=8192 q=17 p=2) with N launch threads on
// ONE GPU.
//
// Per measured round, each thread records:
//   cpu_us : host clock span of rotate()          (= launch-side CPU cost)
//   gpu_us : cudaEvent elapsed on the thread's per-thread stream spanning the
//            rotate kernels                        (= GPU compute cost)
// Reported: per-thread mean/median cpu & gpu, totals, and cpu/gpu ratio —
// the same "launch/compute" ratio nsys gives on machines where it works.
//
// Usage: keyswitch_mt_timed [rounds] [threads] [device]

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <atomic>
#include <memory>
#include <vector>
#include <algorithm>

#include "poseidon/ckks_encoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"
#include "poseidon/basics/randomgen.h"
#include "rmm/mr/device/cuda_memory_resource.hpp"
#include "rmm/mr/device/per_device_resource.hpp"
#include "rmm/mr/device/pool_memory_resource.hpp"

using namespace poseidon;

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

int log2_degree(std::size_t degree) {
    int l = 0;
    while ((std::size_t(1) << l) < degree) ++l;
    return l;
}

ParametersLiteral make_benchmark_parameters(std::size_t degree, std::size_t q_count,
                                            std::size_t p_count = 0) {
    const int log_n = log2_degree(degree);
    ParametersLiteral parms(
        CKKS, log_n, log_n - 1,
        /*log_scale=*/25, /*hamming_weight=*/0, /*q0_level=*/0,
        Modulus(0), std::vector<Modulus>{}, std::vector<Modulus>{},
        poseidon::sec_level_type::none);
    parms.set_log_modulus(
        std::vector<std::uint32_t>(q_count, 30),
        std::vector<std::uint32_t>(p_count, 30));
    return parms;
}

struct BenchState {
    std::atomic<bool> stop{false};
    std::atomic<long> done{0};

    int gpu;
    long rounds;

    // shared, read-only after setup (GpuCiphertextData is non-copyable)
    std::shared_ptr<PoseidonContext> context;
    std::shared_ptr<gpu::GpuParameterData> gpu_params;
    std::shared_ptr<gpu::GpuCiphertextData> ct;
    std::shared_ptr<gpu::GpuGaloisKeysData> galois_keys;
    int rotate_step = 1;

    // per-thread evaluator (each has its own per-thread stream)
    std::unique_ptr<gpu::GpuEvaluator> evaluator;
};

// Shared barrier: ONE counter for all threads. go.store(threads) then each
// thread fetch_sub's the same counter and spins until it reaches 0.
void barrier(std::atomic<int> &go) {
    go.fetch_sub(1);
    while (go.load() != 0) {}
}

void warmup_thread_fn(BenchState &st, std::atomic<int> &go, int warmup_rounds) {
    cudaSetDevice(st.gpu);
    barrier(go);                   // all threads start warmup together
    gpu::GpuCiphertextData output;
    for (int i = 0; i < warmup_rounds; ++i) {
        st.evaluator->rotate(*st.ct, st.rotate_step, *st.galois_keys, output);
    }
}

void measure_thread_fn(BenchState &st, std::atomic<int> &go,
                       std::vector<long> &cpu_us, std::vector<long> &gpu_us) {
    cudaSetDevice(st.gpu);
    barrier(go);                   // all threads start measure together
    gpu::GpuCiphertextData output;
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));
    cpu_us.reserve(st.rounds);
    gpu_us.reserve(st.rounds);
    for (long i = 0; i < st.rounds && !st.stop.load(); ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(cudaEventRecord(ev0, cudaStreamPerThread));
        st.evaluator->rotate(*st.ct, st.rotate_step, *st.galois_keys, output);
        const auto t1 = std::chrono::steady_clock::now();
        CHECK(cudaEventRecord(ev1, cudaStreamPerThread));
        CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        cpu_us.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        gpu_us.push_back(static_cast<long>(ms * 1000.0));
        st.done.fetch_add(1);
    }
    CHECK(cudaEventDestroy(ev0));
    CHECK(cudaEventDestroy(ev1));
}

int main(int argc, char **argv) {
    const std::size_t degree = 8192;
    const std::size_t q_count = 17;
    long rounds = 50;
    int threads = 1;
    int gpu = 0;
    if (argc > 1) rounds = atol(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);
    if (argc > 3) gpu = atoi(argv[3]);

    int dev_count = 0;
    CHECK(cudaGetDeviceCount(&dev_count));
    if (gpu >= dev_count) { fprintf(stderr, "need GPU %d (have %d)\n", gpu, dev_count); return 1; }
    cudaSetDevice(gpu);

    auto parms = make_benchmark_parameters(degree, q_count, /*p_count=*/2);
    auto context = std::make_shared<PoseidonContext>(parms);

    KeyGenerator keygen(*context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys galois_keys;
    const auto galois_tool = context->crt_context()->galois_tool();
    keygen.create_galois_keys(
        std::vector<std::uint32_t>{
            galois_tool->get_elt_from_step(1),
            galois_tool->get_elt_from_step(0)},
        galois_keys);

    CKKSEncoder encoder(*context);
    Encryptor encryptor(*context, public_key, keygen.secret_key());
    Plaintext plain;
    encoder.encode(std::vector<double>{1.0, 2.0, 3.0, 4.0},
                   parms.scale(), plain);
    Ciphertext ct;
    encryptor.encrypt(plain, ct);

    auto gpu_params = std::make_shared<gpu::GpuParameterData>();
    gpu_params->build_from_poseidon_context(*context, gpu);

    std::vector<BenchState> st(threads);
    for (int t = 0; t < threads; ++t) {
        st[t].gpu = gpu;
        st[t].rounds = rounds;
        st[t].context = context;
        st[t].gpu_params = gpu_params;
        st[t].evaluator = std::make_unique<gpu::GpuEvaluator>(*gpu_params);
    }
    st[0].ct = std::make_shared<gpu::GpuCiphertextData>(
        gpu::GpuUploader::upload_ciphertext(ct, gpu));
    st[0].galois_keys = std::make_shared<gpu::GpuGaloisKeysData>(
        gpu::GpuUploader::upload_galois_keys(galois_keys, gpu));
    for (int t = 1; t < threads; ++t) {
        st[t].ct = st[0].ct;                // shared device buffers, read-only
        st[t].galois_keys = st[0].galois_keys;
    }

    printf("degree=%zu q=%zu rounds=%ld threads=%d gpu=%d single-GPU-mt\n",
           degree, q_count, rounds, threads, gpu);
    fflush(stdout);

    std::atomic<int> go{0};   // shared barrier counter for all threads

    // ---- warmup: 3 rounds per thread, JIT excluded from analysis ----
    {
        printf("[warmup] start\n"); fflush(stdout);
        go.store(threads);
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back(warmup_thread_fn, std::ref(st[t]), std::ref(go), 3);
        for (auto &t : ts) t.join();
        cudaDeviceSynchronize();
        printf("[warmup] done\n"); fflush(stdout);
    }

    // ---- measure: `rounds` rounds per thread ----
    {
        printf("[measure] start\n"); fflush(stdout);
        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < threads; ++t) { st[t].done.store(0); }
        go.store(threads);
        std::vector<std::vector<long>> cpu(threads), gpu(threads);
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back(measure_thread_fn, std::ref(st[t]), std::ref(go),
                            std::ref(cpu[t]), std::ref(gpu[t]));
        while (true) {
            long d = 0;
            for (int t = 0; t < threads; ++t) d += st[t].done.load();
            if (d >= rounds * threads) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        st[0].stop.store(true);
        for (auto &t : ts) t.join();
        const auto t1 = std::chrono::steady_clock::now();
        printf("[measure] done wall=%.3f s\n",
               std::chrono::duration<double>(t1 - t0).count());

        long long tot_cpu = 0, tot_gpu = 0;
        printf("%-8s %8s %10s %10s %10s %10s\n",
               "thread", "n", "cpu_avg", "cpu_med", "gpu_avg", "gpu_med");
        for (int t = 0; t < threads; ++t) {
            auto c = cpu[t], g = gpu[t];
            std::sort(c.begin(), c.end());
            std::sort(g.begin(), g.end());
            long long sc = 0, sg = 0;
            for (auto v : c) sc += v;
            for (auto v : g) sg += v;
            tot_cpu += sc; tot_gpu += sg;
            printf("%-8d %8zu %10.1f %10.1f %10.1f %10.1f\n",
                   t, c.size(),
                   (double)sc / c.size(), (double)c[c.size() / 2],
                   (double)sg / g.size(), (double)g[g.size() / 2]);
        }
        printf("TOTAL cpu=%lld us  gpu=%lld us  cpu/gpu=%.3f\n",
               tot_cpu, tot_gpu, (double)tot_cpu / tot_gpu);
        fflush(stdout);
    }
    cudaDeviceSynchronize();

    printf("done\n");
    return 0;
}
