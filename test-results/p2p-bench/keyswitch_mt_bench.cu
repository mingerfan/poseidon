// Multi-thread, single-GPU keyswitch.hybrid launch-overhead benchmark.
//
// Same kernel workload as keyswitch_single_bench (full GpuEvaluator::rotate ->
// all 11 real keyswitch kernels, N=8192 q=17 p=2) but with N launch threads
// ALL on ONE GPU.
//
// Purpose: the real 4-GPU runtime has one launch thread per GPU (4 threads
// hitting libcuda simultaneously -> driver lock contention). A single-GPU
// machine cannot run 4 GPUs, but it CAN run N launch threads on one GPU and
// measure the same driver-side contention. Each thread's kernels go to its own
// per-thread stream (gpu_execution_stream() == cudaStreamPerThread), exactly
// like one real GPU's launch thread.
//
// Usage: keyswitch_mt_bench [rounds] [threads] [device]
//   rounds   rotate() rounds per thread (default 50)
//   threads  number of concurrent launch threads (default 1)
//   device   GPU index (default 0)
//
// Two phases (both in one process, timeline-ordered):
//   warmup : all threads do 3 rounds together (JIT cold-start, excluded)
//   measure: all threads do `rounds` rounds together
// Barrier (go counter) at each phase start so all threads contend from t=0.
//
// Measures via nsys: per-thread CPU cudaLaunchKernel duration and GPU kernel
// durations, threads=1 vs 2 vs 4 vs 8 on the same device.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <atomic>
#include <memory>
#include <vector>

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

void measure_thread_fn(BenchState &st, std::atomic<int> &go) {
    cudaSetDevice(st.gpu);
    barrier(go);                   // all threads start measure together
    gpu::GpuCiphertextData output;
    for (long i = 0; i < st.rounds && !st.stop.load(); ++i) {
        st.evaluator->rotate(*st.ct, st.rotate_step, *st.galois_keys, output);
        st.done.fetch_add(1);
    }
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
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back(measure_thread_fn, std::ref(st[t]), std::ref(go));
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
        fflush(stdout);
    }
    cudaDeviceSynchronize();

    printf("done\n");
    return 0;
}
