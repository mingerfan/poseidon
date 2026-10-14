// Full keyswitch.hybrid dual-thread benchmark.
//
// launch thread: repeatedly calls GpuEvaluator::rotate() -> emits the FULL
//   keyswitch.hybrid kernel sequence (all 11 real kernels: hybrid_forward_ntt_
//   modup_qp_fused_stage, inverse_ntt_fourstep_phase, ... apply_galois,
//   dyadic_product, ...) on its GPU's compute stream.
// comm thread:   issues cudaMemcpyPeerAsync P2P copies on a dedicated stream.
//
// Modes:
//   no-comm: launch threads only
//   comm:    launch threads + comm threads concurrently
//
// Measures via nsys: launch (CPU cudaLaunchKernel) and compute (GPU kernel)
// times per kernel, no-comm vs comm.
//
// N=8192, q=17 (MLP config). Two GPUs, four threads.

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
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

// ---- per-GPU state ----
struct GpuState {
    std::atomic<int> go{0};
    std::atomic<bool> stop{false};
    std::atomic<long> done{0};

    int gpu;
    long rounds;

    // evaluator state (device-gpu)
    std::shared_ptr<PoseidonContext> context;
    std::unique_ptr<gpu::GpuParameterData> gpu_params;
    std::unique_ptr<gpu::GpuEvaluator> evaluator;
    gpu::GpuCiphertextData ct;
    gpu::GpuGaloisKeysData galois_keys;
    int rotate_step = 1;

    // comm state
    gpu::GpuWord *p2p_src, *p2p_dst;
    cudaStream_t copy_stream;
    std::size_t p2p_chunk;
    int p2p_copies;
};

void launch_thread_fn(GpuState &st) {
    cudaSetDevice(st.gpu);
    st.go.fetch_sub(1);
    while (st.go.load() != 0) {}
    gpu::GpuCiphertextData output;
    for (long i = 0; i < st.rounds && !st.stop.load(); ++i) {
        st.evaluator->rotate(st.ct, st.rotate_step, st.galois_keys, output);
        st.done.fetch_add(1);
    }
}

void comm_thread_fn(GpuState &st) {
    cudaSetDevice(st.gpu);
    st.go.fetch_sub(1);
    while (st.go.load() != 0) {}
    const int other = 1 - st.gpu;
    while (!st.stop.load()) {
        for (int c = 0; c < st.p2p_copies; ++c) {
            cudaError_t e = cudaMemcpyPeerAsync(
                (char*)st.p2p_dst + (std::size_t)c * st.p2p_chunk, other,
                (char*)st.p2p_src + (std::size_t)c * st.p2p_chunk, st.gpu,
                st.p2p_chunk, st.copy_stream);
            if (e != cudaSuccess) { fprintf(stderr, "p2p err: %s\n", cudaGetErrorString(e)); exit(1); }
        }
    }
}

int main(int argc, char **argv) {
    const std::size_t degree = 8192;
    const std::size_t q_count = 17;
    long rounds = 50;
    if (argc > 1) rounds = atol(argv[1]);

    int dev_count = 0;
    CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count < 2) { fprintf(stderr, "need 2 GPUs\n"); return 1; }
    cudaSetDevice(0);
    int can = 0; CHECK(cudaDeviceCanAccessPeer(&can, 0, 1));
    if (!can) { fprintf(stderr, "P2P 0<->1 not supported\n"); return 1; }
    cudaSetDevice(0); CHECK(cudaDeviceEnablePeerAccess(1, 0));
    cudaSetDevice(1); CHECK(cudaDeviceEnablePeerAccess(0, 0));

    const std::size_t p2p_chunk = 2u << 20;
    const int p2p_copies = 16;
    const std::size_t p2p_bytes = 64u << 20;

    GpuState st[2];
    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        st[g].gpu = g;
        st[g].rounds = rounds;
        st[g].p2p_chunk = p2p_chunk;
        st[g].p2p_copies = p2p_copies;

        // Build context + GPU params (real poseidon stack)
        auto parms = make_benchmark_parameters(degree, q_count, /*p_count=*/2);
        st[g].context = std::make_shared<PoseidonContext>(parms);

        KeyGenerator keygen(*st[g].context);
        PublicKey public_key;
        keygen.create_public_key(public_key);
        RelinKeys relin_keys;
        keygen.create_relin_keys(relin_keys);
        GaloisKeys galois_keys;
        const auto galois_tool = st[g].context->crt_context()->galois_tool();
        keygen.create_galois_keys(
            std::vector<std::uint32_t>{
                galois_tool->get_elt_from_step(1),
                galois_tool->get_elt_from_step(0)},
            galois_keys);

        CKKSEncoder encoder(*st[g].context);
        Encryptor encryptor(*st[g].context, public_key, keygen.secret_key());
        Plaintext plain;
        encoder.encode(std::vector<double>{1.0, 2.0, 3.0, 4.0},
                       parms.scale(), plain);
        Ciphertext ct;
        encryptor.encrypt(plain, ct);

        st[g].gpu_params = std::make_unique<gpu::GpuParameterData>();
        st[g].gpu_params->build_from_poseidon_context(*st[g].context, g);
        st[g].evaluator = std::make_unique<gpu::GpuEvaluator>(*st[g].gpu_params);
        st[g].ct = gpu::GpuUploader::upload_ciphertext(ct, g);
        st[g].galois_keys = gpu::GpuUploader::upload_galois_keys(galois_keys, g);

        CHECK(cudaMalloc(&st[g].p2p_src, p2p_bytes));
        CHECK(cudaMalloc(&st[g].p2p_dst, p2p_bytes));
        CHECK(cudaStreamCreate(&st[g].copy_stream));
    }

    printf("degree=%zu q=%zu rounds=%ld P2P=2MBx%d\n", degree, q_count, rounds, p2p_copies);
    fflush(stdout);

    // MODE 0: no-comm
    {
        printf("[mode=no-comm] start\n"); fflush(stdout);
        for (int g = 0; g < 2; ++g) { st[g].go.store(1); st[g].stop.store(false); st[g].done.store(0); }
        std::thread l0(launch_thread_fn, std::ref(st[0]));
        std::thread l1(launch_thread_fn, std::ref(st[1]));
        while (st[0].done.load() < rounds || st[1].done.load() < rounds)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        st[0].stop.store(true); st[1].stop.store(true);
        l0.join(); l1.join();
        printf("[mode=no-comm] done %ld/%ld\n", st[0].done.load(), st[1].done.load());
        fflush(stdout);
        for (int g = 0; g < 2; ++g) { cudaSetDevice(g); CHECK(cudaDeviceSynchronize()); }
    }

    // MODE 1: comm
    {
        printf("[mode=comm] start\n"); fflush(stdout);
        for (int g = 0; g < 2; ++g) { st[g].go.store(2); st[g].stop.store(false); st[g].done.store(0); }
        std::thread l0(launch_thread_fn, std::ref(st[0]));
        std::thread l1(launch_thread_fn, std::ref(st[1]));
        std::thread c0(comm_thread_fn, std::ref(st[0]));
        std::thread c1(comm_thread_fn, std::ref(st[1]));
        while (st[0].done.load() < rounds || st[1].done.load() < rounds)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        st[0].stop.store(true); st[1].stop.store(true);
        l0.join(); l1.join(); c0.join(); c1.join();
        printf("[mode=comm] done %ld/%ld\n", st[0].done.load(), st[1].done.load());
        fflush(stdout);
        for (int g = 0; g < 2; ++g) { cudaSetDevice(g); CHECK(cudaDeviceSynchronize()); }
    }

    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        CHECK(cudaFree(st[g].p2p_src)); CHECK(cudaFree(st[g].p2p_dst));
        CHECK(cudaStreamDestroy(st[g].copy_stream));
    }
    printf("done\n");
    return 0;
}
