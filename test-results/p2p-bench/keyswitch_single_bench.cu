// Single-GPU keyswitch.hybrid launch-overhead benchmark.
//
// Same kernel workload as keyswitch_dual_bench (full GpuEvaluator::rotate ->
// all 11 real keyswitch kernels) but on ONE GPU with ONE launch thread.
// No P2P / comm thread (works on single-GPU machines like the local RTX 4060).
//
// Used to compare kernel LAUNCH overhead across machines:
//   - local RTX 4060
//   - 188Server V100
// via nsys (cudaLaunchKernel CPU duration).

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

struct BenchState {
    std::atomic<bool> stop{false};
    std::atomic<long> done{0};

    int gpu;
    long rounds;

    std::shared_ptr<PoseidonContext> context;
    std::unique_ptr<gpu::GpuParameterData> gpu_params;
    std::unique_ptr<gpu::GpuEvaluator> evaluator;
    gpu::GpuCiphertextData ct;
    gpu::GpuGaloisKeysData galois_keys;
    int rotate_step = 1;
};

void launch_thread_fn(BenchState &st) {
    cudaSetDevice(st.gpu);
    gpu::GpuCiphertextData output;
    for (long i = 0; i < st.rounds && !st.stop.load(); ++i) {
        st.evaluator->rotate(st.ct, st.rotate_step, st.galois_keys, output);
        st.done.fetch_add(1);
    }
}

int main(int argc, char **argv) {
    const std::size_t degree = 8192;
    const std::size_t q_count = 17;
    long rounds = 50;
    if (argc > 1) rounds = atol(argv[1]);

    int dev_count = 0;
    CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count < 1) { fprintf(stderr, "need 1 GPU\n"); return 1; }
    const int gpu = 0;
    cudaSetDevice(gpu);

    BenchState st;
    st.gpu = gpu;
    st.rounds = rounds;

    auto parms = make_benchmark_parameters(degree, q_count, /*p_count=*/2);
    st.context = std::make_shared<PoseidonContext>(parms);

    KeyGenerator keygen(*st.context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys galois_keys;
    const auto galois_tool = st.context->crt_context()->galois_tool();
    keygen.create_galois_keys(
        std::vector<std::uint32_t>{
            galois_tool->get_elt_from_step(1),
            galois_tool->get_elt_from_step(0)},
        galois_keys);

    CKKSEncoder encoder(*st.context);
    Encryptor encryptor(*st.context, public_key, keygen.secret_key());
    Plaintext plain;
    encoder.encode(std::vector<double>{1.0, 2.0, 3.0, 4.0},
                   parms.scale(), plain);
    Ciphertext ct;
    encryptor.encrypt(plain, ct);

    st.gpu_params = std::make_unique<gpu::GpuParameterData>();
    st.gpu_params->build_from_poseidon_context(*st.context, gpu);
    st.evaluator = std::make_unique<gpu::GpuEvaluator>(*st.gpu_params);
    st.ct = gpu::GpuUploader::upload_ciphertext(ct, gpu);
    st.galois_keys = gpu::GpuUploader::upload_galois_keys(galois_keys, gpu);

    printf("degree=%zu q=%zu rounds=%ld single-GPU\n", degree, q_count, rounds);
    fflush(stdout);

    // warmup: 3 rounds to trigger JIT, excluded from nsys analysis
    printf("[warmup] start\n"); fflush(stdout);
    {
        gpu::GpuCiphertextData output;
        for (int i = 0; i < 3; ++i) {
            st.evaluator->rotate(st.ct, st.rotate_step, st.galois_keys, output);
        }
    }
    cudaDeviceSynchronize();
    printf("[warmup] done\n"); fflush(stdout);

    // measured rounds
    printf("[measure] start\n"); fflush(stdout);
    {
        std::thread lt(launch_thread_fn, std::ref(st));
        while (st.done.load() < rounds)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        st.stop.store(true);
        lt.join();
    }
    cudaDeviceSynchronize();
    printf("[measure] done %ld\n", st.done.load());
    fflush(stdout);

    printf("done\n");
    return 0;
}
