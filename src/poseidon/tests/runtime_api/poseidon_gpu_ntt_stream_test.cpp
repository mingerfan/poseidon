#include "poseidon/basics/util/ntt.h"
#include "poseidon/gpu/gpu_ntt_handler.h"
#include "poseidon/gpu/gpu_tensor_core_gemm.h"
#include "poseidon/poseidon_context.h"

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace poseidon;
using namespace poseidon::gpu;

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Stream
{
    cudaStream_t value = nullptr;
    Stream()
    {
        gpu_check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking),
                       "NTT test stream create");
    }
    ~Stream() { (void)cudaStreamDestroy(value); }
};

struct TestMemoryPool
{
    rmm::mr::cuda_memory_resource upstream;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool{
        &upstream, 128ULL << 20, 128ULL << 20};
    rmm::mr::device_memory_resource *previous =
        rmm::mr::set_current_device_resource(&pool);
    ~TestMemoryPool() { rmm::mr::set_current_device_resource(previous); }
};

struct PinnedWords
{
    GpuWord *data = nullptr;
    explicit PinnedWords(std::size_t count)
    {
        gpu_check_cuda(cudaMallocHost(&data, count * sizeof(GpuWord)),
                       "NTT test pinned output");
    }
    ~PinnedWords() { (void)cudaFreeHost(data); }
};

// Keep the caller's default stream blocked until the explicit stream has
// finished and its result has been copied back. A misplaced GEMM cannot race
// ahead and accidentally make this numerical regression pass.
struct DefaultStreamGate
{
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false, released = false, timed_out = false;

    static void CUDART_CB wait(void *opaque)
    {
        auto &gate = *static_cast<DefaultStreamGate *>(opaque);
        std::unique_lock<std::mutex> lock(gate.mutex);
        gate.entered = true;
        gate.condition.notify_all();
        // Bound failures caused by an unintended dependency on this stream.
        gate.timed_out = !gate.condition.wait_for(
            lock, std::chrono::seconds(5), [&] { return gate.released; });
    }

    void block()
    {
        gpu_check_cuda(cudaLaunchHostFunc(cudaStreamPerThread, wait, this),
                       "NTT test block default stream");
        std::unique_lock<std::mutex> lock(mutex);
        require(condition.wait_for(lock, std::chrono::seconds(1),
                                   [&] { return entered; }),
                "NTT test default stream gate did not start");
    }

    bool still_blocked()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return !released && !timed_out;
    }

    ~DefaultStreamGate()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            released = true;
            condition.notify_all();
        }
        (void)cudaStreamSynchronize(cudaStreamPerThread);
    }
};

void check_transform(const GpuNTTHandler &handler, const GpuLevelInfo &level,
                     cudaStream_t stream, const std::vector<std::uint64_t> &coefficients,
                     const std::vector<std::uint64_t> &ntt, bool inverse)
{
    const auto &input_words = inverse ? ntt : coefficients;
    const auto &expected = inverse ? coefficients : ntt;
    auto input = GpuPlaintextData::allocate_single_device(level.degree, level.q_count, 0);
    auto output = GpuPlaintextData::allocate_single_device(level.degree, level.q_count, 0);
    input.meta.parms_id = output.meta.parms_id = level.parms_id;
    input.meta.is_ntt_form = inverse;
    output.meta.is_ntt_form = !inverse;
    const std::vector<GpuWord> packed(input_words.begin(), input_words.end());
    auto source = input.make_const_view();
    auto destination = output.make_view();
    // CUDA's first kernel dispatch can load a module and synchronize other
    // streams. Warm this exact path before the gate. Zero input also prevents
    // old scratch results from satisfying the later nonzero numerical oracle.
    input.fields_.front().buffer.fill_zero();
    if (inverse) handler.inverse_plaintext(destination, source, level);
    else handler.forward_plaintext(destination, source, level);
    gpu_check_cuda(cudaDeviceSynchronize(), "NTT test warmup");
    input.fields_.front().buffer.copy_from_host(packed.data(), packed.size());
    PinnedWords actual(expected.size());
    gpu_check_cuda(cudaStreamSynchronize(stream), "NTT test input ready");

    std::size_t mismatches = 0;
    {
        DefaultStreamGate gate;
        gate.block();
        if (inverse) handler.inverse_plaintext(destination, source, level);
        else handler.forward_plaintext(destination, source, level);
        gpu_check_cuda(cudaMemcpyAsync(actual.data, output.fields_.front().data(),
                                       expected.size() * sizeof(GpuWord),
                                       cudaMemcpyDeviceToHost, stream),
                       "NTT test explicit-stream download");
        gpu_check_cuda(cudaStreamSynchronize(stream), "NTT test explicit-stream completion");
        require(gate.still_blocked(), "Tensor NTT unexpectedly waited for the default stream");
        for (std::size_t i = 0; i < expected.size(); ++i)
            if (actual.data[i] != expected[i]) ++mismatches;
    }
    if (mismatches != 0)
        throw std::runtime_error(std::string(inverse ? "inverse" : "forward") +
                                 " Tensor NTT residue mismatches=" + std::to_string(mismatches));
    std::cout << "[PASS] " << (inverse ? "inverse" : "forward")
              << " Tensor NTT: " << expected.size()
              << " CPU reference residues, default stream blocked\n";
}
}  // namespace

int main()
{
    try
    {
        if (!supports_tensor_core_integer_gemm(0))
        {
            std::cout << "[SKIP] Tensor NTT stream regression requires SM 7.5+\n";
            return 77;
        }
        gpu_check_cuda(cudaSetDevice(0), "NTT test device");
        setenv("POSEIDON_NTT_ALGO", "tensor", 1);
        setenv("POSEIDON_NTT_FUSION_STAGES", "4", 1);
        setenv("POSEIDON_NTT_FUSED_MATRIX_STAGES", "4", 1);
        TestMemoryPool pool;
        Stream stream;
        ScopedGpuExecutionStream execution(stream.value);
        ParametersLiteral literal(CKKS, 12, 11, 40, 0, 0, Modulus(0), {}, {},
                                  sec_level_type::none);
        literal.set_log_modulus({30, 30, 30}, {30});
        PoseidonContext context(literal);
        GpuParameterData parameters(context, 0, {3});
        GpuNTTHandler handler(parameters);
        const auto id = context.crt_context()->first_context_data()->parms_id();
        const auto &level = parameters.get_level(id);
        std::vector<std::uint64_t> coefficients(level.degree * level.q_count);
        const auto &moduli = context.parameters_literal()->q();
        for (std::size_t limb = 0; limb < level.q_count; ++limb)
            for (std::size_t i = 0; i < level.degree; ++i)
                coefficients[limb * level.degree + i] =
                    (31 * i * i + 17 * i + 101 * limb + 1) % moduli[limb].value();
        auto ntt = coefficients;
        const auto *tables = context.crt_context()->small_ntt_tables();
        for (std::size_t limb = 0; limb < level.q_count; ++limb)
            util::ntt_negacyclic_harvey(ntt.data() + limb * level.degree, tables[limb]);
        check_transform(handler, level, stream.value, coefficients, ntt, false);
        check_transform(handler, level, stream.value, coefficients, ntt, true);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] Tensor NTT stream regression: " << error.what() << '\n';
        return 1;
    }
}
