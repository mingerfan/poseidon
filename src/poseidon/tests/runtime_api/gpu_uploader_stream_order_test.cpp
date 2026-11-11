// Standalone nonblocking-stream uploader regression; not registered by this task.
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/poseidon_context.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace poseidon;
using namespace poseidon::gpu;
namespace
{
struct Gate
{
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false;
    bool released = false;
};
}
int main()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try
    {
        ParametersLiteral params{CKKS, 12, 11, 44, 64, 0, 0, {}, {}, sec_level_type::none};
        params.set_log_modulus({29, 30, 31, 32}, {30, 31});
        PoseidonContext context(params);
        KeyGenerator generator(context);
        PublicKey public_key;
        generator.create_public_key(public_key);
        CKKSEncoder encoder(context);
        Encryptor encryptor(context, public_key);
        Plaintext plain;
        encoder.encode(std::vector<std::complex<double>>(encoder.slot_count(), {0.1, 0.2}),
                       std::ldexp(1.0, 44), plain);
        Ciphertext source;
        encryptor.encrypt(plain, source);
        cudaStream_t stream{};
        gpu_check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        {
            ScopedGpuExecutionStream scope(stream);
            auto gpu = GpuUploader::upload_ciphertext(source, 0);
            Gate gate;
            gpu_check_cuda(cudaLaunchHostFunc(stream, [](void *opaque) {
                auto &g = *static_cast<Gate *>(opaque);
                std::unique_lock<std::mutex> lock(g.mutex);
                g.entered = true;
                g.cv.notify_all();
                g.cv.wait(lock, [&] { return g.released; });
            }, &gate), "launch gate");
            for (auto &field : gpu.fields_)
                gpu_check_cuda(cudaMemsetAsync(field.data(), 0, field.size() * sizeof(GpuWord), stream),
                               "enqueue producer");
            std::thread release([&] {
                std::unique_lock<std::mutex> lock(gate.mutex);
                gate.cv.wait_for(lock, std::chrono::seconds(2), [&] { return gate.entered; });
                gate.cv.wait_for(lock, std::chrono::milliseconds(100));
                gate.released = true;
                gate.cv.notify_all();
            });
            Ciphertext result;
            try { GpuUploader::download_ciphertext(gpu, result, context); }
            catch (...) { release.join(); cudaStreamSynchronize(stream); throw; }
            release.join();
            const auto words = result.size() * result.poly_modulus_degree() * result.coeff_modulus_size();
            if (!std::all_of(result.data(), result.data() + words, [](auto x) { return x == 0; }))
                throw std::runtime_error("readback missed execution-stream producer");
        }
        gpu_check_cuda(cudaStreamDestroy(stream), "destroy stream");
        std::cout << "PASS uploader execution-stream readback\n";
        return 0;
    }
    catch (const std::exception &e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
