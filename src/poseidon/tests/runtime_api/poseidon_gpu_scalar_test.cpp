#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using poseidon::gpu::GpuCiphertextData;
using poseidon::gpu::GpuWord;

void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
void check(cudaError_t status, const char *message) {
    poseidon::gpu::gpu_check_cuda(status, message);
}

struct Stream {
    cudaStream_t value{};
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create scalar stream"); }
    ~Stream() { if (value) (void)cudaStreamDestroy(value); }
};

// This callback never calls CUDA. The watchdog only bounds a failed test.
struct Gate {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered=false, released=false, timed_out=false;
    std::thread watchdog;
    Gate() : watchdog([this] {
        std::unique_lock<std::mutex> lock(mutex);
        if (!condition.wait_for(lock, std::chrono::seconds(2), [&] { return released; })) {
            timed_out=true; released=true; condition.notify_all();
        }
    }) {}
    void release() {
        std::lock_guard<std::mutex> lock(mutex); released=true; condition.notify_all();
    }
    ~Gate() { release(); if (watchdog.joinable()) watchdog.join(); }
    static void CUDART_CB wait(void *opaque) {
        auto &gate=*static_cast<Gate *>(opaque);
        std::unique_lock<std::mutex> lock(gate.mutex);
        gate.entered=true; gate.condition.notify_all();
        gate.condition.wait(lock, [&] { return gate.released; });
    }
};

std::size_t check_words(const poseidon::Ciphertext &source,
                        const poseidon::Ciphertext &actual,
                        const std::vector<poseidon::Modulus> &moduli,
                        std::uint64_t scalar) {
    require(actual.scale()==source.scale(), "integer scalar changed scale metadata");
    require(actual.parms_id()==source.parms_id() && actual.size()==source.size(),
            "integer scalar changed shape metadata");
    const auto n=source.poly_modulus_degree(), q=source.coeff_modulus_size();
    std::size_t checked=0;
    for (std::size_t c=0;c<source.size();++c)
        for (std::size_t limb=0;limb<q;++limb)
            for (std::size_t i=0;i<n;++i) {
                const auto offset=limb*n+i;
                // Independent CPU oracle multiplies the original uint64
                // scalar in 128 bits, then reduces. No truncation or retag.
                const auto expected=static_cast<std::uint64_t>(
                    (static_cast<unsigned __int128>(source.data(c)[offset])*scalar)%moduli[limb].value());
                if (actual.data(c)[offset]!=expected)
                    throw std::runtime_error("wide scalar RNS word mismatch scalar="+std::to_string(scalar)+
                        " component="+std::to_string(c)+" limb="+std::to_string(limb)+" coeff="+std::to_string(i)+
                        " source="+std::to_string(source.data(c)[offset])+" q="+std::to_string(moduli[limb].value())+
                        " expected="+std::to_string(expected)+" actual="+std::to_string(actual.data(c)[offset]));
                ++checked;
            }
    return checked;
}
}

int main() {
    try {
        setenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB", "512", 1);
        setenv("POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB", "64", 1);
        int count=0; check(cudaGetDeviceCount(&count), "scalar visible devices");
        require(count==1, "scalar test requires one explicitly selected GPU");
        check(cudaSetDevice(0), "scalar device");
        Stream stream;
        poseidon::gpu::ScopedGpuExecutionStream selected(stream.value);
        poseidon::ParametersLiteral literal(CKKS,12,11,40,0,0,poseidon::Modulus(0),{},{},
                                            poseidon::sec_level_type::none);
        literal.set_log_modulus(std::vector<std::uint32_t>(7,30),std::vector<std::uint32_t>(2,30));
        poseidon::PoseidonContext context(literal);
        poseidon::KeyGenerator generator(context);
        poseidon::PublicKey public_key; generator.create_public_key(public_key);
        poseidon::CKKSEncoder encoder(context);
        poseidon::Encryptor encryptor(context,public_key);
        poseidon::Decryptor decryptor(context,generator.secret_key());
        // Reuse the production bounded resource owner; direct evaluator calls
        // below exercise the scalar primitive without any Rescale wrapper.
        poseidon::runtime_api::PoseidonGpuApi api("wide-scalar-test",context,0);
        poseidon::gpu::GpuParameterData parameters(context,0);
        poseidon::gpu::GpuEvaluator evaluator(parameters);
        std::vector<std::complex<double>> slots(2048);
        for (std::size_t i=0;i<slots.size();++i)
            slots[i]={.125*std::sin(i*.071)+.025, .0625*std::cos(i*.037)-.015625};
        poseidon::Plaintext plain; encoder.encode(slots,std::ldexp(1.,40),plain);
        poseidon::Ciphertext source; encryptor.encrypt(plain,source);
        poseidon::Plaintext original_plain; decryptor.decrypt(source,original_plain);
        std::vector<std::complex<double>> original_slots; encoder.decode(original_plain,original_slots);
        const auto &moduli=context.parameters_literal()->q();
        const std::vector<std::uint64_t> scalars={0,1,17,std::numeric_limits<std::uint32_t>::max(),
            std::uint64_t(1)<<32,(std::uint64_t(1)<<34)+123,
            (std::uint64_t(1)<<63)+129,std::numeric_limits<std::uint64_t>::max()};
        std::size_t checked=0; double maximum_normalized_error=0;
        for (auto scalar:scalars) for (bool alias:{false,true}) {
            auto input=poseidon::gpu::GpuUploader::upload_ciphertext(source,0);
            // The direct uploader's packed host copies use the C++ CUDA
            // default stream; establish readiness before another stream.
            check(cudaStreamSynchronize(cudaStreamLegacy), "scalar legacy uploaded input ready");
            check(cudaStreamSynchronize(cudaStreamPerThread), "scalar uploaded input ready");
            GpuCiphertextData output;
            auto &result=alias?input:output;
            evaluator.multiply_scalar(input,scalar,result);
            check(cudaStreamSynchronize(stream.value), "scalar completion");
            poseidon::Ciphertext actual;
            poseidon::gpu::GpuUploader::download_ciphertext(result,actual,context);
            checked+=check_words(source,actual,moduli,scalar);
            poseidon::Plaintext decoded_plain; decryptor.decrypt(actual,decoded_plain);
            std::vector<std::complex<double>> actual_slots; encoder.decode(decoded_plain,actual_slots);
            require(actual_slots.size()==slots.size(), "wide scalar decode slot count");
            for (std::size_t i=0;i<actual_slots.size();++i) {
                const auto expected=scalar?original_slots[i]:std::complex<double>{};
                const auto normalized=scalar?actual_slots[i]/static_cast<double>(scalar):actual_slots[i];
                const double error=std::abs(normalized-expected);
                require(std::isfinite(error) && error<2e-9, "wide scalar independent decoded mismatch");
                maximum_normalized_error=std::max(maximum_normalized_error,error);
            }
        }

        // Caller PTDS remains blocked while a nondefault nonblocking stream
        // completes an aliased wide multiply and explicit D2H word validation.
        auto input=poseidon::gpu::GpuUploader::upload_ciphertext(source,0);
        check(cudaStreamSynchronize(cudaStreamLegacy), "stream oracle legacy uploaded input ready");
        check(cudaStreamSynchronize(cudaStreamPerThread), "stream oracle uploaded input ready");
        check(cudaStreamSynchronize(stream.value), "stream oracle ready input");
        const auto words=source.size()*source.coeff_modulus_size()*source.poly_modulus_degree();
        GpuWord *host_words=nullptr;
        check(cudaMallocHost(&host_words,words*sizeof(GpuWord)), "stream oracle pinned output");
        Gate gate;
        try {
            check(cudaLaunchHostFunc(cudaStreamPerThread,Gate::wait,&gate), "gate caller PTDS");
            { std::unique_lock<std::mutex> lock(gate.mutex);
              require(gate.condition.wait_for(lock,std::chrono::seconds(1),[&] {return gate.entered;}),
                      "caller PTDS gate never entered"); }
            const auto scalar=(std::uint64_t(1)<<34)+123;
            evaluator.multiply_scalar(input,scalar,input);
            const auto view=input.make_const_view(); std::size_t offset=0;
            for (const auto &poly:view.polys) for (const auto &shard:poly.shards) {
                const auto size=shard.limb_count*shard.coeff_count;
                check(cudaMemcpyAsync(host_words+offset,shard.ptr,size*sizeof(GpuWord),cudaMemcpyDeviceToHost,
                                      stream.value), "wide scalar nondefault D2H"); offset+=size;
            }
            check(cudaStreamSynchronize(stream.value), "wide scalar nondefault completion");
            { std::lock_guard<std::mutex> lock(gate.mutex);
              require(!gate.released && !gate.timed_out, "wide scalar waited for caller PTDS"); }
            const auto n=source.poly_modulus_degree(),q=source.coeff_modulus_size();
            for (std::size_t c=0;c<source.size();++c) for(std::size_t limb=0;limb<q;++limb)
                for(std::size_t i=0;i<n;++i) {
                    const auto expected=static_cast<GpuWord>((static_cast<unsigned __int128>(
                        source.data(c)[limb*n+i])*scalar)%moduli[limb].value());
                    require(host_words[(c*q+limb)*n+i]==expected,"nondefault stream scalar RNS mismatch");
                    ++checked;
                }
        } catch (...) {
            gate.release(); (void)cudaStreamSynchronize(cudaStreamPerThread);
            (void)cudaFreeHost(host_words); throw;
        }
        gate.release();check(cudaStreamSynchronize(cudaStreamPerThread),"release scalar gate");
        check(cudaFreeHost(host_words),"release scalar oracle pinned");api.drain();
        std::cout << "[PASS] GPU uint64 scalar: cases=" << scalars.size()*2
                  << " checked_RNS_words=" << checked << " decoded_complex_slots=" << slots.size()
                  << " max_normalized_error=" << maximum_normalized_error
                  << " alias=true PTDS_blocked_nondefault_completed=true max_scalar="
                  << std::numeric_limits<std::uint64_t>::max() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] GPU uint64 scalar: " << error.what() << '\n'; return 1;
    }
}
