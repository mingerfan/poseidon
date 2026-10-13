// Public API regression: AddCP's unchanged c1 must be ready on Waterline's
// foreground stream even while the caller's per-thread default stream is held.
// This is a small synthetic dependency test, not a complete model test.
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

using poseidon::runtime_api::PoseidonGpuApi;
using poseidon::runtime_api::PoseidonGpuValue;
using Clock = std::chrono::steady_clock;
using Word = poseidon::gpu::GpuWord;

void require(bool condition, const char *reason) {
    if (!condition) throw std::runtime_error(reason);
}
void cuda_check(cudaError_t result, const char *reason) {
    poseidon::gpu::gpu_check_cuda(result, reason);
}
struct Gate {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false, submitted = false, finished = false, watchdog = false;
};
struct GateLifetime {
    // Declared after Gate, so callback data survives every failure path.
    ~GateLifetime() { (void)cudaStreamSynchronize(cudaStreamPerThread); }
};
void CUDART_CB hold_caller_ptds(void *opaque) {
    auto &gate = *static_cast<Gate *>(opaque);
    std::unique_lock<std::mutex> lock(gate.mutex);
    gate.entered = true;
    gate.condition.notify_all();
    // Waiting here never calls CUDA. A watchdog also releases the callback if
    // submission blocks; that becomes a test failure instead of a deadlock.
    if (!gate.condition.wait_for(lock, std::chrono::seconds(1), [&] {return gate.submitted;}))
        gate.watchdog = true;
    lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    lock.lock();
    gate.finished = true;
    gate.condition.notify_all();
}

int main() {
    nlohmann::json report{{"passed",false}, {"scope","public AddCP Ready under blocked caller PTDS"}};
    Word *expected = nullptr, *actual = nullptr;
    cudaStream_t verifier = nullptr;
    try {
        setenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB", "128", 1);
        auto moduli = poseidon::CoeffModulus::Create(4096, std::vector<int>(9,30));
        poseidon::ParametersLiteral parameters(CKKS,12,11,40,0,0,poseidon::Modulus(0),
            std::vector<poseidon::Modulus>(moduli.begin(),moduli.begin()+7),
            std::vector<poseidon::Modulus>(moduli.begin()+7,moduli.end()),poseidon::sec_level_type::none);
        poseidon::PoseidonContext context(parameters);
        poseidon::KeyGenerator keys(context);
        poseidon::PublicKey public_key; keys.create_public_key(public_key);
        poseidon::CKKSEncoder encoder(context); poseidon::Encryptor encryptor(context,public_key);
        poseidon::Plaintext encoded, add_encoded;
        encoder.encode(std::vector<double>{.25,-.5,1.25},std::ldexp(1.0,40),encoded);
        encoder.encode(std::vector<double>{.5,.25,-.75},std::ldexp(1.0,40),add_encoded);
        poseidon::Ciphertext encrypted; encryptor.encrypt(encoded,encrypted);
        PoseidonGpuApi api("ptds-copy-test",context,0);
        fhegpu::OperatorSpec spec; spec.id="ptds-copy-test";spec.version=1;spec.status="validated";
        spec.target_id="poseidon-ckks-gpu";spec.rescale_mode=fhegpu::RescaleMode::Lazy;
        spec.context_id="ptds-copy-test";spec.poly_degree=4096;spec.rns_moduli_log2=std::vector<int>(7,30);
        spec.max_modulus_log2=30;spec.default_scale_log2=40;spec.level_upper_bound=6;
        spec.operators[fhegpu::ComputeKind::Rescale].supported=true;
        spec.operators[fhegpu::ComputeKind::Rescale].max_levels_per_op=4;
        const std::string digest="sha256:"+std::string(64,'1');
        fhegpu::TargetConfig target;target.target_id=spec.target_id;target.device_counts={1};target.capability_version=1;
        target.operator_spec={spec.id,spec.version,digest};
        api.preflight(digest,false,target,spec,{});
        fhegpu::RuntimePlan plan;plan.target=target;
        fhegpu::WaterlineConfig config;config.device_capacity_bytes={4ULL<<30};
        config.device_reserve_bytes={512ULL<<20};config.device_fractions={.1};
        config.host_budget_bytes=64ULL<<20;config.pinned_budget_bytes=16ULL<<20;config.raw_budget_bytes=4ULL<<20;
        plan.waterline=config;api.validate_waterline_resources(plan);
        auto source=PoseidonGpuValue::from_device_ciphertext(poseidon::gpu::GpuUploader::upload_ciphertext(encrypted,0));
        auto plain=PoseidonGpuValue::from_device_plaintext(poseidon::gpu::GpuUploader::upload_plaintext(add_encoded,0));
        fhegpu::ComputeOp add;add.kind=fhegpu::ComputeKind::AddCP;
        add.place={fhegpu::PlaceKind::Device,0,0};
        const auto source_shard=source.device_ciphertext().make_const_view().polys.at(1).shards.at(0);
        const auto words=source_shard.limb_count*source_shard.coeff_count;
        const auto bytes=words*sizeof(Word);
        cuda_check(cudaStreamCreateWithFlags(&verifier,cudaStreamNonBlocking),"verification stream");
        cuda_check(cudaMallocHost(reinterpret_cast<void **>(&expected),bytes),"expected pinned storage");
        cuda_check(cudaMallocHost(reinterpret_cast<void **>(&actual),bytes),"actual pinned storage");
        cuda_check(cudaMemcpyAsync(expected,source_shard.ptr,bytes,cudaMemcpyDeviceToHost,verifier),"source c1 snapshot");
        cuda_check(cudaStreamSynchronize(verifier),"snapshot completion");
        require(std::any_of(expected,expected+words,[](Word x){return x!=0;}),"source c1 is unexpectedly zero");
        nlohmann::json rounds=nlohmann::json::array();
        for (int round=0;round<5;++round) {
            // Warm all lazy allocation paths. Poison the free output's c1 so a
            // late copy cannot accidentally pass by reusing a previous AddCP.
            std::optional<PoseidonGpuValue> warm=api.compute(add,{source,plain});api.synchronize(*warm);
            const auto warm_shard=warm->device_ciphertext().make_view().polys.at(1).shards.at(0);
            cuda_check(cudaMemsetAsync(warm_shard.ptr,0,bytes,verifier),"poison warmed output c1");
            cuda_check(cudaStreamSynchronize(verifier),"poison completion");
            warm.reset();api.collect_completed();
            std::fill(actual,actual+words,0);
            Gate gate;
            GateLifetime gate_lifetime;
            cuda_check(cudaLaunchHostFunc(cudaStreamPerThread,hold_caller_ptds,&gate),"hold caller PTDS");
            {
                std::unique_lock<std::mutex> lock(gate.mutex);
                require(gate.condition.wait_for(lock,std::chrono::seconds(1),[&]{return gate.entered;}),"PTDS gate did not enter");
            }
            const auto begin=Clock::now();
            auto result=api.compute(add,{source,plain});
            const auto submit_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
            {
                std::lock_guard<std::mutex> lock(gate.mutex);gate.submitted=true;gate.condition.notify_all();
            }
            api.synchronize(result);
            const auto result_shard=result.device_ciphertext().make_const_view().polys.at(1).shards.at(0);
            cuda_check(cudaMemcpyAsync(actual,result_shard.ptr,bytes,cudaMemcpyDeviceToHost,verifier),"ready c1 snapshot");
            cuda_check(cudaStreamSynchronize(verifier),"ready snapshot completion");
            const auto ready_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
            bool gate_still_blocked=false,watchdog=false;
            {
                std::lock_guard<std::mutex> lock(gate.mutex);
                gate_still_blocked=!gate.finished;watchdog=gate.watchdog;
            }
            const auto mismatches=std::inner_product(expected,expected+words,actual,std::size_t(0),
                std::plus<std::size_t>{},[](Word a,Word b){return std::size_t(a!=b);});
            // Always finish the callback before its stack data is destroyed.
            cuda_check(cudaStreamSynchronize(cudaStreamPerThread),"finish PTDS gate");
            rounds.push_back({{"round",round},{"submit_ms",submit_ms},{"ready_snapshot_ms",ready_ms},
                {"gate_still_blocked_at_snapshot",gate_still_blocked},{"words_checked",words},
                {"c1_mismatches",mismatches},{"reused_warmed_c1_address",warm_shard.ptr==result_shard.ptr},
                {"watchdog",watchdog}});
            report["rounds"]=rounds;
            require(!watchdog && submit_ms<40 && ready_ms<40 && gate_still_blocked,"foreground AddCP waited for caller PTDS");
            require(mismatches==0,"Ready published before unchanged c1 copy completed");
        }
        api.drain();report["passed"]=true;report["injected_delay_ms"]=80;
    } catch(const std::exception &error) {report["error"]=error.what();}
    if (actual) cudaFreeHost(actual);
    if (expected) cudaFreeHost(expected);
    if (verifier) cudaStreamDestroy(verifier);
    std::cout<<report.dump(2)<<'\n';
    return report["passed"].get<bool>()?0:1;
}
