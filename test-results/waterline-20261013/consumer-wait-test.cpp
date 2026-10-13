// Standalone synthetic GPU dependency test. Uses only public Poseidon APIs.
// Producer API keeps the caller's low stream; consumer API enables Waterline.
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using poseidon::runtime_api::PoseidonGpuApi;
using poseidon::runtime_api::PoseidonGpuValue;
using Clock=std::chrono::steady_clock;

void require(bool condition,const char *reason) {
    if(!condition)throw std::runtime_error(reason);
}
struct Gate {
    std::mutex mutex;std::condition_variable condition;
    bool entered=false,submitted=false,release=false,finished=false,watchdog=false;
};
void CUDART_CB delay_producer(void *opaque) {
    auto &gate=*static_cast<Gate *>(opaque);
    std::unique_lock<std::mutex> lock(gate.mutex);
    gate.entered=true;gate.condition.notify_all();
    gate.condition.wait(lock,[&]{return gate.release;});
    gate.finished=true;gate.condition.notify_all();
}
class ReleaseGate {
public:
    ReleaseGate(cudaStream_t stream,Gate &gate):gate_(gate) {
        poseidon::gpu::gpu_check_cuda(cudaLaunchHostFunc(stream,delay_producer,&gate_),"delay producer");
        std::unique_lock<std::mutex> lock(gate_.mutex);
        require(gate_.condition.wait_for(lock,std::chrono::seconds(1),[&]{return gate_.entered;}),"producer gate did not start");
        thread_=std::thread([this] {
            std::unique_lock<std::mutex> lock(gate_.mutex);
            if(!gate_.condition.wait_for(lock,std::chrono::seconds(1),[&]{return gate_.submitted;}))gate_.watchdog=true;
            lock.unlock();std::this_thread::sleep_for(std::chrono::milliseconds(80));lock.lock();
            gate_.release=true;gate_.condition.notify_all();
        });
    }
    void submitted() {std::lock_guard<std::mutex> lock(gate_.mutex);gate_.submitted=true;gate_.condition.notify_all();}
    void finish() {
        if(thread_.joinable())thread_.join();
        std::unique_lock<std::mutex> lock(gate_.mutex);
        gate_.condition.wait(lock,[&]{return gate_.finished;});
    }
    ~ReleaseGate() {finish();}
private:
    Gate &gate_;std::thread thread_;
};
class CollectLoop {
public:
    CollectLoop(PoseidonGpuApi &producer,PoseidonGpuApi &consumer):thread_([&,this] {
        try {
            while(!stop_.load()) {producer.collect_completed();consumer.collect_completed();std::this_thread::yield();}
        } catch(...) {failure_=std::current_exception();}
    }) {}
    void finish() {stop_=true;if(thread_.joinable())thread_.join();if(failure_)std::rethrow_exception(failure_);}
    ~CollectLoop() {stop_=true;if(thread_.joinable())thread_.join();}
private:
    std::atomic<bool> stop_{false};std::exception_ptr failure_;std::thread thread_;
};

int main(int argc,char **argv) {
    if(argc!=2) {std::cerr<<"usage: consumer-wait-test REPORT_JSON\n";return 2;}
    nlohmann::json report{{"passed",false},{"scope","synthetic GPU dependency measurement; not a model test"}};
    try {
        // Includes actual persistent/key/encoder reserve and the 64 MiB
        // allocator headroom validated by the public Waterline API.
        setenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB","128",1);
        const auto moduli=poseidon::CoeffModulus::Create(4096,std::vector<int>(9,30));
        poseidon::ParametersLiteral parameters(CKKS,12,11,40,0,0,poseidon::Modulus(0),
            std::vector<poseidon::Modulus>(moduli.begin(),moduli.begin()+7),
            std::vector<poseidon::Modulus>(moduli.begin()+7,moduli.end()),poseidon::sec_level_type::none);
        poseidon::PoseidonContext context(parameters);poseidon::KeyGenerator keys(context);
        auto relin=std::make_shared<poseidon::RelinKeys>();keys.create_relin_keys(*relin);
        auto galois=std::make_shared<poseidon::GaloisKeys>();keys.create_galois_keys(std::vector<int>{1},*galois);
        poseidon::PublicKey public_key;keys.create_public_key(public_key);
        poseidon::CKKSEncoder encoder(context);poseidon::Encryptor encryptor(context,public_key);
        poseidon::Decryptor decryptor(context,keys.secret_key());
        const std::vector<double> prefix{.25,-.5,1.25},addend{.5,.25,-.75};
        std::vector<double> original(encoder.slot_count());std::copy(prefix.begin(),prefix.end(),original.begin());
        poseidon::Plaintext encoded;encoder.encode(prefix,std::ldexp(1.0,40),encoded);
        poseidon::Ciphertext encrypted;encryptor.encrypt(encoded,encrypted);
        PoseidonGpuApi producer("wait-event-test",context,0,relin,galois);
        PoseidonGpuApi consumer("wait-event-test",context,0,relin,galois);
        const fhegpu::Place device{fhegpu::PlaceKind::Device,0,0};
        fhegpu::OperatorSpec spec;spec.id="wait-event-test";spec.version=1;spec.status="validated";
        spec.target_id="poseidon-ckks-gpu";spec.rescale_mode=fhegpu::RescaleMode::Lazy;
        spec.context_id="wait-event-test";spec.poly_degree=4096;spec.rns_moduli_log2=std::vector<int>(7,30);
        spec.max_modulus_log2=30;spec.default_scale_log2=40;spec.level_upper_bound=6;
        spec.operators[fhegpu::ComputeKind::Rescale].supported=true;
        spec.operators[fhegpu::ComputeKind::Rescale].max_levels_per_op=4;
        const std::string digest="sha256:"+std::string(64,'1');
        fhegpu::TargetConfig target;target.target_id=spec.target_id;target.device_counts={1};target.capability_version=1;
        target.operator_spec={spec.id,spec.version,digest};
        fhegpu::PlanRequirements requirements;
        requirements.keys={{fhegpu::KeyKind::Relin,device,std::nullopt,6},
                           {fhegpu::KeyKind::Galois,device,1,6}};
        producer.preflight(digest,false,target,spec,requirements);
        consumer.preflight(digest,false,target,spec,requirements);
        fhegpu::RuntimePlan configuration;configuration.target=target;
        fhegpu::WaterlineConfig waterline;waterline.device_capacity_bytes={4ULL<<30};
        waterline.device_reserve_bytes={512ULL<<20};waterline.device_fractions={.1};
        waterline.host_budget_bytes=64ULL<<20;waterline.pinned_budget_bytes=16ULL<<20;waterline.raw_budget_bytes=4ULL<<20;
        configuration.waterline=waterline;consumer.validate_waterline_resources(configuration);
        auto source=PoseidonGpuValue::from_device_ciphertext(poseidon::gpu::GpuUploader::upload_ciphertext(encrypted,0));
        poseidon::Plaintext add_plain;encoder.encode(addend,std::ldexp(1.0,40),add_plain);
        auto plain=PoseidonGpuValue::from_device_plaintext(poseidon::gpu::GpuUploader::upload_plaintext(add_plain,0));
        fhegpu::ComputeOp negate;negate.kind=fhegpu::ComputeKind::Negate;negate.place=device;
        fhegpu::ComputeOp add;add.kind=fhegpu::ComputeKind::AddCP;add.place=device;
        fhegpu::ComputeOp rotate;rotate.kind=fhegpu::ComputeKind::Rotate;rotate.place=device;rotate.attrs=fhegpu::RotateAttrs{1};
        fhegpu::ComputeOp multiply;multiply.kind=fhegpu::ComputeKind::MulCC;multiply.place=device;
        auto product=producer.compute(multiply,{source,source});producer.synchronize(product);
        fhegpu::ComputeOp relinearize;relinearize.kind=fhegpu::ComputeKind::Relinearize;relinearize.place=device;
        auto squared=consumer.compute(relinearize,{product});consumer.synchronize(squared);
        fhegpu::ComputeOp rescale;rescale.kind=fhegpu::ComputeKind::Rescale;rescale.place=device;
        rescale.attrs=fhegpu::RescaleAttrs{5,50};
        // Warm lazy allocations, key views and every kernel before the gate.
        for(const auto &op:{negate,rotate,rescale}) {
            auto value=consumer.compute(op,{op.kind==fhegpu::ComputeKind::Rescale?squared:source});consumer.synchronize(value);
        }
        auto add_warm=consumer.compute(add,{source,plain});consumer.synchronize(add_warm);
        auto reuse_op=add;reuse_op.reuse_input=0;
        auto reuse_warm=consumer.compute_reuse(reuse_op,std::move(add_warm),{plain});consumer.synchronize(reuse_warm);
        producer.drain();consumer.drain();
        require(consumer.waterline_measurements().devices[0].consumer_wait_calls==0,"ready warmup counted as dependency wait");
        int least=0,greatest=0;poseidon::gpu::gpu_check_cuda(cudaDeviceGetStreamPriorityRange(&least,&greatest),"priority range");
        cudaStream_t low=nullptr;poseidon::gpu::gpu_check_cuda(cudaStreamCreateWithPriority(&low,cudaStreamNonBlocking,least),"producer low stream");
        CollectLoop collector(producer,consumer);
        const auto check_value=[&](PoseidonGpuValue &value,const std::vector<double> &expected) {
            consumer.synchronize(value);poseidon::Ciphertext host;
            poseidon::gpu::GpuUploader::download_ciphertext(value.device_ciphertext(),host,context);
            poseidon::Plaintext decrypted;decryptor.decrypt(host,decrypted);std::vector<std::complex<double>> actual;encoder.decode(decrypted,actual);
            require(actual.size()==expected.size(),"incorrect decoded slot count");
            double error=0;for(std::size_t index=0;index<actual.size();++index) {
                require(std::isfinite(actual[index].real()) && std::isfinite(actual[index].imag()),"nonfinite decoded slot");
                error=std::max(error,std::abs(actual[index].real()-expected[index]));error=std::max(error,std::abs(actual[index].imag()));
            }
            require(error<1e-4,"delayed consumer numerical error");return error;
        };
        const auto run=[&](const char *name,const fhegpu::ComputeOp &operation,const PoseidonGpuValue &input,
                            std::vector<double> expected,bool reuse,bool same_stream_followup) {
            const auto before=consumer.waterline_measurements().devices[0];
            Gate gate;ReleaseGate release(low,gate);
            PoseidonGpuValue delayed=input;
            {
                poseidon::gpu::ScopedGpuExecutionStream producer_stream(low);
                delayed=producer.compute(negate,{input});
            }
            const auto begin=Clock::now();
            auto result=reuse?consumer.compute_reuse(reuse_op,delayed,{plain}):
                consumer.compute(operation,operation.kind==fhegpu::ComputeKind::AddCP?
                                 std::vector<PoseidonGpuValue>{delayed,plain}:std::vector<PoseidonGpuValue>{delayed});
            if(same_stream_followup) {
                // This Ready is still pending, but its producer and consumer
                // are already ordered on the same actual foreground stream.
                result=consumer.compute(negate,{result});
                for(auto &value:expected)value=-value;
            }
            const double submit_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();release.submitted();
            const double error=check_value(result,expected);release.finish();
            require(!gate.watchdog && submit_ms<40,"consumer submission blocked for producer readiness");
            const auto after=consumer.waterline_measurements().devices[0];
            const auto wait_ns=after.consumer_wait_nanoseconds-before.consumer_wait_nanoseconds;
            require(after.consumer_wait_calls==before.consumer_wait_calls+1,"incorrect dependency interval count");
            require(wait_ns>60000000 && wait_ns<200000000,"GPU dependency interval differs from injected 80ms delay");
            report[name]={{"submit_ms",submit_ms},{"consumer_wait_nanoseconds",wait_ns},
                          {"max_abs_including_imaginary",error},{"slots_checked",expected.size()}};
            std::cout<<"PASS "<<name<<" submit_ms="<<submit_ms<<" wait_ns="<<wait_ns<<" error="<<error<<'\n'<<std::flush;
            return delayed;
        };
        auto negated=original;for(auto &value:negated)value=-value;
        auto delayed=run("Negate_with_same_stream_followup",negate,source,original,false,true);
        const auto before_ready=consumer.waterline_measurements().devices[0];
        auto ready_result=consumer.compute(negate,{delayed});check_value(ready_result,original);
        require(consumer.waterline_measurements().devices[0].consumer_wait_calls==before_ready.consumer_wait_calls,"already-ready input counted again");
        report["already_ready_input_not_counted"]=true;report["same_actual_stream_skip_preserved"]=true;
        auto add_expected=negated;for(std::size_t i=0;i<addend.size();++i)add_expected[i]+=addend[i];
        run("AddCP",add,source,add_expected,false,false);
        run("AddCP_reuse",reuse_op,source,add_expected,true,false);
        auto rotate_expected=negated;std::rotate(rotate_expected.begin(),rotate_expected.begin()+1,rotate_expected.end());
        run("Rotate",rotate,source,rotate_expected,false,false);
        auto square_expected=original;for(auto &value:square_expected)value=-value*value;
        run("Relinearize",relinearize,product,square_expected,false,false);
        // Rescale stores the declared power-of-two scale. Account for the
        // actual dropped prime when comparing this standalone operation.
        auto rescale_expected=square_expected;
        const double factor=std::ldexp(1.0,30)/context.parameters_literal()->q().back().value();
        for(auto &value:rescale_expected)value*=factor;
        run("Rescale",rescale,squared,rescale_expected,false,false);
        collector.finish();producer.drain();consumer.drain();
        const auto measurement=consumer.waterline_measurements().devices[0];
        require(measurement.consumer_wait_calls==6,"unexpected completed measurement count");
        poseidon::gpu::gpu_check_cuda(cudaStreamDestroy(low),"producer stream destroy");
        report["passed"]=true;report["consumer_wait_calls"]=measurement.consumer_wait_calls;
        report["consumer_wait_nanoseconds"]=measurement.consumer_wait_nanoseconds;
        report["delay_semantics"]="80ms host gate in a CUDA low stream followed by actual GPU producer kernels";
        report["measurement_scope"]="CUDA foreground dependency waiting; CPU scheduling delay is excluded";
    } catch(const std::exception &error) {report["error"]=error.what();std::cerr<<"FAIL "<<error.what()<<'\n';}
    std::ofstream output(argv[1]);output<<report.dump(2)<<'\n';
    if(!output)return 1;
    return report["passed"].get<bool>()?0:1;
}
