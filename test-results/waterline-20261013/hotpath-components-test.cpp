// Small concurrent diagnostic, only public APIs. No model input or full graph.
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_memory.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/communication/cuda_local_transfer.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using Api=poseidon::runtime_api::PoseidonGpuApi;
using Value=poseidon::runtime_api::PoseidonGpuValue;
using Pinned=poseidon::runtime_api::communication::PinnedHostBuffer;
using Clock=std::chrono::steady_clock;
using Json=nlohmann::json;
constexpr int Samples=10;
void check(cudaError_t value,const char *reason) {poseidon::gpu::gpu_check_cuda(value,reason);}
void require(bool value,const char *reason) {if(!value)throw std::runtime_error(reason);}
double elapsed(Clock::time_point begin) {return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
std::string timestamp() {
    auto now=std::time(nullptr);std::tm value{};gmtime_r(&now,&value);
    char text[32];std::strftime(text,sizeof(text),"%Y-%m-%dT%H:%M:%SZ",&value);return text;
}
Json statistics(const std::vector<double> &values) {
    auto sorted=values;std::sort(sorted.begin(),sorted.end());double total=0;
    for(auto value:sorted)total+=value;
    return {{"samples",values},{"mean",total/sorted.size()},{"median",(sorted[4]+sorted[5])/2},
            {"min",sorted.front()},{"max",sorted.back()},{"unit","ms"}};
}
struct Events {
    cudaEvent_t begin=nullptr,end=nullptr;
    Events() {check(cudaEventCreate(&begin),"start event");check(cudaEventCreate(&end),"end event");}
    ~Events() {if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);}
    double milliseconds() {float result=0;check(cudaEventElapsedTime(&result,begin,end),"elapsed event");return result;}
};
struct StreamLifetime {
    cudaStream_t value=nullptr;
    ~StreamLifetime() {if(value)(void)cudaStreamDestroy(value);}
};
void CUDART_CB own_host_delay(void *) {std::this_thread::sleep_for(std::chrono::milliseconds(50));}

int main(int argc,char **argv) {
    const bool enable_cache=argc==2 && std::string(argv[1])=="--cache";
    if(argc!=1 && !enable_cache) {std::cerr<<"usage: hotpath-components-test [--cache]\n";return 2;}
    Json report{{"passed",false},{"scope","public synthetic hot-path decomposition; concurrent workloads recorded separately; not end-to-end model performance"},
                {"started_utc",timestamp()},{"samples_per_component",Samples},{"N",16384},{"Q",40},{"P",8},{"scale_log2",120},
                {"pool_maximum_mib",1536},{"pool_initial_mib",64},{"gpu_selection","the one explicitly selected visible GPU, logical GPU0"}};
    report["CUDA_VISIBLE_DEVICES"]=std::getenv("CUDA_VISIBLE_DEVICES")?std::getenv("CUDA_VISIBLE_DEVICES"):"unset";
    report["timing_semantics"]="GPU CUDA event intervals overlap CPU submission/Ready wait; include queue interference and launch gaps, not isolated active-kernel time. Do not add event and submission durations. Ordinary AddCP/Rotate use public API on caller low stream without Waterline foreground override.";
    report["diagnostic_pid"]=getpid();report["CKKS_security_level"]="none; parameter-matched numerical diagnostic";
    report["waterline_raw_pinned_cache"]=enable_cache;
    try {
        setenv("POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB","1536",1);
        setenv("POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB","64",1);
        int visible=0;check(cudaGetDeviceCount(&visible),"visible devices");require(visible==1,"diagnostic must expose exactly one explicitly selected GPU");
        check(cudaSetDevice(0),"select diagnostic GPU");std::size_t free=0,total=0;
        check(cudaMemGetInfo(&free,&total),"free memory before diagnostic");require(free>=(2ULL<<30),"selected GPU must have at least 2GiB free before diagnostics");
        report["free_before_bytes"]=free;report["total_bytes"]=total;
        cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,0),"device properties");report["device_name"]=properties.name;
        int least=0,greatest=0;check(cudaDeviceGetStreamPriorityRange(&least,&greatest),"stream priorities");
        StreamLifetime stream_owner;
        check(cudaStreamCreateWithPriority(&stream_owner.value,cudaStreamNonBlocking,least),"low diagnostic stream");
        const auto stream=stream_owner.value;
        report["ordinary_gpu_stream_priority"]=least;
        poseidon::gpu::ScopedGpuExecutionStream caller_stream(stream);
        const auto cold_begin=Clock::now();
        poseidon::ParametersLiteral parameters(CKKS,14,13,120,0,0,poseidon::Modulus(0),{},{},poseidon::sec_level_type::none);
        parameters.set_log_modulus(std::vector<std::uint32_t>(40,30),std::vector<std::uint32_t>(8,30));
        poseidon::PoseidonContext context(parameters);poseidon::KeyGenerator generator(context);
        auto public_key=std::make_shared<poseidon::PublicKey>();generator.create_public_key(*public_key);
        auto secret_key=std::make_shared<poseidon::SecretKey>(generator.secret_key());
        auto galois=std::make_shared<poseidon::GaloisKeys>();generator.create_galois_keys(std::vector<int>{1},*galois);
        poseidon::CKKSEncoder encoder(context);poseidon::Encryptor encryptor(context,*public_key);
        poseidon::Decryptor decryptor(context,*secret_key);
        Api api("hotpath-profile",context,0,{},galois,public_key,secret_key);
        const fhegpu::Place device{fhegpu::PlaceKind::Device,0,0},host{fhegpu::PlaceKind::Host,0,0};
        fhegpu::OperatorSpec spec;spec.id="hotpath-profile";spec.version=1;spec.status="diagnostic";
        spec.target_id="poseidon-ckks-gpu";spec.context_id="hotpath-profile";spec.poly_degree=16384;
        spec.rns_moduli_log2=std::vector<int>(40,30);spec.max_modulus_log2=30;spec.default_scale_log2=120;
        spec.level_upper_bound=39;spec.rescale_mode=fhegpu::RescaleMode::Lazy;
        for(auto kind:{fhegpu::ComputeKind::AddCP,fhegpu::ComputeKind::Rotate,fhegpu::ComputeKind::Boot,fhegpu::ComputeKind::Rescale})
            spec.operators[kind].supported=true;
        spec.operators[fhegpu::ComputeKind::Rescale].max_levels_per_op=4;
        fhegpu::BootProfile boot_profile;boot_profile.profile_id="ready-host-refresh";
        boot_profile.implementation=fhegpu::BootImplementation::DecryptReencrypt;
        boot_profile.input_level_min=0;boot_profile.input_level_max=39;boot_profile.input_components=2;
        boot_profile.output_level=39;boot_profile.output_scale_log2=120;boot_profile.output_components=2;
        boot_profile.needs_secret_key=true;boot_profile.needs_host_compute=true;spec.boot_profiles={boot_profile};
        const std::string digest="sha256:"+std::string(64,'1');
        fhegpu::TargetConfig target;target.target_id=spec.target_id;target.world_size=1;target.device_counts={1};
        target.capability_version=1;target.operator_spec={spec.id,1,digest};
        fhegpu::PlanRequirements requirements;requirements.capabilities={fhegpu::RequiredCapability::HostCompute,fhegpu::RequiredCapability::BootDecryptReencrypt};
        requirements.keys={{fhegpu::KeyKind::Galois,device,1,39},{fhegpu::KeyKind::Secret,host,std::nullopt}};
        api.preflight(digest,false,target,spec,requirements);
        std::vector<double> slots(8192);for(std::size_t i=0;i<slots.size();++i)slots[i]=.125*std::sin(i*.071)+.025*std::cos(i*.019);
        const auto high=context.crt_context()->parms_id_map().at(39),low=context.crt_context()->parms_id_map().at(7);
        const auto scale=std::ldexp(1.,120);
        fhegpu::ValueDesc desc{77,fhegpu::ValueKind::Plaintext,device,"hotpath-profile",39,120,true,1};
        const auto cpu_cost=api.prepare_resources(desc,fhegpu::PrepareBackend::Cpu);
        const auto gpu_cost=api.prepare_resources(desc,fhegpu::PrepareBackend::Gpu);
        fhegpu::ComputeOp add;add.kind=fhegpu::ComputeKind::AddCP;add.place=device;
        fhegpu::ComputeOp rotate;rotate.kind=fhegpu::ComputeKind::Rotate;rotate.place=device;rotate.attrs=fhegpu::RotateAttrs{1};
        report["resource_queries"]={{"cpu_prepare_host",cpu_cost.peak.host},{"cpu_prepare_pinned",cpu_cost.peak.pinned},
            {"gpu_prepare_plain_peak",gpu_cost.peak.plain[0]},{"gpu_prepare_pinned",gpu_cost.peak.pinned},
            {"rotate_workspace_bound",api.instruction_workspace({0,rotate}).front[0]}};
        auto setup=api.waterline_measurements();report["persistent_allocated_bytes"]=setup.devices[0].allocated_live;
        Json primes=Json::array();for(auto value:context.parameters_literal()->q())primes.push_back(value.value());report["actual_Q_primes"]=primes;
        primes=Json::array();for(auto value:context.parameters_literal()->p())primes.push_back(value.value());report["actual_P_primes"]=primes;
        poseidon::Plaintext full_plain,low_plain;encoder.encode(slots,high,scale,full_plain);encoder.encode(slots,low,scale,low_plain);
        poseidon::Ciphertext high_cipher,low_cipher;encryptor.encrypt(full_plain,high_cipher);encryptor.encrypt(low_plain,low_cipher);
        const auto verify_plain=[&](const poseidon::Plaintext &plain,const std::vector<double> &expected) {
            std::vector<std::complex<double>> actual;encoder.decode(plain,actual);require(actual.size()==expected.size(),"diagnostic slot count changed");
            double maximum=0;for(std::size_t i=0;i<actual.size();++i) {
                require(std::isfinite(actual[i].real()) && std::isfinite(actual[i].imag()),"nonfinite diagnostic output");
                maximum=std::max({maximum,std::abs(actual[i].real()-expected[i]),std::abs(actual[i].imag())});
            }
            require(maximum<1e-8,"diagnostic numerical mismatch");return maximum;
        };
        auto device_cipher=Value::from_device_ciphertext(poseidon::gpu::GpuUploader::upload_ciphertext(high_cipher,0));
        auto device_plain=Value::from_device_plaintext(poseidon::gpu::GpuUploader::upload_plaintext(full_plain,0));
        check(cudaStreamSynchronize(stream),"ready diagnostic inputs");
        auto ready_host=Value::from_host_ciphertext(low_cipher);require(api.is_ready(ready_host),"Host Boot input must already be Ready");
        fhegpu::ComputeOp boot;boot.kind=fhegpu::ComputeKind::Boot;boot.place=host;
        boot.attrs=fhegpu::BootAttrs{39,120,2,"ready-host-refresh",fhegpu::BootImplementation::DecryptReencrypt};
        {auto value=api.compute(boot,{ready_host});}
        {auto value=api.compute(add,{device_cipher,device_plain});api.synchronize(value);}
        {auto value=api.compute(rotate,{device_cipher});api.synchronize(value);}
        {auto value=api.prepare_plaintext_async(desc,slots,fhegpu::PrepareBackend::Gpu);api.synchronize(value);}
        api.drain();report["cold_setup_ms"]=elapsed(cold_begin);
        std::cout<<"HOTPATH_READY utc="<<timestamp()<<" persistent="<<report["persistent_allocated_bytes"]
                 <<" rotate_bound="<<report["resource_queries"]["rotate_workspace_bound"]<<'\n'<<std::flush;

        std::vector<double> cpu_encode,boot_api,decrypt_ms,decode_ms,encode_ms,encrypt_ms;
        for(int i=0;i<Samples;++i) {
            poseidon::Plaintext value;auto begin=Clock::now();encoder.encode(slots,high,scale,value);cpu_encode.push_back(elapsed(begin));
        }
        report["CPU_complete_encode_fullQ_wall"]=statistics(cpu_encode);
        std::optional<Value> last_boot;
        for(int i=0;i<Samples;++i) {
            const auto begin=Clock::now();auto result=api.compute(boot,{ready_host});boot_api.push_back(elapsed(begin));
            last_boot=std::move(result);
        }
        report["Host_secret_refresh_ready_input_API_wall"]=statistics(boot_api);
        {poseidon::Plaintext value;decryptor.decrypt(last_boot->host_ciphertext(),value);report["Host_secret_refresh_sanity_max_error"]=verify_plain(value,slots);}
        last_boot.reset();
        for(int i=0;i<Samples;++i) {
            poseidon::Plaintext decoded,reencoded;poseidon::Ciphertext encrypted;std::vector<std::complex<double>> complex_slots;
            auto begin=Clock::now();decryptor.decrypt(low_cipher,decoded);decrypt_ms.push_back(elapsed(begin));
            begin=Clock::now();encoder.decode(decoded,complex_slots);decode_ms.push_back(elapsed(begin));
            begin=Clock::now();encoder.encode(complex_slots,high,scale,reencoded);encode_ms.push_back(elapsed(begin));
            begin=Clock::now();encryptor.encrypt(reencoded,encrypted);encrypt_ms.push_back(elapsed(begin));
        }
        report["Host_secret_refresh_CPU_stages"]={{"decrypt_Q8",statistics(decrypt_ms)},{"decode_Q8",statistics(decode_ms)},
            {"encode_Q40",statistics(encode_ms)},{"encrypt_Q40",statistics(encrypt_ms)}};

        Json pinned=Json::array();
        for(std::size_t bytes:{std::size_t(65536),std::size_t(16384*40*4)}) {
            {auto warm=std::make_unique<Pinned>(bytes);}
            std::vector<double> allocate,release;
            for(int i=0;i<Samples;++i) {
                auto begin=Clock::now();auto value=std::make_unique<Pinned>(bytes);allocate.push_back(elapsed(begin));
                begin=Clock::now();value.reset();release.push_back(elapsed(begin));
            }
            pinned.push_back({{"bytes",bytes},{"cudaMallocHost_wall",statistics(allocate)},{"cudaFreeHost_wall",statistics(release)}});
        }
        report["PinnedHostBuffer_idle_own_context"]=pinned;
        // One bounded own-context probe of allocation with pending work. It
        // does not inspect or synchronize the concurrently running model.
        check(cudaLaunchHostFunc(stream,own_host_delay,nullptr),"own allocation delay");
        auto begin=Clock::now();auto pending_pin=std::make_unique<Pinned>(65536);
        report["PinnedHostBuffer_with_own_50ms_pending_callback_malloc_ms"]=elapsed(begin);
        check(cudaStreamSynchronize(stream),"finish own allocation probe");pending_pin.reset();

        if(enable_cache) {
            // A resource configuration with an eligible GPU Prepare. The
            // same direct API measurements below select CPU/GPU explicitly;
            // this enables only the production raw-pinned cache contract.
            fhegpu::RuntimePlan plan;plan.format_version=4;plan.target=target;plan.values={desc};
            plan.execution={{0,fhegpu::PreparePlaintextOp{fhegpu::InlineEncodePayload{slots},desc.id}}};
            fhegpu::WaterlineConfig config;config.backend=fhegpu::PrepareBackend::Auto;
            config.device_capacity_bytes={2ULL<<30};config.device_reserve_bytes={1ULL<<30};config.device_fractions={.25};
            config.host_budget_bytes=128ULL<<20;config.pinned_budget_bytes=16ULL<<20;config.raw_budget_bytes=16ULL<<20;
            plan.waterline=config;api.validate_waterline_resources(plan);
            const auto fixed=api.waterline_fixed_resources();
            const auto dynamic=api.prepare_resources(desc,fhegpu::PrepareBackend::Gpu);
            report["cached_resources"]={{"fixed_host",fixed.host},{"fixed_pinned",fixed.pinned},
                {"dynamic_gpu_prepare_host",dynamic.peak.host},{"dynamic_gpu_prepare_pinned",dynamic.peak.pinned}};
        }
        for(auto backend:{fhegpu::PrepareBackend::Cpu,fhegpu::PrepareBackend::Gpu}) {
            {auto warm=api.prepare_plaintext_async(desc,slots,backend);api.synchronize(warm);api.drain();}
            std::vector<double> submit,synchronize,collect,encode_event,h2d_event,cpu_math;
            std::optional<Value> last;
            const auto allocations_before=Pinned::allocation_calls(),frees_before=Pinned::free_calls();
            for(int i=0;i<Samples;++i) {
                const auto before=api.waterline_measurements();auto begin=Clock::now();
                auto value=api.prepare_plaintext_async(desc,slots,backend);submit.push_back(elapsed(begin));
                begin=Clock::now();api.synchronize(value);synchronize.push_back(elapsed(begin));
                begin=Clock::now();api.collect_completed();collect.push_back(elapsed(begin));
                const auto after=api.waterline_measurements();
                encode_event.push_back((after.devices[0].gpu_encode_nanoseconds-before.devices[0].gpu_encode_nanoseconds)/1e6);
                h2d_event.push_back((backend==fhegpu::PrepareBackend::Gpu?
                    after.devices[0].gpu_raw_h2d_nanoseconds-before.devices[0].gpu_raw_h2d_nanoseconds:
                    after.transport_h2d_nanoseconds-before.transport_h2d_nanoseconds)/1e6);
                cpu_math.push_back((after.cpu_encode_nanoseconds-before.cpu_encode_nanoseconds)/1e6);
                last=std::move(value);
            }
            report[backend==fhegpu::PrepareBackend::Cpu?"CPU_Prepare_complete_API":"GPU_Prepare_complete_API"]={
                {"submit_wall",statistics(submit)},{"Ready_synchronize_wall",statistics(synchronize)},
                {"completed_reclamation_wall",statistics(collect)},{"GPU_math_event",statistics(encode_event)},
                {"H2D_event",statistics(h2d_event)},{"CPU_encoder_counter",statistics(cpu_math)},
                {"pinned_allocations",Pinned::allocation_calls()-allocations_before},
                {"pinned_frees_before_drain",Pinned::free_calls()-frees_before},
                {"pinned_live_before_drain",Pinned::live_bytes()}};
            poseidon::Plaintext actual;
            poseidon::gpu::GpuUploader::download_plaintext(last->device_plaintext(),actual,context);
            report[backend==fhegpu::PrepareBackend::Cpu?"CPU_Prepare_sanity_max_error":"GPU_Prepare_sanity_max_error"]=verify_plain(actual,slots);
        }
        // Enabling Waterline also enables its private foreground stream. An
        // external caller-stream event would then describe the wrong stream.
        // Ordinary operation events are measured only in the uncached run.
        if(enable_cache)report["ordinary_operator_events_omitted"]="Waterline foreground stream differs from caller stream; use the uncached ready-input operator decomposition";
        if(!enable_cache)for(const auto &operation:{add,rotate}) {
            const bool addition=operation.kind==fhegpu::ComputeKind::AddCP;
            std::vector<double> submit,event,wait;
            std::optional<Value> last;
            for(int i=0;i<Samples;++i) {
                Events timing;check(cudaEventRecord(timing.begin,stream),"operator event start");auto begin=Clock::now();
                auto output=api.compute(operation,addition?std::vector<Value>{device_cipher,device_plain}:std::vector<Value>{device_cipher});
                submit.push_back(elapsed(begin));check(cudaEventRecord(timing.end,stream),"operator event end");
                begin=Clock::now();api.synchronize(output);check(cudaEventSynchronize(timing.end),"operator event completion");wait.push_back(elapsed(begin));
                event.push_back(timing.milliseconds());
                last=std::move(output);
            }
            report[addition?"GPU_AddCP_ready_inputs":"GPU_Rotate_ready_input"]={
                {"submit_wall",statistics(submit)},{"CUDA_event",statistics(event)},{"Ready_wait_wall",statistics(wait)}};
            poseidon::Ciphertext actual;poseidon::gpu::GpuUploader::download_ciphertext(last->device_ciphertext(),actual,context);
            poseidon::Plaintext plain;decryptor.decrypt(actual,plain);auto expected=slots;
            if(addition)for(auto &value:expected)value*=2;
            else std::rotate(expected.begin(),expected.begin()+1,expected.end());
            report[addition?"GPU_AddCP_sanity_max_error":"GPU_Rotate_sanity_max_error"]=verify_plain(plain,expected);
        }
        api.drain();auto after=api.waterline_measurements();
        report["managed_pool"]={{"allocated_live",after.devices[0].allocated_live},{"allocated_peak",after.devices[0].allocated_peak},
            {"pool_reserved",after.devices[0].pool_reserved},{"pool_capacity",after.devices[0].pool_capacity}};
        report["pinned_live_end"]=Pinned::live_bytes();report["pinned_peak"]=Pinned::peak_bytes();
        require(Pinned::live_bytes()==0,"completed diagnostic still retains pinned staging");
        report["passed"]=true;
    } catch(const std::exception &error) {report["error"]=error.what();std::cerr<<"HOTPATH_FAIL "<<error.what()<<'\n';}
    report["finished_utc"]=timestamp();std::cout<<report.dump(2)<<'\n'<<std::flush;
    return report["passed"].get<bool>()?0:1;
}
