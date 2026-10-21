// Reuse the repository's existing native-bootstrap fixture and its tolerance.
#define main original_native_bootstrap_e2e_main
#include "poseidon/tests/runtime_api/poseidon_gpu_bootstrap_e2e.cpp"
#undef main
#include "poseidon/gpu/gpu_memory.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <numeric>

using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
double elapsed_ms(Clock::time_point begin) {
    return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
}
void save_report(const std::string &path,const Json &report) {
    std::ofstream out(path);out.exceptions(std::ios::badbit|std::ios::failbit);
    out<<report.dump(2)<<'\n';
}
int main(int argc,char **argv) {
    if(argc!=5) {std::cerr<<"usage: native-boot-bench REPORT DEGREE runtime-mixed|qwen30 ITERATIONS\n";return 2;}
    const std::string report_path=argv[1];
    Json report={{"passed",false},{"stage","setup"},{"samples",Json::array()},
        {"scope","Native GPU Boot API with input/output resident on device. Per-call synchronization, no nsys. Setup, explicit transfers and validation excluded from call timings."},
        {"tolerance",kTolerance},{"secret_key_used_for_boot_execution",false}};
    const auto process_begin=Clock::now();
    try {
        const auto degree=std::stoull(argv[2]);const std::string mode=argv[3];
        const auto iterations=std::stoull(argv[4]);
        if(!iterations)throw std::runtime_error("iterations must be positive");
        report["degree"]=degree;report["context_mode"]=mode;
        const auto context_begin=Clock::now();
        auto context=[&] {
            if(mode=="runtime-mixed")return make_context(degree);
            if(mode!="qwen30")throw std::runtime_error("unknown context mode");
            const int log_n=log2_degree(degree);
            poseidon::ParametersLiteral p(CKKS,log_n,log_n-1,kLogScale,0,kQ0Level,
                poseidon::Modulus(0),{},{},poseidon::sec_level_type::none);
            p.set_log_modulus(std::vector<std::uint32_t>(50,30),std::vector<std::uint32_t>(8,30));
            return poseidon::PoseidonContext(p);
        }();
        report["context_ms"]=elapsed_ms(context_begin);
        report["security_level"]="none; numerical validation context";
        for(auto q:context.parameters_literal()->q())report["q_bits"].push_back(q.bit_count());
        for(auto p:context.parameters_literal()->p())report["p_bits"].push_back(p.bit_count());
        const auto keys_begin=Clock::now();
        poseidon::KeyGenerator generator(context);poseidon::PublicKey public_key;
        generator.create_public_key(public_key);
        poseidon::Encryptor encryptor(context,public_key);
        poseidon::Decryptor decryptor(context,generator.secret_key());
        poseidon::CKKSEncoder encoder(context);
        report["base_keys_ms"]=elapsed_ms(keys_begin);
        PoseidonGpuApi api(kContextId,context,0);
        poseidon::gpu::GpuBootstrapProfileConfig config;
        const auto setup_begin=Clock::now();
        std::cout<<"Building native profile degree="<<degree<<" context="<<mode<<std::endl;
        auto profile=poseidon::gpu::GpuBootstrapProfileBuilder::build(context,generator,0,config);
        report["native_profile_setup_ms"]=elapsed_ms(setup_begin);
        report["profile_id"]=profile.profile_id;
        report["output_level"]=profile.output_level;report["output_scale_log2"]=profile.output_scale_log2;
        const auto spec=make_operator_spec(context,poseidon::runtime_api::make_native_boot_profile(profile));
        const auto plan=make_plan(spec,spec.spec.boot_profiles.front());
        api.configure_native_bootstrap(0,std::move(profile));
        const auto message=make_message(degree/2);
        poseidon::Plaintext plain;encoder.encode(message,std::ldexp(1.0,kLogScale),plain);
        poseidon::Ciphertext cipher;encryptor.encrypt(plain,cipher);
        auto upload=api.communicate_async(std::get<fhegpu::CommAction>(plan.plan.initialization.front().body),
            {PoseidonGpuValue::from_host_ciphertext(std::move(cipher))},{plan.plan.values.at(1)});
        auto uploaded=api.wait(upload);auto input=std::move(uploaded.at(0).value());api.synchronize(input);
        const auto &boot=std::get<fhegpu::ComputeOp>(plan.plan.execution.front().body);
        const auto &download=std::get<fhegpu::CommAction>(plan.plan.finalization.front().body);
        cudaEvent_t first=nullptr,last=nullptr;
        poseidon::gpu::gpu_check_cuda(cudaEventCreate(&first),"benchmark start event");
        poseidon::gpu::gpu_check_cuda(cudaEventCreate(&last),"benchmark end event");
        bool all_passed=true;
        // First call is cold; two additional warmups, then requested samples.
        for(std::size_t i=0;i<iterations+3;++i) {
            report["stage"]="native_boot";save_report(report_path,report);
            const auto begin=Clock::now();
            poseidon::gpu::gpu_check_cuda(cudaEventRecord(first,poseidon::gpu::gpu_execution_stream()),"benchmark start record");
            auto output=api.compute(boot,{input});
            poseidon::gpu::gpu_check_cuda(cudaEventRecord(last,poseidon::gpu::gpu_execution_stream()),"benchmark end record");
            poseidon::gpu::gpu_check_cuda(cudaEventSynchronize(last),"benchmark GPU completion");
            api.synchronize(output);
            const auto wall=elapsed_ms(begin);float gpu_span=0;
            poseidon::gpu::gpu_check_cuda(cudaEventElapsedTime(&gpu_span,first,last),"benchmark elapsed event");
            auto handle=api.communicate_async(download,{output},{plan.plan.values.at(3)});
            auto downloaded=api.wait(handle);
            poseidon::Plaintext decoded_plain;decryptor.decrypt(downloaded.at(0)->host_ciphertext(),decoded_plain);
            std::vector<std::complex<double>> decoded;encoder.decode(decoded_plain,decoded);
            if(decoded.size()!=message.size())throw std::runtime_error("decoded slot count mismatch");
            double error=0,rms=0;std::size_t nonfinite=0;
            for(std::size_t j=0;j<message.size();++j) {
                if(!std::isfinite(decoded[j].real()) || !std::isfinite(decoded[j].imag())) {++nonfinite;continue;}
                const auto delta=std::abs(decoded[j]-message[j]);error=std::max(error,delta);rms+=delta*delta;
            }
            const bool passed=!nonfinite && error<=kTolerance;all_passed&=passed;
            const std::string kind=i==0?"cold":i<3?"warmup":"measured";
            report["samples"].push_back({{"index",i},{"kind",kind},{"wall_ms",wall},
                {"gpu_stream_span_ms",gpu_span},{"max_complex_abs_error",error},
                {"rms_complex_error",std::sqrt(rms/message.size())},{"nonfinite",nonfinite},{"passed",passed}});
            std::cout<<kind<<" index="<<i<<" wall_ms="<<wall<<" gpu_span_ms="<<gpu_span<<" max_error="<<error<<" passed="<<passed<<std::endl;
            report["stage"]=passed?"sample_complete":"numerical_failure";save_report(report_path,report);
            if(!passed)break; // Preserve the failure; do not time an invalid path as successful Boot.
        }
        poseidon::gpu::gpu_check_cuda(cudaEventDestroy(first),"benchmark start destroy");
        poseidon::gpu::gpu_check_cuda(cudaEventDestroy(last),"benchmark end destroy");
        api.drain();report["passed"]=all_passed;
        report["process_ms_before_teardown"]=elapsed_ms(process_begin);
        report["stage"]=all_passed?"complete":"numerical_failure";
        save_report(report_path,report);return all_passed?0:1;
    } catch(const std::exception &e) {
        report["error"]=e.what();report["process_ms_before_teardown"]=elapsed_ms(process_begin);
        save_report(report_path,report);std::cerr<<"Native benchmark failed: "<<e.what()<<std::endl;return 1;
    }
}
