#include "gpu_ckks_runtime.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc,char **argv){
    using namespace poseidon::benchmark::resnet50_gpu;
    if(argc!=2 || (std::string(argv[1])!="--metadata-only" && std::string(argv[1])!="--run"))
        throw std::runtime_error("usage: native_nohoist_bootstrap_probe --metadata-only|--run");
    const auto config=make_resnet50_gpu_config();
    if(config.degree()!=65536 || config.q_moduli.size()!=50 || config.p_moduli.size()!=25)
        throw std::runtime_error("unexpected native CKKS configuration");
    std::cout<<std::unitbuf<<std::setprecision(17);
    if(std::string(argv[1])=="--metadata-only"){
        std::cout<<"NATIVE_NOHOIST_PROBE metadata=PASS N=65536 Q=50 P=25 input_q=32 application_q=36 gpu_executed=false\n";
        return 0;
    }
    GpuCkksRuntime runtime(config);
    std::vector<double> input(runtime.slot_count());
    for(std::size_t i=0;i<input.size();++i)input[i]=0.125*std::cos(double(i%101)*0.0625);
    auto encrypted=runtime.encrypt(input);
    auto application=runtime.drop_to_q_count(encrypted,6);
    const auto before=runtime.decrypt(application);
    runtime.initialize_bootstrap();
    auto result=runtime.bootstrap(application);
    const auto after=runtime.decrypt(result);
    double encoding=0,error=0,imaginary=0;
    for(std::size_t i=0;i<input.size();++i){
        if(!std::isfinite(after[i].real()) || !std::isfinite(after[i].imag()))
            throw std::runtime_error("nonfinite native no-hoist bootstrap result");
        encoding=std::max(encoding,std::abs(before[i].real()-input[i]));
        error=std::max(error,std::abs(after[i].real()-before[i].real()));
        imaginary=std::max(imaginary,std::abs(after[i].imag()));
    }
    if(result.meta.q_count!=31 || std::abs(std::log2(result.meta.scale)-40)>1e-9)
        throw std::runtime_error("native no-hoist bootstrap output metadata mismatch");
    std::cout<<"NATIVE_NOHOIST_BOOTSTRAP_COMPLETE q_in=6 q_out=31 log_scale=40"
        <<" encoding_max_error="<<encoding<<" bootstrap_max_error="<<error<<" max_imag="<<imaginary
        <<" gpu_executed=true scope=single_bootstrap_not_full_network_or_DSL\n";
}
