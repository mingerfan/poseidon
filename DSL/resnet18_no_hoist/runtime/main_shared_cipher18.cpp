#include "compiled_replay.cpp"
#include "gpu_resnet18_inference.h"
#include "gpu_activity_timing.h"

namespace poseidon::benchmark::resnet18_gpu {
GpuResNet18Result run_gpu_resnet18_impl(std::size_t,const ResNet18Topology &,const ResNet18Weights &,
    std::size_t,poseidon::benchmark::resnet50_gpu::GpuCkksRuntime &,bool,bool,bool);
}
using namespace poseidon::benchmark::resnet50_gpu;
using namespace poseidon::benchmark::resnet18_gpu;
static double max_difference(const std::vector<double> &a,const std::vector<double> &b){
    require(a.size()==b.size(),"paired logit lengths");double error=0;
    for(std::size_t i=0;i<a.size();++i)error=std::max(error,std::abs(a[i]-b[i]));return error;
}
int main(int argc,char **argv){try{
    std::cout<<std::unitbuf;
    const bool probe=argc==3 && std::string(argv[1])=="--probe-only";
    require(probe || argc==6,"usage: shared_cipher18 --probe-only PREFIX | HEVM CONSTANTS INPUTS PAYLOAD PREFIX");
    const std::string prefix=argv[probe?2:5];
    for(auto suffix:{".native.json",".dsl.json",".paired.json",".native-rns.bin"})require(!std::ifstream(prefix+suffix).good(),"paired artifacts already exist");
    for(auto key:{"POSEIDON_RESCALE_META_BITS","POSEIDON_RESCALE_MODEL_BITS","POSEIDON_DSL_SCALE_SHIFT","POSEIDON_GPU_DNUM"})require(!std::getenv(key),std::string("must unset: ")+key);
    const char *mode=std::getenv("POSEIDON_GPU_LINEAR_TRANSFORM_MODE");require(mode && std::string(mode)=="no_hoist","paired no_hoist required");
    auto config=make_resnet18_gpu_config();GpuCkksRuntime runtime(config);
    runtime.initialize_inference_evaluation_keys();runtime.initialize_bootstrap();
    runtime.dsl_begin_same_cipher_capture(prefix,probe?1:147,probe?1:1000);
    if(probe){
        std::vector<double> input(32768),mask(32768);
        for(std::size_t i=0;i<input.size();++i){input[i]=0.01*std::sin(double(i)*0.001);mask[i]=0.5+0.1*std::cos(double(i)*0.0003);}
        auto a=runtime.encrypt(input);auto one=runtime.encrypt_constant(1.0);const double scale=runtime.last_modulus_value(a);
        auto product=runtime.multiply_plain(a,mask,scale);auto reduced=runtime.rescale(product,1);
        auto low=runtime.drop_to_q_count(reduced,6);auto boot=runtime.bootstrap(low);auto final=runtime.drop_to_q_count(boot,4);
        auto native=runtime.decrypt(final);
        runtime.dsl_begin_same_cipher_replay(prefix);auto b=runtime.encrypt(input);auto same_one=runtime.encrypt_constant(1.0);
        runtime.dsl_require_equal(a,b);runtime.dsl_require_equal(one,same_one);
        auto plain=runtime.dsl_encode(32,mask,scale);auto replay_product=runtime.multiply_plain_preencoded(b,plain);
        auto replay_reduced=runtime.dsl_unary(3,replay_product,0);auto replay_low=runtime.dsl_unary(4,replay_reduced,25);
        auto replay_boot=runtime.dsl_unary(10,replay_low,30);auto replay_final=runtime.dsl_unary(4,replay_boot,27);
        auto dsl=runtime.decrypt(replay_final);auto report=Json::parse(runtime.dsl_finish_same_cipher());
        double error=0;for(std::size_t i=0;i<dsl.size();++i)error=std::max(error,std::abs(dsl[i]-native[i]));
        report["scope"]="controlled small probe: same input/ONE, scalar/vector native calls and original bootstrap; not full-network evidence";
        report["max_decoded_complex_error"]=error;report["status"]=report["bit_exact_final_ciphertexts"].get<bool>() && error==0?"PASS":"FAIL";
        std::ofstream out(prefix+".paired.json");out<<report.dump(2)<<'\n';require(bool(out),"probe report write");
        std::cout<<"SHARED_CIPHER_PROBE "<<report.dump()<<'\n';return report["status"]=="PASS"?0:1;
    }
    const auto topology=make_resnet18_topology();const auto weights=load_resnet18_weights();
    std::cout<<"SHARED_CIPHER_NATIVE_FULL_BEGIN context/keygen=once input_encryptions=captured_host_ciphertexts\n";
    const auto begin=std::chrono::steady_clock::now();
    const auto native=run_gpu_resnet18_impl(0,topology,weights,topology.blocks.size(),runtime,true,false,false);
    require(native.logits.size()==1000 && native.completed_blocks==8,"paired native full");
    Json native_report={{"status","COMPLETE"},{"prediction",native.predicted_label},{"label",native.true_label},{"logits",native.logits},
        {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()}};
    {std::ofstream out(prefix+".native.json");out<<native_report.dump(2)<<'\n';require(bool(out),"native report write");}
    runtime.dsl_begin_same_cipher_replay(prefix);
    std::cout<<"SHARED_CIPHER_DSL_FULL_BEGIN same_context_keys_and_input_ciphertexts=true\n";
    std::vector<std::string> args={"shared_compiled_dsl","18",argv[1],argv[2],argv[3],argv[4],prefix+".dsl.json","0"};
    std::vector<char*> pointers;for(auto &s:args)pointers.push_back(s.data());
    require(run_compiled_dsl_shared(pointers.size(),pointers.data(),runtime)==0,"paired compiled DSL failed");
    std::ifstream dsfile(prefix+".dsl.json");Json ds;dsfile>>ds;require(bool(dsfile),"read paired DSL report");
    auto report=Json::parse(runtime.dsl_finish_same_cipher());
    auto values=ds.at("logits").get<std::vector<double>>();report["max_logit_error"]=max_difference(values,native.logits);
    double sum=0;for(std::size_t i=0;i<values.size();++i){auto d=values[i]-native.logits[i];sum+=d*d;}
    report["rms_logit_error"]=std::sqrt(sum/values.size());report["native_prediction"]=native.predicted_label;report["dsl_prediction"]=ds.at("prediction");
    report["scope"]="full-network same-context/key/input-ciphertext control; all 1000 final ciphertexts compared word-for-word before decrypt";
    report["status"]=report.at("bit_exact_final_ciphertexts").get<bool>() && report["max_logit_error"]==0?"PASS":"FAIL";
    {std::ofstream out(prefix+".paired.json");out<<report.dump(2)<<'\n';require(bool(out),"paired report write");}
    std::cout<<"SHARED_CIPHER_FULL "<<report.dump()<<'\n';return report["status"]=="PASS"?0:1;
}catch(const std::exception &e){std::cerr<<"SHARED_CIPHER_ERROR "<<e.what()<<'\n';return 1;}}
