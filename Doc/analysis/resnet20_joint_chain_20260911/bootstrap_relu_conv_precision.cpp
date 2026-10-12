// One real layer, not a full-network or performance/security approval.
#define POSEIDON_BOOTSTRAP_RELU_LIBRARY
#include "bootstrap_relu_precision.cpp"
#include "original_conv_reference.h"
#include "conv_probe_runtime.h"

namespace conv_probe {
namespace app=poseidon::benchmark::resnet20_gpu::core;
using Runtime=app::DiagnosticConvRuntime;

auto run_conv(const Runtime &runtime,const Runtime::DeviceCiphertext &input,
    const ApplicationConvFixture &f) {
    auto tensor=app::make_shape(32,32,16,1,runtime.slot_count());
    if(tensor.packs.size()!=1 || tensor.pages_per_cipher!=32)throw std::runtime_error("unexpected input packing");
    for(int c=0;c<16;++c)for(int r=0;r<32;++r)for(int col=0;col<32;++col)
        if(tensor.slot_index(c,r,col)!=std::size_t(c*1024+r*32+col) || tensor.pack_index(c)!=0)
            throw std::runtime_error("CHW-to-SIMD packing mismatch");
    tensor.packs[0]=input;
    auto output=app::conv2d_bn(tensor,16,1,3,3,f.weights,f.bn_scale,f.bn_bias,runtime);
    if(output.packs.size()!=1 || output.h!=32 || output.w!=32 || output.c!=16 || output.k!=1)
        throw std::runtime_error("Conv output layout mismatch");
    auto result=std::move(output.packs[0]);
    if(result.meta.q_count!=6 || std::abs(std::log2(result.meta.scale)-40)>1e-8)
        throw std::runtime_error("Conv did not follow Q9 -> Q6 / scale40");
    return result;
}

Values cpu_conv(const BaselineMetadata &m,const Values &v,const ApplicationConvFixture &f) {
    Runtime cpu(m);return *run_conv(cpu,cpu.host(v),f).values;
}

Values pack(const std::vector<double> &v) {
    if(v.size()!=16384)throw std::runtime_error("CHW pack size mismatch");
    Values out(32768);std::copy(v.begin(),v.end(),out.begin());return out;
}

Values plain_relu(const std::vector<double> &v) {
    auto out=pack(v);for(auto &x:out)x=original_application_relu(double(x.real()));return out;
}

std::vector<double> active_real(const Values &v) {
    if(v.size()!=32768)throw std::runtime_error("slot count mismatch");
    std::vector<double> out(16384);for(std::size_t i=0;i<out.size();++i)out[i]=double(v[i].real());return out;
}

Runtime metadata(const BaselineMetadata &m,const ApplicationConvFixture &f) {
    Runtime cpu(m);const auto input=plain_relu(f.stem);
    auto result=run_conv(cpu,cpu.host(input),f);
    const double original_error=compare_values("original_SIMD_Conv_vs_original_CHW_Conv",*result.values,
        pack(original_application_conv(active_real(input),f)));
    // A non-image input catches accidental reliance on particular activations.
    Values edge(32768);
    for(int i=0;i<16384;++i)edge[i]=(i%13-6)*0.03125L;
    const double edge_error=compare_values("SIMD_Conv_layout_edge_reference",cpu_conv(m,edge,f),
        pack(original_application_conv(active_real(edge),f)));
    if(original_error>1e-10 || edge_error>1e-10)throw std::runtime_error("original Conv layout oracle failed");
    if(cpu.counts["rescale"]!=3 || cpu.rotations.size()!=2 || !cpu.rotations.count(9) || !cpu.rotations.count(7))
        throw std::runtime_error("unexpected Conv rescale/rotation levels");
    std::cout<<"CONV_PLAN layer=layer1_0.conv"<<f.conv_index<<"+BN image="<<f.image_id
        <<" shape=16x32x32 kernel=3x3 stride=1 active_slots=16384 total_slots=32768"
        <<" q_in=9 q_out=6 Q_consumed=3 log_scale=40 original_source=true prefix_stem=offline_plaintext\n";
    for(const auto &[q,steps]:cpu.rotations){std::cout<<"CONV_ROTATIONS q="<<q<<" steps=";
        bool first=true;for(int step:steps){if(!first)std::cout<<',';std::cout<<step;first=false;}std::cout<<'\n';}
    std::cout<<"CONV_COUNTS";for(const auto &[name,count]:cpu.counts)std::cout<<' '<<name<<'='<<count;std::cout<<'\n';
    bool encrypt_rejected=false,decrypt_rejected=false,raise_rejected=false,align_rejected=false;
    try{cpu.encrypt({});}catch(const std::runtime_error &){encrypt_rejected=true;}
    try{cpu.decrypt(result);}catch(const std::runtime_error &){decrypt_rejected=true;}
    try{cpu.drop_to_q_count(result,7);}catch(const std::runtime_error &){raise_rejected=true;}
    try{cpu.add(result,cpu.host(input));}catch(const std::runtime_error &){align_rejected=true;}
    if(!encrypt_rejected || !decrypt_rejected || !raise_rejected || !align_rejected)
        throw std::runtime_error("Conv negative checks failed");
    std::cout<<"CONV_NEGATIVE_CHECKS encryption=REJECTED decryption=REJECTED modraise=REJECTED misalignment=REJECTED\n";
    return cpu;
}

struct ConvContinuation {
    const ReluContinuation &relu;
    const CT &output;
    const Values &local,&chain,&original;
};

int execute(const ReluContinuation &r,const ApplicationConvFixture &f,const Runtime &planned,
    const std::function<int(const ConvContinuation &)> &continuation={}) {
    const auto &b=r.bootstrap;const auto &m=b.metadata;
    std::cout<<"PHASE real_application_ConvBN key_setup_after_bootstrap=true global_P=25 direct_keys=true\n";
    std::set<int> all;for(const auto &[q,steps]:planned.rotations)all.insert(steps.begin(),steps.end());
    auto galois=b.rotation_keys(std::vector<int>(all.begin(),all.end()));
    GpuUploader::prepare_key_views_for_q_counts(galois,{7,9});
    Runtime runtime(m,&b.eval,&galois);
    CT input;b.eval.drop_modulus(r.output,input,r.output.meta.parms_id);
    const auto actual_input=decoded_values(b.replay,r.output);
    auto result=run_conv(runtime,runtime.device(std::move(input)),f);
    if(runtime.rotations!=planned.rotations || runtime.counts!=planned.counts)
        throw std::runtime_error("GPU Conv graph differs from pre-key CPU trace");
    const auto local=cpu_conv(m,actual_input,f),chain=cpu_conv(m,r.chain,f);
    const auto original=pack(original_application_conv(active_real(r.original),f));
    const double local_error=b.replay.check("Conv_local_arithmetic_all_slots",*result.gpu,local);
    const double chain_error=b.replay.check("Conv_continuous_polynomial_all_slots",*result.gpu,chain);
    const double end_error=b.replay.check("Conv_original_application_reference_all_slots",*result.gpu,original);
    const double propagated=compare_values("Conv_propagated_input_arithmetic",local,chain);
    compare_values("Conv_bootstrap_approximation_propagation",chain,original);
    const auto decoded=decoded_values(b.replay,*result.gpu);
    double padding=0,imag=0;for(std::size_t i=0;i<decoded.size();++i){
        imag=std::max(imag,double(std::abs(decoded[i].imag())));
        if(i>=16384)padding=std::max(padding,double(std::abs(decoded[i])));
    }
    const bool pass=local_error<=1e-5 && chain_error<=propagated+1e-5 && padding<=1e-5;
    std::cout<<"CONV_RESULT integration="<<(pass?"PASS":"FAIL")<<" local_arithmetic_max="<<local_error
        <<" propagated_input_max="<<propagated<<" chain_arithmetic_max="<<chain_error
        <<" end_to_end_max="<<end_error<<" end_to_end_target_1e_4="<<(end_error<=1e-4?"PASS":"FAIL")
        <<" padding_max="<<padding<<" max_imag="<<imag<<" q=6 log_scale="<<std::log2(result.meta.scale)
        <<" same_ciphertext_chain=true no_reencryption=true full_network_tested=false"
        <<" production_per_level_P_tested=false timing_benchmark=false security_approved=false\n";
    if(pass && continuation) {
        // Public keys are host-cached. Do not carry this GPU upload into a
        // nested high-Q bootstrap; no later Conv operation uses runtime.keys.
        gpu_check_cuda(cudaDeviceSynchronize(),"release Conv rotation keys before continuation");
        galois=GpuGaloisKeysData{};
        return continuation(ConvContinuation{r,*result.gpu,local,chain,original});
    }
    return pass?0:1;
}
} // namespace conv_probe

#ifndef POSEIDON_BOOTSTRAP_CONV_LIBRARY
int main(int argc,char **argv) {
    try {
        if(argc!=4 || (std::string(argv[1])!="--metadata-only" && std::string(argv[1])!="--unsafe-accuracy-only"))
            throw std::runtime_error("usage: bootstrap_relu_conv_precision --metadata-only|--unsafe-accuracy-only fixture.txt image_id");
        std::size_t parsed=0;const int image=std::stoi(argv[3],&parsed);
        if(parsed!=std::string(argv[3]).size() || image<0 || image>=10000)throw std::runtime_error("invalid CIFAR image id");
        ReluFixture relu(argv[2]);const auto fixture=load_application_conv_fixture(image);
        const auto packed=conv_probe::pack(fixture.stem);
        std::vector<std::complex<double>> input;for(auto v:packed)input.emplace_back(v);
        double max_input=0;for(auto v:input)max_input=std::max(max_input,std::abs(v));
        std::cout<<"APPLICATION_INPUT image="<<image<<" stem_max_abs="<<max_input<<" boundary=40 stem_gpu_tested=false\n";
        std::unique_ptr<conv_probe::Runtime> planned;
        std::string profile="q50-fixed45",dataset="grid";
        char *args[]={argv[0],argv[1],profile.data(),dataset.data(),argv[2]};
        return run_bootstrap_baseline(5,args,
            [&](const BaselineMetadata &m){check_continuation_metadata(m,relu);
                planned=std::make_unique<conv_probe::Runtime>(conv_probe::metadata(m,fixture));},
            [&](const BaselineContinuation &b){return continue_to_relu(b,relu,
                [&](const ReluContinuation &r){return conv_probe::execute(r,fixture,*planned);});},&input);
    }catch(const std::exception &e){std::cerr<<"ERROR "<<e.what()<<'\n';return 2;}
}
#endif
