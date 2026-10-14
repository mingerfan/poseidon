// Complete first BasicBlock, with a diagnostic bootstrap before stem ReLU.
// One initial encryption only. No production/security/performance approval.
#define POSEIDON_BOOTSTRAP_CONV_LIBRARY
#include "bootstrap_relu_conv_precision.cpp"

namespace block_probe {
using namespace conv_probe;

auto residual(const Runtime &runtime,const Runtime::DeviceCiphertext &left,
    const Runtime::DeviceCiphertext &right) {
    auto a=app::make_shape(32,32,16,1,32768),b=app::make_shape(32,32,16,1,32768);
    a.packs[0]=left;b.packs[0]=right;
    auto result=app::residual_add(a,b,runtime);
    if(result.packs.size()!=1 || result.packs[0].meta.q_count!=6 ||
        std::abs(std::log2(result.packs[0].meta.scale)-40)>1e-8)
        throw std::runtime_error("residual did not preserve Q6 / scale40");
    return std::move(result.packs[0]);
}

Values add_values(Values left,const Values &right) {
    if(left.size()!=right.size())throw std::runtime_error("reference shape mismatch");
    for(std::size_t i=0;i<left.size();++i)left[i]+=right[i];return left;
}

Values application_relu(Values x) {
    for(auto &v:x) {
        if(v.imag()!=0)throw std::runtime_error("original application reference must remain real");
        v=original_application_relu(double(v.real()));
    }
    return x;
}

void check_metadata(const BaselineMetadata &m,const ApplicationConvFixture &f1,
    const ApplicationConvFixture &f2) {
    const auto shortcut=plain_relu(f1.stem);
    const auto conv1=cpu_conv(m,shortcut,f1);
    const auto conv2=cpu_conv(m,application_relu(conv1),f2);
    Runtime cpu(m);
    const auto sum=residual(cpu,cpu.host(conv2,6),cpu.host(shortcut,9));
    const auto result=application_relu(*sum.values);
    const auto independent=pack(original_application_first_block(f1,f2));
    if(compare_values("block_original_SIMD_vs_independent_CHW",result,independent)>1e-10)
        throw std::runtime_error("independent first-block reference mismatch");
    if(cpu.counts["rescale"]!=0 || cpu.counts["add"]!=1)
        throw std::runtime_error("residual must not rescale");
    bool scale_rejected=false,layout_rejected=false;
    try{residual(cpu,cpu.host(conv2,6),cpu.host(shortcut,9,std::exp2(41)));}
    catch(const std::runtime_error &){scale_rejected=true;}
    auto a=app::make_shape(32,32,16,1,32768),b=app::make_shape(16,16,16,1,32768);
    a.packs[0]=cpu.host(conv2,6);b.packs[0]=cpu.host(shortcut,9);
    try{app::residual_add(a,b,cpu);}catch(const std::invalid_argument &){layout_rejected=true;}
    if(!scale_rejected || !layout_rejected)throw std::runtime_error("residual negative checks failed");
    std::cout<<"BLOCK_PLAN name=layer1_0 image="<<f1.image_id
        <<" stem_prefix=offline_plaintext diagnostic_stem_bootstraps=1 block_bootstraps=2"
        <<" real_convolutions=2 residual_adds=1 final_q=9 final_log_scale=40\n"
        <<"BLOCK_ORDER save_shortcut_Q9,conv1_Q6,act1_bootstrap_Q31,relu_Q9,conv2_Q6,residual_Q6,act2_bootstrap_Q31,relu_Q9\n"
        <<"RESIDUAL_PLAN main_Q=6 shortcut_Q=9 aligned_Q=6 rescale=0 shortcut_preserved=true\n"
        <<"BLOCK_NEGATIVE_CHECKS scale_mismatch=REJECTED layout_mismatch=REJECTED\n";
}

class Probe {
    const ReluFixture &relu;
    const ApplicationConvFixture &f1,&f2;
    const Runtime &plan1,&plan2;
    CT shortcut;
    Values shortcut_reference,shortcut_decoded;
    std::string first_target_failure;
    double maximum_boundary_error=0;
    int bootstraps=1,convolutions=0,residuals=0;
    void record(const std::string &stage,const BaselineContinuation &b,const CT &value,const Values &reference) {
        const double error=b.replay.check("block/"+stage,value,reference);
        maximum_boundary_error=std::max(maximum_boundary_error,error);
        if(error>1e-4 && first_target_failure.empty())first_target_failure=stage;
        double domain=0;for(auto v:decoded_values(b.replay,value))domain=std::max(domain,double(std::abs(v)));
        std::cout<<"BLOCK_BOUNDARY stage="<<stage<<" q="<<value.meta.q_count
            <<" log_scale="<<std::log2(value.meta.scale)<<" max_abs="<<error<<" domain="<<domain
            <<" target_1e_4="<<(error<=1e-4?"PASS":"FAIL")<<'\n';
        if(!std::isfinite(domain) || domain>1)throw std::runtime_error("block activation left verified domain at "+stage);
    }
    int refresh(const std::string &label,const BaselineContinuation &b,const CT &input,const Values &reference,
        const std::function<int(const ReluContinuation &)> &next) {
        ++bootstraps;
        std::cout<<"BLOCK_PHASE "<<label<<" bootstrap_index="<<bootstraps<<" source=previous_GPU_ciphertext\n";
        return bootstrap_existing_ciphertext(b.metadata,b.eval,b.keys,b.replay,input,reference,
            b.rotation_keys,b.bootstrap_rotation_keys,[&](const BaselineContinuation &updated) {
                record(label+".native_bootstrap",updated,updated.native_output,reference);
                return continue_to_relu(updated,relu,next);
            });
    }
    int after_conv1(const ConvContinuation &c) {
        ++convolutions;const auto &b=c.relu.bootstrap;
        record("conv1",b,c.output,c.original);
        return refresh("act1",b,c.output,c.original,[&](const ReluContinuation &r) {
            record("act1.relu",r.bootstrap,r.output,r.original);
            return execute(r,f2,plan2,[&](const ConvContinuation &next){return after_conv2(next);});
        });
    }
    int after_conv2(const ConvContinuation &c) {
        ++convolutions;const auto &b=c.relu.bootstrap;
        record("conv2",b,c.output,c.original);
        if(shortcut.meta.q_count!=9)throw std::runtime_error("saved shortcut was mutated");
        b.replay.scale_check(shortcut,40);
        if(b.replay.check("block/shortcut_retained",shortcut,shortcut_decoded)>1e-9)
            throw std::runtime_error("shortcut data changed while evaluating main branch");
        // No keys needed for add/drop. The adapter uses the exact original
        // residual_add function with actual main/shortcut ciphertexts.
        GpuGaloisKeysData unused;Runtime runtime(b.metadata,&b.eval,&unused);
        CT left,right;b.eval.drop_modulus(c.output,left,c.output.meta.parms_id);
        b.eval.drop_modulus(shortcut,right,shortcut.meta.parms_id);
        const auto local=add_values(decoded_values(b.replay,c.output),shortcut_decoded);
        auto sum=residual(runtime,runtime.device(std::move(left)),runtime.device(std::move(right)));
        ++residuals;
        if(runtime.counts["rescale"]!=0 || shortcut.meta.q_count!=9)
            throw std::runtime_error("residual introduced rescale or changed shortcut");
        const double local_error=b.replay.check("block/residual_local_arithmetic",*sum.gpu,local);
        if(local_error>1e-7)throw std::runtime_error("residual local arithmetic guard failed");
        const auto original=add_values(c.original,shortcut_reference);
        record("residual",b,*sum.gpu,original);
        std::cout<<"RESIDUAL_RESULT local_max="<<local_error<<" main_Q=6 shortcut_Q=9 output_Q=6 rescale=0\n";
        return refresh("act2",b,*sum.gpu,original,[&](const ReluContinuation &r) {
            const auto independent=pack(original_application_first_block(f1,f2));
            if(compare_values("block_final_continuous_reference_vs_independent_CHW",r.original,independent)>1e-10)
                throw std::runtime_error("block plaintext reference drifted from original application");
            record("act2.relu",r.bootstrap,r.output,independent);
            r.bootstrap.replay.scale_check(r.output,40);
            if(r.output.meta.q_count!=9 || bootstraps!=3 || convolutions!=2 || residuals!=1)
                throw std::runtime_error("incomplete BasicBlock execution");
            const double error=r.bootstrap.replay.check("block/final_original_CHW",r.output,independent);
            const bool pass=first_target_failure.empty();
            std::cout<<"BLOCK_RESULT integration=PASS boundary_target_1e_4="<<(pass?"PASS":"FAIL")
                <<" first_target_failure="<<(pass?"none":first_target_failure)
                <<" max_boundary_error="<<maximum_boundary_error<<" final_max="<<error
                <<" final_q=9 final_log_scale="<<std::log2(r.output.meta.scale)
                <<" bootstraps="<<bootstraps<<" real_convolutions="<<convolutions<<" residual_adds="<<residuals
                <<" completed_blocks=1 encryptions=1 no_reencryption=true full_network_tested=false"
                <<" production_per_level_P_tested=false security_approved=false timing_benchmark=false\n";
            return pass?0:1;
        });
    }
public:
    Probe(const ReluFixture &r,const ApplicationConvFixture &a,const ApplicationConvFixture &b,
        const Runtime &p1,const Runtime &p2):relu(r),f1(a),f2(b),plan1(p1),plan2(p2){}
    int start(const ReluContinuation &r) {
        record("stem.relu",r.bootstrap,r.output,r.original);
        r.bootstrap.eval.drop_modulus(r.output,shortcut,r.output.meta.parms_id);
        shortcut_reference=r.original;shortcut_decoded=decoded_values(r.bootstrap.replay,shortcut);
        return execute(r,f1,plan1,[&](const ConvContinuation &c){return after_conv1(c);});
    }
};
} // namespace

int main(int argc,char **argv) {
    try {
        if(argc<4 || argc>6 || (std::string(argv[1])!="--metadata-only" && std::string(argv[1])!="--unsafe-accuracy-only"))
            throw std::runtime_error("usage: bootstrap_block_precision --metadata-only|--unsafe-accuracy-only fixture.txt image_id [--lcdnn-s2c-first-real] [--bootstrap-only|--relu-only]");
        bool output_real_projection=false,bootstrap_only=false,relu_only=false;
        for(int i=4;i<argc;++i) {
            const std::string option=argv[i];
            if(option=="--lcdnn-s2c-first-real")output_real_projection=true;
            else if(option=="--bootstrap-only")bootstrap_only=true;
            else if(option=="--relu-only")relu_only=true;
            else throw std::runtime_error("unknown bootstrap branch");
        }
        if(bootstrap_only && relu_only)throw std::runtime_error("choose only one stop boundary");
        std::size_t consumed=0;const int image=std::stoi(argv[3],&consumed);
        if(consumed!=std::string(argv[3]).size() || image<0 || image>=10000)throw std::runtime_error("invalid CIFAR image id");
        ReluFixture relu(argv[2]);const auto f1=load_application_conv_fixture(image,1),f2=load_application_conv_fixture(image,2);
        const auto packed=conv_probe::pack(f1.stem);
        std::vector<std::complex<double>> input;for(auto v:packed)input.emplace_back(v);
        std::unique_ptr<conv_probe::Runtime> p1,p2;
        std::string profile="q50-fixed45",dataset="grid";
        char *args[]={argv[0],argv[1],profile.data(),dataset.data(),argv[2]};
        return run_bootstrap_baseline(5,args,[&](const BaselineMetadata &m) {
            check_continuation_metadata(m,relu);
            p1=std::make_unique<conv_probe::Runtime>(conv_probe::metadata(m,f1));
            p2=std::make_unique<conv_probe::Runtime>(conv_probe::metadata(m,f2));
            block_probe::check_metadata(m,f1,f2);
        },[&](const BaselineContinuation &b) {
            if(bootstrap_only) {
                std::cout<<"BLOCK_STOP requested=bootstrap_only completed_bootstraps=1"
                    <<" relu_tested=false convolution_tested=false\n";
                return 0;
            }
            std::unique_ptr<block_probe::Probe> block;
            if(!relu_only)block=std::make_unique<block_probe::Probe>(relu,f1,f2,*p1,*p2);
            return continue_to_relu(b,relu,[&](const ReluContinuation &r){
                if(relu_only) {
                    std::cout<<"BLOCK_STOP requested=relu_only completed_bootstraps=1"
                        <<" completed_relus=1 convolution_tested=false\n";
                    return 0;
                }
                return block->start(r);
            });
        },&input,output_real_projection);
    }catch(const std::exception &e){std::cerr<<"ERROR "<<e.what()<<'\n';return 2;}
}
