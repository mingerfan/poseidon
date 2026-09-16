// Bounded-lifetime, correctness-first ResNet20 diagnostic.
//
// The full application graph is evaluated on one continuously encrypted GPU
// state. Decoding is used only as an observer at named boundaries and is never
// fed back into the encrypted computation. The sole encryption API is scoped
// to the original image patches consumed by the unchanged encrypted stem.
#define POSEIDON_BOOTSTRAP_RELU_LIBRARY
#include "bootstrap_relu_precision.cpp"
#include "original_conv_reference.h"
#include "conv_probe_runtime.h"

#include <optional>
#include <type_traits>
#ifdef POSEIDON_S2C_FIRST_ENABLE_TIMING
#include "../../../bench/resnet20_s2c_first/gpu_activity_timing.h"
#endif

namespace network_probe {
namespace app=poseidon::benchmark::resnet20_gpu::core;
using Runtime=app::DiagnosticConvRuntime;
using Tensor=app::DiagnosticConvTensor;

struct DirectPlan {
    std::map<std::size_t,std::set<int>> rotations;
    std::map<std::string,int> counts;
};

struct OwnedActivation {
    CT ciphertext;
    Values reference;
};

Values pack_chw(const std::vector<double> &values,int h,int w,int c,int k) {
    if(values.size()!=static_cast<std::size_t>(h*w*c))
        throw std::runtime_error("CHW oracle shape mismatch");
    auto shape=app::make_shape(h,w,c,k,32768);
    if(shape.packs.size()!=1)throw std::runtime_error("network diagnostic expects one ciphertext per tensor");
    Values packed(32768);
    for(int channel=0;channel<c;++channel)for(int row=0;row<h;++row)for(int col=0;col<w;++col)
        packed[shape.slot_index(channel,row,col)]=values[(static_cast<std::size_t>(channel)*h+row)*w+col];
    return packed;
}

Tensor tensor_from_host(const BaselineMetadata &m,const Values &packed,
    int h,int w,int c,int k,std::size_t q=9) {
    Runtime runtime(m);auto tensor=app::make_shape(h,w,c,k,32768);
    if(tensor.packs.size()!=1)throw std::runtime_error("host tensor unexpectedly spans ciphertexts");
    tensor.packs[0]=runtime.host(packed,q,std::exp2(40));return tensor;
}

Tensor tensor_from_device(const BaselineMetadata &m,CT ciphertext,
    int h,int w,int c,int k,const GpuGaloisKeysData &keys,GpuEvaluator &eval) {
    Runtime runtime(m,&eval,&keys);auto tensor=app::make_shape(h,w,c,k,32768);
    if(tensor.packs.size()!=1)throw std::runtime_error("GPU tensor unexpectedly spans ciphertexts");
    tensor.packs[0]=runtime.device(std::move(ciphertext));return tensor;
}

std::vector<int> rotation_union(const DirectPlan &plan) {
    std::set<int> all;for(const auto &[q,steps]:plan.rotations)all.insert(steps.begin(),steps.end());
    return {all.begin(),all.end()};
}

std::vector<std::size_t> rotation_levels(const DirectPlan &plan) {
    std::vector<std::size_t> levels;for(const auto &[q,steps]:plan.rotations)if(!steps.empty())levels.push_back(q);
    return levels;
}

template<class Function>
auto run_direct(const BaselineSession &s,const DirectPlan &plan,const std::string &label,Function &&function) {
    auto steps=rotation_union(plan);
    GpuGaloisKeysData keys;
    if(!steps.empty()) {
        keys=s.rotation_keys(steps);
        GpuUploader::prepare_key_views_for_q_counts(keys,rotation_levels(plan));
    }
    Runtime runtime(s.metadata,&s.eval,&keys);
    auto result=function(runtime);
    if(runtime.rotations!=plan.rotations || runtime.counts!=plan.counts)
        throw std::runtime_error(label+" GPU graph differs from CPU trace");
    gpu_check_cuda(cudaDeviceSynchronize(),("finish "+label+" before releasing direct keys").c_str());
    std::cout<<"NETWORK_DIRECT stage="<<label<<" rotations="<<steps.size();
    for(const auto &[name,count]:runtime.counts)std::cout<<' '<<name<<'='<<count;
    std::cout<<'\n';
    return result;
}

Tensor apply_conv(const Runtime &runtime,Tensor input,const ApplicationLayerFixture &layer) {
    if(input.h!=layer.input_h || input.w!=layer.input_w || input.c!=layer.input_channels)
        throw std::runtime_error("network convolution input shape mismatch");
    auto output=app::conv2d_bn(input,layer.output_channels,layer.stride,3,3,
        layer.weights,layer.bn_scale,layer.bn_bias,runtime);
    if(output.packs.size()!=1 || output.packs[0].meta.q_count!=6 ||
        std::abs(std::log2(output.packs[0].meta.scale)-40)>1e-7)
        throw std::runtime_error("network convolution did not produce Q6/scale40");
    return output;
}

DirectPlan trace_conv(const BaselineMetadata &m,const Values &input,
    const ApplicationLayerFixture &layer,int k,const Values &expected) {
    Runtime cpu(m);auto output=apply_conv(cpu,tensor_from_host(m,input,
        layer.input_h,layer.input_w,layer.input_channels,k),layer);
    if(compare_values("network/conv_SIMD_vs_CHW_oracle",*output.packs[0].values,expected)>1e-10)
        throw std::runtime_error("network convolution oracle mismatch");
    return {cpu.rotations,cpu.counts};
}

OwnedActivation relu_existing_q31(const BaselineSession &s,const ReluFixture &fixture,
    const CT &input,const Values &reference,const std::string &label) {
    if(input.meta.q_count!=31)throw std::runtime_error(label+" ReLU input must be Q31");
    s.replay.scale_check(input,40);
    const Values decoded=decoded_values(s.replay,input);
    double domain=0,imag=0;for(auto value:decoded) {
        domain=std::max(domain,double(std::abs(value)));
        imag=std::max(imag,double(std::abs(value.imag())));
    }
    if(!std::isfinite(domain) || domain>1)throw std::runtime_error(label+" ReLU input left [-1,1]");
    Ref real_reference;for(auto value:reference) {
        if(std::abs(value.imag())>1e-12)throw std::runtime_error(label+" oracle is not real");
        real_reference.push_back(value.real());
    }
    Values local=decoded,chain=reference;
    auto current=s.replay.drop(input,0);double stage_error=0,max_local=0;
    for(std::size_t i=0;i<fixture.stages.size();++i) {
        current=s.replay.component(fixture.stages[i],current,real_reference,i+1,stage_error);
        for(auto &value:local)value=plain_complex_node(*fixture.stages[i].tree,value);
        for(auto &value:chain)value=plain_complex_node(*fixture.stages[i].tree,value);
        max_local=std::max(max_local,s.replay.check(label+".relu_stage"+std::to_string(i+1)+"_local",current,local));
    }
    auto output=s.replay.tail(input,current,fixture.tail_work,fixture.tail_drop);
    Values original(reference.size());
    for(std::size_t i=0;i<reference.size();++i) {
        local[i]=decoded[i]*(local[i]+0.5L);
        chain[i]=reference[i]*(chain[i]+0.5L);
        original[i]=original_application_relu(double(reference[i].real()));
    }
    const double local_error=s.replay.check(label+".relu_local_final",output,local);
    const double boundary_error=s.replay.check(label+".relu_CHW_oracle",output,original);
    s.replay.scale_check(output,40);
    if(output.meta.q_count!=9 || max_local>1e-5 || local_error>1e-5)
        throw std::runtime_error(label+" ReLU arithmetic guard failed");
    std::cout<<"NETWORK_RELU stage="<<label<<" domain="<<domain<<" max_imag="<<imag
        <<" local_max="<<std::max(max_local,local_error)<<" oracle_max="<<boundary_error
        <<" q=9 log_scale=40\n";
    return {std::move(output),std::move(original)};
}

OwnedActivation refresh_relu(const BaselineSession &s,const ReluFixture &fixture,
    const CT &input,const Values &reference,const std::string &label,
    BootstrapOfflineCache &offline_cache) {
    if(input.meta.q_count!=6)throw std::runtime_error(label+" bootstrap input must be Q6");
    std::optional<OwnedActivation> owned;
    const int rc=bootstrap_existing_ciphertext(s.metadata,s.eval,s.keys,s.replay,input,reference,
        s.rotation_keys,s.bootstrap_rotation_keys,[&](const BaselineContinuation &bootstrap) {
            return continue_to_relu(bootstrap,fixture,[&](const ReluContinuation &relu) {
                CT clone;s.eval.drop_modulus(relu.output,clone,relu.output.meta.parms_id);
                owned.emplace(OwnedActivation{std::move(clone),relu.original});
                return 0;
            });
        },&offline_cache);
    if(rc || !owned)throw std::runtime_error(label+" bootstrap/ReLU failed");
    gpu_check_cuda(cudaDeviceSynchronize(),("release "+label+" bootstrap temporaries").c_str());
    std::cout<<"NETWORK_REFRESH stage="<<label<<" bounded_lifetime=true q=9 log_scale=40\n";
    return std::move(*owned);
}

double check_boundary(const BaselineSession &s,const std::string &label,
    const CT &actual,const Values &expected,double &maximum,std::string &first_failure,
    double target=0.1) {
    const double error=s.replay.check("network/"+label,actual,expected);
    maximum=std::max(maximum,error);
    if((!std::isfinite(error) || error>target) && first_failure.empty())first_failure=label;
    std::cout<<"NETWORK_BOUNDARY stage="<<label<<" q="<<actual.meta.q_count
        <<" log_scale="<<std::log2(actual.meta.scale)<<" max_abs="<<error
        <<" target="<<target<<" result="<<(error<=target?"PASS":"FAIL")<<'\n';
    return error;
}

#ifdef POSEIDON_S2C_FIRST_ENABLE_TIMING
using ActivityTiming=poseidon::benchmark::s2c_first::GpuActivityTiming;

template<class Function>
auto run_activity(ActivityTiming *timing,const std::string &label,Function &&function) {
    using Result=decltype(function());
    if(!timing) {
        if constexpr(std::is_void_v<Result>) {function();return;}
        else return function();
    }
    timing->stage(label,true);
    try {
        if constexpr(std::is_void_v<Result>) {
            function();timing->stage(label,false);return;
        } else {
            Result result=function();timing->stage(label,false);return result;
        }
    } catch(...) {
        try {timing->stage(label,false);} catch(...) {}
        throw;
    }
}

CT evaluate_relu_no_observer(Replay &replay,const ReluFixture &fixture,const CT &input) {
    Ref unused;double stage_error=0;
    auto current=replay.drop(input,0);
    for(std::size_t index=0;index<fixture.stages.size();++index)
        current=replay.component(fixture.stages[index],current,unused,index+1,stage_error);
    auto output=replay.tail(input,current,fixture.tail_work,fixture.tail_drop);
    replay.scale_check(output,40);
    if(output.meta.q_count!=9)throw std::runtime_error("prepared ReLU did not produce Q9");
    return output;
}

void prepare_bootstrap_cpu(const BaselineMetadata &m,const CT &input,
    BootstrapOfflineCache &cache) {
    if(cache.stc_cpu)return;
    const auto nq=m.ctx.parameters_literal()->q().size();
    const double working_scale=std::exp2(45);
    const double c2s_scaling=m.poly.q_div()/(m.poly.k()*m.poly.sc_fac()*m.poly.q_diff());
    HomomorphicDFTMatrixLiteral stc_literal(
        poseidon::decode,16,15,5,{1,1,1},true,1,false,1);
    HomomorphicDFTMatrixLiteral cts_literal(
        poseidon::encode,16,15,nq-1,{1,1,1},true,c2s_scaling,false,1);
    cache.stc_cpu=std::make_unique<LinearMatrixGroup>();
    cache.cts_cpu=std::make_unique<LinearMatrixGroup>();
    stc_literal.create_dynamic(*cache.stc_cpu,m.encoder,input.meta.scale,
        working_scale,working_scale);
    cts_literal.create_dynamic(*cache.cts_cpu,m.encoder,working_scale,
        working_scale,working_scale);
    std::set<int> rotations{0};
    rotations.insert(cache.stc_cpu->rot_index().begin(),cache.stc_cpu->rot_index().end());
    rotations.insert(cache.cts_cpu->rot_index().begin(),cache.cts_cpu->rot_index().end());
    cache.rotations.assign(rotations.begin(),rotations.end());
    std::cout<<"CONTINUOUS_PREPARE_BOOTSTRAP matrices=ready rotations="<<cache.rotations.size()<<'\n';
}

struct ContinuousGraphPlans {
    std::vector<DirectPlan> convolutions;
    std::vector<std::optional<DirectPlan>> shortcuts;
    DirectPlan head;
    std::map<std::size_t,std::set<int>> rotations;

    ContinuousGraphPlans(const BaselineMetadata &m,
        const ApplicationNetworkFixture &fixture) {
        auto merge=[&](const DirectPlan &plan) {
            for(const auto &[q,steps]:plan.rotations)
                rotations[q].insert(steps.begin(),steps.end());
        };
        int h=32,w=32,c=16,k=1;
        shortcuts.resize(fixture.blocks.size());
        for(std::size_t index=0;index<fixture.blocks.size();++index) {
            const auto &block=fixture.blocks[index];
            const int next_h=h/block.conv1.stride,next_w=w/block.conv1.stride;
            const int next_k=k*block.conv1.stride;
            convolutions.push_back(trace_conv(m,pack_chw(block.input,h,w,c,k),
                block.conv1,k,pack_chw(block.conv1_output,next_h,next_w,
                    block.conv1.output_channels,next_k)));
            merge(convolutions.back());
            convolutions.push_back(trace_conv(m,pack_chw(block.act1_output,
                next_h,next_w,block.conv1.output_channels,next_k),block.conv2,
                next_k,pack_chw(block.conv2_output,next_h,next_w,
                    block.conv2.output_channels,next_k)));
            merge(convolutions.back());
            if(block.option_a_shortcut) {
                Runtime cpu(m);(void)app::downsample_shortcut(tensor_from_host(
                    m,pack_chw(block.input,h,w,c,k),h,w,c,k),cpu);
                shortcuts[index]=DirectPlan{cpu.rotations,cpu.counts};
                merge(*shortcuts[index]);
            }
            h=next_h;w=next_w;c=block.conv2.output_channels;k=next_k;
        }
        Runtime cpu(m);auto tensor=tensor_from_host(m,pack_chw(
            fixture.blocks.back().output,h,w,c,k),h,w,c,k);
        auto features=app::global_average_pool(tensor,40,cpu);
        (void)app::fully_connected(features,64,fixture.linear_weight,
            fixture.linear_bias,10,cpu);
        head={cpu.rotations,cpu.counts};merge(head);
    }

    std::vector<int> steps() const {
        std::set<int> all;for(const auto &[q,at_level]:rotations) {
            (void)q;all.insert(at_level.begin(),at_level.end());
        }
        return {all.begin(),all.end()};
    }
    std::vector<std::size_t> levels() const {
        std::vector<std::size_t> result;
        for(const auto &[q,steps]:rotations)if(!steps.empty())result.push_back(q);
        return result;
    }
};

class ContinuousBootstrap {
    const BaselineSession &s;
    GpuGaloisKeysData rotations;
    BootstrapOfflineCache cpu;
    GpuLinearMatrixGroupQP stc,cts;
    GpuBootstrapData data;
    GpuPlaintextData minus_i,plus_i;
    GpuDoubleHoistWorkspace stc_workspace,cts_workspace;
    GpuBootstrapWorkspace evalmod_workspace;
    double alpha=1;
public:
    explicit ContinuousBootstrap(const BaselineSession &session):s(session) {
        CT metadata_input;metadata_input.meta.q_count=6;
        metadata_input.meta.scale=std::exp2(40);
        prepare_bootstrap_cpu(s.metadata,metadata_input,cpu);
        rotations=s.bootstrap_rotation_keys(cpu.rotations);
        const auto nq=s.metadata.ctx.parameters_literal()->q().size();
        GpuUploader::prepare_key_views_for_q_counts(
            rotations,{4,5,6,nq-4,nq-2,nq-1,nq,s.metadata.native_plan.output_q_count});
        stc=GpuUploader::upload_linear_matrix_group_qp(*cpu.stc_cpu,s.metadata.ctx,0,1);
        cts=GpuUploader::upload_linear_matrix_group_qp(*cpu.cts_cpu,s.metadata.ctx,0,1);
        FoldSpec fold(s.metadata);alpha=fold.alpha;data.eval_mod=fold.upload(s.metadata,&s.keys);
        Plaintext host_minus_i,host_plus_i;
        s.metadata.encoder.encode(std::complex<double>(0,-1),s.metadata.c2s_id,1,host_minus_i);
        s.metadata.encoder.encode(std::complex<double>(0,1),
            data.eval_mod.output_parms_id,1,host_plus_i);
        minus_i=GpuUploader::upload_plaintext(host_minus_i,0);
        plus_i=GpuUploader::upload_plaintext(host_plus_i,0);
        std::cout<<"CONTINUOUS_PREPARE_BOOTSTRAP rotations="<<cpu.rotations.size()
            <<" stc_diagonals="<<matrix_plaintext_count(*cpu.stc_cpu)
            <<" cts_diagonals="<<matrix_plaintext_count(*cpu.cts_cpu)
            <<" evalmod_q_out="<<data.eval_mod.output_q_count<<'\n';
    }

    CT run(const CT &encrypted_input,ActivityTiming *timing) {
        const auto &m=s.metadata;const auto &ctx=m.ctx;auto &eval=s.eval;
        const auto nq=ctx.parameters_literal()->q().size();
        const double working_scale=std::exp2(45);
        const double preparation_target=std::exp2(
            std::round(std::log2(ctx.crt_context()->q0()/32)));
        if(encrypted_input.meta.q_count!=6)
            throw std::runtime_error("continuous bootstrap input must be Q6");
        CT stc_out,prepared,raised,real,imag,eval_real,eval_imag;
        run_activity(timing,"bootstrap.S2C",[&]{
            eval.dft_double_hoist(encrypted_input,stc,rotations,
                stc_workspace,stc_out);
        });
        run_activity(timing,"bootstrap.prepare",[&]{
            eval.bootstrap_prepare_modraise_input(stc_out,prepared,
                ctx.crt_context()->parms_id_map().at(1),preparation_target);
        });
        run_activity(timing,"bootstrap.ModRaise",[&]{
            eval.raise_modulus(prepared,raised);
        });
        raised.meta.scale=working_scale;
        run_activity(timing,"bootstrap.C2S",[&]{
            eval.coeff_to_slot_double_hoist(raised,cts,minus_i,rotations,
                cts_workspace,real,imag);
        });
        run_activity(timing,"bootstrap.EvalMod",[&]{
            eval.eval_mod_high_precision(real,data,s.keys,evalmod_workspace,eval_real);
        });
        run_activity(timing,"bootstrap.EvalMod",[&]{
            eval.eval_mod_high_precision(imag,data,s.keys,evalmod_workspace,eval_imag);
        });
        CT scaled_imag,combined,half,conjugated,output;
        run_activity(timing,"bootstrap.recombine",[&]{
            eval.multiply_plain(eval_imag,plus_i,scaled_imag);
            eval.add(eval_real,scaled_imag,combined);
            eval.multiply_scalar(combined,16,half);
            eval.conjugate(half,rotations,conjugated);
            eval.add(half,conjugated,output);
        });
        output.meta.scale*=alpha;
        s.replay.scale_check(output,40);
        if(output.meta.q_count!=31)
            throw std::runtime_error("continuous bootstrap output must be Q31");
        return output;
    }

private:
    static std::size_t matrix_plaintext_count(const LinearMatrixGroup &group) {
        std::size_t result=0;
        for(const auto &matrix:group.data())result+=matrix.plain_vec.size();
        return result;
    }
};

class FixedDnum2ApplicationKeys {
    const BaselineSession &s;
    std::map<std::size_t,std::unique_ptr<ApplicationKeySwitchRuntime>> rotations;
    std::map<std::size_t,std::unique_ptr<ApplicationKeySwitchRuntime>> relin_by_p;
    std::map<std::size_t,std::size_t> relin_q_to_p;
    bool prepared=false;

    static std::size_t p_for_q(std::size_t q) {return std::max<std::size_t>(2,(q+1)/2);}
    static std::size_t key_payload_bytes(std::size_t q,std::size_t p,
        std::size_t key_count) {
        const std::size_t dnum=(q+p-1)/p;
        return key_count*dnum*2*(q+p)*(std::size_t(1)<<16)*sizeof(GpuWord);
    }
    static std::size_t active_parameter_ntt_bytes(std::size_t q,std::size_t p) {
        constexpr std::size_t large_ntt_arrays=7;
        return large_ntt_arrays*(q+p)*(std::size_t(1)<<16)*sizeof(GpuWord);
    }
    static std::size_t actual_key_bytes(const GpuEvaluationKeyData &keys) {
        std::size_t bytes=0;
        for(const auto &field:keys.fields_)bytes+=field.size()*sizeof(GpuWord);
        return bytes;
    }
    static void collect_node_relin_levels(const Node &node,std::set<std::size_t> &levels) {
        if(node.type=="combine") {
            levels.insert(entry_q-node.work);
            collect_node_relin_levels(*node.quotient,levels);
            collect_node_relin_levels(*node.remainder,levels);
        }
    }
    static std::set<std::size_t> relin_levels(const ReluFixture &fixture) {
        std::set<std::size_t> levels;
        for(const auto &stage:fixture.stages) {
            for(const auto &basis:stage.basis)levels.insert(entry_q-basis.work);
            collect_node_relin_levels(*stage.tree,levels);
        }
        levels.insert(entry_q-fixture.tail_work);
        return levels;
    }
    void verify_compatible(const ApplicationKeySwitchRuntime &runtime,
        std::size_t active_q) const {
        const auto primary=s.metadata.ctx.crt_context()->parms_id_map().at(active_q-1);
        const auto local=runtime.context.crt_context()->parms_id_map().at(active_q-1);
        if(primary!=local)throw std::logic_error(
            "level-aware application Q context is incompatible with ciphertext");
    }

public:
    explicit FixedDnum2ApplicationKeys(const BaselineSession &session):s(session) {}

    void prepare(const std::map<std::size_t,std::set<int>> &rotation_plan,
        const ReluFixture &relu) {
        if(prepared || rotation_plan.empty())
            throw std::logic_error("application KeySwitch plan is invalid or already prepared");
        const auto global_p=s.metadata.ctx.parameters_literal()->p().size();
        const auto global_q=s.metadata.ctx.parameters_literal()->q().size();
        const auto relin_plan=relin_levels(relu);
        std::set<std::size_t> relin_p_counts;
        std::map<std::size_t,std::vector<std::size_t>> relin_q_counts_by_p;
        std::size_t planned_rotation_bytes=0,planned_relin_bytes=0;
        std::size_t planned_parameter_ntt_bytes=0;
        for(const auto &[q,steps]:rotation_plan) {
            const auto p=p_for_q(q);
            if(p>global_p || (q+p-1)/p!=2 || steps.empty())
                throw std::logic_error("rotation plan cannot preserve fixed dnum=2");
            const auto bytes=key_payload_bytes(q,p,steps.size());
            planned_rotation_bytes+=bytes;
            planned_parameter_ntt_bytes+=active_parameter_ntt_bytes(q,p);
            std::cout<<"APPLICATION_KEY_PLAN kind=rotation q="<<q<<" p="<<p
                <<" dnum=2 keys="<<steps.size()
                <<" payload_MiB="<<double(bytes)/(1ULL<<20)<<'\n';
        }
        for(auto q:relin_plan) {
            const auto p=p_for_q(q);
            if(p>global_p || (q+p-1)/p!=2)
                throw std::logic_error("relinearization plan cannot preserve fixed dnum=2");
            relin_q_to_p.emplace(q,p);relin_p_counts.insert(p);
            relin_q_counts_by_p[p].push_back(q);
            planned_parameter_ntt_bytes+=active_parameter_ntt_bytes(q,p);
        }
        for(auto p:relin_p_counts) {
            const auto context_q=std::min(global_q,2*p);
            planned_relin_bytes+=key_payload_bytes(context_q,p,1);
        }
        const auto old_global_rotation_bytes=key_payload_bytes(global_q,global_p,109);
        const auto planned_total=planned_rotation_bytes+planned_relin_bytes;
        std::cout<<"APPLICATION_KEY_PREFLIGHT old_global_rotation_MiB="
            <<double(old_global_rotation_bytes)/(1ULL<<20)
            <<" level_rotation_MiB="<<double(planned_rotation_bytes)/(1ULL<<20)
            <<" level_relin_MiB="<<double(planned_relin_bytes)/(1ULL<<20)
            <<" level_total_MiB="<<double(planned_total)/(1ULL<<20)
            <<" active_parameter_ntt_MiB="
            <<double(planned_parameter_ntt_bytes)/(1ULL<<20)
            <<" resident=true swaps=false\n";
        if(planned_total>=old_global_rotation_bytes)
            throw std::runtime_error(
                "level-aware application keys do not fit the replaced global-key budget");

        for(const auto &[q,steps]:rotation_plan) {
            const auto p=p_for_q(q);
            std::vector<int> level_steps(steps.begin(),steps.end());
            auto runtime=s.application_keyswitch(q,p,{q},level_steps,false);
            verify_compatible(*runtime,q);
            if(runtime->effective_dnum!=2)
                throw std::logic_error("rotation runtime effective dnum changed");
            rotations.emplace(q,std::move(runtime));
        }
        for(auto p:relin_p_counts) {
            const auto context_q=std::min(global_q,2*p);
            auto runtime=s.application_keyswitch(
                context_q,p,relin_q_counts_by_p.at(p),{},true);
            if(runtime->effective_dnum!=2)
                throw std::logic_error("relinearization runtime effective dnum changed");
            relin_by_p.emplace(p,std::move(runtime));
        }
        for(const auto &[q,p]:relin_q_to_p)verify_compatible(*relin_by_p.at(p),q);

        std::size_t actual_rotation_bytes=0,actual_relin_bytes=0;
        for(const auto &[q,runtime]:rotations) {
            (void)q;actual_rotation_bytes+=actual_key_bytes(runtime->gpu_galois_keys);
        }
        for(const auto &[p,runtime]:relin_by_p) {
            (void)p;actual_relin_bytes+=actual_key_bytes(runtime->gpu_relin_keys);
        }
        if(actual_rotation_bytes!=planned_rotation_bytes ||
            actual_relin_bytes!=planned_relin_bytes)
            throw std::logic_error("application evaluation-key payload differs from preflight");
        std::cout<<"APPLICATION_KEYS_READY rotation_levels="<<rotations.size()
            <<" relin_levels="<<relin_q_to_p.size()
            <<" relin_contexts="<<relin_by_p.size()
            <<" parameter_levels="<<rotations.size()+relin_q_to_p.size()
            <<" payload_MiB="<<double(actual_rotation_bytes+actual_relin_bytes)/(1ULL<<20)
            <<" dnum=2 resident=true swaps=false\n";
        prepared=true;
    }

    Runtime::RotationBackend rotation_backend(std::size_t q) {
        if(!prepared)throw std::logic_error("application KeySwitch keys are not prepared");
        auto found=rotations.find(q);
        if(found==rotations.end())
            throw std::logic_error("no level-aware rotation runtime for ciphertext Q");
        auto &runtime=*found->second;
        return {runtime.gpu_evaluator.get(),&runtime.gpu_galois_keys,
            &runtime.rotate_many_workspace};
    }

    void relinearize(const CT &source,CT &destination) {
        if(!prepared)throw std::logic_error("application KeySwitch keys are not prepared");
        auto planned=relin_q_to_p.find(source.meta.q_count);
        if(planned==relin_q_to_p.end())
            throw std::logic_error("unplanned application relinearization Q level");
        auto &runtime=*relin_by_p.at(planned->second);
        runtime.gpu_evaluator->relinearize(source,runtime.gpu_relin_keys,destination);
    }
};

class ContinuousNetwork {
    const BaselineSession &s;
    const ReluFixture &relu;
    const ApplicationNetworkFixture &fixture;
    ContinuousGraphPlans plans;
    GpuGaloisKeysData no_keys;
    FixedDnum2ApplicationKeys application_keys;
    std::unique_ptr<ContinuousBootstrap> bootstrap;
    std::unique_ptr<Runtime> stem_runtime,add_runtime,head_runtime;
    std::vector<std::unique_ptr<Runtime>> conv_runtimes;
    std::vector<std::unique_ptr<Runtime>> shortcut_runtimes;
    bool relu_ready=false;

    CT execute_relu(const CT &input,bool capture,ActivityTiming *timing) {
        s.replay.release_scratch();
        if(capture)s.replay.begin_plaintext_capture();
        else s.replay.begin_plaintext_replay();
        CT output=run_activity(timing,"relu",[&]{
            return evaluate_relu_no_observer(s.replay,relu,input);
        });
        if(capture)s.replay.finish_plaintext_capture();
        else s.replay.finish_plaintext_replay();
        s.replay.release_scratch();
        return output;
    }

    CT execute_conv(std::size_t index,CT input,const ApplicationLayerFixture &layer,
        int h,int w,int c,int k,bool capture,ActivityTiming *timing) {
        auto &runtime=*conv_runtimes.at(index);
        if(capture)runtime.begin_prepared_capture();else runtime.begin_prepared_replay();
        auto output=run_activity(timing,"conv_bn",[&]{
            return apply_conv(runtime,tensor_from_device(s.metadata,std::move(input),
                h,w,c,k,no_keys,s.eval),layer);
        });
        if(capture)runtime.finish_prepared_capture();else runtime.finish_prepared_replay();
        if(capture && (runtime.rotations!=plans.convolutions.at(index).rotations ||
            runtime.counts!=plans.convolutions.at(index).counts))
            throw std::runtime_error("continuous Conv capture differs from CPU graph");
        return std::move(*output.packs[0].gpu);
    }

    CT execute_shortcut(std::size_t block,CT input,int h,int w,int c,int k,
        bool capture,ActivityTiming *timing) {
        auto &runtime=*shortcut_runtimes.at(block);
        if(capture)runtime.begin_prepared_capture();else runtime.begin_prepared_replay();
        auto output=run_activity(timing,"shortcut",[&]{
            return app::downsample_shortcut(tensor_from_device(s.metadata,
                std::move(input),h,w,c,k,no_keys,s.eval),runtime);
        });
        if(capture)runtime.finish_prepared_capture();else runtime.finish_prepared_replay();
        if(capture && (runtime.rotations!=plans.shortcuts.at(block)->rotations ||
            runtime.counts!=plans.shortcuts.at(block)->counts))
            throw std::runtime_error("continuous shortcut capture differs from CPU graph");
        return std::move(*output.packs[0].gpu);
    }

    CT execute_graph(bool capture,ActivityTiming *timing=nullptr) {
        if(capture)stem_runtime->begin_prepared_capture();
        else stem_runtime->begin_prepared_replay();
        auto stem=run_activity(timing,"stem.conv_bn",[&]{
            return app::encrypted_stem_conv2d_bn(fixture.image,32,32,3,16,1,3,3,
                fixture.stem.weights,fixture.stem.bn_scale,fixture.stem.bn_bias,*stem_runtime);
        });
        if(capture)stem_runtime->finish_prepared_capture();
        else stem_runtime->finish_prepared_replay();
        if(stem.packs.size()!=1 || stem.packs[0].meta.q_count!=31 ||
            std::abs(std::log2(stem.packs[0].meta.scale)-40)>1e-7)
            throw std::runtime_error("continuous stem did not produce Q31/scale40");
        CT stem_q31=std::move(*stem.packs[0].gpu);
        CT active=execute_relu(stem_q31,capture && !relu_ready,timing);
        relu_ready=true;
        int h=32,w=32,c=16,k=1;std::size_t conv_index=0;
        for(std::size_t block_index=0;block_index<fixture.blocks.size();++block_index) {
            const auto &block=fixture.blocks[block_index];
            CT shortcut;run_activity(timing,"residual.clone",[&]{
                s.eval.drop_modulus(active,shortcut,active.meta.parms_id);
            });
            auto first=execute_conv(conv_index++,std::move(active),block.conv1,
                h,w,c,k,capture,timing);
            auto first_active=execute_relu(bootstrap->run(first,timing),false,timing);
            const int next_h=h/block.conv1.stride,next_w=w/block.conv1.stride;
            const int next_k=k*block.conv1.stride;
            auto second=execute_conv(conv_index++,std::move(first_active),block.conv2,
                next_h,next_w,block.conv1.output_channels,next_k,capture,timing);
            if(block.option_a_shortcut)
                shortcut=execute_shortcut(block_index,std::move(shortcut),
                    h,w,c,k,capture,timing);
            auto left=tensor_from_device(s.metadata,std::move(second),next_h,next_w,
                block.conv2.output_channels,next_k,no_keys,s.eval);
            auto right=tensor_from_device(s.metadata,std::move(shortcut),next_h,next_w,
                block.conv2.output_channels,next_k,no_keys,s.eval);
            auto summed=run_activity(timing,"residual.add",[&]{
                return app::residual_add(left,right,*add_runtime);
            });
            active=execute_relu(bootstrap->run(*summed.packs[0].gpu,timing),false,timing);
            h=next_h;w=next_w;c=block.conv2.output_channels;k=next_k;
        }
        if(capture)head_runtime->begin_prepared_capture();
        else head_runtime->begin_prepared_replay();
        auto logits=run_activity(timing,"head",[&]{
            auto tensor=tensor_from_device(s.metadata,std::move(active),h,w,c,k,
                no_keys,s.eval);
            auto features=app::global_average_pool(tensor,40,*head_runtime);
            return app::fully_connected(features,64,fixture.linear_weight,
                fixture.linear_bias,10,*head_runtime);
        });
        if(capture)head_runtime->finish_prepared_capture();
        else head_runtime->finish_prepared_replay();
        if(capture && (head_runtime->rotations!=plans.head.rotations ||
            head_runtime->counts!=plans.head.counts))
            throw std::runtime_error("continuous head capture differs from CPU graph");
        return std::move(*logits.gpu);
    }

public:
    ContinuousNetwork(const BaselineSession &session,const ReluFixture &r,
        const ApplicationNetworkFixture &f)
        :s(session),relu(r),fixture(f),plans(s.metadata,fixture),application_keys(s) {
        const auto steps=plans.steps();
        if(steps.size()!=109)
            throw std::runtime_error("continuous direct rotation union changed");
        application_keys.prepare(plans.rotations,relu);
        bootstrap=std::make_unique<ContinuousBootstrap>(s);
        stem_runtime=std::make_unique<Runtime>(s.metadata,&s.eval,&no_keys,
            [&](const std::vector<double> &slots){return s.encrypt_application_input(slots);});
        add_runtime=std::make_unique<Runtime>(s.metadata,&s.eval,&no_keys);
        auto rotation_resolver=[this](std::size_t q) {
            return application_keys.rotation_backend(q);
        };
        conv_runtimes.reserve(plans.convolutions.size());
        for(std::size_t i=0;i<plans.convolutions.size();++i)
            conv_runtimes.push_back(std::make_unique<Runtime>(
                s.metadata,&s.eval,&no_keys,Runtime::InputEncryptor{},rotation_resolver));
        shortcut_runtimes.resize(fixture.blocks.size());
        for(std::size_t i=0;i<fixture.blocks.size();++i)if(plans.shortcuts[i])
            shortcut_runtimes[i]=std::make_unique<Runtime>(
                s.metadata,&s.eval,&no_keys,Runtime::InputEncryptor{},rotation_resolver);
        head_runtime=std::make_unique<Runtime>(
            s.metadata,&s.eval,&no_keys,Runtime::InputEncryptor{},rotation_resolver);
        s.replay.set_relinearize_callback([this](const CT &source,CT &destination) {
            application_keys.relinearize(source,destination);
        });
        std::cout<<"CONTINUOUS_PREPARE_DIRECT rotations="<<steps.size()
            <<" levels="<<plans.levels().size()
            <<" hoisted=true level_aware_p=true dnum=2\n";
    }

    ~ContinuousNetwork() {s.replay.clear_relinearize_callback();}

    int run() {
        auto warm=execute_graph(true);
        gpu_check_cuda(cudaDeviceSynchronize(),"continuous prepared warmup");
        warm=CT{};gpu_check_cuda(cudaDeviceSynchronize(),"continuous release warmup output");
        if(stem_runtime->prepared_input_count()!=27 ||
            stem_runtime->prepared_plaintext_count()!=28 ||
            s.replay.plaintext_cache_size()!=46)
            throw std::runtime_error("continuous resident cache contract mismatch");
        std::size_t conv_plaintexts=0,shortcut_plaintexts=0;
        for(const auto &runtime:conv_runtimes)conv_plaintexts+=runtime->prepared_plaintext_count();
        for(const auto &runtime:shortcut_runtimes)if(runtime)
            shortcut_plaintexts+=runtime->prepared_plaintext_count();
        if(conv_plaintexts!=1696 || shortcut_plaintexts!=48 ||
            head_runtime->prepared_plaintext_count()!=93)
            throw std::runtime_error("continuous public plaintext inventory changed");
        std::size_t free_bytes=0,total_bytes=0;
        gpu_check_cuda(cudaMemGetInfo(&free_bytes,&total_bytes),"continuous prepared memory");
        std::cout<<"CONTINUOUS_PREPARED inputs=27 stem_plaintexts=28"
            <<" conv_plaintexts="<<conv_plaintexts
            <<" shortcut_plaintexts="<<shortcut_plaintexts
            <<" relu_plaintexts="<<s.replay.plaintext_cache_size()
            <<" head_plaintexts="<<head_runtime->prepared_plaintext_count()
            <<" free_MiB="<<double(free_bytes)/(1ULL<<20)<<'\n';

        poseidon::benchmark::s2c_first::GpuActivityTiming activity;
        cudaEvent_t start_event=nullptr,stop_event=nullptr;
        gpu_check_cuda(cudaEventCreate(&start_event),"continuous start event");
        gpu_check_cuda(cudaEventCreate(&stop_event),"continuous stop event");
        struct EventCleanup {cudaEvent_t &a,&b;~EventCleanup(){
            if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);}} cleanup{start_event,stop_event};
        gpu_check_cuda(cudaDeviceSynchronize(),"continuous timing start sync");
        activity.begin_continuous();
        gpu_check_cuda(cudaEventRecord(start_event),"continuous record start");
        const auto wall_begin=std::chrono::steady_clock::now();
        auto logits=execute_graph(false,&activity);
        const auto enqueue_end=std::chrono::steady_clock::now();
        gpu_check_cuda(cudaEventRecord(stop_event),"continuous record stop");
        gpu_check_cuda(cudaDeviceSynchronize(),"continuous inference completion");
        const auto wall_end=std::chrono::steady_clock::now();
        float event_ms=0;gpu_check_cuda(cudaEventElapsedTime(
            &event_ms,start_event,stop_event),"continuous event elapsed");
        auto result=activity.finish();result.print();
        const std::map<std::string,int> category_calls{
            {"bootstrap.C2S",18},{"bootstrap.EvalMod",36},
            {"bootstrap.ModRaise",18},{"bootstrap.S2C",18},
            {"bootstrap.prepare",18},{"bootstrap.recombine",18},
            {"conv_bn",18},{"head",1},{"relu",19},
            {"residual.add",9},{"residual.clone",9},{"shortcut",2},
            {"stem.conv_bn",1}};
        double breakdown_sum_ms=0;
        for(const auto &[category,stage]:result.stages) {
            if(!category_calls.count(category))
                throw std::runtime_error("unexpected continuous timing category: "+category);
            breakdown_sum_ms+=stage.gpu_ms;
            std::cout<<"GPU_ACTIVITY_BREAKDOWN category="<<category
                <<" gpu_ms="<<stage.gpu_ms
                <<" calls="<<category_calls.at(category)<<'\n';
        }
        if(result.stages.size()!=category_calls.size())
            throw std::runtime_error("continuous timing category inventory changed");
        std::cout<<"GPU_ACTIVITY_BREAKDOWN_SUM gpu_ms="<<breakdown_sum_ms
            <<" category_count="<<result.stages.size()<<'\n';
        const auto stage_ms=[&](const char *category) {
            return result.stages.at(category).gpu_ms;
        };
        const std::vector<std::pair<std::string,double>> report_categories{
            {"Conv+BN",stage_ms("conv_bn")+stage_ms("stem.conv_bn")},
            {"ReLU",stage_ms("relu")},
            {"Bootstrap",stage_ms("bootstrap.C2S")+stage_ms("bootstrap.EvalMod")+
                stage_ms("bootstrap.ModRaise")+stage_ms("bootstrap.S2C")+
                stage_ms("bootstrap.prepare")+stage_ms("bootstrap.recombine")},
            {"Shortcut",stage_ms("shortcut")},
            {"Pool+FC",stage_ms("head")},
            {"Residual/copy",stage_ms("residual.add")+stage_ms("residual.clone")}};
        double report_category_sum_ms=0;
        for(const auto &[category,gpu_ms]:report_categories) {
            report_category_sum_ms+=gpu_ms;
            std::cout<<"GPU_ACTIVITY_CATEGORY category="<<category
                <<" gpu_ms="<<gpu_ms<<'\n';
        }
        std::cout<<"GPU_ACTIVITY_CATEGORY_SUM gpu_ms="<<report_category_sum_ms
            <<" category_count="<<report_categories.size()<<'\n';
        const double enqueue_ms=std::chrono::duration<double,std::milli>(
            enqueue_end-wall_begin).count();
        const double wall_ms=std::chrono::duration<double,std::milli>(
            wall_end-wall_begin).count();
        // Accuracy observation is deliberately after activity.finish(), the
        // stop event and wall_end. Its D2H/decrypt work is not part of online
        // latency or CUPTI's zero-transfer interval.
        const auto decoded=s.replay.decoded_after_timing(logits);
        if(decoded.size()<fixture.reference_logits.size())
            throw std::runtime_error("post-timing logits are truncated");
        std::vector<double> gpu_logits(fixture.reference_logits.size());
        double max_logit_error=0,max_logit_imag=0;
        for(std::size_t index=0;index<gpu_logits.size();++index) {
            gpu_logits[index]=decoded[index].real();
            max_logit_error=std::max(max_logit_error,
                std::abs(gpu_logits[index]-fixture.reference_logits[index]));
            max_logit_imag=std::max(max_logit_imag,std::abs(decoded[index].imag()));
        }
        const auto argmax=[](const std::vector<double> &values) {
            return int(std::distance(values.begin(),
                std::max_element(values.begin(),values.end())));
        };
        const int gpu_prediction=argmax(gpu_logits);
        const int plain_prediction=argmax(fixture.reference_logits);
        const bool accuracy_pass=std::isfinite(max_logit_error) &&
            max_logit_error<=0.1 && gpu_prediction==plain_prediction;
        std::cout<<"POST_TIMING_LOGITS";
        for(double value:gpu_logits)std::cout<<' '<<value;
        std::cout<<'\n';
        std::cout<<"POST_TIMING_REFERENCE_LOGITS";
        for(double value:fixture.reference_logits)std::cout<<' '<<value;
        std::cout<<'\n';
        std::cout<<"POST_TIMING_ACCURACY result="<<(accuracy_pass?"PASS":"FAIL")
            <<" max_logit_error="<<max_logit_error
            <<" max_logit_imag="<<max_logit_imag
            <<" true_label="<<fixture.true_label
            <<" plain_prediction="<<plain_prediction
            <<" gpu_prediction="<<gpu_prediction<<'\n';
        std::cout<<"CONTINUOUS_TIMING_RESULT mode=single_resident_online_inference"
            <<" wall_ms="<<wall_ms<<" cuda_event_ms="<<event_ms
            <<" gpu_activity_union_ms="<<result.gpu_total_ms
            <<" host_enqueue_ms="<<enqueue_ms
            <<" activities="<<result.activities
            <<" host_transfers="<<result.host_transfers
            <<" blocks=9 bootstraps=18 dnum=2 input_q=32 stem_output_q=31"
            <<" application_keyswitch=fixed_dnum2 hoisted_rotation=true"
            <<" input_upload_included=false public_material_upload_included=false"
            <<" per_stage_sync=false observer_decryptions=0"
            <<" post_timing_decryptions=1 accuracy="<<(accuracy_pass?"PASS":"FAIL")
            <<" max_logit_error="<<max_logit_error
            <<" intermediate_reencryptions=0 swaps=false"
            <<" encrypted_logits_q="<<logits.meta.q_count
            <<" security_approved=false\n";
        return accuracy_pass?0:1;
    }
};
#endif

class Network {
    const BaselineSession &s;
    const ReluFixture &relu;
    const ApplicationNetworkFixture &fixture;
    double maximum_error=0;
    std::string first_failure;
    int bootstraps=0,convolutions=0,residuals=0,input_encryptions=0;
    BootstrapOfflineCache bootstrap_cache;

    CT conv(CT input,const Values &input_reference,const ApplicationLayerFixture &layer,
        int k,const Values &expected,const std::string &label) {
        const auto plan=trace_conv(s.metadata,input_reference,layer,k,expected);
        auto output=run_direct(s,plan,label,[&](const Runtime &runtime) {
            return apply_conv(runtime,tensor_from_device(s.metadata,std::move(input),
                layer.input_h,layer.input_w,layer.input_channels,k,*runtime.keys,s.eval),layer);
        });
        ++convolutions;
        check_boundary(s,label,*output.packs[0].gpu,expected,maximum_error,first_failure);
        return std::move(*output.packs[0].gpu);
    }

public:
    Network(const BaselineSession &session,const ReluFixture &r,
        const ApplicationNetworkFixture &f):s(session),relu(r),fixture(f) {}

    int run(std::size_t max_blocks) {
        if(max_blocks>fixture.blocks.size())throw std::runtime_error("max-blocks exceeds 9");
        std::cout<<"NETWORK_PLAN topology=resnet20 blocks="<<max_blocks
            <<" full_blocks=9 expected_bootstraps="<<2*max_blocks
            <<" input_q=32 stem_output_q=31"
            <<" encrypted_stem=true intermediate_reencryption=false observer_decryption_only=true\n";

        GpuGaloisKeysData unused;
        Runtime stem_runtime(s.metadata,&s.eval,&unused,[&](const std::vector<double> &slots) {
            ++input_encryptions;return s.encrypt_application_input(slots);
        });
        auto stem=app::encrypted_stem_conv2d_bn(fixture.image,32,32,3,16,1,3,3,
            fixture.stem.weights,fixture.stem.bn_scale,fixture.stem.bn_bias,stem_runtime);
        gpu_check_cuda(cudaDeviceSynchronize(),"finish encrypted ResNet20 stem");
        if(stem.packs.size()!=1 || stem.packs[0].meta.q_count!=31 ||
            std::abs(std::log2(stem.packs[0].meta.scale)-40)>1e-7 || input_encryptions!=27)
            throw std::runtime_error("encrypted stem contract mismatch");
        auto stem_reference=pack_chw(fixture.stem_output,32,32,16,1);
        check_boundary(s,"stem.conv1_bn1",*stem.packs[0].gpu,stem_reference,
            maximum_error,first_failure,2e-4);
        CT stem_q31=std::move(*stem.packs[0].gpu);
        auto active=relu_existing_q31(s,relu,stem_q31,stem_reference,"stem");
        active.reference=pack_chw(fixture.stem_relu_output,32,32,16,1);
        check_boundary(s,"stem.relu",active.ciphertext,active.reference,
            maximum_error,first_failure,2e-3);
        int h=32,w=32,c=16,k=1;

        for(std::size_t index=0;index<max_blocks;++index) {
            const auto &block=fixture.blocks[index];
            const std::string prefix="layer"+std::to_string(block.stage)+"."+std::to_string(block.block);
            CT shortcut;s.eval.drop_modulus(active.ciphertext,shortcut,active.ciphertext.meta.parms_id);
            auto shortcut_reference=active.reference;
            auto first=conv(std::move(active.ciphertext),active.reference,block.conv1,k,
                pack_chw(block.conv1_output,h/block.conv1.stride,w/block.conv1.stride,
                    block.conv1.output_channels,k*block.conv1.stride),prefix+".conv1_bn1");
            ++bootstraps;
            auto first_active=refresh_relu(s,relu,first,
                pack_chw(block.conv1_output,h/block.conv1.stride,w/block.conv1.stride,
                    block.conv1.output_channels,k*block.conv1.stride),prefix+".act1",bootstrap_cache);
            const int next_h=h/block.conv1.stride,next_w=w/block.conv1.stride;
            const int next_k=k*block.conv1.stride;
            auto second=conv(std::move(first_active.ciphertext),first_active.reference,
                block.conv2,next_k,pack_chw(block.conv2_output,next_h,next_w,
                    block.conv2.output_channels,next_k),prefix+".conv2_bn2");

            if(block.option_a_shortcut) {
                const auto expected=pack_chw(block.shortcut_output,next_h,next_w,
                    block.conv2.output_channels,next_k);
                Runtime cpu(s.metadata);auto host=app::downsample_shortcut(
                    tensor_from_host(s.metadata,shortcut_reference,h,w,c,k),cpu);
                if(compare_values("network/shortcut_SIMD_vs_CHW_oracle",*host.packs[0].values,expected)>1e-10)
                    throw std::runtime_error("shortcut oracle mismatch");
                DirectPlan plan{cpu.rotations,cpu.counts};
                auto downsampled=run_direct(s,plan,prefix+".shortcut",[&](const Runtime &runtime) {
                    return app::downsample_shortcut(tensor_from_device(s.metadata,std::move(shortcut),
                        h,w,c,k,*runtime.keys,s.eval),runtime);
                });
                shortcut=std::move(*downsampled.packs[0].gpu);shortcut_reference=expected;
                check_boundary(s,prefix+".shortcut",shortcut,expected,maximum_error,first_failure,2e-2);
            }
            GpuGaloisKeysData add_keys;Runtime add_runtime(s.metadata,&s.eval,&add_keys);
            auto left=tensor_from_device(s.metadata,std::move(second),next_h,next_w,
                block.conv2.output_channels,next_k,add_keys,s.eval);
            auto right=tensor_from_device(s.metadata,std::move(shortcut),next_h,next_w,
                block.conv2.output_channels,next_k,add_keys,s.eval);
            auto sum=app::residual_add(left,right,add_runtime);++residuals;
            const auto residual_reference=pack_chw(block.residual_output,next_h,next_w,
                block.conv2.output_channels,next_k);
            check_boundary(s,prefix+".add",*sum.packs[0].gpu,residual_reference,
                maximum_error,first_failure,2e-2);
            ++bootstraps;
            active=refresh_relu(s,relu,*sum.packs[0].gpu,residual_reference,prefix+".act2",bootstrap_cache);
            active.reference=pack_chw(block.output,next_h,next_w,
                block.conv2.output_channels,next_k);
            check_boundary(s,prefix+".output",active.ciphertext,active.reference,
                maximum_error,first_failure,block.stage==1?2e-2:1.0);
            h=next_h;w=next_w;c=block.conv2.output_channels;k=next_k;
            std::size_t free_bytes=0,total_bytes=0;gpu_check_cuda(cudaMemGetInfo(&free_bytes,&total_bytes),"network memory checkpoint");
            std::cout<<"NETWORK_BLOCK_RESULT block="<<prefix<<" completed="<<index+1
                <<" free_GiB="<<double(free_bytes)/(1ULL<<30)<<" q=9 log_scale=40\n";
        }

        if(max_blocks<fixture.blocks.size()) {
            const bool pass=first_failure.empty();
            std::cout<<"NETWORK_RESULT integration="<<(pass?"PASS":"FAIL")
                <<" staged=true completed_blocks="<<max_blocks<<" bootstraps="<<bootstraps
                <<" convolutions="<<convolutions<<" residual_adds="<<residuals
                <<" input_encryptions="<<input_encryptions<<" intermediate_reencryptions=0"
                <<" max_boundary_error="<<maximum_error
                <<" first_failure="<<(first_failure.empty()?"none":first_failure)<<'\n';
            return pass?0:1;
        }

        Runtime cpu(s.metadata);auto host_tensor=tensor_from_host(s.metadata,active.reference,h,w,c,k);
        auto host_features=app::global_average_pool(host_tensor,40,cpu);
        auto host_logits=app::fully_connected(host_features,64,fixture.linear_weight,
            fixture.linear_bias,10,cpu);
        Values expected_logits(32768);for(int i=0;i<10;++i)expected_logits[i]=fixture.reference_logits[i];
        if(compare_values("network/head_SIMD_vs_CHW_oracle",*host_logits.values,expected_logits)>1e-10)
            throw std::runtime_error("head oracle mismatch");
        DirectPlan head_plan{cpu.rotations,cpu.counts};
        auto encrypted_logits=run_direct(s,head_plan,"head",[&](const Runtime &runtime) {
            auto tensor=tensor_from_device(s.metadata,std::move(active.ciphertext),h,w,c,k,*runtime.keys,s.eval);
            auto features=app::global_average_pool(tensor,40,runtime);
            return app::fully_connected(features,64,fixture.linear_weight,fixture.linear_bias,10,runtime);
        });
        const auto decoded=decoded_values(s.replay,*encrypted_logits.gpu);
        std::vector<double> logits(10);double logit_error=0;
        for(int i=0;i<10;++i) {
            logits[i]=double(decoded[i].real());
            logit_error=std::max(logit_error,std::abs(logits[i]-fixture.reference_logits[i]));
        }
        const auto argmax=[](const std::vector<double> &v) {
            return int(std::distance(v.begin(),std::max_element(v.begin(),v.end())));
        };
        const int gpu_prediction=argmax(logits),plain_prediction=argmax(fixture.reference_logits);
        const bool pass=first_failure.empty() && std::isfinite(logit_error) &&
            logit_error<=0.1 && gpu_prediction==plain_prediction;
        std::cout<<"NETWORK_LOGITS";for(double value:logits)std::cout<<' '<<value;std::cout<<'\n';
        std::cout<<"NETWORK_REFERENCE_LOGITS";for(double value:fixture.reference_logits)std::cout<<' '<<value;std::cout<<'\n';
        std::cout<<"NETWORK_RESULT integration="<<(pass?"PASS":"FAIL")
            <<" staged=false completed_blocks=9 bootstraps="<<bootstraps
            <<" convolutions="<<convolutions<<" residual_adds="<<residuals
            <<" input_encryptions="<<input_encryptions<<" intermediate_reencryptions=0"
            <<" true_label="<<fixture.true_label<<" plain_prediction="<<plain_prediction
            <<" gpu_prediction="<<gpu_prediction<<" max_logit_error="<<logit_error
            <<" max_boundary_error="<<maximum_error
            <<" first_failure="<<(first_failure.empty()?"none":first_failure)
            <<" full_network_tested=true security_approved=false timing_benchmark=false\n";
        return pass?0:1;
    }
};
} // namespace network_probe

int run_s2c_first_resnet20_application(int argc,char **argv) {
    try {
        const std::string mode=argc>1?argv[1]:"";
        bool valid_mode=mode=="--metadata-only" || mode=="--unsafe-accuracy-only";
#ifdef POSEIDON_S2C_FIRST_ENABLE_TIMING
        valid_mode=valid_mode || mode=="--unsafe-performance-only";
#endif
        if(argc<4 || argc>6 || !valid_mode)
            throw std::runtime_error("usage: bootstrap_network_precision --metadata-only|--unsafe-accuracy-only|--unsafe-performance-only fixture.txt image_id [--max-blocks N]");
#ifndef POSEIDON_S2C_FIRST_ENABLE_TIMING
        if(mode=="--unsafe-performance-only")
            throw std::runtime_error("performance mode is available only in the dedicated bench entry");
#endif
        std::size_t parsed=0;const int image=std::stoi(argv[3],&parsed);
        if(parsed!=std::string(argv[3]).size() || image<0 || image>=10000)
            throw std::runtime_error("invalid CIFAR image id");
        std::size_t max_blocks=9;
        if(argc>4) {
            if(std::string(argv[4])!="--max-blocks" || argc!=6)
                throw std::runtime_error("expected --max-blocks N");
            parsed=0;max_blocks=std::stoul(argv[5],&parsed);
            if(parsed!=std::string(argv[5]).size() || max_blocks>9)
                throw std::runtime_error("max-blocks must be 0..9");
        }
        if(mode=="--unsafe-performance-only" && max_blocks!=9)
            throw std::runtime_error("performance mode always runs the complete 9-block network");
        ReluFixture relu(argv[2]);auto fixture=load_application_network_fixture(image);
        std::string profile="q50-fixed45",dataset="grid";
        char *args[]={argv[0],argv[1],profile.data(),dataset.data(),argv[2]};
        return run_bootstrap_baseline(5,args,[&](const BaselineMetadata &m) {
            check_continuation_metadata(m,relu);
            if(fixture.blocks.size()!=9 || fixture.stem_output.size()!=16384 ||
                fixture.reference_logits.size()!=10)
                throw std::runtime_error("ResNet20 fixture is incomplete");
            std::cout<<"NETWORK_METADATA topology_blocks=9 bootstrap_points=18"
                <<" input_q=32 stem_output_q=31"
                <<" oracle=independent_original_CHW input_boundary=40 status=PASS\n";
        },{},nullptr,true,[&](const BaselineSession &session) {
#ifdef POSEIDON_S2C_FIRST_ENABLE_TIMING
            if(mode=="--unsafe-performance-only")
                return network_probe::ContinuousNetwork(session,relu,fixture).run();
#endif
            return network_probe::Network(session,relu,fixture).run(max_blocks);
        });
    } catch(const std::exception &e) {std::cerr<<"ERROR "<<e.what()<<'\n';return 2;}
}

#ifndef POSEIDON_S2C_FIRST_RESNET20_LIBRARY
int main(int argc,char **argv) {
    return run_s2c_first_resnet20_application(argc,argv);
}
#endif
