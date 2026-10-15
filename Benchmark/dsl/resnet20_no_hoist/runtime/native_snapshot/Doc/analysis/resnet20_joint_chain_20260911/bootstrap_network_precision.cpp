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
#include <complex>
#ifdef POSEIDON_S2C_FIRST_ENABLE_TIMING
#include "../../../bench/resnet20_s2c_first/gpu_activity_timing.h"
#endif
#ifdef POSEIDON_S2C_FIRST_DSL_RUNNER
#include "poseidon/frontends/dacapo/hevm_static_execution_plan.h"
#include "poseidon/frontends/dacapo/poseidon_gpu_hevm_executor.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
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
        current=s.replay.component(fixture.stages[i],std::move(current),real_reference,i+1,stage_error);
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
        current=replay.component(fixture.stages[index],std::move(current),unused,index+1,stage_error);
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

void report_eval_mod_plain_error(Replay &replay,const char *branch,
    std::size_t call,const Values &input,const CT &actual,const FoldSpec &fold,
    const EvalModPoly &poly) {
    const auto actual_values=decoded_values(replay,actual);
    if(actual_values.size()!=input.size())
        throw std::runtime_error("EvalMod diagnostic slot count changed");
    double max_error=0,max_expected=0,max_actual=0;
    long double squared_error=0;
    for(std::size_t i=0;i<input.size();++i) {
        const Complex input_value{input[i].real(),input[i].imag()};
        const auto expected=fold.reference(poly,input_value);
        const Complex actual_value{actual_values[i].real(),actual_values[i].imag()};
        const double error=static_cast<double>(std::abs(actual_value-expected));
        max_error=std::max(max_error,error);
        max_expected=std::max(max_expected,static_cast<double>(std::abs(expected)));
        max_actual=std::max(max_actual,static_cast<double>(std::abs(actual_value)));
        squared_error+=static_cast<long double>(error)*error;
    }
    std::cout<<"DSL_EVALMOD_PLAIN_CHECK call="<<call<<" branch="<<branch
        <<" max_abs_error="<<max_error
        <<" rms_error="<<std::sqrt(static_cast<double>(squared_error/input.size()))
        <<" expected_max_abs="<<max_expected<<" actual_max_abs="<<max_actual<<'\n';
}

class ContinuousBootstrap {
    static constexpr std::size_t default_c2s_baby_tile=8;
    const BaselineSession &s;
    GpuGaloisKeysData rotations;
    BootstrapOfflineCache cpu;
    GpuLinearMatrixGroupQP stc,cts;
    GpuLinearMatrixGroup stc_q,cts_q;
    GpuBootstrapData data;
    GpuPlaintextData minus_i,plus_i;
    GpuDoubleHoistWorkspace stc_workspace,cts_workspace;
    GpuBootstrapWorkspace evalmod_workspace;
    double alpha=1;
    std::size_t diagnostic_calls=0;
public:
    explicit ContinuousBootstrap(const BaselineSession &session):s(session) {
        // The Q50/P25 C2S matrices have eight baby steps per stage.  Keeping
        // all eight in one tile removes the second Q/P accumulator pass.
        // Only stage 1 has <=4 giant groups and uses the fused KeySwitch/MAC;
        // stages 2/3 have eight groups and retain separate KeySwitch and MAC.
        // QP plaintext compression leaves ample
        // resident-pool headroom for the larger temporary tile.  The generic
        // library default remains unchanged; an explicit environment setting
        // can still override this application-local choice for A/B testing.
        cts_workspace.baby_tile_size=default_c2s_baby_tile;
        CT metadata_input;metadata_input.meta.q_count=6;
        metadata_input.meta.scale=std::exp2(40);
        prepare_bootstrap_cpu(s.metadata,metadata_input,cpu);
        const bool use_no_hoist_keys=
            gpu_linear_transform_mode_from_environment(
                GpuLinearTransformMode::DoubleHoistBsgs)==
            GpuLinearTransformMode::NoHoistBsgs;
        rotations=use_no_hoist_keys
            ? s.rotation_keys(cpu.rotations)
            : s.bootstrap_rotation_keys(cpu.rotations);
        const auto nq=s.metadata.ctx.parameters_literal()->q().size();
        GpuUploader::prepare_key_views_for_q_counts(
            rotations,{4,5,6,nq-4,nq-2,nq-1,nq,s.metadata.native_plan.output_q_count});
        // DFT diagonals have exact power-of-two periods in bit-reversed NTT
        // order.  Keep only one period per QP limb; the uploader verifies the
        // reconstructed Q and P residues before this public material can be
        // used by the online path.
        stc=GpuUploader::upload_linear_matrix_group_qp(
            *cpu.stc_cpu,s.metadata.ctx,0,1,true);
        cts=GpuUploader::upload_linear_matrix_group_qp(
            *cpu.cts_cpu,s.metadata.ctx,0,1,true);
        stc_q=GpuUploader::upload_linear_matrix_group(*cpu.stc_cpu,0);
        cts_q=GpuUploader::upload_linear_matrix_group(*cpu.cts_cpu,0);
        const auto stc_compression=compressed_qp_stats(stc);
        const auto cts_compression=compressed_qp_stats(cts);
        if(stc_compression.diagonals!=matrix_plaintext_count(*cpu.stc_cpu) ||
            cts_compression.diagonals!=matrix_plaintext_count(*cpu.cts_cpu) ||
            !stc_compression.exact || !cts_compression.exact)
            throw std::logic_error(
                "continuous bootstrap QP plaintext compression contract failed");
        const auto full_words=stc_compression.full_words+cts_compression.full_words;
        const auto compact_words=
            stc_compression.compact_words+cts_compression.compact_words;
        if(!compact_words || compact_words>=full_words)
            throw std::logic_error(
                "continuous bootstrap QP plaintexts did not compress");
        std::cout<<"BOOTSTRAP_QP_COMPRESSION enabled=true exact=true"
            <<" stc_diagonals="<<stc_compression.diagonals
            <<" cts_diagonals="<<cts_compression.diagonals
            <<" full_MiB="<<double(full_words*sizeof(GpuWord))/(1ULL<<20)
            <<" compact_MiB="<<double(compact_words*sizeof(GpuWord))/(1ULL<<20)
            <<" ratio="<<double(full_words)/compact_words<<'\n';
        const bool fixed_degree_refit=
            std::getenv("POSEIDON_DSL_EVALMOD_FIXED_DEGREE_REFIT")!=nullptr;
        FoldSpec fold(s.metadata,fixed_degree_refit);
        if(fixed_degree_refit) {
            std::cout<<"DSL_EVALMOD_REFIT enabled=true degree="<<fold.coefficients.degree()
                <<" method=fixed_degree_chebyshev_interpolation same_degree=true\n";
        }
        alpha=fold.alpha;data.eval_mod=fold.upload(s.metadata,&s.keys);
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
        const bool use_no_hoist=
            gpu_linear_transform_mode_from_environment(
                GpuLinearTransformMode::DoubleHoistBsgs)==
            GpuLinearTransformMode::NoHoistBsgs;
        const double working_scale=std::exp2(45);
        const double preparation_target=std::exp2(
            std::round(std::log2(ctx.crt_context()->q0()/32)));
        if(encrypted_input.meta.q_count!=6)
            throw std::runtime_error("continuous bootstrap input must be Q6");
        const bool accuracy_diag=
            std::getenv("POSEIDON_DSL_BOOTSTRAP_ACCURACY_DIAG")!=nullptr;
        Values input_values;
        if(accuracy_diag)input_values=decoded_values(s.replay,encrypted_input);
        CT stc_out,prepared,raised,real,imag,eval_real,eval_imag;
        run_activity(timing,"bootstrap.S2C",[&]{
            if(use_no_hoist)
                eval.dft_loose(encrypted_input,stc_q,rotations,stc_out);
            else
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
            if(use_no_hoist)
                eval.coeff_to_slot_loose(
                    raised,cts_q,minus_i,rotations,real,imag);
            else
                eval.coeff_to_slot_double_hoist(raised,cts,minus_i,rotations,
                    cts_workspace,real,imag);
        });
        Values real_values,imag_values;
        if(accuracy_diag) {
            real_values=decoded_values(s.replay,real);
            imag_values=decoded_values(s.replay,imag);
            double real_domain=0,imag_domain=0;
            std::size_t real_outside=0,imag_outside=0;
            for(std::size_t i=0;i<real_values.size();++i) {
                const double real_abs=std::abs(real_values[i]);
                const double imag_abs=std::abs(imag_values[i]);
                real_domain=std::max(real_domain,real_abs);
                imag_domain=std::max(imag_domain,imag_abs);
                real_outside+=real_abs>1.0;
                imag_outside+=imag_abs>1.0;
            }
            std::cout<<"DSL_BOOTSTRAP_C2S_DOMAIN call="<<++diagnostic_calls
                <<" real_max_abs="<<real_domain
                <<" real_outside_unit="<<real_outside
                <<" imag_max_abs="<<imag_domain
                <<" imag_outside_unit="<<imag_outside<<'\n';
        }
        run_activity(timing,"bootstrap.EvalMod",[&]{
            eval.eval_mod_high_precision(real,data,s.keys,evalmod_workspace,eval_real);
        });
        run_activity(timing,"bootstrap.EvalMod",[&]{
            eval.eval_mod_high_precision(imag,data,s.keys,evalmod_workspace,eval_imag);
        });
        if(accuracy_diag) {
            const FoldSpec fold(s.metadata);
            report_eval_mod_plain_error(s.replay,"real",diagnostic_calls,
                real_values,eval_real,fold,s.metadata.poly);
            report_eval_mod_plain_error(s.replay,"imag",diagnostic_calls,
                imag_values,eval_imag,fold,s.metadata.poly);
        }
        CT scaled_imag,combined,half,conjugated,output;
        run_activity(timing,"bootstrap.recombine",[&]{
            eval.multiply_plain(eval_imag,plus_i,scaled_imag);
            eval.add(eval_real,scaled_imag,combined);
            eval.multiply_scalar(combined,16,half);
            eval.conjugate(half,rotations,conjugated);
            eval.add(half,conjugated,output);
        });
        output.meta.scale*=alpha;
        if(accuracy_diag) {
            const FoldSpec fold(s.metadata);
            const auto output_values=decoded_values(s.replay,output);
            if(output_values.size()!=input_values.size() ||
                real_values.size()!=input_values.size() ||
                imag_values.size()!=input_values.size())
                throw std::runtime_error("bootstrap recombination diagnostic slot count changed");
            double max_formula_error=0,max_formula_input_error=0;
            long double formula_squared=0,input_squared=0;
            double residual_bin_max[5]={},residual_bin_squared[5]={};
            std::size_t residual_bin_count[5]={},max_residual_slot=0;
            for(std::size_t i=0;i<input_values.size();++i) {
                const auto expected_real=fold.reference(s.metadata.poly,real_values[i]);
                const auto expected_imag=fold.reference(s.metadata.poly,imag_values[i]);
                const long double expected_output=
                    32.0L*(expected_real+Complex(0,1)*expected_imag).real()/alpha;
                const long double formula_error=std::abs(
                    output_values[i]-Complex(expected_output,0));
                const long double input_error=std::abs(
                    input_values[i]-Complex(expected_output,0));
                max_formula_error=std::max(max_formula_error,
                    static_cast<double>(formula_error));
                if(input_error>max_formula_input_error) {
                    max_formula_input_error=static_cast<double>(input_error);
                    max_residual_slot=i;
                }
                const auto input_abs=std::abs(input_values[i]);
                const auto bin=std::min<std::size_t>(4,static_cast<std::size_t>(input_abs*5));
                residual_bin_count[bin]++;
                residual_bin_max[bin]=std::max(residual_bin_max[bin],
                    static_cast<double>(input_error));
                residual_bin_squared[bin]+=static_cast<double>(input_error*input_error);
                formula_squared+=formula_error*formula_error;
                input_squared+=input_error*input_error;
            }
            std::cout<<"DSL_BOOTSTRAP_RECOMBINE_CHECK call="<<diagnostic_calls
                <<" output_vs_fold_formula_max="<<max_formula_error
                <<" output_vs_fold_formula_rms="<<std::sqrt(static_cast<double>(
                    formula_squared/input_values.size()))
                <<" fold_formula_vs_input_max="<<max_formula_input_error
                <<" fold_formula_vs_input_rms="<<std::sqrt(static_cast<double>(
                    input_squared/input_values.size()))<<'\n';
            const auto &peak_real=real_values[max_residual_slot];
            const auto &peak_imag=imag_values[max_residual_slot];
            const Complex peak_formula=32.0L*(
                fold.reference(s.metadata.poly,peak_real)+
                Complex(0,1)*fold.reference(s.metadata.poly,peak_imag)).real()/alpha;
            std::cout<<"DSL_BOOTSTRAP_RESIDUAL_C2S call="<<diagnostic_calls
                <<" slot="<<max_residual_slot
                <<" real_input="<<peak_real<<" imag_input="<<peak_imag
                <<" real_fold="<<fold.reference(s.metadata.poly,peak_real)
                <<" imag_fold="<<fold.reference(s.metadata.poly,peak_imag)<<'\n';
            std::cout<<"DSL_BOOTSTRAP_RESIDUAL_PEAK call="<<diagnostic_calls
                <<" slot="<<max_residual_slot<<" input="<<input_values[max_residual_slot]
                <<" fold_formula="<<peak_formula
                <<" error="<<std::abs(input_values[max_residual_slot]-peak_formula)<<'\n';
            for(std::size_t bin=0;bin<5;++bin)if(residual_bin_count[bin])
                std::cout<<"DSL_BOOTSTRAP_RESIDUAL_BUCKET call="<<diagnostic_calls
                    <<" input_abs_range="<<0.2*bin<<':'<<0.2*(bin+1)
                    <<" count="<<residual_bin_count[bin]
                    <<" max="<<residual_bin_max[bin]
                    <<" rms="<<std::sqrt(residual_bin_squared[bin]/residual_bin_count[bin])<<'\n';
        }
        s.replay.scale_check(output,40);
        if(output.meta.q_count!=31)
            throw std::runtime_error("continuous bootstrap output must be Q31");
        return output;
    }

    std::size_t stc_baby_tile_size() const {
        return stc_workspace.baby_tile_size;
    }
    std::size_t c2s_baby_tile_size() const {
        return cts_workspace.baby_tile_size;
    }
    std::size_t c2s_matrix_count() const {
        return cts_workspace.matrix_counts.size();
    }
    std::size_t c2s_tile_batches() const {
        std::size_t result=0;
        for(const auto &counts:cts_workspace.matrix_counts)
            result+=counts.baby_tile_count;
        return result;
    }

private:
    struct QpCompressionStats {
        std::size_t diagonals=0,full_words=0,compact_words=0;
        bool exact=true;
    };

    static QpCompressionStats compressed_qp_stats(
        const GpuLinearMatrixGroupQP &group) {
        QpCompressionStats result;
        for(const auto &matrix:group.data()) {
            if(!matrix.plain_vec_qp.empty() ||
                !matrix.plan.compressed_plaintexts ||
                matrix.plan.terms.size()!=matrix.compressed_plain_vec_qp.size() ||
                matrix.plan.diagonal_periods.size()!=matrix.plan.terms.size())
                throw std::logic_error(
                    "continuous bootstrap compressed QP matrix plan is inconsistent");
            for(const auto &[diagonal,plain]:matrix.compressed_plain_vec_qp) {
                (void)diagonal;
                if(!plain.meta.degree || !plain.meta.p_count || !plain.period ||
                    plain.period>plain.meta.degree ||
                    (plain.period&(plain.period-1)) ||
                    !plain.exact_device_reconstruction)
                    throw std::logic_error(
                        "continuous bootstrap compressed QP plaintext is not exact");
                ++result.diagonals;
                result.full_words+=plain.full_word_count();
                result.compact_words+=plain.compact_word_count();
                result.exact=result.exact&&plain.exact_device_reconstruction;
            }
        }
        return result;
    }

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
    bool conv_plain_batch=true;

    void report_relu_basis_fusion(bool preparation) const {
        const auto &stats=s.replay.get_moddrop_stats();
        const bool enabled=s.replay.basis_fusion_enabled();
        if(stats.fused_basis_calls!=(enabled?285:0) ||
            stats.fused_basis_constants!=(enabled?190:0) ||
            stats.fused_basis_products!=(enabled?95:0) ||
            stats.exact_basis_checks!=(enabled && preparation?285:0) ||
            stats.exact_basis_residues!=(enabled && preparation?824311808:0))
            throw std::runtime_error("continuous ReLU basis fusion inventory changed");
        std::cout<<(preparation?"RELU_BASIS_FUSION_PREPARED":"RELU_BASIS_FUSION")
            <<" enabled="<<(enabled?"true":"false")<<" calls="<<stats.fused_basis_calls
            <<" constants="<<stats.fused_basis_constants<<" products="<<stats.fused_basis_products
            <<" exact_checks="<<stats.exact_basis_checks<<" exact_residues="<<stats.exact_basis_residues
            <<" online_observer=false coefficients_changed=false rescale_changed=false\n";
    }

    void report_conv_plain_batch(bool preparation) const {
        Runtime::PlainBatchStats total;
        for(const auto &runtime:conv_runtimes) {
            const auto &stats=runtime->plain_batch_stats;
            total.chains+=stats.chains;total.terms+=stats.terms;total.calls+=stats.calls;
            total.exact_checks+=stats.exact_checks;total.exact_residues+=stats.exact_residues;
            total.pending_terms+=stats.pending_terms;
        }
        if(total.chains!=(conv_plain_batch?160:0) ||
            total.terms!=(conv_plain_batch?1440:0) ||
            total.calls!=(conv_plain_batch?480:0) || total.pending_terms ||
            total.exact_checks!=(conv_plain_batch && preparation?480:0) ||
            total.exact_residues!=(conv_plain_batch && preparation?566231040:0))
            throw std::runtime_error("continuous Conv plain batch inventory changed");
        std::cout<<(preparation?"CONV_PLAIN_BATCH_PREPARED":"CONV_PLAIN_BATCH")
            <<" enabled="<<(conv_plain_batch?"true":"false")
            <<" chains="<<total.chains<<" terms="<<total.terms<<" calls="<<total.calls
            <<" exact_checks="<<total.exact_checks<<" exact_residues="<<total.exact_residues
            <<" pending="<<total.pending_terms
            <<" online_observer=false coefficients_changed=false rescale_changed=false\n";
    }

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
        if(const char *raw=std::getenv("POSEIDON_CONV_PLAIN_BATCH")) {
            const std::string value(raw);
            if(value!="0" && value!="1")
                throw std::invalid_argument("POSEIDON_CONV_PLAIN_BATCH must be 0 or 1");
            conv_plain_batch=value=="1";
        }
        for(std::size_t i=0;i<plans.convolutions.size();++i) {
            conv_runtimes.push_back(std::make_unique<Runtime>(
                s.metadata,&s.eval,&no_keys,Runtime::InputEncryptor{},rotation_resolver));
            conv_runtimes.back()->plain_batch_enabled=conv_plain_batch;
        }
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
        s.replay.reset_moddrop_stats();
        s.replay.set_basis_preparation_checks(true);
        auto warm=execute_graph(true);
        s.replay.set_basis_preparation_checks(false);
        gpu_check_cuda(cudaDeviceSynchronize(),"continuous prepared warmup");
        warm=CT{};gpu_check_cuda(cudaDeviceSynchronize(),"continuous release warmup output");
        report_conv_plain_batch(true);
        report_relu_basis_fusion(true);
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
        const auto c2s_tile_batches=bootstrap->c2s_tile_batches();
        const char *tile_override=std::getenv("POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE");
        if((tile_override==nullptr || *tile_override=='\0') &&
            (bootstrap->c2s_baby_tile_size()!=8 ||
             bootstrap->c2s_matrix_count()!=3 ||
             c2s_tile_batches!=3))
            throw std::runtime_error(
                "continuous default C2S full-baby tile contract changed");
        std::cout<<"CONTINUOUS_PREPARED inputs=27 stem_plaintexts=28"
            <<" conv_plaintexts="<<conv_plaintexts
            <<" shortcut_plaintexts="<<shortcut_plaintexts
            <<" relu_plaintexts="<<s.replay.plaintext_cache_size()
            <<" head_plaintexts="<<head_runtime->prepared_plaintext_count()
            <<" bootstrap_s2c_baby_tile="<<bootstrap->stc_baby_tile_size()
            <<" bootstrap_c2s_baby_tile="<<bootstrap->c2s_baby_tile_size()
            <<" bootstrap_c2s_tile_batches="<<c2s_tile_batches
            <<" free_MiB="<<double(free_bytes)/(1ULL<<20)<<'\n';

        poseidon::benchmark::s2c_first::GpuActivityTiming activity;
        cudaEvent_t start_event=nullptr,stop_event=nullptr;
        gpu_check_cuda(cudaEventCreate(&start_event),"continuous start event");
        gpu_check_cuda(cudaEventCreate(&stop_event),"continuous stop event");
        struct EventCleanup {cudaEvent_t &a,&b;~EventCleanup(){
            if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);}} cleanup{start_event,stop_event};
        gpu_check_cuda(cudaDeviceSynchronize(),"continuous timing start sync");
        s.replay.reset_moddrop_stats();
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
        report_conv_plain_batch(false);
        report_relu_basis_fusion(false);
        const auto &moddrop_stats=s.replay.get_moddrop_stats();
        const bool leaf_fusion=s.replay.leaf_fusion_enabled();
        const bool basis_fusion=s.replay.basis_fusion_enabled();
        if(moddrop_stats.fused_leaf_calls!=(leaf_fusion?266:0) ||
            moddrop_stats.fused_leaf_terms!=(leaf_fusion?570:0) ||
            moddrop_stats.fused_leaf_adds_removed!=(leaf_fusion?304:0) ||
            moddrop_stats.exact_leaf_checks!=0 || moddrop_stats.exact_leaf_residues!=0)
            throw std::runtime_error("continuous ReLU fused leaf inventory changed");
        if(s.replay.zero_copy_moddrop_enabled() && moddrop_stats.inplace_calls!=494)
            throw std::runtime_error("continuous ReLU zero-copy ModDrop inventory changed");
        if(s.replay.q_prefix_views_enabled() &&
            (moddrop_stats.prefix_multiply_calls!=513 ||
             moddrop_stats.prefix_multiply_plain_calls!=
                (leaf_fusion?95:665)-(basis_fusion?95:0) ||
             moddrop_stats.prefix_source_views!=1463 ||
             moddrop_stats.prefix_discarded_q_limbs!=1159))
            throw std::runtime_error("continuous ReLU Q-prefix view inventory changed");
        if(s.replay.zero_copy_moddrop_enabled() &&
            s.replay.q_prefix_views_enabled() &&
            moddrop_stats.materialized_calls!=19)
            throw std::runtime_error("continuous ReLU materialized ModDrop inventory changed");
        std::cout<<"RELU_MODDROP zero_copy="
            <<(s.replay.zero_copy_moddrop_enabled()?"true":"false")
            <<" q_prefix_views="
            <<(s.replay.q_prefix_views_enabled()?"true":"false")
            <<" materialized_calls="<<moddrop_stats.materialized_calls
            <<" inplace_calls="<<moddrop_stats.inplace_calls
            <<" inplace_discarded_q_limbs="
            <<moddrop_stats.inplace_discarded_q_limbs
            <<" prefix_multiply_calls="<<moddrop_stats.prefix_multiply_calls
            <<" prefix_multiply_plain_calls="
            <<moddrop_stats.prefix_multiply_plain_calls
            <<" prefix_source_views="<<moddrop_stats.prefix_source_views
            <<" prefix_discarded_q_limbs="
            <<moddrop_stats.prefix_discarded_q_limbs<<'\n';
        std::cout<<"RELU_LEAF_FUSION enabled="<<(leaf_fusion?"true":"false")
            <<" calls="<<moddrop_stats.fused_leaf_calls
            <<" terms="<<moddrop_stats.fused_leaf_terms
            <<" adds_removed="<<moddrop_stats.fused_leaf_adds_removed
            <<" exact_checks="<<moddrop_stats.exact_leaf_checks
            <<" online_observer=false coefficients_changed=false rescale_changed=false\n";
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
            <<" compressed_qp_plaintexts=true"
            <<" bootstrap_c2s_baby_tile="<<bootstrap->c2s_baby_tile_size()
            <<" relu_zero_copy_moddrop="
            <<(s.replay.zero_copy_moddrop_enabled()?"true":"false")
            <<" relu_q_prefix_views="
            <<(s.replay.q_prefix_views_enabled()?"true":"false")
            <<" relu_leaf_fusion="<<(s.replay.leaf_fusion_enabled()?"true":"false")
            <<" conv_plain_batch="<<(conv_plain_batch?"true":"false")
            <<" relu_basis_fusion="<<(basis_fusion?"true":"false")
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
#ifdef POSEIDON_S2C_FIRST_DSL_RUNNER
// ---------------------------------------------------------------------------
// Dacapo/Hecate HEVM replay runner.
//
// The reverse-engineered ResNet20 program is loaded from the traced .hevm
// bytecode plus its double-precision constant table, executed on the GPU with
// the same no-hoist bootstrap used by the hand-written application, and the
// packed output is decoded and compared against the PyTorch reference logits.
// ---------------------------------------------------------------------------
std::vector<double> read_double_file(const std::string &path,std::size_t expected) {
    std::ifstream in(path);
    if(!in)throw std::runtime_error("cannot open "+path);
    std::vector<double> values;
    double value=0;
    while(in>>value)values.push_back(value);
    if(values.size()!=expected)
        throw std::runtime_error("unexpected value count in "+path);
    return values;
}

std::vector<int> dsl_rotation_steps(const poseidon::mgpu::MgpuSchedule &schedule) {
    std::vector<int> steps;
    for(const auto &op:schedule.ops) {
        if(op.kind!=poseidon::mgpu::MgpuOpKind::Rotate)continue;
        const auto attr=op.integer_attributes.find("rotate_step");
        if(attr==op.integer_attributes.end())
            throw std::runtime_error("HEVM rotate op without rotate_step attribute");
        steps.push_back(static_cast<int>(attr->second));
    }
    std::sort(steps.begin(),steps.end());
    steps.erase(std::unique(steps.begin(),steps.end()),steps.end());
    return steps;
}

int run_dsl_plan(const BaselineSession &s,const std::string &hevm_path,
    const std::string &constants_path,const std::string &slots_path,
    const std::string &reference_path) {
    using namespace poseidon::mgpu;
    const auto prepared=prepare_hevm_static_execution_plan_from_files(
        s.metadata.ctx,DacapoHevmArtifactPaths{hevm_path,constants_path});
    if(!prepared.ok())
        throw std::runtime_error("HEVM plan failed:\n"+prepared.format_diagnostics());
    // Own a mutable copy: the scale compensation below rewrites the encoded
    // constants in place (the heavy payloads are shared_ptr, so this is cheap).
    HevmStaticExecutionPlan plan=prepared.plan;
    if(plan.io_plan.cipher_inputs.size()!=1)
        throw std::runtime_error("DSL runner expects exactly one cipher input");
    const auto &input_slot=plan.io_plan.cipher_inputs.front();
    const std::size_t nq=s.metadata.ctx.parameters_literal()->q().size();
    std::cout<<std::unitbuf<<std::setprecision(16)
        <<"DSL_PLAN ops="<<plan.schedule.ops.size()
        <<" plaintexts="<<plan.encoded_plaintexts.size()
        <<" results="<<plan.io_plan.results.size()<<" nq="<<nq
        <<" input_level="<<input_slot.level<<" input_init_level="<<input_slot.init_level
        <<" input_log_scale="<<input_slot.scale<<" input_q_count="<<(input_slot.level+1)
        <<'\n';
    // CrtContext keys parms_id_map() with (q_count - 1), so the HEVM level is
    // the remaining-limb index and the input needs level + 1 primes.
    std::size_t q_count=static_cast<std::size_t>(input_slot.level)+1;
    if(q_count<2 || q_count>nq)
        throw std::runtime_error("cannot derive DSL input q_count from HEVM level");
    double log_scale=static_cast<double>(input_slot.scale);
    if(const char *env=std::getenv("POSEIDON_DSL_INPUT_Q"))q_count=std::stoul(env);
    if(const char *env=std::getenv("POSEIDON_DSL_INPUT_LOG_SCALE"))log_scale=std::stod(env);
    const auto slots=read_double_file(slots_path,32768);
    Ciphertext input=s.encrypt_application_input_host(slots,q_count,std::exp2(log_scale));
    std::cout<<"DSL_INPUT q_count="<<input.coeff_modulus_size()
        <<" log_scale="<<std::log2(input.scale())<<'\n';
    // Diagnostic aid: POSEIDON_DSL_MAX_OPS truncates the program so a single
    // segment prefix can be replayed (and traced) without allocating the
    // bootstrap matrices at all.
    std::vector<MgpuOp> truncated;
    if(const char *env=std::getenv("POSEIDON_DSL_MAX_OPS")) {
        const std::size_t max_ops=std::stoul(env);
        if(max_ops>0 && max_ops<plan.schedule.ops.size()) {
            truncated.assign(plan.schedule.ops.begin(),
                plan.schedule.ops.begin()+static_cast<std::ptrdiff_t>(max_ops));
        }
    }
    const auto &ops=truncated.empty()?plan.schedule.ops:truncated;
    std::map<ValueId,std::size_t> defined_at,last_use;
    for(std::size_t i=0;i<ops.size();++i) {
        for(const auto &in:ops[i].inputs)
            last_use[in.id]=i;
        for(const auto &out:ops[i].outputs)
            defined_at[out.id]=i;
    }
    const bool clear_reference_diag=
        std::getenv("POSEIDON_DSL_CLEAR_REFERENCE_DIAG")!=nullptr;
    std::unordered_map<ValueId,const std::vector<double>*> clear_plain_messages;
    std::unordered_map<ValueId,Values> clear_values;
    if(clear_reference_diag) {
        for(const auto &entry:plan.encoded_plaintexts)
            if(entry.message)
                clear_plain_messages.emplace(entry.value_id,entry.message.get());
        Values initial_values;
        initial_values.reserve(slots.size());
        for(const double value:slots)
            initial_values.emplace_back(value,0.0L);
        clear_values.emplace(input_slot.value_id,std::move(initial_values));
    }
    auto evaluate_clear_op=[&](std::size_t index) {
        const MgpuOp &op=ops[index];
        if(op.outputs.empty() || op.kind==MgpuOpKind::Download)
            return;
        const ValueId output_id=op.outputs.front().id;
        auto input_values=[&](std::size_t position)->const Values& {
            if(op.inputs.size()<=position)
                throw std::runtime_error("clear DSL reference is missing an input");
            const auto found=clear_values.find(op.inputs[position].id);
            if(found==clear_values.end())
                throw std::runtime_error("clear DSL reference lost value %"+
                    std::to_string(op.inputs[position].id));
            return found->second;
        };
        if(clear_reference_diag && output_id==634 && op.kind==MgpuOpKind::Add &&
           op.inputs.size()>=2) {
            const Values &left=input_values(0);
            const Values &right=input_values(1);
            auto stats=[](const Values &values) {
                long double maximum=0.0L, squares=0.0L;
                for(const auto &value:values) {
                    const long double magnitude=std::abs(value);
                    maximum=std::max(maximum,magnitude);
                    squares+=magnitude*magnitude;
                }
                return std::pair<long double,long double>{maximum,
                    std::sqrt(squares/static_cast<long double>(values.size()))};
            };
            const auto left_stats=stats(left);
            const auto right_stats=stats(right);
            // h51's runtime metadata showed value628 at 72.0379796037154
            // bits and value633 at 72.01056706838125 bits. Model the actual
            // low-scale encoded-one coefficient rounding, without changing
            // the execution path.
            constexpr double gap_bits=0.02741253533415;
            const double encoded_scale=std::exp2(gap_bits);
            const double rounded_constant=std::round(encoded_scale);
            const double decoded_multiplier=rounded_constant/encoded_scale;
            const long double penalty=std::abs(1.0-decoded_multiplier);
            const long double predicted_max=penalty*right_stats.first;
            const long double predicted_rms=penalty*right_stats.second;
            std::cout<<"DSL_CLEAR_ADD out=634 in0="<<op.inputs[0].id
                <<" in1="<<op.inputs[1].id
                <<" in0_max="<<left_stats.first<<" in0_rms="<<left_stats.second
                <<" in1_max="<<right_stats.first<<" in1_rms="<<right_stats.second
                <<" low_id=633 gap_bits="<<gap_bits
                <<" encoded_scale="<<encoded_scale
                <<" rounded_constant="<<rounded_constant
                <<" decoded_multiplier="<<decoded_multiplier
                <<" predicted_max_error="<<predicted_max
                <<" predicted_rms_error="<<predicted_rms<<'\n';
        }
        Values output;
        switch(op.kind) {
        case MgpuOpKind::UploadCipher:
            if(clear_values.find(output_id)==clear_values.end())
                throw std::runtime_error("clear DSL reference has an unknown cipher input");
            break;
        case MgpuOpKind::UploadPlain: {
            const auto message=clear_plain_messages.find(output_id);
            if(message==clear_plain_messages.end())
                throw std::runtime_error("clear DSL reference is missing a plaintext message");
            output.assign(slots.size(),Complex{});
            if(message->second->size()>output.size())
                throw std::runtime_error("clear DSL plaintext exceeds the slot count");
            for(std::size_t i=0;i<message->second->size();++i)
                output[i]=Complex((*message->second)[i],0.0L);
            clear_values[output_id]=std::move(output);
            break;
        }
        case MgpuOpKind::Add:
        case MgpuOpKind::AddPlain:
        case MgpuOpKind::Sub:
        case MgpuOpKind::MultiplyPlain:
        case MgpuOpKind::Multiply: {
            const Values &left=input_values(0);
            const Values &right=input_values(1);
            if(left.size()!=right.size())
                throw std::runtime_error("clear DSL operands have different slot counts");
            output.resize(left.size());
            for(std::size_t i=0;i<left.size();++i) {
                if(op.kind==MgpuOpKind::Add || op.kind==MgpuOpKind::AddPlain)
                    output[i]=left[i]+right[i];
                else if(op.kind==MgpuOpKind::Sub)
                    output[i]=left[i]-right[i];
                else
                    output[i]=left[i]*right[i];
            }
            clear_values[output_id]=std::move(output);
            break;
        }
        case MgpuOpKind::Negate:
        case MgpuOpKind::CopyPlain:
        case MgpuOpKind::CopyCipher:
        case MgpuOpKind::Relinearize:
        case MgpuOpKind::Rescale:
        case MgpuOpKind::Modswitch:
        case MgpuOpKind::BootstrapFallback:
            output=input_values(0);
            if(op.kind==MgpuOpKind::Negate)
                for(auto &value:output)value=-value;
            clear_values[output_id]=std::move(output);
            break;
        case MgpuOpKind::Rotate: {
            const Values &source=input_values(0);
            const auto step=op.integer_attributes.find("rotate_step");
            if(step==op.integer_attributes.end() || source.empty())
                throw std::runtime_error("clear DSL rotate has no step or slots");
            const auto count=static_cast<std::int64_t>(source.size());
            const auto shift=(step->second%count+count)%count;
            output.resize(source.size());
            for(std::size_t i=0;i<source.size();++i)
                output[i]=source[(i+static_cast<std::size_t>(shift))%source.size()];
            clear_values[output_id]=std::move(output);
            break;
        }
        case MgpuOpKind::Download:
            break;
        }
        for(const auto &input:op.inputs) {
            const auto last=last_use.find(input.id);
            if(last!=last_use.end() && last->second==index)
                clear_values.erase(input.id);
        }
    };
    // The complete rotation-key set holds 238 Galois keys; a single upload
    // would exceed device memory.  Cut the straight-line schedule at the NTT
    // bootstrap boundaries so each segment only owns the rotations it uses.
    std::vector<std::size_t> segment_end;
    for(std::size_t i=0;i<ops.size();++i)
        if(ops[i].kind==MgpuOpKind::BootstrapFallback) {
            // The bootstrap needs its own matrices, workspaces and rotation
            // keys.  Ending the previous segment *before* it lets those
            // segment keys be released first, which keeps the device peak
            // inside a shared-GPU memory budget.
            segment_end.push_back(i);
            segment_end.push_back(i+1);
        }
    // Optional first-window checkpoints for locating clear-vs-ciphertext
    // drift.  Keep this diagnostic-only and limit it to the ordinary segment
    // between the first two bootstraps; the default plan is unchanged.
    if(const char *raw=std::getenv("POSEIDON_DSL_CLEAR_DIAG_CHUNK_OPS")) {
        const std::size_t chunk=static_cast<std::size_t>(std::stoull(raw));
        if(chunk==0)
            throw std::runtime_error("clear diagnostic chunk size must be positive");
        std::vector<std::size_t> bootstrap_indices;
        for(std::size_t i=0;i<ops.size();++i)
            if(ops[i].kind==MgpuOpKind::BootstrapFallback)
                bootstrap_indices.push_back(i);
        if(!bootstrap_indices.empty()) {
            const std::size_t begin=bootstrap_indices[0]+1;
            const std::size_t end=bootstrap_indices.size()>=2
                ? bootstrap_indices[1] : ops.size();
            for(std::size_t checkpoint=begin+chunk;checkpoint<end;checkpoint+=chunk)
                segment_end.push_back(checkpoint);
            std::sort(segment_end.begin(),segment_end.end());
            segment_end.erase(std::unique(segment_end.begin(),segment_end.end()),
                segment_end.end());
            std::cout<<"DSL_CLEAR_DIAG_CHUNKS begin="<<begin<<" end="<<end
                <<" chunk_ops="<<chunk<<'\n';
        }
    }
    if(segment_end.empty() || segment_end.back()!=ops.size())
        segment_end.push_back(ops.size());
    std::unordered_map<ValueId,std::size_t> plain_by_value;
    for(std::size_t i=0;i<plan.encoded_plaintexts.size();++i)
        plain_by_value[plan.encoded_plaintexts[i].value_id]=i;

    // ---- scale-model compensation -------------------------------------
    // A Poseidon rescale drops one ~32-bit prime, while the Dacapo/Hecate
    // model divides the scale by the profile's rescalingFactor (51).  The
    // resulting drift accumulates along the program and by the first
    // bootstrap the ciphertext scale sat at 2^89 instead of the 2^40 the
    // no-hoist continuation is built for.  Replay the deterministic level /
    // scale arithmetic, then re-encode every constant with its own drift
    // removed so the executed scales match the compiled model.
    double model_rescale_bits=51.0;
    if(const char *env=std::getenv("POSEIDON_DSL_MODEL_RESCALE_BITS"))
        model_rescale_bits=std::stod(env);
    const double base_scale_log2=40.0;
    const auto &q_primes=s.metadata.ctx.parameters_literal()->q();
    auto prime_bits=[&](int level)->double {
        if(level<0 || static_cast<std::size_t>(level)>=q_primes.size())return 32.0;
        return std::log2(static_cast<double>(q_primes[static_cast<std::size_t>(level)].value()));
    };
    std::map<ValueId,double> drift,scale_of;
    std::map<ValueId,int> level_of;
    auto define=[&](ValueId id,double d,double s,int l){drift[id]=d;scale_of[id]=s;level_of[id]=l;};
    for(const MgpuOp &op:ops) {
        if(op.outputs.empty())continue;
        const ValueId out=op.outputs[0].id;
        const auto src=[&](std::size_t index) {
            return op.inputs.size()>index?op.inputs[index].id:0;
        };
        switch(op.kind) {
        case MgpuOpKind::UploadCipher:
            define(out,0.0,base_scale_log2,
                static_cast<int>(plan.io_plan.cipher_inputs.front().init_level));
            break;
        case MgpuOpKind::UploadPlain: {
            const auto level=op.integer_attributes.find("encode_level");
            const auto scale=op.integer_attributes.find("encode_scale");
            define(out,0.0,
                scale==op.integer_attributes.end()?0.0:static_cast<double>(scale->second),
                level==op.integer_attributes.end()?0:static_cast<int>(level->second));
            break;
        }
        case MgpuOpKind::Negate:
        case MgpuOpKind::Rotate:
        case MgpuOpKind::AddPlain:
        case MgpuOpKind::Add:
        case MgpuOpKind::Sub:
            define(out,drift[src(0)],scale_of[src(0)],level_of[src(0)]);
            break;
        case MgpuOpKind::Rescale: {
            const int level=level_of[src(0)];
            const double dropped=prime_bits(level);
            define(out,drift[src(0)]+(model_rescale_bits-dropped),
                scale_of[src(0)]-dropped,level-1);
            break;
        }
        case MgpuOpKind::Modswitch: {
            const auto attr=op.integer_attributes.find("down_factor");
            const int down=attr==op.integer_attributes.end()?1:static_cast<int>(attr->second);
            define(out,drift[src(0)],scale_of[src(0)],level_of[src(0)]-down);
            break;
        }
        case MgpuOpKind::MultiplyPlain:
            define(out,drift[src(0)],scale_of[src(0)]+scale_of[src(1)],level_of[src(0)]);
            break;
        case MgpuOpKind::Multiply:
            define(out,drift[src(0)]+drift[src(1)],
                scale_of[src(0)]+scale_of[src(1)],level_of[src(0)]);
            break;
        case MgpuOpKind::BootstrapFallback: {
            const auto attr=op.integer_attributes.find("target_level");
            const int target=attr==op.integer_attributes.end()?0:static_cast<int>(attr->second);
            // ContinuousBootstrap asserts its output on 2^40, but the callback
            // then lifts it to the compiled program's ci<51*0> with a
            // representable one, so the next segment starts from 2^51.
            define(out,51.0-base_scale_log2,51.0,target);
            break;
        }
        default:
            break;
        }
    }
    std::map<ValueId,double> plain_shift;
    for(const MgpuOp &op:ops)
        if(op.kind==MgpuOpKind::AddPlain || op.kind==MgpuOpKind::MultiplyPlain)
            if(op.inputs.size()>1 && plain_shift.find(op.inputs[1].id)==plain_shift.end())
                plain_shift[op.inputs[1].id]=drift[op.inputs[0].id];
    // The GPU rescale divides the scale by one ~32-bit q_last while the
    // compiler models the profile's 51-bit rescalingFactor, so the executed
    // scale drifts above the model.  POSEIDON_DSL_SCALE_SHIFT subtracts a
    // global offset at encode time; the block below solves one *additional*
    // offset per bootstrap so its input lands back on the compiled 2^40 base.
    double global_scale_shift=0.0;
    if(const char *env=std::getenv("POSEIDON_DSL_SCALE_SHIFT"))
        global_scale_shift=std::stod(env);
    // Scale every segment starts from after a no-hoist bootstrap.  This must be
    // the scale the *compiled* program assumes for a bootstrap result, i.e.
    // ci<51*0>.  ContinuousBootstrap itself enforces 2^40, so the callback
    // lifts its output to 2^51 (see below).  Using 2^53.03 here - the EvalMod
    // intermediate - was the earlier mistake.
    const double no_hoist_boot_scale=51.0;
    // The backend emulates the compiler's rescalingFactor with rescale_x2 plus
    // a compensating "one": two limbs consumed, tracked scale dropped by the
    // modelled factor.  The replay below has to use the same arithmetic or the
    // solved per-segment offsets fight the (now correct) trajectory.
    const bool emulate_two_limb_rescale=[] {
        const char *raw=std::getenv("POSEIDON_RESCALE_EMULATE_BITS");
        if(raw==nullptr || *raw=='\0')
            return true;
        return std::stod(raw)>0.0;
    }();
    double max_shift=0;
    {
        // With the 51-bit rescale emulation the executed trajectory follows the
        // compiled model exactly, so a per-segment plaintext offset is no
        // longer wanted: shifting a segment's constants moves the whole
        // segment - including its ciphertext x ciphertext products, where any
        // offset doubles - away from the model again.  Opt in with
        // POSEIDON_DSL_SEGMENT_SHIFT=1 if that ever needs re-testing.
        const bool solve_segment_shifts =
            std::getenv("POSEIDON_DSL_SEGMENT_SHIFT")!=nullptr;
        double segment_shift_scale=1.0;
        if(const char *raw=std::getenv("POSEIDON_DSL_SEGMENT_SHIFT_SCALE"))
            segment_shift_scale=std::stod(raw);
        // Segment that owns each UploadPlain result (by its output value id).
        std::map<ValueId,std::size_t> plain_owner;
        for(std::size_t seg=0;seg<segment_end.size();++seg) {
            const std::size_t b=seg==0?0:segment_end[seg-1];
            const std::size_t e=segment_end[seg];
            for(std::size_t i=b;i<e;++i)
                if(ops[i].kind==MgpuOpKind::UploadPlain && !ops[i].outputs.empty())
                    plain_owner.emplace(ops[i].outputs[0].id,seg);
        }
        // A segment's constants drive the bootstrap of the *next* segment that
        // ends in a BootstrapFallback, so the offset solved there is the one
        // those constants must carry.
        std::vector<std::size_t> eff_seg(segment_end.size(),0);
        {
            std::size_t next=segment_end.empty()?0:segment_end.size()-1;
            for(std::size_t s=segment_end.size();s-->0;) {
                const bool is_boot=segment_end[s]>0 &&
                    ops[segment_end[s]-1].kind==MgpuOpKind::BootstrapFallback;
                if(is_boot)next=s;
                eff_seg[s]=next;
            }
        }
        // Carry-in state for each segment: segment 0 starts from the uploaded
        // input, every later segment from the previous bootstrap output.
        struct CarryState { int level; double scale; };
        std::map<std::size_t,std::map<ValueId,CarryState>> carry_in;
        {
            std::map<ValueId,CarryState> current;
            current[plan.io_plan.cipher_inputs.front().value_id]=
                {static_cast<int>(plan.io_plan.cipher_inputs.front().init_level),40.0};
            carry_in[0]=current;
            for(std::size_t seg=0;seg<segment_end.size();++seg) {
                const std::size_t b=seg==0?0:segment_end[seg-1];
                const std::size_t e=segment_end[seg];
                for(std::size_t i=b;i<e;++i)
                    if(ops[i].kind==MgpuOpKind::BootstrapFallback && !ops[i].outputs.empty()) {
                        const auto at=ops[i].integer_attributes.find("target_level");
                        current.clear();
                        current[ops[i].outputs[0].id]=
                            {at==ops[i].integer_attributes.end()?0:static_cast<int>(at->second),
                             no_hoist_boot_scale};
                    }
                if(seg+1<segment_end.size())
                    carry_in[seg+1]=current;
            }
        }
        std::vector<double> segment_shift(segment_end.size(),0.0);
        std::size_t stuck=0;
        for(std::size_t seg=0;seg<segment_end.size();++seg) {
            const std::size_t e0=segment_end[seg];
            if(e0==0 || ops[e0-1].kind!=MgpuOpKind::BootstrapFallback)
                continue;
            auto owner_shift=[&](ValueId id,double trial)->double {
                const auto owner=plain_owner.find(id);
                if(owner==plain_owner.end())
                    return global_scale_shift;
                const std::size_t e=eff_seg[owner->second];
                return global_scale_shift+(e==seg?trial:segment_shift[e]);
            };
            auto replay=[&](std::size_t upto,double trial)->double {
                std::map<ValueId,std::pair<int,double>> env;
                const auto seed=carry_in.find(seg);
                if(seed!=carry_in.end())
                    for(const auto &kv:seed->second)
                        env[kv.first]={kv.second.level,kv.second.scale};
                double result=40.0;
                for(std::size_t i=0;i<upto;++i) {
                    const MgpuOp &op=ops[i];
                    if(op.outputs.empty())continue;
                    const ValueId out=op.outputs[0].id;
                    auto src=[&](std::size_t k){return op.inputs.size()>k?op.inputs[k].id:0;};
                    switch(op.kind) {
                    case MgpuOpKind::UploadCipher: continue;
                    case MgpuOpKind::UploadPlain: {
                        const auto lv=op.integer_attributes.find("encode_level");
                        const auto sc=op.integer_attributes.find("encode_scale");
                        const double delta=owner_shift(out,trial);
                        env[out]={lv==op.integer_attributes.end()?0:static_cast<int>(lv->second),
                            (sc==op.integer_attributes.end()?0.0:static_cast<double>(sc->second))-delta};
                        continue;
                    }
                    case MgpuOpKind::Modswitch: {
                        const auto at=op.integer_attributes.find("down_factor");
                        const auto in=env.find(src(0));
                        if(in==env.end())continue;
                        env[out]={in->second.first-(at==op.integer_attributes.end()?1:static_cast<int>(at->second)),
                            in->second.second};
                        continue;
                    }
                    case MgpuOpKind::Rescale: {
                        const auto in=env.find(src(0));
                        if(in==env.end())continue;
                        const int lv=in->second.first;
                        if(emulate_two_limb_rescale) {
                            // rescale_x2 consumes two limbs; the compensating
                            // plaintext one restores the excess, so the tracked
                            // scale drops exactly the modelled factor.
                            env[out]={lv-2,in->second.second-model_rescale_bits};
                        } else {
                            const double bits=(lv>=0 && static_cast<std::size_t>(lv)<q_primes.size())
                                ?std::log2(static_cast<double>(q_primes[static_cast<std::size_t>(lv)].value()))
                                :32.0;
                            env[out]={lv-1,in->second.second-bits};
                        }
                        continue;
                    }
                    case MgpuOpKind::AddPlain:
                    case MgpuOpKind::MultiplyPlain: {
                        const auto a=env.find(src(0));const auto b=env.find(src(1));
                        if(a==env.end())continue;
                        double sc=a->second.second;
                        if(op.kind==MgpuOpKind::MultiplyPlain && b!=env.end())sc+=b->second.second;
                        env[out]={a->second.first,sc};
                        continue;
                    }
                    case MgpuOpKind::Add:
                    case MgpuOpKind::Sub: {
                        const auto a=env.find(src(0));const auto b=env.find(src(1));
                        if(a==env.end())continue;
                        double sc=a->second.second;
                        if(b!=env.end())sc=std::max(sc,b->second.second);
                        env[out]={a->second.first,sc};
                        continue;
                    }
                    case MgpuOpKind::Multiply: {
                        const auto a=env.find(src(0));const auto b=env.find(src(1));
                        if(a==env.end())continue;
                        double sc=a->second.second;
                        if(b!=env.end())sc+=b->second.second;
                        env[out]={a->second.first,sc};
                        continue;
                    }
                    case MgpuOpKind::BootstrapFallback: {
                        const auto at=op.integer_attributes.find("target_level");
                        env[out]={at==op.integer_attributes.end()?0:static_cast<int>(at->second),
                            no_hoist_boot_scale};
                        continue;
                    }
                    default: {
                        const auto a=env.find(src(0));
                        if(a!=env.end())env[out]=a->second;
                        continue;
                    }
                    }
                }
                const auto it=env.find(ops[upto-1].inputs[0].id);
                return it==env.end()?result:it->second.second;
            };
            const double s0=replay(e0,0.0);
            const double s1=replay(e0,1.0);
            const double k=s0-s1;
            if(solve_segment_shifts && k>1e-9)
                segment_shift[seg]=std::max(0.0,(s0-40.0)/k)*segment_shift_scale;
            else if(s0>40.0+1e-6)
                ++stuck;
        }
        CKKSEncoder compensator(s.metadata.ctx);
        const auto &map=s.metadata.ctx.crt_context()->parms_id_map();
        std::size_t reencoded=0;
        std::size_t underflowed=0;
        for(HevmEncodedPlaintext &entry:plan.encoded_plaintexts) {
            const auto owner=plain_owner.find(entry.value_id);
            if(owner==plain_owner.end() || entry.message==nullptr)
                continue;
            const double allowed=global_scale_shift+segment_shift[eff_seg[owner->second]];
            max_shift=std::max(max_shift,allowed);
            const double encoded_log2=static_cast<double>(entry.scale)-allowed;
            if(encoded_log2<=0.0)
                continue;
            const auto pid=map.find(static_cast<std::uint32_t>(entry.level));
            if(pid==map.end())continue;
            // A shifted-down constant must still be representable: the encoder
            // rounds |message| * 2^scale, so a constant whose peak lands below
            // one half would silently become the zero plaintext.  Keep the
            // unshifted encoding instead of deleting the constant.
            double peak=0.0;
            for(const double component:*entry.message)
                peak=std::max(peak,std::abs(component));
            if(peak*std::exp2(encoded_log2)<0.5) {
                ++underflowed;
                continue;
            }
            auto replacement=std::make_shared<Plaintext>();
            compensator.encode(*entry.message,pid->second,
                std::exp2(encoded_log2),*replacement);
            entry.plaintext=replacement;
            ++reencoded;
        }
        std::cout<<"DSL_SEGMENT_SHIFTS base="<<global_scale_shift<<" stuck="<<stuck
            <<" reencoded="<<reencoded<<" underflowed="<<underflowed;
        for(std::size_t seg=0;seg<segment_shift.size();++seg)
            if(segment_shift[seg]>1e-9)
                std::cout<<" s"<<seg<<'='<<segment_shift[seg];
        std::cout<<'\n';
    }
    std::cout<<"DSL_SCALE_MODEL rescale_bits="<<model_rescale_bits
        <<" max_plaintext_shift="<<max_shift
        <<" bootstrap_input_drift="<<(drift.empty()?0.0:drift[plan.io_plan.results.front().value_id])
        <<'\n';
    // Built lazily so a prefix without a bootstrap op needs no matrix upload.
    std::unique_ptr<ContinuousBootstrap> bootstrap;
    std::size_t bootstrap_calls=0;
    const int device=ops.empty()?0:ops.front().device_id;
    std::map<ValueId,std::shared_ptr<Ciphertext>> carried;
    const ValueId initial_id=plan.io_plan.cipher_inputs.front().value_id;
    carried[initial_id]=
        std::make_shared<Ciphertext>(std::move(input));
    std::size_t begin=0;
    std::shared_ptr<Ciphertext> final_result;
    for(std::size_t seg=0;seg<segment_end.size();++seg) {
        const std::size_t end=segment_end[seg];
        std::vector<ValueId> segment_inputs,segment_results;
        std::vector<std::shared_ptr<const Ciphertext>> bound;
        std::vector<char> synthetic;
        std::vector<std::size_t> plain_prelude;
        auto add_input=[&](ValueId id) {
            if(std::find(segment_inputs.begin(),segment_inputs.end(),id)!=segment_inputs.end())
                return;
            const auto value=carried.find(id);
            if(value==carried.end())
                throw std::runtime_error("DSL segment "+std::to_string(seg)+
                    " needs an unbound value %"+std::to_string(id));
            const auto defined=defined_at.find(id);
            segment_inputs.push_back(id);
            bound.push_back(value->second);
            // Values defined inside this segment (the HEVM argument upload)
            // already have their own UploadCipher op.
            synthetic.push_back(
                defined!=defined_at.end() && defined->second<begin ? 1 : 0);
        };
        if(seg==0)
            add_input(initial_id);
        for(std::size_t i=begin;i<end;++i)
            for(const auto &in:ops[i].inputs) {
                const auto defined=defined_at.find(in.id);
                if(defined==defined_at.end() || defined->second>=begin)continue;
                // A plaintext produced in an earlier segment (an UploadPlain
                // value reused across a segment boundary) is not a carried
                // ciphertext; re-emit its upload at the head of this segment.
                if(ops[defined->second].kind==MgpuOpKind::UploadPlain)
                    plain_prelude.push_back(defined->second);
                else
                    add_input(in.id);
            }
        HevmStaticExecutionPlan segment;
        for(std::size_t k=0;k<segment_inputs.size();++k) {
            const ValueId id=segment_inputs[k];
            HevmCipherInputSlot slot;
            slot.index=k;
            slot.value_id=id;
            slot.device_id=device;
            segment.io_plan.cipher_inputs.push_back(slot);
            if(synthetic[k]==0)
                continue;
            MgpuOp upload;
            upload.kind=MgpuOpKind::UploadCipher;
            upload.device_id=device;
            upload.outputs.push_back(MgpuValueRef{id});
            upload.debug_name="dsl_segment_input";
            segment.schedule.ops.push_back(std::move(upload));
        }
        for(const std::size_t index:plain_prelude) {
            const auto found=plain_by_value.find(ops[index].outputs[0].id);
            if(found==plain_by_value.end())
                throw std::runtime_error("DSL segment is missing a reused encoded plaintext");
            if(std::find_if(segment.schedule.ops.begin(),segment.schedule.ops.end(),
                    [&](const MgpuOp &op){return op.outputs.size()==1 &&
                        op.outputs[0].id==ops[index].outputs[0].id;})
               !=segment.schedule.ops.end())
                continue;
            segment.encoded_plaintexts.push_back(plan.encoded_plaintexts[found->second]);
            segment.schedule.ops.push_back(ops[index]);
        }
        for(std::size_t i=begin;i<end;++i) {
            // Aggregate the per-rescale model correction: one multiply at the
            // segment's last rescale carrying 2^-(extra * rescale_count) for
            // both value and scale, instead of ~20 tiny multiplies.
            std::size_t last_rescale=end;
            double aggregate_fix=0.0;
            for(std::size_t k=begin;k<end;++k)
                if(ops[k].kind==MgpuOpKind::Rescale && !ops[k].inputs.empty()) {
                    last_rescale=k;
                    const auto lv=level_of.find(ops[k].inputs[0].id);
                    if(lv!=level_of.end() && lv->second>=0 &&
                       static_cast<std::size_t>(lv->second)<q_primes.size())
                        aggregate_fix+=model_rescale_bits-std::log2(
                            static_cast<double>(
                                q_primes[static_cast<std::size_t>(lv->second)]
                                    .value()));
                }
            const int aggregate_fix_bits=static_cast<int>(aggregate_fix+0.5);
            if(ops[i].kind==MgpuOpKind::UploadPlain) {
                const auto found=plain_by_value.find(ops[i].outputs[0].id);
                if(found==plain_by_value.end())
                    throw std::runtime_error("DSL segment is missing an encoded plaintext");
                segment.encoded_plaintexts.push_back(plan.encoded_plaintexts[found->second]);
            }
            MgpuOp current=ops[i];
            segment.schedule.ops.push_back(std::move(current));
            if(ops[i].kind==MgpuOpKind::UploadPlain || ops[i].kind==MgpuOpKind::CopyPlain)
                continue;
            for(const auto &out:ops[i].outputs) {
                const auto use=last_use.find(out.id);
                if(use==last_use.end() || use->second<end)continue;
                if(std::find(segment_results.begin(),segment_results.end(),out.id)!=segment_results.end())continue;
                segment_results.push_back(out.id);
            }
        }
        if(seg+1==segment_end.size() && !plan.io_plan.results.empty()) {
            const ValueId final_id=plan.io_plan.results.front().value_id;
            const auto defined=defined_at.find(final_id);
            if(defined!=defined_at.end() && defined->second>=begin &&
               std::find(segment_results.begin(),segment_results.end(),final_id)==
                   segment_results.end())
                segment_results.push_back(final_id);
        }
        if(clear_reference_diag)
            for(std::size_t i=begin;i<end;++i)
                evaluate_clear_op(i);
        for(std::size_t result_index=0;result_index<segment_results.size();++result_index) {
            const ValueId id=segment_results[result_index];
            MgpuOp download;
            download.kind=MgpuOpKind::Download;
            download.device_id=device;
            download.inputs.push_back(MgpuValueRef{id});
            download.debug_name="dsl_segment_result";
            segment.schedule.ops.push_back(std::move(download));
            HevmResultSlot slot;
            slot.index=result_index;
            slot.value_id=id;
            slot.device_id=device;
            segment.io_plan.results.push_back(slot);
        }
        const auto steps=dsl_rotation_steps(segment.schedule);
        PoseidonGpuHevmExecutionOptions options;
        options.device_count=1;
        if(!steps.empty())
            options.gpu_galois_keys=
                std::make_shared<GpuGaloisKeysData>(s.rotation_keys(steps));
        options.gpu_relin_keys=std::shared_ptr<const GpuRelinKeysData>(&s.keys,
            [](const GpuRelinKeysData *){});
        int bootstrap_target=-1;
        for(const auto &op:segment.schedule.ops)
            if(op.kind==MgpuOpKind::BootstrapFallback) {
                const auto attr=op.integer_attributes.find("target_level");
                if(attr!=op.integer_attributes.end())
                    bootstrap_target=static_cast<int>(attr->second);
            }
        options.bootstrap=[&](const GpuCiphertextData &in_ct,GpuCiphertextData &out_ct,int){
            // The model-aligned rescale metadata override is only meant for the
            // straight-line segment ops; the no-hoist bootstrap internals are
            // built for the natural (~q_last) rescale, so hide the override for
            // the duration of the bootstrap.
            struct RescaleMetaGuard {
                bool had=false;
                std::string value;
                RescaleMetaGuard() {
                    const char *raw=std::getenv("POSEIDON_RESCALE_META_BITS");
                    if(raw) { had=true; value=raw; unsetenv("POSEIDON_RESCALE_META_BITS"); }
                }
                ~RescaleMetaGuard() {
                    if(had) setenv("POSEIDON_RESCALE_META_BITS",value.c_str(),1);
                }
            } rescale_meta_guard;
            if(!bootstrap)
                bootstrap=std::make_unique<ContinuousBootstrap>(s);
            if(std::getenv("POSEIDON_DSL_DIAG")) {
                const auto &map=s.metadata.ctx.crt_context()->parms_id_map();
                const auto q6=map.find(5);
                std::cout<<"DSL_BOOTSTRAP_IN q="<<in_ct.meta.q_count
                    <<" log_scale="<<std::log2(in_ct.meta.scale)
                    <<" parms="<<in_ct.meta.parms_id[0]<<','<<in_ct.meta.parms_id[1]
                    <<','<<in_ct.meta.parms_id[2]<<','<<in_ct.meta.parms_id[3];
                if(q6!=map.end())
                    std::cout<<" q6_parms="<<q6->second[0]<<','<<q6->second[1]
                        <<','<<q6->second[2]<<','<<q6->second[3]
                        <<" match="<<(in_ct.meta.parms_id==q6->second);
                std::cout<<'\n';
            }
            // Bring the segment output onto the no-hoist continuation's exact
            // (Q6, 2^40) input contract while keeping the encoded message
            // intact.
            //
            // Order matters.  A plaintext of value 1 encoded at 2^(40-s) only
            // represents 1 when 40-s >= 0; for s > 40 the encoder rounds
            // 1*2^(40-s) to zero and the multiply silently wipes the whole
            // ciphertext.  So the surplus scale has to be removed with *real*
            // rescales first (each one divides the value by its q_last and
            // drops the tracked scale by the same amount, so the message
            // survives), and only the representable remainder is fixed with a
            // plaintext one.  Dropping modulus limbs is value preserving, so
            // the Q6 reduction is free and can happen in either order.
            const auto &boot_map=s.metadata.ctx.crt_context()->parms_id_map();
            CT rescale_ping,rescale_pong,reduced_input,normalised_input;
            const CT *bootstrap_input=&in_ct;
            bool rescale_toggle=false;
            // A direct 51 -> ~19 scale drop throws away roughly 32 bits of
            // fractional precision before every bootstrap.  Lift by an exact
            // plaintext one so the next real single-limb rescale lands at
            // scale 2^40 instead of 2^19; this preserves the message while
            // retaining about 21 more bits in the rescale rounding.
            if(bootstrap_input->meta.q_count>6 &&
               std::log2(bootstrap_input->meta.scale)>40.0+1e-9) {
                const auto &q=s.metadata.ctx.parameters_literal()->q();
                const std::size_t last=bootstrap_input->meta.q_count-1;
                if(last<q.size()) {
                    const double drop_bits=std::log2(static_cast<double>(q[last].value()));
                    const double lift_bits=40.0+drop_bits-
                        std::log2(bootstrap_input->meta.scale);
                    if(lift_bits>0.0) {
                        CKKSEncoder encoder(s.metadata.ctx);
                        const std::size_t slots=static_cast<std::size_t>(
                            s.metadata.ctx.parameters_literal()->slot());
                        Plaintext one;
                        encoder.encode(std::vector<double>(slots,1.0),
                            bootstrap_input->meta.parms_id,std::exp2(lift_bits),one);
                        auto uploaded=gpu::GpuUploader::upload_plaintext(one,device);
                        CT lifted;
                        s.eval.multiply_plain(*bootstrap_input,uploaded,lifted);
                        std::cout<<"DSL_BOOTSTRAP_PRECISION_LIFT log_scale="
                            <<std::log2(bootstrap_input->meta.scale)
                            <<" lift_bits="<<lift_bits<<'\n';
                        rescale_ping=std::move(lifted);
                        bootstrap_input=&rescale_ping;
                        rescale_toggle=true;
                    }
                }
            }
            while(std::log2(bootstrap_input->meta.scale)>40.0+1e-9 &&
                  bootstrap_input->meta.q_count>6) {
                CT &destination=rescale_toggle?rescale_pong:rescale_ping;
                s.eval.rescale(*bootstrap_input,destination);
                bootstrap_input=&destination;
                rescale_toggle=!rescale_toggle;
            }
            if(std::log2(bootstrap_input->meta.scale)>40.0+1e-9)
                throw std::runtime_error(
                    "bootstrap input is above the 2^40 base scale with no room "
                    "left to rescale (q="+
                    std::to_string(bootstrap_input->meta.q_count)+", log_scale="+
                    std::to_string(std::log2(bootstrap_input->meta.scale))+")");
            if(bootstrap_input->meta.q_count>6) {
                const auto target=boot_map.find(5);
                if(target==boot_map.end())
                    throw std::runtime_error("Q6 parms_id is unavailable");
                s.eval.drop_modulus(*bootstrap_input,reduced_input,target->second);
                bootstrap_input=&reduced_input;
            } else if(bootstrap_input->meta.q_count<6)
                throw std::runtime_error(
                    "bootstrap input has fewer than six modulus limbs");
            if(std::abs(std::log2(bootstrap_input->meta.scale)-40.0)>1e-9) {
                const double gap=40.0-std::log2(bootstrap_input->meta.scale);
                CKKSEncoder encoder(s.metadata.ctx);
                const std::size_t slots=static_cast<std::size_t>(
                    s.metadata.ctx.parameters_literal()->slot());
                Plaintext one;
                encoder.encode(std::vector<double>(slots,1.0),
                    bootstrap_input->meta.parms_id,std::exp2(gap),one);
                auto uploaded=
                    gpu::GpuUploader::upload_plaintext(one,device);
                s.eval.multiply_plain(*bootstrap_input,uploaded,normalised_input);
                bootstrap_input=&normalised_input;
            }
            if(std::getenv("POSEIDON_DSL_DIAG")) {
                const auto prepared_values=decoded_values(s.replay,*bootstrap_input);
                double magnitude=0;
                for(const auto &value:prepared_values)
                    magnitude=std::max(magnitude,static_cast<double>(std::abs(value)));
                std::cout<<"DSL_BOOTSTRAP_PREPARED q="<<bootstrap_input->meta.q_count
                    <<" log_scale="<<std::log2(bootstrap_input->meta.scale)
                    <<" max_abs="<<magnitude<<'\n';
            }
            const bool check_bootstrap_values=
                std::getenv("POSEIDON_DSL_BOOTSTRAP_ACCURACY_DIAG")!=nullptr;
            Values bootstrap_input_values;
            if(check_bootstrap_values)
                bootstrap_input_values=decoded_values(s.replay,*bootstrap_input);
            CT produced=bootstrap->run(*bootstrap_input,nullptr);
            ++bootstrap_calls;
            if(check_bootstrap_values) {
                const auto bootstrap_output_values=decoded_values(s.replay,produced);
                if(bootstrap_output_values.size()!=bootstrap_input_values.size())
                    throw std::runtime_error("bootstrap diagnostic slot count changed");
                double max_error=0,max_input=0;
                long double squared_error=0;
                for(std::size_t i=0;i<bootstrap_input_values.size();++i) {
                    const double error=static_cast<double>(std::abs(
                        bootstrap_output_values[i]-bootstrap_input_values[i]));
                    max_error=std::max(max_error,error);
                    max_input=std::max(max_input,static_cast<double>(
                        std::abs(bootstrap_input_values[i])));
                    squared_error+=static_cast<long double>(error)*error;
                }
                std::cout<<"DSL_BOOTSTRAP_VALUE_CHECK call="<<bootstrap_calls
                    <<" max_abs_error="<<max_error
                    <<" rms_error="<<std::sqrt(static_cast<double>(
                        squared_error/bootstrap_input_values.size()))
                    <<" input_max_abs="<<max_input<<'\n';
            }
            // ContinuousBootstrap pins its output on exactly 2^40, but the
            // compiled program's bootstrap result is ci<51*0> and every later
            // step assumes that: each layer is "+constant scale, -51 rescale"
            // and every earth.upscale is *relative* (target - current), so a
            // segment started 11 bits low keeps that offset and then doubles it
            // on each ciphertext x ciphertext multiply (51->40 vs 40->29, then
            // 40*40=80 vs 29*29=58).  Lift the tracked scale back to the
            // model's 2^51 with a representable "one" encoded at 2^11 (integer
            // coefficient 2^11): the encrypted value is multiplied by one, so
            // only the scale metadata moves.
            const double boot_output_log_scale=[] {
                const char *raw=std::getenv("POSEIDON_DSL_BOOT_OUTPUT_LOG_SCALE");
                return (raw!=nullptr && *raw!='\0') ? std::stod(raw) : 51.0;
            }();
            if(std::abs(std::log2(produced.meta.scale)-boot_output_log_scale)>1e-9) {
                CKKSEncoder encoder(s.metadata.ctx);
                const std::size_t slots=static_cast<std::size_t>(
                    s.metadata.ctx.parameters_literal()->slot());
                Plaintext one;
                encoder.encode(
                    std::vector<double>(slots,1.0), produced.meta.parms_id,
                    std::exp2(boot_output_log_scale-std::log2(produced.meta.scale)),
                    one);
                auto uploaded=gpu::GpuUploader::upload_plaintext(one,device);
                CT lifted;
                s.eval.multiply_plain(produced,uploaded,lifted);
                produced=std::move(lifted);
            }
            const std::size_t target_level=static_cast<std::size_t>(bootstrap_target)+1;
            const std::size_t target_q_count=target_level+1;
            // Keep one guard q limb above the declared target. The final level-2
            // result adds a scale-71 constant, which cannot be re-encoded
            // under only two ~32-bit q limbs.
            if(bootstrap_target<0 || produced.meta.q_count<=target_q_count) {
                out_ct=std::move(produced);
                return;
            }
            const auto &map=s.metadata.ctx.crt_context()->parms_id_map();
            const auto target=map.find(static_cast<std::uint32_t>(target_level));
            if(target==map.end())
                throw std::runtime_error("DSL bootstrap target level is unavailable");
            GpuCiphertextData reduced;
            s.eval.drop_modulus(produced,reduced,target->second);
            out_ct=std::move(reduced);
        };
        std::cout<<"DSL_SEGMENT index="<<seg<<" ops="<<segment.schedule.ops.size()
            <<" inputs="<<segment_inputs.size()<<" results="<<segment_results.size()
            <<" plaintexts="<<segment.encoded_plaintexts.size()
            <<" rotate_steps="<<steps.size()<<'\n';
        const auto execution=execute_hevm_static_plan_with_poseidon_gpu(
            s.metadata.ctx,segment,bound,options);
        if(!execution.ok())
            throw std::runtime_error("DSL segment "+std::to_string(seg)+" failed:\n"+
                execution.format_errors());
        if(execution.results.size()!=segment_results.size())
            throw std::runtime_error("DSL segment result count mismatch");
        for(std::size_t k=0;k<segment_results.size();++k) {
            if(execution.results[k]==nullptr)
                throw std::runtime_error("DSL segment produced a null result");
            carried[segment_results[k]]=execution.results[k];
            const bool print_segment_diag=std::getenv("POSEIDON_DSL_DIAG")!=nullptr;
            if(print_segment_diag || clear_reference_diag) {
                const CT uploaded=GpuUploader::upload_ciphertext(
                    *execution.results[k],device);
                const auto values=s.replay.decoded(uploaded);
                if(print_segment_diag) {
                    double magnitude=0;
                    for(const auto &value:values)
                        magnitude=std::max(magnitude,static_cast<double>(std::abs(value)));
                    std::cout<<"DSL_SEGMENT_RESULT seg="<<seg<<" k="<<k
                        <<" value="<<segment_results[k]
                        <<" q="<<uploaded.meta.q_count
                        <<" log_scale="<<std::log2(uploaded.meta.scale)
                        <<" max_abs="<<magnitude<<'\n';
                }
                if(clear_reference_diag) {
                    const auto expected=clear_values.find(segment_results[k]);
                    if(expected==clear_values.end() || expected->second.size()!=values.size())
                        throw std::runtime_error("clear DSL boundary reference is unavailable");
                    double max_error=0.0;
                    long double squared_error=0.0L;
                    for(std::size_t i=0;i<values.size();++i) {
                        const Complex actual_value(values[i].real(),values[i].imag());
                        const long double error=std::abs(actual_value-expected->second[i]);
                        max_error=std::max(max_error,static_cast<double>(error));
                        squared_error+=error*error;
                    }
                    std::cout<<"DSL_CLEAR_BOUNDARY seg="<<seg
                        <<" value="<<segment_results[k];
                    const auto definition=defined_at.find(segment_results[k]);
                    if(definition!=defined_at.end())
                        std::cout<<" def_op="<<definition->second
                            <<" def_kind="<<to_string(ops[definition->second].kind);
                    std::cout<<" max_error="<<max_error
                        <<" rms_error="<<std::sqrt(static_cast<double>(
                            squared_error/values.size()))<<'\n';
                }
            }
        }
        begin=end;
    }
    if(!truncated.empty()) {
        std::cout<<"DSL_TRUNCATED ops="<<ops.size()
            <<" segments="<<segment_end.size()<<" bootstrap_calls="<<bootstrap_calls<<'\n';
        return 0;
    }
    if(plan.io_plan.results.size()!=1)
        throw std::runtime_error("DSL runner expects exactly one result");
    const auto result_value=carried.find(plan.io_plan.results.front().value_id);
    if(result_value==carried.end() || result_value->second==nullptr)
        throw std::runtime_error("DSL execution produced no result ciphertext");
    final_result=result_value->second;
    std::cout<<"DSL_EXECUTION ok=true bootstrap_calls="<<bootstrap_calls<<'\n';
    CT gpu_result=GpuUploader::upload_ciphertext(*final_result,0);
    const auto decoded=s.replay.decoded(gpu_result);
    const auto reference=read_double_file(reference_path,10);
    // Dacapo's ResNet test applies postprocess() to the HEVM result before
    // comparing it with the PyTorch logits; that function multiplies by 32.
    constexpr double hevm_output_gain=32.0;
    if(clear_reference_diag) {
        const auto clear_result=clear_values.find(
            plan.io_plan.results.front().value_id);
        if(clear_result==clear_values.end() ||
           clear_result->second.size()<reference.size())
            throw std::runtime_error("clear DSL final reference is unavailable");
        double clear_max_error=0.0;
        std::cout<<"DSL_CLEAR_RESULT logits";
        for(std::size_t i=0;i<reference.size();++i) {
            const double value=static_cast<double>(clear_result->second[i].real())*
                hevm_output_gain;
            std::cout<<' '<<value;
            clear_max_error=std::max(clear_max_error,
                std::abs(value-reference[i]));
        }
        std::cout<<'\n'<<"DSL_CLEAR_CHECK max_logit_error="
            <<clear_max_error<<" output_gain="<<hevm_output_gain<<'\n';
    }
    double best=-std::numeric_limits<double>::infinity();
    int prediction=0;
    std::cout<<"DSL_RESULT_RAW logits";
    for(std::size_t i=0;i<reference.size();++i) {
        std::cout<<' '<<decoded[i].real();
    }
    std::cout<<'\n'<<"DSL_RESULT logits";
    for(std::size_t i=0;i<reference.size();++i) {
        const double value=decoded[i].real()*hevm_output_gain;
        std::cout<<' '<<value;
        if(value>best){best=value;prediction=static_cast<int>(i);}
    }
    std::cout<<'\n';
    double max_error=0,raw_max_error=0;
    for(std::size_t i=0;i<reference.size();++i)
    {
        max_error=std::max(max_error,std::abs(
            decoded[i].real()*hevm_output_gain-reference[i]));
        raw_max_error=std::max(raw_max_error,std::abs(
            decoded[i].real()-reference[i]));
    }
    const int reference_prediction=static_cast<int>(
        std::max_element(reference.begin(),reference.end())-reference.begin());
    const bool accuracy_pass=std::isfinite(max_error) && max_error<=0.1 &&
        prediction==reference_prediction;
    std::cout<<"DSL_CHECK prediction="<<prediction
        <<" reference_prediction="<<reference_prediction
        <<" max_logit_error="<<max_error
        <<" raw_max_logit_error="<<raw_max_error
        <<" output_gain="<<hevm_output_gain
        <<" accuracy="<<(accuracy_pass?"PASS":"FAIL")<<'\n';
    return accuracy_pass?0:3;
}
#endif // POSEIDON_S2C_FIRST_DSL_RUNNER
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

#ifdef POSEIDON_S2C_FIRST_DSL_RUNNER
int run_s2c_first_dsl_application(int argc,char **argv) {
    try {
        if(argc!=7 || std::string(argv[1])!="--dsl-run")
            throw std::runtime_error(
                "usage: --dsl-run relu_fixture.txt hevm.bin constants.cst slots.txt reference_logits.txt");
        const std::string fixture=argv[2];
        const std::string hevm=argv[3],constants=argv[4],slots=argv[5],reference=argv[6];
        std::string mode="--unsafe-accuracy-only",profile="q50-fixed45",dataset="grid";
        std::vector<char *> args{argv[0],mode.data(),profile.data(),dataset.data(),
            const_cast<char *>(fixture.c_str())};
        return run_bootstrap_baseline(static_cast<int>(args.size()),args.data(),{},{},
            nullptr,false,[&](const BaselineSession &session) {
                return network_probe::run_dsl_plan(session,hevm,constants,slots,reference);
            });
    } catch(const std::exception &e) {
        std::cerr<<"ERROR "<<e.what()<<'\n';
        return 2;
    }
}
#endif // POSEIDON_S2C_FIRST_DSL_RUNNER

#ifndef POSEIDON_S2C_FIRST_RESNET20_LIBRARY
int main(int argc,char **argv) {
    return run_s2c_first_resnet20_application(argc,argv);
}
#endif



