#pragma once
// Construction-only immutable profiles for a separately labelled FHE accuracy
// control. No shared T/r mutation, shallow profile copy, or default changes.
#include "source_runtime.hpp"
#include "source_manual_c2s.hpp"
#include "poseidon/advance/homomorphic_mod.h"
#include "source_manual_coefficients.hpp"
#include <thread>
#include <set>
namespace cerium_source::manual_backend {
using namespace cerium_source;
using namespace poseidon::gpu;
// One owner for GPU submission, profile construction and the shared
// Runtime encoder. NativeLinear may still run independent CPU preparation
// (its own periodic encoder plus const Api::prime) and drains it per projection.
class SubmissionOwner {
    const std::thread::id owner=std::this_thread::get_id();
public:
    void check() const {if(std::this_thread::get_id()!=owner)throw std::runtime_error("mixed Block requires one CPU submission owner");}
};
inline std::pair<double,std::size_t> planned_scale(Api &api,double s,const GpuLinearMatrixGroupQP &group) {
    if(group.data().empty())throw std::runtime_error("mixed empty DFT plan");
    auto q=std::size_t(group.data().front().level)+1;
    for(std::size_t i=0;i<group.data().size();++i) {
        s*=group.data()[i].scale;
        const auto drops=group.rescale_counts().empty()?group.step():group.rescale_counts().at(i);
        for(unsigned j=0;j<drops;++j)s/=api.prime(q--);
    }return {s,q};
}
inline void check_key_meta(const GpuEvaluationKeyData &key) {
    if(key.empty()||key.meta.degree!=65536||key.meta.q_count!=49||key.meta.p_count!=25||key.meta.component_count!=2||
       (key.level_keys&&!key.level_keys->empty()))throw std::runtime_error("mixed owning ordinary key must be Global Q49/P25");
}
inline void validate_key(const GpuEvaluationKeyData &key) {
    check_key_meta(key);
    std::vector<std::size_t> levels;for(std::size_t q=2;q<=49;++q)levels.push_back(q);
    GpuUploader::prepare_key_views_for_q_counts(key,levels);
}
inline void verify_profile(Runtime &runtime,const GpuBootstrapProfile &profile,unsigned ratio,double eta,double target,double alpha) {
    const auto &d=profile.bootstrap_data;const auto id=runtime.parameters.get_level_by_q_count(27).parms_id;
    if((ratio!=8192&&ratio!=65536)||!(eta>0&&std::isfinite(eta))||
       d.output_ratio!=ratio||d.q0_over_message_ratio!=target||d.output_scale_multiplier!=alpha||
       !d.slot_to_coeff_first||!d.project_real||d.allow_environment_linear_transform_override||
       d.linear_transform_mode!=GpuLinearTransformMode::DoubleHoistBsgs||d.input_q_count!=6||d.input_scale!=std::exp2(40.)||
       d.raised_scale_override!=std::exp2(54.)||d.post_raise_integer_multiplier!=1||d.post_raise_scale_multiplier!=1.||!d.post_raise_plaintext.empty()||
       d.slot_to_coeff_input_q_count!=27||d.slot_to_coeff_input_parms_id!=id||d.slot_to_coeff_output_scale!=std::exp2(50.)||
       d.eval_mod.output_q_count!=27||d.eval_mod.output_parms_id!=id||d.eval_mod.target_scale!=std::exp2(50.)||
       d.eval_mod.polynomial_degree!=30||!d.eval_mod.dynamic_rescale||d.eval_mod.double_angle_constants.size()!=3||
       d.active_coeff_to_slot_matrix_qp().rescale_counts()!=std::vector<std::uint32_t>{1,2,1}||
       d.active_slot_to_coeff_matrix_qp().rescale_counts()!=std::vector<std::uint32_t>{1,1,2}||
       d.plus_i_plaintext.meta.q_count!=27||d.plus_i_plaintext.meta.parms_id!=id||d.plus_i_plaintext.meta.scale!=1.||
       profile.output_level!=26||profile.output_scale_log2!=50||!profile.level_parameters.empty())
        throw std::runtime_error("mixed immutable corrected profile contract");
    // Sparse-low is deliberately a DIFFERENT basis, not Global Q49/P25.
    if(!d.sparse_switch_to_ephemeral||!d.sparse_switch_to_dense||!d.sparse_low_owner||!d.sparse_low_evaluator||
       d.sparse_switch_to_ephemeral->meta.q_count!=2||d.sparse_switch_to_ephemeral->meta.p_count!=2||
       d.sparse_switch_to_ephemeral->meta.degree!=65536||d.sparse_switch_to_ephemeral->meta.component_count!=2)
        throw std::runtime_error("mixed sparse-low Q2/P2 owner/key contract");
    check_key_meta(*d.sparse_switch_to_dense);
    const auto [s,q]=planned_scale(*runtime.api,d.raised_scale_override,d.active_coeff_to_slot_matrix_qp());
    if(q!=45||!Api::same_scale(s,d.eval_mod.input_scale))throw std::runtime_error("mixed actual supplied C2S output/EvalMod input");
    for(const auto &c:d.eval_mod.polynomial_combine_steps)
        if(c.output_q_count!=c.product_q_count||!(c.product_scale>0&&std::isfinite(c.product_scale))||
           !(c.quotient_output_scale>0)||std::log2(c.quotient_output_scale)<60.-1e-10)
            throw std::runtime_error("mixed quotient floor/lazy combine");
}
struct ProfileNumbers {unsigned ratio=0;double eta=0,target=0,alpha=0,m=0,c2s_scale=0;};
inline std::pair<std::unique_ptr<const GpuBootstrapProfile>,ProfileNumbers>
make_profile(Runtime &r,unsigned ratio,const SubmissionOwner &host) {
    host.check();
    if(const char *p=std::getenv("POSEIDON_GPU_SSE_LOW_P");p&&std::string(p)!="2")throw std::invalid_argument("mixed requires sparse-low P2");
    if(ratio!=8192&&ratio!=65536)throw std::invalid_argument("mixed closed profile ratio");
    if(r.context.parameters_literal()->q().size()!=49||r.context.parameters_literal()->p().size()!=25)
        throw std::runtime_error("mixed Q49/P25 fixture required");
    // KeyGenerator/encoder access is serialized on the submission owner.
    const int dev=r.device_id;
    poseidon::gpu::gpu_check_cuda(cudaSetDevice(dev),"mixed profile device");
    GpuResNet20ManualOptions options;options.sine_degree=22;options.double_angle=3;options.k=12;options.log_split=2;
    options.full_coefficients=true;options.sparse_secret_hamming_weight=32;
    options.level_special_modulus=GpuResNet20ManualOptions::LevelSpecialModulus::Global;
    auto profile=GpuBootstrapProfileBuilder::build_resnet20_manual(r.context,r.keys,dev,options);
    if(!profile.level_parameters.empty())throw std::runtime_error("mixed Global profile unexpectedly supplied alternate P tables");
    if(!profile.relin_keys||!profile.galois_keys)throw std::runtime_error("mixed missing owning Boot keys");
    validate_key(*profile.relin_keys);validate_key(*profile.galois_keys);
    auto &d=profile.bootstrap_data;
    if(!d.sparse_switch_to_dense||!d.sparse_switch_to_ephemeral)throw std::runtime_error("mixed absent SSE keys");
    validate_key(*d.sparse_switch_to_dense);GpuUploader::prepare_key_views_for_q_counts(*d.sparse_switch_to_ephemeral,{2});
    ManualC2SPrecision alternate(r.context,r.encoder,*profile.galois_keys,54);
    const auto [low,lowq]=planned_scale(*r.api,std::exp2(40.),d.slot_to_coeff_matrix_qp);
    const auto [c2s,cq]=planned_scale(*r.api,std::exp2(54.),alternate.matrices);
    if(lowq!=2||cq!=45)throw std::runtime_error("mixed physical S2C/C2S endpoint");
    const auto literal=r.context.parameters_literal();
    const double q0=double(static_cast<long double>(literal->q()[0].value())*literal->q()[1].value());
    const double B=std::exp2(std::round(std::log2(q0))),target=std::exp2(std::round(std::log2(q0/ratio)));
    const double m=std::round(target/low),eta=ratio*m*low/B;
    if(m!=(ratio==8192?16.:2.)||!(eta>0&&std::isfinite(eta)))throw std::runtime_error("mixed physical m/eta");
    poseidon::EvalModPoly p(r.context,poseidon::CosDiscrete,std::exp2(50.),44,5,3,12,0,31);p.enable_full_cosine_coefficients();
    auto coefficients=p.sine_poly();auto a=manual_cosine_coefficients_k12_da3();coefficients.data().assign(a.begin(),a.end());
    for(auto &c:coefficients.data())c*=p.sqrt_2pi();coefficients.max_degree()=30;coefficients.is_even()=true;coefficients.is_odd()=false;
    GpuEvalModUploadOptions opt;opt.dynamic_rescale=true;opt.polynomial_log_split=2;opt.flat_bsgs_b8=false;opt.virtual_degree_bound=false;opt.lead_leaf_resplit=false;
    opt.min_quotient_scale_log2=60;
    const auto id=r.parameters.get_level_by_q_count(45).parms_id;
    auto native=GpuUploader::upload_eval_mod_high_precision(p,r.encoder,id,dev,nullptr,poseidon::parms_id_zero,1,
        &coefficients,true,UINT32_MAX,NAN,NAN,true,c2s,true,opt);
    if(native.output_q_count!=27)throw std::runtime_error("mixed native Q27 contract");
    const double alpha=std::exp2(50.)/native.output_scale,seed=std::pow(alpha/eta,1./8);
    if(!(alpha>0&&std::isfinite(alpha)&&seed>0&&std::isfinite(seed)))throw std::runtime_error("mixed physical output fold");
    for(auto &c:coefficients.data())c*=seed;
    auto candidate=GpuUploader::upload_eval_mod_high_precision(p,r.encoder,id,dev,nullptr,poseidon::parms_id_zero,1,
        &coefficients,true,UINT32_MAX,p.sqrt_2pi()*seed,NAN,true,c2s,false,opt);
    if(candidate.output_q_count!=native.output_q_count||!Api::same_scale(candidate.output_scale,native.output_scale))
        throw std::runtime_error("mixed seeded output schedule changed");
    GpuUploader::prepare_key_views_for_q_counts(*profile.relin_keys,candidate.required_relin_q_counts);
    poseidon::Plaintext host_i;r.encoder.encode(std::complex<double>(0,1),candidate.output_parms_id,1.,host_i);
    auto plus_i=GpuUploader::upload_plaintext(host_i,dev);
    gpu_check_cuda(cudaDeviceSynchronize(),"mixed profile construction barrier");
    // The old, unpublished base payload dies only after its uploads complete.
    d.eval_mod=std::move(candidate);d.plus_i_plaintext=std::move(plus_i);alternate.swap_into(d);
    d.q0_over_message_ratio=target;d.output_ratio=ratio;d.output_scale_multiplier=alpha;
    d.slot_to_coeff_input_q_count=27;d.slot_to_coeff_input_parms_id=native.output_parms_id;d.slot_to_coeff_output_scale=std::exp2(50.);
    profile.output_level=26;profile.output_scale_log2=50;profile.profile_id="manual-llama-q27-r"+std::to_string(ratio);
    verify_profile(r,profile,ratio,eta,target,alpha);
    gpu_check_cuda(cudaDeviceSynchronize(),"mixed immutable profile publish");
    ProfileNumbers n{ratio,eta,target,alpha,m,c2s};
    return {std::make_unique<const GpuBootstrapProfile>(std::move(profile)),n};
}
class BoundApi {
    Runtime &runtime;
    const SubmissionOwner &host;
    std::unique_ptr<const GpuBootstrapProfile> profile;
    GpuBootstrapWorkspace workspace;
    Api api;
    ProfileNumbers numbers;
public:
    BoundApi(Runtime &r,const SubmissionOwner &h,std::pair<std::unique_ptr<const GpuBootstrapProfile>,ProfileNumbers> built,std::size_t cache,
             ExpUnitSeedPolicy seeds=ExpUnitSeedPolicy::IntegerPhysicalScale)
        :runtime(r),host(h),profile(std::move(built.first)),api(r.context,r.parameters,r.encoder,r.relin,r.galois,cache,seeds),numbers(built.second) {
        h.check();if(!profile)throw std::runtime_error("mixed absent immutable profile");api.bind_boot(*profile,workspace);verify();
    }
    ~BoundApi(){api.boot_observer={};api.stage_observer={};(void)cudaDeviceSynchronize();}
    BoundApi(const BoundApi&)=delete;BoundApi&operator=(const BoundApi&)=delete;
    BoundApi(BoundApi&&)=delete;BoundApi&operator=(BoundApi&&)=delete;
    Api &get(){host.check();return api;}
    const GpuBootstrapProfile &get_profile() const{return *profile;}
    const ProfileNumbers &config() const{return numbers;}
    void verify() const {host.check();verify_profile(runtime,*profile,numbers.ratio,numbers.eta,numbers.target,numbers.alpha);}
};
inline std::size_t key_bytes(const GpuEvaluationKeyData &key,std::set<const void*> &seen) {
    if(!seen.insert(&key).second)return 0;
    std::size_t bytes=0;for(const auto &f:key.fields_)bytes+=f.buffer.allocated_bytes();
    bytes+=key.galois_key_q0_ptrs.allocated_bytes()+key.galois_key_p0_ptrs.allocated_bytes()+
        key.galois_key_q1_ptrs.allocated_bytes()+key.galois_key_p1_ptrs.allocated_bytes();
    if(key.level_keys)for(const auto &[q,k]:*key.level_keys){(void)q;if(k)bytes+=key_bytes(*k,seen);}
    return bytes;
}
inline std::size_t profile_key_bytes(const GpuBootstrapProfile &p,std::set<const void*> &seen) {
    std::size_t bytes=key_bytes(*p.relin_keys,seen)+key_bytes(*p.galois_keys,seen);
    for(const auto &k:{p.bootstrap_data.sparse_switch_to_ephemeral,p.bootstrap_data.sparse_switch_to_dense})if(k)bytes+=key_bytes(*k,seen);
    return bytes;
}
} // namespace cerium_source::manual_backend
