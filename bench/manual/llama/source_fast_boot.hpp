#pragma once
// Fast per-level special-modulus (P) tables + InversePreRotated Galois keys +
// deduplicated primary(8192)/SiLU(65536) Boot profile construction.
// Opt-in via LLAMA_FAST_P=1 / LLAMA_FAST_BOOT=1 on Mode::Fast.
#include "source_runtime.hpp"
#include "source_manual_boot.hpp"
#include "poseidon/basics/util/ntt.h"
#include "poseidon/basics/util/rlwe.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace cerium_source::fast_boot {
using namespace cerium_source;
using namespace poseidon::gpu;

inline bool fast_p_enabled() {
    const char *v=std::getenv("LLAMA_FAST_P");return v&&std::string(v)=="1";
}
inline bool fast_boot_enabled() {
    const char *v=std::getenv("LLAMA_FAST_BOOT");
    if(v&&*v)return std::string(v)=="1";
    return fast_p_enabled();
}

inline poseidon::SecretKey slice_secret_key(
    const poseidon::SecretKey &src,std::size_t degree,
    std::size_t global_q,std::size_t q_keep,
    const poseidon::PoseidonContext &sub,std::size_t p_keep) {
    poseidon::SecretKey out;
    out.data().resize(sub,sub.crt_context()->key_parms_id(),degree*(q_keep+p_keep));
    std::copy_n(src.data().data(),degree*q_keep,out.data().data());
    std::copy_n(src.data().data()+degree*global_q,degree*p_keep,out.data().data()+degree*q_keep);
    out.parms_id()=sub.crt_context()->key_parms_id();
    return out;
}

struct HostFastRuntimeBand {
    std::size_t top_q=0,p=0;
    std::vector<std::size_t> qs;
    std::vector<int> steps;
    poseidon::PoseidonContext sub;
    poseidon::RelinKeys relin_cpu;
    poseidon::GaloisKeys galois_cpu;
    HostFastRuntimeBand(std::size_t tq,std::size_t pv,std::vector<std::size_t> qv,
                        std::vector<int> sv,poseidon::PoseidonContext sctx)
        :top_q(tq),p(pv),qs(std::move(qv)),steps(std::move(sv)),sub(std::move(sctx)) {}
};

struct HostFastBootBundle {
    std::vector<HostFastRuntimeBand> bands;
    poseidon::LinearMatrixGroup c2s_cpu,s2c_cpu;
    std::shared_ptr<poseidon::GaloisKeys> boot_gk49;
    std::shared_ptr<poseidon::PoseidonContext> sub6_ctx;
    std::shared_ptr<poseidon::GaloisKeys> boot_gk6;
    std::shared_ptr<poseidon::KSwitchKeys> sse_dense_cpu;
    std::shared_ptr<poseidon::ParametersLiteral> sse_low_lit;
    std::shared_ptr<poseidon::PoseidonContext> sse_low_ctx;
    std::shared_ptr<poseidon::KSwitchKeys> sse_eph_cpu;
    double sse_bits=0;
    std::uint32_t sse_hw=32;
};

inline HostFastBootBundle prepare_host_fast_boot_bundle(
    const poseidon::PoseidonContext &context,
    poseidon::KeyGenerator &keys,
    poseidon::CKKSEncoder &encoder,
    const std::vector<int> &app_steps) {
    HostFastBootBundle h;
    const auto &global=*context.parameters_literal();
    const std::size_t global_q=global.q().size(),global_p=global.p().size();
    const std::size_t degree=context.crt_context()->key_context_data()->parms().degree();
    if(global_q!=49||global_p!=25||degree!=65536)throw std::runtime_error("fast Q49/P25 fixture required");

    struct BandSpec {std::size_t top_q,p;std::vector<std::size_t> qs;std::vector<int> steps;};
    std::vector<int> hi_steps;
    for(int s:app_steps) {
        const int a=s<0?-s:s;
        if(a==0||(a<=128&&(a&(a-1))==0))hi_steps.push_back(s);
    }
    if(hi_steps.empty())hi_steps.push_back(0);
    std::vector<BandSpec> specs{
        {6,6,{3,4,5,6},app_steps},
        {10,11,{7,8,9,10},app_steps},
        {13,14,{11,12,13},app_steps},
        {16,16,{14,15,16},app_steps},
        {27,14,{17,18,19,20,21,22,23,24,25,26,27},hi_steps},
    };
    const bool fast_evalmod_p=[]{
        const char *v=std::getenv("LLAMA_FAST_EVALMOD_P");
        return v&&*v&&std::string(v)=="1";
    }();
    if(fast_evalmod_p) {
        specs.push_back({33,17,{28,29,30,31,32,33},{}});
        specs.push_back({37,19,{34,35,36,37},{}});
        specs.push_back({41,21,{38,39,40,41},{}});
        specs.push_back({44,23,{42,43,44},{}});
    }
    for(const auto &sp:specs) {
        poseidon::ParametersLiteral lit=global;
        lit.set_modulus(
            {global.q().begin(),global.q().begin()+static_cast<std::ptrdiff_t>(sp.top_q)},
            {global.p().begin(),global.p().begin()+static_cast<std::ptrdiff_t>(sp.p)});
        poseidon::PoseidonContext sub(lit);
        const poseidon::SecretKey sk=slice_secret_key(keys.secret_key(),degree,global_q,sp.top_q,sub,sp.p);
        poseidon::KeyGenerator sub_kg(sub,sk);
        h.bands.emplace_back(sp.top_q,sp.p,sp.qs,sp.steps,sub);
        auto &dst=h.bands.back();
        sub_kg.create_relin_keys(dst.relin_cpu);
        if(!sp.steps.empty()) {
            const auto gt=sub.crt_context()->galois_tool();
            std::set<std::uint32_t> elts{gt->get_elt_from_step(0)};
            for(int s:sp.steps)elts.insert(gt->get_elt_from_step(s));
            sub_kg.create_galois_keys(std::vector<std::uint32_t>(elts.begin(),elts.end()),dst.galois_cpu);
        }
    }

    const auto t0=Clock::now();
    constexpr double raised_scale=double(std::uint64_t{1}<<54);
    {
        poseidon::EvalModPoly orig(context,poseidon::CosDiscrete,std::exp2(45.),0,5,3,12,0,22);
        const double scaling=orig.q_div()/(orig.k()*orig.sc_fac()*orig.q_diff())*(raised_scale/std::exp2(45.));
        if(std::abs(12*context.crt_context()->q0()*scaling/raised_scale-1)>1e-12)
            throw std::runtime_error("fast C2S integer-period pairing failed");
        poseidon::HomomorphicDFTMatrixLiteral lit(poseidon::encode,16,15,48,{1,1,1},true,scaling,false,1,{},0);
        lit.create_dynamic(h.c2s_cpu,encoder,raised_scale,std::exp2(50.),std::exp2(50.),1.);
    }
    {
        poseidon::HomomorphicDFTMatrixLiteral lit(poseidon::decode,16,15,5,std::vector<std::uint32_t>(3,1),true,1.0,false,1);
        lit.create_dynamic(h.s2c_cpu,encoder,std::exp2(40.),std::exp2(45.),std::exp2(45.),1.0);
    }
    std::cout<<"FAST_BOOT_PHASE dft_cpu="<<seconds(t0)<<"\n";

    {
        const auto gt49=context.crt_context()->galois_tool();
        std::set<std::uint32_t> main_elts{gt49->get_elt_from_step(0)};
        for(int s:h.c2s_cpu.rot_index())main_elts.insert(gt49->get_elt_from_step(s));
        h.boot_gk49=std::make_shared<poseidon::GaloisKeys>();
        keys.create_galois_keys(std::vector<std::uint32_t>(main_elts.begin(),main_elts.end()),*h.boot_gk49);

        poseidon::ParametersLiteral lit6=global;
        lit6.set_modulus({global.q().begin(),global.q().begin()+6},{global.p().begin(),global.p().begin()+6});
        h.sub6_ctx=std::make_shared<poseidon::PoseidonContext>(lit6);
        const auto sk6=slice_secret_key(keys.secret_key(),degree,global_q,6,*h.sub6_ctx,6);
        poseidon::KeyGenerator kg6(*h.sub6_ctx,sk6);
        const auto gt6=h.sub6_ctx->crt_context()->galois_tool();
        std::set<std::uint32_t> elts6{gt6->get_elt_from_step(0)};
        for(int s:h.s2c_cpu.rot_index())elts6.insert(gt6->get_elt_from_step(s));
        h.boot_gk6=std::make_shared<poseidon::GaloisKeys>();
        kg6.create_galois_keys(std::vector<std::uint32_t>(elts6.begin(),elts6.end()),*h.boot_gk6);
    }
    {
        constexpr std::uint32_t hw=32;
        h.sse_hw=hw;
        const auto global_ctx=context.crt_context();
        const auto kp_id=global_ctx->key_parms_id();
        const std::size_t mod_cnt=global_ctx->key_context_data()->parms().coeff_modulus().size();
        poseidon::SecretKey sparse;sparse.data().resize(context,kp_id,degree*mod_cnt);
        poseidon::util::sample_poly_ternary_with_hamming(
            context.random_generator()->create(),context,kp_id,hw,sparse.data().data());
        poseidon::util::RNSIter coeff(sparse.data().data(),degree);
        poseidon::util::ntt_negacyclic_harvey(coeff,mod_cnt,global_ctx->small_ntt_tables());
        sparse.parms_id()=kp_id;

        h.sse_dense_cpu=std::make_shared<poseidon::KSwitchKeys>(keys.create_switch_key(sparse,keys.secret_key()));

        poseidon::ParametersLiteral low_lit=global;
        low_lit.set_modulus({global.q().begin(),global.q().begin()+2},{global.p().begin(),global.p().begin()+2});
        h.sse_low_lit=std::make_shared<poseidon::ParametersLiteral>(low_lit);
        h.sse_low_ctx=std::make_shared<poseidon::PoseidonContext>(low_lit);
        const auto dense_low=slice_secret_key(keys.secret_key(),degree,global_q,2,*h.sse_low_ctx,2);
        const auto sparse_low=slice_secret_key(sparse,degree,global_q,2,*h.sse_low_ctx,2);
        poseidon::KeyGenerator low_kg(*h.sse_low_ctx,dense_low);
        h.sse_eph_cpu=std::make_shared<poseidon::KSwitchKeys>(low_kg.create_switch_key(dense_low,sparse_low));
        double bits=0;for(const auto &qm:h.sse_low_lit->q())bits+=std::log2(double(qm.value()));
        for(const auto &pm:h.sse_low_lit->p())bits+=std::log2(double(pm.value()));
        h.sse_bits=bits;
    }
    return h;
}

inline void install_fast_runtime_levels_from_bundle(Runtime &r,const HostFastBootBundle &h) {
    const int dev=r.device_id;
    gpu_check_cuda(cudaSetDevice(dev),"fast runtime levels device");
    const auto t_start=Clock::now();
    auto relin_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();
    auto galois_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();
    for(const auto &b:h.bands) {
        auto r_gpu=std::make_shared<GpuRelinKeysData>(GpuUploader::upload_relin_keys(b.relin_cpu,dev));
        GpuUploader::prepare_key_views_for_q_counts(*r_gpu,b.qs);
        for(auto q:b.qs)(*relin_map)[q]=r_gpu;
        if(!b.steps.empty()) {
            auto g_gpu=std::make_shared<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(b.galois_cpu,dev));
            GpuUploader::prepare_key_views_for_q_counts(*g_gpu,b.qs);
            for(auto q:b.qs)(*galois_map)[q]=g_gpu;
        }
        GpuParameterData donor(b.sub,dev,b.qs);
        for(auto q:b.qs)r.parameters.adopt_level(donor.take_level_by_q_count(q));
    }
    r.relin.level_keys=std::move(relin_map);
    r.galois.level_keys=std::move(galois_map);
    std::cout<<"FAST_RUNTIME_LEVEL_P installed device="<<dev<<" q3..6=P6 q7..10=P11 q11..13=P14 q14..16=P16 q17..27=P14 InversePreRotated=true seconds="<<seconds(t_start)<<"\n";
}

// Install fast per-level P tables into Runtime::parameters, relin, and galois:
//   q=1..6   -> P=6  (dnum=1, S2C / down mask / proj)
//   q=7..10  -> P=11 (dnum=1, AV / O / SiLU combine / QK; +34b headroom at Q=10)
//   q=11..13 -> P=14 (dnum=1, Linear / QK11 / AV top / Softmax)
//   q=14..16 -> P=16 (dnum=1, Linear in / RMS out / SiLU top)
//   q=17..27 -> P=14 (dnum=2, RMS / Softmax / SiLU top + proj conj)
//   q=28..49 -> P=25 (dnum=2, C2S + EvalMod main context)
inline void install_fast_runtime_levels(
    Runtime &r,const std::vector<int> &app_steps) {
    const int dev=r.device_id;
    gpu_check_cuda(cudaSetDevice(dev),"fast runtime levels device");
    const auto &global=*r.context.parameters_literal();
    const std::size_t global_q=global.q().size();
    const std::size_t degree=r.context.crt_context()->key_context_data()->parms().degree();
    struct Band {std::size_t top_q,p;std::vector<std::size_t> qs;std::vector<int> steps;};
    std::vector<int> hi_steps;
    for(int s:app_steps) {
        const int a=s<0?-s:s;
        if(a==0||(a<=128&&(a&(a-1))==0))hi_steps.push_back(s);
    }
    if(hi_steps.empty())hi_steps.push_back(0);
    std::vector<Band> bands{
        {6,6,{3,4,5,6},app_steps},
        {10,11,{7,8,9,10},app_steps},
        {13,14,{11,12,13},app_steps},
        {16,16,{14,15,16},app_steps},
        {27,14,{17,18,19,20,21,22,23,24,25,26,27},hi_steps},
    };
    const bool fast_evalmod_p=[]{
        const char *v=std::getenv("LLAMA_FAST_EVALMOD_P");
        return v&&*v&&std::string(v)=="1";
    }();
    if(fast_evalmod_p) {
        bands.push_back({33,17,{28,29,30,31,32,33},{}});
        bands.push_back({37,19,{34,35,36,37},{}});
        bands.push_back({41,21,{38,39,40,41},{}});
        bands.push_back({44,23,{42,43,44},{}});
    }
    const auto t_start=Clock::now();
    auto relin_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();
    auto galois_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();
    for(const auto &b:bands) {
        poseidon::ParametersLiteral lit=global;
        lit.set_modulus(
            {global.q().begin(),global.q().begin()+static_cast<std::ptrdiff_t>(b.top_q)},
            {global.p().begin(),global.p().begin()+static_cast<std::ptrdiff_t>(b.p)});
        const poseidon::PoseidonContext sub(lit);
        const poseidon::SecretKey sk=slice_secret_key(r.keys.secret_key(),degree,global_q,b.top_q,sub,b.p);
        poseidon::KeyGenerator sub_kg(sub,sk);
        {
            poseidon::RelinKeys r_cpu;sub_kg.create_relin_keys(r_cpu);
            auto r_gpu=std::make_shared<GpuRelinKeysData>(GpuUploader::upload_relin_keys(r_cpu,dev));
            GpuUploader::prepare_key_views_for_q_counts(*r_gpu,b.qs);
            for(auto q:b.qs)(*relin_map)[q]=r_gpu;
        }
        if(!b.steps.empty()) {
            const auto gt=sub.crt_context()->galois_tool();
            std::set<std::uint32_t> elts{gt->get_elt_from_step(0)};
            for(int s:b.steps)elts.insert(gt->get_elt_from_step(s));
            poseidon::GaloisKeys g_cpu;
            sub_kg.create_galois_keys(std::vector<std::uint32_t>(elts.begin(),elts.end()),g_cpu);
            auto g_gpu=std::make_shared<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(g_cpu,dev));
            GpuUploader::prepare_key_views_for_q_counts(*g_gpu,b.qs);
            for(auto q:b.qs)(*galois_map)[q]=g_gpu;
        }
        GpuParameterData donor(sub,dev,b.qs);
        for(auto q:b.qs)r.parameters.adopt_level(donor.take_level_by_q_count(q));
    }
    r.relin.level_keys=std::move(relin_map);
    r.galois.level_keys=std::move(galois_map);
    std::cout<<"FAST_RUNTIME_LEVEL_P installed device="<<dev<<" q3..6=P6 q7..10=P11 q11..13=P14 q14..16=P16 q17..27=P14 InversePreRotated=true seconds="<<seconds(t_start)<<"\n";
}

// Verify profile contract while permitting fast per-level P keys and shared keys.
inline void verify_fast_profile(Runtime &runtime,const GpuBootstrapProfile &profile,
    unsigned ratio,double eta,double target,double alpha) {
    const auto &d=profile.bootstrap_data;const auto id=runtime.parameters.get_level_by_q_count(27).parms_id;
    if((ratio!=8192&&ratio!=65536)||!(eta>0&&std::isfinite(eta))||
       d.output_ratio!=ratio||d.q0_over_message_ratio!=target||d.output_scale_multiplier!=alpha||
       !d.slot_to_coeff_first||!d.project_real||d.allow_environment_linear_transform_override||
       d.linear_transform_mode!=GpuLinearTransformMode::DoubleHoistBsgs||d.input_q_count!=6||d.input_scale!=std::exp2(40.)||
       d.raised_scale_override!=std::exp2(54.)||d.post_raise_integer_multiplier!=1||d.post_raise_scale_multiplier!=1.||!d.post_raise_plaintext.empty()||
       d.slot_to_coeff_input_q_count!=27||d.slot_to_coeff_input_parms_id!=id||d.slot_to_coeff_output_scale!=std::exp2(50.)||
       d.eval_mod.output_q_count!=27||d.eval_mod.output_parms_id!=id||d.eval_mod.target_scale!=std::exp2(50.)||
       d.eval_mod.polynomial_degree!=30||!d.eval_mod.dynamic_rescale||d.eval_mod.double_angle_constants.size()!=3||
       d.coeff_to_slot_matrix_qp.rescale_counts()!=std::vector<std::uint32_t>{1,2,1}||
       d.slot_to_coeff_matrix_qp.rescale_counts()!=std::vector<std::uint32_t>{1,1,2}||
       d.plus_i_plaintext.meta.q_count!=27||d.plus_i_plaintext.meta.parms_id!=id||d.plus_i_plaintext.meta.scale!=1.||
       profile.output_level!=26||profile.output_scale_log2!=50)
        throw std::runtime_error("fast immutable corrected profile contract");
    if(!d.sparse_switch_to_ephemeral||!d.sparse_switch_to_dense||!d.sparse_low_owner||!d.sparse_low_evaluator||
       d.sparse_switch_to_ephemeral->meta.q_count!=2||d.sparse_switch_to_ephemeral->meta.p_count!=2||
       d.sparse_switch_to_ephemeral->meta.degree!=65536||d.sparse_switch_to_ephemeral->meta.component_count!=2)
        throw std::runtime_error("fast sparse-low Q2/P2 contract");
    const auto [s,q]=manual_backend::planned_scale(*runtime.api,d.raised_scale_override,d.coeff_to_slot_matrix_qp);
    if(q!=45||!Api::same_scale(s,d.eval_mod.input_scale))throw std::runtime_error("fast actual C2S output/EvalMod input");
    for(const auto &c:d.eval_mod.polynomial_combine_steps)
        if(c.output_q_count!=c.product_q_count||!(c.product_scale>0&&std::isfinite(c.product_scale))||
           !(c.quotient_output_scale>0)||std::log2(c.quotient_output_scale)<60.-1e-10)
            throw std::runtime_error("fast quotient floor/lazy combine");
}

inline void validate_fast_runtime_keys(Runtime &r) {
    if(!r.relin.level_keys||r.relin.level_keys->empty()||!r.galois.level_keys||r.galois.level_keys->empty())
        throw std::runtime_error("fast mode expected level_keys on Runtime");
    for(std::size_t q=3;q<=27;++q) {
        const auto &lvl=r.parameters.get_level_by_q_count(q);
        const std::size_t p_lvl=lvl.shards.front().hybrid_base_p_count;
        const auto &rk=r.relin.for_level(q);
        const auto &gk=r.galois.for_level(q);
        if(rk.meta.p_count!=p_lvl||gk.meta.p_count!=p_lvl||gk.meta.galois_format!=GpuGaloisKeyFormat::InversePreRotated)
            throw std::runtime_error("fast level key P/format mismatch at q="+std::to_string(q));
    }
}

namespace detail {
struct SparseLowHolder {
    poseidon::ParametersLiteral parameters;
    poseidon::PoseidonContext context;
    std::unique_ptr<GpuParameterData> gpu_parameters;
    std::unique_ptr<GpuEvaluator> evaluator;
    explicit SparseLowHolder(const poseidon::ParametersLiteral &lit):parameters(lit),context(parameters){}
};
}

// Construct both ratio=8192 and ratio=65536 Boot profiles on r.device_id
// from either a shared HostFastBootBundle (no duplicate KeyGenerator/DFT) or fresh CPU work.
inline std::array<std::pair<std::unique_ptr<const GpuBootstrapProfile>,manual_backend::ProfileNumbers>,2>
make_fast_profile_pair(Runtime &r,const manual_backend::SubmissionOwner &host,
                       const HostFastBootBundle *shared=nullptr) {
    host.check();
    const int dev=r.device_id;
    gpu_check_cuda(cudaSetDevice(dev),"make fast profile pair device");
    const auto t0=Clock::now();
    if(const char *p=std::getenv("POSEIDON_GPU_SSE_LOW_P");p&&std::string(p)!="2")
        throw std::invalid_argument("fast boot requires sparse-low P2");
    const auto &global=*r.context.parameters_literal();
    const std::size_t global_q=global.q().size(),global_p=global.p().size();
    const std::size_t degree=r.context.crt_context()->key_context_data()->parms().degree();
    if(global_q!=49||global_p!=25||degree!=65536)throw std::runtime_error("fast Q49/P25 fixture required");

    constexpr double raised_scale=double(std::uint64_t{1}<<54);
    poseidon::LinearMatrixGroup local_c2s,local_s2c;
    const poseidon::LinearMatrixGroup *c2s_ptr=shared?&shared->c2s_cpu:&local_c2s;
    const poseidon::LinearMatrixGroup *s2c_ptr=shared?&shared->s2c_cpu:&local_s2c;
    if(!shared) {
        poseidon::EvalModPoly orig(r.context,poseidon::CosDiscrete,std::exp2(45.),0,5,3,12,0,22);
        const double scaling=orig.q_div()/(orig.k()*orig.sc_fac()*orig.q_diff())*(raised_scale/std::exp2(45.));
        if(std::abs(12*r.context.crt_context()->q0()*scaling/raised_scale-1)>1e-12)
            throw std::runtime_error("fast C2S integer-period pairing failed");
        poseidon::HomomorphicDFTMatrixLiteral lit(poseidon::encode,16,15,48,{1,1,1},true,scaling,false,1,{},0);
        lit.create_dynamic(local_c2s,r.encoder,raised_scale,std::exp2(50.),std::exp2(50.),1.);
        poseidon::HomomorphicDFTMatrixLiteral lit_s(poseidon::decode,16,15,5,std::vector<std::uint32_t>(3,1),true,1.0,false,1);
        lit_s.create_dynamic(local_s2c,r.encoder,std::exp2(40.),std::exp2(45.),std::exp2(45.),1.0);
        std::cout<<"FAST_BOOT_PHASE dft_cpu="<<seconds(t0)<<"\n";
    }
    const auto &c2s_cpu=*c2s_ptr;
    const auto &s2c_cpu=*s2c_ptr;

    poseidon::EvalModPoly p(r.context,poseidon::CosDiscrete,std::exp2(50.),44,5,3,12,0,31);
    p.enable_full_cosine_coefficients();
    auto base_coeffs=p.sine_poly();
    {
        auto a=manual_cosine_coefficients_k12_da3();
        base_coeffs.data().assign(a.begin(),a.end());
        for(auto &c:base_coeffs.data())c*=p.sqrt_2pi();
        base_coeffs.max_degree()=30;base_coeffs.is_even()=true;base_coeffs.is_odd()=false;
    }
    const auto id45=r.parameters.get_level_by_q_count(45).parms_id;
    auto boot_relin=std::shared_ptr<const GpuRelinKeysData>(&r.relin,[](const GpuRelinKeysData*){});

    std::shared_ptr<GpuGaloisKeysData> boot_galois;
    if(shared) {
        boot_galois=std::make_shared<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(*shared->boot_gk49,dev));
        GpuUploader::prepare_key_views_for_q_counts(*boot_galois,{45,46,48,49});
        auto level_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();
        auto g6=std::make_shared<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(*shared->boot_gk6,dev));
        GpuUploader::prepare_key_views_for_q_counts(*g6,{4,5,6});
        (*level_map)[4]=g6;(*level_map)[5]=g6;(*level_map)[6]=g6;
        if(!r.galois.level_keys||!r.galois.level_keys->count(27))
            throw std::runtime_error("fast boot requires Runtime P=14 key at q=27");
        (*level_map)[27]=r.galois.level_keys->at(27);
        boot_galois->level_keys=std::move(level_map);
    } else {
        const auto gt49=r.context.crt_context()->galois_tool();
        std::set<std::uint32_t> main_elts{gt49->get_elt_from_step(0)};
        for(int s:c2s_cpu.rot_index())main_elts.insert(gt49->get_elt_from_step(s));
        poseidon::GaloisKeys gk49;
        r.keys.create_galois_keys(std::vector<std::uint32_t>(main_elts.begin(),main_elts.end()),gk49);
        boot_galois=std::make_shared<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(gk49,dev));
        GpuUploader::prepare_key_views_for_q_counts(*boot_galois,{45,46,48,49});

        auto level_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();
        poseidon::ParametersLiteral lit6=global;
        lit6.set_modulus({global.q().begin(),global.q().begin()+6},{global.p().begin(),global.p().begin()+6});
        const poseidon::PoseidonContext sub6(lit6);
        const auto sk6=slice_secret_key(r.keys.secret_key(),degree,global_q,6,sub6,6);
        poseidon::KeyGenerator kg6(sub6,sk6);
        const auto gt6=sub6.crt_context()->galois_tool();
        std::set<std::uint32_t> elts6{gt6->get_elt_from_step(0)};
        for(int s:s2c_cpu.rot_index())elts6.insert(gt6->get_elt_from_step(s));
        poseidon::GaloisKeys gk6;kg6.create_galois_keys(std::vector<std::uint32_t>(elts6.begin(),elts6.end()),gk6);
        auto g6=std::make_shared<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(gk6,dev));
        GpuUploader::prepare_key_views_for_q_counts(*g6,{4,5,6});
        (*level_map)[4]=g6;(*level_map)[5]=g6;(*level_map)[6]=g6;
        if(!r.galois.level_keys||!r.galois.level_keys->count(27))
            throw std::runtime_error("fast boot requires Runtime P=14 key at q=27");
        (*level_map)[27]=r.galois.level_keys->at(27);
        boot_galois->level_keys=std::move(level_map);
    }
    std::cout<<"FAST_BOOT_PHASE device="<<dev<<" keys_done="<<seconds(t0)<<"\n";

    std::shared_ptr<const GpuEvaluationKeyData> sse_eph,sse_dense;
    const GpuEvaluator *sse_eval=nullptr;
    std::shared_ptr<const void> sse_owner;
    {
        auto up_sw=[dev](const poseidon::KSwitchKeys &k){
            poseidon::RelinKeys rel;static_cast<poseidon::KSwitchKeys&>(rel)=k;
            return std::make_shared<GpuEvaluationKeyData>(GpuUploader::upload_relin_keys(rel,dev));
        };
        if(shared) {
            sse_dense=up_sw(*shared->sse_dense_cpu);
            auto holder=std::make_shared<detail::SparseLowHolder>(*shared->sse_low_lit);
            sse_eph=up_sw(*shared->sse_eph_cpu);
            GpuUploader::prepare_key_views_for_q_counts(*sse_eph,{2});
            holder->gpu_parameters=std::make_unique<GpuParameterData>(holder->context,dev,std::vector<std::size_t>{2});
            holder->evaluator=std::make_unique<GpuEvaluator>(*holder->gpu_parameters);
            std::cout<<"GPU_BOOT_SSE device="<<dev<<" dense_to_ephemeral_modulus=low q=2 p=2 bits="<<shared->sse_bits<<" ephemeral_h="<<shared->sse_hw<<"\n";
            sse_eval=holder->evaluator.get();
            sse_owner=std::move(holder);
        } else {
            constexpr std::uint32_t hw=32;
            const auto global_ctx=r.context.crt_context();
            const auto kp_id=global_ctx->key_parms_id();
            const std::size_t mod_cnt=global_ctx->key_context_data()->parms().coeff_modulus().size();
            poseidon::SecretKey sparse;sparse.data().resize(r.context,kp_id,degree*mod_cnt);
            poseidon::util::sample_poly_ternary_with_hamming(
                r.context.random_generator()->create(),r.context,kp_id,hw,sparse.data().data());
            poseidon::util::RNSIter coeff(sparse.data().data(),degree);
            poseidon::util::ntt_negacyclic_harvey(coeff,mod_cnt,global_ctx->small_ntt_tables());
            sparse.parms_id()=kp_id;
            sse_dense=up_sw(r.keys.create_switch_key(sparse,r.keys.secret_key()));
            poseidon::ParametersLiteral low_lit=global;
            low_lit.set_modulus({global.q().begin(),global.q().begin()+2},{global.p().begin(),global.p().begin()+2});
            auto holder=std::make_shared<detail::SparseLowHolder>(low_lit);
            const auto dense_low=slice_secret_key(r.keys.secret_key(),degree,global_q,2,holder->context,2);
            const auto sparse_low=slice_secret_key(sparse,degree,global_q,2,holder->context,2);
            poseidon::KeyGenerator low_kg(holder->context,dense_low);
            sse_eph=up_sw(low_kg.create_switch_key(dense_low,sparse_low));
            GpuUploader::prepare_key_views_for_q_counts(*sse_eph,{2});
            holder->gpu_parameters=std::make_unique<GpuParameterData>(holder->context,dev,std::vector<std::size_t>{2});
            holder->evaluator=std::make_unique<GpuEvaluator>(*holder->gpu_parameters);
            double bits=0;for(const auto &qm:holder->parameters.q())bits+=std::log2(double(qm.value()));
            for(const auto &pm:holder->parameters.p())bits+=std::log2(double(pm.value()));
            std::cout<<"GPU_BOOT_SSE device="<<dev<<" dense_to_ephemeral_modulus=low q=2 p=2 bits="<<bits<<" ephemeral_h="<<hw<<"\n";
            sse_eval=holder->evaluator.get();
            sse_owner=std::move(holder);
        }
    }

    const auto build_one=[&](unsigned ratio) -> std::pair<std::unique_ptr<const GpuBootstrapProfile>,manual_backend::ProfileNumbers> {
        GpuBootstrapProfile prof;
        prof.cuda_device_id=dev;prof.input_level_min=5;prof.input_level_max=5;prof.input_components=2;
        prof.input_scale_log2=40;prof.output_level=26;prof.output_scale_log2=50;prof.output_components=2;
        prof.relin_keys=boot_relin;prof.galois_keys=boot_galois;
        auto &d=prof.bootstrap_data;
        d.linear_transform_mode=GpuLinearTransformMode::DoubleHoistBsgs;
        d.allow_environment_linear_transform_override=false;
        d.slot_to_coeff_first=true;d.input_q_count=6;d.input_scale=std::exp2(40.);
        d.raised_scale_override=raised_scale;d.project_real=true;d.output_ratio=ratio;
        d.slot_to_coeff_output_scale=std::exp2(50.);
        d.q0_parms_id=r.context.crt_context()->parms_id_map().at(1);
        d.sparse_switch_to_ephemeral=sse_eph;d.sparse_switch_to_dense=sse_dense;
        d.sparse_low_evaluator=sse_eval;d.sparse_low_owner=sse_owner;

        d.coeff_to_slot_matrix_qp=GpuUploader::upload_linear_matrix_group_qp(c2s_cpu,r.context,dev,1,true);
        d.slot_to_coeff_matrix_qp=GpuUploader::upload_linear_matrix_group_qp(s2c_cpu,r.context,dev,1,true);
        for(const auto &m:d.coeff_to_slot_matrix_qp.data())d.coeff_to_slot_groups_qp.push_back(GpuUploader::upload_double_hoist_group_views(m,dev));
        for(const auto &m:d.slot_to_coeff_matrix_qp.data())d.slot_to_coeff_groups_qp.push_back(GpuUploader::upload_double_hoist_group_views(m,dev));

        const auto [low,lowq]=manual_backend::planned_scale(*r.api,std::exp2(40.),d.slot_to_coeff_matrix_qp);
        const auto [c2s_s,cq]=manual_backend::planned_scale(*r.api,raised_scale,d.coeff_to_slot_matrix_qp);
        if(lowq!=2||cq!=45)throw std::runtime_error("fast S2C/C2S endpoint mismatch");
        const double q0=double(static_cast<long double>(global.q()[0].value())*global.q()[1].value());
        const double B=std::exp2(std::round(std::log2(q0))),target=std::exp2(std::round(std::log2(q0/ratio)));
        const double m=std::round(target/low),eta=ratio*m*low/B;
        if(m!=(ratio==8192?16.:2.)||!(eta>0&&std::isfinite(eta)))throw std::runtime_error("fast m/eta mismatch");
        d.q0_over_message_ratio=target;

        GpuEvalModUploadOptions opt;opt.dynamic_rescale=true;opt.polynomial_log_split=2;opt.flat_bsgs_b8=false;
        opt.virtual_degree_bound=false;opt.lead_leaf_resplit=false;opt.min_quotient_scale_log2=60;
        const auto native=GpuUploader::upload_eval_mod_high_precision(p,r.encoder,id45,dev,nullptr,poseidon::parms_id_zero,1,
            &base_coeffs,true,UINT32_MAX,NAN,NAN,true,c2s_s,true,opt);
        if(native.output_q_count!=27)throw std::runtime_error("fast Q27 contract");
        const double alpha=std::exp2(50.)/native.output_scale,seed=std::pow(alpha/eta,1./8);
        auto seeded=base_coeffs;for(auto &c:seeded.data())c*=seed;
        auto candidate=GpuUploader::upload_eval_mod_high_precision(p,r.encoder,id45,dev,nullptr,poseidon::parms_id_zero,1,
            &seeded,true,UINT32_MAX,p.sqrt_2pi()*seed,NAN,true,c2s_s,false,opt);
        if(candidate.output_q_count!=native.output_q_count||!Api::same_scale(candidate.output_scale,native.output_scale))
            throw std::runtime_error("fast seeded EvalMod changed");
        GpuUploader::prepare_key_views_for_q_counts(*boot_relin,candidate.required_relin_q_counts);
        d.output_scale_multiplier=alpha;
        d.slot_to_coeff_input_q_count=27;d.slot_to_coeff_input_parms_id=native.output_parms_id;

        poseidon::Plaintext h_mi,h_pi;
        r.encoder.encode(std::complex<double>(0,-1),id45,1.,h_mi);
        r.encoder.encode(std::complex<double>(0,1),candidate.output_parms_id,1.,h_pi);
        d.minus_i_plaintext=GpuUploader::upload_plaintext(h_mi,dev);
        d.plus_i_plaintext=GpuUploader::upload_plaintext(h_pi,dev);
        d.eval_mod=std::move(candidate);
        prof.profile_id="fast-llama-q27-r"+std::to_string(ratio);
        verify_fast_profile(r,prof,ratio,eta,target,alpha);
        return {std::make_unique<const GpuBootstrapProfile>(std::move(prof)),manual_backend::ProfileNumbers{ratio,eta,target,alpha,m,c2s_s}};
    };
    auto out=std::array{build_one(8192),build_one(65536)};
    gpu_check_cuda(cudaSetDevice(dev),"fast profile pair sync device");
    gpu_check_cuda(cudaDeviceSynchronize(),"fast profile pair ready");
    std::cout<<"FAST_BOOT_PAIR device="<<dev<<" setup_seconds="<<seconds(t0)<<" per_level_P=true shared_keys=true\n";
    return out;
}
} // namespace cerium_source::fast_boot
