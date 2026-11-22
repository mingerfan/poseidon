// SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Source-order single-lane QK/AV: llama_cerium.py:191-238 and636-682.
#pragma once
#include "source_ops.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <optional>

namespace cerium_source {
inline bool fast_attention_enabled() {
    const char *v=std::getenv("LLAMA_FAST_ATTENTION");return v&&std::string(v)=="1";
}
inline void accumulate(Api &api,std::optional<CT> &sum,CT term) {
    if(sum)*sum=api.add(*sum,term);else sum=std::move(term);
}
inline std::vector<CT> qk_pair_fast(Api &api,const std::array<const CT*,2> &Q,const CT &K) {
    static const std::vector<int> babies={0,256,512,768,1024,1280,1536,1792};
    static const std::vector<int> giants={0,-2048,-4096,-6144,-8192,-10240,-12288,-14336,-16384,-18432,-20480,-22528,-24576,-26624,-28672,-30720};
    static const std::vector<int> d_steps={-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7};
    static const std::vector<std::vector<double>> mod8_masks=[]{
        std::vector<std::vector<double>> m(8,std::vector<double>(32768,0.));
        for(std::size_t b=0;b<8;++b)for(std::size_t j=b;j<32768;j+=8)m[b][j]=1.;
        return m;
    }();
    static const std::vector<double> lt8_vec=[]{
        std::vector<double> v(32768,0.);
        const double inv=1./(80.*std::sqrt(128.));
        for(std::size_t j=0;j<32768;++j)if((j%128)<8)v[j]=inv;
        return v;
    }();
    const auto q=std::min({Q[0]->meta.q_count,Q[1]->meta.q_count,K.meta.q_count});
    static const std::size_t work_q=[]{
        const char *v=std::getenv("LLAMA_FAST_QK_Q");
        if(v&&*v){const auto val=std::stoul(v);if(val==10||val==11||val==12)return std::size_t(val);}
        return std::size_t(10);
    }();
    if(q<work_q)throw std::runtime_error("fast QK insufficient Q budget");
    auto K12=api.drop(K,work_q);
    auto bs=api.hoisted(K12,babies);
    const double kv_mask_scale=api.prime(work_q);
    std::array<std::shared_ptr<poseidon::gpu::GpuPlaintextData>,8> kv_masks;
    for(int b=0;b<8;++b)kv_masks[b]=api.cached_static_vector(K12,mod8_masks[b],kv_mask_scale);
    std::array<std::optional<CT>,15> kv_acc;
    for(int b=0;b<8;++b) {
        std::vector<int> b_d;
        for(int d=-b;d<=7-b;++d)if(d!=0)b_d.push_back(d);
        auto shifted=api.hoisted(bs[b],b_d);
        std::size_t si=0;
        for(int d=-b;d<=7-b;++d) {
            const CT &src=(d==0)?bs[b]:shifted[si++];
            accumulate(api,kv_acc[std::size_t(d+7)],api.cp_prepared(src,*kv_masks[b]));
        }
    }
    std::array<CT,15> KV;
    for(std::size_t i=0;i<15;++i)KV[i]=api.rescale(*kv_acc[i],1);
    const std::size_t q_work=work_q-1;
    const double post_mul_scale=KV[0].meta.scale*Q[0]->meta.scale/api.prime(q_work);
    auto mask_lt8=api.cached_static_vector(api.drop(KV[0],q_work-1),lt8_vec,
        std::exp2(40.)*api.removed(q_work-1,2)/post_mul_scale);
    auto q0_drop=api.drop(*Q[0],q_work),q1_drop=api.drop(*Q[1],q_work);
    auto gs=api.hoisted_pair(q0_drop,q1_drop,giants);
    std::array<std::optional<CT>,2> results;
    for(int g=0;g<16;++g) {
        auto q_ds=api.hoisted_pair(gs[0][g],gs[1][g],d_steps);
        std::optional<CT> group0,group1;
        for(int d=-7;d<=7;++d) {
            const std::size_t idx=std::size_t(d<0?(d+7):(d+6));
            const CT &qd0=(d==0)?gs[0][g]:q_ds[0][idx];
            const CT &qd1=(d==0)?gs[1][g]:q_ds[1][idx];
            auto prod=api.mul_pair(KV[std::size_t(d+7)],qd0,KV[std::size_t(d+7)],qd1);
            if(group0) {
                auto acc=api.add_pair(*group0,prod[0],*group1,prod[1]);
                group0=std::move(acc[0]);group1=std::move(acc[1]);
            } else {
                group0=std::move(prod[0]);group1=std::move(prod[1]);
            }
        }
        auto rel=api.relin_pair(*group0,*group1);
        auto sum_g=api.rescale_pair(rel[0],rel[1],1);
        for(int j:{8,16,32,64}) {
            auto rot=api.rotate_pair(sum_g[0],sum_g[1],j);
            sum_g=api.add_pair(sum_g[0],rot[0],sum_g[1],rot[1]);
        }
        auto masked=api.cp_prepared_pair(sum_g[0],sum_g[1],*mask_lt8);
        if(g)masked=api.rotate_pair(masked[0],masked[1],(2048-8)*g);
        if(results[0]) {
            auto acc=api.add_pair(*results[0],masked[0],*results[1],masked[1]);
            results[0]=std::move(acc[0]);results[1]=std::move(acc[1]);
        } else {
            results[0]=std::move(masked[0]);results[1]=std::move(masked[1]);
        }
    }
    auto rescaled=api.rescale_pair(*results[0],*results[1],2);
    api.require_scale(rescaled[0],std::exp2(40.));
    api.require_scale(rescaled[1],std::exp2(40.));
    std::vector<CT> outputs;outputs.reserve(2);
    outputs.push_back(std::move(rescaled[0]));
    outputs.push_back(std::move(rescaled[1]));
    return outputs;
}
inline std::vector<CT> qk_pair(Api &api,const std::array<const CT*,2> &Q,const CT &K) {
    if(fast_attention_enabled())return qk_pair_fast(api,Q,K);
    std::vector<int> babies,giants;
    for(int i=0;i<8;++i)babies.push_back(256*i);
    for(int i=0;i<16;++i)giants.push_back(-2048*i);
    // Original inputs have a higher scale before one post-rotation Rescale.
    // Interface conversion preserves represented values and actual metadata.
    const auto q=std::min({Q[0]->meta.q_count,Q[1]->meta.q_count,K.meta.q_count});
    if(q<11)throw std::runtime_error("source QK input budget");
    auto kr=api.cp_to(api.drop(K,q),1.,std::exp2(60.)*api.prime(q),0);
    auto bs=api.hoisted(kr,babies);
    for(auto &v:bs)v=api.rescale(v,1);
    std::vector<double> selector(32768);
    for(std::size_t j=0;j<32768;j+=128)selector[j]=1./(80.*std::sqrt(128.));
    auto mask=api.cached_vector(bs[0],selector,
        std::exp2(40.)*api.removed(bs[0].meta.q_count,4)/std::exp2(120.));
    std::vector<CT> outputs;
    for(const auto *query:Q) {
    auto qr=api.cp_to(api.drop(*query,q),1.,std::exp2(60.)*api.prime(q),0);
    auto gs=api.hoisted(qr,giants);
    for(auto &v:gs)v=api.rescale(v,1);
    std::optional<CT> result;
    for(int g=0;g<16;++g) {
        std::optional<CT> group;
        for(int b=0;b<8;++b) {
            auto product=api.mul_r(bs[b],gs[g]);
            for(int j=1;j<128;j*=2)product=api.add(product,api.rotate(product,j));
            auto term=api.cp_prepared(product,*mask);
            accumulate(api,group,api.rotate(term,-b));
        }
        accumulate(api,result,api.rotate(*group,(2048-8)*g));
    }
    outputs.push_back(api.rescale(*result,4));
    }
    return outputs;
}
inline std::vector<CT> av_four_fast(Api &api,const std::array<const CT*,4> &probabilities_times_two,const CT &V) {
    static const bool av_timing=(std::getenv("LLAMA_AV_STAGE_TIMING")&&std::string(std::getenv("LLAMA_AV_STAGE_TIMING"))=="1");
    auto ts=[&]{if(av_timing)cudaDeviceSynchronize();return std::chrono::steady_clock::now();};
    const auto t0=ts();
    static const std::vector<int> babies={0,256,512,768,1024,1280,1536,1792};
    static const std::vector<int> giants={0,-2040,-4080,-6120,-8160,-10200,-12240,-14280,-16320,-18360,-20400,-22440,-24480,-26520,-28560,-30600};
    static const std::vector<int> lr_steps={0,1,2,3,4,5,6,7,-128,-127,-126,-125,-124,-123,-122,-121};
    static const std::vector<std::vector<double>> diag_selectors=[]{
        std::vector<std::vector<double>> s(128,std::vector<double>(32768,0.));
        for(std::size_t off=0;off<128;++off)for(std::size_t j=off;j<32768;j+=128)s[off][j]=1.;
        return s;
    }();
    static const std::vector<std::vector<double>> left_vecs=[]{
        std::vector<std::vector<double>> v(128,std::vector<double>(32768,0.));
        const double val=1./(2.*1.0001);
        for(int off=0;off<128;++off)for(std::size_t j=0;j<32768;++j)if(int(j%128)<128-off)v[off][j]=val;
        return v;
    }();
    static const std::vector<std::vector<double>> right_vecs=[]{
        std::vector<std::vector<double>> v(128,std::vector<double>(32768,0.));
        const double val=1./(2.*1.0001);
        for(int off=0;off<128;++off)for(std::size_t j=0;j<32768;++j)if(int(j%128)>=128-off)v[off][j]=val;
        return v;
    }();
    constexpr std::size_t p_q=10;
    if(V.meta.q_count<p_q+2)throw std::runtime_error("fast AV V Q budget");
    auto V12=api.drop(V,p_q+2);
    auto value_babies=api.hoisted(V12,babies);
    const double sel_scale=api.prime(p_q-1)*api.removed(p_q+2,2)/V12.meta.scale;
    std::optional<CT> diagonal;
    for(int g=0;g<16;++g) {
        std::optional<CT> group;
        for(int b=0;b<8;++b) {
            auto sel=api.cached_static_vector(V12,diag_selectors[8*g+b],sel_scale);
            accumulate(api,group,api.cp_prepared(value_babies[b],*sel));
        }
        accumulate(api,diagonal,g?api.rotate(*group,2048*g):std::move(*group));
    }
    auto diag_bs=api.hoisted(*diagonal,babies);
    for(auto &v:diag_bs)v=api.rescale(v,2);
    const auto t1=ts();
    const auto &shape_raw=*probabilities_times_two[0];
    if(shape_raw.meta.q_count<p_q)throw std::runtime_error("fast AV prob Q budget");
    auto shape=api.drop(shape_raw,p_q);
    const double mask_scale=std::exp2(40.)*api.prime(p_q)/shape.meta.scale;
    std::vector<std::shared_ptr<poseidon::gpu::GpuPlaintextData>> left_masks,right_masks;
    left_masks.reserve(128);right_masks.reserve(128);
    for(int off=0;off<128;++off) {
        left_masks.push_back(api.cached_static_vector(shape,left_vecs[off],mask_scale));
        right_masks.push_back(api.cached_static_vector(shape,right_vecs[off],mask_scale));
    }
    const auto t2=ts();
    double prob_gs_ms=0,lr_ms=0,mul_ms=0,relin_ms=0;
    std::vector<CT> dropped_probs;dropped_probs.reserve(4);
    for(const auto *prob:probabilities_times_two) {
        if(prob->meta.q_count!=shape_raw.meta.q_count||!Api::same_scale(prob->meta.scale,shape_raw.meta.scale))
            throw std::runtime_error("fast AV batch input contract");
        dropped_probs.push_back(api.drop(*prob,p_q));
    }
    const auto tp0=ts();
    auto prob_gs_all=api.hoisted_batch({&dropped_probs[0],&dropped_probs[1],&dropped_probs[2],&dropped_probs[3]},giants);
    const auto tp1=ts();prob_gs_ms+=std::chrono::duration<double,std::milli>(tp1-tp0).count();
    std::array<std::optional<CT>,4> acc_outputs;
    for(int g=0;g<16;++g) {
        const auto tl0=ts();
        auto lr_all=api.hoisted_batch({&prob_gs_all[0][g],&prob_gs_all[1][g],&prob_gs_all[2][g],&prob_gs_all[3][g]},lr_steps);
        const auto tl1=ts();lr_ms+=std::chrono::duration<double,std::milli>(tl1-tl0).count();
        std::array<std::optional<CT>,4> groups;
        const auto tm0=ts();
        for(std::size_t lane=0;lane<4;++lane) {
            for(int b=0;b<8;++b) {
                auto l=api.cp_prepared(lr_all[lane][b],*left_masks[8*g+b]);
                auto r=api.cp_prepared(lr_all[lane][8+b],*right_masks[8*g+b]);
                accumulate(api,groups[lane],api.mul(api.add(l,r),diag_bs[b]));
            }
        }
        const auto tm1=ts();mul_ms+=std::chrono::duration<double,std::milli>(tm1-tm0).count();
        auto rel01=api.relin_pair(*groups[0],*groups[1]);
        auto rel23=api.relin_pair(*groups[2],*groups[3]);
        for(std::size_t lane=0;lane<4;++lane) {
            auto &rel=(lane<2)?rel01[lane]:rel23[lane-2];
            accumulate(api,acc_outputs[lane],g?api.rotate(rel,2048*g):std::move(rel));
        }
        const auto tr1=ts();relin_ms+=std::chrono::duration<double,std::milli>(tr1-tm1).count();
    }
    std::vector<CT> outputs;outputs.reserve(4);
    for(std::size_t lane=0;lane<4;++lane) {
        outputs.push_back(api.rescale(*acc_outputs[lane],2));
        api.require_scale(outputs.back(),std::exp2(40.));
    }
    if(av_timing) {
        const auto t3=ts();
        std::printf("AV_STAGES total=%.3f v_diag=%.3f masks=%.3f prob_gs=%.3f lr_hoist=%.3f mul=%.3f relin_rot=%.3f\n",
            std::chrono::duration<double,std::milli>(t3-t0).count(),
            std::chrono::duration<double,std::milli>(t1-t0).count(),
            std::chrono::duration<double,std::milli>(t2-t1).count(),
            prob_gs_ms,lr_ms,mul_ms,relin_ms);
    }
    return outputs;
}
inline std::vector<CT> av_four(Api &api,const std::array<const CT*,4> &probabilities_times_two,const CT &V) {
    if(fast_attention_enabled())return av_four_fast(api,probabilities_times_two,V);
    std::vector<int> babies,giants;
    for(int i=0;i<8;++i)babies.push_back(256*i);
    for(int i=0;i<16;++i)giants.push_back((8-2048)*i);
    auto Vscaled=api.cp_to(V,1.,std::exp2(90.),0);
    auto value_babies=api.hoisted(Vscaled,babies);
    std::optional<CT> diagonal;
    for(int g=0;g<16;++g) {
        std::optional<CT> group;
        for(int b=0;b<8;++b) {
            std::vector<double> selector(32768);
            for(std::size_t j=8*g+b;j<32768;j+=128)selector[j]=1.;
            auto term=api.cp_vector(value_babies[b],selector,
                std::exp2(40.)*api.removed(Vscaled.meta.q_count,3)/Vscaled.meta.scale);
            accumulate(api,group,std::move(term));
        }
        accumulate(api,diagonal,api.rotate(*group,2048*g));
    }
    *diagonal=api.rescale(*diagonal,1);
    auto diag_bs=api.hoisted(*diagonal,babies);
    for(auto &v:diag_bs)v=api.rescale(v,2);
    const auto &shape=*probabilities_times_two[0];
    const auto common_q=std::min(shape.meta.q_count-2,diag_bs[0].meta.q_count);
    const double prepared_scale=std::exp2(40.)*api.prime(common_q)/diag_bs[0].meta.scale;
    std::vector<std::shared_ptr<poseidon::gpu::GpuPlaintextData>> left_masks,right_masks;
    for(int offset=0;offset<128;++offset) {
        std::vector<double> left(32768),right(32768);
        for(std::size_t j=0;j<32768;++j) {
            const bool use_left=int(j%128)<128-offset;
            left[j]=use_left?1./(2.*1.0001):0.;right[j]=use_left?0.:1./(2.*1.0001);
        }
        const double encoded=prepared_scale*api.removed(shape.meta.q_count,2)/shape.meta.scale;
        left_masks.push_back(api.cached_vector(shape,left,encoded));
        right_masks.push_back(api.cached_vector(shape,right,encoded));
    }
    std::vector<CT> outputs;
    for(const auto *prob:probabilities_times_two) {
    const auto &probability_times_two=*prob;
    if(prob->meta.q_count!=shape.meta.q_count||!Api::same_scale(prob->meta.scale,shape.meta.scale))
        throw std::runtime_error("source AV batch input contract");
    auto prob_gs=api.hoisted(probability_times_two,giants);
    const auto q=std::min(probability_times_two.meta.q_count-2,diag_bs[0].meta.q_count);
    if(q<7)throw std::runtime_error("source AV output budget");
    std::optional<CT> output;
    for(int g=0;g<16;++g) {
        std::vector<int> left_steps,right_steps;
        for(int b=0;b<8;++b){left_steps.push_back(b);right_steps.push_back(b-128);}
        auto l=api.hoisted(prob_gs[g],left_steps),r=api.hoisted(prob_gs[g],right_steps);
        std::optional<CT> group;
        for(int b=0;b<8;++b) {
            auto shifted=api.add(api.cp_prepared(l[b],*left_masks[8*g+b]),api.cp_prepared(r[b],*right_masks[8*g+b]));
            shifted=api.rescale(shifted,2);
            accumulate(api,group,api.mul(shifted,diag_bs[b]));
        }
        auto relinearized=api.relin(*group);
        accumulate(api,output,api.rotate(relinearized,2048*g));
    }
    auto result=api.rescale(*output,1);api.require_scale(result,std::exp2(40.));outputs.push_back(std::move(result));
    }
    return outputs;
}
} // namespace cerium_source
