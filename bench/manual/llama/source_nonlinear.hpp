// SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Handwritten C++ port of CMU-CAOS/Cerium 6b487041:
// benchmarks/llama/llama_cerium.py:353-451,1539-1696 and llama_runner.py.
// Backend adaptation: actual Poseidon primes/scales, native ResNet20 Boot.
// See SOURCE_MAPPING.md for boundaries not yet ported. No claim of full-model
// equivalence is made by these component implementations.
#pragma once
#include "source_ops.hpp"
#include <array>

namespace cerium_source {
constexpr std::array<double,4> g3{4589./1024.,-16577./1024.,25614./1024.,-12860./1024.};
constexpr std::array<double,4> f3{35./16.,-35./16.,21./16.,-5./16.};

// Preserve the original x1/x3/x5/x7 DAG, per-term relinearization, sum order,
// conjugation and single/double Rescale. This is NOT Horner evaluation.
// The original runner halves encoding scales before x+conj(x). Poseidon
// carries physical scales, so encode half the coefficient at the common
// physical scale instead: identical encoded coefficient, correct metadata.
inline CT g3_f3(Api &api,const CT &x,const std::array<double,4> &c,
                bool doubled,unsigned final_rescales,double output_scale) {
    const unsigned d=doubled?2:1;
    const auto q=x.meta.q_count;
    if(q<=2*d+final_rescales)throw std::runtime_error("g3/f3 requires more Q");
    const auto common_q=q-2*d;
    const double common_scale=output_scale*api.removed(common_q,final_rescales);
    const bool fast_nl=[]{const char *v=std::getenv("LLAMA_FAST_NONLINEAR");return v&&v[0]=='1'&&v[1]=='\0';}();
    const bool add_prefix=[]{const char *v=std::getenv("LLAMA_ADD_PREFIX");return v&&v[0]=='1'&&v[1]=='\0';}();
    auto x2=api.square(x,d),x4=api.square(x2,d);
    auto x7=api.rescale(api.cp(x,c[3]/2.,common_scale*api.removed(q,2*d)/
        (x.meta.scale*x2.meta.scale*x4.meta.scale)),d);
    x7=api.mul_rescale(x7,x2,d);
    x7=fast_nl?api.mul(x7,x4):api.mul_r(x7,x4);
    CT x5,x3,x1;
    if(fast_nl) {
        if(doubled) {
            x5=api.rescale(api.cp(x,c[2]/2.,common_scale*api.removed(q,2)/
                (x.meta.scale*x4.meta.scale)),2);
            x5=api.mul(x5,x4);
            x3=api.rescale(api.cp(x,c[1]/2.,common_scale*api.removed(q,2)/
                (x.meta.scale*x2.meta.scale)),2);
            x3=api.mul(api.drop(x3,common_q),x2);
            x1=api.cp(x,c[0]/2.,common_scale/x.meta.scale);
            if(!add_prefix)x1=api.drop(x1,common_q);
        } else {
            x5=api.rescale(api.cp(x,c[2]/2.,common_scale*api.removed(q,2)/
                (x.meta.scale*x4.meta.scale)),2);
            x5=api.mul(x5,x4);
            x3=api.rescale(api.cp(x,c[1]/2.,common_scale*api.removed(q,2)/
                (x.meta.scale*x2.meta.scale)),2);
            x3=api.mul(x3,x2);
            x1=api.rescale(api.cp(x,c[0]/2.,common_scale*api.removed(q,1)/x.meta.scale),1);
            if(!add_prefix)x1=api.drop(x1,common_q);
        }
        auto sum3=api.relin(api.add(api.add(x3,x5),x7));
        auto out=api.add(x1,sum3);
        out=api.add(out,api.conjugate(out));
        out=api.rescale(out,final_rescales);
        api.require_scale(out,output_scale);
        return out;
    }
    if(doubled) {
        x5=api.rescale(api.cp(x,c[2]/2.,common_scale*api.removed(q,2)/
            (x.meta.scale*x4.meta.scale)),2);
        x5=api.mul_r(api.drop(x5,common_q),x4);
        x3=api.rescale(api.cp(x,c[1]/2.,common_scale*api.removed(q,2)/
            (x.meta.scale*x2.meta.scale)),2);
        x3=api.mul_r(api.drop(x3,common_q),api.drop(x2,common_q));
        x1=api.drop(api.cp(x,c[0]/2.,common_scale/x.meta.scale),common_q);
    } else {
        x5=api.rescale(api.cp(x,c[2]/2.,common_scale*api.removed(q,2)/
            (x.meta.scale*x4.meta.scale)),2);
        x5=api.mul_r(x5,x4);
        x3=api.rescale(api.cp(x,c[1]/2.,common_scale*api.removed(q,2)/
            (x.meta.scale*x2.meta.scale)),2);
        x3=api.mul_r(x3,api.drop(x2,common_q));
        x1=api.rescale(api.cp(x,c[0]/2.,common_scale*api.removed(q,1)/x.meta.scale),1);
        x1=api.drop(x1,common_q);
    }
    auto out=api.add(api.add(api.add(x1,x3),x5),x7);
    out=api.add(out,api.conjugate(out));
    out=api.rescale(out,final_rescales);
    api.require_scale(out,output_scale);
    return out;
}

/// Evaluate g3/f3 cubic on N matching ciphertexts simultaneously (up to 8),
/// sharing relinearize/conjugate/rescale passes across all lanes.
inline std::vector<CT> g3_f3_batch(Api &api,const std::vector<const CT*> &xs,
    const std::array<double,4> &c,bool doubled,unsigned final_rescales,double output_scale) {
    if(xs.empty())return {};
    const unsigned d=doubled?2:1;
    const auto q=xs[0]->meta.q_count;
    const double s0=xs[0]->meta.scale;
    const bool fast_nl=[]{const char *v=std::getenv("LLAMA_FAST_NONLINEAR");return v&&v[0]=='1'&&v[1]=='\0';}();
    bool same_shape=fast_nl&&xs.size()>=2;
    for(const auto *x:xs) {
        if(!x)throw std::invalid_argument("g3_f3_batch: null source");
        if(x->meta.parms_id!=xs[0]->meta.parms_id||!Api::same_scale(x->meta.scale,s0))same_shape=false;
    }
    if(!same_shape) {
        std::vector<CT> out;out.reserve(xs.size());
        for(const auto *x:xs)out.push_back(g3_f3(api,*x,c,doubled,final_rescales,output_scale));
        return out;
    }
    if(q<=2*d+final_rescales)throw std::runtime_error("g3/f3 requires more Q");
    const auto common_q=q-2*d;
    const double common_scale=output_scale*api.removed(common_q,final_rescales);
    auto x2=api.square_batch(xs,d);
    auto x4=api.square_batch(Api::ptrs_of(x2),d);
    auto x7=api.rescale_batch(api.cp_batch(xs,c[3]/2.,common_scale*api.removed(q,2*d)/
        (s0*x2[0].meta.scale*x4[0].meta.scale)),d);
    x7=api.mul_rescale_batch(Api::ptrs_of(x7),Api::ptrs_of(x2),d);
    x7=api.mul_batch(Api::ptrs_of(x7),Api::ptrs_of(x4));
    std::vector<CT> x5,x3,x1;
    x5=api.rescale_batch(api.cp_batch(xs,c[2]/2.,common_scale*api.removed(q,2)/
        (s0*x4[0].meta.scale)),2);
    x5=api.mul_batch(Api::ptrs_of(x5),Api::ptrs_of(x4));
    x3=api.rescale_batch(api.cp_batch(xs,c[1]/2.,common_scale*api.removed(q,2)/
        (s0*x2[0].meta.scale)),2);
    x3=api.mul_batch(Api::ptrs_of(x3),Api::ptrs_of(x2));
    if(doubled)x1=api.cp_batch(xs,c[0]/2.,common_scale/s0);
    else x1=api.rescale_batch(api.cp_batch(xs,c[0]/2.,common_scale*api.removed(q,1)/s0),1);
    auto sum3=api.relin_batch(api.add_batch(Api::ptrs_of(api.add_batch(Api::ptrs_of(x3),Api::ptrs_of(x5))),Api::ptrs_of(x7)));
    auto out=api.add_batch(Api::ptrs_of(x1),Api::ptrs_of(sum3));
    out=api.add_conjugate_batch(out);
    if(final_rescales)out=api.rescale_batch(out,final_rescales);
    for(const auto &o:out)api.require_scale(o,output_scale);
    return out;
}

/// Evaluate g3/f3 cubic on two matching ciphertexts simultaneously, sharing
/// relinearize/rescale_x2 passes across both lanes without changing algebra.
inline std::array<CT,2> g3_f3_pair(Api &api,const CT &x0,const CT &x1,
    const std::array<double,4> &c,bool doubled,unsigned final_rescales,double output_scale) {
    if(x0.meta.q_count!=x1.meta.q_count||!Api::same_scale(x0.meta.scale,x1.meta.scale))
        return {g3_f3(api,x0,c,doubled,final_rescales,output_scale),
                g3_f3(api,x1,c,doubled,final_rescales,output_scale)};
    const unsigned d=doubled?2:1;
    const auto q=x0.meta.q_count;
    if(q<=2*d+final_rescales)throw std::runtime_error("g3/f3 requires more Q");
    const auto common_q=q-2*d;
    const double common_scale=output_scale*api.removed(common_q,final_rescales);
    const bool fast_nl=[]{const char *v=std::getenv("LLAMA_FAST_NONLINEAR");return v&&v[0]=='1'&&v[1]=='\0';}();
    const bool add_prefix=[]{const char *v=std::getenv("LLAMA_ADD_PREFIX");return v&&v[0]=='1'&&v[1]=='\0';}();
    const std::array<const CT*,2> xs{&x0,&x1};
    auto x2=api.square_pair(x0,x1,d);
    auto x4=api.square_pair(x2[0],x2[1],d);
    std::array<CT,2> x7p;
    if(Api::same_scale(x2[0].meta.scale,x2[1].meta.scale)&&
       Api::same_scale(x4[0].meta.scale,x4[1].meta.scale)) {
        const double x7_s=common_scale*api.removed(q,2*d)/
            (x0.meta.scale*x2[0].meta.scale*x4[0].meta.scale);
        x7p=api.cp_pair(x0,x1,c[3]/2.,x7_s);
        x7p=api.rescale_pair(x7p[0],x7p[1],d);
    } else {
        for(unsigned l=0;l<2;++l)
            x7p[l]=api.rescale(api.cp(*xs[l],c[3]/2.,common_scale*api.removed(q,2*d)/
                (xs[l]->meta.scale*x2[l].meta.scale*x4[l].meta.scale)),d);
    }
    auto x7=api.mul_rescale_pair(x7p[0],x2[0],x7p[1],x2[1],d);
    std::array<CT,2> x5,x3,x1_term;
    if(fast_nl) {
        const bool pair_scales=
            Api::same_scale(x4[0].meta.scale,x4[1].meta.scale)&&
            Api::same_scale(x2[0].meta.scale,x2[1].meta.scale);
        if(pair_scales) {
            x7=api.mul_pair(x7[0],x4[0],x7[1],x4[1]);
            x5=api.cp_pair(x0,x1,c[2]/2.,common_scale*api.removed(q,2)/(x0.meta.scale*x4[0].meta.scale));
            x5=api.rescale_pair(x5[0],x5[1],2);
            x5=api.mul_pair(x5[0],x4[0],x5[1],x4[1]);
            x3=api.cp_pair(x0,x1,c[1]/2.,common_scale*api.removed(q,2)/(x0.meta.scale*x2[0].meta.scale));
            x3=api.rescale_pair(x3[0],x3[1],2);
            if(doubled) {
                x3={api.drop(x3[0],common_q),api.drop(x3[1],common_q)};
                x3=api.mul_pair(x3[0],x2[0],x3[1],x2[1]);
                x1_term=api.cp_pair(x0,x1,c[0]/2.,common_scale/x0.meta.scale);
            } else {
                x3=api.mul_pair(x3[0],x2[0],x3[1],x2[1]);
                x1_term=api.cp_pair(x0,x1,c[0]/2.,common_scale*api.removed(q,1)/x0.meta.scale);
                x1_term=api.rescale_pair(x1_term[0],x1_term[1],1);
            }
            if(!add_prefix)x1_term={api.drop(x1_term[0],common_q),api.drop(x1_term[1],common_q)};
        } else {
            for(unsigned l=0;l<2;++l) {
                x7[l]=api.mul(x7[l],x4[l]);
                x5[l]=api.rescale(api.cp(*xs[l],c[2]/2.,common_scale*api.removed(q,2)/
                    (xs[l]->meta.scale*x4[l].meta.scale)),2);
                x5[l]=api.mul(x5[l],x4[l]);
                x3[l]=api.rescale(api.cp(*xs[l],c[1]/2.,common_scale*api.removed(q,2)/
                    (xs[l]->meta.scale*x2[l].meta.scale)),2);
                if(doubled) {
                    x3[l]=api.mul(api.drop(x3[l],common_q),x2[l]);
                    x1_term[l]=api.cp(*xs[l],c[0]/2.,common_scale/xs[l]->meta.scale);
                } else {
                    x3[l]=api.mul(x3[l],x2[l]);
                    x1_term[l]=api.rescale(api.cp(*xs[l],c[0]/2.,common_scale*api.removed(q,1)/xs[l]->meta.scale),1);
                }
                if(!add_prefix)x1_term[l]=api.drop(x1_term[l],common_q);
            }
        }
        auto sum_35=api.add_pair(x3[0],x5[0],x3[1],x5[1]);
        auto sum_in=api.add_pair(sum_35[0],x7[0],sum_35[1],x7[1]);
        auto sum3=api.relin_pair(sum_in[0],sum_in[1]);
        auto v=api.add_pair(x1_term[0],sum3[0],x1_term[1],sum3[1]);
        v=api.add_conjugate_pair(v[0],v[1]);
        auto out=api.rescale_pair(v[0],v[1],final_rescales);
        api.require_scale(out[0],output_scale);
        api.require_scale(out[1],output_scale);
        return out;
    }
    x7=api.relin_pair(x7[0],x7[1]);
    x7=api.mul_pair(x7[0],x4[0],x7[1],x4[1]);
    x5=api.cp_pair(x0,x1,c[2]/2.,common_scale*api.removed(q,2)/(x0.meta.scale*x4[0].meta.scale));
    x5=api.rescale_pair(x5[0],x5[1],2);
    x3=api.cp_pair(x0,x1,c[1]/2.,common_scale*api.removed(q,2)/(x0.meta.scale*x2[0].meta.scale));
    x3=api.rescale_pair(x3[0],x3[1],2);
    if(doubled) {
        x5={api.drop(x5[0],common_q),api.drop(x5[1],common_q)};
        x3={api.drop(x3[0],common_q),api.drop(x3[1],common_q)};
        x2={api.drop(x2[0],common_q),api.drop(x2[1],common_q)};
        x1_term=api.cp_pair(x0,x1,c[0]/2.,common_scale/x0.meta.scale);
        x1_term={api.drop(x1_term[0],common_q),api.drop(x1_term[1],common_q)};
    } else {
        x2={api.drop(x2[0],common_q),api.drop(x2[1],common_q)};
        x1_term=api.cp_pair(x0,x1,c[0]/2.,common_scale*api.removed(q,1)/x0.meta.scale);
        x1_term=api.rescale_pair(x1_term[0],x1_term[1],1);
        x1_term={api.drop(x1_term[0],common_q),api.drop(x1_term[1],common_q)};
    }
    x5=api.mul_r_pair(x5[0],x4[0],x5[1],x4[1]);
    x3=api.mul_r_pair(x3[0],x2[0],x3[1],x2[1]);
    x7=api.relin_pair(x7[0],x7[1]);
    auto v13=api.add_pair(x1_term[0],x3[0],x1_term[1],x3[1]);
    auto v57=api.add_pair(x5[0],x7[0],x5[1],x7[1]);
    auto v=api.add_pair(v13[0],v57[0],v13[1],v57[1]);
    v=api.add_conjugate_pair(v[0],v[1]);
    auto out=api.rescale_pair(v[0],v[1],final_rescales);
    api.require_scale(out[0],output_scale);
    api.require_scale(out[1],output_scale);
    return out;
}

// llama_cerium.py sign_new2: g3/rescale, g3/rescale, f3, f3_new/doubleRescale.
inline CT sign_new2(Api &api,const CT &x,double output_scale,
                    const char *prefix="sign_new2") {
    auto y=g3_f3(api,x,g3,false,1,api.prime(x.meta.q_count-4)/1.05);
    if(api.stage_observer)api.observe((std::string(prefix)+":g3_1").c_str(),y);
    y=g3_f3(api,y,g3,false,1,api.prime(y.meta.q_count-4)/1.05);
    if(api.stage_observer)api.observe((std::string(prefix)+":g3_2").c_str(),y);
    y=g3_f3(api,y,f3,false,0,api.removed(y.meta.q_count-2,2)/1.05);
    if(api.stage_observer)api.observe((std::string(prefix)+":f3_1").c_str(),y);
    auto out=g3_f3(api,y,f3,true,2,output_scale);
    if(api.stage_observer)api.observe((std::string(prefix)+":f3_2").c_str(),out);
    return out;
}

// One lane of silu_internal(v,w). Source MakeVector/BreakVector pairing is
// not implemented here; native Boot currently refreshes each lane separately.
// Keep w*P, w*Q, w*v BEFORE gating, and relinearize the three-product SUM.
inline CT silu_internal(Api &api,const CT &input,const CT &w) {
    auto v=api.source_bootstrap(input,"silu_internal:b33_2");
    // Match the refreshed branch to the unrefreshed up projection budget.
    // Only drop unused native Q; do not add another bootstrap or change w.
    const auto q=std::min(v.meta.q_count,w.meta.q_count+13);
    v=api.drop(v,q);
    if(q<27||w.meta.q_count<q-13)
        throw std::runtime_error("source SiLU physical input budget needs v>=Q27, w>=vQ-13");
    const auto poly_q=q-13,weighted_q=poly_q-2;
    const double target=std::exp2(40.);
    const double weighted_scale=target;
    const double pq_scale=weighted_scale*api.removed(poly_q,2)/w.meta.scale;
    const double sign_scale=target*api.removed(weighted_q,2)/weighted_scale;
    // vRF0/vRF1 are encoding-scale transformations in the original runner.
    // Keep the represented x/50 and x values, solving physical scales here.
    const double vr_target=api.prime(q-4)/1.05;
    const char *rational_vr=std::getenv("LLAMA_SILU_RATIONAL_VR_SCALE");
    if(rational_vr&&std::string(rational_vr)!="0"&&std::string(rational_vr)!="1")
        throw std::invalid_argument("LLAMA_SILU_RATIONAL_VR_SCALE must be0 or1");
    CT vr;
    if(rational_vr&&std::string(rational_vr)=="1") {
        // Experimental FHE encoding plan: coefficient m at scale 50*m
        // represents 1/50. Keep the natural CP+RS scale, not vr_target.
        const double ideal=vr_target*api.removed(q,1)/v.meta.scale;
        const double m=std::round(ideal/50.),scale=50.*m;
        if(!(ideal>0&&std::isfinite(ideal)&&m>=1&&scale<=std::exp2(53.))||
           std::round((1./50.)*scale)!=m)
            throw std::runtime_error("SiLU rational vr scale out of bounds");
        vr=api.rescale(api.cp(v,1./50.,scale),1);
    } else vr=api.cp_to(v,1./50.,vr_target,1);
    api.observe("silu:vr",vr);
    auto comp0=api.add_plain(vr,8./50.);
    auto comp1=api.add_plain(vr,4./50.);
    auto comp2=api.add_plain(vr,-4./50.);
    api.observe("silu:comp0",comp0);api.observe("silu:comp1",comp1);api.observe("silu:comp2",comp2);
    auto s0=sign_new2(api,comp0,sign_scale,"silu:sign0");
    auto s1=sign_new2(api,comp1,sign_scale,"silu:sign1");
    auto s2=sign_new2(api,comp2,sign_scale,"silu:sign2");
    auto b1=api.sub(s0,s1),b2=api.sub(s1,s2),b3=api.add_plain(s2,1.);
    v=api.cp_to(v,1.,std::exp2(60.),0);
    v=api.drop(v,q-5);
    auto v2=api.square(v,2),v4=api.square(v2,2);
    auto v6=api.mul_rescale(api.drop(v2,v4.meta.q_count),v4,2);
    const double raw_pq=pq_scale*api.removed(v6.meta.q_count,2);
    auto Q=api.cp(v6,.0002743776353465/2.,raw_pq/v6.meta.scale);
    auto Qt4=api.cp(v4,-.011113046708173/2.,raw_pq/v4.meta.scale);
    Q=api.add(Q,api.drop(Qt4,v6.meta.q_count));
    auto Qt2=api.cp(v2,.2281430841728270/2.,raw_pq/v2.meta.scale);
    Q=api.add(Q,api.drop(Qt2,v6.meta.q_count));
    auto Qt1=api.cp(v,.5/2.,raw_pq/v.meta.scale);
    Q=api.add(Q,api.drop(Qt1,v6.meta.q_count));
    Q=api.add_plain(api.rescale(Q,2),.0085064025895951/2.);
    auto P=api.drop(api.cp(v2,-.0055465625580307/2.,raw_pq/v2.meta.scale),v4.meta.q_count);
    auto Pt1=api.drop(api.cp(v,-.0819767021525476/2.,raw_pq/v.meta.scale),v4.meta.q_count);
    P=api.drop(api.add(P,Pt1),v6.meta.q_count);
    P=api.add_plain(api.rescale(P,2),-.3067541139982155/2.);
    v=api.cp_to(v,.5,pq_scale,2);
    v=api.drop(v,poly_q);
    auto QW=api.mul_rescale(w,Q,2);
    auto PW=api.mul_rescale(w,P,2);
    auto vw=api.mul_rescale(w,v,2);
    auto res=api.add(api.add(api.mul(b1,PW),api.mul(b2,QW)),api.mul(b3,vw));
    // Explicit interface exit conversion: source returns its high-scale raw
    // product; the current Poseidon caller requires scale 2^40.
    res=api.relin_rescale(res,2);api.require_scale(res,target);
    return res;
}
} // namespace cerium_source
