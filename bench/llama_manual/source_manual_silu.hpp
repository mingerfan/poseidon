#pragma once
// Manual all60 source-SiLU implementation, promoted without arithmetic changes.
// New manual entry remains experimental, not a production accuracy release.
// Requires an exclusively submitted immutable
// Boot profile with actual Q27/output2^50; FOUR encrypted Boots per call.
// CPU role/scalar audits do not establish encrypted accuracy.
#include "source_nonlinear.hpp"
namespace cerium_source {
inline CT manual_silu_all60_unpaired(Api &api,const CT &input,const CT &w) {
    if(input.meta.q_count!=6||w.meta.q_count!=14)
        throw std::runtime_error("SiLU60 requires physical inputQ6/upQ14");
    api.require_scale(input,std::exp2(40.));api.require_scale(w,std::exp2(40.));
    auto check_boot=[&](const CT &x) {
        if(x.meta.q_count!=27)throw std::runtime_error("SiLU60 requires native physical Q27");
        api.require_scale(x,std::exp2(50.));
    };
    auto v=api.source_bootstrap(input,"silu_internal:b33_2");check_boot(v);
    constexpr std::size_t q=27,poly_q=14,weighted_q=12;
    const double target=std::exp2(40.),high=std::exp2(60.);
    const double pq_scale=target*api.removed(poly_q,2)/w.meta.scale;
    const double sign_scale=api.removed(weighted_q,2);
    // Integer coefficient20 at physical PTscale1000 represents1/50.
    // NO VR rescale, metadata reset or decryption feedback.
    auto vr=api.cp(v,1./50.,1000.);api.observe("silu60:vr",vr);
    auto comp0=api.add_plain(vr,8./50.);
    auto comp1=api.add_plain(vr,4./50.);
    auto comp2=api.add_plain(vr,-4./50.);
    auto sign=[&](const CT &x,unsigned branch) {
        const std::string prefix="silu60:sign"+std::to_string(branch);
        auto y=g3_f3(api,x,g3,true,2,high);
        api.observe((prefix+":g3_1").c_str(),y);
        y=g3_f3(api,y,g3,true,2,high);
        api.observe((prefix+":g3_2").c_str(),y);
        if(y.meta.q_count!=15)throw std::runtime_error("SiLU60 G2 budget");
        y=api.cp_to(y,1.,target,2);
        api.observe((prefix+":converted").c_str(),y);
        y=api.drop(y,6);api.require_scale(y,target);
        const std::string point="silu60:g2_refresh"+std::to_string(branch);
        y=api.source_bootstrap(y,point.c_str());check_boot(y);
        y=api.cp(y,1.,1024.);api.require_scale(y,high);
        y=g3_f3(api,y,f3,true,2,high);
        api.observe((prefix+":f3_1").c_str(),y);
        y=g3_f3(api,y,f3,true,2,sign_scale);
        api.observe((prefix+":f3_2").c_str(),y);
        if(y.meta.q_count!=15)throw std::runtime_error("SiLU60 F2 budget");
        return api.drop(y,weighted_q);
    };
    auto s0=sign(comp0,0),s1=sign(comp1,1),s2=sign(comp2,2);
    auto b1=api.sub(s0,s1),b2=api.sub(s1,s2),b3=api.add_plain(s2,1.);
    // Downstream source DAG verbatim: P/Q halves, complex arithmetic,
    // weighted-before-gating, source sum order and ONE final relinearization.
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
    api.observe("silu60:P",P);api.observe("silu60:Q",Q);api.observe("silu60:vhalf",v);
    auto QW=api.mul_rescale(w,Q,2);
    auto PW=api.mul_rescale(w,P,2);
    auto vw=api.mul_rescale(w,v,2);
    api.observe("silu60:QW",QW);api.observe("silu60:PW",PW);api.observe("silu60:vw",vw);
    auto res=api.add(api.add(api.mul(b1,PW),api.mul(b2,QW)),api.mul(b3,vw));
    api.observe("silu60:raw_sum",res);
    res=api.relin_rescale(res,2);api.require_scale(res,target);
    if(res.meta.q_count!=10)throw std::runtime_error("SiLU60 final physical Q10");
    return res;
}
} // namespace cerium_source
