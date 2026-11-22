#pragma once
// UNQUALIFIED paired topology of the already audited all60 source-SiLU DAG.
// SameN65536/Q27 profile and per-lane polynomials/scales; FOUR physical Boots
// for TWO independent logical Gate/Up streams (old topology uses eight).
// This changes refresh topology/noise, not an accepted exact-preserving pass.
// Opt-in candidate manual plan only, never a legacy/default model change.
// Requires separate complete-SiLU and new-entry GPU accuracy testing.
#include "source_manual_silu.hpp"
#include <array>
namespace cerium_source {
inline bool silu_1boot_enabled() {
    const char *v=std::getenv("LLAMA_SILU_1BOOT");return v&&std::string(v)=="1";
}
// Shared P/Q/weighting/gating tail for paired SiLU (used by both 4-Boot Q10 and 1-Boot Q8 paths).
inline std::array<CT,2> finish_silu_pair_poly(Api &api,const std::array<CT,2> &gate,
    const CT &up0,const CT &up1,const std::array<std::array<CT,3>,2> &signs,
    std::size_t v_drop_q,std::size_t poly_q,std::size_t final_q,
    const std::function<void(unsigned,const std::string&,const CT&)> &observe={}) {
    const double target=std::exp2(40.),high=std::exp2(60.);
    const double pq_scale=target*api.removed(poly_q,2)/up0.meta.scale;
    const std::array<const CT*,2> ups{&up0,&up1};
    for(unsigned lane=0;lane<2;++lane)
        if(!Api::same_scale(ups[lane]->meta.scale,up0.meta.scale))
            throw std::runtime_error("SiLU pair up scale mismatch");
    std::array<CT,2> b1=api.sub_pair(signs[0][0],signs[0][1],signs[1][0],signs[1][1]);
    std::array<CT,2> b2=api.sub_pair(signs[0][1],signs[0][2],signs[1][1],signs[1][2]);
    std::array<CT,2> b3{api.add_plain(signs[0][2],1.),api.add_plain(signs[1][2],1.)};
    std::array<CT,2> v=api.cp_to_pair(gate[0],gate[1],1.,high,0);
    v={api.drop(v[0],v_drop_q),api.drop(v[1],v_drop_q)};
    auto v2=api.square_pair(v[0],v[1],2);
    auto v4=api.square_pair(v2[0],v2[1],2);
    auto v6=api.mul_rescale_pair(
        api.drop(v2[0],v4[0].meta.q_count),v4[0],
        api.drop(v2[1],v4[1].meta.q_count),v4[1],2);
    std::array<CT,2> Q,P,vh;
    if(v[0].meta.parms_id==v[1].meta.parms_id&&
       Api::same_scale(v[0].meta.scale,v[1].meta.scale)&&
       Api::same_scale(v2[0].meta.scale,v2[1].meta.scale)&&
       Api::same_scale(v4[0].meta.scale,v4[1].meta.scale)&&
       Api::same_scale(v6[0].meta.scale,v6[1].meta.scale)) {
        const double raw_pq=pq_scale*api.removed(v6[0].meta.q_count,2);
        auto q_poly=api.cp_pair(v6[0],v6[1],.0002743776353465/2.,raw_pq/v6[0].meta.scale);
        auto q4=api.cp_pair(v4[0],v4[1],-.011113046708173/2.,raw_pq/v4[0].meta.scale);
        q_poly=api.add_pair(q_poly[0],q4[0],q_poly[1],q4[1]);
        auto q2=api.cp_pair(v2[0],v2[1],.2281430841728270/2.,raw_pq/v2[0].meta.scale);
        q_poly=api.add_pair(q_poly[0],q2[0],q_poly[1],q2[1]);
        auto q1=api.cp_pair(v[0],v[1],.5/2.,raw_pq/v[0].meta.scale);
        q_poly=api.add_pair(q_poly[0],q1[0],q_poly[1],q1[1]);
        q_poly=api.rescale_pair(q_poly[0],q_poly[1],2);
        Q={api.add_plain(q_poly[0],.0085064025895951/2.),api.add_plain(q_poly[1],.0085064025895951/2.)};
        auto p2=api.cp_pair(v2[0],v2[1],-.0055465625580307/2.,raw_pq/v2[0].meta.scale);
        auto p1=api.cp_pair(v[0],v[1],-.0819767021525476/2.,raw_pq/v[0].meta.scale);
        auto p_poly=api.add_pair(p2[0],p1[0],p2[1],p1[1]);
        p_poly={api.drop(p_poly[0],v6[0].meta.q_count),api.drop(p_poly[1],v6[1].meta.q_count)};
        p_poly=api.rescale_pair(p_poly[0],p_poly[1],2);
        P={api.add_plain(p_poly[0],-.3067541139982155/2.),api.add_plain(p_poly[1],-.3067541139982155/2.)};
        vh=api.cp_to_pair(v[0],v[1],.5,pq_scale,2);
        vh={api.drop(vh[0],poly_q),api.drop(vh[1],poly_q)};
        if(observe)for(unsigned lane=0;lane<2;++lane){observe(lane,"P",P[lane]);observe(lane,"Q",Q[lane]);observe(lane,"vhalf",vh[lane]);}
    } else {
        for(unsigned lane=0;lane<2;++lane) {
            const double raw_pq=pq_scale*api.removed(v6[lane].meta.q_count,2);
            auto q_poly=api.cp(v6[lane],.0002743776353465/2.,raw_pq/v6[lane].meta.scale);
            q_poly=api.add(q_poly,api.drop(api.cp(v4[lane],-.011113046708173/2.,raw_pq/v4[lane].meta.scale),v6[lane].meta.q_count));
            q_poly=api.add(q_poly,api.drop(api.cp(v2[lane],.2281430841728270/2.,raw_pq/v2[lane].meta.scale),v6[lane].meta.q_count));
            q_poly=api.add(q_poly,api.drop(api.cp(v[lane],.5/2.,raw_pq/v[lane].meta.scale),v6[lane].meta.q_count));
            Q[lane]=api.add_plain(api.rescale(q_poly,2),.0085064025895951/2.);
            auto p_poly=api.drop(api.cp(v2[lane],-.0055465625580307/2.,raw_pq/v2[lane].meta.scale),v4[lane].meta.q_count);
            p_poly=api.drop(api.add(p_poly,api.drop(api.cp(v[lane],-.0819767021525476/2.,raw_pq/v[lane].meta.scale),v4[lane].meta.q_count)),v6[lane].meta.q_count);
            P[lane]=api.add_plain(api.rescale(p_poly,2),-.3067541139982155/2.);
            vh[lane]=api.drop(api.cp_to(v[lane],.5,pq_scale,2),poly_q);
            if(observe){observe(lane,"P",P[lane]);observe(lane,"Q",Q[lane]);observe(lane,"vhalf",vh[lane]);}
        }
    }
    auto QW=api.mul_rescale_pair(up0,Q[0],up1,Q[1],2);
    auto PW=api.mul_rescale_pair(up0,P[0],up1,P[1],2);
    auto vw=api.mul_rescale_pair(up0,vh[0],up1,vh[1],2);
    if(observe)for(unsigned lane=0;lane<2;++lane){observe(lane,"QW",QW[lane]);observe(lane,"PW",PW[lane]);observe(lane,"vw",vw[lane]);}
    auto m1=api.mul_pair(b1[0],PW[0],b1[1],PW[1]);
    auto m2=api.mul_pair(b2[0],QW[0],b2[1],QW[1]);
    auto m3=api.mul_pair(b3[0],vw[0],b3[1],vw[1]);
    auto sum12=api.add_pair(m1[0],m2[0],m1[1],m2[1]);
    auto sum=api.add_pair(sum12[0],m3[0],sum12[1],m3[1]);
    if(observe)for(unsigned lane=0;lane<2;++lane)observe(lane,"raw_sum",sum[lane]);
    auto output=api.relin_rescale_pair(sum[0],sum[1],2);
    for(unsigned lane=0;lane<2;++lane) {
        api.require_scale(output[lane],target);
        if(output[lane].meta.q_count!=final_q)throw std::runtime_error("SiLU pair finalQ mismatch");
    }
    return output;
}
// Diagnostics only: synchronous const reads, no retained borrows or plaintext
// feedback. Freeze before packing; check that baseline immediately after pack
// and after Boot. Distinct from the old single-output boot_observer contract.
struct ManualSiluRefreshObserver {
    std::function<void(std::size_t,const CT&,const CT&)> before;
    std::function<void(std::size_t,const CT&,const CT&,const CT&)> packed;
    std::function<void(std::size_t,const CT&,const CT&,const CT&,const CT&,const CT&)> after;
};
inline std::array<CT,2> manual_silu_all60_pair_candidate(Api &api,
    const CT &input0,const CT &up0,const CT &input1,const CT &up1,
    const ManualSiluRefreshObserver &observer={}) {
    const auto before=api.boot_calls,pairs_before=api.manual_projection_pairs;
    for(const auto *g:{&input0,&input1}) {
        packing::require_single_device_real_pair_operand(*g);
        if(g->meta.q_count!=6)throw std::runtime_error("SiLU60pair requires GateQ6");
        api.require_scale(*g,std::exp2(40.));
    }
    for(const auto *w:{&up0,&up1}) {
        packing::require_single_device_real_pair_operand(*w);
        if(w->meta.q_count!=14)throw std::runtime_error("SiLU60pair requires UpQ14");
        api.require_scale(*w,std::exp2(40.));
    }
    std::size_t refresh_calls=0;
    auto refresh=[&](const CT &a,const CT &b) {
        const auto ordinal=refresh_calls++;
        if(observer.before)observer.before(ordinal,a,b);
        auto packed=api.pack_independent_reals(a,b);
        if(observer.packed)observer.packed(ordinal,a,b,packed.carrier());
        auto result=api.source_bootstrap_projections_candidate(packed);
        for(const auto *x:{&result.real,&result.imaginary}) {
            if(x->meta.q_count!=27)throw std::runtime_error("SiLU60pair requires nativeQ27");
            api.require_scale(*x,std::exp2(50.));
        }
        if(observer.after)observer.after(ordinal,a,b,packed.carrier(),result.real,result.imaginary);
        return std::array<CT,2>{std::move(result.real),std::move(result.imaginary)};
    };
    constexpr std::size_t poly_q=14,weighted_q=12;
    const double target=std::exp2(40.),high=std::exp2(60.),sign_scale=api.removed(weighted_q,2);
    auto gate=refresh(input0,input1);
    std::array<CT,2> vr;std::array<std::array<CT,3>,2> comp,signs;
    auto observe=[&](unsigned lane,const std::string &point,const CT &x) {
        if(api.stage_observer)api.observe(("silu60pair:lane"+std::to_string(lane)+":"+point).c_str(),x);
    };
    for(unsigned lane=0;lane<2;++lane) {
        vr[lane]=api.cp(gate[lane],1./50.,1000.);observe(lane,"vr",vr[lane]);
        comp[lane][0]=api.add_plain(vr[lane],8./50.);
        comp[lane][1]=api.add_plain(vr[lane],4./50.);
        comp[lane][2]=api.add_plain(vr[lane],-4./50.);
    }
    for(unsigned branch=0;branch<3;++branch) {
        std::array<CT,2> converted;
        const auto prefix="sign"+std::to_string(branch);
        auto y=g3_f3_pair(api,comp[0][branch],comp[1][branch],g3,true,2,high);
        for(unsigned lane=0;lane<2;++lane)observe(lane,prefix+":g3_1",y[lane]);
        y=g3_f3_pair(api,y[0],y[1],g3,true,2,high);
        for(unsigned lane=0;lane<2;++lane) {
            observe(lane,prefix+":g3_2",y[lane]);
            if(y[lane].meta.q_count!=15)throw std::runtime_error("SiLU60pair G2 budget");
            y[lane]=api.cp_to(y[lane],1.,target,2);observe(lane,prefix+":converted",y[lane]);
            converted[lane]=api.drop(y[lane],6);api.require_scale(converted[lane],target);
        }
        auto refreshed=refresh(converted[0],converted[1]);
        std::array<CT,2> f_in{api.cp(refreshed[0],1.,1024.),api.cp(refreshed[1],1.,1024.)};
        for(unsigned lane=0;lane<2;++lane)api.require_scale(f_in[lane],high);
        auto yf=g3_f3_pair(api,f_in[0],f_in[1],f3,true,2,high);
        for(unsigned lane=0;lane<2;++lane)observe(lane,prefix+":f3_1",yf[lane]);
        yf=g3_f3_pair(api,yf[0],yf[1],f3,true,2,sign_scale);
        for(unsigned lane=0;lane<2;++lane) {
            observe(lane,prefix+":f3_2",yf[lane]);
            if(yf[lane].meta.q_count!=15)throw std::runtime_error("SiLU60pair F2 budget");
            signs[lane][branch]=api.drop(yf[lane],weighted_q);
        }
    }
    auto output=finish_silu_pair_poly(api,gate,up0,up1,signs,22,poly_q,10,observe);
    if(refresh_calls!=4||api.boot_calls-before!=4||api.manual_projection_pairs-pairs_before!=4)
        throw std::runtime_error("SiLU60pair actual four physical projection calls required");
    return output;
}

// Fast 1-Boot paired SiLU (opt-in via LLAMA_SILU_1BOOT=1 on Mode::Fast).
// Uses Cerium sign_new2-style 2^60 f3_1->f3_2 exit at q=10 (out_q=8) with
// 1 paired Boot per 2 lanes (28 Boots total instead of 112).
inline std::array<CT,2> manual_silu_1boot_pair_fast(Api &api,
    const CT &input0,const CT &up0,const CT &input1,const CT &up1) {
    const auto before=api.boot_calls,pairs_before=api.manual_projection_pairs;
    for(const auto *g:{&input0,&input1}) {
        packing::require_single_device_real_pair_operand(*g);
        if(g->meta.q_count<6)throw std::runtime_error("SiLU1boot requires GateQ>=6");
        api.require_scale(*g,std::exp2(40.));
    }
    for(const auto *w:{&up0,&up1}) {
        packing::require_single_device_real_pair_operand(*w);
        if(w->meta.q_count<12)throw std::runtime_error("SiLU1boot requires UpQ>=12");
        api.require_scale(*w,std::exp2(40.));
    }
    const auto boot_q=api.bound_boot_profile()?api.bound_boot_profile()->bootstrap_data.input_q_count:6;
    auto packed=api.pack_independent_reals(api.drop(input0,boot_q),api.drop(input1,boot_q));
    auto res=api.source_bootstrap_projections_candidate(packed);
    std::array<CT,2> gate{std::move(res.real),std::move(res.imaginary)};
    for(const auto &g:gate){if(g.meta.q_count!=27)throw std::runtime_error("SiLU1boot requires Q27");api.require_scale(g,std::exp2(50.));}
    constexpr std::size_t poly_q=12,weighted_q=10;
    const double sign_scale=api.removed(weighted_q,2);
    std::array<std::array<CT,3>,2> comp,signs;
    auto vr=api.cp_pair(gate[0],gate[1],1./50.,1000.);
    for(unsigned lane=0;lane<2;++lane) {
        comp[lane][0]=api.add_plain(vr[lane],8./50.);
        comp[lane][1]=api.add_plain(vr[lane],4./50.);
        comp[lane][2]=api.add_plain(vr[lane],-4./50.);
    }
    for(unsigned branch=0;branch<3;++branch) {
        auto y=g3_f3_pair(api,comp[0][branch],comp[1][branch],g3,true,2,api.prime(20)/1.05);
        y=g3_f3_pair(api,y[0],y[1],g3,false,1,api.prime(17)/1.05);
        y=g3_f3_pair(api,y[0],y[1],f3,false,0,api.removed(16,2)/1.05);
        auto done=g3_f3_pair(api,y[0],y[1],f3,true,2,sign_scale);
        signs[0][branch]=std::move(done[0]);
        signs[1][branch]=std::move(done[1]);
        for(unsigned lane=0;lane<2;++lane) {
            if(signs[lane][branch].meta.q_count!=weighted_q)
                throw std::runtime_error("SiLU1boot signQ mismatch");
        }
    }
    auto output=finish_silu_pair_poly(api,gate,up0,up1,signs,20,poly_q,8);
    if(api.boot_calls-before!=1||api.manual_projection_pairs-pairs_before!=1)
        throw std::runtime_error("SiLU1boot one projection call required");
    return output;
}
} // namespace cerium_source
