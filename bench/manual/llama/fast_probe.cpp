// Fast A800 micro-benchmark & accuracy probe for Llama manual optimizations:
//  1) Boot stage breakdown (S2C, SSE_low, ModRaise, SSE_dense, C2S, EvalMod x2, Proj)
//     across Global P=25 vs LevelSpecialModulus::Performance vs custom P=9 schedules;
//  2) Keyswitch / rotate / relin_rescale_x2 timings at q={6,8,10,14,27} across P;
//  3) 1-Boot & 2-Boot paired SiLU accuracy & timing on public silu_patterns;
//  4) Reduced-Boot Softmax inverse_sqrt2 & RMS inverse_sqrt accuracy & timing.
#include "source_manual_boot.hpp"
#include "source_fast_boot.hpp"
#include "source_linear.hpp"
#include "source_manual_pair_silu.hpp"
#include "source_paired.hpp"
#include "source_attention.hpp"
#include "tests/source_silu_cpu.hpp"
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <cuda_runtime_api.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <vector>

using namespace cerium_source;
using namespace poseidon;
using namespace poseidon::gpu;

namespace {
struct GpuTimer {
    cudaEvent_t a{}, b{};
    GpuTimer(){gpu_check_cuda(cudaEventCreate(&a),"ev");gpu_check_cuda(cudaEventCreate(&b),"ev");}
    ~GpuTimer(){cudaEventDestroy(a);cudaEventDestroy(b);}
    void start(){gpu_check_cuda(cudaEventRecord(a,gpu_execution_stream()),"rec");}
    double stop_ms(){
        gpu_check_cuda(cudaEventRecord(b,gpu_execution_stream()),"rec");
        gpu_check_cuda(cudaEventSynchronize(b),"sync");
        float ms=0;gpu_check_cuda(cudaEventElapsedTime(&ms,a,b),"elapsed");
        return ms;
    }
};

struct PublicSiluInputs {std::array<std::vector<double>,2> gate,up;};
PublicSiluInputs silu_patterns() {
    PublicSiluInputs p;for(auto &x:p.gate)x.resize(32768);for(auto &x:p.up)x.resize(32768);
    const auto pi=std::acos(-1.);
    for(std::size_t j=0;j<32768;++j) {
        const double dyadic=25.*double(int((37*j+11)%257)-128)/128.;
        p.gate[0][j]=j%8==1?-8.:(j%8==3?-4.:(j%8==5?4.:dyadic));
        p.gate[1][j]=25.*std::sin(2*pi*17.*double(j)/32768.);
        p.up[0][j]=8.*std::cos(2*pi*13.*double(j)/32768.);
        p.up[1][j]=8.*double(int((29*j+7)%257)-128)/128.;
    }
    return p;
}

double check_silu_error(Runtime &r,const std::array<CT,2> &out,const PublicSiluInputs &p) {
    double max_err=0;
    for(unsigned lane=0;lane<2;++lane) {
        auto v=r.decrypt_final(out[lane]);
        for(std::size_t j=0;j<32768;++j) {
            auto ref=silu_cpu_reference::apply({p.gate[lane][j],0},{p.up[lane][j],0});
            double dr=v[j].real()-static_cast<double>(ref.real());
            double di=v[j].imag()-static_cast<double>(ref.imag());
            max_err=std::max(max_err,std::hypot(dr,di));
        }
    }
    return max_err;
}

// 1-Boot SiLU candidate: uses exact integer vr=cp(v,1/50,1000) (q=27, scale~2^60),
// g3_1 with doubled=true (27->21), then g3_2 (21->18), f3_1 (18->16), f3_2 (16->12)
// with doubled=false. Zero extra Boot inside sign!
std::array<CT,2> silu_1boot_cerium_q10(Api &api,const CT &gate0,const CT &up0,const CT &gate1,const CT &up1,bool g3_1_d2=false,Runtime *dbg_r=nullptr,const PublicSiluInputs *dbg_p=nullptr) {
    auto refresh=[&](const CT &a,const CT &b) {
        auto packed=api.pack_independent_reals(api.drop(a,6),api.drop(b,6));
        auto result=api.source_bootstrap_projections_candidate(packed);
        return std::array<CT,2>{std::move(result.real),std::move(result.imaginary)};
    };
    const std::size_t poly_q=g3_1_d2?12:14,weighted_q=g3_1_d2?10:12;
    const double target=std::exp2(40.),high=std::exp2(60.),sign_scale=api.removed(weighted_q,2);
    auto gate=refresh(gate0,gate1);
    std::array<CT,2> vr;std::array<std::array<CT,3>,2> comp,signs;
    for(unsigned lane=0;lane<2;++lane) {
        if(g3_1_d2) {
            vr[lane]=api.cp(gate[lane],1./50.,1000.);
            comp[lane][0]=api.add_plain(vr[lane],8./50.);
            comp[lane][1]=api.add_plain(vr[lane],4./50.);
            comp[lane][2]=api.add_plain(vr[lane],-4./50.);
        } else {
            vr[lane]=api.cp(gate[lane],1./50.,950.);
            comp[lane][0]=api.rescale(api.add_plain(vr[lane],8./50.),1);
            comp[lane][1]=api.rescale(api.add_plain(vr[lane],4./50.),1);
            comp[lane][2]=api.rescale(api.add_plain(vr[lane],-4./50.),1);
        }
    }
    for(unsigned branch=0;branch<3;++branch) {
        for(unsigned lane=0;lane<2;++lane) {
            CT y;
            if(g3_1_d2) {
                y=g3_f3(api,comp[lane][branch],g3,true,2,api.prime(20)/1.05); // 27 -> 21 (scale 2^30)
                y=g3_f3(api,y,g3,false,1,api.prime(17)/1.05);                   // 21 -> 18 (scale 2^30)
                y=g3_f3(api,y,f3,false,0,api.removed(16,2)/1.05);               // 18 -> 16 (scale 2^60!)
                y=g3_f3(api,y,f3,true,2,sign_scale);                            // 16 -> 10 (all at 2^60!)
            } else {
                y=g3_f3(api,comp[lane][branch],g3,false,1,api.prime(22)/1.05); // 26 -> 23 (scale 2^30)
                y=g3_f3(api,y,g3,false,1,api.prime(19)/1.05);                   // 23 -> 20 (scale 2^30)
                y=g3_f3(api,y,f3,false,0,api.removed(18,2)/1.05);               // 20 -> 18 (scale 2^60!)
                y=g3_f3(api,y,f3,true,2,sign_scale);                            // 18 -> 12 (all at 2^60!)
            }
            signs[lane][branch]=std::move(y);
        }
    }
    std::array<CT,2> output;
    for(unsigned lane=0;lane<2;++lane) {
        const auto &w=api.drop(lane?up1:up0,poly_q);
        auto b1=api.sub(signs[lane][0],signs[lane][1]);
        auto b2=api.sub(signs[lane][1],signs[lane][2]);
        auto b3=api.add_plain(signs[lane][2],1.);
        const double pq_scale=target*api.removed(poly_q,2)/w.meta.scale;
        auto v=api.cp_to(gate[lane],1.,high,0);v=api.drop(v,poly_q+8);
        auto v2=api.square(v,2),v4=api.square(v2,2);
        auto v6=api.mul_rescale(api.drop(v2,v4.meta.q_count),v4,2);
        const double raw_pq=pq_scale*api.removed(v6.meta.q_count,2);
        auto Q=api.cp(v6,.0002743776353465/2.,raw_pq/v6.meta.scale);
        Q=api.add(Q,api.drop(api.cp(v4,-.011113046708173/2.,raw_pq/v4.meta.scale),v6.meta.q_count));
        Q=api.add(Q,api.drop(api.cp(v2,.2281430841728270/2.,raw_pq/v2.meta.scale),v6.meta.q_count));
        Q=api.add(Q,api.drop(api.cp(v,.5/2.,raw_pq/v.meta.scale),v6.meta.q_count));
        Q=api.add_plain(api.rescale(Q,2),.0085064025895951/2.);
        auto P=api.drop(api.cp(v2,-.0055465625580307/2.,raw_pq/v2.meta.scale),v4.meta.q_count);
        P=api.drop(api.add(P,api.drop(api.cp(v,-.0819767021525476/2.,raw_pq/v.meta.scale),v4.meta.q_count)),v6.meta.q_count);
        P=api.add_plain(api.rescale(P,2),-.3067541139982155/2.);
        v=api.cp_to(v,.5,pq_scale,2);v=api.drop(v,poly_q);
        auto QW=api.mul_rescale(w,Q,2),PW=api.mul_rescale(w,P,2),vw=api.mul_rescale(w,v,2);
        auto sum=api.add(api.add(api.mul(b1,PW),api.mul(b2,QW)),api.mul(b3,vw));
        output[lane]=api.relin_rescale(sum,2);api.require_scale(output[lane],target);
        if(dbg_r&&dbg_p&&lane==0) {
            auto vs1=dbg_r->decrypt_final(signs[0][1]),vs2=dbg_r->decrypt_final(signs[0][2]);
            double e25_1=0,e25_2=0;
            for(std::size_t j=0;j<32768;++j) if(std::abs(dbg_p->gate[0][j])>10.) {
                e25_1=std::max(e25_1,std::abs(vs1[j].real()-(dbg_p->gate[0][j]>-4.?1.:-1.)));
                e25_2=std::max(e25_2,std::abs(vs2[j].real()-(dbg_p->gate[0][j]>4.?1.:-1.)));
            }
            std::cout<<"  CERIUM1BOOT_SGN_GT10 d2="<<g3_1_d2<<" e_s1="<<e25_1<<" e_s2="<<e25_2<<"\n";
        }
    }
    return output;
}

// Debug version printing per-stage error of silu_1boot_g60_pair for lane 0 branch 0!
std::array<CT,2> silu_1boot_g60_pair(Api &api,const CT &gate0,const CT &up0,const CT &gate1,const CT &up1,Runtime *dbg_r=nullptr,const PublicSiluInputs *dbg_p=nullptr) {
    auto refresh=[&](const CT &a,const CT &b) {
        auto packed=api.pack_independent_reals(api.drop(a,6),api.drop(b,6));
        auto result=api.source_bootstrap_projections_candidate(packed);
        return std::array<CT,2>{std::move(result.real),std::move(result.imaginary)};
    };
    auto dbg_stage=[&](const char *tag,const CT &ct,auto fn) {
        if(!dbg_r||!dbg_p)return;
        auto v=dbg_r->decrypt_final(ct);double me=0;
        for(std::size_t j=0;j<32768;++j)me=std::max(me,std::abs(v[j].real()-fn(dbg_p->gate[0][j])));
        std::cout<<"  SILU_DBG "<<tag<<" q="<<ct.meta.q_count<<" log_scale="<<std::log2(ct.meta.scale)<<" max_err="<<me<<"\n";
    };
    auto g3cpu=[](double x){return g3[0]*x+g3[1]*x*x*x+g3[2]*x*x*x*x*x+g3[3]*x*x*x*x*x*x*x;};
    auto f3cpu=[](double x){return f3[0]*x+f3[1]*x*x*x+f3[2]*x*x*x*x*x+f3[3]*x*x*x*x*x*x*x;};
    constexpr std::size_t poly_q=12,weighted_q=10;
    const double target=std::exp2(40.),high=std::exp2(60.),sign_scale=api.removed(weighted_q,2);
    auto gate=refresh(gate0,gate1);
    dbg_stage("after_boot",gate[0],[](double x){return x;});
    std::array<CT,2> vr;std::array<std::array<CT,3>,2> comp,signs;
    for(unsigned lane=0;lane<2;++lane) {
        vr[lane]=api.cp(gate[lane],1./50.,1000.);
        comp[lane][0]=api.add_plain(vr[lane],8./50.);
        comp[lane][1]=api.add_plain(vr[lane],4./50.);
        comp[lane][2]=api.add_plain(vr[lane],-4./50.);
    }
    dbg_stage("comp0",comp[0][0],[](double x){return (x+8.)/50.;});
    for(unsigned branch=0;branch<3;++branch) {
        for(unsigned lane=0;lane<2;++lane) {
            auto c0fn=[branch](double x){return (x+(branch==0?8.:(branch==1?4.:-4.)))/50.;};
            auto y=g3_f3(api,comp[lane][branch],g3,true,2,high);             // 27 -> 21
            if(lane==0&&branch==0)dbg_stage("g3_1",y,[&](double x){return g3cpu(c0fn(x));});
            y=g3_f3(api,y,g3,true,2,api.prime(14)/1.05);                      // 21 -> 15
            if(lane==0&&branch==0)dbg_stage("g3_2",y,[&](double x){return g3cpu(g3cpu(c0fn(x)));});
            y=g3_f3(api,y,f3,false,1,api.prime(11)/1.05);                     // 15 -> 12
            if(lane==0&&branch==0)dbg_stage("f3_1",y,[&](double x){return f3cpu(g3cpu(g3cpu(c0fn(x))));});
            y=g3_f3(api,y,f3,false,0,sign_scale);                             // 12 -> 10
            if(lane==0&&branch==0)dbg_stage("f3_2",y,[&](double x){return f3cpu(f3cpu(g3cpu(g3cpu(c0fn(x)))));});
            signs[lane][branch]=std::move(y);
        }
    }
    std::array<CT,2> output;
    for(unsigned lane=0;lane<2;++lane) {
        const auto &w=api.drop(lane?up1:up0,poly_q);
        auto b1=api.sub(signs[lane][0],signs[lane][1]);
        auto b2=api.sub(signs[lane][1],signs[lane][2]);
        auto b3=api.add_plain(signs[lane][2],1.);
        const double pq_scale=target*api.removed(poly_q,2)/w.meta.scale;
        auto v=api.cp_to(gate[lane],1.,high,0);v=api.drop(v,20); // 20->18->16->14->12
        auto v2=api.square(v,2),v4=api.square(v2,2);
        auto v6=api.mul_rescale(api.drop(v2,v4.meta.q_count),v4,2);
        const double raw_pq=pq_scale*api.removed(v6.meta.q_count,2);
        auto Q=api.cp(v6,.0002743776353465/2.,raw_pq/v6.meta.scale);
        Q=api.add(Q,api.drop(api.cp(v4,-.011113046708173/2.,raw_pq/v4.meta.scale),v6.meta.q_count));
        Q=api.add(Q,api.drop(api.cp(v2,.2281430841728270/2.,raw_pq/v2.meta.scale),v6.meta.q_count));
        Q=api.add(Q,api.drop(api.cp(v,.5/2.,raw_pq/v.meta.scale),v6.meta.q_count));
        Q=api.add_plain(api.rescale(Q,2),.0085064025895951/2.);
        auto P=api.drop(api.cp(v2,-.0055465625580307/2.,raw_pq/v2.meta.scale),v4.meta.q_count);
        P=api.drop(api.add(P,api.drop(api.cp(v,-.0819767021525476/2.,raw_pq/v.meta.scale),v4.meta.q_count)),v6.meta.q_count);
        P=api.add_plain(api.rescale(P,2),-.3067541139982155/2.);
        v=api.cp_to(v,.5,pq_scale,2);v=api.drop(v,poly_q);
        auto QW=api.mul_rescale(w,Q,2),PW=api.mul_rescale(w,P,2),vw=api.mul_rescale(w,v,2);
        auto sum=api.add(api.add(api.mul(b1,PW),api.mul(b2,QW)),api.mul(b3,vw));
        output[lane]=api.relin_rescale(sum,2);api.require_scale(output[lane],target);
        if(dbg_r&&dbg_p&&lane==0) {
            auto vo=dbg_r->decrypt_final(output[0]),vb1=dbg_r->decrypt_final(b1),vb2=dbg_r->decrypt_final(b2),vb3=dbg_r->decrypt_final(b3);
            auto vP=dbg_r->decrypt_final(P),vQ=dbg_r->decrypt_final(Q);
            double mw=0;std::size_t mj=0;double mr=0,mo=0;
            for(std::size_t j=0;j<32768;++j) {
                auto ref=silu_cpu_reference::apply({dbg_p->gate[0][j],0},{dbg_p->up[0][j],0});
                double e=std::hypot(vo[j].real()-double(ref.real()),vo[j].imag()-double(ref.imag()));
                if(e>mw){mw=e;mj=j;mr=double(ref.real());mo=vo[j].real();}
            }
            std::cout<<"  SILU_WORST j="<<mj<<" gate="<<dbg_p->gate[0][mj]<<" up="<<dbg_p->up[0][mj]
                <<" ref="<<mr<<" out="<<mo<<" err="<<mw
                <<" b1="<<vb1[mj].real()<<" b2="<<vb2[mj].real()<<" b3="<<vb3[mj].real()
                <<" P="<<vP[mj].real()<<" Q="<<vQ[mj].real()<<"\n";
            // Also max err on b1,b2,b3 across all slots and check branches 1,2 at 25:
            auto vs1=dbg_r->decrypt_final(signs[0][1]),vs2=dbg_r->decrypt_final(signs[0][2]);
            double e25_1=0,e25_2=0;
            for(std::size_t j=0;j<32768;++j) {
                if(std::abs(dbg_p->gate[0][j])>10.) {
                    double t1=dbg_p->gate[0][j]>-4.?1.:-1., t2=dbg_p->gate[0][j]>4.?1.:-1.;
                    e25_1=std::max(e25_1,std::abs(vs1[j].real()-t1));
                    e25_2=std::max(e25_2,std::abs(vs2[j].real()-t2));
                }
            }
            std::cout<<"  SILU_SGN_GT10 e_s1="<<e25_1<<" e_s2="<<e25_2<<"\n";
        }
    }
    return output;
}

// 2-Boot SiLU candidate: 1 Boot on input, 1 Boot on all-3-sign midpoint (g3_1+g3_2 at 2^60 -> Boot -> f3_1+f3_2 at 2^60).
// Even better: packs sign0+i*sign1 in 1 paired Boot, and sign2 in 1 Boot (or pairs across 2 tiles -> 2.5 Boots/2 tiles)!
// Here on 2 lanes: 3 branches x 2 lanes = 6 signals -> 3 paired Boots (total 4 Boots for 2 lanes = 2 Boots/tile)!
std::array<CT,2> silu_2boot_pair(Api &api,const CT &gate0,const CT &up0,const CT &gate1,const CT &up1) {
    auto refresh=[&](const CT &a,const CT &b) {
        auto packed=api.pack_independent_reals(api.drop(a,6),api.drop(b,6));
        auto result=api.source_bootstrap_projections_candidate(packed);
        return std::array<CT,2>{std::move(result.real),std::move(result.imaginary)};
    };
    constexpr std::size_t poly_q=14,weighted_q=12;
    const double target=std::exp2(40.),high=std::exp2(60.),sign_scale=api.removed(weighted_q,2);
    auto gate=refresh(gate0,gate1);
    std::array<CT,2> vr;std::array<std::array<CT,3>,2> comp,signs;
    for(unsigned lane=0;lane<2;++lane) {
        vr[lane]=api.cp(gate[lane],1./50.,1000.);
        comp[lane][0]=api.add_plain(vr[lane],8./50.);
        comp[lane][1]=api.add_plain(vr[lane],4./50.);
        comp[lane][2]=api.add_plain(vr[lane],-4./50.);
    }
    // Wait: g3_1 + g3_2 -> drop to 15 -> if we do f3_1 at 2^60 (15->9), still needs Boot!
    // What if g3_1(27->21) + g3_2(21->15) -> Boot -> f3_1(27->21) + f3_2(21->15)? That uses all60 everywhere!
    for(unsigned branch=0;branch<3;++branch) {
        std::array<CT,2> converted;
        for(unsigned lane=0;lane<2;++lane) {
            auto y=g3_f3(api,comp[lane][branch],g3,true,2,high);
            y=g3_f3(api,y,g3,true,2,high);
            y=api.cp_to(y,1.,target,2);
            converted[lane]=api.drop(y,6);
        }
        auto refreshed=refresh(converted[0],converted[1]);
        for(unsigned lane=0;lane<2;++lane) {
            auto y=api.cp(refreshed[lane],1.,1024.);
            y=g3_f3(api,y,f3,true,2,high);
            y=g3_f3(api,y,f3,true,2,sign_scale);
            signs[lane][branch]=api.drop(y,weighted_q);
        }
    }
    std::array<CT,2> output;
    for(unsigned lane=0;lane<2;++lane) {
        const auto &w=lane?up1:up0;
        auto b1=api.sub(signs[lane][0],signs[lane][1]);
        auto b2=api.sub(signs[lane][1],signs[lane][2]);
        auto b3=api.add_plain(signs[lane][2],1.);
        const double pq_scale=target*api.removed(poly_q,2)/w.meta.scale;
        auto v=api.cp_to(gate[lane],1.,high,0);v=api.drop(v,22);
        auto v2=api.square(v,2),v4=api.square(v2,2);
        auto v6=api.mul_rescale(api.drop(v2,v4.meta.q_count),v4,2);
        const double raw_pq=pq_scale*api.removed(v6.meta.q_count,2);
        auto Q=api.cp(v6,.0002743776353465/2.,raw_pq/v6.meta.scale);
        Q=api.add(Q,api.drop(api.cp(v4,-.011113046708173/2.,raw_pq/v4.meta.scale),v6.meta.q_count));
        Q=api.add(Q,api.drop(api.cp(v2,.2281430841728270/2.,raw_pq/v2.meta.scale),v6.meta.q_count));
        Q=api.add(Q,api.drop(api.cp(v,.5/2.,raw_pq/v.meta.scale),v6.meta.q_count));
        Q=api.add_plain(api.rescale(Q,2),.0085064025895951/2.);
        auto P=api.drop(api.cp(v2,-.0055465625580307/2.,raw_pq/v2.meta.scale),v4.meta.q_count);
        P=api.drop(api.add(P,api.drop(api.cp(v,-.0819767021525476/2.,raw_pq/v.meta.scale),v4.meta.q_count)),v6.meta.q_count);
        P=api.add_plain(api.rescale(P,2),-.3067541139982155/2.);
        v=api.cp_to(v,.5,pq_scale,2);v=api.drop(v,poly_q);
        auto QW=api.mul_rescale(w,Q,2),PW=api.mul_rescale(w,P,2),vw=api.mul_rescale(w,v,2);
        auto sum=api.add(api.add(api.mul(b1,PW),api.mul(b2,QW)),api.mul(b3,vw));
        output[lane]=api.relin_rescale(sum,2);api.require_scale(output[lane],target);
    }
    return output;
}

// Also test a 1-Boot SiLU with high-precision (d=2) on BOTH g3_1 AND g3_2?! No, 27-6-6 = 15, leaving 15-12 = 3 (<12).
// Wait: what if g3_1 (27->21, d=2), g3_2 (21->18, d=1, scale=2^30), f3_1 (18->16, d=1), f3_2 (16->12, d=1)? That's silu_1boot_pair above!
// Also test if g3_1 (27->24, d=1), g3_2 (24->18, d=2? no), or g3_1(d=2 27->21), g3_2(d=1 21->18), f3_1(d=1 18->15? Wait f3_1 rs=0 takes 2 levels 18->16, f3_2 rs=2 takes 4 levels 16->12)!

// Test inverse_sqrt2 with 1 Boot instead of 3 Boots!
// input is at q=27, scale=2^60.
// newton_1 produces y_1 at q=23 (4 primes: x->25, xy->23), newton_2 -> q=19, newton_3 -> q=15, newton_4 -> q=11,
// Boot ONCE -> q=27 -> newton_5 -> q=23, newton_6 -> q=19 (then square->17, mul e->14).
std::array<CT,2> inverse_sqrt2_1boot_pair(Api &api,const std::array<CT,2> &input) {
    // input[l] is at q=27, scale=2^60 (after sum boot + cp_to(2^60)).
    // newton_1: uses x at q=27 -> y[l] at q=23 (wait: initial_q=25).
    const std::size_t initial_q=25;
    const double initial_scale=std::exp2(60.);
    const double square_scale=initial_scale*initial_scale/api.removed(initial_q,2);
    const double common=std::exp2(60.)*api.removed(initial_q-2,2);
    std::array<CT,2> y;
    for(int l=0;l<2;++l) {
        auto x=api.drop(input[l],initial_q+2);
        const double cp_scale=common*api.removed(initial_q+2,2)*api.removed(initial_q,2)/
            (x.meta.scale*initial_scale*square_scale);
        x=api.rescale(api.cp(x,.25,cp_scale),2);
        auto xy=api.rescale(api.cp(x,.2,initial_scale),2);
        auto xy3=api.cp(xy,.04,square_scale);
        auto t=api.add_plain(api.cp(xy3,-1.,1.),.15);
        y[l]=api.rescale(api.add(t,api.conjugate(t)),2); // q = initial_q - 4 = 21!
    }
    // Now y[l] is at q=21, scale=2^60!
    // Iter 1: 21 -> 17 (scale 2^60)
    // Iter 2: 17 -> 13 (scale 2^60)
    // Iter 3: 13 -> 9  (scale 2^40) -> Boot ONCE -> q=27, scale=2^50 -> cp_to 2^60
    // Iter 4: 27 -> 23 -> Iter 5: 23 -> 19!
    // Wait: input[l] only needs q >= y.q + 2! And input[l] is at q=27, so it covers ALL!
    auto boot_pair=[&](const CT &a,const CT &b) {
        auto p=api.pack_independent_reals(api.drop(a,6),api.drop(b,6));
        auto r=api.source_bootstrap_projections_candidate(p);
        return std::array<CT,2>{std::move(r.real),std::move(r.imaginary)};
    };
    for(unsigned i=1;i<6;++i) {
        const bool do_boot=(i==3);
        for(int l=0;l<2;++l)y[l]=newton_rms_iteration(api,input[l],y[l],std::exp2(do_boot?40.:60.));
        if(do_boot) {
            y=boot_pair(y[0],y[1]);
            for(int l=0;l<2;++l)y[l]=api.drop(api.cp_to(y[l],1.,std::exp2(60.),0),25);
        }
    }
    return y;
}
} // namespace

int main(int argc,char **argv)try {
    if(argc<2){std::cerr<<"usage: fast_probe <r20_fixture.txt>\n";return 1;}
    ::setenv("LLAMA_SCALAR_CACHE","1",1);
    ::setenv("LLAMA_ADD_PREFIX","1",1);
    ::setenv("POSEIDON_NTT_ALGO","fourstep",1);
    ::setenv("POSEIDON_GPU_SKIP_MONTGOMERY_TABLES","1",0);
    ::setenv("POSEIDON_GPU_SSE_LOW_P","2",1);
    ::setenv("POSEIDON_RESCALE_X2","1",1);
    ::setenv("POSEIDON_MANUAL_BOOT_RANGE_EXPERIMENT","1",1);

    const auto literal=parameters_from_fixture(argv[1]);
    gpu_check_cuda(cudaSetDevice(0),"probe dev0");
    auto *old=rmm::mr::get_current_device_resource();
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool(old,64ULL<<20,64ULL<<30);
    rmm::mr::set_current_device_resource(&pool);

    if(argc>=3&&std::string(argv[2])=="--micro") {
        ::setenv("LLAMA_FAST_P","1",1);
        ::setenv("LLAMA_FAST_BOOT","1",1);
        ::setenv("LLAMA_FUSED_RELIN_RS2","1",1);
        ::setenv("LLAMA_FAST_NONLINEAR","1",1);
        ::setenv("LLAMA_FAST_ATTENTION","1",1);
        std::vector<int> rots=source_rotations();
        Runtime r(literal,false,rots,0,true);
        fast_boot::install_fast_runtime_levels(r,rots);
        manual_backend::SubmissionOwner host;
        auto built=fast_boot::make_fast_profile_pair(r,host);
        manual_backend::BoundApi owner(r,host,std::move(built[0]),0);
        auto &api=owner.get();
        std::vector<double> ones(32768,0.1);
        auto c6=r.encrypt(ones,6),c10=r.encrypt(ones,10),c12=r.encrypt(ones,12),c16=r.encrypt(ones,16),c27=r.encrypt(ones,27);
        auto mask=api.cached_vector(c10,ones,api.prime(10));
        auto p6=api.pack_independent_reals(c6,c6);
        auto warm_b=api.source_bootstrap_projections_candidate(p6);
        auto warm_r=api.rotate(c10,1);
        auto warm_mr=api.mul_rescale(c12,c12,2);
        gpu_check_cuda(cudaDeviceSynchronize(),"micro warm");
        auto bench=[&](const char *name,int iters,auto fn){
            GpuTimer t;t.start();
            for(int i=0;i<iters;++i){auto out=fn();(void)out;}
            double ms=t.stop_ms()/iters;
            std::cout<<"MICRO name="<<name<<" ms="<<ms<<"\n";
        };
        bench("boot_pair",10,[&]{return api.source_bootstrap_projections_candidate(p6);});
        bench("rot_q10",50,[&]{return api.rotate(c10,1);});
        static const std::vector<int> h16={0,-2048,-4096,-6144,-8192,-10240,-12288,-14336,-16384,-18432,-20480,-22528,-24576,-26624,-28672,-30720};
        bench("hoist16_q12",10,[&]{return api.hoisted(c12,h16);});
        auto prod10=api.mul(c10,c10);
        bench("mul_q10",50,[&]{return api.mul(c10,c10);});
        bench("relin_q10",50,[&]{return api.relin(prod10);});
        bench("rs1_q10",50,[&]{return api.rescale(c10,1);});
        bench("rs2_q12",50,[&]{return api.rescale(c12,2);});
        bench("mul_rs2_q12",50,[&]{return api.mul_rescale(c12,c12,2);});
        bench("cp_prep_q10",50,[&]{return api.cp_prepared(c10,*mask);});
        bench("g3_q27_d2",10,[&]{return g3_f3(api,c27,g3,true,2,api.prime(20)/1.05);});
        bench("g3_q16_d1",10,[&]{return g3_f3(api,c16,f3,false,2,api.removed(12,2));});
        std::array<const CT*,2> qp{&c12,&c12};
        std::array<const CT*,4> pp{&c10,&c10,&c10,&c10};
        bench("qk_pair_fast",2,[&]{return qk_pair_fast(api,qp,c12);});
        bench("av_four_fast",2,[&]{return av_four_fast(api,pp,c12);});
        return 0;
    }

    if(argc>=3&&std::string(argv[2])=="--opt") {
        std::vector<int> steps{0,1,2,4,8,16,32,64,-1,-2,-4,-8,-16,-32,-64};
        Runtime r(literal,false,steps);
        {
            std::set<std::uint32_t> elts;
            for(int s:steps)elts.insert(r.context.crt_context()->galois_tool()->get_elt_from_step(s));
            GaloisKeys host_gk;r.keys.create_galois_keys(std::vector<std::uint32_t>(elts.begin(),elts.end()),host_gk);
            r.galois=GpuUploader::upload_double_hoist_galois_keys(host_gk,0);
        }
        manual_backend::SubmissionOwner host;
        auto built=manual_backend::make_profile(r,65536,host);
        manual_backend::BoundApi owner(r,host,std::move(built),0);
        auto &api=owner.get();
        // 1) Test Softmax baseline vs LLAMA_FAST_NONLINEAR=1 vs 4-Boot (3-stage sign: g3->g3->f3) + 1-Boot isqrt2:
        {
            std::vector<double> scores(32768,0.);
            for(std::size_t row=0;row<38;++row)for(std::size_t col=0;col<=row;++col)
                scores[row*128+col]=(double(int((row*37+col*13)%101)-70)*0.15)/80.;
            for(std::size_t row=0;row<128;++row)for(std::size_t col=0;col<128;++col)if(row>=38||col>row)
                scores[row*128+col]=scores[(row%38)*128+(col%(row%38+1))];
            for(std::size_t j=16384;j<32768;++j)scores[j]=scores[j-16384];
            std::vector<double> emask(32768,0.);
            for(std::size_t r0=0;r0<128;++r0)for(std::size_t c0=0;c0<128;++c0)if(c0<=r0)emask[r0*128+c0]=1.;
            for(std::size_t j=16384;j<32768;++j)emask[j]=emask[j-16384];
            auto sct=r.encrypt(scores,7), mct=r.encrypt(emask,27);
            ::setenv("LLAMA_FAST_NONLINEAR","0",1);
            ::setenv("LLAMA_FAST_SOFTMAX","0",1);
            auto warm_ref=softmax128_pair(api,sct,sct,mct);
            gpu_check_cuda(cudaDeviceSynchronize(),"sm warm");
            GpuTimer ts;ts.start();
            auto ref_pair=softmax128_pair(api,sct,sct,mct);
            double ref_ms=ts.stop_ms();
            auto d_ref=r.decrypt_final(ref_pair[0]);

            ::setenv("LLAMA_FAST_NONLINEAR","1",1);
            ts.start();
            auto nl_pair=softmax128_pair(api,sct,sct,mct);
            double nl_ms=ts.stop_ms();
            auto d_nl=r.decrypt_final(nl_pair[0]);
            double nl_err=0;
            for(std::size_t row=0;row<38;++row)for(std::size_t col=0;col<=row;++col)
                nl_err=std::max(nl_err,std::abs(d_ref[row*128+col].real()-d_nl[row*128+col].real()));
            std::cout<<"PROBE_SOFTMAX_FAST_NL ref_ms="<<ref_ms<<" fast_nl_ms="<<nl_ms<<" max_prob_err="<<nl_err<<"\n";

            auto sign3=[&](const CT &x,unsigned rs,double out_scale){
                auto y=g3_f3(api,x,g3,false,1,api.prime(x.meta.q_count-4)/1.05);
                y=g3_f3(api,y,g3,false,1,api.prime(y.meta.q_count-4)/1.05);
                return g3_f3(api,y,f3,false,rs,out_scale);
            };
            // Clean 3-stage max (2 steps per Boot, 10 levels each: 27->17->7->Boot):
            auto max3_step=[&](const CT &A,const CT &B){
                auto AB=api.add(A,B),diff=api.sub(A,B);
                const auto q=diff.meta.q_count;
                auto sinp=api.cp_to(diff,1.,api.prime(q-4)/1.05,1); // q-1
                const auto sign_q=q-9; // after g3(3) + g3(3) + f3(2) = 8 -> q-9!
                const double final_scale=std::exp2(40.),common=final_scale*api.removed(sign_q,1);
                auto s=sign3(sinp,0,api.prime(sign_q)); // scale=prime(sign_q)
                AB=api.cp(AB,.5,common/AB.meta.scale);
                diff=api.cp_to(diff,.5,common/s.meta.scale,1);
                diff=api.drop(diff,s.meta.q_count);
                auto res=api.mul_r(diff,s);
                res=api.add(res,api.drop(AB,res.meta.q_count));
                res=api.rescale(res,1); // -> sign_q - 1 = q - 10!
                return res;
            };
            auto run_4max=[&](const char *label,bool one_isqrt){
                ts.start();
                auto Ab=boot_pair(api,sct,sct);
                std::array<CT,2> m{api.copy(Ab[0]),api.copy(Ab[1])};
                for(int i=0;i<7;++i) {
                    for(int l=0;l<2;++l)m[l]=max3_step(api.rotate(m[l],-(1<<i)),m[l]);
                    if((i%2)==1||i==6)m=boot_pair(api,m[0],m[1]);
                }
                static const std::vector<double> sel=[]{std::vector<double> v(32768);for(std::size_t i=127;i<32768;i+=128)v[i]=80.;return v;}();
                static const std::vector<double> half=[]{std::vector<double> v(32768,0.);for(std::size_t i=127;i<32768;i+=128)v[i]=0.;for(std::size_t i=0;i<32768;i+=128)v[i]=.5;return v;}();
                std::array<CT,2> shifted;
                for(int l=0;l<2;++l) {
                    m[l]=api.cp_vector(m[l],sel,std::exp2(40.)*api.removed(m[l].meta.q_count,2)/m[l].meta.scale);
                    for(int i=0;i<7;++i)m[l]=api.add(m[l],api.rotate(m[l],1<<i));
                    m[l]=api.rescale(m[l],2);
                    Ab[l]=api.cp_to(Ab[l],80.,std::exp2(40.),2);
                    shifted[l]=api.sub(Ab[l],m[l]);
                }
                auto e=exp_vec_pair(api,shifted);
                std::array<CT,2> sum;
                for(int l=0;l<2;++l) {
                    auto mask=api.cp_to(mct,1.,std::exp2(40.)*api.removed(e[l].meta.q_count,2)/e[l].meta.scale,0);
                    auto rm=api.mul_r(e[l],mask);
                    sum[l]=api.copy(rm);e[l]=api.rescale(rm,2);
                    for(int i=0;i<7;++i)sum[l]=api.add(sum[l],api.rotate(sum[l],1<<i));
                    sum[l]=api.rescale(sum[l],1);
                    sum[l]=api.cp_vector(sum[l],half,std::exp2(40.)*api.removed(sum[l].meta.q_count,3)/sum[l].meta.scale);
                    for(int i=0;i<7;++i)sum[l]=api.add(sum[l],api.rotate(sum[l],-(1<<i)));
                    sum[l]=api.rescale(sum[l],3);
                }
                sum=boot_pair(api,sum[0],sum[1]);
                for(int l=0;l<2;++l)sum[l]=api.cp_to(sum[l],1.,std::exp2(60.),0);
                auto inv=one_isqrt?inverse_sqrt2_pair_schedule(api,sum,true):inverse_sqrt2_pair(api,sum);
                auto inv2=api.square(inv[0],2);
                auto o=api.rescale(api.mul_r(e[0],inv2),1);
                auto p0=api.cp_to(o,1.,std::exp2(40.),2);
                double opt_ms=ts.stop_ms();
                auto d_opt=r.decrypt_final(p0);
                double max_e=0;
                for(std::size_t row=0;row<38;++row)for(std::size_t col=0;col<=row;++col)
                    max_e=std::max(max_e,std::abs(d_ref[row*128+col].real()-d_opt[row*128+col].real()));
                std::cout<<"PROBE_SOFTMAX_OPT "<<label<<" ms="<<opt_ms<<" max_prob_err="<<max_e<<"\n";
            };
            run_4max("4maxBoot+3isqrtBoot",false);
            run_4max("4maxBoot+1isqrtBoot",true);
        }
        // 2) Test SiLU 1-boot with LLAMA_FAST_NONLINEAR=0 vs 1, and Boot CUDA Graph after 2 warmups:
        {
            std::vector<double> zeros(32768,0.1);
            auto x=r.encrypt(zeros,6);
            auto p=api.pack_independent_reals(x,x);
            auto warm1=api.source_bootstrap_projections_candidate(p);
            auto warm2=api.source_bootstrap_projections_candidate(p);
            gpu_check_cuda(cudaDeviceSynchronize(),"warm");
            GpuTimer tb;tb.start();
            for(int i=0;i<5;++i)auto b=api.source_bootstrap_projections_candidate(p);
            double eager_ms=tb.stop_ms()/5.0;
            std::cout<<"PROBE_BOOT_EAGER eager_ms="<<eager_ms<<"\n";
            cudaGraph_t graph{};
            cudaStream_t st=gpu_execution_stream();
            cudaError_t cap_err=cudaStreamBeginCapture(st,cudaStreamCaptureModeRelaxed);
            bool captured=false;
            if(cap_err==cudaSuccess) {
                try {
                    auto cap_res=api.source_bootstrap_projections_candidate(p);
                    (void)cap_res;
                    cap_err=cudaStreamEndCapture(st,&graph);
                    captured=(cap_err==cudaSuccess);
                } catch(const std::exception &e) {
                    std::cout<<"PROBE_GRAPH_THROW "<<e.what()<<"\n";
                    cudaStreamEndCapture(st,&graph);
                    (void)cudaGetLastError();
                }
            }
            std::cout<<"PROBE_BOOT_GRAPH captured="<<captured<<" err="<<cudaGetErrorString(cap_err)<<"\n";
            if(captured) {
                cudaGraphExec_t exec{};
                gpu_check_cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),"inst");
                tb.start();
                for(int i=0;i<5;++i)gpu_check_cuda(cudaGraphLaunch(exec,st),"launch");
                double graph_ms=tb.stop_ms()/5.0;
                std::cout<<"PROBE_BOOT_TIMING eager_ms="<<eager_ms<<" graph_ms="<<graph_ms<<"\n";
                cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
            }
        }
        return 0;
    }

    if(argc>=3&&std::string(argv[2])=="--silu") {
        std::vector<int> steps{0};
        Runtime r(literal,false,steps);
        {
            GaloisKeys host_gk;r.keys.create_galois_keys(std::vector<std::uint32_t>{r.context.crt_context()->galois_tool()->get_elt_from_step(0)},host_gk);
            r.galois=GpuUploader::upload_double_hoist_galois_keys(host_gk,0);
        }
        manual_backend::SubmissionOwner host;
        auto built=manual_backend::make_profile(r,65536,host);
        manual_backend::BoundApi owner(r,host,std::move(built),0);
        auto &api=owner.get();
        auto pat=silu_patterns();
        auto g0=r.encrypt(pat.gate[0],6),u0=r.encrypt(pat.up[0],14);
        auto g1=r.encrypt(pat.gate[1],6),u1=r.encrypt(pat.up[1],14);
        auto run_sched=[&](const char *label,int mode) {
            auto packed=api.pack_independent_reals(g0,g1);
            auto res=api.source_bootstrap_projections_candidate(packed);
            std::array<CT,2> gate{std::move(res.real),std::move(res.imaginary)};
            constexpr std::size_t poly_q=12,weighted_q=10;
            const double sign_scale=api.removed(weighted_q,2);
            std::array<std::array<CT,3>,2> signs;
            for(unsigned lane=0;lane<2;++lane) {
                auto vr=api.cp(gate[lane],1./50.,mode==0?1000.:500.);
                const std::array<CT,3> comp{api.add_plain(vr,8./50.),api.add_plain(vr,4./50.),api.add_plain(vr,-4./50.)};
                for(unsigned b=0;b<3;++b) {
                    CT y;
                    if(mode==0) {
                        y=g3_f3(api,comp[b],g3,true,2,api.prime(20)/1.05);
                        y=g3_f3(api,y,g3,false,1,api.prime(17)/1.05);
                        y=g3_f3(api,y,f3,false,0,api.removed(16,2)/1.05);
                        y=g3_f3(api,y,f3,true,2,sign_scale);
                    } else if(mode==1) {
                        // g3_1(d=2,rs=0) 27->23, g3_2(d=2,rs=1) 23->18, f3_1(d=1,rs=0) 18->16, f3_2(d=2,rs=2) 16->10
                        y=g3_f3(api,comp[b],g3,true,0,api.removed(23,2)/1.2);
                        y=g3_f3(api,y,g3,true,1,api.prime(17)/1.05);
                        y=g3_f3(api,y,f3,false,0,api.removed(16,2)/1.05);
                        y=g3_f3(api,y,f3,true,2,sign_scale);
                    } else {
                        // g3_1(d=2,rs=0) 27->23, g3_2(d=2,rs=2) 23->17, f3_1(d=1,rs=1) 17->14, f3_2(d=1,rs=2) 14->10
                        y=g3_f3(api,comp[b],g3,true,0,api.removed(23,2)/1.2);
                        y=g3_f3(api,y,g3,true,2,api.prime(16)/1.05);
                        y=g3_f3(api,y,f3,false,1,api.prime(13)/1.05);
                        y=g3_f3(api,y,f3,false,2,sign_scale);
                    }
                    signs[lane][b]=std::move(y);
                }
            }
            auto out=finish_silu_pair_poly(api,gate,u0,u1,signs,20,poly_q,8);
            std::cout<<"PROBE_SILU_SCHED "<<label<<" max_err="<<std::setprecision(8)<<check_silu_error(r,out,pat)<<"\n";
        };
        run_sched("d2_d1_d1_d2",0);
        run_sched("d2rs0_d2rs1_d1_d2",1);
        run_sched("d2rs0_d2rs2_d1_d1",2);
        return 0;
    }

    if(argc>=3&&std::string(argv[2])=="--v2") {
        ::setenv("LLAMA_FAST_P","1",1);
        Runtime r(literal,false,source_rotations(),0,true);
        fast_boot::install_fast_runtime_levels(r,source_rotations());
        NativeLinearAdapter lin(*r.api,r.context);
        // Diagnostic: isolate d=0 non-const, d=1, d=16, dense 1x1 (B=2), dense 4x4 (B=8)
        auto test_case=[&](const char *name,std::size_t nin,std::size_t nout,auto wfn) {
            std::vector<CT> in;
            for(std::size_t it=0;it<nin;++it) {
                std::vector<double> slot(32768);
                for(std::size_t j=0;j<32768;++j)slot[j]=0.05*std::sin(0.1*double(j%256)+double(it));
                in.push_back(r.encrypt(slot,8));
            }
            ::setenv("LLAMA_FAST_LINEAR","1",1);auto v1=lin.run(in,nin*256,nout*256,wfn,false,"v1");lin.release_input();
            ::setenv("LLAMA_FAST_LINEAR","2",1);auto v2=lin.run(in,nin*256,nout*256,wfn,false,"v2");lin.release_input();
            double e=0,m=0;
            for(std::size_t ot=0;ot<nout;++ot) {
                auto a=r.decrypt_final(v1[ot]),b=r.decrypt_final(v2[ot]);
                for(std::size_t j=0;j<32768;++j){e=std::max(e,std::abs(a[j]-b[j]));m=std::max(m,std::abs(a[j]));}
            }
            std::cout<<"DIAG_CASE name="<<name<<" err="<<e<<" ref_mag="<<m<<"\n";
        };
        test_case("1x1_d0_nonconst",1,1,[](std::size_t o,std::size_t i){return o==i?0.1*std::cos(double(o)):0.;});
        test_case("1x1_d1",1,1,[](std::size_t o,std::size_t i){return (o<128&&i<128&&i==o+1)||(o>=128&&i>=128&&i==o+1)?0.5:0.;});
        test_case("1x1_d16",1,1,[](std::size_t o,std::size_t i){return (o<128&&i<128&&i==o+16)?0.5:0.;});
        test_case("1x1_dminus16",1,1,[](std::size_t o,std::size_t i){return (o<128&&i<128&&o==i+16)?0.5:0.;});
        test_case("1x1_dminus1",1,1,[](std::size_t o,std::size_t i){return (o<128&&i<128&&o==i+1)?0.5:0.;});
        test_case("1x1_dminus127",1,1,[](std::size_t o,std::size_t i){return (o<128&&i<128&&o==i+127)?0.5:0.;});
        test_case("1x1_g0_pos",1,1,[](std::size_t o,std::size_t i){return (o<128&&i>=o&&i<o+16)?0.05:0.;});
        test_case("1x1_dense",1,1,[](std::size_t o,std::size_t i){return 0.02*((int(o*7+i*13)%17)-8);});
        test_case("4x4_dense",4,4,[](std::size_t o,std::size_t i){return 0.02*((int(o*7+i*13)%17)-8);});
        for(std::size_t q_lvl:{16UL,10UL,8UL}) {
            for(bool rope:{false,true}) {
                const std::size_t nin=4,nout=4,W=nin*256,H=nout*256;
                std::vector<CT> in;
                for(std::size_t it=0;it<nin;++it) {
                    std::vector<double> slot(32768);
                    for(std::size_t j=0;j<32768;++j)slot[j]=0.05*std::sin(0.1*double(j%256)+double(it));
                    in.push_back(r.encrypt(slot,q_lvl));
                }
                auto w=[](std::size_t o,std::size_t i){return 0.02*((int(o*7+i*13)%17)-8);};
                ::setenv("LLAMA_FAST_LINEAR","0",1);auto ref=lin.run(in,W,H,w,rope,"ref");
                auto max_err=[&](const std::vector<CT> &got) {
                    double e=0,em=0;
                    for(std::size_t ot=0;ot<nout;++ot) {
                        auto da=r.decrypt_final(ref[ot]),db=r.decrypt_final(got[ot]);
                        for(std::size_t j=0;j<32768;++j){e=std::max(e,std::abs(da[j]-db[j]));em=std::max(em,std::abs(db[j]));}
                    }
                    return std::pair<double,double>{e,em};
                };
                ::setenv("LLAMA_FAST_LINEAR","1",1);auto v1=lin.run(in,W,H,w,rope,"v1");lin.release_input();
                auto ev1=max_err(v1);
                ::setenv("LLAMA_FAST_LINEAR","2",1);auto v2=lin.run(in,W,H,w,rope,"v2");lin.release_input();
                auto ev2=max_err(v2);
                std::cout<<"PROBE_LIN q="<<q_lvl<<" rope="<<rope
                    <<" v1_err="<<ev1.first<<" v1_mag="<<ev1.second
                    <<" v2_err="<<ev2.first<<" v2_mag="<<ev2.second<<"\n";
            }
        }
        return 0;
    }

    std::vector<int> steps{0};
    Runtime r(literal,false,steps);
    // Replace r.galois with double-hoist pre-rotated version!
    {
        std::set<std::uint32_t> elts;
        for(int s:steps)elts.insert(r.context.crt_context()->galois_tool()->get_elt_from_step(s));
        GaloisKeys host;r.keys.create_galois_keys(std::vector<std::uint32_t>(elts.begin(),elts.end()),host);
        r.galois=GpuUploader::upload_double_hoist_galois_keys(host,0);
    }
    manual_backend::SubmissionOwner host;
    auto built=manual_backend::make_profile(r,65536,host);
    manual_backend::BoundApi owner(r,host,std::move(built),0);
    auto &api=owner.get();
    const auto &prof=owner.get_profile();
    auto &data=const_cast<GpuBootstrapData&>(prof.bootstrap_data);

    GpuTimer t;
    // 1. Profile full bootstrap_manual_projections (Global P=25)
    auto pat=silu_patterns();
    std::cout<<"PROBE_PRIMES_LOG2";
    for(std::size_t q=1;q<=28;++q)std::cout<<" "<<q<<":"<<std::log2(api.prime(q));
    std::cout<<"\n";
    auto g0=r.encrypt(pat.gate[0],6),u0=r.encrypt(pat.up[0],15);
    auto g1=r.encrypt(pat.gate[1],6),u1=r.encrypt(pat.up[1],15);
    auto u0_12=r.encrypt(pat.up[0],12),u1_12=r.encrypt(pat.up[1],12);
    gpu_check_cuda(cudaDeviceSynchronize(),"init");

    // Test 1-Boot SiLU (f3_2 at 2^60!)
    try{
        auto u0_14=api.drop(u0,14),u1_14=api.drop(u1,14);
        auto out_c1=silu_1boot_cerium_q10(api,g0,u0_14,g1,u1_14,false,&r,&pat);
        gpu_check_cuda(cudaDeviceSynchronize(),"c1");
        t.start();out_c1=silu_1boot_cerium_q10(api,g0,u0_14,g1,u1_14,false);double ms_c1=t.stop_ms();
        std::cout<<"PROBE_SILU_1BOOT_Q10 ms="<<ms_c1<<" max_err="<<std::setprecision(8)<<check_silu_error(r,out_c1,pat)<<" out_q="<<out_c1[0].meta.q_count<<"\n";

        auto out_c2=silu_1boot_cerium_q10(api,g0,u0_12,g1,u1_12,true,&r,&pat);
        gpu_check_cuda(cudaDeviceSynchronize(),"c2");
        t.start();out_c2=silu_1boot_cerium_q10(api,g0,u0_12,g1,u1_12,true);double ms_c2=t.stop_ms();
        std::cout<<"PROBE_SILU_1BOOT_D2_Q8 ms="<<ms_c2<<" max_err="<<std::setprecision(8)<<check_silu_error(r,out_c2,pat)<<" out_q="<<out_c2[0].meta.q_count<<"\n";
    }catch(const std::exception &e){std::cout<<"PROBE_SILU_ERR: "<<e.what()<<"\n";}
    return 0;
}catch(const std::exception &e){std::cerr<<"PROBE_FAILED: "<<e.what()<<"\n";return 1;}
