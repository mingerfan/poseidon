// Paired (two real messages per physical Boot) variants of the source
// Softmax / RMSNorm, matching Cerium's *_vec / b35_2 usage (bootstrap.py:
// 560-565). Each lane executes exactly the unpaired source arithmetic; only
// the refresh is shared: a+i*b -> bootstrap_manual_projections -> (a, b).
// Opt-in: LLAMA_PAIR_SOFTMAX=1, LLAMA_PAIR_RMS=1.
#pragma once
#include "source_softmax.hpp"
#include <array>

namespace cerium_source {
inline bool env_flag(const char *name) {const char *v=std::getenv(name);return v&&std::string(v)=="1";}
inline int fast_softmax_mode() {
    const char *v=std::getenv("LLAMA_FAST_SOFTMAX");
    if(!v||!*v||std::string(v)=="0")return 0;
    if(std::string(v)=="2")return 2;
    return std::string(v)=="1"?1:0;
}
inline bool fast_softmax_enabled() {return fast_softmax_mode()!=0;}
inline bool fast_rms_enabled() {return env_flag("LLAMA_FAST_RMS");}
inline std::array<CT,2> boot_pair(Api &api,const CT &a,const CT &b) {
    const auto *profile=api.bound_boot_profile();
    if(!profile)throw std::runtime_error("paired refresh requires a bound profile");
    const auto q=profile->bootstrap_data.input_q_count;
    auto packed=api.pack_independent_reals(api.drop(a,q),api.drop(b,q));
    auto r=api.source_bootstrap_projections_candidate(packed);
    return {std::move(r.real),std::move(r.imaginary)};
}
inline std::array<CT,2> inverse_sqrt2_pair_mode(Api &api,const std::array<CT,2> &input,int boot_mode) {
    const std::size_t initial_q=boot_mode!=0?25:14;
    const double initial_scale=std::exp2(60.);
    const double square_scale=initial_scale*initial_scale/api.removed(initial_q,2);
    const double common=std::exp2(60.)*api.removed(initial_q-2,2);
    std::array<CT,2> y;
    if(input[0].meta.parms_id==input[1].meta.parms_id&&
       Api::same_scale(input[0].meta.scale,input[1].meta.scale)) {
        std::array<CT,2> x{api.drop(input[0],initial_q+2),api.drop(input[1],initial_q+2)};
        const double cp_scale=common*api.removed(initial_q+2,2)*api.removed(initial_q,2)/
            (x[0].meta.scale*initial_scale*square_scale);
        auto xp=api.cp_pair(x[0],x[1],.25,cp_scale);
        x=api.rescale_pair(xp[0],xp[1],2);
        auto xyp=api.cp_pair(x[0],x[1],.2,initial_scale);
        auto xy=api.rescale_pair(xyp[0],xyp[1],2);
        auto xy3=api.cp_pair(xy[0],xy[1],.04,square_scale);
        auto xy3n=api.cp_pair(xy3[0],xy3[1],-1.,1.);
        std::array<CT,2> t{api.add_plain(xy3n[0],.15),api.add_plain(xy3n[1],.15)};
        auto tc=api.add_conjugate_pair(t[0],t[1]);
        y=api.rescale_pair(tc[0],tc[1],2);
    } else {
        for(int l=0;l<2;++l) {
            auto x=api.drop(input[l],initial_q+2);
            const double cp_scale=common*api.removed(initial_q+2,2)*api.removed(initial_q,2)/
                (x.meta.scale*initial_scale*square_scale);
            x=api.rescale(api.cp(x,.25,cp_scale),2);
            auto xy=api.rescale(api.cp(x,.2,initial_scale),2);
            auto xy3=api.cp(xy,.04,square_scale);
            auto t=api.add_plain(api.cp(xy3,-1.,1.),.15);
            y[l]=api.rescale(api.add(t,api.conjugate(t)),2);
        }
    }
    for(unsigned i=1;i<6;++i) {
        const bool at_source_boot=
            boot_mode==1?(i==3):(boot_mode==2?(i==2||i==4):((i%2)==1));
        y=newton_rms_iteration_pair(api,input[0],y[0],input[1],y[1],std::exp2(at_source_boot?40.:60.));
        if(at_source_boot) {
            y=boot_pair(api,y[0],y[1]);
            const std::size_t drop_q=
                boot_mode==1?25:(boot_mode==2?(i==4?20:25):(i==5?16:14));
            y={api.drop(y[0],drop_q),api.drop(y[1],drop_q)};
            y=api.cp_to_pair(y[0],y[1],1.,std::exp2(60.),0);
        }
    }
    return y;
}
inline std::array<CT,2> inverse_sqrt2_pair_schedule(Api &api,const std::array<CT,2> &input,bool one_boot) {
    return inverse_sqrt2_pair_mode(api,input,one_boot?1:0);
}
inline std::array<CT,2> inverse_sqrt2_pair(Api &api,const std::array<CT,2> &input) {
    return inverse_sqrt2_pair_schedule(api,input,false);
}
inline std::array<CT,2> exp_vec_pair(Api &api,const std::array<CT,2> &input) {
    std::array<CT,2> x;
    if(input[0].meta.parms_id==input[1].meta.parms_id&&
       Api::same_scale(input[0].meta.scale,input[1].meta.scale)) {
        x=api.cp_to_pair(input[0],input[1],1./128.,square_seed_scale(api,input[0].meta.q_count-2,5,std::exp2(40.)),2);
        x={api.add_plain(x[0],1.),api.add_plain(x[1],1.)};
    } else {
        for(int l=0;l<2;++l) {
            x[l]=api.cp_to(input[l],1./128.,square_seed_scale(api,input[l].meta.q_count-2,5,std::exp2(40.)),2);
            x[l]=api.add_plain(x[l],1.);
        }
    }
    x=source_squares_pair(api,std::move(x),5);
    x=boot_pair(api,x[0],x[1]);
    if(api.exp_unit_seed_policy()!=ExpUnitSeedPolicy::IntegerPhysicalScale)
        throw std::runtime_error("paired exp requires the integer physical-scale seed policy");
    if(x[0].meta.parms_id==x[1].meta.parms_id&&x[0].meta.scale==x[1].meta.scale) {
        const double seed_scale=square_seed_scale(api,x[0].meta.q_count,2,std::exp2(40.));
        const double ideal=seed_scale/x[0].meta.scale,integer=std::round(ideal);
        if(!(ideal>0&&std::isfinite(ideal)&&integer>=1&&integer<=std::exp2(53.)))
            throw std::runtime_error("integer exp unit seed scale out of bounds");
        x=api.cp_pair(x[0],x[1],1.,integer);
        api.record_exp_seed(true);
        api.record_exp_seed(true);
    } else {
        for(int l=0;l<2;++l) {
            const double seed_scale=square_seed_scale(api,x[l].meta.q_count,2,std::exp2(40.));
            const double ideal=seed_scale/x[l].meta.scale,integer=std::round(ideal);
            if(!(ideal>0&&std::isfinite(ideal)&&integer>=1&&integer<=std::exp2(53.)))
                throw std::runtime_error("integer exp unit seed scale out of bounds");
            x[l]=api.cp(x[l],1.,integer);
            api.record_exp_seed(true);
        }
    }
    x=source_squares_pair(api,std::move(x),2);
    return x;
}
inline std::array<CT,2> softmax128_pair(Api &api,const CT &s0,const CT &s1,const CT &encrypted_mask) {
    auto A=boot_pair(api,s0,s1);
    std::array<CT,2> m{api.copy(A[0]),api.copy(A[1])};
    for(int i=0;i<7;++i) {
        auto rot=api.rotate_pair(m[0],m[1],-(1<<i));
        m=max_new_short_pair(api,rot[0],m[0],rot[1],m[1]);
        m=boot_pair(api,m[0],m[1]);
    }
    static const std::vector<double> selector=[]{
        std::vector<double> v(32768);for(std::size_t i=127;i<32768;i+=128)v[i]=80.;return v;
    }();
    std::array<CT,2> shifted;
    if(m[0].meta.parms_id==m[1].meta.parms_id&&Api::same_scale(m[0].meta.scale,m[1].meta.scale)&&
       A[0].meta.parms_id==A[1].meta.parms_id&&Api::same_scale(A[0].meta.scale,A[1].meta.scale)) {
        m=api.cp_vector_pair(m[0],m[1],selector,std::exp2(40.)*api.removed(m[0].meta.q_count,2)/m[0].meta.scale);
        for(int i=0;i<7;++i) {
            auto rot=api.rotate_pair(m[0],m[1],1<<i);
            m=api.add_pair(m[0],rot[0],m[1],rot[1]);
        }
        m=api.rescale_pair(m[0],m[1],2);
        A=api.cp_to_pair(A[0],A[1],80.,std::exp2(40.),2);
        shifted=api.sub_pair(A[0],m[0],A[1],m[1]);
    } else {
        for(int l=0;l<2;++l) {
            m[l]=api.cp_vector(m[l],selector,std::exp2(40.)*api.removed(m[l].meta.q_count,2)/m[l].meta.scale);
            for(int i=0;i<7;++i)m[l]=api.add(m[l],api.rotate(m[l],1<<i));
            m[l]=api.rescale(m[l],2);
            A[l]=api.cp_to(A[l],80.,std::exp2(40.),2);
            shifted[l]=api.sub(A[l],m[l]);
        }
    }
    auto e=exp_vec_pair(api,shifted);
    std::array<CT,2> sum;
    static const std::vector<double> half=[]{
        std::vector<double> v(32768,0.);for(std::size_t i=0;i<32768;i+=128)v[i]=.5;return v;
    }();
    const bool same_mask=e[0].meta.q_count==e[1].meta.q_count&&Api::same_scale(e[0].meta.scale,e[1].meta.scale);
    auto mask0=api.cp_to(encrypted_mask,1.,std::exp2(40.)*api.removed(e[0].meta.q_count,2)/e[0].meta.scale,0);
    std::optional<CT> mask1_opt;
    if(!same_mask)mask1_opt=api.cp_to(encrypted_mask,1.,std::exp2(40.)*api.removed(e[1].meta.q_count,2)/e[1].meta.scale,0);
    const CT &mask1=same_mask?mask0:*mask1_opt;
    auto raw_masked=api.mul_r_pair(e[0],mask0,e[1],mask1);
    sum={api.copy(raw_masked[0]),api.copy(raw_masked[1])};
    e=api.rescale_pair(raw_masked[0],raw_masked[1],2);
    for(int i=0;i<7;++i) {
        auto rot=api.rotate_pair(sum[0],sum[1],1<<i);
        sum=api.add_pair(sum[0],rot[0],sum[1],rot[1]);
    }
    sum=api.rescale_pair(sum[0],sum[1],1);
    if(sum[0].meta.parms_id==sum[1].meta.parms_id&&Api::same_scale(sum[0].meta.scale,sum[1].meta.scale)) {
        sum=api.cp_vector_pair(sum[0],sum[1],half,std::exp2(40.)*api.removed(sum[0].meta.q_count,3)/sum[0].meta.scale);
    } else {
        for(int l=0;l<2;++l)
            sum[l]=api.cp_vector(sum[l],half,std::exp2(40.)*api.removed(sum[l].meta.q_count,3)/sum[l].meta.scale);
    }
    for(int i=0;i<7;++i) {
        auto rot=api.rotate_pair(sum[0],sum[1],-(1<<i));
        sum=api.add_pair(sum[0],rot[0],sum[1],rot[1]);
    }
    sum=api.rescale_pair(sum[0],sum[1],3);
    sum=boot_pair(api,sum[0],sum[1]);
    sum=api.cp_to_pair(sum[0],sum[1],1.,std::exp2(60.),0);
    auto inverse=inverse_sqrt2_pair_mode(api,sum,fast_softmax_mode());
    auto inv=api.square_pair(inverse[0],inverse[1],2);
    auto scaled_out=api.mul_r_pair(e[0],inv[0],e[1],inv[1]);
    auto o=api.rescale_pair(scaled_out[0],scaled_out[1],1);
    return api.cp_to_pair(o[0],o[1],1.,std::exp2(40.),2);
}
// RMSNorm with the 16 input refreshes paired two-by-two (Cerium b35_2).
inline std::vector<CT> rmsnorm_pair(Api &api,const std::vector<CT> &input,
    const std::vector<double> &gamma,std::size_t layer,double epsilon=1e-5) {
    if(input.size()!=16||gamma.size()!=4096||layer>=32)
        throw std::invalid_argument("source RMSNorm dimensions");
    std::vector<CT> x;x.reserve(16);
    for(std::size_t i=0;i<16;i+=2) {
        auto r=boot_pair(api,input[i],input[i+1]);
        auto p=api.cp_to_pair(r[0],r[1],1.,std::exp2(60.),0);
        x.push_back(std::move(p[0]));x.push_back(std::move(p[1]));
    }
    std::vector<CT> sq;sq.reserve(16);
    for(std::size_t base=0;base<16;base+=8) {
        std::vector<const CT*> half(8);
        for(std::size_t k=0;k<8;++k)half[k]=&x[base+k];
        auto prod=api.mul_batch(half,half);
        for(auto &p:prod)sq.push_back(std::move(p));
    }
    auto s8=api.add_batch(
        {&sq[0],&sq[1],&sq[2],&sq[3],&sq[4],&sq[5],&sq[6],&sq[7]},
        {&sq[8],&sq[9],&sq[10],&sq[11],&sq[12],&sq[13],&sq[14],&sq[15]});
    auto s4=api.add_batch({&s8[0],&s8[1],&s8[2],&s8[3]},{&s8[4],&s8[5],&s8[6],&s8[7]});
    auto s2=api.add_pair(s4[0],s4[2],s4[1],s4[3]);
    auto sum=api.relin(api.add(s2[0],s2[1]));
    for(int i=0;i<8;++i)sum=api.add(sum,api.rotate(sum,-(1<<i)));
    sum=api.add_plain(sum,4096.*epsilon);
    std::vector<double> selector(32768),inverse_divisor(32768);
    for(std::size_t i=0;i<32768;++i) {
        const double divisor=(layer>28||(layer>=2&&i<256))?2.:.01;
        inverse_divisor[i]=1./std::sqrt(divisor);
        if(i%256==255)selector[i]=1./(4096.*divisor);
    }
    auto mean=api.cp_vector(sum,selector,std::exp2(60.)*api.removed(sum.meta.q_count,4)/sum.meta.scale);
    for(int i=0;i<8;++i)mean=api.add(mean,api.rotate(mean,1<<i));
    mean=api.rescale(mean,4);
    const bool fast_rms=fast_rms_enabled();
    auto inverse=inverse_sqrt_rms_schedule(api,mean,fast_rms);
    std::vector<CT> relins;relins.reserve(16);
    std::vector<double> weights(32768);
    const double target_out=fast_rms?std::exp2(40.):std::exp2(60.);
    for(std::size_t i=0;i<16;i+=2) {
        std::array<CT,2> weighted_pair;
        for(std::size_t l=0;l<2;++l) {
            const std::size_t idx=i+l;
            for(std::size_t j=0;j<32768;++j)weights[j]=gamma[idx*256+j%256]*inverse_divisor[j];
            const auto q=std::min(inverse.meta.q_count,x[idx].meta.q_count);
            const double weight_scale=target_out*api.removed(q,3)/(inverse.meta.scale*x[idx].meta.scale);
            weighted_pair[l]=api.cp_vector(inverse,weights,weight_scale);
        }
        auto relin=api.mul_r_pair(weighted_pair[0],x[i],weighted_pair[1],x[i+1]);
        relins.push_back(std::move(relin[0]));relins.push_back(std::move(relin[1]));
    }
    std::vector<CT> out;out.reserve(16);
    for(std::size_t base=0;base<16;base+=8) {
        std::vector<CT> chunk;chunk.reserve(8);
        for(std::size_t k=0;k<8;++k)chunk.push_back(std::move(relins[base+k]));
        auto rescaled=api.rescale_batch(chunk,3);
        if(fast_rms) {
            for(auto &r:rescaled)out.push_back(std::move(r));
        } else {
            for(std::size_t k=0;k<8;k+=2) {
                auto p=api.cp_to_pair(rescaled[k],rescaled[k+1],1.,std::exp2(40.),1);
                out.push_back(std::move(p[0]));out.push_back(std::move(p[1]));
            }
        }
    }
    return out;
}
} // namespace cerium_source
