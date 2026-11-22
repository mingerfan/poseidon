// SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Handwritten source-order port: llama_cerium.py:1053-1167, llama_runner.py
// get_isqrt_inputs_rmsnorm/get_rmsnorm_inputs, pinned 6b487041.
#pragma once
#include "source_ops.hpp"

namespace cerium_source {
// Solve the original square/rescale recurrence using the backend's actual
// primes. This only selects plaintext encoding scales; no metadata rewriting.
inline double square_seed_scale(Api &api,std::size_t q,unsigned count,double output_scale) {
    if(q<=2*count)throw std::runtime_error("insufficient Q for source square chain");
    double log_scale=std::log2(output_scale);
    for(unsigned i=count;i>0;--i)
        log_scale=(log_scale+std::log2(api.removed(q-2*(i-1),2)))/2.;
    return std::exp2(log_scale);
}
inline CT source_squares(Api &api,CT x,unsigned count) {
    for(unsigned i=0;i<count;++i)x=api.square(x,2);
    return x;
}
inline std::array<CT,2> source_squares_pair(Api &api,std::array<CT,2> x,unsigned count) {
    for(unsigned i=0;i<count;++i)x=api.square_pair(x[0],x[1],2);
    return x;
}
// Original source refreshes after every second iteration: logical Q10->6->2.
// Map these public boundaries to physical Q14->10->6, the ResNet20 input Q.
// Data coefficients are unchanged; the half before conjugation comes from
// the original runner's half encoding scale. There is no extra correction.
inline CT newton_rms_iteration(Api &api,const CT &x,const CT &y,double output_scale) {
    const auto q=y.meta.q_count;
    if(q<10||x.meta.q_count<q+2)throw std::runtime_error("source Newton physical budget");
    auto y2=api.square(y,2);
    auto x_=api.drop(x,q+2);
    const double common=output_scale*api.removed(q-2,2);
    const double encoded=common*api.removed(q+2,2)*api.removed(q,2)/
        (x_.meta.scale*y.meta.scale*y2.meta.scale);
    x_=api.rescale(api.cp(x_,.25,encoded),2);
    auto xy=api.mul_rescale(x_,y,2);
    auto xy3=api.mul_r(xy,y2);
    auto y_=api.drop(api.cp(y,.75,common/y.meta.scale),q-2);
    auto result=api.sub(y_,xy3);
    result=api.add(result,api.conjugate(result));
    result=api.rescale(result,2);api.require_scale(result,output_scale);
    return result;
}
inline std::array<CT,2> newton_rms_iteration_pair(Api &api,
    const CT &x0,const CT &y0,const CT &x1,const CT &y1,double output_scale) {
    if(y0.meta.q_count!=y1.meta.q_count||x0.meta.q_count!=x1.meta.q_count)
        return {newton_rms_iteration(api,x0,y0,output_scale),
                newton_rms_iteration(api,x1,y1,output_scale)};
    const auto q=y0.meta.q_count;
    if(q<10||x0.meta.q_count<q+2)throw std::runtime_error("source Newton physical budget");
    const std::array<const CT*,2> xs{&x0,&x1},ys{&y0,&y1};
    auto y2=api.square_pair(y0,y1,2);
    const double common=output_scale*api.removed(q-2,2);
    std::array<CT,2> x_;
    if(Api::same_scale(x0.meta.scale,x1.meta.scale)&&
       Api::same_scale(y0.meta.scale,y1.meta.scale)&&
       Api::same_scale(y2[0].meta.scale,y2[1].meta.scale)) {
        std::array<CT,2> xd{api.drop(x0,q+2),api.drop(x1,q+2)};
        const double encoded=common*api.removed(q+2,2)*api.removed(q,2)/
            (xd[0].meta.scale*y0.meta.scale*y2[0].meta.scale);
        auto xp=api.cp_pair(xd[0],xd[1],.25,encoded);
        x_=api.rescale_pair(xp[0],xp[1],2);
    } else {
        for(unsigned l=0;l<2;++l) {
            x_[l]=api.drop(*xs[l],q+2);
            const double encoded=common*api.removed(q+2,2)*api.removed(q,2)/
                (x_[l].meta.scale*ys[l]->meta.scale*y2[l].meta.scale);
            x_[l]=api.rescale(api.cp(x_[l],.25,encoded),2);
        }
    }
    auto xy=api.mul_rescale_pair(x_[0],y0,x_[1],y1,2);
    auto xy3=api.mul_r_pair(xy[0],y2[0],xy[1],y2[1]);
    std::array<CT,2> y_;
    if(Api::same_scale(y0.meta.scale,y1.meta.scale)) {
        auto yp=api.cp_pair(y0,y1,.75,common/y0.meta.scale);
        y_={api.drop(yp[0],q-2),api.drop(yp[1],q-2)};
    } else {
        for(unsigned l=0;l<2;++l)y_[l]=api.drop(api.cp(*ys[l],.75,common/ys[l]->meta.scale),q-2);
    }
    auto r=api.sub_pair(y_[0],xy3[0],y_[1],xy3[1]);
    r=api.add_conjugate_pair(r[0],r[1]);
    auto out=api.rescale_pair(r[0],r[1],2);
    api.require_scale(out[0],output_scale);
    api.require_scale(out[1],output_scale);
    return out;
}
inline CT inverse_sqrt_rms_schedule(Api &api,const CT &input,bool fast_3per_boot) {
    if(input.meta.q_count<(fast_3per_boot?23:20))throw std::runtime_error("RMS inverse input budget");
    auto x=api.add_plain(api.cp(input,-1.,1.),-.4);
    x=api.cp_to(x,1./256.,square_seed_scale(api,x.meta.q_count-2,4,std::exp2(40.)),2);
    x=api.add_plain(x,1.);
    x=source_squares(api,std::move(x),4);api.observe("rms:seed_after_four_squares",x);
    api.require_scale(x,std::exp2(40.));
    x=api.source_bootstrap(x,"inverse_sqrt:seed_after_four_squares:b35");
    x=api.cp_to(x,1.,square_seed_scale(api,x.meta.q_count,3,std::exp2(60.)),0);
    x=source_squares(api,std::move(x),3);
    auto y=api.add_plain(api.add(x,x),.2);api.observe("rms:y_initial",y);
    y=api.drop(y,fast_3per_boot?21:14);
    for(unsigned i=0;i<10;++i) {
        const bool at_source_boot=fast_3per_boot?(i==2||i==5||i==8):((i%2)==1);
        const double out_scale=std::exp2((at_source_boot||(fast_3per_boot&&i==9))?40.:60.);
        y=newton_rms_iteration(api,input,y,out_scale);
        if(api.stage_observer)api.observe(("rms:newton_"+std::to_string(i+1)).c_str(),y);
        if(at_source_boot) {
            y=api.source_bootstrap(y,"inverse_sqrt:newton_"+std::to_string(i+1)+":b35");
            y=api.drop(y,fast_3per_boot?21:(i==9?20:14));
            y=api.cp_to(y,1.,std::exp2(60.),0);
        }
    }
    return y;
}
inline CT inverse_sqrt_rms(Api &api,const CT &input) {
    return inverse_sqrt_rms_schedule(api,input,false);
}

// Same raw squares -> sum -> one relinearize -> eight rotations -> epsilon
// -> selector -> eight rotations -> four rescale sequence as the source.
// Current lane adaptation uses 16 native refreshes, pending vector pairing.
inline std::vector<CT> rmsnorm(Api &api,const std::vector<CT> &input,
    const std::vector<double> &gamma,std::size_t layer,double epsilon=1e-5) {
    if(input.size()!=16||gamma.size()!=4096||layer>=32)
        throw std::invalid_argument("source RMSNorm dimensions");
    std::vector<CT> x;
    for(std::size_t i=0;i<16;++i) {
        auto refreshed=api.source_bootstrap(input[i],"rmsnorm:input_"+std::to_string(i)+":b35_2_lane");
        x.push_back(api.cp_to(refreshed,1.,std::exp2(60.),0));
    }
    auto sum=api.mul(x[0],x[0]);
    for(std::size_t i=1;i<16;++i)sum=api.add(sum,api.mul(x[i],x[i]));
    sum=api.relin(sum);
    for(int i=0;i<8;++i)sum=api.add(sum,api.rotate(sum,-(1<<i)));
    sum=api.add_plain(sum,4096.*epsilon);
    std::vector<double> selector(32768),inverse_divisor(32768);
    for(std::size_t i=0;i<32768;++i) {
        // Same per-layer/token remappable divisor as original model_params:
        // layers>=29 all2, layer>=2 firsttoken2, other positions .01.
        const double divisor=(layer>28||(layer>=2&&i<256))?2.:.01;
        inverse_divisor[i]=1./std::sqrt(divisor);
        if(i%256==255)selector[i]=1./(4096.*divisor);
    }
    auto mean=api.cp_vector(sum,selector,std::exp2(60.)*api.removed(sum.meta.q_count,4)/sum.meta.scale);
    for(int i=0;i<8;++i)mean=api.add(mean,api.rotate(mean,1<<i));
    mean=api.rescale(mean,4);api.observe("rms:mean",mean);
    auto inverse=inverse_sqrt_rms(api,mean);api.observe("rms:inverse_final",inverse);
    std::vector<CT> out;
    for(std::size_t i=0;i<16;++i) {
        std::vector<double> weights(32768);
        for(std::size_t j=0;j<32768;++j)weights[j]=gamma[i*256+j%256]*inverse_divisor[j];
        const auto q=std::min(inverse.meta.q_count,x[i].meta.q_count);
        const double weight_scale=std::exp2(60.)*api.removed(q,3)/(inverse.meta.scale*x[i].meta.scale);
        auto weighted=api.cp_vector(inverse,weights,weight_scale);
        auto result=api.rescale(api.mul_r(weighted,x[i]),3);
        // Backend entry contract conversion, recorded separately from source.
        out.push_back(api.cp_to(result,1.,std::exp2(40.),1));
    }
    return out;
}
} // namespace cerium_source
