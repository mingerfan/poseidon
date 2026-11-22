// SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Source-order port of llama_cerium.py:276-350,454-615, Cerium 6b487041.
#pragma once
#include "source_nonlinear.hpp"
#include "source_rms.hpp"

namespace cerium_source {
inline constexpr unsigned exp_unit_seed_consumer_version=1;
inline CT sign_new_short(Api &api,const CT &x,double output_scale) {
    api.observe("max_sign:input",x);
    auto y=g3_f3(api,x,g3,false,1,api.prime(x.meta.q_count-4)/1.05);api.observe("max_sign:g3_1",y);
    y=g3_f3(api,y,g3,false,1,api.prime(y.meta.q_count-4)/1.05);api.observe("max_sign:g3_2",y);
    y=g3_f3(api,y,f3,false,1,api.prime(y.meta.q_count-4)/1.05);api.observe("max_sign:f3_1",y);
    auto out=g3_f3(api,y,f3,false,0,output_scale);api.observe("max_sign:f3_2",out);return out;
}
inline CT max_new_short(Api &api,const CT &A,const CT &B) {
    auto AB=api.add(A,B),difference=api.sub(A,B);
    const auto q=difference.meta.q_count;
    if(q<23)throw std::runtime_error("source maximum input Q");
    auto sign_input=api.cp_to(difference,1.,api.prime(q-4)/1.05,1);
    const auto sign_q=q-12;
    const double final_scale=std::exp2(40.);
    const double common=final_scale*api.removed(sign_q,2);
    auto s=sign_new_short(api,sign_input,std::exp2(60.));
    AB=api.cp(AB,.5,common/AB.meta.scale);
    difference=api.cp_to(difference,.5,common/s.meta.scale,2);
    difference=api.drop(difference,s.meta.q_count);
    auto result=api.mul_r(difference,s);
    result=api.add(result,api.drop(AB,result.meta.q_count));
    result=api.rescale(result,2);api.require_scale(result,final_scale);
    return result;
}
inline std::array<CT,2> sign_new_short_pair(Api &api,const CT &x0,const CT &x1,double output_scale) {
    auto y=g3_f3_pair(api,x0,x1,g3,false,1,api.prime(x0.meta.q_count-4)/1.05);
    y=g3_f3_pair(api,y[0],y[1],g3,false,1,api.prime(y[0].meta.q_count-4)/1.05);
    y=g3_f3_pair(api,y[0],y[1],f3,false,1,api.prime(y[0].meta.q_count-4)/1.05);
    return g3_f3_pair(api,y[0],y[1],f3,false,0,output_scale);
}
inline std::array<CT,2> max_new_short_pair(Api &api,
    const CT &A0,const CT &B0,const CT &A1,const CT &B1) {
    if(A0.meta.q_count!=A1.meta.q_count||B0.meta.q_count!=B1.meta.q_count||
       A0.meta.q_count!=B0.meta.q_count)
        return {max_new_short(api,A0,B0),max_new_short(api,A1,B1)};
    auto AB=api.add_pair(A0,B0,A1,B1);
    auto diff=api.sub_pair(A0,B0,A1,B1);
    const auto q=diff[0].meta.q_count;
    if(q<23)throw std::runtime_error("source maximum input Q");
    if(!Api::same_scale(diff[0].meta.scale,diff[1].meta.scale)||
       !Api::same_scale(AB[0].meta.scale,AB[1].meta.scale))
        return {max_new_short(api,A0,B0),max_new_short(api,A1,B1)};
    auto sign_in=api.cp_to_pair(diff[0],diff[1],1.,api.prime(q-4)/1.05,1);
    const auto sign_q=q-12;
    const double final_scale=std::exp2(40.);
    const double common=final_scale*api.removed(sign_q,2);
    auto s=sign_new_short_pair(api,sign_in[0],sign_in[1],std::exp2(60.));
    AB=api.cp_pair(AB[0],AB[1],.5,common/AB[0].meta.scale);
    if(Api::same_scale(s[0].meta.scale,s[1].meta.scale)) {
        diff=api.cp_to_pair(diff[0],diff[1],.5,common/s[0].meta.scale,2);
        diff={api.drop(diff[0],s[0].meta.q_count),api.drop(diff[1],s[1].meta.q_count)};
    } else {
        for(unsigned l=0;l<2;++l) {
            diff[l]=api.cp_to(diff[l],.5,common/s[l].meta.scale,2);
            diff[l]=api.drop(diff[l],s[l].meta.q_count);
        }
    }
    auto res=api.mul_r_pair(diff[0],s[0],diff[1],s[1]);
    auto r=api.add_pair(res[0],api.drop(AB[0],res[0].meta.q_count),
                        res[1],api.drop(AB[1],res[1].meta.q_count));
    auto out=api.rescale_pair(r[0],r[1],2);
    api.require_scale(out[0],final_scale);
    api.require_scale(out[1],final_scale);
    return out;
}
inline CT exp_vec_lane(Api &api,const CT &input) {
    auto x=api.cp_to(input,1./128.,square_seed_scale(api,input.meta.q_count-2,5,std::exp2(40.)),2);
    x=api.add_plain(x,1.);
    x=source_squares(api,std::move(x),5);api.observe("exp:after_five_squares",x);
    x=api.source_bootstrap(x,"exp_vec:after_five_squares:b35_2_lane");
    const double seed_scale=square_seed_scale(api,x.meta.q_count,2,std::exp2(40.));
    bool integer_unit=api.exp_unit_seed_policy()==ExpUnitSeedPolicy::IntegerPhysicalScale;
    if(api.exp_unit_seed_policy()==ExpUnitSeedPolicy::LegacyEnvironment) {
        const char *requested=std::getenv("LLAMA_SOFTMAX_INTEGER_UNIT_SEED");
        if(requested&&std::string(requested)!="0"&&std::string(requested)!="1")
            throw std::invalid_argument("LLAMA_SOFTMAX_INTEGER_UNIT_SEED must be0 or1");
        integer_unit=requested&&std::string(requested)=="1";
    }
    if(integer_unit) {
        // Experimental FHE precision plan: represent plaintext ONE exactly at
        // an integer physical scale. Do not falsely declare seed_scale: retain
        // the actual product scale, including through the following squares.
        // The mask interface below solves its scales from the actual e.scale.
        const double ideal=seed_scale/x.meta.scale,integer=std::round(ideal);
        if(!(ideal>0&&std::isfinite(ideal)&&integer>=1&&integer<=std::exp2(53.)))
            throw std::runtime_error("integer exp unit seed scale out of bounds");
        x=api.cp(x,1.,integer);
    } else x=api.cp_to(x,1.,seed_scale,0);
    api.record_exp_seed(integer_unit); // successful selected consumer, not SETUP intent
    auto out=source_squares(api,std::move(x),2);api.observe("exp:after_seven_squares",out);return out;
}
inline CT inverse_sqrt2(Api &api,const CT &input) {
    const std::size_t initial_q=14;
    const double initial_scale=std::exp2(60.);
    const double square_scale=initial_scale*initial_scale/api.removed(initial_q,2);
    const double common=std::exp2(60.)*api.removed(initial_q-2,2);
    auto x=api.drop(input,initial_q+2);
    const double cp_scale=common*api.removed(initial_q+2,2)*api.removed(initial_q,2)/
        (x.meta.scale*initial_scale*square_scale);
    // Iteration 0 really uses plaintext y=.2, y²=.04, as in the source.
    x=api.rescale(api.cp(x,.25,cp_scale),2);
    auto xy=api.rescale(api.cp(x,.2,initial_scale),2);
    auto xy3=api.cp(xy,.04,square_scale);
    auto y=api.add_plain(api.cp(xy3,-1.,1.),.15);
    y=api.rescale(api.add(y,api.conjugate(y)),2);api.observe("softmax_inverse:newton_1",y);
    for(unsigned i=1;i<6;++i) {
        const bool at_source_boot=(i%2)==1;
        y=newton_rms_iteration(api,input,y,std::exp2(at_source_boot?40.:60.));
        if(api.stage_observer)api.observe(("softmax_inverse:newton_"+std::to_string(i+1)).c_str(),y);
        if(at_source_boot) {
            y=api.source_bootstrap(y,"inverse_sqrt2:newton_"+std::to_string(i+1)+":b35");
            y=api.drop(y,i==5?16:14);
            y=api.cp_to(y,1.,std::exp2(60.),0);
        }
    }
    return y;
}

// Source Softmax output is 2*probability. The original AV masks, rather than
// this function, apply 1/(2*1.0001). Keep that boundary unchanged.
inline CT softmax128(Api &api,const CT &scores_div80,const CT &encrypted_mask) {
    auto A=api.source_bootstrap(scores_div80,"softmax_128x128:input:b35_2_lane");
    auto m=api.copy(A);
    for(int i=0;i<7;++i) {
        m=max_new_short(api,api.rotate(m,-(1<<i)),m);
        if(api.stage_observer)api.observe(("softmax:max_before_boot_"+std::to_string(i+1)).c_str(),m);
        m=api.source_bootstrap(m,"array_max:step_"+std::to_string(i+1)+":b35_2_lane");
    }
    std::vector<double> selector(32768);
    for(std::size_t i=127;i<32768;i+=128)selector[i]=80.;
    m=api.cp_vector(m,selector,std::exp2(40.)*api.removed(m.meta.q_count,2)/m.meta.scale);
    for(int i=0;i<7;++i)m=api.add(m,api.rotate(m,1<<i));
    m=api.rescale(m,2);api.observe("softmax:max_broadcast",m);
    A=api.cp_to(A,80.,std::exp2(40.),2);
    auto shifted=api.sub(A,m);api.observe("softmax:shifted_logits",shifted);
    auto e=exp_vec_lane(api,shifted);
    auto mask=api.cp_to(encrypted_mask,1.,
        std::exp2(40.)*api.removed(e.meta.q_count,2)/e.meta.scale,0);
    auto raw_masked=api.mul_r(e,mask);
    auto sum=api.copy(raw_masked);
    e=api.rescale(raw_masked,2);api.observe("softmax:exp_masked",e);
    for(int i=0;i<7;++i)sum=api.add(sum,api.rotate(sum,1<<i));
    sum=api.rescale(sum,1);
    std::fill(selector.begin(),selector.end(),0.);
    for(std::size_t i=0;i<32768;i+=128)selector[i]=.5;
    sum=api.cp_vector(sum,selector,std::exp2(40.)*api.removed(sum.meta.q_count,3)/sum.meta.scale);
    for(int i=0;i<7;++i)sum=api.add(sum,api.rotate(sum,-(1<<i)));
    sum=api.rescale(sum,3);api.observe("softmax:row_sum_before_boot",sum);
    sum=api.source_bootstrap(sum,"softmax_128x128:row_sum:b35_2_lane");
    sum=api.cp_to(sum,1.,std::exp2(60.),0);
    auto inverse=inverse_sqrt2(api,sum);
    api.observe("softmax:inverse_sqrt",inverse);
    inverse=api.square(inverse,2);api.observe("softmax:inverse_squared",inverse);
    auto out=api.rescale(api.mul_r(e,inverse),1);
    // Actual encoding/rescale to the backend scale40 application contract.
    auto final=api.cp_to(out,1.,std::exp2(40.),2);api.observe("softmax:output",final);return final;
}
} // namespace cerium_source
