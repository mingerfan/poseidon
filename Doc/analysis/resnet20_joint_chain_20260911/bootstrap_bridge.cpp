// Public metadata only. No CUDA initialization, keys, ciphertexts or encoding.
#include "poseidon/advance/homomorphic_mod.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/gpu/gpu_scale_planner.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/parameters_literal.h"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

using namespace poseidon;
void number(double x) { if (!std::isfinite(x)) throw std::runtime_error("nonfinite metadata"); std::cout << x; }
void plain(const gpu::GpuPlaintextData &p) {
  std::cout << "{\"q\":" << p.meta.q_count << ",\"scale\":";
  number(p.meta.scale > 0 ? std::log2(p.meta.scale) : 0);
  std::cout << "}";
}
int main(int argc, char **argv) {
 try {
  std::size_t nq, np; double input_logscale;
  if (!(std::cin >> nq >> np >> input_logscale) || nq < 20 || nq > 80 || np > 80) return 2;
  std::vector<Modulus> q, p;
  for (std::size_t i=0;i<nq+np;++i) {
    std::uint64_t value; if (!(std::cin >> value)) return 2;
    (i<nq?q:p).emplace_back(value);
  }
  setenv("POSEIDON_BOOTSTRAP_EVALMOD_DYNAMIC_RESCALE","1",1);
  setenv("POSEIDON_EVALMOD_LOG_SPLIT","3",1);
  setenv("POSEIDON_EVALMOD_FLAT_BSGS_B8","0",1);
  setenv("POSEIDON_EVALMOD_VIRTUAL_DEGREE_BOUND","0",1);
  setenv("POSEIDON_EVALMOD_LEAD_LEAF_RESPLIT","0",1);
  ParametersLiteral parms(CKKS,16,15,40,192,1,Modulus(0),q,p,sec_level_type::none);
  PoseidonFactory::get_instance()->set_device_type(DEVICE_SOFTWARE);
  auto ctx=PoseidonFactory::get_instance()->create_poseidon_context(parms);
  CKKSEncoder encoder(ctx);
  std::vector<std::uint64_t> active;
  for(auto &v:q) active.push_back(v.value());
  const double target=std::exp2(45.0);
  EvalModPoly poly(ctx,CosDiscrete,target,0,5,2,25,0,59);
  const bool sweep=argc>1 && std::string(argv[1])=="sweep";
  const bool fold=argc>1 && std::string(argv[1])=="fold";
  const int profiles=sweep?41:1;
  std::cout<<std::setprecision(17)<<"[";
  for(int profile=0;profile<profiles;++profile) {
  const double last_plain=sweep?40+profile*0.25:45;
  if(profile) std::cout<<",";
  double scale=std::exp2(input_logscale);
  std::size_t cq=nq;
  std::cout << "{\"last_plain\":"<<last_plain<<",\"c2s\":[";
  for(int i=0;i<3;++i) {
    const double plain_scale=std::exp2(i==2?last_plain:45);
    auto plan=gpu::plan_gpu_dynamic_rescale(scale*plain_scale,target,
        std::span<const std::uint64_t>(active.data(),cq),true);
    if(i)std::cout<<",";
    std::cout<<"{\"q_in\":"<<cq<<",\"s_in\":"<<std::log2(scale)
       <<",\"plain\":"<<std::log2(plain_scale)<<",\"pre\":"<<std::log2(scale*plain_scale)
       <<",\"drop\":"<<plan.rescale_count<<",\"q_out\":"<<plan.output_q_count
       <<",\"s_out\":"<<std::log2(plan.output_scale)<<"}";
    cq=plan.output_q_count;scale=plan.output_scale;
  }
  poly.set_level_start(cq-1);
  auto ep=gpu::GpuUploader::upload_eval_mod_high_precision(poly,encoder,
    ctx.crt_context()->parms_id_map().at(cq-1),0,nullptr,parms_id_zero,1,nullptr,true,
    std::numeric_limits<std::uint32_t>::max(),
    std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),
    true,scale,true);
  const double raw_scale=ep.output_scale;
  const double seed_log=(40-std::log2(raw_scale))/4;
  const double seed=std::exp2(seed_log);
  Polynomial folded_polynomial=poly.sine_poly();
  if(fold) {
    for(auto &coefficient:folded_polynomial.data()) coefficient*=seed;
    auto adjusted=gpu::GpuUploader::upload_eval_mod_high_precision(poly,encoder,
      ctx.crt_context()->parms_id_map().at(cq-1),0,nullptr,parms_id_zero,1,
      &folded_polynomial,true,std::numeric_limits<std::uint32_t>::max(),
      poly.sqrt_2pi()*seed,std::numeric_limits<double>::quiet_NaN(),true,scale,true);
    if(adjusted.output_q_count!=ep.output_q_count ||
       std::abs(std::log2(adjusted.output_scale/raw_scale))>1e-10 ||
       adjusted.polynomial_blocks.size()!=ep.polynomial_blocks.size() ||
       adjusted.basis_steps.size()!=ep.basis_steps.size() ||
       adjusted.polynomial_combine_steps.size()!=ep.polynomial_combine_steps.size())
      throw std::runtime_error("coefficient folding changed the Q/scale DAG");
    for(std::size_t i=0;i<ep.polynomial_blocks.size();++i) {
      const auto &a=ep.polynomial_blocks[i]; const auto &b=adjusted.polynomial_blocks[i];
      if(a.terms.size()!=b.terms.size()) throw std::runtime_error("fold pruned a term");
      for(std::size_t j=0;j<a.terms.size();++j)
        if(a.terms[j].degree!=b.terms[j].degree) throw std::runtime_error("fold changed term degree");
    }
    ep=std::move(adjusted);
  }
  std::cout<<"],\"eval_q_in\":"<<cq<<",\"eval_s_in\":"<<std::log2(scale)
    <<",\"eval_q_out\":"<<ep.output_q_count<<",\"eval_s_out\":"<<std::log2(ep.output_scale)
    <<",\"degree\":"<<ep.polynomial_degree<<",\"requested_degree\":59,\"root_node\":"
    <<ep.polynomial_result_node<<",\"polynomial_scale\":"<<std::log2(ep.polynomial_output_scale)
    <<",\"basis\":[";
  for(std::size_t i=0;i<ep.basis_steps.size();++i) {
    auto &s=ep.basis_steps[i];if(i)std::cout<<",";
    std::cout<<"{\"degree\":"<<s.output_degree<<",\"left\":"<<s.left_degree
      <<",\"right\":"<<s.right_degree<<",\"diff\":"<<s.correction_degree
      <<",\"pre\":"<<std::log2(s.pre_rescale_scale)<<",\"drop\":"<<s.rescale_count
      <<",\"scale\":"<<std::log2(s.output_scale)<<",\"correction\":";
    plain(s.correction_plaintext);std::cout<<"}";
  }
  std::cout<<"],\"leaves\":[";
  for(std::size_t i=0;i<ep.polynomial_blocks.size();++i) {
    auto &b=ep.polynomial_blocks[i];if(i)std::cout<<",";
    std::cout<<"{\"q\":"<<b.output_q_count<<",\"scale\":"<<std::log2(b.output_scale)
      <<",\"drop\":"<<b.rescale_count<<",\"terms\":[";
    for(std::size_t j=0;j<b.terms.size();++j) {
      if(j)std::cout<<",";auto &t=b.terms[j];
      std::cout<<"{\"degree\":"<<t.degree<<",\"plain\":";plain(t.coefficient_plaintext);std::cout<<"}";
    }
    std::cout<<"]}";
  }
  std::cout<<"],\"combines\":[";
  for(std::size_t i=0;i<ep.polynomial_combine_steps.size();++i) {
    auto &s=ep.polynomial_combine_steps[i];if(i)std::cout<<",";
    std::cout<<"{\"node\":"<<s.output_node<<",\"quotient\":"<<s.quotient_node
      <<",\"remainder\":"<<s.remainder_node<<",\"basis\":"<<s.basis_degree
      <<",\"q\":"<<s.output_q_count<<",\"scale\":"<<std::log2(s.output_scale)
      <<",\"quotient_drop\":"<<s.quotient_rescale_count<<",\"quotient_scale\":";
    number(s.quotient_output_scale>0?std::log2(s.quotient_output_scale):0);
    std::cout<<",\"remainder_drop\":"<<s.remainder_rescale_count
      <<",\"product_q\":"<<s.product_q_count<<",\"product_scale\":"<<std::log2(s.product_scale)
      <<",\"product_plain\":";plain(s.product_scale_plaintext);
    std::cout<<",\"product_aligned\":";
    number(s.product_aligned_scale>0?std::log2(s.product_aligned_scale):0);
    std::cout<<",\"remainder_plain\":";plain(s.remainder_scale_plaintext);
    std::cout<<",\"remainder_aligned\":";
    number(s.remainder_aligned_scale>0?std::log2(s.remainder_aligned_scale):0);
    std::cout<<"}";
  }
  std::cout<<"],\"double_angle_drops\":[";
  for(std::size_t i=0;i<ep.double_angle_rescale_counts.size();++i) {
    if(i)std::cout<<",";std::cout<<ep.double_angle_rescale_counts[i];
  }
  std::cout<<"],\"fold\":{\"enabled\":"<<(fold?"true":"false")
    <<",\"seed_log\":"<<seed_log<<",\"value_factor_log\":"<<4*seed_log
    <<",\"application_scale\":"<<(fold?std::log2(raw_scale)+4*seed_log:std::log2(raw_scale))
    <<",\"original_coefficients\":[";
  for(std::size_t i=0;i<poly.sine_poly().data().size();++i) {
    if(i)std::cout<<",";auto v=poly.sine_poly().data()[i];
    std::cout<<"["<<v.real()<<","<<v.imag()<<"]";
  }
  std::cout<<"],\"folded_coefficients\":[";
  for(std::size_t i=0;i<folded_polynomial.data().size();++i) {
    if(i)std::cout<<",";auto v=folded_polynomial.data()[i];
    std::cout<<"["<<v.real()<<","<<v.imag()<<"]";
  }
  std::cout<<"],\"original_constants\":["<<std::pow(poly.sqrt_2pi(),2)<<","<<std::pow(poly.sqrt_2pi(),4)
    <<"],\"folded_constants\":["<<std::pow(poly.sqrt_2pi()*(fold?seed:1),2)<<","
    <<std::pow(poly.sqrt_2pi()*(fold?seed:1),4)<<"]}"
    <<",\"q0_log\":"<<std::log2(ctx.crt_context()->q0())
    <<",\"keys_generated\":false,\"gpu_executed\":false}";
  }
  std::cout<<"]"<<std::endl;
 }catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
}
