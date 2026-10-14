// Isolated approved parameter experiment; not an Agent or Runtime backend.
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/evaluator/evaluator_ckks_base.h"
#include "approved_parameters.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cstdint>
#include <stdexcept>
using namespace poseidon;

void state(const char* name, const Ciphertext& c) {
 if(c.size()) std::cout<<"state="<<name<<" level="<<c.level()<<" scale_log2="<<std::log2(c.scale())<<std::endl;
}
void trace_states(const BootstrapTrace& t) {
 state("prepared_q0",t.prepared_q0);state("raised",t.raised);
 state("coeff_to_slot",t.coeff_to_slot_real_aligned);
 state("eval_mod_input",t.eval_mod_real.input);
 state("eval_mod_polynomial",t.eval_mod_real.polynomial_output);
 for(const auto& c:t.eval_mod_real.double_angle_outputs)state("double_angle",c);
 state("slot_to_coeff",t.slot_to_coeff);state("final_output",t.final_output);
}
bool compare(const char* group,const char* stage,const Ciphertext& c,
             const std::vector<double>& expected,Decryptor& decryptor,CKKSEncoder& encoder) {
 Plaintext p;decryptor.decrypt(c,p);
 std::vector<std::complex<double>> actual;encoder.decode(p,actual);
 if(actual.size()!=expected.size())throw std::runtime_error("decoded slot count mismatch");
 double maximum=0,mae=0,max_relative=0,max_imaginary=0;std::size_t failures=0;
 for(std::size_t i=0;i<expected.size();++i) {
  double error=std::abs(actual[i].real()-expected[i]);
  bool pass=std::isfinite(error)&&error<=1e-5+1e-4*std::abs(expected[i]);
  maximum=std::max(maximum,error);mae+=error;failures+=!pass;
  if(expected[i]!=0)max_relative=std::max(max_relative,error/std::abs(expected[i]));
  max_imaginary=std::max(max_imaginary,std::abs(actual[i].imag()));
  std::cout<<"value group="<<group<<" stage="<<stage<<" index="<<i
           <<" reference="<<expected[i]<<" actual="<<actual[i].real()
           <<" absolute_error="<<error<<" passed="<<pass<<"\n";
 }
 std::cout<<"comparison group="<<group<<" stage="<<stage
          <<" compared="<<expected.size()<<" failed="<<failures<<" max_absolute_error="<<maximum
          <<" mae="<<mae/expected.size()<<" max_relative_error="<<max_relative
          <<" max_imaginary="<<max_imaginary<<" passed="<<(failures==0)<<std::endl;
 return failures==0;
}
int main() {
 std::cout<<std::setprecision(17);
 const char* stage="context";const char* group="none";BootstrapTrace trace;
 try {
  std::vector<Modulus> q,p;
  for(auto v:approved_q)q.emplace_back(v);
  for(auto v:approved_p)p.emplace_back(v);
  if(CoeffModulus::MaxBitCount(32768,sec_level_type::tc128)!=881)
   throw std::runtime_error("security table differs from approved context");
  PoseidonContext context(ParametersLiteral(CKKS,15,14,40,0,0,Modulus(0),q,p,sec_level_type::tc128));
  auto evaluator=PoseidonFactory::get_instance()->create_ckks_evaluator(context);
  std::cout<<"context_admitted=true degree=32768 q_count="<<q.size()<<" p_count="<<p.size()
           <<" exact_qp_bits=876 security=tc128 table_max_bits=881"<<std::endl;
  stage="keys";std::cout<<"stage=keys"<<std::endl;
  KeyGenerator keygen(context);PublicKey pk;keygen.create_public_key(pk);
  RelinKeys relin;keygen.create_relin_keys(relin);
  GaloisKeys galois;keygen.create_galois_keys(galois);
  CKKSEncoder encoder(context);Encryptor encryptor(context,pk);
  Decryptor decryptor(context,keygen.secret_key());
  BootstrapConfig config;config.scaling_log=48;config.trace=&trace;
  std::cout<<"bootstrap_config boundary_k="<<config.boundary_k<<" log_message_ratio="<<config.log_message_ratio
           <<" double_angle="<<config.double_angle<<" scaling_log="<<config.scaling_log
           <<" output_ratio="<<config.output_ratio<<" project_real="<<config.project_real
           <<" logical_rescale_count="<<config.logical_rescale_count
           <<" q0_modulus_count="<<config.q0_modulus_count<<std::endl;
  const char* groups[]={"zero","signed","fixed_random","boundary"};
  for(int g=0;g<4;++g) {
   group=groups[g];std::vector<double> values(16384);
   const double signed_values[]={.125,-.25,.375,-.5};
   const double boundary_values[]={-.5,.5,-.499,.499,0.,1.0/1024,-1.0/1024,.25};
   std::uint32_t seed=233;
   for(std::size_t i=0;i<values.size();++i) {
    seed=1664525U*seed+1013904223U;
    values[i]=g==0?0.:g==1?signed_values[i%4]:g==2?
       (static_cast<double>(seed>>8)/16777215.-.5):boundary_values[i%8];
   }
   stage="encrypt";Plaintext plain;encoder.encode(values,std::exp2(40.),plain);
   Ciphertext input;encryptor.encrypt(plain,input);state("input",input);
   stage="first_refresh";trace=BootstrapTrace{};Ciphertext refreshed;
   std::cout<<"begin group="<<group<<" stage="<<stage<<std::endl;
   evaluator->bootstrap(input,refreshed,relin,galois,encoder,config);
   trace_states(trace);
   if(!compare(group,stage,refreshed,values,decryptor,encoder))return 1;
   stage="square_rescale";
   if(refreshed.level()<1)throw std::runtime_error("no remaining rescale level after refresh");
   Ciphertext product,squared;
   evaluator->multiply_relin(refreshed,refreshed,product,relin);
   evaluator->rescale(product,squared);state("square_rescale",squared);
   std::vector<double> expected(values.size());
   for(std::size_t i=0;i<values.size();++i)expected[i]=values[i]*values[i];
   if(!compare(group,stage,squared,expected,decryptor,encoder))return 1;
   stage="second_refresh";trace=BootstrapTrace{};Ciphertext again;
   evaluator->bootstrap(squared,again,relin,galois,encoder,config);
   trace_states(trace);
   if(!compare(group,stage,again,expected,decryptor,encoder))return 1;
  }
  std::cout<<"all_four_groups_passed=true"<<std::endl;return 0;
 } catch(const std::exception& e) {
  std::cout<<"blocked_group="<<group<<" blocked_stage="<<stage<<" reason="<<e.what()<<std::endl;
  trace_states(trace);return 2;
 }
}
