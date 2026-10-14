// Bounded diagnostic for the existing Bootstrapper under the Agent's fixed tc128 context.
// This is not a Runtime integration test and never performs intermediate decryption.
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/evaluator/evaluator_ckks_base.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <complex>
using namespace poseidon;
int main() {
 const char *stage="context";
 BootstrapTrace trace;
 try {
  const std::vector<std::uint64_t> primes={
   1152921504581419009ULL,1152921504581877761ULL,1152921504583647233ULL,
   1152921504585547777ULL,1152921504586530817ULL,1152921504589938689ULL,
   1152921504592429057ULL,1152921504592822273ULL,1152921504593412097ULL,
   1152921504595640321ULL,1152921504595968001ULL,1152921504597016577ULL,
   1152921504598720513ULL,1152921504606584833ULL};
  std::vector<Modulus> q,p;
  for(std::size_t i=0;i<primes.size();++i)(i<13?q:p).emplace_back(primes[i]);
  PoseidonContext context(ParametersLiteral(CKKS,15,14,45,0,0,Modulus(0),q,p,sec_level_type::tc128));
  auto evaluator=PoseidonFactory::get_instance()->create_ckks_evaluator(context);
  std::cout<<"context=poseidon-agent-n15-q13-p1-tc128-v1 security=tc128"<<std::endl;
  stage="keys";std::cout<<"stage="<<stage<<std::endl;
  KeyGenerator keygen(context);PublicKey pk;keygen.create_public_key(pk);
  RelinKeys relin;keygen.create_relin_keys(relin);
  GaloisKeys galois;keygen.create_galois_keys(galois);
  CKKSEncoder encoder(context);Encryptor encryptor(context,pk);
  Decryptor decryptor(context,keygen.secret_key());
  const std::vector<double> period={0.,.125,-.25,.5};
  std::vector<double> values(16384);
  for(std::size_t i=0;i<values.size();++i)values[i]=period[i%period.size()];
  Plaintext plain;encoder.encode(values,std::exp2(45.),plain);
  Ciphertext encrypted;encryptor.encrypt(plain,encrypted);
  BootstrapConfig config; // Unchanged library defaults, all printed for review.
  config.trace=&trace;
  std::cout<<"bootstrap_config boundary_k="<<config.boundary_k<<" log_message_ratio="<<config.log_message_ratio
   <<" double_angle="<<config.double_angle<<" scaling_log="<<config.scaling_log
   <<" output_ratio="<<config.output_ratio<<" project_real="<<config.project_real
   <<" logical_rescale_count="<<config.logical_rescale_count<<" q0_modulus_count="<<config.q0_modulus_count<<std::endl;
  stage="native_bootstrap";std::cout<<"stage="<<stage<<" input_level="<<encrypted.level()<<" scale_log2="<<std::log2(encrypted.scale())<<std::endl;
  Ciphertext refreshed;
  evaluator->bootstrap(encrypted,refreshed,relin,galois,encoder,config);
  stage="final_decrypt";
  Plaintext decoded_plain;decryptor.decrypt(refreshed,decoded_plain);
  std::vector<std::complex<double>> decoded;encoder.decode(decoded_plain,decoded);
  double maximum=0.;bool passed=true;
  for(std::size_t i=0;i<values.size();++i) {
   double error=std::abs(decoded.at(i).real()-values[i]);maximum=std::max(maximum,error);
   passed &= std::isfinite(error) && error<=1e-5+1e-4*std::abs(values[i]);
  }
  std::cout<<"native_bootstrap_completed=true output_level="<<refreshed.level()
   <<" scale_log2="<<std::log2(refreshed.scale())<<" max_absolute_error="<<maximum
   <<" numerical_passed="<<passed<<" compared_values="<<values.size()<<std::endl;
  return passed?0:1;
 } catch(const std::exception&e) {
  std::cerr<<"blocked_stage="<<stage<<" reason="<<e.what()<<std::endl;
  auto state=[](const char* name,const Ciphertext& value) {
   if(value.size()!=0)std::cerr<<"completed_state="<<name<<" level="<<value.level()<<" scale_log2="<<std::log2(value.scale())<<std::endl;
  };
  state("prepared_q0",trace.prepared_q0);state("raised",trace.raised);
  state("coeff_to_slot",trace.coeff_to_slot_real_aligned);
  state("eval_mod_input",trace.eval_mod_real.input);state("eval_mod_polynomial",trace.eval_mod_real.polynomial_output);
  for(const auto& value:trace.eval_mod_real.double_angle_outputs)state("double_angle",value);
  state("slot_to_coeff",trace.slot_to_coeff);state("final_output",trace.final_output);
  return 2;
 }
}
