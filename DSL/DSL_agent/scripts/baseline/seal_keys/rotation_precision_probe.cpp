// Diagnostic only: not an Agent backend, no simulated bootstrap, no saved keys.
// Compare SEAL error at each stage using the same input ciphertext and key set.
#include <seal/seal.h>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Vec=std::vector<double>;
void emit(const char* name,int trial,int batch,const Vec& actual,const Vec& expected,
          std::size_t level,double scale) {
  double max4=0,mae4=0,maxall=0;
  for(std::size_t i=0;i<actual.size();++i) {
    auto error=std::abs(actual[i]-expected[i]);
    maxall=std::max(maxall,error);
    if(i<4) { max4=std::max(max4,error);mae4+=error/4; }
  }
  std::cout << "{\"stage\":\"" << name << "\",\"trial\":" << trial << ",\"batch\":" << batch
    << ",\"data_moduli\":" << level << ",\"log2_scale\":" << std::log2(scale)
    << ",\"max_abs_first4\":" << max4 << ",\"mae_first4\":" << mae4
    << ",\"max_abs_all_slots\":" << maxall << ",\"error_first4\":[";
  for(int i=0;i<4;++i) { if(i) std::cout << ','; std::cout << actual[i]-expected[i]; }
  std::cout << "]}\n";
}
int main() {
  try {
    std::cout << std::setprecision(17);
    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    parms.set_poly_modulus_degree(32768);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(32768,std::vector<int>(14,60)));
    seal::SEALContext ctx(parms,true,seal::sec_level_type::tc128);
    if(!ctx.parameters_set()) throw std::runtime_error(ctx.parameter_error_message());
    const double scale=std::ldexp(1.0,40);
    for(int trial=0;trial<2;++trial) {
      seal::KeyGenerator keys(ctx);
      seal::PublicKey pub;keys.create_public_key(pub);
      seal::GaloisKeys gal;keys.create_galois_keys(std::vector<int>{1,2,3},gal);
      seal::CKKSEncoder encoder(ctx);
      seal::Encryptor encryptor(ctx,pub);
      seal::Decryptor decryptor(ctx,keys.secret_key());
      seal::Evaluator evaluator(ctx);
      const std::array<std::array<double,4>,3> batches={{{0,0,0,0},{.5,-1,.25,-.75},{-1,1,-1,1}}};
      for(int batch=0;batch<3;++batch) {
        Vec input(encoder.slot_count());
        for(std::size_t i=0;i<input.size();++i) input[i]=batches[batch][i%4];
        seal::Plaintext plain;
        encoder.encode(input,scale,plain);
        Vec decoded;encoder.decode(plain,decoded);
        emit("encode_decode",trial,batch,decoded,input,13,scale);
        auto observe=[&](const char* name,const seal::Ciphertext& c,const Vec& ref) {
          seal::Plaintext p;decryptor.decrypt(c,p);Vec a;encoder.decode(p,a);
          emit(name,trial,batch,a,ref,c.coeff_modulus_size(),c.scale());
        };
        seal::Ciphertext encrypted;encryptor.encrypt(plain,encrypted);
        observe("encrypted_input",encrypted,input);
        seal::Ciphertext low=encrypted;
        evaluator.mod_switch_to_inplace(low,ctx.last_parms_id());
        observe("modswitch_to_level1",low,input);
        seal::Ciphertext low_sum=low,high_sum=encrypted;
        Vec sum=input;
        for(int step=1;step<=3;++step) {
          Vec rotated(input.size());
          for(std::size_t i=0;i<input.size();++i) { rotated[i]=input[(i+step)%input.size()];sum[i]+=rotated[i]; }
          seal::Ciphertext lr,hr;
          evaluator.rotate_vector(low,step,gal,lr);
          evaluator.rotate_vector(encrypted,step,gal,hr);
          std::string label="rotate_level1_"+std::to_string(step);
          observe(label.c_str(),lr,rotated);
          label="rotate_level13_"+std::to_string(step);
          observe(label.c_str(),hr,rotated);
          evaluator.add_inplace(low_sum,lr);
          evaluator.add_inplace(high_sum,hr);
        }
        observe("sum_level1",low_sum,sum);
        observe("sum_level13",high_sum,sum);
        evaluator.mod_switch_to_inplace(high_sum,ctx.last_parms_id());
        observe("sum_high_then_modswitch",high_sum,sum);
      }
    }
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << '\n';return 1; }
}
