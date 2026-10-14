// Independent fixed-SEAL boundary tests. No HEVM/compiler changes or saved keys.
#include <seal/seal.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <string>
int main(int argc,char**argv) {
 try {
  if(argc!=2)return 2;
  seal::EncryptionParameters p(seal::scheme_type::ckks);
  p.set_poly_modulus_degree(32768);
  p.set_coeff_modulus(seal::CoeffModulus::Create(32768,std::vector<int>(14,60)));
  seal::SEALContext c(p,true,seal::sec_level_type::tc128);
  if(!c.parameters_set()||std::string(SEAL_VERSION)!="4.0.0")return 3;
  {std::ofstream f(std::string(argv[1])+"/probe-parm.seal",std::ios::binary);p.save(f);}
  seal::CKKSEncoder enc(c);seal::Evaluator ev(c);seal::KeyGenerator kg(c);
  seal::PublicKey pub;kg.create_public_key(pub);
  seal::RelinKeys rk;kg.create_relin_keys(rk);
  seal::Encryptor encryptor(c,pub);seal::Decryptor decryptor(c,kg.secret_key());
  auto at=[&](size_t n){auto d=c.first_context_data();while(d&&d->parms().coeff_modulus().size()!=n)d=d->next_context_data();if(!d)throw std::runtime_error("level");return d->parms_id();};
  auto encode=[&](int bits,size_t level=13) {seal::Plaintext t;enc.encode(std::vector<double>{0.125,-0.25,0.5,-0.75},at(level),std::ldexp(1.,bits),t);return t;};
  auto cipher=[&](int bits) {seal::Ciphertext t;encryptor.encrypt(encode(bits),t);return t;};
  int failed=0;std::cout<<std::setprecision(17);
  auto test=[&](const char*name,bool want,const std::function<void()>&f){
   bool ok=true;std::string error;
   try{f();}catch(const std::exception&e){ok=false;error=e.what();}
   bool match=ok==want;if(!match)++failed;
   std::cout<<"{\"name\":\""<<name<<"\",\"expected_accept\":"<<(want?"true":"false")
    <<",\"accepted\":"<<(ok?"true":"false")<<",\"passed\":"<<(match?"true":"false")
    <<",\"diagnostic\":\""<<error<<"\"}\n";
  };
  test("encode_above_180",true,[&]{encode(200);});
  test("encode_headroom_778",true,[&]{encode(778);});
  test("encode_headroom_779",false,[&]{encode(779);});
  test("direct_encode_level1_59",false,[&]{encode(59,1);});
  test("runtime_encode_then_modswitch_level1_59",true,[&]{auto t=encode(59);ev.mod_switch_to_inplace(t,at(1));});
  test("runtime_encode_then_modswitch_level1_75",false,[&]{auto t=encode(75);ev.mod_switch_to_inplace(t,at(1));});
  test("intermediate_multiply_overflow_780",false,[&]{auto a=cipher(390);ev.square_inplace(a);});
  test("multiply_above_180_numerical",true,[&]{
   auto a=cipher(200);ev.square_inplace(a);ev.relinearize_inplace(a,rk);
   seal::Plaintext t;decryptor.decrypt(a,t);std::vector<double> v;enc.decode(t,v);
   double x[]={.125,-.25,.5,-.75};double error=0;
   for(int i=0;i<4;++i)error=std::max(error,std::abs(v[i]-x[i]*x[i]));
   if(error>1e-5)throw std::runtime_error("Numerical mismatch");
   std::cout<<"{\"observation\":\"high_scale_square\",\"log2_scale\":"<<std::log2(a.scale())<<",\"max_abs\":"<<error<<"}\n";
  });
  test("actual_prime_rescale",true,[&]{
   auto a=cipher(40);ev.square_inplace(a);ev.relinearize_inplace(a,rk);ev.rescale_to_next_inplace(a);
   double expected=std::ldexp(1.,80)/static_cast<double>(p.coeff_modulus()[12].value());
   if(a.scale()!=expected)throw std::runtime_error("Propagation mismatch");
   std::cout<<"{\"observation\":\"rescale\",\"scale\":"<<a.scale()<<",\"data_moduli\":"<<a.coeff_modulus_size()<<"}\n";
  });
  test("cipher_modswitch_level1_75",false,[&]{auto a=cipher(75);ev.mod_switch_to_inplace(a,at(1));});
  test("rescale_chain_exhausted",false,[&]{auto a=cipher(40);ev.mod_switch_to_inplace(a,at(1));ev.rescale_to_next_inplace(a);});
  test("decode_level1_60",false,[&]{auto t=encode(40,1);t.scale()=std::ldexp(1.,60);std::vector<double>v;enc.decode(t,v);});
  test("add_without_runtime_override",false,[&]{auto a=cipher(40),b=cipher(41);ev.add_inplace(a,b);});
  test("runtime_override_changes_value",true,[&]{
   auto a=cipher(40),b=cipher(41);a.scale()=b.scale();ev.add_inplace(a,b);
   seal::Plaintext t;decryptor.decrypt(a,t);std::vector<double> v;enc.decode(t,v);
   if(std::abs(v[0]-.1875)>1e-5||std::abs(v[0]-.25)<1e-4)throw std::runtime_error("Expected semantic distortion absent");
   std::cout<<"{\"observation\":\"scale_override\",\"actual\":"<<v[0]<<",\"wanted\":0.25}\n";
  });
  return failed?1:0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 4;}
}
