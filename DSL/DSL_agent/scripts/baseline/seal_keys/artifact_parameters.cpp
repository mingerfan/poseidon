// Read-only checker. Does not load keys, evaluate, or modify the HEVM runtime.
#include <seal/seal.h>
#include <fstream>
#include <cstdint>
#include <string>
extern "C" int inspect_seal_parameters(const char *path,uint64_t *moduli,uint64_t *bits,uint64_t *indices) {
 try {
  if(!path||!moduli||!bits||!indices||std::string(SEAL_VERSION)!="4.0.0")return 1;
  std::ifstream f(path,std::ios::binary); seal::EncryptionParameters p;p.load(f);
  if(f.peek()!=std::char_traits<char>::eof())return 2;
  seal::SEALContext c(p,true,seal::sec_level_type::tc128);
  if(!c.parameters_set()||p.scheme()!=seal::scheme_type::ckks||p.poly_modulus_degree()!=32768||p.coeff_modulus().size()!=14)return 3;
  for(size_t i=0;i<14;++i)moduli[i]=p.coeff_modulus()[i].value();
  size_t count=0;
  for(auto d=c.first_context_data();d;d=d->next_context_data()){
   size_t n=d->parms().coeff_modulus().size();
   if(!n||n>13)return 4;
   bits[n-1]=d->total_coeff_modulus_bit_count();indices[n-1]=d->chain_index();++count;
  }
  return count==13?0:5;
 }catch(...){return 6;}
}
