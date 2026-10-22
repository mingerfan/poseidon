// Read-only evaluation-key verification for the new packing ABI only.
#include <seal/seal.h>
#include <seal/util/galois.h>
#include <cstdint>
#include <fstream>

extern "C" int verify_packed_galois_file(const char *parameters,const char *galois,
    const std::int64_t *steps,std::uint64_t count,std::uint64_t period) {
  try {
    if(!parameters || !galois || (count&&!steps) || count>8 || period<4 || period>256 || (period&(period-1))) return 1;
    seal::EncryptionParameters parms;
    std::ifstream p(parameters,std::ios::binary);parms.load(p);
    seal::SEALContext context(parms,true,seal::sec_level_type::tc128);
    if(!context.parameters_set() || parms.poly_modulus_degree()!=32768 || parms.coeff_modulus().size()!=14) return 1;
    for(const auto &q:parms.coeff_modulus()) if(q.bit_count()!=60) return 1;
    seal::GaloisKeys keys;
    std::ifstream k(galois,std::ios::binary);keys.load(context,k);
    for(std::uint64_t i=0;i<count;++i) {
      if(steps[i]<=0 || static_cast<std::uint64_t>(steps[i])>=period || (steps[i]&(steps[i]-1))) return 1;
      auto element=context.key_context_data()->galois_tool()->get_elt_from_step(static_cast<int>(steps[i]));
      if(!keys.has_key(element)) return 2;
    }
    return 0;
  } catch(...){return 1;}
}
