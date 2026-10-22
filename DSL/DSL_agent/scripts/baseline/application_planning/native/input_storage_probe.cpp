// Actual Poseidon coefficient allocation; no inference or bootstrap claim.
#include "poseidon/ciphertext.h"
#include "poseidon/poseidon_context.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdexcept>
int main(int argc,char**argv) {
  try {
    if(argc!=15)throw std::runtime_error("fourteen approved modulus values required");
    std::vector<poseidon::Modulus> q,p;
    for(int i=1;i<argc;i++)(i<=13?q:p).emplace_back(std::stoull(argv[i]));
    poseidon::PoseidonContext context(poseidon::ParametersLiteral(
      CKKS,15,14,45,0,0,poseidon::Modulus(0),q,p,poseidon::sec_level_type::tc128));
    auto id=context.crt_context()->parms_id_map().at(12);
    poseidon::Ciphertext first,second;
    first.resize(context,id,2);second.resize(context,id,2);
    auto elements=first.dyn_array().size();
    auto bytes=elements*sizeof(poseidon::Ciphertext::ct_coeff_type);
    if(first.poly_modulus_degree()!=32768 || first.coeff_modulus_size()!=13 ||
       first.size()!=2 || elements!=2*32768*13 || bytes!=6815744 ||
       first.data()==second.data())throw std::runtime_error("input storage contract mismatch");
    std::cout<<"{\"input_coefficient_bytes\":"<<bytes
             <<",\"components\":"<<first.size()<<",\"data_moduli\":"<<first.coeff_modulus_size()
             <<",\"poly_degree\":"<<first.poly_modulus_degree()
             <<",\"coefficient_size\":"<<sizeof(poseidon::Ciphertext::ct_coeff_type)
             <<",\"separate_allocations\":true,\"encrypted_execution\":false}\n";
    return 0;
  } catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
