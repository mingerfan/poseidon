// Separate opt-in key setup for periodic-packed-v1. Never changes legacy keys.
#include <seal/seal.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> void save_new(const std::filesystem::path &path,const T &value) {
  if(std::filesystem::exists(path)) throw std::runtime_error("Preserving existing key file");
  std::ofstream out(path,std::ios::binary);
  out.exceptions(std::ios::badbit|std::ios::failbit);
  value.save(out);
}

int main(int argc,char **argv) {
  try {
    if(argc!=3) throw std::runtime_error("Usage: seal_packed_keys EMPTY_DIRECTORY PERIOD");
    std::size_t consumed=0;
    const int period=std::stoi(argv[2],&consumed);
    if(consumed!=std::string(argv[2]).size() || period<4 || period>256 || (period&(period-1)))
      throw std::runtime_error("Period must be a power of two from4 to256");
    const std::filesystem::path dir(argv[1]);
    if(!std::filesystem::is_directory(dir) || !std::filesystem::is_empty(dir))
      throw std::runtime_error("Requires a fresh empty key directory");
    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    parms.set_poly_modulus_degree(32768);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(32768,std::vector<int>(14,60)));
    seal::SEALContext context(parms,true,seal::sec_level_type::tc128);
    if(!context.parameters_set() || context.first_context_data()->parms().coeff_modulus().size()!=13)
      throw std::runtime_error("Stock SEAL security profile check failed");
    seal::KeyGenerator generator(context);
    seal::PublicKey pub;generator.create_public_key(pub);
    seal::RelinKeys relin;generator.create_relin_keys(relin);
    std::vector<int> steps;
    for(int step=1;step<period;step*=2) steps.push_back(step);
    seal::GaloisKeys gal;generator.create_galois_keys(steps,gal);
    save_new(dir/"parm.seal",parms);save_new(dir/"pub.seal",pub);
    save_new(dir/"sec.seal",generator.secret_key());save_new(dir/"relin.seal",relin);save_new(dir/"gal.seal",gal);
    std::cout<<"{\"seal_version\":\""<<SEAL_VERSION<<"\",\"polynomial_degree\":32768,\"slots\":16384,"
       "\"security_check\":\"tc128\",\"parameters_set\":true,\"data_modulus_count\":13,"
       "\"execution_abi\":\"periodic-packed-v1\",\"slot_period\":"<<period<<",\"rotation_steps\":[";
    for(std::size_t i=0;i<steps.size();++i){if(i)std::cout<<',';std::cout<<steps[i];}
    std::cout<<"],\"modulus_bits\":[";
    for(std::size_t i=0;i<parms.coeff_modulus().size();++i){if(i)std::cout<<',';std::cout<<parms.coeff_modulus()[i].bit_count();}
    std::cout<<"],\"modulus_values\":[";
    for(std::size_t i=0;i<parms.coeff_modulus().size();++i){if(i)std::cout<<',';std::cout<<'\"'<<parms.coeff_modulus()[i].value()<<'\"';}
    std::cout<<"]}\n";
    return 0;
  } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
