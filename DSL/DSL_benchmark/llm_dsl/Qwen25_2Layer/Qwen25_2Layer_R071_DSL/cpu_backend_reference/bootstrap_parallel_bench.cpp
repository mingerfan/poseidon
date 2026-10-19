// Isolated CPU backend throughput/precision comparison. No graph rewrite.
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/evaluator/software/evaluator_ckks_software.h"
#include <nlohmann/json.hpp>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sys/resource.h>
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
double cpu_seconds(){rusage u{};getrusage(RUSAGE_SELF,&u);return u.ru_utime.tv_sec+u.ru_utime.tv_usec*1e-6+u.ru_stime.tv_sec+u.ru_stime.tv_usec*1e-6;}
uint64_t peak_rss(){rusage u{};getrusage(RUSAGE_SELF,&u);return uint64_t(u.ru_maxrss)*1024;}
int main(int argc,char**argv){
 if(argc!=3){std::cerr<<"bench BOOT_CONFIG REPORT\n";return 2;}
 Json report={{"status","starting"},{"scope","Isolated RNSPoly NTT and pointwise parallel bootstrap benchmark; not full model"},
  {"real_ckks",true},{"decrypt_reencrypt_count",0},{"bootstraps",Json::array()},{"runs",Json::array()}};
 auto save=[&]{report["peak_rss_bytes"]=peak_rss();std::ofstream f(argv[2]);f<<report.dump(2);};
 try{
  Json full;std::ifstream f(argv[1]);f>>full;auto cfg=full.at("bootstrap");
  if(cfg.at("degree")!=48 || cfg.at("scale_bits")!=43 || cfg.at("output_level")!=5)throw std::runtime_error("Unexpected original bootstrap contract");
  omp_set_dynamic(0);omp_set_max_active_levels(1);omp_set_num_threads(1);
  report["omp_binding"]=std::getenv("OMP_PROC_BIND")?std::getenv("OMP_PROC_BIND"):"unset";
  report["omp_places"]=std::getenv("OMP_PLACES")?std::getenv("OMP_PLACES"):"unset";
  const unsigned logn=cfg.at("log_n");
  poseidon::ParametersLiteral params{CKKS,logn,logn-1,43,(2U*(1U<<logn))/3,0,0,{},{},poseidon::sec_level_type::tc128};
  params.set_log_modulus(cfg.at("q_bits").get<std::vector<uint32_t>>(),cfg.at("p_bits").get<std::vector<uint32_t>>());
  poseidon::PoseidonContext context(params);poseidon::CKKSEncoder encoder(context);poseidon::EvaluatorCkksSoftware evaluator(context);
  poseidon::BootstrapWorkspace workspace;
  poseidon::EvalModPoly poly(context,poseidon::CosContinuous,std::exp2(cfg.at("eval_scale_bits").get<int>()),1,cfg.at("log_message_ratio"),cfg.at("double_angle"),cfg.at("k"),0,cfg.at("degree"));
  report["status"]="key-generation";save();
  poseidon::KeyGenerator keygen(context);poseidon::PublicKey pk;keygen.create_public_key(pk);
  poseidon::RelinKeys rk;keygen.create_relin_keys(rk);
  poseidon::GaloisKeys gk;std::set<int> steps{0};
  for(int step=1;step<int(params.slot());step<<=1){steps.insert(step);steps.insert(-step);}
  keygen.create_galois_keys(std::vector<int>(steps.begin(),steps.end()),gk);
  std::vector<double> expected(params.slot());
  for(size_t i=0;i<expected.size();++i)expected[i]=i<expected.size()/4?0.:-2.+4.*(i-expected.size()/4)/(expected.size()*3/4-1);
  poseidon::Plaintext pt;encoder.encode(expected,context.crt_context()->parms_id_map().at(1),std::ldexp(1.,43),pt);
  poseidon::Encryptor encryptor(context,pk);poseidon::Ciphertext input;encryptor.encrypt(pt,input);
  poseidon::Decryptor decryptor(context,keygen.secret_key());
  std::vector<uint64_t> serial_coefficients;bool passed=true;double worst=0,worst_imag=0;
  int sequence[]={1,1,8,16,32,1};
  for(int iteration=0;iteration<6;++iteration){
   int threads=sequence[iteration];omp_set_num_threads(threads);
   report["status"]="executing";report["requested_threads"]=threads;
   report["bootstraps"].push_back({{"status","running"},{"threads",threads},{"warmup",iteration==0}});save();
   poseidon::Ciphertext output;const auto begin=Clock::now();double cpu=cpu_seconds();
   evaluator.bootstrap(input,output,rk,gk,encoder,poly,cfg.at("coeff_to_slot_step"),[&](const char*stage,const poseidon::Ciphertext&v){
     report["bootstraps"].back()["checkpoints"].push_back({{"stage",stage},{"seconds_since_boot_start",std::chrono::duration<double>(Clock::now()-begin).count()},{"cpu_seconds_since_boot_start",cpu_seconds()-cpu},{"level",v.level()}});
     report["bootstraps"].back()["stage"]=stage;report["bootstraps"].back()["stage_level"]=v.level();save();
   },&workspace,cfg.value("coeff_to_slot_levels",uint32_t(3)));
   const double elapsed=std::chrono::duration<double>(Clock::now()-begin).count();cpu=cpu_seconds()-cpu;
   if(output.level()!=5 || output.size()!=2 || std::abs(std::log2(output.scale())-43)>1e-6)throw std::runtime_error("Unexpected output metadata");
   const size_t count=output.size()*output.poly_modulus_degree()*output.coeff_modulus_size();
   if(iteration==1)serial_coefficients.assign(output.data(),output.data()+count);
   const bool equal=iteration<2 || (serial_coefficients.size()==count && std::equal(serial_coefficients.begin(),serial_coefficients.end(),output.data()));
   poseidon::Plaintext decrypted;decryptor.decrypt(output,decrypted);
   std::vector<std::complex<double>> actual;encoder.decode(decrypted,actual);if(actual.size()!=expected.size())throw std::runtime_error("Slot mismatch");
   double error=0,imag=0,zeros=0;
   for(size_t i=0;i<actual.size();++i){if(!std::isfinite(actual[i].real())||!std::isfinite(actual[i].imag()))throw std::runtime_error("Nonfinite output");
     error=std::max(error,std::abs(actual[i].real()-expected[i]));imag=std::max(imag,std::abs(actual[i].imag()));if(i<actual.size()/4)zeros=std::max(zeros,std::abs(actual[i].real()));}
   worst=std::max(worst,error);worst_imag=std::max(worst_imag,imag);passed=passed&&equal&&error<1e-3&&imag<1e-3;
   report["runs"].push_back({{"threads",threads},{"warmup",iteration==0},{"seconds",elapsed},{"cpu_seconds",cpu},{"average_cpu_cores",cpu/elapsed},
     {"bitwise_equal_to_serial",equal},{"serial_comparison_performed",iteration>=2},{"max_abs_error",error},{"max_imaginary",imag},{"zero_quarter_max_abs",zeros}});
   report["bootstraps"].back()["status"]="completed";report["bootstraps"].back()["seconds"]=elapsed;save();
  }
  report["passed"]=passed;report["max_abs_error"]=worst;report["max_imaginary"]=worst_imag;report["threshold"]=1e-3;
  report["status"]=passed?"passed":"failed_precision_or_equivalence";save();return passed?0:1;
 }catch(const std::exception&e){report["status"]="failed";report["error"]=e.what();save();std::cerr<<e.what()<<'\n';return 1;}
}
