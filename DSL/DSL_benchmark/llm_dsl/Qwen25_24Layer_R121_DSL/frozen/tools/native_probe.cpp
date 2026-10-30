// 24-block source-adapted (not yet built or CKKS-tested) FinalOnly CKKS probe. The bootstrap binding matches the passed small-graph probe.
// Native SequentialRuntime with an isolated CPU API binding to the validated
// degree-48 bootstrap. No secret key is supplied to the execution API.
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/poseidon_cpu_api.h"
#include "poseidon/evaluator/software/evaluator_ckks_software.h"
#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/runtime.hpp"
#include <nlohmann/json.hpp>
#include "key_payload_stats.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sys/resource.h>
#include <omp.h>
#include "poseidon/basics/memorymanager.h"
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
using Api=poseidon::runtime_api::PoseidonCpuApi;
using Value=poseidon::runtime_api::PoseidonCpuValue;
Json read(const char *p){std::ifstream f(p);if(!f)throw std::runtime_error("Cannot open input");Json j;f>>j;return j;}
std::uint64_t rss(){rusage r{};if(getrusage(RUSAGE_SELF,&r))throw std::runtime_error("getrusage");return std::uint64_t(r.ru_maxrss)*1024;}
double process_cpu_seconds(){rusage u{};getrusage(RUSAGE_SELF,&u);return u.ru_utime.tv_sec+u.ru_utime.tv_usec*1e-6+u.ru_stime.tv_sec+u.ru_stime.tv_usec*1e-6;}
int main(int argc,char **argv){
 if(argc!=7){std::cerr<<"probe PLAN SPEC BUNDLE FIXTURE CONFIG REPORT\n";return 2;}
 Json report={{"status","starting"},{"native_runtime",true},{"native_cpu_api_binding","isolated callback patch"},{"real_ckks",true},{"decrypt_reencrypt_count",0},{"secret_key_supplied_to_execution_api",false},{"mode","FinalOnly"},{"bootstraps",Json::array()}};
 report["complete_decoder_blocks"]=24;report["sequential_tokens"]=2;report["synthetic_weights"]=true;report["rotation_api_mode"]="native_naf";report["reclaim_policy"]="selective";
 report["configured_omp_threads"]=omp_get_max_threads();report["cpu_parallel_scope"]="Per-prime backend; model DAG remains sequential";
 auto save=[&]{report["peak_rss_bytes"]=rss();std::ofstream out(argv[6]);if(!out)throw std::runtime_error("Cannot save report");out<<report.dump(2)<<'\n';};
 try{
  auto plan=fhegpu::RuntimePlanJsonReader::read_file(argv[1]);auto spec=fhegpu::OperatorSpecReader::read_file(argv[2]);
  auto needs=fhegpu::PlanVerifier::verify(plan.plan,spec,false);
  const auto cfg=read(argv[5]).at("bootstrap"),fixture=read(argv[4]);
  if(cfg.at("degree")!=48 || cfg.at("scale_bits")!=43 || cfg.at("output_level")!=5)throw std::runtime_error("Unexpected bootstrap configuration");
  const unsigned logn=cfg.at("log_n");
  poseidon::ParametersLiteral params{CKKS,logn,logn-1,43,(2U*(1U<<logn))/3,0,0,{},{},poseidon::sec_level_type::tc128};
  params.set_log_modulus(cfg.at("q_bits").get<std::vector<uint32_t>>(),cfg.at("p_bits").get<std::vector<uint32_t>>());
  poseidon::PoseidonContext context(params);
  poseidon::CKKSEncoder encoder(context);poseidon::EvaluatorCkksSoftware evaluator(context);
  poseidon::BootstrapWorkspace workspace;
  poseidon::EvalModPoly poly(context,poseidon::CosContinuous,std::exp2(cfg.at("eval_scale_bits").get<int>()),1,cfg.at("log_message_ratio"),cfg.at("double_angle"),cfg.at("k"),0,cfg.at("degree"));
  report["status"]="key-generation";save();std::cout<<"Generating keys"<<std::endl;
  poseidon::KeyGenerator keygen(context);auto pk=std::make_shared<poseidon::PublicKey>();keygen.create_public_key(*pk);
  auto rk=std::make_shared<poseidon::RelinKeys>();keygen.create_relin_keys(*rk);
  auto gk=std::make_shared<poseidon::GaloisKeys>();std::set<int> steps{0};
  for(int step=1;step<int(params.slot());step<<=1){steps.insert(step);steps.insert(-step);}
  keygen.create_galois_keys(std::vector<int>(steps.begin(),steps.end()),*gk);
  report["rotation_key_count"]=gk->size();report["rss_after_keys"]=rss();
  report["rss_after_keys_is_high_water_mark"]=true;
  report["current_rss_after_keys_bytes"]=current_process_rss_bytes();
  report["public_computation_keys"]={
    {"galois",switching_key_payload_stats(*gk)},
    {"relinearization",switching_key_payload_stats(*rk)},
    {"public_key_coefficient_bytes",public_key_cipher_coeff_bytes(pk->data())},
    {"note","Coefficient payload only; excludes allocator/keygen caches/workspaces; no key contents exported"}};

  Api api(spec.spec.context_id,context,rk,gk);
  api.set_native_boot_handler([&](const poseidon::Ciphertext &input,const fhegpu::BootAttrs &attrs){
   if(attrs.implementation!=fhegpu::BootImplementation::Native || attrs.operator_profile!="cpu-bootstrap48-level1-to5" || attrs.target_level!=5 || attrs.target_scale_log2!=43 || attrs.target_components!=2 || input.level()!=1 || input.size()!=2 || std::abs(std::log2(input.scale())-43)>1e-6)throw std::runtime_error("Native bootstrap contract mismatch");
   const auto start=Clock::now();poseidon::Ciphertext output;
   report["bootstraps"].push_back({{"input_level",input.level()},{"status","running"}});save();
   evaluator.bootstrap(input,output,*rk,*gk,encoder,poly,cfg.at("coeff_to_slot_step"),[&](const char *stage,const poseidon::Ciphertext &v){
    const std::string phase=stage;
    if(phase=="coeff_to_slot" || phase=="eval_mod_real" || phase=="slot_to_coeff"){
     if(omp_in_parallel())throw std::runtime_error("Reclaim requires a quiescent stage");
     auto pool=poseidon::MemoryPoolHandle::Global();const auto before=pool.alloc_byte_count();
     const auto reclaim_start=Clock::now();const auto released=pool.qwen_reclaim_idle(16*1024*1024);
     const auto seconds=std::chrono::duration<double>(Clock::now()-reclaim_start).count();
     if(pool.alloc_byte_count()+released!=before)throw std::runtime_error("Reclaim accounting mismatch");
     report["bootstraps"].back()["reclaims"].push_back({{"stage",stage},{"released_bytes",released},{"seconds",seconds}});
    }
    report["bootstraps"].back()["stage"]=stage;report["bootstraps"].back()["stage_level"]=v.level();save();std::cout<<"Boot "<<report["bootstraps"].size()<<" "<<stage<<" level="<<v.level()<<std::endl;
   },&workspace,cfg.value("coeff_to_slot_levels",uint32_t(3)));
   if(output.level()!=5 || output.size()!=2 || std::abs(std::log2(output.scale())-43)>1e-6)throw std::runtime_error("Native bootstrap output metadata mismatch");
   auto &row=report["bootstraps"].back();row["status"]="completed";row["seconds"]=std::chrono::duration<double>(Clock::now()-start).count();row["output_level"]=output.level();row["output_scale_log2"]=std::log2(output.scale());save();return output;
  });
  if(plan.plan.external_inputs.size()!=2 || plan.plan.final_outputs.size()!=29 ||
     fixture.at("inputs").size()!=2 || fixture.at("expected").size()!=29 ||
     fixture.at("num_layers")!=24)throw std::runtime_error("Expected full 24-block, two-token model");
  report["scope"]="24 real-width blocks, 2-token decode, complete vocabulary and KV";
  std::size_t expected_boots=0;const auto plan_json=read(argv[1]);
  for(const auto *section:{"initialization","execution","finalization"})
   for(const auto &op:plan_json.at(section))if(op.value("op",std::string{})=="boot")++expected_boots;
  if(!expected_boots)throw std::runtime_error("Expected a native-bootstrap plan");
  report["synthetic_weights"]=true;report["expected_boot_count"]=expected_boots;
  std::unordered_map<fhegpu::ValueId,Value> inputs;
  poseidon::Encryptor encryptor(context,*pk);
  for(size_t index=0;index<plan.plan.external_inputs.size();++index){
   const auto id=plan.plan.external_inputs[index];const fhegpu::ValueDesc *in=nullptr;
   for(const auto &v:plan.plan.values)if(v.id==id)in=&v;
   if(!in)throw std::runtime_error("Input descriptor missing");
   const auto slots=fixture.at("inputs").at(index).get<std::vector<double>>();
   if(slots.size()!=params.slot())throw std::runtime_error("Input slot count mismatch");
   poseidon::Plaintext pt;encoder.encode(slots,context.crt_context()->parms_id_map().at(in->level),std::ldexp(1.,in->scale_log2),pt);
   poseidon::Ciphertext ct;encryptor.encrypt(pt,ct);
   inputs.emplace(id,Value::from_ciphertext(std::move(ct)));
  }
  fhegpu::SequentialRuntime<Api> runtime(0,1,0,api);fhegpu::RuntimeResources resources{spec,std::filesystem::path(argv[3]),false};
  report["status"]="executing";save();const auto start=Clock::now();const double cpu_start=process_cpu_seconds();
  const auto result=runtime.run(plan,resources,inputs,fhegpu::DiffMode::FinalOnly);
  report["runtime_seconds"]=std::chrono::duration<double>(Clock::now()-start).count();
  report["runtime_cpu_seconds"]=process_cpu_seconds()-cpu_start;report["runtime_average_cpu_cores"]=report["runtime_cpu_seconds"].get<double>()/report["runtime_seconds"].get<double>();
  report["runtime_timing"]={{"encode_calls",result.timing.encode_calls},{"bundle_read_calls",result.timing.bundle_read_calls},{"bundle_read_bytes",result.timing.bundle_read_bytes},{"raw_peak_bytes",result.timing.raw_peak_bytes},{"bundle_resident_bytes",result.timing.bundle_resident_bytes},{"boot_calls",result.timing.boot_calls}};
  poseidon::Decryptor decryptor(context,keygen.secret_key());
  if(result.values.size()!=29)throw std::runtime_error("FinalOnly output count mismatch");
  report["status"]="checking-outputs";report["outputs"]=Json::array();save();
  double error=0,imag=0,tail=0;
  for(size_t index=0;index<plan.plan.final_outputs.size();++index){
   poseidon::Plaintext output;
   decryptor.decrypt(result.values.at(plan.plan.final_outputs[index]).value.ciphertext(),output);
   std::vector<std::complex<double>> actual;encoder.decode(output,actual);
   const auto expected=fixture.at("expected").at(index).get<std::vector<double>>();
   if(actual.size()!=params.slot() || expected.empty() || expected.size()>actual.size())throw std::runtime_error("Output shape mismatch");
   double row_error=0,row_imag=0,row_tail=0;
   for(size_t i=0;i<actual.size();++i){
    if(!std::isfinite(actual[i].real()) || !std::isfinite(actual[i].imag()))throw std::runtime_error("Nonfinite output");
    if(i<expected.size())row_error=std::max(row_error,std::abs(actual[i].real()-expected[i]));
    else row_tail=std::max(row_tail,std::abs(actual[i].real()));
    row_imag=std::max(row_imag,std::abs(actual[i].imag()));
   }
   error=std::max(error,row_error);imag=std::max(imag,row_imag);tail=std::max(tail,row_tail);
   report["outputs"].push_back({{"name",fixture.at("output_names").at(index)},{"max_abs_error",row_error},
     {"max_imaginary",row_imag},{"max_abs_tail",row_tail}});
   save();
  }
  if(report["bootstraps"].size()!=expected_boots)throw std::runtime_error("Unexpected completed bootstrap count");
  report["diagnostic_ciphertext_copies"]=0;
  report["max_abs_tail"]=tail;
  std::uint64_t plainbytes=0,cipherbytes=0;
  for(const auto &entry:result.values){const auto &v=entry.second.value;if(v.kind()==fhegpu::ValueKind::Plaintext)plainbytes+=v.plaintext().coeff_count()*8ULL;else{const auto &c=v.ciphertext();cipherbytes+=c.size()*c.poly_modulus_degree()*c.coeff_modulus_size()*8ULL;}}
  report["retained_value_count"]=result.values.size();report["retained_plain_coeff_bytes"]=plainbytes;report["retained_cipher_coeff_bytes"]=cipherbytes;
  report["max_abs_error"]=error;report["max_imaginary"]=imag;report["threshold"]=0.001;report["passed"]=error<0.001 && imag<0.001 && tail<0.001;
  report["status"]=report["passed"].get<bool>()?"passed":"failed_precision";save();std::cout<<report.dump(2)<<'\n';return report["passed"].get<bool>()?0:1;
 }catch(const std::exception &e){report["status"]="failed";report["error"]=e.what();save();std::cerr<<e.what()<<'\n';return 1;}
}
