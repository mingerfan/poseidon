#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/runtime_api/poseidon_cpu_api.h"
#include "poseidon/runtime_api/rotation_key_basis.h"
#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/runtime.hpp"
#include "runtime/verifier.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <set>
#include <cmath>
#include <iostream>
using Json=nlohmann::json;
using namespace poseidon;
using namespace poseidon::runtime_api;
static void require(bool ok,const char* reason) {if(!ok) throw std::runtime_error(reason);}
static PoseidonContext context() {
 const std::vector<std::uint64_t> primes={
 1152921504581419009ULL,1152921504581877761ULL,1152921504583647233ULL,
 1152921504585547777ULL,1152921504586530817ULL,1152921504589938689ULL,
 1152921504592429057ULL,1152921504592822273ULL,1152921504593412097ULL,
 1152921504595640321ULL,1152921504595968001ULL,1152921504597016577ULL,
 1152921504598720513ULL,1152921504606584833ULL};
 std::vector<Modulus> q,p;
 for(std::size_t i=0;i<primes.size();++i)(i<13?q:p).emplace_back(primes[i]);
 return PoseidonContext(ParametersLiteral(CKKS,15,14,45,0,0,Modulus(0),q,p,sec_level_type::tc128));
}
int main(int argc,char**argv) {try {
 require(argc==6,"expected plan spec bundle inputs output");
 auto loaded=fhegpu::RuntimePlanJsonReader::read_file(argv[1]);
 auto spec=fhegpu::OperatorSpecReader::read_file(argv[2]);
 require(spec.spec.poly_degree==32768 && spec.spec.rns_moduli_log2==std::vector<int>(13,60)
  && spec.spec.default_scale_log2==45 && spec.spec.level_upper_bound==12,"fixed parameter identity");
 auto req=fhegpu::PlanVerifier::verify(loaded.plan,spec,false);
 for(auto cap:req.capabilities)require(cap!=fhegpu::RequiredCapability::BootNative &&
   cap!=fhegpu::RequiredCapability::BootDecryptReencrypt,"bootstrap unavailable");
 for(auto &v:loaded.plan.values)require(v.place.kind==fhegpu::PlaceKind::Host && v.place.rank==0,"CPU Host only");
 // Validate the complete input protocol before generating any key material.
 Json input;std::ifstream(argv[4])>>input;
 bool wide=input.is_object();std::size_t decode_slots=256,period_size=0;
 const Json *samples=&input;
 if(wide) {
  require(input.size()==7 && input.contains("format") && input.contains("execution_abi") &&
    input.contains("input_slot_period") && input.contains("decode_slots") &&
    input.contains("input_ciphertexts") && input.contains("output_ciphertexts") &&
    input.contains("samples"),"wide input protocol fields");
  require(input["format"]=="poseidon-agent-inputs-v2" &&
    input["execution_abi"]=="unified-capacity-chunked-inputs-v4","wide input protocol identity");
  const auto &p=input["input_slot_period"];
  require(p.is_number_integer() && (p==512 || p==1024 || p==2048 || p==4096),"wide input slot period");
  period_size=p.get<std::size_t>();decode_slots=period_size;
  require(input["decode_slots"].is_number_integer() && input["decode_slots"]==period_size,
    "wide output decode binding");
  require(input["input_ciphertexts"].is_number_integer() &&
    input["input_ciphertexts"]==loaded.plan.external_inputs.size() &&
    loaded.plan.external_inputs.size()>=2 && loaded.plan.external_inputs.size()<=17,"wide input count binding");
  require(input["output_ciphertexts"].is_number_integer() &&
    input["output_ciphertexts"]==loaded.plan.final_outputs.size() &&
    loaded.plan.final_outputs.size()>=1 && loaded.plan.final_outputs.size()<=16,"wide output count binding");
  samples=&input["samples"];
 }
 require(samples->is_array() && samples->size()>=1 && samples->size()<=16,"input sample count");
 for(const auto &sample:*samples) {
  require(sample.is_array() && sample.size()==loaded.plan.external_inputs.size(),"input count");
  for(const auto &vector:sample) {
   require(vector.is_array() && !vector.empty() && 16384%vector.size()==0 &&
      (wide?vector.size()==period_size:vector.size()<=256),"input slot period");
   for(const auto &x:vector)require(x.is_number() && std::isfinite(x.get<double>()),"finite input");
  }
 }
 auto ctx=context();
 KeyGenerator kg(ctx);
 PublicKey pk;kg.create_public_key(pk);
 auto rk=std::make_shared<RelinKeys>();auto gk=std::make_shared<GaloisKeys>();
 std::set<int> steps;bool relin=false;
 for(auto &key:req.keys){
  require(key.kind!=fhegpu::KeyKind::Secret,"executor must not request secret key");
  if(key.kind==fhegpu::KeyKind::Relin)relin=true;
  if(key.kind==fhegpu::KeyKind::Galois){require(key.rotation_step.has_value(),"rotation requirement");steps.insert(*key.rotation_step);}
 }
 if(relin)kg.create_relin_keys(*rk);
 auto basis=binary_rotation_key_basis(steps,ctx.parameters_literal()->slot());
 if(!basis.empty())kg.create_galois_keys(std::vector<int>(basis.begin(),basis.end()),*gk);
 CKKSEncoder encoder(ctx);Encryptor enc(ctx,pk);Decryptor dec(ctx,kg.secret_key());
 // Only evaluation keys enter the executor. Secret key remains with the acceptance owner.
 PoseidonCpuApi api(spec.spec.context_id,ctx,rk,gk);
 Json outputs=Json::array();
 for(const auto&sample:*samples) {
  require(sample.size()==loaded.plan.external_inputs.size(),"input count");
  std::unordered_map<fhegpu::ValueId,PoseidonCpuValue> inputs;
  for(std::size_t i=0;i<sample.size();++i) {
   auto id=loaded.plan.external_inputs[i];
   const fhegpu::ValueDesc*desc=nullptr;for(auto &v:loaded.plan.values)if(v.id==id)desc=&v;
   require(desc!=nullptr && desc->level>=0 && desc->level<=12,"input metadata");
   auto period=sample[i].get<std::vector<double>>();
   std::vector<double> slots(16384);
   for(std::size_t j=0;j<slots.size();++j){slots[j]=period[j%period.size()];require(std::isfinite(slots[j]),"finite input");}
   Plaintext plain;encoder.encode(slots,ctx.crt_context()->parms_id_map().at(desc->level),std::ldexp(1.,desc->scale_log2),plain);
   Ciphertext cipher;enc.encrypt(plain,cipher);
   inputs.emplace(id,PoseidonCpuValue::from_ciphertext(std::move(cipher)));
  }
  fhegpu::SequentialRuntime<PoseidonCpuApi> runtime(0,1,0,api);
  fhegpu::RuntimeResources resources{spec,std::filesystem::path(argv[3]),false};
  auto result=runtime.run(loaded,resources,inputs);
  Json sample_out=Json::array();
  for(auto id:loaded.plan.final_outputs){
   Plaintext plain;dec.decrypt(result.values.at(id).value.ciphertext(),plain);
   std::vector<std::complex<double>> decoded;encoder.decode(plain,decoded);
   std::vector<double> values;for(std::size_t i=0;i<decode_slots;++i){require(std::isfinite(decoded[i].real()),"finite output");values.push_back(decoded[i].real());}
   sample_out.push_back(values);
  }
  outputs.push_back(sample_out);
 }
 Json report={{"backend","poseidon_cpu"},{"encrypted_execution",true},{"bootstrap",false},{"security","tc128"},{"degree",32768},{"data_moduli",13},{"special_moduli",1},{"outputs",outputs}};
 report["q"]=Json::array();for(const auto&m:ctx.parameters_literal()->q())report["q"].push_back(m.value());
 report["p"]=Json::array();for(const auto&m:ctx.parameters_literal()->p())report["p"].push_back(m.value());
 report["executor_secret_key"]=false;
 if(wide) {
  report["format"]="poseidon-agent-execution-v2";
  report["execution_abi"]=input["execution_abi"];
  report["input_slot_period"]=period_size;report["decoded_slots"]=decode_slots;
  report["input_ciphertexts"]=loaded.plan.external_inputs.size();
  report["output_ciphertexts"]=loaded.plan.final_outputs.size();
 }
 std::ofstream(argv[5])<<report.dump(2)<<'\n';
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
