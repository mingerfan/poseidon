// std-only schema/routing declarations. Synthetic manifest files are NOT
// cryptographic payload binding, Context/key identity or encrypted accuracy.
#include "source_manual_io.hpp"
#include <iostream>
#include <limits>
#include <type_traits>
#include <unistd.h>
using namespace cerium_source::manual;
static_assert(!std::is_default_constructible_v<Plan>,"manual plan must be explicit");
namespace {
unsigned checks=0,negative=0;
void check(bool value){++checks;if(!value)throw std::runtime_error("manual plan test failed");}
template<class F>void rejects(F f){++negative;try{f();}catch(const std::exception&){return;}throw std::runtime_error("missing manual plan rejection");}
Observation expected(Plan p,std::size_t i) {
    const auto &b=boundaries[i];return {b.name,b.tiles,b.q,std::exp2(40.),b.primary_boots,
        i>=12?p.secondary_boots():0,i>=12?p.projection_pairs():0,i>=5?16U:0U,0};
}
void metadata_tests() {
    check(parse_mode("manual-highprecision-real-v1")==Mode::HighPrecisionReal);
    check(parse_mode("manual-highprecision-paired-silu-candidate-v1")==Mode::HighPrecisionPairedSiluCandidate);
    for(const auto *bad:{"","native","legacy","manual-highprecision-real-v1x"})rejects([&]{(void)parse_mode(bad);});
    rejects([]{Plan{static_cast<Mode>(99)}.validate();});
    for(auto mode:{Mode::HighPrecisionReal,Mode::HighPrecisionPairedSiluCandidate}) {
        Plan plan{mode};check(plan.total_boots()==(plan.paired_silu()?364U:476U));check(plan.projection_pairs()==(plan.paired_silu()?112U:0U));
        BoundaryTrace good(plan);for(std::size_t i=0;i<15;++i)good.observe(expected(plan,i));good.finish();check(good.observed_boundaries()==15);
        rejects([&]{good.observe(expected(plan,14));});rejects([&]{BoundaryTrace(plan).finish();});
        for(std::size_t point=0;point<15;++point)for(unsigned mutation=0;mutation<10;++mutation) {
            BoundaryTrace bad(plan);for(std::size_t i=0;i<point;++i)bad.observe(expected(plan,i));auto o=expected(plan,point);
            switch(mutation){case 0:o.name="wrong";break;case 1:++o.tiles;break;case 2:++o.q;break;
              case 3:o.scale*=2;break;case 4:++o.primary_boots;break;case 5:++o.secondary_boots;break;
              case 6:++o.projection_pairs;break;case 7:++o.integer_exp_seeds;break;case 8:++o.standard_exp_seeds;break;
              default:o.scale=std::numeric_limits<double>::quiet_NaN();}
            rejects([&]{bad.observe(o);});
        }
    }
    for(const auto *bad:{"","-1","+1","1x"," 1","184467440737095516160"})rejects([&]{(void)decimal(bad);});
    check(decimal("32")==32);
}
void input_tests() {
    const char *args[]{"manual","--test-inputs-cpu","manual-highprecision-real-v1","/not-fixture","/not-weights","/not-index","/not-embeddings","/not-reference","1","38","/not-manifest","/not-report"};
    auto missing=parse_inputs(12,args);rejects([&]{validate_manifest(missing);});
    for(auto index:{1U,2U,8U,9U}) {auto copy=std::array<const char*,12>{};std::copy(std::begin(args),std::end(args),copy.begin());copy[index]="bad";rejects([&]{(void)parse_inputs(12,copy.data());});}
    rejects([&]{(void)parse_inputs(11,args);});
    char owned[]="/mnt/zhitai/KnowledgeLib/Research/FHE-GPU/workspaces/llama-framework-node2/bench/llama_manual/tests/input-v2/g01-before/synthetic-XXXXXX";const auto made=::mkdtemp(owned);if(!made)throw std::runtime_error("owned test directory failed");
    const std::filesystem::path root(made);
    struct Cleanup {std::filesystem::path p;~Cleanup(){std::filesystem::remove_all(p);}}cleanup{root};
    Inputs in{Plan{Mode::HighPrecisionReal},{},1,38,(root/"manifest.tsv").string(),(root/"report.json").string(),true};
    const std::array<const char*,5> names{"fixture","weights","index","embeddings","reference"};
    const std::array<const char*,5> pins{
        "d9a79047125a6506c6b2d4c335737ccdb52a216f398a8150b133d47a44f72f56",
        "3a4780271abf7780094e272be10abe3dcc089f3cbde6301d0cc33329a70e0e7d",
        "36b6f66f819aa0e34683f9adce037b599d2c4572ae97a3ea670b7600a4d49b32",
        "5e1a1e39b2021a66119bdb7c5b2dd8a65d4e6fd701c2ce16f52000f27f00839f",
        "7cbc25944cfeddc720415e9ceb6b61001f92cdf20fd9e93a389dcaabdf11cc25"};
    for(unsigned i=0;i<5;++i){in.payloads[i]=(root/names[i]).string();std::ofstream(in.payloads[i])<<"synthetic metadata policy input, NOT actual pinned payload";}
    auto content=[&](){std::string out="MANUAL_LLAMA_INPUTS_V1\n";for(unsigned i=0;i<5;++i)out+=std::string(names[i])+" "+in.payloads[i]+" "+pins[i]+"\n";return out+"report "+in.report+" -\n";};
    const auto original=content();auto write=[&](const std::string &s){std::ofstream(in.manifest)<<s;};
    write(original);validate_manifest(in);check(!std::filesystem::exists(in.report));
    write(original+"fixture "+in.payloads[0]+" "+pins[0]+"\n");rejects([&]{validate_manifest(in);});
    write(original.substr(0,original.find("report ")));rejects([&]{validate_manifest(in);});
    auto wrong=original;const auto pos=wrong.find(pins[0]);wrong[pos]='0';write(wrong);rejects([&]{validate_manifest(in);});write(original);
    std::filesystem::create_symlink(root/"missing-target",in.report);rejects([&]{validate_manifest(in);});check(!std::filesystem::exists(root/"missing-target"));std::filesystem::remove(in.report);
    const auto real=in.payloads[0];in.payloads[0]=(root/"alias").string();std::filesystem::create_symlink(real,in.payloads[0]);write(content());rejects([&]{validate_manifest(in);});in.payloads[0]=real;write(original);
    const auto mf=in.manifest;in.manifest=(root/"manifest-link").string();std::filesystem::create_symlink(mf,in.manifest);rejects([&]{validate_manifest(in);});in.manifest=mf;
    std::ofstream(in.report)<<"do not overwrite";rejects([&]{validate_manifest(in);});
}
}
int main()try {
    metadata_tests();input_tests();std::cout<<"{\"status\":\"passed\",\"checks\":"<<checks<<",\"negative_cases\":"<<negative
      <<",\"N\":65536,\"plans\":2,\"boundaries_per_plan\":15,\"source_topology_not_executed_Boots\":true,\"synthetic_manifest_declarations_not_payload_hash_proof\":true,\"no_Context_Runtime_keys_GPU\":true}\n";return 0;
}catch(const std::exception &e){std::cerr<<"MANUAL_PLAN_TEST_FAILED "<<e.what()<<std::endl;return 1;}
