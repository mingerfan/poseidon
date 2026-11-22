// std-only schema/routing declarations. Synthetic manifest files are NOT
// cryptographic payload binding, Context/key identity or encrypted accuracy.
#include "source_manual_io.hpp"
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>
#include <unistd.h>
using namespace cerium_source::manual;
static_assert(!std::is_default_constructible_v<Plan>,"manual plan must be explicit");
namespace {
unsigned checks=0,negative=0;
void check(bool value,const std::string &reason="manual plan test failed"){++checks;if(!value)throw std::runtime_error(reason);}
template<class F>void rejects(F f,const std::string &reason="missing manual plan rejection") {
    ++negative;try{f();}catch(const std::exception&){return;}throw std::runtime_error(reason);
}
template<class F>void rejects_with(F f,const std::string &message) {
    ++negative;try{f();}catch(const std::exception &e){
        if(e.what()==message)return;
        throw std::runtime_error("expected rejection '"+message+"', got '"+e.what()+"'");
    }
    throw std::runtime_error("missing rejection: "+message);
}
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
namespace fs=std::filesystem;
const std::array<const char*,5> roles{"fixture","weights","index","embeddings","reference"};
// Independent expected declarations, not values copied at runtime from the reader.
const std::array<const char*,5> v1_pins{
    "d9a79047125a6506c6b2d4c335737ccdb52a216f398a8150b133d47a44f72f56",
    "3a4780271abf7780094e272be10abe3dcc089f3cbde6301d0cc33329a70e0e7d",
    "36b6f66f819aa0e34683f9adce037b599d2c4572ae97a3ea670b7600a4d49b32",
    "5e1a1e39b2021a66119bdb7c5b2dd8a65d4e6fd701c2ce16f52000f27f00839f",
    "7cbc25944cfeddc720415e9ceb6b61001f92cdf20fd9e93a389dcaabdf11cc25"};
const std::array<const char*,4> shard_names{
    "model-00001-of-00004.safetensors","model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors","model-00004-of-00004.safetensors"};
const std::array<const char*,4> shard_pins{
    "f8b9704ab09cdeb097aa4a0a24bca96f906eec36bad63ab495bc21475058601b",
    "c28b25e7541751056ee126627e007f8d4288319733285e9f7b17b9ff6eb313f0",
    "d8e9504dd4e4a146d484c52a97584ec14dac92237c46b064934af67a85e7d383",
    "e4486f35c040f683f7d790354f66c169c109eb9fa0954a4a35d7c458a108405d"};
struct Row {std::string role,path,pin;};
const char *magic(InputVersion version){return version==InputVersion::V1?"MANUAL_LLAMA_INPUTS_V1":"MANUAL_LLAMA_INPUTS_V2";}
std::vector<Row> declarations(const Inputs &in) {
    std::vector<Row> rows;
    for(std::size_t i=0;i<roles.size();++i)rows.push_back({roles[i],in.payloads[i],v1_pins[i]});
    if(in.layers==32)rows[4].pin="1d4a5ec9b4686b7d12a83b29e207dd84a7f9463a80e9a9628267eeacf25c3cc8";
    rows.push_back({"report",in.report,"-"});
    if(in.version==InputVersion::V2) {
        rows[1].pin="-";rows[2].pin="fd89f8e33fac43f92e45d26477628bfa263de5b8e8159874ed22c71c8fd087dd";
        for(std::size_t i=0;i<shard_names.size();++i)
            rows.push_back({shard_names[i],(fs::path(in.payloads[1])/shard_names[i]).string(),shard_pins[i]});
    }
    return rows;
}
void write_file(const fs::path &path,const std::string &text) {
    std::ofstream out(path);if(!(out<<text))throw std::runtime_error("synthetic test file write failed: "+path.string());
}
std::string content(const Inputs &in,const std::vector<Row> &rows) {
    std::string text=std::string(magic(in.version))+"\n";
    for(const auto &r:rows)text+=r.role+" "+r.path+" "+r.pin+"\n";
    return text;
}
void write_manifest(const Inputs &in){write_file(in.manifest,content(in,declarations(in)));}
Inputs synthetic_inputs(const fs::path &root,InputVersion version) {
    fs::create_directory(root);
    Inputs in{Plan{Mode::HighPrecisionReal},{},1,38,(root/"manifest.tsv").string(),(root/"report.json").string(),true,version};
    const std::array<const char*,5> filenames{"fixture.txt","layer.weights.bin","index.tsv","embeddings.bin","reference.bin"};
    for(std::size_t i=0;i<filenames.size();++i) {
        in.payloads[i]=(root/filenames[i]).string();
        if(i==1&&version==InputVersion::V2) {
            in.payloads[i]=(root/"weights").string();fs::create_directory(in.payloads[i]);
            for(const auto *name:shard_names)write_file(fs::path(in.payloads[i])/name,"synthetic shard declaration, NOT model data");
        } else write_file(in.payloads[i],"synthetic declaration, NOT actual pinned payload");
    }
    return in;
}
Inputs parse_cli(const Inputs &in,const std::string &mode,const std::string &layers="") {
    const std::array<std::string,12> args{"manual",mode,in.plan.name(),in.payloads[0],in.payloads[1],in.payloads[2],in.payloads[3],in.payloads[4],
        layers.empty()?std::to_string(in.layers):layers,std::to_string(in.valid_tokens),in.manifest,in.report};
    std::array<const char*,12> ptrs{};for(std::size_t i=0;i<ptrs.size();++i)ptrs[i]=args[i].c_str();
    return parse_inputs(int(ptrs.size()),ptrs.data());
}
void valid_declarations(const Inputs &in) {
    write_manifest(in);validate_manifest(in);
    check(!fs::exists(in.report)&&!fs::is_symlink(in.report),"declaration validation created a report");
    check(model_weights_root(in)==(in.version==InputVersion::V1?fs::path(in.payloads[1]).parent_path():fs::path(in.payloads[1])),
        "wrong SafetensorWeights directory routing");
}
void cli_tests(Inputs v1,Inputs v2) {
    rejects([]{(void)parse_inputs(0,nullptr);});
    const char *args[]{"manual","--test-inputs-cpu","manual-highprecision-real-v1","/not-fixture","/not-weights","/not-index","/not-embeddings","/not-reference","1","38","/not-manifest","/not-report"};
    auto missing=parse_inputs(12,args);rejects_with([&]{validate_manifest(missing);},"manual CPU hash manifest required");
    for(auto index:{1U,2U,8U,9U}) {auto copy=std::array<const char*,12>{};std::copy(std::begin(args),std::end(args),copy.begin());copy[index]="bad";rejects([&]{(void)parse_inputs(12,copy.data());});}
    for(int argc:{1,2,11,13})rejects([&]{(void)parse_inputs(argc,args);});
    for(const auto *bad:{"--run-v3","--test-inputs-cpu-v1","--RUN","--run-v2x"})rejects([&]{(void)parse_cli(v2,bad);});
    for(auto version:{InputVersion::V1,InputVersion::V2}) {
        auto in=version==InputVersion::V1?v1:v2;
        for(const auto *mode:version==InputVersion::V1?std::array<const char*,2>{"--run","--test-inputs-cpu"}:
            std::array<const char*,2>{"--run-v2","--test-inputs-cpu-v2"}) {
            for(auto plan:{Mode::HighPrecisionReal,Mode::HighPrecisionPairedSiluCandidate}) {
                in.plan=Plan{plan};auto parsed=parse_cli(in,mode);
                check(parsed.version==version&&parsed.cpu_only==(std::string_view(mode).find("--test-")==0)&&parsed.plan.mode==plan&&parsed.payloads==in.payloads);
                valid_declarations(parsed);
            }
            for(unsigned layer=0;layer<=33;++layer) {
                if(layer==1||(version==InputVersion::V2&&layer==32))continue;
                rejects([&]{(void)parse_cli(in,mode,std::to_string(layer));},"accepted unpinned layer scope");
            }
        }
        for(std::size_t tokens:{0U,37U,39U,128U}) {auto bad=in;bad.valid_tokens=tokens;rejects([&]{(void)parse_cli(bad,version==InputVersion::V1?"--run":"--run-v2");});}
    }
    v2.layers=32;
    for(const auto *mode:{"--run-v2","--test-inputs-cpu-v2"}) {auto parsed=parse_cli(v2,mode);check(parsed.layers==32&&parsed.version==InputVersion::V2);valid_declarations(parsed);}
    check(std::string(input_version_name(InputVersion::V1))=="MANUAL_LLAMA_INPUTS_V1");
    check(std::string(input_version_name(InputVersion::V2))=="MANUAL_LLAMA_INPUTS_V2");
    check(std::string(model_weight_scope(InputVersion::V1))=="layer0_only_file_and_index");
    check(std::string(model_weight_scope(InputVersion::V2))=="full_32_layer_HF_four_shard_directory_and_index");
    auto bad=v1;bad.layers=32;rejects([&]{validate_manifest(bad);});rejects([&]{(void)model_weights_root(bad);});
    bad=v2;bad.layers=2;rejects([&]{validate_manifest(bad);});
    bad=v2;bad.version=static_cast<InputVersion>(99);rejects([&]{validate_manifest(bad);});
    rejects([]{(void)model_weight_scope(static_cast<InputVersion>(99));});
}
void row_tests(Inputs in) {
    const auto good=declarations(in);const auto original=content(in,good);
    for(std::size_t i=0;i<good.size();++i) {
        auto rows=good;rows[i].pin=std::string(64,'0');write_file(in.manifest,content(in,rows));
        rejects([&]{validate_manifest(in);},"accepted wrong pin for "+good[i].role);
        rows=good;rows[i].path+=".wrong";write_file(in.manifest,content(in,rows));
        rejects([&]{validate_manifest(in);},"accepted wrong path for "+good[i].role);
        rows=good;rows.erase(rows.begin()+i);write_file(in.manifest,content(in,rows));
        rejects([&]{validate_manifest(in);},"accepted missing role "+good[i].role);
        rows=good;rows.push_back(rows[i]);write_file(in.manifest,content(in,rows));
        rejects([&]{validate_manifest(in);},"accepted duplicate role "+good[i].role);
        rows=good;rows[i].role="unknown";write_file(in.manifest,content(in,rows));
        rejects([&]{validate_manifest(in);},"accepted unknown role instead of "+good[i].role);
    }
    for(const auto *tail:{"\n","extra\n","fixture /bad\n","fixture /bad - unexpected\n","unknown /bad -\n"}) {
        write_file(in.manifest,original+tail);rejects([&]{validate_manifest(in);});
    }
    // Both CLI modes must reject the opposite version BEFORE any payload path
    // check: a missing fixture and existing report would otherwise fail later.
    for(const auto *mode:in.version==InputVersion::V1?std::array<const char*,2>{"--run","--test-inputs-cpu"}:
        std::array<const char*,2>{"--run-v2","--test-inputs-cpu-v2"}) {
        auto absent=in;absent.payloads[0]+=".missing";absent=parse_cli(absent,mode);
        const auto text=content(absent,declarations(absent));
        for(const std::string &badmagic:{std::string(""),std::string("MANUAL_LLAMA_INPUTS_V3"),
            std::string(in.version==InputVersion::V1?"MANUAL_LLAMA_INPUTS_V2":"MANUAL_LLAMA_INPUTS_V1")}) {
            write_file(in.report,"existing synthetic report");
            write_file(in.manifest,badmagic+text.substr(text.find('\n')));
            rejects_with([&]{validate_manifest(absent);},"manual input version/manifest magic mismatch");
            check(fs::file_size(in.report)==25,"version mismatch modified an existing report");fs::remove(in.report);
        }
    }
    if(in.version==InputVersion::V2) {
        auto wrong=good;wrong[2].pin=v1_pins[2];write_file(in.manifest,content(in,wrong));rejects([&]{validate_manifest(in);});
        wrong=good;wrong[4].pin=in.layers==1?"1d4a5ec9b4686b7d12a83b29e207dd84a7f9463a80e9a9628267eeacf25c3cc8":v1_pins[4];
        write_file(in.manifest,content(in,wrong));rejects([&]{validate_manifest(in);});
        // Equal-count substitution must not turn V2 into a V1 role set.
        wrong=good;wrong[6].role="shard1";write_file(in.manifest,content(in,wrong));rejects([&]{validate_manifest(in);});
    }
    valid_declarations(in);
}
void path_tests(Inputs in,const fs::path &root) {
    const auto original=in;
    for(std::size_t i=0;i<in.payloads.size();++i) {
        const auto real=fs::path(in.payloads[i]);const auto alias=root/"payload-link";
        fs::create_symlink(real,alias);in.payloads[i]=alias.string();write_manifest(in);rejects([&]{validate_manifest(in);},"accepted payload leaf symlink");
        fs::remove(alias);in=original;
        fs::create_directory_symlink(real.parent_path(),alias);in.payloads[i]=(alias/real.filename()).string();write_manifest(in);
        rejects([&]{validate_manifest(in);},"accepted payload parent symlink");fs::remove(alias);in=original;
        in.payloads[i]=(real.parent_path()/"."/real.filename()).string();write_manifest(in);rejects([&]{validate_manifest(in);},"accepted noncanonical payload");in=original;
        in.payloads[i]="relative-input";write_manifest(in);rejects([&]{validate_manifest(in);},"accepted relative payload");in=original;
        in.payloads[i]=(root/"space input").string();write_manifest(in);rejects([&]{validate_manifest(in);},"accepted whitespace payload");in=original;
        if(i!=1||in.version==InputVersion::V1) {
            in.payloads[i]=root.string();write_manifest(in);
            rejects_with([&]{validate_manifest(in);},"canonical nonsymlink input file required");in=original;
        }
    }
    if(in.version==InputVersion::V1) {
        in.payloads[1]=root.string();write_manifest(in);rejects_with([&]{(void)model_weights_root(in);},"canonical nonsymlink input file required");in=original;
    } else {
        in.payloads[1]=in.payloads[0];write_manifest(in);
        rejects_with([&]{validate_manifest(in);},"canonical nonsymlink input directory required");
        rejects_with([&]{(void)model_weights_root(in);},"canonical nonsymlink input directory required");in=original;
        write_manifest(in);
        for(const auto *name:shard_names) {
            const auto path=fs::path(in.payloads[1])/name;const auto saved=fs::path(path.string()+".saved");
            fs::rename(path,saved);fs::create_symlink(saved,path);
            rejects_with([&]{validate_manifest(in);},"canonical nonsymlink input file required");fs::remove(path);
            fs::create_directory(path);rejects_with([&]{validate_manifest(in);},"canonical nonsymlink input file required");fs::remove(path);
            rejects_with([&]{validate_manifest(in);},"canonical nonsymlink input file required");fs::rename(saved,path);
        }
    }
    write_manifest(in);
    const auto alias=root/"manifest-link";fs::create_symlink(in.manifest,alias);in.manifest=alias.string();
    rejects([&]{validate_manifest(in);},"accepted manifest leaf symlink");fs::remove(alias);in=original;
    fs::create_directory_symlink(fs::path(in.manifest).parent_path(),alias);in.manifest=(alias/fs::path(in.manifest).filename()).string();
    rejects([&]{validate_manifest(in);},"accepted manifest parent symlink");fs::remove(alias);in=original;
    in.manifest=(fs::path(in.manifest).parent_path()/"."/fs::path(in.manifest).filename()).string();
    rejects([&]{validate_manifest(in);},"accepted noncanonical manifest");in=original;
    const auto missing=root/"missing-report-target";
    fs::create_symlink(missing,in.report);rejects([&]{validate_manifest(in);},"accepted dangling report symlink");check(!fs::exists(missing));fs::remove(in.report);
    fs::create_symlink(in.payloads[0],in.report);rejects([&]{validate_manifest(in);},"accepted live report symlink");fs::remove(in.report);
    const auto parent=root/"report-parent-link";fs::create_directory_symlink(fs::path(in.report).parent_path(),parent);
    in.report=(parent/"new-report.json").string();write_manifest(in);rejects([&]{validate_manifest(in);},"accepted report symlink parent");
    check(!fs::exists(in.report));fs::remove(parent);in=original;
    for(const auto &bad:{std::string("relative-report"),(root/"space report").string(),(root/"."/"noncanonical-report").string(),root.string()+"/"}) {
        in.report=bad;write_manifest(in);rejects([&]{validate_manifest(in);},"accepted invalid report path");in=original;
    }
    write_manifest(in);write_file(in.report,"do not overwrite");rejects([&]{validate_manifest(in);},"accepted existing report");
    check(fs::file_size(in.report)==16,"existing report was overwritten");fs::remove(in.report);
    fs::create_directory(in.report);rejects([&]{validate_manifest(in);},"accepted existing report directory");fs::remove(in.report);
    valid_declarations(in);
}
void input_tests() {
    // TMPDIR is set to the exclusive evidence generation by the CPU runner.
    std::string owned=(fs::temp_directory_path()/"manual-llama-inputs-XXXXXX").string();
    const auto made=::mkdtemp(owned.data());if(!made)throw std::runtime_error("owned test directory failed");
    const fs::path root(made);
    struct Cleanup {fs::path p;~Cleanup(){fs::remove_all(p);}}cleanup{root};
    auto v1=synthetic_inputs(root/"v1",InputVersion::V1);auto v2=synthetic_inputs(root/"v2",InputVersion::V2);
    cli_tests(v1,v2);row_tests(v1);row_tests(v2);auto full=v2;full.layers=32;row_tests(full);
    path_tests(v1,root);path_tests(v2,root);
}
}
int main()try {
    metadata_tests();input_tests();std::cout<<"{\"status\":\"passed\",\"checks\":"<<checks<<",\"negative_cases\":"<<negative
      <<",\"N\":65536,\"plans\":2,\"boundaries_per_plan\":15,\"input_versions\":2,\"V1_layers\":[1],\"V2_layers\":[1,32]"
      <<",\"source_topology_not_executed_Boots\":true,\"synthetic_manifest_declarations_not_payload_hash_proof\":true,\"no_Context_Runtime_keys_GPU\":true}\n";return 0;
}catch(const std::exception &e){std::cerr<<"MANUAL_PLAN_TEST_FAILED "<<e.what()<<std::endl;return 1;}
