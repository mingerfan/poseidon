#pragma once
// Closed input declarations for the NEW experimental manual entry. A trusted
// CPU preflight must hash these payloads before launch; this reader validates
// pins/roles/paths, NOT a second cryptographic hash or a decryption oracle.
#include "source_manual_plan.hpp"
#include <charconv>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace cerium_source::manual {
enum class InputVersion {V1,V2};
struct Inputs {
    Plan plan;
    std::array<std::string,5> payloads; // fixture,weights,index,embeddings,reference
    std::size_t layers=0,valid_tokens=0;
    std::string manifest,report;
    bool cpu_only=false;
    InputVersion version=InputVersion::V1;
};
inline const char *input_version_name(InputVersion version) {
    if(version==InputVersion::V1)return "MANUAL_LLAMA_INPUTS_V1";
    if(version==InputVersion::V2)return "MANUAL_LLAMA_INPUTS_V2";
    throw std::invalid_argument("unsupported manual input version");
}
inline const char *model_weight_scope(InputVersion version) {
    if(version==InputVersion::V1)return "layer0_only_file_and_index";
    if(version==InputVersion::V2)return "full_32_layer_HF_four_shard_directory_and_index";
    throw std::invalid_argument("unsupported manual input version");
}
inline void validate_input_scope(const Inputs &in) {
    (void)input_version_name(in.version);
    if(in.version==InputVersion::V1&&in.layers!=1)
        throw std::invalid_argument("manual V1 weights/index support layer1 only");
    if((in.layers!=1&&in.layers!=32)||in.valid_tokens!=38)
        throw std::invalid_argument("manual pinned references support layers1/32 and active38 only");
    in.plan.validate();
}
inline constexpr std::array<const char*,4> hf_shard_names{
    "model-00001-of-00004.safetensors","model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors","model-00004-of-00004.safetensors"};
// Public metadata only: WEIGHT_DOWNLOAD_LOCK.json, LLM-Research/Meta-Llama-3.1-8B.
inline constexpr std::array<const char*,4> hf_shard_pins{
    "f8b9704ab09cdeb097aa4a0a24bca96f906eec36bad63ab495bc21475058601b",
    "c28b25e7541751056ee126627e007f8d4288319733285e9f7b17b9ff6eb313f0",
    "d8e9504dd4e4a146d484c52a97584ec14dac92237c46b064934af67a85e7d383",
    "e4486f35c040f683f7d790354f66c169c109eb9fa0954a4a35d7c458a108405d"};
inline std::size_t decimal(std::string_view text) {
    std::size_t value=0;auto r=std::from_chars(text.data(),text.data()+text.size(),value);
    if(text.empty()||r.ec!=std::errc{}||r.ptr!=text.data()+text.size())throw std::invalid_argument("closed decimal required");
    return value;
}
inline Inputs parse_inputs(int argc,const char *const *argv) {
    if(argc!=12)throw std::invalid_argument("manual needs mode plan fixture weights index embeddings reference layers valid_tokens manifest report");
    const std::string_view mode(argv[1]);
    if(mode!="--run"&&mode!="--test-inputs-cpu"&&mode!="--run-v2"&&mode!="--test-inputs-cpu-v2")
        throw std::invalid_argument("manual mode must be --run|--test-inputs-cpu|--run-v2|--test-inputs-cpu-v2");
    const auto version=(mode=="--run-v2"||mode=="--test-inputs-cpu-v2")?InputVersion::V2:InputVersion::V1;
    Inputs in{Plan{parse_mode(argv[2])},{argv[3],argv[4],argv[5],argv[6],argv[7]},decimal(argv[8]),decimal(argv[9]),argv[10],argv[11],
        mode=="--test-inputs-cpu"||mode=="--test-inputs-cpu-v2",version};
    validate_input_scope(in);return in;
}
inline std::filesystem::path canonical_file(const std::string &name) {
    namespace fs=std::filesystem;const fs::path p(name);
    if(!p.is_absolute()||name.find_first_of(" \t\r\n")!=std::string::npos||
       fs::is_symlink(p)||!fs::is_regular_file(p)||fs::canonical(p)!=p)
        throw std::runtime_error("canonical nonsymlink input file required");
    return p;
}
inline std::filesystem::path canonical_directory(const std::string &name) {
    namespace fs=std::filesystem;const fs::path p(name);
    if(!p.is_absolute()||name.find_first_of(" \t\r\n")!=std::string::npos||
       fs::is_symlink(p)||!fs::is_directory(p)||fs::canonical(p)!=p)
        throw std::runtime_error("canonical nonsymlink input directory required");
    return p;
}
inline std::filesystem::path model_weights_root(const Inputs &in) {
    validate_input_scope(in);
    // SafetensorWeights always takes a DIRECTORY. V1 pins one regular file;
    // its fixed index resolves layer.weights.bin relative to this parent.
    return in.version==InputVersion::V1?canonical_file(in.payloads[1]).parent_path():canonical_directory(in.payloads[1]);
}
inline std::string report_path(const std::string &name) {
    namespace fs=std::filesystem;const fs::path p(name);
    if(!p.is_absolute()||p.filename().empty()||p.filename()=="."||p.filename()==".."||
       name.find_first_of(" \t\r\n")!=std::string::npos||fs::canonical(p.parent_path())!=p.parent_path()||
       fs::exists(p)||fs::is_symlink(p))throw std::runtime_error("canonical exclusive manual report required");
    return name;
}
inline void validate_manifest(const Inputs &in) {
    validate_input_scope(in);
    // Missing manifest is rejected before touching any model payload path.
    if(!std::filesystem::exists(in.manifest))throw std::runtime_error("manual CPU hash manifest required");
    (void)canonical_file(in.manifest); // reject leaf/parent symlinks BEFORE opening
    std::ifstream file(in.manifest);std::string line;
    if(!std::getline(file,line)||line!=input_version_name(in.version))
        throw std::runtime_error("manual input version/manifest magic mismatch");
    std::map<std::string,std::pair<std::string,std::string>> rows;
    while(std::getline(file,line)) {
        std::istringstream s(line);std::string role,path,hash,extra;
        if(!(s>>role>>path>>hash)||(s>>extra)||!rows.emplace(role,std::make_pair(path,hash)).second)
            throw std::runtime_error("malformed/duplicate manual manifest role");
    }
    const std::array<const char*,5> names{"fixture","weights","index","embeddings","reference"};
    const std::array<std::string,5> hashes{
        "d9a79047125a6506c6b2d4c335737ccdb52a216f398a8150b133d47a44f72f56",
        in.version==InputVersion::V1?"3a4780271abf7780094e272be10abe3dcc089f3cbde6301d0cc33329a70e0e7d":"-",
        in.version==InputVersion::V1?"36b6f66f819aa0e34683f9adce037b599d2c4572ae97a3ea670b7600a4d49b32":"fd89f8e33fac43f92e45d26477628bfa263de5b8e8159874ed22c71c8fd087dd",
        "5e1a1e39b2021a66119bdb7c5b2dd8a65d4e6fd701c2ce16f52000f27f00839f",
        in.layers==1?"7cbc25944cfeddc720415e9ceb6b61001f92cdf20fd9e93a389dcaabdf11cc25":"1d4a5ec9b4686b7d12a83b29e207dd84a7f9463a80e9a9628267eeacf25c3cc8"};
    if(rows.size()!=(in.version==InputVersion::V1?6U:10U))
        throw std::runtime_error("manual manifest needs closed version-specific roles");
    // Verify all declarations before filesystem reads of referenced payloads.
    for(std::size_t i=0;i<names.size();++i) {
        const auto it=rows.find(names[i]);
        if(it==rows.end()||it->second.first!=in.payloads[i]||it->second.second!=hashes[i])
            throw std::runtime_error("manual input pin/path declaration mismatch");
    }
    auto out=rows.find("report");
    if(out==rows.end()||out->second.first!=in.report||out->second.second!="-")throw std::runtime_error("manual report role mismatch");
    if(in.version==InputVersion::V2)for(std::size_t i=0;i<hf_shard_names.size();++i) {
        const auto it=rows.find(hf_shard_names[i]);
        const auto path=(std::filesystem::path(in.payloads[1])/hf_shard_names[i]).string();
        if(it==rows.end()||it->second.first!=path||it->second.second!=hf_shard_pins[i])
            throw std::runtime_error("manual HF shard pin/path declaration mismatch");
    }
    for(std::size_t i=0;i<in.payloads.size();++i) {
        if(i==1&&in.version==InputVersion::V2)(void)canonical_directory(in.payloads[i]);
        else (void)canonical_file(in.payloads[i]);
    }
    if(in.version==InputVersion::V2)for(const auto *name:hf_shard_names)
        (void)canonical_file((std::filesystem::path(in.payloads[1])/name).string());
    (void)report_path(in.report);
}
} // namespace cerium_source::manual
