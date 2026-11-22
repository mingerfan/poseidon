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
struct Inputs {
    Plan plan;
    std::array<std::string,5> payloads; // fixture,weights,index,embeddings,reference
    std::size_t layers=0,valid_tokens=0;
    std::string manifest,report;
    bool cpu_only=false;
};
inline std::size_t decimal(std::string_view text) {
    std::size_t value=0;auto r=std::from_chars(text.data(),text.data()+text.size(),value);
    if(text.empty()||r.ec!=std::errc{}||r.ptr!=text.data()+text.size())throw std::invalid_argument("closed decimal required");
    return value;
}
inline Inputs parse_inputs(int argc,const char *const *argv) {
    if(argc!=12||(std::string_view(argv[1])!="--run"&&std::string_view(argv[1])!="--test-inputs-cpu"))
        throw std::invalid_argument("manual --run|--test-inputs-cpu plan fixture weights index embeddings reference layers valid_tokens manifest report");
    Inputs in{Plan{parse_mode(argv[2])},{argv[3],argv[4],argv[5],argv[6],argv[7]},decimal(argv[8]),decimal(argv[9]),argv[10],argv[11],std::string_view(argv[1])=="--test-inputs-cpu"};
    if((in.layers!=1&&in.layers!=32)||in.valid_tokens!=38)
        throw std::invalid_argument("manual pinned references support layers1/32 and active38 only");
    in.plan.validate();return in;
}
inline std::filesystem::path canonical_file(const std::string &name) {
    namespace fs=std::filesystem;const fs::path p(name);
    if(!p.is_absolute()||name.find_first_of(" \t\r\n")!=std::string::npos||
       fs::is_symlink(p)||!fs::is_regular_file(p)||fs::canonical(p)!=p)
        throw std::runtime_error("canonical nonsymlink input file required");
    return p;
}
inline std::string report_path(const std::string &name) {
    namespace fs=std::filesystem;const fs::path p(name);
    if(!p.is_absolute()||p.filename().empty()||p.filename()=="."||p.filename()==".."||
       name.find_first_of(" \t\r\n")!=std::string::npos||fs::canonical(p.parent_path())!=p.parent_path()||
       fs::exists(p)||fs::is_symlink(p))throw std::runtime_error("canonical exclusive manual report required");
    return name;
}
inline void validate_manifest(const Inputs &in) {
    // Missing manifest is rejected before touching any model payload path.
    if(!std::filesystem::exists(in.manifest))throw std::runtime_error("manual CPU hash manifest required");
    (void)canonical_file(in.manifest); // reject leaf/parent symlinks BEFORE opening
    std::ifstream file(in.manifest);std::string line;
    if(!std::getline(file,line)||line!="MANUAL_LLAMA_INPUTS_V1")throw std::runtime_error("manual CPU hash manifest required");
    std::map<std::string,std::pair<std::string,std::string>> rows;
    while(std::getline(file,line)) {
        std::istringstream s(line);std::string role,path,hash,extra;
        if(!(s>>role>>path>>hash)||(s>>extra)||!rows.emplace(role,std::make_pair(path,hash)).second)
            throw std::runtime_error("malformed/duplicate manual manifest role");
    }
    const std::array<const char*,5> names{"fixture","weights","index","embeddings","reference"};
    const std::array<std::string,5> hashes{
        "d9a79047125a6506c6b2d4c335737ccdb52a216f398a8150b133d47a44f72f56",
        "3a4780271abf7780094e272be10abe3dcc089f3cbde6301d0cc33329a70e0e7d",
        "36b6f66f819aa0e34683f9adce037b599d2c4572ae97a3ea670b7600a4d49b32",
        "5e1a1e39b2021a66119bdb7c5b2dd8a65d4e6fd701c2ce16f52000f27f00839f",
        in.layers==1?"7cbc25944cfeddc720415e9ceb6b61001f92cdf20fd9e93a389dcaabdf11cc25":"1d4a5ec9b4686b7d12a83b29e207dd84a7f9463a80e9a9628267eeacf25c3cc8"};
    if(rows.size()!=6)throw std::runtime_error("manual manifest needs five input roles and report");
    // Verify all declarations before filesystem reads of referenced payloads.
    for(std::size_t i=0;i<names.size();++i) {
        const auto it=rows.find(names[i]);
        if(it==rows.end()||it->second.first!=in.payloads[i]||it->second.second!=hashes[i])
            throw std::runtime_error("manual input pin/path declaration mismatch");
    }
    auto out=rows.find("report");
    if(out==rows.end()||out->second.first!=in.report||out->second.second!="-")throw std::runtime_error("manual report role mismatch");
    for(const auto &name:in.payloads)(void)canonical_file(name);
    (void)report_path(in.report);
}
} // namespace cerium_source::manual
