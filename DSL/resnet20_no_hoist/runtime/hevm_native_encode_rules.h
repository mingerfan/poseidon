#pragma once
#include "hevm_native_scale_sidecar.h"
#include <unordered_map>

namespace poseidon::mgpu {
enum class NativeEncodeRuleKind { Constant,Source,Ratio,Exp2OverSource };
struct HevmNativeEncodeRule {
    std::uint64_t index,destination,first,second;
    NativeEncodeRuleKind kind;
    double constant,exponent;
};
inline std::vector<HevmNativeEncodeRule> parse_hevm_native_encode_rules(
    std::string_view binary,std::string_view scales,std::string_view metadata) {
    auto require=[](bool ok,const char *message){if(!ok)throw std::invalid_argument(message);};
    const auto encoded=parse_hevm_native_scales(binary,scales);
    std::unordered_map<std::uint64_t,HevmNativeScale> scale_map;
    for(auto row:encoded)scale_map.emplace(row.operation_index,row);
    auto read=[&](std::size_t at,unsigned bytes=8){require(at<=binary.size() && bytes<=binary.size()-at,"truncated Encode rule HEVM");
        std::uint64_t result=0;for(unsigned i=0;i<bytes;++i)result|=std::uint64_t(static_cast<unsigned char>(binary[at+i]))<<(8*i);return result;};
    auto header=read(4,4),body=read(header),nops=read(header+8),start=header+body;
    std::uint64_t hash=14695981039346656037ULL;
    for(unsigned char byte:binary){hash^=byte;hash*=1099511628211ULL;}
    std::vector<std::uint64_t> defined_at(read(header+16),nops+1);
    for(std::uint64_t i=0;i<read(8);++i)defined_at.at(i)=0;
    for(std::uint64_t i=0;i<nops;++i){auto at=start+8*i,opcode=read(at,2);
        if(opcode!=65535 && opcode!=0)defined_at.at(read(at+2,2))=i+1;}
    std::istringstream input{std::string(metadata)};std::string line,magic,extra;
    require(bool(std::getline(input,line)),"empty Encode rules");
    std::istringstream first_line(line);std::uint64_t count=0,binding=0;
    require(bool(first_line>>magic>>count>>std::hex>>binding) && !(first_line>>extra),"invalid Encode rule header");
    require(magic=="POSEIDON_HEVM_NATIVE_ENCODE_RULE_V1" && count==nops && binding==hash,"Encode rules binding mismatch");
    std::vector<HevmNativeEncodeRule> result;std::unordered_set<std::uint64_t> seen;
    while(std::getline(input,line)) {
        std::istringstream row(line);HevmNativeEncodeRule rule{};std::string kind,target;
        require(bool(row>>rule.index>>rule.destination>>kind>>rule.first>>rule.second>>target) && !(row>>extra),"invalid Encode rule row");
        require(seen.insert(rule.index).second && scale_map.count(rule.index),"duplicate/non-Encode rule");
        require(read(start+8*rule.index+2,2)==rule.destination,"Encode rule destination mismatch");
        if(kind=="constant")rule.kind=NativeEncodeRuleKind::Constant;
        else if(kind=="source")rule.kind=NativeEncodeRuleKind::Source;
        else if(kind=="ratio")rule.kind=NativeEncodeRuleKind::Ratio;
        else if(kind=="exp2_over_source")rule.kind=NativeEncodeRuleKind::Exp2OverSource;
        else throw std::invalid_argument("unknown native Encode rule");
        if(rule.kind!=NativeEncodeRuleKind::Constant)
            require(rule.first<defined_at.size() && defined_at[rule.first]<=rule.index,"Encode rule source not defined");
        if(rule.kind==NativeEncodeRuleKind::Ratio)
            require(rule.second<defined_at.size() && defined_at[rule.second]<=rule.index,"Encode ratio denominator not defined");
        std::size_t used=0;
        if(rule.kind==NativeEncodeRuleKind::Exp2OverSource) {
            rule.exponent=std::stod(target,&used);
            require(used==target.size() && std::isfinite(rule.exponent) && std::isfinite(std::exp2(rule.exponent)),"invalid Encode exponent");
        } else {
            auto bits=std::stoull(target,&used,16);
            require(used==target.size(),"invalid Encode fallback bits");
            rule.constant=std::bit_cast<double>(bits);
            require(bits==scale_map.at(rule.index).bits && rule.constant>0 && std::isfinite(rule.constant),"Encode fallback scale mismatch");
        }
        result.push_back(rule);
    }
    require(result.size()==encoded.size(),"Encode rules do not cover every Encode");
    return result;
}
template<class GetScale>
double evaluate_hevm_native_encode_rule(const HevmNativeEncodeRule &rule,GetScale get) {
    auto scale=[&](std::uint64_t reg){auto value=get(reg);
        if(!(value>0) || !std::isfinite(value))throw std::invalid_argument("invalid native ciphertext scale");return value;};
    double result=rule.constant;
    if(rule.kind==NativeEncodeRuleKind::Source)result=scale(rule.first);
    else if(rule.kind==NativeEncodeRuleKind::Ratio)result=scale(rule.first)/scale(rule.second);
    else if(rule.kind==NativeEncodeRuleKind::Exp2OverSource)result=std::exp2(rule.exponent)/scale(rule.first);
    if(!(result>0) || !std::isfinite(result))throw std::invalid_argument("invalid native Encode expression result");
    return result;
}
} // namespace poseidon::mgpu
