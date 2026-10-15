#pragma once
#include "hevm_native_encode_rules.h"
#include "hevm_native_rescale_groups.h"
#include <unordered_map>

namespace poseidon::mgpu {
struct NativeHevmAction {
    std::uint64_t raw_index,opcode,destination,left,right;
    HevmNativeRescaleGroup group{};
    HevmNativeEncodeRule encode{};
    std::vector<std::uint64_t> cipher_reads,plain_reads;
};
struct NativeHevmSchedule {
    std::uint64_t input_count,result_register;
    std::vector<NativeHevmAction> actions;
    std::unordered_map<std::uint64_t,std::uint64_t> cipher_uses,plain_uses;
};
inline NativeHevmSchedule build_native_hevm_schedule(std::string_view binary,
    std::string_view scales,std::string_view expressions,std::string_view rescale_metadata) {
    const auto rules=parse_hevm_native_encode_rules(binary,scales,expressions);
    const auto groups=parse_hevm_native_rescale_groups(binary,rescale_metadata);
    std::unordered_map<std::uint64_t,HevmNativeEncodeRule> encode;
    std::unordered_map<std::uint64_t,HevmNativeRescaleGroup> rescale;
    for(auto row:rules)encode.emplace(row.index,row);
    for(auto row:groups)rescale.emplace(row.begin,row);
    auto read=[&](std::size_t at,unsigned bytes=8){if(at>binary.size() || bytes>binary.size()-at)throw std::invalid_argument("truncated native schedule");
        std::uint64_t result=0;for(unsigned i=0;i<bytes;++i)result|=std::uint64_t(static_cast<unsigned char>(binary[at+i]))<<(8*i);return result;};
    const auto header=read(4,4),body=read(header),nops=read(header+8),start=header+body;
    NativeHevmSchedule schedule;schedule.input_count=read(8);auto nresults=read(16);
    if(schedule.input_count!=27 || nresults!=1)throw std::invalid_argument("native schedule input/result contract");
    schedule.result_register=read(header+40+8*(2*schedule.input_count+2*nresults));
    std::unordered_set<std::uint64_t> defined_ct,defined_pt;
    for(std::uint64_t i=0;i<schedule.input_count;++i)defined_ct.insert(i);
    for(std::uint64_t i=0;i<nops;) {
        NativeHevmAction action{};action.raw_index=i;auto found=rescale.find(i);
        if(found!=rescale.end()) {
            action.group=found->second;action.opcode=256;
            action.destination=action.group.destination;action.left=action.group.source;
            action.cipher_reads.push_back(action.left);i=action.group.end;
        } else {
            auto at=start+8*i;action.opcode=read(at,2);action.destination=read(at+2,2);
            action.left=read(at+4,2);action.right=read(at+6,2);++i;
            if(action.opcode==65535)continue;
            if(action.opcode==0) {
                action.encode=encode.at(action.raw_index);
                if(action.encode.kind!=NativeEncodeRuleKind::Constant)action.cipher_reads.push_back(action.encode.first);
                if(action.encode.kind==NativeEncodeRuleKind::Ratio)action.cipher_reads.push_back(action.encode.second);
            } else {
                if(action.opcode>10 || action.opcode==5 || action.opcode==3)
                    throw std::invalid_argument("unsupported or ungrouped strict native opcode");
                action.cipher_reads.push_back(action.left);
                if(action.opcode==6 || action.opcode==8)action.cipher_reads.push_back(action.right);
                if(action.opcode==7 || action.opcode==9)action.plain_reads.push_back(action.right);
            }
        }
        for(auto reg:action.cipher_reads){if(!defined_ct.count(reg))throw std::invalid_argument("undefined native ciphertext dependency");++schedule.cipher_uses[reg];}
        for(auto reg:action.plain_reads){if(!defined_pt.count(reg))throw std::invalid_argument("undefined native plaintext dependency");++schedule.plain_uses[reg];}
        if(action.opcode==0){if(!defined_pt.insert(action.destination).second)throw std::invalid_argument("native plaintext register reused");}
        else if(!defined_ct.insert(action.destination).second)throw std::invalid_argument("native ciphertext register reused");
        schedule.actions.push_back(std::move(action));
    }
    if(!defined_ct.count(schedule.result_register))throw std::invalid_argument("missing native result");
    ++schedule.cipher_uses[schedule.result_register];return schedule;
}
template<class Backend>
std::size_t execute_native_hevm_schedule(const NativeHevmSchedule &schedule,Backend &backend,std::size_t max_ops=0) {
    auto ct=schedule.cipher_uses,pt=schedule.plain_uses;std::size_t done=0;
    for(const auto &action:schedule.actions) {
        if(max_ops && done==max_ops)break;
        if(action.opcode==0) {
            auto scale=evaluate_hevm_native_encode_rule(action.encode,[&](auto reg){return backend.cipher_scale(reg);});
            backend.encode(action.destination,action.left,(action.right>>10)+1,scale);
        } else if(action.opcode==256)backend.rescale_group(action.group);
        else backend.operation(action);
        for(auto reg:action.cipher_reads)if(--ct.at(reg)==0)backend.release_cipher(reg);
        for(auto reg:action.plain_reads)if(--pt.at(reg)==0)backend.release_plain(reg);
        ++done;
    }
    return done;
}
} // namespace poseidon::mgpu
