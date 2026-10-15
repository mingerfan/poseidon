#include "hevm_native_encode_rules.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
std::string read(const char *path){std::ifstream in(path,std::ios::binary);assert(in);return {std::istreambuf_iterator<char>(in),{}};}
int main(int argc,char **argv){assert(argc==4);auto binary=read(argv[1]),scales=read(argv[2]),text=read(argv[3]);
    using namespace poseidon::mgpu;
    auto rules=parse_hevm_native_encode_rules(binary,scales,text);
    std::size_t counts[4]={};for(auto rule:rules){++counts[static_cast<int>(rule.kind)];
        auto get=[](std::uint64_t reg){return std::exp2(40.0)*(1.0+double(reg%7)*1e-4);};
        auto actual=evaluate_hevm_native_encode_rule(rule,get);double expected=rule.constant;
        if(rule.kind==NativeEncodeRuleKind::Source)expected=get(rule.first);
        if(rule.kind==NativeEncodeRuleKind::Ratio)expected=get(rule.first)/get(rule.second);
        if(rule.kind==NativeEncodeRuleKind::Exp2OverSource)expected=std::exp2(rule.exponent)/get(rule.first);
        assert(std::bit_cast<std::uint64_t>(actual)==std::bit_cast<std::uint64_t>(expected));}
    assert(rules.size()==2739 && counts[0]==1845 && counts[1]==229 && counts[2]==95 && counts[3]==570);
    auto rejects=[&](std::string b,std::string t){bool rejected=false;try{parse_hevm_native_encode_rules(b,scales,t);}catch(const std::exception &){rejected=true;}assert(rejected);};
    auto corrupt=binary;corrupt.back()^=1;rejects(corrupt,text);
    auto end=text.find('\n'),row_end=text.find('\n',end+1);auto row=text.substr(end+1,row_end-end-1);
    rejects(binary,text+row+"\n");rejects(binary,text.substr(0,text.rfind('\n',text.size()-2)+1));
    auto wrong=text;auto position=wrong.find("constant");assert(position!=std::string::npos);wrong.replace(position,8,"unknown");rejects(binary,wrong);
    auto invalid=rules.back();invalid.kind=NativeEncodeRuleKind::Ratio;bool bad=false;
    try{evaluate_hevm_native_encode_rule(invalid,[](auto){return 0.0;});}catch(const std::invalid_argument &){bad=true;}assert(bad);
    std::cout<<"NATIVE_ENCODE_RULES PASS rules="<<rules.size()<<" dynamic="<<rules.size()-counts[0]<<" negative_tests=5 gpu_executed=false\n";
}
