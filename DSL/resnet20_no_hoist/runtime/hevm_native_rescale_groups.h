#pragma once
#include <cstdint>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <utility>

namespace poseidon::mgpu {
struct HevmNativeRescaleGroup {
    std::uint64_t begin,end,source,destination,count,mode,origin;
};
// Calls the native evaluator directly; no 51-bit compensation or scale overwrite.
template<class Evaluator,class Ciphertext>
void execute_hevm_native_rescale_group(Evaluator &eval,const Ciphertext &source,
    Ciphertext &destination,const HevmNativeRescaleGroup &group) {
    if(!group.count || group.mode>1 || source.meta.q_count<=group.count)
        throw std::invalid_argument("invalid native rescale dispatch");
    const auto input_q=source.meta.q_count;
    if(group.mode==1)eval.rescale_many(source,destination,group.count);
    else {
        eval.rescale(source,destination);
        for(std::uint64_t i=1;i<group.count;++i){Ciphertext next;
            eval.rescale(destination,next);destination=std::move(next);}
    }
    if(destination.meta.q_count!=input_q-group.count)
        throw std::runtime_error("native rescale dispatch Q mismatch");
}
template<class Ciphertext>
void reconcile_hevm_native_scale(const Ciphertext &left,Ciphertext &right) {
    if(left.meta.q_count!=right.meta.q_count || left.meta.parms_id!=right.meta.parms_id ||
        !(left.meta.scale>0) || !(right.meta.scale>0) ||
        !std::isfinite(left.meta.scale) || !std::isfinite(right.meta.scale) ||
        std::abs(std::log2(left.meta.scale/right.meta.scale))>1e-8)
        throw std::invalid_argument("native Add operands not Q/scale aligned");
    right.meta.scale=left.meta.scale;
}
// mode0: original Conv sequential rescale; mode1: original ReLU rescale_many.
inline std::vector<HevmNativeRescaleGroup> parse_hevm_native_rescale_groups(
    std::string_view binary,std::string_view metadata) {
    auto require=[](bool ok,const char *message){if(!ok)throw std::invalid_argument(message);};
    auto read=[&](std::size_t at,unsigned bytes) {
        require(at<=binary.size() && bytes<=binary.size()-at,"truncated rescale HEVM");
        std::uint64_t value=0;
        for(unsigned i=0;i<bytes;++i)value|=std::uint64_t(static_cast<unsigned char>(binary[at+i]))<<(8*i);
        return value;
    };
    require(read(0,4)==0x4845564D,"invalid rescale HEVM magic");
    auto header=read(4,4),body=read(header,8),nops=read(header+8,8);
    require(header<=binary.size() && body<=binary.size()-header,"invalid rescale HEVM sizes");
    auto start=header+body;
    require(nops<=(binary.size()-start)/8 && start+nops*8==binary.size(),"invalid rescale HEVM ops");
    std::uint64_t hash=14695981039346656037ULL;
    for(unsigned char byte:binary){hash^=byte;hash*=1099511628211ULL;}
    std::istringstream input{std::string(metadata)};std::string line,magic,extra;
    require(bool(std::getline(input,line)),"empty rescale metadata");
    std::istringstream first(line);std::uint64_t declared=0,binding=0;
    require(bool(first>>magic>>declared>>std::hex>>binding) && !(first>>extra),"invalid rescale header");
    require(magic=="POSEIDON_HEVM_NATIVE_RESCALE_V1" && declared==nops && binding==hash,"rescale binding mismatch");
    std::vector<std::uint64_t> uses(read(header+16,8));
    for(std::uint64_t i=0;i<nops;++i) {
        auto at=start+8*i,opcode=read(at,2),lhs=read(at+4,2),rhs=read(at+6,2);
        if(opcode!=65535 && opcode!=0){require(lhs<uses.size(),"invalid ciphertext register");++uses[lhs];
            if(opcode==6 || opcode==8){require(rhs<uses.size(),"invalid binary register");++uses[rhs];}}
    }
    const auto nargs=read(8,8),nresults=read(16,8);
    for(std::uint64_t i=0;i<nresults;++i){auto reg=read(header+40+8*(2*nargs+2*nresults+i),8);
        require(reg<uses.size(),"invalid result register");++uses[reg];}
    std::vector<HevmNativeRescaleGroup> result;std::uint64_t previous_end=0;
    while(std::getline(input,line)) {
        std::istringstream row(line);HevmNativeRescaleGroup g{};
        require(bool(row>>g.begin>>g.end>>g.source>>g.destination>>g.count>>g.mode>>g.origin) && !(row>>extra),"invalid rescale row");
        require(g.count>0 && g.mode<=1 && g.begin>=previous_end && g.end<=nops && g.end>g.begin && g.end-g.begin==2*g.count,"invalid overlapping rescale group");
        auto source=g.source;
        for(std::uint64_t j=0;j<g.count;++j){auto alloc=start+8*(g.begin+2*j),at=alloc+8;
            require(read(alloc,2)==65535 && read(at,2)==3 && read(at+4,2)==source && read(at+6,2)==0,"rescale group is not a literal chain");
            auto dst=read(at+2,2);require(dst<uses.size(),"invalid rescale destination");
            if(j+1<g.count)require(uses[dst]==1,"rescale intermediate has another consumer");
            source=dst;
        }
        require(source==g.destination,"rescale final destination mismatch");
        previous_end=g.end;result.push_back(g);
    }
    require(!result.empty(),"rescale metadata has no groups");
    return result;
}
} // namespace poseidon::mgpu
