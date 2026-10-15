#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace poseidon::mgpu {
struct HevmNativeScale {
    std::uint64_t operation_index, value_id, constant_index, level, bits;
    double scale;
};

inline std::vector<HevmNativeScale> parse_hevm_native_scales(
    std::string_view binary, std::string_view metadata) {
    auto require=[](bool ok, const char *message) {
        if(!ok) throw std::invalid_argument(message);
    };
    auto read=[&](std::size_t at, unsigned bytes) {
        require(at<=binary.size() && bytes<=binary.size()-at,"truncated HEVM scale binding");
        std::uint64_t value=0;
        for(unsigned i=0;i<bytes;++i)
            value|=std::uint64_t(static_cast<unsigned char>(binary[at+i]))<<(8*i);
        return value;
    };
    require(read(0,4)==0x4845564D,"invalid HEVM scale binding magic");
    const auto header=read(4,4), nargs=read(8,8);
    require(header>=24 && header<=binary.size(),"invalid HEVM header");
    const auto body=read(header,8), nops=read(header+8,8);
    require(body>=40 && body<=binary.size()-header,"invalid HEVM config size");
    const auto start=header+body;
    require(nops<=(binary.size()-start)/8 && nops*8==binary.size()-start,
            "invalid HEVM operations size");
    std::uint64_t hash=14695981039346656037ULL;
    for(unsigned char byte:binary){hash^=byte;hash*=1099511628211ULL;}
    std::istringstream input{std::string(metadata)};
    std::string line,magic,extra;
    require(bool(std::getline(input,line)),"empty native scale metadata");
    std::istringstream first(line);
    std::uint64_t count=0,binding=0;
    require(bool(first>>magic>>count>>std::hex>>binding) && !(first>>extra),
            "invalid native scale metadata header");
    require(magic=="POSEIDON_HEVM_NATIVE_SCALE_V1" && count==nops && binding==hash,
            "native scale metadata does not bind to this HEVM");
    std::vector<std::uint64_t> value_ids;
    value_ids.reserve(nops);
    std::uint64_t value=nargs;
    for(std::size_t i=0;i<nops;++i) {
        if(read(start+8*i,2)!=65535)++value;
        value_ids.push_back(value);
    }
    std::unordered_set<std::uint64_t> seen;
    std::vector<HevmNativeScale> result;
    while(std::getline(input,line)) {
        require(!line.empty(),"empty native scale row");
        std::istringstream row(line);
        std::uint64_t index=0,constant=0,level=0,bits=0;
        require(bool(row>>index>>constant>>level>>std::hex>>bits) && !(row>>extra),
                "invalid native scale row");
        require(index<nops && seen.insert(index).second,"duplicate/invalid native scale index");
        const auto at=start+8*index;
        require(read(at,2)==0 && read(at+4,2)==constant && (read(at+6,2)>>10)==level,
                "native scale row does not match Encode operands");
        const double scale=std::bit_cast<double>(bits);
        require(scale>0 && std::isfinite(scale),"native Encode scale is not positive finite");
        result.push_back({index,value_ids[index],constant,level,bits,scale});
    }
    require(!result.empty(),"native scale metadata has no rows");
    return result;
}
} // namespace poseidon::mgpu
