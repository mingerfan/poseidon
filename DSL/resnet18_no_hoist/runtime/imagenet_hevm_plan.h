#pragma once
// Reads real compiler V2 artifacts. No GPU calls in this header.
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <zlib.h>

namespace imagenet_dsl {
using Json=nlohmann::json;
inline void require(bool ok,const std::string &message){if(!ok)throw std::runtime_error(message);}
inline double f64(std::uint64_t bits){return std::bit_cast<double>(bits);}
inline std::uint64_t hex(const std::string &s){return std::stoull(s,nullptr,16);}
struct Op {std::uint32_t code,dst,lhs,rhs;};
struct Meta {std::uint32_t q=0,components=2;double scale=0;};
struct Plan {
    std::vector<Op> ops;
    std::vector<std::uint64_t> returns;
    std::vector<std::uint32_t> cipher_uses,plain_uses;
    std::unordered_map<std::size_t,double> exact;
    std::vector<Meta> expected_cipher;
    std::uint64_t hash=14695981039346656037ULL;
    std::size_t boots=0;
    template<class F> static void sources(const Op &o,F consume){
        if(o.code==0 || o.code==0xffffffff)return;
        switch(o.code){
        case 1:case 2:case 3:case 4:case 10:case 13:case 14:case 16:case 17: consume(false,o.lhs);break;
        case 6:case 8:case 15:consume(false,o.lhs);consume(false,o.rhs);break;
        case 7:case 9:consume(false,o.lhs);consume(true,o.rhs);break;
        default:throw std::runtime_error("unsupported ImageNet HEVM opcode "+std::to_string(o.code));
        }
    }
    explicit Plan(const std::string &path){
        std::ifstream in(path,std::ios::binary|std::ios::ate);require(bool(in),"open HEVM");
        auto length=in.tellg();require(length>=64,"short HEVM");std::vector<unsigned char> raw(length);
        in.seekg(0);in.read(reinterpret_cast<char*>(raw.data()),raw.size());require(bool(in),"read HEVM");
        for(auto b:raw){hash^=b;hash*=1099511628211ULL;}
        auto u32=[&](std::size_t off){require(off+4<=raw.size(),"HEVM header range");std::uint32_t v;std::memcpy(&v,raw.data()+off,4);return v;};
        auto u64=[&](std::size_t off){require(off+8<=raw.size(),"HEVM header range");std::uint64_t v;std::memcpy(&v,raw.data()+off,8);return v;};
        const auto header=u32(4);require(u32(0)==0x48455632 && header==24,"expected compiler V2 HEVM");
        require(u64(8)==148 && u64(16)==1000,"ImageNet signature must be 148/1000");
        const auto body=u64(header),count=u64(header+8),nc=u64(header+16),np=u64(header+24);
        require(u64(header+32)==49 && body==40+8*(148*2+1000*3),"Q50/header contract");
        require(count<(1ULL<<31) && nc>=148 && nc<count+148 && np<=count,"HEVM counts");
        require(header+body+count*16==raw.size(),"HEVM length");
        std::size_t off=header+40;
        for(int group=0;group<4;++group){const int n=group<2?148:1000;const int expected=group%2==0?40:(group==1?31:3);
            for(int i=0;i<n;++i,off+=8)require(u64(off)==std::uint64_t(expected),"input/output Q scale header");}
        for(int i=0;i<1000;++i,off+=8){auto r=u64(off);require(r<nc,"return register");returns.push_back(r);}
        ops.resize(count);std::memcpy(ops.data(),raw.data()+header+body,count*16);
        cipher_uses.resize(nc);plain_uses.resize(np);
        std::vector<bool> cd(nc),pd(np);std::fill(cd.begin(),cd.begin()+148,true);
        for(const auto &o:ops){if(o.code==0xffffffff)continue;
            sources(o,[&](bool plain,std::uint32_t r){auto &def=plain?pd:cd;auto &use=plain?plain_uses:cipher_uses;
                require(r<def.size() && def[r],"undefined HEVM source");require(use[r]!=UINT32_MAX,"use overflow");++use[r];});
            auto &def=o.code==0?pd:cd;require(o.dst<def.size() && !def[o.dst],"duplicate/out-of-range SSA destination");def[o.dst]=true;
            if(o.code==10)++boots;
        }
        for(auto r:returns){require(cd[r],"undefined return");++cipher_uses[r];}
        sidecar(path+".native_scales","POSEIDON_HEVM_NATIVE_SCALE_V1",true);
        sidecar(path+".native_primitives","POSEIDON_HEVM_NATIVE_PRIMITIVE_V1",false);
        for(std::size_t i=0;i<ops.size();++i)if(ops[i].code==0 || ops[i].code==13)require(exact.count(i),"missing f64 metadata");
    }
    void sidecar(const std::string &path,const std::string &expected,bool encode){
        std::ifstream in(path);require(bool(in),"open f64 sidecar "+path);std::string magic,bound,bits;std::size_t count;
        in>>magic>>count>>bound;require(magic==expected && count==ops.size() && hex(bound)==hash,"f64 sidecar binding");
        std::size_t i;std::uint64_t arg,level;
        while(in>>i>>arg){if(encode)in>>level;in>>bits;require(bool(in) && i<ops.size(),"f64 row");
            const auto &o=ops[i];require(encode?(o.code==0 && o.lhs==arg && (o.rhs>>10)==level):(o.code==13 && arg==13),"f64 row opcode/args");
            const double value=f64(hex(bits));require(std::isfinite(value) && value>0 && exact.emplace(i,value).second,"f64 duplicate/value");}
        require(in.eof(),"malformed f64 sidecar");
    }
    std::size_t metadata(const std::vector<std::uint64_t> &q,int model) {
        require(q.size()==50 && (model==18 || model==50),"native Q50/model");
        require(boots==std::size_t(model==18?60:351),"native bootstrap call count");
        std::vector<Meta> c(cipher_uses.size()),p(plain_uses.size());
        for(int i=0;i<148;++i)c[i]={32,2,std::exp2(40)};
        for(std::size_t i=0;i<ops.size();++i){auto o=ops[i];if(o.code==0xffffffff)continue;
            if(o.code==0){p[o.dst]={1+(o.rhs>>10),1,exact.at(i)};require(p[o.dst].q<=50,"encode Q");continue;}
            Meta a=c[o.lhs],out=a;
            switch(o.code){
            case 1:case 2:case 14:break;
            case 3:case 17:{const auto n=o.code==3?1:o.rhs;require(n>0 && n<a.q,"rescale Q");
                if(n==2 && a.components==2)out.scale/=double(q[a.q-1]*q[a.q-2]);
                else for(std::uint32_t k=0;k<n;++k)out.scale/=double(q[a.q-1-k]);out.q-=n;break;}
            case 4:require(o.rhs>0 && o.rhs<a.q,"drop Q");out.q-=o.rhs;break;
            case 6:case 7:case 8:case 9:case 15:{const Meta b=(o.code==7 || o.code==9)?p[o.rhs]:c[o.rhs];
                require(a.q==b.q,"binary Q mismatch");
                if(o.code==6 || o.code==7)require(std::abs(a.scale-b.scale)<=1e-6*std::max(a.scale,b.scale),"native add scale mismatch");
                else out.scale*=b.scale;
                if(o.code==15)out.components=3;
                require(o.code!=8,"implicit mulcc/relinearization forbidden in ImageNet schedule");break;}
            case 10:require(a.q==6 && std::abs(std::log2(a.scale)-40)<1e-8 && o.rhs==30,"native bootstrap input");out={31,2,std::exp2(40)};break;
            case 13:out.scale=exact.at(i);break;
            case 16:require(a.components==3,"relinearize requires raw size3");out.components=2;break;
            default:throw std::runtime_error("unsupported metadata opcode");
            }
            require(out.q>0 && std::isfinite(out.scale) && out.scale>0,"invalid metadata at "+std::to_string(i));c[o.dst]=out;
        }
        for(auto r:returns)require(c[r].q==4 && c[r].components==2 && std::abs(std::log2(c[r].scale)-40)<1e-8,"native head Q4/40");
        expected_cipher=std::move(c);
        return ops.size();
    }
};
inline std::vector<Json> jsonlines(const std::string &path){
    std::ifstream in(path);require(bool(in),"open JSONL "+path);std::vector<Json> rows;std::string line;
    while(std::getline(in,line))rows.push_back(Json::parse(line));require(in.eof(),"read JSONL");return rows;
}
inline std::vector<double> payload(std::ifstream &in,const Json &row){
    const auto n=row.at("elements").get<std::uint64_t>(),size=row.at("payload_compressed_bytes").get<std::uint64_t>();
    require(n>0 && n<=32768 && size>0 && size<=compressBound(n*8),"payload dimensions");
    in.clear();in.seekg(row.at("payload_offset").get<std::uint64_t>());std::uint64_t hn,hs;
    in.read(reinterpret_cast<char*>(&hn),8);in.read(reinterpret_cast<char*>(&hs),8);require(bool(in)&&hn==n&&hs==size,"payload header");
    std::vector<unsigned char> compressed(size);in.read(reinterpret_cast<char*>(compressed.data()),size);require(bool(in),"payload read");
    std::vector<double> values(n);uLongf bytes=n*8;
    require(uncompress(reinterpret_cast<Bytef*>(values.data()),&bytes,compressed.data(),size)==Z_OK && bytes==n*8,"payload zlib");
    std::uint64_t hash=14695981039346656037ULL;
    for(double v:values){require(std::isfinite(v),"nonfinite constant/input");hash^=std::bit_cast<std::uint64_t>(v);hash*=1099511628211ULL;}
    require(hash==row.at("fnv64_words").get<std::uint64_t>(),"payload native FNV words");return values;
}
}
