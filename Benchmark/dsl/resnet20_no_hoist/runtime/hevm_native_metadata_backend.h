#pragma once
#include "hevm_native_schedule.h"
#include <algorithm>
#include <cassert>
namespace strict_native_metadata {
using namespace poseidon::mgpu;
struct CT {struct {std::uint64_t q_count,parms_id;double scale;} meta;};
struct Backend {
    std::unordered_map<std::uint64_t,std::uint64_t> rotation_levels;
    std::vector<std::uint64_t> primes;
    std::unordered_map<std::uint64_t,CT> ct,pt;
    std::size_t boot=0,encodes=0,seq=0,many=0,peak=27;
    explicit Backend(const std::vector<std::uint64_t> &q):primes(q){for(int i=0;i<27;++i)ct[i]={{32,31,std::exp2(40)}};}
    double cipher_scale(std::uint64_t reg){return ct.at(reg).meta.scale;}
    void encode(std::uint64_t reg,std::uint64_t,std::uint64_t q,double s){pt[reg]={{q,q-1,s}};++encodes;}
    void rescale_group(const HevmNativeRescaleGroup &g){auto out=ct.at(g.source);auto q=out.meta.q_count;
        if(g.mode==1 && g.count==2)out.meta.scale/=double(primes[q-1]*primes[q-2]);
        else for(std::uint64_t i=0;i<g.count;++i)out.meta.scale/=double(primes[q-1-i]);
        out.meta.q_count-=g.count;out.meta.parms_id=out.meta.q_count-1;ct[g.destination]=out;
        seq+=g.mode==0;many+=g.mode==1;peak=std::max(peak,ct.size());}
    void operation(const NativeHevmAction &a){auto out=ct.at(a.left);
        switch(a.opcode){
        case 1:rotation_levels[a.raw_index]=out.meta.q_count;break;
        case 2:break;
        case 4:out.meta.q_count-=a.right;out.meta.parms_id=out.meta.q_count-1;break;
        case 6:{auto right=ct.at(a.right);reconcile_hevm_native_scale(out,right);break;}
        case 7:{auto right=pt.at(a.right);reconcile_hevm_native_scale(out,right);break;}
        case 8:{auto right=ct.at(a.right);assert(out.meta.q_count==right.meta.q_count);out.meta.scale*=right.meta.scale;break;}
        case 9:{auto right=pt.at(a.right);assert(out.meta.q_count==right.meta.q_count);out.meta.scale*=right.meta.scale;break;}
        case 10:assert(out.meta.q_count==6 && std::abs(std::log2(out.meta.scale)-40)<1e-8);out={{31,30,std::exp2(40)}};++boot;break;
        default:throw std::runtime_error("unexpected opcode in metadata backend");}
        assert(std::isfinite(out.meta.scale) && out.meta.scale>0 && out.meta.q_count>0);
        ct[a.destination]=out;peak=std::max(peak,ct.size());}
    void release_cipher(std::uint64_t reg){assert(ct.erase(reg)==1);}
    void release_plain(std::uint64_t reg){assert(pt.erase(reg)==1);}
};
}
