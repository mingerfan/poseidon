#include "poseidon/basics/util/rlwe.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/poseidon_context.h"
#include <cstdint>
#include <iostream>
#include <vector>
int main(int argc, char **argv){
    using namespace poseidon;
    const unsigned log_n=argc==2?std::stoul(argv[1]):10;
    ParametersLiteral p(CKKS,log_n,log_n-1,20,192,0,Modulus(0),{},{},sec_level_type::none);
    p.set_log_modulus({32,32},{32});
    PoseidonContext context(p);
    const auto id=context.crt_context()->key_parms_id();
    const auto data=context.crt_context()->get_context_data(id);
    const auto &q=data->coeff_modulus();
    const std::size_t n=std::size_t{1}<<log_n;
    constexpr std::uint64_t sentinel=0xdecafbad12345678ULL;
    unsigned guards=0,hamming=0,rns=0;
    for(unsigned trial=0;trial<64;++trial){
        std::vector<std::uint64_t> out(n*q.size()+1,sentinel);
        out.back()=sentinel;
        util::sample_poly_ternary_with_hamming(context.random_generator()->create(),context,id,out.data());
        guards+=(out.back()!=sentinel);
        unsigned nonzero=0;
        for(std::size_t i=0;i<n;++i){
            const int sign=out[i]==0?0:out[i]==1?1:out[i]==q[0].value()-1?-1:2;
            nonzero+=(sign!=0);
            for(std::size_t limb=0;limb<q.size();++limb){
                const auto expected=sign==0?0ULL:sign==1?1ULL:q[limb].value()-1;
                if(sign==2 || out[limb*n+i]!=expected)++rns;
            }
        }
        hamming+=(nonzero!=192);
    }
    const bool pass=guards==0 && hamming==0 && rns==0;
    std::cout<<"SECRET_SAMPLER "<<(pass?"PASS":"FAIL")<<" degree="<<n<<" trials=64 poisoned_buffer=true guard_corruptions="<<guards
             <<" wrong_hamming_trials="<<hamming<<" rns_mismatches="<<rns<<" gpu_executed=false\n";
    return pass?0:1;
}
