// Full Cerium benchmark boundary: encrypted embeddings -> 32 block hidden.
// NativeLinearAdapter and unpaired native Boot are documented API differences.
#pragma once
#include "source_linear.hpp"
#include "source_softmax.hpp"
#include "source_attention.hpp"
#include "source_paired.hpp"
#include "source_timing.hpp"
#include "llama_manual/checkpoint.hpp"
#include <functional>
#include <array>

namespace cerium_source {
inline std::size_t head_order(std::size_t h) {return layout::q_head_order(h);}
// Optional diagnostics observe const ciphertexts only; never feed plaintext back.
// Callers enabling a decrypting observer must not report its wall as performance.
using BlockObserver=std::function<void(const char*,const std::vector<CT>&)>;
// Optional encrypted-operation substitution for separately labelled FHE controls.
// Empty preserves the original source operation, counter and stage order.
using EncryptedSiluOp=std::function<CT(Api&,const CT&,const CT&,std::size_t)>;
// Optional TWO logical real Gate/Up streams -> TWO owning real results, in the
// same tile order. This is a separately qualified refresh-plan substitution,
// not permission to perform the polynomial on a complex carrier. The caller
// binds the same Context/key/layout and immutable distinct SiLU profile.
using EncryptedSiluPairOp=std::function<std::array<CT,2>(Api&,const CT&,const CT&,const CT&,const CT&,std::size_t)>;
struct BlockEncryptedOps {
    Api *silu_api=nullptr;
    EncryptedSiluOp silu;
    EncryptedSiluPairOp silu_pair;
    void validate(const Api *primary) const {
        if((silu&&silu_pair)||bool(silu_api)!=(bool(silu)||bool(silu_pair))||(silu_api&&silu_api==primary))
            throw std::invalid_argument("Block SiLU hook requires a distinct Api and exactly one unary/pair operation");
    }
};
inline std::uint64_t block_boot_total(std::uint64_t primary,std::uint64_t secondary) {
    if(secondary>std::numeric_limits<std::uint64_t>::max()-primary)
        throw std::overflow_error("Block combined Boot counter overflow");
    return primary+secondary;
}
inline std::vector<CT> source_block(Api &api,NativeLinearAdapter &linear,const std::vector<CT> &input,
    const llama_manual::LayerWeights &w,const CT &encrypted_mask,std::size_t layer,const BlockObserver &observer={},const BlockEncryptedOps &ops={}) {
    ops.validate(&api);
    llama_manual::ModelConfig config;w.validate(config);
    OperatorTiming timing;
    auto checkpoint=[&](const char *stage,const std::vector<CT> &values){if(observer)observer(stage,values);};
    auto boot_count=[&](){return ops.silu_api?block_boot_total(api.boot_calls,ops.silu_api->boot_calls):api.boot_calls;};
    auto notify=[&](const char *stage){const auto boots=boot_count();timing.mark(stage,boots);std::cout<<"SOURCE_BLOCK layer="<<layer<<" stage="<<stage<<" boots="<<boots<<std::endl;};
    std::vector<CT> residual;
    {
        notify("attention_rms");auto normalized=env_flag("LLAMA_PAIR_RMS")?rmsnorm_pair(api,input,w.attention_norm,layer):rmsnorm(api,input,w.attention_norm,layer);checkpoint("attention_rms",normalized);
        notify("q_projection");auto q=linear.run(normalized,4096,4096,[&](auto o,auto i){return w.q(head_order(o/128)*128+o%128,i);},true,"q");checkpoint("q_projection",q);
        notify("k_projection");auto k=linear.run(normalized,4096,1024,[&](auto o,auto i){return w.k(o,i);},true,"k");checkpoint("k_projection",k);
        notify("v_projection");auto v=linear.run(normalized,4096,1024,[&](auto o,auto i){return w.v(o,i);},false,"v");checkpoint("v_projection",v);
        linear.release_input();normalized.clear();
        // Preserve attention()'s stage order: all QK, all Softmax, then all AV.
        notify("qk");std::vector<CT> scores;
        for(std::size_t i=0;i<16;i+=2) {
            auto pair=qk_pair(api,{&q[i],&q[i+1]},k[i/4]);
            for(auto &score:pair)scores.push_back(fast_attention_enabled()?std::move(score):api.drop(score,score.meta.q_count-2));
        }
        checkpoint("qk",scores);
        notify("softmax");std::vector<CT> probabilities;
        if(env_flag("LLAMA_PAIR_SOFTMAX")) {
            for(std::size_t i=0;i<scores.size();i+=2) {
                auto pair=softmax128_pair(api,scores[i],scores[i+1],encrypted_mask);
                for(auto &p:pair)probabilities.push_back(std::move(p));
            }
        } else for(const auto &score:scores)probabilities.push_back(softmax128(api,score,encrypted_mask));
        checkpoint("softmax",probabilities);
        notify("av");std::vector<CT> attention;
        for(std::size_t i=0;i<16;i+=4) {
            auto group=av_four(api,{&probabilities[i],&probabilities[i+1],&probabilities[i+2],&probabilities[i+3]},v[i/4]);
            for(auto &value:group)attention.push_back(std::move(value));
            std::cout<<"SOURCE_ATTENTION kv_tile="<<i/4<<" output_tiles="<<attention.size()<<std::endl;
        }
        checkpoint("av",attention);
        q.clear();k.clear();v.clear();
        notify("o_projection");auto projected=linear.run(attention,4096,4096,[&](auto o,auto i){return w.o(o,head_order(i/128)*128+i%128);},false,"o");checkpoint("o_projection",projected);
        linear.release_input();
        for(std::size_t i=0;i<16;i+=2) {
            auto p=api.add_pair(input[i],projected[i],input[i+1],projected[i+1]);
            residual.push_back(std::move(p[0]));residual.push_back(std::move(p[1]));
        }
        checkpoint("attention_residual",residual);
    }
    {
        notify("ffn_rms");auto normalized=env_flag("LLAMA_PAIR_RMS")?rmsnorm_pair(api,residual,w.ffn_norm,layer):rmsnorm(api,residual,w.ffn_norm,layer);checkpoint("ffn_rms",normalized);
        notify("gate_projection");auto gate=linear.run(normalized,4096,14336,[&](auto o,auto i){return w.gate(o,i);},false,"gate");checkpoint("gate_projection",gate);
        notify("up_projection");auto up=linear.run(normalized,4096,14336,[&](auto o,auto i){return w.up(o,i);},false,"up");checkpoint("up_projection",up);
        linear.release_input();normalized.clear();
        notify("swiglu");std::vector<CT> activated;
        if(ops.silu_pair) {
            for(std::size_t i=0;i<56;i+=2) {
                auto pair=ops.silu_pair(*ops.silu_api,gate[i],up[i],gate[i+1],up[i+1],i);
                for(auto &value:pair)activated.push_back(std::move(value));
            }
        } else {
        for(std::size_t i=0;i<56;++i)activated.push_back(ops.silu?ops.silu(*ops.silu_api,gate[i],up[i],i):silu_internal(api,gate[i],up[i]));
        }
        checkpoint("swiglu",activated);
        gate.clear();up.clear();
        notify("down_projection");auto down=linear.run(activated,14336,4096,[&](auto o,auto i){return w.down(o,i);},false,"down");checkpoint("down_projection",down);
        linear.release_input();
        for(std::size_t i=0;i<16;i+=2) {
            auto p=api.add_pair(residual[i],down[i],residual[i+1],down[i+1]);
            residual[i]=std::move(p[0]);residual[i+1]=std::move(p[1]);
        }
    }
    checkpoint("complete",residual);notify("complete");timing.finish(layer);return residual;
}
} // namespace cerium_source
