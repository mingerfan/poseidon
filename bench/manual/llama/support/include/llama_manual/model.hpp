#pragma once

#include "llama_manual/linear.hpp"
#include <functional>
#include <string>

namespace llama_manual {

struct ModelConfig {
    std::size_t hidden = 4096, intermediate = 14336, heads = 32, kv_heads = 8;
    std::size_t head_dim = 128, layers = 32, vocabulary = 128256;
    double rms_epsilon = 1e-5;
    double rope_theta = 500000.0, rope_factor = 8.0;
    double rope_low_factor = 1.0, rope_high_factor = 4.0, rope_original_context = 8192.0;
    bool scaled_rope = true;
    void validate(const Layout &l) const {
        l.validate();
        if (!hidden || !intermediate || !layers || !vocabulary || !heads || !kv_heads ||
            !head_dim || head_dim % 2 || heads % kv_heads || heads * head_dim != hidden ||
            l.feature_tile % head_dim || (l.feature_tile & (l.feature_tile-1)) ||
            (l.token_capacity & (l.token_capacity-1)) ||
            !(rms_epsilon > 0) || !(rope_theta > 1) || !(rope_factor >= 1) ||
            !(rope_high_factor > rope_low_factor) || !(rope_low_factor > 0) ||
            !(rope_original_context > 0)) throw std::invalid_argument("unsupported Llama layout/config");
    }
};

// A calibrated polynomial in z=(x-center)*gain, coefficients lowest degree first.
// These are REQUIRED application inputs. No guessed coefficients are embedded
// for pretrained Llama. Interval validity must be checked against calibration.
struct Polynomial {
    double center = 0, gain = 1, domain_low = 0, domain_high = 0;
    std::vector<double> coefficients;
    void validate() const {
        if (coefficients.empty() || coefficients.size() > 65 || !std::isfinite(center) ||
            !std::isfinite(gain) || gain == 0 || !std::isfinite(domain_low) ||
            !std::isfinite(domain_high) || domain_low >= domain_high)
            throw std::invalid_argument("missing/invalid calibrated polynomial");
        for (double c : coefficients) if (!std::isfinite(c))
            throw std::invalid_argument("non-finite polynomial coefficient");
    }
};

struct Approximation {
    Polynomial attention_rsqrt_seed, ffn_rsqrt_seed, exponential, reciprocal_seed, silu;
    unsigned rsqrt_steps = 0, reciprocal_steps = 0;
    void validate() const {
        attention_rsqrt_seed.validate(); ffn_rsqrt_seed.validate(); exponential.validate();
        reciprocal_seed.validate(); silu.validate();
        if (rsqrt_steps > 12 || reciprocal_steps > 12)
            throw std::invalid_argument("excessive Newton iteration count");
    }
};

// Dense CPU weight ownership is per layer; callers may implement load/release
// with mmap or a bounded cache. No all-layer encoded plaintext cache is required.
struct Matrix {
    std::size_t rows = 0, columns = 0;
    std::vector<double> values;
    double operator()(std::size_t r, std::size_t c) const { return values.at(r * columns + c); }
    void validate(std::size_t r, std::size_t c) const {
        if (rows != r || columns != c || !c || r > std::numeric_limits<std::size_t>::max()/c ||
            values.size() != r*c) throw std::invalid_argument("weight matrix shape mismatch");
    }
};
struct LayerWeights {
    std::vector<double> attention_norm, ffn_norm;
    Matrix q, k, v, o, gate, up, down;
    void validate(const ModelConfig &c) const {
        if (attention_norm.size()!=c.hidden || ffn_norm.size()!=c.hidden)
            throw std::invalid_argument("normalization weight shape mismatch");
        q.validate(c.hidden,c.hidden); k.validate(c.kv_heads*c.head_dim,c.hidden);
        v.validate(c.kv_heads*c.head_dim,c.hidden); o.validate(c.hidden,c.hidden);
        gate.validate(c.intermediate,c.hidden); up.validate(c.intermediate,c.hidden);
        down.validate(c.hidden,c.intermediate);
    }
};

// Cerium uses native Meta interleaved Q/K rows. Providers own at most one
// decoded layer at a time; no encrypted intermediate is exposed to them.
struct NativeWeightProvider {
    virtual ~NativeWeightProvider() = default;
    virtual LayerWeights load_layer(std::size_t layer) = 0;
    virtual void layer_completed(std::size_t) {} // Public progress only; no secret-key feedback.
};

template<class Ops> using Tensor = std::vector<typename Ops::Cipher>;

inline std::vector<double> constant_slots(const Layout &l, double value) {
    return std::vector<double>(l.slots,value);
}

template<class Ops>
typename Ops::Cipher scale(Ops &ops, const Layout &l, const typename Ops::Cipher &x, double a) {
    return ops.mul_plain(x,constant_slots(l,a));
}

template<class Ops>
typename Ops::Cipher polynomial(Ops &ops, const Layout &l,
    const typename Ops::Cipher &x, const Polynomial &p) {
    p.validate();
    // A backend may record calibration-domain observations in floating tests.
    // The GPU implementation performs no decryption or data-dependent checks.
    ops.observe_domain(x,p.domain_low,p.domain_high);
    auto z=ops.add_plain(x,constant_slots(l,-p.center));
    z=scale(ops,l,z,p.gain);
    auto y=ops.constant_like(z,p.coefficients.back());
    for (std::size_t i=p.coefficients.size()-1;i>0;--i) {
        y=ops.multiply(y,z);
        y=ops.add_plain(y,constant_slots(l,p.coefficients[i-1]));
    }
    return y;
}

template<class Ops>
typename Ops::Cipher inverse_sqrt(Ops &ops,const Layout &l,
    const typename Ops::Cipher &x,const Polynomial &seed,unsigned steps) {
    auto y=polynomial(ops,l,x,seed);
    for(unsigned i=0;i<steps;++i) {
        auto yy=ops.multiply(y,y);
        auto xyy=ops.multiply(x,yy);
        auto correction=ops.add_plain(scale(ops,l,xyy,-0.5),constant_slots(l,1.5));
        y=ops.multiply(y,correction);
    }
    return y;
}

template<class Ops>
typename Ops::Cipher inverse(Ops &ops,const Layout &l,
    const typename Ops::Cipher &x,const Polynomial &seed,unsigned steps) {
    auto y=polynomial(ops,l,x,seed);
    for(unsigned i=0;i<steps;++i) {
        auto xy=ops.multiply(x,y);
        auto correction=ops.add_plain(scale(ops,l,xy,-1),constant_slots(l,2));
        y=ops.multiply(y,correction);
    }
    return y;
}

template<class Ops>
Tensor<Ops> project(Ops &ops,const Layout &l,const Tensor<Ops> &x,const Matrix &w) {
    w.validate(w.rows,w.columns);
    auto prepared=ops.prepare_linear(x,l.slots);
    auto at=[&w](std::size_t out,std::size_t in){return w(out,in);};
    return linear(ops,l,prepared,w.columns,w.rows,at,2);
}

template<class Ops>
Tensor<Ops> rmsnorm(Ops &ops,const Layout &l,const ModelConfig &c,
    const Tensor<Ops> &x,std::size_t tokens,const std::vector<double> &gamma,
    const Polynomial &seed,unsigned steps) {
    if(x.size()!=l.tiles(c.hidden)||gamma.size()!=c.hidden)
        throw std::invalid_argument("RMSNorm shape mismatch");
    std::optional<typename Ops::Cipher> total;
    for(const auto &tile:x) {
        auto squared=ops.multiply(tile,tile);
        if(total) total=ops.add(*total,squared); else total.emplace(std::move(squared));
    }
    for(std::size_t step=1;step<l.feature_tile;step*=2) {
        auto rotated=ops.rotate(*total,static_cast<int>(step*l.token_capacity));
        total=ops.add(*total,rotated);
    }
    auto variance=scale(ops,l,*total,1.0/static_cast<double>(c.hidden));
    variance=ops.add_plain(variance,constant_slots(l,c.rms_epsilon));
    std::vector<double> valid(l.slots,0), filler(l.slots,0);
    for(std::size_t s=0;s<l.slots;++s) {
        if(s%l.token_capacity<tokens) valid[s]=1;
        else filler[s]=(seed.domain_low+seed.domain_high)/2;
    }
    // Keep padded token lanes inside the calibrated domain before evaluating
    // the inverse square root; mask them again at the output.
    variance=ops.add_plain(ops.mul_plain(variance,valid),filler);
    auto inv=inverse_sqrt(ops,l,variance,seed,steps);
    Tensor<Ops> out;
    for(std::size_t tile=0;tile<x.size();++tile) {
        std::vector<double> weight(l.slots,0);
        for(std::size_t f=0;f<l.feature_tile;++f) {
            const auto global=tile*l.feature_tile+f;
            if(global>=c.hidden) continue;
            for(std::size_t t=0;t<tokens;++t) weight[f*l.token_capacity+t]=gamma[global];
        }
        auto product=ops.multiply(x[tile],inv);
        out.push_back(ops.mul_plain(product,weight));
    }
    return out;
}

inline double rotary_frequency(const ModelConfig &c,std::size_t pair) {
    double freq=1.0/std::pow(c.rope_theta,2.0*pair/c.head_dim);
    if(!c.scaled_rope) return freq;
    const double wavelength=2.0*std::acos(-1.0)/freq;
    if(wavelength>c.rope_original_context/c.rope_low_factor) return freq/c.rope_factor;
    if(wavelength<c.rope_original_context/c.rope_high_factor) return freq;
    const double blend=(c.rope_original_context/wavelength-c.rope_low_factor)/
        (c.rope_high_factor-c.rope_low_factor);
    return (1.0-blend)*freq/c.rope_factor+blend*freq;
}

// HF split-half RoPE convention. Native Meta interleaved weights require a
// reviewed conversion; do not feed their matrices to this entry point.
template<class Ops>
Tensor<Ops> rope(Ops &ops,const Layout &l,const ModelConfig &c,const Tensor<Ops> &x,
    std::size_t width,std::size_t tokens,std::size_t first_position) {
    Tensor<Ops> out;
    const auto half=c.head_dim/2;
    for(std::size_t tile=0;tile<x.size();++tile) {
        std::vector<double> cosine(l.slots,0),lower(l.slots,0),upper(l.slots,0);
        for(std::size_t f=0;f<l.feature_tile;++f) {
            if(tile*l.feature_tile+f>=width) continue;
            const auto channel=f%c.head_dim;
            for(std::size_t t=0;t<tokens;++t) {
                const double angle=(first_position+t)*rotary_frequency(c,channel%half);
                const auto s=f*l.token_capacity+t;
                cosine[s]=std::cos(angle);
                if(channel<half) lower[s]=-std::sin(angle); else upper[s]=std::sin(angle);
            }
        }
        auto base=ops.mul_plain(x[tile],cosine);
        auto left=ops.rotate(x[tile],static_cast<int>(half*l.token_capacity));
        auto right=ops.rotate(x[tile],-static_cast<int>(half*l.token_capacity));
        auto low=ops.mul_plain(left,lower), high=ops.mul_plain(right,upper);
        out.push_back(ops.add(base,ops.add(low,high)));
    }
    return out;
}

template<class Ops>
typename Ops::Cipher head(Ops &ops,const Layout &l,const ModelConfig &c,
    const Tensor<Ops> &x,std::size_t index,std::size_t tokens) {
    const auto feature=index*c.head_dim;
    auto shifted=ops.rotate(x.at(feature/l.feature_tile),
        static_cast<int>((feature%l.feature_tile)*l.token_capacity));
    std::vector<double> mask(l.slots,0);
    for(std::size_t f=0;f<c.head_dim;++f)
        for(std::size_t t=0;t<tokens;++t) mask[f*l.token_capacity+t]=1;
    return ops.mul_plain(shifted,mask);
}

template<class Ops>
typename Ops::Cipher broadcast_token(Ops &ops,const Layout &l,const ModelConfig &c,
    const typename Ops::Cipher &x,std::size_t token) {
    auto shifted=ops.rotate(x,static_cast<int>(token));
    std::vector<double> mask(l.slots,0);
    for(std::size_t f=0;f<c.head_dim;++f) mask[f*l.token_capacity]=1;
    auto out=ops.mul_plain(shifted,mask);
    for(std::size_t step=1;step<l.token_capacity;step*=2)
        out=ops.add(out,ops.rotate(out,-static_cast<int>(step)));
    return out;
}

// Causal prefill with GQA. Scores for one head are retained; heads execute
// serially to bound memory. Correctness baseline, not Cerium speed parity.
template<class Ops>
Tensor<Ops> attention(Ops &ops,const Layout &l,const ModelConfig &c,
    const Tensor<Ops> &q,const Tensor<Ops> &k,const Tensor<Ops> &v,
    std::size_t tokens,const Approximation &a) {
    using Cipher=typename Ops::Cipher;
    std::vector<std::optional<Cipher>> tiles(l.tiles(c.hidden));
    for(std::size_t h=0;h<c.heads;++h) {
        auto query=head(ops,l,c,q,h,tokens);
        auto key=head(ops,l,c,k,h/(c.heads/c.kv_heads),tokens);
        auto value=head(ops,l,c,v,h/(c.heads/c.kv_heads),tokens);
        std::vector<Cipher> scores;
        std::optional<Cipher> mean;
        for(std::size_t j=0;j<tokens;++j) {
            auto broadcast=broadcast_token(ops,l,c,key,j);
            auto score=ops.multiply(query,broadcast);
            for(std::size_t step=1;step<l.feature_tile;step*=2)
                score=ops.add(score,ops.rotate(score,static_cast<int>(step*l.token_capacity)));
            score=scale(ops,l,score,1/std::sqrt(static_cast<double>(c.head_dim)));
            std::vector<double> mask(l.slots,0);
            for(std::size_t s=0;s<l.slots;++s)
                if(s%l.token_capacity>=j && s%l.token_capacity<tokens) mask[s]=1;
            auto valid=ops.mul_plain(score,mask);
            if(mean) mean=ops.add(*mean,valid); else mean.emplace(std::move(valid));
            scores.push_back(std::move(score));
        }
        std::vector<double> divisor(l.slots,0);
        for(std::size_t s=0;s<l.slots;++s)
            if(s%l.token_capacity<tokens) divisor[s]=1.0/(s%l.token_capacity+1);
        mean=ops.mul_plain(*mean,divisor);
        std::vector<Cipher> exps;
        std::optional<Cipher> denominator;
        for(std::size_t j=0;j<tokens;++j) {
            auto centered=ops.sub(scores[j],*mean);
            std::vector<double> mask(l.slots,0);
            for(std::size_t s=0;s<l.slots;++s)
                if(s%l.token_capacity>=j && s%l.token_capacity<tokens) mask[s]=1;
            // Set excluded scores to 0 BEFORE polynomial evaluation, then
            // mask exp(0) out of the denominator. No finite "-infinity" hack.
            centered=ops.mul_plain(centered,mask);
            auto ex=ops.mul_plain(polynomial(ops,l,centered,a.exponential),mask);
            if(denominator) denominator=ops.add(*denominator,ex);
            else denominator.emplace(ops.copy(ex));
            exps.push_back(std::move(ex));
        }
        scores.clear();
        std::vector<double> filler(l.slots,0);
        for(std::size_t s=0;s<l.slots;++s)
            if(s%l.token_capacity>=tokens) filler[s]=(a.reciprocal_seed.domain_low+a.reciprocal_seed.domain_high)/2;
        denominator=ops.add_plain(*denominator,filler);
        auto inv=inverse(ops,l,*denominator,a.reciprocal_seed,a.reciprocal_steps);
        std::optional<Cipher> sum;
        for(std::size_t j=0;j<tokens;++j) {
            auto probability=ops.multiply(exps[j],inv);
            auto broadcast=broadcast_token(ops,l,c,value,j);
            auto term=ops.multiply(probability,broadcast);
            if(sum) sum=ops.add(*sum,term); else sum.emplace(std::move(term));
        }
        std::vector<double> head_mask(l.slots,0);
        for(std::size_t f=0;f<c.head_dim;++f)
            for(std::size_t t=0;t<tokens;++t) head_mask[f*l.token_capacity+t]=1;
        auto clean=ops.mul_plain(*sum,head_mask);
        const auto feature=h*c.head_dim;
        auto placed=ops.rotate(clean,-static_cast<int>((feature%l.feature_tile)*l.token_capacity));
        auto &target=tiles[feature/l.feature_tile];
        if(target) target=ops.add(*target,placed); else target.emplace(std::move(placed));
    }
    Tensor<Ops> out;
    for(auto &tile:tiles) out.push_back(std::move(*tile));
    return out;
}

template<class Ops>
Tensor<Ops> swiglu(Ops &ops,const Layout &l,const Tensor<Ops> &x,
    const LayerWeights &w,const Approximation &a) {
    auto gate=project(ops,l,x,w.gate), up=project(ops,l,x,w.up);
    Tensor<Ops> activated;
    for(std::size_t i=0;i<gate.size();++i) {
        auto silu=polynomial(ops,l,gate[i],a.silu);
        activated.push_back(ops.multiply(silu,up[i]));
    }
    return project(ops,l,activated,w.down);
}

template<class Ops>
Tensor<Ops> decoder_block(Ops &ops,const Layout &l,const ModelConfig &c,
    const Tensor<Ops> &x,std::size_t tokens,std::size_t first_position,
    const LayerWeights &w,const Approximation &a) {
    c.validate(l); w.validate(c); a.validate();
    if(!tokens||tokens>l.token_capacity) throw std::invalid_argument("invalid token count");
    auto norm=rmsnorm(ops,l,c,x,tokens,w.attention_norm,a.attention_rsqrt_seed,a.rsqrt_steps);
    auto q=rope(ops,l,c,project(ops,l,norm,w.q),c.hidden,tokens,first_position);
    auto k=rope(ops,l,c,project(ops,l,norm,w.k),c.kv_heads*c.head_dim,tokens,first_position);
    auto v=project(ops,l,norm,w.v);
    auto context=attention(ops,l,c,q,k,v,tokens,a);
    auto projected=project(ops,l,context,w.o);
    Tensor<Ops> residual;
    for(std::size_t i=0;i<x.size();++i) residual.push_back(ops.add(x[i],projected[i]));
    auto ffn_input=rmsnorm(ops,l,c,residual,tokens,w.ffn_norm,a.ffn_rsqrt_seed,a.rsqrt_steps);
    auto ffn=swiglu(ops,l,ffn_input,w,a);
    Tensor<Ops> out;
    for(std::size_t i=0;i<x.size();++i) out.push_back(ops.add(residual[i],ffn[i]));
    return out;
}

// Encrypted embeddings in, encrypted hidden vectors out. Provider loads one
// layer's weights/profile at a time. Default ModelConfig has all 32 layers.
template<class Ops,class Provider>
Tensor<Ops> prefill(Ops &ops,const Layout &l,const ModelConfig &c,Tensor<Ops> input,
    std::size_t tokens,std::size_t first_position,Provider &provider) {
    c.validate(l);
    for(std::size_t layer=0;layer<c.layers;++layer) {
        auto weights=provider.load_layer(layer);
        auto approximation=provider.approximation(layer);
        input=decoder_block(ops,l,c,input,tokens,first_position,weights,approximation);
    }
    return input;
}

// Optional server-side final norm and full-vocabulary LM head. Embedding lookup,
// sampling and tokenizer are client-side; no incremental KV-cache entry yet.
template<class Ops>
Tensor<Ops> logits(Ops &ops,const Layout &l,const ModelConfig &c,const Tensor<Ops> &hidden,
    std::size_t tokens,const std::vector<double> &gamma,const Matrix &lm_head,
    const Polynomial &rsqrt_seed,unsigned steps) {
    lm_head.validate(c.vocabulary,c.hidden);
    auto normalized=rmsnorm(ops,l,c,hidden,tokens,gamma,rsqrt_seed,steps);
    return project(ops,l,normalized,lm_head);
}

} // namespace llama_manual
