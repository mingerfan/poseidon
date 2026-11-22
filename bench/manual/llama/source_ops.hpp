// Backend adapter for the handwritten Cerium source-order port.
// Reference: FHE_GPU_Agent ResNet20, bootstrap_network_precision.cpp and
// relu_precision.cpp. No DSL compiler, plaintext execution or secret key here.
#pragma once
#include "poseidon/ckks_encoder.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/gpu/gpu_bootstrap_profile.h"
#include "llama_manual/periodic_plaintext.hpp"
#include "source_cerium_layout.hpp"
#include "source_pair_packing.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstring>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cerium_source {
using CT=poseidon::gpu::GpuCiphertextData;
// Legacy callers retain their environment consumer. The new manual program
// selects a constructor-frozen precision policy, never a live environment knob.
enum class ExpUnitSeedPolicy {LegacyEnvironment,IntegerPhysicalScale};
// LRU residency budget only: live callers may retain evicted plaintexts.
// The original 1GiB default is unchanged; 0 explicitly disables retention.
inline std::size_t real_plain_cache_budget_from_environment() {
    const char *raw=std::getenv("LLAMA_REAL_CACHE_MB");if(!raw)return 1ULL<<30;
    std::uint64_t mb=0;const char *end=raw+std::strlen(raw);const auto result=std::from_chars(raw,end,mb);
    if(raw==end||result.ec!=std::errc{}||result.ptr!=end||mb>8192||mb>(std::numeric_limits<std::size_t>::max()>>20))
        throw std::invalid_argument("LLAMA_REAL_CACHE_MB must be decimal MiB in [0,8192]");
    return static_cast<std::size_t>(mb)<<20;
}
class Api {
public:
    // Diagnostic side channels only. Callers enabling decrypting observers
    // must not treat their wall/events as performance; no plaintext feedback.
    std::function<void(const std::string&,const CT&)> stage_observer;
    std::function<void(const std::string&,const CT&,const CT&)> boot_observer;
    void observe(const char *point,const CT &x) const {if(stage_observer)stage_observer(point,x);}
    Api(const poseidon::PoseidonContext &context,
        const poseidon::gpu::GpuParameterData &parameters,
        const poseidon::CKKSEncoder &encoder,
        const poseidon::gpu::GpuRelinKeysData &relin,
        const poseidon::gpu::GpuGaloisKeysData &galois,
        std::size_t real_cache_budget=real_plain_cache_budget_from_environment(),
        ExpUnitSeedPolicy seed_policy=ExpUnitSeedPolicy::LegacyEnvironment)
        : context_(context),parameters_(parameters),encoder_(encoder),relin_(relin),
          galois_(galois),eval_(parameters),real_cache_budget_(real_cache_budget),seed_policy_(seed_policy),
          device_id_(infer_device_id(parameters_)) {
        if(seed_policy_!=ExpUnitSeedPolicy::LegacyEnvironment&&seed_policy_!=ExpUnitSeedPolicy::IntegerPhysicalScale)
            throw std::invalid_argument("invalid immutable exp unit seed policy");
        const auto literal=context_.parameters_literal();
        layout::require_ring(literal->log_n(),literal->log_slots());
        layout::require_encoder_slots(encoder_.slot_count());
        if(real_cache_budget_>(8192ULL<<20))throw std::invalid_argument("real plaintext cache budget exceeds 8192MiB");
    }
    int device_id() const noexcept{return device_id_;}
    ExpUnitSeedPolicy exp_unit_seed_policy() const noexcept{return seed_policy_;}
    const poseidon::gpu::GpuBootstrapProfile *bound_boot_profile() const noexcept{return profile_;}
    bool shares_application_bindings(const Api &other) const noexcept {
        return &context_==&other.context_ && &parameters_==&other.parameters_ && &encoder_==&other.encoder_ &&
               &relin_==&other.relin_ && &galois_==&other.galois_;
    }
    void record_exp_seed(bool integer) {
        auto &count=integer?integer_exp_seed_calls:standard_exp_seed_calls;
        if(count==UINT64_MAX)throw std::overflow_error("exp seed counter overflow");
        ++count;
    }
    std::size_t real_plain_cache_budget_bytes() const noexcept{return real_cache_budget_;}
    std::size_t real_plain_cache_resident_bytes() const noexcept{return real_cache_bytes_;}
    void bind_boot(const poseidon::gpu::GpuBootstrapProfile &profile,
                   poseidon::gpu::GpuBootstrapWorkspace &workspace) {
        profile_=&profile;workspace_=&workspace;
    }
    double prime(std::size_t count) const {
        if(!count||count>context_.parameters_literal()->q().size())
            throw std::invalid_argument("physical prime index");
        return static_cast<double>(context_.parameters_literal()->q()[count-1].value());
    }
    std::size_t level_p_count(std::size_t q_count) const {
        const auto &lvl=parameters_.get_level_by_q_count(q_count);
        const auto p=(!lvl.shards.empty())?lvl.shards.front().hybrid_base_p_count:0;
        return p?p:context_.parameters_literal()->p().size();
    }
    const poseidon::gpu::GpuEvaluator &eval() const noexcept{return eval_;}
    const poseidon::gpu::GpuGaloisKeysData &galois_keys() const noexcept{return galois_;}
    const poseidon::gpu::GpuParameterData &parameters() const noexcept{return parameters_;}
    poseidon::gpu::GpuDoubleHoistWorkspace &double_hoist_workspace() noexcept{return hoist_;}
    double removed(std::size_t q,std::size_t n) const {
        if(q<=n)throw std::invalid_argument("insufficient physical Q");
        long double result=1;
        for(std::size_t i=0;i<n;++i)result*=prime(q-i);
        return static_cast<double>(result);
    }
    CT drop(const CT &x,std::size_t q) const {
        if(q>x.meta.q_count||!q)throw std::invalid_argument("cannot raise Q by ModDrop");
        CT out;eval_.drop_modulus(x,out,parameters_.get_level_by_q_count(q).parms_id);return out;
    }
    CT copy(const CT &x) const {return drop(x,x.meta.q_count);}
    CT rescale(const CT &x,unsigned n) const {
        if(!n)return copy(x);
        (void)removed(x.meta.q_count,n);
        CT out;eval_.rescale_many(x,out,n);return out;
    }
    CT mul(const CT &a,const CT &b) const {
        CT out;eval_.multiply_q_prefix(a,b,out,parameters_.get_level_by_q_count(
            std::min(a.meta.q_count,b.meta.q_count)).parms_id);return out;
    }
    CT relin(const CT &x) const {
        if(x.meta.component_count==2)return copy(x);
        CT out;eval_.relinearize(x,relin_,out);return out;
    }
    static bool fused_relin_rs2_enabled() {
        const char *v=std::getenv("LLAMA_FUSED_RELIN_RS2");
        const char *rs=std::getenv("POSEIDON_RESCALE_X2");
        return v&&std::string(v)=="1"&&
            (!rs||(std::string(rs)!="0"&&std::string(rs)!="false"&&std::string(rs)!="FALSE"&&std::string(rs)!="off"&&std::string(rs)!="OFF"));
    }
    CT mul_r(const CT &a,const CT &b) const {return relin(mul(a,b));}
    CT relin_rescale(const CT &raw,unsigned n) const {
        if(n==2&&raw.meta.component_count==3&&fused_relin_rs2_enabled()) {
            CT out;eval_.relinearize_rescale_x2_hybrid(raw,relin_,out);return out;
        }
        return rescale(relin(raw),n);
    }
    CT mul_rescale(const CT &a,const CT &b,unsigned n) const {return relin_rescale(mul(a,b),n);}
    CT square(const CT &x,unsigned n) const {return mul_rescale(x,x,n);}

    /// Paired ciphertext helpers sharing relinearization key-switch passes when shapes match.
    std::array<CT,2> relin_pair(const CT &a,const CT &b) const {
        if(a.meta.component_count==2&&b.meta.component_count==2)return {copy(a),copy(b)};
        if(a.meta.component_count==3&&b.meta.component_count==3&&a.meta.parms_id==b.meta.parms_id) {
            std::array<CT,2> out;eval_.relinearize_pair(a,b,relin_,out[0],out[1]);return out;
        }
        return {relin(a),relin(b)};
    }
    std::array<CT,2> relin_rescale_pair(const CT &a,const CT &b,unsigned n) const {
        if(n==2&&fused_relin_rs2_enabled()&&
           a.meta.component_count==3&&b.meta.component_count==3&&a.meta.parms_id==b.meta.parms_id) {
            std::array<CT,2> out;eval_.relinearize_rescale_x2_hybrid_pair(a,b,relin_,out[0],out[1]);return out;
        }
        auto r=relin_pair(a,b);
        return rescale_pair(r[0],r[1],n);
    }
    std::array<CT,2> mul_pair(const CT &a0,const CT &b0,const CT &a1,const CT &b1) const {
        const auto q0=std::min(a0.meta.q_count,b0.meta.q_count);
        const auto q1=std::min(a1.meta.q_count,b1.meta.q_count);
        if(q0!=q1)return {mul(a0,b0),mul(a1,b1)};
        std::array<CT,2> out;
        if(a0.meta.parms_id==b0.meta.parms_id&&a0.meta.parms_id==a1.meta.parms_id&&
           a0.meta.parms_id==b1.meta.parms_id) {
            eval_.multiply_pair(a0,b0,a1,b1,out[0],out[1]);
        } else {
            eval_.multiply_q_prefix_pair(a0,b0,a1,b1,parameters_.get_level_by_q_count(q0).parms_id,out[0],out[1]);
        }
        return out;
    }
    std::array<CT,2> rescale_pair(const CT &a,const CT &b,unsigned n) const {
        if(n==0) return {copy(a),copy(b)};
        std::array<CT,2> out;
        eval_.rescale_many_pair(a,b,out[0],out[1],n);
        return out;
    }
    std::array<CT,2> sum_pair(const CT &a0,const CT &b0,const CT &a1,const CT &b1,bool subtract) const {
        if(!same_scale(a0.meta.scale,b0.meta.scale)||!same_scale(a1.meta.scale,b1.meta.scale))
            throw std::runtime_error("unaligned actual scales in source pair addition");
        const auto q0=std::min(a0.meta.q_count,b0.meta.q_count);
        const auto q1=std::min(a1.meta.q_count,b1.meta.q_count);
        CT l0_drop,r0_drop,l1_drop,r1_drop;
        const CT &l0=(a0.meta.q_count==q0)?a0:(l0_drop=drop(a0,q0));
        const CT &r0=(b0.meta.q_count==q0)?b0:(r0_drop=drop(b0,q0));
        const CT &l1=(a1.meta.q_count==q1)?a1:(l1_drop=drop(a1,q1));
        const CT &r1=(b1.meta.q_count==q1)?b1:(r1_drop=drop(b1,q1));
        std::array<CT,2> out;
        if(subtract) eval_.sub_pair(l0,r0,l1,r1,out[0],out[1]);
        else eval_.add_pair(l0,r0,l1,r1,out[0],out[1]);
        return out;
    }
    std::array<CT,2> add_pair(const CT &a0,const CT &b0,const CT &a1,const CT &b1) const {
        return sum_pair(a0,b0,a1,b1,false);
    }
    std::array<CT,2> sub_pair(const CT &a0,const CT &b0,const CT &a1,const CT &b1) const {
        return sum_pair(a0,b0,a1,b1,true);
    }
    std::array<CT,2> conjugate_pair(const CT &a,const CT &b) const {
        std::array<CT,2> out;
        eval_.conjugate_pair(a,b,galois_,out[0],out[1]);
        return out;
    }
    std::array<CT,2> add_conjugate_pair(const CT &a,const CT &b) const {
        auto c=conjugate_pair(a,b);
        return add_pair(a,c[0],b,c[1]);
    }
    static std::vector<const CT*> ptrs_of(const std::vector<CT> &xs) {
        std::vector<const CT*> out(xs.size());
        for(std::size_t i=0;i<xs.size();++i)out[i]=&xs[i];
        return out;
    }
    std::vector<CT> relin_batch(const std::vector<CT> &xs) const {
        if(xs.empty())return {};
        bool all2=true,all3=true,same_p=true;
        const auto ref=xs[0].meta.parms_id;
        for(const auto &x:xs) {
            if(x.meta.component_count!=2)all2=false;
            if(x.meta.component_count!=3)all3=false;
            if(x.meta.parms_id!=ref)same_p=false;
        }
        if(all3&&same_p&&xs.size()>=2) {
            std::vector<CT> out;eval_.relinearize_batch(ptrs_of(xs),relin_,out);return out;
        }
        std::vector<CT> out;out.reserve(xs.size());
        if(all2){for(const auto &x:xs)out.push_back(copy(x));return out;}
        for(const auto &x:xs)out.push_back(relin(x));
        return out;
    }
    std::vector<CT> rescale_batch(const std::vector<CT> &xs,unsigned n) const {
        if(xs.empty())return {};
        if(n==0) {
            std::vector<CT> out;out.reserve(xs.size());
            for(const auto &x:xs)out.push_back(copy(x));
            return out;
        }
        for(const auto &x:xs)(void)removed(x.meta.q_count,n);
        std::vector<CT> out;eval_.rescale_many_batch(ptrs_of(xs),out,n);return out;
    }
    std::vector<CT> relin_rescale_batch(const std::vector<CT> &xs,unsigned n) const {
        if(xs.empty())return {};
        if(n==2&&fused_relin_rs2_enabled()&&xs.size()>=2) {
            bool ok=true;const auto ref=xs[0].meta.parms_id;
            for(const auto &x:xs)if(x.meta.component_count!=3||x.meta.parms_id!=ref){ok=false;break;}
            if(ok){std::vector<CT> out;eval_.relinearize_rescale_x2_hybrid_batch(ptrs_of(xs),relin_,out);return out;}
        }
        return rescale_batch(relin_batch(xs),n);
    }
    std::vector<CT> mul_batch(const std::vector<const CT*> &as,const std::vector<const CT*> &bs) const {
        if(as.size()!=bs.size())throw std::invalid_argument("mul_batch size mismatch");
        std::vector<CT> out;eval_.multiply_batch(as,bs,out);return out;
    }
    std::vector<CT> square_batch(const std::vector<const CT*> &xs,unsigned n) const {
        std::vector<CT> sq;eval_.square_batch(xs,sq);
        return relin_rescale_batch(sq,n);
    }
    std::vector<CT> mul_rescale_batch(const std::vector<const CT*> &as,const std::vector<const CT*> &bs,unsigned n) const {
        return relin_rescale_batch(mul_batch(as,bs),n);
    }
    std::vector<CT> add_batch(const std::vector<const CT*> &as,const std::vector<const CT*> &bs) const {
        if(as.size()!=bs.size())throw std::invalid_argument("add_batch size mismatch");
        for(std::size_t i=0;i<as.size();++i)
            if(!same_scale(as[i]->meta.scale,bs[i]->meta.scale))
                throw std::runtime_error("unaligned actual scales in source batch addition");
        std::vector<CT> out;eval_.add_batch(as,bs,out);return out;
    }
    std::vector<CT> cp_batch(const std::vector<const CT*> &xs,double value,double scale) const {
        if(xs.empty())return {};
        const auto ref=xs[0]->meta.parms_id;
        for(const auto *x:xs)if(x->meta.parms_id!=ref) {
            std::vector<CT> out;out.reserve(xs.size());
            for(const auto *item:xs)out.push_back(cp(*item,value,scale));
            return out;
        }
        auto plain=scalar_plain(*xs[0],value,scale);
        std::vector<CT> out;eval_.multiply_plain_batch(xs,*plain,out);return out;
    }
    std::vector<CT> conjugate_batch(const std::vector<const CT*> &xs) const {
        std::vector<CT> out;eval_.conjugate_batch(xs,galois_,out);return out;
    }
    std::vector<CT> add_conjugate_batch(const std::vector<CT> &xs) const {
        auto xp=ptrs_of(xs);auto c=conjugate_batch(xp);
        return add_batch(xp,ptrs_of(c));
    }
    std::array<CT,2> mul_r_pair(const CT &a0,const CT &b0,const CT &a1,const CT &b1) const {
        auto m=mul_pair(a0,b0,a1,b1);
        return relin_pair(m[0],m[1]);
    }
    std::array<CT,2> mul_rescale_pair(const CT &a0,const CT &b0,const CT &a1,const CT &b1,unsigned n) const {
        auto m=mul_pair(a0,b0,a1,b1);
        return relin_rescale_pair(m[0],m[1],n);
    }
    std::array<CT,2> square_pair(const CT &x0,const CT &x1,unsigned n) const {
        std::array<CT,2> m;
        eval_.square_pair(x0,x1,m[0],m[1]);
        return relin_rescale_pair(m[0],m[1],n);
    }
    CT cp(const CT &x,double value,double scale) const {
        if(!std::isfinite(value)||!std::isfinite(scale)||scale<1)
            throw std::invalid_argument("invalid physical plaintext encoding scale");
        auto plain=scalar_plain(x,value,scale);
        CT out;eval_.multiply_plain(x,*plain,out);return out;
    }
    std::array<CT,2> cp_pair(const CT &x0,const CT &x1,double value,double scale) const {
        if(!std::isfinite(value)||!std::isfinite(scale)||scale<1)
            throw std::invalid_argument("invalid physical plaintext encoding scale");
        if(x0.meta.parms_id!=x1.meta.parms_id)return {cp(x0,value,scale),cp(x1,value,scale)};
        auto plain=scalar_plain(x0,value,scale);
        std::array<CT,2> out;eval_.multiply_plain_pair(x0,x1,*plain,out[0],out[1]);return out;
    }
    CT cp_to(const CT &x,double value,double output_scale,unsigned n) const {
        return rescale(cp(x,value,output_scale*removed(x.meta.q_count,n)/x.meta.scale),n);
    }
    std::array<CT,2> cp_to_pair(const CT &x0,const CT &x1,double value,double output_scale,unsigned n) const {
        if(x0.meta.parms_id!=x1.meta.parms_id||!same_scale(x0.meta.scale,x1.meta.scale))
            return {cp_to(x0,value,output_scale,n),cp_to(x1,value,output_scale,n)};
        auto p=cp_pair(x0,x1,value,output_scale*removed(x0.meta.q_count,n)/x0.meta.scale);
        if(n==0)return p;
        return rescale_pair(p[0],p[1],n);
    }
    CT cp_vector(const CT &x,const std::vector<double> &values,double scale) const {
        auto plain=cached_vector(x,values,scale);
        return cp_prepared(x,*plain);
    }
    std::array<CT,2> cp_vector_pair(const CT &x0,const CT &x1,const std::vector<double> &values,double scale) const {
        if(x0.meta.parms_id!=x1.meta.parms_id)return {cp_vector(x0,values,scale),cp_vector(x1,values,scale)};
        auto plain=cached_vector(x0,values,scale);
        return cp_prepared_pair(x0,x1,*plain);
    }
    CT cp_complex(const CT &x,const std::vector<std::complex<double>> &values,double scale) const {
        if(values.size()!=x.meta.degree/2||!std::isfinite(scale)||scale<1)
            throw std::invalid_argument("invalid complex mask contract");
        for(auto it=complex_cache_.begin();it!=complex_cache_.end();++it)
            if(it->id==x.meta.parms_id&&it->scale==scale&&it->values==values) {
                auto result=it->plain;complex_cache_.splice(complex_cache_.begin(),complex_cache_,it);
                ++plain_cache_hits;return cp_prepared(x,*result);
            }
        ++plain_cache_misses;
        poseidon::Plaintext host;encoder_.encode(values,x.meta.parms_id,scale,host);
        const int dev=x.fields_.empty()?device_id_:x.fields_.front().device_id;
        auto plain=std::make_shared<poseidon::gpu::GpuPlaintextData>(poseidon::gpu::GpuUploader::upload_plaintext(host,dev));
        const auto bytes=x.meta.degree*x.meta.q_count*sizeof(poseidon::gpu::GpuWord);
        constexpr std::size_t budget=128ULL<<20;
        while(!complex_cache_.empty()&&complex_cache_bytes_+bytes>budget) {
            complex_cache_bytes_-=complex_cache_.back().bytes;complex_cache_.pop_back();
        }
        if(bytes<=budget){complex_cache_.push_front({x.meta.parms_id,scale,values,plain,bytes});complex_cache_bytes_+=bytes;}
        return cp_prepared(x,*plain);
    }
    // Explicit candidate packing only: no paired refresh/model wiring yet.
    packing::PackedRealPair<CT> pack_independent_reals(const CT &a,const CT &b) {
        return packing::merge_for_pair_boot(*this,a,b,packing::InputConvention::BothLogicalReal);
    }
    packing::PackedRealPair<CT> merge_native_prephased_pair(const CT &a,const CT &ib) {
        return packing::merge_for_pair_boot(*this,a,ib,packing::InputConvention::SecondAlreadyImaginary);
    }
    void prepare_linear_babies(const CT &x,const std::vector<int> &steps,
                               poseidon::gpu::GpuDoubleHoistBabyCache &cache) {
        eval_.prepare_double_hoist_babies(x,steps,galois_,hoist_,cache);
    }
    CT native_dft(const CT &x,const poseidon::gpu::GpuLinearMatrixGroupQP &matrix,
                  const poseidon::gpu::GpuDoubleHoistBabyCache *cache=nullptr) {
        CT out;
        if(cache) {
            if(matrix.data().size()!=1||matrix.rescale_min_scale()>0.)
                throw std::invalid_argument("cached linear DFT requires one fixed-rescale matrix");
            eval_.multiply_by_diag_matrix_bsgs_double_hoist(x,matrix.data()[0],galois_,
                std::max(matrix.step(),std::uint32_t{1}),hoist_,out,cache);
        } else eval_.dft_double_hoist(x,matrix,galois_,hoist_,out);
        return out;
    }
    CT native_dft_sum(const CT &x,const poseidon::gpu::GpuLinearMatrixGroupQP &matrix,
                      poseidon::gpu::GpuDoubleHoistSumState &state,bool final_input,
                      const poseidon::gpu::GpuDoubleHoistBabyCache *cache=nullptr) {
        if(matrix.data().size()!=1||matrix.rescale_min_scale()>0.)
            throw std::invalid_argument("linear sum requires one fixed-rescale matrix");
        CT out;eval_.multiply_by_diag_matrix_bsgs_double_hoist_sum(x,matrix.data()[0],galois_,
            std::max(matrix.step(),std::uint32_t{1}),hoist_,state,final_input,out,cache);
        return out;
    }
    poseidon::gpu::GpuPlaintextData prepare_vector(const CT &x,const std::vector<double> &values,double scale) const {
        if(values.size()!=x.meta.degree/2||!std::isfinite(scale)||scale<1||
           !std::all_of(values.begin(),values.end(),[](double v){return std::isfinite(v);}))
            throw std::invalid_argument("invalid vector plaintext contract");
        // Cerium's PeriodicPlaintextInput stores a compact polynomial. The
        // ordinary ResNet20 CP API needs full Q NTT data, reconstructed exactly
        // using the period mapping checked in probe07 at Q8/Q18/Q30.
        bool periodic=x.meta.degree==65536&&scale>1;
        for(std::size_t j=256;periodic&&j<values.size();++j)periodic=values[j]==values[j%256];
        if(periodic) {
            if(!periodic_encoder_)periodic_encoder_=std::make_unique<llama_periodic::Encoder>(context_);
            std::array<double,256> pattern;std::copy_n(values.begin(),256,pattern.begin());
            try {
                const auto encoded=periodic_encoder_->encode(pattern,x.meta.parms_id,scale);
                const int dev=x.fields_.empty()?device_id_:x.fields_.front().device_id;
                auto plain=poseidon::gpu::GpuPlaintextData::allocate_single_device(65536,x.meta.q_count,dev);
                plain.meta.parms_id=x.meta.parms_id;plain.meta.scale=scale;plain.meta.is_ntt_form=true;
                const std::size_t compact_words=x.meta.q_count*llama_periodic::compact_degree;
                periodic_staging_.upload_async(compact_words,encoded.values.data(),dev);
                const auto &shard=plain.poly_.shards.front();
                poseidon::gpu::kernel::launch_expand_periodic_plaintext(
                    plain.fields_[shard.field_index].data()+shard.field_offset,
                    periodic_staging_.device.data(),
                    x.meta.q_count,65536,dev);
                ++compact_plain_preparations;
                return plain;
            }catch(const std::overflow_error &) {
                // Preserve full-precision encoding for unsupported int64
                // coefficients. Never rescale or approximate public values.
            }
        }
        poseidon::Plaintext host;encoder_.encode(values,x.meta.parms_id,scale,host);
        const int dev=x.fields_.empty()?device_id_:x.fields_.front().device_id;
        return poseidon::gpu::GpuUploader::upload_plaintext(host,dev);
    }
    poseidon::gpu::GpuPlaintextData prepare_complex(const CT &x,const std::vector<std::complex<double>> &values,double scale) const {
        if(values.size()!=x.meta.degree/2||!std::isfinite(scale)||scale<1)
            throw std::invalid_argument("invalid complex mask contract");
        poseidon::Plaintext host;encoder_.encode(values,x.meta.parms_id,scale,host);
        const int dev=x.fields_.empty()?device_id_:x.fields_.front().device_id;
        return poseidon::gpu::GpuUploader::upload_plaintext(host,dev);
    }
    std::vector<poseidon::gpu::GpuPlaintextData> prepare_complex_batch(
            const CT &x,const std::vector<std::vector<std::complex<double>>> &values_list,double scale) const {
        if(values_list.empty())return {};
        for(const auto &v:values_list)
            if(v.size()!=x.meta.degree/2||!std::isfinite(scale)||scale<1)
                throw std::invalid_argument("invalid complex mask contract");
        const std::size_t batch=values_list.size();
        const int dev=x.fields_.empty()?device_id_:x.fields_.front().device_id;
        const std::size_t q=x.meta.q_count,deg=x.meta.degree;
        const auto &level_info=parameters_.get_level(x.meta.parms_id);
        const poseidon::gpu::GpuParameterShard *pshard=nullptr;
        for(const auto &cand:level_info.shards) {
            if(cand.device_id==dev&&cand.limb_begin==0&&cand.limb_count>=q&&cand.ntt_tables.size()>=q*deg) {
                pshard=&cand;break;
            }
        }
        if(batch>=2&&deg==65536&&pshard!=nullptr&&batch<=65535) {
            const std::size_t poly_words=q*deg,total_words=batch*poly_words;
            periodic_staging_.prepare_host_and_device(total_words,dev);
            std::atomic<bool> ok{true};
            const int threads=static_cast<int>(std::min<std::size_t>(batch,16));
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(long long i=0;i<static_cast<long long>(batch);++i) {
                if(!ok.load(std::memory_order_relaxed))continue;
                if(!encoder_.encode_rns_coefficients_u32(
                        values_list[static_cast<std::size_t>(i)].data(),
                        values_list[static_cast<std::size_t>(i)].size(),
                        x.meta.parms_id,scale,
                        periodic_staging_.host+static_cast<std::size_t>(i)*poly_words,q)) {
                    ok.store(false,std::memory_order_relaxed);
                }
            }
            if(ok.load(std::memory_order_acquire)) {
                const cudaStream_t stream=poseidon::gpu::gpu_execution_stream();
                if(periodic_staging_.batch_ntt_dst.device_id()!=dev||periodic_staging_.batch_ntt_dst.size()<total_words)
                    periodic_staging_.batch_ntt_dst.allocate(total_words,dev);
                periodic_staging_.upload_host_range_async(total_words,dev,stream);
                poseidon::gpu::GpuPolyShardView dst_view{dev,periodic_staging_.batch_ntt_dst.data(),0,q,0,deg};
                poseidon::gpu::GpuConstPolyShardView src_view{dev,periodic_staging_.device.data(),0,q,0,deg};
                poseidon::gpu::kernel::launch_forward_ntt_poly_shard_batch_fourstep_65536(
                    dst_view,src_view,*pshard,deg,batch);
                std::vector<poseidon::gpu::GpuPlaintextData> out;out.reserve(batch);
                for(std::size_t i=0;i<batch;++i) {
                    auto plain=poseidon::gpu::GpuPlaintextData::allocate_single_device(deg,q,dev);
                    plain.meta.parms_id=x.meta.parms_id;plain.meta.scale=scale;plain.meta.is_ntt_form=true;
                    const auto &shard=plain.poly_.shards.front();
                    poseidon::gpu::gpu_check_cuda(
                        cudaMemcpyAsync(
                            plain.fields_[shard.field_index].data()+shard.field_offset,
                            periodic_staging_.batch_ntt_dst.data()+i*poly_words,
                            poly_words*sizeof(poseidon::gpu::GpuWord),
                            cudaMemcpyDeviceToDevice,stream),
                        "prepare_complex_batch d2d");
                    out.push_back(std::move(plain));
                }
                return out;
            }
        }
        std::vector<poseidon::gpu::GpuPlaintextData> out;out.reserve(batch);
        for(const auto &v:values_list)out.push_back(prepare_complex(x,v,scale));
        return out;
    }
    std::shared_ptr<poseidon::gpu::GpuPlaintextData> cached_vector(const CT &x,const std::vector<double> &values,double scale) const {
        return cached_vector_impl(x,values,scale,false);
    }
    std::shared_ptr<poseidon::gpu::GpuPlaintextData> cached_static_vector(const CT &x,const std::vector<double> &values,double scale) const {
        return cached_vector_impl(x,values,scale,true);
    }
    std::shared_ptr<poseidon::gpu::GpuPlaintextData> cached_vector_impl(const CT &x,const std::vector<double> &values,double scale,bool is_static) const {
        // Same public constant/context/scale identity as source cached input
        // bindings; bounded residency is an explicit native-backend adaptation.
        const std::uint64_t sh=is_static?static_entry_hash(x.meta.parms_id,scale,&values):0;
        if(is_static) {
            if(auto bucket=static_index_.find(sh);bucket!=static_index_.end()) {
                for(auto it:bucket->second) {
                    if(it->static_ptr==&values&&it->id==x.meta.parms_id&&it->scale==scale) {
                        auto result=it->plain;real_cache_.splice(real_cache_.begin(),real_cache_,it);
                        ++plain_cache_hits;return result;
                    }
                }
            }
        }
        const std::uint64_t h=real_entry_hash(x.meta.parms_id,scale,values);
        if(auto bucket=real_index_.find(h);bucket!=real_index_.end()) {
            for(auto it:bucket->second) {
                if(it->id==x.meta.parms_id&&it->scale==scale&&it->values==values) {
                    if(is_static&&!it->static_ptr){it->static_ptr=&values;it->static_hash=sh;static_index_[sh].push_back(it);}
                    auto result=it->plain;real_cache_.splice(real_cache_.begin(),real_cache_,it);
                    ++plain_cache_hits;return result;
                }
            }
        }
        ++plain_cache_misses;
        auto plain=std::make_shared<poseidon::gpu::GpuPlaintextData>(prepare_vector(x,values,scale));
        const auto bytes=x.meta.degree*x.meta.q_count*sizeof(poseidon::gpu::GpuWord);
        const std::size_t budget=real_cache_budget_;
        while(!real_cache_.empty()&&real_cache_bytes_+bytes>budget) {
            const auto &dead=real_cache_.back();
            auto erase_from=[&](auto &idx,std::uint64_t kh){
                if(auto bit=idx.find(kh);bit!=idx.end()) {
                    auto &vec=bit->second;
                    for(std::size_t i=0;i<vec.size();++i) {
                        if(&*vec[i]==&dead){vec[i]=vec.back();vec.pop_back();break;}
                    }
                    if(vec.empty())idx.erase(bit);
                }
            };
            erase_from(real_index_,dead.hash);
            if(dead.static_ptr)erase_from(static_index_,dead.static_hash);
            real_cache_bytes_-=dead.bytes;real_cache_.pop_back();
        }
        if(bytes<=budget) {
            real_cache_.push_front({x.meta.parms_id,scale,values,plain,bytes,h,is_static?&values:nullptr,sh});
            auto it=real_cache_.begin();
            real_index_[h].push_back(it);
            if(is_static)static_index_[sh].push_back(it);
            real_cache_bytes_+=bytes;
        }
        return plain;
    }
    CT cp_prepared(const CT &x,const poseidon::gpu::GpuPlaintextData &plain) const {
        CT out;eval_.multiply_plain(x,plain,out);return out;
    }
    std::array<CT,2> cp_prepared_pair(const CT &x0,const CT &x1,const poseidon::gpu::GpuPlaintextData &plain) const {
        std::array<CT,2> out;eval_.multiply_plain_pair(x0,x1,plain,out[0],out[1]);return out;
    }
    CT rotate(const CT &x,int step) const {
        if(!step)return copy(x);
        CT out;eval_.rotate(x,step,galois_,out);return out;
    }
    std::array<CT,2> rotate_pair(const CT &x0,const CT &x1,int step) const {
        if(!step)return {copy(x0),copy(x1)};
        std::array<CT,2> out;eval_.rotate_pair(x0,x1,step,galois_,out[0],out[1]);return out;
    }
    std::vector<CT> hoisted(const CT &x,const std::vector<int> &steps) {
        std::vector<int> active;
        for(int step:steps)if(step%32768)active.push_back(step);
        std::vector<CT> rotations,out;
        if(!active.empty())eval_.rotate_many_hoisted(x,active,galois_,hoist_,rotations);
        std::size_t i=0;
        for(int step:steps)out.push_back(step%32768?std::move(rotations.at(i++)):copy(x));
        return out;
    }
    std::vector<std::vector<CT>> hoisted_batch(const std::vector<const CT*> &xs,const std::vector<int> &steps) {
        std::vector<int> active;
        for(int step:steps)if(step%32768)active.push_back(step);
        std::vector<std::vector<CT>> rotations,out(xs.size());
        if(!active.empty()&&!xs.empty())eval_.rotate_many_hoisted_batch(xs,active,galois_,hoist_,rotations);
        for(std::size_t s=0;s<xs.size();++s) {
            std::size_t i=0;
            out[s].reserve(steps.size());
            for(int step:steps)out[s].push_back(step%32768?std::move(rotations.at(s).at(i++)):copy(*xs[s]));
        }
        return out;
    }
    std::array<std::vector<CT>,2> hoisted_pair(const CT &x0,const CT &x1,const std::vector<int> &steps) {
        auto res=hoisted_batch({&x0,&x1},steps);
        return {std::move(res[0]),std::move(res[1])};
    }
    CT add_plain(const CT &x,double value) const {
        auto plain=scalar_plain(x,value,x.meta.scale);
        CT out;eval_.add_plain(x,*plain,out);return out;
    }
    CT add(const CT &a,const CT &b) const {return sum(a,b,false);}
    CT sub(const CT &a,const CT &b) const {return sum(a,b,true);}
    CT conjugate(const CT &x) const {
        CT out;eval_.conjugate(x,galois_,out);return out;
    }
    CT source_bootstrap(const CT &x,const std::string &source_point) {
        if(!profile_||!workspace_)throw std::runtime_error("native ResNet20 profile not bound");
        const auto &data=profile_->bootstrap_data;
        if(!data.slot_to_coeff_first||x.meta.q_count<data.input_q_count||
           !same_scale(x.meta.scale,data.input_scale))
            throw std::runtime_error("native ResNet20 bootstrap Q/scale mismatch at "+source_point);
        auto input=drop(x,data.input_q_count);CT out;
        eval_.bootstrap(input,data,*profile_->relin_keys,*profile_->galois_keys,*workspace_,out);
        ++boot_calls;
        if(boot_observer)boot_observer(source_point,input,out);
        std::cout<<"SOURCE_BOOT point="<<source_point<<" input_q="<<input.meta.q_count
            <<" output_q="<<out.meta.q_count<<" log_scale="<<std::log2(out.meta.scale)
            <<" calls="<<boot_calls<<" corrections=0"<<std::endl;
        return out;
    }
    // Candidate component projections, not a qualified two-message refresh.
    // Preserve this adapter's existing explicit drop-to-Boot boundary; packing
    // itself never drops Q. Legacy single-output observers are not called.
    poseidon::gpu::GpuEvaluator::ManualBootstrapProjections
    source_bootstrap_projections_candidate(const packing::PackedRealPair<CT> &packed) {
        if(!profile_||!workspace_||!profile_->relin_keys||!profile_->galois_keys)
            throw std::runtime_error("manual projection profile/owning keys not bound");
        const auto &data=profile_->bootstrap_data;
        const auto &carrier=packed.carrier();
        packing::require_single_device_real_pair_operand(carrier,device_id_);
        if(!data.slot_to_coeff_first||!data.project_real||carrier.meta.q_count<data.input_q_count||
           !same_scale(carrier.meta.scale,data.input_scale))
            throw std::runtime_error("candidate manual projection Q/scale contract");
        auto input=drop(carrier,data.input_q_count);
        auto result=eval_.bootstrap_manual_projections(input,data,*profile_->relin_keys,*profile_->galois_keys,*workspace_);
        ++boot_calls;++manual_projection_pairs;
        std::cout<<"SOURCE_BOOT_PROJECTIONS_CANDIDATE physical_boot_calls="<<boot_calls
                 <<" pairs="<<manual_projection_pairs<<" component_outputs=2 semantic_qualification=false"
                 <<" output_q="<<result.real.meta.q_count<<" log_scale="<<std::log2(result.real.meta.scale)<<std::endl;
        return result;
    }
    static bool same_scale(double a,double b) {
        return std::isfinite(a)&&std::isfinite(b)&&a>0&&b>0&&
            std::abs(a-b)<=1e-10*std::max(a,b);
    }
    void require_scale(const CT &x,double scale) const {
        if(!same_scale(x.meta.scale,scale))throw std::runtime_error("physical scale recurrence mismatch");
    }
    std::uint64_t boot_calls=0,manual_projection_pairs=0;
    std::uint64_t integer_exp_seed_calls=0,standard_exp_seed_calls=0;
    mutable std::uint64_t plain_cache_hits=0,plain_cache_misses=0,compact_plain_preparations=0;
    mutable std::uint64_t scalar_cache_hits=0,scalar_cache_misses=0;
private:
    struct ScalarEntry {
        poseidon::parms_id_type id;double scale,value;
        std::shared_ptr<poseidon::gpu::GpuPlaintextData> plain;std::size_t bytes;
    };
    mutable std::list<ScalarEntry> scalar_cache_;
    mutable std::size_t scalar_cache_bytes_=0;
    std::shared_ptr<poseidon::gpu::GpuPlaintextData> scalar_plain(const CT &x,double value,double scale) const {
        const char *v=std::getenv("LLAMA_SCALAR_CACHE");const bool enabled=v&&std::string(v)=="1";
        if(enabled)for(auto it=scalar_cache_.begin();it!=scalar_cache_.end();++it)
            if(it->id==x.meta.parms_id&&it->scale==scale&&it->value==value) {
                auto out=it->plain;scalar_cache_.splice(scalar_cache_.begin(),scalar_cache_,it);++scalar_cache_hits;return out;
            }
        ++scalar_cache_misses;
        poseidon::Plaintext host;encoder_.encode(value,x.meta.parms_id,scale,host);
        const int dev=x.fields_.empty()?device_id_:x.fields_.front().device_id;
        auto plain=std::make_shared<poseidon::gpu::GpuPlaintextData>(poseidon::gpu::GpuUploader::upload_plaintext(host,dev));
        if(enabled) {
            const auto bytes=x.meta.degree*x.meta.q_count*sizeof(poseidon::gpu::GpuWord);
            std::size_t budget=256ULL<<20;
            if(const char *raw=std::getenv("LLAMA_SCALAR_CACHE_MB"))budget=std::stoull(raw)<<20;
            while(!scalar_cache_.empty()&&scalar_cache_bytes_+bytes>budget){scalar_cache_bytes_-=scalar_cache_.back().bytes;scalar_cache_.pop_back();}
            if(bytes<=budget){scalar_cache_.push_front({x.meta.parms_id,scale,value,plain,bytes});scalar_cache_bytes_+=bytes;}
        }
        return plain;
    }
    static std::uint64_t mix64(std::uint64_t x) noexcept {
        x+=0x9e3779b97f4a7c15ULL;
        x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
        x=(x^(x>>27))*0x94d049bb133111ebULL;
        return x^(x>>31);
    }
    static std::uint64_t real_entry_hash(const poseidon::parms_id_type &id,double scale,const std::vector<double> &values) noexcept {
        std::uint64_t h=mix64(values.size());
        for(auto w:id)h^=mix64(w+(h<<1));
        std::uint64_t sb=0;std::memcpy(&sb,&scale,sizeof(sb));
        if((sb<<1)==0)sb=0;
        h=mix64(h^sb);
        std::uint64_t v0=0,v1=0;
        for(std::size_t i=0;i<values.size();++i) {
            std::uint64_t bits=0;std::memcpy(&bits,&values[i],sizeof(bits));
            if((bits<<1)==0)bits=0;
            if(i&1)v1=v1*0x9e3779b97f4a7c15ULL+bits;
            else v0=v0*0xbf58476d1ce4e5b9ULL+bits;
        }
        return mix64(h^v0^mix64(v1));
    }
    static std::uint64_t static_entry_hash(const poseidon::parms_id_type &id,double scale,const void *ptr) noexcept {
        std::uint64_t h=mix64(reinterpret_cast<std::uintptr_t>(ptr));
        for(auto w:id)h^=mix64(w+(h<<1));
        std::uint64_t sb=0;std::memcpy(&sb,&scale,sizeof(sb));
        if((sb<<1)==0)sb=0;
        return mix64(h^sb);
    }
    struct PeriodicStaging {
        poseidon::gpu::GpuWord *host=nullptr;
        std::size_t cap=0;
        cudaEvent_t copy_done=nullptr;
        poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> device,batch_ntt_dst;
        ~PeriodicStaging() {
            if(copy_done)cudaEventDestroy(copy_done);
            if(host)cudaFreeHost(host);
        }
        void prepare_host_and_device(std::size_t words,int dev) {
            if(words>cap) {
                if(copy_done){cudaEventSynchronize(copy_done);}
                if(host){cudaFreeHost(host);host=nullptr;cap=0;}
                poseidon::gpu::gpu_check_cuda(cudaHostAlloc(&host,words*sizeof(poseidon::gpu::GpuWord),cudaHostAllocDefault),"periodic pinned alloc");
                cap=words;
            } else if(copy_done) {
                poseidon::gpu::gpu_check_cuda(cudaEventSynchronize(copy_done),"periodic host wait");
            }
            if(device.device_id()!=dev||device.size()<words)device.allocate(words,dev);
        }
        void upload_host_range_async(std::size_t words,int dev,cudaStream_t stream) {
            poseidon::gpu::gpu_check_cuda(cudaSetDevice(dev),"periodic set device");
            if(!copy_done)poseidon::gpu::gpu_check_cuda(cudaEventCreateWithFlags(&copy_done,cudaEventDisableTiming),"periodic ev");
            poseidon::gpu::gpu_check_cuda(cudaMemcpyAsync(device.data(),host,words*sizeof(poseidon::gpu::GpuWord),cudaMemcpyHostToDevice,stream),"periodic async h2d");
            poseidon::gpu::gpu_check_cuda(cudaEventRecord(copy_done,stream),"periodic rec");
        }
        void upload_async(std::size_t words,const poseidon::gpu::GpuWord *src,int dev) {
            prepare_host_and_device(words,dev);
            std::memcpy(host,src,words*sizeof(poseidon::gpu::GpuWord));
            upload_host_range_async(words,dev,poseidon::gpu::gpu_execution_stream());
        }
    };
    template<class Value> struct PlainEntry {
        poseidon::parms_id_type id;double scale;std::vector<Value> values;
        std::shared_ptr<poseidon::gpu::GpuPlaintextData> plain;std::size_t bytes;std::uint64_t hash=0;
        const void *static_ptr=nullptr;std::uint64_t static_hash=0;
    };
    mutable std::list<PlainEntry<double>> real_cache_;
    mutable std::unordered_map<std::uint64_t,std::vector<std::list<PlainEntry<double>>::iterator>> real_index_;
    mutable std::unordered_map<std::uint64_t,std::vector<std::list<PlainEntry<double>>::iterator>> static_index_;
    mutable std::list<PlainEntry<std::complex<double>>> complex_cache_;
    mutable std::size_t real_cache_bytes_=0,complex_cache_bytes_=0;
    mutable std::unique_ptr<llama_periodic::Encoder> periodic_encoder_;
    mutable PeriodicStaging periodic_staging_;
    CT sum(const CT &a,const CT &b,bool subtract) const {
        if(!same_scale(a.meta.scale,b.meta.scale))
            throw std::runtime_error("unaligned actual scales in source addition");
        const auto q=std::min(a.meta.q_count,b.meta.q_count);
        if(const char *v=std::getenv("LLAMA_ADD_PREFIX");v&&std::string(v)=="1") {
            CT out;eval_.add_sub_q_prefix(a,b,out,parameters_.get_level_by_q_count(q).parms_id,subtract);return out;
        }
        auto left=drop(a,q),right=drop(b,q);CT out;
        if(subtract)eval_.sub(left,right,out);else eval_.add(left,right,out);
        return out;
    }
    static int infer_device_id(const poseidon::gpu::GpuParameterData &parameters) {
        try {
            const auto &lvl=parameters.get_first_q_level();
            if(!lvl.shards.empty())return lvl.shards.front().device_id;
        }catch(...){}
        return 0;
    }
    const poseidon::PoseidonContext &context_;
    const poseidon::gpu::GpuParameterData &parameters_;
    const poseidon::CKKSEncoder &encoder_;
    const poseidon::gpu::GpuRelinKeysData &relin_;
    const poseidon::gpu::GpuGaloisKeysData &galois_;
    poseidon::gpu::GpuEvaluator eval_;
    const std::size_t real_cache_budget_;
    const ExpUnitSeedPolicy seed_policy_;
    const int device_id_=0;
    poseidon::gpu::GpuDoubleHoistWorkspace hoist_;
    const poseidon::gpu::GpuBootstrapProfile *profile_=nullptr;
    poseidon::gpu::GpuBootstrapWorkspace *workspace_=nullptr;
};
} // namespace cerium_source
