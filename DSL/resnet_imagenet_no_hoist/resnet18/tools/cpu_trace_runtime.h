#pragma once
// Diagnostic CPU metadata backend. Never encrypts or evaluates ciphertexts.
// The unmodified native tensor/ReLU arithmetic drives this recorder.
#include "resnet50_config.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

namespace poseidon::benchmark::resnet50_gpu {
struct TraceComplete {};
struct TraceLimit {};
struct TraceMeta {std::size_t q_count=0; double scale=1;};
struct TraceValue {TraceMeta meta;std::uint64_t id=0;};
class GpuCkksRuntime {
public:
    using DeviceCiphertext=TraceValue;
    using DevicePlaintext=TraceValue;
    inline static std::ofstream output;
    inline static std::ofstream payload;
    inline static std::string stage_name="setup";
    inline static std::uint64_t next_id=0,inputs=0,boots=0;
    inline static std::uint64_t encrypted_constants=0;
    inline static std::uint64_t limit=0;
    ResNet50GpuConfig config;
    mutable TraceValue encrypted_one_cache;
    explicit GpuCkksRuntime(const ResNet50GpuConfig &c,int=0):config(c){config.validate();}
    static void stage(const std::string &name){stage_name=name;}
    static std::uint64_t bits(double x){return std::bit_cast<std::uint64_t>(x);}
    static std::uint64_t fingerprint(const std::vector<double> &v){
        std::uint64_t h=14695981039346656037ULL;
        for(double x:v){if(!std::isfinite(x))throw std::runtime_error("nonfinite payload");h^=bits(x);h*=1099511628211ULL;}
        return h;
    }
    static std::string record_payload(const std::vector<double> &v){
        static_assert(std::endian::native==std::endian::little);
        std::vector<unsigned char> compressed(compressBound(v.size()*8));
        uLongf compressed_size=compressed.size();
        if(compress2(compressed.data(),&compressed_size,
            reinterpret_cast<const unsigned char*>(v.data()),v.size()*8,Z_BEST_SPEED)!=Z_OK)
            throw std::runtime_error("lossless payload compression failed");
        const auto offset=std::uint64_t(payload.tellp());
        const std::uint64_t elements=v.size(),count=compressed_size;
        payload.write(reinterpret_cast<const char*>(&elements),8);
        payload.write(reinterpret_cast<const char*>(&count),8);
        payload.write(reinterpret_cast<const char*>(compressed.data()),count);
        if(!payload)throw std::runtime_error("compressed payload write failure");
        return ",\"payload_offset\":"+std::to_string(offset)+",\"payload_compressed_bytes\":"+std::to_string(count);
    }
    static TraceValue emit(const char *op,TraceMeta meta,const std::vector<TraceValue> &args={},
                           const std::string &extra="") {
        if(limit && next_id>=limit){output.flush();throw TraceLimit{};}
        if(!meta.q_count || !(meta.scale>0) || !std::isfinite(meta.scale))
            throw std::runtime_error(std::string("invalid trace metadata: ")+op+" stage="+stage_name);
        TraceValue value{meta,++next_id};
        output<<"{\"id\":"<<value.id<<",\"op\":\""<<op<<"\",\"stage\":\""<<stage_name
              <<"\",\"q\":"<<meta.q_count<<",\"scale_bits\":"<<bits(meta.scale)<<",\"inputs\":[";
        bool first=true;for(const auto &a:args){if(!first)output<<',';first=false;
            output<<"{\"id\":"<<a.id<<",\"q\":"<<a.meta.q_count<<",\"scale_bits\":"<<bits(a.meta.scale)<<'}';}
        output<<']'<<extra<<"}\n";
        if(!output)throw std::runtime_error("trace write failure");return value;
    }
    TraceValue encrypt(const std::vector<double> &v) const {
        if(v.size()>slot_count())throw std::runtime_error("input exceeds slots");++inputs;
        return emit("input",{config.application_q_count()-4,std::exp2(config.log_scale)}, {},
            ",\"elements\":"+std::to_string(v.size())+",\"fnv64_words\":"+std::to_string(fingerprint(v))+record_payload(v));
    }
    TraceValue encrypt_constant(double v) const {
        if(v==1.0 && encrypted_one_cache.id)return integer(encrypted_one_cache,1);
        ++encrypted_constants;
        auto value=emit("encrypted_constant",{config.application_q_count()-4,std::exp2(config.log_scale)}, {},
            ",\"value_bits\":"+std::to_string(bits(v)));
        if(v==1.0){encrypted_one_cache=value;return integer(encrypted_one_cache,1);}
        return value;
    }
    std::vector<std::complex<double>> decrypt(const TraceValue &) const {
        throw std::runtime_error("metadata-only tracer must not produce decoded numerical results");
    }
    static void finish(const std::vector<TraceValue> &values){
        output<<"{\"op\":\"outputs\",\"values\":[";bool first=true;
        for(const auto &v:values){if(!first)output<<',';first=false;output<<v.id;}
        output<<"],\"inputs\":"<<inputs<<",\"encrypted_constants\":"<<encrypted_constants<<",\"bootstrap_calls\":"<<boots
              <<",\"gpu_executed\":false,\"scope\":\"CPU metadata only; not semantic or encrypted acceptance\"}\n";
        output.flush();payload.flush();if(!output || !payload)throw std::runtime_error("trace flush failure");throw TraceComplete{};
    }
    TraceValue scale(const TraceValue &a,double s) const{return emit("set_scale",{a.meta.q_count,s},{a});}
    TraceValue integer(const TraceValue &a,std::uint64_t n) const {
        return emit("multiply_integer",a.meta,{a},",\"integer\":"+std::to_string(n));
    }
    TraceValue drop_to_q_count(const TraceValue &a,std::size_t q) const {
        if(!q || q>a.meta.q_count)throw std::runtime_error("invalid physical drop");
        return q==a.meta.q_count?integer(a,1):emit("drop",{q,a.meta.scale},{a});
    }
    TraceValue rescale(const TraceValue &a,std::uint32_t n=1) const {
        if(!n || n>=a.meta.q_count)throw std::runtime_error("invalid physical rescale");
        double s=a.meta.scale;const auto q=a.meta.q_count;
        if(n==2)s/=double(config.q_moduli.at(q-1)*config.q_moduli.at(q-2));
        else for(std::uint32_t i=0;i<n;++i)s/=double(config.q_moduli.at(q-1-i));
        return emit("rescale_many",{q-n,s},{a},",\"count\":"+std::to_string(n));
    }
    double last_modulus_value(const TraceValue &a) const {return double(config.q_moduli.at(a.meta.q_count-1));}
    TraceValue encode_and_upload_plain(const TraceValue &a,const std::vector<double> &v,double s) const {
        return emit("encode_vector",{a.meta.q_count,s},{},",\"elements\":"+std::to_string(v.size())+
            ",\"fnv64_words\":"+std::to_string(fingerprint(v))+record_payload(v));
    }
    std::vector<TraceValue> encode_and_upload_plain_batch(const TraceValue &a,const std::vector<std::vector<double>> &v,double s) const {
        std::vector<TraceValue> result;for(const auto &x:v)result.push_back(encode_and_upload_plain(a,x,s));return result;
    }
    TraceValue encode_scalar(const TraceValue &a,double v,double s) const {
        return emit("encode_scalar",{a.meta.q_count,s},{},",\"value_bits\":"+std::to_string(bits(v)));
    }
    TraceValue multiply_plain_preencoded(const TraceValue &a,const TraceValue &p) const {
        if(a.meta.q_count!=p.meta.q_count)throw std::runtime_error("plaintext Q mismatch");
        return emit("multiply_plain",{a.meta.q_count,a.meta.scale*p.meta.scale},{a,p});
    }
    TraceValue multiply_plain(const TraceValue &a,const std::vector<double> &v,double s) const {
        return multiply_plain_preencoded(a,encode_and_upload_plain(a,v,s));
    }
    TraceValue multiply_plain_scalar(const TraceValue &a,double v,double s) const {
        return multiply_plain_preencoded(a,encode_scalar(a,v,s));
    }
    TraceValue multiply_plain_rescale(const TraceValue &a,const std::vector<double> &v) const {
        return rescale(multiply_plain(a,v,last_modulus_value(a)),1);
    }
    TraceValue multiply_plain_rescale_cached(const TraceValue &a,const std::vector<double> &v,const std::string &) const {
        return multiply_plain_rescale(a,v);
    }
    TraceValue multiply_plain_rescale_cached_lazy(const TraceValue &a,const std::string &,const std::function<std::vector<double>()> &builder) const {
        return multiply_plain_rescale(a,builder());
    }
    TraceValue multiply_plain_scalar_rescale(const TraceValue &a,double v) const {
        const double s=last_modulus_value(a);
        if(std::abs(v)*s<0.5)return drop_to_q_count(sub(a,a),a.meta.q_count-1);
        const long double magnitude=std::abs(static_cast<long double>(v))*s;
        if(magnitude<=4294967295.0L){
            auto product=scale(integer(a,std::uint64_t(std::floor(magnitude+0.5L))),a.meta.scale*s);
            if(v<0)product=emit("negate",product.meta,{product});
            return rescale(product,1);
        }
        return rescale(multiply_plain_scalar(a,v,s),1);
    }
    static void align_check(const TraceValue &a,const TraceValue &b){
        if(a.meta.q_count!=b.meta.q_count || std::abs(std::log2(a.meta.scale/b.meta.scale))>1e-6)
            throw std::runtime_error("add/sub metadata mismatch at "+stage_name);
    }
    TraceValue add(const TraceValue &a,const TraceValue &b) const {align_check(a,b);return emit("add",a.meta,{a,b});}
    TraceValue sub(const TraceValue &a,const TraceValue &b) const {align_check(a,b);return emit("sub",a.meta,{a,b});}
    TraceValue add_aligned(const TraceValue &a,const TraceValue &b) const {
        auto q=std::min(a.meta.q_count,b.meta.q_count);
        return add(a.meta.q_count==q?a:drop_to_q_count(a,q),b.meta.q_count==q?b:drop_to_q_count(b,q));
    }
    TraceValue sub_aligned(const TraceValue &a,const TraceValue &b) const {
        auto q=std::min(a.meta.q_count,b.meta.q_count);
        return sub(a.meta.q_count==q?a:drop_to_q_count(a,q),b.meta.q_count==q?b:drop_to_q_count(b,q));
    }
    TraceValue add_plain(const TraceValue &a,const std::vector<double> &v) const {
        auto p=encode_and_upload_plain(a,v,a.meta.scale);return emit("add_plain",a.meta,{a,p});
    }
    TraceValue add_plain_scalar(const TraceValue &a,double v) const {
        auto p=encode_scalar(a,v,a.meta.scale);return emit("add_plain",a.meta,{a,p});
    }
    TraceValue rotate(const TraceValue &a,int step) const{return emit("rotate",a.meta,{a},",\"step\":"+std::to_string(step));}
    TraceValue rotate_composed(const TraceValue &a,long long step) const {
        auto remaining=step%static_cast<long long>(slot_count());if(remaining<0)remaining+=slot_count();
        if(!remaining)return drop_to_q_count(a,a.meta.q_count);
        TraceValue current=a;for(int bit=1;remaining;bit<<=1)if(remaining&bit){current=rotate(current,bit);remaining-=bit;}
        return current;
    }
    TraceValue multiply_relinearize(const TraceValue &a,const TraceValue &b) const {
        auto q=std::min(a.meta.q_count,b.meta.q_count);
        auto left=a.meta.q_count==q?a:drop_to_q_count(a,q),right=b.meta.q_count==q?b:drop_to_q_count(b,q);
        auto raw=emit("multiply_raw",{q,left.meta.scale*right.meta.scale},{left,right});
        return emit("relinearize",raw.meta,{raw});
    }
    TraceValue multiply_relinearize_rescale(const TraceValue &a,const TraceValue &b) const {
        auto reduced=rescale(multiply_relinearize(a,b),config.cipher_product_rescale_primes);
        const double target=a.meta.scale;
        if(std::abs(target/reduced.meta.scale-1)<=1e-6)return scale(reduced,target);
        auto corrected=rescale(multiply_plain_scalar(reduced,1,target*last_modulus_value(reduced)/reduced.meta.scale));
        return scale(corrected,target);
    }
    TraceValue square_relinearize_rescale(const TraceValue &a) const {return multiply_relinearize_rescale(a,a);}
    TraceValue bootstrap(const TraceValue &a) const {
        if(a.meta.q_count!=6 || std::abs(std::log2(a.meta.scale)-40)>1e-8)
            throw std::runtime_error("native bootstrap entry mismatch at "+stage_name);
        ++boots;return emit("bootstrap_native_api",{31,std::exp2(40)},{a});
    }
    TraceValue bootstrap_modraise(const TraceValue &) const {throw std::runtime_error("unexpected raw bootstrap variant");}
    void initialize_evaluation_keys(const std::vector<int> & = {}){}
    void initialize_inference_evaluation_keys(){}
    void initialize_resnet50_evaluation_keys(){}
    void initialize_bootstrap(){}
    bool evaluation_keys_ready() const noexcept{return false;}
    bool bootstrap_ready() const noexcept{return false;}
    void enable_full_device_cache(bool=true){throw std::runtime_error("use normal native inference, not timed cache replay");}
    void clear_plaintext_cache(){}
    void clear_ciphertext_cache(){}
    void synchronize() const{}
    int device_id() const noexcept{return -1;}
    std::size_t slot_count() const noexcept{return config.slot_count();}
};
}
