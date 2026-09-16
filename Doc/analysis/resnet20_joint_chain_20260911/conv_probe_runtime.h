#pragma once
// Adapter for the unchanged application Conv source. The host mode models
// decoded SIMD arithmetic, NOT CKKS noise or encryption. The isolated resident
// GPU path supplies per-level KeySwitch backends and batches sibling rotations
// through one hoisted decomposition; staged diagnostics may still supply the
// original evaluator/key pair directly.
#include "gpu_ckks_runtime.h" // guard original runtime before source-name substitution
#include <map>
#include <set>
#include <memory>

namespace poseidon::benchmark::resnet20_gpu::core {
class DiagnosticConvRuntime {
public:
    using InputEncryptor=std::function<CT(const std::vector<double> &)>;
    struct RotationBackend {
        GpuEvaluator *evaluator;
        const GpuGaloisKeysData *keys;
        GpuDoubleHoistWorkspace *workspace;
    };
    using RotationBackendResolver=std::function<RotationBackend(std::size_t)>;
    struct DeviceCiphertext {
        gpu::GpuCiphertextMeta meta;
        std::shared_ptr<CT> gpu;
        std::shared_ptr<Values> values;
    };
    const BaselineMetadata &m;
    GpuEvaluator *eval=nullptr;
    const GpuGaloisKeysData *keys=nullptr;
    InputEncryptor input_encryptor;
    RotationBackendResolver rotation_backend;
    mutable std::map<std::size_t,std::set<int>> rotations;
    mutable std::map<std::string,int> counts;
    enum class PreparedMode { uncached, capture, replay };
    struct CachedPlain {
        std::size_t q_count;
        double scale;
        std::unique_ptr<GpuPlaintextData> data;
    };
    mutable PreparedMode prepared_mode=PreparedMode::uncached;
    mutable std::vector<CachedPlain> prepared_plains;
    mutable std::vector<std::shared_ptr<CT>> prepared_inputs;
    mutable std::size_t prepared_plain_index=0,prepared_input_index=0;
    mutable std::unique_ptr<GpuPlaintextData> transient_plain;
    explicit DiagnosticConvRuntime(const BaselineMetadata &metadata,
        GpuEvaluator *e=nullptr,const GpuGaloisKeysData *k=nullptr,
        InputEncryptor encryptor={},RotationBackendResolver resolver={})
        :m(metadata),eval(e),keys(k),input_encryptor(std::move(encryptor)),
        rotation_backend(std::move(resolver)) {
        if (bool(e)!=bool(k)) throw std::runtime_error("evaluator/keys must be supplied together");
        if(input_encryptor && !eval)throw std::runtime_error("input encryptor requires GPU mode");
        if(rotation_backend && !eval)throw std::runtime_error("rotation resolver requires GPU mode");
    }
    void begin_prepared_capture() const {
        if(!eval || prepared_mode!=PreparedMode::uncached)
            throw std::logic_error("invalid prepared Conv capture state");
        prepared_plains.clear();prepared_inputs.clear();transient_plain.reset();
        prepared_plain_index=prepared_input_index=0;prepared_mode=PreparedMode::capture;
    }
    void finish_prepared_capture() const {
        if(prepared_mode!=PreparedMode::capture)
            throw std::logic_error("prepared Conv capture is not active");
        prepared_mode=PreparedMode::uncached;
    }
    void begin_prepared_replay() const {
        if(!eval || prepared_mode!=PreparedMode::uncached || prepared_plains.empty())
            throw std::logic_error("prepared Conv cache is unavailable");
        transient_plain.reset();prepared_plain_index=prepared_input_index=0;
        prepared_mode=PreparedMode::replay;
    }
    void finish_prepared_replay() const {
        if(prepared_mode!=PreparedMode::replay ||
            prepared_plain_index!=prepared_plains.size() ||
            prepared_input_index!=prepared_inputs.size())
            throw std::logic_error("prepared Conv replay differs from captured plan");
        prepared_mode=PreparedMode::uncached;
    }
    std::size_t prepared_plaintext_count() const {return prepared_plains.size();}
    std::size_t prepared_input_count() const {return prepared_inputs.size();}
    std::size_t slot_count() const {return 32768;}
    parms_id_type id(std::size_t q) const {return m.ctx.crt_context()->parms_id_map().at(q-1);}
    DeviceCiphertext host(const Values &v,std::size_t q=9,double scale=std::exp2(40)) const {
        if (eval || v.size()!=slot_count()) throw std::runtime_error("invalid plaintext probe input");
        DeviceCiphertext x;x.meta.q_count=q;x.meta.scale=scale;x.meta.parms_id=id(q);
        x.values=std::make_shared<Values>(v);return x;
    }
    DeviceCiphertext device(CT x) const {
        if (!eval) throw std::runtime_error("cannot inject GPU input into plaintext probe");
        DeviceCiphertext out;out.meta=x.meta;out.gpu=std::make_shared<CT>(std::move(x));return out;
    }
    void validate(const DeviceCiphertext &x) const {
        if (bool(x.gpu)!=bool(eval) || bool(x.values)==bool(eval) || !x.meta.q_count ||
            x.meta.parms_id!=id(x.meta.q_count) || !std::isfinite(x.meta.scale) || x.meta.scale<=0)
            throw std::runtime_error("invalid probe ciphertext/mode");
    }
    void aligned(const DeviceCiphertext &a,const DeviceCiphertext &b) const {
        validate(a);validate(b);
        if(a.meta.q_count!=b.meta.q_count || std::abs(std::log2(a.meta.scale/b.meta.scale))>1e-8)
            throw std::runtime_error("Conv operands not Q/scale aligned");
    }
    double modulus_value_from_end(const DeviceCiphertext &x,std::size_t offset) const {
        validate(x);if(offset>=x.meta.q_count)throw std::runtime_error("modulus offset out of range");
        return m.ctx.parameters_literal()->q().at(x.meta.q_count-1-offset).value();
    }
    double last_modulus_value(const DeviceCiphertext &x) const {return modulus_value_from_end(x,0);}
    DeviceCiphertext drop_to_q_count(const DeviceCiphertext &x,std::size_t q) const {
        validate(x);if(!q || q>x.meta.q_count)throw std::runtime_error("invalid modulus drop");
        ++counts["drop_or_clone"];
        if(eval){CT y;eval->drop_modulus(*x.gpu,y,id(q));return device(std::move(y));}
        return host(*x.values,q,x.meta.scale);
    }
    DeviceCiphertext add(const DeviceCiphertext &a,const DeviceCiphertext &b) const {
        aligned(a,b);++counts["add"];
        if(eval){CT y;eval->add(*a.gpu,*b.gpu,y);return device(std::move(y));}
        auto y=host(*a.values,a.meta.q_count,a.meta.scale);
        for(std::size_t i=0;i<slot_count();++i)(*y.values)[i]+=(*b.values)[i];return y;
    }
    DeviceCiphertext add_aligned(const DeviceCiphertext &a,const DeviceCiphertext &b) const {
        const auto q=std::min(a.meta.q_count,b.meta.q_count);
        return add(drop_to_q_count(a,q),drop_to_q_count(b,q));
    }
    const GpuPlaintextData &plain(const DeviceCiphertext &x,const std::vector<double> &v,double s) const {
        if(!eval || v.size()!=slot_count() || !(s>0) || !std::isfinite(s))
            throw std::runtime_error("invalid GPU plaintext");
        if(prepared_mode==PreparedMode::replay) {
            if(prepared_plain_index>=prepared_plains.size())
                throw std::logic_error("prepared plaintext replay exceeded captured plan");
            auto &cached=prepared_plains[prepared_plain_index++];
            if(cached.q_count!=x.meta.q_count || cached.scale!=s)
                throw std::logic_error("prepared plaintext metadata differs from captured plan");
            return *cached.data;
        }
        Plaintext p;m.encoder.encode(v,x.meta.parms_id,s,p);
        auto uploaded=std::make_unique<GpuPlaintextData>(GpuUploader::upload_plaintext(p,0));
        if(prepared_mode==PreparedMode::capture) {
            prepared_plains.push_back({x.meta.q_count,s,std::move(uploaded)});
            return *prepared_plains.back().data;
        }
        transient_plain=std::move(uploaded);return *transient_plain;
    }
    DeviceCiphertext multiply_plain(const DeviceCiphertext &x,const std::vector<double> &v,double s) const {
        validate(x);if(v.size()!=slot_count() || !(s>0) || !std::isfinite(s))throw std::runtime_error("invalid plaintext");
        ++counts["multiply_plain"];
        if(eval){CT y;eval->multiply_plain(*x.gpu,plain(x,v,s),y);return device(std::move(y));}
        auto y=host(*x.values,x.meta.q_count,x.meta.scale*s);
        for(std::size_t i=0;i<slot_count();++i)(*y.values)[i]*=v[i];return y;
    }
    void multiply_plain_accumulate(const DeviceCiphertext &x,const std::vector<double> &v,
        double s,DeviceCiphertext &destination) const {
        validate(x);validate(destination);
        if(x.meta.q_count!=destination.meta.q_count || v.size()!=slot_count() || !(s>0) ||
            !std::isfinite(s) || std::abs(std::log2(x.meta.scale*s/destination.meta.scale))>1e-8)
            throw std::runtime_error("invalid fused accumulation alignment");
        ++counts["multiply_plain_accumulate"];
        if(eval){
            if(!destination.gpu.unique())destination=drop_to_q_count(destination,destination.meta.q_count);
            eval->multiply_plain_accumulate(*x.gpu,plain(x,v,s),*destination.gpu);
            destination.meta=destination.gpu->meta;
        }else{
            if(!destination.values.unique())destination.values=std::make_shared<Values>(*destination.values);
            for(std::size_t i=0;i<slot_count();++i)(*destination.values)[i]+=(*x.values)[i]*static_cast<long double>(v[i]);
        }
    }
    DeviceCiphertext rescale(const DeviceCiphertext &x,std::size_t count) const {
        validate(x);if(!count || count>=x.meta.q_count)throw std::runtime_error("invalid rescale count");
        counts["rescale"]+=count;
        if(eval){
            CT y;eval->rescale(*x.gpu,y);
            for(std::size_t i=1;i<count;++i){CT next;eval->rescale(y,next);y=std::move(next);}
            return device(std::move(y));
        }
        double s=x.meta.scale;for(std::size_t i=0;i<count;++i)s/=modulus_value_from_end(x,i);
        return host(*x.values,x.meta.q_count-count,s);
    }
    DeviceCiphertext multiply_plain_rescale(const DeviceCiphertext &x,const std::vector<double> &v) const {
        return rescale(multiply_plain(x,v,last_modulus_value(x)),1);
    }
    DeviceCiphertext add_plain(const DeviceCiphertext &x,const std::vector<double> &v) const {
        validate(x);if(v.size()!=slot_count())throw std::runtime_error("invalid add_plain shape");
        ++counts["add_plain"];
        if(eval){CT y;eval->add_plain(*x.gpu,plain(x,v,x.meta.scale),y);return device(std::move(y));}
        auto y=host(*x.values,x.meta.q_count,x.meta.scale);
        for(std::size_t i=0;i<slot_count();++i)(*y.values)[i]+=v[i];return y;
    }
    DeviceCiphertext rotate_composed(const DeviceCiphertext &x,long long requested) const {
        validate(x);int step=(requested%32768+32768)%32768;
        if(!step)return drop_to_q_count(x,x.meta.q_count);
        rotations[x.meta.q_count].insert(step);++counts["rotate"];
        if(eval){
            CT y;
            if(rotation_backend) {
                auto backend=rotation_backend(x.meta.q_count);
                backend.evaluator->rotate(*x.gpu,step,*backend.keys,y);
            } else eval->rotate(*x.gpu,step,*keys,y);
            return device(std::move(y));
        }
        auto y=host(*x.values,x.meta.q_count,x.meta.scale);
        for(std::size_t i=0;i<slot_count();++i)(*y.values)[i]=(*x.values)[(i+step)%slot_count()];return y;
    }
    std::vector<DeviceCiphertext> rotate_many_composed(const DeviceCiphertext &x,
        const std::vector<long long> &steps) const {
        if(!eval || !rotation_backend) {
            std::vector<DeviceCiphertext> out;out.reserve(steps.size());
            for(auto step:steps)out.push_back(rotate_composed(x,step));
            return out;
        }
        validate(x);
        std::vector<DeviceCiphertext> result(steps.size());
        std::vector<int> direct_steps;
        std::vector<std::size_t> direct_indices;
        direct_steps.reserve(steps.size());direct_indices.reserve(steps.size());
        for(std::size_t index=0;index<steps.size();++index) {
            const int step=(steps[index]%32768+32768)%32768;
            if(!step) {
                result[index]=drop_to_q_count(x,x.meta.q_count);
                continue;
            }
            rotations[x.meta.q_count].insert(step);++counts["rotate"];
            direct_steps.push_back(step);direct_indices.push_back(index);
        }
        if(!direct_steps.empty()) {
            auto backend=rotation_backend(x.meta.q_count);
            if(!backend.evaluator || !backend.keys || !backend.workspace)
                throw std::logic_error("incomplete hoisted rotation backend");
            std::vector<CT> rotated;
            backend.evaluator->rotate_many_hoisted(*x.gpu,direct_steps,
                *backend.keys,*backend.workspace,rotated);
            if(rotated.size()!=direct_steps.size())
                throw std::logic_error("hoisted rotation output count mismatch");
            for(std::size_t index=0;index<rotated.size();++index)
                result[direct_indices[index]]=device(std::move(rotated[index]));
        }
        return result;
    }
    DeviceCiphertext encrypt(const std::vector<double> &slots) const {
        if(!input_encryptor && prepared_mode!=PreparedMode::replay)
            throw std::runtime_error("intermediate encryption forbidden in Conv probe");
        if(slots.size()!=slot_count())throw std::runtime_error("input encryption slot mismatch");
        ++counts["input_encrypt"];
        if(prepared_mode==PreparedMode::replay) {
            if(prepared_input_index>=prepared_inputs.size())
                throw std::logic_error("prepared input replay exceeded captured plan");
            DeviceCiphertext out;out.gpu=prepared_inputs[prepared_input_index++];out.meta=out.gpu->meta;
            return out;
        }
        auto uploaded=std::make_shared<CT>(input_encryptor(slots));
        if(prepared_mode==PreparedMode::capture)prepared_inputs.push_back(uploaded);
        DeviceCiphertext out;out.gpu=std::move(uploaded);out.meta=out.gpu->meta;return out;
    }
    std::vector<std::complex<double>> decrypt(const DeviceCiphertext &) const {
        throw std::runtime_error("decryption forbidden inside application Conv source");
    }
};
} // namespace

// Distinct types avoid an ODR collision with the independent original CHW TU.
// Only names/types are substituted; all Conv masks, rotations and rescale
// placement are compiled directly from the unchanged application source.
#define GpuCkksRuntime DiagnosticConvRuntime
#define GpuMultiplexedTensor DiagnosticConvTensor
#include "/home/liufuyao/Work/poseidon_gpu_other/resnet20-9.3/benchmark/resnet20_gpu/gpu_multiplexed_tensor.cpp"
#undef GpuMultiplexedTensor
#undef GpuCkksRuntime
