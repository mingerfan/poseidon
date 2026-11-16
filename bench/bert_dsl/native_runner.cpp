// Standard RuntimePlan runner. No secret-key refresh and no numerical pass/fail gate.
#include "poseidon/basics/modulus.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_bootstrap_profile.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "poseidon/runtime_api/rotation_key_basis.h"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/runtime.hpp"
#include "runtime/utils/sha256.hpp"
#ifdef BERT_COMPACT_LINEAR
#include "compact_native/compact_linear.h"
#endif
#ifdef BERT_PREPARED_QK
#include "compact_attention/prepared_qk.h"
#endif
#ifdef BERT_PREPARED_AV
#include "compact_av/prepared_av.h"
#endif
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <sys/resource.h>

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using BaseApi = poseidon::runtime_api::PoseidonGpuApi;
using Value = poseidon::runtime_api::PoseidonGpuValue;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); }
int strict_int_argument(const std::string &value,const std::string &name) {
    int result=0;
    const auto parsed=std::from_chars(value.data(),value.data()+value.size(),result);
    if(parsed.ec!=std::errc{} || parsed.ptr!=value.data()+value.size())
        throw std::runtime_error("invalid integer for "+name);
    return result;
}
void cuda_check(cudaError_t e, const char *op) {
    if(e != cudaSuccess) throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(e));
}
// Optional one-run CUDA-profiler capture. It never changes the HE plan; these
// samples are explicitly diagnostic, not uninstrumented benchmark results.
class ProfilerRange {
    bool active_=false;
public:
    explicit ProfilerRange(bool enabled) {
        if(enabled){cuda_check(cudaProfilerStart(),"cudaProfilerStart");active_=true;}
    }
    void stop() {
        if(active_){auto status=cudaProfilerStop();active_=false;cuda_check(status,"cudaProfilerStop");}
    }
    ~ProfilerRange(){if(active_)(void)cudaProfilerStop();}
    ProfilerRange(const ProfilerRange&)=delete;
    ProfilerRange& operator=(const ProfilerRange&)=delete;
};
// Optional bench-owned allocator. Installed before every GPU API/profile and
// kept outside their scopes, including error unwinding. No classic RMM pool is
// layered on top: BaseApi recognizes this non-cuda_memory_resource as external.
class AsyncAllocator {
    using Statistics=rmm::mr::statistics_resource_adaptor<rmm::mr::cuda_async_memory_resource>;
    int device_=0;
    rmm::mr::device_memory_resource *previous_=nullptr;
    std::size_t initial_free_=0,total_=0;
    std::uint64_t release_threshold_=0;
    std::unique_ptr<rmm::mr::cuda_async_memory_resource> upstream_;
    std::unique_ptr<Statistics> statistics_;
public:
    explicit AsyncAllocator(int device):device_(device) {
        cuda_check(cudaSetDevice(device_),"async allocator cudaSetDevice");
        cuda_check(cudaMemGetInfo(&initial_free_,&total_),"async allocator cudaMemGetInfo");
        release_threshold_=total_; // Retention policy only, NEVER a capacity/safety cap.
        upstream_=std::make_unique<rmm::mr::cuda_async_memory_resource>(std::size_t{0},release_threshold_);
        statistics_=std::make_unique<Statistics>(upstream_.get());
        previous_=rmm::mr::set_current_device_resource(statistics_.get());
    }
    ~AsyncAllocator() noexcept {
        // All Runtime/API/profile/value owners have already gone out of scope.
        // Settle queued cudaFreeAsync work before destroying the pool itself.
        if(cudaSetDevice(device_)==cudaSuccess) {
            (void)cudaDeviceSynchronize();
            rmm::mr::set_current_device_resource(previous_);
        }
    }
    AsyncAllocator(const AsyncAllocator&)=delete;
    AsyncAllocator& operator=(const AsyncAllocator&)=delete;
    Json snapshot(const char *scope) const {
        cuda_check(cudaSetDevice(device_),"async allocator snapshot cudaSetDevice");
        const auto bytes=statistics_->get_bytes_counter(),allocations=statistics_->get_allocations_counter();
        Json j{{"kind","rmm_cuda_async_with_statistics"},{"scope",scope},{"logical_device",device_},
            {"initial_pool_size_bytes",0},{"initial_cuda_free_bytes",initial_free_},{"device_total_bytes",total_},
            {"release_threshold_bytes",release_threshold_},{"release_threshold_is_capacity_cap",false},
            {"allocated_live_bytes",bytes.value},{"allocated_peak_bytes",bytes.peak},{"allocated_total_bytes",bytes.total},
            {"allocations_live",allocations.value},{"allocations_peak",allocations.peak},{"allocations_total",allocations.total},
            {"statistics_scope","cumulative since allocator installation, including profile/keys/setup/warmup/measured runs; RMM counter overhead is included in timings"}};
        const auto attribute=[&](const char *name,cudaMemPoolAttr attr) {
            std::uint64_t value=0;auto status=cudaMemPoolGetAttribute(upstream_->pool_handle(),attr,&value);
            if(status==cudaSuccess) j[name]=value;
            else {j[name]=nullptr;j["cuda_pool_attribute_errors"][name]=cudaGetErrorString(status);}
        };
        attribute("cuda_pool_used_current_bytes",cudaMemPoolAttrUsedMemCurrent);
        attribute("cuda_pool_used_high_bytes",cudaMemPoolAttrUsedMemHigh);
        attribute("cuda_pool_reserved_current_bytes",cudaMemPoolAttrReservedMemCurrent);
        attribute("cuda_pool_reserved_high_bytes",cudaMemPoolAttrReservedMemHigh);
        return j;
    }
};
Json read_json(const std::filesystem::path &p, std::string *digest=nullptr) {
    std::ifstream f(p); if(!f) throw std::runtime_error("cannot open "+p.string());
    std::ostringstream s; s<<f.rdbuf(); if(f.bad()) throw std::runtime_error("read failed "+p.string());
    if(digest) *digest="sha256:"+fhegpu::sha256_hex(s.str());
    return Json::parse(s.str());
}
bool prepared_qk_requires_pack_inputs(const Json &fixture) {
    bool required=false;
    if(fixture.contains("prepared_qk_required_capabilities")) {
        const auto &capabilities=fixture.at("prepared_qk_required_capabilities");
        if(!capabilities.is_array())throw std::runtime_error("prepared_qk_required_capabilities must be an array");
        for(const auto &capability:capabilities) {
            if(!capability.is_string() || capability.get<std::string>()!="pack_inputs")
                throw std::runtime_error("unsupported prepared QK required capability");
            required=true;
        }
    }
    if(fixture.contains("model")) {
        const auto &model=fixture.at("model");
        if(!model.is_object())throw std::runtime_error("fixture model must be an object");
        if(model.contains("qk_pack_inputs")) {
            if(!model.at("qk_pack_inputs").is_boolean())
                throw std::runtime_error("model.qk_pack_inputs must be a JSON boolean");
            required=required || model.at("qk_pack_inputs").get<bool>();
        }
    }
    if(required && !fixture.contains("prepared_qk_manifest"))
        throw std::runtime_error("required pack_inputs capability lacks prepared QK manifest");
    return required;
}
bool prepared_av_requires_packed_partial(const Json &fixture) {
    bool required=false;
    if(fixture.contains("prepared_av_required_capabilities")) {
        const auto &capabilities=fixture.at("prepared_av_required_capabilities");
        if(!capabilities.is_array())throw std::runtime_error("prepared_av_required_capabilities must be an array");
        for(const auto &capability:capabilities) {
            if(!capability.is_string() || capability.get<std::string>()!="packed_input2_partial")
                throw std::runtime_error("unsupported prepared AV required capability");
            required=true;
        }
    }
    if(fixture.contains("model")) {
        const auto &model=fixture.at("model");
        if(!model.is_object())throw std::runtime_error("fixture model must be an object");
        if(model.contains("av_pack_inputs")) {
            if(!model.at("av_pack_inputs").is_boolean())
                throw std::runtime_error("model.av_pack_inputs must be a JSON boolean");
            required=required || model.at("av_pack_inputs").get<bool>();
        }
    }
    if(required && !fixture.contains("prepared_av_manifest"))
        throw std::runtime_error("required packed_input2_partial capability lacks prepared AV manifest");
    return required;
}
void save_json(const std::filesystem::path &p, const Json &j) {
    if(!p.parent_path().empty()) std::filesystem::create_directories(p.parent_path());
    std::ofstream f(p); f<<j.dump(2)<<'\n'; if(!f) throw std::runtime_error("cannot write "+p.string());
}
std::uint32_t log2_exact(std::uint64_t n) {
    if(n<2 || (n&(n-1))) throw std::runtime_error("poly_degree must be a power of two");
    std::uint32_t k=0; while(n>1){++k;n>>=1;} return k;
}
poseidon::PoseidonContext make_context(const Json &c, const fhegpu::OperatorSpec *spec=nullptr) {
    auto n=c.at("poly_degree").get<std::uint64_t>();
    auto logn=log2_exact(n);
    if(n>std::numeric_limits<std::uint64_t>::max()/2)
        throw std::runtime_error("poly_degree is too large");
    auto q=c.at("q_moduli_bits").get<std::vector<int>>();
    auto p=c.at("p_moduli_bits").get<std::vector<int>>();
    auto scale=c.at("default_scale_log2").get<int>();
    if(spec && (n!=spec->poly_degree || q!=spec->rns_moduli_log2 || scale!=spec->default_scale_log2))
        throw std::runtime_error("fixture degree, Q chain or default scale disagrees with OperatorSpec");
    if(q.empty() || p.empty() || scale<=0 || scale>=63) throw std::runtime_error("invalid fixture Q/P/default scale");
    auto bits=q; bits.insert(bits.end(),p.begin(),p.end());
    if(std::any_of(bits.begin(),bits.end(),[](int b){return b<2||b>32;}))
        throw std::runtime_error("GPU CKKS Q/P primes must fit 32 bits");
    std::vector<poseidon::Modulus> qs,ps;
    if(c.contains("q_moduli") || c.contains("p_moduli")) {
        if(!c.contains("q_moduli") || !c.contains("p_moduli"))
            throw std::runtime_error("explicit moduli require both Q and P");
        auto q_values=c.at("q_moduli").get<std::vector<std::uint64_t>>();
        auto p_values=c.at("p_moduli").get<std::vector<std::uint64_t>>();
        if(q_values.size()!=q.size() || p_values.size()!=p.size())
            throw std::runtime_error("explicit Q/P length disagrees with bit widths");
        std::set<std::uint64_t> distinct;
        const auto append=[&](const auto &values,const auto &widths,auto &out) {
            for(std::size_t i=0;i<values.size();++i) {
                const auto prime=values[i];
                if(prime<(std::uint64_t{1}<<(widths[i]-1)) || prime>=(std::uint64_t{1}<<widths[i]) ||
                   (prime-1)%(2*n) || !distinct.insert(prime).second)
                    throw std::runtime_error("invalid explicit NTT modulus, bit width or duplicate");
                out.emplace_back(prime);
            }
        };
        append(q_values,q,qs);append(p_values,p,ps);
        if(spec && !spec->rns_moduli.empty() && q_values!=spec->rns_moduli)
            throw std::runtime_error("explicit Q primes disagree with OperatorSpec");
    } else {
        auto primes=poseidon::CoeffModulus::Create(n,bits);
        qs.assign(primes.begin(),primes.begin()+q.size());ps.assign(primes.begin()+q.size(),primes.end());
    }
    auto hamming=c.value("secret_hamming_weight",std::uint32_t{0});
    auto q0=c.value("q0_level",std::uint32_t{1});
    if(hamming>n || q0>=q.size()-1)throw std::runtime_error("invalid secret Hamming weight or ground Q level");
    poseidon::ParametersLiteral literal(CKKS,logn,logn-1,scale,hamming,q0,poseidon::Modulus(0),qs,ps,poseidon::sec_level_type::none);
    return poseidon::PoseidonContext(literal);
}
void context_report(Json &report, const poseidon::PoseidonContext &c) {
    std::vector<std::uint64_t> q,p;
    for(auto &v:c.parameters_literal()->q()) q.push_back(v.value());
    for(auto &v:c.parameters_literal()->p()) p.push_back(v.value());
    report["actual_q_moduli"]=q; report["actual_p_moduli"]=p;
    report["context_q0_level"]=c.parameters_literal()->q0_level();
    report["context_secret_hamming_weight"]=c.parameters_literal()->hamming_weight();
}
poseidon::gpu::GpuBootstrapProfileConfig boot_config(const Json &fixture, const std::string &id) {
    poseidon::gpu::GpuBootstrapProfileConfig c;
    c.profile_id=id; c.eval_mod_log_scale=fixture.at("context").at("default_scale_log2").get<int>();
    c.eval_mod_sine_degree=58; c.eval_mod_double_angle=2; c.eval_mod_generation_degree=59;
    c.eval_mod_truncate_degree.reset();
    Json j=fixture.value("native_boot",Json::object());
    if(fixture.contains("native_boot_profiles") && fixture.at("native_boot_profiles").contains(id))
        j.update(fixture.at("native_boot_profiles").at(id));
#define BOOT_FIELD(field) if(j.contains(#field)) c.field=j.at(#field).get<decltype(c.field)>();
    BOOT_FIELD(bootstrap_ratio) BOOT_FIELD(eval_mod_log_scale) BOOT_FIELD(eval_mod_double_angle)
    BOOT_FIELD(eval_mod_k) BOOT_FIELD(eval_mod_arcsine_degree) BOOT_FIELD(eval_mod_sine_degree)
    BOOT_FIELD(eval_mod_generation_degree) BOOT_FIELD(eval_mod_fixed_degree_refit)
    BOOT_FIELD(eval_mod_log_split) BOOT_FIELD(eval_mod_virtual_degree_bound)
    BOOT_FIELD(logical_rescale_count) BOOT_FIELD(c2s_log_bsgs_ratio) BOOT_FIELD(c2s_layer_groups)
    BOOT_FIELD(c2s_direct_layer_threshold) BOOT_FIELD(s2c_log_bsgs_ratio) BOOT_FIELD(project_real)
    BOOT_FIELD(sparse_secret_hamming_weight) BOOT_FIELD(eval_mod_full_coefficients)
    BOOT_FIELD(compress_linear_plaintexts)
#undef BOOT_FIELD
    if(j.contains("eval_mod_truncate_degree") && !j.at("eval_mod_truncate_degree").is_null())
        c.eval_mod_truncate_degree=j.at("eval_mod_truncate_degree").get<std::uint32_t>();
    if(j.contains("output_level") && !j.at("output_level").is_null()) c.output_level=j.at("output_level").get<int>();
    return c;
}
Json profile_report(const poseidon::gpu::GpuBootstrapProfile &p, const poseidon::gpu::GpuBootstrapProfileConfig &c) {
    return {{"profile_id",p.profile_id},{"cuda_device",p.cuda_device_id},
        {"input_level_min",p.input_level_min},{"input_level_max",p.input_level_max},
        {"input_scale_log2",p.input_scale_log2 ? Json(*p.input_scale_log2):Json(nullptr)},
        {"input_components",p.input_components},{"output_level",p.output_level},
        {"output_scale_log2",p.output_scale_log2},{"output_components",p.output_components},
        {"bootstrap_constant_bytes",p.bootstrap_data.allocated_bytes()},
        {"config",{{"eval_mod_log_scale",c.eval_mod_log_scale},{"eval_mod_sine_degree",c.eval_mod_sine_degree},
          {"eval_mod_double_angle",c.eval_mod_double_angle},{"eval_mod_generation_degree",c.eval_mod_generation_degree},
          {"eval_mod_truncate_degree",c.eval_mod_truncate_degree?Json(*c.eval_mod_truncate_degree):Json(nullptr)},
          {"output_level",c.output_level?Json(*c.output_level):Json(nullptr)},
          {"project_real",c.project_real},{"bootstrap_ratio",c.bootstrap_ratio},
          {"profile_id",c.profile_id},{"eval_mod_k",c.eval_mod_k},
          {"eval_mod_arcsine_degree",c.eval_mod_arcsine_degree},
          {"eval_mod_fixed_degree_refit",c.eval_mod_fixed_degree_refit},
          {"eval_mod_log_split",c.eval_mod_log_split},
          {"eval_mod_virtual_degree_bound",c.eval_mod_virtual_degree_bound},
          {"eval_mod_full_coefficients",c.eval_mod_full_coefficients},
          {"logical_rescale_count",c.logical_rescale_count},
          {"sparse_secret_hamming_weight",c.sparse_secret_hamming_weight},
          {"c2s_layer_groups",c.c2s_layer_groups},{"c2s_log_bsgs_ratio",c.c2s_log_bsgs_ratio},
          {"s2c_log_bsgs_ratio",c.s2c_log_bsgs_ratio},
          {"c2s_direct_layer_threshold",c.c2s_direct_layer_threshold},
          {"compress_linear_plaintexts",c.compress_linear_plaintexts}}}};
}
fhegpu::PrepareBackend parse_backend(const std::string &s) {
    if(s=="cpu") return fhegpu::PrepareBackend::Cpu;
    if(s=="gpu") return fhegpu::PrepareBackend::Gpu;
    if(s=="auto") return fhegpu::PrepareBackend::Auto;
    throw std::runtime_error("backend must be cpu, gpu or auto");
}
void configure_waterline(fhegpu::RuntimePlan &plan,const Json &fixture,const std::optional<std::string> &backend) {
    if(fixture.contains("waterline")) {
        const auto &j=fixture.at("waterline"); fhegpu::WaterlineConfig w;
        w.device_capacity_bytes=j.at("device_capacity_bytes").get<std::vector<std::uint64_t>>();
        w.device_reserve_bytes=j.at("device_reserve_bytes").get<std::vector<std::uint64_t>>();
        w.device_fractions=j.at("device_fractions").get<std::vector<double>>();
        w.host_budget_bytes=j.at("host_budget_bytes").get<std::uint64_t>();
        w.pinned_budget_bytes=j.at("pinned_budget_bytes").get<std::uint64_t>();
        w.raw_budget_bytes=j.at("raw_budget_bytes").get<std::uint64_t>();
        w.lookahead_groups=j.value("lookahead_groups",8U); w.backend=parse_backend(j.value("backend","cpu")); plan.waterline=w;
    }
    if(backend) { if(!plan.waterline) throw std::runtime_error("--backend requires Waterline"); plan.waterline->backend=parse_backend(*backend); }
}
std::vector<double> slots(const Json &row,std::uint64_t id,std::size_t capacity) {
    if(row.contains("value_id") && row.at("value_id").get<std::uint64_t>()!=id) throw std::runtime_error("fixture value_id order mismatch");
    const auto &a=row.contains("values")?row.at("values"):row.at("slots");
    auto v=a.get<std::vector<double>>();
    if(v.empty() || v.size()>capacity || std::any_of(v.begin(),v.end(),[](double x){return !std::isfinite(x);}))
        throw std::runtime_error("fixture values must be 1..N/2 finite real numbers");
    return v;
}
Json timing(const fhegpu::RuntimeTiming &t) {
    return {{"setup_ms",t.setup_nanoseconds/1e6},{"initialization_ms",t.initialization_nanoseconds/1e6},
        {"online_execution_ms",t.online_execution_nanoseconds/1e6},{"encode_calls",t.encode_calls},
        {"encode_submission_ms",t.encode_nanoseconds/1e6},{"bundle_read_calls",t.bundle_read_calls},
        {"bundle_read_bytes",t.bundle_read_bytes},{"bundle_read_ms",t.bundle_read_nanoseconds/1e6},
        {"bundle_resident_bytes",t.bundle_resident_bytes},{"bundle_resident_load_ms",t.bundle_resident_load_nanoseconds/1e6},
        {"raw_peak_bytes",t.raw_peak_bytes},{"compute_calls",t.compute_calls},{"boot_calls",t.boot_calls},
        {"compute_submission_including_boot_ms",t.compute_including_boot_nanoseconds/1e6},
        {"boot_submission_ms",t.boot_nanoseconds/1e6},{"communication_post_calls",t.communication_post_calls},
        {"communication_wait_calls",t.communication_wait_calls},{"communication_post_ms",t.communication_post_nanoseconds/1e6},
        {"communication_wait_ms",t.communication_wait_nanoseconds/1e6},{"fence_calls",t.fence_calls}};
}
// The Runtime still consumes the unmodified, verified plan. Only immutable
// plaintext producers/transfers can hit this setup-populated cache. Custom
// communication handles return their already-synchronized real GPU values;
// ciphertext transfers and all arithmetic always use the real GPU API.
class RunnerApi : public BaseApi {
public:
    using BaseApi::BaseApi;
    struct CommHandle {
        std::optional<BaseApi::CommHandle> real;
        std::vector<std::optional<Value>> cached;
    };
    std::unordered_map<fhegpu::ValueId,Value> constants;
    std::atomic<std::uint64_t> native_boot_calls{0}, cache_hits{0}, prepared_calls{0};
    std::atomic<std::uint64_t> prepared_linear_calls{0}, prepared_qk_calls{0}, prepared_av_calls{0};
    std::function<Value(const fhegpu::ComputeOp&,const std::vector<Value>&)> prepared_compute;
    Value encode_plaintext(const fhegpu::ValueDesc &d,const std::vector<double> &v) {
        auto i=constants.find(d.id); if(i!=constants.end()){++cache_hits;return i->second;}
        return BaseApi::encode_plaintext(d,v);
    }
    Value prepare_plaintext_async(const fhegpu::ValueDesc &d,const std::vector<double> &v,fhegpu::PrepareBackend b) {
        auto i=constants.find(d.id); if(i!=constants.end()){++cache_hits;return i->second;}
        return BaseApi::prepare_plaintext_async(d,v,b);
    }
    CommHandle communicate_async(const fhegpu::CommAction &a,const std::vector<Value> &v,const std::vector<fhegpu::ValueDesc> &d) {
        CommHandle h;
        bool cached=!a.outputs.empty();
        for(auto id:a.outputs) if(!constants.count(id)) cached=false;
        if(cached) {for(auto id:a.outputs)h.cached.emplace_back(constants.at(id));++cache_hits;}
        else h.real.emplace(BaseApi::communicate_async(a,v,d));
        return h;
    }
    static std::vector<std::optional<Value>> take_cached(CommHandle &h) {
        std::vector<std::optional<Value>> outputs(h.cached.size());
        for(std::size_t i=0;i<h.cached.size();++i) {
            if(h.cached[i]) {
                outputs[i].emplace(std::move(*h.cached[i]));
                h.cached[i].reset();
            }
        }
        return outputs;
    }
    std::vector<std::optional<Value>> posted_outputs(CommHandle &h) { return h.real?BaseApi::posted_outputs(*h.real):take_cached(h); }
    std::vector<std::optional<Value>> wait(CommHandle &h) { return h.real?BaseApi::wait(*h.real):take_cached(h); }
    Value compute(const fhegpu::ComputeOp &op,const std::vector<Value> &v) {
        if(op.kind==fhegpu::ComputeKind::ApiCall) {
            if(!prepared_compute)throw std::runtime_error("prepared ApiCall requires the compact native backend build");
            auto out=prepared_compute(op,v);++prepared_calls;
            const auto &attrs=std::get<fhegpu::ApiCallAttrs>(op.attrs);
            if(attrs.callee=="bert.compact_linear")++prepared_linear_calls;
            else if(attrs.callee=="bert.compact_qk")++prepared_qk_calls;
            else if(attrs.callee=="bert.compact_av" || attrs.callee=="bert.compact_av_packed_partial")++prepared_av_calls;
            return out;
        }
        auto out=BaseApi::compute(op,v); if(op.kind==fhegpu::ComputeKind::Boot) ++native_boot_calls; return out;
    }
    fhegpu::ResourceUsage value_resources(const fhegpu::ValueDesc &d) const {
        return constants.count(d.id)?fhegpu::ResourceUsage(local_device_count()):BaseApi::value_resources(d);
    }
    fhegpu::ResourceUsage waterline_fixed_resources() const {
        auto usage=BaseApi::waterline_fixed_resources();
        usage+=resident_usage; return usage;
    }
    fhegpu::PrepareResources prepare_resources(const fhegpu::ValueDesc &d,fhegpu::PrepareBackend b) {
        if(!constants.count(d.id)) return BaseApi::prepare_resources(d,b);
        fhegpu::ResourceUsage peak(local_device_count()),resident(local_device_count());
        peak.host=peak.raw=slot_bytes; return {peak,resident,true};
    }
    fhegpu::ResourceUsage resident_usage{0};
    std::uint64_t slot_bytes=0;
};
#ifdef BERT_COMPACT_LINEAR
// Bench-owned prepared implementation: the verified V7 call is opaque, but
// every internal weight multiplication/rotation/rescale is actual Poseidon GPU
// arithmetic. Input owners survive the complete output-tile invocation.
class PreparedLinear {
    struct Binding { std::string matrix;std::size_t tile; };
    bert::compact::CompactLinearEngine engine_;
    std::unordered_map<std::uint64_t,Binding> calls_;
    std::vector<Value> sequence_inputs_;
    std::vector<const poseidon::gpu::GpuCiphertextData*> device_inputs_;
    std::unordered_map<std::string,double> matrix_ms_;
    std::unordered_map<std::string,std::uint64_t> matrix_calls_;
    Json setup_=Json::array();
public:
    PreparedLinear(const poseidon::PoseidonContext &context,std::shared_ptr<const poseidon::GaloisKeys> keys,
                   const std::filesystem::path &manifest_path,bool periodic_encoding=false):engine_(context,std::move(keys),0) {
        auto manifest=read_json(manifest_path);
        if(manifest.value("format_version",0)!=1 || manifest.value("dtype","")!="<f4")
            throw std::runtime_error("unsupported compact Linear manifest");
        auto weight_path=manifest_path.parent_path()/manifest.at("weight_file").get<std::string>();
        std::ifstream input(weight_path,std::ios::binary);
        if(!input)throw std::runtime_error("cannot open compact dense weights");
        std::ostringstream raw;raw<<input.rdbuf();auto bytes=raw.str();
        if(input.bad() || "sha256:"+fhegpu::sha256_hex(bytes)!=manifest.at("weight_sha256").get<std::string>())
            throw std::runtime_error("compact dense weight SHA256 mismatch");
        std::unordered_map<std::string,Json> matrices,contracts;
        for(const auto &row:manifest.at("matrices")) {
            auto id=row.at("matrix_id").get<std::string>();
            if(!matrices.emplace(id,row).second)throw std::runtime_error("duplicate compact matrix id");
        }
        for(const auto &call:manifest.at("calls")) {
            auto id=call.at("matrix_id").get<std::string>();
            auto contract=call;contract.erase("call_id");contract.erase("output_tile");
            auto old=contracts.find(id);
            if(old!=contracts.end() && old->second!=contract)
                throw std::runtime_error("one matrix id has inconsistent physical contracts");
            contracts[id]=contract;
            if(!calls_.emplace(std::stoull(call.at("call_id").get<std::string>()),
                Binding{id,call.at("output_tile").get<std::size_t>()}).second)
                throw std::runtime_error("duplicate compact call_id");
        }
        std::vector<std::string> order;for(const auto &[id,c]:contracts)order.push_back(id);
        std::sort(order.begin(),order.end(),[&](const std::string &a,const std::string &b){
            auto x=contracts.at(a).at("input_level").get<int>(),y=contracts.at(b).at("input_level").get<int>();
            return x==y?a<b:x>y;
        });
        const auto floats=[&](std::uint64_t offset,std::size_t count) {
            if(offset>bytes.size() || count>(bytes.size()-offset)/sizeof(float))
                throw std::runtime_error("compact weight range exceeds binary asset");
            std::vector<float> out(count);std::memcpy(out.data(),bytes.data()+offset,count*sizeof(float));return out;
        };
        for(const auto &id:order) {
            const auto &row=matrices.at(id),&call=contracts.at(id);
            bert::compact::MatrixSpec s;s.id=id;s.periodic_encoding=periodic_encoding;
            s.input_width=call.at("input_width").get<std::size_t>();s.output_width=call.at("output_width").get<std::size_t>();
            if(s.input_width!=row.at("input_width").get<std::size_t>() || s.output_width!=row.at("output_width").get<std::size_t>())
                throw std::runtime_error("compact matrix dimensions disagree");
            s.tile=call.at("tile").get<std::size_t>();s.baby=call.at("baby_step").get<std::size_t>();
            if(s.tile!=256 || s.baby!=16 || context.parameters_literal()->slot()%s.tile)
                throw std::runtime_error("unsupported compact tile/baby layout");
            s.tokens=context.parameters_literal()->slot()/s.tile;
            s.input_level=call.at("input_level").get<std::size_t>();s.input_scale_log2=call.at("input_scale_log2").get<double>();
            s.output_level=call.at("output_level").get<std::size_t>();s.output_scale_log2=call.at("output_scale_log2").get<double>();
            s.rescale_count=call.at("rescale_count").get<std::uint32_t>();
            if(!s.input_width || !s.output_width || s.output_width>bytes.size()/sizeof(float)/s.input_width)
                throw std::runtime_error("compact matrix dimensions overflow asset");
            auto weights=floats(row.at("weight_offset").get<std::uint64_t>(),s.input_width*s.output_width);
            auto bias=floats(row.at("bias_offset").get<std::uint64_t>(),s.output_width);
            engine_.prepare(s,weights,bias);const auto &m=engine_.matrix_stats(id);
            setup_.push_back({{"matrix_id",id},{"input_level",s.input_level},{"output_level",s.output_level},
                {"input_scale_log2",s.input_scale_log2},{"output_scale_log2",s.output_scale_log2},
                {"weight_bytes",m.weight_bytes},{"compact_bytes",m.compact_bytes},{"full_equivalent_bytes",m.full_equivalent_bytes},
                {"cp_terms",m.cp_terms},{"baby_rotations",m.baby_rotations},{"giant_rotations",m.giant_rotations},
                {"compression_fallbacks",m.compression_fallbacks},{"max_period",m.max_period},
                {"plaintext_scale_log2",m.plaintext_scale_log2},{"setup_seconds",m.setup_seconds},
                {"encoding_mode",m.encoding_mode},{"periodic_encoded_terms",m.periodic_encoded_terms},
                {"strict_encoded_terms",m.strict_encoded_terms},{"periodic_encoding_fallbacks",m.periodic_encoding_fallbacks},
                {"periodic_fallback_reason",m.periodic_fallback_reason},
                {"periodic_full_ntt_mapping_verified",m.periodic_full_ntt_mapping_verified},
                {"native_encoder_bitwise_guaranteed",m.native_encoder_bitwise_guaranteed},
                {"native_encoder_audit_terms",m.native_encoder_audit_terms},
                {"native_encoder_mismatch_terms",m.native_encoder_mismatch_terms},
                {"native_encoder_mismatch_words",m.native_encoder_mismatch_words}});
            std::cout<<"PREPARED matrix="<<id<<" seconds="<<m.setup_seconds<<" compact_bytes="<<m.compact_bytes<<'\n'<<std::flush;
        }
    }
    Value compute(RunnerApi &api,const fhegpu::ComputeOp &op,const std::vector<Value> &values) {
        const auto *attrs=std::get_if<fhegpu::ApiCallAttrs>(&op.attrs);
        if(!attrs || attrs->callee!="bert.compact_linear" || op.place.index!=0)
            throw std::runtime_error("unsupported prepared GPU callee/device");
        const auto &binding=calls_.at(attrs->call_id);auto begin=Clock::now();
        if(binding.tile==0) {
            sequence_inputs_=values;device_inputs_.clear();
            for(auto &value:sequence_inputs_) {
                api.synchronize(value);device_inputs_.push_back(&value.device_ciphertext());
            }
        } else {
            if(values.size()!=device_inputs_.size())throw std::runtime_error("compact sequence operand count changed");
            for(std::size_t i=0;i<values.size();++i)
                if(&values[i].device_ciphertext()!=device_inputs_[i])
                    throw std::runtime_error("compact sequence operands changed");
        }
        auto result=engine_.run(binding.matrix,binding.tile,device_inputs_);
        matrix_ms_[binding.matrix]+=ms(begin,Clock::now());++matrix_calls_[binding.matrix];
        return Value::from_device_ciphertext(std::move(result));
    }
    void begin_run() {engine_.reset_sequence();sequence_inputs_.clear();device_inputs_.clear();matrix_ms_.clear();matrix_calls_.clear();}
    Json run_report() const {return {{"matrix_synchronized_wall_ms",matrix_ms_},{"matrix_calls",matrix_calls_},
        {"scope","prepared callback wall includes input-ready wait, native GPU linear execution and output synchronization; not isolated GPU kernel timing"}};}
    Json report() const {
        auto s=engine_.engine_stats();return {{"matrices",setup_},{"parameter_bytes",s.parameter_bytes},{"key_bytes",s.key_bytes},
            {"key_storage_is_full_q",s.key_storage_is_full_q},{"run_calls",s.run_calls},{"actual_rotation_calls",s.actual_rotation_calls}};
    }
};
#endif
#ifdef BERT_PREPARED_QK
#include "prepared_qk_adapter.h"
#endif
#ifdef BERT_PREPARED_AV
#include "prepared_av_adapter.h"
#endif
std::uint64_t available_host_bytes() {
    std::ifstream mem("/proc/meminfo"); std::string key,unit; std::uint64_t kb=0,result=0;
    while(mem>>key>>kb>>unit) if(key=="MemAvailable:"){result=kb*1024;break;}
    for(auto paths: {std::pair{"/sys/fs/cgroup/memory.max","/sys/fs/cgroup/memory.current"},
                    std::pair{"/sys/fs/cgroup/memory/memory.limit_in_bytes","/sys/fs/cgroup/memory/memory.usage_in_bytes"}}) {
        std::ifstream limit_file(paths.first),used_file(paths.second); std::uint64_t limit=0,used=0;
        if(limit_file>>limit && used_file>>used && limit>used) result=result?std::min(result,limit-used):limit-used;
    }
    return result;
}
Json prepare_resident(RunnerApi &api,const fhegpu::RuntimePlan &plan,
                      const std::optional<std::filesystem::path> &bundle,std::size_t capacity,
                      const std::string &mode,std::uint64_t budget,const std::string &resident_backend,
                      const std::string &placement,const fhegpu::BundleReadOptions &bundle_options) {
    auto begin=Clock::now();
    const bool host_only=placement=="host";
    std::size_t prepare_count=0;
    api.resident_usage=fhegpu::ResourceUsage(api.local_device_count());
    std::unordered_map<fhegpu::ValueId,const fhegpu::ValueDesc*> desc;
    for(const auto &d:plan.values) desc.emplace(d.id,&d);
    std::set<fhegpu::ValueId> candidates,host_encode_roots;
    std::unordered_map<fhegpu::ValueId,std::size_t> consumer_counts;
    std::vector<std::string> contents;
    fhegpu::ResourceUsage usage(api.local_device_count());
    const auto add=[&](fhegpu::ValueId id) {
        if(candidates.insert(id).second && !host_only) usage+=api.BaseApi::value_resources(*desc.at(id));
    };
    for(const auto *phase:{&plan.initialization,&plan.execution,&plan.finalization}) for(const auto &i:*phase) {
        const fhegpu::EncodePayload *payload=nullptr;
        if(const auto *op=std::get_if<fhegpu::ComputeOp>(&i.body))
            for(auto id:op->inputs) ++consumer_counts[id];
        if(const auto *op=std::get_if<fhegpu::CommAction>(&i.body))
            for(auto id:op->inputs) ++consumer_counts[id];
        if(const auto *op=std::get_if<fhegpu::EncodeOp>(&i.body)) {
            add(op->output);payload=&op->payload;
            const auto &d=*desc.at(op->output);
            if(d.kind==fhegpu::ValueKind::Plaintext && d.place.kind==fhegpu::PlaceKind::Host) {
                host_encode_roots.insert(op->output);
                if(host_only)usage+=api.BaseApi::value_resources(d);
            }
        }
        else if(const auto *op=std::get_if<fhegpu::PreparePlaintextOp>(&i.body)) {
            ++prepare_count;add(op->output);payload=&op->payload;
        }
        else if(const auto *op=std::get_if<fhegpu::CommAction>(&i.body)) {
            bool constant=!op->inputs.empty();
            for(auto id:op->inputs) if(!candidates.count(id)) constant=false;
            for(auto id:op->outputs) if(desc.at(id)->kind!=fhegpu::ValueKind::Plaintext) constant=false;
            if(constant) for(auto id:op->outputs) add(id);
        }
        if(payload) if(auto *p=std::get_if<fhegpu::BundleEncodePayload>(payload)) contents.push_back(p->content);
    }
    // Optimize only a unique, direct Host Encode -> Device plaintext transfer.
    // Every other producer/consumer shape retains its original CPU setup path.
    std::unordered_map<fhegpu::ValueId,const fhegpu::CommAction*> gpu_pairs;
    if(resident_backend=="gpu")
        for(const auto *phase:{&plan.initialization,&plan.execution,&plan.finalization}) for(const auto &i:*phase) {
            const auto *op=std::get_if<fhegpu::CommAction>(&i.body);
            if(!op || op->kind!=fhegpu::CommKind::Transfer || op->inputs.size()!=1 || op->outputs.size()!=1 ||
               op->sources.size()!=1 || op->destinations.size()!=1 || op->output_types.size()!=1 ||
               op->output_types[0]!=fhegpu::ValueKind::Plaintext || !host_encode_roots.count(op->inputs[0]) ||
               consumer_counts.at(op->inputs[0])!=1) continue;
            const auto &host=*desc.at(op->inputs[0]),&device=*desc.at(op->outputs[0]);
            if(device.kind==fhegpu::ValueKind::Plaintext && device.place.kind==fhegpu::PlaceKind::Device &&
               host.place.rank==device.place.rank && host.context==device.context && host.level==device.level &&
               host.scale_log2==device.scale_log2 && host.ntt && device.ntt && host.components==1 && device.components==1)
                gpu_pairs.emplace(host.id,op);
        }
    std::vector<std::uint64_t> budgets,available_bytes;
    Json capacity_sources=Json::array(),capacity_snapshots=Json::array();
    const auto measurements=api.waterline_measurements();
    bool fits=!(host_only && prepare_count);
    for(int d=0;d<api.local_device_count();++d) {
        cuda_check(cudaSetDevice(api.cuda_device_id(d)),"cudaSetDevice"); std::size_t free=0,total=0;
        cuda_check(cudaMemGetInfo(&free,&total),"cudaMemGetInfo");
        const auto &m=measurements.devices.at(d);
        // CUDA free excludes memory already reserved by RMM. Resident values
        // allocate inside that pool, so admission must use its remaining credit.
        const std::uint64_t available=m.managed_pool
            ? (m.pool_capacity>m.allocated_live?m.pool_capacity-m.allocated_live:0)
            : free;
        const char *source=m.managed_pool?"rmm_pool_capacity_minus_allocated_live":"cuda_mem_get_info_free";
        constexpr std::uint64_t gib=1ULL<<30;
        const std::uint64_t safe_available=available>gib?available-gib:0;
        auto allowed=budget?std::min(budget,safe_available):std::min(available*3/5,safe_available);
        available_bytes.push_back(available);capacity_sources.push_back(source);
        capacity_snapshots.push_back({{"logical_device",d},{"source",source},{"available_bytes",available},
            {"safe_available_bytes",safe_available},{"cuda_free_bytes",free},{"managed_pool",m.managed_pool},
            {"pool_capacity_bytes",m.pool_capacity},{"pool_reserved_bytes",m.pool_reserved},
            {"allocated_live_bytes",m.allocated_live}});
        budgets.push_back(allowed); if(usage.plain[d]+usage.front[d]>allowed) fits=false;
    }
    auto host_budget=available_host_bytes()/2; if(!host_budget || usage.host>host_budget) fits=false;
    // Waterline admission owns every allocation and reserve. Do not bypass it
    // with an out-of-band cache; its normal online preparation remains active.
    if(plan.waterline) fits=false;
    Json report{{"requested",mode},{"backend_requested",resident_backend},{"placement",placement},
        {"eligible_gpu_pairs",gpu_pairs.size()},{"prepare_plaintext_instructions",prepare_count},
        {"forward_memory_admitted",false},
        {"admission_scope",host_only?"Host Encode cache only; Device plaintext estimate is zero, NOT a forward GPU memory admission":"persistent Host+Device cache only; full forward peak is not admitted"},
        {"candidate_count",candidates.size()},{"estimated_host_bytes",usage.host},
        {"estimated_device_bytes",usage.plain},{"device_budget_bytes",budgets},{"host_budget_bytes",host_budget},
        {"device_available_bytes",available_bytes},{"device_capacity_sources",capacity_sources},
        {"device_capacity_snapshots",capacity_snapshots}};
    if(mode=="off" || !fits || candidates.empty()) {
        if(mode=="on" && !fits) throw std::runtime_error("resident cache exceeds budget, Waterline active, or host-only cache has unsupported PreparePlaintext; use auto/off or cache both");
        report["enabled"]=false;
        report["reason"]=mode=="off"?"explicit_off":plan.waterline?"waterline_owns_preparation":
            (host_only && prepare_count)?"host_only_prepare_plaintext_unsupported":candidates.empty()?"no_plaintexts":"conservative_memory_budget";
        report["timing_scope"]="original plan: Encode/PreparePlaintext and weight H2D are timed in their original initialization/execution phases";
        return report;
    }
    std::optional<fhegpu::PlaintextBundleLoader> loader;
    if(plan.plaintext_bundle) {
        if(!bundle) throw std::runtime_error("plaintext bundle required");
        loader.emplace(fhegpu::PlaintextBundleLoader::open(*bundle,*plan.plaintext_bundle,contents,capacity,false,bundle_options));
        report["setup_bundle_resident_bytes"]=loader->resident_bytes();
        report["setup_bundle_resident_load_ms"]=loader->resident_load_seconds()*1000;
    }
    const auto payload_slots=[&](const fhegpu::EncodePayload &p) {
        if(auto *v=std::get_if<fhegpu::InlineEncodePayload>(&p)) return v->values;
        if(!loader) throw std::runtime_error("bundle payload without bundle reference");
        return loader->read(std::get<fhegpu::BundleEncodePayload>(p).content);
    };
    // Keep the cache private until it is complete: setup calls must not hit it.
    std::unordered_map<fhegpu::ValueId,Value> values;
    std::uint64_t cpu_encodes=0,cpu_prepares=0,gpu_prepares=0,gpu_pair_count=0,d2h_calls=0,skipped_uploads=0;
    std::uint64_t deferred_transfers=0,temporary_release_syncs=0;
    for(const auto *phase:{&plan.initialization,&plan.execution,&plan.finalization}) for(const auto &i:*phase) {
        if(const auto *op=std::get_if<fhegpu::EncodeOp>(&i.body)) {
            const auto &host_desc=*desc.at(op->output);
            const auto pair=gpu_pairs.find(op->output);
            if(pair!=gpu_pairs.end()) {
              {
                const auto &device_desc=*desc.at(pair->second->outputs[0]);
                auto device_value=api.BaseApi::prepare_plaintext_async(device_desc,payload_slots(op->payload),fhegpu::PrepareBackend::Gpu);
                api.synchronize(device_value);api.validate_value(device_value,device_desc);++gpu_prepares;
                fhegpu::CommAction download;
                download.kind=fhegpu::CommKind::Transfer;download.hint=fhegpu::CommHint::PointToPoint;
                download.inputs={device_desc.id};download.outputs={host_desc.id};
                download.sources={device_desc.place};download.destinations={host_desc.place};
                download.output_types={fhegpu::ValueKind::Plaintext};
                auto handle=api.BaseApi::communicate_async(download,{device_value},{host_desc});
                auto downloaded=api.BaseApi::wait(handle);
                if(downloaded.size()!=1 || !downloaded[0]) throw std::runtime_error("resident GPU encode plaintext download missing value");
                api.synchronize(*downloaded[0]);api.validate_value(*downloaded[0],host_desc);
                values.emplace(host_desc.id,std::move(*downloaded[0]));
                if(!host_only)values.emplace(device_desc.id,std::move(device_value));
                ++gpu_pair_count;++d2h_calls;
              }
                if(host_only) {
                    // Drop the temporary GPU plaintext and all completed API
                    // retainers before the next FFT. No Device value is cached.
                    api.drain();
                    cuda_check(cudaDeviceSynchronize(),"resident Host-only temporary GPU release");
                    ++temporary_release_syncs;
                }
            } else {
                auto value=api.BaseApi::encode_plaintext(host_desc,payload_slots(op->payload));
                api.synchronize(value);api.validate_value(value,host_desc);
                values.emplace(op->output,std::move(value));++cpu_encodes;
            }
        } else if(const auto *op=std::get_if<fhegpu::PreparePlaintextOp>(&i.body)) {
            auto backend=resident_backend=="gpu"?fhegpu::PrepareBackend::Gpu:fhegpu::PrepareBackend::Cpu;
            const auto &d=*desc.at(op->output);
            auto value=api.BaseApi::prepare_plaintext_async(d,payload_slots(op->payload),backend);
            api.synchronize(value);api.validate_value(value,d);values.emplace(op->output,std::move(value));
            if(backend==fhegpu::PrepareBackend::Gpu)++gpu_prepares;else ++cpu_prepares;
        } else if(const auto *op=std::get_if<fhegpu::CommAction>(&i.body)) {
            if(op->outputs.empty() || !candidates.count(op->outputs.front())) continue;
            // Host-only setup does no out-of-band transfer. The unchanged
            // Runtime must perform each original upload in its declared phase.
            if(host_only) {++deferred_transfers;continue;}
            if(std::all_of(op->outputs.begin(),op->outputs.end(),[&](auto id){return values.count(id)!=0;})) {
                ++skipped_uploads;continue;
            }
            std::vector<Value> inputs;std::vector<fhegpu::ValueDesc> outputs;
            for(auto id:op->inputs) inputs.push_back(values.at(id));
            for(auto id:op->outputs) outputs.push_back(*desc.at(id));
            auto h=api.BaseApi::communicate_async(*op,inputs,outputs); auto results=api.BaseApi::wait(h);
            for(std::size_t j=0;j<op->outputs.size();++j) {
                if(!results.at(j)) throw std::runtime_error("resident upload missing value");
                api.synchronize(*results[j]);api.validate_value(*results[j],*desc.at(op->outputs[j]));
                values.emplace(op->outputs[j],std::move(*results[j]));
            }
        }
    }
    api.drain();api.constants=std::move(values);api.resident_usage=usage;api.slot_bytes=capacity*sizeof(double);
    report["enabled"]=true;report["setup_ms"]=ms(begin,Clock::now());report["cached_values"]=api.constants.size();
    std::size_t host_values=0,device_values=0;
    for(const auto &[id,value]:api.constants) {
        if(value.place_kind()==fhegpu::PlaceKind::Host)++host_values;else ++device_values;
    }
    report["cached_host_values"]=host_values;report["cached_device_values"]=device_values;
    report["backend_counts"]={{"cpu_encode_calls",cpu_encodes},{"cpu_prepare_calls",cpu_prepares},
        {"gpu_prepare_calls",gpu_prepares},{"gpu_encode_host_device_pairs",gpu_pair_count},
        {"gpu_plaintext_d2h_calls",d2h_calls},{"preuploaded_transfers_skipped",skipped_uploads},
        {"original_transfers_deferred_to_runtime",deferred_transfers},{"temporary_gpu_release_syncs",temporary_release_syncs},
        {"gpu_requested_cpu_encode_fallbacks",resident_backend=="gpu"?cpu_encodes:0}};
    report["timing_scope"]=host_only
        ? "Host-only cache: one-time GPU Encode+D2H (or CPU Encode) is setup; GPU temporaries are synchronized and released per pair. Only Host Encode producers hit cache. All original Device plaintext uploads remain real H2D in each Runtime plan's original initialization/execution phase, with ordinary Device value resource accounting and last-use release. Raw bundle reads/validation remain in their original phases. Cache admission does NOT admit the forward GPU peak."
        : "all weight preparation is setup-only: eligible gpu mode pairs use real GPU Encode then D2H to cache the same RNS plaintext on Host and Device; other Host Encode shapes use CPU. Original verified plan Encode/upload producers hit these real-value caches online, but raw bundle reads/validation remain in their original phases";
    return report;
}
Value download(RunnerApi &api,fhegpu::ValueId id,const fhegpu::ValueDesc &d,const Value &v) {
    if(d.place.kind==fhegpu::PlaceKind::Host) return v;
    fhegpu::CommAction a; a.kind=fhegpu::CommKind::Transfer;a.hint=fhegpu::CommHint::PointToPoint;
    a.inputs={id};a.outputs={id};a.sources={d.place};a.destinations={{fhegpu::PlaceKind::Host,0,0}};
    a.output_types={fhegpu::ValueKind::Ciphertext};auto out=d;out.place=a.destinations.front();
    auto h=api.communicate_async(a,{v},{out});auto result=api.wait(h);
    if(result.size()!=1 || !result[0]) throw std::runtime_error("output download missing value");return std::move(*result[0]);
}
Json summary(const std::vector<std::complex<double>> &actual,const std::vector<double> &expected) {
    std::size_t finite=0,nonfinite=0,reference_compared=0;double max_real=0,max_imag=0,max_error=0;long double squares=0;
    Json sample=Json::array();
    for(std::size_t i=0;i<actual.size();++i) {
        auto z=actual[i];if(i<8) sample.push_back({z.real(),z.imag()});
        if(!std::isfinite(z.real())||!std::isfinite(z.imag())){++nonfinite;continue;}
        ++finite;max_real=std::max(max_real,std::abs(z.real()));max_imag=std::max(max_imag,std::abs(z.imag()));
        if(i<expected.size()){++reference_compared;double e=std::abs(z.real()-expected[i]);max_error=std::max(max_error,e);squares+=static_cast<long double>(e)*e;}
    }
    Json j{{"slots",actual.size()},{"finite_slots",finite},{"nonfinite_slots",nonfinite},
        {"max_abs_real",max_real},{"max_abs_imaginary",max_imag},{"first_slots_real_imaginary",sample},
        {"numerical_acceptance_enforced",false}};
    if(!expected.empty()) {
        j["reference_slots"]=expected.size();j["reference_compared_finite_slots"]=reference_compared;
        const bool valid=reference_compared==expected.size() && std::isfinite(max_error);
        j["reference_error_valid"]=valid;
        j["max_abs_reference_error"]=valid?Json(max_error):Json(nullptr);
        j["reference_rms_error"]=valid?Json(std::sqrt(double(squares/reference_compared))):Json(nullptr);
    }
    return j;
}
} // namespace

int main(int argc,char **argv) {
    const auto startup=Clock::now();
    bool probe=argc>1 && std::string(argv[1])=="--probe-profile";
    if((probe && argc<4) || (!probe && argc<5)) {
        std::cerr<<"usage: bert_native_runner PLAN SPEC FIXTURE REPORT [DECODED] [--bundle DIR] [--warmup 1] [--iterations 3] [--resident-plain auto|on|off] [--resident-backend cpu|gpu] [--resident-cache host|both] [--allocator managed|async] [--resident-budget-bytes N] [--native-boot-workspace-bytes N] [--trace PATH] [--cuda-profiler-iteration N] [--backend cpu|gpu|auto]\n"
                 <<"       bert_native_runner --probe-profile FIXTURE REPORT [--profile-id ID] [--device 0]\n";return 2;
    }
    const std::filesystem::path report_path=probe?argv[3]:argv[4];
    Json report{{"format_version",1},{"completed",false},{"numerical_acceptance_enforced",false},
                {"secret_key_assisted_refresh",false},{"mode",probe?"profile_probe":"runtime_plan"}};
    std::string stage="arguments";
    // Must outlive every GPU object in either branch and in exception cleanup.
    std::unique_ptr<AsyncAllocator> external_allocator;
    try {
        std::optional<std::filesystem::path> bundle,decoded;
        std::optional<std::string> backend;
        std::string resident="auto",resident_backend="cpu",resident_cache="both",allocator="managed",profile_id="bert-native";
        std::uint64_t workspace=0,budget=0,bundle_budget=0;
        int warmup=1,iterations=3,device=0,profiler_iteration=-1;
        int option=probe?4:5;
        if(!probe && option<argc && std::string(argv[option]).rfind("--",0)!=0) decoded=argv[option++];
        for(;option<argc;option+=2) {
            if(option+1>=argc) throw std::runtime_error("option requires a value");
            std::string name=argv[option],value=argv[option+1];
            if(name=="--bundle")bundle=value;else if(name=="--backend")backend=value;
            else if(name=="--resident-plain")resident=value;
            else if(name=="--resident-backend")resident_backend=value;
            else if(name=="--resident-cache")resident_cache=value;
            else if(name=="--allocator")allocator=value;
            else if(name=="--resident-budget-bytes")budget=std::stoull(value);
            else if(name=="--bundle-resident-bytes")bundle_budget=std::stoull(value);
            else if(name=="--native-boot-workspace-bytes")workspace=std::stoull(value);
            else if(name=="--warmup")warmup=std::stoi(value);else if(name=="--iterations")iterations=std::stoi(value);
            else if(name=="--profile-id")profile_id=value;else if(name=="--device")device=std::stoi(value);
            else if(name=="--cuda-profiler-iteration")profiler_iteration=strict_int_argument(value,name);
            else if(name=="--trace") {
                std::filesystem::path path=value;if(!path.parent_path().empty())std::filesystem::create_directories(path.parent_path());
                if(setenv("POSEIDON_RUNTIME_TRACE",value.c_str(),1))throw std::runtime_error("cannot set trace path");report["trace_path"]=value;
            } else throw std::runtime_error("unknown option: "+name);
        }
        if(iterations>0 && warmup>std::numeric_limits<int>::max()-iterations)
            throw std::runtime_error("warmup+iterations exceeds supported integer range");
        if(profiler_iteration < -1 || (profiler_iteration >= 0 && (probe || profiler_iteration >= static_cast<long long>(warmup)+iterations)))
            throw std::runtime_error("invalid --cuda-profiler-iteration (zero-based run including warmup)");
        report["cuda_profiler_capture_iteration"]=profiler_iteration;
        report["instrumented_diagnostic"]=profiler_iteration>=0;
        if(warmup<0 || iterations<1 || (resident!="auto"&&resident!="on"&&resident!="off") ||
           (resident_backend!="cpu"&&resident_backend!="gpu") || (resident_cache!="host"&&resident_cache!="both") ||
           (allocator!="managed"&&allocator!="async"))
            throw std::runtime_error("invalid warmup/iterations/resident mode/backend/allocator");
        report["allocator"]=allocator;
        if(probe && device<0)throw std::runtime_error("invalid --device");
        if(probe && allocator=="async" && device!=0)
            throw std::runtime_error("--allocator async requires exactly one visible GPU (logical device 0)");
        // Call only after CPU fixture/context/key-plan validation, but before
        // constructing any GPU owner. The outer owner still outlives them all.
        const auto install_allocator=[&] {
            if(allocator!="async")return;
            stage="install_async_allocator";int visible=0;
            cuda_check(cudaGetDeviceCount(&visible),"async allocator cudaGetDeviceCount");
            if(visible!=1)
                throw std::runtime_error("--allocator async requires exactly one visible GPU (logical device 0)");
            external_allocator=std::make_unique<AsyncAllocator>(0);
            report["external_allocator_config"]=external_allocator->snapshot("installed_before_gpu_api_and_profile");
        };
        stage="load_fixture";std::string fixture_digest;
        auto fixture=read_json(probe?argv[2]:argv[3],&fixture_digest);
        if(fixture.value("format_version",1)!=1) throw std::runtime_error("unsupported fixture format_version");
        report["fixture_sha256"]=fixture_digest;report["context"]=fixture.at("context");report["model"]=fixture.value("model",Json::object());
        const bool require_pack_inputs=prepared_qk_requires_pack_inputs(fixture);
#ifndef BERT_PREPARED_QK
        if(require_pack_inputs)throw std::runtime_error("required pack_inputs capability needs a prepared QK backend build");
#endif
        const bool require_av_packed_partial=prepared_av_requires_packed_partial(fixture);
#ifndef BERT_PREPARED_AV
        if(require_av_packed_partial)throw std::runtime_error("required packed_input2_partial capability needs a prepared AV backend build");
#endif
        if(probe) {
            stage="probe_context";auto context=make_context(fixture.at("context"));context_report(report,context);
            if(fixture.contains("native_boot"))profile_id=fixture.at("native_boot").value("profile_id",profile_id);
            auto config=boot_config(fixture,profile_id);
#ifdef BERT_PREPARED_QK
            if(fixture.contains("prepared_qk_manifest")) {
                stage="prepared_qk_key_plan";
                const auto path=std::filesystem::path(argv[2]).parent_path()/fixture.at("prepared_qk_manifest").get<std::string>();
                (void)PreparedQk::cpu_rotation_steps(path,
                    fixture.at("prepared_qk_manifest_sha256").get<std::string>(),true,require_pack_inputs);
            }
#endif
#ifdef BERT_PREPARED_AV
            if(fixture.contains("prepared_av_manifest")) {
                stage="prepared_av_key_plan";
                const auto path=std::filesystem::path(argv[2]).parent_path()/fixture.at("prepared_av_manifest").get<std::string>();
                (void)PreparedAv::cpu_rotation_steps(path,
                    fixture.at("prepared_av_manifest_sha256").get<std::string>(),true,require_av_packed_partial);
                PreparedAv::validate_context_contract(path,
                    fixture.at("prepared_av_manifest_sha256").get<std::string>(),context,require_av_packed_partial);
            }
#endif
            install_allocator();
            poseidon::KeyGenerator keys(context);
            stage="probe_native_profile";auto begin=Clock::now();
            auto profile=poseidon::gpu::GpuBootstrapProfileBuilder::build(context,keys,device,config);
            report["native_boot_profile"]=profile_report(profile,config);report["profile_setup_ms"]=ms(begin,Clock::now());
            report["native_boot_calls"]=0;report["profile_probe_only"]=true;
            report["probe_scope"]="builds and uploads real native bootstrap constants/keys, but runs no ciphertext Boot or inference";
            if(external_allocator)report["external_allocator"]=external_allocator->snapshot("profile_probe_end_before_teardown");
        } else {
            stage="load_and_verify_plan";
            auto loaded=fhegpu::RuntimePlanReader::read_file(argv[1]);auto spec=fhegpu::OperatorSpecReader::read_file(argv[2]);auto &plan=loaded.plan;
            configure_waterline(plan,fixture,backend);
            auto requirements=fhegpu::PlanVerifier::verify(plan,spec,false);
            report["plan_verified"]=true;report["plan_source_sha256"]=loaded.source_sha256;report["operator_spec_sha256"]=spec.source_sha256;
            if(plan.target.world_size!=1 || plan.target.device_counts.size()!=1 || plan.target.device_counts[0]<1)throw std::runtime_error("runner requires one rank and at least one GPU");
            const auto policy=poseidon::runtime_api::rotation_key_policy_from_environment();
            const bool prepared_direct=fixture.value("prepared_direct_rotation_keys",false);
#ifdef BERT_PREPARED_QK
            std::set<int> prepared_qk_cpu_rotations;
            const bool prepared_qk_direct=prepared_direct || policy==poseidon::runtime_api::RotationKeyPolicy::Direct;
            if(fixture.contains("prepared_qk_manifest")) {
                stage="prepared_qk_key_plan";
                const auto path=std::filesystem::path(argv[3]).parent_path()/fixture.at("prepared_qk_manifest").get<std::string>();
                prepared_qk_cpu_rotations=PreparedQk::cpu_rotation_steps(path,
                    fixture.at("prepared_qk_manifest_sha256").get<std::string>(),prepared_qk_direct,require_pack_inputs);
                report["prepared_qk_rotation_key_mode"]=prepared_qk_direct?"direct":"binary";
                stage="load_and_verify_plan";
            }
#endif
#ifdef BERT_PREPARED_AV
            std::set<int> prepared_av_cpu_rotations;
            std::unordered_map<std::uint64_t,std::string> prepared_av_callees;
            const bool prepared_av_direct=prepared_direct || policy==poseidon::runtime_api::RotationKeyPolicy::Direct;
            stage="prepared_av_key_plan";
            if(fixture.contains("prepared_av_manifest")) {
                const auto path=std::filesystem::path(argv[3]).parent_path()/fixture.at("prepared_av_manifest").get<std::string>();
                auto av_plan=PreparedAv::cpu_plan(path,
                    fixture.at("prepared_av_manifest_sha256").get<std::string>(),prepared_av_direct,require_av_packed_partial);
                prepared_av_cpu_rotations=std::move(av_plan.rotation_steps);
                prepared_av_callees=std::move(av_plan.callees);
                report["prepared_av_rotation_key_mode"]=prepared_av_direct?"direct":"binary";
            }
            // The actual Runtime callee must agree with this call ID's mode,
            // not merely with a self-reported manifest capability/algorithm.
            bool referenced_av_packed_partial=false;
            for(const auto *phase:{&plan.initialization,&plan.execution,&plan.finalization})for(const auto &i:*phase) {
                const auto *op=std::get_if<fhegpu::ComputeOp>(&i.body);
                if(!op || op->kind!=fhegpu::ComputeKind::ApiCall)continue;
                const auto *attrs=std::get_if<fhegpu::ApiCallAttrs>(&op->attrs);
                if(!attrs || (attrs->callee!="bert.compact_av" && attrs->callee!="bert.compact_av_packed_partial"))continue;
                const auto expected=prepared_av_callees.find(attrs->call_id);
                if(expected==prepared_av_callees.end() || expected->second!=attrs->callee)
                    throw std::runtime_error("prepared AV Runtime call_id algorithm/callee mismatch or missing manifest");
                if(attrs->callee=="bert.compact_av_packed_partial")referenced_av_packed_partial=true;
            }
            if(require_av_packed_partial && !referenced_av_packed_partial)
                throw std::runtime_error("required packed_input2_partial capability has no matching Runtime AV call");
            stage="load_and_verify_plan";
#endif
            const int devices=plan.target.device_counts[0];
            std::set<std::pair<int,std::string>> profiles;std::size_t boot_instructions=0;
            for(const auto *phase:{&plan.initialization,&plan.execution,&plan.finalization})for(const auto &i:*phase) {
                if(const auto *op=std::get_if<fhegpu::ComputeOp>(&i.body);op && op->kind==fhegpu::ComputeKind::Boot) {
                    const auto &a=std::get<fhegpu::BootAttrs>(op->attrs);
                    if(a.implementation!=fhegpu::BootImplementation::Native || op->place.kind!=fhegpu::PlaceKind::Device)
                        throw std::runtime_error("only true native GPU Boot is permitted (decrypt_reencrypt forbidden)");
                    profiles.emplace(op->place.index,a.operator_profile);++boot_instructions;
                }
            }
            report["native_boot_instructions_per_plan"]=boot_instructions;
            const std::size_t capacity=spec.spec.poly_degree/2;
            std::unordered_map<fhegpu::ValueId,const fhegpu::ValueDesc*> desc;for(const auto &d:plan.values)desc.emplace(d.id,&d);
            const auto &rows=fixture.at("inputs");
            if(!rows.is_array() || rows.size()!=plan.external_inputs.size())throw std::runtime_error("fixture inputs disagree with plan external_inputs");
            if(fixture.contains("expected_outputs") && fixture.at("expected_outputs").size()!=plan.final_outputs.size())throw std::runtime_error("expected output count mismatch");
            stage="validate_fixture";
            std::vector<std::vector<double>> input_values,expected_values(plan.final_outputs.size());
            input_values.reserve(rows.size());
            for(std::size_t i=0;i<rows.size();++i) {
                auto id=plan.external_inputs[i];const auto &d=*desc.at(id);
                if(d.kind!=fhegpu::ValueKind::Ciphertext || d.place.kind!=fhegpu::PlaceKind::Host || d.place.rank!=0)
                    throw std::runtime_error("fixture external inputs must be rank-0 Host ciphertexts");
                input_values.push_back(slots(rows[i],id,capacity));
            }
            for(std::size_t i=0;i<plan.final_outputs.size();++i) {
                auto id=plan.final_outputs[i];
                if(desc.at(id)->kind!=fhegpu::ValueKind::Ciphertext)throw std::runtime_error("final output must be ciphertext");
                if(fixture.contains("expected_outputs"))expected_values[i]=slots(fixture.at("expected_outputs")[i],id,capacity);
            }
            stage="create_context";auto begin=Clock::now();auto context=make_context(fixture.at("context"),&spec.spec);context_report(report,context);report["context_ms"]=ms(begin,Clock::now());
#ifdef BERT_PREPARED_AV
            if(fixture.contains("prepared_av_manifest")) {
                stage="prepared_av_key_plan";
                const auto path=std::filesystem::path(argv[3]).parent_path()/fixture.at("prepared_av_manifest").get<std::string>();
                PreparedAv::validate_context_contract(path,
                    fixture.at("prepared_av_manifest_sha256").get<std::string>(),context,require_av_packed_partial);
            }
#endif
            stage="key_plan";
            bool need_relin=false;std::set<int> rotations;
            for(const auto &k:requirements.keys){if(k.kind==fhegpu::KeyKind::Relin)need_relin=true;if(k.kind==fhegpu::KeyKind::Galois){if(!k.rotation_step)throw std::runtime_error("rotation key lacks step");rotations.insert(*k.rotation_step);}}
            if(fixture.contains("prepared_rotation_steps"))
                for(const auto &step:fixture.at("prepared_rotation_steps"))rotations.insert(step.get<int>());
            if(fixture.contains("prepared_qk_manifest") || fixture.contains("prepared_av_manifest"))need_relin=true;
            auto basis=poseidon::runtime_api::rotation_key_steps(rotations,context.parameters_literal()->slot(),policy);
            // Keep ordinary Runtime rotations on its selected policy, while
            // explicitly allowing the prepared Linear to hoist direct babies.
            if(prepared_direct && fixture.contains("prepared_rotation_steps"))
                for(const auto &step:fixture.at("prepared_rotation_steps"))basis.insert(step.get<int>());
#ifdef BERT_PREPARED_QK
            // Generate real CPU keys for every manifest mode, even if the
            // fixture's legacy prepared_rotation_steps lists only baseline QK.
            // Binary engine paths use RAW signed parts, not Runtime's canonical
            // rotation decomposition; do not normalize/redecompose those parts.
            if(!prepared_qk_cpu_rotations.empty()) {
                if(prepared_qk_direct && !prepared_direct)
                    prepared_qk_cpu_rotations=poseidon::runtime_api::rotation_key_steps(
                        prepared_qk_cpu_rotations,context.parameters_literal()->slot(),
                        poseidon::runtime_api::RotationKeyPolicy::Direct);
                basis.insert(prepared_qk_cpu_rotations.begin(),prepared_qk_cpu_rotations.end());
                report["prepared_qk_cpu_rotation_steps"]=prepared_qk_cpu_rotations;
            }
#endif
#ifdef BERT_PREPARED_AV
            // Union the engine's actual algorithm-specific CPU key needs. Raw
            // signed binary parts must bypass Runtime's canonical decomposition.
            if(!prepared_av_cpu_rotations.empty()) {
                if(prepared_av_direct && !prepared_direct)
                    prepared_av_cpu_rotations=poseidon::runtime_api::rotation_key_steps(
                        prepared_av_cpu_rotations,context.parameters_literal()->slot(),
                        poseidon::runtime_api::RotationKeyPolicy::Direct);
                basis.insert(prepared_av_cpu_rotations.begin(),prepared_av_cpu_rotations.end());
                report["prepared_av_cpu_rotation_steps"]=prepared_av_cpu_rotations;
            }
#endif
            report["prepared_direct_rotation_keys"]=prepared_direct;
            std::map<std::pair<int,std::string>,poseidon::gpu::GpuBootstrapProfileConfig> native_boot_configs;
            for(const auto &[d,id]:profiles)native_boot_configs.emplace(std::make_pair(d,id),boot_config(fixture,id));
            fhegpu::BundleReadOptions bundle_options;bundle_options.resident_byte_limit=bundle_budget;
            if(!bundle_budget) if(auto env=std::getenv("POSEIDON_RUNTIME_BUNDLE_RESIDENT_BYTES"))bundle_options.resident_byte_limit=std::stoull(env);
            install_allocator();
            stage="device_discovery";int visible=0;cuda_check(cudaGetDeviceCount(&visible),"cudaGetDeviceCount");
            if(visible!=devices)throw std::runtime_error("plan device count must match visible CUDA devices; set CUDA_VISIBLE_DEVICES");
            report["devices"]=Json::array();for(int d=0;d<devices;++d){cudaDeviceProp p{};cuda_check(cudaGetDeviceProperties(&p,d),"cudaGetDeviceProperties");report["devices"].push_back({{"logical_device",d},{"name",p.name},{"total_global_bytes",p.totalGlobalMem}});}
            stage="key_generation";begin=Clock::now();poseidon::KeyGenerator keys(context);
            auto secret=std::make_shared<poseidon::SecretKey>(keys.secret_key());auto public_key=std::make_shared<poseidon::PublicKey>();keys.create_public_key(*public_key);
            auto relin=std::make_shared<poseidon::RelinKeys>();auto galois=std::make_shared<poseidon::GaloisKeys>();
            if(need_relin)keys.create_relin_keys(*relin);
            if(!basis.empty())keys.create_galois_keys(std::vector<int>(basis.begin(),basis.end()),*galois);
            report["rotation_key_policy"]=poseidon::runtime_api::rotation_key_policy_name(policy);report["rotation_key_basis"]=basis;report["key_generation_ms"]=ms(begin,Clock::now());
            stage="encrypt_inputs";begin=Clock::now();poseidon::CKKSEncoder encoder(context);poseidon::Encryptor encryptor(context,*public_key);poseidon::Decryptor decryptor(context,*secret);
            std::unordered_map<fhegpu::ValueId,Value> inputs;report["inputs"]=Json::array();
            for(std::size_t i=0;i<rows.size();++i) {
                auto id=plan.external_inputs[i];const auto &d=*desc.at(id);
                const auto &values=input_values[i];poseidon::Plaintext plain;poseidon::Ciphertext ct;
                encoder.encode(values,context.crt_context()->parms_id_map().at(d.level),std::exp2(d.scale_log2),plain);encryptor.encrypt(plain,ct);
                inputs.emplace(id,Value::from_host_ciphertext(std::move(ct)));report["inputs"].push_back({{"value_id",id},{"actual_slots",values.size()},{"level",d.level},{"scale_log2",d.scale_log2}});
            }
            report["input_encryption_ms"]=ms(begin,Clock::now());
            stage="create_gpu_api";std::vector<int> cuda_devices(devices);std::iota(cuda_devices.begin(),cuda_devices.end(),0);
            // Never pass encryption/decryption keys to the GPU Runtime API.
            RunnerApi api(spec.spec.context_id,context,cuda_devices,relin,galois);
            stage="configure_native_boot";report["native_boot_profiles"]=Json::array();
            for(auto &[d,id]:profiles) {
                const auto &config=native_boot_configs.at({d,id});begin=Clock::now();auto p=poseidon::gpu::GpuBootstrapProfileBuilder::build(context,keys,d,config);
                auto declared=std::find_if(spec.spec.boot_profiles.begin(),spec.spec.boot_profiles.end(),[&](const auto &v){return v.profile_id==id;});
                if(declared==spec.spec.boot_profiles.end() || p.output_level!=declared->output_level || p.output_scale_log2!=declared->output_scale_log2 ||
                   declared->input_level_min<p.input_level_min || declared->input_level_max>p.input_level_max ||
                   p.input_components!=declared->input_components || p.output_components!=declared->output_components)
                    throw std::runtime_error("built native profile disagrees with OperatorSpec; calibrate with --probe-profile first");
                auto j=profile_report(p,config);j["setup_ms"]=ms(begin,Clock::now());j["workspace_capacity_bytes"]=workspace?workspace:std::uint64_t(context.parameters_literal()->degree())*65536;
                report["native_boot_profiles"].push_back(j);api.configure_native_bootstrap(d,std::move(p),workspace);
            }
            stage="gpu_preflight";begin=Clock::now();
            api.preflight(loaded.preflight_identity.empty()?loaded.source_sha256:loaded.preflight_identity,
                          false,plan.target,spec.spec,requirements);
            report["gpu_preflight_setup_ms"]=ms(begin,Clock::now());
#ifdef BERT_COMPACT_LINEAR
            std::unique_ptr<PreparedLinear> prepared;
            if(fixture.contains("compact_linear_manifest")) {
                stage="prepared_compact_linear_setup";begin=Clock::now();
                auto manifest_path=std::filesystem::path(argv[3]).parent_path()/fixture.at("compact_linear_manifest").get<std::string>();
                prepared=std::make_unique<PreparedLinear>(context,galois,manifest_path,
                    fixture.value("compact_periodic_encoding",false));
                report["prepared_linear_setup_ms"]=ms(begin,Clock::now());report["prepared_linear"]=prepared->report();
            }
#endif
#ifdef BERT_PREPARED_QK
            std::unique_ptr<PreparedQk> prepared_qk;
            if(fixture.contains("prepared_qk_manifest")) {
                stage="prepared_cyclic_qk_setup";begin=Clock::now();
                auto path=std::filesystem::path(argv[3]).parent_path()/fixture.at("prepared_qk_manifest").get<std::string>();
                prepared_qk=std::make_unique<PreparedQk>(context,relin,galois,path,
                    fixture.at("prepared_qk_manifest_sha256").get<std::string>(),require_pack_inputs);
                report["prepared_qk_setup_ms"]=ms(begin,Clock::now());report["prepared_qk"]=prepared_qk->report();
            }
#endif
#ifdef BERT_PREPARED_AV
            std::unique_ptr<PreparedAv> prepared_av;
            if(fixture.contains("prepared_av_manifest")) {
                stage="prepared_cyclic_av_setup";begin=Clock::now();
                auto path=std::filesystem::path(argv[3]).parent_path()/fixture.at("prepared_av_manifest").get<std::string>();
                prepared_av=std::make_unique<PreparedAv>(context,relin,galois,path,
                    fixture.at("prepared_av_manifest_sha256").get<std::string>(),require_av_packed_partial);
                report["prepared_av_setup_ms"]=ms(begin,Clock::now());report["prepared_av"]=prepared_av->report();
            }
#endif
#if defined(BERT_COMPACT_LINEAR) || defined(BERT_PREPARED_QK) || defined(BERT_PREPARED_AV)
            api.prepared_compute=[&](const fhegpu::ComputeOp &op,const std::vector<Value> &values)->Value {
                const auto &attrs=std::get<fhegpu::ApiCallAttrs>(op.attrs);
#ifdef BERT_COMPACT_LINEAR
                if(attrs.callee=="bert.compact_linear" && prepared)return prepared->compute(api,op,values);
#endif
#ifdef BERT_PREPARED_QK
                if(attrs.callee=="bert.compact_qk" && prepared_qk)return prepared_qk->compute(api,op,values);
#endif
#ifdef BERT_PREPARED_AV
                if((attrs.callee=="bert.compact_av" || attrs.callee=="bert.compact_av_packed_partial") && prepared_av)
                    return prepared_av->compute(api,op,values);
#endif
                throw std::runtime_error("unsupported or unbound prepared callee: "+attrs.callee);
            };
#endif
            stage="resident_plaintext_setup";report["resident_plaintexts"]=prepare_resident(api,plan,bundle,capacity,resident,budget,resident_backend,resident_cache,bundle_options);
            std::cout<<"SETUP ready resident="<<report["resident_plaintexts"]["enabled"]<<" native_boot_instructions="<<boot_instructions<<'\n'<<std::flush;
            fhegpu::RuntimeResources resources{spec,bundle,false,bundle_options};
            fhegpu::SequentialRuntime<RunnerApi> runtime(0,1,devices,api,fhegpu::DeviceExecutionMode::PerDeviceWorkers,devices);
            std::optional<fhegpu::RunArtifact<Value>> last;
            report["runs"]=Json::array();std::vector<double> online,wall;
            stage="run_plan";
            for(int run=0;run<warmup+iterations;++run) {
                last.reset();auto boots_before=api.native_boot_calls.load(),hits_before=api.cache_hits.load(),prepared_before=api.prepared_calls.load();
                auto linear_before=api.prepared_linear_calls.load(),qk_before=api.prepared_qk_calls.load(),av_before=api.prepared_av_calls.load();
#ifdef BERT_COMPACT_LINEAR
                if(prepared)prepared->begin_run();
#endif
#ifdef BERT_PREPARED_QK
                if(prepared_qk)prepared_qk->begin_run();
#endif
#ifdef BERT_PREPARED_AV
                if(prepared_av)prepared_av->begin_run();
#endif
                ProfilerRange profiler(run==profiler_iteration);
                begin=Clock::now();
                auto artifact=runtime.run(loaded,resources,inputs);api.drain();auto elapsed=ms(begin,Clock::now());
                profiler.stop();
                auto j=timing(artifact.timing);j["phase"]=run<warmup?"warmup":"measured";j["index"]=run<warmup?run:run-warmup;
                j["cuda_profiler_range_enabled"]=run==profiler_iteration;
                j["wall_ms"]=elapsed;j["native_boot_calls"]=api.native_boot_calls.load()-boots_before;j["resident_cache_hits"]=api.cache_hits.load()-hits_before;
                j["prepared_calls"]=api.prepared_calls.load()-prepared_before;
                j["prepared_linear_calls"]=api.prepared_linear_calls.load()-linear_before;
                j["prepared_qk_calls"]=api.prepared_qk_calls.load()-qk_before;
                j["prepared_av_calls"]=api.prepared_av_calls.load()-av_before;
#ifdef BERT_COMPACT_LINEAR
                if(prepared)j["prepared_linear"]=prepared->run_report();
#endif
#ifdef BERT_PREPARED_QK
                if(prepared_qk)j["prepared_qk"]=prepared_qk->run_report();
#endif
#ifdef BERT_PREPARED_AV
                if(prepared_av)j["prepared_av"]=prepared_av->run_report();
#endif
                report["runs"].push_back(j);if(run>=warmup){online.push_back(artifact.timing.online_execution_nanoseconds/1e6);wall.push_back(elapsed);}
                std::cout<<"RUN "<<run<<" phase="<<j["phase"]<<" online_ms="<<j["online_execution_ms"]<<" wall_ms="<<elapsed<<" native_boot_calls="<<j["native_boot_calls"]<<'\n'<<std::flush;
                last.emplace(std::move(artifact));
            }
            const auto mean=[](const std::vector<double> &v){return std::accumulate(v.begin(),v.end(),0.0)/v.size();};
            auto sorted=online;std::sort(sorted.begin(),sorted.end());
            report["warmup"]=warmup;report["iterations"]=iterations;report["mean_online_ms"]=mean(online);report["median_online_ms"]=sorted[sorted.size()/2];report["mean_runtime_wall_ms"]=mean(wall);
            report["native_boot_calls_total_including_warmup"]=api.native_boot_calls.load();
#ifdef BERT_COMPACT_LINEAR
            if(prepared)report["prepared_linear"]=prepared->report();
#endif
#ifdef BERT_PREPARED_QK
            if(prepared_qk)report["prepared_qk"]=prepared_qk->report();
#endif
#ifdef BERT_PREPARED_AV
            if(prepared_av)report["prepared_av"]=prepared_av->report();
#endif
            std::uint64_t measured_boots=0;for(const auto &j:report["runs"])if(j.at("phase")=="measured")measured_boots+=j.at("native_boot_calls").get<std::uint64_t>();report["native_boot_calls_measured"]=measured_boots;
            stage="decrypt_summary";begin=Clock::now();report["outputs"]=Json::array();Json decoded_rows=Json::array();
            for(std::size_t i=0;i<plan.final_outputs.size();++i) {
                auto id=plan.final_outputs[i];const auto &d=*desc.at(id);
                auto host=download(api,id,d,last->values.at(id).value);poseidon::Plaintext p;decryptor.decrypt(host.host_ciphertext(),p);
                std::vector<std::complex<double>> actual;encoder.decode(p,actual);const auto &expected=expected_values[i];
                auto j=summary(actual,expected);j["value_id"]=id;j["level"]=d.level;j["scale_log2"]=d.scale_log2;report["outputs"].push_back(j);
                if(decoded){std::vector<double> real,imag;for(auto z:actual){real.push_back(z.real());imag.push_back(z.imag());}decoded_rows.push_back({{"value_id",id},{"real",real},{"imaginary",imag}});}
            }
            if(decoded){save_json(*decoded,{{"format_version",1},{"fixture_sha256",fixture_digest},{"outputs",decoded_rows}});report["decoded_path"]=decoded->string();}
            report["decrypt_and_save_ms"]=ms(begin,Clock::now());api.drain();
            auto m=api.waterline_measurements();report["preparation_measurements_cumulative"]={{"cpu_prepare_calls",m.cpu_prepare_calls},{"cpu_encode_ms",m.cpu_encode_nanoseconds/1e6},{"cpu_plain_h2d_bytes",m.cpu_plain_h2d_bytes},{"transport_h2d_calls",m.transport_h2d_calls},{"transport_h2d_bytes",m.transport_h2d_bytes},{"transport_h2d_ms",m.transport_h2d_nanoseconds/1e6}};
            report["device_measurements"]=Json::array();
            for(std::size_t d=0;d<m.devices.size();++d) {
                const auto &dm=m.devices[d];
                report["device_measurements"].push_back({{"logical_device",d},{"managed_pool",dm.managed_pool},
                    {"unit","bytes"},{"allocated_live",dm.allocated_live},{"allocated_peak",dm.allocated_peak},
                    {"pool_reserved",dm.pool_reserved},{"pool_capacity",dm.pool_capacity}});
            }
            report["timing_scope"]="online_execution_ms is synchronized Runtime execution+finalization, excluding per-run setup/initialization, resident setup, input encryption, profile/key generation, and output decryption. wall_ms includes Runtime setup+initialization+execution+finalization. Cache misses retain original phase encode/H2D costs; raw bundle reads/validation can remain online even with resident plaintexts. Boot/compute component times are CPU submission, not isolated GPU kernel latency.";
            if(external_allocator)report["external_allocator"]=external_allocator->snapshot("run_end_before_runtime_teardown");
        }
        if(external_allocator) {
            cuda_check(cudaDeviceSynchronize(),"async allocator final cleanup synchronization");
            report["external_allocator_after_teardown"]=external_allocator->snapshot("after_all_runtime_profile_values_destroyed");
        }
        struct rusage usage{};if(!getrusage(RUSAGE_SELF,&usage))report["host_max_rss_kib"]=usage.ru_maxrss;
        report["completed"]=true;report["stage"]="complete";report["main_wall_ms"]=ms(startup,Clock::now());
        report["security_level"]="none; experimental parameters, not a security claim";save_json(report_path,report);
        std::cout<<"COMPLETE report="<<report_path<<" numerical_acceptance_enforced=false\n";return 0;
    } catch(const std::exception &e) {
        report["stage"]=stage;report["error"]=e.what();report["main_wall_ms"]=ms(startup,Clock::now());
        if(external_allocator) {
            try {report["external_allocator"]=external_allocator->snapshot("failure_after_runtime_scope_unwound");}
            catch(const std::exception &diagnostic) {report["external_allocator_snapshot_error"]=diagnostic.what();}
        }
        std::cerr<<"FAIL stage="<<stage<<" "<<e.what()<<'\n';try{save_json(report_path,report);}catch(const std::exception &r){std::cerr<<r.what()<<'\n';}return 1;
    }
}
