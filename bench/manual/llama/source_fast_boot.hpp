#pragma once
// Fast per-level special-modulus (P) tables + InversePreRotated Galois keys +
// deduplicated primary(8192)/SiLU(65536) Boot profile construction.
// Opt-in via LLAMA_FAST_P=1 / LLAMA_FAST_BOOT=1 on Mode::Fast.
#include "source_runtime.hpp"
#include "source_manual_boot.hpp"
#include "source_cache.hpp"
#include "poseidon/basics/util/ntt.h"
#include "poseidon/basics/util/rlwe.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace cerium_source::fast_boot {
using namespace cerium_source;
using namespace poseidon::gpu;

inline bool fast_p_enabled() {
    const char *v=std::getenv("LLAMA_FAST_P");return v&&std::string(v)=="1";
}
inline bool fast_boot_enabled() {
    const char *v=std::getenv("LLAMA_FAST_BOOT");
    if(v&&*v)return std::string(v)=="1";
    return fast_p_enabled();
}

inline poseidon::SecretKey slice_secret_key(
    const poseidon::SecretKey &src,std::size_t degree,
    std::size_t global_q,std::size_t q_keep,
    const poseidon::PoseidonContext &sub,std::size_t p_keep) {
    poseidon::SecretKey out;
    out.data().resize(sub,sub.crt_context()->key_parms_id(),degree*(q_keep+p_keep));
    std::copy_n(src.data().data(),degree*q_keep,out.data().data());
    std::copy_n(src.data().data()+degree*global_q,degree*p_keep,out.data().data()+degree*q_keep);
    out.parms_id()=sub.crt_context()->key_parms_id();
    return out;
}

struct HostFastRuntimeBand {
    std::size_t top_q=0,p=0;
    std::vector<std::size_t> qs;
    std::vector<int> steps;
    poseidon::PoseidonContext sub;
    poseidon::RelinKeys relin_cpu;
    poseidon::GaloisKeys galois_cpu;
    HostFastRuntimeBand(std::size_t tq,std::size_t pv,std::vector<std::size_t> qv,
                        std::vector<int> sv,poseidon::PoseidonContext sctx)
        :top_q(tq),p(pv),qs(std::move(qv)),steps(std::move(sv)),sub(std::move(sctx)) {}
};

struct HostFastBootBundle {
    std::vector<HostFastRuntimeBand> bands;
    poseidon::LinearMatrixGroup c2s_cpu,s2c_cpu;
    std::vector<int> c2s_rot_index,s2c_rot_index;
    bool has_cpu_dft=false;
    bool keys_loaded_from_cache=false;
    std::uint64_t bundle_fp=0;
    std::filesystem::path level_gpu_keys_path;
    std::filesystem::path boot_gpu_keys_path;
    std::string public_cache_dir;
    std::string dev_key_cache_dir;
    std::shared_ptr<poseidon::GaloisKeys> boot_gk49;
    std::shared_ptr<poseidon::PoseidonContext> sub6_ctx;
    std::shared_ptr<poseidon::GaloisKeys> boot_gk6;
    std::shared_ptr<poseidon::KSwitchKeys> sse_dense_cpu;
    std::shared_ptr<poseidon::ParametersLiteral> sse_low_lit;
    std::shared_ptr<poseidon::PoseidonContext> sse_low_ctx;
    std::shared_ptr<poseidon::KSwitchKeys> sse_eph_cpu;
    double sse_bits=0;
    std::uint32_t sse_hw=32;
};

namespace detail {
inline std::uint64_t fast_boot_keys_fingerprint(
    const poseidon::ParametersLiteral &global,
    const poseidon::SecretKey &sk,
    bool fast_evalmod_p,
    const std::vector<int> &app_steps,
    const std::vector<int> &c2s_rots,
    const std::vector<int> &s2c_rots,
    std::uint32_t sse_hw) {
    cache::Hasher h;
    h.str("LLAMA_FAST_BOOT_GPU_KEYS_V1");
    h.pod(cache::parameters_fingerprint(global));
    h.pod(cache::secret_key_fingerprint(sk));
    h.pod(static_cast<std::uint8_t>(fast_evalmod_p ? 1 : 0));
    h.pod(sse_hw);
    auto feed_vec = [&](const std::vector<int> &v) {
        h.pod(static_cast<std::uint64_t>(v.size()));
        for (int x : v) h.pod(static_cast<std::int32_t>(x));
    };
    feed_vec(app_steps);
    feed_vec(c2s_rots);
    feed_vec(s2c_rots);
    return h.finish();
}

inline void save_fast_level_gpu_keys_file(
    const std::filesystem::path &path,
    std::uint64_t fp,
    const std::vector<HostFastRuntimeBand> &bands,
    const std::vector<std::shared_ptr<GpuRelinKeysData>> &relins,
    const std::vector<std::shared_ptr<GpuGaloisKeysData>> &galois) {
    cache::AtomicWriter writer(path);
    auto &os = writer.stream();
    cache::detail::write_magic(os, cache::kFastLevelGpuMagic, fp);
    cache::detail::write_pod(os, static_cast<std::uint64_t>(bands.size()));
    for (std::size_t i = 0; i < bands.size(); ++i) {
        cache::detail::write_pod(os, static_cast<std::uint64_t>(bands[i].top_q));
        cache::detail::write_pod(os, static_cast<std::uint64_t>(bands[i].p));
        cache::detail::save_gpu_eval_key(os, *relins[i]);
        const std::uint8_t has_gk = galois[i] ? 1 : 0;
        cache::detail::write_pod(os, has_gk);
        if (has_gk) cache::detail::save_gpu_eval_key(os, *galois[i]);
    }
    writer.commit();
}

inline bool load_fast_level_gpu_keys_file(
    const std::filesystem::path &path,
    std::uint64_t fp,
    int device_id,
    const std::vector<HostFastRuntimeBand> &bands,
    std::vector<std::shared_ptr<GpuRelinKeysData>> &relins,
    std::vector<std::shared_ptr<GpuGaloisKeysData>> &galois) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    try {
        if (!cache::detail::read_and_check_magic(is, cache::kFastLevelGpuMagic, fp)) return false;
        const auto n = static_cast<std::size_t>(cache::detail::read_pod<std::uint64_t>(is));
        if (n != bands.size()) return false;
        relins.resize(n);
        galois.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto tq = static_cast<std::size_t>(cache::detail::read_pod<std::uint64_t>(is));
            const auto pv = static_cast<std::size_t>(cache::detail::read_pod<std::uint64_t>(is));
            if (tq != bands[i].top_q || pv != bands[i].p) return false;
            relins[i] = std::make_shared<GpuRelinKeysData>(cache::detail::load_gpu_eval_key(is, device_id));
            const auto has_gk = cache::detail::read_pod<std::uint8_t>(is);
            if ((has_gk != 0) != !bands[i].steps.empty()) return false;
            if (has_gk) {
                galois[i] = std::make_shared<GpuGaloisKeysData>(cache::detail::load_gpu_eval_key(is, device_id));
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

inline void save_fast_boot_gpu_keys_file(
    const std::filesystem::path &path,
    std::uint64_t fp,
    const GpuGaloisKeysData &gk49,
    const GpuGaloisKeysData &gk6,
    const GpuEvaluationKeyData &sse_dense,
    const GpuEvaluationKeyData &sse_eph) {
    cache::AtomicWriter writer(path);
    auto &os = writer.stream();
    cache::detail::write_magic(os, cache::kFastBootGpuMagic, fp);
    cache::detail::save_gpu_eval_key(os, gk49);
    cache::detail::save_gpu_eval_key(os, gk6);
    cache::detail::save_gpu_eval_key(os, sse_dense);
    cache::detail::save_gpu_eval_key(os, sse_eph);
    writer.commit();
}

inline bool load_fast_boot_gpu_keys_file(
    const std::filesystem::path &path,
    std::uint64_t fp,
    int device_id,
    std::shared_ptr<GpuGaloisKeysData> &gk49,
    std::shared_ptr<GpuGaloisKeysData> &gk6,
    std::shared_ptr<const GpuEvaluationKeyData> &sse_dense,
    std::shared_ptr<const GpuEvaluationKeyData> &sse_eph) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    try {
        if (!cache::detail::read_and_check_magic(is, cache::kFastBootGpuMagic, fp)) return false;
        gk49 = std::make_shared<GpuGaloisKeysData>(cache::detail::load_gpu_eval_key(is, device_id));
        gk6 = std::make_shared<GpuGaloisKeysData>(cache::detail::load_gpu_eval_key(is, device_id));
        sse_dense = std::make_shared<GpuEvaluationKeyData>(cache::detail::load_gpu_eval_key(is, device_id));
        sse_eph = std::make_shared<GpuEvaluationKeyData>(cache::detail::load_gpu_eval_key(is, device_id));
        return true;
    } catch (...) {
        return false;
    }
}
} // namespace detail

inline HostFastBootBundle prepare_host_fast_boot_bundle(
    const poseidon::PoseidonContext &context,
    poseidon::KeyGenerator &keys,
    poseidon::CKKSEncoder &encoder,
    const std::vector<int> &app_steps,
    const std::string &public_cache_override={},
    const std::string &dev_key_cache_override={}) {
    HostFastBootBundle h;
    h.public_cache_dir = cache::public_cache_dir(public_cache_override);
    h.dev_key_cache_dir = cache::dev_key_cache_dir(dev_key_cache_override);
    const auto &global=*context.parameters_literal();
    const std::size_t global_q=global.q().size(),global_p=global.p().size();
    const std::size_t degree=context.crt_context()->key_context_data()->parms().degree();
    if(global_q!=49||global_p!=25||degree!=65536)throw std::runtime_error("fast Q49/P25 fixture required");

    constexpr double raised_scale=double(std::uint64_t{1}<<54);
    const std::uint64_t params_fp=cache::parameters_fingerprint(global);
    const auto pub_ns=cache::public_namespace_key(global,raised_scale,60U);

    // 1. Obtain C2S/S2C rotation indices (skip ~27s CPU DFT build when public cache already holds QP matrices).
    const auto t0=Clock::now();
    bool pub_rot_hit=false;
    if(!h.public_cache_dir.empty()) {
        const auto pub_root=std::filesystem::path(h.public_cache_dir)/pub_ns;
        std::vector<int> c2s_r,s2c_r;
        if(cache::read_qp_group_rot_index(pub_root/"c2s_qp.bin",params_fp,c2s_r)&&
           cache::read_qp_group_rot_index(pub_root/"s2c_qp.bin",params_fp,s2c_r)) {
            h.c2s_rot_index=std::move(c2s_r);
            h.s2c_rot_index=std::move(s2c_r);
            h.has_cpu_dft=false;
            pub_rot_hit=true;
            std::cout<<"LLAMA_PUBLIC_CACHE dft_rot_index_hit=true skipped_cpu_dft=true seconds="<<seconds(t0)<<"\n";
        }
    }
    if(!pub_rot_hit) {
        {
            poseidon::EvalModPoly orig(context,poseidon::CosDiscrete,std::exp2(45.),0,5,3,12,0,22);
            const double scaling=orig.q_div()/(orig.k()*orig.sc_fac()*orig.q_diff())*(raised_scale/std::exp2(45.));
            if(std::abs(12*context.crt_context()->q0()*scaling/raised_scale-1)>1e-12)
                throw std::runtime_error("fast C2S integer-period pairing failed");
            poseidon::HomomorphicDFTMatrixLiteral lit(poseidon::encode,16,15,48,{1,1,1},true,scaling,false,1,{},0);
            lit.create_dynamic(h.c2s_cpu,encoder,raised_scale,std::exp2(50.),std::exp2(50.),1.);
        }
        {
            poseidon::HomomorphicDFTMatrixLiteral lit(poseidon::decode,16,15,5,std::vector<std::uint32_t>(3,1),true,1.0,false,1);
            lit.create_dynamic(h.s2c_cpu,encoder,std::exp2(40.),std::exp2(45.),std::exp2(45.),1.0);
        }
        h.c2s_rot_index=h.c2s_cpu.rot_index();
        h.s2c_rot_index=h.s2c_cpu.rot_index();
        h.has_cpu_dft=true;
        std::cout<<"FAST_BOOT_PHASE dft_cpu="<<seconds(t0)<<"\n";
    }

    // 2. Build sub-contexts in parallel (needed for GpuParameterData levels + SSE low evaluator).
    struct BandSpec {std::size_t top_q,p;std::vector<std::size_t> qs;std::vector<int> steps;};
    std::vector<int> hi_steps;
    for(int s:app_steps) {
        const int a=s<0?-s:s;
        if(a==0||(a<=128&&(a&(a-1))==0))hi_steps.push_back(s);
    }
    if(hi_steps.empty())hi_steps.push_back(0);
    std::vector<BandSpec> specs{
        {6,6,{3,4,5,6},app_steps},
        {10,11,{7,8,9,10},app_steps},
        {13,14,{11,12,13},app_steps},
        {16,16,{14,15,16},app_steps},
        {27,14,{17,18,19,20,21,22,23,24,25,26,27},hi_steps},
    };
    const bool fast_evalmod_p=[]{
        const char *v=std::getenv("LLAMA_FAST_EVALMOD_P");
        return v&&*v&&std::string(v)=="1";
    }();
    if(fast_evalmod_p) {
        specs.push_back({33,17,{28,29,30,31,32,33},{}});
        specs.push_back({37,19,{34,35,36,37},{}});
        specs.push_back({41,21,{38,39,40,41},{}});
        specs.push_back({44,23,{42,43,44},{}});
    }
    const auto t_sub=Clock::now();
    std::vector<std::unique_ptr<poseidon::PoseidonContext>> sub_ctxs(specs.size());
    // Construct bands sequentially so each PoseidonContext uses all OpenMP threads
    // inside CreateNTTTables and CrtContext::validate without nested-OMP serialization.
    for(std::size_t i=0;i<specs.size();++i) {
        poseidon::ParametersLiteral lit=global;
        lit.set_modulus(
            {global.q().begin(),global.q().begin()+static_cast<std::ptrdiff_t>(specs[i].top_q)},
            {global.p().begin(),global.p().begin()+static_cast<std::ptrdiff_t>(specs[i].p)});
        sub_ctxs[i]=std::make_unique<poseidon::PoseidonContext>(lit);
    }
    h.bands.reserve(specs.size());
    for(std::size_t i=0;i<specs.size();++i) {
        h.bands.emplace_back(specs[i].top_q,specs[i].p,specs[i].qs,specs[i].steps,std::move(*sub_ctxs[i]));
    }
    {
        h.sse_hw=32;
        poseidon::ParametersLiteral low_lit=global;
        low_lit.set_modulus({global.q().begin(),global.q().begin()+2},{global.p().begin(),global.p().begin()+2});
        h.sse_low_lit=std::make_shared<poseidon::ParametersLiteral>(low_lit);
        h.sse_low_ctx=std::make_shared<poseidon::PoseidonContext>(low_lit);
        double bits=0;for(const auto &qm:h.sse_low_lit->q())bits+=std::log2(double(qm.value()));
        for(const auto &pm:h.sse_low_lit->p())bits+=std::log2(double(pm.value()));
        h.sse_bits=bits;
    }
    std::cout<<"FAST_BOOT_PHASE sub_contexts="<<seconds(t_sub)<<"\n";

    // 3. Check compact GPU-ready uint32 dev key cache (skips BOTH CPU keygen AND CPU uint64 key loading!).
    h.bundle_fp=detail::fast_boot_keys_fingerprint(
        global,keys.secret_key(),fast_evalmod_p,app_steps,h.c2s_rot_index,h.s2c_rot_index,h.sse_hw);
    if(!h.dev_key_cache_dir.empty()) {
        const auto dev_ns=cache::dev_key_namespace_key(global,app_steps,true,fast_evalmod_p,h.sse_hw);
        const auto dev_root=std::filesystem::path(h.dev_key_cache_dir)/dev_ns;
        h.level_gpu_keys_path=dev_root/"fast_level_gpu_keys.bin";
        h.boot_gpu_keys_path=dev_root/"fast_boot_gpu_keys.bin";
        if(cache::check_cache_file_header(h.level_gpu_keys_path,cache::kFastLevelGpuMagic,h.bundle_fp)&&
           cache::check_cache_file_header(h.boot_gpu_keys_path,cache::kFastBootGpuMagic,h.bundle_fp)) {
            h.keys_loaded_from_cache=true;
            std::cout<<"LLAMA_DEV_KEY_CACHE gpu_ready_keys_hit=true dev_only=true skipped_cpu_keygen=true\n";
            return h;
        }
    }

    // 4. Generate CPU keys on cache miss (parallelized across Galois keys after removing omp critical).
    const auto t_kg=Clock::now();
    {
        poseidon::ParametersLiteral lit6=global;
        lit6.set_modulus({global.q().begin(),global.q().begin()+6},{global.p().begin(),global.p().begin()+6});
        h.sub6_ctx=std::make_shared<poseidon::PoseidonContext>(lit6);
    }
    for(auto &dst:h.bands) {
        const poseidon::SecretKey sk=slice_secret_key(keys.secret_key(),degree,global_q,dst.top_q,dst.sub,dst.p);
        poseidon::KeyGenerator sub_kg(dst.sub,sk);
        sub_kg.create_relin_keys(dst.relin_cpu);
        if(!dst.steps.empty()) {
            const auto gt=dst.sub.crt_context()->galois_tool();
            std::set<std::uint32_t> elts{gt->get_elt_from_step(0)};
            for(int s:dst.steps)elts.insert(gt->get_elt_from_step(s));
            sub_kg.create_galois_keys(std::vector<std::uint32_t>(elts.begin(),elts.end()),dst.galois_cpu);
        }
    }
    {
        const auto gt49=context.crt_context()->galois_tool();
        std::set<std::uint32_t> main_elts{gt49->get_elt_from_step(0)};
        for(int s:h.c2s_rot_index)main_elts.insert(gt49->get_elt_from_step(s));
        h.boot_gk49=std::make_shared<poseidon::GaloisKeys>();
        keys.create_galois_keys(std::vector<std::uint32_t>(main_elts.begin(),main_elts.end()),*h.boot_gk49);

        const auto sk6=slice_secret_key(keys.secret_key(),degree,global_q,6,*h.sub6_ctx,6);
        poseidon::KeyGenerator kg6(*h.sub6_ctx,sk6);
        const auto gt6=h.sub6_ctx->crt_context()->galois_tool();
        std::set<std::uint32_t> elts6{gt6->get_elt_from_step(0)};
        for(int s:h.s2c_rot_index)elts6.insert(gt6->get_elt_from_step(s));
        h.boot_gk6=std::make_shared<poseidon::GaloisKeys>();
        kg6.create_galois_keys(std::vector<std::uint32_t>(elts6.begin(),elts6.end()),*h.boot_gk6);
    }
    {
        const auto global_ctx=context.crt_context();
        const auto kp_id=global_ctx->key_parms_id();
        const std::size_t mod_cnt=global_ctx->key_context_data()->parms().coeff_modulus().size();
        poseidon::SecretKey sparse;sparse.data().resize(context,kp_id,degree*mod_cnt);
        poseidon::util::sample_poly_ternary_with_hamming(
            context.random_generator()->create(),context,kp_id,h.sse_hw,sparse.data().data());
        poseidon::util::RNSIter coeff(sparse.data().data(),degree);
        poseidon::util::ntt_negacyclic_harvey(coeff,mod_cnt,global_ctx->small_ntt_tables());
        sparse.parms_id()=kp_id;

        h.sse_dense_cpu=std::make_shared<poseidon::KSwitchKeys>(keys.create_switch_key(sparse,keys.secret_key()));

        const auto dense_low=slice_secret_key(keys.secret_key(),degree,global_q,2,*h.sse_low_ctx,2);
        const auto sparse_low=slice_secret_key(sparse,degree,global_q,2,*h.sse_low_ctx,2);
        poseidon::KeyGenerator low_kg(*h.sse_low_ctx,dense_low);
        h.sse_eph_cpu=std::make_shared<poseidon::KSwitchKeys>(low_kg.create_switch_key(dense_low,sparse_low));
    }
    std::cout<<"FAST_BOOT_PHASE host_keygen="<<seconds(t_kg)<<"\n";
    return h;
}

inline void install_fast_runtime_levels_from_bundle(Runtime &r,const HostFastBootBundle &h) {
    const int dev=r.device_id;
    gpu_check_cuda(cudaSetDevice(dev),"fast runtime levels device");
    const auto t_start=Clock::now();
    auto relin_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();
    auto galois_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();

    std::vector<std::shared_ptr<GpuRelinKeysData>> band_relins(h.bands.size());
    std::vector<std::shared_ptr<GpuGaloisKeysData>> band_galois(h.bands.size());
    bool gpu_cache_hit=false;
    if(!h.level_gpu_keys_path.empty()) {
        const auto tl=Clock::now();
        if(detail::load_fast_level_gpu_keys_file(h.level_gpu_keys_path,h.bundle_fp,dev,h.bands,band_relins,band_galois)) {
            gpu_cache_hit=true;
            std::cout<<"LLAMA_DEV_KEY_CACHE device="<<dev<<" level_gpu_keys_hit=true dev_only=true seconds="<<seconds(tl)<<"\n";
        }
    }
    for(std::size_t i=0;i<h.bands.size();++i) {
        const auto &b=h.bands[i];
        if(!gpu_cache_hit) {
            band_relins[i]=std::make_shared<GpuRelinKeysData>(GpuUploader::upload_relin_keys(b.relin_cpu,dev));
            if(!b.steps.empty()) {
                band_galois[i]=std::make_shared<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(b.galois_cpu,dev));
            }
        }
        GpuUploader::prepare_key_views_for_q_counts(*band_relins[i],b.qs);
        for(auto q:b.qs)(*relin_map)[q]=band_relins[i];
        if(band_galois[i]) {
            GpuUploader::prepare_key_views_for_q_counts(*band_galois[i],b.qs);
            for(auto q:b.qs)(*galois_map)[q]=band_galois[i];
        }
        GpuParameterData donor(b.sub,dev,b.qs);
        for(auto q:b.qs)r.parameters.adopt_or_insert_level(donor.take_level_by_q_count(q));
    }
    if(!gpu_cache_hit&&!h.level_gpu_keys_path.empty()) {
        const auto ts=Clock::now();
        detail::save_fast_level_gpu_keys_file(h.level_gpu_keys_path,h.bundle_fp,h.bands,band_relins,band_galois);
        std::cout<<"LLAMA_DEV_KEY_CACHE device="<<dev<<" level_gpu_keys_hit=false saved=true dev_only=true save_seconds="<<seconds(ts)<<"\n";
    }
    r.relin.level_keys=std::move(relin_map);
    r.galois.level_keys=std::move(galois_map);
    std::cout<<"FAST_RUNTIME_LEVEL_P installed device="<<dev<<" q3..6=P6 q7..10=P11 q11..13=P14 q14..16=P16 q17..27=P14 InversePreRotated=true seconds="<<seconds(t_start)<<"\n";
}

// Install fast per-level P tables into Runtime::parameters, relin, and galois:
//   q=1..6   -> P=6  (dnum=1, S2C / down mask / proj)
//   q=7..10  -> P=11 (dnum=1, AV / O / SiLU combine / QK; +34b headroom at Q=10)
//   q=11..13 -> P=14 (dnum=1, Linear / QK11 / AV top / Softmax)
//   q=14..16 -> P=16 (dnum=1, Linear in / RMS out / SiLU top)
//   q=17..27 -> P=14 (dnum=2, RMS / Softmax / SiLU top + proj conj)
//   q=28..49 -> P=25 (dnum=2, C2S + EvalMod main context)
inline void install_fast_runtime_levels(
    Runtime &r,const std::vector<int> &app_steps) {
    const int dev=r.device_id;
    gpu_check_cuda(cudaSetDevice(dev),"fast runtime levels device");
    const auto &global=*r.context.parameters_literal();
    const std::size_t global_q=global.q().size();
    const std::size_t degree=r.context.crt_context()->key_context_data()->parms().degree();
    struct Band {std::size_t top_q,p;std::vector<std::size_t> qs;std::vector<int> steps;};
    std::vector<int> hi_steps;
    for(int s:app_steps) {
        const int a=s<0?-s:s;
        if(a==0||(a<=128&&(a&(a-1))==0))hi_steps.push_back(s);
    }
    if(hi_steps.empty())hi_steps.push_back(0);
    std::vector<Band> bands{
        {6,6,{3,4,5,6},app_steps},
        {10,11,{7,8,9,10},app_steps},
        {13,14,{11,12,13},app_steps},
        {16,16,{14,15,16},app_steps},
        {27,14,{17,18,19,20,21,22,23,24,25,26,27},hi_steps},
    };
    const bool fast_evalmod_p=[]{
        const char *v=std::getenv("LLAMA_FAST_EVALMOD_P");
        return v&&*v&&std::string(v)=="1";
    }();
    if(fast_evalmod_p) {
        bands.push_back({33,17,{28,29,30,31,32,33},{}});
        bands.push_back({37,19,{34,35,36,37},{}});
        bands.push_back({41,21,{38,39,40,41},{}});
        bands.push_back({44,23,{42,43,44},{}});
    }
    const auto t_start=Clock::now();
    auto relin_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();
    auto galois_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();
    for(const auto &b:bands) {
        poseidon::ParametersLiteral lit=global;
        lit.set_modulus(
            {global.q().begin(),global.q().begin()+static_cast<std::ptrdiff_t>(b.top_q)},
            {global.p().begin(),global.p().begin()+static_cast<std::ptrdiff_t>(b.p)});
        const poseidon::PoseidonContext sub(lit);
        const poseidon::SecretKey sk=slice_secret_key(r.keys.secret_key(),degree,global_q,b.top_q,sub,b.p);
        poseidon::KeyGenerator sub_kg(sub,sk);
        {
            poseidon::RelinKeys r_cpu;sub_kg.create_relin_keys(r_cpu);
            auto r_gpu=std::make_shared<GpuRelinKeysData>(GpuUploader::upload_relin_keys(r_cpu,dev));
            GpuUploader::prepare_key_views_for_q_counts(*r_gpu,b.qs);
            for(auto q:b.qs)(*relin_map)[q]=r_gpu;
        }
        if(!b.steps.empty()) {
            const auto gt=sub.crt_context()->galois_tool();
            std::set<std::uint32_t> elts{gt->get_elt_from_step(0)};
            for(int s:b.steps)elts.insert(gt->get_elt_from_step(s));
            poseidon::GaloisKeys g_cpu;
            sub_kg.create_galois_keys(std::vector<std::uint32_t>(elts.begin(),elts.end()),g_cpu);
            auto g_gpu=std::make_shared<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(g_cpu,dev));
            GpuUploader::prepare_key_views_for_q_counts(*g_gpu,b.qs);
            for(auto q:b.qs)(*galois_map)[q]=g_gpu;
        }
        GpuParameterData donor(sub,dev,b.qs);
        for(auto q:b.qs)r.parameters.adopt_or_insert_level(donor.take_level_by_q_count(q));
    }
    r.relin.level_keys=std::move(relin_map);
    r.galois.level_keys=std::move(galois_map);
    std::cout<<"FAST_RUNTIME_LEVEL_P installed device="<<dev<<" q3..6=P6 q7..10=P11 q11..13=P14 q14..16=P16 q17..27=P14 InversePreRotated=true seconds="<<seconds(t_start)<<"\n";
}

// Verify profile contract while permitting fast per-level P keys and shared keys.
inline void verify_fast_profile(Runtime &runtime,const GpuBootstrapProfile &profile,
    unsigned ratio,double eta,double target,double alpha) {
    const auto &d=profile.bootstrap_data;const auto id=runtime.parameters.get_level_by_q_count(27).parms_id;
    if((ratio!=8192&&ratio!=65536)||!(eta>0&&std::isfinite(eta))||
       d.output_ratio!=ratio||d.q0_over_message_ratio!=target||d.output_scale_multiplier!=alpha||
       !d.slot_to_coeff_first||!d.project_real||d.allow_environment_linear_transform_override||
       d.linear_transform_mode!=GpuLinearTransformMode::DoubleHoistBsgs||d.input_q_count!=6||d.input_scale!=std::exp2(40.)||
       d.raised_scale_override!=std::exp2(54.)||d.post_raise_integer_multiplier!=1||d.post_raise_scale_multiplier!=1.||!d.post_raise_plaintext.empty()||
       d.slot_to_coeff_input_q_count!=27||d.slot_to_coeff_input_parms_id!=id||d.slot_to_coeff_output_scale!=std::exp2(50.)||
       d.eval_mod.output_q_count!=27||d.eval_mod.output_parms_id!=id||d.eval_mod.target_scale!=std::exp2(50.)||
       d.eval_mod.polynomial_degree!=30||!d.eval_mod.dynamic_rescale||d.eval_mod.double_angle_constants.size()!=3||
       d.active_coeff_to_slot_matrix_qp().rescale_counts()!=std::vector<std::uint32_t>{1,2,1}||
       d.active_slot_to_coeff_matrix_qp().rescale_counts()!=std::vector<std::uint32_t>{1,1,2}||
       d.active_coeff_to_slot_groups_qp().size()!=3||
       d.active_slot_to_coeff_groups_qp().size()!=3||
       d.plus_i_plaintext.meta.q_count!=27||d.plus_i_plaintext.meta.parms_id!=id||d.plus_i_plaintext.meta.scale!=1.||
       profile.output_level!=26||profile.output_scale_log2!=50)
        throw std::runtime_error("fast immutable corrected profile contract");
    if(!d.sparse_switch_to_ephemeral||!d.sparse_switch_to_dense||!d.sparse_low_owner||!d.sparse_low_evaluator||
       d.sparse_switch_to_ephemeral->meta.q_count!=2||d.sparse_switch_to_ephemeral->meta.p_count!=2||
       d.sparse_switch_to_ephemeral->meta.degree!=65536||d.sparse_switch_to_ephemeral->meta.component_count!=2)
        throw std::runtime_error("fast sparse-low Q2/P2 contract");
    const auto [s,q]=manual_backend::planned_scale(*runtime.api,d.raised_scale_override,d.active_coeff_to_slot_matrix_qp());
    if(q!=45||!Api::same_scale(s,d.eval_mod.input_scale))throw std::runtime_error("fast actual C2S output/EvalMod input");
    for(const auto &c:d.eval_mod.polynomial_combine_steps)
        if(c.output_q_count!=c.product_q_count||!(c.product_scale>0&&std::isfinite(c.product_scale))||
           !(c.quotient_output_scale>0)||std::log2(c.quotient_output_scale)<60.-1e-10)
            throw std::runtime_error("fast quotient floor/lazy combine");
}

inline void validate_fast_runtime_keys(Runtime &r) {
    if(!r.relin.level_keys||r.relin.level_keys->empty()||!r.galois.level_keys||r.galois.level_keys->empty())
        throw std::runtime_error("fast mode expected level_keys on Runtime");
    for(std::size_t q=3;q<=27;++q) {
        const auto &lvl=r.parameters.get_level_by_q_count(q);
        const std::size_t p_lvl=lvl.shards.front().hybrid_base_p_count;
        const auto &rk=r.relin.for_level(q);
        const auto &gk=r.galois.for_level(q);
        if(rk.meta.p_count!=p_lvl||gk.meta.p_count!=p_lvl||gk.meta.galois_format!=GpuGaloisKeyFormat::InversePreRotated)
            throw std::runtime_error("fast level key P/format mismatch at q="+std::to_string(q));
    }
}

namespace detail {
struct SparseLowHolder {
    poseidon::ParametersLiteral parameters;
    poseidon::PoseidonContext context;
    std::unique_ptr<GpuParameterData> gpu_parameters;
    std::unique_ptr<GpuEvaluator> evaluator;
    explicit SparseLowHolder(const poseidon::ParametersLiteral &lit):parameters(lit),context(parameters){}
};
}

// Construct both ratio=8192 and ratio=65536 Boot profiles on r.device_id,
// sharing a single GPU C2S/S2C QP matrix allocation across both profiles and
// optionally reading/populating the public constants disk cache.
inline std::array<std::pair<std::unique_ptr<const GpuBootstrapProfile>,manual_backend::ProfileNumbers>,2>
make_fast_profile_pair(Runtime &r,const manual_backend::SubmissionOwner &host,
                       const HostFastBootBundle *shared=nullptr,
                       const std::string &public_cache_override={}) {
    host.check();
    const int dev=r.device_id;
    gpu_check_cuda(cudaSetDevice(dev),"make fast profile pair device");
    const auto t0=Clock::now();
    if(const char *p=std::getenv("POSEIDON_GPU_SSE_LOW_P");p&&std::string(p)!="2")
        throw std::invalid_argument("fast boot requires sparse-low P2");
    const auto &global=*r.context.parameters_literal();
    const std::size_t global_q=global.q().size(),global_p=global.p().size();
    const std::size_t degree=r.context.crt_context()->key_context_data()->parms().degree();
    if(global_q!=49||global_p!=25||degree!=65536)throw std::runtime_error("fast Q49/P25 fixture required");

    constexpr double raised_scale=double(std::uint64_t{1}<<54);
    const std::uint64_t params_fp=cache::parameters_fingerprint(global);
    const std::string pub_dir=public_cache_override.empty()
        ?(shared&&!shared->public_cache_dir.empty()?shared->public_cache_dir:cache::public_cache_dir())
        :public_cache_override;
    const std::filesystem::path pub_root=pub_dir.empty()
        ?std::filesystem::path()
        :std::filesystem::path(pub_dir)/cache::public_namespace_key(global,raised_scale,60U);

    // 1. Load or upload ONE shared GPU copy of C2S/S2C QP matrices + group views for BOTH profiles.
    auto shared_c2s_qp=std::make_shared<GpuLinearMatrixGroupQP>();
    auto shared_s2c_qp=std::make_shared<GpuLinearMatrixGroupQP>();
    bool qp_cache_hit=false;
    if(!pub_root.empty()) {
        const auto t_qp=Clock::now();
        if(cache::load_qp_matrix_group(pub_root/"c2s_qp.bin",params_fp,r.context,dev,*shared_c2s_qp)&&
           cache::load_qp_matrix_group(pub_root/"s2c_qp.bin",params_fp,r.context,dev,*shared_s2c_qp)) {
            qp_cache_hit=true;
            std::cout<<"LLAMA_PUBLIC_CACHE device="<<dev<<" qp_c2s_s2c_hit=true seconds="<<seconds(t_qp)<<"\n";
        }
    }
    poseidon::LinearMatrixGroup local_c2s,local_s2c;
    if(!qp_cache_hit) {
        const poseidon::LinearMatrixGroup *c2s_ptr=(shared&&shared->has_cpu_dft)?&shared->c2s_cpu:&local_c2s;
        const poseidon::LinearMatrixGroup *s2c_ptr=(shared&&shared->has_cpu_dft)?&shared->s2c_cpu:&local_s2c;
        if(!shared||!shared->has_cpu_dft) {
            poseidon::EvalModPoly orig(r.context,poseidon::CosDiscrete,std::exp2(45.),0,5,3,12,0,22);
            const double scaling=orig.q_div()/(orig.k()*orig.sc_fac()*orig.q_diff())*(raised_scale/std::exp2(45.));
            if(std::abs(12*r.context.crt_context()->q0()*scaling/raised_scale-1)>1e-12)
                throw std::runtime_error("fast C2S integer-period pairing failed");
            poseidon::HomomorphicDFTMatrixLiteral lit(poseidon::encode,16,15,48,{1,1,1},true,scaling,false,1,{},0);
            lit.create_dynamic(local_c2s,r.encoder,raised_scale,std::exp2(50.),std::exp2(50.),1.);
            poseidon::HomomorphicDFTMatrixLiteral lit_s(poseidon::decode,16,15,5,std::vector<std::uint32_t>(3,1),true,1.0,false,1);
            lit_s.create_dynamic(local_s2c,r.encoder,std::exp2(40.),std::exp2(45.),std::exp2(45.),1.0);
            std::cout<<"FAST_BOOT_PHASE dft_cpu="<<seconds(t0)<<"\n";
        }
        const auto t_qp=Clock::now();
        *shared_c2s_qp=GpuUploader::upload_linear_matrix_group_qp(*c2s_ptr,r.context,dev,1,true);
        *shared_s2c_qp=GpuUploader::upload_linear_matrix_group_qp(*s2c_ptr,r.context,dev,1,true);
        std::cout<<"FAST_BOOT_PHASE device="<<dev<<" qp_upload_shared="<<seconds(t_qp)<<"\n";
        if(!pub_root.empty()) {
            const auto ts=Clock::now();
            cache::save_qp_matrix_group(pub_root/"c2s_qp.bin",params_fp,*shared_c2s_qp);
            cache::save_qp_matrix_group(pub_root/"s2c_qp.bin",params_fp,*shared_s2c_qp);
            std::cout<<"LLAMA_PUBLIC_CACHE device="<<dev<<" qp_c2s_s2c_hit=false saved=true save_seconds="<<seconds(ts)<<"\n";
        }
    }
    auto shared_c2s_groups=std::make_shared<std::vector<std::vector<GpuMatrixPlainQP>>>();
    auto shared_s2c_groups=std::make_shared<std::vector<std::vector<GpuMatrixPlainQP>>>();
    for(const auto &m:shared_c2s_qp->data())shared_c2s_groups->push_back(GpuUploader::upload_double_hoist_group_views(m,dev));
    for(const auto &m:shared_s2c_qp->data())shared_s2c_groups->push_back(GpuUploader::upload_double_hoist_group_views(m,dev));

    const auto &c2s_rots=shared?shared->c2s_rot_index:shared_c2s_qp->rot_index();
    const auto &s2c_rots=shared?shared->s2c_rot_index:shared_s2c_qp->rot_index();

    const auto id45=r.parameters.get_level_by_q_count(45).parms_id;
    auto boot_relin=std::shared_ptr<const GpuRelinKeysData>(&r.relin,[](const GpuRelinKeysData*){});

    std::shared_ptr<GpuGaloisKeysData> boot_galois;
    std::shared_ptr<const GpuEvaluationKeyData> sse_eph,sse_dense;
    const GpuEvaluator *sse_eval=nullptr;
    std::shared_ptr<const void> sse_owner;
    {
        std::shared_ptr<GpuGaloisKeysData> g6;
        bool boot_gpu_cache_hit=false;
        if(shared&&!shared->boot_gpu_keys_path.empty()) {
            const auto tl=Clock::now();
            if(detail::load_fast_boot_gpu_keys_file(
                   shared->boot_gpu_keys_path,shared->bundle_fp,dev,boot_galois,g6,sse_dense,sse_eph)) {
                boot_gpu_cache_hit=true;
                std::cout<<"LLAMA_DEV_KEY_CACHE device="<<dev<<" boot_gpu_keys_hit=true dev_only=true seconds="<<seconds(tl)<<"\n";
            }
        }
        if(boot_gpu_cache_hit) {
            GpuUploader::prepare_key_views_for_q_counts(*boot_galois,{45,46,48,49});
            GpuUploader::prepare_key_views_for_q_counts(*g6,{4,5,6});
            auto level_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();
            (*level_map)[4]=g6;(*level_map)[5]=g6;(*level_map)[6]=g6;
            if(!r.galois.level_keys||!r.galois.level_keys->count(27))
                throw std::runtime_error("fast boot requires Runtime P=14 key at q=27");
            (*level_map)[27]=r.galois.level_keys->at(27);
            boot_galois->level_keys=std::move(level_map);

            auto holder=std::make_shared<detail::SparseLowHolder>(*shared->sse_low_lit);
            GpuUploader::prepare_key_views_for_q_counts(*sse_eph,{2});
            holder->gpu_parameters=std::make_unique<GpuParameterData>(holder->context,dev,std::vector<std::size_t>{2});
            holder->evaluator=std::make_unique<GpuEvaluator>(*holder->gpu_parameters);
            std::cout<<"GPU_BOOT_SSE device="<<dev<<" dense_to_ephemeral_modulus=low q=2 p=2 bits="<<shared->sse_bits<<" ephemeral_h="<<shared->sse_hw<<"\n";
            sse_eval=holder->evaluator.get();
            sse_owner=std::move(holder);
        } else {
            if(shared) {
                boot_galois=std::make_shared<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(*shared->boot_gk49,dev));
                GpuUploader::prepare_key_views_for_q_counts(*boot_galois,{45,46,48,49});
                auto level_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();
                g6=std::make_shared<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(*shared->boot_gk6,dev));
                GpuUploader::prepare_key_views_for_q_counts(*g6,{4,5,6});
                (*level_map)[4]=g6;(*level_map)[5]=g6;(*level_map)[6]=g6;
                if(!r.galois.level_keys||!r.galois.level_keys->count(27))
                    throw std::runtime_error("fast boot requires Runtime P=14 key at q=27");
                (*level_map)[27]=r.galois.level_keys->at(27);
                boot_galois->level_keys=std::move(level_map);
            } else {
                const auto gt49=r.context.crt_context()->galois_tool();
                std::set<std::uint32_t> main_elts{gt49->get_elt_from_step(0)};
                for(int s:c2s_rots)main_elts.insert(gt49->get_elt_from_step(s));
                poseidon::GaloisKeys gk49;
                r.keys.create_galois_keys(std::vector<std::uint32_t>(main_elts.begin(),main_elts.end()),gk49);
                boot_galois=std::make_shared<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(gk49,dev));
                GpuUploader::prepare_key_views_for_q_counts(*boot_galois,{45,46,48,49});

                auto level_map=std::make_shared<std::map<std::size_t,std::shared_ptr<const GpuEvaluationKeyData>>>();
                poseidon::ParametersLiteral lit6=global;
                lit6.set_modulus({global.q().begin(),global.q().begin()+6},{global.p().begin(),global.p().begin()+6});
                const poseidon::PoseidonContext sub6(lit6);
                const auto sk6=slice_secret_key(r.keys.secret_key(),degree,global_q,6,sub6,6);
                poseidon::KeyGenerator kg6(sub6,sk6);
                const auto gt6=sub6.crt_context()->galois_tool();
                std::set<std::uint32_t> elts6{gt6->get_elt_from_step(0)};
                for(int s:s2c_rots)elts6.insert(gt6->get_elt_from_step(s));
                poseidon::GaloisKeys gk6;kg6.create_galois_keys(std::vector<std::uint32_t>(elts6.begin(),elts6.end()),gk6);
                g6=std::make_shared<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(gk6,dev));
                GpuUploader::prepare_key_views_for_q_counts(*g6,{4,5,6});
                (*level_map)[4]=g6;(*level_map)[5]=g6;(*level_map)[6]=g6;
                if(!r.galois.level_keys||!r.galois.level_keys->count(27))
                    throw std::runtime_error("fast boot requires Runtime P=14 key at q=27");
                (*level_map)[27]=r.galois.level_keys->at(27);
                boot_galois->level_keys=std::move(level_map);
            }
            auto up_sw=[dev](const poseidon::KSwitchKeys &k){
                poseidon::RelinKeys rel;static_cast<poseidon::KSwitchKeys&>(rel)=k;
                return std::make_shared<GpuEvaluationKeyData>(GpuUploader::upload_relin_keys(rel,dev));
            };
            if(shared) {
                sse_dense=up_sw(*shared->sse_dense_cpu);
                auto holder=std::make_shared<detail::SparseLowHolder>(*shared->sse_low_lit);
                sse_eph=up_sw(*shared->sse_eph_cpu);
                GpuUploader::prepare_key_views_for_q_counts(*sse_eph,{2});
                holder->gpu_parameters=std::make_unique<GpuParameterData>(holder->context,dev,std::vector<std::size_t>{2});
                holder->evaluator=std::make_unique<GpuEvaluator>(*holder->gpu_parameters);
                std::cout<<"GPU_BOOT_SSE device="<<dev<<" dense_to_ephemeral_modulus=low q=2 p=2 bits="<<shared->sse_bits<<" ephemeral_h="<<shared->sse_hw<<"\n";
                sse_eval=holder->evaluator.get();
                sse_owner=std::move(holder);
            } else {
                constexpr std::uint32_t hw=32;
                const auto global_ctx=r.context.crt_context();
                const auto kp_id=global_ctx->key_parms_id();
                const std::size_t mod_cnt=global_ctx->key_context_data()->parms().coeff_modulus().size();
                poseidon::SecretKey sparse;sparse.data().resize(r.context,kp_id,degree*mod_cnt);
                poseidon::util::sample_poly_ternary_with_hamming(
                    r.context.random_generator()->create(),r.context,kp_id,hw,sparse.data().data());
                poseidon::util::RNSIter coeff(sparse.data().data(),degree);
                poseidon::util::ntt_negacyclic_harvey(coeff,mod_cnt,global_ctx->small_ntt_tables());
                sparse.parms_id()=kp_id;
                sse_dense=up_sw(r.keys.create_switch_key(sparse,r.keys.secret_key()));
                poseidon::ParametersLiteral low_lit=global;
                low_lit.set_modulus({global.q().begin(),global.q().begin()+2},{global.p().begin(),global.p().begin()+2});
                auto holder=std::make_shared<detail::SparseLowHolder>(low_lit);
                const auto dense_low=slice_secret_key(r.keys.secret_key(),degree,global_q,2,holder->context,2);
                const auto sparse_low=slice_secret_key(sparse,degree,global_q,2,holder->context,2);
                poseidon::KeyGenerator low_kg(holder->context,dense_low);
                sse_eph=up_sw(low_kg.create_switch_key(dense_low,sparse_low));
                GpuUploader::prepare_key_views_for_q_counts(*sse_eph,{2});
                holder->gpu_parameters=std::make_unique<GpuParameterData>(holder->context,dev,std::vector<std::size_t>{2});
                holder->evaluator=std::make_unique<GpuEvaluator>(*holder->gpu_parameters);
                double bits=0;for(const auto &qm:holder->parameters.q())bits+=std::log2(double(qm.value()));
                for(const auto &pm:holder->parameters.p())bits+=std::log2(double(pm.value()));
                std::cout<<"GPU_BOOT_SSE device="<<dev<<" dense_to_ephemeral_modulus=low q=2 p=2 bits="<<bits<<" ephemeral_h="<<hw<<"\n";
                sse_eval=holder->evaluator.get();
                sse_owner=std::move(holder);
            }
            if(shared&&!shared->boot_gpu_keys_path.empty()) {
                const auto ts=Clock::now();
                detail::save_fast_boot_gpu_keys_file(
                    shared->boot_gpu_keys_path,shared->bundle_fp,*boot_galois,*g6,*sse_dense,*sse_eph);
                std::cout<<"LLAMA_DEV_KEY_CACHE device="<<dev<<" boot_gpu_keys_hit=false saved=true dev_only=true save_seconds="<<seconds(ts)<<"\n";
            }
        }
    }
    std::cout<<"FAST_BOOT_PHASE device="<<dev<<" keys_done="<<seconds(t0)<<"\n";

    const auto [low,lowq]=manual_backend::planned_scale(*r.api,std::exp2(40.),*shared_s2c_qp);
    const auto [c2s_s,cq]=manual_backend::planned_scale(*r.api,raised_scale,*shared_c2s_qp);
    if(lowq!=2||cq!=45)throw std::runtime_error("fast S2C/C2S endpoint mismatch");
    const double q0=double(static_cast<long double>(global.q()[0].value())*global.q()[1].value());
    const double B=std::exp2(std::round(std::log2(q0)));

    // Lazily computed once if either ratio misses the EvalMod public cache.
    bool evalmod_base_ready=false;
    std::unique_ptr<poseidon::EvalModPoly> eval_poly;
    poseidon::Polynomial base_coeffs;
    GpuBootstrapData::EvalModData native_meta;
    GpuEvalModUploadOptions eval_opt;
    eval_opt.dynamic_rescale=true;eval_opt.polynomial_log_split=2;eval_opt.flat_bsgs_b8=false;
    eval_opt.virtual_degree_bound=false;eval_opt.lead_leaf_resplit=false;eval_opt.min_quotient_scale_log2=60;
    auto ensure_evalmod_base=[&](){
        if(evalmod_base_ready)return;
        eval_poly=std::make_unique<poseidon::EvalModPoly>(r.context,poseidon::CosDiscrete,std::exp2(50.),44,5,3,12,0,31);
        eval_poly->enable_full_cosine_coefficients();
        base_coeffs=eval_poly->sine_poly();
        auto a=manual_cosine_coefficients_k12_da3();
        base_coeffs.data().assign(a.begin(),a.end());
        for(auto &c:base_coeffs.data())c*=eval_poly->sqrt_2pi();
        base_coeffs.max_degree()=30;base_coeffs.is_even()=true;base_coeffs.is_odd()=false;
        native_meta=GpuUploader::upload_eval_mod_high_precision(*eval_poly,r.encoder,id45,dev,nullptr,poseidon::parms_id_zero,1,
            &base_coeffs,true,UINT32_MAX,NAN,NAN,true,c2s_s,true,eval_opt);
        if(native_meta.output_q_count!=27)throw std::runtime_error("fast Q27 contract");
        evalmod_base_ready=true;
    };

    const auto build_one=[&](unsigned ratio) -> std::pair<std::unique_ptr<const GpuBootstrapProfile>,manual_backend::ProfileNumbers> {
        GpuBootstrapProfile prof;
        prof.cuda_device_id=dev;prof.input_level_min=5;prof.input_level_max=5;prof.input_components=2;
        prof.input_scale_log2=40;prof.output_level=26;prof.output_scale_log2=50;prof.output_components=2;
        prof.relin_keys=boot_relin;prof.galois_keys=boot_galois;
        auto &d=prof.bootstrap_data;
        d.linear_transform_mode=GpuLinearTransformMode::DoubleHoistBsgs;
        d.allow_environment_linear_transform_override=false;
        d.slot_to_coeff_first=true;d.input_q_count=6;d.input_scale=std::exp2(40.);
        d.raised_scale_override=raised_scale;d.project_real=true;d.output_ratio=ratio;
        d.slot_to_coeff_output_scale=std::exp2(50.);
        d.q0_parms_id=r.context.crt_context()->parms_id_map().at(1);
        d.sparse_switch_to_ephemeral=sse_eph;d.sparse_switch_to_dense=sse_dense;
        d.sparse_low_evaluator=sse_eval;d.sparse_low_owner=sse_owner;

        // Share the single GPU copy of C2S/S2C QP matrices and BSGS group views across both profiles.
        d.shared_coeff_to_slot_matrix_qp=shared_c2s_qp;
        d.shared_slot_to_coeff_matrix_qp=shared_s2c_qp;
        d.shared_coeff_to_slot_groups_qp=shared_c2s_groups;
        d.shared_slot_to_coeff_groups_qp=shared_s2c_groups;

        const double target=std::exp2(std::round(std::log2(q0/ratio)));
        const double m=std::round(target/low),eta=ratio*m*low/B;
        if(m!=(ratio==8192?16.:2.)||!(eta>0&&std::isfinite(eta)))throw std::runtime_error("fast m/eta mismatch");
        d.q0_over_message_ratio=target;

        double alpha=0.0;
        bool evalmod_hit=false;
        const std::filesystem::path em_path=pub_root.empty()
            ?std::filesystem::path()
            :pub_root/("evalmod_r"+std::to_string(ratio)+".bin");
        if(!em_path.empty()) {
            const auto t_em=Clock::now();
            cache::CachedEvalModBundle loaded;
            if(cache::load_eval_mod_bundle(em_path,params_fp,ratio,r.context,dev,loaded)) {
                GpuUploader::prepare_key_views_for_q_counts(*boot_relin,loaded.eval_mod.required_relin_q_counts);
                alpha=loaded.alpha;
                d.output_scale_multiplier=alpha;
                d.slot_to_coeff_input_q_count=27;
                d.slot_to_coeff_input_parms_id=loaded.eval_mod.output_parms_id;
                d.minus_i_plaintext=std::move(loaded.minus_i);
                d.plus_i_plaintext=std::move(loaded.plus_i);
                d.eval_mod=std::move(loaded.eval_mod);
                evalmod_hit=true;
                std::cout<<"LLAMA_PUBLIC_CACHE device="<<dev<<" evalmod_r"<<ratio<<"_hit=true seconds="<<seconds(t_em)<<"\n";
            }
        }
        if(!evalmod_hit) {
            ensure_evalmod_base();
            alpha=std::exp2(50.)/native_meta.output_scale;
            const double seed=std::pow(alpha/eta,1./8);
            auto seeded=base_coeffs;for(auto &c:seeded.data())c*=seed;
            auto candidate=GpuUploader::upload_eval_mod_high_precision(*eval_poly,r.encoder,id45,dev,nullptr,poseidon::parms_id_zero,1,
                &seeded,true,UINT32_MAX,eval_poly->sqrt_2pi()*seed,NAN,true,c2s_s,false,eval_opt);
            if(candidate.output_q_count!=native_meta.output_q_count||!Api::same_scale(candidate.output_scale,native_meta.output_scale))
                throw std::runtime_error("fast seeded EvalMod changed");
            GpuUploader::prepare_key_views_for_q_counts(*boot_relin,candidate.required_relin_q_counts);
            d.output_scale_multiplier=alpha;
            d.slot_to_coeff_input_q_count=27;d.slot_to_coeff_input_parms_id=native_meta.output_parms_id;

            poseidon::Plaintext h_mi,h_pi;
            r.encoder.encode(std::complex<double>(0,-1),id45,1.,h_mi);
            r.encoder.encode(std::complex<double>(0,1),candidate.output_parms_id,1.,h_pi);
            d.minus_i_plaintext=GpuUploader::upload_plaintext(h_mi,dev);
            d.plus_i_plaintext=GpuUploader::upload_plaintext(h_pi,dev);
            d.eval_mod=std::move(candidate);

            if(!em_path.empty()) {
                const auto ts=Clock::now();
                cache::CachedEvalModBundle to_save;
                // Temporarily move out to serialize without duplicating GPU buffers, then move back.
                to_save.eval_mod=std::move(d.eval_mod);
                to_save.minus_i=std::move(d.minus_i_plaintext);
                to_save.plus_i=std::move(d.plus_i_plaintext);
                to_save.alpha=alpha;to_save.eta=eta;to_save.target=target;to_save.m=m;to_save.c2s_scale=c2s_s;
                cache::save_eval_mod_bundle(em_path,params_fp,ratio,to_save);
                d.eval_mod=std::move(to_save.eval_mod);
                d.minus_i_plaintext=std::move(to_save.minus_i);
                d.plus_i_plaintext=std::move(to_save.plus_i);
                std::cout<<"LLAMA_PUBLIC_CACHE device="<<dev<<" evalmod_r"<<ratio<<"_hit=false saved=true save_seconds="<<seconds(ts)<<"\n";
            }
        }
        prof.profile_id="fast-llama-q27-r"+std::to_string(ratio);
        verify_fast_profile(r,prof,ratio,eta,target,alpha);
        return {std::make_unique<const GpuBootstrapProfile>(std::move(prof)),manual_backend::ProfileNumbers{ratio,eta,target,alpha,m,c2s_s}};
    };
    auto out=std::array{build_one(8192),build_one(65536)};
    gpu_check_cuda(cudaSetDevice(dev),"fast profile pair sync device");
    gpu_check_cuda(cudaDeviceSynchronize(),"fast profile pair ready");
    std::cout<<"FAST_BOOT_PAIR device="<<dev<<" setup_seconds="<<seconds(t0)<<" per_level_P=true shared_keys=true shared_c2s_s2c=true\n";
    return out;
}
} // namespace cerium_source::fast_boot
