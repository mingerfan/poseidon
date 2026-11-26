// Common benchmark setup. Only the caller owns the secret key; Api does not.
#pragma once
#include "source_ops.hpp"
#include "source_cache.hpp"
#include "poseidon/keygenerator.h"
#include "poseidon/encryptor.h"
#include "poseidon/decryptor.h"
#include <chrono>
#include <fstream>
#include <memory>
#include <set>

namespace cerium_source {
using Clock=std::chrono::steady_clock;
inline double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
// Validate before PoseidonContext/KeyGenerator/GPU tables are constructed.
inline const poseidon::ParametersLiteral &require_cerium_parameters(const poseidon::ParametersLiteral &literal) {
    layout::require_ring(literal.log_n(),literal.log_slots());
    if(literal.scheme()!=CKKS)throw std::invalid_argument("Cerium Llama requires CKKS");
    return literal;
}
inline poseidon::ParametersLiteral parameters_from_fixture(const char *filename) {
    std::ifstream f(filename);std::string magic;int nq=0,np=0;
    if(!(f>>magic>>nq>>np)||magic!="RELU_PRECISION_V1"||nq!=50||np!=25)
        throw std::runtime_error("fixture header");
    std::vector<poseidon::Modulus> q,p;
    for(int i=0;i<nq+np;++i) {
        std::uint64_t prime=0;if(!(f>>prime))throw std::runtime_error("fixture truncated");
        if(i!=8)(i<nq?q:p).emplace_back(prime);
    }
    poseidon::ParametersLiteral literal(CKKS,layout::log_ring_degree,layout::log_slots,40,32768,1,
        poseidon::Modulus(0),{},{},poseidon::sec_level_type::none);
    literal.set_modulus(q,p);return literal;
}
inline std::vector<int> source_rotations() {
    std::set<int> steps{0};
    auto add=[&](int s){s=(s%32768+32768)%32768;if(s>=16384)s-=32768;steps.insert(s);};
    for(int s=1;s<256;s*=2){add(s);add(-s);}
    for(int b=0;b<16;++b)add(b);
    for(int g=-128;g<128;g+=16)add(g);
    for(int b=0;b<8;++b){add(256*b);add(-b);add(b-128);}
    for(int g=0;g<16;++g){add(2048*g);add(-2048*g);add(2040*g);add(-2040*g);}
    for(int s:{-129,-127,127,129})add(s);
    return {steps.begin(),steps.end()};
}
struct HostRuntimeKeys {
    poseidon::PoseidonContext context;
    poseidon::SecretKey secret_key;
    poseidon::PublicKey public_key;
    poseidon::RelinKeys relin;
    poseidon::GaloisKeys galois;
    bool use_double_hoist_galois=false;
    bool loaded_from_dev_cache=false;
    HostRuntimeKeys(const poseidon::PoseidonContext &ctx,const poseidon::KeyGenerator &kg,
                    const std::vector<int> &steps)
        :context(ctx),secret_key(kg.secret_key()) {
        const auto start=Clock::now();
        std::cout<<"SOURCE_SETUP stage=cerium_geometry N="<<context.parameters_literal()->degree()
                 <<" complex_slots="<<(context.parameters_literal()->degree()/2)
                 <<" token_capacity="<<layout::token_capacity
                 <<" feature_tile="<<layout::feature_tile<<" layout_contract_version="<<layout::contract_version<<std::endl;
        std::cout<<"SOURCE_SETUP stage=public_and_rotation_keys"<<std::endl;
        const auto &secret=secret_key.data();
        if(std::all_of(secret.data(),secret.data()+layout::ring_degree,[](auto v){return v==0;}))throw std::runtime_error("zero secret");
        kg.create_public_key(public_key);
        const_cast<poseidon::KeyGenerator&>(kg).create_relin_keys(relin);
        const char *fp=std::getenv("LLAMA_FAST_P");const bool fast_p=fp&&std::string(fp)=="1";
        const char *fl=std::getenv("LLAMA_FAST_LINEAR");const bool fast_l=fl&&*fl&&std::string(fl)!="0";
        const std::vector<int> main_steps=fast_p?std::vector<int>{0}:steps;
        const_cast<poseidon::KeyGenerator&>(kg).create_galois_keys(main_steps,galois);
        use_double_hoist_galois=(fast_p||fast_l);
        std::cout<<"SOURCE_SETUP stage=host_keys_ready seconds="<<seconds(start)<<std::endl;
    }
private:
    HostRuntimeKeys(const poseidon::PoseidonContext &ctx,
                    poseidon::SecretKey sk,
                    poseidon::PublicKey pk,
                    poseidon::RelinKeys rk,
                    poseidon::GaloisKeys gk,
                    bool dh)
        :context(ctx),secret_key(std::move(sk)),public_key(std::move(pk)),
         relin(std::move(rk)),galois(std::move(gk)),use_double_hoist_galois(dh),loaded_from_dev_cache(true) {}
public:
    static HostRuntimeKeys create(const poseidon::ParametersLiteral &literal,
                                  const std::vector<int> &steps=source_rotations(),
                                  const std::string &dev_key_cache_override={}) {
        const auto t_ctx=Clock::now();
        poseidon::PoseidonContext ctx(require_cerium_parameters(literal));
        std::cout<<"SOURCE_SETUP stage=context_ready seconds="<<seconds(t_ctx)<<std::endl;
        const std::string cache_root=cache::dev_key_cache_dir(dev_key_cache_override);
        const char *fp_env=std::getenv("LLAMA_FAST_P");const bool fast_p=fp_env&&std::string(fp_env)=="1";
        const char *fl_env=std::getenv("LLAMA_FAST_LINEAR");const bool fast_l=fl_env&&*fl_env&&std::string(fl_env)!="0";
        const char *fe_env=std::getenv("LLAMA_FAST_EVALMOD_P");const bool fast_evalmod_p=fe_env&&*fe_env&&std::string(fe_env)=="1";
        const std::vector<int> main_steps=fast_p?std::vector<int>{0}:steps;
        if(!cache_root.empty()) {
            const auto ns=cache::dev_key_namespace_key(literal,steps,fast_p,fast_evalmod_p,32U);
            cache::Hasher kh;
            kh.str("LLAMA_HOST_KEYS_V1");
            kh.pod(cache::parameters_fingerprint(literal));
            kh.pod(static_cast<std::uint8_t>(fast_p?1:0));
            kh.pod(static_cast<std::uint64_t>(main_steps.size()));
            for(int s:main_steps)kh.pod(static_cast<std::int32_t>(s));
            const std::uint64_t host_fp=kh.finish();
            const auto path=std::filesystem::path(cache_root)/ns/"host_keys.bin";
            const auto t0=Clock::now();
            poseidon::SecretKey sk;poseidon::PublicKey pk;poseidon::RelinKeys rk;poseidon::GaloisKeys gk;
            if(cache::load_host_keys_file(path,host_fp,ctx,sk,pk,rk,gk)) {
                const auto &secret=sk.data();
                if(std::all_of(secret.data(),secret.data()+layout::ring_degree,[](auto v){return v==0;}))
                    throw std::runtime_error("cached dev secret key is zero");
                std::cout<<"LLAMA_DEV_KEY_CACHE hit=true dev_only=true path="<<path.string()
                         <<" seconds="<<seconds(t0)<<std::endl;
                return HostRuntimeKeys(ctx,std::move(sk),std::move(pk),std::move(rk),std::move(gk),fast_p||fast_l);
            }
            poseidon::KeyGenerator kg(ctx);
            HostRuntimeKeys fresh(ctx,kg,steps);
            const auto ts=Clock::now();
            cache::save_host_keys_file(path,host_fp,fresh.secret_key,fresh.public_key,fresh.relin,fresh.galois);
            std::cout<<"LLAMA_DEV_KEY_CACHE hit=false saved=true dev_only=true path="<<path.string()
                     <<" save_seconds="<<seconds(ts)<<std::endl;
            return fresh;
        }
        poseidon::KeyGenerator kg(ctx);
        return HostRuntimeKeys(ctx,kg,steps);
    }
};
inline std::vector<std::size_t> fast_runtime_global_q_counts(std::size_t global_q) {
    const char *fe=std::getenv("LLAMA_FAST_EVALMOD_P");
    const bool fast_evalmod_p=fe&&*fe&&std::string(fe)=="1";
    const std::size_t first_global=fast_evalmod_p?45:28;
    std::vector<std::size_t> qs;
    qs.reserve(global_q);
    if(global_q>=1)qs.push_back(1);
    if(global_q>=2)qs.push_back(2);
    for(std::size_t q=first_global;q<=global_q;++q)qs.push_back(q);
    return qs;
}
inline poseidon::gpu::GpuParameterData make_runtime_parameters(
    const poseidon::PoseidonContext &ctx,int dev,bool defer_fast_p_levels) {
    const auto t0=Clock::now();
    const std::size_t global_q=ctx.parameters_literal()->q().size();
    poseidon::gpu::GpuParameterData params = (defer_fast_p_levels&&global_q==49)
        ?poseidon::gpu::GpuParameterData(ctx,dev,fast_runtime_global_q_counts(global_q),true)
        :poseidon::gpu::GpuParameterData(ctx,dev);
    std::cout<<"SOURCE_SETUP stage=gpu_parameters_ready device="<<dev
             <<" levels="<<params.level_count()
             <<" gib="<<(double(params.allocated_bytes())/double(1ULL<<30))
             <<" sparse_defer="<<(defer_fast_p_levels&&global_q==49?"true":"false")
             <<" seconds="<<seconds(t0)<<std::endl;
    return params;
}
struct Runtime {
    int device_id=0;
    poseidon::PoseidonContext context;
    poseidon::KeyGenerator keys;
    poseidon::CKKSEncoder encoder;
    poseidon::gpu::GpuParameterData parameters;
    poseidon::PublicKey public_key;
    poseidon::gpu::GpuRelinKeysData relin;
    poseidon::gpu::GpuGaloisKeysData galois;
    poseidon::gpu::GpuBootstrapProfile profile;
    poseidon::gpu::GpuBootstrapWorkspace workspace;
    std::unique_ptr<Api> api;
    // Operator experiments may explicitly skip Boot setup and request only
    // their direct rotation keys. The model keeps the original full setup.
    Runtime(const char *fixture,bool with_boot=true,
            const std::vector<int> &steps=source_rotations(),int dev=0,bool defer_fast_p_levels=false)
        :Runtime(parameters_from_fixture(fixture),with_boot,steps,dev,defer_fast_p_levels) {}
    // Explicit parameter construction is for separately labelled FHE control
    // experiments. The fixture/model constructor above keeps Q49/P25 intact.
    Runtime(const poseidon::ParametersLiteral &literal,bool with_boot,
            const std::vector<int> &steps,int dev=0,bool defer_fast_p_levels=false)
        :device_id(dev),context(require_cerium_parameters(literal)),keys(context),encoder(context),
         parameters(make_runtime_parameters(context,dev,defer_fast_p_levels&&!with_boot)) {
        const auto start=Clock::now();
        std::cout<<"SOURCE_SETUP stage=cerium_geometry device="<<dev<<" N="<<context.parameters_literal()->degree()
                 <<" complex_slots="<<encoder.slot_count()<<" token_capacity="<<layout::token_capacity
                 <<" feature_tile="<<layout::feature_tile<<" layout_contract_version="<<layout::contract_version<<std::endl;
        std::cout<<"SOURCE_SETUP stage=public_and_rotation_keys"<<std::endl;
        const auto &secret=keys.secret_key().data();
        if(std::all_of(secret.data(),secret.data()+layout::ring_degree,[](auto v){return v==0;}))throw std::runtime_error("zero secret");
        keys.create_public_key(public_key);
        {
            poseidon::RelinKeys host;keys.create_relin_keys(host);
            relin=poseidon::gpu::GpuUploader::upload_relin_keys(host,dev);
        }
        {
            const char *fp=std::getenv("LLAMA_FAST_P");const bool fast_p=fp&&std::string(fp)=="1";
            const char *fl=std::getenv("LLAMA_FAST_LINEAR");const bool fast_l=fl&&*fl&&std::string(fl)!="0";
            const std::vector<int> main_steps=fast_p?std::vector<int>{0}:steps;
            poseidon::GaloisKeys host;keys.create_galois_keys(main_steps,host);
            galois=(fast_p||fast_l)?poseidon::gpu::GpuUploader::upload_double_hoist_galois_keys(host,dev)
                                   :poseidon::gpu::GpuUploader::upload_galois_keys(host,dev);
        }
        api=std::make_unique<Api>(context,parameters,encoder,relin,galois);
        if(with_boot) {
        std::cout<<"SOURCE_SETUP stage=native_bootstrap_profile application_keys="<<steps.size()
                 <<" elapsed_seconds="<<seconds(start)<<std::endl;
        poseidon::gpu::GpuResNet20ManualOptions options;
        options.sine_degree=22;options.double_angle=3;options.k=12;options.log_split=2;
        options.full_coefficients=true;options.sparse_secret_hamming_weight=32;
        options.level_special_modulus=poseidon::gpu::GpuResNet20ManualOptions::LevelSpecialModulus::Global;
        profile=poseidon::gpu::GpuBootstrapProfileBuilder::build_resnet20_manual(context,keys,dev,options);
        for(auto &level:profile.level_parameters)parameters.adopt_level(std::move(level));
        profile.level_parameters.clear();
        api->bind_boot(profile,workspace);
        }
        poseidon::gpu::gpu_check_cuda(cudaSetDevice(dev),"source setup set device");
        poseidon::gpu::gpu_check_cuda(cudaDeviceSynchronize(),"source setup complete");
        std::cout<<"SOURCE_SETUP stage=completed device="<<dev<<" seconds="<<seconds(start)<<std::endl;
    }
    Runtime(const HostRuntimeKeys &shared,int dev,bool defer_fast_p_levels=false)
        :device_id(dev),context(shared.context),keys(context,shared.secret_key),encoder(context),
         parameters(make_runtime_parameters(context,dev,defer_fast_p_levels)),public_key(shared.public_key) {
        const auto start=Clock::now();
        poseidon::gpu::gpu_check_cuda(cudaSetDevice(dev),"runtime shared setup set device");
        relin=poseidon::gpu::GpuUploader::upload_relin_keys(shared.relin,dev);
        galois=shared.use_double_hoist_galois
            ?poseidon::gpu::GpuUploader::upload_double_hoist_galois_keys(shared.galois,dev)
            :poseidon::gpu::GpuUploader::upload_galois_keys(shared.galois,dev);
        api=std::make_unique<Api>(context,parameters,encoder,relin,galois);
        std::cout<<"SOURCE_SETUP stage=uploaded_shared device="<<dev<<" seconds="<<seconds(start)<<std::endl;
    }
    CT encrypt(const std::vector<double> &values,std::size_t q,double scale=std::exp2(40.)) {
        poseidon::Plaintext plain;encoder.encode(values,parameters.get_level_by_q_count(q).parms_id,scale,plain);
        poseidon::Encryptor encryptor(context,public_key);poseidon::Ciphertext host;encryptor.encrypt(plain,host);
        return poseidon::gpu::GpuUploader::upload_ciphertext(host,device_id);
    }
    std::vector<std::complex<double>> decrypt_final(const CT &x) {
        poseidon::Ciphertext host;poseidon::gpu::GpuUploader::download_ciphertext(x,host,context);
        poseidon::Decryptor decryptor(context,keys.secret_key());poseidon::Plaintext plain;decryptor.decrypt(host,plain);
        std::vector<std::complex<double>> values;encoder.decode(plain,values);return values;
    }
};
} // namespace cerium_source
