// Restore the historical fixed-2^45 StC-first degree59/DA2 contract, using
// existing double-hoist/EvalMod APIs and a real nonzero random secret.
// This does NOT approve security or fold the native output to scale 2^40.
#define POSEIDON_BOOTSTRAP_PRECISION_LIBRARY
#include "bootstrap_precision.cpp"
#include <chrono>
#include <functional>

// Diagnostic-only continuation hooks. The standalone baseline supplies no
// hooks and retains its native output. Continuations receive ciphertexts,
// never an encryptor or secret key, and can run only after the baseline guard.
struct BaselineMetadata {
    const PoseidonContext &ctx;
    CKKSEncoder &encoder;
    const EvalModPoly &poly;
    parms_id_type c2s_id;
    double c2s_scale;
    const GpuBootstrapData::EvalModData &native_plan;
    // LCDNN's imaginary-removing bootstrap preserves the complete complex
    // bootstrap circuit, then projects its output to Re(x) without consuming
    // a Q level. It does not remove the imaginary EvalMod branch.
    bool output_real_projection;
};
using RotationKeyFactory=std::function<GpuGaloisKeysData(const std::vector<int> &)>;
using ApplicationInputEncryptor=std::function<CT(const std::vector<double> &)>;

ParametersLiteral make_application_keyswitch_parameters(
    const ParametersLiteral &global_parameters,
    std::size_t q_count,
    std::size_t p_count) {
    if(q_count<2 || q_count>global_parameters.q().size() ||
        p_count<2 || p_count>q_count || p_count>global_parameters.p().size())
        throw std::invalid_argument(
            "application KeySwitch requires 2 <= P <= Q and P <= global P");
    ParametersLiteral parameters=global_parameters;
    parameters.set_modulus(
        std::vector<Modulus>(global_parameters.q().begin(),
            global_parameters.q().begin()+static_cast<std::ptrdiff_t>(q_count)),
        std::vector<Modulus>(global_parameters.p().begin(),
            global_parameters.p().begin()+static_cast<std::ptrdiff_t>(p_count)));
    return parameters;
}

SecretKey make_application_keyswitch_secret(
    const KeyGenerator &global_keygen,
    const ParametersLiteral &global_parameters,
    const PoseidonContext &level_context,
    std::size_t q_count,
    std::size_t p_count) {
    const std::size_t degree=global_parameters.degree();
    const std::size_t global_q_count=global_parameters.q().size();
    const std::size_t global_p_count=global_parameters.p().size();
    const auto &source=global_keygen.secret_key().data();
    if(source.coeff_count()!=degree*(global_q_count+global_p_count))
        throw std::logic_error("global secret-key storage does not match Q/P parameters");
    SecretKey result;
    result.data().resize(level_context,level_context.crt_context()->key_parms_id(),
        degree*(q_count+p_count));
    std::copy_n(source.data(),degree*q_count,result.data().data());
    std::copy_n(source.data()+degree*global_q_count,degree*p_count,
        result.data().data()+degree*q_count);
    return result;
}

// Stateless application-side KeySwitch material. Each instance reuses the
// global Q/P prime prefixes and exact secret polynomial, but owns only the P
// prefix required by its actual Q level. It contains public GPU material only
// after construction; the compatible secret never leaves this factory call.
struct ApplicationKeySwitchRuntime {
    ApplicationKeySwitchRuntime(const ParametersLiteral &global_parameters,
        const KeyGenerator &global_keygen,std::size_t q,std::size_t p,
        const std::vector<std::size_t> &active_q_counts,
        const std::vector<int> &rotation_steps,int device_id,
        bool create_relinearization_key)
        :q_count(q),p_count(p),effective_dnum((q+p-1)/p),
        parameters(make_application_keyswitch_parameters(global_parameters,q,p)),
        context(parameters) {
        auto compatible_secret=make_application_keyswitch_secret(global_keygen,
            global_parameters,context,q,p);
        KeyGenerator level_keygen(context,compatible_secret);
        gpu_parameters=std::make_unique<GpuParameterData>(
            context,device_id,active_q_counts);
        if(gpu_parameters->level_count()!=active_q_counts.size())
            throw std::logic_error("sparse application parameter level count changed");
        gpu_evaluator=std::make_unique<GpuEvaluator>(*gpu_parameters);
        if(create_relinearization_key) {
            RelinKeys host;level_keygen.create_relin_keys(host);
            gpu_relin_keys=GpuUploader::upload_relin_keys(host,device_id);
        }
        if(!rotation_steps.empty()) {
            GaloisKeys host;level_keygen.create_galois_keys(rotation_steps,host);
            gpu_galois_keys=GpuUploader::upload_galois_keys(host,device_id);
        }
    }
    std::size_t q_count,p_count,effective_dnum;
    ParametersLiteral parameters;
    PoseidonContext context;
    std::unique_ptr<GpuParameterData> gpu_parameters;
    std::unique_ptr<GpuEvaluator> gpu_evaluator;
    GpuRelinKeysData gpu_relin_keys;
    GpuGaloisKeysData gpu_galois_keys;
    GpuDoubleHoistWorkspace rotate_many_workspace;
};
using ApplicationKeySwitchFactory=std::function<std::unique_ptr<ApplicationKeySwitchRuntime>(
    std::size_t,std::size_t,const std::vector<std::size_t> &,
    const std::vector<int> &,bool)>;

struct BaselineContinuation {
    const BaselineMetadata &metadata;
    GpuEvaluator &eval;
    GpuRelinKeysData &keys;
    Replay &replay;
    const CT &real, &imag, &native_output;
    const Values &input, &noisy_input, &bootstrap_polynomial;
    // Public evaluation keys only; does not expose an encryptor or secret.
    RotationKeyFactory rotation_keys,bootstrap_rotation_keys;
};
struct BaselineSession {
    const BaselineMetadata &metadata;
    GpuEvaluator &eval;
    GpuRelinKeysData &keys;
    Replay &replay;
    // This encryptor is deliberately limited to real, bounded application
    // input slots. It exists for the encrypted ResNet stem only; callers do
    // not receive the secret key and cannot re-encrypt intermediate values.
    ApplicationInputEncryptor encrypt_application_input;
    RotationKeyFactory rotation_keys,bootstrap_rotation_keys;
    ApplicationKeySwitchFactory application_keyswitch;
};

const std::vector<std::uint32_t> historical_q_bits = {
    32,32,32,32,32,32,32,32,32,32,32,32,28,28,31,31,32,
    32,30,31,32,31,32,32,31,31,31,32,32,31,32,32,32,30};

// Q50 double-hoist matrices/scratch need more room than the earlier classic
// BSGS diagnostic. Keep an explicit cap and check an additional 8 GiB margin.
// Do not change the shared ReLU/old failed-bootstrap diagnostic pool policy.
struct BaselinePool {
    rmm::mr::cuda_memory_resource upstream;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool;
    rmm::mr::device_memory_resource *previous;
    explicit BaselinePool(std::size_t cap,std::size_t initial=1<<20)
        : pool(&upstream,initial,cap), previous(rmm::mr::get_current_device_resource()) {
        rmm::mr::set_current_device_resource(&pool);
    }
    ~BaselinePool() { rmm::mr::set_current_device_resource(previous); }
};

ParametersLiteral baseline_parameters(const std::string &profile, const std::string &fixture) {
    ParametersLiteral p(CKKS,16,15,40,192,1,Modulus(0),{}, {},sec_level_type::none);
    if (profile=="historical-q34") {
        // Same constructor and prime generation as make_test_parameters(),
        // except h=192. Never call the historical h=0 test entry point.
        p.set_log_modulus(historical_q_bits,std::vector<std::uint32_t>(9,32));
    } else if (profile=="q50-fixed45") {
        std::ifstream in(fixture); std::string magic; int nq=0,np=0; in>>magic>>nq>>np;
        if (magic!="RELU_PRECISION_V1" || nq!=50 || np!=25) throw std::runtime_error("Q50 fixture required");
        std::vector<Modulus> q,aux;
        for (int i=0;i<nq+np;++i) {
            std::uint64_t v=0;
            if (!(in>>v)) throw std::runtime_error("truncated fixture primes");
            (i<nq?q:aux).emplace_back(v);
        }
        p.set_modulus(q,aux);
    } else throw std::runtime_error("unknown baseline profile");
    return p;
}

void configure_evalmod_baseline() {
    setenv("POSEIDON_BOOTSTRAP_EVALMOD_DYNAMIC_RESCALE","1",1);
    setenv("POSEIDON_EVALMOD_LOG_SPLIT","3",1);
    setenv("POSEIDON_EVALMOD_FLAT_BSGS_B8","0",1);
    setenv("POSEIDON_EVALMOD_VIRTUAL_DEGREE_BOUND","0",1);
    setenv("POSEIDON_EVALMOD_LEAD_LEAF_RESPLIT","0",1);
}

void print_rescale_trace(const std::string &name, const GpuDoubleHoistWorkspace &w) {
    int i=0;
    for (const auto &s:w.matrix_rescale_trace) {
        std::cout<<"TRACE "<<name<<" stage="<<++i<<" q_in="<<s.input_q_count<<" q_out="<<s.output_q_count
            <<" input_log_scale="<<std::log2(s.input_scale)<<" output_log_scale="<<std::log2(s.output_scale)<<'\n';
    }
}

#include "bootstrap_ciphertext_stage.h"

int run_bootstrap_baseline(int argc, char **argv,
    const std::function<void(const BaselineMetadata &)> &on_metadata={},
    const std::function<int(const BaselineContinuation &)> &continuation={},
    const std::vector<std::complex<double>> *application_input=nullptr,
    bool output_real_projection=false,
    const std::function<int(const BaselineSession &)> &session_continuation={}) {
    try {
        if (argc<4 || argc>5) throw std::runtime_error("usage: bootstrap_baseline_precision --unsafe-accuracy-only|--unsafe-performance-only|--metadata-only historical-q34|q50-fixed45 historical|grid [q50_fixture]");
        const std::string mode=argv[1], profile=argv[2], dataset=argv[3];
        const bool metadata=mode=="--metadata-only";
        const bool performance=mode=="--unsafe-performance-only";
        const bool continuous=performance;
        if (!metadata && mode!="--unsafe-accuracy-only" && !performance)
            throw std::runtime_error("explicit accuracy-only or performance-only flag required");
        if (dataset!="historical" && dataset!="grid" && dataset!="grid-near-zero") throw std::runtime_error("unknown dataset");
        if (application_input) {
            if (application_input->size()!=32768) throw std::runtime_error("application input must contain 32768 slots");
            for (auto x:*application_input)
                if (!std::isfinite(x.real()) || x.imag()!=0 || std::abs(x)>1)
                    throw std::runtime_error("application input must be real, finite and within [-1,1]");
            std::cout<<"INPUT_OVERRIDE application_packed_slots=true encrypted_once=true\n";
        }
        if (output_real_projection && !application_input && !session_continuation)
            throw std::runtime_error("LCDNN real projection requires a real application input");
        auto parms=baseline_parameters(profile,argc==5?argv[4]:"");
        const auto nq=parms.q().size(), np=parms.p().size();
        std::cout<<std::unitbuf<<std::setprecision(16)
            <<"BASELINE profile="<<profile<<" dataset="<<dataset<<" N=65536 Q="<<nq<<" P="<<np
            <<" global_dnum="<<(nq+np-1)/np<<" secret_hamming_weight=192 degree=59 DA=2 arcsine_degree=0"
            <<" output_fold=false c2s_log_scale=45 linear_mode=double_hoist metadata_only="<<metadata<<'\n';
        std::cout<<"BOOTSTRAP_BRANCH mode="<<(output_real_projection?"lcdnn_s2c_first_real":"generic_complex")
            <<" active_log_slots=15 evalmod_calls=2"
            <<" output_real_projection="<<(output_real_projection?"true":"false")<<'\n';
        std::cout<<"Q_PRIMES"; for (auto &q:parms.q()) std::cout<<' '<<q.value(); std::cout<<'\n';
        std::cout<<"P_PRIMES"; for (auto &p:parms.p()) std::cout<<' '<<p.value(); std::cout<<'\n';
        configure_evalmod_baseline();
        PoseidonFactory::get_instance()->set_device_type(DEVICE_SOFTWARE);
        auto ctx=PoseidonFactory::get_instance()->create_poseidon_context(parms);
        CKKSEncoder encoder(ctx);
        const double working_scale=std::exp2(45);
        EvalModPoly poly(ctx,CosDiscrete,working_scale,0,5,2,25,0,59);
        const double c2s_scaling=poly.q_div()/(poly.k()*poly.sc_fac()*poly.q_diff());
        // Exact historical q0_over_message_ratio() policy: round the target
        // to a power of two, then approximate it with an integer multiplier.
        const double preparation_target=std::exp2(std::round(std::log2(ctx.crt_context()->q0()/32)));
        const double periods=25*ctx.crt_context()->q0()*c2s_scaling/working_scale;
        if (std::abs(periods-1)>1e-12) throw std::runtime_error("integer-period contract violated");
        std::cout<<"CONTRACT preparation_target_log_scale="<<std::log2(preparation_target)<<" c2s_log_scale=45 periods_per_integer="<<periods<<'\n';
        std::size_t predicted_q=nq; double predicted_scale=working_scale;
        for (int stage=0;stage<3;++stage) {
            auto before=predicted_q; predicted_scale*=working_scale;
            while (predicted_q>1 && predicted_scale/parms.q()[predicted_q-1].value()>=(working_scale+1)/2) {
                predicted_scale/=parms.q()[--predicted_q].value();
            }
            std::cout<<"PLAN C2S stage="<<stage+1<<" q_in="<<before<<" q_out="<<predicted_q<<" output_log_scale="<<std::log2(predicted_scale)<<'\n';
        }
        poly.set_level_start(predicted_q-1);
        const auto c2s_id=ctx.crt_context()->parms_id_map().at(predicted_q-1);
        auto plan=GpuUploader::upload_eval_mod_high_precision(poly,encoder,c2s_id,0,nullptr,parms_id_zero,1,nullptr,true,
            std::numeric_limits<std::uint32_t>::max(),NAN,NAN,true,predicted_scale,true);
        std::cout<<"PLAN EvalMod q_in="<<predicted_q<<" q_out="<<plan.output_q_count<<" output_log_scale="<<std::log2(plan.output_scale)
            <<" effective_degree="<<plan.polynomial_degree<<'\n';
        if (predicted_q!=nq-4 || plan.output_q_count!=nq-19) throw std::runtime_error("baseline Q schedule differs from historical 4+15");
        const BaselineMetadata metadata_state{
            ctx,encoder,poly,c2s_id,predicted_scale,plan,output_real_projection};
        if (on_metadata) on_metadata(metadata_state);
        if (metadata) { std::cout<<"RESULT metadata=PASS keys_generated=false gpu_executed=false\n"; return 0; }

        gpu_check_cuda(cudaSetDevice(0),"cudaSetDevice");
        std::size_t free,total; gpu_check_cuda(cudaMemGetInfo(&free,&total),"cudaMemGetInfo");
        const std::size_t pool_gib=continuous?30:(nq==50?18:12);
        const std::size_t initial_pool_gib=continuous?pool_gib:0;
        const std::size_t required_free=continuous
            ? (pool_gib<<30)+(std::size_t(1)<<30)
            : ((pool_gib+8)<<30);
        if (free<required_free) throw std::runtime_error(continuous
            ? "continuous resident mode needs a 30 GiB pool plus 1 GiB free headroom"
            : "need pool cap plus 8 GiB free headroom");
        std::cout<<"MEMORY pool_cap_GiB="<<pool_gib
            <<" pool_initial_GiB="<<initial_pool_gib
            <<" startup_free_GiB="<<double(free)/(1ULL<<30)<<'\n';
        BaselinePool pool(pool_gib<<30,continuous?(pool_gib<<30):(1<<20));
        auto secret=nonzero_secret(ctx); KeyGenerator generator(ctx,secret);
        PublicKey pub; generator.create_public_key(pub); RelinKeys relin; generator.create_relin_keys(relin);
        Encryptor encryptor(ctx,pub,secret); Decryptor decryptor(ctx,secret);
        GpuParameterData parameters(ctx,0); GpuEvaluator eval(parameters);
        auto keys=GpuUploader::upload_relin_keys(relin,0);
        Replay replay(ctx,encoder,decryptor,parameters,keys,!performance);
        // Cache host public evaluation keys. The session continuation decides
        // whether GPU key uploads have stage-local or whole-graph lifetime.
        std::map<std::vector<int>,GaloisKeys> host_rotation_cache;
        auto upload_keys=[&](const std::vector<int> &steps,bool double_hoist) {
            for(int step:steps)
                if(step<0 || step>=32768 || (!double_hoist && step==0))
                    throw std::runtime_error("invalid public rotation step");
            auto [it,inserted]=host_rotation_cache.try_emplace(steps);
            if(inserted) {
                std::vector<std::uint32_t> elements;
                for(int step:steps)elements.push_back(ctx.crt_context()->galois_tool()->get_elt_from_step(step));
                generator.create_galois_keys(elements,it->second);
            }
            std::cout<<"PUBLIC_KEYS count="<<steps.size()<<" host_cache_hit="<<(!inserted)
                <<" double_hoist="<<double_hoist<<'\n';
            return double_hoist?GpuUploader::upload_double_hoist_galois_keys(it->second,0)
                :GpuUploader::upload_galois_keys(it->second,0);
        };
        RotationKeyFactory direct=[&](const auto &steps){return upload_keys(steps,false);};
        RotationKeyFactory boot=[&](const auto &steps){return upload_keys(steps,true);};
        ApplicationKeySwitchFactory application_keyswitch=
            [&](std::size_t q_count,std::size_t p_count,
                const std::vector<std::size_t> &active_q_counts,
                const std::vector<int> &steps,bool create_relin) {
                return std::make_unique<ApplicationKeySwitchRuntime>(parms,generator,
                    q_count,p_count,active_q_counts,steps,0,create_relin);
            };
        ApplicationInputEncryptor encrypt_application_input=[&](const std::vector<double> &slots) {
            if(slots.size()!=32768)throw std::runtime_error("application encryption requires 32768 real slots");
            for(double value:slots)
                if(!std::isfinite(value) || std::abs(value)>1)
                    throw std::runtime_error("application encryption requires finite slots within [-1,1]");
            // The encrypted stem consumes exactly one Q prime and its ReLU
            // requires Q31. Start each prepared image patch directly at Q32;
            // carrying it through Q50 only to discard Q49..Q31 after the stem
            // wastes limbs without contributing multiplicative depth.
            const auto input_id=ctx.crt_context()->parms_id_map().at(31);
            Plaintext plain;encoder.encode(slots,input_id,std::exp2(40),plain);
            Ciphertext encrypted;encryptor.encrypt(plain,encrypted);
            if(encrypted.coeff_modulus_size()!=32)
                throw std::logic_error("application input encryption did not produce Q32");
            return GpuUploader::upload_ciphertext(encrypted,0);
        };
        if(session_continuation)
            return session_continuation(BaselineSession{metadata_state,eval,keys,replay,
                encrypt_application_input,direct,boot,application_keyswitch});

        std::vector<std::complex<double>> message(32768);
        for (std::size_t i=0;i<message.size();++i)
            message[i]=dataset=="historical"
                ? std::complex<double>(double(i%17+1)/32,double(i%11+1)/64)
                : std::complex<double>(-0.5+double(i)/(message.size()-1),0);
        if (dataset!="historical") {message[0]=0;message[1]=-0.5;message[2]=0.5;}
        if (dataset=="grid-near-zero")
            for (int i=3;i<67;++i) message[i]=(i%2?1:-1)*std::exp2(-1-(i-3)/2);
        if (application_input) message=*application_input;
        Values expected(message.begin(),message.end());
        Plaintext plain; encoder.encode(message,std::exp2(40),plain);
        Ciphertext source; encryptor.encrypt(plain,source);
        auto full_input=GpuUploader::upload_ciphertext(source,0); CT input;
        eval.drop_modulus(full_input,input,parameters.get_level_by_q_count(6).parms_id);
        replay.check("baseline_input_Q6",input,expected);
        const auto noisy_input=decoded_values(replay,input);
        full_input=CT{};
        return bootstrap_existing_ciphertext(metadata_state,eval,keys,replay,input,expected,direct,boot,continuation);
    } catch (const std::exception &e) { std::cerr<<"ERROR "<<e.what()<<std::endl; return 2; }
}

#ifndef POSEIDON_BOOTSTRAP_BASELINE_LIBRARY
int main(int argc, char **argv) { return run_bootstrap_baseline(argc,argv); }
#endif
