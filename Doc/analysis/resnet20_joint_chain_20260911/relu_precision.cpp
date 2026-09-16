// Isolated accuracy diagnostic. UNSAFE parameters; never a production entry.
// Uses a real OS-random, nonzero secret and the original application ReLU.
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/basics/randomtostd.h"
#include "poseidon/basics/util/ntt.h"
#include "poseidon/basics/util/common.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "secure_sparse_secret.h"
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>

using namespace poseidon;
using namespace poseidon::gpu;
using CT = GpuCiphertextData;
using Ref = std::vector<long double>;
using Complex = std::complex<long double>;
double original_application_relu(double x);
constexpr int entry_q = 31;

struct Node {
    std::string type;
    int work, output, degree = 0, split = 0;
    double pre, scale;
    std::vector<double> coefficients;
    std::unique_ptr<Node> quotient, remainder;
};
struct Basis { int degree, work, output; double pre, scale; };
struct Stage {
    int degree, start, end;
    double input_scale, output_scale;
    std::vector<Basis> basis;
    std::unique_ptr<Node> tree;
};
std::unique_ptr<Node> read_node(std::istream &in) {
    auto n = std::make_unique<Node>();
    in >> n->type >> n->work >> n->output >> n->pre >> n->scale;
    if (n->type == "leaf") {
        in >> n->degree;
        if (n->degree < 1 || n->degree > 27 || n->degree%2 != 1)
            throw std::runtime_error("invalid leaf degree");
        n->coefficients.resize((n->degree+1)/2);
        for (auto &v : n->coefficients) in >> v;
    } else if (n->type == "combine") {
        in >> n->split;
        n->quotient = read_node(in); n->remainder = read_node(in);
    } else throw std::runtime_error("invalid node type");
    return n;
}

SecretKey nonzero_secret(const PoseidonContext &ctx) {
    const auto id = ctx.crt_context()->key_parms_id();
    const auto level = ctx.crt_context()->key_context_data();
    const auto &moduli = level->coeff_modulus();
    const auto n = level->parms().degree();
    RandomToStandardAdapter engine(ctx.random_generator()->create());
    auto c = benchmark::resnet20_gpu::core::sample_balanced_sparse_secret(n, 192, engine);
    struct Wipe { std::vector<std::int8_t> &v;
        ~Wipe() { util::poseidon_memzero(v.data(), v.size()); } } wipe{c};
    if (std::count(c.begin(), c.end(), 1) != 96 || std::count(c.begin(), c.end(), -1) != 96)
        throw std::runtime_error("secret support verification failed");
    SecretKey key;
    key.data().resize(ctx, id, n*moduli.size());
    for (std::size_t j=0; j<moduli.size(); ++j)
        for (std::size_t i=0; i<n; ++i)
            key.data().data()[j*n+i] = c[i] < 0 ? moduli[j].value()-1 : c[i];
    util::ntt_negacyclic_harvey(util::RNSIter(key.data().data(), n), moduli.size(), ctx.crt_context()->small_ntt_tables());
    key.parms_id() = id;
    std::cout << "SECRET h=192 positive=96 negative=96 source=OS_random nonzero=true\n";
    return key;
}

struct Pool {
    rmm::mr::cuda_memory_resource upstream;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool{&upstream, 1<<20, std::size_t(12)<<30};
    rmm::mr::device_memory_resource *previous = rmm::mr::get_current_device_resource();
    Pool() { rmm::mr::set_current_device_resource(&pool); }
    ~Pool() { rmm::mr::set_current_device_resource(previous); }
};

long double cheb(int degree, long double x) {
    long double a=1, b=x;
    if (degree == 0) return a;
    for (int d=2; d<=degree; ++d) { auto c=2*x*b-a; a=b; b=c; }
    return b;
}
long double plain_node(const Node &n, long double x) {
    if (n.type == "leaf") {
        long double y=0;
        for (std::size_t i=0; i<n.coefficients.size(); ++i)
            y += static_cast<long double>(n.coefficients[i])*cheb(2*i+1, x);
        return y;
    }
    return cheb(n.split, x)*plain_node(*n.quotient, x)+plain_node(*n.remainder, x);
}
Complex plain_complex_node(const Node &n, Complex x) {
    auto basis = [&](int degree) {
        Complex a=1, b=x;
        for (int d=2; d<=degree; ++d) { auto c=2.0L*x*b-a; a=b; b=c; }
        return b;
    };
    if (n.type == "leaf") {
        Complex y=0;
        for (std::size_t i=0; i<n.coefficients.size(); ++i)
            y += static_cast<long double>(n.coefficients[i])*basis(2*i+1);
        return y;
    }
    return basis(n.split)*plain_complex_node(*n.quotient,x)+plain_complex_node(*n.remainder,x);
}

class Replay {
    const PoseidonContext &ctx;
    CKKSEncoder &encoder;
    Decryptor &decryptor;
    GpuParameterData &parameters;
    GpuRelinKeysData &keys;
    GpuEvaluator eval;
    int stage = 0;
    std::map<int, CT> basis;
    Ref stage_input;
    bool observe;
    enum class PlainMode { uncached, capture, replay };
    struct CachedPlain {
        std::size_t q_count;
        double value,scale;
        std::unique_ptr<GpuPlaintextData> data;
    };
    PlainMode plain_mode=PlainMode::uncached;
    std::vector<CachedPlain> plain_cache;
    std::size_t plain_index=0;
    std::unique_ptr<GpuPlaintextData> transient_plain;
    std::function<void(const CT &,CT &)> relinearize_callback;
public:
    double max_scale_drift = 0;
    // Diagnostic boundary only; callers retain their own ciphertext outputs.
    void release_scratch() { basis.clear(); stage_input.clear(); }
    Replay(const PoseidonContext &c, CKKSEncoder &e, Decryptor &d, GpuParameterData &p, GpuRelinKeysData &k,
        bool enable_observer=true)
        : ctx(c), encoder(e), decryptor(d), parameters(p), keys(k), eval(p), observe(enable_observer) {}
    void begin_plaintext_capture() {
        if(plain_mode!=PlainMode::uncached)throw std::logic_error("plaintext cache mode already active");
        plain_cache.clear();plain_index=0;transient_plain.reset();plain_mode=PlainMode::capture;
    }
    void finish_plaintext_capture() {
        if(plain_mode!=PlainMode::capture)throw std::logic_error("plaintext capture is not active");
        plain_mode=PlainMode::uncached;
    }
    void begin_plaintext_replay() {
        if(plain_mode!=PlainMode::uncached || plain_cache.empty())
            throw std::logic_error("plaintext cache is unavailable for replay");
        plain_index=0;transient_plain.reset();plain_mode=PlainMode::replay;
    }
    void finish_plaintext_replay() {
        if(plain_mode!=PlainMode::replay || plain_index!=plain_cache.size())
            throw std::logic_error("plaintext replay did not consume the captured plan");
        plain_mode=PlainMode::uncached;
    }
    std::size_t plaintext_cache_size() const {return plain_cache.size();}
    void set_relinearize_callback(std::function<void(const CT &,CT &)> callback) {
        relinearize_callback=std::move(callback);
    }
    void clear_relinearize_callback() {relinearize_callback={};}
    CT drop(const CT &x, int consumed) {
        if (consumed<0 || consumed>=entry_q || x.meta.q_count < entry_q-consumed)
            throw std::runtime_error("invalid Q alignment");
        CT y; eval.drop_modulus(x, y, parameters.get_level_by_q_count(entry_q-consumed).parms_id); return y;
    }
    void scale_check(const CT &x, double expected) {
        double delta=std::abs(std::log2(x.meta.scale)-expected);
        if (!std::isfinite(delta) || delta>1e-8) throw std::runtime_error("scale witness mismatch");
        max_scale_drift=std::max(max_scale_drift, delta);
    }
    const GpuPlaintextData &scalar(double value, const CT &x, double scale) {
        if(plain_mode==PlainMode::replay) {
            if(plain_index>=plain_cache.size())throw std::logic_error("plaintext replay exceeded captured plan");
            auto &cached=plain_cache[plain_index++];
            if(cached.q_count!=x.meta.q_count || cached.value!=value || cached.scale!=scale)
                throw std::logic_error("plaintext replay differs from captured plan");
            return *cached.data;
        }
        Plaintext p; encoder.encode(value, x.meta.parms_id, scale, p);
        auto uploaded=std::make_unique<GpuPlaintextData>(GpuUploader::upload_plaintext(p, 0));
        if(plain_mode==PlainMode::capture) {
            plain_cache.push_back({x.meta.q_count,value,scale,std::move(uploaded)});
            return *plain_cache.back().data;
        }
        transient_plain=std::move(uploaded);return *transient_plain;
    }
    CT mul(const CT &a, const CT &b) {
        CT raw,out;eval.multiply(a,b,raw);
        if(relinearize_callback)relinearize_callback(raw,out);
        else eval.relinearize(raw,keys,out);
        return out;
    }
    CT rescale(CT x, int output, double expected) {
        int count = static_cast<int>(x.meta.q_count)-(entry_q-output);
        if (count<0) throw std::runtime_error("negative rescale count");
        if (count) { CT y; eval.rescale_many(x, y, count); x=std::move(y); }
        scale_check(x, expected); return x;
    }
    // Only floating-roundoff metadata reconciliation before add/sub; cannot
    // change a physical scale by a meaningful amount to make a test pass.
    void reconcile(const CT &a, CT &b) {
        scale_check(b, std::log2(a.meta.scale)); b.meta.scale=a.meta.scale;
    }
    std::vector<std::complex<double>> decoded(const CT &x) {
        if(!observe)throw std::logic_error("observer disabled for performance replay");
        Ciphertext c; GpuUploader::download_ciphertext(x, c, ctx);
        Plaintext p; decryptor.decrypt(c, p);
        std::vector<std::complex<double>> got; encoder.decode(p, got);
        return got;
    }
    // Explicitly outside the online interval: the resident performance path
    // calls this only after CUPTI collection and wall/event timing have ended.
    // Keeping it separate from decoded() preserves the fail-closed rule that
    // no observer can be used while performance replay is active.
    std::vector<std::complex<double>> decoded_after_timing(const CT &x) {
        Ciphertext c;GpuUploader::download_ciphertext(x,c,ctx);
        Plaintext p;decryptor.decrypt(c,p);
        std::vector<std::complex<double>> got;encoder.decode(p,got);return got;
    }
    template<class T>
    double check(const std::string &name, const CT &x, const std::vector<T> &expected) {
        if(!observe){(void)name;(void)x;(void)expected;return 0;}
        const auto got=decoded(x);
        if (got.size()!=expected.size()) throw std::runtime_error("slot count mismatch");
        long double squares=0; double worst=0, imag=0, magnitude=0;
        for (std::size_t i=0; i<got.size(); ++i) {
            double error=std::abs(Complex(got[i])-expected[i]);
            if (!std::isfinite(error)) throw std::runtime_error("nonfinite decryption error");
            worst=std::max(worst,error); squares+=error*error;
            imag=std::max(imag,std::abs(got[i].imag())); magnitude=std::max(magnitude,double(std::abs(expected[i])));
        }
        std::cout << "CHECK " << name << " q=" << x.meta.q_count << " log_scale=" << std::log2(x.meta.scale)
            << " max_abs=" << worst << " rms=" << std::sqrt(squares/got.size()) << " max_imag=" << imag
            << " reference_max_abs=" << magnitude << std::endl;
        return worst;
    }
    CT node(const Node &n, int heap=1) {
        CT out;
        if (n.type == "leaf") {
            for (std::size_t i=0; i<n.coefficients.size(); ++i) {
                CT term=drop(basis.at(2*i+1),n.work), product;
                const auto &p=scalar(n.coefficients[i],term,std::exp2(n.pre)/term.meta.scale);
                eval.multiply_plain(term,p,product);
                if (i==0) out=std::move(product);
                else { reconcile(out, product); CT sum; eval.add(out,product,sum); out=std::move(sum); }
            }
        } else {
            auto quotient=node(*n.quotient,2*heap+1), remainder=node(*n.remainder,2*heap);
            auto b=drop(basis.at(n.split),n.work);
            quotient=drop(quotient,n.work); remainder=drop(remainder,n.work);
            auto product=mul(b,quotient); reconcile(product,remainder);
            eval.add(product,remainder,out);
        }
        scale_check(out,n.pre); out=rescale(std::move(out),n.output,n.scale);
        if(observe) {
            Ref ref(stage_input.size());
            for (std::size_t i=0; i<ref.size(); ++i) ref[i]=plain_node(n,stage_input[i]);
            check("stage"+std::to_string(stage)+"/node"+std::to_string(heap),out,ref);
        }
        return out;
    }
    CT component(const Stage &s, const CT &input, Ref &ref, int index, double &max_error) {
        stage=index;if(observe)stage_input=ref;basis.clear();
        scale_check(input,s.input_scale);
        basis.emplace(1,drop(input,s.start));
        for (const auto &b:s.basis) {
            int left=1;
            while (2*left<b.degree) left*=2;
            int right=b.degree-left, diff=left-right;
            auto a=drop(basis.at(left),b.work), c=drop(basis.at(right),b.work);
            auto product=mul(a,c); CT twice; eval.add(product,product,twice);
            CT corrected;
            if (diff==0) {
                const auto &p=scalar(1,twice,twice.meta.scale); eval.sub_plain(twice,p,corrected);
            } else {
                auto d=drop(basis.at(diff),b.work);
                const auto &p=scalar(1,d,twice.meta.scale/d.meta.scale);
                CT aligned; eval.multiply_plain(d,p,aligned); reconcile(twice,aligned);
                eval.sub(twice,aligned,corrected);
            }
            scale_check(corrected,b.pre);
            auto result=rescale(std::move(corrected),b.output,b.scale);
            if(observe) {
                Ref expected(ref.size());
                for (std::size_t i=0; i<ref.size(); ++i) expected[i]=cheb(b.degree,ref[i]);
                check("stage"+std::to_string(stage)+"/T"+std::to_string(b.degree),result,expected);
            }
            basis.emplace(b.degree,std::move(result));
        }
        auto out=node(*s.tree);
        if(observe) {
            for (auto &v:ref) v=plain_node(*s.tree,v);
            max_error=check("P"+std::to_string(s.degree)+"_stage"+std::to_string(stage),out,ref);
        } else max_error=0;
        basis.clear(); return out;
    }
    CT tail(const CT &original, const CT &last, int work, int drops) {
        CT shifted; const auto &half=scalar(0.5,last,last.meta.scale); eval.add_plain(last,half,shifted);
        auto a=drop(original,work), b=drop(shifted,work);
        return rescale(mul(a,b),work+drops,40);
    }
};

#ifndef POSEIDON_RELU_PRECISION_LIBRARY
int main(int argc, char **argv) {
    try {
        if (argc!=3 || std::string(argv[1])!="--unsafe-accuracy-only")
            throw std::runtime_error("usage: relu_precision --unsafe-accuracy-only fixture.txt");
        std::ifstream in(argv[2]); std::string magic; in>>magic;
        if (magic!="RELU_PRECISION_V1") throw std::runtime_error("invalid fixture");
        int nq,np,ns; in>>nq>>np;
        if (nq!=50 || np!=25) throw std::runtime_error("expected saved Q50/P25 dnum2 witness");
        std::vector<Modulus> q,p;
        for (int i=0;i<nq+np;++i) { std::uint64_t v; in>>v; (i<nq?q:p).emplace_back(v); }
        in>>ns; if (ns!=3) throw std::runtime_error("expected three stages");
        std::vector<Stage> stages(ns);
        for (auto &s:stages) {
            int count; in>>s.degree>>s.start>>s.end>>s.input_scale>>s.output_scale>>count;
            if (count<0 || count>32) throw std::runtime_error("invalid basis count");
            s.basis.resize(count);
            for (auto &b:s.basis) in>>b.degree>>b.work>>b.output>>b.pre>>b.scale;
            s.tree=read_node(in);
        }
        int work,drops; std::string coeff_hash,witness_hash; in>>work>>drops>>coeff_hash>>witness_hash;
        if (!in || work+drops!=22) throw std::runtime_error("invalid/truncated ReLU schedule");
        std::cout << std::setprecision(14) << std::unitbuf
            << "ACCURACY_ONLY security_approved=false N=65536 Q=50 P=25 dnum=2 slots=32768\n"
            << "FIXTURE coefficients_sha256=" << coeff_hash << " witness_sha256=" << witness_hash << '\n';
        gpu_check_cuda(cudaSetDevice(0),"cudaSetDevice");
        std::size_t free,total; gpu_check_cuda(cudaMemGetInfo(&free,&total),"cudaMemGetInfo");
        if (free<(std::size_t(16)<<30)) throw std::runtime_error("need at least 16 GiB free for this bounded diagnostic");
        Pool pool;
        ParametersLiteral parms(CKKS,16,15,40,192,1,Modulus(0),q,p,sec_level_type::none);
        PoseidonFactory::get_instance()->set_device_type(DEVICE_SOFTWARE);
        auto ctx=PoseidonFactory::get_instance()->create_poseidon_context(parms);
        CKKSEncoder encoder(ctx);
        auto secret=nonzero_secret(ctx);
        KeyGenerator generator(ctx,secret); PublicKey pub; generator.create_public_key(pub);
        RelinKeys relin; generator.create_relin_keys(relin);
        Encryptor encryptor(ctx,pub,secret); Decryptor decryptor(ctx,secret);
        GpuParameterData parameters(ctx,0);
        auto keys=GpuUploader::upload_relin_keys(relin,0);
        Replay replay(ctx,encoder,decryptor,parameters,keys);
        std::vector<double> input(32768);
        for (std::size_t i=0;i<input.size();++i) input[i]=-1+2.0*i/(input.size()-1);
        // Public corners and finely spaced values near the approximation gap.
        input[0]=0; input[1]=-1; input[2]=1;
        for (int i=3;i<67;++i) input[i]=(i%2?1:-1)*std::exp2(-1-(i-3)/2);
        Ref ref(input.begin(),input.end()), original_ref=ref;
        Plaintext plain; encoder.encode(input,parameters.get_level_by_q_count(entry_q).parms_id,std::exp2(40),plain);
        Ciphertext encrypted; encryptor.encrypt(plain,encrypted);
        auto original=GpuUploader::upload_ciphertext(encrypted,0);
        replay.check("encrypted_input",original,ref);
        auto current=replay.drop(original,0);
        const auto decoded_input=replay.decoded(original);
        std::vector<Complex> propagated(decoded_input.begin(),decoded_input.end());
        const auto noisy_input=propagated;
        double errors[3];
        for (int i=0;i<3;++i) {
            current=replay.component(stages[i],current,ref,i+1,errors[i]);
            for (auto &v:propagated) v=plain_complex_node(*stages[i].tree,v);
            replay.check("stage"+std::to_string(i+1)+"_vs_polynomial_of_decrypted_input",current,propagated);
        }
        auto result=replay.tail(original,current,work,drops);
        long double approximation=0, sum=0, source_delta=0;
        for (std::size_t i=0;i<ref.size();++i) {
            ref[i]=original_ref[i]*(ref[i]+0.5L);
            auto error=std::abs(ref[i]-std::max(0.0L,original_ref[i]));
            approximation=std::max(approximation,error); sum+=error*error;
            source_delta=std::max(source_delta,std::abs(ref[i]-original_application_relu(input[i])));
            propagated[i]=noisy_input[i]*(propagated[i]+0.5L);
        }
        std::cout << "SOURCE_REFERENCE original_application_function_max_difference=" << source_delta << '\n';
        if (!std::isfinite(source_delta) || source_delta>1e-10L)
            throw std::runtime_error("fixture differs from original application reference");
        auto error=replay.check("ReLU_final_vs_original_polynomial",result,ref);
        replay.check("ReLU_final_vs_polynomial_of_decrypted_input",result,propagated);
        std::cout << "APPROXIMATION plaintext_polynomial_vs_ReLU_max=" << approximation
            << " rms=" << std::sqrt(sum/ref.size()) << '\n';
        Ref ideal=original_ref; for (auto &v:ideal) v=std::max(0.0L,v);
        replay.check("ReLU_final_vs_ideal_ReLU",result,ideal);
        const bool pass=error<=1e-5 && std::all_of(std::begin(errors),std::end(errors),[](double x){return x<=1e-5;});
        std::cout << "RESULT precision=" << (pass?"PASS":"FAIL")
            << " final_output=" << (error<=1e-5?"PASS":"FAIL")
            << " stage_guard=" << (pass?"PASS":"FAIL")
            << " he_tolerance=1e-5 max_scale_drift_bits="
            << replay.max_scale_drift << " actual_Q_consumed=" << entry_q-result.meta.q_count
            << " bootstrap_tested=false full_network_tested=false\n";
        return pass?0:1;
    } catch (const std::exception &e) { std::cerr<<"ERROR "<<e.what()<<std::endl; return 2; }
}
#endif
