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
#include <cstdlib>
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

bool use_relu_zero_copy_moddrop() {
    const char *raw=std::getenv("POSEIDON_RELU_ZERO_COPY_MODDROP");
    if(raw==nullptr || *raw=='\0')return true;
    const std::string value(raw);
    return value!="0" && value!="OFF" && value!="off" &&
        value!="false" && value!="FALSE";
}

bool use_relu_q_prefix_views() {
    const char *raw=std::getenv("POSEIDON_RELU_Q_PREFIX_VIEWS");
    if(raw==nullptr || *raw=='\0')return true;
    const std::string value(raw);
    return value!="0" && value!="OFF" && value!="off" &&
        value!="false" && value!="FALSE";
}

bool use_relu_leaf_fusion() {
    const char *raw=std::getenv("POSEIDON_RELU_LEAF_FUSION");
    if(raw==nullptr || *raw=='\0')return true;
    const std::string value(raw);
    return value!="0" && value!="OFF" && value!="off" &&
        value!="false" && value!="FALSE";
}

bool use_relu_basis_fusion() {
    const char *raw=std::getenv("POSEIDON_RELU_BASIS_FUSION");
    if(raw==nullptr || *raw=='\0')return true;
    const std::string value(raw);
    if(value!="0" && value!="1")throw std::invalid_argument("POSEIDON_RELU_BASIS_FUSION must be 0 or 1");
    return value=="1";
}

// Diagnostic only: downloads are never called by observer-free online replay.
// Compare ciphertext residues, not just decoded values or predicted classes.
std::size_t require_exact_ciphertexts(const PoseidonContext &ctx,
    const CT &a,const CT &b,const std::string &label) {
    if(a.meta.parms_id!=b.meta.parms_id || a.meta.q_count!=b.meta.q_count ||
        a.meta.p_count!=b.meta.p_count || a.meta.degree!=b.meta.degree ||
        a.meta.scale!=b.meta.scale || a.meta.is_ntt_form!=b.meta.is_ntt_form ||
        a.size()!=b.size())
        throw std::runtime_error(label+" ciphertext metadata differs");
    Ciphertext host_a,host_b;
    GpuUploader::download_ciphertext(a,host_a,ctx);
    GpuUploader::download_ciphertext(b,host_b,ctx);
    const auto words=a.meta.degree*a.meta.q_count*a.size();
    if(!std::equal(host_a.data(),host_a.data()+words,host_b.data()))
        throw std::runtime_error(label+" ciphertext RNS residues differ");
    return words;
}

struct ReluModDropStats {
    std::size_t materialized_calls=0;
    std::size_t inplace_calls=0;
    std::size_t inplace_discarded_q_limbs=0;
    std::size_t prefix_multiply_calls=0;
    std::size_t prefix_multiply_plain_calls=0;
    std::size_t prefix_source_views=0;
    std::size_t prefix_discarded_q_limbs=0;
    std::size_t fused_leaf_calls=0;
    std::size_t fused_leaf_terms=0;
    std::size_t fused_leaf_adds_removed=0;
    std::size_t exact_leaf_checks=0;
    std::size_t exact_leaf_residues=0;
    std::size_t fused_basis_calls=0;
    std::size_t fused_basis_constants=0;
    std::size_t fused_basis_products=0;
    std::size_t exact_basis_checks=0;
    std::size_t exact_basis_residues=0;
};

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
    const bool zero_copy_moddrop=use_relu_zero_copy_moddrop();
    const bool q_prefix_views=use_relu_q_prefix_views();
    const bool leaf_fusion;
    const bool basis_fusion;
    bool basis_preparation_checks=false;
    ReluModDropStats moddrop_stats;
public:
    double max_scale_drift = 0;
    // Diagnostic boundary only; callers retain their own ciphertext outputs.
    void release_scratch() { basis.clear(); stage_input.clear(); }
    Replay(const PoseidonContext &c, CKKSEncoder &e, Decryptor &d, GpuParameterData &p, GpuRelinKeysData &k,
        bool enable_observer=true,bool enable_leaf_fusion=use_relu_leaf_fusion(),
        bool enable_basis_fusion=use_relu_basis_fusion())
        : ctx(c), encoder(e), decryptor(d), parameters(p), keys(k), eval(p), observe(enable_observer),
          leaf_fusion(enable_leaf_fusion && q_prefix_views),
          basis_fusion(enable_basis_fusion && q_prefix_views) {}
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
    void reset_moddrop_stats() {moddrop_stats={};}
    const ReluModDropStats &get_moddrop_stats() const {return moddrop_stats;}
    bool zero_copy_moddrop_enabled() const {return zero_copy_moddrop;}
    bool q_prefix_views_enabled() const {return q_prefix_views;}
    bool leaf_fusion_enabled() const {return leaf_fusion;}
    bool basis_fusion_enabled() const {return basis_fusion;}
    void set_basis_preparation_checks(bool enabled) {basis_preparation_checks=enabled;}
    CT drop(const CT &x, int consumed) {
        if (consumed<0 || consumed>=entry_q || x.meta.q_count < entry_q-consumed)
            throw std::runtime_error("invalid Q alignment");
        ++moddrop_stats.materialized_calls;
        CT y; eval.drop_modulus(x, y, parameters.get_level_by_q_count(entry_q-consumed).parms_id); return y;
    }
    CT drop_owned(CT x, int consumed) {
        if (consumed<0 || consumed>=entry_q || x.meta.q_count < entry_q-consumed)
            throw std::runtime_error("invalid owned Q alignment");
        if(!zero_copy_moddrop)return drop(x,consumed);
        const auto target_q=static_cast<std::size_t>(entry_q-consumed);
        moddrop_stats.inplace_discarded_q_limbs+=x.meta.q_count-target_q;
        eval.drop_modulus_inplace(
            x,parameters.get_level_by_q_count(target_q).parms_id);
        ++moddrop_stats.inplace_calls;
        return x;
    }
    void scale_check(const CT &x, double expected) {
        double delta=std::abs(std::log2(x.meta.scale)-expected);
        if (!std::isfinite(delta) || delta>1e-8) throw std::runtime_error("scale witness mismatch");
        max_scale_drift=std::max(max_scale_drift, delta);
    }
    const GpuPlaintextData &scalar_at(double value,std::size_t q_count,
        parms_id_type parms_id,double scale) {
        if(plain_mode==PlainMode::replay) {
            if(plain_index>=plain_cache.size())throw std::logic_error("plaintext replay exceeded captured plan");
            auto &cached=plain_cache[plain_index++];
            if(cached.q_count!=q_count || cached.value!=value || cached.scale!=scale)
                throw std::logic_error("plaintext replay differs from captured plan");
            return *cached.data;
        }
        Plaintext p; encoder.encode(value,parms_id,scale,p);
        auto uploaded=std::make_unique<GpuPlaintextData>(GpuUploader::upload_plaintext(p, 0));
        if(plain_mode==PlainMode::capture) {
            plain_cache.push_back({q_count,value,scale,std::move(uploaded)});
            return *plain_cache.back().data;
        }
        transient_plain=std::move(uploaded);return *transient_plain;
    }
    const GpuPlaintextData &scalar(double value,const CT &x,double scale) {
        return scalar_at(value,x.meta.q_count,x.meta.parms_id,scale);
    }
    CT mul(const CT &a, const CT &b) {
        CT raw,out;eval.multiply(a,b,raw);
        if(relinearize_callback)relinearize_callback(raw,out);
        else eval.relinearize(raw,keys,out);
        return out;
    }
    CT mul_q_prefix(const CT &a,const CT &b,int consumed,
        bool view_a,bool view_b) {
        if(consumed<0 || consumed>=entry_q)
            throw std::runtime_error("invalid multiply Q-prefix alignment");
        const auto target_q=static_cast<std::size_t>(entry_q-consumed);
        if(a.meta.q_count<target_q || b.meta.q_count<target_q ||
            (!view_a && a.meta.q_count!=target_q) ||
            (!view_b && b.meta.q_count!=target_q))
            throw std::runtime_error("multiply Q-prefix source level mismatch");
        if(!q_prefix_views) {
            CT a_dropped,b_dropped;
            const CT *left=&a,*right=&b;
            if(view_a) {a_dropped=drop(a,consumed);left=&a_dropped;}
            if(view_b) {b_dropped=drop(b,consumed);right=&b_dropped;}
            return mul(*left,*right);
        }
        ++moddrop_stats.prefix_multiply_calls;
        moddrop_stats.prefix_source_views+=std::size_t(view_a)+std::size_t(view_b);
        if(view_a)moddrop_stats.prefix_discarded_q_limbs+=a.meta.q_count-target_q;
        if(view_b)moddrop_stats.prefix_discarded_q_limbs+=b.meta.q_count-target_q;
        CT raw,out;
        eval.multiply_q_prefix(a,b,raw,
            parameters.get_level_by_q_count(target_q).parms_id);
        if(relinearize_callback)relinearize_callback(raw,out);
        else eval.relinearize(raw,keys,out);
        return out;
    }
    CT multiply_plain_q_prefix(const CT &source,int consumed,
        double value,double plain_scale) {
        if(consumed<0 || consumed>=entry_q)
            throw std::runtime_error("invalid multiply-plain Q-prefix alignment");
        const auto target_q=static_cast<std::size_t>(entry_q-consumed);
        if(source.meta.q_count<target_q)
            throw std::runtime_error("multiply-plain Q-prefix source level mismatch");
        const auto target_id=parameters.get_level_by_q_count(target_q).parms_id;
        const auto &plain=scalar_at(value,target_q,target_id,plain_scale);
        CT product;
        if(!q_prefix_views) {
            auto materialized=drop(source,consumed);
            eval.multiply_plain(materialized,plain,product);
            return product;
        }
        ++moddrop_stats.prefix_multiply_plain_calls;
        ++moddrop_stats.prefix_source_views;
        moddrop_stats.prefix_discarded_q_limbs+=source.meta.q_count-target_q;
        eval.multiply_plain_q_prefix(source,plain,product);
        return product;
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
    CT fused_leaf(const Node &n) {
        if(n.coefficients.empty() || n.coefficients.size()>4 ||
            n.work<0 || n.work>=entry_q)
            throw std::runtime_error("unsupported fused ReLU leaf schedule");
        const auto q=static_cast<std::size_t>(entry_q-n.work);
        const auto id=parameters.get_level_by_q_count(q).parms_id;
        std::vector<const CT *> terms;
        std::vector<const GpuPlaintextData *> plains;
        std::vector<std::unique_ptr<GpuPlaintextData>> uncached_plains;
        for(std::size_t i=0;i<n.coefficients.size();++i) {
            const auto &term=basis.at(2*i+1);
            if(term.meta.q_count<q)throw std::runtime_error("fused leaf Q-prefix underflow");
            // Same coefficients, term order, Q level and encoding scales as
            // the unfused loop. Do not prune even a zero coefficient.
            terms.push_back(&term);
            plains.push_back(&scalar_at(n.coefficients[i],q,id,
                std::exp2(n.pre)/term.meta.scale));
            if(plain_mode==PlainMode::uncached)
                uncached_plains.push_back(std::move(transient_plain));
            ++moddrop_stats.prefix_source_views;
            moddrop_stats.prefix_discarded_q_limbs+=term.meta.q_count-q;
        }
        CT out;eval.multiply_plain_sum_q_prefix(terms,plains,out);
        ++moddrop_stats.fused_leaf_calls;
        moddrop_stats.fused_leaf_terms+=terms.size();
        moddrop_stats.fused_leaf_adds_removed+=terms.size()-1;
        if(observe) {
            CT reference;
            for(std::size_t i=0;i<terms.size();++i) {
                CT product;eval.multiply_plain_q_prefix(*terms[i],*plains[i],product);
                if(i==0)reference=std::move(product);
                else {reconcile(reference,product);CT sum;
                    eval.add(reference,product,sum);reference=std::move(sum);}
            }
            moddrop_stats.exact_leaf_residues+=require_exact_ciphertexts(
                ctx,out,reference,"ReLU fused leaf");
            ++moddrop_stats.exact_leaf_checks;
        }
        return out;
    }
    CT node(const Node &n, int heap=1) {
        CT out;
        if (n.type == "leaf" && leaf_fusion) {
            out=fused_leaf(n);
        } else if (n.type == "leaf") {
            for (std::size_t i=0; i<n.coefficients.size(); ++i) {
                const auto &term=basis.at(2*i+1);
                auto product=multiply_plain_q_prefix(term,n.work,
                    n.coefficients[i],std::exp2(n.pre)/term.meta.scale);
                if (i==0) out=std::move(product);
                else { reconcile(out, product); CT sum; eval.add(out,product,sum); out=std::move(sum); }
            }
        } else {
            auto quotient=node(*n.quotient,2*heap+1), remainder=node(*n.remainder,2*heap);
            quotient=drop_owned(std::move(quotient),n.work);
            remainder=drop_owned(std::move(remainder),n.work);
            auto product=mul_q_prefix(
                basis.at(n.split),quotient,n.work,true,false);
            reconcile(product,remainder);
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
    CT fused_basis_correction(const CT &product,int diff,int work) {
        const CT *correction=diff?&basis.at(diff):nullptr;
        const double plain_scale=correction?product.meta.scale/correction->meta.scale:product.meta.scale;
        // Encode precisely the original positive constant/alignment plaintext.
        // Doubling and subtraction happen in RNS, not through new coefficients.
        const auto &plain=scalar(1,product,plain_scale);
        if(correction) {
            ++moddrop_stats.prefix_source_views;
            moddrop_stats.prefix_discarded_q_limbs+=correction->meta.q_count-product.meta.q_count;
            ++moddrop_stats.fused_basis_products;
        } else ++moddrop_stats.fused_basis_constants;
        CT out;eval.double_sub_plain_q_prefix(product,plain,out,correction);
        ++moddrop_stats.fused_basis_calls;
        if(observe || basis_preparation_checks) {
            CT twice,reference;eval.add(product,product,twice);
            if(correction) {
                CT aligned;eval.multiply_plain_q_prefix(*correction,plain,aligned);
                reconcile(twice,aligned);eval.sub(twice,aligned,reference);
            } else eval.sub_plain(twice,plain,reference);
            moddrop_stats.exact_basis_residues+=require_exact_ciphertexts(
                ctx,out,reference,"ReLU basis correction");
            ++moddrop_stats.exact_basis_checks;
        }
        if(product.meta.q_count!=static_cast<std::size_t>(entry_q-work))
            throw std::runtime_error("basis correction Q witness mismatch");
        return out;
    }
    CT component(const Stage &s, CT input, Ref &ref, int index, double &max_error) {
        stage=index;if(observe)stage_input=ref;basis.clear();
        scale_check(input,s.input_scale);
        basis.emplace(1,drop_owned(std::move(input),s.start));
        for (const auto &b:s.basis) {
            int left=1;
            while (2*left<b.degree) left*=2;
            int right=b.degree-left, diff=left-right;
            auto product=mul_q_prefix(
                basis.at(left),basis.at(right),b.work,true,true);
            CT corrected;
            if(basis_fusion) {
                corrected=fused_basis_correction(product,diff,b.work);
            } else {
                CT twice; eval.add(product,product,twice);
                if (diff==0) {
                    const auto &p=scalar(1,twice,twice.meta.scale); eval.sub_plain(twice,p,corrected);
                } else {
                    const auto &d=basis.at(diff);
                    auto aligned=multiply_plain_q_prefix(
                        d,b.work,1,twice.meta.scale/d.meta.scale);
                    reconcile(twice,aligned);
                    eval.sub(twice,aligned,corrected);
                }
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
        auto b=drop_owned(std::move(shifted),work);
        return rescale(
            mul_q_prefix(original,b,work,true,false),work+drops,40);
    }
};

#ifndef POSEIDON_RELU_PRECISION_LIBRARY
void test_basis_fusion_api(const PoseidonContext &ctx,CKKSEncoder &encoder,
    GpuParameterData &parameters,const CT &input) {
    GpuEvaluator eval(parameters);
    const auto target=parameters.get_level_by_q_count(entry_q-2).parms_id;
    CT source,correction,source_snapshot,correction_snapshot;
    eval.drop_modulus(input,source,target);
    eval.rescale(input,correction); // Higher Q prefix and genuinely different scale.
    eval.drop_modulus(source,source_snapshot,target);
    eval.drop_modulus(correction,correction_snapshot,correction.meta.parms_id);
    std::size_t words=0,checks=0;
    for(double value:{-1.25,0.0,1.0,2.5})for(bool product:{false,true}) {
        Plaintext host;encoder.encode(value,target,
            product?source.meta.scale/correction.meta.scale:source.meta.scale,host);
        auto plain=GpuUploader::upload_plaintext(host,0);
        CT twice,reference,fused;eval.add(source,source,twice);
        if(product) {
            CT aligned;eval.multiply_plain_q_prefix(correction,plain,aligned);
            if(std::abs(std::log2(aligned.meta.scale/twice.meta.scale))>1e-8)
                throw std::runtime_error("basis API reference scale mismatch");
            aligned.meta.scale=twice.meta.scale;eval.sub(twice,aligned,reference);
        } else eval.sub_plain(twice,plain,reference);
        eval.double_sub_plain_q_prefix(source,plain,fused,product?&correction:nullptr);
        words+=require_exact_ciphertexts(ctx,fused,reference,"basis API");++checks;
        CT alias;eval.drop_modulus(source,alias,target);
        eval.double_sub_plain_q_prefix(alias,plain,alias,product?&correction:nullptr);
        words+=require_exact_ciphertexts(ctx,alias,reference,"basis source alias");++checks;
        if(product) {
            eval.drop_modulus(correction,alias,correction.meta.parms_id);
            eval.double_sub_plain_q_prefix(source,plain,alias,&alias);
            words+=require_exact_ciphertexts(ctx,alias,reference,"basis correction alias");++checks;
        }
    }
    words+=require_exact_ciphertexts(ctx,source,source_snapshot,"basis immutable source");++checks;
    words+=require_exact_ciphertexts(ctx,correction,correction_snapshot,"basis immutable correction");++checks;
    Plaintext host;encoder.encode(1.0,target,source.meta.scale,host);
    auto plain=GpuUploader::upload_plaintext(host,0);
    CT unused,empty;
    int rejected=0;
    auto reject=[&](auto operation) {
        try {operation();}catch(const std::invalid_argument &) {++rejected;return;}
        throw std::runtime_error("basis API accepted invalid operands");
    };
    reject([&]{eval.double_sub_plain_q_prefix(empty,plain,unused);});
    reject([&]{eval.double_sub_plain_q_prefix(source,plain,unused,&empty);});
    reject([&]{eval.double_sub_plain_q_prefix(input,plain,unused);});
    plain.meta.scale*=2;
    reject([&]{eval.double_sub_plain_q_prefix(source,plain,unused);});
    plain.meta.scale/=2;plain.meta.is_ntt_form=false;
    reject([&]{eval.double_sub_plain_q_prefix(source,plain,unused);});
    plain.meta.is_ntt_form=true;--plain.poly_.q_count;
    reject([&]{eval.double_sub_plain_q_prefix(source,plain,unused);});
    ++plain.poly_.q_count;source.meta.is_ntt_form=false;
    reject([&]{eval.double_sub_plain_q_prefix(source,plain,unused);});
    source.meta.is_ntt_form=true;
    CT too_low;eval.drop_modulus(source,too_low,parameters.get_level_by_q_count(entry_q-3).parms_id);
    reject([&]{eval.double_sub_plain_q_prefix(source,plain,unused,&too_low);});
    CT raw;eval.multiply(source,source,raw);
    reject([&]{eval.double_sub_plain_q_prefix(raw,plain,unused);});
    std::cout<<"RELU_BASIS_API result=PASS constants_and_products=true zero_negative_coefficients=true"
        <<" mixed_q_and_scales=true source_alias=true correction_alias=true immutable_inputs=true"
        <<" checks="<<checks<<" exact_residues="<<words<<" rejected="<<rejected<<'\n';
}

void test_leaf_fusion_api(const PoseidonContext &ctx,CKKSEncoder &encoder,
    GpuParameterData &parameters,const CT &input) {
    GpuEvaluator eval(parameters);
    const auto target=parameters.get_level_by_q_count(entry_q-2).parms_id;
    CT lower,negative,input_snapshot,lower_snapshot;
    eval.drop_modulus(input,lower,parameters.get_level_by_q_count(entry_q-1).parms_id);
    eval.negate(input,negative);
    eval.drop_modulus(input,input_snapshot,input.meta.parms_id);
    eval.drop_modulus(lower,lower_snapshot,lower.meta.parms_id);
    std::vector<GpuPlaintextData> plains;
    for(double value:{-1.25,0.0,2.5,-0.75}) {
        Plaintext plain;encoder.encode(value,target,std::exp2(40),plain);
        plains.push_back(GpuUploader::upload_plaintext(plain,0));
    }
    std::vector<const CT *> sources{&input,&lower,&negative,&input};
    std::vector<const GpuPlaintextData *> coefficients;
    for(const auto &plain:plains)coefficients.push_back(&plain);
    std::size_t exact_words=0;
    for(std::size_t count=1;count<=4;++count) {
        std::vector<const CT *> xs(sources.begin(),sources.begin()+count);
        std::vector<const GpuPlaintextData *> ps(coefficients.begin(),coefficients.begin()+count);
        CT fused,reference;eval.multiply_plain_sum_q_prefix(xs,ps,fused);
        for(std::size_t i=0;i<count;++i) {
            CT dropped,product;eval.drop_modulus(*xs[i],dropped,target);
            eval.multiply_plain(dropped,*ps[i],product);
            if(i==0)reference=std::move(product);
            else {CT sum;eval.add(reference,product,sum);reference=std::move(sum);}
        }
        exact_words+=require_exact_ciphertexts(ctx,fused,reference,"fused sum API");
    }
    exact_words+=require_exact_ciphertexts(ctx,input,input_snapshot,"immutable input");
    exact_words+=require_exact_ciphertexts(ctx,lower,lower_snapshot,"immutable Q prefix");
    CT alias,alias_reference;
    eval.drop_modulus(input,alias,input.meta.parms_id);
    eval.multiply_plain_q_prefix(input,plains[0],alias_reference);
    eval.multiply_plain_sum_q_prefix({&alias},{&plains[0]},alias);
    exact_words+=require_exact_ciphertexts(ctx,alias,alias_reference,"aliased destination");
    int rejected=0;
    auto reject=[&](auto operation) {
        try {operation();}catch(const std::invalid_argument &) {++rejected;return;}
        throw std::runtime_error("fused sum accepted invalid input");
    };
    CT unused;
    reject([&]{eval.multiply_plain_sum_q_prefix({}, {},unused);});
    reject([&]{eval.multiply_plain_sum_q_prefix(sources,{&plains[0]},unused);});
    reject([&]{eval.multiply_plain_sum_q_prefix({nullptr},{&plains[0]},unused);});
    reject([&]{eval.multiply_plain_sum_q_prefix({&input},{nullptr},unused);});
    auto too_many=sources;too_many.push_back(&input);
    auto too_many_plains=coefficients;too_many_plains.push_back(&plains[0]);
    reject([&]{eval.multiply_plain_sum_q_prefix(too_many,too_many_plains,unused);});
    // Even a zero-valued term must be validated, not silently skipped.
    plains[1].meta.scale*=2;
    reject([&]{eval.multiply_plain_sum_q_prefix(sources,coefficients,unused);});
    plains[1].meta.scale/=2;
    --plains[1].meta.q_count;
    reject([&]{eval.multiply_plain_sum_q_prefix(sources,coefficients,unused);});
    ++plains[1].meta.q_count;
    plains[1].meta.is_ntt_form=false;
    reject([&]{eval.multiply_plain_sum_q_prefix(sources,coefficients,unused);});
    plains[1].meta.is_ntt_form=true;
    CT raw;eval.multiply(input,input,raw);
    reject([&]{eval.multiply_plain_sum_q_prefix({&raw},{&plains[0]},unused);});
    std::cout<<"RELU_LEAF_API result=PASS term_counts=1,2,3,4 zero_and_negative_coefficients=true"
        <<" mixed_q_prefixes=true immutable_inputs=true alias_safe=true rejected="<<rejected
        <<" exact_residues="<<exact_words<<'\n';
}

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
        if(replay.basis_fusion_enabled())test_basis_fusion_api(ctx,encoder,parameters,original);
        if(replay.leaf_fusion_enabled())test_leaf_fusion_api(ctx,encoder,parameters,original);
        replay.check("encrypted_input",original,ref);
        auto current=replay.drop(original,0);
        const auto decoded_input=replay.decoded(original);
        std::vector<Complex> propagated(decoded_input.begin(),decoded_input.end());
        const auto noisy_input=propagated;
        double errors[3];
        for (int i=0;i<3;++i) {
            current=replay.component(stages[i],std::move(current),ref,i+1,errors[i]);
            for (auto &v:propagated) v=plain_complex_node(*stages[i].tree,v);
            replay.check("stage"+std::to_string(i+1)+"_vs_polynomial_of_decrypted_input",current,propagated);
        }
        auto result=replay.tail(original,current,work,drops);
        if(replay.basis_fusion_enabled()) {
            Replay baseline(ctx,encoder,decryptor,parameters,keys,false,replay.leaf_fusion_enabled(),false);
            auto old=baseline.drop(original,0);Ref unused;double unused_error=0;
            for(int i=0;i<3;++i)
                old=baseline.component(stages[i],std::move(old),unused,i+1,unused_error);
            old=baseline.tail(original,old,work,drops);
            const auto words=require_exact_ciphertexts(ctx,result,old,"full ReLU basis A/B");
            const auto &stats=replay.get_moddrop_stats();
            if(stats.fused_basis_calls!=15 || stats.fused_basis_constants!=10 ||
                stats.fused_basis_products!=5 || stats.exact_basis_checks!=15 ||
                stats.exact_basis_residues!=43384832)
                throw std::runtime_error("ReLU basis diagnostic inventory changed");
            std::cout<<"RELU_BASIS_EXACT result=PASS calls=15 constants=10 products=5"
                <<" exact_checks="<<stats.exact_basis_checks<<" exact_residues="<<stats.exact_basis_residues
                <<" full_relu_residues="<<words<<" same_ciphertext=true same_keys=true"
                <<" output_q="<<result.meta.q_count<<" output_log_scale="<<std::log2(result.meta.scale)<<'\n';
        }
        if(replay.leaf_fusion_enabled()) {
            // The old full ReLU runs on the SAME ciphertext and evaluation
            // keys, so random encryption noise cannot hide a regression.
            Replay baseline(ctx,encoder,decryptor,parameters,keys,false,false);
            auto old=baseline.drop(original,0);Ref unused;double unused_error=0;
            for(int i=0;i<3;++i)
                old=baseline.component(stages[i],std::move(old),unused,i+1,unused_error);
            old=baseline.tail(original,old,work,drops);
            const auto exact_words=require_exact_ciphertexts(ctx,result,old,"full ReLU A/B");
            const auto &stats=replay.get_moddrop_stats();
            if(stats.fused_leaf_calls!=14 || stats.fused_leaf_terms!=30 ||
                stats.fused_leaf_adds_removed!=16 || stats.exact_leaf_checks!=14)
                throw std::runtime_error("ReLU leaf fusion diagnostic inventory changed");
            std::cout<<"RELU_LEAF_EXACT result=PASS leaf_checks="<<stats.exact_leaf_checks
                <<" leaf_residues="<<stats.exact_leaf_residues
                <<" full_relu_residues="<<exact_words
                <<" coefficient_terms="<<stats.fused_leaf_terms
                <<" same_ciphertext=true same_keys=true output_q="<<result.meta.q_count
                <<" output_log_scale="<<std::log2(result.meta.scale)<<'\n';
        }
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
