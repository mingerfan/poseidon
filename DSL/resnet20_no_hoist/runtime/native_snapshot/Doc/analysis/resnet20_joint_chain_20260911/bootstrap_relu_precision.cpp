// Same-ciphertext Q50 bootstrap -> scale40 -> original ReLU diagnostic.
// No production/security gate changes; decoded values are reference-only.
#define POSEIDON_BOOTSTRAP_BASELINE_LIBRARY
#include "bootstrap_baseline_precision.cpp"
#include <sstream>

struct ReluFixture {
    std::vector<std::uint64_t> q,p;
    std::vector<Stage> stages;
    int tail_work=0,tail_drop=0;
    std::string coefficients_hash,witness_hash;
    explicit ReluFixture(const std::string &path) {
        std::ifstream in(path); std::string magic; int nq=0,np=0,ns=0;
        in>>magic>>nq>>np;
        if (magic!="RELU_PRECISION_V1" || nq!=50 || np!=25) throw std::runtime_error("Q50/P25 fixture required");
        q.resize(nq);p.resize(np); for(auto &v:q) in>>v; for(auto &v:p) in>>v;
        in>>ns; if (!in || ns!=3) throw std::runtime_error("three ReLU stages required");
        stages.resize(ns);
        const int degrees[]={15,15,27}, starts[]={0,6,12}, ends[]={6,12,19};
        for (int i=0;i<ns;++i) {
            auto &s=stages[i];int count=0;
            in>>s.degree>>s.start>>s.end>>s.input_scale>>s.output_scale>>count;
            if (!in || s.degree!=degrees[i] || s.start!=starts[i] || s.end!=ends[i] || count<1 || count>32)
                throw std::runtime_error("unexpected ReLU schedule");
            s.basis.resize(count);
            for(auto &b:s.basis) in>>b.degree>>b.work>>b.output>>b.pre>>b.scale;
            s.tree=read_node(in);
        }
        in>>tail_work>>tail_drop>>coefficients_hash>>witness_hash;
        std::string extra;
        if (!in || tail_work!=19 || tail_drop!=3 || (in>>extra)) throw std::runtime_error("invalid ReLU fixture tail");
    }
    long double reference(long double x) const {
        auto y=x;for(const auto &s:stages)y=plain_node(*s.tree,y);
        return x*(y+0.5L);
    }
};

// Compare the level/scale DAG, including term degrees and plaintext metadata,
// not coefficient values (which intentionally change by the fold factor).
std::string plan_signature(const GpuBootstrapData::EvalModData &p) {
    std::ostringstream out;out<<std::hexfloat;
    auto put=[&](auto... x){((out<<x<<','),...);out<<';';};
    auto plain=[&](const GpuPlaintextData &x){put(x.meta.q_count,x.meta.scale);};
    put(p.input_scale,p.target_scale,p.output_scale,p.output_q_count,p.polynomial_degree,
        p.polynomial_log_split,p.polynomial_flat_bsgs,p.polynomial_degree_bound_virtual,
        p.polynomial_result_node,p.polynomial_output_scale,p.polynomial_rescale_count,
        p.dynamic_rescale,p.dynamic_min_scale,p.rescale_polynomial_terms_individually);
    plain(p.input_offset_plaintext);
    for(const auto &s:p.basis_steps) {
        put(s.output_degree,s.left_degree,s.right_degree,s.correction_degree,
            s.pre_rescale_scale,s.output_scale,s.rescale_count,s.align_left_operand,
            s.operand_alignment_pre_rescale_scale,s.operand_alignment_output_scale,
            s.operand_alignment_rescale_count,s.correction_alignment_pre_rescale_scale,
            s.correction_alignment_rescale_count);
        plain(s.operand_alignment_plaintext);plain(s.correction_alignment_plaintext);plain(s.correction_plaintext);
    }
    auto terms=[&](const auto &v){put(v.size());for(const auto &t:v){put(t.degree);plain(t.coefficient_plaintext);}};
    terms(p.polynomial_terms);
    for(const auto &b:p.polynomial_blocks){put(b.output_q_count,b.output_scale,b.rescale_count);terms(b.terms);}
    for(const auto &s:p.polynomial_combine_steps) {
        put(s.output_node,s.quotient_node,s.remainder_node,s.basis_degree,s.output_scale,s.output_q_count,
            s.quotient_rescale_count,s.quotient_output_scale,s.remainder_rescale_count,
            s.product_scale,s.product_q_count,s.product_aligned_scale,s.remainder_aligned_scale);
        plain(s.product_scale_plaintext);plain(s.remainder_scale_plaintext);
    }
    for(const auto &p:p.double_angle_constants)plain(p);
    for(auto n:p.double_angle_rescale_counts)put(n);
    for(auto n:p.required_relin_q_counts)put(n);
    return out.str();
}

struct FoldSpec {
    double alpha,seed,da_base;
    Polynomial coefficients;
    explicit FoldSpec(const BaselineMetadata &m,bool fixed_degree_refit=false)
        : alpha(std::exp2(40)/m.native_plan.output_scale),seed(std::pow(alpha,0.25)),
          da_base(m.poly.sqrt_2pi()*seed),coefficients(m.poly.sine_poly()) {
        if (m.native_plan.output_q_count!=31 || m.poly.double_angle()!=2 || !(alpha>0 && alpha<1))
            throw std::runtime_error("fold requires Q31 native degree59/DA2 contract");
        if(fixed_degree_refit) {
            auto refitted=m.poly;
            const auto degree=coefficients.degree();
            refitted.refit_discrete_cosine_fixed_degree(degree);
            if(refitted.sine_poly().degree()!=degree)
                throw std::runtime_error("fixed-degree EvalMod refit changed polynomial degree");
            coefficients=refitted.sine_poly();
        }
        for(auto &v:coefficients.data())v*=seed;
        if(fixed_degree_refit && coefficients.data().size()>32) {
            // Keep the measured Q50 split-leaf degree-0 term at its native value.
            // The refit drives coefficient 32 to 2.85e-15, but native is -0.273968;
            // merely crossing the 1e-14 support threshold changes the EvalMod numerics.
            const auto native_coefficients=m.poly.sine_poly();
            coefficients.data()[32]=native_coefficients.data()[32]*seed;
        }
    }
    auto upload(const BaselineMetadata &m,GpuRelinKeysData *keys=nullptr) const {
        auto p=GpuUploader::upload_eval_mod_high_precision(m.poly,m.encoder,m.c2s_id,0,keys,
            parms_id_zero,1,&coefficients,true,std::numeric_limits<std::uint32_t>::max(),
            da_base,NAN,true,m.c2s_scale,keys==nullptr);
        const auto candidate_signature = plan_signature(p);
        const auto native_signature = plan_signature(m.native_plan);
        if (candidate_signature != native_signature) {
            std::size_t diff = 0;
            while (diff < candidate_signature.size() && diff < native_signature.size() &&
                   candidate_signature[diff] == native_signature[diff]) ++diff;
            auto group = [diff](const std::string &signature) {
                if (signature.empty()) return std::string{};
                const auto pos = std::min(diff, signature.size() - 1);
                const auto prev = signature.rfind(';', pos);
                const auto next = signature.find(';', pos);
                return signature.substr(prev == std::string::npos ? 0 : prev + 1,
                    (next == std::string::npos ? signature.size() : next) -
                    (prev == std::string::npos ? 0 : prev + 1));
            };
            std::size_t group_index = 0;
            for (std::size_t i = 0; i < std::min(diff, native_signature.size()); ++i)
                if (native_signature[i] == ';') ++group_index;
            const auto basis_prefix = std::size_t{2} +
                4 * m.native_plan.basis_steps.size();
            std::cerr << "FOLD_PLAN_DIFF byte=" << diff << " group=" << group_index
                      << " native={" << group(native_signature) << "} candidate={"
                      << group(candidate_signature) << "} basis_steps_native="
                      << m.native_plan.basis_steps.size() << " basis_steps_candidate="
                      << p.basis_steps.size();
            if (group_index >= 2 && group_index < basis_prefix) {
                const auto step_index = (group_index - 2) / 4;
                const auto component = (group_index - 2) % 4;
                static const char *components[] = {
                    "basis_step_fields", "operand_alignment_plaintext",
                    "correction_alignment_plaintext", "correction_plaintext"};
                std::cerr << " section=basis step_index=" << step_index
                          << " output_degree="
                          << m.native_plan.basis_steps.at(step_index).output_degree
                          << " component=" << components[component];
            } else if (group_index >= basis_prefix) {
                auto relative = group_index - basis_prefix;
                std::cerr << " section=after_basis relative_group=" << relative
                          << " polynomial_terms=" << m.native_plan.polynomial_terms.size()
                          << " blocks=" << m.native_plan.polynomial_blocks.size();
                if (relative == 0) {
                    std::cerr << " field=polynomial_terms.size";
                } else {
                    --relative;
                    bool located = false;
                    for (std::size_t block_index = 0;
                         block_index < m.native_plan.polynomial_blocks.size() && !located;
                         ++block_index) {
                        const auto &block = m.native_plan.polynomial_blocks[block_index];
                        if (relative == 0) {
                            std::cerr << " field=block.output_metadata block=" << block_index;
                            located = true;
                            break;
                        }
                        --relative;
                        if (relative == 0) {
                            std::cerr << " field=block.terms.size block=" << block_index
                                      << " terms=" << block.terms.size();
                            located = true;
                            break;
                        }
                        --relative;
                        for (std::size_t term_index = 0;
                             term_index < block.terms.size(); ++term_index) {
                            if (relative < 2) {
                                std::cerr << " field=block.term."
                                          << (relative == 0 ? "degree" : "plaintext_metadata")
                                          << " block=" << block_index << " term=" << term_index
                                          << " degree=" << block.terms[term_index].degree;
                                located = true;
                                break;
                            }
                            relative -= 2;
                        }
                        if (!located) {
                            // The block groups above have been fully consumed.
                        }
                    }
                    if (!located && relative >= m.native_plan.polynomial_terms.size() * 2) {
                        relative -= m.native_plan.polynomial_terms.size() * 2;
                        for (std::size_t block_index = 0;
                             block_index < m.native_plan.polynomial_blocks.size() && !located;
                             ++block_index) {
                            const auto &block = m.native_plan.polynomial_blocks[block_index];
                            const auto width = std::size_t{2} + block.terms.size() * 2;
                            if (relative < width) {
                                std::cerr << " field=block_layout block=" << block_index
                                          << " relative=" << relative;
                                located = true;
                            } else {
                                relative -= width;
                            }
                        }
                    }
                }
            } else {
                std::cerr << " section=top_level";
            }
            std::cerr << "\n";
            throw std::runtime_error("fold changed Q/scale DAG or term degrees");
        }
        return p;
    }
    Complex reference(const EvalModPoly &p,Complex x) const {
        x-=0.5L/(p.sc_fac()*(p.sine_poly_b()-p.sine_poly_a()));
        const auto &c=coefficients.data();Complex a=1,b=x,y=Complex(c.at(0))+Complex(c.at(1))*b;
        for(std::size_t i=2;i<c.size();++i){auto next=2.0L*x*b-a;y+=Complex(c[i])*next;a=b;b=next;}
        long double constant=da_base;
        for(unsigned i=0;i<p.double_angle();++i){constant*=constant;y=2.0L*y*y-constant;}
        return y;
    }
};

void check_continuation_metadata(const BaselineMetadata &m,const ReluFixture &f) {
    const auto &parms=*m.ctx.parameters_literal();
    if (parms.q().size()!=f.q.size() || parms.p().size()!=f.p.size()) throw std::runtime_error("fixture/context shape mismatch");
    for(std::size_t i=0;i<f.q.size();++i)if(parms.q()[i].value()!=f.q[i])throw std::runtime_error("fixture Q mismatch");
    for(std::size_t i=0;i<f.p.size();++i)if(parms.p()[i].value()!=f.p[i])throw std::runtime_error("fixture P mismatch");
    FoldSpec fold(m);auto plan=fold.upload(m);
    long double identity=0,source_error=0;
    for(int i=0;i<=256;++i){
        const double x=-0.8+1.6*i/256;
        identity=std::max(identity,std::abs(fold.reference(m.poly,x)/static_cast<long double>(fold.alpha)-evalmod_plain(m.poly,x)));
        const double v=-0.5+double(i)/256;
        source_error=std::max(source_error,std::abs(f.reference(v)-original_application_relu(v)));
    }
    if (!std::isfinite(identity) || identity>1e-10 || source_error>1e-10) throw std::runtime_error("fold identity or original ReLU reference mismatch");
    std::cout<<"FOLD_PLAN q_in=46 q_out="<<plan.output_q_count<<" alpha="<<fold.alpha<<" seed="<<fold.seed
        <<" native_log_scale="<<std::log2(plan.output_scale)<<" application_log_scale=40 same_DAG=true extra_Q=0"
        <<" identity_max="<<identity<<" source_reference_max="<<source_error<<'\n';
    for(std::size_t i=0;i<f.stages.size();++i){const auto &s=f.stages[i];
        std::cout<<"RELU_PLAN stage="<<i+1<<" degree="<<s.degree<<" q_in="<<31-s.start<<" q_out="<<31-s.end
            <<" input_log_scale="<<s.input_scale<<" output_log_scale="<<s.output_scale<<'\n';}
    std::cout<<"CONTINUATION_PLAN bootstrap_out=31 relu_out=9 relu_Q_consumed=22 output_log_scale=40"
        <<" no_reencryption=true security_approved=false\n";
}

struct ReluContinuation {
    const BaselineContinuation &bootstrap;
    const CT &output;
    const Values &local, &chain, &original;
};

int continue_to_relu(const BaselineContinuation &b,const ReluFixture &f,
    const std::function<int(const ReluContinuation &)> &continuation={}) {
    const auto &m=b.metadata;FoldSpec fold(m);auto &eval=b.eval;auto &replay=b.replay;
    std::cout<<"PHASE same_C2S_folded_EvalMod no_reencryption=true\n";
    GpuBootstrapData data;data.eval_mod=fold.upload(m,&b.keys);
    const bool output_real_projection=m.output_real_projection;
    GpuBootstrapWorkspace workspace;CT real,imag;
    eval.eval_mod_high_precision(b.real,data,b.keys,workspace,real);
    CT imaginary,combined,output;
    eval.eval_mod_high_precision(b.imag,data,b.keys,workspace,imag);
    Plaintext plus_i;m.encoder.encode(std::complex<double>(0,1),imag.meta.parms_id,1,plus_i);
    eval.multiply_plain(imag,GpuUploader::upload_plaintext(plus_i,0),imaginary);
    eval.add(real,imaginary,combined);
    if(output_real_projection) {
        CT half,conjugated;
        eval.multiply_scalar(combined,16,half);
        gpu_check_cuda(cudaDeviceSynchronize(),"release folded EvalMod workspace before projection key");
        workspace=GpuBootstrapWorkspace{};data=GpuBootstrapData{};
        auto projection_keys=b.bootstrap_rotation_keys({0});
        GpuUploader::prepare_key_views_for_q_counts(projection_keys,{m.native_plan.output_q_count});
        eval.conjugate(half,projection_keys,conjugated);
        eval.add(half,conjugated,output);
        std::cout<<"FOLD_REAL_PROJECTION evalmod_calls=2 conjugations=1 q_consumed=0 multiplier_before_add=16\n";
    } else {
        eval.multiply_scalar(combined,32,output);
    }
    replay.scale_check(output,std::log2(m.native_plan.output_scale));
    if(output.meta.q_count!=31)throw std::runtime_error("fold lost Q");
    Values scaled_reference=b.bootstrap_polynomial;
    for(auto &v:scaled_reference)v*=fold.alpha;
    replay.check("fold_before_scale_reconciliation",output,scaled_reference);
    // Paired with ALL polynomial coefficients and BOTH DA constants above.
    // This is never applied to an unfurled/native ciphertext.
    output.meta.scale*=fold.alpha;replay.scale_check(output,40);
    const double differential=replay.check("fold40_vs_same_C2S_native",output,decoded_values(replay,b.native_output));
    const double arithmetic=replay.check("fold40_vs_bootstrap_polynomial",output,b.bootstrap_polynomial);
    const double restore=replay.check("fold40_vs_original_input",output,b.input);
    const bool fold_pass=differential<=1e-4 && arithmetic<=1e-4 && restore<=0.002;
    std::cout<<"FOLD_RESULT integration="<<(fold_pass?"PASS":"FAIL")<<" differential_max="<<differential
        <<" arithmetic_max="<<arithmetic<<" restore_max="<<restore<<" q=31 log_scale=40 extra_Q=0\n";
    if(!fold_pass){std::cout<<"STOP first_failed_boundary=fold40 full_network_tested=false\n";return 1;}
    const Values relu_input=decoded_values(replay,output);
    gpu_check_cuda(cudaDeviceSynchronize(),"release folded EvalMod scratch");
    data=GpuBootstrapData{};workspace=GpuBootstrapWorkspace{};
    real=CT{};imag=CT{};imaginary=CT{};combined=CT{};
    double magnitude=0,imaginary_max=0;
    for(auto v:relu_input){magnitude=std::max(magnitude,double(std::abs(v)));imaginary_max=std::max(imaginary_max,double(std::abs(v.imag())));}
    std::cout<<"RELU_DOMAIN max_abs="<<magnitude<<" max_imag="<<imaginary_max<<'\n';
    if(magnitude>1){std::cout<<"STOP first_failed_boundary=relu_input_domain full_network_tested=false\n";return 1;}
    Ref reference;for(auto v:b.input){if(v.imag()!=0)throw std::runtime_error("real application input required");reference.push_back(v.real());}
    Values local=relu_input,chain=b.bootstrap_polynomial;
    auto current=replay.drop(output,0);double max_local_stage=0,stage_error=0;
    std::cout<<"PHASE continuous_bootstrap40_to_original_ReLU no_reencryption=true\n";
    for(std::size_t i=0;i<f.stages.size();++i){
        current=replay.component(f.stages[i],std::move(current),reference,i+1,stage_error);
        for(auto &v:local)v=plain_complex_node(*f.stages[i].tree,v);
        for(auto &v:chain)v=plain_complex_node(*f.stages[i].tree,v);
        max_local_stage=std::max(max_local_stage,replay.check("relu_stage"+std::to_string(i+1)+"_local_arithmetic",current,local));
        replay.check("relu_stage"+std::to_string(i+1)+"_vs_continuous_bootstrap_polynomial",current,chain);
    }
    auto result=replay.tail(output,current,f.tail_work,f.tail_drop);
    Values original(b.input.size()),ideal(b.input.size());
    for(std::size_t i=0;i<local.size();++i){
        local[i]=relu_input[i]*(local[i]+0.5L);
        chain[i]=b.bootstrap_polynomial[i]*(chain[i]+0.5L);
        original[i]=original_application_relu(double(b.input[i].real()));
        ideal[i]=std::max(0.0L,b.input[i].real());
    }
    const double local_error=replay.check("chain_final_ReLU_local_arithmetic",result,local);
    const double chain_arithmetic=replay.check("chain_final_vs_continuous_plain_polynomial",result,chain);
    const double end_error=replay.check("chain_final_vs_original_application_ReLU",result,original);
    replay.check("chain_final_vs_ideal_ReLU",result,ideal);
    compare_values("bootstrap_approximation_propagation_through_ReLU",chain,original);
    replay.scale_check(result,40);
    if(result.meta.q_count!=9)throw std::runtime_error("continuous ReLU did not consume exactly 22 Q");
    const bool pass=max_local_stage<=1e-5 && local_error<=1e-5 && chain_arithmetic<=1e-4;
    std::cout<<"CHAIN_RESULT integration="<<(pass?"PASS":"FAIL")<<" local_stage_max="<<max_local_stage
        <<" local_final_max="<<local_error<<" chain_arithmetic_max="<<chain_arithmetic
        <<" end_to_end_max="<<end_error<<" end_to_end_target_1e_4="<<(end_error<=1e-4?"PASS":"FAIL")
        <<" q="<<result.meta.q_count<<" log_scale="<<std::log2(result.meta.scale)
        <<" bootstrap_tested=true relu_tested=true convolution_tested=false full_network_tested=false no_reencryption=true\n";
    if (pass && continuation) return continuation(ReluContinuation{b,result,local,chain,original});
    return pass?0:1;
}

#ifndef POSEIDON_BOOTSTRAP_RELU_LIBRARY
int main(int argc,char **argv){
    try{
        if(argc!=4 || (std::string(argv[1])!="--metadata-only" && std::string(argv[1])!="--unsafe-accuracy-only"))
            throw std::runtime_error("usage: bootstrap_relu_precision --metadata-only|--unsafe-accuracy-only fixture.txt grid|grid-near-zero");
        const std::string dataset=argv[3];
        if(dataset!="grid" && dataset!="grid-near-zero")throw std::runtime_error("real grid dataset required");
        ReluFixture fixture(argv[2]);
        std::string profile="q50-fixed45";
        char *args[]={argv[0],argv[1],profile.data(),argv[3],argv[2]};
        return run_bootstrap_baseline(5,args,
            [&](const BaselineMetadata &m){check_continuation_metadata(m,fixture);},
            [&](const BaselineContinuation &b){return continue_to_relu(b,fixture);});
    }catch(const std::exception &e){std::cerr<<"ERROR "<<e.what()<<'\n';return 2;}
}
#endif
