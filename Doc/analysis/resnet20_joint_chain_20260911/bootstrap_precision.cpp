// Continuous real-secret bootstrap -> ReLU diagnostic. No production gate bypass.
#define POSEIDON_RELU_PRECISION_LIBRARY
#include "relu_precision.cpp"
#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/advance/homomorphic_mod.h"
#include <set>

using Values = std::vector<Complex>;
using Diagonals = std::map<int, std::vector<std::complex<double>>>;

Values plain_dft(const Values &x, const Diagonals &matrix) {
    Values out(x.size());
    for (const auto &[rotation, diagonal] : matrix) {
        if (diagonal.size()!=x.size()) throw std::runtime_error("DFT oracle size mismatch");
        for (std::size_t i=0;i<x.size();++i)
            out[i] += Complex(diagonal[i])*x[(i+rotation)&(x.size()-1)];
    }
    return out;
}
double compare_values(const std::string &name, const Values &got, const Values &ref) {
    if (got.size()!=ref.size()) throw std::runtime_error("oracle size mismatch");
    long double worst=0, squares=0;
    for (std::size_t i=0;i<got.size();++i) {
        auto e=std::abs(got[i]-ref[i]);
        if (!std::isfinite(e)) throw std::runtime_error("nonfinite oracle error");
        worst=std::max(worst,e); squares+=e*e;
    }
    std::cout<<"ORACLE "<<name<<" max_abs="<<worst<<" rms="<<std::sqrt(squares/got.size())<<'\n';
    return worst;
}
Values decoded_values(Replay &r, const CT &ct) {
    const auto d=r.decoded(ct); return Values(d.begin(),d.end());
}
Complex evalmod_plain(const EvalModPoly &poly, Complex x, long double fold_seed=1) {
    x += -0.5L/(poly.sc_fac()*(poly.sine_poly_b()-poly.sine_poly_a()));
    Complex prev=1, current=x;
    const auto &c=poly.sine_poly().data();
    Complex y=Complex(c.at(0))*fold_seed+Complex(c.at(1))*fold_seed*x;
    for (std::size_t d=2;d<c.size();++d) {
        auto next=2.0L*x*current-prev;
        y+=Complex(c[d])*fold_seed*next; prev=current; current=next;
    }
    long double constant=poly.sqrt_2pi()*fold_seed;
    for (unsigned i=0;i<poly.double_angle();++i) {
        constant*=constant; y=2.0L*y*y-constant;
    }
    return y;
}

CT trace_dft(const std::string &name, const CT &input,
    const GpuLinearMatrixGroup &matrix, const std::vector<Diagonals> &ideal,
    const GpuGaloisKeysData &keys, GpuEvaluator &eval, Replay &replay, Values ref) {
    CT current;
    for (std::size_t stage=0;stage<matrix.data().size();++stage) {
        const CT &source=stage?current:input;
        auto noisy=decoded_values(replay,source);
        auto local=plain_dft(noisy,ideal.at(stage));
        ref=plain_dft(ref,ideal.at(stage));
        CT product, next;
        eval.multiply_by_diag_matrix_bsgs(source,matrix.data()[stage],keys,0,product);
        eval.rescale_dynamic(product,next,matrix.rescale_min_scale());
        if (next.meta.q_count+matrix.rescale_counts().at(stage)!=source.meta.q_count)
            throw std::runtime_error("DFT Q schedule mismatch");
        const auto tag=name+"_stage"+std::to_string(stage+1);
        replay.check(tag+"_cumulative",next,ref);
        replay.check(tag+"_local_arithmetic",next,local);
        current=std::move(next);
    }
    return current;
}

#ifndef POSEIDON_BOOTSTRAP_PRECISION_LIBRARY
int main(int argc, char **argv) {
    try {
        if (argc<3 || argc>4 || std::string(argv[1])!="--unsafe-accuracy-only")
            throw std::runtime_error("usage: bootstrap_precision --unsafe-accuracy-only fixture.txt [input_amplitude]");
        const double amplitude=argc==4?std::stod(argv[3]):0.5;
        if (!(amplitude>0 && amplitude<=1)) throw std::runtime_error("amplitude must be in (0,1]");
        std::ifstream in(argv[2]); std::string magic; int nq,np,ns;
        in>>magic>>nq>>np;
        if (magic!="RELU_PRECISION_V1" || nq!=50 || np!=25) throw std::runtime_error("expected Q50/P25 witness");
        std::vector<Modulus> q,p;
        for (int i=0;i<nq+np;++i) { std::uint64_t v; in>>v; (i<nq?q:p).emplace_back(v); }
        in>>ns; if (ns!=3) throw std::runtime_error("invalid ReLU stages");
        std::vector<Stage> stages(ns);
        for (auto &s:stages) {
            int count; in>>s.degree>>s.start>>s.end>>s.input_scale>>s.output_scale>>count;
            if (count<0 || count>32) throw std::runtime_error("invalid basis count");
            s.basis.resize(count);
            for (auto &b:s.basis) in>>b.degree>>b.work>>b.output>>b.pre>>b.scale;
            s.tree=read_node(in);
        }
        int tail_work,tail_drop; std::string coeff_hash,witness_hash;
        in>>tail_work>>tail_drop>>coeff_hash>>witness_hash;
        if (!in) throw std::runtime_error("truncated witness");
        std::cout<<std::unitbuf<<std::setprecision(14)
            <<"BOOTSTRAP_ACCURACY_ONLY N=65536 Q=50 P=25 dnum=2 h=192 degree=59 DA=2 arcsine_degree=0 amplitude="<<amplitude<<'\n'
            <<"FIXTURE coefficients_sha256="<<coeff_hash<<" witness_sha256="<<witness_hash<<'\n';
        gpu_check_cuda(cudaSetDevice(0),"cudaSetDevice");
        std::size_t free,total; gpu_check_cuda(cudaMemGetInfo(&free,&total),"cudaMemGetInfo");
        if (free<(std::size_t(18)<<30)) throw std::runtime_error("need 18 GiB free; bounded pool is 12 GiB");
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
        const auto &raise_table=parameters.get_level_by_q_count(2).shards.at(0);
        if (raise_table.bootstrap_raise_source_q_count!=2 || raise_table.bootstrap_raise_target_q_count!=50)
            throw std::runtime_error("context did not configure Q2 -> Q50 ModRaise");
        auto keys=GpuUploader::upload_relin_keys(relin,0);
        GpuEvaluator eval(parameters); Replay replay(ctx,encoder,decryptor,parameters,keys);
        std::vector<double> input(32768);
        for (std::size_t i=0;i<input.size();++i) input[i]=amplitude*(-1+2.0*i/(input.size()-1));
        input[0]=0; input[1]=-amplitude; input[2]=amplitude;
        Values expected(input.begin(),input.end());
        Plaintext plain; encoder.encode(input,parameters.get_level_by_q_count(6).parms_id,std::exp2(40),plain);
        Ciphertext ct; encryptor.encrypt(plain,ct);
        auto start=GpuUploader::upload_ciphertext(ct,0);
        replay.check("input_Q6",start,expected);
        const auto noisy_start=decoded_values(replay,start);

        std::cout<<"PHASE create_DFT_matrices\n";
        const double target=std::exp2(45);
        EvalModPoly poly(ctx,CosDiscrete,target,0,5,2,25,0,59);
        const double c2s_scaling=poly.q_div()/(poly.k()*poly.sc_fac()*poly.q_diff());
        // The saved Q50 witness uses exact q0/32, not rounded-to-2^59.
        const double prep_scale=std::exp2(58.97349886989023);
        const double factor=1/(poly.k()*poly.sc_fac()*poly.q_diff()*32);
        const double logical_scale=prep_scale*c2s_scaling/factor;
        HomomorphicDFTMatrixLiteral stc_literal(poseidon::decode,16,15,5,{1,1,1},true,1,false,1);
        HomomorphicDFTMatrixLiteral cts_literal(poseidon::encode,16,15,49,{1,1,1},true,c2s_scaling,false,1);
        auto stc_ideal=stc_literal.gen_matrices(), cts_ideal=cts_literal.gen_matrices();
        auto inverse_check=expected;
        for (const auto &matrix:stc_ideal) inverse_check=plain_dft(inverse_check,matrix);
        for (const auto &matrix:cts_ideal) inverse_check=plain_dft(inverse_check,matrix);
        for (auto &v:inverse_check) v*=2/c2s_scaling;
        if (compare_values("DFT_pair_inverse",inverse_check,expected)>1e-8)
            throw std::runtime_error("plain DFT pair is not an inverse");
        LinearMatrixGroup stc_cpu, cts_cpu;
        stc_literal.create_dynamic(stc_cpu,encoder,std::exp2(40),target,target);
        cts_literal.create_dynamic(cts_cpu,encoder,logical_scale,target,target);
        if (stc_cpu.rescale_counts()!=std::vector<std::uint32_t>{1,1,2} || cts_cpu.rescale_counts()!=std::vector<std::uint32_t>{1,1,2})
            throw std::runtime_error("DFT chain differs from Q50 witness");
        std::set<int> rotations{0};
        rotations.insert(stc_cpu.rot_index().begin(),stc_cpu.rot_index().end());
        rotations.insert(cts_cpu.rot_index().begin(),cts_cpu.rot_index().end());
        const std::size_t key_bytes=rotations.size()*std::size_t(2)*2*75*65536*4;
        std::cout<<"PHASE rotation_keys count="<<rotations.size()<<" estimated_gpu_GiB="<<double(key_bytes)/(1ULL<<30)<<'\n';
        if (key_bytes>(std::size_t(7)<<30)) throw std::runtime_error("rotation key estimate exceeds diagnostic budget");
        std::vector<std::uint32_t> elts;
        for (int step:rotations) elts.push_back(ctx.crt_context()->galois_tool()->get_elt_from_step(step));
        GaloisKeys galois; generator.create_galois_keys(elts,galois);
        auto gpu_galois=GpuUploader::upload_galois_keys(galois,0);
        auto stc=GpuUploader::upload_linear_matrix_group(stc_cpu,0);
        auto cts=GpuUploader::upload_linear_matrix_group(cts_cpu,0);
        std::cout<<"PHASE S2C\n";
        auto stc_out=trace_dft("S2C",start,stc,stc_ideal,gpu_galois,eval,replay,expected);
        replay.scale_check(stc_out,47.045350653721194);
        const auto decoded_stc=decoded_values(replay,stc_out);
        CT prepared,raised;
        eval.bootstrap_prepare_modraise_input(stc_out,prepared,parameters.get_level_by_q_count(2).parms_id,ctx.crt_context()->q0()/32);
        replay.scale_check(prepared,std::log2(prep_scale));
        replay.check("prepare_integer3897",prepared,decoded_stc);
        eval.raise_modulus(prepared,raised);
        if (raised.meta.q_count!=50) throw std::runtime_error("ModRaise did not produce Q50");
        // ModRaise introduces integer multiples of Q. Raw decoded values are
        // not an identity oracle until EvalMod removes those multiples.
        std::cout<<"PHASE ModRaise_complete q="<<raised.meta.q_count<<" physical_log_scale="<<std::log2(raised.meta.scale)<<'\n';
        raised.meta.scale=logical_scale;
        auto raised_values=decoded_values(replay,raised);
        std::cout<<"PHASE C2S logical_log_scale="<<std::log2(logical_scale)<<'\n';
        auto cts_out=trace_dft("C2S",raised,cts,cts_ideal,gpu_galois,eval,replay,raised_values);
        CT conjugated,real,imag_difference,imag;
        eval.conjugate(cts_out,gpu_galois,conjugated);
        eval.add(cts_out,conjugated,real); eval.sub(cts_out,conjugated,imag_difference);
        Plaintext minus_i; encoder.encode(std::complex<double>(0,-1),cts_out.meta.parms_id,1,minus_i);
        eval.multiply_plain(imag_difference,GpuUploader::upload_plaintext(minus_i,0),imag);
        replay.scale_check(real,54.035945568761946);
        const auto real_values=decoded_values(replay,real), imag_values=decoded_values(replay,imag);
        double max_domain=0; std::size_t outside=0;
        Values sine_output(expected.size()), polynomial_output(expected.size());
        for (std::size_t i=0;i<expected.size();++i) {
            const double domain=std::max(std::abs(real_values[i]),std::abs(imag_values[i]));
            max_domain=std::max(max_domain,domain); outside+=domain>1;
            const long double pi=std::acos(-1.L), k=poly.k()*poly.sc_fac();
            auto ideal_sine=[&](Complex x){return static_cast<long double>(poly.q_diff())/(2*pi)*std::sin(2*pi*k*x);};
            sine_output[i]=32.0L*(ideal_sine(real_values[i])+Complex(0,1)*ideal_sine(imag_values[i]));
            polynomial_output[i]=32.0L*(evalmod_plain(poly,real_values[i])+Complex(0,1)*evalmod_plain(poly,imag_values[i]));
        }
        std::cout<<"DOMAIN evalmod_normalized_max="<<max_domain<<" slots_outside_unit_interval="<<outside<<'\n';
        compare_values("ideal_modular_sine_vs_original_input",sine_output,expected);
        compare_values("degree59_plain_vs_modular_sine",polynomial_output,sine_output);
        compare_values("degree59_plain_vs_original_input",polynomial_output,expected);

        setenv("POSEIDON_BOOTSTRAP_EVALMOD_DYNAMIC_RESCALE","1",1);
        setenv("POSEIDON_EVALMOD_LOG_SPLIT","3",1);
        setenv("POSEIDON_EVALMOD_FLAT_BSGS_B8","0",1);
        setenv("POSEIDON_EVALMOD_VIRTUAL_DEGREE_BOUND","0",1);
        setenv("POSEIDON_EVALMOD_LEAD_LEAF_RESPLIT","0",1);
        poly.set_level_start(real.meta.q_count-1);
        auto plan=GpuUploader::upload_eval_mod_high_precision(poly,encoder,real.meta.parms_id,0,nullptr,parms_id_zero,1,nullptr,true,
            std::numeric_limits<std::uint32_t>::max(),NAN,NAN,true,real.meta.scale,true);
        const double raw_scale=plan.output_scale;
        const double alpha=std::exp2(40)/raw_scale, seed=std::pow(alpha,0.25);
        if (plan.output_q_count!=31 || std::abs(std::log2(raw_scale)-53.03094629532368)>1e-8)
            throw std::runtime_error("EvalMod differs from Q50 witness");
        std::cout<<"PHASE EvalMod_fold seed="<<seed<<" alpha="<<alpha<<" raw_log_scale="<<std::log2(raw_scale)<<'\n';
        Polynomial folded=poly.sine_poly(); for (auto &v:folded.data()) v*=seed;
        GpuBootstrapData data;
        data.eval_mod=GpuUploader::upload_eval_mod_high_precision(poly,encoder,real.meta.parms_id,0,&keys,parms_id_zero,1,&folded,true,
            std::numeric_limits<std::uint32_t>::max(),poly.sqrt_2pi()*seed,NAN,true,real.meta.scale,false);
        GpuBootstrapWorkspace workspace; CT eval_real,eval_imag;
        eval.eval_mod_high_precision(real,data,keys,workspace,eval_real);
        eval.eval_mod_high_precision(imag,data,keys,workspace,eval_imag);
        Values folded_real(expected.size()),folded_imag(expected.size());
        for (std::size_t i=0;i<expected.size();++i) {
            folded_real[i]=evalmod_plain(poly,real_values[i],seed);
            folded_imag[i]=evalmod_plain(poly,imag_values[i],seed);
        }
        replay.check("EvalMod_folded_real_arithmetic",eval_real,folded_real);
        replay.check("EvalMod_folded_imag_arithmetic",eval_imag,folded_imag);
        Plaintext plus_i; encoder.encode(std::complex<double>(0,1),eval_imag.meta.parms_id,1,plus_i);
        CT imaginary,combined,result;
        eval.multiply_plain(eval_imag,GpuUploader::upload_plaintext(plus_i,0),imaginary);
        eval.add(eval_real,imaginary,combined); eval.multiply_scalar(combined,32,result);
        replay.scale_check(result,std::log2(raw_scale));
        result.meta.scale*=alpha;
        const double bootstrap_error=replay.check("bootstrap_final_vs_original_input",result,expected);
        replay.check("bootstrap_final_vs_decrypted_input",result,noisy_start);
        const double arithmetic=replay.check("bootstrap_final_vs_degree59_plain",result,polynomial_output);
        std::cout<<"BOOTSTRAP_RESULT max_abs="<<bootstrap_error<<" arithmetic_max="<<arithmetic
            <<" target_1e_4="<<(bootstrap_error<=1e-4?"PASS":"FAIL")<<" q="<<result.meta.q_count<<'\n';
        if (bootstrap_error>0.05 || arithmetic>0.001 || outside) {
            std::cout<<"STOP first_failed_boundary=bootstrap no_reencryption=true full_network_tested=false\n";
            return 1;
        }
        std::cout<<"PHASE continuous_bootstrap_to_ReLU no_reencryption=true\n";
        Ref ref(input.begin(),input.end()); auto current=replay.drop(result,0); double stage_errors[3];
        for (int i=0;i<3;++i) current=replay.component(stages[i],current,ref,i+1,stage_errors[i]);
        auto after_relu=replay.tail(result,current,tail_work,tail_drop);
        for (std::size_t i=0;i<ref.size();++i) ref[i]=input[i]*(ref[i]+0.5L);
        const double chain_error=replay.check("bootstrap_then_ReLU_vs_original_plain",after_relu,ref);
        std::cout<<"CHAIN_RESULT max_abs="<<chain_error<<" bootstrap_tested=true relu_tested=true convolution_tested=false full_network_tested=false\n";
        return bootstrap_error<=1e-4 && chain_error<=1e-4?0:1;
    } catch (const std::exception &e) { std::cerr<<"ERROR "<<e.what()<<std::endl; return 2; }
}
#endif
