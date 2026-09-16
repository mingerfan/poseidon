// Extracted unchanged bootstrap arithmetic from the validated baseline.
// Accepts an existing ciphertext. No encryptor or secret-key interface.
using IdealDFTMatrices=decltype(std::declval<HomomorphicDFTMatrixLiteral>().gen_matrices());
struct BootstrapOfflineCache {
    IdealDFTMatrices stc_ideal,cts_ideal;
    std::unique_ptr<LinearMatrixGroup> stc_cpu,cts_cpu;
    std::vector<int> rotations;
};

int bootstrap_existing_ciphertext(const BaselineMetadata &m,
    GpuEvaluator &eval,GpuRelinKeysData &keys,Replay &replay,
    const CT &encrypted_input,const Values &expected,
    const RotationKeyFactory &rotation_keys,const RotationKeyFactory &bootstrap_rotation_keys,
    const std::function<int(const BaselineContinuation &)> &continuation={},
    BootstrapOfflineCache *offline_cache=nullptr) {
    const auto &ctx=m.ctx;auto &encoder=m.encoder;const auto &poly=m.poly;
    const auto &parms=*ctx.parameters_literal();
    const auto nq=parms.q().size(),np=parms.p().size();
    const auto predicted_q=nq-4;const double predicted_scale=m.c2s_scale;
    const auto c2s_id=m.c2s_id;const auto &plan=m.native_plan;
    const double working_scale=std::exp2(45);
    const double c2s_scaling=poly.q_div()/(poly.k()*poly.sc_fac()*poly.q_diff());
    const double preparation_target=std::exp2(std::round(std::log2(ctx.crt_context()->q0()/32)));
    std::vector<std::pair<std::string,double>> gpu_timings;
    auto timed_gpu=[&](const std::string &name,const auto &operation) {
        gpu_check_cuda(cudaDeviceSynchronize(),(name+" timing start").c_str());
        const auto begin=std::chrono::steady_clock::now();
        operation();
        gpu_check_cuda(cudaDeviceSynchronize(),(name+" timing end").c_str());
        const double milliseconds=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-begin).count();
        gpu_timings.emplace_back(name,milliseconds);
        std::cout<<"BOOTSTRAP_GPU_TIMING stage="<<name<<" elapsed_ms="<<milliseconds
            <<" excludes_offline_setup=true\n";
    };
    if(encrypted_input.meta.q_count!=6 || expected.size()!=32768)
        throw std::runtime_error("continuation bootstrap requires Q6 and 32768 reference slots");
    replay.scale_check(encrypted_input,40);
    // Clear previous ReLU bases before allocating another high-Q DFT workspace.
    replay.release_scratch();
    CT input;eval.drop_modulus(encrypted_input,input,encrypted_input.meta.parms_id);
    const auto noisy_input=decoded_values(replay,input);
    std::cout<<"PHASE bootstrap_existing_ciphertext q=6 log_scale="<<std::log2(input.meta.scale)
        <<" no_reencryption=true\n";
    const bool output_real_projection=m.output_real_projection;
    constexpr std::uint32_t transform_log_slots=15;
    std::cout<<"PHASE baseline_matrices mode="
        <<(output_real_projection?"lcdnn_s2c_first_real":"generic_complex")
        <<" log_slots="<<transform_log_slots<<'\n';
    HomomorphicDFTMatrixLiteral stc_literal(
        poseidon::decode,16,transform_log_slots,5,{1,1,1},true,1,false,1);
    HomomorphicDFTMatrixLiteral cts_literal(
        poseidon::encode,16,transform_log_slots,nq-1,{1,1,1},true,c2s_scaling,false,1);
    std::unique_ptr<LinearMatrixGroup> local_stc_cpu,local_cts_cpu;
    IdealDFTMatrices local_stc_ideal,local_cts_ideal;
    if(offline_cache && offline_cache->stc_cpu) {
        std::cout<<"BOOTSTRAP_OFFLINE_CACHE matrices=hit\n";
    } else {
        std::cout<<"BOOTSTRAP_OFFLINE_CACHE matrices=miss\n";
        auto stc_cpu=std::make_unique<LinearMatrixGroup>();
        auto cts_cpu=std::make_unique<LinearMatrixGroup>();
        auto stc_ideal=stc_literal.gen_matrices(),cts_ideal=cts_literal.gen_matrices();
        stc_literal.create_dynamic(*stc_cpu,encoder,input.meta.scale,working_scale,working_scale);
        cts_literal.create_dynamic(*cts_cpu,encoder,working_scale,working_scale,working_scale);
        if(offline_cache) {
            offline_cache->stc_ideal=std::move(stc_ideal);offline_cache->cts_ideal=std::move(cts_ideal);
            offline_cache->stc_cpu=std::move(stc_cpu);offline_cache->cts_cpu=std::move(cts_cpu);
        } else {
            local_stc_ideal=std::move(stc_ideal);local_cts_ideal=std::move(cts_ideal);
            local_stc_cpu=std::move(stc_cpu);local_cts_cpu=std::move(cts_cpu);
        }
    }
    auto &stc_cpu=offline_cache?*offline_cache->stc_cpu:*local_stc_cpu;
    auto &cts_cpu=offline_cache?*offline_cache->cts_cpu:*local_cts_cpu;
    auto &stc_ideal=offline_cache?offline_cache->stc_ideal:local_stc_ideal;
    auto &cts_ideal=offline_cache?offline_cache->cts_ideal:local_cts_ideal;
    if (stc_cpu.rescale_counts()!=std::vector<std::uint32_t>{1,1,2} || cts_cpu.rescale_counts()!=std::vector<std::uint32_t>{1,1,2})
        throw std::runtime_error("DFT drop pattern changed");
    std::set<int> rotations{0};
    rotations.insert(stc_cpu.rot_index().begin(),stc_cpu.rot_index().end());
    rotations.insert(cts_cpu.rot_index().begin(),cts_cpu.rot_index().end());
    const auto estimated_bytes=rotations.size()*((nq+np-1)/np)*2*(nq+np)*65536*4;
    if (estimated_bytes>(std::size_t(7)<<30)) throw std::runtime_error("key memory estimate too large");
    std::cout<<"PHASE baseline_keys rotations="<<rotations.size()<<" estimated_gpu_GiB="<<double(estimated_bytes)/(1ULL<<30)<<'\n';
    auto gpu_galois=bootstrap_rotation_keys(std::vector<int>(rotations.begin(),rotations.end()));
    GpuUploader::prepare_key_views_for_q_counts(gpu_galois,{4,5,6,nq-4,nq-2,nq-1,nq});
    auto stc=GpuUploader::upload_linear_matrix_group_qp(stc_cpu,ctx,0,1);
    auto cts=GpuUploader::upload_linear_matrix_group_qp(cts_cpu,ctx,0,1);
    GpuDoubleHoistWorkspace stc_work,cts_work;
    CT stc_out,prepared,raised;
    std::cout<<"PHASE baseline_S2C_double_hoist\n";
    timed_gpu("S2C",[&]{eval.dft_double_hoist(input,stc,gpu_galois,stc_work,stc_out);});
    print_rescale_trace("S2C",stc_work);
    Values expected_stc=noisy_input;
    for (auto &matrix:stc_ideal) expected_stc=plain_dft(expected_stc,matrix);
    replay.check("baseline_S2C_vs_plain_DFT",stc_out,expected_stc);
    timed_gpu("prepare",[&]{eval.bootstrap_prepare_modraise_input(
        stc_out,prepared,ctx.crt_context()->parms_id_map().at(1),preparation_target);});
    if (prepared.meta.q_count!=2) throw std::runtime_error("prepare did not retain Q2");
    const double multiplier=std::round(preparation_target/stc_out.meta.scale);
    if (std::abs(prepared.meta.scale/(stc_out.meta.scale*multiplier)-1)>1e-12)
        throw std::runtime_error("preparation does not match rounded-integer source policy");
    std::cout<<"PREPARE multiplier="<<multiplier<<" actual_log_scale="<<std::log2(prepared.meta.scale)<<'\n';
    replay.check("baseline_prepare_preserves_value",prepared,decoded_values(replay,stc_out));
    // All S2C consumers have completed (the diagnostic download above
    // synchronizes). Release scratch/matrices before the high-Q C2S.
    gpu_check_cuda(cudaDeviceSynchronize(),"baseline release S2C");
    stc_work=GpuDoubleHoistWorkspace{}; stc=GpuLinearMatrixGroupQP{};
    stc_out=CT{}; input=CT{};
    const double preparation_gain=32*prepared.meta.scale/std::exp2(std::round(std::log2(ctx.crt_context()->q0())));
    std::cout<<"CONTRACT small_signal_gain="<<preparation_gain<<" output_gain_compensation=false\n";
    timed_gpu("ModRaise",[&]{eval.raise_modulus(prepared,raised);});
    if (raised.meta.q_count!=nq) throw std::runtime_error("raise did not reach full Q");
    // Historical fixed-scale interpretation is part of the C2S coefficient
    // contract, NOT an application-facing scale reset.
    raised.meta.scale=working_scale;
    auto raised_values=decoded_values(replay,raised);
    prepared=CT{};
    Values cts_expected=raised_values;
    for (auto &matrix:cts_ideal) cts_expected=plain_dft(cts_expected,matrix);
    Values real_expected(cts_expected.size()),imag_expected(cts_expected.size());
    for (std::size_t i=0;i<cts_expected.size();++i) {
        real_expected[i]=2*cts_expected[i].real(); imag_expected[i]=2*cts_expected[i].imag();
    }
    Plaintext minus_i;
    encoder.encode(std::complex<double>(0,-1),c2s_id,1,minus_i);
    CT real,imag;
    std::cout<<"PHASE baseline_C2S_double_hoist branch=real_plus_imag\n";
    const auto minus_i_gpu=GpuUploader::upload_plaintext(minus_i,0);
    timed_gpu("C2S",[&]{eval.coeff_to_slot_double_hoist(
        raised,cts,minus_i_gpu,gpu_galois,cts_work,real,imag);});
    print_rescale_trace("C2S",cts_work);
    replay.scale_check(real,std::log2(predicted_scale));
    replay.check("baseline_C2S_real_vs_plain_DFT",real,real_expected);
    replay.check("baseline_C2S_imag_vs_plain_DFT",imag,imag_expected);
    if (real.meta.q_count!=predicted_q || imag.meta.q_count!=predicted_q)
        throw std::runtime_error("actual C2S Q count differs from plan");
    const auto real_values=decoded_values(replay,real);
    const auto imag_values=decoded_values(replay,imag);
    gpu_check_cuda(cudaDeviceSynchronize(),"baseline release C2S");
    cts_work=GpuDoubleHoistWorkspace{}; cts=GpuLinearMatrixGroupQP{};
    gpu_galois=GpuGaloisKeysData{}; raised=CT{};
    double max_domain=0; std::size_t outside=0;
    Values poly_expected(expected.size()),sine_expected(expected.size());
    const long double pi=std::acos(-1.L),total_k=poly.k()*poly.sc_fac();
    for (std::size_t i=0;i<expected.size();++i) {
        auto d=std::max(std::abs(real_values[i]),std::abs(imag_values[i]));
        max_domain=std::max(max_domain,double(d)); outside+=d>1;
        auto sine=[&](Complex x){return static_cast<long double>(poly.q_diff())/(2*pi)*std::sin(2*pi*total_k*x);};
        sine_expected[i]=32.0L*(sine(real_values[i])+Complex(0,1)*sine(imag_values[i]));
        poly_expected[i]=32.0L*(evalmod_plain(poly,real_values[i])+Complex(0,1)*evalmod_plain(poly,imag_values[i]));
        if(output_real_projection) {
            sine_expected[i]=sine_expected[i].real();
            poly_expected[i]=poly_expected[i].real();
        }
    }
    std::cout<<"DOMAIN max_abs="<<max_domain<<" outside="<<outside<<'\n';
    compare_values("baseline_ideal_sine_vs_input",sine_expected,expected);
    compare_values("baseline_degree59_vs_sine",poly_expected,sine_expected);
    compare_values("baseline_degree59_vs_input",poly_expected,expected);
    std::cout<<"PHASE baseline_original_EvalMod_no_fold\n";
    GpuBootstrapData data;
    data.eval_mod=GpuUploader::upload_eval_mod_high_precision(poly,encoder,c2s_id,0,&keys,parms_id_zero,1,nullptr,true,
        std::numeric_limits<std::uint32_t>::max(),NAN,NAN,true,real.meta.scale,false);
    if (data.eval_mod.output_q_count!=plan.output_q_count || std::abs(data.eval_mod.output_scale/plan.output_scale-1)>1e-12)
        throw std::runtime_error("device upload changed native EvalMod contract");
    GpuBootstrapWorkspace work; CT eval_real,eval_imag;
    timed_gpu("EvalMod_real",[&]{eval.eval_mod_high_precision(real,data,keys,work,eval_real);});
    CT scaled_imag,combined,result;
    timed_gpu("EvalMod_imag",[&]{eval.eval_mod_high_precision(imag,data,keys,work,eval_imag);});
    Plaintext plus_i; encoder.encode(std::complex<double>(0,1),eval_imag.meta.parms_id,1,plus_i);
    const auto plus_i_gpu=GpuUploader::upload_plaintext(plus_i,0);
    if(output_real_projection) {
        // LCDNN computes Re(x)=x/2+conj(x)/2. Folding the historical output
        // multiplier 32 into this projection gives 16*x+16*conj(x), with no
        // rescale and therefore no additional Q consumption.
        CT half,conjugated;
        gpu_check_cuda(cudaDeviceSynchronize(),"release EvalMod workspace before real projection key");
        work=GpuBootstrapWorkspace{};data=GpuBootstrapData{};
        auto projection_keys=bootstrap_rotation_keys({0});
        GpuUploader::prepare_key_views_for_q_counts(projection_keys,{plan.output_q_count});
        timed_gpu("recombine_project_real",[&]{
            eval.multiply_plain(eval_imag,plus_i_gpu,scaled_imag);
            eval.add(eval_real,scaled_imag,combined);
            eval.multiply_scalar(combined,16,half);
            eval.conjugate(half,projection_keys,conjugated);
            eval.add(half,conjugated,result);
        });
        std::cout<<"REAL_PROJECTION evalmod_calls=2 conjugations=1 q_consumed=0 multiplier_before_add=16\n";
    } else {
        timed_gpu("recombine_complex",[&]{
            eval.multiply_plain(eval_imag,plus_i_gpu,scaled_imag);
            eval.add(eval_real,scaled_imag,combined);
            eval.multiply_scalar(combined,32,result);
        });
    }
    double total_gpu_ms=0;for(const auto &[name,milliseconds]:gpu_timings){(void)name;total_gpu_ms+=milliseconds;}
    std::cout<<"BOOTSTRAP_GPU_TIMING total_stages_ms="<<total_gpu_ms
        <<" includes=S2C,prepare,ModRaise,C2S,EvalMod_real,EvalMod_imag,recombine"
        <<" excludes_offline_setup=true excludes_reference_checks=true\n";
    replay.scale_check(result,std::log2(plan.output_scale));
    if (result.meta.q_count!=plan.output_q_count) throw std::runtime_error("actual EvalMod Q count differs from plan");
    const double error=replay.check("baseline_final_vs_original_input",result,expected);
    replay.check("baseline_final_vs_decrypted_input",result,noisy_input);
    const double arithmetic=replay.check("baseline_final_vs_degree59_plain",result,poly_expected);
    const bool baseline_pass=error<=0.002 && arithmetic<=0.002 && outside==0;
    std::cout<<"BASELINE_RESULT legacy_tolerance_0_002="<<(baseline_pass?"PASS":"FAIL")
        <<" strict_target_1e_4="<<(error<=1e-4?"PASS":"FAIL")<<" max_abs="<<error<<" arithmetic_max="<<arithmetic
        <<" q="<<result.meta.q_count<<" log_scale="<<std::log2(result.meta.scale)
        <<" no_reencryption=true full_network_tested=false output_fold=false\n";
    if (continuation && baseline_pass) {
        // Retain only the C2S inputs, the native result and relin keys.
        gpu_check_cuda(cudaDeviceSynchronize(),"baseline continuation boundary");
        work=GpuBootstrapWorkspace{}; data=GpuBootstrapData{};
        eval_real=CT{}; eval_imag=CT{}; scaled_imag=CT{}; combined=CT{};
        return continuation(BaselineContinuation{m,eval,keys,replay,
            real,imag,result,expected,noisy_input,poly_expected,
            rotation_keys,bootstrap_rotation_keys});
    }
    return baseline_pass?0:1;
}
