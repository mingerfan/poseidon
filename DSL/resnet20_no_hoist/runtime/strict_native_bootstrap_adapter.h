#pragma once
// Include after the original bootstrap/network library definitions.
// Arithmetic/error gates below are extracted verbatim from continue_to_relu.
inline int strict_continue_native_fold_only(const BaselineContinuation &b,CT &retained) {
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
    retained=std::move(output);
    return 0;
}
inline CT strict_native_bootstrap_only(const BaselineSession &s,const CT &input,
    const Values &observer_reference,BootstrapOfflineCache &cache) {
    if(input.meta.q_count!=6 || observer_reference.size()!=32768)
        throw std::runtime_error("strict native bootstrap input contract mismatch");
    std::optional<CT> output;
    const int rc=bootstrap_existing_ciphertext(s.metadata,s.eval,s.keys,s.replay,
        input,observer_reference,s.rotation_keys,s.bootstrap_rotation_keys,
        [&](const BaselineContinuation &b){CT retained;
            const int result=strict_continue_native_fold_only(b,retained);
            if(!result)output.emplace(std::move(retained));return result;},&cache);
    if(rc || !output)throw std::runtime_error("strict native bootstrap/fold failed");
    s.replay.scale_check(*output,40);
    if(output->meta.q_count!=31)throw std::runtime_error("strict native bootstrap output Q mismatch");
    return std::move(*output);
}
