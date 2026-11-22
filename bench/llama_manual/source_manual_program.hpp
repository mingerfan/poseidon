#pragma once
// Handwritten model program: explicit plan + owning immutable Boot resources +
// direct Poseidon-API block execution. No DSL runtime or plaintext fallback.
#include "source_manual_plan.hpp"
#include "source_manual_boot.hpp"
#include "source_fast_boot.hpp"
#include "source_manual_silu.hpp"
#include "source_manual_pair_silu.hpp"
#include "source_model.hpp"

namespace cerium_source::manual {
static_assert(exp_unit_seed_consumer_version==1,"manual entry requires the implemented exp unit consumer");
class Llama {
    const Plan plan_;
    Runtime &runtime_;
    manual_backend::SubmissionOwner host_;
    manual_backend::BoundApi primary_,silu_;
    NativeLinearAdapter linear_;
    std::uint64_t completed_blocks_=0;
    static Plan checked(Plan p){p.validate();return p;}
    static Runtime &checked_runtime(Runtime &r) {
        if(!r.api||r.api->bound_boot_profile())throw std::invalid_argument("manual program requires Runtime with_boot=false");
        require_cerium_parameters(*r.context.parameters_literal());return r;
    }
    void shape(const CT &x,std::size_t q,double scale) const {
        packing::require_single_device_real_pair_operand(x);
        if(x.meta.q_count!=q||x.meta.parms_id!=runtime_.parameters.get_level_by_q_count(q).parms_id||
           !Api::same_scale(x.meta.scale,scale))throw std::runtime_error("manual ciphertext boundary mismatch");
    }
    void bindings() {
        host_.check();primary_.verify();silu_.verify();auto &a=primary_.get(),&b=silu_.get();
        if(!runtime_.api||!a.shares_application_bindings(b)||!a.shares_application_bindings(*runtime_.api)||
           a.bound_boot_profile()!=&primary_.get_profile()||
           b.bound_boot_profile()!=&silu_.get_profile()||
           a.exp_unit_seed_policy()!=ExpUnitSeedPolicy::IntegerPhysicalScale||
           b.exp_unit_seed_policy()!=ExpUnitSeedPolicy::IntegerPhysicalScale)
            throw std::runtime_error("manual Context/key/profile/seed-policy binding changed");
        if(a.stage_observer||a.boot_observer||b.stage_observer||b.boot_observer)
            throw std::runtime_error("manual model entry forbids hidden operator observers");
    }
    static Runtime &prepare_runtime(Runtime &r,Plan plan) {
        checked_runtime(r);
        if(plan.mode==Mode::Fast&&fast_boot::fast_p_enabled()&&!r.galois.level_keys)
            fast_boot::install_fast_runtime_levels(r,source_rotations());
        return r;
    }
    using BuiltPair=std::array<std::pair<std::unique_ptr<const poseidon::gpu::GpuBootstrapProfile>,manual_backend::ProfileNumbers>,2>;
    static BuiltPair build_profiles(Runtime &r,Plan plan) {
        prepare_runtime(r,plan);
        manual_backend::SubmissionOwner setup_owner;
        if(plan.mode==Mode::Fast&&fast_boot::fast_boot_enabled())
            return fast_boot::make_fast_profile_pair(r,setup_owner);
        return {manual_backend::make_profile(r,8192,setup_owner),manual_backend::make_profile(r,65536,setup_owner)};
    }
    Llama(Runtime &r,Plan plan,std::size_t linear_cache,BuiltPair built)
        :plan_(checked(plan)),runtime_(r),
         primary_(runtime_,host_,std::move(built[0]),linear_cache),
         silu_(runtime_,host_,std::move(built[1]),linear_cache?std::min(linear_cache,std::size_t(512ULL<<20)):0),
         linear_(primary_.get(),r.context) {
        if(plan_.mode==Mode::Fast&&fast_boot::fast_p_enabled())fast_boot::validate_fast_runtime_keys(runtime_);
        else{manual_backend::validate_key(runtime_.relin);manual_backend::validate_key(runtime_.galois);}
        bindings();
    }
public:
    // Runtime must outlive the model and be constructed with_boot=false. Input
    // ciphertexts must be encrypted under that Runtime's key/layout: their
    // metadata alone cannot prove this caller obligation.
    Llama(Runtime &r,Plan plan,std::size_t linear_cache=2ULL<<30)
        :Llama(prepare_runtime(r,plan),plan,linear_cache,build_profiles(r,checked(plan))){}
    Llama(const Llama&)=delete;Llama&operator=(const Llama&)=delete;
    const Plan &plan() const noexcept{return plan_;}
    std::uint64_t completed_blocks() const noexcept{return completed_blocks_;}
    std::uint64_t primary_boots(){return primary_.get().boot_calls;}
    std::uint64_t silu_boots(){return silu_.get().boot_calls;}
    std::uint64_t projection_pairs(){return silu_.get().manual_projection_pairs;}
    std::uint64_t integer_exp_seeds(){return primary_.get().integer_exp_seed_calls;}
    std::uint64_t standard_exp_seeds(){return primary_.get().standard_exp_seed_calls;}
    const manual_backend::ProfileNumbers &primary_profile_numbers() const {return primary_.config();}
    const manual_backend::ProfileNumbers &silu_profile_numbers() const {return silu_.config();}
    std::size_t actual_boot_output_q() const {return primary_.get_profile().bootstrap_data.eval_mod.output_q_count;}
    double actual_boot_output_scale() const {return primary_.get_profile().bootstrap_data.slot_to_coeff_output_scale;}
    std::vector<CT> decoder_block(const std::vector<CT> &input,const llama_manual::LayerWeights &weights,
                                 const CT &encrypted_mask,std::size_t layer) {
        bindings();if(layer!=completed_blocks_||layer>=32||input.size()!=16)
            throw std::invalid_argument("manual blocks must execute sequentially from layer0");
        for(const auto &x:input)shape(x,6,std::exp2(40.));shape(encrypted_mask,27,std::exp2(40.));
        auto &a=primary_.get(),&b=silu_.get();
        const auto ab=a.boot_calls,bb=b.boot_calls,pairs=b.manual_projection_pairs,primary_pairs=a.manual_projection_pairs;
        const auto integer=a.integer_exp_seed_calls,standard=a.standard_exp_seed_calls;
        BoundaryTrace trace(plan_);BlockEncryptedOps ops;ops.silu_api=&b;
        if(plan_.paired_silu()) {
            const bool use_1boot=plan_.mode==Mode::Fast&&silu_1boot_enabled();
            ops.silu_pair=[use_1boot](Api &api,const CT &g0,const CT &u0,const CT &g1,const CT &u1,std::size_t) {
                auto x0=api.drop(g0,6),x1=api.drop(g1,6);
                return use_1boot?manual_silu_1boot_pair_fast(api,x0,u0,x1,u1)
                                :manual_silu_all60_pair_candidate(api,x0,u0,x1,u1);
            };
        } else {
            ops.silu=[](Api &api,const CT &g,const CT &u,std::size_t) {
                auto x=api.drop(g,6);return manual_silu_all60_unpaired(api,x,u);
            };
        }
        auto boundary=[&](const char *name,const std::vector<CT> &values) {
            const auto index=trace.observed_boundaries();if(index>=boundaries.size())throw std::runtime_error("extra manual boundary");
            const auto &expected=boundaries[index];
            const auto target_q=(plan_.fixed_boot_counts()||values.empty())?expected.q:values[0].meta.q_count;
            for(const auto &x:values)shape(x,target_q,std::exp2(40.));
            if(plan_.fixed_boot_counts()&&a.manual_projection_pairs!=primary_pairs)throw std::runtime_error("unplanned primary paired refresh");
            trace.observe({name,values.size(),values.empty()?0:values[0].meta.q_count,
                values.empty()?0:values[0].meta.scale,a.boot_calls-ab,b.boot_calls-bb,b.manual_projection_pairs-pairs,
                a.integer_exp_seed_calls-integer,a.standard_exp_seed_calls-standard});
        };
        auto result=source_block(a,linear_,input,weights,encrypted_mask,layer,boundary,ops);
        trace.finish();bindings();poseidon::gpu::gpu_check_cuda(cudaDeviceSynchronize(),"manual decoder block completion");
        ++completed_blocks_;return result;
    }
};
} // namespace cerium_source::manual
