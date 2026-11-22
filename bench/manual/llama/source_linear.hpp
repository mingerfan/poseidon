// SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// RoPE mask preparation also follows Meta's llama_models/models/llama3/
// model_params.py; retain LLAMA3_REFERENCE_LICENSE.txt for that reference.
// Cerium diagonal/dual-branch/RoPE formulas; native ResNet20 DFT adapter.
// By default the DFT API finishes/rescales each input tile. The opt-in group
// accumulator aligns the across-input giant schedule without changing Boot.
// Neither mode claims to be a literal port of Cerium's full runtime scheduler.
#pragma once
#include "source_ops.hpp"
#include "source_preparation.hpp"
#include "source_fast_linear.hpp"
#include "llama_manual/periodic_plaintext.hpp"
#include "llama_manual/periodic_plaintext_gpu.hpp"
#include "llama_manual/periodic_plaintext_pipeline.hpp"
#include "llama_manual/model.hpp"
#include <future>
#include <mutex>
#include <optional>
#include <chrono>
#include <cstdlib>

namespace cerium_source {
class NativeLinearAdapter {
    Api &api_;
    llama_periodic::Encoder encoder_;
    std::unique_ptr<llama_periodic::GpuEncoder> gpu_encoder_;
    std::map<int,std::vector<std::complex<double>>> rope_masks_;
    std::unique_ptr<FastLinearEngine> fast_;
public:
    // Drop shared baby-step rotations before the caller frees that input.
    void release_input(){if(fast_)fast_->release_input();}
    FastLinearEngine &fast_engine() {
        if(!fast_)fast_=std::make_unique<FastLinearEngine>(api_,encoder_);
        return *fast_;
    }
    std::vector<CT> run_bf16_device(
        const std::vector<CT> &input,std::size_t in_width,std::size_t out_width,
        const std::uint16_t *weights_bf16_device,bool use_o_col_perm,
        bool with_rope,const std::string &label) {
        return fast_engine().run_bf16_device(input,in_width,out_width,weights_bf16_device,
                                             use_o_col_perm,with_rope,label,rope_masks_);
    }
    NativeLinearAdapter(Api &api,const poseidon::PoseidonContext &context):api_(api),encoder_(context) {
        llama_manual::ModelConfig config;
        std::map<int,std::vector<std::complex<double>>> base;
        for(int d:{-1,0,1})base[d].resize(32768);
        const std::complex<double> imaginary(0.,1.);
        for(std::size_t j=0;j<32768;++j) {
            const auto feature=j%128;
            const double angle=(j/256)*llama_manual::rotary_frequency(config,feature/2);
            const auto freq=std::polar(1.,angle);
            base[0][j]=.5*freq;
            if(feature%2)base[1][j]=.5*imaginary*freq;
            else base[-1][j]=-.5*imaginary*freq;
        }
        // Exact masks from prepare_adjust_and_apply_rotary_embeddings_masks.
        for(int d:{-1,0,1}) {
            rope_masks_[d]=base[d];
            auto &plus=rope_masks_[128+d];auto &minus=rope_masks_[-128+d];
            plus.resize(32768);minus.resize(32768);
            for(std::size_t j=0;j<32768;++j) {
                plus[j]=(j%256>=128)?base[d][(j+32768-128)%32768]:0.;
                minus[j]=(j%256<128)?base[d][(j+128)%32768]:0.;
            }
        }
        if(fast_linear_enabled())fast_=std::make_unique<FastLinearEngine>(api_,encoder_);
    }
    template<class Weight>
    std::vector<CT> run(const std::vector<CT> &input,std::size_t in_width,std::size_t out_width,
                        Weight weight,bool with_rope,const std::string &label) {
        if(input.size()!=in_width/256||in_width%256||out_width%256||input.empty())
            throw std::invalid_argument("source Linear dimensions");
        if(fast_linear_enabled()) {
            if(!fast_)fast_=std::make_unique<FastLinearEngine>(api_,encoder_);
            return fast_->run(input,in_width,out_width,weight,with_rope,label,rope_masks_);
        }
        const auto q=input[0].meta.q_count;const auto id=input[0].meta.parms_id;
        for(const auto &x:input)if(x.meta.parms_id!=id||!Api::same_scale(x.meta.scale,std::exp2(40.)))
            throw std::invalid_argument("source Linear requires aligned scale40 inputs");
        if(q<8)throw std::runtime_error("source Linear output below native bootstrap input budget");
        const char *gpu_switch=std::getenv("LLAMA_LINEAR_GPU_ENCODE");
        const bool gpu_encode=gpu_switch&&std::string(gpu_switch)=="1";
        const char *pipeline_switch=std::getenv("LLAMA_LINEAR_GPU_PIPELINE");
        const bool gpu_pipeline=pipeline_switch&&std::string(pipeline_switch)=="1";
        if(gpu_pipeline&&!gpu_encode)throw std::invalid_argument("GPU pipeline requires GPU periodic encoding");
        if(gpu_encode&&!gpu_encoder_)gpu_encoder_=std::make_unique<llama_periodic::GpuEncoder>(encoder_);
        const char *phase_switch=std::getenv("LLAMA_LINEAR_PHASE_TIMING");
        const bool phase_timing=phase_switch&&std::string(phase_switch)=="1";
        const auto start=std::chrono::steady_clock::now();
        double prepare_wall_sum=0,wait_work_wall=0,upload_call_wall=0,native_call_wall=0,tile_sync_wall=0,projection_finish_wall=0;
        std::size_t diagonal_data_h2d_bytes=0;
        const auto nin=input.size(),nout=out_width/256,total=2*nin*nout;
        struct Work {std::map<int,llama_periodic::Encoded> diagonals;std::size_t it,ot,lane;double prepare_wall=0;};
        auto prepare=[&](std::size_t task) {
            const auto prep_start=phase_timing?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
            Work work;work.ot=task/(2*nin);work.lane=(task/nin)%2;work.it=task%nin;
            std::array<llama_periodic::Encoded,255> prepared;
            std::exception_ptr failure;std::mutex failure_mutex;
            #pragma omp parallel for num_threads(6) schedule(static)
            for(int index=0;index<255;++index)try {
                const int d=index-127,gs=(((d+32768)%32768)/16)*16;
                std::array<double,256> period{};
                for(int k=0;k<256;++k) {
                    const int f=(k-gs+32768)%256,half=f/128,column=f%128,row=column+d;
                    if(row>=0&&row<128)period[k]=weight(work.ot*256+((half+work.lane)%2)*128+column,
                                                        work.it*256+half*128+row);
                }
                prepared[index]=gpu_encode?encoder_.encode_coefficients(period,id,api_.prime(q)):encoder_.encode(period,id,api_.prime(q));
            }catch(...){std::lock_guard<std::mutex> guard(failure_mutex);if(!failure)failure=std::current_exception();}
            if(failure)std::rethrow_exception(failure);
            for(int i=0;i<255;++i)work.diagonals.emplace(i-127,std::move(prepared[i]));
            if(phase_timing)work.prepare_wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-prep_start).count();
            return work;
        };
        // Opt-in graph scheduling optimization; no polynomial, prime, scale,
        // Rescale boundary, or giant rotation changes. Cache only this immutable
        // input vector, bounded by an explicit byte budget (default 4 GiB).
        const char *cache_switch=std::getenv("LLAMA_LINEAR_BABY_CACHE");
        const bool use_cache=cache_switch&&std::string(cache_switch)=="1";
        const char *sum_switch=std::getenv("LLAMA_LINEAR_GROUP_ACCUM");
        const bool group_sum=sum_switch&&std::string(sum_switch)=="1";
        if(gpu_pipeline&&group_sum)throw std::invalid_argument("GPU pipeline/group accumulation combination is unsupported");
        const char *packed_switch=std::getenv("LLAMA_LINEAR_PACKED_UPLOAD");
        const bool packed_upload=packed_switch&&std::string(packed_switch)=="1";
        const char *prep_switch=std::getenv("LLAMA_LINEAR_PERSISTENT_PREP");
        const bool persistent_prep=prep_switch&&std::string(prep_switch)=="1";
        std::array<poseidon::gpu::GpuDoubleHoistSumState,2> sums;
        std::size_t cache_budget=4ULL<<30,cache_bytes=0,cache_hits=0;
        if(const char *v=std::getenv("LLAMA_LINEAR_BABY_CACHE_MB"))cache_budget=std::stoull(v)<<20;
        std::vector<poseidon::gpu::GpuDoubleHoistBabyCache> babies(nin);
        std::vector<CT> output;std::array<std::optional<CT>,2> branch;
        std::unique_ptr<BoundedPreparation<Work>> producer;
        std::future<Work> future;
        // Declared after GPU cache/output owners: its destructor drains first
        // on any get/DFT/postprocess exception, before those owners are freed.
        std::unique_ptr<llama_periodic::PersistentMatrixSession> pipeline;
        if(gpu_pipeline)pipeline=std::make_unique<llama_periodic::PersistentMatrixSession>(*gpu_encoder_);
        if(persistent_prep)producer=std::make_unique<BoundedPreparation<Work>>(total,prepare);
        else future=std::async(std::launch::async,prepare,0);
        for(std::size_t task=0;task<total;++task) {
            auto phase_start=phase_timing?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
            auto work=producer?producer->get():future.get();
            if(phase_timing){wait_work_wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-phase_start).count();prepare_wall_sum+=work.prepare_wall;}
            if(!producer&&task+1<total)future=std::async(std::launch::async,prepare,task+1);
            const bool final_input=work.it+1==nin;
            auto apply_matrix=[&](const poseidon::gpu::GpuLinearMatrixGroupQP &matrix){
                const auto native_start=phase_timing?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
                auto &cache=babies[work.it];const auto &meta=work.diagonals.begin()->second.meta;
                const std::size_t required=matrix.data()[0].plan.baby_steps.size()*2*(meta.q_count+meta.p_count)*meta.degree*sizeof(poseidon::gpu::GpuWord);
                if(use_cache&&!cache.ready&&required<=cache_budget-cache_bytes){api_.prepare_linear_babies(input[work.it],matrix.data()[0].plan.baby_steps,cache);cache_bytes+=required;}
                else if(cache.ready)++cache_hits;
                auto result=group_sum?api_.native_dft_sum(input[work.it],matrix,sums[work.lane],final_input,cache.ready?&cache:nullptr):api_.native_dft(input[work.it],matrix,cache.ready?&cache:nullptr);
                if(phase_timing)native_call_wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-native_start).count();
                return result;
            };
            std::optional<llama_periodic::PackedMatrix> packed;poseidon::gpu::GpuLinearMatrixGroupQP matrix;CT term;
            if(pipeline){
                if(phase_timing)phase_start=std::chrono::steady_clock::now();const double before_native=native_call_wall;
                term=pipeline->consume(work.diagonals,1,apply_matrix);
                if(phase_timing)upload_call_wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-phase_start).count()-(native_call_wall-before_native);
            }else{
                if(phase_timing)phase_start=std::chrono::steady_clock::now();
                if(gpu_encode){packed=gpu_encoder_->upload(work.diagonals,1);matrix=std::move(packed->group);}
                else if(packed_upload){packed=llama_periodic::upload_matrix_packed(work.diagonals,0,1);matrix=std::move(packed->group);}
                else matrix=llama_periodic::upload_matrix(work.diagonals,0,1);
                if(phase_timing)upload_call_wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-phase_start).count();
                term=apply_matrix(matrix);
                if(phase_timing)phase_start=std::chrono::steady_clock::now();
                poseidon::gpu::gpu_check_cuda(cudaStreamSynchronize(poseidon::gpu::gpu_execution_stream()),"native source Linear tile");
                if(phase_timing)tile_sync_wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-phase_start).count();
            }
            if(phase_timing){const auto &m=work.diagonals.begin()->second.meta;diagonal_data_h2d_bytes+=work.diagonals.size()*llama_periodic::compact_degree*(gpu_encode?sizeof(std::int64_t):(m.q_count+m.p_count)*sizeof(poseidon::gpu::GpuWord));}
            if(group_sum&&!final_input)continue;
            api_.require_scale(term,std::exp2(40.));
            if(term.meta.q_count!=q-1)throw std::runtime_error("native DFT output Q");
            auto &sum=branch[work.lane];
            if(sum)*sum=api_.add(*sum,term);else sum=std::move(term);
            if(work.lane!=1||work.it+1!=nin)continue;
            std::optional<CT> joined;
            auto add_term=[&](CT value){if(joined)*joined=api_.add(*joined,value);else joined=std::move(value);};
            const double mask_scale=api_.prime(q-1);
            if(with_rope) {
                // Preserve original products -> rotations -> sum -> conjugate.
                for(int d:{-1,0,1})add_term(api_.rotate(api_.cp_complex(*branch[0],rope_masks_.at(d),mask_scale),d));
                for(int d:{-129,-128,-127,127,128,129})
                    add_term(api_.rotate(api_.cp_complex(*branch[1],rope_masks_.at(d),mask_scale),d));
                *joined=api_.add(*joined,api_.conjugate(*joined));
            } else {
                add_term(api_.cp(*branch[0],1.,mask_scale));
                for(int d:{-128,128}) {
                    std::vector<double> mask(32768);
                    for(std::size_t j=0;j<32768;++j)mask[j]=d==128?(j%256>=128):(j%256<128);
                    add_term(api_.rotate(api_.cp_vector(*branch[1],mask,mask_scale),d));
                }
            }
            output.push_back(api_.rescale(*joined,1));branch[0].reset();branch[1].reset();
            std::cout<<"SOURCE_LINEAR label="<<label<<" tile="<<work.ot+1<<" tiles="<<nout
                <<" input_q="<<q<<" output_q="<<output.back().meta.q_count<<" fused_rope="<<with_rope
                <<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
                <<" adapter=native_dft per_input_rescale="<<(!group_sum)<<" periodic_encoding="<<(gpu_encode?"gpu_integer_ntt":"cpu")<<" gpu_pipeline="<<gpu_pipeline<<" completion_boundary="<<(gpu_pipeline?"projection_finish":"tile_sync")<<" original_scheduler=false"<<std::endl;
        }
        if(pipeline){const auto finish_start=phase_timing?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};pipeline->finish();if(phase_timing)projection_finish_wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-finish_start).count();}
        if(phase_timing)std::cout<<"SOURCE_LINEAR_PHASE {\"label\":\""<<label<<"\",\"scope\":\"host_wall_API_calls_not_kernel_busy\",\"matrices\":"<<total<<",\"gpu_encode\":"<<(gpu_encode?"true":"false")<<",\"prepare_wall_sum_seconds_overlap_allowed\":"<<prepare_wall_sum<<",\"wait_work_wall_seconds\":"<<wait_work_wall<<",\"upload_call_wall_seconds\":"<<upload_call_wall<<",\"native_call_wall_seconds_may_include_internal_wait\":"<<native_call_wall<<",\"gpu_pipeline\":"<<(gpu_pipeline?"true":"false")<<",\"host_slot_wait_wall_seconds_subset_of_upload\":"<<(pipeline?pipeline->host_slot_wait_wall_seconds():0)<<",\"projection_finish_wall_seconds\":"<<projection_finish_wall<<",\"explicit_tile_sync_wall_seconds\":"<<tile_sync_wall<<",\"diagonal_data_h2d_bytes_excludes_metadata_tables\":"<<diagonal_data_h2d_bytes<<"}"<<std::endl;
        if(use_cache)std::cout<<"SOURCE_LINEAR_CACHE label="<<label<<" bytes="<<cache_bytes
            <<" budget="<<cache_budget<<" reuse_hits="<<cache_hits<<" graph_change=baby_reuse_only"<<std::endl;
        return output;
    }
};
} // namespace cerium_source
