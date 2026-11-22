// Fast linear projection path (opt-in: LLAMA_FAST_LINEAR=1).
//
// Same Cerium periodic-diagonal BSGS math as NativeLinearAdapter::run, but:
//  * weights are densified once per projection and kept on the GPU; all 255
//    diagonals of every input tile of an (output tile, lane) pair are encoded
//    by one GPU kernel (identical FFT/rounding/NTT as the CPU encoder);
//  * baby-step rotations of each input tile are computed once and shared by all
//    projections reading the same input (q/k/v and gate/up);
//  * accumulation across input tiles happens in the QP double-hoist
//    accumulators BEFORE the giant rotations (Cerium llama_cerium.py:121-129),
//    so giant rotations/ModDown/Rescale run once per (output tile, lane);
//  * no per-tile host synchronization.
// FHE plan difference vs. the default path: one Rescale after the cross-input
// sum instead of one per input tile (fewer rounding events; same Q/scale out).
#pragma once
#include "source_ops.hpp"
#include "fast_linear_gpu.h"
#include "llama_manual/checkpoint.hpp"
#include "llama_manual/periodic_plaintext.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace cerium_source {
inline bool fast_linear_enabled() {
    const char *v=std::getenv("LLAMA_FAST_LINEAR");return v&&*v&&std::string(v)!="0";
}
inline bool fast_linear_v2() {
    const char *v=std::getenv("LLAMA_FAST_LINEAR");return v&&std::string(v)=="2";
}
inline bool fast_linear_timing() {
    const char *v=std::getenv("LLAMA_FAST_LINEAR_TIMING");return v&&std::string(v)=="1";
}
class FastLinearEngine {
    Api &api_;
    const llama_periodic::Encoder &encoder_;
    llama_periodic::GpuTablesSnapshot tables_;
    poseidon::gpu::DeviceVector<int> slot_map_;
    poseidon::gpu::DeviceVector<double> inverse_roots_;
    poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> moduli_,roots_,shoup_,permutation_;
    poseidon::gpu::DeviceVector<int> overflow_;
    poseidon::gpu::DeviceVector<double> weights_;
    poseidon::gpu::DeviceVector<std::uint16_t> weights_bf16_;
    poseidon::gpu::DeviceVector<int> o_col_perm_;
    struct PinnedDoubles {
        double *ptr=nullptr;
        std::size_t cap=0;
        cudaEvent_t upload_done=nullptr;
        ~PinnedDoubles() {
            if(upload_done)cudaEventDestroy(upload_done);
            if(ptr)cudaFreeHost(ptr);
        }
        PinnedDoubles()=default;
        PinnedDoubles(const PinnedDoubles&)=delete;
        PinnedDoubles&operator=(const PinnedDoubles&)=delete;
        void wait() {
            if(upload_done)poseidon::gpu::gpu_check_cuda(cudaEventSynchronize(upload_done),"fast linear weight wait");
        }
        void ensure(std::size_t need) {
            wait();
            if(need<=cap)return;
            if(ptr){poseidon::gpu::gpu_check_cuda(cudaFreeHost(ptr),"fast linear free pinned weights");ptr=nullptr;cap=0;}
            poseidon::gpu::gpu_check_cuda(cudaHostAlloc(&ptr,need*sizeof(double),cudaHostAllocDefault),"fast linear alloc pinned weights");
            cap=need;
        }
        void mark_done(cudaStream_t stream) {
            if(!upload_done)poseidon::gpu::gpu_check_cuda(cudaEventCreateWithFlags(&upload_done,cudaEventDisableTiming),"fast linear weight event");
            poseidon::gpu::gpu_check_cuda(cudaEventRecord(upload_done,stream),"fast linear record weight upload");
        }
    } weights_host_;
    // Shared baby-step cache, bound to an explicit input generation.
    std::vector<poseidon::gpu::GpuDoubleHoistBabyCache> babies_;
    std::vector<const poseidon::gpu::GpuWord*> baby_sources_;
    // Q-only v2 buffers and cached rotated join masks.
    poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> babies_q_,encoded_q_,out_q_batch_;
    std::vector<CT> batch_out_;
    poseidon::parms_id_type babies_q_id_{};
    std::size_t babies_q_count_=0;
    bool babies_q_valid_=false;
    struct NonRopeMaskPair {
        poseidon::gpu::GpuPlaintextData geq128,lt128;
    };
    std::map<std::pair<poseidon::parms_id_type,double>,NonRopeMaskPair> nonrope_masks_;
    std::map<std::tuple<const void*,poseidon::parms_id_type,double>,
             std::map<int,poseidon::gpu::GpuPlaintextData>> rope_masks_cache_;
    // Per-projection-shape templates: one packed matrix per input tile.
    struct Template {
        std::vector<llama_periodic::PackedMatrix> matrices;
        std::vector<int> ordered_diagonals;
        poseidon::gpu::DeviceVector<int> diagonals_device;
        poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord*> outputs_device;
        poseidon::parms_id_type id{};double scale=0;std::size_t nin=0;
    };
    std::unique_ptr<Template> template_;
public:
    double encode_seconds=0,mac_seconds=0,giant_seconds=0,post_seconds=0,upload_seconds=0,baby_seconds=0;
    FastLinearEngine(Api &api,const llama_periodic::Encoder &encoder):api_(api),encoder_(encoder),tables_(encoder.gpu_tables_snapshot()) {
        const int dev=api_.device_id();
        std::vector<int> map(llama_periodic::compact_degree);
        for(std::size_t i=0;i<map.size();++i)map[i]=static_cast<int>(encoder.slot_map()[i]);
        std::vector<double> roots(2*llama_periodic::compact_degree);
        for(std::size_t i=0;i<llama_periodic::compact_degree;++i){roots[2*i]=encoder.inverse_roots()[i].real();roots[2*i+1]=encoder.inverse_roots()[i].imag();}
        auto up=[dev](auto &dst,const auto &src){dst.allocate(src.size(),dev);dst.copy_from_host(src.data(),src.size());};
        up(slot_map_,map);up(inverse_roots_,roots);up(moduli_,tables_.moduli);up(roots_,tables_.roots);
        up(shoup_,tables_.root_shoup);up(permutation_,tables_.permutation);
        std::vector<int> zero{0};up(overflow_,zero);
        std::vector<int> col_perm(4096);
        for(std::size_t i=0;i<4096;++i)col_perm[i]=static_cast<int>(cerium_source::layout::q_head_order(i/128)*128+i%128);
        up(o_col_perm_,col_perm);
    }
    // Callers must release before the shared input ciphertexts are freed.
    void release_input() {
        babies_.clear();baby_sources_.clear();batch_out_.clear();
        const char *keep=std::getenv("LLAMA_LINEAR_KEEP_WORKSPACE");
        if(!(keep&&std::string(keep)=="1")) {
            babies_q_.release();encoded_q_.release();out_q_batch_.release();
        }
        babies_q_count_=0;babies_q_valid_=false;
    }
    struct GpuBf16LayerShard {
        int device_id=0;
        std::vector<double> attention_norm,ffn_norm;
        poseidon::gpu::DeviceVector<std::uint16_t> q,k,v,o,gate,up,down;
        std::size_t q_rows=0,k_rows=0,v_rows=0,o_rows=0,gate_rows=0,up_rows=0,down_rows=0;
        std::size_t bytes() const noexcept {
            return (q.size()+k.size()+v.size()+o.size()+gate.size()+up.size()+down.size())*sizeof(std::uint16_t);
        }
    };
    static GpuBf16LayerShard upload_bf16_layer_shard(const llama_manual::LayerWeightsBf16Shard &src,int dev) {
        poseidon::gpu::gpu_check_cuda(cudaSetDevice(dev),"upload bf16 shard device");
        GpuBf16LayerShard dst;
        dst.device_id=dev;
        dst.attention_norm=src.attention_norm;
        dst.ffn_norm=src.ffn_norm;
        auto up_mat=[dev](const llama_manual::Bf16MatrixShard &m,poseidon::gpu::DeviceVector<std::uint16_t> &vec,std::size_t &rows) {
            rows=m.rows;
            vec.allocate(m.values.size(),dev);
            vec.copy_from_host(m.values.data(),m.values.size());
        };
        up_mat(src.q,dst.q,dst.q_rows);
        up_mat(src.k,dst.k,dst.k_rows);
        up_mat(src.v,dst.v,dst.v_rows);
        up_mat(src.o,dst.o,dst.o_rows);
        up_mat(src.gate,dst.gate,dst.gate_rows);
        up_mat(src.up,dst.up,dst.up_rows);
        up_mat(src.down,dst.down,dst.down_rows);
        return dst;
    }
    std::vector<CT> run_bf16_device(
        const std::vector<CT> &input,std::size_t in_width,std::size_t out_width,
        const std::uint16_t *weights_bf16_device,bool use_o_col_perm,
        bool with_rope,const std::string &label,
        const std::map<int,std::vector<std::complex<double>>> &rope_masks) {
        using namespace poseidon::gpu;
        using clock=std::chrono::steady_clock;
        const bool timing=fast_linear_timing();
        auto sync=[&]{if(timing)gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()),"fast linear bf16 timing");};
        auto since=[](clock::time_point t){return std::chrono::duration<double>(clock::now()-t).count();};
        const auto start=clock::now();
        validate_inputs(input,in_width,out_width);
        const auto q=input[0].meta.q_count;const auto id=input[0].meta.parms_id;
        const auto nin=input.size(),nout=out_width/256;
        const double scale=api_.prime(q);
        const auto level=tables_.levels.at(id);
        if(level.first!=q)throw std::runtime_error("fast Linear bf16 level Q mismatch");
        const auto &level_info=api_.parameters().get_level(id);
        const GpuParameterShard *shard=nullptr;
        for(const auto &cand:level_info.shards)if(cand.hybrid_base_q_count==q){shard=&cand;break;}
        if(!shard&&!level_info.shards.empty())shard=&level_info.shards.front();
        if(!shard||shard->q_primes.size()<q||shard->q_modulus_constants.size()<q||shard->device_id!=api_.device_id())
            throw std::runtime_error("fast Linear bf16 parameter shard mismatch");
        const cudaStream_t stream=gpu_execution_stream();
        bool reuse=babies_q_valid_&&babies_q_count_==nin&&babies_q_id_==id&&baby_sources_.size()==nin;
        for(std::size_t i=0;reuse&&i<nin;++i)
            reuse=baby_sources_[i]==input[i].make_const_view().polys[0].shards.front().ptr;
        auto t=clock::now();
        if(!reuse) {
            babies_.clear();baby_sources_.clear();
            const std::size_t q_words=q*65536ULL;
            const std::size_t total_words=nin*16ULL*2ULL*q_words;
            if(babies_q_.device_id()!=shard->device_id||babies_q_.size()<total_words)
                babies_q_.allocate(total_words,shard->device_id);
            static const std::vector<int> steps15={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
            for(std::size_t it=0;it<nin;++it) {
                const auto src0=input[it].make_const_view();
                baby_sources_.push_back(src0.polys[0].shards.front().ptr);
                GpuWord *it_base=babies_q_.data()+it*32ULL*q_words;
                for(int c=0;c<2;++c) {
                    gpu_check_cuda(cudaMemcpyAsync(it_base+static_cast<std::size_t>(c)*q_words,
                        src0.polys[c].shards.front().ptr,
                        q_words*sizeof(GpuWord),cudaMemcpyDeviceToDevice,stream),"fast bf16 baby0 copy");
                }
            }
            for(std::size_t it0=0;it0<nin;it0+=8) {
                const std::size_t count=std::min<std::size_t>(8,nin-it0);
                std::vector<const poseidon::gpu::GpuCiphertextData*> bsrcs;
                std::vector<poseidon::gpu::GpuWord*> bdsts;
                bsrcs.reserve(count);bdsts.reserve(count);
                for(std::size_t k=0;k<count;++k) {
                    const std::size_t it=it0+k;
                    bsrcs.push_back(&input[it]);
                    bdsts.push_back(babies_q_.data()+it*32ULL*q_words+2ULL*q_words);
                }
                api_.eval().rotate_many_hoisted_batch_contiguous(
                    bsrcs,steps15,api_.galois_keys(),api_.double_hoist_workspace(),bdsts);
            }
            babies_q_id_=id;babies_q_count_=nin;babies_q_valid_=true;
            sync();baby_seconds+=since(t);
        }
        const std::size_t total_mats=nout*2;
        const std::size_t batch_cap=std::min<std::size_t>(total_mats,8);
        const std::size_t need_enc=batch_cap*nin*256ULL*q*512ULL;
        const std::size_t need_out=batch_cap*16ULL*2ULL*q*65536ULL;
        if(encoded_q_.device_id()!=shard->device_id||encoded_q_.size()<need_enc)
            encoded_q_.allocate(need_enc,shard->device_id);
        if(out_q_batch_.device_id()!=shard->device_id||out_q_batch_.size()<need_out)
            out_q_batch_.allocate(need_out,shard->device_id);
        static const std::vector<int> giant_steps_v2={
            0,16,32,48,64,80,96,112,-128,-112,-96,-80,-64,-48,-32,-16};
        const double product_scale=std::exp2(40.)*scale;
        fast_linear::BatchEncodeQArgs eargs;
        eargs.weights=nullptr;
        eargs.weights_bf16=weights_bf16_device;
        eargs.col_perm=use_o_col_perm?o_col_perm_.data():nullptr;
        eargs.columns=static_cast<long long>(in_width);
        eargs.slot_map=slot_map_.data();eargs.inverse_roots=reinterpret_cast<const double2*>(inverse_roots_.data());
        eargs.fix=scale/llama_periodic::compact_degree;eargs.coefficient_limit=encoder_.coefficient_limit(id);
        eargs.moduli=moduli_.data();eargs.ntt_roots=roots_.data();eargs.ntt_shoup=shoup_.data();
        eargs.permutation=permutation_.data();eargs.nq=static_cast<int>(q);eargs.nin=static_cast<int>(nin);
        eargs.encoded_q=encoded_q_.data();eargs.overflow=overflow_.data();
        fast_linear::BatchMacQArgs margs;
        margs.babies_q=babies_q_.data();margs.encoded_q=encoded_q_.data();margs.out_q_batch=out_q_batch_.data();
        margs.q_primes=shard->q_primes.data();margs.q_barrett=shard->q_modulus_constants.data();
        margs.nq=static_cast<int>(q);margs.nin=static_cast<int>(nin);
        std::vector<CT> output;output.reserve(nout);
        for(std::size_t m0=0;m0<total_mats;m0+=8) {
            const int B=static_cast<int>(std::min<std::size_t>(8,total_mats-m0));
            for(int k=0;k<B;++k){const std::size_t idx=m0+static_cast<std::size_t>(k);eargs.ots[k]=static_cast<int>(idx/2);eargs.lanes[k]=static_cast<int>(idx&1);}
            eargs.batch_count=B;margs.batch_count=B;
            t=clock::now();
            gpu_check_cuda(fast_linear::launch_batch_encode_q_diagonals(eargs,stream),"fast bf16 encode");
            sync();encode_seconds+=since(t);t=clock::now();
            gpu_check_cuda(fast_linear::launch_batched_compact_mac_q(margs,stream),"fast bf16 mac");
            sync();mac_seconds+=since(t);
            const std::size_t out_stride=16ULL*2ULL*q*65536ULL;
            t=clock::now();
            api_.eval().giant_reduce_q_groups_batch(
                out_q_batch_.data(),static_cast<std::size_t>(B),out_stride,
                q,id,product_scale,giant_steps_v2,api_.galois_keys(),1,
                api_.double_hoist_workspace(),batch_out_);
            sync();giant_seconds+=since(t);
            t=clock::now();
            for(int k=0;k<B;k+=2) {
                const auto &b0=batch_out_[static_cast<std::size_t>(k)];
                const auto &b1=batch_out_[static_cast<std::size_t>(k+1)];
                api_.require_scale(b0,std::exp2(40.));
                api_.require_scale(b1,std::exp2(40.));
                if(b0.meta.q_count!=q-1||b1.meta.q_count!=q-1)
                    throw std::runtime_error("fast bf16 DFT output Q");
                join_lanes_pair(b0,b1,with_rope,q,rope_masks,output);
            }
            sync();post_seconds+=since(t);
        }
        int overflow=0;overflow_.copy_to_host(&overflow,1);
        if(overflow)throw std::overflow_error("fast linear bf16 compact coefficient signed range");
        return output;
    }
    template<class Weight>
    std::vector<CT> run(const std::vector<CT> &input,std::size_t in_width,std::size_t out_width,
                        Weight weight,bool with_rope,const std::string &label,
                        const std::map<int,std::vector<std::complex<double>>> &rope_masks) {
        if(fast_linear_v2())
            return run_v2(input,in_width,out_width,weight,with_rope,label,rope_masks);
        using namespace poseidon::gpu;
        using clock=std::chrono::steady_clock;
        const bool timing=fast_linear_timing();
        auto sync=[&]{if(timing)gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()),"fast linear timing");};
        auto since=[](clock::time_point t){return std::chrono::duration<double>(clock::now()-t).count();};
        const auto start=clock::now();
        validate_inputs(input,in_width,out_width);
        const auto q=input[0].meta.q_count;const auto id=input[0].meta.parms_id;
        const auto nin=input.size(),nout=out_width/256;
        const auto p_lvl=api_.level_p_count(q);
        const double scale=api_.prime(q);
        prepare_template(id,scale,nin,p_lvl);
        auto &tpl=*template_;
        // Shared babies: valid while the caller keeps the same input alive.
        bool reuse=babies_.size()==nin;
        for(std::size_t i=0;reuse&&i<nin;++i)
            reuse=baby_sources_[i]==input[i].make_const_view().polys[0].shards.front().ptr&&babies_[i].ready&&
                  babies_[i].source_meta.parms_id==id;
        auto t=clock::now();
        if(!reuse) {
            babies_.clear();baby_sources_.clear();babies_.resize(nin);
            const auto &steps=tpl.matrices[0].group.data()[0].plan.baby_steps;
            for(std::size_t i=0;i<nin;++i) {
                api_.prepare_linear_babies(input[i],steps,babies_[i]);
                baby_sources_.push_back(input[i].make_const_view().polys[0].shards.front().ptr);
            }
            sync();baby_seconds+=since(t);
        }
        t=clock::now();
        upload_weights(in_width,out_width,weight);
        upload_seconds+=since(t);
        fast_linear::EncodeArgs args;
        args.weights=weights_.data();args.columns=static_cast<long long>(in_width);args.it_begin=0;
        args.diagonals=tpl.diagonals_device.data();args.slot_map=slot_map_.data();
        args.inverse_roots=reinterpret_cast<const double2*>(inverse_roots_.data());
        args.fix=scale/llama_periodic::compact_degree;args.coefficient_limit=encoder_.coefficient_limit(id);
        args.moduli=moduli_.data();args.ntt_roots=roots_.data();args.ntt_shoup=shoup_.data();args.permutation=permutation_.data();
        args.full_q=static_cast<int>(tables_.full_q);
        const auto level=tables_.levels.at(id);
        args.nq=static_cast<int>(level.first);args.np=static_cast<int>(p_lvl);
        args.outputs=tpl.outputs_device.data();args.overflow=overflow_.data();
        std::array<GpuDoubleHoistSumState,2> sums;
        std::vector<CT> output;std::array<std::optional<CT>,2> branch;
        for(std::size_t ot=0;ot<nout;++ot)for(int lane=0;lane<2;++lane) {
            t=clock::now();
            args.ot=static_cast<int>(ot);args.lane=lane;
            gpu_check_cuda(fast_linear::launch_encode_diagonals(args,static_cast<int>(tpl.ordered_diagonals.size()),
                static_cast<int>(nin),gpu_execution_stream()),"fast linear encode");
            sync();encode_seconds+=since(t);t=clock::now();
            CT term;
            for(std::size_t it=0;it<nin;++it)
                term=api_.native_dft_sum(input[it],tpl.matrices[it].group,sums[lane],it+1==nin,&babies_[it]);
            sync();mac_seconds+=since(t);t=clock::now();
            api_.require_scale(term,std::exp2(40.));
            if(term.meta.q_count!=q-1)throw std::runtime_error("fast DFT output Q");
            branch[lane]=std::move(term);
            if(lane==1) {
                join_lanes_pair(*branch[0],*branch[1],with_rope,q,rope_masks,output);
                branch[0].reset();branch[1].reset();
                sync();post_seconds+=since(t);
            }
        }
        int overflow=0;overflow_.copy_to_host(&overflow,1);
        if(overflow)throw std::overflow_error("fast linear compact coefficient signed range");
        std::cout<<"FAST_LINEAR label="<<label<<" tiles="<<nout<<" input_q="<<q<<" output_q="<<output.back().meta.q_count
            <<" fused_rope="<<with_rope<<" baby_reuse="<<reuse<<" seconds="<<since(start)
            <<" cumulative_encode="<<encode_seconds<<" mac="<<mac_seconds<<" post="<<post_seconds
            <<" upload="<<upload_seconds<<" baby="<<baby_seconds<<" timing_sync="<<timing<<std::endl;
        return output;
    }
private:
    static void validate_inputs(const std::vector<CT> &input,std::size_t in_width,std::size_t out_width) {
        if(input.size()!=in_width/256||in_width%256||out_width%256||input.empty())
            throw std::invalid_argument("fast Linear dimensions");
        const auto q=input[0].meta.q_count;const auto id=input[0].meta.parms_id;
        for(const auto &x:input)if(x.meta.parms_id!=id||!Api::same_scale(x.meta.scale,std::exp2(40.)))
            throw std::invalid_argument("fast Linear requires aligned scale40 inputs");
        if(q<8)throw std::runtime_error("fast Linear output below native bootstrap input budget");
    }
    template<class Weight>
    void upload_weights(std::size_t in_width,std::size_t out_width,Weight weight) {
        const std::size_t need=out_width*in_width;
        weights_host_.ensure(need);
        #pragma omp parallel for num_threads(16) schedule(static)
        for(long long r=0;r<static_cast<long long>(out_width);++r)
            for(std::size_t c=0;c<in_width;++c)weights_host_.ptr[r*in_width+c]=weight(static_cast<std::size_t>(r),c);
        if(weights_.device_id()!=api_.device_id()||weights_.size()<need)weights_.allocate(need,api_.device_id());
        const cudaStream_t stream=poseidon::gpu::gpu_execution_stream();
        poseidon::gpu::gpu_check_cuda(
            cudaMemcpyAsync(weights_.data(),weights_host_.ptr,need*sizeof(double),cudaMemcpyHostToDevice,stream),
            "fast linear upload weights");
        weights_host_.mark_done(stream);
    }
    const NonRopeMaskPair &cached_nonrope_masks(const CT &sample,double mask_scale) {
        const auto key=std::make_pair(sample.meta.parms_id,mask_scale);
        auto it=nonrope_masks_.find(key);
        if(it!=nonrope_masks_.end())return it->second;
        std::vector<double> geq128(32768),lt128(32768);
        for(std::size_t j=0;j<32768;++j){const bool hi=(j%256)>=128;geq128[j]=hi?1.:0.;lt128[j]=hi?0.:1.;}
        NonRopeMaskPair pair{api_.prepare_vector(sample,geq128,mask_scale),
                             api_.prepare_vector(sample,lt128,mask_scale)};
        return nonrope_masks_.emplace(key,std::move(pair)).first->second;
    }
    const std::map<int,poseidon::gpu::GpuPlaintextData> &cached_rope_masks(
            const CT &sample,const std::map<int,std::vector<std::complex<double>>> &rope_masks,double mask_scale) {
        const auto key=std::make_tuple(static_cast<const void*>(&rope_masks),sample.meta.parms_id,mask_scale);
        auto it=rope_masks_cache_.find(key);
        if(it!=rope_masks_cache_.end())return it->second;
        static constexpr int kSteps[9]={-1,0,1,-129,-128,-127,127,128,129};
        std::vector<std::vector<std::complex<double>>> rotated_list(9,std::vector<std::complex<double>>(32768));
        #pragma omp parallel for num_threads(9) schedule(static)
        for(int idx=0;idx<9;++idx) {
            const int d=kSteps[idx];
            const auto &src=rope_masks.at(d);
            if(src.size()!=32768)throw std::invalid_argument("fast linear rope mask size mismatch");
            const std::size_t shift=static_cast<std::size_t>((d%32768+32768)%32768);
            auto &rotated=rotated_list[static_cast<std::size_t>(idx)];
            for(std::size_t j=0;j<32768;++j)rotated[j]=src[(j+shift)%32768];
        }
        auto pts=api_.prepare_complex_batch(sample,rotated_list,mask_scale);
        std::map<int,poseidon::gpu::GpuPlaintextData> encoded;
        for(int idx=0;idx<9;++idx)encoded.emplace(kSteps[idx],std::move(pts[static_cast<std::size_t>(idx)]));
        return rope_masks_cache_.emplace(key,std::move(encoded)).first->second;
    }
    void join_lanes_pair(const CT &b0,const CT &b1,bool with_rope,std::size_t q,
                         const std::map<int,std::vector<std::complex<double>>> &rope_masks,
                         std::vector<CT> &output) {
        const double mask_scale=api_.prime(q-1);
        if(with_rope) {
            const auto &pts=cached_rope_masks(b0,rope_masks,mask_scale);
            auto rots0=api_.hoisted(b0,{-1,1});
            CT acc=api_.cp_prepared(rots0[0],pts.at(-1));
            acc=api_.add(acc,api_.cp_prepared(b0,pts.at(0)));
            acc=api_.add(acc,api_.cp_prepared(rots0[1],pts.at(1)));
            auto rots1=api_.hoisted(b1,{-129,-128,-127,127,128,129});
            static constexpr int steps1[6]={-129,-128,-127,127,128,129};
            for(int i=0;i<6;++i)acc=api_.add(acc,api_.cp_prepared(rots1[i],pts.at(steps1[i])));
            acc=api_.add(acc,api_.conjugate(acc));
            output.push_back(api_.rescale(acc,1));
        } else {
            CT term0;
            api_.eval().multiply_scalar(b0,static_cast<std::uint64_t>(std::llround(mask_scale)),term0);
            term0.meta.scale=b0.meta.scale*mask_scale;
            const auto &masks=cached_nonrope_masks(b1,mask_scale);
            auto rots1=api_.hoisted(b1,{-128,128});
            CT acc=api_.add(term0,api_.cp_prepared(rots1[0],masks.geq128));
            acc=api_.add(acc,api_.cp_prepared(rots1[1],masks.lt128));
            output.push_back(api_.rescale(acc,1));
        }
    }
    template<class Weight>
    std::vector<CT> run_v2(const std::vector<CT> &input,std::size_t in_width,std::size_t out_width,
                           Weight weight,bool with_rope,const std::string &label,
                           const std::map<int,std::vector<std::complex<double>>> &rope_masks) {
        using namespace poseidon::gpu;
        using clock=std::chrono::steady_clock;
        const bool timing=fast_linear_timing();
        auto sync=[&]{if(timing)gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()),"fast linear v2 timing");};
        auto since=[](clock::time_point t){return std::chrono::duration<double>(clock::now()-t).count();};
        const auto start=clock::now();
        validate_inputs(input,in_width,out_width);
        const auto q=input[0].meta.q_count;const auto id=input[0].meta.parms_id;
        const auto nin=input.size(),nout=out_width/256;
        const double scale=api_.prime(q);
        const auto level=tables_.levels.at(id);
        if(level.first!=q)throw std::runtime_error("fast Linear v2 level Q mismatch");
        const auto &level_info=api_.parameters().get_level(id);
        const GpuParameterShard *shard=nullptr;
        for(const auto &cand:level_info.shards)if(cand.hybrid_base_q_count==q){shard=&cand;break;}
        if(!shard&&!level_info.shards.empty())shard=&level_info.shards.front();
        if(!shard||shard->q_primes.size()<q||shard->q_modulus_constants.size()<q)
            throw std::runtime_error("fast Linear v2 parameter shard mismatch");
        const cudaStream_t stream=gpu_execution_stream();
        bool reuse=babies_q_valid_&&babies_q_count_==nin&&babies_q_id_==id&&baby_sources_.size()==nin;
        for(std::size_t i=0;reuse&&i<nin;++i)
            reuse=baby_sources_[i]==input[i].make_const_view().polys[0].shards.front().ptr;
        auto t=clock::now();
        if(!reuse) {
            babies_.clear();baby_sources_.clear();
            const std::size_t q_words=q*65536ULL;
            const std::size_t total_words=nin*16ULL*2ULL*q_words;
            if(babies_q_.device_id()!=shard->device_id||babies_q_.size()<total_words)
                babies_q_.allocate(total_words,shard->device_id);
            static const std::vector<int> steps15={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
            for(std::size_t it=0;it<nin;++it) {
                const auto src0=input[it].make_const_view();
                baby_sources_.push_back(src0.polys[0].shards.front().ptr);
                GpuWord *it_base=babies_q_.data()+it*32ULL*q_words;
                for(int c=0;c<2;++c) {
                    gpu_check_cuda(cudaMemcpyAsync(it_base+static_cast<std::size_t>(c)*q_words,
                        src0.polys[c].shards.front().ptr,
                        q_words*sizeof(GpuWord),cudaMemcpyDeviceToDevice,stream),"fast v2 baby0 copy");
                }
            }
            for(std::size_t it0=0;it0<nin;it0+=8) {
                const std::size_t count=std::min<std::size_t>(8,nin-it0);
                std::vector<const poseidon::gpu::GpuCiphertextData*> bsrcs;
                std::vector<poseidon::gpu::GpuWord*> bdsts;
                bsrcs.reserve(count);bdsts.reserve(count);
                for(std::size_t k=0;k<count;++k) {
                    const std::size_t it=it0+k;
                    bsrcs.push_back(&input[it]);
                    bdsts.push_back(babies_q_.data()+it*32ULL*q_words+2ULL*q_words);
                }
                api_.eval().rotate_many_hoisted_batch_contiguous(
                    bsrcs,steps15,api_.galois_keys(),api_.double_hoist_workspace(),bdsts);
            }
            babies_q_id_=id;babies_q_count_=nin;babies_q_valid_=true;
            sync();baby_seconds+=since(t);
        }
        t=clock::now();
        upload_weights(in_width,out_width,weight);
        upload_seconds+=since(t);
        const std::size_t total_mats=nout*2;
        const std::size_t batch_cap=std::min<std::size_t>(total_mats,8);
        const std::size_t need_enc=batch_cap*nin*256ULL*q*512ULL;
        const std::size_t need_out=batch_cap*16ULL*2ULL*q*65536ULL;
        if(encoded_q_.device_id()!=shard->device_id||encoded_q_.size()<need_enc)
            encoded_q_.allocate(need_enc,shard->device_id);
        if(out_q_batch_.device_id()!=shard->device_id||out_q_batch_.size()<need_out)
            out_q_batch_.allocate(need_out,shard->device_id);
        static const std::vector<int> giant_steps_v2={
            0,16,32,48,64,80,96,112,-128,-112,-96,-80,-64,-48,-32,-16};
        const double product_scale=std::exp2(40.)*scale;
        fast_linear::BatchEncodeQArgs eargs;
        eargs.weights=weights_.data();eargs.columns=static_cast<long long>(in_width);
        eargs.slot_map=slot_map_.data();eargs.inverse_roots=reinterpret_cast<const double2*>(inverse_roots_.data());
        eargs.fix=scale/llama_periodic::compact_degree;eargs.coefficient_limit=encoder_.coefficient_limit(id);
        eargs.moduli=moduli_.data();eargs.ntt_roots=roots_.data();eargs.ntt_shoup=shoup_.data();
        eargs.permutation=permutation_.data();eargs.nq=static_cast<int>(q);eargs.nin=static_cast<int>(nin);
        eargs.encoded_q=encoded_q_.data();eargs.overflow=overflow_.data();
        fast_linear::BatchMacQArgs margs;
        margs.babies_q=babies_q_.data();margs.encoded_q=encoded_q_.data();margs.out_q_batch=out_q_batch_.data();
        margs.q_primes=shard->q_primes.data();margs.q_barrett=shard->q_modulus_constants.data();
        margs.nq=static_cast<int>(q);margs.nin=static_cast<int>(nin);
        std::vector<CT> output;output.reserve(nout);
        for(std::size_t m0=0;m0<total_mats;m0+=8) {
            const int B=static_cast<int>(std::min<std::size_t>(8,total_mats-m0));
            for(int k=0;k<B;++k){const std::size_t idx=m0+static_cast<std::size_t>(k);eargs.ots[k]=static_cast<int>(idx/2);eargs.lanes[k]=static_cast<int>(idx&1);}
            eargs.batch_count=B;margs.batch_count=B;
            t=clock::now();
            gpu_check_cuda(fast_linear::launch_batch_encode_q_diagonals(eargs,stream),"fast v2 encode");
            sync();encode_seconds+=since(t);t=clock::now();
            gpu_check_cuda(fast_linear::launch_batched_compact_mac_q(margs,stream),"fast v2 mac");
            sync();mac_seconds+=since(t);
            const std::size_t out_stride=16ULL*2ULL*q*65536ULL;
            t=clock::now();
            api_.eval().giant_reduce_q_groups_batch(
                out_q_batch_.data(),static_cast<std::size_t>(B),out_stride,
                q,id,product_scale,giant_steps_v2,api_.galois_keys(),1,
                api_.double_hoist_workspace(),batch_out_);
            sync();giant_seconds+=since(t);
            t=clock::now();
            for(int k=0;k<B;k+=2) {
                const auto &b0=batch_out_[static_cast<std::size_t>(k)];
                const auto &b1=batch_out_[static_cast<std::size_t>(k+1)];
                api_.require_scale(b0,std::exp2(40.));
                api_.require_scale(b1,std::exp2(40.));
                if(b0.meta.q_count!=q-1||b1.meta.q_count!=q-1)
                    throw std::runtime_error("fast v2 DFT output Q");
                join_lanes_pair(b0,b1,with_rope,q,rope_masks,output);
            }
            sync();post_seconds+=since(t);
        }
        int overflow=0;overflow_.copy_to_host(&overflow,1);
        if(overflow)throw std::overflow_error("fast linear v2 compact coefficient signed range");
        std::cout<<"FAST_LINEAR_V2 label="<<label<<" tiles="<<nout<<" input_q="<<q<<" output_q="<<output.back().meta.q_count
            <<" fused_rope="<<with_rope<<" baby_reuse="<<reuse<<" seconds="<<since(start)
            <<" cumulative_encode="<<encode_seconds<<" mac="<<mac_seconds<<" giant="<<giant_seconds
            <<" post="<<post_seconds<<" upload="<<upload_seconds<<" baby="<<baby_seconds<<" timing_sync="<<timing<<std::endl;
        return output;
    }
    void prepare_template(poseidon::parms_id_type id,double scale,std::size_t nin,std::size_t p_lvl) {
        using namespace poseidon::gpu;
        if(template_&&template_->id==id&&template_->scale==scale&&template_->nin>=nin)return;
        template_.reset();
        auto tpl=std::make_unique<Template>();tpl->id=id;tpl->scale=scale;tpl->nin=nin;
        const auto level=tables_.levels.at(id);
        std::map<int,llama_periodic::Encoded> shape;
        for(int d=-127;d<=127;++d) {
            llama_periodic::Encoded e;e.meta={id,scale,true,65536,level.first,p_lvl};
            e.encoder_identity=tables_.encoder_identity;shape.emplace(d,std::move(e));
        }
        std::map<const llama_periodic::Encoded*,int> back;
        for(const auto &[d,e]:shape)back.emplace(&e,d);
        std::vector<GpuWord*> outputs;
        for(std::size_t it=0;it<nin;++it) {
            std::vector<const llama_periodic::Encoded*> ordered;
            tpl->matrices.push_back(llama_periodic::allocate_matrix_packed(shape,ordered,api_.device_id(),1,false));
            if(it==0)for(const auto *e:ordered)tpl->ordered_diagonals.push_back(back.at(e));
            outputs.push_back(tpl->matrices.back().residues.data());
        }
        tpl->diagonals_device.allocate(tpl->ordered_diagonals.size(),api_.device_id());
        tpl->diagonals_device.copy_from_host(tpl->ordered_diagonals.data(),tpl->ordered_diagonals.size());
        tpl->outputs_device.allocate(outputs.size(),api_.device_id());tpl->outputs_device.copy_from_host(outputs.data(),outputs.size());
        template_=std::move(tpl);
    }
};
} // namespace cerium_source
