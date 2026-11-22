// Optional same-stream projection session. Complete DFT submission stays in
// consume(); no view may escape/transfer threads. Encoder/MR/pool/input outlive
// the session. Pinned host slots belong to the encoder across projections.
#pragma once
#include "llama_manual/periodic_plaintext_gpu.hpp"
#include <chrono>
#include <cstring>
#include <type_traits>
#include <utility>
namespace llama_periodic {
class PinnedCoefficientPool {
    friend class PersistentMatrixSession;
    struct Slot {
        std::int64_t *data=nullptr;cudaEvent_t done=nullptr;bool inflight=false,recorded=false;
        Slot()=default;Slot(const Slot&)=delete;Slot&operator=(const Slot&)=delete;
        Slot&operator=(Slot&&x)noexcept{if(this!=&x){release();data=std::exchange(x.data,nullptr);done=std::exchange(x.done,nullptr);inflight=std::exchange(x.inflight,false);recorded=std::exchange(x.recorded,false);}return *this;}
        void release()noexcept{if(done)(void)cudaEventDestroy(done);if(data)(void)cudaFreeHost(data);done=nullptr;data=nullptr;}
        ~Slot(){release();}
    };
    int device_;cudaStream_t stream_;std::thread::id thread_;
    std::array<Slot,2> slots_;std::size_t capacity_=0;unsigned next_=0;
    void drain_noexcept()noexcept{if(cudaSetDevice(device_)!=cudaSuccess)return;if(std::this_thread::get_id()==thread_)(void)cudaStreamSynchronize(stream_);else(void)cudaDeviceSynchronize();}
    void ensure(std::size_t count){
        if(count<=capacity_)return;
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(std::int64_t))throw std::overflow_error("pinned coefficients size");
        gpu_check_cuda(cudaStreamSynchronize(stream_),"pinned pool resize drain");
        // Transactional growth. Partial host/event allocation failure leaves
        // the prior, already-drained pool usable by a fresh session.
        std::array<Slot,2> fresh;
        for(auto &slot:fresh){gpu_check_cuda(cudaMallocHost(reinterpret_cast<void**>(&slot.data),count*sizeof(std::int64_t)),"pinned periodic coefficients");gpu_check_cuda(cudaEventCreateWithFlags(&slot.done,cudaEventDisableTiming),"pinned periodic event");}
        slots_=std::move(fresh);capacity_=count;next_=0;
    }
public:
    PinnedCoefficientPool(int device,cudaStream_t stream,std::thread::id thread):device_(device),stream_(stream),thread_(thread){}
    ~PinnedCoefficientPool(){drain_noexcept();}
    PinnedCoefficientPool(const PinnedCoefficientPool&)=delete;
    PinnedCoefficientPool&operator=(const PinnedCoefficientPool&)=delete;
};
class PersistentMatrixSession {
    GpuEncoder &owner_;
    std::shared_ptr<PinnedCoefficientPool> pinned_;
    DeviceVector<std::int64_t> coefficients_;
    std::unique_ptr<PackedMatrix> matrix_;
    GpuPlaintextMeta meta_{};unsigned drops_=0;
    std::vector<int> keys_,order_;
    bool poisoned_=false,finished_=false,busy_=false;
    struct BusyGuard{bool &busy;explicit BusyGuard(bool &b):busy(b){busy=true;}~BusyGuard(){busy=false;}};
    double slot_wait_wall_=0;
    void check_consumer()const{
        if(std::this_thread::get_id()!=owner_.thread_||gpu_execution_stream()!=owner_.stream_)throw std::invalid_argument("periodic pipeline execution stream/thread changed");
        int device=-1;gpu_check_cuda(cudaGetDevice(&device),"periodic pipeline device");if(device!=owner_.device_)throw std::invalid_argument("periodic pipeline device changed");
    }
    void drain_noexcept()noexcept{if(cudaSetDevice(owner_.device_)!=cudaSuccess)return;if(std::this_thread::get_id()==owner_.thread_)(void)cudaStreamSynchronize(owner_.stream_);else(void)cudaDeviceSynchronize();}
    void configure(const std::map<int,Encoded>&diagonals,unsigned drops){
        std::vector<int> keys;keys.reserve(diagonals.size());for(const auto &[d,e]:diagonals)keys.push_back(d);const auto &meta=diagonals.begin()->second.meta;
        if(matrix_){if(keys!=keys_||drops!=drops_||meta.parms_id!=meta_.parms_id||meta.scale!=meta_.scale||meta.q_count!=meta_.q_count||meta.p_count!=meta_.p_count||meta.degree!=meta_.degree)throw std::invalid_argument("periodic pipeline shape changed; use a new projection session");return;}
        // Keep synchronous pageable descriptor uploads, once per projection.
        std::vector<const Encoded*> ordered;matrix_=std::make_unique<PackedMatrix>(allocate_matrix_packed(diagonals,ordered,owner_.device_,drops,false));std::map<const Encoded*,int> ids;for(const auto &[d,e]:diagonals)ids.emplace(&e,d);for(const auto *encoded:ordered)order_.push_back(ids.at(encoded));keys_=std::move(keys);meta_=meta;drops_=drops;
        const auto count=order_.size()*compact_degree;coefficients_.allocate(count,owner_.device_);pinned_->ensure(count);
    }
public:
    explicit PersistentMatrixSession(GpuEncoder&owner):owner_(owner){check_consumer();if(!owner_.pinned_pool_)owner_.pinned_pool_=std::make_shared<PinnedCoefficientPool>(owner_.device_,owner_.stream_,owner_.thread_);pinned_=owner_.pinned_pool_;}
    PersistentMatrixSession(const PersistentMatrixSession&)=delete;
    PersistentMatrixSession&operator=(const PersistentMatrixSession&)=delete;
    ~PersistentMatrixSession(){drain_noexcept();}
    template<class Consumer>auto consume(const std::map<int,Encoded>&diagonals,unsigned drops,Consumer&&consumer){
        using Result=std::invoke_result_t<Consumer,const GpuLinearMatrixGroupQP&>;static_assert(!std::is_reference_v<Result>&&!std::is_pointer_v<Result>,"consumer must not return an escaping group view");check_consumer();if(busy_)throw std::logic_error("periodic pipeline callback reentrancy");BusyGuard guard(busy_);
        try{
            if(poisoned_||finished_)throw std::logic_error("periodic pipeline is closed/poisoned");owner_.validate(diagonals,drops);configure(diagonals,drops);
            auto &slot=pinned_->slots_[pinned_->next_];pinned_->next_^=1;
            if(slot.inflight){auto start=std::chrono::steady_clock::now();if(slot.recorded)gpu_check_cuda(cudaEventSynchronize(slot.done),"periodic pinned slot wait");else gpu_check_cuda(cudaStreamSynchronize(owner_.stream_),"periodic unrecorded copy drain");slot_wait_wall_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();slot.inflight=false;slot.recorded=false;}
            for(std::size_t i=0;i<order_.size();++i)std::memcpy(slot.data+i*compact_degree,diagonals.at(order_[i]).coefficients.data(),compact_degree*sizeof(std::int64_t));
            slot.inflight=true;slot.recorded=false;
            gpu_check_cuda(cudaMemcpyAsync(coefficients_.data(),slot.data,coefficients_.size()*sizeof(std::int64_t),cudaMemcpyHostToDevice,owner_.stream_),"periodic pinned H2D");gpu_check_cuda(cudaEventRecord(slot.done,owner_.stream_),"periodic H2D event");slot.recorded=true;
            gpu_check_cuda(owner_.launch_(coefficients_.data(),matrix_->residues.data(),owner_.moduli_.data(),owner_.roots_.data(),owner_.shoup_.data(),owner_.permutation_.data(),owner_.full_q_,static_cast<int>(meta_.q_count),static_cast<int>(meta_.p_count),static_cast<int>(order_.size()),owner_.stream_),"periodic pipeline NTT");
            return std::forward<Consumer>(consumer)(static_cast<const GpuLinearMatrixGroupQP&>(matrix_->group));
        }catch(...){poisoned_=true;drain_noexcept();throw;}
    }
    void finish(){check_consumer();if(busy_)throw std::logic_error("periodic pipeline finish inside callback");try{if(poisoned_)throw std::logic_error("poisoned periodic pipeline");gpu_check_cuda(cudaStreamSynchronize(owner_.stream_),"periodic projection finish");finished_=true;}catch(...){poisoned_=true;drain_noexcept();throw;}}
    double host_slot_wait_wall_seconds()const noexcept{return slot_wait_wall_;}
};
} // namespace llama_periodic
