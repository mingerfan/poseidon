// Optional GPU integer encoding after the unchanged CPU FFT/rounding stage.
// The plugin is explicitly selected by absolute path; default CPU behavior is
// unchanged and no extra CUDA kernel is linked into the original backend.
#pragma once
#include "llama_manual/periodic_plaintext.hpp"
#include <dlfcn.h>
#include <cstdint>
#include <climits>
#include <thread>
namespace llama_periodic {
class PersistentMatrixSession;
class PinnedCoefficientPool;
class GpuEncoder {
    friend class PersistentMatrixSession;
    using Launch=cudaError_t(*)(const std::int64_t*,GpuWord*,const GpuWord*,
        const GpuWord*,const GpuWord*,const GpuWord*,int,int,int,int,cudaStream_t);
    std::shared_ptr<void> library_;
    Launch launch_=nullptr;
    int device_,full_q_,full_p_;
    cudaStream_t stream_;
    const std::thread::id thread_=std::this_thread::get_id();
    std::shared_ptr<const void> identity_;
    std::map<parms_id_type,std::pair<std::size_t,std::size_t>> levels_;
    DeviceVector<GpuWord> moduli_,roots_,shoup_,permutation_;
    DeviceVector<std::int64_t> coefficients_;
    // Defined in the optional pipeline header. Lifetime spans projections;
    // never allocate/free pinned memory in every online projection again.
    std::shared_ptr<PinnedCoefficientPool> pinned_pool_;
    void validate(const std::map<int,Encoded> &diagonals,unsigned drops) const {
        if(std::this_thread::get_id()!=thread_||gpu_execution_stream()!=stream_)
            throw std::invalid_argument("GPU encoder execution stream/thread changed");
        if(diagonals.empty()||diagonals.size()>INT_MAX||!drops)throw std::invalid_argument("GPU periodic matrix count/drops");
        const auto &meta=diagonals.begin()->second.meta;
        if(!meta.q_count||meta.q_count>static_cast<std::size_t>(full_q_)||meta.p_count>static_cast<std::size_t>(full_p_)||meta.degree!=65536||!meta.is_ntt_form||!std::isfinite(meta.scale)||meta.scale<=1)
            throw std::invalid_argument("GPU periodic parameter contract");
        const auto level=levels_.find(meta.parms_id);
        if(level==levels_.end()||level->second!=std::make_pair(meta.q_count,meta.p_count))
            throw std::invalid_argument("GPU periodic context/id shape mismatch");
        for(const auto &[d,e]:diagonals)if(e.encoder_identity!=identity_||e.meta.parms_id!=meta.parms_id||e.meta.q_count!=meta.q_count||e.meta.p_count!=meta.p_count||e.meta.degree!=meta.degree||e.meta.scale!=meta.scale||e.meta.is_ntt_form!=meta.is_ntt_form)
            throw std::invalid_argument("GPU periodic diagonal origin/metadata mismatch");
    }
public:
    explicit GpuEncoder(const Encoder &encoder,int device=0):device_(device),stream_(gpu_execution_stream()) {
        int current=-1;gpu_check_cuda(cudaGetDevice(&current),"GPU periodic device");
        if(current!=device_)throw std::invalid_argument("GPU encoder requires current device");
        static_assert(sizeof(GpuWord)==4,"GPU periodic encoder requires 32-bit limbs");
        const char *path=std::getenv("LLAMA_PERIODIC_GPU_LIBRARY");
        if(!path||path[0]!='/')throw std::invalid_argument("LLAMA_PERIODIC_GPU_LIBRARY requires explicit absolute plugin path");
        void *handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);
        if(!handle)throw std::runtime_error(std::string("periodic GPU plugin load: ")+dlerror());
        library_=std::shared_ptr<void>(handle,[](void *p){dlclose(p);});
        dlerror();launch_=reinterpret_cast<Launch>(dlsym(handle,"llama_periodic_ntt_512_batch_v1"));
        if(const char *error=dlerror())throw std::runtime_error(std::string("periodic GPU symbol: ")+error);
        if(!launch_)throw std::runtime_error("periodic GPU symbol is null");
        const auto tables=encoder.gpu_tables_snapshot();identity_=tables.encoder_identity;levels_=tables.levels;full_q_=static_cast<int>(tables.full_q);full_p_=static_cast<int>(tables.moduli.size()-tables.full_q);
        auto upload=[&](auto &target,const auto &source){target.allocate(source.size(),device_);target.copy_from_host(source.data(),source.size());};
        upload(moduli_,tables.moduli);upload(roots_,tables.roots);upload(shoup_,tables.root_shoup);upload(permutation_,tables.permutation);
    }
    ~GpuEncoder() noexcept {
        // Exceptional early exits must also complete NTT before dlclose. A
        // device barrier handles destruction on another host thread (PTDS).
        if(library_&&cudaSetDevice(device_)==cudaSuccess)(void)cudaDeviceSynchronize();
    }
    // Single consumer, fixed execution stream/thread. No concurrent uploads.
    PackedMatrix upload(const std::map<int,Encoded> &diagonals,unsigned drops=1) {
        validate(diagonals,drops);
        const auto &meta=diagonals.begin()->second.meta;
        std::vector<const Encoded*> ordered;
        auto result=allocate_matrix_packed(diagonals,ordered,device_,drops,false);
        std::vector<std::int64_t> coefficients;coefficients.reserve(ordered.size()*compact_degree);
        for(const auto *encoded:ordered)coefficients.insert(coefficients.end(),encoded->coefficients.begin(),encoded->coefficients.end());
        if(coefficients_.size()!=coefficients.size()||!coefficients_.data())coefficients_.allocate(coefficients.size(),device_);
        coefficients_.copy_from_host(coefficients.data(),coefficients.size());
        gpu_check_cuda(launch_(coefficients_.data(),result.residues.data(),moduli_.data(),roots_.data(),shoup_.data(),permutation_.data(),full_q_,static_cast<int>(meta.q_count),static_cast<int>(meta.p_count),static_cast<int>(ordered.size()),stream_),"periodic GPU NTT launch");
        // Caller owns result through the last use. Encoder tables/plugin stay
        // alive until their NTT completes; destruction has a safety barrier.
        // Normal synchronization stays at the existing source tile boundary.
        return result;
    }
};
} // namespace llama_periodic
