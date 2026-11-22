// Cerium's periodic-plaintext representation, adapted to the ResNet20 backend.
// Reference: CMU-CAOS/Cerium 6b487041, runtime/io/ckks-encoder.h.
// This host-only adapter uses the backend's FFT/NTT, generator 5 and rounding.
// CPU reference stays here. Optional GPU integer NTT is implemented separately
// in periodic_plaintext_gpu.hpp; ciphertext evaluation is unchanged. See PORTING.md.
#pragma once
#include "poseidon/ckks_encoder.h"
#include "poseidon/gpu/gpu_double_hoist.h"
#include <array>
#include <algorithm>
#include <map>
#include <memory>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <string>

namespace llama_periodic {
using namespace poseidon;
using namespace poseidon::gpu;
constexpr std::size_t period=256, compact_degree=512;
inline std::size_t rev(std::size_t x,unsigned bits) {
    std::size_t y=0;for(unsigned b=0;b<bits;++b){y=(y<<1)|(x&1);x>>=1;}return y;
}
inline std::uint64_t modpow(std::uint64_t a,std::size_t e,std::uint64_t q) {
    std::uint64_t r=1;for(;e;e>>=1,a=(a*a)%q)if(e&1)r=(r*a)%q;return r;
}
struct Encoded {
    GpuPlaintextMeta meta;
    std::vector<GpuWord> values; // [Q|P][512], natural exponent index.
    std::array<std::int64_t,compact_degree> coefficients{};
    std::shared_ptr<const void> encoder_identity; // GPU accepts only its encoder's coefficients.
};
// Immutable 512-point NTT tables for an optional integer GPU encoder.
struct GpuTablesSnapshot {
    std::size_t full_q=0;
    std::vector<GpuWord> moduli,roots,root_shoup,permutation;
    std::map<parms_id_type,std::pair<std::size_t,std::size_t>> levels;
    std::shared_ptr<const void> encoder_identity;
};
class Encoder {
    struct Limb {
        std::uint64_t modulus;
        std::unique_ptr<util::NTTTables> table;
        std::array<std::size_t,compact_degree> indices;
    };
    const PoseidonContext &context_;
    const std::shared_ptr<const void> identity_=std::make_shared<const int>(0);
    std::size_t degree_,full_q_;
    std::vector<Limb> limbs_;
    std::array<std::size_t,compact_degree> map_{};
    std::array<std::complex<double>,compact_degree> inverse_roots_{};
public:
    explicit Encoder(const PoseidonContext &context):context_(context) {
        degree_=context.parameters_literal()->degree();
        full_q_=context.crt_context()->key_context_data()->parms().q().size();
        if(degree_!=65536)throw std::invalid_argument("periodic adapter requires production N=65536");
        const auto *full=context.crt_context()->small_ntt_tables();
        auto moduli=context.parameters_literal()->q();
        const auto &p=context.parameters_literal()->p();moduli.insert(moduli.end(),p.begin(),p.end());
        if(full_q_+p.size()!=moduli.size())throw std::runtime_error("unexpected QP tables");
        for(std::size_t l=0;l<moduli.size();++l) {
            if(moduli[l].value()>std::numeric_limits<GpuWord>::max())throw std::invalid_argument("requires GPU 32-bit primes");
            Limb limb;limb.modulus=moduli[l].value();
            limb.table=std::make_unique<util::NTTTables>(9,moduli[l]);
            const auto target=modpow(full[l].get_root(),degree_/compact_degree,limb.modulus);
            const auto small=limb.table->get_root();
            std::size_t exponent=0;std::uint64_t power=1;
            for(std::size_t e=0;e<1024;++e,power=power*small%limb.modulus)if(power==target){exponent=e;break;}
            if(!(exponent&1))throw std::runtime_error("compact/full NTT roots are incompatible");
            for(std::size_t k=0;k<compact_degree;++k)
                limb.indices[k]=rev((((exponent*(2*k+1))%1024)-1)/2,9);
            limbs_.push_back(std::move(limb));
        }
        std::size_t pos=1;
        for(std::size_t i=0;i<period;++i,pos=pos*5%1024) {
            map_[i]=rev((pos-1)/2,9);map_[i+period]=rev((1024-pos-1)/2,9);
        }
        util::ComplexRoots roots(1024,MemoryManager::GetPool());
        for(std::size_t i=1;i<compact_degree;++i)
            inverse_roots_[i]=std::conj(roots.get_root(rev(i-1,9)+1));
    }
    // Exact host FFT inputs, exported for the resident-weight GPU encoder.
    const std::array<std::size_t,compact_degree> &slot_map() const noexcept{return map_;}
    const std::array<std::complex<double>,compact_degree> &inverse_roots() const noexcept{return inverse_roots_;}
    double coefficient_limit(parms_id_type id) const {
        const auto data=context_.crt_context()->get_context_data(id);
        if(!data)throw std::invalid_argument("periodic plaintext parameters");
        return std::ldexp(1.,std::min(62,static_cast<int>(data->total_coeff_modulus_bit_count())-2));
    }
    GpuTablesSnapshot gpu_tables_snapshot() const {
        GpuTablesSnapshot out;out.full_q=full_q_;out.encoder_identity=identity_;
        for(auto level=context_.crt_context()->key_context_data();level;level=level->next_context_data())
            out.levels.emplace(level->parms_id(),std::make_pair(level->qp_rns_tool()->base_q()->size(),level->qp_rns_tool()->base_p()->size()));
        for(const auto &limb:limbs_) {
            out.moduli.push_back(static_cast<GpuWord>(limb.modulus));
            const auto *roots=limb.table->get_from_root_powers();
            for(std::size_t j=0;j<compact_degree;++j) {
                out.roots.push_back(static_cast<GpuWord>(roots[j].operand));
                out.root_shoup.push_back(static_cast<GpuWord>(roots[j].quotient>>32));
                out.permutation.push_back(static_cast<GpuWord>(limb.indices[j]));
            }
        }
        return out;
    }
    // Identical CPU FFT/rounding/range checks; leave values empty for GPU NTT.
    Encoded encode_coefficients(const std::array<double,period> &values,parms_id_type id,double scale) const {
        const auto data=context_.crt_context()->get_context_data(id);
        if(!data||!std::isfinite(scale)||scale<=1)throw std::invalid_argument("periodic plaintext parameters");
        const auto nq=data->qp_rns_tool()->base_q()->size(),np=data->qp_rns_tool()->base_p()->size();
        Encoded result;result.meta={id,scale,true,degree_,nq,np};result.encoder_identity=identity_;
        std::array<std::complex<double>,compact_degree> work{};
        for(std::size_t j=0;j<period;++j) {
            if(!std::isfinite(values[j]))throw std::invalid_argument("nonfinite periodic weights");
            work[map_[j]]=work[map_[j+period]]=values[j];
        }
        util::Arithmetic<std::complex<double>,std::complex<double>,double> arithmetic;
        util::DWTHandler<std::complex<double>,std::complex<double>,double> fft(arithmetic);
        const double fix=scale/compact_degree;
        fft.transform_from_rev(work.data(),9,inverse_roots_.data(),&fix);
        const double limit=std::ldexp(1.,std::min(62,static_cast<int>(data->total_coeff_modulus_bit_count())-2));
        for(std::size_t i=0;i<compact_degree;++i) {
            const double value=std::round(work[i].real());
            if(!std::isfinite(value)||std::abs(value)>=limit)throw std::overflow_error("compact coefficient signed range");
            result.coefficients[i]=static_cast<std::int64_t>(value);
        }
        return result;
    }
    Encoded encode(const std::array<double,period> &values,parms_id_type id,double scale) const {
        auto result=encode_coefficients(values,id,scale);
        const auto nq=result.meta.q_count,np=result.meta.p_count;
        result.values.resize((nq+np)*compact_degree);
        // Real Llama diagonals encoded at a ~30-bit prime usually have all
        // signed coefficients strictly below every modulus. Avoid 512 costly
        // uint64 divisions per limb in that case, without changing rounding,
        // residues, or the general large-coefficient fallback.
        const char *raw=std::getenv("LLAMA_PERIODIC_REDUCED_FASTPATH");
        const bool fast=raw&&std::string(raw)=="1";
        std::uint64_t max_magnitude=0;
        if(fast)for(auto v:result.coefficients)max_magnitude=std::max(max_magnitude,static_cast<std::uint64_t>(v<0?-v:v));
        for(std::size_t l=0;l<nq+np;++l) {
            const auto &limb=limbs_.at(l<nq?l:full_q_+l-nq);
            std::array<std::uint64_t,compact_degree> residues{};
            if(fast&&max_magnitude<limb.modulus) {
                for(std::size_t j=0;j<compact_degree;++j) {
                    const auto v=result.coefficients[j];
                    const auto r=static_cast<std::uint64_t>(v<0?-v:v);
                    residues[j]=v<0&&r?limb.modulus-r:r;
                }
            } else for(std::size_t j=0;j<compact_degree;++j) {
                const auto v=result.coefficients[j];
                const auto r=static_cast<std::uint64_t>(v<0?-v:v)%limb.modulus;
                residues[j]=v<0&&r?limb.modulus-r:r;
            }
            util::ntt_negacyclic_harvey(residues.data(),*limb.table);
            for(std::size_t j=0;j<compact_degree;++j)
                result.values[l*compact_degree+j]=static_cast<GpuWord>(residues[limb.indices[j]]);
        }
        return result;
    }
};

// Caller retains each prepared matrix until its last GPU use. At most one
// tile/branch is resident in the initial streaming integration.
inline GpuLinearMatrixGroupQP upload_matrix(const std::map<int,Encoded> &diagonals,int device,unsigned drops=1) {
    if(diagonals.empty()||!drops)throw std::invalid_argument("empty periodic matrix");
    GpuLinearMatrixGroupQP group;group.set_step(drops);group.data().emplace_back();
    auto &matrix=group.data().back();const auto &first=diagonals.begin()->second.meta;
    matrix.log_slots=15;matrix.n1=16;matrix.level=static_cast<unsigned>(first.q_count-1);matrix.scale=first.scale;
    auto &plan=matrix.plan;plan.log_slots=15;plan.n1=16;plan.rescale_count=drops;plan.compressed_plaintexts=true;
    std::map<int,std::vector<std::pair<int,const Encoded*>>> groups;
    for(const auto &[d,encoded]:diagonals) {
        if(encoded.meta.parms_id!=first.parms_id||encoded.meta.scale!=first.scale||encoded.values.size()!=(first.q_count+first.p_count)*compact_degree)
            throw std::invalid_argument("periodic diagonal contract mismatch");
        const int normalized=(d%32768+32768)%32768;
        groups[(normalized/16)*16].push_back({normalized%16,&encoded});
    }
    std::map<int,unsigned> babies;
    for(const auto &[g,terms]:groups)for(const auto &[b,e]:terms)babies.emplace(b,0);
    for(auto &[b,id]:babies){id=static_cast<unsigned>(plan.baby_steps.size());plan.baby_steps.push_back(b);}
    std::vector<const GpuWord*> qptr,pptr;std::vector<std::uint32_t> periods,term_babies;
    plan.group_term_offsets.push_back(0);
    for(const auto &[g,terms]:groups) {
        const auto gi=static_cast<unsigned>(plan.giant_steps.size());plan.giant_steps.push_back(g);
        for(const auto &[b,encoded]:terms) {
            const int d=g+b;auto &plain=matrix.compressed_plain_vec_qp[d];plain.meta=encoded->meta;
            plain.period=compact_degree;plain.exact_device_reconstruction=true;
            plain.residues.allocate(encoded->values.size(),device);
            plain.residues.copy_from_host(encoded->values.data(),encoded->values.size());
            qptr.push_back(plain.residues.data());pptr.push_back(plain.residues.data()+first.q_count*compact_degree);
            periods.push_back(compact_degree);term_babies.push_back(babies.at(b));
            plan.terms.push_back({gi,babies.at(b),static_cast<unsigned>(plan.terms.size())});
        }
        plan.group_term_offsets.push_back(static_cast<unsigned>(plan.terms.size()));
    }
    plan.n2=static_cast<unsigned>(plan.giant_steps.size());
    auto upload=[device](auto &dest,const auto &src){dest.allocate(src.size(),device);dest.copy_from_host(src.data(),src.size());};
    upload(plan.diagonal_q_ptrs,qptr);upload(plan.diagonal_p_ptrs,pptr);upload(plan.diagonal_periods,periods);
    upload(plan.term_baby_indices,term_babies);upload(plan.group_term_offsets_device,plan.group_term_offsets);
    matrix.rot_index=plan.baby_steps;matrix.rot_index.insert(matrix.rot_index.end(),plan.giant_steps.begin(),plan.giant_steps.end());
    group.rot_index()=matrix.rot_index;return group;
}
// A plan view with an explicit owner for one packed upload. The existing
// group_view contract allows plan pointers whose storage is owned externally.
// Keep this object alive through the last GPU use of group.
struct PackedMatrix {
    DeviceVector<GpuWord> residues;
    GpuLinearMatrixGroupQP group;
};
// Shared layout construction for CPU and GPU encoding. ordered is the actual
// giant-group order, which differs from signed diagonal map order.
inline PackedMatrix allocate_matrix_packed(const std::map<int,Encoded> &diagonals,
        std::vector<const Encoded*> &ordered,int device,unsigned drops,bool require_values) {
    if(diagonals.empty()||!drops)throw std::invalid_argument("empty periodic matrix");
    ordered.clear();
    PackedMatrix result;auto &group=result.group;group.set_step(drops);group.data().emplace_back();
    auto &matrix=group.data().back();const auto &first=diagonals.begin()->second.meta;
    if(!first.q_count||first.degree!=65536||!first.is_ntt_form||!std::isfinite(first.scale)||first.scale<=1)
        throw std::invalid_argument("packed periodic metadata");
    matrix.group_view=true;matrix.log_slots=15;matrix.n1=16;matrix.level=static_cast<unsigned>(first.q_count-1);matrix.scale=first.scale;
    auto &plan=matrix.plan;plan.log_slots=15;plan.n1=16;plan.rescale_count=drops;plan.compressed_plaintexts=true;
    std::map<int,std::vector<std::pair<int,const Encoded*>>> groups;
    const auto stride=(first.q_count+first.p_count)*compact_degree;
    for(const auto &[d,encoded]:diagonals) {
        if(encoded.meta.parms_id!=first.parms_id||encoded.meta.scale!=first.scale||encoded.meta.q_count!=first.q_count||encoded.meta.p_count!=first.p_count||encoded.meta.degree!=first.degree||encoded.meta.is_ntt_form!=first.is_ntt_form||(require_values&&encoded.values.size()!=stride))
            throw std::invalid_argument("packed periodic diagonal contract mismatch");
        int normalized=(d%32768+32768)%32768;groups[(normalized/16)*16].push_back({normalized%16,&encoded});
    }
    std::map<int,unsigned> babies;
    for(const auto &[g,terms]:groups)for(const auto &[b,e]:terms)babies.emplace(b,0);
    for(auto &[b,id]:babies){id=static_cast<unsigned>(plan.baby_steps.size());plan.baby_steps.push_back(b);}
    result.residues.allocate(diagonals.size()*stride,device);
    ordered.reserve(diagonals.size());
    std::vector<const GpuWord*> qptr,pptr;std::vector<std::uint32_t> periods,term_babies;
    plan.group_term_offsets.push_back(0);
    for(const auto &[g,terms]:groups) {
        auto gi=static_cast<unsigned>(plan.giant_steps.size());plan.giant_steps.push_back(g);
        for(const auto &[b,encoded]:terms) {
            const auto offset=ordered.size()*stride;ordered.push_back(encoded);
            qptr.push_back(result.residues.data()+offset);pptr.push_back(result.residues.data()+offset+first.q_count*compact_degree);
            periods.push_back(compact_degree);term_babies.push_back(babies.at(b));
            plan.terms.push_back({gi,babies.at(b),static_cast<unsigned>(plan.terms.size())});
        }
        plan.group_term_offsets.push_back(static_cast<unsigned>(plan.terms.size()));
    }
    plan.n2=static_cast<unsigned>(plan.giant_steps.size());
    auto upload=[device](auto &dest,const auto &src){dest.allocate(src.size(),device);dest.copy_from_host(src.data(),src.size());};
    upload(plan.diagonal_q_ptrs,qptr);upload(plan.diagonal_p_ptrs,pptr);upload(plan.diagonal_periods,periods);
    upload(plan.term_baby_indices,term_babies);upload(plan.group_term_offsets_device,plan.group_term_offsets);
    matrix.rot_index=plan.baby_steps;matrix.rot_index.insert(matrix.rot_index.end(),plan.giant_steps.begin(),plan.giant_steps.end());
    group.rot_index()=matrix.rot_index;return result;
}
inline PackedMatrix upload_matrix_packed(const std::map<int,Encoded> &diagonals,int device,unsigned drops=1) {
    std::vector<const Encoded*> ordered;
    auto result=allocate_matrix_packed(diagonals,ordered,device,drops,true);
    const auto &meta=diagonals.begin()->second.meta;
    std::vector<GpuWord> packed;packed.reserve(diagonals.size()*(meta.q_count+meta.p_count)*compact_degree);
    for(const auto *encoded:ordered)packed.insert(packed.end(),encoded->values.begin(),encoded->values.end());
    result.residues.copy_from_host(packed.data(),packed.size());
    return result;
}
} // namespace llama_periodic
