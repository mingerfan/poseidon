#pragma once
// Candidate packing primitive only, NOT a paired Bootstrap implementation.
// Native Cerium prephases its second input by i; its wrapper merely adds.
// Poseidon's current manual real-only Boot must not silently consume this as
// a two-output refresh. No model path enables this helper by default.
#include "source_cerium_layout.hpp"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cerium_source::packing {
enum class InputConvention {BothLogicalReal,SecondAlreadyImaginary};

template<class Cipher> class PackedRealPair {
    Cipher carrier_;
    InputConvention convention_;
public:
    PackedRealPair(Cipher &&carrier,InputConvention convention)
        :carrier_(std::move(carrier)),convention_(convention) {}
    const Cipher &carrier() const & noexcept {return carrier_;}
    const Cipher &carrier() const && = delete; // never borrow from a temporary owner
    InputConvention convention() const noexcept {return convention_;}
    static constexpr unsigned logical_real_channels=2;
    // Deliberately no implicit conversion to an ordinary real ciphertext.
};

template<class Cipher> void require_single_device_real_pair_operand(const Cipher &x,int expected_device=-1) {
    const auto &m=x.meta;
    const int dev=expected_device>=0?expected_device:(x.fields_.empty()?0:x.fields_[0].device_id);
    if(m.degree!=layout::ring_degree || !m.q_count || m.p_count || m.component_count!=2 ||
       !m.is_ntt_form || m.correction_factor!=1 || !std::isfinite(m.scale) || m.scale<=0 ||
       x.empty() || x.fields_.size()!=1 || x.fields_[0].empty() || x.fields_[0].device_id<0 || x.fields_[0].device_id!=dev)
        throw std::invalid_argument("paired packing requires single-device N65536 Q-only NTT two-component ciphertexts");
}
template<class Cipher> void require_same_pair_state(const Cipher &a,const Cipher &b) {
    require_single_device_real_pair_operand(a);require_single_device_real_pair_operand(b);
    if(a.meta.parms_id!=b.meta.parms_id || a.meta.q_count!=b.meta.q_count ||
       a.meta.scale!=b.meta.scale || a.fields_[0].device_id!=b.fields_[0].device_id)
        throw std::invalid_argument("paired packing requires identical Q/id/actual scale/device; no implicit ModDrop or scale relabel");
}
// Caller contract: both values belong to this Ops' Context/key/device and identical
// logical slot layout. Ciphertext metadata cannot prove key identity or real
// semantics. InputConvention is an explicit declaration, not a decrypted test.
// BothLogicalReal: a+i*b. SecondAlreadyImaginary: a+ib, with no second i.
// With real-input imaginary residuals, those residuals cross into the partner;
// no perfect-real or no-cross-talk precision guarantee is made here.
template<class Ops,class Cipher>
PackedRealPair<Cipher> merge_for_pair_boot(Ops &ops,const Cipher &a,const Cipher &b,
                                         InputConvention convention) {
    if(convention!=InputConvention::BothLogicalReal && convention!=InputConvention::SecondAlreadyImaginary)
        throw std::invalid_argument("explicit paired input convention required");
    require_same_pair_state(a,b);
    auto merge=[&](const Cipher &second) {
        require_same_pair_state(a,second);
        auto out=ops.add(a,second);require_same_pair_state(a,out);return out;
    };
    if(convention==InputConvention::SecondAlreadyImaginary)
        return PackedRealPair<Cipher>(merge(b),convention);
    static const std::vector<std::complex<double>> plus_i(layout::slots,{0.,1.});
    // Scale1 exact+i; existing Api caches the encoding at this input Q/id.
    // Never borrow an EvalMod +/-i PT belonging to a different Q level.
    auto imaginary=ops.cp_complex(b,plus_i,1.);
    return PackedRealPair<Cipher>(merge(imaginary),convention);
}
} // namespace cerium_source::packing
