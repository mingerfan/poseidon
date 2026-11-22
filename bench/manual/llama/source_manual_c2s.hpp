#pragma once
// Manual model's physical C2S rebuild, promoted without arithmetic changes.
// A new executable still requires independent encrypted accuracy validation.
// No backend/header ABI/kernel/default change and no metadata-only accuracy fix.
#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/advance/homomorphic_mod.h"
#include "poseidon/gpu/gpu_uploader.h"
#include <utility>
namespace cerium_source::manual_backend {
struct ManualC2SPrecision {
 poseidon::gpu::GpuLinearMatrixGroupQP matrices;
 // Nonowning group pointers must die before their owning matrices.
 std::vector<std::vector<poseidon::gpu::GpuMatrixPlainQP>> groups;
 double raised_scale;
 const unsigned configured_raise_log;
 ManualC2SPrecision(poseidon::PoseidonContext&context,poseidon::CKKSEncoder&encoder,const poseidon::gpu::GpuGaloisKeysData&keys,unsigned raise_log=50):raised_scale(std::exp2(double(raise_log))),configured_raise_log(raise_log){
  using namespace poseidon;using namespace poseidon::gpu;
  if(raise_log!=50&&raise_log!=52&&raise_log!=54)throw std::runtime_error("selected C2S raise log must be50/52/54");if(context.parameters_literal()->q().size()!=49)throw std::runtime_error("selected C2S precision requires Q49");EvalModPoly original(context,CosDiscrete,std::exp2(45.),0,5,3,12,0,22);const double scaling=original.q_div()/(original.k()*original.sc_fac()*original.q_diff())*(raised_scale/std::exp2(45.));if(std::abs(12*context.crt_context()->q0()*scaling/raised_scale-1)>1e-12)throw std::runtime_error("selected C2S integer-period pairing failed");HomomorphicDFTMatrixLiteral literal(encode,16,15,48,{1,1,1},true,scaling,false,1,{},0);LinearMatrixGroup cpu;literal.create_dynamic(cpu,encoder,raised_scale,std::exp2(50.),std::exp2(50.),1.);const std::vector<std::uint32_t> expected{1,2,1};if(cpu.data().size()!=3||cpu.rescale_counts()!=expected)throw std::runtime_error("selected C2S precision unexpected drop schedule");const std::vector<std::size_t> required_q{49,48,46,45};for(std::size_t i=0;i<3;++i)if(std::size_t(cpu.data()[i].level)+1!=required_q[i])throw std::runtime_error("selected C2S physical stage level mismatch");GpuUploader::prepare_key_views_for_q_counts(keys,required_q);matrices=GpuUploader::upload_linear_matrix_group_qp(cpu,context,0,1,true);for(const auto&m:matrices.data())groups.push_back(GpuUploader::upload_double_hoist_group_views(m,0));if(matrices.rescale_counts()!=expected||groups.size()!=3)throw std::runtime_error("selected C2S uploaded plan mismatch");
 }
 void swap_into(poseidon::gpu::GpuBootstrapData&data){using std::swap;swap(matrices,data.coeff_to_slot_matrix_qp);swap(groups,data.coeff_to_slot_groups_qp);swap(raised_scale,data.raised_scale_override);}
};
} // namespace cerium_source::manual_backend
