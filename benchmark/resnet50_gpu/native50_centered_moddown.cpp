// Diagnostic executable override only. Reuse the existing centered base converter.
// Production CPU/GPU libraries are left untouched until this hypothesis is tested.
#include "poseidon/util/rns_tool_qp.h"
#include <iostream>
namespace poseidon::util {
void RNSToolQP::mod_down(ConstRNSIter input_q, ConstRNSIter input_p,
                        RNSIter destination, MemoryPoolHandle pool) const
{
    static bool reported=false;
    if (!reported) {
        std::cout << "CENTERED_MODDOWN_CPU_DIAGNOSTIC_ACTIVE" << std::endl;
        reported=true;
    }
    const auto nq=base_q_->size();
    POSEIDON_ALLOCATE_GET_RNS_ITER(tmp,coeff_count_,nq,pool);
    for (std::size_t j=0;j<nq;++j) {
        RNSBase single(std::vector<Modulus>{base_q_->base()[j]},pool);
        BaseConverter centered(*base_p_,single,pool);
        centered.exact_convert_array(input_p,tmp[j],pool);
    }
    for (std::size_t j=0;j<nq;++j) {
        sub_poly_coeffmod(input_q[j],tmp[j],coeff_count_,base_q_->base()[j],destination[j]);
        multiply_poly_scalar_coeffmod(destination[j],coeff_count_,
            prod_p_inv_mod_qi_[j].operand,base_q_->base()[j],destination[j]);
    }
}
}
