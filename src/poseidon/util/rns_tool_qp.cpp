#include "rns_tool_qp.h"
#include <algorithm>

namespace poseidon
{
namespace util
{
RNSToolQP::RNSToolQP(std::size_t poly_modulus_degree, const RNSBase &coeff_modulus_q,
                     const RNSBase &coeff_modulus_p, const Modulus &plain_modulus,
                     MemoryPoolHandle pool)
    : pool_(std::move(pool))
{
    initialize(poly_modulus_degree, coeff_modulus_q, coeff_modulus_p, plain_modulus);
}

void RNSToolQP::initialize(std::size_t poly_modulus_degree, const RNSBase &q, const RNSBase &p,
                           const Modulus &t)
{

    coeff_count_ = poly_modulus_degree;
    base_q_ = allocate<RNSBase>(pool_, q, pool_);
    base_p_ = allocate<RNSBase>(pool_, p, pool_);

    auto base_qp = base_q_->extend(p);
    base_qp_ = allocate<RNSBase>(pool_, base_qp, pool_);

    size_t base_q_size = base_q_->size();
    size_t base_p_size = base_p_->size();

    auto decomp_count = ceil(static_cast<double>(base_q_size) / static_cast<double>(base_p_size));
    for (size_t i = 0; i < decomp_count; ++i)
    {
        rnstool_decomp_.emplace_back(*base_q_.get(), *base_p_.get(), i);
    }

    base_q_to_p_conv_ = allocate<BaseConverter>(pool_, *base_q_, *base_p_, pool_);
    base_p_to_q_conv_ = allocate<BaseConverter>(pool_, *base_p_, *base_q_, pool_);

    prod_p_mod_qi_ = allocate<MultiplyUIntModOperand>(base_q_size, pool_);
    POSEIDON_ITERATE(iter(prod_p_mod_qi_, base_q_->base()), base_q_size,
                     [&](auto I) {
                         get<0>(I).operand =
                             modulo_uint(base_p_->base_prod(), base_p_size, get<1>(I));
                     });

    prod_p_inv_mod_qi_ = allocate<MultiplyUIntModOperand>(base_q_size, pool_);
    POSEIDON_ITERATE(iter(prod_p_inv_mod_qi_, base_q_->base()), base_q_size,
                     [&](auto I)
                     {
                         get<0>(I).operand =
                             modulo_uint(base_p_->base_prod(), base_p_size, get<1>(I));
                         if (!try_invert_uint_mod(get<0>(I).operand, get<1>(I), get<0>(I).operand))
                         {
                             throw std::logic_error("invalid rns bases");
                         }
                     });
}

void RNSToolQP::mod_up(poseidon::util::ConstRNSIter input, poseidon::util::RNSIter destination,
                       poseidon::MemoryPoolHandle pool) const
{

    /*
    Require: Input in base q
    Ensure: Output in base p
    */

    // Convert q -> p
    base_q_to_p_conv_->fast_convert_array(input, destination, pool);
}
void RNSToolQP::mod_up_copy_q(ConstRNSIter input, size_t base_idx, RNSIter destination,
                              MemoryPoolHandle pool) const
{
    auto &rns_tool_inter = rnstool_decomp_[base_idx];
    auto base_start_index = rns_tool_inter.base_start_idx();
    auto base_last_index = rns_tool_inter.base_last_idx();
    auto coeff_count = input.poly_modulus_degree();
    ConstRNSIter base_input_iter = input;
    RNSIter base_dest_iter = destination;

    for (auto i = base_start_index; i < base_last_index; ++i)
    {
        set_uint(*(base_input_iter + i), coeff_count, *(base_dest_iter + i));
    }
}

void RNSToolQP::mod_up_base_q(RNSIter input, size_t base_idx, MemoryPoolHandle pool) const
{
    auto &rns_tool_inter = rnstool_decomp_[base_idx];
    auto base_start_index = rns_tool_inter.base_start_idx();
    auto base_last_index = rns_tool_inter.base_last_idx();
    auto base_q_size = base_q_->size();
    auto &conv_map = rns_tool_inter.single_obase_q_conv_map();

    if (base_last_index - base_start_index == 1)
    {
        POSEIDON_THROW(invalid_argument_error, "mod_up_base_q : only one base");
    }

    ConstRNSIter base_iter = input + base_start_index;
    for (auto i = 0; i < base_start_index; ++i)
    {
        auto &conv = conv_map.at(i);
        RNSIter iter_tmp = input + i;
        conv->fast_convert_array(base_iter, iter_tmp, pool);
    }

    for (auto i = base_last_index; i < base_q_size; ++i)
    {
        auto &conv = conv_map.at(i);
        RNSIter iter_tmp = input + i;
        conv->fast_convert_array(base_iter, iter_tmp, pool);
    }
}

void RNSToolQP::mod_up_base_p(ConstRNSIter input, size_t base_idx, RNSIter destination,
                              MemoryPoolHandle pool) const
{
    auto &rns_tool_inter = rnstool_decomp_[base_idx];
    auto base_start_index = rns_tool_inter.base_start_idx();
    auto base_last_index = rns_tool_inter.base_last_idx();
    auto &conv = rns_tool_inter.obase_p_conv();

    if (base_last_index - base_start_index == 1)
    {
        POSEIDON_THROW(invalid_argument_error, "mod_up_base_q : only one base");
    }

    ConstRNSIter base_iter = input + base_start_index;
    conv->fast_convert_array(base_iter, destination, pool);
}

void RNSToolQP::mod_up_from_one_base_q(RNSIter input, size_t base_idx, MemoryPoolHandle pool) const
{
    auto &rns_tool_inter = rnstool_decomp_[base_idx];
    auto base_start_index = rns_tool_inter.base_start_idx();
    auto base_last_index = rns_tool_inter.base_last_idx();
    auto coeff_count = this->coeff_count_;
    auto modulus_q = this->base_q_->base();

    if (base_last_index - base_start_index != 1)
    {
        POSEIDON_THROW(invalid_argument_error, "mod_up_from_one_base_q : not one base");
    }

    ConstRNSIter base_iter = input + base_start_index;
    RNSIter iter_tmp = input;

    for (std::size_t i = 0; i < this->base_q_->size(); ++i)
    {
        if (i == base_start_index) continue;
        if (modulus_q[i] >= modulus_q[base_last_index - 1])
        {
            set_uint(*(base_iter), coeff_count, *(iter_tmp + i));
        }
        else
        {
            modulo_poly_coeffs(*(base_iter), coeff_count, modulus_q[i], *(iter_tmp + i));
        }
    }
}

void RNSToolQP::mod_up_from_one_base_p(ConstRNSIter input, size_t base_idx, RNSIter destination,
                                       MemoryPoolHandle pool) const
{
    auto &rns_tool_inter = rnstool_decomp_[base_idx];
    auto base_start_index = rns_tool_inter.base_start_idx();
    auto base_last_index = rns_tool_inter.base_last_idx();
    auto coeff_count = this->coeff_count_;
    auto modulus_q = this->base_q_->base();
    auto modulus_p = this->base_p_->base();
    auto base_p_size = this->base_p_->size();

    if (base_last_index - base_start_index != 1)
    {
        POSEIDON_THROW(invalid_argument_error, "mod_up_from_one_base_p : not one base");
    }

    ConstRNSIter base_input_iter = input + base_start_index;
    RNSIter base_dest_iter = destination;

    for (auto i = 0; i < base_p_size; ++i)
    {
        if (modulus_p[i] >= modulus_q[base_last_index - 1])
        {
            set_uint(*(base_input_iter), coeff_count, *(base_dest_iter + i));
        }
        else
        {
            modulo_poly_coeffs(*(base_input_iter), coeff_count, modulus_p[i],
                               *(base_dest_iter + i));
        }
    }
}

void RNSToolQP::mod_down(ConstRNSIter input_q, ConstRNSIter input_p,
                         poseidon::util::RNSIter destination, poseidon::MemoryPoolHandle pool) const
{

    /*
    Require: Input in base p
    Ensure: Output in base q
    */

    size_t base_q_size = base_q_->size();

    // Convert p -> q
    POSEIDON_ALLOCATE_GET_RNS_ITER(tmp, coeff_count_, base_q_size, pool);

    base_p_to_q_conv_->fast_convert_array(input_p, tmp, pool);
    for (auto i = 0; i < base_q_size; ++i)
    {
        sub_poly_coeffmod(input_q[i], tmp[i], coeff_count_, base_q_->base()[i], destination[i]);
    }

    POSEIDON_ITERATE(iter(destination, base_q_->base(), prod_p_inv_mod_qi_), base_q_size,
                     [&](auto I)
                     {
                         multiply_poly_scalar_coeffmod(get<0>(I), coeff_count_, get<2>(I).operand,
                                                       get<1>(I), get<0>(I));
                     });
}

void RNSToolQP::mod_down_centered(ConstRNSIter input_q, ConstRNSIter input_p,
                                  RNSIter destination, MemoryPoolHandle pool) const
{
    const auto p_size=base_p_->size();
    const auto q_size=base_q_->size();
    auto composed=allocate_uint(coeff_count_*p_size,pool);
    for(std::size_t i=0;i<p_size;++i)
        std::copy_n(&input_p[i][0],coeff_count_,composed.get()+i*coeff_count_);
    // compose_array changes RNS-major storage to coefficient-major integers in [0,P).
    base_p_->compose_array(composed.get(),coeff_count_,pool);
    auto half=allocate_uint(p_size,pool);
    right_shift_uint(base_p_->base_prod(),1,p_size,half.get());
    auto negative=allocate_uint(coeff_count_,pool);
    for(std::size_t j=0;j<coeff_count_;++j)
        negative[j]=is_greater_than_uint(composed.get()+j*p_size,half.get(),p_size);
    for(std::size_t i=0;i<q_size;++i)
    {
        const auto &qi=base_q_->base()[i];
        for(std::size_t j=0;j<coeff_count_;++j)
        {
            auto residue=modulo_uint(composed.get()+j*p_size,p_size,qi);
            if(negative[j]) residue=sub_uint_mod(residue,prod_p_mod_qi_[i].operand,qi);
            const auto difference=sub_uint_mod(input_q[i][j],residue,qi);
            destination[i][j]=multiply_uint_mod(difference,prod_p_inv_mod_qi_[i].operand,qi);
        }
    }
}

void RNSToolQP::RNSToolInter::initialize(const RNSBase &q, const RNSBase &p, size_t decomp_idx)
{
    size_t base_q_size = q.size();
    size_t base_p_size = p.size();

    base_start_idx_ = decomp_idx * base_p_size;
    base_end_idx_ = (decomp_idx + 1) * base_p_size;
    base_last_idx_ = base_q_size < base_end_idx_ ? base_q_size : base_end_idx_;
    if ((base_q_size < base_end_idx_) && (base_q_size % base_p_size == 1))
    {
        // only need copy data,don't need basic conv
    }
    else
    {
        auto base_rns_num = base_last_idx_ - base_start_idx_;
        std::vector<Modulus> coeff_modulus_tmp(base_rns_num);
        for (size_t j = 0; j < base_rns_num; ++j)
        {
            coeff_modulus_tmp[j] = q.base()[base_start_idx_ + j].value();
        }
        base_qi_ = allocate<RNSBase>(pool_, coeff_modulus_tmp, pool_);

        for (auto i = 0; i < base_start_idx_; ++i)
        {
            auto &modulus = q.base()[i];
            auto base_tmp = allocate<RNSBase>(pool_, vector<Modulus>{modulus}, pool_);
            single_obase_q_conv_map_.emplace(
                i, allocate<BaseConverter>(pool_, *base_qi_, *base_tmp, pool_));
        }

        for (auto i = base_last_idx_; i < base_q_size; ++i)
        {
            auto &modulus = q.base()[i];
            auto base_tmp = allocate<RNSBase>(pool_, vector<Modulus>{modulus}, pool_);
            single_obase_q_conv_map_.emplace(
                i, allocate<BaseConverter>(pool_, *base_qi_, *base_tmp, pool_));
        }
        obase_p_conv_ = allocate<BaseConverter>(pool_, *base_qi_, p, pool_);
    }
}

}  // namespace util
}  // namespace poseidon
