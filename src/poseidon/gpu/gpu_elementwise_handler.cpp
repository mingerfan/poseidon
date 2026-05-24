#include "poseidon/gpu/gpu_elementwise_handler.h"
#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"

#include <stdexcept>
#include <algorithm>

namespace poseidon
{
namespace gpu
{
namespace
{

template <typename LeftShard, typename RightShard>
bool same_shard_placement(const LeftShard &left, const RightShard &right)
{
    return left.device_id == right.device_id &&
           left.limb_begin == right.limb_begin &&
           left.limb_count == right.limb_count &&
           left.coeff_begin == right.coeff_begin &&
           left.coeff_count == right.coeff_count;
}

const GpuParameterShard *find_parameter_shard(
    const GpuLevelInfo &level_info,
    const GpuPolyShardView &shard)
{
    for (const auto &candidate : level_info.shards)
    {
        const bool same_device = candidate.device_id == shard.device_id;
        const bool covers_limb =
            shard.limb_begin >= candidate.limb_begin &&
            shard.limb_begin + shard.limb_count <=
                candidate.limb_begin + candidate.limb_count;

        if (same_device && covers_limb)
        {
            return &candidate;
        }
    }

    return nullptr;
}

bool can_fuse_two_component_add_shards(
    const GpuPolyShardView &dst0,
    const GpuPolyShardView &dst1,
    const GpuConstPolyShardView &lhs0,
    const GpuConstPolyShardView &lhs1,
    const GpuConstPolyShardView &rhs0,
    const GpuConstPolyShardView &rhs1)
{
    return same_shard_placement(dst0, dst1) &&
           same_shard_placement(dst0, lhs0) &&
           same_shard_placement(dst0, lhs1) &&
           same_shard_placement(dst0, rhs0) &&
           same_shard_placement(dst0, rhs1);
}

}  // namespace

GpuElementwiseHandler::GpuElementwiseHandler(const GpuParameterData &params)
    : params_(params)
{}

// 最顶层的加法，支持不同component密文的求和，低位求和。
void GpuElementwiseHandler::add_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &left_view,
    const GpuConstCiphertextView &right_view,
    const GpuLevelInfo &level_info) const
{
    if (!(left_view.meta.parms_id == right_view.meta.parms_id) ||
        !(left_view.meta.parms_id == destination_view.meta.parms_id))
    {
        throw std::invalid_argument("add_ciphertext: parms_id mismatch");
    }

    if (left_view.meta.is_ntt_form != right_view.meta.is_ntt_form ||
        left_view.meta.is_ntt_form != destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument("add_ciphertext: NTT form mismatch");
    }

    if (left_view.meta.degree != right_view.meta.degree ||
        left_view.meta.degree != destination_view.meta.degree ||
        left_view.meta.q_count != right_view.meta.q_count ||
        left_view.meta.q_count != destination_view.meta.q_count ||
        left_view.meta.p_count != right_view.meta.p_count ||
        left_view.meta.p_count != destination_view.meta.p_count)
    {
        throw std::invalid_argument("add_ciphertext: shape mismatch");
    }

    const auto common_count = std::min(left_view.polys.size(), right_view.polys.size());
    const auto result_count = std::max(left_view.polys.size(), right_view.polys.size());

    if (destination_view.polys.size() != result_count)
    {
        throw std::invalid_argument("add_ciphertext: destination component count mismatch");
    }

    std::size_t processed_common_count = 0;
    if (common_count >= 2)
    {
        bool can_fuse_c0_c1 =
            destination_view.polys[0].shards.size() ==
                destination_view.polys[1].shards.size() &&
            destination_view.polys[0].shards.size() ==
                left_view.polys[0].shards.size() &&
            destination_view.polys[0].shards.size() ==
                left_view.polys[1].shards.size() &&
            destination_view.polys[0].shards.size() ==
                right_view.polys[0].shards.size() &&
            destination_view.polys[0].shards.size() ==
                right_view.polys[1].shards.size();

        if (can_fuse_c0_c1)
        {
            for (std::size_t shard_index = 0;
                 shard_index < destination_view.polys[0].shards.size();
                 ++shard_index)
            {
                const auto &dst0 = destination_view.polys[0].shards[shard_index];
                const auto &dst1 = destination_view.polys[1].shards[shard_index];
                const auto &lhs0 = left_view.polys[0].shards[shard_index];
                const auto &lhs1 = left_view.polys[1].shards[shard_index];
                const auto &rhs0 = right_view.polys[0].shards[shard_index];
                const auto &rhs1 = right_view.polys[1].shards[shard_index];

                if (!can_fuse_two_component_add_shards(
                        dst0,
                        dst1,
                        lhs0,
                        lhs1,
                        rhs0,
                        rhs1) ||
                    find_parameter_shard(level_info, dst0) == nullptr)
                {
                    can_fuse_c0_c1 = false;
                    break;
                }
            }
        }

        if (can_fuse_c0_c1)
        {
            for (std::size_t shard_index = 0;
                 shard_index < destination_view.polys[0].shards.size();
                 ++shard_index)
            {
                const auto &dst0 = destination_view.polys[0].shards[shard_index];
                const auto &dst1 = destination_view.polys[1].shards[shard_index];
                const auto &lhs0 = left_view.polys[0].shards[shard_index];
                const auto &lhs1 = left_view.polys[1].shards[shard_index];
                const auto &rhs0 = right_view.polys[0].shards[shard_index];
                const auto &rhs1 = right_view.polys[1].shards[shard_index];
                const auto *parameter_shard = find_parameter_shard(level_info, dst0);

                kernel::launch_add_two_poly_shards(
                    dst0,
                    dst1,
                    lhs0,
                    lhs1,
                    rhs0,
                    rhs1,
                    *parameter_shard,
                    level_info.degree);
            }
            processed_common_count = 2;
        }
    }

    for (std::size_t i = processed_common_count; i < common_count; ++i)
    {
        add_poly(destination_view.polys[i], left_view.polys[i], right_view.polys[i], level_info);
    }

    if (left_view.polys.size() > right_view.polys.size())
    {
        for (std::size_t i = common_count; i < left_view.polys.size(); ++i)
        {
            copy_poly(destination_view.polys[i], left_view.polys[i], level_info);
        }
    }
    else
    {
        for (std::size_t i = common_count; i < right_view.polys.size(); ++i)
        {
            copy_poly(destination_view.polys[i], right_view.polys[i], level_info);
        }
    }
}

void GpuElementwiseHandler::sub_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &left_view,
    const GpuConstCiphertextView &right_view,
    const GpuLevelInfo &level_info) const
{
    if (!(left_view.meta.parms_id == right_view.meta.parms_id) ||
        !(left_view.meta.parms_id == destination_view.meta.parms_id))
    {
        throw std::invalid_argument("sub_ciphertext: parms_id mismatch");
    }

    if (left_view.meta.is_ntt_form != right_view.meta.is_ntt_form ||
        left_view.meta.is_ntt_form != destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument("sub_ciphertext: NTT form mismatch");
    }

    if (left_view.meta.degree != right_view.meta.degree ||
        left_view.meta.degree != destination_view.meta.degree ||
        left_view.meta.q_count != right_view.meta.q_count ||
        left_view.meta.q_count != destination_view.meta.q_count ||
        left_view.meta.p_count != right_view.meta.p_count ||
        left_view.meta.p_count != destination_view.meta.p_count)
    {
        throw std::invalid_argument("sub_ciphertext: shape mismatch");
    }

    const auto common_count = std::min(left_view.polys.size(), right_view.polys.size());
    const auto result_count = std::max(left_view.polys.size(), right_view.polys.size());

    if (destination_view.polys.size() != result_count)
    {
        throw std::invalid_argument("sub_ciphertext: destination component count mismatch");
    }

    for (std::size_t i = 0; i < common_count; ++i)
    {
        sub_poly(
            destination_view.polys[i],
            left_view.polys[i],
            right_view.polys[i],
            level_info);
    }

    if (left_view.polys.size() > right_view.polys.size())
    {
        for (std::size_t i = common_count; i < left_view.polys.size(); ++i)
        {
            copy_poly(
                destination_view.polys[i],
                left_view.polys[i],
                level_info);
        }
    }
    else
    {
        for (std::size_t i = common_count; i < right_view.polys.size(); ++i)
        {
            negate_poly(
                destination_view.polys[i],
                right_view.polys[i],
                level_info);
        }
    }
}

void GpuElementwiseHandler::negate_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &level_info) const
{
    if (!(source_view.meta.parms_id == destination_view.meta.parms_id))
    {
        throw std::invalid_argument("negate_ciphertext: parms_id mismatch");
    }

    if (source_view.meta.is_ntt_form != destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument("negate_ciphertext: NTT form mismatch");
    }

    if (source_view.meta.degree != destination_view.meta.degree ||
        source_view.meta.q_count != destination_view.meta.q_count ||
        source_view.meta.p_count != destination_view.meta.p_count)
    {
        throw std::invalid_argument("negate_ciphertext: shape mismatch");
    }

    const auto common_count = source_view.polys.size();

    if (destination_view.polys.size() != source_view.polys.size())
    {
        throw std::invalid_argument("negate_ciphertext: component count mismatch");
    }

    for (std::size_t i = 0; i < common_count; ++i)
    {
        negate_poly(
            destination_view.polys[i],
            source_view.polys[i],
            level_info);
    }
}

void GpuElementwiseHandler::add_plain_to_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &ciphertext_view,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // CKKS add_plain:
    // - destination.c0 = ciphertext.c0 + plaintext.poly;
    // - destination.ci = ciphertext.ci for i > 0.
    //
    // c0 computation should eventually call add_poly().
    // other components should eventually call copy_poly().

    (void)destination_view;
    (void)ciphertext_view;
    (void)plaintext_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::add_plain_to_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::sub_plain_from_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &ciphertext_view,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // CKKS sub_plain:
    // - destination.c0 = ciphertext.c0 - plaintext.poly;
    // - destination.ci = ciphertext.ci for i > 0.
    //
    // c0 computation should eventually call sub_poly().
    // other components should eventually call copy_poly().

    (void)destination_view;
    (void)ciphertext_view;
    (void)plaintext_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::sub_plain_from_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::multiply_plain_with_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &ciphertext_view,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // For every ciphertext component:
    // - destination.ci = ciphertext.ci * plaintext.poly.
    //
    // Each component computation should eventually call multiply_plain_poly().
    // multiply_plain_poly() should call kernel::launch_dyadic_product_poly_shard(...).

    (void)destination_view;
    (void)ciphertext_view;
    (void)plaintext_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::multiply_plain_with_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::multiply_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &left_view,
    const GpuConstCiphertextView &right_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // Implement Poseidon component-vector convolution:
    //
    // for each left component index:
    //   for each right component index:
    //     destination component at index sum receives product accumulation.
    //
    // Each product accumulation should eventually call multiply_accumulate_poly().
    // multiply_accumulate_poly() should call
    // kernel::launch_multiply_accumulate_poly_shard(...).
    //
    // This corresponds to Cheddar-style Tensor, but must use polys[index].

    (void)destination_view;
    (void)left_view;
    (void)right_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::multiply_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::square_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // Implement optimized square using component-vector symmetry.
    //
    // Internally this should reuse multiply_accumulate_poly() where possible.

    (void)destination_view;
    (void)source_view;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::square_ciphertext is not implemented yet");
}

void GpuElementwiseHandler::add_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &left_poly,
    const GpuConstRNSPolyView &right_poly,
    const GpuLevelInfo &level_info) const
{
    if (destination_poly.shards.size() != left_poly.shards.size() ||
        destination_poly.shards.size() != right_poly.shards.size())
    {
        throw std::invalid_argument("add_poly: shard count mismatch");
    }

    for (std::size_t i = 0; i < destination_poly.shards.size(); ++i)
    {
        const auto &dst = destination_poly.shards[i];
        const auto &lhs = left_poly.shards[i];
        const auto &rhs = right_poly.shards[i];

        if (dst.device_id != lhs.device_id ||
            dst.device_id != rhs.device_id ||
            dst.limb_begin != lhs.limb_begin ||
            dst.limb_begin != rhs.limb_begin ||
            dst.limb_count != lhs.limb_count ||
            dst.limb_count != rhs.limb_count ||
            dst.coeff_begin != lhs.coeff_begin ||
            dst.coeff_begin != rhs.coeff_begin ||
            dst.coeff_count != lhs.coeff_count ||
            dst.coeff_count != rhs.coeff_count)
        {
            throw std::invalid_argument("add_poly: shard placement mismatch");
        }

        const GpuParameterShard *parameter_shard =
            find_parameter_shard(level_info, dst);

        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("add_poly: no matching parameter shard");
        }

        kernel::launch_add_poly_shard(dst, lhs, rhs, *parameter_shard, level_info.degree);
    }
}

void GpuElementwiseHandler::sub_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &left_poly,
    const GpuConstRNSPolyView &right_poly,
    const GpuLevelInfo &level_info) const
{
    if (destination_poly.shards.size() != left_poly.shards.size() ||
        destination_poly.shards.size() != right_poly.shards.size())
    {
        throw std::invalid_argument("sub_poly: shard count mismatch");
    }

    for (std::size_t i = 0; i < destination_poly.shards.size(); ++i)
    {
        const auto &dst = destination_poly.shards[i];
        const auto &lhs = left_poly.shards[i];
        const auto &rhs = right_poly.shards[i];

        if (dst.device_id != lhs.device_id ||
            dst.device_id != rhs.device_id ||
            dst.limb_begin != lhs.limb_begin ||
            dst.limb_begin != rhs.limb_begin ||
            dst.limb_count != lhs.limb_count ||
            dst.limb_count != rhs.limb_count ||
            dst.coeff_begin != lhs.coeff_begin ||
            dst.coeff_begin != rhs.coeff_begin ||
            dst.coeff_count != lhs.coeff_count ||
            dst.coeff_count != rhs.coeff_count)
        {
            throw std::invalid_argument("sub_poly: shard placement mismatch");
        }

        const GpuParameterShard *parameter_shard =
            find_parameter_shard(level_info, dst);

        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("sub_poly: no matching parameter shard");
        }

        kernel::launch_sub_poly_shard(dst, lhs, rhs, *parameter_shard, level_info.degree);
    }
}

void GpuElementwiseHandler::negate_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &source_poly,
    const GpuLevelInfo &level_info) const
{
    if (destination_poly.shards.size() != source_poly.shards.size())
    {
        throw std::invalid_argument("negate_poly: shard count mismatch");
    }

    for (std::size_t i = 0; i < destination_poly.shards.size(); ++i)
    {
        const auto &dst = destination_poly.shards[i];
        const auto &src = source_poly.shards[i];

        if (dst.device_id != src.device_id ||
            dst.limb_begin != src.limb_begin ||
            dst.limb_count != src.limb_count ||
            dst.coeff_begin != src.coeff_begin ||
            dst.coeff_count != src.coeff_count)
        {
            throw std::invalid_argument("negate_poly: shard placement mismatch");
        }

        const GpuParameterShard *parameter_shard =
            find_parameter_shard(level_info, dst);

        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("negate_poly: no matching parameter shard");
        }

        kernel::launch_negate_poly_shard(dst, src, *parameter_shard, level_info.degree);
    }
}

void GpuElementwiseHandler::copy_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &source_poly,
    const GpuLevelInfo &level_info) const
{
    if (destination_poly.shards.size() != source_poly.shards.size())
    {
        throw std::invalid_argument("copy_poly: shard count mismatch");
    }

    for (std::size_t i = 0; i < destination_poly.shards.size(); ++i)
    {
        const auto &dst = destination_poly.shards[i];
        const auto &src = source_poly.shards[i];

        if (dst.device_id != src.device_id ||
            dst.limb_begin != src.limb_begin ||
            dst.limb_count != src.limb_count ||
            dst.coeff_begin != src.coeff_begin ||
            dst.coeff_count != src.coeff_count)
        {
            throw std::invalid_argument("copy_poly: shard placement mismatch");
        }

        (void)level_info;
        kernel::launch_copy_poly_shard(dst, src, level_info.degree);
    }
}

void GpuElementwiseHandler::multiply_plain_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &ciphertext_poly,
    const GpuConstRNSPolyView &plaintext_poly,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // 1. Validate placement.
    // 2. Find matching GpuParameterShard.
    // 3. Call kernel::launch_dyadic_product_poly_shard(...).

    (void)destination_poly;
    (void)ciphertext_poly;
    (void)plaintext_poly;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::multiply_plain_poly is not implemented yet");
}

void GpuElementwiseHandler::multiply_accumulate_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &left_poly,
    const GpuConstRNSPolyView &right_poly,
    const GpuLevelInfo &level_info) const
{
    // TODO:
    // 1. Validate shard placement.
    // 2. Find matching GpuParameterShard.
    // 3. Call kernel::launch_multiply_accumulate_poly_shard(...).

    (void)destination_poly;
    (void)left_poly;
    (void)right_poly;
    (void)level_info;

    throw std::runtime_error("GpuElementwiseHandler::multiply_accumulate_poly is not implemented yet");
}

}  // namespace gpu
}  // namespace poseidon
