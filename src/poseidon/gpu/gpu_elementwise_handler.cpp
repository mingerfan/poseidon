#include "poseidon/gpu/gpu_elementwise_handler.h"
#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace poseidon
{
namespace gpu
{
namespace
{

constexpr const char *kFusedCiphertextMultiplyEnv =
    "POSEIDON_ELEMENTWISE_FUSED_CT_MUL";

bool use_fused_ciphertext_multiply()
{
    const char *raw = std::getenv(kFusedCiphertextMultiplyEnv);
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

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

void GpuElementwiseHandler::multiply_scalar_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    GpuWord scalar,
    const GpuLevelInfo &level_info) const
{
    if (!(source_view.meta.parms_id == destination_view.meta.parms_id) ||
        !(source_view.meta.parms_id == level_info.parms_id))
    {
        throw std::invalid_argument("multiply_scalar_ciphertext: parms_id mismatch");
    }
    if (source_view.meta.is_ntt_form != destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument("multiply_scalar_ciphertext: NTT form mismatch");
    }
    if (source_view.meta.degree != destination_view.meta.degree ||
        source_view.meta.degree != level_info.degree ||
        source_view.meta.q_count != destination_view.meta.q_count ||
        source_view.meta.q_count != level_info.q_count ||
        source_view.meta.p_count != destination_view.meta.p_count ||
        source_view.meta.p_count != level_info.p_count)
    {
        throw std::invalid_argument("multiply_scalar_ciphertext: shape mismatch");
    }
    if (destination_view.polys.size() != source_view.polys.size())
    {
        throw std::invalid_argument("multiply_scalar_ciphertext: component count mismatch");
    }

    for (std::size_t i = 0; i < source_view.polys.size(); ++i)
    {
        const auto &dst_poly = destination_view.polys[i];
        const auto &src_poly = source_view.polys[i];
        if (dst_poly.shards.size() != src_poly.shards.size())
        {
            throw std::invalid_argument("multiply_scalar_ciphertext: shard count mismatch");
        }

        for (std::size_t shard_index = 0; shard_index < dst_poly.shards.size(); ++shard_index)
        {
            const auto &dst = dst_poly.shards[shard_index];
            const auto &src = src_poly.shards[shard_index];
            if (dst.device_id != src.device_id ||
                dst.limb_begin != src.limb_begin ||
                dst.limb_count != src.limb_count ||
                dst.coeff_begin != src.coeff_begin ||
                dst.coeff_count != src.coeff_count)
            {
                throw std::invalid_argument("multiply_scalar_ciphertext: shard placement mismatch");
            }

            const GpuParameterShard *parameter_shard =
                find_parameter_shard(level_info, dst);
            if (parameter_shard == nullptr)
            {
                throw std::invalid_argument("multiply_scalar_ciphertext: no matching parameter shard");
            }

            kernel::launch_multiply_scalar_poly_shard(
                dst,
                src,
                scalar,
                *parameter_shard,
                level_info.degree);
        }
    }
}

void GpuElementwiseHandler::add_plain_to_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &ciphertext_view,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
     if (!(ciphertext_view.meta.parms_id == plaintext_view.meta.parms_id) ||
        !(ciphertext_view.meta.parms_id == destination_view.meta.parms_id))
    {
        throw std::invalid_argument("add_plain_to_ciphertext: parms_id mismatch");
    }

    if (ciphertext_view.meta.is_ntt_form != plaintext_view.meta.is_ntt_form ||
        ciphertext_view.meta.is_ntt_form != destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument("add_plain_to_ciphertext: NTT form mismatch");
    }

    if (ciphertext_view.meta.degree != plaintext_view.meta.degree ||
        ciphertext_view.meta.degree != destination_view.meta.degree ||
        ciphertext_view.meta.q_count != plaintext_view.meta.q_count ||
        ciphertext_view.meta.q_count != destination_view.meta.q_count ||
        ciphertext_view.meta.p_count != plaintext_view.meta.p_count ||
        ciphertext_view.meta.p_count != destination_view.meta.p_count)
    {
        throw std::invalid_argument("add_plain_to_ciphertext: shape mismatch");
    }

    if (ciphertext_view.polys.empty())
    {
        throw std::invalid_argument("add_plain_to_ciphertext: empty ciphertext");
    }

    if (destination_view.polys.size() != ciphertext_view.polys.size())
    {
        throw std::invalid_argument(
            "add_plain_to_ciphertext: destination component count mismatch");
    }

    // CKKS rule:
    // destination.c0 = ciphertext.c0 + plaintext
    add_poly(
        destination_view.polys[0],
        ciphertext_view.polys[0],
        plaintext_view.poly,
        level_info);

    // destination.ci = ciphertext.ci for i > 0
    for (std::size_t i = 1; i < ciphertext_view.polys.size(); ++i)
    {
        copy_poly(
            destination_view.polys[i],
            ciphertext_view.polys[i],
            level_info);
    }
}

void GpuElementwiseHandler::sub_plain_from_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &ciphertext_view,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
    if (!(ciphertext_view.meta.parms_id == plaintext_view.meta.parms_id) ||
        !(ciphertext_view.meta.parms_id == destination_view.meta.parms_id))
    {
        throw std::invalid_argument("sub_plain_from_ciphertext: parms_id mismatch");
    }

    if (ciphertext_view.meta.is_ntt_form != plaintext_view.meta.is_ntt_form ||
        ciphertext_view.meta.is_ntt_form != destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument("sub_plain_from_ciphertext: NTT form mismatch");
    }

    if (ciphertext_view.meta.degree != plaintext_view.meta.degree ||
        ciphertext_view.meta.degree != destination_view.meta.degree ||
        ciphertext_view.meta.q_count != plaintext_view.meta.q_count ||
        ciphertext_view.meta.q_count != destination_view.meta.q_count ||
        ciphertext_view.meta.p_count != plaintext_view.meta.p_count ||
        ciphertext_view.meta.p_count != destination_view.meta.p_count)
    {
        throw std::invalid_argument("sub_plain_from_ciphertext: shape mismatch");
    }

    if (ciphertext_view.polys.empty())
    {
        throw std::invalid_argument("sub_plain_from_ciphertext: empty ciphertext");
    }

    if (destination_view.polys.size() != ciphertext_view.polys.size())
    {
        throw std::invalid_argument(
            "sub_plain_from_ciphertext: destination component count mismatch");
    }

    // CKKS rule:
    // destination.c0 = ciphertext.c0 - plaintext
    sub_poly(
        destination_view.polys[0],
        ciphertext_view.polys[0],
        plaintext_view.poly,
        level_info);

    // destination.ci = ciphertext.ci for i > 0
    for (std::size_t i = 1; i < ciphertext_view.polys.size(); ++i)
    {
        copy_poly(
            destination_view.polys[i],
            ciphertext_view.polys[i],
            level_info);
    }
}

void GpuElementwiseHandler::multiply_plain_with_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &ciphertext_view,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
    if (!(ciphertext_view.meta.parms_id == plaintext_view.meta.parms_id) ||
        !(ciphertext_view.meta.parms_id == destination_view.meta.parms_id))
    {
        throw std::invalid_argument("multiply_plain_with_ciphertext: parms_id mismatch");
    }

    if (ciphertext_view.meta.is_ntt_form != plaintext_view.meta.is_ntt_form ||
        ciphertext_view.meta.is_ntt_form != destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument("multiply_plain_with_ciphertext: NTT form mismatch");
    }

    if (destination_view.polys.size() != ciphertext_view.polys.size())
    {
        throw std::invalid_argument(
            "multiply_plain_with_ciphertext: destination component count mismatch");
    }

    for (std::size_t i = 0; i < ciphertext_view.polys.size(); ++i)
    {
        multiply_plain_poly(
            destination_view.polys[i],
            ciphertext_view.polys[i],
            plaintext_view.poly,
            level_info);
    }
}

void GpuElementwiseHandler::multiply_plain_accumulate_with_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &ciphertext_view,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
    if (!(destination_view.meta.parms_id == ciphertext_view.meta.parms_id) ||
        !(destination_view.meta.parms_id == plaintext_view.meta.parms_id) ||
        !destination_view.meta.is_ntt_form ||
        !ciphertext_view.meta.is_ntt_form ||
        !plaintext_view.meta.is_ntt_form ||
        destination_view.polys.size() != 2 ||
        ciphertext_view.polys.size() != 2)
    {
        throw std::invalid_argument(
            "multiply_plain_accumulate_with_ciphertext: incompatible input");
    }
    if (destination_view.polys[0].shards.size() != 1 ||
        destination_view.polys[1].shards.size() != 1 ||
        ciphertext_view.polys[0].shards.size() != 1 ||
        ciphertext_view.polys[1].shards.size() != 1 ||
        plaintext_view.poly.shards.size() != 1)
    {
        throw std::invalid_argument(
            "multiply_plain_accumulate_with_ciphertext: one shard required");
    }

    const auto &destination0 = destination_view.polys[0].shards.front();
    const auto *parameter_shard =
        find_parameter_shard(level_info, destination0);
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "multiply_plain_accumulate_with_ciphertext: no parameter shard");
    }
    kernel::launch_multiply_plain_accumulate_two_components(
        destination0,
        destination_view.polys[1].shards.front(),
        ciphertext_view.polys[0].shards.front(),
        ciphertext_view.polys[1].shards.front(),
        plaintext_view.poly.shards.front(),
        *parameter_shard,
        level_info.degree);
}

/**
 * @brief 真正构建d0 d1 d2的位置，再根据计算需求调用底层的kernel算子
 */
void GpuElementwiseHandler::multiply_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &left_view,
    const GpuConstCiphertextView &right_view,
    const GpuLevelInfo &level_info) const
{
    if (!(left_view.meta.parms_id == right_view.meta.parms_id) ||
        !(left_view.meta.parms_id == destination_view.meta.parms_id) ||
        !(left_view.meta.parms_id == level_info.parms_id))
    {
        throw std::invalid_argument("multiply_ciphertext: parms_id mismatch");
    }

    if (!left_view.meta.is_ntt_form ||
        !right_view.meta.is_ntt_form ||
        !destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument("multiply_ciphertext: CKKS inputs must be in NTT form");
    }

    if (left_view.meta.degree != right_view.meta.degree ||
        left_view.meta.degree != destination_view.meta.degree ||
        left_view.meta.degree != level_info.degree ||
        left_view.meta.q_count != right_view.meta.q_count ||
        left_view.meta.q_count != destination_view.meta.q_count ||
        left_view.meta.q_count != level_info.q_count ||
        left_view.meta.p_count != right_view.meta.p_count ||
        left_view.meta.p_count != destination_view.meta.p_count ||
        left_view.meta.p_count != level_info.p_count)
    {
        throw std::invalid_argument("multiply_ciphertext: shape mismatch");
    }

    if (left_view.meta.p_count != 0)
    {
        throw std::invalid_argument("multiply_ciphertext: p limbs are not supported yet");
    }

    if (left_view.polys.empty() || right_view.polys.empty())
    {
        throw std::invalid_argument("multiply_ciphertext: empty ciphertext");
    }

    const std::size_t result_count =
        left_view.polys.size() + right_view.polys.size() - 1;
    if (destination_view.polys.size() != result_count)
    {
        throw std::invalid_argument(
            "multiply_ciphertext: destination component count mismatch");
    }

    if (use_fused_ciphertext_multiply() &&
        left_view.polys.size() == 2 &&
        right_view.polys.size() == 2 &&
        destination_view.polys.size() == 3)
    {
        const std::size_t shard_count =
            destination_view.polys[0].shards.size();
        bool compatible = shard_count != 0;
        for (const auto &poly : destination_view.polys)
        {
            compatible = compatible && poly.shards.size() == shard_count;
        }
        for (const auto &poly : left_view.polys)
        {
            compatible = compatible && poly.shards.size() == shard_count;
        }
        for (const auto &poly : right_view.polys)
        {
            compatible = compatible && poly.shards.size() == shard_count;
        }

        for (std::size_t shard_index = 0;
             compatible && shard_index < shard_count;
             ++shard_index)
        {
            const auto &dst0 = destination_view.polys[0].shards[shard_index];
            compatible =
                same_shard_placement(
                    dst0,
                    destination_view.polys[1].shards[shard_index]) &&
                same_shard_placement(
                    dst0,
                    destination_view.polys[2].shards[shard_index]) &&
                same_shard_placement(
                    dst0,
                    left_view.polys[0].shards[shard_index]) &&
                same_shard_placement(
                    dst0,
                    left_view.polys[1].shards[shard_index]) &&
                same_shard_placement(
                    dst0,
                    right_view.polys[0].shards[shard_index]) &&
                same_shard_placement(
                    dst0,
                    right_view.polys[1].shards[shard_index]);
        }

        if (compatible)
        {
            for (std::size_t shard_index = 0;
                 shard_index < shard_count;
                 ++shard_index)
            {
                const auto &dst0 =
                    destination_view.polys[0].shards[shard_index];
                const auto *parameter_shard =
                    find_parameter_shard(level_info, dst0);
                if (parameter_shard == nullptr)
                {
                    throw std::invalid_argument(
                        "multiply_ciphertext: no matching parameter shard");
                }
                kernel::launch_multiply_two_component_ciphertexts(
                    dst0,
                    destination_view.polys[1].shards[shard_index],
                    destination_view.polys[2].shards[shard_index],
                    left_view.polys[0].shards[shard_index],
                    left_view.polys[1].shards[shard_index],
                    right_view.polys[0].shards[shard_index],
                    right_view.polys[1].shards[shard_index],
                    *parameter_shard,
                    level_info.degree);
            }
            return;
        }
    }

    for (std::size_t dest_component = 0;
         dest_component < result_count;
         ++dest_component)
    {
        const std::size_t left_first =
            dest_component >= right_view.polys.size()
                ? dest_component - right_view.polys.size() + 1
                : 0;
        const std::size_t left_last =
            std::min(dest_component, left_view.polys.size() - 1);

        bool first_product = true;
        for (std::size_t left_component = left_first;
             left_component <= left_last;
             ++left_component)
        {
            const std::size_t right_component =
                dest_component - left_component;

            if (first_product)
            {
                multiply_plain_poly(
                    destination_view.polys[dest_component],
                    left_view.polys[left_component],
                    right_view.polys[right_component],
                    level_info);
                first_product = false;
            }
            else
            {
                multiply_accumulate_poly(
                    destination_view.polys[dest_component],
                    left_view.polys[left_component],
                    right_view.polys[right_component],
                    level_info);
            }
        }
    }
}

void GpuElementwiseHandler::square_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &level_info) const
{
    if (!use_fused_ciphertext_multiply() ||
        source_view.polys.size() != 2 ||
        destination_view.polys.size() != 3)
    {
        multiply_ciphertext(
            destination_view,
            source_view,
            source_view,
            level_info);
        return;
    }
    if (!(source_view.meta.parms_id == destination_view.meta.parms_id) ||
        !(source_view.meta.parms_id == level_info.parms_id) ||
        !source_view.meta.is_ntt_form ||
        !destination_view.meta.is_ntt_form ||
        source_view.meta.degree != destination_view.meta.degree ||
        source_view.meta.degree != level_info.degree ||
        source_view.meta.q_count != destination_view.meta.q_count ||
        source_view.meta.q_count != level_info.q_count ||
        source_view.meta.p_count != 0 ||
        destination_view.meta.p_count != 0)
    {
        throw std::invalid_argument("square_ciphertext: input shape mismatch");
    }

    const std::size_t shard_count =
        destination_view.polys[0].shards.size();
    bool compatible = shard_count != 0;
    for (const auto &poly : destination_view.polys)
    {
        compatible = compatible && poly.shards.size() == shard_count;
    }
    for (const auto &poly : source_view.polys)
    {
        compatible = compatible && poly.shards.size() == shard_count;
    }
    for (std::size_t shard_index = 0;
         compatible && shard_index < shard_count;
         ++shard_index)
    {
        const auto &dst0 = destination_view.polys[0].shards[shard_index];
        compatible =
            same_shard_placement(
                dst0,
                destination_view.polys[1].shards[shard_index]) &&
            same_shard_placement(
                dst0,
                destination_view.polys[2].shards[shard_index]) &&
            same_shard_placement(
                dst0,
                source_view.polys[0].shards[shard_index]) &&
            same_shard_placement(
                dst0,
                source_view.polys[1].shards[shard_index]);
    }
    if (!compatible)
    {
        multiply_ciphertext(
            destination_view,
            source_view,
            source_view,
            level_info);
        return;
    }

    for (std::size_t shard_index = 0;
         shard_index < shard_count;
         ++shard_index)
    {
        const auto &dst0 =
            destination_view.polys[0].shards[shard_index];
        const auto *parameter_shard =
            find_parameter_shard(level_info, dst0);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument(
                "square_ciphertext: no matching parameter shard");
        }
        kernel::launch_square_two_component_ciphertext(
            dst0,
            destination_view.polys[1].shards[shard_index],
            destination_view.polys[2].shards[shard_index],
            source_view.polys[0].shards[shard_index],
            source_view.polys[1].shards[shard_index],
            *parameter_shard,
            level_info.degree);
    }
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
    if (destination_poly.shards.size() != ciphertext_poly.shards.size() ||
        destination_poly.shards.size() != plaintext_poly.shards.size())
    {
        throw std::invalid_argument("multiply_plain_poly: shard count mismatch");
    }

    for (std::size_t i = 0; i < destination_poly.shards.size(); ++i)
    {
        const auto &dst = destination_poly.shards[i];
        const auto &ct = ciphertext_poly.shards[i];
        const auto &plain = plaintext_poly.shards[i];

        if (dst.device_id != ct.device_id ||
            dst.device_id != plain.device_id ||
            dst.limb_begin != ct.limb_begin ||
            dst.limb_begin != plain.limb_begin ||
            dst.limb_count != ct.limb_count ||
            dst.limb_count != plain.limb_count ||
            dst.coeff_begin != ct.coeff_begin ||
            dst.coeff_begin != plain.coeff_begin ||
            dst.coeff_count != ct.coeff_count ||
            dst.coeff_count != plain.coeff_count)
        {
            throw std::invalid_argument("multiply_plain_poly: shard placement mismatch");
        }

        const GpuParameterShard *parameter_shard =
            find_parameter_shard(level_info, dst);

        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("multiply_plain_poly: no matching parameter shard");
        }

        kernel::launch_dyadic_product_poly_shard(
            dst,
            ct,
            plain,
            *parameter_shard,
            level_info.degree);
    }
}

void GpuElementwiseHandler::multiply_accumulate_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &left_poly,
    const GpuConstRNSPolyView &right_poly,
    const GpuLevelInfo &level_info) const
{
    if (destination_poly.shards.size() != left_poly.shards.size() ||
        destination_poly.shards.size() != right_poly.shards.size())
    {
        throw std::invalid_argument("multiply_accumulate_poly: shard count mismatch");
    }

    for (std::size_t i = 0; i < destination_poly.shards.size(); ++i)
    {
        const auto &dst = destination_poly.shards[i];
        const auto &lhs = left_poly.shards[i];
        const auto &rhs = right_poly.shards[i];

        if (!same_shard_placement(dst, lhs) ||
            !same_shard_placement(dst, rhs))
        {
            throw std::invalid_argument("multiply_accumulate_poly: shard placement mismatch");
        }

        const GpuParameterShard *parameter_shard =
            find_parameter_shard(level_info, dst);

        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("multiply_accumulate_poly: no matching parameter shard");
        }

        kernel::launch_multiply_accumulate_poly_shard(
            dst,
            lhs,
            rhs,
            *parameter_shard,
            level_info.degree);
    }
}

}  // namespace gpu
}  // namespace poseidon
