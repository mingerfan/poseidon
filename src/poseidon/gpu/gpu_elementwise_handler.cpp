#include "poseidon/gpu/gpu_elementwise_handler.h"
#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"

#include <algorithm>
#include <array>
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

void GpuElementwiseHandler::multiply_plain_sum(
    GpuCiphertextView &destination_view,
    const std::vector<GpuConstCiphertextView> &ciphertexts,
    const std::vector<GpuConstPlaintextView> &plaintexts,
    const GpuLevelInfo &level_info,
    bool accumulate) const
{
    if (ciphertexts.empty() || ciphertexts.size() > 4 ||
        ciphertexts.size() != plaintexts.size() ||
        destination_view.polys.size() != 2 ||
        destination_view.polys[0].shards.size() != 1 ||
        destination_view.polys[1].shards.size() != 1)
        throw std::invalid_argument("multiply_plain_sum: unsupported shape");
    const auto &dst0 = destination_view.polys[0].shards.front();
    const auto &dst1 = destination_view.polys[1].shards.front();
    if (!same_shard_placement(dst0, dst1))
        throw std::invalid_argument("multiply_plain_sum: destination shard mismatch");
    std::array<GpuConstPolyShardView, 4> c0{}, c1{}, plain{};
    for (std::size_t i = 0; i < ciphertexts.size(); ++i)
    {
        const auto &ct = ciphertexts[i];
        const auto &pt = plaintexts[i];
        if (ct.polys.size() != 2 || ct.polys[0].shards.size() != 1 ||
            ct.polys[1].shards.size() != 1 || pt.poly.shards.size() != 1 ||
            ct.meta.parms_id != level_info.parms_id ||
            pt.meta.parms_id != level_info.parms_id)
            throw std::invalid_argument("multiply_plain_sum: source shape mismatch");
        c0[i] = ct.polys[0].shards.front();
        c1[i] = ct.polys[1].shards.front();
        plain[i] = pt.poly.shards.front();
        if (!same_shard_placement(dst0, c0[i]) ||
            !same_shard_placement(dst0, c1[i]) ||
            !same_shard_placement(dst0, plain[i]))
            throw std::invalid_argument("multiply_plain_sum: source shard mismatch");
    }
    const auto *shard = find_parameter_shard(level_info, dst0);
    if (shard == nullptr)
        throw std::invalid_argument("multiply_plain_sum: parameter shard absent");
    kernel::launch_multiply_plain_caccumulate_two_components_4(
        dst0, dst1, c0.data(), c1.data(), plain.data(), ciphertexts.size(),
        nullptr, accumulate, *shard, level_info.degree);
}

void GpuElementwiseHandler::double_sub_plain(
    GpuCiphertextView &destination, const GpuConstCiphertextView &source,
    const GpuConstPlaintextView &plaintext, const GpuConstCiphertextView *correction,
    const GpuLevelInfo &level) const
{
    const auto valid = [&](const auto &ct) {
        return ct.meta.parms_id == level.parms_id && ct.polys.size() == 2 &&
            ct.polys[0].shards.size() == 1 && ct.polys[1].shards.size() == 1;
    };
    if (!valid(destination) || !valid(source) || (correction && !valid(*correction)) ||
        plaintext.meta.parms_id != level.parms_id || plaintext.poly.shards.size() != 1)
        throw std::invalid_argument("double_sub_plain: unsupported shape");
    const auto *parameters = find_parameter_shard(level, destination.polys[0].shards[0]);
    if (!parameters) throw std::invalid_argument("double_sub_plain: parameter shard absent");
    kernel::launch_double_sub_plain_two_components(
        destination.polys[0].shards[0], destination.polys[1].shards[0],
        source.polys[0].shards[0], source.polys[1].shards[0], plaintext.poly.shards[0],
        correction ? &correction->polys[0].shards[0] : nullptr,
        correction ? &correction->polys[1].shards[0] : nullptr, *parameters);
}

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
    std::uint64_t scalar,
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

namespace
{

GpuConstPolyShardView const_alias(const GpuPolyShardView &view)
{
    GpuConstPolyShardView result;
    result.device_id = view.device_id;
    result.ptr = view.ptr;
    result.limb_begin = view.limb_begin;
    result.limb_count = view.limb_count;
    result.coeff_begin = view.coeff_begin;
    result.coeff_count = view.coeff_count;
    return result;
}

}  // namespace

void GpuElementwiseHandler::add_plain_to_ciphertext_inplace(
    GpuCiphertextView &ciphertext_view,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
    if (!(ciphertext_view.meta.parms_id == plaintext_view.meta.parms_id) ||
        !ciphertext_view.meta.is_ntt_form ||
        !plaintext_view.meta.is_ntt_form ||
        ciphertext_view.meta.degree != plaintext_view.meta.degree ||
        ciphertext_view.meta.q_count != plaintext_view.meta.q_count ||
        ciphertext_view.meta.p_count != plaintext_view.meta.p_count ||
        ciphertext_view.polys.empty() ||
        ciphertext_view.polys[0].shards.size() != plaintext_view.poly.shards.size())
    {
        throw std::invalid_argument(
            "add_plain_to_ciphertext_inplace: incompatible input");
    }

    for (std::size_t i = 0; i < ciphertext_view.polys[0].shards.size(); ++i)
    {
        const auto &dst = ciphertext_view.polys[0].shards[i];
        const auto &plain = plaintext_view.poly.shards[i];
        const auto *parameter_shard = find_parameter_shard(level_info, dst);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument(
                "add_plain_to_ciphertext_inplace: no parameter shard");
        }
        kernel::launch_add_poly_shard(
            dst,
            const_alias(dst),
            plain,
            *parameter_shard,
            level_info.degree);
    }
}

void GpuElementwiseHandler::sub_plain_from_ciphertext_inplace(
    GpuCiphertextView &ciphertext_view,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
    if (!(ciphertext_view.meta.parms_id == plaintext_view.meta.parms_id) ||
        !ciphertext_view.meta.is_ntt_form ||
        !plaintext_view.meta.is_ntt_form ||
        ciphertext_view.meta.degree != plaintext_view.meta.degree ||
        ciphertext_view.meta.q_count != plaintext_view.meta.q_count ||
        ciphertext_view.meta.p_count != plaintext_view.meta.p_count ||
        ciphertext_view.polys.empty() ||
        ciphertext_view.polys[0].shards.size() != plaintext_view.poly.shards.size())
    {
        throw std::invalid_argument(
            "sub_plain_from_ciphertext_inplace: incompatible input");
    }

    for (std::size_t i = 0; i < ciphertext_view.polys[0].shards.size(); ++i)
    {
        const auto &dst = ciphertext_view.polys[0].shards[i];
        const auto &plain = plaintext_view.poly.shards[i];
        const auto *parameter_shard = find_parameter_shard(level_info, dst);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument(
                "sub_plain_from_ciphertext_inplace: no parameter shard");
        }
        kernel::launch_sub_poly_shard(
            dst,
            const_alias(dst),
            plain,
            *parameter_shard,
            level_info.degree);
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
        destination_view.polys.size() < 2 ||
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

namespace
{

template <typename ViewT>
bool same_ciphertext_shape_as_level(
    const ViewT &view,
    const GpuLevelInfo &level_info,
    std::size_t expected_components = 0)
{
    if (!(view.meta.parms_id == level_info.parms_id) ||
        !view.meta.is_ntt_form ||
        view.meta.degree != level_info.degree ||
        view.meta.q_count != level_info.q_count ||
        view.meta.p_count != 0 ||
        view.polys.empty())
    {
        return false;
    }
    if (expected_components != 0 && view.polys.size() != expected_components)
    {
        return false;
    }
    const std::size_t shard_count = view.polys[0].shards.size();
    if (shard_count == 0)
    {
        return false;
    }
    for (std::size_t c = 0; c < view.polys.size(); ++c)
    {
        if (view.polys[c].shards.size() != shard_count)
        {
            return false;
        }
        for (std::size_t s = 0; s < shard_count; ++s)
        {
            const auto &ref = view.polys[0].shards[s];
            const auto &cur = view.polys[c].shards[s];
            if (cur.device_id != ref.device_id ||
                cur.limb_begin != ref.limb_begin ||
                cur.limb_count != ref.limb_count ||
                cur.coeff_begin != ref.coeff_begin ||
                cur.coeff_count != ref.coeff_count)
            {
                return false;
            }
        }
    }
    return true;
}

template <typename ShardA, typename ShardB>
bool matching_shard_placement(const ShardA &a, const ShardB &b)
{
    return a.device_id == b.device_id &&
           a.limb_begin == b.limb_begin &&
           a.limb_count == b.limb_count &&
           a.coeff_begin == b.coeff_begin &&
           a.coeff_count == b.coeff_count;
}

bool views_share_pair_shards(
    const GpuConstCiphertextView &lhs_a,
    const GpuConstCiphertextView &rhs_a,
    const GpuCiphertextView &dst_a,
    const GpuConstCiphertextView &lhs_b,
    const GpuConstCiphertextView &rhs_b,
    const GpuCiphertextView &dst_b)
{
    const std::size_t shard_count = dst_a.polys[0].shards.size();
    if (lhs_b.polys[0].shards.size() != shard_count ||
        rhs_a.polys[0].shards.size() != shard_count ||
        rhs_b.polys[0].shards.size() != shard_count ||
        dst_b.polys[0].shards.size() != shard_count)
    {
        return false;
    }
    for (std::size_t s = 0; s < shard_count; ++s)
    {
        const auto &ref = dst_a.polys[0].shards[s];
        if (!matching_shard_placement(ref, lhs_a.polys[0].shards[s]) ||
            !matching_shard_placement(ref, rhs_a.polys[0].shards[s]) ||
            !matching_shard_placement(ref, lhs_b.polys[0].shards[s]) ||
            !matching_shard_placement(ref, rhs_b.polys[0].shards[s]) ||
            !matching_shard_placement(ref, dst_b.polys[0].shards[s]))
        {
            return false;
        }
    }
    return true;
}

}  // namespace

void GpuElementwiseHandler::multiply_plain_pair(
    GpuCiphertextView &destination_a_view,
    GpuCiphertextView &destination_b_view,
    const GpuConstCiphertextView &ciphertext_a_view,
    const GpuConstCiphertextView &ciphertext_b_view,
    const GpuConstRNSPolyView &plaintext_poly,
    const GpuLevelInfo &level_info) const
{
    const std::size_t count_a = ciphertext_a_view.polys.size();
    const std::size_t count_b = ciphertext_b_view.polys.size();
    const std::size_t total_count = count_a + count_b;
    const bool shapes_ok =
        total_count > 0 &&
        total_count <= kernel::kMaxElementwisePairPtrs &&
        same_ciphertext_shape_as_level(ciphertext_a_view, level_info) &&
        same_ciphertext_shape_as_level(destination_a_view, level_info, count_a) &&
        same_ciphertext_shape_as_level(ciphertext_b_view, level_info) &&
        same_ciphertext_shape_as_level(destination_b_view, level_info, count_b) &&
        plaintext_poly.shards.size() == destination_a_view.polys[0].shards.size() &&
        destination_b_view.polys[0].shards.size() == destination_a_view.polys[0].shards.size();

    bool shards_ok = shapes_ok;
    if (shards_ok)
    {
        for (std::size_t s = 0; s < destination_a_view.polys[0].shards.size(); ++s)
        {
            const auto &ref = destination_a_view.polys[0].shards[s];
            if (!matching_shard_placement(ref, ciphertext_a_view.polys[0].shards[s]) ||
                !matching_shard_placement(ref, ciphertext_b_view.polys[0].shards[s]) ||
                !matching_shard_placement(ref, destination_b_view.polys[0].shards[s]) ||
                !matching_shard_placement(ref, plaintext_poly.shards[s]))
            {
                shards_ok = false;
                break;
            }
        }
    }

    if (!shards_ok)
    {
        for (std::size_t c = 0; c < count_a; ++c)
        {
            multiply_plain_poly(
                destination_a_view.polys[c], ciphertext_a_view.polys[c], plaintext_poly, level_info);
        }
        for (std::size_t c = 0; c < count_b; ++c)
        {
            multiply_plain_poly(
                destination_b_view.polys[c], ciphertext_b_view.polys[c], plaintext_poly, level_info);
        }
        return;
    }

    const std::size_t shard_count = destination_a_view.polys[0].shards.size();
    for (std::size_t s = 0; s < shard_count; ++s)
    {
        const auto &ref = destination_a_view.polys[0].shards[s];
        const GpuParameterShard *parameter_shard = find_parameter_shard(level_info, ref);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("multiply_plain_pair: no matching parameter shard");
        }
        kernel::MultiplyPlainPtrsArgs args{};
        std::size_t idx = 0;
        for (std::size_t c = 0; c < count_a; ++c, ++idx)
        {
            args.dst[idx] = destination_a_view.polys[c].shards[s].ptr;
            args.src[idx] = ciphertext_a_view.polys[c].shards[s].ptr;
        }
        for (std::size_t c = 0; c < count_b; ++c, ++idx)
        {
            args.dst[idx] = destination_b_view.polys[c].shards[s].ptr;
            args.src[idx] = ciphertext_b_view.polys[c].shards[s].ptr;
        }
        kernel::launch_multiply_plain_ptrs(
            args, total_count, plaintext_poly.shards[s], ref, *parameter_shard, level_info.degree);
    }
}

void GpuElementwiseHandler::add_or_sub_ciphertext_pair(
    GpuCiphertextView &destination_a_view,
    GpuCiphertextView &destination_b_view,
    const GpuConstCiphertextView &left_a_view,
    const GpuConstCiphertextView &right_a_view,
    const GpuConstCiphertextView &left_b_view,
    const GpuConstCiphertextView &right_b_view,
    bool subtract,
    const GpuLevelInfo &level_info) const
{
    const std::size_t count_a = left_a_view.polys.size();
    const std::size_t count_b = left_b_view.polys.size();
    const std::size_t total_count = count_a + count_b;
    const bool fast_ok =
        total_count > 0 &&
        total_count <= kernel::kMaxElementwisePairPtrs &&
        same_ciphertext_shape_as_level(left_a_view, level_info) &&
        same_ciphertext_shape_as_level(right_a_view, level_info, count_a) &&
        same_ciphertext_shape_as_level(destination_a_view, level_info, count_a) &&
        same_ciphertext_shape_as_level(left_b_view, level_info) &&
        same_ciphertext_shape_as_level(right_b_view, level_info, count_b) &&
        same_ciphertext_shape_as_level(destination_b_view, level_info, count_b) &&
        views_share_pair_shards(
            left_a_view, right_a_view, destination_a_view,
            left_b_view, right_b_view, destination_b_view);

    if (!fast_ok)
    {
        if (subtract)
        {
            sub_ciphertext(destination_a_view, left_a_view, right_a_view, level_info);
            sub_ciphertext(destination_b_view, left_b_view, right_b_view, level_info);
        }
        else
        {
            add_ciphertext(destination_a_view, left_a_view, right_a_view, level_info);
            add_ciphertext(destination_b_view, left_b_view, right_b_view, level_info);
        }
        return;
    }

    const std::size_t shard_count = destination_a_view.polys[0].shards.size();
    for (std::size_t s = 0; s < shard_count; ++s)
    {
        const auto &ref = destination_a_view.polys[0].shards[s];
        const GpuParameterShard *parameter_shard = find_parameter_shard(level_info, ref);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("add_or_sub_ciphertext_pair: no matching parameter shard");
        }
        kernel::BinaryPolyPtrsArgs args{};
        std::size_t idx = 0;
        for (std::size_t c = 0; c < count_a; ++c, ++idx)
        {
            args.dst[idx] = destination_a_view.polys[c].shards[s].ptr;
            args.lhs[idx] = left_a_view.polys[c].shards[s].ptr;
            args.rhs[idx] = right_a_view.polys[c].shards[s].ptr;
        }
        for (std::size_t c = 0; c < count_b; ++c, ++idx)
        {
            args.dst[idx] = destination_b_view.polys[c].shards[s].ptr;
            args.lhs[idx] = left_b_view.polys[c].shards[s].ptr;
            args.rhs[idx] = right_b_view.polys[c].shards[s].ptr;
        }
        kernel::launch_add_or_sub_ptrs(
            args, total_count, subtract, ref, *parameter_shard);
    }
}

void GpuElementwiseHandler::multiply_ciphertext_pair(
    GpuCiphertextView &destination_a_view,
    GpuCiphertextView &destination_b_view,
    const GpuConstCiphertextView &left_a_view,
    const GpuConstCiphertextView &right_a_view,
    const GpuConstCiphertextView &left_b_view,
    const GpuConstCiphertextView &right_b_view,
    const GpuLevelInfo &level_info) const
{
    const bool fast_ok =
        use_fused_ciphertext_multiply() &&
        same_ciphertext_shape_as_level(left_a_view, level_info, 2) &&
        same_ciphertext_shape_as_level(right_a_view, level_info, 2) &&
        same_ciphertext_shape_as_level(destination_a_view, level_info, 3) &&
        same_ciphertext_shape_as_level(left_b_view, level_info, 2) &&
        same_ciphertext_shape_as_level(right_b_view, level_info, 2) &&
        same_ciphertext_shape_as_level(destination_b_view, level_info, 3) &&
        views_share_pair_shards(
            left_a_view, right_a_view, destination_a_view,
            left_b_view, right_b_view, destination_b_view);

    if (!fast_ok)
    {
        multiply_ciphertext(destination_a_view, left_a_view, right_a_view, level_info);
        multiply_ciphertext(destination_b_view, left_b_view, right_b_view, level_info);
        return;
    }

    const std::size_t shard_count = destination_a_view.polys[0].shards.size();
    for (std::size_t s = 0; s < shard_count; ++s)
    {
        const auto &ref = destination_a_view.polys[0].shards[s];
        const GpuParameterShard *parameter_shard = find_parameter_shard(level_info, ref);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("multiply_ciphertext_pair: no matching parameter shard");
        }
        const bool is_square =
            left_a_view.polys[0].shards[s].ptr == right_a_view.polys[0].shards[s].ptr &&
            left_a_view.polys[1].shards[s].ptr == right_a_view.polys[1].shards[s].ptr &&
            left_b_view.polys[0].shards[s].ptr == right_b_view.polys[0].shards[s].ptr &&
            left_b_view.polys[1].shards[s].ptr == right_b_view.polys[1].shards[s].ptr;
        kernel::launch_multiply_pair_two_components(
            destination_a_view.polys[0].shards[s],
            destination_a_view.polys[1].shards[s],
            destination_a_view.polys[2].shards[s],
            left_a_view.polys[0].shards[s],
            left_a_view.polys[1].shards[s],
            right_a_view.polys[0].shards[s],
            right_a_view.polys[1].shards[s],
            &destination_b_view.polys[0].shards[s],
            &destination_b_view.polys[1].shards[s],
            &destination_b_view.polys[2].shards[s],
            &left_b_view.polys[0].shards[s],
            &left_b_view.polys[1].shards[s],
            &right_b_view.polys[0].shards[s],
            &right_b_view.polys[1].shards[s],
            is_square,
            *parameter_shard,
            level_info.degree);
    }
}

void GpuElementwiseHandler::square_ciphertext_pair(
    GpuCiphertextView &destination_a_view,
    GpuCiphertextView &destination_b_view,
    const GpuConstCiphertextView &source_a_view,
    const GpuConstCiphertextView &source_b_view,
    const GpuLevelInfo &level_info) const
{
    multiply_ciphertext_pair(
        destination_a_view,
        destination_b_view,
        source_a_view,
        source_a_view,
        source_b_view,
        source_b_view,
        level_info);
}

void GpuElementwiseHandler::double_sub_or_add_plain_pair(
    kernel::DoublePlainPairOp op,
    GpuCiphertextView &destination_a_view,
    GpuCiphertextView *destination_b_view,
    const GpuConstCiphertextView &source_a_view,
    const GpuConstCiphertextView *source_b_view,
    const GpuConstRNSPolyView &plaintext_poly,
    const GpuConstCiphertextView *correction_a_view,
    const GpuConstCiphertextView *correction_b_view,
    const GpuLevelInfo &level_info) const
{
    const bool has_b = destination_b_view != nullptr && source_b_view != nullptr;
    if ((destination_b_view == nullptr) != (source_b_view == nullptr))
    {
        throw std::invalid_argument("double_sub_or_add_plain_pair: lane B mismatch");
    }
    if (!same_ciphertext_shape_as_level(source_a_view, level_info, 2) ||
        !same_ciphertext_shape_as_level(destination_a_view, level_info, 2))
    {
        throw std::invalid_argument("double_sub_or_add_plain_pair: lane A shape mismatch");
    }
    const std::size_t shard_count = destination_a_view.polys[0].shards.size();
    if (plaintext_poly.shards.size() != shard_count)
    {
        throw std::invalid_argument("double_sub_or_add_plain_pair: plaintext shard count mismatch");
    }
    if (has_b &&
        (!same_ciphertext_shape_as_level(*source_b_view, level_info, 2) ||
         !same_ciphertext_shape_as_level(*destination_b_view, level_info, 2) ||
         destination_b_view->polys[0].shards.size() != shard_count))
    {
        throw std::invalid_argument("double_sub_or_add_plain_pair: lane B shape mismatch");
    }
    if (op == kernel::DoublePlainPairOp::SubCorr)
    {
        if (correction_a_view == nullptr ||
            !same_ciphertext_shape_as_level(*correction_a_view, level_info, 2) ||
            correction_a_view->polys[0].shards.size() != shard_count ||
            (has_b &&
             (correction_b_view == nullptr ||
              !same_ciphertext_shape_as_level(*correction_b_view, level_info, 2) ||
              correction_b_view->polys[0].shards.size() != shard_count)))
        {
            throw std::invalid_argument("double_sub_or_add_plain_pair: correction shape mismatch");
        }
    }

    for (std::size_t s = 0; s < shard_count; ++s)
    {
        const auto &dst_a0 = destination_a_view.polys[0].shards[s];
        const GpuParameterShard *parameter_shard = find_parameter_shard(level_info, dst_a0);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("double_sub_or_add_plain_pair: no matching parameter shard");
        }
        kernel::launch_double_sub_or_add_plain_pair(
            op,
            destination_a_view.polys[0].shards[s],
            destination_a_view.polys[1].shards[s],
            source_a_view.polys[0].shards[s],
            source_a_view.polys[1].shards[s],
            has_b ? &destination_b_view->polys[0].shards[s] : nullptr,
            has_b ? &destination_b_view->polys[1].shards[s] : nullptr,
            has_b ? &source_b_view->polys[0].shards[s] : nullptr,
            has_b ? &source_b_view->polys[1].shards[s] : nullptr,
            plaintext_poly.shards[s],
            correction_a_view ? &correction_a_view->polys[0].shards[s] : nullptr,
            correction_a_view ? &correction_a_view->polys[1].shards[s] : nullptr,
            correction_b_view ? &correction_b_view->polys[0].shards[s] : nullptr,
            correction_b_view ? &correction_b_view->polys[1].shards[s] : nullptr,
            *parameter_shard);
    }
}

void GpuElementwiseHandler::multiply_plain_batch(
    std::vector<GpuCiphertextView> &destination_views,
    const std::vector<GpuConstCiphertextView> &ciphertext_views,
    const GpuConstPlaintextView &plaintext_view,
    const GpuLevelInfo &level_info) const
{
    const std::size_t count = ciphertext_views.size();
    if (count == 0 || destination_views.size() != count)
    {
        throw std::invalid_argument("GpuElementwiseHandler::multiply_plain_batch: size mismatch");
    }
    if (count == 1)
    {
        multiply_plain_with_ciphertext(
            destination_views[0], ciphertext_views[0], plaintext_view, level_info);
        return;
    }
    const auto &plaintext_poly = plaintext_view.poly;
    const std::size_t per_ct = ciphertext_views[0].polys.size();
    const std::size_t total_polys = count * per_ct;
    const std::size_t shard_count = plaintext_poly.shards.size();
    bool fast_ok =
        total_polys <= kernel::kMaxElementwisePairPtrs &&
        per_ct > 0 && shard_count > 0 &&
        plaintext_view.meta.is_ntt_form &&
        !(plaintext_view.meta.parms_id != level_info.parms_id) &&
        same_ciphertext_shape_as_level(ciphertext_views[0], level_info, per_ct) &&
        same_ciphertext_shape_as_level(destination_views[0], level_info, per_ct) &&
        ciphertext_views[0].polys[0].shards.size() == shard_count &&
        destination_views[0].polys[0].shards.size() == shard_count;
    for (std::size_t i = 0; fast_ok && i < count; ++i)
    {
        if (ciphertext_views[i].polys.size() != per_ct ||
            destination_views[i].polys.size() != per_ct ||
            !same_ciphertext_shape_as_level(ciphertext_views[i], level_info, per_ct) ||
            !same_ciphertext_shape_as_level(destination_views[i], level_info, per_ct) ||
            ciphertext_views[i].polys[0].shards.size() != shard_count ||
            destination_views[i].polys[0].shards.size() != shard_count)
        {
            fast_ok = false;
        }
        for (std::size_t c = 0; fast_ok && c < per_ct; ++c)
        {
            for (std::size_t s = 0; fast_ok && s < shard_count; ++s)
            {
                if (!matching_shard_placement(
                        ciphertext_views[i].polys[c].shards[s], plaintext_poly.shards[s]) ||
                    !matching_shard_placement(
                        destination_views[i].polys[c].shards[s], plaintext_poly.shards[s]))
                {
                    fast_ok = false;
                }
            }
        }
    }
    if (!fast_ok)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            multiply_plain_with_ciphertext(
                destination_views[i], ciphertext_views[i], plaintext_view, level_info);
        }
        return;
    }

    for (std::size_t s = 0; s < shard_count; ++s)
    {
        const auto &dst0 = destination_views[0].polys[0].shards[s];
        const GpuParameterShard *parameter_shard = find_parameter_shard(level_info, dst0);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("multiply_plain_batch: no matching parameter shard");
        }
        kernel::MultiplyPlainPtrsArgs args{};
        std::size_t idx = 0;
        for (std::size_t i = 0; i < count; ++i)
        {
            for (std::size_t c = 0; c < per_ct; ++c, ++idx)
            {
                args.dst[idx] = destination_views[i].polys[c].shards[s].ptr;
                args.src[idx] = ciphertext_views[i].polys[c].shards[s].ptr;
            }
        }
        kernel::launch_multiply_plain_ptrs(
            args, total_polys, plaintext_poly.shards[s], dst0, *parameter_shard, level_info.degree);
    }
}

void GpuElementwiseHandler::add_or_sub_ciphertext_batch(
    std::vector<GpuCiphertextView> &destination_views,
    const std::vector<GpuConstCiphertextView> &left_views,
    const std::vector<GpuConstCiphertextView> &right_views,
    bool subtract,
    const GpuLevelInfo &level_info) const
{
    const std::size_t count = left_views.size();
    if (count == 0 || right_views.size() != count || destination_views.size() != count)
    {
        throw std::invalid_argument("GpuElementwiseHandler::add_or_sub_ciphertext_batch: size mismatch");
    }
    if (count == 1)
    {
        if (subtract) sub_ciphertext(destination_views[0], left_views[0], right_views[0], level_info);
        else add_ciphertext(destination_views[0], left_views[0], right_views[0], level_info);
        return;
    }
    const std::size_t per_ct = left_views[0].polys.size();
    const std::size_t total_polys = count * per_ct;
    const std::size_t shard_count = per_ct > 0 ? left_views[0].polys[0].shards.size() : 0;
    bool fast_ok =
        total_polys <= kernel::kMaxElementwisePairPtrs &&
        per_ct > 0 && shard_count > 0;
    for (std::size_t i = 0; fast_ok && i < count; ++i)
    {
        if (!same_ciphertext_shape_as_level(left_views[i], level_info, per_ct) ||
            !same_ciphertext_shape_as_level(right_views[i], level_info, per_ct) ||
            !same_ciphertext_shape_as_level(destination_views[i], level_info, per_ct) ||
            left_views[i].polys[0].shards.size() != shard_count ||
            right_views[i].polys[0].shards.size() != shard_count ||
            destination_views[i].polys[0].shards.size() != shard_count)
        {
            fast_ok = false;
        }
        for (std::size_t c = 0; fast_ok && c < per_ct; ++c)
        {
            for (std::size_t s = 0; fast_ok && s < shard_count; ++s)
            {
                const auto &ref = left_views[0].polys[0].shards[s];
                if (!matching_shard_placement(left_views[i].polys[c].shards[s], ref) ||
                    !matching_shard_placement(right_views[i].polys[c].shards[s], ref) ||
                    !matching_shard_placement(destination_views[i].polys[c].shards[s], ref))
                {
                    fast_ok = false;
                }
            }
        }
    }
    if (!fast_ok)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            if (subtract) sub_ciphertext(destination_views[i], left_views[i], right_views[i], level_info);
            else add_ciphertext(destination_views[i], left_views[i], right_views[i], level_info);
        }
        return;
    }

    for (std::size_t s = 0; s < shard_count; ++s)
    {
        const auto &dst0 = destination_views[0].polys[0].shards[s];
        const GpuParameterShard *parameter_shard = find_parameter_shard(level_info, dst0);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("add_or_sub_ciphertext_batch: no matching parameter shard");
        }
        kernel::BinaryPolyPtrsArgs args{};
        std::size_t idx = 0;
        for (std::size_t i = 0; i < count; ++i)
        {
            for (std::size_t c = 0; c < per_ct; ++c, ++idx)
            {
                args.dst[idx] = destination_views[i].polys[c].shards[s].ptr;
                args.lhs[idx] = left_views[i].polys[c].shards[s].ptr;
                args.rhs[idx] = right_views[i].polys[c].shards[s].ptr;
            }
        }
        kernel::launch_add_or_sub_ptrs(
            args, total_polys, subtract, dst0, *parameter_shard);
    }
}

void GpuElementwiseHandler::multiply_ciphertext_batch(
    std::vector<GpuCiphertextView> &destination_views,
    const std::vector<GpuConstCiphertextView> &left_views,
    const std::vector<GpuConstCiphertextView> &right_views,
    bool is_square,
    const GpuLevelInfo &level_info) const
{
    const std::size_t count = left_views.size();
    if (count == 0 || right_views.size() != count || destination_views.size() != count)
    {
        throw std::invalid_argument("GpuElementwiseHandler::multiply_ciphertext_batch: size mismatch");
    }
    if (count == 1)
    {
        if (is_square) square_ciphertext(destination_views[0], left_views[0], level_info);
        else multiply_ciphertext(destination_views[0], left_views[0], right_views[0], level_info);
        return;
    }
    const std::size_t shard_count =
        left_views[0].polys.empty() ? 0 : left_views[0].polys[0].shards.size();
    bool fast_ok =
        count <= kernel::kMaxElementwiseCiphertextBatch &&
        shard_count > 0 &&
        use_fused_ciphertext_multiply();
    for (std::size_t i = 0; fast_ok && i < count; ++i)
    {
        if (!same_ciphertext_shape_as_level(left_views[i], level_info, 2) ||
            !same_ciphertext_shape_as_level(right_views[i], level_info, 2) ||
            !same_ciphertext_shape_as_level(destination_views[i], level_info, 3) ||
            left_views[i].polys[0].shards.size() != shard_count ||
            right_views[i].polys[0].shards.size() != shard_count ||
            destination_views[i].polys[0].shards.size() != shard_count)
        {
            fast_ok = false;
        }
        for (std::size_t s = 0; fast_ok && s < shard_count; ++s)
        {
            const auto &ref = left_views[0].polys[0].shards[s];
            for (std::size_t c = 0; c < 2; ++c)
            {
                if (!matching_shard_placement(left_views[i].polys[c].shards[s], ref) ||
                    !matching_shard_placement(right_views[i].polys[c].shards[s], ref) ||
                    !matching_shard_placement(destination_views[i].polys[c].shards[s], ref))
                {
                    fast_ok = false;
                }
            }
            if (!matching_shard_placement(destination_views[i].polys[2].shards[s], ref))
            {
                fast_ok = false;
            }
        }
    }
    if (!fast_ok)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            if (is_square) square_ciphertext(destination_views[i], left_views[i], level_info);
            else multiply_ciphertext(destination_views[i], left_views[i], right_views[i], level_info);
        }
        return;
    }

    for (std::size_t s = 0; s < shard_count; ++s)
    {
        const auto &dst0 = destination_views[0].polys[0].shards[s];
        const GpuParameterShard *parameter_shard = find_parameter_shard(level_info, dst0);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("multiply_ciphertext_batch: no matching parameter shard");
        }
        kernel::MultiplyTwoComponentsBatchArgs args{};
        for (std::size_t i = 0; i < count; ++i)
        {
            args.dst0[i] = destination_views[i].polys[0].shards[s].ptr;
            args.dst1[i] = destination_views[i].polys[1].shards[s].ptr;
            args.dst2[i] = destination_views[i].polys[2].shards[s].ptr;
            args.lhs0[i] = left_views[i].polys[0].shards[s].ptr;
            args.lhs1[i] = left_views[i].polys[1].shards[s].ptr;
            args.rhs0[i] = right_views[i].polys[0].shards[s].ptr;
            args.rhs1[i] = right_views[i].polys[1].shards[s].ptr;
        }
        kernel::launch_multiply_batch_two_components(
            args, count, is_square, dst0, *parameter_shard, level_info.degree);
    }
}

}  // namespace gpu
}  // namespace poseidon
