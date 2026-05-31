#include "poseidon/gpu/gpu_ntt_handler.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"

#include <stdexcept>
#include <string>

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

void validate_ciphertext_ntt_shape(
    const char *name,
    const GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &level_info)
{
    if (!(destination_view.meta.parms_id == source_view.meta.parms_id) ||
        !(destination_view.meta.parms_id == level_info.parms_id))
    {
        throw std::invalid_argument(std::string(name) + ": parms_id mismatch");
    }
    if (destination_view.meta.degree != source_view.meta.degree ||
        destination_view.meta.degree != level_info.degree ||
        destination_view.meta.q_count != source_view.meta.q_count ||
        destination_view.meta.q_count != level_info.q_count ||
        destination_view.meta.p_count != source_view.meta.p_count ||
        destination_view.meta.p_count != level_info.p_count)
    {
        throw std::invalid_argument(std::string(name) + ": shape mismatch");
    }
    if (destination_view.meta.p_count != 0)
    {
        throw std::invalid_argument(std::string(name) + ": p limbs are not supported yet");
    }
    if (destination_view.polys.size() != source_view.polys.size())
    {
        throw std::invalid_argument(std::string(name) + ": component count mismatch");
    }
}

void validate_plaintext_ntt_shape(
    const char *name,
    const GpuPlaintextView &destination_view,
    const GpuConstPlaintextView &source_view,
    const GpuLevelInfo &level_info)
{
    if (!(destination_view.meta.parms_id == source_view.meta.parms_id) ||
        !(destination_view.meta.parms_id == level_info.parms_id))
    {
        throw std::invalid_argument(std::string(name) + ": parms_id mismatch");
    }
    if (destination_view.meta.degree != source_view.meta.degree ||
        destination_view.meta.degree != level_info.degree ||
        destination_view.meta.q_count != source_view.meta.q_count ||
        destination_view.meta.q_count != level_info.q_count ||
        destination_view.meta.p_count != source_view.meta.p_count ||
        destination_view.meta.p_count != level_info.p_count)
    {
        throw std::invalid_argument(std::string(name) + ": shape mismatch");
    }
    if (destination_view.meta.p_count != 0)
    {
        throw std::invalid_argument(std::string(name) + ": p limbs are not supported yet");
    }
}

}  // namespace

GpuNTTHandler::GpuNTTHandler(const GpuParameterData &params)
    : params_(params)
{}

void GpuNTTHandler::forward_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &level_info) const
{
    validate_ciphertext_ntt_shape(
        "GpuNTTHandler::forward_ciphertext",
        destination_view,
        source_view,
        level_info);

    if (source_view.meta.is_ntt_form || !destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument(
            "GpuNTTHandler::forward_ciphertext: NTT form mismatch");
    }

    // 遍历每一个component，每个component进行一次
    for (std::size_t i = 0; i < destination_view.polys.size(); ++i)
    {
        forward_poly(destination_view.polys[i], source_view.polys[i], level_info);
    }
}

void GpuNTTHandler::inverse_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &level_info) const
{
    validate_ciphertext_ntt_shape(
        "GpuNTTHandler::inverse_ciphertext",
        destination_view,
        source_view,
        level_info);

    if (!source_view.meta.is_ntt_form || destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument(
            "GpuNTTHandler::inverse_ciphertext: NTT form mismatch");
    }

    for (std::size_t i = 0; i < destination_view.polys.size(); ++i)
    {
        inverse_poly(destination_view.polys[i], source_view.polys[i], level_info);
    }
}

void GpuNTTHandler::forward_plaintext(
    GpuPlaintextView &destination_view,
    const GpuConstPlaintextView &source_view,
    const GpuLevelInfo &level_info) const
{
    validate_plaintext_ntt_shape(
        "GpuNTTHandler::forward_plaintext",
        destination_view,
        source_view,
        level_info);

    if (source_view.meta.is_ntt_form || !destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument(
            "GpuNTTHandler::forward_plaintext: NTT form mismatch");
    }

    forward_poly(destination_view.poly, source_view.poly, level_info);
}

void GpuNTTHandler::inverse_plaintext(
    GpuPlaintextView &destination_view,
    const GpuConstPlaintextView &source_view,
    const GpuLevelInfo &level_info) const
{
    validate_plaintext_ntt_shape(
        "GpuNTTHandler::inverse_plaintext",
        destination_view,
        source_view,
        level_info);

    if (!source_view.meta.is_ntt_form || destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument(
            "GpuNTTHandler::inverse_plaintext: NTT form mismatch");
    }

    inverse_poly(destination_view.poly, source_view.poly, level_info);
}

void GpuNTTHandler::forward_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &source_poly,
    const GpuLevelInfo &level_info) const
{
    if (destination_poly.shards.size() != source_poly.shards.size())
    {
        throw std::invalid_argument("GpuNTTHandler::forward_poly: shard count mismatch");
    }

    for (std::size_t i = 0; i < destination_poly.shards.size(); ++i)
    {
        const auto &dst = destination_poly.shards[i];
        const auto &src = source_poly.shards[i];
        if (!same_shard_placement(dst, src))
        {
            throw std::invalid_argument("GpuNTTHandler::forward_poly: shard placement mismatch");
        }

        const auto *parameter_shard = find_parameter_shard(level_info, dst);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("GpuNTTHandler::forward_poly: no matching parameter shard");
        }

        kernel::launch_forward_ntt_poly_shard(
            dst,
            src,
            *parameter_shard,
            level_info.degree);
    }
}

void GpuNTTHandler::inverse_poly(
    GpuRNSPolyView &destination_poly,
    const GpuConstRNSPolyView &source_poly,
    const GpuLevelInfo &level_info) const
{
    if (destination_poly.shards.size() != source_poly.shards.size())
    {
        throw std::invalid_argument("GpuNTTHandler::inverse_poly: shard count mismatch");
    }

    for (std::size_t i = 0; i < destination_poly.shards.size(); ++i)
    {
        const auto &dst = destination_poly.shards[i];
        const auto &src = source_poly.shards[i];
        if (!same_shard_placement(dst, src))
        {
            throw std::invalid_argument("GpuNTTHandler::inverse_poly: shard placement mismatch");
        }

        const auto *parameter_shard = find_parameter_shard(level_info, dst);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument("GpuNTTHandler::inverse_poly: no matching parameter shard");
        }

        kernel::launch_inverse_ntt_poly_shard(
            dst,
            src,
            *parameter_shard,
            level_info.degree);
    }
}

}  // namespace gpu
}  // namespace poseidon
