#include "poseidon/gpu/gpu_modswitch_handler.h"

#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"
#include "poseidon/gpu/kernels/gpu_rescale_kernels.h"

#include <nvtx3/nvToolsExt.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace poseidon
{
namespace gpu
{
namespace
{

class NvtxRange
{
public:
    explicit NvtxRange(std::string name)
        : name_(std::move(name))
    {
        nvtxRangePushA(name_.c_str());
    }

    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;

    ~NvtxRange()
    {
        nvtxRangePop();
    }

private:
    std::string name_;
};

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

void validate_rescale_ciphertext_shape(
    const GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info)
{
    if (!(source_view.meta.parms_id == source_level_info.parms_id) ||
        !(destination_view.meta.parms_id == destination_level_info.parms_id))
    {
        throw std::invalid_argument("GpuModSwitchHandler::rescale_ciphertext: parms_id mismatch");
    }
    if (!source_view.meta.is_ntt_form || !destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuModSwitchHandler::rescale_ciphertext: CKKS rescale requires NTT form");
    }
    if (source_view.meta.p_count != 0 || destination_view.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuModSwitchHandler::rescale_ciphertext: p limbs are not supported yet");
    }
    if (source_view.meta.degree != destination_view.meta.degree ||
        source_view.meta.degree != source_level_info.degree ||
        destination_view.meta.degree != destination_level_info.degree)
    {
        throw std::invalid_argument("GpuModSwitchHandler::rescale_ciphertext: degree mismatch");
    }
    if (source_view.meta.q_count < 2 ||
        destination_view.meta.q_count + 1 != source_view.meta.q_count ||
        source_level_info.q_count != source_view.meta.q_count ||
        destination_level_info.q_count != destination_view.meta.q_count)
    {
        throw std::invalid_argument("GpuModSwitchHandler::rescale_ciphertext: q_count mismatch");
    }
    if (source_view.polys.size() != destination_view.polys.size() ||
        source_view.polys.size() != source_view.meta.component_count ||
        destination_view.polys.size() != destination_view.meta.component_count)
    {
        throw std::invalid_argument("GpuModSwitchHandler::rescale_ciphertext: component count mismatch");
    }
}

void validate_drop_modulus_ciphertext_shape(
    const GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info)
{
    if (!(source_view.meta.parms_id == source_level_info.parms_id) ||
        !(destination_view.meta.parms_id == destination_level_info.parms_id))
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::drop_modulus_ciphertext: parms_id mismatch");
    }
    if (source_view.meta.is_ntt_form != destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::drop_modulus_ciphertext: NTT form mismatch");
    }
    if (source_view.meta.scale != destination_view.meta.scale)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::drop_modulus_ciphertext: scale mismatch");
    }
    if (source_view.meta.p_count != 0 || destination_view.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::drop_modulus_ciphertext: p limbs are not supported yet");
    }
    if (source_view.meta.degree != destination_view.meta.degree ||
        source_view.meta.degree != source_level_info.degree ||
        destination_view.meta.degree != destination_level_info.degree)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::drop_modulus_ciphertext: degree mismatch");
    }
    if (destination_view.meta.q_count >= source_view.meta.q_count ||
        source_level_info.q_count != source_view.meta.q_count ||
        destination_level_info.q_count != destination_view.meta.q_count)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::drop_modulus_ciphertext: q_count mismatch");
    }
    if (source_view.polys.size() != destination_view.polys.size() ||
        source_view.polys.size() != source_view.meta.component_count ||
        destination_view.polys.size() != destination_view.meta.component_count)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::drop_modulus_ciphertext: component count mismatch");
    }
}

void validate_single_shard_poly(
    const char *name,
    const GpuRNSPolyView &poly,
    std::size_t q_count,
    std::size_t degree)
{
    if (poly.shards.size() != 1)
    {
        throw std::invalid_argument(std::string(name) + ": first implementation requires one shard per poly");
    }

    const auto &shard = poly.shards.front();
    if (shard.limb_begin != 0 ||
        shard.limb_count != q_count ||
        shard.coeff_begin != 0 ||
        shard.coeff_count != degree)
    {
        throw std::invalid_argument(std::string(name) + ": shard shape mismatch");
    }
}

void validate_single_shard_poly(
    const char *name,
    const GpuConstRNSPolyView &poly,
    std::size_t q_count,
    std::size_t degree)
{
    if (poly.shards.size() != 1)
    {
        throw std::invalid_argument(std::string(name) + ": first implementation requires one shard per poly");
    }

    const auto &shard = poly.shards.front();
    if (shard.limb_begin != 0 ||
        shard.limb_count != q_count ||
        shard.coeff_begin != 0 ||
        shard.coeff_count != degree)
    {
        throw std::invalid_argument(std::string(name) + ": shard shape mismatch");
    }
}

GpuConstPolyShardView make_source_limb_range(
    const GpuConstPolyShardView &source_shard,
    std::size_t limb_begin,
    std::size_t limb_count,
    std::size_t degree)
{
    if (limb_begin < source_shard.limb_begin ||
        limb_begin + limb_count >
            source_shard.limb_begin + source_shard.limb_count)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler: source shard does not cover requested limb range");
    }

    GpuConstPolyShardView result;
    result.device_id = source_shard.device_id;
    result.ptr =
        source_shard.ptr + (limb_begin - source_shard.limb_begin) * degree;
    result.limb_begin = limb_begin;
    result.limb_count = limb_count;
    result.coeff_begin = 0;
    result.coeff_count = degree;
    return result;
}

GpuConstPolyShardView make_const_shard_view(
    const GpuPolyShardView &shard)
{
    GpuConstPolyShardView result;
    result.device_id = shard.device_id;
    result.ptr = shard.ptr;
    result.limb_begin = shard.limb_begin;
    result.limb_count = shard.limb_count;
    result.coeff_begin = shard.coeff_begin;
    result.coeff_count = shard.coeff_count;
    return result;
}

/* 对component的rescale，主要包含INTT+BConv+NTT+Submult*/
void rescale_poly(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t source_q_count,
    std::size_t degree,
    GpuWord *scratch_q_last,
    GpuWord *scratch_correction,
    GpuWord *scratch_correction_ntt)
{
    const std::size_t destination_q_count = source_q_count - 1;
    const std::size_t q_last_limb = source_q_count - 1;
    const int device_id = destination_shard.device_id;

    GpuPolyShardView q_last_coeff_shard;
    q_last_coeff_shard.device_id = device_id;
    q_last_coeff_shard.ptr = scratch_q_last;
    q_last_coeff_shard.limb_begin = q_last_limb;
    q_last_coeff_shard.limb_count = 1;
    q_last_coeff_shard.coeff_begin = 0;
    q_last_coeff_shard.coeff_count = degree;

    const auto q_last_ntt_source = make_source_limb_range(
        source_shard,
        q_last_limb,
        1,
        degree);

    {
        NvtxRange range("rescale.intt_q_last");
        kernel::launch_inverse_ntt_poly_shard(
            q_last_coeff_shard,
            q_last_ntt_source,
            parameter_shard,
            degree);
    }

    GpuPolyShardView correction_shard;
    correction_shard.device_id = device_id;
    correction_shard.ptr = scratch_correction;
    correction_shard.limb_begin = 0;
    correction_shard.limb_count = destination_q_count;
    correction_shard.coeff_begin = 0;
    correction_shard.coeff_count = degree;

    GpuPolyShardView correction_ntt_shard = correction_shard;
    correction_ntt_shard.ptr = scratch_correction_ntt;

    {
        NvtxRange range("rescale.build_correction");
        kernel::launch_build_q_last_rescale_correction_poly_shard(
            correction_shard,
            make_const_shard_view(q_last_coeff_shard),
            parameter_shard,
            degree);
    }

    {
        NvtxRange range("rescale.forward_ntt_correction");
        kernel::launch_forward_ntt_poly_shard(
            correction_ntt_shard,
            make_const_shard_view(correction_shard),
            parameter_shard,
            degree);
    }

    const auto source_without_q_last = make_source_limb_range(
        source_shard,
        0,
        destination_q_count,
        degree);

    {
        NvtxRange range("rescale.apply_correction");
        kernel::launch_apply_q_last_rescale_correction_poly_shard(
            destination_shard,
            source_without_q_last,
            make_const_shard_view(correction_ntt_shard),
            parameter_shard,
            degree);
    }
}

}  // namespace

GpuModSwitchHandler::GpuModSwitchHandler(const GpuParameterData &params)
    : params_(params)
{}

GpuModSwitchHandler::~GpuModSwitchHandler()
{
    try
    {
        if (rescale_scratch_.device_id >= 0 &&
            (rescale_scratch_.q_last_capacity != 0 ||
             rescale_scratch_.correction_capacity != 0))
        {
            cudaSetDevice(rescale_scratch_.device_id);
            cudaDeviceSynchronize();
        }
    }
    catch (...)
    {}
}

void GpuModSwitchHandler::ensure_rescale_scratch(
    std::size_t degree,
    std::size_t destination_q_count,
    int device_id) const
{
    NvtxRange range("rescale.ensure_scratch");
    if (destination_q_count != 0 &&
        degree > std::numeric_limits<std::size_t>::max() / destination_q_count)
    {
        throw std::overflow_error(
            "GpuModSwitchHandler::ensure_rescale_scratch: scratch size overflow");
    }

    const std::size_t q_last_size = degree;
    const std::size_t correction_size = destination_q_count * degree;
    const bool need_reallocate =
        rescale_scratch_.device_id != device_id ||
        rescale_scratch_.q_last_capacity < q_last_size ||
        rescale_scratch_.correction_capacity < correction_size;

    if (!need_reallocate)
    {
        return;
    }

    if (rescale_scratch_.device_id >= 0 &&
        (rescale_scratch_.q_last_capacity != 0 ||
         rescale_scratch_.correction_capacity != 0))
    {
        gpu_check_cuda(
            cudaSetDevice(rescale_scratch_.device_id),
            "GpuModSwitchHandler::ensure_rescale_scratch cudaSetDevice before realloc");
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            "GpuModSwitchHandler::ensure_rescale_scratch realloc sync");
    }

    rescale_scratch_.q_last.allocate(q_last_size, device_id);
    rescale_scratch_.correction.allocate(correction_size, device_id);
    rescale_scratch_.correction_ntt.allocate(correction_size, device_id);
    rescale_scratch_.q_last_capacity = q_last_size;
    rescale_scratch_.correction_capacity = correction_size;
    rescale_scratch_.device_id = device_id;
}

void GpuModSwitchHandler::rescale_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info) const
{
    NvtxRange range("modswitch.rescale");
    validate_rescale_ciphertext_shape(
        destination_view,
        source_view,
        source_level_info,
        destination_level_info);

    const std::size_t degree = source_view.meta.degree;
    const std::size_t source_q_count = source_view.meta.q_count;
    const std::size_t destination_q_count = destination_view.meta.q_count;
    const int device_id = destination_view.polys.front().shards.front().device_id;

    ensure_rescale_scratch(
        degree,
        destination_q_count,
        device_id);

    for (std::size_t i = 0; i < source_view.polys.size(); ++i)
    {
        validate_single_shard_poly(
            "GpuModSwitchHandler::rescale_ciphertext source",
            source_view.polys[i],
            source_q_count,
            degree);
        validate_single_shard_poly(
            "GpuModSwitchHandler::rescale_ciphertext destination",
            destination_view.polys[i],
            destination_q_count,
            degree);

        const auto &destination_shard = destination_view.polys[i].shards.front();
        const auto &source_shard = source_view.polys[i].shards.front();
        if (destination_shard.device_id != source_shard.device_id)
        {
            throw std::invalid_argument(
                "GpuModSwitchHandler::rescale_ciphertext: source/destination device mismatch");
        }

        const auto *parameter_shard = find_parameter_shard(
            source_level_info,
            destination_shard);
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument(
                "GpuModSwitchHandler::rescale_ciphertext: no matching source parameter shard");
        }

        {
            NvtxRange component_range(
                "rescale.component[" + std::to_string(i) + "]");
            rescale_poly(
                destination_shard,
                source_shard,
                *parameter_shard,
                source_q_count,
                degree,
                rescale_scratch_.q_last.data(),
                rescale_scratch_.correction.data(),
                rescale_scratch_.correction_ntt.data());
        }
    }
}

void GpuModSwitchHandler::rescale_dynamic_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info,
    double min_scale) const
{
    // TODO:
    // Launch GPU dynamic rescale kernels.

    (void)destination_view;
    (void)source_view;
    (void)source_level_info;
    (void)destination_level_info;
    (void)min_scale;

    throw std::runtime_error("GpuModSwitchHandler::rescale_dynamic_ciphertext is not implemented yet");
}

void GpuModSwitchHandler::drop_modulus_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info) const
{
    NvtxRange range("modswitch.drop_modulus");
    validate_drop_modulus_ciphertext_shape(
        destination_view,
        source_view,
        source_level_info,
        destination_level_info);

    const std::size_t degree = source_view.meta.degree;
    const std::size_t source_q_count = source_view.meta.q_count;
    const std::size_t destination_q_count = destination_view.meta.q_count;

    for (std::size_t i = 0; i < source_view.polys.size(); ++i)
    {
        validate_single_shard_poly(
            "GpuModSwitchHandler::drop_modulus_ciphertext source",
            source_view.polys[i],
            source_q_count,
            degree);
        validate_single_shard_poly(
            "GpuModSwitchHandler::drop_modulus_ciphertext destination",
            destination_view.polys[i],
            destination_q_count,
            degree);

        const auto &destination_shard = destination_view.polys[i].shards.front();
        const auto &source_shard = source_view.polys[i].shards.front();
        kernel::launch_copy_poly_shard(
            destination_shard,
            make_source_limb_range(
                source_shard,
                0,
                destination_q_count,
                degree),
            degree);
    }
}

}  // namespace gpu
}  // namespace poseidon
