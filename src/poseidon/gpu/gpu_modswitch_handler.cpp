#include "poseidon/gpu/gpu_modswitch_handler.h"

#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"
#include "poseidon/gpu/kernels/gpu_rescale_kernels.h"

#include <nvtx3/nvToolsExt.h>

#include <cstdlib>
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

template <typename ShardView>
const GpuParameterShard *find_parameter_shard(
    const GpuLevelInfo &level_info,
    const ShardView &shard)
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

void validate_rescale_x2_ciphertext_shape(
    const GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info)
{
    if (!(source_view.meta.parms_id == source_level_info.parms_id) ||
        !(destination_view.meta.parms_id == destination_level_info.parms_id))
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_ciphertext_x2: parms_id mismatch");
    }
    if (!source_view.meta.is_ntt_form || !destination_view.meta.is_ntt_form ||
        source_view.meta.p_count != 0 || destination_view.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_ciphertext_x2: q-only NTT form required");
    }
    if (source_view.meta.degree != destination_view.meta.degree ||
        source_view.meta.degree != source_level_info.degree ||
        destination_view.meta.degree != destination_level_info.degree)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_ciphertext_x2: degree mismatch");
    }
    if (source_view.meta.q_count < 3 ||
        destination_view.meta.q_count + 2 != source_view.meta.q_count ||
        source_level_info.q_count != source_view.meta.q_count ||
        destination_level_info.q_count != destination_view.meta.q_count)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_ciphertext_x2: q_count mismatch");
    }
    if (source_view.polys.size() != 2 ||
        destination_view.polys.size() != 2 ||
        source_view.meta.component_count != 2 ||
        destination_view.meta.component_count != 2)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_ciphertext_x2: exactly two components required");
    }
}

void validate_rescale_many_ciphertext_shape(
    const GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info,
    std::size_t rescale_count)
{
    if (!(source_view.meta.parms_id == source_level_info.parms_id) ||
        !(destination_view.meta.parms_id == destination_level_info.parms_id))
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_ciphertext_many: parms_id mismatch");
    }
    if (!source_view.meta.is_ntt_form || !destination_view.meta.is_ntt_form ||
        source_view.meta.p_count != 0 || destination_view.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_ciphertext_many: q-only NTT form required");
    }
    if (rescale_count == 0 || rescale_count >= source_view.meta.q_count ||
        destination_view.meta.q_count + rescale_count !=
            source_view.meta.q_count ||
        source_level_info.q_count != source_view.meta.q_count ||
        destination_level_info.q_count != destination_view.meta.q_count)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_ciphertext_many: q_count mismatch");
    }
    if (source_view.meta.degree != destination_view.meta.degree ||
        source_view.meta.degree != source_level_info.degree ||
        destination_view.meta.degree != destination_level_info.degree)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_ciphertext_many: degree mismatch");
    }
    if (source_view.polys.empty() ||
        source_view.polys.size() != destination_view.polys.size() ||
        source_view.polys.size() != source_view.meta.component_count ||
        destination_view.polys.size() != destination_view.meta.component_count)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_ciphertext_many: component count mismatch");
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
        throw std::invalid_argument("GpuModSwitchHandler::drop_modulus_ciphertext: parms_id mismatch");
    }
    if (source_view.meta.p_count != 0 || destination_view.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuModSwitchHandler::drop_modulus_ciphertext: p limbs are not supported yet");
    }
    if (source_view.meta.degree != destination_view.meta.degree ||
        source_view.meta.degree != source_level_info.degree ||
        destination_view.meta.degree != destination_level_info.degree)
    {
        throw std::invalid_argument("GpuModSwitchHandler::drop_modulus_ciphertext: degree mismatch");
    }
    if (source_level_info.q_count != source_view.meta.q_count ||
        destination_level_info.q_count != destination_view.meta.q_count ||
        destination_view.meta.q_count == 0 ||
        destination_view.meta.q_count > source_view.meta.q_count)
    {
        throw std::invalid_argument("GpuModSwitchHandler::drop_modulus_ciphertext: q_count mismatch");
    }
    if (source_view.polys.size() != destination_view.polys.size() ||
        source_view.polys.size() != source_view.meta.component_count ||
        destination_view.polys.size() != destination_view.meta.component_count)
    {
        throw std::invalid_argument("GpuModSwitchHandler::drop_modulus_ciphertext: component count mismatch");
    }
}

void validate_raise_modulus_ciphertext_shape(
    const GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info)
{
    if (!(source_view.meta.parms_id == source_level_info.parms_id) ||
        !(destination_view.meta.parms_id == destination_level_info.parms_id))
    {
        throw std::invalid_argument("GpuModSwitchHandler::raise_modulus_ciphertext: parms_id mismatch");
    }
    if (source_view.meta.is_ntt_form || destination_view.meta.is_ntt_form)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::raise_modulus_ciphertext: coefficient-domain input/output required");
    }
    if (source_view.meta.p_count != 0 || destination_view.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuModSwitchHandler::raise_modulus_ciphertext: p limbs are not supported yet");
    }
    if (source_view.meta.degree != destination_view.meta.degree ||
        source_view.meta.degree != source_level_info.degree ||
        destination_view.meta.degree != destination_level_info.degree)
    {
        throw std::invalid_argument("GpuModSwitchHandler::raise_modulus_ciphertext: degree mismatch");
    }
    if (source_level_info.q_count != source_view.meta.q_count ||
        destination_level_info.q_count != destination_view.meta.q_count ||
        source_view.meta.q_count == 0 ||
        destination_view.meta.q_count < source_view.meta.q_count)
    {
        throw std::invalid_argument("GpuModSwitchHandler::raise_modulus_ciphertext: q_count mismatch");
    }
    if (source_view.polys.size() != destination_view.polys.size() ||
        source_view.polys.size() != source_view.meta.component_count ||
        destination_view.polys.size() != destination_view.meta.component_count)
    {
        throw std::invalid_argument("GpuModSwitchHandler::raise_modulus_ciphertext: component count mismatch");
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
            "GpuModSwitchHandler::rescale_ciphertext: source shard does not cover requested limb range");
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

bool is_fourstep_ntt_algo_for_rescale()
{
    const char *raw = std::getenv("POSEIDON_NTT_ALGO");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string algo(raw);
    return algo == "fourstep" || algo == "four_step" || algo == "cheddar";
}

bool can_batch_rescale_fourstep(
    std::size_t degree,
    const GpuParameterShard &parameter_shard)
{
    return degree == 65536 &&
           parameter_shard.limb_begin == 0 &&
           !parameter_shard.fourstep_intt_tables.empty() &&
           is_fourstep_ntt_algo_for_rescale();
}

void launch_correction_forward_ntt_components(
    int device_id,
    GpuWord *correction_ntt_data,
    const GpuWord *correction_data,
    std::size_t destination_q_count,
    std::size_t degree,
    std::size_t component_count,
    const GpuParameterShard &parameter_shard)
{
    const std::size_t component_words = destination_q_count * degree;
    if (component_count >= 2 &&
        can_batch_rescale_fourstep(degree, parameter_shard))
    {
        const GpuPolyShardView first_ntt{
            device_id, correction_ntt_data, 0, destination_q_count, 0, degree};
        const GpuConstPolyShardView first_coeff{
            device_id, correction_data, 0, destination_q_count, 0, degree};
        kernel::launch_forward_ntt_poly_shard_batch_fourstep_65536(
            first_ntt, first_coeff, parameter_shard, degree, component_count);
        return;
    }
    for (std::size_t component = 0; component < component_count; ++component)
    {
        const std::size_t offset = component * component_words;
        const GpuPolyShardView correction_ntt{
            device_id, correction_ntt_data + offset, 0, destination_q_count, 0, degree};
        const GpuConstPolyShardView correction_coeff{
            device_id, correction_data + offset, 0, destination_q_count, 0, degree};
        kernel::launch_forward_ntt_poly_shard(
            correction_ntt, correction_coeff, parameter_shard, degree);
    }
}

void launch_dropped_intt_ptrs_list(
    int device_id,
    GpuWord *dropped_coeff_data,
    const kernel::InntPtrsArgs &src_args,
    std::size_t dropped_begin,
    std::size_t dropped_count,
    std::size_t component_count,
    std::size_t degree,
    const GpuParameterShard &parameter_shard)
{
    if (component_count == 2 &&
        can_batch_rescale_fourstep(degree, parameter_shard))
    {
        kernel::launch_inverse_ntt_two_components_fourstep_65536(
            dropped_coeff_data,
            dropped_coeff_data + dropped_count * degree,
            src_args.sources[0],
            src_args.sources[1],
            dropped_begin,
            dropped_count,
            parameter_shard);
        return;
    }
    if (component_count >= 3 &&
        component_count <= kernel::kMaxInntPtrsBatch &&
        can_batch_rescale_fourstep(degree, parameter_shard))
    {
        kernel::launch_inverse_ntt_ptrs_fourstep_65536(
            dropped_coeff_data,
            src_args,
            dropped_begin - parameter_shard.limb_begin,
            dropped_count,
            component_count,
            parameter_shard);
        return;
    }
    for (std::size_t component = 0; component < component_count; ++component)
    {
        GpuPolyShardView dropped_coefficients;
        dropped_coefficients.device_id = device_id;
        dropped_coefficients.ptr =
            dropped_coeff_data + component * dropped_count * degree;
        dropped_coefficients.limb_begin = dropped_begin;
        dropped_coefficients.limb_count = dropped_count;
        dropped_coefficients.coeff_begin = 0;
        dropped_coefficients.coeff_count = degree;
        GpuConstPolyShardView dropped_ntt;
        dropped_ntt.device_id = device_id;
        dropped_ntt.ptr = src_args.sources[component];
        dropped_ntt.limb_begin = dropped_begin;
        dropped_ntt.limb_count = dropped_count;
        dropped_ntt.coeff_begin = 0;
        dropped_ntt.coeff_count = degree;
        kernel::launch_inverse_ntt_poly_shard(
            dropped_coefficients, dropped_ntt, parameter_shard, degree);
    }
}

void launch_dropped_intt_components(
    int device_id,
    GpuWord *dropped_coeff_data,
    const GpuConstCiphertextView &source_view,
    std::size_t dropped_begin,
    std::size_t dropped_count,
    std::size_t component_count,
    std::size_t degree,
    const GpuParameterShard &parameter_shard)
{
    if (component_count <= kernel::kMaxInntPtrsBatch)
    {
        bool covered = true;
        kernel::InntPtrsArgs src_args{};
        for (std::size_t c = 0; c < component_count; ++c)
        {
            const auto &src = source_view.polys[c].shards.front();
            if (src.limb_begin > dropped_begin ||
                src.limb_begin + src.limb_count < dropped_begin + dropped_count)
            {
                covered = false;
                break;
            }
            src_args.sources[c] = src.ptr + (dropped_begin - src.limb_begin) * degree;
        }
        if (covered)
        {
            launch_dropped_intt_ptrs_list(
                device_id, dropped_coeff_data, src_args,
                dropped_begin, dropped_count, component_count, degree, parameter_shard);
            return;
        }
    }
    for (std::size_t component = 0; component < component_count; ++component)
    {
        const auto &source_shard = source_view.polys[component].shards.front();
        GpuPolyShardView dropped_coefficients;
        dropped_coefficients.device_id = device_id;
        dropped_coefficients.ptr =
            dropped_coeff_data + component * dropped_count * degree;
        dropped_coefficients.limb_begin = dropped_begin;
        dropped_coefficients.limb_count = dropped_count;
        dropped_coefficients.coeff_begin = 0;
        dropped_coefficients.coeff_count = degree;
        const auto dropped_ntt = make_source_limb_range(
            source_shard, dropped_begin, dropped_count, degree);
        kernel::launch_inverse_ntt_poly_shard(
            dropped_coefficients, dropped_ntt, parameter_shard, degree);
    }
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
             rescale_scratch_.correction_capacity != 0 ||
             rescale_scratch_.dropped_two_capacity != 0 ||
             rescale_scratch_.dropped_many_capacity != 0 ||
             rescale_scratch_.centered_remainder_capacity != 0))
        {
            cudaSetDevice(rescale_scratch_.device_id);
            cudaDeviceSynchronize();
        }
    }
    catch (...)
    {}
}

void GpuModSwitchHandler::ensure_rescale_many_scratch(
    std::size_t degree,
    std::size_t destination_q_count,
    std::size_t component_count,
    std::size_t rescale_count,
    int device_id) const
{
    NvtxRange range("rescale_many.ensure_scratch");
    if (component_count == 0 || rescale_count == 0 ||
        component_count > std::numeric_limits<std::size_t>::max() / degree ||
        component_count * degree >
            std::numeric_limits<std::size_t>::max() / rescale_count ||
        destination_q_count >
            std::numeric_limits<std::size_t>::max() / (component_count * degree))
    {
        throw std::overflow_error(
            "GpuModSwitchHandler::ensure_rescale_many_scratch: size overflow");
    }
    const std::size_t dropped_size =
        component_count * rescale_count * degree;
    const std::size_t correction_size =
        component_count * destination_q_count * degree;
    const bool need_reallocate =
        rescale_scratch_.device_id != device_id ||
        rescale_scratch_.dropped_many_capacity < dropped_size ||
        rescale_scratch_.correction_capacity < correction_size;
    if (!need_reallocate)
    {
        return;
    }
    if (rescale_scratch_.device_id >= 0)
    {
        gpu_check_cuda(
            cudaSetDevice(rescale_scratch_.device_id),
            "GpuModSwitchHandler::ensure_rescale_many_scratch cudaSetDevice");
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            "GpuModSwitchHandler::ensure_rescale_many_scratch realloc sync");
    }
    rescale_scratch_.dropped_many.allocate(dropped_size, device_id);
    rescale_scratch_.correction.allocate(correction_size, device_id);
    rescale_scratch_.correction_ntt.allocate(correction_size, device_id);
    rescale_scratch_.dropped_many_capacity = dropped_size;
    rescale_scratch_.correction_capacity = correction_size;
    rescale_scratch_.device_id = device_id;
}

void GpuModSwitchHandler::ensure_rescale_x2_scratch(
    std::size_t degree,
    std::size_t destination_q_count,
    std::size_t component_count,
    int device_id) const
{
    NvtxRange range("rescale_x2.ensure_scratch");
    if (component_count == 0 ||
        destination_q_count >
            std::numeric_limits<std::size_t>::max() / component_count ||
        component_count * destination_q_count >
            std::numeric_limits<std::size_t>::max() / degree ||
        component_count >
            std::numeric_limits<std::size_t>::max() / (2 * degree))
    {
        throw std::overflow_error(
            "GpuModSwitchHandler::ensure_rescale_x2_scratch: size overflow");
    }

    const std::size_t dropped_two_size = component_count * 2 * degree;
    const std::size_t correction_size =
        component_count * destination_q_count * degree;
    const bool need_reallocate =
        rescale_scratch_.device_id != device_id ||
        rescale_scratch_.dropped_two_capacity < dropped_two_size ||
        rescale_scratch_.correction_capacity < correction_size;
    if (!need_reallocate)
    {
        return;
    }

    if (rescale_scratch_.device_id >= 0)
    {
        gpu_check_cuda(
            cudaSetDevice(rescale_scratch_.device_id),
            "GpuModSwitchHandler::ensure_rescale_x2_scratch cudaSetDevice");
        gpu_check_cuda(
            cudaDeviceSynchronize(),
            "GpuModSwitchHandler::ensure_rescale_x2_scratch realloc sync");
    }

    rescale_scratch_.dropped_two.allocate(dropped_two_size, device_id);
    rescale_scratch_.correction.allocate(correction_size, device_id);
    rescale_scratch_.correction_ntt.allocate(correction_size, device_id);
    rescale_scratch_.dropped_two_capacity = dropped_two_size;
    rescale_scratch_.centered_remainder = DeviceVector<GpuWide>{};
    rescale_scratch_.centered_remainder_capacity = 0;
    rescale_scratch_.correction_capacity = correction_size;
    rescale_scratch_.device_id = device_id;
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

    const std::size_t component_count = source_view.polys.size();
    for (std::size_t i = 0; i < component_count; ++i)
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
    }

    if (component_count == 2)
    {
        const auto &dst0 = destination_view.polys[0].shards.front();
        const auto &dst1 = destination_view.polys[1].shards.front();
        const auto &src0 = source_view.polys[0].shards.front();
        const auto &src1 = source_view.polys[1].shards.front();
        const auto *parameter_shard =
            find_parameter_shard(source_level_info, dst0);
        if (parameter_shard != nullptr &&
            dst0.device_id == device_id &&
            dst1.device_id == device_id &&
            dst0.limb_begin == 0 && dst1.limb_begin == 0 &&
            src0.limb_begin == 0 && src1.limb_begin == 0 &&
            can_batch_rescale_fourstep(degree, *parameter_shard))
        {
            ensure_rescale_x2_scratch(
                degree,
                destination_q_count,
                2,
                device_id);
            launch_dropped_intt_components(
                device_id,
                rescale_scratch_.dropped_two.data(),
                source_view,
                source_q_count - 1,
                1,
                2,
                degree,
                *parameter_shard);
            kernel::launch_build_q_last_rescale_correction_batch2(
                rescale_scratch_.correction.data(),
                rescale_scratch_.dropped_two.data(),
                destination_q_count,
                *parameter_shard,
                degree);
            launch_correction_forward_ntt_components(
                device_id,
                rescale_scratch_.correction_ntt.data(),
                rescale_scratch_.correction.data(),
                destination_q_count,
                degree,
                2,
                *parameter_shard);
            const auto src0_prefix = make_source_limb_range(
                src0, 0, destination_q_count, degree);
            const auto src1_prefix = make_source_limb_range(
                src1, 0, destination_q_count, degree);
            kernel::launch_apply_q_last_rescale_correction_batch2(
                dst0,
                dst1,
                src0_prefix,
                src1_prefix,
                rescale_scratch_.correction_ntt.data(),
                *parameter_shard,
                degree);
            return;
        }
    }

    ensure_rescale_scratch(
        degree,
        destination_q_count,
        device_id);

    for (std::size_t i = 0; i < component_count; ++i)
    {
        const auto &destination_shard = destination_view.polys[i].shards.front();
        const auto &source_shard = source_view.polys[i].shards.front();

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

bool GpuModSwitchHandler::rescale_q_batch_to_outputs(
    const GpuWord *source_batch_q,
    std::size_t batch_count,
    std::size_t source_q_count,
    const std::vector<GpuCiphertextView> &destination_views,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info) const
{
    if (batch_count == 0)
    {
        return true;
    }
    const std::size_t component_count = batch_count * 2;
    if (source_batch_q == nullptr ||
        destination_views.size() != batch_count ||
        component_count > kernel::kMaxRescaleBatchComponents ||
        source_q_count < 2 ||
        destination_level_info.q_count + 1 != source_q_count ||
        source_level_info.q_count != source_q_count)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_q_batch_to_outputs: invalid batch shape");
    }

    const std::size_t degree = source_level_info.degree;
    const std::size_t destination_q_count = destination_level_info.q_count;
    const auto &first_dst_shard =
        destination_views[0].polys.at(0).shards.front();
    const int device_id = first_dst_shard.device_id;
    const auto *parameter_shard =
        find_parameter_shard(source_level_info, first_dst_shard);
    if (parameter_shard == nullptr ||
        !can_batch_rescale_fourstep(degree, *parameter_shard))
    {
        return false;
    }

    kernel::ApplyQlastRescaleBatchArgs dst_args{};
    for (std::size_t b = 0; b < batch_count; ++b)
    {
        const auto &view = destination_views[b];
        if (view.polys.size() != 2 || !view.meta.is_ntt_form ||
            view.meta.p_count != 0 || view.meta.q_count != destination_q_count ||
            view.meta.degree != degree)
        {
            throw std::invalid_argument(
                "GpuModSwitchHandler::rescale_q_batch_to_outputs: destination metadata mismatch");
        }
        for (std::size_t c = 0; c < 2; ++c)
        {
            validate_single_shard_poly(
                "GpuModSwitchHandler::rescale_q_batch_to_outputs",
                view.polys[c],
                destination_q_count,
                degree);
            const auto &shard = view.polys[c].shards.front();
            if (shard.device_id != device_id || shard.limb_begin != 0)
            {
                throw std::invalid_argument(
                    "GpuModSwitchHandler::rescale_q_batch_to_outputs: shard device/offset mismatch");
            }
            dst_args.destinations[b * 2 + c] = shard.ptr;
        }
    }

    ensure_rescale_x2_scratch(degree, destination_q_count, component_count, device_id);
    const std::size_t dropped_limb = source_q_count - 1;
    const std::size_t source_component_stride = source_q_count * degree;

    const GpuPolyShardView first_dropped_coeff{
        device_id, rescale_scratch_.dropped_two.data(),
        dropped_limb, 1, 0, degree};
    const GpuConstPolyShardView first_dropped_ntt{
        device_id, source_batch_q + dropped_limb * degree,
        dropped_limb, 1, 0, degree};
    kernel::launch_inverse_ntt_poly_shard_batch_fourstep_65536(
        first_dropped_coeff,
        first_dropped_ntt,
        *parameter_shard,
        degree,
        component_count,
        degree,
        source_component_stride);

    kernel::launch_build_q_last_rescale_correction_batch(
        rescale_scratch_.correction.data(),
        rescale_scratch_.dropped_two.data(),
        component_count,
        destination_q_count,
        *parameter_shard,
        degree);

    const GpuPolyShardView corr_ntt{
        device_id, rescale_scratch_.correction_ntt.data(),
        0, destination_q_count, 0, degree};
    const GpuConstPolyShardView corr_coeff{
        device_id, rescale_scratch_.correction.data(),
        0, destination_q_count, 0, degree};
    kernel::launch_forward_ntt_poly_shard_batch_fourstep_65536(
        corr_ntt,
        corr_coeff,
        *parameter_shard,
        degree,
        component_count);

    kernel::launch_apply_q_last_rescale_correction_batch_ptrs(
        dst_args,
        source_batch_q,
        source_component_stride,
        rescale_scratch_.correction_ntt.data(),
        component_count,
        destination_q_count,
        *parameter_shard,
        degree);
    return true;
}

void GpuModSwitchHandler::rescale_ciphertext_x2(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info) const
{
    NvtxRange range("modswitch.rescale_x2");
    validate_rescale_x2_ciphertext_shape(
        destination_view,
        source_view,
        source_level_info,
        destination_level_info);

    const std::size_t degree = source_view.meta.degree;
    const std::size_t source_q_count = source_view.meta.q_count;
    const std::size_t destination_q_count = destination_view.meta.q_count;
    const std::size_t component_count = 2;
    const int device_id =
        destination_view.polys.front().shards.front().device_id;
    ensure_rescale_x2_scratch(
        degree,
        destination_q_count,
        component_count,
        device_id);

    const GpuParameterShard *parameter_shard = nullptr;
    for (std::size_t component = 0;
         component < component_count;
         ++component)
    {
        validate_single_shard_poly(
            "GpuModSwitchHandler::rescale_ciphertext_x2 source",
            source_view.polys[component],
            source_q_count,
            degree);
        validate_single_shard_poly(
            "GpuModSwitchHandler::rescale_ciphertext_x2 destination",
            destination_view.polys[component],
            destination_q_count,
            degree);
        const auto &source_shard =
            source_view.polys[component].shards.front();
        const auto &destination_shard =
            destination_view.polys[component].shards.front();
        if (source_shard.device_id != destination_shard.device_id ||
            source_shard.device_id != device_id)
        {
            throw std::invalid_argument(
                "GpuModSwitchHandler::rescale_ciphertext_x2: device mismatch");
        }
        if (parameter_shard == nullptr)
        {
            parameter_shard =
                find_parameter_shard(source_level_info, destination_shard);
            if (parameter_shard == nullptr)
            {
                throw std::invalid_argument(
                    "GpuModSwitchHandler::rescale_ciphertext_x2: no matching parameter shard");
            }
        }
    }
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_ciphertext_x2: no parameter shard");
    }

    launch_dropped_intt_components(
        device_id,
        rescale_scratch_.dropped_two.data(),
        source_view,
        source_q_count - 2,
        2,
        component_count,
        degree,
        *parameter_shard);

    kernel::launch_build_q_last_two_rescale_correction_batch_fused(
        rescale_scratch_.correction.data(),
        rescale_scratch_.dropped_two.data(),
        component_count,
        destination_q_count,
        *parameter_shard,
        degree);

    launch_correction_forward_ntt_components(
        device_id,
        rescale_scratch_.correction_ntt.data(),
        rescale_scratch_.correction.data(),
        destination_q_count,
        degree,
        component_count,
        *parameter_shard);

    const auto source0 = make_source_limb_range(
        source_view.polys[0].shards.front(),
        0,
        destination_q_count,
        degree);
    const auto source1 = make_source_limb_range(
        source_view.polys[1].shards.front(),
        0,
        destination_q_count,
        degree);
    kernel::launch_apply_q_last_two_rescale_correction_batch2(
        destination_view.polys[0].shards.front(),
        destination_view.polys[1].shards.front(),
        source0,
        source1,
        rescale_scratch_.correction_ntt.data(),
        *parameter_shard,
        degree);
}

void GpuModSwitchHandler::rescale_ciphertext_many(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info,
    std::size_t rescale_count) const
{
    NvtxRange range("modswitch.rescale_many");
    validate_rescale_many_ciphertext_shape(
        destination_view,
        source_view,
        source_level_info,
        destination_level_info,
        rescale_count);

    const std::size_t degree = source_view.meta.degree;
    const std::size_t source_q_count = source_view.meta.q_count;
    const std::size_t destination_q_count = destination_view.meta.q_count;
    const std::size_t component_count = source_view.polys.size();
    const int device_id =
        destination_view.polys.front().shards.front().device_id;
    ensure_rescale_many_scratch(
        degree,
        destination_q_count,
        component_count,
        rescale_count,
        device_id);

    const GpuParameterShard *parameter_shard = nullptr;
    for (std::size_t component = 0;
         component < component_count;
         ++component)
    {
        validate_single_shard_poly(
            "GpuModSwitchHandler::rescale_ciphertext_many source",
            source_view.polys[component],
            source_q_count,
            degree);
        validate_single_shard_poly(
            "GpuModSwitchHandler::rescale_ciphertext_many destination",
            destination_view.polys[component],
            destination_q_count,
            degree);
        const auto &source_shard =
            source_view.polys[component].shards.front();
        const auto &destination_shard =
            destination_view.polys[component].shards.front();
        if (source_shard.device_id != destination_shard.device_id ||
            source_shard.device_id != device_id)
        {
            throw std::invalid_argument(
                "GpuModSwitchHandler::rescale_ciphertext_many: device mismatch");
        }
        if (parameter_shard == nullptr)
        {
            parameter_shard =
                find_parameter_shard(source_level_info, destination_shard);
            if (parameter_shard == nullptr)
            {
                throw std::invalid_argument(
                    "GpuModSwitchHandler::rescale_ciphertext_many: no matching parameter shard");
            }
        }
    }
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "GpuModSwitchHandler::rescale_ciphertext_many: no parameter shard");
    }

    launch_dropped_intt_components(
        device_id,
        rescale_scratch_.dropped_many.data(),
        source_view,
        destination_q_count,
        rescale_count,
        component_count,
        degree,
        *parameter_shard);

    kernel::launch_convert_rescale_dropped_to_mixed_radix(
        rescale_scratch_.dropped_many.data(),
        component_count,
        source_q_count,
        rescale_count,
        *parameter_shard,
        degree);
    kernel::launch_build_rescale_many_correction_batch(
        rescale_scratch_.correction.data(),
        rescale_scratch_.dropped_many.data(),
        component_count,
        source_q_count,
        rescale_count,
        *parameter_shard,
        degree);

    launch_correction_forward_ntt_components(
        device_id,
        rescale_scratch_.correction_ntt.data(),
        rescale_scratch_.correction.data(),
        destination_q_count,
        degree,
        component_count,
        *parameter_shard);

    if (component_count >= 2 &&
        component_count <= kernel::kMaxRescaleBatchComponents &&
        can_batch_rescale_fourstep(degree, *parameter_shard))
    {
        bool same_limb_begin = true;
        kernel::ApplyRescalePtrsArgs apply_args{};
        for (std::size_t c = 0; c < component_count; ++c)
        {
            const auto &src = source_view.polys[c].shards.front();
            const auto &dst = destination_view.polys[c].shards.front();
            if (src.limb_begin != 0 || dst.limb_begin != 0)
            {
                same_limb_begin = false;
                break;
            }
            apply_args.sources[c] = src.ptr;
            apply_args.destinations[c] = dst.ptr;
        }
        if (same_limb_begin)
        {
            kernel::launch_apply_rescale_many_correction_ptrs(
                apply_args,
                rescale_scratch_.correction_ntt.data(),
                component_count,
                source_q_count,
                rescale_count,
                *parameter_shard,
                degree);
            return;
        }
    }

    for (std::size_t component = 0;
         component < component_count;
         ++component)
    {
        const std::size_t correction_offset =
            component * destination_q_count * degree;
        GpuPolyShardView correction_ntt;
        correction_ntt.device_id = device_id;
        correction_ntt.ptr =
            rescale_scratch_.correction_ntt.data() + correction_offset;
        correction_ntt.limb_begin = 0;
        correction_ntt.limb_count = destination_q_count;
        correction_ntt.coeff_begin = 0;
        correction_ntt.coeff_count = degree;

        const auto retained_source = make_source_limb_range(
            source_view.polys[component].shards.front(),
            0,
            destination_q_count,
            degree);
        kernel::launch_apply_rescale_many_correction_poly_shard(
            destination_view.polys[component].shards.front(),
            retained_source,
            make_const_shard_view(correction_ntt),
            source_q_count,
            rescale_count,
            *parameter_shard,
            degree);
    }
}

bool GpuModSwitchHandler::rescale_ciphertext_pair(
    GpuCiphertextView &destination_a_view,
    GpuCiphertextView &destination_b_view,
    const GpuConstCiphertextView &source_a_view,
    const GpuConstCiphertextView &source_b_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info) const
{
    validate_rescale_ciphertext_shape(
        destination_a_view, source_a_view, source_level_info, destination_level_info);
    validate_rescale_ciphertext_shape(
        destination_b_view, source_b_view, source_level_info, destination_level_info);
    const std::size_t per_ct = source_a_view.polys.size();
    const std::size_t total_comp = per_ct * 2;
    if (per_ct != source_b_view.polys.size() ||
        total_comp > kernel::kMaxInntPtrsBatch ||
        total_comp > kernel::kMaxRescaleBatchComponents)
    {
        return false;
    }
    const std::size_t degree = source_a_view.meta.degree;
    const std::size_t source_q_count = source_a_view.meta.q_count;
    const std::size_t destination_q_count = destination_a_view.meta.q_count;
    const int device_id = destination_a_view.polys[0].shards.front().device_id;
    const auto *parameter_shard =
        find_parameter_shard(source_level_info, destination_a_view.polys[0].shards.front());
    if (parameter_shard == nullptr || !can_batch_rescale_fourstep(degree, *parameter_shard))
    {
        return false;
    }

    kernel::InntPtrsArgs intt_args{};
    kernel::ApplyRescalePtrsArgs apply_args{};
    for (int lane = 0; lane < 2; ++lane)
    {
        const auto &src_v = lane == 0 ? source_a_view : source_b_view;
        const auto &dst_v = lane == 0 ? destination_a_view : destination_b_view;
        for (std::size_t c = 0; c < per_ct; ++c)
        {
            validate_single_shard_poly("rescale_ciphertext_pair src", src_v.polys[c], source_q_count, degree);
            validate_single_shard_poly("rescale_ciphertext_pair dst", dst_v.polys[c], destination_q_count, degree);
            const auto &s = src_v.polys[c].shards.front();
            const auto &d = dst_v.polys[c].shards.front();
            if (s.device_id != device_id || d.device_id != device_id ||
                s.limb_begin != 0 || d.limb_begin != 0)
            {
                return false;
            }
            const std::size_t idx = static_cast<std::size_t>(lane) * per_ct + c;
            intt_args.sources[idx] = s.ptr + (source_q_count - 1) * degree;
            apply_args.sources[idx] = s.ptr;
            apply_args.destinations[idx] = d.ptr;
        }
    }

    ensure_rescale_x2_scratch(degree, destination_q_count, total_comp, device_id);
    launch_dropped_intt_ptrs_list(
        device_id, rescale_scratch_.dropped_two.data(), intt_args,
        source_q_count - 1, 1, total_comp, degree, *parameter_shard);
    kernel::launch_build_q_last_rescale_correction_batch(
        rescale_scratch_.correction.data(),
        rescale_scratch_.dropped_two.data(),
        total_comp, destination_q_count, *parameter_shard, degree);
    launch_correction_forward_ntt_components(
        device_id,
        rescale_scratch_.correction_ntt.data(),
        rescale_scratch_.correction.data(),
        destination_q_count, degree, total_comp, *parameter_shard);
    kernel::launch_apply_q_last_rescale_correction_ptrs(
        apply_args, rescale_scratch_.correction_ntt.data(),
        total_comp, destination_q_count, *parameter_shard, degree);
    return true;
}

bool GpuModSwitchHandler::rescale_ciphertext_x2_pair(
    GpuCiphertextView &destination_a_view,
    GpuCiphertextView &destination_b_view,
    const GpuConstCiphertextView &source_a_view,
    const GpuConstCiphertextView &source_b_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info) const
{
    validate_rescale_x2_ciphertext_shape(
        destination_a_view, source_a_view, source_level_info, destination_level_info);
    validate_rescale_x2_ciphertext_shape(
        destination_b_view, source_b_view, source_level_info, destination_level_info);
    const std::size_t degree = source_a_view.meta.degree;
    const std::size_t source_q_count = source_a_view.meta.q_count;
    const std::size_t destination_q_count = destination_a_view.meta.q_count;
    const int device_id = destination_a_view.polys[0].shards.front().device_id;
    const auto *parameter_shard =
        find_parameter_shard(source_level_info, destination_a_view.polys[0].shards.front());
    if (parameter_shard == nullptr || !can_batch_rescale_fourstep(degree, *parameter_shard))
    {
        return false;
    }

    kernel::InntPtrsArgs intt_args{};
    kernel::ApplyRescalePtrsArgs apply_args{};
    for (int lane = 0; lane < 2; ++lane)
    {
        const auto &src_v = lane == 0 ? source_a_view : source_b_view;
        const auto &dst_v = lane == 0 ? destination_a_view : destination_b_view;
        for (std::size_t c = 0; c < 2; ++c)
        {
            validate_single_shard_poly("rescale_ciphertext_x2_pair src", src_v.polys[c], source_q_count, degree);
            validate_single_shard_poly("rescale_ciphertext_x2_pair dst", dst_v.polys[c], destination_q_count, degree);
            const auto &s = src_v.polys[c].shards.front();
            const auto &d = dst_v.polys[c].shards.front();
            if (s.device_id != device_id || d.device_id != device_id ||
                s.limb_begin != 0 || d.limb_begin != 0)
            {
                return false;
            }
            const std::size_t idx = static_cast<std::size_t>(lane) * 2 + c;
            intt_args.sources[idx] = s.ptr + (source_q_count - 2) * degree;
            apply_args.sources[idx] = s.ptr;
            apply_args.destinations[idx] = d.ptr;
        }
    }

    constexpr std::size_t total_comp = 4;
    ensure_rescale_x2_scratch(degree, destination_q_count, total_comp, device_id);
    launch_dropped_intt_ptrs_list(
        device_id, rescale_scratch_.dropped_two.data(), intt_args,
        source_q_count - 2, 2, total_comp, degree, *parameter_shard);
    kernel::launch_build_q_last_two_rescale_correction_batch_fused(
        rescale_scratch_.correction.data(),
        rescale_scratch_.dropped_two.data(),
        total_comp, destination_q_count, *parameter_shard, degree);
    launch_correction_forward_ntt_components(
        device_id,
        rescale_scratch_.correction_ntt.data(),
        rescale_scratch_.correction.data(),
        destination_q_count, degree, total_comp, *parameter_shard);
    kernel::launch_apply_q_last_two_rescale_correction_ptrs(
        apply_args, rescale_scratch_.correction_ntt.data(),
        total_comp, destination_q_count, *parameter_shard, degree);
    return true;
}

bool GpuModSwitchHandler::rescale_ciphertext_many_pair(
    GpuCiphertextView &destination_a_view,
    GpuCiphertextView &destination_b_view,
    const GpuConstCiphertextView &source_a_view,
    const GpuConstCiphertextView &source_b_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info,
    std::size_t rescale_count) const
{
    if (rescale_count == 1)
    {
        return rescale_ciphertext_pair(
            destination_a_view, destination_b_view,
            source_a_view, source_b_view,
            source_level_info, destination_level_info);
    }
    if (rescale_count == 2 &&
        source_a_view.polys.size() == 2 &&
        source_b_view.polys.size() == 2 &&
        source_a_view.meta.component_count == 2 &&
        source_b_view.meta.component_count == 2)
    {
        return rescale_ciphertext_x2_pair(
            destination_a_view, destination_b_view,
            source_a_view, source_b_view,
            source_level_info, destination_level_info);
    }
    validate_rescale_many_ciphertext_shape(
        destination_a_view, source_a_view, source_level_info, destination_level_info, rescale_count);
    validate_rescale_many_ciphertext_shape(
        destination_b_view, source_b_view, source_level_info, destination_level_info, rescale_count);
    const std::size_t per_ct = source_a_view.polys.size();
    const std::size_t total_comp = per_ct * 2;
    if (per_ct != source_b_view.polys.size() ||
        total_comp > kernel::kMaxInntPtrsBatch ||
        total_comp > kernel::kMaxRescaleBatchComponents)
    {
        return false;
    }
    const std::size_t degree = source_a_view.meta.degree;
    const std::size_t source_q_count = source_a_view.meta.q_count;
    const std::size_t destination_q_count = destination_a_view.meta.q_count;
    const int device_id = destination_a_view.polys[0].shards.front().device_id;
    const auto *parameter_shard =
        find_parameter_shard(source_level_info, destination_a_view.polys[0].shards.front());
    if (parameter_shard == nullptr || !can_batch_rescale_fourstep(degree, *parameter_shard))
    {
        return false;
    }

    kernel::InntPtrsArgs intt_args{};
    kernel::ApplyRescalePtrsArgs apply_args{};
    for (int lane = 0; lane < 2; ++lane)
    {
        const auto &src_v = lane == 0 ? source_a_view : source_b_view;
        const auto &dst_v = lane == 0 ? destination_a_view : destination_b_view;
        for (std::size_t c = 0; c < per_ct; ++c)
        {
            validate_single_shard_poly("rescale_ciphertext_many_pair src", src_v.polys[c], source_q_count, degree);
            validate_single_shard_poly("rescale_ciphertext_many_pair dst", dst_v.polys[c], destination_q_count, degree);
            const auto &s = src_v.polys[c].shards.front();
            const auto &d = dst_v.polys[c].shards.front();
            if (s.device_id != device_id || d.device_id != device_id ||
                s.limb_begin != 0 || d.limb_begin != 0)
            {
                return false;
            }
            const std::size_t idx = static_cast<std::size_t>(lane) * per_ct + c;
            intt_args.sources[idx] = s.ptr + destination_q_count * degree;
            apply_args.sources[idx] = s.ptr;
            apply_args.destinations[idx] = d.ptr;
        }
    }

    ensure_rescale_many_scratch(
        degree, destination_q_count, total_comp, rescale_count, device_id);
    launch_dropped_intt_ptrs_list(
        device_id, rescale_scratch_.dropped_many.data(), intt_args,
        destination_q_count, rescale_count, total_comp, degree, *parameter_shard);
    kernel::launch_convert_rescale_dropped_to_mixed_radix(
        rescale_scratch_.dropped_many.data(),
        total_comp, source_q_count, rescale_count, *parameter_shard, degree);
    kernel::launch_build_rescale_many_correction_batch(
        rescale_scratch_.correction.data(),
        rescale_scratch_.dropped_many.data(),
        total_comp, source_q_count, rescale_count, *parameter_shard, degree);
    launch_correction_forward_ntt_components(
        device_id,
        rescale_scratch_.correction_ntt.data(),
        rescale_scratch_.correction.data(),
        destination_q_count, degree, total_comp, *parameter_shard);
    kernel::launch_apply_rescale_many_correction_ptrs(
        apply_args, rescale_scratch_.correction_ntt.data(),
        total_comp, source_q_count, rescale_count, *parameter_shard, degree);
    return true;
}

bool GpuModSwitchHandler::rescale_ciphertext_many_batch(
    std::vector<GpuCiphertextView> &destination_views,
    const std::vector<GpuConstCiphertextView> &source_views,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info,
    std::size_t rescale_count) const
{
    const std::size_t batch_count = source_views.size();
    if (batch_count == 0 || destination_views.size() != batch_count || rescale_count == 0)
    {
        return false;
    }
    const std::size_t per_ct = source_views[0].polys.size();
    const std::size_t total_comp = per_ct * batch_count;
    if (per_ct == 0 ||
        total_comp > kernel::kMaxInntPtrsBatch ||
        total_comp > kernel::kMaxRescaleBatchComponents)
    {
        return false;
    }
    const std::size_t degree = source_views[0].meta.degree;
    const std::size_t source_q_count = source_views[0].meta.q_count;
    const std::size_t destination_q_count = destination_views[0].meta.q_count;
    if (source_q_count <= rescale_count ||
        destination_q_count + rescale_count != source_q_count)
    {
        return false;
    }
    const auto &first_dst = destination_views[0].polys[0].shards.front();
    const int device_id = first_dst.device_id;
    const auto *parameter_shard = find_parameter_shard(source_level_info, first_dst);
    if (parameter_shard == nullptr ||
        !can_batch_rescale_fourstep(degree, *parameter_shard))
    {
        return false;
    }

    kernel::InntPtrsArgs intt_args{};
    kernel::ApplyRescalePtrsArgs apply_args{};
    for (std::size_t i = 0; i < batch_count; ++i)
    {
        const auto &src_v = source_views[i];
        auto &dst_v = destination_views[i];
        if (src_v.polys.size() != per_ct || dst_v.polys.size() != per_ct)
        {
            return false;
        }
        validate_rescale_many_ciphertext_shape(
            dst_v, src_v, source_level_info, destination_level_info, rescale_count);
        for (std::size_t c = 0; c < per_ct; ++c)
        {
            validate_single_shard_poly(
                "rescale_ciphertext_many_batch src", src_v.polys[c], source_q_count, degree);
            validate_single_shard_poly(
                "rescale_ciphertext_many_batch dst", dst_v.polys[c], destination_q_count, degree);
            const auto &s = src_v.polys[c].shards.front();
            const auto &d = dst_v.polys[c].shards.front();
            if (s.device_id != device_id || d.device_id != device_id ||
                s.limb_begin != 0 || d.limb_begin != 0)
            {
                return false;
            }
            const std::size_t idx = i * per_ct + c;
            intt_args.sources[idx] = s.ptr + destination_q_count * degree;
            apply_args.sources[idx] = s.ptr;
            apply_args.destinations[idx] = d.ptr;
        }
    }

    if (rescale_count == 1)
    {
        ensure_rescale_x2_scratch(degree, destination_q_count, total_comp, device_id);
        launch_dropped_intt_ptrs_list(
            device_id, rescale_scratch_.dropped_two.data(), intt_args,
            source_q_count - 1, 1, total_comp, degree, *parameter_shard);
        kernel::launch_build_q_last_rescale_correction_batch(
            rescale_scratch_.correction.data(),
            rescale_scratch_.dropped_two.data(),
            total_comp, destination_q_count, *parameter_shard, degree);
        launch_correction_forward_ntt_components(
            device_id,
            rescale_scratch_.correction_ntt.data(),
            rescale_scratch_.correction.data(),
            destination_q_count, degree, total_comp, *parameter_shard);
        kernel::launch_apply_q_last_rescale_correction_ptrs(
            apply_args, rescale_scratch_.correction_ntt.data(),
            total_comp, destination_q_count, *parameter_shard, degree);
        return true;
    }

    if (rescale_count == 2 && parameter_shard->q_last_two_product != 0)
    {
        ensure_rescale_x2_scratch(degree, destination_q_count, total_comp, device_id);
        launch_dropped_intt_ptrs_list(
            device_id, rescale_scratch_.dropped_two.data(), intt_args,
            source_q_count - 2, 2, total_comp, degree, *parameter_shard);
        kernel::launch_build_q_last_two_rescale_correction_batch_fused(
            rescale_scratch_.correction.data(),
            rescale_scratch_.dropped_two.data(),
            total_comp, destination_q_count, *parameter_shard, degree);
        launch_correction_forward_ntt_components(
            device_id,
            rescale_scratch_.correction_ntt.data(),
            rescale_scratch_.correction.data(),
            destination_q_count, degree, total_comp, *parameter_shard);
        kernel::launch_apply_q_last_two_rescale_correction_ptrs(
            apply_args, rescale_scratch_.correction_ntt.data(),
            total_comp, destination_q_count, *parameter_shard, degree);
        return true;
    }

    ensure_rescale_many_scratch(
        degree, destination_q_count, total_comp, rescale_count, device_id);
    launch_dropped_intt_ptrs_list(
        device_id, rescale_scratch_.dropped_many.data(), intt_args,
        destination_q_count, rescale_count, total_comp, degree, *parameter_shard);
    kernel::launch_convert_rescale_dropped_to_mixed_radix(
        rescale_scratch_.dropped_many.data(),
        total_comp, source_q_count, rescale_count, *parameter_shard, degree);
    kernel::launch_build_rescale_many_correction_batch(
        rescale_scratch_.correction.data(),
        rescale_scratch_.dropped_many.data(),
        total_comp, source_q_count, rescale_count, *parameter_shard, degree);
    launch_correction_forward_ntt_components(
        device_id,
        rescale_scratch_.correction_ntt.data(),
        rescale_scratch_.correction.data(),
        destination_q_count, degree, total_comp, *parameter_shard);
    kernel::launch_apply_rescale_many_correction_ptrs(
        apply_args, rescale_scratch_.correction_ntt.data(),
        total_comp, source_q_count, rescale_count, *parameter_shard, degree);
    return true;
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
        if (destination_shard.device_id != source_shard.device_id)
        {
            throw std::invalid_argument(
                "GpuModSwitchHandler::drop_modulus_ciphertext: source/destination device mismatch");
        }

        const auto source_prefix = make_source_limb_range(
            source_shard,
            0,
            destination_q_count,
            degree);

        NvtxRange component_range(
            "drop_modulus.component[" + std::to_string(i) + "]");
        kernel::launch_copy_poly_shard(
            destination_shard,
            source_prefix,
            degree);
    }
}

void GpuModSwitchHandler::raise_modulus_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info) const
{
    NvtxRange range("modswitch.raise_modulus");
    validate_raise_modulus_ciphertext_shape(
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
            "GpuModSwitchHandler::raise_modulus_ciphertext source",
            source_view.polys[i],
            source_q_count,
            degree);
        validate_single_shard_poly(
            "GpuModSwitchHandler::raise_modulus_ciphertext destination",
            destination_view.polys[i],
            destination_q_count,
            degree);

        const auto &destination_shard = destination_view.polys[i].shards.front();
        const auto &source_shard = source_view.polys[i].shards.front();
        if (destination_shard.device_id != source_shard.device_id)
        {
            throw std::invalid_argument(
                "GpuModSwitchHandler::raise_modulus_ciphertext: source/destination device mismatch");
        }

        const auto *source_parameter_shard =
            find_parameter_shard(source_level_info, source_shard);
        if (source_parameter_shard == nullptr)
        {
            throw std::invalid_argument(
                "GpuModSwitchHandler::raise_modulus_ciphertext: no matching source parameter shard");
        }

        const auto *target_parameter_shard =
            find_parameter_shard(destination_level_info, destination_shard);
        if (target_parameter_shard == nullptr)
        {
            throw std::invalid_argument(
                "GpuModSwitchHandler::raise_modulus_ciphertext: no matching target parameter shard");
        }

        NvtxRange component_range(
            "raise_modulus.component[" + std::to_string(i) + "]");
        kernel::launch_bootstrap_modraise_poly_shard(
            destination_shard,
            source_shard,
            *source_parameter_shard,
            *target_parameter_shard,
            source_q_count,
            destination_q_count,
            degree);
    }
}

}  // namespace gpu
}  // namespace poseidon
