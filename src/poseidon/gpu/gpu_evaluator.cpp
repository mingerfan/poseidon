#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/kernels/gpu_keyswitch_kernels.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace poseidon
{
namespace gpu
{

namespace
{

bool same_scale(double a, double b)
{
    const double tolerance =
        1e-6 * std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= tolerance;
}

bool same_logical_shard_layout(
    const GpuRNSPoly &reference,
    const GpuRNSPoly &candidate)
{
    if (reference.shards.size() != candidate.shards.size())
    {
        return false;
    }

    for (std::size_t i = 0; i < reference.shards.size(); ++i)
    {
        const auto &lhs = reference.shards[i];
        const auto &rhs = candidate.shards[i];
        if (lhs.limb_begin != rhs.limb_begin ||
            lhs.limb_count != rhs.limb_count ||
            lhs.coeff_begin != rhs.coeff_begin ||
            lhs.coeff_count != rhs.coeff_count)
        {
            return false;
        }
    }

    return true;
}

bool all_components_use_layout(
    const GpuCiphertextData &ciphertext,
    const GpuRNSPoly &reference)
{
    for (const auto &poly : ciphertext.polys_)
    {
        if (!same_logical_shard_layout(reference, poly))
        {
            return false;
        }
    }

    return true;
}

bool ciphertext_storage_matches(
    const GpuCiphertextData &ciphertext,
    std::size_t degree,
    std::size_t q_count,
    std::size_t p_count,
    std::size_t component_count,
    int device_id,
    const GpuRNSPoly &reference_layout)
{
    if (ciphertext.empty() ||
        ciphertext.fields_.empty() ||
        ciphertext.size() != component_count)
    {
        return false;
    }
    if (ciphertext.fields_.front().device_id != device_id)
    {
        return false;
    }
    for (const auto &poly : ciphertext.polys_)
    {
        if (poly.degree != degree ||
            poly.q_count != q_count ||
            poly.p_count != p_count ||
            !same_logical_shard_layout(reference_layout, poly))
        {
            return false;
        }
    }
    return true;
}

void prepare_ciphertext_destination(
    GpuCiphertextData &destination,
    const GpuCiphertextData *alias_source0,
    const GpuCiphertextData *alias_source1,
    const GpuCiphertextMeta &meta,
    std::size_t component_count,
    int device_id,
    const GpuRNSPoly &reference_layout)
{
    const bool aliases_input =
        &destination == alias_source0 ||
        (alias_source1 != nullptr && &destination == alias_source1);
    if (aliases_input ||
        !ciphertext_storage_matches(
            destination,
            meta.degree,
            meta.q_count,
            meta.p_count,
            component_count,
            device_id,
            reference_layout))
    {
        destination =
            GpuCiphertextData::allocate_single_device_sharded(
                meta.degree,
                meta.q_count,
                component_count,
                device_id,
                reference_layout.shards,
                meta.p_count);
    }

    destination.meta = meta;
    destination.meta.component_count = component_count;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char *what)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    {
        throw std::overflow_error(what);
    }
    return a * b;
}

void validate_ntt_ciphertext_input(
    const char *name,
    const GpuCiphertextData &source_ciphertext,
    bool expect_ntt_form)
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument(std::string(name) + ": empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument(std::string(name) + ": empty ciphertext storage");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument(std::string(name) + ": component metadata mismatch");
    }
    if (source_ciphertext.meta.is_ntt_form != expect_ntt_form)
    {
        throw std::invalid_argument(std::string(name) + ": NTT form mismatch");
    }
    if (source_ciphertext.meta.degree == 0 ||
        source_ciphertext.meta.q_count + source_ciphertext.meta.p_count == 0)
    {
        throw std::invalid_argument(std::string(name) + ": invalid ciphertext shape");
    }

    const auto &reference_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument(std::string(name) + ": shard layout mismatch");
    }
}

void zero_poly(
    GpuRNSPolyView &poly,
    const char *name)
{
    for (const auto &shard : poly.shards)
    {
        const std::size_t word_count = checked_mul(
            shard.limb_count,
            shard.coeff_count,
            "GpuEvaluator zero word count overflow");
        gpu_check_cuda(cudaSetDevice(shard.device_id), name);
        gpu_check_cuda(
            cudaMemset(
                shard.ptr,
                0,
                word_count * sizeof(GpuWord)),
            name);
    }
}

void copy_poly(
    GpuRNSPolyView &destination,
    const GpuConstRNSPolyView &source,
    const char *name)
{
    if (destination.shards.size() != source.shards.size())
    {
        throw std::invalid_argument(std::string(name) + ": shard count mismatch");
    }

    for (std::size_t i = 0; i < destination.shards.size(); ++i)
    {
        const auto &dst = destination.shards[i];
        const auto &src = source.shards[i];
        if (dst.device_id != src.device_id ||
            dst.limb_begin != src.limb_begin ||
            dst.limb_count != src.limb_count ||
            dst.coeff_begin != src.coeff_begin ||
            dst.coeff_count != src.coeff_count)
        {
            throw std::invalid_argument(std::string(name) + ": shard placement mismatch");
        }

        const std::size_t word_count = checked_mul(
            dst.limb_count,
            dst.coeff_count,
            "GpuEvaluator copy word count overflow");
        gpu_check_cuda(cudaSetDevice(dst.device_id), name);
        gpu_check_cuda(
            cudaMemcpy(
                dst.ptr,
                src.ptr,
                word_count * sizeof(GpuWord),
                cudaMemcpyDeviceToDevice),
            name);
    }
}

std::uint32_t galois_elt_from_rotation_step(
    std::size_t degree,
    int step)
{
    if (degree == 0 || degree > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max() / 2))
    {
        throw std::invalid_argument("GpuEvaluator::rotate: invalid degree");
    }
    if ((degree & (degree - 1)) != 0)
    {
        throw std::invalid_argument("GpuEvaluator::rotate: degree must be a power of two");
    }

    const std::uint32_t n = static_cast<std::uint32_t>(degree);
    const std::uint32_t m32 = n << 1;
    const std::uint64_t m = static_cast<std::uint64_t>(m32);

    if (step == 0)
    {
        return m32 - 1;
    }

    const bool negative = step < 0;
    const std::int64_t signed_step = static_cast<std::int64_t>(step);
    const std::uint64_t abs_step = negative
        ? static_cast<std::uint64_t>(-signed_step)
        : static_cast<std::uint64_t>(signed_step);

    if (abs_step >= (static_cast<std::uint64_t>(n) >> 1))
    {
        throw std::invalid_argument("GpuEvaluator::rotate: step count too large");
    }

    std::uint32_t rotation_count = static_cast<std::uint32_t>(abs_step);
    if (negative)
    {
        rotation_count = (n >> 1) - rotation_count;
    }

    /* 与 Poseidon::util::GaloisTool::generator_ 保持一致。 */
    constexpr std::uint64_t generator = 5;
    std::uint64_t galois_elt = 1;
    while (rotation_count-- != 0)
    {
        galois_elt *= generator;
        galois_elt &= m - 1;
    }
    return static_cast<std::uint32_t>(galois_elt);
}

std::size_t galois_key_index(std::uint32_t galois_elt)
{
    if ((galois_elt & 1U) == 0U)
    {
        throw std::invalid_argument("GpuEvaluator::rotate: invalid Galois element");
    }
    return static_cast<std::size_t>((galois_elt - 1U) >> 1U);
}

std::uint32_t galois_elt_for_conjugation(std::size_t degree)
{
    if (degree == 0 || degree > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max() / 2))
    {
        throw std::invalid_argument("GpuEvaluator::conjugate: invalid degree");
    }
    if ((degree & (degree - 1)) != 0)
    {
        throw std::invalid_argument("GpuEvaluator::conjugate: degree must be a power of two");
    }

    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(degree) << 1) - 1);
}

}  // namespace

GpuEvaluator::GpuEvaluator(const GpuParameterData &params)
    : params_(params),
      elementwise_handler_(params),
      keyswitch_handler_(params),
      ntt_handler_(params),
      modswitch_handler_(params)
{}

void GpuEvaluator::add(
    const GpuCiphertextData &left_ciphertext,
    const GpuCiphertextData &right_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (left_ciphertext.empty() || right_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::add: empty ciphertext");
    }

    if (!(left_ciphertext.meta.parms_id == right_ciphertext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::add: parms_id mismatch");
    }

    if (left_ciphertext.meta.is_ntt_form != right_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::add: NTT form mismatch");
    }

    if (left_ciphertext.meta.degree != right_ciphertext.meta.degree ||
        left_ciphertext.meta.q_count != right_ciphertext.meta.q_count ||
        left_ciphertext.meta.p_count != right_ciphertext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::add: shape mismatch");
    }

    if (!same_scale(left_ciphertext.meta.scale, right_ciphertext.meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::add: scale mismatch");
    }

    if (left_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::add: p limbs are not supported by add kernel yet");
    }

    const std::size_t result_components =
        std::max(left_ciphertext.size(), right_ciphertext.size());

    const int device_id = left_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = left_ciphertext.polys_.at(0);

    if (left_ciphertext.meta.component_count != left_ciphertext.size() ||
        right_ciphertext.meta.component_count != right_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::add: component metadata mismatch");
    }

    if (!all_components_use_layout(left_ciphertext, reference_layout) ||
        !all_components_use_layout(right_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::add: shard layout mismatch");
    }

    prepare_ciphertext_destination(
        destination_ciphertext,
        &left_ciphertext,
        &right_ciphertext,
        left_ciphertext.meta,
        result_components,
        device_id,
        reference_layout);

    auto left_view = left_ciphertext.make_const_view();
    auto right_view = right_ciphertext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();

    const auto &level_info = params_.get_level(left_ciphertext.meta.parms_id);

    elementwise_handler_.add_ciphertext(
        destination_view,
        left_view,
        right_view,
        level_info);
}

void GpuEvaluator::sub(
    const GpuCiphertextData &left_ciphertext,
    const GpuCiphertextData &right_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (left_ciphertext.empty() || right_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::sub: empty ciphertext");
    }

    if (!(left_ciphertext.meta.parms_id == right_ciphertext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::sub: parms_id mismatch");
    }

    if (left_ciphertext.meta.is_ntt_form != right_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::sub: NTT form mismatch");
    }

    if (left_ciphertext.meta.degree != right_ciphertext.meta.degree ||
        left_ciphertext.meta.q_count != right_ciphertext.meta.q_count ||
        left_ciphertext.meta.p_count != right_ciphertext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::sub: shape mismatch");
    }

    if (!same_scale(left_ciphertext.meta.scale, right_ciphertext.meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::sub: scale mismatch");
    }

    if (left_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::sub: p limbs are not supported by sub kernel yet");
    }

    const std::size_t result_components =
        std::max(left_ciphertext.size(), right_ciphertext.size());

    const int device_id = left_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = left_ciphertext.polys_.at(0);

    if (left_ciphertext.meta.component_count != left_ciphertext.size() ||
        right_ciphertext.meta.component_count != right_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::sub: component metadata mismatch");
    }

    if (!all_components_use_layout(left_ciphertext, reference_layout) ||
        !all_components_use_layout(right_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::sub: shard layout mismatch");
    }

    prepare_ciphertext_destination(
        destination_ciphertext,
        &left_ciphertext,
        &right_ciphertext,
        left_ciphertext.meta,
        result_components,
        device_id,
        reference_layout);

    auto left_view = left_ciphertext.make_const_view();
    auto right_view = right_ciphertext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();

    const auto &level_info = params_.get_level(left_ciphertext.meta.parms_id);

    elementwise_handler_.sub_ciphertext(
        destination_view,
        left_view,
        right_view,
        level_info);
}

void GpuEvaluator::negate(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::negate: empty ciphertext");
    }

    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::negate: component metadata mismatch");
    }

    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::negate: p limbs are not supported by negate kernel yet");
    }

    const std::size_t result_components = source_ciphertext.size();

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::negate: shard layout mismatch");
    }

    prepare_ciphertext_destination(
        destination_ciphertext,
        &source_ciphertext,
        nullptr,
        source_ciphertext.meta,
        result_components,
        device_id,
        reference_layout);

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();

    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    elementwise_handler_.negate_ciphertext(
        destination_view,
        source_view,
        level_info);
}

void GpuEvaluator::add_plain(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty() || source_plaintext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: empty input");
    }

    if (!(source_ciphertext.meta.parms_id == source_plaintext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: parms_id mismatch");
    }

    if (source_ciphertext.meta.is_ntt_form != source_plaintext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: NTT form mismatch");
    }

    // CKKS add_plain usually expects both ciphertext and plaintext in NTT form.
    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: CKKS input must be in NTT form");
    }

    if (source_ciphertext.meta.degree != source_plaintext.meta.degree ||
        source_ciphertext.meta.q_count != source_plaintext.meta.q_count ||
        source_ciphertext.meta.p_count != source_plaintext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: shape mismatch");
    }

    if (!same_scale(source_ciphertext.meta.scale, source_plaintext.meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: scale mismatch");
    }

    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::add_plain: p limbs are not supported by add_plain kernel yet");
    }

    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: component metadata mismatch");
    }

    const std::size_t result_components = source_ciphertext.size();

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: ciphertext shard layout mismatch");
    }

    if (!same_logical_shard_layout(reference_layout, source_plaintext.poly_))
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: plaintext shard layout mismatch");
    }

    prepare_ciphertext_destination(
        destination_ciphertext,
        &source_ciphertext,
        nullptr,
        source_ciphertext.meta,
        result_components,
        device_id,
        reference_layout);

    auto ciphertext_view = source_ciphertext.make_const_view();
    auto plaintext_view = source_plaintext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();

    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    elementwise_handler_.add_plain_to_ciphertext(
        destination_view,
        ciphertext_view,
        plaintext_view,
        level_info);
}

void GpuEvaluator::sub_plain(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty() || source_plaintext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: empty input");
    }

    if (!(source_ciphertext.meta.parms_id == source_plaintext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: parms_id mismatch");
    }

    if (source_ciphertext.meta.is_ntt_form != source_plaintext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: NTT form mismatch");
    }

    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: CKKS input must be in NTT form");
    }

    if (source_ciphertext.meta.degree != source_plaintext.meta.degree ||
        source_ciphertext.meta.q_count != source_plaintext.meta.q_count ||
        source_ciphertext.meta.p_count != source_plaintext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: shape mismatch");
    }

    if (!same_scale(source_ciphertext.meta.scale, source_plaintext.meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: scale mismatch");
    }

    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::sub_plain: p limbs are not supported by sub_plain kernel yet");
    }

    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: component metadata mismatch");
    }

    const std::size_t result_components = source_ciphertext.size();

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: ciphertext shard layout mismatch");
    }

    if (!same_logical_shard_layout(reference_layout, source_plaintext.poly_))
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: plaintext shard layout mismatch");
    }

    prepare_ciphertext_destination(
        destination_ciphertext,
        &source_ciphertext,
        nullptr,
        source_ciphertext.meta,
        result_components,
        device_id,
        reference_layout);

    auto ciphertext_view = source_ciphertext.make_const_view();
    auto plaintext_view = source_plaintext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();

    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    elementwise_handler_.sub_plain_from_ciphertext(
        destination_view,
        ciphertext_view,
        plaintext_view,
        level_info);
}

void GpuEvaluator::multiply_plain(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty() || source_plaintext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::multiply_plain: empty input");
    }

    if (!(source_ciphertext.meta.parms_id == source_plaintext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::multiply_plain: parms_id mismatch");
    }

    if (source_ciphertext.meta.is_ntt_form != source_plaintext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::multiply_plain: NTT form mismatch");
    }

    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::multiply_plain: CKKS input must be in NTT form");
    }

    if (source_ciphertext.meta.degree != source_plaintext.meta.degree ||
        source_ciphertext.meta.q_count != source_plaintext.meta.q_count ||
        source_ciphertext.meta.p_count != source_plaintext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::multiply_plain: shape mismatch");
    }

    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain: p limbs are not supported yet");
    }

    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain: component metadata mismatch");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain: ciphertext shard layout mismatch");
    }

    if (!same_logical_shard_layout(reference_layout, source_plaintext.poly_))
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain: plaintext shard layout mismatch");
    }

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            source_ciphertext.size(),
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);

    result.meta = source_ciphertext.meta;
    result.meta.component_count = source_ciphertext.size();
    result.meta.scale =
        source_ciphertext.meta.scale * source_plaintext.meta.scale;

    if (!(result.meta.scale > 0.0) || !std::isfinite(result.meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::multiply_plain: invalid result scale");
    }

    auto ciphertext_view = source_ciphertext.make_const_view();
    auto plaintext_view = source_plaintext.make_const_view();
    auto destination_view = result.make_view();

    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    elementwise_handler_.multiply_plain_with_ciphertext(
        destination_view,
        ciphertext_view,
        plaintext_view,
        level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::ntt_fwd(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::ntt_fwd",
        source_ciphertext,
        false);

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            source_ciphertext.size(),
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);

    result.meta = source_ciphertext.meta;
    result.meta.component_count = source_ciphertext.size();
    result.meta.is_ntt_form = true;

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();

    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    ntt_handler_.forward_ciphertext(
        destination_view,
        source_view,
        level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::ntt_inv(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::ntt_inv",
        source_ciphertext,
        true);

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            source_ciphertext.size(),
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);

    result.meta = source_ciphertext.meta;
    result.meta.component_count = source_ciphertext.size();
    result.meta.is_ntt_form = false;

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();

    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    ntt_handler_.inverse_ciphertext(
        destination_view,
        source_view,
        level_info);

    destination_ciphertext = std::move(result);
}

/**
 * @brief 用户端顶层算子，进行必要的输入检查，构建结果临时缓存，调用multiply_ciphertext
 */
void GpuEvaluator::multiply(
    const GpuCiphertextData &left_ciphertext,
    const GpuCiphertextData &right_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (left_ciphertext.empty() || right_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::multiply: empty ciphertext");
    }
    if (left_ciphertext.fields_.empty() || right_ciphertext.fields_.empty())
    {
        throw std::invalid_argument("GpuEvaluator::multiply: empty ciphertext storage");
    }

    if (!(left_ciphertext.meta.parms_id == right_ciphertext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::multiply: parms_id mismatch");
    }
    if (!left_ciphertext.meta.is_ntt_form ||
        !right_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::multiply: CKKS inputs must be in NTT form");
    }
    if (left_ciphertext.meta.degree != right_ciphertext.meta.degree ||
        left_ciphertext.meta.q_count != right_ciphertext.meta.q_count ||
        left_ciphertext.meta.p_count != right_ciphertext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::multiply: shape mismatch");
    }
    if (left_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuEvaluator::multiply: p limbs are not supported yet");
    }
    if (left_ciphertext.meta.component_count != left_ciphertext.size() ||
        right_ciphertext.meta.component_count != right_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::multiply: component metadata mismatch");
    }

    const int device_id = left_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = left_ciphertext.polys_.at(0);

    if (!all_components_use_layout(left_ciphertext, reference_layout) ||
        !all_components_use_layout(right_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::multiply: shard layout mismatch");
    }

    const std::size_t result_components =
        left_ciphertext.size() + right_ciphertext.size() - 1;

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            left_ciphertext.meta.degree,
            left_ciphertext.meta.q_count,
            result_components,
            device_id,
            reference_layout.shards,
            left_ciphertext.meta.p_count);

    result.meta = left_ciphertext.meta;
    result.meta.component_count = result_components;
    result.meta.is_ntt_form = true;
    result.meta.scale =
        left_ciphertext.meta.scale * right_ciphertext.meta.scale;

    if (!(result.meta.scale > 0.0) || !std::isfinite(result.meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::multiply: invalid result scale");
    }

    auto left_view = left_ciphertext.make_const_view();
    auto right_view = right_ciphertext.make_const_view();
    auto destination_view = result.make_view();

    const auto &level_info = params_.get_level(left_ciphertext.meta.parms_id);

    elementwise_handler_.multiply_ciphertext(
        destination_view,
        left_view,
        right_view,
        level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::square(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    multiply(source_ciphertext, source_ciphertext, destination_ciphertext);
}

void GpuEvaluator::rescale(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::rescale: empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument("GpuEvaluator::rescale: empty ciphertext storage");
    }
    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::rescale: CKKS input must be in NTT form");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuEvaluator::rescale: p limbs are not supported yet");
    }
    if (source_ciphertext.meta.q_count < 2)
    {
        throw std::invalid_argument("GpuEvaluator::rescale: cannot drop the last q modulus");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::rescale: component metadata mismatch");
    }

    const auto &source_level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    const auto &destination_level_info =
        params_.get_next_level(source_ciphertext.meta.parms_id);

    if (source_level_info.q_count != source_ciphertext.meta.q_count ||
        destination_level_info.q_count + 1 != source_level_info.q_count)
    {
        throw std::invalid_argument("GpuEvaluator::rescale: level q_count mismatch");
    }
    if (source_level_info.shards.empty() ||
        source_level_info.shards.front().q_last == 0)
    {
        throw std::invalid_argument("GpuEvaluator::rescale: missing source q_last parameter");
    }

    const std::size_t destination_q_count =
        source_ciphertext.meta.q_count - 1;
    const int device_id = source_ciphertext.fields_.at(0).device_id;

    GpuPolyShard destination_shard;
    destination_shard.field_index = 0;
    destination_shard.field_offset = 0;
    destination_shard.limb_begin = 0;
    destination_shard.limb_count = destination_q_count;
    destination_shard.coeff_begin = 0;
    destination_shard.coeff_count = source_ciphertext.meta.degree;

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            destination_q_count,
            source_ciphertext.size(),
            device_id,
            std::vector<GpuPolyShard>{destination_shard},
            0);

    result.meta = source_ciphertext.meta;
    result.meta.parms_id = destination_level_info.parms_id;
    result.meta.q_count = destination_q_count;
    result.meta.p_count = 0;
    result.meta.component_count = source_ciphertext.size();
    result.meta.is_ntt_form = true;
    result.meta.scale =
        source_ciphertext.meta.scale /
        static_cast<double>(source_level_info.shards.front().q_last);

    if (!(result.meta.scale > 0.0) || !std::isfinite(result.meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::rescale: invalid result scale");
    }

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();

    modswitch_handler_.rescale_ciphertext(
        destination_view,
        source_view,
        source_level_info,
        destination_level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::rescale_dynamic(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext,
    double min_scale) const
{
    // TODO:
    // 1. Determine how many physical 32-bit primes should be dropped.
    // 2. Determine destination level and scale.
    // 3. Prepare destination metadata and storage.
    // 4. Call modswitch_handler_.rescale_dynamic_ciphertext(...).

    (void)source_ciphertext;
    (void)destination_ciphertext;
    (void)min_scale;

    throw std::runtime_error("GpuEvaluator::rescale_dynamic is not implemented yet");
}

void GpuEvaluator::drop_modulus(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext,
    parms_id_type target_parms_id) const
{
    // TODO:
    // 1. Check target parms_id.
    // 2. Prepare destination metadata and storage.
    // 3. Query source and destination level info.
    // 4. Call modswitch_handler_.drop_modulus_ciphertext(...).

    (void)source_ciphertext;
    (void)destination_ciphertext;
    (void)target_parms_id;

    throw std::runtime_error("GpuEvaluator::drop_modulus is not implemented yet");
}

/* GpuEvaluator::multiply(...) -> 输出 3 component: d0, d1, d2 */
void GpuEvaluator::relinearize(
    const GpuCiphertextData &source_ciphertext,
    const GpuRelinKeysData &relin_keys,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::relinearize: empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument("GpuEvaluator::relinearize: empty ciphertext storage");
    }
    if (relin_keys.empty())
    {
        throw std::invalid_argument("GpuEvaluator::relinearize: empty relin keys");
    }
    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize: CKKS ciphertext must be in NTT form");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize: component metadata mismatch");
    }
    /* 重线性化只支持输入分量个数为3 */
    if (source_ciphertext.size() != 3)
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize: first HYBRID implementation expects a size-3 ciphertext");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize: input ciphertext p limbs are not supported");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::relinearize: shard layout mismatch");
    }

    /* 分配临时结果缓存，只允许2个分量存在 */
    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            2,
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);

    /* 输出仍在同一层级，但重线性化后只保留2个密文分量 */
    result.meta = source_ciphertext.meta;
    result.meta.component_count = 2;
    result.meta.is_ntt_form = true;

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();
    auto relin_keys_view = relin_keys.make_const_view();
    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    /* 通过dnum分解的方式进行重线性化 */
    keyswitch_handler_.relinearize_hybrid_ciphertext(
        destination_view,
        source_view,
        relin_keys_view,
        relin_keys,
        level_info);

    destination_ciphertext = std::move(result);
}

/*顶层旋转操作入口*/
void GpuEvaluator::rotate(
    const GpuCiphertextData &source_ciphertext,
    int step,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &destination_ciphertext) const
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::rotate",
        source_ciphertext,
        true);
    if (source_ciphertext.size() != 2)
    {
        throw std::invalid_argument(
            "GpuEvaluator::rotate: first implementation expects a size-2 ciphertext");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::rotate: input ciphertext p limbs are not supported");
    }
    if (source_ciphertext.polys_.at(0).shards.size() != 1)
    {
        throw std::invalid_argument(
            "GpuEvaluator::rotate: first implementation requires one full shard");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);
    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            2,
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);

    result.meta = source_ciphertext.meta;
    result.meta.component_count = 2;
    result.meta.is_ntt_form = true;

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();

    /*step为0说明没有旋转，直接保持不变*/
    if (step == 0)
    {
        copy_poly(
            destination_view.polys[0],
            source_view.polys[0],
            "GpuEvaluator::rotate copy c0");
        copy_poly(
            destination_view.polys[1],
            source_view.polys[1],
            "GpuEvaluator::rotate copy c1");
        destination_ciphertext = std::move(result);
        return;
    }

    if (galois_keys.empty())
    {
        throw std::invalid_argument("GpuEvaluator::rotate: empty galois keys");
    }

    /*step不为0的情况，step表示左旋转位数，galois_elt_from_rotation_step负责把step转换成密文的Galois element形态。*/
    const std::uint32_t galois_elt = galois_elt_from_rotation_step(source_ciphertext.meta.degree, step);
    /*选择密钥，因为不同的step对应不同的旋转密钥，所以galois_key_index选择对应的高斯密钥*/
    const std::size_t key_index = galois_key_index(galois_elt);

    GpuCiphertextData rotated_c1 =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            1,
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);
    rotated_c1.meta = source_ciphertext.meta;
    rotated_c1.meta.component_count = 1;
    rotated_c1.meta.is_ntt_form = true;

    auto rotated_c1_view = rotated_c1.make_view();

    kernel::launch_apply_galois_ntt_poly_shard(
        destination_view.polys[0].shards.front(),
        source_view.polys[0].shards.front(),
        galois_elt,
        source_ciphertext.meta.degree);
    kernel::launch_apply_galois_ntt_poly_shard(
        rotated_c1_view.polys[0].shards.front(),
        source_view.polys[1].shards.front(),
        galois_elt,
        source_ciphertext.meta.degree);

    /* c1 先清零，后续 switch-key 会把 rotated_c1 * galois_key 累加进去。 */
    zero_poly(destination_view.polys[1], "GpuEvaluator::rotate zero c1");

    auto rotated_c1_const_view = rotated_c1.make_const_view();
    auto galois_keys_view = galois_keys.make_const_view();
    keyswitch_handler_.switch_key_hybrid_ciphertext(
        destination_view,
        rotated_c1_const_view.polys[0],
        galois_keys_view,
        galois_keys,
        key_index,
        level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::conjugate(
    const GpuCiphertextData &source_ciphertext,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &destination_ciphertext) const
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::conjugate",
        source_ciphertext,
        true);
    if (source_ciphertext.size() != 2)
    {
        throw std::invalid_argument(
            "GpuEvaluator::conjugate: first implementation expects a size-2 ciphertext");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::conjugate: input ciphertext p limbs are not supported");
    }
    if (source_ciphertext.polys_.at(0).shards.size() != 1)
    {
        throw std::invalid_argument(
            "GpuEvaluator::conjugate: first implementation requires one full shard");
    }
    if (galois_keys.empty())
    {
        throw std::invalid_argument("GpuEvaluator::conjugate: empty galois keys");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);
    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            2,
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);

    result.meta = source_ciphertext.meta;
    result.meta.component_count = 2;
    result.meta.is_ntt_form = true;

    const std::uint32_t galois_elt =
        galois_elt_for_conjugation(source_ciphertext.meta.degree);
    const std::size_t key_index = galois_key_index(galois_elt);

    GpuCiphertextData conjugated_c1 =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            1,
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);
    conjugated_c1.meta = source_ciphertext.meta;
    conjugated_c1.meta.component_count = 1;
    conjugated_c1.meta.is_ntt_form = true;

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();
    auto conjugated_c1_view = conjugated_c1.make_view();

    kernel::launch_apply_galois_ntt_poly_shard(
        destination_view.polys[0].shards.front(),
        source_view.polys[0].shards.front(),
        galois_elt,
        source_ciphertext.meta.degree);
    kernel::launch_apply_galois_ntt_poly_shard(
        conjugated_c1_view.polys[0].shards.front(),
        source_view.polys[1].shards.front(),
        galois_elt,
        source_ciphertext.meta.degree);

    zero_poly(destination_view.polys[1], "GpuEvaluator::conjugate zero c1");

    auto conjugated_c1_const_view = conjugated_c1.make_const_view();
    auto galois_keys_view = galois_keys.make_const_view();
    keyswitch_handler_.switch_key_hybrid_ciphertext(
        destination_view,
        conjugated_c1_const_view.polys[0],
        galois_keys_view,
        galois_keys,
        key_index,
        level_info);

    destination_ciphertext = std::move(result);
}

}  // namespace gpu
}  // namespace poseidon
