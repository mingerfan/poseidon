#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/kernels/gpu_keyswitch_kernels.h"

#include "poseidon/advance/homomorphic_linear_transform.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <functional>
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

std::vector<GpuEvalModBasisStep> make_gpu_eval_mod_basis_plan(
    GpuEvalModPolynomialBasis basis,
    const std::vector<std::uint32_t> &requested_degrees)
{
    std::vector<GpuEvalModBasisStep> steps;
    std::map<std::uint32_t, bool> available;
    available.emplace(0, true);
    available.emplace(1, true);

    std::function<void(std::uint32_t)> schedule_degree =
        [&](std::uint32_t degree) {
            if (degree <= 1 || available.count(degree) != 0)
            {
                return;
            }
            if (degree == std::numeric_limits<std::uint32_t>::max())
            {
                throw std::invalid_argument(
                    "make_gpu_eval_mod_basis_plan: degree is too large");
            }

            const bool is_power_of_two = (degree & (degree - 1U)) == 0U;
            std::uint32_t left_degree = 0;
            std::uint32_t right_degree = 0;
            std::uint32_t correction_degree = 0;
            if (is_power_of_two)
            {
                left_degree = degree >> 1U;
                right_degree = left_degree;
            }
            else
            {
                std::uint32_t power_of_two = 1;
                const std::uint32_t split_limit = (degree + 1U) >> 1U;
                while (power_of_two < split_limit)
                {
                    power_of_two <<= 1U;
                }
                left_degree = power_of_two - 1U;
                right_degree = degree + 1U - power_of_two;
                if (basis == GpuEvalModPolynomialBasis::Chebyshev)
                {
                    correction_degree = left_degree > right_degree
                        ? left_degree - right_degree
                        : right_degree - left_degree;
                }
            }

            schedule_degree(left_degree);
            schedule_degree(right_degree);
            if (basis == GpuEvalModPolynomialBasis::Chebyshev &&
                correction_degree != 0)
            {
                schedule_degree(correction_degree);
            }

            steps.push_back(GpuEvalModBasisStep{
                degree,
                left_degree,
                right_degree,
                correction_degree});
            available.emplace(degree, true);
        };

    std::vector<std::uint32_t> sorted_degrees = requested_degrees;
    std::sort(sorted_degrees.begin(), sorted_degrees.end());
    sorted_degrees.erase(
        std::unique(sorted_degrees.begin(), sorted_degrees.end()),
        sorted_degrees.end());
    for (const auto degree : sorted_degrees)
    {
        schedule_degree(degree);
    }
    return steps;
}

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
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: empty ciphertext storage");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: p limbs are not supported yet");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: component metadata mismatch");
    }
    const auto &reference_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: shard layout mismatch");
    }
    if (reference_layout.shards.size() != 1 ||
        reference_layout.shards.front().limb_begin != 0 ||
        reference_layout.shards.front().limb_count != source_ciphertext.meta.q_count ||
        reference_layout.shards.front().coeff_begin != 0 ||
        reference_layout.shards.front().coeff_count != source_ciphertext.meta.degree)
    {
        throw std::invalid_argument(
            "GpuEvaluator::drop_modulus: first implementation requires one full q shard");
    }

    const auto &source_level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    const auto &destination_level_info =
        params_.get_level(target_parms_id);

    if (source_level_info.q_count != source_ciphertext.meta.q_count ||
        destination_level_info.p_count != 0 ||
        destination_level_info.q_count == 0 ||
        destination_level_info.q_count > source_level_info.q_count)
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: level q_count mismatch");
    }
    if (destination_level_info.degree != source_ciphertext.meta.degree)
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: degree mismatch");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    GpuPolyShard destination_shard;
    destination_shard.field_index = 0;
    destination_shard.field_offset = 0;
    destination_shard.limb_begin = 0;
    destination_shard.limb_count = destination_level_info.q_count;
    destination_shard.coeff_begin = 0;
    destination_shard.coeff_count = source_ciphertext.meta.degree;

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            destination_level_info.q_count,
            source_ciphertext.size(),
            device_id,
            std::vector<GpuPolyShard>{ destination_shard },
            0);

    result.meta = source_ciphertext.meta;
    result.meta.parms_id = destination_level_info.parms_id;
    result.meta.q_count = destination_level_info.q_count;
    result.meta.p_count = 0;
    result.meta.component_count = source_ciphertext.size();

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();
    modswitch_handler_.drop_modulus_ciphertext(
        destination_view,
        source_view,
        source_level_info,
        destination_level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::multiply_scalar(
    const GpuCiphertextData &source_ciphertext,
    std::uint64_t scalar,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: empty ciphertext storage");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: p limbs are not supported yet");
    }
    if (scalar > std::numeric_limits<GpuWord>::max())
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: scalar exceeds GPU word size");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: component metadata mismatch");
    }

    const auto &reference_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: shard layout mismatch");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);
    if (level_info.q_count != source_ciphertext.meta.q_count ||
        level_info.p_count != source_ciphertext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: level shape mismatch");
    }

    GpuCiphertextMeta result_meta = source_ciphertext.meta;
    prepare_ciphertext_destination(
        destination_ciphertext,
        &source_ciphertext,
        nullptr,
        result_meta,
        source_ciphertext.size(),
        device_id,
        reference_layout);

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();
    elementwise_handler_.multiply_scalar_ciphertext(
        destination_view,
        source_view,
        static_cast<GpuWord>(scalar),
        level_info);
}

void GpuEvaluator::raise_modulus(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::raise_modulus: empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument("GpuEvaluator::raise_modulus: empty ciphertext storage");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuEvaluator::raise_modulus: p limbs are not supported yet");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::raise_modulus: component metadata mismatch");
    }

    const auto &source_level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    const auto &destination_level_info = params_.get_first_q_level();
    if (source_level_info.p_count != 0 ||
        destination_level_info.p_count != 0 ||
        source_level_info.degree != source_ciphertext.meta.degree ||
        source_level_info.q_count != source_ciphertext.meta.q_count ||
        destination_level_info.degree != source_ciphertext.meta.degree ||
        source_level_info.q_count > destination_level_info.q_count)
    {
        throw std::invalid_argument("GpuEvaluator::raise_modulus: level shape mismatch");
    }

    const auto &source_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, source_layout) ||
        source_layout.shards.size() != 1 ||
        source_layout.shards.front().limb_begin != 0 ||
        source_layout.shards.front().limb_count != source_ciphertext.meta.q_count ||
        source_layout.shards.front().coeff_begin != 0 ||
        source_layout.shards.front().coeff_count != source_ciphertext.meta.degree)
    {
        throw std::invalid_argument(
            "GpuEvaluator::raise_modulus: first implementation requires one full q-prefix shard");
    }

    GpuCiphertextData coeff_source_storage;
    const GpuCiphertextData *coeff_source = &source_ciphertext;
    if (source_ciphertext.meta.is_ntt_form)
    {
        ntt_inv(source_ciphertext, coeff_source_storage);
        coeff_source = &coeff_source_storage;
    }

    const int device_id = coeff_source->fields_.at(0).device_id;
    GpuPolyShard destination_shard;
    destination_shard.field_index = 0;
    destination_shard.field_offset = 0;
    destination_shard.limb_begin = 0;
    destination_shard.limb_count = destination_level_info.q_count;
    destination_shard.coeff_begin = 0;
    destination_shard.coeff_count = coeff_source->meta.degree;

    GpuCiphertextData coeff_raised =
        GpuCiphertextData::allocate_single_device_sharded(
            coeff_source->meta.degree,
            destination_level_info.q_count,
            coeff_source->size(),
            device_id,
            std::vector<GpuPolyShard>{ destination_shard },
            0);

    coeff_raised.meta = coeff_source->meta;
    coeff_raised.meta.parms_id = destination_level_info.parms_id;
    coeff_raised.meta.q_count = destination_level_info.q_count;
    coeff_raised.meta.p_count = 0;
    coeff_raised.meta.component_count = coeff_source->size();
    coeff_raised.meta.is_ntt_form = false;

    auto source_view = coeff_source->make_const_view();
    auto destination_view = coeff_raised.make_view();
    modswitch_handler_.raise_modulus_ciphertext(
        destination_view,
        source_view,
        source_level_info,
        destination_level_info);

    ntt_fwd(coeff_raised, destination_ciphertext);
}

void GpuEvaluator::bootstrap_prepare_modraise_input(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext,
    parms_id_type q0_parms_id,
    double q0_over_message_ratio) const
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::bootstrap_prepare_modraise_input",
        source_ciphertext,
        true);
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_prepare_modraise_input: p limbs are not supported");
    }
    if (!(q0_over_message_ratio > 0.0) || !std::isfinite(q0_over_message_ratio))
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_prepare_modraise_input: invalid target scale");
    }

    const auto &q0_level_info = params_.get_level(q0_parms_id);
    if (q0_level_info.p_count != 0 ||
        q0_level_info.degree != source_ciphertext.meta.degree)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_prepare_modraise_input: invalid q0 level");
    }

    const GpuCiphertextData *current = &source_ciphertext;
    GpuCiphertextData scratch0;
    GpuCiphertextData scratch1;

    auto next_scratch = [&]() -> GpuCiphertextData &
    {
        return (current == &scratch0) ? scratch1 : scratch0;
    };

    while (current->meta.scale > std::pow(2.0, 54.0))
    {
        const auto &level_info = params_.get_level(current->meta.parms_id);
        if (level_info.shards.empty() || level_info.shards.front().q_last == 0)
        {
            throw std::invalid_argument(
                "GpuEvaluator::bootstrap_prepare_modraise_input: missing q_last parameter");
        }
        const double q_last = static_cast<double>(level_info.shards.front().q_last);
        if (current->meta.scale / q_last <= 1.6e+07)
        {
            throw std::invalid_argument(
                "GpuEvaluator::bootstrap_prepare_modraise_input: scale cannot be safely rescaled for bootstrap");
        }

        GpuCiphertextData &next = next_scratch();
        rescale(*current, next);
        current = &next;
    }

    if (current->meta.q_count < q0_level_info.q_count)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_prepare_modraise_input: source is below q0 level");
    }

    const std::size_t q0_plus_one_count = q0_level_info.q_count + 1;
    if (current->meta.q_count > q0_plus_one_count)
    {
        const auto &q0_plus_one_level =
            params_.get_level_by_q_count(q0_plus_one_count, 0);
        GpuCiphertextData &next = next_scratch();
        drop_modulus(*current, next, q0_plus_one_level.parms_id);
        current = &next;
    }

    const double scale_multiplier_double =
        std::round(q0_over_message_ratio / current->meta.scale);
    if (!std::isfinite(scale_multiplier_double))
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_prepare_modraise_input: invalid scale multiplier");
    }
    if (scale_multiplier_double > 1.0)
    {
        if (scale_multiplier_double >
            static_cast<double>(std::numeric_limits<GpuWord>::max()))
        {
            throw std::invalid_argument(
                "GpuEvaluator::bootstrap_prepare_modraise_input: scale multiplier exceeds GPU word size");
        }
        const auto scale_multiplier =
            static_cast<std::uint64_t>(scale_multiplier_double);
        GpuCiphertextData &next = next_scratch();
        multiply_scalar(*current, scale_multiplier, next);
        next.meta.scale = current->meta.scale * scale_multiplier_double;
        current = &next;
    }

    drop_modulus(*current, destination_ciphertext, q0_level_info.parms_id);
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
    auto relin_keys_view =
        relin_keys.make_const_view(source_ciphertext.meta.q_count);
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
    auto galois_keys_view =
        galois_keys.make_const_view(source_ciphertext.meta.q_count);
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
    auto galois_keys_view =
        galois_keys.make_const_view(source_ciphertext.meta.q_count);
    keyswitch_handler_.switch_key_hybrid_ciphertext(
        destination_view,
        conjugated_c1_const_view.polys[0],
        galois_keys_view,
        galois_keys,
        key_index,
        level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::multiply_by_diag_matrix_bsgs(
    const GpuCiphertextData &source_ciphertext,
    const GpuMatrixPlain &matrix,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &destination_ciphertext) const
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::multiply_by_diag_matrix_bsgs",
        source_ciphertext,
        true);
    if (source_ciphertext.size() != 2)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs: first implementation expects a size-2 ciphertext");
    }
    if (matrix.plain_vec.empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs: empty diagonal matrix");
    }
    if (matrix.n1 == 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs: invalid BSGS n1");
    }

    const auto [index, unused_rot_n1, rot_n2] =
        poseidon::bsgs_index(
            matrix.plain_vec,
            1 << matrix.log_slots,
            static_cast<int>(matrix.n1));
    (void)unused_rot_n1;

    std::map<int, GpuCiphertextData> baby_rotations;
    for (const int step : rot_n2)
    {
        if (step == 0)
        {
            continue;
        }

        GpuCiphertextData rotated;
        rotate(source_ciphertext, step, galois_keys, rotated);
        baby_rotations.emplace(step, std::move(rotated));
    }

    GpuCiphertextData result_accumulator;
    GpuCiphertextData inner_sum;
    GpuCiphertextData product;
    GpuCiphertextData rotated_inner_sum;
    bool have_result = false;

    for (const auto &giant_entry : index)
    {
        const int giant_step = giant_entry.first;
        bool have_inner_sum = false;

        for (const int baby_step : giant_entry.second)
        {
            const int diagonal_index = giant_step + baby_step;
            const auto plaintext_it = matrix.plain_vec.find(diagonal_index);
            if (plaintext_it == matrix.plain_vec.end())
            {
                throw std::invalid_argument(
                    "GpuEvaluator::multiply_by_diag_matrix_bsgs: missing plaintext diagonal");
            }

            const GpuCiphertextData &rotated_source =
                baby_step == 0 ? source_ciphertext : baby_rotations.at(baby_step);

            multiply_plain(rotated_source, plaintext_it->second, product);

            if (!have_inner_sum)
            {
                inner_sum = std::move(product);
                have_inner_sum = true;
            }
            else
            {
                GpuCiphertextData updated_inner_sum;
                add(inner_sum, product, updated_inner_sum);
                inner_sum = std::move(updated_inner_sum);
            }
        }

        if (!have_inner_sum)
        {
            continue;
        }

        if (!have_result)
        {
            if (giant_step == 0)
            {
                result_accumulator = std::move(inner_sum);
            }
            else
            {
                rotate(inner_sum, giant_step, galois_keys, rotated_inner_sum);
                result_accumulator = std::move(rotated_inner_sum);
            }
            have_result = true;
        }
        else
        {
            if (giant_step == 0)
            {
                GpuCiphertextData updated_accumulator;
                add(result_accumulator, inner_sum, updated_accumulator);
                result_accumulator = std::move(updated_accumulator);
            }
            else
            {
                rotate(inner_sum, giant_step, galois_keys, rotated_inner_sum);
                GpuCiphertextData updated_accumulator;
                add(result_accumulator, rotated_inner_sum, updated_accumulator);
                result_accumulator = std::move(updated_accumulator);
            }
        }
    }

    if (!have_result)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs: no nonzero diagonal contribution");
    }

    rescale(result_accumulator, destination_ciphertext);
}

void GpuEvaluator::dft(
    const GpuCiphertextData &source_ciphertext,
    const GpuLinearMatrixGroup &matrix_group,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &destination_ciphertext) const
{
    if (matrix_group.data().empty())
    {
        throw std::invalid_argument("GpuEvaluator::dft: empty matrix group");
    }

    GpuCiphertextData current;
    multiply_by_diag_matrix_bsgs(
        source_ciphertext,
        matrix_group.data().front(),
        galois_keys,
        current);

    for (std::size_t i = 1; i < matrix_group.data().size(); ++i)
    {
        GpuCiphertextData next;
        multiply_by_diag_matrix_bsgs(
            current,
            matrix_group.data()[i],
            galois_keys,
            next);
        current = std::move(next);
    }

    destination_ciphertext = std::move(current);
}

void GpuEvaluator::coeff_to_slot(
    const GpuCiphertextData &source_ciphertext,
    const GpuLinearMatrixGroup &matrix_group,
    const GpuPlaintextData &minus_i_plaintext,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &result_real,
    GpuCiphertextData &result_imag) const
{
    GpuCiphertextData dft_result;
    dft(source_ciphertext, matrix_group, galois_keys, dft_result);

    GpuCiphertextData conjugated;
    conjugate(dft_result, galois_keys, conjugated);

    add(dft_result, conjugated, result_real);

    GpuCiphertextData imag_difference;
    sub(dft_result, conjugated, imag_difference);
    multiply_plain(imag_difference, minus_i_plaintext, result_imag);
}

void GpuEvaluator::slot_to_coeff(
    const GpuCiphertextData &source_real,
    const GpuCiphertextData &source_imag,
    const GpuLinearMatrixGroup &matrix_group,
    const GpuPlaintextData &plus_i_plaintext,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &result) const
{
    GpuCiphertextData scaled_imag;
    multiply_plain(source_imag, plus_i_plaintext, scaled_imag);

    GpuCiphertextData merged_slots;
    add(scaled_imag, source_real, merged_slots);

    dft(merged_slots, matrix_group, galois_keys, result);
}

void GpuEvaluator::bootstrap(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    const GpuRelinKeysData &relin_keys,
    const GpuGaloisKeysData &galois_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    if (bootstrap_data.coeff_to_slot_matrix.data().empty())
    {
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty CoeffToSlot matrix group");
    }
    if (bootstrap_data.slot_to_coeff_matrix.data().empty())
    {
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty SlotToCoeff matrix group");
    }
    if (bootstrap_data.minus_i_plaintext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty minus-i plaintext");
    }
    if (bootstrap_data.plus_i_plaintext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty plus-i plaintext");
    }
    if (!(bootstrap_data.q0_over_message_ratio > 0.0) ||
        !std::isfinite(bootstrap_data.q0_over_message_ratio))
    {
        throw std::invalid_argument("GpuEvaluator::bootstrap: invalid q0/message ratio");
    }

    bootstrap_prepare_modraise_input(
        source_ciphertext,
        workspace.modraise_input,
        bootstrap_data.q0_parms_id,
        bootstrap_data.q0_over_message_ratio);

    raise_modulus(workspace.modraise_input, workspace.raised);

    const GpuCiphertextData *raised_for_c2s = &workspace.raised;
    if (bootstrap_data.raised_scale_override > 0.0)
    {
        workspace.raised.meta.scale = bootstrap_data.raised_scale_override;
    }

    if (bootstrap_data.post_raise_integer_multiplier > 1)
    {
        multiply_scalar(
            workspace.raised,
            bootstrap_data.post_raise_integer_multiplier,
            workspace.raised_scaled);
        workspace.raised_scaled.meta.scale =
            workspace.raised.meta.scale * bootstrap_data.post_raise_scale_multiplier;
        raised_for_c2s = &workspace.raised_scaled;
    }
    else if (!bootstrap_data.post_raise_plaintext.empty())
    {
        multiply_plain(
            workspace.raised,
            bootstrap_data.post_raise_plaintext,
            workspace.raised_scaled);
        raised_for_c2s = &workspace.raised_scaled;
    }

    coeff_to_slot(
        *raised_for_c2s,
        bootstrap_data.coeff_to_slot_matrix,
        bootstrap_data.minus_i_plaintext,
        galois_keys,
        workspace.coeff_to_slot_real,
        workspace.coeff_to_slot_imag);

    eval_mod_high_precision(
        workspace.coeff_to_slot_real,
        bootstrap_data,
        relin_keys,
        workspace,
        workspace.eval_mod_real);
    eval_mod_high_precision(
        workspace.coeff_to_slot_imag,
        bootstrap_data,
        relin_keys,
        workspace,
        workspace.eval_mod_imag);

    if (bootstrap_data.slot_to_coeff_input_scale > 0.0)
    {
        workspace.eval_mod_real.meta.scale =
            bootstrap_data.slot_to_coeff_input_scale;
        workspace.eval_mod_imag.meta.scale =
            bootstrap_data.slot_to_coeff_input_scale;
    }

    slot_to_coeff(
        workspace.eval_mod_real,
        workspace.eval_mod_imag,
        bootstrap_data.slot_to_coeff_matrix,
        bootstrap_data.plus_i_plaintext,
        galois_keys,
        destination_ciphertext);
}

void GpuEvaluator::eval_mod_high_precision(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    const GpuRelinKeysData &relin_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::eval_mod_high_precision",
        source_ciphertext,
        true);
    if (source_ciphertext.size() != 2)
    {
        throw std::invalid_argument(
            "GpuEvaluator::eval_mod_high_precision: expected a size-2 ciphertext");
    }
    const double configured_target_scale =
        bootstrap_data.eval_mod.target_scale > 0.0
            ? bootstrap_data.eval_mod.target_scale
            : bootstrap_data.eval_mod_target_scale;
    const auto &configured_coefficients =
        !bootstrap_data.eval_mod.polynomial_coefficients.empty()
            ? bootstrap_data.eval_mod.polynomial_coefficients
            : bootstrap_data.eval_mod_polynomial_plaintexts;
    const auto &polynomial_terms = bootstrap_data.eval_mod.polynomial_terms;
    const auto &basis_steps = bootstrap_data.eval_mod.basis_steps;
    const auto &polynomial_blocks = bootstrap_data.eval_mod.polynomial_blocks;
    const auto &polynomial_combine_steps =
        bootstrap_data.eval_mod.polynomial_combine_steps;
    const auto &configured_double_angle_constants =
        !bootstrap_data.eval_mod.double_angle_constants.empty()
            ? bootstrap_data.eval_mod.double_angle_constants
            : bootstrap_data.double_angle_plaintexts;
    const GpuPlaintextData &configured_input_offset =
        !bootstrap_data.eval_mod.input_offset_plaintext.empty()
            ? bootstrap_data.eval_mod.input_offset_plaintext
            : bootstrap_data.eval_mod_input_offset_plaintext;

    if (!(configured_target_scale > 0.0) ||
        !std::isfinite(configured_target_scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::eval_mod_high_precision: invalid target scale");
    }
    if (polynomial_blocks.empty() &&
        polynomial_terms.empty() &&
        configured_coefficients.empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::eval_mod_high_precision: empty polynomial plaintexts");
    }

    const double target_scale = configured_target_scale;

    if (workspace.capture_eval_mod_trace)
    {
        workspace.eval_mod_trace_offset_input = GpuCiphertextData{};
        workspace.eval_mod_trace_polynomial_output = GpuCiphertextData{};
        workspace.eval_mod_trace_double_angle_outputs.clear();
        workspace.eval_mod_trace_double_angle_outputs.reserve(
            configured_double_angle_constants.size());
    }

    /*
     * CPU eval_mod_high_precision first resets the logical scale of the input
     * ciphertext to the EvalMod scaling factor.  Keep that behavior while
     * preserving the existing GPU layout by copying with the scalar kernel.
     */
    multiply_scalar(source_ciphertext, 1, workspace.scratch0);
    workspace.scratch0.meta.scale = target_scale;

    const GpuCiphertextData *x = &workspace.scratch0;
    if (!configured_input_offset.empty())
    {
        add_plain(
            workspace.scratch0,
            configured_input_offset,
            workspace.scratch1);
        workspace.scratch1.meta.scale = target_scale;
        x = &workspace.scratch1;
    }

    if (workspace.capture_eval_mod_trace)
    {
        multiply_scalar(
            *x,
            1,
            workspace.eval_mod_trace_offset_input);
        workspace.eval_mod_trace_offset_input.meta.scale = x->meta.scale;
    }

    /*
     * Keep the old Horner layout only for callers that still populate the
     * early polynomial_coefficients field. The high-precision path below uses
     * a setup-time fixed basis DAG and never performs CPU-side recursive
     * polynomial decomposition in the timed path.
     */
    if (polynomial_blocks.empty() && polynomial_terms.empty())
    {
        GpuCiphertextData accumulator;
        multiply_scalar(*x, 0, accumulator);
        accumulator.meta.scale = target_scale;

        const auto &coefficients = configured_coefficients;
        if (!coefficients.front().empty())
        {
            add_plain(accumulator, coefficients.front(), workspace.scratch2);
            accumulator = std::move(workspace.scratch2);
            accumulator.meta.scale = target_scale;
        }

        for (std::size_t i = 1; i < coefficients.size(); ++i)
        {
            const GpuCiphertextData *x_at_level = x;
            GpuCiphertextData x_dropped;
            if (!(x->meta.parms_id == accumulator.meta.parms_id))
            {
                if (x->meta.q_count < accumulator.meta.q_count)
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: x level is below accumulator level");
                }
                drop_modulus(*x, x_dropped, accumulator.meta.parms_id);
                x_dropped.meta.scale = target_scale;
                x_at_level = &x_dropped;
            }

            multiply(accumulator, *x_at_level, workspace.scratch3);
            relinearize(workspace.scratch3, relin_keys, workspace.scratch4);
            rescale(workspace.scratch4, workspace.scratch5);
            workspace.scratch5.meta.scale = target_scale;
            accumulator = std::move(workspace.scratch5);

            if (!coefficients[i].empty())
            {
                add_plain(accumulator, coefficients[i], workspace.scratch2);
                accumulator = std::move(workspace.scratch2);
                accumulator.meta.scale = target_scale;
            }
        }

        for (const auto &double_angle_plaintext : configured_double_angle_constants)
        {
            square(accumulator, workspace.scratch3);
            relinearize(workspace.scratch3, relin_keys, workspace.scratch4);
            rescale(workspace.scratch4, workspace.scratch5);
            workspace.scratch5.meta.scale = target_scale;
            add(workspace.scratch5, workspace.scratch5, workspace.scratch2);

            if (!double_angle_plaintext.empty())
            {
                add_plain(
                    workspace.scratch2,
                    double_angle_plaintext,
                    workspace.scratch3);
                accumulator = std::move(workspace.scratch3);
            }
            else
            {
                accumulator = std::move(workspace.scratch2);
            }
            accumulator.meta.scale = target_scale;
        }

        destination_ciphertext = std::move(accumulator);
        destination_ciphertext.meta.scale = source_ciphertext.meta.scale;
        return;
    }

    std::uint32_t maximum_degree = 1;
    for (const auto &term : polynomial_terms)
    {
        maximum_degree = std::max(maximum_degree, term.degree);
        if (term.coefficient_plaintext.empty())
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: empty term plaintext");
        }
    }
    for (const auto &block : polynomial_blocks)
    {
        if (block.terms.empty())
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: empty polynomial block");
        }
        for (const auto &term : block.terms)
        {
            maximum_degree = std::max(maximum_degree, term.degree);
            if (term.coefficient_plaintext.empty())
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: empty block term plaintext");
            }
        }
    }
    for (const auto &combine : polynomial_combine_steps)
    {
        maximum_degree = std::max(maximum_degree, combine.basis_degree);
    }
    for (const auto &step : basis_steps)
    {
        maximum_degree = std::max(maximum_degree, step.output_degree);
        maximum_degree = std::max(maximum_degree, step.left_degree);
        maximum_degree = std::max(maximum_degree, step.right_degree);
        maximum_degree = std::max(maximum_degree, step.correction_degree);
    }

    std::vector<bool> basis_available(
        static_cast<std::size_t>(maximum_degree) + 1,
        false);
    basis_available[0] = true;
    basis_available[1] = true;
    for (const auto &step : basis_steps)
    {
        if (step.output_degree <= 1 ||
            step.left_degree == 0 ||
            step.right_degree == 0 ||
            !basis_available[step.left_degree] ||
            !basis_available[step.right_degree] ||
            basis_available[step.output_degree])
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: basis plan is not topologically valid");
        }
        if (bootstrap_data.eval_mod.polynomial_basis ==
                GpuEvalModPolynomialBasis::Chebyshev &&
            step.correction_degree != 0 &&
            !basis_available[step.correction_degree])
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: missing Chebyshev correction basis");
        }
        basis_available[step.output_degree] = true;
    }
    for (const auto &term : polynomial_terms)
    {
        if (term.degree != 0 && !basis_available[term.degree])
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: polynomial term is absent from basis plan");
        }
    }
    for (const auto &block : polynomial_blocks)
    {
        for (const auto &term : block.terms)
        {
            if (term.degree != 0 && !basis_available[term.degree])
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: block term is absent from basis plan");
            }
        }
    }
    for (const auto &combine : polynomial_combine_steps)
    {
        if (combine.basis_degree == 0 ||
            !basis_available[combine.basis_degree])
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: combine basis is absent from basis plan");
        }
    }

    const auto required_basis_slots =
        static_cast<std::size_t>(maximum_degree) + 1;
    if (workspace.eval_mod_basis.size() < required_basis_slots)
    {
        workspace.eval_mod_basis.resize(required_basis_slots);
    }
    multiply_scalar(*x, 1, workspace.eval_mod_basis[1]);
    workspace.eval_mod_basis[1].meta.scale = target_scale;

    auto multiply_relinearize_rescale =
        [&](const GpuCiphertextData &left,
            const GpuCiphertextData &right,
            double expected_output_scale,
            GpuCiphertextData &output) {
            const GpuCiphertextData *left_at_level = &left;
            const GpuCiphertextData *right_at_level = &right;
            if (!(left.meta.parms_id == right.meta.parms_id))
            {
                if (left.meta.q_count == right.meta.q_count)
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: equal q_count has different parms_id");
                }
                if (left.meta.q_count > right.meta.q_count)
                {
                    drop_modulus(left, workspace.scratch3, right.meta.parms_id);
                    left_at_level = &workspace.scratch3;
                }
                else
                {
                    drop_modulus(right, workspace.scratch4, left.meta.parms_id);
                    right_at_level = &workspace.scratch4;
                }
            }

            multiply(*left_at_level, *right_at_level, workspace.scratch5);
            relinearize(workspace.scratch5, relin_keys, workspace.scratch3);
            // rescale_dynamic is deliberately not used in the first GPU path.
            rescale(workspace.scratch3, output);
            if (expected_output_scale > 0.0)
            {
                output.meta.scale = expected_output_scale;
            }
        };

    for (const auto &step : basis_steps)
    {
        auto &output = workspace.eval_mod_basis[step.output_degree];
        multiply_relinearize_rescale(
            workspace.eval_mod_basis[step.left_degree],
            workspace.eval_mod_basis[step.right_degree],
            step.output_scale,
            output);

        if (bootstrap_data.eval_mod.polynomial_basis !=
            GpuEvalModPolynomialBasis::Chebyshev)
        {
            continue;
        }

        add(output, output, workspace.scratch3);
        output = std::move(workspace.scratch3);
        output.meta.scale = step.output_scale;

        if (step.correction_degree == 0)
        {
            if (step.correction_plaintext.empty())
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: missing Chebyshev constant correction");
            }
            sub_plain(output, step.correction_plaintext, workspace.scratch4);
            output = std::move(workspace.scratch4);
            output.meta.scale = step.output_scale;
            continue;
        }

        const GpuCiphertextData *correction =
            &workspace.eval_mod_basis[step.correction_degree];
        if (!(output.meta.parms_id == correction->meta.parms_id))
        {
            if (output.meta.q_count > correction->meta.q_count)
            {
                drop_modulus(output, workspace.scratch4, correction->meta.parms_id);
                output = std::move(workspace.scratch4);
            }
            else if (correction->meta.q_count > output.meta.q_count)
            {
                drop_modulus(
                    *correction,
                    workspace.scratch4,
                    output.meta.parms_id);
                correction = &workspace.scratch4;
            }
            else
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: Chebyshev correction parms_id mismatch");
            }
        }
        if (step.correction_plaintext.empty() ||
            step.correction_plaintext.meta.parms_id != output.meta.parms_id)
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: invalid Chebyshev correction scale plaintext");
        }
        multiply_plain(
            *correction,
            step.correction_plaintext,
            workspace.scratch3);
        workspace.scratch3.meta.scale = step.output_scale;
        sub(output, workspace.scratch3, workspace.scratch5);
        output = std::move(workspace.scratch5);
        output.meta.scale = step.output_scale;
    }

    auto evaluate_term_block =
        [&](const std::vector<GpuEvalModPolynomialTerm> &terms,
            std::uint32_t rescale_count,
            double expected_output_scale,
            GpuCiphertextData &block_output) {
            GpuCiphertextData block_accumulator;
            double block_scale = 0.0;

            auto accumulate_term = [&](GpuCiphertextData &term_ciphertext) {
                if (!(block_scale > 0.0))
                {
                    block_scale = term_ciphertext.meta.scale;
                }
                else if (!same_scale(block_scale, term_ciphertext.meta.scale))
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: polynomial term scale mismatch");
                }
                if (block_accumulator.empty())
                {
                    block_accumulator = std::move(term_ciphertext);
                    return;
                }

                const GpuCiphertextData *left = &block_accumulator;
                const GpuCiphertextData *right = &term_ciphertext;
                if (!(left->meta.parms_id == right->meta.parms_id))
                {
                    if (left->meta.q_count > right->meta.q_count)
                    {
                        drop_modulus(*left, workspace.scratch3, right->meta.parms_id);
                        left = &workspace.scratch3;
                    }
                    else if (right->meta.q_count > left->meta.q_count)
                    {
                        drop_modulus(*right, workspace.scratch4, left->meta.parms_id);
                        right = &workspace.scratch4;
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: term parms_id mismatch");
                    }
                }
                add(*left, *right, workspace.scratch5);
                block_accumulator = std::move(workspace.scratch5);
                block_accumulator.meta.scale = block_scale;
            };

            std::size_t previous_term_q_count = 0;
            for (const auto &term : terms)
            {
                if (term.degree == 0)
                {
                    continue;
                }
                if (term.coefficient_plaintext.meta.q_count <
                    previous_term_q_count)
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: block terms are not in setup-time q_count order");
                }
                previous_term_q_count =
                    term.coefficient_plaintext.meta.q_count;

                const auto &basis = workspace.eval_mod_basis[term.degree];
                const GpuCiphertextData *basis_at_level = &basis;
                if (!(basis.meta.parms_id ==
                      term.coefficient_plaintext.meta.parms_id))
                {
                    if (basis.meta.q_count <=
                        term.coefficient_plaintext.meta.q_count)
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: coefficient plaintext is above its basis level");
                    }
                    drop_modulus(
                        basis,
                        workspace.scratch3,
                        term.coefficient_plaintext.meta.parms_id);
                    basis_at_level = &workspace.scratch3;
                }
                multiply_plain(
                    *basis_at_level,
                    term.coefficient_plaintext,
                    workspace.scratch2);
                accumulate_term(workspace.scratch2);
            }

            for (const auto &term : terms)
            {
                if (term.degree != 0)
                {
                    continue;
                }
                if (block_accumulator.empty())
                {
                    multiply_scalar(
                        workspace.eval_mod_basis[1],
                        0,
                        block_accumulator);
                    block_scale = term.coefficient_plaintext.meta.scale;
                    block_accumulator.meta.scale = block_scale;
                }
                if (!(block_accumulator.meta.parms_id ==
                      term.coefficient_plaintext.meta.parms_id))
                {
                    if (block_accumulator.meta.q_count <=
                        term.coefficient_plaintext.meta.q_count)
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: constant plaintext is above block level");
                    }
                    drop_modulus(
                        block_accumulator,
                        workspace.scratch3,
                        term.coefficient_plaintext.meta.parms_id);
                    block_accumulator = std::move(workspace.scratch3);
                }
                add_plain(
                    block_accumulator,
                    term.coefficient_plaintext,
                    workspace.scratch4);
                block_accumulator = std::move(workspace.scratch4);
                block_accumulator.meta.scale = block_scale;
            }

            if (block_accumulator.empty())
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: polynomial block has no evaluable terms");
            }

            for (std::uint32_t i = 0; i < rescale_count; ++i)
            {
                rescale(block_accumulator, workspace.scratch2);
                block_accumulator = std::move(workspace.scratch2);
            }
            if (expected_output_scale > 0.0)
            {
                block_accumulator.meta.scale = expected_output_scale;
            }
            block_output = std::move(block_accumulator);
        };

    auto add_aligned =
        [&](const GpuCiphertextData &left_input,
            const GpuCiphertextData &right_input,
            double expected_output_scale,
            GpuCiphertextData &output) {
            const GpuCiphertextData *left = &left_input;
            const GpuCiphertextData *right = &right_input;
            if (!(left->meta.parms_id == right->meta.parms_id))
            {
                if (left->meta.q_count > right->meta.q_count)
                {
                    drop_modulus(*left, workspace.scratch3, right->meta.parms_id);
                    left = &workspace.scratch3;
                }
                else if (right->meta.q_count > left->meta.q_count)
                {
                    drop_modulus(*right, workspace.scratch4, left->meta.parms_id);
                    right = &workspace.scratch4;
                }
                else
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: combine parms_id mismatch");
                }
            }
            add(*left, *right, workspace.scratch5);
            if (expected_output_scale > 0.0)
            {
                workspace.scratch5.meta.scale = expected_output_scale;
            }
            output = std::move(workspace.scratch5);
        };

    GpuCiphertextData accumulator;
    if (polynomial_blocks.empty())
    {
        evaluate_term_block(
            polynomial_terms,
            bootstrap_data.eval_mod.polynomial_rescale_count,
            target_scale,
            accumulator);
    }
    else
    {
        std::uint32_t maximum_node =
            bootstrap_data.eval_mod.polynomial_result_node;
        maximum_node = std::max(
            maximum_node,
            static_cast<std::uint32_t>(polynomial_blocks.size() - 1));
        for (const auto &combine : polynomial_combine_steps)
        {
            maximum_node = std::max(maximum_node, combine.output_node);
            maximum_node = std::max(maximum_node, combine.quotient_node);
            maximum_node = std::max(maximum_node, combine.remainder_node);
        }
        if (workspace.eval_mod_nodes.size() <= maximum_node)
        {
            workspace.eval_mod_nodes.resize(
                static_cast<std::size_t>(maximum_node) + 1);
        }

        std::vector<bool> node_available(
            static_cast<std::size_t>(maximum_node) + 1,
            false);
        for (std::size_t i = 0; i < polynomial_blocks.size(); ++i)
        {
            if (i >= node_available.size())
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: block node id exceeds schedule");
            }
            evaluate_term_block(
                polynomial_blocks[i].terms,
                polynomial_blocks[i].rescale_count,
                polynomial_blocks[i].output_scale,
                workspace.eval_mod_nodes[i]);
            node_available[i] = true;
        }

        for (const auto &combine : polynomial_combine_steps)
        {
            if (!node_available[combine.quotient_node] ||
                !node_available[combine.remainder_node] ||
                node_available[combine.output_node])
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: polynomial combine plan is not topologically valid");
            }
            multiply_relinearize_rescale(
                workspace.eval_mod_nodes[combine.quotient_node],
                workspace.eval_mod_basis[combine.basis_degree],
                combine.output_scale,
                workspace.scratch2);
            add_aligned(
                workspace.scratch2,
                workspace.eval_mod_nodes[combine.remainder_node],
                combine.output_scale,
                workspace.eval_mod_nodes[combine.output_node]);
            node_available[combine.output_node] = true;
        }

        const auto result_node =
            bootstrap_data.eval_mod.polynomial_result_node;
        if (!node_available[result_node])
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: polynomial result node is unavailable");
        }
        if (workspace.capture_eval_mod_trace)
        {
            multiply_scalar(
                workspace.eval_mod_nodes[result_node],
                1,
                accumulator);
            accumulator.meta.scale =
                workspace.eval_mod_nodes[result_node].meta.scale;
        }
        else
        {
            accumulator = std::move(workspace.eval_mod_nodes[result_node]);
        }
    }

    if (workspace.capture_eval_mod_trace)
    {
        multiply_scalar(
            accumulator,
            1,
            workspace.eval_mod_trace_polynomial_output);
        workspace.eval_mod_trace_polynomial_output.meta.scale =
            accumulator.meta.scale;
    }

    for (const auto &double_angle_plaintext : configured_double_angle_constants)
    {
        square(accumulator, workspace.scratch5);
        relinearize(workspace.scratch5, relin_keys, workspace.scratch2);
        add(workspace.scratch2, workspace.scratch2, workspace.scratch3);

        if (!double_angle_plaintext.empty())
        {
            if (!(workspace.scratch3.meta.parms_id ==
                  double_angle_plaintext.meta.parms_id))
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: double-angle plaintext level mismatch");
            }
            add_plain(
                workspace.scratch3,
                double_angle_plaintext,
                workspace.scratch4);
            rescale(workspace.scratch4, workspace.scratch5);
        }
        else
        {
            rescale(workspace.scratch3, workspace.scratch5);
        }
        accumulator = std::move(workspace.scratch5);
        if (workspace.capture_eval_mod_trace)
        {
            workspace.eval_mod_trace_double_angle_outputs.emplace_back();
            multiply_scalar(
                accumulator,
                1,
                workspace.eval_mod_trace_double_angle_outputs.back());
            workspace.eval_mod_trace_double_angle_outputs.back().meta.scale =
                accumulator.meta.scale;
        }
    }

    /*
     * The CPU high-precision evaluator may finish at a lower Q prefix because
     * its recursive evaluator performs additional level alignment. Setup can
     * record that observable output parms_id without moving the recursive
     * control flow into the GPU hot path. Dropping a Q suffix is a contiguous,
     * coefficient-parallel operation and preserves the evaluated CKKS value.
     */
    if (bootstrap_data.eval_mod.output_parms_id != parms_id_zero &&
        accumulator.meta.parms_id != bootstrap_data.eval_mod.output_parms_id)
    {
        if (accumulator.meta.q_count < bootstrap_data.eval_mod.output_q_count)
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: GPU polynomial schedule ended below the configured CPU output level");
        }
        drop_modulus(
            accumulator,
            workspace.scratch2,
            bootstrap_data.eval_mod.output_parms_id);
        accumulator = std::move(workspace.scratch2);
    }

    destination_ciphertext = std::move(accumulator);
    destination_ciphertext.meta.scale = source_ciphertext.meta.scale;
}

}  // namespace gpu
}  // namespace poseidon
