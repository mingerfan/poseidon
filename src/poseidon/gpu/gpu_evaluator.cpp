#include "poseidon/gpu/gpu_evaluator.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>
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

}  // namespace

GpuEvaluator::GpuEvaluator(const GpuParameterData &params)
    : params_(params),
      elementwise_handler_(params),
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

    auto left_view = left_ciphertext.make_const_view();
    auto right_view = right_ciphertext.make_const_view();
    auto destination_view = result.make_view();

    const auto &level_info = params_.get_level(left_ciphertext.meta.parms_id);

    elementwise_handler_.add_ciphertext(
        destination_view,
        left_view,
        right_view,
        level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::sub(
    const GpuCiphertextData &left_ciphertext,
    const GpuCiphertextData &right_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // Same high-level structure as add(), but call subtraction handler.

    (void)left_ciphertext;
    (void)right_ciphertext;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::sub is not implemented yet");
}

void GpuEvaluator::negate(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // 1. Check source metadata.
    // 2. Prepare destination metadata and storage.
    // 3. Create views.
    // 4. Call elementwise_handler_.negate_ciphertext(...).

    (void)source_ciphertext;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::negate is not implemented yet");
}

void GpuEvaluator::add_plain(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // 1. Check ciphertext/plaintext semantic compatibility.
    // 2. Prepare destination metadata and storage.
    // 3. Create views.
    // 4. Call elementwise_handler_.add_plain_to_ciphertext(...).
    //
    // CKKS rule:
    // - plaintext is added only to c0.

    (void)source_ciphertext;
    (void)source_plaintext;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::add_plain is not implemented yet");
}

void GpuEvaluator::sub_plain(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // 1. Check ciphertext/plaintext semantic compatibility.
    // 2. Prepare destination metadata and storage.
    // 3. Create views.
    // 4. Call elementwise_handler_.sub_plain_from_ciphertext(...).
    //
    // CKKS rule:
    // - plaintext is subtracted only from c0.

    (void)source_ciphertext;
    (void)source_plaintext;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::sub_plain is not implemented yet");
}

void GpuEvaluator::multiply_plain(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // 1. Check ciphertext/plaintext semantic compatibility.
    // 2. Prepare destination metadata:
    //    - destination scale = ciphertext scale * plaintext scale.
    // 3. Prepare destination storage.
    // 4. Create views.
    // 5. Call elementwise_handler_.multiply_plain_with_ciphertext(...).

    (void)source_ciphertext;
    (void)source_plaintext;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::multiply_plain is not implemented yet");
}

void GpuEvaluator::ntt_fwd(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // 1. Check source is not already in NTT form.
    // 2. Prepare destination metadata with is_ntt_form = true.
    // 3. Prepare destination storage.
    // 4. Create views.
    // 5. Call ntt_handler_.forward_ciphertext(...).

    (void)source_ciphertext;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::ntt_fwd is not implemented yet");
}

void GpuEvaluator::ntt_inv(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // 1. Check source is in NTT form.
    // 2. Prepare destination metadata with is_ntt_form = false.
    // 3. Prepare destination storage.
    // 4. Create views.
    // 5. Call ntt_handler_.inverse_ciphertext(...).

    (void)source_ciphertext;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::ntt_inv is not implemented yet");
}

void GpuEvaluator::multiply(
    const GpuCiphertextData &left_ciphertext,
    const GpuCiphertextData &right_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // 1. Check ciphertext semantic compatibility.
    // 2. Prepare destination metadata:
    //    - destination component count = left_count + right_count - 1.
    //    - destination scale = left scale * right scale.
    // 3. Prepare destination storage.
    // 4. Create views.
    // 5. Call elementwise_handler_.multiply_ciphertext(...).
    //
    // Note:
    // - This corresponds to Cheddar-style Tensor operation,
    //   but must be implemented using Poseidon's polys[index] model.

    (void)left_ciphertext;
    (void)right_ciphertext;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::multiply is not implemented yet");
}

void GpuEvaluator::square(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // 1. Check source metadata.
    // 2. Prepare destination metadata and storage.
    // 3. Create views.
    // 4. Call elementwise_handler_.square_ciphertext(...).

    (void)source_ciphertext;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::square is not implemented yet");
}

void GpuEvaluator::rescale(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // 1. Determine destination level and scale.
    // 2. Prepare destination metadata and storage.
    // 3. Query source and destination level info from params_.
    // 4. Create views.
    // 5. Call modswitch_handler_.rescale_ciphertext(...).

    (void)source_ciphertext;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::rescale is not implemented yet");
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

void GpuEvaluator::relinearize(
    const GpuCiphertextData &source_ciphertext,
    const GpuRelinKeysData &relin_keys,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // GPU relinearization.
    //
    // Current framework decision:
    // - keep this as a top-level TODO for now;
    // - introduce a dedicated key-switch handler only after Poseidon key-switch
    //   layout is fully mapped.
    //
    // Expected future logic:
    // - check source component count;
    // - check relin key compatibility;
    // - prepare destination with two components;
    // - run GPU key-switch pipeline.

    (void)source_ciphertext;
    (void)relin_keys;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::relinearize is not implemented yet");
}

void GpuEvaluator::rotate(
    const GpuCiphertextData &source_ciphertext,
    int step,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // GPU rotation.
    //
    // Current framework decision:
    // - keep this as a top-level TODO for now;
    // - introduce a dedicated key-switch handler only after Poseidon key-switch
    //   layout is fully mapped.
    //
    // Expected future logic:
    // - select Galois key by rotation step;
    // - apply automorphism/permutation;
    // - run GPU key-switch pipeline.

    (void)source_ciphertext;
    (void)step;
    (void)galois_keys;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::rotate is not implemented yet");
}

void GpuEvaluator::conjugate(
    const GpuCiphertextData &source_ciphertext,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &destination_ciphertext) const
{
    // TODO:
    // GPU conjugation.
    //
    // Current framework decision:
    // - keep this as a top-level TODO for now;
    // - introduce a dedicated key-switch handler only after Poseidon key-switch
    //   layout is fully mapped.

    (void)source_ciphertext;
    (void)galois_keys;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::conjugate is not implemented yet");
}

}  // namespace gpu
}  // namespace poseidon
