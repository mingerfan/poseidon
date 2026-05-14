#include "poseidon/gpu/gpu_evaluator.h"

#include <stdexcept>

namespace poseidon
{
namespace gpu
{

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
    // TODO:
    // 1. Check FHE semantic validity:
    //    - same parms_id;
    //    - compatible scale;
    //    - same NTT form;
    //    - same degree and active limb count.
    //
    // 2. Prepare destination metadata:
    //    - destination metadata should follow add semantics;
    //    - destination component count should be max(left_count, right_count).
    //
    // 3. Prepare destination storage.
    //
    // 4. Create views:
    //    - left ciphertext const view;
    //    - right ciphertext const view;
    //    - destination ciphertext mutable view.
    //
    // 5. Query level information from params_.
    //
    // 6. Call elementwise_handler_.add_ciphertext(...).

    (void)left_ciphertext;
    (void)right_ciphertext;
    (void)destination_ciphertext;

    throw std::runtime_error("GpuEvaluator::add is not implemented yet");
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