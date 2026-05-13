#include "poseidon/gpu/gpu_evaluator.h"

#include <stdexcept>

namespace poseidon
{
namespace gpu
{

GpuEvaluator::GpuEvaluator(const GpuParameterData &params)
    : params_(params)
{}

void GpuEvaluator::add(
    const GpuCiphertextData &a,
    const GpuCiphertextData &b,
    GpuCiphertextData &res) const
{
    // TODO:
    // GPU ciphertext addition.
    //
    // Expected steps:
    // - Check metadata compatibility.
    // - Query level info from params_.
    // - Create const views for a/b and mutable view for res.
    // - Launch elementwise add kernel.

    (void)a;
    (void)b;
    (void)res;

    throw std::runtime_error("GpuEvaluator::add is not implemented yet");
}

void GpuEvaluator::sub(
    const GpuCiphertextData &a,
    const GpuCiphertextData &b,
    GpuCiphertextData &res) const
{
    // TODO: GPU ciphertext subtraction.

    (void)a;
    (void)b;
    (void)res;

    throw std::runtime_error("GpuEvaluator::sub is not implemented yet");
}

void GpuEvaluator::negate(
    const GpuCiphertextData &a,
    GpuCiphertextData &res) const
{
    // TODO: GPU ciphertext negation.

    (void)a;
    (void)res;

    throw std::runtime_error("GpuEvaluator::negate is not implemented yet");
}

void GpuEvaluator::add_plain(
    const GpuCiphertextData &ct,
    const GpuPlaintextData &pt,
    GpuCiphertextData &res) const
{
    // TODO:
    // GPU add_plain.
    // For CKKS, plaintext is added to c0 only.

    (void)ct;
    (void)pt;
    (void)res;

    throw std::runtime_error("GpuEvaluator::add_plain is not implemented yet");
}

void GpuEvaluator::sub_plain(
    const GpuCiphertextData &ct,
    const GpuPlaintextData &pt,
    GpuCiphertextData &res) const
{
    // TODO:
    // GPU sub_plain.
    // For CKKS, plaintext is subtracted from c0 only.

    (void)ct;
    (void)pt;
    (void)res;

    throw std::runtime_error("GpuEvaluator::sub_plain is not implemented yet");
}

void GpuEvaluator::multiply_plain(
    const GpuCiphertextData &ct,
    const GpuPlaintextData &pt,
    GpuCiphertextData &res) const
{
    // TODO:
    // GPU multiply_plain.
    // Each ciphertext component should be multiplied by plaintext polynomial.

    (void)ct;
    (void)pt;
    (void)res;

    throw std::runtime_error("GpuEvaluator::multiply_plain is not implemented yet");
}

void GpuEvaluator::ntt_fwd(
    const GpuCiphertextData &ct,
    GpuCiphertextData &res) const
{
    // TODO:
    // GPU forward NTT for every active component and limb.

    (void)ct;
    (void)res;

    throw std::runtime_error("GpuEvaluator::ntt_fwd is not implemented yet");
}

void GpuEvaluator::ntt_inv(
    const GpuCiphertextData &ct,
    GpuCiphertextData &res) const
{
    // TODO:
    // GPU inverse NTT for every active component and limb.

    (void)ct;
    (void)res;

    throw std::runtime_error("GpuEvaluator::ntt_inv is not implemented yet");
}

void GpuEvaluator::multiply(
    const GpuCiphertextData &a,
    const GpuCiphertextData &b,
    GpuCiphertextData &res) const
{
    // TODO:
    // GPU ciphertext-ciphertext multiplication.
    // Common size-2 x size-2 case:
    // res.c0 = a.c0 * b.c0
    // res.c1 = a.c0 * b.c1 + a.c1 * b.c0
    // res.c2 = a.c1 * b.c1

    (void)a;
    (void)b;
    (void)res;

    throw std::runtime_error("GpuEvaluator::multiply is not implemented yet");
}

void GpuEvaluator::square(
    const GpuCiphertextData &a,
    GpuCiphertextData &res) const
{
    // TODO:
    // GPU optimized square.

    (void)a;
    (void)res;

    throw std::runtime_error("GpuEvaluator::square is not implemented yet");
}

void GpuEvaluator::rescale(
    const GpuCiphertextData &ct,
    GpuCiphertextData &res) const
{
    // TODO:
    // GPU rescale.
    // Should support 32-bit physical primes and logical multi-prime dropping.

    (void)ct;
    (void)res;

    throw std::runtime_error("GpuEvaluator::rescale is not implemented yet");
}

void GpuEvaluator::rescale_dynamic(
    const GpuCiphertextData &ct,
    GpuCiphertextData &res,
    double min_scale) const
{
    // TODO:
    // GPU dynamic/logical rescale.

    (void)ct;
    (void)res;
    (void)min_scale;

    throw std::runtime_error("GpuEvaluator::rescale_dynamic is not implemented yet");
}

void GpuEvaluator::drop_modulus(
    const GpuCiphertextData &ct,
    GpuCiphertextData &res,
    parms_id_type target_parms_id) const
{
    // TODO:
    // GPU drop modulus.

    (void)ct;
    (void)res;
    (void)target_parms_id;

    throw std::runtime_error("GpuEvaluator::drop_modulus is not implemented yet");
}

void GpuEvaluator::relinearize(
    const GpuCiphertextData &ct,
    const GpuRelinKeysData &relin_keys,
    GpuCiphertextData &res) const
{
    // TODO:
    // GPU relinearization.
    // Requires GPU key-switching pipeline.

    (void)ct;
    (void)relin_keys;
    (void)res;

    throw std::runtime_error("GpuEvaluator::relinearize is not implemented yet");
}

void GpuEvaluator::rotate(
    const GpuCiphertextData &ct,
    int step,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &res) const
{
    // TODO:
    // GPU rotation.
    // Requires Galois permutation and key switching.

    (void)ct;
    (void)step;
    (void)galois_keys;
    (void)res;

    throw std::runtime_error("GpuEvaluator::rotate is not implemented yet");
}

void GpuEvaluator::conjugate(
    const GpuCiphertextData &ct,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &res) const
{
    // TODO:
    // GPU conjugation.
    // Requires conjugation permutation and key switching.

    (void)ct;
    (void)galois_keys;
    (void)res;

    throw std::runtime_error("GpuEvaluator::conjugate is not implemented yet");
}

}  // namespace gpu
}  // namespace poseidon