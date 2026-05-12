#include "poseidon/gpu/gpu_evaluator.h"

#include <stdexcept>

namespace poseidon
{
namespace gpu
{

void GpuEvaluator::add(
    const GpuCiphertextData &a,
    const GpuCiphertextData &b,
    GpuCiphertextData &res) const
{
    // TODO:
    // 1. Check metadata:
    //    - a.meta.parms_id == b.meta.parms_id
    //    - a.meta.is_ntt_form == b.meta.is_ntt_form
    //    - scale approximately equal
    //    - same component count, degree, q_count
    //
    // 2. Allocate res if empty.
    //
    // 3. Create views:
    //    auto av = a.make_const_view();  // later add const version
    //    auto bv = b.make_const_view();
    //    auto rv = res.make_view();
    //
    // 4. Launch add kernel.
    //
    // Current stage: interface only.

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
    // TODO:
    // Same structure as add, but launch subtraction kernel.

    (void)a;
    (void)b;
    (void)res;

    throw std::runtime_error("GpuEvaluator::sub is not implemented yet");
}

void GpuEvaluator::negate(
    const GpuCiphertextData &a,
    GpuCiphertextData &res) const
{
    // TODO:
    // 1. Check metadata.
    // 2. Allocate res if needed.
    // 3. Launch negation kernel.

    (void)a;
    (void)res;

    throw std::runtime_error("GpuEvaluator::negate is not implemented yet");
}

}  // namespace gpu
}  // namespace poseidon