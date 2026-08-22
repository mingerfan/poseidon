#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace poseidon
{
namespace gpu
{

/**
 * @brief Setup/runtime plan for removing as many physical Q primes as possible
 * while keeping the resulting CKKS scale above half of a logical target.
 *
 * active_moduli must contain the active Q prefix in ascending level order;
 * the last entry is the first modulus removed by CKKS rescale. The planner
 * uses the concrete modulus values instead of their nominal bit widths.
 */
struct GpuDynamicRescalePlan
{
    std::uint32_t rescale_count = 0;
    std::size_t output_q_count = 0;
    double input_scale = 0.0;
    double output_scale = 0.0;
    double target_scale = 0.0;
    double minimum_output_scale = 0.0;
};

inline GpuDynamicRescalePlan plan_gpu_dynamic_rescale(
    double input_scale,
    double target_scale,
    std::span<const std::uint64_t> active_moduli,
    bool require_rescale = false)
{
    if (!(input_scale > 0.0) || !std::isfinite(input_scale))
    {
        throw std::invalid_argument(
            "plan_gpu_dynamic_rescale: input scale must be finite and positive");
    }
    if (!(target_scale > 0.0) || !std::isfinite(target_scale))
    {
        throw std::invalid_argument(
            "plan_gpu_dynamic_rescale: target scale must be finite and positive");
    }
    if (active_moduli.empty())
    {
        throw std::invalid_argument(
            "plan_gpu_dynamic_rescale: active modulus prefix is empty");
    }

    long double output_scale = static_cast<long double>(input_scale);
    // Match EvaluatorCkksBase::rescale_dynamic exactly: min_scale names the
    // logical working target and half of it is the accepted physical floor.
    const long double minimum_scale =
        (static_cast<long double>(target_scale) + 1.0L) / 2.0L;
    std::size_t output_q_count = active_moduli.size();
    std::uint32_t rescale_count = 0;

    // Keep at least q0. A physical prime is removed only when the exact
    // post-rescale scale remains inside the declared logical precision range.
    while (output_q_count > 1)
    {
        const std::uint64_t modulus = active_moduli[output_q_count - 1];
        if (modulus == 0)
        {
            throw std::invalid_argument(
                "plan_gpu_dynamic_rescale: active modulus contains zero");
        }
        const long double next_scale =
            output_scale / static_cast<long double>(modulus);
        if (next_scale < minimum_scale)
        {
            break;
        }
        output_scale = next_scale;
        --output_q_count;
        ++rescale_count;
    }

    if (require_rescale && rescale_count == 0)
    {
        throw std::invalid_argument(
            "plan_gpu_dynamic_rescale: no physical modulus can be removed without violating the minimum scale");
    }

    const double planned_output_scale = static_cast<double>(output_scale);
    if (!(planned_output_scale > 0.0) ||
        !std::isfinite(planned_output_scale))
    {
        throw std::invalid_argument(
            "plan_gpu_dynamic_rescale: planned output scale is invalid");
    }

    return GpuDynamicRescalePlan{
        rescale_count,
        output_q_count,
        input_scale,
        planned_output_scale,
        target_scale,
        static_cast<double>(minimum_scale)};
}

}  // namespace gpu
}  // namespace poseidon
