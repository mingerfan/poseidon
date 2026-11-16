#pragma once
// Pure CPU setup planning shared by the engine and a dependency-free test.
#include <algorithm>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <vector>

namespace bert::attention::detail {
inline std::vector<int> qk_rotation_steps(bool paired, bool packed_inputs = false) {
    if (paired && packed_inputs)
        throw std::invalid_argument("QK pair_products and pack_inputs are mutually exclusive");
    std::set<int> steps;
    for (int b = 1; b < 8; ++b) { steps.insert(256 * b); steps.insert(-b); }
    for (int g = 1; g < 16; ++g) { steps.insert(-2048 * g); steps.insert(2040 * g); }
    for (int shift = 1; shift < 64; shift *= 2) steps.insert(shift);
    if (paired || packed_inputs) { steps.insert(-64); steps.insert(63); }
    if (packed_inputs) steps.insert(192);
    return {steps.begin(), steps.end()};
}
inline std::vector<int> qk_binary_rotation_steps(bool paired, bool packed_inputs = false) {
    std::set<int> steps;
    for (int step : qk_rotation_steps(paired, packed_inputs)) {
        const int sign = step < 0 ? -1 : 1;
        unsigned value = static_cast<unsigned>(step < 0 ? -step : step);
        for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
            if (value & 1) steps.insert(sign * static_cast<int>(bit));
    }
    return {steps.begin(), steps.end()};
}
struct KeyPreparationPlan {
    std::size_t storage_q_count = 0;
    std::set<int> rotation_steps;
    bool needs_upload = false;
};
inline KeyPreparationPlan plan_key_preparation(std::size_t current_q_count,
        const std::set<int> &current_steps, std::size_t requested_q_count,
        const std::vector<int> &requested_steps) {
    if (!requested_q_count || requested_steps.empty())
        throw std::invalid_argument("QK key setup requires nonzero Q count and rotations");
    KeyPreparationPlan plan;
    plan.storage_q_count = std::max(current_q_count, requested_q_count);
    plan.rotation_steps = current_steps;
    for (int step : requested_steps) {
        if (!step) throw std::invalid_argument("QK key setup cannot request rotation zero");
        plan.rotation_steps.insert(step);
    }
    plan.needs_upload = !current_q_count || plan.storage_q_count != current_q_count ||
                        plan.rotation_steps != current_steps;
    return plan;
}
} // namespace bert::attention::detail
