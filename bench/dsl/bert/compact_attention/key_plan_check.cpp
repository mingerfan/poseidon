// Dependency-free mixed-spec setup planning. Does not link or initialize CUDA.
#include "prepared_qk.h"
#include "qk_key_plan.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
std::set<int> as_set(const std::vector<int> &steps) { return {steps.begin(), steps.end()}; }
}
int main() {
    try {
        using namespace bert::attention;
        using namespace bert::attention::detail;
        check(!QkSpec{}.pair_products && !QkSpec{}.pack_inputs, "both experimental modes must default OFF");
        const auto baseline = qk_rotation_steps(false), paired = qk_rotation_steps(true);
        const auto old = as_set(baseline), both = as_set(paired);
        check(old.size() == 50 && both.size() == 52, "direct step counts changed");
        check(std::includes(both.begin(), both.end(), old.begin(), old.end()), "paired keys must retain baseline");
        auto extras = both;
        for (int step : old) extras.erase(step);
        check(extras == std::set<int>{-64, 63}, "paired direct extras must be exactly -64/+63");
        check(!old.count(-64) && !old.count(63), "baseline unexpectedly requires paired keys");
        for (bool paired_mode : {false, true}) {
            const auto binary = as_set(qk_binary_rotation_steps(paired_mode));
            for (int step : qk_rotation_steps(paired_mode)) {
                const int sign = step < 0 ? -1 : 1;
                unsigned value = static_cast<unsigned>(step < 0 ? -step : step);
                int reconstructed = 0;
                for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
                    if (value & 1) {
                        const int piece = sign * static_cast<int>(bit);
                        check(binary.count(piece), "missing binary rotation part");
                        reconstructed += piece;
                    }
                check(reconstructed == step, "binary path changes the rotation");
            }
        }
        const auto first = plan_key_preparation(0, {}, 9, baseline);
        check(first.needs_upload && first.storage_q_count == 9 && first.rotation_steps == old,
              "baseline first setup must upload ONLY its old keys");
        const auto warm = plan_key_preparation(first.storage_q_count, first.rotation_steps, 7, baseline);
        check(!warm.needs_upload && warm.storage_q_count == 9 && warm.rotation_steps == old,
              "warm lower-Q baseline should reuse its owner");
        const auto add_paired = plan_key_preparation(first.storage_q_count, first.rotation_steps, 7, paired);
        check(add_paired.needs_upload && add_paired.storage_q_count == 9 && add_paired.rotation_steps == both,
              "lower-Q paired spec must reupload UNION without shrinking the existing Q owner");
        const auto add_old = plan_key_preparation(add_paired.storage_q_count, add_paired.rotation_steps, 7, baseline);
        check(!add_old.needs_upload && add_old.storage_q_count == 9 && add_old.rotation_steps == both,
              "baseline after paired must keep the superset");
        const auto grow = plan_key_preparation(add_old.storage_q_count, add_old.rotation_steps, 13, baseline);
        check(grow.needs_upload && grow.storage_q_count == 13 && grow.rotation_steps == both,
              "higher-Q baseline must retain previously requested paired keys");
        const auto paired_first = plan_key_preparation(0, {}, 7, paired);
        const auto old_higher = plan_key_preparation(paired_first.storage_q_count, paired_first.rotation_steps, 9, baseline);
        check(paired_first.needs_upload && old_higher.needs_upload && old_higher.storage_q_count == 9 &&
              old_higher.rotation_steps == both, "opposite mixed preparation order failed");
        const auto packed = qk_rotation_steps(false, true);
        const auto all = as_set(packed);
        check(all.size() == 53 && std::includes(all.begin(), all.end(), both.begin(), both.end()),
              "packed-input direct keys must retain baseline and paired keys");
        auto packed_extras = all;
        for (int step : old) packed_extras.erase(step);
        check(packed_extras == std::set<int>{-64, 63, 192}, "packed-input extras must be exactly -64/+63/+192");
        const auto binary_packed = as_set(qk_binary_rotation_steps(false, true));
        for (int step : packed) {
            const int sign = step < 0 ? -1 : 1;
            unsigned value = static_cast<unsigned>(step < 0 ? -step : step);
            int reconstructed = 0;
            for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
                if (value & 1) {
                    const int piece = sign * static_cast<int>(bit);
                    check(binary_packed.count(piece), "packed-input binary rotation part missing");
                    reconstructed += piece;
                }
            check(reconstructed == step, "packed-input binary path changes rotation");
        }
        const auto third_lower = plan_key_preparation(add_paired.storage_q_count, add_paired.rotation_steps, 5, packed);
        check(third_lower.needs_upload && third_lower.storage_q_count == 9 && third_lower.rotation_steps == all,
              "third mode at lower Q must upload +192 without shrinking Q");
        const auto packed_first = plan_key_preparation(0, {}, 5, packed);
        const auto paired_after_packed = plan_key_preparation(packed_first.storage_q_count, packed_first.rotation_steps, 7, paired);
        const auto old_after_packed = plan_key_preparation(paired_after_packed.storage_q_count,
            paired_after_packed.rotation_steps, 13, baseline);
        check(packed_first.needs_upload && paired_after_packed.needs_upload && old_after_packed.needs_upload &&
              old_after_packed.storage_q_count == 13 && old_after_packed.rotation_steps == all,
              "packed->paired->baseline must keep all53 keys while growing Q");
        const auto warm_paired = plan_key_preparation(old_after_packed.storage_q_count,
            old_after_packed.rotation_steps, 7, paired);
        check(!warm_paired.needs_upload && warm_paired.storage_q_count == 13 && warm_paired.rotation_steps == all,
              "existing third-mode union should satisfy lower-Q paired setup");
        // All six preparation orders must converge to the same highest-Q
        // owner and53-key union, not merely the two representative paths.
        const std::vector<std::vector<int>> mode_steps{baseline, paired, packed};
        const std::vector<std::size_t> mode_q{9, 7, 5};
        std::vector<int> order{0, 1, 2};
        std::size_t orders = 0;
        do {
            std::size_t active_q = 0;
            std::set<int> active_steps;
            for (int mode : order) {
                auto next = plan_key_preparation(active_q, active_steps, mode_q[mode], mode_steps[mode]);
                check(next.storage_q_count == std::max(active_q, mode_q[mode]), "mixed-order Q owner shrank");
                check(std::includes(next.rotation_steps.begin(), next.rotation_steps.end(),
                                    active_steps.begin(), active_steps.end()), "mixed-order keys disappeared");
                active_q = next.storage_q_count;
                active_steps = std::move(next.rotation_steps);
            }
            check(active_q == 9 && active_steps == all, "mixed-order final key union differs");
            ++orders;
        } while (std::next_permutation(order.begin(), order.end()));
        check(orders == 6, "not all three-mode preparation orders were checked");
        bool invalid_modes = false, invalid_binary_modes = false;
        try { (void)qk_rotation_steps(true, true); } catch (const std::invalid_argument &) { invalid_modes = true; }
        try { (void)qk_binary_rotation_steps(true, true); } catch (const std::invalid_argument &) { invalid_binary_modes = true; }
        check(invalid_modes && invalid_binary_modes, "conflicting modes must be explicitly rejected");
        bool invalid_q = false, invalid_step = false;
        try { (void)plan_key_preparation(0, {}, 0, paired); } catch (const std::invalid_argument &) { invalid_q = true; }
        try { (void)plan_key_preparation(0, {}, 9, {0}); } catch (const std::invalid_argument &) { invalid_step = true; }
        check(invalid_q && invalid_step, "malformed key setup must fail");
        std::cout << "PASS CPU-only QK key plan: default baseline50; paired52; packed-input53 "
                     "including exact -64/+63/+192; three-mode union/lower-Q reupload/highest-Q retention/"
                     "binary completeness/conflict rejection; no CUDA/context\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << "QK key planning: " << e.what() << '\n'; return 1; }
}
