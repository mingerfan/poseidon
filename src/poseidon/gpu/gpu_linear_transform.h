#pragma once

#include "poseidon/gpu/gpu_plaintext.h"

#include <cstdint>
#include <map>
#include <vector>

namespace poseidon
{
namespace gpu
{

/**
 * @brief GPU-resident form of MatrixPlain.
 *
 * Matrix generation stays on the CPU side. Each encoded plaintext diagonal is
 * uploaded once into this structure, then GPU linear transforms consume these
 * device-resident plaintexts without re-encoding or host transfer.
 */
struct GpuMatrixPlain
{
    std::uint32_t log_slots = 0;
    std::uint32_t n1 = 0;
    std::uint32_t level = 0;
    double scale = 1.0;
    std::vector<int> rot_index;
    std::map<int, GpuPlaintextData> plain_vec;
};

/**
 * One independently schedulable plaintext product in a loose BSGS group.
 *
 * The input is Rotate(source, baby_step), with baby_step zero denoting the
 * original source.  The output remains an ordinary Q-basis ciphertext.
 */
struct GpuLooseBsgsTerm
{
    int baby_step = 0;
    int diagonal_index = 0;
};

/**
 * One independently reducible giant group.  After its term products are
 * summed, giant_step is applied to the ordinary Q-basis group result.
 */
struct GpuLooseBsgsGroup
{
    int giant_step = 0;
    std::vector<GpuLooseBsgsTerm> terms;
};

/**
 * Host-side, device-independent BSGS task graph used by the opt-in
 * loose-coupled path.  No lifted-QP object crosses a task boundary.
 */
struct GpuLooseBsgsPlan
{
    std::vector<int> baby_steps;
    std::vector<GpuLooseBsgsGroup> groups;
};

GpuLooseBsgsPlan make_gpu_loose_bsgs_plan(
    const GpuMatrixPlain &matrix);

/**
 * @brief GPU-resident form of LinearMatrixGroup.
 */
class GpuLinearMatrixGroup
{
public:
    GpuLinearMatrixGroup() = default;

    std::vector<GpuMatrixPlain> &data() noexcept { return matrices_; }
    const std::vector<GpuMatrixPlain> &data() const noexcept { return matrices_; }

    std::vector<int> &rot_index() noexcept { return rotate_index_; }
    const std::vector<int> &rot_index() const noexcept { return rotate_index_; }

    std::uint32_t step() const noexcept { return scalar_step_; }
    void set_step(std::uint32_t step) noexcept { scalar_step_ = step; }

    double rescale_min_scale() const noexcept { return rescale_min_scale_; }
    void set_rescale_min_scale(double scale) noexcept { rescale_min_scale_ = scale; }

    std::vector<std::uint32_t> &rescale_counts() noexcept { return rescale_counts_; }
    const std::vector<std::uint32_t> &rescale_counts() const noexcept
    {
        return rescale_counts_;
    }

private:
    std::vector<GpuMatrixPlain> matrices_;
    std::vector<int> rotate_index_;
    std::uint32_t scalar_step_ = 0;
    double rescale_min_scale_ = 0.0;
    std::vector<std::uint32_t> rescale_counts_;
};

}  // namespace gpu
}  // namespace poseidon
