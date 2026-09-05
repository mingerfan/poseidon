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
