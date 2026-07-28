#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_linear_transform.h"
#include "poseidon/gpu/gpu_memory.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace poseidon
{
namespace gpu
{

enum class GpuLinearTransformMode : std::uint8_t
{
    ClassicBsgs = 0,
    SingleHoistBsgs = 1,
    DoubleHoistBsgs = 2,
};

struct GpuDoubleHoistTerm
{
    std::uint32_t giant_index = 0;
    std::uint32_t baby_index = 0;
    std::uint32_t diagonal_id = 0;
};

/**
 * Compact, setup-time BSGS schedule. Device arrays are immutable in the hot
 * path and retain the same ordering as terms/group_term_offsets.
 */
struct GpuDoubleHoistMatrixPlan
{
    std::uint32_t log_slots = 0;
    std::uint32_t n1 = 0;
    std::uint32_t n2 = 0;
    std::uint32_t rescale_count = 1;

    std::vector<int> baby_steps;
    std::vector<int> giant_steps;
    std::vector<std::uint32_t> group_term_offsets;
    std::vector<GpuDoubleHoistTerm> terms;

    DeviceVector<const GpuWord *> diagonal_q_ptrs;
    DeviceVector<const GpuWord *> diagonal_p_ptrs;
    DeviceVector<std::uint32_t> term_baby_indices;
    DeviceVector<std::uint32_t> group_term_offsets_device;
};

/**
 * One diagonal matrix whose plaintexts have exact Q->P extensions. Q and P
 * residues remain in one [Q|P] plaintext allocation, while kernels consume
 * separate Q/P pointers.
 */
struct GpuMatrixPlainQP
{
    std::uint32_t log_slots = 0;
    std::uint32_t n1 = 0;
    std::uint32_t level = 0;
    double scale = 1.0;
    std::vector<int> rot_index;
    std::map<int, GpuPlaintextData> plain_vec_qp;
    GpuDoubleHoistMatrixPlan plan;
};

class GpuLinearMatrixGroupQP
{
public:
    std::vector<GpuMatrixPlainQP> &data() noexcept { return matrices_; }
    const std::vector<GpuMatrixPlainQP> &data() const noexcept { return matrices_; }

    std::vector<int> &rot_index() noexcept { return rotate_index_; }
    const std::vector<int> &rot_index() const noexcept { return rotate_index_; }

    std::uint32_t step() const noexcept { return scalar_step_; }
    void set_step(std::uint32_t step) noexcept { scalar_step_ = step; }

private:
    std::vector<GpuMatrixPlainQP> matrices_;
    std::vector<int> rotate_index_;
    std::uint32_t scalar_step_ = 0;
};

/**
 * Hoisted HYBRID decomposition in [digit][limb][coeff] order.
 */
struct GpuHoistedDecomposition
{
    parms_id_type parms_id{};
    int device_id = 0;
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;
    std::size_t dnum = 0;

    DeviceVector<GpuWord> source_intt_q;
    DeviceVector<GpuWord> digits_q_ntt;
    DeviceVector<GpuWord> digits_p_ntt;
};

/**
 * Split Q/P ciphertext batch, laid out as
 * [batch][component][limb][coeff].
 */
struct GpuQPCiphertextBuffer
{
    int device_id = 0;
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;
    std::size_t batch_count = 0;

    DeviceVector<GpuWord> q;
    DeviceVector<GpuWord> p;

    void ensure_capacity(
        int requested_device_id,
        std::size_t requested_degree,
        std::size_t requested_q_count,
        std::size_t requested_p_count,
        std::size_t requested_batch_count);

    std::size_t q_words_per_batch() const noexcept
    {
        return 2 * q_count * degree;
    }

    std::size_t p_words_per_batch() const noexcept
    {
        return 2 * p_count * degree;
    }

    GpuWord *q_component(std::size_t batch, std::size_t component);
    GpuWord *p_component(std::size_t batch, std::size_t component);
    const GpuWord *q_component(std::size_t batch, std::size_t component) const;
    const GpuWord *p_component(std::size_t batch, std::size_t component) const;
};

struct GpuQCiphertextBatchBuffer
{
    int device_id = 0;
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t batch_count = 0;
    DeviceVector<GpuWord> q;

    void ensure_capacity(
        int requested_device_id,
        std::size_t requested_degree,
        std::size_t requested_q_count,
        std::size_t requested_batch_count);

    GpuWord *q_component(std::size_t batch, std::size_t component);
    const GpuWord *q_component(
        std::size_t batch,
        std::size_t component) const;
};

/**
 * Reusable scratch used by staged HYBRID KeySwitch. No allocation occurs when
 * a later call fits the recorded capacity.
 */
struct GpuHybridKeySwitchWorkspace
{
    int device_id = 0;
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;

    DeviceVector<GpuWord> permuted_digit_q;
    DeviceVector<GpuWord> permuted_digit_p;
    DeviceVector<GpuWord> p_coeff0;
    DeviceVector<GpuWord> p_coeff1;
    DeviceVector<GpuWord> converted_q0;
    DeviceVector<GpuWord> converted_q1;

    void ensure_capacity(
        int requested_device_id,
        std::size_t requested_degree,
        std::size_t requested_q_count,
        std::size_t requested_p_count);
};

struct GpuDoubleHoistOperationCounts
{
    std::size_t source_decompose_count = 0;
    std::size_t outer_decompose_count = 0;
    std::size_t keymul_count = 0;
    std::size_t inner_moddown_count = 0;
    std::size_t outer_moddown_count = 0;
    std::size_t qp_pmult_count = 0;
    std::size_t permute_count = 0;
    std::size_t baby_tile_count = 0;
    std::size_t workspace_peak_bytes = 0;
};

struct GpuDoubleHoistWorkspace
{
    GpuHoistedDecomposition source_hoist;
    GpuHoistedDecomposition outer_hoist;
    GpuHybridKeySwitchWorkspace keyswitch;

    GpuQPCiphertextBuffer baby_tile;
    GpuQPCiphertextBuffer group_accumulators;
    GpuQPCiphertextBuffer outer_accumulator;
    GpuQCiphertextBatchBuffer inner_q_batch;

    DeviceVector<GpuWord> batch_p_coeff;
    DeviceVector<GpuWord> batch_converted_q;
    DeviceVector<GpuWord> outer_reduced_p;
    std::vector<GpuHoistedDecomposition> outer_group_hoists;

    GpuCiphertextData inner_q;
    GpuCiphertextData result_q;

    std::size_t baby_tile_size = 4;
    std::size_t max_workspace_bytes = 0;
    GpuDoubleHoistOperationCounts last_counts;
    std::vector<GpuDoubleHoistOperationCounts> matrix_counts;
};

GpuLinearTransformMode gpu_linear_transform_mode_from_environment(
    GpuLinearTransformMode fallback);

}  // namespace gpu
}  // namespace poseidon
