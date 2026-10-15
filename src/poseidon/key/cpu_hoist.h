#pragma once

#include "poseidon/parameters_literal.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace poseidon
{
// One source c1, decomposed into HYBRID digits in [digit][limb][coefficient]
// order. Q and P are stored separately, both in NTT form.
struct CpuHoistedDecomposition
{
    parms_id_type parms_id = parms_id_zero;
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;
    std::size_t dnum = 0;
    std::vector<std::uint64_t> digits_q_ntt;
    std::vector<std::uint64_t> digits_p_ntt;
};

struct CpuHoistOperationCounts
{
    std::size_t source_decompose_count = 0;
    std::size_t key_switch_count = 0;
    std::size_t moddown_count = 0; // ciphertexts, each with two components
    std::size_t fallback_rotation_count = 0;
};

// Caller-owned scratch; retain between calls to reuse allocations. Every batch
// recomputes the source digits, including when parms_id is unchanged. Separate
// concurrent calls must use separate workspaces.
struct CpuHoistWorkspace
{
    CpuHoistedDecomposition source_hoist;
    std::vector<std::uint64_t> source_coeff_q;
    std::vector<std::uint64_t> permuted_q;
    std::vector<std::uint64_t> permuted_p;
    std::vector<std::uint64_t> accumulator_q; // [component][limb][coefficient]
    std::vector<std::uint64_t> accumulator_p;
    std::vector<std::uint64_t> product;
    CpuHoistOperationCounts last_counts;

    // Retained vector capacity, excluding outputs and MemoryPool temporaries.
    std::size_t allocated_bytes() const noexcept
    {
        return sizeof(std::uint64_t) *
               (source_hoist.digits_q_ntt.capacity() + source_hoist.digits_p_ntt.capacity() +
                source_coeff_q.capacity() + permuted_q.capacity() + permuted_p.capacity() +
                accumulator_q.capacity() + accumulator_p.capacity() + product.capacity());
    }
};
} // namespace poseidon
