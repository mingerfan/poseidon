#pragma once
#include "poseidon/gpu/gpu_memory.h"
#include <cstddef>
#include <cstdint>

namespace bert::compact::detail {
struct MacTerm {
    const poseidon::gpu::GpuWord *cipher0;
    const poseidon::gpu::GpuWord *cipher1;
    const poseidon::gpu::GpuWord *plaintext;
    std::uint32_t period;
    std::uint32_t reserved = 0;
};
void launch_compact_group_mac(
    poseidon::gpu::GpuWord *output0,
    poseidon::gpu::GpuWord *output1,
    const MacTerm *terms, std::size_t term_count,
    const poseidon::gpu::GpuWord *q_primes,
    const poseidon::gpu::GpuWide *q_barrett,
    std::size_t degree, std::size_t q_count);
} // namespace bert::compact::detail
