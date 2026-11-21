#pragma once

#include "poseidon/basics/util/ntt.h"
#include "poseidon/gpu/gpu_memory.h"

#include <vector>

namespace poseidon::gpu
{
namespace detail { struct WideBvLimb; }

// In-place negacyclic NTT on contiguous [RNS limb][coefficient] storage.
// Forward output uses Poseidon's bit-reversed order. Inputs must be reduced.
class GpuWideNtt
{
public:
    GpuWideNtt(int log_degree, const std::vector<Modulus> &moduli, int device_id);

    void forward(DeviceVector<std::uint64_t> &values) const;
    void inverse(DeviceVector<std::uint64_t> &values) const;
    void forward_accumulate(DeviceVector<std::uint64_t> &values,
                            const std::uint64_t *key, DeviceVector<std::uint64_t> &sum,
                            const DeviceVector<detail::WideBvLimb> &limbs) const;

    // Transform a buffer containing consecutive limbs of the plan.
    void forward_prefix(DeviceVector<std::uint64_t> &values) const;
    void inverse_limb(DeviceVector<std::uint64_t> &values, std::size_t limb) const;

    // Coefficient-domain extension of the centered first-limb representative
    // into this plan's complete RNS base. Source and destination must be distinct.
    void extend_centered_q0(const DeviceVector<std::uint64_t> &source,
                            DeviceVector<std::uint64_t> &destination) const;

private:
    void transform(DeviceVector<std::uint64_t> &values, bool inverse, std::size_t first_limb,
                   const std::uint64_t *key = nullptr, std::uint64_t *sum = nullptr,
                   const detail::WideBvLimb *limbs = nullptr) const;

    int device_id_;
    std::size_t degree_;
    std::size_t limb_count_;
    DeviceVector<std::uint64_t> moduli_;
    DeviceVector<util::MultiplyUIntModOperand> forward_roots_;
    DeviceVector<util::MultiplyUIntModOperand> inverse_roots_;
    DeviceVector<util::MultiplyUIntModOperand> inverse_degrees_;
};

} // namespace poseidon::gpu
