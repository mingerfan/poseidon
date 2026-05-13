#pragma once

#include "poseidon/parameters_literal.h"
#include "poseidon/gpu/gpu_memory.h"

#include <cstddef>
#include <vector>

namespace poseidon
{

class PoseidonContext;

namespace gpu
{

/**
 * @brief GPU-side information for one modulus-chain level.
 *
 * This is the GPU-side counterpart of Poseidon CrtContext::ContextData.
 * It should eventually contain all device-resident parameter tables needed by
 * GPU homomorphic operators.
 */
struct GpuLevelInfo
{
    parms_id_type parms_id{};

    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;

    int device_id = 0;

    /**
     * @brief Device-side q and p primes.
     *
     * Since GPU primes are expected to be below 32 bits, they are stored as
     * GpuWord.
     */
    DeviceVector<GpuWord> q_primes;
    DeviceVector<GpuWord> p_primes;

    /**
     * @brief TODO modular arithmetic constants.
     *
     * Later:
     * - Barrett constants;
     * - Montgomery constants;
     * - inverse primes;
     * - rescale constants.
     */
    DeviceVector<GpuWord> q_modulus_constants;
    DeviceVector<GpuWord> p_modulus_constants;

    /**
     * @brief TODO NTT tables.
     *
     * Later:
     * - forward NTT roots;
     * - inverse NTT roots;
     * - N inverse factors.
     */
    DeviceVector<GpuWord> ntt_tables;
    DeviceVector<GpuWord> intt_tables;
};

/**
 * @brief GPU-side parameter cache.
 *
 * This object should be built from PoseidonContext once.
 * After construction, GpuEvaluator should use this object instead of directly
 * depending on PoseidonContext.
 *
 * Current stage:
 * - Only defines the framework interface.
 * - Real table construction is TODO.
 */
class GpuParameterData
{
public:
    GpuParameterData() = default;

    explicit GpuParameterData(const PoseidonContext &context, int device_id = 0);

    /**
     * @brief Build GPU-side parameter tables from PoseidonContext.
     *
     * TODO:
     * - Traverse modulus-chain levels;
     * - copy q/p primes to GPU 32-bit arrays;
     * - prepare NTT/INTT tables;
     * - prepare reduction constants;
     * - prepare rescale/modswitch/key-switch tables.
     */
    void build_from_poseidon_context(const PoseidonContext &context, int device_id);

    /**
     * @brief Query level information by parms_id.
     *
     * TODO:
     * - Current framework can use linear search;
     * - later replace with hash map if needed.
     */
    const GpuLevelInfo &get_level(const parms_id_type &parms_id) const;

    bool empty() const;

private:
    std::vector<GpuLevelInfo> levels_;
};

}  // namespace gpu
}  // namespace poseidon