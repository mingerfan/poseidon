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
 * @brief GPU-side parameter shard.
 *
 * Parameter data is not an RNS polynomial, so it does not use GpuRNSPoly.
 * However, it still needs to support multi-GPU placement.
 *
 * A parameter shard describes which parameter tables are available on one GPU
 * for a certain RNS limb range.
 */
struct GpuParameterShard
{
    int device_id = 0;

    std::size_t limb_begin = 0;
    std::size_t limb_count = 0;

    /**
     * @brief Device-side q and p primes.
     *
     * Depending on implementation choice:
     * - this shard may hold only the local limb range;
     * - or it may hold a full copy of all primes on this device.
     */
    DeviceVector<GpuWord> q_primes;
    DeviceVector<GpuWord> p_primes;
    DeviceVector<GpuWord> rns_primes;

    /**
     * @brief TODO modular arithmetic constants.
     *
     * Later:
     * - Barrett constants;
     * - Montgomery constants;
     * - inverse primes;
     * - rescale constants.
     */
    DeviceVector<GpuWide> q_modulus_constants;
    DeviceVector<GpuWide> p_modulus_constants;
    DeviceVector<GpuWide> rns_modulus_constants;

    /**
     * @brief Rescale constants for dropping the last q modulus.
     *
     * q_last and half_q_last are host-side scalar constants used as kernel
     * arguments. The two vectors are indexed by q limb i, for
     * i = 0..q_count-2:
     * - half_q_last_mod_q[i] = floor(q_last / 2) mod q_i;
     * - inv_q_last_mod_q[i] = q_last^{-1} mod q_i.
     */
    GpuWord q_last = 0;
    GpuWord half_q_last = 0;
    DeviceVector<GpuWord> half_q_last_mod_q;
    DeviceVector<GpuWord> inv_q_last_mod_q;

    /**
     * @brief NTT tables for q/p RNS limbs.
     *
     * Root tables are stored as [limb][degree] operands copied from Poseidon's
     * CPU NTT tables in [q limbs][p limbs] order. inv_degree_modulo stores
     * N^{-1} modulo the matching q or p prime for inverse NTT.
     */
    DeviceVector<GpuWord> ntt_tables;
    DeviceVector<GpuWord> intt_tables;
    DeviceVector<GpuWord> inv_degree_modulo;
};

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

    /**
     * @brief Per-device or per-limb parameter shards.
     *
     * This makes parameter placement compatible with ciphertext/plaintext/key
     * shard placement.
     */
    std::vector<GpuParameterShard> shards;
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
     * - prepare rescale/modswitch/key-switch tables;
     * - build per-device/per-limb GpuParameterShard objects.
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

    /**
     * @brief Query the next lower modulus-chain level.
     *
     * This is used by rescale/drop-modulus style operations. The next level is
     * the context level after dropping the current last q modulus.
     */
    const GpuLevelInfo &get_next_level(const parms_id_type &parms_id) const;

    bool empty() const;

private:
    std::vector<GpuLevelInfo> levels_;
};

}  // namespace gpu
}  // namespace poseidon
