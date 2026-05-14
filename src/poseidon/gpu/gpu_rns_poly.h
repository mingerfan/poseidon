#pragma once

#include "poseidon/gpu/gpu_memory.h"

#include <cstddef>
#include <vector>

namespace poseidon
{
namespace gpu
{

/**
 * @brief Physical slice of one logical GPU RNS polynomial.
 *
 * A shard says:
 * - which GPU memory block stores this slice;
 * - which RNS limb range it covers;
 * - which coefficient range it covers.
 *
 * This structure is shared by ciphertext, plaintext, and key data.
 */
struct GpuPolyShard
{
    /**
     * @brief Index into the owner object's fields_ array.
     *
     * For example:
     * - GpuCiphertextData::fields_
     * - GpuPlaintextData::fields_
     * - GpuEvaluationKeyData::fields_
     */
    std::size_t field_index = 0;

    /**
     * @brief Offset inside the selected field buffer.
     *
     * First-stage simple layouts usually use 0.
     */
    std::size_t field_offset = 0;

    /**
     * @brief RNS limb range.
     *
     * Example:
     * - limb_begin = 0, limb_count = 4 means q0-q3.
     * - limb_begin = 4, limb_count = 4 means q4-q7.
     */
    std::size_t limb_begin = 0;
    std::size_t limb_count = 0;

    /**
     * @brief Coefficient range.
     *
     * First-stage simple layouts usually cover the full range:
     * - coeff_begin = 0
     * - coeff_count = degree
     */
    std::size_t coeff_begin = 0;
    std::size_t coeff_count = 0;
};

/**
 * @brief Logical GPU-side RNS polynomial.
 *
 * This is a general RNS polynomial descriptor.
 *
 * Usage:
 * - ciphertext: poly_id = 0/1/2 means c0/c1/c2;
 * - plaintext: usually only one polynomial, poly_id = 0;
 * - keys: poly_id indexes one key polynomial.
 *
 * This structure does not own GPU memory.
 * It only records which shards belong to this logical RNS polynomial.
 */
struct GpuRNSPoly
{
    std::size_t poly_id = 0;

    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;

    std::vector<GpuPolyShard> shards;
};

/**
 * @brief Mutable view of one physical shard.
 *
 * View objects are temporary and non-owning.
 * They are created before launching GPU handlers/kernels.
 */
struct GpuPolyShardView
{
    int device_id = 0;
    GpuWord *ptr = nullptr;

    std::size_t limb_begin = 0;
    std::size_t limb_count = 0;

    std::size_t coeff_begin = 0;
    std::size_t coeff_count = 0;
};

/**
 * @brief Const view of one physical shard.
 */
struct GpuConstPolyShardView
{
    int device_id = 0;
    const GpuWord *ptr = nullptr;

    std::size_t limb_begin = 0;
    std::size_t limb_count = 0;

    std::size_t coeff_begin = 0;
    std::size_t coeff_count = 0;
};

/**
 * @brief Mutable view of one logical RNS polynomial.
 */
struct GpuRNSPolyView
{
    std::size_t poly_id = 0;
    std::vector<GpuPolyShardView> shards;
};

/**
 * @brief Const view of one logical RNS polynomial.
 */
struct GpuConstRNSPolyView
{
    std::size_t poly_id = 0;
    std::vector<GpuConstPolyShardView> shards;
};

}  // namespace gpu
}  // namespace poseidon