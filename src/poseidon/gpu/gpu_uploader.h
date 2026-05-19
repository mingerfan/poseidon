#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_key.h"

namespace poseidon
{

class Ciphertext;
class Plaintext;
class PoseidonContext;
class RelinKeys;
class GaloisKeys;

namespace gpu
{

/**
 * @brief CPU/GPU conversion helper.
 */
class GpuUploader
{
public:
    /**
     * @brief Upload CPU Ciphertext to GPU.
     *
     * - Read shape and metadata from CPU Ciphertext.
     * - Allocate GpuCiphertextData according to placement.
     * - Convert CPU uint64_t residues to GPU uint32_t residues.
     * - Copy data to GPU fields.
     */
    static GpuCiphertextData upload_ciphertext(
        const Ciphertext &src,
        int device_id);

    /**
     * @brief Download GPU Ciphertext to CPU.
     *
     * - Resize CPU Ciphertext.
     * - Convert GPU uint32_t residues to CPU uint64_t residues.
     * - Restore metadata.
     */
    static void download_ciphertext(
        const GpuCiphertextData &src,
        Ciphertext &dst);

    static void download_ciphertext(
        const GpuCiphertextData &src,
        Ciphertext &dst,
        const PoseidonContext &context);

    /**
     * @brief Upload CPU Plaintext to GPU.
     *
     * - Support the same shard/placement model as ciphertext.
     */
    static GpuPlaintextData upload_plaintext(
        const Plaintext &src,
        int device_id);

    /**
     * @brief Download GPU Plaintext to CPU.
     */
    static void download_plaintext(
        const GpuPlaintextData &src,
        Plaintext &dst);

    static void download_plaintext(
        const GpuPlaintextData &src,
        Plaintext &dst,
        const PoseidonContext &context);

    /**
     * @brief Upload CPU relinearization keys to GPU.
     *
     * - Preserve key-switching key layout;
     * - support multi-GPU shard placement.
     */
    static GpuRelinKeysData upload_relin_keys(
        const RelinKeys &src,
        int device_id);

    /**
     * @brief Upload CPU Galois keys to GPU.
     *
     * - Preserve Galois element to key mapping;
     * - support multi-GPU shard placement.
     */
    static GpuGaloisKeysData upload_galois_keys(
        const GaloisKeys &src,
        int device_id);
};

}  // namespace gpu
}  // namespace poseidon
