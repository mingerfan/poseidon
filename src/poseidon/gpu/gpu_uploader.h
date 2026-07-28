#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_linear_transform.h"
#include "poseidon/gpu/gpu_evaluator.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace poseidon
{

class Ciphertext;
class Plaintext;
class PoseidonContext;
class RelinKeys;
class GaloisKeys;
class MatrixPlain;
class LinearMatrixGroup;
class CKKSEncoder;
class EvalModPoly;
class Polynomial;

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
     * @brief Upload CPU Ciphertext with a custom single-device shard template.
     *
     * The template describes one ciphertext component. field_index and
     * field_offset in the template are ignored and assigned by the allocator.
     */
    static GpuCiphertextData upload_ciphertext(
        const Ciphertext &src,
        int device_id,
        const std::vector<GpuPolyShard> &shard_template);

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
     * @brief Upload one CPU pre-generated diagonal matrix.
     *
     * This copies MatrixPlain::plain_vec plaintext diagonals to GPU. Matrix
     * generation and CPU encoding are intentionally outside the GPU timed path.
     */
    static GpuMatrixPlain upload_matrix_plain(
        const MatrixPlain &src,
        int device_id);

    /**
     * @brief Upload one CPU pre-generated linear-matrix group.
     */
    static GpuLinearMatrixGroup upload_linear_matrix_group(
        const LinearMatrixGroup &src,
        int device_id);

    /**
     * Upload a linear matrix group with exact Q->P extended diagonals and a
     * compact immutable BSGS plan for double hoisting.
     */
    static GpuLinearMatrixGroupQP upload_linear_matrix_group_qp(
        const LinearMatrixGroup &src,
        const PoseidonContext &context,
        int device_id,
        std::uint32_t rescale_count = 1);

    /**
     * @brief Generate and upload the fixed high-precision EvalMod BSGS plan.
     *
     * Polynomial splitting, level simulation, plaintext encoding and all
     * host-to-device transfers are setup work. No ciphertext evaluation is
     * performed here. When relin_keys is non-null, all zero-copy key levels
     * required by the generated plan are validated during setup. An optional
     * expected_output_parms_id records the output level observed from the CPU
     * high-precision evaluator; the GPU finishes with a parallel Q-prefix
     * drop when its static plan naturally retains more limbs.
     */
    static GpuBootstrapData::EvalModData upload_eval_mod_high_precision(
        const EvalModPoly &eval_mod_poly,
        const CKKSEncoder &encoder,
        parms_id_type input_parms_id,
        int device_id,
        GpuRelinKeysData *relin_keys = nullptr,
        parms_id_type expected_output_parms_id = parms_id_zero,
        std::uint32_t logical_rescale_count = 1,
        const Polynomial *polynomial_override = nullptr,
        bool include_input_offset = true,
        std::uint32_t double_angle_override =
            std::numeric_limits<std::uint32_t>::max(),
        double double_angle_base_override =
            std::numeric_limits<double>::quiet_NaN(),
        double polynomial_output_scale_override =
            std::numeric_limits<double>::quiet_NaN(),
        bool fuse_leaf_terms_before_rescale = true);

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

    /**
     * Upload and inverse-pre-rotate every non-empty Galois key. Runtime
     * double-hoist KeyMult can then permute two outputs instead of all digits.
     */
    static GpuGaloisKeysData upload_double_hoist_galois_keys(
        const GaloisKeys &src,
        int device_id);

    /**
     * @brief Validate setup-time zero-copy views over one [Q_storage | P] key.
     *
     * No device allocation or device-to-device copy is performed. At runtime
     * KeySwitch reads the active Q prefix and addresses P through its fixed
     * offset in the original full-level key allocation.
     */
    static void prepare_key_views_for_q_counts(
        const GpuEvaluationKeyData &keys,
        const std::vector<std::size_t> &q_counts);
};

}  // namespace gpu
}  // namespace poseidon
