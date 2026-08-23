#pragma once

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_double_hoist.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_linear_transform.h"
#include "poseidon/gpu/gpu_parameter.h"

#include "poseidon/gpu/gpu_elementwise_handler.h"
#include "poseidon/gpu/gpu_keyswitch_handler.h"
#include "poseidon/gpu/gpu_ntt_handler.h"
#include "poseidon/gpu/gpu_modswitch_handler.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace poseidon
{
namespace gpu
{

enum class GpuBootstrapSchedule : std::uint8_t
{
    Standard = 0,
    StCFirst = 1,
};

enum class GpuEvalModPolynomialBasis : std::uint8_t
{
    Monomial = 0,
    Chebyshev = 1,
};

struct GpuEvalModPolynomialTerm
{
    std::uint32_t degree = 0;
    GpuPlaintextData coefficient_plaintext;
};

/**
 * @brief One setup-time scheduled ciphertext-basis operation for EvalMod.
 *
 * For the monomial basis this computes X^output = X^left * X^right.
 * For the Chebyshev basis this computes
 * T_output = 2*T_left*T_right - T_correction. A zero correction degree means
 * subtracting the pre-uploaded plaintext constant one.
 *
 * Steps must be topologically sorted by CPU setup. The GPU runtime never
 * recursively discovers powers or branches on polynomial coefficients.
 */
struct GpuEvalModBasisStep
{
    std::uint32_t output_degree = 0;
    std::uint32_t left_degree = 0;
    std::uint32_t right_degree = 0;
    std::uint32_t correction_degree = 0;
    double pre_rescale_scale = 0.0;
    double output_scale = 0.0;
    std::uint32_t rescale_count = 1;

    bool align_left_operand = false;
    GpuPlaintextData operand_alignment_plaintext;
    double operand_alignment_pre_rescale_scale = 0.0;
    double operand_alignment_output_scale = 0.0;
    std::uint32_t operand_alignment_rescale_count = 1;
    GpuPlaintextData correction_alignment_plaintext;
    double correction_alignment_pre_rescale_scale = 0.0;
    std::uint32_t correction_alignment_rescale_count = 1;

    /*
     * Chebyshev only. For correction_degree==0 this is the constant one at
     * output_scale. Otherwise it is a plaintext one whose scale converts the
     * correction basis to output_scale without changing its decoded value.
     */
    GpuPlaintextData correction_plaintext;
};

/**
 * @brief One low-degree leaf of the setup-time polynomial split tree.
 */
struct GpuEvalModPolynomialBlock
{
    std::vector<GpuEvalModPolynomialTerm> terms;
    std::uint32_t rescale_count = 1;
    double output_scale = 0.0;
    std::size_t output_q_count = 0;
};

/**
 * @brief Fixed Q*T_k+R combine node used by the EvalMod BSGS tree.
 */
struct GpuEvalModPolynomialCombineStep
{
    std::uint32_t output_node = 0;
    std::uint32_t quotient_node = 0;
    std::uint32_t remainder_node = 0;
    std::uint32_t basis_degree = 0;
    double output_scale = 0.0;
    std::size_t output_q_count = 0;
    std::uint32_t quotient_rescale_count = 0;
    double quotient_output_scale = 0.0;
    std::uint32_t remainder_rescale_count = 0;
    double product_scale = 0.0;
    std::size_t product_q_count = 0;
    GpuPlaintextData product_scale_plaintext;
    double product_aligned_scale = 0.0;
    GpuPlaintextData remainder_scale_plaintext;
    double remainder_aligned_scale = 0.0;
};

/**
 * @brief Build the fixed EvalMod basis DAG during untimed CPU setup.
 *
 * requested_degrees normally contains the non-zero degrees of the uploaded
 * polynomial terms. The returned steps are topologically sorted and contain
 * no run-time level/scale decisions.
 */
std::vector<GpuEvalModBasisStep> make_gpu_eval_mod_basis_plan(
    GpuEvalModPolynomialBasis basis,
    const std::vector<std::uint32_t> &requested_degrees,
    std::uint32_t preferred_giant_stride = 0);

/**
 * @brief GPU-resident bootstrapping constants and precomputed objects.
 *
 * Matrix/plaintext/key generation stays on the CPU side. This structure only
 * stores objects that have already been uploaded to GPU memory and scalar
 * parameters needed by the GPU bootstrapping scheduler.
 */
struct GpuBootstrapData
{
    GpuBootstrapSchedule schedule = GpuBootstrapSchedule::Standard;
    GpuLinearTransformMode linear_transform_mode =
        GpuLinearTransformMode::ClassicBsgs;
    bool allow_environment_linear_transform_override = true;
    std::size_t double_hoist_baby_tile = 4;

    parms_id_type q0_parms_id{};
    double q0_over_message_ratio = 0.0;

    double raised_scale_override = 0.0;
    /* Logical scale assigned to the raised StC output before post-raise C2S. */
    double post_raise_c2s_input_scale = 0.0;
    std::uint64_t post_raise_integer_multiplier = 1;
    double post_raise_scale_multiplier = 1.0;
    GpuPlaintextData post_raise_plaintext;

    /* Logical CKKS scale restored after EvalMod and before SlotToCoeff. */
    double slot_to_coeff_input_scale = 0.0;

    struct EvalModData
    {
        /* Physical scale supplied by dynamic C2S; target_scale remains min_scale. */
        double input_scale = 0.0;
        double target_scale = 0.0;
        double output_scale = 0.0;
        parms_id_type output_parms_id{};
        std::size_t output_q_count = 0;
        GpuPlaintextData input_offset_plaintext;

        /*
         * Preferred high-precision EvalMod plan.
         *
         * Each term is already encoded/uploaded by CPU setup at the exact
         * parms_id and plaintext scale needed by the corresponding GPU basis
         * ciphertext. Runtime GPU EvalMod only generates ciphertext bases,
         * multiply_plain's these constants, and performs level-aligned sums.
         * Non-constant terms should be ordered by nondecreasing q_count so the
         * accumulator reaches its final level without repeated down-switches.
         */
        GpuEvalModPolynomialBasis polynomial_basis =
            GpuEvalModPolynomialBasis::Monomial;
        std::uint32_t polynomial_degree = 0;
        std::uint32_t polynomial_log_split = 0;
        bool polynomial_flat_bsgs = false;
        bool polynomial_degree_bound_virtual = false;
        std::size_t polynomial_root_anchor_q_count = 0;
        std::vector<GpuEvalModBasisStep> basis_steps;
        std::vector<GpuEvalModPolynomialTerm> polynomial_terms;

        /*
         * Number of physical Q primes removed by one logical EvalMod
         * rescale. A value of two lets 29/30-bit GPU primes implement an
         * approximately 58/60-bit CKKS working scale. Runtime executes this
         * fixed setup-time plan with the exact two-prime rescale_x2 path.
         */
        std::uint32_t logical_rescale_count = 1;
        bool dynamic_rescale = false;
        double dynamic_min_scale = 0.0;

        /*
         * Fixed number of physical primes removed after a polynomial leaf
         * sum. The optimized plan accumulates all leaf terms first and then
         * performs one logical rescale; the legacy plan rescales every term.
         */
        std::uint32_t polynomial_rescale_count = 1;
        double polynomial_output_scale = 0.0;
        bool rescale_polynomial_terms_individually = false;

        /*
         * Preferred BSGS/recurse-equivalent schedule for the 59-degree path.
         * Leaf blocks and combine steps are fully generated during CPU setup.
         * Node ids [0, polynomial_blocks.size()) name leaf outputs; combine
         * steps append/overwrite only the explicitly declared output nodes.
         */
        std::vector<GpuEvalModPolynomialBlock> polynomial_blocks;
        std::vector<GpuEvalModPolynomialCombineStep> polynomial_combine_steps;
        std::uint32_t polynomial_result_node = 0;

        /*
         * Plaintext constants used by GPU Chebyshev basis generation.
         *
         * For T_{2k}=2*T_k^2-1 and T_{a+b}=2*T_a*T_b-T_|a-b|,
         * the degree-zero term requires a plaintext 1 encoded at the generated
         * basis ciphertext parms_id. CPU setup should upload one plaintext for
         * every EvalMod basis level that may be produced.
         */
        std::vector<GpuPlaintextData> chebyshev_one_plaintexts;

        /*
         * Legacy EvalMod polynomial coefficients uploaded as plaintexts.
         *
         * CPU setup may generate/encode these constants, but the polynomial
         * evaluation itself is executed by GpuEvaluator::eval_mod_high_precision.
         * New setup code should prefer polynomial_terms above. This field is
         * kept only for compatibility with old bootstrapping tests.
         */
        std::vector<GpuPlaintextData> polynomial_coefficients;

        /*
         * Double-angle additive constants, one per iteration, already encoded
         * and uploaded to GPU memory.
         */
        std::vector<GpuPlaintextData> double_angle_constants;
        std::vector<std::uint32_t> double_angle_rescale_counts;

        /* Q prefixes whose zero-copy relinearization-key views are used. */
        std::vector<std::size_t> required_relin_q_counts;
    };

    EvalModData eval_mod;

    /*
     * Backward-compatible flat fields for early bootstrap tests/setup code.
     * Prefer eval_mod.* for new call sites.
     */
    double eval_mod_target_scale = 1.0;
    GpuPlaintextData eval_mod_input_offset_plaintext;
    std::vector<GpuPlaintextData> eval_mod_polynomial_plaintexts;
    std::vector<GpuPlaintextData> double_angle_plaintexts;

    GpuLinearMatrixGroup coeff_to_slot_matrix;
    GpuLinearMatrixGroup slot_to_coeff_matrix;
    GpuLinearMatrixGroupQP coeff_to_slot_matrix_qp;
    GpuLinearMatrixGroupQP slot_to_coeff_matrix_qp;
    GpuPlaintextData minus_i_plaintext;
    GpuPlaintextData plus_i_plaintext;

    /* Final reconstruction used by the production CPU Bootstrapper. */
    bool project_real = false;
    std::uint32_t output_ratio = 1;
    double slot_to_coeff_output_scale = 0.0;

    /*
     * Optional exact Runtime scale normalization. The uploaded plaintext is
     * selected so multiply_plain + one rescale preserves the decoded value;
     * output_scale_override then records the exact integer-log2 scale promised
     * by RuntimePlan.
     */
    GpuPlaintextData output_scale_normalization_plaintext;
    double output_scale_override = 0.0;
};

/**
 * @brief Reusable ciphertext storage for the GPU bootstrapping scheduler.
 *
 * The fields are intentionally explicit so the full bootstrap path can reuse
 * storage across calls without changing the component/limb/coeff layout used
 * by the existing GPU primitives.
 */
struct GpuBootstrapWorkspace
{
    GpuCiphertextData modraise_input;
    GpuCiphertextData raised;
    GpuCiphertextData raised_scaled;

    GpuCiphertextData coeff_to_slot_real;
    GpuCiphertextData coeff_to_slot_imag;

    GpuCiphertextData eval_mod_real;
    GpuCiphertextData eval_mod_imag;

    GpuCiphertextData scratch0;
    GpuCiphertextData scratch1;
    GpuCiphertextData scratch2;
    GpuCiphertextData scratch3;
    GpuCiphertextData scratch4;
    GpuCiphertextData scratch5;

    std::vector<GpuCiphertextData> eval_mod_basis;
    std::vector<GpuCiphertextData> eval_mod_nodes;

    GpuDoubleHoistWorkspace coeff_to_slot_double_hoist;
    GpuDoubleHoistWorkspace slot_to_coeff_double_hoist;

    struct EvalModStageTiming
    {
        double input_preparation_ms{0.0};
        double basis_generation_ms{0.0};
        double leaf_evaluation_ms{0.0};
        double bsgs_combine_ms{0.0};
        double double_angle_ms{0.0};
        double output_alignment_ms{0.0};
        double total_ms{0.0};
    };

    struct EvalModMultiplyTiming
    {
        std::string label;
        std::size_t q_count{0};
        std::size_t decomposition_count{0};
        bool is_square{false};
        double gpu_ms{0.0};
    };

    // CUDA-event profiling is opt-in and is disabled for normal timed calls.
    // A caller can profile one EvalMod invocation without introducing event
    // synchronization into the production bootstrap path.
    bool capture_eval_mod_stage_timing{false};
    EvalModStageTiming eval_mod_stage_timing;
    std::vector<EvalModMultiplyTiming> eval_mod_multiply_timings;

    // Optional correctness-only snapshots. They remain disabled in timed and
    // production bootstrap calls, so the hot path pays no copy or storage cost.
    bool capture_eval_mod_trace{false};
    GpuCiphertextData eval_mod_trace_offset_input;
    GpuCiphertextData eval_mod_trace_polynomial_output;
    std::vector<GpuCiphertextData> eval_mod_trace_double_angle_square_outputs;
    std::vector<GpuCiphertextData> eval_mod_trace_double_angle_relin_outputs;
    std::vector<GpuCiphertextData> eval_mod_trace_double_angle_rescaled_square_outputs;
    std::vector<GpuCiphertextData> eval_mod_trace_polynomial_leaf_outputs;
    std::vector<GpuCiphertextData> eval_mod_trace_polynomial_combine_outputs;
    std::vector<GpuCiphertextData> eval_mod_trace_double_angle_outputs;
};

/**
 * @brief Top-level GPU evaluator.
 *
 * This class is the highest-level GPU homomorphic operation interface.
 *
 * It plays a role similar to Cheddar's Context operation entry, but uses
 * Poseidon-style naming and Poseidon-style component vectors.
 *
 * Responsibilities:
 * - check FHE semantic validity;
 * - prepare destination metadata;
 * - prepare destination storage;
 * - create views;
 * - select the proper handler.
 *
 * It does not directly launch CUDA kernels.
 * Kernel launch planning belongs to handler classes.
 */
class GpuEvaluator
{
public:
    explicit GpuEvaluator(const GpuParameterData &params);

    void add(
        const GpuCiphertextData &left_ciphertext,
        const GpuCiphertextData &right_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void sub(
        const GpuCiphertextData &left_ciphertext,
        const GpuCiphertextData &right_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void negate(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void add_plain(
        const GpuCiphertextData &source_ciphertext,
        const GpuPlaintextData &source_plaintext,
        GpuCiphertextData &destination_ciphertext) const;

    void sub_plain(
        const GpuCiphertextData &source_ciphertext,
        const GpuPlaintextData &source_plaintext,
        GpuCiphertextData &destination_ciphertext) const;

    void multiply_plain(
        const GpuCiphertextData &source_ciphertext,
        const GpuPlaintextData &source_plaintext,
        GpuCiphertextData &destination_ciphertext) const;

    void multiply_plain_accumulate(
        const GpuCiphertextData &source_ciphertext,
        const GpuPlaintextData &source_plaintext,
        GpuCiphertextData &destination_ciphertext) const;

    void ntt_fwd(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void ntt_inv(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void multiply(
        const GpuCiphertextData &left_ciphertext,
        const GpuCiphertextData &right_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void square(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    void rescale(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Drop two physical q primes with one centered base conversion.
     *
     * The result is exactly equivalent to two consecutive ordinary rescales.
     * The optimized implementation currently batches c0 and c1.
     */
    void rescale_x2(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Execute a fixed number of CKKS rescale operations.
     *
     * The entire dropped q suffix is processed by one exact centered BConv.
     * Counts one and two retain their specialized fast paths when available.
     */
    void rescale_many(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext,
        std::uint32_t rescale_count) const;

    void rescale_dynamic(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext,
        double min_scale) const;

    void drop_modulus(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext,
        parms_id_type target_parms_id) const;

    /**
     * @brief Multiply every ciphertext component by one integer scalar modulo q.
     *
     * This is primarily used by bootstrap scale matching before ModRaise.
     */
    void multiply_scalar(
        const GpuCiphertextData &source_ciphertext,
        std::uint64_t scalar,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief CKKS bootstrap ModRaise.
     *
     * Mirrors EvaluatorCkksBase::raise_modulus for the bootstrap path:
     * source is transformed to coefficient domain if needed, converted from
     * its current q-only prefix level to the first q-only Q level, then
     * transformed back to NTT form.
     */
    void raise_modulus(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Prepare a CKKS ciphertext for bootstrap ModRaise.
     *
     * Mirrors the CPU bootstrap prefix up to the input of raise_modulus:
     * - repeatedly uses ordinary rescale until scale <= 2^54;
     * - drops to q0+1 if needed;
     * - multiplies by round(q0_over_message_ratio / scale) when > 1;
     * - drops to q0.
     *
     * The result is still q-only and in the same NTT form as the input.
     * This function intentionally does not perform Q0 -> QL basis extension.
     */
    void bootstrap_prepare_modraise_input(
        const GpuCiphertextData &source_ciphertext,
        GpuCiphertextData &destination_ciphertext,
        parms_id_type q0_parms_id,
        double q0_over_message_ratio) const;

    /**
     * @brief Relinearize ciphertext.
     *
     * Dispatches size-3 CKKS ciphertexts to the HYBRID key-switch handler and
     * writes a size-2 ciphertext in NTT form.
     */
    void relinearize(
        const GpuCiphertextData &source_ciphertext,
        const GpuRelinKeysData &relin_keys,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Relinearize and drop two q primes in one HYBRID finalize path.
     *
     * This is equivalent to relinearize(source) followed by rescale_x2, but
     * avoids materializing the full same-level size-2 ciphertext.
     */
    void relinearize_rescale_x2_hybrid(
        const GpuCiphertextData &source_ciphertext,
        const GpuRelinKeysData &relin_keys,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Rotate ciphertext.
     *
     * First CKKS implementation:
     * - apply NTT-domain Galois permutation to c0/c1;
     * - key-switch the permuted c1 with the uploaded Galois key;
     * - currently requires the direct Galois key for the requested step.
     */
    void rotate(
        const GpuCiphertextData &source_ciphertext,
        int step,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Conjugate ciphertext.
     *
     * First CKKS implementation:
     * - apply NTT-domain conjugation Galois permutation to c0/c1;
     * - key-switch the permuted c1 with the uploaded conjugation key.
     */
    void conjugate(
        const GpuCiphertextData &source_ciphertext,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &destination_ciphertext) const;

    void conjugate_pre_rotated(
        const GpuCiphertextData &source_ciphertext,
        const GpuGaloisKeysData &galois_keys,
        GpuDoubleHoistWorkspace &workspace,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Multiply by one pre-uploaded diagonal plaintext matrix using BSGS.
     *
     * This mirrors the current GPU bootstrap test reference while using
     * GPU rotate/multiply_plain/add/rescale primitives. Delayed/dynamic
     * rescale is intentionally not used in this temporary bootstrapping path.
     */
    void multiply_by_diag_matrix_bsgs(
        const GpuCiphertextData &source_ciphertext,
        const GpuMatrixPlain &matrix,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &destination_ciphertext) const;

    void multiply_by_diag_matrix_bsgs(
        const GpuCiphertextData &source_ciphertext,
        const GpuMatrixPlain &matrix,
        const GpuGaloisKeysData &galois_keys,
        std::uint32_t rescale_count,
        GpuCiphertextData &destination_ciphertext) const;

    void multiply_by_diag_matrix_bsgs_double_hoist(
        const GpuCiphertextData &source_ciphertext,
        const GpuMatrixPlainQP &matrix,
        const GpuGaloisKeysData &galois_keys,
        std::uint32_t rescale_count,
        GpuDoubleHoistWorkspace &workspace,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Apply a pre-uploaded DFT linear matrix group.
     */
    void dft(
        const GpuCiphertextData &source_ciphertext,
        const GpuLinearMatrixGroup &matrix_group,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &destination_ciphertext) const;

    void dft_double_hoist(
        const GpuCiphertextData &source_ciphertext,
        const GpuLinearMatrixGroupQP &matrix_group,
        const GpuGaloisKeysData &galois_keys,
        GpuDoubleHoistWorkspace &workspace,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief CKKS CoeffToSlot with CPU-precomputed matrices already on GPU.
     *
     * minus_i_plaintext must be the CPU-encoded plaintext for complex(0, -1)
     * at the DFT output parms_id and scale 1.0, uploaded to GPU outside the
     * timed GPU path.
     */
    void coeff_to_slot(
        const GpuCiphertextData &source_ciphertext,
        const GpuLinearMatrixGroup &matrix_group,
        const GpuPlaintextData &minus_i_plaintext,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &result_real,
        GpuCiphertextData &result_imag) const;

    void coeff_to_slot_double_hoist(
        const GpuCiphertextData &source_ciphertext,
        const GpuLinearMatrixGroupQP &matrix_group,
        const GpuPlaintextData &minus_i_plaintext,
        const GpuGaloisKeysData &galois_keys,
        GpuDoubleHoistWorkspace &workspace,
        GpuCiphertextData &result_real,
        GpuCiphertextData &result_imag) const;

    /**
     * @brief CKKS SlotToCoeff with CPU-precomputed inverse DFT matrices already on GPU.
     *
     * plus_i_plaintext must be the CPU-encoded plaintext for complex(0, 1)
     * at the input imag ciphertext parms_id and scale 1.0, uploaded to GPU
     * outside the timed GPU path.
     */
    void slot_to_coeff(
        const GpuCiphertextData &source_real,
        const GpuCiphertextData &source_imag,
        const GpuLinearMatrixGroup &matrix_group,
        const GpuPlaintextData &plus_i_plaintext,
        const GpuGaloisKeysData &galois_keys,
        GpuCiphertextData &result) const;

    void slot_to_coeff_double_hoist(
        const GpuCiphertextData &source_real,
        const GpuCiphertextData &source_imag,
        const GpuLinearMatrixGroupQP &matrix_group,
        const GpuPlaintextData &plus_i_plaintext,
        const GpuGaloisKeysData &galois_keys,
        GpuDoubleHoistWorkspace &workspace,
        GpuCiphertextData &result) const;

    /**
     * @brief CKKS bootstrapping scheduler using GPU-resident precomputed data.
     *
     * This stitches together the already implemented GPU stages. The profile
     * selects either the standard ModRaise-first schedule or the optimized
     * StC-first schedule used by the current degree-22 bootstrap path.
     * EvalMod uses the setup-time static GPU plan stored in GpuBootstrapData.
     */
    void bootstrap(
        const GpuCiphertextData &source_ciphertext,
        const GpuBootstrapData &bootstrap_data,
        const GpuRelinKeysData &relin_keys,
        const GpuGaloisKeysData &galois_keys,
        GpuBootstrapWorkspace &workspace,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Execute the StC-first prefix through the raw post-raise C2S DFT.
     *
     * The returned ciphertext precedes the conjugation-based real/imaginary
     * split. Keeping this boundary explicit lets a multi-device plan copy one
     * ciphertext and extract the two independent EvalMod branches in parallel.
     */
    void bootstrap_stc_first_transform(
        const GpuCiphertextData &source_ciphertext,
        const GpuBootstrapData &bootstrap_data,
        const GpuGaloisKeysData &galois_keys,
        GpuBootstrapWorkspace &workspace,
        GpuCiphertextData &destination_ciphertext) const;

    /** Extract the real EvalMod input from a raw C2S DFT ciphertext. */
    void bootstrap_extract_real(
        const GpuCiphertextData &source_ciphertext,
        const GpuBootstrapData &bootstrap_data,
        const GpuGaloisKeysData &galois_keys,
        GpuBootstrapWorkspace &workspace,
        GpuCiphertextData &destination_ciphertext) const;

    /** Extract the imaginary EvalMod input from a raw C2S DFT ciphertext. */
    void bootstrap_extract_imag(
        const GpuCiphertextData &source_ciphertext,
        const GpuBootstrapData &bootstrap_data,
        const GpuGaloisKeysData &galois_keys,
        GpuBootstrapWorkspace &workspace,
        GpuCiphertextData &destination_ciphertext) const;

    /** Combine evaluated real/imaginary branches and normalize the output. */
    void bootstrap_stc_first_finalize(
        const GpuCiphertextData &source_real,
        const GpuCiphertextData &source_imag,
        const GpuBootstrapData &bootstrap_data,
        GpuBootstrapWorkspace &workspace,
        GpuCiphertextData &destination_ciphertext) const;

    /**
     * @brief Evaluate the setup-time high-precision EvalMod plan on GPU.
     *
     * This is public so callers can benchmark and validate the EvalMod stage
     * independently while reusing exactly the implementation used by
     * bootstrap(). Polynomial decomposition and plaintext upload remain
     * outside this runtime operation.
     */
    void eval_mod_high_precision(
        const GpuCiphertextData &source_ciphertext,
        const GpuBootstrapData &bootstrap_data,
        const GpuRelinKeysData &relin_keys,
        GpuBootstrapWorkspace &workspace,
        GpuCiphertextData &destination_ciphertext) const;

private:
    void normalize_bootstrap_output_scale(
        GpuCiphertextData &output,
        const GpuBootstrapData &bootstrap_data,
        GpuBootstrapWorkspace &workspace) const;

    const GpuParameterData &params_;

    GpuElementwiseHandler elementwise_handler_;
    GpuKeySwitchHandler keyswitch_handler_;
    GpuNTTHandler ntt_handler_;
    GpuModSwitchHandler modswitch_handler_;
};

}  // namespace gpu
}  // namespace poseidon
