#pragma once

#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_key.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace poseidon
{

class KeyGenerator;
class PoseidonContext;

namespace gpu
{

/**
 * @brief Deterministic setup parameters for one native GPU bootstrap profile.
 *
 * The defaults describe the current degree-22, baby-4, three-double-angle
 * profile. They intentionally do not read process environment variables.
 */
struct GpuBootstrapProfileConfig
{
    std::string profile_id = "poseidon-gpu-degree22-da3";
    std::uint32_t bootstrap_ratio = 32;
    std::uint32_t eval_mod_log_scale = 45;
    std::uint32_t eval_mod_double_angle = 3;
    std::uint32_t eval_mod_k = 25;
    std::uint32_t eval_mod_arcsine_degree = 0;
    std::uint32_t eval_mod_sine_degree = 22;
    std::uint32_t eval_mod_generation_degree = 59;
    std::optional<std::uint32_t> eval_mod_truncate_degree = 22;
    bool eval_mod_fixed_degree_refit = false;
    std::uint32_t eval_mod_log_split = 2;
    bool eval_mod_virtual_degree_bound = true;
    std::uint32_t logical_rescale_count = 1;
    std::uint32_t c2s_log_bsgs_ratio = 1;
    std::vector<std::uint32_t> c2s_layer_groups;
    std::uint32_t c2s_direct_layer_threshold = 0;
    std::uint32_t s2c_log_bsgs_ratio = 1;
    bool project_real = false;
    GpuLinearTransformMode linear_transform_mode =
        GpuLinearTransformMode::DoubleHoistBsgs;

    /*
     * Sparse-secret encapsulation: a non-zero value is the Hamming weight of
     * the ephemeral secret ModRaise runs under (Cerium uses 32). With h=32
     * the ModRaise overflow stays within K=12 often enough for the degree-22
     * fit (about 2^-29 failure per Boot, versus 2^-14 for h=192/K=25).
     * The dense->sparse key is generated over the full key modulus, which a
     * secure deployment must replace with a small-modulus key.
     */
    std::uint32_t sparse_secret_hamming_weight = 0;
    /* Evaluate odd Chebyshev terms too (the discrete fit is not even). */
    bool eval_mod_full_coefficients = false;
    /* Periodic compression of the double-hoist DFT diagonals. */
    bool compress_linear_plaintexts = false;
    /*
     * Declared output level. A cheaper EvalMod leaves surplus Q limbs; they
     * are dropped before SlotToCoeff so the profile still matches the
     * OperatorSpec, and SlotToCoeff runs on fewer limbs.
     */
    std::optional<int> output_level;
};

/**
 * @brief EvalMod choices of the manual S2C-first ResNet20 Boot. The defaults
 * are the original degree-59/DA2/K=25 profile.
 */
struct GpuResNet20ManualOptions
{
    std::uint32_t sine_degree = 59;
    std::uint32_t double_angle = 2;
    std::uint32_t k = 25;
    std::uint32_t log_split = 3;
    bool full_coefficients = false;
    std::uint32_t sparse_secret_hamming_weight = 0;
};

/**
 * @brief Fully uploaded resources and Runtime metadata for one GPU profile.
 */
struct GpuBootstrapProfile
{
    std::string profile_id;
    int cuda_device_id = 0;
    int input_level_min = 0;
    int input_level_max = 0;
    int input_components = 2;
    std::optional<int> input_scale_log2;
    int output_level = 0;
    int output_scale_log2 = 0;
    int output_components = 2;
    GpuBootstrapData bootstrap_data;
    std::shared_ptr<const GpuRelinKeysData> relin_keys;
    std::shared_ptr<const GpuGaloisKeysData> galois_keys;
};

/**
 * @brief Build all CPU plans/keys and upload one immutable GPU profile.
 *
 * Setup is intentionally untimed. The returned resources are complete
 * single-device objects and can be installed directly in PoseidonGpuApi.
 */
class GpuBootstrapProfileBuilder
{
public:
    static GpuBootstrapProfile build(
        const PoseidonContext &context,
        KeyGenerator &key_generator,
        int cuda_device_id,
        const GpuBootstrapProfileConfig &config = {});
    // Manual ResNet20 Q50/P25: Q6/scale40 -> S2C -> raise -> C2S ->
    // degree59/DA2 with folded coefficients -> Q31/scale40.
    static GpuBootstrapProfile build_resnet20_manual(
        const PoseidonContext &context, KeyGenerator &key_generator,
        int cuda_device_id, const GpuResNet20ManualOptions &options = {});
};

} // namespace gpu
} // namespace poseidon
