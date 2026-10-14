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
};

} // namespace gpu
} // namespace poseidon
