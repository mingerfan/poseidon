#pragma once

#include "poseidon/gpu/gpu_bootstrap_profile.h"

#include <nlohmann/json.hpp>

#include <string>

namespace poseidon
{
class PoseidonContext;

namespace runtime_api
{

/**
 * @brief The Boot recipe of a native GPU profile: what expand-ckks-bootstrap needs.
 *
 * Exported on the GPU host from the very profile the runner builds
 * (poseidon_runtime_gpu_mlp_e2e --export-boot-recipe), so the compiler cuts
 * each Boot into steps that this profile can run (RuntimePlan V6). It is a
 * description, not the constants: levels, scales and rescale counts of every
 * GpuEvaluator::bootstrap() stage, and each CoeffToSlot/SlotToCoeff layer's
 * BSGS baby steps and giant groups. The format is
 * compiler/lib/Dialect/CKKS/Transforms/ExpandBootstrap.cpp's input: version 1
 * is a CtS-first Boot (ModRaise, CoeffToSlot, EvalMod, SlotToCoeff); version 2
 * adds "order" = "slot_to_coeff_first" (SlotToCoeff on the input, ModRaise,
 * CoeffToSlot, EvalMod, merge), the manual ResNet20 Boot.
 * "diagonal_plaintexts" says whether an op-granularity cut can name each
 * diagonal.
 */
nlohmann::json make_boot_recipe(const gpu::GpuBootstrapProfile &profile,
                                const PoseidonContext &context);

/**
 * @brief The profile plaintext a boot_step mul_const names.
 *
 * "minus_i", "plus_i", "post_raise", or "<c2s|s2c>/<layer>/<diagonal>" for a
 * linear-transform diagonal -- [Q|P] in a double-hoist profile, of which
 * mul_const uses the Q part (GpuEvaluator::multiply_plain_q_view).
 */
const gpu::GpuPlaintextData &boot_plaintext(const gpu::GpuBootstrapData &data,
                                            const std::string &key);

} // namespace runtime_api
} // namespace poseidon
