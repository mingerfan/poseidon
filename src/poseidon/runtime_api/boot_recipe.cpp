#include "poseidon/runtime_api/boot_recipe.h"

#include "poseidon/gpu/gpu_double_hoist.h"
#include "poseidon/gpu/gpu_linear_transform.h"
#include "poseidon/gpu/gpu_scale_planner.h"
#include "poseidon/poseidon_context.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace poseidon
{
namespace runtime_api
{
namespace
{

using Json = nlohmann::json;

// One layer's BSGS structure, in whichever form the profile uploaded.
struct Layer
{
    std::uint32_t level = 0;
    double plaintext_scale = 0.0;
    Json baby_steps = Json::array();
    Json groups = Json::array();
};

Layer qp_layer(const gpu::GpuMatrixPlainQP &matrix)
{
    Layer layer{matrix.level, matrix.scale};
    const auto &plan = matrix.plan;
    for (const int step : plan.baby_steps)
        layer.baby_steps.push_back(step);
    for (std::size_t group = 0; group < plan.giant_steps.size(); ++group)
    {
        Json terms = Json::array();
        for (auto t = plan.group_term_offsets.at(group); t < plan.group_term_offsets.at(group + 1); ++t)
        {
            const int baby = plan.baby_steps.at(plan.terms.at(t).baby_index);
            terms.push_back({baby, plan.giant_steps[group] + baby});
        }
        layer.groups.push_back({{"giant", plan.giant_steps[group]}, {"terms", terms}});
    }
    return layer;
}

Layer classic_layer(const gpu::GpuMatrixPlain &matrix)
{
    Layer layer{matrix.level, matrix.scale};
    const auto plan = gpu::make_gpu_loose_bsgs_plan(matrix);
    for (const int step : plan.baby_steps)
        layer.baby_steps.push_back(step);
    for (const auto &group : plan.groups)
    {
        Json terms = Json::array();
        for (const auto &term : group.terms)
            terms.push_back({term.baby_step, term.diagonal_index});
        layer.groups.push_back({{"giant", group.giant_step}, {"terms", terms}});
    }
    return layer;
}

// The layers of one transform from (level, scale), as dft*() runs them; both
// are advanced to the transform's output.
Json transform_layers(const gpu::GpuBootstrapData &data, bool coeff_to_slot,
                      const std::vector<std::uint64_t> &moduli, std::uint32_t &level, double &scale)
{
    const bool qp = data.linear_transform_mode == gpu::GpuLinearTransformMode::DoubleHoistBsgs;
    const auto &qp_group = coeff_to_slot ? data.coeff_to_slot_matrix_qp : data.slot_to_coeff_matrix_qp;
    const auto &classic_group = coeff_to_slot ? data.coeff_to_slot_matrix : data.slot_to_coeff_matrix;
    const double min_scale = qp ? qp_group.rescale_min_scale() : classic_group.rescale_min_scale();
    const std::uint32_t step = qp ? qp_group.step() : classic_group.step();
    const std::size_t count = qp ? qp_group.data().size() : classic_group.data().size();
    Json layers = Json::array();
    for (std::size_t index = 0; index < count; ++index)
    {
        const Layer layer = qp ? qp_layer(qp_group.data()[index]) : classic_layer(classic_group.data()[index]);
        if (layer.level != level)
            throw std::runtime_error("boot recipe: a layer's matrix is at level " +
                                     std::to_string(layer.level) + ", its input at " +
                                     std::to_string(level));
        const double product = scale * layer.plaintext_scale;
        std::uint32_t drops = 0;
        double output = product;
        if (min_scale > 0.0)
        {
            const auto plan = gpu::plan_gpu_dynamic_rescale(
                product, min_scale, std::span<const std::uint64_t>(moduli.data(), level + 1));
            drops = plan.rescale_count;
            output = plan.output_scale;
        }
        else
        {
            drops = std::max(step, std::uint32_t{1});
            for (std::uint32_t i = 0; i < drops; ++i)
                output /= static_cast<double>(moduli.at(level - i));
        }
        if (drops == 0 || drops > level)
            throw std::runtime_error("boot recipe: a layer drops no level or the whole chain");
        layers.push_back({{"level", level}, {"plaintext_scale_log2", std::log2(layer.plaintext_scale)},
                          {"rescale_count", drops}, {"output_scale_log2", std::log2(output)},
                          {"baby_steps", layer.baby_steps}, {"groups", layer.groups}});
        level -= drops;
        scale = output;
    }
    return layers;
}

std::uint32_t level_of(std::size_t q_count)
{
    if (q_count == 0)
        throw std::runtime_error("boot recipe: a plaintext has no Q primes");
    return static_cast<std::uint32_t>(q_count - 1);
}

// Whether every linear-transform diagonal has a Q view a mul_const step can
// name ("<c2s|s2c>/<layer>/<diagonal>"): an op-granularity cut needs them. A
// profile that uploads its diagonals compressed keeps only the residues the
// fused double-hoist kernels read.
bool diagonal_plaintexts(const gpu::GpuBootstrapData &data)
{
    if (data.linear_transform_mode != gpu::GpuLinearTransformMode::DoubleHoistBsgs)
        return true;
    for (const auto *group : {&data.coeff_to_slot_matrix_qp, &data.slot_to_coeff_matrix_qp})
        for (const auto &matrix : group->data())
            if (matrix.plain_vec_qp.empty())
                return false;
    return true;
}

} // namespace

Json make_boot_recipe(const gpu::GpuBootstrapProfile &profile, const PoseidonContext &context)
{
    const auto &data = profile.bootstrap_data;
    if (!(data.raised_scale_override > 0.0))
        throw std::runtime_error("boot recipe: the profile has no raised_scale_override, so the "
                                 "raised scale depends on the input and no recipe can fix it");
    if (data.linear_transform_mode == gpu::GpuLinearTransformMode::SingleHoistBsgs)
        throw std::runtime_error("boot recipe: single_hoist is not a bootstrap mode");
    std::vector<std::uint64_t> moduli;
    for (const auto &modulus : context.crt_context()->first_context_data()->coeff_modulus())
        moduli.push_back(modulus.value());
    const auto top = static_cast<std::uint32_t>(moduli.size() - 1);
    const auto q0 = context.parameters_literal()->q0_level();

    // An S2C-first Boot (the manual ResNet20 one) runs SlotToCoeff on its
    // input before ModRaise: the layers, then the rounded integer product
    // that sets ModRaise's input scale at q0, as GpuEvaluator::bootstrap()
    // does. Recipe format 2 says so in "order"; format 1 is CtS-first.
    Json s2c_first = nullptr, prepare = nullptr;
    if (data.slot_to_coeff_first)
    {
        if (data.linear_transform_mode != gpu::GpuLinearTransformMode::DoubleHoistBsgs ||
            profile.input_level_min != profile.input_level_max || !(data.input_scale > 0.0) ||
            static_cast<std::size_t>(profile.input_level_min) + 1 != data.input_q_count)
            throw std::runtime_error("boot recipe: an S2C-first Boot needs double hoist and one "
                                     "declared input level and scale");
        std::uint32_t input_level = static_cast<std::uint32_t>(profile.input_level_min);
        double input_scale = data.input_scale;
        s2c_first = transform_layers(data, false, moduli, input_level, input_scale);
        if (input_level != q0)
            throw std::runtime_error("boot recipe: S2C-first SlotToCoeff ends at level " +
                                     std::to_string(input_level) + ", not at q0");
        const double multiplier = std::round(data.q0_over_message_ratio / input_scale);
        if (!std::isfinite(multiplier) || multiplier < 1.0 || multiplier > std::ldexp(1.0, 53))
            throw std::runtime_error("boot recipe: S2C-first preparation multiplier is outside [1, 2^53]");
        prepare = {{"level", q0}, {"multiplier", static_cast<std::uint64_t>(multiplier)},
                   {"scale_log2", std::log2(input_scale * multiplier)},
                   {"work", Json::array({Json::array({"mul_cp", q0, 1})})}};
    }

    // ModRaise (bootstrap_raise), then the optional post-raise product.
    std::uint32_t level = top;
    double scale = data.raised_scale_override;
    Json post_raise = nullptr;
    if (data.post_raise_integer_multiplier > 1)
    {
        scale *= data.post_raise_scale_multiplier;
        post_raise = {{"kind", "scalar"}, {"multiplier", data.post_raise_integer_multiplier},
                      {"scale_log2", std::log2(scale)}};
    }
    else if (!data.post_raise_plaintext.empty())
    {
        scale *= data.post_raise_plaintext.meta.scale;
        post_raise = {{"kind", "plaintext"}, {"key", "post_raise"}, {"scale_log2", std::log2(scale)}};
    }
    // An S2C-first input is already at q0 and scaled: only the raise itself.
    const Json raise = {{"level", top}, {"scale_log2", std::log2(data.raised_scale_override)},
                        {"work", data.slot_to_coeff_first
                                     ? Json::array({Json::array({"rescale", top, 2})})
                                     : Json::array({Json::array({"mod_switch", q0 + 1, 1}),
                                                    Json::array({"mul_cp", q0 + 1, 1}),
                                                    Json::array({"rescale", q0 + 1, 1}),
                                                    Json::array({"rescale", top, 2})})}};

    const Json c2s = transform_layers(data, true, moduli, level, scale);
    if (level_of(data.minus_i_plaintext.meta.q_count) != level)
        throw std::runtime_error("boot recipe: the -i plaintext is not at the CoeffToSlot output level");
    const std::uint32_t eval_input = level;

    const auto &mod = data.eval_mod;
    const std::uint32_t eval_level = level_of(mod.output_q_count);
    std::int64_t products = 0, polynomial_terms = static_cast<std::int64_t>(mod.polynomial_terms.size());
    for (const auto &block : mod.polynomial_blocks)
        polynomial_terms += static_cast<std::int64_t>(block.terms.size());
    products = static_cast<std::int64_t>(mod.basis_steps.size() + mod.polynomial_combine_steps.size() +
                                         mod.double_angle_constants.size());
    const auto middle = (eval_input + eval_level) / 2;
    const Json eval_mod = {{"level", eval_level}, {"scale_log2", std::log2(mod.output_scale)},
                           {"work", Json::array({Json::array({"mul_cc", middle, products}),
                                                 Json::array({"relinearize", middle, products}),
                                                 Json::array({"rescale", middle, products + static_cast<std::int64_t>(mod.polynomial_blocks.size())}),
                                                 Json::array({"mul_cp", middle, polynomial_terms}),
                                                 Json::array({"add_cc", middle, polynomial_terms})})}};

    level = eval_level;
    scale = data.slot_to_coeff_input_scale > 0.0 ? data.slot_to_coeff_input_scale : mod.output_scale;
    if (data.plus_i_plaintext.meta.scale != 1.0 ||
        level_of(data.plus_i_plaintext.meta.q_count) != level)
        throw std::runtime_error("boot recipe: the +i plaintext is not a unit at the EvalMod output level");
    if (data.slot_to_coeff_first)
    {
        // real + i * imag, times the fold 16, plus its conjugate: the output
        // is labelled with the application scale the coefficient fold makes.
        if (!data.project_real || data.output_ratio != 32 ||
            level != static_cast<std::uint32_t>(profile.output_level) ||
            !(data.slot_to_coeff_output_scale > 0.0) ||
            std::abs(std::log2(mod.output_scale * data.output_scale_multiplier /
                               data.slot_to_coeff_output_scale)) > 1e-8)
            throw std::runtime_error("boot recipe: the S2C-first output fold is not the one "
                                     "GpuEvaluator::bootstrap() checks");
        return {
            {"format", "poseidon-boot-recipe"}, {"format_version", 2},
            {"order", "slot_to_coeff_first"},
            {"profile_id", profile.profile_id},
            {"linear_transform_mode", gpu::gpu_linear_transform_mode_name(data.linear_transform_mode)},
            {"diagonal_plaintexts", diagonal_plaintexts(data)},
            {"context", {{"poly_degree", context.parameters_literal()->degree()}, {"rns_moduli", moduli},
                         {"q0_level", q0}}},
            {"input", {{"level_min", profile.input_level_min}, {"level_max", profile.input_level_max},
                       {"scale_log2", std::log2(data.input_scale)}}},
            {"output", {{"level", profile.output_level}, {"scale_log2", profile.output_scale_log2}}},
            {"slot_to_coeff", {{"layers", s2c_first}}},
            {"prepare", prepare},
            {"raise", raise},
            {"post_raise", post_raise},
            {"coeff_to_slot", {{"layers", c2s}, {"minus_i", {{"key", "minus_i"},
                               {"scale_log2", std::log2(data.minus_i_plaintext.meta.scale)}}}}},
            {"eval_mod", eval_mod},
            {"merge", {{"plus_i", {{"key", "plus_i"}, {"scale_log2", 0.0}}},
                       {"fold_multiplier", data.output_ratio / 2}}},
            {"project_real", true},
            {"output_scale_log2", std::log2(data.slot_to_coeff_output_scale)},
        };
    }
    const Json s2c = transform_layers(data, false, moduli, level, scale);
    if (level != static_cast<std::uint32_t>(profile.output_level))
        throw std::runtime_error("boot recipe: SlotToCoeff ends at level " + std::to_string(level) +
                                 ", the profile says " + std::to_string(profile.output_level));

    const std::uint32_t ratio = data.project_real ? data.output_ratio / 2 : data.output_ratio;
    return {
        {"format", "poseidon-boot-recipe"}, {"format_version", 1},
        {"profile_id", profile.profile_id},
        {"linear_transform_mode", gpu::gpu_linear_transform_mode_name(data.linear_transform_mode)},
        {"diagonal_plaintexts", diagonal_plaintexts(data)},
        {"context", {{"poly_degree", context.parameters_literal()->degree()}, {"rns_moduli", moduli},
                     {"q0_level", q0}}},
        {"input", {{"level_min", profile.input_level_min}, {"level_max", profile.input_level_max}}},
        {"output", {{"level", profile.output_level}, {"scale_log2", profile.output_scale_log2}}},
        {"raise", raise},
        {"post_raise", post_raise},
        {"coeff_to_slot", {{"layers", c2s}, {"minus_i", {{"key", "minus_i"},
                           {"scale_log2", std::log2(data.minus_i_plaintext.meta.scale)}}}}},
        {"eval_mod", eval_mod},
        {"slot_to_coeff_input_scale_log2",
         data.slot_to_coeff_input_scale > 0.0 ? Json(std::log2(data.slot_to_coeff_input_scale)) : Json(nullptr)},
        {"slot_to_coeff", {{"layers", s2c}, {"plus_i", {{"key", "plus_i"}, {"scale_log2", 0.0}}}}},
        {"slot_to_coeff_output_scale_log2",
         data.slot_to_coeff_output_scale > 0.0 ? Json(std::log2(data.slot_to_coeff_output_scale)) : Json(nullptr)},
        {"project_real", data.project_real},
        {"output_multiplier", ratio},
    };
}

const gpu::GpuPlaintextData &boot_plaintext(const gpu::GpuBootstrapData &data, const std::string &key)
{
    if (key == "minus_i") return data.minus_i_plaintext;
    if (key == "plus_i") return data.plus_i_plaintext;
    if (key == "post_raise") return data.post_raise_plaintext;
    // <c2s|s2c>/<layer>/<diagonal>
    const auto first = key.find('/'), second = key.find('/', first + 1);
    const std::string transform = key.substr(0, first);
    if (first == std::string::npos || second == std::string::npos ||
        (transform != "c2s" && transform != "s2c"))
        throw std::invalid_argument("unknown Boot plaintext '" + key + "'");
    const std::size_t layer = std::stoul(key.substr(first + 1, second - first - 1));
    const int diagonal = std::stoi(key.substr(second + 1));
    const bool c2s = transform == "c2s";
    if (data.linear_transform_mode == gpu::GpuLinearTransformMode::DoubleHoistBsgs)
    {
        const auto &matrix = (c2s ? data.coeff_to_slot_matrix_qp : data.slot_to_coeff_matrix_qp).data().at(layer);
        if (matrix.plain_vec_qp.empty())
            throw std::invalid_argument("Boot plaintext '" + key + "' is compressed and has no Q view");
        return matrix.plain_vec_qp.at(diagonal);
    }
    return (c2s ? data.coeff_to_slot_matrix : data.slot_to_coeff_matrix).data().at(layer).plain_vec.at(diagonal);
}

} // namespace runtime_api
} // namespace poseidon
