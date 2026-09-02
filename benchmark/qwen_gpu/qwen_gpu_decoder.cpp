#include "qwen_gpu_decoder.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace poseidon::benchmark::qwen_gpu
{

namespace
{

const qwen::Tensor *optional_bias(const qwen::Tensor &bias)
{
    return bias.numel() == 0 ? nullptr : &bias;
}

const qwen::he::ApproximationConfig &position_zero_config(
    const qwen::he::ApproximationConfig &fallback,
    const std::map<std::size_t, qwen::he::ApproximationConfig> &overrides)
{
    const auto found = overrides.find(0);
    return found == overrides.end() ? fallback : found->second;
}

const qwen::he::ApproximationConfig &position_config(
    const qwen::he::ApproximationConfig &fallback,
    const std::map<std::size_t, qwen::he::ApproximationConfig> &overrides,
    std::size_t position)
{
    const auto found = overrides.find(position);
    return found == overrides.end() ? fallback : found->second;
}

GpuEncryptedTensor calibrated_rms_norm(
    const GpuEncryptedTensor &input, const qwen::Tensor &weight,
    double epsilon, const qwen::he::ApproximationConfig &fallback,
    const std::map<std::size_t, qwen::he::ApproximationConfig> &overrides,
    std::size_t position_offset, const Runtime &runtime)
{
    std::vector<GpuEncryptedTensor> outputs;
    outputs.reserve(input.layout().tokens);
    for (std::size_t token = 0; token < input.layout().tokens; ++token)
    {
        const auto &config = position_config(
            fallback, overrides, position_offset + token);
        outputs.push_back(rms_norm(
            token_view(input, token, runtime), weight, epsilon,
            config.minimum, config.maximum, config.sample_count, runtime));
    }
    return concatenate_tokens(std::move(outputs), runtime);
}

GpuEncryptedTensor calibrated_silu(
    const GpuEncryptedTensor &input,
    const qwen::he::EncryptedDecoderApproximationConfig &approximation,
    std::size_t position_offset, const Runtime &runtime)
{
    std::vector<GpuEncryptedTensor> outputs;
    outputs.reserve(input.layout().tokens);
    for (std::size_t token = 0; token < input.layout().tokens; ++token)
    {
        const std::size_t position = position_offset + token;
        const auto position_features =
            approximation.silu_feature_overrides.find(position);
        std::vector<qwen::he::ApproximationConfig> features;
        if (position_features != approximation.silu_feature_overrides.end())
        {
            features = position_features->second;
        }
        else if (!approximation.silu_feature_configs.empty())
        {
            features = approximation.silu_feature_configs;
        }
        else
        {
            const auto &config = position_config(
                approximation.silu, approximation.silu_overrides,
                position);
            features.assign(input.layout().features, config);
        }
        if (features.size() != input.layout().features)
        {
            throw std::invalid_argument(
                "GPU Qwen calibrated SiLU width does not match input");
        }
        std::vector<double> minimum(features.size());
        std::vector<double> maximum(features.size());
        for (std::size_t feature = 0; feature < features.size(); ++feature)
        {
            minimum[feature] = features[feature].minimum;
            maximum[feature] = features[feature].maximum;
        }
        outputs.push_back(silu(
            token_view(input, token, runtime), minimum, maximum,
            approximation.silu.sample_count, runtime));
    }
    return concatenate_tokens(std::move(outputs), runtime);
}

void require_real_refresh(qwen::he::RefreshMode mode)
{
    if (mode != qwen::he::RefreshMode::none &&
        mode != qwen::he::RefreshMode::bootstrap)
    {
        throw std::invalid_argument(
            "GPU Qwen supports only none or real bootstrap refresh modes");
    }
}

qwen::Tensor repeat_value_weight(const qwen::QwenConfig &config)
{
    const std::size_t query_features =
        config.num_attention_heads * config.head_dim;
    const std::size_t kv_features =
        config.num_key_value_heads * config.head_dim;
    qwen::Tensor weight({query_features, kv_features});
    const std::size_t group_size =
        config.num_attention_heads / config.num_key_value_heads;
    for (std::size_t query_head = 0;
         query_head < config.num_attention_heads; ++query_head)
    {
        const std::size_t kv_head = query_head / group_size;
        for (std::size_t feature = 0; feature < config.head_dim; ++feature)
        {
            weight.at(query_head * config.head_dim + feature,
                      kv_head * config.head_dim + feature) = 1.0;
        }
    }
    return weight;
}

template <typename Function>
GpuEncryptedTensor logged(
    const std::string &name, Function &&function,
    const DecoderTrace &trace)
{
    const auto start = std::chrono::steady_clock::now();
    std::cout << "qwen_gpu_stage=" << name << " event=start\n";
    auto output = function();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "qwen_gpu_stage=" << name << " event=end elapsed_ms="
              << elapsed << " ciphers=" << output.ciphertexts().size()
              << " q_count=" << output.ciphertexts().front().meta.q_count
              << '\n';
    if (trace)
    {
        trace(name, output);
    }
    return output;
}

}  // namespace

GpuEncryptedTensor single_token_decoder_layer(
    const GpuEncryptedTensor &input,
    const qwen::DecoderLayerWeights &weights,
    const qwen::QwenConfig &model_config,
    const qwen::he::EncryptedDecoderApproximationConfig &approximation,
    std::size_t layer_index, const Runtime &runtime,
    const DecoderTrace &trace)
{
    model_config.validate();
    weights.validate(model_config);
    approximation.validate();
    if (input.layout().tokens != 1 ||
        input.layout().features != model_config.hidden_size)
    {
        throw std::invalid_argument(
            "GPU Qwen decoder currently expects one [1, hidden_size] token");
    }

    const auto &input_range = position_zero_config(
        approximation.input_inverse_sqrt,
        approximation.input_inverse_sqrt_overrides);
    auto normalized = logged(
        "layer_" + std::to_string(layer_index) + ".input_rmsnorm",
        [&] {
            return rms_norm(
                input, weights.input_norm, model_config.rms_norm_epsilon,
                input_range.minimum, input_range.maximum,
                input_range.sample_count, runtime);
        }, trace);

    // With one causal key, score/maximum/exp/reciprocal cancel exactly and
    // attention equals V. Q/K/RoPE are therefore intentionally elided.
    auto value = logged(
        "layer_" + std::to_string(layer_index) + ".value_projection",
        [&] {
            return linear(normalized, weights.value_weight,
                          optional_bias(weights.value_bias), runtime);
        }, trace);
    const qwen::Tensor repeat = repeat_value_weight(model_config);
    auto attention = logged(
        "layer_" + std::to_string(layer_index) +
            ".single_token_gqa_attention",
        [&] { return linear(value, repeat, nullptr, runtime); }, trace);
    auto attention_output = logged(
        "layer_" + std::to_string(layer_index) +
            ".attention_output_projection",
        [&] {
            return linear(attention, weights.output_weight, nullptr, runtime);
        }, trace);
    auto post_attention = logged(
        "layer_" + std::to_string(layer_index) +
            ".post_attention_residual",
        [&] { return add(input, attention_output, runtime); }, trace);
    post_attention = logged(
        "layer_" + std::to_string(layer_index) +
            ".post_attention_bootstrap",
        [&] {
            return bootstrap(
                post_attention, runtime,
                approximation.post_attention_bootstrap_value_scale);
        }, trace);

    const auto &post_range = position_zero_config(
        approximation.post_attention_inverse_sqrt,
        approximation.post_attention_inverse_sqrt_overrides);
    auto mlp_input = logged(
        "layer_" + std::to_string(layer_index) +
            ".post_attention_rmsnorm",
        [&] {
            return rms_norm(
                post_attention, weights.post_attention_norm,
                model_config.rms_norm_epsilon,
                post_range.minimum, post_range.maximum,
                post_range.sample_count, runtime);
        }, trace);
    if (approximation.mlp_input_refresh != qwen::he::RefreshMode::none)
    {
        mlp_input = logged(
            "layer_" + std::to_string(layer_index) +
                ".mlp_input_bootstrap",
            [&] {
                return bootstrap(
                    mlp_input, runtime,
                    approximation.mlp_input_bootstrap_value_scale);
            }, trace);
    }

    auto gate = logged(
        "layer_" + std::to_string(layer_index) + ".mlp_gate_projection",
        [&] { return linear(mlp_input, weights.gate_weight, nullptr, runtime); },
        trace);
    auto up = logged(
        "layer_" + std::to_string(layer_index) + ".mlp_up_projection",
        [&] { return linear(mlp_input, weights.up_weight, nullptr, runtime); },
        trace);

    std::vector<qwen::he::ApproximationConfig> silu_features;
    const auto position_features =
        approximation.silu_feature_overrides.find(0);
    if (position_features != approximation.silu_feature_overrides.end())
    {
        silu_features = position_features->second;
    }
    else if (!approximation.silu_feature_configs.empty())
    {
        silu_features = approximation.silu_feature_configs;
    }
    else
    {
        silu_features.assign(
            model_config.intermediate_size, approximation.silu);
    }
    if (silu_features.size() != model_config.intermediate_size)
    {
        throw std::invalid_argument(
            "GPU Qwen SiLU calibration width does not match MLP width");
    }
    std::vector<double> silu_minimum(silu_features.size());
    std::vector<double> silu_maximum(silu_features.size());
    for (std::size_t feature = 0; feature < silu_features.size(); ++feature)
    {
        silu_minimum[feature] = silu_features[feature].minimum;
        silu_maximum[feature] = silu_features[feature].maximum;
    }
    auto activated_gate = logged(
        "layer_" + std::to_string(layer_index) + ".mlp_silu",
        [&] {
            return silu(
                gate, silu_minimum, silu_maximum,
                approximation.silu.sample_count, runtime);
        }, trace);
    auto swiglu = logged(
        "layer_" + std::to_string(layer_index) + ".mlp_swiglu",
        [&] { return multiply(activated_gate, up, runtime); }, trace);
    auto mlp_output = logged(
        "layer_" + std::to_string(layer_index) + ".mlp_down_projection",
        [&] { return linear(swiglu, weights.down_weight, nullptr, runtime); },
        trace);
    auto output = logged(
        "layer_" + std::to_string(layer_index) + ".output_residual",
        [&] { return add(post_attention, mlp_output, runtime); }, trace);
    return logged(
        "layer_" + std::to_string(layer_index) + ".output_bootstrap",
        [&] {
            return bootstrap(
                output, runtime,
                approximation.output_bootstrap_value_scale);
        }, trace);
}

GpuEncryptedTensor decoder_layer(
    const GpuEncryptedTensor &input,
    const qwen::DecoderLayerWeights &weights,
    const qwen::QwenConfig &model_config,
    const qwen::he::EncryptedDecoderApproximationConfig &approximation,
    std::size_t layer_index, std::size_t position_offset,
    const Runtime &runtime, const DecoderTrace &trace,
    GpuKVCache *cache)
{
    model_config.validate();
    weights.validate(model_config);
    approximation.validate();
    require_real_refresh(approximation.input_norm_refresh);
    require_real_refresh(approximation.qkv_refresh);
    require_real_refresh(approximation.attention_output_refresh);
    require_real_refresh(approximation.post_attention_refresh);
    require_real_refresh(approximation.mlp_input_refresh);
    require_real_refresh(approximation.output_refresh);
    if (input.layout().features != model_config.hidden_size)
    {
        throw std::invalid_argument(
            "GPU Qwen decoder input width does not match model");
    }
    const std::string prefix = "layer_" + std::to_string(layer_index) + '.';
    auto normalized = logged(prefix + "input_rmsnorm", [&] {
        return calibrated_rms_norm(
            input, weights.input_norm, model_config.rms_norm_epsilon,
            approximation.input_inverse_sqrt,
            approximation.input_inverse_sqrt_overrides,
            position_offset, runtime);
    }, trace);
    if (approximation.input_norm_refresh == qwen::he::RefreshMode::bootstrap)
    {
        normalized = logged(prefix + "input_rmsnorm_refresh", [&] {
            return bootstrap(
                normalized, runtime,
                approximation.input_norm_bootstrap_value_scale);
        }, trace);
    }

    auto query = logged(prefix + "query_projection", [&] {
        return linear(normalized, weights.query_weight,
                      optional_bias(weights.query_bias), runtime);
    }, trace);
    auto key = logged(prefix + "key_projection", [&] {
        return linear(normalized, weights.key_weight,
                      optional_bias(weights.key_bias), runtime);
    }, trace);
    auto value = logged(prefix + "value_projection", [&] {
        return linear(normalized, weights.value_weight,
                      optional_bias(weights.value_bias), runtime);
    }, trace);
    query = logged(prefix + "query_rope", [&] {
        return rope(
            query, model_config.num_attention_heads,
            model_config.head_dim, position_offset,
            model_config.rope_theta, runtime);
    }, trace);
    key = logged(prefix + "key_rope", [&] {
        return rope(
            key, model_config.num_key_value_heads,
            model_config.head_dim, position_offset,
            model_config.rope_theta, runtime);
    }, trace);
    if (approximation.qkv_refresh == qwen::he::RefreshMode::bootstrap)
    {
        query = logged(prefix + "query_rope_refresh", [&] {
            return bootstrap(
                query, runtime,
                approximation.query_key_bootstrap_value_scale);
        }, trace);
        key = logged(prefix + "key_rope_refresh", [&] {
            return bootstrap(
                key, runtime,
                approximation.query_key_bootstrap_value_scale);
        }, trace);
        value = logged(prefix + "value_refresh", [&] {
            return bootstrap(value, runtime);
        }, trace);
    }

    auto attention = logged(prefix + "attention", [&] {
        return stable_causal_gqa_attention(
            query, key, value, model_config,
            approximation.attention, runtime, cache);
    }, trace);
    if (approximation.attention_output_refresh ==
        qwen::he::RefreshMode::bootstrap)
    {
        attention = logged(prefix + "attention_refresh", [&] {
            return bootstrap(attention, runtime);
        }, trace);
    }
    auto attention_output = logged(
        prefix + "attention_output_projection", [&] {
            return linear(
                attention, weights.output_weight, nullptr, runtime);
        }, trace);
    auto post_attention = logged(prefix + "post_attention_residual", [&] {
        return add(input, attention_output, runtime);
    }, trace);
    if (approximation.post_attention_refresh ==
        qwen::he::RefreshMode::bootstrap)
    {
        post_attention = logged(prefix + "post_attention_bootstrap", [&] {
            return bootstrap(
                post_attention, runtime,
                approximation.post_attention_bootstrap_value_scale);
        }, trace);
    }

    auto mlp_input = logged(prefix + "post_attention_rmsnorm", [&] {
        return calibrated_rms_norm(
            post_attention, weights.post_attention_norm,
            model_config.rms_norm_epsilon,
            approximation.post_attention_inverse_sqrt,
            approximation.post_attention_inverse_sqrt_overrides,
            position_offset, runtime);
    }, trace);
    if (approximation.mlp_input_refresh == qwen::he::RefreshMode::bootstrap)
    {
        mlp_input = logged(prefix + "mlp_input_bootstrap", [&] {
            return bootstrap(
                mlp_input, runtime,
                approximation.mlp_input_bootstrap_value_scale);
        }, trace);
    }
    const auto gate = logged(prefix + "mlp_gate_projection", [&] {
        return linear(mlp_input, weights.gate_weight, nullptr, runtime);
    }, trace);
    const auto up = logged(prefix + "mlp_up_projection", [&] {
        return linear(mlp_input, weights.up_weight, nullptr, runtime);
    }, trace);
    const auto activated_gate = logged(prefix + "mlp_silu", [&] {
        return calibrated_silu(
            gate, approximation, position_offset, runtime);
    }, trace);
    const auto swiglu = logged(prefix + "mlp_swiglu", [&] {
        return multiply(activated_gate, up, runtime);
    }, trace);
    const auto mlp_output = logged(prefix + "mlp_down_projection", [&] {
        return linear(swiglu, weights.down_weight, nullptr, runtime);
    }, trace);
    auto output = logged(prefix + "output_residual", [&] {
        return add(post_attention, mlp_output, runtime);
    }, trace);
    if (approximation.output_refresh == qwen::he::RefreshMode::bootstrap)
    {
        output = logged(prefix + "output_bootstrap", [&] {
            return bootstrap(
                output, runtime,
                approximation.output_bootstrap_value_scale);
        }, trace);
    }
    return output;
}

}  // namespace poseidon::benchmark::qwen_gpu
