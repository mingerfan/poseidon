#include "qwen_gpu_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poseidon::benchmark::qwen_gpu
{

namespace
{

bool attention_validation_enabled()
{
    return std::getenv("POSEIDON_GPU_QWEN_VALIDATE") != nullptr;
}

double tensor_max_error(
    const qwen::Tensor &actual, const qwen::Tensor &expected)
{
    if (actual.shape() != expected.shape())
    {
        throw std::invalid_argument(
            "GPU Qwen attention diagnostic shapes do not match");
    }
    double error = 0.0;
    for (std::size_t index = 0; index < actual.numel(); ++index)
    {
        error = std::max(
            error, std::abs(actual.data()[index] - expected.data()[index]));
    }
    return error;
}

void report_bootstrap_diagnostic(
    const std::string &stage, const qwen::Tensor &before,
    const GpuEncryptedTensor &after, const Runtime &runtime)
{
    const qwen::Tensor refreshed = decrypt_tensor(after, runtime);
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (double value : before.data())
    {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    std::cout << "qwen_gpu_attention_diagnostic stage=" << stage
              << " bootstrap_max_error="
              << tensor_max_error(refreshed, before)
              << " input_min=" << minimum
              << " input_max=" << maximum << '\n';
}

void report_polynomial_diagnostic(
    const std::string &stage, const GpuEncryptedTensor &input,
    const GpuEncryptedTensor &output,
    const qwen::he::ApproximationConfig &config, bool use_softplus,
    const Runtime &runtime)
{
    const qwen::Tensor plain_input = decrypt_tensor(input, runtime);
    const qwen::Tensor plain_output = decrypt_tensor(output, runtime);
    qwen::Tensor expected(plain_input.shape());
    const poseidon::Polynomial polynomial = use_softplus
        ? qwen::he::make_softplus_polynomial(config)
        : qwen::he::make_sigmoid_polynomial(config);
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < plain_input.numel(); ++index)
    {
        const double value = plain_input.data()[index];
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        expected.data()[index] =
            qwen::he::evaluate_chebyshev_plain(value, polynomial);
    }
    std::cout << "qwen_gpu_attention_diagnostic stage=" << stage
              << " polynomial_max_error="
              << tensor_max_error(plain_output, expected)
              << " input_min=" << minimum
              << " input_max=" << maximum
              << " calibrated_min=" << config.minimum
              << " calibrated_max=" << config.maximum << '\n';
}

void report_score_diagnostic(
    const std::string &stage, const GpuEncryptedTensor &query,
    const GpuEncryptedTensor &key, const GpuEncryptedTensor &score,
    const qwen::QwenConfig &config, const Runtime &runtime)
{
    const qwen::Tensor plain_query = decrypt_tensor(query, runtime);
    const qwen::Tensor plain_key = decrypt_tensor(key, runtime);
    const qwen::Tensor plain_score = decrypt_tensor(score, runtime);
    qwen::Tensor expected(plain_score.shape());
    const double scale =
        1.0 / std::sqrt(static_cast<double>(config.head_dim));
    for (std::size_t head = 0; head < config.num_attention_heads; ++head)
    {
        const std::size_t begin = head * config.head_dim;
        double value = 0.0;
        for (std::size_t feature = 0; feature < config.head_dim; ++feature)
        {
            value += plain_query.at(0, begin + feature) *
                     plain_key.at(0, begin + feature);
        }
        value *= scale;
        for (std::size_t feature = 0; feature < config.head_dim; ++feature)
        {
            expected.at(0, begin + feature) = value;
        }
    }
    std::cout << "qwen_gpu_attention_diagnostic stage=" << stage
              << " score_max_error="
              << tensor_max_error(plain_score, expected) << '\n';
}

void validate_qkv(
    const GpuEncryptedTensor &query,
    const GpuEncryptedTensor &key,
    const GpuEncryptedTensor &value,
    const qwen::QwenConfig &config)
{
    config.validate();
    const std::size_t query_features =
        config.num_attention_heads * config.head_dim;
    const std::size_t kv_features =
        config.num_key_value_heads * config.head_dim;
    if (query.layout().tokens == 0 ||
        key.layout().tokens != query.layout().tokens ||
        value.layout().tokens != query.layout().tokens ||
        query.layout().features != query_features ||
        key.layout().features != kv_features ||
        value.layout().features != kv_features ||
        query.layout().feature_chunks() != 1 ||
        key.layout().feature_chunks() != 1 ||
        value.layout().feature_chunks() != 1)
    {
        throw std::invalid_argument(
            "GPU Qwen attention Q/K/V layouts do not match");
    }
}

qwen::Tensor repeat_kv_weight(const qwen::QwenConfig &config)
{
    const std::size_t query_features =
        config.num_attention_heads * config.head_dim;
    const std::size_t kv_features =
        config.num_key_value_heads * config.head_dim;
    qwen::Tensor weight({query_features, kv_features});
    const std::size_t group_size = config.query_group_size();
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

qwen::Tensor head_sum_weight(const qwen::QwenConfig &config)
{
    const std::size_t features =
        config.num_attention_heads * config.head_dim;
    qwen::Tensor weight({features, features});
    const double scale =
        1.0 / std::sqrt(static_cast<double>(config.head_dim));
    for (std::size_t head = 0; head < config.num_attention_heads; ++head)
    {
        const std::size_t begin = head * config.head_dim;
        for (std::size_t output = 0; output < config.head_dim; ++output)
        {
            for (std::size_t input = 0; input < config.head_dim; ++input)
            {
                weight.at(begin + output, begin + input) = scale;
            }
        }
    }
    return weight;
}

struct AttentionData
{
    std::vector<GpuEncryptedTensor> repeated_values;
    std::vector<std::vector<GpuEncryptedTensor>> scores;
};

AttentionData make_attention_data(
    const GpuEncryptedTensor &query,
    const GpuEncryptedTensor &all_key,
    const GpuEncryptedTensor &all_value,
    const qwen::QwenConfig &config,
    std::size_t position_offset, const Runtime &runtime)
{
    const qwen::Tensor repeat_weight = repeat_kv_weight(config);
    auto repeated_keys = linear(
        all_key, repeat_weight, nullptr, runtime);
    auto repeated_values_tensor = linear(
        all_value, repeat_weight, nullptr, runtime);
    std::vector<GpuEncryptedTensor> repeated_key_tokens;
    std::vector<GpuEncryptedTensor> repeated_value_tokens;
    repeated_key_tokens.reserve(all_key.layout().tokens);
    repeated_value_tokens.reserve(all_value.layout().tokens);
    for (std::size_t token = 0; token < all_key.layout().tokens; ++token)
    {
        repeated_key_tokens.push_back(token_view(
            repeated_keys, token, runtime));
        repeated_value_tokens.push_back(token_view(
            repeated_values_tensor, token, runtime));
    }

    const qwen::Tensor sum_weight = head_sum_weight(config);
    std::vector<std::vector<GpuEncryptedTensor>> scores(
        query.layout().tokens);
    for (std::size_t query_token = 0;
         query_token < query.layout().tokens; ++query_token)
    {
        const auto query_view = token_view(query, query_token, runtime);
        const std::size_t visible = position_offset + query_token + 1;
        scores[query_token].reserve(visible);
        for (std::size_t key_token = 0; key_token < visible; ++key_token)
        {
            auto product = multiply(
                query_view, repeated_key_tokens[key_token], runtime);
            auto score = linear(product, sum_weight, nullptr, runtime);
            if (attention_validation_enabled())
            {
                report_score_diagnostic(
                    "row_" + std::to_string(query_token) + ".key_" +
                        std::to_string(key_token),
                    query_view, repeated_key_tokens[key_token], score,
                    config, runtime);
            }
            scores[query_token].push_back(std::move(score));
        }
    }
    return {std::move(repeated_value_tokens), std::move(scores)};
}

qwen::he::ApproximationConfig dual_sigmoid_config(
    const qwen::he::StableAttentionApproximationConfig &config)
{
    const double bound = std::max(
        std::abs(config.exponential.minimum),
        std::abs(config.exponential.maximum));
    return {-bound, bound,
            std::max(32, config.exponential.sample_count)};
}

qwen::he::ApproximationConfig online_config(
    const qwen::he::StableAttentionApproximationConfig &config)
{
    const double score_bound = std::max(
        std::abs(config.exponential.minimum),
        std::abs(config.exponential.maximum));
    const double margin = std::log(std::max(
        1.0, config.reciprocal.maximum));
    const double bound = score_bound + margin;
    return {-bound, bound,
            std::max(128, config.exponential.sample_count)};
}

double gpu_attention_bootstrap_scale(double requested)
{
    // The CUDA bootstrap is most accurate when its external value scaling is
    // a power of two. Attention's Trident calibration can request values such
    // as 1.69, which leaves layer-0 deltas near the EvalMod boundary and was
    // measured to introduce O(1) error. A scale of four keeps the calibrated
    // deltas in the high-precision interval without unnecessary amplification.
    const double bounded = std::max(4.0, requested);
    return std::exp2(std::ceil(std::log2(bounded)));
}

GpuEncryptedTensor maybe_refresh_aggregate(
    const GpuEncryptedTensor &aggregate, const Runtime &runtime)
{
    std::size_t minimum_q = aggregate.ciphertexts().front().meta.q_count;
    for (const auto &cipher : aggregate.ciphertexts())
    {
        minimum_q = std::min(minimum_q, cipher.meta.q_count);
    }
    return minimum_q <= 9
        ? bootstrap(aggregate, runtime)
        : token_view(aggregate, 0, runtime);
}

}  // namespace

bool GpuKVCache::empty() const noexcept
{
    return key_.ciphertexts().empty();
}

std::size_t GpuKVCache::size() const noexcept
{
    return empty() ? 0 : key_.layout().tokens;
}

void GpuKVCache::clear()
{
    key_ = GpuEncryptedTensor{};
    value_ = GpuEncryptedTensor{};
}

void GpuKVCache::append(
    const GpuEncryptedTensor &key,
    const GpuEncryptedTensor &value,
    const Runtime &runtime)
{
    if (key.layout().tokens != value.layout().tokens ||
        key.layout().features != value.layout().features)
    {
        throw std::invalid_argument(
            "GPU Qwen KV cache append layouts do not match");
    }
    std::vector<GpuEncryptedTensor> keys;
    std::vector<GpuEncryptedTensor> values;
    keys.reserve(size() + key.layout().tokens);
    values.reserve(size() + value.layout().tokens);
    if (!empty())
    {
        for (std::size_t token = 0; token < size(); ++token)
        {
            keys.push_back(token_view(key_, token, runtime));
            values.push_back(token_view(value_, token, runtime));
        }
    }
    for (std::size_t token = 0; token < key.layout().tokens; ++token)
    {
        keys.push_back(token_view(key, token, runtime));
        values.push_back(token_view(value, token, runtime));
    }
    key_ = concatenate_tokens(std::move(keys), runtime);
    value_ = concatenate_tokens(std::move(values), runtime);
}

const GpuEncryptedTensor &GpuKVCache::key() const
{
    if (empty())
    {
        throw std::logic_error("GPU Qwen KV cache is empty");
    }
    return key_;
}

const GpuEncryptedTensor &GpuKVCache::value() const
{
    if (empty())
    {
        throw std::logic_error("GPU Qwen KV cache is empty");
    }
    return value_;
}

GpuEncryptedTensor stable_causal_gqa_attention(
    const GpuEncryptedTensor &query,
    const GpuEncryptedTensor &key,
    const GpuEncryptedTensor &value,
    const qwen::QwenConfig &model_config,
    const qwen::he::StableAttentionApproximationConfig &approximation,
    const Runtime &runtime, GpuKVCache *cache)
{
    validate_qkv(query, key, value, model_config);
    approximation.validate();
    const GpuEncryptedTensor *all_key = &key;
    const GpuEncryptedTensor *all_value = &value;
    std::size_t position_offset = 0;
    if (cache != nullptr)
    {
        position_offset = cache->size();
        cache->append(key, value, runtime);
        all_key = &cache->key();
        all_value = &cache->value();
    }
    AttentionData attention = make_attention_data(
        query, *all_key, *all_value, model_config,
        position_offset, runtime);
    const auto dual = dual_sigmoid_config(approximation);
    const auto online = online_config(approximation);
    const double dual_scale = gpu_attention_bootstrap_scale(
        approximation.dual_token_bootstrap_value_scale);
    const double online_scale = gpu_attention_bootstrap_scale(std::max(
        approximation.dual_token_bootstrap_value_scale,
        std::max(std::abs(online.minimum), std::abs(online.maximum)) /
            16.0));
    if (attention_validation_enabled())
    {
        std::cout << "qwen_gpu_attention_diagnostic dual_bootstrap_scale="
                  << dual_scale
                  << " online_bootstrap_scale=" << online_scale << '\n';
    }

    std::vector<GpuEncryptedTensor> outputs;
    outputs.reserve(query.layout().tokens);
    for (std::size_t query_token = 0;
         query_token < query.layout().tokens; ++query_token)
    {
        const auto &scores = attention.scores[query_token];
        if (scores.size() == 1)
        {
            outputs.push_back(token_view(
                attention.repeated_values[0], 0, runtime));
            continue;
        }
        if (scores.size() == 2)
        {
            auto delta = subtract(scores[0], scores[1], runtime);
            std::optional<qwen::Tensor> delta_before;
            if (attention_validation_enabled())
            {
                delta_before.emplace(decrypt_tensor(delta, runtime));
            }
            delta = bootstrap(
                delta, runtime, dual_scale);
            if (delta_before.has_value())
            {
                report_bootstrap_diagnostic(
                    "row_" + std::to_string(query_token) +
                        ".dual_delta",
                    *delta_before, delta, runtime);
            }
            const auto probability = sigmoid(
                delta, dual.minimum, dual.maximum,
                dual.sample_count, runtime);
            if (attention_validation_enabled())
            {
                report_polynomial_diagnostic(
                    "row_" + std::to_string(query_token) +
                        ".dual_sigmoid",
                    delta, probability, dual, false, runtime);
            }
            const auto difference = subtract(
                attention.repeated_values[0],
                attention.repeated_values[1], runtime);
            outputs.push_back(add(
                attention.repeated_values[1],
                multiply(probability, difference, runtime), runtime));
            continue;
        }

        auto log_normalizer = token_view(scores[0], 0, runtime);
        auto aggregate = token_view(
            attention.repeated_values[0], 0, runtime);
        for (std::size_t key_token = 1;
             key_token < scores.size(); ++key_token)
        {
            auto delta = subtract(
                scores[key_token], log_normalizer, runtime);
            std::optional<qwen::Tensor> delta_before;
            if (attention_validation_enabled())
            {
                delta_before.emplace(decrypt_tensor(delta, runtime));
            }
            delta = bootstrap(delta, runtime, online_scale);
            if (delta_before.has_value())
            {
                report_bootstrap_diagnostic(
                    "row_" + std::to_string(query_token) + ".step_" +
                        std::to_string(key_token) + ".delta",
                    *delta_before, delta, runtime);
            }
            const auto probability = sigmoid(
                delta, online.minimum, online.maximum,
                online.sample_count, runtime);
            const auto correction = softplus(
                delta, online.minimum, online.maximum,
                online.sample_count, runtime);
            if (attention_validation_enabled())
            {
                const std::string prefix =
                    "row_" + std::to_string(query_token) + ".step_" +
                    std::to_string(key_token);
                report_polynomial_diagnostic(
                    prefix + ".sigmoid", delta, probability,
                    online, false, runtime);
                report_polynomial_diagnostic(
                    prefix + ".softplus", delta, correction,
                    online, true, runtime);
            }
            log_normalizer = add(
                log_normalizer, correction, runtime);
            aggregate = maybe_refresh_aggregate(aggregate, runtime);
            const auto difference = subtract(
                attention.repeated_values[key_token],
                aggregate, runtime);
            aggregate = add(
                aggregate,
                multiply(probability, difference, runtime), runtime);
        }
        outputs.push_back(std::move(aggregate));
    }
    return concatenate_tokens(std::move(outputs), runtime);
}

}  // namespace poseidon::benchmark::qwen_gpu
