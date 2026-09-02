#include "qwen_gpu_ops.h"
#include "qwen_gpu_decoder.h"
#include "qwen_gpu_tensor.h"

#include "he/qwen25_05b_config.h"
#include "model/plain_qwen.h"
#include "model/qwen_config.h"
#include "ops/plain_ops.h"

#include "resnet50_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using poseidon::benchmark::qwen_gpu::Runtime;

int selected_device()
{
    const char *value = std::getenv("POSEIDON_GPU_QWEN_DEVICE");
    return value == nullptr ? 0 : std::stoi(value);
}

double maximum_error(const qwen::Tensor &actual, const qwen::Tensor &expected)
{
    if (actual.shape() != expected.shape())
    {
        throw std::invalid_argument("comparison tensor shapes do not match");
    }
    double result = 0.0;
    for (std::size_t index = 0; index < actual.numel(); ++index)
    {
        result = std::max(result, std::abs(
            actual.data()[index] - expected.data()[index]));
    }
    return result;
}

double tensor_max_abs(const qwen::Tensor &tensor)
{
    double result = 0.0;
    for (double value : tensor.data())
    {
        result = std::max(result, std::abs(value));
    }
    return result;
}

std::string polynomial_trace_name(const std::string &gpu_stage)
{
    const std::size_t separator = gpu_stage.find('.');
    const std::string name = separator == std::string::npos
        ? gpu_stage : gpu_stage.substr(separator + 1);
    if (name == "single_token_gqa_attention") return "attention";
    if (name == "input_rmsnorm_refresh") return "input_rmsnorm_refreshed";
    if (name == "query_rope_refresh") return "query_rope_refreshed";
    if (name == "key_rope_refresh") return "key_rope_refreshed";
    if (name == "attention_refresh") return "attention_refreshed";
    if (name == "attention_output_projection") return "attention_output";
    if (name == "post_attention_bootstrap") return "post_attention_refreshed";
    if (name == "mlp_input_bootstrap") return "post_attention_rmsnorm_refreshed";
    if (name == "mlp_gate_projection") return "mlp_gate";
    if (name == "mlp_up_projection") return "mlp_up";
    if (name == "mlp_down_projection") return "mlp_output";
    if (name == "output_residual") return "output";
    if (name == "output_bootstrap") return "output_refreshed";
    return name;
}

void run_smoke()
{
    const auto config =
        poseidon::benchmark::resnet50_gpu::make_resnet50_gpu_config();
    Runtime runtime(config, selected_device());
    runtime.initialize_inference_evaluation_keys();

    qwen::Tensor input({1, 4}, {0.25, -0.5, 0.75, 1.25});
    qwen::Tensor weight({3, 4}, {
        0.5, 0.0, -0.25, 0.1,
        -0.2, 0.3, 0.4, 0.0,
        0.0, -0.5, 0.25, 0.75,
    });
    qwen::Tensor bias({3}, {0.1, -0.2, 0.05});
    const qwen::Tensor expected = qwen::linear(input, weight, &bias);

    auto encrypted = poseidon::benchmark::qwen_gpu::encrypt_tensor(
        input, runtime);
    auto encrypted_output = poseidon::benchmark::qwen_gpu::linear(
        encrypted, weight, &bias, runtime);
    const qwen::Tensor output = poseidon::benchmark::qwen_gpu::decrypt_tensor(
        encrypted_output, runtime);
    const double error = maximum_error(output, expected);
    std::cout << "[PASS] GPU Qwen encrypted tensor round-trip\n"
              << "[" << (error < 1.0e-5 ? "PASS" : "FAIL")
              << "] GPU Qwen diagonal Linear max_error=" << error
              << " q_count="
              << encrypted_output.ciphertexts().front().meta.q_count << '\n';
    if (error >= 1.0e-5)
    {
        throw std::runtime_error("GPU Qwen Linear smoke test failed");
    }

    qwen::Tensor multi_input({3, 4}, {
        0.25, -0.5, 0.75, 1.25,
        -1.0, 0.125, 0.5, -0.75,
        0.0, 1.0, -0.25, 0.625,
    });
    const qwen::Tensor multi_expected = qwen::linear(
        multi_input, weight, &bias);
    const auto multi_encrypted =
        poseidon::benchmark::qwen_gpu::encrypt_tensor(
            multi_input, runtime);
    const qwen::Tensor multi_output =
        poseidon::benchmark::qwen_gpu::decrypt_tensor(
            poseidon::benchmark::qwen_gpu::linear(
                multi_encrypted, weight, &bias, runtime),
            runtime);
    const double multi_error = maximum_error(
        multi_output, multi_expected);
    std::cout << "[" << (multi_error < 1.0e-5 ? "PASS" : "FAIL")
              << "] GPU Qwen multi-token reused-plaintext Linear max_error="
              << multi_error << '\n';
    if (multi_error >= 1.0e-5)
    {
        throw std::runtime_error(
            "GPU Qwen multi-token Linear reuse smoke test failed");
    }

    qwen::Tensor giant_input({1, 64});
    qwen::Tensor giant_weight({64, 64});
    for (std::size_t feature = 0; feature < 64; ++feature)
    {
        giant_input.at(0, feature) =
            std::sin(static_cast<double>(feature) * 0.1);
    }
    for (std::size_t output_feature = 0; output_feature < 24;
         ++output_feature)
    {
        giant_weight.at(output_feature, output_feature + 40) = 1.0;
    }
    const qwen::Tensor giant_expected = qwen::linear(
        giant_input, giant_weight);
    const auto giant_encrypted =
        poseidon::benchmark::qwen_gpu::encrypt_tensor(
            giant_input, runtime);
    const qwen::Tensor giant_output =
        poseidon::benchmark::qwen_gpu::decrypt_tensor(
            poseidon::benchmark::qwen_gpu::linear(
                giant_encrypted, giant_weight, nullptr, runtime),
            runtime);
    const double giant_error = maximum_error(
        giant_output, giant_expected);
    std::cout << "[" << (giant_error < 1.0e-5 ? "PASS" : "FAIL")
              << "] GPU Qwen BSGS giant-step Linear max_error="
              << giant_error << '\n';
    if (giant_error >= 1.0e-5)
    {
        throw std::runtime_error(
            "GPU Qwen BSGS giant-step smoke test failed");
    }

    const std::vector<double> minimum(output.dim(1), -2.0);
    const std::vector<double> maximum(output.dim(1), 2.0);
    const auto encrypted_silu = poseidon::benchmark::qwen_gpu::silu(
        encrypted_output, minimum, maximum, 16, runtime);
    const qwen::Tensor silu_output =
        poseidon::benchmark::qwen_gpu::decrypt_tensor(
            encrypted_silu, runtime);
    qwen::Tensor exact_silu(output.shape());
    for (std::size_t index = 0; index < output.numel(); ++index)
    {
        exact_silu.data()[index] = output.data()[index] /
            (1.0 + std::exp(-output.data()[index]));
    }
    const double silu_error = maximum_error(silu_output, exact_silu);
    std::cout << "[" << (silu_error < 1.0e-4 ? "PASS" : "FAIL")
              << "] GPU Qwen degree-15 SiLU max_error=" << silu_error
              << " q_count="
              << encrypted_silu.ciphertexts().front().meta.q_count << '\n';
    if (silu_error >= 1.0e-4)
    {
        throw std::runtime_error("GPU Qwen SiLU smoke test failed");
    }

    qwen::Tensor norm_weight({4}, {1.0, 0.9, 1.1, 0.8});
    const auto encrypted_norm = poseidon::benchmark::qwen_gpu::rms_norm(
        encrypted, norm_weight, 1.0e-6, 0.1, 1.0, 16, runtime);
    const qwen::Tensor norm_output =
        poseidon::benchmark::qwen_gpu::decrypt_tensor(
            encrypted_norm, runtime);
    qwen::Tensor exact_norm(input.shape());
    double square_sum = 0.0;
    for (double value : input.data())
    {
        square_sum += value * value;
    }
    const double inverse_rms = 1.0 / std::sqrt(
        square_sum / static_cast<double>(input.dim(1)) + 1.0e-6);
    for (std::size_t feature = 0; feature < input.dim(1); ++feature)
    {
        exact_norm.at(0, feature) =
            input.at(0, feature) * inverse_rms * norm_weight.at(feature);
    }
    const double norm_error = maximum_error(norm_output, exact_norm);
    std::cout << "[" << (norm_error < 1.0e-4 ? "PASS" : "FAIL")
              << "] GPU Qwen degree-15 RMSNorm max_error=" << norm_error
              << " q_count="
              << encrypted_norm.ciphertexts().front().meta.q_count << '\n';
    if (norm_error >= 1.0e-4)
    {
        throw std::runtime_error("GPU Qwen RMSNorm smoke test failed");
    }
}

void run_bootstrap_scale_check()
{
    const auto config =
        poseidon::benchmark::resnet50_gpu::make_resnet50_gpu_config();
    Runtime runtime(config, selected_device());
    runtime.initialize_inference_evaluation_keys();
    runtime.initialize_bootstrap();
    const qwen::Tensor input(
        {1, 8},
        {-14.2219, -9.4878, -4.53616, -1.0,
         0.0, 1.0, 6.77265, 7.89394});
    const auto encrypted = poseidon::benchmark::qwen_gpu::encrypt_tensor(
        input, runtime);
    for (double scale : {1.7, 4.0, 8.0, 16.0, 22.3, 32.0})
    {
        const qwen::Tensor actual =
            poseidon::benchmark::qwen_gpu::decrypt_tensor(
                poseidon::benchmark::qwen_gpu::bootstrap(
                    encrypted, runtime, scale),
                runtime);
        std::cout << "qwen_gpu_bootstrap_scale_check value_scale=" << scale
                  << " max_error=" << maximum_error(actual, input) << '\n';
    }
}

void print_model_info(const std::filesystem::path &model_directory)
{
    const qwen::QwenConfig config =
        qwen::load_qwen_config(model_directory / "config.json");
    qwen::QwenModelWeights weights =
        qwen::load_qwen_model_weights(model_directory);
    weights.validate(config);
    std::cout << "[PASS] Qwen checkpoint loaded\n"
              << "model_directory=" << model_directory << '\n'
              << "layers=" << config.num_hidden_layers
              << " hidden_size=" << config.hidden_size
              << " intermediate_size=" << config.intermediate_size
              << " query_heads=" << config.num_attention_heads
              << " kv_heads=" << config.num_key_value_heads
              << " head_dim=" << config.head_dim
              << " vocabulary=" << config.vocab_size << '\n';
}

void check_topology(const std::filesystem::path &model_directory)
{
    const qwen::QwenConfig config =
        qwen::load_qwen_config(model_directory / "config.json");
    if (config.hidden_size != 896 || config.intermediate_size != 4864 ||
        config.num_hidden_layers != 24 ||
        config.num_attention_heads != 14 ||
        config.num_key_value_heads != 2 || config.head_dim != 64)
    {
        throw std::invalid_argument(
            "GPU Qwen benchmark expects the Qwen2.5-0.5B topology");
    }
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer)
    {
        auto approximation = qwen::he::qwen25_05b_layer_approximation(
            layer, 1);
        qwen::he::set_decoder_boundary_bootstrap_schedule(approximation);
        qwen::he::remove_single_token_attention_refreshes(approximation);
        qwen::he::remove_redundant_rmsnorm_refreshes(approximation);
        qwen::he::set_qwen25_05b_calibrated_bootstrap_scales(
            approximation, layer);
        approximation.validate();
        const auto feature_ranges =
            approximation.silu_feature_overrides.find(0);
        if (feature_ranges == approximation.silu_feature_overrides.end() ||
            feature_ranges->second.size() != config.intermediate_size)
        {
            throw std::runtime_error(
                "Qwen GPU layer is missing position-zero SiLU calibration");
        }
        auto multi = qwen::he::qwen25_05b_layer_approximation(layer, 2);
        qwen::he::set_decoder_multi_token_bootstrap_schedule(
            multi, qwen::he::RefreshMode::bootstrap, 2);
        qwen::he::remove_redundant_rmsnorm_refreshes(multi);
        qwen::he::set_qwen25_05b_calibrated_bootstrap_scales(
            multi, layer);
        multi.validate();
    }
    auto gpu_config =
        poseidon::benchmark::resnet50_gpu::make_resnet50_gpu_config();
    gpu_config.validate();
    std::cout << "[PASS] GPU Qwen topology and all 24 single/multi-token schedules\n"
              << "hidden=896 intermediate=4864 layers=24"
              << " slots=" << gpu_config.slot_count()
              << " application_q_count="
              << gpu_config.application_q_count() << '\n';
}

qwen::Tensor embedding_row(
    const qwen::Tensor &embedding, std::size_t token)
{
    if (embedding.rank() != 2 || token >= embedding.dim(0))
    {
        throw std::out_of_range("Qwen token ID is outside the embedding table");
    }
    qwen::Tensor result({1, embedding.dim(1)});
    for (std::size_t feature = 0; feature < embedding.dim(1); ++feature)
    {
        result.at(0, feature) = embedding.at(token, feature);
    }
    return result;
}

std::vector<std::size_t> parse_token_ids(const std::string &text)
{
    std::vector<std::size_t> result;
    std::size_t begin = 0;
    while (begin <= text.size())
    {
        const std::size_t comma = text.find(',', begin);
        const std::string item = text.substr(
            begin, comma == std::string::npos
                       ? std::string::npos : comma - begin);
        if (item.empty())
        {
            throw std::invalid_argument("empty Qwen token ID");
        }
        result.push_back(std::stoull(item));
        if (comma == std::string::npos)
        {
            break;
        }
        begin = comma + 1;
    }
    if (result.empty())
    {
        throw std::invalid_argument("Qwen token list must not be empty");
    }
    return result;
}

qwen::Tensor embedding_rows(
    const qwen::Tensor &embedding,
    const std::vector<std::size_t> &token_ids)
{
    if (embedding.rank() != 2 || token_ids.empty())
    {
        throw std::invalid_argument("invalid Qwen embedding request");
    }
    qwen::Tensor result({token_ids.size(), embedding.dim(1)});
    for (std::size_t token = 0; token < token_ids.size(); ++token)
    {
        if (token_ids[token] >= embedding.dim(0))
        {
            throw std::out_of_range(
                "Qwen token ID is outside the embedding table");
        }
        for (std::size_t feature = 0; feature < embedding.dim(1); ++feature)
        {
            result.at(token, feature) =
                embedding.at(token_ids[token], feature);
        }
    }
    return result;
}

qwen::Tensor last_token(const qwen::Tensor &input)
{
    if (input.rank() != 2 || input.dim(0) == 0)
    {
        throw std::invalid_argument("last_token expects a nonempty rank-2 tensor");
    }
    qwen::Tensor result({1, input.dim(1)});
    for (std::size_t feature = 0; feature < input.dim(1); ++feature)
    {
        result.at(0, feature) = input.at(input.dim(0) - 1, feature);
    }
    return result;
}

std::size_t argmax_token(const qwen::Tensor &logits)
{
    if (logits.numel() == 0)
    {
        throw std::invalid_argument("argmax requires a nonempty tensor");
    }
    return static_cast<std::size_t>(std::max_element(
        logits.data().begin(), logits.data().end()) -
        logits.data().begin());
}

qwen::he::ApproximationConfig partial_final_norm_config(
    const qwen::Tensor &hidden, double epsilon)
{
    if (hidden.rank() != 2 || hidden.dim(1) == 0 || epsilon <= 0.0)
    {
        throw std::invalid_argument(
            "partial final RMSNorm calibration received invalid input");
    }
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = 0.0;
    for (std::size_t token = 0; token < hidden.dim(0); ++token)
    {
        double square_sum = 0.0;
        for (std::size_t feature = 0; feature < hidden.dim(1); ++feature)
        {
            const double value = hidden.at(token, feature);
            square_sum += value * value;
        }
        const double variance =
            square_sum / static_cast<double>(hidden.dim(1)) + epsilon;
        minimum = std::min(minimum, variance);
        maximum = std::max(maximum, variance);
    }
    const double lower = std::max(epsilon, minimum * 0.80);
    const double upper = std::max(lower * 1.05, maximum * 1.20);
    const double ratio = upper / lower;
    return {lower, upper, ratio > 20.0 ? 128 : (ratio > 5.0 ? 64 : 32)};
}

qwen::he::EncryptedDecoderApproximationConfig generation_approximation(
    std::size_t layer, std::size_t maximum_attention_tokens)
{
    auto approximation = qwen::he::qwen25_05b_layer_approximation(
        layer, maximum_attention_tokens);
    qwen::he::set_decoder_bootstrap_schedule(
        approximation, qwen::he::RefreshMode::bootstrap);
    if (maximum_attention_tokens > 1)
    {
        qwen::he::set_decoder_multi_token_bootstrap_schedule(
            approximation, qwen::he::RefreshMode::bootstrap,
            maximum_attention_tokens);
    }
    else
    {
        qwen::he::set_decoder_boundary_bootstrap_schedule(approximation);
        qwen::he::remove_single_token_attention_refreshes(approximation);
    }
    qwen::he::remove_redundant_rmsnorm_refreshes(approximation);
    qwen::he::set_qwen25_05b_calibrated_bootstrap_scales(
        approximation, layer);
    approximation.validate();
    return approximation;
}

void run_single_token_inference(
    const std::filesystem::path &model_directory,
    std::size_t token_id, std::size_t maximum_layers)
{
    const qwen::QwenConfig model_config =
        qwen::load_qwen_config(model_directory / "config.json");
    qwen::QwenModelWeights weights =
        qwen::load_qwen_model_weights(model_directory);
    maximum_layers = std::min(maximum_layers, model_config.num_hidden_layers);
    if (maximum_layers == 0)
    {
        throw std::invalid_argument("GPU Qwen inference requires at least one layer");
    }

    const bool validate_stages =
        std::getenv("POSEIDON_GPU_QWEN_VALIDATE") != nullptr;
    const qwen::Tensor input = embedding_row(
        weights.token_embedding, token_id);
    std::vector<qwen::he::EncryptedDecoderApproximationConfig>
        approximations;
    std::vector<qwen::Tensor> exact_outputs;
    std::vector<qwen::Tensor> polynomial_outputs;
    std::vector<std::map<std::string, qwen::Tensor>> polynomial_traces(
        maximum_layers);
    approximations.reserve(maximum_layers);
    exact_outputs.reserve(maximum_layers);
    polynomial_outputs.reserve(maximum_layers);
    qwen::Tensor exact_cursor = input;
    qwen::Tensor polynomial_cursor = input;
    for (std::size_t layer_index = 0;
         layer_index < maximum_layers; ++layer_index)
    {
        auto approximation = qwen::he::qwen25_05b_layer_approximation(
            layer_index, 1);
        qwen::he::set_decoder_boundary_bootstrap_schedule(approximation);
        qwen::he::remove_single_token_attention_refreshes(approximation);
        qwen::he::remove_redundant_rmsnorm_refreshes(approximation);
        qwen::he::set_qwen25_05b_calibrated_bootstrap_scales(
            approximation, layer_index);
        approximation.validate();
        polynomial_cursor = qwen::he::approximate_decoder_layer(
            polynomial_cursor, weights.layers[layer_index], model_config,
            approximation, 0,
            [&](const std::string &name, const qwen::Tensor &tensor) {
                if (validate_stages)
                {
                    polynomial_traces[layer_index].emplace(name, tensor);
                }
            });
        qwen::PlainDecoderLayer exact_layer(
            model_config, weights.layers[layer_index]);
        exact_cursor = exact_layer.forward(exact_cursor);
        const double approximation_error = maximum_error(
            polynomial_cursor, exact_cursor);
        const double approximation_tolerance =
            1.0e-2 + 5.0e-4 * tensor_max_abs(exact_cursor);
        std::cout << "qwen_gpu_preflight layer=" << layer_index
                  << " polynomial_vs_exact_max_error="
                  << approximation_error << " tolerance="
                  << approximation_tolerance << '\n';
        if (approximation_error > approximation_tolerance)
        {
            throw std::runtime_error(
                "Qwen token is outside the calibrated polynomial domain at layer " +
                std::to_string(layer_index));
        }
        approximations.push_back(std::move(approximation));
        exact_outputs.push_back(exact_cursor);
        polynomial_outputs.push_back(polynomial_cursor);
    }

    const auto he_config =
        poseidon::benchmark::resnet50_gpu::make_resnet50_gpu_config();
    Runtime runtime(he_config, selected_device());
    runtime.initialize_inference_evaluation_keys();
    runtime.initialize_bootstrap();
    auto encrypted = poseidon::benchmark::qwen_gpu::encrypt_tensor(
        input, runtime);

    for (std::size_t layer_index = 0;
         layer_index < maximum_layers; ++layer_index)
    {
        const auto &approximation = approximations[layer_index];
        const auto &polynomial_trace = polynomial_traces[layer_index];

        encrypted = poseidon::benchmark::qwen_gpu::single_token_decoder_layer(
            encrypted, weights.layers[layer_index], model_config,
            approximation, layer_index, runtime,
            validate_stages
                ? poseidon::benchmark::qwen_gpu::DecoderTrace{
                      [&](const std::string &stage,
                          const poseidon::benchmark::qwen_gpu::GpuEncryptedTensor
                              &tensor) {
                          const std::string reference_name =
                              polynomial_trace_name(stage);
                          const auto reference =
                              polynomial_trace.find(reference_name);
                          if (reference == polynomial_trace.end())
                          {
                              throw std::runtime_error(
                                  "missing polynomial trace for GPU stage " +
                                  stage);
                          }
                          const qwen::Tensor actual =
                              poseidon::benchmark::qwen_gpu::decrypt_tensor(
                                  tensor, runtime);
                          std::cout
                              << "qwen_gpu_trace stage=" << stage
                              << " max_error="
                              << maximum_error(actual, reference->second)
                              << " reference_max_abs="
                              << tensor_max_abs(reference->second) << '\n';
                      }}
                : poseidon::benchmark::qwen_gpu::DecoderTrace{});
        if (validate_stages || layer_index + 1 == maximum_layers)
        {
            const qwen::Tensor actual =
                poseidon::benchmark::qwen_gpu::decrypt_tensor(
                    encrypted, runtime);
            std::cout << "qwen_gpu_validation layer=" << layer_index
                      << " ckks_vs_polynomial_max_error="
                      << maximum_error(
                             actual, polynomial_outputs[layer_index])
                      << " polynomial_vs_exact_max_error="
                      << maximum_error(
                             polynomial_outputs[layer_index],
                             exact_outputs[layer_index]) << '\n';
        }
    }
    const qwen::Tensor &polynomial = polynomial_outputs.back();
    const qwen::Tensor &exact = exact_outputs.back();
    qwen::Tensor actual = poseidon::benchmark::qwen_gpu::decrypt_tensor(
        encrypted, runtime);
    const double final_error = maximum_error(actual, polynomial);
    const double final_tolerance =
        1.0e-2 + 2.0e-5 * tensor_max_abs(polynomial);
    std::cout << "qwen_gpu_final ckks_vs_polynomial_max_error="
              << final_error << " tolerance=" << final_tolerance << '\n';
    if (final_error > final_tolerance)
    {
        throw std::runtime_error(
            "GPU Qwen encrypted decoder exceeded CKKS tolerance");
    }

    if (maximum_layers == model_config.num_hidden_layers)
    {
        const auto final_config =
            qwen::he::qwen25_05b_final_inverse_sqrt_config();
        const qwen::Tensor exact_final = qwen::rms_norm(
            exact, weights.final_norm, model_config.rms_norm_epsilon);
        const qwen::Tensor polynomial_final =
            qwen::he::approximate_rms_norm_plain(
                polynomial, weights.final_norm,
                model_config.rms_norm_epsilon, final_config);
        auto encrypted_final = poseidon::benchmark::qwen_gpu::rms_norm(
            encrypted, weights.final_norm, model_config.rms_norm_epsilon,
            final_config.minimum, final_config.maximum,
            final_config.sample_count, runtime);
        encrypted_final = poseidon::benchmark::qwen_gpu::bootstrap(
            encrypted_final, runtime);
        const qwen::Tensor decrypted_final =
            poseidon::benchmark::qwen_gpu::decrypt_tensor(
                encrypted_final, runtime);
        const qwen::Tensor &lm_head = weights.lm_head.empty()
            ? weights.token_embedding : weights.lm_head;
        const qwen::Tensor exact_logits =
            qwen::linear(exact_final, lm_head);
        const qwen::Tensor polynomial_logits =
            qwen::linear(polynomial_final, lm_head);
        const qwen::Tensor gpu_logits =
            qwen::linear(decrypted_final, lm_head);
        const auto argmax = [](const qwen::Tensor &logits) {
            return static_cast<std::size_t>(std::max_element(
                logits.data().begin(), logits.data().end()) -
                logits.data().begin());
        };
        const std::size_t exact_token = argmax(exact_logits);
        const std::size_t polynomial_token = argmax(polynomial_logits);
        const std::size_t gpu_token = argmax(gpu_logits);
        const double final_norm_error = maximum_error(
            decrypted_final, polynomial_final);
        const double final_norm_approximation_error = maximum_error(
            polynomial_final, exact_final);
        const double logits_error = maximum_error(
            gpu_logits, polynomial_logits);
        std::cout << "qwen_gpu_final_rmsnorm ckks_vs_polynomial_max_error="
                  << final_norm_error
                  << " polynomial_vs_exact_max_error="
                  << final_norm_approximation_error << '\n'
                  << "qwen_gpu_client_logits ckks_vs_polynomial_max_error="
                  << logits_error << " exact_argmax=" << exact_token
                  << " polynomial_argmax=" << polynomial_token
                  << " gpu_argmax=" << gpu_token << '\n';
        if (exact_token != polynomial_token ||
            polynomial_token != gpu_token || logits_error > 1.0e-1 ||
            final_norm_error >
                1.0e-2 + 5.0e-4 * tensor_max_abs(polynomial_final) ||
            final_norm_approximation_error >
                1.0e-2 + 5.0e-4 * tensor_max_abs(exact_final))
        {
            throw std::runtime_error(
                "GPU Qwen final client logits validation failed");
        }
    }
    std::cout << "[PASS] GPU Qwen encrypted decoder layers="
              << maximum_layers << " token_id=" << token_id << '\n';
}

void run_multi_token_inference(
    const std::filesystem::path &model_directory,
    const std::vector<std::size_t> &token_ids,
    std::size_t maximum_layers)
{
    if (token_ids.size() < 2)
    {
        throw std::invalid_argument(
            "--infer-ids requires at least two token IDs");
    }
    const qwen::QwenConfig model_config =
        qwen::load_qwen_config(model_directory / "config.json");
    qwen::QwenModelWeights weights =
        qwen::load_qwen_model_weights(model_directory);
    maximum_layers = std::min(maximum_layers, model_config.num_hidden_layers);
    if (maximum_layers == 0)
    {
        throw std::invalid_argument("GPU Qwen inference requires at least one layer");
    }
    const bool validate_stages =
        std::getenv("POSEIDON_GPU_QWEN_VALIDATE") != nullptr;
    const qwen::Tensor input = embedding_rows(
        weights.token_embedding, token_ids);
    std::vector<qwen::he::EncryptedDecoderApproximationConfig>
        approximations;
    std::vector<qwen::Tensor> exact_outputs;
    std::vector<qwen::Tensor> polynomial_outputs;
    std::vector<std::map<std::string, qwen::Tensor>> polynomial_traces(
        maximum_layers);
    approximations.reserve(maximum_layers);
    exact_outputs.reserve(maximum_layers);
    polynomial_outputs.reserve(maximum_layers);
    qwen::Tensor exact_cursor = input;
    qwen::Tensor polynomial_cursor = input;
    for (std::size_t layer = 0; layer < maximum_layers; ++layer)
    {
        auto approximation = qwen::he::qwen25_05b_layer_approximation(
            layer, token_ids.size());
        qwen::he::set_decoder_multi_token_bootstrap_schedule(
            approximation, qwen::he::RefreshMode::bootstrap,
            token_ids.size());
        qwen::he::remove_redundant_rmsnorm_refreshes(approximation);
        qwen::he::set_qwen25_05b_calibrated_bootstrap_scales(
            approximation, layer);
        approximation.validate();
        polynomial_cursor = qwen::he::approximate_decoder_layer(
            polynomial_cursor, weights.layers[layer], model_config,
            approximation, 0,
            [&](const std::string &name, const qwen::Tensor &tensor) {
                if (validate_stages)
                {
                    polynomial_traces[layer].emplace(name, tensor);
                }
            });
        qwen::PlainDecoderLayer exact_layer(
            model_config, weights.layers[layer]);
        exact_cursor = exact_layer.forward(exact_cursor);
        const double approximation_error = maximum_error(
            polynomial_cursor, exact_cursor);
        const double approximation_tolerance =
            1.0e-2 + 5.0e-4 * tensor_max_abs(exact_cursor);
        std::cout << "qwen_gpu_preflight layer=" << layer
                  << " tokens=" << token_ids.size()
                  << " polynomial_vs_exact_max_error="
                  << approximation_error << " tolerance="
                  << approximation_tolerance << '\n';
        if (approximation_error > approximation_tolerance)
        {
            throw std::runtime_error(
                "Qwen prompt is outside the calibrated polynomial domain at layer " +
                std::to_string(layer));
        }
        approximations.push_back(std::move(approximation));
        exact_outputs.push_back(exact_cursor);
        polynomial_outputs.push_back(polynomial_cursor);
    }

    const auto he_config =
        poseidon::benchmark::resnet50_gpu::make_resnet50_gpu_config();
    Runtime runtime(he_config, selected_device());
    runtime.initialize_inference_evaluation_keys();
    runtime.initialize_bootstrap();
    auto encrypted = poseidon::benchmark::qwen_gpu::encrypt_tensor(
        input, runtime);
    std::vector<poseidon::benchmark::qwen_gpu::GpuKVCache> caches(
        maximum_layers);
    for (std::size_t layer = 0; layer < maximum_layers; ++layer)
    {
        const auto &polynomial_trace = polynomial_traces[layer];
        encrypted = poseidon::benchmark::qwen_gpu::decoder_layer(
            encrypted, weights.layers[layer], model_config,
            approximations[layer], layer, 0, runtime,
            validate_stages
                ? poseidon::benchmark::qwen_gpu::DecoderTrace{
                      [&](const std::string &stage,
                          const poseidon::benchmark::qwen_gpu::GpuEncryptedTensor
                              &tensor) {
                          const std::string reference_name =
                              polynomial_trace_name(stage);
                          const auto reference =
                              polynomial_trace.find(reference_name);
                          if (reference == polynomial_trace.end())
                          {
                              throw std::runtime_error(
                                  "missing multi-token polynomial trace for " +
                                  stage);
                          }
                          const qwen::Tensor actual =
                              poseidon::benchmark::qwen_gpu::decrypt_tensor(
                                  tensor, runtime);
                          std::cout << "qwen_gpu_trace stage=" << stage
                                    << " max_error="
                                    << maximum_error(
                                           actual, reference->second)
                                    << '\n';
                      }}
                : poseidon::benchmark::qwen_gpu::DecoderTrace{},
            &caches[layer]);
        const qwen::Tensor actual =
            poseidon::benchmark::qwen_gpu::decrypt_tensor(
                encrypted, runtime);
        std::cout << "qwen_gpu_validation layer=" << layer
                  << " tokens=" << token_ids.size()
                  << " ckks_vs_polynomial_max_error="
                  << maximum_error(actual, polynomial_outputs[layer])
                  << " cache_tokens=" << caches[layer].size() << '\n';
    }
    const qwen::Tensor &polynomial = polynomial_outputs.back();
    const qwen::Tensor &exact = exact_outputs.back();
    const qwen::Tensor actual =
        poseidon::benchmark::qwen_gpu::decrypt_tensor(encrypted, runtime);
    const double final_error = maximum_error(actual, polynomial);
    const double tolerance =
        1.0e-2 + 2.0e-5 * tensor_max_abs(polynomial);
    if (final_error > tolerance)
    {
        throw std::runtime_error(
            "multi-token GPU Qwen decoder exceeded CKKS tolerance");
    }

    if (maximum_layers == model_config.num_hidden_layers)
    {
        const auto final_config =
            qwen::he::qwen25_05b_final_inverse_sqrt_config();
        const qwen::Tensor exact_final = qwen::rms_norm(
            exact, weights.final_norm, model_config.rms_norm_epsilon);
        const qwen::Tensor polynomial_final =
            qwen::he::approximate_rms_norm_plain(
                polynomial, weights.final_norm,
                model_config.rms_norm_epsilon, final_config);
        auto encrypted_final = poseidon::benchmark::qwen_gpu::rms_norm(
            encrypted, weights.final_norm, model_config.rms_norm_epsilon,
            final_config.minimum, final_config.maximum,
            final_config.sample_count, runtime);
        encrypted_final = poseidon::benchmark::qwen_gpu::bootstrap(
            encrypted_final, runtime);
        const qwen::Tensor decrypted_final =
            poseidon::benchmark::qwen_gpu::decrypt_tensor(
                encrypted_final, runtime);
        const qwen::Tensor &lm_head = weights.lm_head.empty()
            ? weights.token_embedding : weights.lm_head;
        const qwen::Tensor exact_logits = qwen::linear(
            last_token(exact_final), lm_head);
        const qwen::Tensor polynomial_logits = qwen::linear(
            last_token(polynomial_final), lm_head);
        const qwen::Tensor gpu_logits = qwen::linear(
            last_token(decrypted_final), lm_head);
        const auto argmax = [](const qwen::Tensor &logits) {
            return static_cast<std::size_t>(std::max_element(
                logits.data().begin(), logits.data().end()) -
                logits.data().begin());
        };
        const double final_norm_error = maximum_error(
            decrypted_final, polynomial_final);
        const double final_norm_approximation_error = maximum_error(
            polynomial_final, exact_final);
        const double logits_error = maximum_error(
            gpu_logits, polynomial_logits);
        const std::size_t exact_token = argmax(exact_logits);
        const std::size_t polynomial_token = argmax(polynomial_logits);
        const std::size_t gpu_token = argmax(gpu_logits);
        std::cout << "qwen_gpu_final_rmsnorm ckks_vs_polynomial_max_error="
                  << final_norm_error
                  << " polynomial_vs_exact_max_error="
                  << final_norm_approximation_error << '\n'
                  << "qwen_gpu_client_logits ckks_vs_polynomial_max_error="
                  << logits_error << " exact_argmax=" << exact_token
                  << " polynomial_argmax=" << polynomial_token
                  << " gpu_argmax=" << gpu_token << '\n';
        if (exact_token != polynomial_token ||
            polynomial_token != gpu_token || logits_error > 1.0e-1 ||
            final_norm_error >
                1.0e-2 + 5.0e-4 * tensor_max_abs(polynomial_final) ||
            final_norm_approximation_error >
                1.0e-2 + 5.0e-4 * tensor_max_abs(exact_final))
        {
            throw std::runtime_error(
                "multi-token GPU Qwen final logits validation failed");
        }
    }
    std::cout << "[PASS] GPU Qwen encrypted prefill tokens="
              << token_ids.size() << " layers=" << maximum_layers << '\n';
}

struct GenerationReferenceStep
{
    std::vector<qwen::Tensor> layer_outputs;
    std::vector<std::map<std::string, qwen::Tensor>> layer_traces;
    qwen::he::ApproximationConfig final_norm_config;
    qwen::Tensor polynomial_final;
    std::size_t exact_token = 0;
    std::size_t polynomial_token = 0;
};

void run_generation(
    const std::filesystem::path &model_directory,
    const std::vector<std::size_t> &prompt,
    std::size_t maximum_new_tokens, std::size_t maximum_layers)
{
    if (prompt.empty())
    {
        throw std::invalid_argument(
            "--generate-ids requires a nonempty prompt");
    }
    if (maximum_new_tokens == 0)
    {
        throw std::invalid_argument(
            "--generate-ids requires at least one new token");
    }
    const bool allow_tolerance_miss =
        std::getenv("POSEIDON_GPU_QWEN_ALLOW_TOLERANCE_MISS") != nullptr;
    const bool preflight_only =
        std::getenv("POSEIDON_GPU_QWEN_PREFLIGHT_ONLY") != nullptr;
    const bool validate_stages =
        std::getenv("POSEIDON_GPU_QWEN_VALIDATE") != nullptr;
    const bool run_reference_preflight = validate_stages || preflight_only;
    bool reference_tolerance_miss = false;
    const qwen::QwenConfig model_config =
        qwen::load_qwen_config(model_directory / "config.json");
    qwen::QwenModelWeights weights =
        qwen::load_qwen_model_weights(model_directory);
    maximum_layers = std::min(maximum_layers, model_config.num_hidden_layers);
    if (maximum_layers == 0)
    {
        throw std::invalid_argument(
            "GPU Qwen generation requires at least one layer");
    }
    const char *gpu_layer_limit_text =
        std::getenv("POSEIDON_GPU_QWEN_DIAGNOSTIC_LAYER_LIMIT");
    const std::size_t gpu_layer_limit = gpu_layer_limit_text == nullptr
        ? maximum_layers
        : std::min(maximum_layers,
                   static_cast<std::size_t>(
                       std::stoull(gpu_layer_limit_text)));
    if (gpu_layer_limit == 0)
    {
        throw std::invalid_argument(
            "POSEIDON_GPU_QWEN_DIAGNOSTIC_LAYER_LIMIT must be positive");
    }
    const std::size_t maximum_attention_tokens =
        prompt.size() + maximum_new_tokens - 1;
    std::vector<qwen::he::EncryptedDecoderApproximationConfig>
        approximations;
    approximations.reserve(maximum_layers);
    for (std::size_t layer = 0; layer < maximum_layers; ++layer)
    {
        approximations.push_back(generation_approximation(
            layer, maximum_attention_tokens));
    }

    // Run the exact and polynomial paths first. Besides establishing the
    // expected greedy tokens, this rejects prompts outside Trident's
    // calibrated approximation domains before allocating CUDA memory.
    std::vector<qwen::KVCache> exact_caches(maximum_layers);
    std::vector<qwen::KVCache> polynomial_caches(maximum_layers);
    std::vector<GenerationReferenceStep> references;
    references.reserve(maximum_new_tokens);
    std::vector<std::size_t> reference_ids = prompt;
    std::size_t position_offset = 0;
    const qwen::Tensor &lm_head = weights.lm_head.empty()
        ? weights.token_embedding : weights.lm_head;
    if (run_reference_preflight)
    {
      for (std::size_t step = 0; step < maximum_new_tokens; ++step)
      {
        qwen::Tensor exact_hidden = embedding_rows(
            weights.token_embedding, reference_ids);
        qwen::Tensor polynomial_hidden = exact_hidden;
        GenerationReferenceStep reference;
        reference.layer_outputs.reserve(maximum_layers);
        reference.layer_traces.resize(maximum_layers);
        for (std::size_t layer = 0; layer < maximum_layers; ++layer)
        {
            qwen::PlainDecoderLayer exact_layer(
                model_config, weights.layers[layer]);
            exact_hidden = exact_layer.forward(
                exact_hidden, &exact_caches[layer]);
            polynomial_hidden = qwen::he::approximate_decoder_layer(
                polynomial_hidden, weights.layers[layer], model_config,
                approximations[layer], position_offset,
                [&](const std::string &name, const qwen::Tensor &tensor) {
                    if (validate_stages)
                    {
                        reference.layer_traces[layer].emplace(name, tensor);
                    }
                },
                &polynomial_caches[layer]);
            const double approximation_error = maximum_error(
                polynomial_hidden, exact_hidden);
            const double tolerance =
                1.0e-2 + 5.0e-4 * tensor_max_abs(exact_hidden);
            std::cout << "qwen_gpu_generate_preflight step=" << step
                      << " layer=" << layer
                      << " position_offset=" << position_offset
                      << " cache_tokens=" << polynomial_caches[layer].size()
                      << " polynomial_vs_exact_max_error="
                      << approximation_error << " tolerance=" << tolerance
                      << '\n';
            if (approximation_error > tolerance)
            {
                reference_tolerance_miss = true;
                std::cout << "qwen_gpu_generate_warning step=" << step
                          << " layer=" << layer
                          << " type=polynomial_tolerance_miss\n";
                if (!allow_tolerance_miss)
                {
                    throw std::runtime_error(
                        "Qwen generation input is outside the calibrated "
                        "polynomial domain at step " +
                        std::to_string(step) + ", layer " +
                        std::to_string(layer));
                }
            }
            reference.layer_outputs.push_back(polynomial_hidden);
        }

        reference.final_norm_config =
            maximum_layers == model_config.num_hidden_layers
                ? qwen::he::qwen25_05b_final_inverse_sqrt_config()
                : partial_final_norm_config(
                      polynomial_hidden, model_config.rms_norm_epsilon);
        const qwen::Tensor exact_final = qwen::rms_norm(
            exact_hidden, weights.final_norm,
            model_config.rms_norm_epsilon);
        reference.polynomial_final = qwen::he::approximate_rms_norm_plain(
            polynomial_hidden, weights.final_norm,
            model_config.rms_norm_epsilon,
            reference.final_norm_config);
        const qwen::Tensor exact_logits = qwen::linear(
            last_token(exact_final), lm_head);
        const qwen::Tensor polynomial_logits = qwen::linear(
            last_token(reference.polynomial_final), lm_head);
        reference.exact_token = argmax_token(exact_logits);
        reference.polynomial_token = argmax_token(polynomial_logits);
        const double final_approximation_error = maximum_error(
            reference.polynomial_final, exact_final);
        const double final_tolerance =
            1.0e-2 + 5.0e-4 * tensor_max_abs(exact_final);
        std::cout << "qwen_gpu_generate_preflight step=" << step
                  << " final_polynomial_vs_exact_max_error="
                  << final_approximation_error
                  << " tolerance=" << final_tolerance
                  << " exact_token=" << reference.exact_token
                  << " polynomial_token=" << reference.polynomial_token
                  << '\n';
        if (final_approximation_error > final_tolerance)
        {
            reference_tolerance_miss = true;
            std::cout << "qwen_gpu_generate_warning step=" << step
                      << " type=final_polynomial_tolerance_miss\n";
            if (!allow_tolerance_miss)
            {
                throw std::runtime_error(
                    "Qwen final polynomial result exceeds its tolerance at "
                    "step " + std::to_string(step));
            }
        }
        if (reference.exact_token != reference.polynomial_token)
        {
            throw std::runtime_error(
                "Qwen polynomial token does not match exact greedy "
                "generation at step " + std::to_string(step));
        }
        position_offset += reference_ids.size();
        reference_ids = {reference.polynomial_token};
        references.push_back(std::move(reference));
      }

      std::cout << "qwen_gpu_preflight_exact_generated:";
      for (const auto &reference : references)
      {
        std::cout << ' ' << reference.exact_token;
      }
      std::cout << "\nqwen_gpu_preflight_polynomial_generated:";
      for (const auto &reference : references)
      {
        std::cout << ' ' << reference.polynomial_token;
      }
      std::cout << "\nqwen_gpu_preflight_tolerance_status="
              << (reference_tolerance_miss ? "WARN" : "PASS") << '\n';
      if (preflight_only)
      {
        std::cout << "[PASS] Qwen exact/polynomial greedy-token preflight"
                  << " prompt_tokens=" << prompt.size()
                  << " new_tokens=" << maximum_new_tokens
                  << " layers=" << maximum_layers << '\n';
        return;
      }
    }
    else
    {
        std::cout
            << "qwen_gpu_reference_preflight=SKIPPED"
            << " reason=production_gpu_generation"
            << " validation_status=UNVERIFIED\n";
    }

    const auto he_config =
        poseidon::benchmark::resnet50_gpu::make_resnet50_gpu_config();
    Runtime runtime(he_config, selected_device());
    runtime.initialize_inference_evaluation_keys();
    runtime.initialize_bootstrap();
    std::vector<poseidon::benchmark::qwen_gpu::GpuKVCache> gpu_caches(
        maximum_layers);
    std::vector<std::size_t> current_ids = prompt;
    std::vector<std::size_t> generated;
    generated.reserve(maximum_new_tokens);
    position_offset = 0;
    const auto gpu_start = std::chrono::steady_clock::now();
    for (std::size_t step = 0; step < maximum_new_tokens; ++step)
    {
        const auto step_start = std::chrono::steady_clock::now();
        auto encrypted = poseidon::benchmark::qwen_gpu::encrypt_tensor(
            embedding_rows(weights.token_embedding, current_ids), runtime);
        for (std::size_t layer = 0; layer < maximum_layers; ++layer)
        {
            const auto *polynomial_trace = validate_stages
                ? &references[step].layer_traces[layer] : nullptr;
            encrypted = poseidon::benchmark::qwen_gpu::decoder_layer(
                encrypted, weights.layers[layer], model_config,
                approximations[layer], layer, position_offset, runtime,
                validate_stages
                    ? poseidon::benchmark::qwen_gpu::DecoderTrace{
                          [&](const std::string &stage,
                              const poseidon::benchmark::qwen_gpu::
                                  GpuEncryptedTensor &tensor) {
                              const std::string reference_name =
                                  polynomial_trace_name(stage);
                              const auto reference =
                                  polynomial_trace->find(reference_name);
                              if (reference == polynomial_trace->end())
                              {
                                  std::cout
                                      << "qwen_gpu_generate_trace stage="
                                      << stage
                                      << " reference=missing\n";
                                  return;
                              }
                              const qwen::Tensor actual =
                                  poseidon::benchmark::qwen_gpu::
                                      decrypt_tensor(tensor, runtime);
                              std::cout
                                  << "qwen_gpu_generate_trace step=" << step
                                  << " layer=" << layer
                                  << " stage=" << stage
                                  << " max_error="
                                  << maximum_error(
                                         actual, reference->second)
                                  << " reference_max_abs="
                                  << tensor_max_abs(reference->second)
                                  << '\n';
                          }}
                    : poseidon::benchmark::qwen_gpu::DecoderTrace{},
                &gpu_caches[layer]);
            if (validate_stages)
            {
                const qwen::Tensor decrypted_layer =
                    poseidon::benchmark::qwen_gpu::decrypt_tensor(
                        encrypted, runtime);
                const qwen::Tensor &polynomial_layer =
                    references[step].layer_outputs[layer];
                const double layer_error = maximum_error(
                    decrypted_layer, polynomial_layer);
                const double layer_tolerance =
                    1.0e-2 + 2.0e-5 * tensor_max_abs(polynomial_layer);
                std::cout << "qwen_gpu_generate_validation step=" << step
                          << " layer=" << layer
                          << " cache_tokens=" << gpu_caches[layer].size()
                          << " ckks_vs_polynomial_max_error=" << layer_error
                          << " tolerance=" << layer_tolerance << '\n';
                if (layer_error > layer_tolerance)
                {
                    std::cout << "qwen_gpu_generate_warning step=" << step
                              << " layer=" << layer
                              << " type=ckks_tolerance_miss\n";
                    if (!allow_tolerance_miss)
                    {
                        throw std::runtime_error(
                            "GPU Qwen generation exceeded the CKKS layer "
                            "tolerance at step " + std::to_string(step) +
                            ", layer " + std::to_string(layer));
                    }
                }
            }
            if (gpu_layer_limit_text != nullptr &&
                layer + 1 == gpu_layer_limit)
            {
                std::cout
                    << "[PASS] GPU Qwen diagnostic layer limit reached"
                    << " step=" << step
                    << " completed_layers=" << gpu_layer_limit << '\n';
                return;
            }
        }

        const qwen::he::ApproximationConfig final_config =
            maximum_layers == model_config.num_hidden_layers
                ? qwen::he::qwen25_05b_final_inverse_sqrt_config()
                : (validate_stages
                       ? references[step].final_norm_config
                       : throw std::runtime_error(
                             "partial-layer GPU generation requires "
                             "POSEIDON_GPU_QWEN_VALIDATE=1"));
        auto encrypted_final = poseidon::benchmark::qwen_gpu::rms_norm(
            encrypted, weights.final_norm,
            model_config.rms_norm_epsilon,
            final_config.minimum, final_config.maximum,
            final_config.sample_count, runtime);
        encrypted_final = poseidon::benchmark::qwen_gpu::bootstrap(
            encrypted_final, runtime);
        const qwen::Tensor decrypted_final =
            poseidon::benchmark::qwen_gpu::decrypt_tensor(
                encrypted_final, runtime);
        const qwen::Tensor gpu_logits = qwen::linear(
            last_token(decrypted_final), lm_head);
        const std::size_t gpu_token = argmax_token(gpu_logits);
        const auto step_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - step_start).count();
        std::cout << "qwen_gpu_generate_step step=" << step
                  << " input_tokens=" << current_ids.size()
                  << " position_offset=" << position_offset
                  << " gpu_token=" << gpu_token
                  << " duration_ms=" << step_ms
                  << " validation_status="
                  << (validate_stages ? "VALIDATED" : "UNVERIFIED")
                  << '\n';
        if (validate_stages)
        {
            const qwen::Tensor polynomial_logits = qwen::linear(
                last_token(references[step].polynomial_final), lm_head);
            const double final_error = maximum_error(
                decrypted_final, references[step].polynomial_final);
            const double final_tolerance = 1.0e-2 + 5.0e-4 *
                tensor_max_abs(references[step].polynomial_final);
            const double logits_error = maximum_error(
                gpu_logits, polynomial_logits);
            std::cout
                << "qwen_gpu_generate_reference step=" << step
                << " final_ckks_vs_polynomial_max_error=" << final_error
                << " final_tolerance=" << final_tolerance
                << " logits_ckks_vs_polynomial_max_error=" << logits_error
                << " exact_token=" << references[step].exact_token
                << " polynomial_token="
                << references[step].polynomial_token
                << " gpu_token=" << gpu_token << '\n';
            const bool gpu_numerical_miss =
                final_error > final_tolerance || logits_error > 1.0e-1;
            if ((gpu_numerical_miss && !allow_tolerance_miss) ||
                gpu_token != references[step].exact_token ||
                gpu_token != references[step].polynomial_token)
            {
                throw std::runtime_error(
                    "GPU Qwen greedy token does not match the exact and "
                    "polynomial references at step " +
                    std::to_string(step));
            }
        }
        generated.push_back(gpu_token);
        position_offset += current_ids.size();
        current_ids = {gpu_token};
    }
    const auto gpu_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - gpu_start).count();
    std::cout << "qwen_gpu_generated:";
    for (std::size_t token : generated)
    {
        std::cout << ' ' << token;
    }
    std::cout << "\nqwen_gpu_generation_elapsed_ms=" << gpu_ms
              << (validate_stages
                      ? "\n[PASS] GPU Qwen validated encrypted greedy generation"
                      : "\n[DONE] GPU Qwen unverified encrypted greedy generation")
              << " prompt_tokens="
              << prompt.size() << " new_tokens=" << maximum_new_tokens
              << " layers=" << maximum_layers
              << " validation_status="
              << (validate_stages ? "VALIDATED" : "UNVERIFIED") << '\n';
}

void usage(const char *program)
{
    std::cerr << "Usage: " << program
              << " --smoke | --bootstrap-scale-check"
              << " | --model-info [MODEL_DIRECTORY]"
              << " | --topology-check [MODEL_DIRECTORY]"
              << " | --infer-token TOKEN_ID [MAX_LAYERS] [MODEL_DIRECTORY]"
              << " | --infer-ids ID,ID,... [MAX_LAYERS] [MODEL_DIRECTORY]"
              << " | --generate-ids ID,ID,... MAX_NEW_TOKENS"
              << " [MAX_LAYERS] [MODEL_DIRECTORY]\n";
}

}  // namespace

int main(int argc, char **argv)
{
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    const auto start = std::chrono::steady_clock::now();
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--smoke")
        {
            run_smoke();
        }
        else if (argc == 2 &&
                 std::string(argv[1]) == "--bootstrap-scale-check")
        {
            run_bootstrap_scale_check();
        }
        else if (argc >= 2 && std::string(argv[1]) == "--model-info")
        {
            const std::filesystem::path model = argc >= 3
                ? std::filesystem::path(argv[2])
                : std::filesystem::path(POSEIDON_GPU_QWEN_DEFAULT_MODEL_DIR);
            print_model_info(model);
        }
        else if (argc >= 2 && std::string(argv[1]) == "--topology-check")
        {
            const std::filesystem::path model = argc >= 3
                ? std::filesystem::path(argv[2])
                : std::filesystem::path(POSEIDON_GPU_QWEN_DEFAULT_MODEL_DIR);
            check_topology(model);
        }
        else if (argc >= 3 && std::string(argv[1]) == "--infer-token")
        {
            const std::size_t token_id = std::stoull(argv[2]);
            const std::size_t maximum_layers = argc >= 4
                ? std::stoull(argv[3]) : 24;
            const std::filesystem::path model = argc >= 5
                ? std::filesystem::path(argv[4])
                : std::filesystem::path(POSEIDON_GPU_QWEN_DEFAULT_MODEL_DIR);
            run_single_token_inference(model, token_id, maximum_layers);
        }
        else if (argc >= 3 && std::string(argv[1]) == "--infer-ids")
        {
            const std::vector<std::size_t> token_ids =
                parse_token_ids(argv[2]);
            const std::size_t maximum_layers = argc >= 4
                ? std::stoull(argv[3]) : 24;
            const std::filesystem::path model = argc >= 5
                ? std::filesystem::path(argv[4])
                : std::filesystem::path(POSEIDON_GPU_QWEN_DEFAULT_MODEL_DIR);
            run_multi_token_inference(
                model, token_ids, maximum_layers);
        }
        else if (argc >= 4 && std::string(argv[1]) == "--generate-ids")
        {
            const std::vector<std::size_t> prompt =
                parse_token_ids(argv[2]);
            const std::size_t maximum_new_tokens = std::stoull(argv[3]);
            const std::size_t maximum_layers = argc >= 5
                ? std::stoull(argv[4]) : 24;
            const std::filesystem::path model = argc >= 6
                ? std::filesystem::path(argv[5])
                : std::filesystem::path(POSEIDON_GPU_QWEN_DEFAULT_MODEL_DIR);
            run_generation(
                model, prompt, maximum_new_tokens, maximum_layers);
        }
        else
        {
            usage(argv[0]);
            return 2;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "qwen_gpu_total_elapsed_ms=" << elapsed
                  << " qwen_gpu_total_elapsed_seconds="
                  << static_cast<double>(elapsed) / 1000.0 << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "poseidon_gpu_qwen: " << error.what() << '\n';
        return 1;
    }
}
