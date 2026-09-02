#include "qwen_gpu_ops.h"

#include "poseidon/advance/util/chebyshev_interpolation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace poseidon::benchmark::qwen_gpu
{

namespace
{

constexpr std::size_t kBabyStep = 32;

void require_same_layout(
    const GpuEncryptedTensor &lhs, const GpuEncryptedTensor &rhs)
{
    const auto &a = lhs.layout();
    const auto &b = rhs.layout();
    if (a.tokens != b.tokens || a.features != b.features ||
        a.token_stride != b.token_stride || a.slot_count != b.slot_count)
    {
        throw std::invalid_argument(
            "GPU Qwen binary operation requires matching layouts");
    }
}

std::vector<double> pack_plain_chunk(
    const qwen::Tensor &plain, const GpuTensorLayout &layout,
    std::size_t token, std::size_t chunk)
{
    if (plain.rank() != 2 || plain.dim(0) != layout.tokens ||
        plain.dim(1) != layout.features)
    {
        throw std::invalid_argument(
            "GPU Qwen plaintext tensor does not match encrypted layout");
    }
    std::vector<double> slots(layout.slot_count, 0.0);
    const std::size_t begin = chunk * layout.token_stride;
    const std::size_t end = std::min(
        begin + layout.token_stride, layout.features);
    for (std::size_t feature = begin; feature < end; ++feature)
    {
        slots[feature - begin] = plain.at(token, feature);
    }
    return slots;
}

Runtime::DeviceCiphertext clone_cipher(
    const Runtime::DeviceCiphertext &cipher, const Runtime &runtime)
{
    return runtime.drop_to_q_count(cipher, cipher.meta.q_count);
}

Runtime::DeviceCiphertext make_periodic(
    const Runtime::DeviceCiphertext &input, std::size_t stride,
    const Runtime &runtime)
{
    auto output = clone_cipher(input, runtime);
    for (std::size_t distance = stride;
         distance < runtime.slot_count(); distance *= 2)
    {
        output = runtime.add(
            output, runtime.rotate_composed(output, distance));
    }
    return output;
}

Runtime::DeviceCiphertext reduce_block_sum(
    const Runtime::DeviceCiphertext &input, std::size_t stride,
    const Runtime &runtime)
{
    auto output = make_periodic(input, stride, runtime);
    for (std::size_t step = 1; step < stride; step *= 2)
    {
        output = runtime.add(
            output, runtime.rotate_composed(output, step));
    }
    return output;
}

std::vector<double> repeat_block(
    const std::vector<double> &block, std::size_t slot_count)
{
    if (block.empty() || slot_count % block.size() != 0)
    {
        throw std::invalid_argument("GPU Qwen block cannot fill CKKS slots");
    }
    std::vector<double> result(slot_count, 0.0);
    for (std::size_t offset = 0; offset < slot_count; offset += block.size())
    {
        std::copy(block.begin(), block.end(), result.begin() + offset);
    }
    return result;
}

double inverse_sqrt_function(double value)
{
    return 1.0 / std::sqrt(value);
}

double silu_function(double value)
{
    return value / (1.0 + std::exp(-value));
}

double sigmoid_function(double value)
{
    if (value >= 0.0)
    {
        const double exponential = std::exp(-value);
        return 1.0 / (1.0 + exponential);
    }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

double softplus_function(double value)
{
    return value >= 0.0
        ? value + std::log1p(std::exp(-value))
        : std::log1p(std::exp(value));
}

void validate_polynomial_vector(
    const std::vector<poseidon::Polynomial> &polynomials,
    const std::vector<std::vector<int>> &slot_indexes,
    std::size_t slot_count)
{
    if (polynomials.empty() || polynomials.size() != slot_indexes.size())
    {
        throw std::invalid_argument(
            "GPU Qwen polynomial vector is empty or inconsistent");
    }
    const std::size_t degree = polynomials.front().degree();
    if (degree == 0 || ((degree + 1) & degree) != 0)
    {
        throw std::invalid_argument(
            "GPU Qwen polynomial degree must be 2^k-1");
    }
    for (const auto &polynomial : polynomials)
    {
        if (polynomial.basis_type() != poseidon::Chebyshev ||
            polynomial.degree() != degree)
        {
            throw std::invalid_argument(
                "GPU Qwen polynomials must share Chebyshev degree");
        }
    }
    for (const auto &indexes : slot_indexes)
    {
        for (int slot : indexes)
        {
            if (slot < 0 || static_cast<std::size_t>(slot) >= slot_count)
            {
                throw std::out_of_range(
                    "GPU Qwen polynomial slot is out of range");
            }
        }
    }
}

Runtime::DeviceCiphertext evaluate_polynomial_leaf(
    const Runtime::DeviceCiphertext &input,
    const std::vector<poseidon::Polynomial> &polynomials,
    const std::vector<std::vector<int>> &slot_indexes,
    const Runtime &runtime)
{
    std::vector<double> linear(runtime.slot_count(), 0.0);
    std::vector<double> constant(runtime.slot_count(), 0.0);
    for (std::size_t index = 0; index < polynomials.size(); ++index)
    {
        const auto &coefficients = polynomials[index].data();
        const double c0 = coefficients.empty() ? 0.0 : coefficients[0].real();
        const double c1 = coefficients.size() < 2 ? 0.0 : coefficients[1].real();
        for (int slot : slot_indexes[index])
        {
            linear[static_cast<std::size_t>(slot)] = c1;
            constant[static_cast<std::size_t>(slot)] = c0;
        }
    }
    auto result = runtime.multiply_plain_rescale(input, linear);
    return runtime.add_plain(result, constant);
}

Runtime::DeviceCiphertext evaluate_polynomial_node(
    const Runtime::DeviceCiphertext &input,
    const std::vector<poseidon::Polynomial> &polynomials,
    const std::vector<std::vector<int>> &slot_indexes,
    const std::vector<std::unique_ptr<Runtime::DeviceCiphertext>> &basis,
    const Runtime &runtime)
{
    const std::size_t degree = polynomials.front().degree();
    if (degree <= 1)
    {
        return evaluate_polynomial_leaf(
            input, polynomials, slot_indexes, runtime);
    }
    const int split = static_cast<int>((degree + 1) / 2);
    std::vector<poseidon::Polynomial> left;
    std::vector<poseidon::Polynomial> right;
    left.reserve(polynomials.size());
    right.reserve(polynomials.size());
    for (const auto &polynomial : polynomials)
    {
        auto [quotient, remainder] = poseidon::split_coeffs(polynomial, split);
        left.push_back(std::move(remainder));
        right.push_back(std::move(quotient));
    }
    auto right_value = evaluate_polynomial_node(
        input, right, slot_indexes, basis, runtime);
    auto product = runtime.multiply_relinearize_rescale(
        *basis.at(static_cast<std::size_t>(split)), right_value);
    auto left_value = evaluate_polynomial_node(
        input, left, slot_indexes, basis, runtime);
    return runtime.add_aligned(product, left_value);
}

Runtime::DeviceCiphertext evaluate_chebyshev_cipher(
    const Runtime::DeviceCiphertext &input,
    const std::vector<poseidon::Polynomial> &polynomials,
    const std::vector<std::vector<int>> &slot_indexes,
    const Runtime &runtime)
{
    validate_polynomial_vector(polynomials, slot_indexes, runtime.slot_count());
    const std::size_t degree = polynomials.front().degree();
    std::vector<std::unique_ptr<Runtime::DeviceCiphertext>> basis(degree + 1);
    basis[1] = std::make_unique<Runtime::DeviceCiphertext>(
        clone_cipher(input, runtime));
    for (std::size_t current = 2; current <= degree; current *= 2)
    {
        auto squared = runtime.square_relinearize_rescale(
            *basis[current / 2]);
        auto doubled = runtime.add(squared, squared);
        basis[current] = std::make_unique<Runtime::DeviceCiphertext>(
            runtime.add_plain(doubled, std::vector<double>(
                runtime.slot_count(), -1.0)));
    }
    return evaluate_polynomial_node(
        input, polynomials, slot_indexes, basis, runtime);
}

}  // namespace

GpuEncryptedTensor add(
    const GpuEncryptedTensor &lhs, const GpuEncryptedTensor &rhs,
    const Runtime &runtime)
{
    require_same_layout(lhs, rhs);
    std::vector<GpuEncryptedTensor::Ciphertext> output;
    output.reserve(lhs.ciphertexts().size());
    for (std::size_t index = 0; index < lhs.ciphertexts().size(); ++index)
    {
        output.push_back(runtime.add_aligned(
            lhs.ciphertexts()[index], rhs.ciphertexts()[index]));
    }
    return GpuEncryptedTensor(lhs.layout(), std::move(output));
}

GpuEncryptedTensor subtract(
    const GpuEncryptedTensor &lhs, const GpuEncryptedTensor &rhs,
    const Runtime &runtime)
{
    require_same_layout(lhs, rhs);
    std::vector<GpuEncryptedTensor::Ciphertext> output;
    output.reserve(lhs.ciphertexts().size());
    for (std::size_t index = 0; index < lhs.ciphertexts().size(); ++index)
    {
        output.push_back(runtime.sub_aligned(
            lhs.ciphertexts()[index], rhs.ciphertexts()[index]));
    }
    return GpuEncryptedTensor(lhs.layout(), std::move(output));
}

GpuEncryptedTensor add_plain(
    const GpuEncryptedTensor &input, const qwen::Tensor &plain,
    const Runtime &runtime)
{
    std::vector<GpuEncryptedTensor::Ciphertext> output;
    output.reserve(input.ciphertexts().size());
    for (std::size_t token = 0; token < input.layout().tokens; ++token)
    {
        for (std::size_t chunk = 0;
             chunk < input.layout().feature_chunks(); ++chunk)
        {
            output.push_back(runtime.add_plain(
                input.cipher(token, chunk),
                pack_plain_chunk(plain, input.layout(), token, chunk)));
        }
    }
    return GpuEncryptedTensor(input.layout(), std::move(output));
}

GpuEncryptedTensor multiply_plain(
    const GpuEncryptedTensor &input, const qwen::Tensor &plain,
    const Runtime &runtime)
{
    std::vector<GpuEncryptedTensor::Ciphertext> output;
    output.reserve(input.ciphertexts().size());
    for (std::size_t token = 0; token < input.layout().tokens; ++token)
    {
        for (std::size_t chunk = 0;
             chunk < input.layout().feature_chunks(); ++chunk)
        {
            output.push_back(runtime.multiply_plain_rescale(
                input.cipher(token, chunk),
                pack_plain_chunk(plain, input.layout(), token, chunk)));
        }
    }
    return GpuEncryptedTensor(input.layout(), std::move(output));
}

GpuEncryptedTensor multiply(
    const GpuEncryptedTensor &lhs, const GpuEncryptedTensor &rhs,
    const Runtime &runtime)
{
    require_same_layout(lhs, rhs);
    std::vector<GpuEncryptedTensor::Ciphertext> output;
    output.reserve(lhs.ciphertexts().size());
    for (std::size_t index = 0; index < lhs.ciphertexts().size(); ++index)
    {
        output.push_back(runtime.multiply_relinearize_rescale(
            lhs.ciphertexts()[index], rhs.ciphertexts()[index]));
    }
    return GpuEncryptedTensor(lhs.layout(), std::move(output));
}

GpuEncryptedTensor bootstrap(
    const GpuEncryptedTensor &input, const Runtime &runtime,
    double value_scale)
{
    if (!std::isfinite(value_scale) || value_scale <= 0.0)
    {
        throw std::invalid_argument(
            "GPU Qwen bootstrap scale must be finite and positive");
    }
    std::vector<GpuEncryptedTensor::Ciphertext> output;
    output.reserve(input.ciphertexts().size());
    for (const auto &cipher : input.ciphertexts())
    {
        Runtime::DeviceCiphertext prepared = clone_cipher(cipher, runtime);
        if (value_scale != 1.0)
        {
            prepared = runtime.multiply_plain_scalar_rescale(
                prepared, 1.0 / value_scale);
        }
        auto refreshed = runtime.bootstrap(prepared);
        if (value_scale != 1.0)
        {
            refreshed = runtime.multiply_plain(
                refreshed,
                std::vector<double>(runtime.slot_count(), value_scale),
                1.0);
        }
        output.push_back(std::move(refreshed));
    }
    return GpuEncryptedTensor(input.layout(), std::move(output));
}

GpuEncryptedTensor linear(
    const GpuEncryptedTensor &input, const qwen::Tensor &weight,
    const qwen::Tensor *bias, const Runtime &runtime)
{
    if (weight.rank() != 2 || weight.dim(1) != input.layout().features)
    {
        throw std::invalid_argument("GPU Qwen Linear weight shape is invalid");
    }
    if (bias != nullptr &&
        (bias->rank() != 1 || bias->dim(0) != weight.dim(0)))
    {
        throw std::invalid_argument("GPU Qwen Linear bias shape is invalid");
    }
    const std::size_t stride = input.layout().token_stride;
    if (stride % kBabyStep != 0)
    {
        throw std::invalid_argument(
            "GPU Qwen Linear stride must be divisible by baby-step width");
    }
    const std::size_t output_features = weight.dim(0);
    const std::size_t output_chunks =
        (output_features + stride - 1) / stride;
    const std::size_t token_count = input.layout().tokens;
    const bool profile =
        std::getenv("POSEIDON_GPU_QWEN_PROFILE_LINEAR") != nullptr;
    using Clock = std::chrono::steady_clock;
    const auto linear_start = Clock::now();
    std::chrono::nanoseconds preparation_time{0};
    std::chrono::nanoseconds multiply_time{0};
    std::chrono::nanoseconds rotation_time{0};
    std::chrono::nanoseconds addition_time{0};
    std::size_t prepared_plaintexts = 0;
    std::size_t plaintext_products = 0;

    // Keep every token accumulator live, but stream only one 32-diagonal BSGS
    // group of encoded plaintexts at a time. A full Q52 plaintext is large;
    // this bounded plan reuses each host encoding/upload across all tokens
    // without attempting to retain an entire dense matrix on the device.
    std::vector<std::unique_ptr<Runtime::DeviceCiphertext>> accumulated(
        token_count * output_chunks);
    const auto accumulator_index = [output_chunks](
        std::size_t token, std::size_t output_chunk) {
        return token * output_chunks + output_chunk;
    };

    for (std::size_t input_chunk = 0;
         input_chunk < input.layout().feature_chunks(); ++input_chunk)
    {
        std::vector<std::vector<std::unique_ptr<Runtime::DeviceCiphertext>>>
            token_babies(token_count);
        for (std::size_t token = 0; token < token_count; ++token)
        {
            auto periodic = make_periodic(
                input.cipher(token, input_chunk), stride, runtime);
            auto &babies = token_babies[token];
            babies.resize(kBabyStep);
            babies[0] = std::make_unique<Runtime::DeviceCiphertext>(
                clone_cipher(periodic, runtime));
            for (std::size_t baby = 1; baby < kBabyStep; ++baby)
            {
                const auto start = Clock::now();
                babies[baby] = std::make_unique<Runtime::DeviceCiphertext>(
                    runtime.rotate_composed(periodic, baby));
                rotation_time += Clock::now() - start;
            }
        }

        const auto &encoding_source = *token_babies.front().front();
        const double plain_scale =
            runtime.last_modulus_value(encoding_source);
        for (std::size_t token = 1; token < token_count; ++token)
        {
            if (token_babies[token].front()->meta.parms_id !=
                encoding_source.meta.parms_id)
            {
                throw std::invalid_argument(
                    "GPU Qwen Linear token levels do not match");
            }
        }

        for (std::size_t output_chunk = 0;
             output_chunk < output_chunks; ++output_chunk)
        {
            for (std::size_t giant = 0; giant < stride;
                 giant += kBabyStep)
            {
                struct PreparedDiagonal
                {
                    std::size_t baby;
                    Runtime::DevicePlaintext plaintext;
                };
                std::vector<std::size_t> prepared_babies;
                std::vector<std::vector<double>> prepared_values;
                prepared_babies.reserve(kBabyStep);
                prepared_values.reserve(kBabyStep);
                for (std::size_t baby = 0; baby < kBabyStep; ++baby)
                {
                    const std::size_t diagonal = giant + baby;
                    std::vector<double> diagonal_values(
                        runtime.slot_count(), 0.0);
                    bool nonzero = false;
                    const std::size_t output_begin = output_chunk * stride;
                    const std::size_t input_begin = input_chunk * stride;
                    for (std::size_t output_local = 0;
                         output_local < stride; ++output_local)
                    {
                        const std::size_t output_index =
                            output_begin + output_local;
                        const std::size_t input_local =
                            (output_local + diagonal) % stride;
                        const std::size_t input_index =
                            input_begin + input_local;
                        if (output_index >= output_features ||
                            input_index >= input.layout().features)
                        {
                            continue;
                        }
                        const double value =
                            weight.at(output_index, input_index);
                        diagonal_values[output_local] = value;
                        nonzero = nonzero || value != 0.0;
                    }
                    if (!nonzero)
                    {
                        continue;
                    }
                    if (giant != 0)
                    {
                        std::rotate(
                            diagonal_values.begin(),
                            diagonal_values.end() -
                                static_cast<std::ptrdiff_t>(giant),
                            diagonal_values.end());
                    }
                    prepared_babies.push_back(baby);
                    prepared_values.push_back(std::move(diagonal_values));
                }
                if (prepared_values.empty())
                {
                    continue;
                }
                const auto preparation_start = Clock::now();
                auto device_plaintexts =
                    runtime.encode_and_upload_plain_batch(
                        encoding_source, prepared_values, plain_scale);
                preparation_time += Clock::now() - preparation_start;
                std::vector<PreparedDiagonal> prepared;
                prepared.reserve(device_plaintexts.size());
                for (std::size_t index = 0;
                     index < device_plaintexts.size(); ++index)
                {
                    prepared.push_back({
                        prepared_babies[index],
                        std::move(device_plaintexts[index])});
                }
                prepared_plaintexts += prepared.size();

                for (std::size_t token = 0; token < token_count; ++token)
                {
                    std::unique_ptr<Runtime::DeviceCiphertext> partial;
                    for (const auto &entry : prepared)
                    {
                        const auto multiply_start = Clock::now();
                        auto term = runtime.multiply_plain_preencoded(
                            *token_babies[token][entry.baby],
                            entry.plaintext);
                        multiply_time += Clock::now() - multiply_start;
                        ++plaintext_products;
                        if (!partial)
                        {
                            partial =
                                std::make_unique<Runtime::DeviceCiphertext>(
                                    std::move(term));
                        }
                        else
                        {
                            const auto add_start = Clock::now();
                            *partial = runtime.add(*partial, term);
                            addition_time += Clock::now() - add_start;
                        }
                    }
                    if (giant != 0)
                    {
                        const auto start = Clock::now();
                        *partial = runtime.rotate_composed(*partial, giant);
                        rotation_time += Clock::now() - start;
                    }
                    auto &destination = accumulated[
                        accumulator_index(token, output_chunk)];
                    if (!destination)
                    {
                        destination =
                            std::make_unique<Runtime::DeviceCiphertext>(
                                std::move(*partial));
                    }
                    else
                    {
                        const auto start = Clock::now();
                        *destination = runtime.add_aligned(
                            *destination, *partial);
                        addition_time += Clock::now() - start;
                    }
                }
            }
        }
    }

    std::vector<GpuEncryptedTensor::Ciphertext> outputs;
    outputs.reserve(token_count * output_chunks);
    for (std::size_t token = 0; token < token_count; ++token)
    {
        for (std::size_t output_chunk = 0;
             output_chunk < output_chunks; ++output_chunk)
        {
            auto &result = accumulated[
                accumulator_index(token, output_chunk)];
            if (!result)
            {
                throw std::logic_error(
                    "GPU Qwen Linear produced an empty output chunk");
            }
            *result = runtime.rescale(*result, 1);
            if (bias != nullptr)
            {
                std::vector<double> values(runtime.slot_count(), 0.0);
                const std::size_t begin = output_chunk * stride;
                const std::size_t end = std::min(
                    begin + stride, output_features);
                for (std::size_t index = begin; index < end; ++index)
                {
                    values[index - begin] = bias->at(index);
                }
                *result = runtime.add_plain(*result, values);
            }
            outputs.push_back(std::move(*result));
        }
    }
    if (profile)
    {
        const auto to_ms = [](std::chrono::nanoseconds value) {
            return std::chrono::duration<double, std::milli>(value).count();
        };
        const auto total = std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - linear_start);
        std::cout << "qwen_gpu_linear_profile"
                  << " tokens=" << token_count
                  << " input_features=" << input.layout().features
                  << " output_features=" << output_features
                  << " prepared_plaintexts=" << prepared_plaintexts
                  << " plaintext_products=" << plaintext_products
                  << " reuse_factor="
                  << (prepared_plaintexts == 0
                          ? 0.0
                          : static_cast<double>(plaintext_products) /
                                static_cast<double>(prepared_plaintexts))
                  << " preparation_ms=" << to_ms(preparation_time)
                  << " multiply_ms=" << to_ms(multiply_time)
                  << " rotation_ms=" << to_ms(rotation_time)
                  << " addition_ms=" << to_ms(addition_time)
                  << " total_ms=" << to_ms(total) << '\n';
    }
    return GpuEncryptedTensor(
        {input.layout().tokens, output_features, stride,
         input.layout().slot_count},
        std::move(outputs));
}

GpuEncryptedTensor rope(
    const GpuEncryptedTensor &input, std::size_t head_count,
    std::size_t head_dim, std::size_t position_offset, double theta,
    const Runtime &runtime)
{
    if (head_dim == 0 || head_dim % 2 != 0 || theta <= 0.0 ||
        input.layout().features != head_count * head_dim ||
        input.layout().feature_chunks() != 1)
    {
        throw std::invalid_argument("invalid GPU Qwen RoPE input");
    }
    const std::size_t half = head_dim / 2;
    std::vector<GpuEncryptedTensor::Ciphertext> outputs;
    outputs.reserve(input.layout().tokens);
    for (std::size_t token = 0; token < input.layout().tokens; ++token)
    {
        const auto &source = input.cipher(token, 0);
        std::vector<double> cosine(runtime.slot_count(), 0.0);
        std::vector<double> first_sine(runtime.slot_count(), 0.0);
        std::vector<double> second_sine(runtime.slot_count(), 0.0);
        for (std::size_t head = 0; head < head_count; ++head)
        {
            const std::size_t head_offset = head * head_dim;
            for (std::size_t index = 0; index < half; ++index)
            {
                const double frequency = std::pow(
                    theta, -2.0 * static_cast<double>(index) /
                               static_cast<double>(head_dim));
                const double angle = static_cast<double>(
                    position_offset + token) * frequency;
                const double cos_value = std::cos(angle);
                const double sin_value = std::sin(angle);
                const std::size_t first = head_offset + index;
                const std::size_t second = first + half;
                cosine[first] = cos_value;
                cosine[second] = cos_value;
                first_sine[first] = -sin_value;
                second_sine[second] = sin_value;
            }
        }
        const double plain_scale = runtime.last_modulus_value(source);
        auto result = runtime.multiply_plain(source, cosine, plain_scale);
        const auto rotated_left = runtime.rotate_composed(source, half);
        const auto rotated_right = runtime.rotate_composed(
            source, -static_cast<long long>(half));
        result = runtime.add(
            result, runtime.multiply_plain(
                rotated_left, first_sine, plain_scale));
        result = runtime.add(
            result, runtime.multiply_plain(
                rotated_right, second_sine, plain_scale));
        outputs.push_back(runtime.rescale(result, 1));
    }
    return GpuEncryptedTensor(input.layout(), std::move(outputs));
}

GpuEncryptedTensor token_view(
    const GpuEncryptedTensor &input, std::size_t token,
    const Runtime &runtime)
{
    if (token >= input.layout().tokens)
    {
        throw std::out_of_range("GPU Qwen token view is out of range");
    }
    std::vector<GpuEncryptedTensor::Ciphertext> ciphertexts;
    ciphertexts.reserve(input.layout().feature_chunks());
    for (std::size_t chunk = 0;
         chunk < input.layout().feature_chunks(); ++chunk)
    {
        ciphertexts.push_back(clone_cipher(
            input.cipher(token, chunk), runtime));
    }
    return GpuEncryptedTensor(
        {1, input.layout().features, input.layout().token_stride,
         input.layout().slot_count},
        std::move(ciphertexts));
}

GpuEncryptedTensor concatenate_tokens(
    std::vector<GpuEncryptedTensor> tokens, const Runtime &runtime)
{
    if (tokens.empty())
    {
        throw std::invalid_argument(
            "cannot concatenate an empty GPU Qwen token list");
    }
    const auto first = tokens.front().layout();
    std::size_t target_q_count = std::numeric_limits<std::size_t>::max();
    std::size_t total_tokens = 0;
    for (const auto &token : tokens)
    {
        if (token.layout().features != first.features ||
            token.layout().token_stride != first.token_stride ||
            token.layout().slot_count != first.slot_count)
        {
            throw std::invalid_argument(
                "GPU Qwen concatenated token layouts do not match");
        }
        total_tokens += token.layout().tokens;
        for (const auto &cipher : token.ciphertexts())
        {
            target_q_count = std::min(
                target_q_count, cipher.meta.q_count);
        }
    }
    std::vector<GpuEncryptedTensor::Ciphertext> ciphertexts;
    ciphertexts.reserve(
        total_tokens * first.feature_chunks());
    for (const auto &token : tokens)
    {
        for (const auto &cipher : token.ciphertexts())
        {
            ciphertexts.push_back(runtime.drop_to_q_count(
                cipher, target_q_count));
        }
    }
    return GpuEncryptedTensor(
        {total_tokens, first.features, first.token_stride,
         first.slot_count},
        std::move(ciphertexts));
}

GpuEncryptedTensor evaluate_chebyshev(
    const GpuEncryptedTensor &input,
    const poseidon::Polynomial &polynomial,
    const Runtime &runtime)
{
    std::vector<int> slots(runtime.slot_count());
    std::iota(slots.begin(), slots.end(), 0);
    return evaluate_chebyshev(input, {polynomial}, {std::move(slots)}, runtime);
}

GpuEncryptedTensor evaluate_chebyshev(
    const GpuEncryptedTensor &input,
    const std::vector<poseidon::Polynomial> &polynomials,
    const std::vector<std::vector<int>> &slot_indexes,
    const Runtime &runtime)
{
    std::vector<GpuEncryptedTensor::Ciphertext> output;
    output.reserve(input.ciphertexts().size());
    for (const auto &cipher : input.ciphertexts())
    {
        output.push_back(evaluate_chebyshev_cipher(
            cipher, polynomials, slot_indexes, runtime));
    }
    return GpuEncryptedTensor(input.layout(), std::move(output));
}

GpuEncryptedTensor affine_to_chebyshev_domain(
    const GpuEncryptedTensor &input,
    const std::vector<double> &minimum_by_feature,
    const std::vector<double> &maximum_by_feature,
    const Runtime &runtime)
{
    if (minimum_by_feature.size() != input.layout().features ||
        maximum_by_feature.size() != input.layout().features)
    {
        throw std::invalid_argument(
            "GPU Qwen approximation ranges do not match tensor width");
    }
    qwen::Tensor multipliers(
        {input.layout().tokens, input.layout().features});
    qwen::Tensor offsets(
        {input.layout().tokens, input.layout().features});
    for (std::size_t token = 0; token < input.layout().tokens; ++token)
    {
        for (std::size_t feature = 0;
             feature < input.layout().features; ++feature)
        {
            const double minimum = minimum_by_feature[feature];
            const double maximum = maximum_by_feature[feature];
            if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
                minimum >= maximum)
            {
                throw std::invalid_argument(
                    "GPU Qwen approximation interval is invalid");
            }
            multipliers.at(token, feature) = 2.0 / (maximum - minimum);
            offsets.at(token, feature) =
                -(maximum + minimum) / (maximum - minimum);
        }
    }
    return add_plain(multiply_plain(input, multipliers, runtime), offsets, runtime);
}

GpuEncryptedTensor rms_norm(
    const GpuEncryptedTensor &input, const qwen::Tensor &weight,
    double epsilon, double minimum_variance, double maximum_variance,
    int polynomial_samples, const Runtime &runtime)
{
    if (weight.rank() != 1 || weight.dim(0) != input.layout().features ||
        input.layout().features > input.layout().token_stride ||
        epsilon <= 0.0 || minimum_variance <= 0.0 ||
        minimum_variance >= maximum_variance || polynomial_samples < 2)
    {
        throw std::invalid_argument("GPU Qwen RMSNorm parameters are invalid");
    }
    auto squared = multiply(input, input, runtime);
    std::vector<GpuEncryptedTensor::Ciphertext> variance_ciphers;
    variance_ciphers.reserve(input.layout().tokens);
    for (std::size_t token = 0; token < input.layout().tokens; ++token)
    {
        auto sum = reduce_block_sum(
            squared.cipher(token, 0), input.layout().token_stride, runtime);
        std::vector<double> scale(runtime.slot_count(),
            1.0 / static_cast<double>(input.layout().features));
        auto variance = runtime.multiply_plain_rescale(sum, scale);
        variance = runtime.add_plain_scalar(variance, epsilon);
        variance_ciphers.push_back(std::move(variance));
    }
    GpuEncryptedTensor variance(
        {input.layout().tokens, input.layout().features,
         input.layout().token_stride, input.layout().slot_count},
        std::move(variance_ciphers));
    std::vector<double> minimum(input.layout().features, minimum_variance);
    std::vector<double> maximum(input.layout().features, maximum_variance);
    auto normalized = affine_to_chebyshev_domain(
        variance, minimum, maximum, runtime);
    auto polynomial = poseidon::util::approximate(
        inverse_sqrt_function, minimum_variance, maximum_variance,
        polynomial_samples);
    polynomial.lead() = true;
    auto inverse = evaluate_chebyshev(normalized, polynomial, runtime);
    auto output = multiply(input, inverse, runtime);
    qwen::Tensor weights(
        {input.layout().tokens, input.layout().features});
    for (std::size_t token = 0; token < input.layout().tokens; ++token)
    {
        for (std::size_t feature = 0;
             feature < input.layout().features; ++feature)
        {
            weights.at(token, feature) = weight.at(feature);
        }
    }
    return multiply_plain(output, weights, runtime);
}

GpuEncryptedTensor silu(
    const GpuEncryptedTensor &input,
    const std::vector<double> &minimum_by_feature,
    const std::vector<double> &maximum_by_feature,
    int polynomial_samples, const Runtime &runtime)
{
    if (polynomial_samples < 2 ||
        minimum_by_feature.size() != input.layout().features ||
        maximum_by_feature.size() != input.layout().features)
    {
        throw std::invalid_argument("GPU Qwen SiLU parameters are invalid");
    }
    auto normalized = affine_to_chebyshev_domain(
        input, minimum_by_feature, maximum_by_feature, runtime);
    std::vector<GpuEncryptedTensor::Ciphertext> outputs;
    outputs.reserve(normalized.ciphertexts().size());
    for (std::size_t token = 0; token < input.layout().tokens; ++token)
    {
        for (std::size_t chunk = 0;
             chunk < input.layout().feature_chunks(); ++chunk)
        {
            std::map<std::pair<double, double>, std::size_t> groups;
            std::vector<poseidon::Polynomial> polynomials;
            std::vector<std::vector<int>> slot_indexes;
            const std::size_t begin = chunk * input.layout().token_stride;
            const std::size_t end = std::min(
                begin + input.layout().token_stride,
                input.layout().features);
            for (std::size_t feature = begin; feature < end; ++feature)
            {
                const auto key = std::make_pair(
                    minimum_by_feature[feature],
                    maximum_by_feature[feature]);
                auto found = groups.find(key);
                if (found == groups.end())
                {
                    const std::size_t index = polynomials.size();
                    auto polynomial = poseidon::util::approximate(
                        silu_function, key.first, key.second,
                        polynomial_samples);
                    polynomial.lead() = true;
                    polynomials.push_back(std::move(polynomial));
                    slot_indexes.emplace_back();
                    found = groups.emplace(key, index).first;
                }
                for (std::size_t block = 0;
                     block < runtime.slot_count() /
                         input.layout().token_stride;
                     ++block)
                {
                    slot_indexes[found->second].push_back(static_cast<int>(
                        block * input.layout().token_stride +
                        (feature - begin)));
                }
            }
            outputs.push_back(evaluate_chebyshev_cipher(
                normalized.cipher(token, chunk), polynomials,
                slot_indexes, runtime));
        }
    }
    return GpuEncryptedTensor(input.layout(), std::move(outputs));
}

GpuEncryptedTensor sigmoid(
    const GpuEncryptedTensor &input, double minimum, double maximum,
    int polynomial_samples, const Runtime &runtime)
{
    if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
        minimum >= maximum || polynomial_samples < 2)
    {
        throw std::invalid_argument("GPU Qwen Sigmoid interval is invalid");
    }
    std::vector<double> minimums(input.layout().features, minimum);
    std::vector<double> maximums(input.layout().features, maximum);
    auto normalized = affine_to_chebyshev_domain(
        input, minimums, maximums, runtime);
    auto polynomial = poseidon::util::approximate(
        sigmoid_function, minimum, maximum, polynomial_samples);
    polynomial.lead() = true;
    return evaluate_chebyshev(normalized, polynomial, runtime);
}

GpuEncryptedTensor softplus(
    const GpuEncryptedTensor &input, double minimum, double maximum,
    int polynomial_samples, const Runtime &runtime)
{
    if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
        minimum >= maximum || polynomial_samples < 2)
    {
        throw std::invalid_argument("GPU Qwen Softplus interval is invalid");
    }
    std::vector<double> minimums(input.layout().features, minimum);
    std::vector<double> maximums(input.layout().features, maximum);
    auto normalized = affine_to_chebyshev_domain(
        input, minimums, maximums, runtime);
    auto polynomial = poseidon::util::approximate(
        softplus_function, minimum, maximum, polynomial_samples);
    polynomial.lead() = true;
    return evaluate_chebyshev(normalized, polynomial, runtime);
}

}  // namespace poseidon::benchmark::qwen_gpu
