#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/encryptor.h"
#include "poseidon/evaluator/evaluator_ckks_base.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

#include <cuda_runtime_api.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int kSkip = 77;

int require_cuda_device()
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess)
    {
        std::cerr << "[SKIP] cudaGetDeviceCount failed: "
                  << cudaGetErrorString(status) << "\n";
        return kSkip;
    }
    if (device_count == 0)
    {
        std::cerr << "[SKIP] No CUDA device is visible\n";
        return kSkip;
    }
    return EXIT_SUCCESS;
}

std::size_t env_size_or(const char *name, std::size_t fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return fallback;
    }
    return static_cast<std::size_t>(std::stoull(value));
}

double env_double_or(const char *name, double fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return fallback;
    }
    return std::stod(value);
}

int log2_degree(std::size_t degree)
{
    if (degree < 2 || (degree & (degree - 1)) != 0)
    {
        throw std::invalid_argument("degree must be a power of two");
    }

    int result = 0;
    while (degree > 1)
    {
        degree >>= 1;
        ++result;
    }
    return result;
}

poseidon::ParametersLiteral make_test_parameters(
    std::size_t degree,
    std::size_t q_count,
    std::size_t p_count,
    std::uint32_t log_q,
    std::uint32_t log_p,
    std::uint32_t log_scale,
    std::uint32_t q0_level)
{
    const int log_n = log2_degree(degree);
    poseidon::ParametersLiteral parms(
        CKKS,
        log_n,
        log_n - 1,
        log_scale,
        /*hamming_weight=*/0,
        q0_level,
        poseidon::Modulus(0),
        std::vector<poseidon::Modulus>{},
        std::vector<poseidon::Modulus>{},
        poseidon::sec_level_type::none);

    parms.set_log_modulus(
        std::vector<std::uint32_t>(q_count, log_q),
        std::vector<std::uint32_t>(p_count, log_p));
    return parms;
}

class RmmPoolScope
{
public:
    explicit RmmPoolScope(int device_id)
        : device_id_(device_id), pool_(&upstream_, 1 << 20, std::nullopt)
    {
        const cudaError_t status = cudaSetDevice(device_id_);
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("cudaSetDevice failed: ") + cudaGetErrorString(status));
        }

        previous_ = rmm::mr::get_current_device_resource();
        rmm::mr::set_current_device_resource(&pool_);
    }

    RmmPoolScope(const RmmPoolScope &) = delete;
    RmmPoolScope &operator=(const RmmPoolScope &) = delete;

    ~RmmPoolScope()
    {
        try
        {
            cudaSetDevice(device_id_);
            rmm::mr::set_current_device_resource(previous_);
        }
        catch (...)
        {}
    }

private:
    int device_id_ = 0;
    rmm::mr::cuda_memory_resource upstream_;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_;
    rmm::mr::device_memory_resource *previous_ = nullptr;
};

std::vector<std::complex<double>> make_message(std::size_t slot_count)
{
    std::vector<std::complex<double>> message(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i)
    {
        const double real = static_cast<double>((i % 17) + 1) / 32.0;
        const double imag = static_cast<double>((i % 11) + 1) / 64.0;
        message[i] = {real, imag};
    }
    return message;
}

double q0_over_message_ratio(
    const poseidon::PoseidonContext &context,
    std::uint32_t bootstrap_ratio)
{
    double value = context.crt_context()->q0();
    value /= static_cast<double>(bootstrap_ratio);
    return std::exp2(std::round(std::log2(value)));
}

poseidon::Ciphertext cpu_bootstrap_prepare_and_raise(
    const poseidon::Ciphertext &source,
    const poseidon::EvaluatorCkksBase &evaluator,
    const poseidon::PoseidonContext &context,
    const poseidon::CKKSEncoder &encoder,
    double target_q0_over_message_ratio)
{
    poseidon::Ciphertext result = source;
    const auto q0_level = context.parameters_literal()->q0_level();
    const auto parms_id_map = context.crt_context()->parms_id_map();

    while (result.scale() > std::pow(2.0, 54.0))
    {
        evaluator.rescale(result, result);
    }

    const auto level = result.level();
    if (level < q0_level)
    {
        throw std::runtime_error("source ciphertext is below q0 level");
    }
    const auto level_diff = level - q0_level;
    if (level_diff > 1)
    {
        evaluator.drop_modulus(result, result, parms_id_map.at(q0_level + 1));
    }

    double scale_multiplier =
        std::round(target_q0_over_message_ratio / result.scale());
    if (scale_multiplier > 1.0)
    {
        if (scale_multiplier >
            static_cast<double>(std::numeric_limits<int>::max()))
        {
            throw std::runtime_error("CPU scale multiplier exceeds int range");
        }
        evaluator.multiply_const_direct(
            result,
            static_cast<int>(scale_multiplier),
            result,
            encoder);
        result.scale() *= scale_multiplier;
    }

    evaluator.drop_modulus(result, result, parms_id_map.at(q0_level));

    poseidon::Ciphertext raised;
    evaluator.raise_modulus(result, raised);
    return raised;
}

poseidon::LinearMatrixGroup make_coeff_to_slot_matrix_group(
    const poseidon::PoseidonContext &context,
    poseidon::CKKSEncoder &encoder,
    double scaling,
    std::uint32_t log_bsgs_ratio,
    std::uint32_t step)
{
    poseidon::HomomorphicDFTMatrixLiteral matrix_literal(
        poseidon::encode,
        context.parameters_literal()->log_n(),
        context.parameters_literal()->log_slots(),
        static_cast<std::uint32_t>(context.parameters_literal()->q().size() - 1),
        std::vector<std::uint32_t>(3, 1),
        /*repack_imag_to_real=*/true,
        scaling,
        /*bit_reversed=*/false,
        log_bsgs_ratio);

    poseidon::LinearMatrixGroup matrix_group;
    matrix_literal.create(matrix_group, encoder, step);
    return matrix_group;
}

poseidon::LinearMatrixGroup make_slot_to_coeff_matrix_group(
    const poseidon::PoseidonContext &context,
    poseidon::CKKSEncoder &encoder,
    std::uint32_t level_start,
    double scaling,
    std::uint32_t log_bsgs_ratio,
    std::uint32_t step)
{
    poseidon::HomomorphicDFTMatrixLiteral matrix_literal(
        poseidon::decode,
        context.parameters_literal()->log_n(),
        context.parameters_literal()->log_slots(),
        level_start,
        std::vector<std::uint32_t>(3, 1),
        /*repack_imag_to_real=*/true,
        scaling,
        /*bit_reversed=*/false,
        log_bsgs_ratio);

    poseidon::LinearMatrixGroup matrix_group;
    matrix_literal.create(matrix_group, encoder, step);
    return matrix_group;
}

poseidon::GaloisKeys make_galois_keys_for_matrix_group(
    const poseidon::PoseidonContext &context,
    poseidon::KeyGenerator &keygen,
    const poseidon::LinearMatrixGroup &matrix_group)
{
    std::set<int> steps(
        matrix_group.rot_index().begin(),
        matrix_group.rot_index().end());
    steps.insert(0);  // conjugation key used by coeff_to_slot after DFT

    const auto galois_tool = context.crt_context()->galois_tool();
    std::vector<std::uint32_t> galois_elts;
    galois_elts.reserve(steps.size());
    for (const int step : steps)
    {
        galois_elts.push_back(galois_tool->get_elt_from_step(step));
    }

    poseidon::GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_elts, galois_keys);
    return galois_keys;
}

poseidon::GaloisKeys make_galois_keys_for_matrix_groups(
    const poseidon::PoseidonContext &context,
    poseidon::KeyGenerator &keygen,
    const std::vector<const poseidon::LinearMatrixGroup *> &matrix_groups)
{
    std::set<int> steps;
    steps.insert(0);  // conjugation key used by coeff_to_slot after DFT
    for (const auto *matrix_group : matrix_groups)
    {
        if (matrix_group == nullptr)
        {
            continue;
        }
        steps.insert(
            matrix_group->rot_index().begin(),
            matrix_group->rot_index().end());
    }

    const auto galois_tool = context.crt_context()->galois_tool();
    std::vector<std::uint32_t> galois_elts;
    galois_elts.reserve(steps.size());
    for (const int step : steps)
    {
        galois_elts.push_back(galois_tool->get_elt_from_step(step));
    }

    poseidon::GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_elts, galois_keys);
    return galois_keys;
}

std::optional<std::uint32_t> find_context_level(
    const poseidon::PoseidonContext &context,
    const poseidon::parms_id_type &parms_id)
{
    const auto parms_id_map = context.crt_context()->parms_id_map();
    for (const auto &entry : parms_id_map)
    {
        if (entry.second == parms_id)
        {
            return entry.first;
        }
    }
    return std::nullopt;
}

std::string format_context_level(
    const poseidon::PoseidonContext &context,
    const poseidon::parms_id_type &parms_id)
{
    const auto level = find_context_level(context, parms_id);
    if (level.has_value())
    {
        return std::to_string(*level);
    }
    return "not-in-context-map";
}

void print_matrix_group_levels(
    const poseidon::PoseidonContext &context,
    const poseidon::LinearMatrixGroup &matrix_group,
    const std::string &label)
{
    for (std::size_t i = 0; i < matrix_group.data().size(); ++i)
    {
        const auto &matrix = matrix_group.data()[i];
        std::cout << label << " matrix[" << i << "] level=" << matrix.level;
        if (!matrix.plain_vec.empty())
        {
            const auto &plaintext = matrix.plain_vec.begin()->second;
            std::cout << " q_count=" << plaintext.coeff_count() / context.parameters_literal()->degree()
                      << " parms_level="
                      << format_context_level(context, plaintext.parms_id());
        }
        std::cout << "\n";
    }
}

void cpu_multiply_by_diag_matrix_bsgs_rescale(
    const poseidon::Ciphertext &ciphertext,
    const poseidon::MatrixPlain &matrix,
    poseidon::Ciphertext &result,
    const poseidon::EvaluatorCkksBase &evaluator,
    const poseidon::GaloisKeys &galois_keys)
{
    auto [index, unused_rot_n1, rot_n2] =
        poseidon::bsgs_index(
            matrix.plain_vec,
            1 << matrix.log_slots,
            static_cast<int>(matrix.n1));
    (void)unused_rot_n1;

    std::map<int, poseidon::Ciphertext> rotated_ciphertexts;
    poseidon::Ciphertext inner_sum;
    poseidon::Ciphertext inner_product;
    poseidon::Ciphertext accumulator;

    for (const int step : rot_n2)
    {
        if (step != 0)
        {
            evaluator.rotate(ciphertext, rotated_ciphertexts[step], step, galois_keys);
        }
    }

    int giant_count = 0;
    for (const auto &giant_entry : index)
    {
        int baby_count = 0;
        for (const int baby_step : giant_entry.second)
        {
            const int diagonal_index = giant_entry.first + baby_step;
            const auto &plaintext = matrix.plain_vec.at(diagonal_index);

            const poseidon::Ciphertext &rotated_source =
                baby_step == 0 ? ciphertext : rotated_ciphertexts.at(baby_step);

            poseidon::Ciphertext product;
            evaluator.multiply_plain(rotated_source, plaintext, product);

            if (giant_count == 0)
            {
                if (baby_count == 0)
                {
                    accumulator = std::move(product);
                }
                else
                {
                    poseidon::Ciphertext updated_accumulator;
                    evaluator.add(accumulator, product, updated_accumulator);
                    accumulator = std::move(updated_accumulator);
                }
            }
            else
            {
                if (baby_count == 0)
                {
                    inner_sum = std::move(product);
                }
                else
                {
                    poseidon::Ciphertext updated_inner_sum;
                    evaluator.add(inner_sum, product, updated_inner_sum);
                    inner_sum = std::move(updated_inner_sum);
                }
            }

            ++baby_count;
        }

        if (giant_count != 0)
        {
            evaluator.rotate(inner_sum, inner_product, giant_entry.first, galois_keys);
            poseidon::Ciphertext updated_accumulator;
            evaluator.add(accumulator, inner_product, updated_accumulator);
            accumulator = std::move(updated_accumulator);
        }
        ++giant_count;
    }

    evaluator.rescale(accumulator, result);
}

void cpu_dft_rescale(
    const poseidon::Ciphertext &ciphertext,
    const poseidon::LinearMatrixGroup &matrix_group,
    poseidon::Ciphertext &result,
    const poseidon::EvaluatorCkksBase &evaluator,
    const poseidon::GaloisKeys &galois_keys)
{
    if (matrix_group.data().empty())
    {
        throw std::invalid_argument("cpu_dft_rescale: empty matrix group");
    }

    cpu_multiply_by_diag_matrix_bsgs_rescale(
        ciphertext,
        matrix_group.data().front(),
        result,
        evaluator,
        galois_keys);

    for (std::size_t i = 1; i < matrix_group.data().size(); ++i)
    {
        poseidon::Ciphertext next;
        cpu_multiply_by_diag_matrix_bsgs_rescale(
            result,
            matrix_group.data()[i],
            next,
            evaluator,
            galois_keys);
        result = std::move(next);
    }
}

void cpu_coeff_to_slot_rescale(
    const poseidon::Ciphertext &ciphertext,
    const poseidon::LinearMatrixGroup &matrix_group,
    poseidon::Ciphertext &result_real,
    poseidon::Ciphertext &result_imag,
    const poseidon::EvaluatorCkksBase &evaluator,
    const poseidon::GaloisKeys &galois_keys,
    const poseidon::CKKSEncoder &encoder)
{
    poseidon::Ciphertext dft_result;
    cpu_dft_rescale(
        ciphertext,
        matrix_group,
        dft_result,
        evaluator,
        galois_keys);

    evaluator.conjugate(dft_result, galois_keys, result_imag);
    evaluator.add(dft_result, result_imag, result_real);
    evaluator.sub(dft_result, result_imag, result_imag);
    evaluator.multiply_const(
        result_imag,
        std::complex<double>(0.0, -1.0),
        1.0,
        result_imag,
        encoder);
}

void cpu_slot_to_coeff_rescale(
    const poseidon::Ciphertext &real,
    const poseidon::Ciphertext &imag,
    const poseidon::LinearMatrixGroup &matrix_group,
    poseidon::Ciphertext &result,
    const poseidon::EvaluatorCkksBase &evaluator,
    const poseidon::GaloisKeys &galois_keys,
    const poseidon::CKKSEncoder &encoder)
{
    poseidon::Ciphertext scaled_imag;
    evaluator.multiply_const(
        imag,
        std::complex<double>(0.0, 1.0),
        1.0,
        scaled_imag,
        encoder);

    poseidon::Ciphertext merged_slots;
    evaluator.add(scaled_imag, real, merged_slots);

    cpu_dft_rescale(
        merged_slots,
        matrix_group,
        result,
        evaluator,
        galois_keys);
}

void cpu_evalmod_identity_horner(
    const poseidon::Ciphertext &input,
    poseidon::Ciphertext &output,
    const poseidon::EvaluatorCkksBase &evaluator,
    const poseidon::RelinKeys &relin_keys,
    const poseidon::CKKSEncoder &encoder,
    double target_scale)
{
    poseidon::Ciphertext x;
    poseidon::Ciphertext accumulator;
    poseidon::Ciphertext with_one;
    poseidon::Ciphertext multiplied;
    poseidon::Ciphertext relined;
    poseidon::Plaintext one_plain;

    evaluator.multiply_const_direct(input, 1, x, encoder);
    x.scale() = target_scale;
    evaluator.multiply_const_direct(x, 0, accumulator, encoder);
    accumulator.scale() = target_scale;
    encoder.encode(
        std::complex<double>(1.0, 0.0),
        x.parms_id(),
        target_scale,
        one_plain);
    evaluator.add_plain(accumulator, one_plain, with_one);
    with_one.scale() = target_scale;
    evaluator.multiply(with_one, x, multiplied);
    evaluator.relinearize(multiplied, relined, relin_keys);
    evaluator.rescale(relined, output);
    output.scale() = target_scale;
}

struct RawComparison
{
    bool equal = false;
    std::size_t expected_words = 0;
    std::size_t actual_words = 0;
    std::size_t mismatch_count = 0;
};

struct TimingRow
{
    std::string operation;
    double cpu_avg_ms = 0.0;
    double gpu_avg_ms = 0.0;
    std::string correctness = "N/A";
};

RawComparison compare_ciphertexts(
    const poseidon::Ciphertext &expected,
    const poseidon::Ciphertext &actual,
    std::size_t max_printed_mismatches)
{
    RawComparison result;
    result.expected_words =
        expected.size() * expected.poly_modulus_degree() *
        expected.coeff_modulus_size();
    result.actual_words =
        actual.size() * actual.poly_modulus_degree() *
        actual.coeff_modulus_size();

    const std::size_t compared_words =
        std::min(result.expected_words, result.actual_words);
    for (std::size_t i = 0; i < compared_words; ++i)
    {
        if (expected.data()[i] != actual.data()[i])
        {
            if (result.mismatch_count < max_printed_mismatches)
            {
                std::cout << "mismatch[" << result.mismatch_count
                          << "] index=" << i
                          << " cpu=" << expected.data()[i]
                          << " gpu=" << actual.data()[i] << "\n";
            }
            ++result.mismatch_count;
        }
    }

    result.equal =
        result.mismatch_count == 0 &&
        result.expected_words == result.actual_words &&
        expected.parms_id() == actual.parms_id() &&
        expected.is_ntt_form() == actual.is_ntt_form() &&
        expected.size() == actual.size() &&
        expected.poly_modulus_degree() == actual.poly_modulus_degree() &&
        expected.coeff_modulus_size() == actual.coeff_modulus_size() &&
        expected.scale() == actual.scale();
    return result;
}

template <typename Func>
double time_cpu_ms(std::size_t iterations, Func &&func)
{
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i)
    {
        func();
    }
    const auto stop = std::chrono::steady_clock::now();
    const auto total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    return static_cast<double>(total_us) / 1000.0 / static_cast<double>(iterations);
}

template <typename Func>
double time_gpu_ms(std::size_t iterations, Func &&func)
{
    cudaDeviceSynchronize();
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i)
    {
        func();
    }
    cudaDeviceSynchronize();
    const auto stop = std::chrono::steady_clock::now();
    const auto total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    return static_cast<double>(total_us) / 1000.0 / static_cast<double>(iterations);
}

std::string format_ms(double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << value;
    return stream.str();
}

std::string format_speedup(double cpu_ms, double gpu_ms)
{
    if (!(gpu_ms > 0.0))
    {
        return "n/a";
    }

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << (cpu_ms / gpu_ms) << "x";
    return stream.str();
}

std::vector<std::size_t> required_dft_key_q_counts(
    std::size_t input_q_count,
    std::size_t matrix_count,
    bool include_post_dft_conjugation)
{
    if (matrix_count > input_q_count)
    {
        throw std::invalid_argument("DFT matrix count exceeds input q_count");
    }

    std::vector<std::size_t> q_counts;
    q_counts.reserve(matrix_count + (include_post_dft_conjugation ? 1 : 0));
    for (std::size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index)
    {
        q_counts.push_back(input_q_count - matrix_index);
    }
    if (include_post_dft_conjugation)
    {
        q_counts.push_back(input_q_count - matrix_count);
    }

    std::sort(q_counts.begin(), q_counts.end());
    q_counts.erase(
        std::unique(q_counts.begin(), q_counts.end()),
        q_counts.end());
    return q_counts;
}

std::vector<std::size_t> merge_q_counts(
    std::vector<std::size_t> lhs,
    const std::vector<std::size_t> &rhs)
{
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    std::sort(lhs.begin(), lhs.end());
    lhs.erase(std::unique(lhs.begin(), lhs.end()), lhs.end());
    return lhs;
}

std::string join_q_counts(const std::vector<std::size_t> &q_counts)
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < q_counts.size(); ++i)
    {
        if (i != 0)
        {
            stream << ",";
        }
        stream << q_counts[i];
    }
    return stream.str();
}

void print_bootstrap_timing_table(
    const std::vector<TimingRow> &rows,
    std::size_t iterations,
    std::size_t warmup,
    std::optional<std::size_t> full_smoke_iterations = std::nullopt,
    std::optional<std::size_t> full_smoke_warmup = std::nullopt)
{
    constexpr int op_width = 42;
    constexpr int cpu_width = 14;
    constexpr int gpu_width = 14;
    constexpr int speedup_width = 12;
    constexpr int correctness_width = 11;

    auto print_separator = [&]() {
        std::cout << "|-" << std::string(op_width, '-')
                  << "-|-" << std::string(cpu_width, '-')
                  << "-|-" << std::string(gpu_width, '-')
                  << "-|-" << std::string(speedup_width, '-')
                  << "-|-" << std::string(correctness_width, '-')
                  << "-|\n";
    };

    auto print_row = [&](
        const std::string &operation,
        const std::string &cpu_ms,
        const std::string &gpu_ms,
        const std::string &speedup,
        const std::string &correctness) {
        std::cout << "| " << std::left << std::setw(op_width) << operation
                  << " | " << std::right << std::setw(cpu_width) << cpu_ms
                  << " | " << std::right << std::setw(gpu_width) << gpu_ms
                  << " | " << std::right << std::setw(speedup_width) << speedup
                  << " | " << std::right << std::setw(correctness_width) << correctness
                  << " |\n";
    };

    std::cout << "\n[bootstrap timing summary]\n";
    print_separator();
    print_row("operation", "CPU avg ms", "GPU avg ms", "speedup", "correct");
    print_separator();
    for (const auto &row : rows)
    {
        print_row(
            row.operation,
            format_ms(row.cpu_avg_ms),
            format_ms(row.gpu_avg_ms),
            format_speedup(row.cpu_avg_ms, row.gpu_avg_ms),
            row.correctness);
    }
    print_separator();
    std::cout << "iterations=" << iterations
              << ", warmup=" << warmup << "\n";
    if (full_smoke_iterations.has_value())
    {
        std::cout << "full/evalmod smoke iterations=" << *full_smoke_iterations;
        if (full_smoke_warmup.has_value())
        {
            std::cout << ", warmup=" << *full_smoke_warmup;
        }
        std::cout << "\n";
    }
    std::cout << "excluded from timing: matrix generation/upload, Galois key generation/upload, "
                 "per-level compact key precompute, constant plaintext encode/upload, "
                 "ciphertext download/compare\n";
    std::cout << "note: GPU rotate/key-switch reuses setup-time compacted per-q keys\n";
}

}  // namespace

int main()
{
    try
    {
        const int cuda_status = require_cuda_device();
        if (cuda_status != EXIT_SUCCESS)
        {
            return cuda_status;
        }

        const int device_id = 0;
        RmmPoolScope rmm_scope(device_id);

        const std::size_t degree =
            env_size_or("POSEIDON_BOOTSTRAP_DEGREE", 65536);
        const std::size_t q_count =
            env_size_or("POSEIDON_BOOTSTRAP_Q_COUNT", 8);
        const std::size_t p_count =
            env_size_or("POSEIDON_BOOTSTRAP_P_COUNT", 2);
        const std::uint32_t log_q = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_LOG_Q", 30));
        const std::uint32_t log_p = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_LOG_P", log_q));
        const std::uint32_t log_scale = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_LOG_SCALE", 25));
        const std::uint32_t q0_level = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_Q0_LEVEL", 0));
        const std::uint32_t bootstrap_ratio = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_MESSAGE_RATIO", 1));
        const std::size_t warmup =
            env_size_or("POSEIDON_BOOTSTRAP_WARMUP", 2);
        const std::size_t iterations =
            env_size_or("POSEIDON_BOOTSTRAP_ITERATIONS", 10);
        const std::size_t full_smoke_warmup =
            env_size_or(
                "POSEIDON_BOOTSTRAP_FULL_WARMUP",
                std::min<std::size_t>(warmup, 1));
        const std::size_t full_smoke_iterations =
            env_size_or(
                "POSEIDON_BOOTSTRAP_FULL_ITERATIONS",
                std::min<std::size_t>(iterations, 3));
        const double c2s_scaling =
            env_double_or("POSEIDON_BOOTSTRAP_C2S_SCALING", 1.0);
        const std::uint32_t c2s_log_bsgs_ratio = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_C2S_LOG_BSGS_RATIO", 1));
        const std::uint32_t c2s_step = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_C2S_STEP", 1));
        const double s2c_scaling =
            env_double_or("POSEIDON_BOOTSTRAP_S2C_SCALING", 1.0);
        const std::uint32_t s2c_log_bsgs_ratio = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_S2C_LOG_BSGS_RATIO", 1));
        const std::uint32_t s2c_step = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_S2C_STEP", 1));

        if (q_count <= q0_level + 1)
        {
            throw std::invalid_argument(
                "POSEIDON_BOOTSTRAP_Q_COUNT must be greater than q0_level + 1");
        }
        if (iterations == 0)
        {
            throw std::invalid_argument("POSEIDON_BOOTSTRAP_ITERATIONS must be nonzero");
        }
        if (full_smoke_iterations == 0)
        {
            throw std::invalid_argument(
                "POSEIDON_BOOTSTRAP_FULL_ITERATIONS must be nonzero");
        }

        auto parms = make_test_parameters(
            degree,
            q_count,
            p_count,
            log_q,
            log_p,
            log_scale,
            q0_level);

        poseidon::PoseidonFactory::get_instance()->set_device_type(
            poseidon::DEVICE_SOFTWARE);
        auto context =
            poseidon::PoseidonFactory::get_instance()->create_poseidon_context(parms);
        auto cpu_evaluator =
            poseidon::PoseidonFactory::get_instance()->create_ckks_evaluator(context);
        poseidon::CKKSEncoder encoder(context);

        poseidon::KeyGenerator keygen(context);
        poseidon::PublicKey public_key;
        keygen.create_public_key(public_key);
        poseidon::RelinKeys relin_keys;
        keygen.create_relin_keys(relin_keys);
        poseidon::Encryptor encryptor(context, public_key, keygen.secret_key());

        const auto message = make_message(std::size_t{1} << parms.log_slots());
        poseidon::Plaintext plain;
        encoder.encode(message, std::exp2(static_cast<double>(log_scale)), plain);

        poseidon::Ciphertext source;
        encryptor.encrypt(plain, source);

        const double target_q0_scale =
            q0_over_message_ratio(context, bootstrap_ratio);
        const auto q0_parms_id =
            context.crt_context()->parms_id_map().at(q0_level);

        std::cout << "[bootstrap ModRaise test parameters]\n";
        std::cout << "degree          = " << degree << "\n";
        std::cout << "q_count         = " << q_count << "\n";
        std::cout << "p_count         = " << p_count << "\n";
        std::cout << "log_q           = " << log_q << "\n";
        std::cout << "log_p           = " << log_p << "\n";
        std::cout << "log_scale       = " << log_scale << "\n";
        std::cout << "q0_level        = " << q0_level << "\n";
        std::cout << "message_ratio   = " << bootstrap_ratio << "\n";
        std::cout << "c2s_scaling     = " << c2s_scaling << "\n";
        std::cout << "c2s_bsgs_ratio  = " << c2s_log_bsgs_ratio << "\n";
        std::cout << "c2s_step        = " << c2s_step << "\n";
        std::cout << "c2s_rescale     = ordinary\n";
        std::cout << "s2c_scaling     = " << s2c_scaling << "\n";
        std::cout << "s2c_bsgs_ratio  = " << s2c_log_bsgs_ratio << "\n";
        std::cout << "s2c_step        = " << s2c_step << "\n";
        std::cout << "s2c_rescale     = ordinary\n";
        std::cout << "iterations      = " << iterations << "\n";
        std::cout << "full_iterations = " << full_smoke_iterations << "\n";
        std::cout << "full_warmup     = " << full_smoke_warmup << "\n";

        poseidon::gpu::GpuParameterData gpu_params(context, device_id);
        poseidon::gpu::GpuEvaluator gpu_evaluator(gpu_params);
        auto gpu_relin_keys =
            poseidon::gpu::GpuUploader::upload_relin_keys(
                relin_keys,
                device_id);
        auto gpu_source =
            poseidon::gpu::GpuUploader::upload_ciphertext(source, device_id);

        poseidon::Ciphertext cpu_reference =
            cpu_bootstrap_prepare_and_raise(
                source,
                *cpu_evaluator,
                context,
                encoder,
                target_q0_scale);

        poseidon::gpu::GpuCiphertextData gpu_prepared;
        poseidon::gpu::GpuCiphertextData gpu_raised;
        gpu_evaluator.bootstrap_prepare_modraise_input(
            gpu_source,
            gpu_prepared,
            q0_parms_id,
            target_q0_scale);
        gpu_evaluator.raise_modulus(gpu_prepared, gpu_raised);
        cudaDeviceSynchronize();

        poseidon::Ciphertext gpu_download;
        poseidon::gpu::GpuUploader::download_ciphertext(
            gpu_raised,
            gpu_download,
            context);

        const auto comparison =
            compare_ciphertexts(cpu_reference, gpu_download, 8);

        std::cout << "\n[CPU/GPU bootstrap ModRaise comparison]\n";
        std::cout << "cpu raw words   = " << comparison.expected_words << "\n";
        std::cout << "gpu raw words   = " << comparison.actual_words << "\n";
        std::cout << "mismatches      = " << comparison.mismatch_count << "\n";
        std::cout << "raw_equal       = "
                  << (comparison.equal ? "YES" : "NO") << "\n";
        std::cout << "result q_count  = " << gpu_download.coeff_modulus_size()
                  << "\n";
        std::cout << "result NTT form = " << gpu_download.is_ntt_form() << "\n";
        std::cout << "result scale    = " << std::setprecision(12)
                  << gpu_download.scale() << "\n";

        if (!comparison.equal)
        {
            return EXIT_FAILURE;
        }

        for (std::size_t i = 0; i < warmup; ++i)
        {
            (void)cpu_bootstrap_prepare_and_raise(
                source,
                *cpu_evaluator,
                context,
                encoder,
                target_q0_scale);
            gpu_evaluator.bootstrap_prepare_modraise_input(
                gpu_source,
                gpu_prepared,
                q0_parms_id,
                target_q0_scale);
            gpu_evaluator.raise_modulus(gpu_prepared, gpu_raised);
        }
        cudaDeviceSynchronize();

        double cpu_ms = 0.0;
        poseidon::Ciphertext cpu_timing_sink;
        cpu_ms = time_cpu_ms(iterations, [&]() {
            cpu_timing_sink =
                cpu_bootstrap_prepare_and_raise(
                    source,
                    *cpu_evaluator,
                    context,
                    encoder,
                    target_q0_scale);
        });

        double gpu_ms = 0.0;
        gpu_ms = time_gpu_ms(iterations, [&]() {
            gpu_evaluator.bootstrap_prepare_modraise_input(
                gpu_source,
                gpu_prepared,
                q0_parms_id,
                target_q0_scale);
            gpu_evaluator.raise_modulus(gpu_prepared, gpu_raised);
        });

        std::cout << "\n[CoeffToSlot setup]\n";
        std::cout << "Generating CPU C2S matrix and rotation keys outside timing...\n";
        auto c2s_matrix_group =
            make_coeff_to_slot_matrix_group(
                context,
                encoder,
                c2s_scaling,
                c2s_log_bsgs_ratio,
                c2s_step);
        auto galois_keys =
            make_galois_keys_for_matrix_group(
                context,
                keygen,
                c2s_matrix_group);
        auto gpu_c2s_matrix_group =
            poseidon::gpu::GpuUploader::upload_linear_matrix_group(
                c2s_matrix_group,
                device_id);
        auto gpu_galois_keys =
            poseidon::gpu::GpuUploader::upload_galois_keys(
                galois_keys,
                device_id);
        const auto c2s_key_q_counts = required_dft_key_q_counts(
            gpu_raised.meta.q_count,
            c2s_matrix_group.data().size(),
            true);
        poseidon::gpu::GpuUploader::precompute_compacted_keys_for_q_counts(
            gpu_galois_keys,
            c2s_key_q_counts);
        std::cout << "c2s matrices    = " << c2s_matrix_group.data().size() << "\n";
        std::cout << "rotation keys   = " << c2s_matrix_group.rot_index().size() + 1
                  << "\n";
        std::cout << "compact key q   = " << join_q_counts(c2s_key_q_counts) << "\n";
        std::cout << "gpu input q     = " << gpu_raised.meta.q_count
                  << " parms_level="
                  << format_context_level(context, gpu_raised.meta.parms_id)
                  << "\n";
        print_matrix_group_levels(context, c2s_matrix_group, "c2s");

        poseidon::Ciphertext cpu_dft_for_plain;
        cpu_dft_rescale(
            cpu_reference,
            c2s_matrix_group,
            cpu_dft_for_plain,
            *cpu_evaluator,
            galois_keys);
        poseidon::Plaintext minus_i_plain;
        encoder.encode(
            std::complex<double>(0.0, -1.0),
            cpu_dft_for_plain.parms_id(),
            1.0,
            minus_i_plain);
        auto gpu_minus_i_plain =
            poseidon::gpu::GpuUploader::upload_plaintext(
                minus_i_plain,
                device_id);

        poseidon::Ciphertext cpu_c2s_real;
        poseidon::Ciphertext cpu_c2s_imag;
        cpu_coeff_to_slot_rescale(
            cpu_reference,
            c2s_matrix_group,
            cpu_c2s_real,
            cpu_c2s_imag,
            *cpu_evaluator,
            galois_keys,
            encoder);

        poseidon::gpu::GpuCiphertextData gpu_c2s_real;
        poseidon::gpu::GpuCiphertextData gpu_c2s_imag;
        try
        {
            gpu_evaluator.coeff_to_slot(
                gpu_raised,
                gpu_c2s_matrix_group,
                gpu_minus_i_plain,
                gpu_galois_keys,
                gpu_c2s_real,
                gpu_c2s_imag);
        }
        catch (const std::exception &ex)
        {
            throw std::runtime_error(
                std::string("GPU coeff_to_slot correctness failed: input q_count=") +
                std::to_string(gpu_raised.meta.q_count) +
                ", input parms_level=" +
                format_context_level(context, gpu_raised.meta.parms_id) +
                ", minus_i parms_level=" +
                format_context_level(context, gpu_minus_i_plain.meta.parms_id) +
                ": " + ex.what());
        }
        cudaDeviceSynchronize();

        poseidon::Ciphertext gpu_c2s_real_download;
        poseidon::Ciphertext gpu_c2s_imag_download;
        poseidon::gpu::GpuUploader::download_ciphertext(
            gpu_c2s_real,
            gpu_c2s_real_download,
            context);
        poseidon::gpu::GpuUploader::download_ciphertext(
            gpu_c2s_imag,
            gpu_c2s_imag_download,
            context);

        const auto c2s_real_comparison =
            compare_ciphertexts(cpu_c2s_real, gpu_c2s_real_download, 8);
        const auto c2s_imag_comparison =
            compare_ciphertexts(cpu_c2s_imag, gpu_c2s_imag_download, 8);

        std::cout << "\n[CPU/GPU bootstrap CoeffToSlot comparison]\n";
        std::cout << "real raw words  = " << c2s_real_comparison.expected_words << "\n";
        std::cout << "real mismatches = " << c2s_real_comparison.mismatch_count << "\n";
        std::cout << "real raw_equal  = "
                  << (c2s_real_comparison.equal ? "YES" : "NO") << "\n";
        std::cout << "imag raw words  = " << c2s_imag_comparison.expected_words << "\n";
        std::cout << "imag mismatches = " << c2s_imag_comparison.mismatch_count << "\n";
        std::cout << "imag raw_equal  = "
                  << (c2s_imag_comparison.equal ? "YES" : "NO") << "\n";
        std::cout << "result q_count  = "
                  << gpu_c2s_real_download.coeff_modulus_size() << "\n";
        std::cout << "result NTT form = "
                  << gpu_c2s_real_download.is_ntt_form() << "\n";

        if (!c2s_real_comparison.equal || !c2s_imag_comparison.equal)
        {
            return EXIT_FAILURE;
        }

        for (std::size_t i = 0; i < warmup; ++i)
        {
            cpu_coeff_to_slot_rescale(
                cpu_reference,
                c2s_matrix_group,
                cpu_c2s_real,
                cpu_c2s_imag,
                *cpu_evaluator,
                galois_keys,
                encoder);
            gpu_evaluator.coeff_to_slot(
                gpu_raised,
                gpu_c2s_matrix_group,
                gpu_minus_i_plain,
                gpu_galois_keys,
                gpu_c2s_real,
                gpu_c2s_imag);
        }
        cudaDeviceSynchronize();

        double cpu_c2s_ms = 0.0;
        cpu_c2s_ms = time_cpu_ms(iterations, [&]() {
            poseidon::Ciphertext real;
            poseidon::Ciphertext imag;
            cpu_coeff_to_slot_rescale(
                cpu_reference,
                c2s_matrix_group,
                real,
                imag,
                *cpu_evaluator,
                galois_keys,
                encoder);
        });

        double gpu_c2s_ms = 0.0;
        gpu_c2s_ms = time_gpu_ms(iterations, [&]() {
            gpu_evaluator.coeff_to_slot(
                gpu_raised,
                gpu_c2s_matrix_group,
                gpu_minus_i_plain,
                gpu_galois_keys,
                gpu_c2s_real,
                gpu_c2s_imag);
        });

        std::cout << "\n[SlotToCoeff setup]\n";
        std::cout << "Generating CPU S2C matrix and rotation keys outside timing...\n";
        const auto s2c_level_start = static_cast<std::uint32_t>(
            context.crt_context()->get_context_data(cpu_c2s_real.parms_id())->level());
        auto s2c_matrix_group =
            make_slot_to_coeff_matrix_group(
                context,
                encoder,
                s2c_level_start,
                s2c_scaling,
                s2c_log_bsgs_ratio,
                s2c_step);
        auto s2c_galois_keys =
            make_galois_keys_for_matrix_group(
                context,
                keygen,
                s2c_matrix_group);
        auto gpu_s2c_matrix_group =
            poseidon::gpu::GpuUploader::upload_linear_matrix_group(
                s2c_matrix_group,
                device_id);
        auto gpu_s2c_galois_keys =
            poseidon::gpu::GpuUploader::upload_galois_keys(
                s2c_galois_keys,
                device_id);
        const auto s2c_key_q_counts = required_dft_key_q_counts(
            gpu_c2s_real.meta.q_count,
            s2c_matrix_group.data().size(),
            false);
        poseidon::gpu::GpuUploader::precompute_compacted_keys_for_q_counts(
            gpu_s2c_galois_keys,
            s2c_key_q_counts);

        poseidon::Plaintext plus_i_plain;
        encoder.encode(
            std::complex<double>(0.0, 1.0),
            cpu_c2s_imag.parms_id(),
            1.0,
            plus_i_plain);
        auto gpu_plus_i_plain =
            poseidon::gpu::GpuUploader::upload_plaintext(
                plus_i_plain,
                device_id);

        std::cout << "s2c matrices    = " << s2c_matrix_group.data().size() << "\n";
        std::cout << "rotation keys   = " << s2c_matrix_group.rot_index().size() + 1
                  << "\n";
        std::cout << "compact key q   = " << join_q_counts(s2c_key_q_counts) << "\n";
        std::cout << "gpu input q     = " << gpu_c2s_real.meta.q_count
                  << " parms_level="
                  << format_context_level(context, gpu_c2s_real.meta.parms_id)
                  << "\n";
        print_matrix_group_levels(context, s2c_matrix_group, "s2c");

        poseidon::Ciphertext cpu_s2c_result;
        cpu_slot_to_coeff_rescale(
            cpu_c2s_real,
            cpu_c2s_imag,
            s2c_matrix_group,
            cpu_s2c_result,
            *cpu_evaluator,
            s2c_galois_keys,
            encoder);

        poseidon::gpu::GpuCiphertextData gpu_s2c_result;
        try
        {
            gpu_evaluator.slot_to_coeff(
                gpu_c2s_real,
                gpu_c2s_imag,
                gpu_s2c_matrix_group,
                gpu_plus_i_plain,
                gpu_s2c_galois_keys,
                gpu_s2c_result);
        }
        catch (const std::exception &ex)
        {
            throw std::runtime_error(
                std::string("GPU slot_to_coeff correctness failed: real q_count=") +
                std::to_string(gpu_c2s_real.meta.q_count) +
                ", real parms_level=" +
                format_context_level(context, gpu_c2s_real.meta.parms_id) +
                ", plus_i parms_level=" +
                format_context_level(context, gpu_plus_i_plain.meta.parms_id) +
                ": " + ex.what());
        }
        cudaDeviceSynchronize();

        poseidon::Ciphertext gpu_s2c_download;
        poseidon::gpu::GpuUploader::download_ciphertext(
            gpu_s2c_result,
            gpu_s2c_download,
            context);

        const auto s2c_comparison =
            compare_ciphertexts(cpu_s2c_result, gpu_s2c_download, 8);

        std::cout << "\n[CPU/GPU bootstrap SlotToCoeff comparison]\n";
        std::cout << "cpu raw words   = " << s2c_comparison.expected_words << "\n";
        std::cout << "gpu raw words   = " << s2c_comparison.actual_words << "\n";
        std::cout << "mismatches      = " << s2c_comparison.mismatch_count << "\n";
        std::cout << "raw_equal       = "
                  << (s2c_comparison.equal ? "YES" : "NO") << "\n";
        std::cout << "result q_count  = "
                  << gpu_s2c_download.coeff_modulus_size() << "\n";
        std::cout << "result NTT form = "
                  << gpu_s2c_download.is_ntt_form() << "\n";

        if (!s2c_comparison.equal)
        {
            return EXIT_FAILURE;
        }

        double cpu_evalmod_identity_ms = 0.0;
        double gpu_evalmod_identity_ms = 0.0;
        double cpu_full_bootstrap_ms = 0.0;
        double gpu_full_bootstrap_ms = 0.0;
        bool evalmod_identity_correct = false;
        bool full_scheduler_correct = false;

        {
        std::cout << "\n[Full GPU bootstrap scheduler smoke]\n";
        const auto eval_mod_target_scale = gpu_c2s_real.meta.scale;
        if (gpu_c2s_real.meta.q_count < 2)
        {
            throw std::invalid_argument(
                "full bootstrap smoke requires at least two q limbs after CoeffToSlot");
        }
        const auto eval_mod_output_q_count = gpu_c2s_real.meta.q_count - 1;
        if (eval_mod_output_q_count < 1)
        {
            throw std::invalid_argument(
                "full bootstrap smoke EvalMod output q_count is invalid");
        }
        const auto eval_mod_output_level =
            static_cast<std::uint32_t>(eval_mod_output_q_count - 1);
        const auto eval_mod_output_parms_id =
            context.crt_context()->parms_id_map().at(eval_mod_output_level);

        auto full_s2c_matrix_group =
            make_slot_to_coeff_matrix_group(
                context,
                encoder,
                eval_mod_output_level,
                s2c_scaling,
                s2c_log_bsgs_ratio,
                s2c_step);
        auto full_galois_keys =
            make_galois_keys_for_matrix_groups(
                context,
                keygen,
                std::vector<const poseidon::LinearMatrixGroup *>{
                    &c2s_matrix_group,
                    &full_s2c_matrix_group});
        auto gpu_full_galois_keys =
            poseidon::gpu::GpuUploader::upload_galois_keys(
                full_galois_keys,
                device_id);

        const auto full_c2s_key_q_counts = required_dft_key_q_counts(
            gpu_raised.meta.q_count,
            c2s_matrix_group.data().size(),
            true);
        const auto full_s2c_key_q_counts = required_dft_key_q_counts(
            eval_mod_output_q_count,
            full_s2c_matrix_group.data().size(),
            false);
        const auto full_key_q_counts =
            merge_q_counts(full_c2s_key_q_counts, full_s2c_key_q_counts);
        poseidon::gpu::GpuUploader::precompute_compacted_keys_for_q_counts(
            gpu_full_galois_keys,
            full_key_q_counts);
        poseidon::gpu::GpuUploader::precompute_compacted_keys_for_q_counts(
            gpu_relin_keys,
            std::vector<std::size_t>{gpu_c2s_real.meta.q_count});

        auto gpu_full_s2c_matrix_group =
            poseidon::gpu::GpuUploader::upload_linear_matrix_group(
                full_s2c_matrix_group,
                device_id);

        auto gpu_bootstrap_c2s_matrix_group =
            poseidon::gpu::GpuUploader::upload_linear_matrix_group(
                c2s_matrix_group,
                device_id);
        auto gpu_bootstrap_minus_i_plain =
            poseidon::gpu::GpuUploader::upload_plaintext(
                minus_i_plain,
                device_id);

        poseidon::Plaintext full_plus_i_plain;
        encoder.encode(
            std::complex<double>(0.0, 1.0),
            eval_mod_output_parms_id,
            1.0,
            full_plus_i_plain);
        auto gpu_bootstrap_full_plus_i_plain =
            poseidon::gpu::GpuUploader::upload_plaintext(
                full_plus_i_plain,
                device_id);

        poseidon::Plaintext evalmod_one_plain;
        encoder.encode(
            std::complex<double>(1.0, 0.0),
            gpu_c2s_real.meta.parms_id,
            eval_mod_target_scale,
            evalmod_one_plain);
        auto gpu_evalmod_one_plain =
            poseidon::gpu::GpuUploader::upload_plaintext(
                evalmod_one_plain,
                device_id);

        poseidon::gpu::GpuBootstrapData bootstrap_data;
        bootstrap_data.q0_parms_id = q0_parms_id;
        bootstrap_data.q0_over_message_ratio = target_q0_scale;
        bootstrap_data.coeff_to_slot_matrix = std::move(gpu_bootstrap_c2s_matrix_group);
        bootstrap_data.slot_to_coeff_matrix = std::move(gpu_full_s2c_matrix_group);
        bootstrap_data.minus_i_plaintext = std::move(gpu_bootstrap_minus_i_plain);
        bootstrap_data.plus_i_plaintext = std::move(gpu_bootstrap_full_plus_i_plain);
        bootstrap_data.eval_mod.target_scale = eval_mod_target_scale;
        bootstrap_data.eval_mod.polynomial_coefficients.push_back(
            std::move(gpu_evalmod_one_plain));
        bootstrap_data.eval_mod.polynomial_coefficients.emplace_back();

        poseidon::gpu::GpuBootstrapWorkspace bootstrap_workspace;
        poseidon::gpu::GpuCiphertextData gpu_full_bootstrap_result;
        gpu_evaluator.bootstrap(
            gpu_source,
            bootstrap_data,
            gpu_relin_keys,
            gpu_full_galois_keys,
            bootstrap_workspace,
            gpu_full_bootstrap_result);
        cudaDeviceSynchronize();

        auto apply_identity_evalmod =
            [&](const poseidon::gpu::GpuCiphertextData &input,
                poseidon::gpu::GpuCiphertextData &output) {
                poseidon::gpu::GpuCiphertextData x;
                poseidon::gpu::GpuCiphertextData accumulator;
                poseidon::gpu::GpuCiphertextData with_one;
                poseidon::gpu::GpuCiphertextData multiplied;
                poseidon::gpu::GpuCiphertextData relined;
                poseidon::gpu::GpuCiphertextData rescaled;

                gpu_evaluator.multiply_scalar(input, 1, x);
                x.meta.scale = eval_mod_target_scale;
                gpu_evaluator.multiply_scalar(x, 0, accumulator);
                accumulator.meta.scale = eval_mod_target_scale;
                gpu_evaluator.add_plain(
                    accumulator,
                    bootstrap_data.eval_mod.polynomial_coefficients.front(),
                    with_one);
                with_one.meta.scale = eval_mod_target_scale;
                gpu_evaluator.multiply(with_one, x, multiplied);
                gpu_evaluator.relinearize(multiplied, gpu_relin_keys, relined);
                gpu_evaluator.rescale(relined, rescaled);
                rescaled.meta.scale = eval_mod_target_scale;
                output = std::move(rescaled);
            };

        poseidon::gpu::GpuCiphertextData manual_c2s_real;
        poseidon::gpu::GpuCiphertextData manual_c2s_imag;
        poseidon::gpu::GpuCiphertextData manual_eval_real;
        poseidon::gpu::GpuCiphertextData manual_eval_imag;
        poseidon::gpu::GpuCiphertextData manual_full_result;
        gpu_evaluator.coeff_to_slot(
            gpu_raised,
            bootstrap_data.coeff_to_slot_matrix,
            bootstrap_data.minus_i_plaintext,
            gpu_full_galois_keys,
            manual_c2s_real,
            manual_c2s_imag);
        apply_identity_evalmod(manual_c2s_real, manual_eval_real);
        apply_identity_evalmod(manual_c2s_imag, manual_eval_imag);

        poseidon::Ciphertext cpu_full_c2s_real;
        poseidon::Ciphertext cpu_full_c2s_imag;
        poseidon::Ciphertext cpu_full_eval_real;
        poseidon::Ciphertext cpu_full_eval_imag;
        poseidon::Ciphertext cpu_full_result;
        cpu_coeff_to_slot_rescale(
            cpu_reference,
            c2s_matrix_group,
            cpu_full_c2s_real,
            cpu_full_c2s_imag,
            *cpu_evaluator,
            full_galois_keys,
            encoder);
        cpu_evalmod_identity_horner(
            cpu_full_c2s_real,
            cpu_full_eval_real,
            *cpu_evaluator,
            relin_keys,
            encoder,
            eval_mod_target_scale);
        cpu_evalmod_identity_horner(
            cpu_full_c2s_imag,
            cpu_full_eval_imag,
            *cpu_evaluator,
            relin_keys,
            encoder,
            eval_mod_target_scale);

        poseidon::Ciphertext full_c2s_real_download;
        poseidon::Ciphertext full_c2s_imag_download;
        poseidon::Ciphertext full_eval_real_download;
        poseidon::Ciphertext full_eval_imag_download;
        poseidon::Ciphertext manual_c2s_real_download;
        poseidon::Ciphertext manual_c2s_imag_download;
        poseidon::Ciphertext manual_eval_real_download;
        poseidon::Ciphertext manual_eval_imag_download;
        poseidon::gpu::GpuUploader::download_ciphertext(
            bootstrap_workspace.coeff_to_slot_real,
            full_c2s_real_download,
            context);
        poseidon::gpu::GpuUploader::download_ciphertext(
            bootstrap_workspace.coeff_to_slot_imag,
            full_c2s_imag_download,
            context);
        poseidon::gpu::GpuUploader::download_ciphertext(
            bootstrap_workspace.eval_mod_real,
            full_eval_real_download,
            context);
        poseidon::gpu::GpuUploader::download_ciphertext(
            bootstrap_workspace.eval_mod_imag,
            full_eval_imag_download,
            context);
        poseidon::gpu::GpuUploader::download_ciphertext(
            manual_c2s_real,
            manual_c2s_real_download,
            context);
        poseidon::gpu::GpuUploader::download_ciphertext(
            manual_c2s_imag,
            manual_c2s_imag_download,
            context);
        poseidon::gpu::GpuUploader::download_ciphertext(
            manual_eval_real,
            manual_eval_real_download,
            context);
        poseidon::gpu::GpuUploader::download_ciphertext(
            manual_eval_imag,
            manual_eval_imag_download,
            context);
        const auto full_c2s_real_comparison =
            compare_ciphertexts(manual_c2s_real_download, full_c2s_real_download, 0);
        const auto full_c2s_imag_comparison =
            compare_ciphertexts(manual_c2s_imag_download, full_c2s_imag_download, 0);
        const auto full_eval_real_comparison =
            compare_ciphertexts(manual_eval_real_download, full_eval_real_download, 0);
        const auto full_eval_imag_comparison =
            compare_ciphertexts(manual_eval_imag_download, full_eval_imag_download, 0);
        const auto cpu_gpu_eval_real_comparison =
            compare_ciphertexts(cpu_full_eval_real, manual_eval_real_download, 0);
        const auto cpu_gpu_eval_imag_comparison =
            compare_ciphertexts(cpu_full_eval_imag, manual_eval_imag_download, 0);

        gpu_evaluator.slot_to_coeff(
            manual_eval_real,
            manual_eval_imag,
            bootstrap_data.slot_to_coeff_matrix,
            bootstrap_data.plus_i_plaintext,
            gpu_full_galois_keys,
            manual_full_result);
        cudaDeviceSynchronize();
        cpu_slot_to_coeff_rescale(
            cpu_full_eval_real,
            cpu_full_eval_imag,
            full_s2c_matrix_group,
            cpu_full_result,
            *cpu_evaluator,
            full_galois_keys,
            encoder);

        poseidon::Ciphertext gpu_full_bootstrap_download;
        poseidon::Ciphertext manual_full_download;
        poseidon::gpu::GpuUploader::download_ciphertext(
            gpu_full_bootstrap_result,
            gpu_full_bootstrap_download,
            context);
        poseidon::gpu::GpuUploader::download_ciphertext(
            manual_full_result,
            manual_full_download,
            context);
        const auto full_bootstrap_comparison =
            compare_ciphertexts(manual_full_download, gpu_full_bootstrap_download, 8);
        const auto cpu_gpu_full_bootstrap_comparison =
            compare_ciphertexts(cpu_full_result, gpu_full_bootstrap_download, 8);
        evalmod_identity_correct =
            cpu_gpu_eval_real_comparison.equal &&
            cpu_gpu_eval_imag_comparison.equal;
        full_scheduler_correct =
            full_bootstrap_comparison.equal &&
            cpu_gpu_full_bootstrap_comparison.equal;
        std::cout << "evalmod smoke   = f(x)=x\n";
        std::cout << "compact key q   = " << join_q_counts(full_key_q_counts) << "\n";
        std::cout << "c2s real equal  = "
                  << (full_c2s_real_comparison.equal ? "YES" : "NO") << "\n";
        std::cout << "c2s imag equal  = "
                  << (full_c2s_imag_comparison.equal ? "YES" : "NO") << "\n";
        std::cout << "eval real equal = "
                  << (full_eval_real_comparison.equal ? "YES" : "NO") << "\n";
        std::cout << "eval imag equal = "
                  << (full_eval_imag_comparison.equal ? "YES" : "NO") << "\n";
        std::cout << "eval CPU/GPU    = "
                  << (evalmod_identity_correct ? "YES" : "NO") << "\n";
        std::cout << "result q_count  = "
                  << gpu_full_bootstrap_download.coeff_modulus_size() << "\n";
        std::cout << "mismatches      = "
                  << full_bootstrap_comparison.mismatch_count << "\n";
        std::cout << "raw_equal       = "
                  << (full_scheduler_correct ? "YES" : "NO") << "\n";
        if (!full_scheduler_correct)
        {
            return EXIT_FAILURE;
        }

        auto run_cpu_full_bootstrap_smoke = [&]() {
            poseidon::Ciphertext raised =
                cpu_bootstrap_prepare_and_raise(
                    source,
                    *cpu_evaluator,
                    context,
                    encoder,
                    target_q0_scale);
            poseidon::Ciphertext real;
            poseidon::Ciphertext imag;
            poseidon::Ciphertext eval_real;
            poseidon::Ciphertext eval_imag;
            poseidon::Ciphertext result;
            cpu_coeff_to_slot_rescale(
                raised,
                c2s_matrix_group,
                real,
                imag,
                *cpu_evaluator,
                full_galois_keys,
                encoder);
            cpu_evalmod_identity_horner(
                real,
                eval_real,
                *cpu_evaluator,
                relin_keys,
                encoder,
                eval_mod_target_scale);
            cpu_evalmod_identity_horner(
                imag,
                eval_imag,
                *cpu_evaluator,
                relin_keys,
                encoder,
                eval_mod_target_scale);
            cpu_slot_to_coeff_rescale(
                eval_real,
                eval_imag,
                full_s2c_matrix_group,
                result,
                *cpu_evaluator,
                full_galois_keys,
                encoder);
            return result;
        };

        for (std::size_t i = 0; i < full_smoke_warmup; ++i)
        {
            poseidon::Ciphertext cpu_eval_real_warmup;
            poseidon::Ciphertext cpu_eval_imag_warmup;
            cpu_evalmod_identity_horner(
                cpu_full_c2s_real,
                cpu_eval_real_warmup,
                *cpu_evaluator,
                relin_keys,
                encoder,
                eval_mod_target_scale);
            cpu_evalmod_identity_horner(
                cpu_full_c2s_imag,
                cpu_eval_imag_warmup,
                *cpu_evaluator,
                relin_keys,
                encoder,
                eval_mod_target_scale);

            poseidon::gpu::GpuCiphertextData gpu_eval_real_warmup;
            poseidon::gpu::GpuCiphertextData gpu_eval_imag_warmup;
            apply_identity_evalmod(manual_c2s_real, gpu_eval_real_warmup);
            apply_identity_evalmod(manual_c2s_imag, gpu_eval_imag_warmup);

            (void)run_cpu_full_bootstrap_smoke();
            gpu_evaluator.bootstrap(
                gpu_source,
                bootstrap_data,
                gpu_relin_keys,
                gpu_full_galois_keys,
                bootstrap_workspace,
                gpu_full_bootstrap_result);
        }
        cudaDeviceSynchronize();

        cpu_evalmod_identity_ms = time_cpu_ms(full_smoke_iterations, [&]() {
            poseidon::Ciphertext eval_real;
            poseidon::Ciphertext eval_imag;
            cpu_evalmod_identity_horner(
                cpu_full_c2s_real,
                eval_real,
                *cpu_evaluator,
                relin_keys,
                encoder,
                eval_mod_target_scale);
            cpu_evalmod_identity_horner(
                cpu_full_c2s_imag,
                eval_imag,
                *cpu_evaluator,
                relin_keys,
                encoder,
                eval_mod_target_scale);
        });

        gpu_evalmod_identity_ms = time_gpu_ms(full_smoke_iterations, [&]() {
            poseidon::gpu::GpuCiphertextData eval_real;
            poseidon::gpu::GpuCiphertextData eval_imag;
            apply_identity_evalmod(manual_c2s_real, eval_real);
            apply_identity_evalmod(manual_c2s_imag, eval_imag);
        });

        poseidon::Ciphertext cpu_full_timing_sink;
        cpu_full_bootstrap_ms = time_cpu_ms(full_smoke_iterations, [&]() {
            cpu_full_timing_sink = run_cpu_full_bootstrap_smoke();
        });

        gpu_full_bootstrap_ms = time_gpu_ms(full_smoke_iterations, [&]() {
            gpu_evaluator.bootstrap(
                gpu_source,
                bootstrap_data,
                gpu_relin_keys,
                gpu_full_galois_keys,
                bootstrap_workspace,
                gpu_full_bootstrap_result);
        });
        }

        for (std::size_t i = 0; i < warmup; ++i)
        {
            cpu_slot_to_coeff_rescale(
                cpu_c2s_real,
                cpu_c2s_imag,
                s2c_matrix_group,
                cpu_s2c_result,
                *cpu_evaluator,
                s2c_galois_keys,
                encoder);
            gpu_evaluator.slot_to_coeff(
                gpu_c2s_real,
                gpu_c2s_imag,
                gpu_s2c_matrix_group,
                gpu_plus_i_plain,
                gpu_s2c_galois_keys,
                gpu_s2c_result);
        }
        cudaDeviceSynchronize();

        double cpu_s2c_ms = 0.0;
        cpu_s2c_ms = time_cpu_ms(iterations, [&]() {
            poseidon::Ciphertext result;
            cpu_slot_to_coeff_rescale(
                cpu_c2s_real,
                cpu_c2s_imag,
                s2c_matrix_group,
                result,
                *cpu_evaluator,
                s2c_galois_keys,
                encoder);
        });

        double gpu_s2c_ms = 0.0;
        gpu_s2c_ms = time_gpu_ms(iterations, [&]() {
            gpu_evaluator.slot_to_coeff(
                gpu_c2s_real,
                gpu_c2s_imag,
                gpu_s2c_matrix_group,
                gpu_plus_i_plain,
                gpu_s2c_galois_keys,
                gpu_s2c_result);
        });

        print_bootstrap_timing_table(
            std::vector<TimingRow>{
                TimingRow{
                    "prepare_modraise_input + raise_modulus",
                    cpu_ms,
                    gpu_ms,
                    comparison.equal ? "YES" : "NO"},
                TimingRow{
                    "coeff_to_slot",
                    cpu_c2s_ms,
                    gpu_c2s_ms,
                    (c2s_real_comparison.equal && c2s_imag_comparison.equal) ? "YES" : "NO"},
                TimingRow{
                    "slot_to_coeff",
                    cpu_s2c_ms,
                    gpu_s2c_ms,
                    s2c_comparison.equal ? "YES" : "NO"},
                TimingRow{
                    "eval_mod_identity_smoke",
                    cpu_evalmod_identity_ms,
                    gpu_evalmod_identity_ms,
                    evalmod_identity_correct ? "YES" : "NO"},
                TimingRow{
                    "full_bootstrap_scheduler_smoke",
                    cpu_full_bootstrap_ms,
                    gpu_full_bootstrap_ms,
                    full_scheduler_correct ? "YES" : "NO"}},
            iterations,
            warmup,
            full_smoke_iterations,
            full_smoke_warmup);

        std::cout << "\n[OK] GPU bootstrap ModRaise, CoeffToSlot, SlotToCoeff, "
                     "EvalMod smoke, and full scheduler smoke match CPU reference\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[FAILED] " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
