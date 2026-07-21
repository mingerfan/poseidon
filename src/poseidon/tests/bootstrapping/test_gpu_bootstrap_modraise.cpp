#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/advance/homomorphic_mod.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/decryptor.h"
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

std::uint32_t log2_power_of_two(std::uint32_t value, const char *name)
{
    if (value == 0 || (value & (value - 1)) != 0)
    {
        throw std::invalid_argument(std::string(name) + " must be a power of two");
    }

    std::uint32_t result = 0;
    while (value > 1)
    {
        value >>= 1;
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

struct RawComparison
{
    bool equal = false;
    std::size_t expected_words = 0;
    std::size_t actual_words = 0;
    std::size_t mismatch_count = 0;
};

struct ApproxComparison
{
    bool equal = false;
    double max_abs_error = 0.0;
    double rms_error = 0.0;
};

struct TimingRow
{
    std::string operation;
    double cpu_avg_ms = 0.0;
    double gpu_avg_ms = 0.0;
    std::string correctness = "N/A";
};

struct EvalModTraceRow
{
    std::string stage;
    std::size_t cpu_q_count = 0;
    std::size_t gpu_q_count = 0;
    double cpu_log2_scale = 0.0;
    double gpu_log2_scale = 0.0;
    ApproxComparison comparison;
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

std::vector<std::complex<double>> decrypt_decode(
    const poseidon::Ciphertext &ciphertext,
    poseidon::Decryptor &decryptor,
    const poseidon::CKKSEncoder &encoder)
{
    poseidon::Plaintext plaintext;
    std::vector<std::complex<double>> values;
    decryptor.decrypt(ciphertext, plaintext);
    encoder.decode(plaintext, values);
    return values;
}

ApproxComparison compare_approx(
    const std::vector<std::complex<double>> &expected,
    const std::vector<std::complex<double>> &actual,
    double tolerance)
{
    ApproxComparison result;
    if (expected.size() != actual.size() || expected.empty())
    {
        return result;
    }

    long double squared_error_sum = 0.0L;
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const double error = std::abs(expected[i] - actual[i]);
        if (!std::isfinite(error))
        {
            result.max_abs_error = std::numeric_limits<double>::infinity();
            result.rms_error = std::numeric_limits<double>::infinity();
            return result;
        }
        result.max_abs_error = std::max(result.max_abs_error, error);
        squared_error_sum += static_cast<long double>(error) * error;
    }
    result.rms_error = std::sqrt(
        static_cast<double>(squared_error_sum / expected.size()));
    result.equal = result.max_abs_error <= tolerance;
    return result;
}

ApproxComparison compare_decrypted_ciphertexts(
    const poseidon::Ciphertext &expected,
    const poseidon::Ciphertext &actual,
    poseidon::Decryptor &decryptor,
    const poseidon::CKKSEncoder &encoder,
    double tolerance)
{
    return compare_approx(
        decrypt_decode(expected, decryptor, encoder),
        decrypt_decode(actual, decryptor, encoder),
        tolerance);
}

poseidon::Ciphertext download_gpu_ciphertext(
    const poseidon::gpu::GpuCiphertextData &source,
    const poseidon::PoseidonContext &context)
{
    poseidon::Ciphertext result;
    poseidon::gpu::GpuUploader::download_ciphertext(source, result, context);
    return result;
}

EvalModTraceRow compare_eval_mod_trace_stage(
    std::string stage,
    const poseidon::Ciphertext &cpu,
    const poseidon::Ciphertext &gpu,
    poseidon::Decryptor &decryptor,
    const poseidon::CKKSEncoder &encoder,
    double tolerance)
{
    EvalModTraceRow row;
    row.stage = std::move(stage);
    row.cpu_q_count = cpu.coeff_modulus_size();
    row.gpu_q_count = gpu.coeff_modulus_size();
    row.cpu_log2_scale = std::log2(cpu.scale());
    row.gpu_log2_scale = std::log2(gpu.scale());
    row.comparison = compare_decrypted_ciphertexts(
        cpu,
        gpu,
        decryptor,
        encoder,
        tolerance);
    return row;
}

void print_eval_mod_trace_table(const std::vector<EvalModTraceRow> &rows)
{
    constexpr int stage_width = 28;
    constexpr int q_width = 7;
    constexpr int scale_width = 12;
    constexpr int error_width = 16;
    constexpr int correct_width = 9;

    auto separator = [&]() {
        std::cout << "|-" << std::string(stage_width, '-')
                  << "-|-" << std::string(q_width, '-')
                  << "-|-" << std::string(q_width, '-')
                  << "-|-" << std::string(scale_width, '-')
                  << "-|-" << std::string(scale_width, '-')
                  << "-|-" << std::string(error_width, '-')
                  << "-|-" << std::string(correct_width, '-') << "-|\n";
    };
    auto print_row = [&](const std::string &stage,
                         const std::string &cpu_q,
                         const std::string &gpu_q,
                         const std::string &cpu_scale,
                         const std::string &gpu_scale,
                         const std::string &error,
                         const std::string &correct) {
        std::cout << "| " << std::left << std::setw(stage_width) << stage
                  << " | " << std::right << std::setw(q_width) << cpu_q
                  << " | " << std::right << std::setw(q_width) << gpu_q
                  << " | " << std::right << std::setw(scale_width) << cpu_scale
                  << " | " << std::right << std::setw(scale_width) << gpu_scale
                  << " | " << std::right << std::setw(error_width) << error
                  << " | " << std::right << std::setw(correct_width) << correct
                  << " |\n";
    };

    std::cout << "\n[EvalMod CPU/GPU stage trace: real branch]\n";
    separator();
    print_row(
        "stage", "CPU q", "GPU q", "CPU log2(s)", "GPU log2(s)",
        "max abs error", "correct");
    separator();

    std::string first_divergent_stage = "none";
    for (const auto &row : rows)
    {
        std::ostringstream cpu_scale;
        std::ostringstream gpu_scale;
        std::ostringstream error;
        cpu_scale << std::fixed << std::setprecision(3) << row.cpu_log2_scale;
        gpu_scale << std::fixed << std::setprecision(3) << row.gpu_log2_scale;
        error << std::scientific << std::setprecision(6)
              << row.comparison.max_abs_error;
        print_row(
            row.stage,
            std::to_string(row.cpu_q_count),
            std::to_string(row.gpu_q_count),
            cpu_scale.str(),
            gpu_scale.str(),
            error.str(),
            row.comparison.equal ? "YES" : "NO");
        if (!row.comparison.equal && first_divergent_stage == "none")
        {
            first_divergent_stage = row.stage;
        }
    }
    separator();
    std::cout << "first divergent stage = " << first_divergent_stage << "\n";
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
    std::optional<std::size_t> full_iterations = std::nullopt,
    std::optional<std::size_t> full_warmup = std::nullopt)
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
    if (full_iterations.has_value())
    {
        std::cout << "full/evalmod iterations=" << *full_iterations;
        if (full_warmup.has_value())
        {
            std::cout << ", warmup=" << *full_warmup;
        }
        std::cout << "\n";
    }
    std::cout << "excluded from timing: matrix generation/upload, Galois key generation/upload, "
                 "CPU EvalMod level probe, zero-copy key-view validation, "
                 "constant plaintext encode/upload, "
                 "EvalMod stage-trace snapshots/download, ciphertext download/compare\n";
    std::cout << "note: GPU rotate/key-switch reads zero-copy Q-prefix/P-tail key views\n";
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
            env_size_or("POSEIDON_BOOTSTRAP_DEGREE", 32768);
        const std::size_t q_count =
            env_size_or("POSEIDON_BOOTSTRAP_Q_COUNT", 20);
        const std::size_t p_count =
            env_size_or("POSEIDON_BOOTSTRAP_P_COUNT", 5);
        const std::uint32_t log_q = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_LOG_Q", 30));
        const std::uint32_t log_p = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_LOG_P", log_q));
        const std::uint32_t log_scale = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_LOG_SCALE", 25));
        const std::uint32_t q0_level = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_Q0_LEVEL", 1));
        const std::uint32_t bootstrap_ratio = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_MESSAGE_RATIO", 512));
        const std::size_t warmup =
            env_size_or("POSEIDON_BOOTSTRAP_WARMUP", 0);
        const std::size_t iterations =
            env_size_or("POSEIDON_BOOTSTRAP_ITERATIONS", 1);
        const std::size_t full_warmup =
            env_size_or(
                "POSEIDON_BOOTSTRAP_FULL_WARMUP",
                0);
        const std::size_t full_iterations =
            env_size_or(
                "POSEIDON_BOOTSTRAP_FULL_ITERATIONS",
                1);
        double c2s_scaling = env_double_or(
            "POSEIDON_BOOTSTRAP_C2S_SCALING",
            std::numeric_limits<double>::quiet_NaN());
        const std::uint32_t c2s_log_bsgs_ratio = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_C2S_LOG_BSGS_RATIO", 1));
        const std::uint32_t c2s_step = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_C2S_STEP", 1));
        double s2c_scaling = env_double_or(
            "POSEIDON_BOOTSTRAP_S2C_SCALING",
            std::numeric_limits<double>::quiet_NaN());
        const std::uint32_t s2c_log_bsgs_ratio = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_S2C_LOG_BSGS_RATIO", 1));
        const std::uint32_t s2c_step = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_S2C_STEP", 1));
        const std::uint32_t evalmod_log_scale = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_EVALMOD_LOG_SCALE", log_q));
        const std::uint32_t evalmod_double_angle = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_EVALMOD_DOUBLE_ANGLE", 3));
        const std::uint32_t evalmod_k = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_EVALMOD_K", 16));
        const std::uint32_t evalmod_arcsine_degree = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_EVALMOD_ARCSINE_DEGREE", 0));
        const std::uint32_t evalmod_sine_degree = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_EVALMOD_SINE_DEGREE", 30));
        const double correctness_tolerance = env_double_or(
            "POSEIDON_BOOTSTRAP_CORRECTNESS_TOLERANCE", 1.0e-3);
        const std::uint32_t evalmod_log_message_ratio =
            log2_power_of_two(bootstrap_ratio, "POSEIDON_BOOTSTRAP_MESSAGE_RATIO");

        if (q_count <= q0_level + 1)
        {
            throw std::invalid_argument(
                "POSEIDON_BOOTSTRAP_Q_COUNT must be greater than q0_level + 1");
        }
        if (iterations == 0)
        {
            throw std::invalid_argument("POSEIDON_BOOTSTRAP_ITERATIONS must be nonzero");
        }
        if (full_iterations == 0)
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

        const double evalmod_scale =
            std::exp2(static_cast<double>(evalmod_log_scale));
        poseidon::EvalModPoly eval_mod_poly(
            context,
            poseidon::CosDiscrete,
            evalmod_scale,
            /*level_start=*/0,
            evalmod_log_message_ratio,
            evalmod_double_angle,
            evalmod_k,
            evalmod_arcsine_degree,
            evalmod_sine_degree);
        if (!std::isfinite(c2s_scaling))
        {
            c2s_scaling =
                eval_mod_poly.q_div() /
                (eval_mod_poly.k() * eval_mod_poly.sc_fac() * eval_mod_poly.q_diff());
        }
        if (!std::isfinite(s2c_scaling))
        {
            s2c_scaling =
                context.parameters_literal()->scale() /
                (eval_mod_poly.scaling_factor() / eval_mod_poly.message_ratio());
        }

        poseidon::KeyGenerator keygen(context);
        poseidon::PublicKey public_key;
        keygen.create_public_key(public_key);
        poseidon::RelinKeys relin_keys;
        keygen.create_relin_keys(relin_keys);
        poseidon::Encryptor encryptor(context, public_key, keygen.secret_key());
        poseidon::Decryptor decryptor(context, keygen.secret_key());

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
        std::cout << "evalmod_scale   = 2^" << evalmod_log_scale << "\n";
        std::cout << "evalmod_degree  = " << evalmod_sine_degree << "\n";
        std::cout << "double_angle    = " << evalmod_double_angle << "\n";
        std::cout << "evalmod_k       = " << evalmod_k << "\n";
        std::cout << "tolerance       = " << correctness_tolerance << "\n";
        std::cout << "c2s_scaling     = " << c2s_scaling << "\n";
        std::cout << "c2s_bsgs_ratio  = " << c2s_log_bsgs_ratio << "\n";
        std::cout << "c2s_step        = " << c2s_step << "\n";
        std::cout << "c2s_rescale     = ordinary\n";
        std::cout << "s2c_scaling     = " << s2c_scaling << "\n";
        std::cout << "s2c_bsgs_ratio  = " << s2c_log_bsgs_ratio << "\n";
        std::cout << "s2c_step        = " << s2c_step << "\n";
        std::cout << "s2c_rescale     = ordinary\n";
        std::cout << "iterations      = " << iterations << "\n";
        std::cout << "full_iterations = " << full_iterations << "\n";
        std::cout << "full_warmup     = " << full_warmup << "\n";

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

        std::cout << "\n[Unified high-precision bootstrap setup]\n";
        std::cout << "Generating CPU matrices/plans/keys and uploading one GPU data set "
                     "outside timing...\n";

        const double raised_scale = static_cast<double>(
            context.crt_context()->first_context_data()->coeff_modulus()[0].value());
        const double post_raise_multiplier = std::round(
            eval_mod_poly.scaling_factor() /
            raised_scale /
            eval_mod_poly.message_ratio());
        if (post_raise_multiplier >
            static_cast<double>(std::numeric_limits<int>::max()))
        {
            throw std::invalid_argument(
                "post-raise multiplier exceeds CPU/GPU direct integer path");
        }

        auto c2s_matrix_group = make_coeff_to_slot_matrix_group(
            context,
            encoder,
            c2s_scaling,
            c2s_log_bsgs_ratio,
            c2s_step);
        if (c2s_matrix_group.data().size() >= gpu_raised.meta.q_count)
        {
            throw std::invalid_argument(
                "CoeffToSlot consumes the complete modulus chain");
        }

        const std::size_t c2s_output_q_count =
            gpu_raised.meta.q_count - c2s_matrix_group.data().size();
        const auto c2s_output_parms_id =
            context.crt_context()->parms_id_map().at(
                static_cast<std::uint32_t>(c2s_output_q_count - 1));
        const auto c2s_output_context =
            context.crt_context()->get_context_data(c2s_output_parms_id);
        if (!c2s_output_context)
        {
            throw std::runtime_error(
                "CoeffToSlot output parms_id is absent from the context");
        }

        eval_mod_poly.set_level_start(
            static_cast<std::uint32_t>(c2s_output_context->level()));

        /*
         * Probe the CPU high-precision schedule once during untimed setup.
         * The GPU uploader records this observable output level while still
         * generating a fixed BSGS/ordinary-rescale GPU execution plan.
         */
        poseidon::Plaintext cpu_evalmod_probe_plain;
        encoder.encode(
            std::complex<double>(0.125, 0.0),
            c2s_output_parms_id,
            evalmod_scale,
            cpu_evalmod_probe_plain);
        poseidon::Ciphertext cpu_evalmod_probe_input;
        encryptor.encrypt(cpu_evalmod_probe_plain, cpu_evalmod_probe_input);
        cpu_evalmod_probe_input.scale() = evalmod_scale;
        poseidon::Ciphertext cpu_evalmod_probe_output;
        cpu_evaluator->eval_mod_high_precision(
            cpu_evalmod_probe_input,
            cpu_evalmod_probe_output,
            eval_mod_poly,
            relin_keys,
            encoder);
        const auto cpu_evalmod_output_parms_id =
            cpu_evalmod_probe_output.parms_id();
        const auto cpu_evalmod_output_context =
            context.crt_context()->get_context_data(
                cpu_evalmod_output_parms_id);
        if (!cpu_evalmod_output_context)
        {
            throw std::runtime_error(
                "CPU EvalMod setup probe returned an unknown output parms_id");
        }

        auto gpu_evalmod_data =
            poseidon::gpu::GpuUploader::upload_eval_mod_high_precision(
                eval_mod_poly,
                encoder,
                c2s_output_parms_id,
                device_id,
                &gpu_relin_keys,
                cpu_evalmod_output_parms_id);
        const auto evalmod_output_q_count =
            gpu_evalmod_data.output_q_count;
        const auto evalmod_output_parms_id =
            gpu_evalmod_data.output_parms_id;
        const auto evalmod_output_context =
            context.crt_context()->get_context_data(evalmod_output_parms_id);
        if (!evalmod_output_context ||
            evalmod_output_parms_id != cpu_evalmod_output_parms_id)
        {
            throw std::runtime_error(
                "GPU EvalMod setup did not preserve the CPU output parms_id");
        }

        auto s2c_matrix_group = make_slot_to_coeff_matrix_group(
            context,
            encoder,
            static_cast<std::uint32_t>(evalmod_output_context->level()),
            s2c_scaling,
            s2c_log_bsgs_ratio,
            s2c_step);
        auto full_galois_keys = make_galois_keys_for_matrix_groups(
            context,
            keygen,
            std::vector<const poseidon::LinearMatrixGroup *>{
                &c2s_matrix_group,
                &s2c_matrix_group});

        /*
         * Keep the library bootstrap as the correctness oracle. The staged
         * CPU path below intentionally mirrors the GPU ordinary-rescale
         * scheduler and must independently agree with this result.
         */
        auto cpu_library_eval_mod_poly = eval_mod_poly;
        poseidon::Ciphertext cpu_library_bootstrap_result;
        cpu_evaluator->bootstrap(
            source,
            cpu_library_bootstrap_result,
            relin_keys,
            full_galois_keys,
            encoder,
            cpu_library_eval_mod_poly);

        poseidon::Plaintext minus_i_plain;
        encoder.encode(
            std::complex<double>(0.0, -1.0),
            c2s_output_parms_id,
            1.0,
            minus_i_plain);
        poseidon::Plaintext plus_i_plain;
        encoder.encode(
            std::complex<double>(0.0, 1.0),
            evalmod_output_parms_id,
            1.0,
            plus_i_plain);

        auto gpu_full_galois_keys =
            poseidon::gpu::GpuUploader::upload_galois_keys(
                full_galois_keys,
                device_id);
        const auto c2s_key_q_counts = required_dft_key_q_counts(
            gpu_raised.meta.q_count,
            c2s_matrix_group.data().size(),
            true);
        const auto s2c_key_q_counts = required_dft_key_q_counts(
            evalmod_output_q_count,
            s2c_matrix_group.data().size(),
            false);
        const auto full_key_q_counts =
            merge_q_counts(c2s_key_q_counts, s2c_key_q_counts);
        poseidon::gpu::GpuUploader::prepare_key_views_for_q_counts(
            gpu_full_galois_keys,
            full_key_q_counts);

        poseidon::gpu::GpuBootstrapData bootstrap_data;
        bootstrap_data.q0_parms_id = q0_parms_id;
        bootstrap_data.q0_over_message_ratio = target_q0_scale;
        bootstrap_data.raised_scale_override = raised_scale;
        bootstrap_data.slot_to_coeff_input_scale =
            context.parameters_literal()->scale();
        if (post_raise_multiplier > 1.0)
        {
            bootstrap_data.post_raise_integer_multiplier =
                static_cast<std::uint64_t>(post_raise_multiplier);
            bootstrap_data.post_raise_scale_multiplier =
                post_raise_multiplier;
        }
        bootstrap_data.coeff_to_slot_matrix =
            poseidon::gpu::GpuUploader::upload_linear_matrix_group(
                c2s_matrix_group,
                device_id);
        bootstrap_data.slot_to_coeff_matrix =
            poseidon::gpu::GpuUploader::upload_linear_matrix_group(
                s2c_matrix_group,
                device_id);
        bootstrap_data.minus_i_plaintext =
            poseidon::gpu::GpuUploader::upload_plaintext(
                minus_i_plain,
                device_id);
        bootstrap_data.plus_i_plaintext =
            poseidon::gpu::GpuUploader::upload_plaintext(
                plus_i_plain,
                device_id);
        bootstrap_data.eval_mod = std::move(gpu_evalmod_data);

        std::cout << "c2s matrices     = "
                  << c2s_matrix_group.data().size() << "\n";
        std::cout << "s2c matrices     = "
                  << s2c_matrix_group.data().size() << "\n";
        std::cout << "rotation keys    = "
                  << full_galois_keys.data().size() << "\n";
        std::cout << "Galois key views = "
                  << join_q_counts(full_key_q_counts) << "\n";
        std::cout << "polynomial degree= "
                  << eval_mod_poly.sine_poly().degree() << "\n";
        std::cout << "basis steps       = "
                  << bootstrap_data.eval_mod.basis_steps.size() << "\n";
        std::cout << "BSGS leaf blocks  = "
                  << bootstrap_data.eval_mod.polynomial_blocks.size() << "\n";
        std::cout << "BSGS combines     = "
                  << bootstrap_data.eval_mod.polynomial_combine_steps.size() << "\n";
        std::cout << "double-angle      = "
                  << bootstrap_data.eval_mod.double_angle_constants.size() << "\n";
        std::cout << "EvalMod input q   = "
                  << c2s_output_q_count << "\n";
        std::cout << "EvalMod output q  = "
                  << evalmod_output_q_count << "\n";
        std::cout << "CPU EvalMod out q = "
                  << cpu_evalmod_probe_output.coeff_modulus_size() << "\n";
        std::cout << "Relin key views   = "
                  << join_q_counts(
                         bootstrap_data.eval_mod.required_relin_q_counts)
                  << "\n";
        std::cout << "GPU setup objects = one C2S matrix group, one S2C matrix group, "
                     "one Galois-key set\n";

        poseidon::Ciphertext cpu_full_raised = cpu_reference;
        cpu_full_raised.scale() = raised_scale;
        if (post_raise_multiplier > 1.0)
        {
            cpu_evaluator->multiply_const_direct(
                cpu_full_raised,
                static_cast<int>(post_raise_multiplier),
                cpu_full_raised,
                encoder);
            cpu_full_raised.scale() *= post_raise_multiplier;
        }

        poseidon::gpu::GpuCiphertextData gpu_full_raised;
        gpu_evaluator.multiply_scalar(
            gpu_raised,
            post_raise_multiplier > 1.0
                ? static_cast<std::uint64_t>(post_raise_multiplier)
                : std::uint64_t{1},
            gpu_full_raised);
        gpu_full_raised.meta.scale =
            raised_scale *
            (post_raise_multiplier > 1.0
                 ? post_raise_multiplier
                 : 1.0);

        poseidon::Ciphertext cpu_c2s_real;
        poseidon::Ciphertext cpu_c2s_imag;
        cpu_coeff_to_slot_rescale(
            cpu_full_raised,
            c2s_matrix_group,
            cpu_c2s_real,
            cpu_c2s_imag,
            *cpu_evaluator,
            full_galois_keys,
            encoder);

        poseidon::gpu::GpuBootstrapWorkspace bootstrap_workspace;
        auto &gpu_c2s_real = bootstrap_workspace.coeff_to_slot_real;
        auto &gpu_c2s_imag = bootstrap_workspace.coeff_to_slot_imag;
        gpu_evaluator.coeff_to_slot(
            gpu_full_raised,
            bootstrap_data.coeff_to_slot_matrix,
            bootstrap_data.minus_i_plaintext,
            gpu_full_galois_keys,
            gpu_c2s_real,
            gpu_c2s_imag);
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
        const bool c2s_correct =
            c2s_real_comparison.equal && c2s_imag_comparison.equal;

        std::cout << "\n[CPU/GPU CoeffToSlot correctness]\n";
        std::cout << "real raw_equal = "
                  << (c2s_real_comparison.equal ? "YES" : "NO") << "\n";
        std::cout << "imag raw_equal = "
                  << (c2s_imag_comparison.equal ? "YES" : "NO") << "\n";
        if (!c2s_correct)
        {
            return EXIT_FAILURE;
        }

        poseidon::Ciphertext cpu_eval_real;
        poseidon::Ciphertext cpu_eval_imag;
        poseidon::EvalModTrace cpu_eval_real_trace;
        cpu_evaluator->eval_mod_high_precision(
            cpu_c2s_real,
            cpu_eval_real,
            eval_mod_poly,
            relin_keys,
            encoder,
            &cpu_eval_real_trace);
        cpu_evaluator->eval_mod_high_precision(
            cpu_c2s_imag,
            cpu_eval_imag,
            eval_mod_poly,
            relin_keys,
            encoder);
        if (cpu_eval_real.parms_id() != cpu_evalmod_output_parms_id ||
            cpu_eval_imag.parms_id() != cpu_evalmod_output_parms_id)
        {
            throw std::runtime_error(
                "CPU EvalMod output level changed between setup probe and correctness execution");
        }

        auto &gpu_eval_real = bootstrap_workspace.eval_mod_real;
        auto &gpu_eval_imag = bootstrap_workspace.eval_mod_imag;
        bootstrap_workspace.capture_eval_mod_trace = true;
        gpu_evaluator.eval_mod_high_precision(
            gpu_c2s_real,
            bootstrap_data,
            gpu_relin_keys,
            bootstrap_workspace,
            gpu_eval_real);
        cudaDeviceSynchronize();

        const auto gpu_trace_offset_input = download_gpu_ciphertext(
            bootstrap_workspace.eval_mod_trace_offset_input,
            context);
        const auto gpu_trace_polynomial_output = download_gpu_ciphertext(
            bootstrap_workspace.eval_mod_trace_polynomial_output,
            context);
        std::map<std::uint32_t, poseidon::Ciphertext> gpu_trace_basis;
        for (const auto &[degree_index, cpu_basis_ciphertext] :
             cpu_eval_real_trace.basis)
        {
            (void)cpu_basis_ciphertext;
            if (degree_index < bootstrap_workspace.eval_mod_basis.size() &&
                !bootstrap_workspace.eval_mod_basis[degree_index].empty())
            {
                gpu_trace_basis.emplace(
                    degree_index,
                    download_gpu_ciphertext(
                        bootstrap_workspace.eval_mod_basis[degree_index],
                        context));
            }
        }
        std::vector<poseidon::Ciphertext> gpu_trace_polynomial_leaves;
        gpu_trace_polynomial_leaves.reserve(
            bootstrap_data.eval_mod.polynomial_blocks.size());
        for (std::size_t block_index = 0;
             block_index < bootstrap_data.eval_mod.polynomial_blocks.size();
             ++block_index)
        {
            gpu_trace_polynomial_leaves.push_back(download_gpu_ciphertext(
                bootstrap_workspace.eval_mod_nodes.at(block_index),
                context));
        }
        std::vector<poseidon::Ciphertext> gpu_trace_polynomial_combines;
        gpu_trace_polynomial_combines.reserve(
            bootstrap_data.eval_mod.polynomial_combine_steps.size());
        for (const auto &combine :
             bootstrap_data.eval_mod.polynomial_combine_steps)
        {
            gpu_trace_polynomial_combines.push_back(download_gpu_ciphertext(
                bootstrap_workspace.eval_mod_nodes.at(combine.output_node),
                context));
        }
        std::vector<poseidon::Ciphertext> gpu_trace_double_angle_outputs;
        gpu_trace_double_angle_outputs.reserve(
            bootstrap_workspace.eval_mod_trace_double_angle_outputs.size());
        for (const auto &gpu_trace_ciphertext :
             bootstrap_workspace.eval_mod_trace_double_angle_outputs)
        {
            gpu_trace_double_angle_outputs.push_back(download_gpu_ciphertext(
                gpu_trace_ciphertext,
                context));
        }

        bootstrap_workspace.capture_eval_mod_trace = false;
        gpu_evaluator.eval_mod_high_precision(
            gpu_c2s_imag,
            bootstrap_data,
            gpu_relin_keys,
            bootstrap_workspace,
            gpu_eval_imag);
        cudaDeviceSynchronize();

        poseidon::Ciphertext gpu_eval_real_download;
        poseidon::Ciphertext gpu_eval_imag_download;
        poseidon::gpu::GpuUploader::download_ciphertext(
            gpu_eval_real,
            gpu_eval_real_download,
            context);
        poseidon::gpu::GpuUploader::download_ciphertext(
            gpu_eval_imag,
            gpu_eval_imag_download,
            context);
        if (gpu_eval_real_download.parms_id() != cpu_evalmod_output_parms_id ||
            gpu_eval_imag_download.parms_id() != cpu_evalmod_output_parms_id)
        {
            throw std::runtime_error(
                "GPU EvalMod output level does not match the CPU high-precision schedule");
        }
        const auto eval_real_comparison =
            compare_decrypted_ciphertexts(
                cpu_eval_real,
                gpu_eval_real_download,
                decryptor,
                encoder,
                correctness_tolerance);
        const auto eval_imag_comparison =
            compare_decrypted_ciphertexts(
                cpu_eval_imag,
                gpu_eval_imag_download,
                decryptor,
                encoder,
                correctness_tolerance);
        const bool evalmod_correct =
            eval_real_comparison.equal && eval_imag_comparison.equal;

        std::vector<EvalModTraceRow> eval_mod_trace_rows;
        eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
            "input + offset",
            cpu_eval_real_trace.offset_input,
            gpu_trace_offset_input,
            decryptor,
            encoder,
            correctness_tolerance));
        for (const auto &[degree_index, gpu_basis_ciphertext] : gpu_trace_basis)
        {
            const auto cpu_basis_iter =
                cpu_eval_real_trace.basis.find(degree_index);
            if (cpu_basis_iter == cpu_eval_real_trace.basis.end())
            {
                continue;
            }
            eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                "basis T_" + std::to_string(degree_index),
                cpu_basis_iter->second,
                gpu_basis_ciphertext,
                decryptor,
                encoder,
                correctness_tolerance));
        }
        const auto compared_leaf_count = std::min(
            cpu_eval_real_trace.polynomial_leaves.size(),
            gpu_trace_polynomial_leaves.size());
        for (std::size_t leaf_index = 0;
             leaf_index < compared_leaf_count;
             ++leaf_index)
        {
            eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                "polynomial leaf " + std::to_string(leaf_index),
                cpu_eval_real_trace.polynomial_leaves[leaf_index],
                gpu_trace_polynomial_leaves[leaf_index],
                decryptor,
                encoder,
                correctness_tolerance));
        }
        const auto compared_combine_count = std::min(
            cpu_eval_real_trace.polynomial_combines.size(),
            gpu_trace_polynomial_combines.size());
        for (std::size_t combine_index = 0;
             combine_index < compared_combine_count;
             ++combine_index)
        {
            eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                "polynomial combine " + std::to_string(combine_index),
                cpu_eval_real_trace.polynomial_combines[combine_index],
                gpu_trace_polynomial_combines[combine_index],
                decryptor,
                encoder,
                correctness_tolerance));
        }
        eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
            "polynomial output",
            cpu_eval_real_trace.polynomial_output,
            gpu_trace_polynomial_output,
            decryptor,
            encoder,
            correctness_tolerance));
        const auto compared_double_angle_count = std::min(
            cpu_eval_real_trace.double_angle_outputs.size(),
            gpu_trace_double_angle_outputs.size());
        for (std::size_t double_angle_index = 0;
             double_angle_index < compared_double_angle_count;
             ++double_angle_index)
        {
            eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                "double angle " + std::to_string(double_angle_index),
                cpu_eval_real_trace.double_angle_outputs[double_angle_index],
                gpu_trace_double_angle_outputs[double_angle_index],
                decryptor,
                encoder,
                correctness_tolerance));
        }
        print_eval_mod_trace_table(eval_mod_trace_rows);
        std::cout << "trace counts: basis CPU/GPU="
                  << cpu_eval_real_trace.basis.size() << "/"
                  << gpu_trace_basis.size()
                  << ", leaves CPU/GPU="
                  << cpu_eval_real_trace.polynomial_leaves.size() << "/"
                  << gpu_trace_polynomial_leaves.size()
                  << ", combines CPU/GPU="
                  << cpu_eval_real_trace.polynomial_combines.size() << "/"
                  << gpu_trace_polynomial_combines.size()
                  << ", double-angle CPU/GPU="
                  << cpu_eval_real_trace.double_angle_outputs.size() << "/"
                  << gpu_trace_double_angle_outputs.size() << "\n";

        cpu_eval_real.scale() = context.parameters_literal()->scale();
        cpu_eval_imag.scale() = context.parameters_literal()->scale();
        gpu_eval_real.meta.scale = context.parameters_literal()->scale();
        gpu_eval_imag.meta.scale = context.parameters_literal()->scale();

        poseidon::Ciphertext cpu_s2c_result;
        cpu_slot_to_coeff_rescale(
            cpu_eval_real,
            cpu_eval_imag,
            s2c_matrix_group,
            cpu_s2c_result,
            *cpu_evaluator,
            full_galois_keys,
            encoder);
        poseidon::gpu::GpuCiphertextData gpu_s2c_result;
        gpu_evaluator.slot_to_coeff(
            gpu_eval_real,
            gpu_eval_imag,
            bootstrap_data.slot_to_coeff_matrix,
            bootstrap_data.plus_i_plaintext,
            gpu_full_galois_keys,
            gpu_s2c_result);
        cudaDeviceSynchronize();

        poseidon::Ciphertext gpu_s2c_download;
        poseidon::gpu::GpuUploader::download_ciphertext(
            gpu_s2c_result,
            gpu_s2c_download,
            context);
        const auto s2c_comparison =
            compare_decrypted_ciphertexts(
                cpu_s2c_result,
                gpu_s2c_download,
                decryptor,
                encoder,
                correctness_tolerance);
        const bool s2c_correct = s2c_comparison.equal;

        auto run_cpu_full_bootstrap = [&]() {
            auto raised = cpu_bootstrap_prepare_and_raise(
                source,
                *cpu_evaluator,
                context,
                encoder,
                target_q0_scale);
            raised.scale() = raised_scale;
            if (post_raise_multiplier > 1.0)
            {
                cpu_evaluator->multiply_const_direct(
                    raised,
                    static_cast<int>(post_raise_multiplier),
                    raised,
                    encoder);
                raised.scale() *= post_raise_multiplier;
            }

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
            cpu_evaluator->eval_mod_high_precision(
                real,
                eval_real,
                eval_mod_poly,
                relin_keys,
                encoder);
            cpu_evaluator->eval_mod_high_precision(
                imag,
                eval_imag,
                eval_mod_poly,
                relin_keys,
                encoder);
            eval_real.scale() = context.parameters_literal()->scale();
            eval_imag.scale() = context.parameters_literal()->scale();
            cpu_slot_to_coeff_rescale(
                eval_real,
                eval_imag,
                s2c_matrix_group,
                result,
                *cpu_evaluator,
                full_galois_keys,
                encoder);
            return result;
        };

        poseidon::Ciphertext cpu_full_result =
            run_cpu_full_bootstrap();
        poseidon::gpu::GpuCiphertextData gpu_full_result;
        gpu_evaluator.bootstrap(
            gpu_source,
            bootstrap_data,
            gpu_relin_keys,
            gpu_full_galois_keys,
            bootstrap_workspace,
            gpu_full_result);
        cudaDeviceSynchronize();

        poseidon::Ciphertext gpu_full_download;
        poseidon::gpu::GpuUploader::download_ciphertext(
            gpu_full_result,
            gpu_full_download,
            context);
        const auto full_cpu_gpu_comparison =
            compare_decrypted_ciphertexts(
                cpu_full_result,
                gpu_full_download,
                decryptor,
                encoder,
                correctness_tolerance);
        const auto cpu_library_source_comparison =
            compare_approx(
                message,
                decrypt_decode(
                    cpu_library_bootstrap_result,
                    decryptor,
                    encoder),
                correctness_tolerance);
        const auto staged_cpu_library_comparison =
            compare_decrypted_ciphertexts(
                cpu_library_bootstrap_result,
                cpu_full_result,
                decryptor,
                encoder,
                correctness_tolerance);
        const auto gpu_library_comparison =
            compare_decrypted_ciphertexts(
                cpu_library_bootstrap_result,
                gpu_full_download,
                decryptor,
                encoder,
                correctness_tolerance);
        const auto cpu_source_comparison =
            compare_approx(
                message,
                decrypt_decode(cpu_full_result, decryptor, encoder),
                correctness_tolerance);
        const auto gpu_source_comparison =
            compare_approx(
                message,
                decrypt_decode(gpu_full_download, decryptor, encoder),
                correctness_tolerance);
        const bool full_correct =
            cpu_library_source_comparison.equal &&
            staged_cpu_library_comparison.equal &&
            gpu_library_comparison.equal &&
            full_cpu_gpu_comparison.equal &&
            cpu_source_comparison.equal &&
            gpu_source_comparison.equal;

        std::cout << "\n[High-precision correctness]\n";
        std::cout << "EvalMod real max error = "
                  << eval_real_comparison.max_abs_error << "\n";
        std::cout << "EvalMod imag max error = "
                  << eval_imag_comparison.max_abs_error << "\n";
        std::cout << "S2C CPU/GPU max error  = "
                  << s2c_comparison.max_abs_error << "\n";
        std::cout << "full CPU/GPU max error = "
                  << full_cpu_gpu_comparison.max_abs_error << "\n";
        std::cout << "CPU library/source err = "
                  << cpu_library_source_comparison.max_abs_error << "\n";
        std::cout << "CPU staged/library err = "
                  << staged_cpu_library_comparison.max_abs_error << "\n";
        std::cout << "GPU/CPU library error  = "
                  << gpu_library_comparison.max_abs_error << "\n";
        std::cout << "CPU/source max error   = "
                  << cpu_source_comparison.max_abs_error << "\n";
        std::cout << "GPU/source max error   = "
                  << gpu_source_comparison.max_abs_error << "\n";

        for (std::size_t i = 0; i < warmup; ++i)
        {
            poseidon::Ciphertext cpu_real;
            poseidon::Ciphertext cpu_imag;
            cpu_coeff_to_slot_rescale(
                cpu_full_raised,
                c2s_matrix_group,
                cpu_real,
                cpu_imag,
                *cpu_evaluator,
                full_galois_keys,
                encoder);
            gpu_evaluator.coeff_to_slot(
                gpu_full_raised,
                bootstrap_data.coeff_to_slot_matrix,
                bootstrap_data.minus_i_plaintext,
                gpu_full_galois_keys,
                gpu_c2s_real,
                gpu_c2s_imag);
        }
        cudaDeviceSynchronize();

        const double cpu_c2s_ms =
            time_cpu_ms(iterations, [&]() {
                poseidon::Ciphertext real;
                poseidon::Ciphertext imag;
                cpu_coeff_to_slot_rescale(
                    cpu_full_raised,
                    c2s_matrix_group,
                    real,
                    imag,
                    *cpu_evaluator,
                    full_galois_keys,
                    encoder);
            });
        const double gpu_c2s_ms =
            time_gpu_ms(iterations, [&]() {
                gpu_evaluator.coeff_to_slot(
                    gpu_full_raised,
                    bootstrap_data.coeff_to_slot_matrix,
                    bootstrap_data.minus_i_plaintext,
                    gpu_full_galois_keys,
                    gpu_c2s_real,
                    gpu_c2s_imag);
            });

        for (std::size_t i = 0; i < full_warmup; ++i)
        {
            poseidon::Ciphertext cpu_real;
            poseidon::Ciphertext cpu_imag;
            cpu_evaluator->eval_mod_high_precision(
                cpu_c2s_real,
                cpu_real,
                eval_mod_poly,
                relin_keys,
                encoder);
            cpu_evaluator->eval_mod_high_precision(
                cpu_c2s_imag,
                cpu_imag,
                eval_mod_poly,
                relin_keys,
                encoder);
            gpu_evaluator.eval_mod_high_precision(
                gpu_c2s_real,
                bootstrap_data,
                gpu_relin_keys,
                bootstrap_workspace,
                gpu_eval_real);
            gpu_evaluator.eval_mod_high_precision(
                gpu_c2s_imag,
                bootstrap_data,
                gpu_relin_keys,
                bootstrap_workspace,
                gpu_eval_imag);
            (void)run_cpu_full_bootstrap();
            gpu_evaluator.bootstrap(
                gpu_source,
                bootstrap_data,
                gpu_relin_keys,
                gpu_full_galois_keys,
                bootstrap_workspace,
                gpu_full_result);
        }
        cudaDeviceSynchronize();

        const double cpu_evalmod_ms =
            time_cpu_ms(full_iterations, [&]() {
                poseidon::Ciphertext real;
                poseidon::Ciphertext imag;
                cpu_evaluator->eval_mod_high_precision(
                    cpu_c2s_real,
                    real,
                    eval_mod_poly,
                    relin_keys,
                    encoder);
                cpu_evaluator->eval_mod_high_precision(
                    cpu_c2s_imag,
                    imag,
                    eval_mod_poly,
                    relin_keys,
                    encoder);
            });
        const double gpu_evalmod_ms =
            time_gpu_ms(full_iterations, [&]() {
                gpu_evaluator.eval_mod_high_precision(
                    gpu_c2s_real,
                    bootstrap_data,
                    gpu_relin_keys,
                    bootstrap_workspace,
                    gpu_eval_real);
                gpu_evaluator.eval_mod_high_precision(
                    gpu_c2s_imag,
                    bootstrap_data,
                    gpu_relin_keys,
                    bootstrap_workspace,
                    gpu_eval_imag);
            });

        // EvalMod restores the C2S input scale. CPU bootstrap resets both
        // branches to the library scale before the inverse DFT.
        gpu_eval_real.meta.scale = context.parameters_literal()->scale();
        gpu_eval_imag.meta.scale = context.parameters_literal()->scale();

        const double cpu_s2c_ms =
            time_cpu_ms(full_iterations, [&]() {
                poseidon::Ciphertext result;
                cpu_slot_to_coeff_rescale(
                    cpu_eval_real,
                    cpu_eval_imag,
                    s2c_matrix_group,
                    result,
                    *cpu_evaluator,
                    full_galois_keys,
                    encoder);
            });
        const double gpu_s2c_ms =
            time_gpu_ms(full_iterations, [&]() {
                gpu_evaluator.slot_to_coeff(
                    gpu_eval_real,
                    gpu_eval_imag,
                    bootstrap_data.slot_to_coeff_matrix,
                    bootstrap_data.plus_i_plaintext,
                    gpu_full_galois_keys,
                    gpu_s2c_result);
            });

        const double cpu_full_bootstrap_ms =
            time_cpu_ms(full_iterations, [&]() {
                cpu_full_result = run_cpu_full_bootstrap();
            });
        const double gpu_full_bootstrap_ms =
            time_gpu_ms(full_iterations, [&]() {
                gpu_evaluator.bootstrap(
                    gpu_source,
                    bootstrap_data,
                    gpu_relin_keys,
                    gpu_full_galois_keys,
                    bootstrap_workspace,
                    gpu_full_result);
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
                    c2s_correct ? "YES" : "NO"},
                TimingRow{
                    "eval_mod_high_precision (real + imag)",
                    cpu_evalmod_ms,
                    gpu_evalmod_ms,
                    evalmod_correct ? "YES" : "NO"},
                TimingRow{
                    "slot_to_coeff",
                    cpu_s2c_ms,
                    gpu_s2c_ms,
                    s2c_correct ? "YES" : "NO"},
                TimingRow{
                    "full_bootstrap_high_precision",
                    cpu_full_bootstrap_ms,
                    gpu_full_bootstrap_ms,
                    full_correct ? "YES" : "NO"}},
            iterations,
            warmup,
            full_iterations,
            full_warmup);

        if (!evalmod_correct || !s2c_correct || !full_correct)
        {
            return EXIT_FAILURE;
        }

        std::cout << "\n[OK] One GPU bootstrap data set passed ModRaise, "
                     "CoeffToSlot, high-precision EvalMod, SlotToCoeff, "
                     "and full-bootstrap checks\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[FAILED] " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
