#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/advance/homomorphic_mod.h"
#include "poseidon/advance/bootstrapper.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/evaluator/evaluator_ckks_base.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_keyswitch_handler.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/gpu/kernels/gpu_double_hoist_kernels.h"
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

bool env_flag_enabled(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return false;
    }
    const std::string text(value);
    return text != "0" &&
           text != "OFF" &&
           text != "off" &&
           text != "false" &&
           text != "FALSE";
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

std::uint32_t diagnostic_galois_elt_from_rotation_step(
    std::size_t degree,
    int step)
{
    if (degree == 0 ||
        (degree & (degree - 1)) != 0 ||
        degree > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max() / 2))
    {
        throw std::invalid_argument(
            "diagnostic rotation requires a power-of-two degree");
    }

    const std::uint32_t n = static_cast<std::uint32_t>(degree);
    const std::uint32_t m = n << 1;
    if (step == 0)
    {
        return m - 1;
    }

    const std::int64_t signed_step = static_cast<std::int64_t>(step);
    const bool negative = signed_step < 0;
    const std::uint64_t absolute_step = negative
        ? static_cast<std::uint64_t>(-signed_step)
        : static_cast<std::uint64_t>(signed_step);
    if (absolute_step >= (static_cast<std::uint64_t>(n) >> 1))
    {
        throw std::invalid_argument(
            "diagnostic rotation step is out of range");
    }

    std::uint32_t rotation_count =
        static_cast<std::uint32_t>(absolute_step);
    if (negative)
    {
        rotation_count = (n >> 1) - rotation_count;
    }

    constexpr std::uint64_t generator = 5;
    std::uint64_t galois_elt = 1;
    while (rotation_count-- != 0)
    {
        galois_elt = (galois_elt * generator) & (m - 1);
    }
    return static_cast<std::uint32_t>(galois_elt);
}

std::size_t diagnostic_galois_key_index(std::uint32_t galois_elt)
{
    if ((galois_elt & 1U) == 0U)
    {
        throw std::invalid_argument(
            "diagnostic Galois element must be odd");
    }
    return static_cast<std::size_t>((galois_elt - 1U) >> 1U);
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
    poseidon::EvaluatorCkksBase &evaluator,
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

    poseidon::Bootstrapper bootstrapper(
        context,
        evaluator,
        encoder,
        context.parameters_literal()->log_slots(),
        /*boundary_k=*/25,
        source.scale(),
        context.parameters_literal()->scale(),
        {},
        /*logical_rescale_count=*/2,
        q0_level + 1,
        context.crt_context()->q0());
    poseidon::Ciphertext raised;
    bootstrapper.mod_raise(result, raised);
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
    const poseidon::GaloisKeys &galois_keys,
    std::uint32_t rescale_count)
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

    if (rescale_count == 0)
    {
        throw std::invalid_argument(
            "cpu_multiply_by_diag_matrix_bsgs_rescale: rescale_count must be positive");
    }
    poseidon::Ciphertext current = std::move(accumulator);
    for (std::uint32_t index = 0; index < rescale_count; ++index)
    {
        poseidon::Ciphertext next;
        evaluator.rescale(current, next);
        current = std::move(next);
    }
    result = std::move(current);
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
        galois_keys,
        std::max(matrix_group.step(), std::uint32_t{1}));

    for (std::size_t i = 1; i < matrix_group.data().size(); ++i)
    {
        poseidon::Ciphertext next;
        cpu_multiply_by_diag_matrix_bsgs_rescale(
            result,
            matrix_group.data()[i],
            next,
            evaluator,
            galois_keys,
            std::max(matrix_group.step(), std::uint32_t{1}));
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

struct BootstrapTimingParameters
{
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;
    std::uint32_t log_q = 0;
    std::uint32_t log_p = 0;
    std::uint32_t input_log_scale = 0;
    std::uint32_t evalmod_log_scale = 0;
    std::uint32_t q0_level = 0;
    std::uint32_t message_ratio = 0;
    std::uint32_t evalmod_degree = 0;
    std::uint32_t evalmod_rescale_count = 0;
    std::uint32_t double_angle = 0;
    std::uint32_t c2s_step = 0;
    std::uint32_t s2c_step = 0;
    std::string linear_transform_mode;
};

struct CorrectnessRow
{
    std::string operation;
    bool correct = false;
    double max_abs_error = 0.0;
    std::string comparison;
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

struct DoubleHoistDiagnosticRow
{
    std::string invariant;
    std::string detail;
    std::size_t expected_q_count = 0;
    std::size_t actual_q_count = 0;
    double max_abs_error = 0.0;
    bool raw_equal = false;
    bool correct = false;
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

void print_double_hoist_diagnostic_table(
    const std::vector<DoubleHoistDiagnosticRow> &rows)
{
    constexpr int invariant_width = 28;
    constexpr int detail_width = 14;
    constexpr int q_width = 10;
    constexpr int error_width = 16;
    constexpr int raw_width = 9;
    constexpr int correct_width = 9;

    auto separator = [&]() {
        std::cout << "|-" << std::string(invariant_width, '-')
                  << "-|-" << std::string(detail_width, '-')
                  << "-|-" << std::string(q_width, '-')
                  << "-|-" << std::string(q_width, '-')
                  << "-|-" << std::string(error_width, '-')
                  << "-|-" << std::string(raw_width, '-')
                  << "-|-" << std::string(correct_width, '-') << "-|\n";
    };
    auto print_row = [&](const std::string &invariant,
                         const std::string &detail,
                         const std::string &expected_q,
                         const std::string &actual_q,
                         const std::string &error,
                         const std::string &raw,
                         const std::string &correct) {
        std::cout << "| " << std::left << std::setw(invariant_width)
                  << invariant
                  << " | " << std::setw(detail_width) << detail
                  << " | " << std::right << std::setw(q_width) << expected_q
                  << " | " << std::setw(q_width) << actual_q
                  << " | " << std::setw(error_width) << error
                  << " | " << std::setw(raw_width) << raw
                  << " | " << std::setw(correct_width) << correct << " |\n";
    };

    std::cout << "\n[Double-hoist primitive diagnostics]\n";
    separator();
    print_row(
        "invariant",
        "detail",
        "expected q",
        "actual q",
        "max abs error",
        "raw equal",
        "correct");
    separator();
    for (const auto &row : rows)
    {
        std::ostringstream error;
        error << std::scientific << std::setprecision(6)
              << row.max_abs_error;
        print_row(
            row.invariant,
            row.detail,
            std::to_string(row.expected_q_count),
            std::to_string(row.actual_q_count),
            error.str(),
            row.raw_equal ? "YES" : "NO",
            row.correct ? "YES" : "NO");
    }
    separator();

    const auto first_failure = std::find_if(
        rows.begin(),
        rows.end(),
        [](const DoubleHoistDiagnosticRow &row) {
            return !row.correct;
        });
    std::cout << "first failing invariant = "
              << (first_failure == rows.end()
                      ? "none"
                      : first_failure->invariant)
              << "\n";

    auto passed = [&](const std::string &name) {
        const auto row = std::find_if(
            rows.begin(),
            rows.end(),
            [&](const DoubleHoistDiagnosticRow &candidate) {
                return candidate.invariant == name;
            });
        return row != rows.end() && row->correct;
    };
    std::cout << "diagnosis               = ";
    if (!passed("ModDown(LiftP(ct)) == ct"))
    {
        std::cout << "LiftP or the P=0 ModDown path is incorrect\n";
    }
    else if (!passed("standard monolithic rotate"))
    {
        std::cout << "the existing standard GPU rotate baseline is incorrect\n";
    }
    else if (!passed("standard staged rotate"))
    {
        std::cout << "Hoist, standard staged KeyMult, or nonzero-P ModDown is incorrect\n";
    }
    else if (!passed("pre-rotated staged rotate"))
    {
        std::cout << "inverse-pre-rotated key generation or pre-rotated KeyMult is incorrect\n";
    }
    else if (!passed("DoubleHoist(one matrix)"))
    {
        std::cout << "QP plaintext extension/multiply or inner/outer BSGS scheduling is incorrect\n";
    }
    else
    {
        std::cout << "all isolated double-hoist invariants passed\n";
    }
}

void print_double_hoist_counts(
    const char *stage,
    const poseidon::gpu::GpuDoubleHoistWorkspace &workspace)
{
    std::cout << "\n[double-hoist operation counts: " << stage << "]\n";
    std::cout
        << "| matrix | source D | outer D | keymul | inner R | outer R |"
           " QP PMult | permute | tiles | workspace MiB |\n";
    std::cout
        << "|--------|----------|---------|--------|---------|---------|"
           "----------|---------|-------|---------------|\n";
    for (std::size_t matrix = 0;
         matrix < workspace.matrix_counts.size();
         ++matrix)
    {
        const auto &counts = workspace.matrix_counts[matrix];
        std::cout << "| " << std::setw(6) << matrix
                  << " | " << std::setw(8) << counts.source_decompose_count
                  << " | " << std::setw(7) << counts.outer_decompose_count
                  << " | " << std::setw(6) << counts.keymul_count
                  << " | " << std::setw(7) << counts.inner_moddown_count
                  << " | " << std::setw(7) << counts.outer_moddown_count
                  << " | " << std::setw(8) << counts.qp_pmult_count
                  << " | " << std::setw(7) << counts.permute_count
                  << " | " << std::setw(5) << counts.baby_tile_count
                  << " | " << std::fixed << std::setprecision(2)
                  << std::setw(13)
                  << static_cast<double>(counts.workspace_peak_bytes) /
                         (1024.0 * 1024.0)
                  << " |\n";
    }
    std::cout << std::defaultfloat << std::setprecision(12);
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
    std::uint32_t rescale_count,
    bool include_post_dft_conjugation)
{
    if (rescale_count == 0 ||
        matrix_count > input_q_count / rescale_count)
    {
        throw std::invalid_argument("DFT matrix count exceeds input q_count");
    }

    std::vector<std::size_t> q_counts;
    q_counts.reserve(matrix_count + (include_post_dft_conjugation ? 1 : 0));
    for (std::size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index)
    {
        q_counts.push_back(
            input_q_count - matrix_index * rescale_count);
    }
    if (include_post_dft_conjugation)
    {
        q_counts.push_back(
            input_q_count - matrix_count * rescale_count);
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
    const BootstrapTimingParameters &parameters,
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
    std::cout << "parameters: N=" << parameters.degree
              << ", Q=" << parameters.q_count
              << "x" << parameters.log_q << "-bit"
              << ", P=" << parameters.p_count
              << "x" << parameters.log_p << "-bit"
              << ", input_scale=2^" << parameters.input_log_scale
              << "\n";
    std::cout << "bootstrap: mode=" << parameters.linear_transform_mode
              << ", q0_level=" << parameters.q0_level
              << ", message_ratio=" << parameters.message_ratio
              << ", EvalMod_scale=2^" << parameters.evalmod_log_scale
              << ", degree=" << parameters.evalmod_degree
              << ", rescale_width=" << parameters.evalmod_rescale_count
              << ", double_angle=" << parameters.double_angle
              << ", C2S/S2C_step=" << parameters.c2s_step
              << "/" << parameters.s2c_step
              << "\n";
    std::cout << "excluded from timing: matrix generation/upload, Galois key generation/upload, "
                 "CPU EvalMod level probe, zero-copy key-view validation, "
                 "constant plaintext encode/upload, "
                 "EvalMod stage-trace snapshots/download, ciphertext download/compare\n";
    std::cout << "timing order: CPU warmup -> CPU timing; "
                 "GPU warmup -> GPU timing, independently for every operation\n";
    std::cout << "note: GPU rotate/key-switch reads zero-copy Q-prefix/P-tail key views\n";
}

void accumulate_eval_mod_stage_timing(
    poseidon::gpu::GpuBootstrapWorkspace::EvalModStageTiming &sum,
    const poseidon::gpu::GpuBootstrapWorkspace::EvalModStageTiming &sample)
{
    sum.input_preparation_ms += sample.input_preparation_ms;
    sum.basis_generation_ms += sample.basis_generation_ms;
    sum.leaf_evaluation_ms += sample.leaf_evaluation_ms;
    sum.bsgs_combine_ms += sample.bsgs_combine_ms;
    sum.double_angle_ms += sample.double_angle_ms;
    sum.output_alignment_ms += sample.output_alignment_ms;
    sum.total_ms += sample.total_ms;
}

void print_eval_mod_stage_timing_table(
    const poseidon::gpu::GpuBootstrapWorkspace::EvalModStageTiming &sum,
    double profiled_wall_ms_sum,
    std::size_t iterations)
{
    const double divisor = static_cast<double>(iterations);
    const double total = sum.total_ms / divisor;
    const double profiled_wall_total = profiled_wall_ms_sum / divisor;
    const double host_gap = std::max(profiled_wall_total - total, 0.0);
    const auto average = [divisor](double value) {
        return value / divisor;
    };
    const auto share = [profiled_wall_total](double value) {
        if (!(profiled_wall_total > 0.0))
        {
            return std::string("N/A");
        }
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(1)
               << value * 100.0 / profiled_wall_total << "%";
        return stream.str();
    };

    constexpr int op_width = 30;
    constexpr int time_width = 14;
    constexpr int share_width = 10;
    const auto separator = [&]() {
        std::cout << "|-" << std::string(op_width, '-')
                  << "-|-" << std::string(time_width, '-')
                  << "-|-" << std::string(share_width, '-')
                  << "-|\n";
    };
    const auto row = [&](const std::string &operation, double milliseconds) {
        std::cout << "| " << std::left << std::setw(op_width) << operation
                  << " | " << std::right << std::setw(time_width)
                  << format_ms(milliseconds)
                  << " | " << std::right << std::setw(share_width)
                  << share(milliseconds)
                  << " |\n";
    };

    std::cout << "\n[GPU EvalMod stage timing: real + imag]\n";
    separator();
    std::cout << "| " << std::left << std::setw(op_width) << "stage"
              << " | " << std::right << std::setw(time_width) << "GPU avg ms"
              << " | " << std::right << std::setw(share_width) << "share"
              << " |\n";
    separator();
    row("input preparation", average(sum.input_preparation_ms));
    row("Chebyshev basis generation", average(sum.basis_generation_ms));
    row("fused leaf evaluation", average(sum.leaf_evaluation_ms));
    row("BSGS tree combine", average(sum.bsgs_combine_ms));
    row("double-angle iterations", average(sum.double_angle_ms));
    row("output level alignment", average(sum.output_alignment_ms));
    separator();
    row("CUDA-event active total", total);
    row("host/allocator/sync gap", host_gap);
    row("profiled wall total", profiled_wall_total);
    separator();
    std::cout << "profile iterations=" << iterations
              << "; real + imag; setup/download excluded\n";
}

void print_correctness_summary(const std::vector<CorrectnessRow> &rows)
{
    constexpr int op_width = 34;
    constexpr int comparison_width = 22;
    constexpr int error_width = 16;
    constexpr int correct_width = 9;
    const auto separator = [&]() {
        std::cout << "|-" << std::string(op_width, '-')
                  << "-|-" << std::string(comparison_width, '-')
                  << "-|-" << std::string(error_width, '-')
                  << "-|-" << std::string(correct_width, '-')
                  << "-|\n";
    };

    std::cout << "\n[bootstrap correctness summary]\n";
    separator();
    std::cout << "| " << std::left << std::setw(op_width) << "operation"
              << " | " << std::left << std::setw(comparison_width) << "comparison"
              << " | " << std::right << std::setw(error_width) << "max abs error"
              << " | " << std::right << std::setw(correct_width) << "correct"
              << " |\n";
    separator();
    for (const auto &row : rows)
    {
        std::ostringstream error;
        error << std::scientific << std::setprecision(6)
              << row.max_abs_error;
        std::cout << "| " << std::left << std::setw(op_width) << row.operation
                  << " | " << std::left << std::setw(comparison_width)
                  << row.comparison
                  << " | " << std::right << std::setw(error_width)
                  << error.str()
                  << " | " << std::right << std::setw(correct_width)
                  << (row.correct ? "YES" : "NO")
                  << " |\n";
    }
    separator();
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
            env_size_or("POSEIDON_BOOTSTRAP_DEGREE", 16384);
        const std::size_t q_count =
            env_size_or("POSEIDON_BOOTSTRAP_Q_COUNT", 34);
        const std::size_t p_count =
            env_size_or("POSEIDON_BOOTSTRAP_P_COUNT", 9);
        const std::uint32_t log_q = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_LOG_Q", 30));
        const std::uint32_t log_p = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_LOG_P", log_q));
        const std::uint32_t log_scale = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_LOG_SCALE", 30));
        const std::uint32_t q0_level = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_Q0_LEVEL", 1));
        const std::uint32_t bootstrap_ratio = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_MESSAGE_RATIO", 512));
        const std::size_t warmup =
            env_size_or("POSEIDON_BOOTSTRAP_WARMUP", 1);
        const std::size_t iterations =
            env_size_or("POSEIDON_BOOTSTRAP_ITERATIONS", 1);
        const std::size_t full_warmup =
            env_size_or(
                "POSEIDON_BOOTSTRAP_FULL_WARMUP",
                1);
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
        const std::uint32_t evalmod_rescale_count = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_EVALMOD_RESCALE_COUNT", 1));
        const std::uint32_t evalmod_log_scale = static_cast<std::uint32_t>(
            env_size_or(
                "POSEIDON_BOOTSTRAP_EVALMOD_LOG_SCALE",
                static_cast<std::size_t>(log_q) * evalmod_rescale_count));
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
        const bool detailed_diagnostics =
            env_flag_enabled("POSEIDON_BOOTSTRAP_DETAILED_DIAGNOSTICS");
        const std::size_t evalmod_stage_profile_iterations =
            env_size_or(
                "POSEIDON_BOOTSTRAP_STAGE_PROFILE_ITERATIONS",
                3);
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
        if (evalmod_stage_profile_iterations == 0)
        {
            throw std::invalid_argument(
                "POSEIDON_BOOTSTRAP_STAGE_PROFILE_ITERATIONS must be nonzero");
        }
        if (evalmod_rescale_count == 0)
        {
            throw std::invalid_argument(
                "POSEIDON_BOOTSTRAP_EVALMOD_RESCALE_COUNT must be nonzero");
        }
        if (evalmod_log_scale >= 63)
        {
            throw std::invalid_argument(
                "POSEIDON_BOOTSTRAP_EVALMOD_LOG_SCALE must be less than 63");
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
        std::cout << "evalmod_rescale = " << evalmod_rescale_count
                  << " physical prime(s) per logical step\n";
        std::cout << "double_angle    = " << evalmod_double_angle << "\n";
        std::cout << "evalmod_k       = " << evalmod_k << "\n";
        std::cout << "tolerance       = " << correctness_tolerance << "\n";
        std::cout << "c2s_scaling     = " << c2s_scaling << "\n";
        std::cout << "c2s_bsgs_ratio  = " << c2s_log_bsgs_ratio << "\n";
        std::cout << "c2s_step        = " << c2s_step << "\n";
        std::cout << "c2s_rescale     = " << c2s_step
                  << " physical prime(s) per matrix\n";
        std::cout << "s2c_scaling     = " << s2c_scaling << "\n";
        std::cout << "s2c_bsgs_ratio  = " << s2c_log_bsgs_ratio << "\n";
        std::cout << "s2c_step        = " << s2c_step << "\n";
        std::cout << "s2c_rescale     = " << s2c_step
                  << " physical prime(s) per matrix\n";
        std::cout << "iterations      = " << iterations << "\n";
        std::cout << "full_iterations = " << full_iterations << "\n";
        std::cout << "full_warmup     = " << full_warmup << "\n";
        std::cout << "stage_profile   = "
                  << evalmod_stage_profile_iterations << " iteration(s)\n";
        std::cout << "diagnostics     = "
                  << (detailed_diagnostics ? "detailed" : "summary") << "\n";

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

        if (detailed_diagnostics)
        {
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
        }

        if (!comparison.equal)
        {
            std::cerr << "[FAILED] ModRaise CPU/GPU raw RNS mismatch: "
                      << comparison.mismatch_count << " word(s)\n";
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
        }

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

        for (std::size_t i = 0; i < warmup; ++i)
        {
            gpu_evaluator.bootstrap_prepare_modraise_input(
                gpu_source,
                gpu_prepared,
                q0_parms_id,
                target_q0_scale);
            gpu_evaluator.raise_modulus(gpu_prepared, gpu_raised);
        }
        cudaDeviceSynchronize();

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

        // The correct dual-prime bootstrap path anchors ModRaise at the exact
        // q0 base product, not at the first physical prime.
        const double raised_scale = context.crt_context()->q0();
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

        poseidon::Bootstrapper cpu_bootstrapper(
            context,
            *cpu_evaluator,
            encoder,
            context.parameters_literal()->log_slots(),
            /*boundary_k=*/evalmod_k,
            source.scale(),
            context.parameters_literal()->scale(),
            {},
            evalmod_rescale_count,
            q0_level + 1,
            raised_scale);
        cpu_bootstrapper.generate_linear_coefficients();
        const double c2s_input_scale = raised_scale *
            (post_raise_multiplier > 1.0 ? post_raise_multiplier : 1.0);
        auto c2s_matrix_group =
            cpu_bootstrapper.create_coeff_to_slot_matrix_group(
                context.crt_context()->first_parms_id(),
                c2s_input_scale,
                c2s_log_bsgs_ratio);
        const std::size_t c2s_rescale_count =
            std::max(c2s_matrix_group.step(), std::uint32_t{1});
        const std::size_t c2s_consumed_q =
            c2s_matrix_group.data().size() * c2s_rescale_count;
        if (c2s_consumed_q >= gpu_raised.meta.q_count)
        {
            throw std::invalid_argument(
                "CoeffToSlot consumes the complete modulus chain");
        }

        const std::size_t c2s_output_q_count =
            gpu_raised.meta.q_count - c2s_consumed_q;
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

        auto c2s_setup_galois_keys = make_galois_keys_for_matrix_group(
            context,
            keygen,
            c2s_matrix_group);

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

        poseidon::Ciphertext cpu_c2s_real_raw;
        poseidon::Ciphertext cpu_c2s_imag_raw;
        cpu_coeff_to_slot_rescale(
            cpu_full_raised,
            c2s_matrix_group,
            cpu_c2s_real_raw,
            cpu_c2s_imag_raw,
            *cpu_evaluator,
            c2s_setup_galois_keys,
            encoder);

        if (cpu_c2s_real_raw.parms_id() != cpu_c2s_imag_raw.parms_id() ||
            std::abs(
                std::log2(cpu_c2s_real_raw.scale()) -
                std::log2(cpu_c2s_imag_raw.scale())) > 1.0e-9)
        {
            throw std::runtime_error(
                "CoeffToSlot real/imag branches cannot share one scale-alignment plan");
        }
        if (c2s_output_context->coeff_modulus().size() <= evalmod_rescale_count)
        {
            throw std::runtime_error(
                "CoeffToSlot output has insufficient levels for scale alignment");
        }
        long double c2s_alignment_modulus_product = 1.0L;
        const auto &c2s_output_moduli = c2s_output_context->coeff_modulus();
        for (std::uint32_t index = 0; index < evalmod_rescale_count; ++index)
        {
            c2s_alignment_modulus_product *= static_cast<long double>(
                c2s_output_moduli[c2s_output_moduli.size() - 1 - index].value());
        }
        const double c2s_alignment_plain_scale =
            evalmod_scale *
            static_cast<double>(c2s_alignment_modulus_product) /
            cpu_c2s_real_raw.scale();
        poseidon::Plaintext c2s_alignment_plain;
        encoder.encode(
            std::complex<double>(1.0, 0.0),
            c2s_output_parms_id,
            c2s_alignment_plain_scale,
            c2s_alignment_plain);

        poseidon::Ciphertext cpu_c2s_real;
        poseidon::Ciphertext cpu_c2s_imag;
        cpu_evaluator->multiply_plain(
            cpu_c2s_real_raw,
            c2s_alignment_plain,
            cpu_c2s_real);
        cpu_evaluator->multiply_plain(
            cpu_c2s_imag_raw,
            c2s_alignment_plain,
            cpu_c2s_imag);
        for (std::uint32_t index = 0; index < evalmod_rescale_count; ++index)
        {
            cpu_evaluator->rescale(cpu_c2s_real, cpu_c2s_real);
            cpu_evaluator->rescale(cpu_c2s_imag, cpu_c2s_imag);
        }
        cpu_c2s_real.scale() = evalmod_scale;
        cpu_c2s_imag.scale() = evalmod_scale;

        const auto evalmod_input_parms_id = cpu_c2s_real.parms_id();
        const auto evalmod_input_context =
            context.crt_context()->get_context_data(evalmod_input_parms_id);
        if (!evalmod_input_context)
        {
            throw std::runtime_error(
                "aligned CoeffToSlot output parms_id is absent from the context");
        }
        eval_mod_poly.set_level_start(
            static_cast<std::uint32_t>(evalmod_input_context->level()));

        const auto bootstrap_evalmod_polynomial =
            cpu_bootstrapper.eval_mod_polynomial();
        const double bootstrap_inverse_coefficient =
            cpu_bootstrapper.inverse_coefficient(evalmod_double_angle);
        const double bootstrap_double_angle_base = std::pow(
            bootstrap_inverse_coefficient,
            1.0 / static_cast<double>(std::uint64_t{1} << evalmod_double_angle));

        /*
         * Probe the CPU high-precision schedule once during untimed setup.
         * The GPU uploader records this observable output level while still
         * generating a fixed BSGS/ordinary-rescale GPU execution plan.
         */
        poseidon::Plaintext cpu_evalmod_probe_plain;
        encoder.encode(
            std::complex<double>(0.125, 0.0),
            evalmod_input_parms_id,
            evalmod_scale,
            cpu_evalmod_probe_plain);
        poseidon::Ciphertext cpu_evalmod_probe_input;
        encryptor.encrypt(cpu_evalmod_probe_plain, cpu_evalmod_probe_input);
        cpu_evalmod_probe_input.scale() = evalmod_scale;
        poseidon::Ciphertext cpu_evalmod_probe_output;
        poseidon::BootstrapEvalModTrace cpu_evalmod_probe_trace;
        cpu_bootstrapper.eval_mod(
            cpu_evalmod_probe_input,
            cpu_evalmod_probe_output,
            relin_keys,
            evalmod_double_angle,
            bootstrap_inverse_coefficient,
            evalmod_scale,
            &cpu_evalmod_probe_trace);
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

        const bool fused_leaf_ab_enabled =
            env_flag_enabled("POSEIDON_BOOTSTRAP_FUSED_LEAF_AB");
        auto gpu_evalmod_data =
            poseidon::gpu::GpuUploader::upload_eval_mod_high_precision(
                eval_mod_poly,
                encoder,
                evalmod_input_parms_id,
                device_id,
                &gpu_relin_keys,
                cpu_evalmod_output_parms_id,
                evalmod_rescale_count,
                &bootstrap_evalmod_polynomial,
                /*include_input_offset=*/false,
                evalmod_double_angle,
                bootstrap_double_angle_base,
                cpu_evalmod_probe_trace.polynomial_output.scale(),
                /*fuse_leaf_terms_before_rescale=*/true);
        poseidon::gpu::GpuBootstrapData legacy_leaf_evalmod_data;
        if (fused_leaf_ab_enabled)
        {
            legacy_leaf_evalmod_data.eval_mod =
                poseidon::gpu::GpuUploader::upload_eval_mod_high_precision(
                    eval_mod_poly,
                    encoder,
                    evalmod_input_parms_id,
                    device_id,
                    &gpu_relin_keys,
                    cpu_evalmod_output_parms_id,
                    evalmod_rescale_count,
                    &bootstrap_evalmod_polynomial,
                    /*include_input_offset=*/false,
                    evalmod_double_angle,
                    bootstrap_double_angle_base,
                    cpu_evalmod_probe_trace.polynomial_output.scale(),
                    /*fuse_leaf_terms_before_rescale=*/false);
        }
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

        // Fuse the exact q0 / 2^scaling_log correction into the inverse DFT
        // plaintexts. This is the same zero-runtime-cost normalization used
        // by the verified CPU Bootstrapper path.
        const double s2c_normalization = raised_scale / evalmod_scale;
        auto s2c_matrix_group =
            cpu_bootstrapper.create_slot_to_coeff_matrix_group(
                evalmod_output_parms_id,
                cpu_evalmod_probe_output.scale(),
                s2c_normalization,
                s2c_log_bsgs_ratio);
        auto full_galois_keys = make_galois_keys_for_matrix_groups(
            context,
            keygen,
            std::vector<const poseidon::LinearMatrixGroup *>{
                &c2s_matrix_group,
                &s2c_matrix_group});

        // The verified production Bootstrapper path is the correctness
        // oracle. The staged CPU/GPU paths below use the same cosine-heap
        // polynomial and dual-prime logical-rescale policy.
        poseidon::BootstrapConfig cpu_library_bootstrap_config;
        cpu_library_bootstrap_config.boundary_k = evalmod_k;
        cpu_library_bootstrap_config.log_message_ratio =
            evalmod_log_message_ratio;
        cpu_library_bootstrap_config.double_angle = evalmod_double_angle;
        cpu_library_bootstrap_config.scaling_log = evalmod_log_scale;
        cpu_library_bootstrap_config.output_ratio = bootstrap_ratio;
        cpu_library_bootstrap_config.project_real = false;
        cpu_library_bootstrap_config.logical_rescale_count =
            evalmod_rescale_count;
        cpu_library_bootstrap_config.q0_modulus_count = q0_level + 1;
        poseidon::GaloisKeys cpu_library_galois_keys;
        keygen.create_galois_keys(cpu_library_galois_keys);
        poseidon::Ciphertext cpu_library_bootstrap_result;
        cpu_evaluator->bootstrap(
            source,
            cpu_library_bootstrap_result,
            relin_keys,
            cpu_library_galois_keys,
            encoder,
            cpu_library_bootstrap_config);

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

        const auto configured_linear_transform_mode =
            poseidon::gpu::gpu_linear_transform_mode_from_environment(
                poseidon::gpu::GpuLinearTransformMode::ClassicBsgs);
        auto gpu_full_galois_keys =
            configured_linear_transform_mode ==
                    poseidon::gpu::GpuLinearTransformMode::DoubleHoistBsgs
                ? poseidon::gpu::GpuUploader::
                      upload_double_hoist_galois_keys(
                          full_galois_keys,
                          device_id)
                : poseidon::gpu::GpuUploader::upload_galois_keys(
                      full_galois_keys,
                      device_id);
        const auto c2s_key_q_counts = required_dft_key_q_counts(
            gpu_raised.meta.q_count,
            c2s_matrix_group.data().size(),
            c2s_matrix_group.step(),
            true);
        const auto s2c_key_q_counts = required_dft_key_q_counts(
            evalmod_output_q_count,
            s2c_matrix_group.data().size(),
            s2c_matrix_group.step(),
            false);
        const auto full_key_q_counts =
            merge_q_counts(c2s_key_q_counts, s2c_key_q_counts);
        poseidon::gpu::GpuUploader::prepare_key_views_for_q_counts(
            gpu_full_galois_keys,
            full_key_q_counts);

        poseidon::gpu::GpuBootstrapData bootstrap_data;
        bootstrap_data.linear_transform_mode =
            configured_linear_transform_mode;
        bootstrap_data.q0_parms_id = q0_parms_id;
        bootstrap_data.q0_over_message_ratio = target_q0_scale;
        bootstrap_data.raised_scale_override = raised_scale;
        // The cosine-heap EvalMod keeps its actual post-rescale scale. Do not
        // reinterpret the residues by overwriting metadata before S2C.
        bootstrap_data.slot_to_coeff_input_scale = 0.0;
        bootstrap_data.coeff_to_slot_scale_alignment_plaintext =
            poseidon::gpu::GpuUploader::upload_plaintext(
                c2s_alignment_plain,
                device_id);
        bootstrap_data.coeff_to_slot_scale_alignment_rescale_count =
            evalmod_rescale_count;
        bootstrap_data.coeff_to_slot_aligned_scale = evalmod_scale;
        bootstrap_data.project_real = false;
        bootstrap_data.output_ratio = bootstrap_ratio;
        const double s2c_output_scale =
            raised_scale * context.parameters_literal()->scale() /
            source.scale();
        bootstrap_data.slot_to_coeff_output_scale = s2c_output_scale;
        if (post_raise_multiplier > 1.0)
        {
            bootstrap_data.post_raise_integer_multiplier =
                static_cast<std::uint64_t>(post_raise_multiplier);
            bootstrap_data.post_raise_scale_multiplier =
                post_raise_multiplier;
        }
        if (bootstrap_data.linear_transform_mode ==
            poseidon::gpu::GpuLinearTransformMode::DoubleHoistBsgs)
        {
            bootstrap_data.coeff_to_slot_matrix_qp =
                poseidon::gpu::GpuUploader::upload_linear_matrix_group_qp(
                    c2s_matrix_group,
                    context,
                    device_id,
                    std::max(c2s_matrix_group.step(), std::uint32_t{1}));
            bootstrap_data.slot_to_coeff_matrix_qp =
                poseidon::gpu::GpuUploader::upload_linear_matrix_group_qp(
                    s2c_matrix_group,
                    context,
                    device_id,
                    std::max(s2c_matrix_group.step(), std::uint32_t{1}));
        }
        else
        {
            bootstrap_data.coeff_to_slot_matrix =
                poseidon::gpu::GpuUploader::upload_linear_matrix_group(
                    c2s_matrix_group,
                    device_id);
            bootstrap_data.slot_to_coeff_matrix =
                poseidon::gpu::GpuUploader::upload_linear_matrix_group(
                    s2c_matrix_group,
                    device_id);
        }
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
        std::cout << "linear transform = "
                  << (bootstrap_data.linear_transform_mode ==
                              poseidon::gpu::GpuLinearTransformMode::DoubleHoistBsgs
                          ? "double_hoist"
                          : "classic")
                  << "\n";
        std::cout << "Galois key views = "
                  << join_q_counts(full_key_q_counts) << "\n";
        std::cout << "polynomial degree= "
                  << bootstrap_evalmod_polynomial.degree() << "\n";
        std::cout << "basis steps       = "
                  << bootstrap_data.eval_mod.basis_steps.size() << "\n";
        std::cout << "BSGS leaf blocks  = "
                  << bootstrap_data.eval_mod.polynomial_blocks.size() << "\n";
        std::cout << "BSGS combines     = "
                  << bootstrap_data.eval_mod.polynomial_combine_steps.size() << "\n";
        std::cout << "leaf accumulation = fused MAC, one rescale per leaf\n";
        std::size_t leaf_nonconstant_terms = 0;
        for (const auto &block :
             bootstrap_data.eval_mod.polynomial_blocks)
        {
            for (const auto &term : block.terms)
            {
                leaf_nonconstant_terms += term.degree != 0 ? 1 : 0;
            }
        }
        std::cout << "leaf term rescales= "
                  << leaf_nonconstant_terms << " -> "
                  << bootstrap_data.eval_mod.polynomial_blocks.size()
                  << "\n";
        std::cout << "double-angle      = "
                  << bootstrap_data.eval_mod.double_angle_constants.size() << "\n";
        std::cout << "C2S rescale width = "
                  << c2s_matrix_group.step() << " prime(s)\n";
        std::cout << "EvalMod rescale   = "
                  << evalmod_rescale_count << " prime(s)\n";
        std::cout << "S2C rescale width = "
                  << s2c_matrix_group.step() << " prime(s)\n";
        std::cout << "EvalMod input q   = "
                  << cpu_c2s_real.coeff_modulus_size() << "\n";
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

        // Recompute the staged CPU C2S values with the exact same randomized
        // Galois keys used by the GPU upload. The earlier setup-only key set
        // is sufficient for deriving levels/scales but cannot be raw-RNS
        // compared with a transform evaluated under another key ciphertext.
        cpu_coeff_to_slot_rescale(
            cpu_full_raised,
            c2s_matrix_group,
            cpu_c2s_real_raw,
            cpu_c2s_imag_raw,
            *cpu_evaluator,
            full_galois_keys,
            encoder);
        cpu_evaluator->multiply_plain(
            cpu_c2s_real_raw,
            c2s_alignment_plain,
            cpu_c2s_real);
        cpu_evaluator->multiply_plain(
            cpu_c2s_imag_raw,
            c2s_alignment_plain,
            cpu_c2s_imag);
        for (std::uint32_t index = 0; index < evalmod_rescale_count; ++index)
        {
            cpu_evaluator->rescale(cpu_c2s_real, cpu_c2s_real);
            cpu_evaluator->rescale(cpu_c2s_imag, cpu_c2s_imag);
        }
        cpu_c2s_real.scale() = evalmod_scale;
        cpu_c2s_imag.scale() = evalmod_scale;

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

        poseidon::gpu::GpuBootstrapWorkspace bootstrap_workspace;
        const bool use_double_hoist =
            bootstrap_data.linear_transform_mode ==
            poseidon::gpu::GpuLinearTransformMode::DoubleHoistBsgs;
        if (use_double_hoist && detailed_diagnostics)
        {
            std::vector<DoubleHoistDiagnosticRow> diagnostic_rows;
            const auto &diagnostic_level =
                gpu_params.get_level(gpu_full_raised.meta.parms_id);
            const auto diagnostic_source =
                gpu_full_raised.make_const_view();
            if (diagnostic_source.polys.size() != 2 ||
                diagnostic_source.polys[0].shards.size() != 1 ||
                diagnostic_source.polys[1].shards.size() != 1)
            {
                throw std::runtime_error(
                    "double-hoist diagnostics require a one-shard size-2 input");
            }
            const auto &diagnostic_c0 =
                diagnostic_source.polys[0].shards.front();
            const auto &diagnostic_c1 =
                diagnostic_source.polys[1].shards.front();
            const poseidon::gpu::GpuParameterShard *diagnostic_parameter_shard =
                nullptr;
            for (const auto &candidate : diagnostic_level.shards)
            {
                if (candidate.device_id == diagnostic_c0.device_id &&
                    candidate.hybrid_base_q_count ==
                        gpu_full_raised.meta.q_count &&
                    candidate.hybrid_base_p_count != 0)
                {
                    diagnostic_parameter_shard = &candidate;
                    break;
                }
            }
            if (diagnostic_parameter_shard == nullptr)
            {
                throw std::runtime_error(
                    "double-hoist diagnostics cannot find the HYBRID parameter shard");
            }

            poseidon::gpu::GpuKeySwitchHandler diagnostic_keyswitch(
                gpu_params);
            poseidon::gpu::GpuHybridKeySwitchWorkspace
                diagnostic_keyswitch_workspace;

            /*
             * Invariant 1:
             * ModDownP(LiftP(ct)) must reproduce the input ciphertext exactly.
             * This isolates LiftP and the standalone delayed ModDown path from
             * decomposition, Galois permutation, and evaluation keys.
             */
            {
                poseidon::gpu::GpuQPCiphertextBuffer lifted;
                lifted.ensure_capacity(
                    diagnostic_c0.device_id,
                    gpu_full_raised.meta.degree,
                    gpu_full_raised.meta.q_count,
                    diagnostic_parameter_shard->hybrid_base_p_count,
                    1);
                poseidon::gpu::kernel::launch_double_hoist_lift_identity(
                    lifted.q_component(0, 0),
                    lifted.q_component(0, 1),
                    lifted.p_component(0, 0),
                    lifted.p_component(0, 1),
                    diagnostic_c0.ptr,
                    diagnostic_c1.ptr,
                    *diagnostic_parameter_shard,
                    gpu_full_raised.meta.degree,
                    false);

                poseidon::gpu::GpuCiphertextData roundtrip_gpu;
                diagnostic_keyswitch.moddown_qp_ciphertext_to_q(
                    lifted,
                    0,
                    roundtrip_gpu,
                    gpu_full_raised.meta,
                    diagnostic_level,
                    diagnostic_keyswitch_workspace);
                cudaDeviceSynchronize();

                const auto roundtrip_cpu = download_gpu_ciphertext(
                    roundtrip_gpu,
                    context);
                const auto raw = compare_ciphertexts(
                    cpu_full_raised,
                    roundtrip_cpu,
                    0);
                const auto approximate =
                    compare_decrypted_ciphertexts(
                        cpu_full_raised,
                        roundtrip_cpu,
                        decryptor,
                        encoder,
                        correctness_tolerance);
                diagnostic_rows.push_back(DoubleHoistDiagnosticRow{
                    "ModDown(LiftP(ct)) == ct",
                    "identity",
                    cpu_full_raised.coeff_modulus_size(),
                    roundtrip_cpu.coeff_modulus_size(),
                    approximate.max_abs_error,
                    raw.equal,
                    raw.equal && approximate.equal});
            }

            /*
             * Invariant 2:
             * Split one nonzero baby rotation into three GPU paths which all
             * use the exact same freshly generated one-step CPU Galois key:
             *
             *   a) existing monolithic standard-key rotation;
             *   b) standard-key Hoist/KeyMult/no-ModDown/ModDown;
             *   c) inverse-pre-rotated-key staged rotation.
             *
             * This avoids duplicating the complete bootstrap key set while
             * making key randomness identical across all four CPU/GPU
             * ciphertexts.
             */
            const auto &diagnostic_matrix =
                bootstrap_data.coeff_to_slot_matrix_qp.data().front();
            const auto nonzero_baby =
                std::find_if(
                    diagnostic_matrix.plan.baby_steps.begin(),
                    diagnostic_matrix.plan.baby_steps.end(),
                    [](int step) { return step != 0; });
            if (nonzero_baby ==
                diagnostic_matrix.plan.baby_steps.end())
            {
                throw std::runtime_error(
                    "double-hoist diagnostics require a nonzero baby rotation");
            }
            const int diagnostic_rotation_step = *nonzero_baby;
            {
                const std::uint32_t galois_elt =
                    diagnostic_galois_elt_from_rotation_step(
                        gpu_full_raised.meta.degree,
                        diagnostic_rotation_step);
                const std::size_t key_index =
                    diagnostic_galois_key_index(galois_elt);

                poseidon::GaloisKeys diagnostic_cpu_galois_keys;
                keygen.create_galois_keys(
                    std::vector<std::uint32_t>{galois_elt},
                    diagnostic_cpu_galois_keys);
                poseidon::Ciphertext cpu_rotated;
                cpu_evaluator->rotate(
                    cpu_full_raised,
                    cpu_rotated,
                    diagnostic_rotation_step,
                    diagnostic_cpu_galois_keys);

                auto diagnostic_standard_keys =
                    poseidon::gpu::GpuUploader::upload_galois_keys(
                        diagnostic_cpu_galois_keys,
                        device_id);
                auto diagnostic_pre_rotated_keys =
                    poseidon::gpu::GpuUploader::
                        upload_double_hoist_galois_keys(
                            diagnostic_cpu_galois_keys,
                            device_id);
                poseidon::gpu::GpuUploader::prepare_key_views_for_q_counts(
                    diagnostic_standard_keys,
                    std::vector<std::size_t>{
                        gpu_full_raised.meta.q_count});
                poseidon::gpu::GpuUploader::prepare_key_views_for_q_counts(
                    diagnostic_pre_rotated_keys,
                    std::vector<std::size_t>{
                        gpu_full_raised.meta.q_count});

                auto append_rotation_result =
                    [&](const std::string &invariant,
                        const poseidon::gpu::GpuCiphertextData &gpu_result)
                {
                    cudaDeviceSynchronize();
                    const auto downloaded = download_gpu_ciphertext(
                        gpu_result,
                        context);
                    const auto raw = compare_ciphertexts(
                        cpu_rotated,
                        downloaded,
                        0);
                    const auto approximate =
                        compare_decrypted_ciphertexts(
                            cpu_rotated,
                            downloaded,
                            decryptor,
                            encoder,
                            correctness_tolerance);
                    diagnostic_rows.push_back(DoubleHoistDiagnosticRow{
                        invariant,
                        "step=" +
                            std::to_string(diagnostic_rotation_step),
                        cpu_rotated.coeff_modulus_size(),
                        downloaded.coeff_modulus_size(),
                        approximate.max_abs_error,
                        raw.equal,
                        approximate.equal &&
                            cpu_rotated.parms_id() ==
                                downloaded.parms_id()});
                };

                poseidon::gpu::GpuCiphertextData monolithic_rotated_gpu;
                gpu_evaluator.rotate(
                    gpu_full_raised,
                    diagnostic_rotation_step,
                    diagnostic_standard_keys,
                    monolithic_rotated_gpu);
                append_rotation_result(
                    "standard monolithic rotate",
                    monolithic_rotated_gpu);

                poseidon::gpu::GpuHoistedDecomposition hoisted;
                diagnostic_keyswitch.hoist_decompose_modup_ntt(
                    diagnostic_source.polys[1],
                    diagnostic_level,
                    hoisted,
                    diagnostic_keyswitch_workspace);

                auto run_staged_rotation =
                    [&](const poseidon::gpu::GpuGaloisKeysData &keys,
                        const std::string &invariant)
                {
                    poseidon::gpu::GpuQPCiphertextBuffer lifted_rotation;
                    lifted_rotation.ensure_capacity(
                        diagnostic_c0.device_id,
                        gpu_full_raised.meta.degree,
                        gpu_full_raised.meta.q_count,
                        diagnostic_parameter_shard->hybrid_base_p_count,
                        1);
                    const auto key_view = keys.make_const_view(
                        gpu_full_raised.meta.q_count);
                    diagnostic_keyswitch.keyswitch_multsum_no_moddown(
                        hoisted,
                        galois_elt,
                        key_view,
                        keys,
                        key_index,
                        lifted_rotation,
                        0,
                        true,
                        diagnostic_level,
                        diagnostic_keyswitch_workspace);
                    poseidon::gpu::kernel::
                        launch_double_hoist_add_lifted_galois_c0(
                            lifted_rotation.q_component(0, 0),
                            diagnostic_c0.ptr,
                            galois_elt,
                            *diagnostic_parameter_shard,
                            gpu_full_raised.meta.degree);

                    poseidon::gpu::GpuCiphertextData staged_rotated_gpu;
                    diagnostic_keyswitch.moddown_qp_ciphertext_to_q(
                        lifted_rotation,
                        0,
                        staged_rotated_gpu,
                        gpu_full_raised.meta,
                        diagnostic_level,
                        diagnostic_keyswitch_workspace);
                    append_rotation_result(
                        invariant,
                        staged_rotated_gpu);
                };

                run_staged_rotation(
                    diagnostic_standard_keys,
                    "standard staged rotate");
                run_staged_rotation(
                    diagnostic_pre_rotated_keys,
                    "pre-rotated staged rotate");
            }

            /*
             * Invariant 3:
             * one complete double-hoisted BSGS matrix must agree with the
             * existing CPU classic-BSGS implementation before later C2S
             * matrices and conjugation can hide the first divergence.
             */
            {
                poseidon::Ciphertext cpu_matrix_result;
                cpu_multiply_by_diag_matrix_bsgs_rescale(
                    cpu_full_raised,
                    c2s_matrix_group.data().front(),
                    cpu_matrix_result,
                    *cpu_evaluator,
                    full_galois_keys,
                    std::max(
                        c2s_matrix_group.step(),
                        std::uint32_t{1}));

                poseidon::gpu::GpuDoubleHoistWorkspace
                    diagnostic_matrix_workspace;
                poseidon::gpu::GpuCiphertextData gpu_matrix_result;
                gpu_evaluator.multiply_by_diag_matrix_bsgs_double_hoist(
                    gpu_full_raised,
                    diagnostic_matrix,
                    gpu_full_galois_keys,
                    std::max(
                        c2s_matrix_group.step(),
                        std::uint32_t{1}),
                    diagnostic_matrix_workspace,
                    gpu_matrix_result);
                cudaDeviceSynchronize();

                const auto matrix_download = download_gpu_ciphertext(
                    gpu_matrix_result,
                    context);
                const auto raw = compare_ciphertexts(
                    cpu_matrix_result,
                    matrix_download,
                    0);
                const auto approximate =
                    compare_decrypted_ciphertexts(
                        cpu_matrix_result,
                        matrix_download,
                        decryptor,
                        encoder,
                        correctness_tolerance);
                diagnostic_rows.push_back(DoubleHoistDiagnosticRow{
                    "DoubleHoist(one matrix)",
                    "matrix=0",
                    cpu_matrix_result.coeff_modulus_size(),
                    matrix_download.coeff_modulus_size(),
                    approximate.max_abs_error,
                    raw.equal,
                    approximate.equal &&
                        cpu_matrix_result.parms_id() ==
                            matrix_download.parms_id()});
            }
            print_double_hoist_diagnostic_table(diagnostic_rows);
        }
        auto run_gpu_coeff_to_slot =
            [&](const poseidon::gpu::GpuCiphertextData &input,
                poseidon::gpu::GpuCiphertextData &real,
                poseidon::gpu::GpuCiphertextData &imag)
        {
            if (use_double_hoist)
            {
                gpu_evaluator.coeff_to_slot_double_hoist(
                    input,
                    bootstrap_data.coeff_to_slot_matrix_qp,
                    bootstrap_data.minus_i_plaintext,
                    gpu_full_galois_keys,
                    bootstrap_workspace.coeff_to_slot_double_hoist,
                    real,
                    imag);
            }
            else
            {
                gpu_evaluator.coeff_to_slot(
                    input,
                    bootstrap_data.coeff_to_slot_matrix,
                    bootstrap_data.minus_i_plaintext,
                    gpu_full_galois_keys,
                    real,
                    imag);
            }
        };
        auto run_gpu_slot_to_coeff =
            [&](const poseidon::gpu::GpuCiphertextData &real,
                const poseidon::gpu::GpuCiphertextData &imag,
                poseidon::gpu::GpuCiphertextData &output)
        {
            if (use_double_hoist)
            {
                gpu_evaluator.slot_to_coeff_double_hoist(
                    real,
                    imag,
                    bootstrap_data.slot_to_coeff_matrix_qp,
                    bootstrap_data.plus_i_plaintext,
                    gpu_full_galois_keys,
                    bootstrap_workspace.slot_to_coeff_double_hoist,
                    output);
            }
            else
            {
                gpu_evaluator.slot_to_coeff(
                    real,
                    imag,
                    bootstrap_data.slot_to_coeff_matrix,
                    bootstrap_data.plus_i_plaintext,
                    gpu_full_galois_keys,
                    output);
            }
        };
        auto &gpu_c2s_real = bootstrap_workspace.coeff_to_slot_real;
        auto &gpu_c2s_imag = bootstrap_workspace.coeff_to_slot_imag;
        run_gpu_coeff_to_slot(
            gpu_full_raised,
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
        bool c2s_correct = false;
        double c2s_max_error = 0.0;
        if (use_double_hoist)
        {
            const auto real_comparison = compare_decrypted_ciphertexts(
                cpu_c2s_real_raw,
                gpu_c2s_real_download,
                decryptor,
                encoder,
                correctness_tolerance);
            const auto imag_comparison = compare_decrypted_ciphertexts(
                cpu_c2s_imag_raw,
                gpu_c2s_imag_download,
                decryptor,
                encoder,
                correctness_tolerance);
            c2s_correct =
                real_comparison.equal && imag_comparison.equal;
            c2s_max_error = std::max(
                real_comparison.max_abs_error,
                imag_comparison.max_abs_error);
            if (detailed_diagnostics)
            {
                std::cout << "\n[CPU/GPU CoeffToSlot correctness]\n";
                std::cout << "real max error = "
                          << real_comparison.max_abs_error << "\n";
                std::cout << "imag max error = "
                          << imag_comparison.max_abs_error << "\n";
            }
        }
        else
        {
            const auto real_comparison = compare_ciphertexts(
                cpu_c2s_real_raw,
                gpu_c2s_real_download,
                8);
            const auto imag_comparison = compare_ciphertexts(
                cpu_c2s_imag_raw,
                gpu_c2s_imag_download,
                8);
            c2s_correct =
                real_comparison.equal && imag_comparison.equal;
            if (detailed_diagnostics)
            {
                std::cout << "\n[CPU/GPU CoeffToSlot correctness]\n";
                std::cout << "real raw_equal = "
                          << (real_comparison.equal ? "YES" : "NO") << "\n";
                std::cout << "imag raw_equal = "
                          << (imag_comparison.equal ? "YES" : "NO") << "\n";
            }
        }
        if (!c2s_correct)
        {
            std::cerr << "[FAILED] CoeffToSlot CPU/GPU comparison exceeded "
                      << "the configured tolerance\n";
            return EXIT_FAILURE;
        }
        if (use_double_hoist && detailed_diagnostics)
        {
            print_double_hoist_counts(
                "CoeffToSlot",
                bootstrap_workspace.coeff_to_slot_double_hoist);
        }

        gpu_evaluator.multiply_plain(
            gpu_c2s_real,
            bootstrap_data.coeff_to_slot_scale_alignment_plaintext,
            bootstrap_workspace.scratch0);
        gpu_evaluator.multiply_plain(
            gpu_c2s_imag,
            bootstrap_data.coeff_to_slot_scale_alignment_plaintext,
            bootstrap_workspace.scratch1);
        if (env_flag_enabled("POSEIDON_BOOTSTRAP_RESCALE_X2_AB"))
        {
            const char *original_raw = std::getenv("POSEIDON_RESCALE_X2");
            const bool had_original = original_raw != nullptr;
            const std::string original_value =
                had_original ? std::string(original_raw) : std::string();
            const auto set_rescale_x2_mode = [](bool enabled) {
                if (setenv(
                        "POSEIDON_RESCALE_X2",
                        enabled ? "1" : "0",
                        1) != 0)
                {
                    throw std::runtime_error(
                        "failed to set rescale_x2 mode");
                }
            };

            poseidon::gpu::GpuCiphertextData legacy_rescaled;
            poseidon::gpu::GpuCiphertextData x2_rescaled;
            set_rescale_x2_mode(false);
            gpu_evaluator.rescale_many(
                bootstrap_workspace.scratch0,
                legacy_rescaled,
                2);
            set_rescale_x2_mode(true);
            gpu_evaluator.rescale_many(
                bootstrap_workspace.scratch0,
                x2_rescaled,
                2);
            cudaDeviceSynchronize();

            const auto legacy_download = download_gpu_ciphertext(
                legacy_rescaled,
                context);
            const auto x2_download = download_gpu_ciphertext(
                x2_rescaled,
                context);
            const auto comparison = compare_ciphertexts(
                legacy_download,
                x2_download,
                8);
            const bool metadata_equal =
                legacy_rescaled.meta.parms_id == x2_rescaled.meta.parms_id &&
                legacy_rescaled.meta.q_count == x2_rescaled.meta.q_count &&
                legacy_rescaled.meta.scale == x2_rescaled.meta.scale;
            std::cout << "\n[rescale_x2 exact equivalence]\n";
            std::cout << "raw_equal      = "
                      << (comparison.equal ? "YES" : "NO") << "\n";
            std::cout << "metadata_equal = "
                      << (metadata_equal ? "YES" : "NO") << "\n";
            if (!comparison.equal || !metadata_equal)
            {
                throw std::runtime_error(
                    "rescale_x2 is not exactly equivalent to two ordinary rescales");
            }

            if (had_original)
            {
                (void)setenv(
                    "POSEIDON_RESCALE_X2",
                    original_value.c_str(),
                    1);
            }
            else
            {
                (void)unsetenv("POSEIDON_RESCALE_X2");
            }
        }
        gpu_evaluator.rescale_many(
            bootstrap_workspace.scratch0,
            gpu_c2s_real,
            evalmod_rescale_count);
        gpu_evaluator.rescale_many(
            bootstrap_workspace.scratch1,
            gpu_c2s_imag,
            evalmod_rescale_count);
        gpu_c2s_real.meta.scale = evalmod_scale;
        gpu_c2s_imag.meta.scale = evalmod_scale;

        poseidon::Ciphertext cpu_eval_real;
        poseidon::Ciphertext cpu_eval_imag;
        poseidon::BootstrapEvalModTrace cpu_eval_real_trace;
        cpu_bootstrapper.eval_mod(
            cpu_c2s_real,
            cpu_eval_real,
            relin_keys,
            evalmod_double_angle,
            bootstrap_inverse_coefficient,
            evalmod_scale,
            detailed_diagnostics ? &cpu_eval_real_trace : nullptr);
        cpu_bootstrapper.eval_mod(
            cpu_c2s_imag,
            cpu_eval_imag,
            relin_keys,
            evalmod_double_angle,
            bootstrap_inverse_coefficient,
            evalmod_scale);
        if (cpu_eval_real.parms_id() != cpu_evalmod_output_parms_id ||
            cpu_eval_imag.parms_id() != cpu_evalmod_output_parms_id)
        {
            throw std::runtime_error(
                "CPU EvalMod output level changed between setup probe and correctness execution");
        }

        auto &gpu_eval_real = bootstrap_workspace.eval_mod_real;
        auto &gpu_eval_imag = bootstrap_workspace.eval_mod_imag;
        bootstrap_workspace.capture_eval_mod_trace = detailed_diagnostics;
        gpu_evaluator.eval_mod_high_precision(
            gpu_c2s_real,
            bootstrap_data,
            gpu_relin_keys,
            bootstrap_workspace,
            gpu_eval_real);
        cudaDeviceSynchronize();
        poseidon::gpu::GpuCiphertextData gpu_eval_real_stable;
        gpu_evaluator.multiply_scalar(
            gpu_eval_real,
            1,
            gpu_eval_real_stable);
        gpu_eval_real_stable.meta.scale = gpu_eval_real.meta.scale;

        poseidon::Ciphertext gpu_trace_offset_input;
        poseidon::Ciphertext gpu_trace_polynomial_output;
        std::vector<poseidon::Ciphertext> gpu_trace_double_angle_outputs;
        if (detailed_diagnostics)
        {
            gpu_trace_offset_input = download_gpu_ciphertext(
                bootstrap_workspace.eval_mod_trace_offset_input,
                context);
            gpu_trace_polynomial_output = download_gpu_ciphertext(
                bootstrap_workspace.eval_mod_trace_polynomial_output,
                context);
            gpu_trace_double_angle_outputs.reserve(
                bootstrap_workspace.eval_mod_trace_double_angle_outputs.size());
            for (const auto &gpu_trace_ciphertext :
                 bootstrap_workspace.eval_mod_trace_double_angle_outputs)
            {
                gpu_trace_double_angle_outputs.push_back(
                    download_gpu_ciphertext(gpu_trace_ciphertext, context));
            }
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
            gpu_eval_real_stable,
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

        if (fused_leaf_ab_enabled)
        {
            poseidon::gpu::GpuCiphertextData legacy_leaf_real;
            poseidon::gpu::GpuCiphertextData legacy_leaf_imag;
            gpu_evaluator.eval_mod_high_precision(
                gpu_c2s_real,
                legacy_leaf_evalmod_data,
                gpu_relin_keys,
                bootstrap_workspace,
                legacy_leaf_real);
            gpu_evaluator.eval_mod_high_precision(
                gpu_c2s_imag,
                legacy_leaf_evalmod_data,
                gpu_relin_keys,
                bootstrap_workspace,
                legacy_leaf_imag);
            cudaDeviceSynchronize();
            const auto legacy_leaf_real_download =
                download_gpu_ciphertext(legacy_leaf_real, context);
            const auto legacy_leaf_imag_download =
                download_gpu_ciphertext(legacy_leaf_imag, context);
            const auto leaf_real_comparison =
                compare_decrypted_ciphertexts(
                    legacy_leaf_real_download,
                    gpu_eval_real_download,
                    decryptor,
                    encoder,
                    correctness_tolerance);
            const auto leaf_imag_comparison =
                compare_decrypted_ciphertexts(
                    legacy_leaf_imag_download,
                    gpu_eval_imag_download,
                    decryptor,
                    encoder,
                    correctness_tolerance);
            std::cout << "\n[EvalMod fused leaf correctness]\n";
            std::cout << "legacy/fused real max error = "
                      << leaf_real_comparison.max_abs_error << "\n";
            std::cout << "legacy/fused imag max error = "
                      << leaf_imag_comparison.max_abs_error << "\n";
            std::cout << "within tolerance             = "
                      << (leaf_real_comparison.equal &&
                                  leaf_imag_comparison.equal
                              ? "YES"
                              : "NO")
                      << "\n";
            if (!leaf_real_comparison.equal ||
                !leaf_imag_comparison.equal)
            {
                throw std::runtime_error(
                    "fused EvalMod leaf accumulation exceeds correctness tolerance");
            }
        }

        if (detailed_diagnostics)
        {
            std::vector<EvalModTraceRow> eval_mod_trace_rows;
            eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                "input",
                cpu_eval_real_trace.input,
                gpu_trace_offset_input,
                decryptor,
                encoder,
                correctness_tolerance));
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
            std::cout << "trace counts: double-angle CPU/GPU="
                      << cpu_eval_real_trace.double_angle_outputs.size() << "/"
                      << gpu_trace_double_angle_outputs.size() << "\n";
        }

        poseidon::Ciphertext cpu_s2c_result;
        cpu_slot_to_coeff_rescale(
            cpu_eval_real,
            cpu_eval_imag,
            s2c_matrix_group,
            cpu_s2c_result,
            *cpu_evaluator,
            full_galois_keys,
            encoder);
        cpu_s2c_result.scale() = s2c_output_scale;
        poseidon::gpu::GpuCiphertextData gpu_s2c_result;
        run_gpu_slot_to_coeff(
            gpu_eval_real_stable,
            gpu_eval_imag,
            gpu_s2c_result);
        gpu_s2c_result.meta.scale = s2c_output_scale;
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
        if (use_double_hoist && detailed_diagnostics)
        {
            print_double_hoist_counts(
                "SlotToCoeff",
                bootstrap_workspace.slot_to_coeff_double_hoist);
        }

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
            cpu_evaluator->multiply_plain(real, c2s_alignment_plain, real);
            cpu_evaluator->multiply_plain(imag, c2s_alignment_plain, imag);
            for (std::uint32_t index = 0;
                 index < evalmod_rescale_count;
                 ++index)
            {
                cpu_evaluator->rescale(real, real);
                cpu_evaluator->rescale(imag, imag);
            }
            real.scale() = evalmod_scale;
            imag.scale() = evalmod_scale;
            cpu_bootstrapper.eval_mod(
                real,
                eval_real,
                relin_keys,
                evalmod_double_angle,
                bootstrap_inverse_coefficient,
                evalmod_scale);
            cpu_bootstrapper.eval_mod(
                imag,
                eval_imag,
                relin_keys,
                evalmod_double_angle,
                bootstrap_inverse_coefficient,
                evalmod_scale);
            cpu_slot_to_coeff_rescale(
                eval_real,
                eval_imag,
                s2c_matrix_group,
                result,
                *cpu_evaluator,
                full_galois_keys,
                encoder);
            result.scale() = s2c_output_scale;
            cpu_evaluator->multiply_const_direct(
                result,
                static_cast<int>(bootstrap_ratio),
                result,
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

        if (detailed_diagnostics)
        {
            std::cout << "\n[High-precision correctness details]\n";
            std::cout << std::scientific << std::setprecision(12);
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
            std::cout << std::defaultfloat << std::setprecision(12);
        }

        print_correctness_summary(
            std::vector<CorrectnessRow>{
                CorrectnessRow{
                    "ModRaise",
                    comparison.equal,
                    0.0,
                    "CPU/GPU raw RNS"},
                CorrectnessRow{
                    "CoeffToSlot",
                    c2s_correct,
                    c2s_max_error,
                    use_double_hoist
                        ? "CPU/GPU decoded"
                        : "CPU/GPU raw RNS"},
                CorrectnessRow{
                    "EvalMod (real + imag)",
                    evalmod_correct,
                    std::max(
                        eval_real_comparison.max_abs_error,
                        eval_imag_comparison.max_abs_error),
                    "CPU/GPU decoded"},
                CorrectnessRow{
                    "SlotToCoeff",
                    s2c_correct,
                    s2c_comparison.max_abs_error,
                    "CPU/GPU decoded"},
                CorrectnessRow{
                    "Full bootstrap",
                    full_correct,
                    gpu_source_comparison.max_abs_error,
                    "GPU/source decoded"}});

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
        }

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

        for (std::size_t i = 0; i < warmup; ++i)
        {
            run_gpu_coeff_to_slot(
                gpu_full_raised,
                gpu_c2s_real,
                gpu_c2s_imag);
        }
        cudaDeviceSynchronize();

        const double gpu_c2s_ms =
            time_gpu_ms(iterations, [&]() {
                run_gpu_coeff_to_slot(
                    gpu_full_raised,
                    gpu_c2s_real,
                    gpu_c2s_imag);
            });

        // coeff_to_slot timing leaves the persistent GPU branches at the raw
        // DFT level. Restore the setup-time value-preserving alignment before
        // the independent EvalMod/S2C timing measurements.
        gpu_evaluator.multiply_plain(
            gpu_c2s_real,
            bootstrap_data.coeff_to_slot_scale_alignment_plaintext,
            bootstrap_workspace.scratch0);
        gpu_evaluator.multiply_plain(
            gpu_c2s_imag,
            bootstrap_data.coeff_to_slot_scale_alignment_plaintext,
            bootstrap_workspace.scratch1);
        gpu_evaluator.rescale_many(
            bootstrap_workspace.scratch0,
            gpu_c2s_real,
            evalmod_rescale_count);
        gpu_evaluator.rescale_many(
            bootstrap_workspace.scratch1,
            gpu_c2s_imag,
            evalmod_rescale_count);
        gpu_c2s_real.meta.scale = evalmod_scale;
        gpu_c2s_imag.meta.scale = evalmod_scale;

        for (std::size_t i = 0; i < full_warmup; ++i)
        {
            poseidon::Ciphertext cpu_real;
            poseidon::Ciphertext cpu_imag;
            cpu_bootstrapper.eval_mod(
                cpu_c2s_real,
                cpu_real,
                relin_keys,
                evalmod_double_angle,
                bootstrap_inverse_coefficient,
                evalmod_scale);
            cpu_bootstrapper.eval_mod(
                cpu_c2s_imag,
                cpu_imag,
                relin_keys,
                evalmod_double_angle,
                bootstrap_inverse_coefficient,
                evalmod_scale);
        }

        const double cpu_evalmod_ms =
            time_cpu_ms(full_iterations, [&]() {
                poseidon::Ciphertext real;
                poseidon::Ciphertext imag;
                cpu_bootstrapper.eval_mod(
                    cpu_c2s_real,
                    real,
                    relin_keys,
                    evalmod_double_angle,
                    bootstrap_inverse_coefficient,
                    evalmod_scale);
                cpu_bootstrapper.eval_mod(
                    cpu_c2s_imag,
                    imag,
                    relin_keys,
                    evalmod_double_angle,
                    bootstrap_inverse_coefficient,
                    evalmod_scale);
            });

        for (std::size_t i = 0; i < full_warmup; ++i)
        {
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
        }
        cudaDeviceSynchronize();

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

        poseidon::gpu::GpuBootstrapWorkspace::EvalModStageTiming
            evalmod_stage_timing_sum;
        double evalmod_stage_profile_wall_ms_sum = 0.0;
        bootstrap_workspace.capture_eval_mod_stage_timing = true;
        for (std::size_t iteration = 0;
             iteration < evalmod_stage_profile_iterations;
             ++iteration)
        {
            const auto profile_start = std::chrono::steady_clock::now();
            gpu_evaluator.eval_mod_high_precision(
                gpu_c2s_real,
                bootstrap_data,
                gpu_relin_keys,
                bootstrap_workspace,
                gpu_eval_real);
            accumulate_eval_mod_stage_timing(
                evalmod_stage_timing_sum,
                bootstrap_workspace.eval_mod_stage_timing);
            gpu_evaluator.eval_mod_high_precision(
                gpu_c2s_imag,
                bootstrap_data,
                gpu_relin_keys,
                bootstrap_workspace,
                gpu_eval_imag);
            accumulate_eval_mod_stage_timing(
                evalmod_stage_timing_sum,
                bootstrap_workspace.eval_mod_stage_timing);
            const auto profile_stop = std::chrono::steady_clock::now();
            evalmod_stage_profile_wall_ms_sum +=
                static_cast<double>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        profile_stop - profile_start)
                        .count()) /
                1000.0;
        }
        bootstrap_workspace.capture_eval_mod_stage_timing = false;

        bool fused_leaf_ab_ran = false;
        double legacy_leaf_evalmod_ab_ms = 0.0;
        double fused_leaf_evalmod_ab_ms = 0.0;
        if (fused_leaf_ab_enabled)
        {
            const std::size_t ab_rounds = env_size_or(
                "POSEIDON_BOOTSTRAP_FUSED_LEAF_AB_ROUNDS",
                10);
            const std::size_t ab_batch = env_size_or(
                "POSEIDON_BOOTSTRAP_FUSED_LEAF_AB_BATCH",
                2);
            if (ab_rounds == 0 || ab_batch == 0)
            {
                throw std::invalid_argument(
                    "fused leaf A/B rounds and batch must be positive");
            }
            const auto run_legacy_leaf_pair = [&]() {
                gpu_evaluator.eval_mod_high_precision(
                    gpu_c2s_real,
                    legacy_leaf_evalmod_data,
                    gpu_relin_keys,
                    bootstrap_workspace,
                    gpu_eval_real);
                gpu_evaluator.eval_mod_high_precision(
                    gpu_c2s_imag,
                    legacy_leaf_evalmod_data,
                    gpu_relin_keys,
                    bootstrap_workspace,
                    gpu_eval_imag);
            };
            const auto run_fused_leaf_pair = [&]() {
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
            };

            (void)time_gpu_ms(1, run_legacy_leaf_pair);
            (void)time_gpu_ms(1, run_fused_leaf_pair);
            for (std::size_t round = 0; round < ab_rounds; ++round)
            {
                const auto measure_legacy = [&]() {
                    legacy_leaf_evalmod_ab_ms +=
                        time_gpu_ms(ab_batch, run_legacy_leaf_pair);
                };
                const auto measure_fused = [&]() {
                    fused_leaf_evalmod_ab_ms +=
                        time_gpu_ms(ab_batch, run_fused_leaf_pair);
                };
                if ((round & 1U) == 0)
                {
                    measure_legacy();
                    measure_fused();
                }
                else
                {
                    measure_fused();
                    measure_legacy();
                }
            }
            legacy_leaf_evalmod_ab_ms /=
                static_cast<double>(ab_rounds);
            fused_leaf_evalmod_ab_ms /=
                static_cast<double>(ab_rounds);
            fused_leaf_ab_ran = true;
        }

        bool fused_multiply_ab_ran = false;
        double legacy_evalmod_ab_ms = 0.0;
        double fused_evalmod_ab_ms = 0.0;
        if (env_flag_enabled("POSEIDON_BOOTSTRAP_FUSED_MUL_AB"))
        {
            const std::size_t ab_rounds = env_size_or(
                "POSEIDON_BOOTSTRAP_FUSED_MUL_AB_ROUNDS",
                10);
            const std::size_t ab_batch = env_size_or(
                "POSEIDON_BOOTSTRAP_FUSED_MUL_AB_BATCH",
                2);
            if (ab_rounds == 0 || ab_batch == 0)
            {
                throw std::invalid_argument(
                    "fused multiply A/B rounds and batch must be positive");
            }

            const char *original_raw =
                std::getenv("POSEIDON_ELEMENTWISE_FUSED_CT_MUL");
            const bool had_original = original_raw != nullptr;
            const std::string original_value =
                had_original ? std::string(original_raw) : std::string();
            const auto set_fused_mode = [](bool enabled) {
                if (setenv(
                        "POSEIDON_ELEMENTWISE_FUSED_CT_MUL",
                        enabled ? "1" : "0",
                        1) != 0)
                {
                    throw std::runtime_error(
                        "failed to set fused ciphertext multiply mode");
                }
            };
            const auto run_evalmod_pair = [&]() {
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
            };

            set_fused_mode(false);
            (void)time_gpu_ms(1, run_evalmod_pair);
            set_fused_mode(true);
            (void)time_gpu_ms(1, run_evalmod_pair);

            for (std::size_t round = 0; round < ab_rounds; ++round)
            {
                const auto measure_legacy = [&]() {
                    set_fused_mode(false);
                    legacy_evalmod_ab_ms +=
                        time_gpu_ms(ab_batch, run_evalmod_pair);
                };
                const auto measure_fused = [&]() {
                    set_fused_mode(true);
                    fused_evalmod_ab_ms +=
                        time_gpu_ms(ab_batch, run_evalmod_pair);
                };
                if ((round & 1U) == 0)
                {
                    measure_legacy();
                    measure_fused();
                }
                else
                {
                    measure_fused();
                    measure_legacy();
                }
            }
            legacy_evalmod_ab_ms /= static_cast<double>(ab_rounds);
            fused_evalmod_ab_ms /= static_cast<double>(ab_rounds);
            fused_multiply_ab_ran = true;

            if (had_original)
            {
                (void)setenv(
                    "POSEIDON_ELEMENTWISE_FUSED_CT_MUL",
                    original_value.c_str(),
                    1);
            }
            else
            {
                (void)unsetenv("POSEIDON_ELEMENTWISE_FUSED_CT_MUL");
            }
        }

        bool rescale_x2_ab_ran = false;
        double legacy_rescale_evalmod_ab_ms = 0.0;
        double x2_evalmod_ab_ms = 0.0;
        if (env_flag_enabled("POSEIDON_BOOTSTRAP_RESCALE_X2_AB"))
        {
            const std::size_t ab_rounds = env_size_or(
                "POSEIDON_BOOTSTRAP_RESCALE_X2_AB_ROUNDS",
                10);
            const std::size_t ab_batch = env_size_or(
                "POSEIDON_BOOTSTRAP_RESCALE_X2_AB_BATCH",
                2);
            if (ab_rounds == 0 || ab_batch == 0)
            {
                throw std::invalid_argument(
                    "rescale_x2 A/B rounds and batch must be positive");
            }

            const char *original_raw = std::getenv("POSEIDON_RESCALE_X2");
            const bool had_original = original_raw != nullptr;
            const std::string original_value =
                had_original ? std::string(original_raw) : std::string();
            const auto set_rescale_x2_mode = [](bool enabled) {
                if (setenv(
                        "POSEIDON_RESCALE_X2",
                        enabled ? "1" : "0",
                        1) != 0)
                {
                    throw std::runtime_error(
                        "failed to set rescale_x2 mode");
                }
            };
            const auto run_evalmod_pair = [&]() {
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
            };

            set_rescale_x2_mode(false);
            (void)time_gpu_ms(1, run_evalmod_pair);
            set_rescale_x2_mode(true);
            (void)time_gpu_ms(1, run_evalmod_pair);
            for (std::size_t round = 0; round < ab_rounds; ++round)
            {
                const auto measure_legacy = [&]() {
                    set_rescale_x2_mode(false);
                    legacy_rescale_evalmod_ab_ms +=
                        time_gpu_ms(ab_batch, run_evalmod_pair);
                };
                const auto measure_x2 = [&]() {
                    set_rescale_x2_mode(true);
                    x2_evalmod_ab_ms +=
                        time_gpu_ms(ab_batch, run_evalmod_pair);
                };
                if ((round & 1U) == 0)
                {
                    measure_legacy();
                    measure_x2();
                }
                else
                {
                    measure_x2();
                    measure_legacy();
                }
            }
            legacy_rescale_evalmod_ab_ms /=
                static_cast<double>(ab_rounds);
            x2_evalmod_ab_ms /= static_cast<double>(ab_rounds);
            rescale_x2_ab_ran = true;

            if (had_original)
            {
                (void)setenv(
                    "POSEIDON_RESCALE_X2",
                    original_value.c_str(),
                    1);
            }
            else
            {
                (void)unsetenv("POSEIDON_RESCALE_X2");
            }
        }

        for (std::size_t i = 0; i < full_warmup; ++i)
        {
            poseidon::Ciphertext result;
            cpu_slot_to_coeff_rescale(
                cpu_eval_real,
                cpu_eval_imag,
                s2c_matrix_group,
                result,
                *cpu_evaluator,
                full_galois_keys,
                encoder);
        }
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

        for (std::size_t i = 0; i < full_warmup; ++i)
        {
            run_gpu_slot_to_coeff(
                gpu_eval_real,
                gpu_eval_imag,
                gpu_s2c_result);
        }
        cudaDeviceSynchronize();

        const double gpu_s2c_ms =
            time_gpu_ms(full_iterations, [&]() {
                run_gpu_slot_to_coeff(
                    gpu_eval_real,
                    gpu_eval_imag,
                    gpu_s2c_result);
            });

        for (std::size_t i = 0; i < full_warmup; ++i)
        {
            cpu_full_result = run_cpu_full_bootstrap();
        }
        const double cpu_full_bootstrap_ms =
            time_cpu_ms(full_iterations, [&]() {
                cpu_full_result = run_cpu_full_bootstrap();
            });

        for (std::size_t i = 0; i < full_warmup; ++i)
        {
            gpu_evaluator.bootstrap(
                gpu_source,
                bootstrap_data,
                gpu_relin_keys,
                gpu_full_galois_keys,
                bootstrap_workspace,
                gpu_full_result);
        }
        cudaDeviceSynchronize();

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

        if (fused_multiply_ab_ran)
        {
            std::cout << "\n[EvalMod fused multiply/square A/B]\n";
            std::cout << "legacy GPU avg ms = "
                      << format_ms(legacy_evalmod_ab_ms) << "\n";
            std::cout << "fused GPU avg ms  = "
                      << format_ms(fused_evalmod_ab_ms) << "\n";
            std::cout << "legacy/fused      = "
                      << format_speedup(
                             legacy_evalmod_ab_ms,
                             fused_evalmod_ab_ms)
                      << "\n";
            std::cout << "measurement       = alternating legacy/fused "
                      << "inside one process\n";
        }
        if (fused_leaf_ab_ran)
        {
            std::cout << "\n[EvalMod fused leaf A/B]\n";
            std::cout << "per-term rescale GPU avg ms = "
                      << format_ms(legacy_leaf_evalmod_ab_ms) << "\n";
            std::cout << "fused leaf GPU avg ms       = "
                      << format_ms(fused_leaf_evalmod_ab_ms) << "\n";
            std::cout << "legacy/fused                = "
                      << format_speedup(
                             legacy_leaf_evalmod_ab_ms,
                             fused_leaf_evalmod_ab_ms)
                      << "\n";
            std::cout << "measurement                 = alternating "
                      << "per-term/fused leaf inside one process\n";
        }
        if (rescale_x2_ab_ran)
        {
            std::cout << "\n[EvalMod rescale_x2 A/B]\n";
            std::cout << "legacy GPU avg ms = "
                      << format_ms(legacy_rescale_evalmod_ab_ms) << "\n";
            std::cout << "x2 GPU avg ms     = "
                      << format_ms(x2_evalmod_ab_ms) << "\n";
            std::cout << "legacy/x2         = "
                      << format_speedup(
                             legacy_rescale_evalmod_ab_ms,
                             x2_evalmod_ab_ms)
                      << "\n";
            std::cout << "measurement       = alternating legacy/x2 "
                      << "inside one process\n";
        }

        print_eval_mod_stage_timing_table(
            evalmod_stage_timing_sum,
            evalmod_stage_profile_wall_ms_sum,
            evalmod_stage_profile_iterations);

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
            BootstrapTimingParameters{
                degree,
                q_count,
                p_count,
                log_q,
                log_p,
                log_scale,
                evalmod_log_scale,
                q0_level,
                bootstrap_ratio,
                evalmod_sine_degree,
                evalmod_rescale_count,
                evalmod_double_angle,
                c2s_step,
                s2c_step,
                use_double_hoist ? "double_hoist" : "classic"},
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
