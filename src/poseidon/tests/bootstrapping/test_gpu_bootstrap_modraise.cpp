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
#include "poseidon/gpu/gpu_scale_planner.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/gpu/kernels/gpu_double_hoist_kernels.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
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
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

bool env_flag_enabled_or(const char *name, bool fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return fallback;
    }
    const std::string text(value);
    return text != "0" &&
           text != "OFF" &&
           text != "off" &&
           text != "false" &&
           text != "FALSE";
}

std::optional<std::vector<std::uint32_t>> env_u32_list(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return std::nullopt;
    }

    std::vector<std::uint32_t> result;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        if (token.empty())
        {
            throw std::invalid_argument(
                std::string(name) + " contains an empty entry");
        }
        const auto parsed = std::stoul(token);
        if (parsed > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::invalid_argument(
                std::string(name) + " contains an out-of-range entry");
        }
        result.push_back(static_cast<std::uint32_t>(parsed));
    }
    if (result.empty())
    {
        throw std::invalid_argument(std::string(name) + " is empty");
    }
    return result;
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
    const std::vector<std::uint32_t> &log_q_chain,
    std::size_t p_count,
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
        log_q_chain,
        std::vector<std::uint32_t>(p_count, log_p));
    return parms;
}

std::vector<std::uint32_t> make_mixed_45_bootstrap_q_chain(
    std::size_t q_count,
    std::uint32_t fallback_log_q)
{
    if (q_count != 34)
    {
        return std::vector<std::uint32_t>(q_count, fallback_log_q);
    }

    // q[0..10] remain available after the fully verified GS-aligned path.
    // Rescaling from q[33] down currently follows C2S=4, EvalMod=15,
    // S2C=4. Setup always prints the concrete-prime schedule so a future
    // EvalMod DAG improvement can retune this chain without hard-coded drops.
    return {
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        28, 28, 31, 31, 32, 32, 30, 31, 32, 31, 32,
        32, 31, 31, 31, 32, 32, 31, 32, 32, 32, 30};
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
    std::uint32_t step,
    double input_scale = 0.0,
    double min_scale = 0.0,
    double value_normalization = 1.0,
    const std::vector<std::uint32_t> &layer_groups = {},
    std::uint32_t direct_layer_threshold = 0)
{
    const std::size_t matrix_depth =
        layer_groups.empty() ? 3 : layer_groups.size();
    poseidon::HomomorphicDFTMatrixLiteral matrix_literal(
        poseidon::encode,
        context.parameters_literal()->log_n(),
        context.parameters_literal()->log_slots(),
        static_cast<std::uint32_t>(context.parameters_literal()->q().size() - 1),
        std::vector<std::uint32_t>(matrix_depth, 1),
        /*repack_imag_to_real=*/true,
        scaling,
        /*bit_reversed=*/false,
        log_bsgs_ratio,
        layer_groups,
        direct_layer_threshold);

    poseidon::LinearMatrixGroup matrix_group;
    if (min_scale > 0.0)
    {
        matrix_literal.create_dynamic(
            matrix_group,
            encoder,
            input_scale,
            min_scale,
            min_scale,
            value_normalization);
    }
    else
    {
        matrix_literal.create(matrix_group, encoder, step);
    }
    return matrix_group;
}

poseidon::LinearMatrixGroup make_slot_to_coeff_matrix_group(
    const poseidon::PoseidonContext &context,
    poseidon::CKKSEncoder &encoder,
    std::uint32_t level_start,
    double scaling,
    std::uint32_t log_bsgs_ratio,
    std::uint32_t step,
    double input_scale = 0.0,
    double min_scale = 0.0,
    double value_normalization = 1.0)
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
    if (min_scale > 0.0)
    {
        matrix_literal.create_dynamic(
            matrix_group,
            encoder,
            input_scale,
            min_scale,
            min_scale,
            value_normalization);
    }
    else
    {
        matrix_literal.create(matrix_group, encoder, step);
    }
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

struct PlaintextCompressionStats
{
    std::size_t diagonal_count = 0;
    std::size_t full_q_words = 0;
    std::size_t compressed_q_words = 0;
    std::size_t full_qp_words = 0;
    std::size_t compressed_qp_words = 0;
    std::map<std::size_t, std::size_t> compression_histogram;
    bool exact_q_reconstruction = true;
    bool exact_qp_device_reconstruction = true;
};

bool plaintext_has_bit_reversed_period(
    const poseidon::Plaintext &plaintext,
    std::size_t degree,
    std::size_t period,
    const std::vector<std::uint32_t> &bit_reversed_indices)
{
    if (!plaintext.is_ntt_form() || degree == 0 || period == 0 ||
        period > degree || degree % period != 0 ||
        plaintext.coeff_count() % degree != 0 ||
        bit_reversed_indices.size() != degree)
    {
        return false;
    }

    const std::size_t q_count = plaintext.coeff_count() / degree;
    std::vector<std::uint64_t> representatives(period);
    std::vector<bool> initialized(period);
    const std::size_t mask = period - 1;
    for (std::size_t limb = 0; limb < q_count; ++limb)
    {
        std::fill(initialized.begin(), initialized.end(), false);
        const auto *values = plaintext.data() + limb * degree;
        for (std::size_t coefficient = 0;
             coefficient < degree;
             ++coefficient)
        {
            const std::size_t compact_index =
                bit_reversed_indices[coefficient] & mask;
            if (!initialized[compact_index])
            {
                representatives[compact_index] = values[coefficient];
                initialized[compact_index] = true;
            }
            else if (representatives[compact_index] != values[coefficient])
            {
                return false;
            }
        }
    }
    return true;
}

std::size_t plaintext_bit_reversed_period(
    const poseidon::Plaintext &plaintext,
    std::size_t degree,
    const std::vector<std::uint32_t> &bit_reversed_indices)
{
    for (std::size_t period = 1; period <= degree; period <<= 1)
    {
        if (plaintext_has_bit_reversed_period(
                plaintext,
                degree,
                period,
                bit_reversed_indices))
        {
            return period;
        }
    }
    return degree;
}

PlaintextCompressionStats analyze_plaintext_compression(
    const std::string &label,
    const poseidon::LinearMatrixGroup &matrix_group,
    std::size_t degree,
    std::size_t p_count)
{
    if (degree == 0 || (degree & (degree - 1)) != 0)
    {
        throw std::invalid_argument(
            "plaintext compression probe requires a power-of-two degree");
    }
    int log_degree = 0;
    for (std::size_t value = degree; value > 1; value >>= 1)
    {
        ++log_degree;
    }
    std::vector<std::uint32_t> bit_reversed_indices(degree);
    for (std::size_t coefficient = 0; coefficient < degree; ++coefficient)
    {
        bit_reversed_indices[coefficient] =
            poseidon::util::reverse_bits(
                static_cast<std::uint32_t>(coefficient),
                log_degree);
    }

    PlaintextCompressionStats total;
    std::cout << "\n[WHET plaintext compression probe: " << label << "]\n";
    for (std::size_t stage = 0;
         stage < matrix_group.data().size();
         ++stage)
    {
        PlaintextCompressionStats stage_stats;
        const auto &matrix = matrix_group.data()[stage];
        for (const auto &entry : matrix.plain_vec)
        {
            const auto &plaintext = entry.second;
            if (plaintext.coeff_count() % degree != 0)
            {
                throw std::runtime_error(
                    "plaintext compression probe found an invalid plaintext shape");
            }
            const std::size_t q_count =
                plaintext.coeff_count() / degree;
            const std::size_t period = plaintext_bit_reversed_period(
                plaintext,
                degree,
                bit_reversed_indices);
            const bool exact = plaintext_has_bit_reversed_period(
                plaintext,
                degree,
                period,
                bit_reversed_indices);
            const std::size_t compression = degree / period;

            ++stage_stats.diagonal_count;
            stage_stats.full_q_words += q_count * degree;
            stage_stats.compressed_q_words += q_count * period;
            stage_stats.full_qp_words += (q_count + p_count) * degree;
            stage_stats.compressed_qp_words +=
                (q_count + p_count) * period;
            ++stage_stats.compression_histogram[compression];
            stage_stats.exact_q_reconstruction &= exact;
        }

        total.diagonal_count += stage_stats.diagonal_count;
        total.full_q_words += stage_stats.full_q_words;
        total.compressed_q_words += stage_stats.compressed_q_words;
        total.full_qp_words += stage_stats.full_qp_words;
        total.compressed_qp_words += stage_stats.compressed_qp_words;
        total.exact_q_reconstruction &= stage_stats.exact_q_reconstruction;
        for (const auto &entry : stage_stats.compression_histogram)
        {
            total.compression_histogram[entry.first] += entry.second;
        }

        const double qp_ratio = stage_stats.compressed_qp_words == 0
            ? 1.0
            : static_cast<double>(stage_stats.full_qp_words) /
                  static_cast<double>(stage_stats.compressed_qp_words);
        std::cout << "stage=" << stage
                  << " q=" << (matrix.level + 1)
                  << " diagonals=" << stage_stats.diagonal_count
                  << " QP compression=" << qp_ratio << "x"
                  << " exact_Q="
                  << (stage_stats.exact_q_reconstruction ? "YES" : "NO")
                  << " factors=";
        bool first = true;
        for (const auto &entry : stage_stats.compression_histogram)
        {
            if (!first)
            {
                std::cout << ",";
            }
            first = false;
            std::cout << entry.first << "x:" << entry.second;
        }
        std::cout << "\n";
    }

    const double gpu_word_mib =
        static_cast<double>(sizeof(poseidon::gpu::GpuWord)) /
        (1024.0 * 1024.0);
    const double total_ratio = total.compressed_qp_words == 0
        ? 1.0
        : static_cast<double>(total.full_qp_words) /
              static_cast<double>(total.compressed_qp_words);
    std::cout << "total diagonals       = " << total.diagonal_count << "\n"
              << "full QP storage MiB   = "
              << total.full_qp_words * gpu_word_mib << "\n"
              << "compact QP storage MiB= "
              << total.compressed_qp_words * gpu_word_mib << "\n"
              << "aggregate compression = " << total_ratio << "x\n"
              << "exact Q reconstruction= "
              << (total.exact_q_reconstruction ? "YES" : "NO") << "\n";
    return total;
}

PlaintextCompressionStats analyze_uploaded_compressed_qp(
    const std::string &label,
    const poseidon::gpu::GpuLinearMatrixGroupQP &matrix_group)
{
    PlaintextCompressionStats total;
    for (const auto &matrix : matrix_group.data())
    {
        if (!matrix.plain_vec_qp.empty() ||
            !matrix.plan.compressed_plaintexts ||
            matrix.plan.terms.size() !=
                matrix.compressed_plain_vec_qp.size() ||
            matrix.plan.diagonal_periods.size() !=
                matrix.plan.terms.size())
        {
            throw std::runtime_error(
                "compressed QP upload produced an inconsistent matrix plan");
        }

        std::vector<std::uint32_t> device_periods(
            matrix.plan.diagonal_periods.size());
        matrix.plan.diagonal_periods.copy_to_host(
            device_periods.data(),
            device_periods.size());

        std::map<std::size_t, std::size_t> expected_period_counts;
        for (const auto &entry : matrix.compressed_plain_vec_qp)
        {
            const auto &plaintext = entry.second;
            if (plaintext.meta.degree == 0 || plaintext.period == 0 ||
                plaintext.meta.p_count == 0 ||
                plaintext.period > plaintext.meta.degree ||
                (plaintext.period & (plaintext.period - 1)) != 0)
            {
                throw std::runtime_error(
                    "compressed QP upload produced invalid plaintext metadata");
            }
            ++total.diagonal_count;
            total.full_q_words +=
                plaintext.meta.q_count * plaintext.meta.degree;
            total.compressed_q_words +=
                plaintext.meta.q_count * plaintext.period;
            total.full_qp_words += plaintext.full_word_count();
            total.compressed_qp_words += plaintext.compact_word_count();
            ++total.compression_histogram[
                plaintext.meta.degree / plaintext.period];
            ++expected_period_counts[plaintext.period];
            total.exact_qp_device_reconstruction &=
                plaintext.exact_device_reconstruction;
        }

        std::map<std::size_t, std::size_t> device_period_counts;
        for (const auto period : device_periods)
        {
            ++device_period_counts[period];
        }
        if (device_period_counts != expected_period_counts)
        {
            throw std::runtime_error(
                "compressed QP plan periods do not match device plaintexts");
        }
    }

    const double gpu_word_mib =
        static_cast<double>(sizeof(poseidon::gpu::GpuWord)) /
        (1024.0 * 1024.0);
    const double ratio = total.compressed_qp_words == 0
        ? 1.0
        : static_cast<double>(total.full_qp_words) /
              static_cast<double>(total.compressed_qp_words);
    std::cout << "\n[Exact compact QP device upload: " << label << "]\n"
              << "diagonals             = " << total.diagonal_count << "\n"
              << "full QP storage MiB   = "
              << total.full_qp_words * gpu_word_mib << "\n"
              << "compact QP storage MiB= "
              << total.compressed_qp_words * gpu_word_mib << "\n"
              << "aggregate compression = " << ratio << "x\n"
              << "exact device Q+P      = "
              << (total.exact_qp_device_reconstruction ? "YES" : "NO")
              << "\n";
    return total;
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
    std::uint32_t rescale_count,
    double rescale_min_scale = 0.0)
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

    if (rescale_min_scale > 0.0)
    {
        evaluator.rescale_dynamic(
            accumulator,
            result,
            rescale_min_scale);
        return;
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
    const poseidon::GaloisKeys &galois_keys,
    std::vector<poseidon::Ciphertext> *stage_trace = nullptr)
{
    if (matrix_group.data().empty())
    {
        throw std::invalid_argument("cpu_dft_rescale: empty matrix group");
    }
    if (stage_trace)
    {
        stage_trace->clear();
        stage_trace->reserve(matrix_group.data().size());
    }

    cpu_multiply_by_diag_matrix_bsgs_rescale(
        ciphertext,
        matrix_group.data().front(),
        result,
        evaluator,
        galois_keys,
        std::max(matrix_group.step(), std::uint32_t{1}),
        matrix_group.rescale_min_scale());
    if (stage_trace)
    {
        stage_trace->push_back(result);
    }

    for (std::size_t i = 1; i < matrix_group.data().size(); ++i)
    {
        poseidon::Ciphertext next;
        cpu_multiply_by_diag_matrix_bsgs_rescale(
            result,
            matrix_group.data()[i],
            next,
            evaluator,
            galois_keys,
            std::max(matrix_group.step(), std::uint32_t{1}),
            matrix_group.rescale_min_scale());
        result = std::move(next);
        if (stage_trace)
        {
            stage_trace->push_back(result);
        }
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

RawComparison compare_device_words_exact(
    const poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> &expected,
    const poseidon::gpu::DeviceVector<poseidon::gpu::GpuWord> &actual,
    std::size_t active_words,
    const std::string &label,
    std::size_t max_printed_mismatches = 8)
{
    RawComparison result;
    result.expected_words = active_words;
    result.actual_words = active_words;
    if (expected.size() < active_words || actual.size() < active_words)
    {
        result.actual_words = std::min(expected.size(), actual.size());
        return result;
    }

    constexpr std::size_t kChunkWords = 1U << 20;
    std::vector<poseidon::gpu::GpuWord> expected_host(kChunkWords);
    std::vector<poseidon::gpu::GpuWord> actual_host(kChunkWords);
    for (std::size_t offset = 0; offset < active_words;
         offset += kChunkWords)
    {
        const std::size_t count =
            std::min(kChunkWords, active_words - offset);
        const auto expected_status = cudaMemcpy(
            expected_host.data(),
            expected.data() + offset,
            count * sizeof(poseidon::gpu::GpuWord),
            cudaMemcpyDeviceToHost);
        const auto actual_status = cudaMemcpy(
            actual_host.data(),
            actual.data() + offset,
            count * sizeof(poseidon::gpu::GpuWord),
            cudaMemcpyDeviceToHost);
        if (expected_status != cudaSuccess || actual_status != cudaSuccess)
        {
            throw std::runtime_error(
                label + ": device QP comparison copy failed");
        }
        for (std::size_t index = 0; index < count; ++index)
        {
            if (expected_host[index] != actual_host[index])
            {
                if (result.mismatch_count < max_printed_mismatches)
                {
                    std::cout
                        << label << " mismatch[" << result.mismatch_count
                        << "] index=" << (offset + index)
                        << " full=" << expected_host[index]
                        << " compressed=" << actual_host[index] << "\n";
                }
                ++result.mismatch_count;
            }
        }
    }
    result.equal = result.mismatch_count == 0;
    return result;
}

void run_dynamic_scale_planner_tests(
    const poseidon::PoseidonContext &context,
    poseidon::EvaluatorCkksBase &cpu_evaluator,
    poseidon::gpu::GpuEvaluator &gpu_evaluator,
    const poseidon::Ciphertext &source,
    double minimum_scale,
    int device_id)
{
    const auto first_context = context.crt_context()->first_context_data();
    if (!first_context)
    {
        throw std::runtime_error("dynamic scale test has no first context level");
    }
    std::vector<std::uint64_t> active_moduli;
    active_moduli.reserve(first_context->coeff_modulus().size());
    for (const auto &modulus : first_context->coeff_modulus())
    {
        active_moduli.push_back(modulus.value());
    }

    struct ScaleCase
    {
        int input_log_scale;
        std::uint32_t expected_rescale_count;
    };
    const std::vector<ScaleCase> cases{
        {102, 1},
        {121, 2},
        {108, 1},
        {152, 3},
        // Exercise the fully generic exact mixed-radix path beyond the
        // three-prime bootstrap case as well.
        {185, 4},
        {217, 5},
        // These two cases distinguish the GS half-target floor from a hard
        // 2^51 floor: 2^82 can drop once to about 2^50, while 2^81 cannot.
        {82, 1},
        {81, 0},
        {60, 0},
    };

    const double minimum_output_scale = (minimum_scale + 1.0) / 2.0;
    std::cout << "\n[dynamic scale planner: GS min_scale/2 policy]\n";
    for (const auto &test : cases)
    {
        const double input_scale = std::exp2(test.input_log_scale);
        const auto plan = poseidon::gpu::plan_gpu_dynamic_rescale(
            input_scale,
            minimum_scale,
            active_moduli);
        if (plan.rescale_count != test.expected_rescale_count)
        {
            throw std::runtime_error(
                "dynamic scale planner returned unexpected rescale count for 2^" +
                std::to_string(test.input_log_scale));
        }
        const double output_log_scale = std::log2(plan.output_scale);
        if (output_log_scale < std::log2(minimum_output_scale) - 1.0e-9 ||
            output_log_scale >= std::log2(minimum_output_scale) + 32.0)
        {
            throw std::runtime_error(
                "dynamic scale planner output escaped the GS half-target interval");
        }

        poseidon::Ciphertext cpu_result = source;
        cpu_result.scale() = input_scale;
        for (std::uint32_t index = 0; index < plan.rescale_count; ++index)
        {
            cpu_evaluator.rescale(cpu_result, cpu_result);
        }

        poseidon::Ciphertext cpu_dynamic = source;
        cpu_dynamic.scale() = input_scale;
        cpu_evaluator.rescale_dynamic(
            cpu_dynamic,
            cpu_dynamic,
            minimum_scale);
        if (cpu_dynamic.coeff_modulus_size() != plan.output_q_count)
        {
            throw std::runtime_error(
                "GPU planner rescale count disagrees with CPU rescale_dynamic");
        }
        const double cpu_scale_error =
            std::abs(cpu_dynamic.scale() - plan.output_scale) /
            plan.output_scale;
        if (cpu_scale_error > 1.0e-12)
        {
            throw std::runtime_error(
                "GPU planner scale disagrees with CPU rescale_dynamic");
        }
        cpu_dynamic.scale() = cpu_result.scale();
        const auto cpu_raw = compare_ciphertexts(cpu_result, cpu_dynamic, 4);
        if (!cpu_raw.equal)
        {
            throw std::runtime_error(
                "CPU rescale_dynamic disagrees with its repeated-rescale reference");
        }

        // Upload the original level for the GPU test. CPU rescaling above is
        // deliberately performed on a separate ciphertext.
        poseidon::Ciphertext gpu_source_cpu = source;
        gpu_source_cpu.scale() = input_scale;
        auto gpu_input = poseidon::gpu::GpuUploader::upload_ciphertext(
            gpu_source_cpu,
            device_id);
        poseidon::gpu::GpuCiphertextData gpu_result;
        gpu_evaluator.rescale_dynamic(
            gpu_input,
            gpu_result,
            minimum_scale);
        poseidon::Ciphertext downloaded;
        poseidon::gpu::GpuUploader::download_ciphertext(
            gpu_result,
            downloaded,
            context);

        const double relative_scale_error =
            std::abs(downloaded.scale() - plan.output_scale) /
            plan.output_scale;
        if (relative_scale_error > 1.0e-12)
        {
            throw std::runtime_error(
                "dynamic GPU rescale metadata disagrees with its setup plan");
        }
        cpu_result.scale() = downloaded.scale();
        const auto raw = compare_ciphertexts(cpu_result, downloaded, 4);
        if (!raw.equal)
        {
            throw std::runtime_error(
                "dynamic GPU rescale disagrees with repeated CPU rescale for 2^" +
                std::to_string(test.input_log_scale));
        }

        if (plan.rescale_count == 3)
        {
            auto gpu_inplace = poseidon::gpu::GpuUploader::upload_ciphertext(
                gpu_source_cpu,
                device_id);
            gpu_evaluator.rescale_dynamic(
                gpu_inplace,
                gpu_inplace,
                minimum_scale);
            poseidon::Ciphertext inplace_downloaded;
            poseidon::gpu::GpuUploader::download_ciphertext(
                gpu_inplace,
                inplace_downloaded,
                context);
            cpu_result.scale() = inplace_downloaded.scale();
            const auto inplace_raw = compare_ciphertexts(
                cpu_result,
                inplace_downloaded,
                4);
            if (!inplace_raw.equal)
            {
                throw std::runtime_error(
                    "in-place generic GPU rescale disagrees with CPU reference");
            }
        }

        std::cout << "input=2^" << test.input_log_scale
                  << " drop=" << plan.rescale_count
                  << " output=2^" << std::fixed << std::setprecision(6)
                  << output_log_scale
                  << " q=" << active_moduli.size()
                  << "->" << plan.output_q_count
                  << " cpu_dynamic=YES raw_equal=YES\n";
        std::cout.unsetf(std::ios::floatfield);
    }

    bool rejected_required_rescale = false;
    try
    {
        (void)poseidon::gpu::plan_gpu_dynamic_rescale(
            std::exp2(60.0),
            minimum_scale,
            active_moduli,
            /*require_rescale=*/true);
    }
    catch (const std::invalid_argument &)
    {
        rejected_required_rescale = true;
    }
    if (!rejected_required_rescale)
    {
        throw std::runtime_error(
            "dynamic scale planner accepted an impossible required rescale");
    }
    std::cout << "required-rescale underflow rejected=YES\n";
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

std::complex<double> evaluate_chebyshev_series(
    const poseidon::Polynomial &polynomial,
    const std::complex<double> &input)
{
    const auto &coefficients = polynomial.data();
    if (coefficients.empty())
    {
        return {};
    }

    std::complex<double> value = coefficients[0];
    if (coefficients.size() == 1)
    {
        return value;
    }

    std::complex<double> previous = 1.0;
    std::complex<double> current = input;
    value += coefficients[1] * current;
    for (std::size_t degree = 2; degree < coefficients.size(); ++degree)
    {
        const auto next = 2.0 * input * current - previous;
        value += coefficients[degree] * next;
        previous = current;
        current = next;
    }
    return value;
}

std::complex<double> evaluate_evalmod_polynomial_plain(
    const poseidon::EvalModPoly &eval_mod_poly,
    const std::complex<double> &input)
{
    const double interval_width =
        eval_mod_poly.sine_poly_b() - eval_mod_poly.sine_poly_a();
    const double offset =
        -0.5 / (eval_mod_poly.sc_fac() * interval_width);
    auto value = evaluate_chebyshev_series(
        eval_mod_poly.sine_poly(),
        input + offset);

    double double_angle_constant = eval_mod_poly.sqrt_2pi();
    for (std::uint32_t index = 0;
         index < eval_mod_poly.double_angle();
         ++index)
    {
        double_angle_constant *= double_angle_constant;
        value = 2.0 * value * value - double_angle_constant;
    }
    return value;
}

std::complex<double> evaluate_evalmod_ideal_sine(
    const poseidon::EvalModPoly &eval_mod_poly,
    const std::complex<double> &input)
{
    const double pi = std::acos(-1.0);
    const double total_k = eval_mod_poly.k() * eval_mod_poly.sc_fac();
    return eval_mod_poly.q_diff() / (2.0 * pi) *
           std::sin(2.0 * pi * total_k * input);
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
    if (!std::isfinite(value))
    {
        return "SKIP";
    }
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << value;
    return stream.str();
}

std::string format_speedup(double cpu_ms, double gpu_ms)
{
    if (!std::isfinite(cpu_ms) || !std::isfinite(gpu_ms) ||
        !(gpu_ms > 0.0))
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
    bool include_post_dft_conjugation,
    const std::vector<std::uint32_t> &stage_rescale_counts = {})
{
    if ((!stage_rescale_counts.empty() &&
         stage_rescale_counts.size() != matrix_count) ||
        (stage_rescale_counts.empty() && rescale_count == 0))
    {
        throw std::invalid_argument("DFT rescale plan is inconsistent");
    }

    std::vector<std::size_t> q_counts;
    q_counts.reserve(matrix_count + (include_post_dft_conjugation ? 1 : 0));
    std::size_t current_q_count = input_q_count;
    for (std::size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index)
    {
        q_counts.push_back(current_q_count);
        const std::uint32_t stage_drop = stage_rescale_counts.empty()
            ? rescale_count
            : stage_rescale_counts[matrix_index];
        if (stage_drop == 0 || stage_drop >= current_q_count)
        {
            throw std::invalid_argument("DFT matrix count exceeds input q_count");
        }
        current_q_count -= stage_drop;
    }
    if (include_post_dft_conjugation)
    {
        q_counts.push_back(current_q_count);
    }

    std::sort(q_counts.begin(), q_counts.end());
    q_counts.erase(
        std::unique(q_counts.begin(), q_counts.end()),
        q_counts.end());
    return q_counts;
}

std::vector<std::uint32_t> dft_stage_rescale_counts(
    const poseidon::LinearMatrixGroup &matrix_group)
{
    if (!matrix_group.rescale_counts().empty())
    {
        return matrix_group.rescale_counts();
    }
    return std::vector<std::uint32_t>(
        matrix_group.data().size(),
        std::max(matrix_group.step(), std::uint32_t{1}));
}

double planned_dft_output_scale(
    const poseidon::PoseidonContext &context,
    double input_scale,
    const poseidon::LinearMatrixGroup &matrix_group)
{
    const auto first = context.crt_context()->first_context_data();
    if (!first)
    {
        throw std::invalid_argument("DFT scale plan has no context");
    }
    const auto counts = dft_stage_rescale_counts(matrix_group);
    std::size_t q_count =
        static_cast<std::size_t>(matrix_group.data().front().level) + 1;
    if (q_count == 0 || q_count > first->coeff_modulus().size())
    {
        throw std::invalid_argument("DFT scale plan has an invalid start level");
    }
    double scale = input_scale;
    for (std::size_t stage = 0; stage < matrix_group.data().size(); ++stage)
    {
        scale *= matrix_group.data()[stage].scale;
        for (std::uint32_t drop = 0; drop < counts[stage]; ++drop)
        {
            scale /= static_cast<double>(
                first->coeff_modulus().at(q_count - 1).value());
            --q_count;
        }
    }
    return scale;
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
    std::cout << "GPU-only timing mode: set POSEIDON_BOOTSTRAP_GPU_ONLY_TIMING=1 "
                 "to skip CPU timed benchmarks while keeping CPU setup/probes\n";
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

struct EvalModMultiplyTimingAggregate
{
    poseidon::gpu::GpuBootstrapWorkspace::EvalModMultiplyTiming metadata;
    double total_gpu_ms{0.0};
    std::size_t samples{0};
};

using EvalModMultiplyTimingMap =
    std::map<std::string, EvalModMultiplyTimingAggregate>;

void accumulate_eval_mod_multiply_timings(
    EvalModMultiplyTimingMap &aggregates,
    const std::vector<
        poseidon::gpu::GpuBootstrapWorkspace::EvalModMultiplyTiming> &samples)
{
    for (const auto &sample : samples)
    {
        auto &aggregate = aggregates[sample.label];
        if (aggregate.samples == 0)
        {
            aggregate.metadata = sample;
        }
        else if (aggregate.metadata.q_count != sample.q_count ||
                 aggregate.metadata.decomposition_count !=
                     sample.decomposition_count ||
                 aggregate.metadata.is_square != sample.is_square)
        {
            throw std::runtime_error(
                "EvalMod multiply timing metadata changed between samples");
        }
        aggregate.total_gpu_ms += sample.gpu_ms;
        ++aggregate.samples;
    }
}

void print_eval_mod_multiply_timing_table(
    const EvalModMultiplyTimingMap &aggregates,
    std::size_t profile_iterations)
{
    std::vector<EvalModMultiplyTimingAggregate> sorted;
    sorted.reserve(aggregates.size());
    for (const auto &entry : aggregates)
    {
        sorted.push_back(entry.second);
    }
    std::sort(
        sorted.begin(),
        sorted.end(),
        [](const auto &left, const auto &right) {
            return left.total_gpu_ms > right.total_gpu_ms;
        });

    constexpr int label_width = 37;
    constexpr int type_width = 8;
    constexpr int count_width = 6;
    constexpr int time_width = 14;
    const auto separator = [&]() {
        std::cout << "|-" << std::string(label_width, '-')
                  << "-|-" << std::string(type_width, '-')
                  << "-|-" << std::string(count_width, '-')
                  << "-|-" << std::string(count_width, '-')
                  << "-|-" << std::string(time_width, '-')
                  << "-|-" << std::string(time_width, '-')
                  << "-|\n";
    };

    std::cout << "\n[GPU EvalMod ciphertext multiply + relinearize timing]\n";
    separator();
    std::cout << "| " << std::left << std::setw(label_width) << "node"
              << " | " << std::setw(type_width) << "type"
              << " | " << std::right << std::setw(count_width) << "Q"
              << " | " << std::setw(count_width) << "dnum"
              << " | " << std::setw(time_width) << "avg ms/call"
              << " | " << std::setw(time_width) << "real+imag ms"
              << " |\n";
    separator();
    double pair_total_ms = 0.0;
    for (const auto &aggregate : sorted)
    {
        const double average = aggregate.samples == 0
            ? 0.0
            : aggregate.total_gpu_ms /
                  static_cast<double>(aggregate.samples);
        const double pair_time = profile_iterations == 0
            ? 0.0
            : aggregate.total_gpu_ms /
                  static_cast<double>(profile_iterations);
        pair_total_ms += pair_time;
        std::cout << "| " << std::left << std::setw(label_width)
                  << aggregate.metadata.label
                  << " | " << std::setw(type_width)
                  << (aggregate.metadata.is_square ? "square" : "multiply")
                  << " | " << std::right << std::setw(count_width)
                  << aggregate.metadata.q_count
                  << " | " << std::setw(count_width)
                  << aggregate.metadata.decomposition_count
                  << " | " << std::setw(time_width) << format_ms(average)
                  << " | " << std::setw(time_width) << format_ms(pair_time)
                  << " |\n";
    }
    separator();
    std::cout << "Multiply+Relin real+imag total = "
              << format_ms(pair_total_ms)
              << "; sorted by GPU contribution; CUDA events exclude host launch gaps\n";
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
            env_size_or("POSEIDON_BOOTSTRAP_LOG_Q", 32));
        const std::uint32_t log_p = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_LOG_P", log_q));
        const std::uint32_t log_scale = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_LOG_SCALE", 40));
        const std::uint32_t q0_level = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_Q0_LEVEL", 1));
        const std::uint32_t bootstrap_ratio = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_MESSAGE_RATIO", 32));
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
                45));
        const std::uint32_t evalmod_double_angle = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_EVALMOD_DOUBLE_ANGLE", 3));
        const std::uint32_t evalmod_k = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_EVALMOD_K", 16));
        const std::uint32_t evalmod_arcsine_degree = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_EVALMOD_ARCSINE_DEGREE", 0));
        const std::uint32_t evalmod_sine_degree = static_cast<std::uint32_t>(
            env_size_or("POSEIDON_BOOTSTRAP_EVALMOD_SINE_DEGREE", 30));
        const std::uint32_t evalmod_generation_degree =
            static_cast<std::uint32_t>(env_size_or(
                "POSEIDON_BOOTSTRAP_EVALMOD_GENERATION_DEGREE",
                evalmod_sine_degree));
        const std::optional<std::uint32_t> evalmod_truncate_degree =
            std::getenv("POSEIDON_BOOTSTRAP_EVALMOD_TRUNCATE_DEGREE")
                ? std::optional<std::uint32_t>(
                      static_cast<std::uint32_t>(env_size_or(
                          "POSEIDON_BOOTSTRAP_EVALMOD_TRUNCATE_DEGREE",
                          evalmod_sine_degree)))
                : std::nullopt;
        const bool evalmod_fixed_degree_refit =
            env_flag_enabled(
                "POSEIDON_BOOTSTRAP_EVALMOD_FIXED_DEGREE_REFIT");
        const double correctness_tolerance = env_double_or(
            "POSEIDON_BOOTSTRAP_CORRECTNESS_TOLERANCE", 1.0e-3);
        const bool detailed_diagnostics =
            env_flag_enabled("POSEIDON_BOOTSTRAP_DETAILED_DIAGNOSTICS");
        const bool skip_library_oracle =
            env_flag_enabled("POSEIDON_BOOTSTRAP_SKIP_LIBRARY_ORACLE");
        const bool gpu_only_timing =
            env_flag_enabled("POSEIDON_BOOTSTRAP_GPU_ONLY_TIMING");
        const bool nsys_capture_full =
            env_flag_enabled("POSEIDON_BOOTSTRAP_NSYS_CAPTURE_FULL");
        const bool ignore_correctness_failure =
            env_flag_enabled("POSEIDON_BOOTSTRAP_IGNORE_CORRECTNESS_FAILURE");
        const bool evalmod_dynamic_rescale =
            env_flag_enabled("POSEIDON_BOOTSTRAP_EVALMOD_DYNAMIC_RESCALE");
        const bool setup_only =
            env_flag_enabled("POSEIDON_BOOTSTRAP_SETUP_ONLY");
        const bool scale_planner_only =
            env_flag_enabled("POSEIDON_BOOTSTRAP_SCALE_PLANNER_ONLY");
        const bool slim_scale_chain_plan_only =
            env_flag_enabled(
                "POSEIDON_BOOTSTRAP_SLIM_SCALE_CHAIN_PLAN_ONLY");
        const bool slim_global_scale_chain_search =
            env_flag_enabled(
                "POSEIDON_BOOTSTRAP_SLIM_GLOBAL_SCALE_CHAIN_SEARCH");
        const bool c2s_only =
            env_flag_enabled("POSEIDON_BOOTSTRAP_C2S_ONLY");
        const bool evalmod_only =
            env_flag_enabled("POSEIDON_BOOTSTRAP_EVALMOD_ONLY");
        const bool slim_stc_first_probe =
            env_flag_enabled("POSEIDON_BOOTSTRAP_SLIM_STC_FIRST_PROBE");
        const bool slim_stc_modraise_probe =
            env_flag_enabled("POSEIDON_BOOTSTRAP_SLIM_STC_MODRAISE_PROBE");
        const bool slim_stc_c2s_probe =
            env_flag_enabled("POSEIDON_BOOTSTRAP_SLIM_STC_C2S_PROBE");
        const bool slim_stc_evalmod_probe =
            env_flag_enabled("POSEIDON_BOOTSTRAP_SLIM_STC_EVALMOD_PROBE");
        const bool slim_c2s_5433 =
            env_flag_enabled("POSEIDON_BOOTSTRAP_SLIM_C2S_5433");
        const bool plaintext_compression_probe =
            env_flag_enabled(
                "POSEIDON_BOOTSTRAP_PLAINTEXT_COMPRESSION_PROBE");
        const bool compressed_qp_mac_probe =
            env_flag_enabled(
                "POSEIDON_BOOTSTRAP_COMPRESSED_QP_MAC_PROBE");
        const bool slim_stc_run_c2s =
            slim_stc_c2s_probe || slim_stc_evalmod_probe ||
            plaintext_compression_probe || compressed_qp_mac_probe ||
            slim_c2s_5433;
        const bool slim_stc_run_modraise =
            slim_stc_modraise_probe || slim_stc_run_c2s;
        const std::size_t slim_stc_input_q_count =
            env_size_or("POSEIDON_BOOTSTRAP_SLIM_STC_INPUT_Q_COUNT", 6);
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

        const bool mixed_45_q_chain =
            env_size_or("POSEIDON_BOOTSTRAP_MIXED_45_Q_CHAIN", 1) != 0;
        auto log_q_chain = mixed_45_q_chain
            ? make_mixed_45_bootstrap_q_chain(q_count, log_q)
            : std::vector<std::uint32_t>(q_count, log_q);
        if (const auto override_chain =
                env_u32_list("POSEIDON_BOOTSTRAP_Q_BIT_CHAIN"))
        {
            if (override_chain->size() != q_count)
            {
                throw std::invalid_argument(
                    "POSEIDON_BOOTSTRAP_Q_BIT_CHAIN must contain exactly " +
                    std::to_string(q_count) + " comma-separated entries");
            }
            log_q_chain = *override_chain;
        }
        auto parms = make_test_parameters(
            degree,
            log_q_chain,
            p_count,
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
            evalmod_generation_degree);
        if (evalmod_fixed_degree_refit)
        {
            if (evalmod_truncate_degree.has_value())
            {
                throw std::invalid_argument(
                    "fixed-degree EvalMod refit and truncation are mutually exclusive");
            }
            eval_mod_poly.refit_discrete_cosine_fixed_degree(
                evalmod_sine_degree);
            if (eval_mod_poly.sine_poly().degree() != evalmod_sine_degree)
            {
                throw std::runtime_error(
                    "fixed-degree EvalMod refit produced an unexpected degree");
            }
        }
        if (evalmod_truncate_degree.has_value())
        {
            if (*evalmod_truncate_degree != evalmod_sine_degree)
            {
                throw std::invalid_argument(
                    "EvalMod truncate degree must match the requested sine degree");
            }
            eval_mod_poly.truncate_sine_polynomial(
                *evalmod_truncate_degree);
            if (eval_mod_poly.sine_poly().degree() !=
                *evalmod_truncate_degree)
            {
                throw std::runtime_error(
                    "EvalMod truncated polynomial has an unexpected effective degree");
            }
        }
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

        if (slim_global_scale_chain_search)
        {
            if (!evalmod_dynamic_rescale ||
                !env_flag_enabled("POSEIDON_BOOTSTRAP_SLIM_C2S_5433"))
            {
                throw std::invalid_argument(
                    "POSEIDON_BOOTSTRAP_SLIM_GLOBAL_SCALE_CHAIN_SEARCH "
                    "requires dynamic rescale and "
                    "POSEIDON_BOOTSTRAP_SLIM_C2S_5433=1");
            }

            struct ScaleChainCandidate
            {
                std::vector<std::uint32_t> bits;
                std::vector<std::uint32_t> c2s_drops;
                std::vector<double> c2s_stage_log_scales;
                std::size_t c2s_q_count = 0;
                double c2s_log_scale = 0.0;
                std::size_t eval_q_count = 0;
                double eval_log_scale = 0.0;
                double score = -std::numeric_limits<double>::infinity();
            };

            const std::size_t search_budget = env_size_or(
                "POSEIDON_BOOTSTRAP_SLIM_GLOBAL_SEARCH_BUDGET", 320);
            const std::uint32_t search_min_bits =
                static_cast<std::uint32_t>(env_size_or(
                    "POSEIDON_BOOTSTRAP_SLIM_GLOBAL_SEARCH_MIN_BITS", 20));
            const std::uint32_t search_max_bits =
                static_cast<std::uint32_t>(env_size_or(
                    "POSEIDON_BOOTSTRAP_SLIM_GLOBAL_SEARCH_MAX_BITS", 32));
            const std::size_t search_first_index = env_size_or(
                "POSEIDON_BOOTSTRAP_SLIM_GLOBAL_SEARCH_FIRST_INDEX", 13);
            if (search_budget == 0 || search_min_bits < 20 ||
                search_min_bits > search_max_bits || search_max_bits > 32 ||
                search_first_index >= q_count)
            {
                throw std::invalid_argument(
                    "invalid slim global scale-chain search bounds");
            }

            auto chain_key = [](const std::vector<std::uint32_t> &bits) {
                std::string key;
                key.reserve(bits.size() * 3);
                for (const auto bit : bits)
                {
                    key.append(std::to_string(bit));
                    key.push_back(',');
                }
                return key;
            };

            auto evaluate_candidate =
                [&](const std::vector<std::uint32_t> &candidate_bits)
                    -> std::optional<ScaleChainCandidate> {
                try
                {
                    auto candidate_parms = make_test_parameters(
                        degree,
                        candidate_bits,
                        p_count,
                        log_p,
                        log_scale,
                        q0_level);
                    auto candidate_context =
                        poseidon::PoseidonFactory::get_instance()
                            ->create_poseidon_context(candidate_parms);
                    poseidon::CKKSEncoder candidate_encoder(
                        candidate_context);
                    const auto first_context_data =
                        candidate_context.crt_context()
                            ->first_context_data();
                    if (!first_context_data ||
                        first_context_data->parms().q().size() != q_count)
                    {
                        return std::nullopt;
                    }

                    std::vector<std::uint64_t> active_moduli;
                    active_moduli.reserve(q_count);
                    for (const auto &modulus :
                         first_context_data->parms().q())
                    {
                        active_moduli.push_back(modulus.value());
                    }

                    ScaleChainCandidate candidate;
                    candidate.bits = candidate_bits;
                    candidate.c2s_q_count = q_count;
                    double c2s_scale = evalmod_scale;
                    for (std::size_t stage = 0; stage < 4; ++stage)
                    {
                        const auto plan =
                            poseidon::gpu::plan_gpu_dynamic_rescale(
                                c2s_scale * evalmod_scale,
                                evalmod_scale,
                                std::span<const std::uint64_t>(
                                    active_moduli.data(),
                                    candidate.c2s_q_count),
                                /*require_rescale=*/true);
                        c2s_scale = plan.output_scale;
                        candidate.c2s_q_count = plan.output_q_count;
                        candidate.c2s_drops.push_back(
                            plan.rescale_count);
                        candidate.c2s_stage_log_scales.push_back(
                            std::log2(plan.output_scale));
                    }
                    candidate.c2s_log_scale = std::log2(c2s_scale);

                    // This is the production scale interval for the reordered
                    // C2S path: target/2 through target times one remaining
                    // <=32-bit prime. Keep the explicit 44-bit lower guard
                    // used by the correctness tests and reject pathological
                    // chains before building the EvalMod metadata DAG.
                    for (const double stage_log_scale :
                         candidate.c2s_stage_log_scales)
                    {
                        if (stage_log_scale < 44.0 ||
                            stage_log_scale >= 77.0)
                        {
                            return std::nullopt;
                        }
                    }

                    const auto c2s_output_parms_id =
                        candidate_context.crt_context()
                            ->parms_id_map()
                            .at(static_cast<std::uint32_t>(
                                candidate.c2s_q_count - 1));
                    const auto c2s_output_context =
                        candidate_context.crt_context()->get_context_data(
                            c2s_output_parms_id);
                    if (!c2s_output_context)
                    {
                        return std::nullopt;
                    }
                    eval_mod_poly.set_level_start(
                        static_cast<std::uint32_t>(
                            c2s_output_context->level()));
                    const auto evalmod_plan =
                        poseidon::gpu::GpuUploader::
                            upload_eval_mod_high_precision(
                                eval_mod_poly,
                                candidate_encoder,
                                c2s_output_parms_id,
                                device_id,
                                nullptr,
                                poseidon::parms_id_zero,
                                evalmod_rescale_count,
                                &eval_mod_poly.sine_poly(),
                                /*include_input_offset=*/true,
                                std::numeric_limits<std::uint32_t>::max(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                /*fuse_leaf_terms_before_rescale=*/true,
                                c2s_scale,
                                /*metadata_only=*/true);
                    candidate.eval_q_count = evalmod_plan.output_q_count;
                    candidate.eval_log_scale =
                        std::log2(evalmod_plan.output_scale);
                    if (!std::isfinite(candidate.eval_log_scale) ||
                        candidate.eval_log_scale < 44.0 ||
                        candidate.eval_log_scale >= 77.0)
                    {
                        return std::nullopt;
                    }

                    // Remaining Q limbs dominate. Within the same q-count,
                    // prefer the native EvalMod target 2^45, then a C2S output
                    // near 2^45. A tiny penalty discourages gratuitously tiny
                    // physical primes when scale behavior is otherwise equal.
                    double small_prime_penalty = 0.0;
                    for (std::size_t index = search_first_index;
                         index < candidate.bits.size();
                         ++index)
                    {
                        small_prime_penalty +=
                            static_cast<double>(
                                search_max_bits - candidate.bits[index]);
                    }
                    candidate.score =
                        10000.0 *
                            static_cast<double>(candidate.eval_q_count) -
                        20.0 * std::abs(candidate.eval_log_scale - 45.0) -
                        0.5 * std::abs(candidate.c2s_log_scale - 45.0) -
                        0.01 * small_prime_penalty;
                    return candidate;
                }
                catch (const std::exception &)
                {
                    return std::nullopt;
                }
            };

            std::vector<ScaleChainCandidate> population;
            std::unordered_set<std::string> visited;
            std::mt19937 generator(static_cast<std::uint32_t>(env_size_or(
                "POSEIDON_BOOTSTRAP_SLIM_GLOBAL_SEARCH_SEED",
                0x5343414cU)));
            std::uniform_int_distribution<std::uint32_t> random_bit(
                search_min_bits, search_max_bits);
            std::uniform_int_distribution<std::size_t> random_index(
                search_first_index, q_count - 1);

            auto try_candidate =
                [&](const std::vector<std::uint32_t> &candidate_bits) {
                if (visited.size() >= search_budget ||
                    !visited.emplace(chain_key(candidate_bits)).second)
                {
                    return;
                }
                if (auto candidate = evaluate_candidate(candidate_bits))
                {
                    population.push_back(std::move(*candidate));
                }
                if (visited.size() % 25 == 0)
                {
                    std::cout
                        << "[scale-search] tried=" << visited.size()
                        << " valid=" << population.size() << "\n"
                        << std::flush;
                }
            };

            try_candidate(log_q_chain);
            auto all_max_chain = log_q_chain;
            std::fill(
                all_max_chain.begin() +
                    static_cast<std::ptrdiff_t>(search_first_index),
                all_max_chain.end(),
                search_max_bits);
            try_candidate(all_max_chain);

            // Seed the population with the previously observed 34->14
            // transition (low q[28]) and with broad random chains. The later
            // evolutionary rounds search all q[14..33], not just the former
            // two hand-tuned boundary primes. q[13] is included because the
            // dynamic leaf/combine alignment can use the highest retained
            // prime even when the final result keeps q[0..13].
            auto low_q28_chain = log_q_chain;
            if (q_count > 13)
            {
                low_q28_chain[13] = search_max_bits;
            }
            if (q_count > 28)
            {
                low_q28_chain[28] = std::max(search_min_bits, 22U);
                try_candidate(low_q28_chain);
            }
            const std::size_t initial_random_budget = env_size_or(
                "POSEIDON_BOOTSTRAP_SLIM_GLOBAL_SEARCH_INITIAL_RANDOM", 12);
            while (visited.size() < std::min<std::size_t>(
                       search_budget, initial_random_budget))
            {
                auto random_chain = all_max_chain;
                const std::size_t changes =
                    2 + static_cast<std::size_t>(generator() % 9U);
                for (std::size_t change = 0; change < changes; ++change)
                {
                    random_chain[random_index(generator)] =
                        random_bit(generator);
                }
                try_candidate(random_chain);
            }

            auto sort_population = [&]() {
                std::sort(
                    population.begin(),
                    population.end(),
                    [](const auto &lhs, const auto &rhs) {
                        return lhs.score > rhs.score;
                    });
            };
            sort_population();
            while (visited.size() < search_budget && !population.empty())
            {
                sort_population();
                const std::size_t parent_count =
                    std::min<std::size_t>(8, population.size());
                const auto parent = population[
                    static_cast<std::size_t>(generator()) % parent_count];
                auto child = parent.bits;
                const std::size_t changes =
                    1 + static_cast<std::size_t>(generator() % 4U);
                for (std::size_t change = 0; change < changes; ++change)
                {
                    const auto index = random_index(generator);
                    if ((generator() & 1U) != 0U)
                    {
                        const int delta =
                            static_cast<int>(generator() % 7U) - 3;
                        child[index] = static_cast<std::uint32_t>(
                            std::clamp(
                                static_cast<int>(child[index]) + delta,
                                static_cast<int>(search_min_bits),
                                static_cast<int>(search_max_bits)));
                    }
                    else
                    {
                        child[index] = random_bit(generator);
                    }
                }
                try_candidate(child);
            }
            sort_population();

            std::cout
                << "\n[Slim post-ModRaise C2S+EvalMod global scale search]\n"
                << "searched q indexes = " << search_first_index << ".."
                << (q_count - 1) << "\n"
                << "candidate bit range= " << search_min_bits << ".."
                << search_max_bits << "\n"
                << "objective          = maximize remaining Q, then "
                   "minimize |final_log_scale-45|\n"
                << "tried/valid        = " << visited.size() << "/"
                << population.size() << "\n";
            const std::size_t report_count =
                std::min<std::size_t>(12, population.size());
            for (std::size_t rank = 0; rank < report_count; ++rank)
            {
                const auto &candidate = population[rank];
                std::cout
                    << "SEARCH_RESULT rank=" << (rank + 1)
                    << " c2s_q=" << candidate.c2s_q_count
                    << " c2s_log_scale=" << std::setprecision(12)
                    << candidate.c2s_log_scale
                    << " eval_q=" << candidate.eval_q_count
                    << " eval_log_scale=" << candidate.eval_log_scale
                    << " dist40="
                    << std::abs(candidate.eval_log_scale - 40.0)
                    << " dist45="
                    << std::abs(candidate.eval_log_scale - 45.0)
                    << " c2s_drops=";
                for (std::size_t index = 0;
                     index < candidate.c2s_drops.size();
                     ++index)
                {
                    if (index != 0)
                        std::cout << ',';
                    std::cout << candidate.c2s_drops[index];
                }
                std::cout << " chain=";
                for (std::size_t index = 0;
                     index < candidate.bits.size();
                     ++index)
                {
                    if (index != 0)
                        std::cout << ',';
                    std::cout << candidate.bits[index];
                }
                std::cout << "\n";
            }
            return population.empty() ? EXIT_FAILURE : EXIT_SUCCESS;
        }

        if (slim_scale_chain_plan_only)
        {
            if (!evalmod_dynamic_rescale ||
                !env_flag_enabled("POSEIDON_BOOTSTRAP_SLIM_C2S_5433"))
            {
                throw std::invalid_argument(
                    "POSEIDON_BOOTSTRAP_SLIM_SCALE_CHAIN_PLAN_ONLY requires "
                    "dynamic rescale and POSEIDON_BOOTSTRAP_SLIM_C2S_5433=1");
            }

            const auto first_context_data =
                context.crt_context()->first_context_data();
            if (!first_context_data ||
                first_context_data->parms().q().size() != q_count)
            {
                throw std::runtime_error(
                    "scale-chain planner could not load the full Q chain");
            }
            std::vector<std::uint64_t> active_moduli;
            active_moduli.reserve(q_count);
            for (const auto &modulus : first_context_data->parms().q())
            {
                active_moduli.push_back(modulus.value());
            }

            std::size_t c2s_q_count = q_count;
            double c2s_scale = evalmod_scale;
            std::vector<std::uint32_t> c2s_drops;
            std::vector<double> c2s_output_scales;
            c2s_drops.reserve(4);
            c2s_output_scales.reserve(4);
            for (std::size_t stage = 0; stage < 4; ++stage)
            {
                const double pre_rescale_scale =
                    c2s_scale * evalmod_scale;
                const auto plan = poseidon::gpu::plan_gpu_dynamic_rescale(
                    pre_rescale_scale,
                    evalmod_scale,
                    std::span<const std::uint64_t>(
                        active_moduli.data(),
                        c2s_q_count),
                    /*require_rescale=*/true);
                c2s_drops.push_back(plan.rescale_count);
                c2s_output_scales.push_back(plan.output_scale);
                c2s_q_count = plan.output_q_count;
                c2s_scale = plan.output_scale;
            }

            const auto c2s_output_parms_id =
                context.crt_context()->parms_id_map().at(
                    static_cast<std::uint32_t>(c2s_q_count - 1));
            const auto c2s_output_context =
                context.crt_context()->get_context_data(
                    c2s_output_parms_id);
            if (!c2s_output_context)
            {
                throw std::runtime_error(
                    "scale-chain planner C2S output level is absent");
            }
            eval_mod_poly.set_level_start(
                static_cast<std::uint32_t>(c2s_output_context->level()));
            const auto evalmod_plan =
                poseidon::gpu::GpuUploader::upload_eval_mod_high_precision(
                    eval_mod_poly,
                    encoder,
                    c2s_output_parms_id,
                    device_id,
                    nullptr,
                    poseidon::parms_id_zero,
                    evalmod_rescale_count,
                    nullptr,
                    /*include_input_offset=*/true,
                    std::numeric_limits<std::uint32_t>::max(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    /*fuse_leaf_terms_before_rescale=*/true,
                    c2s_scale,
                    /*metadata_only=*/true);

            std::cout << "[Slim C2S+EvalMod scale-chain plan]\n";
            std::cout << "q bit chain       = ";
            for (std::size_t index = 0; index < log_q_chain.size(); ++index)
            {
                if (index != 0)
                    std::cout << ',';
                std::cout << log_q_chain[index];
            }
            std::cout << "\nC2S drops         = ";
            for (std::size_t stage = 0; stage < c2s_drops.size(); ++stage)
            {
                if (stage != 0)
                    std::cout << ',';
                std::cout << c2s_drops[stage];
            }
            std::cout << "\nC2S stage scales  = ";
            for (std::size_t stage = 0;
                 stage < c2s_output_scales.size();
                 ++stage)
            {
                if (stage != 0)
                    std::cout << ',';
                std::cout << std::log2(c2s_output_scales[stage]);
            }
            std::cout
                << "\nC2S q/output scale= " << c2s_q_count
                << "/2^" << std::log2(c2s_scale)
                << "\nEvalMod q/scale   = " << c2s_q_count << "->"
                << evalmod_plan.output_q_count << "/2^"
                << std::log2(evalmod_plan.output_scale)
                << "\nTotal Q consumed  = "
                << (q_count - evalmod_plan.output_q_count) << "\n";
            std::cout << "Eval basis plan   = ";
            for (std::size_t index = 0;
                 index < evalmod_plan.basis_steps.size();
                 ++index)
            {
                const auto &step = evalmod_plan.basis_steps[index];
                if (index != 0)
                    std::cout << ";";
                std::cout << "T" << step.output_degree
                          << ":q" << step.correction_plaintext.meta.q_count
                          << "-" << step.rescale_count
                          << "@" << std::log2(step.output_scale);
            }
            std::cout << "\nEval leaf plan    = ";
            for (std::size_t index = 0;
                 index < evalmod_plan.polynomial_blocks.size();
                 ++index)
            {
                const auto &block = evalmod_plan.polynomial_blocks[index];
                if (index != 0)
                    std::cout << ";";
                std::cout << index << ":q" << block.output_q_count
                          << "-" << block.rescale_count
                          << "@" << std::log2(block.output_scale);
            }
            std::cout << "\nEval combine plan = ";
            for (std::size_t index = 0;
                 index < evalmod_plan.polynomial_combine_steps.size();
                 ++index)
            {
                const auto &step =
                    evalmod_plan.polynomial_combine_steps[index];
                if (index != 0)
                    std::cout << ";";
                std::cout << index << ":T" << step.basis_degree
                          << "/q" << step.product_q_count
                          << "-" << step.quotient_rescale_count
                          << "-" << step.remainder_rescale_count
                          << "->q" << step.output_q_count
                          << "@" << std::log2(step.output_scale);
            }
            std::cout << "\nEval double-angle= ";
            for (std::size_t index = 0;
                 index < evalmod_plan.double_angle_rescale_counts.size();
                 ++index)
            {
                if (index != 0)
                    std::cout << ',';
                std::cout << evalmod_plan.double_angle_rescale_counts[index];
            }
            std::cout
                << "\nPLAN_RESULT c2s_q=" << c2s_q_count
                << " c2s_log_scale=" << std::setprecision(12)
                << std::log2(c2s_scale)
                << " eval_q=" << evalmod_plan.output_q_count
                << " eval_log_scale="
                << std::log2(evalmod_plan.output_scale)
                << " total_consumed="
                << (q_count - evalmod_plan.output_q_count)
                << "\n";
            return EXIT_SUCCESS;
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
        std::cout << "q bit chain     = ";
        for (std::size_t index = 0; index < log_q_chain.size(); ++index)
        {
            if (index != 0)
            {
                std::cout << ",";
            }
            std::cout << log_q_chain[index];
        }
        std::cout << "\n";
        std::cout << "log_p           = " << log_p << "\n";
        std::cout << "log_scale       = " << log_scale << "\n";
        std::cout << "q0_level        = " << q0_level << "\n";
        std::cout << "message_ratio   = " << bootstrap_ratio << "\n";
        std::cout << "evalmod_scale   = 2^" << evalmod_log_scale << "\n";
        std::cout << "evalmod_degree  = " << evalmod_sine_degree << "\n";
        std::cout << "generation_degree= "
                  << evalmod_generation_degree << "\n";
        std::cout << "effective_degree= "
                  << eval_mod_poly.sine_poly().degree() << "\n";
        std::cout << "coefficient_src = "
                  << (evalmod_fixed_degree_refit
                          ? "fixed-degree Chebyshev interpolation"
                          : (evalmod_truncate_degree.has_value()
                                 ? "higher-degree truncation"
                                 : "native ApproximateCos"))
                  << "\n";
        std::cout << "evalmod_rescale = " << evalmod_rescale_count
                  << " physical prime(s) per logical step\n";
        std::cout << "evalmod_dynamic = "
                  << (evalmod_dynamic_rescale ? "ON" : "OFF") << "\n";
        std::cout << "double_angle    = " << evalmod_double_angle << "\n";
        std::cout << "evalmod_k       = " << evalmod_k << "\n";
        std::cout << "tolerance       = " << correctness_tolerance << "\n";
        std::cout << "c2s_scaling     = " << c2s_scaling << "\n";
        std::cout << "c2s_bsgs_ratio  = " << c2s_log_bsgs_ratio << "\n";
        std::cout << "c2s_step        = " << c2s_step << "\n";
        if (evalmod_dynamic_rescale)
        {
            std::cout << "c2s_rescale     = dynamic (GS min_scale/2)\n";
        }
        else
        {
            std::cout << "c2s_rescale     = " << c2s_step
                      << " physical prime(s) per matrix\n";
        }
        std::cout << "s2c_scaling     = " << s2c_scaling << "\n";
        std::cout << "s2c_bsgs_ratio  = " << s2c_log_bsgs_ratio << "\n";
        std::cout << "s2c_step        = " << s2c_step << "\n";
        if (evalmod_dynamic_rescale)
        {
            std::cout << "s2c_rescale     = dynamic (GS min_scale/2)\n";
        }
        else
        {
            std::cout << "s2c_rescale     = " << s2c_step
                      << " physical prime(s) per matrix\n";
        }
        std::cout << "iterations      = " << iterations << "\n";
        std::cout << "full_iterations = " << full_iterations << "\n";
        std::cout << "full_warmup     = " << full_warmup << "\n";
        std::cout << "stage_profile   = "
                  << evalmod_stage_profile_iterations << " iteration(s)\n";
        std::cout << "timing_mode     = "
                  << (gpu_only_timing ? "GPU-only" : "CPU/GPU") << "\n";
        std::cout << "diagnostics     = "
                  << (detailed_diagnostics ? "detailed" : "summary") << "\n";
        std::cout << "nsys_capture    = "
                  << (nsys_capture_full ? "full bootstrap only" : "OFF")
                  << "\n";
        std::cout << "ignore_failure  = "
                  << (ignore_correctness_failure ? "YES" : "NO") << "\n";
        std::cout << "setup_only      = "
                  << (setup_only ? "YES" : "NO") << "\n";
        std::cout << "scale_plan_only = "
                  << (scale_planner_only ? "YES" : "NO") << "\n";
        std::cout << "c2s_only        = "
                  << (c2s_only ? "YES" : "NO") << "\n";
        std::cout << "evalmod_only    = "
                  << (evalmod_only ? "YES" : "NO") << "\n";
        std::cout << "slim_stc_probe  = "
                  << (slim_stc_first_probe ? "YES" : "NO") << "\n";
        std::cout << "slim_stc_raise  = "
                  << (slim_stc_modraise_probe ? "YES" : "NO") << "\n";
        std::cout << "slim_stc_c2s    = "
                  << (slim_stc_c2s_probe ? "YES" : "NO") << "\n";
        std::cout << "slim_c2s_5433   = "
                  << (slim_c2s_5433 ? "YES" : "NO") << "\n";
        std::cout << "slim_stc_evalmod= "
                  << (slim_stc_evalmod_probe ? "YES" : "NO") << "\n";
        std::cout << "ptxt_compress   = "
                  << (plaintext_compression_probe ? "YES" : "NO") << "\n";
        std::cout << "compact_qp_mac  = "
                  << (compressed_qp_mac_probe ? "YES" : "NO") << "\n";
        if (slim_stc_first_probe || slim_stc_run_modraise)
        {
            std::cout << "slim_stc_input_q= "
                      << slim_stc_input_q_count << "\n";
        }

        poseidon::gpu::GpuParameterData gpu_params(context, device_id);
        poseidon::gpu::GpuEvaluator gpu_evaluator(gpu_params);
        if (scale_planner_only)
        {
            run_dynamic_scale_planner_tests(
                context,
                *cpu_evaluator,
                gpu_evaluator,
                source,
                evalmod_scale,
                device_id);
            std::cout << "\n[OK] Stage 1 GS-compatible dynamic scale planner "
                         "and GPU rescale tests passed\n";
            return EXIT_SUCCESS;
        }
        if (slim_stc_first_probe || slim_stc_run_modraise)
        {
            if (!evalmod_dynamic_rescale)
            {
                throw std::invalid_argument(
                    "POSEIDON_BOOTSTRAP_SLIM_STC_FIRST_PROBE requires "
                    "POSEIDON_BOOTSTRAP_EVALMOD_DYNAMIC_RESCALE=1");
            }
            if (slim_stc_input_q_count <= 1 ||
                slim_stc_input_q_count > q_count)
            {
                throw std::invalid_argument(
                    "POSEIDON_BOOTSTRAP_SLIM_STC_INPUT_Q_COUNT must be in "
                    "[2, POSEIDON_BOOTSTRAP_Q_COUNT]");
            }

            const auto slim_input_parms_id =
                context.crt_context()->parms_id_map().at(
                    static_cast<std::uint32_t>(
                        slim_stc_input_q_count - 1));
            poseidon::Ciphertext cpu_slim_input;
            cpu_evaluator->drop_modulus(
                source,
                cpu_slim_input,
                slim_input_parms_id);
            cpu_slim_input.scale() = source.scale();

            // StC-first changes only the position in the modulus chain. Keep
            // the production DFT scale policy unchanged: every plaintext
            // matrix uses EvalMod's 2^45 working scale and dynamic rescale
            // keeps each physical result above the 2^44 half-target floor.
            const double slim_plaintext_scale = evalmod_scale;
            const double slim_min_scale = evalmod_scale;
            // In the production order, the trailing StC normalization
            // includes message_ratio because it follows EvalMod.  Once StC
            // is moved before ModRaise, that factor would be applied a second
            // time by the final EvalMod output multiplication.  Keep the
            // reordered transform pair normalized independently of the
            // message ratio.  The explicit override is retained for focused
            // experiments without changing the production path.
            const double slim_stc_scaling = env_double_or(
                "POSEIDON_BOOTSTRAP_SLIM_STC_SCALING",
                1.0);
            const double slim_semantic_target_tolerance = env_double_or(
                "POSEIDON_BOOTSTRAP_SLIM_STC_TARGET_TOLERANCE",
                1.0e-4);
            const double slim_semantic_minimum_tolerance = env_double_or(
                "POSEIDON_BOOTSTRAP_SLIM_STC_MINIMUM_TOLERANCE",
                1.0e-3);
            if (!(slim_semantic_target_tolerance > 0.0) ||
                !(slim_semantic_minimum_tolerance >=
                  slim_semantic_target_tolerance))
            {
                throw std::invalid_argument(
                    "slim StC semantic tolerances must satisfy "
                    "0 < target <= minimum");
            }

            std::cout
                << "\n[Slim StC-first phase 1 probe]\n"
                << "Creating only the low-level StC matrices/keys; "
                   "full bootstrap remains unchanged.\n"
                << "[phase] create low-level StC matrix group"
                << " input_q=" << slim_stc_input_q_count
                << " input_scale=2^" << std::log2(cpu_slim_input.scale())
                << " plaintext_scale=2^"
                << std::log2(slim_plaintext_scale)
                << " min_scale=2^" << std::log2(slim_min_scale)
                << " normalization=" << slim_stc_scaling
                << "\n" << std::flush;

            auto slim_stc_matrix_group =
                make_slot_to_coeff_matrix_group(
                    context,
                    encoder,
                    static_cast<std::uint32_t>(
                        slim_stc_input_q_count - 1),
                    slim_stc_scaling,
                    s2c_log_bsgs_ratio,
                    s2c_step,
                    cpu_slim_input.scale(),
                    slim_min_scale);
            const std::size_t slim_reference_input_q_count =
                env_size_or(
                    "POSEIDON_BOOTSTRAP_SLIM_STC_REFERENCE_Q_COUNT",
                    15);
            if (slim_reference_input_q_count <= slim_stc_input_q_count ||
                slim_reference_input_q_count > q_count)
            {
                throw std::invalid_argument(
                    "POSEIDON_BOOTSTRAP_SLIM_STC_REFERENCE_Q_COUNT must be "
                    "greater than the low-level input and no greater than Q");
            }
            poseidon::Ciphertext cpu_slim_reference_input;
            cpu_evaluator->drop_modulus(
                source,
                cpu_slim_reference_input,
                context.crt_context()->parms_id_map().at(
                    static_cast<std::uint32_t>(
                        slim_reference_input_q_count - 1)));
            cpu_slim_reference_input.scale() = source.scale();
            auto slim_stc_reference_matrix_group =
                make_slot_to_coeff_matrix_group(
                    context,
                    encoder,
                    static_cast<std::uint32_t>(
                        slim_reference_input_q_count - 1),
                    slim_stc_scaling,
                    s2c_log_bsgs_ratio,
                    s2c_step,
                    cpu_slim_reference_input.scale(),
                    evalmod_scale);
            poseidon::LinearMatrixGroup slim_c2s_matrix_group;
            if (slim_stc_run_c2s)
            {
                std::cout
                    << "[phase] create post-ModRaise C2S matrix group"
                    << " input_q=" << q_count
                    << " logical_input_scale=2^"
                    << std::log2(evalmod_scale)
                    << " plaintext_scale=2^"
                    << std::log2(evalmod_scale)
                    << " min_scale=2^"
                    << std::log2(evalmod_scale) << "\n" << std::flush;
                slim_c2s_matrix_group = make_coeff_to_slot_matrix_group(
                    context,
                    encoder,
                    c2s_scaling,
                    c2s_log_bsgs_ratio,
                    c2s_step,
                    evalmod_scale,
                    evalmod_scale,
                    1.0,
                    slim_c2s_5433
                        ? std::vector<std::uint32_t>{5, 4, 3, 3}
                        : std::vector<std::uint32_t>{},
                    slim_c2s_5433 ? 3 : 0);
                if (slim_c2s_5433)
                {
                    const std::vector<std::uint32_t> layer_groups{
                        5, 4, 3, 3};
                    const std::vector<std::size_t> expected_diagonals{
                        32, 31, 15, 15};
                    const int slots =
                        1 << context.parameters_literal()->log_slots();
                    if (slim_c2s_matrix_group.data().size() !=
                        layer_groups.size())
                    {
                        throw std::runtime_error(
                            "slim C2S [5,4,3,3] matrix count mismatch");
                    }
                    std::cout
                        << "[Slim C2S 5+4+3+3 matrix plan]\n";
                    for (std::size_t stage = 0;
                         stage < layer_groups.size();
                         ++stage)
                    {
                        const auto &matrix =
                            slim_c2s_matrix_group.data()[stage];
                        const auto [index, unused_giant_steps, baby_steps] =
                            poseidon::bsgs_index(
                                matrix.plain_vec,
                                slots,
                                static_cast<int>(matrix.n1));
                        (void)unused_giant_steps;
                        const bool direct = layer_groups[stage] <= 3;
                        std::cout
                            << "stage=" << stage
                            << " fused_layers=" << layer_groups[stage]
                            << " diagonals=" << matrix.plain_vec.size()
                            << " n1=" << matrix.n1
                            << " baby=" << baby_steps.size()
                            << " giant=" << index.size()
                            << " mode=" << (direct ? "direct" : "BSGS")
                            << "\n" << std::flush;
                        if (matrix.plain_vec.size() !=
                                expected_diagonals[stage] ||
                            direct != (matrix.n1 ==
                                       static_cast<std::uint32_t>(slots)) ||
                            (direct && index.size() != 1))
                        {
                            throw std::runtime_error(
                                "slim C2S [5,4,3,3] stage plan mismatch");
                        }
                    }
                }
            }
            poseidon::gpu::GpuLinearMatrixGroupQP compact_probe_stc;
            poseidon::gpu::GpuLinearMatrixGroupQP compact_probe_c2s;
            if (plaintext_compression_probe || compressed_qp_mac_probe)
            {
                const auto stc_stats = analyze_plaintext_compression(
                    "front-loaded StC",
                    slim_stc_matrix_group,
                    degree,
                    p_count);
                const auto c2s_stats = analyze_plaintext_compression(
                    "post-ModRaise C2S",
                    slim_c2s_matrix_group,
                    degree,
                    p_count);
                if (!stc_stats.exact_q_reconstruction ||
                    !c2s_stats.exact_q_reconstruction)
                {
                    std::cerr
                        << "[FAILED] WHET plaintext compression probe did "
                           "not reconstruct every Q plaintext exactly\n";
                    return EXIT_FAILURE;
                }

                std::cout
                    << "\n[phase] construct compact QP device plaintexts "
                       "and verify every residue\n"
                    << std::flush;
                compact_probe_stc =
                    poseidon::gpu::GpuUploader::
                        upload_linear_matrix_group_qp(
                            slim_stc_matrix_group,
                            context,
                            device_id,
                            std::max(
                                slim_stc_matrix_group.step(),
                                std::uint32_t{1}),
                            true);
                compact_probe_c2s =
                    poseidon::gpu::GpuUploader::
                        upload_linear_matrix_group_qp(
                            slim_c2s_matrix_group,
                            context,
                            device_id,
                            std::max(
                                slim_c2s_matrix_group.step(),
                                std::uint32_t{1}),
                            true);
                const auto actual_stc_stats =
                    analyze_uploaded_compressed_qp(
                        "front-loaded StC",
                        compact_probe_stc);
                const auto actual_c2s_stats =
                    analyze_uploaded_compressed_qp(
                        "post-ModRaise C2S",
                        compact_probe_c2s);
                if (!actual_stc_stats.exact_qp_device_reconstruction ||
                    !actual_c2s_stats.exact_qp_device_reconstruction)
                {
                    std::cerr
                        << "[FAILED] compact QP device plaintexts did not "
                           "reconstruct every Q/P residue exactly\n";
                    return EXIT_FAILURE;
                }
                const auto estimated_words =
                    stc_stats.compressed_qp_words +
                    c2s_stats.compressed_qp_words;
                const auto actual_words =
                    actual_stc_stats.compressed_qp_words +
                    actual_c2s_stats.compressed_qp_words;
                std::cout
                    << "[OK] WHET bit-reversed periodicity reconstructed "
                       "all encoded Q plaintexts exactly. Compact device "
                       "storage also reconstructed every Q residue and all "
                    << p_count
                    << " P residues bit-for-bit. Q-only period estimate "
                       "matches exact QP upload="
                    << (estimated_words == actual_words ? "YES" : "NO")
                    << (compressed_qp_mac_probe
                            ? ". Proceeding to the isolated compressed QP "
                              "MAC compute probe.\n"
                            : ". No compressed plaintext entered a compute "
                              "kernel.\n");
                if (!compressed_qp_mac_probe)
                {
                    return EXIT_SUCCESS;
                }
            }
            const auto slim_rescale_counts =
                dft_stage_rescale_counts(slim_stc_matrix_group);
            const std::size_t slim_consumed_q = std::accumulate(
                slim_rescale_counts.begin(),
                slim_rescale_counts.end(),
                std::size_t{0});
            if (slim_consumed_q >= slim_stc_input_q_count)
            {
                throw std::runtime_error(
                    "low-level StC consumes all available Q moduli");
            }
            const std::size_t slim_output_q_count =
                slim_stc_input_q_count - slim_consumed_q;
            const std::size_t modraise_base_q_count =
                static_cast<std::size_t>(q0_level) + 1;
            if (slim_output_q_count != modraise_base_q_count)
            {
                throw std::runtime_error(
                    "phase-1 low-level StC probe must finish at the configured "
                    "ModRaise base q_count; "
                    "adjust POSEIDON_BOOTSTRAP_SLIM_STC_INPUT_Q_COUNT");
            }

            std::vector<const poseidon::LinearMatrixGroup *>
                slim_matrix_groups_for_keys{
                    &slim_stc_matrix_group,
                    &slim_stc_reference_matrix_group};
            if (slim_stc_run_c2s)
            {
                slim_matrix_groups_for_keys.push_back(
                    &slim_c2s_matrix_group);
            }
            auto slim_galois_keys = make_galois_keys_for_matrix_groups(
                context,
                keygen,
                slim_matrix_groups_for_keys);
            const auto slim_mode =
                poseidon::gpu::gpu_linear_transform_mode_from_environment(
                    poseidon::gpu::GpuLinearTransformMode::ClassicBsgs);
            if (compressed_qp_mac_probe &&
                slim_mode !=
                    poseidon::gpu::GpuLinearTransformMode::DoubleHoistBsgs)
            {
                throw std::invalid_argument(
                    "POSEIDON_BOOTSTRAP_COMPRESSED_QP_MAC_PROBE requires "
                    "POSEIDON_GPU_LINEAR_TRANSFORM_MODE=double_hoist");
            }
            auto gpu_slim_galois_keys =
                slim_mode ==
                        poseidon::gpu::GpuLinearTransformMode::DoubleHoistBsgs
                    ? poseidon::gpu::GpuUploader::
                          upload_double_hoist_galois_keys(
                              slim_galois_keys,
                              device_id)
                    : poseidon::gpu::GpuUploader::upload_galois_keys(
                          slim_galois_keys,
                          device_id);
            const auto slim_key_q_counts = required_dft_key_q_counts(
                slim_stc_input_q_count,
                slim_stc_matrix_group.data().size(),
                slim_stc_matrix_group.step(),
                false,
                slim_rescale_counts);
            auto slim_all_key_q_counts = slim_key_q_counts;
            if (slim_stc_run_c2s)
            {
                const auto slim_c2s_rescale_counts =
                    dft_stage_rescale_counts(slim_c2s_matrix_group);
                const auto slim_c2s_key_q_counts =
                    required_dft_key_q_counts(
                        q_count,
                        slim_c2s_matrix_group.data().size(),
                        slim_c2s_matrix_group.step(),
                        true,
                        slim_c2s_rescale_counts);
                slim_all_key_q_counts = merge_q_counts(
                    slim_all_key_q_counts,
                    slim_c2s_key_q_counts);
            }
            poseidon::gpu::GpuUploader::prepare_key_views_for_q_counts(
                gpu_slim_galois_keys,
                slim_all_key_q_counts);

            poseidon::gpu::GpuLinearMatrixGroup gpu_slim_matrix;
            poseidon::gpu::GpuLinearMatrixGroupQP gpu_slim_matrix_qp;
            if (slim_mode ==
                poseidon::gpu::GpuLinearTransformMode::DoubleHoistBsgs)
            {
                gpu_slim_matrix_qp =
                    poseidon::gpu::GpuUploader::upload_linear_matrix_group_qp(
                        slim_stc_matrix_group,
                        context,
                        device_id,
                        std::max(
                            slim_stc_matrix_group.step(),
                            std::uint32_t{1}));
            }
            else
            {
                gpu_slim_matrix =
                    poseidon::gpu::GpuUploader::upload_linear_matrix_group(
                        slim_stc_matrix_group,
                        device_id);
            }
            poseidon::gpu::GpuLinearMatrixGroup gpu_slim_c2s_matrix;
            poseidon::gpu::GpuLinearMatrixGroupQP gpu_slim_c2s_matrix_qp;
            if (slim_stc_run_c2s)
            {
                if (slim_mode ==
                    poseidon::gpu::GpuLinearTransformMode::DoubleHoistBsgs)
                {
                    gpu_slim_c2s_matrix_qp =
                        poseidon::gpu::GpuUploader::
                            upload_linear_matrix_group_qp(
                                slim_c2s_matrix_group,
                                context,
                                device_id,
                                std::max(
                                    slim_c2s_matrix_group.step(),
                                    std::uint32_t{1}));
                }
                else
                {
                    gpu_slim_c2s_matrix =
                        poseidon::gpu::GpuUploader::
                            upload_linear_matrix_group(
                                slim_c2s_matrix_group,
                                device_id);
                }
            }

            poseidon::Ciphertext cpu_slim_output;
            std::vector<poseidon::Ciphertext> cpu_slim_stage_trace;
            std::cout << "[phase] CPU low-level StC correctness\n"
                      << std::flush;
            cpu_dft_rescale(
                cpu_slim_input,
                slim_stc_matrix_group,
                cpu_slim_output,
                *cpu_evaluator,
                slim_galois_keys,
                &cpu_slim_stage_trace);
            poseidon::Ciphertext cpu_slim_reference_output;
            std::vector<poseidon::Ciphertext> cpu_slim_reference_stage_trace;
            std::cout << "[phase] high-level CPU StC semantic oracle\n"
                      << std::flush;
            cpu_dft_rescale(
                cpu_slim_reference_input,
                slim_stc_reference_matrix_group,
                cpu_slim_reference_output,
                *cpu_evaluator,
                slim_galois_keys,
                &cpu_slim_reference_stage_trace);
            const auto slim_input_comparison =
                compare_decrypted_ciphertexts(
                    cpu_slim_reference_input,
                    cpu_slim_input,
                    decryptor,
                    encoder,
                    slim_semantic_minimum_tolerance);
            const auto slim_semantic_comparison =
                compare_decrypted_ciphertexts(
                    cpu_slim_reference_output,
                    cpu_slim_output,
                    decryptor,
                    encoder,
                    slim_semantic_minimum_tolerance);
            if (cpu_slim_stage_trace.size() !=
                    cpu_slim_reference_stage_trace.size())
            {
                throw std::runtime_error(
                    "low/high StC stage trace size mismatch");
            }
            std::vector<ApproxComparison> slim_stage_comparisons;
            slim_stage_comparisons.reserve(cpu_slim_stage_trace.size());
            for (std::size_t stage = 0;
                 stage < cpu_slim_stage_trace.size();
                 ++stage)
            {
                slim_stage_comparisons.push_back(
                    compare_decrypted_ciphertexts(
                        cpu_slim_reference_stage_trace[stage],
                        cpu_slim_stage_trace[stage],
                        decryptor,
                        encoder,
                        slim_semantic_minimum_tolerance));
            }

            auto gpu_slim_input =
                poseidon::gpu::GpuUploader::upload_ciphertext(
                    cpu_slim_input,
                    device_id);
            poseidon::gpu::GpuCiphertextData gpu_slim_output;
            poseidon::gpu::GpuDoubleHoistWorkspace slim_workspace;
            poseidon::gpu::GpuCiphertextData gpu_compact_slim_output;
            poseidon::gpu::GpuDoubleHoistWorkspace compact_slim_workspace;
            auto run_gpu_slim_stc = [&]() {
                if (slim_mode ==
                    poseidon::gpu::GpuLinearTransformMode::DoubleHoistBsgs)
                {
                    gpu_evaluator.dft_double_hoist(
                        gpu_slim_input,
                        gpu_slim_matrix_qp,
                        gpu_slim_galois_keys,
                        slim_workspace,
                        gpu_slim_output);
                }
                else
                {
                    gpu_evaluator.dft(
                        gpu_slim_input,
                        gpu_slim_matrix,
                        gpu_slim_galois_keys,
                    gpu_slim_output);
                }
            };
            auto run_gpu_compact_slim_stc = [&]() {
                gpu_evaluator.dft_double_hoist(
                    gpu_slim_input,
                    compact_probe_stc,
                    gpu_slim_galois_keys,
                    compact_slim_workspace,
                    gpu_compact_slim_output);
            };

            std::cout << "[phase] GPU low-level StC correctness\n"
                      << std::flush;
            run_gpu_slim_stc();
            cudaDeviceSynchronize();
            poseidon::Ciphertext gpu_slim_download;
            poseidon::gpu::GpuUploader::download_ciphertext(
                gpu_slim_output,
                gpu_slim_download,
                context);
            RawComparison compact_slim_qp_q;
            RawComparison compact_slim_qp_p;
            RawComparison compact_slim_output_raw;
            ApproxComparison compact_slim_output_approx;
            if (compressed_qp_mac_probe)
            {
                std::cout
                    << "[phase] compressed QP MAC low-level StC exactness\n"
                    << std::flush;
                run_gpu_compact_slim_stc();
                cudaDeviceSynchronize();
                const auto compact_slim_download =
                    download_gpu_ciphertext(
                        gpu_compact_slim_output,
                        context);
                compact_slim_output_raw = compare_ciphertexts(
                    gpu_slim_download,
                    compact_slim_download,
                    8);
                compact_slim_output_approx =
                    compare_decrypted_ciphertexts(
                        gpu_slim_download,
                        compact_slim_download,
                        decryptor,
                        encoder,
                        0.0);

                const auto &full_qp =
                    slim_workspace.group_accumulators;
                const auto &compact_qp =
                    compact_slim_workspace.group_accumulators;
                if (full_qp.degree != compact_qp.degree ||
                    full_qp.q_count != compact_qp.q_count ||
                    full_qp.p_count != compact_qp.p_count ||
                    full_qp.batch_count != compact_qp.batch_count)
                {
                    throw std::runtime_error(
                        "compressed StC QP MAC workspace shape mismatch");
                }
                const std::size_t active_q_words =
                    full_qp.batch_count * 2 * full_qp.q_count *
                    full_qp.degree;
                const std::size_t active_p_words =
                    full_qp.batch_count * 2 * full_qp.p_count *
                    full_qp.degree;
                compact_slim_qp_q = compare_device_words_exact(
                    full_qp.q,
                    compact_qp.q,
                    active_q_words,
                    "StC compact Q MAC");
                compact_slim_qp_p = compare_device_words_exact(
                    full_qp.p,
                    compact_qp.p,
                    active_p_words,
                    "StC compact P MAC");
                if (!compact_slim_qp_q.equal ||
                    !compact_slim_qp_p.equal ||
                    !compact_slim_output_raw.equal ||
                    !compact_slim_output_approx.equal)
                {
                    std::cerr
                        << "[FAILED] compressed QP MAC changed low-level "
                           "StC residues or output\n";
                    return EXIT_FAILURE;
                }
            }
            const auto slim_comparison = compare_decrypted_ciphertexts(
                cpu_slim_output,
                gpu_slim_download,
                decryptor,
                encoder,
                correctness_tolerance);
            if (cpu_slim_output.coeff_modulus_size() !=
                    modraise_base_q_count ||
                gpu_slim_download.coeff_modulus_size() !=
                    modraise_base_q_count)
            {
                throw std::runtime_error(
                    "low-level StC CPU/GPU execution did not finish at the "
                    "configured ModRaise base q_count");
            }

            for (std::size_t index = 0; index < warmup; ++index)
            {
                run_gpu_slim_stc();
            }
            cudaDeviceSynchronize();
            const double slim_gpu_ms = time_gpu_ms(
                iterations,
                run_gpu_slim_stc);
            double compact_slim_gpu_ms =
                std::numeric_limits<double>::quiet_NaN();
            if (compressed_qp_mac_probe)
            {
                for (std::size_t index = 0; index < warmup; ++index)
                {
                    run_gpu_compact_slim_stc();
                }
                cudaDeviceSynchronize();
                compact_slim_gpu_ms = time_gpu_ms(
                    iterations,
                    run_gpu_compact_slim_stc);
            }

            std::cout << "\n[low-level StC q-count/scale trace]\n";
            std::size_t trace_q_count = slim_stc_input_q_count;
            double trace_scale = cpu_slim_input.scale();
            const auto first_context =
                context.crt_context()->first_context_data();
            const double slim_scale_floor =
                (slim_min_scale + 1.0) / 2.0;
            for (std::size_t stage = 0;
                 stage < slim_stc_matrix_group.data().size();
                 ++stage)
            {
                const auto input_q_count = trace_q_count;
                const auto input_scale = trace_scale;
                const auto plaintext_scale =
                    slim_stc_matrix_group.data()[stage].scale;
                trace_scale *= plaintext_scale;
                for (std::uint32_t drop = 0;
                     drop < slim_rescale_counts[stage];
                     ++drop)
                {
                    trace_scale /= static_cast<double>(
                        first_context->coeff_modulus()
                            .at(trace_q_count - 1)
                            .value());
                    --trace_q_count;
                }
                const double stage_scale_upper =
                    slim_scale_floor * static_cast<double>(
                        first_context->coeff_modulus()
                            .at(trace_q_count - 1)
                            .value());
                if (trace_scale < slim_scale_floor ||
                    trace_scale >= stage_scale_upper)
                {
                    throw std::runtime_error(
                        "low-level StC output escaped the production dynamic "
                        "scale interval");
                }
                std::cout << "stage=" << stage
                          << " q=" << input_q_count
                          << "->" << trace_q_count
                          << " input=2^" << std::log2(input_scale)
                          << " plain=2^" << std::log2(plaintext_scale)
                          << " drop=" << slim_rescale_counts[stage]
                          << " output=2^" << std::log2(trace_scale)
                          << " allowed=[2^"
                          << std::log2(slim_scale_floor) << ",2^"
                          << std::log2(stage_scale_upper) << ")"
                          << "\n";
            }
            std::cout << "mode             = "
                      << (slim_mode ==
                                  poseidon::gpu::GpuLinearTransformMode::
                                      DoubleHoistBsgs
                              ? "double_hoist"
                              : "classic")
                      << "\n"
                      << "matrix groups    = "
                      << slim_stc_matrix_group.data().size() << "\n"
                      << "rescale pattern  = ";
            for (std::size_t index = 0;
                 index < slim_rescale_counts.size();
                 ++index)
            {
                if (index != 0)
                {
                    std::cout << ",";
                }
                std::cout << slim_rescale_counts[index];
            }
            std::cout << " (total=" << slim_consumed_q << ")\n"
                      << "key q views      = "
                      << join_q_counts(slim_key_q_counts) << "\n"
                      << "CPU/GPU max error= "
                      << slim_comparison.max_abs_error << "\n"
                      << "low/high input err= "
                      << slim_input_comparison.max_abs_error << " (rms="
                      << slim_input_comparison.rms_error << ")\n";
            for (std::size_t stage = 0;
                 stage < slim_stage_comparisons.size();
                 ++stage)
            {
                std::cout << "low/high stage " << stage << " err= "
                          << slim_stage_comparisons[stage].max_abs_error
                          << " (rms="
                          << slim_stage_comparisons[stage].rms_error << ")\n";
            }
            std::cout
                      << "low/high StC delta= "
                      << slim_semantic_comparison.max_abs_error << "\n"
                      << "semantic target  = "
                      << slim_semantic_target_tolerance << "\n"
                      << "semantic minimum = "
                      << slim_semantic_minimum_tolerance << "\n"
                      << "delta scope      = intermediate coefficient "
                         "representation (not end-to-end precision)\n"
                      << "GPU avg latency  = " << slim_gpu_ms << " ms\n";
            if (compressed_qp_mac_probe)
            {
                std::cout
                    << "compact Q MAC exact= "
                    << (compact_slim_qp_q.equal ? "YES" : "NO") << "\n"
                    << "compact P MAC exact= "
                    << (compact_slim_qp_p.equal ? "YES" : "NO") << "\n"
                    << "compact output exact= "
                    << (compact_slim_output_raw.equal ? "YES" : "NO")
                    << "\n"
                    << "compact GPU latency= "
                    << compact_slim_gpu_ms << " ms\n";
            }
            if (!slim_comparison.equal || !slim_semantic_comparison.equal)
            {
                std::cerr
                    << "[FAILED] low-level StC implementation or semantic "
                       "oracle mismatch\n";
                return EXIT_FAILURE;
            }
            if (slim_semantic_comparison.max_abs_error >
                slim_semantic_target_tolerance)
            {
                std::cout
                    << "[WARN] The intermediate low/high StC delta exceeds "
                       "the final bootstrap target. This is diagnostic only; "
                       "validate precision after the complete StC-first "
                       "bootstrap schedule.\n";
            }
            if (slim_stc_run_modraise)
            {
                std::cout
                    << "\n[Slim StC-first phase 2: ModRaise]\n"
                    << "Preparing the q=" << modraise_base_q_count
                    << " StC output at q0/message_ratio and raising it to q="
                    << q_count << ".\n" << std::flush;

                poseidon::Ciphertext cpu_slim_raised =
                    cpu_bootstrap_prepare_and_raise(
                        cpu_slim_output,
                        *cpu_evaluator,
                        context,
                        encoder,
                        target_q0_scale);

                poseidon::gpu::GpuCiphertextData gpu_slim_prepared;
                poseidon::gpu::GpuCiphertextData gpu_slim_raised;
                auto run_gpu_slim_modraise = [&]() {
                    gpu_evaluator.bootstrap_prepare_modraise_input(
                        gpu_slim_output,
                        gpu_slim_prepared,
                        q0_parms_id,
                        target_q0_scale);
                    gpu_evaluator.raise_modulus(
                        gpu_slim_prepared,
                        gpu_slim_raised);
                };
                run_gpu_slim_modraise();
                cudaDeviceSynchronize();

                const auto gpu_slim_modraise_input_cpu =
                    download_gpu_ciphertext(gpu_slim_output, context);
                const auto cpu_from_gpu_slim_raised =
                    cpu_bootstrap_prepare_and_raise(
                        gpu_slim_modraise_input_cpu,
                        *cpu_evaluator,
                        context,
                        encoder,
                        target_q0_scale);
                const auto cpu_gpu_slim_raised =
                    download_gpu_ciphertext(gpu_slim_raised, context);
                const auto gpu_slim_prepared_cpu =
                    download_gpu_ciphertext(gpu_slim_prepared, context);
                const auto slim_modraise_raw = compare_ciphertexts(
                    cpu_from_gpu_slim_raised,
                    cpu_gpu_slim_raised,
                    8);
                const auto slim_prepare_semantic =
                    compare_decrypted_ciphertexts(
                        cpu_slim_output,
                        gpu_slim_prepared_cpu,
                        decryptor,
                        encoder,
                        slim_semantic_minimum_tolerance);
                const auto slim_cpu_raise_semantic =
                    compare_decrypted_ciphertexts(
                        cpu_slim_output,
                        cpu_slim_raised,
                        decryptor,
                        encoder,
                        slim_semantic_minimum_tolerance);
                const auto slim_gpu_raise_semantic =
                    compare_decrypted_ciphertexts(
                        cpu_slim_output,
                        cpu_gpu_slim_raised,
                        decryptor,
                        encoder,
                        slim_semantic_minimum_tolerance);

                if (gpu_slim_prepared.meta.q_count !=
                        modraise_base_q_count ||
                    gpu_slim_raised.meta.q_count != q_count ||
                    cpu_slim_raised.coeff_modulus_size() != q_count)
                {
                    throw std::runtime_error(
                        "slim StC-first ModRaise q-count invariant failed");
                }

                for (std::size_t index = 0; index < warmup; ++index)
                {
                    run_gpu_slim_modraise();
                }
                cudaDeviceSynchronize();
                const double slim_modraise_gpu_ms = time_gpu_ms(
                    iterations,
                    run_gpu_slim_modraise);
                auto run_gpu_slim_stc_and_modraise = [&]() {
                    run_gpu_slim_stc();
                    run_gpu_slim_modraise();
                };
                for (std::size_t index = 0; index < warmup; ++index)
                {
                    run_gpu_slim_stc_and_modraise();
                }
                cudaDeviceSynchronize();
                const double slim_stc_modraise_gpu_ms = time_gpu_ms(
                    iterations,
                    run_gpu_slim_stc_and_modraise);

                std::cout
                    << "input q/scale    = "
                    << cpu_slim_output.coeff_modulus_size() << "/2^"
                    << std::log2(cpu_slim_output.scale()) << "\n"
                    << "prepared q/scale = "
                    << gpu_slim_prepared.meta.q_count << "/2^"
                    << std::log2(gpu_slim_prepared.meta.scale) << "\n"
                    << "raised q/scale   = "
                    << gpu_slim_raised.meta.q_count << "/2^"
                    << std::log2(gpu_slim_raised.meta.scale) << "\n"
                    << "CPU/GPU raw equal= "
                    << (slim_modraise_raw.equal ? "YES" : "NO") << "\n"
                    << "prepare delta    = "
                    << slim_prepare_semantic.max_abs_error << " (rms="
                    << slim_prepare_semantic.rms_error << ")\n"
                    << "CPU raise delta  = "
                    << slim_cpu_raise_semantic.max_abs_error << " (rms="
                    << slim_cpu_raise_semantic.rms_error << ")\n"
                    << "GPU raise delta  = "
                    << slim_gpu_raise_semantic.max_abs_error << " (rms="
                    << slim_gpu_raise_semantic.rms_error << ")\n"
                    << "ModRaise GPU ms  = " << slim_modraise_gpu_ms << "\n"
                    << "StC+raise GPU ms = "
                    << slim_stc_modraise_gpu_ms << "\n";

                if (!slim_modraise_raw.equal ||
                    !slim_prepare_semantic.equal ||
                    !slim_cpu_raise_semantic.equal ||
                    !slim_gpu_raise_semantic.equal)
                {
                    std::cerr
                        << "[FAILED] Slim StC-first ModRaise correctness "
                           "validation failed\n";
                    return EXIT_FAILURE;
                }
                std::cout
                    << "[OK] Slim StC q=" << slim_stc_input_q_count
                    << "->" << modraise_base_q_count
                    << " and ModRaise q=" << modraise_base_q_count
                    << "->" << q_count << " passed\n";
                if (!slim_stc_run_c2s)
                {
                    return EXIT_SUCCESS;
                }

                std::cout
                    << "\n[Slim StC-first phase 3: C2S]\n"
                    << "Calibrating the raised physical scale for the "
                       "reordered C2S input and applying C2S.\n"
                    << std::flush;

                const double slim_physical_raised_scale =
                    gpu_slim_raised.meta.scale;
                const double slim_ideal_input_factor =
                    1.0 /
                    (eval_mod_poly.k() * eval_mod_poly.sc_fac() *
                     eval_mod_poly.q_diff() *
                     static_cast<double>(bootstrap_ratio));
                const bool slim_calibrate_c2s_scale = env_flag_enabled_or(
                    "POSEIDON_BOOTSTRAP_SLIM_STC_CALIBRATE_C2S_SCALE",
                    true);
                const double slim_c2s_logical_input_scale =
                    slim_calibrate_c2s_scale
                        ? slim_physical_raised_scale * c2s_scaling /
                              slim_ideal_input_factor
                        : evalmod_scale;
                const double slim_c2s_expected_value_factor =
                    slim_physical_raised_scale /
                    slim_c2s_logical_input_scale * c2s_scaling;
                cpu_slim_raised.scale() = slim_c2s_logical_input_scale;
                gpu_slim_raised.meta.scale = slim_c2s_logical_input_scale;

                const auto slim_c2s_rescale_counts =
                    dft_stage_rescale_counts(slim_c2s_matrix_group);
                const std::size_t slim_c2s_consumed_q = std::accumulate(
                    slim_c2s_rescale_counts.begin(),
                    slim_c2s_rescale_counts.end(),
                    std::size_t{0});
                if (slim_c2s_consumed_q >= q_count)
                {
                    throw std::runtime_error(
                        "slim StC-first C2S consumes the full Q chain");
                }
                const std::size_t slim_c2s_output_q_count =
                    q_count - slim_c2s_consumed_q;
                const auto slim_c2s_output_parms_id =
                    context.crt_context()->parms_id_map().at(
                        static_cast<std::uint32_t>(
                            slim_c2s_output_q_count - 1));
                poseidon::Plaintext slim_minus_i_plain;
                encoder.encode(
                    std::complex<double>(0.0, -1.0),
                    slim_c2s_output_parms_id,
                    1.0,
                    slim_minus_i_plain);
                auto gpu_slim_minus_i_plain =
                    poseidon::gpu::GpuUploader::upload_plaintext(
                        slim_minus_i_plain,
                        device_id);

                poseidon::Ciphertext cpu_slim_c2s_real;
                poseidon::Ciphertext cpu_slim_c2s_imag;
                cpu_coeff_to_slot_rescale(
                    cpu_slim_raised,
                    slim_c2s_matrix_group,
                    cpu_slim_c2s_real,
                    cpu_slim_c2s_imag,
                    *cpu_evaluator,
                    slim_galois_keys,
                    encoder);

                poseidon::gpu::GpuCiphertextData gpu_slim_c2s_real;
                poseidon::gpu::GpuCiphertextData gpu_slim_c2s_imag;
                poseidon::gpu::GpuDoubleHoistWorkspace slim_c2s_workspace;
                poseidon::gpu::GpuCiphertextData gpu_compact_c2s_real;
                poseidon::gpu::GpuCiphertextData gpu_compact_c2s_imag;
                poseidon::gpu::GpuDoubleHoistWorkspace
                    compact_c2s_workspace;
                auto run_gpu_slim_c2s = [&]() {
                    if (slim_mode ==
                        poseidon::gpu::GpuLinearTransformMode::DoubleHoistBsgs)
                    {
                        gpu_evaluator.coeff_to_slot_double_hoist(
                            gpu_slim_raised,
                            gpu_slim_c2s_matrix_qp,
                            gpu_slim_minus_i_plain,
                            gpu_slim_galois_keys,
                            slim_c2s_workspace,
                            gpu_slim_c2s_real,
                            gpu_slim_c2s_imag);
                    }
                    else
                    {
                        gpu_evaluator.coeff_to_slot(
                            gpu_slim_raised,
                            gpu_slim_c2s_matrix,
                            gpu_slim_minus_i_plain,
                            gpu_slim_galois_keys,
                            gpu_slim_c2s_real,
                        gpu_slim_c2s_imag);
                    }
                };
                auto run_gpu_compact_c2s = [&]() {
                    gpu_evaluator.coeff_to_slot_double_hoist(
                        gpu_slim_raised,
                        compact_probe_c2s,
                        gpu_slim_minus_i_plain,
                        gpu_slim_galois_keys,
                        compact_c2s_workspace,
                        gpu_compact_c2s_real,
                        gpu_compact_c2s_imag);
                };
                run_gpu_slim_c2s();
                cudaDeviceSynchronize();

                const auto cpu_gpu_slim_c2s_real =
                    download_gpu_ciphertext(gpu_slim_c2s_real, context);
                const auto cpu_gpu_slim_c2s_imag =
                    download_gpu_ciphertext(gpu_slim_c2s_imag, context);
                RawComparison compact_c2s_qp_q;
                RawComparison compact_c2s_qp_p;
                RawComparison compact_c2s_real_raw;
                RawComparison compact_c2s_imag_raw;
                ApproxComparison compact_c2s_real_approx;
                ApproxComparison compact_c2s_imag_approx;
                if (compressed_qp_mac_probe)
                {
                    std::cout
                        << "[phase] compressed QP MAC post-ModRaise C2S "
                           "exactness\n"
                        << std::flush;
                    run_gpu_compact_c2s();
                    cudaDeviceSynchronize();
                    const auto compact_c2s_real =
                        download_gpu_ciphertext(
                            gpu_compact_c2s_real,
                            context);
                    const auto compact_c2s_imag =
                        download_gpu_ciphertext(
                            gpu_compact_c2s_imag,
                            context);
                    compact_c2s_real_raw = compare_ciphertexts(
                        cpu_gpu_slim_c2s_real,
                        compact_c2s_real,
                        8);
                    compact_c2s_imag_raw = compare_ciphertexts(
                        cpu_gpu_slim_c2s_imag,
                        compact_c2s_imag,
                        8);
                    compact_c2s_real_approx =
                        compare_decrypted_ciphertexts(
                            cpu_gpu_slim_c2s_real,
                            compact_c2s_real,
                            decryptor,
                            encoder,
                            0.0);
                    compact_c2s_imag_approx =
                        compare_decrypted_ciphertexts(
                            cpu_gpu_slim_c2s_imag,
                            compact_c2s_imag,
                            decryptor,
                            encoder,
                            0.0);

                    const auto &full_qp =
                        slim_c2s_workspace.group_accumulators;
                    const auto &compact_qp =
                        compact_c2s_workspace.group_accumulators;
                    if (full_qp.degree != compact_qp.degree ||
                        full_qp.q_count != compact_qp.q_count ||
                        full_qp.p_count != compact_qp.p_count ||
                        full_qp.batch_count != compact_qp.batch_count)
                    {
                        throw std::runtime_error(
                            "compressed C2S QP MAC workspace shape mismatch");
                    }
                    const std::size_t active_q_words =
                        full_qp.batch_count * 2 * full_qp.q_count *
                        full_qp.degree;
                    const std::size_t active_p_words =
                        full_qp.batch_count * 2 * full_qp.p_count *
                        full_qp.degree;
                    compact_c2s_qp_q = compare_device_words_exact(
                        full_qp.q,
                        compact_qp.q,
                        active_q_words,
                        "C2S compact Q MAC");
                    compact_c2s_qp_p = compare_device_words_exact(
                        full_qp.p,
                        compact_qp.p,
                        active_p_words,
                        "C2S compact P MAC");
                    if (!compact_c2s_qp_q.equal ||
                        !compact_c2s_qp_p.equal ||
                        !compact_c2s_real_raw.equal ||
                        !compact_c2s_imag_raw.equal ||
                        !compact_c2s_real_approx.equal ||
                        !compact_c2s_imag_approx.equal)
                    {
                        std::cerr
                            << "[FAILED] compressed QP MAC changed C2S "
                               "residues or output\n";
                        return EXIT_FAILURE;
                    }
                }
                const auto slim_c2s_real_cpu_gpu =
                    compare_decrypted_ciphertexts(
                        cpu_slim_c2s_real,
                        cpu_gpu_slim_c2s_real,
                        decryptor,
                        encoder,
                        slim_semantic_minimum_tolerance);
                const auto slim_c2s_imag_cpu_gpu =
                    compare_decrypted_ciphertexts(
                        cpu_slim_c2s_imag,
                        cpu_gpu_slim_c2s_imag,
                        decryptor,
                        encoder,
                        slim_semantic_minimum_tolerance);

                const auto source_values = decrypt_decode(
                    source,
                    decryptor,
                    encoder);
                std::vector<std::complex<double>> expected_c2s_real(
                    source_values.size());
                std::vector<std::complex<double>> expected_c2s_imag(
                    source_values.size());
                for (std::size_t index = 0;
                     index < source_values.size();
                     ++index)
                {
                    expected_c2s_real[index] = {
                        source_values[index].real() *
                            slim_c2s_expected_value_factor,
                        0.0};
                    expected_c2s_imag[index] = {
                        source_values[index].imag() *
                            slim_c2s_expected_value_factor,
                        0.0};
                }
                const auto slim_c2s_real_semantic = compare_approx(
                    expected_c2s_real,
                    decrypt_decode(
                        cpu_gpu_slim_c2s_real,
                        decryptor,
                        encoder),
                    slim_semantic_minimum_tolerance);
                const auto slim_c2s_imag_semantic = compare_approx(
                    expected_c2s_imag,
                    decrypt_decode(
                        cpu_gpu_slim_c2s_imag,
                        decryptor,
                        encoder),
                    slim_semantic_minimum_tolerance);

                if (gpu_slim_c2s_real.meta.q_count !=
                        slim_c2s_output_q_count ||
                    gpu_slim_c2s_imag.meta.q_count !=
                        slim_c2s_output_q_count)
                {
                    throw std::runtime_error(
                        "slim StC-first C2S q-count invariant failed");
                }

                for (std::size_t index = 0; index < warmup; ++index)
                {
                    run_gpu_slim_c2s();
                }
                cudaDeviceSynchronize();
                const double slim_c2s_gpu_ms = time_gpu_ms(
                    iterations,
                    run_gpu_slim_c2s);
                double compact_c2s_gpu_ms =
                    std::numeric_limits<double>::quiet_NaN();
                if (compressed_qp_mac_probe)
                {
                    for (std::size_t index = 0; index < warmup; ++index)
                    {
                        run_gpu_compact_c2s();
                    }
                    cudaDeviceSynchronize();
                    compact_c2s_gpu_ms = time_gpu_ms(
                        iterations,
                        run_gpu_compact_c2s);
                }
                auto run_gpu_slim_through_c2s = [&]() {
                    run_gpu_slim_stc();
                    run_gpu_slim_modraise();
                    gpu_slim_raised.meta.scale =
                        slim_c2s_logical_input_scale;
                    run_gpu_slim_c2s();
                };
                for (std::size_t index = 0; index < warmup; ++index)
                {
                    run_gpu_slim_through_c2s();
                }
                cudaDeviceSynchronize();
                const double slim_through_c2s_gpu_ms = time_gpu_ms(
                    iterations,
                    run_gpu_slim_through_c2s);

                std::cout << "\n[post-ModRaise C2S scale trace]\n";
                std::size_t slim_c2s_trace_q = q_count;
                double slim_c2s_trace_scale =
                    slim_c2s_logical_input_scale;
                const double slim_c2s_scale_floor =
                    (evalmod_scale + 1.0) / 2.0;
                for (std::size_t stage = 0;
                     stage < slim_c2s_matrix_group.data().size();
                     ++stage)
                {
                    const auto input_q_count = slim_c2s_trace_q;
                    const auto input_scale = slim_c2s_trace_scale;
                    const auto plaintext_scale =
                        slim_c2s_matrix_group.data()[stage].scale;
                    slim_c2s_trace_scale *= plaintext_scale;
                    for (std::uint32_t drop = 0;
                         drop < slim_c2s_rescale_counts[stage];
                         ++drop)
                    {
                        slim_c2s_trace_scale /= static_cast<double>(
                            first_context->coeff_modulus()
                                .at(slim_c2s_trace_q - 1)
                                .value());
                        --slim_c2s_trace_q;
                    }
                    const double stage_scale_upper =
                        slim_c2s_scale_floor * static_cast<double>(
                            first_context->coeff_modulus()
                                .at(slim_c2s_trace_q - 1)
                                .value());
                    if (slim_c2s_trace_scale < slim_c2s_scale_floor ||
                        slim_c2s_trace_scale >= stage_scale_upper)
                    {
                        throw std::runtime_error(
                            "post-ModRaise C2S output escaped the production "
                            "dynamic scale interval");
                    }
                    std::cout
                        << "stage=" << stage << " q=" << input_q_count
                        << "->" << slim_c2s_trace_q
                        << " input=2^" << std::log2(input_scale)
                        << " plain=2^" << std::log2(plaintext_scale)
                        << " drop=" << slim_c2s_rescale_counts[stage]
                        << " output=2^"
                        << std::log2(slim_c2s_trace_scale) << "\n";
                }

                std::cout
                    << "physical/logical factor = 2^"
                    << std::log2(
                           slim_physical_raised_scale /
                           slim_c2s_logical_input_scale)
                    << "\n"
                    << "logical input scale     = 2^"
                    << std::log2(slim_c2s_logical_input_scale)
                    << " (calibration="
                    << (slim_calibrate_c2s_scale ? "ON" : "OFF") << ")"
                    << "\n"
                    << "expected value factor   = "
                    << slim_c2s_expected_value_factor << "\n"
                    << "C2S output q/scale      = "
                    << gpu_slim_c2s_real.meta.q_count << "/2^"
                    << std::log2(gpu_slim_c2s_real.meta.scale) << "\n"
                    << "C2S real CPU/GPU delta  = "
                    << slim_c2s_real_cpu_gpu.max_abs_error << " (rms="
                    << slim_c2s_real_cpu_gpu.rms_error << ")\n"
                    << "C2S imag CPU/GPU delta  = "
                    << slim_c2s_imag_cpu_gpu.max_abs_error << " (rms="
                    << slim_c2s_imag_cpu_gpu.rms_error << ")\n"
                    << "roundtrip real delta    = "
                    << slim_c2s_real_semantic.max_abs_error << " (rms="
                    << slim_c2s_real_semantic.rms_error << ")\n"
                    << "roundtrip imag delta    = "
                    << slim_c2s_imag_semantic.max_abs_error << " (rms="
                    << slim_c2s_imag_semantic.rms_error << ")\n"
                    << "C2S GPU ms              = "
                    << slim_c2s_gpu_ms << "\n"
                    << "StC+raise+C2S GPU ms    = "
                    << slim_through_c2s_gpu_ms << "\n";
                if (compressed_qp_mac_probe)
                {
                    std::cout
                        << "compact Q MAC exact   = "
                        << (compact_c2s_qp_q.equal ? "YES" : "NO")
                        << "\n"
                        << "compact P MAC exact   = "
                        << (compact_c2s_qp_p.equal ? "YES" : "NO")
                        << "\n"
                        << "compact real exact    = "
                        << (compact_c2s_real_raw.equal ? "YES" : "NO")
                        << "\n"
                        << "compact imag exact    = "
                        << (compact_c2s_imag_raw.equal ? "YES" : "NO")
                        << "\n"
                        << "compact C2S GPU ms    = "
                        << compact_c2s_gpu_ms << "\n";
                }

                if (!slim_c2s_real_cpu_gpu.equal ||
                    !slim_c2s_imag_cpu_gpu.equal ||
                    !slim_c2s_real_semantic.equal ||
                    !slim_c2s_imag_semantic.equal)
                {
                    std::cerr
                        << "[FAILED] Slim StC-first post-ModRaise C2S "
                           "validation failed\n";
                    return EXIT_FAILURE;
                }
                std::cout
                    << "[OK] Slim StC->ModRaise->C2S phase passed\n";
                if (!slim_stc_evalmod_probe)
                {
                    return EXIT_SUCCESS;
                }

                std::cout
                    << "\n[Slim StC-first phase 4: EvalMod]\n"
                    << "Uploading the existing degree-"
                    << evalmod_sine_degree
                    << " dynamic EvalMod plan for the C2S q="
                    << slim_c2s_output_q_count << " output.\n"
                    << std::flush;

                const auto slim_evalmod_input_parms_id =
                    cpu_gpu_slim_c2s_real.parms_id();
                const auto slim_evalmod_input_context =
                    context.crt_context()->get_context_data(
                        slim_evalmod_input_parms_id);
                if (!slim_evalmod_input_context)
                {
                    throw std::runtime_error(
                        "slim StC-first EvalMod input level is absent");
                }
                eval_mod_poly.set_level_start(
                    static_cast<std::uint32_t>(
                        slim_evalmod_input_context->level()));

                auto gpu_slim_relin_keys =
                    poseidon::gpu::GpuUploader::upload_relin_keys(
                        relin_keys,
                        device_id);
                poseidon::gpu::GpuBootstrapData slim_evalmod_bootstrap_data;
                slim_evalmod_bootstrap_data.eval_mod =
                    poseidon::gpu::GpuUploader::
                        upload_eval_mod_high_precision(
                            eval_mod_poly,
                            encoder,
                            slim_evalmod_input_parms_id,
                            device_id,
                            &gpu_slim_relin_keys,
                            poseidon::parms_id_zero,
                            evalmod_rescale_count,
                            nullptr,
                            /*include_input_offset=*/true,
                            std::numeric_limits<std::uint32_t>::max(),
                            std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::quiet_NaN(),
                            /*fuse_leaf_terms_before_rescale=*/true,
                            cpu_gpu_slim_c2s_real.scale());
                const auto slim_evalmod_output_q_count =
                    slim_evalmod_bootstrap_data.eval_mod.output_q_count;
                const auto slim_evalmod_output_parms_id =
                    slim_evalmod_bootstrap_data.eval_mod.output_parms_id;
                if (slim_evalmod_output_q_count == 0 ||
                    slim_evalmod_output_q_count >=
                        slim_c2s_output_q_count)
                {
                    throw std::runtime_error(
                        "slim StC-first EvalMod returned an invalid output "
                        "q-count");
                }

                const auto &slim_evalmod_plan =
                    slim_evalmod_bootstrap_data.eval_mod;
                std::uint32_t slim_degree_bound = 1;
                while (slim_degree_bound < slim_evalmod_plan.polynomial_degree)
                {
                    slim_degree_bound <<= 1U;
                }
                bool slim_degree_bound_materialized = false;
                std::vector<std::uint32_t> slim_basis_degrees;
                slim_basis_degrees.reserve(slim_evalmod_plan.basis_steps.size());
                for (const auto &step : slim_evalmod_plan.basis_steps)
                {
                    slim_basis_degrees.push_back(step.output_degree);
                    slim_degree_bound_materialized |=
                        step.output_degree == slim_degree_bound;
                }
                std::vector<std::size_t> slim_node_use_counts(
                    slim_evalmod_plan.polynomial_blocks.size() +
                        slim_evalmod_plan.polynomial_combine_steps.size(),
                    0);
                for (const auto &combine :
                     slim_evalmod_plan.polynomial_combine_steps)
                {
                    if (combine.quotient_node < slim_node_use_counts.size())
                    {
                        ++slim_node_use_counts[combine.quotient_node];
                    }
                    if (combine.remainder_node < slim_node_use_counts.size())
                    {
                        ++slim_node_use_counts[combine.remainder_node];
                    }
                }
                std::size_t slim_lazy_deferred_outputs = 0;
                for (const auto &parent :
                     slim_evalmod_plan.polynomial_combine_steps)
                {
                    if (parent.remainder_node >=
                            slim_evalmod_plan.polynomial_blocks.size() &&
                        parent.remainder_node < slim_node_use_counts.size() &&
                        slim_node_use_counts[parent.remainder_node] == 1 &&
                        parent.remainder_rescale_count == 0 &&
                        parent.remainder_scale_plaintext.empty())
                    {
                        ++slim_lazy_deferred_outputs;
                    }
                }
                const bool slim_lazy_relin_enabled =
                    env_flag_enabled_or("POSEIDON_EVALMOD_LAZY_RELIN", true);
                std::cout
                    << "[Slim EvalMod execution plan]\n"
                    << "split log/baby width = "
                    << slim_evalmod_plan.polynomial_log_split << "/"
                    << (1U << slim_evalmod_plan.polynomial_log_split) << "\n"
                    << "basis degrees        = "
                    << join_q_counts(std::vector<std::size_t>(
                           slim_basis_degrees.begin(),
                           slim_basis_degrees.end()))
                    << "\n"
                    << "leaf/combine counts  = "
                    << slim_evalmod_plan.polynomial_blocks.size() << "/"
                    << slim_evalmod_plan.polynomial_combine_steps.size() << "\n"
                    << "combine relin plan   = "
                    << slim_evalmod_plan.polynomial_combine_steps.size()
                    << " -> "
                    << (slim_lazy_relin_enabled
                            ? slim_evalmod_plan.polynomial_combine_steps.size() -
                                  slim_lazy_deferred_outputs
                            : slim_evalmod_plan.polynomial_combine_steps.size())
                    << " (deferred="
                    << (slim_lazy_relin_enabled
                            ? slim_lazy_deferred_outputs
                            : 0)
                    << ")\n"
                    << "degree-bound T" << slim_degree_bound << "      = "
                    << (slim_degree_bound_materialized
                            ? "materialized"
                            : "planning-only")
                    << " (virtual_flag="
                    << (slim_evalmod_plan.polynomial_degree_bound_virtual
                            ? "ON"
                            : "OFF")
                    << ", root_q="
                    << slim_evalmod_plan.polynomial_root_anchor_q_count
                    << ")"
                    << "\n";

                if (evalmod_sine_degree == 22 &&
                    slim_evalmod_plan.polynomial_log_split == 2 &&
                    env_flag_enabled(
                        "POSEIDON_EVALMOD_VIRTUAL_DEGREE_BOUND"))
                {
                    const std::vector<std::uint32_t> expected_basis{
                        2, 3, 4, 8, 16};
                    if (slim_basis_degrees != expected_basis ||
                        slim_evalmod_plan.polynomial_blocks.size() != 6 ||
                        slim_evalmod_plan.polynomial_combine_steps.size() != 5 ||
                        (slim_lazy_relin_enabled &&
                         slim_lazy_deferred_outputs != 2) ||
                        slim_degree_bound_materialized)
                    {
                        throw std::runtime_error(
                            "slim22 baby-4 plan contains an unexpected or unused computation");
                    }

                }

                poseidon::Ciphertext cpu_slim_eval_real;
                poseidon::Ciphertext cpu_slim_eval_imag;
                std::cout
                    << "[phase] CPU EvalMod correctness oracle\n"
                    << std::flush;
                cpu_evaluator->eval_mod_high_precision(
                    cpu_gpu_slim_c2s_real,
                    cpu_slim_eval_real,
                    eval_mod_poly,
                    relin_keys,
                    encoder,
                    nullptr,
                    /*preserve_input_scale=*/true);
                cpu_evaluator->eval_mod_high_precision(
                    cpu_gpu_slim_c2s_imag,
                    cpu_slim_eval_imag,
                    eval_mod_poly,
                    relin_keys,
                    encoder,
                    nullptr,
                    /*preserve_input_scale=*/true);
                if (slim_evalmod_bootstrap_data.eval_mod.polynomial_flat_bsgs &&
                    cpu_slim_eval_real.coeff_modulus_size() >
                        slim_evalmod_output_q_count)
                {
                    cpu_evaluator->drop_modulus(
                        cpu_slim_eval_real,
                        cpu_slim_eval_real,
                        slim_evalmod_output_parms_id);
                    cpu_evaluator->drop_modulus(
                        cpu_slim_eval_imag,
                        cpu_slim_eval_imag,
                        slim_evalmod_output_parms_id);
                }

                poseidon::gpu::GpuBootstrapWorkspace slim_evalmod_workspace;
                poseidon::gpu::GpuCiphertextData gpu_slim_eval_real;
                poseidon::gpu::GpuCiphertextData gpu_slim_eval_imag;
                auto run_gpu_slim_evalmod = [&]() {
                    gpu_evaluator.eval_mod_high_precision(
                        gpu_slim_c2s_real,
                        slim_evalmod_bootstrap_data,
                        gpu_slim_relin_keys,
                        slim_evalmod_workspace,
                        gpu_slim_eval_real);
                    gpu_evaluator.eval_mod_high_precision(
                        gpu_slim_c2s_imag,
                        slim_evalmod_bootstrap_data,
                        gpu_slim_relin_keys,
                        slim_evalmod_workspace,
                        gpu_slim_eval_imag);
                };
                std::cout
                    << "[phase] GPU EvalMod correctness\n"
                    << std::flush;
                run_gpu_slim_evalmod();
                cudaDeviceSynchronize();

                const auto cpu_gpu_slim_eval_real =
                    download_gpu_ciphertext(gpu_slim_eval_real, context);
                const auto cpu_gpu_slim_eval_imag =
                    download_gpu_ciphertext(gpu_slim_eval_imag, context);
                const auto slim_eval_real_cpu_gpu =
                    compare_decrypted_ciphertexts(
                        cpu_slim_eval_real,
                        cpu_gpu_slim_eval_real,
                        decryptor,
                        encoder,
                        correctness_tolerance);
                const auto slim_eval_imag_cpu_gpu =
                    compare_decrypted_ciphertexts(
                        cpu_slim_eval_imag,
                        cpu_gpu_slim_eval_imag,
                        decryptor,
                        encoder,
                        correctness_tolerance);

                poseidon::Plaintext slim_plus_i_plain;
                encoder.encode(
                    std::complex<double>(0.0, 1.0),
                    slim_evalmod_output_parms_id,
                    1.0,
                    slim_plus_i_plain);
                auto gpu_slim_plus_i_plain =
                    poseidon::gpu::GpuUploader::upload_plaintext(
                        slim_plus_i_plain,
                        device_id);

                poseidon::Ciphertext cpu_slim_scaled_imag;
                poseidon::Ciphertext cpu_slim_final;
                cpu_evaluator->multiply_const(
                    cpu_slim_eval_imag,
                    std::complex<double>(0.0, 1.0),
                    1.0,
                    cpu_slim_scaled_imag,
                    encoder);
                cpu_evaluator->add(
                    cpu_slim_eval_real,
                    cpu_slim_scaled_imag,
                    cpu_slim_final);
                if (bootstrap_ratio > 1)
                {
                    cpu_evaluator->multiply_const_direct(
                        cpu_slim_final,
                        static_cast<int>(bootstrap_ratio),
                        cpu_slim_final,
                        encoder);
                }

                poseidon::gpu::GpuCiphertextData gpu_slim_scaled_imag;
                poseidon::gpu::GpuCiphertextData gpu_slim_combined;
                poseidon::gpu::GpuCiphertextData gpu_slim_final;
                auto combine_gpu_slim_evalmod = [&]() {
                    gpu_evaluator.multiply_plain(
                        gpu_slim_eval_imag,
                        gpu_slim_plus_i_plain,
                        gpu_slim_scaled_imag);
                    gpu_evaluator.add(
                        gpu_slim_eval_real,
                        gpu_slim_scaled_imag,
                        gpu_slim_combined);
                    if (bootstrap_ratio > 1)
                    {
                        gpu_evaluator.multiply_scalar(
                            gpu_slim_combined,
                            bootstrap_ratio,
                            gpu_slim_final);
                    }
                    else
                    {
                        gpu_slim_final = std::move(gpu_slim_combined);
                    }
                };
                combine_gpu_slim_evalmod();
                cudaDeviceSynchronize();
                const auto cpu_gpu_slim_final =
                    download_gpu_ciphertext(gpu_slim_final, context);
                const auto slim_final_cpu_gpu =
                    compare_decrypted_ciphertexts(
                        cpu_slim_final,
                        cpu_gpu_slim_final,
                        decryptor,
                        encoder,
                        correctness_tolerance);
                const auto slim_final_source =
                    compare_decrypted_ciphertexts(
                        source,
                        cpu_gpu_slim_final,
                        decryptor,
                        encoder,
                        slim_semantic_minimum_tolerance);

                // Decompose the end-to-end error without introducing another
                // encrypted computation.  The CosDiscrete polynomial is a
                // Chebyshev approximation to the scaled sine map below.  By
                // evaluating both maps on the decoded C2S inputs, the report
                // separates input/linear-transform error, the inherent
                // sin(z) versus z error, polynomial approximation error, and
                // homomorphic arithmetic error.
                const auto slim_c2s_real_values = decrypt_decode(
                    cpu_gpu_slim_c2s_real,
                    decryptor,
                    encoder);
                const auto slim_c2s_imag_values = decrypt_decode(
                    cpu_gpu_slim_c2s_imag,
                    decryptor,
                    encoder);
                const auto slim_final_values = decrypt_decode(
                    cpu_gpu_slim_final,
                    decryptor,
                    encoder);
                if (slim_c2s_real_values.size() != source_values.size() ||
                    slim_c2s_imag_values.size() != source_values.size())
                {
                    throw std::runtime_error(
                        "slim EvalMod plaintext diagnostic slot-count mismatch");
                }

                std::vector<std::complex<double>> slim_ideal_c2s_real(
                    source_values.size());
                std::vector<std::complex<double>> slim_ideal_c2s_imag(
                    source_values.size());
                std::vector<std::complex<double>> slim_plain_poly_final(
                    source_values.size());
                std::vector<std::complex<double>> slim_ideal_actual_final(
                    source_values.size());
                std::vector<std::complex<double>> slim_ideal_linear_final(
                    source_values.size());
                double slim_max_actual_sine_argument = 0.0;
                double slim_max_ideal_sine_argument = 0.0;
                const double two_pi_k =
                    2.0 * std::acos(-1.0) * eval_mod_poly.k() *
                    eval_mod_poly.sc_fac();
                for (std::size_t index = 0;
                     index < source_values.size();
                     ++index)
                {
                    slim_ideal_c2s_real[index] =
                        source_values[index].real() *
                        slim_ideal_input_factor;
                    slim_ideal_c2s_imag[index] =
                        source_values[index].imag() *
                        slim_ideal_input_factor;

                    const auto polynomial_real =
                        evaluate_evalmod_polynomial_plain(
                            eval_mod_poly,
                            slim_c2s_real_values[index]);
                    const auto polynomial_imag =
                        evaluate_evalmod_polynomial_plain(
                            eval_mod_poly,
                            slim_c2s_imag_values[index]);
                    const auto ideal_actual_real =
                        evaluate_evalmod_ideal_sine(
                            eval_mod_poly,
                            slim_c2s_real_values[index]);
                    const auto ideal_actual_imag =
                        evaluate_evalmod_ideal_sine(
                            eval_mod_poly,
                            slim_c2s_imag_values[index]);
                    const auto ideal_linear_real =
                        evaluate_evalmod_ideal_sine(
                            eval_mod_poly,
                            slim_ideal_c2s_real[index]);
                    const auto ideal_linear_imag =
                        evaluate_evalmod_ideal_sine(
                            eval_mod_poly,
                            slim_ideal_c2s_imag[index]);
                    slim_plain_poly_final[index] =
                        static_cast<double>(bootstrap_ratio) *
                        (polynomial_real +
                         std::complex<double>(0.0, 1.0) * polynomial_imag);
                    slim_ideal_actual_final[index] =
                        static_cast<double>(bootstrap_ratio) *
                        (ideal_actual_real +
                         std::complex<double>(0.0, 1.0) * ideal_actual_imag);
                    slim_ideal_linear_final[index] =
                        static_cast<double>(bootstrap_ratio) *
                        (ideal_linear_real +
                         std::complex<double>(0.0, 1.0) * ideal_linear_imag);
                    slim_max_actual_sine_argument = std::max(
                        slim_max_actual_sine_argument,
                        std::max(
                            std::abs(two_pi_k * slim_c2s_real_values[index]),
                            std::abs(two_pi_k * slim_c2s_imag_values[index])));
                    slim_max_ideal_sine_argument = std::max(
                        slim_max_ideal_sine_argument,
                        std::max(
                            std::abs(two_pi_k * slim_ideal_c2s_real[index]),
                            std::abs(two_pi_k * slim_ideal_c2s_imag[index])));
                }
                const auto slim_c2s_real_ideal = compare_approx(
                    slim_ideal_c2s_real,
                    slim_c2s_real_values,
                    slim_semantic_minimum_tolerance);
                const auto slim_c2s_imag_ideal = compare_approx(
                    slim_ideal_c2s_imag,
                    slim_c2s_imag_values,
                    slim_semantic_minimum_tolerance);
                const auto slim_plain_predicted_total = compare_approx(
                    source_values,
                    slim_plain_poly_final,
                    slim_semantic_minimum_tolerance);
                const auto slim_homomorphic_arithmetic = compare_approx(
                    slim_plain_poly_final,
                    slim_final_values,
                    slim_semantic_minimum_tolerance);
                const auto slim_polynomial_approximation = compare_approx(
                    slim_ideal_actual_final,
                    slim_plain_poly_final,
                    slim_semantic_minimum_tolerance);
                const auto slim_linear_pipeline_effect = compare_approx(
                    slim_ideal_linear_final,
                    slim_ideal_actual_final,
                    slim_semantic_minimum_tolerance);
                const auto slim_sine_linearity = compare_approx(
                    source_values,
                    slim_ideal_linear_final,
                    slim_semantic_minimum_tolerance);

                if (gpu_slim_eval_real.meta.q_count !=
                        slim_evalmod_output_q_count ||
                    gpu_slim_eval_imag.meta.q_count !=
                        slim_evalmod_output_q_count ||
                    gpu_slim_final.meta.q_count !=
                        slim_evalmod_output_q_count)
                {
                    throw std::runtime_error(
                        "slim StC-first EvalMod q-count invariant failed");
                }

                for (std::size_t index = 0; index < warmup; ++index)
                {
                    run_gpu_slim_evalmod();
                }
                cudaDeviceSynchronize();
                const double slim_evalmod_gpu_ms = time_gpu_ms(
                    iterations,
                    run_gpu_slim_evalmod);
                auto run_gpu_slim_full = [&]() {
                    nvtxRangePushA("bootstrap_once");
                    nvtxRangePushA("StC");
                    run_gpu_slim_stc();
                    nvtxRangePop();
                    nvtxRangePushA("ModRaise");
                    run_gpu_slim_modraise();
                    nvtxRangePop();
                    gpu_slim_raised.meta.scale =
                        slim_c2s_logical_input_scale;
                    nvtxRangePushA("CtS");
                    run_gpu_slim_c2s();
                    nvtxRangePop();
                    nvtxRangePushA("EvalMod");
                    run_gpu_slim_evalmod();
                    combine_gpu_slim_evalmod();
                    nvtxRangePop();
                    nvtxRangePop();
                };
                for (std::size_t index = 0; index < full_warmup; ++index)
                {
                    run_gpu_slim_full();
                }
                cudaDeviceSynchronize();
                double slim_full_gpu_ms =
                    std::numeric_limits<double>::quiet_NaN();
                if (nsys_capture_full)
                {
                    std::cout
                        << "[phase] nsys capture: one profiled slim StC-first "
                           "GPU bootstrap after warmup\n"
                        << std::flush;
                    const auto capture_start_status = cudaProfilerStart();
                    if (capture_start_status != cudaSuccess)
                    {
                        throw std::runtime_error(
                            std::string("cudaProfilerStart failed: ") +
                            cudaGetErrorString(capture_start_status));
                    }
                    nvtxRangePushA(
                        "profiled_slim_stc_first_bootstrap_once");
                    const auto start = std::chrono::steady_clock::now();
                    run_gpu_slim_full();
                    cudaDeviceSynchronize();
                    const auto stop = std::chrono::steady_clock::now();
                    nvtxRangePop();
                    const auto capture_stop_status = cudaProfilerStop();
                    if (capture_stop_status != cudaSuccess)
                    {
                        throw std::runtime_error(
                            std::string("cudaProfilerStop failed: ") +
                            cudaGetErrorString(capture_stop_status));
                    }
                    const auto elapsed_us =
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            stop - start).count();
                    slim_full_gpu_ms =
                        static_cast<double>(elapsed_us) / 1000.0;
                }
                else
                {
                    slim_full_gpu_ms = time_gpu_ms(
                        iterations,
                        run_gpu_slim_full);
                }

                std::cout
                    << "EvalMod q              = "
                    << slim_c2s_output_q_count << "->"
                    << slim_evalmod_output_q_count << " (consumed="
                    << (slim_c2s_output_q_count -
                        slim_evalmod_output_q_count) << ")\n"
                    << "EvalMod output scale   = 2^"
                    << std::log2(gpu_slim_eval_real.meta.scale) << "\n"
                    << "EvalMod real CPU/GPU   = "
                    << slim_eval_real_cpu_gpu.max_abs_error << " (rms="
                    << slim_eval_real_cpu_gpu.rms_error << ")\n"
                    << "EvalMod imag CPU/GPU   = "
                    << slim_eval_imag_cpu_gpu.max_abs_error << " (rms="
                    << slim_eval_imag_cpu_gpu.rms_error << ")\n"
                    << "final CPU/GPU delta    = "
                    << slim_final_cpu_gpu.max_abs_error << " (rms="
                    << slim_final_cpu_gpu.rms_error << ")\n"
                    << "final source error     = "
                    << slim_final_source.max_abs_error << " (rms="
                    << slim_final_source.rms_error << ")\n"
                    << "\n[EvalMod plaintext error decomposition]\n"
                    << "ideal C2S factor       = "
                    << slim_ideal_input_factor << "\n"
                    << "measured C2S factor    = "
                    << slim_c2s_expected_value_factor << "\n"
                    << "C2S->ideal real delta  = "
                    << slim_c2s_real_ideal.max_abs_error << " (rms="
                    << slim_c2s_real_ideal.rms_error << ")\n"
                    << "C2S->ideal imag delta  = "
                    << slim_c2s_imag_ideal.max_abs_error << " (rms="
                    << slim_c2s_imag_ideal.rms_error << ")\n"
                    << "max sine argument      = "
                    << slim_max_actual_sine_argument << " (ideal="
                    << slim_max_ideal_sine_argument << ")\n"
                    << "sine nonlinearity      = "
                    << slim_sine_linearity.max_abs_error << " (rms="
                    << slim_sine_linearity.rms_error << ")\n"
                    << "linear pipeline effect = "
                    << slim_linear_pipeline_effect.max_abs_error << " (rms="
                    << slim_linear_pipeline_effect.rms_error << ")\n"
                    << "polynomial approx      = "
                    << slim_polynomial_approximation.max_abs_error << " (rms="
                    << slim_polynomial_approximation.rms_error << ")\n"
                    << "HE arithmetic          = "
                    << slim_homomorphic_arithmetic.max_abs_error << " (rms="
                    << slim_homomorphic_arithmetic.rms_error << ")\n"
                    << "plaintext predicted    = "
                    << slim_plain_predicted_total.max_abs_error << " (rms="
                    << slim_plain_predicted_total.rms_error << ")\n"
                    << "final target/minimum   = "
                    << slim_semantic_target_tolerance << "/"
                    << slim_semantic_minimum_tolerance << "\n"
                    << "EvalMod GPU ms         = "
                    << slim_evalmod_gpu_ms << "\n"
                    << "full slim bootstrap ms = "
                    << slim_full_gpu_ms << "\n";

                if (!slim_eval_real_cpu_gpu.equal ||
                    !slim_eval_imag_cpu_gpu.equal ||
                    !slim_final_cpu_gpu.equal)
                {
                    std::cerr
                        << "[FAILED] Slim StC-first GPU/CPU EvalMod "
                           "validation failed\n";
                    return EXIT_FAILURE;
                }
                if (!slim_final_source.equal &&
                    !ignore_correctness_failure)
                {
                    std::cerr
                        << "[FAILED] Slim StC-first end-to-end semantic "
                           "validation failed\n";
                    return EXIT_FAILURE;
                }
                if (!slim_final_source.equal)
                {
                    std::cout
                        << "[WARN] Known end-to-end semantic error ignored "
                           "because "
                           "POSEIDON_BOOTSTRAP_IGNORE_CORRECTNESS_FAILURE=1; "
                           "GPU/CPU arithmetic checks remain mandatory.\n";
                }
                if (slim_final_source.max_abs_error >
                    slim_semantic_target_tolerance)
                {
                    std::cout
                        << "[WARN] Slim StC-first full bootstrap meets the "
                           "minimum precision but misses the target "
                           "precision.\n";
                }
                std::cout
                    << "[OK] Slim StC-first full bootstrap passed\n";
                return EXIT_SUCCESS;
            }
            std::cout
                << "\n[OK] Slim StC-first phase 1 probe passed scale-range, "
                   "input-equivalence, and CPU/GPU checks for q="
                << slim_stc_input_q_count << "->"
                << modraise_base_q_count << " CPU/GPU validation\n";
            return EXIT_SUCCESS;
        }
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

        if (!gpu_only_timing)
        {
            for (std::size_t i = 0; i < warmup; ++i)
            {
                (void)cpu_bootstrap_prepare_and_raise(
                    source,
                    *cpu_evaluator,
                    context,
                    encoder,
                    target_q0_scale);
            }
        }

        double cpu_ms = std::numeric_limits<double>::quiet_NaN();
        poseidon::Ciphertext cpu_timing_sink;
        if (!gpu_only_timing)
        {
            cpu_ms = time_cpu_ms(iterations, [&]() {
                cpu_timing_sink =
                    cpu_bootstrap_prepare_and_raise(
                        source,
                        *cpu_evaluator,
                        context,
                        encoder,
                        target_q0_scale);
            });
        }

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
        const double q0_product_scale = context.crt_context()->q0();
        // The mixed-prime path uses the GS logical contract: normal CKKS
        // computation runs at 2^40 while ModRaise presents a 2^45 working
        // scale to C2S/EvalMod. The physical two-prime q0 product remains an
        // implementation detail and is compensated by the DFT coefficients.
        const double raised_scale = evalmod_dynamic_rescale
            ? evalmod_scale
            : q0_product_scale;
        const double post_raise_multiplier = evalmod_dynamic_rescale
            ? 1.0
            : std::round(
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
            q0_product_scale);
        cpu_bootstrapper.generate_linear_coefficients();
        const double c2s_input_scale = raised_scale *
            (post_raise_multiplier > 1.0 ? post_raise_multiplier : 1.0);
        const double c2s_value_normalization =
            1.0;
        std::cout << "[phase] create CoeffToSlot matrix group"
                  << " physical_raised_log2_scale="
                  << std::log2(gpu_raised.meta.scale)
                  << " logical_input_log2_scale="
                  << std::log2(c2s_input_scale)
                  << " normalization=2^"
                  << std::log2(c2s_value_normalization)
                  << "\n" << std::flush;
        auto c2s_matrix_group = evalmod_dynamic_rescale
            ? make_coeff_to_slot_matrix_group(
                  context,
                  encoder,
                  c2s_scaling,
                  c2s_log_bsgs_ratio,
                  c2s_step,
                  c2s_input_scale,
                  evalmod_scale,
                  c2s_value_normalization)
            : cpu_bootstrapper.create_coeff_to_slot_matrix_group(
                context.crt_context()->first_parms_id(),
                c2s_input_scale,
                c2s_log_bsgs_ratio,
                evalmod_scale);
        const auto c2s_stage_rescale_counts =
            dft_stage_rescale_counts(c2s_matrix_group);
        const std::size_t c2s_consumed_q = std::accumulate(
            c2s_stage_rescale_counts.begin(),
            c2s_stage_rescale_counts.end(),
            std::size_t{0});
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
        std::cout << "[phase] staged CPU CoeffToSlot setup probe\n" << std::flush;
        cpu_coeff_to_slot_rescale(
            cpu_full_raised,
            c2s_matrix_group,
            cpu_c2s_real_raw,
            cpu_c2s_imag_raw,
            *cpu_evaluator,
            c2s_setup_galois_keys,
            encoder);

        const double expected_c2s_output_scale = planned_dft_output_scale(
            context,
            c2s_input_scale,
            c2s_matrix_group);
        if (cpu_c2s_real_raw.parms_id() != cpu_c2s_imag_raw.parms_id() ||
            std::abs(
                std::log2(cpu_c2s_real_raw.scale()) -
                std::log2(cpu_c2s_imag_raw.scale())) > 1.0e-9 ||
            std::abs(
                std::log2(cpu_c2s_real_raw.scale()) -
                std::log2(expected_c2s_output_scale)) > 1.0e-9)
        {
            throw std::runtime_error(
                "CoeffToSlot did not produce the expected EvalMod input scale");
        }

        poseidon::Ciphertext cpu_c2s_real = cpu_c2s_real_raw;
        poseidon::Ciphertext cpu_c2s_imag = cpu_c2s_imag_raw;

        const auto evalmod_input_parms_id = cpu_c2s_real.parms_id();
        const auto evalmod_input_context =
            context.crt_context()->get_context_data(evalmod_input_parms_id);
        if (!evalmod_input_context)
        {
            throw std::runtime_error(
                "fused CoeffToSlot output parms_id is absent from the context");
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
        poseidon::Ciphertext cpu_evalmod_probe_output;
        poseidon::BootstrapEvalModTrace cpu_evalmod_probe_trace;
        poseidon::parms_id_type cpu_evalmod_output_parms_id =
            poseidon::parms_id_zero;
        double polynomial_output_scale_override =
            std::numeric_limits<double>::quiet_NaN();
        if (!evalmod_dynamic_rescale)
        {
            poseidon::Plaintext cpu_evalmod_probe_plain;
            std::cout << "[phase] CPU EvalMod setup probe encode\n" << std::flush;
            encoder.encode(
                std::complex<double>(0.125, 0.0),
                evalmod_input_parms_id,
                evalmod_scale,
                cpu_evalmod_probe_plain);
            poseidon::Ciphertext cpu_evalmod_probe_input;
            encryptor.encrypt(cpu_evalmod_probe_plain, cpu_evalmod_probe_input);
            cpu_evalmod_probe_input.scale() = evalmod_scale;
            std::cout << "[phase] CPU EvalMod setup probe execute\n" << std::flush;
            cpu_bootstrapper.eval_mod(
                cpu_evalmod_probe_input,
                cpu_evalmod_probe_output,
                relin_keys,
                evalmod_double_angle,
                bootstrap_inverse_coefficient,
                evalmod_scale,
                &cpu_evalmod_probe_trace);
            cpu_evalmod_output_parms_id =
                cpu_evalmod_probe_output.parms_id();
            const auto cpu_evalmod_output_context =
                context.crt_context()->get_context_data(
                    cpu_evalmod_output_parms_id);
            if (!cpu_evalmod_output_context)
            {
                throw std::runtime_error(
                    "CPU EvalMod setup probe returned an unknown output parms_id");
            }
            polynomial_output_scale_override =
                cpu_evalmod_probe_trace.polynomial_output.scale();
        }

        const bool fused_leaf_ab_enabled =
            env_flag_enabled("POSEIDON_BOOTSTRAP_FUSED_LEAF_AB");
        std::cout << "[phase] upload GPU EvalMod dynamic/static plan\n" << std::flush;
        auto gpu_evalmod_data =
            poseidon::gpu::GpuUploader::upload_eval_mod_high_precision(
                eval_mod_poly,
                encoder,
                evalmod_input_parms_id,
                device_id,
                &gpu_relin_keys,
                evalmod_dynamic_rescale
                    ? poseidon::parms_id_zero
                    : cpu_evalmod_output_parms_id,
                evalmod_rescale_count,
                evalmod_dynamic_rescale
                    ? nullptr
                    : &bootstrap_evalmod_polynomial,
                /*include_input_offset=*/evalmod_dynamic_rescale,
                evalmod_dynamic_rescale
                    ? std::numeric_limits<std::uint32_t>::max()
                    : evalmod_double_angle,
                evalmod_dynamic_rescale
                    ? std::numeric_limits<double>::quiet_NaN()
                    : bootstrap_double_angle_base,
                polynomial_output_scale_override,
                /*fuse_leaf_terms_before_rescale=*/true,
                cpu_c2s_real.scale());
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
                    evalmod_dynamic_rescale
                        ? poseidon::parms_id_zero
                        : cpu_evalmod_output_parms_id,
                    evalmod_rescale_count,
                    evalmod_dynamic_rescale
                        ? nullptr
                        : &bootstrap_evalmod_polynomial,
                    /*include_input_offset=*/evalmod_dynamic_rescale,
                    evalmod_dynamic_rescale
                        ? std::numeric_limits<std::uint32_t>::max()
                        : evalmod_double_angle,
                    evalmod_dynamic_rescale
                        ? std::numeric_limits<double>::quiet_NaN()
                        : bootstrap_double_angle_base,
                    polynomial_output_scale_override,
                    /*fuse_leaf_terms_before_rescale=*/false,
                    cpu_c2s_real.scale());
        }
        const auto evalmod_output_q_count =
            gpu_evalmod_data.output_q_count;
        const auto evalmod_output_parms_id =
            gpu_evalmod_data.output_parms_id;
        const auto evalmod_output_context =
            context.crt_context()->get_context_data(evalmod_output_parms_id);
        if (!evalmod_output_context ||
            (!evalmod_dynamic_rescale &&
             evalmod_output_parms_id != cpu_evalmod_output_parms_id))
        {
            throw std::runtime_error(
                "GPU EvalMod setup did not preserve the CPU output parms_id");
        }

        // Fuse the exact q0 / 2^scaling_log correction into the inverse DFT
        // plaintexts. This is the same zero-runtime-cost normalization used
        // by the verified CPU Bootstrapper path.
        // EvalMod's exact post-rescale scale depends on the concrete primes.
        // Reinterpret it at the logical 2^evalmod_log_scale scale before S2C
        // and compensate the resulting value change in the final S2C matrix.
        // This keeps every S2C matrix multiplication near the logical scale
        // without spending another pair of physical modulus primes.
        const double evalmod_setup_output_scale =
            gpu_evalmod_data.output_scale > 0.0
                ? gpu_evalmod_data.output_scale
                : cpu_evalmod_probe_output.scale();
        const double s2c_input_scale = evalmod_dynamic_rescale
            ? evalmod_setup_output_scale
            : evalmod_scale;
        const double s2c_scale_compensation =
            s2c_input_scale / evalmod_setup_output_scale;
        const double s2c_normalization = evalmod_dynamic_rescale
            ? s2c_scaling
            : raised_scale / evalmod_scale * s2c_scale_compensation;
        std::cout << "[phase] create SlotToCoeff matrix group"
                  << " input_log2_scale=" << std::log2(s2c_input_scale)
                  << " evalmod_output_log2_scale="
                  << std::log2(evalmod_setup_output_scale)
                  << " normalization=" << s2c_normalization << "\n"
                  << std::flush;
        auto s2c_matrix_group = evalmod_dynamic_rescale
            ? make_slot_to_coeff_matrix_group(
                  context,
                  encoder,
                  static_cast<std::uint32_t>(evalmod_output_context->level()),
                  s2c_normalization,
                  s2c_log_bsgs_ratio,
                  s2c_step,
                  s2c_input_scale,
                  evalmod_scale)
            : cpu_bootstrapper.create_slot_to_coeff_matrix_group(
                  evalmod_output_parms_id,
                  s2c_input_scale,
                  s2c_normalization,
                  s2c_log_bsgs_ratio);
        if (evalmod_dynamic_rescale)
        {
            const double raw_s2c_output_scale = planned_dft_output_scale(
                context,
                s2c_input_scale,
                s2c_matrix_group);
            const double output_scale_correction =
                evalmod_scale / raw_s2c_output_scale;
            std::cout << "[phase] SlotToCoeff output normalization"
                      << " raw_log2_scale=" << std::log2(raw_s2c_output_scale)
                      << " correction=2^" << std::log2(output_scale_correction)
                      << "\n" << std::flush;
            s2c_matrix_group = make_slot_to_coeff_matrix_group(
                context,
                encoder,
                static_cast<std::uint32_t>(evalmod_output_context->level()),
                s2c_normalization,
                s2c_log_bsgs_ratio,
                s2c_step,
                s2c_input_scale,
                evalmod_scale,
                output_scale_correction);
        }
        const auto s2c_stage_rescale_counts =
            dft_stage_rescale_counts(s2c_matrix_group);
        const std::size_t s2c_consumed_q = std::accumulate(
            s2c_stage_rescale_counts.begin(),
            s2c_stage_rescale_counts.end(),
            std::size_t{0});
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
        poseidon::Ciphertext cpu_library_bootstrap_result;
        if (!skip_library_oracle)
        {
            poseidon::GaloisKeys cpu_library_galois_keys;
            keygen.create_galois_keys(cpu_library_galois_keys);
            cpu_evaluator->bootstrap(
                source,
                cpu_library_bootstrap_result,
                relin_keys,
                cpu_library_galois_keys,
                encoder,
                cpu_library_bootstrap_config);
        }
        else
        {
            std::cout
                << "CPU production-library oracle = skipped by "
                   "POSEIDON_BOOTSTRAP_SKIP_LIBRARY_ORACLE; staged CPU/GPU "
                   "and source-message checks remain enabled\n";
        }

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
            true,
            c2s_stage_rescale_counts);
        const auto s2c_key_q_counts = required_dft_key_q_counts(
            evalmod_output_q_count,
            s2c_matrix_group.data().size(),
            s2c_matrix_group.step(),
            false,
            s2c_stage_rescale_counts);
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
        // The inverse scale change is fused into the final S2C matrix above,
        // so restoring this logical scale consumes no modulus level.
        bootstrap_data.slot_to_coeff_input_scale = s2c_input_scale;
        bootstrap_data.project_real = false;
        bootstrap_data.output_ratio = bootstrap_ratio;
        const double s2c_output_scale = evalmod_dynamic_rescale
            ? evalmod_scale
            : raised_scale * context.parameters_literal()->scale() /
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
        std::cout << "lazy relin       = "
                  << (env_flag_enabled_or("POSEIDON_EVALMOD_LAZY_RELIN", true)
                          ? "ON"
                          : "OFF")
                  << ", effective_degree="
                  << bootstrap_data.eval_mod.polynomial_degree
                  << ", basis="
                  << (bootstrap_data.eval_mod.polynomial_basis ==
                              poseidon::gpu::GpuEvalModPolynomialBasis::Chebyshev
                          ? "Chebyshev"
                          : "Monomial")
                  << "\n";
        std::cout << "evalmod split    = log_split="
                  << bootstrap_data.eval_mod.polynomial_log_split
                  << ", baby_width="
                  << (std::uint64_t{1}
                      << bootstrap_data.eval_mod.polynomial_log_split)
                  << ", layout="
                  << (bootstrap_data.eval_mod.polynomial_flat_bsgs
                          ? "flat_b8"
                          : "recursive")
                  << "\n";
        std::cout << "basis steps       = "
                  << bootstrap_data.eval_mod.basis_steps.size() << "\n";
        if (bootstrap_data.eval_mod.dynamic_rescale)
        {
            std::cout << "dynamic min scale = "
                      << bootstrap_data.eval_mod.dynamic_min_scale << "\n";
            std::cout << "basis rescale pat = ";
            for (std::size_t i = 0;
                 i < bootstrap_data.eval_mod.basis_steps.size();
                 ++i)
            {
                const auto &step = bootstrap_data.eval_mod.basis_steps[i];
                if (i != 0)
                {
                    std::cout << ",";
                }
                std::cout << "T" << step.output_degree
                          << ":" << step.rescale_count;
            }
            std::cout << "\n";
            if (detailed_diagnostics)
            {
                std::cout << "basis scale plan= ";
                for (std::size_t i = 0;
                     i < bootstrap_data.eval_mod.basis_steps.size();
                     ++i)
                {
                    const auto &step = bootstrap_data.eval_mod.basis_steps[i];
                    if (i != 0)
                    {
                        std::cout << ",";
                    }
                    std::cout << "T" << step.output_degree
                              << "(" << step.left_degree << "*"
                              << step.right_degree;
                    if (step.correction_degree != 0)
                    {
                        std::cout << "-T" << step.correction_degree;
                    }
                    else
                    {
                        std::cout << "-1";
                    }
                    std::cout << "):pre="
                              << std::fixed << std::setprecision(3)
                              << std::log2(step.pre_rescale_scale)
                              << ",out="
                              << std::log2(step.output_scale);
                    std::cout.unsetf(std::ios::floatfield);
                }
                std::cout << "\n";
            }
            std::cout << "double rescale pat= ";
            for (std::size_t i = 0;
                 i < bootstrap_data.eval_mod.double_angle_rescale_counts.size();
                 ++i)
            {
                if (i != 0)
                {
                    std::cout << ",";
                }
                std::cout
                    << bootstrap_data.eval_mod.double_angle_rescale_counts[i];
            }
            std::cout << "\n";
        }
        std::cout << "BSGS leaf blocks  = "
                  << bootstrap_data.eval_mod.polynomial_blocks.size() << "\n";
        std::cout << "BSGS combines     = "
                  << bootstrap_data.eval_mod.polynomial_combine_steps.size() << "\n";
        if (bootstrap_data.eval_mod.dynamic_rescale && detailed_diagnostics)
        {
            std::cout << "combine scale plan= ";
            for (std::size_t i = 0;
                 i < bootstrap_data.eval_mod.polynomial_combine_steps.size();
                 ++i)
            {
                const auto &step =
                    bootstrap_data.eval_mod.polynomial_combine_steps[i];
                if (i != 0)
                {
                    std::cout << ",";
                }
                std::cout << "#" << i
                          << "(out_node=" << step.output_node
                          << ",qnode=" << step.quotient_node
                          << ",rnode=" << step.remainder_node
                          << ",T" << step.basis_degree
                          << "):qr=" << step.quotient_rescale_count
                          << ",rr=" << step.remainder_rescale_count
                          << ",prod_q=" << step.product_q_count
                          << ",out_q=" << step.output_q_count
                          << ",prod="
                          << std::fixed << std::setprecision(3)
                          << std::log2(step.product_scale);
                if (step.product_aligned_scale > 0.0)
                {
                    std::cout << "->"
                              << std::log2(step.product_aligned_scale);
                }
                if (step.remainder_aligned_scale > 0.0)
                {
                    std::cout << ",rem->"
                              << std::log2(step.remainder_aligned_scale);
                }
                std::cout << ",out=" << std::log2(step.output_scale);
                std::cout.unsetf(std::ios::floatfield);
            }
            std::cout << "\n";
        }
        std::cout << "leaf accumulation = fused MAC, one rescale per leaf\n";
        if (bootstrap_data.eval_mod.dynamic_rescale && detailed_diagnostics)
        {
            std::cout << "leaf scale plan   = ";
            for (std::size_t i = 0;
                 i < bootstrap_data.eval_mod.polynomial_blocks.size();
                 ++i)
            {
                const auto &block =
                    bootstrap_data.eval_mod.polynomial_blocks[i];
                if (i != 0)
                {
                    std::cout << ",";
                }
                std::cout << "#" << i
                          << "(out_q=" << block.output_q_count
                          << ",out="
                          << std::fixed << std::setprecision(3)
                          << std::log2(block.output_scale)
                          << ",terms=";
                bool first_term = true;
                for (const auto &term : block.terms)
                {
                    if (!first_term)
                    {
                        std::cout << "/";
                    }
                    first_term = false;
                    std::cout << "d" << term.degree
                              << "@q"
                              << term.coefficient_plaintext.meta.q_count
                              << ":p="
                              << std::log2(
                                     term.coefficient_plaintext.meta.scale);
                }
                std::cout << ")";
                std::cout.unsetf(std::ios::floatfield);
            }
            std::cout << "\n";
        }
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
                  << join_q_counts(std::vector<std::size_t>(
                         c2s_stage_rescale_counts.begin(),
                         c2s_stage_rescale_counts.end()))
                  << " prime(s), total=" << c2s_consumed_q << "\n";
        if (c2s_matrix_group.rescale_min_scale() > 0.0)
        {
            std::cout << "C2S plaintext scale = 2^"
                      << std::log2(c2s_matrix_group.rescale_min_scale())
                      << " with GS min_scale/2 dynamic drops\n";
        }
        else
        {
            std::cout << "C2S scale align   = fused into final matrix (0 extra prime(s))\n";
        }
        std::cout << "EvalMod rescale   = "
                  << evalmod_rescale_count << " prime(s)\n";
        std::cout << "S2C rescale width = "
                  << join_q_counts(std::vector<std::size_t>(
                         s2c_stage_rescale_counts.begin(),
                         s2c_stage_rescale_counts.end()))
                  << " prime(s), total=" << s2c_consumed_q << "\n";
        std::cout << "EvalMod input q   = "
                  << cpu_c2s_real.coeff_modulus_size() << "\n";
        std::cout << "EvalMod output q  = "
                  << evalmod_output_q_count << "\n";
        std::cout << "EvalMod out scale = "
                  << evalmod_setup_output_scale << "\n";
        std::cout << "CPU EvalMod out q = ";
        if (cpu_evalmod_probe_output.is_valid())
        {
            std::cout << cpu_evalmod_probe_output.coeff_modulus_size();
        }
        else
        {
            std::cout << "N/A (GPU dynamic setup plan)";
        }
        std::cout << "\n";
        std::cout << "Relin key views   = "
                  << join_q_counts(
                         bootstrap_data.eval_mod.required_relin_q_counts)
                  << "\n";
        std::cout << "GPU setup objects = one C2S matrix group, one S2C matrix group, "
                     "one Galois-key set\n";
        if (setup_only)
        {
            std::cout << "\n[OK] Bootstrap setup-only path generated and uploaded "
                         "matrices, keys, and EvalMod plan\n";
            return EXIT_SUCCESS;
        }

        // Recompute the staged CPU C2S values with the exact same randomized
        // Galois keys used by the GPU upload. The earlier setup-only key set
        // is sufficient for deriving levels/scales but cannot be raw-RNS
        // compared with a transform evaluated under another key ciphertext.
        std::cout << "[phase] staged CPU CoeffToSlot correctness\n" << std::flush;
        cpu_coeff_to_slot_rescale(
            cpu_full_raised,
            c2s_matrix_group,
            cpu_c2s_real_raw,
            cpu_c2s_imag_raw,
            *cpu_evaluator,
            full_galois_keys,
            encoder);
        cpu_c2s_real = cpu_c2s_real_raw;
        cpu_c2s_imag = cpu_c2s_imag_raw;

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
        std::cout << "[phase] staged GPU CoeffToSlot correctness\n" << std::flush;
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
        if (use_double_hoist)
        {
            const auto &trace = bootstrap_workspace
                .coeff_to_slot_double_hoist.matrix_rescale_trace;
            if (trace.size() != c2s_stage_rescale_counts.size())
            {
                throw std::runtime_error(
                    "CoeffToSlot dynamic rescale trace size mismatch");
            }
            std::cout << "\n[C2S dynamic rescale trace]\n";
            for (std::size_t stage = 0; stage < trace.size(); ++stage)
            {
                const auto &entry = trace[stage];
                if (entry.rescale_count != c2s_stage_rescale_counts[stage])
                {
                    throw std::runtime_error(
                        "CoeffToSlot runtime drop count disagrees with setup plan");
                }
                std::cout << "stage=" << stage
                          << " q=" << entry.input_q_count
                          << "->" << entry.output_q_count
                          << " input=2^" << std::fixed << std::setprecision(6)
                          << std::log2(entry.input_scale)
                          << " plain=2^" << std::log2(entry.plaintext_scale)
                          << " product=2^" << std::log2(entry.product_scale)
                          << " drop=" << entry.rescale_count
                          << " output=2^" << std::log2(entry.output_scale)
                          << "\n";
            }
            std::cout.unsetf(std::ios::floatfield);
        }
        std::cout << "C2S CPU/GPU correct=YES max_error="
                  << c2s_max_error << " output_q="
                  << gpu_c2s_real.meta.q_count << " output_scale=2^"
                  << std::log2(gpu_c2s_real.meta.scale) << "\n";
        if (detailed_diagnostics)
        {
            const auto c2s_values = decrypt_decode(
                cpu_c2s_real, decryptor, encoder);
            double c2s_max_abs = 0.0;
            for (const auto &value : c2s_values)
            {
                c2s_max_abs = std::max(c2s_max_abs, std::abs(value));
            }
            std::cout << "C2S EvalMod-domain coefficient normalization=2^"
                      << std::log2(c2s_value_normalization)
                      << " physical output max_abs=" << c2s_max_abs
                      << " (input scale preserved)"
                      << "\n";
        }
        if (c2s_only)
        {
            std::cout << "\n[OK] Stage 2 dynamic CoeffToSlot passed\n";
            return EXIT_SUCCESS;
        }

        poseidon::Ciphertext cpu_eval_real;
        poseidon::Ciphertext cpu_eval_imag;
        poseidon::BootstrapEvalModTrace cpu_eval_real_bootstrap_trace;
        poseidon::EvalModTrace cpu_eval_real_dynamic_trace;
        const bool cpu_evalmod_oracle_available = true;
        const bool cpu_full_oracle_available =
            !evalmod_dynamic_rescale;
        if (cpu_evalmod_oracle_available)
        {
            std::cout << "[phase] staged CPU EvalMod correctness\n" << std::flush;
            if (evalmod_dynamic_rescale)
            {
                // Stage 3 isolates EvalMod from the already-validated C2S
                // implementation. High-degree Chebyshev bases amplify even a
                // tiny difference between independently computed CPU/GPU C2S
                // ciphertexts, so both backends must start from identical RNS
                // words here. End-to-end source accuracy is checked separately.
                cpu_evaluator->eval_mod_high_precision(
                    gpu_c2s_real_download,
                    cpu_eval_real,
                    eval_mod_poly,
                    relin_keys,
                    encoder,
                    detailed_diagnostics ? &cpu_eval_real_dynamic_trace : nullptr,
                    /*preserve_input_scale=*/true);
                cpu_evaluator->eval_mod_high_precision(
                    gpu_c2s_imag_download,
                    cpu_eval_imag,
                    eval_mod_poly,
                    relin_keys,
                    encoder,
                    nullptr,
                    /*preserve_input_scale=*/true);
            }
            else
            {
                cpu_bootstrapper.eval_mod(
                    cpu_c2s_real,
                    cpu_eval_real,
                    relin_keys,
                    evalmod_double_angle,
                    bootstrap_inverse_coefficient,
                    evalmod_scale,
                    detailed_diagnostics ? &cpu_eval_real_bootstrap_trace : nullptr);
                cpu_bootstrapper.eval_mod(
                    cpu_c2s_imag,
                    cpu_eval_imag,
                    relin_keys,
                    evalmod_double_angle,
                    bootstrap_inverse_coefficient,
                    evalmod_scale);
            }
            const auto expected_cpu_evalmod_parms_id =
                evalmod_dynamic_rescale
                    ? evalmod_output_parms_id
                    : cpu_evalmod_output_parms_id;
            // The experimental flat-b8 GPU circuit has one more
            // multiplicative level than the balanced CPU Chebyshev tree.
            // Dropping the CPU oracle to the GPU plan's level preserves the
            // represented value and allows a decoded-value comparison without
            // changing the production CPU or default GPU evaluation paths.
            if (evalmod_dynamic_rescale &&
                bootstrap_data.eval_mod.polynomial_flat_bsgs &&
                cpu_eval_real.coeff_modulus_size() > evalmod_output_q_count)
            {
                cpu_evaluator->drop_modulus(
                    cpu_eval_real,
                    cpu_eval_real,
                    expected_cpu_evalmod_parms_id);
                cpu_evaluator->drop_modulus(
                    cpu_eval_imag,
                    cpu_eval_imag,
                    expected_cpu_evalmod_parms_id);
            }
            if (cpu_eval_real.parms_id() != expected_cpu_evalmod_parms_id ||
                cpu_eval_imag.parms_id() != expected_cpu_evalmod_parms_id)
            {
                throw std::runtime_error(
                    "CPU EvalMod output level changed between setup probe and correctness execution");
            }
        }

        auto &gpu_eval_real = bootstrap_workspace.eval_mod_real;
        auto &gpu_eval_imag = bootstrap_workspace.eval_mod_imag;
        bootstrap_workspace.capture_eval_mod_trace = detailed_diagnostics;
        std::cout << "[phase] staged GPU EvalMod correctness\n" << std::flush;
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
        std::vector<poseidon::Ciphertext> gpu_trace_polynomial_leaf_outputs;
        std::vector<poseidon::Ciphertext> gpu_trace_polynomial_combine_outputs;
        std::vector<poseidon::Ciphertext> gpu_trace_double_angle_square_outputs;
        std::vector<poseidon::Ciphertext> gpu_trace_double_angle_relin_outputs;
        std::vector<poseidon::Ciphertext> gpu_trace_double_angle_rescaled_square_outputs;
        std::vector<poseidon::Ciphertext> gpu_trace_double_angle_outputs;
        std::map<std::uint32_t, poseidon::Ciphertext> gpu_trace_basis_outputs;
        if (detailed_diagnostics)
        {
            gpu_trace_offset_input = download_gpu_ciphertext(
                bootstrap_workspace.eval_mod_trace_offset_input,
                context);
            gpu_trace_polynomial_output = download_gpu_ciphertext(
                bootstrap_workspace.eval_mod_trace_polynomial_output,
                context);
            gpu_trace_polynomial_leaf_outputs.reserve(
                bootstrap_workspace
                    .eval_mod_trace_polynomial_leaf_outputs.size());
            for (const auto &gpu_trace_ciphertext :
                 bootstrap_workspace
                     .eval_mod_trace_polynomial_leaf_outputs)
            {
                gpu_trace_polynomial_leaf_outputs.push_back(
                    download_gpu_ciphertext(gpu_trace_ciphertext, context));
            }
            gpu_trace_polynomial_combine_outputs.reserve(
                bootstrap_workspace
                    .eval_mod_trace_polynomial_combine_outputs.size());
            for (const auto &gpu_trace_ciphertext :
                 bootstrap_workspace
                     .eval_mod_trace_polynomial_combine_outputs)
            {
                gpu_trace_polynomial_combine_outputs.push_back(
                    download_gpu_ciphertext(gpu_trace_ciphertext, context));
            }
            gpu_trace_double_angle_outputs.reserve(
                bootstrap_workspace.eval_mod_trace_double_angle_outputs.size());
            gpu_trace_double_angle_square_outputs.reserve(
                bootstrap_workspace
                    .eval_mod_trace_double_angle_square_outputs.size());
            for (const auto &gpu_trace_ciphertext :
                 bootstrap_workspace
                     .eval_mod_trace_double_angle_square_outputs)
            {
                gpu_trace_double_angle_square_outputs.push_back(
                    download_gpu_ciphertext(gpu_trace_ciphertext, context));
            }
            gpu_trace_double_angle_relin_outputs.reserve(
                bootstrap_workspace
                    .eval_mod_trace_double_angle_relin_outputs.size());
            for (const auto &gpu_trace_ciphertext :
                 bootstrap_workspace
                     .eval_mod_trace_double_angle_relin_outputs)
            {
                gpu_trace_double_angle_relin_outputs.push_back(
                    download_gpu_ciphertext(gpu_trace_ciphertext, context));
            }
            gpu_trace_double_angle_rescaled_square_outputs.reserve(
                bootstrap_workspace
                    .eval_mod_trace_double_angle_rescaled_square_outputs.size());
            for (const auto &gpu_trace_ciphertext :
                 bootstrap_workspace
                     .eval_mod_trace_double_angle_rescaled_square_outputs)
            {
                gpu_trace_double_angle_rescaled_square_outputs.push_back(
                    download_gpu_ciphertext(gpu_trace_ciphertext, context));
            }
            for (const auto &gpu_trace_ciphertext :
                 bootstrap_workspace.eval_mod_trace_double_angle_outputs)
            {
                gpu_trace_double_angle_outputs.push_back(
                    download_gpu_ciphertext(gpu_trace_ciphertext, context));
            }
            if (evalmod_dynamic_rescale)
            {
                for (const auto &step : bootstrap_data.eval_mod.basis_steps)
                {
                    const auto degree = step.output_degree;
                    if (degree < bootstrap_workspace.eval_mod_basis.size() &&
                        !bootstrap_workspace.eval_mod_basis[degree].empty())
                    {
                        gpu_trace_basis_outputs.emplace(
                            degree,
                            download_gpu_ciphertext(
                                bootstrap_workspace.eval_mod_basis[degree],
                                context));
                    }
                }
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
        if (detailed_diagnostics)
        {
            const auto print_decoded_magnitude =
                [&](const char *label, const poseidon::Ciphertext &ciphertext) {
                    const auto values = decrypt_decode(ciphertext, decryptor, encoder);
                    double maximum = 0.0;
                    double rms = 0.0;
                    for (const auto &value : values)
                    {
                        maximum = std::max(maximum, std::abs(value));
                        rms += std::norm(value);
                    }
                    rms = values.empty()
                        ? 0.0
                        : std::sqrt(rms / static_cast<double>(values.size()));
                    std::cout << label << " decoded max/rms=" << maximum
                              << "/" << rms << "\n";
                };
            print_decoded_magnitude("EvalMod real", gpu_eval_real_download);
            print_decoded_magnitude("EvalMod imag", gpu_eval_imag_download);
            print_decoded_magnitude("EvalMod polynomial", gpu_trace_polynomial_output);
            for (const auto &entry : gpu_trace_basis_outputs)
            {
                const auto label = "  basis T" + std::to_string(entry.first);
                print_decoded_magnitude(label.c_str(), entry.second);
            }
            for (std::size_t index = 0;
                 index < gpu_trace_polynomial_leaf_outputs.size(); ++index)
            {
                const auto label = "  leaf #" + std::to_string(index);
                print_decoded_magnitude(
                    label.c_str(), gpu_trace_polynomial_leaf_outputs[index]);
            }
            for (std::size_t index = 0;
                 index < gpu_trace_polynomial_combine_outputs.size(); ++index)
            {
                const auto label = "  combine #" + std::to_string(index);
                print_decoded_magnitude(
                    label.c_str(), gpu_trace_polynomial_combine_outputs[index]);
            }
            if (gpu_trace_polynomial_leaf_outputs.size() > 4)
            {
                auto cpu_rescaled_leaf =
                    gpu_trace_polynomial_leaf_outputs[4];
                for (std::uint32_t index = 0; index < 3; ++index)
                {
                    cpu_evaluator->rescale(
                        cpu_rescaled_leaf, cpu_rescaled_leaf);
                }
                auto gpu_leaf = poseidon::gpu::GpuUploader::upload_ciphertext(
                    gpu_trace_polynomial_leaf_outputs[4], device_id);
                poseidon::gpu::GpuCiphertextData gpu_rescaled_leaf;
                gpu_evaluator.rescale_many(gpu_leaf, gpu_rescaled_leaf, 3);
                cudaDeviceSynchronize();
                auto downloaded_rescaled_leaf = download_gpu_ciphertext(
                    gpu_rescaled_leaf, context);
                cpu_rescaled_leaf.scale() = downloaded_rescaled_leaf.scale();
                const auto rescale_raw = compare_ciphertexts(
                    cpu_rescaled_leaf, downloaded_rescaled_leaf, 4);
                const auto rescale_approx = compare_decrypted_ciphertexts(
                    cpu_rescaled_leaf,
                    downloaded_rescaled_leaf,
                    decryptor,
                    encoder,
                    correctness_tolerance);
                std::cout << "  mid-level q25->q22 rescale_many(3) raw/err="
                          << (rescale_raw.equal ? "YES" : "NO") << "/"
                          << rescale_approx.max_abs_error << "\n";
            }
        }
        if (gpu_eval_real_download.parms_id() != evalmod_output_parms_id ||
            gpu_eval_imag_download.parms_id() != evalmod_output_parms_id)
        {
            throw std::runtime_error(
                "GPU EvalMod output level does not match the configured high-precision schedule");
        }
        ApproxComparison eval_real_comparison{true, 0.0, 0.0};
        ApproxComparison eval_imag_comparison{true, 0.0, 0.0};
        bool evalmod_correct = true;
        std::string evalmod_comparison_label = "GPU level/metadata";
        if (cpu_evalmod_oracle_available)
        {
            eval_real_comparison =
                compare_decrypted_ciphertexts(
                    cpu_eval_real,
                    gpu_eval_real_download,
                    decryptor,
                    encoder,
                    correctness_tolerance);
            eval_imag_comparison =
                compare_decrypted_ciphertexts(
                    cpu_eval_imag,
                    gpu_eval_imag_download,
                    decryptor,
                    encoder,
                    correctness_tolerance);
            evalmod_correct =
                eval_real_comparison.equal && eval_imag_comparison.equal;
            evalmod_comparison_label = "CPU/GPU decoded";
        }
        else
        {
            const auto compare_with_plain_eval =
                [&](const poseidon::Ciphertext &input,
                    const poseidon::Ciphertext &actual) {
                    const auto input_values = decrypt_decode(
                        input, decryptor, encoder);
                    std::vector<std::complex<double>> expected_values;
                    expected_values.reserve(input_values.size());
                    for (const auto &value : input_values)
                    {
                        expected_values.push_back(
                            cpu_bootstrapper.eval_mod_plain_trace(
                                value,
                                evalmod_double_angle,
                                bootstrap_inverse_coefficient).back());
                    }
                    return compare_approx(
                        expected_values,
                        decrypt_decode(actual, decryptor, encoder),
                        correctness_tolerance);
                };
            eval_real_comparison = compare_with_plain_eval(
                gpu_c2s_real_download, gpu_eval_real_download);
            eval_imag_comparison = compare_with_plain_eval(
                gpu_c2s_imag_download, gpu_eval_imag_download);
            evalmod_correct =
                eval_real_comparison.equal && eval_imag_comparison.equal;
            evalmod_comparison_label = "GPU/plain cosine heap";
        }

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

        if (detailed_diagnostics && cpu_evalmod_oracle_available)
        {
            const auto &cpu_trace_input = evalmod_dynamic_rescale
                ? cpu_eval_real_dynamic_trace.offset_input
                : cpu_eval_real_bootstrap_trace.input;
            const auto &cpu_trace_polynomial_output = evalmod_dynamic_rescale
                ? cpu_eval_real_dynamic_trace.polynomial_output
                : cpu_eval_real_bootstrap_trace.polynomial_output;
            const auto &cpu_trace_double_angle_outputs = evalmod_dynamic_rescale
                ? cpu_eval_real_dynamic_trace.double_angle_outputs
                : cpu_eval_real_bootstrap_trace.double_angle_outputs;
            std::vector<EvalModTraceRow> eval_mod_trace_rows;
            eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                "input",
                cpu_trace_input,
                gpu_trace_offset_input,
                decryptor,
                encoder,
                correctness_tolerance));
            eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                "polynomial output",
                cpu_trace_polynomial_output,
                gpu_trace_polynomial_output,
                decryptor,
                encoder,
                correctness_tolerance));
            const auto polynomial_output_raw =
                compare_ciphertexts(
                    cpu_trace_polynomial_output,
                    gpu_trace_polynomial_output,
                    0);
            std::cout << "\n[EvalMod polynomial output raw diagnostic]\n";
            std::cout << "cpu size/ntt/q/log2(scale) = "
                      << cpu_trace_polynomial_output.size() << "/"
                      << (cpu_trace_polynomial_output.is_ntt_form() ? 1 : 0)
                      << "/"
                      << cpu_trace_polynomial_output.coeff_modulus_size()
                      << "/"
                      << std::fixed << std::setprecision(3)
                      << std::log2(cpu_trace_polynomial_output.scale())
                      << "\n";
            std::cout << "gpu size/ntt/q/log2(scale) = "
                      << gpu_trace_polynomial_output.size() << "/"
                      << (gpu_trace_polynomial_output.is_ntt_form() ? 1 : 0)
                      << "/"
                      << gpu_trace_polynomial_output.coeff_modulus_size()
                      << "/"
                      << std::fixed << std::setprecision(3)
                      << std::log2(gpu_trace_polynomial_output.scale())
                      << "\n";
            std::cout << "raw words/mismatches/equal = "
                      << polynomial_output_raw.expected_words << "/"
                      << polynomial_output_raw.mismatch_count << "/"
                      << (polynomial_output_raw.equal ? "YES" : "NO")
                      << "\n";
            if (evalmod_dynamic_rescale)
            {
                std::cout << "leaf trace counts CPU/GPU = "
                          << cpu_eval_real_dynamic_trace
                                 .polynomial_leaves.size()
                          << "/"
                          << gpu_trace_polynomial_leaf_outputs.size()
                          << "\n";
                std::cout << "combine trace counts CPU/GPU = "
                          << cpu_eval_real_dynamic_trace
                                 .polynomial_combines.size()
                          << "/"
                          << gpu_trace_polynomial_combine_outputs.size()
                          << "\n";
                const auto compared_leaf_count = std::min(
                    cpu_eval_real_dynamic_trace.polynomial_leaves.size(),
                    gpu_trace_polynomial_leaf_outputs.size());
                for (std::size_t leaf_index = 0;
                     leaf_index < compared_leaf_count;
                     ++leaf_index)
                {
                    eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                        "polynomial leaf " +
                            std::to_string(leaf_index),
                        cpu_eval_real_dynamic_trace
                            .polynomial_leaves[leaf_index],
                        gpu_trace_polynomial_leaf_outputs[leaf_index],
                        decryptor,
                        encoder,
                        correctness_tolerance));

                    poseidon::Ciphertext cpu_leaf_square;
                    poseidon::Ciphertext gpu_leaf_cpu_square;
                    cpu_evaluator->square(
                        cpu_eval_real_dynamic_trace
                            .polynomial_leaves[leaf_index],
                        cpu_leaf_square);
                    cpu_evaluator->square(
                        gpu_trace_polynomial_leaf_outputs[leaf_index],
                        gpu_leaf_cpu_square);
                    eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                        "polynomial leaf CPU-square " +
                            std::to_string(leaf_index),
                        cpu_leaf_square,
                        gpu_leaf_cpu_square,
                        decryptor,
                        encoder,
                        correctness_tolerance));
                }
                const auto compared_combine_count = std::min(
                    cpu_eval_real_dynamic_trace.polynomial_combines.size(),
                    gpu_trace_polynomial_combine_outputs.size());
                for (std::size_t combine_index = 0;
                     combine_index < compared_combine_count;
                     ++combine_index)
                {
                    eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                        "polynomial combine " +
                            std::to_string(combine_index),
                        cpu_eval_real_dynamic_trace
                            .polynomial_combines[combine_index],
                        gpu_trace_polynomial_combine_outputs[combine_index],
                        decryptor,
                        encoder,
                        correctness_tolerance));

                    poseidon::Ciphertext cpu_combine_square;
                    poseidon::Ciphertext gpu_combine_cpu_square;
                    cpu_evaluator->square(
                        cpu_eval_real_dynamic_trace
                            .polynomial_combines[combine_index],
                        cpu_combine_square);
                    cpu_evaluator->square(
                        gpu_trace_polynomial_combine_outputs[combine_index],
                        gpu_combine_cpu_square);
                    eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                        "polynomial combine CPU-square " +
                            std::to_string(combine_index),
                        cpu_combine_square,
                        gpu_combine_cpu_square,
                        decryptor,
                        encoder,
                        correctness_tolerance));
                }
            }
            const auto compared_double_angle_count = std::min(
                cpu_trace_double_angle_outputs.size(),
                gpu_trace_double_angle_outputs.size());
            if (evalmod_dynamic_rescale)
            {
                std::vector<poseidon::Ciphertext> cpu_da_square_outputs;
                std::vector<poseidon::Ciphertext> cpu_da_relin_outputs;
                std::vector<poseidon::Ciphertext> cpu_da_rescaled_square_outputs;
                cpu_da_square_outputs.reserve(compared_double_angle_count);
                cpu_da_relin_outputs.reserve(compared_double_angle_count);
                cpu_da_rescaled_square_outputs.reserve(compared_double_angle_count);
                poseidon::Ciphertext cpu_da_input =
                    cpu_trace_polynomial_output;
                double diagnostic_sqrt2pi = eval_mod_poly.sqrt_2pi();
                for (std::size_t double_angle_index = 0;
                     double_angle_index < compared_double_angle_count;
                     ++double_angle_index)
                {
                    poseidon::Ciphertext cpu_da_square;
                    cpu_evaluator->square(
                        cpu_da_input,
                        cpu_da_square);
                    cpu_da_square_outputs.push_back(cpu_da_square);

                    poseidon::Ciphertext cpu_da_relin;
                    cpu_evaluator->multiply_relin_dynamic(
                        cpu_da_input,
                        cpu_da_input,
                        cpu_da_relin,
                        relin_keys);
                    cpu_da_relin_outputs.push_back(cpu_da_relin);

                    diagnostic_sqrt2pi *= diagnostic_sqrt2pi;
                    poseidon::Ciphertext cpu_da_pre_rescale;
                    cpu_evaluator->add(
                        cpu_da_relin,
                        cpu_da_relin,
                        cpu_da_pre_rescale);
                    cpu_evaluator->add_const(
                        cpu_da_pre_rescale,
                        -diagnostic_sqrt2pi,
                        cpu_da_pre_rescale,
                        encoder);
                    poseidon::Ciphertext cpu_da_rescaled;
                    cpu_evaluator->rescale_dynamic(
                        cpu_da_pre_rescale,
                        cpu_da_rescaled,
                        evalmod_scale);
                    cpu_da_rescaled_square_outputs.push_back(cpu_da_rescaled);
                    cpu_da_input =
                        cpu_trace_double_angle_outputs[double_angle_index];
                }

                if (!cpu_da_square_outputs.empty())
                {
                    poseidon::Ciphertext gpu_poly_cpu_square;
                    cpu_evaluator->square(
                        gpu_trace_polynomial_output,
                        gpu_poly_cpu_square);
                    eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                        "polynomial output CPU-square",
                        cpu_da_square_outputs.front(),
                        gpu_poly_cpu_square,
                        decryptor,
                        encoder,
                        correctness_tolerance));
                }

                const auto compared_square_count = std::min(
                    cpu_da_square_outputs.size(),
                    gpu_trace_double_angle_square_outputs.size());
                for (std::size_t double_angle_index = 0;
                     double_angle_index < compared_square_count;
                     ++double_angle_index)
                {
                    eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                        "double angle square " +
                            std::to_string(double_angle_index),
                        cpu_da_square_outputs[double_angle_index],
                        gpu_trace_double_angle_square_outputs[double_angle_index],
                        decryptor,
                        encoder,
                        correctness_tolerance));
                }
                for (std::size_t double_angle_index = 0;
                     double_angle_index < compared_square_count;
                     ++double_angle_index)
                {
                    poseidon::Ciphertext gpu_square_cpu_relin;
                    cpu_evaluator->relinearize(
                        gpu_trace_double_angle_square_outputs[double_angle_index],
                        gpu_square_cpu_relin,
                        relin_keys);
                    eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                        "double angle square CPU-relin " +
                            std::to_string(double_angle_index),
                        cpu_da_relin_outputs[double_angle_index],
                        gpu_square_cpu_relin,
                        decryptor,
                        encoder,
                        correctness_tolerance));
                }

                const auto compared_relin_count = std::min(
                    cpu_da_relin_outputs.size(),
                    gpu_trace_double_angle_relin_outputs.size());
                for (std::size_t double_angle_index = 0;
                     double_angle_index < compared_relin_count;
                     ++double_angle_index)
                {
                    eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                        "double angle relin " +
                            std::to_string(double_angle_index),
                        cpu_da_relin_outputs[double_angle_index],
                        gpu_trace_double_angle_relin_outputs[double_angle_index],
                        decryptor,
                        encoder,
                        correctness_tolerance));
                }

                const auto compared_rescaled_square_count = std::min(
                    cpu_da_rescaled_square_outputs.size(),
                    gpu_trace_double_angle_rescaled_square_outputs.size());
                for (std::size_t double_angle_index = 0;
                     double_angle_index < compared_rescaled_square_count;
                     ++double_angle_index)
                {
                    eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                        "double angle square-rescale " +
                            std::to_string(double_angle_index),
                        cpu_da_rescaled_square_outputs[double_angle_index],
                        gpu_trace_double_angle_rescaled_square_outputs
                            [double_angle_index],
                        decryptor,
                        encoder,
                        correctness_tolerance));
                }
            }
            for (std::size_t double_angle_index = 0;
                 double_angle_index < compared_double_angle_count;
                 ++double_angle_index)
            {
                eval_mod_trace_rows.push_back(compare_eval_mod_trace_stage(
                    "double angle " + std::to_string(double_angle_index),
                    cpu_trace_double_angle_outputs[double_angle_index],
                    gpu_trace_double_angle_outputs[double_angle_index],
                    decryptor,
                    encoder,
                    correctness_tolerance));
            }
            print_eval_mod_trace_table(eval_mod_trace_rows);
            std::cout << "trace counts: double-angle CPU/GPU="
                      << cpu_trace_double_angle_outputs.size() << "/"
                      << gpu_trace_double_angle_outputs.size() << "\n";

            if (evalmod_dynamic_rescale)
            {
                std::cout << "\n[EvalMod basis trace: real branch]\n";
                std::cout << "|--------|---------|---------|--------------|--------------|------------------|-----------|\n";
                std::cout << "| degree |   CPU q |   GPU q |  CPU log2(s) |  GPU log2(s) |    max abs error |   correct |\n";
                std::cout << "|--------|---------|---------|--------------|--------------|------------------|-----------|\n";
                for (const auto &entry : cpu_eval_real_dynamic_trace.basis)
                {
                    const auto degree = entry.first;
                    const auto gpu_basis_iter =
                        gpu_trace_basis_outputs.find(degree);
                    if (gpu_basis_iter == gpu_trace_basis_outputs.end())
                    {
                        std::cout << "| " << std::setw(6) << degree
                                  << " | " << std::setw(7)
                                  << entry.second.coeff_modulus_size()
                                  << " | " << std::setw(7) << "-"
                                  << " | " << std::setw(12)
                                  << std::fixed << std::setprecision(3)
                                  << std::log2(entry.second.scale())
                                  << " | " << std::setw(12) << "-"
                                  << " | " << std::setw(16) << "-"
                                  << " | " << std::setw(9) << "NO"
                                  << " |\n";
                        continue;
                    }
                    const auto row = compare_eval_mod_trace_stage(
                        "basis T" + std::to_string(degree),
                        entry.second,
                        gpu_basis_iter->second,
                        decryptor,
                        encoder,
                        correctness_tolerance);
                    std::cout << "| " << std::setw(6) << degree
                              << " | " << std::setw(7) << row.cpu_q_count
                              << " | " << std::setw(7) << row.gpu_q_count
                              << " | " << std::setw(12)
                              << std::fixed << std::setprecision(3)
                              << row.cpu_log2_scale
                              << " | " << std::setw(12)
                              << row.gpu_log2_scale
                              << " | " << std::setw(16)
                              << std::scientific << std::setprecision(6)
                              << row.comparison.max_abs_error
                              << " | " << std::setw(9)
                              << (row.comparison.equal ? "YES" : "NO")
                              << " |\n";
                    std::cout.unsetf(std::ios::floatfield);
                }
                std::cout << "|--------|---------|---------|--------------|--------------|------------------|-----------|\n";
            }
        }

        std::cout << "EvalMod CPU/GPU correct="
                  << (evalmod_correct ? "YES" : "NO")
                  << " real_max_error=" << eval_real_comparison.max_abs_error
                  << " imag_max_error=" << eval_imag_comparison.max_abs_error
                  << " input_q=" << gpu_c2s_real.meta.q_count
                  << " output_q=" << gpu_eval_real.meta.q_count
                  << " output_scale=2^"
                  << std::log2(gpu_eval_real.meta.scale) << "\n";
        if (evalmod_only)
        {
            if (!evalmod_correct)
            {
                std::cerr << "[FAILED] Stage 3 dynamic EvalMod comparison\n";
                return EXIT_FAILURE;
            }
            std::cout << "\n[OK] Stage 3 dynamic EvalMod passed\n";
            return EXIT_SUCCESS;
        }

        poseidon::Ciphertext cpu_s2c_result;
        if (cpu_evalmod_oracle_available)
        {
            std::cout << "[phase] staged CPU SlotToCoeff correctness\n" << std::flush;
            std::cout << "  CPU EvalMod real/imag log2(scale) = "
                      << std::log2(cpu_eval_real.scale()) << "/"
                      << std::log2(cpu_eval_imag.scale())
                      << ", q_count = " << cpu_eval_real.coeff_modulus_size()
                      << ", total modulus bits = "
                      << context.crt_context()
                             ->get_context_data(cpu_eval_real.parms_id())
                             ->total_coeff_modulus_bit_count()
                      << "\n";
            for (std::size_t matrix_index = 0;
                 matrix_index < s2c_matrix_group.data().size();
                 ++matrix_index)
            {
                const auto &matrix = s2c_matrix_group.data()[matrix_index];
                const auto &matrix_plain = matrix.plain_vec.begin()->second;
                std::cout << "  S2C matrix " << matrix_index
                          << " log2(scale) = " << std::log2(matrix_plain.scale())
                          << ", q_count = "
                          << matrix_plain.coeff_count() / degree << "\n";
            }
            std::cout << std::flush;
            cpu_eval_real.scale() = s2c_input_scale;
            cpu_eval_imag.scale() = s2c_input_scale;
            cpu_slot_to_coeff_rescale(
                cpu_eval_real,
                cpu_eval_imag,
                s2c_matrix_group,
                cpu_s2c_result,
                *cpu_evaluator,
                full_galois_keys,
                encoder);
            cpu_s2c_result.scale() = s2c_output_scale;
        }
        else
        {
            std::cout << "[phase] staged CPU SlotToCoeff correctness skipped "
                      << "(dynamic GPU EvalMod output has no CPU ciphertext oracle yet)\n"
                      << std::flush;
        }
        poseidon::gpu::GpuCiphertextData gpu_s2c_result;
        std::cout << "[phase] staged GPU SlotToCoeff correctness\n" << std::flush;
        gpu_eval_real_stable.meta.scale = s2c_input_scale;
        gpu_eval_imag.meta.scale = s2c_input_scale;
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
        if (detailed_diagnostics)
        {
            const auto s2c_values = decrypt_decode(
                gpu_s2c_download, decryptor, encoder);
            double s2c_maximum = 0.0;
            for (const auto &value : s2c_values)
            {
                s2c_maximum = std::max(s2c_maximum, std::abs(value));
            }
            std::cout << "S2C decoded max_abs=" << s2c_maximum << "\n";
        }
        ApproxComparison s2c_comparison{true, 0.0, 0.0};
        bool s2c_correct = true;
        std::string s2c_comparison_label = "GPU staged only";
        if (cpu_evalmod_oracle_available)
        {
            s2c_comparison =
                compare_decrypted_ciphertexts(
                    cpu_s2c_result,
                    gpu_s2c_download,
                    decryptor,
                    encoder,
                    correctness_tolerance);
            s2c_correct = s2c_comparison.equal;
            s2c_comparison_label = "CPU/GPU decoded";
        }
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
            eval_real.scale() = s2c_input_scale;
            eval_imag.scale() = s2c_input_scale;
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

        poseidon::Ciphertext cpu_full_result;
        if (cpu_full_oracle_available)
        {
            std::cout << "[phase] full staged CPU bootstrap correctness\n" << std::flush;
            cpu_full_result = run_cpu_full_bootstrap();
        }
        else
        {
            std::cout << "[phase] full staged CPU bootstrap correctness skipped "
                      << "(dynamic GPU schedule has no fixed-rescale CPU oracle yet)\n"
                      << std::flush;
        }
        poseidon::gpu::GpuCiphertextData gpu_full_result;
        std::cout << "[phase] full GPU bootstrap correctness\n" << std::flush;
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
        ApproxComparison full_cpu_gpu_comparison{true, 0.0, 0.0};
        if (cpu_full_oracle_available)
        {
            full_cpu_gpu_comparison =
                compare_decrypted_ciphertexts(
                    cpu_full_result,
                    gpu_full_download,
                    decryptor,
                    encoder,
                    correctness_tolerance);
        }
        ApproxComparison cpu_library_source_comparison{true, 0.0, 0.0};
        ApproxComparison staged_cpu_library_comparison{true, 0.0, 0.0};
        ApproxComparison gpu_library_comparison{true, 0.0, 0.0};
        if (!skip_library_oracle && cpu_full_oracle_available)
        {
            cpu_library_source_comparison =
                compare_approx(
                    message,
                    decrypt_decode(
                        cpu_library_bootstrap_result,
                        decryptor,
                        encoder),
                    correctness_tolerance);
            staged_cpu_library_comparison =
                compare_decrypted_ciphertexts(
                    cpu_library_bootstrap_result,
                    cpu_full_result,
                    decryptor,
                    encoder,
                    correctness_tolerance);
            gpu_library_comparison =
                compare_decrypted_ciphertexts(
                    cpu_library_bootstrap_result,
                    gpu_full_download,
                    decryptor,
                    encoder,
                    correctness_tolerance);
        }
        ApproxComparison cpu_source_comparison{true, 0.0, 0.0};
        if (cpu_full_oracle_available)
        {
            cpu_source_comparison =
                compare_approx(
                    message,
                    decrypt_decode(cpu_full_result, decryptor, encoder),
                    correctness_tolerance);
        }
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
                    evalmod_comparison_label},
                CorrectnessRow{
                    "SlotToCoeff",
                    s2c_correct,
                    s2c_comparison.max_abs_error,
                    s2c_comparison_label},
                CorrectnessRow{
                    "Full bootstrap",
                    full_correct,
                    gpu_source_comparison.max_abs_error,
                    "GPU/source decoded"}});

        if (!gpu_only_timing)
        {
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
        }

        double cpu_c2s_ms = std::numeric_limits<double>::quiet_NaN();
        if (!gpu_only_timing)
        {
            cpu_c2s_ms = time_cpu_ms(iterations, [&]() {
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
        }

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

        if (!gpu_only_timing && cpu_evalmod_oracle_available)
        {
            for (std::size_t i = 0; i < full_warmup; ++i)
            {
                poseidon::Ciphertext cpu_real;
                poseidon::Ciphertext cpu_imag;
                if (evalmod_dynamic_rescale)
                {
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
                }
                else
                {
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
            }
        }

        double cpu_evalmod_ms = std::numeric_limits<double>::quiet_NaN();
        if (!gpu_only_timing && cpu_evalmod_oracle_available)
        {
            cpu_evalmod_ms = time_cpu_ms(full_iterations, [&]() {
                poseidon::Ciphertext real;
                poseidon::Ciphertext imag;
                if (evalmod_dynamic_rescale)
                {
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
                }
                else
                {
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
                }
            });
        }

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
        EvalModMultiplyTimingMap evalmod_multiply_timing_sum;
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
            accumulate_eval_mod_multiply_timings(
                evalmod_multiply_timing_sum,
                bootstrap_workspace.eval_mod_multiply_timings);
            gpu_evaluator.eval_mod_high_precision(
                gpu_c2s_imag,
                bootstrap_data,
                gpu_relin_keys,
                bootstrap_workspace,
                gpu_eval_imag);
            accumulate_eval_mod_stage_timing(
                evalmod_stage_timing_sum,
                bootstrap_workspace.eval_mod_stage_timing);
            accumulate_eval_mod_multiply_timings(
                evalmod_multiply_timing_sum,
                bootstrap_workspace.eval_mod_multiply_timings);
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

        if (!gpu_only_timing && cpu_evalmod_oracle_available)
        {
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
        }
        double cpu_s2c_ms = std::numeric_limits<double>::quiet_NaN();
        if (!gpu_only_timing && cpu_evalmod_oracle_available)
        {
            cpu_s2c_ms = time_cpu_ms(full_iterations, [&]() {
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
        }

        gpu_eval_real.meta.scale = s2c_input_scale;
        gpu_eval_imag.meta.scale = s2c_input_scale;
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

        if (!gpu_only_timing && cpu_full_oracle_available)
        {
            for (std::size_t i = 0; i < full_warmup; ++i)
            {
                cpu_full_result = run_cpu_full_bootstrap();
            }
        }
        double cpu_full_bootstrap_ms =
            std::numeric_limits<double>::quiet_NaN();
        if (!gpu_only_timing && cpu_full_oracle_available)
        {
            cpu_full_bootstrap_ms = time_cpu_ms(full_iterations, [&]() {
                cpu_full_result = run_cpu_full_bootstrap();
            });
        }

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

        double gpu_full_bootstrap_ms = std::numeric_limits<double>::quiet_NaN();
        if (nsys_capture_full)
        {
            std::cout << "[phase] nsys capture: one profiled full GPU bootstrap after warmup\n"
                      << std::flush;
            const auto capture_start_status = cudaProfilerStart();
            if (capture_start_status != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("cudaProfilerStart failed: ") +
                    cudaGetErrorString(capture_start_status));
            }
            nvtxRangePushA("profiled_full_bootstrap_once");
            const bool capture_keyswitch_detail = env_flag_enabled(
                "POSEIDON_BOOTSTRAP_NSYS_KEYSWITCH_DETAIL");
            bootstrap_workspace.capture_eval_mod_stage_timing =
                capture_keyswitch_detail;
            const auto start = std::chrono::steady_clock::now();
            gpu_evaluator.bootstrap(
                gpu_source,
                bootstrap_data,
                gpu_relin_keys,
                gpu_full_galois_keys,
                bootstrap_workspace,
                gpu_full_result);
            cudaDeviceSynchronize();
            const auto stop = std::chrono::steady_clock::now();
            bootstrap_workspace.capture_eval_mod_stage_timing = false;
            nvtxRangePop();
            const auto capture_stop_status = cudaProfilerStop();
            if (capture_stop_status != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("cudaProfilerStop failed: ") +
                    cudaGetErrorString(capture_stop_status));
            }
            const auto elapsed_us =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    stop - start).count();
            gpu_full_bootstrap_ms =
                static_cast<double>(elapsed_us) / 1000.0;
        }
        else
        {
            gpu_full_bootstrap_ms = time_gpu_ms(full_iterations, [&]() {
                gpu_evaluator.bootstrap(
                    gpu_source,
                    bootstrap_data,
                    gpu_relin_keys,
                    gpu_full_galois_keys,
                    bootstrap_workspace,
                    gpu_full_result);
            });
        }

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
        print_eval_mod_multiply_timing_table(
            evalmod_multiply_timing_sum,
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

        if (use_double_hoist)
        {
            const bool specialized_shape = degree == 65536 && p_count == 9;
            const bool modup_row_tiled8 =
                specialized_shape && env_flag_enabled_or(
                    "POSEIDON_DOUBLE_HOIST_P9_MODUP_ROW_TILED_8",
                    true);
            const bool moddown_preweight =
                specialized_shape && env_flag_enabled_or(
                    "POSEIDON_DOUBLE_HOIST_P9_PREWEIGHT_P",
                    true);
            const bool moddown_p_to_q_row_tiled8 =
                specialized_shape && env_flag_enabled_or(
                    "POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_ROW_TILED_8",
                    true);
            const bool qp_fourstep =
                specialized_shape && env_flag_enabled_or(
                    "POSEIDON_DOUBLE_HOIST_P9_QP_FOURSTEP",
                    true);
            const bool p_to_q_fourstep =
                specialized_shape && env_flag_enabled_or(
                    "POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_FOURSTEP",
                    true);
            const bool keyswitch_p_to_q_fourstep =
                specialized_shape && env_flag_enabled_or(
                    "POSEIDON_KEYSWITCH_P9_P_TO_Q_FOURSTEP",
                    true);
            const bool qp_mac_group_tiled8 = env_flag_enabled_or(
                "POSEIDON_DOUBLE_HOIST_QP_MAC_GROUP_TILED_8",
                true);
            const bool qp_mac_component_fused = env_flag_enabled_or(
                "POSEIDON_DOUBLE_HOIST_QP_MAC_COMPONENT_FUSED",
                true);
            const bool direct_giant_accumulate = env_flag_enabled_or(
                "POSEIDON_DOUBLE_HOIST_DIRECT_GIANT_ACCUMULATE",
                true);
            const auto state = [](bool enabled) {
                return enabled ? "ON" : "OFF";
            };
            const char *p_to_q_row8_state =
                p_to_q_fourstep && moddown_p_to_q_row_tiled8
                    ? "FALLBACK"
                    : state(moddown_p_to_q_row_tiled8);
            std::cout
                << "\n[WARN] Double-Hoist optimized paths (validated for "
                   "N=65536, P=9): ModUp-row8="
                << state(modup_row_tiled8)
                << ", ModDown-P-preweight="
                << state(moddown_preweight)
                << ", ModDown-P-to-Q-row8="
                << p_to_q_row8_state
                << ", QP-Four-step="
                << state(qp_fourstep)
                << ", P-to-Q-Four-step="
                << state(p_to_q_fourstep)
                << ", KeySwitch-P-to-Q-Four-step="
                << state(keyswitch_p_to_q_fourstep)
                << ", QP-MAC-group-tiled4/8="
                << state(qp_mac_group_tiled8)
                << ", QP-MAC-component-fused="
                << state(qp_mac_component_fused)
                << ", direct-giant-accumulate="
                << state(direct_giant_accumulate)
                << ", QP-phase2-MAC-alias/occupancy=ON"
                << ". Set the corresponding POSEIDON_DOUBLE_HOIST_* "
                   "or POSEIDON_KEYSWITCH_P9_* variable to 0 for runtime "
                   "path rollback; phase2-MAC alias/occupancy tuning is a "
                   "compile-time N=65536 default.\n";
        }

        std::cout
            << "\n[WARN] EvalMod degree-59 lazy relinearization="
            << (env_flag_enabled_or("POSEIDON_EVALMOD_LAZY_RELIN", true)
                    ? "ON"
                    : "OFF")
            << ", log_split="
            << bootstrap_data.eval_mod.polynomial_log_split
            << ", layout="
            << (bootstrap_data.eval_mod.polynomial_flat_bsgs
                    ? "flat_b8"
                    : "recursive")
            << ", leaves="
            << bootstrap_data.eval_mod.polynomial_blocks.size()
            << ", combines="
            << bootstrap_data.eval_mod.polynomial_combine_steps.size()
            << ". Eligible remainder-chain outputs stay as size-3 "
               "ciphertexts and share one later relinearization. Set "
               "POSEIDON_EVALMOD_LAZY_RELIN=0 for rollback.\n";

        if (!evalmod_correct || !s2c_correct || !full_correct)
        {
            if (!ignore_correctness_failure)
            {
                return EXIT_FAILURE;
            }
            std::cout << "\n[WARN] Correctness failure ignored because "
                         "POSEIDON_BOOTSTRAP_IGNORE_CORRECTNESS_FAILURE=1\n";
        }

        if (evalmod_correct && s2c_correct && full_correct)
        {
            std::cout << "\n[OK] One GPU bootstrap data set passed ModRaise, "
                         "CoeffToSlot, high-precision EvalMod, SlotToCoeff, "
                         "and full-bootstrap checks\n";
        }
        return EXIT_SUCCESS;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[FAILED] " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
