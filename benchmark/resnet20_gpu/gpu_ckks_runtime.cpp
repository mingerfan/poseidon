#include "gpu_ckks_runtime.h"

#include "poseidon/ckks_encoder.h"
#include "poseidon/advance/homomorphic_dft.h"
#include "poseidon/advance/homomorphic_mod.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

#include <cuda_runtime_api.h>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

namespace poseidon::benchmark::resnet20_gpu::core
{
namespace
{

std::uint64_t double_bits(double value)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::string scalar_plaintext_cache_key(
    const char *operation,
    std::size_t q_count,
    double value,
    double scale)
{
    return std::string(operation) + ':' + std::to_string(q_count) + ':' +
           std::to_string(double_bits(value)) + ':' +
           std::to_string(double_bits(scale));
}

std::string vector_cache_key(
    const char *operation,
    const std::vector<double> &values)
{
    // Two independent 64-bit accumulators make accidental reuse of a
    // different model operand vanishingly unlikely without retaining a second
    // host copy of every plaintext vector.
    std::uint64_t first = 1469598103934665603ULL;
    std::uint64_t second = 0x9e3779b97f4a7c15ULL;
    for (const double value : values)
    {
        const auto bits = double_bits(value);
        first ^= bits;
        first *= 1099511628211ULL;
        second ^= bits + 0x9e3779b97f4a7c15ULL + (second << 6) + (second >> 2);
    }
    return std::string(operation) + ':' + std::to_string(values.size()) + ':' +
           std::to_string(first) + ':' + std::to_string(second);
}

void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

bool environment_flag_enabled(const char *name)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
    {
        return false;
    }
    const std::string value(raw);
    return value != "0" && value != "OFF" && value != "off" &&
           value != "false" && value != "FALSE";
}

enum class ApplicationKeySwitchPMode
{
    single_digit,
    fixed_dnum,
};

ApplicationKeySwitchPMode configured_application_keyswitch_p_mode()
{
    const char *raw = std::getenv("POSEIDON_APPLICATION_KEYSWITCH_P_MODE");
    if (raw == nullptr || *raw == '\0' || std::string(raw) == "single_digit")
    {
        return ApplicationKeySwitchPMode::single_digit;
    }
    const std::string value(raw);
    if (value == "fixed_dnum" || value == "fixed-dnum")
    {
        return ApplicationKeySwitchPMode::fixed_dnum;
    }
    throw std::invalid_argument(
        "POSEIDON_APPLICATION_KEYSWITCH_P_MODE must be single_digit or fixed_dnum");
}

const char *application_keyswitch_p_mode_name(ApplicationKeySwitchPMode mode)
{
    return mode == ApplicationKeySwitchPMode::fixed_dnum
        ? "fixed_dnum"
        : "single_digit";
}

std::size_t fixed_dnum_p_count(
    std::size_t q_count,
    std::size_t target_dnum)
{
    if (q_count < 2 || target_dnum == 0)
    {
        throw std::invalid_argument("fixed-dnum KeySwitch shape is invalid");
    }
    // Poseidon selects its BV implementation when P has one prime. Keep two
    // primes as the smallest basis accepted by the GPU HYBRID implementation.
    return std::max<std::size_t>(
        2, (q_count + target_dnum - 1) / target_dnum);
}

std::optional<std::size_t> configured_rmm_pool_limit()
{
    const char *raw = std::getenv("POSEIDON_GPU_MAX_POOL_MB");
    if (raw == nullptr || *raw == '\0')
    {
        return std::nullopt;
    }
    try
    {
        const std::string text(raw);
        std::size_t parsed = 0;
        const auto value_mb = std::stoull(text, &parsed, 10);
        if (parsed != text.size())
        {
            throw std::invalid_argument("trailing characters");
        }
        if (value_mb == 0)
        {
            return std::nullopt;
        }
        constexpr std::size_t mib = 1024 * 1024;
        if (value_mb > std::numeric_limits<std::size_t>::max() / mib)
        {
            throw std::out_of_range("pool limit overflow");
        }
        return static_cast<std::size_t>(value_mb) * mib;
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument(
            "POSEIDON_GPU_MAX_POOL_MB must be a non-negative integer");
    }
}

class RmmPoolScope
{
public:
    explicit RmmPoolScope(int device_id)
        : device_id_(device_id),
          maximum_pool_size_(configured_rmm_pool_limit()),
          pool_(&upstream_, 1 << 20, maximum_pool_size_)
    {
        check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
        previous_ = rmm::mr::get_current_device_resource();
        rmm::mr::set_current_device_resource(&pool_);
        if (maximum_pool_size_)
        {
            std::cout << "[GPU memory] RMM pool hard limit="
                      << *maximum_pool_size_ / (1024 * 1024) << " MiB\n";
        }
    }

    RmmPoolScope(const RmmPoolScope &) = delete;
    RmmPoolScope &operator=(const RmmPoolScope &) = delete;

    ~RmmPoolScope()
    {
        try
        {
            (void)cudaSetDevice(device_id_);
            rmm::mr::set_current_device_resource(previous_);
        }
        catch (...)
        {}
    }

private:
    int device_id_;
    rmm::mr::cuda_memory_resource upstream_;
    std::optional<std::size_t> maximum_pool_size_;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_;
    rmm::mr::device_memory_resource *previous_ = nullptr;
};

ParametersLiteral make_parameters(const GpuConfig &config)
{
    config.validate();
    ParametersLiteral parameters(
        CKKS,
        config.log_n,
        config.log_slots,
        config.log_scale,
        /*hamming_weight=*/0,
        config.q0_level,
        Modulus(0),
        {},
        {},
        sec_level_type::none);
    parameters.set_log_modulus(config.log_q, config.log_p);
    return parameters;
}

ParametersLiteral make_level_keyswitch_parameters(
    const ParametersLiteral &global_parameters,
    std::size_t q_count,
    std::size_t p_count)
{
    if (q_count < 2 || q_count > global_parameters.q().size() ||
        p_count < 2 || p_count > q_count ||
        p_count > global_parameters.p().size())
    {
        throw std::invalid_argument(
            "level-aware key switching requires 2 <= P <= Q and P <= global P");
    }

    ParametersLiteral parameters = global_parameters;
    const std::vector<Modulus> q(
        global_parameters.q().begin(),
        global_parameters.q().begin() + static_cast<std::ptrdiff_t>(q_count));
    const std::vector<Modulus> p(
        global_parameters.p().begin(),
        global_parameters.p().begin() + static_cast<std::ptrdiff_t>(p_count));
    parameters.set_modulus(q, p);
    return parameters;
}

SecretKey make_level_keyswitch_secret_key(
    const KeyGenerator &global_keygen,
    const ParametersLiteral &global_parameters,
    const PoseidonContext &level_context,
    std::size_t q_count,
    std::size_t p_count)
{
    const std::size_t degree = global_parameters.degree();
    const std::size_t global_q_count = global_parameters.q().size();
    const std::size_t global_p_count = global_parameters.p().size();
    const auto &source = global_keygen.secret_key().data();
    const std::size_t expected_source_words =
        degree * (global_q_count + global_p_count);
    if (source.coeff_count() != expected_source_words)
    {
        throw std::logic_error(
            "global secret-key storage does not match the Q/P parameter shape");
    }

    SecretKey result;
    const std::size_t target_words = degree * (q_count + p_count);
    result.data().resize(
        level_context,
        level_context.crt_context()->key_parms_id(),
        target_words);

    // The level context reuses the same Q and P prime prefixes. Copying
    // their NTT residues preserves the exact ternary secret polynomial while
    // rebinding it to the level-specific Q_q union P_q key context.
    std::copy_n(source.data(), degree * q_count, result.data().data());
    std::copy_n(
        source.data() + degree * global_q_count,
        degree * p_count,
        result.data().data() + degree * q_count);
    return result;
}

class LevelAwareKeySwitchRuntime
{
public:
    LevelAwareKeySwitchRuntime(
        const ParametersLiteral &global_parameters,
        const KeyGenerator &global_keygen,
        std::size_t q_count,
        std::size_t p_count,
        const std::vector<int> &rotation_steps,
        int device_id,
        bool create_relinearization_key = true)
        : q_count(q_count),
          p_count(p_count),
          effective_dnum((q_count + p_count - 1) / p_count),
          parameters(make_level_keyswitch_parameters(
              global_parameters, q_count, p_count)),
          context(parameters)
    {
        auto compatible_secret = make_level_keyswitch_secret_key(
            global_keygen, global_parameters, context, q_count, p_count);
        KeyGenerator level_keygen(context, compatible_secret);

        gpu_parameters =
            std::make_unique<gpu::GpuParameterData>(context, device_id);
        gpu_evaluator =
            std::make_unique<gpu::GpuEvaluator>(*gpu_parameters);
        if (create_relinearization_key)
        {
            RelinKeys host_relin_keys;
            level_keygen.create_relin_keys(host_relin_keys);
            gpu_relin_keys = gpu::GpuUploader::upload_relin_keys(
                host_relin_keys, device_id);
        }
        if (!rotation_steps.empty())
        {
            GaloisKeys host_galois_keys;
            level_keygen.create_galois_keys(rotation_steps, host_galois_keys);
            gpu_galois_keys = gpu::GpuUploader::upload_galois_keys(
                host_galois_keys, device_id);
        }
    }

    std::size_t q_count;
    std::size_t p_count;
    std::size_t effective_dnum;
    ParametersLiteral parameters;
    PoseidonContext context;
    std::unique_ptr<gpu::GpuParameterData> gpu_parameters;
    std::unique_ptr<gpu::GpuEvaluator> gpu_evaluator;
    gpu::GpuRelinKeysData gpu_relin_keys;
    gpu::GpuGaloisKeysData gpu_galois_keys;
    mutable gpu::GpuDoubleHoistWorkspace rotate_many_workspace;
};

LinearMatrixGroup make_dynamic_dft_group(
    const PoseidonContext &context,
    CKKSEncoder &encoder,
    LinearType type,
    std::uint32_t level_start,
    double scaling,
    double input_scale,
    double minimum_scale,
    double value_normalization)
{
    HomomorphicDFTMatrixLiteral literal(
        type,
        context.parameters_literal()->log_n(),
        context.parameters_literal()->log_slots(),
        level_start,
        std::vector<std::uint32_t>(3, 1),
        /*repack_imag_to_real=*/true,
        scaling,
        /*bit_reversed=*/false,
        /*log_bsgs_ratio=*/1);
    LinearMatrixGroup result;
    literal.create_dynamic(
        result,
        encoder,
        input_scale,
        minimum_scale,
        minimum_scale,
        value_normalization);
    return result;
}

std::vector<std::uint32_t> dft_rescale_counts(
    const LinearMatrixGroup &group)
{
    if (!group.rescale_counts().empty())
    {
        return group.rescale_counts();
    }
    return std::vector<std::uint32_t>(
        group.data().size(), std::max(group.step(), std::uint32_t{1}));
}

double planned_dft_output_scale(
    const PoseidonContext &context,
    double input_scale,
    const LinearMatrixGroup &group)
{
    const auto first = context.crt_context()->first_context_data();
    if (!first || group.data().empty())
    {
        throw std::invalid_argument("GPU bootstrap DFT has no context or matrices");
    }
    const auto counts = dft_rescale_counts(group);
    std::size_t q_count = static_cast<std::size_t>(group.data().front().level) + 1;
    double scale = input_scale;
    for (std::size_t stage = 0; stage < group.data().size(); ++stage)
    {
        scale *= group.data()[stage].scale;
        for (std::uint32_t drop = 0; drop < counts[stage]; ++drop)
        {
            scale /= static_cast<double>(
                first->coeff_modulus().at(q_count - 1).value());
            --q_count;
        }
    }
    return scale;
}

GaloisKeys make_bootstrap_galois_keys(
    const PoseidonContext &context,
    KeyGenerator &keygen,
    const LinearMatrixGroup &coeff_to_slot,
    const LinearMatrixGroup &slot_to_coeff)
{
    std::set<int> steps{ 0 };
    steps.insert(coeff_to_slot.rot_index().begin(), coeff_to_slot.rot_index().end());
    steps.insert(slot_to_coeff.rot_index().begin(), slot_to_coeff.rot_index().end());
    // Report which power-of-two rotations are already covered by the
    // DFT key set. Missing rotations are composed from this set by the runtime
    // rather than allocating a second full-chain key set.
    const std::size_t slot_count = std::size_t{1}
        << context.parameters_literal()->log_slots();
    std::cerr << "[GPU bootstrap] DFT rotation steps=" << steps.size()
              << " missing powers=";
    bool first_missing = true;
    for (std::size_t step = 1; step < slot_count; step <<= 1)
    {
        if (steps.count(static_cast<int>(step)) == 0)
        {
            std::cerr << (first_missing ? "" : ",") << step;
            first_missing = false;
        }
    }
    std::cerr << (first_missing ? "none" : "") << '\n';
    const auto tool = context.crt_context()->galois_tool();
    std::vector<std::uint32_t> elements;
    elements.reserve(steps.size());
    for (const int step : steps)
    {
        elements.push_back(tool->get_elt_from_step(step));
    }
    GaloisKeys result;
    keygen.create_galois_keys(elements, result);
    return result;
}

std::vector<std::size_t> required_dft_key_q_counts(
    std::size_t input_q_count,
    const LinearMatrixGroup &group,
    bool include_conjugation)
{
    const auto counts = dft_rescale_counts(group);
    std::vector<std::size_t> result;
    std::size_t current = input_q_count;
    for (std::size_t matrix = 0; matrix < group.data().size(); ++matrix)
    {
        result.push_back(current);
        if (counts[matrix] == 0 || counts[matrix] >= current)
        {
            throw std::invalid_argument("GPU bootstrap DFT consumes its modulus chain");
        }
        current -= counts[matrix];
    }
    if (include_conjugation)
    {
        result.push_back(current);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

using BootstrapRotationRequirements =
    std::map<std::size_t, std::set<int>>;

void add_matrix_rotation_requirements(
    const MatrixPlain &matrix,
    std::set<int> &steps)
{
    const auto [index, unused_giant_steps, baby_steps] = poseidon::bsgs_index(
        matrix.plain_vec,
        1 << matrix.log_slots,
        static_cast<int>(matrix.n1));
    (void)unused_giant_steps;
    for (const int step : baby_steps)
    {
        if (step != 0)
        {
            steps.insert(step);
        }
    }
    for (const auto &entry : index)
    {
        if (entry.first != 0)
        {
            steps.insert(entry.first);
        }
    }
}

std::size_t add_dft_rotation_requirements(
    std::size_t input_q_count,
    const LinearMatrixGroup &group,
    bool conjugate_output,
    BootstrapRotationRequirements &requirements)
{
    const auto counts = dft_rescale_counts(group);
    std::size_t current_q_count = input_q_count;
    for (std::size_t stage = 0; stage < group.data().size(); ++stage)
    {
        add_matrix_rotation_requirements(
            group.data()[stage], requirements[current_q_count]);
        if (counts[stage] == 0 || counts[stage] >= current_q_count)
        {
            throw std::invalid_argument(
                "GPU bootstrap DFT consumes its modulus chain");
        }
        current_q_count -= counts[stage];
    }
    if (conjugate_output)
    {
        // Poseidon uses step zero when asking KeyGenerator for the
        // conjugation Galois element.
        requirements[current_q_count].insert(0);
    }
    return current_q_count;
}

}  // namespace

class GpuCkksRuntime::Impl
{
public:
    Impl(const GpuConfig &config, int device_id)
        : device_id(device_id),
          rmm_scope(device_id),
          parameters(make_parameters(config)),
          context(parameters),
          encoder(context),
          keygen(context),
          gpu_parameters(context, device_id),
          gpu_evaluator(gpu_parameters),
          config(config),
          application_scale(std::ldexp(1.0, config.application_log_scale)),
          application_parms_id(
              gpu_parameters.get_level_by_q_count(config.application_q_count()).parms_id),
          trace_rotation_steps(
              environment_flag_enabled("POSEIDON_TRACE_ROTATION_STEPS"))
    {
        keygen.create_public_key(public_key);
        encryptor = std::make_unique<Encryptor>(
            context, public_key, keygen.secret_key());
        decryptor = std::make_unique<Decryptor>(context, keygen.secret_key());
    }

    int device_id;
    // Must be declared before every GPU allocation so it is destroyed last.
    RmmPoolScope rmm_scope;
    ParametersLiteral parameters;
    PoseidonContext context;
    CKKSEncoder encoder;
    KeyGenerator keygen;
    PublicKey public_key;
    std::unique_ptr<Encryptor> encryptor;
    std::unique_ptr<Decryptor> decryptor;
    gpu::GpuParameterData gpu_parameters;
    gpu::GpuEvaluator gpu_evaluator;
    GpuConfig config;
    double application_scale;
    parms_id_type application_parms_id;
    std::unique_ptr<gpu::GpuRelinKeysData> gpu_relin_keys;
    std::unique_ptr<gpu::GpuGaloisKeysData> gpu_galois_keys;
    std::unique_ptr<RelinKeys> host_relin_keys;
    std::unique_ptr<gpu::GpuGaloisKeysData> gpu_bootstrap_galois_keys;
    std::unique_ptr<gpu::GpuBootstrapData> bootstrap_data;
    std::unique_ptr<gpu::GpuBootstrapWorkspace> bootstrap_workspace;
    std::unordered_map<std::string, gpu::GpuPlaintextData> plaintext_cache;
    std::unordered_map<std::string, gpu::GpuCiphertextData> ciphertext_cache;
    std::unique_ptr<gpu::GpuCiphertextData> encrypted_one_cache;
    std::set<int> direct_rotation_steps;
    std::map<std::size_t, std::set<int>> direct_rotation_steps_by_q;
    mutable gpu::GpuDoubleHoistWorkspace rotate_many_workspace;
    // Several active Q levels can share a fixed-dnum key generated at the
    // largest Q using the same P prefix (for example Q29..Q32 share Q32/P8).
    std::unordered_map<
        std::size_t,
        std::shared_ptr<LevelAwareKeySwitchRuntime>> level_keyswitch_runtimes;
    std::unordered_map<
        std::size_t,
        std::unique_ptr<LevelAwareKeySwitchRuntime>>
        bootstrap_level_keyswitch_runtimes;
    bool use_direct_rotation_keys = false;
    bool level_aware_application_keyswitch_enabled = false;
    ApplicationKeySwitchPMode application_keyswitch_p_mode =
        ApplicationKeySwitchPMode::single_digit;
    bool full_device_cache = false;
    bool trace_rotation_steps = false;
    mutable std::map<std::size_t, std::set<int>> observed_rotation_steps_by_q;
};

GpuCkksRuntime::GpuCkksRuntime(const GpuConfig &config, int device_id)
    : impl_(std::make_unique<Impl>(config, device_id))
{}

GpuCkksRuntime::~GpuCkksRuntime() = default;

GpuCkksRuntime::DeviceCiphertext
GpuCkksRuntime::encrypt(const std::vector<double> &slots) const
{
    std::string cache_key;
    if (impl_->full_device_cache)
    {
        cache_key = vector_cache_key("encrypted_input", slots);
        const auto found = impl_->ciphertext_cache.find(cache_key);
        if (found != impl_->ciphertext_cache.end())
        {
            DeviceCiphertext copy;
            impl_->gpu_evaluator.multiply_scalar(found->second, 1, copy);
            return copy;
        }
    }
    Plaintext plaintext;
    impl_->encoder.encode(
        slots,
        impl_->application_parms_id,
        impl_->application_scale,
        plaintext);

    Ciphertext ciphertext;
    impl_->encryptor->encrypt(plaintext, ciphertext);
    auto uploaded =
        gpu::GpuUploader::upload_ciphertext(ciphertext, impl_->device_id);
    if (!impl_->full_device_cache)
    {
        return uploaded;
    }
    auto found = impl_->ciphertext_cache.emplace(
        std::move(cache_key), std::move(uploaded)).first;
    DeviceCiphertext copy;
    impl_->gpu_evaluator.multiply_scalar(found->second, 1, copy);
    return copy;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::encrypt_constant(double value) const
{
    if (value == 1.0 && impl_->encrypted_one_cache)
    {
        DeviceCiphertext copy;
        impl_->gpu_evaluator.multiply_scalar(
            *impl_->encrypted_one_cache, 1, copy);
        return copy;
    }
    Plaintext plaintext;
    impl_->encoder.encode(
        value,
        impl_->application_parms_id,
        impl_->application_scale,
        plaintext);
    Ciphertext ciphertext;
    impl_->encryptor->encrypt(plaintext, ciphertext);
    auto uploaded =
        gpu::GpuUploader::upload_ciphertext(ciphertext, impl_->device_id);
    if (value == 1.0)
    {
        impl_->encrypted_one_cache =
            std::make_unique<DeviceCiphertext>(std::move(uploaded));
        DeviceCiphertext copy;
        impl_->gpu_evaluator.multiply_scalar(
            *impl_->encrypted_one_cache, 1, copy);
        return copy;
    }
    return uploaded;
}

std::vector<std::complex<double>>
GpuCkksRuntime::decrypt(const DeviceCiphertext &ciphertext) const
{
    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    Ciphertext host_ciphertext;
    gpu::GpuUploader::download_ciphertext(
        ciphertext, host_ciphertext, impl_->context);
    Plaintext plaintext;
    impl_->decryptor->decrypt(host_ciphertext, plaintext);
    std::vector<std::complex<double>> result;
    impl_->encoder.decode(plaintext, result);
    return result;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::multiply_plain_rescale(
    const DeviceCiphertext &source,
    const std::vector<double> &plain_slots) const
{
    if (impl_->full_device_cache)
    {
        return multiply_plain_rescale_cached(
            source, plain_slots, vector_cache_key("multiply_rescale", plain_slots));
    }
    const auto context_data =
        impl_->context.crt_context()->get_context_data(source.meta.parms_id);
    if (!context_data || context_data->coeff_modulus().empty())
    {
        throw std::invalid_argument(
            "multiply_plain_rescale received an unknown or empty modulus level");
    }

    // Encoding at the exact value of the q prime being removed minimizes the
    // post-rescale scale drift: (source.scale * q_last) / q_last.
    const double plain_scale = static_cast<double>(
        context_data->coeff_modulus().back().value());
    Plaintext host_plaintext;
    impl_->encoder.encode(
        plain_slots, source.meta.parms_id, plain_scale, host_plaintext);
    auto device_plaintext = gpu::GpuUploader::upload_plaintext(
        host_plaintext, impl_->device_id);

    DeviceCiphertext product;
    impl_->gpu_evaluator.multiply_plain(source, device_plaintext, product);
    DeviceCiphertext result;
    impl_->gpu_evaluator.rescale(product, result);
    return result;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::multiply_plain_rescale_cached(
    const DeviceCiphertext &source,
    const std::vector<double> &plain_slots,
    const std::string &cache_key) const
{
    const auto context_data =
        impl_->context.crt_context()->get_context_data(source.meta.parms_id);
    if (!context_data || context_data->coeff_modulus().empty())
    {
        throw std::invalid_argument(
            "multiply_plain_rescale_cached received an unknown modulus level");
    }
    const double plain_scale = static_cast<double>(
        context_data->coeff_modulus().back().value());
    const auto full_key = cache_key + ":q=" +
                          std::to_string(source.meta.q_count);
    auto found = impl_->plaintext_cache.find(full_key);
    if (found == impl_->plaintext_cache.end())
    {
        Plaintext host_plaintext;
        impl_->encoder.encode(
            plain_slots, source.meta.parms_id, plain_scale, host_plaintext);
        auto uploaded = gpu::GpuUploader::upload_plaintext(
            host_plaintext, impl_->device_id);
        found = impl_->plaintext_cache.emplace(
            full_key, std::move(uploaded)).first;
    }
    DeviceCiphertext product;
    impl_->gpu_evaluator.multiply_plain(source, found->second, product);
    return rescale(product, 1);
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::multiply_plain(
    const DeviceCiphertext &source,
    const std::vector<double> &plain_slots,
    double plain_scale) const
{
    if (impl_->full_device_cache)
    {
        const auto key = vector_cache_key("multiply_lazy", plain_slots) +
                         ":q=" + std::to_string(source.meta.q_count) +
                         ":s=" + std::to_string(double_bits(plain_scale));
        auto found = impl_->plaintext_cache.find(key);
        if (found == impl_->plaintext_cache.end())
        {
            auto uploaded = encode_and_upload_plain(
                source, plain_slots, plain_scale);
            found = impl_->plaintext_cache.emplace(
                key, std::move(uploaded)).first;
        }
        return multiply_plain_preencoded(source, found->second);
    }
    auto device_plaintext = encode_and_upload_plain(
        source, plain_slots, plain_scale);
    return multiply_plain_preencoded(source, device_plaintext);
}

void GpuCkksRuntime::multiply_plain_accumulate(
    const DeviceCiphertext &source,
    const std::vector<double> &plain_slots,
    double plain_scale,
    DeviceCiphertext &destination) const
{
    if (!(plain_scale > 0.0) || !std::isfinite(plain_scale))
    {
        throw std::invalid_argument(
            "multiply_plain_accumulate requires a positive finite scale");
    }

    if (impl_->full_device_cache)
    {
        const auto key = vector_cache_key("multiply_lazy", plain_slots) +
                         ":q=" + std::to_string(source.meta.q_count) +
                         ":s=" + std::to_string(double_bits(plain_scale));
        auto found = impl_->plaintext_cache.find(key);
        if (found == impl_->plaintext_cache.end())
        {
            auto uploaded = encode_and_upload_plain(
                source, plain_slots, plain_scale);
            found = impl_->plaintext_cache.emplace(
                key, std::move(uploaded)).first;
        }
        impl_->gpu_evaluator.multiply_plain_accumulate(
            source, found->second, destination);
        return;
    }

    auto plaintext = encode_and_upload_plain(
        source, plain_slots, plain_scale);
    impl_->gpu_evaluator.multiply_plain_accumulate(
        source, plaintext, destination);
}

GpuCkksRuntime::DevicePlaintext GpuCkksRuntime::encode_and_upload_plain(
    const DeviceCiphertext &source,
    const std::vector<double> &plain_slots,
    double plain_scale) const
{
    if (!(plain_scale > 0.0) || !std::isfinite(plain_scale))
    {
        throw std::invalid_argument("multiply_plain requires a positive finite scale");
    }
    Plaintext host_plaintext;
    impl_->encoder.encode(
        plain_slots, source.meta.parms_id, plain_scale, host_plaintext);
    return gpu::GpuUploader::upload_plaintext(
        host_plaintext, impl_->device_id);
}

std::vector<GpuCkksRuntime::DevicePlaintext>
GpuCkksRuntime::encode_and_upload_plain_batch(
    const DeviceCiphertext &source,
    const std::vector<std::vector<double>> &plain_slots,
    double plain_scale) const
{
    if (!(plain_scale > 0.0) || !std::isfinite(plain_scale))
    {
        throw std::invalid_argument(
            "encode_and_upload_plain_batch requires a positive finite scale");
    }
    if (plain_slots.empty())
    {
        return {};
    }
    for (const auto &slots : plain_slots)
    {
        if (slots.size() > slot_count())
        {
            throw std::invalid_argument(
                "encode_and_upload_plain_batch input exceeds slot count");
        }
    }

    std::vector<Plaintext> host_plaintexts(plain_slots.size());
    int thread_count = static_cast<int>(std::min<std::size_t>(
        plain_slots.size(), 8));
    if (const char *text =
            std::getenv("POSEIDON_GPU_QWEN_ENCODE_THREADS"))
    {
        const int requested = std::stoi(text);
        if (requested <= 0)
        {
            throw std::invalid_argument(
                "POSEIDON_GPU_QWEN_ENCODE_THREADS must be positive");
        }
        thread_count = static_cast<int>(std::min<std::size_t>(
            plain_slots.size(), static_cast<std::size_t>(requested)));
    }

    std::exception_ptr encoding_error;
    std::mutex encoding_error_mutex;
#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(plain_slots.size()); ++index)
    {
        try
        {
            impl_->encoder.encode(
                plain_slots[static_cast<std::size_t>(index)],
                source.meta.parms_id,
                plain_scale,
                host_plaintexts[static_cast<std::size_t>(index)]);
        }
        catch (...)
        {
            std::lock_guard<std::mutex> guard(encoding_error_mutex);
            if (!encoding_error)
            {
                encoding_error = std::current_exception();
            }
        }
    }
    if (encoding_error)
    {
        std::rethrow_exception(encoding_error);
    }

    std::vector<DevicePlaintext> result;
    result.reserve(host_plaintexts.size());
    for (const auto &plaintext : host_plaintexts)
    {
        result.push_back(gpu::GpuUploader::upload_plaintext(
            plaintext, impl_->device_id));
    }
    return result;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::multiply_plain_preencoded(
    const DeviceCiphertext &source,
    const DevicePlaintext &plaintext) const
{
    if (source.meta.parms_id != plaintext.meta.parms_id)
    {
        throw std::invalid_argument(
            "multiply_plain_preencoded parameter levels do not match");
    }
    DeviceCiphertext result;
    impl_->gpu_evaluator.multiply_plain(source, plaintext, result);
    return result;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::multiply_plain_scalar_rescale(
    const DeviceCiphertext &source,
    double value) const
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument(
            "multiply_plain_scalar_rescale requires a finite scalar");
    }
    const auto context_data =
        impl_->context.crt_context()->get_context_data(source.meta.parms_id);
    if (!context_data || context_data->coeff_modulus().empty())
    {
        throw std::invalid_argument(
            "multiply_plain_scalar_rescale received an unknown modulus level");
    }
    const double plain_scale = static_cast<double>(
        context_data->coeff_modulus().back().value());
    if (std::abs(value) * plain_scale < 0.5)
    {
        // The scalar rounds to zero at this plaintext scale. Poseidon's scalar
        // encoder computes log2(abs(value * scale)); values in (0, 0.5) would
        // otherwise underflow its unsigned bit-count calculation. Preserve the
        // normal multiply/rescale metadata by producing a device-side zero and
        // dropping the same q prime that the product would consume.
        if (source.meta.q_count <= 1)
        {
            throw std::invalid_argument(
                "multiply_plain_scalar_rescale cannot consume the final q prime");
        }
        auto zero = sub(source, source);
        return drop_to_q_count(zero, source.meta.q_count - 1);
    }
    const auto key = scalar_plaintext_cache_key(
        "multiply_scalar", source.meta.q_count, value, plain_scale);
    auto found = impl_->plaintext_cache.find(key);
    if (found == impl_->plaintext_cache.end())
    {
        Plaintext host_plaintext;
        impl_->encoder.encode(
            value, source.meta.parms_id, plain_scale, host_plaintext);
        auto uploaded = gpu::GpuUploader::upload_plaintext(
            host_plaintext, impl_->device_id);
        found = impl_->plaintext_cache.emplace(
            key, std::move(uploaded)).first;
    }
    DeviceCiphertext product;
    impl_->gpu_evaluator.multiply_plain(source, found->second, product);
    return rescale(product, 1);
}

double GpuCkksRuntime::last_modulus_value(
    const DeviceCiphertext &source) const
{
    return modulus_value_from_end(source, 0);
}

double GpuCkksRuntime::modulus_value_from_end(
    const DeviceCiphertext &source,
    std::size_t reverse_index) const
{
    const auto context_data =
        impl_->context.crt_context()->get_context_data(source.meta.parms_id);
    if (!context_data ||
        context_data->coeff_modulus().size() <= reverse_index)
    {
        throw std::invalid_argument(
            "modulus_value_from_end received an unavailable modulus prime");
    }
    const auto &moduli = context_data->coeff_modulus();
    return static_cast<double>(
        moduli[moduli.size() - reverse_index - 1].value());
}

void GpuCkksRuntime::initialize_evaluation_keys(
    const std::vector<int> &rotation_steps)
{
    if (!impl_->gpu_relin_keys)
    {
        impl_->host_relin_keys = std::make_unique<RelinKeys>();
        impl_->keygen.create_relin_keys(*impl_->host_relin_keys);
        auto uploaded = gpu::GpuUploader::upload_relin_keys(
            *impl_->host_relin_keys, impl_->device_id);
        impl_->gpu_relin_keys =
            std::make_unique<gpu::GpuRelinKeysData>(std::move(uploaded));
    }

    if (!rotation_steps.empty())
    {
        GaloisKeys host_galois_keys;
        impl_->keygen.create_galois_keys(rotation_steps, host_galois_keys);
        auto uploaded = gpu::GpuUploader::upload_galois_keys(
            host_galois_keys, impl_->device_id);
        impl_->gpu_galois_keys =
            std::make_unique<gpu::GpuGaloisKeysData>(std::move(uploaded));
    }
}

bool GpuCkksRuntime::evaluation_keys_ready() const noexcept
{
    return static_cast<bool>(impl_->gpu_relin_keys);
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::add(
    const DeviceCiphertext &left,
    const DeviceCiphertext &right) const
{
    DeviceCiphertext result;
    impl_->gpu_evaluator.add(left, right, result);
    return result;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::drop_to_q_count(
    const DeviceCiphertext &source,
    std::size_t target_q_count) const
{
    if (target_q_count > source.meta.q_count || target_q_count == 0)
    {
        throw std::invalid_argument("drop_to_q_count target is not a lower valid level");
    }
    if (target_q_count == source.meta.q_count)
    {
        DeviceCiphertext copy;
        impl_->gpu_evaluator.multiply_scalar(source, 1, copy);
        return copy;
    }
    const auto &target = impl_->gpu_parameters.get_level_by_q_count(target_q_count);
    DeviceCiphertext result;
    impl_->gpu_evaluator.drop_modulus(source, result, target.parms_id);
    return result;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::add_aligned(
    const DeviceCiphertext &left,
    const DeviceCiphertext &right) const
{
    const std::size_t target_q_count =
        std::min(left.meta.q_count, right.meta.q_count);
    std::unique_ptr<DeviceCiphertext> adjusted_left;
    std::unique_ptr<DeviceCiphertext> adjusted_right;
    const DeviceCiphertext *left_view = &left;
    const DeviceCiphertext *right_view = &right;
    if (left.meta.q_count != target_q_count)
    {
        adjusted_left = std::make_unique<DeviceCiphertext>(
            drop_to_q_count(left, target_q_count));
        left_view = adjusted_left.get();
    }
    if (right.meta.q_count != target_q_count)
    {
        adjusted_right = std::make_unique<DeviceCiphertext>(
            drop_to_q_count(right, target_q_count));
        right_view = adjusted_right.get();
    }
    return add(*left_view, *right_view);
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::sub(
    const DeviceCiphertext &left,
    const DeviceCiphertext &right) const
{
    DeviceCiphertext result;
    impl_->gpu_evaluator.sub(left, right, result);
    return result;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::sub_aligned(
    const DeviceCiphertext &left,
    const DeviceCiphertext &right) const
{
    const std::size_t target_q_count =
        std::min(left.meta.q_count, right.meta.q_count);
    std::unique_ptr<DeviceCiphertext> adjusted_left;
    std::unique_ptr<DeviceCiphertext> adjusted_right;
    const DeviceCiphertext *left_view = &left;
    const DeviceCiphertext *right_view = &right;
    if (left.meta.q_count != target_q_count)
    {
        adjusted_left = std::make_unique<DeviceCiphertext>(
            drop_to_q_count(left, target_q_count));
        left_view = adjusted_left.get();
    }
    if (right.meta.q_count != target_q_count)
    {
        adjusted_right = std::make_unique<DeviceCiphertext>(
            drop_to_q_count(right, target_q_count));
        right_view = adjusted_right.get();
    }
    return sub(*left_view, *right_view);
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::add_plain(
    const DeviceCiphertext &source,
    const std::vector<double> &plain_slots) const
{
    if (impl_->full_device_cache)
    {
        const auto key = vector_cache_key("add_plain", plain_slots) + ":q=" +
                         std::to_string(source.meta.q_count) + ":scale=" +
                         std::to_string(double_bits(source.meta.scale));
        auto found = impl_->plaintext_cache.find(key);
        if (found == impl_->plaintext_cache.end())
        {
            Plaintext host_plaintext;
            impl_->encoder.encode(
                plain_slots,
                source.meta.parms_id,
                source.meta.scale,
                host_plaintext);
            auto uploaded = gpu::GpuUploader::upload_plaintext(
                host_plaintext, impl_->device_id);
            found = impl_->plaintext_cache.emplace(
                key, std::move(uploaded)).first;
        }
        DeviceCiphertext result;
        impl_->gpu_evaluator.add_plain(source, found->second, result);
        return result;
    }
    Plaintext host_plaintext;
    impl_->encoder.encode(
        plain_slots,
        source.meta.parms_id,
        source.meta.scale,
        host_plaintext);
    auto device_plaintext = gpu::GpuUploader::upload_plaintext(
        host_plaintext, impl_->device_id);
    DeviceCiphertext result;
    impl_->gpu_evaluator.add_plain(source, device_plaintext, result);
    return result;
}

void GpuCkksRuntime::enable_full_device_cache(bool enable)
{
    impl_->full_device_cache = enable;
}

void GpuCkksRuntime::synchronize() const
{
    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::add_plain_scalar(
    const DeviceCiphertext &source,
    double value) const
{
    const auto key = scalar_plaintext_cache_key(
        "add_scalar", source.meta.q_count, value, source.meta.scale);
    auto found = impl_->plaintext_cache.find(key);
    if (found == impl_->plaintext_cache.end())
    {
        Plaintext host_plaintext;
        impl_->encoder.encode(
            value, source.meta.parms_id, source.meta.scale, host_plaintext);
        auto uploaded = gpu::GpuUploader::upload_plaintext(
            host_plaintext, impl_->device_id);
        found = impl_->plaintext_cache.emplace(
            key, std::move(uploaded)).first;
    }
    DeviceCiphertext result;
    impl_->gpu_evaluator.add_plain(source, found->second, result);
    return result;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::rotate(
    const DeviceCiphertext &source,
    int step) const
{
    const auto level_runtime = impl_->level_keyswitch_runtimes.find(
        source.meta.q_count);
    if (impl_->use_direct_rotation_keys &&
        level_runtime != impl_->level_keyswitch_runtimes.end())
    {
        const auto planned = impl_->direct_rotation_steps_by_q.find(
            source.meta.q_count);
        long long normalized = static_cast<long long>(step) %
                               static_cast<long long>(slot_count());
        if (normalized < 0)
        {
            normalized += static_cast<long long>(slot_count());
        }
        if (planned == impl_->direct_rotation_steps_by_q.end() ||
            planned->second.count(static_cast<int>(normalized)) == 0)
        {
            throw std::logic_error(
                "direct rotation key is missing at the ciphertext Q level");
        }
        DeviceCiphertext result;
        level_runtime->second->gpu_evaluator->rotate(
            source,
            step,
            level_runtime->second->gpu_galois_keys,
            result);
        return result;
    }

    const gpu::GpuGaloisKeysData *keys = impl_->gpu_galois_keys
        ? impl_->gpu_galois_keys.get()
        : impl_->gpu_bootstrap_galois_keys.get();
    if (impl_->use_direct_rotation_keys && !impl_->gpu_galois_keys)
    {
        throw std::logic_error(
            "direct rotation keys were not prepared for this q level");
    }
    if (!keys)
    {
        throw std::logic_error(
            "rotate requires inference or bootstrap Galois keys");
    }
    DeviceCiphertext result;
    impl_->gpu_evaluator.rotate(
        source, step, *keys, result);
    return result;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::rotate_composed(
    const DeviceCiphertext &source,
    long long step) const
{
    const long long slots = static_cast<long long>(slot_count());
    long long remaining = step % slots;
    if (remaining < 0)
    {
        remaining += slots;
    }
    if (remaining == 0)
    {
        return drop_to_q_count(source, source.meta.q_count);
    }

    const int direct_step = static_cast<int>(remaining);
    if (impl_->trace_rotation_steps)
    {
        impl_->observed_rotation_steps_by_q[source.meta.q_count].insert(
            direct_step);
    }
    if (impl_->use_direct_rotation_keys)
    {
        const auto planned = impl_->direct_rotation_steps_by_q.find(
            source.meta.q_count);
        if (planned == impl_->direct_rotation_steps_by_q.end() ||
            planned->second.count(direct_step) == 0)
        {
            throw std::logic_error(
                "direct rotation key is missing for the requested Q/step");
        }
        return rotate(source, direct_step);
    }

    std::unique_ptr<DeviceCiphertext> current;
    for (int bit = 1; remaining != 0; bit <<= 1)
    {
        if ((remaining & bit) == 0)
        {
            continue;
        }
        if (current)
        {
            current = std::make_unique<DeviceCiphertext>(rotate(*current, bit));
        }
        else
        {
            current = std::make_unique<DeviceCiphertext>(rotate(source, bit));
        }
        remaining -= bit;
    }
    return std::move(*current);
}

std::vector<GpuCkksRuntime::DeviceCiphertext>
GpuCkksRuntime::rotate_many_composed(
    const DeviceCiphertext &source,
    const std::vector<long long> &steps) const
{
    std::vector<DeviceCiphertext> result(steps.size());
    if (steps.empty())
    {
        return result;
    }
    if (!impl_->use_direct_rotation_keys)
    {
        for (std::size_t index = 0; index < steps.size(); ++index)
        {
            result[index] = rotate_composed(source, steps[index]);
        }
        return result;
    }

    const long long slots = static_cast<long long>(slot_count());
    std::vector<int> direct_steps;
    std::vector<std::size_t> direct_indices;
    direct_steps.reserve(steps.size());
    direct_indices.reserve(steps.size());
    for (std::size_t index = 0; index < steps.size(); ++index)
    {
        long long normalized = steps[index] % slots;
        if (normalized < 0)
        {
            normalized += slots;
        }
        if (normalized == 0)
        {
            result[index] = drop_to_q_count(source, source.meta.q_count);
            continue;
        }
        const int direct_step = static_cast<int>(normalized);
        if (impl_->trace_rotation_steps)
        {
            impl_->observed_rotation_steps_by_q[source.meta.q_count].insert(
                direct_step);
        }
        const auto planned = impl_->direct_rotation_steps_by_q.find(
            source.meta.q_count);
        if (planned == impl_->direct_rotation_steps_by_q.end() ||
            planned->second.count(direct_step) == 0)
        {
            throw std::logic_error(
                "direct rotation key is missing for a hoisted Q/step");
        }
        direct_steps.push_back(direct_step);
        direct_indices.push_back(index);
    }

    if (!direct_steps.empty())
    {
        std::vector<DeviceCiphertext> direct_results;
        const auto level_runtime = impl_->level_keyswitch_runtimes.find(
            source.meta.q_count);
        if (level_runtime != impl_->level_keyswitch_runtimes.end())
        {
            level_runtime->second->gpu_evaluator->rotate_many_hoisted(
                source,
                direct_steps,
                level_runtime->second->gpu_galois_keys,
                level_runtime->second->rotate_many_workspace,
                direct_results);
        }
        else
        {
            if (!impl_->gpu_galois_keys)
            {
                throw std::logic_error(
                    "direct rotation keys were not prepared for this q level");
            }
            impl_->gpu_evaluator.rotate_many_hoisted(
                source,
                direct_steps,
                *impl_->gpu_galois_keys,
                impl_->rotate_many_workspace,
                direct_results);
        }
        for (std::size_t index = 0; index < direct_results.size(); ++index)
        {
            result[direct_indices[index]] = std::move(direct_results[index]);
        }
    }
    return result;
}

GpuCkksRuntime::ApplicationKeySwitchShape
GpuCkksRuntime::application_keyswitch_shape(std::size_t q_count) const
{
    if (q_count < 2 || q_count > impl_->parameters.q().size())
    {
        throw std::invalid_argument(
            "application KeySwitch q_count is outside the global Q chain");
    }

    const std::size_t global_p_count = impl_->parameters.p().size();
    ApplicationKeySwitchShape result;
    result.q_count = q_count;
    result.p_count = global_p_count;
    result.effective_dnum =
        (q_count + global_p_count - 1) / global_p_count;

    if (!impl_->level_aware_application_keyswitch_enabled)
    {
        return result;
    }

    if (impl_->application_keyswitch_p_mode ==
        ApplicationKeySwitchPMode::single_digit)
    {
        if (q_count <= global_p_count)
        {
            result.p_count = q_count;
            result.effective_dnum = 1;
            result.level_aware = true;
        }
        return result;
    }

    const std::size_t p_count = fixed_dnum_p_count(
        q_count, impl_->config.dnum);
    if (p_count > global_p_count)
    {
        throw std::logic_error(
            "fixed-dnum application P exceeds the global P prefix");
    }
    result.p_count = p_count;
    result.effective_dnum = (q_count + p_count - 1) / p_count;
    // When the selected P is already the global P, the global key material
    // has exactly the requested decomposition and avoids a duplicate context.
    result.level_aware = p_count < global_p_count;
    return result;
}

void GpuCkksRuntime::initialize_direct_rotation_keys(
    const std::vector<int> &rotation_steps,
    const std::vector<std::size_t> &level_aware_q_counts)
{
    if (rotation_steps.empty())
    {
        throw std::logic_error(
            "direct rotation step list must not be empty");
    }

    std::vector<std::size_t> q_counts = level_aware_q_counts;
    if (q_counts.empty())
    {
        q_counts.push_back(impl_->config.application_q_count());
    }
    std::map<std::size_t, std::vector<int>> rotation_steps_by_q;
    for (const std::size_t q_count : q_counts)
    {
        rotation_steps_by_q.emplace(q_count, rotation_steps);
    }
    initialize_direct_rotation_keys(rotation_steps_by_q);
}

void GpuCkksRuntime::initialize_direct_rotation_keys(
    const std::map<std::size_t, std::vector<int>> &rotation_steps_by_q)
{
    if (rotation_steps_by_q.empty())
    {
        throw std::logic_error(
            "direct rotation level plan must not be empty");
    }

    impl_->direct_rotation_steps.clear();
    impl_->direct_rotation_steps_by_q.clear();
    const auto slots = static_cast<long long>(slot_count());
    for (const auto &[q_count, requested_steps] : rotation_steps_by_q)
    {
        if (q_count < 2 || q_count > impl_->parameters.q().size())
        {
            throw std::invalid_argument(
                "direct rotation q_count is outside the global Q chain");
        }
        auto &level_steps = impl_->direct_rotation_steps_by_q[q_count];
        for (const int step : requested_steps)
        {
            long long normalized = static_cast<long long>(step) % slots;
            if (normalized < 0)
            {
                normalized += slots;
            }
            if (normalized == 0)
            {
                throw std::invalid_argument(
                    "direct rotation level plan contains a zero rotation");
            }
            level_steps.insert(static_cast<int>(normalized));
            impl_->direct_rotation_steps.insert(static_cast<int>(normalized));
        }
        if (level_steps.empty())
        {
            throw std::invalid_argument(
                "direct rotation level plan contains an empty level");
        }
    }

    const std::vector<int> steps(
        impl_->direct_rotation_steps.begin(),
        impl_->direct_rotation_steps.end());
    std::cout << "[GPU ResNet20] generating direct rotation keys before "
                 "inference union_count=" << steps.size() << '\n';

    // Direct rotations and application-level relinearization share the
    // level-aware contexts below so their P basis follows the actual
    // ciphertext level instead of retaining every global P.
    initialize_evaluation_keys();
    impl_->level_keyswitch_runtimes.clear();
    impl_->gpu_galois_keys.reset();

    const char *level_aware_env =
        std::getenv("POSEIDON_LEVEL_AWARE_ROTATION_P");
    const bool level_aware_enabled =
        level_aware_env == nullptr || std::string(level_aware_env) != "0";
    impl_->level_aware_application_keyswitch_enabled = level_aware_enabled;
    impl_->application_keyswitch_p_mode =
        configured_application_keyswitch_p_mode();
    if (!level_aware_enabled)
    {
        std::cout << "[GPU ResNet20] level-aware rotation P disabled; "
                     "using global key-switch basis\n";
    }
    else
    {
        std::cout << "[GPU ResNet20] application KeySwitch P mode="
                  << application_keyswitch_p_mode_name(
                         impl_->application_keyswitch_p_mode)
                  << " target_dnum=" << impl_->config.dnum << '\n';
    }

    bool needs_global_rotation_keys = !level_aware_enabled;
    std::set<int> global_steps;
    for (const auto &[q_count, level_step_set] :
         impl_->direct_rotation_steps_by_q)
    {
        const std::vector<int> level_steps(
            level_step_set.begin(), level_step_set.end());
        const auto shape = application_keyswitch_shape(q_count);
        if (!shape.level_aware)
        {
            needs_global_rotation_keys = true;
            global_steps.insert(level_step_set.begin(), level_step_set.end());
            std::cout << "[GPU ResNet20] direct rotation level q=" << q_count
                      << " keys=" << level_steps.size()
                      << " uses global p=" << shape.p_count
                      << " effective_dnum=" << shape.effective_dnum
                      << '\n';
            continue;
        }

        auto level_runtime = std::make_shared<LevelAwareKeySwitchRuntime>(
            impl_->parameters,
            impl_->keygen,
            q_count,
            shape.p_count,
            level_steps,
            impl_->device_id);
        const auto primary_parms_id = impl_->context.crt_context()
            ->parms_id_map().at(static_cast<std::uint32_t>(q_count - 1));
        const auto level_parms_id = level_runtime->context.crt_context()
            ->parms_id_map().at(static_cast<std::uint32_t>(q_count - 1));
        if (primary_parms_id != level_parms_id)
        {
            throw std::logic_error(
                "level-aware rotation Q context is incompatible with ciphertext");
        }
        std::cout << "[GPU ResNet20] application keyswitch level q=" << q_count
                  << " p=" << level_runtime->p_count
                  << " target_dnum=" << impl_->config.dnum
                  << " effective_dnum=" << level_runtime->effective_dnum
                  << " keys=" << level_steps.size() << '\n';
        impl_->level_keyswitch_runtimes.emplace(
            q_count, std::move(level_runtime));
    }

    if (needs_global_rotation_keys)
    {
        const std::vector<int> uploaded_global_steps(
            global_steps.empty() ? impl_->direct_rotation_steps.begin()
                                 : global_steps.begin(),
            global_steps.empty() ? impl_->direct_rotation_steps.end()
                                 : global_steps.end());
        initialize_evaluation_keys(uploaded_global_steps);
    }
    impl_->use_direct_rotation_keys = true;
    std::cout << "[GPU ResNet20] direct rotation keys ready steps=";
    for (std::size_t index = 0; index < steps.size(); ++index)
    {
        std::cout << (index == 0 ? "" : ",") << steps[index];
    }
    std::cout << '\n';
}

void GpuCkksRuntime::print_rotation_step_trace() const
{
    if (!impl_->trace_rotation_steps)
    {
        return;
    }
    std::set<int> union_steps;
    std::size_t total_level_keys = 0;
    for (const auto &[q_count, steps] : impl_->observed_rotation_steps_by_q)
    {
        total_level_keys += steps.size();
        union_steps.insert(steps.begin(), steps.end());
        std::cout << "[GPU rotation plan] q=" << q_count
                  << " count=" << steps.size() << " steps=";
        std::size_t index = 0;
        for (const int step : steps)
        {
            std::cout << (index++ == 0 ? "" : ",") << step;
        }
        std::cout << '\n';
    }
    std::cout << "[GPU rotation plan] levels="
              << impl_->observed_rotation_steps_by_q.size()
              << " total_level_keys=" << total_level_keys
              << " union_keys=" << union_steps.size() << '\n';
}

void GpuCkksRuntime::initialize_inference_evaluation_keys()
{
    impl_->direct_rotation_steps.clear();
    impl_->direct_rotation_steps_by_q.clear();
    impl_->level_keyswitch_runtimes.clear();
    impl_->use_direct_rotation_keys = false;
    impl_->level_aware_application_keyswitch_enabled = false;
    std::vector<int> steps;
    for (std::size_t step = 1; step < slot_count(); step <<= 1)
    {
        steps.push_back(static_cast<int>(step));
    }
    initialize_evaluation_keys(steps);
}

void GpuCkksRuntime::initialize_all_evaluation_keys()
{
    initialize_inference_evaluation_keys();
}

void GpuCkksRuntime::initialize_bootstrap()
{
    if (impl_->bootstrap_data)
    {
        return;
    }
    initialize_evaluation_keys();
    impl_->gpu_parameters.configure_bootstrap_raise_target(
        impl_->context,
        static_cast<std::size_t>(impl_->config.q0_level + 1),
        impl_->config.bootstrap_q_count);
    impl_->bootstrap_level_keyswitch_runtimes.clear();
    if (::setenv("POSEIDON_BOOTSTRAP_EVALMOD_DYNAMIC_RESCALE", "1", 1) != 0)
    {
        throw std::runtime_error("failed to enable GPU bootstrap dynamic rescale");
    }

    const double evalmod_scale = std::exp2(
        static_cast<double>(impl_->config.evalmod_log_scale));
    const std::uint32_t log_message_ratio = static_cast<std::uint32_t>(
        std::llround(std::log2(impl_->config.message_ratio)));
    EvalModPoly eval_mod_poly(
        impl_->context,
        CosDiscrete,
        evalmod_scale,
        /*level_start=*/0,
        log_message_ratio,
        impl_->config.double_angle,
        impl_->config.boundary_k,
        /*arcsine_degree=*/0,
        impl_->config.evalmod_degree);
    const double c2s_scaling =
        eval_mod_poly.q_div() /
        (eval_mod_poly.k() * eval_mod_poly.sc_fac() * eval_mod_poly.q_diff());
    const double bootstrap_native_scale = std::exp2(
        static_cast<double>(impl_->config.bootstrap_output_log_scale));
    const double s2c_scaling =
        bootstrap_native_scale /
        (eval_mod_poly.scaling_factor() / eval_mod_poly.message_ratio());
    const double raised_scale = evalmod_scale;

    auto coeff_to_slot = make_dynamic_dft_group(
        impl_->context,
        impl_->encoder,
        encode,
        impl_->config.bootstrap_q_count - 1,
        c2s_scaling,
        raised_scale,
        evalmod_scale,
        1.0);
    const auto c2s_counts = dft_rescale_counts(coeff_to_slot);
    const std::size_t c2s_consumed = std::accumulate(
        c2s_counts.begin(), c2s_counts.end(), std::size_t{0});
    if (c2s_consumed >= impl_->config.bootstrap_q_count)
    {
        throw std::runtime_error("GPU bootstrap C2S consumes the modulus chain");
    }
    const std::size_t c2s_output_q_count =
        impl_->config.bootstrap_q_count - c2s_consumed;
    const auto evalmod_input_parms_id = impl_->context.crt_context()
        ->parms_id_map().at(static_cast<std::uint32_t>(c2s_output_q_count - 1));
    const auto evalmod_context = impl_->context.crt_context()->get_context_data(
        evalmod_input_parms_id);
    if (!evalmod_context)
    {
        throw std::runtime_error("GPU bootstrap EvalMod input level is absent");
    }
    eval_mod_poly.set_level_start(
        static_cast<std::uint32_t>(evalmod_context->level()));
    const double c2s_output_scale = planned_dft_output_scale(
        impl_->context, raised_scale, coeff_to_slot);
    auto evalmod_data = gpu::GpuUploader::upload_eval_mod_high_precision(
        eval_mod_poly,
        impl_->encoder,
        evalmod_input_parms_id,
        impl_->device_id,
        impl_->gpu_relin_keys.get(),
        parms_id_zero,
        /*logical_rescale_count=*/1,
        /*polynomial_override=*/nullptr,
        /*include_input_offset=*/true,
        std::numeric_limits<std::uint32_t>::max(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        /*fuse_leaf_terms_before_rescale=*/true,
        c2s_output_scale);

    const auto evalmod_output_context = impl_->context.crt_context()->get_context_data(
        evalmod_data.output_parms_id);
    if (!evalmod_output_context)
    {
        throw std::runtime_error("GPU bootstrap EvalMod output level is absent");
    }
    const double s2c_input_scale = evalmod_data.output_scale;
    const double s2c_minimum_scale = std::exp2(60.0);
    auto slot_to_coeff = make_dynamic_dft_group(
        impl_->context,
        impl_->encoder,
        decode,
        static_cast<std::uint32_t>(evalmod_output_context->level()),
        s2c_scaling,
        s2c_input_scale,
        s2c_minimum_scale,
        1.0);
    const double raw_s2c_output_scale = planned_dft_output_scale(
        impl_->context, s2c_input_scale, slot_to_coeff);
    slot_to_coeff = make_dynamic_dft_group(
        impl_->context,
        impl_->encoder,
        decode,
        static_cast<std::uint32_t>(evalmod_output_context->level()),
        s2c_scaling,
        s2c_input_scale,
        s2c_minimum_scale,
        evalmod_scale / raw_s2c_output_scale);

    const auto linear_transform_mode =
        gpu::gpu_linear_transform_mode_from_environment(
            gpu::GpuLinearTransformMode::ClassicBsgs);
    if (linear_transform_mode == gpu::GpuLinearTransformMode::SingleHoistBsgs)
    {
        throw std::invalid_argument(
            "GPU ResNet20 Bootstrap supports classic or double_hoist mode");
    }

    auto host_galois_keys = make_bootstrap_galois_keys(
        impl_->context, impl_->keygen, coeff_to_slot, slot_to_coeff);
    auto uploaded_galois =
        linear_transform_mode == gpu::GpuLinearTransformMode::DoubleHoistBsgs
            ? gpu::GpuUploader::upload_double_hoist_galois_keys(
                  host_galois_keys, impl_->device_id)
            : gpu::GpuUploader::upload_galois_keys(
                  host_galois_keys, impl_->device_id);
    impl_->gpu_bootstrap_galois_keys =
        std::make_unique<gpu::GpuGaloisKeysData>(std::move(uploaded_galois));
    auto key_q_counts = required_dft_key_q_counts(
        impl_->config.bootstrap_q_count, coeff_to_slot, true);
    const auto s2c_key_q_counts = required_dft_key_q_counts(
        evalmod_data.output_q_count, slot_to_coeff, false);
    key_q_counts.insert(
        key_q_counts.end(), s2c_key_q_counts.begin(), s2c_key_q_counts.end());
    std::sort(key_q_counts.begin(), key_q_counts.end());
    key_q_counts.erase(
        std::unique(key_q_counts.begin(), key_q_counts.end()),
        key_q_counts.end());
    gpu::GpuUploader::prepare_key_views_for_q_counts(
        *impl_->gpu_bootstrap_galois_keys, key_q_counts);

    const char *level_aware_bootstrap_env =
        std::getenv("POSEIDON_LEVEL_AWARE_BOOTSTRAP_P");
    const bool level_aware_bootstrap_enabled =
        level_aware_bootstrap_env == nullptr ||
        std::string(level_aware_bootstrap_env) != "0";
    if (level_aware_bootstrap_enabled)
    {
        BootstrapRotationRequirements rotation_requirements;
        const std::size_t c2s_final_q_count =
            add_dft_rotation_requirements(
                impl_->config.bootstrap_q_count,
                coeff_to_slot,
                /*conjugate_output=*/true,
                rotation_requirements);
        if (c2s_final_q_count != c2s_output_q_count)
        {
            throw std::logic_error(
                "GPU bootstrap C2S KeySwitch level plan is inconsistent");
        }
        const std::size_t s2c_final_q_count =
            add_dft_rotation_requirements(
                evalmod_data.output_q_count,
                slot_to_coeff,
                /*conjugate_output=*/false,
                rotation_requirements);
        // project_real performs one final conjugation after SlotToCoeff.
        rotation_requirements[s2c_final_q_count].insert(0);

        std::set<std::size_t> relinearization_q_counts(
            evalmod_data.required_relin_q_counts.begin(),
            evalmod_data.required_relin_q_counts.end());
        std::set<std::size_t> candidate_q_counts =
            relinearization_q_counts;
        for (const auto &entry : rotation_requirements)
        {
            candidate_q_counts.insert(entry.first);
        }

        const std::size_t global_p_count = impl_->parameters.p().size();
        for (const std::size_t q_count : candidate_q_counts)
        {
            // A separate context helps only after Q has fallen below the
            // global P width. At higher levels the configured global HYBRID
            // decomposition remains the intended path.
            if (q_count < 2 || q_count >= global_p_count)
            {
                continue;
            }

            std::vector<int> rotation_steps;
            const auto rotation_iter = rotation_requirements.find(q_count);
            if (rotation_iter != rotation_requirements.end())
            {
                rotation_steps.assign(
                    rotation_iter->second.begin(),
                    rotation_iter->second.end());
            }
            const bool needs_relinearization =
                relinearization_q_counts.count(q_count) != 0;
            if (rotation_steps.empty() && !needs_relinearization)
            {
                continue;
            }

            auto level_runtime =
                std::make_unique<LevelAwareKeySwitchRuntime>(
                    impl_->parameters,
                    impl_->keygen,
                    q_count,
                    q_count,
                    rotation_steps,
                    impl_->device_id,
                    needs_relinearization);
            const auto primary_parms_id = impl_->context.crt_context()
                ->parms_id_map().at(static_cast<std::uint32_t>(q_count - 1));
            const auto level_parms_id = level_runtime->context.crt_context()
                ->parms_id_map().at(static_cast<std::uint32_t>(q_count - 1));
            if (primary_parms_id != level_parms_id)
            {
                throw std::logic_error(
                    "level-aware bootstrap Q context is incompatible with ciphertext");
            }
            std::cout << "[GPU bootstrap] level-aware keyswitch q="
                      << q_count << " p=" << q_count
                      << " dnum=1 rotations=" << rotation_steps.size()
                      << " relin=" << (needs_relinearization ? 1 : 0)
                      << '\n';
            impl_->bootstrap_level_keyswitch_runtimes.emplace(
                q_count, std::move(level_runtime));
        }
    }
    else
    {
        std::cout << "[GPU bootstrap] level-aware P disabled; using global "
                     "KeySwitch basis\n";
    }

    Plaintext minus_i;
    impl_->encoder.encode(
        std::complex<double>(0.0, -1.0),
        impl_->context.crt_context()->parms_id_map().at(
            static_cast<std::uint32_t>(c2s_output_q_count - 1)),
        1.0,
        minus_i);
    Plaintext plus_i;
    impl_->encoder.encode(
        std::complex<double>(0.0, 1.0),
        evalmod_data.output_parms_id,
        1.0,
        plus_i);

    auto data = std::make_unique<gpu::GpuBootstrapData>();
    data->linear_transform_mode = linear_transform_mode;
    data->q0_parms_id = impl_->context.crt_context()->parms_id_map().at(
        impl_->config.q0_level);
    data->raised_parms_id = impl_->context.crt_context()->parms_id_map().at(
        impl_->config.bootstrap_q_count - 1);
    data->q0_over_message_ratio = std::exp2(std::round(std::log2(
        impl_->context.crt_context()->q0() /
        static_cast<double>(impl_->config.message_ratio))));
    data->raised_scale_override = raised_scale;
    data->slot_to_coeff_input_scale = s2c_input_scale;
    data->project_real = true;
    data->output_ratio = impl_->config.message_ratio;
    data->slot_to_coeff_output_scale = evalmod_scale;
    if (linear_transform_mode == gpu::GpuLinearTransformMode::DoubleHoistBsgs)
    {
        data->coeff_to_slot_matrix_qp =
            gpu::GpuUploader::upload_linear_matrix_group_qp(
                coeff_to_slot,
                impl_->context,
                impl_->device_id,
                std::max(coeff_to_slot.step(), std::uint32_t{1}));
        data->slot_to_coeff_matrix_qp =
            gpu::GpuUploader::upload_linear_matrix_group_qp(
                slot_to_coeff,
                impl_->context,
                impl_->device_id,
                std::max(slot_to_coeff.step(), std::uint32_t{1}));
    }
    else
    {
        data->coeff_to_slot_matrix =
            gpu::GpuUploader::upload_linear_matrix_group(
                coeff_to_slot, impl_->device_id);
        data->slot_to_coeff_matrix =
            gpu::GpuUploader::upload_linear_matrix_group(
                slot_to_coeff, impl_->device_id);
    }
    data->minus_i_plaintext = gpu::GpuUploader::upload_plaintext(
        minus_i, impl_->device_id);
    data->plus_i_plaintext = gpu::GpuUploader::upload_plaintext(
        plus_i, impl_->device_id);
    data->eval_mod = std::move(evalmod_data);
    impl_->bootstrap_data = std::move(data);
    impl_->bootstrap_workspace = std::make_unique<gpu::GpuBootstrapWorkspace>();
    std::cout << "[GPU bootstrap] linear_transform="
              << (linear_transform_mode ==
                          gpu::GpuLinearTransformMode::DoubleHoistBsgs
                      ? "double_hoist"
                      : "classic")
              << '\n';
}

bool GpuCkksRuntime::bootstrap_ready() const noexcept
{
    return static_cast<bool>(impl_->bootstrap_data);
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::rescale(
    const DeviceCiphertext &source,
    std::uint32_t physical_prime_count) const
{
    DeviceCiphertext result;
    impl_->gpu_evaluator.rescale_many(source, result, physical_prime_count);
    return result;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::bootstrap_modraise(
    const DeviceCiphertext &source) const
{
    const auto q0_parms_id = impl_->context.crt_context()->parms_id_map().at(
        impl_->config.q0_level);
    double q0_over_message_ratio = impl_->context.crt_context()->q0() /
        static_cast<double>(impl_->config.message_ratio);
    q0_over_message_ratio = std::exp2(
        std::round(std::log2(q0_over_message_ratio)));
    DeviceCiphertext prepared;
    impl_->gpu_evaluator.bootstrap_prepare_modraise_input(
        source,
        prepared,
        q0_parms_id,
        q0_over_message_ratio);
    DeviceCiphertext raised;
    impl_->gpu_evaluator.raise_modulus(prepared, raised);
    return raised;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::bootstrap(
    const DeviceCiphertext &source) const
{
    if (!impl_->bootstrap_data || !impl_->bootstrap_workspace ||
        !impl_->gpu_relin_keys || !impl_->gpu_bootstrap_galois_keys)
    {
        throw std::logic_error("bootstrap requires initialize_bootstrap");
    }
    impl_->gpu_parameters.configure_bootstrap_raise_target(
        impl_->context,
        static_cast<std::size_t>(impl_->config.q0_level + 1),
        impl_->config.bootstrap_q_count);
    gpu::GpuKeySwitchDispatch keyswitch_dispatch;
    keyswitch_dispatch.rotate =
        [this](const DeviceCiphertext &input,
               int step,
               DeviceCiphertext &output) {
            const auto found =
                impl_->bootstrap_level_keyswitch_runtimes.find(
                    input.meta.q_count);
            if (found == impl_->bootstrap_level_keyswitch_runtimes.end() ||
                found->second->gpu_galois_keys.empty())
            {
                return false;
            }
            found->second->gpu_evaluator->rotate(
                input, step, found->second->gpu_galois_keys, output);
            return true;
        };
    keyswitch_dispatch.conjugate =
        [this](const DeviceCiphertext &input, DeviceCiphertext &output) {
            const auto found =
                impl_->bootstrap_level_keyswitch_runtimes.find(
                    input.meta.q_count);
            if (found == impl_->bootstrap_level_keyswitch_runtimes.end() ||
                found->second->gpu_galois_keys.empty())
            {
                return false;
            }
            found->second->gpu_evaluator->conjugate(
                input, found->second->gpu_galois_keys, output);
            return true;
        };
    keyswitch_dispatch.relinearize =
        [this](const DeviceCiphertext &input, DeviceCiphertext &output) {
            const auto found =
                impl_->bootstrap_level_keyswitch_runtimes.find(
                    input.meta.q_count);
            if (found == impl_->bootstrap_level_keyswitch_runtimes.end() ||
                found->second->gpu_relin_keys.empty())
            {
                return false;
            }
            found->second->gpu_evaluator->relinearize(
                input, found->second->gpu_relin_keys, output);
            return true;
        };
    keyswitch_dispatch.relinearize_rescale_x2 =
        [this](const DeviceCiphertext &input, DeviceCiphertext &output) {
            const auto found =
                impl_->bootstrap_level_keyswitch_runtimes.find(
                    input.meta.q_count);
            if (found == impl_->bootstrap_level_keyswitch_runtimes.end() ||
                found->second->gpu_relin_keys.empty())
            {
                return false;
            }
            found->second->gpu_evaluator->relinearize_rescale_x2_hybrid(
                input, found->second->gpu_relin_keys, output);
            return true;
        };

    class ScopedKeySwitchDispatch
    {
    public:
        ScopedKeySwitchDispatch(
            const gpu::GpuEvaluator &evaluator,
            const gpu::GpuKeySwitchDispatch *dispatch)
            : evaluator_(evaluator)
        {
            evaluator_.set_keyswitch_dispatch(dispatch);
        }

        ~ScopedKeySwitchDispatch()
        {
            evaluator_.set_keyswitch_dispatch(nullptr);
        }

    private:
        const gpu::GpuEvaluator &evaluator_;
    };

    const char *level_aware_bootstrap_env =
        std::getenv("POSEIDON_LEVEL_AWARE_BOOTSTRAP_P");
    const bool use_level_aware_bootstrap =
        !impl_->bootstrap_level_keyswitch_runtimes.empty() &&
        (level_aware_bootstrap_env == nullptr ||
         std::string(level_aware_bootstrap_env) != "0");
    ScopedKeySwitchDispatch scoped_dispatch(
        impl_->gpu_evaluator,
        use_level_aware_bootstrap ? &keyswitch_dispatch : nullptr);

    DeviceCiphertext result;
    impl_->gpu_evaluator.bootstrap(
        source,
        *impl_->bootstrap_data,
        *impl_->gpu_relin_keys,
        *impl_->gpu_bootstrap_galois_keys,
        *impl_->bootstrap_workspace,
        result);

    // Normalize the refreshed ciphertext to the configured bootstrap output
    // scale. For ResNet20 this is the 2^40 application scale, while EvalMod
    // uses its independent 2^45 internal scale.
    const double target_scale = std::exp2(
        static_cast<double>(impl_->config.bootstrap_output_log_scale));
    DeviceCiphertext normalized;
    if (std::abs(result.meta.scale / target_scale - 1.0) <= 1.0e-6)
    {
        result.meta.scale = target_scale;
        normalized = std::move(result);
    }
    else
    {
        const auto context_data = impl_->context.crt_context()->get_context_data(
            result.meta.parms_id);
        if (!context_data || context_data->coeff_modulus().empty())
        {
            throw std::runtime_error("bootstrap output has no scale-correction level");
        }
        const double modulus = static_cast<double>(
            context_data->coeff_modulus().back().value());
        const double plain_scale = target_scale * modulus / result.meta.scale;
        const auto correction_key = scalar_plaintext_cache_key(
            "bootstrap_correction", result.meta.q_count, 1.0, plain_scale);
        auto correction_found = impl_->plaintext_cache.find(correction_key);
        if (correction_found == impl_->plaintext_cache.end())
        {
            Plaintext correction;
            impl_->encoder.encode(
                1.0, result.meta.parms_id, plain_scale, correction);
            auto uploaded = gpu::GpuUploader::upload_plaintext(
                correction, impl_->device_id);
            correction_found = impl_->plaintext_cache.emplace(
                correction_key, std::move(uploaded)).first;
        }
        DeviceCiphertext product;
        impl_->gpu_evaluator.multiply_plain(
            result, correction_found->second, product);
        normalized = rescale(product, 1);
        normalized.meta.scale = target_scale;
    }
    DeviceCiphertext application_level;
    if (normalized.meta.q_count > impl_->config.application_q_count())
    {
        application_level = drop_to_q_count(
            normalized, impl_->config.application_q_count());
    }
    else if (normalized.meta.q_count < impl_->config.application_q_count())
    {
        const auto q0_parms_id = impl_->context.crt_context()->parms_id_map().at(
            impl_->config.q0_level);
        DeviceCiphertext prepared;
        impl_->gpu_evaluator.bootstrap_prepare_modraise_input(
            normalized,
            prepared,
            q0_parms_id,
            target_scale);
        impl_->gpu_parameters.configure_bootstrap_raise_target(
            impl_->context,
            static_cast<std::size_t>(impl_->config.q0_level + 1),
            impl_->config.application_q_count());
        DeviceCiphertext raised;
        impl_->gpu_evaluator.raise_modulus(
            prepared, impl_->application_parms_id, raised);
        raised.meta.scale = target_scale;
        application_level = std::move(raised);
    }
    else
    {
        application_level = std::move(normalized);
    }

    if (std::abs(impl_->application_scale / target_scale - 1.0) <= 1.0e-12)
    {
        application_level.meta.scale = impl_->application_scale;
        return application_level;
    }
    const double promotion_scale = impl_->application_scale / target_scale;
    // The matrix normalization is calibrated at message_ratio=32. Ratios
    // above or below it change the raw output magnitude proportionally, so
    // fold the reciprocal correction into this exact scale-promotion plain.
    const double message_ratio_correction =
        32.0 / static_cast<double>(impl_->config.message_ratio);
    const auto promotion_key = scalar_plaintext_cache_key(
        "bootstrap_promotion", application_level.meta.q_count,
        message_ratio_correction, promotion_scale);
    auto promotion_found = impl_->plaintext_cache.find(promotion_key);
    if (promotion_found == impl_->plaintext_cache.end())
    {
        Plaintext promotion_plaintext;
        impl_->encoder.encode(
            message_ratio_correction,
            application_level.meta.parms_id,
            promotion_scale,
            promotion_plaintext);
        auto uploaded = gpu::GpuUploader::upload_plaintext(
            promotion_plaintext, impl_->device_id);
        promotion_found = impl_->plaintext_cache.emplace(
            promotion_key, std::move(uploaded)).first;
    }
    DeviceCiphertext promoted;
    impl_->gpu_evaluator.multiply_plain(
        application_level, promotion_found->second, promoted);
    promoted.meta.scale = impl_->application_scale;
    return promoted;
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::square_relinearize_rescale(
    const DeviceCiphertext &source) const
{
    return multiply_relinearize_rescale(source, source);
}

GpuCkksRuntime::DeviceCiphertext GpuCkksRuntime::multiply_relinearize_rescale(
    const DeviceCiphertext &left,
    const DeviceCiphertext &right) const
{
    if (!impl_->gpu_relin_keys)
    {
        throw std::logic_error(
            "square_relinearize_rescale requires initialize_evaluation_keys");
    }

    const std::size_t target_q_count =
        std::min(left.meta.q_count, right.meta.q_count);
    std::unique_ptr<DeviceCiphertext> adjusted_left;
    std::unique_ptr<DeviceCiphertext> adjusted_right;
    const DeviceCiphertext *left_view = &left;
    const DeviceCiphertext *right_view = &right;
    if (left.meta.q_count != target_q_count)
    {
        adjusted_left = std::make_unique<DeviceCiphertext>(
            drop_to_q_count(left, target_q_count));
        left_view = adjusted_left.get();
    }
    if (right.meta.q_count != target_q_count)
    {
        adjusted_right = std::make_unique<DeviceCiphertext>(
            drop_to_q_count(right, target_q_count));
        right_view = adjusted_right.get();
    }

    DeviceCiphertext multiplied;
    impl_->gpu_evaluator.multiply(*left_view, *right_view, multiplied);
    DeviceCiphertext relinearized;
    auto level_runtime = impl_->level_keyswitch_runtimes.find(
        target_q_count);
    if (level_runtime == impl_->level_keyswitch_runtimes.end() &&
        impl_->level_aware_application_keyswitch_enabled &&
        impl_->application_keyswitch_p_mode ==
            ApplicationKeySwitchPMode::fixed_dnum)
    {
        const auto shape = application_keyswitch_shape(target_q_count);
        if (shape.level_aware)
        {
            std::shared_ptr<LevelAwareKeySwitchRuntime> runtime;
            for (const auto &[unused_q_count, candidate] :
                 impl_->level_keyswitch_runtimes)
            {
                (void)unused_q_count;
                if (candidate->p_count == shape.p_count &&
                    candidate->q_count >= target_q_count)
                {
                    runtime = candidate;
                    break;
                }
            }
            if (!runtime)
            {
                const std::size_t context_q_count = std::min(
                    impl_->parameters.q().size(),
                    shape.p_count *
                        static_cast<std::size_t>(impl_->config.dnum));
                if (context_q_count < target_q_count)
                {
                    throw std::logic_error(
                        "fixed-dnum relinearization context is too small");
                }
                runtime = std::make_shared<LevelAwareKeySwitchRuntime>(
                    impl_->parameters,
                    impl_->keygen,
                    context_q_count,
                    shape.p_count,
                    std::vector<int>{},
                    impl_->device_id,
                    /*create_relinearization_key=*/true);
            }
            const auto primary_parms_id = impl_->context.crt_context()
                ->parms_id_map().at(
                    static_cast<std::uint32_t>(target_q_count - 1));
            const auto level_parms_id = runtime->context.crt_context()
                ->parms_id_map().at(
                    static_cast<std::uint32_t>(target_q_count - 1));
            if (primary_parms_id != level_parms_id)
            {
                throw std::logic_error(
                    "fixed-dnum relinearization Q context is incompatible with ciphertext");
            }
            std::cout
                << "[GPU ResNet20] application relinearize keyswitch level q="
                << target_q_count << " p=" << runtime->p_count
                << " target_dnum=" << impl_->config.dnum
                << " effective_dnum=" << shape.effective_dnum
                << " shared_context_q=" << runtime->q_count << '\n';
            level_runtime = impl_->level_keyswitch_runtimes.emplace(
                target_q_count, std::move(runtime)).first;
        }
    }
    if (level_runtime != impl_->level_keyswitch_runtimes.end())
    {
        level_runtime->second->gpu_evaluator->relinearize(
            multiplied,
            level_runtime->second->gpu_relin_keys,
            relinearized);
    }
    else
    {
        impl_->gpu_evaluator.relinearize(
            multiplied, *impl_->gpu_relin_keys, relinearized);
    }
    auto reduced = rescale(
        relinearized,
        impl_->config.cipher_product_rescale_primes);

    const double target_scale = left_view->meta.scale;
    const double scale_ratio = target_scale / reduced.meta.scale;
    if (std::abs(scale_ratio - 1.0) <= 1.0e-6)
    {
        reduced.meta.scale = target_scale;
        return reduced;
    }
    const auto context_data =
        impl_->context.crt_context()->get_context_data(reduced.meta.parms_id);
    if (!context_data || context_data->coeff_modulus().empty())
    {
        throw std::runtime_error("GPU scale correction has no remaining modulus");
    }
    const double correction_modulus = static_cast<double>(
        context_data->coeff_modulus().back().value());
    const double correction_plain_scale =
        target_scale * correction_modulus / reduced.meta.scale;
    if (!(correction_plain_scale > 0.0) ||
        !std::isfinite(correction_plain_scale))
    {
        throw std::runtime_error("GPU post-multiply scale correction is invalid");
    }
    const auto correction_key = scalar_plaintext_cache_key(
        "multiply_scale_correction", reduced.meta.q_count, 1.0,
        correction_plain_scale);
    auto correction_found = impl_->plaintext_cache.find(correction_key);
    if (correction_found == impl_->plaintext_cache.end())
    {
        Plaintext correction_plaintext;
        impl_->encoder.encode(
            1.0,
            reduced.meta.parms_id,
            correction_plain_scale,
            correction_plaintext);
        auto uploaded = gpu::GpuUploader::upload_plaintext(
            correction_plaintext, impl_->device_id);
        correction_found = impl_->plaintext_cache.emplace(
            correction_key, std::move(uploaded)).first;
    }
    DeviceCiphertext normalized;
    impl_->gpu_evaluator.multiply_plain(
        reduced, correction_found->second, normalized);
    normalized = rescale(normalized, 1);
    normalized.meta.scale = target_scale;
    return normalized;
}

int GpuCkksRuntime::device_id() const noexcept
{
    return impl_->device_id;
}

std::size_t GpuCkksRuntime::slot_count() const noexcept
{
    return impl_->config.slot_count();
}

}  // namespace poseidon::benchmark::resnet20_gpu::core
