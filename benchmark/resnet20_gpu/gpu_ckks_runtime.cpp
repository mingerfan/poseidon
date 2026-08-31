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

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
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

class RmmPoolScope
{
public:
    explicit RmmPoolScope(int device_id)
        : device_id_(device_id), pool_(&upstream_, 1 << 20, std::nullopt)
    {
        check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
        previous_ = rmm::mr::get_current_device_resource();
        rmm::mr::set_current_device_resource(&pool_);
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
              gpu_parameters.get_level_by_q_count(config.application_q_count()).parms_id)
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
    bool use_direct_rotation_keys = false;
    bool full_device_cache = false;
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
    const auto context_data =
        impl_->context.crt_context()->get_context_data(source.meta.parms_id);
    if (!context_data || context_data->coeff_modulus().empty())
    {
        throw std::invalid_argument(
            "last_modulus_value received an unknown or empty modulus level");
    }
    return static_cast<double>(context_data->coeff_modulus().back().value());
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
    const gpu::GpuGaloisKeysData *keys = impl_->gpu_galois_keys
        ? impl_->gpu_galois_keys.get()
        : impl_->gpu_bootstrap_galois_keys.get();
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
    if (impl_->use_direct_rotation_keys)
    {
        if (impl_->direct_rotation_steps.count(direct_step) == 0)
        {
            throw std::logic_error(
                "direct rotation key is missing for the requested step");
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

void GpuCkksRuntime::initialize_direct_rotation_keys(
    const std::vector<int> &rotation_steps)
{
    if (rotation_steps.empty())
    {
        throw std::logic_error(
            "direct rotation step list must not be empty");
    }

    impl_->direct_rotation_steps.clear();
    const auto slots = static_cast<long long>(slot_count());
    for (const int step : rotation_steps)
    {
        long long normalized = static_cast<long long>(step) % slots;
        if (normalized < 0)
        {
            normalized += slots;
        }
        if (normalized == 0)
        {
            throw std::invalid_argument(
                "direct rotation step list contains a zero rotation");
        }
        impl_->direct_rotation_steps.insert(static_cast<int>(normalized));
    }

    const std::vector<int> steps(
        impl_->direct_rotation_steps.begin(),
        impl_->direct_rotation_steps.end());
    std::cout << "[GPU ResNet20] generating direct rotation keys before "
                 "inference count=" << steps.size() << '\n';
    initialize_evaluation_keys(steps);
    impl_->use_direct_rotation_keys = true;
    std::cout << "[GPU ResNet20] direct rotation keys ready steps=";
    for (std::size_t index = 0; index < steps.size(); ++index)
    {
        std::cout << (index == 0 ? "" : ",") << steps[index];
    }
    std::cout << '\n';
}

void GpuCkksRuntime::initialize_inference_evaluation_keys()
{
    impl_->direct_rotation_steps.clear();
    impl_->use_direct_rotation_keys = false;
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

    auto host_galois_keys = make_bootstrap_galois_keys(
        impl_->context, impl_->keygen, coeff_to_slot, slot_to_coeff);
    auto uploaded_galois = gpu::GpuUploader::upload_galois_keys(
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
    data->linear_transform_mode = gpu::GpuLinearTransformMode::ClassicBsgs;
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
    data->coeff_to_slot_matrix =
        gpu::GpuUploader::upload_linear_matrix_group(
            coeff_to_slot, impl_->device_id);
    data->slot_to_coeff_matrix =
        gpu::GpuUploader::upload_linear_matrix_group(
            slot_to_coeff, impl_->device_id);
    data->minus_i_plaintext = gpu::GpuUploader::upload_plaintext(
        minus_i, impl_->device_id);
    data->plus_i_plaintext = gpu::GpuUploader::upload_plaintext(
        plus_i, impl_->device_id);
    data->eval_mod = std::move(evalmod_data);
    impl_->bootstrap_data = std::move(data);
    impl_->bootstrap_workspace = std::make_unique<gpu::GpuBootstrapWorkspace>();
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
    impl_->gpu_evaluator.relinearize(
        multiplied, *impl_->gpu_relin_keys, relinearized);
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
