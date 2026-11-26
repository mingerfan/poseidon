#include "native_boot.h"
#include "poseidon/runtime_api/s2c_bootstrap.h"
#include "poseidon/s2c_gpu/gpu_bootstrap_profile.h"
#include "poseidon/basics/randomtostd.h"
#include "poseidon/basics/util/ntt.h"
#include "bounded_keys.h"
#include "q50_p25.h"
#include "secret_audit.h"
#include "support/secure_sparse_secret.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace poseidon::runtime_api
{
namespace
{
size_t profile_index(std::string_view id)
{
    const auto it = std::find(s2c_bootstrap_profiles.begin(), s2c_bootstrap_profiles.end(), id);
    if (it == s2c_bootstrap_profiles.end())
        throw std::invalid_argument("unknown S2C-first Boot profile: " + std::string(id));
    return static_cast<size_t>(it - s2c_bootstrap_profiles.begin());
}
ParametersLiteral parameters(size_t index)
{
    ParametersLiteral literal(CKKS, 16, 15, 40, index >= 2 ? 32768 : 192, 1, Modulus(0), {}, {},
                              sec_level_type::none);
    if (index == 1)
    {
        literal.set_log_modulus({32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 28, 28, 31, 31, 32,
                                 32, 30, 31, 32, 31, 32, 32, 31, 31, 31, 32, 32, 31, 32, 32, 32, 30},
                                std::vector<std::uint32_t>(9, 32));
    }
    else if (index == 0)
    {
        literal.set_modulus(std::vector<Modulus>(q50.begin(), q50.end()),
                            std::vector<Modulus>(p25.begin(), p25.end()));
    }
    else
    {
        literal.set_modulus(std::vector<Modulus>(agent22::q_primes.begin(), agent22::q_primes.end()),
                            std::vector<Modulus>(agent22::p_primes.begin(), agent22::p_primes.end()));
    }
    return literal;
}
void check_context(const PoseidonContext &context, size_t index)
{
    const auto expected = parameters(index);
    const auto &actual = *context.parameters_literal();
    if (actual.scheme() != CKKS || actual.log_n() != 16 || actual.log_slots() != 15 ||
        actual.log_scale() != 40 || actual.q0_level() != 1 || actual.q() != expected.q() ||
        actual.p() != expected.p() || actual.hamming_weight() != expected.hamming_weight())
        throw std::invalid_argument("S2C-first profile does not match the encryption context");
    if (!context.random_generator() || !context.random_generator()->use_random_seed())
        throw std::invalid_argument("S2C-first key preparation requires operating-system entropy");
}
} // namespace
PoseidonContext make_s2c_bootstrap_context(std::string_view id)
{
    return PoseidonContext(parameters(profile_index(id)));
}
SecretKey make_s2c_bootstrap_secret(const PoseidonContext &context, std::string_view id)
{
    check_context(context, profile_index(id));
    const auto crt = context.crt_context();
    const auto key_id = crt->key_parms_id();
    const auto &key_context = *crt->key_context_data();
    const auto &moduli = key_context.coeff_modulus();
    const auto n = key_context.parms().degree();
    const auto h = context.parameters_literal()->hamming_weight();
    RandomToStandardAdapter engine(context.random_generator()->create());
    auto coefficients = benchmark::resnet20_gpu::core::sample_balanced_sparse_secret(n, h, engine);
    struct Wipe
    {
        std::vector<std::int8_t> &data;
        ~Wipe()
        {
            util::poseidon_memzero(data.data(), data.size());
        }
    } wipe{coefficients};
    SecretKey key;
    key.data().resize(context, key_id, n * moduli.size());
    auto *data = key.data().data();
    for (size_t j = 0; j < moduli.size(); ++j)
        for (size_t i = 0; i < n; ++i)
            data[j * n + i] = coefficients[i] < 0 ? moduli[j].value() - 1 : coefficients[i];
    agent22::audit_secret("main", data, n, moduli, h, true);
    util::ntt_negacyclic_harvey(util::RNSIter(data, n), moduli.size(), crt->small_ntt_tables());
    key.parms_id() = key_id;
    return key;
}
namespace s2c
{
struct NativeBoot::Impl
{
    int device;
    s2c_gpu::GpuParameterData params;
    s2c_gpu::GpuBootstrapProfile profile;
    s2c_gpu::GpuEvaluator evaluator;
    s2c_gpu::GpuBootstrapWorkspace workspace;
    s2c_gpu::GpuCiphertextData input, output;
    cudaEvent_t latest{};
    bool pending = false;
    std::mutex mutex;

    Impl(const PoseidonContext &ctx, KeyGenerator &gen, int dev, size_t index)
        : device(dev), params(ctx, dev), evaluator(params)
    {
        s2c_gpu::GpuResNet20ManualOptions options;
        if (index == 1)
        {
            options.sine_degree = 30;
            options.double_angle = 3;
            options.k = 16;
            options.full_coefficients = true;
            options.sparse_secret_hamming_weight = 32;
            options.target_output_q = 14;
            options.complex_native_output = true;
        }
        else if (index >= 2)
        {
            options.sine_degree = 22;
            options.double_angle = 3;
            options.k = 12;
            options.log_split = 2;
            options.full_coefficients = true;
            options.sparse_secret_hamming_weight = 32;
            options.level_special_modulus =
                s2c_gpu::GpuResNet20ManualOptions::LevelSpecialModulus::Secure;
            options.dnum1_max_q = 22;
            options.secure_modulus_bits = agent22::pq_budget_bits;
            options.target_output_q = 24;
        }
        profile = s2c_gpu::GpuBootstrapProfileBuilder::build_resnet20_manual(ctx, gen, dev, options);
        if (index >= 2)
            agent22::audit_bounded_keys(profile, *ctx.parameters_literal());
        for (auto &level : profile.level_parameters)
            params.adopt_level(std::move(level));
        workspace.coeff_to_slot_double_hoist.baby_tile_size = 8;
        input = s2c_gpu::GpuCiphertextData::allocate_single_device(65536, 6, 2, dev);
        input.meta.parms_id = ctx.crt_context()->parms_id_map().at(5);
        input.meta.scale = std::exp2(40.0);
        input.meta.is_ntt_form = true;
        gpu::gpu_check_cuda(cudaStreamSynchronize(s2c_gpu::gpu_execution_stream()),
                            "finish S2C-first setup");
        gpu::gpu_check_cuda(cudaEventCreateWithFlags(&latest, cudaEventDisableTiming),
                            "S2C-first workspace event");
    }
    ~Impl()
    {
        (void)cudaSetDevice(device);
        if (pending)
            (void)cudaEventSynchronize(latest);
        (void)cudaEventDestroy(latest);
    }
};

// Copy the canonical limb/coefficient rectangles, including sharded runtime inputs.
template <class Source, class Destination>
void copy_to_packed(const Source &source, Destination &dest, cudaStream_t stream)
{
    const auto view = source.make_const_view();
    const auto packed = dest.make_view();
    for (size_t j = 0; j < view.polys.size(); ++j)
    {
        const auto &target = packed.polys.at(j).shards.at(0);
        for (const auto &s : view.polys[j].shards)
        {
            if (s.device_id != target.device_id || s.limb_begin > target.limb_count ||
                s.limb_count > target.limb_count - s.limb_begin || s.coeff_begin > target.coeff_count ||
                s.coeff_count > target.coeff_count - s.coeff_begin)
                throw std::invalid_argument("incompatible native Boot ciphertext shard");
            gpu::gpu_check_cuda(
                cudaMemcpy2DAsync(target.ptr + s.limb_begin * target.coeff_count + s.coeff_begin,
                                  target.coeff_count * sizeof(*target.ptr), s.ptr,
                                  s.coeff_count * sizeof(*s.ptr), s.coeff_count * sizeof(*s.ptr),
                                  s.limb_count, cudaMemcpyDeviceToDevice, stream),
                "S2C-first device copy");
        }
    }
}

NativeBoot::NativeBoot(const PoseidonContext &ctx, KeyGenerator &gen, int device, std::string_view id)
{
    const auto index = profile_index(id);
    check_context(ctx, index);
    if (gen.secret_key().parms_id() != ctx.crt_context()->key_parms_id() ||
        gen.actual_secret_weight() != ctx.parameters_literal()->hamming_weight() ||
        (index >= 2 && gen.actual_positive_count() * 2 != gen.actual_secret_weight()))
        throw std::invalid_argument("S2C-first main secret does not match the profile");
    impl_ = std::make_unique<Impl>(ctx, gen, device, index);
}
NativeBoot::~NativeBoot() = default;
int NativeBoot::output_scale_log2() const
{
    const double scale = impl_->profile.bootstrap_data.slot_to_coeff_output_scale;
    int exponent = 0;
    if (std::frexp(scale, &exponent) != 0.5 || exponent < 1)
        throw std::logic_error("S2C-first output scale must be an exact power of two");
    return exponent - 1;
}
int NativeBoot::output_level() const
{
    return impl_->profile.output_level;
}

void NativeBoot::execute(const gpu::GpuCiphertextData &source, bool real_only,
                         gpu::GpuCiphertextData &dest)
{
    auto &s = *impl_;
    const auto &m = source.meta;
    if (m.parms_id != s.input.meta.parms_id || m.degree != 65536 || m.q_count != 6 || m.p_count != 0 ||
        m.scale != s.input.meta.scale || m.component_count != 2 || source.size() != 2 || !m.is_ntt_form)
        throw std::invalid_argument(
            "native S2C-first Boot requires a two-component NTT Q6 ciphertext at 2^40");
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto stream = gpu::gpu_execution_stream();
    s2c_gpu::ScopedGpuExecutionStream scoped(stream);
    if (s.pending)
        gpu::gpu_check_cuda(cudaStreamWaitEvent(stream, s.latest, 0), "reuse native Boot workspace");
    try
    {
        s.input.meta.correction_factor = m.correction_factor;
        copy_to_packed(source, s.input, stream);
        s.evaluator.bootstrap(s.input, s.profile.bootstrap_data, *s.profile.relin_keys,
                              *s.profile.galois_keys, s.workspace, s.output, real_only);
        const auto &out = s.output.meta;
        dest = gpu::GpuCiphertextData::allocate_single_device(out.degree, out.q_count,
                                                              out.component_count, s.device);
        dest.meta.parms_id = out.parms_id;
        dest.meta.scale = out.scale;
        dest.meta.correction_factor = out.correction_factor;
        dest.meta.is_ntt_form = out.is_ntt_form;
        copy_to_packed(s.output, dest, stream);
        gpu::gpu_check_cuda(cudaEventRecord(s.latest, stream),
                            "record native Boot workspace completion");
        s.pending = true;
    }
    catch (...)
    {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
}
} // namespace s2c
} // namespace poseidon::runtime_api
