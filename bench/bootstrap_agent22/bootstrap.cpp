#include "default_optimizations.h"
#include "gpu_activity_timing.h"
#include "bounded_keys.h"
#include "secret_audit.h"
#include "support/secure_sparse_secret.h"

#include "poseidon/basics/randomtostd.h"
#include "poseidon/basics/util/ntt.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_bootstrap_profile.h"
#include "poseidon/gpu/gpu_opt_flags.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
using namespace poseidon;
using namespace poseidon::gpu;
using namespace poseidon::agent22;
using poseidon::benchmark::s2c_first::GpuActivityTiming;
using Clock = std::chrono::steady_clock;
constexpr std::size_t slots = 32768;
constexpr std::size_t main_weight = 32768;
constexpr std::size_t input_q = 6;
constexpr std::size_t output_q = 24;
constexpr double scale = double(std::uint64_t{1} << 40);
constexpr double tolerance = 1e-3;
constexpr char usage[] =
    "usage: agent22_bootstrap [repetitions: 1..100] "
    "[9 | 25 | dnum4 | s2c-first-degree22-da3-q43-dnum4 | "
    "s2c-first-degree22-da3-q43-dnum4-real-only] [--compare]";

struct Profile
{
    std::string_view name;
    int dnum1_max_q;
    bool bounded_keys;
    bool real_only;
};

constexpr Profile dual{"s2c-first-degree22-da3-q43-dnum4", int(p_primes.size()), true, false};
constexpr Profile real{"s2c-first-degree22-da3-q43-dnum4-real-only", int(p_primes.size()), true, true};

Profile select_profile(std::string_view name)
{
    if (name == dual.name || name == "dnum4" ||
        name == "s2c-first-degree22-da3-full-coefficients-q43-layered-dnum4")
        return dual;
    if (name == real.name)
        return real;
    if (name == "9" || name == "25")
        return {"s2c-first-degree22-da3-full-coefficients-q43p22", name == "9" ? 9 : 25, false, false};
    throw std::invalid_argument(usage);
}

int integer_argument(std::string_view text, int low, int high)
{
    int value = 0;
    const auto end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end || value < low || value > high)
        throw std::invalid_argument(usage);
    return value;
}

struct Event
{
    cudaEvent_t handle{};
    Event() { gpu_check_cuda(cudaEventCreate(&handle), "create timing event"); }
    ~Event() { cudaEventDestroy(handle); }
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
};

struct Pool
{
    rmm::mr::cuda_memory_resource upstream;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool{&upstream, 1 << 20};
    rmm::mr::device_memory_resource *previous = rmm::mr::get_current_device_resource();
    Pool() { rmm::mr::set_current_device_resource(&pool); }
    ~Pool() { rmm::mr::set_current_device_resource(previous); }
};

SecretKey make_main_secret(const PoseidonContext &context)
{
    const auto crt = context.crt_context();
    const auto id = crt->key_parms_id();
    const auto data = crt->key_context_data();
    const auto &moduli = data->coeff_modulus();
    const auto n = data->parms().degree();
    RandomToStandardAdapter engine(context.random_generator()->create());
    using poseidon::benchmark::resnet20_gpu::core::sample_balanced_sparse_secret;
    auto coefficients = sample_balanced_sparse_secret(n, main_weight, engine);
    struct Wipe
    {
        std::vector<std::int8_t> &data;
        ~Wipe() { util::poseidon_memzero(data.data(), data.size()); }
    } wipe{coefficients};

    SecretKey key;
    key.data().resize(context, id, n * moduli.size());
    auto *limbs = key.data().data();
    for (std::size_t j = 0; j < moduli.size(); ++j)
        for (std::size_t i = 0; i < n; ++i)
            limbs[j * n + i] = coefficients[i] < 0 ? moduli[j].value() - 1 : coefficients[i];
    audit_secret("main", limbs, n, moduli, main_weight, true);
    util::ntt_negacyclic_harvey(util::RNSIter(limbs, n), moduli.size(), crt->small_ntt_tables());
    key.parms_id() = id;
    return key;
}

std::vector<std::complex<double>> verify_output(
    const GpuCiphertextData &output, const PoseidonContext &context, const CKKSEncoder &encoder,
    Decryptor &decryptor, const std::vector<double> &reference, const char *label)
{
    if (output.meta.q_count != output_q || !std::isfinite(output.meta.scale) ||
        std::abs(std::log2(output.meta.scale) - 40.0) > 1e-7)
        throw std::runtime_error("Agent22 bootstrap output must be Q24 / 2^40");
    Ciphertext ciphertext;
    GpuUploader::download_ciphertext(output, ciphertext, context);
    Plaintext plaintext;
    decryptor.decrypt(ciphertext, plaintext);
    std::vector<std::complex<double>> decoded;
    encoder.decode(plaintext, decoded);
    if (decoded.size() != reference.size())
        throw std::runtime_error("unexpected decoded slot count");

    double max_abs = 0;
    double sum_squared = 0;
    for (std::size_t i = 0; i < decoded.size(); ++i)
    {
        const double error = std::abs(decoded[i] - reference[i]);
        if (!std::isfinite(error))
            throw std::runtime_error("non-finite error in a decoded slot");
        max_abs = std::max(max_abs, error);
        sum_squared += error * error;
    }
    std::cout << "PRECISION sample=" << label << " slots=" << decoded.size()
              << " output_q=" << output.meta.q_count << " log2_scale=" << std::log2(output.meta.scale)
              << " max_abs=" << max_abs << " rms=" << std::sqrt(sum_squared / decoded.size())
              << " max_abs_limit=" << tolerance << std::endl;
    if (max_abs > tolerance)
        throw std::runtime_error("Agent22 bootstrap exceeded the precision limit");
    return decoded;
}

void print_samples(const char *metric, std::vector<double> samples)
{
    std::cout << metric << " samples_ms=";
    for (std::size_t i = 0; i < samples.size(); ++i)
        std::cout << (i ? "," : "") << samples[i];
    std::sort(samples.begin(), samples.end());
    const auto middle = samples.size() / 2;
    const double median = (samples[middle] + samples[(samples.size() - 1) / 2]) / 2;
    std::cout << " median_ms=" << median << std::endl;
}

void configure_profile(Profile selected)
{
    // Agent selects its real-projection path through this process-wide flag.
    if (setenv("POSEIDON_OPT_BOOT_REAL_ONLY", selected.real_only ? "1" : "0", 1) != 0)
        throw std::runtime_error("cannot configure Agent EvalMod branches");
    poseidon::benchmark::s2c_first::apply_default_optimizations();
    reload_opt_flags();
}

std::vector<std::complex<double>> measure_profile(
    Profile selected, int repetitions, const cudaDeviceProp &device, Pool &memory,
    GpuEvaluator &evaluator, const GpuBootstrapProfile &profile, const GpuCiphertextData &input,
    const PoseidonContext &context, const CKKSEncoder &encoder, Decryptor &decryptor,
    const std::vector<double> &reference)
{
    std::cout << std::setprecision(9) << "PROFILE id=" << selected.name
              << " gpu=\"" << device.name << "\""
              << " N=" << 2 * slots << " degree=22 double_angle=3 K=12 Q=43 available_p=22"
              << " input_q=6 output_q=24 dnum1_max_q=" << selected.dnum1_max_q
              << " input_encryption=" << (selected.bounded_keys ? "symmetric_Q6" : "public_key")
              << " evalmod_branches=" << (selected.real_only ? 1 : 2)
              << " entropy=OS_random" << std::endl;

    const auto &plan = profile.bootstrap_data;
    GpuBootstrapWorkspace workspace;
    workspace.coeff_to_slot_double_hoist.baby_tile_size = 8;
    GpuCiphertextData output;
    const auto boot = [&] {
        evaluator.bootstrap(input, plan, *profile.relin_keys, *profile.galois_keys, workspace, output);
    };
    const auto stream = gpu_execution_stream();
    constexpr int warmups = 3;
    for (int i = 0; i < warmups; ++i)
        boot();
    gpu_check_cuda(cudaStreamSynchronize(stream), "finish warmup");
    const bool has_real = !workspace.eval_mod_real.empty();
    const bool has_imag = !workspace.eval_mod_imag.empty();
    if (!has_real || has_imag == selected.real_only)
        throw std::runtime_error("EvalMod workspace outputs disagree with the selected profile");
    std::cout << "EVALMOD_OUTPUTS real=" << has_real << " imag=" << has_imag << std::endl;
    verify_output(output, context, encoder, decryptor, reference, "warmup");

    std::vector<double> events, wall;
    events.reserve(repetitions);
    wall.reserve(repetitions);
    Event start, end;
    for (int i = 0; i < repetitions; ++i)
    {
        const auto wall_start = Clock::now();
        gpu_check_cuda(cudaEventRecord(start.handle, stream), "record bootstrap start");
        boot();
        gpu_check_cuda(cudaEventRecord(end.handle, stream), "record bootstrap end");
        gpu_check_cuda(cudaEventSynchronize(end.handle), "finish bootstrap");
        wall.push_back(std::chrono::duration<double, std::milli>(Clock::now() - wall_start).count());
        float elapsed = 0;
        gpu_check_cuda(cudaEventElapsedTime(&elapsed, start.handle, end.handle), "read bootstrap events");
        events.push_back(elapsed);
    }
    verify_output(output, context, encoder, decryptor, reference, "timed");
    print_samples("CUDA_EVENT", std::move(events));
    print_samples("ONLINE_WALL", std::move(wall));

    GpuActivityTiming activity;
    activity.begin_continuous();
    activity.stage("Bootstrap", true);
    for (int i = 0; i < repetitions; ++i)
        boot();
    gpu_check_cuda(cudaStreamSynchronize(stream), "finish CUPTI bootstrap replay");
    activity.stage("Bootstrap", false);
    const auto result = activity.finish();
    result.print();
    std::cout << "CUPTI calls=" << repetitions << " activity_ms_per_boot="
              << result.gpu_total_ns / (1e6 * repetitions)
              << " pool_reserved_MiB=" << memory.pool.pool_size() / double(1 << 20) << std::endl;
    return verify_output(output, context, encoder, decryptor, reference, "cupti");
}

void run_benchmark(int repetitions, Profile selected, bool compare)
{
    gpu_check_cuda(cudaSetDevice(0), "select benchmark GPU");
    cudaDeviceProp device;
    gpu_check_cuda(cudaGetDeviceProperties(&device, 0), "read GPU properties");
    if (setenv("POSEIDON_GPU_SSE_LOW_P", "2", 1) != 0)
        throw std::runtime_error("cannot configure sparse secret switching");
    configure_profile(selected);
    const bool bounded_keys = selected.bounded_keys;
    std::cout << std::setprecision(9);
    Pool pool;
    const auto setup_start = Clock::now();
    ParametersLiteral literal(CKKS, 16, 15, 40, main_weight, 1, Modulus(0), {}, {}, sec_level_type::none);
    const std::vector<Modulus> q(q_primes.begin(), q_primes.end());
    const std::vector<Modulus> p(p_primes.begin(), p_primes.end());
    literal.set_modulus(q, p);
    PoseidonContext context(literal);
    const auto secret = make_main_secret(context);
    KeyGenerator generator(context, secret);
    Decryptor decryptor(context, secret);
    CKKSEncoder encoder(context);

    GpuParameterData parameters(context, 0);
    GpuResNet20ManualOptions options;
    options.sine_degree = 22;
    options.double_angle = 3;
    options.k = 12;
    options.log_split = 2;
    options.full_coefficients = true;
    options.sparse_secret_hamming_weight = 32;
    using LevelSpecialModulus = GpuResNet20ManualOptions::LevelSpecialModulus;
    options.level_special_modulus = bounded_keys ? LevelSpecialModulus::Secure : LevelSpecialModulus::Performance;
    options.secure_modulus_bits = pq_budget_bits;
    options.dnum1_max_q = selected.dnum1_max_q;
    options.target_output_q = output_q;
    auto profile = GpuBootstrapProfileBuilder::build_resnet20_manual(context, generator, 0, options);
    const auto &plan = profile.bootstrap_data;
    if (!plan.slot_to_coeff_first || plan.input_q_count != input_q || !plan.project_real ||
        plan.eval_mod.output_q_count != output_q || !profile.relin_keys || !profile.galois_keys)
        throw std::runtime_error("unexpected Agent22 bootstrap plan");
    if (bounded_keys)
        audit_bounded_keys(profile, literal);
    std::sort(profile.level_parameters.begin(), profile.level_parameters.end(),
        [](const auto &a, const auto &b) { return a.q_count < b.q_count; });
    for (auto &level : profile.level_parameters)
    {
        // The ciphertext level is Q-only. The associated P basis lives
        // in that level's parameter shard for use by key switching.
        const auto limb_count = level.shards.at(0).limb_count;
        if (limb_count <= level.q_count)
            throw std::runtime_error("Agent22 key-switch level is missing its special basis");
        const auto p_count = limb_count - level.q_count;
        std::cout << "KEYSWITCH Q=" << level.q_count << " P=" << p_count
                  << " dnum=" << (level.q_count + p_count - 1) / p_count << '\n';
        parameters.adopt_level(std::move(level));
    }
    GpuEvaluator evaluator(parameters);

    std::vector<double> reference(slots);
    for (std::size_t i = 0; i < slots; ++i)
        reference[i] = 0.3 * std::sin(double(i) * 0.07) + 0.2 * std::cos(double(i) * 0.17);
    Plaintext plaintext;
    encoder.encode(reference, context.crt_context()->parms_id_map().at(input_q - 1), scale, plaintext);
    Ciphertext ciphertext;
    if (bounded_keys)
    {
        Encryptor(context, secret).encrypt_symmetric(plaintext, ciphertext);
    }
    else
    {
        PublicKey public_key;
        generator.create_public_key(public_key);
        Encryptor(context, public_key, secret).encrypt(plaintext, ciphertext);
    }
    const auto input = GpuUploader::upload_ciphertext(ciphertext, 0);
    gpu_check_cuda(cudaDeviceSynchronize(), "finish uploading bootstrap resources");
    std::cout << "SETUP seconds=" << std::chrono::duration<double>(Clock::now() - setup_start).count()
              << " pool_reserved_MiB=" << pool.pool.pool_size() / double(1 << 20) << std::endl;

    const auto measure = [&](Profile which) {
        return measure_profile(which, repetitions, device, pool, evaluator, profile, input,
                               context, encoder, decryptor, reference);
    };
    const auto decoded = measure(selected);
    if (compare)
    {
        const auto alternate = selected.real_only ? dual : real;
        configure_profile(alternate);
        const auto other = measure(alternate);
        double max_delta = 0;
        double sum_squared = 0;
        for (std::size_t i = 0; i < decoded.size(); ++i)
        {
            const double delta = std::abs(decoded[i] - other[i]);
            max_delta = std::max(max_delta, delta);
            sum_squared += delta * delta;
        }
        std::cout << "BRANCH_COMPARISON slots=" << decoded.size() << " max_abs_delta=" << max_delta
                  << " rms_delta=" << std::sqrt(sum_squared / decoded.size()) << std::endl;
    }
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc > 4)
            throw std::invalid_argument(usage);
        const int repetitions = argc > 1 ? integer_argument(argv[1], 1, 100) : 18;
        const std::string_view mode = argc > 2 ? argv[2] : "9";
        const auto selected = select_profile(mode);
        const bool compare = argc == 4;
        if (compare && (std::string_view(argv[3]) != "--compare" || !selected.bounded_keys))
            throw std::invalid_argument(usage);
        run_benchmark(repetitions, selected, compare);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Agent22 benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
