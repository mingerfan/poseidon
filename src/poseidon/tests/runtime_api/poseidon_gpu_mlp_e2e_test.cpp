#include "poseidon/basics/modulus.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/gpu/gpu_bootstrap_profile.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/boot_recipe.h"
#include "poseidon/runtime_api/communication/cuda_local_transfer.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "poseidon/runtime_api/rotation_key_basis.h"
#include "runtime/execution_config.hpp"
#include "runtime/json_execution_config_reader.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/runtime.hpp"
#include "runtime/verifier.hpp"
#include "runtime_bundle_options.h"

#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <iterator>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

using Json = nlohmann::json;
using poseidon::runtime_api::PoseidonGpuApi;
using poseidon::runtime_api::PoseidonGpuValue;

constexpr double kAbsoluteTolerance = 0.1;
constexpr double kRelativeTolerance = 6e-3;
constexpr double kImaginaryTolerance = 1e-2;
// Runs of one process see the same input and the same plan; their outputs may
// differ only by floating-point noise, never by a corrupted value.
constexpr double kRunConsistencyTolerance = 1e-6;
constexpr const char *kAllowNoBootEnv =
    "POSEIDON_GPU_MLP_ALLOW_NO_BOOT";
constexpr const char *kDeviceWorkersEnv =
    "POSEIDON_RUNTIME_DEVICE_WORKERS";
constexpr const char *kDeviceBackfillEnv =
    "POSEIDON_RUNTIME_DEVICE_BACKFILL";
constexpr const char *kDeviceDependenciesEnv =
    "POSEIDON_RUNTIME_DEVICE_DEPENDENCIES";
// Kept so a caller that still sets it is recognised rather than silently
// ignored.  It no longer gates anything: publishing a Device output after
// submission and letting consumers wait on its CUDA event is now how the GPU
// Api always behaves, so the report states that rather than echoing the flag.
constexpr const char *kTransferEventsEnv =
    "POSEIDON_RUNTIME_TRANSFER_EVENTS";
constexpr const char *kRotationKeyPolicyEnv =
    "POSEIDON_RUNTIME_ROTATION_KEY_POLICY";
constexpr const char *kCudaProfilerRangeEnv =
    "POSEIDON_RUNTIME_CUDA_PROFILER_RANGE";

void require_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

bool env_flag_enabled(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" &&
           text != "off" && text != "OFF";
}

std::size_t env_positive_size(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return 0;
    }
    const std::string text(value);
    std::size_t consumed = 0;
    const unsigned long parsed = std::stoul(text, &consumed);
    if (consumed != text.size() || parsed == 0)
    {
        throw std::runtime_error(
            std::string(name) + " must be a positive integer");
    }
    return static_cast<std::size_t>(parsed);
}

/**
 * @brief One NVTX range, closed by the destructor or by an explicit end().
 *
 * The names this runner pushes -- "setup", "warmup", "online.iteration.N" --
 * are the same ones poseidon_gpu_mpi_runtime_e2e pushes, so a timeline
 * collected from the search backend can be read with the same selectors as one
 * collected from the final backend.
 */
class NvtxRange
{
public:
    explicit NvtxRange(const char *name)
    {
        nvtxRangePushA(name);
    }

    ~NvtxRange()
    {
        end();
    }

    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;

    void end()
    {
        if (!ended_)
        {
            ended_ = true;
            nvtxRangePop();
        }
    }

private:
    bool ended_ = false;
};

/**
 * @brief How many times to run the plan inside this one process, and with
 * which ExecutionConfig.
 *
 * The defaults reproduce the historical behaviour exactly: no warm-up, one
 * measured iteration, and scheduling taken from the environment, which is what
 * the caller gets when it passes no flags.
 */
struct SamplingOptions
{
    std::size_t warmups = 0;
    std::size_t iterations = 1;
    std::optional<std::string> execution_config;
    // Native no-hoist bootstrapping profile (generate_profiles.py output);
    // without it a Boot can only be the Host decrypt_reencrypt refresh.
    std::optional<std::string> native_config;
};

SamplingOptions parse_sampling_options(int argc, char **argv, int first)
{
    SamplingOptions options;
    for (int index = first; index < argc; ++index)
    {
        const std::string option(argv[index]);
        if (option != "--warmups" && option != "--iterations" &&
            option != "--execution-config" && option != "--native-config")
        {
            throw std::invalid_argument("unknown option: " + option);
        }
        if (index + 1 >= argc)
        {
            throw std::invalid_argument(option + " needs a value");
        }
        if (option == "--execution-config")
        {
            options.execution_config = std::string(argv[++index]);
            continue;
        }
        if (option == "--native-config")
        {
            options.native_config = std::string(argv[++index]);
            continue;
        }
        const std::string text(argv[++index]);
        std::size_t consumed = 0;
        const unsigned long parsed = std::stoul(text, &consumed);
        if (consumed != text.size())
        {
            throw std::invalid_argument(option + " needs an integer");
        }
        if (option == "--warmups")
        {
            options.warmups = static_cast<std::size_t>(parsed);
        }
        else
        {
            options.iterations = static_cast<std::size_t>(parsed);
        }
    }
    if (options.iterations == 0)
    {
        throw std::invalid_argument("--iterations must be positive");
    }
    return options;
}

double median_of(std::vector<double> values)
{
    if (values.empty())
    {
        throw std::invalid_argument("median of an empty sample");
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1)
    {
        return values[middle];
    }
    return 0.5 * (values[middle - 1] + values[middle]);
}

/**
 * @brief The online-latency statistics for one candidate.
 *
 * ``median_seconds`` is what adjudication compares; the rest are reported so a
 * run can be lined up against a poseidon final-test report, which carries
 * average/min/max and the per-iteration vector but no median.
 */
Json online_timing_json(const std::vector<double> &per_iteration_seconds)
{
    if (per_iteration_seconds.empty())
    {
        throw std::invalid_argument("no measured iterations");
    }
    const auto [minimum, maximum] = std::minmax_element(
        per_iteration_seconds.begin(), per_iteration_seconds.end());
    const double sum = std::accumulate(
        per_iteration_seconds.begin(), per_iteration_seconds.end(), 0.0);
    return {
        {"per_iteration_seconds", per_iteration_seconds},
        {"median_seconds", median_of(per_iteration_seconds)},
        {"average_seconds", sum / per_iteration_seconds.size()},
        {"min_seconds", *minimum},
        {"max_seconds", *maximum},
    };
}

struct Comparison
{
    double max_abs = 0.0;
    double rms = 0.0;
    double max_relative = 0.0;
    bool within_tolerance = true;
};

Json read_json(const std::filesystem::path &path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("cannot open JSON file: " + path.string());
    }
    Json value;
    input >> value;
    return value;
}

std::vector<double> read_numbers(const Json &root, const char *member,
                                 std::size_t expected_size)
{
    const auto &array = root.at(member);
    if (!array.is_array() || array.size() != expected_size)
    {
        throw std::runtime_error(std::string(member) + " has the wrong length");
    }
    std::vector<double> result;
    result.reserve(array.size());
    for (const auto &value : array)
    {
        const double number = value.get<double>();
        if (!std::isfinite(number))
        {
            throw std::runtime_error(std::string(member) + " contains a non-finite number");
        }
        result.push_back(number);
    }
    return result;
}

std::vector<double> pack_mlp_input(const std::vector<double> &logical)
{
    if (logical.size() != 784)
    {
        throw std::runtime_error("MLP logical input must contain 784 values");
    }
    std::vector<double> packed;
    packed.reserve(1600);
    for (std::size_t block_index = 0; block_index < 8; ++block_index)
    {
        std::vector<double> block(100, 0.0);
        const std::size_t begin = block_index * 100;
        const std::size_t count =
            std::min<std::size_t>(100, logical.size() - begin);
        std::copy_n(logical.begin() + begin, count, block.begin());
        packed.insert(packed.end(), block.begin(), block.end());
        packed.insert(packed.end(), block.begin(), block.end());
    }
    return packed;
}

std::vector<double> pack_resnet20_input(const std::vector<double> &logical)
{
    constexpr std::size_t image_values = 3 * 32 * 32;
    constexpr std::size_t packed_block = 4096;
    constexpr std::size_t repetitions = 4;
    constexpr double activation_scale = 32.0;
    if (logical.size() != image_values)
    {
        throw std::runtime_error(
            "ResNet-20 logical input must contain 3072 values");
    }
    std::vector<double> block(packed_block, 0.0);
    for (std::size_t index = 0; index < logical.size(); ++index)
    {
        block[index] = logical[index] / activation_scale;
    }
    std::vector<double> packed;
    packed.reserve(packed_block * repetitions);
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
    {
        packed.insert(packed.end(), block.begin(), block.end());
    }
    return packed;
}

std::uint32_t exact_log2(std::uint64_t value)
{
    if (value < 2 || (value & (value - 1)) != 0)
    {
        throw std::runtime_error("poly_degree must be a power of two");
    }
    std::uint32_t result = 0;
    while (value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

poseidon::PoseidonContext make_context(const fhegpu::OperatorSpec &spec,
                                      const Json &native_config)
{
    if (!native_config.is_null() && native_config.contains("p_moduli"))
    {
        // A hand-built chain (resnet20-manual-q50): the OperatorSpec's own Q
        // primes and the config's P primes, exactly as the Boot was validated.
        if (spec.rns_moduli.size() != spec.rns_moduli_log2.size())
            throw std::runtime_error("native-config p_moduli needs an OperatorSpec with rns_moduli");
        std::vector<poseidon::Modulus> q, p;
        for (auto value : spec.rns_moduli) q.emplace_back(value);
        for (const auto &value : native_config.at("p_moduli")) p.emplace_back(value.get<std::uint64_t>());
        const auto log_n = exact_log2(spec.poly_degree);
        poseidon::ParametersLiteral parameters(CKKS, log_n, log_n - 1,
            spec.default_scale_log2, native_config.at("hamming_weight").get<std::uint32_t>(),
            native_config.at("q0_level").get<std::uint32_t>(), poseidon::Modulus(0),
            q, p, poseidon::sec_level_type::none);
        return poseidon::PoseidonContext(parameters);
    }
    if (!native_config.is_null())
    {
        const auto p_bits = native_config.at("p_bits").get<std::vector<std::uint32_t>>();
        if (p_bits.empty()) throw std::runtime_error("Native bootstrap needs P moduli");
        const auto log_n = exact_log2(spec.poly_degree);
        poseidon::ParametersLiteral parameters(CKKS, log_n, log_n - 1,
            spec.default_scale_log2, native_config.at("hamming_weight").get<std::uint32_t>(),
            native_config.at("q0_level").get<std::uint32_t>(), poseidon::Modulus(0),
            {}, {}, poseidon::sec_level_type::none);
        // Match the Native component's concrete-prime generator, including
        // prime order. Equal bit widths alone do not identify a CKKS context.
        parameters.set_log_modulus(
            std::vector<std::uint32_t>(spec.rns_moduli_log2.begin(), spec.rns_moduli_log2.end()),
            p_bits);
        return poseidon::PoseidonContext(parameters);
    }
    std::vector<int> modulus_bits = spec.rns_moduli_log2;
    modulus_bits.insert(modulus_bits.end(), 2, spec.max_modulus_log2);
    const auto moduli = poseidon::CoeffModulus::Create(spec.poly_degree, modulus_bits);
    const std::size_t q_count = spec.rns_moduli_log2.size();
    std::vector<poseidon::Modulus> q(moduli.begin(), moduli.begin() + q_count);
    std::vector<poseidon::Modulus> p(moduli.begin() + q_count, moduli.end());
    const std::uint32_t log_n = exact_log2(spec.poly_degree);
    poseidon::ParametersLiteral parameters(
        CKKS, log_n, log_n - 1,
        static_cast<std::uint32_t>(spec.default_scale_log2), 0, 0,
        poseidon::Modulus(0), q, p, poseidon::sec_level_type::none);
    return poseidon::PoseidonContext(parameters);
}

const fhegpu::ValueDesc &find_value(const fhegpu::RuntimePlan &plan,
                                    fhegpu::ValueId id)
{
    const auto found = std::find_if(
        plan.values.begin(), plan.values.end(),
        [id](const fhegpu::ValueDesc &value) { return value.id == id; });
    if (found == plan.values.end())
    {
        throw std::runtime_error("missing ValueDesc " + std::to_string(id));
    }
    return *found;
}

std::size_t count_transfers(const fhegpu::RuntimePlan &plan)
{
    std::size_t count = 0;
    const auto add_phase = [&count](const std::vector<fhegpu::Instruction> &phase) {
        count += static_cast<std::size_t>(std::count_if(
            phase.begin(), phase.end(), [](const fhegpu::Instruction &instruction) {
                return std::holds_alternative<fhegpu::CommAction>(instruction.body);
            }));
    };
    add_phase(plan.initialization);
    add_phase(plan.execution);
    add_phase(plan.finalization);
    return count;
}

std::size_t count_boots(const fhegpu::RuntimePlan &plan)
{
    return static_cast<std::size_t>(std::count_if(
        plan.execution.begin(), plan.execution.end(),
        [](const fhegpu::Instruction &instruction) {
            const auto *op = std::get_if<fhegpu::ComputeOp>(&instruction.body);
            return op != nullptr && op->kind == fhegpu::ComputeKind::Boot;
        }));
}

// Boots the compiler cut into steps (V6): one raise each.
std::size_t count_expanded_boots(const fhegpu::RuntimePlan &plan)
{
    return static_cast<std::size_t>(std::count_if(
        plan.execution.begin(), plan.execution.end(),
        [](const fhegpu::Instruction &instruction) {
            const auto *op = std::get_if<fhegpu::ComputeOp>(&instruction.body);
            const auto *step = op ? std::get_if<fhegpu::BootStepAttrs>(&op->attrs) : nullptr;
            return step != nullptr && step->step == fhegpu::BootStepKind::Raise;
        }));
}

// The native profile a --native-config names. The fields are
// generate_profiles.py's; double_angle is optional (the builder's default).
poseidon::gpu::GpuBootstrapProfileConfig native_profile_config(const Json &native_config)
{
    poseidon::gpu::GpuBootstrapProfileConfig config;
    config.profile_id = native_config.at("profile_id").get<std::string>();
    const int mode = native_config.at("linear_mode").get<int>();
    if (mode != 0 && mode != 2 && mode != 3 && mode != 4)
        throw std::runtime_error("native-config linear_mode must be 0 (classic), 2 (double_hoist), "
                                 "3 (loose_coupled) or 4 (no_hoist)");
    config.linear_transform_mode = static_cast<poseidon::gpu::GpuLinearTransformMode>(mode);
    config.eval_mod_sine_degree = native_config.at("evalmod_degree").get<std::uint32_t>();
    config.eval_mod_generation_degree = native_config.at("generation_degree").get<std::uint32_t>();
    config.eval_mod_log_scale = native_config.at("evalmod_log_scale").get<std::uint32_t>();
    config.project_real = native_config.at("project_real").get<bool>();
    if (native_config.contains("double_angle"))
        config.eval_mod_double_angle = native_config.at("double_angle").get<std::uint32_t>();
    // q0 / message ratio (upstream ResNet18 128, ResNet50 2048); a larger one
    // keeps EvalMod's sine error small for Boot inputs well above 1.
    config.bootstrap_ratio = native_config.value("bootstrap_ratio", config.bootstrap_ratio);
    config.eval_mod_truncate_degree.reset();
    // Sparse-secret encapsulation and the low-degree EvalMod it allows.
    config.eval_mod_k = native_config.value("evalmod_k", config.eval_mod_k);
    config.eval_mod_log_split = native_config.value("evalmod_log_split", config.eval_mod_log_split);
    config.eval_mod_full_coefficients = native_config.value("full_cosine_coefficients", false);
    config.sparse_secret_hamming_weight = native_config.value("sparse_secret_hamming_weight", 0U);
    config.compress_linear_plaintexts = native_config.value("compress_linear_plaintexts", false);
    if (native_config.contains("output_level"))
        config.output_level = native_config.at("output_level").get<int>();
    return config;
}

poseidon::gpu::GpuResNet20ManualOptions manual_profile_options(const Json &native_config)
{
    poseidon::gpu::GpuResNet20ManualOptions options;
    options.sine_degree = native_config.value("evalmod_degree", options.sine_degree);
    options.double_angle = native_config.value("double_angle", options.double_angle);
    options.k = native_config.value("evalmod_k", options.k);
    options.log_split = native_config.value("evalmod_log_split", options.log_split);
    options.full_coefficients = native_config.value("full_cosine_coefficients", false);
    options.sparse_secret_hamming_weight = native_config.value("sparse_secret_hamming_weight", 0U);
    return options;
}

// The built profile must be the one the OperatorSpec plans with.
void require_spec_profile(const fhegpu::OperatorSpec &spec,
                          const poseidon::gpu::GpuBootstrapProfile &profile)
{
    const auto expected = std::find_if(spec.boot_profiles.begin(), spec.boot_profiles.end(),
        [&](const auto &candidate) { return candidate.profile_id == profile.profile_id; });
    if (expected == spec.boot_profiles.end() ||
        expected->output_level != profile.output_level ||
        expected->output_scale_log2 != profile.output_scale_log2 ||
        expected->input_level_min < profile.input_level_min ||
        expected->input_level_max > profile.input_level_max ||
        (expected->input_scale_log2 && profile.input_scale_log2 &&
         *expected->input_scale_log2 != *profile.input_scale_log2))
        throw std::runtime_error("built Native bootstrap does not match OperatorSpec");
}

// builder "resnet20-manual-q50" is the hand-written S2C-first Boot (Q6/s40 ->
// Q31/s40, fixed degree59/DA2); any other config goes through the generic builder.
poseidon::gpu::GpuBootstrapProfile build_native_profile(
    const poseidon::PoseidonContext &context, poseidon::KeyGenerator &key_generator,
    int cuda_device, const Json &native_config)
{
    const std::string builder = native_config.value("builder", "generic");
    if (builder == "resnet20-manual-q50")
        return poseidon::gpu::GpuBootstrapProfileBuilder::build_resnet20_manual(
            context, key_generator, cuda_device, manual_profile_options(native_config));
    if (builder != "generic")
        throw std::runtime_error("native-config builder must be generic or resnet20-manual-q50");
    return poseidon::gpu::GpuBootstrapProfileBuilder::build(
        context, key_generator, cuda_device, native_profile_config(native_config));
}

std::uint64_t device_plaintext_storage_lower_bound(
    const fhegpu::RuntimePlan &plan, std::uint64_t poly_degree)
{
    std::uint64_t bytes = 0, peak = 0;
    std::unordered_map<fhegpu::ValueId, std::uint64_t> live;
    const auto publish = [&](fhegpu::ValueId id) {
        const auto &value = find_value(plan, id);
        if (value.kind == fhegpu::ValueKind::Plaintext &&
            value.place.kind == fhegpu::PlaceKind::Device && !live.count(id))
        {
            const auto size = static_cast<std::uint64_t>(value.level + 1) * poly_degree *
                              sizeof(std::uint32_t);
            live.emplace(id, size);
            bytes += size;
            peak = std::max(peak, bytes);
        }
    };
    // Release planning makes the sum of all encoded weights a lifetime total,
    // not a simultaneous memory lower bound.
    const auto phase = [&](const std::vector<fhegpu::Instruction> &instructions) {
        for (const auto &instruction : instructions)
        {
            if (const auto *op = std::get_if<fhegpu::EncodeOp>(&instruction.body)) publish(op->output);
            else if (const auto *op = std::get_if<fhegpu::PreparePlaintextOp>(&instruction.body)) publish(op->output);
            else if (const auto *op = std::get_if<fhegpu::CommAction>(&instruction.body))
                for (auto id : op->outputs) publish(id);
            else if (const auto *op = std::get_if<fhegpu::ReleaseOp>(&instruction.body))
            {
                const auto found = live.find(op->value);
                if (found != live.end()) { bytes -= found->second; live.erase(found); }
            }
        }
    };
    phase(plan.initialization);
    phase(plan.execution);
    phase(plan.finalization);
    return peak;
}

std::uint64_t hybrid_key_storage_lower_bound(
    std::uint64_t poly_degree, std::size_t q_count, std::size_t p_count,
    std::size_t key_count)
{
    const std::size_t decomposition_count =
        (q_count + p_count - 1) / p_count;
    return static_cast<std::uint64_t>(key_count) * decomposition_count *
           (q_count + p_count) * 2 * poly_degree * sizeof(std::uint32_t);
}

void require_device_memory_lower_bound(
    const fhegpu::RuntimePlan &plan, const fhegpu::OperatorSpec &spec,
    std::size_t rotation_key_count, bool needs_relin, std::size_t p_count)
{
    // A waterline plan lists every PreparePlaintext up front and the executor
    // admits them against the plan's own device budget as the run reaches
    // them, so scanning the instruction list in order would count every
    // plaintext of the run as live at once.  The ledger enforces that budget;
    // only the keys are a static lower bound here.
    const std::uint64_t plaintext_bytes =
        plan.waterline ? 0 : device_plaintext_storage_lower_bound(plan, spec.poly_degree);
    const std::uint64_t key_bytes = hybrid_key_storage_lower_bound(
        spec.poly_degree, spec.rns_moduli_log2.size(), p_count,
        rotation_key_count + (needs_relin ? 1 : 0));
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(status));
    }
    const std::uint64_t lower_bound = plaintext_bytes + key_bytes;
    if (lower_bound > free_bytes)
    {
        throw std::runtime_error(
            "GPU static memory lower bound exceeds free memory: lower_bound=" +
            std::to_string(lower_bound) + " free=" +
            std::to_string(free_bytes) + " total=" +
            std::to_string(total_bytes) + " plaintext=" +
            std::to_string(plaintext_bytes) + " keys=" +
            std::to_string(key_bytes));
    }
}

PoseidonGpuValue download(PoseidonGpuApi &api, fhegpu::ValueId id,
                          const fhegpu::ValueDesc &source_desc,
                          const PoseidonGpuValue &value)
{
    if (source_desc.place.kind == fhegpu::PlaceKind::Host)
    {
        return value;
    }
    fhegpu::CommAction action;
    action.id = 0;
    action.kind = fhegpu::CommKind::Transfer;
    action.hint = fhegpu::CommHint::PointToPoint;
    action.inputs = {id};
    action.outputs = {id + 1};
    action.sources = {source_desc.place};
    action.destinations = {{fhegpu::PlaceKind::Host, 0, 0}};
    action.output_types = {fhegpu::ValueKind::Ciphertext};
    auto output_desc = source_desc;
    output_desc.id = id + 1;
    output_desc.place = action.destinations.front();
    auto handle = api.communicate_async(action, {value}, {output_desc});
    auto outputs = api.wait(handle);
    if (outputs.size() != 1 || !outputs.front())
    {
        throw std::runtime_error("GPU result download produced the wrong output count");
    }
    return std::move(*outputs.front());
}

Comparison compare(const std::vector<double> &expected,
                   const std::vector<double> &actual,
                   double absolute_tolerance = kAbsoluteTolerance,
                   double relative_tolerance = kRelativeTolerance)
{
    Comparison result;
    double squared_error = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        const double difference = std::abs(expected[index] - actual[index]);
        result.max_abs = std::max(result.max_abs, difference);
        squared_error += difference * difference;
        const double relative =
            difference / std::max(std::abs(expected[index]), absolute_tolerance);
        result.max_relative = std::max(result.max_relative, relative);
        if (difference > absolute_tolerance +
                             relative_tolerance * std::abs(expected[index]))
        {
            result.within_tolerance = false;
        }
    }
    result.rms = std::sqrt(squared_error / expected.size());
    return result;
}

Json comparison_json(const Comparison &comparison)
{
    return {{"max_abs", comparison.max_abs},
            {"rms", comparison.rms},
            {"max_relative", comparison.max_relative},
            {"within_tolerance", comparison.within_tolerance}};
}

void write_json(const std::filesystem::path &path, const Json &value)
{
    if (!path.parent_path().empty())
    {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output)
    {
        throw std::runtime_error("cannot write JSON file: " + path.string());
    }
    output << value.dump(2) << '\n';
}

/// Exit status for a run refused before any GPU work: an ExecutionConfig that
/// does not belong to this plan, or scheduling given two ways at once.  Kept
/// apart from 1 (ran and disagreed, or failed) so a caller never mistakes a
/// refusal for a wrong answer it may step over.
constexpr int kRefusedExitCode = 3;

/**
 * @brief What this runner reads from an ExecutionConfig, printed without
 * touching a GPU so a caller can find out before it builds one.
 */
Json capabilities_json()
{
    return {
        {"format_version", 1},
        {"runner", "poseidon_runtime_gpu_mlp_e2e"},
        {"options", {"--warmups", "--iterations", "--execution-config", "--native-config"}},
        // "qwen": slot-vector fixtures (several inputs and outputs, own tolerances).
        {"fixture_models", {"mlp", "resnet20", "qwen"}},
        {"native_config", {{"bootstrap_ratio", true}}},
        // Plans this runner executes: V6 adds the steps of compiler-expanded Boots.
        {"runtime_plan", {{"format_versions", {1, 2, 3, 4, 5, 6}}, {"boot_steps", true}}},
        {"modes", {"--export-boot-recipe OPERATOR_SPEC NATIVE_CONFIG OUT_JSON [CUDA_DEVICE]"}},
        {"execution_config",
         {{"format_version", 1},
          {"applied", fhegpu::applied_fields()},
          {"ignored", fhegpu::ignored_fields()},
          {"refused_exit_code", kRefusedExitCode}}},
    };
}

/**
 * @brief The scheduling environment variables an ExecutionConfig replaces.
 *
 * Setting one of them beside a config would leave two answers to the same
 * question, so the run is refused rather than one silently winning.
 */
std::vector<std::string> scheduling_environment_in_use()
{
    std::vector<std::string> set;
    for (const char *name :
         {kDeviceWorkersEnv, kDeviceBackfillEnv, kDeviceDependenciesEnv})
    {
        const char *value = std::getenv(name);
        if (value != nullptr && value[0] != '\0')
        {
            set.emplace_back(name);
        }
    }
    return set;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc == 2 && std::string(argv[1]) == "--capabilities")
    {
        std::cout << capabilities_json().dump(2) << '\n';
        return 0;
    }
    // The Boot recipe expand-ckks-bootstrap reads, from the very profile a
    // run with this OperatorSpec and native config builds.
    if (argc >= 5 && argc <= 6 && std::string(argv[1]) == "--export-boot-recipe")
    {
        try
        {
            const auto loaded_spec = fhegpu::OperatorSpecReader::read_file(argv[2]);
            const Json native_config = read_json(argv[3]);
            const int cuda_device = argc == 6 ? std::stoi(argv[5]) : 0;
            poseidon::PoseidonContext context = make_context(loaded_spec.spec, native_config);
            poseidon::KeyGenerator key_generator(context);
            const auto profile =
                build_native_profile(context, key_generator, cuda_device, native_config);
            require_spec_profile(loaded_spec.spec, profile);
            std::ofstream output(argv[4]);
            output << poseidon::runtime_api::make_boot_recipe(profile, context).dump(1) << '\n';
            if (!output)
                throw std::runtime_error(std::string("cannot write ") + argv[4]);
            std::cout << "BOOT_RECIPE profile=" << profile.profile_id << " out=" << argv[4] << '\n';
            return 0;
        }
        catch (const std::exception &error)
        {
            std::cerr << "poseidon_runtime_gpu_mlp_e2e: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc < 7)
    {
        std::cerr << "usage: poseidon_runtime_gpu_mlp_e2e PLAN OPERATOR_SPEC "
                     "BUNDLE_DIR FIXTURE MOCK_RESULT REPORT_JSON "
                     "[--warmups N] [--iterations N] [--execution-config PATH] "
                     "[--native-config PATH]\n"
                     "       poseidon_runtime_gpu_mlp_e2e --capabilities\n";
        return 2;
    }

    SamplingOptions sampling;
    try
    {
        sampling = parse_sampling_options(argc, argv, 7);
    }
    catch (const std::exception &error)
    {
        std::cerr << "poseidon_runtime_gpu_mlp_e2e: " << error.what() << '\n';
        return 2;
    }

    try
    {
        const auto loaded_plan = fhegpu::RuntimePlanReader::read_file(argv[1]);
        const auto loaded_spec = fhegpu::OperatorSpecReader::read_file(argv[2]);
        const auto &plan = loaded_plan.plan;
        // One input and one output, except for a slot-vector fixture (checked
        // against the fixture once it is read).
        if (plan.target.world_size != 1 || plan.target.device_counts.size() != 1 ||
            plan.target.device_counts.front() <= 0 ||
            plan.external_inputs.empty() || plan.final_outputs.empty())
        {
            throw std::runtime_error(
                "GPU MLP test requires one rank, at least one device, an input, and an output");
        }
        const int local_device_count = plan.target.device_counts.front();

        // The ExecutionConfig is read and checked here, before the context,
        // the keys (minutes at 65536) or any device is touched: a config that
        // was generated for another plan must cost nothing to reject.
        std::optional<fhegpu::LoadedExecutionConfig> execution_config;
        if (sampling.execution_config)
        {
            try
            {
                execution_config = fhegpu::ExecutionConfigJsonReader::read_file(
                    *sampling.execution_config);
                fhegpu::verify_execution_config(
                    *execution_config, loaded_plan, 0, 1, local_device_count);
                const auto conflicting = scheduling_environment_in_use();
                if (!conflicting.empty())
                {
                    std::string names;
                    for (const auto &name : conflicting)
                    {
                        names += (names.empty() ? "" : ", ") + name;
                    }
                    throw std::runtime_error(
                        "scheduling is given twice: --execution-config and " + names);
                }
                int visible_devices = 0;
                require_cuda(cudaGetDeviceCount(&visible_devices), "cudaGetDeviceCount");
                std::set<int> seen;
                for (const int id : fhegpu::cuda_device_ids_of(execution_config->config))
                {
                    if (id < 0 || id >= visible_devices || !seen.insert(id).second)
                    {
                        throw std::runtime_error(
                            "devices[].cuda_device_id " + std::to_string(id) +
                            " is out of range or repeated for " +
                            std::to_string(visible_devices) + " visible device(s)");
                    }
                }
            }
            catch (const std::exception &error)
            {
                std::cerr << "REFUSED " << error.what() << '\n';
                return kRefusedExitCode;
            }
        }

        // Only now the rest of the inputs: a refused config never reads them.
        const Json fixture = read_json(argv[4]);
        const Json mock_result = read_json(argv[5]);

        std::vector<int> cuda_device_ids;
        if (execution_config)
        {
            cuda_device_ids = fhegpu::cuda_device_ids_of(execution_config->config);
        }
        else
        {
            cuda_device_ids.reserve(static_cast<std::size_t>(local_device_count));
            for (int device = 0; device < local_device_count; ++device)
            {
                cuda_device_ids.push_back(device);
            }
        }
        if (fixture.at("format_version") != 1 ||
            mock_result.at("format_version") != 1 ||
            mock_result.at("passed") != true ||
            fixture.at("seed") != mock_result.at("seed") ||
            fixture.at("model_sha256") != mock_result.at("model_sha256"))
        {
            throw std::runtime_error("fixture and MockVecApi result do not match");
        }
        const std::string model = fixture.value("model", "mlp");
        if (model != "mlp" && model != "resnet20" && model != "qwen")
        {
            throw std::runtime_error("unsupported model fixture");
        }
        // model "qwen": whole slot vectors.  inputs[i] encrypts to
        // plan.external_inputs[i]; expected[i] is the valid prefix of
        // plan.final_outputs[i] and every slot after it must decrypt to ~0.
        const bool slot_fixture = model == "qwen";
        if (slot_fixture
                ? (plan.external_inputs.size() != fixture.at("inputs").size() ||
                   plan.final_outputs.size() != fixture.at("expected").size())
                : (plan.external_inputs.size() != 1 || plan.final_outputs.size() != 1))
        {
            throw std::runtime_error("the plan's inputs/outputs do not match the fixture");
        }
        const Json native_config =
            sampling.native_config ? read_json(*sampling.native_config) : Json();
        const bool native_nohoist = !native_config.is_null();
        const bool frontend_packed =
            fixture.value("input_encoding", "legacy") == "frontend-packed-slots";
        if (fixture.value("bootstrap_requirement", "") == "native-no-hoist" && !native_nohoist)
            throw std::runtime_error("frontend fixture requires --native-config; Host refresh is forbidden");
        if (native_nohoist && !frontend_packed)
            throw std::runtime_error("a native Boot needs the frontend-packed fixture");
        const std::size_t plan_boots = count_boots(plan);
        const std::size_t plan_expanded_boots = count_expanded_boots(plan);
        if (plan_expanded_boots != 0 && !native_nohoist)
            throw std::runtime_error("a plan with expanded Boots needs --native-config for their profile");
        const bool allow_no_boot = env_flag_enabled(kAllowNoBootEnv);
        if (plan_boots == 0 && plan_expanded_boots == 0 && !allow_no_boot)
        {
            throw std::runtime_error(
                "GPU MLP RuntimePlan contains no Host Boot; set " +
                std::string(kAllowNoBootEnv) +
                "=1 to run a no-Boot calibration plan");
        }

        const auto requirements = fhegpu::PlanVerifier::verify(
            plan, loaded_spec, false);
        if (native_nohoist)
            for (const auto &profile : loaded_spec.spec.boot_profiles)
                if (profile.implementation != fhegpu::BootImplementation::Native)
                    throw std::runtime_error("Native frontend forbids decrypt_reencrypt profiles");
        poseidon::PoseidonContext context = make_context(loaded_spec.spec, native_config);
        poseidon::KeyGenerator key_generator(context);
        auto secret_key =
            std::make_shared<poseidon::SecretKey>(key_generator.secret_key());
        auto public_key = std::make_shared<poseidon::PublicKey>();
        auto relin_keys = std::make_shared<poseidon::RelinKeys>();
        auto galois_keys = std::make_shared<poseidon::GaloisKeys>();
        bool needs_relin = false;
        std::set<int> logical_rotation_steps;
        for (const auto &key : requirements.keys)
        {
            if (key.kind == fhegpu::KeyKind::Relin)
            {
                needs_relin = true;
            }
            else if (key.kind == fhegpu::KeyKind::Galois)
            {
                if (!key.rotation_step)
                {
                    throw std::runtime_error("Galois key requirement has no rotation step");
                }
                logical_rotation_steps.insert(*key.rotation_step);
            }
        }
        const char *rotation_key_policy_value =
            std::getenv(kRotationKeyPolicyEnv);
        const std::string rotation_key_policy =
            rotation_key_policy_value == nullptr || rotation_key_policy_value[0] == '\0'
                ? "binary"
                : rotation_key_policy_value;
        if (rotation_key_policy != "binary" && rotation_key_policy != "direct" &&
            rotation_key_policy != "auto")
        {
            throw std::runtime_error(
                std::string(kRotationKeyPolicyEnv) + " must be binary, direct or auto");
        }

        std::set<int> rotation_key_steps;
        std::size_t auto_direct_key_count = 0;
        if (rotation_key_policy == "auto")
        {
            // The power-of-two basis (NAF by default), plus direct keys for
            // the steps that save the most key switches, within a byte budget.
            const std::size_t slot_count = context.parameters_literal()->slot();
            rotation_key_steps = poseidon::runtime_api::binary_rotation_key_basis(
                logical_rotation_steps, slot_count);
            std::map<int, std::size_t> uses;
            for (const auto *section : {&plan.initialization, &plan.execution, &plan.finalization})
                for (const auto &instruction : *section)
                    if (const auto *op = std::get_if<fhegpu::ComputeOp>(&instruction.body);
                        op != nullptr && op->kind == fhegpu::ComputeKind::Rotate)
                    {
                        const int step = poseidon::runtime_api::normalize_rotation_step(
                            std::get<fhegpu::RotateAttrs>(op->attrs).steps, slot_count);
                        if (step != 0)
                            ++uses[step];
                    }
            std::vector<std::pair<std::size_t, int>> savings;
            for (const auto &[step, count] : uses)
            {
                const auto terms =
                    poseidon::runtime_api::decompose_rotation_step(step, slot_count).size();
                if (!rotation_key_steps.count(step) && terms > 1)
                    savings.emplace_back(count * (terms - 1), step);
            }
            std::sort(savings.begin(), savings.end(), std::greater<>());
            const char *budget_raw = std::getenv("POSEIDON_RUNTIME_DIRECT_KEY_BUDGET_MB");
            const std::uint64_t budget = budget_raw == nullptr ? 0
                : static_cast<std::uint64_t>(std::stoull(budget_raw)) << 20;
            const std::uint64_t per_key = hybrid_key_storage_lower_bound(
                loaded_spec.spec.poly_degree, loaded_spec.spec.rns_moduli_log2.size(),
                context.parameters_literal()->p().size(), 1);
            for (const auto &[saved, step] : savings)
            {
                if (per_key * (auto_direct_key_count + 1) > budget)
                    break;
                rotation_key_steps.insert(step);
                ++auto_direct_key_count;
            }
            std::cerr << "ROTATION_KEYS policy=auto basis=" << rotation_key_steps.size() - auto_direct_key_count
                      << " direct=" << auto_direct_key_count << " of " << savings.size()
                      << " per_key_bytes=" << per_key << '\n';
        }
        else if (rotation_key_policy == "direct")
        {
            const std::size_t slot_count = context.parameters_literal()->slot();
            for (int step : logical_rotation_steps)
            {
                const int normalized =
                    poseidon::runtime_api::normalize_rotation_step(step, slot_count);
                if (normalized != 0)
                {
                    rotation_key_steps.insert(normalized);
                }
            }
        }
        else
        {
            rotation_key_steps = poseidon::runtime_api::binary_rotation_key_basis(
                logical_rotation_steps, context.parameters_literal()->slot());
        }

        require_device_memory_lower_bound(
            plan, loaded_spec.spec, rotation_key_steps.size(), needs_relin,
            context.parameters_literal()->p().size());

        const auto key_start = std::chrono::steady_clock::now();
        key_generator.create_public_key(*public_key);
        if (needs_relin)
        {
            key_generator.create_relin_keys(*relin_keys);
        }
        if (!rotation_key_steps.empty())
        {
            key_generator.create_galois_keys(
                std::vector<int>(rotation_key_steps.begin(), rotation_key_steps.end()),
                *galois_keys);
        }
        const auto key_finish = std::chrono::steady_clock::now();

        // Inputs as packed slot vectors, one per external input, and the
        // values compared after a run: the first 10 slots of the one output,
        // or every slot of every output (the expected prefix padded with the
        // zeros its tail must decrypt to) for a slot fixture.
        const std::size_t slot_count = loaded_spec.spec.poly_degree / 2;
        std::vector<std::vector<double>> input_slots;
        std::vector<double> python_output, mock_output;
        if (slot_fixture)
        {
            const auto &rows = fixture.at("inputs");
            for (std::size_t i = 0; i < rows.size(); ++i)
            {
                input_slots.push_back(read_numbers(Json{{"row", rows.at(i)}}, "row", slot_count));
            }
            const auto &expected = fixture.at("expected");
            const auto &mock_rows = mock_result.at("outputs");
            if (mock_rows.size() != expected.size())
                throw std::runtime_error("MockVec result has the wrong output count");
            for (std::size_t i = 0; i < expected.size(); ++i)
            {
                const Json row{{"row", expected.at(i)}};
                auto prefix = read_numbers(row, "row", expected.at(i).size());
                if (prefix.empty() || prefix.size() > slot_count)
                    throw std::runtime_error("fixture expected row does not fit a ciphertext");
                prefix.resize(slot_count, 0.0);
                python_output.insert(python_output.end(), prefix.begin(), prefix.end());
                const auto mock = read_numbers(Json{{"row", mock_rows.at(i)}}, "row", slot_count);
                mock_output.insert(mock_output.end(), mock.begin(), mock.end());
            }
        }
        else
        {
            const std::size_t logical_input_size =
                model == "resnet20" ? 3 * 32 * 32 : 784;
            const auto logical_input =
                read_numbers(fixture, "input", logical_input_size);
            input_slots.push_back(
                frontend_packed ? read_numbers(fixture, "packed_input", slot_count)
                    : model == "resnet20" ? pack_resnet20_input(logical_input)
                                          : pack_mlp_input(logical_input));
            python_output = read_numbers(fixture, "python_output", 10);
            mock_output = read_numbers(mock_result, "output", 10);
        }
        const std::size_t decoded_per_output = slot_fixture ? slot_count : 10;
        const fhegpu::ValueId input_id = plan.external_inputs.front();
        const auto &input_desc = find_value(plan, input_id);
        poseidon::CKKSEncoder encoder(context);
        poseidon::Encryptor encryptor(context, *public_key);
        std::vector<poseidon::Ciphertext> input_ciphers(plan.external_inputs.size());
        for (std::size_t i = 0; i < plan.external_inputs.size(); ++i)
        {
            const auto &desc = find_value(plan, plan.external_inputs[i]);
            poseidon::Plaintext input_plain;
            encoder.encode(
                input_slots.at(i),
                context.crt_context()->parms_id_map().at(static_cast<std::uint32_t>(desc.level)),
                std::ldexp(1.0, desc.scale_log2), input_plain);
            encryptor.encrypt(input_plain, input_ciphers[i]);
        }

        const bool transfer_events_requested =
            env_flag_enabled(kTransferEventsEnv);
        std::size_t device_worker_count = 0;
        fhegpu::DeviceExecutionMode device_execution_mode =
            fhegpu::DeviceExecutionMode::Sequential;
        if (execution_config)
        {
            device_worker_count =
                fhegpu::device_worker_count_of(execution_config->config);
            device_execution_mode =
                fhegpu::device_execution_mode_of(execution_config->config);
        }
        else
        {
            device_worker_count = env_positive_size(kDeviceWorkersEnv);
            const bool backfill = env_flag_enabled(kDeviceBackfillEnv);
            const bool dependencies = env_flag_enabled(kDeviceDependenciesEnv);
            if ((backfill || dependencies) && device_worker_count == 0)
            {
                throw std::runtime_error(
                    std::string("device backfill/dependencies require ") +
                    kDeviceWorkersEnv);
            }
            if (backfill && dependencies)
            {
                throw std::runtime_error(
                    std::string(kDeviceBackfillEnv) + " and " +
                    kDeviceDependenciesEnv + " are mutually exclusive");
            }
            device_execution_mode =
                dependencies ? fhegpu::DeviceExecutionMode::PerDeviceDependencyWorkers
                : backfill   ? fhegpu::DeviceExecutionMode::PerDeviceReadyWorkers
                : device_worker_count != 0
                    ? fhegpu::DeviceExecutionMode::PerDeviceWorkers
                    : fhegpu::DeviceExecutionMode::Sequential;
        }
        const bool device_workers =
            device_execution_mode != fhegpu::DeviceExecutionMode::Sequential;
        const bool device_backfill =
            device_execution_mode == fhegpu::DeviceExecutionMode::PerDeviceReadyWorkers;
        const bool device_dependencies =
            device_execution_mode == fhegpu::DeviceExecutionMode::PerDeviceDependencyWorkers;
        PoseidonGpuApi api(loaded_spec.spec.context_id, context, cuda_device_ids,
                           relin_keys, galois_keys,
                           native_nohoist ? nullptr : public_key,
                           native_nohoist ? nullptr : secret_key);
        if (native_nohoist)
        {
            // Every device that may run a Boot gets its own profile: the
            // bootstrapping keys and tables live on that device.
            for (int device = 0; device < local_device_count; ++device)
            {
                auto profile = build_native_profile(
                    context, key_generator, cuda_device_ids.at(device), native_config);
                require_spec_profile(loaded_spec.spec, profile);
                std::cout << "NATIVE_BOOT_CONFIG linear_mode="
                          << static_cast<int>(profile.bootstrap_data.linear_transform_mode)
                          << " profile=" << profile.profile_id
                          << " device=" << device << " cuda_device=" << cuda_device_ids.at(device)
                          << " output_level=" << profile.output_level << std::endl;
                api.configure_native_bootstrap(device, std::move(profile));
            }
        }
        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(
            0, 1, local_device_count, api, device_execution_mode,
            device_worker_count);
        // The runtime checks the binding again on every run; a config that
        // passed above passes there too, and one that did not never got here.
        const fhegpu::RuntimeResources resources{
            loaded_spec, std::filesystem::path(argv[3]), false,
            runtime_bundle_options_from_env(),
            execution_config ? &*execution_config : nullptr};
        std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
        for (std::size_t i = 0; i < plan.external_inputs.size(); ++i)
        {
            inputs.emplace(plan.external_inputs[i],
                           PoseidonGpuValue::from_host_ciphertext(std::move(input_ciphers[i])));
        }
        const bool cuda_profiler_range = env_flag_enabled(kCudaProfilerRangeEnv);

        const fhegpu::ValueId output_id = plan.final_outputs.front();
        const auto &output_desc = find_value(plan, output_id);
        poseidon::Decryptor decryptor(context, *secret_key);

        // Decrypt one run's outputs, in plan order, concatenated.  Done for
        // every run, outside the timed window: a resource released before the
        // GPU finished with it would corrupt some later run, not necessarily
        // the last one.
        // The scale each output ciphertext carries when it is decoded, last
        // run: what decode divides by, beside the plan's nominal scale.
        std::vector<double> output_scales(plan.final_outputs.size(), 0.0);
        const auto decode_output = [&](const auto &run_artifact) {
            std::vector<double> values;
            values.reserve(decoded_per_output * plan.final_outputs.size());
            double imaginary = 0.0;
            for (std::size_t o = 0; o < plan.final_outputs.size(); ++o)
            {
                const fhegpu::ValueId id = plan.final_outputs[o];
                auto host_output = download(
                    api, id, find_value(plan, id), run_artifact.values.at(id).value);
                output_scales[o] = host_output.host_ciphertext().scale();
                poseidon::Plaintext output_plain;
                decryptor.decrypt(host_output.host_ciphertext(), output_plain);
                std::vector<std::complex<double>> decoded;
                encoder.decode(output_plain, decoded);
                for (std::size_t index = 0; index < decoded_per_output; ++index)
                {
                    values.push_back(decoded.at(index).real());
                    imaginary = std::max(imaginary, std::abs(decoded.at(index).imag()));
                }
            }
            return std::make_pair(values, imaginary);
        };

        // Per run, outside the timed window: what was still held when it
        // ended, the answer it produced, then everything it held released
        // (drain) and the device memory left in use.  A run starting with
        // entries still pending would be paying for the previous run inside
        // its own timing; memory that keeps rising from run to run is a leak.
        Json runs_detail = Json::array();
        std::vector<double> first_output;
        double largest_deviation = 0.0;
        const auto run_checked = [&](const std::string &label, bool timed) {
            const auto at_start = api.in_flight_stats();
            auto run_artifact = [&] {
                NvtxRange range(label.c_str());
                return runtime.run(loaded_plan, resources, inputs);
            }();
            const auto after_run = api.in_flight_stats();
            const auto [values, imaginary] = decode_output(run_artifact);
            const auto drain_start = std::chrono::steady_clock::now();
            const std::size_t drained = api.drain_in_flight();
            const double drain_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - drain_start).count();
            std::vector<double> used_mib;
            int previous = 0;
            require_cuda(cudaGetDevice(&previous), "cudaGetDevice");
            for (const int id : cuda_device_ids)
            {
                std::size_t free_bytes = 0;
                std::size_t total_bytes = 0;
                require_cuda(cudaSetDevice(id), "cudaSetDevice");
                require_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
                used_mib.push_back(static_cast<double>(total_bytes - free_bytes) / (1 << 20));
            }
            require_cuda(cudaSetDevice(previous), "cudaSetDevice restore");
            if (first_output.empty())
            {
                first_output = values;
            }
            double deviation = 0.0;
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                deviation = std::max(deviation, std::abs(values[index] - first_output[index]));
            }
            largest_deviation = std::max(largest_deviation, deviation);
            const Comparison against = compare(mock_output, values);
            runs_detail.push_back({
                {"label", label},
                {"timed", timed},
                {"online_execution_seconds",
                 static_cast<double>(run_artifact.timing.online_execution_nanoseconds) * 1e-9},
                {"mock_max_abs", against.max_abs},
                {"deviation_from_first_run", deviation},
                {"max_imaginary", imaginary},
                {"in_flight_at_start", at_start.pending},
                {"released_during_run", after_run.released - at_start.released},
                {"in_flight_after_run", after_run.pending},
                {"drained_after_run", drained},
                {"drain_seconds", drain_seconds},
                {"device_used_mib", used_mib},
            });
            return std::make_tuple(std::move(run_artifact), values, imaginary);
        };

        // Warm-ups run outside the profiler range and are never recorded. The
        // first run in a process pays CUDA context creation, memory-pool
        // allocation and kernel warm-up; charging that to a candidate is what
        // forced the single-card reference to grow a warm-up pass of its own.
        for (std::size_t index = 0; index < sampling.warmups; ++index)
        {
            (void)run_checked("warmup." + std::to_string(index + 1), false);
        }

        if (cuda_profiler_range)
        {
            require_cuda(cudaProfilerStart(), "cudaProfilerStart");
        }
        // Optional CPU profile of the timed iterations only, with gperftools
        // preloaded (LD_PRELOAD=libprofiler.so); looked up at run time so the
        // runner has no link dependency on it.
        using ProfilerStopFn = void (*)();
        ProfilerStopFn cpu_profiler_stop = nullptr;
        if (const char *cpu_profile = std::getenv("POSEIDON_RUNNER_CPU_PROFILE"))
        {
            using ProfilerStartFn = int (*)(const char *);
            const auto start = reinterpret_cast<ProfilerStartFn>(dlsym(RTLD_DEFAULT, "ProfilerStart"));
            cpu_profiler_stop = reinterpret_cast<ProfilerStopFn>(dlsym(RTLD_DEFAULT, "ProfilerStop"));
            if (start == nullptr || cpu_profiler_stop == nullptr || start(cpu_profile) == 0)
                throw std::runtime_error("POSEIDON_RUNNER_CPU_PROFILE needs libprofiler preloaded");
        }

        // Every iteration is timed; the last one's artifact is the one the
        // numerical comparison below reads, so the answer that gets validated
        // is the answer the final measured run produced.
        std::vector<double> per_iteration_seconds;
        per_iteration_seconds.reserve(sampling.iterations);
        using RunResult = decltype(run_checked(std::string(), true));
        std::optional<RunResult> last;
        for (std::size_t index = 0; index < sampling.iterations; ++index)
        {
            last.emplace(run_checked(
                "online.iteration." + std::to_string(index + 1), true));
            per_iteration_seconds.push_back(
                static_cast<double>(
                    std::get<0>(*last).timing.online_execution_nanoseconds) * 1e-9);
        }

        if (cuda_profiler_range)
        {
            require_cuda(cudaProfilerStop(), "cudaProfilerStop");
        }
        if (cpu_profiler_stop != nullptr)
        {
            cpu_profiler_stop();
        }

        const auto &artifact = std::get<0>(*last);
        const std::vector<double> poseidon_output = std::get<1>(*last);
        const double max_imaginary = std::get<2>(*last);
        // The same input through the same plan must give the same answer on
        // every run of this process; a drift is state leaking between runs.
        const bool runs_consistent = largest_deviation <= kRunConsistencyTolerance;

        // ResNet20's encrypted logits land 0.07-0.10 from the clear reference
        // (CKKS noise through 466 bootstraps), so a 0.1 bound failed about one
        // run in ten for nothing that matters to the answer.  Its bound is 0.2,
        // and the predicted class must also be the reference's.
        // A slot fixture carries its own bounds: an absolute one on every slot
        // (prefix and zero tail alike) and none relative.
        const double python_absolute_tolerance =
            slot_fixture ? fixture.at("tolerance").at("absolute").get<double>()
            : model == "resnet20" ? 0.2 : kAbsoluteTolerance;
        const double relative_tolerance = slot_fixture ? 0.0 : kRelativeTolerance;
        const double imaginary_tolerance =
            slot_fixture ? fixture.at("tolerance").at("imaginary").get<double>()
                         : kImaginaryTolerance;
        const Comparison against_python = compare(
            python_output, poseidon_output, python_absolute_tolerance,
            relative_tolerance);
        const Comparison against_mock = compare(
            mock_output, poseidon_output, python_absolute_tolerance, relative_tolerance);
        // Per output, for a slot fixture: worst error on the valid prefix, on
        // the tail, and in the imaginary part is not separable from the flat
        // vectors above, so it is recomputed here (decode is not repeated).
        Json per_output = Json::array();
        if (slot_fixture)
        {
            for (std::size_t i = 0; i < plan.final_outputs.size(); ++i)
            {
                const std::size_t valid = fixture.at("expected").at(i).size();
                const std::size_t base = i * slot_count;
                double prefix = 0.0, tail = 0.0, versus_mock = 0.0;
                for (std::size_t s = 0; s < slot_count; ++s)
                {
                    const double error = std::abs(poseidon_output[base + s] - python_output[base + s]);
                    if (s < valid)
                        prefix = std::max(prefix, error);
                    else
                        tail = std::max(tail, error);
                    versus_mock = std::max(versus_mock,
                                           std::abs(poseidon_output[base + s] - mock_output[base + s]));
                }
                per_output.push_back({
                    {"name", fixture.contains("output_names")
                                 ? fixture.at("output_names").at(i) : Json(std::to_string(i))},
                    {"value_id", plan.final_outputs[i]},
                    {"valid", valid},
                    {"max_abs_error", prefix},
                    {"max_abs_tail", tail},
                    {"max_abs_vs_mock", versus_mock},
                    {"plan_scale_log2", find_value(plan, plan.final_outputs[i]).scale_log2},
                    {"ciphertext_scale_log2", std::log2(output_scales[i])},
                });
            }
            // Diagnostics: the decoded outputs of the last run, concatenated
            // in plan order, as raw little-endian float64.
            if (const char *dump = std::getenv("POSEIDON_GPU_MLP_OUTPUT_DUMP"); dump && *dump)
            {
                std::ofstream out(dump, std::ios::binary);
                out.write(reinterpret_cast<const char *>(poseidon_output.data()),
                          static_cast<std::streamsize>(poseidon_output.size() * sizeof(double)));
                if (!out)
                    throw std::runtime_error(std::string("cannot write ") + dump);
            }
        }
        const auto argmax = [](const std::vector<double> &values) {
            return static_cast<std::size_t>(std::distance(
                values.begin(), std::max_element(values.begin(), values.end())));
        };
        const std::size_t prediction = argmax(poseidon_output);
        const std::size_t reference_prediction = argmax(python_output);
        const bool prediction_matches =
            model != "resnet20" || prediction == reference_prediction;
        const bool require_python_match = frontend_packed || model != "resnet20";
        const bool passed =
            (!require_python_match || against_python.within_tolerance) &&
            against_mock.within_tolerance && prediction_matches &&
            max_imaginary <= imaginary_tolerance &&
            artifact.timing.boot_calls == plan_boots &&
            runs_consistent;
        const double key_seconds =
            std::chrono::duration<double>(key_finish - key_start).count();
        const double compute_seconds =
            artifact.timing.compute_including_boot_nanoseconds * 1e-9;
        const double boot_seconds = artifact.timing.boot_nanoseconds * 1e-9;
        const double setup_seconds = artifact.timing.setup_nanoseconds * 1e-9;
        const double initialization_seconds =
            artifact.timing.initialization_nanoseconds * 1e-9;
        const double online_execution_seconds =
            artifact.timing.online_execution_nanoseconds * 1e-9;
        const double communication_post_seconds =
            artifact.timing.communication_post_nanoseconds * 1e-9;
        const double communication_wait_seconds =
            artifact.timing.communication_wait_nanoseconds * 1e-9;
        // The median over the measured iterations, which is what adjudication
        // compares. With the default single iteration it is that one value, so
        // a caller that passes no sampling flags gets the same number as before.
        const double runtime_seconds = median_of(per_iteration_seconds);
        // What this run actually executed with, and where each value came
        // from.  ``applied`` holds the values in effect, not the file's text,
        // so a caller can compare it with what it meant to send.
        const std::string execution_mode_name =
            device_execution_mode == fhegpu::DeviceExecutionMode::PerDeviceDependencyWorkers
                ? "dependency_workers"
            : device_execution_mode == fhegpu::DeviceExecutionMode::PerDeviceReadyWorkers
                ? "ready_backfill"
            : device_execution_mode == fhegpu::DeviceExecutionMode::PerDeviceWorkers
                ? "per_device"
                : "sequential";
        const Json applied{
            {"runtime.device_execution_mode", execution_mode_name},
            {"runtime.device_worker_count", device_worker_count},
            {"devices[].cuda_device_id", cuda_device_ids},
        };
        const Json execution_config_json =
            execution_config
                ? Json{{"source", "file"},
                       {"path", *sampling.execution_config},
                       {"source_sha256", execution_config->source_sha256},
                       {"config_id", execution_config->config.config_id},
                       {"binding",
                        {{"plan_sha256", execution_config->config.binding.plan_sha256},
                         {"operator_spec_sha256",
                          execution_config->config.binding.operator_spec_sha256},
                         {"decision_record_sha256",
                          execution_config->config.binding.decision_record_sha256}}},
                       {"applied", applied},
                       {"ignored", fhegpu::ignored_fields()}}
                : Json{{"source", "environment"},
                       {"applied", applied},
                       {"ignored", nullptr}};
        const Json report{
            {"format_version", 1},
            {"passed", passed},
            {"model", model},
            {"native_nohoist", native_nohoist},
            {"input_encoding", fixture.value("input_encoding", "legacy")},
            {"require_python_match", require_python_match},
            {"cuda_devices", cuda_device_ids},
            {"execution_config", execution_config_json},
            {"device_execution_mode", execution_mode_name},
            {"device_workers", device_workers},
            {"device_worker_count", device_worker_count},
            {"device_backfill", device_backfill},
            {"device_dependencies", device_dependencies},
            // Always true now: asynchronous publication is unconditional in
            // this GPU Api.  The requested flag is reported beside it so a
            // caller that set it can see it was recognised and is redundant.
            {"transfer_events", true},
            {"transfer_events_requested", transfer_events_requested},
            {"cuda_profiler_range", cuda_profiler_range},
            {"warmups", sampling.warmups},
            {"iterations", sampling.iterations},
            {"online_timing", online_timing_json(per_iteration_seconds)},
            {"runs", runs_detail},
            {"runs_consistent", runs_consistent},
            {"largest_run_deviation", largest_deviation},
            {"seed", fixture.at("seed")},
            {"model_sha256", fixture.at("model_sha256")},
            {"plan_sha256", loaded_plan.source_sha256},
            {"operator_spec_sha256", loaded_spec.source_sha256},
            {"poly_degree", loaded_spec.spec.poly_degree},
            {"q_modulus_count", loaded_spec.spec.rns_moduli_log2.size()},
            {"rotation_key_policy", rotation_key_policy},
            {"rotation_key_count", rotation_key_steps.size()},
            {"logical_rotation_count", logical_rotation_steps.size()},
            {"allow_no_boot", allow_no_boot},
            {"transfer_count", count_transfers(plan)},
            {"key_generation_seconds", key_seconds},
            {"runtime_seconds", runtime_seconds},
            {"runtime_scope", "online_execution_only"},
            {"runtime_timing",
             {{"setup_seconds", setup_seconds},
              {"initialization_seconds", initialization_seconds},
              {"online_execution_seconds", online_execution_seconds},
              {"encode_calls", artifact.timing.encode_calls},
              {"encode_seconds", artifact.timing.encode_nanoseconds * 1e-9},
              {"bundle_read_calls", artifact.timing.bundle_read_calls},
              {"bundle_read_seconds", artifact.timing.bundle_read_nanoseconds * 1e-9},
              {"bundle_read_bytes", artifact.timing.bundle_read_bytes},
              {"bundle_resident_bytes", artifact.timing.bundle_resident_bytes},
              {"bundle_resident_load_seconds", artifact.timing.bundle_resident_load_nanoseconds * 1e-9},
              {"bundle_verify_seconds", artifact.timing.bundle_verify_nanoseconds * 1e-9},
              {"raw_peak_bytes", artifact.timing.raw_peak_bytes},
              {"fence_calls", artifact.timing.fence_calls},
              {"fence_seconds", artifact.timing.fence_nanoseconds * 1e-9},
              {"compute_calls", artifact.timing.compute_calls},
              {"boot_calls", artifact.timing.boot_calls},
              {"expanded_boots", plan_expanded_boots},
              {"boot_step_calls", artifact.timing.boot_step_calls},
              {"boot_step_seconds", artifact.timing.boot_step_nanoseconds * 1e-9},
              {"communication_post_calls",
               artifact.timing.communication_post_calls},
              {"communication_wait_calls",
               artifact.timing.communication_wait_calls},
              {"communication_post_seconds", communication_post_seconds},
              {"communication_wait_seconds", communication_wait_seconds},
              {"communication_total_seconds",
               communication_post_seconds + communication_wait_seconds},
              {"device_backfill_tasks", artifact.timing.device_backfill_tasks},
              {"device_ready_wait_calls",
               artifact.timing.device_ready_wait_calls},
              {"device_ready_wait_seconds",
               artifact.timing.device_ready_wait_nanoseconds * 1e-9},
              {"compute_including_boot_seconds", compute_seconds},
              {"boot_seconds", boot_seconds},
              {"compute_excluding_boot_seconds", compute_seconds - boot_seconds}}},
            // Whole process, initialization included: pinned buffers handed
            // out, and how many of them needed a new cudaMallocHost.
            {"pinned_staging",
             {{"buffers", poseidon::runtime_api::communication::PinnedHostBuffer::allocation_calls()},
              {"cuda_allocations", poseidon::runtime_api::communication::PinnedHostBuffer::cuda_allocation_calls()},
              {"reuses", poseidon::runtime_api::communication::PinnedHostBuffer::reuse_calls()},
              {"peak_live_bytes", poseidon::runtime_api::communication::PinnedHostBuffer::peak_bytes()},
              {"cached_bytes", poseidon::runtime_api::communication::PinnedHostBuffer::cached_bytes()}}},
            {"input_level", input_desc.level},
            {"input_scale_log2", input_desc.scale_log2},
            {"output_level", output_desc.level},
            {"output_scale_log2", output_desc.scale_log2},
            {"tolerances",
             {{"absolute", python_absolute_tolerance},
              {"relative", relative_tolerance},
              {"imaginary", imaginary_tolerance},
              {"python_absolute", python_absolute_tolerance}}},
            {"prediction", prediction},
            {"reference_prediction", reference_prediction},
            {"prediction_matches", prediction_matches},
            {"python_comparison", comparison_json(against_python)},
            {"mock_comparison", comparison_json(against_mock)},
            {"max_imaginary", max_imaginary},
            // A slot fixture's outputs are ~229k slots each way: per-output
            // summaries instead of the vectors.
            {"python_output", slot_fixture ? Json::array() : Json(python_output)},
            {"mock_output", slot_fixture ? Json::array() : Json(mock_output)},
            {"poseidon_output", slot_fixture ? Json::array() : Json(poseidon_output)},
            {"output_count", plan.final_outputs.size()},
            {"input_count", plan.external_inputs.size()},
            {"per_output", per_output},
        };
        write_json(argv[6], report);
        std::cout << (passed ? "PASS" : "FAIL")
                  << " boots=" << artifact.timing.boot_calls
                  << " transfers=" << count_transfers(plan)
                  << " execution_mode=" << execution_mode_name
                  << " device_workers=" << device_worker_count
                  << " config=" << (execution_config ? execution_config->source_sha256 : "none")
                  << " rotation_key_policy=" << rotation_key_policy
                  << " rotations=" << rotation_key_steps.size()
                  << " key_seconds=" << key_seconds
                  << " runtime_seconds=" << runtime_seconds
                  << " initialization_seconds=" << initialization_seconds
                  << " online_execution_seconds=" << online_execution_seconds
                  << " boot_seconds=" << boot_seconds
                  << " python_max_abs=" << against_python.max_abs
                  << " prediction=" << prediction << "/" << reference_prediction
                  << " max_imaginary=" << max_imaginary << '\n'
                  << "report=" << argv[6] << '\n';
        return passed ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
