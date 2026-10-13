#include "poseidon/basics/modulus.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/runtime_api/communication/cuda_local_transfer.h"
#include "poseidon/runtime_api/poseidon_gpu_api.h"
#include "poseidon/runtime_api/rotation_key_basis.h"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/runtime.hpp"
#include "runtime/utils/sha256.hpp"
#include "runtime_bundle_options.h"

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/resource.h>

namespace
{
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using poseidon::runtime_api::PoseidonGpuApi;
using poseidon::runtime_api::PoseidonGpuValue;

constexpr double kAbsoluteTolerance = 0.001;
constexpr double kRelativeTolerance = 0.0001;

double milliseconds(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void check_cuda(cudaError_t error, const char *operation)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

std::ofstream open_output(const std::filesystem::path &path)
{
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("cannot write output: " + path.string());
    return output;
}

void write_report(const std::filesystem::path &path, const Json &report)
{
    auto output = open_output(path);
    output << report.dump(2) << '\n';
    if (!output)
        throw std::runtime_error("failed writing report: " + path.string());
}

Json read_fixture(const std::filesystem::path &path, std::string &digest)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open fixture: " + path.string());
    std::ostringstream bytes;
    bytes << input.rdbuf();
    if (input.bad())
        throw std::runtime_error("failed reading fixture: " + path.string());
    const std::string text = bytes.str();
    digest = "sha256:" + fhegpu::sha256_hex(text);
    return Json::parse(text);
}

std::uint32_t exact_log2(std::uint64_t value)
{
    if (value < 2 || (value & (value - 1)) != 0)
        throw std::runtime_error("poly_degree must be a power of two");
    std::uint32_t result = 0;
    while (value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

poseidon::PoseidonContext make_context(const fhegpu::OperatorSpec &spec,
                                      const Json &configuration)
{
    const auto degree = configuration.at("poly_degree").get<std::uint64_t>();
    const auto q_bits = configuration.at("q_moduli_bits").get<std::vector<int>>();
    const auto p_bits = configuration.at("p_moduli_bits").get<std::vector<int>>();
    const int scale = configuration.at("default_scale_log2").get<int>();
    if (degree != spec.poly_degree || q_bits != spec.rns_moduli_log2 ||
        scale != spec.default_scale_log2)
        throw std::runtime_error("fixture CKKS degree, Q chain or default scale disagrees with operator spec");
    if (q_bits.empty() || p_bits.empty() || scale <= 0 || scale > 1023)
        throw std::runtime_error("fixture CKKS Q/P chains must be nonempty and scale must be positive");
    auto bits = q_bits;
    bits.insert(bits.end(), p_bits.begin(), p_bits.end());
    if (std::any_of(bits.begin(), bits.end(), [](int bit) { return bit < 2 || bit > 32; }))
        throw std::runtime_error("Poseidon GPU CKKS Q/P primes must fit the 32-bit GPU representation");
    const auto primes = poseidon::CoeffModulus::Create(degree, bits);
    const std::vector<poseidon::Modulus> q(primes.begin(), primes.begin() + q_bits.size());
    const std::vector<poseidon::Modulus> p(primes.begin() + q_bits.size(), primes.end());
    const auto log_n = exact_log2(degree);
    poseidon::ParametersLiteral parameters(
        CKKS, log_n, log_n - 1, static_cast<std::uint32_t>(scale), 0, 0,
        poseidon::Modulus(0), q, p, poseidon::sec_level_type::none);
    return poseidon::PoseidonContext(parameters);
}

std::unordered_map<fhegpu::ValueId, const fhegpu::ValueDesc *>
descriptors(const fhegpu::RuntimePlan &plan)
{
    std::unordered_map<fhegpu::ValueId, const fhegpu::ValueDesc *> result;
    for (const auto &value : plan.values)
        if (!result.emplace(value.id, &value).second)
            throw std::runtime_error("duplicate ValueDesc " + std::to_string(value.id));
    return result;
}

std::vector<double> row_values(const Json &row, fhegpu::ValueId id,
                               std::size_t slot_count)
{
    if (row.contains("value_id") && row.at("value_id").get<fhegpu::ValueId>() != id)
        throw std::runtime_error("fixture value_id does not match the plan's declared input/output order");
    if (!row.at("name").is_string() || row.at("name").get<std::string>().empty())
        throw std::runtime_error("fixture input/output name must be a nonempty string");
    const auto &numbers = row.at("values");
    if (!numbers.is_array() || numbers.empty() || numbers.size() > slot_count)
        throw std::runtime_error("fixture values must be a nonempty array of at most N/2 real slots");
    auto values = numbers.get<std::vector<double>>();
    if (std::any_of(values.begin(), values.end(), [](double value) { return !std::isfinite(value); }))
        throw std::runtime_error("fixture contains a non-finite value");
    return values;
}

fhegpu::PrepareBackend parse_backend(const std::string &name)
{
    if (name == "cpu") return fhegpu::PrepareBackend::Cpu;
    if (name == "gpu") return fhegpu::PrepareBackend::Gpu;
    if (name == "auto") return fhegpu::PrepareBackend::Auto;
    throw std::runtime_error("--backend must be cpu, gpu or auto");
}

std::string backend_name(fhegpu::PrepareBackend backend)
{
    switch (backend)
    {
    case fhegpu::PrepareBackend::Cpu: return "cpu";
    case fhegpu::PrepareBackend::Gpu: return "gpu";
    case fhegpu::PrepareBackend::Auto: return "auto";
    }
    throw std::runtime_error("unknown Prepare backend");
}

void configure_waterline(fhegpu::RuntimePlan &plan, const Json &fixture,
                         const std::optional<fhegpu::PrepareBackend> &backend)
{
    if (fixture.contains("waterline"))
    {
        const auto &configuration = fixture.at("waterline");
        fhegpu::WaterlineConfig waterline;
        waterline.device_capacity_bytes = configuration.at("device_capacity_bytes").get<std::vector<std::uint64_t>>();
        waterline.device_reserve_bytes = configuration.at("device_reserve_bytes").get<std::vector<std::uint64_t>>();
        waterline.device_fractions = configuration.at("device_fractions").get<std::vector<double>>();
        waterline.host_budget_bytes = configuration.at("host_budget_bytes").get<std::uint64_t>();
        waterline.pinned_budget_bytes = configuration.at("pinned_budget_bytes").get<std::uint64_t>();
        waterline.raw_budget_bytes = configuration.at("raw_budget_bytes").get<std::uint64_t>();
        waterline.lookahead_groups = configuration.at("lookahead_groups").get<std::size_t>();
        waterline.backend = configuration.contains("backend")
            ? parse_backend(configuration.at("backend").get<std::string>())
            : (plan.waterline ? plan.waterline->backend : fhegpu::PrepareBackend::Cpu);
        plan.waterline = std::move(waterline);
    }
    if (backend)
    {
        if (!plan.waterline)
            throw std::runtime_error("--backend requires Waterline in the plan or fixture");
        plan.waterline->backend = *backend;
    }
}

Json waterline_report(const fhegpu::RuntimePlan &plan)
{
    if (!plan.waterline) return nullptr;
    const auto &configuration = *plan.waterline;
    return {
        {"backend", backend_name(configuration.backend)},
        {"device_capacity_bytes", configuration.device_capacity_bytes},
        {"device_reserve_bytes", configuration.device_reserve_bytes},
        {"device_fractions", configuration.device_fractions},
        {"host_budget_bytes", configuration.host_budget_bytes},
        {"pinned_budget_bytes", configuration.pinned_budget_bytes},
        {"raw_budget_bytes", configuration.raw_budget_bytes},
        {"lookahead_groups", configuration.lookahead_groups},
    };
}

Json configure_pool_cap(const fhegpu::RuntimePlan &plan)
{
    if (!plan.waterline) return nullptr;
    constexpr std::uint64_t mib = 1ULL << 20;
    constexpr std::uint64_t outside_pool_bytes = 576 * mib;
    constexpr const char *environment = "POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB";
    const auto &capacities = plan.waterline->device_capacity_bytes;
    std::vector<std::uint64_t> maximum_mb, selected_mb;
    for (std::size_t device=0;device<capacities.size();++device)
    {
        check_cuda(cudaSetDevice(static_cast<int>(device)), "pool cap device");
        std::size_t free=0,total=0;
        check_cuda(cudaMemGetInfo(&free,&total), "pool cap free memory");
        const auto available=std::min(std::uint64_t(free),capacities[device]);
        if (available < outside_pool_bytes + 64 * mib)
            throw std::runtime_error("Waterline requires 640 MiB available per card for pool and CUDA objects");
        maximum_mb.push_back((available-outside_pool_bytes)/mib);
    }
    const char *existing = std::getenv(environment);
    if (existing)
    {
        const std::string text(existing);
        std::size_t begin=0;
        do {
            const auto end=text.find(',',begin);
            const auto part=text.substr(begin,end==std::string::npos?end:end-begin);
            std::uint64_t value=0;
            const auto parsed=std::from_chars(part.data(),part.data()+part.size(),value);
            if(parsed.ec!=std::errc{} || parsed.ptr!=part.data()+part.size() || value<64)
                throw std::runtime_error(std::string(environment)+" requires integer MiB caps >=64");
            selected_mb.push_back(value);
            if(end==std::string::npos)break;
            begin=end+1;
        } while(true);
        if(selected_mb.size()==1)selected_mb.resize(capacities.size(),selected_mb.front());
        if(selected_mb.size()!=capacities.size())
            throw std::runtime_error("GPU pool caps require one scalar or one cap for every visible GPU");
        for(std::size_t device=0;device<capacities.size();++device)
            if(selected_mb[device]>maximum_mb[device])
                throw std::runtime_error("GPU pool cap exceeds configured or physically available capacity on GPU"+std::to_string(device));
    }
    else selected_mb=maximum_mb;
    std::string caps;
    for(auto value:selected_mb){if(!caps.empty())caps+=',';caps+=std::to_string(value);}
    if(setenv(environment,caps.c_str(),1)!=0)
        throw std::runtime_error(std::string("cannot set ")+environment);
    std::cout << "GPU_POOL maximum_mb=" << caps << " source="
              << (existing ? "user_environment" : "waterline_and_free_memory")
              << " outside_pool_reserve_bytes=" << outside_pool_bytes << '\n' << std::flush;
    return {{"environment", environment}, {"maximum_mb", selected_mb},
            {"maximum_allowed_mb", maximum_mb},
            {"source", existing ? "user_environment" : "waterline_and_free_memory"},
            {"outside_pool_reserve_bytes", outside_pool_bytes}};
}

Json device_report(int device_count, const fhegpu::RuntimePlan &plan)
{
    Json devices = Json::array();
    for (int device = 0; device < device_count; ++device)
    {
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        if (plan.waterline && plan.waterline->device_capacity_bytes.at(device) > properties.totalGlobalMem)
            throw std::runtime_error("configured Waterline GPU capacity exceeds actual device capacity");
        char pci_bus[32]{};
        check_cuda(cudaDeviceGetPCIBusId(pci_bus, sizeof(pci_bus), device), "cudaDeviceGetPCIBusId");
        std::ostringstream uuid;
        uuid << "GPU-" << std::hex;
        constexpr char digits[] = "0123456789abcdef";
        for (int index = 0; index < 16; ++index)
        {
            if (index == 4 || index == 6 || index == 8 || index == 10) uuid << '-';
            const auto byte = static_cast<unsigned char>(properties.uuid.bytes[index]);
            uuid << digits[byte >> 4] << digits[byte & 15];
        }
        devices.push_back({{"logical_device", device}, {"name", properties.name},
                           {"uuid", uuid.str()}, {"pci_bus_id", pci_bus},
                           {"total_global_bytes", properties.totalGlobalMem}});
    }
    return devices;
}

std::size_t host_boot_count(const fhegpu::RuntimePlan &plan)
{
    std::size_t count = 0;
    const auto visit = [&count](const std::vector<fhegpu::Instruction> &instructions) {
        for (const auto &instruction : instructions)
        {
            const auto *compute = std::get_if<fhegpu::ComputeOp>(&instruction.body);
            if (!compute || compute->kind != fhegpu::ComputeKind::Boot) continue;
            const auto &attributes = std::get<fhegpu::BootAttrs>(compute->attrs);
            if (attributes.implementation != fhegpu::BootImplementation::DecryptReencrypt ||
                compute->place.kind != fhegpu::PlaceKind::Host)
                throw std::runtime_error("Qwen fixture runner supports only explicit Host decrypt-reencrypt Boot; native Bootstrap is not configured");
            ++count;
        }
    };
    visit(plan.initialization);
    visit(plan.execution);
    visit(plan.finalization);
    return count;
}

PoseidonGpuValue download(PoseidonGpuApi &api, fhegpu::ValueId id,
                          const fhegpu::ValueDesc &source,
                          const PoseidonGpuValue &value)
{
    if (source.place.kind == fhegpu::PlaceKind::Host) return value;
    fhegpu::CommAction action;
    action.kind = fhegpu::CommKind::Transfer;
    action.hint = fhegpu::CommHint::PointToPoint;
    action.inputs = {id};
    action.outputs = {id};
    action.sources = {source.place};
    action.destinations = {{fhegpu::PlaceKind::Host, 0, 0}};
    action.output_types = {fhegpu::ValueKind::Ciphertext};
    auto destination = source;
    destination.place = action.destinations.front();
    auto handle = api.communicate_async(action, {value}, {destination});
    auto results = api.wait(handle);
    if (results.size() != 1 || !results.front())
        throw std::runtime_error("final output download produced an incorrect output count");
    return std::move(*results.front());
}

struct Comparison
{
    std::size_t slots = 0, finite_slots = 0, bad_real = 0, bad_imaginary = 0, nonfinite = 0;
    std::size_t bad_prefix = 0, bad_tail = 0;
    double max_abs = 0, max_relative = 0, max_imaginary = 0, max_tail_abs = 0;
    long double square_error = 0;

    bool passed() const { return bad_real == 0 && bad_imaginary == 0 && nonfinite == 0; }

    Json report() const
    {
        return {{"passed", passed()}, {"slots_checked", slots}, {"finite_slots", finite_slots},
                {"max_abs_error", max_abs}, {"max_relative_error_floor_atol", max_relative},
                {"rms_error", finite_slots ? std::sqrt(static_cast<double>(square_error / finite_slots)) : 0.0},
                {"max_imaginary", max_imaginary}, {"max_tail_abs", max_tail_abs},
                {"bad_real_slots", bad_real}, {"bad_imaginary_slots", bad_imaginary},
                {"nonfinite_slots", nonfinite}, {"bad_prefix_slots", bad_prefix},
                {"bad_tail_slots", bad_tail}};
    }

    void append(const Comparison &other)
    {
        slots += other.slots; finite_slots += other.finite_slots;
        bad_real += other.bad_real; bad_imaginary += other.bad_imaginary;
        nonfinite += other.nonfinite; bad_prefix += other.bad_prefix; bad_tail += other.bad_tail;
        max_abs = std::max(max_abs, other.max_abs);
        max_relative = std::max(max_relative, other.max_relative);
        max_imaginary = std::max(max_imaginary, other.max_imaginary);
        max_tail_abs = std::max(max_tail_abs, other.max_tail_abs);
        square_error += other.square_error;
    }
};

Comparison compare(const std::vector<std::complex<double>> &actual,
                   const std::vector<double> &expected)
{
    Comparison result;
    result.slots = actual.size();
    for (std::size_t index = 0; index < actual.size(); ++index)
    {
        const auto value = actual[index];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        {
            ++result.nonfinite;
            if (index < expected.size()) ++result.bad_prefix; else ++result.bad_tail;
            continue;
        }
        ++result.finite_slots;
        const double reference = index < expected.size() ? expected[index] : 0.0;
        const double error = std::abs(value.real() - reference);
        const double imaginary = std::abs(value.imag());
        const bool bad_real = error > kAbsoluteTolerance + kRelativeTolerance * std::abs(reference);
        const bool bad_imaginary = imaginary > kAbsoluteTolerance;
        result.bad_real += bad_real;
        result.bad_imaginary += bad_imaginary;
        if (bad_real || bad_imaginary)
        {
            if (index < expected.size()) ++result.bad_prefix; else ++result.bad_tail;
        }
        result.max_abs = std::max(result.max_abs, error);
        result.max_relative = std::max(result.max_relative, error / std::max(std::abs(reference), kAbsoluteTolerance));
        result.max_imaginary = std::max(result.max_imaginary, imaginary);
        if (index >= expected.size()) result.max_tail_abs = std::max(result.max_tail_abs, std::abs(value.real()));
        result.square_error += static_cast<long double>(error) * error;
    }
    return result;
}

Json timing_report(const fhegpu::RuntimeTiming &timing)
{
    return {{"setup_ms", timing.setup_nanoseconds / 1e6},
            {"initialization_ms", timing.initialization_nanoseconds / 1e6},
            {"online_execution_ms", timing.online_execution_nanoseconds / 1e6},
            {"encode_calls", timing.encode_calls}, {"encode_submission_ms", timing.encode_nanoseconds / 1e6},
            {"bundle_read_calls", timing.bundle_read_calls}, {"bundle_read_bytes", timing.bundle_read_bytes},
            {"bundle_read_ms", timing.bundle_read_nanoseconds / 1e6},
            {"raw_peak_bytes", timing.raw_peak_bytes}, {"bundle_resident_bytes", timing.bundle_resident_bytes},
            {"compute_calls", timing.compute_calls}, {"compute_submission_including_boot_ms", timing.compute_including_boot_nanoseconds / 1e6},
            {"boot_calls", timing.boot_calls}, {"host_boot_ms", timing.boot_nanoseconds / 1e6},
            {"communication_post_calls", timing.communication_post_calls},
            {"communication_wait_calls", timing.communication_wait_calls},
            {"communication_post_ms", timing.communication_post_nanoseconds / 1e6},
            {"communication_wait_ms", timing.communication_wait_nanoseconds / 1e6},
            {"fence_calls", timing.fence_calls}};
}
} // namespace

int main(int argc, char **argv)
{
    const auto startup = Clock::now();
    if (argc < 6)
    {
        std::cerr << "usage: poseidon_runtime_gpu_qwen_e2e PLAN SPEC FIXTURE REPORT DECODED "
                     "[--bundle DIR] [--backend cpu|gpu|auto] [--trace CSV]\n";
        return 2;
    }
    Json report{{"format_version", 1}, {"passed", false}};
    std::string stage = "arguments";
    try
    {
        std::optional<std::filesystem::path> bundle;
        std::optional<fhegpu::PrepareBackend> backend;
        for (int argument = 6; argument < argc; argument += 2)
        {
            if (argument + 1 >= argc) throw std::runtime_error("option requires a value");
            const std::string option = argv[argument];
            if (option == "--bundle") bundle = argv[argument + 1];
            else if (option == "--backend") backend = parse_backend(argv[argument + 1]);
            else if (option == "--trace")
            {
                const std::filesystem::path trace = argv[argument + 1];
                if (!trace.parent_path().empty()) std::filesystem::create_directories(trace.parent_path());
                if (setenv("POSEIDON_RUNTIME_TRACE", trace.c_str(), 1) != 0)
                    throw std::runtime_error("cannot set POSEIDON_RUNTIME_TRACE");
                report["trace_path"] = trace.string();
            }
            else throw std::runtime_error("unknown option: " + option);
        }
        stage = "load_artifacts";
        std::string fixture_digest;
        const auto fixture = read_fixture(argv[3], fixture_digest);
        if (fixture.at("format_version").get<int>() != 1)
            throw std::runtime_error("unsupported Qwen fixture format_version");
        if (fixture.at("tolerances").at("atol").get<double>() != kAbsoluteTolerance ||
            fixture.at("tolerances").at("rtol").get<double>() != kRelativeTolerance)
            throw std::runtime_error("Qwen fixture tolerances must be fixed at atol=0.001, rtol=0.0001");
        auto loaded_plan = fhegpu::RuntimePlanReader::read_file(argv[1]);
        const auto loaded_spec = fhegpu::OperatorSpecReader::read_file(argv[2]);
        auto &plan = loaded_plan.plan;
        if (plan.target.world_size != 1 || plan.target.device_counts.size() != 1 ||
            plan.target.device_counts.front() <= 0)
            throw std::runtime_error("Qwen fixture runner requires one rank and one or more GPUs");
        const int device_count = plan.target.device_counts.front();
        configure_waterline(plan, fixture, backend);
        const auto requirements = fhegpu::PlanVerifier::verify(plan, loaded_spec, false);
        const auto host_boots = host_boot_count(plan);
        const auto descriptions = descriptors(plan);
        const auto slot_count = static_cast<std::size_t>(loaded_spec.spec.poly_degree / 2);
        const auto &input_rows = fixture.at("inputs");
        const auto &output_rows = fixture.at("expected_outputs");
        if (!input_rows.is_array() || input_rows.size() != plan.external_inputs.size() ||
            !output_rows.is_array() || output_rows.size() != plan.final_outputs.size() ||
            input_rows.empty() || output_rows.empty())
            throw std::runtime_error("fixture input/output counts disagree with the complete plan");
        std::set<std::string> names;
        for (std::size_t index = 0; index < input_rows.size(); ++index)
        {
            row_values(input_rows[index], plan.external_inputs[index], slot_count);
            if (!names.insert(input_rows[index].at("name").get<std::string>()).second)
                throw std::runtime_error("duplicate fixture input/output name");
        }
        for (std::size_t index = 0; index < output_rows.size(); ++index)
        {
            row_values(output_rows[index], plan.final_outputs[index], slot_count);
            if (!names.insert(output_rows[index].at("name").get<std::string>()).second)
                throw std::runtime_error("duplicate fixture input/output name");
            if (descriptions.at(plan.final_outputs[index])->kind != fhegpu::ValueKind::Ciphertext)
                throw std::runtime_error("all Qwen fixture final outputs must be ciphertexts");
        }
        report.update({{"fixture_sha256", fixture_digest}, {"plan_source_sha256", loaded_plan.source_sha256},
                       {"operator_spec_sha256", loaded_spec.source_sha256}, {"model", fixture.at("model")},
                       {"context", fixture.at("context")}, {"tolerances", fixture.at("tolerances")},
                       {"waterline", waterline_report(plan)}, {"input_count", input_rows.size()},
                       {"output_count", output_rows.size()}, {"slots_per_output", slot_count},
                       {"execution_mode", "per_device_workers"}, {"device_worker_count", device_count},
                       {"boot_implementation", host_boots ? "host_decrypt_reencrypt" : "none"},
                       {"host_boot_instructions", host_boots}, {"native_boot_instructions", 0},
                       {"secret_key_assisted_refresh", host_boots != 0},
                       {"security_level", "none; numerical validation context"}});
        int visible_device_count = 0;
        check_cuda(cudaGetDeviceCount(&visible_device_count), "cudaGetDeviceCount");
        if (visible_device_count != device_count)
            throw std::runtime_error("plan GPU count must equal all visible CUDA devices; single GPU runs must mask to the selected physical GPU");
        report["devices"] = device_report(device_count, plan);
        const char *visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
        report["cuda_visible_devices"] = visible_devices ? Json(visible_devices) : Json(nullptr);
        report["load_artifacts_ms"] = milliseconds(startup, Clock::now());
        std::cout << "QWEN inputs=" << input_rows.size() << " outputs=" << output_rows.size()
                  << " devices=" << device_count << " slots=" << slot_count
                  << " backend=" << (plan.waterline ? backend_name(plan.waterline->backend) : "explicit")
                  << " host_boot_instructions=" << host_boots << '\n';
        if (host_boots)
            std::cout << "BOOT semantics=Host decrypt-reencrypt uses_secret_key=true native_bootstrap=false\n";
        std::cout << std::flush;

        stage = "create_context";
        const auto context_begin = Clock::now();
        auto context = make_context(loaded_spec.spec, fixture.at("context"));
        report["context_ms"] = milliseconds(context_begin, Clock::now());
        std::vector<std::uint64_t> q_primes, p_primes;
        for (const auto &prime : context.parameters_literal()->q()) q_primes.push_back(prime.value());
        for (const auto &prime : context.parameters_literal()->p()) p_primes.push_back(prime.value());
        report["actual_q_moduli"] = q_primes;
        report["actual_p_moduli"] = p_primes;
        stage = "key_generation";
        const auto keys_begin = Clock::now();
        poseidon::KeyGenerator keys(context);
        auto secret_key = std::make_shared<poseidon::SecretKey>(keys.secret_key());
        auto public_key = std::make_shared<poseidon::PublicKey>();
        keys.create_public_key(*public_key);
        auto relin_keys = std::make_shared<poseidon::RelinKeys>();
        auto galois_keys = std::make_shared<poseidon::GaloisKeys>();
        bool needs_relin = false;
        std::set<int> rotation_steps;
        for (const auto &key : requirements.keys)
        {
            if (key.kind == fhegpu::KeyKind::Relin) needs_relin = true;
            if (key.kind == fhegpu::KeyKind::Galois)
            {
                if (!key.rotation_step) throw std::runtime_error("Galois key requirement lacks rotation_step");
                rotation_steps.insert(*key.rotation_step);
            }
        }
        if (needs_relin) keys.create_relin_keys(*relin_keys);
        const auto basis = poseidon::runtime_api::binary_rotation_key_basis(rotation_steps, context.parameters_literal()->slot());
        if (!basis.empty()) keys.create_galois_keys(std::vector<int>(basis.begin(), basis.end()), *galois_keys);
        report["rotation_key_basis"] = basis;
        report["key_generation_ms"] = milliseconds(keys_begin, Clock::now());
        std::cout << "KEYS ready rotation_basis=" << basis.size() << " key_generation_ms="
                  << report["key_generation_ms"] << '\n' << std::flush;

        stage = "encrypt_inputs";
        const auto input_begin = Clock::now();
        poseidon::CKKSEncoder encoder(context);
        poseidon::Encryptor encryptor(context, *public_key);
        poseidon::Decryptor decryptor(context, *secret_key);
        std::unordered_map<fhegpu::ValueId, PoseidonGpuValue> inputs;
        Json input_descriptions = Json::array();
        for (std::size_t index = 0; index < input_rows.size(); ++index)
        {
            const auto id = plan.external_inputs[index];
            const auto &description = *descriptions.at(id);
            if (description.kind != fhegpu::ValueKind::Ciphertext ||
                description.place.kind != fhegpu::PlaceKind::Host || description.place.rank != 0)
                throw std::runtime_error("Qwen fixture external inputs must be rank-0 Host ciphertexts");
            poseidon::Plaintext plaintext;
            encoder.encode(row_values(input_rows[index], id, slot_count),
                           context.crt_context()->parms_id_map().at(static_cast<std::uint32_t>(description.level)),
                           std::ldexp(1.0, description.scale_log2), plaintext);
            poseidon::Ciphertext ciphertext;
            encryptor.encrypt(plaintext, ciphertext);
            inputs.emplace(id, PoseidonGpuValue::from_host_ciphertext(std::move(ciphertext)));
            input_descriptions.push_back({{"name", input_rows[index].at("name")}, {"value_id", id},
                                          {"level", description.level}, {"scale_log2", description.scale_log2},
                                          {"prefix_size", input_rows[index].at("values").size()}});
        }
        report["inputs"] = std::move(input_descriptions);
        report["input_encryption_ms"] = milliseconds(input_begin, Clock::now());
        stage = "create_gpu_api";
        const auto api_begin = Clock::now();
        report["gpu_pool_cap"] = configure_pool_cap(plan);
        std::vector<int> cuda_devices(device_count);
        std::iota(cuda_devices.begin(), cuda_devices.end(), 0);
        PoseidonGpuApi api(loaded_spec.spec.context_id, context, cuda_devices, relin_keys, galois_keys,
                           host_boots ? public_key : nullptr, host_boots ? secret_key : nullptr);
        report["gpu_api_setup_ms"] = milliseconds(api_begin, Clock::now());
        fhegpu::SequentialRuntime<PoseidonGpuApi> runtime(
            0, 1, device_count, api, fhegpu::DeviceExecutionMode::PerDeviceWorkers, device_count);
        const fhegpu::RuntimeResources resources{loaded_spec, bundle, false, runtime_bundle_options_from_env()};
        stage = "execute_complete_plan";
        const auto runtime_begin = Clock::now();
        std::cout << "FORWARD begin\n" << std::flush;
        const auto artifact = runtime.run(loaded_plan, resources, inputs);
        report["runtime_ms"] = milliseconds(runtime_begin, Clock::now());
        report["runtime_timing"] = timing_report(artifact.timing);
        const auto measured=api.waterline_measurements();
        report["preparation_measurements"]={{"cpu_prepare_calls",measured.cpu_prepare_calls},
            {"cpu_encode_ms",measured.cpu_encode_nanoseconds/1e6},{"cpu_plain_h2d_bytes",measured.cpu_plain_h2d_bytes},
            {"transport_h2d_calls",measured.transport_h2d_calls},{"transport_h2d_bytes",measured.transport_h2d_bytes},
            {"transport_h2d_ms",measured.transport_h2d_nanoseconds/1e6},
            {"transport_scope","Host cipher and CPU plaintext uploads; excludes direct key uploader and direct GPU raw upload"}};
        report["device_measurements"]=Json::array();
        for(std::size_t d=0;d<measured.devices.size();++d) {
            const auto &m=measured.devices[d];
            report["device_measurements"].push_back({{"logical_device",d},{"managed_pool",m.managed_pool},
                {"allocated_live_bytes",m.allocated_live},{"allocated_peak_bytes",m.allocated_peak},
                {"pool_reserved_bytes",m.pool_reserved},{"pool_capacity_bytes",m.pool_capacity},
                {"consumer_wait_calls",m.consumer_wait_calls},{"consumer_wait_ms",m.consumer_wait_nanoseconds/1e6},
                {"gpu_encode_calls",m.gpu_encode_calls},{"gpu_raw_h2d_bytes",m.gpu_raw_h2d_bytes},
                {"gpu_raw_h2d_ms",m.gpu_raw_h2d_nanoseconds/1e6},{"gpu_encode_ms",m.gpu_encode_nanoseconds/1e6}});
        }
        using poseidon::runtime_api::communication::PinnedHostBuffer;
        report["pinned_transfer_peak_after_plan_bytes"] = PinnedHostBuffer::peak_bytes();
        report["pinned_transfer_live_after_plan_bytes"] = PinnedHostBuffer::live_bytes();
        std::cout << "FORWARD complete runtime_ms=" << report["runtime_ms"] << '\n' << std::flush;

        stage = "decrypt_and_validate_all_outputs";
        const auto validation_begin = Clock::now();
        auto decoded_output = open_output(argv[5]);
        decoded_output << "{\"format_version\":1,\"fixture_sha256\":" << Json(fixture_digest).dump()
                       << ",\"slots_per_output\":" << slot_count << ",\"outputs\":[\n";
        Comparison total;
        Json output_reports = Json::array();
        for (std::size_t index = 0; index < output_rows.size(); ++index)
        {
            const auto id = plan.final_outputs[index];
            const auto &description = *descriptions.at(id);
            const auto &value = artifact.values.at(id).value;
            auto host_value = download(api, id, description, value);
            poseidon::Plaintext plaintext;
            decryptor.decrypt(host_value.host_ciphertext(), plaintext);
            std::vector<std::complex<double>> actual;
            encoder.decode(plaintext, actual);
            if (actual.size() != slot_count) throw std::runtime_error("decoder returned an incorrect slot count");
            if (index == 0) report["main_entry_to_first_output_ms"] = milliseconds(startup, Clock::now());
            const auto expected = row_values(output_rows[index], id, slot_count);
            const auto comparison = compare(actual, expected);
            total.append(comparison);
            auto output_report = comparison.report();
            output_report.update({{"name", output_rows[index].at("name")}, {"value_id", id},
                                  {"prefix_size", expected.size()}, {"level", description.level},
                                  {"scale_log2", description.scale_log2}});
            output_reports.push_back(output_report);
            std::vector<double> real, imaginary;
            real.reserve(slot_count); imaginary.reserve(slot_count);
            for (const auto value : actual) { real.push_back(value.real()); imaginary.push_back(value.imag()); }
            const Json decoded_row{{"name", output_rows[index].at("name")}, {"value_id", id},
                                   {"prefix_size", expected.size()}, {"real", real}, {"imaginary", imaginary}};
            if (index) decoded_output << ",\n";
            decoded_output << decoded_row.dump();
            if (!decoded_output) throw std::runtime_error("failed writing full decoded outputs");
            std::cout << (comparison.passed() ? "PASS" : "FAIL") << " output=" << index
                      << " name=" << output_rows[index].at("name").get<std::string>()
                      << " max_abs=" << comparison.max_abs << " max_relative=" << comparison.max_relative
                      << " max_imaginary=" << comparison.max_imaginary
                      << " bad_real=" << comparison.bad_real << " bad_imaginary=" << comparison.bad_imaginary
                      << " nonfinite=" << comparison.nonfinite << '\n' << std::flush;
        }
        decoded_output << "\n]}\n";
        decoded_output.close();
        if (!decoded_output) throw std::runtime_error("failed closing decoded output file");
        report["output_validation_and_save_ms"] = milliseconds(validation_begin, Clock::now());
        report["main_entry_to_all_outputs_ms"] = milliseconds(startup, Clock::now());
        report["outputs"] = std::move(output_reports);
        report["comparison"] = total.report();
        report["decoded_path"] = argv[5];
        report["passed"] = total.passed();
        if(total.passed() && report["model"].value("complete_qwen",false))
            report["model"]["full_model_ckks_validated"]=true;
        api.drain();
        report["pinned_transfer_peak_including_validation_bytes"] = PinnedHostBuffer::peak_bytes();
        report["pinned_transfer_live_after_validation_bytes"] = PinnedHostBuffer::live_bytes();
        struct rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0) report["host_max_rss_kib"] = usage.ru_maxrss;
        report["time_scope"] = "steady-clock from main entry; dynamic-loader/process time requires external /usr/bin/time";
        stage = "complete";
        report["stage"] = stage;
        write_report(argv[4], report);
        std::cout << (total.passed() ? "PASS" : "FAIL") << " ALL_OUTPUTS count=" << output_rows.size()
                  << " slots=" << total.slots << " max_abs=" << total.max_abs
                  << " max_relative=" << total.max_relative << " max_imaginary=" << total.max_imaginary
                  << " bad_real=" << total.bad_real << " bad_imaginary=" << total.bad_imaginary
                  << " nonfinite=" << total.nonfinite << '\n'
                  << "report=" << argv[4] << " decoded=" << argv[5] << '\n';
        return total.passed() ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        report["stage"] = stage;
        report["error"] = error.what();
        report["main_entry_to_failure_ms"] = milliseconds(startup, Clock::now());
        std::cerr << "FAIL stage=" << stage << " " << error.what() << '\n';
        try { write_report(argv[4], report); }
        catch (const std::exception &report_error) { std::cerr << "report error: " << report_error.what() << '\n'; }
        return 1;
    }
}
