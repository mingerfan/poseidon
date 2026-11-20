#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/runtime_api/poseidon_cpu_api.h"
#include "poseidon/runtime_api/rotation_key_basis.h"
#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/runtime.hpp"
#include "runtime/verifier.hpp"

#include <nlohmann/json.hpp>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef PT_R12_APPROVED_DSO
#error "The approved DSO path must be supplied by the R12 CMake target"
#endif

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using poseidon::runtime_api::PoseidonCpuApi;
using poseidon::runtime_api::PoseidonCpuValue;
constexpr std::size_t kSlots = 16384;
constexpr const char *kContextId = "poseidon-agent-n15-q13-p1-tc128-v1";
constexpr double kAtol = 1e-5;
constexpr double kRtol = 1e-4;

class ContractError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
class NumericalFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void require(bool condition, const std::string &reason) {
    if (!condition) throw ContractError(reason);
}

std::uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<
        std::chrono::nanoseconds>(Clock::now() - start).count());
}

template<class Function>
auto timed(Json &times, const std::string &name, Function &&function)
    -> std::invoke_result_t<Function> {
    const auto start = Clock::now();
    const auto finish = [&] {
        times[name] = times.value(name, std::uint64_t{0}) + elapsed_ns(start);
    };
    if constexpr (std::is_void_v<std::invoke_result_t<Function>>) {
        try {
            function();
        } catch (...) {
            finish();
            throw;
        }
        finish();
    } else {
        auto result = [&]() -> std::invoke_result_t<Function> {
            try {
                return function();
            } catch (...) {
                finish();
                throw;
            }
        }();
        finish();
        return result;
    }
}

void write_json(const fs::path &path, const Json &value) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    fs::path temporary = path;
    temporary += ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open result: " + temporary.string());
    output << value.dump(2) << '\n';
    output.close();
    if (!output) throw std::runtime_error("cannot write result: " + temporary.string());
    fs::rename(temporary, path);
}

Json read_json(const fs::path &path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open JSON: " + path.string());
    Json value;
    input >> value;
    input >> std::ws;
    if (!input.eof()) throw std::runtime_error("trailing JSON data: " + path.string());
    return value;
}

struct Options {
    fs::path plan, spec, inputs, output;
    bool check_only = false;
};

Options parse_options(int argc, char **argv) {
    Options options;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (!seen.insert(name).second) throw std::runtime_error("duplicate argument: " + name);
        if (name == "--check-only") {
            options.check_only = true;
            continue;
        }
        fs::path *target = nullptr;
        if (name == "--plan") target = &options.plan;
        else if (name == "--spec") target = &options.spec;
        else if (name == "--inputs") target = &options.inputs;
        else if (name == "--output") target = &options.output;
        else throw std::runtime_error("unknown argument: " + name);
        if (++i == argc) throw std::runtime_error("missing value for " + name);
        *target = argv[i];
    }
    if (options.plan.empty() || options.spec.empty() || options.inputs.empty() || options.output.empty())
        throw std::runtime_error("required: --plan --spec --inputs --output [--check-only]");
    return options;
}

const std::vector<std::uint64_t> &fixed_primes() {
    static const std::vector<std::uint64_t> primes{
        1152921504581419009ULL, 1152921504581877761ULL, 1152921504583647233ULL,
        1152921504585547777ULL, 1152921504586530817ULL, 1152921504589938689ULL,
        1152921504592429057ULL, 1152921504592822273ULL, 1152921504593412097ULL,
        1152921504595640321ULL, 1152921504595968001ULL, 1152921504597016577ULL,
        1152921504598720513ULL, 1152921504606584833ULL};
    return primes;
}

poseidon::PoseidonContext fixed_context() {
    std::vector<poseidon::Modulus> q, p;
    const auto &primes = fixed_primes();
    for (std::size_t i = 0; i < primes.size(); ++i)
        (i < 13 ? q : p).emplace_back(primes[i]);
    return poseidon::PoseidonContext(poseidon::ParametersLiteral(
        ::CKKS, 15, 14, 45, 0, 0, poseidon::Modulus(0), q, p,
        poseidon::sec_level_type::tc128));
}

Json context_parameters(const poseidon::PoseidonContext &context) {
    const auto parameters = context.parameters_literal();
    Json q = Json::array(), p = Json::array();
    for (const auto &modulus : parameters->q()) q.push_back(modulus.value());
    for (const auto &modulus : parameters->p()) p.push_back(modulus.value());
    return {{"context_id", kContextId}, {"scheme", "CKKS"},
            {"poly_degree", parameters->degree()}, {"slots", parameters->slot()},
            {"scale_log2", parameters->log_scale()},
            {"hamming_weight", parameters->hamming_weight()},
            {"q0_level", parameters->q0_level()}, {"security", "tc128"},
            {"q", q}, {"p", p}};
}

std::string loaded_library_path() {
    dlerror();
    void *symbol = dlsym(RTLD_DEFAULT,
        "_ZN8poseidon29UniformRandomGeneratorFactory14DefaultFactoryEv");
    const char *error = dlerror();
    require(error == nullptr && symbol != nullptr, "approved Poseidon DSO symbol is unavailable");
    Dl_info info{};
    require(dladdr(symbol, &info) != 0 && info.dli_fname != nullptr,
            "cannot locate the loaded Poseidon DSO");
    try {
        const auto actual = fs::canonical(info.dli_fname);
        const auto approved = fs::canonical(PT_R12_APPROVED_DSO);
        require(actual == approved, "loaded Poseidon DSO path differs from approved path");
        return actual.string();
    } catch (const fs::filesystem_error &exception) {
        throw ContractError(std::string("cannot resolve Poseidon DSO path: ") + exception.what());
    }
}

void validate_source_path(const Json &source, const char *name) {
    require(source.contains(name) && source[name].is_string(),
            std::string("sample source lacks ") + name);
    const fs::path path = source[name].get<std::string>();
    require(path.is_absolute(), std::string("sample source must be absolute: ") + name);
    require(fs::canonical(path) == path,
            std::string("sample source must be canonical: ") + name);
}

void validate_samples(const Json &document) {
    require(document.is_object() && document.contains("format") &&
            document["format"] == "pt-agent-r12-mlp-samples-v1",
            "unsupported sample protocol");
    require(document.contains("source") && document["source"].is_object(), "sample source is missing");
    const auto &source = document["source"];
    require(source.value("split", "") == "train" &&
            source.value("normalization", "") == "float64(uint8)/255", "sample source policy differs");
    validate_source_path(source, "model");
    validate_source_path(source, "mnist");
    require(document.contains("samples") && document["samples"].is_array() &&
            document["samples"].size() == 3, "exactly three samples are required");
    for (std::size_t i = 0; i < 3; ++i) {
        const auto &sample = document["samples"][i];
        require(sample.is_object() && sample.contains("index") && sample["index"].is_number_integer() &&
                sample["index"] == i, "sample indices must be 0, 1, 2");
        for (const auto &field : {std::pair<const char *, std::size_t>{"input784", 784},
                                  {"reference10", 10}}) {
            require(sample.contains(field.first) && sample[field.first].is_array() &&
                    sample[field.first].size() == field.second, std::string("sample shape: ") + field.first);
            for (const auto &value : sample[field.first]) {
                require(value.is_number(), std::string("sample must be numeric: ") + field.first);
                const double number = value.get<double>();
                require(std::isfinite(number), std::string("sample must be finite: ") + field.first);
                if (field.second == 784) require(number >= 0.0 && number <= 1.0, "input value outside [0,1]");
            }
        }
    }
}

const fhegpu::ValueDesc &description(const fhegpu::RuntimePlan &plan, fhegpu::ValueId id) {
    const auto found = std::find_if(plan.values.begin(), plan.values.end(),
        [&](const auto &value) { return value.id == id; });
    require(found != plan.values.end(), "plan ValueDesc is missing");
    return *found;
}

void validate_contract(const fhegpu::RuntimePlan &plan, const fhegpu::OperatorSpec &spec) {
    require(plan.format_version == 1, "R12 requires RuntimePlan format 1");
    require(!plan.plaintext_bundle, "R12 forbids plaintext_bundle");
    require(plan.target.target_id == "poseidon-ckks-cpu" && plan.target.capability_version == 1 &&
            plan.target.world_size == 1 && plan.target.device_counts == std::vector<int>{0}, "R12 requires one Host rank");
    require(plan.external_inputs.size() == 1 && plan.final_outputs.size() == 1, "R12 requires one input and one output");
    require(spec.context_id == kContextId && spec.target_id == "poseidon-ckks-cpu" &&
            spec.id == "poseidon-agent-cpu-v1" && spec.version == 1 &&
            spec.poly_degree == 32768 && spec.rns_moduli_log2 == std::vector<int>(13, 60) &&
            spec.max_modulus_log2 == 60 && spec.default_scale_log2 == 45 &&
            spec.level_lower_bound == 0 && spec.level_upper_bound == 12,
            "OperatorSpec differs from fixed CPU parameters");
    for (const auto &value : plan.values) {
        require(value.place.kind == fhegpu::PlaceKind::Host && value.place.rank == 0 && value.place.index == 0,
                "all values must use Host rank 0");
        require(value.context == kContextId && value.level >= 0 && value.level <= 12 &&
                value.scale_log2 >= 0 && value.scale_log2 <= 1023 && value.ntt,
                "unsupported value context, level, scale or NTT metadata");
    }
    for (const auto id : {plan.external_inputs.front(), plan.final_outputs.front()})
        require(description(plan, id).kind == fhegpu::ValueKind::Ciphertext &&
                description(plan, id).components == 2, "I/O must be size-2 ciphertext");
    const auto check_phase = [&](const std::vector<fhegpu::Instruction> &phase) {
        for (const auto &instruction : phase) {
            if (const auto *encode = std::get_if<fhegpu::EncodeOp>(&instruction.body)) {
                const auto *payload = std::get_if<fhegpu::InlineEncodePayload>(&encode->payload);
                require(payload != nullptr && !payload->values.empty() && payload->values.size() <= kSlots,
                        "all Encode payloads must be nonempty inline vectors within 16384 slots");
                for (const double value : payload->values) require(std::isfinite(value), "nonfinite inline payload");
            } else if (const auto *op = std::get_if<fhegpu::ComputeOp>(&instruction.body)) {
                require(op->place.kind == fhegpu::PlaceKind::Host && op->place.rank == 0 && op->place.index == 0,
                        "compute must use Host rank 0");
                require(op->kind != fhegpu::ComputeKind::Boot, "bootstrap is unavailable in R12");
            } else {
                throw ContractError("R12 format-1 plan requires local Encode/Compute instructions");
            }
        }
    };
    check_phase(plan.initialization);
    check_phase(plan.execution);
    check_phase(plan.finalization);
}

void validate_requirements(const fhegpu::PlanRequirements &requirements) {
    for (const auto capability : requirements.capabilities)
        require(capability == fhegpu::RequiredCapability::Encode ||
                capability == fhegpu::RequiredCapability::HostCompute, "unsupported runtime capability");
    for (const auto &key : requirements.keys) {
        require(key.kind != fhegpu::KeyKind::Secret, "secret-key operations are forbidden");
        require(key.place.kind == fhegpu::PlaceKind::Host && key.place.rank == 0 && key.place.index == 0,
                "evaluation key requires local Host");
        if (key.kind == fhegpu::KeyKind::Galois)
            require(key.rotation_step.has_value(), "rotation key lacks its step");
    }
}

class TimedRepeatingCpuApi final : public PoseidonCpuApi {
public:
    using Base = PoseidonCpuApi;
    using Value = Base::Value;
    using Base::Base;

    void begin_sample(std::size_t index, const fhegpu::RuntimePlan &plan, std::ofstream &output) {
        sample_index_ = index;
        call_index_ = 0;
        plan_ = &plan;
        output_ = &output;
        calls = Json::array();
        compute_ns = 0;
        constant_encode_ns = 0;
        encode_calls = 0;
    }

    Value encode_plaintext(const fhegpu::ValueDesc &output_desc, const std::vector<double> &slots) {
        require(!slots.empty() && slots.size() <= kSlots, "invalid inline period");
        std::vector<double> repeated(kSlots);
        for (std::size_t i = 0; i < repeated.size(); ++i) repeated[i] = slots[i % slots.size()];
        const auto start = Clock::now();
        try {
            auto result = Base::encode_plaintext(output_desc, repeated);
            constant_encode_ns += elapsed_ns(start);
            ++encode_calls;
            return result;
        } catch (...) {
            constant_encode_ns += elapsed_ns(start);
            throw;
        }
    }

    Value compute(const fhegpu::ComputeOp &op, const std::vector<Value> &inputs) {
        const Json key = make_key(op, "compute");
        const auto start = Clock::now();
        auto result = [&]() -> Value {
            try {
                return Base::compute(op, inputs);
            } catch (...) {
                record(key, elapsed_ns(start), "error");
                throw;
            }
        }();
        record(key, elapsed_ns(start), "ok");
        return result;
    }

    Value compute_reuse(const fhegpu::ComputeOp &op, Value input, const std::vector<Value> &other_inputs) {
        const Json key = make_key(op, "compute_reuse");
        const auto start = Clock::now();
        auto result = [&]() -> Value {
            try {
                return Base::compute_reuse(op, std::move(input), other_inputs);
            } catch (...) {
                record(key, elapsed_ns(start), "error");
                throw;
            }
        }();
        record(key, elapsed_ns(start), "ok");
        return result;
    }

    Json calls = Json::array();
    std::uint64_t compute_ns = 0, constant_encode_ns = 0;
    std::size_t encode_calls = 0;

private:
    Json make_key(const fhegpu::ComputeOp &op, const char *mode) const {
        Json levels = Json::array(), scales = Json::array();
        for (const auto id : op.inputs) {
            const auto &input = description(*plan_, id);
            levels.push_back(input.level);
            scales.push_back(input.scale_log2);
        }
        const auto &output = description(*plan_, op.output);
        Json rotation = nullptr;
        if (op.kind == fhegpu::ComputeKind::Rotate) rotation = std::get<fhegpu::RotateAttrs>(op.attrs).steps;
        return {{"op", fhegpu::to_string(op.kind)}, {"mode", mode}, {"input_levels", levels},
                {"input_scale_log2", scales}, {"output_level", output.level},
                {"output_scale_log2", output.scale_log2}, {"rotation_step", rotation}};
    }

    void record(const Json &key, std::uint64_t duration, const char *status) {
        Json event{{"sample_index", sample_index_}, {"call_index", call_index_++},
                   {"key", key}, {"elapsed_ns", duration}, {"status", status}};
        calls.push_back(event);
        compute_ns += duration;
        *output_ << event.dump() << '\n';
        output_->flush();
        if (!*output_) throw std::runtime_error("cannot write per-op call record");
    }

    std::size_t sample_index_ = 0, call_index_ = 0;
    const fhegpu::RuntimePlan *plan_ = nullptr;
    std::ofstream *output_ = nullptr;
};

Json freeze_profile(const Json &calls) {
    struct Aggregate { Json key; std::uint64_t count = 0, total = 0; };
    std::map<std::string, Aggregate> groups;
    for (const auto &call : calls) {
        if (call.at("status") != "ok") continue;
        const Json &key = call.at("key");
        auto &group = groups[key.dump()];
        group.key = key;
        ++group.count;
        group.total += call.at("elapsed_ns").get<std::uint64_t>();
    }
    Json entries = Json::array();
    for (const auto &item : groups) {
        const auto &group = item.second;
        entries.push_back({{"key", group.key}, {"call_count", group.count},
                           {"total_ns", group.total},
                           {"mean_ns", static_cast<double>(group.total) / static_cast<double>(group.count)}});
    }
    return {{"format", "pt-agent-r12-per-op-profile-v1"}, {"time_unit", "ns"},
            {"context_id", kContextId}, {"calibration_sample_index", 0}, {"frozen", true},
            {"entries", entries}, {"bootstrap_latency_ns", nullptr}};
}

Json prediction(std::size_t index, const Json &calls, const Json &profile,
                std::uint64_t measured, std::uint64_t runtime_ns) {
    std::map<std::string, double> means;
    for (const auto &entry : profile.at("entries"))
        means.emplace(entry.at("key").dump(), entry.at("mean_ns").get<double>());
    double predicted = 0.0;
    Json missing = Json::array();
    std::set<std::string> seen_missing;
    for (const auto &call : calls) {
        const auto key = call.at("key").dump();
        const auto found = means.find(key);
        if (found == means.end()) {
            if (seen_missing.insert(key).second) missing.push_back(call.at("key"));
        } else {
            predicted += found->second;
        }
    }
    const bool complete = missing.empty();
    return {{"sample_index", index}, {"prediction_complete", complete}, {"missing_keys", missing},
            {"predicted_compute_ns", complete ? Json(predicted) : Json(nullptr)},
            {"measured_compute_ns", measured},
            {"relative_abs_error", complete && measured != 0
                ? Json(std::abs(predicted - static_cast<double>(measured)) / static_cast<double>(measured))
                : Json(nullptr)}, {"runtime_run_ns", runtime_ns}};
}

Json initial_report() {
    Json samples = Json::array();
    for (std::size_t i = 0; i < 3; ++i)
        samples.push_back({{"index", i}, {"status", "skipped"}, {"error", nullptr},
            {"encrypted_execution_attempted", false}, {"encrypted_execution_completed", false},
            {"output10", nullptr}, {"imaginary10", nullptr}, {"reference10", nullptr},
            {"abs_error10", nullptr}, {"tolerance10", nullptr}, {"passed", nullptr},
            {"measured_compute_ns", nullptr}, {"runtime_run_ns", nullptr},
            {"phase_elapsed_ns", Json::object()}, {"runtime_attribution", nullptr}});
    return {{"format", "pt-agent-r12-native-result-v1"}, {"status", "running"},
        {"failed_phase", nullptr}, {"error", nullptr}, {"check_only", false},
        {"context_started", false}, {"keygen_started", false},
        {"encrypted_execution_attempts", 0}, {"encrypted_execution_completed", 0},
        {"ckks_tolerance_passed", nullptr}, {"compiled_bootstraps", nullptr},
        {"executor_secret_key", false}, {"parameters", nullptr},
        {"loaded_native_library", nullptr}, {"phase_elapsed_ns", Json::object()},
        {"samples", samples}, {"predictions", Json::array()},
        {"compiler_profile_prediction", nullptr},
        {"artifacts", {{"per_op_calls", nullptr}, {"per_op_profile", nullptr}}}};
}

void add_runtime_attribution(Json &sample, const TimedRepeatingCpuApi &api, std::uint64_t runtime_ns) {
    sample["measured_compute_ns"] = api.compute_ns;
    sample["runtime_run_ns"] = runtime_ns;
    const auto covered = api.compute_ns + api.constant_encode_ns;
    require(covered <= runtime_ns, "runtime child timing exceeds its parent");
    sample["runtime_attribution"] = {{"runtime_run_ns", runtime_ns},
        {"compute_ns", api.compute_ns}, {"constant_encode_ns", api.constant_encode_ns},
        {"constant_encode_calls", api.encode_calls}, {"uncovered_ns", runtime_ns - covered},
        {"children_are_inclusive_in_runtime_run", true}};
}

void add_exclusive_attribution(Json &record) {
    const auto &times = record.at("phase_elapsed_ns");
    const auto total = times.at("total").get<std::uint64_t>();
    std::uint64_t children = 0;
    Json child_times = Json::object();
    for (auto it = times.begin(); it != times.end(); ++it) {
        if (it.key() == "total" || it.value().is_null()) continue;
        const auto duration = it.value().get<std::uint64_t>();
        children += duration;
        child_times[it.key()] = duration;
    }
    require(children <= total, "native child timing exceeds its parent");
    record["timing_attribution"] = {{"clock", "std::chrono::steady_clock"},
        {"total_ns", total}, {"child_phases_ns", child_times}, {"exclusive_ns", total - children}};
}

} // namespace

int main(int argc, char **argv) {
    const auto total_start = Clock::now();
    Json report = initial_report();
    Options options;
    std::string phase = "arguments";
    int exit_code = 1;
    // Retain the requested result destination even if another argument is invalid.
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--output") options.output = argv[i + 1];
    const auto checkpoint = [&] {
        report["current_phase"] = phase;
        report["phase_elapsed_ns"]["total"] = elapsed_ns(total_start);
        add_exclusive_attribution(report);
        if (!options.output.empty()) write_json(options.output, report);
    };
    try {
        options = parse_options(argc, argv);
        report["check_only"] = options.check_only;
        checkpoint();
        phase = "plan_read";
        auto loaded = timed(report["phase_elapsed_ns"], phase, [&] {
            return fhegpu::RuntimePlanJsonReader::read_file(options.plan.string());
        });
        phase = "operator_spec_read";
        auto spec = timed(report["phase_elapsed_ns"], phase, [&] {
            return fhegpu::OperatorSpecReader::read_file(options.spec.string());
        });
        phase = "input_read";
        auto document = timed(report["phase_elapsed_ns"], phase, [&] { return read_json(options.inputs); });
        phase = "contract_check";
        auto requirements = timed(report["phase_elapsed_ns"], phase, [&] {
            validate_samples(document);
            validate_contract(loaded.plan, spec.spec);
            fhegpu::PlanRequirements result;
            try {
                result = fhegpu::PlanVerifier::verify(loaded.plan, spec, true);
                fhegpu::PlanVerifier::verify_runtime_target(loaded.plan, 0, 1, 0);
            } catch (const std::exception &exception) {
                throw ContractError(exception.what());
            }
            validate_requirements(result);
            return result;
        });
        phase = "dso_path_check";
        report["loaded_native_library"] = timed(report["phase_elapsed_ns"], phase, loaded_library_path);
        report["compiled_bootstraps"] = 0;
        for (std::size_t i = 0; i < 3; ++i)
            report["samples"][i]["reference10"] = document["samples"][i]["reference10"];
        if (options.check_only) {
            report["status"] = "ok";
            phase = "check_only_complete";
            checkpoint();
            return 0;
        }

        phase = "context";
        report["context_started"] = true;
        checkpoint();
        auto context = timed(report["phase_elapsed_ns"], phase, fixed_context);
        report["parameters"] = context_parameters(context);
        phase = "keygen_secret";
        report["keygen_started"] = true;
        checkpoint();
        auto keygen = timed(report["phase_elapsed_ns"], phase,
            [&] { return std::make_unique<poseidon::KeyGenerator>(context); });
        poseidon::PublicKey public_key;
        phase = "keygen_public";
        timed(report["phase_elapsed_ns"], phase, [&] { keygen->create_public_key(public_key); });
        auto relin_keys = std::make_shared<poseidon::RelinKeys>();
        auto galois_keys = std::make_shared<poseidon::GaloisKeys>();
        std::set<int> rotations;
        bool needs_relin = false;
        for (const auto &key : requirements.keys) {
            if (key.kind == fhegpu::KeyKind::Relin) needs_relin = true;
            if (key.kind == fhegpu::KeyKind::Galois) rotations.insert(*key.rotation_step);
        }
        phase = "keygen_relin";
        if (needs_relin)
            timed(report["phase_elapsed_ns"], phase, [&] { keygen->create_relin_keys(*relin_keys); });
        else report["phase_elapsed_ns"][phase] = nullptr;
        const auto basis = poseidon::runtime_api::binary_rotation_key_basis(rotations, kSlots);
        report["rotation_steps"] = rotations;
        report["rotation_key_basis"] = basis;
        phase = "keygen_galois";
        if (!basis.empty())
            timed(report["phase_elapsed_ns"], phase, [&] {
                keygen->create_galois_keys(std::vector<int>(basis.begin(), basis.end()), *galois_keys);
            });
        else report["phase_elapsed_ns"][phase] = nullptr;
        phase = "execution_setup";
        checkpoint();
        const auto execution_setup_start = Clock::now();
        poseidon::CKKSEncoder encoder(context);
        poseidon::Encryptor encryptor(context, public_key);
        poseidon::Decryptor decryptor(context, keygen->secret_key());
        TimedRepeatingCpuApi api(kContextId, context, relin_keys, galois_keys);
        const auto calls_path = options.output.parent_path() / "per_op_calls.jsonl";
        const auto profile_path = options.output.parent_path() / "per_op_profile.json";
        std::ofstream calls_output(calls_path, std::ios::trunc);
        if (!calls_output) throw std::runtime_error("cannot open per-op call output");
        report["artifacts"]["per_op_calls"] = fs::absolute(calls_path).string();
        Json profile;
        const auto &input_desc = description(loaded.plan, loaded.plan.external_inputs.front());
        report["phase_elapsed_ns"]["execution_setup"] = elapsed_ns(execution_setup_start);
        for (std::size_t i = 0; i < 3; ++i) {
            const auto sample_start = Clock::now();
            Json &sample = report["samples"][i];
            Json &times = sample["phase_elapsed_ns"];
            sample["status"] = "running";
            const auto sample_prefix = "sample_" + std::to_string(i) + ".";
            api.begin_sample(i, loaded.plan, calls_output);
            try {
                phase = sample_prefix + "input_pack";
                auto slots = timed(times, "input_pack", [&] {
                    auto period = document["samples"][i]["input784"].get<std::vector<double>>();
                    period.resize(800, 0.0);
                    std::vector<double> result(kSlots);
                    for (std::size_t j = 0; j < kSlots; ++j) result[j] = period[j % period.size()];
                    return result;
                });
                poseidon::Plaintext plain;
                phase = sample_prefix + "input_encode";
                timed(times, "input_encode", [&] {
                    encoder.encode(slots, context.crt_context()->parms_id_map().at(
                        static_cast<std::uint32_t>(input_desc.level)),
                        std::ldexp(1.0, input_desc.scale_log2), plain);
                });
                phase = sample_prefix + "input_encrypt";
                sample["encrypted_execution_attempted"] = true;
                report["encrypted_execution_attempts"] = report["encrypted_execution_attempts"].get<int>() + 1;
                checkpoint();
                poseidon::Ciphertext ciphertext;
                timed(times, "input_encrypt", [&] { encryptor.encrypt(plain, ciphertext); });
                std::unordered_map<fhegpu::ValueId, PoseidonCpuValue> inputs;
                inputs.emplace(input_desc.id, PoseidonCpuValue::from_ciphertext(std::move(ciphertext)));
                fhegpu::RuntimeResources resources{spec, std::nullopt, true};
                fhegpu::SequentialRuntime<TimedRepeatingCpuApi> runtime(0, 1, 0, api);
                phase = sample_prefix + "runtime_run";
                auto artifact = timed(times, "runtime_run", [&] {
                    return runtime.run(loaded, resources, inputs, fhegpu::DiffMode::FinalOnly);
                });
                sample["encrypted_execution_completed"] = true;
                report["encrypted_execution_completed"] = report["encrypted_execution_completed"].get<int>() + 1;
                add_runtime_attribution(sample, api, times["runtime_run"].get<std::uint64_t>());
                sample["runtime_timing_metadata"] = {
                    {"encode_calls", artifact.timing.encode_calls},
                    {"encode_ns", artifact.timing.encode_nanoseconds},
                    {"compute_calls", artifact.timing.compute_calls},
                    {"compute_including_boot_ns", artifact.timing.compute_including_boot_nanoseconds},
                    {"boot_calls", artifact.timing.boot_calls},
                    {"setup_ns", artifact.timing.setup_nanoseconds},
                    {"initialization_ns", artifact.timing.initialization_nanoseconds},
                    {"online_execution_ns", artifact.timing.online_execution_nanoseconds},
                    {"usage", "metadata_only_not_added_to_adapter_times"}};
                const auto &output = artifact.values.at(loaded.plan.final_outputs.front()).value;
                poseidon::Plaintext final_plain;
                phase = sample_prefix + "final_decrypt";
                timed(times, "final_decrypt", [&] { decryptor.decrypt(output.ciphertext(), final_plain); });
                std::vector<std::complex<double>> decoded;
                phase = sample_prefix + "final_decode";
                timed(times, "final_decode", [&] { encoder.decode(final_plain, decoded); });
                phase = sample_prefix + "ckks_tolerance";
                timed(times, "ckks_tolerance", [&] {
                    require(decoded.size() >= 10, "decoded output has fewer than ten slots");
                    Json real = Json::array(), imaginary = Json::array(), errors = Json::array(), tolerances = Json::array();
                    bool passed = true;
                    for (std::size_t j = 0; j < 10; ++j) {
                        const auto value = decoded[j];
                        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
                            sample["passed"] = false;
                            throw NumericalFailure("nonfinite CKKS output");
                        }
                        const double reference = document["samples"][i]["reference10"][j].get<double>();
                        const double error = std::abs(value.real() - reference);
                        const double tolerance = kAtol + kRtol * std::abs(reference);
                        real.push_back(value.real()); imaginary.push_back(value.imag());
                        errors.push_back(error); tolerances.push_back(tolerance);
                        passed = passed && error <= tolerance;
                    }
                    sample["output10"] = real;
                    sample["imaginary10"] = imaginary;
                    sample["abs_error10"] = errors;
                    sample["tolerance10"] = tolerances;
                    sample["passed"] = passed;
                    if (!passed) throw NumericalFailure("CKKS output exceeds the unchanged tolerance");
                });
                sample["status"] = "ok";
                if (i == 0) {
                    profile = freeze_profile(api.calls);
                    write_json(profile_path, profile);
                    report["artifacts"]["per_op_profile"] = fs::absolute(profile_path).string();
                } else {
                    report["predictions"].push_back(prediction(i, api.calls, profile, api.compute_ns,
                        times["runtime_run"].get<std::uint64_t>()));
                }
                times["total"] = elapsed_ns(sample_start);
                add_exclusive_attribution(sample);
                report["phase_elapsed_ns"]["sample_" + std::to_string(i) + "_total"] = times["total"];
                checkpoint();
            } catch (...) {
                times["total"] = elapsed_ns(sample_start);
                add_exclusive_attribution(sample);
                report["phase_elapsed_ns"]["sample_" + std::to_string(i) + "_total"] = times["total"];
                if (times.contains("runtime_run"))
                    add_runtime_attribution(sample, api, times["runtime_run"].get<std::uint64_t>());
                throw;
            }
        }
        report["ckks_tolerance_passed"] = true;
        report["status"] = "ok";
        phase = "complete";
        exit_code = 0;
    } catch (const ContractError &exception) {
        report["status"] = "unsupported_contract";
        report["error"] = exception.what();
        exit_code = 2;
    } catch (const NumericalFailure &exception) {
        report["status"] = "ckks_tolerance_failed";
        report["ckks_tolerance_passed"] = false;
        report["error"] = exception.what();
        exit_code = 1;
    } catch (const std::exception &exception) {
        report["status"] = "error";
        report["error"] = exception.what();
        exit_code = 1;
    }
    if (exit_code != 0) {
        report["failed_phase"] = phase;
        for (auto &sample : report["samples"])
            if (sample["status"] == "running") {
                sample["status"] = report["status"];
                sample["error"] = report["error"];
            }
        std::cerr << report["error"] << '\n';
    }
    try {
        checkpoint();
    } catch (const std::exception &exception) {
        std::cerr << "cannot retain client result: " << exception.what() << '\n';
        return 1;
    }
    return exit_code;
}
