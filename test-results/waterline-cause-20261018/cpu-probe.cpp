// CPU control experiment: real Runtime scheduling, scalar token arithmetic.
// No CUDA, CKKS encoding, weight I/O or GPU event queries are performed.
#include "runtime/runtime.hpp"
#include "runtime/json_plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "nlohmann/json.hpp"
#include <sys/resource.h>
#include <atomic>
#include <iostream>

using namespace fhegpu;
struct Token { ValueDesc desc; double value; };
class TokenApi {
public:
    using Value = std::shared_ptr<Token>;
    struct CommHandle { std::vector<std::optional<Value>> outputs; };
    const std::thread::id owner = std::this_thread::get_id();
    std::uint64_t compute_calls = 0, main_collections = 0, worker_collections = 0;
    std::string name() const { return "CPU scheduling control"; }
    Value encode_plaintext(const ValueDesc &desc, const std::vector<double> &slots) {
        return std::make_shared<Token>(Token{desc, slots.at(0)});
    }
    Value compute(const ComputeOp &op, const std::vector<Value> &inputs) {
        ++compute_calls;
        auto result = std::make_shared<Token>(*inputs.at(0));
        result->desc.id = op.output;
        if (op.kind == ComputeKind::MulCP) {
            result->value *= inputs.at(1)->value;
            result->desc.scale_log2 += inputs.at(1)->desc.scale_log2;
        } else if (op.kind == ComputeKind::AddCC) {
            result->value += inputs.at(1)->value;
        } else throw std::runtime_error("probe supports only MulCP/AddCC");
        return result;
    }
    CommHandle communicate_async(const CommAction &, const std::vector<Value> &inputs,
                                 const std::vector<ValueDesc> &outputs) {
        CommHandle handle;
        for (const auto &desc : outputs)
            handle.outputs.push_back(std::make_shared<Token>(Token{desc, inputs.at(0)->value}));
        return handle;
    }
    std::vector<std::optional<Value>> posted_outputs(CommHandle &h) {
        auto outputs = std::move(h.outputs);
        h.outputs.resize(outputs.size());
        return outputs;
    }
    std::vector<std::optional<Value>> wait(CommHandle &h) { return posted_outputs(h); }
    void collect_completed() {
        if (std::this_thread::get_id() == owner) ++main_collections;
        else ++worker_collections;
    }
    void drain() {}
    void synchronize(Value &) {}
    void preflight(std::string_view, bool, const TargetConfig &, const OperatorSpec &, const PlanRequirements &) {}
    [[noreturn]] void abort_all(int, const std::string &reason) { throw std::runtime_error(reason); }
    void validate_value(const Value &v, const ValueDesc &d) const {
        if (!v || v->desc.kind != d.kind || v->desc.context != d.context ||
            v->desc.level != d.level || v->desc.scale_log2 != d.scale_log2 ||
            v->desc.components != d.components)
            throw std::runtime_error("token metadata mismatch");
    }
};

int main(int argc, char **argv) {
    if (argc != 4) throw std::runtime_error("usage: probe template-plan.json operator-spec.json weights");
    auto plan = RuntimePlanJsonReader::read_file(argv[1]).plan;
    const auto spec = OperatorSpecReader::read_file(argv[2]);
    const std::size_t count = std::stoull(argv[3]);
    if (!count) throw std::runtime_error("positive weight count required");
    plan.plan_id = 9918; plan.format_version = 2; plan.waterline.reset();
    plan.values.clear(); plan.initialization.clear(); plan.execution.clear(); plan.finalization.clear();
    plan.external_inputs.clear(); plan.final_outputs.clear(); plan.plaintext_bundle.reset();
    const Place host{PlaceKind::Host, 0, 0}, device{PlaceKind::Device, 0, 0};
    const auto define = [&](ValueKind kind, Place place, int scale) {
        ValueId id = plan.values.size();
        plan.values.push_back({id, kind, place, spec.spec.context_id, 49, scale, true,
                               kind == ValueKind::Ciphertext ? 2 : 1});
        return id;
    };
    std::uint64_t transfer = 0;
    const auto upload = [&](ValueId from, ValueId to) {
        CommAction action;
        action.id = transfer++; action.hint = CommHint::Auto;
        action.inputs = {from}; action.outputs = {to};
        action.sources = {host}; action.destinations = {device};
        action.output_types = {plan.values.at(to).kind};
        plan.initialization.push_back({0, action});
    };
    const ValueId input = define(ValueKind::Ciphertext, host, 40);
    const ValueId x = define(ValueKind::Ciphertext, device, 40);
    plan.external_inputs = {input}; upload(input, x);
    plan.initialization.push_back({0, ReleaseOp{input}});
    ValueId sum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto hp = define(ValueKind::Plaintext, host, 40);
        const auto dp = define(ValueKind::Plaintext, device, 40);
        const auto product = define(ValueKind::Ciphertext, device, 80);
        plan.initialization.push_back({0, EncodeOp{InlineEncodePayload{{double(i + 1)}}, hp}});
        upload(hp, dp); plan.initialization.push_back({0, ReleaseOp{hp}});
        plan.execution.push_back({0, ComputeOp{ComputeKind::MulCP, {x, dp}, product, device, {}}});
        plan.execution.push_back({0, ReleaseOp{dp}});
        if (!i) sum = product;
        else {
            const auto next = define(ValueKind::Ciphertext, device, 80);
            plan.execution.push_back({0, ComputeOp{ComputeKind::AddCC, {sum, product}, next, device, {}}});
            plan.execution.push_back({0, ReleaseOp{sum}});
            plan.execution.push_back({0, ReleaseOp{product}});
            sum = next;
        }
    }
    plan.execution.push_back({0, ReleaseOp{x}}); plan.final_outputs = {sum};
    std::size_t ordinal = 0;
    for (auto *phase : {&plan.initialization, &plan.execution, &plan.finalization})
        for (auto &op : *phase) op.ordinal = ordinal++;
    TokenApi api;
    SequentialRuntime<TokenApi> runtime(0, 1, 1, api, DeviceExecutionMode::PerDeviceWorkers, 1);
    const RuntimeResources resources{spec, std::nullopt, false};
    rusage before{}, after{}; getrusage(RUSAGE_SELF, &before);
    const auto result = runtime.run({plan, "sha256:" + std::string(64, '1')}, resources,
        {{input, std::make_shared<Token>(Token{plan.values.at(input), 2.0})}});
    getrusage(RUSAGE_SELF, &after);
    const double expected = double(count) * (count + 1);
    const bool passed = result.values.at(sum).value->value == expected && api.compute_calls == 2 * count - 1;
    if (!passed) throw std::runtime_error("probe result/count mismatch");
    const auto cpu = [](const timeval &t) { return t.tv_sec + t.tv_usec / 1e6; };
    std::cout << nlohmann::json({{"passed", passed}, {"weights", count},
        {"execution_tasks", plan.execution.size()}, {"compute_calls", api.compute_calls},
        {"online_ms", result.timing.online_execution_nanoseconds / 1e6},
        {"compute_api_ms", result.timing.compute_including_boot_nanoseconds / 1e6},
        {"main_collections", api.main_collections}, {"worker_collections", api.worker_collections},
        {"voluntary_context_switches", after.ru_nvcsw - before.ru_nvcsw},
        {"involuntary_context_switches", after.ru_nivcsw - before.ru_nivcsw},
        {"process_cpu_seconds", cpu(after.ru_utime) + cpu(after.ru_stime) - cpu(before.ru_utime) - cpu(before.ru_stime)}}).dump() << '\n';
}
