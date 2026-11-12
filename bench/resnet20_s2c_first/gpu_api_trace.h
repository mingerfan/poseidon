#pragma once

// Runtime API tracing: no model topology, encoder, uploader, or secret key is
// involved here. Resource pointers below are process-local replay bindings;
// only deterministic registration-order resource numbers enter the MLIR.
// The caller must keep every recorded evaluator, key, plaintext and workspace
// immutable and alive until replay finishes. Only trace a prepared replay, not
// an uncached call that can destroy/replace transient resources. The ResNet
// entry owns those resources for the entire recorder/runtime scope.
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/poseidon_context.h"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace poseidon::gpu {

class ApiTraceValue {
public:
    using CT = GpuCiphertextData;
    using Hoist = GpuHoistedDecomposition;
    using Storage = std::variant<std::shared_ptr<CT>, std::shared_ptr<Hoist>>;
    ApiTraceValue() = default;
    explicit ApiTraceValue(std::shared_ptr<CT> value) : storage_(std::move(value)) {}
    explicit ApiTraceValue(std::shared_ptr<Hoist> value) : storage_(std::move(value)) {}
    explicit ApiTraceValue(CT &&value) : ApiTraceValue(std::make_shared<CT>(std::move(value))) {}
    explicit ApiTraceValue(Hoist &&value) : ApiTraceValue(std::make_shared<Hoist>(std::move(value))) {}
    bool is_ciphertext() const { return std::holds_alternative<std::shared_ptr<CT>>(storage_); }
    bool is_buffer() const { return !is_ciphertext(); }
    explicit operator bool() const {
        return std::visit([](const auto &p) { return bool(p); }, storage_);
    }
    CT &ciphertext() const {
        const auto *p = std::get_if<std::shared_ptr<CT>>(&storage_);
        if (!p || !*p) throw std::runtime_error("API trace expected a ciphertext value");
        return **p;
    }
    const Hoist &hoisted() const { return *hoisted_ptr(); }
    std::shared_ptr<Hoist> hoisted_ptr() const {
        const auto *p = std::get_if<std::shared_ptr<Hoist>>(&storage_);
        if (!p || !*p) throw std::runtime_error("API trace expected a hoisted buffer value");
        return *p;
    }
    // Convenient final-output access; these deliberately reject Buffer values.
    CT *operator->() const { return &ciphertext(); }
    CT &operator*() const { return ciphertext(); }
private:
    Storage storage_;
};

namespace api_trace_detail {
inline bool same_meta(const GpuCiphertextMeta &a, const GpuCiphertextMeta &b) {
    return a.parms_id == b.parms_id && a.scale == b.scale &&
        a.correction_factor == b.correction_factor && a.is_ntt_form == b.is_ntt_form &&
        a.degree == b.degree && a.q_count == b.q_count && a.p_count == b.p_count &&
        a.component_count == b.component_count;
}
inline std::string scalar_signature(double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "IEEE double required");
    std::memcpy(&bits, &value, sizeof(bits));
    return std::to_string(bits);
}
inline std::string parms_signature(const parms_id_type &id) {
    std::string result;
    for (auto word : id) result += "." + std::to_string(word);
    return result;
}
inline void check_ct(const GpuCiphertextData &ct, const PoseidonContext &context) {
    const auto &m = ct.meta;
    const auto &ids = context.crt_context()->parms_id_map();
    if (ct.empty() || !m.q_count || m.q_count > ids.size() ||
        m.degree != context.parameters_literal()->degree() ||
        m.parms_id != ids.at(m.q_count - 1) || !std::isfinite(m.scale) || m.scale <= 0 ||
        !m.component_count || ct.polys_.size() != m.component_count || ct.fields_.empty())
        throw std::runtime_error("API trace ciphertext is incompatible with the physical context");
    for (const auto &field : ct.fields_)
        if (field.device_id != 0)
            throw std::runtime_error("API trace requires physical CUDA device zero");
    for (const auto &poly : ct.polys_) {
        if (poly.degree != m.degree || poly.q_count != m.q_count || poly.p_count != m.p_count ||
            poly.shards.empty())
            throw std::runtime_error("API trace ciphertext polynomial metadata mismatch");
        for (const auto &shard : poly.shards) {
            if (shard.field_index >= ct.fields_.size() ||
                shard.limb_begin + shard.limb_count > m.q_count + m.p_count ||
                shard.coeff_begin + shard.coeff_count > m.degree ||
                shard.field_offset > ct.fields_[shard.field_index].buffer.size() ||
                shard.limb_count * shard.coeff_count >
                    ct.fields_[shard.field_index].buffer.size() - shard.field_offset)
                throw std::runtime_error("API trace ciphertext storage/shard metadata mismatch");
        }
    }
}
} // namespace api_trace_detail

struct ApiTraceHoistDesc {
    parms_id_type parms_id{};
    int device_id = 0;
    std::size_t degree = 0, q_count = 0, p_count = 0, dnum = 0;
    std::uint64_t bytes = 0;
    std::string layout;
    static ApiTraceHoistDesc snapshot(const GpuHoistedDecomposition &h) {
        return {h.parms_id, h.device_id, h.degree, h.q_count, h.p_count, h.dnum,
            h.allocated_bytes(), "ckks.hoisted.v1.q" + std::to_string(h.q_count) +
            ".p" + std::to_string(h.p_count) + ".d" + std::to_string(h.dnum) +
            ".n" + std::to_string(h.degree)};
    }
    bool matches(const GpuHoistedDecomposition &h) const {
        const auto other = snapshot(h);
        return parms_id == other.parms_id && device_id == other.device_id &&
            degree == other.degree && q_count == other.q_count && p_count == other.p_count &&
            dnum == other.dnum && bytes == other.bytes && layout == other.layout;
    }
};

class ApiTraceRecorder {
public:
    using CT = GpuCiphertextData;
    using Value = ApiTraceValue;
    using ValueId = std::uint64_t;
    // For mutation nodes destination is a last-use move or one preserving GPU
    // clone of input zero. Const nodes receive an empty destination and read
    // their operands directly, without gratuitous operand copies.
    using Replay = std::function<Value(const std::vector<Value> &, CT destination)>;
    struct ValueDesc {
        ValueId id = 0;
        bool buffer = false;
        GpuCiphertextMeta meta;
        ApiTraceHoistDesc hoist;
    };
    struct Node {
        std::string opcode, callee;
        std::vector<ValueId> inputs;
        ValueId output = 0;
        std::uint64_t latency_us = 1;
        bool mutates_first = false;
        Replay replay;
    };
    struct Resource { const char *kind; const void *pointer; };

    ApiTraceRecorder(const PoseidonContext &context, GpuEvaluator &main_eval)
        : context_(context), main_eval_(main_eval) { resource_id("eval", &main_eval); }
    ApiTraceRecorder(const ApiTraceRecorder &) = delete;
    ApiTraceRecorder &operator=(const ApiTraceRecorder &) = delete;

    ValueId register_input(std::shared_ptr<CT> ct) {
        require_open();
        if (!nodes_.empty()) throw std::runtime_error("register API trace inputs before the first call");
        if (!ct) throw std::runtime_error("null API trace external input");
        for (const auto &existing : registered_inputs_)
            if (existing.get() == ct.get()) throw std::runtime_error("duplicate API trace external input");
        api_trace_detail::check_ct(*ct, context_);
        const auto id = add_ct(*ct);
        registered_inputs_.push_back(std::move(ct));
        input_ids_.push_back(id);
        return id;
    }
    // External inputs are the original resident handles used for replay.
    // Never let capture overwrite one, even if its metadata stays unchanged:
    // the replay would otherwise read the captured output instead of the input.
    void reject_registered_output(const CT &ct) const {
        for (const auto &input : registered_inputs_)
            if (input.get() == &ct)
                throw std::runtime_error("API trace cannot overwrite a registered external input");
    }
    ValueId input(const CT &ct) const {
        const auto id = ct.api_trace_value_id;
        const auto found = values_.find(id);
        if (!id || found == values_.end() || found->second.buffer)
            throw std::runtime_error("unregistered API trace ciphertext input (prepare/register all sources first)");
        if (!api_trace_detail::same_meta(ct.meta, found->second.meta))
            throw std::runtime_error("unrecorded API trace ciphertext metadata change; use trace_gpu(eval).set_scale");
        return id;
    }
    std::vector<ValueId> inputs(const std::vector<const CT *> &cts) const {
        std::vector<ValueId> result;
        for (const auto *ct : cts) {
            if (!ct) throw std::runtime_error("null API trace ciphertext operand");
            result.push_back(input(*ct));
        }
        return result;
    }
    std::string signature(const std::string &opcode, const GpuEvaluator *eval,
        const std::vector<Resource> &resources = {}, const std::string &parameters = {}) {
        require_open();
        std::string result = "gpu." + opcode + ".e" + std::to_string(resource_id("eval", eval));
        for (const auto &resource : resources)
            result += "." + std::string(resource.kind) + std::to_string(resource_id(resource.kind, resource.pointer));
        return result + parameters;
    }
    void record_ct(std::string opcode, std::string callee, std::vector<ValueId> inputs,
        CT &output, Replay replay, bool mutates_first = false, std::uint64_t latency_us = 1) {
        require_open();
        api_trace_detail::check_ct(output, context_);
        record(std::move(opcode), std::move(callee), std::move(inputs), add_ct(output),
            std::move(replay), mutates_first, latency_us);
    }
    ValueId record_buffer(std::string opcode, std::string callee, std::vector<ValueId> inputs,
        const GpuHoistedDecomposition &output, Replay replay, std::uint64_t latency_us = 1) {
        require_open();
        const auto id = next_id_++;
        ValueDesc desc; desc.id = id; desc.buffer = true; desc.hoist = ApiTraceHoistDesc::snapshot(output);
        if (!desc.hoist.bytes || desc.hoist.device_id != 0 ||
            desc.hoist.degree != context_.parameters_literal()->degree())
            throw std::runtime_error("invalid API trace hoist buffer");
        values_.emplace(id, std::move(desc));
        record(std::move(opcode), std::move(callee), std::move(inputs), id,
            std::move(replay), false, latency_us);
        return id;
    }
    void finish(const CT &ct) {
        require_open();
        final_value_ = input(ct);
        finished_ = true;
    }
    const std::vector<Node> &nodes() const { return nodes_; }
    const std::vector<std::shared_ptr<CT>> &registered_inputs() const { return registered_inputs_; }
    const std::vector<ValueId> &input_ids() const { return input_ids_; }
    const std::unordered_map<ValueId, ValueDesc> &values() const { return values_; }
    const ValueDesc &value(ValueId id) const { return values_.at(id); }
    ValueId final_value_id() const {
        if (!finished_) throw std::runtime_error("API trace has not been finished");
        return final_value_;
    }
    const PoseidonContext &context() const { return context_; }
    GpuEvaluator &main_evaluator() const { return main_eval_; }
    bool finished() const { return finished_; }

    void write_mlir(std::ostream &out) const {
        final_value_id();
        const auto moduli = [&] {
            out << "array<i64: ";
            const auto &q = context_.parameters_literal()->q();
            for (std::size_t i = 0; i < q.size(); ++i) out << (i ? ", " : "") << q[i].value();
            out << ">";
        };
        const auto type = [&](ValueId id) {
            const auto &desc = value(id);
            if (desc.buffer)
                return "!ckks.buffer<" + std::to_string(desc.hoist.bytes) + ", \"" + desc.hoist.layout + "\">";
            const auto exponent = std::log2(desc.meta.scale);
            if (exponent < -0.5 || exponent > double(std::numeric_limits<unsigned>::max()) - 0.5)
                throw std::runtime_error("API trace scale is outside nominal CKKS type range");
            return "tensor<1x!ckks.poly<" + std::to_string(desc.meta.component_count) + " * " +
                std::to_string(std::llround(exponent)) + " * " + std::to_string(desc.meta.q_count - 1) + ">>";
        };
        const auto saved_flags = out.flags();
        const auto saved_precision = out.precision();
        out << std::scientific << std::setprecision(17);
        out << "module attributes {poseidon.task_granularity = \"gpu_api_trace\", ckks.rns_moduli = ";
        moduli();
        out << "} {\n  func.func @gpu_api_trace(";
        for (std::size_t i = 0; i < input_ids_.size(); ++i)
            out << (i ? ", " : "") << "%v" << input_ids_[i] << ": " << type(input_ids_[i]);
        out << ") -> " << type(final_value_) << " attributes {runtime.external_inputs_device = 0 : i64, ckks.rns_moduli = ";
        moduli();
        // earth.arg_scales is the exporter's floating log2 scale convention.
        out << ", earth.arg_scales = array<f64";
        if (!input_ids_.empty()) out << ": ";
        for (std::size_t i = 0; i < input_ids_.size(); ++i)
            out << (i ? ", " : "") << std::log2(value(input_ids_[i]).meta.scale);
        out << ">} {\n";
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            const auto &node = nodes_[i];
            out << "    %v" << node.output << " = \"ckks.apicall\"(";
            for (std::size_t j = 0; j < node.inputs.size(); ++j)
                out << (j ? ", " : "") << "%v" << node.inputs[j];
            out << ") {callee = \"" << node.callee << "\", call_id = " << i << " : i64, latency_us = "
                << node.latency_us << " : i64";
            if (!value(node.output).buffer)
                out << ", runtime.scale_log2 = " << std::log2(value(node.output).meta.scale) << " : f64";
            out << "} : (";
            for (std::size_t j = 0; j < node.inputs.size(); ++j)
                out << (j ? ", " : "") << type(node.inputs[j]);
            out << ") -> " << type(node.output) << "\n";
        }
        out << "    return %v" << final_value_ << " : " << type(final_value_) << "\n  }\n}\n";
        out.flags(saved_flags);
        out.precision(saved_precision);
        if (!out) throw std::runtime_error("cannot write API trace MLIR");
    }
private:
    void require_open() const {
        if (finished_) throw std::runtime_error("cannot append to a finished API trace");
    }
    ValueId add_ct(CT &ct) {
        const auto id = next_id_++;
        ValueDesc desc; desc.id = id; desc.meta = ct.meta;
        values_.emplace(id, std::move(desc));
        ct.api_trace_value_id = id;
        return id;
    }
    void record(std::string opcode, std::string callee, std::vector<ValueId> inputs,
        ValueId output, Replay replay, bool mutates_first, std::uint64_t latency_us) {
        for (const auto id : inputs) {
            if (!values_.count(id) || id >= output)
                throw std::runtime_error("invalid API trace dependency");
        }
        if (mutates_first && (inputs.empty() || value(inputs.front()).buffer))
            throw std::runtime_error("API trace mutation needs a ciphertext input zero");
        nodes_.push_back({std::move(opcode), std::move(callee), std::move(inputs), output,
            std::max<std::uint64_t>(1, latency_us), mutates_first, std::move(replay)});
    }
    std::uint64_t resource_id(const std::string &kind, const void *pointer) {
        if (!pointer) throw std::runtime_error("null API trace resident resource");
        for (std::size_t i = 0; i < resources_.size(); ++i)
            if (resources_[i].first == kind && resources_[i].second == pointer) return i;
        resources_.emplace_back(kind, pointer);
        return resources_.size() - 1;
    }
    const PoseidonContext &context_;
    GpuEvaluator &main_eval_;
    std::vector<std::pair<std::string, const void *>> resources_;
    std::vector<std::shared_ptr<CT>> registered_inputs_;
    std::vector<ValueId> input_ids_;
    std::unordered_map<ValueId, ValueDesc> values_;
    std::vector<Node> nodes_;
    ValueId next_id_ = 1, final_value_ = 0;
    bool finished_ = false;
};

inline thread_local ApiTraceRecorder *active_api_trace_recorder = nullptr;
class ScopedApiTrace {
public:
    explicit ScopedApiTrace(ApiTraceRecorder &recorder) {
        if (active_api_trace_recorder) throw std::runtime_error("nested API tracing is unsupported");
        if (recorder.finished()) throw std::runtime_error("cannot activate a finished API trace");
        active_api_trace_recorder = &recorder;
    }
    ~ScopedApiTrace() { active_api_trace_recorder = nullptr; }
    ScopedApiTrace(const ScopedApiTrace &) = delete;
    ScopedApiTrace &operator=(const ScopedApiTrace &) = delete;
};

// Adapter for the actual SequentialRuntime, not a second schedule executor.
class TracedPlanApi {
public:
    using CT = GpuCiphertextData;
    using Value = ApiTraceValue;
    using PlanValueId = fhegpu::ValueId;
    struct CommHandle {};
    struct HoistStats {
        std::uint64_t live_bytes = 0, peak_bytes = 0, released_bytes = 0;
        std::size_t releases = 0;
    };
    TracedPlanApi(const ApiTraceRecorder &recorder, const fhegpu::RuntimePlan &plan,
        const PoseidonContext &context, GpuEvaluator &main_eval)
        : recorder_(recorder), context_(context), eval_(main_eval), stats_(std::make_shared<HoistStats>()) {
        recorder.final_value_id();
        if (&recorder.context() != &context || &recorder.main_evaluator() != &main_eval)
            throw std::runtime_error("API trace replay must bind its original resident context/evaluator");
        for (const auto &desc : plan.values) {
            if (!single_gpu(desc.place) || !descriptors_.emplace(desc.id, desc).second)
                throw std::runtime_error("API trace plan has duplicate descriptors or a nonlocal place");
        }
        if (plan.external_inputs.size() != recorder.input_ids().size())
            throw std::runtime_error("API trace plan external input count mismatch");
        for (std::size_t i = 0; i < plan.external_inputs.size(); ++i) {
            const auto plan_id = plan.external_inputs[i];
            const auto trace_id = recorder.input_ids()[i];
            if (!trace_ids_.emplace(plan_id, trace_id).second)
                throw std::runtime_error("duplicate API trace plan external input");
            match_descriptor(descriptors_.at(plan_id), recorder.value(trace_id));
            external_inputs_.emplace(plan_id, Value(recorder.registered_inputs()[i]));
            // Resident inputs are not consumed: the same backend may replay
            // again and the reference path still owns these ciphertexts.
            ++initial_remaining_[plan_id];
        }
        std::vector<bool> seen(recorder.nodes().size(), false);
        const auto scan = [&](const std::vector<fhegpu::Instruction> &phase) {
            for (const auto &instruction : phase) {
                const auto *op = std::get_if<fhegpu::ComputeOp>(&instruction.body);
                if (!op) {
                    if (const auto *release = std::get_if<fhegpu::ReleaseOp>(&instruction.body)) {
                        if (!trace_ids_.count(release->value))
                            throw std::runtime_error("API trace plan releases an undefined value");
                    } else if (!std::holds_alternative<fhegpu::FenceOp>(instruction.body))
                        throw std::runtime_error("API trace plan accepts only ApiCall/Release/Fence (no encode/communication)");
                    continue;
                }
                if (op->kind != fhegpu::ComputeKind::ApiCall || op->reuse_input || !single_gpu(op->place))
                    throw std::runtime_error("API trace plan needs single-GPU ApiCalls without reuse_input");
                const auto *attrs = std::get_if<fhegpu::ApiCallAttrs>(&op->attrs);
                if (!attrs || attrs->call_id >= recorder.nodes().size())
                    throw std::runtime_error("API trace plan has an unknown call_id");
                const auto &node = recorder.nodes()[attrs->call_id];
                if (seen[attrs->call_id] || node.callee != attrs->callee || op->inputs.size() != node.inputs.size())
                    throw std::runtime_error("API trace opcode/resource/parameter signature differs from registry");
                for (std::size_t i = 0; i < op->inputs.size(); ++i) {
                    if (trace_ids_.at(op->inputs[i]) != node.inputs[i])
                        throw std::runtime_error("API trace plan dependency mismatch");
                    ++initial_remaining_[op->inputs[i]];
                }
                if (!trace_ids_.emplace(op->output, node.output).second)
                    throw std::runtime_error("API trace plan defines a value twice");
                match_descriptor(descriptors_.at(op->output), recorder.value(node.output));
                seen[attrs->call_id] = true;
            }
        };
        scan(plan.initialization); scan(plan.execution); scan(plan.finalization);
        for (bool present : seen) if (!present) throw std::runtime_error("API trace plan is missing a recorded call");
        if (trace_ids_.size() != descriptors_.size() || plan.final_outputs.size() != 1 ||
            trace_ids_.at(plan.final_outputs.front()) != recorder.final_value_id())
            throw std::runtime_error("API trace plan has extra values or the wrong final ciphertext");
        ++initial_remaining_[plan.final_outputs.front()]; // never move a retained final output
        remaining_ = initial_remaining_;
    }
    std::string name() const { return "TracedGpuApi"; }
    // SequentialRuntime can receive a different LoadedRuntimePlan than the
    // constructor did. Recheck the actual graph and its ValueId binding rather
    // than trusting a digest (binary preflight digests only metadata).
    void validate_plan(const fhegpu::RuntimePlan &plan) {
        TracedPlanApi checked(recorder_, plan, context_, eval_);
        if (checked.trace_ids_ != trace_ids_ || checked.initial_remaining_ != initial_remaining_)
            throw std::runtime_error("runtime plan differs from the validated API trace binding");
    }
    static const CT &result_ciphertext(const Value &value) { return value.ciphertext(); }
    const std::unordered_map<PlanValueId, Value> &external_inputs() const { return external_inputs_; }
    Value compute(const fhegpu::ComputeOp &op, const std::vector<Value> &inputs) {
        if (std::this_thread::get_id() != execution_thread_ || gpu_execution_stream() != execution_stream_)
            throw std::runtime_error("API trace replay requires one submission thread and one CUDA stream");
        if (active_api_trace_recorder) throw std::runtime_error("cannot replay a plan while tracing");
        if (op.kind != fhegpu::ComputeKind::ApiCall || op.reuse_input || !single_gpu(op.place))
            throw std::runtime_error("unsupported operation in traced GPU API");
        const auto &attrs = std::get<fhegpu::ApiCallAttrs>(op.attrs);
        const auto &node = recorder_.nodes().at(attrs.call_id);
        if (attrs.callee != node.callee || inputs.size() != node.inputs.size() || op.inputs.size() != inputs.size() ||
            trace_ids_.at(op.output) != node.output)
            throw std::runtime_error("API trace dispatch differs from preflighted plan");
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (trace_ids_.at(op.inputs[i]) != node.inputs[i])
                throw std::runtime_error("API trace dispatch dependency mismatch");
            validate_value(inputs[i], descriptors_.at(op.inputs[i]));
            auto &uses = remaining_.at(op.inputs[i]);
            if (!uses) throw std::runtime_error("API trace input submitted too many times");
            --uses;
        }
        CT destination;
        if (node.mutates_first) {
            // An aliased source operand in this same call still needs the old
            // storage, even if all its submitted uses happen in this call.
            const bool aliases_operand = std::count(op.inputs.begin(), op.inputs.end(), op.inputs.front()) > 1;
            if (!remaining_.at(op.inputs.front()) && !aliases_operand)
                destination = std::move(inputs.front().ciphertext());
            else eval_.drop_modulus(inputs.front().ciphertext(), destination, inputs.front()->meta.parms_id);
        }
        auto output = node.replay(inputs, std::move(destination));
        if (output.is_buffer()) {
            auto owned = output.hoisted_ptr();
            const auto bytes = owned->allocated_bytes();
            const auto stats = stats_;
            stats->live_bytes += bytes;
            stats->peak_bytes = std::max(stats->peak_bytes, stats->live_bytes);
            // The runtime's ValueStore is the only owner of intermediate
            // handles. This deleter observes actual final-owner release.
            output = Value(std::shared_ptr<GpuHoistedDecomposition>(owned.get(),
                [owned, stats, bytes](GpuHoistedDecomposition *) mutable {
                    owned.reset();
                    stats->live_bytes -= bytes;
                    stats->released_bytes += bytes;
                    ++stats->releases;
                }));
        }
        if (output.is_ciphertext()) output->api_trace_value_id = node.output;
        validate_value(output, descriptors_.at(op.output));
        ++calls_;
        if (node.opcode == "bootstrap_raise") ++boots_;
        if (node.opcode == "bootstrap_linear_layer") ++boot_layers_;
        if (node.opcode == "eval_mod_high_precision") ++boot_eval_mods_;
        if (node.opcode == "rotation_hoist") ++hoists_;
        if (node.opcode == "rotate_from_hoist") ++suffixes_;
        dispatch_order_.push_back(attrs.call_id);
        return output;
    }
    void validate_value(const Value &value, const fhegpu::ValueDesc &expected) const {
        if (!value || !single_gpu(expected.place)) throw std::runtime_error("null/nonlocal traced API value");
        const auto &snapshot = recorder_.value(trace_ids_.at(expected.id));
        match_descriptor(expected, snapshot);
        if (snapshot.buffer) {
            if (!value.is_buffer() || !snapshot.hoist.matches(value.hoisted()))
                throw std::runtime_error("API trace runtime hoist buffer differs from its bytes/layout/context descriptor");
        } else {
            if (!value.is_ciphertext() || !api_trace_detail::same_meta(value->meta, snapshot.meta))
                throw std::runtime_error("API trace runtime ciphertext differs from recorded physical metadata");
            api_trace_detail::check_ct(*value, context_);
        }
    }
    void preflight(std::string_view, bool, const fhegpu::TargetConfig &target,
        const fhegpu::OperatorSpec &spec, const fhegpu::PlanRequirements &) {
        if (target.world_size != 1 || target.device_counts != std::vector<int>{1})
            throw std::runtime_error("traced GPU API requires exactly one GPU");
        std::vector<std::uint64_t> moduli;
        for (const auto &q : context_.parameters_literal()->q()) moduli.push_back(q.value());
        if (spec.poly_degree != context_.parameters_literal()->degree() || spec.rns_moduli != moduli)
            throw std::runtime_error("API trace OperatorSpec has different physical N/Q primes");
        if (stats_->live_bytes) throw std::runtime_error("previous API trace replay still owns hoisted buffers");
        execution_thread_ = std::this_thread::get_id();
        execution_stream_ = gpu_execution_stream();
        remaining_ = initial_remaining_;
        calls_ = boots_ = boot_layers_ = boot_eval_mods_ = hoists_ = suffixes_ = 0;
        *stats_ = HoistStats{};
        dispatch_order_.clear();
    }
    void drain() { gpu_check_cuda(cudaDeviceSynchronize(), "traced API runtime drain"); }
    void synchronize(Value &) { drain(); }
    void collect_completed() {}
    [[noreturn]] void abort_all(int, const std::string &message) { throw std::runtime_error(message); }
    Value encode_plaintext(const fhegpu::ValueDesc &, const std::vector<double> &) {
        throw std::runtime_error("API trace cannot encode/upload; prepare resident constants before capture");
    }
    CommHandle communicate_async(const fhegpu::CommAction &, const std::vector<Value> &,
        const std::vector<fhegpu::ValueDesc> &) {
        throw std::runtime_error("single-GPU API trace does not support communication");
    }
    std::vector<std::optional<Value>> wait(CommHandle &) {
        throw std::runtime_error("single-GPU API trace does not support communication");
    }
    std::size_t calls() const { return calls_; }
    // One raise stage occurs per complete traced Boot.
    std::size_t boot_calls() const { return boots_; }
    std::size_t boot_layer_calls() const { return boot_layers_; }
    std::size_t boot_eval_mod_calls() const { return boot_eval_mods_; }
    std::size_t hoist_calls() const { return hoists_; }
    std::size_t rotation_suffix_calls() const { return suffixes_; }
    std::uint64_t peak_hoist_bytes() const { return stats_->peak_bytes; }
    std::uint64_t live_hoist_bytes() const { return stats_->live_bytes; }
    std::uint64_t released_hoist_bytes() const { return stats_->released_bytes; }
    std::size_t hoist_releases() const { return stats_->releases; }
    const std::vector<std::uint64_t> &dispatch_order() const { return dispatch_order_; }
private:
    static bool single_gpu(const fhegpu::Place &place) {
        return place.kind == fhegpu::PlaceKind::Device && place.rank == 0 && place.index == 0;
    }
    static void match_descriptor(const fhegpu::ValueDesc &expected, const ApiTraceRecorder::ValueDesc &actual) {
        if (!single_gpu(expected.place)) throw std::runtime_error("nonlocal API trace value descriptor");
        if (actual.buffer) {
            if (expected.kind != fhegpu::ValueKind::Buffer || expected.level != 0 || expected.scale_log2 != 0 ||
                expected.ntt || expected.components != 0 || expected.buffer_bytes != actual.hoist.bytes ||
                expected.buffer_layout != actual.hoist.layout)
                throw std::runtime_error("API trace plan buffer descriptor mismatch");
        } else if (expected.kind != fhegpu::ValueKind::Ciphertext ||
            expected.level != int(actual.meta.q_count - 1) || expected.components != int(actual.meta.component_count) ||
            expected.ntt != actual.meta.is_ntt_form || expected.buffer_bytes || !expected.buffer_layout.empty() ||
            !std::isfinite(expected.scale_log2) ||
            std::abs(expected.scale_log2 - std::log2(actual.meta.scale)) > 1e-10)
            throw std::runtime_error("API trace plan ciphertext shape/floating scale descriptor mismatch");
    }
    const ApiTraceRecorder &recorder_;
    const PoseidonContext &context_;
    GpuEvaluator &eval_;
    std::unordered_map<PlanValueId, Value> external_inputs_;
    std::unordered_map<PlanValueId, fhegpu::ValueDesc> descriptors_;
    std::unordered_map<PlanValueId, ApiTraceRecorder::ValueId> trace_ids_;
    std::unordered_map<PlanValueId, std::size_t> initial_remaining_, remaining_;
    std::shared_ptr<HoistStats> stats_;
    std::thread::id execution_thread_{};
    cudaStream_t execution_stream_ = nullptr;
    std::size_t calls_ = 0, boots_ = 0, boot_layers_ = 0, boot_eval_mods_ = 0,
        hoists_ = 0, suffixes_ = 0;
    std::vector<std::uint64_t> dispatch_order_;
};

} // namespace poseidon::gpu
