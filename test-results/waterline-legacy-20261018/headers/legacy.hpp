#pragma once

#include "runtime/plaintext_bundle.hpp"
#include "runtime/thread_trace.hpp"
#include "runtime/verifier.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace fhegpu {

namespace detail {

template <class Api, class = void>
struct HasPostedOutputs : std::false_type {};

template <class Api>
struct HasPostedOutputs<
    Api, std::void_t<decltype(std::declval<Api &>().posted_outputs(
             std::declval<typename Api::CommHandle &>()))>> : std::true_type {};

template <class Api, class = void>
struct UsesBackgroundCommunicationIssuer : std::false_type {};

template <class Api>
struct UsesBackgroundCommunicationIssuer<
    Api, std::void_t<decltype(Api::background_communication_issuer)>>
    : std::bool_constant<Api::background_communication_issuer> {};

template <class Api, class = void>
struct HasComputeReuse : std::false_type {};

template <class Api>
struct HasComputeReuse<Api, std::void_t<
    decltype(std::declval<Api &>().supports_reuse(std::declval<const ComputeOp &>())),
    decltype(std::declval<Api &>().compute_reuse(std::declval<const ComputeOp &>(),
        std::declval<typename Api::Value>(),
        std::declval<const std::vector<typename Api::Value> &>()))>> : std::true_type {};

} // namespace detail

enum class DiffMode { FinalOnly, AllValuesAfterRun };
enum class DeviceExecutionMode { Sequential, PerDeviceWorkers };

struct RuntimeResources {
    const LoadedOperatorSpec &operator_spec;
    std::optional<std::filesystem::path> plaintext_bundle_dir;
    bool skip_artifact_digest_checks = false;
    BundleReadOptions plaintext_bundle_options{};
};

template <class Value>
struct ArtifactValue { Place place; Value value; };

// Wall time inside Api::compute. Asynchronous APIs report submission time.
struct RuntimeTiming {
    std::size_t encode_calls = 0, bundle_read_calls = 0, fence_calls = 0;
    std::uint64_t encode_nanoseconds = 0, bundle_read_nanoseconds = 0;
    std::uint64_t bundle_read_bytes = 0, raw_peak_bytes = 0, fence_nanoseconds = 0;
    std::uint64_t bundle_resident_bytes = 0, bundle_resident_load_nanoseconds = 0;
    std::size_t compute_calls = 0;
    std::size_t boot_calls = 0;
    std::uint64_t compute_including_boot_nanoseconds = 0;
    std::uint64_t boot_nanoseconds = 0;
    std::uint64_t setup_nanoseconds = 0;
    std::uint64_t initialization_nanoseconds = 0;
    std::uint64_t online_execution_nanoseconds = 0;
    std::size_t communication_post_calls = 0;
    std::size_t communication_wait_calls = 0;
    std::uint64_t communication_post_nanoseconds = 0;
    std::uint64_t communication_wait_nanoseconds = 0;
    std::uint64_t compute_excluding_boot_nanoseconds() const {
        return compute_including_boot_nanoseconds - boot_nanoseconds;
    }
};

template <class Value>
struct RunArtifact {
    std::map<ValueId, ArtifactValue<Value>> values;
    RuntimeTiming timing;
};

template <class Api>
class ValueStore {
public:
    using Value = typename Api::Value;
    struct Ready { Place place; Value value; };
    struct Pending { Place place; std::size_t group; std::size_t local_slot; };
    struct Released { Place place; };
    using Entry = std::variant<Ready, Pending, Released>;

    void define_ready(ValueId id, Place place, Value value) {
        if (!entries_.emplace(id, Ready{place, std::move(value)}).second)
            throw std::runtime_error("ValueId defined twice in ValueStore: " + std::to_string(id));
    }
    void define_pending(ValueId id, Place place, std::size_t group, std::size_t local_slot) {
        if (!entries_.emplace(id, Pending{place, group, local_slot}).second)
            throw std::runtime_error("ValueId defined twice in ValueStore: " + std::to_string(id));
    }
    Entry &lookup(ValueId id) {
        auto it = entries_.find(id);
        if (it == entries_.end()) throw std::runtime_error("ValueId is absent from local ValueStore: " + std::to_string(id));
        return it->second;
    }
    const std::unordered_map<ValueId, Entry> &entries() const { return entries_; }
private:
    std::unordered_map<ValueId, Entry> entries_;
};

template <class Api>
class SequentialRuntime {
public:
    using Value = typename Api::Value;
    using CommHandle = typename Api::CommHandle;

    SequentialRuntime(int rank, int world_size, int local_devices, Api &api,
                      DeviceExecutionMode device_execution_mode =
                          DeviceExecutionMode::Sequential,
                      std::size_t device_worker_count = 0)
        : rank_(rank), world_size_(world_size), local_devices_(local_devices), api_(api),
          device_execution_mode_(device_execution_mode),
          device_worker_count_(device_worker_count) {}

    RunArtifact<Value> run(const LoadedRuntimePlan &loaded_plan,
                           const RuntimeResources &resources,
                           const std::unordered_map<ValueId, Value> &local_inputs,
                           DiffMode diff_mode = DiffMode::FinalOnly) {
        plan_ = &loaded_plan.plan;
        plan_source_sha256_ = loaded_plan.source_sha256;
        current_ = nullptr;
        current_value_.reset();
        store_ = ValueStore<Api>{};
        groups_.clear();
        uses_.clear();
        reset_parallel_execution_state();
        bundle_slots_.clear();
        bundle_index_.reset();
        descriptors_.clear();
        for (const auto &value : plan_->values) descriptors_.emplace(value.id, &value);
        return_all_values_ = diff_mode == DiffMode::AllValuesAfterRun;
        timing_ = RuntimeTiming{};
        ThreadTrace::set_thread_name("rank-" + std::to_string(rank_) + "-main");
        open_trace();
        try {
            const auto setup_start = std::chrono::steady_clock::now();
            if (resources.skip_artifact_digest_checks) {
                std::cerr << "WARNING: rank " << rank_
                          << " is running with skip_artifact_digest_checks=true; artifact digest comparisons are disabled\n";
                std::cerr.flush();
            }
            const PlanRequirements requirements = PlanVerifier::verify(
                *plan_, resources.operator_spec, resources.skip_artifact_digest_checks);
            PlanVerifier::verify_runtime_target(*plan_, rank_, world_size_, local_devices_);
            check_memory_instruction_support(diff_mode);
            count_uses();
            load_bundle(resources);
            api_.preflight(loaded_plan.preflight_identity.empty() ? loaded_plan.source_sha256
                                                                : loaded_plan.preflight_identity,
                           resources.skip_artifact_digest_checks,
                           plan_->target, resources.operator_spec.spec, requirements);
            bind_inputs(local_inputs);

            const auto initialization_start = std::chrono::steady_clock::now();
            timing_.setup_nanoseconds = elapsed_nanoseconds(
                setup_start, initialization_start);
            execute_phase(plan_->initialization);
            finish_all_groups();

            const auto online_execution_start = std::chrono::steady_clock::now();
            timing_.initialization_nanoseconds = elapsed_nanoseconds(
                initialization_start, online_execution_start);
            if (device_execution_mode_ == DeviceExecutionMode::PerDeviceWorkers) {
                execute_device_parallel_phases();
            } else {
                execute_phase(plan_->execution);
                execute_phase(plan_->finalization);
                finish_all_groups();
            }
            synchronize_final_outputs();
            api_.drain();
            timing_.online_execution_nanoseconds = elapsed_nanoseconds(
                online_execution_start, std::chrono::steady_clock::now());
            auto artifact = make_artifact(diff_mode);
            store_ = ValueStore<Api>{};
            groups_.clear();
            reset_parallel_execution_state();
            ThreadTrace::write_json(rank_);
            return artifact;
        } catch (const std::exception &error) {
            const std::string diagnostic = format_error(error.what());
            std::cerr << diagnostic << std::endl;
            std::cerr.flush();
            api_.abort_all(1, diagnostic);
        }
    }

private:
    void check_memory_instruction_support(DiffMode mode) {
        const auto check_phase = [&](const std::vector<Instruction> &instructions) {
            for (const auto &instruction : instructions) {
                const bool release = std::holds_alternative<ReleaseOp>(instruction.body);
                const auto *op = std::get_if<ComputeOp>(&instruction.body);
                if (!release && (!op || !op->reuse_input)) continue;
                current_ = &instruction;
                if (mode == DiffMode::AllValuesAfterRun)
                    throw std::runtime_error("AllValuesAfterRun is incompatible with Release/reuse_input");
                if (op && op->reuse_input) {
                    if constexpr (detail::HasComputeReuse<Api>::value) {
                        if (!api_.supports_reuse(*op))
                            throw std::runtime_error("Api does not support reuse_input for " + to_string(op->kind));
                    } else {
                        throw std::runtime_error("Api does not support reuse_input");
                    }
                }
                current_ = nullptr;
            }
        };
        check_phase(plan_->initialization);
        check_phase(plan_->execution);
        check_phase(plan_->finalization);
    }

    static std::uint64_t elapsed_nanoseconds(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point finish) {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start)
                .count());
    }

    void open_trace() {
        runtime_trace_.reset();
        const char *trace_path = std::getenv("POSEIDON_RUNTIME_TRACE");
        if (trace_path == nullptr || trace_path[0] == '\0' ||
            std::string_view(trace_path) == "0")
            return;
        std::string path = std::string_view(trace_path) == "1"
                               ? "/tmp/poseidon-runtime"
                               : std::string(trace_path);
        const std::string rank_text = std::to_string(rank_);
        std::size_t offset = 0;
        bool replaced = false;
        while ((offset = path.find("%r", offset)) != std::string::npos) {
            path.replace(offset, 2, rank_text);
            offset += rank_text.size();
            replaced = true;
        }
        if (!replaced) path += ".rank" + rank_text + ".csv";
        runtime_trace_ = std::make_unique<std::ofstream>(
            path, std::ios::out | std::ios::trunc);
        if (!*runtime_trace_)
            throw std::runtime_error("cannot open Poseidon Runtime trace file");
        *runtime_trace_
            << "rank,event,id,ordinal,consumer_ordinal,op,duration_ns\n";
        runtime_trace_->flush();
    }

    void trace_event(const char *event, std::uint64_t id,
                     std::uint64_t ordinal,
                     std::optional<std::uint64_t> consumer_ordinal,
                     std::string_view op,
                     std::uint64_t duration_nanoseconds) {
        if (!runtime_trace_) return;
        ThreadTraceLockGuard lock(trace_mutex_, "runtime.trace");
        *runtime_trace_ << rank_ << ',' << event << ',' << id << ','
                        << ordinal << ',';
        if (consumer_ordinal) *runtime_trace_ << *consumer_ordinal;
        *runtime_trace_ << ',' << op << ',' << duration_nanoseconds << '\n';
        runtime_trace_->flush();
    }

    struct PendingGroup {
        CommAction action;
        std::vector<ValueId> local_output_ids;
        std::vector<std::size_t> local_output_slots;
        std::vector<bool> local_output_posted;
        CommHandle handle;
        std::uint64_t instruction_ordinal = 0;
        bool completed = false;
    };

    struct ParallelGroup;

    struct ValueUses {
        std::size_t remaining = 0;
        bool release_requested = false;
    };

    struct ParallelValue {
        Place place;
        std::mutex mutex;
        std::condition_variable condition;
        bool defined = false;
        ValueUses uses;
        std::optional<Value> value;
        std::shared_ptr<ParallelGroup> group;
    };

    struct ParallelOutput {
        ValueId id = 0;
        std::size_t action_slot = 0;
        std::shared_ptr<ParallelValue> value;
    };

    struct ParallelGroup {
        CommAction action;
        std::uint64_t instruction_ordinal = 0;
        std::vector<ValueId> local_input_ids;
        std::vector<ParallelOutput> local_outputs;
        CommHandle handle;
        std::mutex mutex;
        bool cross_rank = false;
        bool posted = false;
        bool completed = false;
    };

    using ParallelTask = std::function<void()>;

    const ValueDesc &desc(ValueId id) const {
        auto found = descriptors_.find(id);
        if (found != descriptors_.end()) return *found->second;
        throw std::runtime_error("missing value descriptor for " + std::to_string(id));
    }

    void count_uses() {
        for (const auto &value : plan_->values)
            if (value.place.rank == rank_) uses_.emplace(value.id, ValueUses{});
        const auto count_phase = [&](const auto &instructions, bool parallel) {
            for (const auto &instruction : instructions) {
                if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
                    if (op->place.rank == rank_)
                        for (ValueId id : op->inputs) ++uses_.at(id).remaining;
                } else if (const auto *action = std::get_if<CommAction>(&instruction.body)) {
                    std::size_t submissions = 1;
                    if (parallel) {
                        // Workers post one slice per local destination; the
                        // communication issuer posts all remote destinations together.
                        submissions = is_cross_rank_action(*action) ? 1 : 0;
                        submissions += std::count_if(
                            action->destinations.begin(), action->destinations.end(),
                            [this](const Place &place) { return place.rank == rank_; });
                    }
                    for (std::size_t i = 0; i < action->inputs.size(); ++i)
                        if (action->sources[i].rank == rank_)
                            uses_.at(action->inputs[i]).remaining += submissions;
                }
            }
        };
        count_phase(plan_->initialization, false);
        const bool parallel = device_execution_mode_ == DeviceExecutionMode::PerDeviceWorkers;
        count_phase(plan_->execution, parallel);
        count_phase(plan_->finalization, parallel);
    }

    bool can_clear(const ValueUses &uses) const {
        return uses.release_requested && uses.remaining == 0;
    }

    void clear_sequential_value(ValueId id) {
        if (!can_clear(uses_.at(id))) return;
        auto &entry = store_.lookup(id);
        if (const auto *ready = std::get_if<typename ValueStore<Api>::Ready>(&entry)) {
            const auto place = ready->place;
            entry = typename ValueStore<Api>::Released{place};
        }
        // Pending data is discarded when the communication result arrives.
    }

    void submitted_sequential_uses(const std::vector<ValueId> &ids) {
        for (ValueId id : ids) {
            auto &uses = uses_.at(id);
            if (uses.remaining == 0)
                throw std::runtime_error("submission count underflow for ValueId " + std::to_string(id));
            --uses.remaining;
            clear_sequential_value(id);
        }
    }

    void execute_release(ValueId id, std::uint64_t ordinal, bool parallel) {
        if (desc(id).place.rank != rank_) return;
        if (parallel) {
            auto value = parallel_value(id);
            {
                ThreadTraceLockGuard lock(value->mutex, "runtime.parallel_value");
                value->uses.release_requested = true;
                if (can_clear(value->uses)) value->value.reset();
            }
            value->condition.notify_all();
        } else {
            uses_.at(id).release_requested = true;
            clear_sequential_value(id);
        }
        api_.collect_completed();
        trace_event("release", id, ordinal, std::nullopt, "Release", 0);
    }

    void submitted_parallel_uses(const std::vector<ValueId> &ids) {
        for (ValueId id : ids) {
            auto value = parallel_value(id);
            ThreadTraceLockGuard lock(value->mutex, "runtime.parallel_value");
            if (value->uses.remaining == 0)
                throw std::runtime_error("parallel submission count underflow for ValueId " + std::to_string(id));
            --value->uses.remaining;
            if (can_clear(value->uses)) value->value.reset();
        }
    }

    std::vector<ValueDesc> communication_output_descs(
        const CommAction &action) const {
        std::vector<ValueDesc> result;
        result.reserve(action.outputs.size());
        for (ValueId id : action.outputs) result.push_back(desc(id));
        return result;
    }

    std::vector<std::optional<Value>> posted_outputs(
        const CommAction &action, CommHandle &handle) {
        if constexpr (detail::HasPostedOutputs<Api>::value) {
            auto outputs = api_.posted_outputs(handle);
            if (outputs.size() != action.outputs.size())
                throw std::runtime_error(
                    "Api posted_outputs returned wrong output count");
            for (std::size_t slot = 0; slot < outputs.size(); ++slot)
                if (action.destinations[slot].rank != rank_ && outputs[slot])
                    throw std::runtime_error("Api posted_outputs returned a remote output");
            return outputs;
        }
        return std::vector<std::optional<Value>>(action.outputs.size());
    }

    void load_bundle(const RuntimeResources &resources) {
        if (!plan_->plaintext_bundle) return;
        if (!resources.plaintext_bundle_dir)
            throw std::runtime_error("RuntimePlan requires a plaintext bundle directory");
        std::vector<std::string> local_contents;
        for (const auto *phase : {&plan_->initialization, &plan_->execution})
        for (const auto &instruction : *phase) {
            const auto *encode = std::get_if<EncodeOp>(&instruction.body);
            if (!encode || desc(encode->output).place.rank != rank_) continue;
            if (const auto *payload = std::get_if<BundleEncodePayload>(&encode->payload))
                local_contents.push_back(payload->content);
        }
        if (plan_->format_version == 3) {
            bundle_index_ = PlaintextBundleLoader::open(*resources.plaintext_bundle_dir,
                *plan_->plaintext_bundle, local_contents,
                resources.operator_spec.spec.poly_degree / 2,
                resources.skip_artifact_digest_checks, resources.plaintext_bundle_options);
            timing_.bundle_resident_bytes = bundle_index_->resident_bytes();
            timing_.bundle_resident_load_nanoseconds = static_cast<std::uint64_t>(bundle_index_->resident_load_seconds() * 1e9);
            return;
        }
        auto bundle = PlaintextBundleLoader::load(*resources.plaintext_bundle_dir,
                                                   *plan_->plaintext_bundle,
                                                   local_contents,
                                                   resources.operator_spec.spec.poly_degree / 2,
                                                   resources.skip_artifact_digest_checks, resources.plaintext_bundle_options);
        bundle_slots_ = std::move(bundle.slots_by_content);
    }

    void bind_inputs(const std::unordered_map<ValueId, Value> &inputs) {
        std::size_t expected = 0;
        for (ValueId id : plan_->external_inputs) {
            const auto &expected_desc = desc(id);
            if (expected_desc.place.rank != rank_) continue;
            ++expected;
            auto it = inputs.find(id);
            if (it == inputs.end()) throw std::runtime_error("missing local external input " + std::to_string(id));
            api_.validate_value(it->second, expected_desc);
            store_.define_ready(id, expected_desc.place, it->second);
        }
        if (inputs.size() != expected) throw std::runtime_error("unexpected local external input binding");
    }

    void execute_phase(const std::vector<Instruction> &instructions) {
        for (const auto &instruction : instructions) {
            current_ = &instruction;
            if (const auto *encode = std::get_if<EncodeOp>(&instruction.body)) execute_encode(*encode);
            else if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) execute_compute(*op);
            else if (const auto *action = std::get_if<CommAction>(&instruction.body)) execute_communication(*action);
            else if (std::holds_alternative<FenceOp>(instruction.body)) execute_fence();
            else execute_release(std::get<ReleaseOp>(instruction.body).value,
                                 instruction.ordinal, false);
        }
        current_ = nullptr;
    }

    Value encode_value(const EncodeOp &op, std::uint64_t ordinal) {
        const auto &output_desc = desc(op.output);
        std::vector<double> slots;
        if (const auto *inline_payload = std::get_if<InlineEncodePayload>(&op.payload))
            slots = inline_payload->values;
        else {
            const auto &content = std::get<BundleEncodePayload>(op.payload).content;
            const auto start = std::chrono::steady_clock::now();
            slots = bundle_index_ ? bundle_index_->read(content) : bundle_slots_.at(content);
            if (bundle_index_) {
                const auto elapsed = elapsed_nanoseconds(start, std::chrono::steady_clock::now());
                ++timing_.bundle_read_calls;
                timing_.bundle_read_nanoseconds += elapsed;
                timing_.bundle_read_bytes += slots.size() * sizeof(double);
                timing_.raw_peak_bytes = std::max(timing_.raw_peak_bytes,
                    static_cast<std::uint64_t>(slots.size() * sizeof(double) * 2));
                trace_event("bundle_read", op.output, ordinal, std::nullopt, "Read", elapsed);
            }
        }
        for (double &value : slots) if (value == 0.0) value = 0.0;
        const auto start = std::chrono::steady_clock::now();
        Value output = api_.encode_plaintext(output_desc, slots);
        api_.validate_value(output, output_desc);
        const auto elapsed = elapsed_nanoseconds(start, std::chrono::steady_clock::now());
        ++timing_.encode_calls;
        timing_.encode_nanoseconds += elapsed;
        trace_event("encode", op.output, ordinal, std::nullopt, "Encode", elapsed);
        return output;
    }

    void execute_encode(const EncodeOp &op) {
        const auto &output_desc = desc(op.output);
        if (output_desc.place.rank != rank_) return;
        store_.define_ready(op.output, output_desc.place, encode_value(op, current_->ordinal));
    }

    void execute_fence() {
        const auto start = std::chrono::steady_clock::now();
        finish_all_groups();
        api_.drain();
        record_fence(current_ ? current_->ordinal : 0, start);
    }

    void record_fence(std::uint64_t ordinal, std::chrono::steady_clock::time_point start) {
        const auto elapsed = elapsed_nanoseconds(start, std::chrono::steady_clock::now());
        ++timing_.fence_calls;
        timing_.fence_nanoseconds += elapsed;
        trace_event("fence", 0, ordinal, std::nullopt, "Fence", elapsed);
    }

    Value &ensure_ready(ValueId id, const Place &expected_place) {
        auto &entry = store_.lookup(id);
        if (std::holds_alternative<typename ValueStore<Api>::Released>(entry))
            throw std::runtime_error("ValueId was released: " + std::to_string(id));
        if (auto *ready = std::get_if<typename ValueStore<Api>::Ready>(&entry)) {
            if (ready->place != expected_place) throw std::runtime_error("Ready value Place mismatch for " + std::to_string(id));
            return ready->value;
        }
        const auto pending = std::get<typename ValueStore<Api>::Pending>(entry);
        if (pending.place != expected_place) throw std::runtime_error("Pending value Place mismatch for " + std::to_string(id));
        finish_group(pending.group);
        auto &ready = std::get<typename ValueStore<Api>::Ready>(store_.lookup(id));
        return ready.value;
    }

    void execute_compute(const ComputeOp &op) {
        if (op.place.rank != rank_) return;
        std::vector<Value> inputs;
        for (ValueId id : op.inputs) inputs.push_back(ensure_ready(id, op.place));
        const auto start = std::chrono::steady_clock::now();
        if (op.reuse_input) {
            auto &entry = store_.lookup(op.inputs.front());
            entry = typename ValueStore<Api>::Released{op.place};
        }
        Value output = compute_output(op, inputs);
        const auto finish = std::chrono::steady_clock::now();
        inputs.clear();
        submitted_sequential_uses(op.inputs);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            finish - start);
        const auto elapsed_nanoseconds =
            static_cast<std::uint64_t>(elapsed.count());
        trace_event("compute", op.output, current_->ordinal, std::nullopt,
                    to_string(op.kind), elapsed_nanoseconds);
        ++timing_.compute_calls;
        timing_.compute_including_boot_nanoseconds += elapsed_nanoseconds;
        if (op.kind == ComputeKind::Boot) {
            ++timing_.boot_calls;
            timing_.boot_nanoseconds += elapsed_nanoseconds;
        }
        const auto &output_desc = desc(op.output);
        api_.validate_value(output, output_desc);
        store_.define_ready(op.output, op.place, std::move(output));
    }

    Value compute_output(const ComputeOp &op, std::vector<Value> &inputs) {
        if (!op.reuse_input) return api_.compute(op, inputs);
        if constexpr (detail::HasComputeReuse<Api>::value) {
            Value reused = std::move(inputs.front());
            inputs.erase(inputs.begin());
            return api_.compute_reuse(op, std::move(reused), inputs);
        } else {
            throw std::runtime_error("Api does not support reuse_input");
        }
    }

    void execute_communication(const CommAction &action) {
        std::vector<Value> local_inputs;
        for (std::size_t i = 0; i < action.inputs.size(); ++i)
            if (action.sources[i].rank == rank_) local_inputs.push_back(ensure_ready(action.inputs[i], action.sources[i]));
        PendingGroup group;
        group.action = action;
        group.instruction_ordinal = current_->ordinal;
        const auto post_start = std::chrono::steady_clock::now();
        group.handle = api_.communicate_async(
            action, local_inputs, communication_output_descs(action));
        auto posted = posted_outputs(action, group.handle);
        const auto post_elapsed = elapsed_nanoseconds(
            post_start, std::chrono::steady_clock::now());
        ++timing_.communication_post_calls;
        timing_.communication_post_nanoseconds += post_elapsed;
        trace_event("comm_post", action.id, group.instruction_ordinal,
                    std::nullopt, to_string(action.kind), post_elapsed);
        const std::size_t group_id = groups_.size();
        for (std::size_t i = 0; i < action.outputs.size(); ++i) {
            if (action.destinations[i].rank != rank_) continue;
            const std::size_t local_slot = group.local_output_ids.size();
            group.local_output_ids.push_back(action.outputs[i]);
            group.local_output_slots.push_back(i);
            group.local_output_posted.push_back(posted[i].has_value());
            if (posted[i]) {
                const auto &expected_desc = desc(action.outputs[i]);
                api_.validate_value(*posted[i], expected_desc);
                store_.define_ready(action.outputs[i], action.destinations[i],
                                    std::move(*posted[i]));
            } else {
                store_.define_pending(action.outputs[i], action.destinations[i],
                                      group_id, local_slot);
            }
        }
        groups_.push_back(std::move(group));
        local_inputs.clear();
        std::vector<ValueId> local_ids;
        for (std::size_t i = 0; i < action.inputs.size(); ++i)
            if (action.sources[i].rank == rank_) local_ids.push_back(action.inputs[i]);
        submitted_sequential_uses(local_ids);
    }

    void finish_group(std::size_t group_id) {
        auto &group = groups_.at(group_id);
        if (group.completed) return;
        const auto wait_start = std::chrono::steady_clock::now();
        const std::uint64_t trace_wait_start =
            ThreadTrace::enabled() ? ThreadTrace::timestamp_ns() : 0;
        auto outputs = api_.wait(group.handle);
        if (ThreadTrace::enabled())
            ThreadTrace::record_duration(
                "runtime.communication_wait", &group, trace_wait_start,
                ThreadTrace::timestamp_ns());
        const auto wait_elapsed = elapsed_nanoseconds(
            wait_start, std::chrono::steady_clock::now());
        ++timing_.communication_wait_calls;
        timing_.communication_wait_nanoseconds += wait_elapsed;
        trace_event("comm_wait", group.action.id, group.instruction_ordinal,
                    current_ ? std::optional<std::uint64_t>(current_->ordinal)
                             : std::nullopt,
                    to_string(group.action.kind), wait_elapsed);
        if (outputs.size() != group.action.outputs.size()) throw std::runtime_error("Api wait returned wrong output count");
        for (std::size_t slot = 0; slot < outputs.size(); ++slot)
            if (group.action.destinations[slot].rank != rank_ && outputs[slot])
                throw std::runtime_error("Api wait returned a remote output");
        for (std::size_t i = 0; i < group.local_output_ids.size(); ++i) {
            const ValueId id = group.local_output_ids[i];
            auto &output = outputs[group.local_output_slots[i]];
            if (group.local_output_posted[i]) {
                if (output) throw std::runtime_error("Api delivered a communication output twice");
                continue;
            }
            if (!output) throw std::runtime_error("Api wait omitted a pending communication output");
            const auto &expected_desc = desc(id);
            api_.validate_value(*output, expected_desc);
            if (can_clear(uses_.at(id))) {
                output.reset();
                store_.lookup(id) = typename ValueStore<Api>::Released{expected_desc.place};
            } else {
                store_.lookup(id) = typename ValueStore<Api>::Ready{
                    expected_desc.place, std::move(*output)};
            }
        }
        group.completed = true;
    }

    std::shared_ptr<ParallelValue> parallel_value(ValueId id) const {
        const auto found = parallel_values_.find(id);
        if (found == parallel_values_.end())
            throw std::runtime_error(
                "ValueId is absent from parallel ValueStore: " +
                std::to_string(id));
        return found->second;
    }

    std::shared_ptr<ParallelValue> define_parallel_value(
        ValueId id, const Place &place) {
        auto value = std::make_shared<ParallelValue>();
        value->place = place;
        value->uses = uses_.at(id);
        if (!parallel_values_.emplace(id, value).second)
            throw std::runtime_error(
                "ValueId defined twice in parallel ValueStore: " +
                std::to_string(id));
        return value;
    }

    void prepare_parallel_values() {
        if (return_all_values_)
            throw std::runtime_error(
                "per-device worker execution does not support AllValuesAfterRun");
        if (local_devices_ <= 0)
            throw std::runtime_error(
                "per-device worker execution requires at least one local device");

        for (const auto &item : store_.entries()) {
            const auto *ready =
                std::get_if<typename ValueStore<Api>::Ready>(&item.second);
            if (std::holds_alternative<typename ValueStore<Api>::Released>(item.second)) {
                auto value = define_parallel_value(item.first, desc(item.first).place);
                value->defined = true;
                continue;
            }
            if (ready == nullptr)
                throw std::runtime_error(
                    "initialization left a pending value before parallel execution");
            auto value = define_parallel_value(item.first, ready->place);
            value->value.emplace(ready->value);
            value->defined = true;
        }
        store_ = ValueStore<Api>{};
    }

    void reset_parallel_execution_state() {
        parallel_values_.clear();
        parallel_groups_.clear();
        coordinator_groups_.clear();
        parallel_failed_.store(false);
        parallel_failure_ = nullptr;
    }

    void record_parallel_failure(std::exception_ptr failure) {
        bool expected = false;
        if (!parallel_failed_.compare_exchange_strong(expected, true)) return;
        {
            ThreadTraceLockGuard lock(
                parallel_failure_mutex_, "runtime.parallel_failure");
            parallel_failure_ = std::move(failure);
        }
        for (const auto &item : parallel_values_)
            item.second->condition.notify_all();
    }

    void finish_parallel_group(
        const std::shared_ptr<ParallelGroup> &group,
        std::optional<std::uint64_t> consumer_ordinal) {
        ThreadTraceUniqueLock lock(
            group->mutex, "runtime.parallel_communication_group");
        if (group->completed) return;
        if (!group->posted)
            throw std::runtime_error(
                "parallel communication was waited before it was posted");

        const auto wait_start = std::chrono::steady_clock::now();
        const std::uint64_t trace_wait_start =
            ThreadTrace::enabled() ? ThreadTrace::timestamp_ns() : 0;
        auto outputs = api_.wait(group->handle);
        if (ThreadTrace::enabled())
            ThreadTrace::record_duration(
                "runtime.communication_wait", group.get(), trace_wait_start,
                ThreadTrace::timestamp_ns());
        const auto wait_elapsed = elapsed_nanoseconds(
            wait_start, std::chrono::steady_clock::now());
        {
            ThreadTraceLockGuard timing_lock(
                parallel_timing_mutex_, "runtime.parallel_timing");
            ++timing_.communication_wait_calls;
            timing_.communication_wait_nanoseconds += wait_elapsed;
        }
        trace_event("comm_wait", group->action.id,
                    group->instruction_ordinal, consumer_ordinal,
                    to_string(group->action.kind), wait_elapsed);
        if (outputs.size() != group->action.outputs.size())
            throw std::runtime_error("Api wait returned wrong parallel output count");
        for (std::size_t slot = 0; slot < outputs.size(); ++slot)
            if (group->action.destinations[slot].rank != rank_ && outputs[slot])
                throw std::runtime_error("Api wait returned a remote parallel output");

        for (const auto &local_output : group->local_outputs) {
            const ValueId id = local_output.id;
            if (local_output.action_slot >= group->action.outputs.size() ||
                group->action.outputs[local_output.action_slot] != id)
                throw std::runtime_error(
                    "parallel communication output slot mapping is invalid");
            const auto &expected_desc = desc(id);
            auto &output = outputs[local_output.action_slot];
            if (output) api_.validate_value(*output, expected_desc);
            const auto &value = local_output.value;
            {
                ThreadTraceLockGuard value_lock(
                    value->mutex, "runtime.parallel_value");
                if (!value->defined)
                    throw std::runtime_error(
                        "parallel communication output state is invalid");
                if (!value->group) {
                    if (output)
                        throw std::runtime_error("Api delivered a parallel communication output twice");
                } else {
                    if (!output)
                        throw std::runtime_error("Api wait omitted a pending parallel communication output");
                    if (value->group.get() != group.get())
                        throw std::runtime_error(
                            "pending communication output group is invalid");
                    if (!can_clear(value->uses))
                        value->value.emplace(std::move(*output));
                    output.reset();
                    value->group.reset();
                }
            }
            value->condition.notify_all();
        }
        group->completed = true;
    }

    Value resolve_parallel_value(
        ValueId id, const Place &expected_place,
        std::optional<std::uint64_t> consumer_ordinal) {
        auto value = parallel_value(id);
        for (;;) {
            std::shared_ptr<ParallelGroup> group;
            {
                ThreadTraceUniqueLock lock(
                    value->mutex, "runtime.parallel_value");
                lock.wait(value->condition, [&] {
                    return value->defined || parallel_failed_.load();
                }, "runtime.value_dependency_wait");
                if (parallel_failed_.load())
                    throw std::runtime_error("parallel execution was cancelled");
                if (value->place != expected_place)
                    throw std::runtime_error(
                        "parallel value Place mismatch for " +
                        std::to_string(id));
                if (value->value) return *value->value;
                if (can_clear(value->uses))
                    throw std::runtime_error("parallel ValueId was released: " + std::to_string(id));
                group = value->group;
            }
            if (!group)
                throw std::runtime_error(
                    "parallel value is defined without data or communication");
            if (group->cross_rank) {
                ThreadTraceUniqueLock lock(
                    value->mutex, "runtime.remote_value_wait");
                lock.wait(value->condition, [&] {
                    return value->value.has_value() ||
                           parallel_failed_.load();
                }, "runtime.remote_value_wait");
                if (parallel_failed_.load())
                    throw std::runtime_error(
                        "parallel execution was cancelled");
                continue;
            }
            finish_parallel_group(group, consumer_ordinal);
        }
    }

    void execute_parallel_compute(
        ComputeOp op, std::uint64_t instruction_ordinal,
        const std::shared_ptr<ParallelValue> &output_value) {
        std::vector<Value> inputs;
        inputs.reserve(op.inputs.size());
        for (ValueId id : op.inputs)
            inputs.push_back(resolve_parallel_value(
                id, op.place, instruction_ordinal));

        const auto start = std::chrono::steady_clock::now();
        const std::uint64_t trace_compute_start =
            ThreadTrace::enabled() ? ThreadTrace::timestamp_ns() : 0;
        if (op.reuse_input) {
            const auto &old = parallel_values_.at(op.inputs.front());
            ThreadTraceLockGuard lock(old->mutex, "runtime.parallel_value");
            old->value.reset();
            old->uses.release_requested = true;
        }
        Value output = compute_output(op, inputs);
        const auto finish = std::chrono::steady_clock::now();
        inputs.clear();
        submitted_parallel_uses(op.inputs);
        if (ThreadTrace::enabled())
            ThreadTrace::record_duration(
                "runtime.compute", output_value.get(), trace_compute_start,
                ThreadTrace::timestamp_ns());
        const auto elapsed = elapsed_nanoseconds(
            start, finish);
        trace_event("compute", op.output, instruction_ordinal,
                    std::nullopt, to_string(op.kind), elapsed);
        {
            ThreadTraceLockGuard lock(
                parallel_timing_mutex_, "runtime.parallel_timing");
            ++timing_.compute_calls;
            timing_.compute_including_boot_nanoseconds += elapsed;
            if (op.kind == ComputeKind::Boot) {
                ++timing_.boot_calls;
                timing_.boot_nanoseconds += elapsed;
            }
        }

        const auto &output_desc = desc(op.output);
        api_.validate_value(output, output_desc);
        {
            ThreadTraceLockGuard lock(
                output_value->mutex, "runtime.parallel_value");
            if (!can_clear(output_value->uses))
                output_value->value.emplace(std::move(output));
            output_value->defined = true;
        }
        output_value->condition.notify_all();
    }

    void execute_parallel_communication(
        CommAction action, std::uint64_t instruction_ordinal,
        const std::shared_ptr<ParallelGroup> &group) {
        std::vector<Value> local_inputs;
        for (std::size_t i = 0; i < action.inputs.size(); ++i) {
            if (action.sources[i].rank != rank_) continue;
            local_inputs.push_back(resolve_parallel_value(
                action.inputs[i], action.sources[i], instruction_ordinal));
        }

        const auto post_start = std::chrono::steady_clock::now();
        CommHandle handle = api_.communicate_async(
            action, local_inputs, communication_output_descs(action));
        auto posted = posted_outputs(action, handle);
        const auto post_elapsed = elapsed_nanoseconds(
            post_start, std::chrono::steady_clock::now());
        {
            ThreadTraceLockGuard lock(
                parallel_timing_mutex_, "runtime.parallel_timing");
            ++timing_.communication_post_calls;
            timing_.communication_post_nanoseconds += post_elapsed;
        }
        trace_event("comm_post", action.id, instruction_ordinal,
                    std::nullopt, to_string(action.kind), post_elapsed);

        {
            ThreadTraceLockGuard lock(
                group->mutex, "runtime.parallel_communication_group");
            group->handle = std::move(handle);
            group->posted = true;
        }

        for (const auto &local_output : group->local_outputs) {
            const auto &value = local_output.value;
            auto &output = posted.at(local_output.action_slot);
            if (group->cross_rank &&
                detail::UsesBackgroundCommunicationIssuer<Api>::value &&
                !output)
                throw std::runtime_error(
                    "background communication did not publish a local output");
            {
                ThreadTraceLockGuard lock(
                    value->mutex, "runtime.parallel_value");
                if (output) {
                    const auto &expected_desc = desc(local_output.id);
                    api_.validate_value(*output, expected_desc);
                    if (!can_clear(value->uses))
                        value->value.emplace(std::move(*output));
                    output.reset();
                } else {
                    value->group = group;
                }
                value->defined = true;
            }
            value->condition.notify_all();
        }
        local_inputs.clear();
        submitted_parallel_uses(group->local_input_ids);
    }

    bool is_cross_rank_action(const CommAction &action) const {
        if (action.sources.empty())
            throw std::runtime_error("parallel communication has no source");
        const int source_rank = action.sources.front().rank;
        return std::any_of(
            action.destinations.begin(), action.destinations.end(),
            [source_rank](const Place &place) {
                return place.rank != source_rank;
            });
    }

    std::shared_ptr<ParallelGroup> make_parallel_group(
        CommAction action, std::uint64_t instruction_ordinal,
        bool cross_rank) {
        auto group = std::make_shared<ParallelGroup>();
        group->action = std::move(action);
        group->instruction_ordinal = instruction_ordinal;
        group->cross_rank = cross_rank;
        for (std::size_t i = 0; i < group->action.inputs.size(); ++i)
            if (group->action.sources[i].rank == rank_)
                group->local_input_ids.push_back(group->action.inputs[i]);
        for (std::size_t i = 0; i < group->action.outputs.size(); ++i) {
            if (group->action.destinations[i].rank != rank_) continue;
            group->local_outputs.push_back(ParallelOutput{
                group->action.outputs[i], i,
                define_parallel_value(group->action.outputs[i],
                                      group->action.destinations[i])});
        }
        parallel_groups_.push_back(group);
        return group;
    }

    int local_communication_worker(const CommAction &action) const {
        if (action.outputs.size() != 1 || action.destinations.size() != 1 ||
            action.sources.size() != 1)
            throw std::runtime_error(
                "local parallel communication must have one destination");
        const Place &destination = action.destinations.front();
        const Place &source = action.sources.front();
        // Device-to-Device copies follow the producer so the destination
        // worker can keep executing until it consumes the copied value.
        const Place *device_place =
            source.kind == PlaceKind::Device ? &source :
            destination.kind == PlaceKind::Device ? &destination : nullptr;
        if (device_place == nullptr || device_place->rank != rank_ ||
            device_place->index < 0 || device_place->index >= local_devices_)
            throw std::runtime_error(
                "local parallel communication has no available device worker");
        return device_place->index;
    }

    CommAction local_communication_slice(const CommAction &action,
                                         std::size_t slot) const {
        if (action.inputs.size() != 1 || action.sources.size() != 1 ||
            slot >= action.outputs.size() ||
            slot >= action.destinations.size() ||
            slot >= action.output_types.size())
            throw std::runtime_error(
                "local parallel communication mapping is invalid");
        CommAction result;
        result.id = action.id;
        result.kind = CommKind::Transfer;
        result.hint = action.hint;
        result.inputs = action.inputs;
        result.sources = action.sources;
        result.outputs = {action.outputs[slot]};
        result.destinations = {action.destinations[slot]};
        result.output_types = {action.output_types[slot]};
        return result;
    }

    CommAction cross_rank_communication_slice(
        const CommAction &action) const {
        if (action.inputs.size() != 1 || action.sources.size() != 1 ||
            action.outputs.size() != action.destinations.size() ||
            action.outputs.size() != action.output_types.size())
            throw std::runtime_error(
                "cross-rank parallel communication mapping is invalid");
        CommAction result;
        result.id = action.id;
        result.hint = action.hint;
        result.inputs = action.inputs;
        result.sources = action.sources;
        const int source_rank = action.sources.front().rank;
        for (std::size_t slot = 0; slot < action.outputs.size(); ++slot) {
            if (action.destinations[slot].rank == source_rank) continue;
            result.outputs.push_back(action.outputs[slot]);
            result.destinations.push_back(action.destinations[slot]);
            result.output_types.push_back(action.output_types[slot]);
        }
        if (result.outputs.empty())
            throw std::runtime_error(
                "cross-rank parallel communication has no remote destination");
        result.kind = result.outputs.size() == 1
            ? CommKind::Transfer : CommKind::Replicate;
        return result;
    }

    void compile_parallel_phase(
        const std::vector<Instruction> &instructions,
        std::vector<std::vector<ParallelTask>> &tasks,
        std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const auto &instruction = instructions[i];
            if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
                if (op->place.rank != rank_) continue;
                if (op->place.kind == PlaceKind::Host && plan_->format_version < 3)
                    throw std::runtime_error("per-device worker Host compute requires V3");
                if (op->place.kind == PlaceKind::Device &&
                    (op->place.index < 0 || op->place.index >= local_devices_))
                    throw std::runtime_error(
                        "per-device worker execution requires Device compute operations");
                auto output = define_parallel_value(
                    op->output, desc(op->output).place);
                const std::size_t worker =
                    op->place.kind == PlaceKind::Host ? tasks.size() - 1 :
                    static_cast<std::size_t>(op->place.index) % (tasks.size() - 1);
                tasks[worker].push_back(
                    [this, op = *op, ordinal = instruction.ordinal, output] {
                        execute_parallel_compute(op, ordinal, output);
                    });
                continue;
            }
            if (const auto *encode = std::get_if<EncodeOp>(&instruction.body)) {
                if (desc(encode->output).place.rank != rank_) continue;
                auto output = define_parallel_value(encode->output, desc(encode->output).place);
                tasks.back().push_back([this, op = *encode, ordinal = instruction.ordinal, output] {
                    auto encoded = encode_value(op, ordinal);
                    {
                        ThreadTraceLockGuard lock(output->mutex, "runtime.parallel_value");
                        if (!can_clear(output->uses)) output->value.emplace(std::move(encoded));
                        output->defined = true;
                    }
                    output->condition.notify_all();
                });
                continue;
            }
            if (const auto *release = std::get_if<ReleaseOp>(&instruction.body)) {
                const auto &place = desc(release->value).place;
                if (place.rank != rank_) continue;
                const std::size_t worker = place.kind == PlaceKind::Device
                    ? static_cast<std::size_t>(place.index) % (tasks.size() - 1) : tasks.size() - 1;
                tasks[worker].push_back([this, id = release->value, ordinal = instruction.ordinal] {
                    execute_release(id, ordinal, true);
                });
                continue;
            }

            const auto action = std::get<CommAction>(instruction.body);
            const bool has_cross_rank_destination =
                is_cross_rank_action(action);
            if (has_cross_rank_destination) {
                const auto device_endpoint = [](const Place &place) {
                    return place.kind == PlaceKind::Device;
                };
                if (!std::all_of(action.sources.begin(), action.sources.end(),
                                 device_endpoint) ||
                    !std::all_of(action.destinations.begin(),
                                 action.destinations.end(), device_endpoint))
                    throw std::runtime_error(
                        "multi-rank device workers require Device-only online communication");
                auto cross_rank_action =
                    cross_rank_communication_slice(action);
                coordinator_groups_.push_back(make_parallel_group(
                    std::move(cross_rank_action), instruction.ordinal, true));
            }
            if (action.sources.empty() ||
                action.sources.front().rank != rank_) continue;
            for (std::size_t slot = 0; slot < action.outputs.size(); ++slot) {
                if (has_cross_rank_destination &&
                    action.destinations[slot].rank != rank_)
                    continue;
                CommAction local = local_communication_slice(action, slot);
                auto group = make_parallel_group(
                    local, instruction.ordinal, false);
                const int device = local_communication_worker(local);
                const std::size_t worker =
                    static_cast<std::size_t>(device) % (tasks.size() - 1);
                tasks[worker].push_back(
                    [this, local = std::move(local),
                     ordinal = instruction.ordinal, group] {
                        execute_parallel_communication(
                            local, ordinal, group);
                    });
            }
        }
    }

    void execute_device_parallel_phases() {
        prepare_parallel_values();
        const std::size_t worker_count =
            device_worker_count_ == 0
                ? static_cast<std::size_t>(local_devices_)
                : device_worker_count_;
        if (worker_count == 0 ||
            worker_count > static_cast<std::size_t>(local_devices_))
            throw std::runtime_error(
                "device worker count must be between one and local device count");
        for (const auto *phase : {&plan_->execution, &plan_->finalization}) {
            std::size_t begin = 0;
            for (std::size_t end = 0; end <= phase->size(); ++end) {
                if (end != phase->size() && !std::holds_alternative<FenceOp>((*phase)[end].body)) continue;
                std::vector<std::vector<ParallelTask>> tasks(worker_count + 1);
                compile_parallel_phase(*phase, tasks, begin, end);
                const auto start = std::chrono::steady_clock::now();
                run_parallel_batch(tasks);
                if (end != phase->size()) {
                    api_.drain();
                    record_fence((*phase)[end].ordinal, start);
                }
                begin = end + 1;
            }
        }
        for (ValueId id : plan_->final_outputs) {
            const auto &output_desc = desc(id);
            if (output_desc.place.rank != rank_) continue;
            Value output = resolve_parallel_value(id, output_desc.place, std::nullopt);
            store_.define_ready(id, output_desc.place, std::move(output));
        }
    }

    void run_parallel_batch(std::vector<std::vector<ParallelTask>> &tasks) {
        std::vector<std::thread> workers;
        workers.reserve(tasks.size());
        for (std::size_t device = 0; device < tasks.size(); ++device) {
            if (tasks[device].empty()) continue;
            workers.emplace_back([this, &tasks, device] {
                ThreadTrace::set_thread_name(
                    "rank-" + std::to_string(rank_) +
                    (device + 1 == tasks.size() ? "-cpu-worker" :
                     "-device-worker-" + std::to_string(device)));
                try {
                    for (auto &task : tasks[device]) {
                        if (parallel_failed_.load()) return;
                        task();
                    }
                } catch (...) {
                    record_parallel_failure(std::current_exception());
                }
            });
        }
        std::thread communication_issuer;
        if constexpr (detail::UsesBackgroundCommunicationIssuer<Api>::value) {
            communication_issuer = std::thread([this] {
                ThreadTrace::set_thread_name(
                    "rank-" + std::to_string(rank_) +
                    "-communication-issuer");
                try {
                    for (const auto &group : coordinator_groups_) {
                        if (parallel_failed_.load()) return;
                        execute_parallel_communication(
                            group->action, group->instruction_ordinal, group);
                    }
                } catch (...) {
                    record_parallel_failure(std::current_exception());
                }
            });
        } else {
            try {
                for (const auto &group : coordinator_groups_) {
                    if (parallel_failed_.load()) break;
                    execute_parallel_communication(
                        group->action, group->instruction_ordinal, group);
                    finish_parallel_group(group, std::nullopt);
                }
            } catch (...) {
                record_parallel_failure(std::current_exception());
            }
        }
        for (auto &worker : workers) worker.join();
        if (communication_issuer.joinable()) communication_issuer.join();
        if (parallel_failed_.load()) {
            ThreadTraceLockGuard lock(
                parallel_failure_mutex_, "runtime.parallel_failure");
            std::rethrow_exception(parallel_failure_);
        }

        for (const auto &group : parallel_groups_)
            if (!group->completed)
                finish_parallel_group(group, std::nullopt);

        parallel_groups_.clear();
        coordinator_groups_.clear();
    }

    void finish_all_groups() {
        for (std::size_t i = 0; i < groups_.size(); ++i) finish_group(i);
        // finish_group replaces every Pending entry before indices are reused.
        groups_.clear();
    }

    void synchronize_final_outputs() {
        for (ValueId id : plan_->final_outputs) {
            const auto &output_desc = desc(id);
            if (output_desc.place.rank != rank_) continue;
            current_value_ = id;
            api_.synchronize(ensure_ready(id, output_desc.place));
        }
        current_value_.reset();
    }

    RunArtifact<Value> make_artifact(DiffMode mode) {
        RunArtifact<Value> result;
        result.timing = timing_;
        if (mode == DiffMode::AllValuesAfterRun) {
            for (const auto &item : store_.entries())
                if (const auto *ready = std::get_if<typename ValueStore<Api>::Ready>(&item.second))
                    result.values.emplace(item.first, ArtifactValue<Value>{ready->place, ready->value});
        } else {
            for (ValueId id : plan_->final_outputs) {
                const auto &output_desc = desc(id);
                if (output_desc.place.rank != rank_) continue;
                Value &value = ensure_ready(id, output_desc.place);
                result.values.emplace(id, ArtifactValue<Value>{output_desc.place, value});
            }
        }
        return result;
    }

    std::string format_error(const std::string &reason) const {
        std::ostringstream out;
        out << "fatal runtime error: reason=" << reason
            << " plan_id=" << (plan_ ? plan_->plan_id : 0)
            << " plan_source_sha256=" << plan_source_sha256_
            << " local_rank=" << rank_ << " api=" << api_.name();
        if (current_value_) out << " value_id=" << *current_value_;
        if (current_) {
            out << " op_ordinal=" << current_->ordinal;
            if (const auto *encode = std::get_if<EncodeOp>(&current_->body))
                out << " op=Encode output=" << encode->output;
            else if (const auto *op = std::get_if<ComputeOp>(&current_->body))
                out << " op=" << to_string(op->kind) << " output=" << op->output << " place=" << to_string(op->place);
            else if (std::holds_alternative<FenceOp>(current_->body))
                out << " op=Fence";
            else if (const auto *release = std::get_if<ReleaseOp>(&current_->body))
                out << " op=Release value_id=" << release->value;
            else {
                const auto &action = std::get<CommAction>(current_->body);
                out << " op=" << to_string(action.kind) << " transfer_id=" << action.id;
            }
        }
        return out.str();
    }

    int rank_;
    int world_size_;
    int local_devices_;
    Api &api_;
    DeviceExecutionMode device_execution_mode_;
    std::size_t device_worker_count_ = 0;
    const RuntimePlan *plan_ = nullptr;
    std::string plan_source_sha256_;
    const Instruction *current_ = nullptr;
    std::optional<ValueId> current_value_;
    ValueStore<Api> store_;
    std::vector<PendingGroup> groups_;
    std::unordered_map<ValueId, ValueUses> uses_;
    std::map<std::string, std::vector<double>> bundle_slots_;
    std::optional<PlaintextBundleLoader> bundle_index_;
    std::unordered_map<ValueId, const ValueDesc *> descriptors_;
    std::unordered_map<ValueId, std::shared_ptr<ParallelValue>> parallel_values_;
    std::vector<std::shared_ptr<ParallelGroup>> parallel_groups_;
    std::vector<std::shared_ptr<ParallelGroup>> coordinator_groups_;
    RuntimeTiming timing_;
    std::unique_ptr<std::ofstream> runtime_trace_;
    std::mutex trace_mutex_;
    std::mutex parallel_timing_mutex_;
    std::atomic<bool> parallel_failed_{false};
    std::mutex parallel_failure_mutex_;
    std::exception_ptr parallel_failure_;
    bool return_all_values_ = false;
};

} // namespace fhegpu
