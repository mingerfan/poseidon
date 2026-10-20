#pragma once

#include "api/vec_value.hpp"
#include "runtime/plan_reader.hpp"
#include "runtime/operator_spec_reader.hpp"
#include "runtime/plaintext_bundle.hpp"
#include "runtime/runtime.hpp"
#include "runtime/utils/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fhegpu::experiment {

struct Context {
    LoadedRuntimePlan loaded_plan;
    LoadedOperatorSpec operator_spec;
    std::filesystem::path bundle_dir;
    ValueId input_id = 0;
    VecValue input;
    std::size_t slot_capacity = 0;
};

struct DiffSummary {
    std::size_t compared_lines = 0;
    std::size_t mismatch_lines = 0;
    std::size_t mismatch_slots = 0;
    double max_abs_diff = 0.0;
    std::string final_sha256;
    std::vector<double> final_prefix;
};

inline const ValueDesc &find_value(const RuntimePlan &plan, ValueId id) {
    for (const auto &value : plan.values)
        if (value.id == id) return value;
    throw std::runtime_error("missing ValueDesc " + std::to_string(id));
}

inline std::vector<double> make_input(std::size_t slot_count) {
    std::vector<double> input(slot_count, 0.0);
    for (std::size_t i = 0; i < 784; ++i)
        input[i] = static_cast<double>((i * 17 + 3) % 256) / 255.0;
    return input;
}

inline std::vector<double> pack_mlp_input(const std::vector<double> &logical) {
    if (logical.size() != 784)
        throw std::runtime_error("MLP logical input must contain 784 values");
    std::vector<double> packed;
    packed.reserve(1600);
    for (std::size_t block_index = 0; block_index < 8; ++block_index) {
        std::vector<double> block(100, 0.0);
        const std::size_t begin = block_index * 100;
        const std::size_t count = std::min<std::size_t>(100, logical.size() - begin);
        std::copy_n(logical.begin() + begin, count, block.begin());
        packed.insert(packed.end(), block.begin(), block.end());
        packed.insert(packed.end(), block.begin(), block.end());
    }
    return packed;
}

inline std::vector<double> pack_resnet20_input(
    const std::vector<double> &logical) {
    constexpr std::size_t image_values = 3 * 32 * 32;
    constexpr std::size_t packed_block = 4096;
    constexpr std::size_t repetitions = 4;
    constexpr double activation_scale = 32.0;
    if (logical.size() != image_values)
        throw std::runtime_error(
            "ResNet-20 logical input must contain 3072 values");
    std::vector<double> block(packed_block, 0.0);
    for (std::size_t index = 0; index < logical.size(); ++index)
        block[index] = logical[index] / activation_scale;
    std::vector<double> packed;
    packed.reserve(packed_block * repetitions);
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
        packed.insert(packed.end(), block.begin(), block.end());
    return packed;
}

inline void set_input(Context &context, std::vector<double> input) {
    if (input.size() > context.slot_capacity)
        throw std::runtime_error("model input exceeds CKKS slot capacity");
    input.resize(context.slot_capacity, 0.0);
    const auto &input_desc =
        find_value(context.loaded_plan.plan, context.input_id);
    context.input = make_cipher(
        std::move(input), input_desc.context,
        context.operator_spec.spec.poly_degree, input_desc.level,
        input_desc.scale_log2, input_desc.ntt, input_desc.components);
}

inline Context load_context(const std::filesystem::path &plan_path,
                            const std::filesystem::path &operator_spec_path,
                            std::filesystem::path bundle_dir) {
    Context context;
    context.loaded_plan = RuntimePlanReader::read_file(plan_path.string());
    context.operator_spec = OperatorSpecReader::read_file(operator_spec_path.string());
    context.bundle_dir = std::move(bundle_dir);
    if (context.loaded_plan.plan.external_inputs.empty())
        throw std::runtime_error("experiment requires external inputs");
    context.input_id = context.loaded_plan.plan.external_inputs.front();
    context.slot_capacity =
        static_cast<std::size_t>(context.operator_spec.spec.poly_degree / 2);
    set_input(context, make_input(context.slot_capacity));
    return context;
}

inline std::string slots_sha256(const std::vector<double> &slots) {
    const auto *data = reinterpret_cast<const char *>(slots.data());
    return sha256_hex(std::string_view(data, slots.size() * sizeof(double)));
}

inline std::unordered_map<ValueId, std::size_t>
count_uses(const RuntimePlan &plan) {
    std::unordered_map<ValueId, std::size_t> uses;
    const auto count_phase = [&](const std::vector<Instruction> &phase) {
        for (const auto &instruction : phase) {
            if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
                for (ValueId id : op->inputs) ++uses[id];
            } else if (const auto *action = std::get_if<CommAction>(&instruction.body)) {
                for (ValueId id : action->inputs) ++uses[id];
            }
        }
    };
    count_phase(plan.initialization);
    count_phase(plan.execution);
    count_phase(plan.finalization);
    for (ValueId id : plan.final_outputs) ++uses[id];
    return uses;
}

inline bool metadata_matches(const VecPayload &actual, const ValueDesc &expected,
                             std::uint64_t poly_degree) {
    return actual.kind == expected.kind &&
           actual.metadata.context == expected.context &&
           actual.metadata.degree == poly_degree &&
           actual.metadata.level == expected.level &&
           actual.metadata.scale_log2 == expected.scale_log2 &&
           actual.metadata.ntt == expected.ntt &&
           actual.metadata.components == expected.components;
}

struct LineDiff {
    std::size_t mismatches = 0;
    double max_abs_diff = 0.0;
};

inline LineDiff compare_slots(const std::vector<double> &expected,
                              const std::vector<double> &actual) {
    LineDiff diff;
    if (expected.size() != actual.size()) {
        diff.mismatches = std::max(expected.size(), actual.size());
        diff.max_abs_diff = std::numeric_limits<double>::infinity();
        return diff;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] == actual[i]) continue;
        ++diff.mismatches;
        diff.max_abs_diff =
            std::max(diff.max_abs_diff, std::abs(expected[i] - actual[i]));
    }
    return diff;
}

inline std::vector<double> evaluate_compute(
    const ComputeOp &op,
    const std::unordered_map<ValueId, std::vector<double>> &values) {
    const auto unary = [&]() {
        if (op.inputs.size() != 1)
            throw std::runtime_error("reference unary operation has wrong arity");
        return values.at(op.inputs[0]);
    };
    const auto binary = [&](const auto &function) {
        if (op.inputs.size() != 2)
            throw std::runtime_error("reference binary operation has wrong arity");
        std::vector<double> result = values.at(op.inputs[0]);
        const auto &rhs = values.at(op.inputs[1]);
        if (result.size() != rhs.size())
            throw std::runtime_error("reference slot count mismatch");
        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = function(result[i], rhs[i]);
        return result;
    };

    switch (op.kind) {
    case ComputeKind::AddCC:
    case ComputeKind::AddCP:
        return binary([](double lhs, double rhs) { return lhs + rhs; });
    case ComputeKind::SubCC:
    case ComputeKind::SubCP:
        return binary([](double lhs, double rhs) { return lhs - rhs; });
    case ComputeKind::MulCC:
    case ComputeKind::MulCP:
        return binary([](double lhs, double rhs) { return lhs * rhs; });
    case ComputeKind::Negate: {
        auto result = unary();
        for (double &value : result) value = -value;
        return result;
    }
    case ComputeKind::Rotate: {
        auto result = unary();
        const long long size = static_cast<long long>(result.size());
        long long shift = static_cast<long long>(
            std::get<RotateAttrs>(op.attrs).steps) % size;
        if (shift < 0) shift += size;
        std::rotate(result.begin(), result.begin() + shift, result.end());
        return result;
    }
    case ComputeKind::Rescale:
    case ComputeKind::ModSwitch:
    case ComputeKind::Relinearize:
    case ComputeKind::Boot:
        return unary();
    }
    throw std::runtime_error("unsupported reference compute operation");
}

inline void record_diff(std::ostream &report, DiffSummary &summary,
                        std::string_view source, std::size_t ordinal,
                        std::string_view operation, ValueId output_id,
                        const ValueDesc &output_desc,
                        const std::vector<double> &expected,
                        const RunArtifact<VecValue> &artifact,
                        std::uint64_t poly_degree) {
    const VecPayload actual = artifact.values.at(output_id).value.materialize();
    const LineDiff line = compare_slots(expected, actual.slots);
    const bool metadata_match = metadata_matches(actual, output_desc, poly_degree);
    const bool mismatch = line.mismatches != 0 || !metadata_match;
    ++summary.compared_lines;
    summary.mismatch_slots += line.mismatches;
    summary.max_abs_diff = std::max(summary.max_abs_diff, line.max_abs_diff);
    if (mismatch) ++summary.mismatch_lines;
    report << "source=" << source
           << " ordinal=" << ordinal
           << " op=" << operation
           << " output=" << output_id
           << " slots=" << actual.slots.size()
           << " metadata=" << (metadata_match ? "MATCH" : "MISMATCH")
           << " mismatch_slots=" << line.mismatches
           << " max_abs_diff=" << std::setprecision(17) << line.max_abs_diff
           << '\n';
}

inline DiffSummary diff_against_reference(const Context &context,
                                          const RunArtifact<VecValue> &artifact,
                                          std::ostream &report) {
    const RuntimePlan &plan = context.loaded_plan.plan;
    std::vector<std::string> required_contents;
    for (const auto &instruction : plan.initialization) {
        const auto *encode = std::get_if<EncodeOp>(&instruction.body);
        if (!encode) continue;
        if (const auto *bundle = std::get_if<BundleEncodePayload>(&encode->payload))
            required_contents.push_back(bundle->content);
    }
    const auto bundle = PlaintextBundleLoader::load(
        context.bundle_dir, *plan.plaintext_bundle, required_contents,
        context.slot_capacity, false);

    std::unordered_map<ValueId, std::vector<double>> reference;
    reference.emplace(context.input_id, context.input.materialize().slots);
    auto uses = count_uses(plan);
    DiffSummary summary;
    record_diff(report, summary, "external_input", 0, "Input", context.input_id,
                find_value(plan, context.input_id), reference.at(context.input_id),
                artifact, context.operator_spec.spec.poly_degree);

    const auto release = [&](ValueId id) {
        auto &count = uses.at(id);
        if (--count == 0) reference.erase(id);
    };
    const auto process_phase = [&](std::string_view phase,
                                   const std::vector<Instruction> &instructions) {
        for (const auto &instruction : instructions) {
            if (const auto *encode = std::get_if<EncodeOp>(&instruction.body)) {
                std::vector<double> slots;
                if (const auto *inline_payload =
                        std::get_if<InlineEncodePayload>(&encode->payload))
                    slots = inline_payload->values;
                else
                    slots = bundle.slots_by_content.at(
                        std::get<BundleEncodePayload>(encode->payload).content);
                slots.resize(context.slot_capacity, 0.0);
                record_diff(report, summary, phase, instruction.ordinal, "Encode",
                            encode->output, find_value(plan, encode->output), slots,
                            artifact, context.operator_spec.spec.poly_degree);
                if (uses[encode->output] != 0)
                    reference.emplace(encode->output, std::move(slots));
                continue;
            }
            if (const auto *op = std::get_if<ComputeOp>(&instruction.body)) {
                auto output = evaluate_compute(*op, reference);
                record_diff(report, summary, phase, instruction.ordinal,
                            to_string(op->kind), op->output,
                            find_value(plan, op->output), output, artifact,
                            context.operator_spec.spec.poly_degree);
                if (uses[op->output] != 0)
                    reference.emplace(op->output, std::move(output));
                for (ValueId id : op->inputs) release(id);
                continue;
            }
            throw std::runtime_error(
                "MLP reference experiment does not accept communication instructions");
        }
    };
    process_phase("initialization", plan.initialization);
    process_phase("execution", plan.execution);
    process_phase("finalization", plan.finalization);

    if (plan.final_outputs.size() != 1)
        throw std::runtime_error("experiment requires exactly one final output");
    const ValueId final_id = plan.final_outputs.front();
    const auto actual = artifact.values.at(final_id).value.materialize();
    summary.final_sha256 = slots_sha256(actual.slots);
    const std::size_t prefix_size = std::min<std::size_t>(10, actual.slots.size());
    summary.final_prefix.assign(actual.slots.begin(), actual.slots.begin() + prefix_size);
    release(final_id);
    return summary;
}

inline void print_summary(std::ostream &out, const DiffSummary &summary) {
    out << "compared_lines=" << summary.compared_lines
        << " mismatch_lines=" << summary.mismatch_lines
        << " mismatch_slots=" << summary.mismatch_slots
        << " max_abs_diff=" << std::setprecision(17) << summary.max_abs_diff
        << " final_sha256=" << summary.final_sha256 << '\n';
    out << "final_prefix=";
    for (std::size_t i = 0; i < summary.final_prefix.size(); ++i) {
        if (i != 0) out << ',';
        out << std::setprecision(17) << summary.final_prefix[i];
    }
    out << '\n';
}

} // namespace fhegpu::experiment
