#include "poseidon/mgpu/runtime/hevm_io_binding.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

constexpr const char *kHevmArgIndex = "hevm_arg_index";
constexpr const char *kHevmArgScale = "hevm_arg_scale";
constexpr const char *kHevmArgLevel = "hevm_arg_level";
constexpr const char *kHevmInitLevel = "hevm_init_level";
constexpr const char *kHevmPlainRegister = "hevm_plain_register";
constexpr const char *kHevmConstantIndex = "hevm_constant_index";
constexpr const char *kEncodeScale = "encode_scale";
constexpr const char *kEncodeLevel = "encode_level";
constexpr const char *kHevmResultIndex = "hevm_result_index";
constexpr const char *kHevmResultRegister = "hevm_result_register";
constexpr const char *kHevmResultScale = "hevm_result_scale";
constexpr const char *kHevmResultLevel = "hevm_result_level";

void add_diagnostic(
    HevmIoBindingPlanResult &result, std::size_t op_index, std::string message)
{
    result.diagnostics.push_back(HevmIoBindingDiagnostic{ op_index, std::move(message) });
}

bool has_attribute(const MgpuOp &op, const char *name)
{
    return op.integer_attributes.find(name) != op.integer_attributes.end();
}

bool has_any_attribute(const MgpuOp &op, const std::vector<const char *> &names)
{
    return std::any_of(names.begin(), names.end(), [&op](const char *name) {
        return has_attribute(op, name);
    });
}

bool read_u64_attribute(
    HevmIoBindingPlanResult &result, const MgpuOp &op, std::size_t op_index,
    const char *name, std::uint64_t &value)
{
    const auto iter = op.integer_attributes.find(name);
    if (iter == op.integer_attributes.end())
    {
        add_diagnostic(
            result, op_index, std::string("missing HEVM integer attribute '") + name + "'");
        return false;
    }
    if (iter->second < 0)
    {
        add_diagnostic(
            result, op_index,
            std::string("HEVM integer attribute '") + name + "' must be non-negative");
        return false;
    }

    value = static_cast<std::uint64_t>(iter->second);
    return true;
}

bool read_single_output(
    HevmIoBindingPlanResult &result, const MgpuOp &op, std::size_t op_index,
    ValueId &value_id)
{
    if (op.outputs.size() != 1)
    {
        std::ostringstream stream;
        stream << "HEVM cipher input upload expected one output, got " << op.outputs.size();
        add_diagnostic(result, op_index, stream.str());
        return false;
    }

    value_id = op.outputs[0].id;
    return true;
}

bool read_single_input(
    HevmIoBindingPlanResult &result, const MgpuOp &op, std::size_t op_index,
    ValueId &value_id)
{
    if (op.inputs.size() != 1)
    {
        std::ostringstream stream;
        stream << "HEVM result download expected one input, got " << op.inputs.size();
        add_diagnostic(result, op_index, stream.str());
        return false;
    }

    value_id = op.inputs[0].id;
    return true;
}

void record_cipher_input(
    HevmIoBindingPlanResult &result, const MgpuOp &op, std::size_t op_index,
    std::unordered_map<std::uint64_t, std::size_t> &seen_inputs)
{
    HevmCipherInputSlot slot;
    const bool valid =
        read_single_output(result, op, op_index, slot.value_id) &&
        read_u64_attribute(result, op, op_index, kHevmArgIndex, slot.index) &&
        read_u64_attribute(result, op, op_index, kHevmArgScale, slot.scale) &&
        read_u64_attribute(result, op, op_index, kHevmArgLevel, slot.level) &&
        read_u64_attribute(result, op, op_index, kHevmInitLevel, slot.init_level);
    if (!valid)
    {
        return;
    }

    const auto [iter, inserted] = seen_inputs.emplace(slot.index, op_index);
    if (!inserted)
    {
        std::ostringstream stream;
        stream << "duplicate HEVM cipher input index " << slot.index
               << " first seen at op #" << iter->second;
        add_diagnostic(result, op_index, stream.str());
        return;
    }

    slot.device_id = op.device_id;
    result.plan.cipher_inputs.push_back(slot);
}

void record_result(
    HevmIoBindingPlanResult &result, const MgpuOp &op, std::size_t op_index,
    std::unordered_map<std::uint64_t, std::size_t> &seen_results)
{
    HevmResultSlot slot;
    const bool valid =
        read_single_input(result, op, op_index, slot.value_id) &&
        read_u64_attribute(result, op, op_index, kHevmResultIndex, slot.index) &&
        read_u64_attribute(result, op, op_index, kHevmResultRegister, slot.register_id) &&
        read_u64_attribute(result, op, op_index, kHevmResultScale, slot.scale) &&
        read_u64_attribute(result, op, op_index, kHevmResultLevel, slot.level);
    if (!valid)
    {
        return;
    }

    const auto [iter, inserted] = seen_results.emplace(slot.index, op_index);
    if (!inserted)
    {
        std::ostringstream stream;
        stream << "duplicate HEVM result index " << slot.index << " first seen at op #"
               << iter->second;
        add_diagnostic(result, op_index, stream.str());
        return;
    }

    slot.device_id = op.device_id;
    result.plan.results.push_back(slot);
}

void record_plain_input(
    HevmIoBindingPlanResult &result, const MgpuOp &op, std::size_t op_index)
{
    HevmPlainInputSlot slot;
    const bool valid =
        read_single_output(result, op, op_index, slot.value_id) &&
        read_u64_attribute(result, op, op_index, kHevmPlainRegister, slot.register_id) &&
        read_u64_attribute(result, op, op_index, kHevmConstantIndex, slot.constant_index) &&
        read_u64_attribute(result, op, op_index, kEncodeScale, slot.scale) &&
        read_u64_attribute(result, op, op_index, kEncodeLevel, slot.level);
    if (!valid)
    {
        return;
    }

    slot.device_id = op.device_id;
    result.plan.plain_inputs.push_back(slot);
}

void validate_index_range(std::uint64_t index, std::size_t count, const char *what)
{
    if (index >= count)
    {
        std::ostringstream stream;
        stream << "HEVM " << what << " index " << index
               << " is outside object count " << count;
        throw std::invalid_argument(stream.str());
    }
}

}  // namespace

std::string HevmIoBindingPlanResult::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << "op #" << diagnostics[i].op_index << ": " << diagnostics[i].message;
    }
    return stream.str();
}

HevmIoBindingPlanResult build_hevm_io_binding_plan(const MgpuSchedule &schedule)
{
    HevmIoBindingPlanResult result;
    std::unordered_map<std::uint64_t, std::size_t> seen_inputs;
    std::unordered_map<std::uint64_t, std::size_t> seen_results;
    const std::vector<const char *> arg_attrs{
        kHevmArgIndex,
        kHevmArgScale,
        kHevmArgLevel,
        kHevmInitLevel,
    };
    const std::vector<const char *> plain_attrs{
        kHevmPlainRegister,
        kHevmConstantIndex,
    };
    const std::vector<const char *> result_attrs{
        kHevmResultIndex,
        kHevmResultRegister,
        kHevmResultScale,
        kHevmResultLevel,
    };

    for (std::size_t op_index = 0; op_index < schedule.ops.size(); ++op_index)
    {
        const MgpuOp &op = schedule.ops[op_index];
        const bool has_arg_attrs = has_any_attribute(op, arg_attrs);
        const bool has_plain_attrs = has_any_attribute(op, plain_attrs);
        const bool has_result_attrs = has_any_attribute(op, result_attrs);

        if (has_arg_attrs && op.kind != MgpuOpKind::UploadCipher)
        {
            add_diagnostic(
                result, op_index,
                "HEVM cipher input metadata is only valid on upload_cipher");
            continue;
        }
        if (has_plain_attrs && op.kind != MgpuOpKind::UploadPlain)
        {
            add_diagnostic(
                result, op_index,
                "HEVM plaintext input metadata is only valid on upload_plain");
            continue;
        }
        if (has_result_attrs && op.kind != MgpuOpKind::Download)
        {
            add_diagnostic(
                result, op_index, "HEVM result metadata is only valid on download");
            continue;
        }

        if (has_arg_attrs)
        {
            record_cipher_input(result, op, op_index, seen_inputs);
        }
        if (has_plain_attrs)
        {
            record_plain_input(result, op, op_index);
        }
        if (has_result_attrs)
        {
            record_result(result, op, op_index, seen_results);
        }
    }

    std::sort(
        result.plan.cipher_inputs.begin(), result.plan.cipher_inputs.end(),
        [](const HevmCipherInputSlot &left, const HevmCipherInputSlot &right) {
            return left.index < right.index;
        });
    std::sort(
        result.plan.plain_inputs.begin(), result.plan.plain_inputs.end(),
        [](const HevmPlainInputSlot &left, const HevmPlainInputSlot &right) {
            if (left.register_id != right.register_id)
            {
                return left.register_id < right.register_id;
            }
            return left.value_id < right.value_id;
        });
    std::sort(
        result.plan.results.begin(), result.plan.results.end(),
        [](const HevmResultSlot &left, const HevmResultSlot &right) {
            return left.index < right.index;
        });

    return result;
}

void bind_hevm_cipher_inputs(
    IoBindingExecutionBackend &io, const HevmIoBindingPlan &plan,
    const std::vector<std::shared_ptr<void>> &cipher_inputs)
{
    if (cipher_inputs.size() != plan.cipher_inputs.size())
    {
        std::ostringstream stream;
        stream << "HEVM cipher input object count " << cipher_inputs.size()
               << " does not match schedule input count " << plan.cipher_inputs.size();
        throw std::invalid_argument(stream.str());
    }

    for (const HevmCipherInputSlot &slot : plan.cipher_inputs)
    {
        validate_index_range(slot.index, cipher_inputs.size(), "cipher input");
        io.bind_cipher_upload(slot.value_id, cipher_inputs[static_cast<std::size_t>(slot.index)]);
    }
}

void bind_hevm_plain_inputs_by_constant_index(
    IoBindingExecutionBackend &io, const HevmIoBindingPlan &plan,
    const std::unordered_map<std::uint64_t, std::shared_ptr<void>> &plain_inputs)
{
    for (const HevmPlainInputSlot &slot : plan.plain_inputs)
    {
        const auto iter = plain_inputs.find(slot.constant_index);
        if (iter == plain_inputs.end())
        {
            std::ostringstream stream;
            stream << "missing HEVM plaintext object for constant index "
                   << slot.constant_index;
            throw std::invalid_argument(stream.str());
        }
        io.bind_plain_upload(slot.value_id, iter->second);
    }
}

std::vector<std::shared_ptr<void>> collect_hevm_results(
    const IoBindingExecutionBackend &io, const HevmIoBindingPlan &plan)
{
    std::vector<std::shared_ptr<void>> results(plan.results.size());
    for (const HevmResultSlot &slot : plan.results)
    {
        validate_index_range(slot.index, results.size(), "result");
        results[static_cast<std::size_t>(slot.index)] = io.downloaded_object(slot.value_id);
    }
    return results;
}

}  // namespace poseidon::mgpu
