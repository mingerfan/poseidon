#include "poseidon/mgpu/compiler/copy_insertion.h"

#include <algorithm>
#include <sstream>
#include <unordered_map>

namespace poseidon::mgpu
{
namespace
{

struct ValueState
{
    MgpuValueKind kind = MgpuValueKind::Ciphertext;
    int device_id = 0;
};

void add_diagnostic(
    CopyInsertionResult &result, std::size_t op_index, const std::string &message)
{
    result.diagnostics.push_back(CopyInsertionDiagnostic{ op_index, message });
}

ValueId max_value_id(const MgpuSchedule &schedule)
{
    ValueId max_id = 0;
    for (const MgpuOp &op : schedule.ops)
    {
        for (const MgpuValueRef &input : op.inputs)
        {
            max_id = std::max(max_id, input.id);
        }
        for (const MgpuValueRef &output : op.outputs)
        {
            max_id = std::max(max_id, output.id);
        }
    }
    return max_id;
}

bool define_value(
    CopyInsertionResult &result, std::unordered_map<ValueId, ValueState> &values,
    std::size_t op_index, ValueId id, MgpuValueKind kind, int device_id)
{
    if (id == 0)
    {
        add_diagnostic(result, op_index, "output value id 0 is reserved");
        return false;
    }

    const auto [iter, inserted] = values.emplace(id, ValueState{ kind, device_id });
    if (!inserted)
    {
        std::ostringstream stream;
        stream << "duplicate output value %" << id;
        add_diagnostic(result, op_index, stream.str());
        return false;
    }
    return true;
}

const ValueState *lookup_value(
    CopyInsertionResult &result, const std::unordered_map<ValueId, ValueState> &values,
    std::size_t op_index, ValueId id)
{
    const auto iter = values.find(id);
    if (iter == values.end())
    {
        std::ostringstream stream;
        stream << "unknown input value %" << id;
        add_diagnostic(result, op_index, stream.str());
        return nullptr;
    }
    return &iter->second;
}

bool expected_input_kind(MgpuOpKind kind, std::size_t input_index, MgpuValueKind &expected)
{
    switch (kind)
    {
    case MgpuOpKind::Add:
    case MgpuOpKind::Sub:
    case MgpuOpKind::Multiply:
        expected = MgpuValueKind::Ciphertext;
        return input_index < 2;
    case MgpuOpKind::AddPlain:
    case MgpuOpKind::MultiplyPlain:
        if (input_index == 0)
        {
            expected = MgpuValueKind::Ciphertext;
            return true;
        }
        if (input_index == 1)
        {
            expected = MgpuValueKind::Plaintext;
            return true;
        }
        return false;
    case MgpuOpKind::Relinearize:
    case MgpuOpKind::Rescale:
    case MgpuOpKind::Negate:
    case MgpuOpKind::Rotate:
    case MgpuOpKind::BootstrapFallback:
        expected = MgpuValueKind::Ciphertext;
        return input_index == 0;
    case MgpuOpKind::Download:
        return false;
    case MgpuOpKind::CopyPlain:
        expected = MgpuValueKind::Plaintext;
        return input_index == 0;
    case MgpuOpKind::CopyCipher:
        expected = MgpuValueKind::Ciphertext;
        return input_index == 0;
    case MgpuOpKind::UploadPlain:
    case MgpuOpKind::UploadCipher:
        return false;
    }
    return false;
}

bool output_kind(MgpuOpKind kind, MgpuValueKind &output)
{
    switch (kind)
    {
    case MgpuOpKind::UploadPlain:
    case MgpuOpKind::CopyPlain:
        output = MgpuValueKind::Plaintext;
        return true;
    case MgpuOpKind::UploadCipher:
    case MgpuOpKind::CopyCipher:
    case MgpuOpKind::Add:
    case MgpuOpKind::AddPlain:
    case MgpuOpKind::Sub:
    case MgpuOpKind::MultiplyPlain:
    case MgpuOpKind::Multiply:
    case MgpuOpKind::Negate:
    case MgpuOpKind::Relinearize:
    case MgpuOpKind::Rescale:
    case MgpuOpKind::Rotate:
    case MgpuOpKind::BootstrapFallback:
        output = MgpuValueKind::Ciphertext;
        return true;
    case MgpuOpKind::Download:
        return false;
    }
    return false;
}

MgpuOpKind copy_kind_for_value(MgpuValueKind kind)
{
    return kind == MgpuValueKind::Plaintext ? MgpuOpKind::CopyPlain : MgpuOpKind::CopyCipher;
}

void maybe_define_output(
    CopyInsertionResult &result, std::unordered_map<ValueId, ValueState> &values,
    std::size_t op_index, const MgpuOp &op)
{
    MgpuValueKind kind;
    if (output_kind(op.kind, kind))
    {
        if (op.outputs.size() == 1)
        {
            define_value(result, values, op_index, op.outputs[0].id, kind, op.device_id);
        }
    }
}

}  // namespace

std::string CopyInsertionResult::format_diagnostics() const
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

CopyInsertionResult insert_required_copies(
    const MgpuSchedule &schedule, const CopyInsertionOptions &options)
{
    CopyInsertionResult result;
    std::unordered_map<ValueId, ValueState> values;
    ValueId next_value_id = std::max(options.next_value_id, max_value_id(schedule) + 1);

    for (std::size_t op_index = 0; op_index < schedule.ops.size(); ++op_index)
    {
        MgpuOp op = schedule.ops[op_index];

        if (!is_upload_op(op.kind))
        {
            for (std::size_t input_index = 0; input_index < op.inputs.size(); ++input_index)
            {
                MgpuValueKind expected_kind = MgpuValueKind::Ciphertext;
                const bool has_expected_kind =
                    expected_input_kind(op.kind, input_index, expected_kind);
                const ValueState *state =
                    lookup_value(result, values, op_index, op.inputs[input_index].id);
                if (state == nullptr)
                {
                    continue;
                }

                if (has_expected_kind && state->kind != expected_kind)
                {
                    std::ostringstream stream;
                    stream << "input value %" << op.inputs[input_index].id << " expected "
                           << to_string(expected_kind) << ", got " << to_string(state->kind);
                    add_diagnostic(result, op_index, stream.str());
                    continue;
                }

                if (!is_copy_op(op.kind) && state->device_id != op.device_id)
                {
                    const ValueId copied_id = next_value_id++;
                    MgpuOp copy_op;
                    copy_op.kind = copy_kind_for_value(state->kind);
                    copy_op.device_id = op.device_id;
                    copy_op.inputs = { op.inputs[input_index] };
                    copy_op.outputs = { MgpuValueRef{ copied_id } };
                    copy_op.debug_name = "auto_copy";

                    result.schedule.ops.push_back(copy_op);
                    define_value(result, values, op_index, copied_id, state->kind, op.device_id);
                    op.inputs[input_index] = MgpuValueRef{ copied_id };
                }
            }
        }

        result.schedule.ops.push_back(op);
        maybe_define_output(result, values, op_index, op);
    }

    return result;
}

}  // namespace poseidon::mgpu
