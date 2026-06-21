#include "poseidon/mgpu/compiler/schedule_verifier.h"

#include <sstream>
#include <string>
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

void add_error(
    ScheduleVerificationResult &result, std::size_t op_index, const std::string &message)
{
    result.errors.push_back(ScheduleVerificationError{ op_index, message });
}

bool validate_device(
    ScheduleVerificationResult &result, std::size_t op_index, int device_id, int device_count)
{
    if (device_count <= 0)
    {
        add_error(result, op_index, "device_count must be positive");
        return false;
    }

    if (device_id < 0 || device_id >= device_count)
    {
        std::ostringstream stream;
        stream << "invalid device " << device_id << " for device_count " << device_count;
        add_error(result, op_index, stream.str());
        return false;
    }

    return true;
}

bool expect_arity(
    ScheduleVerificationResult &result, const MgpuOp &op, std::size_t op_index,
    std::size_t input_count, std::size_t output_count)
{
    bool valid = true;
    if (op.inputs.size() != input_count)
    {
        std::ostringstream stream;
        stream << "expected " << input_count << " inputs, got " << op.inputs.size();
        add_error(result, op_index, stream.str());
        valid = false;
    }

    if (op.outputs.size() != output_count)
    {
        std::ostringstream stream;
        stream << "expected " << output_count << " outputs, got " << op.outputs.size();
        add_error(result, op_index, stream.str());
        valid = false;
    }

    return valid;
}

bool define_output(
    ScheduleVerificationResult &result, std::unordered_map<ValueId, ValueState> &values,
    std::size_t op_index, ValueId id, MgpuValueKind kind, int device_id)
{
    if (id == 0)
    {
        add_error(result, op_index, "output value id 0 is reserved");
        return false;
    }

    if (values.find(id) != values.end())
    {
        std::ostringstream stream;
        stream << "duplicate output value %" << id;
        add_error(result, op_index, stream.str());
        return false;
    }

    values.emplace(id, ValueState{ kind, device_id });
    return true;
}

const ValueState *lookup_input(
    ScheduleVerificationResult &result, const std::unordered_map<ValueId, ValueState> &values,
    std::size_t op_index, ValueId id)
{
    const auto iter = values.find(id);
    if (iter == values.end())
    {
        std::ostringstream stream;
        stream << "unknown input value %" << id;
        add_error(result, op_index, stream.str());
        return nullptr;
    }

    return &iter->second;
}

bool expect_input(
    ScheduleVerificationResult &result, const std::unordered_map<ValueId, ValueState> &values,
    std::size_t op_index, ValueId id, MgpuValueKind expected_kind, int expected_device)
{
    const ValueState *state = lookup_input(result, values, op_index, id);
    if (state == nullptr)
    {
        return false;
    }

    bool valid = true;
    if (state->kind != expected_kind)
    {
        std::ostringstream stream;
        stream << "input value %" << id << " expected " << to_string(expected_kind)
               << ", got " << to_string(state->kind);
        add_error(result, op_index, stream.str());
        valid = false;
    }

    if (state->device_id != expected_device)
    {
        std::ostringstream stream;
        stream << "input value %" << id << " is on device " << state->device_id
               << " but op runs on device " << expected_device;
        add_error(result, op_index, stream.str());
        valid = false;
    }

    return valid;
}

bool expect_copy_input(
    ScheduleVerificationResult &result, const std::unordered_map<ValueId, ValueState> &values,
    std::size_t op_index, ValueId id, MgpuValueKind expected_kind)
{
    const ValueState *state = lookup_input(result, values, op_index, id);
    if (state == nullptr)
    {
        return false;
    }

    if (state->kind != expected_kind)
    {
        std::ostringstream stream;
        stream << "copy input value %" << id << " expected " << to_string(expected_kind)
               << ", got " << to_string(state->kind);
        add_error(result, op_index, stream.str());
        return false;
    }

    return true;
}

}  // namespace

std::string ScheduleVerificationResult::format_errors() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < errors.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << "op #" << errors[i].op_index << ": " << errors[i].message;
    }
    return stream.str();
}

ScheduleVerificationResult verify_schedule(
    const MgpuSchedule &schedule, const ScheduleVerifierOptions &options)
{
    ScheduleVerificationResult result;
    std::unordered_map<ValueId, ValueState> values;

    for (std::size_t op_index = 0; op_index < schedule.ops.size(); ++op_index)
    {
        const MgpuOp &op = schedule.ops[op_index];
        const bool device_valid =
            validate_device(result, op_index, op.device_id, options.device_count);

        switch (op.kind)
        {
        case MgpuOpKind::UploadPlain:
            if (expect_arity(result, op, op_index, 0, 1) && device_valid)
            {
                define_output(
                    result, values, op_index, op.outputs[0].id, MgpuValueKind::Plaintext,
                    op.device_id);
            }
            break;
        case MgpuOpKind::UploadCipher:
            if (expect_arity(result, op, op_index, 0, 1) && device_valid)
            {
                define_output(
                    result, values, op_index, op.outputs[0].id, MgpuValueKind::Ciphertext,
                    op.device_id);
            }
            break;
        case MgpuOpKind::CopyPlain:
            if (expect_arity(result, op, op_index, 1, 1) && device_valid &&
                expect_copy_input(
                    result, values, op_index, op.inputs[0].id, MgpuValueKind::Plaintext))
            {
                define_output(
                    result, values, op_index, op.outputs[0].id, MgpuValueKind::Plaintext,
                    op.device_id);
            }
            break;
        case MgpuOpKind::CopyCipher:
            if (expect_arity(result, op, op_index, 1, 1) && device_valid &&
                expect_copy_input(
                    result, values, op_index, op.inputs[0].id, MgpuValueKind::Ciphertext))
            {
                define_output(
                    result, values, op_index, op.outputs[0].id, MgpuValueKind::Ciphertext,
                    op.device_id);
            }
            break;
        case MgpuOpKind::Add:
        case MgpuOpKind::Sub:
        case MgpuOpKind::Multiply:
            if (expect_arity(result, op, op_index, 2, 1) && device_valid)
            {
                const bool inputs_valid =
                    expect_input(
                        result, values, op_index, op.inputs[0].id, MgpuValueKind::Ciphertext,
                        op.device_id) &&
                    expect_input(
                        result, values, op_index, op.inputs[1].id, MgpuValueKind::Ciphertext,
                        op.device_id);
                if (inputs_valid)
                {
                    define_output(
                        result, values, op_index, op.outputs[0].id,
                        MgpuValueKind::Ciphertext, op.device_id);
                }
            }
            break;
        case MgpuOpKind::AddPlain:
        case MgpuOpKind::MultiplyPlain:
            if (expect_arity(result, op, op_index, 2, 1) && device_valid)
            {
                const bool inputs_valid =
                    expect_input(
                        result, values, op_index, op.inputs[0].id, MgpuValueKind::Ciphertext,
                        op.device_id) &&
                    expect_input(
                        result, values, op_index, op.inputs[1].id, MgpuValueKind::Plaintext,
                        op.device_id);
                if (inputs_valid)
                {
                    define_output(
                        result, values, op_index, op.outputs[0].id,
                        MgpuValueKind::Ciphertext, op.device_id);
                }
            }
            break;
        case MgpuOpKind::Relinearize:
        case MgpuOpKind::Rescale:
        case MgpuOpKind::Rotate:
        case MgpuOpKind::BootstrapFallback:
            if (expect_arity(result, op, op_index, 1, 1) && device_valid &&
                expect_input(
                    result, values, op_index, op.inputs[0].id, MgpuValueKind::Ciphertext,
                    op.device_id))
            {
                define_output(
                    result, values, op_index, op.outputs[0].id, MgpuValueKind::Ciphertext,
                    op.device_id);
            }
            break;
        case MgpuOpKind::Download:
            if (expect_arity(result, op, op_index, 1, 0) && device_valid)
            {
                const ValueState *state = lookup_input(result, values, op_index, op.inputs[0].id);
                if (state != nullptr && state->device_id != op.device_id)
                {
                    std::ostringstream stream;
                    stream << "download input value %" << op.inputs[0].id << " is on device "
                           << state->device_id << " but op runs on device " << op.device_id;
                    add_error(result, op_index, stream.str());
                }
            }
            break;
        }
    }

    return result;
}

}  // namespace poseidon::mgpu
