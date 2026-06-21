#include "poseidon/mgpu/compiler/static_placement.h"

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
    StaticPlacementResult &result, std::size_t op_index, const std::string &message)
{
    result.diagnostics.push_back(StaticPlacementDiagnostic{ op_index, message });
}

bool is_valid_device(int device_id, int device_count)
{
    return device_id >= 0 && device_id < device_count;
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

const ValueState *lookup_value(
    StaticPlacementResult &result, const std::unordered_map<ValueId, ValueState> &values,
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

void define_output(
    StaticPlacementResult &result, std::unordered_map<ValueId, ValueState> &values,
    std::size_t op_index, const MgpuOp &op)
{
    MgpuValueKind kind;
    if (!output_kind(op.kind, kind) || op.outputs.empty())
    {
        return;
    }

    if (op.outputs.size() != 1)
    {
        std::ostringstream stream;
        stream << "expected 1 output, got " << op.outputs.size();
        add_diagnostic(result, op_index, stream.str());
        return;
    }

    const ValueId output_id = op.outputs[0].id;
    if (output_id == 0)
    {
        add_diagnostic(result, op_index, "output value id 0 is reserved");
        return;
    }

    const auto [iter, inserted] = values.emplace(output_id, ValueState{ kind, op.device_id });
    if (!inserted)
    {
        std::ostringstream stream;
        stream << "duplicate output value %" << output_id;
        add_diagnostic(result, op_index, stream.str());
    }
}

int next_round_robin_device(std::size_t &next_compute_device, int device_count)
{
    const int device_id = static_cast<int>(next_compute_device % static_cast<std::size_t>(device_count));
    ++next_compute_device;
    return device_id;
}

int choose_unassigned_device(
    StaticPlacementResult &result, const std::unordered_map<ValueId, ValueState> &values,
    std::size_t op_index, const MgpuOp &op, const StaticPlacementOptions &options,
    std::size_t &next_compute_device)
{
    if (is_upload_op(op.kind))
    {
        return options.default_device;
    }

    if (is_download_op(op.kind))
    {
        if (!op.inputs.empty())
        {
            const ValueState *state = lookup_value(result, values, op_index, op.inputs[0].id);
            if (state != nullptr)
            {
                return state->device_id;
            }
        }
        return options.default_device;
    }

    if (is_copy_op(op.kind))
    {
        add_diagnostic(
            result, op_index,
            "copy operation destination device must be assigned before placement");
        return options.default_device;
    }

    switch (options.policy)
    {
    case StaticPlacementPolicy::SingleDevice:
        return options.default_device;
    case StaticPlacementPolicy::RoundRobinCompute:
        return next_round_robin_device(next_compute_device, options.device_count);
    }
    return options.default_device;
}

}  // namespace

const char *to_string(StaticPlacementPolicy policy) noexcept
{
    switch (policy)
    {
    case StaticPlacementPolicy::SingleDevice:
        return "single_device";
    case StaticPlacementPolicy::RoundRobinCompute:
        return "round_robin_compute";
    }
    return "unknown";
}

std::string StaticPlacementResult::format_diagnostics() const
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

StaticPlacementResult place_static_schedule(
    const MgpuSchedule &schedule, const StaticPlacementOptions &options)
{
    StaticPlacementResult result;
    std::unordered_map<ValueId, ValueState> values;
    std::size_t next_compute_device = 0;

    if (options.device_count <= 0)
    {
        add_diagnostic(result, 0, "device_count must be positive");
        return result;
    }

    if (!is_valid_device(options.default_device, options.device_count))
    {
        std::ostringstream stream;
        stream << "invalid default device " << options.default_device << " for device_count "
               << options.device_count;
        add_diagnostic(result, 0, stream.str());
        return result;
    }

    for (std::size_t op_index = 0; op_index < schedule.ops.size(); ++op_index)
    {
        MgpuOp op = schedule.ops[op_index];
        const bool already_assigned =
            options.preserve_existing_devices && op.device_id != kUnassignedDevice;

        if (already_assigned)
        {
            if (!is_valid_device(op.device_id, options.device_count))
            {
                std::ostringstream stream;
                stream << "invalid assigned device " << op.device_id << " for device_count "
                       << options.device_count;
                add_diagnostic(result, op_index, stream.str());
            }
        }
        else
        {
            op.device_id = choose_unassigned_device(
                result, values, op_index, op, options, next_compute_device);
        }

        result.schedule.ops.push_back(op);
        define_output(result, values, op_index, op);
    }

    return result;
}

}  // namespace poseidon::mgpu
