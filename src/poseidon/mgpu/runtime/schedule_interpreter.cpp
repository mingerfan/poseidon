#include "poseidon/mgpu/runtime/schedule_interpreter.h"

#include <exception>
#include <sstream>

namespace poseidon::mgpu
{
namespace
{

void add_error(ScheduleExecutionResult &result, std::size_t op_index, const std::string &message)
{
    result.errors.push_back(ScheduleExecutionError{ op_index, message });
}

void copy_verifier_errors(
    ScheduleExecutionResult &result, const ScheduleVerificationResult &verification)
{
    for (const ScheduleVerificationError &error : verification.errors)
    {
        add_error(result, error.op_index, error.message);
    }
}

void define_output(
    MgpuObjectStore &object_store, const MgpuOp &op, MgpuValueKind kind)
{
    object_store.define(op.outputs[0].id, kind, op.device_id);
}

void apply_completed_op(MgpuObjectStore &object_store, const MgpuOp &op)
{
    switch (op.kind)
    {
    case MgpuOpKind::UploadPlain:
    case MgpuOpKind::CopyPlain:
        define_output(object_store, op, MgpuValueKind::Plaintext);
        break;
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
        define_output(object_store, op, MgpuValueKind::Ciphertext);
        break;
    case MgpuOpKind::Download:
        break;
    }
}

}  // namespace

std::string ScheduleExecutionResult::format_errors() const
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

ScheduleInterpreter::ScheduleInterpreter(ScheduleInterpreterOptions options)
    : options_(options)
{
}

ScheduleExecutionResult ScheduleInterpreter::run(
    const MgpuSchedule &schedule, ScheduleOpHandler &handler) const
{
    ScheduleExecutionResult result;

    const ScheduleVerificationResult verification =
        verify_schedule(schedule, ScheduleVerifierOptions{ options_.device_count });
    if (!verification.ok())
    {
        copy_verifier_errors(result, verification);
        return result;
    }

    for (std::size_t op_index = 0; op_index < schedule.ops.size(); ++op_index)
    {
        const MgpuOp &op = schedule.ops[op_index];

        try
        {
            handler.execute(op, result.object_store);
            apply_completed_op(result.object_store, op);
        }
        catch (const std::exception &ex)
        {
            add_error(result, op_index, ex.what());
            return result;
        }
        catch (...)
        {
            add_error(result, op_index, "unknown execution error");
            return result;
        }
    }

    return result;
}

}  // namespace poseidon::mgpu
