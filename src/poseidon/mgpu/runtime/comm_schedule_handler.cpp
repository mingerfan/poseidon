#include "poseidon/mgpu/runtime/comm_schedule_handler.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

bool is_comm_copy_op(MgpuOpKind kind)
{
    return kind == MgpuOpKind::CopyPlain || kind == MgpuOpKind::CopyCipher;
}

MgpuValueKind value_kind_for_copy(MgpuOpKind kind)
{
    return kind == MgpuOpKind::CopyPlain ? MgpuValueKind::Plaintext
                                         : MgpuValueKind::Ciphertext;
}

void validate_copy_source_has_object(
    const MgpuOp &op, const MgpuObjectMetadata &source)
{
    if (source.object != nullptr)
    {
        return;
    }

    std::ostringstream stream;
    stream << "copy source value %" << op.inputs[0].id
           << " has no object handle";
    throw std::runtime_error(stream.str());
}

}  // namespace

CopyDispatchingScheduleHandler::CopyDispatchingScheduleHandler(
    GpuComm &comm, ScheduleOpHandler *fallback)
    : comm_(comm), fallback_(fallback)
{
}

void CopyDispatchingScheduleHandler::execute(
    const MgpuOp &op, MgpuObjectStore &object_store)
{
    if (!is_comm_copy_op(op.kind))
    {
        if (fallback_ != nullptr)
        {
            fallback_->execute(op, object_store);
        }
        return;
    }

    const MgpuObjectMetadata &source = object_store.at(op.inputs[0].id);
    validate_copy_source_has_object(op, source);
    std::shared_ptr<void> copied_object = comm_.copy(GpuCommCopyRequest{
        op.inputs[0].id,
        op.outputs[0].id,
        value_kind_for_copy(op.kind),
        source.device_id,
        op.device_id,
        source.object,
    });
    if (copied_object != nullptr)
    {
        object_store.define(
            op.outputs[0].id, value_kind_for_copy(op.kind), op.device_id,
            std::move(copied_object));
    }
}

}  // namespace poseidon::mgpu
