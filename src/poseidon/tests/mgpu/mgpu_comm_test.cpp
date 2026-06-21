#include "poseidon/mgpu/runtime/comm_schedule_handler.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace poseidon::mgpu;

namespace
{

MgpuValueRef value(ValueId id)
{
    return MgpuValueRef{ id };
}

MgpuOp op(
    MgpuOpKind kind, int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs)
{
    return MgpuOp{ kind, device_id, std::move(inputs), std::move(outputs), {} };
}

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void require_contains(const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos)
    {
        throw std::runtime_error("expected text to contain: " + needle + "\ntext:\n" + text);
    }
}

class RecordingGpuComm final : public GpuComm
{
public:
    std::shared_ptr<void> copy(const GpuCommCopyRequest &request) override
    {
        requests.push_back(request);
        return request.source_object;
    }

    std::vector<GpuCommCopyRequest> requests;
};

class RecordingFallback final : public ScheduleOpHandler
{
public:
    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override
    {
        if (is_upload_op(op.kind))
        {
            const MgpuValueKind kind = op.kind == MgpuOpKind::UploadPlain
                                           ? MgpuValueKind::Plaintext
                                           : MgpuValueKind::Ciphertext;
            object_store.define(
                op.outputs[0].id, kind, op.device_id,
                std::make_shared<std::string>("uploaded"));
        }
        op_kinds.push_back(op.kind);
    }

    std::vector<MgpuOpKind> op_kinds;
};

MgpuSchedule make_same_device_copy_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 0, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(2) }, {}));
    return schedule;
}

MgpuSchedule make_cross_device_copy_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 1, { value(2) }, {}));
    return schedule;
}

void test_copy_dispatch_uses_comm_layer()
{
    RecordingGpuComm comm;
    RecordingFallback fallback;
    CopyDispatchingScheduleHandler handler(comm, &fallback);
    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 1 });

    const ScheduleExecutionResult result =
        interpreter.run(make_same_device_copy_schedule(), handler);

    require(result.ok(), "same-device copy schedule should pass:\n" + result.format_errors());
    require(comm.requests.size() == 1, "copy op should produce one comm request");
    require(comm.requests[0].source_id == 1, "comm source id mismatch");
    require(comm.requests[0].destination_id == 2, "comm destination id mismatch");
    require(comm.requests[0].kind == MgpuValueKind::Ciphertext, "comm kind mismatch");
    require(comm.requests[0].source_device == 0, "comm source device mismatch");
    require(comm.requests[0].destination_device == 0, "comm destination device mismatch");
    require(comm.requests[0].source_object != nullptr, "comm source object should be provided");
    require(result.object_store.has_object(2), "copy output object should be retained");
    require(
        *result.object_store.object_as<std::string>(2) == "uploaded",
        "copy output object mismatch");
    require(fallback.op_kinds.size() == 2, "fallback should receive non-copy ops only");
    require(fallback.op_kinds[0] == MgpuOpKind::UploadCipher, "fallback first op mismatch");
    require(fallback.op_kinds[1] == MgpuOpKind::Download, "fallback second op mismatch");
}

void test_same_device_comm_rejects_cross_device_copy()
{
    SameDeviceGpuComm comm;
    CopyDispatchingScheduleHandler handler(comm);
    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 2 });

    const ScheduleExecutionResult result =
        interpreter.run(make_cross_device_copy_schedule(), handler);

    require(!result.ok(), "same-device comm should reject cross-device copy");
    require_contains(result.format_errors(), "requires a multi-GPU communication backend");
    require(result.object_store.contains(1), "upload before failed copy should be recorded");
    require(!result.object_store.contains(2), "failed copy output should not be recorded");
}

}  // namespace

int main()
{
    try
    {
        test_copy_dispatch_uses_comm_layer();
        test_same_device_comm_rejects_cross_device_copy();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu comm test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu comm tests passed\n";
    return EXIT_SUCCESS;
}
