#include "poseidon/mgpu/runtime/copy_dispatching_backend.h"
#include "poseidon/mgpu/runtime/sequential_schedule_executor.h"

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

class RecordingBackend final : public ScheduleExecutionBackend
{
public:
    explicit RecordingBackend(std::size_t fail_at = static_cast<std::size_t>(-1))
        : fail_at_(fail_at)
    {
    }

    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override
    {
        if (records.size() == fail_at_)
        {
            throw std::runtime_error("injected backend failure");
        }

        if (!op.inputs.empty())
        {
            const MgpuObjectMetadata &metadata = object_store.at(op.inputs[0].id);
            observed_first_input_devices.push_back(metadata.device_id);
        }

        records.push_back(Record{ op.kind, op.device_id });
    }

    struct Record
    {
        MgpuOpKind kind = MgpuOpKind::Download;
        int device_id = 0;
    };

    std::vector<Record> records;
    std::vector<int> observed_first_input_devices;

private:
    std::size_t fail_at_ = static_cast<std::size_t>(-1);
};

class ObjectDefiningBackend final : public ScheduleExecutionBackend
{
public:
    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override
    {
        switch (op.kind)
        {
        case MgpuOpKind::UploadCipher:
            object_store.define(
                op.outputs[0].id, MgpuValueKind::Ciphertext, op.device_id,
                std::make_shared<std::string>("cipher-upload"));
            break;
        case MgpuOpKind::UploadPlain:
            object_store.define(
                op.outputs[0].id, MgpuValueKind::Plaintext, op.device_id,
                std::make_shared<std::string>("plain-upload"));
            break;
        case MgpuOpKind::MultiplyPlain: {
            const auto input = object_store.object_as<std::string>(op.inputs[0].id);
            const auto plain = object_store.object_as<std::string>(op.inputs[1].id);
            object_store.define(
                op.outputs[0].id, MgpuValueKind::Ciphertext, op.device_id,
                std::make_shared<std::string>(*input + "+" + *plain));
            break;
        }
        default:
            break;
        }
    }
};

class MetadataOnlyUploadBackend final : public ScheduleExecutionBackend
{
public:
    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override
    {
        if (op.kind == MgpuOpKind::UploadCipher)
        {
            object_store.define(
                op.outputs[0].id, MgpuValueKind::Ciphertext, op.device_id);
        }
    }
};

class WrongOutputKindBackend final : public ScheduleExecutionBackend
{
public:
    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override
    {
        if (op.kind == MgpuOpKind::UploadCipher)
        {
            object_store.define(
                op.outputs[0].id, MgpuValueKind::Plaintext, op.device_id);
        }
    }
};

class WrongOutputDeviceBackend final : public ScheduleExecutionBackend
{
public:
    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override
    {
        if (op.kind == MgpuOpKind::UploadCipher)
        {
            object_store.define(
                op.outputs[0].id, MgpuValueKind::Ciphertext, op.device_id + 1);
        }
    }
};

class ReturningGpuComm final : public GpuComm
{
public:
    std::shared_ptr<void> copy(const GpuCommCopyRequest &request) override
    {
        requests.push_back(request);
        return request.source_object;
    }

    std::vector<GpuCommCopyRequest> requests;
};

MgpuSchedule make_single_device_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::MultiplyPlain, 0, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 0, { value(3) }, { value(4) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(4) }, {}));
    return schedule;
}

void test_object_store()
{
    MgpuObjectStore store;
    require(store.empty(), "new object store should be empty");
    store.define(7, MgpuValueKind::Ciphertext, 3);
    require(store.size() == 1, "object store size mismatch");
    require(store.contains(7), "object store should contain value 7");
    require(store.at(7).kind == MgpuValueKind::Ciphertext, "object kind mismatch");
    require(store.at(7).device_id == 3, "object device mismatch");
    require(!store.has_object(7), "metadata-only object should not have a handle");

    auto object = std::make_shared<std::string>("payload");
    store.define(8, MgpuValueKind::Plaintext, 1, object);
    require(store.has_object(8), "object-backed value should have a handle");
    require(*store.object_as<std::string>(8) == "payload", "stored object mismatch");

    store.set_object(7, std::make_shared<std::string>("late"));
    require(store.has_object(7), "set_object should attach a handle");
    require(*store.object_as<std::string>(7) == "late", "set object mismatch");

    bool duplicate_failed = false;
    try
    {
        store.define(7, MgpuValueKind::Plaintext, 3);
    }
    catch (const std::invalid_argument &)
    {
        duplicate_failed = true;
    }
    require(duplicate_failed, "duplicate object define should fail");
}

ScheduleExecutionResult run_with_copy_dispatch(
    const MgpuSchedule &schedule, GpuComm &comm, ScheduleExecutionBackend &backend,
    SequentialScheduleExecutorOptions options)
{
    CopyDispatchingExecutionBackend copy_backend(comm, &backend);
    SequentialScheduleExecutor executor(options);
    return executor.run(schedule, copy_backend);
}

void test_sequential_executor_preserves_backend_defined_objects()
{
    ObjectDefiningBackend backend;
    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 1 });
    const ScheduleExecutionResult result = executor.run(make_single_device_schedule(), backend);

    require(result.ok(), "expected executor success, got:\n" + result.format_errors());
    require(result.object_store.has_object(1), "upload ciphertext object should be retained");
    require(result.object_store.has_object(2), "upload plaintext object should be retained");
    require(result.object_store.has_object(3), "compute output object should be retained");
    require(!result.object_store.has_object(4), "metadata-only rescale output should not have object");
    require(
        *result.object_store.object_as<std::string>(3) == "cipher-upload+plain-upload",
        "compute output object mismatch");
}

void test_sequential_executor_runs_static_order()
{
    RecordingBackend backend;
    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 1 });
    const ScheduleExecutionResult result = executor.run(make_single_device_schedule(), backend);

    require(result.ok(), "expected executor success, got:\n" + result.format_errors());
    require(backend.records.size() == 5, "backend record count mismatch");
    require(backend.records[0].kind == MgpuOpKind::UploadCipher, "first op mismatch");
    require(backend.records[2].kind == MgpuOpKind::MultiplyPlain, "third op mismatch");
    require(backend.records[4].kind == MgpuOpKind::Download, "last op mismatch");
    require(backend.records[2].device_id == 0, "executor should use scheduled device");
    require(backend.observed_first_input_devices[0] == 0, "input device should come from store");

    require(result.object_store.size() == 4, "download should not define a new GPU object");
    require(result.object_store.at(4).kind == MgpuValueKind::Ciphertext, "rescale output kind mismatch");
    require(result.object_store.at(4).device_id == 0, "rescale output device mismatch");
}

void test_sequential_executor_rejects_invalid_schedule_before_execution()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 1, { value(1) }, { value(2) }));

    RecordingBackend backend;
    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 2 });
    const ScheduleExecutionResult result = executor.run(schedule, backend);

    require(!result.ok(), "invalid schedule should fail");
    require(backend.records.empty(), "invalid schedule should not execute any op");
    require_contains(result.format_errors(), "input value %1 is on device 0 but op runs on device 1");
}

void test_backend_failure_stops_execution()
{
    RecordingBackend backend(2);
    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 1 });
    const ScheduleExecutionResult result = executor.run(make_single_device_schedule(), backend);

    require(!result.ok(), "backend failure should fail execution");
    require(backend.records.size() == 2, "backend should stop at injected failure");
    require(result.object_store.contains(1), "completed upload should remain in result store");
    require(!result.object_store.contains(3), "failed op output should not be defined");
    require_contains(result.format_errors(), "injected backend failure");
}

void test_sequential_executor_rejects_wrong_backend_output_kind()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));

    WrongOutputKindBackend backend;
    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 1 });
    const ScheduleExecutionResult result = executor.run(schedule, backend);

    require(!result.ok(), "wrong backend output kind should fail execution");
    require_contains(
        result.format_errors(),
        "backend defined output %1 as plaintext, expected ciphertext");
}

void test_sequential_executor_rejects_wrong_backend_output_device()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));

    WrongOutputDeviceBackend backend;
    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 2 });
    const ScheduleExecutionResult result = executor.run(schedule, backend);

    require(!result.ok(), "wrong backend output device should fail execution");
    require_contains(
        result.format_errors(),
        "backend defined output %1 on device 1, expected device 0");
}

void test_copy_dispatch_backend_routes_copies_to_comm()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 1, { value(2) }, {}));

    ReturningGpuComm comm;
    ObjectDefiningBackend backend;
    const ScheduleExecutionResult result = run_with_copy_dispatch(
        schedule, comm, backend, SequentialScheduleExecutorOptions{ 2 });

    require(result.ok(), "copy-dispatch execution should run schedule:\n" + result.format_errors());
    require(comm.requests.size() == 1, "copy should be dispatched through comm");
    require(comm.requests[0].source_id == 1, "comm source id mismatch");
    require(comm.requests[0].destination_id == 2, "comm destination id mismatch");
    require(comm.requests[0].source_device == 0, "comm source device mismatch");
    require(comm.requests[0].destination_device == 1, "comm destination device mismatch");
    require(result.object_store.at(2).device_id == 1, "copied object should land on destination");
    require(result.object_store.has_object(2), "copied object handle should be retained");
}

void test_copy_dispatch_backend_reports_comm_errors()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 1, { value(2) }, {}));

    SameDeviceGpuComm comm;
    ObjectDefiningBackend backend;
    const ScheduleExecutionResult result = run_with_copy_dispatch(
        schedule, comm, backend, SequentialScheduleExecutorOptions{ 2 });

    require(!result.ok(), "same-device comm should reject executor cross-device copy");
    require_contains(result.format_errors(), "requires a multi-GPU communication backend");
    require(result.object_store.contains(1), "upload before failed copy should be retained");
    require(
        !result.object_store.contains(2),
        "failed copy output should not be retained");
}

void test_copy_dispatch_backend_rejects_metadata_only_copy_source()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(
        op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));

    ReturningGpuComm comm;
    MetadataOnlyUploadBackend backend;
    const ScheduleExecutionResult result = run_with_copy_dispatch(
        schedule, comm, backend, SequentialScheduleExecutorOptions{ 2 });

    require(!result.ok(), "metadata-only copy source should fail");
    require_contains(
        result.format_errors(), "copy source value %1 has no object handle");
    require(comm.requests.empty(), "comm should not receive metadata-only copy");
    require(result.object_store.contains(1), "metadata-only upload should remain");
    require(
        !result.object_store.contains(2),
        "failed copy output should not be retained");
}

}  // namespace

int main()
{
    try
    {
        test_object_store();
        test_sequential_executor_runs_static_order();
        test_sequential_executor_preserves_backend_defined_objects();
        test_sequential_executor_rejects_invalid_schedule_before_execution();
        test_backend_failure_stops_execution();
        test_sequential_executor_rejects_wrong_backend_output_kind();
        test_sequential_executor_rejects_wrong_backend_output_device();
        test_copy_dispatch_backend_routes_copies_to_comm();
        test_copy_dispatch_backend_reports_comm_errors();
        test_copy_dispatch_backend_rejects_metadata_only_copy_source();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu runtime test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu runtime tests passed\n";
    return EXIT_SUCCESS;
}
