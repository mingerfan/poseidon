#include "poseidon/mgpu/runtime/schedule_interpreter.h"

#include <cstdlib>
#include <iostream>
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

class RecordingHandler final : public ScheduleOpHandler
{
public:
    explicit RecordingHandler(std::size_t fail_at = static_cast<std::size_t>(-1))
        : fail_at_(fail_at)
    {
    }

    void execute(const MgpuOp &op, const MgpuObjectStore &object_store) override
    {
        if (records.size() == fail_at_)
        {
            throw std::runtime_error("injected handler failure");
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

void test_interpreter_runs_static_order()
{
    RecordingHandler handler;
    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 1 });
    const ScheduleExecutionResult result = interpreter.run(make_single_device_schedule(), handler);

    require(result.ok(), "expected interpreter success, got:\n" + result.format_errors());
    require(handler.records.size() == 5, "handler record count mismatch");
    require(handler.records[0].kind == MgpuOpKind::UploadCipher, "first op mismatch");
    require(handler.records[2].kind == MgpuOpKind::MultiplyPlain, "third op mismatch");
    require(handler.records[4].kind == MgpuOpKind::Download, "last op mismatch");
    require(handler.records[2].device_id == 0, "interpreter should use scheduled device");
    require(handler.observed_first_input_devices[0] == 0, "input device should come from store");

    require(result.object_store.size() == 4, "download should not define a new GPU object");
    require(result.object_store.at(4).kind == MgpuValueKind::Ciphertext, "rescale output kind mismatch");
    require(result.object_store.at(4).device_id == 0, "rescale output device mismatch");
}

void test_interpreter_rejects_invalid_schedule_before_execution()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 1, { value(1) }, { value(2) }));

    RecordingHandler handler;
    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 2 });
    const ScheduleExecutionResult result = interpreter.run(schedule, handler);

    require(!result.ok(), "invalid schedule should fail");
    require(handler.records.empty(), "invalid schedule should not execute any op");
    require_contains(result.format_errors(), "input value %1 is on device 0 but op runs on device 1");
}

void test_handler_failure_stops_execution()
{
    RecordingHandler handler(2);
    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 1 });
    const ScheduleExecutionResult result = interpreter.run(make_single_device_schedule(), handler);

    require(!result.ok(), "handler failure should fail execution");
    require(handler.records.size() == 2, "handler should stop at injected failure");
    require(result.object_store.contains(1), "completed upload should remain in result store");
    require(!result.object_store.contains(3), "failed op output should not be defined");
    require_contains(result.format_errors(), "injected handler failure");
}

}  // namespace

int main()
{
    try
    {
        test_object_store();
        test_interpreter_runs_static_order();
        test_interpreter_rejects_invalid_schedule_before_execution();
        test_handler_failure_stops_execution();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu runtime test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu runtime tests passed\n";
    return EXIT_SUCCESS;
}
