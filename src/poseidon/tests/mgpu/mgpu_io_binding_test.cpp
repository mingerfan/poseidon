#include "poseidon/mgpu/runtime/io_binding_handler.h"

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

class StringComputeHandler final : public ScheduleOpHandler
{
public:
    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override
    {
        if (op.kind != MgpuOpKind::AddPlain)
        {
            return;
        }

        const auto cipher = object_store.object_as<std::string>(op.inputs[0].id);
        const auto plain = object_store.object_as<std::string>(op.inputs[1].id);
        object_store.define(
            op.outputs[0].id, MgpuValueKind::Ciphertext, op.device_id,
            std::make_shared<std::string>(*cipher + "+" + *plain));
    }
};

MgpuSchedule make_bound_io_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::AddPlain, 0, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(3) }, {}));
    return schedule;
}

void test_bound_uploads_feed_fallback_and_downloads()
{
    StringComputeHandler compute;
    IoBindingScheduleHandler io(&compute);
    io.bind_cipher_upload(1, std::make_shared<std::string>("cipher"));
    io.bind_plain_upload(2, std::make_shared<std::string>("plain"));

    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 1 });
    const ScheduleExecutionResult result = interpreter.run(make_bound_io_schedule(), io);

    require(result.ok(), "expected interpreter success, got:\n" + result.format_errors());
    require(result.object_store.has_object(1), "cipher upload should define an object");
    require(result.object_store.has_object(2), "plain upload should define an object");
    require(result.object_store.has_object(3), "fallback compute should define an object");
    require(io.has_download(3), "download should record the source value");

    const auto downloaded = std::static_pointer_cast<std::string>(io.downloaded_object(3));
    require(*downloaded == "cipher+plain", "downloaded object mismatch");
}

void test_missing_upload_binding_fails()
{
    StringComputeHandler compute;
    IoBindingScheduleHandler io(&compute);
    io.bind_cipher_upload(1, std::make_shared<std::string>("cipher"));

    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 1 });
    const ScheduleExecutionResult result = interpreter.run(make_bound_io_schedule(), io);

    require(!result.ok(), "missing upload binding should fail");
    require_contains(result.format_errors(), "missing upload binding for %2");
}

void test_upload_kind_mismatch_fails()
{
    StringComputeHandler compute;
    IoBindingScheduleHandler io(&compute);
    io.bind_cipher_upload(1, std::make_shared<std::string>("cipher"));
    io.bind_upload(2, MgpuValueKind::Ciphertext, std::make_shared<std::string>("plain"));

    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 1 });
    const ScheduleExecutionResult result = interpreter.run(make_bound_io_schedule(), io);

    require(!result.ok(), "upload kind mismatch should fail");
    require_contains(result.format_errors(), "expected plaintext");
}

void test_download_requires_object_handle()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 0, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(2) }, {}));

    IoBindingScheduleHandler io;
    io.bind_cipher_upload(1, std::make_shared<std::string>("cipher"));

    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 1 });
    const ScheduleExecutionResult result = interpreter.run(schedule, io);

    require(!result.ok(), "metadata-only output download should fail");
    require_contains(result.format_errors(), "download input %2 has no object handle");
}

}  // namespace

int main()
{
    try
    {
        test_bound_uploads_feed_fallback_and_downloads();
        test_missing_upload_binding_fails();
        test_upload_kind_mismatch_fails();
        test_download_requires_object_handle();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu IO binding test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu IO binding tests passed\n";
    return EXIT_SUCCESS;
}
