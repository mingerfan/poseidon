#include "poseidon/mgpu/compiler/static_schedule_pipeline.h"
#include "poseidon/mgpu/runtime/comm_schedule_handler.h"
#include "poseidon/mgpu/runtime/io_binding_handler.h"
#include "poseidon/tests/mgpu/hevm_test_utils.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace poseidon::mgpu;

namespace
{

struct MockObject
{
    std::string expression;
};

std::shared_ptr<MockObject> object(std::string expression)
{
    return std::make_shared<MockObject>(MockObject{ std::move(expression) });
}

std::string value_name(ValueId id)
{
    std::ostringstream stream;
    stream << '%' << id;
    return stream.str();
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
        const auto source = std::static_pointer_cast<MockObject>(request.source_object);
        return object(source->expression);
    }

    std::vector<GpuCommCopyRequest> requests;
};

class ExpressionComputeHandler final : public ScheduleOpHandler
{
public:
    void execute(const MgpuOp &op, MgpuObjectStore &object_store) override
    {
        switch (op.kind)
        {
        case MgpuOpKind::Add:
            define_binary(op, object_store, "add");
            return;
        case MgpuOpKind::AddPlain:
            define_binary(op, object_store, "add_plain");
            return;
        case MgpuOpKind::Multiply:
            define_binary(op, object_store, "mul");
            return;
        case MgpuOpKind::MultiplyPlain:
            define_binary(op, object_store, "mul_plain");
            return;
        case MgpuOpKind::Relinearize:
            define_unary(op, object_store, "relinearize");
            return;
        case MgpuOpKind::Rescale:
            define_unary(op, object_store, "rescale");
            return;
        case MgpuOpKind::Rotate:
            define_rotate(op, object_store);
            return;
        case MgpuOpKind::BootstrapFallback:
            define_unary(op, object_store, "bootstrap");
            return;
        default:
            return;
        }
    }

    std::vector<MgpuOpKind> executed_ops;

private:
    static std::string expression_for(const MgpuObjectStore &object_store, ValueId id)
    {
        const auto input = object_store.object_as<MockObject>(id);
        if (input == nullptr)
        {
            throw std::runtime_error("missing object payload for " + value_name(id));
        }
        return input->expression;
    }

    void define_binary(
        const MgpuOp &op, MgpuObjectStore &object_store, const char *name)
    {
        const std::string left = expression_for(object_store, op.inputs[0].id);
        const std::string right = expression_for(object_store, op.inputs[1].id);
        define_output(op, object_store, std::string(name) + "(" + left + "," + right + ")");
    }

    void define_unary(
        const MgpuOp &op, MgpuObjectStore &object_store, const char *name)
    {
        const std::string input = expression_for(object_store, op.inputs[0].id);
        define_output(op, object_store, std::string(name) + "(" + input + ")");
    }

    void define_rotate(const MgpuOp &op, MgpuObjectStore &object_store)
    {
        const std::string input = expression_for(object_store, op.inputs[0].id);
        const auto iter = op.integer_attributes.find("rotate_step");
        if (iter == op.integer_attributes.end())
        {
            throw std::runtime_error("missing rotate_step in compute handler");
        }

        std::ostringstream stream;
        stream << "rotate(" << iter->second << "," << input << ")";
        define_output(op, object_store, stream.str());
    }

    void define_output(
        const MgpuOp &op, MgpuObjectStore &object_store, std::string expression)
    {
        object_store.define(
            op.outputs[0].id, MgpuValueKind::Ciphertext, op.device_id,
            object(std::move(expression)));
        executed_ops.push_back(op.kind);
    }
};

std::string make_resnet_like_hevm_binary()
{
    return test::make_hevm_binary(
        2, 1, 9, 2, { 8 },
        {
            test::HevmOpRecord{ 0, 0, 0, test::make_hevm_encode_attr(5, 45) },
            test::HevmOpRecord{ 0, 1, 0, test::make_hevm_encode_attr(4, 40) },
            test::HevmOpRecord{ 9, 2, 0, 0 },
            test::HevmOpRecord{ 3, 3, 2, 0 },
            test::HevmOpRecord{ 1, 4, 3, 1 },
            test::HevmOpRecord{ 9, 5, 1, 1 },
            test::HevmOpRecord{ 6, 6, 4, 5 },
            test::HevmOpRecord{ 8, 7, 6, 6 },
            test::HevmOpRecord{ 3, 8, 7, 0 },
        });
}

StaticSchedulePipelineResult prepare_resnet_like_schedule()
{
    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.policy = StaticPlacementPolicy::RoundRobinCompute;

    return prepare_dacapo_static_schedule(
        make_resnet_like_hevm_binary(),
        DacapoAdapterOptions{ DacapoInputFormat::HevmBinary }, options);
}

void bind_uploads(const MgpuSchedule &schedule, IoBindingScheduleHandler &io)
{
    int cipher_index = 0;
    int plain_index = 0;
    for (const MgpuOp &op : schedule.ops)
    {
        if (op.kind == MgpuOpKind::UploadCipher)
        {
            io.bind_cipher_upload(
                op.outputs[0].id, object("cipher_arg_" + std::to_string(cipher_index++)));
        }
        else if (op.kind == MgpuOpKind::UploadPlain)
        {
            io.bind_plain_upload(
                op.outputs[0].id, object("plain_const_" + std::to_string(plain_index++)));
        }
    }
}

void test_static_hevm_pipeline_executes_through_interpreter_handlers()
{
    const StaticSchedulePipelineResult pipeline = prepare_resnet_like_schedule();
    require(pipeline.ok(), "pipeline failed:\n" + pipeline.format_diagnostics());

    ExpressionComputeHandler compute;
    RecordingGpuComm comm;
    CopyDispatchingScheduleHandler copy_handler(comm, &compute);
    IoBindingScheduleHandler io(&copy_handler);
    bind_uploads(pipeline.schedule, io);

    ScheduleInterpreter interpreter(ScheduleInterpreterOptions{ 2 });
    const ScheduleExecutionResult execution = interpreter.run(pipeline.schedule, io);

    require(execution.ok(), "interpreter failed:\n" + execution.format_errors());
    require(comm.requests.size() == 8, "expected eight explicit copy requests");
    require(compute.executed_ops.size() == 7, "expected seven compute operations");

    std::size_t cross_device_copies = 0;
    std::size_t plain_copies = 0;
    for (const GpuCommCopyRequest &request : comm.requests)
    {
        if (request.source_device != request.destination_device)
        {
            ++cross_device_copies;
        }
        if (request.kind == MgpuValueKind::Plaintext)
        {
            ++plain_copies;
        }
    }
    require(cross_device_copies == comm.requests.size(), "all inserted copies should cross devices");
    require(plain_copies == 1, "expected one plaintext copy for the second mul_plain");

    const ValueId download_id = pipeline.schedule.ops.back().inputs[0].id;
    require(io.has_download(download_id), "download should record the final value");
    const auto downloaded = std::static_pointer_cast<MockObject>(io.downloaded_object(download_id));
    require(downloaded != nullptr, "downloaded object should be materialized");
    require_contains(downloaded->expression, "rotate(1,rescale(mul_plain(cipher_arg_0");
    require_contains(downloaded->expression, "mul_plain(cipher_arg_1,plain_const_1)");
    require_contains(downloaded->expression, "rescale(mul(add(");
}

}  // namespace

int main()
{
    try
    {
        test_static_hevm_pipeline_executes_through_interpreter_handlers();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu static pipeline execution test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu static pipeline execution tests passed\n";
    return EXIT_SUCCESS;
}
