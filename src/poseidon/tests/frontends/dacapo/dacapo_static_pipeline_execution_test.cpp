#include "poseidon/frontends/dacapo/dacapo_constants.h"
#include "poseidon/frontends/dacapo/dacapo_artifacts.h"
#include "poseidon/mgpu/compiler/static_schedule_pipeline.h"
#include "poseidon/frontends/dacapo/hevm_io_binding.h"
#include "poseidon/mgpu/runtime/backend/io_binding_backend.h"
#include "poseidon/mgpu/runtime/backend/copy_dispatching_backend.h"
#include "poseidon/mgpu/runtime/executor/sequential_schedule_executor.h"
#include "poseidon/tests/frontends/dacapo/hevm_test_utils.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
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

std::shared_ptr<void> erased_object(std::string expression)
{
    return object(std::move(expression));
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

void append_i64(std::string &output, std::int64_t value)
{
    const auto bits = static_cast<std::uint64_t>(value);
    for (int i = 0; i < 8; ++i)
    {
        output.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
    }
}

void append_double(std::string &output, double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 8; ++i)
    {
        output.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
    }
}

std::string make_hevm_constant_file()
{
    std::string output;
    append_i64(output, 2);
    append_i64(output, 1);
    append_double(output, 3.5);
    append_i64(output, 1);
    append_double(output, -2.0);
    return output;
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

class ExpressionComputeBackend final : public ScheduleExecutionBackend
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
        case MgpuOpKind::Negate:
            define_unary(op, object_store, "negate");
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
            throw std::runtime_error("missing rotate_step in compute backend");
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
        2, 1, 10, 2, { 9 },
        {
            test::HevmOpRecord{ 0, 0, 0, test::make_hevm_encode_attr(5, 45) },
            test::HevmOpRecord{ 0, 1, 0, test::make_hevm_encode_attr(4, 40) },
            test::HevmOpRecord{ 9, 2, 0, 0 },
            test::HevmOpRecord{ 3, 3, 2, 0 },
            test::HevmOpRecord{ 1, 4, 3, 1 },
            test::HevmOpRecord{ 9, 5, 1, 1 },
            test::HevmOpRecord{ 6, 6, 4, 5 },
            test::HevmOpRecord{ 2, 7, 6, 0 },
            test::HevmOpRecord{ 8, 8, 7, 7 },
            test::HevmOpRecord{ 3, 9, 8, 0 },
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

std::string make_constants_pipeline_hevm_binary()
{
    return test::make_hevm_binary(
        1, 1, 2, 2, { 1 },
        {
            test::HevmOpRecord{ 0, 0, 1, test::make_hevm_encode_attr(3, 20) },
            test::HevmOpRecord{ 0, 1, 0, test::make_hevm_encode_attr(3, 20) },
            test::HevmOpRecord{ 9, 1, 0, 0 },
        },
        test::HevmConfigMetadata{
            { 20 },
            { 3 },
            { 40 },
            { 2 },
            3,
        });
}

StaticSchedulePipelineResult prepare_constants_pipeline_schedule()
{
    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.policy = StaticPlacementPolicy::RoundRobinCompute;

    return prepare_dacapo_static_schedule(
        make_constants_pipeline_hevm_binary(),
        DacapoAdapterOptions{ DacapoInputFormat::HevmBinary }, options);
}

void bind_uploads(const MgpuSchedule &schedule, IoBindingExecutionBackend &io)
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

ScheduleExecutionResult run_with_copy_dispatch(
    const MgpuSchedule &schedule, GpuComm &comm, ScheduleExecutionBackend &backend,
    SequentialScheduleExecutorOptions options)
{
    CopyDispatchingExecutionBackend copy_backend(comm, &backend);
    SequentialScheduleExecutor executor(options);
    return executor.run(schedule, copy_backend);
}

void test_static_hevm_pipeline_executes_through_executor_backends()
{
    const StaticSchedulePipelineResult pipeline = prepare_resnet_like_schedule();
    require(pipeline.ok(), "pipeline failed:\n" + pipeline.format_diagnostics());

    ExpressionComputeBackend compute;
    RecordingGpuComm comm;
    IoBindingExecutionBackend io(&compute);
    bind_uploads(pipeline.schedule, io);

    const ScheduleExecutionResult execution = run_with_copy_dispatch(
        pipeline.schedule, comm, io, SequentialScheduleExecutorOptions{ 2 });

    require(execution.ok(), "executor failed:\n" + execution.format_errors());
    require(comm.requests.size() == 9, "expected nine explicit copy requests");
    require(compute.executed_ops.size() == 8, "expected eight compute operations");

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
    require_contains(downloaded->expression, "rescale(mul(negate(add(");
}

void test_static_hevm_pipeline_binds_constants_by_dacapo_index()
{
    const StaticSchedulePipelineResult pipeline = prepare_constants_pipeline_schedule();
    require(pipeline.ok(), "pipeline failed:\n" + pipeline.format_diagnostics());

    const DacapoConstantParseResult constants =
        parse_dacapo_constant_file(make_hevm_constant_file());
    require(constants.ok(), "constant parse failed:\n" + constants.format_diagnostics());

    const HevmIoBindingPlanResult plan_result =
        build_hevm_io_binding_plan(pipeline.schedule);
    require(plan_result.ok(), "HEVM IO plan failed:\n" + plan_result.format_diagnostics());
    require(plan_result.plan.cipher_inputs.size() == 1, "expected one cipher input");
    require(plan_result.plan.plain_inputs.size() == 2, "expected two plaintext constants");
    require(plan_result.plan.results.size() == 1, "expected one result");

    ExpressionComputeBackend compute;
    RecordingGpuComm comm;
    IoBindingExecutionBackend io(&compute);
    bind_hevm_cipher_inputs(io, plan_result.plan, { erased_object("cipher_arg_0") });
    bind_hevm_plain_inputs_by_constant_index(
        io, plan_result.plan,
        {
            { 0, erased_object("const0:3.5") },
            { 1, erased_object("const1:-2.0") },
        });

    const ScheduleExecutionResult execution = run_with_copy_dispatch(
        pipeline.schedule, comm, io, SequentialScheduleExecutorOptions{ 2 });
    require(execution.ok(), "executor failed:\n" + execution.format_errors());

    const std::vector<std::shared_ptr<void>> raw_results =
        collect_hevm_results(io, plan_result.plan);
    require(raw_results.size() == 1, "expected one collected HEVM result");
    const auto downloaded = std::static_pointer_cast<MockObject>(raw_results[0]);
    require(downloaded != nullptr, "downloaded object should be materialized");
    require_contains(downloaded->expression, "mul_plain(cipher_arg_0,const1:-2.0)");
    require(
        downloaded->expression.find("const0:3.5") == std::string::npos,
        "multiply should use constant index 1, not upload order");
}

}  // namespace

int main()
{
    try
    {
        test_static_hevm_pipeline_executes_through_executor_backends();
        test_static_hevm_pipeline_binds_constants_by_dacapo_index();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu static pipeline execution test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu static pipeline execution tests passed\n";
    return EXIT_SUCCESS;
}
