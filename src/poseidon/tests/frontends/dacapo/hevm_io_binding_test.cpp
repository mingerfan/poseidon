#include "poseidon/frontends/dacapo/dacapo_artifacts.h"
#include "poseidon/mgpu/compiler/static_schedule_pipeline.h"
#include "poseidon/frontends/dacapo/hevm_io_binding.h"
#include "poseidon/mgpu/runtime/executor/sequential_schedule_executor.h"
#include "poseidon/tests/frontends/dacapo/hevm_test_utils.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace poseidon::mgpu;

namespace
{

MgpuValueRef value(ValueId id)
{
    return MgpuValueRef{ id };
}

MgpuOp op_with_attrs(
    MgpuOpKind kind, int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs, std::unordered_map<std::string, std::int64_t> attrs)
{
    MgpuOp result{ kind, device_id, std::move(inputs), std::move(outputs), {} };
    result.integer_attributes = std::move(attrs);
    return result;
}

std::shared_ptr<std::string> string_object(const char *value)
{
    return std::make_shared<std::string>(value);
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

std::string make_two_result_hevm_binary()
{
    return test::make_hevm_binary(
        2, 2, 2, 0, { 1, 0 }, {},
        test::HevmConfigMetadata{
            { 45, 46 },
            { 12, 13 },
            { 40, 41 },
            { 8, 9 },
            16,
        });
}

StaticSchedulePipelineResult prepare_two_result_hevm_schedule()
{
    StaticSchedulePipelineOptions options;
    options.device_count = 1;

    return prepare_dacapo_static_schedule(
        make_two_result_hevm_binary(),
        DacapoAdapterOptions{ DacapoInputFormat::HevmBinary }, options);
}

StaticSchedulePipelineResult prepare_plain_input_hevm_schedule()
{
    const std::string hevm = test::make_hevm_binary(
        1, 1, 1, 1, { 0 },
        {
            test::HevmOpRecord{ 0, 0, 7, test::make_hevm_encode_attr(4, 40) },
        },
        test::HevmConfigMetadata{
            { 45 },
            { 12 },
            { 45 },
            { 12 },
            16,
        });

    StaticSchedulePipelineOptions options;
    options.device_count = 1;

    return prepare_dacapo_static_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary }, options);
}

void test_builds_plan_and_binds_cipher_inputs_by_hevm_index()
{
    const StaticSchedulePipelineResult pipeline = prepare_two_result_hevm_schedule();
    require(pipeline.ok(), "pipeline failed:\n" + pipeline.format_diagnostics());

    const HevmIoBindingPlanResult plan_result =
        build_hevm_io_binding_plan(pipeline.schedule);
    require(plan_result.ok(), "HEVM IO plan failed:\n" + plan_result.format_diagnostics());

    const HevmIoBindingPlan &plan = plan_result.plan;
    require(plan.cipher_inputs.size() == 2, "expected two HEVM cipher inputs");
    require(plan.cipher_inputs[0].index == 0, "first cipher input index mismatch");
    require(plan.cipher_inputs[0].scale == 45, "first cipher input scale mismatch");
    require(plan.cipher_inputs[0].level == 12, "first cipher input level mismatch");
    require(plan.cipher_inputs[0].init_level == 16, "first cipher init level mismatch");
    require(plan.cipher_inputs[1].index == 1, "second cipher input index mismatch");
    require(plan.cipher_inputs[1].scale == 46, "second cipher input scale mismatch");

    require(plan.results.size() == 2, "expected two HEVM results");
    require(plan.results[0].index == 0, "first result index mismatch");
    require(plan.results[0].register_id == 1, "first result register mismatch");
    require(plan.results[0].scale == 40, "first result scale mismatch");
    require(plan.results[0].level == 8, "first result level mismatch");
    require(plan.results[1].index == 1, "second result index mismatch");
    require(plan.results[1].register_id == 0, "second result register mismatch");
    require(plan.results[1].scale == 41, "second result scale mismatch");
    require(plan.results[1].level == 9, "second result level mismatch");

    IoBindingExecutionBackend io;
    bind_hevm_cipher_inputs(
        io, plan,
        {
            string_object("cipher_arg_0"),
            string_object("cipher_arg_1"),
        });

    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 1 });
    const ScheduleExecutionResult execution = executor.run(pipeline.schedule, io);
    require(execution.ok(), "executor failed:\n" + execution.format_errors());

    const std::vector<std::shared_ptr<void>> raw_results = collect_hevm_results(io, plan);
    require(raw_results.size() == 2, "expected two collected HEVM results");
    const auto result_0 = std::static_pointer_cast<std::string>(raw_results[0]);
    const auto result_1 = std::static_pointer_cast<std::string>(raw_results[1]);
    require(*result_0 == "cipher_arg_1", "HEVM result 0 should map to register 1");
    require(*result_1 == "cipher_arg_0", "HEVM result 1 should map to register 0");
}

void test_binds_plain_inputs_by_hevm_constant_index()
{
    const StaticSchedulePipelineResult pipeline = prepare_plain_input_hevm_schedule();
    require(pipeline.ok(), "pipeline failed:\n" + pipeline.format_diagnostics());

    const HevmIoBindingPlanResult plan_result =
        build_hevm_io_binding_plan(pipeline.schedule);
    require(plan_result.ok(), "HEVM IO plan failed:\n" + plan_result.format_diagnostics());

    const HevmIoBindingPlan &plan = plan_result.plan;
    require(plan.cipher_inputs.size() == 1, "expected one HEVM cipher input");
    require(plan.plain_inputs.size() == 1, "expected one HEVM plaintext input");
    require(plan.plain_inputs[0].register_id == 0, "plain register metadata mismatch");
    require(plan.plain_inputs[0].constant_index == 7, "constant index metadata mismatch");
    require(plan.plain_inputs[0].level == 4, "plain encode level metadata mismatch");
    require(plan.plain_inputs[0].scale == 40, "plain encode scale metadata mismatch");

    IoBindingExecutionBackend io;
    bind_hevm_cipher_inputs(io, plan, { string_object("cipher_arg_0") });
    bind_hevm_plain_inputs_by_constant_index(
        io, plan,
        {
            { 7, string_object("plain_const_7") },
        });

    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 1 });
    const ScheduleExecutionResult execution = executor.run(pipeline.schedule, io);
    require(execution.ok(), "executor failed:\n" + execution.format_errors());

    const auto uploaded_plain =
        execution.object_store.object_as<std::string>(plan.plain_inputs[0].value_id);
    require(*uploaded_plain == "plain_const_7", "plain upload object mismatch");
}

void test_binds_reused_plain_constant_to_each_upload_value()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op_with_attrs(
        MgpuOpKind::UploadPlain, 0, {}, { value(10) },
        {
            { "hevm_plain_register", 1 },
            { "hevm_constant_index", 7 },
            { "encode_level", 4 },
            { "encode_scale", 40 },
        }));
    schedule.ops.push_back(op_with_attrs(
        MgpuOpKind::UploadPlain, 0, {}, { value(11) },
        {
            { "hevm_plain_register", 0 },
            { "hevm_constant_index", 7 },
            { "encode_level", 4 },
            { "encode_scale", 40 },
        }));

    const HevmIoBindingPlanResult plan_result =
        build_hevm_io_binding_plan(schedule);
    require(
        plan_result.ok(),
        "reused HEVM plaintext constant plan should pass:\n" +
            plan_result.format_diagnostics());
    require(
        plan_result.plan.plain_inputs.size() == 2,
        "expected two plaintext upload slots");
    require(
        plan_result.plan.plain_inputs[0].register_id == 0,
        "plaintext slots should be sorted by register id");
    require(
        plan_result.plan.plain_inputs[1].register_id == 1,
        "second plaintext register mismatch");

    IoBindingExecutionBackend io;
    const auto reused_plain = string_object("plain_const_7");
    bind_hevm_plain_inputs_by_constant_index(
        io, plan_result.plan,
        {
            { 7, reused_plain },
        });

    SequentialScheduleExecutor executor(SequentialScheduleExecutorOptions{ 1 });
    const ScheduleExecutionResult execution = executor.run(schedule, io);
    require(execution.ok(), "executor failed:\n" + execution.format_errors());
    require(
        execution.object_store.object_as<std::string>(10) == reused_plain,
        "first reused plaintext upload mismatch");
    require(
        execution.object_store.object_as<std::string>(11) == reused_plain,
        "second reused plaintext upload mismatch");
}

void test_reports_duplicate_hevm_input_indices()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op_with_attrs(
        MgpuOpKind::UploadCipher, 0, {}, { value(1) },
        {
            { "hevm_arg_index", 0 },
            { "hevm_arg_scale", 45 },
            { "hevm_arg_level", 12 },
            { "hevm_init_level", 16 },
        }));
    schedule.ops.push_back(op_with_attrs(
        MgpuOpKind::UploadCipher, 0, {}, { value(2) },
        {
            { "hevm_arg_index", 0 },
            { "hevm_arg_scale", 46 },
            { "hevm_arg_level", 13 },
            { "hevm_init_level", 16 },
        }));

    const HevmIoBindingPlanResult result = build_hevm_io_binding_plan(schedule);
    require(!result.ok(), "duplicate HEVM input indices should fail");
    require_contains(result.format_diagnostics(), "duplicate HEVM cipher input index 0");
}

void test_reports_incomplete_hevm_metadata()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op_with_attrs(
        MgpuOpKind::Download, 0, { value(1) }, {},
        {
            { "hevm_result_index", 0 },
            { "hevm_result_register", 1 },
            { "hevm_result_scale", 40 },
        }));

    const HevmIoBindingPlanResult result = build_hevm_io_binding_plan(schedule);
    require(!result.ok(), "incomplete HEVM result metadata should fail");
    require_contains(result.format_diagnostics(), "missing HEVM integer attribute");
    require_contains(result.format_diagnostics(), "hevm_result_level");
}

void test_bind_rejects_input_count_mismatch()
{
    const StaticSchedulePipelineResult pipeline = prepare_two_result_hevm_schedule();
    require(pipeline.ok(), "pipeline failed:\n" + pipeline.format_diagnostics());
    const HevmIoBindingPlanResult plan_result =
        build_hevm_io_binding_plan(pipeline.schedule);
    require(plan_result.ok(), "HEVM IO plan failed:\n" + plan_result.format_diagnostics());

    IoBindingExecutionBackend io;
    try
    {
        bind_hevm_cipher_inputs(io, plan_result.plan, { string_object("only_arg") });
    }
    catch (const std::invalid_argument &ex)
    {
        require_contains(ex.what(), "HEVM cipher input object count 1");
        return;
    }

    throw std::runtime_error("input count mismatch should throw");
}

void test_plain_bind_rejects_missing_constant()
{
    const StaticSchedulePipelineResult pipeline = prepare_plain_input_hevm_schedule();
    require(pipeline.ok(), "pipeline failed:\n" + pipeline.format_diagnostics());
    const HevmIoBindingPlanResult plan_result =
        build_hevm_io_binding_plan(pipeline.schedule);
    require(plan_result.ok(), "HEVM IO plan failed:\n" + plan_result.format_diagnostics());

    IoBindingExecutionBackend io;
    try
    {
        bind_hevm_plain_inputs_by_constant_index(io, plan_result.plan, {});
    }
    catch (const std::invalid_argument &ex)
    {
        require_contains(ex.what(), "missing HEVM plaintext object for constant index 7");
        return;
    }

    throw std::runtime_error("missing plaintext constant should throw");
}

void test_collect_results_rejects_missing_download()
{
    HevmIoBindingPlan plan;
    plan.results.push_back(HevmResultSlot{
        /*index=*/0,
        /*register_id=*/0,
        /*value_id=*/42,
        /*device_id=*/0,
        /*scale=*/40,
        /*level=*/2,
    });

    IoBindingExecutionBackend io;
    try
    {
        (void)collect_hevm_results(io, plan);
    }
    catch (const std::out_of_range &ex)
    {
        require_contains(ex.what(), "missing download for %42");
        return;
    }

    throw std::runtime_error("missing HEVM result download should throw");
}

}  // namespace

int main()
{
    try
    {
        test_builds_plan_and_binds_cipher_inputs_by_hevm_index();
        test_binds_plain_inputs_by_hevm_constant_index();
        test_binds_reused_plain_constant_to_each_upload_value();
        test_reports_duplicate_hevm_input_indices();
        test_reports_incomplete_hevm_metadata();
        test_bind_rejects_input_count_mismatch();
        test_plain_bind_rejects_missing_constant();
        test_collect_results_rejects_missing_download();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu HEVM IO binding test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu HEVM IO binding tests passed\n";
    return EXIT_SUCCESS;
}
