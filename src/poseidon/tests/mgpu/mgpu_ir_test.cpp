#include "poseidon/mgpu/compiler/schedule_verifier.h"
#include "poseidon/mgpu/ir/schedule.h"
#include "poseidon/mgpu/ir/schedule_summary.h"

#include <cstdlib>
#include <iostream>
#include <limits>
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

MgpuOp op(
    MgpuOpKind kind, int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs, std::string debug_name = {})
{
    return MgpuOp{ kind, device_id, std::move(inputs), std::move(outputs), std::move(debug_name) };
}

MgpuOp op_with_attrs(
    MgpuOpKind kind, int device_id, std::vector<MgpuValueRef> inputs,
    std::vector<MgpuValueRef> outputs, std::unordered_map<std::string, std::int64_t> attrs)
{
    MgpuOp result{ kind, device_id, std::move(inputs), std::move(outputs), {} };
    result.integer_attributes = std::move(attrs);
    return result;
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

void require_valid(const MgpuSchedule &schedule, int device_count)
{
    const ScheduleVerificationResult result =
        verify_schedule(schedule, ScheduleVerifierOptions{ device_count });
    require(result.ok(), "expected valid schedule, got:\n" + result.format_errors());
}

void require_invalid_contains(
    const MgpuSchedule &schedule, int device_count, const std::string &needle)
{
    const ScheduleVerificationResult result =
        verify_schedule(schedule, ScheduleVerifierOptions{ device_count });
    require(!result.ok(), "expected invalid schedule");
    require_contains(result.format_errors(), needle);
}

void test_kind_strings()
{
    require(std::string(to_string(MgpuOpKind::MultiplyPlain)) == "multiply_plain",
            "MultiplyPlain string mismatch");
    require(std::string(to_string(MgpuOpKind::AddPlain)) == "add_plain",
            "AddPlain string mismatch");
    require(mgpu_op_kind_from_string("add_plain") == MgpuOpKind::AddPlain,
            "add_plain string should map to AddPlain");
    require(mgpu_op_kind_from_string("negate") == MgpuOpKind::Negate,
            "negate string should map to Negate");
    require(mgpu_op_kind_from_string("rotate") == MgpuOpKind::Rotate,
            "rotate string should map to Rotate");
    require(!mgpu_op_kind_from_string("unknown_op").has_value(),
            "unknown op should not parse");
    require(is_upload_op(MgpuOpKind::UploadCipher), "UploadCipher should be upload op");
    require(is_copy_op(MgpuOpKind::CopyPlain), "CopyPlain should be copy op");
    require(is_compute_op(MgpuOpKind::AddPlain), "AddPlain should be compute op");
    require(is_compute_op(MgpuOpKind::Negate), "Negate should be compute op");
    require(is_compute_op(MgpuOpKind::Rescale), "Rescale should be compute op");
    require(is_download_op(MgpuOpKind::Download), "Download should be download op");
}

MgpuSchedule make_valid_two_device_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }, "input_ct"));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(2) }, "weights"));
    schedule.ops.push_back(op(MgpuOpKind::MultiplyPlain, 0, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 1, { value(3) }, { value(4) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 1, { value(4) }, { value(5) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 1, { value(5) }, {}));
    return schedule;
}

void test_dump()
{
    const std::string dumped = dump_schedule(make_valid_two_device_schedule());
    require_contains(dumped, "mgpu.schedule");
    require_contains(dumped, "#0 [%1] = mgpu.upload_cipher device=0 name=\"input_ct\"");
    require_contains(dumped, "#2 [%3] = mgpu.multiply_plain device=0 inputs=[%1, %2]");
    require_contains(dumped, "#5 mgpu.download device=1 inputs=[%5]");
}

void test_valid_schedule()
{
    require_valid(make_valid_two_device_schedule(), 2);
}

void test_add_plain_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::AddPlain, 0, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(3) }, {}));

    require_valid(schedule, 1);
    require_contains(
        dump_schedule(schedule), "#2 [%3] = mgpu.add_plain device=0 inputs=[%1, %2]");
}

void test_negate_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Negate, 0, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(2) }, {}));

    require_valid(schedule, 1);
    require_contains(dump_schedule(schedule), "#1 [%2] = mgpu.negate device=0 inputs=[%1]");
}

void test_integer_attribute_dump()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op_with_attrs(
        MgpuOpKind::Rotate, 0, { value(1) }, { value(2) },
        { { "rotate_step", -3 }, { "level", 4 } }));

    const std::string dumped = dump_schedule(schedule);
    require_contains(
        dumped,
        "#1 [%2] = mgpu.rotate device=0 inputs=[%1] attrs={level=4, rotate_step=-3}");
}

std::size_t count_for_kind(const MgpuScheduleSummary &summary, MgpuOpKind kind)
{
    for (const MgpuOpKindCount &count : summary.op_counts)
    {
        if (count.kind == kind)
        {
            return count.count;
        }
    }
    throw std::runtime_error("missing op kind count");
}

void test_schedule_summary()
{
    const MgpuScheduleSummary summary = summarize_schedule(make_valid_two_device_schedule(), 2);
    require(summary.total_ops == 6, "summary total op count mismatch");
    require(summary.upload_ops == 2, "summary upload count mismatch");
    require(summary.copy_ops == 1, "summary copy count mismatch");
    require(summary.compute_ops == 2, "summary compute count mismatch");
    require(summary.download_ops == 1, "summary download count mismatch");
    require(count_for_kind(summary, MgpuOpKind::CopyCipher) == 1, "copy count mismatch");
    require(count_for_kind(summary, MgpuOpKind::Rescale) == 1, "rescale count mismatch");
    require(summary.device_op_counts.size() == 2, "summary device count size mismatch");
    require(summary.device_op_counts[0].count == 3, "device 0 count mismatch");
    require(summary.device_op_counts[1].count == 3, "device 1 count mismatch");

    const std::string text = dump_schedule_summary(summary);
    require_contains(text, "schedule_ops: 6");
    require_contains(text, "copy_cipher: 1");
    require_contains(text, "device 1: 3");

    const std::string json = schedule_summary_to_json(summary);
    require_contains(json, "\"total_ops\": 6");
    require_contains(json, "\"copy_cipher\": 1");
    require_contains(json, "\"device_id\": 1");
}

void test_schedule_summary_device_anomalies()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, -1, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 5, { value(1) }, {}));

    const MgpuScheduleSummary summary = summarize_schedule(schedule, 2);
    require(summary.unassigned_device_ops == 1, "unassigned device count mismatch");
    require(summary.invalid_device_ops == 1, "invalid device count mismatch");
    require(summary.device_op_counts[0].count == 0, "device 0 should not receive invalid ops");
    require(summary.device_op_counts[1].count == 0, "device 1 should not receive invalid ops");

    const std::string text = dump_schedule_summary(summary);
    require_contains(text, "unassigned: 1");
    require_contains(text, "invalid: 1");
}

void test_required_static_attributes()
{
    MgpuSchedule valid;
    valid.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    valid.ops.push_back(op_with_attrs(
        MgpuOpKind::Rotate, 0, { value(1) }, { value(2) },
        { { "rotate_step", 1 } }));
    valid.ops.push_back(op_with_attrs(
        MgpuOpKind::BootstrapFallback, 0, { value(2) }, { value(3) },
        { { "target_level", 0 } }));
    require_valid(valid, 1);

    MgpuSchedule missing_rotate_step;
    missing_rotate_step.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    missing_rotate_step.ops.push_back(
        op(MgpuOpKind::Rotate, 0, { value(1) }, { value(2) }));
    require_invalid_contains(
        missing_rotate_step, 1, "missing integer attribute 'rotate_step'");

    MgpuSchedule out_of_range_rotate_step;
    out_of_range_rotate_step.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    out_of_range_rotate_step.ops.push_back(op_with_attrs(
        MgpuOpKind::Rotate, 0, { value(1) }, { value(2) },
        { { "rotate_step", std::numeric_limits<std::int64_t>::max() } }));
    require_invalid_contains(
        out_of_range_rotate_step, 1,
        "integer attribute 'rotate_step' is out of int range");

    MgpuSchedule negative_target_level;
    negative_target_level.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    negative_target_level.ops.push_back(op_with_attrs(
        MgpuOpKind::BootstrapFallback, 0, { value(1) }, { value(2) },
        { { "target_level", -1 } }));
    require_invalid_contains(
        negative_target_level, 1,
        "integer attribute 'target_level' must be non-negative");
}

void test_invalid_device()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 2, {}, { value(1) }));
    require_invalid_contains(schedule, 1, "invalid device 2 for device_count 1");
}

void test_missing_input()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::MultiplyPlain, 0, { value(42), value(1) }, { value(2) }));
    require_invalid_contains(schedule, 1, "unknown input value %42");
}

void test_missing_copy()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Relinearize, 1, { value(1) }, { value(2) }));
    require_invalid_contains(
        schedule, 2, "input value %1 is on device 0 but op runs on device 1");
}

void test_download_missing_copy()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 1, { value(1) }, {}));
    require_invalid_contains(
        schedule, 2,
        "download input value %1 is on device 0 but op runs on device 1");
}

void test_wrong_input_kind()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Relinearize, 0, { value(1) }, { value(2) }));
    require_invalid_contains(schedule, 1, "input value %1 expected ciphertext, got plaintext");
}

void test_copy_wrong_input_kind()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));
    require_invalid_contains(
        schedule, 2, "copy input value %1 expected ciphertext, got plaintext");
}

void test_duplicate_output()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    require_invalid_contains(schedule, 1, "duplicate output value %1");
}

void test_reserved_output_id()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(0) }));
    require_invalid_contains(schedule, 1, "output value id 0 is reserved");
}

void test_invalid_device_count()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    require_invalid_contains(schedule, 0, "device_count must be positive");
}

}  // namespace

int main()
{
    try
    {
        test_kind_strings();
        test_dump();
        test_valid_schedule();
        test_add_plain_schedule();
        test_negate_schedule();
        test_integer_attribute_dump();
        test_schedule_summary();
        test_schedule_summary_device_anomalies();
        test_required_static_attributes();
        test_invalid_device();
        test_missing_input();
        test_missing_copy();
        test_download_missing_copy();
        test_wrong_input_kind();
        test_copy_wrong_input_kind();
        test_duplicate_output();
        test_reserved_output_id();
        test_invalid_device_count();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu IR test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu IR tests passed\n";
    return EXIT_SUCCESS;
}
