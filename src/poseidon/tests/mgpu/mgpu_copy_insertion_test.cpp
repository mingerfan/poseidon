#include "poseidon/mgpu/compiler/copy_insertion.h"
#include "poseidon/mgpu/compiler/schedule_verifier.h"

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
    std::vector<MgpuValueRef> outputs, std::string debug_name = {})
{
    return MgpuOp{ kind, device_id, std::move(inputs), std::move(outputs), std::move(debug_name) };
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

void require_verifies(const MgpuSchedule &schedule, int device_count)
{
    const ScheduleVerificationResult verification =
        verify_schedule(schedule, ScheduleVerifierOptions{ device_count });
    require(verification.ok(), "rewritten schedule did not verify:\n" + verification.format_errors());
}

void test_inserts_cipher_copy_for_unary_op()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 1, { value(1) }, { value(2) }));

    const CopyInsertionResult result = insert_required_copies(schedule);
    require(result.ok(), "copy insertion failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 3, "expected one inserted copy op");
    require(result.schedule.ops[1].kind == MgpuOpKind::CopyCipher, "inserted op kind mismatch");
    require(result.schedule.ops[1].device_id == 1, "copy destination device mismatch");
    require(result.schedule.ops[1].inputs[0].id == 1, "copy input mismatch");
    require(result.schedule.ops[1].outputs[0].id == 3, "copy output id mismatch");
    require(result.schedule.ops[2].inputs[0].id == 3, "rescale input should be rewritten");
    require_verifies(result.schedule, 2);
}

void test_inserts_only_missing_inputs()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 1, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::MultiplyPlain, 1, { value(1), value(2) }, { value(3) }));

    const CopyInsertionResult result = insert_required_copies(schedule);
    require(result.ok(), "copy insertion failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 4, "expected one inserted copy op");
    require(result.schedule.ops[2].kind == MgpuOpKind::CopyCipher, "expected ciphertext copy");
    require(result.schedule.ops[2].outputs[0].id == 4, "copy id should be allocated after max id");
    require(result.schedule.ops[3].inputs[0].id == 4, "cipher input should be rewritten");
    require(result.schedule.ops[3].inputs[1].id == 2, "local plaintext should stay unchanged");
    require_verifies(result.schedule, 2);
}

void test_add_plain_inserts_cipher_and_plain_copies()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 1, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::AddPlain, 2, { value(1), value(2) }, { value(3) }));

    const CopyInsertionResult result = insert_required_copies(schedule);
    require(result.ok(), "copy insertion failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 5, "expected two inserted copy ops");
    require(result.schedule.ops[2].kind == MgpuOpKind::CopyCipher, "expected ciphertext copy");
    require(result.schedule.ops[2].outputs[0].id == 4, "cipher copy id mismatch");
    require(result.schedule.ops[3].kind == MgpuOpKind::CopyPlain, "expected plaintext copy");
    require(result.schedule.ops[3].outputs[0].id == 5, "plain copy id mismatch");
    require(result.schedule.ops[4].inputs[0].id == 4, "cipher input should be rewritten");
    require(result.schedule.ops[4].inputs[1].id == 5, "plain input should be rewritten");
    require_verifies(result.schedule, 3);
}

void test_existing_copy_is_preserved()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::CopyCipher, 1, { value(1) }, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 1, { value(2) }, { value(3) }));

    const CopyInsertionResult result = insert_required_copies(schedule);
    require(result.ok(), "copy insertion failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 3, "existing copy should not create another copy");
    require(result.schedule.ops[1].kind == MgpuOpKind::CopyCipher, "existing copy changed");
    require(result.schedule.ops[2].inputs[0].id == 2, "rescale should use existing copied value");
    require_verifies(result.schedule, 2);
}

void test_download_plaintext_copy_does_not_require_ciphertext()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 1, { value(1) }, {}));

    const CopyInsertionResult result = insert_required_copies(schedule);
    require(result.ok(), "copy insertion failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 3, "expected plaintext copy before download");
    require(result.schedule.ops[1].kind == MgpuOpKind::CopyPlain, "expected plaintext copy");
    require(result.schedule.ops[2].inputs[0].id == 2, "download input should be rewritten");
    require_verifies(result.schedule, 2);
}

void test_unknown_input_reports_diagnostic()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 0, { value(99) }, { value(1) }));

    const CopyInsertionResult result = insert_required_copies(schedule);
    require(!result.ok(), "unknown input should fail copy insertion");
    require_contains(result.format_diagnostics(), "unknown input value %99");
}

}  // namespace

int main()
{
    try
    {
        test_inserts_cipher_copy_for_unary_op();
        test_inserts_only_missing_inputs();
        test_add_plain_inserts_cipher_and_plain_copies();
        test_existing_copy_is_preserved();
        test_download_plaintext_copy_does_not_require_ciphertext();
        test_unknown_input_reports_diagnostic();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu copy insertion test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu copy insertion tests passed\n";
    return EXIT_SUCCESS;
}
