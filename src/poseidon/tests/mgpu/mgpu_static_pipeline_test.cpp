#include "poseidon/mgpu/compiler/static_schedule_pipeline.h"
#include "poseidon/tests/mgpu/hevm_test_utils.h"

#include <cstdlib>
#include <cstdint>
#include <iostream>
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
    std::vector<MgpuValueRef> outputs)
{
    return MgpuOp{ kind, device_id, std::move(inputs), std::move(outputs), {} };
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

MgpuSchedule make_resnet_like_schedule()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 1, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::MultiplyPlain, 1, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(4) }));
    schedule.ops.push_back(op(MgpuOpKind::Add, 1, { value(3), value(4) }, { value(5) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(5) }, {}));
    return schedule;
}

void test_pipeline_inserts_copies_and_dumps()
{
    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.emit_debug_dump = true;

    const StaticSchedulePipelineResult result =
        prepare_static_schedule(make_resnet_like_schedule(), options);

    require(result.ok(), "pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 9, "expected three inserted copies");
    require(result.schedule.ops[2].kind == MgpuOpKind::CopyCipher,
            "multiply input should be copied to device 1");
    require(result.schedule.ops[5].kind == MgpuOpKind::CopyCipher,
            "add input should be copied to device 1");
    require(result.schedule.ops[8].kind == MgpuOpKind::Download,
            "last op should remain download");
    require_contains(result.debug_dump, "mgpu.schedule");
    require_contains(result.debug_dump, "name=\"auto_copy\"");
    require_contains(result.debug_dump, "mgpu.download device=0");
}

void test_pipeline_reports_copy_insertion_errors()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::Rescale, 0, { value(99) }, { value(1) }));

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule);
    require(!result.ok(), "pipeline should fail for unknown input");
    require_contains(result.format_diagnostics(), "copy_insertion op #0");
    require_contains(result.format_diagnostics(), "unknown input value %99");
}

void test_pipeline_reports_verifier_errors()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::Rotate, kUnassignedDevice, { value(1) }, { value(2) }));

    StaticSchedulePipelineOptions options;
    options.device_count = 1;

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(!result.ok(), "pipeline should fail when required static attributes are missing");
    require_contains(result.format_diagnostics(), "verify op #1");
    require_contains(result.format_diagnostics(), "missing integer attribute 'rotate_step'");
}

void test_pipeline_places_unassigned_ops_before_copy_insertion()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, kUnassignedDevice, {}, { value(2) }));
    schedule.ops.push_back(
        op(MgpuOpKind::MultiplyPlain, kUnassignedDevice, { value(1), value(2) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(3) }, {}));

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.default_device = 1;

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(result.ok(), "pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 4, "no copies should be required after single-device placement");
    for (const MgpuOp &placed_op : result.schedule.ops)
    {
        require(placed_op.device_id == 1, "pipeline should place all ops on default device");
    }
}

void test_pipeline_preserves_integer_attributes()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, kUnassignedDevice, {}, { value(1) }));
    schedule.ops.push_back(op_with_attrs(
        MgpuOpKind::Rotate, kUnassignedDevice, { value(1) }, { value(2) },
        { { "rotate_step", 5 } }));
    schedule.ops.push_back(op(MgpuOpKind::Download, kUnassignedDevice, { value(2) }, {}));

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.default_device = 1;
    options.emit_debug_dump = true;

    const StaticSchedulePipelineResult result = prepare_static_schedule(schedule, options);
    require(result.ok(), "pipeline failed:\n" + result.format_diagnostics());
    require(
        result.schedule.ops[1].integer_attributes.at("rotate_step") == 5,
        "pipeline should preserve rotate_step");
    require_contains(result.debug_dump, "attrs={rotate_step=5}");
}

void test_pipeline_prepares_dacapo_json_input()
{
    const char *json = R"json(
{
  "version": 1,
  "ops": [
    {"kind": "upload_cipher", "device": 0, "outputs": [1]},
    {"kind": "upload_plain", "device": 1, "outputs": [2]},
    {"kind": "multiply_plain", "device": 1, "inputs": [1, 2], "outputs": [3]},
    {"kind": "download", "device": 0, "inputs": [3]}
  ]
}
)json";

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.emit_debug_dump = true;

    const StaticSchedulePipelineResult result = prepare_dacapo_static_schedule(
        json, DacapoAdapterOptions{ DacapoInputFormat::Json }, options);

    require(result.ok(), "Dacapo JSON pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 6, "expected two inserted copies");
    require(result.schedule.ops[2].kind == MgpuOpKind::CopyCipher,
            "multiply input should be copied to device 1");
    require(result.schedule.ops[5].kind == MgpuOpKind::Download,
            "last op should remain download");
    require_contains(result.debug_dump, "mgpu.copy_cipher");
    require_contains(result.debug_dump, "mgpu.download device=0");
}

void test_pipeline_prepares_hevm_binary_input()
{
    const std::string hevm = test::make_hevm_binary(
        1, 1, 3, 1, { 2 },
        {
            test::HevmOpRecord{ 0, 0, 0, test::make_hevm_encode_attr(4, 40) },
            test::HevmOpRecord{ 9, 1, 0, 0 },
            test::HevmOpRecord{ 1, 2, 1, static_cast<std::uint16_t>(-1) },
        },
        test::HevmConfigMetadata{
            { 45 },
            { 12 },
            { 40 },
            { 8 },
            16,
        });

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.policy = StaticPlacementPolicy::RoundRobinCompute;
    options.emit_debug_dump = true;

    const StaticSchedulePipelineResult result = prepare_dacapo_static_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary }, options);

    require(result.ok(), "Dacapo HEVM pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 6, "expected one inserted copy");
    require(result.schedule.ops[0].kind == MgpuOpKind::UploadCipher, "expected HEVM arg op");
    require(
        result.schedule.ops[0].integer_attributes.at("hevm_arg_scale") == 45,
        "HEVM arg scale metadata should survive pipeline");
    require(
        result.schedule.ops[0].integer_attributes.at("hevm_arg_level") == 12,
        "HEVM arg level metadata should survive pipeline");
    require(
        result.schedule.ops[0].integer_attributes.at("hevm_init_level") == 16,
        "HEVM init level metadata should survive pipeline");
    require(result.schedule.ops[1].kind == MgpuOpKind::UploadPlain, "expected HEVM encode op");
    require(
        result.schedule.ops[1].integer_attributes.at("encode_level") == 4,
        "HEVM encode level should survive pipeline");
    require(
        result.schedule.ops[1].integer_attributes.at("encode_scale") == 40,
        "HEVM encode scale should survive pipeline");
    require(result.schedule.ops[3].kind == MgpuOpKind::CopyCipher,
            "rotate should receive copied ciphertext");
    require(
        result.schedule.ops[4].integer_attributes.at("rotate_step") == -1,
        "HEVM rotate_step should survive copy insertion");
    require(result.schedule.ops[5].kind == MgpuOpKind::Download,
            "last HEVM op should remain download");
    require(
        result.schedule.ops[5].integer_attributes.at("hevm_result_scale") == 40,
        "HEVM result scale metadata should survive pipeline");
    require(
        result.schedule.ops[5].integer_attributes.at("hevm_result_level") == 8,
        "HEVM result level metadata should survive pipeline");
    require(
        result.schedule.ops[4].inputs[0].id == result.schedule.ops[3].outputs[0].id,
        "rotate should consume the copied mulcp result");
    require_contains(result.debug_dump, "mgpu.multiply_plain");
    require_contains(result.debug_dump, "attrs={rotate_step=-1}");
    require_contains(result.debug_dump, "hevm_arg_scale=45");
    require_contains(result.debug_dump, "hevm_result_level=8");
    require_contains(result.debug_dump, "name=\"auto_copy\"");
}

void test_pipeline_prepares_resnet_like_hevm_binary()
{
    const std::string hevm = test::make_hevm_binary(
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

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.policy = StaticPlacementPolicy::RoundRobinCompute;
    options.emit_debug_dump = true;

    const StaticSchedulePipelineResult result = prepare_dacapo_static_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary }, options);

    require(result.ok(), "ResNet-like HEVM pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 22, "expected static copies for round-robin HEVM graph");
    require(result.schedule.ops.front().kind == MgpuOpKind::UploadCipher,
            "first HEVM op should upload the first argument");
    require(result.schedule.ops[1].kind == MgpuOpKind::UploadCipher,
            "second HEVM op should upload the residual argument");
    require(
        result.schedule.ops[2].integer_attributes.at("encode_level") == 5,
        "first HEVM encode level should survive pipeline");
    require(
        result.schedule.ops[3].integer_attributes.at("encode_scale") == 40,
        "second HEVM encode scale should survive pipeline");
    require(result.schedule.ops[8].kind == MgpuOpKind::Rotate,
            "rotate should remain after inserted input copies");
    require(result.schedule.ops[8].integer_attributes.at("rotate_step") == 1,
            "positive HEVM rotate_step should survive pipeline");
    require(result.schedule.ops[13].kind == MgpuOpKind::Add,
            "residual add should remain a ciphertext add");
    require(result.schedule.ops[15].kind == MgpuOpKind::Negate,
            "negate should remain a ciphertext unary op");
    require(result.schedule.ops[18].kind == MgpuOpKind::Multiply,
            "square activation should remain a ciphertext multiply");
    require(result.schedule.ops.back().kind == MgpuOpKind::Download,
            "last HEVM op should remain download");
    require_contains(result.debug_dump, "mgpu.add");
    require_contains(result.debug_dump, "mgpu.negate");
    require_contains(result.debug_dump, "mgpu.multiply");
    require_contains(result.debug_dump, "attrs={rotate_step=1}");
    require_contains(result.debug_dump, "name=\"auto_copy\"");
}

void test_pipeline_reports_dacapo_adapter_errors()
{
    const StaticSchedulePipelineResult result = prepare_dacapo_static_schedule(
        R"json({"version": 1, "ops": [{"kind": "bad", "device": 0}]})json",
        DacapoAdapterOptions{ DacapoInputFormat::Json });

    require(!result.ok(), "invalid Dacapo JSON pipeline input should fail");
    require_contains(result.format_diagnostics(), "dacapo_adapter");
    require_contains(result.format_diagnostics(), "unknown op kind");
    require(result.schedule.ops.empty(), "failed Dacapo adapter should not produce a schedule");
}

}  // namespace

int main()
{
    try
    {
        test_pipeline_inserts_copies_and_dumps();
        test_pipeline_reports_copy_insertion_errors();
        test_pipeline_reports_verifier_errors();
        test_pipeline_places_unassigned_ops_before_copy_insertion();
        test_pipeline_preserves_integer_attributes();
        test_pipeline_prepares_dacapo_json_input();
        test_pipeline_prepares_hevm_binary_input();
        test_pipeline_prepares_resnet_like_hevm_binary();
        test_pipeline_reports_dacapo_adapter_errors();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu static pipeline test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu static pipeline tests passed\n";
    return EXIT_SUCCESS;
}
