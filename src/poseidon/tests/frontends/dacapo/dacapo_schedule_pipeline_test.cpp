#include "poseidon/frontends/dacapo/dacapo_artifacts.h"
#include "poseidon/tests/frontends/dacapo/hevm_test_utils.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace poseidon::mgpu;

namespace
{

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

const MgpuOp *find_op_with_attr(
    const MgpuSchedule &schedule, MgpuOpKind kind, const std::string &attr)
{
    for (const MgpuOp &op : schedule.ops)
    {
        if (op.kind == kind && op.integer_attributes.find(attr) != op.integer_attributes.end())
        {
            return &op;
        }
    }
    return nullptr;
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
    options.scheduler.kind = StaticSchedulerKind::GreedyReady;
    options.scheduler.compute_devices = { 0, 1 };
    options.scheduler.upload_device = 0;
    options.scheduler.download_device = 0;
    options.emit_debug_dump = true;

    const StaticSchedulePipelineResult result = prepare_dacapo_static_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary }, options);

    require(result.ok(), "Dacapo HEVM pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() >= 5, "expected scheduled HEVM ops");
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
    require(
        result.schedule.ops[1].integer_attributes.at("hevm_constant_index") == 0,
        "HEVM constant index should survive pipeline");
    const MgpuOp *rotate =
        find_op_with_attr(result.schedule, MgpuOpKind::Rotate, "rotate_step");
    require(rotate != nullptr, "expected HEVM rotate op");
    require(rotate->integer_attributes.at("rotate_step") == -1,
            "HEVM rotate_step should survive scheduling");
    require(result.schedule.ops.back().kind == MgpuOpKind::Download,
            "last HEVM op should remain download");
    require(
        result.schedule.ops.back().integer_attributes.at("hevm_result_scale") == 40,
        "HEVM result scale metadata should survive pipeline");
    require(
        result.schedule.ops.back().integer_attributes.at("hevm_result_level") == 8,
        "HEVM result level metadata should survive pipeline");
    require_contains(result.debug_dump, "mgpu.multiply_plain");
    require_contains(result.debug_dump, "attrs={rotate_step=-1}");
    require_contains(result.debug_dump, "hevm_constant_index=0");
    require_contains(result.debug_dump, "hevm_arg_scale=45");
    require_contains(result.debug_dump, "hevm_result_level=8");
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
    options.scheduler.kind = StaticSchedulerKind::GreedyReady;
    options.scheduler.compute_devices = { 0, 1 };
    options.scheduler.upload_device = 0;
    options.scheduler.download_device = 0;
    options.emit_debug_dump = true;

    const StaticSchedulePipelineResult result = prepare_dacapo_static_schedule(
        hevm, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary }, options);

    require(result.ok(), "ResNet-like HEVM pipeline failed:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() >= 12, "expected scheduled HEVM graph");
    require(result.schedule.ops.front().kind == MgpuOpKind::UploadCipher,
            "first HEVM op should upload the first argument");
    require(result.schedule.ops[1].kind == MgpuOpKind::UploadCipher,
            "second HEVM op should upload the residual argument");
    require_contains(result.debug_dump, "encode_level=5");
    require_contains(result.debug_dump, "encode_scale=40");
    require(result.schedule.ops.back().kind == MgpuOpKind::Download,
            "last HEVM op should remain download");
    require_contains(result.debug_dump, "mgpu.add");
    require_contains(result.debug_dump, "mgpu.negate");
    require_contains(result.debug_dump, "mgpu.multiply");
    require_contains(result.debug_dump, "attrs={rotate_step=1}");
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
        test_pipeline_prepares_dacapo_json_input();
        test_pipeline_prepares_hevm_binary_input();
        test_pipeline_prepares_resnet_like_hevm_binary();
        test_pipeline_reports_dacapo_adapter_errors();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Dacapo schedule pipeline test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Dacapo schedule pipeline tests passed\n";
    return EXIT_SUCCESS;
}
