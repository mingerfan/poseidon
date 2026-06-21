#include "poseidon/mgpu/compiler/static_schedule_pipeline.h"
#include "poseidon/mgpu/ir/schedule_json.h"

#include <cstdlib>
#include <iostream>
#include <limits>
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

void test_parse_valid_schedule_json()
{
    const char *json = R"json(
{
  "version": 1,
  "ops": [
    {"kind": "upload_cipher", "device": 0, "outputs": [1], "name": "input"},
    {"kind": "upload_plain", "device": 0, "outputs": [2]},
    {"kind": "multiply_plain", "device": 0, "inputs": [1, 2], "outputs": [3]},
    {"kind": "download", "device": 0, "inputs": [3]}
  ]
}
)json";

    const ScheduleJsonParseResult result = parse_schedule_json(json);
    require(result.ok(), "valid JSON schedule should parse:\n" + result.format_diagnostics());
    require(result.schedule.ops.size() == 4, "parsed op count mismatch");
    require(result.schedule.ops[0].kind == MgpuOpKind::UploadCipher, "first op kind mismatch");
    require(result.schedule.ops[0].debug_name == "input", "debug name mismatch");
    require(result.schedule.ops[2].inputs[1].id == 2, "input id mismatch");
    require(result.schedule.ops[3].outputs.empty(), "download outputs should be empty");
}

void test_json_round_trip()
{
    MgpuSchedule schedule;
    schedule.ops.push_back(op(MgpuOpKind::UploadCipher, 0, {}, { value(1) }, "input"));
    schedule.ops.push_back(op(MgpuOpKind::UploadPlain, 0, {}, { value(2) }));
    schedule.ops.push_back(op(MgpuOpKind::Add, 0, { value(1), value(1) }, { value(3) }));
    schedule.ops.push_back(op(MgpuOpKind::AddPlain, 0, { value(3), value(2) }, { value(4) }));
    schedule.ops.push_back(op(MgpuOpKind::Download, 0, { value(4) }, {}));

    const std::string json = schedule_to_json(schedule);
    require_contains(json, "\"kind\": \"add\"");
    require_contains(json, "\"kind\": \"add_plain\"");
    require_contains(json, "\"name\": \"input\"");

    const ScheduleJsonParseResult parsed = parse_schedule_json(json);
    require(parsed.ok(), "round-trip JSON should parse:\n" + parsed.format_diagnostics());
    require(parsed.schedule.ops.size() == schedule.ops.size(), "round-trip op count mismatch");
    require(parsed.schedule.ops[2].kind == MgpuOpKind::Add, "round-trip op kind mismatch");
    require(parsed.schedule.ops[3].kind == MgpuOpKind::AddPlain, "round-trip add_plain kind mismatch");
    require(parsed.schedule.ops[2].outputs[0].id == 3, "round-trip output id mismatch");
}

void test_parse_large_unsigned_value_id()
{
    const char *json = R"json(
{
  "version": 1,
  "ops": [
    {"kind": "upload_cipher", "device": 0, "outputs": [18446744073709551615]}
  ]
}
)json";

    const ScheduleJsonParseResult result = parse_schedule_json(json);
    require(result.ok(), "large unsigned value id should parse:\n" + result.format_diagnostics());
    require(
        result.schedule.ops[0].outputs[0].id == std::numeric_limits<ValueId>::max(),
        "large unsigned value id mismatch");
}

void test_parse_reports_errors()
{
    const char *json = R"json(
{
  "version": 1,
  "ops": [
    {"kind": "not_an_op", "device": 0, "outputs": [1]},
    {"kind": "upload_cipher", "device": 0, "outputs": [-1]},
    {"kind": "download", "device": "gpu0", "inputs": [1]}
  ]
}
)json";

    const ScheduleJsonParseResult result = parse_schedule_json(json);
    require(!result.ok(), "invalid JSON schedule should fail");
    require_contains(result.format_diagnostics(), "unknown op kind");
    require_contains(result.format_diagnostics(), "value id must be non-negative");
    require_contains(result.format_diagnostics(), "/ops/2/device");
}

void test_malformed_json_reports_error()
{
    const ScheduleJsonParseResult result = parse_schedule_json("{");
    require(!result.ok(), "malformed JSON should fail");
    require_contains(result.format_diagnostics(), "/:");
}

void test_json_schedule_pipeline_integration()
{
    const char *json = R"json(
{
  "version": 1,
  "ops": [
    {"kind": "upload_cipher", "device": -1, "outputs": [1]},
    {"kind": "upload_plain", "device": -1, "outputs": [2]},
    {"kind": "multiply_plain", "device": -1, "inputs": [1, 2], "outputs": [3]},
    {"kind": "download", "device": -1, "inputs": [3]}
  ]
}
)json";

    const ScheduleJsonParseResult parsed = parse_schedule_json(json);
    require(parsed.ok(), "JSON schedule should parse:\n" + parsed.format_diagnostics());

    StaticSchedulePipelineOptions options;
    options.device_count = 2;
    options.placement.default_device = 1;
    const StaticSchedulePipelineResult prepared =
        prepare_static_schedule(parsed.schedule, options);
    require(prepared.ok(), "prepared JSON schedule should verify:\n" + prepared.format_diagnostics());
    require(prepared.schedule.ops.size() == 4, "single-device placement should not add copies");
    require(prepared.schedule.ops[2].device_id == 1, "compute should be placed on default device");
}

}  // namespace

int main()
{
    try
    {
        test_parse_valid_schedule_json();
        test_json_round_trip();
        test_parse_large_unsigned_value_id();
        test_parse_reports_errors();
        test_malformed_json_reports_error();
        test_json_schedule_pipeline_integration();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu schedule JSON test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu schedule JSON tests passed\n";
    return EXIT_SUCCESS;
}
