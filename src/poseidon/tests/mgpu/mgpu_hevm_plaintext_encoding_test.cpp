#include "poseidon/ckks_encoder.h"
#include "poseidon/mgpu/runtime/hevm_plaintext_encoding.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace poseidon;
using namespace poseidon::mgpu;

namespace
{

ParametersLiteral make_ckks_test_parameters()
{
    ParametersLiteral parms(
        CKKS,
        /*log_n=*/12,
        /*log_slots=*/11,
        /*log_scale=*/20,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        Modulus(0),
        std::vector<Modulus>{},
        std::vector<Modulus>{},
        sec_level_type::none);
    parms.set_log_modulus(std::vector<std::uint32_t>(3, 30), {});
    return parms;
}

ParametersLiteral make_bfv_test_parameters()
{
    ParametersLiteral parms(
        BFV,
        /*log_n=*/12,
        /*log_slots=*/11,
        /*log_scale=*/20,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        Modulus(65537),
        std::vector<Modulus>{},
        std::vector<Modulus>{},
        sec_level_type::none);
    parms.set_log_modulus(std::vector<std::uint32_t>(3, 30), {});
    return parms;
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

void require_close(double actual, double expected, const std::string &message)
{
    if (std::fabs(actual - expected) > 1e-3)
    {
        std::ostringstream stream;
        stream << message << ": expected " << expected << ", got " << actual;
        throw std::runtime_error(stream.str());
    }
}

HevmIoBindingPlan make_plan()
{
    HevmIoBindingPlan plan;
    plan.plain_inputs.push_back(HevmPlainInputSlot{
        /*register_id=*/0,
        /*constant_index=*/1,
        /*value_id=*/10,
        /*device_id=*/0,
        /*scale=*/20,
        /*level=*/2,
    });
    plan.plain_inputs.push_back(HevmPlainInputSlot{
        /*register_id=*/1,
        /*constant_index=*/1,
        /*value_id=*/11,
        /*device_id=*/0,
        /*scale=*/18,
        /*level=*/1,
    });
    return plan;
}

DacapoConstantTable make_constants()
{
    DacapoConstantTable constants;
    constants.values.push_back({ 99.0 });
    constants.values.push_back({ 1.25, -2.5, 3.75 });
    return constants;
}

void test_encodes_plaintexts_for_each_upload_value()
{
    const ParametersLiteral parms = make_ckks_test_parameters();
    const PoseidonContext context(parms);
    const HevmPlaintextEncodingResult result =
        encode_hevm_plain_inputs(context, make_plan(), make_constants());

    require(result.ok(), "encoding failed:\n" + result.format_diagnostics());
    require(result.plaintexts.size() == 2, "encoded plaintext count mismatch");

    const auto &parms_id_map = context.crt_context()->parms_id_map();
    require(
        result.plaintexts[0].value_id == 10,
        "first encoded plaintext value id mismatch");
    require(
        result.plaintexts[0].plaintext->parms_id() == parms_id_map.at(2),
        "first encoded plaintext level mismatch");
    require(
        result.plaintexts[0].plaintext->scale() == std::ldexp(1.0, 20),
        "first encoded plaintext scale mismatch");

    require(
        result.plaintexts[1].value_id == 11,
        "second encoded plaintext value id mismatch");
    require(
        result.plaintexts[1].plaintext->parms_id() == parms_id_map.at(1),
        "second encoded plaintext level mismatch");
    require(
        result.plaintexts[1].plaintext->scale() == std::ldexp(1.0, 18),
        "second encoded plaintext scale mismatch");

    CKKSEncoder encoder(context);
    std::vector<double> decoded;
    encoder.decode(*result.plaintexts[0].plaintext, decoded);
    require(decoded.size() >= 3, "decoded plaintext slot count mismatch");
    require_close(decoded[0], 1.25, "decoded slot 0 mismatch");
    require_close(decoded[1], -2.5, "decoded slot 1 mismatch");
    require_close(decoded[2], 3.75, "decoded slot 2 mismatch");
}

void test_reports_missing_constant()
{
    const ParametersLiteral parms = make_ckks_test_parameters();
    const PoseidonContext context(parms);
    DacapoConstantTable constants;
    constants.values.push_back({ 1.0 });

    const HevmPlaintextEncodingResult result =
        encode_hevm_plain_inputs(context, make_plan(), constants);
    require(!result.ok(), "missing constant should fail");
    require(result.plaintexts.empty(), "failed encoding should not return partial plaintexts");
    require_contains(result.format_diagnostics(), "missing Dacapo constant index 1");
}

void test_reports_missing_level()
{
    const ParametersLiteral parms = make_ckks_test_parameters();
    const PoseidonContext context(parms);
    HevmIoBindingPlan plan = make_plan();
    plan.plain_inputs[0].level = 99;

    const HevmPlaintextEncodingResult result =
        encode_hevm_plain_inputs(context, plan, make_constants());
    require(!result.ok(), "missing level should fail");
    require_contains(result.format_diagnostics(), "missing Poseidon parms_id");
}

void test_reports_non_ckks_context()
{
    const ParametersLiteral parms = make_bfv_test_parameters();
    const PoseidonContext context(parms);
    const HevmPlaintextEncodingResult result =
        encode_hevm_plain_inputs(context, make_plan(), make_constants());

    require(!result.ok(), "non-CKKS context should fail");
    require(result.plaintexts.empty(), "non-CKKS encoding should not return plaintexts");
    require_contains(
        result.format_diagnostics(),
        "HEVM plaintext constants require a CKKS context");
}

void test_bind_rejects_null_encoded_plaintext()
{
    IoBindingExecutionBackend io;
    HevmEncodedPlaintext plaintext;
    plaintext.value_id = 10;

    bool failed = false;
    try
    {
        bind_hevm_encoded_plain_inputs(io, { plaintext });
    }
    catch (const std::invalid_argument &ex)
    {
        failed = true;
        require_contains(ex.what(), "encoded HEVM plaintext must not be null");
    }
    require(failed, "null encoded plaintext should fail binding");
}

}  // namespace

int main()
{
    try
    {
        test_encodes_plaintexts_for_each_upload_value();
        test_reports_missing_constant();
        test_reports_missing_level();
        test_reports_non_ckks_context();
        test_bind_rejects_null_encoded_plaintext();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mgpu HEVM plaintext encoding test failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "mgpu HEVM plaintext encoding tests passed\n";
    return EXIT_SUCCESS;
}
