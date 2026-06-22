#include "poseidon/mgpu/runtime/hevm_plaintext_encoding.h"

#include "poseidon/ckks_encoder.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

void add_diagnostic(
    HevmPlaintextEncodingResult &result, ValueId value_id, std::string message)
{
    result.diagnostics.push_back(
        HevmPlaintextEncodingDiagnostic{ value_id, std::move(message) });
}

std::string value_name(ValueId id)
{
    std::ostringstream stream;
    stream << '%' << id;
    return stream.str();
}

bool scale_to_double(
    HevmPlaintextEncodingResult &result, const HevmPlainInputSlot &slot, double &scale)
{
    if (slot.scale > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    {
        add_diagnostic(
            result, slot.value_id,
            "HEVM plaintext scale exponent is too large for Poseidon CKKS encoding");
        return false;
    }

    scale = std::ldexp(1.0, static_cast<int>(slot.scale));
    if (!std::isfinite(scale) || scale <= 0.0)
    {
        add_diagnostic(
            result, slot.value_id,
            "HEVM plaintext scale exponent produced an invalid CKKS scale");
        return false;
    }
    return true;
}

bool parms_id_for_level(
    HevmPlaintextEncodingResult &result, const PoseidonContext &context,
    const HevmPlainInputSlot &slot, parms_id_type &parms_id)
{
    if (slot.level > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        add_diagnostic(
            result, slot.value_id, "HEVM plaintext level exceeds uint32_t");
        return false;
    }

    const auto &parms_id_map = context.crt_context()->parms_id_map();
    const auto iter = parms_id_map.find(static_cast<std::uint32_t>(slot.level));
    if (iter == parms_id_map.end())
    {
        std::ostringstream stream;
        stream << "missing Poseidon parms_id for HEVM plaintext level " << slot.level;
        add_diagnostic(result, slot.value_id, stream.str());
        return false;
    }

    parms_id = iter->second;
    return true;
}

const std::vector<double> *constant_for_slot(
    HevmPlaintextEncodingResult &result, const HevmPlainInputSlot &slot,
    const DacapoConstantTable &constants)
{
    if (slot.constant_index >= constants.values.size())
    {
        std::ostringstream stream;
        stream << "missing Dacapo constant index " << slot.constant_index
               << " for HEVM plaintext upload " << value_name(slot.value_id);
        add_diagnostic(result, slot.value_id, stream.str());
        return nullptr;
    }

    return &constants.values[static_cast<std::size_t>(slot.constant_index)];
}

}  // namespace

std::string HevmPlaintextEncodingResult::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << value_name(diagnostics[i].value_id) << ": " << diagnostics[i].message;
    }
    return stream.str();
}

HevmPlaintextEncodingResult encode_hevm_plain_inputs(
    const PoseidonContext &context, const HevmIoBindingPlan &plan,
    const DacapoConstantTable &constants)
{
    HevmPlaintextEncodingResult result;
    if (context.parameters_literal()->scheme() != CKKS)
    {
        add_diagnostic(result, 0, "HEVM plaintext constants require a CKKS context");
        return result;
    }

    CKKSEncoder encoder(context);
    for (const HevmPlainInputSlot &slot : plan.plain_inputs)
    {
        const std::vector<double> *constant = constant_for_slot(result, slot, constants);
        double scale = 0.0;
        parms_id_type parms_id = parms_id_zero;
        if (constant == nullptr ||
            !scale_to_double(result, slot, scale) ||
            !parms_id_for_level(result, context, slot, parms_id))
        {
            continue;
        }

        auto plaintext = std::make_shared<Plaintext>();
        try
        {
            encoder.encode(*constant, parms_id, scale, *plaintext);
        }
        catch (const std::exception &ex)
        {
            add_diagnostic(result, slot.value_id, ex.what());
            continue;
        }

        result.plaintexts.push_back(HevmEncodedPlaintext{
            slot.value_id,
            slot.constant_index,
            slot.register_id,
            slot.device_id,
            slot.scale,
            slot.level,
            std::move(plaintext),
        });
    }

    if (!result.ok())
    {
        result.plaintexts.clear();
    }
    return result;
}

void bind_hevm_encoded_plain_inputs(
    IoBindingExecutionBackend &io, const std::vector<HevmEncodedPlaintext> &plaintexts)
{
    for (const HevmEncodedPlaintext &plaintext : plaintexts)
    {
        if (plaintext.plaintext == nullptr)
        {
            throw std::invalid_argument("encoded HEVM plaintext must not be null");
        }
        io.bind_plain_upload(plaintext.value_id, plaintext.plaintext);
    }
}

}  // namespace poseidon::mgpu
