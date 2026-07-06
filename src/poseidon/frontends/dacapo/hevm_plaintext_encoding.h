#pragma once

#include "poseidon/frontends/dacapo/dacapo_constants.h"
#include "poseidon/frontends/dacapo/hevm_io_binding.h"
#include "poseidon/mgpu/runtime/backend/io_binding_backend.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct HevmEncodedPlaintext
{
    ValueId value_id = 0;
    std::uint64_t constant_index = 0;
    std::uint64_t register_id = 0;
    int device_id = 0;
    std::uint64_t scale = 0;
    std::uint64_t level = 0;
    std::shared_ptr<Plaintext> plaintext;
};

struct HevmPlaintextEncodingDiagnostic
{
    ValueId value_id = 0;
    std::string message;
};

struct HevmPlaintextEncodingResult
{
    std::vector<HevmEncodedPlaintext> plaintexts;
    std::vector<HevmPlaintextEncodingDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

HevmPlaintextEncodingResult encode_hevm_plain_inputs(
    const PoseidonContext &context, const HevmIoBindingPlan &plan,
    const DacapoConstantTable &constants);

void bind_hevm_encoded_plain_inputs(
    IoBindingExecutionBackend &io, const std::vector<HevmEncodedPlaintext> &plaintexts);

}  // namespace poseidon::mgpu
