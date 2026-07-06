#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace poseidon::mgpu
{

enum class DacapoInputFormat
{
    Unknown,
    EarthMlirText,
    HevmBinary,
    Json
};

struct DacapoAdapterOptions
{
    DacapoInputFormat input_format = DacapoInputFormat::Unknown;
};

struct DacapoAdapterDiagnostic
{
    std::size_t offset = 0;
    std::string message;
};

struct DacapoAdapterResult
{
    MgpuSchedule schedule;
    std::vector<DacapoAdapterDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

struct DacapoHevmOpcodeCount
{
    std::uint16_t opcode = 0;
    std::uint64_t count = 0;
    std::string name;
    bool supported = false;
};

struct DacapoHevmOpcodeSummary
{
    std::uint64_t operation_count = 0;
    std::uint64_t alloc_count = 0;
    std::vector<DacapoHevmOpcodeCount> opcode_counts;
    std::vector<DacapoAdapterDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

const char *to_string(DacapoInputFormat format) noexcept;

std::optional<std::string_view> hevm_opcode_name(std::uint16_t opcode) noexcept;
bool is_supported_hevm_opcode(std::uint16_t opcode) noexcept;

// Scans HEVM operation records without translating registers into MgpuSchedule.
DacapoHevmOpcodeSummary summarize_hevm_opcodes(std::string_view input);

DacapoAdapterResult translate_dacapo_schedule(
    std::string_view input, const DacapoAdapterOptions &options = {});

}  // namespace poseidon::mgpu
