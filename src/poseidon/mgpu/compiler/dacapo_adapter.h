#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace poseidon::mgpu
{

enum class DacapoInputFormat
{
    Unknown,
    EarthMlirText,
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

const char *to_string(DacapoInputFormat format) noexcept;

DacapoAdapterResult translate_dacapo_schedule(
    std::string_view input, const DacapoAdapterOptions &options = {});

}  // namespace poseidon::mgpu
