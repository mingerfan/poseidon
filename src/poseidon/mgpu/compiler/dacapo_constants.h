#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace poseidon::mgpu
{

struct DacapoConstantTable
{
    std::vector<std::vector<double>> values;
};

struct DacapoConstantDiagnostic
{
    std::size_t offset = 0;
    std::string message;
};

struct DacapoConstantParseResult
{
    DacapoConstantTable table;
    std::vector<DacapoConstantDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

DacapoConstantParseResult parse_dacapo_constant_file(std::string_view input);

}  // namespace poseidon::mgpu
