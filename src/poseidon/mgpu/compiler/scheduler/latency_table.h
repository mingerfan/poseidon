#pragma once

#include "poseidon/mgpu/ir/schedule.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace poseidon::mgpu
{

struct LatencyTableParseDiagnostic
{
    std::string path;
    std::string message;
};

struct LatencyTable
{
    std::unordered_map<std::string, std::vector<double>> entries;

    double latency_for(MgpuOpKind kind, std::size_t index = 0) const;
};

struct LatencyTableParseResult
{
    LatencyTable table;
    std::vector<LatencyTableParseDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

LatencyTable make_default_latency_table();
LatencyTableParseResult parse_latency_table_json(std::string_view text);

const char *latency_table_key_for_op(MgpuOpKind kind) noexcept;

}  // namespace poseidon::mgpu
