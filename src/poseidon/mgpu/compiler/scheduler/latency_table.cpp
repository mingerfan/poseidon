#include "poseidon/mgpu/compiler/scheduler/latency_table.h"

#include "poseidon/util/json.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

using Json = nlohmann::json;

constexpr double kLatencyUnitSeconds = 1.0e-4;

void add_diagnostic(
    LatencyTableParseResult &result, std::string path, std::string message)
{
    result.diagnostics.push_back(
        LatencyTableParseDiagnostic{ std::move(path), std::move(message) });
}

void set_entry(LatencyTable &table, std::string key, std::vector<double> values)
{
    table.entries[std::move(key)] = std::move(values);
}

}  // namespace

const char *latency_table_key_for_op(MgpuOpKind kind) noexcept
{
    switch (kind)
    {
    case MgpuOpKind::Add:
    case MgpuOpKind::Sub:
        return "add_double";
    case MgpuOpKind::AddPlain:
    case MgpuOpKind::Negate:
        return "add_single";
    case MgpuOpKind::Multiply:
    case MgpuOpKind::Relinearize:
        return "mul_double";
    case MgpuOpKind::MultiplyPlain:
        return "mul_single";
    case MgpuOpKind::Rescale:
        return "rescale_single";
    case MgpuOpKind::Rotate:
        return "rotate_single";
    case MgpuOpKind::UploadPlain:
    case MgpuOpKind::UploadCipher:
    case MgpuOpKind::CopyPlain:
    case MgpuOpKind::CopyCipher:
    case MgpuOpKind::BootstrapFallback:
    case MgpuOpKind::Download:
        return "";
    }
    return "";
}

double LatencyTable::latency_for(MgpuOpKind kind, std::size_t index) const
{
    const char *key = latency_table_key_for_op(kind);
    if (key[0] == '\0')
    {
        return 0.0;
    }

    const auto iter = entries.find(key);
    if (iter == entries.end() || iter->second.empty())
    {
        return kLatencyUnitSeconds;
    }

    const std::size_t clamped = std::min(index, iter->second.size() - 1);
    return iter->second[clamped];
}

std::string LatencyTableParseResult::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << diagnostics[i].path << ": " << diagnostics[i].message;
    }
    return stream.str();
}

LatencyTable make_default_latency_table()
{
    LatencyTable table;
    set_entry(table, "add_single", { 2.0 * kLatencyUnitSeconds });
    set_entry(table, "add_double", { 3.0 * kLatencyUnitSeconds });
    set_entry(table, "mul_single", { 3.0 * kLatencyUnitSeconds });
    set_entry(table, "mul_double", { 697.0 * kLatencyUnitSeconds });
    set_entry(table, "rescale_single", { 15.0 * kLatencyUnitSeconds });
    set_entry(table, "rotate_single", { 685.0 * kLatencyUnitSeconds });
    return table;
}

LatencyTableParseResult parse_latency_table_json(std::string_view text)
{
    LatencyTableParseResult result;

    Json root;
    try
    {
        root = Json::parse(text.begin(), text.end());
    }
    catch (const std::exception &ex)
    {
        add_diagnostic(result, "/", ex.what());
        return result;
    }

    const auto table_iter = root.find("latencyTable");
    if (table_iter == root.end() || !table_iter->is_object())
    {
        add_diagnostic(result, "/latencyTable", "expected an object");
        return result;
    }

    for (auto iter = table_iter->begin(); iter != table_iter->end(); ++iter)
    {
        const std::string path = "/latencyTable/" + iter.key();
        if (!iter.value().is_array())
        {
            add_diagnostic(result, path, "expected an array");
            continue;
        }

        std::vector<double> values;
        values.reserve(iter.value().size());
        for (std::size_t i = 0; i < iter.value().size(); ++i)
        {
            const Json &item = iter.value()[i];
            if (!item.is_number())
            {
                add_diagnostic(
                    result, path + "/" + std::to_string(i), "expected a number");
                continue;
            }
            values.push_back(item.get<double>() * kLatencyUnitSeconds);
        }
        result.table.entries.emplace(iter.key(), std::move(values));
    }

    if (!result.ok())
    {
        result.table.entries.clear();
    }
    return result;
}

}  // namespace poseidon::mgpu
