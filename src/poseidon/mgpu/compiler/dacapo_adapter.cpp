#include "poseidon/mgpu/compiler/dacapo_adapter.h"

#include "poseidon/mgpu/ir/schedule_json.h"

#include <sstream>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

void add_diagnostic(DacapoAdapterResult &result, std::size_t offset, const std::string &message)
{
    result.diagnostics.push_back(DacapoAdapterDiagnostic{ offset, message });
}

}  // namespace

const char *to_string(DacapoInputFormat format) noexcept
{
    switch (format)
    {
    case DacapoInputFormat::Unknown:
        return "unknown";
    case DacapoInputFormat::EarthMlirText:
        return "earth_mlir_text";
    case DacapoInputFormat::Json:
        return "json";
    }
    return "unknown";
}

std::string DacapoAdapterResult::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }
        stream << "offset " << diagnostics[i].offset << ": " << diagnostics[i].message;
    }
    return stream.str();
}

DacapoAdapterResult translate_dacapo_schedule(
    std::string_view input, const DacapoAdapterOptions &options)
{
    DacapoAdapterResult result;

    if (input.empty())
    {
        add_diagnostic(result, 0, "Dacapo adapter input is empty");
        return result;
    }

    if (options.input_format == DacapoInputFormat::Json)
    {
        ScheduleJsonParseResult parsed = parse_schedule_json(input);
        if (parsed.ok())
        {
            result.schedule = std::move(parsed.schedule);
            return result;
        }

        for (const ScheduleJsonDiagnostic &diagnostic : parsed.diagnostics)
        {
            add_diagnostic(
                result, 0,
                "JSON schedule " + diagnostic.path + ": " + diagnostic.message);
        }
        return result;
    }

    std::ostringstream stream;
    stream << "Dacapo " << to_string(options.input_format)
           << " translation is not implemented; capture the optimized Dacapo output and "
              "map it into poseidon::mgpu::MgpuSchedule before enabling execution";
    add_diagnostic(result, 0, stream.str());
    return result;
}

}  // namespace poseidon::mgpu
