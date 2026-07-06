#include "poseidon/frontends/dacapo/dacapo_artifacts.h"

#include <fstream>
#include <sstream>
#include <utility>

namespace poseidon::mgpu
{
namespace
{

void add_diagnostic(
    DacapoHevmArtifactResult &result, std::string stage, std::string path,
    std::size_t location, std::string message)
{
    result.diagnostics.push_back(DacapoHevmArtifactDiagnostic{
        std::move(stage),
        std::move(path),
        location,
        std::move(message),
    });
}

void add_diagnostic(
    StaticSchedulePipelineResult &result, std::string stage, std::size_t op_index,
    std::string message)
{
    result.diagnostics.push_back(
        StaticSchedulePipelineDiagnostic{ std::move(stage), op_index, std::move(message) });
}

std::string read_binary_file(
    DacapoHevmArtifactResult &result, const std::string &path, const char *stage)
{
    if (path.empty())
    {
        add_diagnostic(result, stage, path, 0, "Dacapo artifact path must not be empty");
        return {};
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        add_diagnostic(result, stage, path, 0, "failed to open Dacapo artifact file");
        return {};
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (stream.bad())
    {
        add_diagnostic(result, stage, path, 0, "failed to read Dacapo artifact file");
        return {};
    }

    return buffer.str();
}

void add_pipeline_diagnostics(
    DacapoHevmArtifactResult &result, const std::string &path,
    const StaticSchedulePipelineResult &pipeline)
{
    for (const StaticSchedulePipelineDiagnostic &diagnostic : pipeline.diagnostics)
    {
        add_diagnostic(
            result, diagnostic.stage, path, diagnostic.op_index, diagnostic.message);
    }
}

void add_constant_diagnostics(
    DacapoHevmArtifactResult &result, const std::string &path,
    const DacapoConstantParseResult &constants)
{
    for (const DacapoConstantDiagnostic &diagnostic : constants.diagnostics)
    {
        add_diagnostic(result, "constants", path, diagnostic.offset, diagnostic.message);
    }
}

}  // namespace

StaticSchedulePipelineResult prepare_dacapo_static_schedule(
    std::string_view input, const DacapoAdapterOptions &adapter_options,
    const StaticSchedulePipelineOptions &pipeline_options)
{
    StaticSchedulePipelineResult result;

    const DacapoAdapterResult adapter_result =
        translate_dacapo_schedule(input, adapter_options);
    for (const DacapoAdapterDiagnostic &diagnostic : adapter_result.diagnostics)
    {
        add_diagnostic(
            result, "dacapo_adapter", diagnostic.offset, diagnostic.message);
    }
    if (!adapter_result.ok())
    {
        return result;
    }

    return prepare_static_schedule(adapter_result.schedule, pipeline_options);
}

std::string DacapoHevmArtifactResult::format_diagnostics() const
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i > 0)
        {
            stream << '\n';
        }

        const DacapoHevmArtifactDiagnostic &diagnostic = diagnostics[i];
        stream << diagnostic.stage;
        if (!diagnostic.path.empty())
        {
            stream << " " << diagnostic.path;
        }
        stream << " @" << diagnostic.location << ": " << diagnostic.message;
    }
    return stream.str();
}

DacapoHevmArtifactResult prepare_dacapo_hevm_artifacts_from_files(
    const DacapoHevmArtifactPaths &paths,
    const StaticSchedulePipelineOptions &pipeline_options)
{
    DacapoHevmArtifactResult result;

    const std::string hevm_input = read_binary_file(result, paths.hevm_path, "read_hevm");
    const std::string constants_input =
        read_binary_file(result, paths.constants_path, "read_constants");
    if (!result.ok())
    {
        return result;
    }

    const StaticSchedulePipelineResult pipeline = prepare_dacapo_static_schedule(
        hevm_input, DacapoAdapterOptions{ DacapoInputFormat::HevmBinary },
        pipeline_options);
    add_pipeline_diagnostics(result, paths.hevm_path, pipeline);

    const DacapoConstantParseResult constants =
        parse_dacapo_constant_file(constants_input);
    add_constant_diagnostics(result, paths.constants_path, constants);

    if (!result.ok())
    {
        return result;
    }

    result.schedule = pipeline.schedule;
    result.debug_dump = pipeline.debug_dump;
    result.constants = constants.table;
    return result;
}

}  // namespace poseidon::mgpu
