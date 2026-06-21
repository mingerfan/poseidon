#pragma once

#include "poseidon/mgpu/compiler/dacapo_constants.h"
#include "poseidon/mgpu/compiler/static_schedule_pipeline.h"

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct DacapoHevmArtifactPaths
{
    std::string hevm_path;
    std::string constants_path;
};

struct DacapoHevmArtifactDiagnostic
{
    std::string stage;
    std::string path;
    std::size_t location = 0;
    std::string message;
};

struct DacapoHevmArtifactResult
{
    MgpuSchedule schedule;
    DacapoConstantTable constants;
    std::string debug_dump;
    std::vector<DacapoHevmArtifactDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

DacapoHevmArtifactResult prepare_dacapo_hevm_artifacts_from_files(
    const DacapoHevmArtifactPaths &paths,
    const StaticSchedulePipelineOptions &pipeline_options = {});

}  // namespace poseidon::mgpu
