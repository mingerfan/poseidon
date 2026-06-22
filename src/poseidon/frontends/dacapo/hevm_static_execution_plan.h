#pragma once

#include "poseidon/frontends/dacapo/dacapo_artifacts.h"
#include "poseidon/frontends/dacapo/hevm_io_binding.h"
#include "poseidon/frontends/dacapo/hevm_plaintext_encoding.h"
#include "poseidon/poseidon_context.h"

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::mgpu
{

struct HevmStaticExecutionPlan
{
    MgpuSchedule schedule;
    DacapoConstantTable constants;
    HevmIoBindingPlan io_plan;
    std::vector<HevmEncodedPlaintext> encoded_plaintexts;
    std::string debug_dump;
};

struct HevmStaticExecutionPlanDiagnostic
{
    std::string stage;
    std::string path;
    std::size_t location = 0;
    std::string message;
};

struct HevmStaticExecutionPlanResult
{
    HevmStaticExecutionPlan plan;
    std::vector<HevmStaticExecutionPlanDiagnostic> diagnostics;

    bool ok() const noexcept
    {
        return diagnostics.empty();
    }

    std::string format_diagnostics() const;
};

HevmStaticExecutionPlanResult prepare_hevm_static_execution_plan(
    const PoseidonContext &context, const DacapoHevmArtifactResult &artifacts);

HevmStaticExecutionPlanResult prepare_hevm_static_execution_plan_from_files(
    const PoseidonContext &context, const DacapoHevmArtifactPaths &paths,
    const StaticSchedulePipelineOptions &pipeline_options = {});

}  // namespace poseidon::mgpu
