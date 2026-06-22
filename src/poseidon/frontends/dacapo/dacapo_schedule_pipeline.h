#pragma once

#include "poseidon/frontends/dacapo/dacapo_adapter.h"
#include "poseidon/mgpu/compiler/static_schedule_pipeline.h"

#include <string_view>

namespace poseidon::mgpu
{

StaticSchedulePipelineResult prepare_dacapo_static_schedule(
    std::string_view input, const DacapoAdapterOptions &adapter_options,
    const StaticSchedulePipelineOptions &pipeline_options = {});

}  // namespace poseidon::mgpu
