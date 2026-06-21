#pragma once

#include "poseidon/mgpu/comm/routed_object_copy.h"

#include <cstddef>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace poseidon::mgpu
{

class PlannedMaterializedGpuComm final : public GpuComm
{
public:
    PlannedMaterializedGpuComm(
        const MgpuCommunicationPlan &plan,
        GpuObjectCopyMaterializer &materializer,
        RoutedGpuObjectCopyBackend &backend);

    std::shared_ptr<void> copy(const GpuCommCopyRequest &request) override;

private:
    std::vector<MgpuCopyRoute> routes_;
    std::map<std::pair<ValueId, ValueId>, std::size_t> route_indices_;
    GpuObjectCopyMaterializer &materializer_;
    RoutedGpuObjectCopyBackend &backend_;
};

}  // namespace poseidon::mgpu
