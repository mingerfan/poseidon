#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::benchmark::resnet50_gpu
{

struct BottleneckSpec
{
    int stage = 0;
    int block = 0;
    int input_channels = 0;
    int bottleneck_channels = 0;
    int output_channels = 0;
    int stride = 1;
    bool projection = false;
};

struct ResNet50Topology
{
    std::vector<BottleneckSpec> blocks;

    std::size_t bootstrap_point_count() const noexcept;
    void validate() const;
};

ResNet50Topology make_resnet50_topology();
std::string bottleneck_name(const BottleneckSpec &spec);

}  // namespace poseidon::benchmark::resnet50_gpu

