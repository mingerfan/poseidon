#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::benchmark::resnet18_gpu
{

struct BasicBlockSpec
{
    int stage = 0;
    int block = 0;
    int input_channels = 0;
    int output_channels = 0;
    int stride = 1;
    bool projection = false;
};

struct ResNet18Topology
{
    std::vector<BasicBlockSpec> blocks;

    std::size_t bootstrap_point_count() const noexcept;
    void validate() const;
};

ResNet18Topology make_resnet18_topology();
std::string basic_block_name(const BasicBlockSpec &spec);

}  // namespace poseidon::benchmark::resnet18_gpu
