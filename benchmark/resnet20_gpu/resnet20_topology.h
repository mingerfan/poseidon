#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace poseidon::benchmark::resnet20_gpu
{

struct BasicBlockSpec
{
    int stage = 0;
    int block = 0;
    int input_channels = 0;
    int output_channels = 0;
    int stride = 1;
    bool option_a_shortcut = false;
};

struct ResNet20Topology
{
    std::vector<BasicBlockSpec> blocks;

    std::size_t bootstrap_point_count() const noexcept;
    void validate() const;
};

ResNet20Topology make_resnet20_topology();
std::string basic_block_name(const BasicBlockSpec &spec);

}  // namespace poseidon::benchmark::resnet20_gpu
