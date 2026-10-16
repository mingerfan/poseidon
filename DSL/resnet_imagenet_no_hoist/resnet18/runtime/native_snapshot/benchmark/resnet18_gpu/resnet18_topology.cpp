#include "resnet18_topology.h"

#include <array>
#include <stdexcept>

namespace poseidon::benchmark::resnet18_gpu
{

std::size_t ResNet18Topology::bootstrap_point_count() const noexcept
{
    return blocks.size() * 2;
}

void ResNet18Topology::validate() const
{
    constexpr std::array<int, 4> expected_blocks{2, 2, 2, 2};
    if (blocks.size() != 8 || bootstrap_point_count() != 16)
    {
        throw std::invalid_argument(
            "ResNet18 topology must contain 8 basic blocks and 16 bootstraps");
    }
    std::array<int, 4> actual{};
    for (const auto &spec : blocks)
    {
        if (spec.stage < 1 || spec.stage > 4 || spec.block < 0 ||
            spec.input_channels <= 0 || spec.output_channels <= 0 ||
            (spec.stride != 1 && spec.stride != 2))
        {
            throw std::invalid_argument("invalid ResNet18 basic-block specification");
        }
        const bool expected_projection = spec.stage > 1 && spec.block == 0;
        if (spec.projection != expected_projection ||
            spec.stride != (expected_projection ? 2 : 1))
        {
            throw std::invalid_argument("invalid ResNet18 shortcut specification");
        }
        ++actual[static_cast<std::size_t>(spec.stage - 1)];
    }
    if (actual != expected_blocks)
    {
        throw std::invalid_argument("ResNet18 stage block counts must be 2/2/2/2");
    }
}

ResNet18Topology make_resnet18_topology()
{
    constexpr std::array<int, 4> widths{64, 128, 256, 512};
    ResNet18Topology topology;
    int input_channels = 64;
    for (std::size_t stage_index = 0; stage_index < widths.size(); ++stage_index)
    {
        for (int block = 0; block < 2; ++block)
        {
            const bool projection = stage_index > 0 && block == 0;
            topology.blocks.push_back(BasicBlockSpec{
                static_cast<int>(stage_index + 1),
                block,
                input_channels,
                widths[stage_index],
                projection ? 2 : 1,
                projection});
            input_channels = widths[stage_index];
        }
    }
    topology.validate();
    return topology;
}

std::string basic_block_name(const BasicBlockSpec &spec)
{
    return "layer" + std::to_string(spec.stage) + "." +
        std::to_string(spec.block);
}

}  // namespace poseidon::benchmark::resnet18_gpu
