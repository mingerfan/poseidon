#include "resnet20_topology.h"

#include <array>
#include <stdexcept>

namespace poseidon::benchmark::resnet20_gpu
{

std::size_t ResNet20Topology::bootstrap_point_count() const noexcept
{
    return blocks.size() * 2;
}

void ResNet20Topology::validate() const
{
    constexpr std::array<int, 3> expected_blocks{3, 3, 3};
    if (blocks.size() != 9 || bootstrap_point_count() != 18)
    {
        throw std::invalid_argument(
            "ResNet20 topology must contain 9 basic blocks and 18 bootstraps");
    }
    std::array<int, 3> actual{};
    for (const auto &spec : blocks)
    {
        if (spec.stage < 1 || spec.stage > 3 || spec.block < 0 ||
            spec.input_channels <= 0 || spec.output_channels <= 0 ||
            (spec.stride != 1 && spec.stride != 2))
        {
            throw std::invalid_argument("invalid ResNet20 basic-block specification");
        }
        const bool expected_shortcut = spec.stage > 1 && spec.block == 0;
        if (spec.option_a_shortcut != expected_shortcut ||
            spec.stride != (expected_shortcut ? 2 : 1) ||
            spec.output_channels != (16 << (spec.stage - 1)) ||
            spec.input_channels !=
                (expected_shortcut ? spec.output_channels / 2 : spec.output_channels))
        {
            throw std::invalid_argument("invalid ResNet20 shortcut specification");
        }
        ++actual[static_cast<std::size_t>(spec.stage - 1)];
    }
    if (actual != expected_blocks)
    {
        throw std::invalid_argument("ResNet20 stage block counts must be 3/3/3");
    }
}

ResNet20Topology make_resnet20_topology()
{
    constexpr std::array<int, 3> widths{16, 32, 64};
    ResNet20Topology topology;
    int input_channels = 16;
    for (std::size_t stage_index = 0; stage_index < widths.size(); ++stage_index)
    {
        for (int block = 0; block < 3; ++block)
        {
            const bool shortcut = stage_index > 0 && block == 0;
            topology.blocks.push_back(
                BasicBlockSpec{static_cast<int>(stage_index + 1), block, input_channels,
                               widths[stage_index], shortcut ? 2 : 1, shortcut});
            input_channels = widths[stage_index];
        }
    }
    topology.validate();
    return topology;
}

std::string basic_block_name(const BasicBlockSpec &spec)
{
    return "layer" + std::to_string(spec.stage) + "." + std::to_string(spec.block);
}

}  // namespace poseidon::benchmark::resnet20_gpu
