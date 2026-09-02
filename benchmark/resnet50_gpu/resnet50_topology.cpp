#include "resnet50_topology.h"

#include <array>
#include <stdexcept>

namespace poseidon::benchmark::resnet50_gpu
{

std::size_t ResNet50Topology::bootstrap_point_count() const noexcept
{
    return blocks.size() * 3;
}

void ResNet50Topology::validate() const
{
    constexpr std::array<int, 4> expected_blocks{3, 4, 6, 3};
    if (blocks.size() != 16 || bootstrap_point_count() != 48)
    {
        throw std::invalid_argument("ResNet50 topology must contain 16 blocks and 48 activations");
    }
    std::array<int, 4> actual{};
    for (const auto &spec : blocks)
    {
        if (spec.stage < 1 || spec.stage > 4 || spec.block < 0 ||
            spec.input_channels <= 0 || spec.bottleneck_channels <= 0 ||
            spec.output_channels != spec.bottleneck_channels * 4)
        {
            throw std::invalid_argument("invalid ResNet50 bottleneck specification");
        }
        ++actual[static_cast<std::size_t>(spec.stage - 1)];
    }
    if (actual != expected_blocks)
    {
        throw std::invalid_argument("ResNet50 stage block counts must be 3/4/6/3");
    }
}

ResNet50Topology make_resnet50_topology()
{
    constexpr std::array<int, 4> block_counts{3, 4, 6, 3};
    constexpr std::array<int, 4> widths{64, 128, 256, 512};
    ResNet50Topology topology;
    int input_channels = 64;
    for (std::size_t stage_index = 0; stage_index < block_counts.size(); ++stage_index)
    {
        for (int block = 0; block < block_counts[stage_index]; ++block)
        {
            const int output_channels = widths[stage_index] * 4;
            const bool projection = block == 0;
            topology.blocks.push_back(BottleneckSpec{
                static_cast<int>(stage_index + 1),
                block,
                input_channels,
                widths[stage_index],
                output_channels,
                stage_index != 0 && block == 0 ? 2 : 1,
                projection});
            input_channels = output_channels;
        }
    }
    topology.validate();
    return topology;
}

std::string bottleneck_name(const BottleneckSpec &spec)
{
    return "layer" + std::to_string(spec.stage) + "." +
        std::to_string(spec.block);
}

}  // namespace poseidon::benchmark::resnet50_gpu

