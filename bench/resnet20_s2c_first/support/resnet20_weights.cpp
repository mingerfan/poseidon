#include "resnet20_weights.h"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace poseidon::benchmark::resnet20_gpu
{
namespace
{

constexpr int kStageChannels[3] = {16, 32, 64};
constexpr std::size_t kImageValues = 32 * 32 * 3;

std::vector<double> read_values(const std::filesystem::path &path, std::size_t count)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open ResNet20 data: " + path.string());
    }
    std::vector<double> result(count);
    for (double &value : result)
    {
        if (!(stream >> value))
        {
            throw std::runtime_error("truncated ResNet20 data: " + path.string());
        }
    }
    return result;
}

}  // namespace

std::filesystem::path default_trident_resnet20_root()
{
    if (const char *override_root = std::getenv("POSEIDON_TRIDENT_RESNET20_ROOT"))
    {
        return override_root;
    }
    return std::filesystem::path(POSEIDON_GPU_RESNET20_SOURCE_DIR) /
           "data/resnet20";
}

ResNet20Weights load_resnet20_weights(const std::filesystem::path &resnet20_root)
{
    const auto root = resnet20_root / "pretrained_parameters/resnet20_new";
    ResNet20Weights weights;
    constexpr std::size_t convolution_count = 19;
    weights.conv_weight.resize(convolution_count);
    weights.bn_bias.resize(convolution_count);
    weights.bn_running_mean.resize(convolution_count);
    weights.bn_running_var.resize(convolution_count);
    weights.bn_weight.resize(convolution_count);

    std::size_t conv = 0;
    std::size_t bn = 0;
    weights.conv_weight[conv++] =
        read_values(root / "conv1_weight.txt", 3 * 3 * 3 * 16);
    weights.bn_bias[bn] = read_values(root / "bn1_bias.txt", 16);
    weights.bn_running_mean[bn] = read_values(root / "bn1_running_mean.txt", 16);
    weights.bn_running_var[bn] = read_values(root / "bn1_running_var.txt", 16);
    weights.bn_weight[bn] = read_values(root / "bn1_weight.txt", 16);
    ++bn;

    for (int stage = 1; stage <= 3; ++stage)
    {
        const int output_channels = kStageChannels[stage - 1];
        for (int block = 0; block < 3; ++block)
        {
            const int input_channels =
                stage > 1 && block == 0 ? kStageChannels[stage - 2] : output_channels;
            const std::string prefix =
                "layer" + std::to_string(stage) + "_" + std::to_string(block);
            weights.conv_weight[conv++] =
                read_values(root / (prefix + "_conv1_weight.txt"),
                            3 * 3 * input_channels * output_channels);
            weights.conv_weight[conv++] =
                read_values(root / (prefix + "_conv2_weight.txt"),
                            3 * 3 * output_channels * output_channels);
            for (int layer = 1; layer <= 2; ++layer)
            {
                const auto bn_prefix = prefix + "_bn" + std::to_string(layer);
                weights.bn_bias[bn] =
                    read_values(root / (bn_prefix + "_bias.txt"), output_channels);
                weights.bn_running_mean[bn] = read_values(
                    root / (bn_prefix + "_running_mean.txt"), output_channels);
                weights.bn_running_var[bn] = read_values(
                    root / (bn_prefix + "_running_var.txt"), output_channels);
                weights.bn_weight[bn] =
                    read_values(root / (bn_prefix + "_weight.txt"), output_channels);
                ++bn;
            }
        }
    }
    if (conv != convolution_count || bn != convolution_count)
    {
        throw std::runtime_error("ResNet20 parameter traversal count mismatch");
    }
    weights.linear_weight = read_values(root / "linear_weight.txt", 10 * 64);
    weights.linear_bias = read_values(root / "linear_bias.txt", 10);
    return weights;
}

std::vector<double> load_cifar10_image_chw(std::size_t image_id, double boundary,
                                           const std::filesystem::path &resnet20_root)
{
    if (!(boundary > 0.0))
    {
        throw std::invalid_argument("CIFAR-10 boundary must be positive");
    }
    std::ifstream stream(resnet20_root / "testFile/test_values.txt");
    if (!stream)
    {
        throw std::runtime_error("failed to open ResNet20 CIFAR-10 test values");
    }
    double value = 0.0;
    for (std::size_t index = 0; index < image_id * kImageValues; ++index)
    {
        if (!(stream >> value))
        {
            throw std::runtime_error("failed to skip ResNet20 CIFAR-10 image values");
        }
    }
    std::vector<double> result(kImageValues);
    for (double &slot : result)
    {
        if (!(stream >> slot))
        {
            throw std::runtime_error("failed to read ResNet20 CIFAR-10 image values");
        }
        slot /= boundary;
    }
    return result;
}

int load_cifar10_label(std::size_t image_id, const std::filesystem::path &resnet20_root)
{
    std::ifstream stream(resnet20_root / "testFile/test_label.txt");
    if (!stream)
    {
        throw std::runtime_error("failed to open ResNet20 CIFAR-10 labels");
    }
    int label = -1;
    for (std::size_t index = 0; index <= image_id; ++index)
    {
        if (!(stream >> label))
        {
            throw std::runtime_error("failed to read ResNet20 CIFAR-10 label");
        }
    }
    return label;
}

}  // namespace poseidon::benchmark::resnet20_gpu
