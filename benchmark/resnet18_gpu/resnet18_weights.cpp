#include "resnet18_weights.h"

#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace poseidon::benchmark::resnet18_gpu
{
namespace
{

constexpr int kStageCount = 4;
constexpr int kStageChannels[kStageCount] = {64, 128, 256, 512};
constexpr int kClassCount = 1000;
constexpr int kFinalChannels = 512;
constexpr std::size_t kImageValues = 224 * 224 * 3;

std::vector<double> read_values(
    const std::filesystem::path &path,
    std::size_t count)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open ResNet18 data: " + path.string());
    }
    std::vector<double> result(count);
    for (double &value : result)
    {
        if (!(stream >> value))
        {
            throw std::runtime_error("truncated ResNet18 data: " + path.string());
        }
    }
    return result;
}

std::vector<double> read_first(
    std::initializer_list<std::filesystem::path> paths,
    std::size_t count)
{
    for (const auto &path : paths)
    {
        if (std::filesystem::exists(path))
        {
            return read_values(path, count);
        }
    }
    throw std::runtime_error("no ResNet18 parameter candidate exists");
}

}  // namespace

std::filesystem::path default_trident_resnet18_root()
{
    if (const char *override_root = std::getenv("POSEIDON_TRIDENT_RESNET18_ROOT"))
    {
        return override_root;
    }
    return std::filesystem::path(POSEIDON_GPU_RESNET18_SOURCE_DIR) /
           "../../../Trident/resnet18";
}

ResNet18Weights load_resnet18_weights(const std::filesystem::path &resnet18_root)
{
    const auto root = resnet18_root / "pretrained_parameters/resnet18_imagenet";
    ResNet18Weights weights;
    constexpr int convolution_count = 1 + 4 * 2 * 2;
    weights.conv_weight.resize(convolution_count);
    weights.bn_bias.resize(convolution_count);
    weights.bn_running_mean.resize(convolution_count);
    weights.bn_running_var.resize(convolution_count);
    weights.bn_weight.resize(convolution_count);
    weights.downsample_weight.resize(3);
    weights.downsample_bn_bias.resize(3);
    weights.downsample_bn_running_mean.resize(3);
    weights.downsample_bn_running_var.resize(3);
    weights.downsample_bn_weight.resize(3);

    std::size_t conv = 0;
    std::size_t bn = 0;
    weights.conv_weight[conv++] = read_values(root / "conv1_weight.txt", 7 * 7 * 3 * 64);
    weights.bn_bias[bn] = read_values(root / "bn1_bias.txt", 64);
    weights.bn_running_mean[bn] = read_values(root / "bn1_running_mean.txt", 64);
    weights.bn_running_var[bn] = read_values(root / "bn1_running_var.txt", 64);
    weights.bn_weight[bn] = read_values(root / "bn1_weight.txt", 64);
    ++bn;

    for (int stage = 1; stage <= kStageCount; ++stage)
    {
        const int output_channels = kStageChannels[stage - 1];
        for (int block = 0; block < 2; ++block)
        {
            const int input_channels =
                stage > 1 && block == 0 ? kStageChannels[stage - 2] : output_channels;
            const std::string prefix =
                "layer" + std::to_string(stage) + "_" + std::to_string(block);
            weights.conv_weight[conv++] = read_values(
                root / (prefix + "_conv1_weight.txt"),
                3 * 3 * input_channels * output_channels);
            weights.conv_weight[conv++] = read_values(
                root / (prefix + "_conv2_weight.txt"),
                3 * 3 * output_channels * output_channels);
            for (int layer = 1; layer <= 2; ++layer)
            {
                const std::string bn_prefix =
                    prefix + "_bn" + std::to_string(layer);
                weights.bn_bias[bn] = read_values(
                    root / (bn_prefix + "_bias.txt"), output_channels);
                weights.bn_running_mean[bn] = read_values(
                    root / (bn_prefix + "_running_mean.txt"), output_channels);
                weights.bn_running_var[bn] = read_values(
                    root / (bn_prefix + "_running_var.txt"), output_channels);
                weights.bn_weight[bn] = read_values(
                    root / (bn_prefix + "_weight.txt"), output_channels);
                ++bn;
            }
        }
        if (stage > 1)
        {
            const auto index = static_cast<std::size_t>(stage - 2);
            const int input_channels = kStageChannels[stage - 2];
            const std::string prefix =
                "layer" + std::to_string(stage) + "_0_downsample";
            weights.downsample_weight[index] = read_values(
                root / (prefix + "_0_weight.txt"),
                input_channels * output_channels);
            weights.downsample_bn_bias[index] = read_values(
                root / (prefix + "_1_bias.txt"), output_channels);
            weights.downsample_bn_running_mean[index] = read_values(
                root / (prefix + "_1_running_mean.txt"), output_channels);
            weights.downsample_bn_running_var[index] = read_values(
                root / (prefix + "_1_running_var.txt"), output_channels);
            weights.downsample_bn_weight[index] = read_values(
                root / (prefix + "_1_weight.txt"), output_channels);
        }
    }
    if (conv != weights.conv_weight.size() || bn != weights.bn_bias.size())
    {
        throw std::runtime_error("ResNet18 parameter traversal count mismatch");
    }
    weights.linear_weight = read_first(
        {root / "fc_weight.txt", root / "linear_weight.txt"},
        kClassCount * kFinalChannels);
    weights.linear_bias = read_first(
        {root / "fc_bias.txt", root / "linear_bias.txt"}, kClassCount);
    return weights;
}

std::vector<double> load_imagenet_image_chw(
    std::size_t image_id,
    double boundary,
    const std::filesystem::path &resnet18_root)
{
    if (!(boundary > 0.0))
    {
        throw std::invalid_argument("ImageNet boundary must be positive");
    }
    std::ifstream stream(resnet18_root / "testFile/test_values.txt");
    if (!stream)
    {
        throw std::runtime_error("failed to open ResNet18 ImageNet test values");
    }
    double value = 0.0;
    for (std::size_t index = 0; index < image_id * kImageValues; ++index)
    {
        if (!(stream >> value))
        {
            throw std::runtime_error("failed to skip ResNet18 ImageNet image values");
        }
    }
    std::vector<double> result(kImageValues);
    for (double &slot : result)
    {
        if (!(stream >> slot))
        {
            throw std::runtime_error("failed to read ResNet18 ImageNet image values");
        }
        slot /= boundary;
    }
    return result;
}

int load_imagenet_label(
    std::size_t image_id,
    const std::filesystem::path &resnet18_root)
{
    std::ifstream stream(resnet18_root / "testFile/test_label.txt");
    if (!stream)
    {
        throw std::runtime_error("failed to open ResNet18 ImageNet labels");
    }
    int label = -1;
    for (std::size_t index = 0; index <= image_id; ++index)
    {
        if (!(stream >> label))
        {
            throw std::runtime_error("failed to read ResNet18 ImageNet label");
        }
    }
    return label;
}

}  // namespace poseidon::benchmark::resnet18_gpu
