#include "resnet50_weights.h"

#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace poseidon::benchmark::resnet50_gpu
{
namespace
{

constexpr int kStageCount = 4;
constexpr int kBlocks[kStageCount] = { 3, 4, 6, 3 };
constexpr int kPlanes[kStageCount] = { 64, 128, 256, 512 };
constexpr int kStageChannels[kStageCount] = { 256, 512, 1024, 2048 };
constexpr int kClassCount = 1000;
constexpr int kFinalChannels = 2048;
constexpr std::size_t kImageValues = 224 * 224 * 3;

std::vector<double> read_values(
    const std::filesystem::path &path,
    std::size_t count)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("failed to open ResNet50 data: " + path.string());
    }
    std::vector<double> result(count);
    for (double &value : result)
    {
        if (!(stream >> value))
        {
            throw std::runtime_error("truncated ResNet50 data: " + path.string());
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
    throw std::runtime_error("no ResNet50 parameter candidate exists");
}

int stage_input_channels(int stage, int block)
{
    if (stage == 1 && block == 0)
    {
        return 64;
    }
    return block == 0 ? kStageChannels[stage - 2] : kStageChannels[stage - 1];
}

}  // namespace

std::filesystem::path default_trident_resnet50_root()
{
    if (const char *override_root = std::getenv("POSEIDON_TRIDENT_RESNET50_ROOT"))
    {
        return override_root;
    }
    return std::filesystem::path(POSEIDON_GPU_RESNET50_SOURCE_DIR) /
           "../../../Trident/resnet50";
}

ResNet50Weights load_resnet50_weights(const std::filesystem::path &resnet50_root)
{
    const auto root = resnet50_root / "pretrained_parameters/resnet50_imagenet";
    ResNet50Weights weights;
    constexpr int convolution_count = 1 + 3 * (3 + 4 + 6 + 3);
    weights.conv_weight.resize(convolution_count);
    weights.bn_bias.resize(convolution_count);
    weights.bn_running_mean.resize(convolution_count);
    weights.bn_running_var.resize(convolution_count);
    weights.bn_weight.resize(convolution_count);
    weights.downsample_weight.resize(kStageCount);
    weights.downsample_bn_bias.resize(kStageCount);
    weights.downsample_bn_running_mean.resize(kStageCount);
    weights.downsample_bn_running_var.resize(kStageCount);
    weights.downsample_bn_weight.resize(kStageCount);

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
        const int planes = kPlanes[stage - 1];
        const int output_channels = kStageChannels[stage - 1];
        for (int block = 0; block < kBlocks[stage - 1]; ++block)
        {
            const int input_channels = stage_input_channels(stage, block);
            const std::string prefix =
                "layer" + std::to_string(stage) + "_" + std::to_string(block);
            weights.conv_weight[conv++] = read_values(
                root / (prefix + "_conv1_weight.txt"), input_channels * planes);
            weights.conv_weight[conv++] = read_values(
                root / (prefix + "_conv2_weight.txt"), 3 * 3 * planes * planes);
            weights.conv_weight[conv++] = read_values(
                root / (prefix + "_conv3_weight.txt"), planes * output_channels);
            for (int layer = 1; layer <= 3; ++layer)
            {
                const int channels = layer == 3 ? output_channels : planes;
                const std::string bn_prefix =
                    prefix + "_bn" + std::to_string(layer);
                weights.bn_bias[bn] = read_values(
                    root / (bn_prefix + "_bias.txt"), channels);
                weights.bn_running_mean[bn] = read_values(
                    root / (bn_prefix + "_running_mean.txt"), channels);
                weights.bn_running_var[bn] = read_values(
                    root / (bn_prefix + "_running_var.txt"), channels);
                weights.bn_weight[bn] = read_values(
                    root / (bn_prefix + "_weight.txt"), channels);
                ++bn;
            }
        }
        const std::string prefix =
            "layer" + std::to_string(stage) + "_0_downsample";
        const int input_channels = stage == 1 ? 64 : kStageChannels[stage - 2];
        const std::size_t index = static_cast<std::size_t>(stage - 1);
        weights.downsample_weight[index] = read_values(
            root / (prefix + "_0_weight.txt"), input_channels * output_channels);
        weights.downsample_bn_bias[index] = read_values(
            root / (prefix + "_1_bias.txt"), output_channels);
        weights.downsample_bn_running_mean[index] = read_values(
            root / (prefix + "_1_running_mean.txt"), output_channels);
        weights.downsample_bn_running_var[index] = read_values(
            root / (prefix + "_1_running_var.txt"), output_channels);
        weights.downsample_bn_weight[index] = read_values(
            root / (prefix + "_1_weight.txt"), output_channels);
    }
    weights.linear_weight = read_first(
        { root / "fc_weight.txt", root / "linear_weight.txt" },
        kClassCount * kFinalChannels);
    weights.linear_bias = read_first(
        { root / "fc_bias.txt", root / "linear_bias.txt" }, kClassCount);
    return weights;
}

std::vector<double> load_imagenet_image_chw(
    std::size_t image_id,
    double boundary,
    const std::filesystem::path &resnet50_root)
{
    if (!(boundary > 0.0))
    {
        throw std::invalid_argument("ImageNet boundary must be positive");
    }
    std::ifstream stream(resnet50_root / "testFile/test_values.txt");
    if (!stream)
    {
        throw std::runtime_error("failed to open ImageNet test values");
    }
    double value = 0.0;
    for (std::size_t index = 0; index < image_id * kImageValues; ++index)
    {
        if (!(stream >> value))
        {
            throw std::runtime_error("failed to skip ImageNet image values");
        }
    }
    std::vector<double> result(kImageValues);
    for (double &slot : result)
    {
        if (!(stream >> slot))
        {
            throw std::runtime_error("failed to read ImageNet image values");
        }
        slot /= boundary;
    }
    return result;
}

int load_imagenet_label(
    std::size_t image_id,
    const std::filesystem::path &resnet50_root)
{
    std::ifstream stream(resnet50_root / "testFile/test_label.txt");
    if (!stream)
    {
        throw std::runtime_error("failed to open ImageNet labels");
    }
    int label = -1;
    for (std::size_t index = 0; index <= image_id; ++index)
    {
        if (!(stream >> label))
        {
            throw std::runtime_error("failed to read ImageNet label");
        }
    }
    return label;
}

}  // namespace poseidon::benchmark::resnet50_gpu
