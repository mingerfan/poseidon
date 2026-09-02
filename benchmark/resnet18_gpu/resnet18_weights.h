#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace poseidon::benchmark::resnet18_gpu
{

struct ResNet18Weights
{
    std::vector<double> linear_weight;
    std::vector<double> linear_bias;
    std::vector<std::vector<double>> conv_weight;
    std::vector<std::vector<double>> bn_bias;
    std::vector<std::vector<double>> bn_running_mean;
    std::vector<std::vector<double>> bn_running_var;
    std::vector<std::vector<double>> bn_weight;
    std::vector<std::vector<double>> downsample_weight;
    std::vector<std::vector<double>> downsample_bn_bias;
    std::vector<std::vector<double>> downsample_bn_running_mean;
    std::vector<std::vector<double>> downsample_bn_running_var;
    std::vector<std::vector<double>> downsample_bn_weight;
};

std::filesystem::path default_trident_resnet18_root();
ResNet18Weights load_resnet18_weights(
    const std::filesystem::path &resnet18_root = default_trident_resnet18_root());
std::vector<double> load_imagenet_image_chw(
    std::size_t image_id,
    double boundary,
    const std::filesystem::path &resnet18_root = default_trident_resnet18_root());
int load_imagenet_label(
    std::size_t image_id,
    const std::filesystem::path &resnet18_root = default_trident_resnet18_root());

}  // namespace poseidon::benchmark::resnet18_gpu
