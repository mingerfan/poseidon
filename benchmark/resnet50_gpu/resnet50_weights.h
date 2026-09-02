#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace poseidon::benchmark::resnet50_gpu
{

struct ResNet50Weights
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

std::filesystem::path default_trident_resnet50_root();
ResNet50Weights load_resnet50_weights(
    const std::filesystem::path &resnet50_root = default_trident_resnet50_root());
std::vector<double> load_imagenet_image_chw(
    std::size_t image_id,
    double boundary,
    const std::filesystem::path &resnet50_root = default_trident_resnet50_root());
int load_imagenet_label(
    std::size_t image_id,
    const std::filesystem::path &resnet50_root = default_trident_resnet50_root());

}  // namespace poseidon::benchmark::resnet50_gpu
