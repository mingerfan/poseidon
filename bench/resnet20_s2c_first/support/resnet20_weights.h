#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace poseidon::benchmark::resnet20_gpu
{

struct ResNet20Weights
{
    std::vector<double> linear_weight;
    std::vector<double> linear_bias;
    std::vector<std::vector<double>> conv_weight;
    std::vector<std::vector<double>> bn_bias;
    std::vector<std::vector<double>> bn_running_mean;
    std::vector<std::vector<double>> bn_running_var;
    std::vector<std::vector<double>> bn_weight;
};

std::filesystem::path default_trident_resnet20_root();
ResNet20Weights load_resnet20_weights(
    const std::filesystem::path &resnet20_root = default_trident_resnet20_root());
std::vector<double> load_cifar10_image_chw(
    std::size_t image_id, double boundary,
    const std::filesystem::path &resnet20_root = default_trident_resnet20_root());
int load_cifar10_label(
    std::size_t image_id,
    const std::filesystem::path &resnet20_root = default_trident_resnet20_root());

}  // namespace poseidon::benchmark::resnet20_gpu
