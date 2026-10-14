#include "secure_sparse_secret.h"
#include "resnet20_weights.h"

#include <array>
#include <iostream>
#include <random>
#include <stdexcept>

int main()
{
    using poseidon::benchmark::resnet20_gpu::core::sample_balanced_sparse_secret;
    // A deterministic engine is appropriate only for this non-encrypting test.
    std::mt19937_64 engine(20260911);
    std::size_t cases = 0;
    for (std::size_t n : {2, 16, 4096, 65536})
        for (std::size_t h : {std::size_t{2}, std::min(n, std::size_t{192}), n})
            for (unsigned trial = 0; trial < 8; ++trial)
            {
                const auto secret = sample_balanced_sparse_secret(n, h, engine);
                if (secret.size() != n ||
                    std::count(secret.begin(), secret.end(), 1) != h / 2 ||
                    std::count(secret.begin(), secret.end(), -1) != h / 2 ||
                    std::count(secret.begin(), secret.end(), 0) != n - h)
                    throw std::runtime_error("sparse secret support/sign invariant failed");
                ++cases;
            }
    for (const auto bad : std::array<std::array<std::size_t, 2>, 4>{
             {{0, 0}, {16, 0}, {16, 3}, {16, 18}}})
    {
        bool rejected = false;
        try { (void)sample_balanced_sparse_secret(bad[0], bad[1], engine); }
        catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid/zero secret accepted");
        ++cases;
    }
    namespace model = poseidon::benchmark::resnet20_gpu;
    const auto root = model::default_trident_resnet20_root();
    const auto weights = model::load_resnet20_weights(root);
    const auto image = model::load_cifar10_image_chw(0, 40.0, root);
    if (weights.conv_weight.size() != 19 || weights.linear_weight.size() != 640 ||
        image.size() != 3072)
        throw std::runtime_error("local model asset shape changed");
    std::cout << "support: " << cases << " sampler cases and local model assets passed\n";
}
