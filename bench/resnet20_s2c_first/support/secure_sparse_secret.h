#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace poseidon::benchmark::resnet20_gpu::core
{
// Uniform support, exactly h/2 positive and h/2 negative entries. Production
// callers MUST use Poseidon's cryptographic RandomToStandardAdapter, not a
// simulation PRNG. Fixed sign counts match Lattice Estimator SparseTernary.
template <typename CryptographicEngine>
std::vector<std::int8_t> sample_balanced_sparse_secret(
    std::size_t n, std::size_t h, CryptographicEngine &engine)
{
    if (n == 0 || h == 0 || h > n || h % 2 != 0)
        throw std::invalid_argument("secret requires nonzero even h <= N");
    std::vector<std::size_t> positions(n);
    std::iota(positions.begin(), positions.end(), std::size_t{0});
    std::vector<std::int8_t> coefficients(n, 0);
    for (std::size_t i = 0; i < h; ++i)
    {
        std::uniform_int_distribution<std::size_t> choose(i, n - 1);
        std::swap(positions[i], positions[choose(engine)]);
        coefficients[positions[i]] = i < h / 2 ? 1 : -1;
    }
    return coefficients;
}
} // namespace poseidon::benchmark::resnet20_gpu::core
