#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace poseidon::benchmark::resnet20_gpu::core {

// Diagnostic-only balanced sparse ternary secret sampler.
template <class Engine>
std::vector<std::int8_t> sample_balanced_sparse_secret(
    std::size_t degree, std::size_t hamming_weight, Engine &engine) {
  if (degree == 0) {
    throw std::invalid_argument("secret degree must be nonzero");
  }
  if (hamming_weight == 0 || hamming_weight > degree || hamming_weight % 2 != 0) {
    throw std::invalid_argument(
        "balanced sparse secret requires an even nonzero hamming weight "
        "no greater than the degree");
  }

  std::vector<std::size_t> positions(degree);
  std::iota(positions.begin(), positions.end(), std::size_t{0});
  std::shuffle(positions.begin(), positions.end(), engine);

  std::vector<std::int8_t> secret(degree, 0);
  const std::size_t half = hamming_weight / 2;
  for (std::size_t i = 0; i < half; ++i) {
    secret[positions[i]] = 1;
  }
  for (std::size_t i = half; i < hamming_weight; ++i) {
    secret[positions[i]] = -1;
  }
  return secret;
}

}  // namespace poseidon::benchmark::resnet20_gpu::core
