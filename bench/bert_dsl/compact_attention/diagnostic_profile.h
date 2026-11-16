#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bert::attention::diagnostic {
// Exact ordered pool from ../generate_aligned.py::CERIUM_PRIMES. q52-p11
// takes the first52 and LAST11 (not the immediately following11) primes.
inline const std::vector<std::uint64_t> &prime_pool() {
    static const std::vector<std::uint64_t> primes{
        204865537,205651969,206307329,207880193,209059841,210370561,211025921,211812353,
        214171649,215482369,215744513,216137729,216924161,217317377,218628097,219676673,
        220594177,221249537,222035969,222167041,222953473,223215617,224002049,224133121,
        225574913,228065281,228458497,228720641,230424577,230686721,230817793,231473153,
        232390657,232652801,234356737,235798529,236584961,236716033,239337473,239861761,
        240648193,241827841,244842497,244973569,245235713,245760001,246415361,249561089,
        253100033,253493249,254279681,256376833,256770049,257949697,258605057,260571137,
        260702209,261488641,261881857,263323649,263454721,264634369,265420801,268042241};
    return primes;
}
inline std::vector<std::uint64_t> q52() {
    return {prime_pool().begin(), prime_pool().begin() + 52};
}
inline std::vector<std::uint64_t> p11() {
    return {prime_pool().end() - 11, prime_pool().end()};
}
inline double canonical_scale_log2(std::size_t level) {
    const auto q = q52();
    if (level >= q.size()) throw std::invalid_argument("canonical physical level absent from q52");
    std::vector<double> scales{56, 56};
    for (std::size_t l = 2; l <= level; ++l)
        scales.push_back((scales[l - 2] + std::log2(double(q[l])) + std::log2(double(q[l - 1]))) / 2);
    return scales[level];
}
} // namespace bert::attention::diagnostic
