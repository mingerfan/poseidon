#include "compact_layout.h"
#include <iostream>
#include <random>
#include <string>

using namespace bert::compact;
namespace {
void require(bool p, const char *why) { if (!p) throw std::runtime_error(why); }
void test_compression() {
    constexpr std::size_t n = 1024, limbs = 3, period = 32;
    std::vector<std::uint64_t> full(n * limbs);
    for (std::size_t l = 0; l < limbs; ++l)
        for (std::size_t j = 0; j < n; ++j)
            full[l * n + j] = l * 1000 + reverse_bits(j, 10) % period;
    auto c = compress_exact(full.data(), n, limbs, 512);
    require(c.period == period, "minimal period was not found");
    full[n + 37] ^= 1; // any one residue must reject a false compact period
    auto fallback = compress_exact(full.data(), n, limbs, 32);
    require(fallback.fallback(), "corrupted full residue did not force fallback");
    for (std::size_t i = 0; i < full.size(); ++i)
        require(fallback.values[i] == full[i], "fallback changed ordinary NTT order");
    std::fill(full.begin(), full.end(), 42);
    require(compress_exact(full.data(), n, limbs).period == 1, "constant period is not1");
    full[0] = std::uint64_t{1} << 40;
    bool threw = false;
    try { compress_exact(full.data(), n, limbs); } catch (const std::overflow_error &) { threw = true; }
    require(threw, "large residue silently truncated");
}
void test_dense() {
    // Tiny algebra test, independent of CUDA/Poseidon: multiple input/output
    // tiles, all diagonals, distinct token rows, negative giant groups.
    constexpr std::size_t tile = 16, baby = 4, nt = 3, ni = 2, no = 3;
    constexpr std::size_t iw = ni * tile, ow = no * tile, slots = nt * tile;
    std::mt19937 rng(20261115);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> w(iw * ow), bias(ow), x(nt * iw);
    for (auto &v : w) v = dist(rng);
    for (auto &v : bias) v = dist(rng);
    for (auto &v : x) v = dist(rng);
    double max_error = 0;
    for (std::size_t o = 0; o < no; ++o) {
        std::vector<double> result(slots);
        for (int g = -static_cast<int>(tile); g < static_cast<int>(tile); g += baby) {
            std::vector<double> subtotal(slots);
            for (int d = -static_cast<int>(tile) + 1; d < static_cast<int>(tile); ++d) {
                auto split = split_diagonal(d, baby);
                require(split.baby < baby && split.giant + static_cast<int>(split.baby) == d,
                        "negative diagonal split is invalid");
                if (split.giant != g) continue;
                for (std::size_t i = 0; i < ni; ++i) {
                    auto p = diagonal_period(w.data(), iw, ow, i, o, tile, d, g);
                    for (std::size_t j = 0; j < slots; ++j) {
                        const auto rotated = (j + split.baby) % slots;
                        subtotal[j] += x[(rotated / tile) * iw + i * tile + rotated % tile] * p[j % tile];
                    }
                }
            }
            for (std::size_t j = 0; j < slots; ++j)
                result[j] += subtotal[(j + slots + g) % slots];
        }
        for (std::size_t t = 0; t < nt; ++t)
            for (std::size_t f = 0; f < tile; ++f) {
                double reference = bias[o * tile + f];
                for (std::size_t k = 0; k < iw; ++k)
                    reference += double(w[(o * tile + f) * iw + k]) * x[t * iw + k];
                const double actual = result[t * tile + f] + bias[o * tile + f];
                max_error = std::max(max_error, std::abs(reference - actual));
            }
    }
    require(max_error < 1e-12, "rectangular token-preserving dense BSGS mismatch");
    std::cout << "{\"layout_check\":\"ok\",\"dense_max_abs\":" << max_error << ",\"exact_compression\":true}\n";
}
} // namespace
int main() {
    try { test_compression(); test_dense(); return 0; }
    catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
