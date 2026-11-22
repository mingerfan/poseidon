// CPU-only real Poseidon encoding/compression check. No CUDA headers, GPU
// allocation, key generation, encryption, or full dense encoded pack.
#include "compact_layout.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/basics/modulus.h"
#include <chrono>
#include <iostream>
#include <random>
#include <string>

using namespace bert::compact;
int main(int argc, char **argv) {
    try {
        std::size_t degree = 1024, q_count = 6, p_count = 2, terms = 8;
        double scale_log2 = 30;
        int q_bits = 30;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("option requires a value");
            const std::string value = argv[++i];
            if (arg == "--degree") degree = std::stoull(value);
            else if (arg == "--q-count") q_count = std::stoull(value);
            else if (arg == "--p-count") p_count = std::stoull(value);
            else if (arg == "--terms") terms = std::stoull(value);
            else if (arg == "--scale-log2") scale_log2 = std::stod(value);
            else if (arg == "--q-bits") q_bits = std::stoi(value);
            else throw std::invalid_argument("unknown option " + arg);
        }
        const auto start = std::chrono::steady_clock::now();
        constexpr std::size_t tile = 256;
        if (degree < 512 || !power_of_two(degree) || !q_count || !p_count || !terms)
            throw std::invalid_argument("invalid CPU check parameters");
        std::vector<int> bits(q_count + p_count, q_bits);
        bits.front() = 31;
        for (std::size_t i = q_count; i < bits.size(); ++i) bits[i] = 31;
        auto primes = poseidon::CoeffModulus::Create(degree, bits);
        std::vector<poseidon::Modulus> q(primes.begin(), primes.begin() + q_count),
            p(primes.begin() + q_count, primes.end());
        const auto logn = log2_exact(degree);
        poseidon::ParametersLiteral literal(CKKS, logn, logn - 1, scale_log2, 0, 1,
            poseidon::Modulus(0), q, p, poseidon::sec_level_type::none);
        poseidon::PoseidonContext context(literal);
        poseidon::CKKSEncoder encoder(context);
        std::mt19937 rng(20261115);
        std::normal_distribution<float> dist(0, 0.03f);
        std::vector<float> w(tile * tile);
        for (auto &v : w) v = dist(rng);
        std::vector<double> slots(degree / 2);
        poseidon::Plaintext pt;
        std::uint64_t compact_bytes = 0, full_bytes = 0, fallbacks = 0, max_period = 0;
        double decode_error = 0;
        for (std::size_t t = 0; t < terms; ++t) {
            const int d = t == 0 ? -255 : t == 1 ? 255 : t == 2 ? 0 :
                static_cast<int>((t * 73) % 511) - 255;
            const auto split = split_diagonal(d, 16);
            const auto period = diagonal_period(w.data(), tile, tile, 0, 0, tile, d, split.giant);
            for (std::size_t i = 0; i < slots.size(); ++i) slots[i] = period[i % tile];
            encoder.encode(slots, std::exp2(scale_log2), pt);
            auto compact = compress_exact(pt.data(), degree, q_count);
            compact_bytes += compact.values.size() * sizeof(std::uint32_t);
            full_bytes += degree * q_count * sizeof(std::uint32_t);
            fallbacks += compact.fallback();
            max_period = std::max<std::uint64_t>(max_period, compact.period);
            std::vector<double> decoded;
            encoder.decode(pt, decoded);
            for (std::size_t i = 0; i < slots.size(); ++i)
                decode_error = std::max(decode_error, std::abs(decoded[i] - slots[i]));
        }
        std::cout.precision(16);
        std::cout << "{\"cpu_only\":true,\"degree\":" << degree << ",\"q_count\":" << q_count
            << ",\"p_count\":" << p_count << ",\"plaintext_scale_log2\":" << scale_log2
            << ",\"terms\":" << terms << ",\"compact_bytes\":" << compact_bytes
            << ",\"full_equivalent_bytes\":" << full_bytes << ",\"fallbacks\":" << fallbacks
            << ",\"max_period\":" << max_period << ",\"decode_max_abs\":" << decode_error
            << ",\"seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
            << "}\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
