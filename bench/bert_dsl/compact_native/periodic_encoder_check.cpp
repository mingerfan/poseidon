// CPU-only optional-encoder oracle. This never changes CompactLinearEngine.
#include "periodic_encoder_experiment.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
    try {
        using namespace bert::compact;
        using namespace poseidon;
        using Clock = std::chrono::steady_clock;
        std::size_t n = 1024, nq = 6, terms = 8, iw = 256, ow = 256;
        int qbits = 28, q0bits = 31;
        double scale_log2 = 56;
        std::string weights_path, q_prime_list;
        bool synthetic = false, require_bitwise = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--synthetic-validation") { synthetic = true; continue; }
            if (arg == "--require-native-bitwise") { require_bitwise = true; continue; }
            if (i + 1 >= argc) throw std::invalid_argument("option needs a value");
            const std::string value = argv[++i];
            if (arg == "--degree") n = std::stoull(value);
            else if (arg == "--q-count") nq = std::stoull(value);
            else if (arg == "--q-bits") qbits = std::stoi(value);
            else if (arg == "--q0-bits") q0bits = std::stoi(value);
            else if (arg == "--terms") terms = std::stoull(value);
            else if (arg == "--input-width") iw = std::stoull(value);
            else if (arg == "--output-width") ow = std::stoull(value);
            else if (arg == "--scale-log2") scale_log2 = std::stod(value);
            else if (arg == "--weights") weights_path = value;
            else if (arg == "--q-primes") q_prime_list = value;
            else throw std::invalid_argument("unknown option " + arg);
        }
        if (!power_of_two(n) || n < 1024 || !nq || !terms || !iw || !ow || iw%256 || ow%256 ||
            (!synthetic && weights_path.empty()) || !std::isfinite(scale_log2))
            throw std::invalid_argument("invalid oracle options (need --weights or explicit --synthetic-validation)");
        std::vector<int> bits(nq, qbits); bits.front() = q0bits;
        std::vector<Modulus> q;
        if (q_prime_list.empty()) q = CoeffModulus::Create(n, bits);
        else {
            std::istringstream stream(q_prime_list);
            std::string item;
            while (std::getline(stream, item, ',')) q.emplace_back(std::stoull(item));
            if (q.size() != nq) throw std::invalid_argument("--q-primes count differs from --q-count");
            for (const auto &prime : q)
                if ((prime.value()-1)%(2*n)) throw std::invalid_argument("Q does not support target full NTT");
        }
        ParametersLiteral params(CKKS, log2_exact(n), log2_exact(n)-1, 28, 0, 0,
            Modulus(0), q, {}, sec_level_type::none);
        PoseidonContext context(params);
        CKKSEncoder native(context);
        const auto target = context.crt_context()->first_parms_id();
        PeriodicEncoderExperiment fast(context, target);
        fast.check_monomial_mapping();
        std::vector<float> weights(iw*ow);
        if (!weights_path.empty()) {
            std::ifstream in(weights_path, std::ios::binary | std::ios::ate);
            if (!in || in.tellg() != static_cast<std::streamoff>(weights.size()*sizeof(float)))
                throw std::runtime_error("wrong original raw float32 matrix size");
            in.seekg(0);
            in.read(reinterpret_cast<char *>(weights.data()), weights.size()*sizeof(float));
            if (!in) throw std::runtime_error("failed to read original float32 matrix");
        } else {
            // Explicitly labelled unit-test fixture, never a workload substitute.
            std::mt19937 rng(20261115);
            std::normal_distribution<float> dist(0, 0.03f);
            for (float &v : weights) v = dist(rng);
        }
        double max_weight = 0;
        for (float v : weights) {
            if (!std::isfinite(v)) throw std::invalid_argument("nonfinite original weight");
            max_weight = std::max(max_weight, std::abs(double(v)));
        }
        double native_seconds = 0, fast_seconds = 0, fast_pipeline_seconds = 0, native_decode_max = 0, fast_decode_max = 0;
        std::uint64_t mismatch_words = 0, mismatch_terms = 0, checked_words = 0;
        std::uint64_t coefficient_differences_q0 = 0, native_outside_subring_q0 = 0;
        std::vector<double> slots(n/2);
        const double scale = std::exp2(scale_log2);
        for (std::size_t t = 0; t < terms; ++t) {
            int d = t == 0 ? -255 : t == 1 ? 255 : t == 2 ? 0 : static_cast<int>(((t-3)*137)%511)-255;
            const auto split = split_diagonal(d, 16);
            auto period = diagonal_period(weights.data(), iw, ow, t%(iw/256), (t/3)%(ow/256), 256, d, split.giant);
            for (std::size_t j = 0; j < slots.size(); ++j) slots[j] = period[j%256];
            Plaintext oracle;
            auto begin = Clock::now();
            native.encode(slots, target, scale, oracle);
            native_seconds += std::chrono::duration<double>(Clock::now()-begin).count();
            begin = Clock::now();
            auto candidate = fast.encode(period, scale);
            const double small_seconds = std::chrono::duration<double>(Clock::now()-begin).count();
            fast_seconds += small_seconds;
            begin = Clock::now();
            Plaintext lifted;
            fast.materialize(candidate, scale, lifted);
            if (lifted.poly().poly_degree() != n || lifted.poly().rns_num_q() != nq ||
                lifted.poly().rns_num_p() || lifted.poly().parms_id() != target)
                throw std::logic_error("materialized Plaintext internal RNSPoly metadata differs");
            // Exact compression of the ACTUAL lifted integer polynomial must
            // still be lossless, irrespective of native FFT rounding equality.
            auto exact = compress_exact(lifted.data(), n, nq, 512);
            if (exact.period > 512) throw std::logic_error("sparse lift lost exact periodicity");
            fast_pipeline_seconds += small_seconds + std::chrono::duration<double>(Clock::now()-begin).count();
            std::uint64_t differences = 0;
            for (std::size_t word = 0; word < n*nq; ++word)
                differences += lifted.data()[word] != oracle.data()[word];
            checked_words += n*nq;
            mismatch_words += differences;
            mismatch_terms += differences != 0;
            std::vector<double> decoded;
            native.decode(oracle, decoded);
            for (std::size_t j = 0; j < slots.size(); ++j) {
                if (!std::isfinite(decoded[j])) throw std::runtime_error("nonfinite native decode");
                native_decode_max = std::max(native_decode_max, std::abs(decoded[j]-slots[j]));
            }
            native.decode(lifted, decoded);
            for (std::size_t j = 0; j < slots.size(); ++j) {
                if (!std::isfinite(decoded[j])) throw std::runtime_error("nonfinite lifted decode");
                fast_decode_max = std::max(fast_decode_max, std::abs(decoded[j]-slots[j]));
            }
            if (differences) {
                std::vector<std::uint64_t> a(oracle.data(), oracle.data()+n), b(lifted.data(), lifted.data()+n);
                util::inverse_ntt_negacyclic_harvey(a.data(), context.crt_context()->small_ntt_tables()[0]);
                util::inverse_ntt_negacyclic_harvey(b.data(), context.crt_context()->small_ntt_tables()[0]);
                for (std::size_t j = 0; j < n; ++j) {
                    coefficient_differences_q0 += a[j] != b[j];
                    if (j%(n/512)) native_outside_subring_q0 += a[j] != 0;
                }
            }
        }
        const double encoding_error_bound = std::max(1e-12*std::max(max_weight,1e-12), 1024.0/scale);
        if (native_decode_max > encoding_error_bound || fast_decode_max > encoding_error_bound)
            throw std::runtime_error("periodic encoding decoded error exceeds CPU oracle bound");
        std::cout.precision(16);
        std::cout << "{\"cpu_only\":true,\"experimental_only\":true,\"engine_fast_path_enabled\":false,\"degree\":" << n
            << ",\"q_count\":" << nq << ",\"scale_log2\":" << scale_log2 << ",\"terms\":" << terms
            << ",\"original_weights_file\":" << (weights_path.empty()?"false":"true")
            << ",\"monomial_mapping_exact\":true,\"lift_rns_compression_lossless\":true,\"checked_words\":" << checked_words
            << ",\"native_bitwise_equal\":" << (mismatch_words?"false":"true")
            << ",\"native_mismatch_terms\":" << mismatch_terms << ",\"native_mismatch_words\":" << mismatch_words
            << ",\"coefficient_differences_q0\":" << coefficient_differences_q0
            << ",\"native_outside_subring_q0\":" << native_outside_subring_q0
            << ",\"native_encode_seconds\":" << native_seconds << ",\"small_encode_remap_seconds\":" << fast_seconds
            << ",\"fast_verified_pipeline_seconds\":" << fast_pipeline_seconds
            << ",\"native_decode_max_abs\":" << native_decode_max << ",\"lift_decode_max_abs\":" << fast_decode_max
            << ",\"encoding_error_bound\":" << encoding_error_bound << "}\n";
        return require_bitwise && mismatch_words ? 3 : 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
