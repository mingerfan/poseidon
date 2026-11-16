#include "compact_linear.h"
#include "compact_layout.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/encryptor.h"
#include "poseidon/decryptor.h"
#include "poseidon/keygenerator.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "rmm/mr/device/managed_memory_resource.hpp"
#include "rmm/mr/device/pool_memory_resource.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
std::vector<float> read_raw(const std::string &path, std::size_t n) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(n * sizeof(float)))
        throw std::runtime_error("raw float32 file has wrong byte count: " + path);
    in.seekg(0);
    std::vector<float> result(n);
    if (!in.read(reinterpret_cast<char *>(result.data()), n * sizeof(float)))
        throw std::runtime_error("raw float32 read failed: " + path);
    return result;
}
struct Pool {
    rmm::mr::managed_memory_resource upstream;
    rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource> pool;
    rmm::mr::device_memory_resource *previous;
    explicit Pool(std::size_t max_bytes) : pool(&upstream, 0, max_bytes),
        previous(rmm::mr::set_current_device_resource(&pool)) {}
    ~Pool() { cudaDeviceSynchronize(); rmm::mr::set_current_device_resource(previous); }
};
void help() {
    std::cout << "Opt-in real GPU dense Linear probe (never run without parent authorization).\n"
        "Required: --run-gpu --weights original.f32 --bias original-bias.f32\n"
        "Optional: --degree 65536 --q-count 32 --p-count 16\n"
        "--q-bits 30 --q0-bits 31 --p-bits 31 --input-width 256 --output-width 256\n"
        "--level 5 --rescale-count 1 --input-scale-log2 30 --output-scale-log2 30\n"
        "--weights raw-f32.bin --bias raw-f32.bin --input raw-f32.bin\n"
        "--key-policy direct|binary --warmups 1 --iterations 3 --device 0\n"
        "--pool-mib 60000 --output raw-f32.bin --max-abs-error 0.01\n"
        "--periodic-encoding (explicit small-ring setup opt-in; strict remains default)\n";
}
} // namespace
int main(int argc, char **argv) {
    try {
        using namespace poseidon;
        using namespace poseidon::gpu;
        using namespace bert::compact;
        bool authorized = false, periodic_encoding = false;
        std::size_t degree = 65536, nq = 32, np = 16, iw = 256, ow = 256, level = 5;
        unsigned count = 1, iterations = 3, warmups = 1;
        int qbits = 30, q0bits = 31, pbits = 31, device = 0;
        double input_scale = 30, output_scale = 30, max_abs_error = 0.01;
        std::size_t pool_mib = 60000;
        std::string weights_path, bias_path, input_path, output_path, key_policy = "direct";
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help") { help(); return 0; }
            if (arg == "--run-gpu") { authorized = true; continue; }
            if (arg == "--periodic-encoding") { periodic_encoding = true; continue; }
            if (i + 1 >= argc) throw std::invalid_argument("option needs value: " + arg);
            const std::string value = argv[++i];
            if (arg == "--degree") degree = std::stoull(value);
            else if (arg == "--q-count") nq = std::stoull(value);
            else if (arg == "--p-count") np = std::stoull(value);
            else if (arg == "--q-bits") qbits = std::stoi(value);
            else if (arg == "--q0-bits") q0bits = std::stoi(value);
            else if (arg == "--p-bits") pbits = std::stoi(value);
            else if (arg == "--input-width") iw = std::stoull(value);
            else if (arg == "--output-width") ow = std::stoull(value);
            else if (arg == "--level") level = std::stoull(value);
            else if (arg == "--rescale-count") count = std::stoul(value);
            else if (arg == "--input-scale-log2") input_scale = std::stod(value);
            else if (arg == "--output-scale-log2") output_scale = std::stod(value);
            else if (arg == "--max-abs-error") max_abs_error = std::stod(value);
            else if (arg == "--iterations") iterations = std::stoul(value);
            else if (arg == "--warmups") warmups = std::stoul(value);
            else if (arg == "--pool-mib") pool_mib = std::stoull(value);
            else if (arg == "--device") device = std::stoi(value);
            else if (arg == "--weights") weights_path = value;
            else if (arg == "--bias") bias_path = value;
            else if (arg == "--input") input_path = value;
            else if (arg == "--output") output_path = value;
            else if (arg == "--key-policy") key_policy = value;
            else throw std::invalid_argument("unknown option " + arg);
        }
        if (!authorized) { help(); return 2; } // no CUDA call before this gate
        if (weights_path.empty() || bias_path.empty())
            throw std::invalid_argument("probe requires original dense --weights and --bias raw float32 files; no synthetic weight substitute");
        if (!power_of_two(degree) || degree < 1024 || !nq || !np || !iw || !ow ||
            iw % 256 || ow % 256 || !count || level < count || level >= nq || !iterations ||
            !std::isfinite(input_scale) || !std::isfinite(output_scale) ||
            !(max_abs_error > 0) || !std::isfinite(max_abs_error) ||
            (key_policy != "direct" && key_policy != "binary"))
            throw std::invalid_argument("invalid probe parameters");
        const auto setup_start = Clock::now();
        std::vector<int> bits(nq + np, qbits);
        bits.front() = q0bits;
        for (std::size_t i = nq; i < bits.size(); ++i) bits[i] = pbits;
        auto primes = CoeffModulus::Create(degree, bits);
        std::vector<Modulus> q(primes.begin(), primes.begin()+nq), p(primes.begin()+nq, primes.end());
        const auto logn = log2_exact(degree);
        ParametersLiteral literal(CKKS, logn, logn-1, input_scale, 0, 1, Modulus(0), q, p, sec_level_type::none);
        PoseidonContext context(literal);
        const auto tokens = degree / 512;
        std::mt19937 rng(20261115);
        std::normal_distribution<float> input_dist(0, 1);
        std::vector<float> weights = read_raw(weights_path, iw*ow);
        std::vector<float> bias = read_raw(bias_path, ow);
        std::vector<float> input(tokens*iw);
        if (!input_path.empty()) input = read_raw(input_path, input.size());
        else for (auto &v : input) v = input_dist(rng);
        for (float v : input)
            if (!std::isfinite(v)) throw std::invalid_argument("probe input contains nonfinite values");
        KeyGenerator keygen(context);
        PublicKey public_key;
        keygen.create_public_key(public_key);
        auto galois = std::make_shared<GaloisKeys>();
        auto steps = key_policy == "direct" ? CompactLinearEngine::required_rotation_steps() :
            CompactLinearEngine::required_binary_rotation_steps();
        keygen.create_galois_keys(steps, *galois);
        CKKSEncoder encoder(context);
        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, keygen.secret_key());
        gpu_check_cuda(cudaSetDevice(device), "probe set device");
        Pool pool(pool_mib * 1024 * 1024);
        CompactLinearEngine engine(context, galois, device);
        MatrixSpec spec;
        spec.id = "probe"; spec.input_width = iw; spec.output_width = ow;
        spec.tokens = tokens; spec.input_level = level; spec.output_level = level-count;
        spec.input_scale_log2 = input_scale; spec.output_scale_log2 = output_scale;
        spec.rescale_count = count;
        spec.periodic_encoding = periodic_encoding;
        engine.prepare(spec, weights, bias);
        std::vector<GpuCiphertextData> inputs;
        inputs.reserve(iw/256);
        std::vector<const GpuCiphertextData *> input_ptrs;
        auto context_data = context.crt_context()->first_context_data();
        while (context_data && context_data->level() != level) context_data = context_data->next_context_data();
        if (!context_data) throw std::runtime_error("input level not found");
        std::vector<double> slots(degree/2);
        for (std::size_t i = 0; i < iw/256; ++i) {
            for (std::size_t t = 0; t < tokens; ++t)
                for (std::size_t f = 0; f < 256; ++f) slots[t*256+f] = input[t*iw+i*256+f];
            Plaintext pt;
            encoder.encode(slots, context_data->parms_id(), std::exp2(input_scale), pt);
            Ciphertext ct;
            encryptor.encrypt(pt, ct);
            inputs.push_back(GpuUploader::upload_ciphertext(ct, device));
        }
        for (auto &ct : inputs) input_ptrs.push_back(&ct);
        const double setup_seconds = std::chrono::duration<double>(Clock::now()-setup_start).count();
        std::cerr << "SETUP complete seconds=" << setup_seconds << " compact_bytes="
            << engine.matrix_stats("probe").compact_bytes << '\n';
        std::vector<double> samples;
        std::vector<GpuCiphertextData> outputs;
        for (unsigned it = 0; it < warmups + iterations; ++it) {
            outputs.clear(); outputs.reserve(ow/256);
            const auto begin = Clock::now();
            for (std::size_t o = 0; o < ow/256; ++o) outputs.push_back(engine.run("probe", o, input_ptrs));
            const auto seconds = std::chrono::duration<double>(Clock::now()-begin).count();
            if (it >= warmups) samples.push_back(seconds);
            std::cerr << "ITER " << it << " seconds=" << seconds << '\n';
        }
        std::vector<float> decoded(tokens * ow);
        for (std::size_t o = 0; o < outputs.size(); ++o) {
            Ciphertext cpu;
            GpuUploader::download_ciphertext(outputs[o], cpu, context);
            Plaintext pt;
            decryptor.decrypt(cpu, pt);
            encoder.decode(pt, slots);
            for (std::size_t t = 0; t < tokens; ++t)
                for (std::size_t f = 0; f < 256; ++f) {
                    if (!std::isfinite(slots[t*256+f])) throw std::runtime_error("nonfinite decrypted GPU result");
                    decoded[t*ow+o*256+f] = slots[t*256+f];
                    if (!std::isfinite(decoded[t*ow+o*256+f])) throw std::runtime_error("decrypted result overflows output float32");
                }
        }
        double max_abs = 0, err2 = 0, ref2 = 0;
        for (std::size_t t = 0; t < tokens; ++t)
            for (std::size_t o = 0; o < ow; ++o) {
                double reference = bias[o];
                for (std::size_t i = 0; i < iw; ++i)
                    reference += double(weights[o*iw+i]) * input[t*iw+i];
                const double error = double(decoded[t*ow+o])-reference;
                if (!std::isfinite(reference) || !std::isfinite(error))
                    throw std::runtime_error("nonfinite CPU reference or GPU error");
                max_abs = std::max(max_abs, std::abs(error)); err2 += error*error; ref2 += reference*reference;
            }
        const double relative_l2 = std::sqrt(err2/std::max(ref2,1e-300));
        if (!std::isfinite(err2) || !std::isfinite(ref2) || !std::isfinite(relative_l2))
            throw std::runtime_error("nonfinite aggregate GPU correctness metrics");
        if (!output_path.empty()) {
            std::ofstream out(output_path, std::ios::binary);
            out.write(reinterpret_cast<const char *>(decoded.data()), decoded.size()*sizeof(float));
            if (!out) throw std::runtime_error("output write failed");
        }
        std::sort(samples.begin(), samples.end());
        const auto &ms = engine.matrix_stats("probe"); const auto es = engine.engine_stats();
        std::cout << std::setprecision(16) << "{\"gpu_probe\":true,\"genuine_dense_weights\":true,\"fixture_weights\":"
            << (weights_path.empty() ? "false" : "true") << ",\"input_width\":" << iw << ",\"output_width\":" << ow
            << ",\"tokens\":" << tokens << ",\"setup_seconds\":" << setup_seconds
            << ",\"weight_bytes\":" << ms.weight_bytes << ",\"compact_bytes\":" << ms.compact_bytes
            << ",\"full_equivalent_bytes\":" << ms.full_equivalent_bytes << ",\"cp_terms\":" << ms.cp_terms
            << ",\"baby_rotations\":" << ms.baby_rotations << ",\"giant_rotations\":" << ms.giant_rotations
            << ",\"fallbacks\":" << ms.compression_fallbacks << ",\"max_period\":" << ms.max_period
            << ",\"encoding_mode\":\"" << ms.encoding_mode << "\",\"periodic_encoded_terms\":" << ms.periodic_encoded_terms
            << ",\"strict_encoded_terms\":" << ms.strict_encoded_terms
            << ",\"periodic_encoding_fallbacks\":" << ms.periodic_encoding_fallbacks
            << ",\"periodic_full_ntt_mapping_verified\":" << (ms.periodic_full_ntt_mapping_verified?"true":"false")
            << ",\"native_encoder_bitwise_guaranteed\":" << (ms.native_encoder_bitwise_guaranteed?"true":"false")
            << ",\"native_encoder_audit_terms\":" << ms.native_encoder_audit_terms
            << ",\"native_encoder_mismatch_terms\":" << ms.native_encoder_mismatch_terms
            << ",\"native_encoder_mismatch_words\":" << ms.native_encoder_mismatch_words
            << ",\"key_bytes\":" << es.key_bytes << ",\"key_storage_full_q\":" << (es.key_storage_is_full_q ? "true" : "false")
            << ",\"parameter_bytes\":" << es.parameter_bytes << ",\"actual_rotation_calls\":" << es.actual_rotation_calls
            << ",\"median_seconds\":" << samples[samples.size()/2] << ",\"max_abs\":" << max_abs
            << ",\"relative_l2\":" << relative_l2 << ",\"samples_seconds\":[";
        for (std::size_t i = 0; i < samples.size(); ++i) { if (i) std::cout << ','; std::cout << samples[i]; }
        std::cout << "]}\n";
        return std::isfinite(max_abs) && max_abs <= max_abs_error ? 0 : 3;
    } catch (const std::exception &e) { std::cerr << "compact probe: " << e.what() << '\n'; return 1; }
}
