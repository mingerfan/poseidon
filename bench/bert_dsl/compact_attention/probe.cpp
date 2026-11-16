#include "prepared_qk.h"
#include "qk_reference.h"
#include "diagnostic_profile.h"
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
void help() {
    std::cout << "Opt-in genuine encrypted GPU CYCLIC QK SINGLE-PAIR probe. NOT a full-model benchmark.\n"
        "NO CUDA call or GPU context without --run-gpu; requires separate parent GPU authorization.\n"
        "Default input: clearly diagnostic random two-head Q/K, real zero gap lanes.\n"
        "Fixed N65536/T128/tile256/D64/baby8, exact q52-p11 primes, 1 warmup + 3 samples.\n"
        "--run-gpu [--pair-products | --pack-inputs] [--device 0] [--pool-mib 16000] [--key-policy direct|binary]\n"
        "[--input-level 8] [--input-scale-log2 actual] [--output-scale-log2 actual]\n"
        "[--query packed-q.f32 --key packed-k.f32] [--max-abs-error 0.0001]\n"
        "[--output decoded-cyclic-scores.f64]\n"
        "Packed files: 32768 float32 slots each; features0..63/128..191 per256 row, gaps exactly zero.\n"
        "Provided inputs are not claimed to be original model projections. All128 logical key offsets retained.\n"
        "Both experimental flags default OFF and are mutually exclusive; ideal-message zero gaps required.\n"
        "CKKS noise/rounding may change. Paired2: CC128/rot597/add575, direct extras -64/+63.\n"
        "Packed inputs: true CC64/rot531/add513, direct extras -64/+63/+192; baseline CC128/rot917/add895.\n";
}
std::vector<double> read_packed(const std::string &path) {
    constexpr std::size_t slots = 32768;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(slots * sizeof(float)))
        throw std::invalid_argument("packed input must contain exactly32768 float32 values: " + path);
    in.seekg(0);
    std::vector<float> raw(slots);
    if (!in.read(reinterpret_cast<char *>(raw.data()), raw.size() * sizeof(float)))
        throw std::runtime_error("failed reading " + path);
    return {raw.begin(), raw.end()};
}
void check_input(const std::vector<double> &input) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input[i])) throw std::invalid_argument("nonfinite packed Q/K input");
        if (i % 128 >= 64 && input[i] != 0)
            throw std::invalid_argument("packed Q/K input has nonzero feature-gap lane");
    }
}
// Independent O(T*T*D) oracle: not the BSGS execution being tested.
std::vector<double> oracle(const std::vector<double> &q, const std::vector<double> &k) {
    std::vector<double> output(32768);
    for (std::size_t t = 0; t < 128; ++t)
        for (std::size_t h = 0; h < 2; ++h)
            for (std::size_t d = 0; d < 128; ++d) {
                double dot = 0;
                for (std::size_t f = 0; f < 64; ++f)
                    dot += q[t * 256 + h * 128 + f] * k[((t + d) % 128) * 256 + h * 128 + f];
                output[t * 256 + h * 128 + d] = dot / 8.0;
                if (!std::isfinite(output[t * 256 + h * 128 + d]))
                    throw std::invalid_argument("nonfinite independent CPU oracle");
            }
    return output;
}
struct Pool {
    rmm::mr::managed_memory_resource upstream;
    rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource> pool;
    rmm::mr::device_memory_resource *previous;
    explicit Pool(std::size_t bytes) : pool(&upstream, 0, bytes),
        previous(rmm::mr::set_current_device_resource(&pool)) {}
    ~Pool() { (void)cudaDeviceSynchronize(); rmm::mr::set_current_device_resource(previous); }
};
}
int main(int argc, char **argv) {
    try {
        using namespace poseidon;
        using namespace poseidon::gpu;
        using namespace bert::attention;
        bool authorized = false, have_input_scale = false, have_output_scale = false;
        int device = 0;
        std::size_t pool_mib = 16000;
        double max_abs_error = 1e-4;
        std::string key_policy = "direct", q_path, k_path, output_path;
        QkSpec spec;
        spec.id = "diagnostic-single-pair";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") { help(); return 0; }
            if (arg == "--run-gpu") { authorized = true; continue; }
            if (arg == "--pair-products") { spec.pair_products = true; continue; }
            if (arg == "--pack-inputs") { spec.pack_inputs = true; continue; }
            if (i + 1 >= argc) throw std::invalid_argument("missing option value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--device") device = std::stoi(value);
            else if (arg == "--pool-mib") pool_mib = std::stoull(value);
            else if (arg == "--key-policy") key_policy = value;
            else if (arg == "--input-level") spec.input_level = std::stoull(value);
            else if (arg == "--input-scale-log2") { spec.input_scale_log2 = std::stod(value); have_input_scale = true; }
            else if (arg == "--output-scale-log2") { spec.output_scale_log2 = std::stod(value); have_output_scale = true; }
            else if (arg == "--query") q_path = value;
            else if (arg == "--key") k_path = value;
            else if (arg == "--output") output_path = value;
            else if (arg == "--max-abs-error") max_abs_error = std::stod(value);
            else throw std::invalid_argument("unknown option " + arg);
        }
        // Reject conflicting modes even without GPU authorization, entirely on
        // the CPU; neither --run-gpu nor a device/context is needed to reject.
        if (spec.pair_products && spec.pack_inputs)
            throw std::invalid_argument("--pair-products and --pack-inputs are mutually exclusive");
        // No CUDA call, allocator, PoseidonContext or key generation above this
        // gate. Linking the executable is NOT permission to run CUDA.
        if (!authorized) { help(); return 2; }
        if (device < 0 || !pool_mib || spec.input_level < 4 || spec.input_level >= 52 ||
            !std::isfinite(max_abs_error) || max_abs_error <= 0 ||
            (key_policy != "direct" && key_policy != "binary") || q_path.empty() != k_path.empty())
            throw std::invalid_argument("invalid probe options (provide BOTH --query and --key)");
        spec.output_level = spec.input_level - 4;
        const auto ordered_q = diagnostic::q52();
        if (!have_input_scale) spec.input_scale_log2 = diagnostic::canonical_scale_log2(spec.input_level);
        if (!have_output_scale) spec.output_scale_log2 = diagnostic::canonical_scale_log2(spec.output_level);
        spec.intermediate_scale_log2 = 2 * spec.input_scale_log2 -
            std::log2(double(ordered_q[spec.input_level])) - std::log2(double(ordered_q[spec.input_level - 1]));
        const auto scale_plan = reference::derive_scale_plan(spec, ordered_q);
        std::vector<double> query(32768, 0), key(query.size(), 0);
        const bool diagnostic_random = q_path.empty();
        if (diagnostic_random) {
            std::mt19937_64 rng(20261115);
            std::normal_distribution<double> normal(0, 0.25);
            for (std::size_t t = 0; t < 128; ++t)
                for (std::size_t h = 0; h < 2; ++h)
                    for (std::size_t f = 0; f < 64; ++f) {
                        const auto i = t * 256 + h * 128 + f;
                        query[i] = normal(rng) * (h ? 1.4 : 0.8) + (h ? -0.1 : 0.03);
                        key[i] = normal(rng) * (h ? 0.7 : 1.2) + (h ? 0.08 : -0.05);
                    }
        } else { query = read_packed(q_path); key = read_packed(k_path); }
        check_input(query); check_input(key);
        const auto expected = oracle(query, key); // independent CPU, not timed
        const auto setup_begin = Clock::now();
        std::vector<Modulus> q, p;
        for (auto prime : ordered_q) q.emplace_back(prime);
        for (auto prime : diagnostic::p11()) p.emplace_back(prime);
        ParametersLiteral literal(CKKS, 16, 15, 28, 32768, 1, Modulus(0), q, p, sec_level_type::none);
        PoseidonContext context(literal);
        KeyGenerator keygen(context);
        PublicKey public_key;
        keygen.create_public_key(public_key);
        auto relin = std::make_shared<RelinKeys>();
        auto galois = std::make_shared<GaloisKeys>();
        keygen.create_relin_keys(*relin);
        const auto direct_steps = spec.pack_inputs ? PreparedQkEngine::required_packed_input_rotation_steps() :
                                  spec.pair_products ? PreparedQkEngine::required_paired_rotation_steps() :
                                                       PreparedQkEngine::required_rotation_steps();
        const auto binary_steps = spec.pack_inputs ? PreparedQkEngine::required_packed_input_binary_rotation_steps() :
                                  spec.pair_products ? PreparedQkEngine::required_paired_binary_rotation_steps() :
                                                       PreparedQkEngine::required_binary_rotation_steps();
        keygen.create_galois_keys(key_policy == "direct" ? direct_steps : binary_steps, *galois);
        if ((spec.pair_products || spec.pack_inputs) && key_policy == "direct") {
            for (int step : {-64, 63})
                if (!galois->has_key(context.crt_context()->galois_tool()->get_elt_from_step(step)))
                    throw std::runtime_error("experimental direct probe did not generate genuine -64/+63 keys");
            if (spec.pack_inputs && !galois->has_key(context.crt_context()->galois_tool()->get_elt_from_step(192)))
                throw std::runtime_error("packed-input direct probe did not generate genuine +192 key");
        }
        CKKSEncoder encoder(context);
        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, keygen.secret_key());
        auto input_level = context.crt_context()->first_context_data();
        while (input_level && (input_level->parms().q().size() != spec.input_level + 1 ||
                              !input_level->parms().p().empty())) input_level = input_level->next_context_data();
        if (!input_level) throw std::runtime_error("probe input physical level not found");
        Plaintext pq, pk;
        encoder.encode(query, input_level->parms_id(), std::exp2(spec.input_scale_log2), pq);
        encoder.encode(key, input_level->parms_id(), std::exp2(spec.input_scale_log2), pk);
        Ciphertext cq, ck;
        encryptor.encrypt(pq, cq); encryptor.encrypt(pk, ck);
        gpu_check_cuda(cudaSetDevice(device), "QK probe select device");
        Pool pool(pool_mib * 1024 * 1024);
        PreparedQkEngine engine(context, relin, galois, device);
        engine.prepare(spec);
        auto gq = GpuUploader::upload_ciphertext(cq, device);
        auto gk = GpuUploader::upload_ciphertext(ck, device);
        gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "QK probe synchronize caller inputs");
        const auto setup_seconds = std::chrono::duration<double>(Clock::now() - setup_begin).count();
        std::cerr << "SETUP input_kind=" << (diagnostic_random ? "diagnostic_random_gapped_qk" : "provided_gapped_qk")
                  << " pair_products=" << (spec.pair_products ? "true" : "false")
                  << " pack_inputs=" << (spec.pack_inputs ? "true" : "false")
                  << " seconds=" << setup_seconds << '\n';
        std::vector<double> samples;
        GpuCiphertextData output;
        for (unsigned iteration = 0; iteration < 4; ++iteration) {
            const auto begin = Clock::now();
            output = engine.run(spec.id, gq, gk);
            const auto seconds = std::chrono::duration<double>(Clock::now() - begin).count();
            if (!std::isfinite(seconds) || seconds < 0) throw std::runtime_error("nonfinite GPU sample time");
            if (iteration) samples.push_back(seconds);
            std::cerr << (iteration ? "SAMPLE " : "WARMUP ") << iteration << " seconds=" << seconds << '\n';
        }
        Ciphertext cpu;
        GpuUploader::download_ciphertext(output, cpu, context);
        Plaintext decrypted;
        decryptor.decrypt(cpu, decrypted);
        std::vector<double> decoded;
        encoder.decode(decrypted, decoded);
        if (decoded.size() != expected.size()) throw std::runtime_error("decoded slot count mismatch");
        double max_abs = 0, err2 = 0, ref2 = 0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const double error = decoded[i] - expected[i];
            if (!std::isfinite(decoded[i]) || !std::isfinite(error))
                throw std::runtime_error("nonfinite genuine GPU result/error");
            max_abs = std::max(max_abs, std::abs(error));
            err2 += error * error; ref2 += expected[i] * expected[i];
        }
        const auto relative_l2 = std::sqrt(err2 / std::max(ref2, 1e-300));
        if (!std::isfinite(err2) || !std::isfinite(ref2) || !std::isfinite(relative_l2))
            throw std::runtime_error("nonfinite aggregate GPU correctness metrics");
        if (!output_path.empty()) {
            std::ofstream out(output_path, std::ios::binary);
            out.write(reinterpret_cast<const char *>(decoded.data()), decoded.size() * sizeof(double));
            if (!out) throw std::runtime_error("failed writing decoded cyclic scores");
        }
        const auto stats = engine.engine_stats();
        const auto &qk_stats = engine.qk_stats(spec.id);
        const std::uint64_t expected_adds = 4 * (spec.pack_inputs ? 513 : spec.pair_products ? 575 : 895);
        const std::uint64_t expected_rotations = 4 * (spec.pack_inputs ? 531 : spec.pair_products ? 597 : 917);
        const std::uint64_t expected_cc = 4 * (spec.pack_inputs ? 64 : 128);
        const std::uint64_t expected_r2 = 4 * (spec.pack_inputs ? 65 : 129);
        if (stats.run_calls != 4 || stats.baseline_run_calls != ((!spec.pair_products && !spec.pack_inputs) ? 4U : 0U) ||
            stats.paired_run_calls != (spec.pair_products ? 4U : 0U) ||
            stats.packed_input_run_calls != (spec.pack_inputs ? 4U : 0U) ||
            stats.cc_multiply_calls != expected_cc || stats.relinearize_calls != expected_cc ||
            stats.cp_multiply_calls != 512 || stats.rescale2_calls != expected_r2 || stats.physical_rescale_primes != 2 * expected_r2 ||
            stats.add_calls != expected_adds || (key_policy == "direct" &&
                (stats.rotation_calls != expected_rotations || stats.hoisted_rotation_batches != 8)))
            throw std::runtime_error("actual QK primitive counters differ from complete four-pair schedule");
        auto sorted = samples; std::sort(sorted.begin(), sorted.end());
        std::cout << std::setprecision(16) << "{\"gpu_probe\":true,\"full_model_measurement\":false,\"input_kind\":\""
            << (diagnostic_random ? "diagnostic_random_gapped_qk" : "provided_gapped_qk")
            << "\",\"layout\":\"cyclic_key_offset\",\"tokens\":128,\"heads_per_pair\":2,\"logical_key_offsets_per_pair\":128,"
            << "\"cc_products_per_pair\":" << expected_cc / 4 << ",\"pair_products\":" << (spec.pair_products ? "true" : "false")
            << ",\"pack_inputs\":" << (spec.pack_inputs ? "true" : "false")
            << ",\"schedule\":\"" << qk_stats.schedule << "\",\"encoding\":\"" << qk_stats.encoding << "\","
            << "\"ideal_message_gap_zero_required\":true,\"ciphertext_bitwise_equivalence_claimed\":false,"
            << "\"q_moduli_count\":52,\"p_moduli_count\":11,\"key_policy\":\"" << key_policy << "\","
            << "\"input_level\":" << spec.input_level << ",\"input_scale_log2\":" << spec.input_scale_log2
            << ",\"intermediate_scale_log2\":" << spec.intermediate_scale_log2 << ",\"output_level\":" << spec.output_level
            << ",\"output_scale_log2\":" << std::log2(output.meta.scale) << ",\"mask_scale_log2\":" << scale_plan.mask_scale_log2
            << ",\"setup_seconds\":" << setup_seconds << ",\"engine_setup_seconds\":" << stats.setup_seconds
            << ",\"allocated_bytes\":" << stats.allocated_bytes << ",\"parameter_bytes\":" << stats.parameter_bytes
            << ",\"galois_key_bytes\":" << stats.galois_key_bytes << ",\"relin_key_bytes\":" << stats.relin_key_bytes
            << ",\"mask_bytes\":" << stats.mask_bytes << ",\"key_storage_q_count\":" << stats.key_storage_q_count
            << ",\"key_storage_p_count\":" << stats.key_storage_p_count
            << ",\"key_uploads\":" << stats.key_uploads << ",\"uploaded_rotation_steps\":" << stats.uploaded_rotation_steps
            << ",\"cc_multiply_calls\":" << stats.cc_multiply_calls << ",\"relinearize_calls\":" << stats.relinearize_calls
            << ",\"cp_multiply_calls\":" << stats.cp_multiply_calls << ",\"rescale2_calls\":" << stats.rescale2_calls
            << ",\"rotation_calls\":" << stats.rotation_calls << ",\"hoisted_rotation_batches\":" << stats.hoisted_rotation_batches
            << ",\"add_calls\":" << stats.add_calls << ",\"warmups\":1,\"median_seconds\":" << sorted[1]
            << ",\"max_abs\":" << max_abs << ",\"relative_l2\":" << relative_l2 << ",\"samples_seconds\":["
            << samples[0] << ',' << samples[1] << ',' << samples[2] << "]}\n";
        return max_abs <= max_abs_error ? 0 : 3;
    } catch (const std::exception &e) { std::cerr << "QK probe: " << e.what() << '\n'; return 1; }
}
