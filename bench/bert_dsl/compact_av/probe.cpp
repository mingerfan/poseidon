#include "prepared_av.h"
#include "av_reference.h"
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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
void help() {
    std::cout << "Opt-in genuine encrypted GPU CYCLIC AV SINGLE-PAIR diagnostic; NOT full BERT.\n"
        "No CUDA device call/context without --run-gpu. GPU execution requires separate authorization.\n"
        "Fixed N65536/T128/tile256/head_dim64/baby8, q52-p11, L10/L8 -> L6 -> L4.\n"
        "Full128-key weighted V BEFORE normalization; no Softmax/model inference is measured.\n"
        "Default: diagnostic RANDOM full cyclic scores and gapped two-head V, not model scores.\n"
        "1 warmup + 3 samples; independent direct oracle checks ALL32768 slots and rejects nonfinite.\n"
        "--run-gpu [--device 0] [--pool-mib 16000] [--key-policy direct|binary]\n"
        "[--pack-inputs] [--prepare-both-algorithms] (packed mode requires direct keys)\n"
        "PackedInput2Partial: ALL32768 defined partial slots checked; live16384 match full AV,\n"
        "gaps intentionally nonzero B_current+A_next_head/row. NOT baseline zero-gap AV.\n"
        "Do NOT connect partial output to old unmasked reciprocal/merge consumers.\n"
        "[--prepared-ids 72] [--scores packed-scores.f32 --value packed-value.f32]\n"
        "[--score-scale-log2 S10] [--value-scale-log2 S8] [--common-scale-log2 S6]\n"
        "[--output-scale-log2 S4] [--max-abs-error 0.0001] [--output decoded.f64]\n"
        "Each packed file:32768 float32. Scores full[t,head,d128]; V[t,head,feature128],\n"
        "V feature64..127 of EACH head must be exactly zero. Both heads isolated.\n";
}
std::vector<double> read_packed(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(32768 * sizeof(float)))
        throw std::invalid_argument("packed input must contain exactly32768 float32: " + path);
    in.seekg(0);
    std::vector<float> raw(32768);
    if (!in.read(reinterpret_cast<char *>(raw.data()), raw.size() * sizeof(float)))
        throw std::runtime_error("failed reading " + path);
    return {raw.begin(), raw.end()};
}
void finite(const std::vector<double> &values, const char *name) {
    for (double x : values) if (!std::isfinite(x)) throw std::invalid_argument(std::string("nonfinite ") + name);
}
struct Pool {
    rmm::mr::managed_memory_resource upstream;
    rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource> pool;
    rmm::mr::device_memory_resource *previous;
    explicit Pool(std::size_t bytes) : pool(&upstream, 0, bytes), previous(rmm::mr::set_current_device_resource(&pool)) {}
    ~Pool() { (void)cudaDeviceSynchronize(); rmm::mr::set_current_device_resource(previous); }
};
void verify_counts(const bert::attention::AvEngineStats &x, bool direct, bool packed) {
    // These are checks of counters incremented at REAL evaluator call sites,
    // not values substituted into reported stats. Packed mode pairs full-key
    // terms in disjoint lanes; baseline still performs all128 original CCs.
    constexpr std::uint64_t runs = 4, T = 128, baby = 8, giants = T / baby;
    constexpr std::uint64_t D = T/2;
    const auto cc = packed ? T/2 : T;
    const auto cp = packed ? T + 1 + (T-1) + (D-1) : T + 2*(T-1) + 1;
    const auto adds = packed ? (T-1)+1+(D-1)+T/2+(T/2-1)+1 : 3*(T-1);
    const auto rotations = packed ? (baby-1)+(giants-1)+1+(baby/2-1)+(giants-1)+
        giants*(baby-1)+(D-1)+T/2+(giants-1)+1 :
        (baby-1)*2+(giants-1)*3+giants*(baby-1)+(T-1);
    if (x.run_calls != runs || x.cc_multiply_calls != runs*cc || x.relinearize_calls != runs*cc ||
        x.baseline_run_calls != (packed ? 0 : runs) || x.packed_input2_partial_run_calls != (packed ? runs : 0) ||
        x.cp_multiply_calls != runs*cp || x.rescale_calls != runs*(1 + T + cc) ||
        x.rescale2_calls != runs*(1 + cc) || x.rescale4_calls != runs*T ||
        x.physical_rescale_primes != runs*(2 + 4*T + 2*cc) || x.add_calls != runs*adds ||
        (direct && (x.rotation_calls != runs*rotations || x.hoisted_rotation_batches != runs*(3 + giants))))
        throw std::runtime_error("actual AV evaluator counters disagree with complete four-pair schedule");
}
} // namespace
int main(int argc, char **argv) {
    try {
        using namespace poseidon;
        using namespace poseidon::gpu;
        using namespace bert::attention;
        bool authorized = false, prepare_both = false;
        int device = 0;
        std::size_t pool_mib = 16000, prepared_ids = 72;
        double max_abs_error = 1e-4;
        std::string key_policy = "direct", scores_path, value_path, output_path;
        AvSpec spec; spec.id = "diagnostic-single-pair";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") { help(); return 0; }
            if (arg == "--run-gpu") { authorized = true; continue; }
            if (arg == "--pack-inputs") { spec.algorithm = AvAlgorithm::PackedInput2Partial; continue; }
            if (arg == "--prepare-both-algorithms") { prepare_both = true; continue; }
            if (i + 1 >= argc) throw std::invalid_argument("missing option value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--device") device = std::stoi(value);
            else if (arg == "--pool-mib") pool_mib = std::stoull(value);
            else if (arg == "--prepared-ids") prepared_ids = std::stoull(value);
            else if (arg == "--key-policy") key_policy = value;
            else if (arg == "--scores") scores_path = value;
            else if (arg == "--value") value_path = value;
            else if (arg == "--output") output_path = value;
            else if (arg == "--max-abs-error") max_abs_error = std::stod(value);
            else if (arg == "--score-scale-log2") spec.score_input_scale_log2 = std::stod(value);
            else if (arg == "--value-scale-log2") spec.value_input_scale_log2 = std::stod(value);
            else if (arg == "--common-scale-log2") spec.common_scale_log2 = std::stod(value);
            else if (arg == "--output-scale-log2") spec.output_scale_log2 = std::stod(value);
            else throw std::invalid_argument("unknown option " + arg);
        }
        // NO allocator, context, key generation or explicit CUDA device calls precede
        // this gate. Linked CUDA kernels can register fatbinaries at process start
        // without creating a GPU context or executing GPU work.
        if (!authorized) { help(); return 2; }
        if (device < 0 || !pool_mib || !prepared_ids || prepared_ids > 10000 ||
            !std::isfinite(max_abs_error) || max_abs_error <= 0 ||
            (key_policy != "direct" && key_policy != "binary") || scores_path.empty() != value_path.empty())
            throw std::invalid_argument("invalid options (provide BOTH --scores and --value)");
        const bool packed = spec.algorithm == AvAlgorithm::PackedInput2Partial;
        if ((packed || prepare_both) && key_policy != "direct")
            throw std::invalid_argument("packed/mixed diagnostic probe requires genuine55 direct keys");
        for (double scale : {spec.score_input_scale_log2, spec.value_input_scale_log2, spec.common_scale_log2, spec.output_scale_log2})
            if (!std::isfinite(scale) || scale <= 0 || scale > 200) throw std::invalid_argument("invalid scale option");
        std::vector<double> scores(32768), value(32768, 0);
        const bool random_input = scores_path.empty();
        if (random_input) {
            std::mt19937_64 rng(20261115);
            std::normal_distribution<double> normal(0, 0.1);
            for (std::size_t t = 0; t < 128; ++t)
                for (std::size_t h = 0; h < 2; ++h)
                    for (std::size_t f = 0; f < 128; ++f) {
                        const auto i = t * 256 + h * 128 + f;
                        scores[i] = normal(rng) + (h ? 0.2 : 0.1);
                        if (f < 64) value[i] = normal(rng) * (h ? 1.4 : 0.7) + (h ? -0.08 : 0.03);
                    }
        } else { scores = read_packed(scores_path); value = read_packed(value_path); }
        finite(scores, "scores"); finite(value, "V");
        av_reference::Layout layout;
        av_reference::validate_values(value, layout);
        const auto full_av = av_reference::direct(scores, value, layout);
        const auto expected = packed ? av_reference::partial_direct_oracle(scores, value, layout) : full_av;
        finite(expected, "independent defined-output CPU oracle");
        const auto setup_begin = Clock::now();
        std::vector<Modulus> q, p;
        for (auto prime : av_diagnostic::q52()) q.emplace_back(prime);
        for (auto prime : av_diagnostic::p11()) p.emplace_back(prime);
        ParametersLiteral literal(CKKS, 16, 15, 28, 32768, 1, Modulus(0), q, p, sec_level_type::none);
        PoseidonContext context(literal);
        KeyGenerator keygen(context);
        PublicKey public_key; keygen.create_public_key(public_key);
        auto relin = std::make_shared<RelinKeys>(); auto galois = std::make_shared<GaloisKeys>();
        keygen.create_relin_keys(*relin);
        keygen.create_galois_keys((packed || prepare_both) ? PreparedAvEngine::required_packed_input_rotation_steps() :
            (key_policy == "direct" ? PreparedAvEngine::required_rotation_steps() :
             PreparedAvEngine::required_binary_rotation_steps()), *galois);
        CKKSEncoder encoder(context); Encryptor encryptor(context, public_key); Decryptor decryptor(context, keygen.secret_key());
        auto find_level = [&](std::size_t level) {
            auto data = context.crt_context()->first_context_data();
            while (data && (data->parms().q().size() != level + 1 || !data->parms().p().empty())) data = data->next_context_data();
            if (!data) throw std::runtime_error("probe physical level absent");
            return data;
        };
        Plaintext ps, pv;
        encoder.encode(scores, find_level(spec.score_input_level)->parms_id(), std::exp2(spec.score_input_scale_log2), ps);
        encoder.encode(value, find_level(spec.value_input_level)->parms_id(), std::exp2(spec.value_input_scale_log2), pv);
        Ciphertext cs, cv; encryptor.encrypt(ps, cs); encryptor.encrypt(pv, cv);
        gpu_check_cuda(cudaSetDevice(device), "AV probe select device");
        Pool pool(pool_mib * 1024 * 1024);
        PreparedAvEngine engine(context, relin, galois, device);
        auto prepare_mode = [&](AvAlgorithm algorithm) {
            auto mode = spec; mode.algorithm = algorithm;
            mode.id = algorithm == spec.algorithm ? spec.id : std::string("other-") + av_algorithm_name(algorithm);
            engine.prepare(mode);
            if (engine.av_stats(mode.id).mask_plaintext_count !=
                (algorithm == AvAlgorithm::Baseline ? 383u : 319u))
                throw std::runtime_error("actual encoded mask inventory mismatch");
            const auto first = engine.engine_stats();
            for (std::size_t i = 1; i < prepared_ids; ++i) {
                auto other = mode; other.id = mode.id + "-shared-" + std::to_string(i); engine.prepare(other);
                const auto current = engine.engine_stats();
                if (!engine.av_stats(other.id).reused_mask_bundle || current.mask_bundle_count != first.mask_bundle_count ||
                    current.mask_bytes != first.mask_bytes || current.mask_plaintext_count != first.mask_plaintext_count ||
                    current.allocated_bytes != first.allocated_bytes ||
                    current.galois_key_upload_calls != first.galois_key_upload_calls)
                    throw std::runtime_error("same-physical+algorithm IDs duplicated masks/keys/parameters");
            }
        };
        if (prepare_both) {
            // Baseline first deliberately tests52->55 key extension/reupload.
            prepare_mode(AvAlgorithm::Baseline); prepare_mode(AvAlgorithm::PackedInput2Partial);
            const auto mixed = engine.engine_stats();
            if (mixed.mask_bundle_count != 2 || mixed.mask_plaintext_count != 383+319 || mixed.galois_key_upload_calls != 2 ||
                mixed.relin_key_upload_calls != 2 || mixed.key_storage_q_count != 11 || mixed.key_storage_p_count != 11)
                throw std::runtime_error("mixed algorithms failed isolated masks / owning key extension");
        } else prepare_mode(spec.algorithm);
        auto gs = GpuUploader::upload_ciphertext(cs, device), gv = GpuUploader::upload_ciphertext(cv, device);
        gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "AV caller input synchronize");
        const double setup_seconds = std::chrono::duration<double>(Clock::now() - setup_begin).count();
        std::cerr << "SETUP input_kind=" << (random_input ? "diagnostic_random_scores_and_gapped_V" : "provided_cyclic_scores_and_gapped_V")
                  << " algorithm=" << av_algorithm_name(spec.algorithm) << " partial_output=" << packed
                  << " full_model=false seconds=" << setup_seconds << '\n';
        std::vector<double> samples; GpuCiphertextData output;
        for (unsigned iteration = 0; iteration < 4; ++iteration) {
            const auto begin = Clock::now(); output = engine.run(spec.id, gs, gv);
            const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
            if (!std::isfinite(seconds) || seconds < 0) throw std::runtime_error("nonfinite GPU sample time");
            if (iteration) samples.push_back(seconds);
            std::cerr << (iteration ? "SAMPLE " : "WARMUP ") << iteration << " seconds=" << seconds << '\n';
        }
        Ciphertext cpu; GpuUploader::download_ciphertext(output, cpu, context);
        Plaintext decrypted; decryptor.decrypt(cpu, decrypted);
        std::vector<double> decoded; encoder.decode(decrypted, decoded);
        if (decoded.size() != expected.size()) throw std::runtime_error("decoded slot count mismatch");
        double max_abs = 0, live_full_av_max_abs = 0, gap_max_abs = 0, err2 = 0, ref2 = 0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const double error = decoded[i] - expected[i];
            if (!std::isfinite(decoded[i]) || !std::isfinite(error)) throw std::runtime_error("nonfinite real GPU output/error");
            max_abs = std::max(max_abs, std::abs(error));
            if (i % 128 >= 64) gap_max_abs = std::max(gap_max_abs, std::abs(decoded[i]));
            else live_full_av_max_abs = std::max(live_full_av_max_abs, std::abs(decoded[i] - full_av[i]));
            err2 += error*error; ref2 += expected[i]*expected[i];
        }
        const double relative_l2 = std::sqrt(err2/std::max(ref2, 1e-300));
        if (!std::isfinite(err2) || !std::isfinite(ref2) || !std::isfinite(relative_l2)) throw std::runtime_error("nonfinite aggregate metrics");
        if (!output_path.empty()) {
            std::ofstream out(output_path, std::ios::binary);
            out.write(reinterpret_cast<const char *>(decoded.data()), decoded.size()*sizeof(double));
            if (!out) throw std::runtime_error("failed writing AV decoded output");
        }
        const auto x = engine.engine_stats(); verify_counts(x, key_policy == "direct", packed);
        bool late_prepare_rejected = false;
        try { auto late = spec; late.id = "illegal-late-prepare"; engine.prepare(late); }
        catch (const std::logic_error &) { late_prepare_rejected = true; }
        if (!late_prepare_rejected) throw std::runtime_error("prepare after run was not rejected");
        const auto &a = engine.av_stats(spec.id);
        auto sorted = samples; std::sort(sorted.begin(), sorted.end());
        std::cout << std::setprecision(16) << "{\"gpu_probe\":true,\"full_model_measurement\":false,\"input_kind\":\""
            << (random_input ? "diagnostic_random_scores_and_gapped_V" : "provided_cyclic_scores_and_gapped_V")
            << "\",\"tokens\":128,\"heads_per_pair\":2,\"keys_per_row\":128,\"features_per_head\":64,\"prepared_ids\":" << prepared_ids
            << ",\"algorithm\":\"" << av_algorithm_name(spec.algorithm) << "\",\"schedule\":\"" << a.schedule
            << "\",\"output_semantics\":\"" << a.output_semantics
            << "\",\"partial_output\":" << (packed ? "true" : "false") << ",\"gap_nonzero_is_expected\":" << (packed ? "true" : "false")
            << ",\"prepare_both_algorithms\":" << (prepare_both ? "true" : "false")
            << ",\"key_policy\":\"" << key_policy << "\",\"q_moduli_count\":52,\"p_moduli_count\":11"
            << ",\"score_input_level\":10,\"value_input_level\":8,\"common_level\":6,\"output_level\":4"
            << ",\"score_input_scale_log2\":" << spec.score_input_scale_log2 << ",\"value_input_scale_log2\":" << spec.value_input_scale_log2
            << ",\"common_scale_log2\":" << spec.common_scale_log2 << ",\"output_scale_log2\":" << std::log2(output.meta.scale)
            << ",\"diagonal_mask_scale_log2\":" << a.diagonal_mask_scale_log2 << ",\"score_mask_scale_log2\":" << a.score_mask_scale_log2
            << ",\"setup_seconds\":" << setup_seconds << ",\"engine_setup_seconds\":" << x.setup_seconds
            << ",\"allocated_bytes\":" << x.allocated_bytes << ",\"parameter_bytes\":" << x.parameter_bytes
            << ",\"galois_key_bytes\":" << x.galois_key_bytes << ",\"relin_key_bytes\":" << x.relin_key_bytes
            << ",\"mask_bytes\":" << x.mask_bytes << ",\"mask_bundle_count\":" << x.mask_bundle_count
            << ",\"unique_mask_plaintext_count\":" << x.mask_plaintext_count << ",\"id_shared_mask_plaintext_count\":" << a.mask_plaintext_count
            << ",\"key_storage_q_count\":" << x.key_storage_q_count << ",\"key_storage_p_count\":" << x.key_storage_p_count
            << ",\"relin_key_upload_calls\":" << x.relin_key_upload_calls << ",\"galois_key_upload_calls\":" << x.galois_key_upload_calls
            << ",\"preparation_locked\":" << (x.preparation_locked ? "true" : "false")
            << ",\"run_calls\":" << x.run_calls << ",\"baseline_run_calls\":" << x.baseline_run_calls
            << ",\"packed_input2_partial_run_calls\":" << x.packed_input2_partial_run_calls << ",\"cc_multiply_calls\":" << x.cc_multiply_calls
            << ",\"relinearize_calls\":" << x.relinearize_calls << ",\"cp_multiply_calls\":" << x.cp_multiply_calls
            << ",\"rescale_calls\":" << x.rescale_calls << ",\"rescale2_calls\":" << x.rescale2_calls << ",\"rescale4_calls\":" << x.rescale4_calls
            << ",\"physical_rescale_primes\":" << x.physical_rescale_primes << ",\"rotation_calls\":" << x.rotation_calls
            << ",\"hoisted_rotation_batches\":" << x.hoisted_rotation_batches << ",\"add_calls\":" << x.add_calls
            << ",\"warmups\":1,\"median_seconds\":" << sorted[1] << ",\"max_abs\":" << max_abs
            << ",\"live_full_av_max_abs\":" << live_full_av_max_abs << ",\"live_checked_slots\":16384,\"defined_output_checked_slots\":32768"
            << ",\"gap_max_abs\":" << gap_max_abs << ",\"gap_max_abs_is_not_error\":true,\"relative_l2\":" << relative_l2 << ",\"samples_seconds\":["
            << samples[0] << ',' << samples[1] << ',' << samples[2] << "]}\n";
        return max_abs <= max_abs_error && live_full_av_max_abs <= max_abs_error ? 0 : 3;
    } catch (const std::exception &e) { std::cerr << "AV probe: " << e.what() << '\n'; return 1; }
}
