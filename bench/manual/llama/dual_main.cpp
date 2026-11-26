// Dual-GPU handwritten LLaMA runner for 2xA800 execution, BF16 GPU weight
// loading, compute/comm ratio sweeps, and simulator calibration telemetry.
#include "source_dual_model.hpp"
#include "source_checkpoint.hpp"
#include "llama_manual/benchmark_io.hpp"
#include <condition_variable>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

class ExclusiveReport {
    int fd_;
public:
    explicit ExclusiveReport(const std::string &path)
        : fd_(::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW | O_CLOEXEC, 0600)) {
        if (fd_ < 0) throw std::runtime_error("exclusive dual report open failed: " + path);
    }
    ~ExclusiveReport() { if (fd_ >= 0) ::close(fd_); }
    void finish(const std::string &text) {
        std::size_t at = 0;
        while (at < text.size()) {
            const auto n = ::write(fd_, text.data() + at, text.size() - at);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("dual report write failed");
            at += static_cast<std::size_t>(n);
        }
        if (::fsync(fd_)) throw std::runtime_error("dual report fsync failed");
    }
};

struct SweepJob {
    cerium_source::dual::RunOptions opt;
    int p2p_mode = -1; // -1: keep env, 0: host_staged, 1: p2p_direct
    bool check_accuracy = false;
    std::string report_path;
};

struct CliArgs {
    int gpus = 2;
    cerium_source::manual::Mode mode = cerium_source::manual::Mode::Fast;
    std::string fixture, weights, index, embeddings, reference, report, sweep_jobs_path;
    std::string public_cache, dev_key_cache, ckpt_save_dir, ckpt_load;
    std::size_t layers = 1;
    std::size_t start_layer = 0;
    std::size_t end_layer = 0;
    bool start_layer_explicit = false;
    bool end_layer_explicit = false;
    std::size_t ckpt_save_every = 1;
    std::size_t valid_tokens = 38;
    cerium_source::dual::RunOptions opt;
    bool check_accuracy = true;
};

CliArgs parse_cli(int argc, char **argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&](const char *flag) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (k == "--gpus") a.gpus = std::stoi(next("--gpus"));
        else if (k == "--plan") a.mode = cerium_source::manual::parse_mode(next("--plan"));
        else if (k == "--fixture") a.fixture = next("--fixture");
        else if (k == "--weights") a.weights = next("--weights");
        else if (k == "--index") a.index = next("--index");
        else if (k == "--embeddings") a.embeddings = next("--embeddings");
        else if (k == "--reference") a.reference = next("--reference");
        else if (k == "--report") a.report = next("--report");
        else if (k == "--sweep-jobs") a.sweep_jobs_path = next("--sweep-jobs");
        else if (k == "--public-cache") a.public_cache = next("--public-cache");
        else if (k == "--dev-key-cache") a.dev_key_cache = next("--dev-key-cache");
        else if (k == "--ckpt-save-dir") a.ckpt_save_dir = next("--ckpt-save-dir");
        else if (k == "--ckpt-save-every") a.ckpt_save_every = std::stoull(next("--ckpt-save-every"));
        else if (k == "--ckpt-load") a.ckpt_load = next("--ckpt-load");
        else if (k == "--layers") a.layers = std::stoull(next("--layers"));
        else if (k == "--start-layer") { a.start_layer = std::stoull(next("--start-layer")); a.start_layer_explicit = true; }
        else if (k == "--end-layer") { a.end_layer = std::stoull(next("--end-layer")); a.end_layer_explicit = true; }
        else if (k == "--valid-tokens") a.valid_tokens = std::stoull(next("--valid-tokens"));
        else if (k == "--subbench") a.opt.subbench = next("--subbench");
        else if (k == "--compute-scale") a.opt.compute_scale = std::stoi(next("--compute-scale"));
        else if (k == "--serial-compute-scale") a.opt.serial_compute_scale = std::stoi(next("--serial-compute-scale"));
        else if (k == "--comm-repeat") a.opt.comm_repeat = std::stoull(next("--comm-repeat"));
        else if (k == "--stage-timing") a.opt.stage_timing = (std::stoi(next("--stage-timing")) != 0);
        else if (k == "--no-accuracy") a.check_accuracy = false;
        else throw std::invalid_argument("unknown argument: " + k);
    }
    if (a.gpus != 1 && a.gpus != 2) throw std::invalid_argument("--gpus must be 1 or 2");
    if (a.ckpt_save_every == 0) throw std::invalid_argument("--ckpt-save-every must be >= 1");
    a.public_cache = cerium_source::cache::public_cache_dir(a.public_cache);
    a.dev_key_cache = cerium_source::cache::dev_key_cache_dir(a.dev_key_cache);
    if (a.ckpt_save_dir.empty()) a.ckpt_save_dir = cerium_source::cache::env_path("LLAMA_CKPT_SAVE_DIR");
    if (a.ckpt_load.empty()) a.ckpt_load = cerium_source::cache::env_path("LLAMA_CKPT_LOAD");
    if (!a.end_layer_explicit) a.end_layer = a.layers;
    if (a.end_layer == 0 || a.end_layer > a.layers) throw std::invalid_argument("--end-layer out of range");
    a.opt.gpus = a.gpus;
    if (a.fixture.empty() || a.weights.empty() || a.index.empty() ||
        a.embeddings.empty() || a.reference.empty() || a.report.empty()) {
        throw std::invalid_argument("missing required file arguments");
    }
    if (a.opt.subbench != "full") a.check_accuracy = false;
    return a;
}

std::vector<SweepJob> load_sweep_jobs(const std::string &path, const CliArgs &base) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open sweep jobs file: " + path);
    std::vector<SweepJob> jobs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string subbench, p2p_tok, rep_path;
        int cs = 1, scs = 0, acc_flag = 0;
        std::size_t rep = 1;
        if (!(iss >> subbench >> cs >> scs >> rep >> p2p_tok >> acc_flag >> rep_path)) {
            throw std::runtime_error("invalid sweep job line: " + line);
        }
        SweepJob j;
        j.opt = base.opt;
        j.opt.subbench = subbench;
        j.opt.compute_scale = cs;
        j.opt.serial_compute_scale = scs;
        j.opt.comm_repeat = rep;
        if (p2p_tok == "auto") j.p2p_mode = -1;
        else if (p2p_tok == "0" || p2p_tok == "staged") j.p2p_mode = 0;
        else if (p2p_tok == "1" || p2p_tok == "p2p") j.p2p_mode = 1;
        else throw std::runtime_error("invalid p2p mode in sweep job: " + p2p_tok);
        j.check_accuracy = (acc_flag != 0) && (subbench == "full");
        j.report_path = rep_path;
        jobs.push_back(std::move(j));
    }
    if (jobs.empty()) throw std::runtime_error("sweep jobs file has zero entries");
    return jobs;
}

std::string json_escape(const std::string &s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

// Persistent OS thread per GPU so SubmissionOwner, RMM pool, and CUDA stream belong to the same thread across setup/exec/decrypt/teardown.
class PersistentWorkerThread {
    std::mutex mu_;
    std::condition_variable cv_;
    std::function<void()> task_;
    bool has_task_ = false;
    bool done_ = true;
    bool stop_ = false;
    std::exception_ptr err_;
    std::thread thread_;

public:
    PersistentWorkerThread() {
        thread_ = std::thread([this]() {
            while (true) {
                std::function<void()> fn;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait(lk, [this]() { return stop_ || has_task_; });
                    if (stop_) return;
                    fn = std::move(task_);
                    has_task_ = false;
                }
                std::exception_ptr e;
                try {
                    fn();
                } catch (...) {
                    e = std::current_exception();
                }
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    err_ = e;
                    done_ = true;
                }
                cv_.notify_all();
            }
        });
    }

    ~PersistentWorkerThread() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    PersistentWorkerThread(const PersistentWorkerThread &) = delete;
    PersistentWorkerThread &operator=(const PersistentWorkerThread &) = delete;

    void start(std::function<void()> fn) {
        std::lock_guard<std::mutex> lk(mu_);
        task_ = std::move(fn);
        has_task_ = true;
        done_ = false;
        err_ = nullptr;
        cv_.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]() { return done_; });
        if (err_) std::rethrow_exception(err_);
    }
};

template <class Fn>
void run_on_workers(std::array<std::unique_ptr<PersistentWorkerThread>, 2> &pool, int gpus, Fn &&fn) {
    for (int r = 0; r < gpus; ++r) {
        pool[r]->start([&, r]() { fn(r); });
    }
    for (int r = 0; r < gpus; ++r) {
        pool[r]->wait();
    }
}

std::string format_job_json(
    int gpus, const std::string &route, const std::string &plan_name,
    const cerium_source::dual::RunOptions &opt, std::size_t layers,
    bool check_accuracy, bool pass, const cerium_source::ComplexReferenceSummary &acc,
    double setup_s, double weight_load_s, double exec_wall_s, double decode_s, double whole_s,
    const std::array<cerium_source::dual::WorkerResult, 2> &results) {
    std::ostringstream json;
    json << std::setprecision(17) << "{\n"
         << "  \"status\": \"completed_dual_gpu_manual\",\n"
         << "  \"gpus\": " << gpus << ",\n"
         << "  \"route\": \"" << json_escape(route) << "\",\n"
         << "  \"plan\": \"" << json_escape(plan_name) << "\",\n"
         << "  \"subbench\": \"" << json_escape(opt.subbench) << "\",\n"
         << "  \"compute_scale\": " << opt.compute_scale << ",\n"
         << "  \"serial_compute_scale\": " << (opt.serial_compute_scale > 0 ? opt.serial_compute_scale : opt.compute_scale) << ",\n"
         << "  \"comm_repeat\": " << opt.comm_repeat << ",\n"
         << "  \"blocks\": " << layers << ",\n"
         << "  \"accuracy_checked\": " << (check_accuracy ? "true" : "false") << ",\n"
         << "  \"precision_passed\": " << (pass ? "true" : "false") << ",\n"
         << "  \"maximum_complex_error\": " << acc.maximum_complex_error << ",\n"
         << "  \"maximum_real_error\": " << acc.maximum_real_error << ",\n"
         << "  \"maximum_imaginary\": " << acc.maximum_imaginary << ",\n"
         << "  \"real_RMSE\": " << acc.rmse << ",\n"
         << "  \"setup_seconds\": " << setup_s << ",\n"
         << "  \"weight_load_seconds\": " << weight_load_s << ",\n"
         << "  \"encrypted_execution_wall_seconds\": " << exec_wall_s << ",\n"
         << "  \"final_decode_seconds\": " << decode_s << ",\n"
         << "  \"whole_seconds\": " << whole_s << ",\n"
         << "  \"workers\": [\n";
    for (int r = 0; r < gpus; ++r) {
        const auto &wr = results[r];
        json << "    {\n"
             << "      \"rank\": " << r << ",\n"
             << "      \"exec_wall_seconds\": " << wr.exec_wall_s << ",\n"
             << "      \"compute_sum_ms\": " << wr.compute_sum_ms << ",\n"
             << "      \"serial_sum_ms\": " << wr.serial_sum_ms << ",\n"
             << "      \"comm_dma_sum_ms\": " << wr.comm_dma_sum_ms << ",\n"
             << "      \"comm_wall_sum_ms\": " << wr.comm_wall_sum_ms << ",\n"
             << "      \"comm_bytes_rx\": " << wr.comm_bytes_rx << ",\n"
             << "      \"comm_messages\": " << wr.comm_messages << ",\n"
             << "      \"primary_boots\": " << wr.primary_boots << ",\n"
             << "      \"silu_boots\": " << wr.silu_boots << ",\n"
             << "      \"rmm_live_peak_bytes\": " << wr.rmm_live_peak_bytes << ",\n"
             << "      \"stages\": [";
        for (std::size_t i = 0; i < wr.stages.size(); ++i) {
            const auto &st = wr.stages[i];
            if (i) json << ",";
            json << "{\"name\":\"" << json_escape(st.name) << "\",\"gpu_ms\":" << st.gpu_ms << ",\"wall_ms\":" << st.wall_ms << "}";
        }
        json << "],\n      \"comms\": [";
        for (std::size_t i = 0; i < wr.comms.size(); ++i) {
            const auto &cm = wr.comms[i];
            if (i) json << ",";
            json << "{\"stage\":\"" << json_escape(cm.stage) << "\",\"src\":" << cm.src_device
                 << ",\"dst\":" << cm.dst_device << ",\"count\":" << cm.object_count
                 << ",\"repeat\":" << cm.repeat << ",\"bytes_per_rep\":" << cm.logical_bytes_per_repeat
                 << ",\"total_bytes\":" << cm.total_bytes << ",\"msgs\":" << cm.wire_messages
                 << ",\"dma_ms\":" << cm.dma_ms << ",\"wall_ms\":" << cm.wall_ms << "}";
        }
        json << "]\n    }" << (r + 1 < gpus ? "," : "") << "\n";
    }
    json << "  ]\n}\n";
    return json.str();
}

} // namespace

int main(int argc, char **argv) {
    try {
        using namespace cerium_source;
        using namespace cerium_source::dual;
        using namespace poseidon::gpu;
        std::cout << std::setprecision(17);

        const auto cli = parse_cli(argc, argv);
        if (cli.opt.keep_linear_workspace) ::setenv("LLAMA_LINEAR_KEEP_WORKSPACE", "1", 1);
        ::setenv("LLAMA_SCALAR_CACHE", "1", 0);
        ::setenv("LLAMA_ADD_PREFIX", "1", 0);
        ::setenv("POSEIDON_NTT_ALGO", "fourstep", 1);
        ::setenv("POSEIDON_GPU_SKIP_MONTGOMERY_TABLES", "1", 0);
        ::setenv("POSEIDON_GPU_SSE_LOW_P", "2", 1);
        ::setenv("POSEIDON_RESCALE_X2", "1", 1);
        ::setenv("POSEIDON_MANUAL_BOOT_RANGE_EXPERIMENT", "1", 1);

        std::vector<SweepJob> jobs;
        const bool is_sweep = !cli.sweep_jobs_path.empty();
        if (is_sweep) {
            jobs = load_sweep_jobs(cli.sweep_jobs_path, cli);
        } else {
            SweepJob single;
            single.opt = cli.opt;
            single.check_accuracy = cli.check_accuracy;
            single.report_path = cli.report;
            jobs.push_back(std::move(single));
        }

        ExclusiveReport main_report(cli.report);
        const auto whole0 = Clock::now();

        const auto literal = parameters_from_fixture(cli.fixture.c_str());
        require_cerium_parameters(literal);
        const std::filesystem::path weights_path(cli.weights);
        const std::filesystem::path weights_root = std::filesystem::is_directory(weights_path)
            ? weights_path : weights_path.parent_path();
        llama_manual::SafetensorWeights weights_reader(weights_root, cli.index);
        weights_reader.validate_model_index(cli.layers);
        const auto embeddings = llama_manual::benchmark::read_tensor(cli.embeddings, "embeddings", 0);
        const auto reference = llama_manual::benchmark::read_tensor(cli.reference, "hidden", cli.layers);
        if (embeddings.size() != 128 * 4096 || reference.size() != 128 * 4096)
            throw std::runtime_error("tensor geometry mismatch");

        std::array<std::unique_ptr<PersistentWorkerThread>, 2> pool;
        for (int r = 0; r < cli.gpus; ++r) {
            pool[r] = std::make_unique<PersistentWorkerThread>();
        }

        // 1. Setup: Host KeyGen/DFT ONCE (or from opt-in disk cache) + parallel GPU init on persistent threads.
        const auto setup0 = Clock::now();
        const bool is_fast = (cli.mode == manual::Mode::Fast);
        const auto rot_steps = source_rotations();
        HostRuntimeKeys host_keys = HostRuntimeKeys::create(literal, rot_steps, cli.dev_key_cache);
        std::unique_ptr<fast_boot::HostFastBootBundle> fast_bundle;
        if (is_fast && fast_boot::fast_p_enabled()) {
            poseidon::KeyGenerator setup_kg(host_keys.context, host_keys.secret_key);
            poseidon::CKKSEncoder setup_enc(host_keys.context);
            fast_bundle = std::make_unique<fast_boot::HostFastBootBundle>(
                fast_boot::prepare_host_fast_boot_bundle(
                    host_keys.context, setup_kg, setup_enc, rot_steps, cli.public_cache, cli.dev_key_cache));
        }

        // Optional checkpoint load: validate before uploading weights so we skip layers < start_layer.
        std::size_t start_layer = cli.start_layer;
        const std::size_t end_layer = cli.end_layer;
        std::vector<poseidon::Ciphertext> ckpt_cpu_tiles;
        if (!cli.ckpt_load.empty()) {
            checkpoint::Metadata meta{};
            ckpt_cpu_tiles = checkpoint::load_cpu_tiles(
                cli.ckpt_load, host_keys.context, host_keys.secret_key, meta);
            if (ckpt_cpu_tiles.size() != 16)
                throw std::runtime_error("dual checkpoint tile count must be 16");
            if (cli.start_layer_explicit && start_layer != meta.completed_layers)
                throw std::invalid_argument("--start-layer disagrees with checkpoint completed_layers");
            start_layer = static_cast<std::size_t>(meta.completed_layers);
            if (start_layer > end_layer)
                throw std::invalid_argument("checkpoint completed_layers exceeds --end-layer");
            std::cout << "DUAL_LLAMA_CKPT_LOADED path=" << cli.ckpt_load
                      << " start_layer=" << start_layer << " end_layer=" << end_layer << std::endl;
        } else if (start_layer != 0) {
            throw std::invalid_argument("--start-layer > 0 requires --ckpt-load");
        }

        std::unique_ptr<DualPeerTransfer> peer_comm;
        if (cli.gpus == 2) {
            peer_comm = std::make_unique<DualPeerTransfer>(32);
        }

        std::array<std::unique_ptr<GpuWorkerContext>, 2> workers;
        run_on_workers(pool, cli.gpus, [&](int r) {
            workers[r] = std::make_unique<GpuWorkerContext>(
                r, cli.gpus, r, host_keys, fast_bundle.get(), cli.opt.linear_cache_bytes, is_fast);
        });

        const std::size_t local_tiles = 16 / static_cast<std::size_t>(cli.gpus);
        const auto mask_vec = layout::causal_mask(cli.valid_tokens);
        std::array<CT, 2> worker_masks;
        run_on_workers(pool, cli.gpus, [&](int r) {
            auto &w = *workers[r];
            gpu_check_cuda(cudaSetDevice(w.device_id), "encrypt mask dev");
            ScopedGpuExecutionStream sc_s(w.stream);
            ScopedGpuMemoryResource sc_m(w.memory.get());
            worker_masks[r] = w.runtime->encrypt(mask_vec, 27);
            gpu_check_cuda(cudaStreamSynchronize(w.stream), "mask ready");
        });
        const double setup_s = seconds(setup0);

        // 2. Load & upload per-layer BF16 shards cleanly onto persistent worker threads
        //    (only for layers in [start_layer, end_layer)).
        const auto load0 = Clock::now();
        std::array<std::vector<FastLinearEngine::GpuBf16LayerShard>, 2> gpu_weights;
        for (int r = 0; r < cli.gpus; ++r) gpu_weights[r].resize(cli.layers);
        for (std::size_t l = start_layer; l < end_layer; ++l) {
            std::array<llama_manual::LayerWeightsBf16Shard, 2> host_shards;
            for (int r = 0; r < cli.gpus; ++r) {
                host_shards[r] = weights_reader.load_layer_bf16_shard(l, r, cli.gpus);
            }
            std::array<FastLinearEngine::GpuBf16LayerShard, 2> uploaded;
            run_on_workers(pool, cli.gpus, [&](int r) {
                auto &w = *workers[r];
                gpu_check_cuda(cudaSetDevice(w.device_id), "upload weight dev");
                ScopedGpuExecutionStream sc_s(w.stream);
                ScopedGpuMemoryResource sc_m(w.memory.get());
                uploaded[r] = FastLinearEngine::upload_bf16_layer_shard(host_shards[r], w.device_id);
                gpu_check_cuda(cudaStreamSynchronize(w.stream), "upload weight sync");
            });
            for (int r = 0; r < cli.gpus; ++r) gpu_weights[r][l] = std::move(uploaded[r]);
        }
        const double weight_load_s = seconds(load0);

        const bool use_1boot_silu = is_fast && silu_1boot_enabled();
        const std::string plan_name = manual::Plan(cli.mode).name();
        bool overall_pass = true;
        std::ostringstream sweep_index_json;
        if (is_sweep) {
            sweep_index_json << std::setprecision(17)
                             << "{\n  \"status\": \"completed_dual_sweep\",\n"
                             << "  \"gpus\": " << cli.gpus << ",\n"
                             << "  \"setup_seconds\": " << setup_s << ",\n"
                             << "  \"weight_load_seconds\": " << weight_load_s << ",\n"
                             << "  \"runs\": [\n";
        }

        for (std::size_t jidx = 0; jidx < jobs.size(); ++jidx) {
            const auto &job = jobs[jidx];
            if (peer_comm && job.p2p_mode >= 0) {
                peer_comm->set_p2p(job.p2p_mode == 1);
            }
            const std::string route = peer_comm ? peer_comm->route_name() : "single_gpu";

            std::cout << "DUAL_LLAMA_BEGIN gpus=" << cli.gpus
                      << " route=" << route
                      << " subbench=" << job.opt.subbench
                      << " compute_scale=" << job.opt.compute_scale
                      << " serial_compute_scale=" << (job.opt.serial_compute_scale > 0 ? job.opt.serial_compute_scale : job.opt.compute_scale)
                      << " comm_repeat=" << job.opt.comm_repeat
                      << " layers=" << cli.layers
                      << " setup_seconds=" << setup_s
                      << " weight_load_seconds=" << weight_load_s << std::endl;

            std::array<WorkerResult, 2> results;
            std::array<std::vector<CT>, 2> cur_shards;
            run_on_workers(pool, cli.gpus, [&](int r) {
                auto &w = *workers[r];
                gpu_check_cuda(cudaSetDevice(w.device_id), "job encrypt dev");
                ScopedGpuExecutionStream sc_s(w.stream);
                ScopedGpuMemoryResource sc_m(w.memory.get());
                w.linear->fast_engine().release_input();
                cur_shards[r].reserve(local_tiles);
                const std::size_t offset = static_cast<std::size_t>(r) * local_tiles;
                if (!ckpt_cpu_tiles.empty()) {
                    for (std::size_t lt = 0; lt < local_tiles; ++lt) {
                        cur_shards[r].push_back(GpuUploader::upload_ciphertext(ckpt_cpu_tiles[offset + lt], w.device_id));
                    }
                } else {
                    for (std::size_t lt = 0; lt < local_tiles; ++lt) {
                        cur_shards[r].push_back(w.runtime->encrypt(layout::pack_tile(embeddings, 4096, offset + lt), 6));
                    }
                }
                gpu_check_cuda(cudaStreamSynchronize(w.stream), "job encrypt sync");
            });

            const auto exec0 = Clock::now();
            for (std::size_t l = start_layer; l < end_layer; ++l) {
                run_on_workers(pool, cli.gpus, [&](int r) {
                    auto &w = *workers[r];
                    WorkerResult layer_res;
                    cur_shards[r] = run_worker_block(w, peer_comm.get(), gpu_weights[r][l], std::move(cur_shards[r]),
                                                     worker_masks[r], l, job.opt, use_1boot_silu, layer_res);
                    if (l == start_layer) results[r] = std::move(layer_res);
                    else {
                        results[r].exec_wall_s += layer_res.exec_wall_s;
                        results[r].compute_sum_ms += layer_res.compute_sum_ms;
                        results[r].serial_sum_ms += layer_res.serial_sum_ms;
                        results[r].comm_dma_sum_ms += layer_res.comm_dma_sum_ms;
                        results[r].comm_wall_sum_ms += layer_res.comm_wall_sum_ms;
                        results[r].comm_bytes_rx += layer_res.comm_bytes_rx;
                        results[r].comm_messages += layer_res.comm_messages;
                        results[r].primary_boots += layer_res.primary_boots;
                        results[r].silu_boots += layer_res.silu_boots;
                        results[r].rmm_live_peak_bytes = std::max(results[r].rmm_live_peak_bytes, layer_res.rmm_live_peak_bytes);
                        results[r].stages.insert(results[r].stages.end(), layer_res.stages.begin(), layer_res.stages.end());
                        results[r].comms.insert(results[r].comms.end(), layer_res.comms.begin(), layer_res.comms.end());
                    }
                });
                if (!cli.ckpt_save_dir.empty() && job.opt.subbench == "full" &&
                    (((l + 1) % cli.ckpt_save_every == 0) || (l + 1 == end_layer))) {
                    const auto t_sv = Clock::now();
                    std::vector<poseidon::Ciphertext> global_tiles(16);
                    run_on_workers(pool, cli.gpus, [&](int r) {
                        auto &w = *workers[r];
                        gpu_check_cuda(cudaSetDevice(w.device_id), "ckpt download dev");
                        ScopedGpuExecutionStream sc_s(w.stream);
                        ScopedGpuMemoryResource sc_m(w.memory.get());
                        const std::size_t offset = static_cast<std::size_t>(r) * local_tiles;
                        for (std::size_t lt = 0; lt < local_tiles; ++lt) {
                            GpuUploader::download_ciphertext(cur_shards[r][lt], global_tiles[offset + lt], w.runtime->context);
                        }
                        gpu_check_cuda(cudaStreamSynchronize(w.stream), "ckpt download sync");
                    });
                    const auto ck_path = checkpoint::layer_checkpoint_path(cli.ckpt_save_dir, l + 1);
                    checkpoint::save_cpu_tiles(ck_path, host_keys.context, host_keys.secret_key,
                                               global_tiles, l + 1, cli.valid_tokens);
                    std::cout << "DUAL_LLAMA_CKPT_SAVED path=" << ck_path.string()
                              << " completed_layers=" << (l + 1)
                              << " seconds=" << seconds(t_sv) << std::endl;
                }
            }
            auto &final_shards = cur_shards;
            const double exec_wall_s = seconds(exec0);

            const auto dec0 = Clock::now();
            ComplexReferenceSummary acc{};
            const bool do_accuracy = job.check_accuracy && (end_layer == cli.layers);
            bool pass = true;
            if (do_accuracy) {
                ComplexReferenceMetric metric;
                std::array<std::vector<std::complex<double>>, 2> decoded_tiles;
                run_on_workers(pool, cli.gpus, [&](int r) {
                    auto &w = *workers[r];
                    gpu_check_cuda(cudaSetDevice(w.device_id), "decrypt dev");
                    ScopedGpuExecutionStream sc_s(w.stream);
                    ScopedGpuMemoryResource sc_m(w.memory.get());
                    decoded_tiles[r].resize(local_tiles * 32768);
                    for (std::size_t lt = 0; lt < local_tiles; ++lt) {
                        const auto vals = w.runtime->decrypt_final(final_shards[r][lt]);
                        if (vals.size() != 32768) throw std::runtime_error("unexpected slot count at decrypt");
                        std::copy(vals.begin(), vals.end(), decoded_tiles[r].begin() + lt * 32768);
                    }
                });
                for (int r = 0; r < cli.gpus; ++r) {
                    const std::size_t tile_offset = static_cast<std::size_t>(r) * local_tiles;
                    for (std::size_t lt = 0; lt < local_tiles; ++lt) {
                        const auto *vals = decoded_tiles[r].data() + lt * 32768;
                        for (std::size_t token = 0; token < 128; ++token) {
                            for (std::size_t feature = 0; feature < 256; ++feature) {
                                metric.observe(vals[token * 256 + feature],
                                               reference[token * 4096 + (tile_offset + lt) * 256 + feature],
                                               token < cli.valid_tokens);
                            }
                        }
                    }
                }
                acc = metric.finish();
                pass = acc.passed();
            }
            const double decode_s = seconds(dec0);
            const double whole_s = seconds(whole0);

            // Release job output tensors on each worker's thread before next sweep iteration.
            run_on_workers(pool, cli.gpus, [&](int r) {
                auto &w = *workers[r];
                gpu_check_cuda(cudaSetDevice(w.device_id), "release job outputs");
                ScopedGpuExecutionStream sc_s(w.stream);
                ScopedGpuMemoryResource sc_m(w.memory.get());
                w.linear->fast_engine().release_input();
                final_shards[r].clear();
                gpu_check_cuda(cudaStreamSynchronize(w.stream), "release sync");
            });

            const std::string job_json = format_job_json(
                cli.gpus, route, plan_name, job.opt, cli.layers,
                do_accuracy, pass, acc,
                setup_s, weight_load_s, exec_wall_s, decode_s, whole_s, results);
            if (is_sweep) {
                ExclusiveReport rep(job.report_path);
                rep.finish(job_json);
                sweep_index_json << "    {\"index\":" << jidx
                                 << ",\"route\":\"" << json_escape(route)
                                 << "\",\"subbench\":\"" << json_escape(job.opt.subbench)
                                 << "\",\"compute_scale\":" << job.opt.compute_scale
                                 << ",\"serial_compute_scale\":" << (job.opt.serial_compute_scale > 0 ? job.opt.serial_compute_scale : job.opt.compute_scale)
                                 << ",\"comm_repeat\":" << job.opt.comm_repeat
                                 << ",\"exec_wall_seconds\":" << exec_wall_s
                                 << ",\"precision_passed\":" << (pass ? "true" : "false")
                                 << ",\"report\":\"" << json_escape(job.report_path) << "\"}"
                                 << (jidx + 1 < jobs.size() ? "," : "") << "\n";
            } else {
                main_report.finish(job_json);
            }
            if (!pass) overall_pass = false;

            std::cout << "DUAL_LLAMA_DONE gpus=" << cli.gpus
                      << " route=" << route
                      << " subbench=" << job.opt.subbench
                      << " cs=" << job.opt.compute_scale
                      << " scs=" << (job.opt.serial_compute_scale > 0 ? job.opt.serial_compute_scale : job.opt.compute_scale)
                      << " repeat=" << job.opt.comm_repeat
                      << " exec_wall_s=" << exec_wall_s
                      << " weight_load_s=" << weight_load_s
                      << " precision_pass=" << pass
                      << " max_complex_error=" << acc.maximum_complex_error << std::endl;
        }

        if (is_sweep) {
            sweep_index_json << "  ]\n}\n";
            main_report.finish(sweep_index_json.str());
        }

        // Teardown GPU resources cleanly on the exact same persistent worker threads.
        run_on_workers(pool, cli.gpus, [&](int r) {
            auto &w = *workers[r];
            gpu_check_cuda(cudaSetDevice(w.device_id), "cleanup dev");
            cudaDeviceSynchronize();
            {
                ScopedGpuExecutionStream sc_s(w.stream);
                ScopedGpuMemoryResource sc_m(w.memory.get());
                worker_masks[r] = CT();
                gpu_weights[r].clear();
            }
            workers[r].reset();
        });
        peer_comm.reset();
        return overall_pass ? 0 : 2;
    } catch (const std::exception &e) {
        std::cerr << "DUAL_LLAMA_FAILED " << e.what() << std::endl;
        return 1;
    }
}
