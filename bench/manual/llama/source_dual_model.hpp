#pragma once
// Dual-GPU handwritten LLaMA-3.1-8B block execution + single/dual parity,
// BF16 GPU weight decoding, compute/comm ratio scaling, and per-stage profiling.
#include "dual_peer_transfer.hpp"
#include "source_fast_boot.hpp"
#include "source_fast_linear.hpp"
#include "source_linear.hpp"
#include "source_attention.hpp"
#include "source_rms.hpp"
#include "source_softmax.hpp"
#include "source_paired.hpp"
#include "source_manual_pair_silu.hpp"
#include "source_manual_plan.hpp"
#include "source_manual_boot.hpp"
#include "source_accuracy.hpp"
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace cerium_source::dual {

struct StageTiming {
    std::string name;
    int rank = 0;
    double gpu_ms = 0.0;
    double wall_ms = 0.0;
};

struct RunOptions {
    int gpus = 2;
    int compute_scale = 1;
    int serial_compute_scale = 0; // 0 => use compute_scale
    std::size_t comm_repeat = 1;
    std::string subbench = "full"; // full | ffn | ffn_linear | attn_linear | comm_probe
    bool stage_timing = true;
    bool keep_linear_workspace = true;
    std::size_t linear_cache_bytes = 2ULL << 30;
};

struct WorkerResult {
    double exec_wall_s = 0.0;
    double weight_upload_s = 0.0;
    double compute_sum_ms = 0.0;
    double serial_sum_ms = 0.0;
    double comm_dma_sum_ms = 0.0;
    double comm_wall_sum_ms = 0.0;
    std::size_t comm_bytes_rx = 0;
    std::size_t comm_messages = 0;
    std::uint64_t primary_boots = 0;
    std::uint64_t silu_boots = 0;
    std::uint64_t rmm_live_peak_bytes = 0;
    std::vector<StageTiming> stages;
    std::vector<CommRecord> comms;
};

struct DualRunSummary {
    double setup_s = 0.0;
    double weight_load_s = 0.0;
    double exec_wall_s = 0.0;
    double decode_s = 0.0;
    double whole_s = 0.0;
    ComplexReferenceSummary accuracy;
    bool precision_passed = false;
    std::string route;
    std::array<WorkerResult, 2> workers{};
};

namespace detail {

template <class Fn>
auto run_scaled(int scale, Fn &&fn) {
    if (scale < 1) throw std::invalid_argument("compute_scale must be >= 1");
    decltype(fn()) out{};
    for (int i = 0; i < scale; ++i) {
        if (i + 1 < scale) {
            auto sink = fn();
            (void)sink;
        } else {
            out = fn();
        }
    }
    return out;
}

class StageTimer {
    cudaStream_t stream_;
    bool enabled_;
    cudaEvent_t ev0_{nullptr}, ev1_{nullptr};
    std::chrono::steady_clock::time_point t0_;
public:
    StageTimer(cudaStream_t stream, bool enabled, int dev) : stream_(stream), enabled_(enabled) {
        using poseidon::gpu::gpu_check_cuda;
        if (enabled_) {
            gpu_check_cuda(cudaSetDevice(dev), "StageTimer dev");
            gpu_check_cuda(cudaEventCreate(&ev0_), "StageTimer ev0");
            gpu_check_cuda(cudaEventCreate(&ev1_), "StageTimer ev1");
        }
    }
    ~StageTimer() {
        if (ev0_) cudaEventDestroy(ev0_);
        if (ev1_) cudaEventDestroy(ev1_);
    }
    void begin() {
        using poseidon::gpu::gpu_check_cuda;
        if (enabled_) {
            gpu_check_cuda(cudaEventRecord(ev0_, stream_), "StageTimer rec0");
        }
        t0_ = std::chrono::steady_clock::now();
    }
    void end(const std::string &name, int rank, std::vector<StageTiming> &out) {
        using poseidon::gpu::gpu_check_cuda;
        double gpu_ms = 0.0;
        if (enabled_) {
            gpu_check_cuda(cudaEventRecord(ev1_, stream_), "StageTimer rec1");
            gpu_check_cuda(cudaEventSynchronize(ev1_), "StageTimer sync1");
            float fms = 0.0f;
            gpu_check_cuda(cudaEventElapsedTime(&fms, ev0_, ev1_), "StageTimer elapsed");
            gpu_ms = static_cast<double>(fms);
        }
        const double wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0_).count();
        out.push_back(StageTiming{name, rank, gpu_ms, wall_ms});
    }
};

// RMSNorm Stage A (local tiles): boot_pair + cp_to(2^60) + square + local partial sum
// + 8 downward rotations at Q27 (scalar slot 0 accumulates all 8 local tiles' sums).
inline std::pair<std::vector<CT>, CT> rms_local_part(
    Api &api, const std::vector<CT> &local_input, std::size_t shard_tile_offset,
    std::size_t layer, double epsilon, bool add_eps) {
    const std::size_t n = local_input.size();
    std::vector<CT> x;
    x.reserve(n);
    for (std::size_t i = 0; i < n; i += 2) {
        auto r = boot_pair(api, local_input[i], local_input[i + 1]);
        auto p = api.cp_to_pair(r[0], r[1], 1., std::exp2(60.), 0);
        x.push_back(std::move(p[0]));
        x.push_back(std::move(p[1]));
    }
    std::vector<CT> sq;
    sq.reserve(n);
    for (std::size_t base = 0; base < n; base += 8) {
        const std::size_t chunk = std::min<std::size_t>(8, n - base);
        std::vector<const CT *> ptrs(chunk);
        for (std::size_t k = 0; k < chunk; ++k) ptrs[k] = &x[base + k];
        auto prod = api.mul_batch(ptrs, ptrs);
        for (auto &p : prod) sq.push_back(std::move(p));
    }
    std::optional<CT> acc;
    if (sq.size() == 16) {
        auto s8 = api.add_batch(
            {&sq[0], &sq[1], &sq[2], &sq[3], &sq[4], &sq[5], &sq[6], &sq[7]},
            {&sq[8], &sq[9], &sq[10], &sq[11], &sq[12], &sq[13], &sq[14], &sq[15]});
        auto s4 = api.add_batch({&s8[0], &s8[1], &s8[2], &s8[3]}, {&s8[4], &s8[5], &s8[6], &s8[7]});
        auto s2 = api.add_pair(s4[0], s4[2], s4[1], s4[3]);
        acc = api.relin(api.add(s2[0], s2[1]));
    } else if (sq.size() == 8) {
        auto s4 = api.add_batch({&sq[0], &sq[1], &sq[2], &sq[3]}, {&sq[4], &sq[5], &sq[6], &sq[7]});
        auto s2 = api.add_pair(s4[0], s4[2], s4[1], s4[3]);
        acc = api.relin(api.add(s2[0], s2[1]));
    } else {
        CT s = std::move(sq[0]);
        for (std::size_t i = 1; i < sq.size(); ++i) s = api.add(s, sq[i]);
        acc = api.relin(s);
    }
    CT sum = std::move(*acc);
    for (int i = 0; i < 8; ++i) sum = api.add(sum, api.rotate(sum, -(1 << i)));
    if (add_eps) sum = api.add_plain(sum, 4096. * epsilon);
    (void)shard_tile_offset;
    (void)layer;
    return {std::move(x), std::move(sum)};
}

// RMSNorm Stage B (serial root): sum -> mean -> inverse_sqrt
inline CT rms_inverse_serial(Api &api, const CT &sum, std::size_t layer) {
    std::vector<double> selector(32768, 0.0);
    for (std::size_t i = 0; i < 32768; ++i) {
        const double divisor = (layer > 28 || (layer >= 2 && i < 256)) ? 2. : .01;
        if (i % 256 == 255) selector[i] = 1. / (4096. * divisor);
    }
    auto mean = api.cp_vector(sum, selector, std::exp2(60.) * api.removed(sum.meta.q_count, 4) / sum.meta.scale);
    for (int i = 0; i < 8; ++i) mean = api.add(mean, api.rotate(mean, 1 << i));
    mean = api.rescale(mean, 4);
    const bool fast_rms = fast_rms_enabled();
    return inverse_sqrt_rms_schedule(api, mean, fast_rms);
}

// RMSNorm Stage C (local tiles): apply gamma * inverse_divisor * inverse * x[tile]
inline std::vector<CT> rms_scale_local(
    Api &api, const std::vector<CT> &x_local, const CT &inverse,
    const std::vector<double> &gamma, std::size_t shard_tile_offset, std::size_t layer) {
    const std::size_t n = x_local.size();
    std::vector<double> inverse_divisor(32768);
    for (std::size_t i = 0; i < 32768; ++i) {
        const double divisor = (layer > 28 || (layer >= 2 && i < 256)) ? 2. : .01;
        inverse_divisor[i] = 1. / std::sqrt(divisor);
    }
    const bool fast_rms = fast_rms_enabled();
    const double target_out = fast_rms ? std::exp2(40.) : std::exp2(60.);
    std::vector<CT> relins;
    relins.reserve(n);
    std::vector<double> weights(32768);
    for (std::size_t i = 0; i < n; i += 2) {
        std::array<CT, 2> weighted_pair;
        for (std::size_t l = 0; l < 2; ++l) {
            const std::size_t global_tile = shard_tile_offset + i + l;
            for (std::size_t j = 0; j < 32768; ++j)
                weights[j] = gamma[global_tile * 256 + j % 256] * inverse_divisor[j];
            const auto q = std::min(inverse.meta.q_count, x_local[i + l].meta.q_count);
            const double weight_scale = target_out * api.removed(q, 3) / (inverse.meta.scale * x_local[i + l].meta.scale);
            weighted_pair[l] = api.cp_vector(inverse, weights, weight_scale);
        }
        auto relin = api.mul_r_pair(weighted_pair[0], x_local[i], weighted_pair[1], x_local[i + 1]);
        relins.push_back(std::move(relin[0]));
        relins.push_back(std::move(relin[1]));
    }
    std::vector<CT> out;
    out.reserve(n);
    for (std::size_t base = 0; base < n; base += 8) {
        const std::size_t chunk = std::min<std::size_t>(8, n - base);
        std::vector<CT> batch;
        batch.reserve(chunk);
        for (std::size_t k = 0; k < chunk; ++k) batch.push_back(std::move(relins[base + k]));
        auto rescaled = api.rescale_batch(batch, 3);
        if (fast_rms) {
            for (auto &r : rescaled) out.push_back(std::move(r));
        } else {
            for (std::size_t k = 0; k < chunk; k += 2) {
                auto p = api.cp_to_pair(rescaled[k], rescaled[k + 1], 1., std::exp2(40.), 1);
                out.push_back(std::move(p[0]));
                out.push_back(std::move(p[1]));
            }
        }
    }
    return out;
}

inline std::vector<CT> add_residual_local(Api &api, const std::vector<CT> &a, const std::vector<CT> &b) {
    if (a.size() != b.size() || (a.size() % 2))
        throw std::invalid_argument("add_residual_local shard size mismatch");
    std::vector<CT> out;
    out.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); i += 2) {
        auto p = api.add_pair(a[i], b[i], a[i + 1], b[i + 1]);
        out.push_back(std::move(p[0]));
        out.push_back(std::move(p[1]));
    }
    return out;
}

} // namespace detail

struct GpuWorkerContext {
    int rank = 0;
    int world_size = 1;
    int device_id = 0;
    cudaStream_t stream{nullptr};
    std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>> pool;
    std::unique_ptr<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>> memory;
    rmm::mr::device_memory_resource *old_resource{nullptr};
    std::unique_ptr<Runtime> runtime;
    std::unique_ptr<manual_backend::SubmissionOwner> sub_owner;
    std::unique_ptr<manual_backend::BoundApi> primary;
    std::unique_ptr<manual_backend::BoundApi> silu;
    std::unique_ptr<NativeLinearAdapter> linear;

    GpuWorkerContext(int r, int w, int dev, const HostRuntimeKeys &host_keys,
                     const fast_boot::HostFastBootBundle *fast_bundle,
                     std::size_t linear_cache_bytes, bool is_fast)
        : rank(r), world_size(w), device_id(dev) {
        using poseidon::gpu::gpu_check_cuda;
        gpu_check_cuda(cudaSetDevice(device_id), "worker setDevice");
        gpu_check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "worker stream");
        poseidon::gpu::ScopedGpuExecutionStream scoped_stream(stream);
        old_resource = rmm::mr::get_current_device_resource();
        pool = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>>(
            old_resource, 64ULL << 20, 64ULL << 30);
        memory = std::make_unique<rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>>(pool.get());
        rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_id}, memory.get());
        poseidon::gpu::ScopedGpuMemoryResource scoped_mem(memory.get());

        runtime = std::make_unique<Runtime>(host_keys, device_id);
        sub_owner = std::make_unique<manual_backend::SubmissionOwner>();
        if (is_fast && fast_bundle) {
            fast_boot::install_fast_runtime_levels_from_bundle(*runtime, *fast_bundle);
            auto built = fast_boot::make_fast_profile_pair(*runtime, *sub_owner, fast_bundle);
            primary = std::make_unique<manual_backend::BoundApi>(*runtime, *sub_owner, std::move(built[0]), linear_cache_bytes);
            silu = std::make_unique<manual_backend::BoundApi>(
                *runtime, *sub_owner, std::move(built[1]),
                linear_cache_bytes ? std::min(linear_cache_bytes, std::size_t(512ULL << 20)) : 0);
        } else {
            auto b0 = manual_backend::make_profile(*runtime, 8192, *sub_owner);
            auto b1 = manual_backend::make_profile(*runtime, 65536, *sub_owner);
            primary = std::make_unique<manual_backend::BoundApi>(*runtime, *sub_owner, std::move(b0), linear_cache_bytes);
            silu = std::make_unique<manual_backend::BoundApi>(
                *runtime, *sub_owner, std::move(b1),
                linear_cache_bytes ? std::min(linear_cache_bytes, std::size_t(512ULL << 20)) : 0);
        }
        linear = std::make_unique<NativeLinearAdapter>(primary->get(), runtime->context);
    }

    ~GpuWorkerContext() {
        cudaSetDevice(device_id);
        poseidon::gpu::ScopedGpuExecutionStream scoped_stream(stream);
        poseidon::gpu::ScopedGpuMemoryResource scoped_mem(memory.get());
        linear.reset();
        silu.reset();
        primary.reset();
        runtime.reset();
        if (old_resource) {
            rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_id}, old_resource);
        }
        memory.reset();
        pool.reset();
        if (stream) cudaStreamDestroy(stream);
    }
};

// Executes one full or subgraph LLaMA block on `worker` (rank 0 for 1-GPU; rank 0 & 1 concurrently for 2-GPU).
inline std::vector<CT> run_worker_block(
    GpuWorkerContext &worker,
    DualPeerTransfer *comm,
    const FastLinearEngine::GpuBf16LayerShard &w,
    std::vector<CT> local_input,
    const CT &encrypted_mask,
    std::size_t layer,
    const RunOptions &opt,
    bool use_1boot_silu,
    WorkerResult &stats) {
    using poseidon::gpu::gpu_check_cuda;
    gpu_check_cuda(cudaSetDevice(worker.device_id), "worker block setDevice");
    poseidon::gpu::ScopedGpuExecutionStream scoped_stream(worker.stream);
    poseidon::gpu::ScopedGpuMemoryResource scoped_mem(worker.memory.get());

    const int rank = worker.rank;
    const int world = worker.world_size;
    const int cs = opt.compute_scale;
    const int scs = opt.serial_compute_scale > 0 ? opt.serial_compute_scale : opt.compute_scale;
    const bool is_dual = (world == 2);
    if (is_dual && !comm) throw std::invalid_argument("2-GPU run requires DualPeerTransfer");

    Api &api = worker.primary->get();
    Api &s_api = worker.silu->get();
    NativeLinearAdapter &linear = *worker.linear;
    detail::StageTimer timer(worker.stream, opt.stage_timing, worker.device_id);

    const std::size_t hidden_tiles_local = 16 / static_cast<std::size_t>(world);
    const std::size_t hidden_tile_offset = static_cast<std::size_t>(rank) * hidden_tiles_local;
    const std::size_t q_out_width = 4096 / static_cast<std::size_t>(world);
    const std::size_t kv_out_width = 1024 / static_cast<std::size_t>(world);
    const std::size_t o_out_width = 4096 / static_cast<std::size_t>(world);
    const std::size_t ffn_out_width = 14336 / static_cast<std::size_t>(world);
    const std::size_t kv_groups_local = 4 / static_cast<std::size_t>(world);
    const std::size_t silu_pairs_local = (56 / static_cast<std::size_t>(world)) / 2;

    const auto ab0 = api.boot_calls;
    const auto sb0 = s_api.boot_calls;
    const auto t_block0 = std::chrono::steady_clock::now();

    std::vector<CT> residual = std::move(local_input);
    if (residual.size() != hidden_tiles_local)
        throw std::invalid_argument("worker local_input shard tile count mismatch");

    auto run_rmsnorm_step = [&](const std::string &prefix, const std::vector<CT> &in_shard,
                                const std::vector<double> &gamma) -> std::vector<CT> {
        timer.begin();
        auto local_pair = detail::run_scaled(cs, [&]() {
            return detail::rms_local_part(api, in_shard, hidden_tile_offset, layer, 1e-5, rank == 0);
        });
        timer.end(prefix + "_local_sq", rank, stats.stages);

        CT total_sum = std::move(local_pair.second);
        if (is_dual) {
            std::vector<CT> send_vec;
            if (rank == 1) send_vec.push_back(std::move(total_sum));
            auto rx = comm->send_recv(rank, 1, std::move(send_vec), worker.stream, opt.comm_repeat,
                                      prefix + "_sum_1to0", opt.stage_timing, &stats.comms);
            if (rank == 0) total_sum = api.add(total_sum, rx.at(0));
        }

        CT inverse;
        if (rank == 0) {
            timer.begin();
            inverse = detail::run_scaled(scs, [&]() {
                return detail::rms_inverse_serial(api, total_sum, layer);
            });
            timer.end(prefix + "_inv_sqrt_serial", rank, stats.stages);
        }
        if (is_dual) {
            std::vector<CT> bcast_vec;
            if (rank == 0) bcast_vec.push_back(api.copy(inverse));
            auto rx = comm->send_recv(rank, 0, std::move(bcast_vec), worker.stream, opt.comm_repeat,
                                      prefix + "_inv_0to1", opt.stage_timing, &stats.comms);
            if (rank == 1) inverse = std::move(rx.at(0));
        }

        timer.begin();
        auto local_norm = detail::run_scaled(cs, [&]() {
            return detail::rms_scale_local(api, local_pair.first, inverse, gamma, hidden_tile_offset, layer);
        });
        timer.end(prefix + "_scale_local", rank, stats.stages);

        if (!is_dual) return local_norm;
        return comm->allgather_shards(rank, std::move(local_norm), worker.stream, opt.comm_repeat,
                                      prefix + "_allgather", opt.stage_timing, &stats.comms);
    };

    std::chrono::steady_clock::time_point t_exec_begin = t_block0;
    if (opt.subbench == "full") {
        // 1. Attention RMSNorm -> AllGather normalized (16 tiles on each GPU)
        auto normalized = run_rmsnorm_step("attn_rms", residual, w.attention_norm);

        // 2. Q / K / V projections -> QK -> Softmax -> AV (100% parallel, ZERO comms!)
        timer.begin();
        auto q = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(normalized, 4096, q_out_width, w.q.data(), false, true, "q");
        });
        timer.end("q_proj", rank, stats.stages);

        timer.begin();
        auto k = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(normalized, 4096, kv_out_width, w.k.data(), false, true, "k");
        });
        timer.end("k_proj", rank, stats.stages);

        timer.begin();
        auto v = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(normalized, 4096, kv_out_width, w.v.data(), false, false, "v");
        });
        timer.end("v_proj", rank, stats.stages);
        linear.release_input();
        normalized.clear();

        timer.begin();
        auto scores = detail::run_scaled(cs, [&]() {
            std::vector<CT> out;
            out.reserve(kv_groups_local * 4);
            for (std::size_t i = 0; i < q.size(); i += 2) {
                auto pair = qk_pair(api, {&q[i], &q[i + 1]}, k[i / 4]);
                for (auto &sc : pair) out.push_back(fast_attention_enabled() ? std::move(sc) : api.drop(sc, sc.meta.q_count - 2));
            }
            return out;
        });
        timer.end("qk", rank, stats.stages);

        timer.begin();
        auto probs = detail::run_scaled(cs, [&]() {
            std::vector<CT> out;
            out.reserve(scores.size());
            if (env_flag("LLAMA_PAIR_SOFTMAX")) {
                for (std::size_t i = 0; i < scores.size(); i += 2) {
                    auto pair = softmax128_pair(api, scores[i], scores[i + 1], encrypted_mask);
                    for (auto &p : pair) out.push_back(std::move(p));
                }
            } else {
                for (const auto &sc : scores) out.push_back(softmax128(api, sc, encrypted_mask));
            }
            return out;
        });
        timer.end("softmax", rank, stats.stages);

        timer.begin();
        auto attention_local = detail::run_scaled(cs, [&]() {
            std::vector<CT> out;
            out.reserve(q.size());
            for (std::size_t i = 0; i < q.size(); i += 4) {
                auto grp = av_four(api, {&probs[i], &probs[i + 1], &probs[i + 2], &probs[i + 3]}, v[i / 4]);
                for (auto &val : grp) out.push_back(std::move(val));
            }
            return out;
        });
        timer.end("av", rank, stats.stages);
        q.clear(); k.clear(); v.clear(); scores.clear(); probs.clear();

        // 3. AllGather attention (8 CTs each way @ Q8 = 32 MiB) -> O proj + Attention Residual
        std::vector<CT> attention_all = is_dual
            ? comm->allgather_shards(rank, std::move(attention_local), worker.stream, opt.comm_repeat,
                                     "attn_allgather", opt.stage_timing, &stats.comms)
            : std::move(attention_local);

        timer.begin();
        auto projected = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(attention_all, 4096, o_out_width, w.o.data(), true, false, "o");
        });
        linear.release_input();
        residual = detail::run_scaled(cs, [&]() {
            return detail::add_residual_local(api, residual, projected);
        });
        timer.end("o_proj_res", rank, stats.stages);
        attention_all.clear();
        projected.clear();

        // 4. FFN RMSNorm -> AllGather ffn_norm
        auto ffn_norm = run_rmsnorm_step("ffn_rms", residual, w.ffn_norm);

        // 5. Gate + Up + SwiGLU (100% parallel, ZERO comms!)
        timer.begin();
        auto gate = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(ffn_norm, 4096, ffn_out_width, w.gate.data(), false, false, "gate");
        });
        timer.end("gate_proj", rank, stats.stages);

        timer.begin();
        auto up = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(ffn_norm, 4096, ffn_out_width, w.up.data(), false, false, "up");
        });
        timer.end("up_proj", rank, stats.stages);
        linear.release_input();
        ffn_norm.clear();

        timer.begin();
        auto activated_local = detail::run_scaled(cs, [&]() {
            std::vector<CT> out;
            out.reserve(gate.size());
            for (std::size_t i = 0; i < gate.size(); i += 2) {
                auto x0 = s_api.drop(gate[i], 6);
                auto x1 = s_api.drop(gate[i + 1], 6);
                auto pair = use_1boot_silu
                    ? manual_silu_1boot_pair_fast(s_api, x0, up[i], x1, up[i + 1])
                    : manual_silu_all60_pair_candidate(s_api, x0, up[i], x1, up[i + 1]);
                out.push_back(std::move(pair[0]));
                out.push_back(std::move(pair[1]));
            }
            return out;
        });
        timer.end("swiglu", rank, stats.stages);
        gate.clear(); up.clear();

        // 6. AllGather activated (28 CTs each way @ Q8 = 112 MiB) -> Down proj + FFN Residual
        std::vector<CT> activated_all = is_dual
            ? comm->allgather_shards(rank, std::move(activated_local), worker.stream, opt.comm_repeat,
                                     "ffn_act_allgather", opt.stage_timing, &stats.comms)
            : std::move(activated_local);

        timer.begin();
        auto down = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(activated_all, 14336, o_out_width, w.down.data(), false, false, "down");
        });
        linear.release_input();
        residual = detail::run_scaled(cs, [&]() {
            return detail::add_residual_local(api, residual, down);
        });
        timer.end("down_proj_res", rank, stats.stages);
        (void)silu_pairs_local;
    } else if (opt.subbench == "ffn") {
        auto ffn_norm = run_rmsnorm_step("ffn_rms", residual, w.ffn_norm);
        timer.begin();
        auto gate = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(ffn_norm, 4096, ffn_out_width, w.gate.data(), false, false, "gate");
        });
        timer.end("gate_proj", rank, stats.stages);
        timer.begin();
        auto up = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(ffn_norm, 4096, ffn_out_width, w.up.data(), false, false, "up");
        });
        timer.end("up_proj", rank, stats.stages);
        linear.release_input();
        ffn_norm.clear();
        timer.begin();
        auto activated_local = detail::run_scaled(cs, [&]() {
            std::vector<CT> out;
            out.reserve(gate.size());
            for (std::size_t i = 0; i < gate.size(); i += 2) {
                auto x0 = s_api.drop(gate[i], 6);
                auto x1 = s_api.drop(gate[i + 1], 6);
                auto pair = use_1boot_silu
                    ? manual_silu_1boot_pair_fast(s_api, x0, up[i], x1, up[i + 1])
                    : manual_silu_all60_pair_candidate(s_api, x0, up[i], x1, up[i + 1]);
                out.push_back(std::move(pair[0]));
                out.push_back(std::move(pair[1]));
            }
            return out;
        });
        timer.end("swiglu", rank, stats.stages);
        gate.clear(); up.clear();
        std::vector<CT> activated_all = is_dual
            ? comm->allgather_shards(rank, std::move(activated_local), worker.stream, opt.comm_repeat,
                                     "ffn_act_allgather", opt.stage_timing, &stats.comms)
            : std::move(activated_local);
        timer.begin();
        auto down = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(activated_all, 14336, o_out_width, w.down.data(), false, false, "down");
        });
        linear.release_input();
        residual = detail::run_scaled(cs, [&]() {
            return detail::add_residual_local(api, residual, down);
        });
        timer.end("down_proj_res", rank, stats.stages);
    } else if (opt.subbench == "ffn_linear") {
        // Focused FFN-Linear benchmark (Q14 input -> gate -> up -> Q8 -> AllGather -> down -> Q6)
        // for sweeping compute-to-comm ratios cleanly without 156 Boots.
        std::vector<CT> q14_local;
        q14_local.reserve(hidden_tiles_local);
        for (std::size_t i = 0; i < residual.size(); i += 2) {
            auto r = boot_pair(api, residual[i], residual[i + 1]);
            auto p = api.cp_to_pair(r[0], r[1], 1., std::exp2(40.), 1);
            q14_local.push_back(api.drop(p[0], 14));
            q14_local.push_back(api.drop(p[1], 14));
        }
        std::vector<CT> full_in = is_dual
            ? comm->allgather_shards(rank, std::move(q14_local), worker.stream, 1, "sub_init", false, nullptr)
            : std::move(q14_local);
        gpu_check_cuda(cudaStreamSynchronize(worker.stream), "ffn_linear init sync");
        if (is_dual) comm->barrier();

        t_exec_begin = std::chrono::steady_clock::now();
        timer.begin();
        auto gate = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(full_in, 4096, ffn_out_width, w.gate.data(), false, false, "gate");
        });
        timer.end("sub_gate_proj", rank, stats.stages);

        timer.begin();
        auto up = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(full_in, 4096, ffn_out_width, w.up.data(), false, false, "up");
        });
        timer.end("sub_up_proj", rank, stats.stages);
        linear.release_input();
        full_in.clear();

        timer.begin();
        auto act_local = detail::run_scaled(cs, [&]() {
            std::vector<CT> out;
            out.reserve(gate.size());
            for (std::size_t i = 0; i < gate.size(); i += 2) {
                auto x0 = api.drop(gate[i], 10);
                auto x1 = api.drop(gate[i + 1], 10);
                auto u0 = api.drop(up[i], 10);
                auto u1 = api.drop(up[i + 1], 10);
                auto m = api.mul_rescale_pair(x0, u0, x1, u1, 1); // -> Q9
                auto q8 = api.cp_to_pair(m[0], m[1], 1., std::exp2(40.), 1); // -> Q8, exact 2^40
                out.push_back(std::move(q8[0]));
                out.push_back(std::move(q8[1]));
            }
            return out;
        });
        timer.end("sub_gate_up_mul", rank, stats.stages);
        gate.clear(); up.clear();

        std::vector<CT> act_all = is_dual
            ? comm->allgather_shards(rank, std::move(act_local), worker.stream, opt.comm_repeat,
                                     "sub_act_allgather", opt.stage_timing, &stats.comms)
            : std::move(act_local);

        timer.begin();
        auto down = detail::run_scaled(cs, [&]() {
            return linear.run_bf16_device(act_all, 14336, o_out_width, w.down.data(), false, false, "down");
        });
        linear.release_input();
        residual = detail::run_scaled(cs, [&]() {
            return detail::add_residual_local(api, residual, down);
        });
        timer.end("sub_down_proj_res", rank, stats.stages);
    } else if (opt.subbench == "comm_probe") {
        if (is_dual) {
            std::vector<std::pair<std::string, std::pair<std::size_t, std::size_t>>> sweep = {
                {"q6_1ct",  {1, 6}},
                {"q8_1ct",  {1, 8}},
                {"q8_8ct",  {8, 8}},
                {"q16_8ct", {8, 16}},
                {"q27_1ct", {1, 27}},
                {"q8_28ct", {28, 8}},
            };
            auto make_batch = [&](std::size_t cnt, std::size_t q) {
                std::vector<CT> v;
                v.reserve(cnt);
                for (std::size_t i = 0; i < cnt; ++i) v.push_back(api.drop(encrypted_mask, q));
                return v;
            };
            for (const auto &item : sweep) {
                const auto &tag = item.first;
                const std::size_t cnt = item.second.first;
                const std::size_t q = item.second.second;
                {
                    auto local = make_batch(cnt, q);
                    gpu_check_cuda(cudaStreamSynchronize(worker.stream), "comm probe prep sync");
                    comm->barrier();
                    auto rx_bi = comm->allgather_shards(rank, std::move(local),
                                                        worker.stream, opt.comm_repeat,
                                                        "probe_bidir_" + tag, true, &stats.comms);
                    gpu_check_cuda(cudaStreamSynchronize(worker.stream), "comm probe bidir sync");
                    comm->barrier();
                }
                {
                    auto s0 = (rank == 0) ? make_batch(cnt, q) : std::vector<CT>{};
                    gpu_check_cuda(cudaStreamSynchronize(worker.stream), "comm probe uni01 prep");
                    comm->barrier();
                    auto rx01 = comm->send_recv(rank, 0, std::move(s0), worker.stream, opt.comm_repeat,
                                                "probe_uni01_" + tag, true, &stats.comms);
                    gpu_check_cuda(cudaStreamSynchronize(worker.stream), "comm probe uni01 sync");
                    comm->barrier();
                }
                {
                    auto s1 = (rank == 1) ? make_batch(cnt, q) : std::vector<CT>{};
                    gpu_check_cuda(cudaStreamSynchronize(worker.stream), "comm probe uni10 prep");
                    comm->barrier();
                    auto rx10 = comm->send_recv(rank, 1, std::move(s1), worker.stream, opt.comm_repeat,
                                                "probe_uni10_" + tag, true, &stats.comms);
                    gpu_check_cuda(cudaStreamSynchronize(worker.stream), "comm probe uni10 sync");
                    comm->barrier();
                }
            }
        }
    } else {
        throw std::invalid_argument("unsupported subbench mode: " + opt.subbench);
    }

    gpu_check_cuda(cudaStreamSynchronize(worker.stream), "worker block finish sync");
    stats.exec_wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_exec_begin).count();
    stats.primary_boots = api.boot_calls - ab0;
    stats.silu_boots = s_api.boot_calls - sb0;
    stats.rmm_live_peak_bytes = worker.memory->get_bytes_counter().peak;
    for (const auto &st : stats.stages) {
        const double t_ms = (st.gpu_ms > 0.0) ? st.gpu_ms : st.wall_ms;
        if (st.name.find("_serial") != std::string::npos) stats.serial_sum_ms += t_ms;
        else stats.compute_sum_ms += t_ms;
    }
    for (const auto &cm : stats.comms) {
        stats.comm_dma_sum_ms += cm.dma_ms;
        stats.comm_wall_sum_ms += cm.wall_ms;
        stats.comm_bytes_rx += cm.total_bytes;
        stats.comm_messages += cm.wire_messages;
    }
    return residual;
}

} // namespace cerium_source::dual
