#pragma once
// Bidirectional multi-ciphertext P2P / chunked host-staged transfer engine
// and rendezvous barrier for dual-GPU LLaMA execution.
#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_memory.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace cerium_source::dual {

using CT = poseidon::gpu::GpuCiphertextData;
using Word = poseidon::gpu::GpuWord;

struct CommRecord {
    std::string stage;
    int src_device = 0;
    int dst_device = 0;
    std::size_t object_count = 0;
    std::size_t repeat = 1;
    std::size_t logical_bytes_per_repeat = 0;
    std::size_t total_bytes = 0;
    std::size_t wire_messages = 0;
    double dma_ms = 0.0;
    double wall_ms = 0.0;
};

class RendezvousBarrier {
    std::mutex mu_;
    std::condition_variable cv_;
    int count_ = 0;
    int generation_ = 0;
    const int threads_;
public:
    explicit RendezvousBarrier(int threads = 2) : threads_(threads) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mu_);
        const int gen = generation_;
        if (++count_ == threads_) {
            count_ = 0;
            ++generation_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [&] { return generation_ != gen; });
        }
    }
};

class DualPeerTransfer {
    static bool query_p2p_requested() {
        const char *raw = std::getenv("POSEIDON_PEER_P2P");
        if (!raw || !*raw) raw = std::getenv("POSEIDON_S2C_PEER_P2P");
        if (!raw || !*raw) return true;
        const std::string v(raw);
        if (v != "0" && v != "1")
            throw std::invalid_argument("POSEIDON_PEER_P2P must be 0 or 1");
        return v == "1";
    }

    std::size_t max_q_ = 32;
    std::size_t capacity_words_ = 0;
    std::size_t chunk_words_ = 0;
    bool p2p_supported_ = false;
    bool p2p_enabled_ = false;
    std::array<Word *, 2> staging_{nullptr, nullptr};
    std::array<cudaStream_t, 2> dma_{nullptr, nullptr};
    std::array<cudaStream_t, 2> d2h_{nullptr, nullptr};
    std::array<cudaEvent_t, 2> prod_ev_{nullptr, nullptr};
    std::array<cudaEvent_t, 2> done_ev_{nullptr, nullptr};
    std::array<cudaEvent_t, 2> stage_reuse_ev_{nullptr, nullptr};
    std::array<cudaEvent_t, 2> time_start_{nullptr, nullptr};
    std::array<cudaEvent_t, 2> time_stop_{nullptr, nullptr};
    std::array<std::vector<cudaEvent_t>, 2> chunk_ev_{};
    RendezvousBarrier barrier_{2};

    // Shared rendezvous slots protected by barrier_ phases.
    struct SlotMeta {
        poseidon::gpu::GpuCiphertextMeta meta{};
        const Word *src0 = nullptr;
        const Word *src1 = nullptr;
        bool contiguous = false;
        std::size_t q_words = 0;
    };
    std::array<std::vector<SlotMeta>, 2> shared_meta_{};
    std::array<std::vector<CT>, 2> recv_slot_{};
    bool p2p_sync_after_wait_ = false;

    static void validate_ct(const CT &ct, int expected_dev, std::size_t capacity_words) {
        const auto words = ct.meta.degree * ct.meta.q_count;
        if (ct.meta.degree != 65536 || ct.meta.p_count ||
            ct.meta.component_count != 2 || !words || 2 * words > capacity_words)
            throw std::invalid_argument("DualPeerTransfer: unsupported CT geometry or exceeds max_q");
        const auto view = ct.make_const_view();
        if (view.polys.size() != 2)
            throw std::invalid_argument("DualPeerTransfer: expected 2-component CT");
        for (const auto &poly : view.polys) {
            const auto &shards = poly.shards;
            if (shards.size() != 1 || shards[0].limb_begin || shards[0].coeff_begin ||
                shards[0].limb_count != ct.meta.q_count ||
                shards[0].coeff_count != ct.meta.degree ||
                shards[0].device_id != expected_dev)
                throw std::invalid_argument("DualPeerTransfer: CT layout/device mismatch");
        }
    }

    static SlotMeta describe(const CT &ct, int dev, std::size_t cap_words) {
        validate_ct(ct, dev, cap_words);
        const auto view = ct.make_const_view();
        const std::size_t w = ct.meta.degree * ct.meta.q_count;
        const Word *p0 = view.polys[0].shards[0].ptr;
        const Word *p1 = view.polys[1].shards[0].ptr;
        return SlotMeta{ct.meta, p0, p1, p1 == p0 + w, w};
    }

    void copy_ct_async(const SlotMeta &src_info, int src, CT &dst_ct, int dst,
                       std::size_t &wire_msgs_out) {
        using poseidon::gpu::gpu_check_cuda;
        const std::size_t words = src_info.q_words;
        Word *dst_ptr = dst_ct.fields_[0].data();
        if (p2p_enabled_) {
            gpu_check_cuda(cudaSetDevice(dst), "peer p2p set dst");
            if (src_info.contiguous) {
                gpu_check_cuda(cudaMemcpyPeerAsync(
                    dst_ptr, dst, src_info.src0, src,
                    2 * words * sizeof(Word), dma_[dst]), "peer p2p 2c");
                wire_msgs_out += 1;
            } else {
                gpu_check_cuda(cudaMemcpyPeerAsync(
                    dst_ptr, dst, src_info.src0, src,
                    words * sizeof(Word), dma_[dst]), "peer p2p c0");
                gpu_check_cuda(cudaMemcpyPeerAsync(
                    dst_ptr + words, dst, src_info.src1, src,
                    words * sizeof(Word), dma_[dst]), "peer p2p c1");
                wire_msgs_out += 2;
            }
            return;
        }
        // Chunked host-staged DMA pipeline (d2h_[src] -> staging_[src] -> dma_[dst]).
        gpu_check_cuda(cudaSetDevice(src), "staged d2h wait reuse");
        gpu_check_cuda(cudaStreamWaitEvent(d2h_[src], stage_reuse_ev_[dst], 0), "staged wait reuse");
        Word *stage = staging_[src];
        std::size_t ev_idx = 0;
        const std::size_t step = chunk_words_;
        const Word *src_ptrs[2] = {src_info.src0, src_info.src1};
        for (int c = 0; c < 2; ++c) {
            for (std::size_t off = 0; off < words; off += step) {
                const std::size_t cnt = std::min(step, words - off);
                gpu_check_cuda(cudaSetDevice(src), "staged d2h dev");
                gpu_check_cuda(cudaMemcpyAsync(
                    stage + std::size_t(c) * words + off, src_ptrs[c] + off,
                    cnt * sizeof(Word), cudaMemcpyDeviceToHost, d2h_[src]), "staged d2h");
                const cudaEvent_t ev = chunk_ev_[src].at(ev_idx++);
                gpu_check_cuda(cudaEventRecord(ev, d2h_[src]), "staged d2h ev");
                gpu_check_cuda(cudaSetDevice(dst), "staged h2d dev");
                gpu_check_cuda(cudaStreamWaitEvent(dma_[dst], ev, 0), "staged h2d wait");
                gpu_check_cuda(cudaMemcpyAsync(
                    dst_ptr + std::size_t(c) * words + off, stage + std::size_t(c) * words + off,
                    cnt * sizeof(Word), cudaMemcpyHostToDevice, dma_[dst]), "staged h2d");
            }
        }
        gpu_check_cuda(cudaSetDevice(dst), "staged record reuse");
        gpu_check_cuda(cudaEventRecord(stage_reuse_ev_[dst], dma_[dst]), "staged reuse ev");
        wire_msgs_out += src_info.contiguous ? 1 : 2;
    }

public:
    explicit DualPeerTransfer(std::size_t max_q = 32, std::size_t chunk_bytes = 4ULL << 20)
        : max_q_(max_q),
          capacity_words_(2ULL * 65536ULL * max_q),
          chunk_words_(chunk_bytes / sizeof(Word)) {
        using poseidon::gpu::gpu_check_cuda;
        if (!max_q || max_q > 64 || !chunk_words_ || (chunk_bytes % sizeof(Word)))
            throw std::invalid_argument("DualPeerTransfer: invalid max_q or chunk_bytes");
        int can01 = 0, can10 = 0;
        gpu_check_cuda(cudaDeviceCanAccessPeer(&can01, 0, 1), "canAccessPeer 0->1");
        gpu_check_cuda(cudaDeviceCanAccessPeer(&can10, 1, 0), "canAccessPeer 1->0");
        if (can01 && can10) {
            gpu_check_cuda(cudaSetDevice(0), "enable p2p dev0");
            cudaError_t e0 = cudaDeviceEnablePeerAccess(1, 0);
            if (e0 != cudaSuccess && e0 != cudaErrorPeerAccessAlreadyEnabled)
                gpu_check_cuda(e0, "enablePeerAccess 0->1");
            cudaGetLastError();
            gpu_check_cuda(cudaSetDevice(1), "enable p2p dev1");
            cudaError_t e1 = cudaDeviceEnablePeerAccess(0, 0);
            if (e1 != cudaSuccess && e1 != cudaErrorPeerAccessAlreadyEnabled)
                gpu_check_cuda(e1, "enablePeerAccess 1->0");
            cudaGetLastError();
            p2p_supported_ = true;
        }
        p2p_enabled_ = query_p2p_requested() && p2p_supported_;
        gpu_check_cuda(cudaSetDevice(0), "alloc host staging dev0");
        for (int d = 0; d < 2; ++d) {
            gpu_check_cuda(cudaMallocHost(reinterpret_cast<void **>(&staging_[d]),
                                          capacity_words_ * sizeof(Word)), "alloc peer pinned staging");
            gpu_check_cuda(cudaSetDevice(d), "init peer streams/events");
            gpu_check_cuda(cudaStreamCreateWithFlags(&dma_[d], cudaStreamNonBlocking), "dma stream");
            gpu_check_cuda(cudaStreamCreateWithFlags(&d2h_[d], cudaStreamNonBlocking), "d2h stream");
            gpu_check_cuda(cudaEventCreateWithFlags(&prod_ev_[d], cudaEventDisableTiming), "prod ev");
            gpu_check_cuda(cudaEventCreateWithFlags(&done_ev_[d], cudaEventDisableTiming), "done ev");
            gpu_check_cuda(cudaEventCreateWithFlags(&stage_reuse_ev_[d], cudaEventDisableTiming), "reuse ev");
            gpu_check_cuda(cudaEventCreate(&time_start_[d]), "time start ev");
            gpu_check_cuda(cudaEventCreate(&time_stop_[d]), "time stop ev");
            const std::size_t chunks = 2ULL * ((capacity_words_ / 2 + chunk_words_ - 1) / chunk_words_);
            chunk_ev_[d].resize(chunks);
            for (auto &ev : chunk_ev_[d])
                gpu_check_cuda(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "chunk ev");
        }
        gpu_check_cuda(cudaSetDevice(0), "restore dev0");
    }

    ~DualPeerTransfer() {
        for (int d = 0; d < 2; ++d) {
            cudaSetDevice(d);
            if (dma_[d]) { cudaStreamSynchronize(dma_[d]); cudaStreamDestroy(dma_[d]); }
            if (d2h_[d]) { cudaStreamSynchronize(d2h_[d]); cudaStreamDestroy(d2h_[d]); }
            if (prod_ev_[d]) cudaEventDestroy(prod_ev_[d]);
            if (done_ev_[d]) cudaEventDestroy(done_ev_[d]);
            if (stage_reuse_ev_[d]) cudaEventDestroy(stage_reuse_ev_[d]);
            if (time_start_[d]) cudaEventDestroy(time_start_[d]);
            if (time_stop_[d]) cudaEventDestroy(time_stop_[d]);
            for (auto &ev : chunk_ev_[d]) cudaEventDestroy(ev);
        }
        for (int d = 0; d < 2; ++d) {
            if (staging_[d]) cudaFreeHost(staging_[d]);
        }
        cudaSetDevice(0);
    }

    DualPeerTransfer(const DualPeerTransfer &) = delete;
    DualPeerTransfer &operator=(const DualPeerTransfer &) = delete;

    bool p2p_supported() const noexcept { return p2p_supported_; }
    bool p2p_enabled() const noexcept { return p2p_enabled_; }
    void set_p2p(bool on) {
        if (on && !p2p_supported_)
            throw std::runtime_error("DualPeerTransfer: requested p2p=1 on unsupported pair");
        p2p_enabled_ = on;
    }
    const char *route_name() const noexcept { return p2p_enabled_ ? "p2p_direct" : "host_staged"; }
    void barrier() { barrier_.arrive_and_wait(); }

    // Bidirectional AllGather: rank0 contributes local_cts (M CTs), rank1 contributes
    // local_cts (M CTs). Each rank receives the peer's M CTs allocated under its OWN
    // thread-local RMM pool and returns ordered [rank0_cts..., rank1_cts...].
    std::vector<CT> allgather_shards(
        int rank, std::vector<CT> &&local_cts, cudaStream_t compute_stream,
        std::size_t repeat, const std::string &stage, bool record_timing,
        std::vector<CommRecord> *ledger = nullptr) {
        using poseidon::gpu::gpu_check_cuda;
        if (rank < 0 || rank > 1) throw std::invalid_argument("allgather invalid rank");
        if (!repeat) throw std::invalid_argument("comm_repeat must be >= 1");
        const int peer = 1 - rank;
        gpu_check_cuda(cudaSetDevice(rank), "allgather set rank device");
        gpu_check_cuda(cudaEventRecord(prod_ev_[rank], compute_stream), "allgather record prod");

        shared_meta_[rank].clear();
        shared_meta_[rank].reserve(local_cts.size());
        for (const auto &ct : local_cts)
            shared_meta_[rank].push_back(describe(ct, rank, capacity_words_));

        barrier_.arrive_and_wait();

        const auto &peer_metas = shared_meta_[peer];
        // Allocate destination buffers on THIS rank's RMM pool and compute_stream's device.
        recv_slot_[rank].clear();
        recv_slot_[rank].reserve(peer_metas.size());
        std::size_t bytes_one_way = 0;
        for (const auto &m : peer_metas) {
            CT dst = CT::allocate_single_device(m.meta.degree, m.meta.q_count, 2, rank);
            dst.meta = m.meta;
            recv_slot_[rank].push_back(std::move(dst));
            bytes_one_way += 2ULL * m.q_words * sizeof(Word);
        }

        const auto t_wall0 = std::chrono::steady_clock::now();
        gpu_check_cuda(cudaSetDevice(rank), "allgather dma wait setup");
        gpu_check_cuda(cudaStreamWaitEvent(dma_[rank], prod_ev_[peer], 0), "dma wait remote prod");
        if (!p2p_enabled_) {
            gpu_check_cuda(cudaSetDevice(peer), "staged d2h wait peer prod");
            gpu_check_cuda(cudaStreamWaitEvent(d2h_[peer], prod_ev_[peer], 0), "staged d2h prod");
        }
        // Ensure destination allocations on rank's compute_stream exist before writing into them?
        // RMM pool memory is already allocated synchronously on host, so dma_[rank] can write immediately,
        // wait prod_ev_[rank] is free/fast safety if reuse happened on rank's compute_stream:
        gpu_check_cuda(cudaSetDevice(rank), "allgather dma wait local alloc");
        gpu_check_cuda(cudaStreamWaitEvent(dma_[rank], prod_ev_[rank], 0), "dma wait local alloc");

        if (record_timing) {
            gpu_check_cuda(cudaSetDevice(rank), "allgather timing start");
            gpu_check_cuda(cudaEventRecord(time_start_[rank], dma_[rank]), "allgather start ev");
        }

        std::size_t wire_msgs = 0;
        for (std::size_t rep = 0; rep < repeat; ++rep) {
            for (std::size_t i = 0; i < peer_metas.size(); ++i) {
                copy_ct_async(peer_metas[i], peer, recv_slot_[rank][i], rank, wire_msgs);
            }
        }
        gpu_check_cuda(cudaSetDevice(rank), "allgather record done");
        gpu_check_cuda(cudaEventRecord(done_ev_[rank], dma_[rank]), "allgather done ev");
        if (record_timing) {
            gpu_check_cuda(cudaEventRecord(time_stop_[rank], dma_[rank]), "allgather stop ev");
            gpu_check_cuda(cudaEventSynchronize(time_stop_[rank]), "allgather timing sync");
        }

        const double wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_wall0).count();
        double dma_ms = 0.0;
        if (record_timing) {
            float fms = 0.0f;
            gpu_check_cuda(cudaEventElapsedTime(&fms, time_start_[rank], time_stop_[rank]), "allgather elapsed");
            dma_ms = static_cast<double>(fms);
            if (ledger) {
                ledger->push_back(CommRecord{
                    stage, peer, rank, peer_metas.size(), repeat,
                    bytes_one_way, bytes_one_way * repeat, wire_msgs, dma_ms, wall_ms});
            }
        }

        // Wait until BOTH peer DMA streams finish reading source CTs BEFORE any rank frees local_cts.
        barrier_.arrive_and_wait();
        gpu_check_cuda(cudaSetDevice(rank), "allgather compute wait");
        gpu_check_cuda(cudaStreamWaitEvent(compute_stream, done_ev_[rank], 0), "compute wait local rx");
        gpu_check_cuda(cudaStreamWaitEvent(compute_stream, done_ev_[peer], 0), "compute wait remote rx done reading");

        std::vector<CT> out;
        out.reserve(local_cts.size() + recv_slot_[rank].size());
        if (rank == 0) {
            for (auto &ct : local_cts) out.push_back(std::move(ct));
            for (auto &ct : recv_slot_[0]) out.push_back(std::move(ct));
        } else {
            for (auto &ct : recv_slot_[1]) out.push_back(std::move(ct));
            for (auto &ct : local_cts) out.push_back(std::move(ct));
        }
        recv_slot_[rank].clear();
        return out;
    }

    // Point-to-point asymmetric send from src_rank -> dst_rank (e.g., RMS partial sum 1->0,
    // scalar inv_sqrt broadcast 0->1, or final gather 1->0). Both threads must participate.
    std::vector<CT> send_recv(
        int rank, int src_rank, std::vector<CT> &&src_cts, cudaStream_t compute_stream,
        std::size_t repeat, const std::string &stage, bool record_timing,
        std::vector<CommRecord> *ledger = nullptr) {
        using poseidon::gpu::gpu_check_cuda;
        const int dst_rank = 1 - src_rank;
        if (rank < 0 || rank > 1 || (src_rank != 0 && src_rank != 1))
            throw std::invalid_argument("send_recv invalid rank");
        if (!repeat) throw std::invalid_argument("comm_repeat must be >= 1");

        gpu_check_cuda(cudaSetDevice(rank), "send_recv set dev");
        if (rank == src_rank) {
            gpu_check_cuda(cudaEventRecord(prod_ev_[rank], compute_stream), "send_recv src prod");
            shared_meta_[src_rank].clear();
            shared_meta_[src_rank].reserve(src_cts.size());
            for (const auto &ct : src_cts)
                shared_meta_[src_rank].push_back(describe(ct, src_rank, capacity_words_));
        } else {
            gpu_check_cuda(cudaEventRecord(prod_ev_[rank], compute_stream), "send_recv dst ready");
        }

        barrier_.arrive_and_wait();

        std::vector<CT> dst_cts;
        if (rank == dst_rank) {
            const auto &metas = shared_meta_[src_rank];
            dst_cts.reserve(metas.size());
            std::size_t bytes_one_way = 0;
            for (const auto &m : metas) {
                CT dst = CT::allocate_single_device(m.meta.degree, m.meta.q_count, 2, dst_rank);
                dst.meta = m.meta;
                dst_cts.push_back(std::move(dst));
                bytes_one_way += 2ULL * m.q_words * sizeof(Word);
            }
            const auto t_wall0 = std::chrono::steady_clock::now();
            gpu_check_cuda(cudaSetDevice(dst_rank), "send_recv dma wait");
            gpu_check_cuda(cudaStreamWaitEvent(dma_[dst_rank], prod_ev_[src_rank], 0), "send_recv wait src");
            gpu_check_cuda(cudaStreamWaitEvent(dma_[dst_rank], prod_ev_[dst_rank], 0), "send_recv wait dst alloc");
            if (!p2p_enabled_) {
                gpu_check_cuda(cudaSetDevice(src_rank), "send_recv d2h wait src");
                gpu_check_cuda(cudaStreamWaitEvent(d2h_[src_rank], prod_ev_[src_rank], 0), "send_recv d2h src");
            }
            if (record_timing) {
                gpu_check_cuda(cudaSetDevice(dst_rank), "send_recv time start");
                gpu_check_cuda(cudaEventRecord(time_start_[dst_rank], dma_[dst_rank]), "send_recv start");
            }
            std::size_t wire_msgs = 0;
            for (std::size_t rep = 0; rep < repeat; ++rep) {
                for (std::size_t i = 0; i < metas.size(); ++i)
                    copy_ct_async(metas[i], src_rank, dst_cts[i], dst_rank, wire_msgs);
            }
            gpu_check_cuda(cudaSetDevice(dst_rank), "send_recv record done");
            gpu_check_cuda(cudaEventRecord(done_ev_[dst_rank], dma_[dst_rank]), "send_recv done ev");
            if (record_timing) {
                gpu_check_cuda(cudaEventRecord(time_stop_[dst_rank], dma_[dst_rank]), "send_recv stop ev");
                gpu_check_cuda(cudaEventSynchronize(time_stop_[dst_rank]), "send_recv timing sync");
            }
            const double wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_wall0).count();
            if (record_timing) {
                float fms = 0.0f;
                gpu_check_cuda(cudaEventElapsedTime(&fms, time_start_[dst_rank], time_stop_[dst_rank]), "send_recv elapsed");
                if (ledger) {
                    ledger->push_back(CommRecord{
                        stage, src_rank, dst_rank, metas.size(), repeat,
                        bytes_one_way, bytes_one_way * repeat, wire_msgs, static_cast<double>(fms), wall_ms});
                }
            }
        }

        barrier_.arrive_and_wait();
        gpu_check_cuda(cudaSetDevice(rank), "send_recv final wait");
        gpu_check_cuda(cudaStreamWaitEvent(compute_stream, done_ev_[dst_rank], 0), "send_recv compute wait");
        return dst_cts;
    }
};

} // namespace cerium_source::dual
