#pragma once
// Opt-in disk caches for LLaMA manual runners:
//  1) Public constants cache (LLAMA_PUBLIC_CACHE_DIR / --public-cache):
//     GPU-ready compressed C2S/S2C QP diagonals + EvalMod plans/plaintexts +
//     +/-i plaintexts. Contains zero secret material and is safe to share
//     across fresh keys.
//  2) Dev key cache (LLAMA_DEV_KEY_CACHE_DIR / --dev-key-cache):
//     Development-only SecretKey/PublicKey/RelinKeys/GaloisKeys + per-level
//     fast-P/SSE key bundles. Never enabled by default; explicitly logged.
#include "poseidon/gpu/gpu_double_hoist.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_key.h"
#include "poseidon/gpu/gpu_plaintext.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/kswitchkeys.h"
#include "poseidon/key/publickey.h"
#include "poseidon/key/relinkeys.h"
#include "poseidon/key/secretkey.h"
#include "poseidon/poseidon_context.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace cerium_source::cache {

namespace fs = std::filesystem;

inline std::string env_path(const char *name) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::string(v) : std::string();
}

inline std::string public_cache_dir(const std::string &override_dir = {}) {
    return override_dir.empty() ? env_path("LLAMA_PUBLIC_CACHE_DIR") : override_dir;
}

inline std::string dev_key_cache_dir(const std::string &override_dir = {}) {
    return override_dir.empty() ? env_path("LLAMA_DEV_KEY_CACHE_DIR") : override_dir;
}

// 64-bit FNV-1a hasher for deterministic cache namespace keys.
class Hasher {
    std::uint64_t h_ = 14695981039346656037ULL;
public:
    void bytes(const void *data, std::size_t n) noexcept {
        const auto *p = static_cast<const std::uint8_t *>(data);
        for (std::size_t i = 0; i < n; ++i) {
            h_ ^= p[i];
            h_ *= 1099511628211ULL;
        }
    }
    template <class T>
    void pod(const T &v) noexcept {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&v, sizeof(T));
    }
    void str(const std::string &s) noexcept {
        const std::uint64_t n = s.size();
        pod(n);
        if (n) bytes(s.data(), n);
    }
    std::uint64_t finish() const noexcept { return h_; }
    std::string hex() const {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0') << std::setw(16) << h_;
        return oss.str();
    }
};

inline std::uint64_t parameters_fingerprint(const poseidon::ParametersLiteral &lit) {
    Hasher h;
    h.str("LLAMA_PARAMS_V1");
    h.pod(static_cast<std::uint64_t>(lit.degree()));
    h.pod(static_cast<std::uint64_t>(lit.log_n()));
    h.pod(static_cast<std::uint64_t>(lit.log_slots()));
    h.pod(static_cast<std::uint64_t>(lit.hamming_weight()));
    h.pod(static_cast<std::uint64_t>(lit.q0_level()));
    h.pod(static_cast<std::uint64_t>(lit.q().size()));
    for (const auto &m : lit.q()) h.pod(static_cast<std::uint64_t>(m.value()));
    h.pod(static_cast<std::uint64_t>(lit.p().size()));
    for (const auto &m : lit.p()) h.pod(static_cast<std::uint64_t>(m.value()));
    return h.finish();
}

inline std::string hex64(std::uint64_t v) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << v;
    return oss.str();
}

inline std::uint64_t secret_key_fingerprint(const poseidon::SecretKey &sk) {
    Hasher h;
    h.str("LLAMA_SK_FP_V1");
    const auto &poly = sk.data();
    h.pod(static_cast<std::uint64_t>(poly.coeff_count()));
    h.bytes(poly.data(), poly.coeff_count() * sizeof(std::uint64_t));
    return h.finish();
}

inline std::string public_namespace_key(const poseidon::ParametersLiteral &lit,
                                        double raised_scale,
                                        std::uint32_t min_quotient_log2) {
    Hasher h;
    h.str("LLAMA_PUBLIC_CACHE_V1");
    h.pod(parameters_fingerprint(lit));
    h.pod(raised_scale);
    h.pod(min_quotient_log2);
    return h.hex();
}

inline std::string dev_key_namespace_key(const poseidon::ParametersLiteral &lit,
                                         const std::vector<int> &rot_steps,
                                         bool fast_p,
                                         bool fast_evalmod_p,
                                         std::uint32_t sse_hw) {
    Hasher h;
    h.str("LLAMA_DEV_KEY_CACHE_V1");
    h.pod(parameters_fingerprint(lit));
    h.pod(static_cast<std::uint8_t>(fast_p ? 1 : 0));
    h.pod(static_cast<std::uint8_t>(fast_evalmod_p ? 1 : 0));
    h.pod(sse_hw);
    h.pod(static_cast<std::uint64_t>(rot_steps.size()));
    for (int s : rot_steps) h.pod(static_cast<std::int32_t>(s));
    return h.hex();
}

// Atomic binary file writer (write to .tmp.<pid>.<seq> then rename).
class AtomicWriter {
    fs::path final_path_;
    fs::path tmp_path_;
    std::ofstream out_;
    bool committed_ = false;
    static std::uint64_t next_seq() noexcept {
        static std::atomic<std::uint64_t> counter{0};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }
public:
    explicit AtomicWriter(const fs::path &path)
        : final_path_(path),
          tmp_path_(path.string() + ".tmp." + std::to_string(::getpid()) + "." + std::to_string(next_seq())) {
        if (!final_path_.parent_path().empty()) {
            fs::create_directories(final_path_.parent_path());
        }
        out_.open(tmp_path_, std::ios::binary | std::ios::trunc);
        if (!out_) throw std::runtime_error("cache: cannot open temp file " + tmp_path_.string());
    }
    ~AtomicWriter() {
        if (!committed_) {
            if (out_.is_open()) out_.close();
            std::error_code ec;
            fs::remove(tmp_path_, ec);
        }
    }
    std::ofstream &stream() { return out_; }
    void commit() {
        out_.flush();
        if (!out_) throw std::runtime_error("cache: write failed " + tmp_path_.string());
        out_.close();
        fs::rename(tmp_path_, final_path_);
        committed_ = true;
    }
};

namespace detail {

template <class T>
void write_pod(std::ostream &os, const T &v) {
    static_assert(std::is_trivially_copyable_v<T>);
    os.write(reinterpret_cast<const char *>(&v), sizeof(T));
    if (!os) throw std::runtime_error("cache: stream write error");
}

template <class T>
T read_pod(std::istream &is) {
    static_assert(std::is_trivially_copyable_v<T>);
    T v{};
    is.read(reinterpret_cast<char *>(&v), sizeof(T));
    if (!is) throw std::runtime_error("cache: stream read error");
    return v;
}

template <class T>
void write_vec(std::ostream &os, const std::vector<T> &v) {
    static_assert(std::is_trivially_copyable_v<T>);
    const std::uint64_t n = v.size();
    write_pod(os, n);
    if (n) {
        os.write(reinterpret_cast<const char *>(v.data()), n * sizeof(T));
        if (!os) throw std::runtime_error("cache: vector write error");
    }
}

template <class T>
std::vector<T> read_vec(std::istream &is) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto n = read_pod<std::uint64_t>(is);
    std::vector<T> v(static_cast<std::size_t>(n));
    if (n) {
        is.read(reinterpret_cast<char *>(v.data()), n * sizeof(T));
        if (!is) throw std::runtime_error("cache: vector read error");
    }
    return v;
}

inline void write_magic(std::ostream &os, const char magic[16], std::uint64_t fp) {
    os.write(magic, 16);
    write_pod(os, fp);
}

inline bool read_and_check_magic(std::istream &is, const char expected[16], std::uint64_t expected_fp) {
    char got[16]{};
    is.read(got, 16);
    if (!is || std::memcmp(got, expected, 16) != 0) return false;
    const auto fp = read_pod<std::uint64_t>(is);
    return is && fp == expected_fp;
}

inline void save_gpu_plaintext(std::ostream &os, const poseidon::gpu::GpuPlaintextData &pt) {
    const std::uint8_t present = pt.empty() ? 0 : 1;
    write_pod(os, present);
    if (!present) return;
    if (pt.fields_.size() != 1)
        throw std::runtime_error("cache: only single-device GpuPlaintextData is supported");
    write_pod(os, pt.meta.parms_id);
    write_pod(os, pt.meta.scale);
    write_pod(os, static_cast<std::uint8_t>(pt.meta.is_ntt_form ? 1 : 0));
    write_pod(os, static_cast<std::uint64_t>(pt.meta.degree));
    write_pod(os, static_cast<std::uint64_t>(pt.meta.q_count));
    write_pod(os, static_cast<std::uint64_t>(pt.meta.p_count));
    const std::size_t words = pt.fields_[0].buffer.size();
    std::vector<poseidon::gpu::GpuWord> host(words);
    if (words) pt.fields_[0].buffer.copy_to_host(host.data(), words);
    write_vec(os, host);
}

inline poseidon::gpu::GpuPlaintextData load_gpu_plaintext(
    std::istream &is, const poseidon::PoseidonContext &context, int device_id) {
    const auto present = read_pod<std::uint8_t>(is);
    if (!present) return {};
    const auto parms_id = read_pod<poseidon::parms_id_type>(is);
    const auto scale = read_pod<double>(is);
    const bool is_ntt = read_pod<std::uint8_t>(is) != 0;
    const auto degree = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
    const auto q_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
    const auto p_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
    auto host = read_vec<poseidon::gpu::GpuWord>(is);
    if (parms_id != poseidon::parms_id_zero) {
        const auto cd = context.crt_context()->get_context_data(parms_id);
        if (!cd || cd->parms().degree() != degree)
            throw std::runtime_error("cache: cached GpuPlaintextData parms_id mismatch");
    }
    auto pt = poseidon::gpu::GpuPlaintextData::allocate_single_device(degree, q_count, device_id, p_count);
    pt.meta.parms_id = parms_id;
    pt.meta.scale = scale;
    pt.meta.is_ntt_form = is_ntt;
    if (host.size() != pt.fields_[0].buffer.size())
        throw std::runtime_error("cache: cached GpuPlaintextData word count mismatch");
    if (!host.empty()) pt.fields_[0].buffer.copy_from_host(host.data(), host.size());
    return pt;
}

} // namespace detail

// ============================================================================
// Public Constants Cache: GPU-ready compressed QP C2S/S2C matrices
// ============================================================================
inline constexpr char kQpGroupMagic[16] = {'L','L','A','M','A','_','Q','P','_','V','1',0,0,0,0,0};

inline bool read_qp_group_rot_index(const fs::path &path,
                                    std::uint64_t params_fp,
                                    std::vector<int> &rot_index_out) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    try {
        if (!detail::read_and_check_magic(is, kQpGroupMagic, params_fp)) return false;
        rot_index_out = detail::read_vec<int>(is);
        return is.good();
    } catch (...) {
        return false;
    }
}

inline void save_qp_matrix_group(const fs::path &path,
                                 std::uint64_t params_fp,
                                 const poseidon::gpu::GpuLinearMatrixGroupQP &group) {
    AtomicWriter writer(path);
    auto &os = writer.stream();
    detail::write_magic(os, kQpGroupMagic, params_fp);
    detail::write_vec(os, group.rot_index());
    detail::write_pod(os, group.step());
    detail::write_pod(os, group.rescale_min_scale());
    detail::write_vec(os, group.rescale_counts());
    detail::write_pod(os, static_cast<std::uint64_t>(group.data().size()));
    for (const auto &mat : group.data()) {
        if (!mat.plan.compressed_plaintexts || mat.compressed_plain_vec_qp.empty())
            throw std::runtime_error("cache: save_qp_matrix_group requires compressed QP matrices");
        detail::write_pod(os, mat.log_slots);
        detail::write_pod(os, mat.n1);
        detail::write_pod(os, mat.level);
        detail::write_pod(os, mat.scale);
        detail::write_vec(os, mat.rot_index);

        detail::write_pod(os, static_cast<std::uint64_t>(mat.compressed_plain_vec_qp.size()));
        for (const auto &[diag, cplain] : mat.compressed_plain_vec_qp) {
            detail::write_pod(os, static_cast<std::int32_t>(diag));
            detail::write_pod(os, cplain.meta.parms_id);
            detail::write_pod(os, cplain.meta.scale);
            detail::write_pod(os, static_cast<std::uint8_t>(cplain.meta.is_ntt_form ? 1 : 0));
            detail::write_pod(os, static_cast<std::uint64_t>(cplain.meta.degree));
            detail::write_pod(os, static_cast<std::uint64_t>(cplain.meta.q_count));
            detail::write_pod(os, static_cast<std::uint64_t>(cplain.meta.p_count));
            detail::write_pod(os, static_cast<std::uint64_t>(cplain.period));
            std::vector<poseidon::gpu::GpuWord> host(cplain.residues.size());
            if (!host.empty()) cplain.residues.copy_to_host(host.data(), host.size());
            detail::write_vec(os, host);
        }

        const auto &plan = mat.plan;
        detail::write_pod(os, plan.log_slots);
        detail::write_pod(os, plan.n1);
        detail::write_pod(os, plan.n2);
        detail::write_pod(os, plan.rescale_count);
        detail::write_vec(os, plan.baby_steps);
        detail::write_vec(os, plan.giant_steps);
        detail::write_vec(os, plan.group_term_offsets);
        detail::write_vec(os, plan.terms);
    }
    writer.commit();
}

inline bool load_qp_matrix_group(const fs::path &path,
                                 std::uint64_t params_fp,
                                 const poseidon::PoseidonContext &context,
                                 int device_id,
                                 poseidon::gpu::GpuLinearMatrixGroupQP &out) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    try {
        if (!detail::read_and_check_magic(is, kQpGroupMagic, params_fp)) return false;
        poseidon::gpu::GpuLinearMatrixGroupQP result;
        result.rot_index() = detail::read_vec<int>(is);
        result.set_step(detail::read_pod<std::uint32_t>(is));
        result.set_rescale_min_scale(detail::read_pod<double>(is));
        result.rescale_counts() = detail::read_vec<std::uint32_t>(is);
        const auto num_mats = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
        result.data().reserve(num_mats);

        for (std::size_t mi = 0; mi < num_mats; ++mi) {
            poseidon::gpu::GpuMatrixPlainQP mat;
            mat.log_slots = detail::read_pod<std::uint32_t>(is);
            mat.n1 = detail::read_pod<std::uint32_t>(is);
            mat.level = detail::read_pod<std::uint32_t>(is);
            mat.scale = detail::read_pod<double>(is);
            mat.rot_index = detail::read_vec<int>(is);

            const auto num_diags = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
            for (std::size_t di = 0; di < num_diags; ++di) {
                const int diag = static_cast<int>(detail::read_pod<std::int32_t>(is));
                poseidon::gpu::GpuCompressedPlaintextQP cp;
                cp.meta.parms_id = detail::read_pod<poseidon::parms_id_type>(is);
                cp.meta.scale = detail::read_pod<double>(is);
                cp.meta.is_ntt_form = detail::read_pod<std::uint8_t>(is) != 0;
                cp.meta.degree = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
                cp.meta.q_count = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
                cp.meta.p_count = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
                cp.period = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
                auto host = detail::read_vec<poseidon::gpu::GpuWord>(is);
                const auto cd = context.crt_context()->get_context_data(cp.meta.parms_id);
                if (!cd || cd->parms().degree() != cp.meta.degree)
                    return false;
                if (host.size() != (cp.meta.q_count + cp.meta.p_count) * cp.period)
                    return false;
                cp.residues.allocate(host.size(), device_id);
                if (!host.empty()) cp.residues.copy_from_host(host.data(), host.size());
                cp.exact_device_reconstruction = true;
                mat.compressed_plain_vec_qp.emplace(diag, std::move(cp));
            }

            auto &plan = mat.plan;
            plan.log_slots = detail::read_pod<std::uint32_t>(is);
            plan.n1 = detail::read_pod<std::uint32_t>(is);
            plan.n2 = detail::read_pod<std::uint32_t>(is);
            plan.rescale_count = detail::read_pod<std::uint32_t>(is);
            plan.baby_steps = detail::read_vec<int>(is);
            plan.giant_steps = detail::read_vec<int>(is);
            plan.group_term_offsets = detail::read_vec<std::uint32_t>(is);
            plan.terms = detail::read_vec<poseidon::gpu::GpuDoubleHoistTerm>(is);
            plan.compressed_plaintexts = true;

            std::vector<const poseidon::gpu::GpuWord *> q_ptrs;
            std::vector<const poseidon::gpu::GpuWord *> p_ptrs;
            std::vector<std::uint32_t> periods;
            std::vector<std::uint32_t> term_babies;
            q_ptrs.reserve(plan.terms.size());
            p_ptrs.reserve(plan.terms.size());
            periods.reserve(plan.terms.size());
            term_babies.reserve(plan.terms.size());
            for (const auto &term : plan.terms) {
                const int diag = plan.giant_steps.at(term.giant_index) + plan.baby_steps.at(term.baby_index);
                const auto &cp = mat.compressed_plain_vec_qp.at(diag);
                q_ptrs.push_back(cp.residues.data());
                p_ptrs.push_back(cp.residues.data() + cp.meta.q_count * cp.period);
                periods.push_back(static_cast<std::uint32_t>(cp.period));
                term_babies.push_back(term.baby_index);
            }
            plan.diagonal_q_ptrs.allocate(q_ptrs.size(), device_id);
            plan.diagonal_p_ptrs.allocate(p_ptrs.size(), device_id);
            plan.diagonal_periods.allocate(periods.size(), device_id);
            plan.term_baby_indices.allocate(term_babies.size(), device_id);
            plan.group_term_offsets_device.allocate(plan.group_term_offsets.size(), device_id);
            plan.diagonal_q_ptrs.copy_from_host(q_ptrs.data(), q_ptrs.size());
            plan.diagonal_p_ptrs.copy_from_host(p_ptrs.data(), p_ptrs.size());
            plan.diagonal_periods.copy_from_host(periods.data(), periods.size());
            plan.term_baby_indices.copy_from_host(term_babies.data(), term_babies.size());
            plan.group_term_offsets_device.copy_from_host(
                plan.group_term_offsets.data(), plan.group_term_offsets.size());

            result.data().push_back(std::move(mat));
        }
        out = std::move(result);
        return true;
    } catch (...) {
        return false;
    }
}

// ============================================================================
// Public Constants Cache: EvalModData + +/-i plaintexts + scalar constants
// ============================================================================
inline constexpr char kEvalModMagic[16] = {'L','L','A','M','A','_','E','M','_','V','1',0,0,0,0,0};

struct CachedEvalModBundle {
    poseidon::gpu::GpuBootstrapData::EvalModData eval_mod;
    poseidon::gpu::GpuPlaintextData minus_i;
    poseidon::gpu::GpuPlaintextData plus_i;
    double alpha = 0.0;
    double eta = 0.0;
    double target = 0.0;
    double m = 0.0;
    double c2s_scale = 0.0;
};

inline void save_eval_mod_bundle(const fs::path &path,
                                 std::uint64_t params_fp,
                                 unsigned ratio,
                                 const CachedEvalModBundle &b) {
    AtomicWriter writer(path);
    auto &os = writer.stream();
    detail::write_magic(os, kEvalModMagic, params_fp);
    detail::write_pod(os, static_cast<std::uint32_t>(ratio));
    detail::write_pod(os, b.alpha);
    detail::write_pod(os, b.eta);
    detail::write_pod(os, b.target);
    detail::write_pod(os, b.m);
    detail::write_pod(os, b.c2s_scale);
    detail::save_gpu_plaintext(os, b.minus_i);
    detail::save_gpu_plaintext(os, b.plus_i);

    const auto &em = b.eval_mod;
    detail::write_pod(os, em.input_scale);
    detail::write_pod(os, em.target_scale);
    detail::write_pod(os, em.output_scale);
    detail::write_pod(os, em.output_parms_id);
    detail::write_pod(os, static_cast<std::uint64_t>(em.output_q_count));
    detail::save_gpu_plaintext(os, em.input_offset_plaintext);
    detail::write_pod(os, static_cast<std::uint8_t>(em.polynomial_basis));
    detail::write_pod(os, em.polynomial_degree);
    detail::write_pod(os, em.polynomial_log_split);
    detail::write_pod(os, static_cast<std::uint8_t>(em.polynomial_flat_bsgs ? 1 : 0));
    detail::write_pod(os, static_cast<std::uint8_t>(em.polynomial_degree_bound_virtual ? 1 : 0));
    detail::write_pod(os, static_cast<std::uint64_t>(em.polynomial_root_anchor_q_count));

    detail::write_pod(os, static_cast<std::uint64_t>(em.basis_steps.size()));
    for (const auto &s : em.basis_steps) {
        detail::write_pod(os, s.output_degree);
        detail::write_pod(os, s.left_degree);
        detail::write_pod(os, s.right_degree);
        detail::write_pod(os, s.correction_degree);
        detail::write_pod(os, s.pre_rescale_scale);
        detail::write_pod(os, s.output_scale);
        detail::write_pod(os, s.rescale_count);
        detail::write_pod(os, static_cast<std::uint8_t>(s.align_left_operand ? 1 : 0));
        detail::save_gpu_plaintext(os, s.operand_alignment_plaintext);
        detail::write_pod(os, s.operand_alignment_pre_rescale_scale);
        detail::write_pod(os, s.operand_alignment_output_scale);
        detail::write_pod(os, s.operand_alignment_rescale_count);
        detail::save_gpu_plaintext(os, s.correction_alignment_plaintext);
        detail::write_pod(os, s.correction_alignment_pre_rescale_scale);
        detail::write_pod(os, s.correction_alignment_rescale_count);
        detail::save_gpu_plaintext(os, s.correction_plaintext);
    }

    auto write_terms = [&](const std::vector<poseidon::gpu::GpuEvalModPolynomialTerm> &terms) {
        detail::write_pod(os, static_cast<std::uint64_t>(terms.size()));
        for (const auto &t : terms) {
            detail::write_pod(os, t.degree);
            detail::save_gpu_plaintext(os, t.coefficient_plaintext);
        }
    };
    write_terms(em.polynomial_terms);

    detail::write_pod(os, em.logical_rescale_count);
    detail::write_pod(os, static_cast<std::uint8_t>(em.dynamic_rescale ? 1 : 0));
    detail::write_pod(os, em.dynamic_min_scale);
    detail::write_pod(os, em.polynomial_rescale_count);
    detail::write_pod(os, em.polynomial_final_rescale_count);
    detail::write_pod(os, em.polynomial_output_scale);
    detail::write_pod(os, static_cast<std::uint8_t>(em.rescale_polynomial_terms_individually ? 1 : 0));

    detail::write_pod(os, static_cast<std::uint64_t>(em.polynomial_blocks.size()));
    for (const auto &blk : em.polynomial_blocks) {
        write_terms(blk.terms);
        detail::write_pod(os, blk.rescale_count);
        detail::write_pod(os, blk.output_scale);
        detail::write_pod(os, static_cast<std::uint64_t>(blk.output_q_count));
    }

    detail::write_pod(os, static_cast<std::uint64_t>(em.polynomial_combine_steps.size()));
    for (const auto &c : em.polynomial_combine_steps) {
        detail::write_pod(os, c.output_node);
        detail::write_pod(os, c.quotient_node);
        detail::write_pod(os, c.remainder_node);
        detail::write_pod(os, c.basis_degree);
        detail::write_pod(os, c.output_scale);
        detail::write_pod(os, static_cast<std::uint64_t>(c.output_q_count));
        detail::write_pod(os, c.quotient_rescale_count);
        detail::write_pod(os, c.quotient_output_scale);
        detail::write_pod(os, c.remainder_rescale_count);
        detail::write_pod(os, c.product_scale);
        detail::write_pod(os, static_cast<std::uint64_t>(c.product_q_count));
        detail::save_gpu_plaintext(os, c.product_scale_plaintext);
        detail::write_pod(os, c.product_aligned_scale);
        detail::save_gpu_plaintext(os, c.remainder_scale_plaintext);
        detail::write_pod(os, c.remainder_aligned_scale);
    }

    detail::write_pod(os, em.polynomial_result_node);

    auto write_pt_vec = [&](const std::vector<poseidon::gpu::GpuPlaintextData> &pts) {
        detail::write_pod(os, static_cast<std::uint64_t>(pts.size()));
        for (const auto &p : pts) detail::save_gpu_plaintext(os, p);
    };
    write_pt_vec(em.chebyshev_one_plaintexts);
    write_pt_vec(em.polynomial_coefficients);
    write_pt_vec(em.double_angle_constants);
    detail::write_vec(os, em.double_angle_rescale_counts);
    std::vector<std::uint64_t> req_q(em.required_relin_q_counts.begin(), em.required_relin_q_counts.end());
    detail::write_vec(os, req_q);
    writer.commit();
}

inline bool load_eval_mod_bundle(const fs::path &path,
                                 std::uint64_t params_fp,
                                 unsigned expected_ratio,
                                 const poseidon::PoseidonContext &context,
                                 int device_id,
                                 CachedEvalModBundle &out) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    try {
        if (!detail::read_and_check_magic(is, kEvalModMagic, params_fp)) return false;
        const auto ratio = detail::read_pod<std::uint32_t>(is);
        if (ratio != expected_ratio) return false;
        CachedEvalModBundle b;
        b.alpha = detail::read_pod<double>(is);
        b.eta = detail::read_pod<double>(is);
        b.target = detail::read_pod<double>(is);
        b.m = detail::read_pod<double>(is);
        b.c2s_scale = detail::read_pod<double>(is);
        b.minus_i = detail::load_gpu_plaintext(is, context, device_id);
        b.plus_i = detail::load_gpu_plaintext(is, context, device_id);

        auto &em = b.eval_mod;
        em.input_scale = detail::read_pod<double>(is);
        em.target_scale = detail::read_pod<double>(is);
        em.output_scale = detail::read_pod<double>(is);
        em.output_parms_id = detail::read_pod<poseidon::parms_id_type>(is);
        em.output_q_count = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
        em.input_offset_plaintext = detail::load_gpu_plaintext(is, context, device_id);
        em.polynomial_basis = static_cast<poseidon::gpu::GpuEvalModPolynomialBasis>(detail::read_pod<std::uint8_t>(is));
        em.polynomial_degree = detail::read_pod<std::uint32_t>(is);
        em.polynomial_log_split = detail::read_pod<std::uint32_t>(is);
        em.polynomial_flat_bsgs = detail::read_pod<std::uint8_t>(is) != 0;
        em.polynomial_degree_bound_virtual = detail::read_pod<std::uint8_t>(is) != 0;
        em.polynomial_root_anchor_q_count = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));

        const auto n_basis = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
        em.basis_steps.resize(n_basis);
        for (auto &s : em.basis_steps) {
            s.output_degree = detail::read_pod<std::uint32_t>(is);
            s.left_degree = detail::read_pod<std::uint32_t>(is);
            s.right_degree = detail::read_pod<std::uint32_t>(is);
            s.correction_degree = detail::read_pod<std::uint32_t>(is);
            s.pre_rescale_scale = detail::read_pod<double>(is);
            s.output_scale = detail::read_pod<double>(is);
            s.rescale_count = detail::read_pod<std::uint32_t>(is);
            s.align_left_operand = detail::read_pod<std::uint8_t>(is) != 0;
            s.operand_alignment_plaintext = detail::load_gpu_plaintext(is, context, device_id);
            s.operand_alignment_pre_rescale_scale = detail::read_pod<double>(is);
            s.operand_alignment_output_scale = detail::read_pod<double>(is);
            s.operand_alignment_rescale_count = detail::read_pod<std::uint32_t>(is);
            s.correction_alignment_plaintext = detail::load_gpu_plaintext(is, context, device_id);
            s.correction_alignment_pre_rescale_scale = detail::read_pod<double>(is);
            s.correction_alignment_rescale_count = detail::read_pod<std::uint32_t>(is);
            s.correction_plaintext = detail::load_gpu_plaintext(is, context, device_id);
        }

        auto read_terms = [&]() {
            const auto n = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
            std::vector<poseidon::gpu::GpuEvalModPolynomialTerm> terms(n);
            for (auto &t : terms) {
                t.degree = detail::read_pod<std::uint32_t>(is);
                t.coefficient_plaintext = detail::load_gpu_plaintext(is, context, device_id);
            }
            return terms;
        };
        em.polynomial_terms = read_terms();

        em.logical_rescale_count = detail::read_pod<std::uint32_t>(is);
        em.dynamic_rescale = detail::read_pod<std::uint8_t>(is) != 0;
        em.dynamic_min_scale = detail::read_pod<double>(is);
        em.polynomial_rescale_count = detail::read_pod<std::uint32_t>(is);
        em.polynomial_final_rescale_count = detail::read_pod<std::uint32_t>(is);
        em.polynomial_output_scale = detail::read_pod<double>(is);
        em.rescale_polynomial_terms_individually = detail::read_pod<std::uint8_t>(is) != 0;

        const auto n_blocks = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
        em.polynomial_blocks.resize(n_blocks);
        for (auto &blk : em.polynomial_blocks) {
            blk.terms = read_terms();
            blk.rescale_count = detail::read_pod<std::uint32_t>(is);
            blk.output_scale = detail::read_pod<double>(is);
            blk.output_q_count = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
        }

        const auto n_comb = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
        em.polynomial_combine_steps.resize(n_comb);
        for (auto &c : em.polynomial_combine_steps) {
            c.output_node = detail::read_pod<std::uint32_t>(is);
            c.quotient_node = detail::read_pod<std::uint32_t>(is);
            c.remainder_node = detail::read_pod<std::uint32_t>(is);
            c.basis_degree = detail::read_pod<std::uint32_t>(is);
            c.output_scale = detail::read_pod<double>(is);
            c.output_q_count = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
            c.quotient_rescale_count = detail::read_pod<std::uint32_t>(is);
            c.quotient_output_scale = detail::read_pod<double>(is);
            c.remainder_rescale_count = detail::read_pod<std::uint32_t>(is);
            c.product_scale = detail::read_pod<double>(is);
            c.product_q_count = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
            c.product_scale_plaintext = detail::load_gpu_plaintext(is, context, device_id);
            c.product_aligned_scale = detail::read_pod<double>(is);
            c.remainder_scale_plaintext = detail::load_gpu_plaintext(is, context, device_id);
            c.remainder_aligned_scale = detail::read_pod<double>(is);
        }

        em.polynomial_result_node = detail::read_pod<std::uint32_t>(is);

        auto read_pt_vec = [&]() {
            const auto n = static_cast<std::size_t>(detail::read_pod<std::uint64_t>(is));
            std::vector<poseidon::gpu::GpuPlaintextData> pts;
            pts.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
                pts.push_back(detail::load_gpu_plaintext(is, context, device_id));
            return pts;
        };
        em.chebyshev_one_plaintexts = read_pt_vec();
        em.polynomial_coefficients = read_pt_vec();
        em.double_angle_constants = read_pt_vec();
        em.double_angle_rescale_counts = detail::read_vec<std::uint32_t>(is);
        const auto req_q = detail::read_vec<std::uint64_t>(is);
        em.required_relin_q_counts.assign(req_q.begin(), req_q.end());
        out = std::move(b);
        return true;
    } catch (...) {
        return false;
    }
}

// ============================================================================
// Dev-only Key Disk Cache (Poseidon host keys + compact GPU-ready uint32 keys)
// ============================================================================
inline constexpr char kHostKeysMagic[16] = {'L','L','A','M','A','_','H','K','_','V','1',0,0,0,0,0};
inline constexpr char kFastLevelGpuMagic[16] = {'L','L','A','M','A','_','F','L','G','_','V','1',0,0,0,0};
inline constexpr char kFastBootGpuMagic[16] = {'L','L','A','M','A','_','F','B','G','_','V','1',0,0,0,0};

inline bool check_cache_file_header(const fs::path &path,
                                    const char expected_magic[16],
                                    std::uint64_t expected_fp) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    try {
        return detail::read_and_check_magic(is, expected_magic, expected_fp);
    } catch (...) {
        return false;
    }
}

namespace detail {

inline void save_gpu_ptr_table(std::ostream &os,
                               const poseidon::gpu::DeviceVector<const poseidon::gpu::GpuWord *> &table,
                               const std::vector<poseidon::gpu::GpuFieldData> &fields) {
    const std::size_t n = table.size();
    write_pod(os, static_cast<std::uint64_t>(n));
    if (n == 0) return;
    std::vector<const poseidon::gpu::GpuWord *> ptrs(n);
    table.copy_to_host(ptrs.data(), n);
    std::vector<std::uint64_t> encoded(n * 2);
    for (std::size_t k = 0; k < n; ++k) {
        if (ptrs[k] == nullptr) {
            encoded[2 * k] = std::numeric_limits<std::uint64_t>::max();
            encoded[2 * k + 1] = 0;
            continue;
        }
        const auto addr = reinterpret_cast<std::uintptr_t>(ptrs[k]);
        bool found = false;
        std::uint64_t one_past_fi = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t one_past_off = 0;
        for (std::size_t fi = 0; fi < fields.size(); ++fi) {
            if (fields[fi].empty()) continue;
            const auto begin = reinterpret_cast<std::uintptr_t>(fields[fi].data());
            const auto bytes = fields[fi].size() * sizeof(poseidon::gpu::GpuWord);
            if (addr < begin) continue;
            const auto diff = addr - begin;
            if (diff < bytes && (diff % sizeof(poseidon::gpu::GpuWord)) == 0) {
                encoded[2 * k] = static_cast<std::uint64_t>(fi);
                encoded[2 * k + 1] = static_cast<std::uint64_t>(diff / sizeof(poseidon::gpu::GpuWord));
                found = true;
                break;
            }
            if (diff == bytes) {
                one_past_fi = static_cast<std::uint64_t>(fi);
                one_past_off = static_cast<std::uint64_t>(fields[fi].size());
            }
        }
        if (!found) {
            if (one_past_fi != std::numeric_limits<std::uint64_t>::max()) {
                encoded[2 * k] = one_past_fi;
                encoded[2 * k + 1] = one_past_off;
            } else {
                throw std::runtime_error("cache: GPU key pointer outside key fields");
            }
        }
    }
    write_vec(os, encoded);
}

inline void load_gpu_ptr_table(std::istream &is,
                               int device_id,
                               const std::vector<poseidon::gpu::GpuFieldData> &fields,
                               poseidon::gpu::DeviceVector<const poseidon::gpu::GpuWord *> &table) {
    const auto n = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
    if (n == 0) return;
    const auto encoded = read_vec<std::uint64_t>(is);
    if (encoded.size() != n * 2) throw std::runtime_error("cache: GPU key pointer table size mismatch");
    std::vector<const poseidon::gpu::GpuWord *> ptrs(n, nullptr);
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint64_t fi = encoded[2 * k];
        const std::uint64_t off = encoded[2 * k + 1];
        if (fi == std::numeric_limits<std::uint64_t>::max()) {
            ptrs[k] = nullptr;
        } else {
            const auto &f = fields.at(static_cast<std::size_t>(fi));
            if (off > f.size()) throw std::runtime_error("cache: GPU key pointer offset out of bounds");
            ptrs[k] = f.data() + static_cast<std::size_t>(off);
        }
    }
    table.allocate(n, device_id);
    table.copy_from_host(ptrs.data(), n);
}

inline void save_gpu_eval_key(std::ostream &os, const poseidon::gpu::GpuEvaluationKeyData &key) {
    write_pod(os, key.meta.key_parms_id);
    write_pod(os, static_cast<std::uint64_t>(key.meta.degree));
    write_pod(os, static_cast<std::uint64_t>(key.meta.q_count));
    write_pod(os, static_cast<std::uint64_t>(key.meta.p_count));
    write_pod(os, static_cast<std::uint64_t>(key.meta.key_count));
    write_pod(os, static_cast<std::uint64_t>(key.meta.decomposition_count));
    write_pod(os, static_cast<std::uint64_t>(key.meta.component_count));
    write_pod(os, static_cast<std::uint8_t>(key.meta.galois_format));

    write_pod(os, static_cast<std::uint64_t>(key.fields_.size()));
    constexpr std::size_t kChunkWords = 16ULL << 20; // 64 MiB streaming chunk
    std::vector<poseidon::gpu::GpuWord> chunk;
    for (const auto &f : key.fields_) {
        const std::size_t total = f.size();
        write_pod(os, static_cast<std::uint64_t>(total));
        std::size_t offset = 0;
        while (offset < total) {
            const std::size_t n = std::min(kChunkWords, total - offset);
            chunk.resize(n);
            poseidon::gpu::gpu_check_cuda(
                cudaMemcpyAsync(chunk.data(), f.data() + offset, n * sizeof(poseidon::gpu::GpuWord),
                                cudaMemcpyDeviceToHost, poseidon::gpu::gpu_execution_stream()),
                "cache save gpu key D2H");
            poseidon::gpu::gpu_check_cuda(
                cudaStreamSynchronize(poseidon::gpu::gpu_execution_stream()),
                "cache save gpu key sync");
            os.write(reinterpret_cast<const char *>(chunk.data()), n * sizeof(poseidon::gpu::GpuWord));
            if (!os) throw std::runtime_error("cache: GPU key write error");
            offset += n;
        }
    }

    write_pod(os, static_cast<std::uint64_t>(key.polys_.size()));
    for (const auto &poly : key.polys_) {
        write_pod(os, static_cast<std::uint64_t>(poly.degree));
        write_pod(os, static_cast<std::uint64_t>(poly.q_count));
        write_pod(os, static_cast<std::uint64_t>(poly.p_count));
        write_pod(os, static_cast<std::uint64_t>(poly.shards.size()));
        for (const auto &sh : poly.shards) {
            write_pod(os, static_cast<std::uint64_t>(sh.field_index));
            write_pod(os, static_cast<std::uint64_t>(sh.field_offset));
            write_pod(os, static_cast<std::uint64_t>(sh.limb_begin));
            write_pod(os, static_cast<std::uint64_t>(sh.limb_count));
            write_pod(os, static_cast<std::uint64_t>(sh.coeff_begin));
            write_pod(os, static_cast<std::uint64_t>(sh.coeff_count));
        }
    }

    write_vec(os, key.poly_metadata_);
    write_vec(os, key.galois_elts_by_key_index);
    save_gpu_ptr_table(os, key.galois_key_q0_ptrs, key.fields_);
    save_gpu_ptr_table(os, key.galois_key_p0_ptrs, key.fields_);
    save_gpu_ptr_table(os, key.galois_key_q1_ptrs, key.fields_);
    save_gpu_ptr_table(os, key.galois_key_p1_ptrs, key.fields_);
}

inline poseidon::gpu::GpuEvaluationKeyData load_gpu_eval_key(std::istream &is, int device_id) {
    poseidon::gpu::GpuEvaluationKeyData key;
    key.meta.key_parms_id = read_pod<poseidon::parms_id_type>(is);
    key.meta.degree = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
    key.meta.q_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
    key.meta.p_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
    key.meta.key_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
    key.meta.decomposition_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
    key.meta.component_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
    key.meta.galois_format = static_cast<poseidon::gpu::GpuGaloisKeyFormat>(read_pod<std::uint8_t>(is));

    const auto n_fields = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
    key.fields_.reserve(n_fields);
    constexpr std::size_t kChunkWords = 16ULL << 20; // 64 MiB streaming pinned chunk
    poseidon::gpu::GpuWord *pinned = nullptr;
    struct PinnedGuard {
        poseidon::gpu::GpuWord **p;
        ~PinnedGuard() { if (*p) cudaFreeHost(*p); }
    } guard{&pinned};
    poseidon::gpu::gpu_check_cuda(
        cudaHostAlloc(&pinned, kChunkWords * sizeof(poseidon::gpu::GpuWord), cudaHostAllocDefault),
        "cache load gpu key pinned alloc");

    for (std::size_t fi = 0; fi < n_fields; ++fi) {
        const auto total = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
        key.fields_.emplace_back(device_id, total);
        auto &f = key.fields_.back();
        std::size_t offset = 0;
        while (offset < total) {
            const std::size_t n = std::min(kChunkWords, total - offset);
            is.read(reinterpret_cast<char *>(pinned), n * sizeof(poseidon::gpu::GpuWord));
            if (!is) throw std::runtime_error("cache: GPU key read error");
            poseidon::gpu::gpu_check_cuda(
                cudaMemcpyAsync(f.data() + offset, pinned, n * sizeof(poseidon::gpu::GpuWord),
                                cudaMemcpyHostToDevice, poseidon::gpu::gpu_execution_stream()),
                "cache load gpu key H2D");
            poseidon::gpu::gpu_check_cuda(
                cudaStreamSynchronize(poseidon::gpu::gpu_execution_stream()),
                "cache load gpu key sync");
            offset += n;
        }
    }

    const auto n_polys = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
    key.polys_.resize(n_polys);
    for (auto &poly : key.polys_) {
        poly.degree = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
        poly.q_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
        poly.p_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
        const auto n_shards = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
        poly.shards.resize(n_shards);
        for (auto &sh : poly.shards) {
            sh.field_index = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
            sh.field_offset = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
            sh.limb_begin = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
            sh.limb_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
            sh.coeff_begin = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
            sh.coeff_count = static_cast<std::size_t>(read_pod<std::uint64_t>(is));
        }
    }

    key.poly_metadata_ = read_vec<poseidon::gpu::GpuEvaluationKeyPolyMeta>(is);
    key.galois_elts_by_key_index = read_vec<std::uint32_t>(is);
    load_gpu_ptr_table(is, device_id, key.fields_, key.galois_key_q0_ptrs);
    load_gpu_ptr_table(is, device_id, key.fields_, key.galois_key_p0_ptrs);
    load_gpu_ptr_table(is, device_id, key.fields_, key.galois_key_q1_ptrs);
    load_gpu_ptr_table(is, device_id, key.fields_, key.galois_key_p1_ptrs);
    return key;
}

} // namespace detail

inline void save_host_keys_file(const fs::path &path,
                                std::uint64_t fp,
                                const poseidon::SecretKey &sk,
                                const poseidon::PublicKey &pk,
                                const poseidon::RelinKeys &rk,
                                const poseidon::GaloisKeys &gk) {
    AtomicWriter writer(path);
    auto &os = writer.stream();
    detail::write_magic(os, kHostKeysMagic, fp);
    sk.save(os, poseidon::compr_mode_type::none);
    pk.save(os, poseidon::compr_mode_type::none);
    rk.save(os, poseidon::compr_mode_type::none);
    gk.save(os, poseidon::compr_mode_type::none);
    writer.commit();
}

inline bool load_host_keys_file(const fs::path &path,
                                std::uint64_t fp,
                                const poseidon::PoseidonContext &ctx,
                                poseidon::SecretKey &sk,
                                poseidon::PublicKey &pk,
                                poseidon::RelinKeys &rk,
                                poseidon::GaloisKeys &gk) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    try {
        if (!detail::read_and_check_magic(is, kHostKeysMagic, fp)) return false;
        poseidon::SecretKey sk_tmp;
        poseidon::PublicKey pk_tmp;
        poseidon::RelinKeys rk_tmp;
        poseidon::GaloisKeys gk_tmp;
        sk_tmp.load(ctx, is);
        pk_tmp.load(ctx, is);
        rk_tmp.load(ctx, is);
        gk_tmp.load(ctx, is);
        sk = std::move(sk_tmp);
        pk = std::move(pk_tmp);
        rk = std::move(rk_tmp);
        gk = std::move(gk_tmp);
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace cerium_source::cache
