#include "prepared_qk.h"
#include "qk_reference.h"
#include "qk_key_plan.h"

#include "poseidon/ckks_encoder.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/gpu/kernels/gpu_keyswitch_kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace bert::attention {
namespace {
using namespace poseidon;
using namespace poseidon::gpu;
using Clock = std::chrono::steady_clock;

// Declare this AFTER every local GPU owner / asynchronous host source. It
// synchronizes before those owners unwind, still inside the stream scope.
struct FailureSync {
    cudaStream_t stream;
    int exceptions = std::uncaught_exceptions();
    ~FailureSync() {
        if (std::uncaught_exceptions() > exceptions)
            (void)cudaStreamSynchronize(stream);
    }
};

const GpuWord *component(const GpuCiphertextData &ct, std::size_t c, int device) {
    if (ct.polys_.size() != 2 || ct.meta.component_count != 2 ||
        ct.meta.p_count || !ct.meta.is_ntt_form)
        throw std::invalid_argument("prepared QK requires size2 Q-only NTT ciphertexts");
    const auto &poly = ct.polys_.at(c);
    if (poly.degree != ct.meta.degree || poly.q_count != ct.meta.q_count ||
        poly.p_count || poly.shards.size() != 1)
        throw std::invalid_argument("prepared QK requires one full shard per component");
    const auto &shard = poly.shards.front();
    const auto &field = ct.fields_.at(shard.field_index);
    if (field.device_id != device || shard.limb_begin || shard.coeff_begin ||
        shard.limb_count != ct.meta.q_count || shard.coeff_count != ct.meta.degree ||
        shard.field_offset + ct.meta.q_count * ct.meta.degree > field.size())
        throw std::invalid_argument("prepared QK ciphertext shard mismatch");
    return field.data() + shard.field_offset;
}
GpuWord *component(GpuCiphertextData &ct, std::size_t c, int device) {
    return const_cast<GpuWord *>(component(static_cast<const GpuCiphertextData &>(ct), c, device));
}

std::shared_ptr<const CrtContext::ContextData> level_data(
        const PoseidonContext &context, std::size_t level) {
    for (auto data = context.crt_context()->first_context_data(); data;
         data = data->next_context_data())
        if (data->parms().p().empty() && data->parms().q().size() == level + 1)
            return data;
    throw std::invalid_argument("QkSpec physical level is absent from context");
}

std::uint64_t key_bytes(const GpuEvaluationKeyData &keys) {
    std::uint64_t bytes = 0;
    for (const auto &field : keys.fields_) bytes += field.size() * sizeof(GpuWord);
    return bytes + (keys.galois_key_q0_ptrs.size() + keys.galois_key_q1_ptrs.size() +
        keys.galois_key_p0_ptrs.size() + keys.galois_key_p1_ptrs.size()) * sizeof(const GpuWord *);
}

// Equivalent to the mature uploader's inverse pre-rotation, but with an INNER
// failure guard: its stock helper can unwind a local temporary after queued
// permutations if a later allocation throws, before a caller can synchronize.
// Keep this bench-owned hardening local; do not change the frozen core library.
void pre_rotate_keys_safely(GpuGaloisKeysData &keys, int device, cudaStream_t stream) {
    const auto words = (keys.meta.q_count + keys.meta.p_count) * keys.meta.degree;
    const auto pointers = keys.meta.key_count * keys.meta.decomposition_count;
    DeviceVector<GpuWord> temporary;
    std::vector<const GpuWord *> q0(pointers, nullptr), p0(pointers, nullptr);
    std::vector<const GpuWord *> q1(pointers, nullptr), p1(pointers, nullptr);
    auto destination = keys.make_view();
    const auto source = keys.make_const_view();
    FailureSync failure{stream};
    keys.galois_elts_by_key_index.assign(keys.meta.key_count, 0);
    temporary.allocate(words, device);
    // All potentially throwing host/device allocations precede permutations.
    keys.galois_key_q0_ptrs.allocate(pointers, device);
    keys.galois_key_p0_ptrs.allocate(pointers, device);
    keys.galois_key_q1_ptrs.allocate(pointers, device);
    keys.galois_key_p1_ptrs.allocate(pointers, device);
    const std::uint64_t modulus = static_cast<std::uint64_t>(keys.meta.degree) << 1;
    for (const auto &metadata : keys.poly_metadata_) {
        const auto element = static_cast<std::uint32_t>(2 * metadata.key_index + 1);
        keys.galois_elts_by_key_index.at(metadata.key_index) = element;
        std::uint64_t inverse = 1;
        for (int iteration = 0; iteration < 6; ++iteration) {
            inverse *= 2 - static_cast<std::uint64_t>(element) * inverse;
            inverse &= modulus - 1;
        }
        if (((static_cast<std::uint64_t>(element) * inverse) & (modulus - 1)) != 1)
            throw std::logic_error("QK inverse automorphism failed");
        const auto &src = source.polys.at(metadata.poly_id).shards.front();
        auto &dst = destination.polys.at(metadata.poly_id).shards.front();
        GpuPolyShardView scratch{device, temporary.data(), 0,
            keys.meta.q_count + keys.meta.p_count, 0, keys.meta.degree};
        kernel::launch_apply_galois_ntt_poly_shard(scratch, src,
            static_cast<std::uint32_t>(inverse), keys.meta.degree);
        gpu_check_cuda(cudaMemcpyAsync(dst.ptr, temporary.data(), words * sizeof(GpuWord),
            cudaMemcpyDeviceToDevice, stream), "QK inverse pre-rotate key copy");
        const auto flat = metadata.key_index * keys.meta.decomposition_count + metadata.decomposition_index;
        if (metadata.component_index == 0) {
            q0.at(flat) = src.ptr;
            p0.at(flat) = src.ptr + keys.meta.q_count * keys.meta.degree;
        } else if (metadata.component_index == 1) {
            q1.at(flat) = src.ptr;
            p1.at(flat) = src.ptr + keys.meta.q_count * keys.meta.degree;
        }
    }
    keys.galois_key_q0_ptrs.copy_from_host(q0.data(), pointers);
    keys.galois_key_p0_ptrs.copy_from_host(p0.data(), pointers);
    keys.galois_key_q1_ptrs.copy_from_host(q1.data(), pointers);
    keys.galois_key_p1_ptrs.copy_from_host(p1.data(), pointers);
    keys.meta.galois_format = GpuGaloisKeyFormat::InversePreRotated;
}

reference::ScalePlan validate_spec(const QkSpec &s, const PoseidonContext &context) {
    if (s.pair_products && s.pack_inputs)
        throw std::invalid_argument("QK pair_products and pack_inputs are mutually exclusive");
    if (s.id.empty() || s.tokens != 128 || s.tile != 256 || s.head_dim != 64 || s.baby != 8 ||
        context.parameters_literal()->degree() != 65536 || context.parameters_literal()->slot() != 32768)
        throw std::invalid_argument("prepared QK requires id and N65536/T128/tile256/D64/baby8");
    std::vector<std::uint64_t> q;
    for (const auto &prime : context.parameters_literal()->q()) {
        if (prime.value() > std::numeric_limits<GpuWord>::max())
            throw std::invalid_argument("prepared QK requires <=32-bit Q primes");
        q.push_back(prime.value());
    }
    if (context.parameters_literal()->p().empty())
        throw std::invalid_argument("prepared QK requires HYBRID special primes");
    for (const auto &prime : context.parameters_literal()->p())
        if (prime.value() > std::numeric_limits<GpuWord>::max())
            throw std::invalid_argument("prepared QK requires <=32-bit P primes");
    const auto plan = reference::derive_scale_plan(s, q);
    level_data(context, s.input_level);
    level_data(context, s.input_level - 2);
    level_data(context, s.output_level);
    return plan;
}
} // namespace

struct PreparedQkEngine::Impl {
    struct Prepared {
        QkSpec spec;
        QkStats stats;
        GpuPlaintextData mask;
        GpuPlaintextData gap_mask; // paired-only starts at64/192 in each256 row
    };
    struct Runtime {
        GpuParameterData parameters;
        GpuEvaluator evaluator;
        GpuRotateWorkspace rotate_workspace;
        GpuDoubleHoistWorkspace hoist_workspace;
        Runtime(const PoseidonContext &context, int device, std::size_t qin, std::size_t qout)
            : parameters(context, device, counts(qin, qout)), evaluator(parameters) {}
        static std::vector<std::size_t> counts(std::size_t qin, std::size_t qout) {
            std::vector<std::size_t> result(qin - qout + 1);
            std::iota(result.begin(), result.end(), qout);
            return result;
        }
    };
    struct RunStorage {
        // Every ordinary evaluator output has its own, distinct owner. In
        // particular add(a,b,a), multiply/relinearize/rescale aliases are NOT
        // allowed. Swaps implement safe accumulator ping-pong.
        std::vector<GpuCiphertextData> queries, keys;
        GpuCiphertextData product, relinearized, reduced, rotated, add_scratch;
        GpuCiphertextData masked, group, total, result;
        GpuCiphertextData other_reduced, combined, other_masked; // shared-reduction mode owners
        GpuCiphertextData packed_query, packed_key; // PRE-CC sources; immutable during their own hoists
    };

    PoseidonContext context;
    std::shared_ptr<const RelinKeys> cpu_relin;
    std::shared_ptr<const GaloisKeys> cpu_galois;
    int device;
    cudaStream_t stream = nullptr;
    std::map<std::string, Prepared> prepared;
    std::map<std::pair<std::size_t, std::size_t>, std::unique_ptr<Runtime>> runtimes;
    std::unique_ptr<GpuRelinKeysData> relin;
    std::unique_ptr<GpuGaloisKeysData> galois;
    std::map<int, std::vector<int>> rotation_paths;
    std::set<int> prepared_rotation_steps;
    bool has_run = false;
    EngineStats stats;

    Impl(const PoseidonContext &ctx, std::shared_ptr<const RelinKeys> r,
         std::shared_ptr<const GaloisKeys> g, int dev)
        : context(ctx), cpu_relin(std::move(r)), cpu_galois(std::move(g)), device(dev) {}
    ~Impl() {
        if (!stream) return;
        (void)cudaSetDevice(device);
        {
            ScopedGpuExecutionStream guard(stream);
            (void)cudaStreamSynchronize(stream);
            prepared.clear(); runtimes.clear(); galois.reset(); relin.reset();
        }
        (void)cudaStreamDestroy(stream);
    }
    std::uint32_t elt(int step) const {
        return context.crt_context()->galois_tool()->get_elt_from_step(step);
    }
    std::vector<int> path(int step) const {
        if (cpu_galois->has_key(elt(step))) return {step};
        std::vector<int> result;
        const int sign = step < 0 ? -1 : 1;
        unsigned value = static_cast<unsigned>(step < 0 ? -step : step);
        for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
            if (value & 1) result.push_back(sign * static_cast<int>(bit));
        if (std::all_of(result.begin(), result.end(), [&](int s) { return cpu_galois->has_key(elt(s)); }))
            return result;
        result.clear();
        const auto slots = static_cast<int>(context.parameters_literal()->slot());
        value = static_cast<unsigned>((step % slots + slots) % slots);
        for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
            if (value & 1) {
                if (!cpu_galois->has_key(elt(static_cast<int>(bit))))
                    throw std::invalid_argument("missing direct/binary prepared QK rotation " + std::to_string(step));
                result.push_back(static_cast<int>(bit));
            }
        return result;
    }
    void update_bytes() {
        stats.allocated_bytes = stats.parameter_bytes + stats.galois_key_bytes +
                                stats.relin_key_bytes + stats.mask_bytes;
    }
    void ensure_gpu(std::size_t requested_q, bool paired, bool packed_inputs) {
        gpu_check_cuda(cudaSetDevice(device), "QK prepare select device");
        auto plan = detail::plan_key_preparation(stats.key_storage_q_count, prepared_rotation_steps,
            requested_q, packed_inputs ? PreparedQkEngine::required_packed_input_rotation_steps() :
                         paired ? PreparedQkEngine::required_paired_rotation_steps() :
                                  PreparedQkEngine::required_rotation_steps());
        if (galois && !plan.needs_upload) return;
        // A new mode may need new keys even when its Q is LOWER than the old
        // maximum. Never early-return merely on Q, and never shrink that owner.
        const auto required_q = plan.storage_q_count;
        if (!cpu_relin || !cpu_galois ||
            cpu_relin->parms_id() != context.crt_context()->key_parms_id() ||
            cpu_galois->parms_id() != context.crt_context()->key_parms_id())
            throw std::invalid_argument("prepared QK requires matching CPU relin/Galois keys");
        if (!stream)
            gpu_check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "QK stream create");
        ScopedGpuExecutionStream guard(stream);
        std::unique_ptr<GpuRelinKeysData> new_relin;
        std::unique_ptr<GpuGaloisKeysData> new_galois;
        FailureSync failure{stream};
        std::set<std::uint32_t> elements;
        std::map<int, std::vector<int>> paths;
        for (int step : plan.rotation_steps) {
            auto parts = path(step);
            for (int part : parts) elements.insert(elt(part));
            paths.emplace(step, std::move(parts));
        }
        // Genuine owning crop, including HYBRID digits, while preserving ALL
        // context P. Lower runtime views never pretend to crop that storage.
        new_relin = std::make_unique<GpuRelinKeysData>();
        new_galois = std::make_unique<GpuGaloisKeysData>();
        *new_relin = GpuUploader::upload_relin_keys(*cpu_relin, device, required_q);
        *new_galois = GpuUploader::upload_galois_keys(*cpu_galois, device, required_q,
            std::vector<std::uint32_t>(elements.begin(), elements.end()));
        pre_rotate_keys_safely(*new_galois, device, stream);
        const auto np = context.parameters_literal()->p().size();
        if (new_relin->meta.q_count != required_q || new_galois->meta.q_count != required_q ||
            new_relin->meta.p_count != np || new_galois->meta.p_count != np)
            throw std::runtime_error("prepared QK owning keys did not retain requested Q and full P");
        for (const auto &entry : runtimes) {
            const auto qi = entry.first.first, qm = qi - 2;
            GpuUploader::prepare_key_views_for_q_counts(*new_relin, {qi});
            GpuUploader::prepare_key_views_for_q_counts(*new_galois, {qi, qm});
        }
        gpu_check_cuda(cudaStreamSynchronize(stream), "QK key setup synchronize");
        relin.swap(new_relin); galois.swap(new_galois); rotation_paths.swap(paths);
        prepared_rotation_steps.swap(plan.rotation_steps);
        ++stats.key_uploads;
        stats.uploaded_rotation_steps = elements.size();
        stats.key_storage_q_count = required_q;
        stats.key_storage_p_count = np;
        stats.key_storage_is_full_q = required_q == context.parameters_literal()->q().size();
        stats.relin_key_bytes = key_bytes(*relin);
        stats.galois_key_bytes = key_bytes(*galois);
        update_bytes();
    }
    Runtime &runtime(const QkSpec &s) {
        const auto key = std::make_pair(s.input_level + 1, s.output_level + 1);
        auto it = runtimes.find(key);
        if (it == runtimes.end()) {
            auto r = std::make_unique<Runtime>(context, device, key.first, key.second);
            GpuUploader::prepare_key_views_for_q_counts(*relin, {key.first});
            GpuUploader::prepare_key_views_for_q_counts(*galois, {key.first, key.first - 2});
            const auto bytes = r->parameters.allocated_bytes();
            it = runtimes.emplace(key, std::move(r)).first;
            stats.parameter_bytes += bytes;
            update_bytes();
        }
        return *it->second;
    }
    void clone_into(const GpuCiphertextData &source, GpuCiphertextData &result) {
        if (&source == &result) throw std::logic_error("QK clone alias");
        if (result.meta.degree != source.meta.degree || result.meta.q_count != source.meta.q_count ||
            result.meta.component_count != 2)
            result = GpuCiphertextData::allocate_single_device(source.meta.degree, source.meta.q_count, 2, device);
        result.meta = source.meta;
        for (std::size_t c = 0; c < 2; ++c)
            gpu_check_cuda(cudaMemcpyAsync(component(result, c, device), component(source, c, device),
                source.meta.degree * source.meta.q_count * sizeof(GpuWord), cudaMemcpyDeviceToDevice, stream), "QK clone");
    }
    void rotate_inplace(Runtime &r, GpuCiphertextData &ct, int step) {
        for (int part : rotation_paths.at(step)) {
            r.evaluator.rotate_inplace(ct, part, *galois, r.rotate_workspace);
            ++stats.rotation_calls;
        }
    }
    void rotate_into(Runtime &r, const GpuCiphertextData &source, int step, GpuCiphertextData &result) {
        // Use the specialized in-place implementation on an independent copy:
        // unlike ordinary rotate(), this retains caller-owned KS workspaces
        // across all 768 changing-source head-reduction rotations. No false
        // same-source cache, temporary core workspace, or per-rotation sync.
        clone_into(source, result);
        rotate_inplace(r, result, step);
    }
    void shared_rotations(Runtime &r, const GpuCiphertextData &source,
                          const std::vector<int> &steps, std::vector<GpuCiphertextData> &out) {
        const bool direct = std::all_of(steps.begin(), steps.end(), [&](int s) {
            return rotation_paths.at(s).size() == 1 && rotation_paths.at(s).front() == s;
        });
        if (direct) {
            r.evaluator.rotate_many_hoisted(source, steps, *galois, r.hoist_workspace, out);
            stats.rotation_calls += steps.size();
            ++stats.hoisted_rotation_batches;
        } else {
            out.resize(steps.size());
            for (std::size_t i = 0; i < steps.size(); ++i) rotate_into(r, source, steps[i], out[i]);
        }
    }
    void add_into(Runtime &r, GpuCiphertextData &accumulator,
                  const GpuCiphertextData &term, GpuCiphertextData &scratch) {
        r.evaluator.add(accumulator, term, scratch);
        std::swap(accumulator, scratch);
        ++stats.add_calls;
    }
    void require_meta(const GpuCiphertextData &ct, Runtime &r, std::size_t physical_level,
                      double scale_log2, const char *stage) const {
        const auto &level = r.parameters.get_level_by_q_count(physical_level + 1);
        if (ct.meta.degree != level.degree || ct.meta.q_count != level.q_count ||
            ct.meta.parms_id != level.parms_id || !(ct.meta.scale > 0) ||
            !std::isfinite(ct.meta.scale) || std::abs(std::log2(ct.meta.scale) - scale_log2) > 1e-7)
            throw std::runtime_error(std::string("prepared QK physical level/scale mismatch at ") + stage);
        component(ct, 0, device); component(ct, 1, device);
    }
};

PreparedQkEngine::PreparedQkEngine(const PoseidonContext &context,
        std::shared_ptr<const RelinKeys> relin, std::shared_ptr<const GaloisKeys> galois, int device)
    : impl_(std::make_unique<Impl>(context, std::move(relin), std::move(galois), device)) {}
PreparedQkEngine::~PreparedQkEngine() = default;

std::vector<int> PreparedQkEngine::required_rotation_steps() {
    return detail::qk_rotation_steps(false);
}
std::vector<int> PreparedQkEngine::required_binary_rotation_steps() {
    return detail::qk_binary_rotation_steps(false);
}
std::vector<int> PreparedQkEngine::required_paired_rotation_steps() {
    return detail::qk_rotation_steps(true);
}
std::vector<int> PreparedQkEngine::required_paired_binary_rotation_steps() {
    return detail::qk_binary_rotation_steps(true);
}
std::vector<int> PreparedQkEngine::required_packed_input_rotation_steps() {
    return detail::qk_rotation_steps(false, true);
}
std::vector<int> PreparedQkEngine::required_packed_input_binary_rotation_steps() {
    return detail::qk_binary_rotation_steps(false, true);
}

void PreparedQkEngine::prepare(const QkSpec &s) {
    const auto begin = Clock::now();
    auto &p = *impl_;
    const auto plan = validate_spec(s, p.context);
    if (p.has_run) throw std::logic_error("prepare all QK specs before the first run");
    if (p.prepared.count(s.id)) throw std::invalid_argument("duplicate prepared QK id " + s.id);
    p.ensure_gpu(s.input_level + 1, s.pair_products, s.pack_inputs);
    ScopedGpuExecutionStream guard(p.stream);
    Impl::Prepared item;
    Plaintext encoded;
    std::vector<double> slots(s.tokens * s.tile, 0.0);
    FailureSync failure{p.stream};
    item.spec = s;
    item.stats.id = s.id;
    item.stats.pair_products = s.pair_products;
    item.stats.pack_inputs = s.pack_inputs;
    if (s.pack_inputs) {
        item.stats.schedule = "packed_input2_pre_cc_gap";
        item.stats.encoding = "full_ckks_dual_head_start_masks";
    } else if (s.pair_products) {
        item.stats.schedule = "paired2_gap_shared_headreduce";
        item.stats.encoding = "full_ckks_dual_head_start_masks";
    }
    item.stats.mask_scale_log2 = plan.mask_scale_log2;
    (void)p.runtime(s); // every physical Q count from output+1 through input+1
    const auto middle = level_data(p.context, s.input_level - 2);
    for (std::size_t i = 0; i < slots.size(); i += s.tokens)
        slots[i] = 1.0 / std::sqrt(static_cast<double>(s.head_dim));
    CKKSEncoder encoder(p.context);
    encoder.encode(slots, middle->parms_id(), std::exp2(plan.mask_scale_log2), encoded);
    if (!std::isfinite(encoded.scale()) || encoded.scale() <= 1.0 ||
        std::abs(std::log2(encoded.scale()) - plan.mask_scale_log2) > 1e-10)
        throw std::runtime_error("QK real mask encoding scale differs from derived physical scale");
    item.mask = GpuUploader::upload_plaintext(encoded, p.device);
    for (const auto &field : item.mask.fields_) item.stats.mask_bytes += field.size() * sizeof(GpuWord);
    if (s.pair_products || s.pack_inputs) {
        std::fill(slots.begin(), slots.end(), 0.0);
        for (std::size_t i = s.head_dim; i < slots.size(); i += s.tokens)
            slots[i] = 1.0 / std::sqrt(static_cast<double>(s.head_dim));
        encoder.encode(slots, middle->parms_id(), std::exp2(plan.mask_scale_log2), encoded);
        if (!std::isfinite(encoded.scale()) || encoded.scale() <= 1.0 ||
            std::abs(std::log2(encoded.scale()) - plan.mask_scale_log2) > 1e-10)
            throw std::runtime_error("QK paired gap-mask encoding scale mismatch");
        item.gap_mask = GpuUploader::upload_plaintext(encoded, p.device);
        for (const auto &field : item.gap_mask.fields_) item.stats.mask_bytes += field.size() * sizeof(GpuWord);
    }
    gpu_check_cuda(cudaStreamSynchronize(p.stream), "QK mask/parameter setup synchronize");
    item.stats.setup_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    const auto bytes = item.stats.mask_bytes;
    const auto seconds = item.stats.setup_seconds;
    p.prepared.emplace(s.id, std::move(item));
    p.stats.mask_bytes += bytes;
    p.stats.setup_seconds += seconds;
    p.update_bytes();
}

GpuCiphertextData PreparedQkEngine::run(const std::string &id,
        const GpuCiphertextData &query, const GpuCiphertextData &key) {
    auto &p = *impl_;
    const auto &m = p.prepared.at(id);
    const auto &s = m.spec;
    gpu_check_cuda(cudaSetDevice(p.device), "QK run select device");
    ScopedGpuExecutionStream guard(p.stream);
    Impl::RunStorage w;
    FailureSync failure{p.stream};
    auto &r = *p.runtimes.at({s.input_level + 1, s.output_level + 1});
    p.require_meta(query, r, s.input_level, s.input_scale_log2, "query input");
    p.require_meta(key, r, s.input_level, s.input_scale_log2, "key input");
    if (std::abs(std::log2(query.meta.scale / key.meta.scale)) > 1e-10)
        throw std::invalid_argument("prepared QK query/key actual scales differ");
    p.has_run = true;
    if (s.pack_inputs) {
        // PRE-CC SIMD packing, using only the IDEAL message's zero gaps.
        // Keep raw caller inputs immutable and own the true packed sources.
        p.rotate_into(r, query, -static_cast<int>(s.head_dim), w.rotated);
        r.evaluator.add(query, w.rotated, w.packed_query); ++p.stats.add_calls;
        p.rotate_into(r, key, static_cast<int>(s.tile - s.head_dim), w.rotated);
        r.evaluator.add(key, w.rotated, w.packed_key); ++p.stats.add_calls;
        p.require_meta(w.packed_query, r, s.input_level, s.input_scale_log2, "packed query add");
        p.require_meta(w.packed_key, r, s.input_level, s.input_scale_log2, "packed key add");
    }
    const auto &query_source = s.pack_inputs ? w.packed_query : query;
    const auto &key_source = s.pack_inputs ? w.packed_key : key;
    std::vector<int> query_steps, key_steps;
    for (std::size_t g = 1; g < s.tokens / s.baby; ++g)
        query_steps.push_back(-static_cast<int>(s.tile * s.baby * g));
    const std::size_t key_stride = s.pack_inputs ? 2 : 1;
    for (std::size_t b = key_stride; b < s.baby; b += key_stride)
        key_steps.push_back(static_cast<int>(s.tile * b));
    // Exactly two same-source hoist islands. All head-reduction sources below
    // CHANGE after each add and therefore must never reuse a source hoist.
    // In packed-input mode these hoists decompose the ACTUAL packed Q/K,
    // never a falsely reused decomposition of the corresponding raw input.
    p.shared_rotations(r, query_source, query_steps, w.queries);
    p.shared_rotations(r, key_source, key_steps, w.keys);
    bool have_total = false;
    for (std::size_t g = 0; g < s.tokens / s.baby; ++g) {
        const auto &q = g ? w.queries[g - 1] : query_source;
        bool have_group = false;
        if (s.pack_inputs) {
            for (std::size_t b = 0; b < s.baby; b += 2) {
                const auto &k = b ? w.keys[b / 2 - 1] : key_source;
                // One real CC occupies all four D64 blocks and contains TWO
                // distinct baby products. All128 key offsets/heads remain.
                r.evaluator.multiply(q, k, w.product); ++p.stats.cc_multiply_calls;
                r.evaluator.relinearize(w.product, *p.relin, w.relinearized); ++p.stats.relinearize_calls;
                r.evaluator.rescale_many(w.relinearized, w.combined, 2);
                ++p.stats.rescale2_calls; p.stats.physical_rescale_primes += 2;
                p.require_meta(w.combined, r, s.input_level - 2, s.intermediate_scale_log2, "packed CC Rescale2");
                for (std::size_t shift = 1; shift < s.head_dim; shift *= 2) {
                    p.rotate_into(r, w.combined, static_cast<int>(shift), w.rotated);
                    p.add_into(r, w.combined, w.rotated, w.add_scratch);
                }
                r.evaluator.multiply_plain(w.combined, m.mask, w.masked); ++p.stats.cp_multiply_calls;
                r.evaluator.multiply_plain(w.combined, m.gap_mask, w.other_masked); ++p.stats.cp_multiply_calls;
                p.rotate_inplace(r, w.other_masked, static_cast<int>(s.head_dim - 1));
                p.add_into(r, w.masked, w.other_masked, w.add_scratch);
                if (b) p.rotate_inplace(r, w.masked, -static_cast<int>(b));
                if (!have_group) { std::swap(w.group, w.masked); have_group = true; }
                else p.add_into(r, w.group, w.masked, w.add_scratch);
            }
        } else if (s.pair_products) {
            // Ideal-message head gaps MUST be zero: P1 is shifted into P0's
            // spare64 lanes. Ciphertext noise/encoding gaps are not bitwise0,
            // so this changes their propagation; no decrypt check/refresh.
            for (std::size_t b = 0; b < s.baby; b += 2) {
                const auto &k0 = b ? w.keys[b - 1] : key;
                const auto &k1 = w.keys[b];
                r.evaluator.multiply(q, k0, w.product); ++p.stats.cc_multiply_calls;
                r.evaluator.relinearize(w.product, *p.relin, w.relinearized); ++p.stats.relinearize_calls;
                r.evaluator.rescale_many(w.relinearized, w.reduced, 2);
                ++p.stats.rescale2_calls; p.stats.physical_rescale_primes += 2;
                p.require_meta(w.reduced, r, s.input_level - 2, s.intermediate_scale_log2, "paired P0 Rescale2");
                r.evaluator.multiply(q, k1, w.product); ++p.stats.cc_multiply_calls;
                r.evaluator.relinearize(w.product, *p.relin, w.relinearized); ++p.stats.relinearize_calls;
                r.evaluator.rescale_many(w.relinearized, w.other_reduced, 2);
                ++p.stats.rescale2_calls; p.stats.physical_rescale_primes += 2;
                p.require_meta(w.other_reduced, r, s.input_level - 2, s.intermediate_scale_log2, "paired P1 Rescale2");
                p.rotate_inplace(r, w.other_reduced, -static_cast<int>(s.head_dim));
                r.evaluator.add(w.reduced, w.other_reduced, w.combined); ++p.stats.add_calls;
                for (std::size_t shift = 1; shift < s.head_dim; shift *= 2) {
                    p.rotate_into(r, w.combined, static_cast<int>(shift), w.rotated);
                    p.add_into(r, w.combined, w.rotated, w.add_scratch);
                }
                // TWO genuine encoded-mask CP products, not a selector or
                // metadata trick. Their scales are identical and physical.
                r.evaluator.multiply_plain(w.combined, m.mask, w.masked); ++p.stats.cp_multiply_calls;
                r.evaluator.multiply_plain(w.combined, m.gap_mask, w.other_masked); ++p.stats.cp_multiply_calls;
                p.rotate_inplace(r, w.other_masked, static_cast<int>(s.head_dim - 1));
                p.add_into(r, w.masked, w.other_masked, w.add_scratch);
                if (b) p.rotate_inplace(r, w.masked, -static_cast<int>(b));
                if (!have_group) { std::swap(w.group, w.masked); have_group = true; }
                else p.add_into(r, w.group, w.masked, w.add_scratch);
            }
        } else {
            // Original baseline arithmetic and call order intentionally intact.
            for (std::size_t b = 0; b < s.baby; ++b) {
                const auto &k = b ? w.keys[b - 1] : key;
                r.evaluator.multiply(q, k, w.product); ++p.stats.cc_multiply_calls;
                r.evaluator.relinearize(w.product, *p.relin, w.relinearized); ++p.stats.relinearize_calls;
                r.evaluator.rescale_many(w.relinearized, w.reduced, 2);
                ++p.stats.rescale2_calls; p.stats.physical_rescale_primes += 2;
                p.require_meta(w.reduced, r, s.input_level - 2, s.intermediate_scale_log2, "CC/relin/Rescale2");
                for (std::size_t shift = 1; shift < s.head_dim; shift *= 2) {
                    p.rotate_into(r, w.reduced, static_cast<int>(shift), w.rotated);
                    p.add_into(r, w.reduced, w.rotated, w.add_scratch);
                }
                r.evaluator.multiply_plain(w.reduced, m.mask, w.masked); ++p.stats.cp_multiply_calls;
                if (b) p.rotate_inplace(r, w.masked, -static_cast<int>(b));
                if (!have_group) { std::swap(w.group, w.masked); have_group = true; }
                else p.add_into(r, w.group, w.masked, w.add_scratch);
            }
        }
        if (g) p.rotate_inplace(r, w.group, static_cast<int>((s.tile * s.baby - s.baby) * g));
        if (!have_total) { std::swap(w.total, w.group); have_total = true; }
        else p.add_into(r, w.total, w.group, w.add_scratch);
    }
    // One final logical Rescale2 AFTER accumulating every giant, not per mask.
    r.evaluator.rescale_many(w.total, w.result, 2);
    ++p.stats.rescale2_calls; p.stats.physical_rescale_primes += 2;
    p.require_meta(w.result, r, s.output_level, s.output_scale_log2, "final Rescale2");
    gpu_check_cuda(cudaStreamSynchronize(p.stream), "QK pair output synchronize");
    ++p.stats.run_calls;
    if (s.pack_inputs) ++p.stats.packed_input_run_calls;
    else if (s.pair_products) ++p.stats.paired_run_calls;
    else ++p.stats.baseline_run_calls;
    return std::move(w.result);
}

const QkStats &PreparedQkEngine::qk_stats(const std::string &id) const {
    return impl_->prepared.at(id).stats;
}
EngineStats PreparedQkEngine::engine_stats() const { return impl_->stats; }

} // namespace bert::attention
