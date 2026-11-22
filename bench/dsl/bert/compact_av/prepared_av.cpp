#include "prepared_av.h"
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
#include <tuple>
#include <utility>

namespace bert::attention {
namespace {
using namespace poseidon;
using namespace poseidon::gpu;
using Clock = std::chrono::steady_clock;
// Declare AFTER GPU owners / async host sources, inside the stream scope.
struct FailureSync {
    cudaStream_t stream;
    int exceptions = std::uncaught_exceptions();
    ~FailureSync() {
        if (std::uncaught_exceptions() > exceptions) (void)cudaStreamSynchronize(stream);
    }
};
const GpuWord *component(const GpuCiphertextData &ct, std::size_t c, int device) {
    if (ct.polys_.size() != ct.meta.component_count || ct.meta.p_count || !ct.meta.is_ntt_form)
        throw std::invalid_argument("prepared AV requires Q-only NTT ciphertexts");
    const auto &poly = ct.polys_.at(c);
    if (poly.degree != ct.meta.degree || poly.q_count != ct.meta.q_count ||
        poly.p_count || poly.shards.size() != 1)
        throw std::invalid_argument("prepared AV requires one full shard per component");
    const auto &shard = poly.shards.front();
    const auto &field = ct.fields_.at(shard.field_index);
    if (field.device_id != device || shard.limb_begin || shard.coeff_begin ||
        shard.limb_count != ct.meta.q_count || shard.coeff_count != ct.meta.degree ||
        shard.field_offset + ct.meta.q_count * ct.meta.degree > field.size())
        throw std::invalid_argument("prepared AV ciphertext shard mismatch");
    return field.data() + shard.field_offset;
}
GpuWord *component(GpuCiphertextData &ct, std::size_t c, int device) {
    return const_cast<GpuWord *>(component(static_cast<const GpuCiphertextData &>(ct), c, device));
}
std::shared_ptr<const CrtContext::ContextData> level_data(const PoseidonContext &context, std::size_t level) {
    for (auto data = context.crt_context()->first_context_data(); data; data = data->next_context_data())
        if (data->parms().p().empty() && data->parms().q().size() == level + 1) return data;
    throw std::invalid_argument("AvSpec physical level is absent from context");
}
std::uint64_t key_bytes(const GpuEvaluationKeyData &keys) {
    std::uint64_t bytes = 0;
    for (const auto &field : keys.fields_) bytes += field.size() * sizeof(GpuWord);
    return bytes + (keys.galois_key_q0_ptrs.size() + keys.galois_key_q1_ptrs.size() +
        keys.galois_key_p0_ptrs.size() + keys.galois_key_p1_ptrs.size()) * sizeof(const GpuWord *);
}
// Local version of the mature inverse pre-rotation helper. Allocations precede
// queued kernels, and its inner guard closes the stock helper's exception gap.
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
            throw std::logic_error("AV inverse automorphism failed");
        const auto &src = source.polys.at(metadata.poly_id).shards.front();
        auto &dst = destination.polys.at(metadata.poly_id).shards.front();
        GpuPolyShardView scratch{device, temporary.data(), 0,
            keys.meta.q_count + keys.meta.p_count, 0, keys.meta.degree};
        kernel::launch_apply_galois_ntt_poly_shard(scratch, src,
            static_cast<std::uint32_t>(inverse), keys.meta.degree);
        gpu_check_cuda(cudaMemcpyAsync(dst.ptr, temporary.data(), words * sizeof(GpuWord),
            cudaMemcpyDeviceToDevice, stream), "AV inverse pre-rotate key copy");
        const auto flat = metadata.key_index * keys.meta.decomposition_count + metadata.decomposition_index;
        if (metadata.component_index == 0) {
            q0.at(flat) = src.ptr; p0.at(flat) = src.ptr + keys.meta.q_count * keys.meta.degree;
        } else if (metadata.component_index == 1) {
            q1.at(flat) = src.ptr; p1.at(flat) = src.ptr + keys.meta.q_count * keys.meta.degree;
        }
    }
    keys.galois_key_q0_ptrs.copy_from_host(q0.data(), pointers);
    keys.galois_key_p0_ptrs.copy_from_host(p0.data(), pointers);
    keys.galois_key_q1_ptrs.copy_from_host(q1.data(), pointers);
    keys.galois_key_p1_ptrs.copy_from_host(p1.data(), pointers);
    keys.meta.galois_format = GpuGaloisKeyFormat::InversePreRotated;
}
struct ScalePlan { double diagonal, score; };
ScalePlan validate_spec(const AvSpec &s, const PoseidonContext &context) {
    if (s.algorithm != AvAlgorithm::Baseline && s.algorithm != AvAlgorithm::PackedInput2Partial)
        throw std::invalid_argument("unknown prepared AV algorithm");
    if (s.id.empty() || s.tokens != 128 || s.tile != 256 || s.head_dim != 64 || s.baby != 8 ||
        s.score_input_level != 10 || s.value_input_level != 8 || s.common_level != 6 || s.output_level != 4 ||
        context.parameters_literal()->degree() != 65536 || context.parameters_literal()->slot() != 32768)
        throw std::invalid_argument("prepared AV requires id/N65536/T128/tile256/D64/baby8 and physical L10/L8/L6/L4");
    const auto &q = context.parameters_literal()->q();
    if (q.size() <= s.score_input_level || context.parameters_literal()->p().empty())
        throw std::invalid_argument("prepared AV requires sufficient Q and HYBRID P primes");
    for (const auto &prime : q)
        if (prime.value() > std::numeric_limits<GpuWord>::max())
            throw std::invalid_argument("prepared AV requires <=32-bit Q primes");
    for (const auto &prime : context.parameters_literal()->p())
        if (prime.value() > std::numeric_limits<GpuWord>::max())
            throw std::invalid_argument("prepared AV requires <=32-bit P primes");
    for (double scale : {s.score_input_scale_log2, s.value_input_scale_log2,
                         s.common_scale_log2, s.output_scale_log2})
        if (!std::isfinite(scale) || scale <= 0 || scale > 200)
            throw std::invalid_argument("invalid AV physical scale");
    auto drop_log = [&](std::size_t from, std::size_t to) {
        double result = 0;
        for (auto l = from; l > to; --l) result += std::log2(double(q.at(l).value()));
        return result;
    };
    ScalePlan result{s.common_scale_log2 + drop_log(s.value_input_level, s.common_level) - s.value_input_scale_log2,
                     s.common_scale_log2 + drop_log(s.score_input_level, s.common_level) - s.score_input_scale_log2};
    const double output = 2 * s.common_scale_log2 - drop_log(s.common_level, s.output_level);
    if (std::abs(output - s.output_scale_log2) > 1e-8 || result.diagonal <= 1 || result.score <= 1)
        throw std::invalid_argument("AV scale plan does not match actual Q-prime drops");
    // Metadata equations alone do not establish CKKS representability. Keep
    // 32 bits of modulus headroom at EVERY input/plain/product/output boundary.
    // This is a scale-capacity envelope, not a proof for unbounded encrypted
    // messages: the caller remains responsible for bounded message amplitudes.
    auto require_capacity = [&](std::size_t level, double scale, const char *stage) {
        double capacity = 0;
        for (std::size_t l = 0; l <= level; ++l) capacity += std::log2(double(q.at(l).value()));
        if (!std::isfinite(scale) || scale + 32.0 >= capacity)
            throw std::invalid_argument(std::string("AV scale exceeds actual Q capacity/headroom at ") + stage);
    };
    require_capacity(s.score_input_level, s.score_input_scale_log2, "score input");
    require_capacity(s.value_input_level, s.value_input_scale_log2, "value input");
    require_capacity(s.score_input_level, result.score, "score mask plaintext");
    require_capacity(s.value_input_level, result.diagonal, "diagonal mask plaintext");
    require_capacity(s.score_input_level, s.score_input_scale_log2 + result.score, "score CP");
    require_capacity(s.value_input_level, s.value_input_scale_log2 + result.diagonal, "diagonal CP");
    require_capacity(s.common_level, s.common_scale_log2, "common ciphertext");
    require_capacity(s.common_level, 2 * s.common_scale_log2, "CC before rescale");
    require_capacity(s.output_level, s.output_scale_log2, "output ciphertext");
    for (std::size_t l = s.output_level; l <= s.score_input_level; ++l) (void)level_data(context, l);
    return result;
}
} // namespace

struct PreparedAvEngine::Impl {
    struct MaskBundle {
        std::vector<GpuPlaintextData> diagonal, left, right;
        GpuPlaintextData unity;
        std::uint64_t bytes = 0, plaintext_count = 0;
    };
    struct Prepared { AvSpec spec; AvStats stats; std::shared_ptr<MaskBundle> masks; };
    struct Runtime {
        GpuParameterData parameters;
        GpuEvaluator evaluator;
        GpuRotateWorkspace rotate_workspace;
        GpuDoubleHoistWorkspace hoist_workspace;
        Runtime(const PoseidonContext &context, int device)
            : parameters(context, device, {5,6,7,8,9,10,11}), evaluator(parameters) {}
    };
    struct RunStorage {
        std::vector<GpuCiphertextData> values, diagonals, scores, little;
        GpuCiphertextData diagonal, diagonal_group, diagonal_term, normalized_diagonal;
        GpuCiphertextData left, right, cyclic, product, relinearized, term, group, total, add_scratch;
        GpuCiphertextData even_cyclic, rotated;
    };
    // Algorithm is part of mask SEMANTICS, not merely scheduling. Full masks
    // must never be reused as live-only masks at the same physical scales.
    // IDs never participate in this identity; each mode shares across IDs.
    using BundleKey = std::tuple<AvAlgorithm, double, double, double, double>;
    PoseidonContext context;
    std::shared_ptr<const RelinKeys> cpu_relin;
    std::shared_ptr<const GaloisKeys> cpu_galois;
    int device;
    cudaStream_t stream = nullptr;
    std::map<std::string, Prepared> prepared;
    std::map<BundleKey, std::shared_ptr<MaskBundle>> bundles;
    std::unique_ptr<Runtime> runtime;
    std::unique_ptr<GpuRelinKeysData> relin;
    std::unique_ptr<GpuGaloisKeysData> galois;
    std::map<int, std::vector<int>> rotation_paths;
    AvEngineStats stats;
    Impl(const PoseidonContext &ctx, std::shared_ptr<const RelinKeys> r,
         std::shared_ptr<const GaloisKeys> g, int dev)
        : context(ctx), cpu_relin(std::move(r)), cpu_galois(std::move(g)), device(dev) {}
    ~Impl() {
        if (!stream) return;
        (void)cudaSetDevice(device);
        {
            ScopedGpuExecutionStream guard(stream);
            (void)cudaStreamSynchronize(stream);
            prepared.clear(); bundles.clear(); runtime.reset(); galois.reset(); relin.reset();
        }
        (void)cudaStreamDestroy(stream);
    }
    std::uint32_t elt(int step) const { return context.crt_context()->galois_tool()->get_elt_from_step(step); }
    std::vector<int> path(int step) const {
        if (cpu_galois->has_key(elt(step))) return {step};
        std::vector<int> result;
        const int sign = step < 0 ? -1 : 1;
        unsigned value = static_cast<unsigned>(step < 0 ? -step : step);
        for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
            if (value & 1) result.push_back(sign * static_cast<int>(bit));
        if (std::all_of(result.begin(), result.end(), [&](int s) { return cpu_galois->has_key(elt(s)); })) return result;
        result.clear();
        const auto slots = static_cast<int>(context.parameters_literal()->slot());
        value = static_cast<unsigned>((step % slots + slots) % slots);
        for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
            if (value & 1) {
                if (!cpu_galois->has_key(elt(static_cast<int>(bit))))
                    throw std::invalid_argument("missing direct/binary AV rotation " + std::to_string(step));
                result.push_back(static_cast<int>(bit));
            }
        return result;
    }
    void update_bytes() {
        stats.allocated_bytes = stats.parameter_bytes + stats.galois_key_bytes + stats.relin_key_bytes + stats.mask_bytes;
    }
    void ensure_gpu(AvAlgorithm algorithm) {
        gpu_check_cuda(cudaSetDevice(device), "AV prepare select device");
        std::set<int> required;
        for (const auto &entry : rotation_paths) required.insert(entry.first);
        const auto added = algorithm == AvAlgorithm::PackedInput2Partial ?
            PreparedAvEngine::required_packed_input_rotation_steps() : PreparedAvEngine::required_rotation_steps();
        required.insert(added.begin(), added.end());
        if (runtime && required.size() == rotation_paths.size()) return;
        // prepare() is forbidden after first run: no live evaluator work or
        // cached hoist can observe stale owners/views during this extension.
        if (!cpu_relin || !cpu_galois || cpu_relin->parms_id() != context.crt_context()->key_parms_id() ||
            cpu_galois->parms_id() != context.crt_context()->key_parms_id())
            throw std::invalid_argument("prepared AV requires matching CPU relin/Galois keys");
        if (!stream) gpu_check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "AV stream create");
        ScopedGpuExecutionStream guard(stream);
        std::unique_ptr<GpuRelinKeysData> new_relin;
        std::unique_ptr<GpuGaloisKeysData> new_galois;
        std::unique_ptr<Runtime> new_runtime;
        FailureSync failure{stream};
        std::set<std::uint32_t> elements;
        std::map<int, std::vector<int>> paths;
        for (int step : required) {
            auto parts = path(step);
            for (int part : parts) elements.insert(elt(part));
            paths.emplace(step, std::move(parts));
        }
        // Owning Q11 crop retains all P11. Lower views only reference these
        // genuinely cropped owners; no dynamic P or fake full-Q storage view.
        new_relin = std::make_unique<GpuRelinKeysData>();
        new_galois = std::make_unique<GpuGaloisKeysData>();
        *new_relin = GpuUploader::upload_relin_keys(*cpu_relin, device, 11);
        ++stats.relin_key_upload_calls;
        *new_galois = GpuUploader::upload_galois_keys(*cpu_galois, device, 11,
            std::vector<std::uint32_t>(elements.begin(), elements.end()));
        ++stats.galois_key_upload_calls;
        pre_rotate_keys_safely(*new_galois, device, stream);
        const auto np = context.parameters_literal()->p().size();
        if (new_relin->meta.q_count != 11 || new_galois->meta.q_count != 11 ||
            new_relin->meta.p_count != np || new_galois->meta.p_count != np)
            throw std::runtime_error("AV owning keys did not retain Q11/full P");
        GpuUploader::prepare_key_views_for_q_counts(*new_relin, {7});
        GpuUploader::prepare_key_views_for_q_counts(*new_galois, {5,7,9,11});
        if (!runtime) new_runtime = std::make_unique<Runtime>(context, device);
        gpu_check_cuda(cudaStreamSynchronize(stream), "AV key/parameter setup synchronize");
        relin.swap(new_relin); galois.swap(new_galois);
        if (new_runtime) runtime.swap(new_runtime);
        rotation_paths.swap(paths);
        stats.key_storage_q_count = 11; stats.key_storage_p_count = np;
        stats.key_storage_is_full_q = 11 == context.parameters_literal()->q().size();
        stats.relin_key_bytes = key_bytes(*relin); stats.galois_key_bytes = key_bytes(*galois);
        stats.parameter_bytes = runtime->parameters.allocated_bytes();
        update_bytes();
    }
    void require_meta(const GpuCiphertextData &ct, std::size_t physical_level, double scale_log2,
                      const char *stage, std::size_t components = 2) const {
        const auto &level = runtime->parameters.get_level_by_q_count(physical_level + 1);
        if (ct.meta.degree != level.degree || ct.meta.q_count != level.q_count || ct.meta.parms_id != level.parms_id ||
            ct.meta.component_count != components || !(ct.meta.scale > 0) || !std::isfinite(ct.meta.scale) ||
            std::abs(std::log2(ct.meta.scale) - scale_log2) > 1e-7)
            throw std::runtime_error(std::string("prepared AV physical level/scale mismatch at ") + stage);
        for (std::size_t c = 0; c < components; ++c) (void)component(ct, c, device);
    }
    void same_meta(const GpuCiphertextData &ct, const GpuCiphertextData &source, const char *stage) const {
        require_meta(ct, source.meta.q_count - 1, std::log2(source.meta.scale), stage, source.meta.component_count);
    }
    void clone_into(const GpuCiphertextData &source, GpuCiphertextData &result) {
        if (&source == &result || source.meta.component_count != 2) throw std::logic_error("AV clone alias/size");
        if (result.meta.degree != source.meta.degree || result.meta.q_count != source.meta.q_count || result.meta.component_count != 2)
            result = GpuCiphertextData::allocate_single_device(source.meta.degree, source.meta.q_count, 2, device);
        result.meta = source.meta;
        for (std::size_t c = 0; c < 2; ++c)
            gpu_check_cuda(cudaMemcpyAsync(component(result, c, device), component(source, c, device),
                source.meta.degree * source.meta.q_count * sizeof(GpuWord), cudaMemcpyDeviceToDevice, stream), "AV clone");
    }
    void rotate_inplace(GpuCiphertextData &ct, int step) {
        const auto level = ct.meta.q_count - 1;
        const auto scale = std::log2(ct.meta.scale);
        for (int part : rotation_paths.at(step)) {
            runtime->evaluator.rotate_inplace(ct, part, *galois, runtime->rotate_workspace);
            ++stats.rotation_calls;
            require_meta(ct, level, scale, "rotation");
        }
    }
    void rotate_into(const GpuCiphertextData &source, int step, GpuCiphertextData &result) {
        clone_into(source, result); rotate_inplace(result, step);
    }
    void shared_rotations(const GpuCiphertextData &source, const std::vector<int> &steps,
                          std::vector<GpuCiphertextData> &out) {
        const bool direct = std::all_of(steps.begin(), steps.end(), [&](int s) {
            return rotation_paths.at(s).size() == 1 && rotation_paths.at(s).front() == s;
        });
        if (direct) {
            runtime->evaluator.rotate_many_hoisted(source, steps, *galois, runtime->hoist_workspace, out);
            stats.rotation_calls += steps.size(); ++stats.hoisted_rotation_batches;
            for (const auto &ct : out) same_meta(ct, source, "hoisted rotation");
        } else {
            out.resize(steps.size());
            for (std::size_t i = 0; i < steps.size(); ++i) {
                clone_into(source, out[i]); rotate_inplace(out[i], steps[i]);
            }
        }
    }
    void add_into(GpuCiphertextData &accumulator, const GpuCiphertextData &term, GpuCiphertextData &scratch) {
        // Ordinary add(a,b,a) is NOT safe in the frozen evaluator.
        runtime->evaluator.add(accumulator, term, scratch); ++stats.add_calls;
        same_meta(scratch, accumulator, "add"); std::swap(accumulator, scratch);
    }
    void multiply_plain(const GpuCiphertextData &source, const GpuPlaintextData &plain, GpuCiphertextData &result) {
        runtime->evaluator.multiply_plain(source, plain, result); ++stats.cp_multiply_calls;
        require_meta(result, source.meta.q_count - 1, std::log2(source.meta.scale) + std::log2(plain.meta.scale), "CP");
    }
    void rescale(const GpuCiphertextData &source, GpuCiphertextData &result, unsigned drops,
                 std::size_t level, double scale, const char *stage) {
        runtime->evaluator.rescale_many(source, result, drops);
        ++stats.rescale_calls; stats.physical_rescale_primes += drops;
        if (drops == 2) ++stats.rescale2_calls;
        if (drops == 4) ++stats.rescale4_calls;
        require_meta(result, level, scale, stage);
    }
};

PreparedAvEngine::PreparedAvEngine(const PoseidonContext &context, std::shared_ptr<const RelinKeys> relin,
        std::shared_ptr<const GaloisKeys> galois, int device)
    : impl_(std::make_unique<Impl>(context, std::move(relin), std::move(galois), device)) {}
PreparedAvEngine::~PreparedAvEngine() = default;
const char *av_algorithm_name(AvAlgorithm algorithm) noexcept {
    switch (algorithm) {
    case AvAlgorithm::Baseline: return "Baseline";
    case AvAlgorithm::PackedInput2Partial: return "PackedInput2Partial";
    }
    return "Unknown";
}
const char *av_output_semantics(AvAlgorithm algorithm) noexcept {
    switch (algorithm) {
    case AvAlgorithm::Baseline: return "full_cyclic_av_all_features";
    case AvAlgorithm::PackedInput2Partial: return "live_full_av_gap_B_current_plus_A_next_head_row";
    }
    return "unknown";
}
std::vector<int> PreparedAvEngine::required_rotation_steps() {
    std::set<int> steps;
    for (int b = 1; b < 8; ++b) { steps.insert(256 * b); steps.insert(b); }
    for (int b = 0; b < 8; ++b) steps.insert(b - 128);
    for (int g = 1; g < 16; ++g) { steps.insert(2048 * g); steps.insert(-2040 * g); }
    return {steps.begin(), steps.end()};
}
std::vector<int> PreparedAvEngine::required_binary_rotation_steps() {
    std::set<int> steps;
    for (int step : required_rotation_steps()) {
        const int sign = step < 0 ? -1 : 1;
        unsigned value = static_cast<unsigned>(step < 0 ? -step : step);
        for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
            if (value & 1) steps.insert(sign * static_cast<int>(bit));
    }
    return {steps.begin(), steps.end()};
}
std::vector<int> PreparedAvEngine::required_packed_input_rotation_steps() {
    const auto baseline = required_rotation_steps();
    std::set<int> steps(baseline.begin(), baseline.end());
    steps.insert(-64); steps.insert(64); steps.insert(192);
    return {steps.begin(), steps.end()};
}
std::vector<int> PreparedAvEngine::required_packed_input_binary_rotation_steps() {
    std::set<int> steps;
    for (int step : required_packed_input_rotation_steps()) {
        const int sign = step < 0 ? -1 : 1;
        unsigned value = static_cast<unsigned>(step < 0 ? -step : step);
        for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
            if (value & 1) steps.insert(sign * static_cast<int>(bit));
    }
    return {steps.begin(), steps.end()};
}
void PreparedAvEngine::prepare(const AvSpec &s) {
    const auto begin = Clock::now();
    auto &p = *impl_;
    if (p.stats.preparation_locked) throw std::logic_error("prepare all AV IDs/algorithms before first run");
    const auto plan = validate_spec(s, p.context);
    if (p.prepared.count(s.id)) throw std::invalid_argument("duplicate prepared AV id " + s.id);
    p.ensure_gpu(s.algorithm);
    ScopedGpuExecutionStream guard(p.stream);
    Impl::Prepared item;
    Plaintext encoded;
    std::vector<double> slots(s.tokens * s.tile);
    std::shared_ptr<Impl::MaskBundle> bundle;
    FailureSync failure{p.stream};
    item.spec = s; item.stats.id = s.id;
    item.stats.algorithm = s.algorithm;
    item.stats.output_semantics = av_output_semantics(s.algorithm);
    const bool packed = s.algorithm == AvAlgorithm::PackedInput2Partial;
    item.stats.schedule = packed ? "packed_input2_live_masks_even_babies" : "cerium_bsgs_baby8_giants16";
    item.stats.partial_output = packed; item.stats.client_value_gap_zero_required = packed;
    item.stats.diagonal_mask_scale_log2 = plan.diagonal; item.stats.score_mask_scale_log2 = plan.score;
    const auto key = Impl::BundleKey{s.algorithm, s.score_input_scale_log2, s.value_input_scale_log2,
                                     s.common_scale_log2, s.output_scale_log2};
    const auto existing = p.bundles.find(key);
    if (existing != p.bundles.end()) {
        item.masks = existing->second; item.stats.reused_mask_bundle = true;
    } else {
        bundle = std::make_shared<Impl::MaskBundle>();
        bundle->diagonal.resize(s.tokens); bundle->left.resize(s.tokens - 1); bundle->right.resize(s.tokens - 1);
        CKKSEncoder encoder(p.context);
        auto encode = [&](std::size_t level, double scale, GpuPlaintextData &destination) {
            encoder.encode(slots, level_data(p.context, level)->parms_id(), std::exp2(scale), encoded);
            if (!std::isfinite(encoded.scale()) || std::abs(std::log2(encoded.scale()) - scale) > 1e-10)
                throw std::runtime_error("AV real mask encoder scale mismatch");
            destination = GpuUploader::upload_plaintext(encoded, p.device);
            if (destination.meta.q_count != level + 1 || destination.meta.parms_id != encoded.parms_id() ||
                !destination.meta.is_ntt_form || destination.meta.p_count ||
                std::abs(std::log2(destination.meta.scale) - scale) > 1e-10)
                throw std::runtime_error("AV uploaded mask physical metadata mismatch");
            for (const auto &field : destination.fields_) bundle->bytes += field.size() * sizeof(GpuWord);
            ++bundle->plaintext_count;
        };
        for (std::size_t shift = 0; shift < s.tokens; ++shift) {
            for (std::size_t i = 0; i < slots.size(); ++i) slots[i] = i % s.tokens == shift ? 1.0 : 0.0;
            encode(s.value_input_level, plan.diagonal, bundle->diagonal[shift]);
        }
        for (std::size_t shift = 1; shift < s.tokens; ++shift) {
            for (std::size_t i = 0; i < slots.size(); ++i) {
                const auto j = i % s.tokens;
                slots[i] = (!packed || j < s.head_dim) && j < s.tokens - shift ? 1.0 : 0.0;
            }
            encode(s.score_input_level, plan.score, bundle->left[shift - 1]);
            // For s1..64 the live-intersected right mask is IDENTICALLY zero.
            // Do not encode/upload it. All128 diagonal selectors remain real.
            if (!packed || shift > s.tokens - s.head_dim) {
                for (std::size_t i = 0; i < slots.size(); ++i) {
                    const auto j = i % s.tokens;
                    slots[i] = (!packed || j < s.head_dim) && j >= s.tokens - shift ? 1.0 : 0.0;
                }
                encode(s.score_input_level, plan.score, bundle->right[shift - 1]);
            }
        }
        for (std::size_t i = 0; i < slots.size(); ++i)
            slots[i] = !packed || i % s.tokens < s.head_dim ? 1.0 : 0.0;
        // Baseline unity or packed live mask, explicitly repeated over ALL slots.
        encode(s.score_input_level, plan.score, bundle->unity);
        gpu_check_cuda(cudaStreamSynchronize(p.stream), "AV mask setup synchronize");
        item.masks = bundle;
        p.bundles.emplace(key, bundle);
        p.stats.mask_bytes += bundle->bytes; ++p.stats.mask_bundle_count;
        p.stats.mask_plaintext_count += bundle->plaintext_count;
        p.update_bytes();
    }
    item.stats.mask_bytes = item.masks->bytes;
    item.stats.mask_plaintext_count = item.masks->plaintext_count;
    item.stats.setup_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    p.stats.setup_seconds += item.stats.setup_seconds;
    p.prepared.emplace(s.id, std::move(item));
}
GpuCiphertextData PreparedAvEngine::run(const std::string &id, const GpuCiphertextData &scores,
                                       const GpuCiphertextData &value) {
    auto &p = *impl_;
    const auto &m = p.prepared.at(id); const auto &s = m.spec; const auto &masks = *m.masks;
    const bool packed = s.algorithm == AvAlgorithm::PackedInput2Partial;
    p.stats.preparation_locked = true;
    gpu_check_cuda(cudaSetDevice(p.device), "AV run select device");
    ScopedGpuExecutionStream guard(p.stream);
    Impl::RunStorage w;
    std::vector<int> value_steps, score_steps, little_steps;
    std::vector<std::size_t> right_index(s.baby);
    FailureSync failure{p.stream};
    auto &e = p.runtime->evaluator;
    p.require_meta(scores, s.score_input_level, s.score_input_scale_log2, "scores input");
    p.require_meta(value, s.value_input_level, s.value_input_scale_log2, "value input");
    for (std::size_t b = 1; b < s.baby; ++b) value_steps.push_back(static_cast<int>(s.tile * b));
    for (std::size_t g = 1; g < s.tokens / s.baby; ++g)
        score_steps.push_back(static_cast<int>(s.baby * g) - static_cast<int>(s.tile * s.baby * g));
    p.shared_rotations(value, value_steps, w.values);
    bool have_diagonal = false;
    for (std::size_t g = 0; g < s.tokens / s.baby; ++g) {
        bool have_group = false;
        for (std::size_t b = 0; b < s.baby; ++b) {
            const auto &v = b ? w.values[b - 1] : value;
            p.multiply_plain(v, masks.diagonal[g * s.baby + b], w.diagonal_term);
            if (!have_group) { std::swap(w.diagonal_group, w.diagonal_term); have_group = true; }
            else p.add_into(w.diagonal_group, w.diagonal_term, w.add_scratch);
        }
        if (g) p.rotate_inplace(w.diagonal_group, static_cast<int>(s.tile * s.baby * g));
        if (!have_diagonal) { std::swap(w.diagonal, w.diagonal_group); have_diagonal = true; }
        else p.add_into(w.diagonal, w.diagonal_group, w.add_scratch);
    }
    // All 128 selectors share this ONE physical Rescale2; only then make babies.
    p.rescale(w.diagonal, w.normalized_diagonal, 2, s.common_level, s.common_scale_log2, "V diagonal Rescale2");
    if (packed) {
        // +192, NOT -64: each gap receives the NEXT TOKEN's same-head
        // diagonal. This supplies the odd offset paired with score's -64 pack.
        p.rotate_into(w.normalized_diagonal, static_cast<int>(s.tile - s.head_dim), w.rotated);
        p.add_into(w.normalized_diagonal, w.rotated, w.add_scratch);
        value_steps.clear();
        for (std::size_t b = 2; b < s.baby; b += 2)
            value_steps.push_back(static_cast<int>(s.tile * b));
    }
    p.shared_rotations(w.normalized_diagonal, value_steps, w.diagonals);
    p.shared_rotations(scores, score_steps, w.scores);
    bool have_total = false;
    for (std::size_t g = 0; g < s.tokens / s.baby; ++g) {
        const auto &base = g ? w.scores[g - 1] : scores;
        little_steps.clear();
        for (std::size_t b = 1; b < s.baby; ++b) little_steps.push_back(static_cast<int>(b));
        // Each batch has ONE immutable source. Packed right branches exist only
        // for shifts65..127; do not rotate branches whose public live mask is0.
        std::fill(right_index.begin(), right_index.end(), std::numeric_limits<std::size_t>::max());
        for (std::size_t b = 0; b < s.baby; ++b) {
            const auto shift = g * s.baby + b;
            if (shift && (!packed || shift > s.tokens - s.head_dim)) {
                right_index[b] = little_steps.size();
                little_steps.push_back(static_cast<int>(b) - static_cast<int>(s.tokens));
            }
        }
        p.shared_rotations(base, little_steps, w.little);
        bool have_group = false;
        for (std::size_t b = 0; b < s.baby; ++b) {
            const auto shift = g * s.baby + b;
            if (!shift) p.multiply_plain(base, masks.unity, w.left);
            else {
                const auto &left = b ? w.little[b - 1] : base;
                p.multiply_plain(left, masks.left[shift - 1], w.left);
                if (right_index[b] != std::numeric_limits<std::size_t>::max()) {
                    const auto &right = w.little[right_index[b]];
                    p.multiply_plain(right, masks.right[shift - 1], w.right);
                    p.add_into(w.left, w.right, w.add_scratch);
                }
            }
            // Wide PTscale combines mask AND old unity alignment arithmetically.
            // No scale/parms metadata is changed: 4 actual primes are removed.
            p.rescale(w.left, w.cyclic, 4, s.common_level, s.common_scale_log2, "score mask Rescale4");
            if (packed && b % 2 == 0) {
                std::swap(w.even_cyclic, w.cyclic);
                continue;
            }
            if (packed) {
                // Both inputs are normalized to the original canonical S6.
                // -64 moves odd's live values into gaps without live overlap.
                p.rotate_into(w.cyclic, -static_cast<int>(s.head_dim), w.rotated);
                p.add_into(w.even_cyclic, w.rotated, w.add_scratch);
            }
            const auto &score_operand = packed ? w.even_cyclic : w.cyclic;
            const auto diagonal_baby = packed ? (b - 1) / 2 : b;
            const auto &diagonal = diagonal_baby ? w.diagonals[diagonal_baby - 1] : w.normalized_diagonal;
            e.multiply(score_operand, diagonal, w.product); ++p.stats.cc_multiply_calls;
            p.require_meta(w.product, s.common_level, 2 * s.common_scale_log2, "CC", 3);
            e.relinearize(w.product, *p.relin, w.relinearized); ++p.stats.relinearize_calls;
            p.require_meta(w.relinearized, s.common_level, 2 * s.common_scale_log2, "relinearize");
            p.rescale(w.relinearized, w.term, 2, s.output_level, s.output_scale_log2, "CC Rescale2");
            if (!have_group) { std::swap(w.group, w.term); have_group = true; }
            else p.add_into(w.group, w.term, w.add_scratch);
        }
        if (g) p.rotate_inplace(w.group, static_cast<int>(s.tile * s.baby * g));
        if (!have_total) { std::swap(w.total, w.group); have_total = true; }
        else p.add_into(w.total, w.group, w.add_scratch);
    }
    if (packed) {
        // P=[A0,B0,A1,B1]. Live=A+B; gap=B_current+A_next_head/row.
        // Gaps are defined nonzero partials, NOT a zero-gap AV ciphertext.
        p.rotate_into(w.total, static_cast<int>(s.head_dim), w.rotated);
        p.add_into(w.total, w.rotated, w.add_scratch);
    }
    p.require_meta(w.total, s.output_level, s.output_scale_log2, "AV output");
    gpu_check_cuda(cudaStreamSynchronize(p.stream), "AV pair output synchronize");
    ++p.stats.run_calls;
    if (packed) ++p.stats.packed_input2_partial_run_calls;
    else ++p.stats.baseline_run_calls;
    return std::move(w.total);
}
const AvStats &PreparedAvEngine::av_stats(const std::string &id) const { return impl_->prepared.at(id).stats; }
AvEngineStats PreparedAvEngine::engine_stats() const { return impl_->stats; }
} // namespace bert::attention
