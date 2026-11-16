#include "compact_linear.h"
#include "compact_layout.h"
#include "compact_mac.h"
#include "periodic_encoder_experiment.h"

#include "poseidon/ckks_encoder.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_uploader.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace bert::compact {
namespace {
using namespace poseidon;
using namespace poseidon::gpu;
using Clock = std::chrono::steady_clock;

const GpuWord *component(const GpuCiphertextData &ct, std::size_t c, int device) {
    if (ct.polys_.size() != 2 || ct.meta.component_count != 2 ||
        ct.meta.p_count || !ct.meta.is_ntt_form)
        throw std::invalid_argument("compact Linear requires size2 Q-only NTT ciphertexts");
    const auto &poly = ct.polys_.at(c);
    if (poly.degree != ct.meta.degree || poly.q_count != ct.meta.q_count || poly.p_count ||
        poly.shards.size() != 1)
        throw std::invalid_argument("compact Linear requires one full shard per component");
    const auto &shard = poly.shards.front();
    const auto &field = ct.fields_.at(shard.field_index);
    if (field.device_id != device || shard.limb_begin || shard.coeff_begin ||
        shard.limb_count != ct.meta.q_count || shard.coeff_count != ct.meta.degree ||
        shard.field_offset + ct.meta.q_count * ct.meta.degree > field.size())
        throw std::invalid_argument("compact Linear ciphertext shard mismatch");
    return field.data() + shard.field_offset;
}
GpuWord *component(GpuCiphertextData &ct, std::size_t c, int device) {
    return const_cast<GpuWord *>(component(static_cast<const GpuCiphertextData &>(ct), c, device));
}

std::shared_ptr<const CrtContext::ContextData> level_data(
        const PoseidonContext &ctx, std::size_t level) {
    for (auto data = ctx.crt_context()->first_context_data(); data;
         data = data->next_context_data())
        if (data->parms().p().empty() && data->parms().q().size() == level + 1)
            return data;
    throw std::invalid_argument("MatrixSpec physical level is absent from context");
}

void validate_spec(const MatrixSpec &s, const PoseidonContext &ctx) {
    if (s.id.empty() || s.tile != 256 || s.baby != 16 || s.tokens < 2 ||
        !s.input_width || !s.output_width || s.input_width % s.tile || s.output_width % s.tile)
        throw std::invalid_argument("compact Linear needs id, dense widths divisible by256, tile256/baby16");
    if (s.tokens * s.tile != ctx.parameters_literal()->slot())
        throw std::invalid_argument("MatrixSpec tokens*tile must exactly fill CKKS slots");
    if (!s.rescale_count || s.input_level < s.rescale_count ||
        s.output_level != s.input_level - s.rescale_count)
        throw std::invalid_argument("MatrixSpec output_level must be input_level-rescale_count");
    if (!std::isfinite(s.input_scale_log2) || !std::isfinite(s.output_scale_log2))
        throw std::invalid_argument("MatrixSpec scales must be finite");
    level_data(ctx, s.input_level);
    level_data(ctx, s.output_level);
    for (const auto &q : ctx.parameters_literal()->q())
        if (q.value() > std::numeric_limits<GpuWord>::max())
            throw std::invalid_argument("compact GPU Linear requires <=32-bit Q primes");
    for (const auto &p : ctx.parameters_literal()->p())
        if (p.value() > std::numeric_limits<GpuWord>::max())
            throw std::invalid_argument("compact GPU Linear requires <=32-bit P primes");
}
} // namespace

struct CompactLinearEngine::Impl {
    struct Term {
        std::size_t input_tile;
        std::size_t baby;
        GpuCompressedPlaintextQP plain;
    };
    struct Group { int giant; std::vector<Term> terms; };
    struct Output { std::vector<Group> groups; GpuPlaintextData bias; };
    struct Matrix {
        MatrixSpec spec;
        MatrixStats stats;
        std::vector<float> weights, bias;
        std::vector<Output> outputs;
    };
    struct Runtime {
        GpuParameterData parameters;
        GpuEvaluator evaluator;
        GpuRotateWorkspace rotate_workspace;
        Runtime(const PoseidonContext &ctx, int dev, std::size_t qin, std::size_t qout)
            : parameters(ctx, dev, counts(qin, qout)), evaluator(parameters) {}
        static std::vector<std::size_t> counts(std::size_t qin, std::size_t qout) {
            std::vector<std::size_t> result(qin - qout + 1);
            std::iota(result.begin(), result.end(), qout);
            return result;
        }
    };
    PoseidonContext context;
    std::shared_ptr<const GaloisKeys> cpu_keys;
    int device;
    cudaStream_t stream = nullptr;
    std::map<std::string, Matrix> matrices;
    std::map<std::pair<std::size_t, std::size_t>, std::unique_ptr<Runtime>> runtimes;
    std::unique_ptr<GpuGaloisKeysData> keys;
    std::size_t key_storage_q = 0;
    std::map<int, std::vector<int>> rotation_paths;
    GpuDoubleHoistWorkspace rotation_workspace;
    std::vector<std::vector<GpuCiphertextData>> babies; // baby0 uses caller input
    std::vector<const GpuCiphertextData *> sequence_inputs;
    std::vector<const GpuWord *> sequence_allocations;
    std::string sequence_matrix;
    std::size_t next_output = 0;
    DeviceVector<detail::MacTerm> descriptors;
    EngineStats stats;

    Impl(const PoseidonContext &ctx, std::shared_ptr<const GaloisKeys> k, int d)
        : context(ctx), cpu_keys(std::move(k)), device(d) {}
    ~Impl() {
        if (!stream) return;
        cudaSetDevice(device);
        ScopedGpuExecutionStream guard(stream);
        cudaStreamSynchronize(stream);
        babies.clear(); matrices.clear(); descriptors.release();
        rotation_workspace = GpuDoubleHoistWorkspace{};
        runtimes.clear(); keys.reset();
        cudaStreamDestroy(stream);
    }
    std::uint32_t elt(int step) const {
        return context.crt_context()->galois_tool()->get_elt_from_step(step);
    }
    std::vector<int> path(int step) const {
        if (cpu_keys->has_key(elt(step))) return {step};
        // Prefer short signed binary steps when those keys are present.
        std::vector<int> result;
        const int sign = step < 0 ? -1 : 1;
        unsigned value = static_cast<unsigned>(step < 0 ? -step : step);
        for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
            if (value & 1) result.push_back(sign * static_cast<int>(bit));
        if (std::all_of(result.begin(), result.end(), [&](int s) { return cpu_keys->has_key(elt(s)); }))
            return result;
        // Compatible with positive-power-only key policies: normalize the
        // complete cyclic rotation, rather than treating a negative as zero.
        result.clear();
        const auto slots = context.parameters_literal()->slot();
        value = static_cast<unsigned>((step + static_cast<int>(slots)) % static_cast<int>(slots));
        for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
            if (value & 1) {
                if (!cpu_keys->has_key(elt(static_cast<int>(bit))))
                    throw std::invalid_argument("missing direct/binary Galois key for compact Linear rotation " + std::to_string(step));
                result.push_back(static_cast<int>(bit));
            }
        return result;
    }
    void ensure_gpu(std::size_t required_q) {
        gpu_check_cuda(cudaSetDevice(device), "compact cudaSetDevice");
        if (keys && required_q <= key_storage_q) return;
        if (!cpu_keys) throw std::invalid_argument("compact Linear requires CPU Galois keys");
        if (!stream)
            gpu_check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "compact stream create");
        ScopedGpuExecutionStream guard(stream);
        std::set<std::uint32_t> elements;
        for (int step : CompactLinearEngine::required_rotation_steps()) {
            auto p = path(step);
            for (int part : p) elements.insert(elt(part));
            rotation_paths[step] = std::move(p);
        }
        // The existing uploader really crops owning Q and HYBRID digits;
        // it keeps all P. Prepare matrices highest-input-level first to
        // avoid later growing/re-uploading this immutable key owner.
        keys = std::make_unique<GpuGaloisKeysData>(GpuUploader::upload_double_hoist_galois_keys(
            *cpu_keys, device, required_q, std::vector<std::uint32_t>(elements.begin(), elements.end())));
        key_storage_q = required_q;
        stats.key_storage_is_full_q = key_storage_q == context.parameters_literal()->q().size();
        stats.key_bytes = 0;
        for (const auto &f : keys->fields_) stats.key_bytes += f.size() * sizeof(GpuWord);
        // Include owning pointer tables as well as key residues.
        stats.key_bytes += (keys->galois_key_q0_ptrs.size() + keys->galois_key_q1_ptrs.size() +
                            keys->galois_key_p0_ptrs.size() + keys->galois_key_p1_ptrs.size()) * sizeof(const GpuWord *);
        for (const auto &runtime : runtimes)
            (void)keys->const_view(runtime.first.first);
    }
    Runtime &runtime(const MatrixSpec &s) {
        const auto key = std::make_pair(s.input_level + 1, s.output_level + 1);
        auto it = runtimes.find(key);
        if (it == runtimes.end()) {
            auto runtime = std::make_unique<Runtime>(context, device, key.first, key.second);
            stats.parameter_bytes += runtime->parameters.allocated_bytes();
            GpuUploader::prepare_key_views_for_q_counts(*keys, {key.first});
            it = runtimes.emplace(key, std::move(runtime)).first;
        }
        return *it->second;
    }
    void rotate_inplace(Runtime &r, GpuCiphertextData &ciphertext, int step) {
        for (int part : rotation_paths.at(step)) {
            r.evaluator.rotate_inplace(ciphertext, part, *keys, r.rotate_workspace);
            ++stats.actual_rotation_calls;
        }
    }
    GpuCiphertextData rotate(Runtime &r, const GpuCiphertextData &source, int step) {
        auto result = GpuCiphertextData::allocate_single_device(source.meta.degree, source.meta.q_count, 2, device);
        result.meta = source.meta;
        for (std::size_t c = 0; c < 2; ++c)
            gpu_check_cuda(cudaMemcpyAsync(component(result, c, device), component(source, c, device),
                source.meta.degree * source.meta.q_count * sizeof(GpuWord), cudaMemcpyDeviceToDevice, stream), "compact fallback clone");
        rotate_inplace(r, result, step);
        return result;
    }
    void clear_sequence() {
        babies.clear(); sequence_inputs.clear(); sequence_allocations.clear();
        sequence_matrix.clear(); next_output = 0;
    }
    void build_babies(Matrix &m, Runtime &r,
            const std::vector<const GpuCiphertextData *> &inputs) {
        clear_sequence();
        std::vector<int> direct;
        bool all_direct = true;
        for (int b = 1; b < static_cast<int>(m.spec.baby); ++b) {
            direct.push_back(b);
            all_direct = all_direct && rotation_paths.at(b).size() == 1;
        }
        babies.resize(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (all_direct) {
                r.evaluator.rotate_many_hoisted(*inputs[i], direct, *keys, rotation_workspace, babies[i]);
                stats.actual_rotation_calls += direct.size();
            } else {
                babies[i].reserve(m.spec.baby - 1);
                for (int b : direct) babies[i].push_back(rotate(r, *inputs[i], b));
            }
            sequence_allocations.push_back(component(*inputs[i], 0, device));
        }
        sequence_inputs = inputs;
        sequence_matrix = m.spec.id;
        next_output = 0;
    }
};

CompactLinearEngine::CompactLinearEngine(const PoseidonContext &context,
        std::shared_ptr<const GaloisKeys> keys, int device)
    : impl_(std::make_unique<Impl>(context, std::move(keys), device)) {}
CompactLinearEngine::~CompactLinearEngine() = default;

std::vector<int> CompactLinearEngine::required_rotation_steps(std::size_t tile, std::size_t baby) {
    if (!power_of_two(tile) || !power_of_two(baby) || baby > tile || tile % baby)
        throw std::invalid_argument("invalid tile/baby BSGS strides");
    std::set<int> result;
    for (std::size_t b = 1; b < baby; ++b) result.insert(static_cast<int>(b));
    for (int d = -static_cast<int>(tile) + 1; d < static_cast<int>(tile); ++d) {
        const int g = split_diagonal(d, baby).giant;
        if (g) result.insert(g);
    }
    return {result.begin(), result.end()};
}
std::vector<int> CompactLinearEngine::required_binary_rotation_steps(std::size_t tile, std::size_t baby) {
    std::set<int> result;
    for (int s : required_rotation_steps(tile, baby)) {
        const int sign = s < 0 ? -1 : 1;
        unsigned v = static_cast<unsigned>(s < 0 ? -s : s);
        for (unsigned b = 1; v; b <<= 1, v >>= 1)
            if (v & 1) result.insert(sign * static_cast<int>(b));
    }
    return {result.begin(), result.end()};
}

void CompactLinearEngine::prepare(const MatrixSpec &s, const std::vector<float> &weights,
        const std::vector<float> &bias) {
    if (weights.size() != s.input_width * s.output_width || bias.size() != s.output_width)
        throw std::invalid_argument("compact Linear weight/bias vector size mismatch");
    prepare(s, weights.data(), bias.data());
}
void CompactLinearEngine::prepare(const MatrixSpec &s, const float *weights, const float *bias) {
    const auto start = Clock::now();
    validate_spec(s, impl_->context);
    if (!weights || !bias) throw std::invalid_argument("null dense weights or bias");
    if (impl_->matrices.count(s.id)) throw std::invalid_argument("duplicate compact matrix id " + s.id);
    std::vector<float> original_weights(weights, weights + s.input_width * s.output_width);
    std::vector<float> original_bias(bias, bias + s.output_width);
    for (float v : original_weights) if (!std::isfinite(v)) throw std::invalid_argument("nonfinite dense weight");
    for (float v : original_bias) if (!std::isfinite(v)) throw std::invalid_argument("nonfinite dense bias");
    if (!impl_->sequence_matrix.empty())
        throw std::logic_error("cannot prepare a matrix during an active output sequence");
    impl_->ensure_gpu(s.input_level + 1);
    ScopedGpuExecutionStream guard(impl_->stream);
    // GPU owners MUST be declared after the stream guard: partial prepare
    // cleanup also releases under the private stream on every exception.
    Impl::Matrix matrix;
    matrix.spec = s;
    matrix.weights = std::move(original_weights);
    matrix.bias = std::move(original_bias);
    matrix.stats.id = s.id;
    matrix.stats.weight_bytes = (matrix.weights.size() + matrix.bias.size()) * sizeof(float);
    impl_->runtime(s);
    CKKSEncoder encoder(impl_->context);
    const auto input = level_data(impl_->context, s.input_level);
    const auto output = level_data(impl_->context, s.output_level);
    const auto degree = impl_->context.parameters_literal()->degree();
    const auto qin = s.input_level + 1;
    const auto qout = s.output_level + 1;
    double plain_log2 = s.output_scale_log2 - s.input_scale_log2;
    for (std::size_t q = qout; q < qin; ++q)
        plain_log2 += std::log2(static_cast<double>(input->parms().q()[q].value()));
    const double plain_scale = std::exp2(plain_log2);
    if (!(plain_scale > 1.0) || !std::isfinite(plain_scale))
        throw std::invalid_argument("compact dense plaintext scale must be finite and >1");
    matrix.stats.plaintext_scale_log2 = plain_log2;
    std::unique_ptr<PeriodicEncoderExperiment> periodic;
    if (s.periodic_encoding) {
        try {
            periodic = std::make_unique<PeriodicEncoderExperiment>(impl_->context, input->parms_id(), s.tile);
            periodic->check_monomial_mapping();
            matrix.stats.periodic_full_ntt_mapping_verified = true;
        } catch (const std::bad_alloc &) { throw; }
        catch (const std::exception &e) {
            matrix.stats.periodic_fallback_reason = e.what();
            periodic.reset();
        }
    }
    const auto in_tiles = s.input_width / s.tile, out_tiles = s.output_width / s.tile;
    matrix.outputs.resize(out_tiles);
    std::vector<double> slots(degree / 2);
    Plaintext encoded;
    for (std::size_t o = 0; o < out_tiles; ++o) {
        auto &out = matrix.outputs[o];
        std::map<int, std::vector<Impl::Term>> groups;
        for (int d = -static_cast<int>(s.tile) + 1; d < static_cast<int>(s.tile); ++d) {
            const auto split = split_diagonal(d, s.baby);
            for (std::size_t i = 0; i < in_tiles; ++i) {
                const auto period = diagonal_period(matrix.weights.data(), s.input_width, s.output_width,
                                                    i, o, s.tile, d, split.giant);
                bool used_periodic = false;
                if (periodic) {
                    try {
                        const auto small = periodic->encode(period, plain_scale);
                        periodic->materialize(small, plain_scale, encoded);
                        used_periodic = true;
                    } catch (const std::bad_alloc &) { throw; }
                    catch (const std::exception &e) {
                        if (matrix.stats.periodic_fallback_reason.empty())
                            matrix.stats.periodic_fallback_reason = e.what();
                    }
                }
                const bool audit_native = used_periodic && o == 0 && i == 0 && (d == -255 || d == 0 || d == 255);
                if (!used_periodic || audit_native)
                    for (std::size_t j = 0; j < slots.size(); ++j) slots[j] = period[j % s.tile];
                if (!used_periodic) {
                    encoder.encode(slots, input->parms_id(), plain_scale, encoded);
                    ++matrix.stats.strict_encoded_terms;
                    matrix.stats.periodic_encoding_fallbacks += s.periodic_encoding;
                } else {
                    ++matrix.stats.periodic_encoded_terms;
                    if (audit_native) {
                        Plaintext oracle;
                        encoder.encode(slots, input->parms_id(), plain_scale, oracle);
                        std::uint64_t differences = 0;
                        for (std::size_t word = 0; word < degree * qin; ++word)
                            differences += oracle.data()[word] != encoded.data()[word];
                        ++matrix.stats.native_encoder_audit_terms;
                        matrix.stats.native_encoder_mismatch_terms += differences != 0;
                        matrix.stats.native_encoder_mismatch_words += differences;
                    }
                }
                // This exact full-N verifier is deliberately common to both
                // encoders: compact storage never approximates RNS residues.
                auto host = compress_exact(encoded.data(), degree, qin, 2 * s.tile);
                GpuCompressedPlaintextQP compact;
                compact.meta = {encoded.parms_id(), encoded.scale(), true, degree, qin, 0};
                compact.period = host.period;
                compact.residues.allocate(host.values.size(), impl_->device);
                compact.residues.copy_from_host(host.values.data(), host.values.size());
                std::vector<GpuWord> actual(host.values.size());
                compact.residues.copy_to_host(actual.data(), actual.size());
                if (actual != host.values) throw std::runtime_error("compact plaintext device roundtrip differs");
                compact.exact_device_reconstruction = true;
                matrix.stats.compact_bytes += compact.residues.size() * sizeof(GpuWord);
                matrix.stats.full_equivalent_bytes += degree * qin * sizeof(GpuWord);
                matrix.stats.compression_fallbacks += host.fallback();
                matrix.stats.max_period = std::max<std::uint64_t>(matrix.stats.max_period, host.period);
                ++matrix.stats.cp_terms;
                groups[split.giant].push_back({i, split.baby, std::move(compact)});
            }
        }
        for (auto &entry : groups) out.groups.push_back({entry.first, std::move(entry.second)});
        for (std::size_t j = 0; j < slots.size(); ++j) slots[j] = matrix.bias[o * s.tile + j % s.tile];
        encoder.encode(slots, output->parms_id(), std::exp2(s.output_scale_log2), encoded);
        out.bias = GpuUploader::upload_plaintext(encoded, impl_->device);
        const auto bias_bytes = degree * qout * sizeof(GpuWord);
        matrix.stats.compact_bytes += bias_bytes;
        matrix.stats.full_equivalent_bytes += bias_bytes;
    }
    if (matrix.stats.periodic_encoded_terms) {
        matrix.stats.encoding_mode = "subring512_ckks_lift";
        matrix.stats.native_encoder_bitwise_guaranteed = false;
    }
    matrix.stats.baby_rotations = in_tiles * (s.baby - 1);
    matrix.stats.giant_rotations = out_tiles * (2 * s.tile / s.baby - 1);
    gpu_check_cuda(cudaStreamSynchronize(impl_->stream), "compact prepare synchronize");
    matrix.stats.setup_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    impl_->matrices.emplace(s.id, std::move(matrix));
}

GpuCiphertextData CompactLinearEngine::run(const std::string &id, std::size_t output_tile,
        const std::vector<const GpuCiphertextData *> &inputs) {
    auto &p = *impl_;
    auto &m = p.matrices.at(id);
    const auto &s = m.spec;
    if (output_tile >= m.outputs.size() || inputs.size() != s.input_width / s.tile)
        throw std::invalid_argument("compact Linear input/output tile count mismatch");
    gpu_check_cuda(cudaSetDevice(p.device), "compact run cudaSetDevice");
    ScopedGpuExecutionStream guard(p.stream);
    // Keep the asynchronous H2D source alive past FailureGuard's sync, even
    // if a later scratch allocation/kernel/scale check throws.
    std::vector<detail::MacTerm> host_terms;
    struct FailureGuard {
        Impl &impl;
        int exceptions = std::uncaught_exceptions();
        ~FailureGuard() {
            if (std::uncaught_exceptions() > exceptions) {
                (void)cudaStreamSynchronize(impl.stream);
                impl.clear_sequence();
            }
        }
    } failure_guard{p};
    auto &runtime = p.runtime(s);
    const auto &level = runtime.parameters.get_level_by_q_count(s.input_level + 1);
    for (const auto *ct : inputs) {
        if (!ct || ct->meta.degree != level.degree || ct->meta.q_count != level.q_count ||
            ct->meta.parms_id != level.parms_id || !(ct->meta.scale > 0) || !std::isfinite(ct->meta.scale) ||
            std::abs(std::log2(ct->meta.scale) - s.input_scale_log2) > 1e-8 ||
            std::abs(std::log2(ct->meta.scale / inputs.front()->meta.scale)) > 1e-10)
            throw std::invalid_argument("compact Linear input physical level/scale mismatch");
        component(*ct, 0, p.device); component(*ct, 1, p.device);
    }
    if (output_tile == 0) p.build_babies(m, runtime, inputs);
    else {
        if (p.sequence_matrix != id || p.next_output != output_tile || p.sequence_inputs != inputs)
            throw std::logic_error("compact Linear must run output0,1,...,last for one immutable input sequence");
        for (std::size_t i = 0; i < inputs.size(); ++i)
            if (component(*inputs[i], 0, p.device) != p.sequence_allocations[i])
                throw std::logic_error("compact Linear input allocation changed within sequence");
    }
    const auto &out = m.outputs[output_tile];
    std::vector<std::size_t> offsets{0};
    for (const auto &group : out.groups) {
        for (const auto &term : group.terms) {
            const auto &ct = term.baby ? p.babies[term.input_tile][term.baby - 1] : *inputs[term.input_tile];
            host_terms.push_back({component(ct, 0, p.device), component(ct, 1, p.device),
                term.plain.residues.data(), static_cast<std::uint32_t>(term.plain.period), 0});
        }
        offsets.push_back(host_terms.size());
    }
    if (p.descriptors.size() < host_terms.size()) p.descriptors.allocate(host_terms.size(), p.device);
    // One descriptor transfer per output, not one sync per MAC. host_terms
    // remains alive until the single final stream synchronization below.
    gpu_check_cuda(cudaMemcpyAsync(p.descriptors.data(), host_terms.data(),
        host_terms.size() * sizeof(detail::MacTerm), cudaMemcpyHostToDevice, p.stream), "compact descriptor upload");
    auto meta = inputs.front()->meta;
    meta.scale *= std::exp2(m.stats.plaintext_scale_log2);
    auto group = GpuCiphertextData::allocate_single_device(level.degree, level.q_count, 2, p.device);
    group.meta = meta;
    GpuCiphertextData accumulator;
    GpuCiphertextData sum_scratch;
    bool initialized = false;
    const auto &tables = level.shards.at(0);
    for (std::size_t g = 0; g < out.groups.size(); ++g) {
        detail::launch_compact_group_mac(component(group, 0, p.device), component(group, 1, p.device),
            p.descriptors.data() + offsets[g], offsets[g + 1] - offsets[g],
            tables.q_primes.data(), tables.q_modulus_constants.data(), level.degree, level.q_count);
        // All input tiles were accumulated by this one Q MAC before the
        // group's single giant rotation. NEVER rotate per input tile.
        if (out.groups[g].giant)
            p.rotate_inplace(runtime, group, out.groups[g].giant);
        if (!initialized) {
            accumulator = GpuCiphertextData::allocate_single_device(level.degree, level.q_count, 2, p.device);
            accumulator.meta = meta;
            for (std::size_t c = 0; c < 2; ++c)
                gpu_check_cuda(cudaMemcpyAsync(component(accumulator, c, p.device), component(group, c, p.device),
                    level.degree * level.q_count * sizeof(GpuWord), cudaMemcpyDeviceToDevice, p.stream), "compact initialize accumulator");
            initialized = true;
        } else {
            // Ordinary GpuEvaluator::add is NOT safe when destination is one
            // of its inputs: its prepare helper can invalidate a referenced
            // metadata/source object. Ping-pong distinct owners instead.
            runtime.evaluator.add(accumulator, group, sum_scratch);
            std::swap(accumulator, sum_scratch);
        }
    }
    GpuCiphertextData result;
    runtime.evaluator.rescale_many(accumulator, result, s.rescale_count);
    if (result.meta.q_count != s.output_level + 1 ||
        std::abs(std::log2(result.meta.scale) - s.output_scale_log2) > 1e-6)
        throw std::runtime_error("compact Linear real rescale did not produce declared physical output");
    runtime.evaluator.add_plain_inplace(result, out.bias);
    gpu_check_cuda(cudaStreamSynchronize(p.stream), "compact Linear output synchronize");
    ++p.stats.run_calls;
    p.next_output = output_tile + 1;
    if (p.next_output == m.outputs.size()) p.clear_sequence();
    return result;
}

void CompactLinearEngine::reset_sequence() {
    if (!impl_->stream) return;
    ScopedGpuExecutionStream guard(impl_->stream);
    gpu_check_cuda(cudaStreamSynchronize(impl_->stream), "compact reset synchronize");
    impl_->clear_sequence();
}
const MatrixStats &CompactLinearEngine::matrix_stats(const std::string &id) const {
    return impl_->matrices.at(id).stats;
}
EngineStats CompactLinearEngine::engine_stats() const { return impl_->stats; }

} // namespace bert::compact
