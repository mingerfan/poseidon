#pragma once

#ifdef POSEIDON_S2C_FIRST_ENABLE_TIMING
#include "gpu_api_trace.h"
#include <initializer_list>
#include <utility>

namespace poseidon::gpu {

// A non-owning facade. With no ScopedApiTrace it calls the original evaluator;
// with tracing enabled it first resolves real ciphertext SSA versions, executes
// the same GPU API, and snapshots the actual result (including raw size-3 mul).
class TracedGpuEvaluator {
public:
    using CT = GpuCiphertextData;
    using Value = ApiTraceValue;
    explicit TracedGpuEvaluator(const GpuEvaluator &eval) : eval_(&eval) {}

#define POSEIDON_TRACE_BINARY(NAME) \
    void NAME(const CT &a, const CT &b, CT &out) const { \
        const auto *e = eval_; \
        capture(#NAME, {&a, &b}, {}, "", out, \
            [e](const std::vector<Value> &v, CT) { \
                CT result; e->NAME(*v[0], *v[1], result); return Value(std::move(result)); \
            }, [&] { e->NAME(a, b, out); }); \
    }
    POSEIDON_TRACE_BINARY(add)
    POSEIDON_TRACE_BINARY(sub)
    POSEIDON_TRACE_BINARY(multiply)
#undef POSEIDON_TRACE_BINARY

#define POSEIDON_TRACE_UNARY(NAME) \
    void NAME(const CT &a, CT &out) const { \
        const auto *e = eval_; \
        capture(#NAME, {&a}, {}, "", out, \
            [e](const std::vector<Value> &v, CT) { \
                CT result; e->NAME(*v[0], result); return Value(std::move(result)); \
            }, [&] { e->NAME(a, out); }); \
    }
    POSEIDON_TRACE_UNARY(negate)
    POSEIDON_TRACE_UNARY(square)
    POSEIDON_TRACE_UNARY(rescale)
    POSEIDON_TRACE_UNARY(rescale_x2)
#undef POSEIDON_TRACE_UNARY

#define POSEIDON_TRACE_PLAIN(NAME) \
    void NAME(const CT &a, const GpuPlaintextData &plain, CT &out) const { \
        const auto *e = eval_; const auto *p = &plain; \
        capture(#NAME, {&a}, {{"plain", p}}, "", out, \
            [e, p](const std::vector<Value> &v, CT) { \
                CT result; e->NAME(*v[0], *p, result); return Value(std::move(result)); \
            }, [&] { e->NAME(a, plain, out); }); \
    }
    POSEIDON_TRACE_PLAIN(add_plain)
    POSEIDON_TRACE_PLAIN(sub_plain)
    POSEIDON_TRACE_PLAIN(multiply_plain)
    POSEIDON_TRACE_PLAIN(multiply_plain_q_prefix)
#undef POSEIDON_TRACE_PLAIN

    void add_plain_inplace(CT &a, const GpuPlaintextData &plain) const {
        const auto *e = eval_; const auto *p = &plain;
        capture("add_plain_inplace", {&a}, {{"plain", p}}, "", a,
            [e, p](const std::vector<Value> &, CT result) {
                e->add_plain_inplace(result, *p); return Value(std::move(result));
            }, [&] { e->add_plain_inplace(a, plain); }, true);
    }
    void sub_plain_inplace(CT &a, const GpuPlaintextData &plain) const {
        const auto *e = eval_; const auto *p = &plain;
        capture("sub_plain_inplace", {&a}, {{"plain", p}}, "", a,
            [e, p](const std::vector<Value> &, CT result) {
                e->sub_plain_inplace(result, *p); return Value(std::move(result));
            }, [&] { e->sub_plain_inplace(a, plain); }, true);
    }
    void multiply_q_prefix(const CT &a, const CT &b, CT &out, parms_id_type target) const {
        const auto *e = eval_;
        capture("multiply_q_prefix", {&a, &b}, {}, ".parms" + api_trace_detail::parms_signature(target), out,
            [e, target](const std::vector<Value> &v, CT) {
                CT result; e->multiply_q_prefix(*v[0], *v[1], result, target); return Value(std::move(result));
            }, [&] { e->multiply_q_prefix(a, b, out, target); });
    }
    void relinearize(const CT &a, const GpuRelinKeysData &keys, CT &out) const {
        const auto *e = eval_; const auto *k = &keys;
        capture("relinearize", {&a}, {{"relin", k}}, "", out,
            [e, k](const std::vector<Value> &v, CT) {
                CT result; e->relinearize(*v[0], *k, result); return Value(std::move(result));
            }, [&] { e->relinearize(a, keys, out); });
    }
    void rescale_many(const CT &a, CT &out, std::uint32_t count) const {
        const auto *e = eval_;
        capture("rescale_many", {&a}, {}, ".count" + std::to_string(count), out,
            [e, count](const std::vector<Value> &v, CT) {
                CT result; e->rescale_many(*v[0], result, count); return Value(std::move(result));
            }, [&] { e->rescale_many(a, out, count); });
    }
    void drop_modulus(const CT &a, CT &out, parms_id_type target) const {
        const auto *e = eval_;
        capture("drop_modulus", {&a}, {}, ".parms" + api_trace_detail::parms_signature(target), out,
            [e, target](const std::vector<Value> &v, CT) {
                CT result; e->drop_modulus(*v[0], result, target); return Value(std::move(result));
            }, [&] { e->drop_modulus(a, out, target); });
    }
    void drop_modulus_inplace(CT &a, parms_id_type target) const {
        const auto *e = eval_;
        capture("drop_modulus_inplace", {&a}, {}, ".parms" + api_trace_detail::parms_signature(target), a,
            [e, target](const std::vector<Value> &, CT result) {
                e->drop_modulus_inplace(result, target); return Value(std::move(result));
            }, [&] { e->drop_modulus_inplace(a, target); }, true);
    }
    void set_scale(CT &a, double scale) const {
        if (!std::isfinite(scale) || scale <= 0) throw std::runtime_error("invalid traced ciphertext scale");
        capture("set_scale", {&a}, {}, ".bits" + api_trace_detail::scalar_signature(scale), a,
            [scale](const std::vector<Value> &, CT result) {
                result.meta.scale = scale; return Value(std::move(result));
            }, [&] { a.meta.scale = scale; }, true);
    }
    void multiply_plain_accumulate(const CT &source, const GpuPlaintextData &plain, CT &accumulator) const {
        const auto *e = eval_; const auto *p = &plain;
        // The old accumulator version is an explicit dependency, even though
        // the original C++ API spells it as the destination argument.
        capture("multiply_plain_accumulate", {&accumulator, &source}, {{"plain", p}}, "", accumulator,
            [e, p](const std::vector<Value> &v, CT result) {
                e->multiply_plain_accumulate(*v[1], *p, result); return Value(std::move(result));
            }, [&] { e->multiply_plain_accumulate(source, plain, accumulator); }, true);
    }
    void multiply_plain_sum_q_prefix(const std::vector<const CT *> &sources,
        const std::vector<const GpuPlaintextData *> &plains, CT &out, bool accumulate = false) const {
        const auto *e = eval_;
        if (!active_api_trace_recorder) { e->multiply_plain_sum_q_prefix(sources, plains, out, accumulate); return; }
        std::vector<const CT *> dependencies;
        if (accumulate) dependencies.push_back(&out);
        dependencies.insert(dependencies.end(), sources.begin(), sources.end());
        std::vector<ApiTraceRecorder::Resource> resources;
        for (const auto *p : plains) resources.push_back({"plain", p});
        // Capture the resident plaintext pointers, never the source CT pointers.
        capture("multiply_plain_sum_q_prefix", dependencies, resources,
            ".terms" + std::to_string(sources.size()) + ".acc" + std::to_string(accumulate), out,
            [e, plains, accumulate](const std::vector<Value> &v, CT result) {
                std::vector<const CT *> terms;
                for (std::size_t i = accumulate ? 1 : 0; i < v.size(); ++i) terms.push_back(&*v[i]);
                e->multiply_plain_sum_q_prefix(terms, plains, result, accumulate);
                return Value(std::move(result));
            }, [&] { e->multiply_plain_sum_q_prefix(sources, plains, out, accumulate); }, accumulate);
    }
    void double_sub_plain_q_prefix(const CT &source, const GpuPlaintextData &plain,
        CT &out, const CT *correction = nullptr) const {
        const auto *e = eval_; const auto *p = &plain;
        std::vector<const CT *> dependencies{&source};
        if (correction) dependencies.push_back(correction);
        const bool corrected = correction != nullptr;
        capture("double_sub_plain_q_prefix", dependencies, {{"plain", p}},
            ".correction" + std::to_string(corrected), out,
            [e, p, corrected](const std::vector<Value> &v, CT) {
                CT result; e->double_sub_plain_q_prefix(*v[0], *p, result, corrected ? &*v[1] : nullptr);
                return Value(std::move(result));
            }, [&] { e->double_sub_plain_q_prefix(source, plain, out, correction); });
    }
    void rotate(const CT &a, int step, const GpuGaloisKeysData &keys, CT &out) const {
        const auto *e = eval_; const auto *k = &keys;
        capture("rotate", {&a}, {{"galois", k}}, ".step" + std::to_string(step), out,
            [e, k, step](const std::vector<Value> &v, CT) {
                CT result; e->rotate(*v[0], step, *k, result); return Value(std::move(result));
            }, [&] { e->rotate(a, step, keys, out); });
    }
    void rotate(const CT &a, int step, const GpuGaloisKeysData &keys, CT &out,
        GpuRotationHoistCache *cache, std::uintptr_t tag) const {
        reject_legacy("rotate_with_cache");
        eval_->rotate(a, step, keys, out, cache, tag);
    }
    void rotate_many_hoisted(const CT &source, const std::vector<int> &steps,
        const GpuGaloisKeysData &keys, GpuDoubleHoistWorkspace &workspace,
        std::vector<CT> &out) const {
        const auto *e = eval_; const auto *k = &keys; auto *w = &workspace;
        auto *recorder = active_api_trace_recorder;
        if (!recorder) { e->rotate_many_hoisted(source, steps, keys, workspace, out); return; }
        const auto source_id = recorder->input(source);
        out.clear();
        out.resize(steps.size());
        if (steps.empty()) return;
        GpuHoistedDecomposition hoisted;
        const auto prefix_callee = recorder->signature("rotation_hoist", e, {{"galois", k}, {"hoist_workspace", w}});
        auto start = Clock::now();
        e->rotation_hoist(source, keys, hoisted, workspace);
        const auto hoist_id = recorder->record_buffer("rotation_hoist", prefix_callee, {source_id}, hoisted,
            [e, k, w](const std::vector<Value> &v, CT) {
                GpuHoistedDecomposition result;
                e->rotation_hoist(*v[0], *k, result, *w);
                return Value(std::move(result));
            }, elapsed(start));
        for (std::size_t i = 0; i < steps.size(); ++i) {
            const int step = steps[i];
            const auto callee = recorder->signature("rotate_from_hoist", e,
                {{"galois", k}, {"hoist_workspace", w}}, ".step" + std::to_string(step));
            start = Clock::now();
            e->rotate_from_hoist(source, hoisted, step, keys, workspace, out[i]);
            recorder->record_ct("rotate_from_hoist", callee, {source_id, hoist_id}, out[i],
                [e, k, w, step](const std::vector<Value> &v, CT) {
                    CT result; e->rotate_from_hoist(*v[0], v[1].hoisted(), step, *k, *w, result);
                    return Value(std::move(result));
                }, false, elapsed(start));
        }
        // No recorded lambda owns hoisted. Only its immutable descriptor was
        // retained; this capture-time allocation dies at the end of this loop.
    }
    void bootstrap(const CT &a, const GpuBootstrapData &data, const GpuRelinKeysData &relin,
        const GpuGaloisKeysData &galois, GpuBootstrapWorkspace &workspace, CT &out) const {
        const auto *e = eval_; const auto *d = &data; const auto *r = &relin;
        const auto *g = &galois; auto *w = &workspace;
        capture("bootstrap", {&a}, {{"boot_data", d}, {"relin", r}, {"galois", g}, {"boot_workspace", w}}, "", out,
            [e, d, r, g, w](const std::vector<Value> &v, CT) {
                CT result; e->bootstrap(*v[0], *d, *r, *g, *w, result); return Value(std::move(result));
            }, [&] { e->bootstrap(a, data, relin, galois, workspace, out); });
    }

    // Old diagnostic/expanded-Boot entry points still compile and behave
    // identically outside tracing. Online capture intentionally admits only
    // native bootstrap as one atomic API, never a synthetic expanded fallback.
#define POSEIDON_TRACE_LEGACY(NAME) \
    template<class... Args> decltype(auto) NAME(Args &&...args) const { \
        reject_legacy(#NAME); return eval_->NAME(std::forward<Args>(args)...); \
    }
    POSEIDON_TRACE_LEGACY(dft)
    POSEIDON_TRACE_LEGACY(dft_loose)
    POSEIDON_TRACE_LEGACY(dft_double_hoist)
    POSEIDON_TRACE_LEGACY(coeff_to_slot)
    POSEIDON_TRACE_LEGACY(coeff_to_slot_loose)
    POSEIDON_TRACE_LEGACY(coeff_to_slot_double_hoist)
    POSEIDON_TRACE_LEGACY(raise_modulus)
    POSEIDON_TRACE_LEGACY(eval_mod_high_precision)
    POSEIDON_TRACE_LEGACY(conjugate)
    POSEIDON_TRACE_LEGACY(conjugate_pre_rotated)
    POSEIDON_TRACE_LEGACY(multiply_scalar)
    POSEIDON_TRACE_LEGACY(bootstrap_prepare_modraise_input)
    POSEIDON_TRACE_LEGACY(bootstrap_raise)
    POSEIDON_TRACE_LEGACY(bootstrap_linear_layer)
    POSEIDON_TRACE_LEGACY(bootstrap_bsgs_group)
    POSEIDON_TRACE_LEGACY(bootstrap_conjugate)
    POSEIDON_TRACE_LEGACY(bootstrap_set_scale)
    POSEIDON_TRACE_LEGACY(multiply_plain_q_view)
    POSEIDON_TRACE_LEGACY(multiply_by_diag_matrix_bsgs)
    POSEIDON_TRACE_LEGACY(multiply_by_diag_matrix_bsgs_loose)
    POSEIDON_TRACE_LEGACY(multiply_by_diag_matrix_bsgs_double_hoist)
    POSEIDON_TRACE_LEGACY(relinearize_rescale_x2_hybrid)
    POSEIDON_TRACE_LEGACY(rotate_inplace)
    POSEIDON_TRACE_LEGACY(switch_key)
    POSEIDON_TRACE_LEGACY(ntt_fwd)
    POSEIDON_TRACE_LEGACY(ntt_inv)
    POSEIDON_TRACE_LEGACY(rescale_dynamic)
    POSEIDON_TRACE_LEGACY(retained_workspace_bytes)
#undef POSEIDON_TRACE_LEGACY

private:
    using Clock = std::chrono::steady_clock;
    static std::uint64_t elapsed(Clock::time_point start) {
        return std::max<std::uint64_t>(1, std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
    }
    static void reject_legacy(const char *name) {
        if (active_api_trace_recorder)
            throw std::runtime_error(std::string("unsupported legacy GPU API while tracing: ") + name +
                "; trace native bootstrap atomically");
    }
    template<class Replay, class Run>
    void capture(const char *opcode, const std::vector<const CT *> &sources,
        const std::vector<ApiTraceRecorder::Resource> &resources, const std::string &parameters,
        CT &out, Replay replay, Run run, bool mutates_first = false) const {
        auto *recorder = active_api_trace_recorder;
        if (!recorder) { run(); return; }
        auto dependencies = recorder->inputs(sources); // before any aliased/in-place write
        auto callee = recorder->signature(opcode, eval_, resources, parameters);
        const auto start = Clock::now();
        run();
        recorder->record_ct(opcode, std::move(callee), std::move(dependencies), out,
            std::move(replay), mutates_first, elapsed(start));
    }
    const GpuEvaluator *eval_;
};

inline TracedGpuEvaluator trace_gpu(GpuEvaluator &eval) { return TracedGpuEvaluator(eval); }
inline TracedGpuEvaluator trace_gpu(const GpuEvaluator &eval) { return TracedGpuEvaluator(eval); }

inline void set_trace_scale(GpuEvaluator &eval, GpuCiphertextData &value, double scale) {
    trace_gpu(eval).set_scale(value, scale);
}

} // namespace poseidon::gpu
#else
// Standalone diagnostics do not link the RuntimePlan frontend.
#include "poseidon/gpu/gpu_evaluator.h"
namespace poseidon::gpu {
inline GpuEvaluator &trace_gpu(GpuEvaluator &eval) { return eval; }
inline const GpuEvaluator &trace_gpu(const GpuEvaluator &eval) { return eval; }
inline void set_trace_scale(GpuEvaluator &, GpuCiphertextData &value, double scale) {
    value.meta.scale = scale;
}
} // namespace poseidon::gpu
#endif
