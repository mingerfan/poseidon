#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_scale_planner.h"
#include "poseidon/gpu/kernels/gpu_double_hoist_kernels.h"
#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"
#include "poseidon/gpu/kernels/gpu_keyswitch_kernels.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"
#include "poseidon/gpu/kernels/gpu_rescale_kernels.h"

#include "poseidon/advance/homomorphic_linear_transform.h"

#include <nvtx3/nvToolsExt.h>

#include <stdexcept>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <map>
#include <functional>
#include <string>
#include <utility>

namespace poseidon
{
namespace gpu
{

namespace
{

class NvtxRange
{
public:
    explicit NvtxRange(const char *name)
    {
        nvtxRangePushA(name);
    }

    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;

    ~NvtxRange()
    {
        nvtxRangePop();
    }
};

bool same_scale(double a, double b)
{
    const double tolerance =
        1e-6 * std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= tolerance;
}

bool use_rescale_x2()
{
    const char *raw = std::getenv("POSEIDON_RESCALE_X2");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_relinearize_rescale_x2()
{
    const char *raw = std::getenv("POSEIDON_RELIN_RESCALE_X2");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_double_hoist_direct_giant_accumulate()
{
    const char *raw =
        std::getenv("POSEIDON_DOUBLE_HOIST_DIRECT_GIANT_ACCUMULATE");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

// The key a key switch at this ciphertext's level uses: its per-level
// override when the key carries one (level-aware special modulus), checked
// against the level's own P so a key and a level built for different special
// moduli never meet silently.
const GpuEvaluationKeyData &level_key_for(
    const GpuParameterData &params,
    const GpuEvaluationKeyData &keys,
    const GpuCiphertextMeta &meta,
    const char *name)
{
    const auto &selected = keys.for_level(meta.q_count);
    if (selected.meta.p_count == 0 || meta.q_count == 0)
    {
        return selected;  // empty input: the entry point's own validation reports it
    }
    const GpuLevelInfo *found = nullptr;
    try
    {
        found = &params.get_level(meta.parms_id);
    }
    catch (const std::out_of_range &)
    {
        return selected;
    }
    const auto &level = *found;
    if (!level.shards.empty() && level.shards.front().hybrid_base_p_count != 0 &&
        level.shards.front().hybrid_base_p_count != selected.meta.p_count)
    {
        throw std::invalid_argument(
            std::string(name) + ": key special modulus (P=" +
            std::to_string(selected.meta.p_count) + ") differs from level q" +
            std::to_string(meta.q_count) + " (P=" +
            std::to_string(level.shards.front().hybrid_base_p_count) + ")");
    }
    return selected;
}

// Keep c0 of every giant group in QP (Lattigo's layout): only c1 of the
// non-identity groups is moved down to Q for the giant key switch, and the
// giant accumulation reads c0 straight from the QP group accumulators. This
// removes group_count + 1 of the 2 * group_count inner ModDown components.
// On by default (Boot -7% on the 4090, same errors); POSEIDON_DOUBLE_HOIST_QP_C0=0
// restores the lifted path, whose rounding differs.
bool use_double_hoist_qp_c0()
{
    const char *raw = std::getenv("POSEIDON_DOUBLE_HOIST_QP_C0");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_double_hoist_qp_mac_direct_init()
{
    const char *raw =
        std::getenv("POSEIDON_DOUBLE_HOIST_QP_MAC_DIRECT_INIT");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_double_hoist_fused_baby_keyswitch_c0()
{
    const char *raw = std::getenv(
        "POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_C0");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_double_hoist_fused_baby_keyswitch_plain_mac()
{
    const char *raw = std::getenv(
        "POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_PLAIN_MAC");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_double_hoist_batched_giant_intt()
{
    const char *raw = std::getenv(
        "POSEIDON_DOUBLE_HOIST_BATCHED_GIANT_INTT");
    bool enabled = true;
    if (raw != nullptr && *raw != '\0')
    {
        const std::string value(raw);
        enabled = value != "0" &&
                  value != "OFF" &&
                  value != "off" &&
                  value != "false" &&
                  value != "FALSE";
    }
    if (!enabled)
    {
        return false;
    }

    const char *ntt_raw = std::getenv("POSEIDON_NTT_ALGO");
    if (ntt_raw == nullptr || *ntt_raw == '\0')
    {
        return true;
    }
    const std::string ntt_value(ntt_raw);
    return ntt_value == "fourstep" ||
           ntt_value == "four_step" ||
           ntt_value == "cheddar";
}

bool use_evalmod_inline_leaf()
{
    const char *raw = std::getenv("POSEIDON_EVALMOD_INLINE_LEAF");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_evalmod_lazy_relinearization()
{
    const char *raw = std::getenv("POSEIDON_EVALMOD_LAZY_RELIN");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_evalmod_caccum_leaf()
{
    const char *raw = std::getenv("POSEIDON_EVALMOD_CACCUM_LEAF");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_evalmod_pair()
{
    const char *raw = std::getenv("POSEIDON_EVAL_MOD_PAIR");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_evalmod_d2d_free_dataflow()
{
    const char *raw = std::getenv("POSEIDON_EVALMOD_D2D_FREE_DATAFLOW");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_evalmod_zero_copy_moddrop()
{
    const char *raw = std::getenv("POSEIDON_EVALMOD_ZERO_COPY_MODDROP");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_evalmod_q_prefix_views()
{
    const char *raw = std::getenv("POSEIDON_EVALMOD_Q_PREFIX_VIEWS");
    if (raw == nullptr || *raw == '\0')
    {
        return true;
    }
    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

class EvalModStageEventRecorder
{
public:
    explicit EvalModStageEventRecorder(bool enabled)
        : enabled_(enabled)
    {
        if (!enabled_)
        {
            return;
        }
        try
        {
            for (auto &event : events_)
            {
                gpu_check_cuda(
                    cudaEventCreate(&event),
                    "EvalMod stage cudaEventCreate");
                ++created_count_;
            }
            record(0);
        }
        catch (...)
        {
            destroy_events();
            throw;
        }
    }

    EvalModStageEventRecorder(const EvalModStageEventRecorder &) = delete;
    EvalModStageEventRecorder &operator=(
        const EvalModStageEventRecorder &) = delete;

    ~EvalModStageEventRecorder()
    {
        destroy_events();
    }

    void record(std::size_t boundary)
    {
        if (!enabled_)
        {
            return;
        }
        if (boundary >= events_.size())
        {
            throw std::out_of_range("EvalMod stage boundary is out of range");
        }
        gpu_check_cuda(
            cudaEventRecord(events_[boundary]),
            "EvalMod stage cudaEventRecord");
    }

    void finish(GpuBootstrapWorkspace::EvalModStageTiming &timing)
    {
        if (!enabled_)
        {
            return;
        }
        record(6);
        gpu_check_cuda(
            cudaEventSynchronize(events_[6]),
            "EvalMod stage cudaEventSynchronize");

        timing.input_preparation_ms = elapsed(0, 1);
        timing.basis_generation_ms = elapsed(1, 2);
        timing.leaf_evaluation_ms = elapsed(2, 3);
        timing.bsgs_combine_ms = elapsed(3, 4);
        timing.double_angle_ms = elapsed(4, 5);
        timing.output_alignment_ms = elapsed(5, 6);
        timing.total_ms = elapsed(0, 6);
    }

private:
    double elapsed(std::size_t begin, std::size_t end) const
    {
        float milliseconds = 0.0F;
        gpu_check_cuda(
            cudaEventElapsedTime(
                &milliseconds,
                events_[begin],
                events_[end]),
            "EvalMod stage cudaEventElapsedTime");
        return static_cast<double>(milliseconds);
    }

    void destroy_events() noexcept
    {
        for (std::size_t index = 0; index < created_count_; ++index)
        {
            (void)cudaEventDestroy(events_[index]);
            events_[index] = nullptr;
        }
        created_count_ = 0;
    }

    bool enabled_{false};
    std::array<cudaEvent_t, 7> events_{};
    std::size_t created_count_{0};
};

class EvalModMultiplyEventRecorder
{
public:
    explicit EvalModMultiplyEventRecorder(bool enabled)
        : enabled_(enabled)
    {}

    EvalModMultiplyEventRecorder(const EvalModMultiplyEventRecorder &) = delete;
    EvalModMultiplyEventRecorder &operator=(
        const EvalModMultiplyEventRecorder &) = delete;

    ~EvalModMultiplyEventRecorder()
    {
        for (auto &entry : entries_)
        {
            if (entry.begin != nullptr)
            {
                (void)cudaEventDestroy(entry.begin);
            }
            if (entry.end != nullptr)
            {
                (void)cudaEventDestroy(entry.end);
            }
        }
    }

    void begin(
        std::string label,
        std::size_t q_count,
        std::size_t p_count,
        bool is_square)
    {
        if (!enabled_)
        {
            return;
        }
        if (active_)
        {
            throw std::logic_error(
                "EvalMod multiply timing ranges cannot overlap");
        }
        Entry entry;
        entry.timing.label = std::move(label);
        entry.timing.q_count = q_count;
        entry.timing.decomposition_count = p_count == 0
            ? 0
            : (q_count + p_count - 1) / p_count;
        entry.timing.is_square = is_square;
        gpu_check_cuda(
            cudaEventCreate(&entry.begin),
            "EvalMod multiply begin cudaEventCreate");
        try
        {
            gpu_check_cuda(
                cudaEventCreate(&entry.end),
                "EvalMod multiply end cudaEventCreate");
            gpu_check_cuda(
                cudaEventRecord(entry.begin),
                "EvalMod multiply begin cudaEventRecord");
            nvtxRangePushA(entry.timing.label.c_str());
            entries_.push_back(std::move(entry));
            entry.begin = nullptr;
            entry.end = nullptr;
            active_ = true;
        }
        catch (...)
        {
            if (entry.begin != nullptr)
            {
                (void)cudaEventDestroy(entry.begin);
            }
            if (entry.end != nullptr)
            {
                (void)cudaEventDestroy(entry.end);
            }
            throw;
        }
    }

    void end()
    {
        if (!enabled_)
        {
            return;
        }
        if (!active_ || entries_.empty())
        {
            throw std::logic_error(
                "EvalMod multiply timing range is not active");
        }
        gpu_check_cuda(
            cudaEventRecord(entries_.back().end),
            "EvalMod multiply end cudaEventRecord");
        nvtxRangePop();
        active_ = false;
    }

    void finish(
        std::vector<GpuBootstrapWorkspace::EvalModMultiplyTiming> &timings)
    {
        if (!enabled_)
        {
            return;
        }
        if (active_)
        {
            throw std::logic_error(
                "EvalMod multiply timing range was not closed");
        }
        timings.clear();
        timings.reserve(entries_.size());
        if (!entries_.empty())
        {
            gpu_check_cuda(
                cudaEventSynchronize(entries_.back().end),
                "EvalMod multiply cudaEventSynchronize");
        }
        for (const auto &entry : entries_)
        {
            float milliseconds = 0.0F;
            gpu_check_cuda(
                cudaEventElapsedTime(
                    &milliseconds,
                    entry.begin,
                    entry.end),
                "EvalMod multiply cudaEventElapsedTime");
            auto timing = entry.timing;
            timing.gpu_ms = static_cast<double>(milliseconds);
            timings.push_back(std::move(timing));
        }
    }

private:
    struct Entry
    {
        GpuBootstrapWorkspace::EvalModMultiplyTiming timing;
        cudaEvent_t begin{nullptr};
        cudaEvent_t end{nullptr};
    };

    bool enabled_{false};
    bool active_{false};
    std::vector<Entry> entries_;
};

bool same_logical_shard_layout(
    const GpuRNSPoly &reference,
    const GpuRNSPoly &candidate)
{
    if (reference.shards.size() != candidate.shards.size())
    {
        return false;
    }

    for (std::size_t i = 0; i < reference.shards.size(); ++i)
    {
        const auto &lhs = reference.shards[i];
        const auto &rhs = candidate.shards[i];
        if (lhs.limb_begin != rhs.limb_begin ||
            lhs.limb_count != rhs.limb_count ||
            lhs.coeff_begin != rhs.coeff_begin ||
            lhs.coeff_count != rhs.coeff_count)
        {
            return false;
        }
    }

    return true;
}

bool all_components_use_layout(
    const GpuCiphertextData &ciphertext,
    const GpuRNSPoly &reference)
{
    for (const auto &poly : ciphertext.polys_)
    {
        if (!same_logical_shard_layout(reference, poly))
        {
            return false;
        }
    }

    return true;
}

void validate_q_prefix_source(
    const char *name,
    const GpuCiphertextData &source,
    const GpuLevelInfo &source_level,
    const GpuLevelInfo &target_level)
{
    if (source.empty() || source.fields_.size() != 1)
    {
        throw std::invalid_argument(std::string(name) +
                                    ": requires single-device ciphertext storage");
    }
    if (!source.meta.is_ntt_form ||
        source.meta.p_count != 0 ||
        source.meta.component_count != source.size() ||
        source_level.q_count != source.meta.q_count ||
        target_level.p_count != 0 ||
        target_level.q_count == 0 ||
        target_level.q_count > source.meta.q_count ||
        target_level.degree != source.meta.degree)
    {
        throw std::invalid_argument(std::string(name) +
                                    ": incompatible Q-prefix metadata");
    }
    for (const auto &poly : source.polys_)
    {
        if (poly.degree != source.meta.degree ||
            poly.q_count != source.meta.q_count ||
            poly.p_count != 0 ||
            poly.shards.size() != 1)
        {
            throw std::invalid_argument(std::string(name) +
                                        ": unsupported component layout");
        }
        const auto &shard = poly.shards.front();
        if (shard.field_index != 0 ||
            shard.limb_begin != 0 ||
            shard.limb_count != source.meta.q_count ||
            shard.coeff_begin != 0 ||
            shard.coeff_count != source.meta.degree)
        {
            throw std::invalid_argument(std::string(name) +
                                        ": requires one full q shard");
        }
    }
}

GpuConstCiphertextView make_q_prefix_view(
    const GpuCiphertextData &source,
    const GpuLevelInfo &target_level)
{
    auto view = source.make_const_view();
    view.meta.parms_id = target_level.parms_id;
    view.meta.q_count = target_level.q_count;
    view.meta.p_count = 0;
    for (auto &poly : view.polys)
    {
        poly.shards.front().limb_count = target_level.q_count;
    }
    return view;
}

GpuRNSPoly make_q_prefix_layout(
    const GpuCiphertextData &source,
    const GpuLevelInfo &target_level)
{
    GpuRNSPoly layout = source.polys_.front();
    layout.q_count = target_level.q_count;
    layout.p_count = 0;
    layout.shards.front().limb_count = target_level.q_count;
    return layout;
}

bool ciphertext_storage_matches(
    const GpuCiphertextData &ciphertext,
    std::size_t degree,
    std::size_t q_count,
    std::size_t p_count,
    std::size_t component_count,
    int device_id,
    const GpuRNSPoly &reference_layout)
{
    if (ciphertext.empty() ||
        ciphertext.fields_.empty() ||
        ciphertext.size() != component_count)
    {
        return false;
    }
    if (ciphertext.fields_.front().device_id != device_id)
    {
        return false;
    }
    for (const auto &poly : ciphertext.polys_)
    {
        if (poly.degree != degree ||
            poly.q_count != q_count ||
            poly.p_count != p_count ||
            !same_logical_shard_layout(reference_layout, poly))
        {
            return false;
        }
    }
    return true;
}

void prepare_ciphertext_destination(
    GpuCiphertextData &destination,
    const GpuCiphertextData *alias_source0,
    const GpuCiphertextData *alias_source1,
    const GpuCiphertextMeta &meta,
    std::size_t component_count,
    int device_id,
    const GpuRNSPoly &reference_layout)
{
    const bool aliases_input =
        &destination == alias_source0 ||
        (alias_source1 != nullptr && &destination == alias_source1);
    if (aliases_input ||
        !ciphertext_storage_matches(
            destination,
            meta.degree,
            meta.q_count,
            meta.p_count,
            component_count,
            device_id,
            reference_layout))
    {
        destination =
            GpuCiphertextData::allocate_single_device_sharded(
                meta.degree,
                meta.q_count,
                component_count,
                device_id,
                reference_layout.shards,
                meta.p_count);
    }

    destination.meta = meta;
    destination.meta.component_count = component_count;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char *what)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    {
        throw std::overflow_error(what);
    }
    return a * b;
}

void validate_ntt_ciphertext_input(
    const char *name,
    const GpuCiphertextData &source_ciphertext,
    bool expect_ntt_form)
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument(std::string(name) + ": empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument(std::string(name) + ": empty ciphertext storage");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument(std::string(name) + ": component metadata mismatch");
    }
    if (source_ciphertext.meta.is_ntt_form != expect_ntt_form)
    {
        throw std::invalid_argument(std::string(name) + ": NTT form mismatch");
    }
    if (source_ciphertext.meta.degree == 0 ||
        source_ciphertext.meta.q_count + source_ciphertext.meta.p_count == 0)
    {
        throw std::invalid_argument(std::string(name) + ": invalid ciphertext shape");
    }

    const auto &reference_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument(std::string(name) + ": shard layout mismatch");
    }
}

void zero_poly(
    GpuRNSPolyView &poly,
    const char *name)
{
    for (const auto &shard : poly.shards)
    {
        const std::size_t word_count = checked_mul(
            shard.limb_count,
            shard.coeff_count,
            "GpuEvaluator zero word count overflow");
        gpu_check_cuda(cudaSetDevice(shard.device_id), name);
        gpu_check_cuda(
            cudaMemsetAsync(
                shard.ptr,
                0,
                word_count * sizeof(GpuWord),
                gpu_execution_stream()),
            name);
    }
}

void copy_poly(
    GpuRNSPolyView &destination,
    const GpuConstRNSPolyView &source,
    const char *name)
{
    if (destination.shards.size() != source.shards.size())
    {
        throw std::invalid_argument(std::string(name) + ": shard count mismatch");
    }

    for (std::size_t i = 0; i < destination.shards.size(); ++i)
    {
        const auto &dst = destination.shards[i];
        const auto &src = source.shards[i];
        if (dst.device_id != src.device_id ||
            dst.limb_begin != src.limb_begin ||
            dst.limb_count != src.limb_count ||
            dst.coeff_begin != src.coeff_begin ||
            dst.coeff_count != src.coeff_count)
        {
            throw std::invalid_argument(std::string(name) + ": shard placement mismatch");
        }

        const std::size_t word_count = checked_mul(
            dst.limb_count,
            dst.coeff_count,
            "GpuEvaluator copy word count overflow");
        gpu_check_cuda(cudaSetDevice(dst.device_id), name);
        gpu_check_cuda(
            cudaMemcpyAsync(
                dst.ptr,
                src.ptr,
                word_count * sizeof(GpuWord),
                cudaMemcpyDeviceToDevice,
                gpu_execution_stream()),
            name);
    }
}

void copy_ciphertext_data(
    const GpuCiphertextData &source,
    GpuCiphertextData &destination,
    const char *name)
{
    if (source.empty())
    {
        throw std::invalid_argument(std::string(name) + ": empty source");
    }
    if (source.fields_.empty())
    {
        throw std::invalid_argument(std::string(name) + ": empty source storage");
    }
    if (source.meta.component_count != source.size())
    {
        throw std::invalid_argument(std::string(name) + ": component metadata mismatch");
    }

    const auto &reference_layout = source.polys_.at(0);
    if (!all_components_use_layout(source, reference_layout))
    {
        throw std::invalid_argument(std::string(name) + ": shard layout mismatch");
    }

    const int device_id = source.fields_.at(0).device_id;
    prepare_ciphertext_destination(
        destination,
        &source,
        nullptr,
        source.meta,
        source.meta.component_count,
        device_id,
        reference_layout);

    auto dst_view = destination.make_view();
    auto src_view = source.make_const_view();
    if (dst_view.polys.size() != src_view.polys.size())
    {
        throw std::invalid_argument(std::string(name) + ": component count mismatch");
    }
    for (std::size_t component = 0; component < src_view.polys.size(); ++component)
    {
        copy_poly(dst_view.polys[component], src_view.polys[component], name);
    }
    destination.meta = source.meta;
}

std::uint32_t galois_elt_from_rotation_step(
    std::size_t degree,
    int step)
{
    if (degree == 0 || degree > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max() / 2))
    {
        throw std::invalid_argument("GpuEvaluator::rotate: invalid degree");
    }
    if ((degree & (degree - 1)) != 0)
    {
        throw std::invalid_argument("GpuEvaluator::rotate: degree must be a power of two");
    }

    const std::uint32_t n = static_cast<std::uint32_t>(degree);
    const std::uint32_t m32 = n << 1;
    const std::uint64_t m = static_cast<std::uint64_t>(m32);

    if (step == 0)
    {
        return m32 - 1;
    }

    const bool negative = step < 0;
    const std::int64_t signed_step = static_cast<std::int64_t>(step);
    const std::uint64_t abs_step = negative
        ? static_cast<std::uint64_t>(-signed_step)
        : static_cast<std::uint64_t>(signed_step);

    if (abs_step >= (static_cast<std::uint64_t>(n) >> 1))
    {
        throw std::invalid_argument("GpuEvaluator::rotate: step count too large");
    }

    std::uint32_t rotation_count = static_cast<std::uint32_t>(abs_step);
    if (negative)
    {
        rotation_count = (n >> 1) - rotation_count;
    }

    /* 与 Poseidon::util::GaloisTool::generator_ 保持一致。 */
    constexpr std::uint64_t generator = 5;
    std::uint64_t galois_elt = 1;
    while (rotation_count-- != 0)
    {
        galois_elt *= generator;
        galois_elt &= m - 1;
    }
    return static_cast<std::uint32_t>(galois_elt);
}

std::size_t galois_key_index(std::uint32_t galois_elt)
{
    if ((galois_elt & 1U) == 0U)
    {
        throw std::invalid_argument("GpuEvaluator::rotate: invalid Galois element");
    }
    return static_cast<std::size_t>((galois_elt - 1U) >> 1U);
}

std::uint32_t galois_elt_for_conjugation(std::size_t degree)
{
    if (degree == 0 || degree > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max() / 2))
    {
        throw std::invalid_argument("GpuEvaluator::conjugate: invalid degree");
    }
    if ((degree & (degree - 1)) != 0)
    {
        throw std::invalid_argument("GpuEvaluator::conjugate: degree must be a power of two");
    }

    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(degree) << 1) - 1);
}

}  // namespace

GpuLooseBsgsPlan make_gpu_loose_bsgs_plan(
    const GpuMatrixPlain &matrix)
{
    if (matrix.plain_vec.empty())
    {
        throw std::invalid_argument(
            "make_gpu_loose_bsgs_plan: empty diagonal matrix");
    }
    if (matrix.n1 == 0)
    {
        throw std::invalid_argument(
            "make_gpu_loose_bsgs_plan: invalid BSGS n1");
    }
    if (matrix.log_slots >=
        static_cast<std::uint32_t>(std::numeric_limits<int>::digits))
    {
        throw std::invalid_argument(
            "make_gpu_loose_bsgs_plan: log_slots exceeds int range");
    }

    const auto [index, unused_rot_n1, baby_steps] =
        poseidon::bsgs_index(
            matrix.plain_vec,
            1 << matrix.log_slots,
            static_cast<int>(matrix.n1));
    (void)unused_rot_n1;

    GpuLooseBsgsPlan plan;
    plan.baby_steps = baby_steps;
    plan.groups.reserve(index.size());
    for (const auto &giant_entry : index)
    {
        GpuLooseBsgsGroup group;
        group.giant_step = giant_entry.first;
        group.terms.reserve(giant_entry.second.size());
        for (const int baby_step : giant_entry.second)
        {
            const int diagonal_index = group.giant_step + baby_step;
            if (matrix.plain_vec.find(diagonal_index) ==
                matrix.plain_vec.end())
            {
                throw std::invalid_argument(
                    "make_gpu_loose_bsgs_plan: missing plaintext diagonal");
            }
            group.terms.push_back(
                GpuLooseBsgsTerm{baby_step, diagonal_index});
        }
        if (!group.terms.empty())
        {
            plan.groups.push_back(std::move(group));
        }
    }
    if (plan.groups.empty())
    {
        throw std::invalid_argument(
            "make_gpu_loose_bsgs_plan: no nonzero diagonal contribution");
    }
    return plan;
}

std::vector<GpuEvalModBasisStep> make_gpu_eval_mod_basis_plan(
    GpuEvalModPolynomialBasis basis,
    const std::vector<std::uint32_t> &requested_degrees,
    std::uint32_t preferred_giant_stride)
{
    std::vector<GpuEvalModBasisStep> steps;
    std::map<std::uint32_t, bool> available;
    available.emplace(0, true);
    available.emplace(1, true);

    std::function<void(std::uint32_t)> schedule_degree =
        [&](std::uint32_t degree) {
            if (degree <= 1 || available.count(degree) != 0)
            {
                return;
            }
            if (degree == std::numeric_limits<std::uint32_t>::max())
            {
                throw std::invalid_argument(
                    "make_gpu_eval_mod_basis_plan: degree is too large");
            }

            const bool is_power_of_two = (degree & (degree - 1U)) == 0U;
            std::uint32_t left_degree = 0;
            std::uint32_t right_degree = 0;
            std::uint32_t correction_degree = 0;
            bool used_preferred_giant_pair = false;
            if (basis == GpuEvalModPolynomialBasis::Chebyshev &&
                preferred_giant_stride > 0 &&
                degree > preferred_giant_stride &&
                degree % preferred_giant_stride == 0)
            {
                /*
                 * The flat-b8 plan requests T8,T16,... in ascending order.
                 * Prefer the most balanced pair of already-live giant bases,
                 * e.g. T24=2*T8*T16-T8 and T48=2*T24^2-1.  Keeping this
                 * opt-in avoids changing the production recursive DAG.
                 */
                for (std::uint32_t candidate_left = preferred_giant_stride;
                     candidate_left <= degree / 2;
                     candidate_left += preferred_giant_stride)
                {
                    const auto candidate_right = degree - candidate_left;
                    const auto candidate_correction =
                        candidate_right - candidate_left;
                    if (available.count(candidate_left) != 0 &&
                        available.count(candidate_right) != 0 &&
                        (candidate_correction == 0 ||
                         available.count(candidate_correction) != 0))
                    {
                        left_degree = candidate_left;
                        right_degree = candidate_right;
                        correction_degree = candidate_correction;
                        used_preferred_giant_pair = true;
                    }
                }
            }
            if (used_preferred_giant_pair)
            {
                // All dependencies were checked above and are already live.
            }
            else if (is_power_of_two)
            {
                left_degree = degree >> 1U;
                right_degree = left_degree;
            }
            else
            {
                std::uint32_t power_of_two = 1;
                const std::uint32_t split_limit = (degree + 1U) >> 1U;
                while (power_of_two < split_limit)
                {
                    power_of_two <<= 1U;
                }
                left_degree = power_of_two - 1U;
                right_degree = degree + 1U - power_of_two;
                if (basis == GpuEvalModPolynomialBasis::Chebyshev)
                {
                    correction_degree = left_degree > right_degree
                        ? left_degree - right_degree
                        : right_degree - left_degree;
                }
            }

            schedule_degree(left_degree);
            schedule_degree(right_degree);
            if (basis == GpuEvalModPolynomialBasis::Chebyshev &&
                correction_degree != 0)
            {
                schedule_degree(correction_degree);
            }

            steps.push_back(GpuEvalModBasisStep{
                degree,
                left_degree,
                right_degree,
                correction_degree});
            available.emplace(degree, true);
        };

    std::vector<std::uint32_t> sorted_degrees = requested_degrees;
    std::sort(sorted_degrees.begin(), sorted_degrees.end());
    sorted_degrees.erase(
        std::unique(sorted_degrees.begin(), sorted_degrees.end()),
        sorted_degrees.end());
    for (const auto degree : sorted_degrees)
    {
        schedule_degree(degree);
    }
    return steps;
}

GpuEvaluator::GpuEvaluator(const GpuParameterData &params)
    : params_(params),
      elementwise_handler_(params),
      keyswitch_handler_(params),
      ntt_handler_(params),
      modswitch_handler_(params)
{}

void GpuEvaluator::add(
    const GpuCiphertextData &left_ciphertext,
    const GpuCiphertextData &right_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (left_ciphertext.empty() || right_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::add: empty ciphertext");
    }

    if (!(left_ciphertext.meta.parms_id == right_ciphertext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::add: parms_id mismatch");
    }

    if (left_ciphertext.meta.is_ntt_form != right_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::add: NTT form mismatch");
    }

    if (left_ciphertext.meta.degree != right_ciphertext.meta.degree ||
        left_ciphertext.meta.q_count != right_ciphertext.meta.q_count ||
        left_ciphertext.meta.p_count != right_ciphertext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::add: shape mismatch");
    }

    if (!same_scale(left_ciphertext.meta.scale, right_ciphertext.meta.scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::add: scale mismatch (left=2^" +
            std::to_string(std::log2(left_ciphertext.meta.scale)) +
            ", right=2^" +
            std::to_string(std::log2(right_ciphertext.meta.scale)) +
            ", q=" + std::to_string(left_ciphertext.meta.q_count) + ")");
    }

    if (left_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::add: p limbs are not supported by add kernel yet");
    }

    const std::size_t result_components =
        std::max(left_ciphertext.size(), right_ciphertext.size());

    const int device_id = left_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = left_ciphertext.polys_.at(0);

    if (left_ciphertext.meta.component_count != left_ciphertext.size() ||
        right_ciphertext.meta.component_count != right_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::add: component metadata mismatch");
    }

    if (!all_components_use_layout(left_ciphertext, reference_layout) ||
        !all_components_use_layout(right_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::add: shard layout mismatch");
    }

    prepare_ciphertext_destination(
        destination_ciphertext,
        &left_ciphertext,
        &right_ciphertext,
        left_ciphertext.meta,
        result_components,
        device_id,
        reference_layout);

    auto left_view = left_ciphertext.make_const_view();
    auto right_view = right_ciphertext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();

    const auto &level_info = params_.get_level(left_ciphertext.meta.parms_id);

    elementwise_handler_.add_ciphertext(
        destination_view,
        left_view,
        right_view,
        level_info);
}

void GpuEvaluator::sub(
    const GpuCiphertextData &left_ciphertext,
    const GpuCiphertextData &right_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (left_ciphertext.empty() || right_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::sub: empty ciphertext");
    }

    if (!(left_ciphertext.meta.parms_id == right_ciphertext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::sub: parms_id mismatch");
    }

    if (left_ciphertext.meta.is_ntt_form != right_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::sub: NTT form mismatch");
    }

    if (left_ciphertext.meta.degree != right_ciphertext.meta.degree ||
        left_ciphertext.meta.q_count != right_ciphertext.meta.q_count ||
        left_ciphertext.meta.p_count != right_ciphertext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::sub: shape mismatch");
    }

    if (!same_scale(left_ciphertext.meta.scale, right_ciphertext.meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::sub: scale mismatch");
    }

    if (left_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::sub: p limbs are not supported by sub kernel yet");
    }

    const std::size_t result_components =
        std::max(left_ciphertext.size(), right_ciphertext.size());

    const int device_id = left_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = left_ciphertext.polys_.at(0);

    if (left_ciphertext.meta.component_count != left_ciphertext.size() ||
        right_ciphertext.meta.component_count != right_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::sub: component metadata mismatch");
    }

    if (!all_components_use_layout(left_ciphertext, reference_layout) ||
        !all_components_use_layout(right_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::sub: shard layout mismatch");
    }

    prepare_ciphertext_destination(
        destination_ciphertext,
        &left_ciphertext,
        &right_ciphertext,
        left_ciphertext.meta,
        result_components,
        device_id,
        reference_layout);

    auto left_view = left_ciphertext.make_const_view();
    auto right_view = right_ciphertext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();

    const auto &level_info = params_.get_level(left_ciphertext.meta.parms_id);

    elementwise_handler_.sub_ciphertext(
        destination_view,
        left_view,
        right_view,
        level_info);
}

void GpuEvaluator::negate(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::negate: empty ciphertext");
    }

    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::negate: component metadata mismatch");
    }

    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::negate: p limbs are not supported by negate kernel yet");
    }

    const std::size_t result_components = source_ciphertext.size();

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::negate: shard layout mismatch");
    }

    prepare_ciphertext_destination(
        destination_ciphertext,
        &source_ciphertext,
        nullptr,
        source_ciphertext.meta,
        result_components,
        device_id,
        reference_layout);

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();

    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    elementwise_handler_.negate_ciphertext(
        destination_view,
        source_view,
        level_info);
}

void GpuEvaluator::validate_plain_input(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext) const
{
    if (source_ciphertext.empty() || source_plaintext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::plain arithmetic: empty input");
    }

    if (!(source_ciphertext.meta.parms_id == source_plaintext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::plain arithmetic: parms_id mismatch");
    }

    if (source_ciphertext.meta.is_ntt_form != source_plaintext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::plain arithmetic: NTT form mismatch");
    }

    // CKKS add_plain usually expects both ciphertext and plaintext in NTT form.
    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::plain arithmetic: CKKS input must be in NTT form");
    }

    if (source_ciphertext.meta.degree != source_plaintext.meta.degree ||
        source_ciphertext.meta.q_count != source_plaintext.meta.q_count ||
        source_ciphertext.meta.p_count != source_plaintext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::plain arithmetic: shape mismatch");
    }

    if (!same_scale(source_ciphertext.meta.scale, source_plaintext.meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::plain arithmetic: scale mismatch");
    }

    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::plain arithmetic: p limbs are not supported by add_plain kernel yet");
    }

    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::plain arithmetic: component metadata mismatch");
    }

    const auto &reference_layout = source_ciphertext.polys_.at(0);

    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::plain arithmetic: ciphertext shard layout mismatch");
    }

    if (!same_logical_shard_layout(reference_layout, source_plaintext.poly_))
    {
        throw std::invalid_argument("GpuEvaluator::plain arithmetic: plaintext shard layout mismatch");
    }

}

void GpuEvaluator::add_plain(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    validate_plain_input(source_ciphertext, source_plaintext);
    const std::size_t result_components = source_ciphertext.size();
    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    prepare_ciphertext_destination(
        destination_ciphertext,
        &source_ciphertext,
        nullptr,
        source_ciphertext.meta,
        result_components,
        device_id,
        reference_layout);

    auto ciphertext_view = source_ciphertext.make_const_view();
    auto plaintext_view = source_plaintext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();

    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    elementwise_handler_.add_plain_to_ciphertext(
        destination_view,
        ciphertext_view,
        plaintext_view,
        level_info);
}

void GpuEvaluator::sub_plain(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    validate_plain_input(source_ciphertext, source_plaintext);
    const std::size_t result_components = source_ciphertext.size();
    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    prepare_ciphertext_destination(
        destination_ciphertext,
        &source_ciphertext,
        nullptr,
        source_ciphertext.meta,
        result_components,
        device_id,
        reference_layout);

    auto ciphertext_view = source_ciphertext.make_const_view();
    auto plaintext_view = source_plaintext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();

    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    elementwise_handler_.sub_plain_from_ciphertext(
        destination_view,
        ciphertext_view,
        plaintext_view,
        level_info);
}

void GpuEvaluator::add_plain_inplace(
    GpuCiphertextData &ciphertext, const GpuPlaintextData &plaintext) const
{
    validate_plain_input(ciphertext, plaintext);
    auto view = ciphertext.make_view();
    auto plain = plaintext.make_const_view();
    elementwise_handler_.add_plain_to_ciphertext_inplace(
        view, plain, params_.get_level(ciphertext.meta.parms_id));
}

void GpuEvaluator::sub_plain_inplace(
    GpuCiphertextData &ciphertext, const GpuPlaintextData &plaintext) const
{
    validate_plain_input(ciphertext, plaintext);
    auto view = ciphertext.make_view();
    auto plain = plaintext.make_const_view();
    elementwise_handler_.sub_plain_from_ciphertext_inplace(
        view, plain, params_.get_level(ciphertext.meta.parms_id));
}

void GpuEvaluator::multiply_plain(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty() || source_plaintext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::multiply_plain: empty input");
    }

    if (!(source_ciphertext.meta.parms_id == source_plaintext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::multiply_plain: parms_id mismatch");
    }

    if (source_ciphertext.meta.is_ntt_form != source_plaintext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::multiply_plain: NTT form mismatch");
    }

    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::multiply_plain: CKKS input must be in NTT form");
    }

    if (source_ciphertext.meta.degree != source_plaintext.meta.degree ||
        source_ciphertext.meta.q_count != source_plaintext.meta.q_count ||
        source_ciphertext.meta.p_count != source_plaintext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::multiply_plain: shape mismatch");
    }

    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain: p limbs are not supported yet");
    }

    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain: component metadata mismatch");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain: ciphertext shard layout mismatch");
    }

    if (!same_logical_shard_layout(reference_layout, source_plaintext.poly_))
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain: plaintext shard layout mismatch");
    }

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            source_ciphertext.size(),
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);

    result.meta = source_ciphertext.meta;
    result.meta.component_count = source_ciphertext.size();
    result.meta.scale =
        source_ciphertext.meta.scale * source_plaintext.meta.scale;

    if (!(result.meta.scale > 0.0) || !std::isfinite(result.meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::multiply_plain: invalid result scale");
    }

    auto ciphertext_view = source_ciphertext.make_const_view();
    auto plaintext_view = source_plaintext.make_const_view();
    auto destination_view = result.make_view();

    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    elementwise_handler_.multiply_plain_with_ciphertext(
        destination_view,
        ciphertext_view,
        plaintext_view,
        level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::multiply_plain_q_prefix(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_plaintext.empty() || source_plaintext.fields_.size() != 1)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain_q_prefix: invalid plaintext storage");
    }
    if (!source_plaintext.meta.is_ntt_form ||
        source_plaintext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain_q_prefix: plaintext must be q-only NTT");
    }

    const auto &source_level =
        params_.get_level(source_ciphertext.meta.parms_id);
    const auto &target_level =
        params_.get_level(source_plaintext.meta.parms_id);
    validate_q_prefix_source(
        "GpuEvaluator::multiply_plain_q_prefix",
        source_ciphertext,
        source_level,
        target_level);
    if (source_plaintext.meta.degree != target_level.degree ||
        source_plaintext.meta.q_count != target_level.q_count ||
        source_plaintext.poly_.degree != target_level.degree ||
        source_plaintext.poly_.q_count != target_level.q_count ||
        source_plaintext.poly_.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain_q_prefix: plaintext level mismatch");
    }

    const auto prefix_layout =
        make_q_prefix_layout(source_ciphertext, target_level);
    if (!same_logical_shard_layout(prefix_layout, source_plaintext.poly_))
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain_q_prefix: plaintext shard mismatch");
    }

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            target_level.degree,
            target_level.q_count,
            source_ciphertext.size(),
            source_ciphertext.fields_.front().device_id,
            prefix_layout.shards,
            0);
    result.meta = source_ciphertext.meta;
    result.meta.parms_id = target_level.parms_id;
    result.meta.q_count = target_level.q_count;
    result.meta.p_count = 0;
    result.meta.component_count = source_ciphertext.size();
    result.meta.scale =
        source_ciphertext.meta.scale * source_plaintext.meta.scale;
    if (!(result.meta.scale > 0.0) || !std::isfinite(result.meta.scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain_q_prefix: invalid result scale");
    }

    auto ciphertext_view =
        make_q_prefix_view(source_ciphertext, target_level);
    auto plaintext_view = source_plaintext.make_const_view();
    auto destination_view = result.make_view();
    elementwise_handler_.multiply_plain_with_ciphertext(
        destination_view,
        ciphertext_view,
        plaintext_view,
        target_level);
    destination_ciphertext = std::move(result);
}

void GpuEvaluator::multiply_plain_sum_q_prefix(
    const std::vector<const GpuCiphertextData *> &ciphertexts,
    const std::vector<const GpuPlaintextData *> &plaintexts,
    GpuCiphertextData &destination_ciphertext,
    bool accumulate) const
{
    if (ciphertexts.empty() || ciphertexts.size() > 4 ||
        ciphertexts.size() != plaintexts.size())
        throw std::invalid_argument("multiply_plain_sum_q_prefix: expected 1..4 terms");
    for (std::size_t i = 0; i < ciphertexts.size(); ++i)
        if (ciphertexts[i] == nullptr || plaintexts[i] == nullptr)
            throw std::invalid_argument("multiply_plain_sum_q_prefix: null term");
    const auto &first = *ciphertexts.front();
    const auto &target = params_.get_level(plaintexts.front()->meta.parms_id);
    const double result_scale = first.meta.scale * plaintexts.front()->meta.scale;
    if (!(result_scale > 0) || !std::isfinite(result_scale))
        throw std::invalid_argument("multiply_plain_sum_q_prefix: invalid scale");
    std::vector<GpuConstCiphertextView> ciphertext_views;
    std::vector<GpuConstPlaintextView> plaintext_views;
    ciphertext_views.reserve(ciphertexts.size());
    plaintext_views.reserve(plaintexts.size());
    for (std::size_t i = 0; i < ciphertexts.size(); ++i)
    {
        const auto &ct = *ciphertexts[i];
        const auto &pt = *plaintexts[i];
        validate_q_prefix_source("multiply_plain_sum_q_prefix", ct,
            params_.get_level(ct.meta.parms_id), target);
        if (ct.size() != 2 || pt.empty() || pt.fields_.size() != 1 ||
            !pt.meta.is_ntt_form || pt.meta.parms_id != target.parms_id ||
            pt.meta.degree != target.degree || pt.meta.q_count != target.q_count ||
            pt.meta.p_count != 0 || pt.poly_.degree != target.degree ||
            pt.poly_.q_count != target.q_count || pt.poly_.p_count != 0 ||
            ct.fields_.front().device_id != first.fields_.front().device_id ||
            !same_logical_shard_layout(make_q_prefix_layout(ct, target), pt.poly_))
            throw std::invalid_argument("multiply_plain_sum_q_prefix: term shape mismatch");
        const double product_scale = ct.meta.scale * pt.meta.scale;
        if (!(ct.meta.scale > 0) || !(pt.meta.scale > 0) ||
            !(product_scale > 0) || !std::isfinite(product_scale) ||
            std::abs(std::log2(product_scale / result_scale)) > 1e-8)
            throw std::invalid_argument("multiply_plain_sum_q_prefix: product scales differ");
        ciphertext_views.push_back(make_q_prefix_view(ct, target));
        plaintext_views.push_back(pt.make_const_view());
    }
    const auto layout = make_q_prefix_layout(first, target);
    if (accumulate)
    {
        const auto &dst = destination_ciphertext;
        if (dst.empty())
            throw std::invalid_argument("multiply_plain_sum_q_prefix: empty destination");
        validate_q_prefix_source("multiply_plain_sum_q_prefix destination", dst,
            params_.get_level(dst.meta.parms_id), target);
        if (dst.size() != 2 || dst.meta.parms_id != target.parms_id ||
            dst.fields_.front().device_id != first.fields_.front().device_id ||
            !same_logical_shard_layout(dst.polys_.front(), layout) ||
            !(dst.meta.scale > 0) || !std::isfinite(dst.meta.scale) ||
            std::abs(std::log2(dst.meta.scale / result_scale)) > 1e-8)
            throw std::invalid_argument("multiply_plain_sum_q_prefix: destination mismatch");
        auto destination_view = destination_ciphertext.make_view();
        elementwise_handler_.multiply_plain_sum(
            destination_view, ciphertext_views, plaintext_views, target, true);
        return;
    }
    auto result = GpuCiphertextData::allocate_single_device_sharded(
        target.degree, target.q_count, 2, first.fields_.front().device_id,
        layout.shards, 0);
    result.meta = first.meta;
    result.meta.parms_id = target.parms_id;
    result.meta.q_count = target.q_count;
    result.meta.p_count = 0;
    result.meta.component_count = 2;
    result.meta.scale = result_scale;
    auto destination_view = result.make_view();
    elementwise_handler_.multiply_plain_sum(
        destination_view, ciphertext_views, plaintext_views, target);
    destination_ciphertext = std::move(result);
}

void GpuEvaluator::double_sub_plain_q_prefix(
    const GpuCiphertextData &source, const GpuPlaintextData &plaintext,
    GpuCiphertextData &destination, const GpuCiphertextData *correction) const
{
    constexpr const char *name = "double_sub_plain_q_prefix";
    if (source.empty() || plaintext.empty() || (correction && correction->empty()))
        throw std::invalid_argument("double_sub_plain_q_prefix: empty input");
    const auto &level = params_.get_level(plaintext.meta.parms_id);
    validate_q_prefix_source(name, source, params_.get_level(source.meta.parms_id), level);
    if (source.size() != 2 || source.meta.parms_id != level.parms_id ||
        plaintext.fields_.size() != 1 || !plaintext.meta.is_ntt_form ||
        plaintext.meta.degree != level.degree || plaintext.meta.q_count != level.q_count ||
        plaintext.meta.p_count != 0 || plaintext.poly_.degree != level.degree ||
        plaintext.poly_.q_count != level.q_count || plaintext.poly_.p_count != 0 ||
        !same_logical_shard_layout(source.polys_[0], plaintext.poly_))
        throw std::invalid_argument("double_sub_plain_q_prefix: source/plaintext mismatch");
    const auto layout = make_q_prefix_layout(source, level);
    GpuConstCiphertextView correction_view;
    double subtrahend_scale = plaintext.meta.scale;
    if (correction)
    {
        validate_q_prefix_source(name, *correction,
            params_.get_level(correction->meta.parms_id), level);
        if (correction->size() != 2 ||
            correction->fields_[0].device_id != source.fields_[0].device_id ||
            !same_logical_shard_layout(make_q_prefix_layout(*correction, level), layout) ||
            !(correction->meta.scale > 0) || !std::isfinite(correction->meta.scale))
            throw std::invalid_argument("double_sub_plain_q_prefix: correction mismatch");
        subtrahend_scale *= correction->meta.scale;
        correction_view = make_q_prefix_view(*correction, level);
    }
    if (!(source.meta.scale > 0) || !std::isfinite(source.meta.scale) ||
        !(plaintext.meta.scale > 0) || !std::isfinite(plaintext.meta.scale) ||
        !(subtrahend_scale > 0) || !std::isfinite(subtrahend_scale) ||
        std::abs(std::log2(subtrahend_scale / source.meta.scale)) > 1e-8)
        throw std::invalid_argument("double_sub_plain_q_prefix: scales differ");
    auto result = GpuCiphertextData::allocate_single_device_sharded(
        level.degree, level.q_count, 2, source.fields_[0].device_id, layout.shards, 0);
    result.meta = source.meta;
    auto view = result.make_view();
    elementwise_handler_.double_sub_plain(view, source.make_const_view(),
        plaintext.make_const_view(), correction ? &correction_view : nullptr, level);
    destination = std::move(result);
}

void GpuEvaluator::multiply_plain_accumulate(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty() || source_plaintext.empty() ||
        destination_ciphertext.empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain_accumulate: empty input");
    }
    if (!(source_ciphertext.meta.parms_id ==
          source_plaintext.meta.parms_id) ||
        !(source_ciphertext.meta.parms_id ==
          destination_ciphertext.meta.parms_id) ||
        !source_ciphertext.meta.is_ntt_form ||
        !source_plaintext.meta.is_ntt_form ||
        !destination_ciphertext.meta.is_ntt_form ||
        source_ciphertext.size() != 2 ||
        destination_ciphertext.size() != 2)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain_accumulate: incompatible input");
    }
    const double product_scale =
        source_ciphertext.meta.scale * source_plaintext.meta.scale;
    if (!same_scale(product_scale, destination_ciphertext.meta.scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain_accumulate: scale mismatch");
    }
    if (!same_logical_shard_layout(
            source_ciphertext.polys_.front(),
            source_plaintext.poly_) ||
        !same_logical_shard_layout(
            source_ciphertext.polys_.front(),
            destination_ciphertext.polys_.front()) ||
        !all_components_use_layout(
            source_ciphertext,
            source_ciphertext.polys_.front()) ||
        !all_components_use_layout(
            destination_ciphertext,
            destination_ciphertext.polys_.front()))
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain_accumulate: shard mismatch");
    }

    auto destination_view = destination_ciphertext.make_view();
    auto ciphertext_view = source_ciphertext.make_const_view();
    auto plaintext_view = source_plaintext.make_const_view();
    const auto &level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    elementwise_handler_.multiply_plain_accumulate_with_ciphertext(
        destination_view,
        ciphertext_view,
        plaintext_view,
        level_info);
}

void GpuEvaluator::ntt_fwd(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::ntt_fwd",
        source_ciphertext,
        false);

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            source_ciphertext.size(),
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);

    result.meta = source_ciphertext.meta;
    result.meta.component_count = source_ciphertext.size();
    result.meta.is_ntt_form = true;

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();

    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    ntt_handler_.forward_ciphertext(
        destination_view,
        source_view,
        level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::ntt_inv(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::ntt_inv",
        source_ciphertext,
        true);

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            source_ciphertext.size(),
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);

    result.meta = source_ciphertext.meta;
    result.meta.component_count = source_ciphertext.size();
    result.meta.is_ntt_form = false;

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();

    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    ntt_handler_.inverse_ciphertext(
        destination_view,
        source_view,
        level_info);

    destination_ciphertext = std::move(result);
}

/**
 * @brief 用户端顶层算子，进行必要的输入检查，构建结果临时缓存，调用multiply_ciphertext
 */
void GpuEvaluator::multiply(
    const GpuCiphertextData &left_ciphertext,
    const GpuCiphertextData &right_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (left_ciphertext.empty() || right_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::multiply: empty ciphertext");
    }
    if (left_ciphertext.fields_.empty() || right_ciphertext.fields_.empty())
    {
        throw std::invalid_argument("GpuEvaluator::multiply: empty ciphertext storage");
    }

    if (!(left_ciphertext.meta.parms_id == right_ciphertext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::multiply: parms_id mismatch");
    }
    if (!left_ciphertext.meta.is_ntt_form ||
        !right_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::multiply: CKKS inputs must be in NTT form");
    }
    if (left_ciphertext.meta.degree != right_ciphertext.meta.degree ||
        left_ciphertext.meta.q_count != right_ciphertext.meta.q_count ||
        left_ciphertext.meta.p_count != right_ciphertext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::multiply: shape mismatch");
    }
    if (left_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuEvaluator::multiply: p limbs are not supported yet");
    }
    if (left_ciphertext.meta.component_count != left_ciphertext.size() ||
        right_ciphertext.meta.component_count != right_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::multiply: component metadata mismatch");
    }

    const int device_id = left_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = left_ciphertext.polys_.at(0);

    if (!all_components_use_layout(left_ciphertext, reference_layout) ||
        !all_components_use_layout(right_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::multiply: shard layout mismatch");
    }

    const std::size_t result_components =
        left_ciphertext.size() + right_ciphertext.size() - 1;

    GpuCiphertextMeta result_meta = left_ciphertext.meta;
    result_meta.component_count = result_components;
    result_meta.is_ntt_form = true;
    result_meta.scale =
        left_ciphertext.meta.scale * right_ciphertext.meta.scale;

    if (!(result_meta.scale > 0.0) || !std::isfinite(result_meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::multiply: invalid result scale");
    }

    const bool aliases_input =
        &destination_ciphertext == &left_ciphertext ||
        &destination_ciphertext == &right_ciphertext;
    GpuCiphertextData local_result;
    GpuCiphertextData *result = &destination_ciphertext;
    if (aliases_input)
    {
        local_result = GpuCiphertextData::allocate_single_device_sharded(
            left_ciphertext.meta.degree,
            left_ciphertext.meta.q_count,
            result_components,
            device_id,
            reference_layout.shards,
            left_ciphertext.meta.p_count);
        local_result.meta = result_meta;
        result = &local_result;
    }
    else
    {
        prepare_ciphertext_destination(
            destination_ciphertext,
            nullptr,
            nullptr,
            result_meta,
            result_components,
            device_id,
            reference_layout);
    }

    auto left_view = left_ciphertext.make_const_view();
    auto right_view = right_ciphertext.make_const_view();
    auto destination_view = result->make_view();

    const auto &level_info = params_.get_level(left_ciphertext.meta.parms_id);

    elementwise_handler_.multiply_ciphertext(
        destination_view,
        left_view,
        right_view,
        level_info);

    if (aliases_input)
    {
        destination_ciphertext = std::move(local_result);
    }
}

void GpuEvaluator::multiply_q_prefix(
    const GpuCiphertextData &left_ciphertext,
    const GpuCiphertextData &right_ciphertext,
    GpuCiphertextData &destination_ciphertext,
    parms_id_type target_parms_id) const
{
    const auto &left_level =
        params_.get_level(left_ciphertext.meta.parms_id);
    const auto &right_level =
        params_.get_level(right_ciphertext.meta.parms_id);
    const auto &target_level = params_.get_level(target_parms_id);
    validate_q_prefix_source(
        "GpuEvaluator::multiply_q_prefix left",
        left_ciphertext,
        left_level,
        target_level);
    validate_q_prefix_source(
        "GpuEvaluator::multiply_q_prefix right",
        right_ciphertext,
        right_level,
        target_level);
    if (left_ciphertext.meta.degree != right_ciphertext.meta.degree ||
        left_ciphertext.fields_.front().device_id !=
            right_ciphertext.fields_.front().device_id)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_q_prefix: incompatible inputs");
    }

    const auto result_components =
        left_ciphertext.size() + right_ciphertext.size() - 1;
    const auto prefix_layout =
        make_q_prefix_layout(left_ciphertext, target_level);
    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            target_level.degree,
            target_level.q_count,
            result_components,
            left_ciphertext.fields_.front().device_id,
            prefix_layout.shards,
            0);
    result.meta = left_ciphertext.meta;
    result.meta.parms_id = target_level.parms_id;
    result.meta.q_count = target_level.q_count;
    result.meta.p_count = 0;
    result.meta.component_count = result_components;
    result.meta.is_ntt_form = true;
    result.meta.scale =
        left_ciphertext.meta.scale * right_ciphertext.meta.scale;
    if (!(result.meta.scale > 0.0) || !std::isfinite(result.meta.scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_q_prefix: invalid result scale");
    }

    auto left_view = make_q_prefix_view(left_ciphertext, target_level);
    auto right_view = make_q_prefix_view(right_ciphertext, target_level);
    auto destination_view = result.make_view();
    elementwise_handler_.multiply_ciphertext(
        destination_view,
        left_view,
        right_view,
        target_level);
    destination_ciphertext = std::move(result);
}

void GpuEvaluator::add_sub_q_prefix(
    const GpuCiphertextData &left, const GpuCiphertextData &right,
    GpuCiphertextData &destination, parms_id_type target_parms_id,
    bool subtract) const
{
    const auto &target = params_.get_level(target_parms_id);
    validate_q_prefix_source("add_sub_q_prefix left", left,
        params_.get_level(left.meta.parms_id), target);
    validate_q_prefix_source("add_sub_q_prefix right", right,
        params_.get_level(right.meta.parms_id), target);
    if (left.meta.degree != right.meta.degree ||
        left.fields_.front().device_id != right.fields_.front().device_id ||
        !same_scale(left.meta.scale, right.meta.scale) ||
        !(left.meta.scale > 0.0) || !std::isfinite(left.meta.scale) ||
        !(right.meta.scale > 0.0) || !std::isfinite(right.meta.scale))
        throw std::invalid_argument("add_sub_q_prefix: incompatible shape/device/scale");
    const auto layout = make_q_prefix_layout(left, target);
    const auto right_layout = make_q_prefix_layout(right, target);
    if (!same_logical_shard_layout(layout, right_layout))
        throw std::invalid_argument("add_sub_q_prefix: incompatible shard layouts");
    const auto components = std::max(left.size(), right.size());
    auto result = GpuCiphertextData::allocate_single_device_sharded(
        target.degree, target.q_count, components,
        left.fields_.front().device_id, layout.shards, 0);
    result.meta = left.meta;
    result.meta.parms_id = target.parms_id;
    result.meta.q_count = target.q_count;
    result.meta.p_count = 0;
    result.meta.component_count = components;
    auto left_view = make_q_prefix_view(left, target);
    auto right_view = make_q_prefix_view(right, target);
    auto result_view = result.make_view();
    if (subtract)elementwise_handler_.sub_ciphertext(result_view, left_view, right_view, target);
    else elementwise_handler_.add_ciphertext(result_view, left_view, right_view, target);
    destination = std::move(result);
}

void GpuEvaluator::square(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::square: empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::square: empty ciphertext storage");
    }
    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument(
            "GpuEvaluator::square: CKKS input must be in NTT form");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::square: p limbs are not supported yet");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument(
            "GpuEvaluator::square: component metadata mismatch");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument(
            "GpuEvaluator::square: shard layout mismatch");
    }

    const std::size_t result_components =
        source_ciphertext.size() * 2 - 1;
    GpuCiphertextMeta result_meta = source_ciphertext.meta;
    result_meta.component_count = result_components;
    result_meta.is_ntt_form = true;
    result_meta.scale =
        source_ciphertext.meta.scale * source_ciphertext.meta.scale;
    if (!(result_meta.scale > 0.0) || !std::isfinite(result_meta.scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::square: invalid result scale");
    }

    const bool aliases_input = &destination_ciphertext == &source_ciphertext;
    GpuCiphertextData local_result;
    GpuCiphertextData *result = &destination_ciphertext;
    if (aliases_input)
    {
        local_result = GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            result_components,
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);
        local_result.meta = result_meta;
        result = &local_result;
    }
    else
    {
        prepare_ciphertext_destination(
            destination_ciphertext,
            nullptr,
            nullptr,
            result_meta,
            result_components,
            device_id,
            reference_layout);
    }

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result->make_view();
    const auto &level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    elementwise_handler_.square_ciphertext(
        destination_view,
        source_view,
        level_info);
    if (aliases_input)
    {
        destination_ciphertext = std::move(local_result);
    }
}

void GpuEvaluator::rescale(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::rescale: empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument("GpuEvaluator::rescale: empty ciphertext storage");
    }
    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::rescale: CKKS input must be in NTT form");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuEvaluator::rescale: p limbs are not supported yet");
    }
    if (source_ciphertext.meta.q_count < 2)
    {
        throw std::invalid_argument("GpuEvaluator::rescale: cannot drop the last q modulus");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::rescale: component metadata mismatch");
    }

    const auto &source_level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    const auto &destination_level_info =
        params_.get_next_level(source_ciphertext.meta.parms_id);

    if (source_level_info.q_count != source_ciphertext.meta.q_count ||
        destination_level_info.q_count + 1 != source_level_info.q_count)
    {
        throw std::invalid_argument("GpuEvaluator::rescale: level q_count mismatch");
    }
    if (source_level_info.shards.empty() ||
        source_level_info.shards.front().q_last == 0)
    {
        throw std::invalid_argument("GpuEvaluator::rescale: missing source q_last parameter");
    }

    const std::size_t destination_q_count =
        source_ciphertext.meta.q_count - 1;
    const int device_id = source_ciphertext.fields_.at(0).device_id;

    GpuPolyShard destination_shard;
    destination_shard.field_index = 0;
    destination_shard.field_offset = 0;
    destination_shard.limb_begin = 0;
    destination_shard.limb_count = destination_q_count;
    destination_shard.coeff_begin = 0;
    destination_shard.coeff_count = source_ciphertext.meta.degree;

    GpuCiphertextMeta result_meta = source_ciphertext.meta;
    result_meta.parms_id = destination_level_info.parms_id;
    result_meta.q_count = destination_q_count;
    result_meta.p_count = 0;
    result_meta.component_count = source_ciphertext.size();
    result_meta.is_ntt_form = true;
    result_meta.scale =
        source_ciphertext.meta.scale /
        static_cast<double>(source_level_info.shards.front().q_last);

    if (!(result_meta.scale > 0.0) || !std::isfinite(result_meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::rescale: invalid result scale");
    }

    GpuRNSPoly reference_layout;
    reference_layout.degree = source_ciphertext.meta.degree;
    reference_layout.q_count = destination_q_count;
    reference_layout.p_count = 0;
    reference_layout.shards.push_back(destination_shard);

    const bool aliases_input = &destination_ciphertext == &source_ciphertext;
    GpuCiphertextData local_result;
    GpuCiphertextData *result = &destination_ciphertext;
    if (aliases_input)
    {
        local_result = GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            destination_q_count,
            source_ciphertext.size(),
            device_id,
            std::vector<GpuPolyShard>{destination_shard},
            0);
        local_result.meta = result_meta;
        result = &local_result;
    }
    else
    {
        prepare_ciphertext_destination(
            destination_ciphertext,
            nullptr,
            nullptr,
            result_meta,
            source_ciphertext.size(),
            device_id,
            reference_layout);
    }

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result->make_view();

    modswitch_handler_.rescale_ciphertext(
        destination_view,
        source_view,
        source_level_info,
        destination_level_info);

    if (aliases_input)
    {
        destination_ciphertext = std::move(local_result);
    }
}

void GpuEvaluator::rescale_x2(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty() ||
        source_ciphertext.fields_.empty() ||
        !source_ciphertext.meta.is_ntt_form ||
        source_ciphertext.meta.p_count != 0 ||
        source_ciphertext.meta.q_count < 3 ||
        source_ciphertext.meta.component_count != 2 ||
        source_ciphertext.size() != 2)
    {
        throw std::invalid_argument(
            "GpuEvaluator::rescale_x2: unsupported ciphertext shape");
    }

    const auto &source_level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    const auto &intermediate_level_info =
        params_.get_next_level(source_ciphertext.meta.parms_id);
    const auto &destination_level_info =
        params_.get_next_level(intermediate_level_info.parms_id);
    if (source_level_info.q_count != source_ciphertext.meta.q_count ||
        destination_level_info.q_count + 2 != source_level_info.q_count ||
        source_level_info.shards.empty() ||
        source_level_info.shards.front().q_last_two_product == 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::rescale_x2: level/constants mismatch");
    }

    const std::size_t destination_q_count =
        source_ciphertext.meta.q_count - 2;
    const int device_id = source_ciphertext.fields_.front().device_id;
    GpuPolyShard destination_shard;
    destination_shard.field_index = 0;
    destination_shard.field_offset = 0;
    destination_shard.limb_begin = 0;
    destination_shard.limb_count = destination_q_count;
    destination_shard.coeff_begin = 0;
    destination_shard.coeff_count = source_ciphertext.meta.degree;

    GpuCiphertextMeta result_meta = source_ciphertext.meta;
    result_meta.parms_id = destination_level_info.parms_id;
    result_meta.q_count = destination_q_count;
    result_meta.p_count = 0;
    result_meta.component_count = 2;
    result_meta.is_ntt_form = true;
    result_meta.scale =
        source_ciphertext.meta.scale /
        static_cast<double>(
            source_level_info.shards.front().q_last_two_product);
    if (!(result_meta.scale > 0.0) || !std::isfinite(result_meta.scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::rescale_x2: invalid result scale");
    }

    GpuRNSPoly reference_layout;
    reference_layout.degree = source_ciphertext.meta.degree;
    reference_layout.q_count = destination_q_count;
    reference_layout.p_count = 0;
    reference_layout.shards.push_back(destination_shard);

    const bool aliases_input = &destination_ciphertext == &source_ciphertext;
    GpuCiphertextData local_result;
    GpuCiphertextData *result = &destination_ciphertext;
    if (aliases_input)
    {
        local_result = GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            destination_q_count,
            2,
            device_id,
            std::vector<GpuPolyShard>{destination_shard},
            0);
        local_result.meta = result_meta;
        result = &local_result;
    }
    else
    {
        prepare_ciphertext_destination(
            destination_ciphertext,
            nullptr,
            nullptr,
            result_meta,
            2,
            device_id,
            reference_layout);
    }

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result->make_view();
    modswitch_handler_.rescale_ciphertext_x2(
        destination_view,
        source_view,
        source_level_info,
        destination_level_info);
    if (aliases_input)
    {
        destination_ciphertext = std::move(local_result);
    }
}

void GpuEvaluator::rescale_many(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext,
    std::uint32_t rescale_count) const
{
    if (rescale_count == 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::rescale_many: rescale_count must be positive");
    }
    if (source_ciphertext.meta.q_count <= rescale_count)
    {
        throw std::invalid_argument(
            "GpuEvaluator::rescale_many: modulus chain is too short");
    }
    if (rescale_count == 1)
    {
        rescale(source_ciphertext, destination_ciphertext);
        return;
    }
    const bool can_use_x2 =
        source_ciphertext.meta.component_count == 2 &&
        source_ciphertext.size() == 2 &&
        use_rescale_x2();
    if (rescale_count == 2 && can_use_x2)
    {
        rescale_x2(source_ciphertext, destination_ciphertext);
        return;
    }

    if (source_ciphertext.empty() || source_ciphertext.fields_.empty() ||
        !source_ciphertext.meta.is_ntt_form ||
        source_ciphertext.meta.p_count != 0 ||
        source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument(
            "GpuEvaluator::rescale_many: unsupported ciphertext shape");
    }

    const auto &source_level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    const GpuLevelInfo *destination_level_info = &source_level_info;
    double result_scale = source_ciphertext.meta.scale;
    for (std::uint32_t index = 0; index < rescale_count; ++index)
    {
        if (destination_level_info->shards.empty() ||
            destination_level_info->shards.front().q_last == 0)
        {
            throw std::invalid_argument(
                "GpuEvaluator::rescale_many: missing q_last parameter");
        }
        result_scale /= static_cast<double>(
            destination_level_info->shards.front().q_last);
        destination_level_info = &params_.get_next_level(
            destination_level_info->parms_id);
    }
    const std::size_t destination_q_count =
        source_ciphertext.meta.q_count - rescale_count;
    if (source_level_info.q_count != source_ciphertext.meta.q_count ||
        destination_level_info->q_count != destination_q_count ||
        !(result_scale > 0.0) || !std::isfinite(result_scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::rescale_many: level or scale mismatch");
    }

    const int device_id = source_ciphertext.fields_.front().device_id;
    GpuPolyShard destination_shard;
    destination_shard.field_index = 0;
    destination_shard.field_offset = 0;
    destination_shard.limb_begin = 0;
    destination_shard.limb_count = destination_q_count;
    destination_shard.coeff_begin = 0;
    destination_shard.coeff_count = source_ciphertext.meta.degree;

    GpuCiphertextMeta result_meta = source_ciphertext.meta;
    result_meta.parms_id = destination_level_info->parms_id;
    result_meta.q_count = destination_q_count;
    result_meta.p_count = 0;
    result_meta.component_count = source_ciphertext.size();
    result_meta.is_ntt_form = true;
    result_meta.scale = result_scale;

    GpuRNSPoly reference_layout;
    reference_layout.degree = source_ciphertext.meta.degree;
    reference_layout.q_count = destination_q_count;
    reference_layout.p_count = 0;
    reference_layout.shards.push_back(destination_shard);

    const bool aliases_input = &destination_ciphertext == &source_ciphertext;
    GpuCiphertextData local_result;
    GpuCiphertextData *result = &destination_ciphertext;
    if (aliases_input)
    {
        local_result = GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            destination_q_count,
            source_ciphertext.size(),
            device_id,
            std::vector<GpuPolyShard>{destination_shard},
            0);
        local_result.meta = result_meta;
        result = &local_result;
    }
    else
    {
        prepare_ciphertext_destination(
            destination_ciphertext,
            nullptr,
            nullptr,
            result_meta,
            source_ciphertext.size(),
            device_id,
            reference_layout);
    }

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result->make_view();
    modswitch_handler_.rescale_ciphertext_many(
        destination_view,
        source_view,
        source_level_info,
        *destination_level_info,
        rescale_count);
    if (aliases_input)
    {
        destination_ciphertext = std::move(local_result);
    }
}

void GpuEvaluator::rescale_dynamic(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext,
    double min_scale) const
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::rescale_dynamic: empty ciphertext");
    }
    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument(
            "GpuEvaluator::rescale_dynamic: CKKS input must be in NTT form");
    }
    if (!(min_scale > 0.0) || !std::isfinite(min_scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::rescale_dynamic: invalid minimum scale");
    }
    if (!(source_ciphertext.meta.scale > 0.0) ||
        !std::isfinite(source_ciphertext.meta.scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::rescale_dynamic: invalid input scale");
    }

    std::vector<std::uint64_t> active_moduli(
        source_ciphertext.meta.q_count,
        0);
    parms_id_type current_parms_id = source_ciphertext.meta.parms_id;
    while (true)
    {
        const auto &level_info = params_.get_level(current_parms_id);
        if (level_info.q_count == 0 ||
            level_info.q_count > active_moduli.size() ||
            level_info.shards.empty() ||
            level_info.shards.front().q_last == 0)
        {
            throw std::invalid_argument(
                "GpuEvaluator::rescale_dynamic: incomplete active modulus chain");
        }
        active_moduli[level_info.q_count - 1] =
            level_info.shards.front().q_last;
        if (level_info.q_count == 1)
        {
            break;
        }
        current_parms_id = params_.get_next_level(current_parms_id).parms_id;
    }

    const auto plan = plan_gpu_dynamic_rescale(
        source_ciphertext.meta.scale,
        min_scale,
        active_moduli);

    if (plan.rescale_count == 0)
    {
        if (&source_ciphertext == &destination_ciphertext)
        {
            return;
        }
        copy_ciphertext_data(
            source_ciphertext,
            destination_ciphertext,
            "GpuEvaluator::rescale_dynamic no-op copy");
        return;
    }

    rescale_many(
        source_ciphertext,
        destination_ciphertext,
        plan.rescale_count);
    destination_ciphertext.meta.scale = plan.output_scale;
}

void GpuEvaluator::drop_modulus(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext,
    parms_id_type target_parms_id) const
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: empty ciphertext storage");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: p limbs are not supported yet");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: component metadata mismatch");
    }
    const auto &reference_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: shard layout mismatch");
    }
    if (reference_layout.shards.size() != 1 ||
        reference_layout.shards.front().limb_begin != 0 ||
        reference_layout.shards.front().limb_count != source_ciphertext.meta.q_count ||
        reference_layout.shards.front().coeff_begin != 0 ||
        reference_layout.shards.front().coeff_count != source_ciphertext.meta.degree)
    {
        throw std::invalid_argument(
            "GpuEvaluator::drop_modulus: first implementation requires one full q shard");
    }

    const auto &source_level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    const auto &destination_level_info =
        params_.get_level(target_parms_id);

    if (source_level_info.q_count != source_ciphertext.meta.q_count ||
        destination_level_info.p_count != 0 ||
        destination_level_info.q_count == 0 ||
        destination_level_info.q_count > source_level_info.q_count)
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: level q_count mismatch");
    }
    if (destination_level_info.degree != source_ciphertext.meta.degree)
    {
        throw std::invalid_argument("GpuEvaluator::drop_modulus: degree mismatch");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    GpuPolyShard destination_shard;
    destination_shard.field_index = 0;
    destination_shard.field_offset = 0;
    destination_shard.limb_begin = 0;
    destination_shard.limb_count = destination_level_info.q_count;
    destination_shard.coeff_begin = 0;
    destination_shard.coeff_count = source_ciphertext.meta.degree;

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            destination_level_info.q_count,
            source_ciphertext.size(),
            device_id,
            std::vector<GpuPolyShard>{ destination_shard },
            0);

    result.meta = source_ciphertext.meta;
    result.meta.parms_id = destination_level_info.parms_id;
    result.meta.q_count = destination_level_info.q_count;
    result.meta.p_count = 0;
    result.meta.component_count = source_ciphertext.size();

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();
    modswitch_handler_.drop_modulus_ciphertext(
        destination_view,
        source_view,
        source_level_info,
        destination_level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::drop_modulus_inplace(
    GpuCiphertextData &ciphertext,
    parms_id_type target_parms_id) const
{
    if (ciphertext.empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::drop_modulus_inplace: empty ciphertext");
    }
    if (ciphertext.fields_.empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::drop_modulus_inplace: empty ciphertext storage");
    }
    if (ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::drop_modulus_inplace: p limbs are not supported");
    }
    if (ciphertext.meta.component_count != ciphertext.size())
    {
        throw std::invalid_argument(
            "GpuEvaluator::drop_modulus_inplace: component metadata mismatch");
    }

    const auto &source_level_info =
        params_.get_level(ciphertext.meta.parms_id);
    const auto &destination_level_info =
        params_.get_level(target_parms_id);
    if (source_level_info.q_count != ciphertext.meta.q_count ||
        destination_level_info.p_count != 0 ||
        destination_level_info.q_count == 0 ||
        destination_level_info.q_count > source_level_info.q_count)
    {
        throw std::invalid_argument(
            "GpuEvaluator::drop_modulus_inplace: level q_count mismatch");
    }
    if (destination_level_info.degree != ciphertext.meta.degree)
    {
        throw std::invalid_argument(
            "GpuEvaluator::drop_modulus_inplace: degree mismatch");
    }

    for (const auto &poly : ciphertext.polys_)
    {
        if (poly.degree != ciphertext.meta.degree ||
            poly.q_count != ciphertext.meta.q_count ||
            poly.p_count != 0 ||
            poly.shards.size() != 1)
        {
            throw std::invalid_argument(
                "GpuEvaluator::drop_modulus_inplace: unsupported component layout");
        }
        const auto &shard = poly.shards.front();
        if (shard.limb_begin != 0 ||
            shard.limb_count != ciphertext.meta.q_count ||
            shard.coeff_begin != 0 ||
            shard.coeff_count != ciphertext.meta.degree)
        {
            throw std::invalid_argument(
                "GpuEvaluator::drop_modulus_inplace: requires one full q shard");
        }
    }

    ciphertext.meta.parms_id = destination_level_info.parms_id;
    ciphertext.meta.q_count = destination_level_info.q_count;
    ciphertext.meta.p_count = 0;
    for (auto &poly : ciphertext.polys_)
    {
        poly.q_count = destination_level_info.q_count;
        poly.p_count = 0;
        poly.shards.front().limb_count = destination_level_info.q_count;
    }
}

namespace
{

bool pair_has_single_full_shard(const GpuCiphertextData &ct)
{
    if (ct.empty() || ct.fields_.empty() || ct.meta.p_count != 0 ||
        ct.meta.component_count != ct.size() ||
        !all_components_use_layout(ct, ct.polys_.at(0)) ||
        ct.polys_.at(0).shards.size() != 1)
    {
        return false;
    }
    const auto &s = ct.polys_.at(0).shards.front();
    return s.limb_begin == 0 && s.limb_count == ct.meta.q_count &&
           s.coeff_begin == 0 && s.coeff_count == ct.meta.degree;
}

bool plain_has_single_full_shard(const GpuPlaintextData &pt)
{
    if (pt.empty() || pt.fields_.empty() || pt.meta.p_count != 0 || pt.poly_.shards.size() != 1)
    {
        return false;
    }
    const auto &s = pt.poly_.shards.front();
    return s.limb_begin == 0 && s.limb_count == pt.meta.q_count &&
           s.coeff_begin == 0 && s.coeff_count == pt.meta.degree;
}

GpuConstCiphertextView make_q_prefix_const_view(
    const GpuCiphertextData &ct,
    const GpuLevelInfo &dst_level)
{
    GpuConstCiphertextView view = ct.make_const_view();
    view.meta.parms_id = dst_level.parms_id;
    view.meta.q_count = dst_level.q_count;
    for (auto &poly : view.polys)
    {
        poly.shards.front().limb_count = dst_level.q_count;
    }
    return view;
}

GpuCiphertextView make_q_prefix_mutable_view(
    GpuCiphertextData &ct,
    const GpuCiphertextMeta &dst_meta)
{
    GpuCiphertextView view = ct.make_view();
    view.meta = dst_meta;
    for (auto &poly : view.polys)
    {
        poly.shards.front().limb_count = dst_meta.q_count;
    }
    return view;
}

GpuConstRNSPolyView make_plain_q_prefix_const_view(
    const GpuPlaintextData &pt,
    std::size_t q_count)
{
    GpuConstRNSPolyView view = pt.make_const_view().poly;
    view.shards.front().limb_count = q_count;
    return view;
}

bool ciphertext_storage_supports_q_prefix(
    const GpuCiphertextData &dst,
    const GpuCiphertextMeta &meta,
    std::size_t component_count,
    int device_id)
{
    return !dst.empty() &&
           !dst.fields_.empty() &&
           (dst.fields_.size() == 1 || dst.fields_.size() == component_count) &&
           dst.polys_.size() == component_count &&
           dst.meta.degree == meta.degree &&
           dst.meta.q_count >= meta.q_count &&
           dst.meta.p_count == 0 &&
           all_components_use_layout(dst, dst.polys_.at(0)) &&
           dst.polys_.at(0).shards.size() == 1 &&
           dst.fields_.at(0).device_id == device_id &&
           dst.polys_.at(0).shards.front().limb_begin == 0 &&
           dst.polys_.at(0).shards.front().limb_count >= meta.q_count &&
           dst.polys_.at(0).shards.front().coeff_begin == 0 &&
           dst.polys_.at(0).shards.front().coeff_count == meta.degree;
}

void commit_q_prefix_metadata(
    GpuCiphertextData &dst,
    const GpuCiphertextMeta &meta)
{
    dst.meta = meta;
    for (auto &poly : dst.polys_)
    {
        poly.q_count = meta.q_count;
        poly.p_count = 0;
        poly.shards.front().limb_count = meta.q_count;
    }
}

}  // namespace

void GpuEvaluator::add_pair(
    const GpuCiphertextData &left_a,
    const GpuCiphertextData &right_a,
    const GpuCiphertextData &left_b,
    const GpuCiphertextData &right_b,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    const bool no_cross_alias =
        &destination_a != &destination_b &&
        &destination_a != &left_b && &destination_a != &right_b &&
        &destination_b != &left_a && &destination_b != &right_a;
    if (!no_cross_alias ||
        !pair_has_single_full_shard(left_a) || !pair_has_single_full_shard(right_a) ||
        !pair_has_single_full_shard(left_b) || !pair_has_single_full_shard(right_b) ||
        !(left_a.meta.parms_id == right_a.meta.parms_id) ||
        !(left_a.meta.parms_id == left_b.meta.parms_id) ||
        !(left_a.meta.parms_id == right_b.meta.parms_id) ||
        !left_a.meta.is_ntt_form || !right_a.meta.is_ntt_form ||
        !left_b.meta.is_ntt_form || !right_b.meta.is_ntt_form ||
        left_a.size() != right_a.size() ||
        left_b.size() != right_b.size() ||
        left_a.size() + left_b.size() > kernel::kMaxElementwisePairPtrs ||
        !same_scale(left_a.meta.scale, right_a.meta.scale) ||
        !same_scale(left_b.meta.scale, right_b.meta.scale) ||
        left_a.fields_[0].device_id != left_b.fields_[0].device_id)
    {
        add(left_a, right_a, destination_a);
        add(left_b, right_b, destination_b);
        return;
    }

    const int device_id = left_a.fields_[0].device_id;
    const auto &level_info = params_.get_level(left_a.meta.parms_id);
    prepare_ciphertext_destination(
        destination_a, nullptr, nullptr, left_a.meta, left_a.size(), device_id, left_a.polys_[0]);
    prepare_ciphertext_destination(
        destination_b, nullptr, nullptr, left_b.meta, left_b.size(), device_id, left_b.polys_[0]);
    auto dst_a_view = destination_a.make_view();
    auto dst_b_view = destination_b.make_view();
    elementwise_handler_.add_or_sub_ciphertext_pair(
        dst_a_view, dst_b_view,
        left_a.make_const_view(), right_a.make_const_view(),
        left_b.make_const_view(), right_b.make_const_view(),
        false, level_info);
}

void GpuEvaluator::sub_pair(
    const GpuCiphertextData &left_a,
    const GpuCiphertextData &right_a,
    const GpuCiphertextData &left_b,
    const GpuCiphertextData &right_b,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    const bool no_cross_alias =
        &destination_a != &destination_b &&
        &destination_a != &left_b && &destination_a != &right_b &&
        &destination_b != &left_a && &destination_b != &right_a;
    if (!no_cross_alias ||
        !pair_has_single_full_shard(left_a) || !pair_has_single_full_shard(right_a) ||
        !pair_has_single_full_shard(left_b) || !pair_has_single_full_shard(right_b) ||
        !(left_a.meta.parms_id == right_a.meta.parms_id) ||
        !(left_a.meta.parms_id == left_b.meta.parms_id) ||
        !(left_a.meta.parms_id == right_b.meta.parms_id) ||
        !left_a.meta.is_ntt_form || !right_a.meta.is_ntt_form ||
        !left_b.meta.is_ntt_form || !right_b.meta.is_ntt_form ||
        left_a.size() != right_a.size() ||
        left_b.size() != right_b.size() ||
        left_a.size() + left_b.size() > kernel::kMaxElementwisePairPtrs ||
        !same_scale(left_a.meta.scale, right_a.meta.scale) ||
        !same_scale(left_b.meta.scale, right_b.meta.scale) ||
        left_a.fields_[0].device_id != left_b.fields_[0].device_id)
    {
        sub(left_a, right_a, destination_a);
        sub(left_b, right_b, destination_b);
        return;
    }

    const int device_id = left_a.fields_[0].device_id;
    const auto &level_info = params_.get_level(left_a.meta.parms_id);
    prepare_ciphertext_destination(
        destination_a, nullptr, nullptr, left_a.meta, left_a.size(), device_id, left_a.polys_[0]);
    prepare_ciphertext_destination(
        destination_b, nullptr, nullptr, left_b.meta, left_b.size(), device_id, left_b.polys_[0]);
    auto dst_a_view = destination_a.make_view();
    auto dst_b_view = destination_b.make_view();
    elementwise_handler_.add_or_sub_ciphertext_pair(
        dst_a_view, dst_b_view,
        left_a.make_const_view(), right_a.make_const_view(),
        left_b.make_const_view(), right_b.make_const_view(),
        true, level_info);
}

void GpuEvaluator::multiply_pair(
    const GpuCiphertextData &left_a,
    const GpuCiphertextData &right_a,
    const GpuCiphertextData &left_b,
    const GpuCiphertextData &right_b,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    const bool no_cross_alias =
        &destination_a != &destination_b &&
        &destination_a != &left_b && &destination_a != &right_b &&
        &destination_b != &left_a && &destination_b != &right_a;
    if (!no_cross_alias ||
        !pair_has_single_full_shard(left_a) || !pair_has_single_full_shard(right_a) ||
        !pair_has_single_full_shard(left_b) || !pair_has_single_full_shard(right_b) ||
        !(left_a.meta.parms_id == right_a.meta.parms_id) ||
        !(left_a.meta.parms_id == left_b.meta.parms_id) ||
        !(left_a.meta.parms_id == right_b.meta.parms_id) ||
        !left_a.meta.is_ntt_form || !right_a.meta.is_ntt_form ||
        !left_b.meta.is_ntt_form || !right_b.meta.is_ntt_form ||
        left_a.size() != 2 || right_a.size() != 2 ||
        left_b.size() != 2 || right_b.size() != 2 ||
        left_a.fields_[0].device_id != left_b.fields_[0].device_id)
    {
        multiply(left_a, right_a, destination_a);
        multiply(left_b, right_b, destination_b);
        return;
    }

    GpuCiphertextMeta meta_a = left_a.meta;
    meta_a.component_count = 3;
    meta_a.is_ntt_form = true;
    meta_a.scale = left_a.meta.scale * right_a.meta.scale;
    GpuCiphertextMeta meta_b = left_b.meta;
    meta_b.component_count = 3;
    meta_b.is_ntt_form = true;
    meta_b.scale = left_b.meta.scale * right_b.meta.scale;
    if (!(meta_a.scale > 0.0) || !std::isfinite(meta_a.scale) ||
        !(meta_b.scale > 0.0) || !std::isfinite(meta_b.scale))
    {
        throw std::invalid_argument("GpuEvaluator::multiply_pair: invalid result scale");
    }

    const int device_id = left_a.fields_[0].device_id;
    const auto &level_info = params_.get_level(left_a.meta.parms_id);
    const bool alias_a = &destination_a == &left_a || &destination_a == &right_a;
    const bool alias_b = &destination_b == &left_b || &destination_b == &right_b;
    GpuCiphertextData local_a, local_b;
    GpuCiphertextData *out_a = alias_a ? &local_a : &destination_a;
    GpuCiphertextData *out_b = alias_b ? &local_b : &destination_b;
    prepare_ciphertext_destination(
        *out_a, nullptr, nullptr, meta_a, 3, device_id, left_a.polys_[0]);
    prepare_ciphertext_destination(
        *out_b, nullptr, nullptr, meta_b, 3, device_id, left_b.polys_[0]);

    auto dst_a_view = out_a->make_view();
    auto dst_b_view = out_b->make_view();
    elementwise_handler_.multiply_ciphertext_pair(
        dst_a_view, dst_b_view,
        left_a.make_const_view(), right_a.make_const_view(),
        left_b.make_const_view(), right_b.make_const_view(),
        level_info);
    if (alias_a) destination_a = std::move(local_a);
    if (alias_b) destination_b = std::move(local_b);
}

void GpuEvaluator::multiply_q_prefix_pair(
    const GpuCiphertextData &left_a,
    const GpuCiphertextData &right_a,
    const GpuCiphertextData &left_b,
    const GpuCiphertextData &right_b,
    parms_id_type target_parms_id,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    const auto &target_level = params_.get_level(target_parms_id);
    const bool no_cross_alias =
        &destination_a != &destination_b &&
        &destination_a != &left_b && &destination_a != &right_b &&
        &destination_b != &left_a && &destination_b != &right_a;
    const bool fast_ok =
        no_cross_alias &&
        target_level.p_count == 0 && target_level.q_count != 0 &&
        pair_has_single_full_shard(left_a) && pair_has_single_full_shard(right_a) &&
        pair_has_single_full_shard(left_b) && pair_has_single_full_shard(right_b) &&
        left_a.size() == 2 && right_a.size() == 2 &&
        left_b.size() == 2 && right_b.size() == 2 &&
        left_a.meta.is_ntt_form && right_a.meta.is_ntt_form &&
        left_b.meta.is_ntt_form && right_b.meta.is_ntt_form &&
        left_a.meta.degree == target_level.degree &&
        right_a.meta.degree == target_level.degree &&
        left_b.meta.degree == target_level.degree &&
        right_b.meta.degree == target_level.degree &&
        left_a.meta.q_count >= target_level.q_count &&
        right_a.meta.q_count >= target_level.q_count &&
        left_b.meta.q_count >= target_level.q_count &&
        right_b.meta.q_count >= target_level.q_count &&
        left_a.fields_[0].device_id == left_b.fields_[0].device_id;

    if (!fast_ok)
    {
        multiply_q_prefix(left_a, right_a, destination_a, target_parms_id);
        multiply_q_prefix(left_b, right_b, destination_b, target_parms_id);
        return;
    }

    GpuCiphertextMeta meta_a = left_a.meta;
    meta_a.parms_id = target_level.parms_id;
    meta_a.q_count = target_level.q_count;
    meta_a.component_count = 3;
    meta_a.scale = left_a.meta.scale * right_a.meta.scale;
    GpuCiphertextMeta meta_b = left_b.meta;
    meta_b.parms_id = target_level.parms_id;
    meta_b.q_count = target_level.q_count;
    meta_b.component_count = 3;
    meta_b.scale = left_b.meta.scale * right_b.meta.scale;
    if (!(meta_a.scale > 0.0) || !std::isfinite(meta_a.scale) ||
        !(meta_b.scale > 0.0) || !std::isfinite(meta_b.scale))
    {
        throw std::invalid_argument("GpuEvaluator::multiply_q_prefix_pair: invalid result scale");
    }

    const int device_id = left_a.fields_[0].device_id;
    GpuRNSPoly ref_layout = left_a.polys_[0];
    ref_layout.q_count = target_level.q_count;
    ref_layout.shards.front().limb_count = target_level.q_count;
    const bool alias_a = &destination_a == &left_a || &destination_a == &right_a;
    const bool alias_b = &destination_b == &left_b || &destination_b == &right_b;
    GpuCiphertextData local_a, local_b;
    GpuCiphertextData *out_a = alias_a ? &local_a : &destination_a;
    GpuCiphertextData *out_b = alias_b ? &local_b : &destination_b;
    prepare_ciphertext_destination(
        *out_a, nullptr, nullptr, meta_a, 3, device_id, ref_layout);
    prepare_ciphertext_destination(
        *out_b, nullptr, nullptr, meta_b, 3, device_id, ref_layout);

    auto dst_a_view = out_a->make_view();
    auto dst_b_view = out_b->make_view();
    elementwise_handler_.multiply_ciphertext_pair(
        dst_a_view, dst_b_view,
        make_q_prefix_const_view(left_a, target_level),
        make_q_prefix_const_view(right_a, target_level),
        make_q_prefix_const_view(left_b, target_level),
        make_q_prefix_const_view(right_b, target_level),
        target_level);
    if (alias_a) destination_a = std::move(local_a);
    if (alias_b) destination_b = std::move(local_b);
}

void GpuEvaluator::square_pair(
    const GpuCiphertextData &source_a,
    const GpuCiphertextData &source_b,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    multiply_pair(source_a, source_a, source_b, source_b, destination_a, destination_b);
}

void GpuEvaluator::multiply_plain_pair(
    const GpuCiphertextData &source_a,
    const GpuCiphertextData &source_b,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    const bool no_cross_alias =
        &destination_a != &destination_b &&
        &destination_a != &source_b && &destination_b != &source_a;
    if (!no_cross_alias ||
        !pair_has_single_full_shard(source_a) || !pair_has_single_full_shard(source_b) ||
        !plain_has_single_full_shard(source_plaintext) ||
        !(source_a.meta.parms_id == source_plaintext.meta.parms_id) ||
        !(source_b.meta.parms_id == source_plaintext.meta.parms_id) ||
        !source_a.meta.is_ntt_form || !source_b.meta.is_ntt_form ||
        !source_plaintext.meta.is_ntt_form ||
        source_a.size() + source_b.size() > kernel::kMaxElementwisePairPtrs ||
        source_a.fields_[0].device_id != source_b.fields_[0].device_id ||
        source_a.fields_[0].device_id != source_plaintext.fields_.front().device_id)
    {
        multiply_plain(source_a, source_plaintext, destination_a);
        multiply_plain(source_b, source_plaintext, destination_b);
        return;
    }

    GpuCiphertextMeta meta_a = source_a.meta;
    meta_a.scale = source_a.meta.scale * source_plaintext.meta.scale;
    GpuCiphertextMeta meta_b = source_b.meta;
    meta_b.scale = source_b.meta.scale * source_plaintext.meta.scale;
    const int device_id = source_a.fields_[0].device_id;
    const auto &level_info = params_.get_level(source_plaintext.meta.parms_id);

    prepare_ciphertext_destination(
        destination_a, nullptr, nullptr, meta_a, source_a.size(), device_id, source_a.polys_[0]);
    prepare_ciphertext_destination(
        destination_b, nullptr, nullptr, meta_b, source_b.size(), device_id, source_b.polys_[0]);

    auto dst_a_view = destination_a.make_view();
    auto dst_b_view = destination_b.make_view();
    elementwise_handler_.multiply_plain_pair(
        dst_a_view, dst_b_view,
        source_a.make_const_view(), source_b.make_const_view(),
        source_plaintext.make_const_view().poly,
        level_info);
}

void GpuEvaluator::multiply_plain_q_prefix_pair(
    const GpuCiphertextData &source_a,
    const GpuCiphertextData &source_b,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    const auto &level_info = params_.get_level(source_plaintext.meta.parms_id);
    const bool no_cross_alias =
        &destination_a != &destination_b &&
        &destination_a != &source_b && &destination_b != &source_a;
    const bool fast_ok =
        no_cross_alias &&
        level_info.p_count == 0 && level_info.q_count != 0 &&
        pair_has_single_full_shard(source_a) && pair_has_single_full_shard(source_b) &&
        plain_has_single_full_shard(source_plaintext) &&
        source_a.meta.is_ntt_form && source_b.meta.is_ntt_form &&
        source_plaintext.meta.is_ntt_form &&
        source_a.meta.degree == level_info.degree &&
        source_b.meta.degree == level_info.degree &&
        source_a.meta.q_count >= level_info.q_count &&
        source_b.meta.q_count >= level_info.q_count &&
        source_a.size() + source_b.size() <= kernel::kMaxElementwisePairPtrs &&
        source_a.fields_[0].device_id == source_b.fields_[0].device_id &&
        source_a.fields_[0].device_id == source_plaintext.fields_.front().device_id;

    if (!fast_ok)
    {
        multiply_plain_q_prefix(source_a, source_plaintext, destination_a);
        multiply_plain_q_prefix(source_b, source_plaintext, destination_b);
        return;
    }

    GpuCiphertextMeta meta_a = source_a.meta;
    meta_a.parms_id = level_info.parms_id;
    meta_a.q_count = level_info.q_count;
    meta_a.scale = source_a.meta.scale * source_plaintext.meta.scale;
    GpuCiphertextMeta meta_b = source_b.meta;
    meta_b.parms_id = level_info.parms_id;
    meta_b.q_count = level_info.q_count;
    meta_b.scale = source_b.meta.scale * source_plaintext.meta.scale;
    const int device_id = source_a.fields_[0].device_id;

    GpuRNSPoly ref_layout = source_a.polys_[0];
    ref_layout.q_count = level_info.q_count;
    ref_layout.shards.front().limb_count = level_info.q_count;

    const bool reuse_a =
        ciphertext_storage_supports_q_prefix(destination_a, meta_a, source_a.size(), device_id);
    const bool reuse_b =
        ciphertext_storage_supports_q_prefix(destination_b, meta_b, source_b.size(), device_id);
    if (!reuse_a)
    {
        prepare_ciphertext_destination(
            destination_a, &source_a, nullptr, meta_a, source_a.size(), device_id, ref_layout);
    }
    if (!reuse_b)
    {
        prepare_ciphertext_destination(
            destination_b, &source_b, nullptr, meta_b, source_b.size(), device_id, ref_layout);
    }

    auto dst_a_view = reuse_a
        ? make_q_prefix_mutable_view(destination_a, meta_a)
        : destination_a.make_view();
    auto dst_b_view = reuse_b
        ? make_q_prefix_mutable_view(destination_b, meta_b)
        : destination_b.make_view();
    elementwise_handler_.multiply_plain_pair(
        dst_a_view, dst_b_view,
        make_q_prefix_const_view(source_a, level_info),
        make_q_prefix_const_view(source_b, level_info),
        source_plaintext.make_const_view().poly,
        level_info);
    if (reuse_a)
    {
        commit_q_prefix_metadata(destination_a, meta_a);
    }
    if (reuse_b)
    {
        commit_q_prefix_metadata(destination_b, meta_b);
    }
}

void GpuEvaluator::double_sub_or_add_plain_pair(
    kernel::DoublePlainPairOp op,
    const GpuCiphertextData &source_a,
    const GpuCiphertextData *source_b,
    const GpuPlaintextData &plaintext,
    const GpuCiphertextData *correction_a,
    const GpuCiphertextData *correction_b,
    GpuCiphertextData &destination_a,
    GpuCiphertextData *destination_b,
    const char *caller_name) const
{
    if (!pair_has_single_full_shard(source_a) ||
        source_a.size() != 2 ||
        !source_a.meta.is_ntt_form ||
        !plain_has_single_full_shard(plaintext) ||
        !plaintext.meta.is_ntt_form)
    {
        throw std::invalid_argument(std::string(caller_name) + ": invalid lane A / plaintext input");
    }
    const bool has_b = source_b != nullptr && destination_b != nullptr;
    if ((source_b == nullptr) != (destination_b == nullptr))
    {
        throw std::invalid_argument(std::string(caller_name) + ": invalid lane B pair pointers");
    }
    if (has_b &&
        (!pair_has_single_full_shard(*source_b) ||
         source_b->size() != 2 ||
         !source_b->meta.is_ntt_form ||
         !(source_b->meta.parms_id == source_a.meta.parms_id) ||
         source_b->fields_[0].device_id != source_a.fields_[0].device_id))
    {
        throw std::invalid_argument(std::string(caller_name) + ": lane B mismatch");
    }

    const int device_id = source_a.fields_[0].device_id;
    const auto &level_info = params_.get_level(source_a.meta.parms_id);
    if (plaintext.meta.q_count < source_a.meta.q_count)
    {
        throw std::invalid_argument(std::string(caller_name) + ": plaintext q_count below source");
    }
    if (plaintext.meta.q_count == source_a.meta.q_count &&
        !(plaintext.meta.parms_id == source_a.meta.parms_id))
    {
        throw std::invalid_argument(std::string(caller_name) + ": plaintext parms_id mismatch");
    }
    const GpuConstRNSPolyView plain_view = plaintext.meta.q_count == level_info.q_count
        ? plaintext.make_const_view().poly
        : make_plain_q_prefix_const_view(plaintext, level_info.q_count);

    GpuConstCiphertextView corr_a_view;
    GpuConstCiphertextView corr_b_view;
    if (op == kernel::DoublePlainPairOp::SubCorr)
    {
        if (correction_a == nullptr || !pair_has_single_full_shard(*correction_a) ||
            correction_a->size() != 2 || !correction_a->meta.is_ntt_form ||
            correction_a->meta.q_count < level_info.q_count ||
            (correction_a->meta.q_count == level_info.q_count &&
             !(correction_a->meta.parms_id == level_info.parms_id)))
        {
            throw std::invalid_argument(std::string(caller_name) + ": invalid correction A");
        }
        corr_a_view = correction_a->meta.q_count == level_info.q_count
            ? correction_a->make_const_view()
            : make_q_prefix_const_view(*correction_a, level_info);
        if (has_b)
        {
            if (correction_b == nullptr || !pair_has_single_full_shard(*correction_b) ||
                correction_b->size() != 2 || !correction_b->meta.is_ntt_form ||
                correction_b->meta.q_count < level_info.q_count ||
                (correction_b->meta.q_count == level_info.q_count &&
                 !(correction_b->meta.parms_id == level_info.parms_id)))
            {
                throw std::invalid_argument(std::string(caller_name) + ": invalid correction B");
            }
            corr_b_view = correction_b->meta.q_count == level_info.q_count
                ? correction_b->make_const_view()
                : make_q_prefix_const_view(*correction_b, level_info);
        }
    }

    // Alias-safe per-lane (src -> dst); allocate only on cross-lane aliasing.
    prepare_ciphertext_destination(
        destination_a,
        has_b ? source_b : nullptr,
        has_b ? correction_b : nullptr,
        source_a.meta,
        2,
        device_id,
        source_a.polys_[0]);
    if (has_b)
    {
        prepare_ciphertext_destination(
            *destination_b,
            &source_a,
            correction_a,
            source_b->meta,
            2,
            device_id,
            source_b->polys_[0]);
    }

    auto dst_a_view = destination_a.make_view();
    GpuCiphertextView dst_b_view;
    GpuConstCiphertextView src_b_view;
    if (has_b)
    {
        dst_b_view = destination_b->make_view();
        src_b_view = source_b->make_const_view();
    }
    elementwise_handler_.double_sub_or_add_plain_pair(
        op,
        dst_a_view,
        has_b ? &dst_b_view : nullptr,
        source_a.make_const_view(),
        has_b ? &src_b_view : nullptr,
        plain_view,
        op == kernel::DoublePlainPairOp::SubCorr ? &corr_a_view : nullptr,
        (has_b && op == kernel::DoublePlainPairOp::SubCorr) ? &corr_b_view : nullptr,
        level_info);
}

void GpuEvaluator::rescale_many_pair(
    const GpuCiphertextData &source_a,
    const GpuCiphertextData &source_b,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b,
    std::uint32_t rescale_count) const
{
    if (rescale_count == 0)
    {
        if (&destination_a != &source_a)
        {
            copy_ciphertext_data(source_a, destination_a, "rescale_many_pair zero copy A");
        }
        if (&destination_b != &source_b)
        {
            copy_ciphertext_data(source_b, destination_b, "rescale_many_pair zero copy B");
        }
        return;
    }

    const bool no_cross_alias =
        &destination_a != &destination_b &&
        &destination_a != &source_b &&
        &destination_b != &source_a;
    const bool fast_ok =
        no_cross_alias &&
        pair_has_single_full_shard(source_a) &&
        pair_has_single_full_shard(source_b) &&
        !(source_a.meta.parms_id != source_b.meta.parms_id) &&
        source_a.meta.is_ntt_form && source_b.meta.is_ntt_form &&
        source_a.meta.q_count > rescale_count &&
        source_a.size() == source_b.size() &&
        source_a.fields_[0].device_id == source_b.fields_[0].device_id;

    if (!fast_ok)
    {
        rescale_many(source_a, destination_a, rescale_count);
        rescale_many(source_b, destination_b, rescale_count);
        return;
    }

    const auto &src_level_info = params_.get_level(source_a.meta.parms_id);
    const GpuLevelInfo *cur = &src_level_info;
    long double scale_a = static_cast<long double>(source_a.meta.scale);
    long double scale_b = static_cast<long double>(source_b.meta.scale);
    for (std::uint32_t s = 0; s < rescale_count; ++s)
    {
        const GpuWord q_last = cur->shards.front().q_last;
        scale_a /= static_cast<long double>(q_last);
        scale_b /= static_cast<long double>(q_last);
        cur = &params_.get_next_level(cur->parms_id);
    }
    const auto &dst_level_info = *cur;
    GpuCiphertextMeta meta_a = source_a.meta;
    meta_a.parms_id = dst_level_info.parms_id;
    meta_a.q_count = dst_level_info.q_count;
    meta_a.scale = static_cast<double>(scale_a);
    GpuCiphertextMeta meta_b = source_b.meta;
    meta_b.parms_id = dst_level_info.parms_id;
    meta_b.q_count = dst_level_info.q_count;
    meta_b.scale = static_cast<double>(scale_b);

    const int device_id = source_a.fields_[0].device_id;
    const std::size_t components = source_a.size();
    GpuRNSPoly ref_layout = source_a.polys_[0];
    ref_layout.q_count = dst_level_info.q_count;
    ref_layout.shards.front().limb_count = dst_level_info.q_count;

    auto src_a_view = source_a.make_const_view();
    auto src_b_view = source_b.make_const_view();

    const bool reuse_a =
        ciphertext_storage_supports_q_prefix(destination_a, meta_a, components, device_id);
    const bool reuse_b =
        ciphertext_storage_supports_q_prefix(destination_b, meta_b, components, device_id);
    const bool need_local_a = !reuse_a && &destination_a == &source_a;
    const bool need_local_b = !reuse_b && &destination_b == &source_b;
    GpuCiphertextData local_a, local_b;
    GpuCiphertextData *out_a = need_local_a ? &local_a : &destination_a;
    GpuCiphertextData *out_b = need_local_b ? &local_b : &destination_b;
    if (!reuse_a)
    {
        prepare_ciphertext_destination(
            *out_a, nullptr, nullptr, meta_a, components, device_id, ref_layout);
    }
    if (!reuse_b)
    {
        prepare_ciphertext_destination(
            *out_b, nullptr, nullptr, meta_b, components, device_id, ref_layout);
    }

    auto dst_a_view = reuse_a
        ? make_q_prefix_mutable_view(*out_a, meta_a)
        : out_a->make_view();
    auto dst_b_view = reuse_b
        ? make_q_prefix_mutable_view(*out_b, meta_b)
        : out_b->make_view();

    bool handled = false;
    if (rescale_count == 1)
    {
        handled = modswitch_handler_.rescale_ciphertext_pair(
            dst_a_view, dst_b_view, src_a_view, src_b_view, src_level_info, dst_level_info);
    }
    else if (rescale_count == 2)
    {
        handled = modswitch_handler_.rescale_ciphertext_x2_pair(
            dst_a_view, dst_b_view, src_a_view, src_b_view, src_level_info, dst_level_info);
    }
    else
    {
        handled = modswitch_handler_.rescale_ciphertext_many_pair(
            dst_a_view, dst_b_view, src_a_view, src_b_view, src_level_info, dst_level_info, rescale_count);
    }

    if (!handled)
    {
        rescale_many(source_a, destination_a, rescale_count);
        rescale_many(source_b, destination_b, rescale_count);
        return;
    }

    if (reuse_a)
    {
        commit_q_prefix_metadata(*out_a, meta_a);
    }
    else if (need_local_a)
    {
        destination_a = std::move(local_a);
    }
    if (reuse_b)
    {
        commit_q_prefix_metadata(*out_b, meta_b);
    }
    else if (need_local_b)
    {
        destination_b = std::move(local_b);
    }
}

void GpuEvaluator::rescale_pair(
    const GpuCiphertextData &source_a,
    const GpuCiphertextData &source_b,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    rescale_many_pair(source_a, source_b, destination_a, destination_b, 1);
}

void GpuEvaluator::rescale_x2_pair(
    const GpuCiphertextData &source_a,
    const GpuCiphertextData &source_b,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    rescale_many_pair(source_a, source_b, destination_a, destination_b, 2);
}

void GpuEvaluator::rescale_dynamic_pair(
    const GpuCiphertextData &source_a,
    const GpuCiphertextData &source_b,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b,
    double min_scale) const
{
    if (!pair_has_single_full_shard(source_a) ||
        !pair_has_single_full_shard(source_b) ||
        !(source_a.meta.parms_id == source_b.meta.parms_id) ||
        !source_a.meta.is_ntt_form || !source_b.meta.is_ntt_form ||
        source_a.size() != source_b.size() ||
        !same_scale(source_a.meta.scale, source_b.meta.scale))
    {
        rescale_dynamic(source_a, destination_a, min_scale);
        rescale_dynamic(source_b, destination_b, min_scale);
        return;
    }

    const std::size_t q_count = source_a.meta.q_count;
    std::vector<std::uint64_t> active_moduli(q_count);
    const GpuLevelInfo *lvl = &params_.get_level(source_a.meta.parms_id);
    for (std::size_t idx = q_count; idx-- > 0;)
    {
        const GpuWord ql = lvl->shards.front().q_last;
        active_moduli[idx] = ql != 0 ? ql : 1ULL;
        if (idx > 0)
        {
            lvl = &params_.get_next_level(lvl->parms_id);
        }
    }
    const auto plan = plan_gpu_dynamic_rescale(
        source_a.meta.scale, min_scale, active_moduli, false);
    if (plan.rescale_count == 0)
    {
        if (&destination_a != &source_a)
        {
            copy_ciphertext_data(source_a, destination_a, "rescale_dynamic_pair zero A");
        }
        if (&destination_b != &source_b)
        {
            copy_ciphertext_data(source_b, destination_b, "rescale_dynamic_pair zero B");
        }
        return;
    }
    rescale_many_pair(source_a, source_b, destination_a, destination_b, plan.rescale_count);
    destination_a.meta.scale = plan.output_scale;
    destination_b.meta.scale = plan.output_scale;
}

void GpuEvaluator::multiply_scalar(
    const GpuCiphertextData &source_ciphertext,
    std::uint64_t scalar,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: empty ciphertext storage");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: p limbs are not supported yet");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: component metadata mismatch");
    }

    const auto &reference_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: shard layout mismatch");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);
    if (level_info.q_count != source_ciphertext.meta.q_count ||
        level_info.p_count != source_ciphertext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: level shape mismatch");
    }

    // Each coefficient is read and written independently, so the existing
    // source storage is safe for an aliased scalar multiplication.
    if (&destination_ciphertext != &source_ciphertext)
    {
        prepare_ciphertext_destination(
            destination_ciphertext,
            &source_ciphertext,
            nullptr,
            source_ciphertext.meta,
            source_ciphertext.size(),
            device_id,
            reference_layout);
    }

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();
    elementwise_handler_.multiply_scalar_ciphertext(
        destination_view,
        source_view,
        scalar,
        level_info);
}

void GpuEvaluator::raise_modulus(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::raise_modulus: empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument("GpuEvaluator::raise_modulus: empty ciphertext storage");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument("GpuEvaluator::raise_modulus: p limbs are not supported yet");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::raise_modulus: component metadata mismatch");
    }

    const auto &source_level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    const auto &destination_level_info = params_.get_first_q_level();
    if (source_level_info.p_count != 0 ||
        destination_level_info.p_count != 0 ||
        source_level_info.degree != source_ciphertext.meta.degree ||
        source_level_info.q_count != source_ciphertext.meta.q_count ||
        destination_level_info.degree != source_ciphertext.meta.degree ||
        source_level_info.q_count > destination_level_info.q_count)
    {
        throw std::invalid_argument("GpuEvaluator::raise_modulus: level shape mismatch");
    }

    const auto &source_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, source_layout) ||
        source_layout.shards.size() != 1 ||
        source_layout.shards.front().limb_begin != 0 ||
        source_layout.shards.front().limb_count != source_ciphertext.meta.q_count ||
        source_layout.shards.front().coeff_begin != 0 ||
        source_layout.shards.front().coeff_count != source_ciphertext.meta.degree)
    {
        throw std::invalid_argument(
            "GpuEvaluator::raise_modulus: first implementation requires one full q-prefix shard");
    }

    GpuCiphertextData coeff_source_storage;
    const GpuCiphertextData *coeff_source = &source_ciphertext;
    if (source_ciphertext.meta.is_ntt_form)
    {
        ntt_inv(source_ciphertext, coeff_source_storage);
        coeff_source = &coeff_source_storage;
    }

    const int device_id = coeff_source->fields_.at(0).device_id;
    GpuPolyShard destination_shard;
    destination_shard.field_index = 0;
    destination_shard.field_offset = 0;
    destination_shard.limb_begin = 0;
    destination_shard.limb_count = destination_level_info.q_count;
    destination_shard.coeff_begin = 0;
    destination_shard.coeff_count = coeff_source->meta.degree;

    GpuCiphertextData coeff_raised =
        GpuCiphertextData::allocate_single_device_sharded(
            coeff_source->meta.degree,
            destination_level_info.q_count,
            coeff_source->size(),
            device_id,
            std::vector<GpuPolyShard>{ destination_shard },
            0);

    coeff_raised.meta = coeff_source->meta;
    coeff_raised.meta.parms_id = destination_level_info.parms_id;
    coeff_raised.meta.q_count = destination_level_info.q_count;
    coeff_raised.meta.p_count = 0;
    coeff_raised.meta.component_count = coeff_source->size();
    coeff_raised.meta.is_ntt_form = false;

    auto source_view = coeff_source->make_const_view();
    auto destination_view = coeff_raised.make_view();
    modswitch_handler_.raise_modulus_ciphertext(
        destination_view,
        source_view,
        source_level_info,
        destination_level_info);

    ntt_fwd(coeff_raised, destination_ciphertext);
}

void GpuEvaluator::bootstrap_prepare_modraise_input(
    const GpuCiphertextData &source_ciphertext,
    GpuCiphertextData &destination_ciphertext,
    parms_id_type q0_parms_id,
    double q0_over_message_ratio) const
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::bootstrap_prepare_modraise_input",
        source_ciphertext,
        true);
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_prepare_modraise_input: p limbs are not supported");
    }
    if (!(q0_over_message_ratio > 0.0) || !std::isfinite(q0_over_message_ratio))
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_prepare_modraise_input: invalid target scale");
    }

    const auto &q0_level_info = params_.get_level(q0_parms_id);
    if (q0_level_info.p_count != 0 ||
        q0_level_info.degree != source_ciphertext.meta.degree)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_prepare_modraise_input: invalid q0 level");
    }

    if (source_ciphertext.meta.q_count < q0_level_info.q_count + 1)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_prepare_modraise_input: input needs one q modulus above q0 for scale normalization");
    }
    // Multiply before rescaling: rounding target/scale alone loses precision
    // when that ratio is small. The extra q gives the integer multiplier
    // another modulus worth of precision without lowering the input scale.
    //
    // One extra q stays the choice whenever its multiplier rounds within
    // kOneModulusRounding.  For an input scale far above q0/ratio (2^80
    // against 2^53) that multiplier is ~32 and rounds the scale by ~1.2%;
    // the output scale records it, but a Boot's raised-scale override then
    // discards it and the message comes back scaled.  There, further moduli
    // are divided out until the multiplier rounds within kPreciseRounding,
    // keeping it inside the exact integer range.  With no better choice the
    // most precise one is kept (bootstrap() refuses what it cannot use).
    constexpr double kOneModulusRounding = 1e-6;
    constexpr double kPreciseRounding = 1e-9;
    const double exact_integer_limit = std::ldexp(1.0, 53);
    std::size_t extra_moduli = 0;
    double scale_multiplier_double = 0.0;
    double rounding_error = std::numeric_limits<double>::infinity();
    double divisor = 1.0;
    for (std::size_t extra = 1;
         q0_level_info.q_count + extra <= source_ciphertext.meta.q_count; ++extra)
    {
        const auto &level =
            params_.get_level_by_q_count(q0_level_info.q_count + extra, 0);
        if (level.shards.empty() || level.shards.front().q_last == 0)
        {
            throw std::invalid_argument(
                "GpuEvaluator::bootstrap_prepare_modraise_input: missing q_last parameter");
        }
        divisor *= static_cast<double>(level.shards.front().q_last);
        const double exact =
            q0_over_message_ratio * divisor / source_ciphertext.meta.scale;
        if (!std::isfinite(exact) || exact > exact_integer_limit)
            break;
        const double rounded = std::round(exact);
        if (rounded < 1.0)
            continue;
        const double error = std::abs(rounded / exact - 1.0);
        if (error < rounding_error)
        {
            extra_moduli = extra;
            scale_multiplier_double = rounded;
            rounding_error = error;
        }
        if (error <= (extra == 1 ? kOneModulusRounding : kPreciseRounding))
            break;
    }
    if (extra_moduli == 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_prepare_modraise_input: scale multiplier is outside the exact integer range [1, 2^53]");
    }
    const std::size_t normalization_q_count = q0_level_info.q_count + extra_moduli;
    const auto &normalization_level =
        params_.get_level_by_q_count(normalization_q_count, 0);

    const GpuCiphertextData *current = &source_ciphertext;
    GpuCiphertextData dropped;
    if (source_ciphertext.meta.q_count > normalization_q_count)
    {
        drop_modulus(source_ciphertext, dropped, normalization_level.parms_id);
        current = &dropped;
    }
    GpuCiphertextData scaled;
    multiply_scalar(*current, static_cast<std::uint64_t>(scale_multiplier_double), scaled);
    scaled.meta.scale = current->meta.scale * scale_multiplier_double;
    if (extra_moduli == 1)
        rescale(scaled, destination_ciphertext);
    else
        rescale_many(scaled, destination_ciphertext, static_cast<std::uint32_t>(extra_moduli));
}

/* GpuEvaluator::multiply(...) -> 输出 3 component: d0, d1, d2 */
void GpuEvaluator::relinearize(
    const GpuCiphertextData &source_ciphertext,
    const GpuRelinKeysData &relin_keys_in,
    GpuCiphertextData &destination_ciphertext) const
{
    const auto &relin_keys = level_key_for(params_, relin_keys_in, source_ciphertext.meta, "GpuEvaluator::relinearize");
    if (source_ciphertext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::relinearize: empty ciphertext");
    }
    if (source_ciphertext.fields_.empty())
    {
        throw std::invalid_argument("GpuEvaluator::relinearize: empty ciphertext storage");
    }
    if (relin_keys.empty())
    {
        throw std::invalid_argument("GpuEvaluator::relinearize: empty relin keys");
    }
    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize: CKKS ciphertext must be in NTT form");
    }
    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize: component metadata mismatch");
    }
    /* 重线性化只支持输入分量个数为3 */
    if (source_ciphertext.size() != 3)
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize: first HYBRID implementation expects a size-3 ciphertext");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize: input ciphertext p limbs are not supported");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::relinearize: shard layout mismatch");
    }

    /* 输出仍在同一层级，但重线性化后只保留2个密文分量 */
    GpuCiphertextMeta result_meta = source_ciphertext.meta;
    result_meta.component_count = 2;
    result_meta.is_ntt_form = true;

    const bool aliases_input = &destination_ciphertext == &source_ciphertext;
    GpuCiphertextData local_result;
    GpuCiphertextData *result = &destination_ciphertext;
    if (aliases_input)
    {
        local_result = GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            2,
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);
        local_result.meta = result_meta;
        result = &local_result;
    }
    else
    {
        prepare_ciphertext_destination(
            destination_ciphertext,
            nullptr,
            nullptr,
            result_meta,
            2,
            device_id,
            reference_layout);
    }

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result->make_view();
    const auto &relin_keys_view =
        relin_keys.const_view(source_ciphertext.meta.q_count);
    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    /* 通过dnum分解的方式进行重线性化 */
    keyswitch_handler_.relinearize_hybrid_ciphertext(
        destination_view,
        source_view,
        relin_keys_view,
        relin_keys,
        level_info);

    if (aliases_input)
    {
        destination_ciphertext = std::move(local_result);
    }
}

void GpuEvaluator::relinearize_pair(
    const GpuCiphertextData &source_a,
    const GpuCiphertextData &source_b,
    const GpuRelinKeysData &relin_keys_in,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    if (&destination_a == &destination_b)
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize_pair: destinations must be distinct");
    }
    if (!(source_a.meta.parms_id == source_b.meta.parms_id) ||
        source_a.meta.degree != source_b.meta.degree ||
        source_a.meta.q_count != source_b.meta.q_count ||
        source_a.meta.p_count != 0 || source_b.meta.p_count != 0)
    {
        relinearize(source_a, relin_keys_in, destination_a);
        relinearize(source_b, relin_keys_in, destination_b);
        return;
    }

    const auto &relin_keys = level_key_for(
        params_, relin_keys_in, source_a.meta, "GpuEvaluator::relinearize_pair");
    const std::array<const GpuCiphertextData *, 2> srcs{&source_a, &source_b};
    const std::array<GpuCiphertextData *, 2> dsts{&destination_a, &destination_b};
    for (int lane = 0; lane < 2; ++lane)
    {
        const auto &src = *srcs[lane];
        if (src.empty() || src.fields_.empty() || relin_keys.empty() ||
            !src.meta.is_ntt_form || src.meta.component_count != src.size() ||
            src.size() != 3 || src.meta.p_count != 0 ||
            !all_components_use_layout(src, src.polys_.at(0)))
        {
            throw std::invalid_argument(
                "GpuEvaluator::relinearize_pair: invalid input ciphertext");
        }
    }

    std::array<bool, 2> aliases{};
    std::array<GpuCiphertextData, 2> locals;
    std::array<GpuCiphertextData *, 2> targets{};
    for (int lane = 0; lane < 2; ++lane)
    {
        const auto &src = *srcs[lane];
        const auto &ref_layout = src.polys_.at(0);
        const int device_id = src.fields_.at(0).device_id;
        GpuCiphertextMeta meta = src.meta;
        meta.component_count = 2;
        meta.is_ntt_form = true;
        aliases[lane] = (dsts[lane] == &source_a || dsts[lane] == &source_b);
        if (aliases[lane])
        {
            locals[lane] = GpuCiphertextData::allocate_single_device_sharded(
                src.meta.degree, src.meta.q_count, 2, device_id,
                ref_layout.shards, 0);
            locals[lane].meta = meta;
            targets[lane] = &locals[lane];
        }
        else
        {
            prepare_ciphertext_destination(
                *dsts[lane], nullptr, nullptr, meta, 2, device_id, ref_layout);
            targets[lane] = dsts[lane];
        }
    }

    auto src_a_view = source_a.make_const_view();
    auto src_b_view = source_b.make_const_view();
    auto dst_a_view = targets[0]->make_view();
    auto dst_b_view = targets[1]->make_view();
    const auto &relin_keys_view = relin_keys.const_view(source_a.meta.q_count);
    const auto &level_info = params_.get_level(source_a.meta.parms_id);
    keyswitch_handler_.relinearize_hybrid_ciphertext_pair(
        dst_a_view, dst_b_view, src_a_view, src_b_view,
        relin_keys_view, relin_keys, level_info);

    for (int lane = 0; lane < 2; ++lane)
    {
        if (aliases[lane])
        {
            *dsts[lane] = std::move(locals[lane]);
        }
    }
}

void GpuEvaluator::relinearize_rescale_x2_hybrid(
    const GpuCiphertextData &source_ciphertext,
    const GpuRelinKeysData &relin_keys_in,
    GpuCiphertextData &destination_ciphertext) const
{
    const auto &relin_keys = level_key_for(params_, relin_keys_in, source_ciphertext.meta, "GpuEvaluator::relinearize_rescale_x2_hybrid");
    if (source_ciphertext.empty() ||
        source_ciphertext.fields_.empty() ||
        relin_keys.empty() ||
        !source_ciphertext.meta.is_ntt_form ||
        source_ciphertext.meta.p_count != 0 ||
        source_ciphertext.meta.component_count != 3 ||
        source_ciphertext.size() != 3 ||
        source_ciphertext.meta.q_count < 3)
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize_rescale_x2_hybrid: unsupported ciphertext shape");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &source_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, source_layout))
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize_rescale_x2_hybrid: shard layout mismatch");
    }

    const auto &source_level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    const auto &intermediate_level_info =
        params_.get_next_level(source_ciphertext.meta.parms_id);
    const auto &destination_level_info =
        params_.get_next_level(intermediate_level_info.parms_id);
    if (source_level_info.q_count != source_ciphertext.meta.q_count ||
        destination_level_info.q_count + 2 != source_level_info.q_count ||
        source_level_info.shards.empty() ||
        source_level_info.shards.front().q_last_two_product == 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize_rescale_x2_hybrid: level/constants mismatch");
    }

    const std::size_t destination_q_count =
        source_ciphertext.meta.q_count - 2;
    GpuPolyShard destination_shard;
    destination_shard.field_index = 0;
    destination_shard.field_offset = 0;
    destination_shard.limb_begin = 0;
    destination_shard.limb_count = destination_q_count;
    destination_shard.coeff_begin = 0;
    destination_shard.coeff_count = source_ciphertext.meta.degree;

    GpuCiphertextMeta result_meta = source_ciphertext.meta;
    result_meta.parms_id = destination_level_info.parms_id;
    result_meta.q_count = destination_q_count;
    result_meta.p_count = 0;
    result_meta.component_count = 2;
    result_meta.is_ntt_form = true;
    result_meta.scale =
        source_ciphertext.meta.scale /
        static_cast<double>(
            source_level_info.shards.front().q_last_two_product);
    if (!(result_meta.scale > 0.0) || !std::isfinite(result_meta.scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize_rescale_x2_hybrid: invalid result scale");
    }

    GpuRNSPoly reference_layout;
    reference_layout.degree = source_ciphertext.meta.degree;
    reference_layout.q_count = destination_q_count;
    reference_layout.p_count = 0;
    reference_layout.shards.push_back(destination_shard);

    const bool aliases_input = &destination_ciphertext == &source_ciphertext;
    GpuCiphertextData local_result;
    GpuCiphertextData *result = &destination_ciphertext;
    if (aliases_input)
    {
        local_result = GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            destination_q_count,
            2,
            device_id,
            std::vector<GpuPolyShard>{destination_shard},
            0);
        local_result.meta = result_meta;
        result = &local_result;
    }
    else
    {
        prepare_ciphertext_destination(
            destination_ciphertext,
            nullptr,
            nullptr,
            result_meta,
            2,
            device_id,
            reference_layout);
    }

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result->make_view();
    const auto &relin_keys_view =
        relin_keys.const_view(source_ciphertext.meta.q_count);
    keyswitch_handler_.relinearize_hybrid_ciphertext_rescale_x2(
        destination_view,
        source_view,
        relin_keys_view,
        relin_keys,
        source_level_info,
        destination_level_info);

    if (aliases_input)
    {
        destination_ciphertext = std::move(local_result);
    }
}

void GpuEvaluator::relinearize_rescale_x2_hybrid_pair(
    const GpuCiphertextData &source_a,
    const GpuCiphertextData &source_b,
    const GpuRelinKeysData &relin_keys_in,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    if (&destination_a == &destination_b)
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize_rescale_x2_hybrid_pair: destinations must be distinct");
    }
    if (!(source_a.meta.parms_id == source_b.meta.parms_id) ||
        source_a.meta.degree != source_b.meta.degree ||
        source_a.meta.q_count != source_b.meta.q_count ||
        source_a.meta.p_count != 0 || source_b.meta.p_count != 0)
    {
        relinearize_rescale_x2_hybrid(source_a, relin_keys_in, destination_a);
        relinearize_rescale_x2_hybrid(source_b, relin_keys_in, destination_b);
        return;
    }

    const auto &relin_keys = level_key_for(
        params_, relin_keys_in, source_a.meta,
        "GpuEvaluator::relinearize_rescale_x2_hybrid_pair");
    const std::array<const GpuCiphertextData *, 2> srcs{&source_a, &source_b};
    const std::array<GpuCiphertextData *, 2> dsts{&destination_a, &destination_b};
    for (int lane = 0; lane < 2; ++lane)
    {
        const auto &src = *srcs[lane];
        if (src.empty() || src.fields_.empty() || relin_keys.empty() ||
            !src.meta.is_ntt_form || src.meta.p_count != 0 ||
            src.meta.component_count != 3 || src.size() != 3 ||
            src.meta.q_count < 3 ||
            !all_components_use_layout(src, src.polys_.at(0)))
        {
            throw std::invalid_argument(
                "GpuEvaluator::relinearize_rescale_x2_hybrid_pair: unsupported ciphertext shape");
        }
    }

    const auto &source_level_info = params_.get_level(source_a.meta.parms_id);
    const auto &intermediate_level_info =
        params_.get_next_level(source_a.meta.parms_id);
    const auto &destination_level_info =
        params_.get_next_level(intermediate_level_info.parms_id);
    if (source_level_info.q_count != source_a.meta.q_count ||
        destination_level_info.q_count + 2 != source_level_info.q_count ||
        source_level_info.shards.empty() ||
        source_level_info.shards.front().q_last_two_product == 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::relinearize_rescale_x2_hybrid_pair: level/constants mismatch");
    }

    const std::size_t dst_q_count = source_a.meta.q_count - 2;
    const double q_last_two = static_cast<double>(
        source_level_info.shards.front().q_last_two_product);
    GpuPolyShard dst_shard;
    dst_shard.field_index = 0;
    dst_shard.field_offset = 0;
    dst_shard.limb_begin = 0;
    dst_shard.limb_count = dst_q_count;
    dst_shard.coeff_begin = 0;
    dst_shard.coeff_count = source_a.meta.degree;
    GpuRNSPoly ref_layout;
    ref_layout.degree = source_a.meta.degree;
    ref_layout.q_count = dst_q_count;
    ref_layout.p_count = 0;
    ref_layout.shards.push_back(dst_shard);

    std::array<bool, 2> aliases{};
    std::array<GpuCiphertextData, 2> locals;
    std::array<GpuCiphertextData *, 2> targets{};
    for (int lane = 0; lane < 2; ++lane)
    {
        const auto &src = *srcs[lane];
        const int device_id = src.fields_.at(0).device_id;
        GpuCiphertextMeta meta = src.meta;
        meta.parms_id = destination_level_info.parms_id;
        meta.q_count = dst_q_count;
        meta.p_count = 0;
        meta.component_count = 2;
        meta.is_ntt_form = true;
        meta.scale = src.meta.scale / q_last_two;
        if (!(meta.scale > 0.0) || !std::isfinite(meta.scale))
        {
            throw std::invalid_argument(
                "GpuEvaluator::relinearize_rescale_x2_hybrid_pair: invalid result scale");
        }
        aliases[lane] = (dsts[lane] == &source_a || dsts[lane] == &source_b);
        if (aliases[lane])
        {
            locals[lane] = GpuCiphertextData::allocate_single_device_sharded(
                src.meta.degree, dst_q_count, 2, device_id,
                std::vector<GpuPolyShard>{dst_shard}, 0);
            locals[lane].meta = meta;
            targets[lane] = &locals[lane];
        }
        else
        {
            prepare_ciphertext_destination(
                *dsts[lane], nullptr, nullptr, meta, 2, device_id, ref_layout);
            targets[lane] = dsts[lane];
        }
    }

    auto src_a_view = source_a.make_const_view();
    auto src_b_view = source_b.make_const_view();
    auto dst_a_view = targets[0]->make_view();
    auto dst_b_view = targets[1]->make_view();
    const auto &relin_keys_view = relin_keys.const_view(source_a.meta.q_count);
    keyswitch_handler_.relinearize_hybrid_ciphertext_rescale_x2_pair(
        dst_a_view, dst_b_view, src_a_view, src_b_view,
        relin_keys_view, relin_keys, source_level_info, destination_level_info);

    for (int lane = 0; lane < 2; ++lane)
    {
        if (aliases[lane])
        {
            *dsts[lane] = std::move(locals[lane]);
        }
    }
}

namespace
{

bool prepare_binary_prefix_batch(
    const GpuParameterData &params,
    const std::vector<const GpuCiphertextData *> &lefts,
    const std::vector<const GpuCiphertextData *> &rights,
    bool allow_size3,
    std::size_t max_total_polys,
    std::size_t max_ciphertexts,
    const GpuLevelInfo *&target_level_out,
    GpuRNSPoly &dst_layout_out,
    int &device_id_out,
    std::vector<GpuConstCiphertextView> &lhs_views_out,
    std::vector<GpuConstCiphertextView> &rhs_views_out)
{
    const std::size_t count = lefts.size();
    if (count < 2 || count > max_ciphertexts || rights.size() != count)
    {
        return false;
    }
    const auto &ref_l = *lefts[0];
    const auto &ref_r = *rights[0];
    if (ref_l.empty() || ref_r.empty() || ref_l.fields_.empty() || ref_r.fields_.empty())
    {
        return false;
    }
    const std::size_t comp_count = ref_l.size();
    if ((comp_count != 2 && !(allow_size3 && comp_count == 3)) ||
        comp_count * count > max_total_polys ||
        !ref_l.meta.is_ntt_form || ref_l.meta.p_count != 0)
    {
        return false;
    }
    const std::size_t dst_q = std::min(ref_l.meta.q_count, ref_r.meta.q_count);
    const std::size_t degree = ref_l.meta.degree;
    const int device_id = ref_l.fields_[0].device_id;
    const parms_id_type target_parms_id =
        ref_l.meta.q_count == dst_q ? ref_l.meta.parms_id : ref_r.meta.parms_id;

    auto same_chain_prefix = [&](const GpuCiphertextData &ct) {
        if (ct.meta.q_count < dst_q) return false;
        if (ct.meta.parms_id == target_parms_id) return true;
        if (ct.meta.q_count == dst_q) return false;
        parms_id_type cur = ct.meta.parms_id;
        for (std::size_t s = 0; s < ct.meta.q_count - dst_q; ++s)
        {
            cur = params.get_next_level(cur).parms_id;
        }
        return cur == target_parms_id;
    };

    for (std::size_t i = 0; i < count; ++i)
    {
        const auto &l = *lefts[i];
        const auto &r = *rights[i];
        if (l.empty() || r.empty() || l.fields_.empty() || r.fields_.empty() ||
            !l.meta.is_ntt_form || !r.meta.is_ntt_form ||
            l.meta.p_count != 0 || r.meta.p_count != 0 ||
            l.meta.degree != degree || r.meta.degree != degree ||
            std::min(l.meta.q_count, r.meta.q_count) != dst_q ||
            l.size() != comp_count || r.size() != comp_count ||
            l.meta.component_count != comp_count || r.meta.component_count != comp_count ||
            l.polys_[0].shards.size() != 1 || r.polys_[0].shards.size() != 1 ||
            l.fields_[0].device_id != device_id || r.fields_[0].device_id != device_id ||
            !all_components_use_layout(l, l.polys_[0]) ||
            !all_components_use_layout(r, r.polys_[0]) ||
            l.polys_[0].shards[0].limb_begin != 0 || l.polys_[0].shards[0].limb_count < dst_q ||
            l.polys_[0].shards[0].coeff_begin != 0 || l.polys_[0].shards[0].coeff_count != degree ||
            r.polys_[0].shards[0].limb_begin != 0 || r.polys_[0].shards[0].limb_count < dst_q ||
            r.polys_[0].shards[0].coeff_begin != 0 || r.polys_[0].shards[0].coeff_count != degree ||
            !same_chain_prefix(l) || !same_chain_prefix(r))
        {
            return false;
        }
    }

    target_level_out = &params.get_level(target_parms_id);
    if (target_level_out->q_count != dst_q)
    {
        return false;
    }
    device_id_out = device_id;
    GpuPolyShard dst_shard;
    dst_shard.field_index = 0;
    dst_shard.field_offset = 0;
    dst_shard.limb_begin = 0;
    dst_shard.limb_count = dst_q;
    dst_shard.coeff_begin = 0;
    dst_shard.coeff_count = degree;
    dst_layout_out = GpuRNSPoly{};
    dst_layout_out.degree = degree;
    dst_layout_out.q_count = dst_q;
    dst_layout_out.p_count = 0;
    dst_layout_out.shards.push_back(dst_shard);

    auto make_narrowed = [&](const GpuCiphertextData &ct) {
        auto view = ct.make_const_view();
        view.meta.parms_id = target_parms_id;
        view.meta.q_count = dst_q;
        for (auto &poly : view.polys)
        {
            poly.shards[0].limb_count = dst_q;
        }
        return view;
    };
    lhs_views_out.clear();
    rhs_views_out.clear();
    lhs_views_out.reserve(count);
    rhs_views_out.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        lhs_views_out.push_back(make_narrowed(*lefts[i]));
        rhs_views_out.push_back(make_narrowed(*rights[i]));
    }
    return true;
}

}  // namespace

void GpuEvaluator::add_or_sub_batch_impl(
    const std::vector<const GpuCiphertextData *> &lefts,
    const std::vector<const GpuCiphertextData *> &rights,
    bool subtract,
    std::vector<GpuCiphertextData> &destinations) const
{
    const std::size_t count = lefts.size();
    if (rights.size() != count)
    {
        throw std::invalid_argument("GpuEvaluator::add_or_sub_batch: input size mismatch");
    }
    destinations.resize(count);
    if (count == 0)
    {
        return;
    }
    for (std::size_t i = 0; i < count; ++i)
    {
        if (lefts[i] == nullptr || rights[i] == nullptr)
        {
            throw std::invalid_argument("GpuEvaluator::add_or_sub_batch: null source");
        }
    }
    auto fallback = [&]() {
        for (std::size_t i = 0; i < count; ++i)
        {
            if (subtract) sub(*lefts[i], *rights[i], destinations[i]);
            else add(*lefts[i], *rights[i], destinations[i]);
        }
    };
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!same_scale(lefts[i]->meta.scale, rights[i]->meta.scale))
        {
            throw std::invalid_argument("GpuEvaluator::add_or_sub_batch: scale mismatch");
        }
    }

    const GpuLevelInfo *level_info_ptr = nullptr;
    GpuRNSPoly dst_layout;
    int device_id = 0;
    std::vector<GpuConstCiphertextView> lhs_views, rhs_views;
    if (!prepare_binary_prefix_batch(
            params_, lefts, rights, true,
            kernel::kMaxElementwisePairPtrs, kernel::kMaxElementwisePairPtrs / 2,
            level_info_ptr, dst_layout, device_id, lhs_views, rhs_views))
    {
        fallback();
        return;
    }

    const std::size_t comp_count = lefts[0]->size();
    std::vector<GpuCiphertextData> local_dsts;
    std::vector<GpuCiphertextData *> targets(count);
    bool any_alias = false;
    for (std::size_t i = 0; i < count && !any_alias; ++i)
    {
        for (std::size_t j = 0; j < count; ++j)
        {
            if (&destinations[i] == lefts[j] || &destinations[i] == rights[j])
            {
                any_alias = true;
                break;
            }
        }
    }
    if (any_alias) local_dsts.resize(count);
    std::vector<GpuCiphertextView> dst_views;
    dst_views.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        targets[i] = any_alias ? &local_dsts[i] : &destinations[i];
        GpuCiphertextMeta meta = lefts[i]->meta;
        meta.parms_id = level_info_ptr->parms_id;
        meta.q_count = level_info_ptr->q_count;
        meta.component_count = comp_count;
        meta.scale = std::max(lefts[i]->meta.scale, rights[i]->meta.scale);
        prepare_ciphertext_destination(
            *targets[i], lefts[i], rights[i], meta, comp_count, device_id, dst_layout);
        dst_views.push_back(targets[i]->make_view());
    }
    elementwise_handler_.add_or_sub_ciphertext_batch(
        dst_views, lhs_views, rhs_views, subtract, *level_info_ptr);
    if (any_alias)
    {
        for (std::size_t i = 0; i < count; ++i) destinations[i] = std::move(local_dsts[i]);
    }
}

void GpuEvaluator::add_batch(
    const std::vector<const GpuCiphertextData *> &lefts,
    const std::vector<const GpuCiphertextData *> &rights,
    std::vector<GpuCiphertextData> &destinations) const
{
    add_or_sub_batch_impl(lefts, rights, false, destinations);
}

void GpuEvaluator::sub_batch(
    const std::vector<const GpuCiphertextData *> &lefts,
    const std::vector<const GpuCiphertextData *> &rights,
    std::vector<GpuCiphertextData> &destinations) const
{
    add_or_sub_batch_impl(lefts, rights, true, destinations);
}

void GpuEvaluator::multiply_batch(
    const std::vector<const GpuCiphertextData *> &lefts,
    const std::vector<const GpuCiphertextData *> &rights,
    std::vector<GpuCiphertextData> &destinations) const
{
    const std::size_t count = lefts.size();
    if (rights.size() != count)
    {
        throw std::invalid_argument("GpuEvaluator::multiply_batch: input size mismatch");
    }
    destinations.resize(count);
    if (count == 0)
    {
        return;
    }
    for (std::size_t i = 0; i < count; ++i)
    {
        if (lefts[i] == nullptr || rights[i] == nullptr)
        {
            throw std::invalid_argument("GpuEvaluator::multiply_batch: null source");
        }
        const double out_scale = lefts[i]->meta.scale * rights[i]->meta.scale;
        if (!(out_scale > 0.0) || !std::isfinite(out_scale))
        {
            throw std::invalid_argument("GpuEvaluator::multiply_batch: invalid result scale");
        }
    }
    bool is_square = true;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (lefts[i] != rights[i])
        {
            is_square = false;
            break;
        }
    }
    auto fallback = [&]() {
        for (std::size_t i = 0; i < count; ++i)
        {
            if (lefts[i] == rights[i])
            {
                square(*lefts[i], destinations[i]);
            }
            else if (!(lefts[i]->meta.parms_id == rights[i]->meta.parms_id))
            {
                const std::size_t q = std::min(lefts[i]->meta.q_count, rights[i]->meta.q_count);
                multiply_q_prefix(
                    *lefts[i], *rights[i], destinations[i],
                    params_.get_level_by_q_count(q).parms_id);
            }
            else
            {
                multiply(*lefts[i], *rights[i], destinations[i]);
            }
        }
    };

    const GpuLevelInfo *level_info_ptr = nullptr;
    GpuRNSPoly dst_layout;
    int device_id = 0;
    std::vector<GpuConstCiphertextView> lhs_views, rhs_views;
    if (!prepare_binary_prefix_batch(
            params_, lefts, rights, false,
            kernel::kMaxElementwiseCiphertextBatch * 2, kernel::kMaxElementwiseCiphertextBatch,
            level_info_ptr, dst_layout, device_id, lhs_views, rhs_views))
    {
        fallback();
        return;
    }

    std::vector<GpuCiphertextData> local_dsts;
    std::vector<GpuCiphertextData *> targets(count);
    bool any_alias = false;
    for (std::size_t i = 0; i < count && !any_alias; ++i)
    {
        for (std::size_t j = 0; j < count; ++j)
        {
            if (&destinations[i] == lefts[j] || &destinations[i] == rights[j])
            {
                any_alias = true;
                break;
            }
        }
    }
    if (any_alias) local_dsts.resize(count);
    std::vector<GpuCiphertextView> dst_views;
    dst_views.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        targets[i] = any_alias ? &local_dsts[i] : &destinations[i];
        GpuCiphertextMeta meta = lefts[i]->meta;
        meta.parms_id = level_info_ptr->parms_id;
        meta.q_count = level_info_ptr->q_count;
        meta.component_count = 3;
        meta.scale = lefts[i]->meta.scale * rights[i]->meta.scale;
        prepare_ciphertext_destination(
            *targets[i], lefts[i], rights[i], meta, 3, device_id, dst_layout);
        dst_views.push_back(targets[i]->make_view());
    }
    elementwise_handler_.multiply_ciphertext_batch(
        dst_views, lhs_views, rhs_views, is_square, *level_info_ptr);
    if (any_alias)
    {
        for (std::size_t i = 0; i < count; ++i) destinations[i] = std::move(local_dsts[i]);
    }
}

void GpuEvaluator::square_batch(
    const std::vector<const GpuCiphertextData *> &sources,
    std::vector<GpuCiphertextData> &destinations) const
{
    multiply_batch(sources, sources, destinations);
}

void GpuEvaluator::multiply_plain_batch(
    const std::vector<const GpuCiphertextData *> &sources,
    const GpuPlaintextData &source_plaintext,
    std::vector<GpuCiphertextData> &destinations) const
{
    const std::size_t count = sources.size();
    destinations.resize(count);
    if (count == 0)
    {
        return;
    }
    for (const auto *src : sources)
    {
        if (src == nullptr)
        {
            throw std::invalid_argument("GpuEvaluator::multiply_plain_batch: null source");
        }
    }
    auto fallback = [&]() {
        for (std::size_t i = 0; i < count; ++i)
        {
            multiply_plain(*sources[i], source_plaintext, destinations[i]);
        }
    };
    const auto &ref = *sources[0];
    bool fast_ok =
        count >= 2 &&
        !ref.empty() && !ref.fields_.empty() &&
        !source_plaintext.empty() &&
        ref.size() == 2 && ref.meta.component_count == 2 &&
        ref.size() * count <= kernel::kMaxElementwisePairPtrs &&
        ref.meta.is_ntt_form && source_plaintext.meta.is_ntt_form &&
        ref.meta.p_count == 0 &&
        !(ref.meta.parms_id != source_plaintext.meta.parms_id) &&
        ref.polys_[0].shards.size() == 1 &&
        source_plaintext.poly_.shards.size() == 1;
    const int device_id = fast_ok ? ref.fields_[0].device_id : 0;
    for (std::size_t i = 0; fast_ok && i < count; ++i)
    {
        const auto &s = *sources[i];
        const double out_scale = s.meta.scale * source_plaintext.meta.scale;
        if (s.empty() || s.fields_.empty() ||
            !(s.meta.parms_id == ref.meta.parms_id) ||
            !s.meta.is_ntt_form || s.meta.p_count != 0 ||
            s.size() != 2 || s.meta.component_count != 2 ||
            !(out_scale > 0.0) || !std::isfinite(out_scale) ||
            s.polys_[0].shards.size() != 1 ||
            s.fields_[0].device_id != device_id ||
            !all_components_use_layout(s, s.polys_[0]))
        {
            fast_ok = false;
        }
    }
    if (!fast_ok)
    {
        fallback();
        return;
    }

    const auto &level_info = params_.get_level(ref.meta.parms_id);
    std::vector<GpuCiphertextData> local_dsts;
    std::vector<GpuCiphertextData *> targets(count);
    bool any_alias = false;
    for (std::size_t i = 0; i < count && !any_alias; ++i)
    {
        for (std::size_t j = 0; j < count; ++j)
        {
            if (&destinations[i] == sources[j])
            {
                any_alias = true;
                break;
            }
        }
    }
    if (any_alias) local_dsts.resize(count);
    std::vector<GpuConstCiphertextView> src_views;
    std::vector<GpuCiphertextView> dst_views;
    src_views.reserve(count);
    dst_views.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        targets[i] = any_alias ? &local_dsts[i] : &destinations[i];
        GpuCiphertextMeta meta = sources[i]->meta;
        meta.component_count = 2;
        meta.scale = sources[i]->meta.scale * source_plaintext.meta.scale;
        prepare_ciphertext_destination(
            *targets[i], sources[i], nullptr, meta, 2, device_id, sources[i]->polys_[0]);
        src_views.push_back(sources[i]->make_const_view());
        dst_views.push_back(targets[i]->make_view());
    }
    elementwise_handler_.multiply_plain_batch(
        dst_views, src_views, source_plaintext.make_const_view(), level_info);
    if (any_alias)
    {
        for (std::size_t i = 0; i < count; ++i) destinations[i] = std::move(local_dsts[i]);
    }
}

void GpuEvaluator::rescale_many_batch(
    const std::vector<const GpuCiphertextData *> &sources,
    std::vector<GpuCiphertextData> &destinations,
    std::uint32_t rescale_count) const
{
    const std::size_t count = sources.size();
    destinations.resize(count);
    if (count == 0)
    {
        return;
    }
    for (const auto *src : sources)
    {
        if (src == nullptr)
        {
            throw std::invalid_argument("GpuEvaluator::rescale_many_batch: null source");
        }
    }
    auto fallback = [&]() {
        for (std::size_t i = 0; i < count; ++i)
        {
            rescale_many(*sources[i], destinations[i], rescale_count);
        }
    };
    if (rescale_count == 0 || count < 2)
    {
        fallback();
        return;
    }
    const auto &ref = *sources[0];
    bool fast_ok =
        !ref.empty() && !ref.fields_.empty() &&
        ref.size() == 2 && ref.meta.component_count == 2 &&
        ref.size() * count <= kernel::kMaxInntPtrsBatch &&
        ref.size() * count <= kernel::kMaxRescaleBatchComponents &&
        ref.meta.is_ntt_form && ref.meta.p_count == 0 &&
        ref.meta.q_count > rescale_count &&
        ref.polys_[0].shards.size() == 1;
    const int device_id = fast_ok ? ref.fields_[0].device_id : 0;
    for (std::size_t i = 0; fast_ok && i < count; ++i)
    {
        const auto &s = *sources[i];
        if (s.empty() || s.fields_.empty() ||
            !(s.meta.parms_id == ref.meta.parms_id) ||
            !s.meta.is_ntt_form || s.meta.p_count != 0 ||
            s.size() != 2 || s.meta.component_count != 2 ||
            s.polys_[0].shards.size() != 1 ||
            s.fields_[0].device_id != device_id ||
            !all_components_use_layout(s, s.polys_[0]))
        {
            fast_ok = false;
        }
    }
    if (!fast_ok)
    {
        fallback();
        return;
    }

    const auto &source_level = params_.get_level(ref.meta.parms_id);
    const GpuLevelInfo *dest_level = &source_level;
    long double divisor = 1.0L;
    for (std::uint32_t step = 0; step < rescale_count; ++step)
    {
        if (dest_level->shards.empty() || dest_level->shards.front().q_last == 0)
        {
            fallback();
            return;
        }
        divisor *= static_cast<long double>(dest_level->shards.front().q_last);
        dest_level = &params_.get_next_level(dest_level->parms_id);
    }
    const std::size_t dst_q = ref.meta.q_count - rescale_count;
    GpuPolyShard dst_shard;
    dst_shard.field_index = 0;
    dst_shard.field_offset = 0;
    dst_shard.limb_begin = 0;
    dst_shard.limb_count = dst_q;
    dst_shard.coeff_begin = 0;
    dst_shard.coeff_count = ref.meta.degree;
    GpuRNSPoly dst_layout;
    dst_layout.degree = ref.meta.degree;
    dst_layout.q_count = dst_q;
    dst_layout.p_count = 0;
    dst_layout.shards.push_back(dst_shard);

    std::vector<GpuCiphertextData> local_dsts;
    std::vector<GpuCiphertextData *> targets(count);
    bool any_alias = false;
    for (std::size_t i = 0; i < count && !any_alias; ++i)
    {
        for (std::size_t j = 0; j < count; ++j)
        {
            if (&destinations[i] == sources[j])
            {
                any_alias = true;
                break;
            }
        }
    }
    if (any_alias) local_dsts.resize(count);
    std::vector<GpuConstCiphertextView> src_views;
    std::vector<GpuCiphertextView> dst_views;
    src_views.reserve(count);
    dst_views.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        targets[i] = any_alias ? &local_dsts[i] : &destinations[i];
        GpuCiphertextMeta meta = sources[i]->meta;
        meta.parms_id = dest_level->parms_id;
        meta.q_count = dst_q;
        meta.p_count = 0;
        meta.component_count = 2;
        meta.is_ntt_form = true;
        meta.scale = static_cast<double>(
            static_cast<long double>(sources[i]->meta.scale) / divisor);
        if (!(meta.scale > 0.0) || !std::isfinite(meta.scale))
        {
            throw std::invalid_argument("GpuEvaluator::rescale_many_batch: invalid result scale");
        }
        prepare_ciphertext_destination(
            *targets[i], sources[i], nullptr, meta, 2, device_id, dst_layout);
        src_views.push_back(sources[i]->make_const_view());
        dst_views.push_back(targets[i]->make_view());
    }
    if (!modswitch_handler_.rescale_ciphertext_many_batch(
            dst_views, src_views, source_level, *dest_level, rescale_count))
    {
        fallback();
        return;
    }
    if (any_alias)
    {
        for (std::size_t i = 0; i < count; ++i) destinations[i] = std::move(local_dsts[i]);
    }
}

void GpuEvaluator::relinearize_batch_impl(
    const std::vector<const GpuCiphertextData *> &sources,
    const GpuRelinKeysData &relin_keys_in,
    bool rescale_x2,
    std::vector<GpuCiphertextData> &destinations) const
{
    const std::size_t count = sources.size();
    destinations.resize(count);
    if (count == 0)
    {
        return;
    }
    for (const auto *src : sources)
    {
        if (src == nullptr)
        {
            throw std::invalid_argument("GpuEvaluator::relinearize_batch: null source");
        }
    }
    auto fallback = [&]() {
        for (std::size_t i = 0; i < count; ++i)
        {
            if (rescale_x2) relinearize_rescale_x2_hybrid(*sources[i], relin_keys_in, destinations[i]);
            else relinearize(*sources[i], relin_keys_in, destinations[i]);
        }
    };
    const auto &ref = *sources[0];
    if (count < 2 || count > kernel::kMaxRelinBatchSources ||
        ref.empty() || ref.fields_.empty())
    {
        fallback();
        return;
    }
    const char *op_name = rescale_x2
        ? "GpuEvaluator::relinearize_rescale_x2_hybrid_batch"
        : "GpuEvaluator::relinearize_batch";
    const auto &relin_keys = level_key_for(params_, relin_keys_in, ref.meta, op_name);
    const int device_id = ref.fields_[0].device_id;
    bool fast_ok =
        !relin_keys.empty() &&
        ref.size() == 3 && ref.meta.component_count == 3 &&
        ref.meta.is_ntt_form && ref.meta.p_count == 0 &&
        (!rescale_x2 || ref.meta.q_count >= 3) &&
        ref.polys_[0].shards.size() == 1;
    for (std::size_t i = 0; fast_ok && i < count; ++i)
    {
        const auto &s = *sources[i];
        if (s.empty() || s.fields_.empty() ||
            !(s.meta.parms_id == ref.meta.parms_id) ||
            !s.meta.is_ntt_form || s.meta.p_count != 0 ||
            s.size() != 3 || s.meta.component_count != 3 ||
            s.polys_[0].shards.size() != 1 ||
            s.fields_[0].device_id != device_id ||
            !all_components_use_layout(s, s.polys_[0]))
        {
            fast_ok = false;
        }
    }
    if (!fast_ok)
    {
        fallback();
        return;
    }

    const auto &source_level = params_.get_level(ref.meta.parms_id);
    const GpuLevelInfo *dest_level = &source_level;
    double q_last_two = 1.0;
    if (rescale_x2)
    {
        const auto &mid = params_.get_next_level(source_level.parms_id);
        dest_level = &params_.get_next_level(mid.parms_id);
        if (source_level.q_count != ref.meta.q_count ||
            dest_level->q_count + 2 != source_level.q_count ||
            source_level.shards.empty() ||
            source_level.shards.front().q_last_two_product == 0 ||
            count * 2 > kernel::kMaxInntPtrsBatch ||
            count * 2 > kernel::kMaxRescaleBatchComponents)
        {
            fallback();
            return;
        }
        q_last_two = static_cast<double>(source_level.shards.front().q_last_two_product);
    }

    static const bool persistent_scratch = [] {
        const char *raw = std::getenv("POSEIDON_GPU_PERSISTENT_ROTATE_WORKSPACE");
        return raw != nullptr && std::string(raw) == "1";
    }();
    GpuDoubleHoistWorkspace local_staged;
    std::unique_lock<std::mutex> scratch_lock;
    if (persistent_scratch)
    {
        scratch_lock = std::unique_lock<std::mutex>(rotate_scratch_->mutex);
    }
    GpuDoubleHoistWorkspace &staged =
        persistent_scratch ? rotate_scratch_->workspace : local_staged;

    std::vector<GpuConstCiphertextView> src_views;
    src_views.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        src_views.push_back(sources[i]->make_const_view());
    }
    const auto &relin_keys_view = relin_keys.const_view(ref.meta.q_count);
    if (!keyswitch_handler_.relinearize_hybrid_ciphertext_batch_to_buffer(
            src_views, relin_keys_view, relin_keys, source_level, staged))
    {
        fallback();
        return;
    }

    const std::size_t degree = ref.meta.degree;
    const std::size_t src_q = ref.meta.q_count;
    const std::size_t dst_q = rescale_x2 ? (src_q - 2) : src_q;
    const std::size_t src_q_words = src_q * degree;
    GpuPolyShard dst_shard;
    dst_shard.field_index = 0;
    dst_shard.field_offset = 0;
    dst_shard.limb_begin = 0;
    dst_shard.limb_count = dst_q;
    dst_shard.coeff_begin = 0;
    dst_shard.coeff_count = degree;
    GpuRNSPoly dst_layout;
    dst_layout.degree = degree;
    dst_layout.q_count = dst_q;
    dst_layout.p_count = 0;
    dst_layout.shards.push_back(dst_shard);

    std::vector<GpuCiphertextData> local_dsts;
    std::vector<GpuCiphertextData *> targets(count);
    bool cross_alias = false;
    for (std::size_t i = 0; i < count && !cross_alias; ++i)
    {
        for (std::size_t j = 0; j < count; ++j)
        {
            if (&destinations[i] == sources[j])
            {
                cross_alias = true;
                break;
            }
        }
    }
    if (cross_alias) local_dsts.resize(count);

    std::vector<GpuCiphertextView> dst_views;
    dst_views.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        targets[i] = cross_alias ? &local_dsts[i] : &destinations[i];
        GpuCiphertextMeta meta = sources[i]->meta;
        meta.parms_id = dest_level->parms_id;
        meta.q_count = dst_q;
        meta.p_count = 0;
        meta.component_count = 2;
        meta.is_ntt_form = true;
        if (rescale_x2)
        {
            meta.scale = sources[i]->meta.scale / q_last_two;
            if (!(meta.scale > 0.0) || !std::isfinite(meta.scale))
            {
                throw std::invalid_argument(
                    "GpuEvaluator::relinearize_rescale_x2_hybrid_batch: invalid result scale");
            }
        }
        prepare_ciphertext_destination(
            *targets[i], nullptr, nullptr, meta, 2, device_id, dst_layout);
        dst_views.push_back(targets[i]->make_view());
    }

    if (!rescale_x2)
    {
        const auto stream = gpu_execution_stream();
        gpu_check_cuda(cudaSetDevice(device_id), "relinearize_batch cudaSetDevice");
        for (std::size_t i = 0; i < count; ++i)
        {
            GpuWord *dst0 = dst_views[i].polys[0].shards.front().ptr;
            GpuWord *dst1 = dst_views[i].polys[1].shards.front().ptr;
            const GpuWord *src0 =
                staged.inner_q_batch.q.data() + (i * 2) * src_q_words;
            if (dst1 == dst0 + src_q_words)
            {
                gpu_check_cuda(
                    cudaMemcpyAsync(
                        dst0, src0, 2 * src_q_words * sizeof(GpuWord),
                        cudaMemcpyDeviceToDevice, stream),
                    "relinearize_batch d2d");
            }
            else
            {
                gpu_check_cuda(
                    cudaMemcpyAsync(
                        dst0, src0, src_q_words * sizeof(GpuWord),
                        cudaMemcpyDeviceToDevice, stream),
                    "relinearize_batch d2d c0");
                gpu_check_cuda(
                    cudaMemcpyAsync(
                        dst1, src0 + src_q_words, src_q_words * sizeof(GpuWord),
                        cudaMemcpyDeviceToDevice, stream),
                    "relinearize_batch d2d c1");
            }
        }
    }
    else
    {
        std::vector<GpuConstCiphertextView> relin_views(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            auto &v = relin_views[i];
            v.meta = sources[i]->meta;
            v.meta.component_count = 2;
            v.meta.is_ntt_form = true;
            v.polys.resize(2);
            for (std::size_t c = 0; c < 2; ++c)
            {
                v.polys[c].shards.push_back(GpuConstPolyShardView{
                    device_id,
                    staged.inner_q_batch.q.data() + (i * 2 + c) * src_q_words,
                    0, src_q, 0, degree});
            }
        }
        if (!modswitch_handler_.rescale_ciphertext_many_batch(
                dst_views, relin_views, source_level, *dest_level, 2))
        {
            fallback();
            return;
        }
    }

    if (cross_alias)
    {
        for (std::size_t i = 0; i < count; ++i) destinations[i] = std::move(local_dsts[i]);
    }
}

void GpuEvaluator::relinearize_batch(
    const std::vector<const GpuCiphertextData *> &sources,
    const GpuRelinKeysData &relin_keys,
    std::vector<GpuCiphertextData> &destinations) const
{
    relinearize_batch_impl(sources, relin_keys, false, destinations);
}

void GpuEvaluator::relinearize_rescale_x2_hybrid_batch(
    const std::vector<const GpuCiphertextData *> &sources,
    const GpuRelinKeysData &relin_keys,
    std::vector<GpuCiphertextData> &destinations) const
{
    relinearize_batch_impl(sources, relin_keys, true, destinations);
}
static void validate_rotate_input(const GpuCiphertextData &source_ciphertext)
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::rotate",
        source_ciphertext,
        true);
    if (source_ciphertext.size() != 2)
    {
        throw std::invalid_argument(
            "GpuEvaluator::rotate: first implementation expects a size-2 ciphertext");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::rotate: input ciphertext p limbs are not supported");
    }
    if (source_ciphertext.polys_.at(0).shards.size() != 1)
    {
        throw std::invalid_argument(
            "GpuEvaluator::rotate: first implementation requires one full shard");
    }

}

void GpuEvaluator::rotate(
    const GpuCiphertextData &source_ciphertext,
    int step,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &destination_ciphertext) const
{
    rotate(source_ciphertext, step, galois_keys, destination_ciphertext, nullptr, 0);
}

void GpuEvaluator::rotate(
    const GpuCiphertextData &source_ciphertext,
    int step,
    const GpuGaloisKeysData &galois_keys_in,
    GpuCiphertextData &destination_ciphertext,
    GpuRotationHoistCache *hoist_cache,
    std::uintptr_t source_tag) const
{
    const auto &galois_keys = level_key_for(params_, galois_keys_in, source_ciphertext.meta, "GpuEvaluator::rotate");
    validate_rotate_input(source_ciphertext);

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);
    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            2,
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);

    result.meta = source_ciphertext.meta;
    result.meta.component_count = 2;
    result.meta.is_ntt_form = true;

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();

    /*step为0说明没有旋转，直接保持不变*/
    if (step == 0)
    {
        copy_poly(
            destination_view.polys[0],
            source_view.polys[0],
            "GpuEvaluator::rotate copy c0");
        copy_poly(
            destination_view.polys[1],
            source_view.polys[1],
            "GpuEvaluator::rotate copy c1");
        destination_ciphertext = std::move(result);
        return;
    }

    if (galois_keys.empty())
    {
        throw std::invalid_argument("GpuEvaluator::rotate: empty galois keys");
    }

    /*step不为0的情况，step表示左旋转位数，galois_elt_from_rotation_step负责把step转换成密文的Galois element形态。*/
    const std::uint32_t galois_elt = galois_elt_from_rotation_step(source_ciphertext.meta.degree, step);
    /*选择密钥，因为不同的step对应不同的旋转密钥，所以galois_key_index选择对应的高斯密钥*/
    const std::size_t key_index = galois_key_index(galois_elt);

    if (galois_keys.meta.galois_format ==
        GpuGaloisKeyFormat::InversePreRotated)
    {
        static const bool persistent_scratch = [] {
            const char *raw = std::getenv("POSEIDON_GPU_PERSISTENT_ROTATE_WORKSPACE");
            return raw != nullptr && std::string(raw) == "1";
        }();
        GpuDoubleHoistWorkspace local_staged;
        std::unique_lock<std::mutex> scratch_lock;
        if (persistent_scratch)
        {
            scratch_lock = std::unique_lock<std::mutex>(rotate_scratch_->mutex);
        }
        GpuDoubleHoistWorkspace &staged =
            persistent_scratch ? rotate_scratch_->workspace : local_staged;
        staged.outer_accumulator.ensure_capacity(
            device_id,
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            galois_keys.meta.p_count,
            1);
        // Runtime hoisting: the ModUp of c1 depends only on the source, so
        // rotations of the same source share one decomposition.
        GpuHoistedDecomposition *hoisted = &staged.source_hoist;
        if (hoist_cache != nullptr && source_tag != 0)
        {
            hoisted = &hoist_cache->decomposition;
            if (hoist_cache->source_tag != source_tag ||
                hoisted->parms_id != source_ciphertext.meta.parms_id ||
                hoisted->q_count != source_ciphertext.meta.q_count ||
                hoisted->device_id != device_id)
            {
                hoist_cache->source_tag = 0;
                keyswitch_handler_.hoist_decompose_modup_ntt(
                    source_view.polys[1],
                    level_info,
                    *hoisted,
                    staged.keyswitch);
                hoist_cache->source_tag = source_tag;
                ++hoist_cache->misses;
            }
            else
            {
                ++hoist_cache->hits;
            }
        }
        else
        {
            keyswitch_handler_.hoist_decompose_modup_ntt(
                source_view.polys[1],
                level_info,
                staged.source_hoist,
                staged.keyswitch);
        }
        const auto &key_view =
            galois_keys.const_view(source_ciphertext.meta.q_count);
        keyswitch_handler_.keyswitch_multsum_no_moddown(
            *hoisted,
            galois_elt,
            key_view,
            galois_keys,
            key_index,
            staged.outer_accumulator,
            0,
            true,
            level_info,
            staged.keyswitch);
        const GpuParameterShard *parameter_shard = nullptr;
        for (const auto &candidate : level_info.shards)
        {
            if (candidate.device_id == device_id &&
                candidate.hybrid_base_q_count ==
                    source_ciphertext.meta.q_count)
            {
                parameter_shard = &candidate;
                break;
            }
        }
        if (parameter_shard == nullptr)
        {
            throw std::invalid_argument(
                "GpuEvaluator::rotate: HYBRID parameter shard is absent");
        }
        kernel::launch_double_hoist_add_lifted_galois_c0(
            staged.outer_accumulator.q_component(0, 0),
            source_view.polys[0].shards.front().ptr,
            galois_elt,
            *parameter_shard,
            source_ciphertext.meta.degree);
        keyswitch_handler_.moddown_qp_ciphertext_to_q(
            staged.outer_accumulator,
            0,
            result,
            source_ciphertext.meta,
            level_info,
            staged.keyswitch);
        destination_ciphertext = std::move(result);
        return;
    }

    GpuCiphertextData rotated_c1 =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            1,
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);
    rotated_c1.meta = source_ciphertext.meta;
    rotated_c1.meta.component_count = 1;
    rotated_c1.meta.is_ntt_form = true;

    auto rotated_c1_view = rotated_c1.make_view();

    kernel::launch_apply_galois_ntt_poly_shard(
        destination_view.polys[0].shards.front(),
        source_view.polys[0].shards.front(),
        galois_elt,
        source_ciphertext.meta.degree);
    kernel::launch_apply_galois_ntt_poly_shard(
        rotated_c1_view.polys[0].shards.front(),
        source_view.polys[1].shards.front(),
        galois_elt,
        source_ciphertext.meta.degree);

    /* c1 先清零，后续 switch-key 会把 rotated_c1 * galois_key 累加进去。 */
    zero_poly(destination_view.polys[1], "GpuEvaluator::rotate zero c1");

    auto rotated_c1_const_view = rotated_c1.make_const_view();
    const auto &galois_keys_view =
        galois_keys.const_view(source_ciphertext.meta.q_count);
    keyswitch_handler_.switch_key_hybrid_ciphertext(
        destination_view,
        rotated_c1_const_view.polys[0],
        galois_keys_view,
        galois_keys,
        key_index,
        level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::switch_key(
    const GpuCiphertextData &source_ciphertext,
    const GpuEvaluationKeyData &switch_key,
    GpuCiphertextData &destination_ciphertext) const
{
    validate_rotate_input(source_ciphertext);
    // Resolve once and use the level-specific owner for both validation and
    // the storage-aware key view (a dispatch-only parent may itself be empty).
    const auto &selected_key = level_key_for(
        params_, switch_key, source_ciphertext.meta, "GpuEvaluator::switch_key");
    if (selected_key.empty())
    {
        throw std::invalid_argument("GpuEvaluator::switch_key: empty key");
    }
    NvtxRange range("SwitchKey");
    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);
    if (source_ciphertext.meta.p_count != 0 ||
        selected_key.meta.degree != source_ciphertext.meta.degree ||
        selected_key.meta.q_count < source_ciphertext.meta.q_count ||
        selected_key.meta.component_count != 2 || selected_key.meta.key_count != 1 ||
        selected_key.meta.galois_format != GpuGaloisKeyFormat::Standard ||
        level_info.shards.empty() || selected_key.meta.p_count == 0 ||
        selected_key.meta.p_count != level_info.shards.front().hybrid_base_p_count ||
        std::any_of(selected_key.fields_.begin(), selected_key.fields_.end(),
                    [device_id](const auto &field) { return field.device_id != device_id; }))
        throw std::invalid_argument("GpuEvaluator::switch_key: incompatible HYBRID key layout");
    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            2,
            device_id,
            source_ciphertext.polys_.at(0).shards,
            source_ciphertext.meta.p_count);
    result.meta = source_ciphertext.meta;
    result.meta.component_count = 2;
    result.meta.is_ntt_form = true;

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();
    copy_poly(destination_view.polys[0], source_view.polys[0],
              "GpuEvaluator::switch_key copy c0");
    zero_poly(destination_view.polys[1], "GpuEvaluator::switch_key zero c1");
    keyswitch_handler_.switch_key_hybrid_ciphertext(
        destination_view,
        source_view.polys[1],
        selected_key.const_view(source_ciphertext.meta.q_count),
        selected_key,
        0,
        level_info);
    destination_ciphertext = std::move(result);
}

namespace
{

void collect_and_hoist_giant_groups(
    const GpuKeySwitchHandler &keyswitch_handler,
    const GpuLevelInfo &level_info,
    GpuDoubleHoistWorkspace &workspace,
    int device_id,
    std::size_t degree,
    std::size_t q_count,
    const std::vector<int> &giant_steps,
    const GpuWord *inner_q_batch,
    std::size_t group_stride_words,
    bool batched_giant_intt,
    std::vector<const GpuWord *> &digit_q_ptrs,
    std::vector<const GpuWord *> &digit_p_ptrs,
    std::vector<std::uint32_t> &group_indices,
    std::vector<std::uint32_t> &galois_elts,
    std::vector<std::uint32_t> &key_indices,
    std::size_t *decompose_count = nullptr)
{
    const std::size_t group_count = giant_steps.size();
    const std::size_t q_words = q_count * degree;
    if (workspace.outer_group_hoists.size() < group_count)
    {
        workspace.outer_group_hoists.resize(group_count);
    }
    digit_q_ptrs.clear();
    digit_p_ptrs.clear();
    group_indices.clear();
    galois_elts.clear();
    key_indices.clear();
    digit_q_ptrs.reserve(group_count);
    digit_p_ptrs.reserve(group_count);
    group_indices.reserve(group_count);
    galois_elts.reserve(group_count);
    key_indices.reserve(group_count);

    std::vector<GpuConstRNSPolyView> switch_polys_ntt;
    std::vector<GpuHoistedDecomposition *> destinations;
    switch_polys_ntt.reserve(group_count);
    destinations.reserve(group_count);

    for (std::size_t group = 0; group < group_count; ++group)
    {
        const int giant_step = giant_steps[group];
        if (giant_step == 0)
        {
            continue;
        }
        GpuConstRNSPolyView inner_c1;
        inner_c1.poly_id = 1;
        inner_c1.shards.push_back(GpuConstPolyShardView{
            device_id,
            inner_q_batch + group * group_stride_words + q_words,
            0,
            q_count,
            0,
            degree});
        switch_polys_ntt.push_back(std::move(inner_c1));
        destinations.push_back(&workspace.outer_group_hoists[group]);

        const std::uint32_t galois_elt =
            galois_elt_from_rotation_step(degree, giant_step);
        const std::size_t key_index = galois_key_index(galois_elt);
        if (key_index > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::overflow_error(
                "double-hoist giant key index exceeds uint32_t");
        }
        group_indices.push_back(static_cast<std::uint32_t>(group));
        galois_elts.push_back(galois_elt);
        key_indices.push_back(static_cast<std::uint32_t>(key_index));
    }

    if (switch_polys_ntt.empty())
    {
        return;
    }

    keyswitch_handler.hoist_decompose_modup_ntt_batch(
        switch_polys_ntt,
        level_info,
        destinations,
        workspace.keyswitch,
        batched_giant_intt ? workspace.outer_source_intt_q_batch.data() : nullptr,
        batched_giant_intt ? q_words : 0);

    for (const GpuHoistedDecomposition *hoist : destinations)
    {
        digit_q_ptrs.push_back(hoist->digits_q_ntt.data());
        digit_p_ptrs.push_back(hoist->digits_p_ntt.data());
    }
    if (decompose_count != nullptr)
    {
        *decompose_count += destinations.size();
    }
}

bool launch_hoisted_keymul_many_if_eligible(
    const GpuHoistedDecomposition &source_hoist,
    const std::vector<int> &steps,
    const GpuGaloisKeysData &galois_keys,
    const GpuParameterShard &parameter_shard,
    const GpuWord *source_c0,
    const GpuWord *source_c1,
    bool fuse_c0,
    GpuQPCiphertextBuffer &destination)
{
    if (galois_keys.meta.galois_format !=
            GpuGaloisKeyFormat::InversePreRotated ||
        galois_keys.galois_key_q0_ptrs.empty() ||
        galois_keys.galois_key_p0_ptrs.empty() ||
        galois_keys.galois_key_q1_ptrs.empty() ||
        galois_keys.galois_key_p1_ptrs.empty())
    {
        return false;
    }

    const std::size_t degree = source_hoist.degree;
    const std::size_t q_count = parameter_shard.hybrid_base_q_count;
    const std::size_t p_count = parameter_shard.hybrid_base_p_count;
    const std::size_t dnum = source_hoist.dnum;
    const std::size_t storage_dnum = galois_keys.meta.decomposition_count;
    if (dnum == 0 || storage_dnum < dnum)
    {
        return false;
    }

    const std::size_t q_words_per_batch = 2 * q_count * degree;
    const std::size_t p_words_per_batch = 2 * p_count * degree;
    for (std::size_t offset = 0;
         offset < steps.size();
         offset += kernel::kMaxDoubleHoistKeymulManySteps)
    {
        const std::size_t chunk = std::min(
            kernel::kMaxDoubleHoistKeymulManySteps,
            steps.size() - offset);
        kernel::DoubleHoistKeymulManyArgs args{};
        for (std::size_t s = 0; s < chunk; ++s)
        {
            const int step = steps[offset + s];
            if (step == 0)
            {
                args.galois_elts[s] = 0;
                args.key_indices[s] = 0;
                continue;
            }
            const std::uint32_t galois_elt =
                galois_elt_from_rotation_step(degree, step);
            const std::size_t key_index = galois_key_index(galois_elt);
            if (key_index > std::numeric_limits<std::uint32_t>::max() ||
                galois_keys.galois_elts_by_key_index.size() <= key_index ||
                galois_keys.galois_elts_by_key_index[key_index] != galois_elt ||
                (key_index + 1) * storage_dnum >
                    galois_keys.galois_key_q0_ptrs.size())
            {
                throw std::invalid_argument(
                    "launch_hoisted_keymul_many_if_eligible: Galois key mismatch");
            }
            args.galois_elts[s] = galois_elt;
            args.key_indices[s] = static_cast<std::uint32_t>(key_index);
        }
        kernel::launch_double_hoist_pre_rotated_keymul_many(
            destination.q.data() + offset * q_words_per_batch,
            destination.p.data() + offset * p_words_per_batch,
            source_hoist.digits_q_ntt.data(),
            source_hoist.digits_p_ntt.data(),
            fuse_c0 ? source_c0 : nullptr,
            source_c1,
            galois_keys.galois_key_q0_ptrs.data(),
            galois_keys.galois_key_p0_ptrs.data(),
            galois_keys.galois_key_q1_ptrs.data(),
            galois_keys.galois_key_p1_ptrs.data(),
            args,
            chunk,
            dnum,
            storage_dnum,
            parameter_shard,
            degree);
    }
    return true;
}

bool is_workspace_hoist(
    const GpuHoistedDecomposition &hoisted,
    const GpuDoubleHoistWorkspace &workspace)
{
    if (&hoisted == &workspace.source_hoist || &hoisted == &workspace.outer_hoist)
        return true;
    for (const auto &group : workspace.outer_group_hoists)
        if (&hoisted == &group) return true;
    return false;
}

const GpuParameterShard &validate_split_rotation_input(
    const char *name,
    const GpuCiphertextData &source,
    const GpuLevelInfo &level,
    const GpuGaloisKeysData &keys)
{
    validate_ntt_ciphertext_input(name, source, true);
    if (source.size() != 2 || source.meta.p_count != 0 ||
        source.meta.degree != level.degree || source.meta.q_count != level.q_count ||
        source.polys_.front().shards.size() != 1)
    {
        throw std::invalid_argument(std::string(name) + ": requires one-shard, size-2, Q-only input at this level");
    }
    const auto &first = source.polys_.front().shards.front();
    const int device_id = source.fields_.at(first.field_index).device_id;
    for (const auto &poly : source.polys_)
    {
        const auto &shard = poly.shards.front();
        if (poly.degree != source.meta.degree || poly.q_count != source.meta.q_count ||
            poly.p_count != 0 || shard.limb_begin != 0 ||
            shard.limb_count != source.meta.q_count || shard.coeff_begin != 0 ||
            shard.coeff_count != source.meta.degree ||
            source.fields_.at(shard.field_index).device_id != device_id)
        {
            throw std::invalid_argument(std::string(name) + ": requires full component shards on the same device");
        }
    }
    const GpuParameterShard *parameter_shard = nullptr;
    for (const auto &candidate : level.shards)
    {
        if (candidate.device_id == device_id &&
            candidate.hybrid_base_q_count == source.meta.q_count)
        {
            parameter_shard = &candidate;
            break;
        }
    }
    if (parameter_shard == nullptr || parameter_shard->hybrid_base_p_count == 0)
    {
        throw std::invalid_argument(std::string(name) + ": HYBRID parameter shard is absent");
    }
    const auto p_count = parameter_shard->hybrid_base_p_count;
    const auto dnum = (source.meta.q_count + p_count - 1) / p_count;
    if (parameter_shard->hybrid_decomp_count != dnum || keys.empty() ||
        keys.meta.degree != source.meta.degree || keys.meta.q_count < source.meta.q_count ||
        keys.meta.p_count != p_count || keys.meta.decomposition_count < dnum ||
        keys.meta.component_count != 2 || keys.fields_.empty() ||
        (keys.meta.galois_format != GpuGaloisKeyFormat::Standard &&
         keys.meta.galois_format != GpuGaloisKeyFormat::InversePreRotated))
    {
        throw std::invalid_argument(std::string(name) + ": incompatible Galois key/level metadata");
    }
    for (const auto &field : keys.fields_)
    {
        if (field.device_id != device_id)
        {
            throw std::invalid_argument(std::string(name) + ": Galois key device mismatch");
        }
    }
    if (keys.meta.galois_format == GpuGaloisKeyFormat::InversePreRotated)
    {
        for (const auto *table : {&keys.galois_key_q0_ptrs, &keys.galois_key_p0_ptrs,
                                  &keys.galois_key_q1_ptrs, &keys.galois_key_p1_ptrs})
        {
            if (table->empty() || table->device_id() != device_id)
            {
                throw std::invalid_argument(std::string(name) + ": pre-rotated key table/device mismatch");
            }
        }
    }
    return *parameter_shard;
}

}  // namespace

void GpuEvaluator::rotation_hoist(
    const GpuCiphertextData &source,
    const GpuGaloisKeysData &keys,
    GpuHoistedDecomposition &destination,
    GpuDoubleHoistWorkspace &workspace) const
{
    const char *name = "GpuEvaluator::rotation_hoist";
    if (is_workspace_hoist(destination, workspace))
    {
        throw std::invalid_argument(std::string(name) + ": result must be owned independently of workspace scratch");
    }
    const auto &selected_keys = level_key_for(params_, keys, source.meta, name);
    const auto &level = params_.get_level(source.meta.parms_id);
    (void)validate_split_rotation_input(name, source, level, selected_keys);
    const auto source_view = source.make_const_view();
    // Materialize/validate the owning-storage-aware view before GPU work.
    (void)selected_keys.const_view(source.meta.q_count);
    // Invalidate any old binding before refilling, including on failure.
    destination.rotation_source_c0 = nullptr;
    destination.rotation_source_c1 = nullptr;
    keyswitch_handler_.hoist_decompose_modup_ntt(
        source_view.polys[1], level, destination, workspace.keyswitch);
    destination.rotation_source_meta = source.meta;
    destination.rotation_key_parms_id = selected_keys.meta.key_parms_id;
    destination.rotation_key_format = selected_keys.meta.galois_format;
    destination.rotation_source_c0 = source_view.polys[0].shards.front().ptr;
    destination.rotation_source_c1 = source_view.polys[1].shards.front().ptr;
}

void GpuEvaluator::rotate_from_hoist(
    const GpuCiphertextData &source,
    const GpuHoistedDecomposition &hoisted,
    int step,
    const GpuGaloisKeysData &keys,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &destination) const
{
    const char *name = "GpuEvaluator::rotate_from_hoist";
    if (step == 0 || &source == &destination || is_workspace_hoist(hoisted, workspace))
    {
        throw std::invalid_argument(std::string(name) + ": zero step, in-place destination or scratch-owned hoist is not supported");
    }
    const auto &selected_keys = level_key_for(params_, keys, source.meta, name);
    const auto &level = params_.get_level(source.meta.parms_id);
    const auto &parameter_shard = validate_split_rotation_input(name, source, level, selected_keys);
    const auto source_view = source.make_const_view();
    const auto &source0 = source_view.polys[0].shards.front();
    const auto p_count = parameter_shard.hybrid_base_p_count;
    const auto dnum = (source.meta.q_count + p_count - 1) / p_count;
    const auto &captured = hoisted.rotation_source_meta;
    if (hoisted.parms_id != source.meta.parms_id || hoisted.degree != source.meta.degree ||
        hoisted.q_count != source.meta.q_count || hoisted.p_count != p_count ||
        hoisted.dnum != dnum || hoisted.device_id != source0.device_id ||
        hoisted.rotation_source_c0 != source0.ptr ||
        hoisted.rotation_source_c1 != source_view.polys[1].shards.front().ptr ||
        captured.parms_id != source.meta.parms_id || captured.degree != source.meta.degree ||
        captured.q_count != source.meta.q_count || captured.p_count != source.meta.p_count ||
        captured.component_count != source.meta.component_count ||
        captured.is_ntt_form != source.meta.is_ntt_form || captured.scale != source.meta.scale ||
        captured.correction_factor != source.meta.correction_factor ||
        hoisted.rotation_key_parms_id != selected_keys.meta.key_parms_id ||
        hoisted.rotation_key_format != selected_keys.meta.galois_format)
    {
        throw std::invalid_argument(std::string(name) + ": hoisted source/key binding mismatch");
    }
    const auto q_words = checked_mul(source.meta.q_count, source.meta.degree, name);
    const auto p_words = checked_mul(p_count, source.meta.degree, name);
    if (hoisted.source_intt_q.data() == nullptr || hoisted.source_intt_q.size() < q_words ||
        hoisted.digits_q_ntt.data() == nullptr ||
        hoisted.digits_q_ntt.size() < checked_mul(dnum, q_words, name) ||
        hoisted.digits_p_ntt.data() == nullptr ||
        hoisted.digits_p_ntt.size() < checked_mul(dnum, p_words, name) ||
        hoisted.source_intt_q.device_id() != source0.device_id ||
        hoisted.digits_q_ntt.device_id() != source0.device_id ||
        hoisted.digits_p_ntt.device_id() != source0.device_id)
    {
        throw std::invalid_argument(std::string(name) + ": hoisted allocation/device mismatch");
    }
    const auto &key_view = selected_keys.const_view(source.meta.q_count);
    const auto galois_elt = galois_elt_from_rotation_step(source.meta.degree, step);
    const auto key_index = galois_key_index(galois_elt);
    workspace.outer_accumulator.ensure_capacity(
        source0.device_id, source.meta.degree, source.meta.q_count, p_count, 1);
    // Keep the reference suffix's fused KeyMult/Galois path and c0 ordering.
    // In particular, never refill a hoist through the suffix workspace.
    keyswitch_handler_.keyswitch_multsum_no_moddown(
        hoisted, galois_elt, key_view, selected_keys, key_index,
        workspace.outer_accumulator, 0, true, level, workspace.keyswitch);
    kernel::launch_double_hoist_add_lifted_galois_c0(
        workspace.outer_accumulator.q_component(0, 0), source0.ptr, galois_elt,
        parameter_shard, source.meta.degree);
    keyswitch_handler_.moddown_qp_ciphertext_to_q(
        workspace.outer_accumulator, 0, destination, source.meta, level, workspace.keyswitch);
}

namespace
{

const GpuParameterShard &rotate_many_hoisted_qp_impl(
    const char *caller,
    const GpuParameterData &params,
    const GpuKeySwitchHandler &keyswitch_handler,
    const GpuCiphertextData &source_ciphertext,
    const std::vector<int> &steps,
    const GpuGaloisKeysData &galois_keys_in,
    GpuDoubleHoistWorkspace &workspace,
    const GpuLevelInfo *&level_info_out)
{
    const auto &galois_keys =
        level_key_for(params, galois_keys_in, source_ciphertext.meta, caller);
    if (source_ciphertext.empty() || source_ciphertext.fields_.empty() ||
        !source_ciphertext.meta.is_ntt_form || source_ciphertext.size() != 2 ||
        source_ciphertext.meta.p_count != 0 ||
        source_ciphertext.polys_.at(0).shards.size() != 1)
    {
        throw std::invalid_argument(
            std::string(caller) + ": requires one-shard, size-2, Q-only NTT input");
    }
    if (galois_keys.empty())
    {
        throw std::invalid_argument(
            std::string(caller) + ": empty Galois keys");
    }
    const auto &level_info = params.get_level(source_ciphertext.meta.parms_id);
    const auto source_view = source_ciphertext.make_const_view();
    const auto &source0 = source_view.polys[0].shards.front();
    const GpuParameterShard *parameter_shard = nullptr;
    for (const auto &candidate : level_info.shards)
    {
        if (candidate.device_id == source0.device_id &&
            candidate.hybrid_base_q_count == source_ciphertext.meta.q_count)
        {
            parameter_shard = &candidate;
            break;
        }
    }
    if (parameter_shard == nullptr ||
        parameter_shard->hybrid_base_p_count == 0)
    {
        throw std::invalid_argument(
            std::string(caller) + ": HYBRID parameter shard is absent");
    }
    for (const int step : steps)
    {
        if (step == 0)
        {
            throw std::invalid_argument(
                std::string(caller) + ": zero step is not supported");
        }
    }

    workspace.outer_accumulator.ensure_capacity(
        source0.device_id,
        source_ciphertext.meta.degree,
        source_ciphertext.meta.q_count,
        parameter_shard->hybrid_base_p_count,
        steps.size());
    keyswitch_handler.hoist_decompose_modup_ntt(
        source_view.polys[1],
        level_info,
        workspace.source_hoist,
        workspace.keyswitch);
    const auto &key_view =
        galois_keys.const_view(source_ciphertext.meta.q_count);

    const bool use_fused_keymul_many =
        steps.size() >= 2 &&
        use_double_hoist_fused_baby_keyswitch_c0() &&
        launch_hoisted_keymul_many_if_eligible(
            workspace.source_hoist,
            steps,
            galois_keys,
            *parameter_shard,
            source0.ptr,
            source_view.polys[1].shards.front().ptr,
            true,
            workspace.outer_accumulator);
    if (!use_fused_keymul_many)
    {
        for (std::size_t index = 0; index < steps.size(); ++index)
        {
            const std::uint32_t galois_elt =
                galois_elt_from_rotation_step(
                    source_ciphertext.meta.degree, steps[index]);
            const std::size_t key_index = galois_key_index(galois_elt);
            keyswitch_handler.keyswitch_multsum_no_moddown(
                workspace.source_hoist,
                galois_elt,
                key_view,
                galois_keys,
                key_index,
                workspace.outer_accumulator,
                index,
                true,
                level_info,
                workspace.keyswitch);
            kernel::launch_double_hoist_add_lifted_galois_c0(
                workspace.outer_accumulator.q_component(index, 0),
                source0.ptr,
                galois_elt,
                *parameter_shard,
                source_ciphertext.meta.degree);
        }
    }
    level_info_out = &level_info;
    return *parameter_shard;
}

}  // namespace

void GpuEvaluator::rotate_many_hoisted(
    const GpuCiphertextData &source_ciphertext,
    const std::vector<int> &steps,
    const GpuGaloisKeysData &galois_keys_in,
    GpuDoubleHoistWorkspace &workspace,
    std::vector<GpuCiphertextData> &destination_ciphertexts) const
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::rotate_many_hoisted",
        source_ciphertext,
        true);
    if (steps.empty())
    {
        destination_ciphertexts.clear();
        return;
    }
    const GpuLevelInfo *level_info_ptr = nullptr;
    (void)rotate_many_hoisted_qp_impl(
        "GpuEvaluator::rotate_many_hoisted",
        params_,
        keyswitch_handler_,
        source_ciphertext,
        steps,
        galois_keys_in,
        workspace,
        level_info_ptr);
    const auto &level_info = *level_info_ptr;
    const int device_id = source_ciphertext.fields_.front().device_id;

    destination_ciphertexts.resize(steps.size());
    if (steps.size() >= 2)
    {
        keyswitch_handler_.moddown_qp_ciphertext_batch_to_q(
            workspace.outer_accumulator,
            steps.size(),
            workspace.inner_q_batch,
            level_info,
            workspace.batch_p_coeff,
            workspace.batch_converted_q);
        const std::size_t q_words =
            source_ciphertext.meta.q_count * source_ciphertext.meta.degree;
        const auto stream = gpu_execution_stream();
        gpu_check_cuda(
            cudaSetDevice(device_id),
            "GpuEvaluator::rotate_many_hoisted batch cudaSetDevice");
        GpuRNSPoly layout;
        layout.degree = source_ciphertext.meta.degree;
        layout.q_count = source_ciphertext.meta.q_count;
        layout.p_count = 0;
        layout.shards.push_back(GpuPolyShard{
            0, 0, 0, source_ciphertext.meta.q_count, 0,
            source_ciphertext.meta.degree});
        for (std::size_t index = 0; index < steps.size(); ++index)
        {
            prepare_ciphertext_destination(
                destination_ciphertexts[index],
                &source_ciphertext,
                nullptr,
                source_ciphertext.meta,
                2,
                device_id,
                layout);
            const auto dst_view = destination_ciphertexts[index].make_view();
            GpuWord *dst0 = dst_view.polys[0].shards.front().ptr;
            GpuWord *dst1 = dst_view.polys[1].shards.front().ptr;
            const GpuWord *src0 =
                workspace.inner_q_batch.q.data() + (index * 2) * q_words;
            if (dst1 == dst0 + q_words)
            {
                gpu_check_cuda(
                    cudaMemcpyAsync(
                        dst0, src0, 2 * q_words * sizeof(GpuWord),
                        cudaMemcpyDeviceToDevice, stream),
                    "GpuEvaluator::rotate_many_hoisted batch d2d");
            }
            else
            {
                gpu_check_cuda(
                    cudaMemcpyAsync(
                        dst0, src0, q_words * sizeof(GpuWord),
                        cudaMemcpyDeviceToDevice, stream),
                    "GpuEvaluator::rotate_many_hoisted batch d2d c0");
                gpu_check_cuda(
                    cudaMemcpyAsync(
                        dst1, src0 + q_words, q_words * sizeof(GpuWord),
                        cudaMemcpyDeviceToDevice, stream),
                    "GpuEvaluator::rotate_many_hoisted batch d2d c1");
            }
        }
        return;
    }
    for (std::size_t index = 0; index < steps.size(); ++index)
    {
        keyswitch_handler_.moddown_qp_ciphertext_to_q(
            workspace.outer_accumulator,
            index,
            destination_ciphertexts[index],
            source_ciphertext.meta,
            level_info,
            workspace.keyswitch);
    }
}

namespace
{
template <typename FallbackFn, typename EmitChunkFn>
void rotate_many_hoisted_batch_impl(
    const char *caller,
    const GpuParameterData &params,
    const GpuKeySwitchHandler &keyswitch_handler,
    const std::vector<const GpuCiphertextData *> &sources,
    const std::vector<int> &steps,
    const GpuGaloisKeysData &galois_keys_in,
    GpuDoubleHoistWorkspace &workspace,
    FallbackFn &&fallback,
    EmitChunkFn &&emit_chunk)
{
    for (const auto *src : sources)
    {
        if (src == nullptr)
        {
            throw std::invalid_argument(std::string(caller) + ": null source pointer");
        }
    }
    for (const int step : steps)
    {
        if (step == 0)
        {
            throw std::invalid_argument(std::string(caller) + ": zero step is not supported");
        }
    }

    const std::size_t source_count = sources.size();
    if (source_count < 2 || source_count > kernel::kMaxDoubleHoistKeymulBatchSources)
    {
        fallback();
        return;
    }

    const auto &ref = *sources[0];
    validate_ntt_ciphertext_input(caller, ref, true);
    const auto &galois_keys = level_key_for(params, galois_keys_in, ref.meta, caller);
    const int device_id = ref.fields_.front().device_id;
    const std::size_t degree = ref.meta.degree;
    const std::size_t q_count = ref.meta.q_count;
    const auto &level_info = params.get_level(ref.meta.parms_id);

    const GpuParameterShard *parameter_shard = nullptr;
    for (const auto &candidate : level_info.shards)
    {
        if (candidate.device_id == device_id &&
            candidate.hybrid_base_q_count == q_count)
        {
            parameter_shard = &candidate;
            break;
        }
    }

    bool fast_ok =
        parameter_shard != nullptr &&
        parameter_shard->hybrid_base_p_count > 0 &&
        !galois_keys.empty() &&
        galois_keys.meta.galois_format == GpuGaloisKeyFormat::InversePreRotated &&
        galois_keys.meta.degree == degree &&
        galois_keys.meta.q_count >= q_count &&
        galois_keys.meta.p_count == parameter_shard->hybrid_base_p_count &&
        !galois_keys.galois_key_q0_ptrs.empty() &&
        !galois_keys.galois_key_p0_ptrs.empty() &&
        !galois_keys.galois_key_q1_ptrs.empty() &&
        !galois_keys.galois_key_p1_ptrs.empty();

    const std::size_t p_count = fast_ok ? parameter_shard->hybrid_base_p_count : 0;
    const std::size_t dnum = fast_ok ? (q_count + p_count - 1) / p_count : 0;
    const std::size_t storage_dnum = galois_keys.meta.decomposition_count;
    if (dnum == 0 || storage_dnum < dnum)
    {
        fast_ok = false;
    }

    for (std::size_t s = 0; fast_ok && s < source_count; ++s)
    {
        const auto &src = *sources[s];
        if (src.empty() || src.fields_.empty() ||
            !(src.meta.parms_id == ref.meta.parms_id) ||
            !src.meta.is_ntt_form ||
            src.meta.degree != degree ||
            src.meta.q_count != q_count ||
            src.meta.p_count != 0 ||
            src.size() != 2 ||
            src.meta.component_count != 2 ||
            src.polys_[0].shards.size() != 1 ||
            !all_components_use_layout(src, src.polys_[0]) ||
            src.fields_[0].device_id != device_id ||
            src.polys_[0].shards[0].limb_begin != 0 ||
            src.polys_[0].shards[0].limb_count != q_count ||
            src.polys_[0].shards[0].coeff_begin != 0 ||
            src.polys_[0].shards[0].coeff_count != degree)
        {
            fast_ok = false;
        }
    }

    std::vector<std::uint32_t> step_elts(steps.size());
    std::vector<std::uint32_t> step_key_indices(steps.size());
    for (std::size_t i = 0; fast_ok && i < steps.size(); ++i)
    {
        const std::uint32_t elt = galois_elt_from_rotation_step(degree, steps[i]);
        const std::size_t key_index = galois_key_index(elt);
        if (key_index > std::numeric_limits<std::uint32_t>::max() ||
            galois_keys.galois_elts_by_key_index.size() <= key_index ||
            galois_keys.galois_elts_by_key_index[key_index] != elt ||
            (key_index + 1) * storage_dnum > galois_keys.galois_key_q0_ptrs.size())
        {
            fast_ok = false;
        }
        else
        {
            step_elts[i] = elt;
            step_key_indices[i] = static_cast<std::uint32_t>(key_index);
        }
    }

    if (!fast_ok)
    {
        fallback();
        return;
    }

    std::vector<GpuConstCiphertextView> src_views;
    src_views.reserve(source_count);
    std::vector<GpuConstRNSPolyView> c1_polys;
    c1_polys.reserve(source_count);
    workspace.outer_group_hoists.resize(source_count);
    std::vector<GpuHoistedDecomposition *> hoist_dests;
    hoist_dests.reserve(source_count);
    for (std::size_t s = 0; s < source_count; ++s)
    {
        src_views.push_back(sources[s]->make_const_view());
        c1_polys.push_back(src_views.back().polys[1]);
        hoist_dests.push_back(&workspace.outer_group_hoists[s]);
    }

    (void)galois_keys.const_view(q_count);
    keyswitch_handler.hoist_decompose_modup_ntt_batch(
        c1_polys, level_info, hoist_dests, workspace.keyswitch);

    const std::size_t q_words = q_count * degree;
    const std::size_t p_words = p_count * degree;
    constexpr std::size_t kMaxBatchModDownCts = 32;
    const std::size_t max_chunk_steps = std::min(
        kernel::kMaxDoubleHoistKeymulManySteps,
        std::max<std::size_t>(1, kMaxBatchModDownCts / source_count));
    const std::size_t max_chunk_cts =
        std::min(steps.size(), max_chunk_steps) * source_count;
    workspace.outer_accumulator.ensure_capacity(
        device_id, degree, q_count, p_count, max_chunk_cts);

    GpuRNSPoly layout;
    layout.degree = degree;
    layout.q_count = q_count;
    layout.p_count = 0;
    layout.shards.push_back(GpuPolyShard{0, 0, 0, q_count, 0, degree});

    const auto stream = gpu_execution_stream();
    for (std::size_t offset = 0; offset < steps.size(); offset += max_chunk_steps)
    {
        const std::size_t chunk_steps = std::min(max_chunk_steps, steps.size() - offset);
        const std::size_t chunk_cts = chunk_steps * source_count;

        kernel::DoubleHoistKeymulManyBatchArgs args{};
        for (std::size_t k = 0; k < chunk_steps; ++k)
        {
            args.galois_elts[k] = step_elts[offset + k];
            args.key_indices[k] = step_key_indices[offset + k];
        }
        for (std::size_t s = 0; s < source_count; ++s)
        {
            const auto &h = workspace.outer_group_hoists[s];
            args.digits_q[s] = h.digits_q_ntt.data();
            args.digits_p[s] = h.digits_p_ntt.data();
            args.lifted_c0_q[s] = src_views[s].polys[0].shards.front().ptr;
            args.destination_q[s] = workspace.outer_accumulator.q.data() + s * (2 * q_words);
            args.destination_p[s] = workspace.outer_accumulator.p.data() + s * (2 * p_words);
        }

        kernel::launch_double_hoist_pre_rotated_keymul_many_batch(
            args,
            source_count,
            chunk_steps,
            source_count * 2 * q_words,
            source_count * 2 * p_words,
            galois_keys.galois_key_q0_ptrs.data(),
            galois_keys.galois_key_p0_ptrs.data(),
            galois_keys.galois_key_q1_ptrs.data(),
            galois_keys.galois_key_p1_ptrs.data(),
            dnum,
            storage_dnum,
            *parameter_shard,
            degree);

        keyswitch_handler.moddown_qp_ciphertext_batch_to_q(
            workspace.outer_accumulator,
            chunk_cts,
            workspace.inner_q_batch,
            level_info,
            workspace.batch_p_coeff,
            workspace.batch_converted_q);

        gpu_check_cuda(
            cudaSetDevice(device_id),
            (std::string(caller) + " cudaSetDevice").c_str());
        emit_chunk(offset, chunk_steps, q_words, device_id, layout, stream);
    }
}
}  // namespace

void GpuEvaluator::rotate_many_hoisted_batch(
    const std::vector<const GpuCiphertextData *> &sources,
    const std::vector<int> &steps,
    const GpuGaloisKeysData &galois_keys_in,
    GpuDoubleHoistWorkspace &workspace,
    std::vector<std::vector<GpuCiphertextData>> &destination_batches) const
{
    const char *caller = "GpuEvaluator::rotate_many_hoisted_batch";
    destination_batches.resize(sources.size());
    if (sources.empty() || steps.empty())
    {
        for (auto &batch : destination_batches)
        {
            batch.clear();
        }
        return;
    }
    auto fallback = [&]() {
        for (std::size_t s = 0; s < sources.size(); ++s)
        {
            rotate_many_hoisted(
                *sources[s], steps, galois_keys_in, workspace, destination_batches[s]);
        }
    };
    const std::size_t source_count = sources.size();
    bool resized = false;
    auto emit_chunk = [&](std::size_t offset, std::size_t chunk_steps,
                          std::size_t q_words, int device_id,
                          const GpuRNSPoly &layout, cudaStream_t stream) {
        if (!resized)
        {
            for (std::size_t s = 0; s < source_count; ++s)
            {
                destination_batches[s].resize(steps.size());
            }
            resized = true;
        }
        for (std::size_t k = 0; k < chunk_steps; ++k)
        {
            for (std::size_t s = 0; s < source_count; ++s)
            {
                auto &dst = destination_batches[s][offset + k];
                prepare_ciphertext_destination(
                    dst,
                    sources[s],
                    nullptr,
                    sources[s]->meta,
                    2,
                    device_id,
                    layout);
                const auto dst_view = dst.make_view();
                GpuWord *dst0 = dst_view.polys[0].shards.front().ptr;
                GpuWord *dst1 = dst_view.polys[1].shards.front().ptr;
                const std::size_t batch_idx = k * source_count + s;
                const GpuWord *src0 =
                    workspace.inner_q_batch.q.data() + (batch_idx * 2) * q_words;
                if (dst1 == dst0 + q_words)
                {
                    gpu_check_cuda(
                        cudaMemcpyAsync(
                            dst0, src0, 2 * q_words * sizeof(GpuWord),
                            cudaMemcpyDeviceToDevice, stream),
                        "GpuEvaluator::rotate_many_hoisted_batch d2d");
                }
                else
                {
                    gpu_check_cuda(
                        cudaMemcpyAsync(
                            dst0, src0, q_words * sizeof(GpuWord),
                            cudaMemcpyDeviceToDevice, stream),
                        "GpuEvaluator::rotate_many_hoisted_batch d2d c0");
                    gpu_check_cuda(
                        cudaMemcpyAsync(
                            dst1, src0 + q_words, q_words * sizeof(GpuWord),
                            cudaMemcpyDeviceToDevice, stream),
                        "GpuEvaluator::rotate_many_hoisted_batch d2d c1");
                }
            }
        }
    };
    rotate_many_hoisted_batch_impl(
        caller, params_, keyswitch_handler_, sources, steps,
        galois_keys_in, workspace, fallback, emit_chunk);
}

void GpuEvaluator::rotate_many_hoisted_batch_contiguous(
    const std::vector<const GpuCiphertextData *> &sources,
    const std::vector<int> &steps,
    const GpuGaloisKeysData &galois_keys_in,
    GpuDoubleHoistWorkspace &workspace,
    const std::vector<GpuWord *> &destination_qs) const
{
    const char *caller = "GpuEvaluator::rotate_many_hoisted_batch_contiguous";
    if (destination_qs.size() != sources.size())
    {
        throw std::invalid_argument(
            std::string(caller) + ": destination_qs size mismatch");
    }
    if (sources.empty() || steps.empty())
    {
        return;
    }
    for (const auto *dst : destination_qs)
    {
        if (dst == nullptr)
        {
            throw std::invalid_argument(std::string(caller) + ": null destination");
        }
    }
    auto fallback = [&]() {
        for (std::size_t s = 0; s < sources.size(); ++s)
        {
            rotate_many_hoisted_contiguous(
                *sources[s], steps, galois_keys_in, workspace, destination_qs[s]);
        }
    };
    const std::size_t source_count = sources.size();
    auto emit_chunk = [&](std::size_t offset, std::size_t chunk_steps,
                          std::size_t q_words, int, const GpuRNSPoly &,
                          cudaStream_t stream) {
        const std::size_t row_bytes = 2 * q_words * sizeof(GpuWord);
        for (std::size_t s = 0; s < source_count; ++s)
        {
            gpu_check_cuda(
                cudaMemcpy2DAsync(
                    destination_qs[s] + offset * 2 * q_words,
                    row_bytes,
                    workspace.inner_q_batch.q.data() + s * 2 * q_words,
                    source_count * row_bytes,
                    row_bytes,
                    chunk_steps,
                    cudaMemcpyDeviceToDevice,
                    stream),
                "GpuEvaluator::rotate_many_hoisted_batch_contiguous d2d");
        }
    };
    rotate_many_hoisted_batch_impl(
        caller, params_, keyswitch_handler_, sources, steps,
        galois_keys_in, workspace, fallback, emit_chunk);
}

void GpuEvaluator::rotate_many_hoisted_contiguous(
    const GpuCiphertextData &source_ciphertext,
    const std::vector<int> &steps,
    const GpuGaloisKeysData &galois_keys_in,
    GpuDoubleHoistWorkspace &workspace,
    GpuWord *destination_q) const
{
    validate_ntt_ciphertext_input(
        "GpuEvaluator::rotate_many_hoisted_contiguous",
        source_ciphertext,
        true);
    if (steps.empty())
    {
        return;
    }
    if (destination_q == nullptr)
    {
        throw std::invalid_argument(
            "GpuEvaluator::rotate_many_hoisted_contiguous: null destination");
    }
    const GpuLevelInfo *level_info_ptr = nullptr;
    (void)rotate_many_hoisted_qp_impl(
        "GpuEvaluator::rotate_many_hoisted_contiguous",
        params_,
        keyswitch_handler_,
        source_ciphertext,
        steps,
        galois_keys_in,
        workspace,
        level_info_ptr);
    const auto &level_info = *level_info_ptr;
    const int device_id = source_ciphertext.fields_.front().device_id;
    const std::size_t q_words =
        source_ciphertext.meta.q_count * source_ciphertext.meta.degree;
    const auto stream = gpu_execution_stream();
    gpu_check_cuda(
        cudaSetDevice(device_id),
        "GpuEvaluator::rotate_many_hoisted_contiguous cudaSetDevice");

    if (steps.size() >= 2)
    {
        keyswitch_handler_.moddown_qp_ciphertext_batch_to_q(
            workspace.outer_accumulator,
            steps.size(),
            workspace.inner_q_batch,
            level_info,
            workspace.batch_p_coeff,
            workspace.batch_converted_q);
        gpu_check_cuda(
            cudaMemcpyAsync(
                destination_q,
                workspace.inner_q_batch.q.data(),
                steps.size() * 2 * q_words * sizeof(GpuWord),
                cudaMemcpyDeviceToDevice,
                stream),
            "GpuEvaluator::rotate_many_hoisted_contiguous batch d2d");
        return;
    }
    keyswitch_handler_.moddown_qp_ciphertext_to_q(
        workspace.outer_accumulator,
        0,
        workspace.inner_q,
        source_ciphertext.meta,
        level_info,
        workspace.keyswitch);
    const auto dst_view = workspace.inner_q.make_const_view();
    gpu_check_cuda(
        cudaMemcpyAsync(
            destination_q,
            dst_view.polys[0].shards.front().ptr,
            q_words * sizeof(GpuWord),
            cudaMemcpyDeviceToDevice,
            stream),
        "GpuEvaluator::rotate_many_hoisted_contiguous single d2d c0");
    gpu_check_cuda(
        cudaMemcpyAsync(
            destination_q + q_words,
            dst_view.polys[1].shards.front().ptr,
            q_words * sizeof(GpuWord),
            cudaMemcpyDeviceToDevice,
            stream),
        "GpuEvaluator::rotate_many_hoisted_contiguous single d2d c1");
}

void GpuEvaluator::rotate_inplace(
    GpuCiphertextData &ciphertext, int step,
    const GpuGaloisKeysData &galois_keys, GpuRotateWorkspace &workspace) const
{
    rotate_inplace(ciphertext, step, galois_keys, workspace, nullptr, 0);
}

void GpuEvaluator::rotate_inplace(
    GpuCiphertextData &ciphertext, int step,
    const GpuGaloisKeysData &galois_keys_in, GpuRotateWorkspace &workspace,
    const GpuRotationHoistCache *hoist_cache, std::uintptr_t source_tag) const
{
    const auto &galois_keys = level_key_for(params_, galois_keys_in, ciphertext.meta, "GpuEvaluator::rotate_inplace");
    validate_rotate_input(ciphertext);
    if (step == 0) return;
    if (galois_keys.empty())
        throw std::invalid_argument("GpuEvaluator::rotate_inplace: empty galois keys");
    const auto galois_elt = galois_elt_from_rotation_step(ciphertext.meta.degree, step);
    const auto key_index = galois_key_index(galois_elt);
    const int device_id = ciphertext.fields_.front().device_id;
    const auto &level_info = params_.get_level(ciphertext.meta.parms_id);
    auto source_view = ciphertext.make_const_view();
    auto destination_view = ciphertext.make_view();

    if (galois_keys.meta.galois_format == GpuGaloisKeyFormat::InversePreRotated)
    {
        auto &staged = workspace.pre_rotated;
        staged.outer_accumulator.ensure_capacity(device_id, ciphertext.meta.degree,
            ciphertext.meta.q_count, galois_keys.meta.p_count, 1);
        // The input is consumed here, so a cached decomposition is only read.
        const GpuHoistedDecomposition *hoisted = &staged.source_hoist;
        if (hoist_cache != nullptr && source_tag != 0 && hoist_cache->source_tag == source_tag &&
            hoist_cache->decomposition.parms_id == ciphertext.meta.parms_id &&
            hoist_cache->decomposition.q_count == ciphertext.meta.q_count &&
            hoist_cache->decomposition.device_id == device_id)
            hoisted = &hoist_cache->decomposition;
        else
            keyswitch_handler_.hoist_decompose_modup_ntt(source_view.polys[1],
                level_info, staged.source_hoist, staged.keyswitch);
        const auto &key_view = galois_keys.const_view(ciphertext.meta.q_count);
        keyswitch_handler_.keyswitch_multsum_no_moddown(*hoisted,
            galois_elt, key_view, galois_keys, key_index,
            staged.outer_accumulator, 0, true, level_info, staged.keyswitch);
        const GpuParameterShard *parameter_shard = nullptr;
        for (const auto &candidate : level_info.shards)
            if (candidate.device_id == device_id &&
                candidate.hybrid_base_q_count == ciphertext.meta.q_count)
            {
                parameter_shard = &candidate;
                break;
            }
        if (!parameter_shard)
            throw std::invalid_argument("GpuEvaluator::rotate_inplace: HYBRID parameter shard is absent");
        kernel::launch_double_hoist_add_lifted_galois_c0(
            staged.outer_accumulator.q_component(0, 0),
            source_view.polys[0].shards.front().ptr, galois_elt,
            *parameter_shard, ciphertext.meta.degree);
        const auto meta = ciphertext.meta;
        keyswitch_handler_.moddown_qp_ciphertext_to_q(staged.outer_accumulator,
            0, ciphertext, meta, level_info, staged.keyswitch);
        return;
    }

    auto &temporary = workspace.permutation;
    if (temporary.empty())
        temporary = GpuCiphertextData::allocate_single_device_sharded(
            ciphertext.meta.degree, ciphertext.meta.q_count, 1, device_id,
            ciphertext.polys_.front().shards, 0);
    if (temporary.meta.degree != ciphertext.meta.degree ||
        temporary.meta.q_count != ciphertext.meta.q_count ||
        temporary.size() != 1 || temporary.fields_.front().device_id != device_id)
        throw std::invalid_argument("GpuEvaluator::rotate_inplace: workspace shape mismatch");
    temporary.meta = ciphertext.meta;
    temporary.meta.component_count = 1;
    auto temporary_view = temporary.make_view();
    kernel::launch_apply_galois_ntt_poly_shard(temporary_view.polys[0].shards.front(),
        source_view.polys[0].shards.front(), galois_elt, ciphertext.meta.degree);
    copy_poly(destination_view.polys[0], temporary.make_const_view().polys[0],
              "GpuEvaluator::rotate_inplace copy c0");
    kernel::launch_apply_galois_ntt_poly_shard(temporary_view.polys[0].shards.front(),
        source_view.polys[1].shards.front(), galois_elt, ciphertext.meta.degree);
    zero_poly(destination_view.polys[1], "GpuEvaluator::rotate_inplace zero c1");
    auto temporary_const_view = temporary.make_const_view();
    const auto &key_view = galois_keys.const_view(ciphertext.meta.q_count);
    keyswitch_handler_.switch_key_hybrid_ciphertext(destination_view,
        temporary_const_view.polys[0], key_view, galois_keys, key_index,
        level_info, &workspace.key_switch);
}

void GpuEvaluator::conjugate(
    const GpuCiphertextData &source_ciphertext,
    const GpuGaloisKeysData &galois_keys_in,
    GpuCiphertextData &destination_ciphertext) const
{
    const auto &galois_keys = level_key_for(params_, galois_keys_in, source_ciphertext.meta, "GpuEvaluator::conjugate");
    validate_ntt_ciphertext_input(
        "GpuEvaluator::conjugate",
        source_ciphertext,
        true);
    if (source_ciphertext.size() != 2)
    {
        throw std::invalid_argument(
            "GpuEvaluator::conjugate: first implementation expects a size-2 ciphertext");
    }
    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::conjugate: input ciphertext p limbs are not supported");
    }
    if (source_ciphertext.polys_.at(0).shards.size() != 1)
    {
        throw std::invalid_argument(
            "GpuEvaluator::conjugate: first implementation requires one full shard");
    }
    if (galois_keys.empty())
    {
        throw std::invalid_argument("GpuEvaluator::conjugate: empty galois keys");
    }
    if (galois_keys.meta.galois_format ==
        GpuGaloisKeyFormat::InversePreRotated)
    {
        static const bool persistent_scratch = [] {
            const char *raw = std::getenv("POSEIDON_GPU_PERSISTENT_ROTATE_WORKSPACE");
            return raw != nullptr && std::string(raw) == "1";
        }();
        GpuDoubleHoistWorkspace local_staged;
        std::unique_lock<std::mutex> scratch_lock;
        if (persistent_scratch)
        {
            scratch_lock = std::unique_lock<std::mutex>(rotate_scratch_->mutex);
        }
        GpuDoubleHoistWorkspace &staged =
            persistent_scratch ? rotate_scratch_->workspace : local_staged;
        conjugate_pre_rotated(
            source_ciphertext,
            galois_keys,
            staged,
            destination_ciphertext);
        return;
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);
    const auto &level_info = params_.get_level(source_ciphertext.meta.parms_id);

    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            2,
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);

    result.meta = source_ciphertext.meta;
    result.meta.component_count = 2;
    result.meta.is_ntt_form = true;

    const std::uint32_t galois_elt =
        galois_elt_for_conjugation(source_ciphertext.meta.degree);
    const std::size_t key_index = galois_key_index(galois_elt);

    GpuCiphertextData conjugated_c1 =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            1,
            device_id,
            reference_layout.shards,
            source_ciphertext.meta.p_count);
    conjugated_c1.meta = source_ciphertext.meta;
    conjugated_c1.meta.component_count = 1;
    conjugated_c1.meta.is_ntt_form = true;

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();
    auto conjugated_c1_view = conjugated_c1.make_view();

    kernel::launch_apply_galois_ntt_poly_shard(
        destination_view.polys[0].shards.front(),
        source_view.polys[0].shards.front(),
        galois_elt,
        source_ciphertext.meta.degree);
    kernel::launch_apply_galois_ntt_poly_shard(
        conjugated_c1_view.polys[0].shards.front(),
        source_view.polys[1].shards.front(),
        galois_elt,
        source_ciphertext.meta.degree);

    zero_poly(destination_view.polys[1], "GpuEvaluator::conjugate zero c1");

    auto conjugated_c1_const_view = conjugated_c1.make_const_view();
    const auto &galois_keys_view =
        galois_keys.const_view(source_ciphertext.meta.q_count);
    keyswitch_handler_.switch_key_hybrid_ciphertext(
        destination_view,
        conjugated_c1_const_view.polys[0],
        galois_keys_view,
        galois_keys,
        key_index,
        level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::conjugate_pair(
    const GpuCiphertextData &source_a,
    const GpuCiphertextData &source_b,
    const GpuGaloisKeysData &galois_keys_in,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    automorphism_batch_impl(
        {&source_a, &source_b},
        true,
        0,
        galois_keys_in,
        {&destination_a, &destination_b});
}

void GpuEvaluator::rotate_pair(
    const GpuCiphertextData &source_a,
    const GpuCiphertextData &source_b,
    int step,
    const GpuGaloisKeysData &galois_keys_in,
    GpuCiphertextData &destination_a,
    GpuCiphertextData &destination_b) const
{
    automorphism_batch_impl(
        {&source_a, &source_b},
        false,
        step,
        galois_keys_in,
        {&destination_a, &destination_b});
}

void GpuEvaluator::conjugate_batch(
    const std::vector<const GpuCiphertextData *> &sources,
    const GpuGaloisKeysData &galois_keys_in,
    std::vector<GpuCiphertextData> &destinations) const
{
    if (sources.empty())
    {
        destinations.clear();
        return;
    }
    destinations.resize(sources.size());
    std::vector<GpuCiphertextData *> dst_ptrs(sources.size());
    for (std::size_t i = 0; i < sources.size(); ++i)
    {
        if (sources[i] == nullptr)
        {
            throw std::invalid_argument("GpuEvaluator::conjugate_batch: null source");
        }
        dst_ptrs[i] = &destinations[i];
    }
    automorphism_batch_impl(sources, true, 0, galois_keys_in, dst_ptrs);
}

void GpuEvaluator::automorphism_batch_impl(
    const std::vector<const GpuCiphertextData *> &sources,
    bool is_conjugate,
    int step,
    const GpuGaloisKeysData &galois_keys_in,
    const std::vector<GpuCiphertextData *> &destinations) const
{
    const std::size_t count = sources.size();
    auto fallback = [&]() {
        for (std::size_t i = 0; i < count; ++i)
        {
            if (is_conjugate) conjugate(*sources[i], galois_keys_in, *destinations[i]);
            else rotate(*sources[i], step, galois_keys_in, *destinations[i]);
        }
    };

    if (count < 2 || count > kernel::kMaxDoubleHoistKeymulBatchSources ||
        (!is_conjugate && step == 0))
    {
        fallback();
        return;
    }
    for (std::size_t i = 0; i < count; ++i)
    {
        for (std::size_t j = i + 1; j < count; ++j)
        {
            if (destinations[i] == destinations[j])
            {
                fallback();
                return;
            }
        }
        for (std::size_t j = 0; j < count; ++j)
        {
            if (i != j && destinations[i] == sources[j])
            {
                fallback();
                return;
            }
        }
    }

    const auto &ref = *sources[0];
    if (ref.empty() || ref.fields_.empty())
    {
        fallback();
        return;
    }
    const char *op_name =
        is_conjugate ? "GpuEvaluator::conjugate_batch" : "GpuEvaluator::rotate_pair";
    const auto &galois_keys = level_key_for(params_, galois_keys_in, ref.meta, op_name);
    const int device_id = ref.fields_[0].device_id;
    const std::size_t degree = ref.meta.degree;
    const std::size_t q_count = ref.meta.q_count;
    const auto &level_info = params_.get_level(ref.meta.parms_id);
    const GpuParameterShard *parameter_shard = nullptr;
    for (const auto &candidate : level_info.shards)
    {
        if (candidate.device_id == device_id &&
            candidate.hybrid_base_q_count == q_count)
        {
            parameter_shard = &candidate;
            break;
        }
    }

    bool fast_ok =
        parameter_shard != nullptr &&
        parameter_shard->hybrid_base_p_count > 0 &&
        !galois_keys.empty() &&
        galois_keys.meta.galois_format == GpuGaloisKeyFormat::InversePreRotated &&
        galois_keys.meta.degree == degree &&
        galois_keys.meta.q_count >= q_count &&
        galois_keys.meta.p_count == parameter_shard->hybrid_base_p_count &&
        !galois_keys.galois_key_q0_ptrs.empty() &&
        !galois_keys.galois_key_p0_ptrs.empty() &&
        !galois_keys.galois_key_q1_ptrs.empty() &&
        !galois_keys.galois_key_p1_ptrs.empty();
    for (std::size_t i = 0; fast_ok && i < count; ++i)
    {
        const auto &s = *sources[i];
        if (s.empty() || s.fields_.empty() ||
            !(s.meta.parms_id == ref.meta.parms_id) ||
            !s.meta.is_ntt_form ||
            s.meta.degree != degree ||
            s.meta.q_count != q_count ||
            s.meta.p_count != 0 ||
            s.size() != 2 || s.meta.component_count != 2 ||
            s.polys_[0].shards.size() != 1 ||
            !all_components_use_layout(s, s.polys_[0]) ||
            s.fields_[0].device_id != device_id ||
            s.polys_[0].shards[0].limb_begin != 0 ||
            s.polys_[0].shards[0].limb_count != q_count ||
            s.polys_[0].shards[0].coeff_begin != 0 ||
            s.polys_[0].shards[0].coeff_count != degree)
        {
            fast_ok = false;
        }
    }

    const std::size_t p_count = fast_ok ? parameter_shard->hybrid_base_p_count : 0;
    const std::size_t dnum = fast_ok ? (q_count + p_count - 1) / p_count : 0;
    const std::size_t storage_dnum = galois_keys.meta.decomposition_count;
    const std::uint32_t galois_elt = fast_ok
        ? (is_conjugate ? galois_elt_for_conjugation(degree)
                        : galois_elt_from_rotation_step(degree, step))
        : 0;
    const std::size_t key_index = fast_ok ? galois_key_index(galois_elt) : 0;
    if (!fast_ok || dnum == 0 || storage_dnum < dnum ||
        key_index >= galois_keys.meta.key_count ||
        galois_keys.galois_elts_by_key_index.size() <= key_index ||
        galois_keys.galois_elts_by_key_index[key_index] != galois_elt ||
        (key_index + 1) * storage_dnum > galois_keys.galois_key_q0_ptrs.size())
    {
        fallback();
        return;
    }

    static const bool persistent_scratch = [] {
        const char *raw = std::getenv("POSEIDON_GPU_PERSISTENT_ROTATE_WORKSPACE");
        return raw != nullptr && std::string(raw) == "1";
    }();
    GpuDoubleHoistWorkspace local_staged;
    std::unique_lock<std::mutex> scratch_lock;
    if (persistent_scratch)
    {
        scratch_lock = std::unique_lock<std::mutex>(rotate_scratch_->mutex);
    }
    GpuDoubleHoistWorkspace &staged =
        persistent_scratch ? rotate_scratch_->workspace : local_staged;

    std::vector<GpuConstCiphertextView> src_views;
    std::vector<GpuConstRNSPolyView> c1_polys;
    src_views.reserve(count);
    c1_polys.reserve(count);
    staged.outer_group_hoists.resize(count);
    std::vector<GpuHoistedDecomposition *> hoist_dests(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        src_views.push_back(sources[i]->make_const_view());
        c1_polys.push_back(src_views.back().polys[1]);
        hoist_dests[i] = &staged.outer_group_hoists[i];
    }

    (void)galois_keys.const_view(q_count);
    keyswitch_handler_.hoist_decompose_modup_ntt_batch(
        c1_polys, level_info, hoist_dests, staged.keyswitch);

    const std::size_t q_words = q_count * degree;
    const std::size_t p_words = p_count * degree;
    staged.baby_tile.ensure_capacity(device_id, degree, q_count, p_count, count);

    kernel::DoubleHoistKeymulManyBatchArgs args{};
    args.galois_elts[0] = galois_elt;
    args.key_indices[0] = static_cast<std::uint32_t>(key_index);
    for (std::size_t i = 0; i < count; ++i)
    {
        args.digits_q[i] = staged.outer_group_hoists[i].digits_q_ntt.data();
        args.digits_p[i] = staged.outer_group_hoists[i].digits_p_ntt.data();
        args.lifted_c0_q[i] = src_views[i].polys[0].shards.front().ptr;
        args.destination_q[i] = staged.baby_tile.q.data() + i * (2 * q_words);
        args.destination_p[i] = staged.baby_tile.p.data() + i * (2 * p_words);
    }
    kernel::launch_double_hoist_pre_rotated_keymul_many_batch(
        args,
        count,
        1,
        count * 2 * q_words,
        count * 2 * p_words,
        galois_keys.galois_key_q0_ptrs.data(),
        galois_keys.galois_key_p0_ptrs.data(),
        galois_keys.galois_key_q1_ptrs.data(),
        galois_keys.galois_key_p1_ptrs.data(),
        dnum,
        storage_dnum,
        *parameter_shard,
        degree);

    keyswitch_handler_.moddown_qp_ciphertext_batch_to_q(
        staged.baby_tile,
        count,
        staged.inner_q_batch,
        level_info,
        staged.batch_p_coeff,
        staged.batch_converted_q);

    const auto stream = gpu_execution_stream();
    gpu_check_cuda(cudaSetDevice(device_id), "automorphism_batch cudaSetDevice");
    for (std::size_t i = 0; i < count; ++i)
    {
        GpuCiphertextMeta meta = sources[i]->meta;
        meta.component_count = 2;
        meta.is_ntt_form = true;
        prepare_ciphertext_destination(
            *destinations[i], nullptr, nullptr, meta, 2, device_id, sources[i]->polys_[0]);
        auto dst_view = destinations[i]->make_view();
        GpuWord *dst0 = dst_view.polys[0].shards[0].ptr;
        GpuWord *dst1 = dst_view.polys[1].shards[0].ptr;
        const GpuWord *src0 = staged.inner_q_batch.q.data() + (i * 2) * q_words;
        if (dst1 == dst0 + q_words)
        {
            gpu_check_cuda(
                cudaMemcpyAsync(
                    dst0, src0, 2 * q_words * sizeof(GpuWord),
                    cudaMemcpyDeviceToDevice, stream),
                "automorphism_batch copy");
        }
        else
        {
            gpu_check_cuda(
                cudaMemcpyAsync(
                    dst0, src0, q_words * sizeof(GpuWord),
                    cudaMemcpyDeviceToDevice, stream),
                "automorphism_batch copy c0");
            gpu_check_cuda(
                cudaMemcpyAsync(
                    dst1, src0 + q_words, q_words * sizeof(GpuWord),
                    cudaMemcpyDeviceToDevice, stream),
                "automorphism_batch copy c1");
        }
    }
}

void GpuEvaluator::conjugate_pre_rotated(
    const GpuCiphertextData &source_ciphertext,
    const GpuGaloisKeysData &galois_keys_in,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    const auto &galois_keys = level_key_for(params_, galois_keys_in, source_ciphertext.meta, "GpuEvaluator::conjugate_pre_rotated");
    validate_ntt_ciphertext_input(
        "GpuEvaluator::conjugate_pre_rotated",
        source_ciphertext,
        true);
    if (source_ciphertext.size() != 2 ||
        source_ciphertext.meta.p_count != 0 ||
        source_ciphertext.polys_.at(0).shards.size() != 1 ||
        galois_keys.meta.galois_format !=
            GpuGaloisKeyFormat::InversePreRotated)
    {
        throw std::invalid_argument(
            "GpuEvaluator::conjugate_pre_rotated: invalid input/key format");
    }

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    const auto source_view = source_ciphertext.make_const_view();
    const std::uint32_t galois_elt =
        galois_elt_for_conjugation(source_ciphertext.meta.degree);
    const std::size_t key_index = galois_key_index(galois_elt);

    workspace.outer_accumulator.ensure_capacity(
        device_id,
        source_ciphertext.meta.degree,
        source_ciphertext.meta.q_count,
        galois_keys.meta.p_count,
        1);
    keyswitch_handler_.hoist_decompose_modup_ntt(
        source_view.polys[1],
        level_info,
        workspace.source_hoist,
        workspace.keyswitch);
    const auto &key_view =
        galois_keys.const_view(source_ciphertext.meta.q_count);
    keyswitch_handler_.keyswitch_multsum_no_moddown(
        workspace.source_hoist,
        galois_elt,
        key_view,
        galois_keys,
        key_index,
        workspace.outer_accumulator,
        0,
        true,
        level_info,
        workspace.keyswitch);

    const GpuParameterShard *parameter_shard = nullptr;
    for (const auto &candidate : level_info.shards)
    {
        if (candidate.device_id == device_id &&
            candidate.hybrid_base_q_count ==
                source_ciphertext.meta.q_count)
        {
            parameter_shard = &candidate;
            break;
        }
    }
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "GpuEvaluator::conjugate_pre_rotated: HYBRID parameter shard is absent");
    }
    kernel::launch_double_hoist_add_lifted_galois_c0(
        workspace.outer_accumulator.q_component(0, 0),
        source_view.polys[0].shards.front().ptr,
        galois_elt,
        *parameter_shard,
        source_ciphertext.meta.degree);
    keyswitch_handler_.moddown_qp_ciphertext_to_q(
        workspace.outer_accumulator,
        0,
        workspace.result_q,
        source_ciphertext.meta,
        level_info,
        workspace.keyswitch);
    destination_ciphertext = std::move(workspace.result_q);
}

void GpuEvaluator::multiply_by_diag_matrix_bsgs(
    const GpuCiphertextData &source_ciphertext,
    const GpuMatrixPlain &matrix,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &destination_ciphertext) const
{
    multiply_by_diag_matrix_bsgs(
        source_ciphertext,
        matrix,
        galois_keys,
        1,
        destination_ciphertext);
}

void GpuEvaluator::multiply_by_diag_matrix_bsgs(
    const GpuCiphertextData &source_ciphertext,
    const GpuMatrixPlain &matrix,
    const GpuGaloisKeysData &galois_keys_in,
    std::uint32_t rescale_count,
    GpuCiphertextData &destination_ciphertext) const
{
    const auto &galois_keys = level_key_for(params_, galois_keys_in, source_ciphertext.meta, "GpuEvaluator::multiply_by_diag_matrix_bsgs");
    validate_ntt_ciphertext_input(
        "GpuEvaluator::multiply_by_diag_matrix_bsgs",
        source_ciphertext,
        true);
    if (source_ciphertext.size() != 2)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs: first implementation expects a size-2 ciphertext");
    }
    if (matrix.plain_vec.empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs: empty diagonal matrix");
    }
    if (matrix.n1 == 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs: invalid BSGS n1");
    }

    const auto [index, unused_rot_n1, rot_n2] =
        poseidon::bsgs_index(
            matrix.plain_vec,
            1 << matrix.log_slots,
            static_cast<int>(matrix.n1));
    (void)unused_rot_n1;

    std::map<int, GpuCiphertextData> baby_rotations;
    for (const int step : rot_n2)
    {
        if (step == 0)
        {
            continue;
        }

        GpuCiphertextData rotated;
        rotate(source_ciphertext, step, galois_keys, rotated);
        baby_rotations.emplace(step, std::move(rotated));
    }

    GpuCiphertextData result_accumulator;
    GpuCiphertextData inner_sum;
    GpuCiphertextData product;
    GpuCiphertextData rotated_inner_sum;
    bool have_result = false;

    for (const auto &giant_entry : index)
    {
        const int giant_step = giant_entry.first;
        bool have_inner_sum = false;

        for (const int baby_step : giant_entry.second)
        {
            const int diagonal_index = giant_step + baby_step;
            const auto plaintext_it = matrix.plain_vec.find(diagonal_index);
            if (plaintext_it == matrix.plain_vec.end())
            {
                throw std::invalid_argument(
                    "GpuEvaluator::multiply_by_diag_matrix_bsgs: missing plaintext diagonal");
            }

            const GpuCiphertextData &rotated_source =
                baby_step == 0 ? source_ciphertext : baby_rotations.at(baby_step);

            multiply_plain(rotated_source, plaintext_it->second, product);

            if (!have_inner_sum)
            {
                inner_sum = std::move(product);
                have_inner_sum = true;
            }
            else
            {
                GpuCiphertextData updated_inner_sum;
                add(inner_sum, product, updated_inner_sum);
                inner_sum = std::move(updated_inner_sum);
            }
        }

        if (!have_inner_sum)
        {
            continue;
        }

        if (!have_result)
        {
            if (giant_step == 0)
            {
                result_accumulator = std::move(inner_sum);
            }
            else
            {
                rotate(inner_sum, giant_step, galois_keys, rotated_inner_sum);
                result_accumulator = std::move(rotated_inner_sum);
            }
            have_result = true;
        }
        else
        {
            if (giant_step == 0)
            {
                GpuCiphertextData updated_accumulator;
                add(result_accumulator, inner_sum, updated_accumulator);
                result_accumulator = std::move(updated_accumulator);
            }
            else
            {
                rotate(inner_sum, giant_step, galois_keys, rotated_inner_sum);
                GpuCiphertextData updated_accumulator;
                add(result_accumulator, rotated_inner_sum, updated_accumulator);
                result_accumulator = std::move(updated_accumulator);
            }
        }
    }

    if (!have_result)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs: no nonzero diagonal contribution");
    }

    if (rescale_count == 0)
    {
        destination_ciphertext = std::move(result_accumulator);
    }
    else
    {
        rescale_many(
            result_accumulator,
            destination_ciphertext,
            rescale_count);
    }
}

void GpuEvaluator::multiply_by_diag_matrix_bsgs_loose(
    const GpuCiphertextData &source_ciphertext,
    const GpuMatrixPlain &matrix,
    const GpuGaloisKeysData &galois_keys_in,
    std::uint32_t rescale_count,
    GpuCiphertextData &destination_ciphertext) const
{
    const auto &galois_keys = level_key_for(params_, galois_keys_in, source_ciphertext.meta, "GpuEvaluator::multiply_by_diag_matrix_bsgs_loose");
    validate_ntt_ciphertext_input(
        "GpuEvaluator::multiply_by_diag_matrix_bsgs_loose",
        source_ciphertext,
        true);
    if (source_ciphertext.size() != 2 ||
        source_ciphertext.meta.p_count != 0 ||
        source_ciphertext.polys_.at(0).shards.size() != 1)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs_loose: "
            "requires one-shard, size-2, Q-only input");
    }

    const auto plan = make_gpu_loose_bsgs_plan(matrix);

    // Each non-zero baby rotation is a complete Q-basis ciphertext.  No
    // hoisted decomposition or lifted-QP accumulator is shared with another
    // task.  A future multi-GPU lowering can therefore place these rotations
    // independently and insert ordinary CopyCipher operations when needed.
    std::map<int, GpuCiphertextData> baby_rotations;
    for (const int step : plan.baby_steps)
    {
        if (step == 0)
        {
            continue;
        }
        GpuCiphertextData rotated;
        rotate(source_ciphertext, step, galois_keys, rotated);
        baby_rotations.emplace(step, std::move(rotated));
    }

    // Materialize one ordinary Q-basis result per giant group before the
    // final reduction.  Term products are reduced locally as a bounded-memory
    // reference executor; the public plan retains every independent term for
    // a scheduler that wants to distribute them further.
    std::vector<GpuCiphertextData> group_outputs;
    group_outputs.reserve(plan.groups.size());
    for (const auto &group : plan.groups)
    {
        GpuCiphertextData inner_sum;
        bool have_inner_sum = false;
        for (const auto &term : group.terms)
        {
            const auto plaintext_it =
                matrix.plain_vec.find(term.diagonal_index);
            if (plaintext_it == matrix.plain_vec.end())
            {
                throw std::invalid_argument(
                    "GpuEvaluator::multiply_by_diag_matrix_bsgs_loose: "
                    "missing plaintext diagonal");
            }
            const GpuCiphertextData &rotated_source =
                term.baby_step == 0
                    ? source_ciphertext
                    : baby_rotations.at(term.baby_step);

            GpuCiphertextData product;
            multiply_plain(rotated_source, plaintext_it->second, product);
            if (!have_inner_sum)
            {
                inner_sum = std::move(product);
                have_inner_sum = true;
            }
            else
            {
                GpuCiphertextData updated_inner_sum;
                add(inner_sum, product, updated_inner_sum);
                inner_sum = std::move(updated_inner_sum);
            }
        }
        if (!have_inner_sum)
        {
            throw std::logic_error(
                "GpuEvaluator::multiply_by_diag_matrix_bsgs_loose: "
                "empty giant group");
        }

        if (group.giant_step == 0)
        {
            group_outputs.push_back(std::move(inner_sum));
        }
        else
        {
            GpuCiphertextData rotated_group;
            rotate(inner_sum, group.giant_step, galois_keys, rotated_group);
            group_outputs.push_back(std::move(rotated_group));
        }
    }

    GpuCiphertextData result_accumulator = std::move(group_outputs.front());
    for (std::size_t index = 1; index < group_outputs.size(); ++index)
    {
        GpuCiphertextData updated_accumulator;
        add(
            result_accumulator,
            group_outputs[index],
            updated_accumulator);
        result_accumulator = std::move(updated_accumulator);
    }

    if (rescale_count == 0)
    {
        destination_ciphertext = std::move(result_accumulator);
    }
    else
    {
        rescale_many(
            result_accumulator,
            destination_ciphertext,
            rescale_count);
    }
}

void GpuEvaluator::prepare_double_hoist_babies(
    const GpuCiphertextData &source_ciphertext,
    const std::vector<int> &steps,
    const GpuGaloisKeysData &galois_keys_in,
    GpuDoubleHoistWorkspace &workspace,
    GpuDoubleHoistBabyCache &cache) const
{
    cache.ready = false;
    validate_ntt_ciphertext_input("GpuEvaluator::prepare_double_hoist_babies", source_ciphertext, true);
    if (source_ciphertext.size() != 2 || source_ciphertext.meta.p_count != 0 ||
        source_ciphertext.polys_.at(0).shards.size() != 1 || steps.empty())
        throw std::invalid_argument("prepare_double_hoist_babies: requires a nonempty step list and one-shard Q-only size-2 source");
    const auto &galois_keys = level_key_for(params_, galois_keys_in,
        source_ciphertext.meta, "GpuEvaluator::prepare_double_hoist_babies");
    if (galois_keys.empty())throw std::invalid_argument("prepare_double_hoist_babies: empty Galois keys");
    const auto &level = params_.get_level(source_ciphertext.meta.parms_id);
    const auto view = source_ciphertext.make_const_view();
    const auto &c0 = view.polys[0].shards.front();
    const auto &c1 = view.polys[1].shards.front();
    const GpuParameterShard *shard = nullptr;
    for (const auto &candidate : level.shards)
        if (candidate.device_id == c0.device_id && candidate.hybrid_base_q_count == source_ciphertext.meta.q_count)
        {shard = &candidate;break;}
    if (shard == nullptr || shard->hybrid_base_p_count == 0)
        throw std::invalid_argument("prepare_double_hoist_babies: missing HYBRID parameter shard");
    const auto degree = source_ciphertext.meta.degree;
    const auto q = source_ciphertext.meta.q_count;
    const auto p = shard->hybrid_base_p_count;
    const auto dnum = (q + p - 1) / p;
    cache.rotations.ensure_capacity(c0.device_id, degree, q, p, steps.size());
    keyswitch_handler_.hoist_decompose_modup_ntt(view.polys[1], level, workspace.source_hoist, workspace.keyswitch);
    const auto &key_view = galois_keys.const_view(q);
    const bool fused_c0 = use_double_hoist_fused_baby_keyswitch_c0() &&
        dnum == 1 && galois_keys.meta.galois_format == GpuGaloisKeyFormat::InversePreRotated;
    if (!(fused_c0 && steps.size() >= 2 &&
          launch_hoisted_keymul_many_if_eligible(
              workspace.source_hoist, steps, galois_keys, *shard,
              c0.ptr, c1.ptr, true, cache.rotations)))
    {
        for (std::size_t i = 0; i < steps.size(); ++i)
        {
            if (steps[i] == 0)
            {
                kernel::launch_double_hoist_lift_identity(
                    cache.rotations.q_component(i, 0), cache.rotations.q_component(i, 1),
                    cache.rotations.p_component(i, 0), cache.rotations.p_component(i, 1),
                    c0.ptr, c1.ptr, *shard, degree, false);
                continue;
            }
            const auto elt = galois_elt_from_rotation_step(degree, steps[i]);
            const auto index = galois_key_index(elt);
            keyswitch_handler_.keyswitch_multsum_no_moddown(
                workspace.source_hoist, elt, key_view, galois_keys, index,
                cache.rotations, i, true, level, workspace.keyswitch,
                fused_c0 ? c0.ptr : nullptr);
            if (!fused_c0)kernel::launch_double_hoist_add_lifted_galois_c0(
                cache.rotations.q_component(i, 0), c0.ptr, elt, *shard, degree);
        }
    }
    cache.source_meta = source_ciphertext.meta;
    cache.source_c0 = c0.ptr;cache.source_c1 = c1.ptr;
    cache.galois_key_identity = &galois_keys;
    cache.steps = steps;cache.ready = true;
}

void GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist(
    const GpuCiphertextData &source_ciphertext,
    const GpuMatrixPlainQP &matrix,
    const GpuGaloisKeysData &galois_keys_in,
    std::uint32_t rescale_count,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    multiply_by_diag_matrix_bsgs_double_hoist(source_ciphertext, matrix,
        galois_keys_in, rescale_count, workspace, destination_ciphertext, nullptr);
}

void GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist(
    const GpuCiphertextData &source_ciphertext, const GpuMatrixPlainQP &matrix,
    const GpuGaloisKeysData &galois_keys, std::uint32_t rescale_count,
    GpuDoubleHoistWorkspace &workspace, GpuCiphertextData &destination_ciphertext,
    const GpuDoubleHoistBabyCache *baby_cache) const
{
    double_hoist_impl(source_ciphertext, matrix, galois_keys, rescale_count,
        workspace, destination_ciphertext, baby_cache, nullptr, true);
}

void GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist_sum(
    const GpuCiphertextData &source_ciphertext, const GpuMatrixPlainQP &matrix,
    const GpuGaloisKeysData &galois_keys, std::uint32_t rescale_count,
    GpuDoubleHoistWorkspace &workspace, GpuDoubleHoistSumState &sum_state,
    bool final_input, GpuCiphertextData &destination_ciphertext,
    const GpuDoubleHoistBabyCache *baby_cache) const
{
    double_hoist_impl(source_ciphertext, matrix, galois_keys, rescale_count,
        workspace, destination_ciphertext, baby_cache, &sum_state, final_input);
}

void GpuEvaluator::double_hoist_impl(
    const GpuCiphertextData &source_ciphertext,
    const GpuMatrixPlainQP &matrix,
    const GpuGaloisKeysData &galois_keys_in,
    std::uint32_t rescale_count,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext,
    const GpuDoubleHoistBabyCache *baby_cache,
    GpuDoubleHoistSumState *sum_state, bool final_input) const
{
    const auto &galois_keys = level_key_for(params_, galois_keys_in, source_ciphertext.meta, "GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist");
    validate_ntt_ciphertext_input(
        "GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist",
        source_ciphertext,
        true);
    if (source_ciphertext.size() != 2 ||
        source_ciphertext.meta.p_count != 0 ||
        source_ciphertext.polys_.at(0).shards.size() != 1)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist: "
            "requires one-shard, size-2, Q-only input");
    }
    const bool has_full_plaintexts = !matrix.plain_vec_qp.empty();
    const bool has_compressed_plaintexts =
        !matrix.compressed_plain_vec_qp.empty();
    // A group view reads its parent matrix's plaintexts through plan pointers.
    if ((matrix.group_view
             ? has_full_plaintexts || has_compressed_plaintexts
             : has_full_plaintexts == has_compressed_plaintexts) ||
        matrix.plan.terms.empty() ||
        matrix.plan.baby_steps.empty() ||
        matrix.plan.giant_steps.empty() ||
        matrix.plan.group_term_offsets.size() !=
            matrix.plan.giant_steps.size() + 1)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist: "
            "matrix plan is incomplete");
    }
    if (galois_keys.empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist: "
            "empty Galois keys");
    }

    const auto &level_info =
        params_.get_level(source_ciphertext.meta.parms_id);
    const auto source_view = source_ciphertext.make_const_view();
    const auto &source_shard0 = source_view.polys[0].shards.front();
    const auto &source_shard1 = source_view.polys[1].shards.front();
    const GpuParameterShard *parameter_shard = nullptr;
    for (const auto &candidate : level_info.shards)
    {
        if (candidate.device_id == source_shard0.device_id &&
            candidate.hybrid_base_q_count ==
                source_ciphertext.meta.q_count)
        {
            parameter_shard = &candidate;
            break;
        }
    }
    if (parameter_shard == nullptr ||
        parameter_shard->hybrid_base_p_count == 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist: "
            "HYBRID parameter shard is absent");
    }

    const std::size_t degree = source_ciphertext.meta.degree;
    const std::size_t q_count = source_ciphertext.meta.q_count;
    const std::size_t p_count =
        parameter_shard->hybrid_base_p_count;
    const std::size_t dnum = (q_count + p_count - 1) / p_count;
    const std::size_t group_count = matrix.plan.giant_steps.size();
    const std::size_t baby_count = matrix.plan.baby_steps.size();
    const bool requested_batched_giant_intt =
        use_double_hoist_batched_giant_intt() &&
        degree == 65536 &&
        group_count > 1;
    std::size_t requested_tile = workspace.baby_tile_size;
    if (const char *raw = std::getenv("POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE"))
    {
        const auto parsed = std::strtoull(raw, nullptr, 10);
        if (parsed != 0)
        {
            requested_tile = static_cast<std::size_t>(parsed);
        }
    }
    if (dnum == 1)
    {
        if (const char *raw = std::getenv(
                "POSEIDON_GPU_DOUBLE_HOIST_DNUM1_BABY_TILE"))
        {
            const auto parsed = std::strtoull(raw, nullptr, 10);
            if (parsed != 0)
            {
                requested_tile = static_cast<std::size_t>(parsed);
            }
        }
        else
        {
            requested_tile = 8;
        }
    }
    requested_tile = std::max<std::size_t>(
        1,
        std::min(requested_tile, baby_count));
    workspace.baby_tile_size = requested_tile;

    const std::size_t qp_poly_bytes =
        (q_count + p_count) * degree * sizeof(GpuWord);
    const std::size_t estimated_workspace =
        dnum * qp_poly_bytes +
        requested_tile * 2 * qp_poly_bytes +
        group_count * 2 * qp_poly_bytes +
        2 * qp_poly_bytes +
        (4 * q_count + 4 * p_count) * degree * sizeof(GpuWord) +
        (requested_batched_giant_intt
             ? (group_count - 1) * q_count * degree * sizeof(GpuWord)
             : 0);
    std::size_t max_workspace_bytes = workspace.max_workspace_bytes;
    if (const char *raw =
            std::getenv("POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB"))
    {
        const auto parsed = std::strtoull(raw, nullptr, 10);
        if (parsed != 0)
        {
            max_workspace_bytes =
                static_cast<std::size_t>(parsed) * 1024 * 1024;
        }
    }
    if (max_workspace_bytes != 0 &&
        estimated_workspace > max_workspace_bytes)
    {
        throw std::runtime_error(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist: "
            "estimated workspace exceeds POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB");
    }

    if (baby_cache != nullptr &&
        (!baby_cache->ready || baby_cache->source_c0 != source_shard0.ptr ||
         baby_cache->source_c1 != source_shard1.ptr ||
         baby_cache->source_meta.parms_id != source_ciphertext.meta.parms_id ||
         baby_cache->source_meta.scale != source_ciphertext.meta.scale ||
         baby_cache->source_meta.degree != degree || baby_cache->source_meta.q_count != q_count ||
         baby_cache->galois_key_identity != &galois_keys || baby_cache->steps != matrix.plan.baby_steps ||
         baby_cache->rotations.device_id != source_shard0.device_id || baby_cache->rotations.degree != degree ||
         baby_cache->rotations.q.size() < baby_count * 2 * q_count * degree ||
         baby_cache->rotations.p.size() < baby_count * 2 * p_count * degree ||
         baby_cache->rotations.q_count != q_count || baby_cache->rotations.p_count != p_count ||
         baby_cache->rotations.batch_count != baby_count))
        throw std::invalid_argument("double-hoist baby cache does not match immutable source, keys or matrix steps");
    const bool accumulate_existing = sum_state && sum_state->initialized;
    if (accumulate_existing &&
        (sum_state->source_meta.parms_id != source_ciphertext.meta.parms_id ||
         sum_state->source_meta.scale != source_ciphertext.meta.scale ||
         sum_state->source_meta.degree != degree || sum_state->source_meta.q_count != q_count ||
         sum_state->plaintext_scale != matrix.scale || sum_state->galois_key_identity != &galois_keys ||
         sum_state->giant_steps != matrix.plan.giant_steps ||
         sum_state->groups.device_id != source_shard0.device_id || sum_state->groups.degree != degree ||
         sum_state->groups.q.size() < group_count * 2 * q_count * degree ||
         sum_state->groups.p.size() < group_count * 2 * p_count * degree ||
         sum_state->groups.q_count != q_count || sum_state->groups.p_count != p_count ||
         sum_state->groups.batch_count != group_count))
        throw std::invalid_argument("double-hoist sum physical contract or giant schedule mismatch");
    if (sum_state)std::swap(workspace.group_accumulators, sum_state->groups);
    workspace.last_counts = {};
    workspace.last_counts.workspace_peak_bytes = estimated_workspace;
    workspace.group_accumulators.ensure_capacity(
        source_shard0.device_id,
        degree,
        q_count,
        p_count,
        group_count);
    const bool direct_qp_mac_init =
        use_double_hoist_qp_mac_direct_init() && !accumulate_existing;
    if (!direct_qp_mac_init && !accumulate_existing)
    {
        workspace.group_accumulators.q.fill_zero();
        workspace.group_accumulators.p.fill_zero();
    }

    if (baby_cache == nullptr)
    {
        keyswitch_handler_.hoist_decompose_modup_ntt(
            source_view.polys[1], level_info, workspace.source_hoist, workspace.keyswitch);
        workspace.last_counts.source_decompose_count = 1;
    }

    const auto &galois_keys_view =
        galois_keys.const_view(q_count);
    const bool fused_baby_keyswitch_c0 =
        use_double_hoist_fused_baby_keyswitch_c0() &&
        dnum == 1 &&
        galois_keys.meta.galois_format ==
            GpuGaloisKeyFormat::InversePreRotated;
    bool fused_baby_keyswitch_plain_mac =
        use_double_hoist_fused_baby_keyswitch_plain_mac() &&
        degree == 65536 &&
        group_count <= kernel::kMaxDoubleHoistFusedBabyGroups &&
        galois_keys.meta.galois_format ==
            GpuGaloisKeyFormat::InversePreRotated;
    const std::size_t fused_baby_tile = [&]() {
        if (const char *raw =
                std::getenv("POSEIDON_GPU_DOUBLE_HOIST_FUSED_BABY_TILE"))
        {
            const auto parsed = std::strtoull(raw, nullptr, 10);
            if (parsed != 0)
            {
                return std::min<std::size_t>(
                    baby_count,
                    std::min<std::size_t>(
                        static_cast<std::size_t>(parsed),
                        kernel::kMaxDoubleHoistFusedBabySteps));
            }
        }
        return std::min(baby_count, kernel::kMaxDoubleHoistFusedBabySteps);
    }();
    if (fused_baby_keyswitch_plain_mac)
    {
        std::vector<bool> term_seen(group_count * baby_count, false);
        for (const auto &term : matrix.plan.terms)
        {
            if (term.giant_index >= group_count ||
                term.baby_index >= baby_count)
            {
                fused_baby_keyswitch_plain_mac = false;
                break;
            }
            const std::size_t slot =
                static_cast<std::size_t>(term.giant_index) * baby_count +
                term.baby_index;
            if (term_seen[slot])
            {
                fused_baby_keyswitch_plain_mac = false;
                break;
            }
            term_seen[slot] = true;
        }
    }
    const std::size_t step_tile =
        (baby_cache == nullptr && fused_baby_keyswitch_plain_mac)
            ? fused_baby_tile
            : requested_tile;
    for (std::size_t tile_begin = 0;
         tile_begin < baby_count;
         tile_begin += step_tile)
    {
        const std::size_t tile_count =
            std::min(step_tile, baby_count - tile_begin);
        ++workspace.last_counts.baby_tile_count;
        if (baby_cache != nullptr)
        {
            kernel::launch_double_hoist_qp_plain_mul_accumulate_groups(
                workspace.group_accumulators.q.data(), workspace.group_accumulators.p.data(),
                baby_cache->rotations.q_component(tile_begin, 0), baby_cache->rotations.p_component(tile_begin, 0),
                matrix.plan.diagonal_q_ptrs.data(), matrix.plan.diagonal_p_ptrs.data(),
                matrix.plan.diagonal_periods.data(), matrix.plan.term_baby_indices.data(),
                matrix.plan.group_term_offsets_device.data(), group_count, matrix.plan.terms.size(),
                tile_begin, tile_count, *parameter_shard, degree, matrix.plan.compressed_plaintexts,
                direct_qp_mac_init && tile_begin == 0);
            continue;
        }
        if (fused_baby_keyswitch_plain_mac)
        {
            std::array<
                std::uint32_t,
                kernel::kMaxDoubleHoistFusedBabySteps> galois_elts{};
            std::array<
                std::uint32_t,
                kernel::kMaxDoubleHoistFusedBabySteps> key_indices{};
            std::array<
                std::uint32_t,
                kernel::kMaxDoubleHoistFusedBabyGroups *
                    kernel::kMaxDoubleHoistFusedBabySteps> term_indices{};
            term_indices.fill(std::numeric_limits<std::uint32_t>::max());

            for (std::size_t local = 0; local < tile_count; ++local)
            {
                const int step =
                    matrix.plan.baby_steps[tile_begin + local];
                if (step == 0)
                {
                    continue;
                }
                const std::uint32_t galois_elt =
                    galois_elt_from_rotation_step(degree, step);
                const std::size_t key_index =
                    galois_key_index(galois_elt);
                if (key_index > std::numeric_limits<std::uint32_t>::max() ||
                    galois_keys.galois_elts_by_key_index.size() <= key_index ||
                    galois_keys.galois_elts_by_key_index[key_index] !=
                        galois_elt)
                {
                    throw std::invalid_argument(
                        "fused baby KeySwitch/plain MAC key mismatch");
                }
                const std::size_t pointer_end =
                    (key_index + 1) *
                    galois_keys.meta.decomposition_count;
                if (galois_keys.galois_key_q0_ptrs.size() < pointer_end ||
                    galois_keys.galois_key_p0_ptrs.size() < pointer_end ||
                    galois_keys.galois_key_q1_ptrs.size() < pointer_end ||
                    galois_keys.galois_key_p1_ptrs.size() < pointer_end)
                {
                    throw std::invalid_argument(
                        "fused baby KeySwitch/plain MAC pointer table mismatch");
                }
                galois_elts[local] = galois_elt;
                key_indices[local] =
                    static_cast<std::uint32_t>(key_index);
                ++workspace.last_counts.keymul_count;
                ++workspace.last_counts.permute_count;
            }
            for (std::size_t term_index = 0;
                 term_index < matrix.plan.terms.size();
                 ++term_index)
            {
                const auto &term = matrix.plan.terms[term_index];
                if (term.baby_index < tile_begin ||
                    term.baby_index >= tile_begin + tile_count)
                {
                    continue;
                }
                const std::size_t local = term.baby_index - tile_begin;
                term_indices[
                    static_cast<std::size_t>(term.giant_index) *
                            kernel::kMaxDoubleHoistFusedBabySteps +
                        local] =
                        static_cast<std::uint32_t>(term_index);
            }
            kernel::launch_double_hoist_fused_baby_keyswitch_plain_accumulate(
                workspace.group_accumulators.q.data(),
                workspace.group_accumulators.p.data(),
                workspace.source_hoist.digits_q_ntt.data(),
                workspace.source_hoist.digits_p_ntt.data(),
                source_shard0.ptr,
                source_shard1.ptr,
                galois_elts.data(),
                key_indices.data(),
                term_indices.data(),
                galois_keys.galois_key_q0_ptrs.data(),
                galois_keys.galois_key_p0_ptrs.data(),
                galois_keys.galois_key_q1_ptrs.data(),
                galois_keys.galois_key_p1_ptrs.data(),
                matrix.plan.diagonal_q_ptrs.data(),
                matrix.plan.diagonal_p_ptrs.data(),
                matrix.plan.diagonal_periods.data(),
                group_count,
                matrix.plan.terms.size(),
                tile_count,
                dnum,
                galois_keys.meta.decomposition_count,
                *parameter_shard,
                degree,
                matrix.plan.compressed_plaintexts,
                direct_qp_mac_init && tile_begin == 0);
            continue;
        }
        workspace.baby_tile.ensure_capacity(
            source_shard0.device_id,
            degree,
            q_count,
            p_count,
            tile_count);

        for (std::size_t local = 0; local < tile_count; ++local)
        {
            const int step =
                matrix.plan.baby_steps[tile_begin + local];
            if (step == 0)
            {
                kernel::launch_double_hoist_lift_identity(
                    workspace.baby_tile.q_component(local, 0),
                    workspace.baby_tile.q_component(local, 1),
                    workspace.baby_tile.p_component(local, 0),
                    workspace.baby_tile.p_component(local, 1),
                    source_shard0.ptr,
                    source_shard1.ptr,
                    *parameter_shard,
                    degree,
                    false);
                continue;
            }

            const std::uint32_t galois_elt =
                galois_elt_from_rotation_step(degree, step);
            const std::size_t key_index =
                galois_key_index(galois_elt);
            keyswitch_handler_.keyswitch_multsum_no_moddown(
                workspace.source_hoist,
                galois_elt,
                galois_keys_view,
                galois_keys,
                key_index,
                workspace.baby_tile,
                local,
                true,
                level_info,
                workspace.keyswitch,
                fused_baby_keyswitch_c0
                    ? source_shard0.ptr
                    : nullptr);
            if (!fused_baby_keyswitch_c0)
            {
                kernel::launch_double_hoist_add_lifted_galois_c0(
                    workspace.baby_tile.q_component(local, 0),
                    source_shard0.ptr,
                    galois_elt,
                    *parameter_shard,
                    degree);
            }
            ++workspace.last_counts.keymul_count;
            workspace.last_counts.permute_count +=
                galois_keys.meta.galois_format ==
                        GpuGaloisKeyFormat::InversePreRotated
                    ? 1
                    : 2 * workspace.source_hoist.dnum + 1;
        }

        kernel::launch_double_hoist_qp_plain_mul_accumulate_groups(
            workspace.group_accumulators.q.data(),
            workspace.group_accumulators.p.data(),
            workspace.baby_tile.q.data(),
            workspace.baby_tile.p.data(),
            matrix.plan.diagonal_q_ptrs.data(),
            matrix.plan.diagonal_p_ptrs.data(),
            matrix.plan.diagonal_periods.data(),
            matrix.plan.term_baby_indices.data(),
            matrix.plan.group_term_offsets_device.data(),
            group_count,
            matrix.plan.terms.size(),
            tile_begin,
            tile_count,
            *parameter_shard,
            degree,
            matrix.plan.compressed_plaintexts,
            direct_qp_mac_init && tile_begin == 0);
    }
    workspace.last_counts.qp_pmult_count = matrix.plan.terms.size();
    if (sum_state)
    {
        if (!final_input)
        {
            sum_state->source_meta = source_ciphertext.meta;sum_state->plaintext_scale = matrix.scale;
            sum_state->galois_key_identity = &galois_keys;sum_state->giant_steps = matrix.plan.giant_steps;
            sum_state->initialized = true;
            std::swap(workspace.group_accumulators, sum_state->groups);
            return;
        }
        sum_state->initialized = false;
    }

    GpuCiphertextMeta product_meta = source_ciphertext.meta;
    product_meta.scale *= matrix.scale;
    bool have_outer = false;
    const bool grouped_outer_moddown =
        galois_keys.meta.galois_format ==
        GpuGaloisKeyFormat::InversePreRotated;
    const auto identity_it = std::find(
        matrix.plan.giant_steps.begin(),
        matrix.plan.giant_steps.end(),
        0);
    const bool direct_giant_accumulate =
        grouped_outer_moddown &&
        use_double_hoist_direct_giant_accumulate() &&
        identity_it != matrix.plan.giant_steps.end() &&
        group_count > 1;
    const std::size_t identity_group_index =
        identity_it == matrix.plan.giant_steps.end()
            ? group_count
            : static_cast<std::size_t>(
                  identity_it - matrix.plan.giant_steps.begin());
    const bool direct_identity_only =
        grouped_outer_moddown &&
        group_count == 1 &&
        identity_group_index == 0;
    if (direct_identity_only)
    {
        /*
         * A direct diagonal plan places every term in the identity giant
         * group. Its QP plaintext-MAC accumulator is already the final
         * linear-transform result, so a single ModDown is sufficient. Do not
         * run the generic identity path, which would ModDown, lift back to QP,
         * and ModDown a second time only to satisfy the outer BSGS pipeline.
         */
        keyswitch_handler_.moddown_qp_ciphertext_to_q(
            workspace.group_accumulators,
            0,
            workspace.result_q,
            product_meta,
            level_info,
            workspace.keyswitch);
        workspace.last_counts.outer_moddown_count = 1;
        if (rescale_count == 0)
        {
            destination_ciphertext = std::move(workspace.result_q);
        }
        else
        {
            rescale_many(
                workspace.result_q,
                destination_ciphertext,
                rescale_count);
        }
        return;
    }
    const bool qp_c0 = direct_giant_accumulate && use_double_hoist_qp_c0();
    if (grouped_outer_moddown)
    {
        if (qp_c0)
        {
            // Only c1 of the non-identity groups is decomposed for the giant
            // key switch; everything else stays in QP.
            std::vector<std::uint32_t> c1_groups;
            c1_groups.reserve(group_count);
            for (std::size_t group = 0; group < group_count; ++group)
            {
                if (matrix.plan.giant_steps[group] != 0)
                {
                    c1_groups.push_back(static_cast<std::uint32_t>(group));
                }
            }
            keyswitch_handler_.moddown_qp_c1_groups_to_q(
                workspace.group_accumulators,
                c1_groups,
                group_count,
                workspace.inner_q_batch,
                level_info,
                workspace.batch_p_coeff,
                workspace.batch_converted_q);
            workspace.last_counts.inner_moddown_count += c1_groups.size();
            workspace.last_counts.inner_moddown_component_count += c1_groups.size();
        }
        else
        {
            /*
             * All group accumulators have the same shape. Batch their P-side
             * INTT/P->Q/forward-NTT/ModDown instead of completing one full
             * pipeline before launching the next group.
             */
            keyswitch_handler_.moddown_qp_ciphertext_batch_to_q(
                workspace.group_accumulators,
                group_count,
                workspace.inner_q_batch,
                level_info,
                workspace.batch_p_coeff,
                workspace.batch_converted_q);
            workspace.last_counts.inner_moddown_count += group_count;
            workspace.last_counts.inner_moddown_component_count += 2 * group_count;
        }

        bool batched_giant_intt =
            requested_batched_giant_intt &&
            identity_group_index == 0;
        for (std::size_t group = 1;
             batched_giant_intt && group < group_count;
             ++group)
        {
            batched_giant_intt = matrix.plan.giant_steps[group] != 0;
        }
        const std::size_t batched_giant_count =
            batched_giant_intt ? group_count - 1 : 0;
        const std::size_t q_words = q_count * degree;
        if (batched_giant_intt)
        {
            const std::size_t required_words =
                batched_giant_count * q_words;
            if (workspace.outer_source_intt_q_batch.device_id() !=
                    source_shard0.device_id ||
                workspace.outer_source_intt_q_batch.size() < required_words)
            {
                workspace.outer_source_intt_q_batch.allocate(
                    required_words,
                    source_shard0.device_id);
            }

            GpuPolyShardView destination_intt;
            destination_intt.device_id = source_shard0.device_id;
            destination_intt.ptr =
                workspace.outer_source_intt_q_batch.data();
            destination_intt.limb_begin = 0;
            destination_intt.limb_count = q_count;
            destination_intt.coeff_begin = 0;
            destination_intt.coeff_count = degree;

            GpuConstPolyShardView source_ntt;
            source_ntt.device_id = source_shard0.device_id;
            source_ntt.ptr =
                workspace.inner_q_batch.q_component(1, 1);
            source_ntt.limb_begin = 0;
            source_ntt.limb_count = q_count;
            source_ntt.coeff_begin = 0;
            source_ntt.coeff_count = degree;

            kernel::launch_inverse_ntt_poly_shard_batch_fourstep_65536(
                destination_intt,
                source_ntt,
                *parameter_shard,
                degree,
                batched_giant_count,
                q_words,
                2 * q_words);
        }

        std::vector<const GpuWord *> digit_q_ptrs;
        std::vector<const GpuWord *> digit_p_ptrs;
        std::vector<std::uint32_t> group_indices;
        std::vector<std::uint32_t> galois_elts;
        std::vector<std::uint32_t> key_indices;
        for (std::size_t group = 0; group < group_count; ++group)
        {
            if (matrix.plan.giant_steps[group] == 0)
            {
                if (!direct_giant_accumulate)
                {
                    kernel::launch_double_hoist_lift_identity(
                        workspace.group_accumulators.q_component(group, 0),
                        workspace.group_accumulators.q_component(group, 1),
                        workspace.group_accumulators.p_component(group, 0),
                        workspace.group_accumulators.p_component(group, 1),
                        workspace.inner_q_batch.q_component(group, 0),
                        workspace.inner_q_batch.q_component(group, 1),
                        *parameter_shard,
                        degree,
                        false);
                }
                have_outer = true;
            }
        }
        collect_and_hoist_giant_groups(
            keyswitch_handler_,
            level_info,
            workspace,
            source_shard0.device_id,
            degree,
            q_count,
            matrix.plan.giant_steps,
            workspace.inner_q_batch.q.data(),
            2 * q_words,
            batched_giant_intt,
            digit_q_ptrs,
            digit_p_ptrs,
            group_indices,
            galois_elts,
            key_indices,
            &workspace.last_counts.outer_decompose_count);

        const std::size_t active_group_count = group_indices.size();
        if (active_group_count != 0)
        {
            if (qp_c0)
            {
                workspace.outer_accumulator.ensure_capacity(
                    source_shard0.device_id,
                    degree,
                    q_count,
                    p_count,
                    1);
                kernel::launch_double_hoist_pre_rotated_giant_group_accumulate_qp_c0(
                    workspace.outer_accumulator.q_component(0, 0),
                    workspace.outer_accumulator.q_component(0, 1),
                    workspace.outer_accumulator.p_component(0, 0),
                    workspace.outer_accumulator.p_component(0, 1),
                    workspace.group_accumulators.q.data(),
                    workspace.group_accumulators.p.data(),
                    identity_group_index,
                    digit_q_ptrs.data(),
                    digit_p_ptrs.data(),
                    group_indices.data(),
                    galois_elts.data(),
                    key_indices.data(),
                    galois_keys.galois_key_q0_ptrs.data(),
                    galois_keys.galois_key_p0_ptrs.data(),
                    galois_keys.galois_key_q1_ptrs.data(),
                    galois_keys.galois_key_p1_ptrs.data(),
                    active_group_count,
                    dnum,
                    galois_keys.meta.decomposition_count,
                    *parameter_shard,
                    degree);
            }
            else if (direct_giant_accumulate)
            {
                workspace.outer_accumulator.ensure_capacity(
                    source_shard0.device_id,
                    degree,
                    q_count,
                    p_count,
                    1);
                kernel::launch_double_hoist_pre_rotated_giant_group_accumulate(
                    workspace.outer_accumulator.q_component(0, 0),
                    workspace.outer_accumulator.q_component(0, 1),
                    workspace.outer_accumulator.p_component(0, 0),
                    workspace.outer_accumulator.p_component(0, 1),
                    workspace.inner_q_batch.q.data(),
                    identity_group_index,
                    digit_q_ptrs.data(),
                    digit_p_ptrs.data(),
                    group_indices.data(),
                    galois_elts.data(),
                    key_indices.data(),
                    galois_keys.galois_key_q0_ptrs.data(),
                    galois_keys.galois_key_p0_ptrs.data(),
                    galois_keys.galois_key_q1_ptrs.data(),
                    galois_keys.galois_key_p1_ptrs.data(),
                    active_group_count,
                    dnum,
                    galois_keys.meta.decomposition_count,
                    *parameter_shard,
                    degree);
            }
            else
            {
                kernel::launch_double_hoist_pre_rotated_giant_group_reduce(
                    workspace.group_accumulators.q.data(),
                    workspace.group_accumulators.p.data(),
                    workspace.inner_q_batch.q.data(),
                    digit_q_ptrs.data(),
                    digit_p_ptrs.data(),
                    group_indices.data(),
                    galois_elts.data(),
                    key_indices.data(),
                    galois_keys.galois_key_q0_ptrs.data(),
                    galois_keys.galois_key_p0_ptrs.data(),
                    galois_keys.galois_key_q1_ptrs.data(),
                    galois_keys.galois_key_p1_ptrs.data(),
                    active_group_count,
                    dnum,
                    galois_keys.meta.decomposition_count,
                    *parameter_shard,
                    degree);
            }
            have_outer = true;
            workspace.last_counts.keymul_count += active_group_count;
            workspace.last_counts.permute_count += active_group_count;
        }
    }
    else
    {
        workspace.outer_accumulator.ensure_capacity(
            source_shard0.device_id,
            degree,
            q_count,
            p_count,
            1);
        workspace.outer_accumulator.q.fill_zero();
        workspace.outer_accumulator.p.fill_zero();
        for (std::size_t group = 0; group < group_count; ++group)
        {
            keyswitch_handler_.moddown_qp_ciphertext_to_q(
                workspace.group_accumulators,
                group,
                workspace.inner_q,
                product_meta,
                level_info,
                workspace.keyswitch);
            ++workspace.last_counts.inner_moddown_count;
            workspace.last_counts.inner_moddown_component_count += 2;

            const auto inner_view = workspace.inner_q.make_const_view();
            const auto &inner0 = inner_view.polys[0].shards.front();
            const auto &inner1 = inner_view.polys[1].shards.front();
            const int giant_step = matrix.plan.giant_steps[group];
            if (giant_step == 0)
            {
                kernel::launch_double_hoist_lift_identity(
                    workspace.outer_accumulator.q_component(0, 0),
                    workspace.outer_accumulator.q_component(0, 1),
                    workspace.outer_accumulator.p_component(0, 0),
                    workspace.outer_accumulator.p_component(0, 1),
                    inner0.ptr,
                    inner1.ptr,
                    *parameter_shard,
                    degree,
                    have_outer);
                have_outer = true;
                continue;
            }

            keyswitch_handler_.hoist_decompose_modup_ntt(
                inner_view.polys[1],
                level_info,
                workspace.outer_hoist,
                workspace.keyswitch);
            ++workspace.last_counts.outer_decompose_count;
            const std::uint32_t galois_elt =
                galois_elt_from_rotation_step(degree, giant_step);
            const std::size_t key_index =
                galois_key_index(galois_elt);
            keyswitch_handler_.keyswitch_multsum_no_moddown(
                workspace.outer_hoist,
                galois_elt,
                galois_keys_view,
                galois_keys,
                key_index,
                workspace.outer_accumulator,
                0,
                !have_outer,
                level_info,
                workspace.keyswitch);
            kernel::launch_double_hoist_add_lifted_galois_c0(
                workspace.outer_accumulator.q_component(0, 0),
                inner0.ptr,
                galois_elt,
                *parameter_shard,
                degree);
            have_outer = true;
            ++workspace.last_counts.keymul_count;
            workspace.last_counts.permute_count +=
                2 * workspace.outer_hoist.dnum + 1;
        }
    }
    if (!have_outer)
    {
        throw std::logic_error(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist: "
            "outer accumulator is empty");
    }

    if (direct_giant_accumulate)
    {
        keyswitch_handler_.moddown_qp_ciphertext_to_q(
            workspace.outer_accumulator,
            0,
            workspace.result_q,
            product_meta,
            level_info,
            workspace.keyswitch);
    }
    else if (grouped_outer_moddown)
    {
        keyswitch_handler_.moddown_qp_groups_to_q(
            workspace.group_accumulators,
            group_count,
            workspace.outer_reduced_p,
            workspace.result_q,
            product_meta,
            level_info,
            workspace.keyswitch);
    }
    else
    {
        keyswitch_handler_.moddown_qp_ciphertext_to_q(
            workspace.outer_accumulator,
            0,
            workspace.result_q,
            product_meta,
            level_info,
            workspace.keyswitch);
    }
    workspace.last_counts.outer_moddown_count = 1;
    if (rescale_count == 0)
    {
        destination_ciphertext = std::move(workspace.result_q);
    }
    else
    {
        rescale_many(
            workspace.result_q,
            destination_ciphertext,
            rescale_count);
    }
}

void GpuEvaluator::dft(
    const GpuCiphertextData &source_ciphertext,
    const GpuLinearMatrixGroup &matrix_group,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &destination_ciphertext) const
{
    if (matrix_group.data().empty())
    {
        throw std::invalid_argument("GpuEvaluator::dft: empty matrix group");
    }

    const bool dynamic_rescale = matrix_group.rescale_min_scale() > 0.0;
    if (dynamic_rescale &&
        matrix_group.rescale_counts().size() != matrix_group.data().size())
    {
        throw std::invalid_argument(
            "GpuEvaluator::dft: dynamic rescale plan size mismatch");
    }

    GpuCiphertextData product;
    GpuCiphertextData current;
    multiply_by_diag_matrix_bsgs(
        source_ciphertext,
        matrix_group.data().front(),
        galois_keys,
        dynamic_rescale ? 0 : std::max(matrix_group.step(), std::uint32_t{1}),
        dynamic_rescale ? product : current);
    if (dynamic_rescale)
    {
        rescale_dynamic(
            product,
            current,
            matrix_group.rescale_min_scale());
    }

    for (std::size_t i = 1; i < matrix_group.data().size(); ++i)
    {
        GpuCiphertextData next_product;
        GpuCiphertextData next;
        multiply_by_diag_matrix_bsgs(
            current,
            matrix_group.data()[i],
            galois_keys,
            dynamic_rescale ? 0 : std::max(matrix_group.step(), std::uint32_t{1}),
            dynamic_rescale ? next_product : next);
        if (dynamic_rescale)
        {
            rescale_dynamic(
                next_product,
                next,
                matrix_group.rescale_min_scale());
        }
        current = std::move(next);
    }

    destination_ciphertext = std::move(current);
}

void GpuEvaluator::dft_loose(
    const GpuCiphertextData &source_ciphertext,
    const GpuLinearMatrixGroup &matrix_group,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &destination_ciphertext) const
{
    if (matrix_group.data().empty())
    {
        throw std::invalid_argument("GpuEvaluator::dft_loose: empty matrix group");
    }

    const bool dynamic_rescale = matrix_group.rescale_min_scale() > 0.0;
    if (dynamic_rescale &&
        matrix_group.rescale_counts().size() != matrix_group.data().size())
    {
        throw std::invalid_argument(
            "GpuEvaluator::dft_loose: dynamic rescale plan size mismatch");
    }

    GpuCiphertextData product;
    GpuCiphertextData current;
    multiply_by_diag_matrix_bsgs_loose(
        source_ciphertext,
        matrix_group.data().front(),
        galois_keys,
        dynamic_rescale ? 0 : std::max(matrix_group.step(), std::uint32_t{1}),
        dynamic_rescale ? product : current);
    if (dynamic_rescale)
    {
        rescale_dynamic(
            product,
            current,
            matrix_group.rescale_min_scale());
    }

    for (std::size_t i = 1; i < matrix_group.data().size(); ++i)
    {
        GpuCiphertextData next_product;
        GpuCiphertextData next;
        multiply_by_diag_matrix_bsgs_loose(
            current,
            matrix_group.data()[i],
            galois_keys,
            dynamic_rescale ? 0 : std::max(matrix_group.step(), std::uint32_t{1}),
            dynamic_rescale ? next_product : next);
        if (dynamic_rescale)
        {
            rescale_dynamic(
                next_product,
                next,
                matrix_group.rescale_min_scale());
        }
        current = std::move(next);
    }

    destination_ciphertext = std::move(current);
}

void GpuEvaluator::giant_reduce_q_groups(
    const GpuWord *inner_q_batch,
    std::size_t q_count,
    parms_id_type parms_id,
    double product_scale,
    const std::vector<int> &giant_steps,
    const GpuGaloisKeysData &galois_keys_in,
    std::uint32_t rescale_count,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    if (inner_q_batch == nullptr || q_count == 0 || giant_steps.size() < 2)
    {
        throw std::invalid_argument("GpuEvaluator::giant_reduce_q_groups: invalid inputs");
    }
    const auto &level_info = params_.get_level(parms_id);
    const GpuParameterShard *parameter_shard = nullptr;
    for (const auto &candidate : level_info.shards)
    {
        if (candidate.hybrid_base_q_count == q_count)
        {
            parameter_shard = &candidate;
            break;
        }
    }
    if (parameter_shard == nullptr || parameter_shard->hybrid_base_p_count == 0)
    {
        throw std::invalid_argument("GpuEvaluator::giant_reduce_q_groups: HYBRID parameter shard absent");
    }
    const int device_id = parameter_shard->device_id;
    const std::size_t degree = level_info.degree;
    const std::size_t p_count = parameter_shard->hybrid_base_p_count;
    const std::size_t dnum = (q_count + p_count - 1) / p_count;
    const std::size_t group_count = giant_steps.size();

    GpuCiphertextMeta product_meta;
    product_meta.parms_id = parms_id;
    product_meta.scale = product_scale;
    product_meta.is_ntt_form = true;
    product_meta.degree = degree;
    product_meta.q_count = q_count;
    product_meta.p_count = 0;
    product_meta.component_count = 2;

    const auto &galois_keys = level_key_for(
        params_, galois_keys_in, product_meta, "GpuEvaluator::giant_reduce_q_groups");
    if (galois_keys.empty() ||
        galois_keys.meta.galois_format != GpuGaloisKeyFormat::InversePreRotated)
    {
        throw std::invalid_argument(
            "GpuEvaluator::giant_reduce_q_groups: requires InversePreRotated Galois keys");
    }
    const auto &galois_keys_view = galois_keys.const_view(q_count);
    (void)galois_keys_view;

    const auto identity_it = std::find(giant_steps.begin(), giant_steps.end(), 0);
    if (identity_it != giant_steps.begin())
    {
        throw std::invalid_argument(
            "GpuEvaluator::giant_reduce_q_groups: identity step 0 must be at group index 0");
    }
    const std::size_t q_words = q_count * degree;
    const std::size_t batched_giant_count = group_count - 1;
    const bool batched_giant_intt =
        use_double_hoist_batched_giant_intt() && degree == 65536;
    if (batched_giant_intt)
    {
        const std::size_t required_words = batched_giant_count * q_words;
        if (workspace.outer_source_intt_q_batch.device_id() != device_id ||
            workspace.outer_source_intt_q_batch.size() < required_words)
        {
            workspace.outer_source_intt_q_batch.allocate(required_words, device_id);
        }
        GpuPolyShardView destination_intt;
        destination_intt.device_id = device_id;
        destination_intt.ptr = workspace.outer_source_intt_q_batch.data();
        destination_intt.limb_begin = 0;
        destination_intt.limb_count = q_count;
        destination_intt.coeff_begin = 0;
        destination_intt.coeff_count = degree;

        GpuConstPolyShardView source_ntt;
        source_ntt.device_id = device_id;
        source_ntt.ptr = inner_q_batch + 3 * q_words;
        source_ntt.limb_begin = 0;
        source_ntt.limb_count = q_count;
        source_ntt.coeff_begin = 0;
        source_ntt.coeff_count = degree;

        kernel::launch_inverse_ntt_poly_shard_batch_fourstep_65536(
            destination_intt, source_ntt, *parameter_shard,
            degree, batched_giant_count, q_words, 2 * q_words);
    }

    for (std::size_t group = 1; group < group_count; ++group)
    {
        if (giant_steps[group] == 0)
        {
            throw std::invalid_argument(
                "GpuEvaluator::giant_reduce_q_groups: duplicate identity step 0");
        }
    }
    std::vector<const GpuWord *> digit_q_ptrs;
    std::vector<const GpuWord *> digit_p_ptrs;
    std::vector<std::uint32_t> group_indices;
    std::vector<std::uint32_t> galois_elts;
    std::vector<std::uint32_t> key_indices;
    collect_and_hoist_giant_groups(
        keyswitch_handler_,
        level_info,
        workspace,
        device_id,
        degree,
        q_count,
        giant_steps,
        inner_q_batch,
        2 * q_words,
        batched_giant_intt,
        digit_q_ptrs,
        digit_p_ptrs,
        group_indices,
        galois_elts,
        key_indices);

    workspace.outer_accumulator.ensure_capacity(device_id, degree, q_count, p_count, 1);
    kernel::launch_double_hoist_pre_rotated_giant_group_accumulate(
        workspace.outer_accumulator.q_component(0, 0),
        workspace.outer_accumulator.q_component(0, 1),
        workspace.outer_accumulator.p_component(0, 0),
        workspace.outer_accumulator.p_component(0, 1),
        inner_q_batch,
        0,
        digit_q_ptrs.data(),
        digit_p_ptrs.data(),
        group_indices.data(),
        galois_elts.data(),
        key_indices.data(),
        galois_keys.galois_key_q0_ptrs.data(),
        galois_keys.galois_key_p0_ptrs.data(),
        galois_keys.galois_key_q1_ptrs.data(),
        galois_keys.galois_key_p1_ptrs.data(),
        group_indices.size(),
        dnum,
        galois_keys.meta.decomposition_count,
        *parameter_shard,
        degree);

    keyswitch_handler_.moddown_qp_ciphertext_to_q(
        workspace.outer_accumulator,
        0,
        workspace.result_q,
        product_meta,
        level_info,
        workspace.keyswitch);
    if (rescale_count == 0)
    {
        destination_ciphertext = std::move(workspace.result_q);
    }
    else
    {
        rescale_many(workspace.result_q, destination_ciphertext, rescale_count);
    }
}

void GpuEvaluator::giant_reduce_q_groups_batch(
    const GpuWord *inner_q_batch,
    std::size_t batch_count,
    std::size_t inner_batch_stride_words,
    std::size_t q_count,
    parms_id_type parms_id,
    double product_scale,
    const std::vector<int> &giant_steps,
    const GpuGaloisKeysData &galois_keys_in,
    std::uint32_t rescale_count,
    GpuDoubleHoistWorkspace &workspace,
    std::vector<GpuCiphertextData> &destination_ciphertexts) const
{
    if (batch_count == 0)
    {
        destination_ciphertexts.clear();
        return;
    }
    if (inner_q_batch == nullptr || q_count == 0 || giant_steps.size() < 2)
    {
        throw std::invalid_argument(
            "GpuEvaluator::giant_reduce_q_groups_batch: invalid inputs");
    }
    if (giant_steps[0] != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::giant_reduce_q_groups_batch: identity step 0 must be at group index 0");
    }
    for (std::size_t group = 1; group < giant_steps.size(); ++group)
    {
        if (giant_steps[group] == 0)
        {
            throw std::invalid_argument(
                "GpuEvaluator::giant_reduce_q_groups_batch: duplicate identity step 0");
        }
    }

    const auto &level_info = params_.get_level(parms_id);
    const GpuParameterShard *parameter_shard = nullptr;
    for (const auto &candidate : level_info.shards)
    {
        if (candidate.hybrid_base_q_count == q_count)
        {
            parameter_shard = &candidate;
            break;
        }
    }
    if (parameter_shard == nullptr || parameter_shard->hybrid_base_p_count == 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::giant_reduce_q_groups_batch: HYBRID parameter shard absent");
    }
    const std::size_t degree = level_info.degree;
    const std::size_t group_count = giant_steps.size();
    const std::size_t batched_giant_count = group_count - 1;
    const bool use_batch =
        batch_count >= 2 &&
        batch_count <= kernel::kMaxGiantReduceBatch &&
        batched_giant_count <= kernel::kMaxGiantReduceBatchGroups &&
        degree == 65536 &&
        use_double_hoist_batched_giant_intt() &&
        rescale_count <= 1 &&
        (rescale_count == 0 || q_count >= 2);
    if (!use_batch)
    {
        destination_ciphertexts.resize(batch_count);
        for (std::size_t b = 0; b < batch_count; ++b)
        {
            giant_reduce_q_groups(
                inner_q_batch + b * inner_batch_stride_words,
                q_count,
                parms_id,
                product_scale,
                giant_steps,
                galois_keys_in,
                rescale_count,
                workspace,
                destination_ciphertexts[b]);
        }
        return;
    }

    const int device_id = parameter_shard->device_id;
    const std::size_t p_count = parameter_shard->hybrid_base_p_count;
    const std::size_t dnum = (q_count + p_count - 1) / p_count;
    const std::size_t q_words = q_count * degree;

    GpuCiphertextMeta product_meta;
    product_meta.parms_id = parms_id;
    product_meta.scale = product_scale;
    product_meta.is_ntt_form = true;
    product_meta.degree = degree;
    product_meta.q_count = q_count;
    product_meta.p_count = 0;
    product_meta.component_count = 2;

    const auto &galois_keys = level_key_for(
        params_, galois_keys_in, product_meta,
        "GpuEvaluator::giant_reduce_q_groups_batch");
    if (galois_keys.empty() ||
        galois_keys.meta.galois_format != GpuGaloisKeyFormat::InversePreRotated)
    {
        throw std::invalid_argument(
            "GpuEvaluator::giant_reduce_q_groups_batch: requires InversePreRotated Galois keys");
    }

    const std::size_t total_hoists = batch_count * batched_giant_count;
    const std::size_t required_intt_words = total_hoists * q_words;
    if (workspace.outer_source_intt_q_batch.device_id() != device_id ||
        workspace.outer_source_intt_q_batch.size() < required_intt_words)
    {
        workspace.outer_source_intt_q_batch.allocate(required_intt_words, device_id);
    }
    for (std::size_t b = 0; b < batch_count; ++b)
    {
        GpuPolyShardView destination_intt{
            device_id,
            workspace.outer_source_intt_q_batch.data() + b * batched_giant_count * q_words,
            0, q_count, 0, degree};
        GpuConstPolyShardView source_ntt{
            device_id,
            inner_q_batch + b * inner_batch_stride_words + 3 * q_words,
            0, q_count, 0, degree};
        kernel::launch_inverse_ntt_poly_shard_batch_fourstep_65536(
            destination_intt, source_ntt, *parameter_shard,
            degree, batched_giant_count, q_words, 2 * q_words);
    }

    if (workspace.outer_group_hoists.size() < total_hoists)
    {
        workspace.outer_group_hoists.resize(total_hoists);
    }
    std::vector<GpuConstRNSPolyView> switch_polys_ntt;
    std::vector<GpuHoistedDecomposition *> destinations;
    switch_polys_ntt.reserve(total_hoists);
    destinations.reserve(total_hoists);

    std::vector<std::uint32_t> group_indices(batched_giant_count);
    std::vector<std::uint32_t> galois_elts(batched_giant_count);
    std::vector<std::uint32_t> key_indices(batched_giant_count);
    for (std::size_t g = 1; g < group_count; ++g)
    {
        const std::size_t active_idx = g - 1;
        const std::uint32_t galois_elt =
            galois_elt_from_rotation_step(degree, giant_steps[g]);
        const std::size_t key_index = galois_key_index(galois_elt);
        if (key_index > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::overflow_error(
                "double-hoist giant key index exceeds uint32_t");
        }
        group_indices[active_idx] = static_cast<std::uint32_t>(g);
        galois_elts[active_idx] = galois_elt;
        key_indices[active_idx] = static_cast<std::uint32_t>(key_index);
    }

    for (std::size_t b = 0; b < batch_count; ++b)
    {
        const GpuWord *batch_ptr =
            inner_q_batch + b * inner_batch_stride_words;
        for (std::size_t g = 1; g < group_count; ++g)
        {
            const std::size_t idx = b * batched_giant_count + (g - 1);
            GpuConstRNSPolyView inner_c1;
            inner_c1.poly_id = 1;
            inner_c1.shards.push_back(GpuConstPolyShardView{
                device_id,
                batch_ptr + g * 2 * q_words + q_words,
                0, q_count, 0, degree});
            switch_polys_ntt.push_back(std::move(inner_c1));
            destinations.push_back(&workspace.outer_group_hoists[idx]);
        }
    }

    keyswitch_handler_.hoist_decompose_modup_ntt_batch(
        switch_polys_ntt,
        level_info,
        destinations,
        workspace.keyswitch,
        workspace.outer_source_intt_q_batch.data(),
        q_words);

    std::vector<const GpuWord *> digit_q_ptrs(total_hoists);
    std::vector<const GpuWord *> digit_p_ptrs(total_hoists);
    for (std::size_t i = 0; i < total_hoists; ++i)
    {
        digit_q_ptrs[i] = destinations[i]->digits_q_ntt.data();
        digit_p_ptrs[i] = destinations[i]->digits_p_ntt.data();
    }

    workspace.outer_accumulator.ensure_capacity(
        device_id, degree, q_count, p_count, batch_count);
    kernel::launch_double_hoist_pre_rotated_giant_group_accumulate_batch(
        workspace.outer_accumulator.q.data(),
        workspace.outer_accumulator.p.data(),
        batch_count,
        inner_q_batch,
        inner_batch_stride_words,
        0,
        digit_q_ptrs.data(),
        digit_p_ptrs.data(),
        group_indices.data(),
        galois_elts.data(),
        key_indices.data(),
        galois_keys.galois_key_q0_ptrs.data(),
        galois_keys.galois_key_p0_ptrs.data(),
        galois_keys.galois_key_q1_ptrs.data(),
        galois_keys.galois_key_p1_ptrs.data(),
        batched_giant_count,
        dnum,
        galois_keys.meta.decomposition_count,
        *parameter_shard,
        degree);

    keyswitch_handler_.moddown_qp_ciphertext_batch_to_q(
        workspace.outer_accumulator,
        batch_count,
        workspace.inner_q_batch,
        level_info,
        workspace.batch_p_coeff,
        workspace.batch_converted_q);

    destination_ciphertexts.resize(batch_count);
    if (rescale_count == 0)
    {
        GpuRNSPoly layout;
        layout.degree = degree;
        layout.q_count = q_count;
        layout.p_count = 0;
        layout.shards.push_back(GpuPolyShard{0, 0, 0, q_count, 0, degree});
        const auto stream = gpu_execution_stream();
        gpu_check_cuda(
            cudaSetDevice(device_id),
            "GpuEvaluator::giant_reduce_q_groups_batch copy cudaSetDevice");
        for (std::size_t b = 0; b < batch_count; ++b)
        {
            prepare_ciphertext_destination(
                destination_ciphertexts[b],
                nullptr,
                nullptr,
                product_meta,
                2,
                device_id,
                layout);
            const auto dst_view = destination_ciphertexts[b].make_view();
            const GpuWord *src0 =
                workspace.inner_q_batch.q.data() + b * 2 * q_words;
            gpu_check_cuda(
                cudaMemcpyAsync(
                    dst_view.polys[0].shards.front().ptr,
                    src0,
                    q_words * sizeof(GpuWord),
                    cudaMemcpyDeviceToDevice,
                    stream),
                "GpuEvaluator::giant_reduce_q_groups_batch copy c0");
            gpu_check_cuda(
                cudaMemcpyAsync(
                    dst_view.polys[1].shards.front().ptr,
                    src0 + q_words,
                    q_words * sizeof(GpuWord),
                    cudaMemcpyDeviceToDevice,
                    stream),
                "GpuEvaluator::giant_reduce_q_groups_batch copy c1");
        }
        return;
    }

    const auto &next_level_info = params_.get_next_level(parms_id);
    if (level_info.shards.empty() || level_info.shards.front().q_last == 0 ||
        next_level_info.q_count + 1 != q_count)
    {
        throw std::invalid_argument(
            "GpuEvaluator::giant_reduce_q_groups_batch: rescale level mismatch");
    }
    const std::size_t dst_q_count = q_count - 1;
    GpuCiphertextMeta dst_meta = product_meta;
    dst_meta.parms_id = next_level_info.parms_id;
    dst_meta.q_count = dst_q_count;
    dst_meta.scale =
        product_scale / static_cast<double>(level_info.shards.front().q_last);
    if (!(dst_meta.scale > 0.0) || !std::isfinite(dst_meta.scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::giant_reduce_q_groups_batch: invalid rescaled scale");
    }

    GpuRNSPoly dst_layout;
    dst_layout.degree = degree;
    dst_layout.q_count = dst_q_count;
    dst_layout.p_count = 0;
    dst_layout.shards.push_back(GpuPolyShard{0, 0, 0, dst_q_count, 0, degree});

    std::vector<GpuCiphertextView> dst_views;
    dst_views.reserve(batch_count);
    for (std::size_t b = 0; b < batch_count; ++b)
    {
        prepare_ciphertext_destination(
            destination_ciphertexts[b],
            nullptr,
            nullptr,
            dst_meta,
            2,
            device_id,
            dst_layout);
        dst_views.push_back(destination_ciphertexts[b].make_view());
    }
    if (!modswitch_handler_.rescale_q_batch_to_outputs(
            workspace.inner_q_batch.q.data(),
            batch_count,
            q_count,
            dst_views,
            level_info,
            next_level_info))
    {
        for (std::size_t b = 0; b < batch_count; ++b)
        {
            GpuConstCiphertextView src_view;
            src_view.meta = product_meta;
            src_view.polys.resize(2);
            const GpuWord *src0 =
                workspace.inner_q_batch.q.data() + b * 2 * q_words;
            for (std::size_t c = 0; c < 2; ++c)
            {
                src_view.polys[c].poly_id = c;
                src_view.polys[c].shards.push_back(GpuConstPolyShardView{
                    device_id,
                    src0 + c * q_words,
                    0, q_count, 0, degree});
            }
            modswitch_handler_.rescale_ciphertext(
                dst_views[b], src_view, level_info, next_level_info);
        }
    }
}

void GpuEvaluator::dft_double_hoist(
    const GpuCiphertextData &source_ciphertext,
    const GpuLinearMatrixGroupQP &matrix_group,
    const GpuGaloisKeysData &galois_keys,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    if (matrix_group.data().empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::dft_double_hoist: empty matrix group");
    }

    const bool dynamic_rescale = matrix_group.rescale_min_scale() > 0.0;
    if (dynamic_rescale &&
        matrix_group.rescale_counts().size() != matrix_group.data().size())
    {
        throw std::invalid_argument(
            "GpuEvaluator::dft_double_hoist: dynamic rescale plan size mismatch");
    }

    GpuCiphertextData product;
    GpuCiphertextData current;
    workspace.matrix_counts.clear();
    workspace.matrix_rescale_trace.clear();
    multiply_by_diag_matrix_bsgs_double_hoist(
        source_ciphertext,
        matrix_group.data().front(),
        galois_keys,
        dynamic_rescale ? 0 : std::max(matrix_group.step(), std::uint32_t{1}),
        workspace,
        dynamic_rescale ? product : current);
    if (dynamic_rescale)
    {
        rescale_dynamic(
            product,
            current,
            matrix_group.rescale_min_scale());
    }
    workspace.matrix_counts.push_back(workspace.last_counts);
    workspace.matrix_rescale_trace.push_back(
        GpuDoubleHoistWorkspace::MatrixRescaleTrace{
            source_ciphertext.meta.q_count,
            current.meta.q_count,
            source_ciphertext.meta.scale,
            matrix_group.data().front().scale,
            source_ciphertext.meta.scale * matrix_group.data().front().scale,
            current.meta.scale,
            static_cast<std::uint32_t>(
                source_ciphertext.meta.q_count - current.meta.q_count)});
    for (std::size_t i = 1; i < matrix_group.data().size(); ++i)
    {
        const std::size_t input_q_count = current.meta.q_count;
        const double input_scale = current.meta.scale;
        GpuCiphertextData next_product;
        GpuCiphertextData next;
        multiply_by_diag_matrix_bsgs_double_hoist(
            current,
            matrix_group.data()[i],
            galois_keys,
            dynamic_rescale ? 0 : std::max(matrix_group.step(), std::uint32_t{1}),
            workspace,
            dynamic_rescale ? next_product : next);
        if (dynamic_rescale)
        {
            rescale_dynamic(
                next_product,
                next,
                matrix_group.rescale_min_scale());
        }
        workspace.matrix_counts.push_back(workspace.last_counts);
        workspace.matrix_rescale_trace.push_back(
            GpuDoubleHoistWorkspace::MatrixRescaleTrace{
                input_q_count,
                next.meta.q_count,
                input_scale,
                matrix_group.data()[i].scale,
                input_scale * matrix_group.data()[i].scale,
                next.meta.scale,
                static_cast<std::uint32_t>(input_q_count - next.meta.q_count)});
        current = std::move(next);
    }
    destination_ciphertext = std::move(current);
}

void GpuEvaluator::coeff_to_slot(
    const GpuCiphertextData &source_ciphertext,
    const GpuLinearMatrixGroup &matrix_group,
    const GpuPlaintextData &minus_i_plaintext,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &result_real,
    GpuCiphertextData &result_imag) const
{
    GpuCiphertextData dft_result;
    dft(source_ciphertext, matrix_group, galois_keys, dft_result);

    GpuCiphertextData conjugated;
    conjugate(dft_result, galois_keys, conjugated);

    add(dft_result, conjugated, result_real);

    GpuCiphertextData imag_difference;
    sub(dft_result, conjugated, imag_difference);
    multiply_plain(imag_difference, minus_i_plaintext, result_imag);
}

void GpuEvaluator::coeff_to_slot_loose(
    const GpuCiphertextData &source_ciphertext,
    const GpuLinearMatrixGroup &matrix_group,
    const GpuPlaintextData &minus_i_plaintext,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &result_real,
    GpuCiphertextData &result_imag) const
{
    GpuCiphertextData dft_result;
    dft_loose(
        source_ciphertext,
        matrix_group,
        galois_keys,
        dft_result);

    GpuCiphertextData conjugated;
    conjugate(dft_result, galois_keys, conjugated);
    add(dft_result, conjugated, result_real);

    GpuCiphertextData imag_difference;
    sub(dft_result, conjugated, imag_difference);
    multiply_plain(imag_difference, minus_i_plaintext, result_imag);
}

void GpuEvaluator::coeff_to_slot_double_hoist(
    const GpuCiphertextData &source_ciphertext,
    const GpuLinearMatrixGroupQP &matrix_group,
    const GpuPlaintextData &minus_i_plaintext,
    const GpuGaloisKeysData &galois_keys,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &result_real,
    GpuCiphertextData &result_imag) const
{
    GpuCiphertextData dft_result;
    dft_double_hoist(
        source_ciphertext,
        matrix_group,
        galois_keys,
        workspace,
        dft_result);

    GpuCiphertextData conjugated;
    if (galois_keys.meta.galois_format ==
        GpuGaloisKeyFormat::InversePreRotated)
    {
        conjugate_pre_rotated(
            dft_result,
            galois_keys,
            workspace,
            conjugated);
    }
    else
    {
        conjugate(dft_result, galois_keys, conjugated);
    }
    add(dft_result, conjugated, result_real);

    GpuCiphertextData imag_difference;
    sub(dft_result, conjugated, imag_difference);
    multiply_plain(imag_difference, minus_i_plaintext, result_imag);
}

void GpuEvaluator::slot_to_coeff(
    const GpuCiphertextData &source_real,
    const GpuCiphertextData &source_imag,
    const GpuLinearMatrixGroup &matrix_group,
    const GpuPlaintextData &plus_i_plaintext,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &result) const
{
    GpuCiphertextData scaled_imag;
    multiply_plain(source_imag, plus_i_plaintext, scaled_imag);

    GpuCiphertextData merged_slots;
    add(scaled_imag, source_real, merged_slots);

    dft(merged_slots, matrix_group, galois_keys, result);
}

void GpuEvaluator::slot_to_coeff_loose(
    const GpuCiphertextData &source_real,
    const GpuCiphertextData &source_imag,
    const GpuLinearMatrixGroup &matrix_group,
    const GpuPlaintextData &plus_i_plaintext,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &result) const
{
    GpuCiphertextData scaled_imag;
    multiply_plain(source_imag, plus_i_plaintext, scaled_imag);
    GpuCiphertextData merged_slots;
    add(scaled_imag, source_real, merged_slots);
    dft_loose(
        merged_slots,
        matrix_group,
        galois_keys,
        result);
}

void GpuEvaluator::slot_to_coeff_double_hoist(
    const GpuCiphertextData &source_real,
    const GpuCiphertextData &source_imag,
    const GpuLinearMatrixGroupQP &matrix_group,
    const GpuPlaintextData &plus_i_plaintext,
    const GpuGaloisKeysData &galois_keys,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &result) const
{
    GpuCiphertextData scaled_imag;
    multiply_plain(source_imag, plus_i_plaintext, scaled_imag);
    GpuCiphertextData merged_slots;
    add(scaled_imag, source_real, merged_slots);
    dft_double_hoist(
        merged_slots,
        matrix_group,
        galois_keys,
        workspace,
        result);
}

namespace
{
// A profile's raised-scale override assumes ModRaise starts from exactly
// q0/ratio, the scale its CoeffToSlot constants were built for.  Whatever the
// integer normalization left over would scale the message by that factor,
// with no metadata left to record it.
void require_modraise_scale(const GpuCiphertextData &input, double q0_over_message_ratio,
                            const char *where)
{
    const double relative = std::abs(input.meta.scale / q0_over_message_ratio - 1.0);
    if (!(relative <= 1e-6))
    {
        throw std::runtime_error(
            std::string(where) + ": ModRaise input scale is " + std::to_string(relative) +
            " off q0/ratio; the raised-scale override would scale the message by as much");
    }
}
} // namespace

namespace
{

// ModRaise, with sparse-secret encapsulation when the profile has its keys:
// dense -> ephemeral at q0, raise, ephemeral -> dense at the top level.
void raise_modulus_encapsulated(
    const GpuEvaluator &evaluator,
    const GpuCiphertextData &input,
    const GpuBootstrapData &data,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &raised)
{
    if (!data.sparse_switch_to_ephemeral && !data.sparse_switch_to_dense)
    {
        evaluator.raise_modulus(input, raised);
        return;
    }
    if (!data.sparse_switch_to_ephemeral || !data.sparse_switch_to_dense)
    {
        throw std::invalid_argument(
            "GPU bootstrap sparse-secret encapsulation needs both switching keys");
    }
    NvtxRange range("SparseSecretEncapsulation");
    // The dense -> ephemeral key may live on its own q0 * P_low modulus.
    const GpuEvaluator &to_ephemeral =
        data.sparse_low_evaluator ? *data.sparse_low_evaluator : evaluator;
    to_ephemeral.switch_key(input, *data.sparse_switch_to_ephemeral, workspace.scratch4);
    evaluator.raise_modulus(workspace.scratch4, workspace.scratch5);
    evaluator.switch_key(workspace.scratch5, *data.sparse_switch_to_dense, raised);
}

// A profile whose EvalMod ends above its declared SlotToCoeff input drops the
// surplus Q limbs without a copy, before SlotToCoeff.
void drop_to_slot_to_coeff_input(
    const GpuEvaluator &evaluator,
    const GpuBootstrapData &data,
    GpuCiphertextData &ciphertext)
{
    if (data.slot_to_coeff_input_q_count == 0 ||
        ciphertext.meta.q_count == data.slot_to_coeff_input_q_count)
    {
        return;
    }
    if (ciphertext.meta.q_count < data.slot_to_coeff_input_q_count)
    {
        throw std::runtime_error(
            "GPU bootstrap EvalMod output is below the SlotToCoeff input level");
    }
    evaluator.drop_modulus_inplace(ciphertext, data.slot_to_coeff_input_parms_id);
}

} // namespace

GpuLinearTransformMode bootstrap_validate_schedule(
    const GpuCiphertextData &source, const GpuBootstrapData &data)
{
    const auto mode = data.allow_environment_linear_transform_override
        ? gpu_linear_transform_mode_from_environment(data.linear_transform_mode)
        : data.linear_transform_mode;
    const bool dh = mode == GpuLinearTransformMode::DoubleHoistBsgs;
    if (mode == GpuLinearTransformMode::SingleHoistBsgs)
        throw std::invalid_argument("GpuEvaluator::bootstrap: single_hoist mode is reserved for staged validation; select classic, double_hoist, loose_coupled, or no_hoist");
    if ((!dh && data.coeff_to_slot_matrix.data().empty()) ||
        (dh && data.active_coeff_to_slot_matrix_qp().data().empty()))
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty CoeffToSlot matrix group");
    if ((!dh && data.slot_to_coeff_matrix.data().empty()) ||
        (dh && data.active_slot_to_coeff_matrix_qp().data().empty()))
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty SlotToCoeff matrix group");
    if (data.minus_i_plaintext.empty())
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty minus-i plaintext");
    if (data.plus_i_plaintext.empty())
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty plus-i plaintext");
    if (!(data.q0_over_message_ratio > 0.0) || !std::isfinite(data.q0_over_message_ratio))
        throw std::invalid_argument("GpuEvaluator::bootstrap: invalid q0/message ratio");
    if (data.slot_to_coeff_first &&
        (source.meta.q_count != data.input_q_count || !(source.meta.scale > 0.0) ||
         std::abs(std::log2(source.meta.scale / data.input_scale)) > 1e-8 || !dh))
        throw std::invalid_argument("S2C-first Boot requires its declared input Q/scale and double-hoist mode");
    // dft variants used to validate the dynamic rescale plan before the first layer.
    const auto check = [](const auto &group) {
        if (group.rescale_min_scale() > 0.0 &&
            group.rescale_counts().size() != group.data().size())
            throw std::invalid_argument("GpuEvaluator::bootstrap: dynamic rescale plan size mismatch");
    };
    if (dh) { check(data.active_coeff_to_slot_matrix_qp()); check(data.active_slot_to_coeff_matrix_qp()); }
    else { check(data.coeff_to_slot_matrix); check(data.slot_to_coeff_matrix); }
    return mode;
}

void GpuEvaluator::bootstrap(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    const GpuRelinKeysData &relin_keys,
    const GpuGaloisKeysData &galois_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    bootstrap_impl(source_ciphertext, bootstrap_data, relin_keys, galois_keys,
                   workspace, destination_ciphertext, nullptr);
}

GpuEvaluator::ManualBootstrapProjections GpuEvaluator::bootstrap_manual_projections(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    const GpuRelinKeysData &relin_keys,
    const GpuGaloisKeysData &galois_keys,
    GpuBootstrapWorkspace &workspace) const
{
    const auto &m = source_ciphertext.meta;
    if (!bootstrap_data.slot_to_coeff_first || !bootstrap_data.project_real ||
        source_ciphertext.empty() || m.degree != 65536 || m.p_count != 0 ||
        m.component_count != 2 || !m.is_ntt_form || m.correction_factor != 1 ||
        m.q_count != bootstrap_data.input_q_count ||
        !(m.scale > 0.0) || !std::isfinite(m.scale) ||
        !(bootstrap_data.input_scale > 0.0) || !std::isfinite(bootstrap_data.input_scale) ||
        std::abs(std::log2(m.scale / bootstrap_data.input_scale)) > 1e-8 ||
        source_ciphertext.fields_.size() != 1)
        throw std::invalid_argument("manual projections require external N65536 Q-only NTT input and real S2C-first contract");
    const auto &level = params_.get_level(m.parms_id);
    if (level.degree != 65536 || level.q_count != m.q_count || level.shards.size() != 1 ||
        level.shards[0].device_id != source_ciphertext.fields_[0].device_id)
        throw std::invalid_argument("manual projections input must use this evaluator's single device");
    const auto output_q = bootstrap_data.slot_to_coeff_input_q_count
        ? bootstrap_data.slot_to_coeff_input_q_count : bootstrap_data.eval_mod.output_q_count;
    const auto output_id = bootstrap_data.slot_to_coeff_input_q_count
        ? bootstrap_data.slot_to_coeff_input_parms_id : bootstrap_data.eval_mod.output_parms_id;
    const auto &pi = bootstrap_data.plus_i_plaintext;
    if (pi.empty() || pi.fields_.size() != 1 || pi.fields_[0].empty() ||
        pi.fields_[0].device_id != source_ciphertext.fields_[0].device_id ||
        pi.meta.degree != 65536 || pi.meta.p_count != 0 || !pi.meta.is_ntt_form ||
        pi.meta.q_count != output_q || pi.meta.parms_id != output_id || pi.meta.scale != 1.0 ||
        !(bootstrap_data.output_scale_multiplier > 0.0) || !std::isfinite(bootstrap_data.output_scale_multiplier) ||
        !(bootstrap_data.slot_to_coeff_output_scale > 0.0) || !std::isfinite(bootstrap_data.slot_to_coeff_output_scale))
        throw std::invalid_argument("manual projections require matching output-level scale1 +i and finite physical fold");
    for (const auto *temporary : {
            &workspace.modraise_input, &workspace.raised, &workspace.raised_scaled,
            &workspace.coeff_to_slot_real, &workspace.coeff_to_slot_imag,
            &workspace.eval_mod_real, &workspace.eval_mod_imag,
            &workspace.scratch0, &workspace.scratch1, &workspace.scratch2,
            &workspace.scratch3, &workspace.scratch4, &workspace.scratch5,
            &workspace.eval_mod_trace_offset_input, &workspace.eval_mod_trace_polynomial_output,
            &workspace.coeff_to_slot_double_hoist.inner_q, &workspace.coeff_to_slot_double_hoist.result_q,
            &workspace.slot_to_coeff_double_hoist.inner_q, &workspace.slot_to_coeff_double_hoist.result_q})
        if (&source_ciphertext == temporary)throw std::invalid_argument("manual projections input aliases workspace");
    for (const auto *collection : {
            &workspace.eval_mod_basis, &workspace.eval_mod_nodes,
            &workspace.eval_mod_trace_double_angle_square_outputs,
            &workspace.eval_mod_trace_double_angle_relin_outputs,
            &workspace.eval_mod_trace_double_angle_rescaled_square_outputs,
            &workspace.eval_mod_trace_polynomial_leaf_outputs,
            &workspace.eval_mod_trace_polynomial_combine_outputs,
            &workspace.eval_mod_trace_double_angle_outputs})
        for (const auto &temporary : *collection)
            if (&source_ciphertext == &temporary)throw std::invalid_argument("manual projections input aliases workspace vector");
    ManualBootstrapProjections result;
    bootstrap_impl(source_ciphertext, bootstrap_data, relin_keys, galois_keys,
                   workspace, result.real, &result.imaginary);
    return result;
}

void GpuEvaluator::bootstrap_impl(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    const GpuRelinKeysData &relin_keys,
    const GpuGaloisKeysData &galois_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext,
    GpuCiphertextData *imaginary_projection) const
{
    const auto linear_transform_mode =
        bootstrap_data.allow_environment_linear_transform_override
            ? gpu_linear_transform_mode_from_environment(
                  bootstrap_data.linear_transform_mode)
            : bootstrap_data.linear_transform_mode;
    const bool use_double_hoist =
        linear_transform_mode == GpuLinearTransformMode::DoubleHoistBsgs;
    const bool use_loose_coupled =
        linear_transform_mode == GpuLinearTransformMode::LooseCoupledBsgs ||
        linear_transform_mode == GpuLinearTransformMode::NoHoistBsgs;
    if (linear_transform_mode ==
        GpuLinearTransformMode::SingleHoistBsgs)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap: single_hoist mode is reserved for "
            "staged validation; select classic, double_hoist, "
            "loose_coupled, or no_hoist");
    }
    if ((!use_double_hoist &&
         bootstrap_data.coeff_to_slot_matrix.data().empty()) ||
        (use_double_hoist &&
         bootstrap_data.active_coeff_to_slot_matrix_qp().data().empty()))
    {
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty CoeffToSlot matrix group");
    }
    if ((!use_double_hoist &&
         bootstrap_data.slot_to_coeff_matrix.data().empty()) ||
        (use_double_hoist &&
         bootstrap_data.active_slot_to_coeff_matrix_qp().data().empty()))
    {
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty SlotToCoeff matrix group");
    }
    if (bootstrap_data.minus_i_plaintext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty minus-i plaintext");
    }
    if (bootstrap_data.plus_i_plaintext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty plus-i plaintext");
    }
    if (!(bootstrap_data.q0_over_message_ratio > 0.0) ||
        !std::isfinite(bootstrap_data.q0_over_message_ratio))
    {
        throw std::invalid_argument("GpuEvaluator::bootstrap: invalid q0/message ratio");
    }

    NvtxRange bootstrap_range("bootstrap_once");
    const char *stage_timing_env = std::getenv("POSEIDON_BOOT_STAGE_TIMING");
    const bool boot_stage_timing =
        stage_timing_env != nullptr && std::string(stage_timing_env) == "1";
    struct BootEventGuard
    {
        cudaEvent_t ev[7]{};
        bool active = false;
        GpuBootstrapWorkspace *ws = nullptr;
        bool prev_timing = false;
        ~BootEventGuard()
        {
            if (ws != nullptr)
            {
                ws->capture_eval_mod_stage_timing = prev_timing;
            }
            if (active)
            {
                for (auto &e : ev)
                {
                    if (e != nullptr)
                    {
                        cudaEventDestroy(e);
                    }
                }
            }
        }
    };
    BootEventGuard stage_guard;
    stage_guard.ws = &workspace;
    stage_guard.prev_timing = workspace.capture_eval_mod_stage_timing;
    cudaEvent_t (&ev)[7] = stage_guard.ev;
    auto mark = [&](int idx) {
        if (boot_stage_timing)
        {
            gpu_check_cuda(cudaEventRecord(ev[idx], gpu_execution_stream()), "boot stage ev");
        }
    };
    if (boot_stage_timing)
    {
        for (auto &e : ev) gpu_check_cuda(cudaEventCreate(&e), "boot stage create");
        stage_guard.active = true;
        workspace.capture_eval_mod_stage_timing = true;
    }
    mark(0);
    const GpuCiphertextData *raised_for_c2s = &workspace.raised;
    {
        NvtxRange range("ModRaise");
        if (bootstrap_data.slot_to_coeff_first)
        {
            if (source_ciphertext.meta.q_count != bootstrap_data.input_q_count ||
                !(source_ciphertext.meta.scale > 0.0) ||
                std::abs(std::log2(source_ciphertext.meta.scale / bootstrap_data.input_scale)) > 1e-8 ||
                !use_double_hoist)
                throw std::invalid_argument("S2C-first Boot requires its declared input Q/scale and double-hoist mode");
            dft_double_hoist(source_ciphertext,
                bootstrap_data.active_slot_to_coeff_matrix_qp(), galois_keys,
                workspace.slot_to_coeff_double_hoist, workspace.scratch0);
            if (workspace.scratch0.meta.parms_id != bootstrap_data.q0_parms_id)
                throw std::runtime_error("S2C-first Boot S2C output level differs from q0");
            // Match the manual scheduler: scale the Q2 result by a rounded
            // integer, without consuming another Q before ModRaise.
            const double multiplier = std::round(
                bootstrap_data.q0_over_message_ratio / workspace.scratch0.meta.scale);
            if (!std::isfinite(multiplier) || multiplier < 1.0 || multiplier > std::ldexp(1.0, 53))
                throw std::runtime_error("S2C-first Boot preparation multiplier is outside [1, 2^53]");
            multiply_scalar(workspace.scratch0, static_cast<std::uint64_t>(multiplier),
                workspace.modraise_input);
            workspace.modraise_input.meta.scale = workspace.scratch0.meta.scale * multiplier;
        }
        else
        {
            bootstrap_prepare_modraise_input(
                source_ciphertext,
                workspace.modraise_input,
                bootstrap_data.q0_parms_id,
                bootstrap_data.q0_over_message_ratio);
            if (bootstrap_data.raised_scale_override > 0.0)
                require_modraise_scale(workspace.modraise_input,
                                       bootstrap_data.q0_over_message_ratio,
                                       "GpuEvaluator::bootstrap");
        }
        mark(1);

        raise_modulus_encapsulated(
            *this, workspace.modraise_input, bootstrap_data, workspace, workspace.raised);

        if (bootstrap_data.raised_scale_override > 0.0)
        {
            workspace.raised.meta.scale = bootstrap_data.raised_scale_override;
        }

        if (bootstrap_data.post_raise_integer_multiplier > 1)
        {
            multiply_scalar(
                workspace.raised,
                bootstrap_data.post_raise_integer_multiplier,
                workspace.raised_scaled);
            workspace.raised_scaled.meta.scale =
                workspace.raised.meta.scale * bootstrap_data.post_raise_scale_multiplier;
            raised_for_c2s = &workspace.raised_scaled;
        }
        else if (!bootstrap_data.post_raise_plaintext.empty())
        {
            multiply_plain(
                workspace.raised,
                bootstrap_data.post_raise_plaintext,
                workspace.raised_scaled);
            raised_for_c2s = &workspace.raised_scaled;
        }
    }
    mark(2);

    {
        NvtxRange range("CtS");
        if (use_double_hoist)
        {
            coeff_to_slot_double_hoist(
                *raised_for_c2s,
                bootstrap_data.active_coeff_to_slot_matrix_qp(),
                bootstrap_data.minus_i_plaintext,
                galois_keys,
                workspace.coeff_to_slot_double_hoist,
                workspace.coeff_to_slot_real,
                workspace.coeff_to_slot_imag);
        }
        else if (use_loose_coupled)
        {
            coeff_to_slot_loose(
                *raised_for_c2s,
                bootstrap_data.coeff_to_slot_matrix,
                bootstrap_data.minus_i_plaintext,
                galois_keys,
                workspace.coeff_to_slot_real,
                workspace.coeff_to_slot_imag);
        }
        else
        {
            coeff_to_slot(
                *raised_for_c2s,
                bootstrap_data.coeff_to_slot_matrix,
                bootstrap_data.minus_i_plaintext,
                galois_keys,
                workspace.coeff_to_slot_real,
                workspace.coeff_to_slot_imag);
        }
    }
    mark(3);

    GpuBootstrapWorkspace::EvalModStageTiming eval_real_stages{};
    {
        NvtxRange range("EvalMod");
        if (!workspace.capture_eval_mod_trace &&
            use_evalmod_pair() &&
            !bootstrap_data.eval_mod.polynomial_blocks.empty())
        {
            NvtxRange branch_range("EvalMod pair");
            eval_mod_high_precision_pair(
                workspace.coeff_to_slot_real,
                workspace.coeff_to_slot_imag,
                bootstrap_data,
                relin_keys,
                workspace,
                workspace.eval_mod_real,
                workspace.eval_mod_imag);
            if (boot_stage_timing) eval_real_stages = workspace.eval_mod_stage_timing;
            mark(4);
        }
        else
        {
            {
                NvtxRange branch_range("EvalMod real");
                eval_mod_high_precision(
                    workspace.coeff_to_slot_real,
                    bootstrap_data,
                    relin_keys,
                    workspace,
                    workspace.eval_mod_real);
            }
            if (boot_stage_timing) eval_real_stages = workspace.eval_mod_stage_timing;
            mark(4);
            {
                NvtxRange branch_range("EvalMod imag");
                eval_mod_high_precision(
                    workspace.coeff_to_slot_imag,
                    bootstrap_data,
                    relin_keys,
                    workspace,
                    workspace.eval_mod_imag);
            }
        }
    }
    mark(5);

    if (!bootstrap_data.slot_to_coeff_first && bootstrap_data.slot_to_coeff_input_scale > 0.0)
    {
        workspace.eval_mod_real.meta.scale =
            bootstrap_data.slot_to_coeff_input_scale;
        workspace.eval_mod_imag.meta.scale =
            bootstrap_data.slot_to_coeff_input_scale;
    }
    drop_to_slot_to_coeff_input(*this, bootstrap_data, workspace.eval_mod_real);
    drop_to_slot_to_coeff_input(*this, bootstrap_data, workspace.eval_mod_imag);

    {
        NvtxRange range("StC");
        if (bootstrap_data.slot_to_coeff_first)
        {
            if (!bootstrap_data.project_real)
                throw std::invalid_argument("manual S2C-first Boot requires real projection");
            const auto ratio = bootstrap_data.output_ratio;
            if (ratio != 32)
            {
                const char *enabled = std::getenv("POSEIDON_MANUAL_BOOT_RANGE_EXPERIMENT");
                if (!enabled || std::string(enabled) != "1" || ratio < 32 ||
                    ratio > (1U << 17) || (ratio & (ratio - 1U)))
                    throw std::invalid_argument("manual Boot range experiment requires opt-in and power-of-two ratio [32,131072]");
            }
            multiply_plain(workspace.eval_mod_imag, bootstrap_data.plus_i_plaintext,
                workspace.scratch0);
            add(workspace.eval_mod_real, workspace.scratch0, workspace.scratch1);
            // Match the manual order. Scaling after conjugation would also
            // amplify its key-switch rounding noise by 16.
            multiply_scalar(workspace.scratch1, ratio / 2U, workspace.scratch0);
            conjugate(workspace.scratch0, galois_keys, workspace.scratch1);
            add(workspace.scratch0, workspace.scratch1, destination_ciphertext);
            destination_ciphertext.meta.scale *= bootstrap_data.output_scale_multiplier;
            if (imaginary_projection && (!(destination_ciphertext.meta.scale > 0.0) ||
                                         !std::isfinite(destination_ciphertext.meta.scale)))
                throw std::runtime_error("nonfinite manual real projection scale");
            if (std::abs(std::log2(destination_ciphertext.meta.scale /
                    bootstrap_data.slot_to_coeff_output_scale)) > 1e-8)
                throw std::runtime_error("manual S2C-first Boot output scale disagrees with its coefficient fold");
            destination_ciphertext.meta.scale = bootstrap_data.slot_to_coeff_output_scale;
            if (imaginary_projection)
            {
                // h=(ratio/2)*(e0+i*e1); reuse its conjugation, no scale1 half.
                sub(workspace.scratch1, workspace.scratch0, workspace.scratch2);
                multiply_plain(workspace.scratch2, bootstrap_data.plus_i_plaintext, *imaginary_projection);
                imaginary_projection->meta.scale *= bootstrap_data.output_scale_multiplier;
                if (!(imaginary_projection->meta.scale > 0.0) || !std::isfinite(imaginary_projection->meta.scale))
                    throw std::runtime_error("nonfinite manual imaginary projection scale");
                if (std::abs(std::log2(imaginary_projection->meta.scale /
                        bootstrap_data.slot_to_coeff_output_scale)) > 1e-8)
                    throw std::runtime_error("manual imaginary projection scale disagrees with its coefficient fold");
                imaginary_projection->meta.scale = bootstrap_data.slot_to_coeff_output_scale;
            }
            if (boot_stage_timing)
            {
                mark(6);
                gpu_check_cuda(cudaEventSynchronize(ev[6]), "boot stage sync");
                float d[6]{};
                for (int i = 0; i < 6; ++i) gpu_check_cuda(cudaEventElapsedTime(&d[i], ev[i], ev[i + 1]), "boot stage elapsed");
                float tot = 0; gpu_check_cuda(cudaEventElapsedTime(&tot, ev[0], ev[6]), "boot stage tot");
                std::printf("BOOT_STAGES total=%.3f stc=%.3f sse_raise=%.3f cts=%.3f eval_r=%.3f eval_i=%.3f proj=%.3f | eval_r(in=%.3f basis=%.3f leaf=%.3f comb=%.3f da=%.3f)\n",
                            tot, d[0], d[1], d[2], d[3], d[4], d[5],
                            eval_real_stages.input_preparation_ms,
                            eval_real_stages.basis_generation_ms,
                            eval_real_stages.leaf_evaluation_ms,
                            eval_real_stages.bsgs_combine_ms,
                            eval_real_stages.double_angle_ms);
                static bool printed_detail = false;
                if (!printed_detail)
                {
                    printed_detail = true;
                    for (std::size_t mi = 0; mi < workspace.coeff_to_slot_double_hoist.matrix_counts.size(); ++mi)
                    {
                        const auto &mc = workspace.coeff_to_slot_double_hoist.matrix_counts[mi];
                        std::printf("  CTS_LAYER[%zu] baby_tiles=%zu in_moddn=%zu out_dec=%zu out_moddn=%zu keymul=%zu rot_k=%zu\n",
                                    mi, mc.baby_tile_count, mc.inner_moddown_count,
                                    mc.outer_decompose_count, mc.outer_moddown_count,
                                    mc.keymul_count, mc.permute_count);
                    }
                    for (const auto &mt : workspace.eval_mod_multiply_timings)
                    {
                        std::printf("  EVAL_MUL %s q=%zu dnum=%zu sq=%d ms=%.3f\n",
                                    mt.label.c_str(), mt.q_count, mt.decomposition_count,
                                    mt.is_square ? 1 : 0, mt.gpu_ms);
                    }
                }
            }
            return;
        }
        else if (use_double_hoist)
        {
            slot_to_coeff_double_hoist(
                workspace.eval_mod_real,
                workspace.eval_mod_imag,
                bootstrap_data.active_slot_to_coeff_matrix_qp(),
                bootstrap_data.plus_i_plaintext,
                galois_keys,
                workspace.slot_to_coeff_double_hoist,
                destination_ciphertext);
        }
        else if (use_loose_coupled)
        {
            slot_to_coeff_loose(
                workspace.eval_mod_real,
                workspace.eval_mod_imag,
                bootstrap_data.slot_to_coeff_matrix,
                bootstrap_data.plus_i_plaintext,
                galois_keys,
                destination_ciphertext);
        }
        else
        {
            slot_to_coeff(
                workspace.eval_mod_real,
                workspace.eval_mod_imag,
                bootstrap_data.slot_to_coeff_matrix,
                bootstrap_data.plus_i_plaintext,
                galois_keys,
                destination_ciphertext);
        }
    }

    if (bootstrap_data.slot_to_coeff_output_scale > 0.0)
    {
        destination_ciphertext.meta.scale =
            bootstrap_data.slot_to_coeff_output_scale;
    }

    if (bootstrap_data.project_real)
    {
        conjugate(destination_ciphertext, galois_keys, workspace.scratch0);
        add(destination_ciphertext, workspace.scratch0, workspace.scratch1);
        destination_ciphertext = std::move(workspace.scratch1);
    }

    if (bootstrap_data.output_ratio == 0 ||
        (bootstrap_data.project_real &&
         (bootstrap_data.output_ratio & 1U) != 0))
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap: invalid output ratio");
    }
    const std::uint32_t effective_ratio = bootstrap_data.project_real
        ? bootstrap_data.output_ratio / 2
        : bootstrap_data.output_ratio;
    if (effective_ratio > 1)
    {
        NvtxRange range("Output normalize");
        multiply_scalar(
            destination_ciphertext,
            effective_ratio,
            workspace.scratch0);
        destination_ciphertext = std::move(workspace.scratch0);
    }
}

void GpuEvaluator::bootstrap_raise_stage(
    const GpuCiphertextData &source, const GpuBootstrapData &data,
    GpuBootstrapWorkspace &workspace, GpuCiphertextData &destination) const
{
    NvtxRange range("ModRaise");
    if (data.slot_to_coeff_first)
    {
        if (source.meta.parms_id != data.q0_parms_id)
            throw std::runtime_error("S2C-first Boot S2C output level differs from q0");
        const double multiplier = std::round(data.q0_over_message_ratio / source.meta.scale);
        if (!std::isfinite(multiplier) || multiplier < 1.0 || multiplier > std::ldexp(1.0, 53))
            throw std::runtime_error("S2C-first Boot preparation multiplier is outside [1, 2^53]");
        multiply_scalar(source, static_cast<std::uint64_t>(multiplier), workspace.modraise_input);
        workspace.modraise_input.meta.scale = source.meta.scale * multiplier;
    }
    else
    {
        bootstrap_prepare_modraise_input(source, workspace.modraise_input,
                                        data.q0_parms_id, data.q0_over_message_ratio);
        if (data.raised_scale_override > 0.0)
            require_modraise_scale(workspace.modraise_input, data.q0_over_message_ratio,
                                   "GpuEvaluator::bootstrap");
    }
    GpuCiphertextData raised;
    raise_modulus_encapsulated(*this, workspace.modraise_input, data, workspace, raised);
    if (data.raised_scale_override > 0.0) raised.meta.scale = data.raised_scale_override;
    if (data.post_raise_integer_multiplier > 1)
    {
        multiply_scalar(raised, data.post_raise_integer_multiplier, destination);
        destination.meta.scale = raised.meta.scale * data.post_raise_scale_multiplier;
    }
    else if (!data.post_raise_plaintext.empty())
        multiply_plain(raised, data.post_raise_plaintext, destination);
    else
        destination = std::move(raised);
}

void GpuEvaluator::bootstrap_raise(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    // Without the override the raised scale follows the input, and an
    // expanded Boot's later steps were planned for one fixed scale.
    if (!(bootstrap_data.raised_scale_override > 0.0))
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_raise: the profile has no raised_scale_override");
    }
    NvtxRange range("ModRaise");
    if (bootstrap_data.slot_to_coeff_first)
    {
        // The steps before this one ran SlotToCoeff down to q0 and the
        // rounded product that sets the scale (a mul_scalar step), as
        // bootstrap() does before raise_modulus.
        if (source_ciphertext.meta.parms_id != bootstrap_data.q0_parms_id)
        {
            throw std::invalid_argument(
                "GpuEvaluator::bootstrap_raise: an S2C-first Boot raises from q0");
        }
        raise_modulus_encapsulated(
            *this, source_ciphertext, bootstrap_data, workspace, destination_ciphertext);
    }
    else
    {
        bootstrap_prepare_modraise_input(
            source_ciphertext,
            workspace.modraise_input,
            bootstrap_data.q0_parms_id,
            bootstrap_data.q0_over_message_ratio);
        require_modraise_scale(workspace.modraise_input, bootstrap_data.q0_over_message_ratio,
                               "GpuEvaluator::bootstrap_raise");
        raise_modulus_encapsulated(
            *this, workspace.modraise_input, bootstrap_data, workspace, destination_ciphertext);
    }
    destination_ciphertext.meta.scale = bootstrap_data.raised_scale_override;
}

void GpuEvaluator::bootstrap_linear_layer(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    bool coeff_to_slot,
    std::size_t layer,
    const GpuGaloisKeysData &galois_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    const auto mode =
        bootstrap_data.allow_environment_linear_transform_override
            ? gpu_linear_transform_mode_from_environment(
                  bootstrap_data.linear_transform_mode)
            : bootstrap_data.linear_transform_mode;
    NvtxRange range(coeff_to_slot ? "CtS layer" : "StC layer");
    // One iteration of dft_double_hoist / dft_loose / dft.
    if (mode == GpuLinearTransformMode::DoubleHoistBsgs)
    {
        const auto &group = coeff_to_slot
            ? bootstrap_data.active_coeff_to_slot_matrix_qp()
            : bootstrap_data.active_slot_to_coeff_matrix_qp();
        const bool dynamic_rescale = group.rescale_min_scale() > 0.0;
        GpuCiphertextData product;
        multiply_by_diag_matrix_bsgs_double_hoist(
            source_ciphertext,
            group.data().at(layer),
            galois_keys,
            dynamic_rescale ? 0 : std::max(group.step(), std::uint32_t{1}),
            coeff_to_slot ? workspace.coeff_to_slot_double_hoist
                          : workspace.slot_to_coeff_double_hoist,
            dynamic_rescale ? product : destination_ciphertext);
        if (dynamic_rescale)
        {
            rescale_dynamic(product, destination_ciphertext, group.rescale_min_scale());
        }
        return;
    }
    if (mode == GpuLinearTransformMode::SingleHoistBsgs)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_linear_layer: single_hoist is not a bootstrap mode");
    }
    const auto &group = coeff_to_slot
        ? bootstrap_data.coeff_to_slot_matrix
        : bootstrap_data.slot_to_coeff_matrix;
    const bool dynamic_rescale = group.rescale_min_scale() > 0.0;
    const std::uint32_t rescale_count =
        dynamic_rescale ? 0 : std::max(group.step(), std::uint32_t{1});
    GpuCiphertextData product;
    GpuCiphertextData &target = dynamic_rescale ? product : destination_ciphertext;
    if (mode == GpuLinearTransformMode::LooseCoupledBsgs ||
        mode == GpuLinearTransformMode::NoHoistBsgs)
    {
        multiply_by_diag_matrix_bsgs_loose(
            source_ciphertext, group.data().at(layer), galois_keys, rescale_count, target);
    }
    else
    {
        multiply_by_diag_matrix_bsgs(
            source_ciphertext, group.data().at(layer), galois_keys, rescale_count, target);
    }
    if (dynamic_rescale)
    {
        rescale_dynamic(product, destination_ciphertext, group.rescale_min_scale());
    }
}

void GpuEvaluator::bootstrap_bsgs_group(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    bool coeff_to_slot,
    std::size_t layer,
    std::size_t group,
    const GpuGaloisKeysData &galois_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    const auto &groups = coeff_to_slot
        ? bootstrap_data.active_coeff_to_slot_groups_qp()
        : bootstrap_data.active_slot_to_coeff_groups_qp();
    if (groups.empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_bsgs_group: the profile has no group views "
            "(a double-hoist profile builds them)");
    }
    NvtxRange range(coeff_to_slot ? "CtS group" : "StC group");
    multiply_by_diag_matrix_bsgs_double_hoist(
        source_ciphertext,
        groups.at(layer).at(group),
        galois_keys,
        0,
        coeff_to_slot ? workspace.coeff_to_slot_double_hoist
                      : workspace.slot_to_coeff_double_hoist,
        destination_ciphertext);
}

void GpuEvaluator::bootstrap_conjugate(
    const GpuCiphertextData &source_ciphertext,
    const GpuGaloisKeysData &galois_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    // As coeff_to_slot_double_hoist chooses.
    if (galois_keys.meta.galois_format == GpuGaloisKeyFormat::InversePreRotated)
    {
        conjugate_pre_rotated(
            source_ciphertext,
            galois_keys,
            workspace.coeff_to_slot_double_hoist,
            destination_ciphertext);
    }
    else
    {
        conjugate(source_ciphertext, galois_keys, destination_ciphertext);
    }
}

void GpuEvaluator::bootstrap_set_scale(
    const GpuCiphertextData &source_ciphertext,
    double scale,
    GpuCiphertextData &destination_ciphertext) const
{
    if (!(scale > 0.0) || !std::isfinite(scale))
    {
        throw std::invalid_argument("GpuEvaluator::bootstrap_set_scale: invalid scale");
    }
    copy_ciphertext_data(
        source_ciphertext,
        destination_ciphertext,
        "GpuEvaluator::bootstrap_set_scale");
    destination_ciphertext.meta.scale = scale;
}

void GpuEvaluator::multiply_plain_q_view(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_plaintext.meta.p_count == 0)
    {
        multiply_plain(source_ciphertext, source_plaintext, destination_ciphertext);
        return;
    }
    if (source_ciphertext.empty() || source_plaintext.empty() ||
        !(source_ciphertext.meta.parms_id == source_plaintext.meta.parms_id) ||
        !source_ciphertext.meta.is_ntt_form || !source_plaintext.meta.is_ntt_form ||
        source_ciphertext.meta.degree != source_plaintext.meta.degree ||
        source_ciphertext.meta.q_count != source_plaintext.meta.q_count ||
        source_ciphertext.meta.p_count != 0 ||
        source_ciphertext.meta.component_count != source_ciphertext.size() ||
        source_plaintext.poly_.shards.size() != 1)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain_q_view: the plaintext's Q prefix does not "
            "match the ciphertext");
    }
    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);
    if (!all_components_use_layout(source_ciphertext, reference_layout) ||
        reference_layout.shards.size() != 1)
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_plain_q_view: ciphertext shard layout mismatch");
    }
    GpuCiphertextData result =
        GpuCiphertextData::allocate_single_device_sharded(
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            source_ciphertext.size(),
            device_id,
            reference_layout.shards,
            0);
    result.meta = source_ciphertext.meta;
    result.meta.component_count = source_ciphertext.size();
    result.meta.scale = source_ciphertext.meta.scale * source_plaintext.meta.scale;
    // [Q|P] residues are stored Q first, so the Q prefix is the same pointer
    // with fewer limbs.
    auto plaintext_view = source_plaintext.make_const_view();
    plaintext_view.meta.p_count = 0;
    plaintext_view.poly.shards.front().limb_count = source_ciphertext.meta.q_count;
    auto ciphertext_view = source_ciphertext.make_const_view();
    auto destination_view = result.make_view();
    elementwise_handler_.multiply_plain_with_ciphertext(
        destination_view,
        ciphertext_view,
        plaintext_view,
        params_.get_level(source_ciphertext.meta.parms_id));
    destination_ciphertext = std::move(result);
}

void GpuEvaluator::eval_mod_high_precision(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    const GpuRelinKeysData &relin_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    eval_mod_high_precision_impl(
        {&source_ciphertext, nullptr},
        1,
        bootstrap_data,
        relin_keys,
        workspace,
        {&destination_ciphertext, nullptr});
}

void GpuEvaluator::eval_mod_high_precision_pair(
    const GpuCiphertextData &source_real,
    const GpuCiphertextData &source_imag,
    const GpuBootstrapData &bootstrap_data,
    const GpuRelinKeysData &relin_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_real,
    GpuCiphertextData &destination_imag) const
{
    eval_mod_high_precision_impl(
        {&source_real, &source_imag},
        2,
        bootstrap_data,
        relin_keys,
        workspace,
        {&destination_real, &destination_imag});
}

void GpuEvaluator::eval_mod_high_precision_impl(
    std::array<const GpuCiphertextData *, 2> sources,
    std::size_t lane_count,
    const GpuBootstrapData &bootstrap_data,
    const GpuRelinKeysData &relin_keys,
    GpuBootstrapWorkspace &workspace,
    std::array<GpuCiphertextData *, 2> destinations) const
{
    const GpuCiphertextData &source_ciphertext = *sources[0];
    validate_ntt_ciphertext_input(
        "GpuEvaluator::eval_mod_high_precision",
        source_ciphertext,
        true);
    if (source_ciphertext.size() != 2)
    {
        throw std::invalid_argument(
            "GpuEvaluator::eval_mod_high_precision: expected a size-2 ciphertext");
    }
    const double configured_target_scale =
        bootstrap_data.eval_mod.target_scale > 0.0
            ? bootstrap_data.eval_mod.target_scale
            : bootstrap_data.eval_mod_target_scale;
    const double configured_input_scale =
        bootstrap_data.eval_mod.input_scale > 0.0
            ? bootstrap_data.eval_mod.input_scale
            : configured_target_scale;
    const auto &configured_coefficients =
        !bootstrap_data.eval_mod.polynomial_coefficients.empty()
            ? bootstrap_data.eval_mod.polynomial_coefficients
            : bootstrap_data.eval_mod_polynomial_plaintexts;
    const auto &polynomial_terms = bootstrap_data.eval_mod.polynomial_terms;
    const auto &basis_steps = bootstrap_data.eval_mod.basis_steps;
    const auto &polynomial_blocks = bootstrap_data.eval_mod.polynomial_blocks;
    const auto &polynomial_combine_steps =
        bootstrap_data.eval_mod.polynomial_combine_steps;
    const auto &configured_double_angle_constants =
        !bootstrap_data.eval_mod.double_angle_constants.empty()
            ? bootstrap_data.eval_mod.double_angle_constants
            : bootstrap_data.double_angle_plaintexts;
    const auto &configured_double_angle_rescale_counts =
        bootstrap_data.eval_mod.double_angle_rescale_counts;
    const GpuPlaintextData &configured_input_offset =
        !bootstrap_data.eval_mod.input_offset_plaintext.empty()
            ? bootstrap_data.eval_mod.input_offset_plaintext
            : bootstrap_data.eval_mod_input_offset_plaintext;

    if (!(configured_target_scale > 0.0) ||
        !std::isfinite(configured_target_scale))
    {
        throw std::invalid_argument(
            "GpuEvaluator::eval_mod_high_precision: invalid target scale");
    }
    if (polynomial_blocks.empty() &&
        polynomial_terms.empty() &&
        configured_coefficients.empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::eval_mod_high_precision: empty polynomial plaintexts");
    }

    const double target_scale = configured_target_scale;
    const std::uint32_t logical_rescale_count =
        std::max(
            bootstrap_data.eval_mod.logical_rescale_count,
            std::uint32_t{1});

    if (workspace.capture_eval_mod_trace)
    {
        workspace.eval_mod_trace_offset_input = GpuCiphertextData{};
        workspace.eval_mod_trace_polynomial_output = GpuCiphertextData{};
        workspace.eval_mod_trace_double_angle_square_outputs.clear();
        workspace.eval_mod_trace_double_angle_relin_outputs.clear();
        workspace.eval_mod_trace_double_angle_rescaled_square_outputs.clear();
        workspace.eval_mod_trace_double_angle_outputs.clear();
        workspace.eval_mod_trace_double_angle_square_outputs.reserve(
            configured_double_angle_constants.size());
        workspace.eval_mod_trace_double_angle_relin_outputs.reserve(
            configured_double_angle_constants.size());
        workspace.eval_mod_trace_double_angle_rescaled_square_outputs.reserve(
            configured_double_angle_constants.size());
        workspace.eval_mod_trace_double_angle_outputs.reserve(
            configured_double_angle_constants.size());
    }
    if (workspace.capture_eval_mod_stage_timing)
    {
        workspace.eval_mod_stage_timing =
            GpuBootstrapWorkspace::EvalModStageTiming{};
        workspace.eval_mod_multiply_timings.clear();
    }
    EvalModStageEventRecorder stage_recorder(
        workspace.capture_eval_mod_stage_timing);
    EvalModMultiplyEventRecorder multiply_recorder(
        workspace.capture_eval_mod_stage_timing);
    const bool d2d_free_dataflow = use_evalmod_d2d_free_dataflow();
    const bool zero_copy_moddrop =
        d2d_free_dataflow &&
        use_evalmod_zero_copy_moddrop() &&
        !workspace.capture_eval_mod_trace;
    const bool q_prefix_views =
        zero_copy_moddrop && use_evalmod_q_prefix_views();

    /*
     * Pure ModDrop only discards an RNS-Q suffix.  If the ciphertext storage
     * is dead after the current consumer, retain its allocation and shorten
     * the logical prefix in place instead of materializing that prefix via a
     * synchronous D2D copy.  Physical component strides may remain larger
     * than the logical view; every GPU operator consumes the per-component
     * shard pointers and logical limb counts, so the retained padding is not
     * read.  General/shared inputs continue through drop_modulus().
     */
    auto drop_modulus_inplace =
        [&](GpuCiphertextData &ciphertext,
            parms_id_type target_parms_id) {
            if (!zero_copy_moddrop ||
                ciphertext.empty() ||
                ciphertext.fields_.empty() ||
                ciphertext.meta.p_count != 0 ||
                ciphertext.meta.component_count != ciphertext.size())
            {
                return false;
            }

            const auto &source_level =
                params_.get_level(ciphertext.meta.parms_id);
            const auto &target_level =
                params_.get_level(target_parms_id);
            if (source_level.q_count != ciphertext.meta.q_count ||
                target_level.p_count != 0 ||
                target_level.q_count == 0 ||
                target_level.q_count > source_level.q_count ||
                target_level.degree != ciphertext.meta.degree)
            {
                return false;
            }

            for (const auto &poly : ciphertext.polys_)
            {
                if (poly.degree != ciphertext.meta.degree ||
                    poly.q_count != ciphertext.meta.q_count ||
                    poly.p_count != 0 ||
                    poly.shards.size() != 1)
                {
                    return false;
                }
                const auto &shard = poly.shards.front();
                if (shard.limb_begin != 0 ||
                    shard.limb_count != ciphertext.meta.q_count ||
                    shard.coeff_begin != 0 ||
                    shard.coeff_count != ciphertext.meta.degree)
                {
                    return false;
                }
            }

            ciphertext.meta.parms_id = target_level.parms_id;
            ciphertext.meta.q_count = target_level.q_count;
            for (auto &poly : ciphertext.polys_)
            {
                poly.q_count = target_level.q_count;
                poly.p_count = 0;
                poly.shards.front().limb_count = target_level.q_count;
            }
            return true;
        };

    auto supports_q_prefix_multiply =
        [&](const GpuCiphertextData &left,
            const GpuCiphertextData &right,
            parms_id_type target_parms_id) {
            if (!q_prefix_views ||
                left.empty() || right.empty() ||
                left.fields_.empty() || right.fields_.empty() ||
                left.size() != 2 || right.size() != 2 ||
                left.meta.component_count != left.size() ||
                right.meta.component_count != right.size() ||
                !left.meta.is_ntt_form || !right.meta.is_ntt_form ||
                left.meta.p_count != 0 || right.meta.p_count != 0 ||
                left.meta.degree != right.meta.degree ||
                left.fields_.front().device_id != right.fields_.front().device_id)
            {
                return false;
            }

            const auto &target_level = params_.get_level(target_parms_id);
            if (target_level.p_count != 0 ||
                target_level.q_count == 0 ||
                target_level.degree != left.meta.degree ||
                target_level.q_count > left.meta.q_count ||
                target_level.q_count > right.meta.q_count)
            {
                return false;
            }

            auto supports_source =
                [&](const GpuCiphertextData &source) {
                    for (const auto &poly : source.polys_)
                    {
                        if (poly.degree != source.meta.degree ||
                            poly.q_count != source.meta.q_count ||
                            poly.p_count != 0 ||
                            poly.shards.size() != 1)
                        {
                            return false;
                        }
                        const auto &shard = poly.shards.front();
                        if (shard.limb_begin != 0 ||
                            shard.limb_count != source.meta.q_count ||
                            shard.coeff_begin != 0 ||
                            shard.coeff_count != source.meta.degree)
                        {
                            return false;
                        }
                    }
                    return true;
                };
            return supports_source(left) && supports_source(right);
        };

    auto multiply_q_prefix =
        [&](const GpuCiphertextData &left,
            const GpuCiphertextData &right,
            parms_id_type target_parms_id,
            GpuCiphertextData &output) {
            if (!supports_q_prefix_multiply(left, right, target_parms_id))
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: unsupported Q-prefix multiply");
            }

            const auto &target_level = params_.get_level(target_parms_id);
            auto left_view = left.make_const_view();
            auto right_view = right.make_const_view();
            auto trim_view =
                [&](GpuConstCiphertextView &view) {
                    view.meta.parms_id = target_level.parms_id;
                    view.meta.q_count = target_level.q_count;
                    view.meta.p_count = 0;
                    for (auto &poly : view.polys)
                    {
                        poly.shards.front().limb_count = target_level.q_count;
                    }
                };
            trim_view(left_view);
            trim_view(right_view);

            GpuRNSPoly prefix_layout = left.polys_.front();
            prefix_layout.q_count = target_level.q_count;
            prefix_layout.p_count = 0;
            prefix_layout.shards.front().limb_count = target_level.q_count;

            GpuCiphertextMeta result_meta = left.meta;
            result_meta.parms_id = target_level.parms_id;
            result_meta.q_count = target_level.q_count;
            result_meta.p_count = 0;
            result_meta.component_count = 3;
            result_meta.scale = left.meta.scale * right.meta.scale;
            if (!(result_meta.scale > 0.0) ||
                !std::isfinite(result_meta.scale))
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: invalid Q-prefix multiply scale");
            }

            prepare_ciphertext_destination(
                output,
                nullptr,
                nullptr,
                result_meta,
                3,
                left.fields_.front().device_id,
                prefix_layout);
            auto output_view = output.make_view();
            elementwise_handler_.multiply_ciphertext(
                output_view,
                left_view,
                right_view,
                target_level);
        };

    auto add_plain_inplace =
        [&](GpuCiphertextData &ciphertext,
            const GpuPlaintextData &plaintext,
            const char *name) {
            if (!d2d_free_dataflow)
            {
                GpuCiphertextData materialized;
                add_plain(ciphertext, plaintext, materialized);
                ciphertext = std::move(materialized);
                return;
            }
            if (ciphertext.empty() || plaintext.empty() ||
                !(ciphertext.meta.parms_id == plaintext.meta.parms_id) ||
                ciphertext.meta.is_ntt_form != plaintext.meta.is_ntt_form ||
                ciphertext.meta.degree != plaintext.meta.degree ||
                ciphertext.meta.q_count != plaintext.meta.q_count ||
                ciphertext.meta.p_count != plaintext.meta.p_count ||
                !same_scale(ciphertext.meta.scale, plaintext.meta.scale))
            {
                throw std::invalid_argument(std::string(name) +
                                            ": incompatible input");
            }
            auto ciphertext_view = ciphertext.make_view();
            auto plaintext_view = plaintext.make_const_view();
            const auto &level_info = params_.get_level(ciphertext.meta.parms_id);
            elementwise_handler_.add_plain_to_ciphertext_inplace(
                ciphertext_view,
                plaintext_view,
                level_info);
        };

    auto sub_plain_inplace =
        [&](GpuCiphertextData &ciphertext,
            const GpuPlaintextData &plaintext,
            const char *name) {
            if (!d2d_free_dataflow)
            {
                GpuCiphertextData materialized;
                sub_plain(ciphertext, plaintext, materialized);
                ciphertext = std::move(materialized);
                return;
            }
            if (ciphertext.empty() || plaintext.empty() ||
                !(ciphertext.meta.parms_id == plaintext.meta.parms_id) ||
                ciphertext.meta.is_ntt_form != plaintext.meta.is_ntt_form ||
                ciphertext.meta.degree != plaintext.meta.degree ||
                ciphertext.meta.q_count != plaintext.meta.q_count ||
                ciphertext.meta.p_count != plaintext.meta.p_count ||
                !same_scale(ciphertext.meta.scale, plaintext.meta.scale))
            {
                throw std::invalid_argument(std::string(name) +
                                            ": incompatible input");
            }
            auto ciphertext_view = ciphertext.make_view();
            auto plaintext_view = plaintext.make_const_view();
            const auto &level_info = params_.get_level(ciphertext.meta.parms_id);
            elementwise_handler_.sub_plain_from_ciphertext_inplace(
                ciphertext_view,
                plaintext_view,
                level_info);
        };

    /*
     * Dynamic C2S may finish above the EvalMod minimum scale. Keep that
     * physical input scale instead of relabeling it to target_scale: a metadata
     * relabel would amplify the decoded C2S error. Legacy setup leaves
     * input_scale unset and therefore retains the original target-scale path.
     */
    if (lane_count == 2)
    {
        const GpuCiphertextData &source_imag = *sources[1];
        if (source_imag.empty() || source_imag.fields_.empty() ||
            !(source_imag.meta.parms_id == source_ciphertext.meta.parms_id) ||
            source_imag.meta.is_ntt_form != source_ciphertext.meta.is_ntt_form ||
            source_imag.meta.degree != source_ciphertext.meta.degree ||
            source_imag.meta.q_count != source_ciphertext.meta.q_count ||
            source_imag.meta.p_count != source_ciphertext.meta.p_count ||
            source_imag.meta.component_count != 2 || source_imag.size() != 2 ||
            source_imag.fields_.front().device_id != source_ciphertext.fields_.front().device_id ||
            !same_scale(source_imag.meta.scale, source_ciphertext.meta.scale))
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision_pair: mismatched real/imag inputs");
        }
    }
    const bool legacy_horner =
        polynomial_blocks.empty() && polynomial_terms.empty();
    if (lane_count == 2 &&
        (legacy_horner || polynomial_blocks.empty() ||
         workspace.capture_eval_mod_trace || !use_evalmod_pair() ||
         destinations[0] == destinations[1]))
    {
        eval_mod_high_precision(*sources[0], bootstrap_data, relin_keys, workspace, *destinations[0]);
        eval_mod_high_precision(*sources[1], bootstrap_data, relin_keys, workspace, *destinations[1]);
        return;
    }
    const GpuCiphertextData *x = nullptr;
    if (legacy_horner)
    {
        NvtxRange range("EvalMod input preparation");
        copy_ciphertext_data(
            source_ciphertext,
            workspace.scratch0,
            "GpuEvaluator::eval_mod_high_precision input copy");
        workspace.scratch0.meta.scale = configured_input_scale;

        x = &workspace.scratch0;
        if (!configured_input_offset.empty())
        {
            add_plain_inplace(
                workspace.scratch0,
                configured_input_offset,
                "GpuEvaluator::eval_mod_high_precision input offset");
        }

        if (workspace.capture_eval_mod_trace)
        {
            copy_ciphertext_data(
                *x,
                workspace.eval_mod_trace_offset_input,
                "GpuEvaluator::eval_mod_high_precision trace input copy");
            workspace.eval_mod_trace_offset_input.meta.scale = x->meta.scale;
        }
        stage_recorder.record(1);
    }

    /*
     * Keep the old Horner layout only for callers that still populate the
     * early polynomial_coefficients field. The high-precision path below uses
     * a setup-time fixed basis DAG and never performs CPU-side recursive
     * polynomial decomposition in the timed path.
     */
    if (legacy_horner)
    {
        stage_recorder.record(2);
        GpuCiphertextData accumulator;
        multiply_scalar(*x, 0, accumulator);
        accumulator.meta.scale = target_scale;

        const auto &coefficients = configured_coefficients;
        if (!coefficients.front().empty())
        {
            add_plain_inplace(
                accumulator,
                coefficients.front(),
                "GpuEvaluator::eval_mod_high_precision Horner constant");
            accumulator.meta.scale = target_scale;
        }

        for (std::size_t i = 1; i < coefficients.size(); ++i)
        {
            const GpuCiphertextData *x_at_level = x;
            GpuCiphertextData x_dropped;
            if (!(x->meta.parms_id == accumulator.meta.parms_id))
            {
                if (x->meta.q_count < accumulator.meta.q_count)
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: x level is below accumulator level");
                }
                drop_modulus(*x, x_dropped, accumulator.meta.parms_id);
                x_dropped.meta.scale = target_scale;
                x_at_level = &x_dropped;
            }

            multiply(accumulator, *x_at_level, workspace.scratch3);
            relinearize(workspace.scratch3, relin_keys, workspace.scratch4);
            rescale_many(
                workspace.scratch4,
                workspace.scratch5,
                logical_rescale_count);
            workspace.scratch5.meta.scale = target_scale;
            accumulator = std::move(workspace.scratch5);

            if (!coefficients[i].empty())
            {
                add_plain_inplace(
                    accumulator,
                    coefficients[i],
                    "GpuEvaluator::eval_mod_high_precision Horner coefficient");
                accumulator.meta.scale = target_scale;
            }
        }
        stage_recorder.record(3);
        stage_recorder.record(4);

        for (const auto &double_angle_plaintext : configured_double_angle_constants)
        {
            square(accumulator, workspace.scratch3);
            relinearize(workspace.scratch3, relin_keys, workspace.scratch4);
            rescale_many(
                workspace.scratch4,
                workspace.scratch5,
                logical_rescale_count);
            workspace.scratch5.meta.scale = target_scale;
            add(workspace.scratch5, workspace.scratch5, workspace.scratch2);

            if (!double_angle_plaintext.empty())
            {
                add_plain_inplace(
                    workspace.scratch2,
                    double_angle_plaintext,
                    "GpuEvaluator::eval_mod_high_precision Horner double-angle");
                accumulator = std::move(workspace.scratch2);
            }
            else
            {
                accumulator = std::move(workspace.scratch2);
            }
            accumulator.meta.scale = target_scale;
        }
        stage_recorder.record(5);

        *destinations[0] = std::move(accumulator);
        destinations[0]->meta.scale = source_ciphertext.meta.scale;
        stage_recorder.finish(workspace.eval_mod_stage_timing);
        return;
    }

    std::uint32_t maximum_degree = 1;
    for (const auto &term : polynomial_terms)
    {
        maximum_degree = std::max(maximum_degree, term.degree);
        if (term.coefficient_plaintext.empty())
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: empty term plaintext");
        }
    }
    for (const auto &block : polynomial_blocks)
    {
        if (block.terms.empty())
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: empty polynomial block");
        }
        for (const auto &term : block.terms)
        {
            maximum_degree = std::max(maximum_degree, term.degree);
            if (term.coefficient_plaintext.empty())
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: empty block term plaintext");
            }
        }
    }
    for (const auto &combine : polynomial_combine_steps)
    {
        maximum_degree = std::max(maximum_degree, combine.basis_degree);
    }
    for (const auto &step : basis_steps)
    {
        maximum_degree = std::max(maximum_degree, step.output_degree);
        maximum_degree = std::max(maximum_degree, step.left_degree);
        maximum_degree = std::max(maximum_degree, step.right_degree);
        maximum_degree = std::max(maximum_degree, step.correction_degree);
    }

    std::vector<bool> basis_available(
        static_cast<std::size_t>(maximum_degree) + 1,
        false);
    basis_available[0] = true;
    basis_available[1] = true;
    for (const auto &step : basis_steps)
    {
        if (step.output_degree <= 1 ||
            step.left_degree == 0 ||
            step.right_degree == 0 ||
            !basis_available[step.left_degree] ||
            !basis_available[step.right_degree] ||
            basis_available[step.output_degree])
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: basis plan is not topologically valid");
        }
        if (bootstrap_data.eval_mod.polynomial_basis ==
                GpuEvalModPolynomialBasis::Chebyshev &&
            step.correction_degree != 0 &&
            !basis_available[step.correction_degree])
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: missing Chebyshev correction basis");
        }
        basis_available[step.output_degree] = true;
    }
    for (const auto &term : polynomial_terms)
    {
        if (term.degree != 0 && !basis_available[term.degree])
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: polynomial term is absent from basis plan");
        }
    }
    for (const auto &block : polynomial_blocks)
    {
        for (const auto &term : block.terms)
        {
            if (term.degree != 0 && !basis_available[term.degree])
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: block term is absent from basis plan");
            }
        }
    }
    for (const auto &combine : polynomial_combine_steps)
    {
        if (combine.basis_degree == 0 ||
            !basis_available[combine.basis_degree])
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_high_precision: combine basis is absent from basis plan");
        }
    }

    const auto required_basis_slots =
        static_cast<std::size_t>(maximum_degree) + 1;
    struct EvalModLane
    {
        std::vector<GpuCiphertextData> *basis = nullptr;
        std::vector<GpuCiphertextData> *nodes = nullptr;
        std::array<GpuCiphertextData *, 6> scratch{};
        GpuCiphertextData accumulator;
    };
    EvalModLane lanes[2];
    lanes[0].basis = &workspace.eval_mod_basis;
    lanes[0].nodes = &workspace.eval_mod_nodes;
    lanes[0].scratch = {
        &workspace.scratch0, &workspace.scratch1, &workspace.scratch2,
        &workspace.scratch3, &workspace.scratch4, &workspace.scratch5};
    if (lane_count == 2)
    {
        lanes[1].basis = &workspace.eval_mod_basis_imag;
        lanes[1].nodes = &workspace.eval_mod_nodes_imag;
        lanes[1].scratch = {
            &workspace.eval_mod_scratch_imag[0], &workspace.eval_mod_scratch_imag[1],
            &workspace.eval_mod_scratch_imag[2], &workspace.eval_mod_scratch_imag[3],
            &workspace.eval_mod_scratch_imag[4], &workspace.eval_mod_scratch_imag[5]};
    }
    for (std::size_t lane = 0; lane < lane_count; ++lane)
    {
        if (lanes[lane].basis->size() < required_basis_slots)
        {
            lanes[lane].basis->resize(required_basis_slots);
        }
        auto &b1 = (*lanes[lane].basis)[1];
        if (d2d_free_dataflow)
        {
            NvtxRange range("EvalMod input preparation");
            copy_ciphertext_data(
                *sources[lane],
                b1,
                "GpuEvaluator::eval_mod_high_precision basis T1 input copy");
            b1.meta.scale = configured_input_scale;
            if (!configured_input_offset.empty())
            {
                add_plain_inplace(
                    b1,
                    configured_input_offset,
                    "GpuEvaluator::eval_mod_high_precision basis T1 input offset");
            }
        }
        else
        {
            NvtxRange range("EvalMod input preparation");
            copy_ciphertext_data(
                *sources[lane],
                *lanes[lane].scratch[0],
                "GpuEvaluator::eval_mod_high_precision input copy");
            lanes[lane].scratch[0]->meta.scale = configured_input_scale;
            GpuCiphertextData *prepared_input = lanes[lane].scratch[0];
            if (!configured_input_offset.empty())
            {
                add_plain(
                    *lanes[lane].scratch[0],
                    configured_input_offset,
                    *lanes[lane].scratch[1]);
                lanes[lane].scratch[1]->meta.scale = configured_input_scale;
                prepared_input = lanes[lane].scratch[1];
            }
            copy_ciphertext_data(
                *prepared_input,
                b1,
                "GpuEvaluator::eval_mod_high_precision basis T1 copy");
            b1.meta.scale = configured_input_scale;
        }
    }
    x = &(*lanes[0].basis)[1];
    if (workspace.capture_eval_mod_trace)
    {
        copy_ciphertext_data(
            *x,
            workspace.eval_mod_trace_offset_input,
            "GpuEvaluator::eval_mod_high_precision trace input copy");
        workspace.eval_mod_trace_offset_input.meta.scale = x->meta.scale;
    }
    stage_recorder.record(1);

    auto relinearize_lanes =
        [&](int src_idx, int dst_idx) {
            if (lane_count == 2)
            {
                relinearize_pair(
                    *lanes[0].scratch[src_idx],
                    *lanes[1].scratch[src_idx],
                    relin_keys,
                    *lanes[0].scratch[dst_idx],
                    *lanes[1].scratch[dst_idx]);
            }
            else
            {
                relinearize(
                    *lanes[0].scratch[src_idx],
                    relin_keys,
                    *lanes[0].scratch[dst_idx]);
            }
        };

    auto multiply_relinearize_rescale_lanes =
        [&](std::array<const GpuCiphertextData *, 2> left_in,
            std::array<const GpuCiphertextData *, 2> right_in,
            double expected_output_scale,
            std::uint32_t rescale_count,
            std::array<GpuCiphertextData *, 2> outputs) {
            for (std::size_t lane = 0; lane < lane_count; ++lane)
            {
                auto &L = lanes[lane];
                const GpuCiphertextData *left_at_level = left_in[lane];
                const GpuCiphertextData *right_at_level = right_in[lane];
                if (!(left_at_level->meta.parms_id == right_at_level->meta.parms_id))
                {
                    if (left_at_level->meta.q_count == right_at_level->meta.q_count)
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: equal q_count has different parms_id");
                    }
                    if (left_at_level->meta.q_count > right_at_level->meta.q_count)
                    {
                        drop_modulus(*left_at_level, *L.scratch[3], right_at_level->meta.parms_id);
                        left_at_level = L.scratch[3];
                    }
                    else
                    {
                        drop_modulus(*right_at_level, *L.scratch[4], left_at_level->meta.parms_id);
                        right_at_level = L.scratch[4];
                    }
                }
                if (left_at_level == right_at_level)
                {
                    square(*left_at_level, *L.scratch[5]);
                }
                else
                {
                    multiply(*left_at_level, *right_at_level, *L.scratch[5]);
                }
            }
            if (rescale_count == 2 &&
                use_rescale_x2() &&
                use_relinearize_rescale_x2())
            {
                if (lane_count == 2)
                {
                    relinearize_rescale_x2_hybrid_pair(
                        *lanes[0].scratch[5],
                        *lanes[1].scratch[5],
                        relin_keys,
                        *outputs[0],
                        *outputs[1]);
                }
                else
                {
                    relinearize_rescale_x2_hybrid(
                        *lanes[0].scratch[5],
                        relin_keys,
                        *outputs[0]);
                }
            }
            else
            {
                relinearize_lanes(5, 3);
                for (std::size_t lane = 0; lane < lane_count; ++lane)
                {
                    rescale_many(
                        *lanes[lane].scratch[3],
                        *outputs[lane],
                        rescale_count);
                }
            }
            if (expected_output_scale > 0.0)
            {
                for (std::size_t lane = 0; lane < lane_count; ++lane)
                {
                    outputs[lane]->meta.scale = expected_output_scale;
                }
            }
        };

    auto multiply_plain_accumulate_first_two =
        [&](const GpuCiphertextData &source_ciphertext,
            const GpuPlaintextData &source_plaintext,
            GpuCiphertextData &destination_ciphertext) {
            if (source_ciphertext.empty() || source_plaintext.empty() ||
                destination_ciphertext.empty())
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: empty inline leaf accumulate input");
            }
            if (!(source_ciphertext.meta.parms_id ==
                  source_plaintext.meta.parms_id) ||
                !(source_ciphertext.meta.parms_id ==
                  destination_ciphertext.meta.parms_id) ||
                !source_ciphertext.meta.is_ntt_form ||
                !source_plaintext.meta.is_ntt_form ||
                !destination_ciphertext.meta.is_ntt_form ||
                source_ciphertext.size() != 2 ||
                destination_ciphertext.size() < 2)
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: incompatible inline leaf accumulate input");
            }
            const double product_scale =
                source_ciphertext.meta.scale * source_plaintext.meta.scale;
            if (!same_scale(product_scale, destination_ciphertext.meta.scale))
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: inline leaf accumulate scale mismatch");
            }
            if (!same_logical_shard_layout(
                    source_ciphertext.polys_.front(),
                    source_plaintext.poly_) ||
                !same_logical_shard_layout(
                    source_ciphertext.polys_.front(),
                    destination_ciphertext.polys_.front()) ||
                !all_components_use_layout(
                    source_ciphertext,
                    source_ciphertext.polys_.front()) ||
                !all_components_use_layout(
                    destination_ciphertext,
                    destination_ciphertext.polys_.front()))
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: inline leaf accumulate shard mismatch");
            }

            auto destination_view = destination_ciphertext.make_view();
            auto ciphertext_view = source_ciphertext.make_const_view();
            auto plaintext_view = source_plaintext.make_const_view();
            const auto &level_info =
                params_.get_level(source_ciphertext.meta.parms_id);
            elementwise_handler_.multiply_plain_accumulate_with_ciphertext(
                destination_view,
                ciphertext_view,
                plaintext_view,
                level_info);
        };

    {
        NvtxRange range("EvalMod Chebyshev basis generation");
        for (const auto &step : basis_steps)
        {
            std::array<const GpuCiphertextData *, 2> basis_left{};
            std::array<const GpuCiphertextData *, 2> basis_right{};
            std::array<GpuCiphertextData *, 2> out_ptrs{};
            for (std::size_t lane = 0; lane < lane_count; ++lane)
            {
                auto &L = lanes[lane];
                out_ptrs[lane] = &(*L.basis)[step.output_degree];
                basis_left[lane] = &(*L.basis)[step.left_degree];
                basis_right[lane] = &(*L.basis)[step.right_degree];
            }
            if (!step.operand_alignment_plaintext.empty())
            {
                const bool pair_align_ok =
                    lane_count == 2 &&
                    (step.align_left_operand ? basis_left[0] : basis_right[0])->meta.parms_id ==
                        (step.align_left_operand ? basis_left[1] : basis_right[1])->meta.parms_id;
                if (pair_align_ok)
                {
                    const auto &h0 = step.align_left_operand ? *basis_left[0] : *basis_right[0];
                    const auto &h1 = step.align_left_operand ? *basis_left[1] : *basis_right[1];
                    multiply_plain_pair(
                        h0, h1, step.operand_alignment_plaintext, *lanes[0].scratch[0], *lanes[1].scratch[0]);
                    lanes[0].scratch[0]->meta.scale = step.operand_alignment_pre_rescale_scale;
                    lanes[1].scratch[0]->meta.scale = step.operand_alignment_pre_rescale_scale;
                    rescale_many_pair(
                        *lanes[0].scratch[0], *lanes[1].scratch[0],
                        *lanes[0].scratch[1], *lanes[1].scratch[1],
                        step.operand_alignment_rescale_count);
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        lanes[lane].scratch[1]->meta.scale = step.operand_alignment_output_scale;
                        if (step.align_left_operand)
                        {
                            basis_left[lane] = lanes[lane].scratch[1];
                        }
                        else
                        {
                            basis_right[lane] = lanes[lane].scratch[1];
                        }
                    }
                }
                else
                {
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        auto &L = lanes[lane];
                        const GpuCiphertextData &high_operand = step.align_left_operand
                            ? *basis_left[lane]
                            : *basis_right[lane];
                        multiply_plain(
                            high_operand,
                            step.operand_alignment_plaintext,
                            *L.scratch[0]);
                        L.scratch[0]->meta.scale =
                            step.operand_alignment_pre_rescale_scale;
                        rescale_many(
                            *L.scratch[0],
                            *L.scratch[1],
                            step.operand_alignment_rescale_count);
                        L.scratch[1]->meta.scale =
                            step.operand_alignment_output_scale;
                        if (step.align_left_operand)
                        {
                            basis_left[lane] = L.scratch[1];
                        }
                        else
                        {
                            basis_right[lane] = L.scratch[1];
                        }
                    }
                }
            }
            if (bootstrap_data.eval_mod.dynamic_rescale)
            {
                std::size_t multiply_q_count_0 = 0;
                bool is_square_0 = false;
                const bool paired_basis_mul =
                    lane_count == 2 &&
                    basis_left[0]->meta.parms_id == basis_left[1]->meta.parms_id &&
                    basis_right[0]->meta.parms_id == basis_right[1]->meta.parms_id;
                if (paired_basis_mul)
                {
                    const GpuCiphertextData *bl0 = basis_left[0];
                    const GpuCiphertextData *br0 = basis_right[0];
                    const GpuCiphertextData *bl1 = basis_left[1];
                    const GpuCiphertextData *br1 = basis_right[1];
                    bool use_basis_prefix_multiply = false;
                    parms_id_type basis_prefix_parms_id = parms_id_zero;
                    if (!(bl0->meta.parms_id == br0->meta.parms_id))
                    {
                        if (bl0->meta.q_count == br0->meta.q_count)
                        {
                            throw std::invalid_argument(
                                "GpuEvaluator::eval_mod_high_precision: dynamic basis equal q_count has different parms_id");
                        }
                        basis_prefix_parms_id =
                            bl0->meta.q_count < br0->meta.q_count
                                ? bl0->meta.parms_id
                                : br0->meta.parms_id;
                        use_basis_prefix_multiply =
                            supports_q_prefix_multiply(*bl0, *br0, basis_prefix_parms_id) &&
                            supports_q_prefix_multiply(*bl1, *br1, basis_prefix_parms_id);
                        if (!use_basis_prefix_multiply && bl0->meta.q_count > br0->meta.q_count)
                        {
                            drop_modulus(*bl0, *lanes[0].scratch[0], br0->meta.parms_id);
                            drop_modulus(*bl1, *lanes[1].scratch[0], br1->meta.parms_id);
                            bl0 = lanes[0].scratch[0];
                            bl1 = lanes[1].scratch[0];
                        }
                        else if (!use_basis_prefix_multiply)
                        {
                            drop_modulus(*br0, *lanes[0].scratch[1], bl0->meta.parms_id);
                            drop_modulus(*br1, *lanes[1].scratch[1], bl1->meta.parms_id);
                            br0 = lanes[0].scratch[1];
                            br1 = lanes[1].scratch[1];
                        }
                    }
                    is_square_0 = !use_basis_prefix_multiply && bl0 == br0 && bl1 == br1;
                    multiply_q_count_0 =
                        use_basis_prefix_multiply
                            ? params_.get_level(basis_prefix_parms_id).q_count
                            : bl0->meta.q_count;
                    if (use_basis_prefix_multiply)
                    {
                        multiply_q_prefix_pair(
                            *bl0, *br0, *bl1, *br1, basis_prefix_parms_id,
                            *lanes[0].scratch[5], *lanes[1].scratch[5]);
                    }
                    else if (is_square_0)
                    {
                        square_pair(
                            *bl0, *bl1, *lanes[0].scratch[5], *lanes[1].scratch[5]);
                    }
                    else
                    {
                        multiply_pair(
                            *bl0, *br0, *bl1, *br1,
                            *lanes[0].scratch[5], *lanes[1].scratch[5]);
                    }
                }
                else
                {
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        auto &L = lanes[lane];
                        const GpuCiphertextData *bl = basis_left[lane];
                        const GpuCiphertextData *br = basis_right[lane];
                        bool use_basis_prefix_multiply = false;
                        parms_id_type basis_prefix_parms_id = parms_id_zero;
                        if (!(bl->meta.parms_id == br->meta.parms_id))
                        {
                            if (bl->meta.q_count == br->meta.q_count)
                            {
                                throw std::invalid_argument(
                                    "GpuEvaluator::eval_mod_high_precision: dynamic basis equal q_count has different parms_id");
                            }
                            basis_prefix_parms_id =
                                bl->meta.q_count < br->meta.q_count
                                    ? bl->meta.parms_id
                                    : br->meta.parms_id;
                            use_basis_prefix_multiply =
                                supports_q_prefix_multiply(
                                    *bl, *br, basis_prefix_parms_id);
                            if (!use_basis_prefix_multiply &&
                                bl->meta.q_count > br->meta.q_count)
                            {
                                drop_modulus(*bl, *L.scratch[0], br->meta.parms_id);
                                bl = L.scratch[0];
                            }
                            else if (!use_basis_prefix_multiply)
                            {
                                drop_modulus(*br, *L.scratch[1], bl->meta.parms_id);
                                br = L.scratch[1];
                            }
                        }
                        const bool is_square =
                            !use_basis_prefix_multiply && bl == br;
                        const std::size_t multiply_q_count =
                            use_basis_prefix_multiply
                                ? params_.get_level(basis_prefix_parms_id).q_count
                                : bl->meta.q_count;
                        if (lane == 0)
                        {
                            multiply_q_count_0 = multiply_q_count;
                            is_square_0 = is_square;
                        }
                        if (use_basis_prefix_multiply)
                        {
                            multiply_q_prefix(*bl, *br, basis_prefix_parms_id, *L.scratch[5]);
                        }
                        else if (is_square)
                        {
                            square(*bl, *L.scratch[5]);
                        }
                        else
                        {
                            multiply(*bl, *br, *L.scratch[5]);
                        }
                    }
                }
                if (workspace.capture_eval_mod_stage_timing)
                {
                    multiply_recorder.begin(
                        "EvalMod.mul.basis.T" +
                            std::to_string(step.output_degree) +
                            "=" + std::to_string(step.left_degree) +
                            "x" + std::to_string(step.right_degree) +
                            ".q" + std::to_string(multiply_q_count_0),
                        multiply_q_count_0,
                        relin_keys.meta.p_count,
                        is_square_0);
                }
                relinearize_lanes(5, 3);
                multiply_recorder.end();

                const bool paired_cheb_ok =
                    bootstrap_data.eval_mod.polynomial_basis ==
                        GpuEvalModPolynomialBasis::Chebyshev &&
                    (lane_count == 1 ||
                     (lanes[0].scratch[3]->meta.parms_id == lanes[1].scratch[3]->meta.parms_id &&
                      (step.correction_degree == 0 ||
                       (*lanes[0].basis)[step.correction_degree].meta.parms_id ==
                           (*lanes[1].basis)[step.correction_degree].meta.parms_id)));

                if (paired_cheb_ok)
                {
                    if (step.correction_plaintext.empty())
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: missing dynamic Chebyshev correction plaintext");
                    }
                    const GpuCiphertextData *src_b =
                        lane_count == 2 ? lanes[1].scratch[3] : nullptr;
                    GpuCiphertextData *dst_b =
                        lane_count == 2 ? out_ptrs[1] : nullptr;
                    if (step.correction_degree == 0)
                    {
                        double_sub_or_add_plain_pair(
                            kernel::DoublePlainPairOp::SubPlainC0,
                            *lanes[0].scratch[3],
                            src_b,
                            step.correction_plaintext,
                            nullptr,
                            nullptr,
                            *out_ptrs[0],
                            dst_b,
                            "GpuEvaluator::eval_mod_high_precision dynamic Chebyshev constant");
                    }
                    else
                    {
                        const GpuCiphertextData *src_a = lanes[0].scratch[3];
                        const GpuCiphertextData *corr_a =
                            &(*lanes[0].basis)[step.correction_degree];
                        const GpuCiphertextData *corr_b =
                            lane_count == 2 ? &(*lanes[1].basis)[step.correction_degree] : nullptr;
                        if (src_a->meta.q_count > corr_a->meta.q_count)
                        {
                            for (std::size_t lane = 0; lane < lane_count; ++lane)
                            {
                                if (!q_prefix_views ||
                                    !drop_modulus_inplace(
                                        *lanes[lane].scratch[3],
                                        corr_a->meta.parms_id))
                                {
                                    drop_modulus(
                                        *lanes[lane].scratch[3],
                                        *lanes[lane].scratch[0],
                                        corr_a->meta.parms_id);
                                    if (lane == 0)
                                    {
                                        src_a = lanes[0].scratch[0];
                                    }
                                    else
                                    {
                                        src_b = lanes[1].scratch[0];
                                    }
                                }
                            }
                        }
                        else if (src_a->meta.parms_id != corr_a->meta.parms_id &&
                                 src_a->meta.q_count == corr_a->meta.q_count)
                        {
                            throw std::invalid_argument(
                                "GpuEvaluator::eval_mod_high_precision: dynamic Chebyshev correction parms_id mismatch");
                        }
                        double_sub_or_add_plain_pair(
                            kernel::DoublePlainPairOp::SubCorr,
                            *src_a,
                            src_b,
                            step.correction_plaintext,
                            corr_a,
                            corr_b,
                            *out_ptrs[0],
                            dst_b,
                            "GpuEvaluator::eval_mod_high_precision dynamic Chebyshev correction");
                    }
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        out_ptrs[lane]->meta.scale = step.pre_rescale_scale;
                    }
                }
                else
                {
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        auto &L = lanes[lane];
                        auto &output = *out_ptrs[lane];
                        output = std::move(*L.scratch[3]);
                        output.meta.scale = step.pre_rescale_scale;

                        if (bootstrap_data.eval_mod.polynomial_basis ==
                            GpuEvalModPolynomialBasis::Chebyshev)
                        {
                            add(output, output, *L.scratch[3]);
                            output = std::move(*L.scratch[3]);
                            output.meta.scale = step.pre_rescale_scale;

                            if (step.correction_plaintext.empty())
                            {
                                throw std::invalid_argument(
                                    "GpuEvaluator::eval_mod_high_precision: missing dynamic Chebyshev correction plaintext");
                            }
                            if (step.correction_degree == 0)
                            {
                                sub_plain_inplace(
                                    output,
                                    step.correction_plaintext,
                                    "GpuEvaluator::eval_mod_high_precision dynamic Chebyshev constant");
                                output.meta.scale = step.pre_rescale_scale;
                            }
                            else
                            {
                                const auto &correction_basis =
                                    (*L.basis)[step.correction_degree];
                                multiply_plain(
                                    correction_basis,
                                    step.correction_plaintext,
                                    *L.scratch[4]);
                                L.scratch[4]->meta.scale = step.pre_rescale_scale;

                                const GpuCiphertextData *left = &output;
                                const GpuCiphertextData *right = L.scratch[4];
                                if (!(left->meta.parms_id == right->meta.parms_id))
                                {
                                    if (left->meta.q_count > right->meta.q_count)
                                    {
                                        if (!q_prefix_views ||
                                            !drop_modulus_inplace(
                                                output,
                                                right->meta.parms_id))
                                        {
                                            drop_modulus(
                                                *left,
                                                *L.scratch[0],
                                                right->meta.parms_id);
                                            left = L.scratch[0];
                                        }
                                    }
                                    else if (right->meta.q_count > left->meta.q_count)
                                    {
                                        if (!q_prefix_views ||
                                            !drop_modulus_inplace(
                                                *L.scratch[4],
                                                left->meta.parms_id))
                                        {
                                            drop_modulus(
                                                *right,
                                                *L.scratch[1],
                                                left->meta.parms_id);
                                            right = L.scratch[1];
                                        }
                                    }
                                    else
                                    {
                                        throw std::invalid_argument(
                                            "GpuEvaluator::eval_mod_high_precision: dynamic Chebyshev correction parms_id mismatch");
                                    }
                                }
                                sub(*left, *right, *L.scratch[5]);
                                output = std::move(*L.scratch[5]);
                                output.meta.scale = step.pre_rescale_scale;
                            }
                        }
                    }
                }

                if (lane_count == 2 &&
                    out_ptrs[0]->meta.parms_id == out_ptrs[1]->meta.parms_id &&
                    same_scale(out_ptrs[0]->meta.scale, out_ptrs[1]->meta.scale))
                {
                    if (d2d_free_dataflow)
                    {
                        rescale_dynamic_pair(
                            *out_ptrs[0], *out_ptrs[1],
                            *out_ptrs[0], *out_ptrs[1],
                            target_scale);
                    }
                    else
                    {
                        rescale_dynamic_pair(
                            *out_ptrs[0], *out_ptrs[1],
                            *lanes[0].scratch[2], *lanes[1].scratch[2],
                            target_scale);
                        *out_ptrs[0] = std::move(*lanes[0].scratch[2]);
                        *out_ptrs[1] = std::move(*lanes[1].scratch[2]);
                    }
                }
                else
                {
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        auto &L = lanes[lane];
                        auto &output = *out_ptrs[lane];
                        if (d2d_free_dataflow)
                        {
                            rescale_dynamic(output, output, target_scale);
                        }
                        else
                        {
                            rescale_dynamic(output, *L.scratch[2], target_scale);
                            output = std::move(*L.scratch[2]);
                        }
                    }
                }
                for (std::size_t lane = 0; lane < lane_count; ++lane)
                {
                    out_ptrs[lane]->meta.scale = step.output_scale;
                }
                continue;
            }

            if (workspace.capture_eval_mod_stage_timing)
            {
                multiply_recorder.begin(
                    "EvalMod.mul.basis.T" +
                        std::to_string(step.output_degree) +
                        "=" + std::to_string(step.left_degree) +
                        "x" + std::to_string(step.right_degree) +
                        ".q" + std::to_string(basis_left[0]->meta.q_count),
                    basis_left[0]->meta.q_count,
                    relin_keys.meta.p_count,
                    basis_left[0] == basis_right[0]);
            }
            multiply_relinearize_rescale_lanes(
                basis_left,
                basis_right,
                step.output_scale,
                step.rescale_count,
                out_ptrs);
            multiply_recorder.end();

            if (bootstrap_data.eval_mod.polynomial_basis !=
                GpuEvalModPolynomialBasis::Chebyshev)
            {
                continue;
            }

            for (std::size_t lane = 0; lane < lane_count; ++lane)
            {
                auto &L = lanes[lane];
                auto &output = *out_ptrs[lane];
                add(output, output, *L.scratch[3]);
                output = std::move(*L.scratch[3]);
                output.meta.scale = step.output_scale;

                if (step.correction_degree == 0)
                {
                    if (step.correction_plaintext.empty())
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: missing Chebyshev constant correction");
                    }
                    sub_plain_inplace(
                        output,
                        step.correction_plaintext,
                        "GpuEvaluator::eval_mod_high_precision Chebyshev constant");
                    output.meta.scale = step.output_scale;
                    continue;
                }

                const GpuCiphertextData *correction =
                    &(*L.basis)[step.correction_degree];
                if (!step.correction_alignment_plaintext.empty())
                {
                    multiply_plain(
                        *correction,
                        step.correction_alignment_plaintext,
                        *L.scratch[0]);
                    L.scratch[0]->meta.scale =
                        step.correction_alignment_pre_rescale_scale;
                    rescale_many(
                        *L.scratch[0],
                        *L.scratch[1],
                        step.correction_alignment_rescale_count);
                    L.scratch[1]->meta.scale = step.output_scale;
                    correction = L.scratch[1];
                }
                if (!(output.meta.parms_id == correction->meta.parms_id))
                {
                    if (output.meta.q_count > correction->meta.q_count)
                    {
                        drop_modulus(output, *L.scratch[4], correction->meta.parms_id);
                        output = std::move(*L.scratch[4]);
                    }
                    else if (correction->meta.q_count > output.meta.q_count)
                    {
                        drop_modulus(
                            *correction,
                            *L.scratch[4],
                            output.meta.parms_id);
                        correction = L.scratch[4];
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: Chebyshev correction parms_id mismatch");
                    }
                }
                if (!step.correction_alignment_plaintext.empty())
                {
                    sub(output, *correction, *L.scratch[5]);
                    output = std::move(*L.scratch[5]);
                    output.meta.scale = step.output_scale;
                    continue;
                }
                if (step.correction_plaintext.empty() ||
                    step.correction_plaintext.meta.parms_id != output.meta.parms_id)
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: invalid Chebyshev correction scale plaintext");
                }
                multiply_plain(
                    *correction,
                    step.correction_plaintext,
                    *L.scratch[3]);
                L.scratch[3]->meta.scale = step.output_scale;
                sub(output, *L.scratch[3], *L.scratch[5]);
                output = std::move(*L.scratch[5]);
                output.meta.scale = step.output_scale;
            }
        }
    }
    stage_recorder.record(2);

    std::function<bool(
        EvalModLane &,
        const std::vector<GpuEvalModPolynomialTerm> &,
        std::uint32_t,
        double,
        std::size_t,
        GpuCiphertextData &)> evaluate_term_block_caccum;

    auto evaluate_term_block =
        [&](EvalModLane &L,
            const std::vector<GpuEvalModPolynomialTerm> &terms,
            std::uint32_t rescale_count,
            double expected_output_scale,
            std::size_t expected_output_q_count,
            GpuCiphertextData &block_output) {
            if (evaluate_term_block_caccum &&
                evaluate_term_block_caccum(
                    L,
                    terms,
                    rescale_count,
                    expected_output_scale,
                    expected_output_q_count,
                    block_output))
            {
                return;
            }

            GpuCiphertextData block_accumulator;
            double block_scale = 0.0;

            auto accumulate_term = [&](GpuCiphertextData &term_ciphertext) {
                if (!(block_scale > 0.0))
                {
                    block_scale = term_ciphertext.meta.scale;
                }
                else if (!same_scale(block_scale, term_ciphertext.meta.scale))
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: polynomial term scale mismatch");
                }
                if (block_accumulator.empty())
                {
                    block_accumulator = std::move(term_ciphertext);
                    return;
                }

                const GpuCiphertextData *left = &block_accumulator;
                const GpuCiphertextData *right = &term_ciphertext;
                if (!(left->meta.parms_id == right->meta.parms_id))
                {
                    if (left->meta.q_count > right->meta.q_count)
                    {
                        drop_modulus(*left, *L.scratch[3], right->meta.parms_id);
                        left = L.scratch[3];
                    }
                    else if (right->meta.q_count > left->meta.q_count)
                    {
                        drop_modulus(*right, *L.scratch[4], left->meta.parms_id);
                        right = L.scratch[4];
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: term parms_id mismatch");
                    }
                }
                add(*left, *right, *L.scratch[5]);
                block_accumulator = std::move(*L.scratch[5]);
                block_accumulator.meta.scale = block_scale;
            };

            std::size_t previous_term_q_count = 0;
            for (const auto &term : terms)
            {
                if (term.degree == 0)
                {
                    continue;
                }
                if (!bootstrap_data.eval_mod.dynamic_rescale &&
                    term.coefficient_plaintext.meta.q_count <
                        previous_term_q_count)
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: block terms are not in setup-time q_count order");
                }
                previous_term_q_count =
                    term.coefficient_plaintext.meta.q_count;

                const auto &basis = (*L.basis)[term.degree];
                const GpuCiphertextData *basis_at_level = &basis;
                if (!(basis.meta.parms_id ==
                      term.coefficient_plaintext.meta.parms_id))
                {
                    if (basis.meta.q_count <=
                        term.coefficient_plaintext.meta.q_count)
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: coefficient plaintext is above its basis level");
                    }
                    drop_modulus(
                        basis,
                        *L.scratch[3],
                        term.coefficient_plaintext.meta.parms_id);
                    basis_at_level = L.scratch[3];
                }
                if (!bootstrap_data.eval_mod
                         .rescale_polynomial_terms_individually &&
                    !block_accumulator.empty() &&
                    block_accumulator.meta.parms_id ==
                        basis_at_level->meta.parms_id)
                {
                    multiply_plain_accumulate(
                        *basis_at_level,
                        term.coefficient_plaintext,
                        block_accumulator);
                    continue;
                }
                multiply_plain(
                    *basis_at_level,
                    term.coefficient_plaintext,
                    *L.scratch[2]);
                if (bootstrap_data.eval_mod.rescale_polynomial_terms_individually)
                {
                    rescale_many(
                        *L.scratch[2],
                        *L.scratch[4],
                        rescale_count);
                    L.scratch[4]->meta.scale = expected_output_scale;
                    *L.scratch[2] = std::move(*L.scratch[4]);
                }
                accumulate_term(*L.scratch[2]);
            }

            for (const auto &term : terms)
            {
                if (term.degree != 0)
                {
                    continue;
                }
                if (block_accumulator.empty())
                {
                    multiply_scalar(
                        (*L.basis)[1],
                        0,
                        block_accumulator);
                    block_scale = term.coefficient_plaintext.meta.scale;
                    block_accumulator.meta.scale = block_scale;
                }
                if (!(block_accumulator.meta.parms_id ==
                      term.coefficient_plaintext.meta.parms_id))
                {
                    if (block_accumulator.meta.q_count <=
                        term.coefficient_plaintext.meta.q_count)
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: constant plaintext is above block level");
                    }
                    drop_modulus(
                        block_accumulator,
                        *L.scratch[3],
                        term.coefficient_plaintext.meta.parms_id);
                    block_accumulator = std::move(*L.scratch[3]);
                }
                add_plain_inplace(
                    block_accumulator,
                    term.coefficient_plaintext,
                    "GpuEvaluator::eval_mod_high_precision leaf constant");
                block_accumulator.meta.scale = block_scale;
            }

            if (block_accumulator.empty())
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: polynomial block has no evaluable terms");
            }

            if (!bootstrap_data.eval_mod.rescale_polynomial_terms_individually &&
                rescale_count > 0)
            {
                rescale_many(
                    block_accumulator,
                    *L.scratch[2],
                    rescale_count);
                block_accumulator = std::move(*L.scratch[2]);
            }
            if (expected_output_scale > 0.0)
            {
                block_accumulator.meta.scale = expected_output_scale;
            }
            if (expected_output_q_count > 0 &&
                block_accumulator.meta.q_count > expected_output_q_count)
            {
                const auto &target_level =
                    params_.get_level_by_q_count(expected_output_q_count, 0);
                drop_modulus(
                    block_accumulator,
                    *L.scratch[2],
                    target_level.parms_id);
                block_accumulator = std::move(*L.scratch[2]);
                if (expected_output_scale > 0.0)
                {
                    block_accumulator.meta.scale = expected_output_scale;
                }
            }
            block_output = std::move(block_accumulator);
        };

    auto add_aligned =
        [&](EvalModLane &L,
            const GpuCiphertextData &left_input,
            const GpuCiphertextData &right_input,
            double expected_output_scale,
            GpuCiphertextData &output) {
            const GpuCiphertextData *left = &left_input;
            const GpuCiphertextData *right = &right_input;
            if (!(left->meta.parms_id == right->meta.parms_id))
            {
                if (left->meta.q_count > right->meta.q_count)
                {
                    drop_modulus(*left, *L.scratch[3], right->meta.parms_id);
                    left = L.scratch[3];
                }
                else if (right->meta.q_count > left->meta.q_count)
                {
                    drop_modulus(*right, *L.scratch[4], left->meta.parms_id);
                    right = L.scratch[4];
                }
                else
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: combine parms_id mismatch");
                }
            }
            add(*left, *right, *L.scratch[5]);
            if (expected_output_scale > 0.0)
            {
                L.scratch[5]->meta.scale = expected_output_scale;
            }
            output = std::move(*L.scratch[5]);
        };

    evaluate_term_block_caccum =
        [&](EvalModLane &L,
            const std::vector<GpuEvalModPolynomialTerm> &terms,
            std::uint32_t rescale_count,
            double expected_output_scale,
            std::size_t expected_output_q_count,
            GpuCiphertextData &block_output) {
            const bool dynamic_leaf =
                bootstrap_data.eval_mod.dynamic_rescale;
            if (!use_evalmod_caccum_leaf() ||
                (dynamic_leaf && !d2d_free_dataflow) ||
                bootstrap_data.eval_mod.rescale_polynomial_terms_individually ||
                (!dynamic_leaf && rescale_count != logical_rescale_count) ||
                (dynamic_leaf &&
                 (rescale_count != 0 || expected_output_q_count == 0)) ||
                terms.empty())
            {
                return false;
            }

            const GpuEvalModPolynomialTerm *first_nonconstant = nullptr;
            for (const auto &term : terms)
            {
                if (term.degree != 0)
                {
                    first_nonconstant = &term;
                    break;
                }
            }
            if (first_nonconstant == nullptr ||
                first_nonconstant->coefficient_plaintext.empty())
            {
                return false;
            }

            const auto &reference_plaintext =
                first_nonconstant->coefficient_plaintext;
            if (!reference_plaintext.meta.is_ntt_form ||
                reference_plaintext.meta.p_count != 0 ||
                reference_plaintext.poly_.shards.size() != 1 ||
                reference_plaintext.fields_.empty())
            {
                return false;
            }
            const std::size_t target_q_count =
                dynamic_leaf
                    ? expected_output_q_count
                    : reference_plaintext.meta.q_count;
            const auto &target_level =
                params_.get_level_by_q_count(target_q_count, 0);
            const auto target_parms_id = target_level.parms_id;
            if (first_nonconstant->degree >= L.basis->size() ||
                (*L.basis)[first_nonconstant->degree].empty())
            {
                return false;
            }
            const double pre_rescale_scale =
                (*L.basis)[first_nonconstant->degree].meta.scale *
                reference_plaintext.meta.scale;
            if (!(pre_rescale_scale > 0.0) ||
                !std::isfinite(pre_rescale_scale))
            {
                return false;
            }

            for (const auto &term : terms)
            {
                if (term.coefficient_plaintext.empty() ||
                    term.coefficient_plaintext.meta.q_count < target_q_count ||
                    term.coefficient_plaintext.meta.degree !=
                        source_ciphertext.meta.degree ||
                    term.coefficient_plaintext.meta.p_count != 0 ||
                    !term.coefficient_plaintext.meta.is_ntt_form ||
                    term.coefficient_plaintext.poly_.shards.size() != 1)
                {
                    return false;
                }
                if (term.degree == 0)
                {
                    if (!same_scale(
                            term.coefficient_plaintext.meta.scale,
                            pre_rescale_scale))
                    {
                        return false;
                    }
                    continue;
                }
                if (term.degree >= L.basis->size())
                {
                    return false;
                }
                const auto &basis = (*L.basis)[term.degree];
                if (basis.empty() ||
                    basis.size() != 2 ||
                    !basis.meta.is_ntt_form ||
                    basis.meta.p_count != 0 ||
                    basis.meta.degree != source_ciphertext.meta.degree ||
                    basis.meta.q_count < target_q_count ||
                    basis.polys_[0].shards.size() != 1 ||
                    basis.polys_[1].shards.size() != 1)
                {
                    return false;
                }
                const double term_scale =
                    basis.meta.scale *
                    term.coefficient_plaintext.meta.scale;
                if (!same_scale(term_scale, pre_rescale_scale))
                {
                    return false;
                }
            }

            const int device_id = reference_plaintext.fields_.front().device_id;
            GpuPolyShard accumulator_shard;
            accumulator_shard.field_index = 0;
            accumulator_shard.field_offset = 0;
            accumulator_shard.limb_begin = 0;
            accumulator_shard.limb_count = target_q_count;
            accumulator_shard.coeff_begin = 0;
            accumulator_shard.coeff_count = source_ciphertext.meta.degree;
            auto accumulator =
                GpuCiphertextData::allocate_single_device_sharded(
                    source_ciphertext.meta.degree,
                    target_q_count,
                    2,
                    device_id,
                    std::vector<GpuPolyShard>{accumulator_shard},
                    0);
            accumulator.meta = source_ciphertext.meta;
            accumulator.meta.parms_id = target_parms_id;
            accumulator.meta.scale = pre_rescale_scale;
            accumulator.meta.q_count = target_q_count;
            accumulator.meta.p_count = 0;
            accumulator.meta.component_count = 2;
            accumulator.meta.is_ntt_form = true;

            auto accumulator_view = accumulator.make_view();
            if (accumulator_view.polys.size() != 2 ||
                accumulator_view.polys[0].shards.size() != 1 ||
                accumulator_view.polys[1].shards.size() != 1)
            {
                return false;
            }
            const auto &destination0 =
                accumulator_view.polys[0].shards.front();
            const auto &destination1 =
                accumulator_view.polys[1].shards.front();

            const GpuParameterShard *parameter_shard = nullptr;
            for (const auto &candidate : target_level.shards)
            {
                if (candidate.device_id == destination0.device_id &&
                    destination0.limb_begin >= candidate.limb_begin &&
                    destination0.limb_begin + destination0.limb_count <=
                        candidate.limb_begin + candidate.limb_count)
                {
                    parameter_shard = &candidate;
                    break;
                }
            }
            if (parameter_shard == nullptr)
            {
                return false;
            }

            constexpr std::size_t terms_per_launch = 4;
            std::array<GpuConstPolyShardView, terms_per_launch> c0_terms{};
            std::array<GpuConstPolyShardView, terms_per_launch> c1_terms{};
            std::array<GpuConstPolyShardView, terms_per_launch> plain_terms{};
            GpuConstPolyShardView constant_plaintext{};
            bool has_constant_plaintext = false;
            std::size_t batch_count = 0;
            bool wrote_accumulator = false;

            for (const auto &term : terms)
            {
                if (term.degree != 0)
                {
                    continue;
                }
                if (has_constant_plaintext)
                {
                    return false;
                }
                const auto plaintext_view =
                    term.coefficient_plaintext.make_const_view();
                constant_plaintext = plaintext_view.poly.shards.front();
                if (constant_plaintext.device_id != destination0.device_id ||
                    constant_plaintext.limb_begin != 0 ||
                    constant_plaintext.limb_count < target_q_count ||
                    constant_plaintext.coeff_begin != destination0.coeff_begin ||
                    constant_plaintext.coeff_count != destination0.coeff_count)
                {
                    return false;
                }
                constant_plaintext.limb_count = target_q_count;
                has_constant_plaintext = true;
            }

            const auto flush_batch = [&]() {
                if (batch_count == 0)
                {
                    return;
                }
                kernel::launch_multiply_plain_caccumulate_two_components_4(
                    destination0,
                    destination1,
                    c0_terms.data(),
                    c1_terms.data(),
                    plain_terms.data(),
                    batch_count,
                    !wrote_accumulator && has_constant_plaintext
                        ? &constant_plaintext
                        : nullptr,
                    wrote_accumulator,
                    *parameter_shard,
                    target_level.degree);
                wrote_accumulator = true;
                batch_count = 0;
            };

            for (const auto &term : terms)
            {
                if (term.degree == 0)
                {
                    continue;
                }
                const auto &basis = (*L.basis)[term.degree];
                const auto basis_view = basis.make_const_view();
                const auto plaintext_view =
                    term.coefficient_plaintext.make_const_view();
                auto c0_view = basis_view.polys[0].shards.front();
                auto c1_view = basis_view.polys[1].shards.front();
                auto plain_view = plaintext_view.poly.shards.front();
                if (c0_view.device_id != destination0.device_id ||
                    c1_view.device_id != destination0.device_id ||
                    plain_view.device_id != destination0.device_id ||
                    c0_view.limb_begin != 0 ||
                    c1_view.limb_begin != 0 ||
                    plain_view.limb_begin != 0 ||
                    c0_view.coeff_begin != destination0.coeff_begin ||
                    c1_view.coeff_begin != destination0.coeff_begin ||
                    plain_view.coeff_begin != destination0.coeff_begin ||
                    c0_view.coeff_count != destination0.coeff_count ||
                    c1_view.coeff_count != destination0.coeff_count ||
                    plain_view.coeff_count != destination0.coeff_count ||
                    c0_view.limb_count < destination0.limb_count ||
                    c1_view.limb_count < destination0.limb_count ||
                    plain_view.limb_count < destination0.limb_count)
                {
                    return false;
                }
                c0_view.limb_count = destination0.limb_count;
                c1_view.limb_count = destination0.limb_count;
                plain_view.limb_count = destination0.limb_count;
                c0_terms[batch_count] = c0_view;
                c1_terms[batch_count] = c1_view;
                plain_terms[batch_count] = plain_view;
                ++batch_count;
                if (batch_count == terms_per_launch)
                {
                    flush_batch();
                }
            }
            flush_batch();

            if (!wrote_accumulator)
            {
                return false;
            }

            if (dynamic_leaf)
            {
                accumulator.meta.scale = expected_output_scale;
                block_output = std::move(accumulator);
            }
            else
            {
                rescale_many(
                    accumulator,
                    *L.scratch[2],
                    rescale_count);
                L.scratch[2]->meta.scale = expected_output_scale;
                block_output = std::move(*L.scratch[2]);
            }
            return true;
        };

    auto accumulate_leaf_pre_rescale =
        [&](EvalModLane &L,
            const GpuEvalModPolynomialBlock &block,
            GpuCiphertextData &accumulator_pre_rescale) {
            if (bootstrap_data.eval_mod.rescale_polynomial_terms_individually ||
                block.rescale_count != logical_rescale_count)
            {
                return false;
            }
            if (accumulator_pre_rescale.empty() ||
                accumulator_pre_rescale.size() < 3 ||
                !accumulator_pre_rescale.meta.is_ntt_form)
            {
                return false;
            }

            const double accumulator_scale =
                accumulator_pre_rescale.meta.scale;

            for (const auto &term : block.terms)
            {
                if (term.degree == 0)
                {
                    continue;
                }
                if (term.coefficient_plaintext.empty() ||
                    !(term.coefficient_plaintext.meta.parms_id ==
                      accumulator_pre_rescale.meta.parms_id))
                {
                    return false;
                }
                if (term.degree >= L.basis->size())
                {
                    return false;
                }

                const auto &basis = (*L.basis)[term.degree];
                if (basis.empty())
                {
                    return false;
                }
                if (!(basis.meta.parms_id ==
                      term.coefficient_plaintext.meta.parms_id))
                {
                    if (basis.meta.q_count <=
                        term.coefficient_plaintext.meta.q_count)
                    {
                        return false;
                    }
                }
                const double term_scale =
                    basis.meta.scale *
                    term.coefficient_plaintext.meta.scale;
                if (!same_scale(term_scale, accumulator_scale))
                {
                    return false;
                }
            }

            for (const auto &term : block.terms)
            {
                if (term.degree != 0)
                {
                    continue;
                }
                if (term.coefficient_plaintext.empty() ||
                    !(term.coefficient_plaintext.meta.parms_id ==
                      accumulator_pre_rescale.meta.parms_id) ||
                    !same_scale(
                        term.coefficient_plaintext.meta.scale,
                        accumulator_scale))
                {
                    return false;
                }
            }

            bool accumulated_any = false;
            for (const auto &term : block.terms)
            {
                if (term.degree == 0)
                {
                    continue;
                }
                const auto &basis = (*L.basis)[term.degree];
                const GpuCiphertextData *basis_at_level = &basis;
                if (!(basis.meta.parms_id ==
                      term.coefficient_plaintext.meta.parms_id))
                {
                    drop_modulus(
                        basis,
                        *L.scratch[3],
                        term.coefficient_plaintext.meta.parms_id);
                    basis_at_level = L.scratch[3];
                }
                multiply_plain_accumulate_first_two(
                    *basis_at_level,
                    term.coefficient_plaintext,
                    accumulator_pre_rescale);
                accumulated_any = true;
            }

            for (const auto &term : block.terms)
            {
                if (term.degree != 0)
                {
                    continue;
                }
                add_plain_inplace(
                    accumulator_pre_rescale,
                    term.coefficient_plaintext,
                    "GpuEvaluator::eval_mod_high_precision inline leaf constant");
                accumulator_pre_rescale.meta.scale = accumulator_scale;
                accumulated_any = true;
            }

            return accumulated_any;
        };

    auto multiply_inline_leaf_relinearize_rescale_lanes =
        [&](std::array<const GpuCiphertextData *, 2> left_in,
            std::array<const GpuCiphertextData *, 2> right_in,
            const GpuEvalModPolynomialBlock &leaf_block,
            double expected_output_scale,
            std::array<GpuCiphertextData *, 2> outputs) {
            if (!use_evalmod_inline_leaf())
            {
                return false;
            }

            for (std::size_t lane = 0; lane < lane_count; ++lane)
            {
                auto &L = lanes[lane];
                const GpuCiphertextData *left_at_level = left_in[lane];
                const GpuCiphertextData *right_at_level = right_in[lane];
                if (!(left_at_level->meta.parms_id == right_at_level->meta.parms_id))
                {
                    if (left_at_level->meta.q_count == right_at_level->meta.q_count)
                    {
                        return false;
                    }
                    if (left_at_level->meta.q_count > right_at_level->meta.q_count)
                    {
                        drop_modulus(*left_at_level, *L.scratch[3], right_at_level->meta.parms_id);
                        left_at_level = L.scratch[3];
                    }
                    else
                    {
                        drop_modulus(*right_at_level, *L.scratch[4], left_at_level->meta.parms_id);
                        right_at_level = L.scratch[4];
                    }
                }

                if (left_at_level == right_at_level)
                {
                    square(*left_at_level, *L.scratch[5]);
                }
                else
                {
                    multiply(*left_at_level, *right_at_level, *L.scratch[5]);
                }

                if (!accumulate_leaf_pre_rescale(L, leaf_block, *L.scratch[5]))
                {
                    return false;
                }
            }

            if (logical_rescale_count == 2 &&
                use_rescale_x2() &&
                use_relinearize_rescale_x2())
            {
                if (lane_count == 2)
                {
                    relinearize_rescale_x2_hybrid_pair(
                        *lanes[0].scratch[5],
                        *lanes[1].scratch[5],
                        relin_keys,
                        *outputs[0],
                        *outputs[1]);
                }
                else
                {
                    relinearize_rescale_x2_hybrid(
                        *lanes[0].scratch[5],
                        relin_keys,
                        *outputs[0]);
                }
            }
            else
            {
                relinearize_lanes(5, 3);
                for (std::size_t lane = 0; lane < lane_count; ++lane)
                {
                    rescale_many(
                        *lanes[lane].scratch[3],
                        *outputs[lane],
                        logical_rescale_count);
                }
            }
            if (expected_output_scale > 0.0)
            {
                for (std::size_t lane = 0; lane < lane_count; ++lane)
                {
                    outputs[lane]->meta.scale = expected_output_scale;
                }
            }
            return true;
        };

    if (polynomial_blocks.empty())
    {
        NvtxRange range("EvalMod fused leaf evaluation");
        evaluate_term_block(
            lanes[0],
            polynomial_terms,
            bootstrap_data.eval_mod.polynomial_rescale_count,
            target_scale,
            0,
            lanes[0].accumulator);
        stage_recorder.record(3);
    }
    else
    {
        std::uint32_t maximum_node =
            bootstrap_data.eval_mod.polynomial_result_node;
        maximum_node = std::max(
            maximum_node,
            static_cast<std::uint32_t>(polynomial_blocks.size() - 1));
        for (const auto &combine : polynomial_combine_steps)
        {
            maximum_node = std::max(maximum_node, combine.output_node);
            maximum_node = std::max(maximum_node, combine.quotient_node);
            maximum_node = std::max(maximum_node, combine.remainder_node);
        }
        for (std::size_t lane = 0; lane < lane_count; ++lane)
        {
            if (lanes[lane].nodes->size() <= maximum_node)
            {
                lanes[lane].nodes->resize(
                    static_cast<std::size_t>(maximum_node) + 1);
            }
        }

        std::vector<std::size_t> node_use_counts(
            static_cast<std::size_t>(maximum_node) + 1,
            0);
        std::vector<std::size_t> basis_last_combine_use(
            workspace.eval_mod_basis.size(),
            std::numeric_limits<std::size_t>::max());
        for (std::size_t combine_index = 0;
             combine_index < polynomial_combine_steps.size();
             ++combine_index)
        {
            const auto &combine = polynomial_combine_steps[combine_index];
            if (combine.quotient_node < node_use_counts.size())
            {
                ++node_use_counts[combine.quotient_node];
            }
            if (combine.remainder_node < node_use_counts.size())
            {
                ++node_use_counts[combine.remainder_node];
            }
            if (combine.basis_degree < basis_last_combine_use.size())
            {
                basis_last_combine_use[combine.basis_degree] = combine_index;
            }
        }

        /*
         * A size-3 Q*T+R result may remain unrelinearized when it is consumed
         * only as the remainder of its parent. The parent product is also a
         * size-3 ciphertext, so the two raw ciphertexts can be added and one
         * HYBRID key-switch can relinearize their combined c2 component.
         *
         * Quotient children are deliberately excluded: multiplying a size-3
         * quotient by a size-2 basis would create a size-4 ciphertext and
         * require higher-power relinearization keys. Each candidate remainder
         * edge is checked independently below; an edge that needs an
         * intervening rescale or scale multiplication remains eagerly
         * relinearized. The validated plans are degree 22 with baby width 4,
         * degree 30 with baby width 8, and degree 58/59 with their existing
         * recursive splits.
         */
        const bool lazy_relinearization =
            use_evalmod_lazy_relinearization() &&
            bootstrap_data.eval_mod.dynamic_rescale &&
            (bootstrap_data.eval_mod.polynomial_degree == 22 ||
             bootstrap_data.eval_mod.polynomial_degree == 30 ||
             bootstrap_data.eval_mod.polynomial_degree == 58 ||
             bootstrap_data.eval_mod.polynomial_degree == 59) &&
            bootstrap_data.eval_mod.polynomial_basis ==
                GpuEvalModPolynomialBasis::Chebyshev;
        std::vector<bool> lazy_raw_output(
            static_cast<std::size_t>(maximum_node) + 1,
            false);
        if (lazy_relinearization)
        {
            for (const auto &parent : polynomial_combine_steps)
            {
                if (parent.remainder_node < polynomial_blocks.size() ||
                    parent.remainder_node >= lazy_raw_output.size() ||
                    parent.remainder_node >= node_use_counts.size() ||
                    node_use_counts[parent.remainder_node] != 1 ||
                    parent.remainder_rescale_count != 0 ||
                    !parent.remainder_scale_plaintext.empty())
                {
                    continue;
                }
                lazy_raw_output[parent.remainder_node] = true;
            }
        }

        std::vector<bool> inline_leaf_candidate(polynomial_blocks.size(), false);
        if (use_evalmod_inline_leaf() &&
            !bootstrap_data.eval_mod.dynamic_rescale)
        {
            for (const auto &combine : polynomial_combine_steps)
            {
                if (combine.remainder_node < polynomial_blocks.size() &&
                    combine.remainder_node < node_use_counts.size() &&
                    node_use_counts[combine.remainder_node] == 1)
                {
                    inline_leaf_candidate[combine.remainder_node] = true;
                }
            }
        }

        std::vector<bool> node_available(
            static_cast<std::size_t>(maximum_node) + 1,
            false);
        {
            NvtxRange range("EvalMod fused leaf evaluation");
            for (std::size_t i = 0; i < polynomial_blocks.size(); ++i)
            {
                if (i >= node_available.size())
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: block node id exceeds schedule");
                }
                if (inline_leaf_candidate[i])
                {
                    continue;
                }
                for (std::size_t lane = 0; lane < lane_count; ++lane)
                {
                    evaluate_term_block(
                        lanes[lane],
                        polynomial_blocks[i].terms,
                        polynomial_blocks[i].rescale_count,
                        polynomial_blocks[i].output_scale,
                        polynomial_blocks[i].output_q_count,
                        (*lanes[lane].nodes)[i]);
                }
                node_available[i] = true;
                if (workspace.capture_eval_mod_trace)
                {
                    workspace.eval_mod_trace_polynomial_leaf_outputs.emplace_back();
                    copy_ciphertext_data(
                        (*lanes[0].nodes)[i],
                        workspace.eval_mod_trace_polynomial_leaf_outputs.back(),
                        "GpuEvaluator::eval_mod_high_precision trace leaf copy");
                    workspace.eval_mod_trace_polynomial_leaf_outputs.back().meta.scale =
                        (*lanes[0].nodes)[i].meta.scale;
                }
            }
        }
        stage_recorder.record(3);

        {
            NvtxRange range("EvalMod BSGS tree combine");
            std::array<GpuCiphertextData, 2> comb_prod_scaled;
            std::array<GpuCiphertextData, 2> comb_rem_rescaled;
            std::array<GpuCiphertextData, 2> comb_rem_scaled;
            std::array<GpuCiphertextData, 2> comb_prod_dropped;
            std::array<GpuCiphertextData, 2> comb_rem_dropped;
            std::array<GpuCiphertextData, 2> comb_raw_scratch;
            for (std::size_t combine_index = 0;
                 combine_index < polynomial_combine_steps.size();
                 ++combine_index)
            {
                const auto &combine = polynomial_combine_steps[combine_index];
                const bool remainder_is_inline_leaf =
                    combine.remainder_node < inline_leaf_candidate.size() &&
                    inline_leaf_candidate[combine.remainder_node];
                if (!node_available[combine.quotient_node] ||
                    (!node_available[combine.remainder_node] &&
                     !remainder_is_inline_leaf) ||
                    node_available[combine.output_node])
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: polynomial combine plan is not topologically valid");
                }
                if (remainder_is_inline_leaf &&
                    !node_available[combine.remainder_node])
                {
                    std::array<const GpuCiphertextData *, 2> q_ptrs{};
                    std::array<const GpuCiphertextData *, 2> b_ptrs{};
                    std::array<GpuCiphertextData *, 2> out_ptrs{};
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        q_ptrs[lane] = &(*lanes[lane].nodes)[combine.quotient_node];
                        b_ptrs[lane] = &(*lanes[lane].basis)[combine.basis_degree];
                        out_ptrs[lane] = &(*lanes[lane].nodes)[combine.output_node];
                    }
                    if (multiply_inline_leaf_relinearize_rescale_lanes(
                            q_ptrs, b_ptrs,
                            polynomial_blocks[combine.remainder_node],
                            combine.output_scale, out_ptrs))
                    {
                        node_available[combine.output_node] = true;
                        continue;
                    }
                }
                if (!node_available[combine.remainder_node])
                {
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        evaluate_term_block(
                            lanes[lane],
                            polynomial_blocks[combine.remainder_node].terms,
                            polynomial_blocks[combine.remainder_node].rescale_count,
                            polynomial_blocks[combine.remainder_node].output_scale,
                            polynomial_blocks[combine.remainder_node].output_q_count,
                            (*lanes[lane].nodes)[combine.remainder_node]);
                    }
                    node_available[combine.remainder_node] = true;
                }
                if (bootstrap_data.eval_mod.dynamic_rescale)
                {
                    if (lazy_relinearization)
                    {
                        if (!node_available[combine.quotient_node] ||
                            !node_available[combine.remainder_node])
                        {
                            throw std::invalid_argument(
                                "GpuEvaluator::eval_mod_high_precision: lazy combine input is unavailable");
                        }
                        if (combine.output_q_count == 0 ||
                            combine.product_q_count == 0 ||
                            combine.output_q_count != combine.product_q_count ||
                            !(combine.product_scale > 0.0) ||
                            !std::isfinite(combine.product_scale))
                        {
                            throw std::invalid_argument(
                                "GpuEvaluator::eval_mod_high_precision: unsupported lazy dynamic combine plan");
                        }

                        std::size_t multiply_q_count_0 = 0;
                        bool is_square_0 = false;
                        const bool defer_output =
                            combine.output_node < lazy_raw_output.size() &&
                            lazy_raw_output[combine.output_node];
                        std::array<GpuCiphertextData *, 2> combined_raw_ptrs{
                            defer_output
                                ? &(*lanes[0].nodes)[combine.output_node]
                                : &comb_raw_scratch[0],
                            (lane_count == 2)
                                ? (defer_output
                                       ? &(*lanes[1].nodes)[combine.output_node]
                                       : &comb_raw_scratch[1])
                                : nullptr};

                        const bool symmetric_pair =
                            lane_count == 2 &&
                            (*lanes[0].nodes)[combine.quotient_node].meta.parms_id ==
                                (*lanes[1].nodes)[combine.quotient_node].meta.parms_id &&
                            (*lanes[0].basis)[combine.basis_degree].meta.parms_id ==
                                (*lanes[1].basis)[combine.basis_degree].meta.parms_id &&
                            (*lanes[0].nodes)[combine.remainder_node].meta.parms_id ==
                                (*lanes[1].nodes)[combine.remainder_node].meta.parms_id;

                        if (symmetric_pair)
                        {
                            std::array<const GpuCiphertextData *, 2> q_ptrs{
                                &(*lanes[0].nodes)[combine.quotient_node],
                                &(*lanes[1].nodes)[combine.quotient_node]};
                            std::array<GpuCiphertextData *, 2> q_mut{
                                zero_copy_moddrop &&
                                        combine.quotient_node < node_use_counts.size() &&
                                        node_use_counts[combine.quotient_node] == 1
                                    ? &(*lanes[0].nodes)[combine.quotient_node]
                                    : nullptr,
                                zero_copy_moddrop &&
                                        combine.quotient_node < node_use_counts.size() &&
                                        node_use_counts[combine.quotient_node] == 1
                                    ? &(*lanes[1].nodes)[combine.quotient_node]
                                    : nullptr};
                            if (q_ptrs[0]->size() != 2 || q_ptrs[1]->size() != 2)
                            {
                                throw std::invalid_argument(
                                    "GpuEvaluator::eval_mod_high_precision: lazy quotient must be relinearized");
                            }
                            if (combine.quotient_rescale_count > 0)
                            {
                                rescale_many_pair(
                                    *q_ptrs[0], *q_ptrs[1],
                                    *lanes[0].scratch[2], *lanes[1].scratch[2],
                                    combine.quotient_rescale_count);
                                lanes[0].scratch[2]->meta.scale = combine.quotient_output_scale;
                                lanes[1].scratch[2]->meta.scale = combine.quotient_output_scale;
                                q_ptrs = {lanes[0].scratch[2], lanes[1].scratch[2]};
                                q_mut = {lanes[0].scratch[2], lanes[1].scratch[2]};
                            }

                            std::array<const GpuCiphertextData *, 2> b_ptrs{
                                &(*lanes[0].basis)[combine.basis_degree],
                                &(*lanes[1].basis)[combine.basis_degree]};
                            std::array<GpuCiphertextData *, 2> b_mut{
                                zero_copy_moddrop &&
                                        combine.basis_degree < basis_last_combine_use.size() &&
                                        basis_last_combine_use[combine.basis_degree] == combine_index
                                    ? &(*lanes[0].basis)[combine.basis_degree]
                                    : nullptr,
                                zero_copy_moddrop &&
                                        combine.basis_degree < basis_last_combine_use.size() &&
                                        basis_last_combine_use[combine.basis_degree] == combine_index
                                    ? &(*lanes[1].basis)[combine.basis_degree]
                                    : nullptr};
                            bool use_combine_prefix_multiply = false;
                            parms_id_type combine_prefix_parms_id = parms_id_zero;
                            if (!(q_ptrs[0]->meta.parms_id == b_ptrs[0]->meta.parms_id))
                            {
                                if (q_ptrs[0]->meta.q_count == b_ptrs[0]->meta.q_count)
                                {
                                    throw std::invalid_argument(
                                        "GpuEvaluator::eval_mod_high_precision: lazy combine equal q_count has different parms_id");
                                }
                                combine_prefix_parms_id =
                                    q_ptrs[0]->meta.q_count < b_ptrs[0]->meta.q_count
                                        ? q_ptrs[0]->meta.parms_id
                                        : b_ptrs[0]->meta.parms_id;
                                use_combine_prefix_multiply =
                                    supports_q_prefix_multiply(
                                        *q_ptrs[0], *b_ptrs[0], combine_prefix_parms_id) &&
                                    supports_q_prefix_multiply(
                                        *q_ptrs[1], *b_ptrs[1], combine_prefix_parms_id);
                                if (!use_combine_prefix_multiply &&
                                    q_ptrs[0]->meta.q_count > b_ptrs[0]->meta.q_count)
                                {
                                    for (std::size_t lane = 0; lane < 2; ++lane)
                                    {
                                        if (q_mut[lane] == nullptr ||
                                            !drop_modulus_inplace(*q_mut[lane], b_ptrs[lane]->meta.parms_id))
                                        {
                                            drop_modulus(*q_ptrs[lane], *lanes[lane].scratch[3], b_ptrs[lane]->meta.parms_id);
                                            q_ptrs[lane] = lanes[lane].scratch[3];
                                            q_mut[lane] = lanes[lane].scratch[3];
                                        }
                                    }
                                }
                                else if (!use_combine_prefix_multiply)
                                {
                                    for (std::size_t lane = 0; lane < 2; ++lane)
                                    {
                                        if (b_mut[lane] == nullptr ||
                                            !drop_modulus_inplace(*b_mut[lane], q_ptrs[lane]->meta.parms_id))
                                        {
                                            drop_modulus(*b_ptrs[lane], *lanes[lane].scratch[4], q_ptrs[lane]->meta.parms_id);
                                            b_ptrs[lane] = lanes[lane].scratch[4];
                                            b_mut[lane] = lanes[lane].scratch[4];
                                        }
                                    }
                                }
                            }

                            is_square_0 =
                                !use_combine_prefix_multiply &&
                                q_ptrs[0] == b_ptrs[0] &&
                                q_ptrs[1] == b_ptrs[1];
                            multiply_q_count_0 =
                                use_combine_prefix_multiply
                                    ? params_.get_level(combine_prefix_parms_id).q_count
                                    : q_ptrs[0]->meta.q_count;
                            if (use_combine_prefix_multiply)
                            {
                                multiply_q_prefix_pair(
                                    *q_ptrs[0], *b_ptrs[0], *q_ptrs[1], *b_ptrs[1],
                                    combine_prefix_parms_id,
                                    *lanes[0].scratch[5], *lanes[1].scratch[5]);
                            }
                            else if (is_square_0)
                            {
                                square_pair(
                                    *q_ptrs[0], *q_ptrs[1],
                                    *lanes[0].scratch[5], *lanes[1].scratch[5]);
                            }
                            else
                            {
                                multiply_pair(
                                    *q_ptrs[0], *b_ptrs[0], *q_ptrs[1], *b_ptrs[1],
                                    *lanes[0].scratch[5], *lanes[1].scratch[5]);
                            }
                            lanes[0].scratch[5]->meta.scale = combine.product_scale;
                            lanes[1].scratch[5]->meta.scale = combine.product_scale;

                            std::array<const GpuCiphertextData *, 2> p_ptrs{
                                lanes[0].scratch[5], lanes[1].scratch[5]};
                            std::array<GpuCiphertextData *, 2> p_mut{
                                lanes[0].scratch[5], lanes[1].scratch[5]};
                            if (!combine.product_scale_plaintext.empty())
                            {
                                multiply_plain_pair(
                                    *p_ptrs[0], *p_ptrs[1],
                                    combine.product_scale_plaintext,
                                    comb_prod_scaled[0], comb_prod_scaled[1]);
                                comb_prod_scaled[0].meta.scale = combine.product_aligned_scale;
                                comb_prod_scaled[1].meta.scale = combine.product_aligned_scale;
                                p_ptrs = {&comb_prod_scaled[0], &comb_prod_scaled[1]};
                                p_mut = {&comb_prod_scaled[0], &comb_prod_scaled[1]};
                            }

                            std::array<const GpuCiphertextData *, 2> r_ptrs{
                                &(*lanes[0].nodes)[combine.remainder_node],
                                &(*lanes[1].nodes)[combine.remainder_node]};
                            std::array<GpuCiphertextData *, 2> r_mut{
                                zero_copy_moddrop &&
                                        combine.remainder_node < node_use_counts.size() &&
                                        node_use_counts[combine.remainder_node] == 1
                                    ? &(*lanes[0].nodes)[combine.remainder_node]
                                    : nullptr,
                                zero_copy_moddrop &&
                                        combine.remainder_node < node_use_counts.size() &&
                                        node_use_counts[combine.remainder_node] == 1
                                    ? &(*lanes[1].nodes)[combine.remainder_node]
                                    : nullptr};
                            if (combine.remainder_rescale_count > 0)
                            {
                                rescale_many_pair(
                                    *r_ptrs[0], *r_ptrs[1],
                                    comb_rem_rescaled[0], comb_rem_rescaled[1],
                                    combine.remainder_rescale_count);
                                if (combine.remainder_scale_plaintext.empty() &&
                                    combine.remainder_aligned_scale > 0.0)
                                {
                                    comb_rem_rescaled[0].meta.scale = combine.remainder_aligned_scale;
                                    comb_rem_rescaled[1].meta.scale = combine.remainder_aligned_scale;
                                }
                                r_ptrs = {&comb_rem_rescaled[0], &comb_rem_rescaled[1]};
                                r_mut = {&comb_rem_rescaled[0], &comb_rem_rescaled[1]};
                            }
                            if (!combine.remainder_scale_plaintext.empty())
                            {
                                multiply_plain_pair(
                                    *r_ptrs[0], *r_ptrs[1],
                                    combine.remainder_scale_plaintext,
                                    comb_rem_scaled[0], comb_rem_scaled[1]);
                                comb_rem_scaled[0].meta.scale = combine.remainder_aligned_scale;
                                comb_rem_scaled[1].meta.scale = combine.remainder_aligned_scale;
                                r_ptrs = {&comb_rem_scaled[0], &comb_rem_scaled[1]};
                                r_mut = {&comb_rem_scaled[0], &comb_rem_scaled[1]};
                            }

                            if (!(p_ptrs[0]->meta.parms_id == r_ptrs[0]->meta.parms_id))
                            {
                                if (p_ptrs[0]->meta.q_count == r_ptrs[0]->meta.q_count)
                                {
                                    throw std::invalid_argument(
                                        "GpuEvaluator::eval_mod_high_precision: lazy remainder equal q_count has different parms_id");
                                }
                                if (p_ptrs[0]->meta.q_count > r_ptrs[0]->meta.q_count)
                                {
                                    for (std::size_t lane = 0; lane < 2; ++lane)
                                    {
                                        if (p_mut[lane] == nullptr ||
                                            !drop_modulus_inplace(*p_mut[lane], r_ptrs[lane]->meta.parms_id))
                                        {
                                            drop_modulus(*p_ptrs[lane], comb_prod_dropped[lane], r_ptrs[lane]->meta.parms_id);
                                            comb_prod_dropped[lane].meta.scale = p_ptrs[lane]->meta.scale;
                                            p_ptrs[lane] = &comb_prod_dropped[lane];
                                            p_mut[lane] = &comb_prod_dropped[lane];
                                        }
                                    }
                                }
                                else
                                {
                                    for (std::size_t lane = 0; lane < 2; ++lane)
                                    {
                                        if (r_mut[lane] == nullptr ||
                                            !drop_modulus_inplace(*r_mut[lane], p_ptrs[lane]->meta.parms_id))
                                        {
                                            drop_modulus(*r_ptrs[lane], comb_rem_dropped[lane], p_ptrs[lane]->meta.parms_id);
                                            comb_rem_dropped[lane].meta.scale = r_ptrs[lane]->meta.scale;
                                            r_ptrs[lane] = &comb_rem_dropped[lane];
                                            r_mut[lane] = &comb_rem_dropped[lane];
                                        }
                                    }
                                }
                            }

                            add_pair(
                                *p_ptrs[0], *r_ptrs[0],
                                *p_ptrs[1], *r_ptrs[1],
                                *combined_raw_ptrs[0], *combined_raw_ptrs[1]);
                            combined_raw_ptrs[0]->meta.scale = combine.output_scale;
                            combined_raw_ptrs[1]->meta.scale = combine.output_scale;
                            if (combined_raw_ptrs[0]->size() != 3 ||
                                combined_raw_ptrs[1]->size() != 3)
                            {
                                throw std::invalid_argument(
                                    "GpuEvaluator::eval_mod_high_precision: lazy combine did not produce size-3 output");
                            }
                        }
                        else
                        {
                            for (std::size_t lane = 0; lane < lane_count; ++lane)
                            {
                                auto &L = lanes[lane];
                                const GpuCiphertextData *quotient =
                                    &(*L.nodes)[combine.quotient_node];
                                GpuCiphertextData *quotient_mutable =
                                    zero_copy_moddrop &&
                                            combine.quotient_node < node_use_counts.size() &&
                                            node_use_counts[combine.quotient_node] == 1
                                        ? &(*L.nodes)[combine.quotient_node]
                                        : nullptr;
                                if (quotient->size() != 2)
                                {
                                    throw std::invalid_argument(
                                        "GpuEvaluator::eval_mod_high_precision: lazy quotient must be relinearized");
                                }
                                if (combine.quotient_rescale_count > 0)
                                {
                                    rescale_many(
                                        *quotient,
                                        *L.scratch[2],
                                        combine.quotient_rescale_count);
                                    L.scratch[2]->meta.scale =
                                        combine.quotient_output_scale;
                                    quotient = L.scratch[2];
                                    quotient_mutable = L.scratch[2];
                                }

                                const GpuCiphertextData *basis =
                                    &(*L.basis)[combine.basis_degree];
                                GpuCiphertextData *basis_mutable =
                                    zero_copy_moddrop &&
                                            combine.basis_degree < basis_last_combine_use.size() &&
                                            basis_last_combine_use[combine.basis_degree] == combine_index
                                        ? &(*L.basis)[combine.basis_degree]
                                        : nullptr;
                                bool use_combine_prefix_multiply = false;
                                parms_id_type combine_prefix_parms_id = parms_id_zero;
                                if (!(quotient->meta.parms_id == basis->meta.parms_id))
                                {
                                    if (quotient->meta.q_count == basis->meta.q_count)
                                    {
                                        throw std::invalid_argument(
                                            "GpuEvaluator::eval_mod_high_precision: lazy combine equal q_count has different parms_id");
                                    }
                                    combine_prefix_parms_id =
                                        quotient->meta.q_count < basis->meta.q_count
                                            ? quotient->meta.parms_id
                                            : basis->meta.parms_id;
                                    use_combine_prefix_multiply =
                                        supports_q_prefix_multiply(
                                            *quotient,
                                            *basis,
                                            combine_prefix_parms_id);
                                    if (!use_combine_prefix_multiply &&
                                        quotient->meta.q_count > basis->meta.q_count)
                                    {
                                        if (quotient_mutable == nullptr ||
                                            !drop_modulus_inplace(
                                                *quotient_mutable,
                                                basis->meta.parms_id))
                                        {
                                            drop_modulus(
                                                *quotient,
                                                *L.scratch[3],
                                                basis->meta.parms_id);
                                            quotient = L.scratch[3];
                                            quotient_mutable = L.scratch[3];
                                        }
                                    }
                                    else if (!use_combine_prefix_multiply)
                                    {
                                        if (basis_mutable == nullptr ||
                                            !drop_modulus_inplace(
                                                *basis_mutable,
                                                quotient->meta.parms_id))
                                        {
                                            drop_modulus(
                                                *basis,
                                                *L.scratch[4],
                                                quotient->meta.parms_id);
                                            basis = L.scratch[4];
                                            basis_mutable = L.scratch[4];
                                        }
                                    }
                                }

                                const bool is_square =
                                    !use_combine_prefix_multiply && quotient == basis;
                                const std::size_t multiply_q_count =
                                    use_combine_prefix_multiply
                                        ? params_.get_level(combine_prefix_parms_id).q_count
                                        : quotient->meta.q_count;
                                if (lane == 0)
                                {
                                    multiply_q_count_0 = multiply_q_count;
                                    is_square_0 = is_square;
                                }
                                if (use_combine_prefix_multiply)
                                {
                                    multiply_q_prefix(
                                        *quotient,
                                        *basis,
                                        combine_prefix_parms_id,
                                        *L.scratch[5]);
                                }
                                else if (is_square)
                                {
                                    square(*quotient, *L.scratch[5]);
                                }
                                else
                                {
                                    multiply(*quotient, *basis, *L.scratch[5]);
                                }
                                L.scratch[5]->meta.scale = combine.product_scale;

                                const GpuCiphertextData *product = L.scratch[5];
                                GpuCiphertextData *product_mutable = L.scratch[5];
                                if (!combine.product_scale_plaintext.empty())
                                {
                                    multiply_plain(
                                        *product,
                                        combine.product_scale_plaintext,
                                        comb_prod_scaled[lane]);
                                    comb_prod_scaled[lane].meta.scale =
                                        combine.product_aligned_scale;
                                    product = &comb_prod_scaled[lane];
                                    product_mutable = &comb_prod_scaled[lane];
                                }

                                const GpuCiphertextData *remainder =
                                    &(*L.nodes)[combine.remainder_node];
                                GpuCiphertextData *remainder_mutable =
                                    zero_copy_moddrop &&
                                            combine.remainder_node < node_use_counts.size() &&
                                            node_use_counts[combine.remainder_node] == 1
                                        ? &(*L.nodes)[combine.remainder_node]
                                        : nullptr;

                                if (combine.remainder_rescale_count > 0)
                                {
                                    rescale_many(
                                        *remainder,
                                        comb_rem_rescaled[lane],
                                        combine.remainder_rescale_count);
                                    if (combine.remainder_scale_plaintext.empty() &&
                                        combine.remainder_aligned_scale > 0.0)
                                    {
                                        comb_rem_rescaled[lane].meta.scale =
                                            combine.remainder_aligned_scale;
                                    }
                                    remainder = &comb_rem_rescaled[lane];
                                    remainder_mutable = &comb_rem_rescaled[lane];
                                }
                                if (!combine.remainder_scale_plaintext.empty())
                                {
                                    multiply_plain(
                                        *remainder,
                                        combine.remainder_scale_plaintext,
                                        comb_rem_scaled[lane]);
                                    comb_rem_scaled[lane].meta.scale =
                                        combine.remainder_aligned_scale;
                                    remainder = &comb_rem_scaled[lane];
                                    remainder_mutable = &comb_rem_scaled[lane];
                                }

                                if (!(product->meta.parms_id == remainder->meta.parms_id))
                                {
                                    if (product->meta.q_count == remainder->meta.q_count)
                                    {
                                        throw std::invalid_argument(
                                            "GpuEvaluator::eval_mod_high_precision: lazy remainder equal q_count has different parms_id");
                                    }
                                    if (product->meta.q_count > remainder->meta.q_count)
                                    {
                                        if (product_mutable == nullptr ||
                                            !drop_modulus_inplace(
                                                *product_mutable,
                                                remainder->meta.parms_id))
                                        {
                                            drop_modulus(
                                                *product,
                                                comb_prod_dropped[lane],
                                                remainder->meta.parms_id);
                                            comb_prod_dropped[lane].meta.scale = product->meta.scale;
                                            product = &comb_prod_dropped[lane];
                                            product_mutable = &comb_prod_dropped[lane];
                                        }
                                    }
                                    else
                                    {
                                        if (remainder_mutable == nullptr ||
                                            !drop_modulus_inplace(
                                                *remainder_mutable,
                                                product->meta.parms_id))
                                        {
                                            drop_modulus(
                                                *remainder,
                                                comb_rem_dropped[lane],
                                                product->meta.parms_id);
                                            comb_rem_dropped[lane].meta.scale = remainder->meta.scale;
                                            remainder = &comb_rem_dropped[lane];
                                            remainder_mutable = &comb_rem_dropped[lane];
                                        }
                                    }
                                }

                                add(*product, *remainder, *combined_raw_ptrs[lane]);
                                combined_raw_ptrs[lane]->meta.scale = combine.output_scale;
                                if (combined_raw_ptrs[lane]->size() != 3)
                                {
                                    throw std::invalid_argument(
                                        "GpuEvaluator::eval_mod_high_precision: lazy combine did not produce size-3 output");
                                }
                            }
                        }

                        if (workspace.capture_eval_mod_stage_timing)
                        {
                            multiply_recorder.begin(
                                "EvalMod.mul.combine.lazy." +
                                    std::to_string(combine_index) +
                                    ".T" + std::to_string(combine.basis_degree) +
                                    ".q" + std::to_string(multiply_q_count_0),
                                multiply_q_count_0,
                                relin_keys.meta.p_count,
                                is_square_0);
                        }
                        if (defer_output)
                        {
                        }
                        else if (lane_count == 2)
                        {
                            relinearize_pair(
                                *combined_raw_ptrs[0],
                                *combined_raw_ptrs[1],
                                relin_keys,
                                (*lanes[0].nodes)[combine.output_node],
                                (*lanes[1].nodes)[combine.output_node]);
                            for (std::size_t lane = 0; lane < lane_count; ++lane)
                            {
                                (*lanes[lane].nodes)[combine.output_node].meta.scale =
                                    combine.output_scale;
                            }
                        }
                        else
                        {
                            relinearize(
                                *combined_raw_ptrs[0],
                                relin_keys,
                                (*lanes[0].nodes)[combine.output_node]);
                            (*lanes[0].nodes)[combine.output_node].meta.scale =
                                combine.output_scale;
                        }
                        multiply_recorder.end();
                        node_available[combine.output_node] = true;
                        if (workspace.capture_eval_mod_trace)
                        {
                            const GpuCiphertextData *trace_source =
                                &(*lanes[0].nodes)[combine.output_node];
                            GpuCiphertextData trace_relinearized;
                            if (trace_source->size() == 3)
                            {
                                relinearize(
                                    *trace_source,
                                    relin_keys,
                                    trace_relinearized);
                                trace_relinearized.meta.scale =
                                    trace_source->meta.scale;
                                trace_source = &trace_relinearized;
                            }
                            workspace.eval_mod_trace_polynomial_combine_outputs.emplace_back();
                            copy_ciphertext_data(
                                *trace_source,
                                workspace.eval_mod_trace_polynomial_combine_outputs.back(),
                                "GpuEvaluator::eval_mod_high_precision trace lazy combine copy");
                            workspace.eval_mod_trace_polynomial_combine_outputs.back().meta.scale =
                                trace_source->meta.scale;
                        }
                        continue;
                    }
                    if (!node_available[combine.quotient_node])
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: dynamic combine quotient is unavailable");
                    }
                    if (combine.output_q_count == 0 ||
                        combine.product_q_count == 0 ||
                        !(combine.product_scale > 0.0) ||
                        !std::isfinite(combine.product_scale))
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: incomplete dynamic combine plan");
                    }

                    std::size_t q_count_0 = 0;
                    bool is_square_0 = false;
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        auto &L = lanes[lane];
                        const GpuCiphertextData *quotient =
                            &(*L.nodes)[combine.quotient_node];
                        if (combine.quotient_rescale_count > 0)
                        {
                            rescale_many(
                                *quotient,
                                *L.scratch[2],
                                combine.quotient_rescale_count);
                            L.scratch[2]->meta.scale =
                                combine.quotient_output_scale;
                            quotient = L.scratch[2];
                        }

                        const GpuCiphertextData *basis =
                            &(*L.basis)[combine.basis_degree];
                        if (!(quotient->meta.parms_id == basis->meta.parms_id))
                        {
                            if (quotient->meta.q_count == basis->meta.q_count)
                            {
                                throw std::invalid_argument(
                                    "GpuEvaluator::eval_mod_high_precision: dynamic combine equal q_count has different parms_id");
                            }
                            if (quotient->meta.q_count > basis->meta.q_count)
                            {
                                drop_modulus(*quotient, *L.scratch[3], basis->meta.parms_id);
                                quotient = L.scratch[3];
                            }
                            else
                            {
                                drop_modulus(*basis, *L.scratch[4], quotient->meta.parms_id);
                                basis = L.scratch[4];
                            }
                        }

                        const bool is_square = quotient == basis;
                        if (lane == 0)
                        {
                            q_count_0 = quotient->meta.q_count;
                            is_square_0 = is_square;
                        }
                        if (is_square)
                        {
                            square(*quotient, *L.scratch[5]);
                        }
                        else
                        {
                            multiply(*quotient, *basis, *L.scratch[5]);
                        }
                    }
                    if (workspace.capture_eval_mod_stage_timing)
                    {
                        multiply_recorder.begin(
                            "EvalMod.mul.combine." +
                                std::to_string(combine_index) +
                                ".T" + std::to_string(combine.basis_degree) +
                                ".q" + std::to_string(q_count_0),
                            q_count_0,
                            relin_keys.meta.p_count,
                            is_square_0);
                    }
                    relinearize_lanes(5, 1);
                    multiply_recorder.end();

                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        auto &L = lanes[lane];
                        L.scratch[1]->meta.scale = combine.product_scale;

                        GpuCiphertextData product_scaled;
                        const GpuCiphertextData *product_base = L.scratch[1];
                        if (!combine.product_scale_plaintext.empty())
                        {
                            multiply_plain(
                                *product_base,
                                combine.product_scale_plaintext,
                                product_scaled);
                            product_scaled.meta.scale =
                                combine.product_aligned_scale;
                            product_base = &product_scaled;
                        }

                        if (node_available[combine.remainder_node])
                        {
                            const GpuCiphertextData *product = product_base;
                            const GpuCiphertextData *remainder =
                                &(*L.nodes)[combine.remainder_node];
                            GpuCiphertextData remainder_rescaled;
                            GpuCiphertextData remainder_scaled;
                            GpuCiphertextData product_dropped;
                            GpuCiphertextData remainder_dropped;

                            if (combine.remainder_rescale_count > 0)
                            {
                                rescale_many(
                                    *remainder,
                                    remainder_rescaled,
                                    combine.remainder_rescale_count);
                                if (combine.remainder_scale_plaintext.empty() &&
                                    combine.remainder_aligned_scale > 0.0)
                                {
                                    remainder_rescaled.meta.scale =
                                        combine.remainder_aligned_scale;
                                }
                                remainder = &remainder_rescaled;
                            }
                            if (!combine.remainder_scale_plaintext.empty())
                            {
                                multiply_plain(
                                    *remainder,
                                    combine.remainder_scale_plaintext,
                                    remainder_scaled);
                                remainder_scaled.meta.scale =
                                    combine.remainder_aligned_scale;
                                remainder = &remainder_scaled;
                            }

                            if (!(product->meta.parms_id == remainder->meta.parms_id))
                            {
                                if (product->meta.q_count == remainder->meta.q_count)
                                {
                                    throw std::invalid_argument(
                                        "GpuEvaluator::eval_mod_high_precision: dynamic remainder equal q_count has different parms_id");
                                }
                                if (product->meta.q_count > remainder->meta.q_count)
                                {
                                    drop_modulus(*product, product_dropped, remainder->meta.parms_id);
                                    product_dropped.meta.scale = product->meta.scale;
                                    product = &product_dropped;
                                }
                                else
                                {
                                    drop_modulus(*remainder, remainder_dropped, product->meta.parms_id);
                                    remainder_dropped.meta.scale = remainder->meta.scale;
                                    remainder = &remainder_dropped;
                                }
                            }

                            add(*product, *remainder, *L.scratch[0]);
                            L.scratch[0]->meta.scale = combine.output_scale;
                            (*L.nodes)[combine.output_node] =
                                std::move(*L.scratch[0]);
                        }
                        else
                        {
                            if (product_base == &product_scaled)
                            {
                                (*L.nodes)[combine.output_node] =
                                    std::move(product_scaled);
                            }
                            else
                            {
                                (*L.nodes)[combine.output_node] =
                                    std::move(*L.scratch[1]);
                            }
                            (*L.nodes)[combine.output_node].meta.scale =
                                combine.output_scale;
                        }

                        if ((*L.nodes)[combine.output_node].meta.q_count >
                            combine.output_q_count)
                        {
                            const auto &target_level =
                                params_.get_level_by_q_count(combine.output_q_count, 0);
                            drop_modulus(
                                (*L.nodes)[combine.output_node],
                                *L.scratch[2],
                                target_level.parms_id);
                            (*L.nodes)[combine.output_node] =
                                std::move(*L.scratch[2]);
                            (*L.nodes)[combine.output_node].meta.scale =
                                combine.output_scale;
                        }
                    }
                }
                else
                {
                    if (workspace.capture_eval_mod_stage_timing)
                    {
                        const auto &quotient =
                            (*lanes[0].nodes)[combine.quotient_node];
                        const auto &basis =
                            (*lanes[0].basis)[combine.basis_degree];
                        multiply_recorder.begin(
                            "EvalMod.mul.combine." +
                                std::to_string(combine_index) +
                                ".T" + std::to_string(combine.basis_degree) +
                                ".q" + std::to_string(
                                    std::min(quotient.meta.q_count, basis.meta.q_count)),
                            std::min(quotient.meta.q_count, basis.meta.q_count),
                            relin_keys.meta.p_count,
                            &quotient == &basis);
                    }
                    std::array<const GpuCiphertextData *, 2> q_ptrs{};
                    std::array<const GpuCiphertextData *, 2> b_ptrs{};
                    std::array<GpuCiphertextData *, 2> out_ptrs{};
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        q_ptrs[lane] = &(*lanes[lane].nodes)[combine.quotient_node];
                        b_ptrs[lane] = &(*lanes[lane].basis)[combine.basis_degree];
                        out_ptrs[lane] = lanes[lane].scratch[2];
                    }
                    multiply_relinearize_rescale_lanes(
                        q_ptrs, b_ptrs, combine.output_scale, logical_rescale_count, out_ptrs);
                    multiply_recorder.end();
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        add_aligned(
                            lanes[lane],
                            *lanes[lane].scratch[2],
                            (*lanes[lane].nodes)[combine.remainder_node],
                            combine.output_scale,
                            (*lanes[lane].nodes)[combine.output_node]);
                    }
                }
                node_available[combine.output_node] = true;
                if (workspace.capture_eval_mod_trace)
                {
                    workspace.eval_mod_trace_polynomial_combine_outputs.emplace_back();
                    copy_ciphertext_data(
                        (*lanes[0].nodes)[combine.output_node],
                        workspace.eval_mod_trace_polynomial_combine_outputs.back(),
                        "GpuEvaluator::eval_mod_high_precision trace combine copy");
                    workspace.eval_mod_trace_polynomial_combine_outputs.back().meta.scale =
                        (*lanes[0].nodes)[combine.output_node].meta.scale;
                }
            }

            const auto result_node =
                bootstrap_data.eval_mod.polynomial_result_node;
            if (!node_available[result_node])
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: polynomial result node is unavailable");
            }
            for (std::size_t lane = 0; lane < lane_count; ++lane)
            {
                auto &L = lanes[lane];
                if (workspace.capture_eval_mod_trace)
                {
                    copy_ciphertext_data(
                        (*L.nodes)[result_node],
                        L.accumulator,
                        "GpuEvaluator::eval_mod_high_precision trace accumulator copy");
                    L.accumulator.meta.scale = (*L.nodes)[result_node].meta.scale;
                }
                else
                {
                    L.accumulator = std::move((*L.nodes)[result_node]);
                }
            }
            if (bootstrap_data.eval_mod.dynamic_rescale)
            {
                const auto final_rescale_count =
                    bootstrap_data.eval_mod.polynomial_final_rescale_count;
                if (final_rescale_count > 0)
                {
                    if (lane_count == 2 &&
                        lanes[0].accumulator.meta.parms_id == lanes[1].accumulator.meta.parms_id)
                    {
                        if (d2d_free_dataflow)
                        {
                            rescale_many_pair(
                                lanes[0].accumulator, lanes[1].accumulator,
                                lanes[0].accumulator, lanes[1].accumulator,
                                final_rescale_count);
                        }
                        else
                        {
                            rescale_many_pair(
                                lanes[0].accumulator, lanes[1].accumulator,
                                *lanes[0].scratch[2], *lanes[1].scratch[2],
                                final_rescale_count);
                            lanes[0].accumulator = std::move(*lanes[0].scratch[2]);
                            lanes[1].accumulator = std::move(*lanes[1].scratch[2]);
                        }
                    }
                    else
                    {
                        for (std::size_t lane = 0; lane < lane_count; ++lane)
                        {
                            auto &L = lanes[lane];
                            if (d2d_free_dataflow)
                            {
                                rescale_many(L.accumulator, L.accumulator, final_rescale_count);
                            }
                            else
                            {
                                rescale_many(L.accumulator, *L.scratch[2], final_rescale_count);
                                L.accumulator = std::move(*L.scratch[2]);
                            }
                        }
                    }
                }
                if (bootstrap_data.eval_mod.polynomial_output_scale > 0.0)
                {
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        lanes[lane].accumulator.meta.scale =
                            bootstrap_data.eval_mod.polynomial_output_scale;
                    }
                }
            }
        }
    }
    stage_recorder.record(4);

    if (workspace.capture_eval_mod_trace)
    {
        copy_ciphertext_data(
            lanes[0].accumulator,
            workspace.eval_mod_trace_polynomial_output,
            "GpuEvaluator::eval_mod_high_precision trace polynomial copy");
        workspace.eval_mod_trace_polynomial_output.meta.scale =
            lanes[0].accumulator.meta.scale;
    }

    {
        NvtxRange range("EvalMod double-angle iterations");
        for (std::size_t double_angle_index = 0;
             double_angle_index < configured_double_angle_constants.size();
             ++double_angle_index)
        {
            const auto &double_angle_plaintext =
                configured_double_angle_constants[double_angle_index];
            const std::uint32_t double_angle_rescale_count =
                double_angle_index < configured_double_angle_rescale_counts.size()
                    ? std::max(
                          configured_double_angle_rescale_counts[double_angle_index],
                          std::uint32_t{1})
                    : logical_rescale_count;
            if (workspace.capture_eval_mod_stage_timing)
            {
                multiply_recorder.begin(
                    "EvalMod.mul.double_angle." +
                        std::to_string(double_angle_index) +
                        ".q" + std::to_string(lanes[0].accumulator.meta.q_count),
                    lanes[0].accumulator.meta.q_count,
                    relin_keys.meta.p_count,
                    true);
            }
            if (lane_count == 2 &&
                lanes[0].accumulator.meta.parms_id == lanes[1].accumulator.meta.parms_id)
            {
                square_pair(
                    lanes[0].accumulator, lanes[1].accumulator,
                    *lanes[0].scratch[5], *lanes[1].scratch[5]);
            }
            else
            {
                for (std::size_t lane = 0; lane < lane_count; ++lane)
                {
                    square(lanes[lane].accumulator, *lanes[lane].scratch[5]);
                }
            }
            if (workspace.capture_eval_mod_trace)
            {
                workspace.eval_mod_trace_double_angle_square_outputs.emplace_back();
                copy_ciphertext_data(
                    *lanes[0].scratch[5],
                    workspace.eval_mod_trace_double_angle_square_outputs.back(),
                    "GpuEvaluator::eval_mod_high_precision trace square copy");
                workspace.eval_mod_trace_double_angle_square_outputs.back().meta.scale =
                    lanes[0].scratch[5]->meta.scale;
            }
            const bool use_fused_da_x2 =
                double_angle_rescale_count == 2 &&
                !workspace.capture_eval_mod_trace &&
                use_rescale_x2() &&
                use_relinearize_rescale_x2();
            if (use_fused_da_x2)
            {
                if (lane_count == 2 &&
                    lanes[0].scratch[5]->meta.parms_id == lanes[1].scratch[5]->meta.parms_id)
                {
                    add_pair(
                        *lanes[0].scratch[5], *lanes[0].scratch[5],
                        *lanes[1].scratch[5], *lanes[1].scratch[5],
                        *lanes[0].scratch[3], *lanes[1].scratch[3]);
                }
                else
                {
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        add(*lanes[lane].scratch[5], *lanes[lane].scratch[5], *lanes[lane].scratch[3]);
                    }
                }
                if (!double_angle_plaintext.empty())
                {
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        auto &L = lanes[lane];
                        if (!(L.scratch[3]->meta.parms_id ==
                              double_angle_plaintext.meta.parms_id))
                        {
                            throw std::invalid_argument(
                                "GpuEvaluator::eval_mod_high_precision: double-angle plaintext level mismatch");
                        }
                        add_plain_inplace(
                            *L.scratch[3],
                            double_angle_plaintext,
                            "GpuEvaluator::eval_mod_high_precision double-angle constant");
                    }
                }
                if (lane_count == 2)
                {
                    relinearize_rescale_x2_hybrid_pair(
                        *lanes[0].scratch[3],
                        *lanes[1].scratch[3],
                        relin_keys,
                        *lanes[0].scratch[5],
                        *lanes[1].scratch[5]);
                }
                else
                {
                    relinearize_rescale_x2_hybrid(
                        *lanes[0].scratch[3],
                        relin_keys,
                        *lanes[0].scratch[5]);
                }
                multiply_recorder.end();
                for (std::size_t lane = 0; lane < lane_count; ++lane)
                {
                    lanes[lane].accumulator = std::move(*lanes[lane].scratch[5]);
                }
            }
            else if (bootstrap_data.eval_mod.dynamic_rescale)
            {
                relinearize_lanes(5, 2);
                multiply_recorder.end();
                if (workspace.capture_eval_mod_trace)
                {
                    workspace.eval_mod_trace_double_angle_relin_outputs.emplace_back();
                    copy_ciphertext_data(
                        *lanes[0].scratch[2],
                        workspace.eval_mod_trace_double_angle_relin_outputs.back(),
                        "GpuEvaluator::eval_mod_high_precision trace relin copy");
                    workspace.eval_mod_trace_double_angle_relin_outputs.back().meta.scale =
                        lanes[0].scratch[2]->meta.scale;
                }
                if (lane_count == 2 &&
                    lanes[0].scratch[2]->meta.parms_id == lanes[1].scratch[2]->meta.parms_id)
                {
                    if (!double_angle_plaintext.empty())
                    {
                        if (!(lanes[0].scratch[2]->meta.parms_id ==
                              double_angle_plaintext.meta.parms_id))
                        {
                            throw std::invalid_argument(
                                "GpuEvaluator::eval_mod_high_precision: dynamic "
                                "double-angle plaintext level mismatch at step " +
                                std::to_string(double_angle_index) +
                                " (cipher q=" +
                                std::to_string(lanes[0].scratch[2]->meta.q_count) +
                                ", plaintext q=" +
                                std::to_string(double_angle_plaintext.meta.q_count) +
                                ")");
                        }
                        double_sub_or_add_plain_pair(
                            kernel::DoublePlainPairOp::AddPlainC0,
                            *lanes[0].scratch[2],
                            lanes[1].scratch[2],
                            double_angle_plaintext,
                            nullptr,
                            nullptr,
                            *lanes[0].scratch[3],
                            lanes[1].scratch[3],
                            "GpuEvaluator::eval_mod_high_precision dynamic double-angle constant");
                    }
                    else
                    {
                        add_pair(
                            *lanes[0].scratch[2], *lanes[0].scratch[2],
                            *lanes[1].scratch[2], *lanes[1].scratch[2],
                            *lanes[0].scratch[3], *lanes[1].scratch[3]);
                    }
                    if (double_angle_rescale_count > 0)
                    {
                        rescale_many_pair(
                            *lanes[0].scratch[3], *lanes[1].scratch[3],
                            *lanes[0].scratch[5], *lanes[1].scratch[5],
                            double_angle_rescale_count);
                        lanes[0].accumulator = std::move(*lanes[0].scratch[5]);
                        lanes[1].accumulator = std::move(*lanes[1].scratch[5]);
                    }
                    else if (d2d_free_dataflow)
                    {
                        lanes[0].accumulator = std::move(*lanes[0].scratch[3]);
                        lanes[1].accumulator = std::move(*lanes[1].scratch[3]);
                    }
                    else
                    {
                        copy_ciphertext_data(
                            *lanes[0].scratch[3],
                            lanes[0].accumulator,
                            "GpuEvaluator::eval_mod_high_precision double-angle no-rescale copy");
                        copy_ciphertext_data(
                            *lanes[1].scratch[3],
                            lanes[1].accumulator,
                            "GpuEvaluator::eval_mod_high_precision double-angle no-rescale copy");
                    }
                }
                else
                {
                    for (std::size_t lane = 0; lane < lane_count; ++lane)
                    {
                        auto &L = lanes[lane];
                        add(*L.scratch[2], *L.scratch[2], *L.scratch[3]);
                        const GpuCiphertextData *pre_rescale_double_angle =
                            L.scratch[3];
                        if (!double_angle_plaintext.empty())
                        {
                            if (!(L.scratch[3]->meta.parms_id ==
                                  double_angle_plaintext.meta.parms_id))
                            {
                                throw std::invalid_argument(
                                    "GpuEvaluator::eval_mod_high_precision: dynamic "
                                    "double-angle plaintext level mismatch at step " +
                                    std::to_string(double_angle_index) +
                                    " (cipher q=" +
                                    std::to_string(L.scratch[3]->meta.q_count) +
                                    ", plaintext q=" +
                                    std::to_string(double_angle_plaintext.meta.q_count) +
                                    ")");
                            }
                            add_plain_inplace(
                                *L.scratch[3],
                                double_angle_plaintext,
                                "GpuEvaluator::eval_mod_high_precision dynamic double-angle constant");
                        }
                        if (double_angle_rescale_count > 0)
                        {
                            rescale_many(
                                *pre_rescale_double_angle,
                                *L.scratch[5],
                                double_angle_rescale_count);
                            L.accumulator = std::move(*L.scratch[5]);
                        }
                        else
                        {
                            if (d2d_free_dataflow)
                            {
                                L.accumulator = std::move(*L.scratch[3]);
                            }
                            else
                            {
                                copy_ciphertext_data(
                                    *pre_rescale_double_angle,
                                    L.accumulator,
                                    "GpuEvaluator::eval_mod_high_precision double-angle no-rescale copy");
                            }
                        }
                    }
                }
                if (workspace.capture_eval_mod_trace)
                {
                    workspace.eval_mod_trace_double_angle_rescaled_square_outputs.emplace_back();
                    copy_ciphertext_data(
                        lanes[0].accumulator,
                        workspace.eval_mod_trace_double_angle_rescaled_square_outputs.back(),
                        "GpuEvaluator::eval_mod_high_precision trace rescaled double-angle copy");
                    workspace.eval_mod_trace_double_angle_rescaled_square_outputs.back().meta.scale =
                        lanes[0].accumulator.meta.scale;
                }
            }
            else
            {
                relinearize_lanes(5, 2);
                multiply_recorder.end();
                for (std::size_t lane = 0; lane < lane_count; ++lane)
                {
                    auto &L = lanes[lane];
                    add(*L.scratch[2], *L.scratch[2], *L.scratch[3]);
                    if (!double_angle_plaintext.empty())
                    {
                        if (!(L.scratch[3]->meta.parms_id ==
                              double_angle_plaintext.meta.parms_id))
                        {
                            throw std::invalid_argument(
                                "GpuEvaluator::eval_mod_high_precision: double-angle plaintext level mismatch");
                        }
                        add_plain_inplace(
                            *L.scratch[3],
                            double_angle_plaintext,
                            "GpuEvaluator::eval_mod_high_precision fallback double-angle constant");
                    }
                    rescale_many(
                        *L.scratch[3],
                        *L.scratch[5],
                        double_angle_rescale_count);
                }
                for (std::size_t lane = 0; lane < lane_count; ++lane)
                {
                    lanes[lane].accumulator = std::move(*lanes[lane].scratch[5]);
                }
            }
            if (workspace.capture_eval_mod_trace)
            {
                workspace.eval_mod_trace_double_angle_outputs.emplace_back();
                copy_ciphertext_data(
                    lanes[0].accumulator,
                    workspace.eval_mod_trace_double_angle_outputs.back(),
                    "GpuEvaluator::eval_mod_high_precision trace double-angle copy");
                workspace.eval_mod_trace_double_angle_outputs.back().meta.scale =
                    lanes[0].accumulator.meta.scale;
            }
        }
    }
    stage_recorder.record(5);

    {
        NvtxRange range("EvalMod output level alignment");
        for (std::size_t lane = 0; lane < lane_count; ++lane)
        {
            auto &L = lanes[lane];
            if (bootstrap_data.eval_mod.output_parms_id != parms_id_zero &&
                L.accumulator.meta.parms_id != bootstrap_data.eval_mod.output_parms_id)
            {
                if (L.accumulator.meta.q_count < bootstrap_data.eval_mod.output_q_count)
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: GPU polynomial schedule ended below the configured CPU output level");
                }
                if (!q_prefix_views ||
                    !drop_modulus_inplace(
                        L.accumulator,
                        bootstrap_data.eval_mod.output_parms_id))
                {
                    drop_modulus(
                        L.accumulator,
                        *L.scratch[2],
                        bootstrap_data.eval_mod.output_parms_id);
                    L.accumulator = std::move(*L.scratch[2]);
                }
            }
            if (bootstrap_data.eval_mod.output_scale > 0.0)
            {
                L.accumulator.meta.scale = bootstrap_data.eval_mod.output_scale;
            }

            *destinations[lane] = std::move(L.accumulator);
        }
    }
    stage_recorder.finish(workspace.eval_mod_stage_timing);
    multiply_recorder.finish(workspace.eval_mod_multiply_timings);
}

}  // namespace gpu
}  // namespace poseidon
