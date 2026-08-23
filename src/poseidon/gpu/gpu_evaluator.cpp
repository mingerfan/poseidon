#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_scale_planner.h"
#include "poseidon/gpu/kernels/gpu_double_hoist_kernels.h"
#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"
#include "poseidon/gpu/kernels/gpu_keyswitch_kernels.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"

#include "poseidon/advance/homomorphic_linear_transform.h"

#include <nvtx3/nvToolsExt.h>

#include <stdexcept>
#include <algorithm>
#include <array>
#include <cmath>
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
            cudaMemset(
                shard.ptr,
                0,
                word_count * sizeof(GpuWord)),
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
            cudaMemcpy(
                dst.ptr,
                src.ptr,
                word_count * sizeof(GpuWord),
                cudaMemcpyDeviceToDevice),
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

void GpuEvaluator::add_plain(
    const GpuCiphertextData &source_ciphertext,
    const GpuPlaintextData &source_plaintext,
    GpuCiphertextData &destination_ciphertext) const
{
    if (source_ciphertext.empty() || source_plaintext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: empty input");
    }

    if (!(source_ciphertext.meta.parms_id == source_plaintext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: parms_id mismatch");
    }

    if (source_ciphertext.meta.is_ntt_form != source_plaintext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: NTT form mismatch");
    }

    // CKKS add_plain usually expects both ciphertext and plaintext in NTT form.
    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: CKKS input must be in NTT form");
    }

    if (source_ciphertext.meta.degree != source_plaintext.meta.degree ||
        source_ciphertext.meta.q_count != source_plaintext.meta.q_count ||
        source_ciphertext.meta.p_count != source_plaintext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: shape mismatch");
    }

    if (!same_scale(source_ciphertext.meta.scale, source_plaintext.meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: scale mismatch");
    }

    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::add_plain: p limbs are not supported by add_plain kernel yet");
    }

    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: component metadata mismatch");
    }

    const std::size_t result_components = source_ciphertext.size();

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: ciphertext shard layout mismatch");
    }

    if (!same_logical_shard_layout(reference_layout, source_plaintext.poly_))
    {
        throw std::invalid_argument("GpuEvaluator::add_plain: plaintext shard layout mismatch");
    }

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
    if (source_ciphertext.empty() || source_plaintext.empty())
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: empty input");
    }

    if (!(source_ciphertext.meta.parms_id == source_plaintext.meta.parms_id))
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: parms_id mismatch");
    }

    if (source_ciphertext.meta.is_ntt_form != source_plaintext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: NTT form mismatch");
    }

    if (!source_ciphertext.meta.is_ntt_form)
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: CKKS input must be in NTT form");
    }

    if (source_ciphertext.meta.degree != source_plaintext.meta.degree ||
        source_ciphertext.meta.q_count != source_plaintext.meta.q_count ||
        source_ciphertext.meta.p_count != source_plaintext.meta.p_count)
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: shape mismatch");
    }

    if (!same_scale(source_ciphertext.meta.scale, source_plaintext.meta.scale))
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: scale mismatch");
    }

    if (source_ciphertext.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::sub_plain: p limbs are not supported by sub_plain kernel yet");
    }

    if (source_ciphertext.meta.component_count != source_ciphertext.size())
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: component metadata mismatch");
    }

    const std::size_t result_components = source_ciphertext.size();

    const int device_id = source_ciphertext.fields_.at(0).device_id;
    const auto &reference_layout = source_ciphertext.polys_.at(0);

    if (!all_components_use_layout(source_ciphertext, reference_layout))
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: ciphertext shard layout mismatch");
    }

    if (!same_logical_shard_layout(reference_layout, source_plaintext.poly_))
    {
        throw std::invalid_argument("GpuEvaluator::sub_plain: plaintext shard layout mismatch");
    }

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
    if (scalar > std::numeric_limits<GpuWord>::max())
    {
        throw std::invalid_argument("GpuEvaluator::multiply_scalar: scalar exceeds GPU word size");
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

    GpuCiphertextMeta result_meta = source_ciphertext.meta;
    prepare_ciphertext_destination(
        destination_ciphertext,
        &source_ciphertext,
        nullptr,
        result_meta,
        source_ciphertext.size(),
        device_id,
        reference_layout);

    auto source_view = source_ciphertext.make_const_view();
    auto destination_view = destination_ciphertext.make_view();
    elementwise_handler_.multiply_scalar_ciphertext(
        destination_view,
        source_view,
        static_cast<GpuWord>(scalar),
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

    const GpuCiphertextData *current = &source_ciphertext;
    GpuCiphertextData scratch0;
    GpuCiphertextData scratch1;

    auto next_scratch = [&]() -> GpuCiphertextData &
    {
        return (current == &scratch0) ? scratch1 : scratch0;
    };

    while (current->meta.scale > std::pow(2.0, 54.0))
    {
        const auto &level_info = params_.get_level(current->meta.parms_id);
        if (level_info.shards.empty() || level_info.shards.front().q_last == 0)
        {
            throw std::invalid_argument(
                "GpuEvaluator::bootstrap_prepare_modraise_input: missing q_last parameter");
        }
        const double q_last = static_cast<double>(level_info.shards.front().q_last);
        if (current->meta.scale / q_last <= 1.6e+07)
        {
            throw std::invalid_argument(
                "GpuEvaluator::bootstrap_prepare_modraise_input: scale cannot be safely rescaled for bootstrap");
        }

        GpuCiphertextData &next = next_scratch();
        rescale(*current, next);
        current = &next;
    }

    if (current->meta.q_count < q0_level_info.q_count)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_prepare_modraise_input: source is below q0 level");
    }

    const std::size_t q0_plus_one_count = q0_level_info.q_count + 1;
    if (current->meta.q_count > q0_plus_one_count)
    {
        const auto &q0_plus_one_level =
            params_.get_level_by_q_count(q0_plus_one_count, 0);
        GpuCiphertextData &next = next_scratch();
        drop_modulus(*current, next, q0_plus_one_level.parms_id);
        current = &next;
    }

    const double scale_multiplier_double =
        std::round(q0_over_message_ratio / current->meta.scale);
    if (!std::isfinite(scale_multiplier_double))
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_prepare_modraise_input: invalid scale multiplier");
    }
    if (scale_multiplier_double > 1.0)
    {
        if (scale_multiplier_double >
            static_cast<double>(std::numeric_limits<GpuWord>::max()))
        {
            throw std::invalid_argument(
                "GpuEvaluator::bootstrap_prepare_modraise_input: scale multiplier exceeds GPU word size");
        }
        const auto scale_multiplier =
            static_cast<std::uint64_t>(scale_multiplier_double);
        GpuCiphertextData &next = next_scratch();
        multiply_scalar(*current, scale_multiplier, next);
        next.meta.scale = current->meta.scale * scale_multiplier_double;
        current = &next;
    }

    drop_modulus(*current, destination_ciphertext, q0_level_info.parms_id);
}

/* GpuEvaluator::multiply(...) -> 输出 3 component: d0, d1, d2 */
void GpuEvaluator::relinearize(
    const GpuCiphertextData &source_ciphertext,
    const GpuRelinKeysData &relin_keys,
    GpuCiphertextData &destination_ciphertext) const
{
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
    auto relin_keys_view =
        relin_keys.make_const_view(source_ciphertext.meta.q_count);
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

void GpuEvaluator::relinearize_rescale_x2_hybrid(
    const GpuCiphertextData &source_ciphertext,
    const GpuRelinKeysData &relin_keys,
    GpuCiphertextData &destination_ciphertext) const
{
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
    auto relin_keys_view =
        relin_keys.make_const_view(source_ciphertext.meta.q_count);
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

/*顶层旋转操作入口*/
void GpuEvaluator::rotate(
    const GpuCiphertextData &source_ciphertext,
    int step,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &destination_ciphertext) const
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
        GpuDoubleHoistWorkspace staged;
        staged.outer_accumulator.ensure_capacity(
            device_id,
            source_ciphertext.meta.degree,
            source_ciphertext.meta.q_count,
            galois_keys.meta.p_count,
            1);
        keyswitch_handler_.hoist_decompose_modup_ntt(
            source_view.polys[1],
            level_info,
            staged.source_hoist,
            staged.keyswitch);
        const auto key_view =
            galois_keys.make_const_view(source_ciphertext.meta.q_count);
        keyswitch_handler_.keyswitch_multsum_no_moddown(
            staged.source_hoist,
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
    auto galois_keys_view =
        galois_keys.make_const_view(source_ciphertext.meta.q_count);
    keyswitch_handler_.switch_key_hybrid_ciphertext(
        destination_view,
        rotated_c1_const_view.polys[0],
        galois_keys_view,
        galois_keys,
        key_index,
        level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::conjugate(
    const GpuCiphertextData &source_ciphertext,
    const GpuGaloisKeysData &galois_keys,
    GpuCiphertextData &destination_ciphertext) const
{
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
        GpuDoubleHoistWorkspace staged;
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
    auto galois_keys_view =
        galois_keys.make_const_view(source_ciphertext.meta.q_count);
    keyswitch_handler_.switch_key_hybrid_ciphertext(
        destination_view,
        conjugated_c1_const_view.polys[0],
        galois_keys_view,
        galois_keys,
        key_index,
        level_info);

    destination_ciphertext = std::move(result);
}

void GpuEvaluator::conjugate_pre_rotated(
    const GpuCiphertextData &source_ciphertext,
    const GpuGaloisKeysData &galois_keys,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
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
    const auto key_view =
        galois_keys.make_const_view(source_ciphertext.meta.q_count);
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
    const GpuGaloisKeysData &galois_keys,
    std::uint32_t rescale_count,
    GpuCiphertextData &destination_ciphertext) const
{
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

void GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist(
    const GpuCiphertextData &source_ciphertext,
    const GpuMatrixPlainQP &matrix,
    const GpuGaloisKeysData &galois_keys,
    std::uint32_t rescale_count,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext,
    GpuQPCiphertextBuffer *destination_partial_qp,
    GpuCiphertextMeta *destination_partial_meta) const
{
    const bool produce_partial_qp = destination_partial_qp != nullptr;
    if (produce_partial_qp != (destination_partial_meta != nullptr) ||
        (produce_partial_qp && rescale_count != 0))
    {
        throw std::invalid_argument(
            "GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist: "
            "invalid partial-QP output request");
    }
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
    if (has_full_plaintexts == has_compressed_plaintexts ||
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

    workspace.last_counts = {};
    workspace.last_counts.workspace_peak_bytes = estimated_workspace;
    workspace.group_accumulators.ensure_capacity(
        source_shard0.device_id,
        degree,
        q_count,
        p_count,
        group_count);
    const bool direct_qp_mac_init =
        use_double_hoist_qp_mac_direct_init();
    if (!direct_qp_mac_init)
    {
        workspace.group_accumulators.q.fill_zero();
        workspace.group_accumulators.p.fill_zero();
    }

    keyswitch_handler_.hoist_decompose_modup_ntt(
        source_view.polys[1],
        level_info,
        workspace.source_hoist,
        workspace.keyswitch);
    workspace.last_counts.source_decompose_count = 1;

    const auto galois_keys_view =
        galois_keys.make_const_view(q_count);
    const bool fused_baby_keyswitch_c0 =
        use_double_hoist_fused_baby_keyswitch_c0() &&
        dnum == 1 &&
        galois_keys.meta.galois_format ==
            GpuGaloisKeyFormat::InversePreRotated;
    bool fused_baby_keyswitch_plain_mac =
        use_double_hoist_fused_baby_keyswitch_plain_mac() &&
        degree == 65536 &&
        requested_tile <= kernel::kMaxDoubleHoistFusedBabySteps &&
        group_count <= 4 &&
        galois_keys.meta.galois_format ==
            GpuGaloisKeyFormat::InversePreRotated;
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
    for (std::size_t tile_begin = 0;
         tile_begin < baby_count;
         tile_begin += requested_tile)
    {
        const std::size_t tile_count =
            std::min(requested_tile, baby_count - tile_begin);
        ++workspace.last_counts.baby_tile_count;
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
                4 * kernel::kMaxDoubleHoistFusedBabySteps> term_indices{};
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
        if (produce_partial_qp)
        {
            *destination_partial_qp =
                std::move(workspace.group_accumulators);
            *destination_partial_meta = product_meta;
            return;
        }
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
    if (grouped_outer_moddown)
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
        digit_q_ptrs.reserve(group_count);
        digit_p_ptrs.reserve(group_count);
        group_indices.reserve(group_count);
        galois_elts.reserve(group_count);
        key_indices.reserve(group_count);
        if (workspace.outer_group_hoists.size() < group_count)
        {
            workspace.outer_group_hoists.resize(group_count);
        }

        for (std::size_t group = 0; group < group_count; ++group)
        {
            const int giant_step = matrix.plan.giant_steps[group];
            if (giant_step == 0)
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
                continue;
            }

            GpuConstRNSPolyView inner_c1;
            inner_c1.poly_id = 1;
            inner_c1.shards.push_back(GpuConstPolyShardView{
                source_shard0.device_id,
                workspace.inner_q_batch.q_component(group, 1),
                0,
                q_count,
                0,
                degree});
            const std::size_t non_identity_group = group - 1;
            auto &group_hoist = workspace.outer_group_hoists[group];
            const GpuWord *precomputed_source_intt_q =
                batched_giant_intt
                    ? workspace.outer_source_intt_q_batch.data() +
                          non_identity_group * q_words
                    : nullptr;
            keyswitch_handler_.hoist_decompose_modup_ntt(
                inner_c1,
                level_info,
                group_hoist,
                workspace.keyswitch,
                precomputed_source_intt_q);
            const GpuWord *digit_q_ptr =
                group_hoist.digits_q_ntt.data();
            const GpuWord *digit_p_ptr =
                group_hoist.digits_p_ntt.data();
            ++workspace.last_counts.outer_decompose_count;

            const std::uint32_t galois_elt =
                galois_elt_from_rotation_step(degree, giant_step);
            const std::size_t key_index =
                galois_key_index(galois_elt);
            if (key_index > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::overflow_error(
                    "double-hoist giant key index exceeds uint32_t");
            }
            digit_q_ptrs.push_back(digit_q_ptr);
            digit_p_ptrs.push_back(digit_p_ptr);
            group_indices.push_back(static_cast<std::uint32_t>(group));
            galois_elts.push_back(galois_elt);
            key_indices.push_back(static_cast<std::uint32_t>(key_index));
        }

        const std::size_t active_group_count = group_indices.size();
        if (active_group_count != 0)
        {
            if (direct_giant_accumulate)
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

    if (produce_partial_qp)
    {
        if (!grouped_outer_moddown)
        {
            throw std::invalid_argument(
                "GpuEvaluator::multiply_by_diag_matrix_bsgs_double_hoist: "
                "partial QP requires inverse-pre-rotated Galois keys");
        }
        if (!direct_giant_accumulate)
        {
            workspace.outer_accumulator.ensure_capacity(
                source_shard0.device_id,
                degree,
                q_count,
                p_count,
                1);
            kernel::launch_double_hoist_reduce_qp_groups(
                workspace.outer_accumulator.q_component(0, 0),
                workspace.outer_accumulator.q_component(0, 1),
                workspace.outer_accumulator.p_component(0, 0),
                workspace.outer_accumulator.p_component(0, 1),
                workspace.group_accumulators.q.data(),
                workspace.group_accumulators.p.data(),
                group_count,
                *parameter_shard,
                degree);
        }
        *destination_partial_qp = std::move(workspace.outer_accumulator);
        *destination_partial_meta = product_meta;
        return;
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

void GpuEvaluator::dft_double_hoist_layer(
    const GpuCiphertextData &source_ciphertext,
    const GpuLinearMatrixGroupQP &matrix_group,
    std::size_t layer_index,
    const GpuGaloisKeysData &galois_keys,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    if (layer_index >= matrix_group.data().size())
    {
        throw std::out_of_range(
            "GpuEvaluator::dft_double_hoist_layer: layer index is out of range");
    }
    const bool dynamic_rescale = matrix_group.rescale_min_scale() > 0.0;
    if (dynamic_rescale &&
        matrix_group.rescale_counts().size() != matrix_group.data().size())
    {
        throw std::invalid_argument(
            "GpuEvaluator::dft_double_hoist_layer: dynamic rescale plan size mismatch");
    }

    const auto &matrix = matrix_group.data()[layer_index];
    GpuCiphertextData product;
    multiply_by_diag_matrix_bsgs_double_hoist(
        source_ciphertext,
        matrix,
        galois_keys,
        dynamic_rescale ? 0 : std::max(matrix_group.step(), std::uint32_t{1}),
        workspace,
        dynamic_rescale ? product : destination_ciphertext);
    if (dynamic_rescale)
    {
        rescale_dynamic(
            product,
            destination_ciphertext,
            matrix_group.rescale_min_scale());
    }
}

void GpuEvaluator::dft_double_hoist_layer_partial_qp(
    const GpuCiphertextData &source_ciphertext,
    const GpuLinearMatrixGroupQP &matrix_group,
    std::size_t layer_index,
    const GpuGaloisKeysData &galois_keys,
    GpuDoubleHoistWorkspace &workspace,
    GpuQPCiphertextBuffer &destination_partial_qp,
    GpuCiphertextMeta &destination_partial_meta) const
{
    if (layer_index >= matrix_group.data().size())
    {
        throw std::out_of_range(
            "GpuEvaluator::dft_double_hoist_layer_partial_qp: layer index is out of range");
    }
    GpuCiphertextData unused;
    multiply_by_diag_matrix_bsgs_double_hoist(
        source_ciphertext,
        matrix_group.data()[layer_index],
        galois_keys,
        0,
        workspace,
        unused,
        &destination_partial_qp,
        &destination_partial_meta);
}

void GpuEvaluator::add_double_hoist_partial_qp_inplace(
    GpuQPCiphertextBuffer &destination,
    const GpuQPCiphertextBuffer &source,
    const GpuCiphertextMeta &meta) const
{
    if (destination.batch_count != 1 || source.batch_count != 1 ||
        destination.device_id != source.device_id ||
        destination.degree != source.degree ||
        destination.q_count != source.q_count ||
        destination.p_count != source.p_count ||
        destination.q_count != meta.q_count ||
        destination.degree != meta.degree || destination.p_count == 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::add_double_hoist_partial_qp_inplace: shape mismatch");
    }
    const auto &level_info = params_.get_level(meta.parms_id);
    const GpuParameterShard *parameter_shard = nullptr;
    for (const auto &candidate : level_info.shards)
    {
        if (candidate.device_id == destination.device_id &&
            candidate.hybrid_base_q_count == destination.q_count &&
            candidate.hybrid_base_p_count == destination.p_count)
        {
            parameter_shard = &candidate;
            break;
        }
    }
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "GpuEvaluator::add_double_hoist_partial_qp_inplace: parameter shard is absent");
    }
    kernel::launch_double_hoist_add_qp_inplace(
        destination.q.data(),
        destination.p.data(),
        source.q.data(),
        source.p.data(),
        *parameter_shard,
        destination.degree);
}

void GpuEvaluator::finalize_dft_double_hoist_layer_partial_qp(
    GpuQPCiphertextBuffer &source_partial_qp,
    const GpuCiphertextMeta &source_partial_meta,
    const GpuLinearMatrixGroupQP &matrix_group,
    std::size_t layer_index,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    if (layer_index >= matrix_group.data().size() ||
        source_partial_qp.batch_count != 1 ||
        source_partial_qp.degree != source_partial_meta.degree ||
        source_partial_qp.q_count != source_partial_meta.q_count ||
        source_partial_qp.p_count == 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::finalize_dft_double_hoist_layer_partial_qp: invalid input");
    }
    const auto &level_info = params_.get_level(source_partial_meta.parms_id);
    keyswitch_handler_.moddown_qp_ciphertext_to_q(
        source_partial_qp,
        0,
        workspace.result_q,
        source_partial_meta,
        level_info,
        workspace.keyswitch);

    const bool dynamic_rescale = matrix_group.rescale_min_scale() > 0.0;
    if (dynamic_rescale)
    {
        if (matrix_group.rescale_counts().size() !=
            matrix_group.data().size())
        {
            throw std::invalid_argument(
                "GpuEvaluator::finalize_dft_double_hoist_layer_partial_qp: "
                "dynamic rescale plan size mismatch");
        }
        rescale_dynamic(
            workspace.result_q,
            destination_ciphertext,
            matrix_group.rescale_min_scale());
    }
    else
    {
        rescale_many(
            workspace.result_q,
            destination_ciphertext,
            std::max(matrix_group.step(), std::uint32_t{1}));
    }
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

void GpuEvaluator::normalize_bootstrap_output_scale(
    GpuCiphertextData &output,
    const GpuBootstrapData &bootstrap_data,
    GpuBootstrapWorkspace &workspace) const
{
    if (!bootstrap_data.output_scale_normalization_plaintext.empty())
    {
        if (!(bootstrap_data.output_scale_override > 0.0) ||
            !std::isfinite(bootstrap_data.output_scale_override))
        {
            throw std::invalid_argument(
                "GpuEvaluator::bootstrap: invalid normalized output scale");
        }
        multiply_plain(
            output,
            bootstrap_data.output_scale_normalization_plaintext,
            workspace.scratch3);
        rescale(workspace.scratch3, workspace.scratch4);
        output = std::move(workspace.scratch4);
    }
    if (bootstrap_data.output_scale_override > 0.0)
    {
        output.meta.scale = bootstrap_data.output_scale_override;
    }
}

void GpuEvaluator::bootstrap_stc_first_prepare_c2s(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    const GpuGaloisKeysData &galois_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    if (bootstrap_data.schedule != GpuBootstrapSchedule::StCFirst)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_stc_first_prepare_c2s: profile is not StC-first");
    }
    if (bootstrap_data.project_real)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_stc_first_prepare_c2s: project_real is unsupported");
    }
    if (!(bootstrap_data.q0_over_message_ratio > 0.0) ||
        !std::isfinite(bootstrap_data.q0_over_message_ratio))
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_stc_first_prepare_c2s: invalid q0/message ratio");
    }

    const auto linear_transform_mode =
        bootstrap_data.allow_environment_linear_transform_override
            ? gpu_linear_transform_mode_from_environment(
                  bootstrap_data.linear_transform_mode)
            : bootstrap_data.linear_transform_mode;
    const bool use_double_hoist =
        linear_transform_mode == GpuLinearTransformMode::DoubleHoistBsgs;
    if (linear_transform_mode == GpuLinearTransformMode::SingleHoistBsgs)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_stc_first_prepare_c2s: single_hoist is unsupported");
    }
    if (use_double_hoist)
    {
        if (bootstrap_data.double_hoist_baby_tile == 0)
        {
            throw std::invalid_argument(
                "GpuEvaluator::bootstrap_stc_first_prepare_c2s: double-hoist baby tile is zero");
        }
        workspace.coeff_to_slot_double_hoist.baby_tile_size =
            bootstrap_data.double_hoist_baby_tile;
        workspace.slot_to_coeff_double_hoist.baby_tile_size =
            bootstrap_data.double_hoist_baby_tile;
    }
    if ((!use_double_hoist &&
         (bootstrap_data.coeff_to_slot_matrix.data().empty() ||
          bootstrap_data.slot_to_coeff_matrix.data().empty())) ||
        (use_double_hoist &&
         (bootstrap_data.coeff_to_slot_matrix_qp.data().empty() ||
          bootstrap_data.slot_to_coeff_matrix_qp.data().empty())))
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_stc_first_prepare_c2s: empty linear transform");
    }

    NvtxRange prefix_range("bootstrap_stc_first_prepare_c2s");
    {
        NvtxRange range("StC");
        if (use_double_hoist)
        {
            dft_double_hoist(
                source_ciphertext,
                bootstrap_data.slot_to_coeff_matrix_qp,
                galois_keys,
                workspace.slot_to_coeff_double_hoist,
                workspace.scratch0);
        }
        else
        {
            dft(
                source_ciphertext,
                bootstrap_data.slot_to_coeff_matrix,
                galois_keys,
                workspace.scratch0);
        }
    }

    GpuCiphertextData *raised_for_c2s = &workspace.raised;
    {
        NvtxRange range("ModRaise");
        bootstrap_prepare_modraise_input(
            workspace.scratch0,
            workspace.modraise_input,
            bootstrap_data.q0_parms_id,
            bootstrap_data.q0_over_message_ratio);
        raise_modulus(workspace.modraise_input, workspace.raised);

        if (bootstrap_data.post_raise_c2s_input_scale > 0.0)
        {
            workspace.raised.meta.scale =
                bootstrap_data.post_raise_c2s_input_scale;
        }
        else if (bootstrap_data.raised_scale_override > 0.0)
        {
            workspace.raised.meta.scale =
                bootstrap_data.raised_scale_override;
        }

        if (bootstrap_data.post_raise_integer_multiplier > 1)
        {
            multiply_scalar(
                workspace.raised,
                bootstrap_data.post_raise_integer_multiplier,
                workspace.raised_scaled);
            workspace.raised_scaled.meta.scale =
                workspace.raised.meta.scale *
                bootstrap_data.post_raise_scale_multiplier;
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

    destination_ciphertext = std::move(*raised_for_c2s);
}

void GpuEvaluator::bootstrap_stc_first_transform(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    const GpuGaloisKeysData &galois_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    GpuCiphertextData raised_for_c2s;
    bootstrap_stc_first_prepare_c2s(
        source_ciphertext,
        bootstrap_data,
        galois_keys,
        workspace,
        raised_for_c2s);

    const auto linear_transform_mode =
        bootstrap_data.allow_environment_linear_transform_override
            ? gpu_linear_transform_mode_from_environment(
                  bootstrap_data.linear_transform_mode)
            : bootstrap_data.linear_transform_mode;
    NvtxRange range("CtS raw DFT");
    if (linear_transform_mode == GpuLinearTransformMode::DoubleHoistBsgs)
    {
        dft_double_hoist(
            raised_for_c2s,
            bootstrap_data.coeff_to_slot_matrix_qp,
            galois_keys,
            workspace.coeff_to_slot_double_hoist,
            destination_ciphertext);
    }
    else
    {
        dft(
            raised_for_c2s,
            bootstrap_data.coeff_to_slot_matrix,
            galois_keys,
            destination_ciphertext);
    }
}

void GpuEvaluator::bootstrap_extract_real(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    const GpuGaloisKeysData &galois_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    if (bootstrap_data.schedule != GpuBootstrapSchedule::StCFirst)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_extract_real: profile is not StC-first");
    }
    GpuCiphertextData conjugated;
    if (galois_keys.meta.galois_format ==
        GpuGaloisKeyFormat::InversePreRotated)
    {
        conjugate_pre_rotated(
            source_ciphertext,
            galois_keys,
            workspace.coeff_to_slot_double_hoist,
            conjugated);
    }
    else
    {
        conjugate(source_ciphertext, galois_keys, conjugated);
    }
    add(source_ciphertext, conjugated, destination_ciphertext);
}

void GpuEvaluator::bootstrap_extract_imag(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    const GpuGaloisKeysData &galois_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    if (bootstrap_data.schedule != GpuBootstrapSchedule::StCFirst ||
        bootstrap_data.minus_i_plaintext.empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_extract_imag: invalid StC-first profile");
    }
    GpuCiphertextData conjugated;
    if (galois_keys.meta.galois_format ==
        GpuGaloisKeyFormat::InversePreRotated)
    {
        conjugate_pre_rotated(
            source_ciphertext,
            galois_keys,
            workspace.coeff_to_slot_double_hoist,
            conjugated);
    }
    else
    {
        conjugate(source_ciphertext, galois_keys, conjugated);
    }
    GpuCiphertextData difference;
    sub(source_ciphertext, conjugated, difference);
    multiply_plain(
        difference,
        bootstrap_data.minus_i_plaintext,
        destination_ciphertext);
}

void GpuEvaluator::bootstrap_stc_first_finalize(
    const GpuCiphertextData &source_real,
    const GpuCiphertextData &source_imag,
    const GpuBootstrapData &bootstrap_data,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    if (bootstrap_data.schedule != GpuBootstrapSchedule::StCFirst ||
        bootstrap_data.plus_i_plaintext.empty() ||
        bootstrap_data.output_ratio == 0)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap_stc_first_finalize: invalid StC-first profile");
    }

    multiply_plain(
        source_imag,
        bootstrap_data.plus_i_plaintext,
        workspace.scratch0);
    add(source_real, workspace.scratch0, workspace.scratch1);
    if (bootstrap_data.output_ratio > 1)
    {
        multiply_scalar(
            workspace.scratch1,
            bootstrap_data.output_ratio,
            workspace.scratch2);
        destination_ciphertext = std::move(workspace.scratch2);
    }
    else
    {
        destination_ciphertext = std::move(workspace.scratch1);
    }
    normalize_bootstrap_output_scale(
        destination_ciphertext, bootstrap_data, workspace);
}

void GpuEvaluator::bootstrap(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    const GpuRelinKeysData &relin_keys,
    const GpuGaloisKeysData &galois_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    const auto linear_transform_mode =
        bootstrap_data.allow_environment_linear_transform_override
            ? gpu_linear_transform_mode_from_environment(
                  bootstrap_data.linear_transform_mode)
            : bootstrap_data.linear_transform_mode;
    const bool use_double_hoist =
        linear_transform_mode == GpuLinearTransformMode::DoubleHoistBsgs;
    if (use_double_hoist)
    {
        if (bootstrap_data.double_hoist_baby_tile == 0)
        {
            throw std::invalid_argument(
                "GpuEvaluator::bootstrap: double-hoist baby tile is zero");
        }
        workspace.coeff_to_slot_double_hoist.baby_tile_size =
            bootstrap_data.double_hoist_baby_tile;
        workspace.slot_to_coeff_double_hoist.baby_tile_size =
            bootstrap_data.double_hoist_baby_tile;
    }
    if (linear_transform_mode ==
        GpuLinearTransformMode::SingleHoistBsgs)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap: single_hoist mode is reserved for "
            "staged validation; select classic or double_hoist");
    }
    if ((!use_double_hoist &&
         bootstrap_data.coeff_to_slot_matrix.data().empty()) ||
        (use_double_hoist &&
         bootstrap_data.coeff_to_slot_matrix_qp.data().empty()))
    {
        throw std::invalid_argument("GpuEvaluator::bootstrap: empty CoeffToSlot matrix group");
    }
    if ((!use_double_hoist &&
         bootstrap_data.slot_to_coeff_matrix.data().empty()) ||
        (use_double_hoist &&
         bootstrap_data.slot_to_coeff_matrix_qp.data().empty()))
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

    if (bootstrap_data.schedule == GpuBootstrapSchedule::StCFirst)
    {
        NvtxRange bootstrap_range("bootstrap_stc_first_once");
        GpuCiphertextData c2s_dft_result;
        bootstrap_stc_first_transform(
            source_ciphertext,
            bootstrap_data,
            galois_keys,
            workspace,
            c2s_dft_result);

        GpuCiphertextData conjugated;
        if (use_double_hoist &&
            galois_keys.meta.galois_format ==
                GpuGaloisKeyFormat::InversePreRotated)
        {
            conjugate_pre_rotated(
                c2s_dft_result,
                galois_keys,
                workspace.coeff_to_slot_double_hoist,
                conjugated);
        }
        else
        {
            conjugate(c2s_dft_result, galois_keys, conjugated);
        }
        add(
            c2s_dft_result,
            conjugated,
            workspace.coeff_to_slot_real);
        sub(c2s_dft_result, conjugated, workspace.scratch0);
        multiply_plain(
            workspace.scratch0,
            bootstrap_data.minus_i_plaintext,
            workspace.coeff_to_slot_imag);

        {
            NvtxRange range("EvalMod");
            eval_mod_high_precision(
                workspace.coeff_to_slot_real,
                bootstrap_data,
                relin_keys,
                workspace,
                workspace.eval_mod_real);
            eval_mod_high_precision(
                workspace.coeff_to_slot_imag,
                bootstrap_data,
                relin_keys,
                workspace,
                workspace.eval_mod_imag);
        }
        bootstrap_stc_first_finalize(
            workspace.eval_mod_real,
            workspace.eval_mod_imag,
            bootstrap_data,
            workspace,
            destination_ciphertext);
        return;
    }

    if (bootstrap_data.schedule != GpuBootstrapSchedule::Standard)
    {
        throw std::invalid_argument(
            "GpuEvaluator::bootstrap: unknown bootstrap schedule");
    }

    NvtxRange bootstrap_range("bootstrap_once");
    const GpuCiphertextData *raised_for_c2s = &workspace.raised;
    {
        NvtxRange range("ModRaise");
        bootstrap_prepare_modraise_input(
            source_ciphertext,
            workspace.modraise_input,
            bootstrap_data.q0_parms_id,
            bootstrap_data.q0_over_message_ratio);

        raise_modulus(workspace.modraise_input, workspace.raised);

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

    {
        NvtxRange range("CtS");
        if (use_double_hoist)
        {
            coeff_to_slot_double_hoist(
                *raised_for_c2s,
                bootstrap_data.coeff_to_slot_matrix_qp,
                bootstrap_data.minus_i_plaintext,
                galois_keys,
                workspace.coeff_to_slot_double_hoist,
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

    {
        NvtxRange range("EvalMod");
        {
            NvtxRange branch_range("EvalMod real");
            eval_mod_high_precision(
                workspace.coeff_to_slot_real,
                bootstrap_data,
                relin_keys,
                workspace,
                workspace.eval_mod_real);
        }
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

    if (bootstrap_data.slot_to_coeff_input_scale > 0.0)
    {
        workspace.eval_mod_real.meta.scale =
            bootstrap_data.slot_to_coeff_input_scale;
        workspace.eval_mod_imag.meta.scale =
            bootstrap_data.slot_to_coeff_input_scale;
    }

    {
        NvtxRange range("StC");
        if (use_double_hoist)
        {
            slot_to_coeff_double_hoist(
                workspace.eval_mod_real,
                workspace.eval_mod_imag,
                bootstrap_data.slot_to_coeff_matrix_qp,
                bootstrap_data.plus_i_plaintext,
                galois_keys,
                workspace.slot_to_coeff_double_hoist,
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
    normalize_bootstrap_output_scale(
        destination_ciphertext, bootstrap_data, workspace);
}

void GpuEvaluator::eval_mod_high_precision(
    const GpuCiphertextData &source_ciphertext,
    const GpuBootstrapData &bootstrap_data,
    const GpuRelinKeysData &relin_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext,
    const GpuEvalModPolynomialPartition *polynomial_partition) const
{
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
    const bool partitioned_polynomial = polynomial_partition != nullptr;
    const std::size_t combine_begin = partitioned_polynomial
        ? polynomial_partition->combine_begin
        : 0;
    const std::size_t combine_end = partitioned_polynomial
        ? std::min(
              polynomial_partition->combine_end,
              polynomial_combine_steps.size())
        : polynomial_combine_steps.size();
    const std::uint32_t polynomial_result_node = partitioned_polynomial
        ? polynomial_partition->result_node
        : bootstrap_data.eval_mod.polynomial_result_node;
    const std::size_t basis_step_end = partitioned_polynomial
        ? std::min(
              polynomial_partition->basis_step_end,
              basis_steps.size())
        : basis_steps.size();
    const std::size_t leaf_begin = partitioned_polynomial
        ? polynomial_partition->leaf_begin
        : 0;
    const std::size_t leaf_end = partitioned_polynomial
        ? std::min(
              polynomial_partition->leaf_end,
              polynomial_blocks.size())
        : polynomial_blocks.size();
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
    if (partitioned_polynomial &&
        (polynomial_blocks.empty() ||
         combine_begin > combine_end ||
         combine_end > polynomial_combine_steps.size() ||
         basis_step_end > basis_steps.size() ||
         leaf_begin > leaf_end ||
         leaf_end > polynomial_blocks.size() ||
         polynomial_result_node ==
             std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::invalid_argument(
            "GpuEvaluator::eval_mod_high_precision: invalid polynomial partition");
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
    const bool legacy_horner =
        polynomial_blocks.empty() && polynomial_terms.empty();
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

        destination_ciphertext = std::move(accumulator);
        destination_ciphertext.meta.scale = source_ciphertext.meta.scale;
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
    if (workspace.eval_mod_basis.size() < required_basis_slots)
    {
        workspace.eval_mod_basis.resize(required_basis_slots);
    }
    if (d2d_free_dataflow)
    {
        NvtxRange range("EvalMod input preparation");
        copy_ciphertext_data(
            source_ciphertext,
            workspace.eval_mod_basis[1],
            "GpuEvaluator::eval_mod_high_precision basis T1 input copy");
        workspace.eval_mod_basis[1].meta.scale = configured_input_scale;
        if (!configured_input_offset.empty())
        {
            add_plain_inplace(
                workspace.eval_mod_basis[1],
                configured_input_offset,
                "GpuEvaluator::eval_mod_high_precision basis T1 input offset");
        }
    }
    else
    {
        NvtxRange range("EvalMod input preparation");
        copy_ciphertext_data(
            source_ciphertext,
            workspace.scratch0,
            "GpuEvaluator::eval_mod_high_precision input copy");
        workspace.scratch0.meta.scale = configured_input_scale;
        GpuCiphertextData *prepared_input = &workspace.scratch0;
        if (!configured_input_offset.empty())
        {
            add_plain(
                workspace.scratch0,
                configured_input_offset,
                workspace.scratch1);
            workspace.scratch1.meta.scale = configured_input_scale;
            prepared_input = &workspace.scratch1;
        }
        copy_ciphertext_data(
            *prepared_input,
            workspace.eval_mod_basis[1],
            "GpuEvaluator::eval_mod_high_precision basis T1 copy");
        workspace.eval_mod_basis[1].meta.scale = configured_input_scale;
    }
    x = &workspace.eval_mod_basis[1];
    if (workspace.capture_eval_mod_trace)
    {
        copy_ciphertext_data(
            *x,
            workspace.eval_mod_trace_offset_input,
            "GpuEvaluator::eval_mod_high_precision trace input copy");
        workspace.eval_mod_trace_offset_input.meta.scale = x->meta.scale;
    }
    stage_recorder.record(1);

    auto multiply_relinearize_rescale =
        [&](const GpuCiphertextData &left,
            const GpuCiphertextData &right,
            double expected_output_scale,
            std::uint32_t rescale_count,
            GpuCiphertextData &output) {
            const GpuCiphertextData *left_at_level = &left;
            const GpuCiphertextData *right_at_level = &right;
            if (!(left.meta.parms_id == right.meta.parms_id))
            {
                if (left.meta.q_count == right.meta.q_count)
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: equal q_count has different parms_id");
                }
                if (left.meta.q_count > right.meta.q_count)
                {
                    drop_modulus(left, workspace.scratch3, right.meta.parms_id);
                    left_at_level = &workspace.scratch3;
                }
                else
                {
                    drop_modulus(right, workspace.scratch4, left.meta.parms_id);
                    right_at_level = &workspace.scratch4;
                }
            }

            if (left_at_level == right_at_level)
            {
                square(*left_at_level, workspace.scratch5);
            }
            else
            {
                multiply(*left_at_level, *right_at_level, workspace.scratch5);
            }
            if (rescale_count == 2 &&
                use_rescale_x2() &&
                use_relinearize_rescale_x2())
            {
                relinearize_rescale_x2_hybrid(
                    workspace.scratch5,
                    relin_keys,
                    output);
            }
            else
            {
                relinearize(workspace.scratch5, relin_keys, workspace.scratch3);
                // rescale_dynamic is deliberately not used in the first GPU path.
                rescale_many(
                    workspace.scratch3,
                    output,
                    rescale_count);
            }
            if (expected_output_scale > 0.0)
            {
                output.meta.scale = expected_output_scale;
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
        for (std::size_t basis_step_index = 0;
             basis_step_index < basis_step_end;
             ++basis_step_index)
        {
            const auto &step = basis_steps[basis_step_index];
            auto &output = workspace.eval_mod_basis[step.output_degree];
            const GpuCiphertextData *basis_left =
                &workspace.eval_mod_basis[step.left_degree];
            const GpuCiphertextData *basis_right =
                &workspace.eval_mod_basis[step.right_degree];
            if (!step.operand_alignment_plaintext.empty())
            {
                const GpuCiphertextData &high_operand = step.align_left_operand
                    ? *basis_left
                    : *basis_right;
                multiply_plain(
                    high_operand,
                    step.operand_alignment_plaintext,
                    workspace.scratch0);
                workspace.scratch0.meta.scale =
                    step.operand_alignment_pre_rescale_scale;
                rescale_many(
                    workspace.scratch0,
                    workspace.scratch1,
                    step.operand_alignment_rescale_count);
                workspace.scratch1.meta.scale =
                    step.operand_alignment_output_scale;
                if (step.align_left_operand)
                {
                    basis_left = &workspace.scratch1;
                }
                else
                {
                    basis_right = &workspace.scratch1;
                }
            }
            if (bootstrap_data.eval_mod.dynamic_rescale)
            {
                bool use_basis_prefix_multiply = false;
                parms_id_type basis_prefix_parms_id = parms_id_zero;
                if (!(basis_left->meta.parms_id == basis_right->meta.parms_id))
                {
                    if (basis_left->meta.q_count == basis_right->meta.q_count)
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: dynamic basis equal q_count has different parms_id");
                    }
                    basis_prefix_parms_id =
                        basis_left->meta.q_count < basis_right->meta.q_count
                            ? basis_left->meta.parms_id
                            : basis_right->meta.parms_id;
                    use_basis_prefix_multiply =
                        supports_q_prefix_multiply(
                            *basis_left,
                            *basis_right,
                            basis_prefix_parms_id);
                    if (!use_basis_prefix_multiply &&
                        basis_left->meta.q_count > basis_right->meta.q_count)
                    {
                        drop_modulus(
                            *basis_left,
                            workspace.scratch0,
                            basis_right->meta.parms_id);
                        basis_left = &workspace.scratch0;
                    }
                    else if (!use_basis_prefix_multiply)
                    {
                        drop_modulus(
                            *basis_right,
                            workspace.scratch1,
                            basis_left->meta.parms_id);
                        basis_right = &workspace.scratch1;
                    }
                }

                const bool is_square =
                    !use_basis_prefix_multiply && basis_left == basis_right;
                const std::size_t multiply_q_count =
                    use_basis_prefix_multiply
                        ? params_.get_level(basis_prefix_parms_id).q_count
                        : basis_left->meta.q_count;
                if (workspace.capture_eval_mod_stage_timing)
                {
                    multiply_recorder.begin(
                        "EvalMod.mul.basis.T" +
                        std::to_string(step.output_degree) +
                            "=" + std::to_string(step.left_degree) +
                            "x" + std::to_string(step.right_degree) +
                            ".q" + std::to_string(multiply_q_count),
                        multiply_q_count,
                        relin_keys.meta.p_count,
                        is_square);
                }
                if (use_basis_prefix_multiply)
                {
                    multiply_q_prefix(
                        *basis_left,
                        *basis_right,
                        basis_prefix_parms_id,
                        workspace.scratch5);
                }
                else if (is_square)
                {
                    square(*basis_left, workspace.scratch5);
                }
                else
                {
                    multiply(*basis_left, *basis_right, workspace.scratch5);
                }
                relinearize(workspace.scratch5, relin_keys, workspace.scratch3);
                multiply_recorder.end();
                output = std::move(workspace.scratch3);
                output.meta.scale = step.pre_rescale_scale;

                if (bootstrap_data.eval_mod.polynomial_basis ==
                    GpuEvalModPolynomialBasis::Chebyshev)
                {
                    add(output, output, workspace.scratch3);
                    output = std::move(workspace.scratch3);
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
                            workspace.eval_mod_basis[step.correction_degree];
                        multiply_plain(
                            correction_basis,
                            step.correction_plaintext,
                            workspace.scratch4);
                        workspace.scratch4.meta.scale = step.pre_rescale_scale;

                        const GpuCiphertextData *left = &output;
                        const GpuCiphertextData *right = &workspace.scratch4;
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
                                        workspace.scratch0,
                                        right->meta.parms_id);
                                    left = &workspace.scratch0;
                                }
                            }
                            else if (right->meta.q_count > left->meta.q_count)
                            {
                                if (!q_prefix_views ||
                                    !drop_modulus_inplace(
                                        workspace.scratch4,
                                        left->meta.parms_id))
                                {
                                    drop_modulus(
                                        *right,
                                        workspace.scratch1,
                                        left->meta.parms_id);
                                    right = &workspace.scratch1;
                                }
                            }
                            else
                            {
                                throw std::invalid_argument(
                                    "GpuEvaluator::eval_mod_high_precision: dynamic Chebyshev correction parms_id mismatch");
                            }
                        }
                        sub(*left, *right, workspace.scratch5);
                        output = std::move(workspace.scratch5);
                        output.meta.scale = step.pre_rescale_scale;
                    }
                }

                if (d2d_free_dataflow)
                {
                    rescale_dynamic(output, output, target_scale);
                }
                else
                {
                    rescale_dynamic(output, workspace.scratch2, target_scale);
                    output = std::move(workspace.scratch2);
                }
                output.meta.scale = step.output_scale;
                continue;
            }

            if (workspace.capture_eval_mod_stage_timing)
            {
                multiply_recorder.begin(
                    "EvalMod.mul.basis.T" +
                        std::to_string(step.output_degree) +
                        "=" + std::to_string(step.left_degree) +
                        "x" + std::to_string(step.right_degree) +
                        ".q" + std::to_string(basis_left->meta.q_count),
                    basis_left->meta.q_count,
                    relin_keys.meta.p_count,
                    basis_left == basis_right);
            }
            multiply_relinearize_rescale(
                *basis_left,
                *basis_right,
                step.output_scale,
                step.rescale_count,
                output);
            multiply_recorder.end();

            if (bootstrap_data.eval_mod.polynomial_basis !=
                GpuEvalModPolynomialBasis::Chebyshev)
            {
                continue;
            }

            add(output, output, workspace.scratch3);
            output = std::move(workspace.scratch3);
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
                &workspace.eval_mod_basis[step.correction_degree];
            if (!step.correction_alignment_plaintext.empty())
            {
                multiply_plain(
                    *correction,
                    step.correction_alignment_plaintext,
                    workspace.scratch0);
                workspace.scratch0.meta.scale =
                    step.correction_alignment_pre_rescale_scale;
                rescale_many(
                    workspace.scratch0,
                    workspace.scratch1,
                    step.correction_alignment_rescale_count);
                workspace.scratch1.meta.scale = step.output_scale;
                correction = &workspace.scratch1;
            }
            if (!(output.meta.parms_id == correction->meta.parms_id))
            {
                if (output.meta.q_count > correction->meta.q_count)
                {
                    drop_modulus(output, workspace.scratch4, correction->meta.parms_id);
                    output = std::move(workspace.scratch4);
                }
                else if (correction->meta.q_count > output.meta.q_count)
                {
                    drop_modulus(
                        *correction,
                        workspace.scratch4,
                        output.meta.parms_id);
                    correction = &workspace.scratch4;
                }
                else
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: Chebyshev correction parms_id mismatch");
                }
            }
            if (!step.correction_alignment_plaintext.empty())
            {
                sub(output, *correction, workspace.scratch5);
                output = std::move(workspace.scratch5);
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
                workspace.scratch3);
            workspace.scratch3.meta.scale = step.output_scale;
            sub(output, workspace.scratch3, workspace.scratch5);
            output = std::move(workspace.scratch5);
            output.meta.scale = step.output_scale;
        }
    }
    stage_recorder.record(2);

    std::function<bool(
        const std::vector<GpuEvalModPolynomialTerm> &,
        std::uint32_t,
        double,
        std::size_t,
        GpuCiphertextData &)> evaluate_term_block_caccum;

    auto evaluate_term_block =
        [&](const std::vector<GpuEvalModPolynomialTerm> &terms,
            std::uint32_t rescale_count,
            double expected_output_scale,
            std::size_t expected_output_q_count,
            GpuCiphertextData &block_output) {
            if (evaluate_term_block_caccum &&
                evaluate_term_block_caccum(
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
                        drop_modulus(*left, workspace.scratch3, right->meta.parms_id);
                        left = &workspace.scratch3;
                    }
                    else if (right->meta.q_count > left->meta.q_count)
                    {
                        drop_modulus(*right, workspace.scratch4, left->meta.parms_id);
                        right = &workspace.scratch4;
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: term parms_id mismatch");
                    }
                }
                add(*left, *right, workspace.scratch5);
                block_accumulator = std::move(workspace.scratch5);
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

                const auto &basis = workspace.eval_mod_basis[term.degree];
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
                        workspace.scratch3,
                        term.coefficient_plaintext.meta.parms_id);
                    basis_at_level = &workspace.scratch3;
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
                    workspace.scratch2);
                if (bootstrap_data.eval_mod.rescale_polynomial_terms_individually)
                {
                    rescale_many(
                        workspace.scratch2,
                        workspace.scratch4,
                        rescale_count);
                    workspace.scratch4.meta.scale = expected_output_scale;
                    workspace.scratch2 = std::move(workspace.scratch4);
                }
                accumulate_term(workspace.scratch2);
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
                        workspace.eval_mod_basis[1],
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
                        workspace.scratch3,
                        term.coefficient_plaintext.meta.parms_id);
                    block_accumulator = std::move(workspace.scratch3);
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
                    workspace.scratch2,
                    rescale_count);
                block_accumulator = std::move(workspace.scratch2);
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
                    workspace.scratch2,
                    target_level.parms_id);
                block_accumulator = std::move(workspace.scratch2);
                if (expected_output_scale > 0.0)
                {
                    block_accumulator.meta.scale = expected_output_scale;
                }
            }
            block_output = std::move(block_accumulator);
        };

    auto add_aligned =
        [&](const GpuCiphertextData &left_input,
            const GpuCiphertextData &right_input,
            double expected_output_scale,
            GpuCiphertextData &output) {
            const GpuCiphertextData *left = &left_input;
            const GpuCiphertextData *right = &right_input;
            if (!(left->meta.parms_id == right->meta.parms_id))
            {
                if (left->meta.q_count > right->meta.q_count)
                {
                    drop_modulus(*left, workspace.scratch3, right->meta.parms_id);
                    left = &workspace.scratch3;
                }
                else if (right->meta.q_count > left->meta.q_count)
                {
                    drop_modulus(*right, workspace.scratch4, left->meta.parms_id);
                    right = &workspace.scratch4;
                }
                else
                {
                    throw std::invalid_argument(
                        "GpuEvaluator::eval_mod_high_precision: combine parms_id mismatch");
                }
            }
            add(*left, *right, workspace.scratch5);
            if (expected_output_scale > 0.0)
            {
                workspace.scratch5.meta.scale = expected_output_scale;
            }
            output = std::move(workspace.scratch5);
        };

    evaluate_term_block_caccum =
        [&](const std::vector<GpuEvalModPolynomialTerm> &terms,
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
            if (first_nonconstant->degree >= workspace.eval_mod_basis.size() ||
                workspace.eval_mod_basis[first_nonconstant->degree].empty())
            {
                return false;
            }
            const double pre_rescale_scale =
                workspace.eval_mod_basis[first_nonconstant->degree].meta.scale *
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
                if (term.degree >= workspace.eval_mod_basis.size())
                {
                    return false;
                }
                const auto &basis = workspace.eval_mod_basis[term.degree];
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
                const auto &basis = workspace.eval_mod_basis[term.degree];
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
                    workspace.scratch2,
                    rescale_count);
                workspace.scratch2.meta.scale = expected_output_scale;
                block_output = std::move(workspace.scratch2);
            }
            return true;
        };

    auto accumulate_leaf_pre_rescale =
        [&](const GpuEvalModPolynomialBlock &block,
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
                if (term.degree >= workspace.eval_mod_basis.size())
                {
                    return false;
                }

                const auto &basis = workspace.eval_mod_basis[term.degree];
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
                const auto &basis = workspace.eval_mod_basis[term.degree];
                const GpuCiphertextData *basis_at_level = &basis;
                if (!(basis.meta.parms_id ==
                      term.coefficient_plaintext.meta.parms_id))
                {
                    drop_modulus(
                        basis,
                        workspace.scratch3,
                        term.coefficient_plaintext.meta.parms_id);
                    basis_at_level = &workspace.scratch3;
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

    auto multiply_inline_leaf_relinearize_rescale =
        [&](const GpuCiphertextData &left,
            const GpuCiphertextData &right,
            const GpuEvalModPolynomialBlock &leaf_block,
            double expected_output_scale,
            GpuCiphertextData &output) {
            if (!use_evalmod_inline_leaf())
            {
                return false;
            }

            const GpuCiphertextData *left_at_level = &left;
            const GpuCiphertextData *right_at_level = &right;
            if (!(left.meta.parms_id == right.meta.parms_id))
            {
                if (left.meta.q_count == right.meta.q_count)
                {
                    return false;
                }
                if (left.meta.q_count > right.meta.q_count)
                {
                    drop_modulus(left, workspace.scratch3, right.meta.parms_id);
                    left_at_level = &workspace.scratch3;
                }
                else
                {
                    drop_modulus(right, workspace.scratch4, left.meta.parms_id);
                    right_at_level = &workspace.scratch4;
                }
            }

            if (left_at_level == right_at_level)
            {
                square(*left_at_level, workspace.scratch5);
            }
            else
            {
                multiply(*left_at_level, *right_at_level, workspace.scratch5);
            }

            if (!accumulate_leaf_pre_rescale(
                    leaf_block,
                    workspace.scratch5))
            {
                return false;
            }

            if (logical_rescale_count == 2 &&
                use_rescale_x2() &&
                use_relinearize_rescale_x2())
            {
                relinearize_rescale_x2_hybrid(
                    workspace.scratch5,
                    relin_keys,
                    output);
            }
            else
            {
                relinearize(workspace.scratch5, relin_keys, workspace.scratch3);
                rescale_many(
                    workspace.scratch3,
                    output,
                    logical_rescale_count);
            }
            if (expected_output_scale > 0.0)
            {
                output.meta.scale = expected_output_scale;
            }
            return true;
        };

    GpuCiphertextData accumulator;
    if (polynomial_blocks.empty())
    {
        NvtxRange range("EvalMod fused leaf evaluation");
        evaluate_term_block(
            polynomial_terms,
            bootstrap_data.eval_mod.polynomial_rescale_count,
            target_scale,
            0,
            accumulator);
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
        if (workspace.eval_mod_nodes.size() <= maximum_node)
        {
            workspace.eval_mod_nodes.resize(
                static_cast<std::size_t>(maximum_node) + 1);
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
         * and degree 58/59 with their existing recursive splits.
         */
        const bool lazy_relinearization =
            use_evalmod_lazy_relinearization() &&
            bootstrap_data.eval_mod.dynamic_rescale &&
            (bootstrap_data.eval_mod.polynomial_degree == 22 ||
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
            for (std::size_t i = leaf_begin; i < leaf_end; ++i)
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
                evaluate_term_block(
                    polynomial_blocks[i].terms,
                    polynomial_blocks[i].rescale_count,
                    polynomial_blocks[i].output_scale,
                    polynomial_blocks[i].output_q_count,
                    workspace.eval_mod_nodes[i]);
                node_available[i] = true;
                if (workspace.capture_eval_mod_trace)
                {
                    workspace.eval_mod_trace_polynomial_leaf_outputs.emplace_back();
                    copy_ciphertext_data(
                        workspace.eval_mod_nodes[i],
                        workspace.eval_mod_trace_polynomial_leaf_outputs.back(),
                        "GpuEvaluator::eval_mod_high_precision trace leaf copy");
                    workspace.eval_mod_trace_polynomial_leaf_outputs.back().meta.scale =
                        workspace.eval_mod_nodes[i].meta.scale;
                }
            }
        }
        stage_recorder.record(3);

        {
            NvtxRange range("EvalMod BSGS tree combine");
            for (std::size_t combine_index = combine_begin;
                 combine_index < combine_end;
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
                    !node_available[combine.remainder_node] &&
                    multiply_inline_leaf_relinearize_rescale(
                        workspace.eval_mod_nodes[combine.quotient_node],
                        workspace.eval_mod_basis[combine.basis_degree],
                        polynomial_blocks[combine.remainder_node],
                        combine.output_scale,
                        workspace.eval_mod_nodes[combine.output_node]))
                {
                    node_available[combine.output_node] = true;
                    continue;
                }
                if (!node_available[combine.remainder_node])
                {
                    evaluate_term_block(
                        polynomial_blocks[combine.remainder_node].terms,
                        polynomial_blocks[combine.remainder_node].rescale_count,
                        polynomial_blocks[combine.remainder_node].output_scale,
                        polynomial_blocks[combine.remainder_node].output_q_count,
                        workspace.eval_mod_nodes[combine.remainder_node]);
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

                        const GpuCiphertextData *quotient =
                            &workspace.eval_mod_nodes[combine.quotient_node];
                        GpuCiphertextData *quotient_mutable =
                            zero_copy_moddrop &&
                                    combine.quotient_node < node_use_counts.size() &&
                                    node_use_counts[combine.quotient_node] == 1
                                ? &workspace.eval_mod_nodes[combine.quotient_node]
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
                                workspace.scratch2,
                                combine.quotient_rescale_count);
                            workspace.scratch2.meta.scale =
                                combine.quotient_output_scale;
                            quotient = &workspace.scratch2;
                            quotient_mutable = &workspace.scratch2;
                        }

                        const GpuCiphertextData *basis =
                            &workspace.eval_mod_basis[combine.basis_degree];
                        GpuCiphertextData *basis_mutable =
                            zero_copy_moddrop &&
                                    combine.basis_degree < basis_last_combine_use.size() &&
                                    basis_last_combine_use[combine.basis_degree] == combine_index
                                ? &workspace.eval_mod_basis[combine.basis_degree]
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
                                        workspace.scratch3,
                                        basis->meta.parms_id);
                                    quotient = &workspace.scratch3;
                                    quotient_mutable = &workspace.scratch3;
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
                                        workspace.scratch4,
                                        quotient->meta.parms_id);
                                    basis = &workspace.scratch4;
                                    basis_mutable = &workspace.scratch4;
                                }
                            }
                        }

                        const bool is_square =
                            !use_combine_prefix_multiply && quotient == basis;
                        const std::size_t multiply_q_count =
                            use_combine_prefix_multiply
                                ? params_.get_level(combine_prefix_parms_id).q_count
                                : quotient->meta.q_count;
                        if (workspace.capture_eval_mod_stage_timing)
                        {
                            multiply_recorder.begin(
                                "EvalMod.mul.combine.lazy." +
                                    std::to_string(combine_index) +
                                    ".T" + std::to_string(combine.basis_degree) +
                                    ".q" + std::to_string(multiply_q_count),
                                multiply_q_count,
                                relin_keys.meta.p_count,
                                is_square);
                        }
                        if (use_combine_prefix_multiply)
                        {
                            multiply_q_prefix(
                                *quotient,
                                *basis,
                                combine_prefix_parms_id,
                                workspace.scratch5);
                        }
                        else if (is_square)
                        {
                            square(*quotient, workspace.scratch5);
                        }
                        else
                        {
                            multiply(*quotient, *basis, workspace.scratch5);
                        }
                        workspace.scratch5.meta.scale = combine.product_scale;

                        GpuCiphertextData product_scaled;
                        const GpuCiphertextData *product = &workspace.scratch5;
                        GpuCiphertextData *product_mutable = &workspace.scratch5;
                        if (!combine.product_scale_plaintext.empty())
                        {
                            multiply_plain(
                                *product,
                                combine.product_scale_plaintext,
                                product_scaled);
                            product_scaled.meta.scale =
                                combine.product_aligned_scale;
                            product = &product_scaled;
                            product_mutable = &product_scaled;
                        }

                        const GpuCiphertextData *remainder =
                            &workspace.eval_mod_nodes[combine.remainder_node];
                        GpuCiphertextData *remainder_mutable =
                            zero_copy_moddrop &&
                                    combine.remainder_node < node_use_counts.size() &&
                                    node_use_counts[combine.remainder_node] == 1
                                ? &workspace.eval_mod_nodes[combine.remainder_node]
                                : nullptr;
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
                            remainder_mutable = &remainder_rescaled;
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
                            remainder_mutable = &remainder_scaled;
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
                                        product_dropped,
                                        remainder->meta.parms_id);
                                    product_dropped.meta.scale = product->meta.scale;
                                    product = &product_dropped;
                                    product_mutable = &product_dropped;
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
                                        remainder_dropped,
                                        product->meta.parms_id);
                                    remainder_dropped.meta.scale = remainder->meta.scale;
                                    remainder = &remainder_dropped;
                                    remainder_mutable = &remainder_dropped;
                                }
                            }
                        }

                        GpuCiphertextData combined_raw;
                        add(*product, *remainder, combined_raw);
                        combined_raw.meta.scale = combine.output_scale;
                        if (combined_raw.size() != 3)
                        {
                            throw std::invalid_argument(
                                "GpuEvaluator::eval_mod_high_precision: lazy combine did not produce size-3 output");
                        }

                        const bool defer_output =
                            combine.output_node < lazy_raw_output.size() &&
                            lazy_raw_output[combine.output_node];
                        if (defer_output)
                        {
                            workspace.eval_mod_nodes[combine.output_node] =
                                std::move(combined_raw);
                        }
                        else
                        {
                            relinearize(
                                combined_raw,
                                relin_keys,
                                workspace.eval_mod_nodes[combine.output_node]);
                            workspace.eval_mod_nodes[combine.output_node].meta.scale =
                                combine.output_scale;
                        }
                        multiply_recorder.end();
                        node_available[combine.output_node] = true;
                        if (workspace.capture_eval_mod_trace)
                        {
                            const GpuCiphertextData *trace_source =
                                &workspace.eval_mod_nodes[combine.output_node];
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

                    const GpuCiphertextData *quotient =
                        &workspace.eval_mod_nodes[combine.quotient_node];
                    if (combine.quotient_rescale_count > 0)
                    {
                        rescale_many(
                            *quotient,
                            workspace.scratch2,
                            combine.quotient_rescale_count);
                        workspace.scratch2.meta.scale =
                            combine.quotient_output_scale;
                        quotient = &workspace.scratch2;
                    }

                    const GpuCiphertextData *basis =
                        &workspace.eval_mod_basis[combine.basis_degree];
                    if (!(quotient->meta.parms_id == basis->meta.parms_id))
                    {
                        if (quotient->meta.q_count == basis->meta.q_count)
                        {
                            throw std::invalid_argument(
                                "GpuEvaluator::eval_mod_high_precision: dynamic combine equal q_count has different parms_id");
                        }
                        if (quotient->meta.q_count > basis->meta.q_count)
                        {
                            drop_modulus(
                                *quotient,
                                workspace.scratch3,
                                basis->meta.parms_id);
                            quotient = &workspace.scratch3;
                        }
                        else
                        {
                            drop_modulus(
                                *basis,
                                workspace.scratch4,
                                quotient->meta.parms_id);
                            basis = &workspace.scratch4;
                        }
                    }

                    const bool is_square = quotient == basis;
                    if (workspace.capture_eval_mod_stage_timing)
                    {
                        multiply_recorder.begin(
                            "EvalMod.mul.combine." +
                                std::to_string(combine_index) +
                                ".T" + std::to_string(combine.basis_degree) +
                                ".q" + std::to_string(quotient->meta.q_count),
                            quotient->meta.q_count,
                            relin_keys.meta.p_count,
                            is_square);
                    }
                    if (is_square)
                    {
                        square(*quotient, workspace.scratch5);
                    }
                    else
                    {
                        multiply(*quotient, *basis, workspace.scratch5);
                    }
                    relinearize(workspace.scratch5, relin_keys, workspace.scratch1);
                    multiply_recorder.end();
                    workspace.scratch1.meta.scale = combine.product_scale;

                    GpuCiphertextData product_scaled;
                    const GpuCiphertextData *product_base = &workspace.scratch1;
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
                            &workspace.eval_mod_nodes[combine.remainder_node];
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
                                // The setup planner treats nearby CKKS scales as
                                // add-compatible, matching the CPU evaluator.  GPU
                                // add is intentionally strict, so preserve the
                                // same represented value while canonicalizing the
                                // metadata to the planner's common add scale.
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
                                drop_modulus(
                                    *product,
                                    product_dropped,
                                    remainder->meta.parms_id);
                                product_dropped.meta.scale =
                                    product->meta.scale;
                                product = &product_dropped;
                            }
                            else
                            {
                                drop_modulus(
                                    *remainder,
                                    remainder_dropped,
                                    product->meta.parms_id);
                                remainder_dropped.meta.scale =
                                    remainder->meta.scale;
                                remainder = &remainder_dropped;
                            }
                        }

                        add(*product, *remainder, workspace.scratch0);
                        workspace.scratch0.meta.scale = combine.output_scale;
                        workspace.eval_mod_nodes[combine.output_node] =
                            std::move(workspace.scratch0);
                    }
                    else
                    {
                        if (product_base == &product_scaled)
                        {
                            workspace.eval_mod_nodes[combine.output_node] =
                                std::move(product_scaled);
                        }
                        else
                        {
                            workspace.eval_mod_nodes[combine.output_node] =
                                std::move(workspace.scratch1);
                        }
                        workspace.eval_mod_nodes[combine.output_node].meta.scale =
                            combine.output_scale;
                    }

                    if (workspace.eval_mod_nodes[combine.output_node].meta.q_count >
                        combine.output_q_count)
                    {
                        const auto &target_level =
                            params_.get_level_by_q_count(combine.output_q_count, 0);
                        drop_modulus(
                            workspace.eval_mod_nodes[combine.output_node],
                            workspace.scratch2,
                            target_level.parms_id);
                        workspace.eval_mod_nodes[combine.output_node] =
                            std::move(workspace.scratch2);
                        workspace.eval_mod_nodes[combine.output_node].meta.scale =
                            combine.output_scale;
                    }
                }
                else
                {
                    if (workspace.capture_eval_mod_stage_timing)
                    {
                        const auto &quotient =
                            workspace.eval_mod_nodes[combine.quotient_node];
                        const auto &basis =
                            workspace.eval_mod_basis[combine.basis_degree];
                        multiply_recorder.begin(
                            "EvalMod.mul.combine." +
                                std::to_string(combine_index) +
                                ".T" + std::to_string(combine.basis_degree) +
                                ".q" + std::to_string(
                                    std::min(
                                        quotient.meta.q_count,
                                        basis.meta.q_count)),
                            std::min(
                                quotient.meta.q_count,
                                basis.meta.q_count),
                            relin_keys.meta.p_count,
                            &quotient == &basis);
                    }
                    multiply_relinearize_rescale(
                        workspace.eval_mod_nodes[combine.quotient_node],
                        workspace.eval_mod_basis[combine.basis_degree],
                        combine.output_scale,
                        logical_rescale_count,
                        workspace.scratch2);
                    multiply_recorder.end();
                    add_aligned(
                        workspace.scratch2,
                        workspace.eval_mod_nodes[combine.remainder_node],
                        combine.output_scale,
                        workspace.eval_mod_nodes[combine.output_node]);
                }
                node_available[combine.output_node] = true;
                if (workspace.capture_eval_mod_trace)
                {
                    workspace.eval_mod_trace_polynomial_combine_outputs.emplace_back();
                    copy_ciphertext_data(
                        workspace.eval_mod_nodes[combine.output_node],
                        workspace.eval_mod_trace_polynomial_combine_outputs.back(),
                        "GpuEvaluator::eval_mod_high_precision trace combine copy");
                    workspace.eval_mod_trace_polynomial_combine_outputs.back().meta.scale =
                        workspace.eval_mod_nodes[combine.output_node].meta.scale;
                }
            }

            const auto result_node = polynomial_result_node;
            if (!node_available[result_node])
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: polynomial result node is unavailable");
            }
            if (workspace.capture_eval_mod_trace)
            {
                copy_ciphertext_data(
                    workspace.eval_mod_nodes[result_node],
                    accumulator,
                    "GpuEvaluator::eval_mod_high_precision trace accumulator copy");
                accumulator.meta.scale =
                    workspace.eval_mod_nodes[result_node].meta.scale;
            }
            else
            {
                accumulator = std::move(workspace.eval_mod_nodes[result_node]);
            }
            if (partitioned_polynomial)
            {
                destination_ciphertext = std::move(accumulator);
                stage_recorder.record(4);
                stage_recorder.record(5);
                stage_recorder.finish(workspace.eval_mod_stage_timing);
                multiply_recorder.finish(workspace.eval_mod_multiply_timings);
                return;
            }
            if (bootstrap_data.eval_mod.dynamic_rescale)
            {
                if (d2d_free_dataflow)
                {
                    rescale_dynamic(accumulator, accumulator, target_scale);
                }
                else
                {
                    rescale_dynamic(accumulator, workspace.scratch2, target_scale);
                    accumulator = std::move(workspace.scratch2);
                }
                if (bootstrap_data.eval_mod.polynomial_output_scale > 0.0)
                {
                    accumulator.meta.scale =
                        bootstrap_data.eval_mod.polynomial_output_scale;
                }
            }
        }
    }
    stage_recorder.record(4);

    if (workspace.capture_eval_mod_trace)
    {
        copy_ciphertext_data(
            accumulator,
            workspace.eval_mod_trace_polynomial_output,
            "GpuEvaluator::eval_mod_high_precision trace polynomial copy");
        workspace.eval_mod_trace_polynomial_output.meta.scale =
            accumulator.meta.scale;
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
                        ".q" + std::to_string(accumulator.meta.q_count),
                    accumulator.meta.q_count,
                    relin_keys.meta.p_count,
                    true);
            }
            square(accumulator, workspace.scratch5);
            if (workspace.capture_eval_mod_trace)
            {
                workspace.eval_mod_trace_double_angle_square_outputs.emplace_back();
                copy_ciphertext_data(
                    workspace.scratch5,
                    workspace.eval_mod_trace_double_angle_square_outputs.back(),
                    "GpuEvaluator::eval_mod_high_precision trace square copy");
                workspace.eval_mod_trace_double_angle_square_outputs.back().meta.scale =
                    workspace.scratch5.meta.scale;
            }
            if (bootstrap_data.eval_mod.dynamic_rescale)
            {
                relinearize(workspace.scratch5, relin_keys, workspace.scratch2);
                multiply_recorder.end();
                if (workspace.capture_eval_mod_trace)
                {
                    workspace.eval_mod_trace_double_angle_relin_outputs.emplace_back();
                    copy_ciphertext_data(
                        workspace.scratch2,
                        workspace.eval_mod_trace_double_angle_relin_outputs.back(),
                        "GpuEvaluator::eval_mod_high_precision trace relin copy");
                    workspace.eval_mod_trace_double_angle_relin_outputs.back().meta.scale =
                        workspace.scratch2.meta.scale;
                }
                add(workspace.scratch2, workspace.scratch2, workspace.scratch3);
                const GpuCiphertextData *pre_rescale_double_angle =
                    &workspace.scratch3;
                if (!double_angle_plaintext.empty())
                {
                    if (!(workspace.scratch3.meta.parms_id ==
                          double_angle_plaintext.meta.parms_id))
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: dynamic double-angle plaintext level mismatch");
                    }
                    add_plain_inplace(
                        workspace.scratch3,
                        double_angle_plaintext,
                        "GpuEvaluator::eval_mod_high_precision dynamic double-angle constant");
                }
                if (double_angle_rescale_count > 0)
                {
                    rescale_many(
                        *pre_rescale_double_angle,
                        workspace.scratch5,
                        double_angle_rescale_count);
                    accumulator = std::move(workspace.scratch5);
                }
                else
                {
                    if (d2d_free_dataflow)
                    {
                        accumulator = std::move(workspace.scratch3);
                    }
                    else
                    {
                        copy_ciphertext_data(
                            *pre_rescale_double_angle,
                            accumulator,
                            "GpuEvaluator::eval_mod_high_precision double-angle no-rescale copy");
                    }
                }
                if (workspace.capture_eval_mod_trace)
                {
                    workspace.eval_mod_trace_double_angle_rescaled_square_outputs.emplace_back();
                    copy_ciphertext_data(
                        accumulator,
                        workspace.eval_mod_trace_double_angle_rescaled_square_outputs.back(),
                        "GpuEvaluator::eval_mod_high_precision trace rescaled double-angle copy");
                    workspace.eval_mod_trace_double_angle_rescaled_square_outputs.back().meta.scale =
                        accumulator.meta.scale;
                }
            }
            else if (double_angle_rescale_count == 2 &&
                     use_rescale_x2() &&
                     use_relinearize_rescale_x2())
            {
                add(workspace.scratch5, workspace.scratch5, workspace.scratch3);
                if (!double_angle_plaintext.empty())
                {
                    if (!(workspace.scratch3.meta.parms_id ==
                          double_angle_plaintext.meta.parms_id))
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: double-angle plaintext level mismatch");
                    }
                    add_plain_inplace(
                        workspace.scratch3,
                        double_angle_plaintext,
                        "GpuEvaluator::eval_mod_high_precision double-angle constant");
                    relinearize_rescale_x2_hybrid(
                        workspace.scratch3,
                        relin_keys,
                        workspace.scratch5);
                }
                else
                {
                    relinearize_rescale_x2_hybrid(
                        workspace.scratch3,
                        relin_keys,
                        workspace.scratch5);
                }
                multiply_recorder.end();
            }
            else
            {
                relinearize(workspace.scratch5, relin_keys, workspace.scratch2);
                multiply_recorder.end();
                add(workspace.scratch2, workspace.scratch2, workspace.scratch3);

                if (!double_angle_plaintext.empty())
                {
                    if (!(workspace.scratch3.meta.parms_id ==
                          double_angle_plaintext.meta.parms_id))
                    {
                        throw std::invalid_argument(
                            "GpuEvaluator::eval_mod_high_precision: double-angle plaintext level mismatch");
                    }
                    add_plain_inplace(
                        workspace.scratch3,
                        double_angle_plaintext,
                        "GpuEvaluator::eval_mod_high_precision fallback double-angle constant");
                    rescale_many(
                        workspace.scratch3,
                        workspace.scratch5,
                        double_angle_rescale_count);
                }
                else
                {
                    rescale_many(
                        workspace.scratch3,
                        workspace.scratch5,
                        double_angle_rescale_count);
                }
            }
            if (!bootstrap_data.eval_mod.dynamic_rescale)
            {
                accumulator = std::move(workspace.scratch5);
            }
            if (workspace.capture_eval_mod_trace)
            {
                workspace.eval_mod_trace_double_angle_outputs.emplace_back();
                copy_ciphertext_data(
                    accumulator,
                    workspace.eval_mod_trace_double_angle_outputs.back(),
                    "GpuEvaluator::eval_mod_high_precision trace double-angle copy");
                workspace.eval_mod_trace_double_angle_outputs.back().meta.scale =
                    accumulator.meta.scale;
            }
        }
    }
    stage_recorder.record(5);

    /*
     * The CPU high-precision evaluator may finish at a lower Q prefix because
     * its recursive evaluator performs additional level alignment. Setup can
     * record that observable output parms_id without moving the recursive
     * control flow into the GPU hot path. Dropping a Q suffix is a contiguous,
     * coefficient-parallel operation and preserves the evaluated CKKS value.
     */
    {
        NvtxRange range("EvalMod output level alignment");
        if (bootstrap_data.eval_mod.output_parms_id != parms_id_zero &&
            accumulator.meta.parms_id != bootstrap_data.eval_mod.output_parms_id)
        {
            if (accumulator.meta.q_count < bootstrap_data.eval_mod.output_q_count)
            {
                throw std::invalid_argument(
                    "GpuEvaluator::eval_mod_high_precision: GPU polynomial schedule ended below the configured CPU output level");
            }
            if (!q_prefix_views ||
                !drop_modulus_inplace(
                    accumulator,
                    bootstrap_data.eval_mod.output_parms_id))
            {
                drop_modulus(
                    accumulator,
                    workspace.scratch2,
                    bootstrap_data.eval_mod.output_parms_id);
                accumulator = std::move(workspace.scratch2);
            }
        }
        if (bootstrap_data.eval_mod.output_scale > 0.0)
        {
            accumulator.meta.scale = bootstrap_data.eval_mod.output_scale;
        }

        destination_ciphertext = std::move(accumulator);
    }
    stage_recorder.finish(workspace.eval_mod_stage_timing);
    multiply_recorder.finish(workspace.eval_mod_multiply_timings);
}

void GpuEvaluator::eval_mod_degree22_root_product(
    const GpuCiphertextData &source_quotient,
    const GpuBootstrapData &bootstrap_data,
    const GpuRelinKeysData &,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_product) const
{
    const auto &eval_mod = bootstrap_data.eval_mod;
    if (eval_mod.polynomial_degree != 22 ||
        eval_mod.polynomial_blocks.size() != 6 ||
        eval_mod.polynomial_combine_steps.size() != 5)
    {
        throw std::invalid_argument(
            "GpuEvaluator::eval_mod_degree22_root_product: expected degree-22 baby-4 plan");
    }
    const auto &root = eval_mod.polynomial_combine_steps.back();
    if (root.basis_degree >= workspace.eval_mod_basis.size() ||
        workspace.eval_mod_basis[root.basis_degree].empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::eval_mod_degree22_root_product: root basis is unavailable");
    }

    const GpuCiphertextData *quotient = &source_quotient;
    if (root.quotient_rescale_count > 0)
    {
        rescale_many(
            source_quotient,
            workspace.scratch0,
            root.quotient_rescale_count);
        workspace.scratch0.meta.scale = root.quotient_output_scale;
        quotient = &workspace.scratch0;
    }

    const GpuCiphertextData *basis =
        &workspace.eval_mod_basis[root.basis_degree];
    if (!(quotient->meta.parms_id == basis->meta.parms_id))
    {
        if (quotient->meta.q_count == basis->meta.q_count)
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_degree22_root_product: equal q-count operands have different parms_id");
        }
        if (quotient->meta.q_count > basis->meta.q_count)
        {
            drop_modulus(
                *quotient,
                workspace.scratch1,
                basis->meta.parms_id);
            workspace.scratch1.meta.scale = quotient->meta.scale;
            quotient = &workspace.scratch1;
        }
        else
        {
            drop_modulus(
                *basis,
                workspace.scratch2,
                quotient->meta.parms_id);
            workspace.scratch2.meta.scale = basis->meta.scale;
            basis = &workspace.scratch2;
        }
    }

    multiply(*quotient, *basis, workspace.scratch5);
    workspace.scratch5.meta.scale = root.product_scale;
    const GpuCiphertextData *product = &workspace.scratch5;
    if (!root.product_scale_plaintext.empty())
    {
        multiply_plain(
            *product,
            root.product_scale_plaintext,
            workspace.scratch3);
        workspace.scratch3.meta.scale = root.product_aligned_scale;
        product = &workspace.scratch3;
    }

    if (root.product_q_count != 0 &&
        product->meta.q_count > root.product_q_count)
    {
        const auto &target_level =
            params_.get_level_by_q_count(root.product_q_count, 0);
        drop_modulus(
            *product,
            destination_product,
            target_level.parms_id);
        destination_product.meta.scale = product->meta.scale;
    }
    else
    {
        destination_product = std::move(
            product == &workspace.scratch3
                ? workspace.scratch3
                : workspace.scratch5);
    }
    if (destination_product.size() != 3)
    {
        throw std::logic_error(
            "GpuEvaluator::eval_mod_degree22_root_product: raw root product is not size 3");
    }
}

void GpuEvaluator::eval_mod_degree22_finish_partials(
    const GpuCiphertextData &source_product,
    const GpuCiphertextData &source_remainder,
    const GpuBootstrapData &bootstrap_data,
    const GpuRelinKeysData &relin_keys,
    GpuBootstrapWorkspace &workspace,
    GpuCiphertextData &destination_ciphertext) const
{
    const auto &eval_mod = bootstrap_data.eval_mod;
    if (eval_mod.polynomial_degree != 22 ||
        eval_mod.polynomial_combine_steps.size() != 5 ||
        source_product.size() != 3 ||
        source_remainder.size() != 3)
    {
        throw std::invalid_argument(
            "GpuEvaluator::eval_mod_degree22_finish_partials: invalid degree-22 partials");
    }
    const auto &root = eval_mod.polynomial_combine_steps.back();
    if (root.remainder_rescale_count != 0 ||
        !root.remainder_scale_plaintext.empty())
    {
        throw std::invalid_argument(
            "GpuEvaluator::eval_mod_degree22_finish_partials: root remainder is not lazy-compatible");
    }

    const GpuCiphertextData *product = &source_product;
    const GpuCiphertextData *remainder = &source_remainder;
    if (!(product->meta.parms_id == remainder->meta.parms_id))
    {
        if (product->meta.q_count == remainder->meta.q_count)
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_degree22_finish_partials: equal q-count partials have different parms_id");
        }
        if (product->meta.q_count > remainder->meta.q_count)
        {
            drop_modulus(
                *product,
                workspace.scratch0,
                remainder->meta.parms_id);
            workspace.scratch0.meta.scale = product->meta.scale;
            product = &workspace.scratch0;
        }
        else
        {
            drop_modulus(
                *remainder,
                workspace.scratch1,
                product->meta.parms_id);
            workspace.scratch1.meta.scale = remainder->meta.scale;
            remainder = &workspace.scratch1;
        }
    }

    add(*product, *remainder, workspace.scratch5);
    workspace.scratch5.meta.scale = root.output_scale;
    relinearize(workspace.scratch5, relin_keys, workspace.scratch2);
    workspace.scratch2.meta.scale = root.output_scale;
    GpuCiphertextData accumulator = std::move(workspace.scratch2);

    if (root.output_q_count != 0 &&
        accumulator.meta.q_count > root.output_q_count)
    {
        const auto &target_level =
            params_.get_level_by_q_count(root.output_q_count, 0);
        drop_modulus(
            accumulator,
            workspace.scratch0,
            target_level.parms_id);
        accumulator = std::move(workspace.scratch0);
        accumulator.meta.scale = root.output_scale;
    }

    const double target_scale = eval_mod.target_scale > 0.0
        ? eval_mod.target_scale
        : bootstrap_data.eval_mod_target_scale;
    if (eval_mod.dynamic_rescale)
    {
        rescale_dynamic(accumulator, workspace.scratch2, target_scale);
        accumulator = std::move(workspace.scratch2);
        if (eval_mod.polynomial_output_scale > 0.0)
        {
            accumulator.meta.scale = eval_mod.polynomial_output_scale;
        }
    }

    const auto &double_angle_constants =
        !eval_mod.double_angle_constants.empty()
            ? eval_mod.double_angle_constants
            : bootstrap_data.double_angle_plaintexts;
    const auto &double_angle_rescale_counts =
        eval_mod.double_angle_rescale_counts;
    const auto logical_rescale_count =
        std::max(eval_mod.logical_rescale_count, std::uint32_t{1});
    for (std::size_t index = 0;
         index < double_angle_constants.size();
         ++index)
    {
        const auto rescale_count =
            index < double_angle_rescale_counts.size()
                ? std::max(
                      double_angle_rescale_counts[index],
                      std::uint32_t{1})
                : logical_rescale_count;
        square(accumulator, workspace.scratch5);
        relinearize(workspace.scratch5, relin_keys, workspace.scratch2);
        add(workspace.scratch2, workspace.scratch2, workspace.scratch3);
        if (!double_angle_constants[index].empty())
        {
            add_plain(
                workspace.scratch3,
                double_angle_constants[index],
                workspace.scratch4);
            workspace.scratch3 = std::move(workspace.scratch4);
        }
        if (rescale_count > 0)
        {
            rescale_many(
                workspace.scratch3,
                workspace.scratch5,
                rescale_count);
            accumulator = std::move(workspace.scratch5);
        }
        else
        {
            accumulator = std::move(workspace.scratch3);
        }
    }

    if (eval_mod.output_parms_id != parms_id_zero &&
        accumulator.meta.parms_id != eval_mod.output_parms_id)
    {
        if (accumulator.meta.q_count < eval_mod.output_q_count)
        {
            throw std::invalid_argument(
                "GpuEvaluator::eval_mod_degree22_finish_partials: output fell below configured level");
        }
        drop_modulus(
            accumulator,
            workspace.scratch2,
            eval_mod.output_parms_id);
        accumulator = std::move(workspace.scratch2);
    }
    if (eval_mod.output_scale > 0.0)
    {
        accumulator.meta.scale = eval_mod.output_scale;
    }
    destination_ciphertext = std::move(accumulator);
}

}  // namespace gpu
}  // namespace poseidon
