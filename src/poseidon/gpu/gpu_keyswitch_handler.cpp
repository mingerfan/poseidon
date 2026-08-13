#include "poseidon/gpu/gpu_keyswitch_handler.h"
#include "poseidon/gpu/kernels/gpu_double_hoist_kernels.h"
#include "poseidon/gpu/kernels/gpu_keyswitch_kernels.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"
#include "poseidon/gpu/kernels/gpu_rescale_kernels.h"

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poseidon
{
namespace gpu
{
namespace
{

constexpr std::size_t kRelinKeyPower2Index = 0;
constexpr std::size_t kSwitchKeyComponentCount = 2;
constexpr const char *kFuseDecompQEnv =
    "POSEIDON_KEYSWITCH_FUSE_DECOMP_Q";
constexpr const char *kFuseModupNttHeadEnv =
    "POSEIDON_KEYSWITCH_FUSE_MODUP_NTT_HEAD";
constexpr const char *kBconvRowTiledEnv =
    "POSEIDON_KEYSWITCH_BCONV_ROW_TILED";
constexpr const char *kBconvRowTiled8Env =
    "POSEIDON_KEYSWITCH_BCONV_ROW_TILED_8";
constexpr const char *kPToQRowTiled8Env =
    "POSEIDON_KEYSWITCH_P_TO_Q_ROW_TILED_8";
constexpr const char *kP9PreweightPEnv =
    "POSEIDON_KEYSWITCH_P9_PREWEIGHT_P";
constexpr const char *kP9PToQRowTiled8Env =
    "POSEIDON_KEYSWITCH_P9_P_TO_Q_ROW_TILED_8";
constexpr const char *kP9FourstepPInttEnv =
    "POSEIDON_KEYSWITCH_P9_FOURSTEP_P_INTT";
constexpr const char *kP9FourstepQpEnv =
    "POSEIDON_KEYSWITCH_P9_FOURSTEP_QP";
constexpr const char *kP9PToQFourstepEnv =
    "POSEIDON_KEYSWITCH_P9_P_TO_Q_FOURSTEP";
constexpr const char *kDoubleHoistP9ModupRowTiled8Env =
    "POSEIDON_DOUBLE_HOIST_P9_MODUP_ROW_TILED_8";
constexpr const char *kDoubleHoistP9PreweightPEnv =
    "POSEIDON_DOUBLE_HOIST_P9_PREWEIGHT_P";
constexpr const char *kDoubleHoistP9PToQRowTiled8Env =
    "POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_ROW_TILED_8";
constexpr const char *kDoubleHoistP9QpFourstepEnv =
    "POSEIDON_DOUBLE_HOIST_P9_QP_FOURSTEP";
constexpr const char *kDoubleHoistP9PToQFourstepEnv =
    "POSEIDON_DOUBLE_HOIST_P9_P_TO_Q_FOURSTEP";
constexpr const char *kP9ModupFourstepPhase1FusedEnv =
    "POSEIDON_P9_MODUP_FOURSTEP_PHASE1_FUSED";
constexpr const char *kFourstepC2InttEnv =
    "POSEIDON_KEYSWITCH_FOURSTEP_C2_INTT";
constexpr const char *kFourstepAllNttEnv =
    "POSEIDON_KEYSWITCH_FOURSTEP_ALL_NTT";
constexpr const char *kFourstepPhase2MacEnv =
    "POSEIDON_KEYSWITCH_FOURSTEP_PHASE2_MAC";
constexpr const char *kFourstepFinalizeFusedEnv =
    "POSEIDON_KEYSWITCH_FOURSTEP_FINALIZE_FUSED";
constexpr const char *kPersistentRelinearizeEnv =
    "POSEIDON_KEYSWITCH_PERSISTENT_RELIN";

class NvtxRange
{
public:
    explicit NvtxRange(std::string name)
        : name_(std::move(name))
    {
        nvtxRangePushA(name_.c_str());
    }

    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;

    ~NvtxRange()
    {
        nvtxRangePop();
    }

private:
    std::string name_;
};

std::size_t checked_mul(std::size_t a, std::size_t b, const char *what)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    {
        throw std::overflow_error(what);
    }
    return a * b;
}

bool use_fused_decomp_q()
{
    const char *raw = std::getenv(kFuseDecompQEnv);
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

bool use_persistent_relinearize()
{
    const char *raw = std::getenv(kPersistentRelinearizeEnv);
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

bool use_fused_modup_ntt_head()
{
    const char *raw = std::getenv(kFuseModupNttHeadEnv);
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

bool use_bconv_row_tiled()
{
    const char *raw = std::getenv(kBconvRowTiledEnv);
    if (raw == nullptr || *raw == '\0')
    {
        return false;
    }

    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_bconv_row_tiled8()
{
    const char *raw = std::getenv(kBconvRowTiled8Env);
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

bool use_p_to_q_row_tiled8()
{
    const char *raw = std::getenv(kPToQRowTiled8Env);
    if (raw == nullptr || *raw == '\0')
    {
        return false;
    }

    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_p9_preweight_p(
    std::size_t degree,
    std::size_t base_p_size)
{
    if (degree != 65536 || base_p_size != 9)
    {
        return false;
    }

    const char *raw = std::getenv(kP9PreweightPEnv);
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

bool use_p9_p_to_q_row_tiled8(
    std::size_t degree,
    std::size_t base_p_size)
{
    if (degree != 65536 || base_p_size != 9)
    {
        return false;
    }

    const char *raw = std::getenv(kP9PToQRowTiled8Env);
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

bool use_p9_fourstep_p_intt(
    std::size_t degree,
    std::size_t base_p_size)
{
    if (degree != 65536 || base_p_size != 9)
    {
        return false;
    }

    const char *raw = std::getenv(kP9FourstepPInttEnv);
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

bool use_p9_fourstep_qp(
    std::size_t degree,
    std::size_t base_p_size)
{
    if (degree != 65536 || base_p_size != 9)
    {
        return false;
    }

    const char *raw = std::getenv(kP9FourstepQpEnv);
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

bool use_p9_p_to_q_fourstep(
    std::size_t degree,
    std::size_t base_p_size)
{
    if (degree != 65536 || base_p_size != 9)
    {
        return false;
    }

    const char *raw = std::getenv(kP9PToQFourstepEnv);
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

bool use_double_hoist_p9_modup_row_tiled8(
    std::size_t degree,
    std::size_t base_p_size)
{
    if (degree != 65536 || base_p_size != 9)
    {
        return false;
    }

    const char *raw = std::getenv(kDoubleHoistP9ModupRowTiled8Env);
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

bool use_double_hoist_p9_preweight_p(
    std::size_t degree,
    std::size_t base_p_size)
{
    if (degree != 65536 || base_p_size != 9)
    {
        return false;
    }

    const char *raw = std::getenv(kDoubleHoistP9PreweightPEnv);
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

bool use_double_hoist_p9_p_to_q_row_tiled8(
    std::size_t degree,
    std::size_t base_p_size)
{
    if (degree != 65536 || base_p_size != 9)
    {
        return false;
    }

    const char *raw = std::getenv(kDoubleHoistP9PToQRowTiled8Env);
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

bool use_double_hoist_p9_qp_fourstep(
    std::size_t degree,
    std::size_t base_p_size)
{
    if (degree != 65536 || base_p_size != 9)
    {
        return false;
    }

    const char *raw = std::getenv(kDoubleHoistP9QpFourstepEnv);
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

bool use_double_hoist_p9_p_to_q_fourstep(
    std::size_t degree,
    std::size_t base_p_size)
{
    if (degree != 65536 || base_p_size != 9)
    {
        return false;
    }

    const char *raw = std::getenv(kDoubleHoistP9PToQFourstepEnv);
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

bool use_p9_modup_fourstep_phase1_fused(
    std::size_t degree,
    std::size_t base_p_size)
{
    if (degree != 65536 || base_p_size != 9)
    {
        return false;
    }

    const char *raw = std::getenv(kP9ModupFourstepPhase1FusedEnv);
    if (raw == nullptr || *raw == '\0')
    {
        return false;
    }

    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

void launch_double_hoist_p_to_q_forward_ntt(
    GpuWord *converted_q0,
    GpuWord *converted_q1,
    GpuWord *p_coeff0,
    GpuWord *p_coeff1,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    const std::size_t p_count = parameter_shard.hybrid_base_p_count;
    const bool preweight =
        use_double_hoist_p9_preweight_p(degree, p_count);
    const bool row_tiled8 =
        use_double_hoist_p9_p_to_q_row_tiled8(degree, p_count);
    const bool fourstep =
        use_double_hoist_p9_p_to_q_fourstep(degree, p_count);

    if (preweight)
    {
        NvtxRange range("double_hoist.moddown.preweight_p9");
        kernel::launch_hybrid_preweight_p_two_components(
            p_coeff0,
            p_coeff1,
            parameter_shard,
            degree);
    }

    if (fourstep)
    {
        NvtxRange range("double_hoist.moddown.p_to_q.fourstep_p9");
        kernel::launch_hybrid_convert_p9_to_q_forward_ntt_two_components_fourstep_65536(
            converted_q0,
            converted_q1,
            p_coeff0,
            p_coeff1,
            parameter_shard,
            degree,
            preweight);
    }
    else if (row_tiled8)
    {
        NvtxRange range("double_hoist.moddown.p_to_q.row_tiled8");
        kernel::launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8(
            converted_q0,
            converted_q1,
            p_coeff0,
            p_coeff1,
            parameter_shard,
            degree,
            preweight);
    }
    else
    {
        NvtxRange range("double_hoist.moddown.p_to_q.generic");
        kernel::launch_hybrid_convert_p_to_q_forward_ntt(
            converted_q0,
            converted_q1,
            p_coeff0,
            p_coeff1,
            parameter_shard,
            degree,
            preweight);
    }
}

bool use_fourstep_c2_intt()
{
    const char *raw = std::getenv(kFourstepC2InttEnv);
    if (raw == nullptr || *raw == '\0')
    {
        return false;
    }

    const std::string value(raw);
    return value != "0" &&
           value != "OFF" &&
           value != "off" &&
           value != "false" &&
           value != "FALSE";
}

bool use_fourstep_all_ntt(
    std::size_t degree,
    std::size_t base_q_size,
    std::size_t base_p_size)
{
    const bool supported =
        degree == 65536 &&
        base_p_size == 2 &&
        base_q_size >= 2 &&
        (base_q_size & 1U) == 0;
    if (!supported)
    {
        return false;
    }

    const char *raw = std::getenv(kFourstepAllNttEnv);
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

bool use_fourstep_phase2_mac()
{
    const char *raw = std::getenv(kFourstepPhase2MacEnv);
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

bool use_fourstep_finalize_fused()
{
    const char *raw = std::getenv(kFourstepFinalizeFusedEnv);
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

bool same_shard_placement(
    const GpuConstPolyShardView &lhs,
    const GpuConstPolyShardView &rhs)
{
    return lhs.device_id == rhs.device_id &&
           lhs.limb_begin == rhs.limb_begin &&
           lhs.limb_count == rhs.limb_count &&
           lhs.coeff_begin == rhs.coeff_begin &&
           lhs.coeff_count == rhs.coeff_count;
}

bool same_shard_placement(
    const GpuPolyShardView &lhs,
    const GpuConstPolyShardView &rhs)
{
    return lhs.device_id == rhs.device_id &&
           lhs.limb_begin == rhs.limb_begin &&
           lhs.limb_count == rhs.limb_count &&
           lhs.coeff_begin == rhs.coeff_begin &&
           lhs.coeff_count == rhs.coeff_count;
}

const GpuConstRNSPolyView &find_key_poly(
    const GpuConstEvaluationKeyView &key_view,
    const GpuEvaluationKeyData &key_data,
    std::size_t key_index,
    std::size_t decomposition_index,
    std::size_t component_index)
{
    for (std::size_t i = 0; i < key_data.poly_metadata_.size(); ++i)
    {
        const auto &metadata = key_data.poly_metadata_[i];
        if (metadata.key_index == key_index &&
            metadata.decomposition_index == decomposition_index &&
            metadata.component_index == component_index)
        {
            if (metadata.poly_id >= key_view.polys.size())
            {
                throw std::out_of_range(
                    "GpuKeySwitchHandler: key poly metadata poly_id is out of range");
            }
            return key_view.polys[metadata.poly_id];
        }
    }

    throw std::invalid_argument(
        "GpuKeySwitchHandler: missing key-switch key polynomial");
}

struct HybridKeyComponentView
{
    int device_id = 0;
    const GpuWord *q_ptr = nullptr;
    const GpuWord *p_ptr = nullptr;
};

HybridKeyComponentView make_hybrid_key_component_view(
    const GpuConstRNSPolyView &poly,
    const GpuConstEvaluationKeyView &level_view,
    const GpuEvaluationKeyData &storage)
{
    if (poly.shards.size() != 1)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler: zero-copy key view requires one full shard");
    }
    if (level_view.storage_q_count != storage.meta.q_count ||
        level_view.meta.q_count == 0 ||
        level_view.meta.q_count > level_view.storage_q_count)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler: invalid zero-copy key Q prefix metadata");
    }

    const auto &shard = poly.shards.front();
    if (shard.ptr == nullptr ||
        shard.limb_begin != 0 ||
        shard.limb_count != storage.meta.q_count + storage.meta.p_count ||
        shard.coeff_begin != 0 ||
        shard.coeff_count != storage.meta.degree)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler: key must retain one full [Q_storage | P] allocation");
    }

    HybridKeyComponentView result;
    result.device_id = shard.device_id;
    result.q_ptr = shard.ptr;
    result.p_ptr = shard.ptr + storage.meta.q_count * storage.meta.degree;
    return result;
}

const GpuParameterShard *find_parameter_shard(
    const GpuLevelInfo &level_info,
    const GpuConstPolyShardView &shard)
{
    for (const auto &candidate : level_info.shards)
    {
        const bool same_device = candidate.device_id == shard.device_id;
        const bool covers_limb =
            shard.limb_begin >= candidate.limb_begin &&
            shard.limb_begin + shard.limb_count <=
                candidate.limb_begin + candidate.limb_count;

        if (same_device && covers_limb)
        {
            return &candidate;
        }
    }

    return nullptr;
}

void copy_poly_shard(
    const char *name,
    const GpuPolyShardView &destination,
    const GpuConstPolyShardView &source)
{
    if (!same_shard_placement(destination, source))
    {
        throw std::invalid_argument(std::string(name) + ": shard placement mismatch");
    }

    const std::size_t word_count = checked_mul(
        destination.limb_count,
        destination.coeff_count,
        "GpuKeySwitchHandler copy word count overflow");
    gpu_check_cuda(cudaSetDevice(destination.device_id), name);
    gpu_check_cuda(
        cudaMemcpy(
            destination.ptr,
            source.ptr,
            word_count * sizeof(GpuWord),
            cudaMemcpyDeviceToDevice),
        name);
}

void copy_initial_components(
    GpuCiphertextView &destination,
    const GpuConstCiphertextView &source)
{
    for (std::size_t component = 0; component < destination.polys.size(); ++component)
    {
        copy_poly_shard(
            "GpuKeySwitchHandler::copy_initial_components",
            destination.polys[component].shards.front(),
            source.polys[component].shards.front());
    }
}

struct HybridScratch
{
    int device_id = 0;
    std::size_t degree = 0;
    std::size_t base_q_size = 0;
    std::size_t base_p_size = 0;
    std::size_t q_word_count = 0;
    std::size_t p_word_count = 0;

    DeviceVector<GpuWord> c2_intt;
    DeviceVector<GpuWord> modup_q;
    DeviceVector<GpuWord> modup_p;
    DeviceVector<GpuWord> accum_q0;
    DeviceVector<GpuWord> accum_q1;
    DeviceVector<GpuWord> accum_p0;
    DeviceVector<GpuWord> accum_p1;
    DeviceVector<GpuWord> fourstep_q0;
    DeviceVector<GpuWord> fourstep_q1;
    DeviceVector<GpuWord> fourstep_p0;
    DeviceVector<GpuWord> fourstep_p1;
    DeviceVector<GpuWord> relin_rescale_dropped_ntt;
    DeviceVector<GpuWord> relin_rescale_dropped_coeff;
    DeviceVector<GpuWord> relin_rescale_correction;
    DeviceVector<GpuWord> relin_rescale_correction_ntt;

    GpuPolyShardView c2_intt_view()
    {
        GpuPolyShardView result;
        result.device_id = device_id;
        result.ptr = c2_intt.data();
        result.limb_begin = 0;
        result.limb_count = base_q_size;
        result.coeff_begin = 0;
        result.coeff_count = degree;
        return result;
    }
};

GpuPolyShardView make_scratch_q_view(
    GpuWord *ptr,
    const HybridScratch &scratch)
{
    GpuPolyShardView result;
    result.device_id = scratch.device_id;
    result.ptr = ptr;
    result.limb_begin = 0;
    result.limb_count = scratch.base_q_size;
    result.coeff_begin = 0;
    result.coeff_count = scratch.degree;
    return result;
}

GpuPolyShardView make_scratch_p_view(
    GpuWord *ptr,
    const HybridScratch &scratch)
{
    GpuPolyShardView result;
    result.device_id = scratch.device_id;
    result.ptr = ptr;
    result.limb_begin = scratch.base_q_size;
    result.limb_count = scratch.base_p_size;
    result.coeff_begin = 0;
    result.coeff_count = scratch.degree;
    return result;
}

GpuConstPolyShardView as_const_shard(const GpuPolyShardView &shard)
{
    GpuConstPolyShardView result;
    result.device_id = shard.device_id;
    result.ptr = shard.ptr;
    result.limb_begin = shard.limb_begin;
    result.limb_count = shard.limb_count;
    result.coeff_begin = shard.coeff_begin;
    result.coeff_count = shard.coeff_count;
    return result;
}

void validate_hybrid_parameter_shape(
    const char *name,
    const HybridScratch &scratch,
    const GpuParameterShard &parameter_shard)
{
    if (parameter_shard.hybrid_base_q_count != scratch.base_q_size ||
        parameter_shard.hybrid_base_p_count != scratch.base_p_size)
    {
        throw std::invalid_argument(
            std::string(name) + ": HYBRID parameter shape mismatch");
    }
}

HybridScratch allocate_hybrid_scratch(
    int device_id,
    std::size_t degree,
    std::size_t base_q_size,
    std::size_t base_p_size)
{
    HybridScratch scratch;
    scratch.device_id = device_id;
    scratch.degree = degree;
    scratch.base_q_size = base_q_size;
    scratch.base_p_size = base_p_size;
    scratch.q_word_count = checked_mul(
        base_q_size,
        degree,
        "GpuKeySwitchHandler q scratch size overflow");
    scratch.p_word_count = checked_mul(
        base_p_size,
        degree,
        "GpuKeySwitchHandler p scratch size overflow");

    scratch.c2_intt.allocate(scratch.q_word_count, device_id);
    scratch.modup_q.allocate(scratch.q_word_count, device_id);
    scratch.modup_p.allocate(scratch.p_word_count, device_id);
    /* 累加后ksk0和ksk1对应的结果 */
    scratch.accum_q0.allocate(scratch.q_word_count, device_id);
    scratch.accum_q1.allocate(scratch.q_word_count, device_id);
    scratch.accum_p0.allocate(scratch.p_word_count, device_id);
    scratch.accum_p1.allocate(scratch.p_word_count, device_id);

    if (use_fourstep_all_ntt(degree, base_q_size, base_p_size))
    {
        scratch.fourstep_q0.allocate(scratch.q_word_count, device_id);
        scratch.fourstep_q1.allocate(scratch.q_word_count, device_id);
        scratch.fourstep_p0.allocate(scratch.p_word_count, device_id);
        scratch.fourstep_p1.allocate(scratch.p_word_count, device_id);
    }
    else if (use_p9_fourstep_p_intt(degree, base_p_size) ||
             use_p9_p_to_q_fourstep(
                 degree,
                 base_p_size) ||
             use_p9_fourstep_qp(degree, base_p_size))
    {
        if (use_p9_fourstep_qp(degree, base_p_size))
        {
            scratch.fourstep_q0.allocate(scratch.q_word_count, device_id);
        }
        scratch.fourstep_p0.allocate(scratch.p_word_count, device_id);
        scratch.fourstep_p1.allocate(scratch.p_word_count, device_id);
    }
    return scratch;
}

void ensure_hybrid_scratch(
    HybridScratch &scratch,
    int device_id,
    std::size_t degree,
    std::size_t base_q_size,
    std::size_t base_p_size)
{
    const std::size_t q_word_count = checked_mul(
        base_q_size,
        degree,
        "GpuKeySwitchHandler persistent q scratch size overflow");
    const std::size_t p_word_count = checked_mul(
        base_p_size,
        degree,
        "GpuKeySwitchHandler persistent p scratch size overflow");

    const auto ensure_capacity =
        [device_id](DeviceVector<GpuWord> &buffer, std::size_t word_count)
        {
            if (buffer.size() < word_count ||
                (!buffer.empty() && buffer.device_id() != device_id))
            {
                buffer.allocate(word_count, device_id);
            }
        };

    ensure_capacity(scratch.c2_intt, q_word_count);
    ensure_capacity(scratch.modup_q, q_word_count);
    ensure_capacity(scratch.modup_p, p_word_count);
    ensure_capacity(scratch.accum_q0, q_word_count);
    ensure_capacity(scratch.accum_q1, q_word_count);
    ensure_capacity(scratch.accum_p0, p_word_count);
    ensure_capacity(scratch.accum_p1, p_word_count);

    if (use_fourstep_all_ntt(degree, base_q_size, base_p_size))
    {
        ensure_capacity(scratch.fourstep_q0, q_word_count);
        ensure_capacity(scratch.fourstep_q1, q_word_count);
        ensure_capacity(scratch.fourstep_p0, p_word_count);
        ensure_capacity(scratch.fourstep_p1, p_word_count);
    }
    else if (use_p9_fourstep_p_intt(degree, base_p_size) ||
             use_p9_p_to_q_fourstep(
                 degree,
                 base_p_size) ||
             use_p9_fourstep_qp(degree, base_p_size))
    {
        if (use_p9_fourstep_qp(degree, base_p_size))
        {
            ensure_capacity(scratch.fourstep_q0, q_word_count);
        }
        ensure_capacity(scratch.fourstep_p0, p_word_count);
        ensure_capacity(scratch.fourstep_p1, p_word_count);
    }
    scratch.device_id = device_id;
    scratch.degree = degree;
    scratch.base_q_size = base_q_size;
    scratch.base_p_size = base_p_size;
    scratch.q_word_count = q_word_count;
    scratch.p_word_count = p_word_count;
}

void ensure_hybrid_relin_rescale_x2_scratch(
    HybridScratch &scratch)
{
    if (scratch.degree == 0 || scratch.base_q_size < 3)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler relin-rescale_x2 scratch requires q_count >= 3");
    }
    const auto ensure_capacity =
        [&scratch](DeviceVector<GpuWord> &buffer, std::size_t word_count)
        {
            if (buffer.size() < word_count ||
                (!buffer.empty() && buffer.device_id() != scratch.device_id))
            {
                buffer.allocate(word_count, scratch.device_id);
            }
        };
    const std::size_t dropped_words = 2 * 2 * scratch.degree;
    const std::size_t correction_words =
        2 * (scratch.base_q_size - 2) * scratch.degree;
    ensure_capacity(scratch.relin_rescale_dropped_ntt, dropped_words);
    ensure_capacity(scratch.relin_rescale_dropped_coeff, dropped_words);
    ensure_capacity(scratch.relin_rescale_correction, correction_words);
    ensure_capacity(scratch.relin_rescale_correction_ntt, correction_words);
}

void inverse_ntt_switch_poly(
    HybridScratch &scratch,
    const GpuConstRNSPolyView &switch_poly_ntt,
    const GpuLevelInfo &level_info)
{
    const bool fourstep = use_fourstep_c2_intt() || use_fourstep_all_ntt(
        scratch.degree,
        scratch.base_q_size,
        scratch.base_p_size);
    NvtxRange range(fourstep
        ? "keyswitch.intt_switch_poly.fourstep"
        : "keyswitch.intt_switch_poly.fused3");
    const auto &source_shard = switch_poly_ntt.shards.front();
    auto destination_shard = scratch.c2_intt_view();
    const auto *parameter_shard = find_parameter_shard(level_info, source_shard);
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::inverse_ntt_switch_poly: no matching parameter shard");
    }

    if (fourstep)
    {
        kernel::launch_inverse_ntt_poly_shard_fourstep_65536(
            destination_shard,
            source_shard,
            *parameter_shard,
            scratch.degree);
    }
    else
    {
        kernel::launch_inverse_ntt_poly_shard(
            destination_shard,
            source_shard,
            *parameter_shard,
            scratch.degree);
    }
}

/* 处理单个dnum分量的计算函数 */
void process_hybrid_decomposition_block(
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    HybridScratch &scratch,
    const GpuConstRNSPolyView &switch_poly_ntt,
    const HybridKeyComponentView &key_component0,
    const HybridKeyComponentView &key_component1,
    const GpuLevelInfo &level_info,
    bool fuse_decomp_q,
    bool fuse_modup_ntt_head,
    bool bconv_row_tiled,
    bool bconv_row_tiled8)
{
    NvtxRange block_range(
        "keyswitch.dnum[" + std::to_string(decomp_index) + "]");
    const auto &switch_poly_shard = switch_poly_ntt.shards.front();
    if (key_component0.device_id != key_component1.device_id ||
        key_component0.device_id != switch_poly_shard.device_id)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::process_hybrid_decomposition_block: key device placement mismatch");
    }

    const auto *parameter_shard = find_parameter_shard(
        level_info,
        switch_poly_shard);
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::process_hybrid_decomposition_block: no matching parameter shard");
    }
    validate_hybrid_parameter_shape(
        "GpuKeySwitchHandler::process_hybrid_decomposition_block",
        scratch,
        *parameter_shard);

    const bool fourstep_all_ntt = use_fourstep_all_ntt(
        scratch.degree,
        scratch.base_q_size,
        scratch.base_p_size);
    const bool p9_fourstep_qp = use_p9_fourstep_qp(
        scratch.degree,
        scratch.base_p_size);
    if (fourstep_all_ntt || p9_fourstep_qp)
    {
        const bool fuse_phase2_mac = use_fourstep_phase2_mac();
        const bool fuse_modup_phase1 =
            use_p9_modup_fourstep_phase1_fused(
                scratch.degree,
                scratch.base_p_size);
        if (fuse_modup_phase1)
        {
            NvtxRange range(
                "keyswitch.dnum.modup_fourstep_phase1.fused_p9");
            kernel::launch_hybrid_modup_p9_forward_ntt_qp_active_phase1_fourstep_65536(
                scratch.fourstep_q0.data(),
                scratch.fourstep_p0.data(),
                scratch.c2_intt.data(),
                decomp_index,
                decomp_limb_begin,
                decomp_limb_count,
                *parameter_shard,
                scratch.degree);
        }
        else if (fourstep_all_ntt)
        {
            NvtxRange range("keyswitch.dnum.modup.row_tiled8.coeff");
            kernel::launch_hybrid_modup_decomposition_row_tiled8(
                scratch.modup_q.data(),
                scratch.modup_p.data(),
                scratch.c2_intt.data(),
                decomp_index,
                decomp_limb_begin,
                decomp_limb_count,
                *parameter_shard,
                scratch.degree);
        }
        else
        {
            NvtxRange range("keyswitch.dnum.modup.p9.row_tiled8.coeff");
            kernel::launch_hybrid_modup_decomposition_row_tiled8(
                scratch.modup_q.data(),
                scratch.modup_p.data(),
                scratch.c2_intt.data(),
                decomp_index,
                decomp_limb_begin,
                decomp_limb_count,
                *parameter_shard,
                scratch.degree);
        }

        if (fuse_phase2_mac)
        {
            NvtxRange range(
                "keyswitch.dnum.fourstep_forward_ntt_qp_active.phase2_mac");
            kernel::launch_forward_ntt_qp_active_fourstep_mul_accumulate_two_components_65536(
                scratch.fourstep_q0.data(),
                scratch.fourstep_p0.data(),
                fuse_modup_phase1
                    ? scratch.fourstep_q0.data()
                    : scratch.modup_q.data(),
                fuse_modup_phase1
                    ? scratch.fourstep_p0.data()
                    : scratch.modup_p.data(),
                scratch.accum_q0.data(),
                scratch.accum_p0.data(),
                scratch.accum_q1.data(),
                scratch.accum_p1.data(),
                switch_poly_shard.ptr,
                key_component0.q_ptr,
                key_component0.p_ptr,
                key_component1.q_ptr,
                key_component1.p_ptr,
                decomp_limb_begin,
                decomp_limb_count,
                decomp_index == 0,
                *parameter_shard,
                scratch.degree,
                fuse_modup_phase1);
        }
        else
        {
            {
                NvtxRange range(
                    "keyswitch.dnum.fourstep_forward_ntt_qp_active");
                kernel::launch_forward_ntt_qp_active_fourstep_65536(
                    scratch.fourstep_q0.data(),
                    scratch.fourstep_p0.data(),
                    fuse_modup_phase1
                        ? scratch.fourstep_q0.data()
                        : scratch.modup_q.data(),
                    fuse_modup_phase1
                        ? scratch.fourstep_p0.data()
                        : scratch.modup_p.data(),
                    decomp_limb_begin,
                    decomp_limb_count,
                    *parameter_shard,
                    scratch.degree,
                    fuse_modup_phase1);
            }
            {
                NvtxRange range(
                    "keyswitch.dnum.fourstep_multiply_accumulate.c01");
                kernel::launch_hybrid_multiply_accumulate_two_components(
                    scratch.accum_q0.data(),
                    scratch.accum_p0.data(),
                    scratch.accum_q1.data(),
                    scratch.accum_p1.data(),
                    scratch.fourstep_q0.data(),
                    scratch.fourstep_p0.data(),
                    switch_poly_shard.ptr,
                    key_component0.q_ptr,
                    key_component0.p_ptr,
                    key_component1.q_ptr,
                    key_component1.p_ptr,
                    *parameter_shard,
                    scratch.degree,
                    decomp_limb_begin,
                    decomp_limb_count,
                    decomp_index == 0);
            }
        }
        return;
    }

    /* Compatibility fallback for parameter shapes not supported by the
       N=65536, P=2 four-step pipeline, or for an explicit env rollback. */
    /* 将dnum片段进行模升，每个片段扩展到完整的32+6=38个模数 */
    {
        NvtxRange range(!fuse_modup_ntt_head
            ? "keyswitch.dnum.modup"
            : (bconv_row_tiled8
                ? "keyswitch.dnum.modup_forward_ntt_head.row_tiled8"
                : (bconv_row_tiled
                    ? "keyswitch.dnum.modup_forward_ntt_head.row_tiled"
                    : "keyswitch.dnum.modup_forward_ntt_head")));
        if (fuse_modup_ntt_head)
        {
            if (bconv_row_tiled8)
            {
                kernel::launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled8(
                    scratch.modup_q.data(),
                    scratch.modup_p.data(),
                    scratch.c2_intt.data(),
                    decomp_index,
                    decomp_limb_begin,
                    decomp_limb_count,
                    *parameter_shard,
                    scratch.degree);
            }
            else if (bconv_row_tiled)
            {
                kernel::launch_hybrid_modup_decomposition_forward_ntt_first_stage_row_tiled(
                    scratch.modup_q.data(),
                    scratch.modup_p.data(),
                    scratch.c2_intt.data(),
                    decomp_index,
                    decomp_limb_begin,
                    decomp_limb_count,
                    *parameter_shard,
                    scratch.degree);
            }
            else
            {
                kernel::launch_hybrid_modup_decomposition_forward_ntt_first_stage(
                    scratch.modup_q.data(),
                    scratch.modup_p.data(),
                    scratch.c2_intt.data(),
                    decomp_index,
                    decomp_limb_begin,
                    decomp_limb_count,
                    *parameter_shard,
                    scratch.degree);
            }
        }
        else
        {
            kernel::launch_hybrid_modup_decomposition(
                scratch.modup_q.data(),
                scratch.modup_p.data(),
                scratch.c2_intt.data(),
                switch_poly_shard.ptr,
                decomp_index,
                decomp_limb_begin,
                decomp_limb_count,
                *parameter_shard,
                scratch.degree);
        }
    }

    /* 当前dnum块内的Q limb直接复用原始c2的NTT值；新模升产生的Q/P limb在最后一段NTT中直接完成IP乘加。 */
    {
        NvtxRange range("keyswitch.dnum.forward_ntt_qp_mul_accum.c01");
        kernel::launch_hybrid_forward_ntt_qp_mul_accumulate_two_components(
            scratch.accum_q0.data(),
            scratch.accum_p0.data(),
            scratch.accum_q1.data(),
            scratch.accum_p1.data(),
            scratch.modup_q.data(),
            scratch.modup_p.data(),
            switch_poly_shard.ptr,
            key_component0.q_ptr,
            key_component0.p_ptr,
            key_component1.q_ptr,
            key_component1.p_ptr,
            decomp_limb_begin,
            decomp_limb_count,
            *parameter_shard,
            scratch.degree,
            decomp_index == 0,
            fuse_decomp_q,
            fuse_modup_ntt_head);
    }
}

void finalize_hybrid_relinearize(
    GpuCiphertextView &destination,
    HybridScratch &scratch,
    const GpuLevelInfo &level_info,
    const GpuConstRNSPolyView *add_source0,
    const GpuConstRNSPolyView *add_source1)
{
    NvtxRange finalize_range("keyswitch.finalize");
    const auto &destination_shard0 = destination.polys[0].shards.front();
    const auto &destination_shard1 = destination.polys[1].shards.front();
    const GpuConstPolyShardView *add_source_shard0 =
        add_source0 == nullptr ? nullptr : &add_source0->shards.front();
    const GpuConstPolyShardView *add_source_shard1 =
        add_source1 == nullptr ? nullptr : &add_source1->shards.front();
    /* 查找参数表，为后续计算做准备 */
    const auto *parameter_shard = find_parameter_shard(
        level_info,
        as_const_shard(destination_shard0));
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::finalize_hybrid_relinearize: no matching parameter shard");
    }
    validate_hybrid_parameter_shape(
        "GpuKeySwitchHandler::finalize_hybrid_relinearize",
        scratch,
        *parameter_shard);

    auto accum_p0_view = make_scratch_p_view(scratch.accum_p0.data(), scratch);
    auto accum_p1_view = make_scratch_p_view(scratch.accum_p1.data(), scratch);

    if (use_fourstep_all_ntt(
            scratch.degree,
            scratch.base_q_size,
            scratch.base_p_size))
    {
        auto p_coeff0_view = make_scratch_p_view(
            scratch.fourstep_p0.data(),
            scratch);
        auto p_coeff1_view = make_scratch_p_view(
            scratch.fourstep_p1.data(),
            scratch);
        {
            NvtxRange range("keyswitch.finalize.fourstep_intt_p0");
            kernel::launch_inverse_ntt_poly_shard_fourstep_65536(
                p_coeff0_view,
                as_const_shard(accum_p0_view),
                *parameter_shard,
                scratch.degree);
        }
        {
            NvtxRange range("keyswitch.finalize.fourstep_intt_p1");
            kernel::launch_inverse_ntt_poly_shard_fourstep_65536(
                p_coeff1_view,
                as_const_shard(accum_p1_view),
                *parameter_shard,
                scratch.degree);
        }

        if (use_fourstep_finalize_fused())
        {
            NvtxRange range(
                "keyswitch.finalize.convert_p_to_q_fourstep_q01");
            kernel::launch_hybrid_convert_p_to_q_forward_ntt_two_components_fourstep_65536(
                scratch.fourstep_q0.data(),
                scratch.fourstep_q1.data(),
                scratch.fourstep_p0.data(),
                scratch.fourstep_p1.data(),
                *parameter_shard,
                scratch.degree);
        }
        else
        {
            {
                NvtxRange range("keyswitch.finalize.convert_p_to_q.coeff");
                kernel::launch_hybrid_convert_p_to_q(
                    scratch.c2_intt.data(),
                    scratch.modup_q.data(),
                    scratch.fourstep_p0.data(),
                    scratch.fourstep_p1.data(),
                    *parameter_shard,
                    scratch.degree);
            }

            auto q_coeff0_view = make_scratch_q_view(
                scratch.c2_intt.data(),
                scratch);
            auto q_coeff1_view = make_scratch_q_view(
                scratch.modup_q.data(),
                scratch);
            auto q_ntt0_view = make_scratch_q_view(
                scratch.fourstep_q0.data(),
                scratch);
            auto q_ntt1_view = make_scratch_q_view(
                scratch.fourstep_q1.data(),
                scratch);
            {
                NvtxRange range("keyswitch.finalize.fourstep_forward_ntt_q0");
                kernel::launch_forward_ntt_poly_shard_fourstep_65536(
                    q_ntt0_view,
                    as_const_shard(q_coeff0_view),
                    *parameter_shard,
                    scratch.degree);
            }
            {
                NvtxRange range("keyswitch.finalize.fourstep_forward_ntt_q1");
                kernel::launch_forward_ntt_poly_shard_fourstep_65536(
                    q_ntt1_view,
                    as_const_shard(q_coeff1_view),
                    *parameter_shard,
                    scratch.degree);
            }
        }

        {
            NvtxRange range("keyswitch.finalize.fourstep_apply_moddown_add_back");
            kernel::launch_hybrid_apply_moddown_ntt_add_back(
                destination_shard0,
                destination_shard1,
                scratch.accum_q0.data(),
                scratch.accum_q1.data(),
                scratch.fourstep_q0.data(),
                scratch.fourstep_q1.data(),
                *parameter_shard,
                scratch.degree,
                add_source_shard0,
                add_source_shard1);
        }
        return;
    }

    /* 只把P部分转回系数域，Q部分保持在NTT域 */
    const bool p9_fourstep_intt = use_p9_fourstep_p_intt(
        scratch.degree,
        scratch.base_p_size);
    const bool p_source_preweighted =
        use_p9_preweight_p(scratch.degree, scratch.base_p_size);
    GpuWord *p_coeff0 = scratch.accum_p0.data();
    GpuWord *p_coeff1 = scratch.accum_p1.data();
    if (p9_fourstep_intt)
    {
        p_coeff0 = scratch.fourstep_p0.data();
        p_coeff1 = scratch.fourstep_p1.data();
        auto p_coeff0_view = make_scratch_p_view(p_coeff0, scratch);
        auto p_coeff1_view = make_scratch_p_view(p_coeff1, scratch);
        {
            NvtxRange range("keyswitch.finalize.intt_p0.fourstep_p9");
            kernel::launch_inverse_ntt_poly_shard_fourstep_65536(
                p_coeff0_view,
                as_const_shard(accum_p0_view),
                *parameter_shard,
                scratch.degree);
        }
        {
            NvtxRange range("keyswitch.finalize.intt_p1.fourstep_p9");
            kernel::launch_inverse_ntt_poly_shard_fourstep_65536(
                p_coeff1_view,
                as_const_shard(accum_p1_view),
                *parameter_shard,
                scratch.degree);
        }
    }
    else
    {
        {
            NvtxRange range("keyswitch.finalize.intt_p0");
            kernel::launch_inverse_ntt_poly_shard(
                accum_p0_view,
                as_const_shard(accum_p0_view),
                *parameter_shard,
                scratch.degree);
        }
        {
            NvtxRange range("keyswitch.finalize.intt_p1");
            kernel::launch_inverse_ntt_poly_shard(
                accum_p1_view,
                as_const_shard(accum_p1_view),
                *parameter_shard,
                scratch.degree);
        }
    }

    const bool p_to_q_row_tiled8 =
        use_p_to_q_row_tiled8() ||
        use_p9_p_to_q_row_tiled8(
            scratch.degree,
            scratch.base_p_size);
    const bool p_to_q_fourstep = use_p9_p_to_q_fourstep(
        scratch.degree,
        scratch.base_p_size);
    if (p_source_preweighted)
    {
        NvtxRange range("keyswitch.finalize.preweight_p9");
        kernel::launch_hybrid_preweight_p_two_components(
            p_coeff0,
            p_coeff1,
            *parameter_shard,
            scratch.degree);
    }

    if (p_to_q_fourstep)
    {
        NvtxRange range(
            "keyswitch.finalize.convert_p_to_q_forward_ntt.fourstep_p9");
        kernel::launch_hybrid_convert_p9_to_q_forward_ntt_two_components_fourstep_65536(
            scratch.c2_intt.data(),
            scratch.modup_q.data(),
            p_coeff0,
            p_coeff1,
            *parameter_shard,
            scratch.degree,
            p_source_preweighted);
    }
    else if (p_to_q_row_tiled8)
    {
        NvtxRange range(
            "keyswitch.finalize.convert_p_to_q_forward_ntt.row_tiled8");
        kernel::launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8(
            scratch.c2_intt.data(),
            scratch.modup_q.data(),
            p_coeff0,
            p_coeff1,
            *parameter_shard,
            scratch.degree,
            p_source_preweighted);
    }
    else
    {
        NvtxRange range("keyswitch.finalize.convert_p_to_q_forward_ntt");
        kernel::launch_hybrid_convert_p_to_q_forward_ntt(
            scratch.c2_intt.data(),
            scratch.modup_q.data(),
            p_coeff0,
            p_coeff1,
            *parameter_shard,
            scratch.degree,
            p_source_preweighted);
    }

#if 0
    /* Legacy rollback path: separate P->Q conversion and Q forward NTT. */
    {
        auto converted_q0_view =
            make_scratch_q_view(scratch.c2_intt.data(), scratch);
        auto converted_q1_view =
            make_scratch_q_view(scratch.modup_q.data(), scratch);

        {
            NvtxRange range("keyswitch.finalize.convert_p_to_q");
            kernel::launch_hybrid_convert_p_to_q(
                scratch.c2_intt.data(),
                scratch.modup_q.data(),
                scratch.accum_p0.data(),
                scratch.accum_p1.data(),
                *parameter_shard,
                scratch.degree);
        }

#if 1
        /* Batched Q NTT rollback variant. */
        {
            NvtxRange range("keyswitch.finalize.forward_ntt_q_batched");
            kernel::launch_hybrid_forward_ntt_q_two_components(
                scratch.c2_intt.data(),
                scratch.modup_q.data(),
                *parameter_shard,
                scratch.degree);
        }
#else
        /* Separate-component Q NTT rollback variant. */
        {
            {
                NvtxRange range("keyswitch.finalize.forward_ntt_q0");
                kernel::launch_forward_ntt_poly_shard(
                    converted_q0_view,
                    as_const_shard(converted_q0_view),
                    *parameter_shard,
                    scratch.degree);
            }
            {
                NvtxRange range("keyswitch.finalize.forward_ntt_q1");
                kernel::launch_forward_ntt_poly_shard(
                    converted_q1_view,
                    as_const_shard(converted_q1_view),
                    *parameter_shard,
                    scratch.degree);
            }
        }
#endif
    }
#endif

    /* 在NTT域完成模降，并直接和d0/d1累加，避免额外读写accum_q0/accum_q1 */
    {
        NvtxRange range("keyswitch.finalize.apply_moddown_ntt_add_back");
        kernel::launch_hybrid_apply_moddown_ntt_add_back(
            destination_shard0,
            destination_shard1,
            scratch.accum_q0.data(),
            scratch.accum_q1.data(),
            scratch.c2_intt.data(),
            scratch.modup_q.data(),
            *parameter_shard,
            scratch.degree,
            add_source_shard0,
            add_source_shard1);
    }
}

void finalize_hybrid_relinearize_rescale_x2(
    GpuCiphertextView &destination,
    HybridScratch &scratch,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info,
    const GpuConstRNSPolyView *add_source0,
    const GpuConstRNSPolyView *add_source1)
{
    NvtxRange finalize_range("keyswitch.finalize_rescale_x2");
    if (add_source0 == nullptr || add_source1 == nullptr)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::finalize_hybrid_relinearize_rescale_x2: add-source components are required");
    }
    if (scratch.base_q_size < 3 ||
        destination_level_info.q_count + 2 != scratch.base_q_size ||
        destination.meta.q_count != destination_level_info.q_count)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::finalize_hybrid_relinearize_rescale_x2: level mismatch");
    }

    const auto &destination_shard0 = destination.polys[0].shards.front();
    const auto &destination_shard1 = destination.polys[1].shards.front();
    const auto &add_source_full0 = add_source0->shards.front();
    const auto &add_source_full1 = add_source1->shards.front();
    const auto *parameter_shard = find_parameter_shard(
        source_level_info,
        add_source_full0);
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::finalize_hybrid_relinearize_rescale_x2: no matching parameter shard");
    }
    validate_hybrid_parameter_shape(
        "GpuKeySwitchHandler::finalize_hybrid_relinearize_rescale_x2",
        scratch,
        *parameter_shard);
    if (parameter_shard->q_last_two_product == 0 ||
        parameter_shard->inv_q_last_two_product_mod_q.size() <
            destination_level_info.q_count)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::finalize_hybrid_relinearize_rescale_x2: missing rescale_x2 constants");
    }

    ensure_hybrid_relin_rescale_x2_scratch(scratch);

    auto accum_p0_view = make_scratch_p_view(scratch.accum_p0.data(), scratch);
    auto accum_p1_view = make_scratch_p_view(scratch.accum_p1.data(), scratch);

    const GpuWord *converted_q0 = nullptr;
    const GpuWord *converted_q1 = nullptr;
    if (use_fourstep_all_ntt(
            scratch.degree,
            scratch.base_q_size,
            scratch.base_p_size))
    {
        auto p_coeff0_view = make_scratch_p_view(
            scratch.fourstep_p0.data(),
            scratch);
        auto p_coeff1_view = make_scratch_p_view(
            scratch.fourstep_p1.data(),
            scratch);
        {
            NvtxRange range("keyswitch.finalize_rescale_x2.fourstep_intt_p0");
            kernel::launch_inverse_ntt_poly_shard_fourstep_65536(
                p_coeff0_view,
                as_const_shard(accum_p0_view),
                *parameter_shard,
                scratch.degree);
        }
        {
            NvtxRange range("keyswitch.finalize_rescale_x2.fourstep_intt_p1");
            kernel::launch_inverse_ntt_poly_shard_fourstep_65536(
                p_coeff1_view,
                as_const_shard(accum_p1_view),
                *parameter_shard,
                scratch.degree);
        }
        if (use_fourstep_finalize_fused())
        {
            NvtxRange range(
                "keyswitch.finalize_rescale_x2.convert_p_to_q_fourstep_q01");
            kernel::launch_hybrid_convert_p_to_q_forward_ntt_two_components_fourstep_65536(
                scratch.fourstep_q0.data(),
                scratch.fourstep_q1.data(),
                scratch.fourstep_p0.data(),
                scratch.fourstep_p1.data(),
                *parameter_shard,
                scratch.degree);
        }
        else
        {
            {
                NvtxRange range(
                    "keyswitch.finalize_rescale_x2.convert_p_to_q.coeff");
                kernel::launch_hybrid_convert_p_to_q(
                    scratch.c2_intt.data(),
                    scratch.modup_q.data(),
                    scratch.fourstep_p0.data(),
                    scratch.fourstep_p1.data(),
                    *parameter_shard,
                    scratch.degree);
            }
            auto q_coeff0_view = make_scratch_q_view(
                scratch.c2_intt.data(),
                scratch);
            auto q_coeff1_view = make_scratch_q_view(
                scratch.modup_q.data(),
                scratch);
            auto q_ntt0_view = make_scratch_q_view(
                scratch.fourstep_q0.data(),
                scratch);
            auto q_ntt1_view = make_scratch_q_view(
                scratch.fourstep_q1.data(),
                scratch);
            {
                NvtxRange range(
                    "keyswitch.finalize_rescale_x2.fourstep_forward_ntt_q0");
                kernel::launch_forward_ntt_poly_shard_fourstep_65536(
                    q_ntt0_view,
                    as_const_shard(q_coeff0_view),
                    *parameter_shard,
                    scratch.degree);
            }
            {
                NvtxRange range(
                    "keyswitch.finalize_rescale_x2.fourstep_forward_ntt_q1");
                kernel::launch_forward_ntt_poly_shard_fourstep_65536(
                    q_ntt1_view,
                    as_const_shard(q_coeff1_view),
                    *parameter_shard,
                    scratch.degree);
            }
        }
        converted_q0 = scratch.fourstep_q0.data();
        converted_q1 = scratch.fourstep_q1.data();
    }
    else
    {
        {
            NvtxRange range("keyswitch.finalize_rescale_x2.intt_p0");
            kernel::launch_inverse_ntt_poly_shard(
                accum_p0_view,
                as_const_shard(accum_p0_view),
                *parameter_shard,
                scratch.degree);
        }
        {
            NvtxRange range("keyswitch.finalize_rescale_x2.intt_p1");
            kernel::launch_inverse_ntt_poly_shard(
                accum_p1_view,
                as_const_shard(accum_p1_view),
                *parameter_shard,
                scratch.degree);
        }
        if (use_p_to_q_row_tiled8())
        {
            NvtxRange range(
                "keyswitch.finalize_rescale_x2.convert_p_to_q_forward_ntt.row_tiled8");
            kernel::launch_hybrid_convert_p_to_q_forward_ntt_row_tiled8(
                scratch.c2_intt.data(),
                scratch.modup_q.data(),
                scratch.accum_p0.data(),
                scratch.accum_p1.data(),
                *parameter_shard,
                scratch.degree);
        }
        else
        {
            NvtxRange range(
                "keyswitch.finalize_rescale_x2.convert_p_to_q_forward_ntt");
            kernel::launch_hybrid_convert_p_to_q_forward_ntt(
                scratch.c2_intt.data(),
                scratch.modup_q.data(),
                scratch.accum_p0.data(),
                scratch.accum_p1.data(),
                *parameter_shard,
                scratch.degree);
        }
        converted_q0 = scratch.c2_intt.data();
        converted_q1 = scratch.modup_q.data();
    }

    GpuPolyShardView dropped_ntt0{
        scratch.device_id,
        scratch.relin_rescale_dropped_ntt.data(),
        scratch.base_q_size - 2,
        2,
        0,
        scratch.degree};
    GpuPolyShardView dropped_ntt1{
        scratch.device_id,
        scratch.relin_rescale_dropped_ntt.data() + 2 * scratch.degree,
        scratch.base_q_size - 2,
        2,
        0,
        scratch.degree};
    GpuConstPolyShardView add_tail0{
        add_source_full0.device_id,
        add_source_full0.ptr + (scratch.base_q_size - 2) * scratch.degree,
        scratch.base_q_size - 2,
        2,
        0,
        scratch.degree};
    GpuConstPolyShardView add_tail1{
        add_source_full1.device_id,
        add_source_full1.ptr + (scratch.base_q_size - 2) * scratch.degree,
        scratch.base_q_size - 2,
        2,
        0,
        scratch.degree};

    {
        NvtxRange range("keyswitch.finalize_rescale_x2.build_dropped_ntt");
        kernel::launch_hybrid_apply_moddown_ntt_add_back(
            dropped_ntt0,
            dropped_ntt1,
            scratch.accum_q0.data(),
            scratch.accum_q1.data(),
            converted_q0,
            converted_q1,
            *parameter_shard,
            scratch.degree,
            &add_tail0,
            &add_tail1);
    }

    GpuPolyShardView dropped_coeff0{
        scratch.device_id,
        scratch.relin_rescale_dropped_coeff.data(),
        scratch.base_q_size - 2,
        2,
        0,
        scratch.degree};
    GpuPolyShardView dropped_coeff1{
        scratch.device_id,
        scratch.relin_rescale_dropped_coeff.data() + 2 * scratch.degree,
        scratch.base_q_size - 2,
        2,
        0,
        scratch.degree};
    {
        NvtxRange range("keyswitch.finalize_rescale_x2.intt_dropped0");
        kernel::launch_inverse_ntt_poly_shard(
            dropped_coeff0,
            as_const_shard(dropped_ntt0),
            *parameter_shard,
            scratch.degree);
    }
    {
        NvtxRange range("keyswitch.finalize_rescale_x2.intt_dropped1");
        kernel::launch_inverse_ntt_poly_shard(
            dropped_coeff1,
            as_const_shard(dropped_ntt1),
            *parameter_shard,
            scratch.degree);
    }

    const std::size_t destination_q_count = destination_level_info.q_count;
    {
        NvtxRange range("keyswitch.finalize_rescale_x2.build_correction");
        kernel::launch_build_q_last_two_rescale_correction_batch_fused(
            scratch.relin_rescale_correction.data(),
            scratch.relin_rescale_dropped_coeff.data(),
            2,
            destination_q_count,
            *parameter_shard,
            scratch.degree);
    }

    GpuPolyShardView correction_ntt0{
        scratch.device_id,
        scratch.relin_rescale_correction_ntt.data(),
        0,
        destination_q_count,
        0,
        scratch.degree};
    GpuPolyShardView correction_ntt1{
        scratch.device_id,
        scratch.relin_rescale_correction_ntt.data() +
            destination_q_count * scratch.degree,
        0,
        destination_q_count,
        0,
        scratch.degree};
    GpuConstPolyShardView correction0{
        scratch.device_id,
        scratch.relin_rescale_correction.data(),
        0,
        destination_q_count,
        0,
        scratch.degree};
    GpuConstPolyShardView correction1{
        scratch.device_id,
        scratch.relin_rescale_correction.data() +
            destination_q_count * scratch.degree,
        0,
        destination_q_count,
        0,
        scratch.degree};
    {
        NvtxRange range("keyswitch.finalize_rescale_x2.ntt_correction0");
        kernel::launch_forward_ntt_poly_shard(
            correction_ntt0,
            correction0,
            *parameter_shard,
            scratch.degree);
    }
    {
        NvtxRange range("keyswitch.finalize_rescale_x2.ntt_correction1");
        kernel::launch_forward_ntt_poly_shard(
            correction_ntt1,
            correction1,
            *parameter_shard,
            scratch.degree);
    }

    {
        NvtxRange range("keyswitch.finalize_rescale_x2.apply_retained");
        kernel::launch_hybrid_apply_moddown_ntt_add_back_rescale_x2(
            destination_shard0,
            destination_shard1,
            as_const_shard(correction_ntt0),
            as_const_shard(correction_ntt1),
            scratch.accum_q0.data(),
            scratch.accum_q1.data(),
            converted_q0,
            converted_q1,
            *parameter_shard,
            scratch.degree,
            &add_source_full0,
            &add_source_full1);
    }
}

void validate_single_full_shard(
    const char *name,
    const GpuConstRNSPolyView &poly,
    std::size_t degree,
    std::size_t limb_count)
{
    if (poly.shards.size() != 1)
    {
        throw std::invalid_argument(std::string(name) + ": expected one shard");
    }

    const auto &shard = poly.shards.front();
    if (shard.limb_begin != 0 ||
        shard.limb_count != limb_count ||
        shard.coeff_begin != 0 ||
        shard.coeff_count != degree)
    {
        throw std::invalid_argument(
            std::string(name) + ": expected a full coefficient, full limb shard");
    }
}

void validate_single_full_shard(
    const char *name,
    const GpuRNSPolyView &poly,
    std::size_t degree,
    std::size_t limb_count)
{
    if (poly.shards.size() != 1)
    {
        throw std::invalid_argument(std::string(name) + ": expected one shard");
    }

    const auto &shard = poly.shards.front();
    if (shard.limb_begin != 0 ||
        shard.limb_count != limb_count ||
        shard.coeff_begin != 0 ||
        shard.coeff_count != degree)
    {
        throw std::invalid_argument(
            std::string(name) + ": expected a full coefficient, full limb shard");
    }
}

void validate_hybrid_relinearize_shape(
    const GpuCiphertextView &destination,
    const GpuConstCiphertextView &source,
    const GpuConstEvaluationKeyView &relin_keys,
    const GpuEvaluationKeyData &relin_key_data,
    const GpuLevelInfo &level_info)
{
    if (!(destination.meta.parms_id == source.meta.parms_id) ||
        !(source.meta.parms_id == level_info.parms_id))
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::relinearize_hybrid_ciphertext: parms_id mismatch");
    }
    if (!source.meta.is_ntt_form || !destination.meta.is_ntt_form)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::relinearize_hybrid_ciphertext: CKKS inputs must be in NTT form");
    }
    if (source.polys.size() != 3 || source.meta.component_count != 3)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::relinearize_hybrid_ciphertext: source must have three components");
    }
    if (destination.polys.size() != 2 || destination.meta.component_count != 2)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::relinearize_hybrid_ciphertext: destination must have two components");
    }
    if (source.meta.degree != destination.meta.degree ||
        source.meta.degree != level_info.degree ||
        source.meta.q_count != destination.meta.q_count ||
        source.meta.q_count != level_info.q_count ||
        source.meta.p_count != 0 ||
        destination.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::relinearize_hybrid_ciphertext: shape mismatch");
    }
    if (!(relin_keys.meta.key_parms_id == relin_key_data.meta.key_parms_id) ||
        relin_keys.storage_q_count != relin_key_data.meta.q_count ||
        relin_keys.meta.q_count != source.meta.q_count ||
        relin_keys.meta.key_count == 0 ||
        relin_keys.meta.decomposition_count == 0 ||
        relin_keys.meta.component_count < kSwitchKeyComponentCount)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::relinearize_hybrid_ciphertext: invalid relin key metadata");
    }
    if (relin_key_data.poly_metadata_.empty() ||
        relin_key_data.polys_.size() != relin_keys.polys.size())
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::relinearize_hybrid_ciphertext: invalid relin key layout");
    }

    for (std::size_t component = 0; component < source.polys.size(); ++component)
    {
        validate_single_full_shard(
            "GpuKeySwitchHandler source",
            source.polys[component],
            source.meta.degree,
            source.meta.q_count);
        if (!same_shard_placement(
                source.polys[0].shards.front(),
                source.polys[component].shards.front()))
        {
            throw std::invalid_argument(
                "GpuKeySwitchHandler::relinearize_hybrid_ciphertext: source shard placement mismatch");
        }
    }

    for (std::size_t component = 0; component < destination.polys.size(); ++component)
    {
        validate_single_full_shard(
            "GpuKeySwitchHandler destination",
            destination.polys[component],
            destination.meta.degree,
            destination.meta.q_count);
        if (!same_shard_placement(
                destination.polys[component].shards.front(),
                source.polys[component].shards.front()))
        {
            throw std::invalid_argument(
                "GpuKeySwitchHandler::relinearize_hybrid_ciphertext: destination/source shard placement mismatch");
        }
    }
}

void validate_hybrid_switch_key_shape(
    const GpuCiphertextView &destination,
    const GpuConstRNSPolyView &switch_poly_ntt,
    const GpuConstEvaluationKeyView &switch_keys,
    const GpuEvaluationKeyData &switch_key_data,
    std::size_t key_index,
    const GpuLevelInfo &level_info)
{
    if (!(destination.meta.parms_id == level_info.parms_id))
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: parms_id mismatch");
    }
    if (!destination.meta.is_ntt_form)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: destination must be in NTT form");
    }
    if (destination.polys.size() != 2 || destination.meta.component_count != 2)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: destination must have two components");
    }
    if (destination.meta.degree != level_info.degree ||
        destination.meta.q_count != level_info.q_count ||
        destination.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: destination shape mismatch");
    }
    if (switch_poly_ntt.shards.size() != 1)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: switch poly must have one shard");
    }
    validate_single_full_shard(
        "GpuKeySwitchHandler switch poly",
        switch_poly_ntt,
        destination.meta.degree,
        destination.meta.q_count);

    for (std::size_t component = 0; component < destination.polys.size(); ++component)
    {
        validate_single_full_shard(
            "GpuKeySwitchHandler switch destination",
            destination.polys[component],
            destination.meta.degree,
            destination.meta.q_count);
        if (!same_shard_placement(
                destination.polys[component].shards.front(),
                switch_poly_ntt.shards.front()))
        {
            throw std::invalid_argument(
                "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: destination/switch-poly shard placement mismatch");
        }
    }

    if (!(switch_keys.meta.key_parms_id == switch_key_data.meta.key_parms_id) ||
        switch_keys.storage_q_count != switch_key_data.meta.q_count ||
        switch_keys.meta.q_count != destination.meta.q_count ||
        switch_keys.meta.key_count <= key_index ||
        switch_keys.meta.decomposition_count == 0 ||
        switch_keys.meta.component_count < kSwitchKeyComponentCount)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: invalid key metadata");
    }
    if (switch_key_data.poly_metadata_.empty() ||
        switch_key_data.polys_.size() != switch_keys.polys.size())
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: invalid key layout");
    }
}

}  // namespace

struct GpuKeySwitchHandler::PersistentWorkspace
{
    HybridScratch relinearize;
};

GpuKeySwitchHandler::GpuKeySwitchHandler(const GpuParameterData &params)
    : params_(params),
      persistent_workspace_(new PersistentWorkspace())
{}

GpuKeySwitchHandler::~GpuKeySwitchHandler() = default;

void GpuKeySwitchHandler::hoist_decompose_modup_ntt(
    const GpuConstRNSPolyView &switch_poly_ntt,
    const GpuLevelInfo &level_info,
    GpuHoistedDecomposition &destination,
    GpuHybridKeySwitchWorkspace &workspace) const
{
    NvtxRange range("double_hoist.decompose");
    validate_single_full_shard(
        "GpuKeySwitchHandler::hoist_decompose_modup_ntt",
        switch_poly_ntt,
        level_info.degree,
        level_info.q_count);
    const auto &source_shard = switch_poly_ntt.shards.front();
    const auto *parameter_shard =
        find_parameter_shard(level_info, source_shard);
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::hoist_decompose_modup_ntt: no matching parameter shard");
    }

    const std::size_t q_count = level_info.q_count;
    const std::size_t p_count = parameter_shard->hybrid_base_p_count;
    if (p_count == 0 ||
        parameter_shard->hybrid_base_q_count != q_count)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::hoist_decompose_modup_ntt: invalid HYBRID base");
    }
    const std::size_t dnum = (q_count + p_count - 1) / p_count;
    const std::size_t q_words = q_count * level_info.degree;
    const std::size_t p_words = p_count * level_info.degree;

    workspace.ensure_capacity(
        source_shard.device_id,
        level_info.degree,
        q_count,
        p_count);
    const bool capacity_insufficient =
        destination.device_id != source_shard.device_id ||
        destination.degree != level_info.degree ||
        destination.source_intt_q.size() < q_words ||
        destination.digits_q_ntt.size() < dnum * q_words ||
        destination.digits_p_ntt.size() < dnum * p_words;
    if (capacity_insufficient)
    {
        destination.source_intt_q.allocate(
            q_words,
            source_shard.device_id);
        destination.digits_q_ntt.allocate(
            dnum * q_words,
            source_shard.device_id);
        destination.digits_p_ntt.allocate(
            dnum * p_words,
            source_shard.device_id);
    }
    destination.parms_id = level_info.parms_id;
    destination.device_id = source_shard.device_id;
    destination.degree = level_info.degree;
    destination.q_count = q_count;
    destination.p_count = p_count;
    destination.dnum = dnum;

    GpuPolyShardView intt_q;
    intt_q.device_id = source_shard.device_id;
    intt_q.ptr = destination.source_intt_q.data();
    intt_q.limb_begin = 0;
    intt_q.limb_count = q_count;
    intt_q.coeff_begin = 0;
    intt_q.coeff_count = level_info.degree;
    kernel::launch_inverse_ntt_poly_shard(
        intt_q,
        source_shard,
        *parameter_shard,
        level_info.degree);

    GpuConstPolyShardView intt_q_const;
    intt_q_const.device_id = intt_q.device_id;
    intt_q_const.ptr = intt_q.ptr;
    intt_q_const.limb_begin = intt_q.limb_begin;
    intt_q_const.limb_count = intt_q.limb_count;
    intt_q_const.coeff_begin = intt_q.coeff_begin;
    intt_q_const.coeff_count = intt_q.coeff_count;

    for (std::size_t digit = 0; digit < dnum; ++digit)
    {
        const std::size_t limb_begin = digit * p_count;
        const std::size_t limb_count =
            std::min(p_count, q_count - limb_begin);
        GpuWord *digit_q =
            destination.digits_q_ntt.data() + digit * q_words;
        GpuWord *digit_p =
            destination.digits_p_ntt.data() + digit * p_words;
        const bool qp_fourstep = use_double_hoist_p9_qp_fourstep(
            level_info.degree,
            p_count);
        const bool fuse_modup_phase1 = qp_fourstep &&
            use_p9_modup_fourstep_phase1_fused(
                level_info.degree,
                p_count);
        GpuWord *modup_q = qp_fourstep && !fuse_modup_phase1
            ? workspace.permuted_digit_q.data()
            : digit_q;
        GpuWord *modup_p = qp_fourstep && !fuse_modup_phase1
            ? workspace.permuted_digit_p.data()
            : digit_p;

        if (fuse_modup_phase1)
        {
            NvtxRange modup_range(
                "double_hoist.decompose.modup_fourstep_phase1.fused_p9");
            kernel::launch_hybrid_modup_p9_forward_ntt_qp_active_phase1_fourstep_65536(
                digit_q,
                digit_p,
                destination.source_intt_q.data(),
                digit,
                limb_begin,
                limb_count,
                *parameter_shard,
                level_info.degree);
        }
        else if (use_double_hoist_p9_modup_row_tiled8(
                level_info.degree,
                p_count))
        {
            NvtxRange modup_range(
                "double_hoist.decompose.modup.p9.row_tiled8");
            kernel::launch_hybrid_modup_decomposition_row_tiled8(
                modup_q,
                modup_p,
                destination.source_intt_q.data(),
                digit,
                limb_begin,
                limb_count,
                *parameter_shard,
                level_info.degree);
        }
        else
        {
            NvtxRange modup_range(
                "double_hoist.decompose.modup.generic");
            kernel::launch_hybrid_modup_decomposition(
                modup_q,
                modup_p,
                destination.source_intt_q.data(),
                source_shard.ptr,
                digit,
                limb_begin,
                limb_count,
                *parameter_shard,
                level_info.degree);
        }

        /*
         * launch_hybrid_modup_decomposition deliberately skips the Q limbs
         * which belong to this decomposition block: the monolithic
         * KeySwitch path reads those values directly from switch_poly_ntt.
         * A persistent hoisted digit must instead be self-contained because
         * later Galois permutations and pre-rotated KeyMult consume the full
         * [Q|P] digit. Fill the skipped, already-NTT Q interval from the
         * original switch polynomial before transforming the active limbs.
         *
         * Across all digits these contiguous copies cover Q exactly once.
         */
        const std::size_t copied_q_words =
            limb_count * level_info.degree;
        gpu_check_cuda(
            cudaMemcpyAsync(
                digit_q + limb_begin * level_info.degree,
                source_shard.ptr + limb_begin * level_info.degree,
                copied_q_words * sizeof(GpuWord),
                cudaMemcpyDeviceToDevice,
                nullptr),
            "double-hoist copy decomposition Q limbs");
        if (qp_fourstep)
        {
            NvtxRange ntt_range(
                "double_hoist.decompose.qp_forward_ntt.fourstep_p9");
            kernel::launch_forward_ntt_qp_active_fourstep_65536(
                digit_q,
                digit_p,
                modup_q,
                modup_p,
                limb_begin,
                limb_count,
                *parameter_shard,
                level_info.degree,
                fuse_modup_phase1);
        }
        else
        {
            NvtxRange ntt_range(
                "double_hoist.decompose.qp_forward_ntt.generic");
            kernel::launch_hybrid_forward_ntt_qp_active(
                digit_q,
                digit_p,
                limb_begin,
                limb_count,
                *parameter_shard,
                level_info.degree);
        }
    }
}

void GpuKeySwitchHandler::keyswitch_multsum_no_moddown(
    const GpuHoistedDecomposition &hoisted,
    std::uint32_t galois_elt,
    const GpuConstEvaluationKeyView &keys,
    const GpuEvaluationKeyData &key_storage,
    std::size_t key_index,
    GpuQPCiphertextBuffer &destination,
    std::size_t destination_batch,
    bool initialize,
    const GpuLevelInfo &level_info,
    GpuHybridKeySwitchWorkspace &workspace) const
{
    NvtxRange range("double_hoist.keymult_no_moddown");
    if (!(hoisted.parms_id == level_info.parms_id) ||
        hoisted.degree != level_info.degree ||
        hoisted.q_count != level_info.q_count ||
        hoisted.p_count == 0 || hoisted.dnum == 0)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::keyswitch_multsum_no_moddown: hoisted shape mismatch");
    }
    if (destination_batch >= destination.batch_count ||
        destination.degree != hoisted.degree ||
        destination.q_count != hoisted.q_count ||
        destination.p_count != hoisted.p_count ||
        destination.device_id != hoisted.device_id)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::keyswitch_multsum_no_moddown: destination shape mismatch");
    }
    if (keys.meta.q_count != hoisted.q_count ||
        keys.meta.p_count != hoisted.p_count ||
        keys.storage_q_count != key_storage.meta.q_count ||
        key_index >= keys.meta.key_count)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::keyswitch_multsum_no_moddown: key metadata mismatch");
    }

    GpuConstPolyShardView level_probe;
    level_probe.device_id = hoisted.device_id;
    level_probe.ptr = hoisted.digits_q_ntt.data();
    level_probe.limb_begin = 0;
    level_probe.limb_count = hoisted.q_count;
    level_probe.coeff_begin = 0;
    level_probe.coeff_count = hoisted.degree;
    const auto *parameter_shard =
        find_parameter_shard(level_info, level_probe);
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::keyswitch_multsum_no_moddown: no matching parameter shard");
    }
    workspace.ensure_capacity(
        hoisted.device_id,
        hoisted.degree,
        hoisted.q_count,
        hoisted.p_count);

    const std::size_t q_words = hoisted.q_count * hoisted.degree;
    const std::size_t p_words = hoisted.p_count * hoisted.degree;
    const bool inverse_pre_rotated =
        keys.meta.galois_format ==
        GpuGaloisKeyFormat::InversePreRotated;
    if (inverse_pre_rotated &&
        (key_storage.galois_elts_by_key_index.size() <= key_index ||
         key_storage.galois_elts_by_key_index[key_index] != galois_elt))
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::keyswitch_multsum_no_moddown: "
            "pre-rotated key/Galois-element mismatch");
    }
    if (inverse_pre_rotated)
    {
        const std::size_t pointer_offset =
            key_index * key_storage.meta.decomposition_count;
        const std::size_t pointer_end =
            pointer_offset + hoisted.dnum;
        if (key_storage.galois_key_q0_ptrs.size() < pointer_end ||
            key_storage.galois_key_p0_ptrs.size() < pointer_end ||
            key_storage.galois_key_q1_ptrs.size() < pointer_end ||
            key_storage.galois_key_p1_ptrs.size() < pointer_end)
        {
            throw std::invalid_argument(
                "GpuKeySwitchHandler::keyswitch_multsum_no_moddown: "
                "pre-rotated key pointer tables are incomplete");
        }
        kernel::launch_double_hoist_pre_rotated_keymul_batch(
            destination.q_component(destination_batch, 0),
            destination.p_component(destination_batch, 0),
            destination.q_component(destination_batch, 1),
            destination.p_component(destination_batch, 1),
            hoisted.digits_q_ntt.data(),
            hoisted.digits_p_ntt.data(),
            key_storage.galois_key_q0_ptrs.data() + pointer_offset,
            key_storage.galois_key_p0_ptrs.data() + pointer_offset,
            key_storage.galois_key_q1_ptrs.data() + pointer_offset,
            key_storage.galois_key_p1_ptrs.data() + pointer_offset,
            hoisted.dnum,
            galois_elt,
            initialize,
            *parameter_shard,
            hoisted.degree);
        return;
    }
    for (std::size_t digit = 0; digit < hoisted.dnum; ++digit)
    {
        const GpuWord *digit_q =
            hoisted.digits_q_ntt.data() + digit * q_words;
        const GpuWord *digit_p =
            hoisted.digits_p_ntt.data() + digit * p_words;

        const GpuWord *active_digit_q = digit_q;
        const GpuWord *active_digit_p = digit_p;
        if (!inverse_pre_rotated)
        {
            GpuPolyShardView permuted_q{
                hoisted.device_id,
                workspace.permuted_digit_q.data(),
                0,
                hoisted.q_count,
                0,
                hoisted.degree};
            GpuConstPolyShardView digit_q_view{
                hoisted.device_id,
                digit_q,
                0,
                hoisted.q_count,
                0,
                hoisted.degree};
            kernel::launch_apply_galois_ntt_poly_shard(
                permuted_q,
                digit_q_view,
                galois_elt,
                hoisted.degree);

            GpuPolyShardView permuted_p{
                hoisted.device_id,
                workspace.permuted_digit_p.data(),
                hoisted.q_count,
                hoisted.p_count,
                0,
                hoisted.degree};
            GpuConstPolyShardView digit_p_view{
                hoisted.device_id,
                digit_p,
                hoisted.q_count,
                hoisted.p_count,
                0,
                hoisted.degree};
            kernel::launch_apply_galois_ntt_poly_shard(
                permuted_p,
                digit_p_view,
                galois_elt,
                hoisted.degree);
            active_digit_q = workspace.permuted_digit_q.data();
            active_digit_p = workspace.permuted_digit_p.data();
        }

        const auto &key0_poly = find_key_poly(
            keys,
            key_storage,
            key_index,
            digit,
            0);
        const auto &key1_poly = find_key_poly(
            keys,
            key_storage,
            key_index,
            digit,
            1);
        const auto key0 =
            make_hybrid_key_component_view(key0_poly, keys, key_storage);
        const auto key1 =
            make_hybrid_key_component_view(key1_poly, keys, key_storage);

        const std::size_t decomp_limb_begin = digit * hoisted.p_count;
        const std::size_t decomp_limb_count = std::min(
            hoisted.p_count,
            hoisted.q_count - decomp_limb_begin);
        if (inverse_pre_rotated)
        {
            kernel::launch_double_hoist_pre_rotated_keymul_digit(
                destination.q_component(destination_batch, 0),
                destination.p_component(destination_batch, 0),
                destination.q_component(destination_batch, 1),
                destination.p_component(destination_batch, 1),
                active_digit_q,
                active_digit_p,
                key0.q_ptr,
                key0.p_ptr,
                key1.q_ptr,
                key1.p_ptr,
                galois_elt,
                initialize && digit == 0,
                *parameter_shard,
                hoisted.degree);
        }
        else
        {
            kernel::launch_hybrid_multiply_accumulate_two_components(
                destination.q_component(destination_batch, 0),
                destination.p_component(destination_batch, 0),
                destination.q_component(destination_batch, 1),
                destination.p_component(destination_batch, 1),
                active_digit_q,
                active_digit_p,
                active_digit_q,
                key0.q_ptr,
                key0.p_ptr,
                key1.q_ptr,
                key1.p_ptr,
                *parameter_shard,
                hoisted.degree,
                decomp_limb_begin,
                decomp_limb_count,
                initialize && digit == 0);
        }
    }
}

void GpuKeySwitchHandler::moddown_qp_ciphertext_to_q(
    GpuQPCiphertextBuffer &source,
    std::size_t source_batch,
    GpuCiphertextData &destination,
    const GpuCiphertextMeta &destination_meta,
    const GpuLevelInfo &level_info,
    GpuHybridKeySwitchWorkspace &workspace) const
{
    NvtxRange range("double_hoist.moddown");
    if (source_batch >= source.batch_count ||
        source.degree != level_info.degree ||
        source.q_count != level_info.q_count ||
        source.p_count == 0)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::moddown_qp_ciphertext_to_q: source shape mismatch");
    }
    GpuConstPolyShardView level_probe{
        source.device_id,
        source.q_component(source_batch, 0),
        0,
        source.q_count,
        0,
        source.degree};
    const auto *parameter_shard =
        find_parameter_shard(level_info, level_probe);
    if (parameter_shard == nullptr ||
        parameter_shard->hybrid_base_p_count != source.p_count)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::moddown_qp_ciphertext_to_q: no matching parameter shard");
    }
    workspace.ensure_capacity(
        source.device_id,
        source.degree,
        source.q_count,
        source.p_count);

    const bool reusable =
        !destination.empty() &&
        destination.meta.degree == source.degree &&
        destination.meta.q_count == source.q_count &&
        destination.meta.p_count == 0 &&
        destination.meta.component_count == 2 &&
        destination.fields_.size() == 1 &&
        destination.fields_.front().device_id == source.device_id;
    if (!reusable)
    {
        destination = GpuCiphertextData::allocate_single_device(
            source.degree,
            source.q_count,
            2,
            source.device_id);
    }
    destination.meta = destination_meta;
    destination.meta.degree = source.degree;
    destination.meta.q_count = source.q_count;
    destination.meta.p_count = 0;
    destination.meta.component_count = 2;
    destination.meta.is_ntt_form = true;

    auto destination_view = destination.make_view();

    GpuPolyShardView p_coeff0{
        source.device_id,
        workspace.p_coeff0.data(),
        source.q_count,
        source.p_count,
        0,
        source.degree};
    GpuPolyShardView p_coeff1{
        source.device_id,
        workspace.p_coeff1.data(),
        source.q_count,
        source.p_count,
        0,
        source.degree};
    GpuConstPolyShardView p_ntt0{
        source.device_id,
        source.p_component(source_batch, 0),
        source.q_count,
        source.p_count,
        0,
        source.degree};
    GpuConstPolyShardView p_ntt1{
        source.device_id,
        source.p_component(source_batch, 1),
        source.q_count,
        source.p_count,
        0,
        source.degree};
    kernel::launch_inverse_ntt_poly_shard(
        p_coeff0,
        p_ntt0,
        *parameter_shard,
        source.degree);
    kernel::launch_inverse_ntt_poly_shard(
        p_coeff1,
        p_ntt1,
        *parameter_shard,
        source.degree);
    launch_double_hoist_p_to_q_forward_ntt(
        workspace.converted_q0.data(),
        workspace.converted_q1.data(),
        workspace.p_coeff0.data(),
        workspace.p_coeff1.data(),
        *parameter_shard,
        source.degree);
    kernel::launch_hybrid_apply_moddown_ntt_out_of_place(
        destination_view.polys[0].shards.front().ptr,
        destination_view.polys[1].shards.front().ptr,
        source.q_component(source_batch, 0),
        source.q_component(source_batch, 1),
        workspace.converted_q0.data(),
        workspace.converted_q1.data(),
        *parameter_shard,
        source.degree);
}

void GpuKeySwitchHandler::moddown_qp_ciphertext_batch_to_q(
    GpuQPCiphertextBuffer &source,
    std::size_t batch_count,
    GpuQCiphertextBatchBuffer &destination,
    const GpuLevelInfo &level_info,
    DeviceVector<GpuWord> &p_coeff,
    DeviceVector<GpuWord> &converted_q) const
{
    NvtxRange range("double_hoist.moddown_batch");
    if (batch_count == 0 || batch_count > source.batch_count ||
        source.degree != level_info.degree ||
        source.q_count != level_info.q_count || source.p_count == 0)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::moddown_qp_ciphertext_batch_to_q: "
            "source shape mismatch");
    }

    GpuConstPolyShardView level_probe{
        source.device_id,
        source.q_component(0, 0),
        0,
        source.q_count,
        0,
        source.degree};
    const auto *parameter_shard =
        find_parameter_shard(level_info, level_probe);
    if (parameter_shard == nullptr ||
        parameter_shard->hybrid_base_p_count != source.p_count)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::moddown_qp_ciphertext_batch_to_q: "
            "no matching parameter shard");
    }

    destination.ensure_capacity(
        source.device_id,
        source.degree,
        source.q_count,
        batch_count);
    const std::size_t component_count = 2 * batch_count;
    const std::size_t p_component_words =
        source.p_count * source.degree;
    const std::size_t q_component_words =
        source.q_count * source.degree;
    const std::size_t p_words = component_count * p_component_words;
    const std::size_t q_words = component_count * q_component_words;
    if (p_coeff.size() < p_words ||
        p_coeff.device_id() != source.device_id)
    {
        p_coeff.allocate(p_words, source.device_id);
    }
    if (converted_q.size() < q_words ||
        converted_q.device_id() != source.device_id)
    {
        converted_q.allocate(q_words, source.device_id);
    }

    /*
     * Keep the parameter-selected NTT implementation. The tensor component
     * launcher requires setup-time TAM matrices which are intentionally not
     * mandatory for bootstrap parameters. We still stage all P INTTs before
     * the fused conversion/forward-NTT phase, instead of serializing a
     * complete ModDown pipeline per group.
     */
    for (std::size_t component = 0;
         component < component_count;
         ++component)
    {
        const GpuPolyShardView p_coeff_component{
            source.device_id,
            p_coeff.data() + component * p_component_words,
            source.q_count,
            source.p_count,
            0,
            source.degree};
        const GpuConstPolyShardView p_ntt_component{
            source.device_id,
            source.p.data() + component * p_component_words,
            source.q_count,
            source.p_count,
            0,
            source.degree};
        kernel::launch_inverse_ntt_poly_shard(
            p_coeff_component,
            p_ntt_component,
            *parameter_shard,
            source.degree);
    }

    /*
     * Preserve the existing two-component fusion: it combines P->Q basis
     * conversion with the first forward-NTT stage. Splitting that kernel just
     * to expose a larger batch increases global-memory traffic at N=16384.
     * The groups are still phase-scheduled (all INTTs first, then all fused
     * convert/NTTs, then one out-of-place apply).
     */
    for (std::size_t batch = 0; batch < batch_count; ++batch)
    {
        launch_double_hoist_p_to_q_forward_ntt(
            converted_q.data() +
                (batch * 2) * q_component_words,
            converted_q.data() +
                (batch * 2 + 1) * q_component_words,
            p_coeff.data() +
                (batch * 2) * p_component_words,
            p_coeff.data() +
                (batch * 2 + 1) * p_component_words,
            *parameter_shard,
            source.degree);
    }

    kernel::launch_hybrid_apply_moddown_ntt_out_of_place_batch(
        destination.q.data(),
        source.q.data(),
        converted_q.data(),
        batch_count,
        *parameter_shard,
        source.degree);
}

void GpuKeySwitchHandler::moddown_qp_groups_to_q(
    GpuQPCiphertextBuffer &source_groups,
    std::size_t group_count,
    DeviceVector<GpuWord> &reduced_p_ntt,
    GpuCiphertextData &destination,
    const GpuCiphertextMeta &destination_meta,
    const GpuLevelInfo &level_info,
    GpuHybridKeySwitchWorkspace &workspace) const
{
    NvtxRange range("double_hoist.moddown_grouped_outer");
    if (group_count == 0 || group_count > source_groups.batch_count ||
        source_groups.degree != level_info.degree ||
        source_groups.q_count != level_info.q_count ||
        source_groups.p_count == 0)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::moddown_qp_groups_to_q: "
            "source shape mismatch");
    }
    GpuConstPolyShardView level_probe{
        source_groups.device_id,
        source_groups.q_component(0, 0),
        0,
        source_groups.q_count,
        0,
        source_groups.degree};
    const auto *parameter_shard =
        find_parameter_shard(level_info, level_probe);
    if (parameter_shard == nullptr ||
        parameter_shard->hybrid_base_p_count != source_groups.p_count)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::moddown_qp_groups_to_q: "
            "no matching parameter shard");
    }
    workspace.ensure_capacity(
        source_groups.device_id,
        source_groups.degree,
        source_groups.q_count,
        source_groups.p_count);

    const std::size_t p_component_words =
        source_groups.p_count * source_groups.degree;
    const std::size_t required_p_words = 2 * p_component_words;
    if (reduced_p_ntt.size() < required_p_words ||
        reduced_p_ntt.device_id() != source_groups.device_id)
    {
        reduced_p_ntt.allocate(
            required_p_words,
            source_groups.device_id);
    }
    kernel::launch_double_hoist_reduce_p_groups(
        reduced_p_ntt.data(),
        reduced_p_ntt.data() + p_component_words,
        source_groups.p.data(),
        group_count,
        *parameter_shard,
        source_groups.degree);

    const GpuPolyShardView p_coeff0{
        source_groups.device_id,
        workspace.p_coeff0.data(),
        source_groups.q_count,
        source_groups.p_count,
        0,
        source_groups.degree};
    const GpuPolyShardView p_coeff1{
        source_groups.device_id,
        workspace.p_coeff1.data(),
        source_groups.q_count,
        source_groups.p_count,
        0,
        source_groups.degree};
    const GpuConstPolyShardView p_ntt0{
        source_groups.device_id,
        reduced_p_ntt.data(),
        source_groups.q_count,
        source_groups.p_count,
        0,
        source_groups.degree};
    const GpuConstPolyShardView p_ntt1{
        source_groups.device_id,
        reduced_p_ntt.data() + p_component_words,
        source_groups.q_count,
        source_groups.p_count,
        0,
        source_groups.degree};
    kernel::launch_inverse_ntt_poly_shard(
        p_coeff0,
        p_ntt0,
        *parameter_shard,
        source_groups.degree);
    kernel::launch_inverse_ntt_poly_shard(
        p_coeff1,
        p_ntt1,
        *parameter_shard,
        source_groups.degree);
    launch_double_hoist_p_to_q_forward_ntt(
        workspace.converted_q0.data(),
        workspace.converted_q1.data(),
        workspace.p_coeff0.data(),
        workspace.p_coeff1.data(),
        *parameter_shard,
        source_groups.degree);

    const bool reusable =
        !destination.empty() &&
        destination.meta.degree == source_groups.degree &&
        destination.meta.q_count == source_groups.q_count &&
        destination.meta.p_count == 0 &&
        destination.meta.component_count == 2 &&
        destination.fields_.size() == 1 &&
        destination.fields_.front().device_id ==
            source_groups.device_id;
    if (!reusable)
    {
        destination = GpuCiphertextData::allocate_single_device(
            source_groups.degree,
            source_groups.q_count,
            2,
            source_groups.device_id);
    }
    destination.meta = destination_meta;
    destination.meta.degree = source_groups.degree;
    destination.meta.q_count = source_groups.q_count;
    destination.meta.p_count = 0;
    destination.meta.component_count = 2;
    destination.meta.is_ntt_form = true;
    auto destination_view = destination.make_view();
    kernel::launch_hybrid_apply_moddown_ntt_from_q_groups(
        destination_view.polys[0].shards.front().ptr,
        destination_view.polys[1].shards.front().ptr,
        source_groups.q.data(),
        workspace.converted_q0.data(),
        workspace.converted_q1.data(),
        group_count,
        *parameter_shard,
        source_groups.degree);
}

void GpuKeySwitchHandler::switch_key_hybrid_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstRNSPolyView &switch_poly_ntt,/*可以自由选择需要密钥切换的密文分量位置*/
    const GpuConstEvaluationKeyView &switch_keys_view,
    const GpuEvaluationKeyData &switch_keys_data,
    std::size_t key_index,/*可以自由选择密钥切换的密钥类型*/
    const GpuLevelInfo &level_info) const
{
    switch_key_hybrid_ciphertext_impl(
        destination_view,
        switch_poly_ntt,
        switch_keys_view,
        switch_keys_data,
        key_index,
        level_info,
        nullptr,
        nullptr);
}

void GpuKeySwitchHandler::switch_key_hybrid_ciphertext_impl(
    GpuCiphertextView &destination_view,
    const GpuConstRNSPolyView &switch_poly_ntt,
    const GpuConstEvaluationKeyView &switch_keys_view,
    const GpuEvaluationKeyData &switch_keys_data,
    std::size_t key_index,
    const GpuLevelInfo &level_info,
    const GpuConstRNSPolyView *add_source0,
    const GpuConstRNSPolyView *add_source1,
    const GpuLevelInfo *rescale_x2_destination_level) const
{
    (void)params_;
    if (rescale_x2_destination_level == nullptr)
    {
        validate_hybrid_switch_key_shape(
            destination_view,
            switch_poly_ntt,
            switch_keys_view,
            switch_keys_data,
            key_index,
            level_info);
    }
    else
    {
        if (!(switch_poly_ntt.shards.size() == 1) ||
            !(destination_view.meta.parms_id ==
              rescale_x2_destination_level->parms_id) ||
            destination_view.meta.degree != level_info.degree ||
            destination_view.meta.q_count + 2 != level_info.q_count ||
            destination_view.meta.q_count !=
                rescale_x2_destination_level->q_count ||
            destination_view.meta.p_count != 0 ||
            destination_view.polys.size() != 2 ||
            destination_view.meta.component_count != 2 ||
            switch_keys_view.meta.q_count != level_info.q_count ||
            switch_keys_view.storage_q_count != switch_keys_data.meta.q_count ||
            switch_keys_view.meta.key_count <= key_index ||
            switch_keys_view.meta.decomposition_count == 0 ||
            switch_keys_view.meta.component_count < kSwitchKeyComponentCount ||
            !(switch_keys_view.meta.key_parms_id ==
              switch_keys_data.meta.key_parms_id) ||
            switch_keys_data.poly_metadata_.empty())
        {
            throw std::invalid_argument(
                "GpuKeySwitchHandler::switch_key_hybrid_ciphertext_rescale_x2: shape mismatch");
        }
        validate_single_full_shard(
            "GpuKeySwitchHandler switch poly rescale_x2",
            switch_poly_ntt,
            level_info.degree,
            level_info.q_count);
        for (const auto &poly : destination_view.polys)
        {
            validate_single_full_shard(
                "GpuKeySwitchHandler destination rescale_x2",
                poly,
                destination_view.meta.degree,
                destination_view.meta.q_count);
        }
        if (switch_keys_data.polys_.size() != switch_keys_view.polys.size())
        {
            throw std::invalid_argument(
                "GpuKeySwitchHandler::switch_key_hybrid_ciphertext_rescale_x2: invalid key layout");
        }
    }

    const std::size_t base_q_size = level_info.q_count;
    const std::size_t base_p_size = switch_keys_view.meta.p_count;
    if (switch_keys_view.meta.q_count != base_q_size)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: active key Q prefix must match ciphertext q_count");
    }
    if (switch_keys_view.storage_q_count != switch_keys_data.meta.q_count ||
        switch_keys_view.storage_q_count < base_q_size)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: invalid full-key storage Q count");
    }
    if (base_p_size == 0)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: HYBRID requires key p limbs");
    }
    NvtxRange range(
        "keyswitch.hybrid.q" + std::to_string(base_q_size) +
        ".p" + std::to_string(base_p_size) +
        ".d" + std::to_string(
            (base_q_size + base_p_size - 1) / base_p_size));
    /* 计算dnum的分块数，向上取整，按base_p_size分块 */
    const std::size_t expected_decomposition_count =
        (base_q_size + base_p_size - 1) / base_p_size;
    if (switch_keys_view.meta.decomposition_count < expected_decomposition_count)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: key decomposition count is too small");
    }

    const int device_id = destination_view.polys[0].shards.front().device_id;
    /*
     * Relinearization repeatedly visits descending Q levels. Keep the largest
     * buffers alive and reinterpret only their active prefixes at lower
     * levels. The legacy path deliberately retains per-call allocation for
     * same-binary A/B measurements.
     */
    HybridScratch local_scratch;
    HybridScratch *scratch_ptr = nullptr;
    const bool persistent_relinearize =
        add_source0 != nullptr &&
        add_source1 != nullptr &&
        use_persistent_relinearize();
    if (persistent_relinearize)
    {
        ensure_hybrid_scratch(
            persistent_workspace_->relinearize,
            device_id,
            destination_view.meta.degree,
            base_q_size,
            base_p_size);
        scratch_ptr = &persistent_workspace_->relinearize;
    }
    else
    {
        local_scratch = allocate_hybrid_scratch(
            device_id,
            destination_view.meta.degree,
            base_q_size,
            base_p_size);
        scratch_ptr = &local_scratch;
    }
    HybridScratch &scratch = *scratch_ptr;
    /* 将待切换分量通过 INTT 从点值域转换到系数域 */
    inverse_ntt_switch_poly(
        scratch,
        switch_poly_ntt,
        level_info);

#if 0
    /* Retired all-dnum PAccum experiments. */
    if (use_paccum_final_tail())
    {
        NvtxRange paccum_range("keyswitch.paccum_final_tail");
        allocate_hybrid_paccum_all_dnum_scratch(
            scratch,
            expected_decomposition_count);

        std::vector<const GpuWord *> key0_ptrs(expected_decomposition_count);
        std::vector<const GpuWord *> key1_ptrs(expected_decomposition_count);
        const GpuParameterShard *paccum_parameter_shard = nullptr;

        for (std::size_t decomp_index = 0;
             decomp_index < expected_decomposition_count;
             ++decomp_index)
        {
            const std::size_t decomp_limb_begin = decomp_index * base_p_size;
            const std::size_t decomp_limb_count = std::min(
                base_p_size,
                base_q_size - decomp_limb_begin);

            const auto &key_component0 = find_key_poly(
                switch_keys_view,
                switch_keys_data,
                key_index,
                decomp_index,
                0);
            const auto &key_component1 = find_key_poly(
                switch_keys_view,
                switch_keys_data,
                key_index,
                decomp_index,
                1);

            validate_single_full_shard(
                "GpuKeySwitchHandler switch key c0",
                key_component0,
                switch_keys_view.meta.degree,
                switch_keys_view.meta.q_count + switch_keys_view.meta.p_count);
            validate_single_full_shard(
                "GpuKeySwitchHandler switch key c1",
                key_component1,
                switch_keys_view.meta.degree,
                switch_keys_view.meta.q_count + switch_keys_view.meta.p_count);

            const auto &key0_shard = key_component0.shards.front();
            const auto *parameter_shard =
                find_parameter_shard(level_info, key0_shard);
            if (parameter_shard == nullptr)
            {
                throw std::invalid_argument(
                    "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: no matching final-tail parameter shard");
            }
            if (paccum_parameter_shard == nullptr)
            {
                paccum_parameter_shard = parameter_shard;
            }

            key0_ptrs[decomp_index] = key0_shard.ptr;
            key1_ptrs[decomp_index] =
                key_component1.shards.front().ptr;

            prepare_hybrid_decomposition_final_tail_block(
                decomp_index,
                decomp_limb_begin,
                decomp_limb_count,
                scratch,
                scratch.all_modup_q.data() +
                    decomp_index * scratch.q_word_count,
                scratch.all_modup_p.data() +
                    decomp_index * scratch.p_word_count,
                key_component0,
                key_component1,
                level_info);
        }

        scratch.key_qp0_by_dnum.copy_from_host(
            key0_ptrs.data(),
            key0_ptrs.size());
        scratch.key_qp1_by_dnum.copy_from_host(
            key1_ptrs.data(),
            key1_ptrs.size());

        if (paccum_parameter_shard == nullptr)
        {
            throw std::invalid_argument(
                "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: missing final-tail PAccum parameter shard");
        }

        {
            NvtxRange range(
                "keyswitch.paccum_final_tail.final_ntt_accumulate_all_dnum");
            kernel::launch_hybrid_final_ntt_paccum_all_dnum_two_components(
                scratch.accum_q0.data(),
                scratch.accum_p0.data(),
                scratch.accum_q1.data(),
                scratch.accum_p1.data(),
                scratch.all_modup_q.data(),
                scratch.all_modup_p.data(),
                switch_poly_ntt.shards.front().ptr,
                scratch.key_qp0_by_dnum.data(),
                scratch.key_qp1_by_dnum.data(),
                expected_decomposition_count,
                *paccum_parameter_shard,
                scratch.degree);
        }
    }
    else if (use_paccum_all_dnum())
    {
        NvtxRange paccum_range("keyswitch.paccum_all_dnum");
        allocate_hybrid_paccum_all_dnum_scratch(
            scratch,
            expected_decomposition_count);

        std::vector<const GpuWord *> key0_ptrs(expected_decomposition_count);
        std::vector<const GpuWord *> key1_ptrs(expected_decomposition_count);
        const GpuParameterShard *paccum_parameter_shard = nullptr;

        for (std::size_t decomp_index = 0;
             decomp_index < expected_decomposition_count;
             ++decomp_index)
        {
            const std::size_t decomp_limb_begin = decomp_index * base_p_size;
            const std::size_t decomp_limb_count = std::min(
                base_p_size,
                base_q_size - decomp_limb_begin);

            const auto &key_component0 = find_key_poly(
                switch_keys_view,
                switch_keys_data,
                key_index,
                decomp_index,
                0);
            const auto &key_component1 = find_key_poly(
                switch_keys_view,
                switch_keys_data,
                key_index,
                decomp_index,
                1);

            validate_single_full_shard(
                "GpuKeySwitchHandler switch key c0",
                key_component0,
                switch_keys_view.meta.degree,
                switch_keys_view.meta.q_count + switch_keys_view.meta.p_count);
            validate_single_full_shard(
                "GpuKeySwitchHandler switch key c1",
                key_component1,
                switch_keys_view.meta.degree,
                switch_keys_view.meta.q_count + switch_keys_view.meta.p_count);

            const auto &key0_shard = key_component0.shards.front();
            const auto *parameter_shard =
                find_parameter_shard(level_info, key0_shard);
            if (parameter_shard == nullptr)
            {
                throw std::invalid_argument(
                    "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: no matching parameter shard");
            }
            if (paccum_parameter_shard == nullptr)
            {
                paccum_parameter_shard = parameter_shard;
            }

            key0_ptrs[decomp_index] = key0_shard.ptr;
            key1_ptrs[decomp_index] =
                key_component1.shards.front().ptr;

            prepare_hybrid_decomposition_ntt_block(
                decomp_index,
                decomp_limb_begin,
                decomp_limb_count,
                scratch,
                scratch.all_modup_q.data() +
                    decomp_index * scratch.q_word_count,
                scratch.all_modup_p.data() +
                    decomp_index * scratch.p_word_count,
                switch_poly_ntt,
                key_component0,
                key_component1,
                level_info);
        }

        scratch.key_qp0_by_dnum.copy_from_host(
            key0_ptrs.data(),
            key0_ptrs.size());
        scratch.key_qp1_by_dnum.copy_from_host(
            key1_ptrs.data(),
            key1_ptrs.size());

        if (paccum_parameter_shard == nullptr)
        {
            throw std::invalid_argument(
                "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: missing PAccum parameter shard");
        }

        {
            NvtxRange range("keyswitch.paccum.accumulate_all_dnum");
            kernel::launch_hybrid_paccum_all_dnum_two_components(
                scratch.accum_q0.data(),
                scratch.accum_p0.data(),
                scratch.accum_q1.data(),
                scratch.accum_p1.data(),
                scratch.all_modup_q.data(),
                scratch.all_modup_p.data(),
                switch_poly_ntt.shards.front().ptr,
                scratch.key_qp0_by_dnum.data(),
                scratch.key_qp1_by_dnum.data(),
                expected_decomposition_count,
                *paccum_parameter_shard,
                scratch.degree);
        }
    }
    else
#endif
    {
        const bool fuse_decomp_q = use_fused_decomp_q();
        const bool fuse_modup_ntt_head = use_fused_modup_ntt_head();
        const bool bconv_row_tiled = use_bconv_row_tiled();
        const bool bconv_row_tiled8 = use_bconv_row_tiled8();
        /* 分块循环：每个 dnum 分块做 ModUp + NTT + 密钥乘加累加 */
        for (std::size_t decomp_index = 0;
             decomp_index < expected_decomposition_count;
             ++decomp_index)
        {
            /* 计算在dnum块中的起始位置，因为是按照p_size进行分块的，所以现在在块内循环，块index*块大小可以定位起始地址位置*/
            const std::size_t decomp_limb_begin = decomp_index * base_p_size;
            const std::size_t decomp_limb_count = std::min(
                base_p_size,
                base_q_size - decomp_limb_begin);

            /* find_key_poly主要用来从已经上传到 GPU 里，找到某一个具体的 key 多项式 */
            /* 主要原因是 CPU 侧的密钥是分层的：key index + dnum 分块 + ksk0/ksk1 分量 */
            const auto &key_component0 = find_key_poly(
                switch_keys_view,
                switch_keys_data,
                key_index,
                decomp_index,
                0);
            const auto &key_component1 = find_key_poly(
                switch_keys_view,
                switch_keys_data,
                key_index,
                decomp_index,
                1);

            validate_single_full_shard(
                "GpuKeySwitchHandler switch key c0",
                key_component0,
                switch_keys_data.meta.degree,
                switch_keys_data.meta.q_count + switch_keys_data.meta.p_count);
            validate_single_full_shard(
                "GpuKeySwitchHandler switch key c1",
                key_component1,
                switch_keys_data.meta.degree,
                switch_keys_data.meta.q_count + switch_keys_data.meta.p_count);

            const auto key_component0_level =
                make_hybrid_key_component_view(
                    key_component0,
                    switch_keys_view,
                    switch_keys_data);
            const auto key_component1_level =
                make_hybrid_key_component_view(
                    key_component1,
                    switch_keys_view,
                    switch_keys_data);

            /* 处理每一个 dnum 分块的函数，负责模升+NTT+乘密钥累加*/
            process_hybrid_decomposition_block(
                decomp_index,
                decomp_limb_begin,
                decomp_limb_count,
                scratch,
                switch_poly_ntt,
                key_component0_level,
                key_component1_level,
                level_info,
                fuse_decomp_q,
                fuse_modup_ntt_head,
                bconv_row_tiled,
                bconv_row_tiled8);
        }
    }

    /* INTT 模降 NTT 和原密文分量求和 */
    if (rescale_x2_destination_level == nullptr)
    {
        finalize_hybrid_relinearize(
            destination_view,
            scratch,
            level_info,
            add_source0,
            add_source1);
    }
    else
    {
        finalize_hybrid_relinearize_rescale_x2(
            destination_view,
            scratch,
            level_info,
            *rescale_x2_destination_level,
            add_source0,
            add_source1);
    }
}

void GpuKeySwitchHandler::relinearize_hybrid_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuConstEvaluationKeyView &relin_keys_view,
    const GpuEvaluationKeyData &relin_keys_data,
    const GpuLevelInfo &level_info) const
{
    validate_hybrid_relinearize_shape(
        destination_view,
        source_view,
        relin_keys_view,
        relin_keys_data,
        level_info);

    if (!use_persistent_relinearize())
    {
        /*
         * Legacy A/B path: two synchronous D2D copies followed by a fresh
         * KeySwitch scratch allocation.
         */
        copy_initial_components(destination_view, source_view);
        switch_key_hybrid_ciphertext(
            destination_view,
            source_view.polys[2],
            relin_keys_view,
            relin_keys_data,
            kRelinKeyPower2Index,
            level_info);
        return;
    }

    /*
     * Out-of-place finalize: the ModDown kernel reads source c0/c1 directly
     * and writes source + switched(c2) to the destination. No synchronous
     * pre-copy is needed.
     */
    switch_key_hybrid_ciphertext_impl(
        destination_view,
        source_view.polys[2],
        relin_keys_view,
        relin_keys_data,
        kRelinKeyPower2Index,
        level_info,
        &source_view.polys[0],
        &source_view.polys[1]);
}

void GpuKeySwitchHandler::relinearize_hybrid_ciphertext_rescale_x2(
    GpuCiphertextView &destination_view,
    const GpuConstCiphertextView &source_view,
    const GpuConstEvaluationKeyView &relin_keys_view,
    const GpuEvaluationKeyData &relin_keys_data,
    const GpuLevelInfo &source_level_info,
    const GpuLevelInfo &destination_level_info) const
{
    if (!(source_view.meta.parms_id == source_level_info.parms_id) ||
        !(destination_view.meta.parms_id ==
          destination_level_info.parms_id) ||
        !source_view.meta.is_ntt_form ||
        !destination_view.meta.is_ntt_form ||
        source_view.polys.size() != 3 ||
        source_view.meta.component_count != 3 ||
        destination_view.polys.size() != 2 ||
        destination_view.meta.component_count != 2 ||
        source_view.meta.degree != destination_view.meta.degree ||
        source_view.meta.degree != source_level_info.degree ||
        source_view.meta.degree != destination_level_info.degree ||
        source_view.meta.q_count != source_level_info.q_count ||
        destination_view.meta.q_count != destination_level_info.q_count ||
        destination_view.meta.q_count + 2 != source_view.meta.q_count ||
        source_view.meta.p_count != 0 ||
        destination_view.meta.p_count != 0)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::relinearize_hybrid_ciphertext_rescale_x2: shape mismatch");
    }
    if (!(relin_keys_view.meta.key_parms_id ==
          relin_keys_data.meta.key_parms_id) ||
        relin_keys_view.storage_q_count != relin_keys_data.meta.q_count ||
        relin_keys_view.meta.q_count != source_view.meta.q_count ||
        relin_keys_view.meta.key_count == 0 ||
        relin_keys_view.meta.decomposition_count == 0 ||
        relin_keys_view.meta.component_count < kSwitchKeyComponentCount ||
        relin_keys_data.poly_metadata_.empty() ||
        relin_keys_data.polys_.size() != relin_keys_view.polys.size())
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::relinearize_hybrid_ciphertext_rescale_x2: invalid relin key metadata");
    }
    for (std::size_t component = 0; component < source_view.polys.size();
         ++component)
    {
        validate_single_full_shard(
            "GpuKeySwitchHandler relin-rescale source",
            source_view.polys[component],
            source_view.meta.degree,
            source_view.meta.q_count);
        if (!same_shard_placement(
                source_view.polys[0].shards.front(),
                source_view.polys[component].shards.front()))
        {
            throw std::invalid_argument(
                "GpuKeySwitchHandler::relinearize_hybrid_ciphertext_rescale_x2: source shard placement mismatch");
        }
    }
    for (const auto &poly : destination_view.polys)
    {
        validate_single_full_shard(
            "GpuKeySwitchHandler relin-rescale destination",
            poly,
            destination_view.meta.degree,
            destination_view.meta.q_count);
    }

    switch_key_hybrid_ciphertext_impl(
        destination_view,
        source_view.polys[2],
        relin_keys_view,
        relin_keys_data,
        kRelinKeyPower2Index,
        source_level_info,
        &source_view.polys[0],
        &source_view.polys[1],
        &destination_level_info);
}

}  // namespace gpu
}  // namespace poseidon
