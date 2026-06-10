#include "poseidon/gpu/gpu_keyswitch_handler.h"
#include "poseidon/gpu/kernels/gpu_elementwise_kernels.h"
#include "poseidon/gpu/kernels/gpu_keyswitch_kernels.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace poseidon
{
namespace gpu
{
namespace
{

constexpr std::size_t kRelinKeyPower2Index = 0;
constexpr std::size_t kSwitchKeyComponentCount = 2;

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

    scratch.accum_q0.fill_zero();
    scratch.accum_q1.fill_zero();
    scratch.accum_p0.fill_zero();
    scratch.accum_p1.fill_zero();
    return scratch;
}

void inverse_ntt_switch_poly(
    HybridScratch &scratch,
    const GpuConstRNSPolyView &switch_poly_ntt,
    const GpuLevelInfo &level_info)
{
    NvtxRange range("keyswitch.intt_switch_poly");
    const auto &source_shard = switch_poly_ntt.shards.front();
    auto destination_shard = scratch.c2_intt_view();
    const auto *parameter_shard = find_parameter_shard(level_info, source_shard);
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::inverse_ntt_switch_poly: no matching parameter shard");
    }

    kernel::launch_inverse_ntt_poly_shard(
        destination_shard,
        source_shard,
        *parameter_shard,
        scratch.degree);
}

/* 处理单个dnum分量的计算函数 */
void process_hybrid_decomposition_block(
    std::size_t decomp_index,
    std::size_t decomp_limb_begin,
    std::size_t decomp_limb_count,
    HybridScratch &scratch,
    const GpuConstRNSPolyView &switch_poly_ntt,
    const GpuConstRNSPolyView &key_component0,
    const GpuConstRNSPolyView &key_component1,
    const GpuLevelInfo &level_info)
{
    NvtxRange block_range(
        "keyswitch.dnum[" + std::to_string(decomp_index) + "]");
    const auto &switch_poly_shard = switch_poly_ntt.shards.front();
    const auto &key0_shard = key_component0.shards.front();
    const auto &key1_shard = key_component1.shards.front();
    if (!same_shard_placement(key0_shard, key1_shard))
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::process_hybrid_decomposition_block: key shard placement mismatch");
    }

    const auto *parameter_shard = find_parameter_shard(level_info, key0_shard);
    if (parameter_shard == nullptr)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::process_hybrid_decomposition_block: no matching parameter shard");
    }
    validate_hybrid_parameter_shape(
        "GpuKeySwitchHandler::process_hybrid_decomposition_block",
        scratch,
        *parameter_shard);

    /* 将dnum片段进行模升，每个片段扩展到完整的32+6=38个模数 */
    {
        NvtxRange range("keyswitch.dnum.modup");
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

    /* 当前dnum块内的Q limb直接复用原始c2的NTT值，只对新模升产生的 Q/P limb做NTT操作 */
    {
        NvtxRange range("keyswitch.dnum.forward_ntt_qp");
        kernel::launch_hybrid_forward_ntt_qp(
            scratch.modup_q.data(),
            scratch.modup_p.data(),
            decomp_limb_begin,
            decomp_limb_count,
            *parameter_shard,
            scratch.degree);
    }

    /* 同时和ksk0/ksk1相乘并且累加 */
    {
        NvtxRange range("keyswitch.dnum.mul_accum.c01");
        kernel::launch_hybrid_multiply_accumulate_two_components(
            scratch.accum_q0.data(),
            scratch.accum_p0.data(),
            scratch.accum_q1.data(),
            scratch.accum_p1.data(),
            scratch.modup_q.data(),
            scratch.modup_p.data(),
            key0_shard.ptr,
            key1_shard.ptr,
            *parameter_shard,
            scratch.degree);
    }
}

void finalize_hybrid_relinearize(
    GpuCiphertextView &destination,
    HybridScratch &scratch,
    const GpuLevelInfo &level_info)
{
    NvtxRange finalize_range("keyswitch.finalize");
    const auto &destination_shard0 = destination.polys[0].shards.front();
    const auto &destination_shard1 = destination.polys[1].shards.front();
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

    auto accum_q0_view = make_scratch_q_view(scratch.accum_q0.data(), scratch);
    auto accum_q1_view = make_scratch_q_view(scratch.accum_q1.data(), scratch);
    auto accum_p0_view = make_scratch_p_view(scratch.accum_p0.data(), scratch);
    auto accum_p1_view = make_scratch_p_view(scratch.accum_p1.data(), scratch);
    auto converted_q0_view = make_scratch_q_view(scratch.c2_intt.data(), scratch);
    auto converted_q1_view = make_scratch_q_view(scratch.modup_q.data(), scratch);

    /* 只把P部分转回系数域，Q部分保持在NTT域 */
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

    /* P->Q基转换，结果先放在系数域临时buffer里 */
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

    /* 将转换到Q的部分转回NTT域，然后在NTT域完成模降的减法和乘P逆 */
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
    {
        NvtxRange range("keyswitch.finalize.apply_moddown_ntt");
        kernel::launch_hybrid_apply_moddown_ntt(
            scratch.accum_q0.data(),
            scratch.accum_q1.data(),
            scratch.c2_intt.data(),
            scratch.modup_q.data(),
            *parameter_shard,
            scratch.degree);
    }

    /* 和d0/d1累加 */
    {
        NvtxRange range("keyswitch.finalize.add_back");
        kernel::launch_add_two_poly_shards(
            destination_shard0,
            destination_shard1,
            as_const_shard(destination_shard0),
            as_const_shard(destination_shard1),
            as_const_shard(accum_q0_view),
            as_const_shard(accum_q1_view),
            *parameter_shard,
            scratch.degree);
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

GpuKeySwitchHandler::GpuKeySwitchHandler(const GpuParameterData &params)
    : params_(params)
{}

void GpuKeySwitchHandler::switch_key_hybrid_ciphertext(
    GpuCiphertextView &destination_view,
    const GpuConstRNSPolyView &switch_poly_ntt,/*可以自由选择需要密钥切换的密文分量位置*/
    const GpuConstEvaluationKeyView &switch_keys_view,
    const GpuEvaluationKeyData &switch_keys_data,
    std::size_t key_index,/*可以自由选择密钥切换的密钥类型*/
    const GpuLevelInfo &level_info) const
{
    NvtxRange range("keyswitch.hybrid");
    (void)params_;
    validate_hybrid_switch_key_shape(
        destination_view,
        switch_poly_ntt,
        switch_keys_view,
        switch_keys_data,
        key_index,
        level_info);

    const std::size_t base_q_size = destination_view.meta.q_count;
    const std::size_t base_p_size = switch_keys_view.meta.p_count;
    if (switch_keys_view.meta.q_count != base_q_size)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: current implementation requires key q_count to match ciphertext q_count");
    }
    if (base_p_size == 0)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: HYBRID requires key p limbs");
    }
    /* 计算dnum的分块数，向上取整，按base_p_size分块 */
    const std::size_t expected_decomposition_count =
        (base_q_size + base_p_size - 1) / base_p_size;
    if (switch_keys_view.meta.decomposition_count < expected_decomposition_count)
    {
        throw std::invalid_argument(
            "GpuKeySwitchHandler::switch_key_hybrid_ciphertext: key decomposition count is too small");
    }

    const int device_id = destination_view.polys[0].shards.front().device_id;
    /* HYBRID key-switch过程里的临时显存缓存，存放中间变量 */
    auto scratch = allocate_hybrid_scratch(
        device_id,
        destination_view.meta.degree,
        base_q_size,
        base_p_size);
    /* 将待切换分量通过 INTT 从点值域转换到系数域 */
    inverse_ntt_switch_poly(
        scratch,
        switch_poly_ntt,
        level_info);

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
            switch_keys_view.meta.degree,
            switch_keys_view.meta.q_count + switch_keys_view.meta.p_count);
        validate_single_full_shard(
            "GpuKeySwitchHandler switch key c1",
            key_component1,
            switch_keys_view.meta.degree,
            switch_keys_view.meta.q_count + switch_keys_view.meta.p_count);

        /* 处理每一个 dnum 分块的函数，负责模升+NTT+乘密钥累加*/
        process_hybrid_decomposition_block(
            decomp_index,
            decomp_limb_begin,
            decomp_limb_count,
            scratch,
            switch_poly_ntt,
            key_component0,
            key_component1,
            level_info);
    }

    /* INTT 模降 NTT 和原密文分量求和*/
    finalize_hybrid_relinearize(
        destination_view,
        scratch,
        level_info);
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

    /*relin = 先复制 c0/c1，再对 c2 做通用 switch-key*/
    copy_initial_components(destination_view, source_view);
    /*直接调用密钥切换,第三个分量需要被密钥切换*/
    switch_key_hybrid_ciphertext(
        destination_view,
        source_view.polys[2],
        relin_keys_view,
        relin_keys_data,
        kRelinKeyPower2Index,/*数值为0，表示该密钥为重线性化密钥*/
        level_info);
}

}  // namespace gpu
}  // namespace poseidon
