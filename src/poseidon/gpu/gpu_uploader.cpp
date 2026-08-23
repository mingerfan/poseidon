#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/gpu/gpu_scale_planner.h"
#include "poseidon/gpu/kernels/gpu_keyswitch_kernels.h"

#include "poseidon/advance/homomorphic_linear_transform.h"
#include "poseidon/advance/homomorphic_mod.h"
#include "poseidon/ciphertext.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"
#include "poseidon/basics/util/ntt.h"
#include "poseidon/basics/util/uintarith.h"
#include "poseidon/basics/util/uintarithsmallmod.h"
#include "poseidon/basics/util/uintcore.h"
#include "poseidon/util/rns_tool_qp.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
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

std::size_t checked_mul(std::size_t a, std::size_t b, const char *what)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    {
        throw std::overflow_error(what);
    }
    return a * b;
}

GpuWord checked_gpu_word(std::uint64_t value, const char *what)
{
    if (value > std::numeric_limits<GpuWord>::max())
    {
        throw std::invalid_argument(what);
    }
    return static_cast<GpuWord>(value);
}

void copy_uint64_to_device_field(
    const std::uint64_t *src,
    std::size_t count,
    GpuFieldData &dst,
    const char *what)
{
    if (count > dst.size())
    {
        throw std::out_of_range("source data exceeds GPU field allocation");
    }
    if (count != 0 && src == nullptr)
    {
        throw std::invalid_argument("source pointer is null");
    }

    std::vector<GpuWord> tmp(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        tmp[i] = checked_gpu_word(src[i], what);
    }

    dst.buffer.copy_from_host(tmp.data(), tmp.size());
}

void copy_gpu_words_to_device_field_offset(
    const GpuWord *src,
    std::size_t count,
    GpuFieldData &dst,
    std::size_t dst_offset)
{
    if (dst_offset > dst.size() || count > dst.size() - dst_offset)
    {
        throw std::out_of_range("source data exceeds GPU field allocation");
    }
    if (count != 0 && src == nullptr)
    {
        throw std::invalid_argument("source pointer is null");
    }
    if (count == 0)
    {
        return;
    }

    gpu_check_cuda(cudaSetDevice(dst.device_id), "cudaSetDevice");
    gpu_check_cuda(
        cudaMemcpy(
            dst.data() + dst_offset,
            src,
            count * sizeof(GpuWord),
            cudaMemcpyHostToDevice),
        "cudaMemcpyHostToDevice shard");
}

void copy_device_field_to_uint64(
    const GpuFieldData &src,
    std::uint64_t *dst,
    std::size_t count)
{
    if (count > src.size())
    {
        throw std::out_of_range("GPU field is smaller than requested download size");
    }
    if (count != 0 && dst == nullptr)
    {
        throw std::invalid_argument("destination pointer is null");
    }

    std::vector<GpuWord> tmp(count);
    src.buffer.copy_to_host(tmp.data(), tmp.size());
    std::transform(tmp.cbegin(), tmp.cend(), dst, [](GpuWord value) {
        return static_cast<std::uint64_t>(value);
    });
}

void copy_device_field_offset_to_gpu_words(
    const GpuFieldData &src,
    std::size_t src_offset,
    GpuWord *dst,
    std::size_t count)
{
    if (src_offset > src.size() || count > src.size() - src_offset)
    {
        throw std::out_of_range("GPU field is smaller than requested shard download size");
    }
    if (count != 0 && dst == nullptr)
    {
        throw std::invalid_argument("destination pointer is null");
    }
    if (count == 0)
    {
        return;
    }

    gpu_check_cuda(cudaSetDevice(src.device_id), "cudaSetDevice");
    gpu_check_cuda(
        cudaMemcpy(
            dst,
            src.data() + src_offset,
            count * sizeof(GpuWord),
            cudaMemcpyDeviceToHost),
        "cudaMemcpyDeviceToHost shard");
}

void ciphertext_limb_shape(
    const Ciphertext &src,
    std::size_t &q_count,
    std::size_t &p_count)
{
    q_count = src.coeff_modulus_size();
    p_count = 0;

    if (!src.polys().empty() && src.polys()[0].poly_degree() != 0)
    {
        q_count = src.polys()[0].rns_num_q();
        p_count = src.polys()[0].rns_num_p();
    }
}

void plaintext_limb_shape(
    const Plaintext &src,
    std::size_t &degree,
    std::size_t &q_count,
    std::size_t &p_count)
{
    if (src.is_ntt_form())
    {
        const auto &poly = src.poly();
        degree = poly.poly_degree();
        q_count = poly.rns_num_q();
        p_count = poly.rns_num_p();
        if (degree == 0 || q_count + p_count == 0)
        {
            throw std::invalid_argument(
                "NTT plaintext does not carry a usable RNSPoly shape");
        }
        return;
    }

    degree = src.coeff_count();
    q_count = src.coeff_count() == 0 ? 0 : 1;
    p_count = 0;
}

void upload_ciphertext_component_shards(
    const Ciphertext &src,
    std::size_t component,
    const GpuRNSPoly &poly,
    std::vector<GpuFieldData> &fields,
    const char *what)
{
    const auto degree = src.poly_modulus_degree();
    const auto *component_data = src.data(component);

    for (const auto &shard : poly.shards)
    {
        const auto shard_word_count = checked_mul(
            shard.limb_count,
            shard.coeff_count,
            "ciphertext shard word count overflow");
        std::vector<GpuWord> packed(shard_word_count);

        for (std::size_t local_limb = 0; local_limb < shard.limb_count; ++local_limb)
        {
            for (std::size_t local_coeff = 0; local_coeff < shard.coeff_count; ++local_coeff)
            {
                const auto cpu_index =
                    checked_mul(
                        shard.limb_begin + local_limb,
                        degree,
                        "ciphertext CPU shard index overflow") +
                    shard.coeff_begin + local_coeff;
                const auto gpu_index =
                    checked_mul(
                        local_limb,
                        shard.coeff_count,
                        "ciphertext GPU shard index overflow") +
                    local_coeff;

                packed[gpu_index] = checked_gpu_word(component_data[cpu_index], what);
            }
        }

        copy_gpu_words_to_device_field_offset(
            packed.data(),
            packed.size(),
            fields.at(shard.field_index),
            shard.field_offset);
    }
}

void download_ciphertext_component_shards(
    const GpuRNSPoly &poly,
    const std::vector<GpuFieldData> &fields,
    std::uint64_t *component_data)
{
    for (const auto &shard : poly.shards)
    {
        const auto shard_word_count = checked_mul(
            shard.limb_count,
            shard.coeff_count,
            "ciphertext shard word count overflow");
        std::vector<GpuWord> packed(shard_word_count);

        copy_device_field_offset_to_gpu_words(
            fields.at(shard.field_index),
            shard.field_offset,
            packed.data(),
            packed.size());

        for (std::size_t local_limb = 0; local_limb < shard.limb_count; ++local_limb)
        {
            for (std::size_t local_coeff = 0; local_coeff < shard.coeff_count; ++local_coeff)
            {
                const auto cpu_index =
                    checked_mul(
                        shard.limb_begin + local_limb,
                        poly.degree,
                        "ciphertext CPU shard index overflow") +
                    shard.coeff_begin + local_coeff;
                const auto gpu_index =
                    checked_mul(
                        local_limb,
                        shard.coeff_count,
                        "ciphertext GPU shard index overflow") +
                    local_coeff;

                component_data[cpu_index] = static_cast<std::uint64_t>(packed[gpu_index]);
            }
        }
    }
}

void append_uploaded_ciphertext_as_key_polys(
    GpuCiphertextData &&uploaded,
    std::size_t key_index,
    std::size_t decomposition_index,
    GpuEvaluationKeyData &dst)
{
    if (dst.meta.degree == 0)
    {
        dst.meta.degree = uploaded.meta.degree;
        dst.meta.q_count = uploaded.meta.q_count;
        dst.meta.p_count = uploaded.meta.p_count;
    }
    else if (dst.meta.degree != uploaded.meta.degree ||
             dst.meta.q_count != uploaded.meta.q_count ||
             dst.meta.p_count != uploaded.meta.p_count)
    {
        throw std::invalid_argument("inconsistent evaluation-key polynomial shape");
    }

    dst.meta.component_count = std::max(dst.meta.component_count, uploaded.meta.component_count);

    const auto field_base = dst.fields_.size();
    for (auto &field : uploaded.fields_)
    {
        dst.fields_.push_back(std::move(field));
    }

    for (std::size_t component = 0; component < uploaded.polys_.size(); ++component)
    {
        auto poly = std::move(uploaded.polys_[component]);
        poly.poly_id = dst.polys_.size();
        for (auto &shard : poly.shards)
        {
            shard.field_index += field_base;
        }

        GpuEvaluationKeyPolyMeta poly_meta;
        poly_meta.poly_id = poly.poly_id;
        poly_meta.key_index = key_index;
        poly_meta.decomposition_index = decomposition_index;
        poly_meta.component_index = component;

        dst.poly_metadata_.push_back(poly_meta);
        dst.polys_.push_back(std::move(poly));
    }
}

template <typename KSwitchKeyType>
GpuEvaluationKeyData upload_kswitch_keys(
    const KSwitchKeyType &src,
    int device_id,
    const std::vector<std::size_t> *selected_key_indices = nullptr)
{
    GpuEvaluationKeyData dst;
    dst.meta.key_parms_id = src.parms_id();
    dst.meta.key_count = src.data().size();

    const auto upload_key = [&](std::size_t key_index)
    {
        if (key_index >= src.data().size() || src.data()[key_index].empty())
        {
            throw std::invalid_argument("selected evaluation key does not exist");
        }
        const auto &decompositions = src.data()[key_index];
        dst.meta.decomposition_count =
            std::max(dst.meta.decomposition_count, decompositions.size());

        for (std::size_t decomp_index = 0; decomp_index < decompositions.size(); ++decomp_index)
        {
            auto uploaded = GpuUploader::upload_ciphertext(
                decompositions[decomp_index].data(),
                device_id);
            append_uploaded_ciphertext_as_key_polys(
                std::move(uploaded),
                key_index,
                decomp_index,
                dst);
        }
    };

    if (selected_key_indices == nullptr)
    {
        for (std::size_t key_index = 0; key_index < src.data().size(); ++key_index)
        {
            if (!src.data()[key_index].empty())
            {
                upload_key(key_index);
            }
        }
    }
    else
    {
        for (const auto key_index : *selected_key_indices)
        {
            upload_key(key_index);
        }
    }

    return dst;
}

struct EvalModSplitNode
{
    Polynomial polynomial;
    std::uint32_t split_degree = 0;
    std::unique_ptr<EvalModSplitNode> quotient;
    std::unique_ptr<EvalModSplitNode> remainder;
    std::uint32_t node_id = std::numeric_limits<std::uint32_t>::max();

    bool is_leaf() const noexcept
    {
        return quotient == nullptr && remainder == nullptr;
    }
};

std::uint32_t floor_log2_nonzero(std::size_t value)
{
    if (value == 0)
    {
        throw std::invalid_argument("floor_log2_nonzero requires non-zero input");
    }
    std::uint32_t result = 0;
    while (value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

bool use_evalmod_lead_leaf_resplit()
{
    const char *raw = std::getenv("POSEIDON_EVALMOD_LEAD_LEAF_RESPLIT");
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

bool use_evalmod_flat_bsgs_b8()
{
    const char *raw = std::getenv("POSEIDON_EVALMOD_FLAT_BSGS_B8");
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

bool use_evalmod_virtual_degree_bound()
{
    const char *raw = std::getenv("POSEIDON_EVALMOD_VIRTUAL_DEGREE_BOUND");
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

std::uint32_t evalmod_log_split_or(
    std::uint32_t default_log_split,
    std::uint32_t log_degree)
{
    const char *raw = std::getenv("POSEIDON_EVALMOD_LOG_SPLIT");
    if (raw == nullptr || *raw == '\0')
    {
        return default_log_split;
    }

    const std::string value(raw);
    std::size_t consumed = 0;
    unsigned long parsed = 0;
    try
    {
        parsed = std::stoul(value, &consumed, 10);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument(
            "POSEIDON_EVALMOD_LOG_SPLIT must be an integer");
    }
    if (consumed != value.size() || parsed == 0 || parsed >= 31 ||
        parsed > log_degree)
    {
        throw std::invalid_argument(
            "POSEIDON_EVALMOD_LOG_SPLIT must be in [1, log_degree]");
    }
    return static_cast<std::uint32_t>(parsed);
}

std::unique_ptr<EvalModSplitNode> build_eval_mod_split_tree(
    Polynomial polynomial,
    std::uint32_t log_split,
    std::uint32_t log_degree,
    bool resplit_lead_leaf)
{
    if (polynomial.data().empty())
    {
        throw std::invalid_argument("EvalMod polynomial is empty");
    }
    if (log_split >= 31 || log_degree >= 31)
    {
        throw std::invalid_argument("EvalMod polynomial degree is too large");
    }

    auto node = std::make_unique<EvalModSplitNode>();
    node->polynomial = std::move(polynomial);

    const auto degree = static_cast<std::uint32_t>(node->polynomial.degree());
    const std::uint32_t leaf_degree = 1U << log_split;
    if (degree < leaf_degree)
    {
        if (resplit_lead_leaf && node->polynomial.lead() && log_split > 1)
        {
            const std::uint32_t split_period = 1U << (log_split + 1U);
            const std::uint32_t lead_threshold = 1U << (log_split - 1U);
            const auto max_degree = static_cast<std::uint32_t>(
                std::max(node->polynomial.max_degree(), 0));
            if ((max_degree % split_period) > lead_threshold)
            {
                const auto adjusted_log_degree =
                    floor_log2_nonzero(static_cast<std::size_t>(degree) + 1);
                const auto adjusted_log_split = adjusted_log_degree >> 1U;
                if (adjusted_log_split < log_split)
                {
                    return build_eval_mod_split_tree(
                        std::move(node->polynomial),
                        adjusted_log_split,
                        adjusted_log_degree,
                        resplit_lead_leaf);
                }
            }
        }
        return node;
    }

    std::uint32_t next_power = leaf_degree;
    const std::uint32_t split_bound = (degree >> 1U) + 1U;
    while (next_power < split_bound)
    {
        if (next_power > (std::numeric_limits<std::uint32_t>::max() >> 1U))
        {
            throw std::overflow_error("EvalMod polynomial split overflow");
        }
        next_power <<= 1U;
    }

    auto split = split_coeffs(node->polynomial, static_cast<int>(next_power));
    node->split_degree = next_power;
    node->quotient = build_eval_mod_split_tree(
        std::move(std::get<0>(split)),
        log_split,
        log_degree,
        resplit_lead_leaf);
    node->remainder = build_eval_mod_split_tree(
        std::move(std::get<1>(split)),
        log_split,
        log_degree,
        resplit_lead_leaf);
    return node;
}

std::unique_ptr<EvalModSplitNode> build_eval_mod_flat_split_tree(
    Polynomial polynomial,
    std::uint32_t log_split)
{
    if (polynomial.data().empty())
    {
        throw std::invalid_argument("EvalMod flat polynomial is empty");
    }
    if (log_split >= 31)
    {
        throw std::invalid_argument("EvalMod flat split is too large");
    }

    auto node = std::make_unique<EvalModSplitNode>();
    node->polynomial = std::move(polynomial);
    const auto degree = static_cast<std::uint32_t>(node->polynomial.degree());
    const std::uint32_t leaf_degree = 1U << log_split;
    if (degree < leaf_degree)
    {
        return node;
    }

    /*
     * Peel the highest complete baby-width block. For degree 58 and b=8,
     * recursive calls split at T56,T48,...,T8 and create
     * P=L0+L1*T8+...+L7*T56. split_coeffs applies the reflected lower-degree
     * Chebyshev corrections, so this is algebraically the same polynomial.
     */
    const auto split_degree = (degree / leaf_degree) * leaf_degree;
    if (split_degree == 0 || split_degree >
        static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument("EvalMod flat split degree is invalid");
    }
    auto split = split_coeffs(
        node->polynomial,
        static_cast<int>(split_degree));
    node->split_degree = split_degree;
    node->quotient = build_eval_mod_flat_split_tree(
        std::move(std::get<0>(split)),
        log_split);
    node->remainder = build_eval_mod_flat_split_tree(
        std::move(std::get<1>(split)),
        log_split);
    return node;
}

void collect_eval_mod_leaves(
    EvalModSplitNode &node,
    std::vector<EvalModSplitNode *> &leaves)
{
    if (node.is_leaf())
    {
        node.node_id = static_cast<std::uint32_t>(leaves.size());
        leaves.push_back(&node);
        return;
    }
    collect_eval_mod_leaves(*node.quotient, leaves);
    collect_eval_mod_leaves(*node.remainder, leaves);
}

std::uint32_t schedule_eval_mod_combines(
    EvalModSplitNode &node,
    std::uint32_t &next_node_id,
    std::vector<GpuEvalModPolynomialCombineStep> &steps)
{
    if (node.is_leaf())
    {
        return node.node_id;
    }

    const auto quotient_node = schedule_eval_mod_combines(
        *node.quotient,
        next_node_id,
        steps);
    const auto remainder_node = schedule_eval_mod_combines(
        *node.remainder,
        next_node_id,
        steps);
    node.node_id = next_node_id++;
    steps.push_back(GpuEvalModPolynomialCombineStep{
        node.node_id,
        quotient_node,
        remainder_node,
        node.split_degree});
    return node.node_id;
}

bool eval_mod_coefficient_is_nonzero(const std::complex<double> &value)
{
    return std::abs(value) > util::IsNegligibleThreshold;
}

bool env_flag_enabled(const char *name)
{
    const char *raw = std::getenv(name);
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

}  // namespace

GpuCiphertextData GpuUploader::upload_ciphertext(
    const Ciphertext &src,
    int device_id)
{
    if (!src.is_valid() || src.size() == 0)
    {
        return {};
    }

    std::size_t q_count = 0;
    std::size_t p_count = 0;
    ciphertext_limb_shape(src, q_count, p_count);

    GpuPolyShard shard;
    shard.limb_begin = 0;
    shard.limb_count = q_count + p_count;
    shard.coeff_begin = 0;
    shard.coeff_count = src.poly_modulus_degree();

    return upload_ciphertext(
        src,
        device_id,
        std::vector<GpuPolyShard>{shard});
}

GpuCiphertextData GpuUploader::upload_ciphertext(
    const Ciphertext &src,
    int device_id,
    const std::vector<GpuPolyShard> &shard_template)
{
    if (!src.is_valid() || src.size() == 0)
    {
        return {};
    }

    std::size_t q_count = 0;
    std::size_t p_count = 0;
    ciphertext_limb_shape(src, q_count, p_count);

    auto dst = GpuCiphertextData::allocate_single_device_sharded(
        src.poly_modulus_degree(),
        q_count,
        src.size(),
        device_id,
        shard_template,
        p_count);

    dst.meta.parms_id = src.parms_id();
    dst.meta.scale = src.scale();
    dst.meta.correction_factor = src.correction_factor();
    dst.meta.is_ntt_form = src.is_ntt_form();

    for (std::size_t component = 0; component < src.size(); ++component)
    {
        upload_ciphertext_component_shards(
            src,
            component,
            dst.polys_.at(component),
            dst.fields_,
            "GpuUploader only supports ciphertext residues that fit in GpuWord");
    }

    return dst;
}

void GpuUploader::download_ciphertext(
    const GpuCiphertextData &src,
    Ciphertext &dst)
{
    (void)src;
    (void)dst;
    throw std::invalid_argument(
        "GpuUploader::download_ciphertext requires PoseidonContext to rebuild CPU RNSPoly views");
}

void GpuUploader::download_ciphertext(
    const GpuCiphertextData &src,
    Ciphertext &dst,
    const PoseidonContext &context)
{
    if (src.empty())
    {
        dst.release();
        return;
    }

    const auto limb_count = src.meta.q_count + src.meta.p_count;
    dst.resize(context, src.meta.parms_id, src.meta.component_count);

    if (dst.poly_modulus_degree() != src.meta.degree ||
        dst.coeff_modulus_size() != limb_count ||
        dst.size() != src.meta.component_count)
    {
        throw std::invalid_argument(
            "GPU ciphertext shape does not match PoseidonContext target level");
    }

    dst.scale() = src.meta.scale;
    dst.correction_factor() = src.meta.correction_factor;
    dst.is_ntt_form() = src.meta.is_ntt_form;

    for (std::size_t component = 0; component < src.meta.component_count; ++component)
    {
        download_ciphertext_component_shards(
            src.polys_.at(component),
            src.fields_,
            dst.data(component));
    }
}

GpuPlaintextData GpuUploader::upload_plaintext(
    const Plaintext &src,
    int device_id)
{
    if (!src.is_valid())
    {
        return {};
    }

    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;
    plaintext_limb_shape(src, degree, q_count, p_count);

    auto dst = GpuPlaintextData::allocate_single_device(
        degree,
        q_count,
        device_id,
        p_count);

    dst.meta.parms_id = src.parms_id();
    dst.meta.scale = src.scale();
    dst.meta.is_ntt_form = src.is_ntt_form();

    const auto word_count = checked_mul(
        degree,
        q_count + p_count,
        "plaintext size overflow");
    if (word_count != src.coeff_count())
    {
        throw std::invalid_argument("plaintext coefficient count does not match GPU shape");
    }

    copy_uint64_to_device_field(
        src.data(),
        word_count,
        dst.fields_[0],
        "GpuUploader only supports plaintext coefficients that fit in GpuWord");

    return dst;
}

void GpuUploader::download_plaintext(
    const GpuPlaintextData &src,
    Plaintext &dst)
{
    if (src.empty())
    {
        dst.release();
        return;
    }

    if (src.meta.is_ntt_form || src.meta.parms_id != parms_id_zero)
    {
        throw std::invalid_argument(
            "GpuUploader::download_plaintext requires PoseidonContext for NTT plaintext");
    }

    const auto word_count = src.fields_[0].size();
    dst.resize(word_count);
    dst.parms_id() = parms_id_zero;
    dst.scale() = src.meta.scale;

    copy_device_field_to_uint64(src.fields_[0], dst.data(), word_count);
}

void GpuUploader::download_plaintext(
    const GpuPlaintextData &src,
    Plaintext &dst,
    const PoseidonContext &context)
{
    if (src.empty())
    {
        dst.release();
        return;
    }

    const auto word_count = checked_mul(
        src.meta.degree,
        src.meta.q_count + src.meta.p_count,
        "plaintext size overflow");

    if (src.meta.is_ntt_form || src.meta.parms_id != parms_id_zero)
    {
        dst.resize(context, src.meta.parms_id, word_count);
    }
    else
    {
        dst.resize(word_count);
        dst.parms_id() = parms_id_zero;
    }

    dst.scale() = src.meta.scale;
    copy_device_field_to_uint64(src.fields_[0], dst.data(), word_count);
}

GpuMatrixPlain GpuUploader::upload_matrix_plain(
    const MatrixPlain &src,
    int device_id)
{
    GpuMatrixPlain dst;
    dst.log_slots = src.log_slots;
    dst.n1 = src.n1;
    dst.level = src.level;
    dst.scale = src.scale;
    dst.rot_index = src.rot_index;

    for (const auto &entry : src.plain_vec)
    {
        dst.plain_vec.emplace(
            entry.first,
            upload_plaintext(entry.second, device_id));
    }

    return dst;
}

GpuLinearMatrixGroup GpuUploader::upload_linear_matrix_group(
    const LinearMatrixGroup &src,
    int device_id)
{
    GpuLinearMatrixGroup dst;
    dst.rot_index() = src.rot_index();
    dst.set_step(src.step());
    dst.set_rescale_min_scale(src.rescale_min_scale());
    dst.rescale_counts() = src.rescale_counts();
    dst.data().reserve(src.data().size());

    for (const auto &matrix : src.data())
    {
        dst.data().push_back(upload_matrix_plain(matrix, device_id));
    }

    return dst;
}

namespace
{

struct ExactQpPlaintextHost
{
    parms_id_type parms_id{};
    double scale = 1.0;
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;
    std::vector<std::uint64_t> values;
};

ExactQpPlaintextHost make_qp_plaintext_exact_host(
    const Plaintext &src,
    const PoseidonContext &context)
{
    if (!src.is_ntt_form())
    {
        throw std::invalid_argument(
            "upload_qp_plaintext_exact: plaintext must be in NTT form");
    }
    const auto context_data =
        context.crt_context()->get_context_data(src.parms_id());
    if (!context_data)
    {
        throw std::invalid_argument(
            "upload_qp_plaintext_exact: plaintext parms_id is absent from context");
    }
    const auto *rns_qp = context_data->qp_rns_tool();
    if (rns_qp == nullptr || rns_qp->base_q() == nullptr ||
        rns_qp->base_p() == nullptr)
    {
        throw std::invalid_argument(
            "upload_qp_plaintext_exact: QP RNS tool is unavailable");
    }

    const std::size_t degree = context_data->parms().degree();
    const std::size_t q_count = rns_qp->base_q()->size();
    const std::size_t p_count = rns_qp->base_p()->size();
    if (src.coeff_count() != q_count * degree || p_count == 0)
    {
        throw std::invalid_argument(
            "upload_qp_plaintext_exact: plaintext shape mismatch");
    }

    std::vector<std::uint64_t> qp((q_count + p_count) * degree, 0);
    std::copy_n(src.data(), q_count * degree, qp.data());

    const auto *ntt_tables = context.crt_context()->small_ntt_tables();
    if (ntt_tables == nullptr)
    {
        throw std::invalid_argument(
            "upload_qp_plaintext_exact: CPU NTT tables are unavailable");
    }
    for (std::size_t limb = 0; limb < q_count; ++limb)
    {
        util::inverse_ntt_negacyclic_harvey(
            qp.data() + limb * degree,
            ntt_tables[limb]);
    }

    /*
     * Matrix coefficients are signed CKKS integers. A generic fast Q->P
     * converter may select x+kQ and therefore does not preserve the centered
     * representative needed by delayed ModDown. Compose once during setup,
     * center in (-Q/2,Q/2], and reduce that exact signed integer into every P
     * limb.
     */
    std::vector<std::uint64_t> composed(
        qp.begin(),
        qp.begin() + q_count * degree);
    rns_qp->base_q()->compose_array(
        composed.data(),
        degree,
        MemoryManager::GetPool());
    std::vector<std::uint64_t> magnitude(q_count);
    for (std::size_t coeff = 0; coeff < degree; ++coeff)
    {
        const std::uint64_t *value =
            composed.data() + coeff * q_count;
        const bool negative = util::is_greater_than_or_equal_uint(
            value,
            context_data->upper_half_threshold(),
            q_count);
        const std::uint64_t *unsigned_value = value;
        if (negative)
        {
            util::sub_uint(
                context_data->total_coeff_modulus(),
                value,
                q_count,
                magnitude.data());
            unsigned_value = magnitude.data();
        }

        for (std::size_t p_limb = 0; p_limb < p_count; ++p_limb)
        {
            const auto &modulus = (*rns_qp->base_p())[p_limb];
            const std::uint64_t residue = util::modulo_uint(
                unsigned_value,
                q_count,
                modulus);
            qp[(q_count + p_limb) * degree + coeff] =
                negative && residue != 0
                    ? modulus.value() - residue
                    : residue;
        }
    }

    for (std::size_t limb = 0; limb < q_count; ++limb)
    {
        util::ntt_negacyclic_harvey(
            qp.data() + limb * degree,
            ntt_tables[limb]);
    }
    const auto key_context = context.crt_context()->key_context_data();
    if (!key_context)
    {
        throw std::invalid_argument(
            "upload_qp_plaintext_exact: key context is unavailable");
    }
    const std::size_t p_table_offset = key_context->parms().q().size();
    for (std::size_t limb = 0; limb < p_count; ++limb)
    {
        util::ntt_negacyclic_harvey(
            qp.data() + (q_count + limb) * degree,
            ntt_tables[p_table_offset + limb]);
    }

    ExactQpPlaintextHost result;
    result.parms_id = src.parms_id();
    result.scale = src.scale();
    result.degree = degree;
    result.q_count = q_count;
    result.p_count = p_count;
    result.values = std::move(qp);
    return result;
}

GpuPlaintextData upload_qp_plaintext_exact(
    const Plaintext &src,
    const PoseidonContext &context,
    int device_id)
{
    auto host = make_qp_plaintext_exact_host(src, context);
    auto result = GpuPlaintextData::allocate_single_device(
        host.degree,
        host.q_count,
        device_id,
        host.p_count);
    result.meta.parms_id = src.parms_id();
    result.meta.scale = src.scale();
    result.meta.is_ntt_form = true;
    copy_uint64_to_device_field(
        host.values.data(),
        host.values.size(),
        result.fields_.front(),
        "QP plaintext residue exceeds GpuWord");
    return result;
}

std::vector<std::uint32_t> bit_reversed_indices(std::size_t degree)
{
    if (degree == 0 || (degree & (degree - 1)) != 0 ||
        degree > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::invalid_argument(
            "compressed QP plaintext requires a power-of-two degree");
    }
    int log_degree = 0;
    for (std::size_t value = degree; value > 1; value >>= 1)
    {
        ++log_degree;
    }
    std::vector<std::uint32_t> result(degree);
    for (std::size_t coefficient = 0; coefficient < degree; ++coefficient)
    {
        result[coefficient] = util::reverse_bits(
            static_cast<std::uint32_t>(coefficient),
            log_degree);
    }
    return result;
}

bool exact_qp_has_period(
    const ExactQpPlaintextHost &host,
    std::size_t period,
    const std::vector<std::uint32_t> &bit_reversed)
{
    if (period == 0 || period > host.degree ||
        (period & (period - 1)) != 0 ||
        bit_reversed.size() != host.degree)
    {
        return false;
    }
    std::vector<std::uint64_t> representatives(period);
    std::vector<bool> initialized(period);
    const std::size_t period_mask = period - 1;
    for (std::size_t limb = 0;
         limb < host.q_count + host.p_count;
         ++limb)
    {
        std::fill(initialized.begin(), initialized.end(), false);
        const auto *values = host.values.data() + limb * host.degree;
        for (std::size_t coefficient = 0;
             coefficient < host.degree;
             ++coefficient)
        {
            const std::size_t compact_index =
                period == host.degree
                    ? coefficient
                    : bit_reversed[coefficient] & period_mask;
            if (!initialized[compact_index])
            {
                representatives[compact_index] = values[coefficient];
                initialized[compact_index] = true;
            }
            else if (representatives[compact_index] != values[coefficient])
            {
                return false;
            }
        }
    }
    return true;
}

GpuCompressedPlaintextQP upload_qp_plaintext_exact_compressed(
    const Plaintext &src,
    const PoseidonContext &context,
    int device_id)
{
    auto host = make_qp_plaintext_exact_host(src, context);
    const auto bit_reversed = bit_reversed_indices(host.degree);
    std::size_t period = 1;
    while (period < host.degree &&
           !exact_qp_has_period(host, period, bit_reversed))
    {
        period <<= 1;
    }
    if (!exact_qp_has_period(host, period, bit_reversed))
    {
        throw std::logic_error(
            "compressed QP plaintext period did not reconstruct exactly");
    }

    const std::size_t limb_count = host.q_count + host.p_count;
    std::vector<GpuWord> compact(limb_count * period);
    std::vector<bool> initialized(period);
    const std::size_t period_mask = period - 1;
    for (std::size_t limb = 0; limb < limb_count; ++limb)
    {
        std::fill(initialized.begin(), initialized.end(), false);
        for (std::size_t coefficient = 0;
             coefficient < host.degree;
             ++coefficient)
        {
            const std::size_t compact_index =
                period == host.degree
                    ? coefficient
                    : bit_reversed[coefficient] & period_mask;
            if (!initialized[compact_index])
            {
                const auto value =
                    host.values[limb * host.degree + coefficient];
                if (value > std::numeric_limits<GpuWord>::max())
                {
                    throw std::overflow_error(
                        "compressed QP plaintext residue exceeds GpuWord");
                }
                compact[limb * period + compact_index] =
                    static_cast<GpuWord>(value);
                initialized[compact_index] = true;
            }
        }
    }

    GpuCompressedPlaintextQP result;
    result.meta.parms_id = host.parms_id;
    result.meta.scale = host.scale;
    result.meta.is_ntt_form = true;
    result.meta.degree = host.degree;
    result.meta.q_count = host.q_count;
    result.meta.p_count = host.p_count;
    result.period = period;
    result.residues.allocate(compact.size(), device_id);
    result.residues.copy_from_host(compact.data(), compact.size());

    // Compression is a setup-time, exact representation change.  Read the
    // actual device allocation back once and require every Q/P residue to
    // reconstruct bit-for-bit before the object can enter an experimental
    // compute plan.
    std::vector<GpuWord> device_compact(compact.size());
    result.residues.copy_to_host(
        device_compact.data(),
        device_compact.size());
    bool exact = true;
    for (std::size_t limb = 0; limb < limb_count && exact; ++limb)
    {
        for (std::size_t coefficient = 0;
             coefficient < host.degree;
             ++coefficient)
        {
            const std::size_t compact_index =
                period == host.degree
                    ? coefficient
                    : bit_reversed[coefficient] & period_mask;
            if (static_cast<std::uint64_t>(
                    device_compact[limb * period + compact_index]) !=
                host.values[limb * host.degree + coefficient])
            {
                exact = false;
                break;
            }
        }
    }
    result.exact_device_reconstruction = exact;
    if (!exact)
    {
        throw std::runtime_error(
            "compressed QP device round-trip was not bit-exact");
    }
    return result;
}

GpuDoubleHoistMatrixPlan make_double_hoist_plan(
    const MatrixPlain &src,
    const std::map<int, GpuPlaintextData> &plain_vec_qp,
    const std::map<int, GpuCompressedPlaintextQP> &compressed_plain_vec_qp,
    int device_id,
    std::uint32_t rescale_count)
{
    const bool use_compressed = !compressed_plain_vec_qp.empty();
    if (use_compressed == !plain_vec_qp.empty())
    {
        throw std::invalid_argument(
            "make_double_hoist_plan: exactly one plaintext layout is required");
    }

    GpuDoubleHoistMatrixPlan plan;
    plan.log_slots = src.log_slots;
    plan.n1 = src.n1;
    plan.rescale_count = std::max(rescale_count, std::uint32_t{1});
    plan.compressed_plaintexts = use_compressed;

    const auto [index, unused_giant_steps, baby_steps] =
        poseidon::bsgs_index(
            src.plain_vec,
            1 << src.log_slots,
            static_cast<int>(src.n1));
    (void)unused_giant_steps;
    plan.baby_steps = baby_steps;
    plan.giant_steps.reserve(index.size());
    plan.group_term_offsets.reserve(index.size() + 1);
    plan.group_term_offsets.push_back(0);

    std::map<int, std::uint32_t> baby_ids;
    for (std::size_t i = 0; i < plan.baby_steps.size(); ++i)
    {
        baby_ids.emplace(
            plan.baby_steps[i],
            static_cast<std::uint32_t>(i));
    }

    std::vector<const GpuWord *> q_ptrs;
    std::vector<const GpuWord *> p_ptrs;
    std::vector<std::uint32_t> diagonal_periods;
    std::vector<std::uint32_t> term_baby_indices;
    for (const auto &group : index)
    {
        const auto giant_index =
            static_cast<std::uint32_t>(plan.giant_steps.size());
        plan.giant_steps.push_back(group.first);
        for (const int baby_step : group.second)
        {
            const int diagonal = group.first + baby_step;
            const auto baby_it = baby_ids.find(baby_step);
            if (baby_it == baby_ids.end())
            {
                throw std::logic_error(
                    "make_double_hoist_plan: incomplete BSGS schedule");
            }

            if (use_compressed)
            {
                const auto plain_it =
                    compressed_plain_vec_qp.find(diagonal);
                if (plain_it == compressed_plain_vec_qp.end())
                {
                    throw std::logic_error(
                        "make_double_hoist_plan: incomplete compressed BSGS schedule");
                }
                const auto &plain = plain_it->second;
                if (!plain.exact_device_reconstruction ||
                    plain.meta.p_count == 0 || plain.period == 0 ||
                    plain.period > plain.meta.degree ||
                    (plain.period & (plain.period - 1)) != 0 ||
                    plain.period >
                        std::numeric_limits<std::uint32_t>::max())
                {
                    throw std::invalid_argument(
                        "make_double_hoist_plan: invalid compressed QP diagonal");
                }
                q_ptrs.push_back(plain.residues.data());
                p_ptrs.push_back(
                    plain.residues.data() +
                    plain.meta.q_count * plain.period);
                diagonal_periods.push_back(
                    static_cast<std::uint32_t>(plain.period));
            }
            else
            {
                const auto plain_it = plain_vec_qp.find(diagonal);
                if (plain_it == plain_vec_qp.end())
                {
                    throw std::logic_error(
                        "make_double_hoist_plan: incomplete BSGS schedule");
                }
                const auto view = plain_it->second.make_const_view();
                if (view.poly.shards.size() != 1 ||
                    view.meta.p_count == 0 ||
                    view.meta.degree >
                        std::numeric_limits<std::uint32_t>::max())
                {
                    throw std::invalid_argument(
                        "make_double_hoist_plan: QP diagonal must have one shard");
                }
                const auto &shard = view.poly.shards.front();
                q_ptrs.push_back(shard.ptr);
                p_ptrs.push_back(
                    shard.ptr + view.meta.q_count * view.meta.degree);
                diagonal_periods.push_back(
                    static_cast<std::uint32_t>(view.meta.degree));
            }
            term_baby_indices.push_back(baby_it->second);
            plan.terms.push_back(GpuDoubleHoistTerm{
                giant_index,
                baby_it->second,
                static_cast<std::uint32_t>(plan.terms.size())});
        }
        plan.group_term_offsets.push_back(
            static_cast<std::uint32_t>(plan.terms.size()));
    }
    plan.n2 = static_cast<std::uint32_t>(plan.giant_steps.size());
    if (plan.terms.empty() || plan.n2 == 0)
    {
        throw std::invalid_argument(
            "make_double_hoist_plan: empty diagonal schedule");
    }

    plan.diagonal_q_ptrs.allocate(q_ptrs.size(), device_id);
    plan.diagonal_p_ptrs.allocate(p_ptrs.size(), device_id);
    plan.diagonal_periods.allocate(diagonal_periods.size(), device_id);
    plan.term_baby_indices.allocate(term_baby_indices.size(), device_id);
    plan.group_term_offsets_device.allocate(
        plan.group_term_offsets.size(),
        device_id);
    plan.diagonal_q_ptrs.copy_from_host(q_ptrs.data(), q_ptrs.size());
    plan.diagonal_p_ptrs.copy_from_host(p_ptrs.data(), p_ptrs.size());
    plan.diagonal_periods.copy_from_host(
        diagonal_periods.data(),
        diagonal_periods.size());
    plan.term_baby_indices.copy_from_host(
        term_baby_indices.data(),
        term_baby_indices.size());
    plan.group_term_offsets_device.copy_from_host(
        plan.group_term_offsets.data(),
        plan.group_term_offsets.size());
    return plan;
}

}  // namespace

GpuLinearMatrixGroupQP GpuUploader::upload_linear_matrix_group_qp(
    const LinearMatrixGroup &src,
    const PoseidonContext &context,
    int device_id,
    std::uint32_t rescale_count,
    bool compress_plaintexts)
{
    GpuLinearMatrixGroupQP result;
    result.rot_index() = src.rot_index();
    result.set_step(src.step());
    result.set_rescale_min_scale(src.rescale_min_scale());
    result.rescale_counts() = src.rescale_counts();
    result.data().reserve(src.data().size());

    for (const auto &matrix : src.data())
    {
        GpuMatrixPlainQP uploaded;
        uploaded.log_slots = matrix.log_slots;
        uploaded.n1 = matrix.n1;
        uploaded.level = matrix.level;
        uploaded.scale = matrix.scale;
        uploaded.rot_index = matrix.rot_index;
        for (const auto &entry : matrix.plain_vec)
        {
            if (compress_plaintexts)
            {
                uploaded.compressed_plain_vec_qp.emplace(
                    entry.first,
                    upload_qp_plaintext_exact_compressed(
                        entry.second,
                        context,
                        device_id));
            }
            else
            {
                uploaded.plain_vec_qp.emplace(
                    entry.first,
                    upload_qp_plaintext_exact(
                        entry.second,
                        context,
                        device_id));
            }
        }
        uploaded.plan = make_double_hoist_plan(
            matrix,
            uploaded.plain_vec_qp,
            uploaded.compressed_plain_vec_qp,
            device_id,
            rescale_count);
        result.data().push_back(std::move(uploaded));
    }
    return result;
}

void GpuUploader::restrict_double_hoist_giant_groups(
    GpuMatrixPlainQP &matrix,
    std::size_t group_begin,
    std::size_t group_end)
{
    const auto &source = matrix.plan;
    const std::size_t group_count = source.giant_steps.size();
    if (group_begin >= group_end || group_end > group_count)
    {
        throw std::invalid_argument(
            "GpuUploader::restrict_double_hoist_giant_groups: invalid group range");
    }
    if (source.n2 != group_count || source.terms.empty() ||
        source.group_term_offsets.size() != group_count + 1 ||
        source.diagonal_q_ptrs.size() != source.terms.size() ||
        source.diagonal_p_ptrs.size() != source.terms.size() ||
        source.diagonal_periods.size() != source.terms.size() ||
        source.term_baby_indices.size() != source.terms.size())
    {
        throw std::invalid_argument(
            "GpuUploader::restrict_double_hoist_giant_groups: incomplete source plan");
    }

    const int device_id = source.diagonal_q_ptrs.device_id();
    std::vector<const GpuWord *> source_q_ptrs(source.terms.size());
    std::vector<const GpuWord *> source_p_ptrs(source.terms.size());
    std::vector<std::uint32_t> source_periods(source.terms.size());
    std::vector<std::uint32_t> source_baby_indices(source.terms.size());
    source.diagonal_q_ptrs.copy_to_host(
        source_q_ptrs.data(), source_q_ptrs.size());
    source.diagonal_p_ptrs.copy_to_host(
        source_p_ptrs.data(), source_p_ptrs.size());
    source.diagonal_periods.copy_to_host(
        source_periods.data(), source_periods.size());
    source.term_baby_indices.copy_to_host(
        source_baby_indices.data(), source_baby_indices.size());

    GpuDoubleHoistMatrixPlan restricted;
    restricted.log_slots = source.log_slots;
    restricted.n1 = source.n1;
    restricted.rescale_count = source.rescale_count;
    restricted.baby_steps = source.baby_steps;
    restricted.compressed_plaintexts = source.compressed_plaintexts;
    restricted.giant_steps.reserve(group_end - group_begin);
    restricted.group_term_offsets.reserve(group_end - group_begin + 1);
    restricted.group_term_offsets.push_back(0);

    std::vector<const GpuWord *> q_ptrs;
    std::vector<const GpuWord *> p_ptrs;
    std::vector<std::uint32_t> periods;
    std::vector<std::uint32_t> baby_indices;
    for (std::size_t group = group_begin; group < group_end; ++group)
    {
        const std::uint32_t restricted_group =
            static_cast<std::uint32_t>(restricted.giant_steps.size());
        restricted.giant_steps.push_back(source.giant_steps[group]);
        const std::size_t term_begin = source.group_term_offsets[group];
        const std::size_t term_end = source.group_term_offsets[group + 1];
        if (term_begin >= term_end || term_end > source.terms.size())
        {
            throw std::invalid_argument(
                "GpuUploader::restrict_double_hoist_giant_groups: invalid term range");
        }
        for (std::size_t term = term_begin; term < term_end; ++term)
        {
            const std::uint32_t restricted_term =
                static_cast<std::uint32_t>(restricted.terms.size());
            const std::uint32_t baby_index = source_baby_indices[term];
            q_ptrs.push_back(source_q_ptrs[term]);
            p_ptrs.push_back(source_p_ptrs[term]);
            periods.push_back(source_periods[term]);
            baby_indices.push_back(baby_index);
            restricted.terms.push_back(GpuDoubleHoistTerm{
                restricted_group, baby_index, restricted_term});
        }
        restricted.group_term_offsets.push_back(
            static_cast<std::uint32_t>(restricted.terms.size()));
    }
    restricted.n2 =
        static_cast<std::uint32_t>(restricted.giant_steps.size());

    restricted.diagonal_q_ptrs.allocate(q_ptrs.size(), device_id);
    restricted.diagonal_p_ptrs.allocate(p_ptrs.size(), device_id);
    restricted.diagonal_periods.allocate(periods.size(), device_id);
    restricted.term_baby_indices.allocate(baby_indices.size(), device_id);
    restricted.group_term_offsets_device.allocate(
        restricted.group_term_offsets.size(), device_id);
    restricted.diagonal_q_ptrs.copy_from_host(q_ptrs.data(), q_ptrs.size());
    restricted.diagonal_p_ptrs.copy_from_host(p_ptrs.data(), p_ptrs.size());
    restricted.diagonal_periods.copy_from_host(periods.data(), periods.size());
    restricted.term_baby_indices.copy_from_host(
        baby_indices.data(), baby_indices.size());
    restricted.group_term_offsets_device.copy_from_host(
        restricted.group_term_offsets.data(),
        restricted.group_term_offsets.size());
    matrix.plan = std::move(restricted);
}

GpuBootstrapData::EvalModData GpuUploader::upload_eval_mod_high_precision(
    const EvalModPoly &eval_mod_poly,
    const CKKSEncoder &encoder,
    parms_id_type input_parms_id,
    int device_id,
    GpuRelinKeysData *relin_keys,
    parms_id_type expected_output_parms_id,
    std::uint32_t logical_rescale_count,
    const Polynomial *polynomial_override,
    bool include_input_offset,
    std::uint32_t double_angle_override,
    double double_angle_base_override,
    double polynomial_output_scale_override,
    bool fuse_leaf_terms_before_rescale,
    double input_scale,
    bool metadata_only,
    const GpuEvalModUploadOptions &options)
{
    const auto &context = encoder.context();
    const auto crt_context = context.crt_context();
    if (!crt_context)
    {
        throw std::invalid_argument(
            "GpuUploader::upload_eval_mod_high_precision: empty CRT context");
    }
    const auto input_context_data =
        crt_context->get_context_data(input_parms_id);
    if (!input_context_data)
    {
        throw std::invalid_argument(
            "GpuUploader::upload_eval_mod_high_precision: invalid input parms_id");
    }

    const std::size_t input_q_count =
        input_context_data->coeff_modulus().size();
    if (logical_rescale_count == 0)
    {
        throw std::invalid_argument(
            "GpuUploader::upload_eval_mod_high_precision: logical_rescale_count must be positive");
    }
    if (input_q_count <= logical_rescale_count)
    {
        throw std::invalid_argument(
            "GpuUploader::upload_eval_mod_high_precision: insufficient input modulus chain");
    }
    if (eval_mod_poly.level_start() >= 0 &&
        static_cast<std::size_t>(eval_mod_poly.level_start()) !=
            input_context_data->level())
    {
        throw std::invalid_argument(
            "GpuUploader::upload_eval_mod_high_precision: EvalMod level_start does not match input parms_id");
    }

    const double target_scale = eval_mod_poly.scaling_factor();
    if (!(target_scale > 0.0) ||
        !std::isfinite(target_scale))
    {
        throw std::invalid_argument(
            "GpuUploader::upload_eval_mod_high_precision: invalid scaling factor");
    }
    const bool dynamic_rescale = options.dynamic_rescale.has_value()
        ? *options.dynamic_rescale
        : env_flag_enabled("POSEIDON_BOOTSTRAP_EVALMOD_DYNAMIC_RESCALE");
    const double effective_input_scale = input_scale > 0.0
        ? input_scale
        : target_scale;
    if (!std::isfinite(effective_input_scale))
    {
        throw std::invalid_argument(
            "GpuUploader::upload_eval_mod_high_precision: invalid input scale");
    }

    auto parms_id_for_q_count =
        [&](std::size_t q_count) -> parms_id_type {
            if (q_count == 0 ||
                q_count - 1 > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::invalid_argument(
                    "GpuUploader::upload_eval_mod_high_precision: invalid q_count");
            }
            const auto id_map = crt_context->parms_id_map();
            const auto iter = id_map.find(
                static_cast<std::uint32_t>(q_count - 1));
            if (iter == id_map.end())
            {
                throw std::invalid_argument(
                    "GpuUploader::upload_eval_mod_high_precision: modulus level is absent from context");
            }
            const auto level_data = crt_context->get_context_data(iter->second);
            if (!level_data || level_data->coeff_modulus().size() != q_count)
            {
                throw std::invalid_argument(
                    "GpuUploader::upload_eval_mod_high_precision: q_count/parms_id mismatch");
            }
            return iter->second;
        };

    auto encode_and_upload =
        [&](std::complex<double> value,
            std::size_t q_count,
            double scale) -> GpuPlaintextData {
            if (metadata_only)
            {
                /*
                 * Scale-chain search needs the exact EvalMod DAG and level
                 * transitions, but none of the coefficient payloads.  Keep
                 * enough metadata for diagnostics while avoiding N=65536
                 * plaintext encoding and device allocation for every search
                 * candidate.  The default runtime path never sets this flag.
                 */
                GpuPlaintextData result;
                result.meta.parms_id = parms_id_for_q_count(q_count);
                result.meta.scale = scale;
                result.meta.is_ntt_form = true;
                result.meta.degree =
                    std::size_t{1} << context.parameters_literal()->log_n();
                result.meta.q_count = q_count;
                return result;
            }
            Plaintext plaintext;
            try
            {
                encoder.encode(
                    value,
                    parms_id_for_q_count(q_count),
                    scale,
                    plaintext);
            }
            catch (const std::exception &ex)
            {
                throw std::invalid_argument(
                    std::string(
                        "GpuUploader::upload_eval_mod_high_precision: encode failed: ") +
                    ex.what() +
                    ", q_count=" + std::to_string(q_count) +
                    ", log2(scale)=" +
                    std::to_string(std::log2(scale)) +
                    ", value_abs=" +
                    std::to_string(std::abs(value)));
            }
            return upload_plaintext(plaintext, device_id);
        };

    auto rescale_modulus_product =
        [&](std::size_t q_count, std::uint32_t rescale_count) -> double {
            if (rescale_count == 0 || q_count <= rescale_count)
            {
                throw std::invalid_argument(
                    "GpuUploader::upload_eval_mod_high_precision: invalid logical rescale span");
            }
            const auto level_data = crt_context->get_context_data(
                parms_id_for_q_count(q_count));
            if (!level_data ||
                level_data->coeff_modulus().size() != q_count)
            {
                throw std::invalid_argument(
                    "GpuUploader::upload_eval_mod_high_precision: missing rescale modulus");
            }
            long double product = 1.0L;
            const auto &moduli = level_data->coeff_modulus();
            for (std::uint32_t index = 0; index < rescale_count; ++index)
            {
                product *= static_cast<long double>(
                    moduli[q_count - 1 - index].value());
            }
            const double result = static_cast<double>(product);
            if (!(result > 0.0) || !std::isfinite(result))
            {
                throw std::invalid_argument(
                    "GpuUploader::upload_eval_mod_high_precision: invalid rescale modulus product");
            }
            return result;
        };

    auto choose_dynamic_rescale =
        [&](std::size_t q_count, double input_scale) -> std::uint32_t {
            const auto level_data = crt_context->get_context_data(
                parms_id_for_q_count(q_count));
            if (!level_data ||
                level_data->coeff_modulus().size() != q_count)
            {
                throw std::invalid_argument(
                    "GpuUploader::upload_eval_mod_high_precision: missing dynamic rescale modulus");
            }
            std::vector<std::uint64_t> active_moduli;
            active_moduli.reserve(q_count);
            for (const auto &modulus : level_data->coeff_modulus())
            {
                active_moduli.push_back(modulus.value());
            }
            return plan_gpu_dynamic_rescale(
                       input_scale,
                       target_scale,
                       active_moduli,
                       /*require_rescale=*/true)
                .rescale_count;
        };

    auto require_valid_scale = [](double scale, const char *what) {
        if (!(scale > 0.0) || !std::isfinite(scale))
        {
            throw std::invalid_argument(
                std::string("GpuUploader::upload_eval_mod_high_precision: invalid ") + what);
        }
    };

    const auto &sine_polynomial = polynomial_override != nullptr
        ? *polynomial_override
        : eval_mod_poly.sine_poly();
    if (sine_polynomial.data().empty())
    {
        throw std::invalid_argument(
            "GpuUploader::upload_eval_mod_high_precision: empty sine polynomial");
    }
    const auto polynomial_degree =
        static_cast<std::uint32_t>(sine_polynomial.degree());
    std::uint32_t log_degree = 0;
    std::uint32_t degree_bound = 1;
    while (degree_bound < std::max(polynomial_degree, 1U))
    {
        if (degree_bound >
            (std::numeric_limits<std::uint32_t>::max() >> 1U))
        {
            throw std::overflow_error(
                "GpuUploader::upload_eval_mod_high_precision: polynomial degree overflow");
        }
        degree_bound <<= 1U;
        ++log_degree;
    }
    const auto default_log_split =
        static_cast<std::uint32_t>(optimal_split(static_cast<int>(log_degree)));
    const auto log_split = options.polynomial_log_split.has_value()
        ? *options.polynomial_log_split
        : evalmod_log_split_or(default_log_split, log_degree);
    if (log_split == 0 || log_split > log_degree)
    {
        throw std::invalid_argument(
            "GpuUploader::upload_eval_mod_high_precision: polynomial_log_split "
            "must be in [1, log_degree]");
    }
    const bool flat_bsgs_b8 = options.flat_bsgs_b8.has_value()
        ? *options.flat_bsgs_b8
        : use_evalmod_flat_bsgs_b8();
    const bool virtual_degree_bound =
        dynamic_rescale &&
        (options.virtual_degree_bound.has_value()
             ? *options.virtual_degree_bound
             : use_evalmod_virtual_degree_bound());
    if (flat_bsgs_b8 &&
        (log_split != 3 || !dynamic_rescale ||
         sine_polynomial.basis_type() != Chebyshev ||
         (polynomial_degree != 58 && polynomial_degree != 59)))
    {
        throw std::invalid_argument(
            "POSEIDON_EVALMOD_FLAT_BSGS_B8 requires dynamic Chebyshev "
            "degree 58/59 with log_split=3");
    }

    auto split_tree = flat_bsgs_b8
        ? build_eval_mod_flat_split_tree(sine_polynomial, log_split)
        : build_eval_mod_split_tree(
              sine_polynomial,
              log_split,
              log_degree,
              !dynamic_rescale ||
                  (options.lead_leaf_resplit.has_value()
                       ? *options.lead_leaf_resplit
                       : use_evalmod_lead_leaf_resplit()));
    std::vector<EvalModSplitNode *> leaves;
    collect_eval_mod_leaves(*split_tree, leaves);
    if (leaves.empty() ||
        leaves.size() > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::invalid_argument(
            "GpuUploader::upload_eval_mod_high_precision: invalid split tree");
    }

    GpuBootstrapData::EvalModData result;
    result.input_scale = effective_input_scale;
    result.target_scale = target_scale;
    result.logical_rescale_count = logical_rescale_count;
    result.dynamic_rescale = dynamic_rescale;
    result.dynamic_min_scale = dynamic_rescale ? target_scale : 0.0;
    result.polynomial_rescale_count = logical_rescale_count;
    result.rescale_polynomial_terms_individually =
        polynomial_override != nullptr &&
        !fuse_leaf_terms_before_rescale;
    result.polynomial_basis =
        sine_polynomial.basis_type() == Chebyshev
            ? GpuEvalModPolynomialBasis::Chebyshev
            : GpuEvalModPolynomialBasis::Monomial;
    result.polynomial_degree = polynomial_degree;
    result.polynomial_log_split = log_split;
    result.polynomial_flat_bsgs = flat_bsgs_b8;
    result.polynomial_degree_bound_virtual = virtual_degree_bound;

    std::uint32_t next_node_id = static_cast<std::uint32_t>(leaves.size());
    result.polynomial_result_node = schedule_eval_mod_combines(
        *split_tree,
        next_node_id,
        result.polynomial_combine_steps);

    std::vector<std::uint32_t> requested_degrees;
    for (const auto *leaf : leaves)
    {
        const auto &coefficients = leaf->polynomial.data();
        for (std::size_t degree = 1; degree < coefficients.size(); ++degree)
        {
            if (eval_mod_coefficient_is_nonzero(coefficients[degree]))
            {
                requested_degrees.push_back(static_cast<std::uint32_t>(degree));
            }
        }
    }
    for (const auto &combine : result.polynomial_combine_steps)
    {
        requested_degrees.push_back(combine.basis_degree);
    }
    if (dynamic_rescale && !virtual_degree_bound)
    {
        requested_degrees.push_back(degree_bound);
    }
    result.basis_steps = make_gpu_eval_mod_basis_plan(
        result.polynomial_basis,
        requested_degrees,
        flat_bsgs_b8 ? (1U << log_split) : 0U);

    std::map<std::uint32_t, std::size_t> basis_q_counts;
    std::map<std::uint32_t, double> basis_scales;
    basis_q_counts.emplace(0, input_q_count);
    basis_q_counts.emplace(1, input_q_count);
    basis_scales.emplace(1, effective_input_scale);
    std::set<std::size_t> required_relin_q_counts;
    for (auto &step : result.basis_steps)
    {
        const auto left_iter = basis_q_counts.find(step.left_degree);
        const auto right_iter = basis_q_counts.find(step.right_degree);
        const auto left_scale_iter = basis_scales.find(step.left_degree);
        const auto right_scale_iter = basis_scales.find(step.right_degree);
        if (left_iter == basis_q_counts.end() ||
            right_iter == basis_q_counts.end() ||
            left_scale_iter == basis_scales.end() ||
            right_scale_iter == basis_scales.end())
        {
            throw std::logic_error(
                "GpuUploader::upload_eval_mod_high_precision: basis plan is not topological");
        }
        const std::size_t multiply_q_count =
            std::min(left_iter->second, right_iter->second);
        if (multiply_q_count <= logical_rescale_count)
        {
            throw std::invalid_argument(
                "GpuUploader::upload_eval_mod_high_precision: modulus chain is too short for basis DAG");
        }
        required_relin_q_counts.insert(multiply_q_count);
        std::size_t output_q_count =
            multiply_q_count - logical_rescale_count;
        double multiply_left_scale = left_scale_iter->second;
        double multiply_right_scale = right_scale_iter->second;
        if (!dynamic_rescale && left_iter->second != right_iter->second)
        {
            const bool align_left = left_iter->second > right_iter->second;
            const std::size_t high_q_count = align_left
                ? left_iter->second
                : right_iter->second;
            const std::size_t low_q_count = align_left
                ? right_iter->second
                : left_iter->second;
            const double high_scale = align_left
                ? left_scale_iter->second
                : right_scale_iter->second;
            const double low_scale = align_left
                ? right_scale_iter->second
                : left_scale_iter->second;
            if (high_q_count <= low_q_count ||
                high_q_count - low_q_count >
                    std::numeric_limits<std::uint32_t>::max())
            {
                throw std::invalid_argument(
                    "GpuUploader::upload_eval_mod_high_precision: invalid basis operand alignment span");
            }
            const auto alignment_rescale_count =
                static_cast<std::uint32_t>(high_q_count - low_q_count);
            const double alignment_value =
                low_scale *
                rescale_modulus_product(high_q_count, alignment_rescale_count) /
                (high_scale * high_scale);
            require_valid_scale(alignment_value, "basis operand alignment value");
            step.align_left_operand = align_left;
            step.operand_alignment_rescale_count = alignment_rescale_count;
            step.operand_alignment_plaintext = encode_and_upload(
                {alignment_value, 0.0}, high_q_count, high_scale);
            step.operand_alignment_pre_rescale_scale =
                low_scale *
                rescale_modulus_product(high_q_count, alignment_rescale_count);
            step.operand_alignment_output_scale = low_scale;
            if (align_left)
            {
                multiply_left_scale = low_scale;
            }
            else
            {
                multiply_right_scale = low_scale;
            }
        }
        const double pre_rescale_scale =
            multiply_left_scale * multiply_right_scale;
        const std::uint32_t step_rescale_count = dynamic_rescale
            ? choose_dynamic_rescale(multiply_q_count, pre_rescale_scale)
            : logical_rescale_count;
        if (multiply_q_count <= step_rescale_count)
        {
            throw std::invalid_argument(
                "GpuUploader::upload_eval_mod_high_precision: modulus chain is too short for basis dynamic rescale");
        }
        const double output_scale =
            pre_rescale_scale /
            rescale_modulus_product(
                multiply_q_count,
                step_rescale_count);
        require_valid_scale(output_scale, "basis output scale");
        step.pre_rescale_scale = pre_rescale_scale;
        step.rescale_count = step_rescale_count;
        output_q_count = multiply_q_count - step_rescale_count;

        if (result.polynomial_basis ==
                GpuEvalModPolynomialBasis::Chebyshev &&
            step.correction_degree == 0)
        {
            const std::size_t correction_q_count = dynamic_rescale
                ? multiply_q_count
                : output_q_count;
            const double correction_scale = dynamic_rescale
                ? pre_rescale_scale
                : output_scale;
            step.correction_plaintext = encode_and_upload(
                {1.0, 0.0},
                correction_q_count,
                correction_scale);
        }
        else if (result.polynomial_basis ==
                     GpuEvalModPolynomialBasis::Chebyshev &&
                 step.correction_degree != 0)
        {
            const auto correction_iter =
                basis_q_counts.find(step.correction_degree);
            if (correction_iter == basis_q_counts.end())
            {
                throw std::logic_error(
                    "GpuUploader::upload_eval_mod_high_precision: missing correction basis level");
            }
            const auto correction_scale_iter =
                basis_scales.find(step.correction_degree);
            if (correction_scale_iter == basis_scales.end())
            {
                throw std::logic_error(
                    "GpuUploader::upload_eval_mod_high_precision: missing correction basis scale");
            }
            if (dynamic_rescale)
            {
                const double correction_plaintext_scale =
                    pre_rescale_scale / correction_scale_iter->second;
                require_valid_scale(
                    correction_plaintext_scale,
                    "dynamic Chebyshev correction plaintext scale");
                step.correction_plaintext = encode_and_upload(
                    {1.0, 0.0},
                    correction_iter->second,
                    correction_plaintext_scale);
            }
            else
            if (correction_iter->second > output_q_count)
            {
                if (correction_iter->second - output_q_count >
                    std::numeric_limits<std::uint32_t>::max())
                {
                    throw std::invalid_argument(
                        "GpuUploader::upload_eval_mod_high_precision: invalid Chebyshev correction alignment span");
                }
                const auto correction_rescale_count =
                    static_cast<std::uint32_t>(
                        correction_iter->second - output_q_count);
                const double alignment_value =
                    output_scale *
                    rescale_modulus_product(
                        correction_iter->second,
                        correction_rescale_count) /
                    (correction_scale_iter->second *
                     correction_scale_iter->second);
                require_valid_scale(
                    alignment_value,
                    "Chebyshev correction alignment value");
                step.correction_alignment_rescale_count =
                    correction_rescale_count;
                step.correction_alignment_plaintext = encode_and_upload(
                    {alignment_value, 0.0},
                    correction_iter->second,
                    correction_scale_iter->second);
                step.correction_alignment_pre_rescale_scale =
                    output_scale *
                    rescale_modulus_product(
                        correction_iter->second,
                        correction_rescale_count);
            }
            else
            {
                const double correction_plaintext_scale =
                    output_scale / correction_scale_iter->second;
                require_valid_scale(
                    correction_plaintext_scale,
                    "Chebyshev correction plaintext scale");
                step.correction_plaintext = encode_and_upload(
                    {1.0, 0.0},
                    output_q_count,
                    correction_plaintext_scale);
            }
        }
        step.output_scale = output_scale;
        basis_q_counts.emplace(step.output_degree, output_q_count);
        basis_scales.emplace(step.output_degree, output_scale);
    }

    std::size_t dynamic_root_anchor_q_count = 0;
    if (dynamic_rescale)
    {
        const auto materialized_anchor = basis_q_counts.find(degree_bound);
        if (materialized_anchor != basis_q_counts.end())
        {
            dynamic_root_anchor_q_count = materialized_anchor->second;
        }
        else
        {
            if (!virtual_degree_bound || degree_bound <= 1)
            {
                throw std::logic_error(
                    "GpuUploader::upload_eval_mod_high_precision: dynamic polynomial root basis is absent");
            }

            /*
             * The dynamic level planner historically used T_degree_bound as
             * a depth sentinel, even when the polynomial split never consumes
             * that basis. Reproduce only its q-count transition here. This
             * preserves the existing level/scale schedule without emitting a
             * GPU square, relinearization, correction, rescale, or buffer for
             * the unused Chebyshev basis.
             */
            const auto parent_degree = degree_bound >> 1U;
            const auto parent_q = basis_q_counts.find(parent_degree);
            const auto parent_scale = basis_scales.find(parent_degree);
            if (parent_q == basis_q_counts.end() ||
                parent_scale == basis_scales.end())
            {
                throw std::logic_error(
                    "GpuUploader::upload_eval_mod_high_precision: virtual root parent basis is absent");
            }
            const double virtual_pre_rescale_scale =
                parent_scale->second * parent_scale->second;
            require_valid_scale(
                virtual_pre_rescale_scale,
                "virtual degree-bound pre-rescale scale");
            const auto virtual_rescale_count = choose_dynamic_rescale(
                parent_q->second,
                virtual_pre_rescale_scale);
            if (parent_q->second <= virtual_rescale_count)
            {
                throw std::invalid_argument(
                    "GpuUploader::upload_eval_mod_high_precision: virtual degree-bound exhausts modulus chain");
            }
            dynamic_root_anchor_q_count =
                parent_q->second - virtual_rescale_count;
        }
        result.polynomial_root_anchor_q_count =
            dynamic_root_anchor_q_count;
    }

    struct LeafTermSpec
    {
        std::uint32_t degree = 0;
        std::complex<double> coefficient{};
    };

    result.polynomial_blocks.resize(leaves.size());
    std::vector<std::size_t> node_q_counts(next_node_id, 0);
    std::vector<double> node_scales(next_node_id, 0.0);
    std::vector<std::size_t> leaf_input_q_counts(leaves.size(), 0);
    std::vector<std::vector<LeafTermSpec>> leaf_term_specs(leaves.size());
    std::vector<std::complex<double>> leaf_constants(leaves.size());
    std::vector<bool> leaf_has_constant(leaves.size(), false);
    std::vector<bool> node_valid(next_node_id, false);

    auto rescaled_scale =
        [&](std::size_t q_count,
            double input_scale,
            std::uint32_t rescale_count) -> double {
            if (rescale_count == 0)
            {
                return input_scale;
            }
            return input_scale /
                   rescale_modulus_product(q_count, rescale_count);
        };

    auto choose_dynamic_rescale_allow_zero =
        [&](std::size_t q_count,
            double input_scale,
            double min_scale) -> std::uint32_t {
            const auto level_data = crt_context->get_context_data(
                parms_id_for_q_count(q_count));
            if (!level_data ||
                level_data->coeff_modulus().size() != q_count)
            {
                throw std::invalid_argument(
                    "GpuUploader::upload_eval_mod_high_precision: missing dynamic polynomial modulus");
            }
            std::vector<std::uint64_t> active_moduli;
            active_moduli.reserve(q_count);
            for (const auto &modulus : level_data->coeff_modulus())
            {
                active_moduli.push_back(modulus.value());
            }
            return plan_gpu_dynamic_rescale(
                       input_scale,
                       min_scale,
                       active_moduli)
                .rescale_count;
        };

    Polynomial sine_for_parity = sine_polynomial;
    auto [is_odd_polynomial, is_even_polynomial] =
        is_odd_or_even_polynomial(sine_for_parity);
    const auto &full_q_moduli =
        crt_context->first_context_data()->coeff_modulus();

    auto pre_scalar_plan =
        [&](const Polynomial &polynomial,
            std::uint32_t current_level,
            double current_scale) -> std::pair<std::uint32_t, double> {
            std::uint32_t target_level = current_level;
            double target_scale_for_leaf = current_scale;
            if (polynomial.data().empty())
            {
                return {target_level, target_scale_for_leaf};
            }

            std::size_t minimum_degree_non_zero_coefficient =
                polynomial.data().size() - 1;
            if (is_even_polynomial &&
                minimum_degree_non_zero_coefficient > 0)
            {
                --minimum_degree_non_zero_coefficient;
            }
            if (minimum_degree_non_zero_coefficient == 0)
            {
                if (target_scale_for_leaf + 1000000000.0 < target_scale)
                {
                    throw std::invalid_argument(
                        "GpuUploader::upload_eval_mod_high_precision: unsupported constant leaf scale");
                }
                return {target_level, target_scale_for_leaf};
            }

            for (int key = static_cast<int>(polynomial.degree());
                 key > 0;
                 --key)
            {
                if (static_cast<std::size_t>(key) >=
                    polynomial.data().size())
                {
                    continue;
                }
                if (!eval_mod_coefficient_is_nonzero(
                        polynomial.data()[static_cast<std::size_t>(key)]))
                {
                    continue;
                }
                const auto basis_scale_iter =
                    basis_scales.find(static_cast<std::uint32_t>(key));
                if (basis_scale_iter == basis_scales.end())
                {
                    throw std::logic_error(
                        "GpuUploader::upload_eval_mod_high_precision: missing leaf basis scale");
                }
                while (true)
                {
                    const double plaintext_scale =
                        target_scale_for_leaf /
                        basis_scale_iter->second;
                    if (plaintext_scale >= target_scale)
                    {
                        break;
                    }
                    ++target_level;
                    if (target_level >= full_q_moduli.size())
                    {
                        throw std::invalid_argument(
                            "GpuUploader::upload_eval_mod_high_precision: dynamic leaf target level exceeds Q chain");
                    }
                    target_scale_for_leaf *=
                        static_cast<double>(
                            full_q_moduli[target_level].value());
                }
            }
            return {target_level, target_scale_for_leaf};
        };

    auto polynomial_has_evaluable_terms =
        [&](const Polynomial &polynomial) -> bool {
            for (const auto &coefficient : polynomial.data())
            {
                if (eval_mod_coefficient_is_nonzero(coefficient))
                {
                    return true;
                }
            }
            return false;
        };

    std::uint32_t dynamic_recursive_num = 0;
    struct PlannedNode
    {
        bool valid = false;
        std::size_t q_count = 0;
        double scale = 0.0;
    };

    std::function<PlannedNode(
        EvalModSplitNode &,
        std::uint32_t,
        double)> plan_dynamic_node;
    plan_dynamic_node =
        [&](EvalModSplitNode &node,
            std::uint32_t target_level,
            double requested_scale) -> PlannedNode {
            if (node.is_leaf())
            {
                if (!polynomial_has_evaluable_terms(node.polynomial))
                {
                    return {};
                }
                const auto [leaf_level, leaf_scale] =
                    pre_scalar_plan(
                        node.polynomial,
                        target_level,
                        requested_scale);
                const std::size_t leaf_q_count =
                    static_cast<std::size_t>(leaf_level) + 1;
                node_valid[node.node_id] = true;
                node_q_counts[node.node_id] = leaf_q_count;
                node_scales[node.node_id] = leaf_scale;
                return {true, leaf_q_count, leaf_scale};
            }

            const auto basis_degree = node.split_degree;
            const auto basis_q_iter = basis_q_counts.find(basis_degree);
            const auto basis_scale_iter = basis_scales.find(basis_degree);
            if (basis_q_iter == basis_q_counts.end() ||
                basis_scale_iter == basis_scales.end())
            {
                throw std::logic_error(
                    "GpuUploader::upload_eval_mod_high_precision: missing combine basis plan");
            }

            double quotient_target_scale = requested_scale;
            double pow_scale = 0.0;
            std::uint32_t quotient_target_level = target_level;
            bool target_scale_pass = false;
            if (dynamic_recursive_num == 0 && node.polynomial.lead())
            {
                while (!target_scale_pass)
                {
                    if (quotient_target_level < dynamic_recursive_num)
                    {
                        throw std::invalid_argument(
                            "GpuUploader::upload_eval_mod_high_precision: invalid first lead target level");
                    }
                    const auto modulus_index =
                        quotient_target_level - dynamic_recursive_num;
                    quotient_target_scale *=
                        static_cast<double>(
                            full_q_moduli[modulus_index].value());
                    const double tmp_scale =
                        quotient_target_scale /
                        basis_scale_iter->second;
                    ++dynamic_recursive_num;
                    if (tmp_scale + 1000000000.0 >= target_scale)
                    {
                        quotient_target_scale = tmp_scale;
                        target_scale_pass = true;
                    }
                }
            }
            else if (node.polynomial.lead())
            {
                while (!target_scale_pass)
                {
                    ++quotient_target_level;
                    if (quotient_target_level >= full_q_moduli.size())
                    {
                        throw std::invalid_argument(
                            "GpuUploader::upload_eval_mod_high_precision: dynamic lead target level exceeds Q chain");
                    }
                    quotient_target_scale *=
                        static_cast<double>(
                            full_q_moduli[quotient_target_level].value());
                    const double tmp_scale =
                        quotient_target_scale /
                        basis_scale_iter->second;
                    if (tmp_scale + 1000000000.0 >= target_scale)
                    {
                        quotient_target_scale = tmp_scale;
                        target_scale_pass = true;
                    }
                }
            }
            else
            {
                quotient_target_scale /= basis_scale_iter->second;
                pow_scale = quotient_target_scale;
                while (!target_scale_pass)
                {
                    ++quotient_target_level;
                    if (quotient_target_level >= full_q_moduli.size())
                    {
                        throw std::invalid_argument(
                            "GpuUploader::upload_eval_mod_high_precision: dynamic non-lead target level exceeds Q chain");
                    }
                    quotient_target_scale *=
                        static_cast<double>(
                            full_q_moduli[quotient_target_level].value());
                    const double tmp_scale =
                        quotient_target_scale /
                        basis_scale_iter->second;
                    if (tmp_scale + 1000000000.0 >= target_scale)
                    {
                        target_scale_pass = true;
                    }
                }
            }

            auto quotient =
                plan_dynamic_node(
                    *node.quotient,
                    quotient_target_level,
                    quotient_target_scale);
            if (!quotient.valid)
            {
                auto remainder_only =
                    plan_dynamic_node(
                        *node.remainder,
                        target_level,
                        requested_scale);
                node_valid[node.node_id] = remainder_only.valid;
                if (remainder_only.valid)
                {
                    node_q_counts[node.node_id] = remainder_only.q_count;
                    node_scales[node.node_id] = remainder_only.scale;
                }
                return remainder_only;
            }

            const double quotient_min_scale = node.polynomial.lead()
                ? context.parameters_literal()->scale()
                : pow_scale;
            const auto quotient_rescale_count =
                choose_dynamic_rescale_allow_zero(
                    quotient.q_count,
                    quotient.scale,
                    quotient_min_scale);
            const auto quotient_q_count =
                quotient.q_count - quotient_rescale_count;
            const auto quotient_scale_after =
                rescaled_scale(
                    quotient.q_count,
                    quotient.scale,
                    quotient_rescale_count);
            const std::size_t product_q_count =
                std::min(quotient_q_count, basis_q_iter->second);
            const double product_scale =
                quotient_scale_after * basis_scale_iter->second;
            required_relin_q_counts.insert(product_q_count);

            auto remainder =
                plan_dynamic_node(
                    *node.remainder,
                    static_cast<std::uint32_t>(product_q_count - 1),
                    product_scale);

            std::uint32_t remainder_rescale_count = 0;
            std::size_t output_q_count = product_q_count;
            double output_scale = product_scale;
            std::size_t remainder_q_count = 0;
            double remainder_scale_after = 0.0;
            if (remainder.valid)
            {
                remainder_rescale_count =
                    choose_dynamic_rescale_allow_zero(
                        remainder.q_count,
                        remainder.scale,
                        product_scale);
                remainder_q_count =
                    remainder.q_count - remainder_rescale_count;
                remainder_scale_after =
                    rescaled_scale(
                        remainder.q_count,
                        remainder.scale,
                        remainder_rescale_count);
                output_q_count =
                    std::min(product_q_count, remainder_q_count);
                output_scale = product_scale;
                if (!util::are_approximate<double>(
                        product_scale,
                        remainder_scale_after))
                {
                    output_scale =
                        std::max(product_scale, remainder_scale_after);
                }
            }

            for (auto &combine : result.polynomial_combine_steps)
            {
                if (combine.output_node == node.node_id)
                {
                    combine.quotient_rescale_count =
                        quotient_rescale_count;
                    combine.quotient_output_scale =
                        quotient_scale_after;
                    combine.remainder_rescale_count =
                        remainder_rescale_count;
                    combine.product_scale = product_scale;
                    combine.product_q_count = product_q_count;
                    combine.output_scale = output_scale;
                    combine.output_q_count = output_q_count;
                    combine.product_aligned_scale = product_scale;
                    combine.remainder_aligned_scale = remainder_scale_after;
                    if (remainder.valid &&
                        !util::are_approximate<double>(
                            product_scale,
                            remainder_scale_after))
                    {
                        if (product_scale > remainder_scale_after)
                        {
                            const double scale_ratio =
                                product_scale / remainder_scale_after + 0.5;
                            require_valid_scale(
                                scale_ratio,
                                "dynamic remainder add scale ratio");
                            combine.remainder_scale_plaintext =
                                encode_and_upload(
                                    {scale_ratio, 0.0},
                                    remainder_q_count,
                                    1.0);
                            combine.remainder_aligned_scale = product_scale;
                        }
                        else
                        {
                            const double scale_ratio =
                                remainder_scale_after / product_scale + 0.5;
                            require_valid_scale(
                                scale_ratio,
                                "dynamic product add scale ratio");
                            combine.product_scale_plaintext =
                                encode_and_upload(
                                    {scale_ratio, 0.0},
                                    product_q_count,
                                    1.0);
                            combine.product_aligned_scale =
                                remainder_scale_after;
                        }
                    }
                    break;
                }
            }

            node_valid[node.node_id] = true;
            node_q_counts[node.node_id] = output_q_count;
            node_scales[node.node_id] = output_scale;
            return {true, output_q_count, output_scale};
        };

    for (std::size_t leaf_index = 0; leaf_index < leaves.size(); ++leaf_index)
    {
        const auto &coefficients = leaves[leaf_index]->polynomial.data();
        std::size_t block_q_count = input_q_count;
        bool has_nonconstant_term = false;
        for (std::size_t degree = 1; degree < coefficients.size(); ++degree)
        {
            if (!eval_mod_coefficient_is_nonzero(coefficients[degree]))
            {
                continue;
            }
            const auto basis_iter = basis_q_counts.find(
                static_cast<std::uint32_t>(degree));
            if (basis_iter == basis_q_counts.end())
            {
                throw std::logic_error(
                    "GpuUploader::upload_eval_mod_high_precision: leaf basis level is absent");
            }
            block_q_count = has_nonconstant_term
                ? std::min(block_q_count, basis_iter->second)
                : basis_iter->second;
            has_nonconstant_term = true;
            leaf_term_specs[leaf_index].push_back(LeafTermSpec{
                static_cast<std::uint32_t>(degree),
                coefficients[degree]});
        }
        if (dynamic_rescale)
        {
            std::sort(
                leaf_term_specs[leaf_index].begin(),
                leaf_term_specs[leaf_index].end(),
                [](const LeafTermSpec &lhs, const LeafTermSpec &rhs) {
                    return lhs.degree > rhs.degree;
                });
        }
        leaf_has_constant[leaf_index] =
            !coefficients.empty() &&
            eval_mod_coefficient_is_nonzero(coefficients.front());
        if (leaf_has_constant[leaf_index])
        {
            leaf_constants[leaf_index] = coefficients.front();
        }
        if (block_q_count <= logical_rescale_count)
        {
            throw std::invalid_argument(
                "GpuUploader::upload_eval_mod_high_precision: modulus chain is too short for leaf rescale");
        }
        std::sort(
            leaf_term_specs[leaf_index].begin(),
            leaf_term_specs[leaf_index].end(),
            [](const LeafTermSpec &lhs, const LeafTermSpec &rhs) {
                return lhs.degree > rhs.degree;
            });
        leaf_input_q_counts[leaf_index] = block_q_count;
        node_q_counts[leaf_index] =
            block_q_count - logical_rescale_count;
    }

    if (dynamic_rescale)
    {
        if (dynamic_root_anchor_q_count == 0 ||
            dynamic_root_anchor_q_count - 1 >
                std::numeric_limits<std::uint32_t>::max())
        {
            throw std::invalid_argument(
                "GpuUploader::upload_eval_mod_high_precision: invalid dynamic root anchor level");
        }
        auto root_node =
            plan_dynamic_node(
                *split_tree,
                static_cast<std::uint32_t>(dynamic_root_anchor_q_count - 1),
                target_scale);
        if (!root_node.valid)
        {
            throw std::invalid_argument(
                "GpuUploader::upload_eval_mod_high_precision: dynamic polynomial produced no output");
        }

        const auto final_rescale_count =
            choose_dynamic_rescale_allow_zero(
                root_node.q_count,
                root_node.scale,
                target_scale);
        node_q_counts[result.polynomial_result_node] =
            root_node.q_count - final_rescale_count;
        node_scales[result.polynomial_result_node] = rescaled_scale(
            root_node.q_count,
            root_node.scale,
            final_rescale_count);
    }
    else
    {
        for (const auto &combine : result.polynomial_combine_steps)
        {
            const auto basis_iter = basis_q_counts.find(combine.basis_degree);
            if (basis_iter == basis_q_counts.end() ||
                combine.quotient_node >= node_q_counts.size() ||
                combine.remainder_node >= node_q_counts.size() ||
                combine.output_node >= node_q_counts.size() ||
                node_q_counts[combine.quotient_node] == 0 ||
                node_q_counts[combine.remainder_node] == 0)
            {
                throw std::logic_error(
                    "GpuUploader::upload_eval_mod_high_precision: invalid combine level schedule");
            }
            const std::size_t multiply_q_count = std::min(
                node_q_counts[combine.quotient_node],
                basis_iter->second);
            if (multiply_q_count <= logical_rescale_count)
            {
                throw std::invalid_argument(
                    "GpuUploader::upload_eval_mod_high_precision: modulus chain is too short for polynomial combine");
            }
            required_relin_q_counts.insert(multiply_q_count);
            node_q_counts[combine.output_node] = std::min(
                multiply_q_count - logical_rescale_count,
                node_q_counts[combine.remainder_node]);
        }

        node_scales[result.polynomial_result_node] =
            std::isfinite(polynomial_output_scale_override)
                ? polynomial_output_scale_override
                : target_scale;
        for (auto step_iter = result.polynomial_combine_steps.rbegin();
             step_iter != result.polynomial_combine_steps.rend();
             ++step_iter)
        {
            auto &combine = *step_iter;
            const double output_scale = node_scales[combine.output_node];
            require_valid_scale(output_scale, "polynomial combine output scale");
            const auto basis_q_iter = basis_q_counts.find(combine.basis_degree);
            const auto basis_scale_iter = basis_scales.find(combine.basis_degree);
            if (basis_q_iter == basis_q_counts.end() ||
                basis_scale_iter == basis_scales.end())
            {
                throw std::logic_error(
                    "GpuUploader::upload_eval_mod_high_precision: combine basis schedule is absent");
            }
            const std::size_t multiply_q_count = std::min(
                node_q_counts[combine.quotient_node],
                basis_q_iter->second);
            const double quotient_scale =
                output_scale *
                rescale_modulus_product(
                    multiply_q_count,
                    logical_rescale_count) /
                basis_scale_iter->second;
            require_valid_scale(quotient_scale, "polynomial quotient scale");
            node_scales[combine.quotient_node] = quotient_scale;
            node_scales[combine.remainder_node] = output_scale;
            combine.output_scale = output_scale;
        }
    }

    for (std::size_t leaf_index = 0; leaf_index < leaves.size(); ++leaf_index)
    {
        const double leaf_output_scale = node_scales[leaf_index];
        require_valid_scale(leaf_output_scale, "polynomial leaf output scale");
        const std::size_t block_q_count = dynamic_rescale
            ? node_q_counts[leaf_index]
            : leaf_input_q_counts[leaf_index];
        const std::uint32_t block_rescale_count = dynamic_rescale
            ? 0U
            : logical_rescale_count;
        const double pre_rescale_scale = dynamic_rescale
            ? leaf_output_scale
            : leaf_output_scale *
                  rescale_modulus_product(
                      block_q_count,
                      logical_rescale_count);
        require_valid_scale(pre_rescale_scale, "polynomial leaf accumulator scale");

        auto &block = result.polynomial_blocks[leaf_index];
        block.rescale_count = block_rescale_count;
        block.output_scale = leaf_output_scale;
        block.output_q_count = node_q_counts[leaf_index];
        for (const auto &term : leaf_term_specs[leaf_index])
        {
            const auto basis_scale_iter = basis_scales.find(term.degree);
            const auto basis_q_iter = basis_q_counts.find(term.degree);
            if (basis_scale_iter == basis_scales.end())
            {
                throw std::logic_error(
                    "GpuUploader::upload_eval_mod_high_precision: leaf basis scale is absent");
            }
            if (basis_q_iter == basis_q_counts.end())
            {
                throw std::logic_error(
                    "GpuUploader::upload_eval_mod_high_precision: leaf basis level is absent");
            }
            const std::size_t term_q_count = dynamic_rescale
                ? basis_q_iter->second
                : block_q_count;
            const double plaintext_scale =
                pre_rescale_scale / basis_scale_iter->second;
            require_valid_scale(plaintext_scale, "polynomial coefficient scale");
            block.terms.push_back(GpuEvalModPolynomialTerm{
                term.degree,
                encode_and_upload(
                    term.coefficient,
                    term_q_count,
                    plaintext_scale)});
        }

        if (leaf_has_constant[leaf_index] || block.terms.empty())
        {
            const std::size_t constant_q_count =
                dynamic_rescale
                    ? (!leaf_term_specs[leaf_index].empty()
                          ? leaf_input_q_counts[leaf_index]
                          : block_q_count)
                    : result.rescale_polynomial_terms_individually
                    ? node_q_counts[leaf_index]
                    : block_q_count;
            const double constant_scale =
                dynamic_rescale
                    ? leaf_output_scale
                    : result.rescale_polynomial_terms_individually
                    ? leaf_output_scale
                    : pre_rescale_scale;
            block.terms.push_back(GpuEvalModPolynomialTerm{
                0,
                encode_and_upload(
                    leaf_has_constant[leaf_index]
                        ? leaf_constants[leaf_index]
                        : std::complex<double>{0.0, 0.0},
                    constant_q_count,
                    constant_scale)});
        }
    }

    std::size_t output_q_count =
        node_q_counts[result.polynomial_result_node];
    result.polynomial_output_scale =
        node_scales[result.polynomial_result_node];
    if (output_q_count == 0)
    {
        throw std::logic_error(
            "GpuUploader::upload_eval_mod_high_precision: polynomial output level is unavailable");
    }

    if (include_input_offset &&
        (eval_mod_poly.type() == CosDiscrete ||
         eval_mod_poly.type() == CosContinuous))
    {
        const double interval_width =
            eval_mod_poly.sine_poly_b() - eval_mod_poly.sine_poly_a();
        const double denominator = eval_mod_poly.sc_fac() * interval_width;
        if (!(std::abs(denominator) > 0.0) || !std::isfinite(denominator))
        {
            throw std::invalid_argument(
                "GpuUploader::upload_eval_mod_high_precision: invalid cosine offset denominator");
        }
        result.input_offset_plaintext = encode_and_upload(
            {-0.5 / denominator, 0.0},
            input_q_count,
            effective_input_scale);
    }

    const std::uint32_t double_angle_count =
        double_angle_override == std::numeric_limits<std::uint32_t>::max()
            ? eval_mod_poly.double_angle()
            : double_angle_override;
    double sqrt_2pi = std::isfinite(double_angle_base_override)
        ? double_angle_base_override
        : eval_mod_poly.sqrt_2pi();
    double double_angle_input_scale =
        node_scales[result.polynomial_result_node];
    result.double_angle_constants.reserve(double_angle_count);
    result.double_angle_rescale_counts.reserve(double_angle_count);
    for (std::uint32_t i = 0; i < double_angle_count; ++i)
    {
        if (output_q_count <= logical_rescale_count)
        {
            throw std::invalid_argument(
                "GpuUploader::upload_eval_mod_high_precision: modulus chain is too short for double-angle steps");
        }
        required_relin_q_counts.insert(output_q_count);
        sqrt_2pi *= sqrt_2pi;
        const double pre_rescale_scale =
            double_angle_input_scale * double_angle_input_scale;
        require_valid_scale(pre_rescale_scale, "double-angle pre-rescale scale");
        const std::uint32_t double_angle_rescale_count = dynamic_rescale
            ? choose_dynamic_rescale(output_q_count, pre_rescale_scale)
            : logical_rescale_count;
        if (output_q_count <= double_angle_rescale_count)
        {
            throw std::invalid_argument(
                "GpuUploader::upload_eval_mod_high_precision: modulus chain is too short for dynamic double-angle step");
        }
        const double double_angle_output_scale =
            pre_rescale_scale /
            rescale_modulus_product(
                output_q_count,
                double_angle_rescale_count);
        const std::size_t double_angle_output_q_count =
            output_q_count - double_angle_rescale_count;
        result.double_angle_constants.push_back(
            encode_and_upload(
                {-sqrt_2pi, 0.0},
                output_q_count,
                pre_rescale_scale));
        result.double_angle_rescale_counts.push_back(
            double_angle_rescale_count);
        double_angle_input_scale = double_angle_output_scale;
        require_valid_scale(double_angle_input_scale, "double-angle output scale");
        output_q_count = double_angle_output_q_count;
    }

    if (expected_output_parms_id != parms_id_zero)
    {
        const auto expected_output_context =
            crt_context->get_context_data(expected_output_parms_id);
        if (!expected_output_context)
        {
            throw std::invalid_argument(
                "GpuUploader::upload_eval_mod_high_precision: expected CPU output parms_id is absent from context");
        }
        const auto expected_output_q_count =
            expected_output_context->coeff_modulus().size();
        if (expected_output_q_count > output_q_count)
        {
            throw std::invalid_argument(
                "GpuUploader::upload_eval_mod_high_precision: static GPU schedule consumes more levels than CPU reference");
        }
        output_q_count = expected_output_q_count;
    }

    result.output_q_count = output_q_count;
    result.output_scale = double_angle_input_scale;
    result.output_parms_id = expected_output_parms_id != parms_id_zero
        ? expected_output_parms_id
        : parms_id_for_q_count(output_q_count);

    /*
     * Dynamic planning may request an extra power-of-two Chebyshev basis only
     * as a level/scale anchor (for degree 59 this is T64).  It is useful while
     * constructing node_q_counts, but the runtime evaluator must not spend a
     * ciphertext square/relinearize/rescale on a basis that no polynomial
     * term, combine node, or Chebyshev correction consumes.  Compute liveness
     * from the final uploaded DAG and discard planning-only basis steps.
     */
    if (dynamic_rescale && !result.basis_steps.empty())
    {
        std::map<std::uint32_t, const GpuEvalModBasisStep *> step_by_degree;
        for (const auto &step : result.basis_steps)
        {
            step_by_degree.emplace(step.output_degree, &step);
        }

        std::set<std::uint32_t> live_basis_degrees;
        std::function<void(std::uint32_t)> mark_live_basis =
            [&](std::uint32_t degree) {
                if (degree <= 1 || !live_basis_degrees.insert(degree).second)
                {
                    return;
                }
                const auto step_iter = step_by_degree.find(degree);
                if (step_iter == step_by_degree.end())
                {
                    throw std::logic_error(
                        "GpuUploader::upload_eval_mod_high_precision: live basis step is absent");
                }
                const auto &step = *step_iter->second;
                mark_live_basis(step.left_degree);
                mark_live_basis(step.right_degree);
                if (step.correction_degree != 0)
                {
                    mark_live_basis(step.correction_degree);
                }
            };

        for (const auto &block : result.polynomial_blocks)
        {
            for (const auto &term : block.terms)
            {
                if (term.degree != 0)
                {
                    mark_live_basis(term.degree);
                }
            }
        }
        for (const auto &combine : result.polynomial_combine_steps)
        {
            mark_live_basis(combine.basis_degree);
        }

        result.basis_steps.erase(
            std::remove_if(
                result.basis_steps.begin(),
                result.basis_steps.end(),
                [&](const GpuEvalModBasisStep &step) {
                    return live_basis_degrees.count(step.output_degree) == 0;
                }),
            result.basis_steps.end());
    }

    result.required_relin_q_counts.assign(
        required_relin_q_counts.begin(),
        required_relin_q_counts.end());

    if (relin_keys != nullptr)
    {
        prepare_key_views_for_q_counts(
            *relin_keys,
            result.required_relin_q_counts);
    }
    return result;
}

GpuRelinKeysData GpuUploader::upload_relin_keys(
    const RelinKeys &src,
    int device_id)
{
    return upload_kswitch_keys(src, device_id);
}

GpuRelinKeysData GpuUploader::upload_relin_keys(
    const RelinKeys &src,
    int device_id,
    std::size_t q_count)
{
    auto result = upload_kswitch_keys(src, device_id);
    prepare_key_views_for_q_counts(result, {q_count});
    return result;
}

GpuGaloisKeysData GpuUploader::upload_galois_keys(
    const GaloisKeys &src,
    int device_id)
{
    return upload_kswitch_keys(src, device_id);
}

GpuGaloisKeysData GpuUploader::upload_galois_keys(
    const GaloisKeys &src,
    int device_id,
    std::size_t q_count)
{
    auto result = upload_kswitch_keys(src, device_id);
    prepare_key_views_for_q_counts(result, {q_count});
    return result;
}

GpuGaloisKeysData GpuUploader::upload_galois_keys(
    const GaloisKeys &src,
    int device_id,
    std::size_t q_count,
    const std::vector<std::uint32_t> &galois_elts)
{
    std::vector<std::size_t> key_indices;
    key_indices.reserve(galois_elts.size());
    for (const auto galois_elt : galois_elts)
    {
        if (!src.has_key(galois_elt))
        {
            throw std::invalid_argument("selected Galois key does not exist");
        }
        key_indices.push_back(GaloisKeys::get_index(galois_elt));
    }
    auto result = upload_kswitch_keys(src, device_id, &key_indices);
    prepare_key_views_for_q_counts(result, {q_count});
    return result;
}

GpuGaloisKeysData GpuUploader::upload_double_hoist_galois_keys(
    const GaloisKeys &src,
    int device_id)
{
    auto result = upload_kswitch_keys(src, device_id);
    result.meta.galois_format =
        GpuGaloisKeyFormat::InversePreRotated;
    result.galois_elts_by_key_index.assign(
        result.meta.key_count,
        0);

    const std::uint64_t modulus =
        static_cast<std::uint64_t>(result.meta.degree) << 1;
    auto inverse_odd_mod_power_of_two =
        [modulus](std::uint32_t value)
    {
        std::uint64_t inverse = 1;
        /*
         * Newton iteration doubles the number of correct low bits each step.
         * Masking is valid because 2N is a power of two.
         */
        for (int iteration = 0; iteration < 6; ++iteration)
        {
            inverse *= 2 - static_cast<std::uint64_t>(value) * inverse;
            inverse &= modulus - 1;
        }
        if (((static_cast<std::uint64_t>(value) * inverse) &
             (modulus - 1)) != 1)
        {
            throw std::logic_error(
                "upload_double_hoist_galois_keys: inverse automorphism failed");
        }
        return static_cast<std::uint32_t>(inverse);
    };

    const std::size_t poly_words =
        (result.meta.q_count + result.meta.p_count) *
        result.meta.degree;
    DeviceVector<GpuWord> temporary(poly_words, device_id);
    auto mutable_view = result.make_view();
    const auto const_view = result.make_const_view();
    for (std::size_t poly = 0; poly < result.poly_metadata_.size(); ++poly)
    {
        const auto &metadata = result.poly_metadata_[poly];
        const std::uint32_t galois_elt = static_cast<std::uint32_t>(
            2 * metadata.key_index + 1);
        result.galois_elts_by_key_index[metadata.key_index] =
            galois_elt;
        const std::uint32_t inverse_elt =
            inverse_odd_mod_power_of_two(galois_elt);

        const auto &source = const_view.polys[metadata.poly_id].shards.front();
        auto &destination =
            mutable_view.polys[metadata.poly_id].shards.front();
        GpuPolyShardView temporary_view{
            device_id,
            temporary.data(),
            0,
            result.meta.q_count + result.meta.p_count,
            0,
            result.meta.degree};
        kernel::launch_apply_galois_ntt_poly_shard(
            temporary_view,
            source,
            inverse_elt,
            result.meta.degree);
        gpu_check_cuda(
            cudaMemcpy(
                destination.ptr,
                temporary.data(),
                poly_words * sizeof(GpuWord),
                cudaMemcpyDeviceToDevice),
            "upload_double_hoist_galois_keys copy pre-rotated key");
    }

    const std::size_t pointer_count =
        result.meta.key_count * result.meta.decomposition_count;
    std::vector<const GpuWord *> q0_ptrs(pointer_count, nullptr);
    std::vector<const GpuWord *> p0_ptrs(pointer_count, nullptr);
    std::vector<const GpuWord *> q1_ptrs(pointer_count, nullptr);
    std::vector<const GpuWord *> p1_ptrs(pointer_count, nullptr);
    const auto transformed_view = result.make_const_view();
    for (const auto &metadata : result.poly_metadata_)
    {
        const std::size_t flat =
            metadata.key_index * result.meta.decomposition_count +
            metadata.decomposition_index;
        const auto &shard =
            transformed_view.polys[metadata.poly_id].shards.front();
        const GpuWord *q_ptr = shard.ptr;
        const GpuWord *p_ptr =
            shard.ptr + result.meta.q_count * result.meta.degree;
        if (metadata.component_index == 0)
        {
            q0_ptrs[flat] = q_ptr;
            p0_ptrs[flat] = p_ptr;
        }
        else if (metadata.component_index == 1)
        {
            q1_ptrs[flat] = q_ptr;
            p1_ptrs[flat] = p_ptr;
        }
    }
    result.galois_key_q0_ptrs.allocate(pointer_count, device_id);
    result.galois_key_p0_ptrs.allocate(pointer_count, device_id);
    result.galois_key_q1_ptrs.allocate(pointer_count, device_id);
    result.galois_key_p1_ptrs.allocate(pointer_count, device_id);
    result.galois_key_q0_ptrs.copy_from_host(q0_ptrs.data(), pointer_count);
    result.galois_key_p0_ptrs.copy_from_host(p0_ptrs.data(), pointer_count);
    result.galois_key_q1_ptrs.copy_from_host(q1_ptrs.data(), pointer_count);
    result.galois_key_p1_ptrs.copy_from_host(p1_ptrs.data(), pointer_count);
    return result;
}

void GpuUploader::prepare_key_views_for_q_counts(
    const GpuEvaluationKeyData &keys,
    const std::vector<std::size_t> &q_counts)
{
    if (keys.empty())
    {
        throw std::invalid_argument(
            "GpuUploader::prepare_key_views_for_q_counts: empty keys");
    }

    for (const auto q_count : q_counts)
    {
        if (q_count == 0 || q_count > keys.meta.q_count)
        {
            throw std::invalid_argument(
                "GpuUploader::prepare_key_views_for_q_counts: invalid q_count");
        }

        /* Construction validates the active prefix. The returned object owns
           no memory and is intentionally discarded; runtime views are equally
           cheap and retain no setup-time pointer lifetime beyond keys. */
        (void)keys.make_const_view(q_count);
    }
}

}  // namespace gpu
}  // namespace poseidon
