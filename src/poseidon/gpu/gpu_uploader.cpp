#include "poseidon/gpu/gpu_uploader.h"

#include "poseidon/advance/homomorphic_linear_transform.h"
#include "poseidon/advance/homomorphic_mod.h"
#include "poseidon/ciphertext.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
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
    int device_id)
{
    GpuEvaluationKeyData dst;
    dst.meta.key_parms_id = src.parms_id();
    dst.meta.key_count = src.data().size();

    for (std::size_t key_index = 0; key_index < src.data().size(); ++key_index)
    {
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

std::unique_ptr<EvalModSplitNode> build_eval_mod_split_tree(
    Polynomial polynomial,
    std::uint32_t log_split,
    std::uint32_t log_degree)
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
        if (node->polynomial.lead() && log_split > 1)
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
                        adjusted_log_degree);
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
        log_degree);
    node->remainder = build_eval_mod_split_tree(
        std::move(std::get<1>(split)),
        log_split,
        log_degree);
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
    dst.data().reserve(src.data().size());

    for (const auto &matrix : src.data())
    {
        dst.data().push_back(upload_matrix_plain(matrix, device_id));
    }

    return dst;
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
    double polynomial_output_scale_override)
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
            Plaintext plaintext;
            encoder.encode(
                value,
                parms_id_for_q_count(q_count),
                scale,
                plaintext);
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
    const auto log_split =
        static_cast<std::uint32_t>(optimal_split(static_cast<int>(log_degree)));

    auto split_tree = build_eval_mod_split_tree(
        sine_polynomial,
        log_split,
        log_degree);
    std::vector<EvalModSplitNode *> leaves;
    collect_eval_mod_leaves(*split_tree, leaves);
    if (leaves.empty() ||
        leaves.size() > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::invalid_argument(
            "GpuUploader::upload_eval_mod_high_precision: invalid split tree");
    }

    GpuBootstrapData::EvalModData result;
    result.target_scale = target_scale;
    result.logical_rescale_count = logical_rescale_count;
    result.polynomial_rescale_count = logical_rescale_count;
    result.rescale_polynomial_terms_individually =
        polynomial_override != nullptr;
    result.polynomial_basis =
        sine_polynomial.basis_type() == Chebyshev
            ? GpuEvalModPolynomialBasis::Chebyshev
            : GpuEvalModPolynomialBasis::Monomial;

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
    result.basis_steps = make_gpu_eval_mod_basis_plan(
        result.polynomial_basis,
        requested_degrees);

    std::map<std::uint32_t, std::size_t> basis_q_counts;
    std::map<std::uint32_t, double> basis_scales;
    basis_q_counts.emplace(0, input_q_count);
    basis_q_counts.emplace(1, input_q_count);
    basis_scales.emplace(1, target_scale);
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
        if (left_iter->second != right_iter->second)
        {
            const bool align_left = left_iter->second > right_iter->second;
            const std::size_t high_q_count = align_left
                ? left_iter->second
                : right_iter->second;
            const double high_scale = align_left
                ? left_scale_iter->second
                : right_scale_iter->second;
            const double low_scale = align_left
                ? right_scale_iter->second
                : left_scale_iter->second;
            const double alignment_value =
                low_scale *
                rescale_modulus_product(high_q_count, logical_rescale_count) /
                (high_scale * high_scale);
            require_valid_scale(alignment_value, "basis operand alignment value");
            step.align_left_operand = align_left;
            step.operand_alignment_plaintext = encode_and_upload(
                {alignment_value, 0.0}, high_q_count, high_scale);
            step.operand_alignment_pre_rescale_scale =
                low_scale *
                rescale_modulus_product(high_q_count, logical_rescale_count);
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
        const double output_scale =
            multiply_left_scale * multiply_right_scale /
            rescale_modulus_product(
                multiply_q_count,
                logical_rescale_count);
        require_valid_scale(output_scale, "basis output scale");

        if (result.polynomial_basis ==
                GpuEvalModPolynomialBasis::Chebyshev &&
            step.correction_degree == 0)
        {
            step.correction_plaintext = encode_and_upload(
                {1.0, 0.0},
                output_q_count,
                output_scale);
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
            if (correction_iter->second > output_q_count)
            {
                const double alignment_value =
                    output_scale *
                    rescale_modulus_product(
                        correction_iter->second,
                        logical_rescale_count) /
                    (correction_scale_iter->second *
                     correction_scale_iter->second);
                require_valid_scale(
                    alignment_value,
                    "Chebyshev correction alignment value");
                step.correction_alignment_plaintext = encode_and_upload(
                    {alignment_value, 0.0},
                    correction_iter->second,
                    correction_scale_iter->second);
                step.correction_alignment_pre_rescale_scale =
                    output_scale *
                    rescale_modulus_product(
                        correction_iter->second,
                        logical_rescale_count);
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

    struct LeafTermSpec
    {
        std::uint32_t degree = 0;
        std::complex<double> coefficient{};
    };

    result.polynomial_blocks.resize(leaves.size());
    std::vector<std::size_t> node_q_counts(next_node_id, 0);
    std::vector<std::size_t> leaf_input_q_counts(leaves.size(), 0);
    std::vector<std::vector<LeafTermSpec>> leaf_term_specs(leaves.size());
    std::vector<std::complex<double>> leaf_constants(leaves.size());
    std::vector<bool> leaf_has_constant(leaves.size(), false);
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
        leaf_input_q_counts[leaf_index] = block_q_count;
        node_q_counts[leaf_index] =
            block_q_count - logical_rescale_count;
    }

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

    /*
     * Assign a desired scale to every polynomial-tree node. For
     * Q*T_k+R, choose Q's scale so that the ordinary-rescaled product and R
     * both land exactly at the parent's desired scale. This turns the CPU
     * recursive scale choices into a fixed GPU schedule.
     */
    std::vector<double> node_scales(next_node_id, 0.0);
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

    for (std::size_t leaf_index = 0; leaf_index < leaves.size(); ++leaf_index)
    {
        const double leaf_output_scale = node_scales[leaf_index];
        require_valid_scale(leaf_output_scale, "polynomial leaf output scale");
        const std::size_t block_q_count = leaf_input_q_counts[leaf_index];
        const double pre_rescale_scale =
            leaf_output_scale *
            rescale_modulus_product(
                block_q_count,
                logical_rescale_count);
        require_valid_scale(pre_rescale_scale, "polynomial leaf accumulator scale");

        auto &block = result.polynomial_blocks[leaf_index];
        block.rescale_count = logical_rescale_count;
        block.output_scale = leaf_output_scale;
        for (const auto &term : leaf_term_specs[leaf_index])
        {
            const auto basis_scale_iter = basis_scales.find(term.degree);
            if (basis_scale_iter == basis_scales.end())
            {
                throw std::logic_error(
                    "GpuUploader::upload_eval_mod_high_precision: leaf basis scale is absent");
            }
            const double plaintext_scale =
                pre_rescale_scale / basis_scale_iter->second;
            require_valid_scale(plaintext_scale, "polynomial coefficient scale");
            block.terms.push_back(GpuEvalModPolynomialTerm{
                term.degree,
                encode_and_upload(
                    term.coefficient,
                    block_q_count,
                    plaintext_scale)});
        }

        if (leaf_has_constant[leaf_index] || block.terms.empty())
        {
            const std::size_t constant_q_count =
                result.rescale_polynomial_terms_individually
                    ? node_q_counts[leaf_index]
                    : block_q_count;
            const double constant_scale =
                result.rescale_polynomial_terms_individually
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
            target_scale);
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
        result.double_angle_constants.push_back(
            encode_and_upload(
                {-sqrt_2pi, 0.0},
                output_q_count,
                pre_rescale_scale));
        double_angle_input_scale =
            pre_rescale_scale /
            rescale_modulus_product(
                output_q_count,
                logical_rescale_count);
        require_valid_scale(double_angle_input_scale, "double-angle output scale");
        output_q_count -= logical_rescale_count;
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
    result.output_parms_id = expected_output_parms_id != parms_id_zero
        ? expected_output_parms_id
        : parms_id_for_q_count(output_q_count);
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

GpuGaloisKeysData GpuUploader::upload_galois_keys(
    const GaloisKeys &src,
    int device_id)
{
    return upload_kswitch_keys(src, device_id);
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
