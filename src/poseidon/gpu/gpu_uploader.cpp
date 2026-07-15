#include "poseidon/gpu/gpu_uploader.h"

#include "poseidon/advance/homomorphic_linear_transform.h"
#include "poseidon/ciphertext.h"
#include "poseidon/key/galoiskeys.h"
#include "poseidon/key/relinkeys.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

#include <algorithm>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <limits>
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

GpuEvaluationKeyData compact_evaluation_key_to_q_count(
    const GpuEvaluationKeyData &src,
    std::size_t q_count)
{
    if (src.empty())
    {
        throw std::invalid_argument("cannot compact an empty evaluation key");
    }
    if (q_count == 0 || q_count > src.meta.q_count)
    {
        throw std::invalid_argument("invalid compacted evaluation-key q_count");
    }

    GpuEvaluationKeyData dst;
    dst.meta = src.meta;
    dst.meta.q_count = q_count;

    const std::size_t compact_limb_count = q_count + src.meta.p_count;
    const std::size_t compact_word_count = checked_mul(
        compact_limb_count,
        src.meta.degree,
        "compacted evaluation-key word count overflow");
    const std::size_t q_word_count = checked_mul(
        q_count,
        src.meta.degree,
        "compacted evaluation-key Q word count overflow");
    const std::size_t p_word_count = checked_mul(
        src.meta.p_count,
        src.meta.degree,
        "compacted evaluation-key P word count overflow");
    const std::size_t source_p_offset = checked_mul(
        src.meta.q_count,
        src.meta.degree,
        "compacted evaluation-key source P offset overflow");

    if (src.poly_metadata_.size() != src.polys_.size())
    {
        throw std::invalid_argument(
            "evaluation-key metadata/poly count mismatch during compaction");
    }

    dst.fields_.reserve(src.polys_.size());
    dst.polys_.reserve(src.polys_.size());
    dst.poly_metadata_.reserve(src.poly_metadata_.size());

    for (std::size_t poly_index = 0; poly_index < src.polys_.size(); ++poly_index)
    {
        const auto &source_poly = src.polys_[poly_index];
        if (source_poly.degree != src.meta.degree ||
            source_poly.q_count != src.meta.q_count ||
            source_poly.p_count != src.meta.p_count ||
            source_poly.shards.size() != 1)
        {
            throw std::invalid_argument(
                "unexpected evaluation-key poly shape during compaction");
        }

        const auto &source_shard = source_poly.shards.front();
        if (source_shard.field_index >= src.fields_.size())
        {
            throw std::out_of_range(
                "evaluation-key shard field_index is out of range during compaction");
        }
        if (source_shard.limb_begin != 0 ||
            source_shard.limb_count != src.meta.q_count + src.meta.p_count ||
            source_shard.coeff_begin != 0 ||
            source_shard.coeff_count != src.meta.degree)
        {
            throw std::invalid_argument(
                "evaluation-key shard is not a full [Q|P] shard during compaction");
        }

        const auto &source_field = src.fields_[source_shard.field_index];
        const auto source_field_end =
            source_shard.field_offset + checked_mul(
                source_shard.limb_count,
                source_shard.coeff_count,
                "evaluation-key source shard size overflow during compaction");
        if (source_field_end > source_field.size())
        {
            throw std::out_of_range(
                "evaluation-key source shard exceeds field during compaction");
        }

        GpuFieldData compact_field(source_field.device_id, compact_word_count);
        gpu_check_cuda(
            cudaSetDevice(source_field.device_id),
            "compact evaluation key cudaSetDevice");

        const auto *source_ptr = source_field.data() + source_shard.field_offset;
        if (q_word_count != 0)
        {
            gpu_check_cuda(
                cudaMemcpy(
                    compact_field.data(),
                    source_ptr,
                    q_word_count * sizeof(GpuWord),
                    cudaMemcpyDeviceToDevice),
                "compact evaluation key copy Q");
        }
        if (p_word_count != 0)
        {
            gpu_check_cuda(
                cudaMemcpy(
                    compact_field.data() + q_word_count,
                    source_ptr + source_p_offset,
                    p_word_count * sizeof(GpuWord),
                    cudaMemcpyDeviceToDevice),
                "compact evaluation key copy P");
        }

        const std::size_t compact_field_index = dst.fields_.size();
        dst.fields_.push_back(std::move(compact_field));

        GpuRNSPoly compact_poly;
        compact_poly.poly_id = dst.polys_.size();
        compact_poly.degree = src.meta.degree;
        compact_poly.q_count = q_count;
        compact_poly.p_count = src.meta.p_count;

        GpuPolyShard compact_shard;
        compact_shard.field_index = compact_field_index;
        compact_shard.field_offset = 0;
        compact_shard.limb_begin = 0;
        compact_shard.limb_count = compact_limb_count;
        compact_shard.coeff_begin = 0;
        compact_shard.coeff_count = src.meta.degree;
        compact_poly.shards.push_back(compact_shard);

        auto compact_meta = src.poly_metadata_[poly_index];
        compact_meta.poly_id = compact_poly.poly_id;
        dst.poly_metadata_.push_back(compact_meta);
        dst.polys_.push_back(std::move(compact_poly));
    }

    return dst;
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

void GpuUploader::precompute_compacted_keys_for_q_counts(
    GpuEvaluationKeyData &keys,
    const std::vector<std::size_t> &q_counts)
{
    if (keys.empty())
    {
        throw std::invalid_argument(
            "GpuUploader::precompute_compacted_keys_for_q_counts: empty keys");
    }

    for (const auto q_count : q_counts)
    {
        if (q_count == keys.meta.q_count)
        {
            continue;
        }
        if (q_count == 0 || q_count > keys.meta.q_count)
        {
            throw std::invalid_argument(
                "GpuUploader::precompute_compacted_keys_for_q_counts: invalid q_count");
        }
        if (keys.has_compacted_key(q_count))
        {
            continue;
        }

        keys.store_compacted_key(
            q_count,
            compact_evaluation_key_to_q_count(keys, q_count));
    }
}

}  // namespace gpu
}  // namespace poseidon
