#include "poseidon/gpu/gpu_key.h"

#include <limits>
#include <stdexcept>
#include <utility>

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

std::size_t checked_add(std::size_t a, std::size_t b, const char *what)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
    {
        throw std::overflow_error(what);
    }
    return a + b;
}

void validate_shard(
    const GpuRNSPoly &poly,
    const GpuPolyShard &shard,
    const std::vector<GpuFieldData> &fields)
{
    if (shard.field_index >= fields.size())
    {
        throw std::out_of_range("GpuEvaluationKeyData shard field_index is out of range");
    }
    if (shard.limb_count == 0 || shard.coeff_count == 0)
    {
        throw std::invalid_argument("GpuEvaluationKeyData shard must be non-empty");
    }
    if (shard.limb_begin + shard.limb_count > poly.q_count + poly.p_count)
    {
        throw std::out_of_range("GpuEvaluationKeyData shard limb range is out of range");
    }
    if (shard.coeff_begin + shard.coeff_count > poly.degree)
    {
        throw std::out_of_range("GpuEvaluationKeyData shard coefficient range is out of range");
    }

    const auto &field = fields[shard.field_index];
    const auto shard_size = checked_mul(
        shard.limb_count,
        shard.coeff_count,
        "GpuEvaluationKeyData shard size overflow");
    const auto physical_end = checked_add(
        shard.field_offset,
        shard_size,
        "GpuEvaluationKeyData shard physical end overflow");
    if (physical_end > field.size())
    {
        throw std::out_of_range("GpuEvaluationKeyData shard exceeds field allocation");
    }
}

}  // namespace

bool GpuEvaluationKeyData::empty() const
{
    return polys_.empty();
}

bool GpuEvaluationKeyData::has_compacted_key(std::size_t q_count) const
{
    return compacted_by_q_count_.find(q_count) != compacted_by_q_count_.end();
}

void GpuEvaluationKeyData::store_compacted_key(
    std::size_t q_count,
    GpuEvaluationKeyData compacted_key)
{
    compacted_by_q_count_[q_count] =
        std::make_shared<GpuEvaluationKeyData>(std::move(compacted_key));
}

const GpuEvaluationKeyData &GpuEvaluationKeyData::key_for_q_count(
    std::size_t q_count) const
{
    if (q_count == meta.q_count)
    {
        return *this;
    }

    const auto iter = compacted_by_q_count_.find(q_count);
    if (iter == compacted_by_q_count_.end() || iter->second == nullptr)
    {
        throw std::invalid_argument(
            "GpuEvaluationKeyData::key_for_q_count: compacted key for current q_count was not precomputed");
    }
    return *iter->second;
}

GpuEvaluationKeyView GpuEvaluationKeyData::make_view()
{
    GpuEvaluationKeyView result;
    result.meta = meta;
    result.polys.reserve(polys_.size());

    for (const auto &poly : polys_)
    {
        GpuRNSPolyView poly_view;
        poly_view.poly_id = poly.poly_id;
        poly_view.shards.reserve(poly.shards.size());

        for (const auto &shard : poly.shards)
        {
            validate_shard(poly, shard, fields_);

            auto &field = fields_[shard.field_index];
            GpuPolyShardView shard_view;
            shard_view.device_id = field.device_id;
            shard_view.ptr = field.data() + shard.field_offset;
            shard_view.limb_begin = shard.limb_begin;
            shard_view.limb_count = shard.limb_count;
            shard_view.coeff_begin = shard.coeff_begin;
            shard_view.coeff_count = shard.coeff_count;
            poly_view.shards.push_back(shard_view);
        }

        result.polys.push_back(std::move(poly_view));
    }

    return result;
}

GpuConstEvaluationKeyView GpuEvaluationKeyData::make_const_view() const
{
    GpuConstEvaluationKeyView result;
    result.meta = meta;
    result.polys.reserve(polys_.size());

    for (const auto &poly : polys_)
    {
        GpuConstRNSPolyView poly_view;
        poly_view.poly_id = poly.poly_id;
        poly_view.shards.reserve(poly.shards.size());

        for (const auto &shard : poly.shards)
        {
            validate_shard(poly, shard, fields_);

            const auto &field = fields_[shard.field_index];
            GpuConstPolyShardView shard_view;
            shard_view.device_id = field.device_id;
            shard_view.ptr = field.data() + shard.field_offset;
            shard_view.limb_begin = shard.limb_begin;
            shard_view.limb_count = shard.limb_count;
            shard_view.coeff_begin = shard.coeff_begin;
            shard_view.coeff_count = shard.coeff_count;
            poly_view.shards.push_back(shard_view);
        }

        result.polys.push_back(std::move(poly_view));
    }

    return result;
}

}  // namespace gpu
}  // namespace poseidon
