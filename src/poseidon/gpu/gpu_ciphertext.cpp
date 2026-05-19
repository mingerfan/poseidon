#include "poseidon/gpu/gpu_ciphertext.h"

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

void validate_shard(
    const GpuRNSPoly &poly,
    const GpuPolyShard &shard,
    const std::vector<GpuFieldData> &fields)
{
    if (shard.field_index >= fields.size())
    {
        throw std::out_of_range("GpuCiphertextData shard field_index is out of range");
    }
    if (shard.limb_count == 0 || shard.coeff_count == 0)
    {
        throw std::invalid_argument("GpuCiphertextData shard must be non-empty");
    }
    if (shard.limb_begin + shard.limb_count > poly.q_count + poly.p_count)
    {
        throw std::out_of_range("GpuCiphertextData shard limb range is out of range");
    }
    if (shard.coeff_begin + shard.coeff_count > poly.degree)
    {
        throw std::out_of_range("GpuCiphertextData shard coefficient range is out of range");
    }

    const auto &field = fields[shard.field_index];
    const auto last_limb_offset =
        checked_mul(shard.limb_count - 1, poly.degree, "GpuCiphertextData shard size overflow");
    const auto physical_end =
        shard.field_offset + last_limb_offset + shard.coeff_begin + shard.coeff_count;
    if (physical_end > field.size())
    {
        throw std::out_of_range("GpuCiphertextData shard exceeds field allocation");
    }
}

}  // namespace

std::size_t GpuCiphertextData::size() const
{
    return polys_.size();
}

bool GpuCiphertextData::empty() const
{
    return polys_.empty();
}

GpuCiphertextView GpuCiphertextData::make_view()
{
    GpuCiphertextView result;
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
            shard_view.ptr = field.data() + shard.field_offset + shard.coeff_begin;
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

GpuConstCiphertextView GpuCiphertextData::make_const_view() const
{
    GpuConstCiphertextView result;
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
            shard_view.ptr = field.data() + shard.field_offset + shard.coeff_begin;
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

GpuCiphertextData GpuCiphertextData::allocate_single_device(
    std::size_t degree,
    std::size_t q_count,
    std::size_t component_count,
    int device_id,
    std::size_t p_count)
{
    if (degree == 0 || q_count + p_count == 0 || component_count == 0)
    {
        throw std::invalid_argument(
            "GpuCiphertextData::allocate_single_device requires non-zero shape");
    }

    GpuCiphertextData result;
    result.meta.degree = degree;
    result.meta.q_count = q_count;
    result.meta.p_count = p_count;
    result.meta.component_count = component_count;

    const auto limb_count = q_count + p_count;
    const auto field_size = checked_mul(degree, limb_count, "GpuCiphertextData field size overflow");
    result.fields_.reserve(component_count);
    result.polys_.reserve(component_count);

    for (std::size_t component = 0; component < component_count; ++component)
    {
        result.fields_.emplace_back(device_id, field_size);

        GpuRNSPoly poly;
        poly.poly_id = component;
        poly.degree = degree;
        poly.q_count = q_count;
        poly.p_count = p_count;

        GpuPolyShard shard;
        shard.field_index = component;
        shard.field_offset = 0;
        shard.limb_begin = 0;
        shard.limb_count = limb_count;
        shard.coeff_begin = 0;
        shard.coeff_count = degree;
        poly.shards.push_back(shard);

        result.polys_.push_back(std::move(poly));
    }

    return result;
}

}  // namespace gpu
}  // namespace poseidon
