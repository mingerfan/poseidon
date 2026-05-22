#include "poseidon/gpu/gpu_plaintext.h"

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
        throw std::out_of_range("GpuPlaintextData shard field_index is out of range");
    }
    if (shard.limb_count == 0 || shard.coeff_count == 0)
    {
        throw std::invalid_argument("GpuPlaintextData shard must be non-empty");
    }
    if (shard.limb_begin + shard.limb_count > poly.q_count + poly.p_count)
    {
        throw std::out_of_range("GpuPlaintextData shard limb range is out of range");
    }
    if (shard.coeff_begin + shard.coeff_count > poly.degree)
    {
        throw std::out_of_range("GpuPlaintextData shard coefficient range is out of range");
    }

    const auto &field = fields[shard.field_index];
    const auto shard_size = checked_mul(
        shard.limb_count,
        shard.coeff_count,
        "GpuPlaintextData shard size overflow");
    const auto physical_end = checked_add(
        shard.field_offset,
        shard_size,
        "GpuPlaintextData shard physical end overflow");
    if (physical_end > field.size())
    {
        throw std::out_of_range("GpuPlaintextData shard exceeds field allocation");
    }
}

}  // namespace

bool GpuPlaintextData::empty() const
{
    return fields_.empty();
}

GpuPlaintextView GpuPlaintextData::make_view()
{
    GpuPlaintextView result;
    result.meta = meta;
    result.poly.poly_id = poly_.poly_id;
    result.poly.shards.reserve(poly_.shards.size());

    for (const auto &shard : poly_.shards)
    {
        validate_shard(poly_, shard, fields_);

        auto &field = fields_[shard.field_index];
        GpuPolyShardView shard_view;
        shard_view.device_id = field.device_id;
        shard_view.ptr = field.data() + shard.field_offset;
        shard_view.limb_begin = shard.limb_begin;
        shard_view.limb_count = shard.limb_count;
        shard_view.coeff_begin = shard.coeff_begin;
        shard_view.coeff_count = shard.coeff_count;
        result.poly.shards.push_back(shard_view);
    }

    return result;
}

GpuConstPlaintextView GpuPlaintextData::make_const_view() const
{
    GpuConstPlaintextView result;
    result.meta = meta;
    result.poly.poly_id = poly_.poly_id;
    result.poly.shards.reserve(poly_.shards.size());

    for (const auto &shard : poly_.shards)
    {
        validate_shard(poly_, shard, fields_);

        const auto &field = fields_[shard.field_index];
        GpuConstPolyShardView shard_view;
        shard_view.device_id = field.device_id;
        shard_view.ptr = field.data() + shard.field_offset;
        shard_view.limb_begin = shard.limb_begin;
        shard_view.limb_count = shard.limb_count;
        shard_view.coeff_begin = shard.coeff_begin;
        shard_view.coeff_count = shard.coeff_count;
        result.poly.shards.push_back(shard_view);
    }

    return result;
}

GpuPlaintextData GpuPlaintextData::allocate_single_device(
    std::size_t degree,
    std::size_t q_count,
    int device_id,
    std::size_t p_count)
{
    if (degree == 0 || q_count + p_count == 0)
    {
        throw std::invalid_argument(
            "GpuPlaintextData::allocate_single_device requires non-zero shape");
    }

    GpuPlaintextData result;
    result.meta.degree = degree;
    result.meta.q_count = q_count;
    result.meta.p_count = p_count;

    const auto limb_count = q_count + p_count;
    const auto field_size = checked_mul(degree, limb_count, "GpuPlaintextData field size overflow");
    result.fields_.emplace_back(device_id, field_size);

    result.poly_.poly_id = 0;
    result.poly_.degree = degree;
    result.poly_.q_count = q_count;
    result.poly_.p_count = p_count;

    GpuPolyShard shard;
    shard.field_index = 0;
    shard.field_offset = 0;
    shard.limb_begin = 0;
    shard.limb_count = limb_count;
    shard.coeff_begin = 0;
    shard.coeff_count = degree;
    result.poly_.shards.push_back(shard);

    return result;
}

}  // namespace gpu
}  // namespace poseidon
