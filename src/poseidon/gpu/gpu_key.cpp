#include "poseidon/gpu/gpu_key.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

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

// Device tables contain pointers into fields_, so copying their bytes alone
// would leave entries pointing into the source owner.
const GpuWord *rebase_key_pointer(
    const GpuWord *pointer,
    const std::vector<GpuFieldData> &source_fields,
    const std::vector<GpuFieldData> &destination_fields)
{
    if (pointer == nullptr)
    {
        return nullptr;
    }

    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const GpuWord *one_past_end = nullptr;
    for (std::size_t i = 0; i < source_fields.size(); ++i)
    {
        const auto &source = source_fields[i];
        if (source.empty())
        {
            continue;
        }
        const auto begin = reinterpret_cast<std::uintptr_t>(source.data());
        const auto bytes = checked_mul(
            source.size(), sizeof(GpuWord), "GPU key pointer range overflow");
        if (address < begin)
        {
            continue;
        }
        const auto offset = address - begin;
        if (offset < bytes && offset % sizeof(GpuWord) == 0)
        {
            return destination_fields[i].data() + offset / sizeof(GpuWord);
        }
        if (offset == bytes)
        {
            // A zero-length P segment can be one-past. Prefer an actual
            // in-allocation match when allocations are adjacent.
            one_past_end = destination_fields[i].data() + source.size();
        }
    }
    if (one_past_end != nullptr)
    {
        return one_past_end;
    }
    throw std::invalid_argument(
        "GpuEvaluationKeyData copy: pointer table entry is outside key storage");
}

void copy_key_pointer_table(
    const DeviceVector<const GpuWord *> &source,
    DeviceVector<const GpuWord *> &destination,
    const std::vector<GpuFieldData> &source_fields,
    const std::vector<GpuFieldData> &destination_fields)
{
    std::vector<const GpuWord *> pointers(source.size());
    source.copy_to_host(pointers.data(), pointers.size());
    for (auto &pointer : pointers)
    {
        pointer = rebase_key_pointer(pointer, source_fields, destination_fields);
    }
    destination.allocate(source.size(), source.device_id());
    destination.copy_from_host(pointers.data(), pointers.size());
}

}  // namespace

GpuEvaluationKeyData::GpuEvaluationKeyData(const GpuEvaluationKeyData &other)
    : meta(other.meta), polys_(other.polys_),
      poly_metadata_(other.poly_metadata_),
      galois_elts_by_key_index(other.galois_elts_by_key_index),
      level_keys(other.level_keys)
{
    fields_.reserve(other.fields_.size());
    for (const auto &field : other.fields_)
    {
        fields_.emplace_back(field.device_id, field.size());
        if (!field.empty())
        {
            gpu_check_cuda(
                cudaMemcpyAsync(
                    fields_.back().data(), field.data(),
                    checked_mul(field.size(), sizeof(GpuWord), "GPU key copy size overflow"),
                    cudaMemcpyDeviceToDevice, gpu_execution_stream()),
                "GpuEvaluationKeyData copy cudaMemcpyAsync");
            gpu_check_cuda(
                cudaStreamSynchronize(gpu_execution_stream()),
                "GpuEvaluationKeyData copy cudaStreamSynchronize");
        }
    }
    copy_key_pointer_table(
        other.galois_key_q0_ptrs, galois_key_q0_ptrs, other.fields_, fields_);
    copy_key_pointer_table(
        other.galois_key_p0_ptrs, galois_key_p0_ptrs, other.fields_, fields_);
    copy_key_pointer_table(
        other.galois_key_q1_ptrs, galois_key_q1_ptrs, other.fields_, fields_);
    copy_key_pointer_table(
        other.galois_key_p1_ptrs, galois_key_p1_ptrs, other.fields_, fields_);
    // view_cache_ starts empty: its pointers must be rebuilt from our fields_.
}

GpuEvaluationKeyData &GpuEvaluationKeyData::operator=(const GpuEvaluationKeyData &other)
{
    if (this != &other)
    {
        GpuEvaluationKeyData copy(other);
        *this = std::move(copy);
    }
    return *this;
}

GpuEvaluationKeyData &GpuEvaluationKeyData::operator=(GpuEvaluationKeyData &&other) noexcept
{
    if (this != &other)
    {
        meta = other.meta;
        fields_ = std::move(other.fields_);
        polys_ = std::move(other.polys_);
        poly_metadata_ = std::move(other.poly_metadata_);
        galois_elts_by_key_index = std::move(other.galois_elts_by_key_index);
        galois_key_q0_ptrs = std::move(other.galois_key_q0_ptrs);
        galois_key_p0_ptrs = std::move(other.galois_key_p0_ptrs);
        galois_key_q1_ptrs = std::move(other.galois_key_q1_ptrs);
        galois_key_p1_ptrs = std::move(other.galois_key_p1_ptrs);
        level_keys = std::move(other.level_keys);
        view_cache_ = std::move(other.view_cache_);
    }
    return *this;
}

bool GpuEvaluationKeyData::empty() const
{
    return polys_.empty();
}

GpuEvaluationKeyView GpuEvaluationKeyData::make_view()
{
    invalidate_const_view_cache();
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
    result.storage_q_count = meta.q_count;
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

GpuConstEvaluationKeyView GpuEvaluationKeyData::make_const_view(
    std::size_t active_q_count) const
{
    if (active_q_count == 0 || active_q_count > meta.q_count)
    {
        throw std::invalid_argument(
            "GpuEvaluationKeyData::make_const_view: active q_count is outside the stored Q prefix");
    }

    auto result = make_const_view();
    result.meta.q_count = active_q_count;
    return result;
}

const GpuConstEvaluationKeyView &GpuEvaluationKeyData::const_view(
    std::size_t active_q_count) const
{
    std::lock_guard<std::mutex> lock(view_cache_.mutex);
    const char *cache_setting = std::getenv("POSEIDON_GPU_KEY_VIEW_CACHE");
    if (cache_setting != nullptr && std::strcmp(cache_setting, "0") == 0)
    {
        // Keep every snapshot alive until invalidation, including across lookups.
        view_cache_.uncached_views.push_back(make_const_view(active_q_count));
        return view_cache_.uncached_views.back();
    }
    auto it = view_cache_.views.find(active_q_count);
    if (it == view_cache_.views.end())
    {
        it = view_cache_.views.emplace(
            active_q_count, std::make_unique<GpuConstEvaluationKeyView>(
                                make_const_view(active_q_count))).first;
    }
    return *it->second;
}

void GpuEvaluationKeyData::invalidate_const_view_cache() const
{
    std::lock_guard<std::mutex> lock(view_cache_.mutex);
    view_cache_.views.clear();
    view_cache_.uncached_views.clear();
}

const GpuEvaluationKeyData &GpuEvaluationKeyData::for_level(std::size_t q_count) const
{
    if (level_keys)
    {
        const auto found = level_keys->find(q_count);
        if (found != level_keys->end() && found->second)
        {
            return *found->second;
        }
    }
    return *this;
}

}  // namespace gpu
}  // namespace poseidon
