#include "poseidon/gpu/gpu_key.h"

#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{
using namespace poseidon::gpu;

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

class CacheSetting
{
public:
    explicit CacheSetting(const char *value)
    {
        const char *previous = std::getenv("POSEIDON_GPU_KEY_VIEW_CACHE");
        had_previous_ = previous != nullptr;
        if (had_previous_) previous_ = previous;
        require((value ? setenv("POSEIDON_GPU_KEY_VIEW_CACHE", value, 1)
                       : unsetenv("POSEIDON_GPU_KEY_VIEW_CACHE")) == 0,
                "could not set cache switch");
    }
    ~CacheSetting()
    {
        if (had_previous_) (void)setenv("POSEIDON_GPU_KEY_VIEW_CACHE", previous_.c_str(), 1);
        else (void)unsetenv("POSEIDON_GPU_KEY_VIEW_CACHE");
    }
private:
    bool had_previous_ = false;
    std::string previous_;
};

void check_cache_switch_and_invalidation()
{
    CacheSetting default_enabled(nullptr);
    GpuEvaluationKeyData key;
    key.meta.degree = 8;
    key.meta.q_count = 5;
    const auto &low = key.const_view(1);
    const auto &high = key.const_view(5);
    require(&low == &key.const_view(1), "same Q lookup did not reuse its view");
    require(&high == &key.const_view(5), "second Q lookup did not reuse its view");
    require(&low != &high && low.meta.q_count == 1 && high.meta.q_count == 5,
            "different Q lookups clobbered one another");
    {
        CacheSetting disabled("0");
        const auto &first = key.const_view(1);
        const auto &second = key.const_view(1);
        require(&first != &second && &first != &low,
                "disabled cache reused a snapshot");
        for (std::size_t i = 0; i < 128; ++i) (void)key.const_view(i % 5 + 1);
        require(first.meta.q_count == 1 && first.storage_q_count == 5 &&
                    second.meta.degree == 8 && low.meta.q_count == 1,
                "later disabled lookups invalidated an outstanding reference");
    }
    require(&low == &key.const_view(1), "re-enabling cache lost its reusable snapshot");

    // Retire every borrowed reference before invalidating or mutating the key.
    key.invalidate_const_view_cache();
    key.meta.degree = 16;
    key.polys_.push_back(GpuRNSPoly{});
    key.polys_.back().poly_id = 7;
    require(key.const_view(1).meta.degree == 16 &&
                key.const_view(1).polys.front().poly_id == 7,
            "explicit invalidation did not rebuild changed metadata");
    (void)key.make_view(); // Also retires the cached const view.
    key.meta.degree = 32;
    key.polys_.front().poly_id = 9;
    require(key.const_view(1).meta.degree == 32 &&
                key.const_view(1).polys.front().poly_id == 9,
            "mutable make_view did not invalidate const views");
    for (const char *setting : {"0", "1"})
    {
        CacheSetting cache_setting(setting);
        key.invalidate_const_view_cache();
        key.meta.degree += 8;
        const auto &view = key.const_view(1);
        require(view.meta.degree == key.meta.degree,
                "invalidation did not rebuild the current cache mode");
        for (std::size_t invalid : {std::size_t{0}, std::size_t{6}})
        {
            bool rejected = false;
            try { (void)key.const_view(invalid); }
            catch (const std::invalid_argument &) { rejected = true; }
            require(rejected, "lookup accepted an invalid active Q count");
        }
    }
}

void check_concurrent_lookups(const char *setting)
{
    CacheSetting cache_setting(setting);
    GpuEvaluationKeyData key;
    key.meta.degree = 8;
    key.meta.q_count = 3;
    constexpr std::size_t thread_count = 4, lookups_per_thread = 24;
    std::array<std::array<const GpuConstEvaluationKeyView *, lookups_per_thread>, thread_count> views{};
    std::array<std::exception_ptr, thread_count> errors{};
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < thread_count; ++t)
    {
        threads.emplace_back([&, t] {
            try
            {
                for (std::size_t i = 0; i < lookups_per_thread; ++i)
                    views[t][i] = &key.const_view(i % 3 + 1);
            }
            catch (...) { errors[t] = std::current_exception(); }
        });
    }
    for (auto &thread : threads) thread.join();
    for (const auto &error : errors) if (error) std::rethrow_exception(error);
    std::set<const GpuConstEvaluationKeyView *> distinct;
    for (const auto &per_thread : views)
    {
        for (std::size_t i = 0; i < lookups_per_thread; ++i)
        {
            require(per_thread[i]->meta.q_count == i % 3 + 1 &&
                        per_thread[i]->storage_q_count == 3,
                    "concurrent lookup returned corrupted/dangling metadata");
            distinct.insert(per_thread[i]);
        }
    }
    require(distinct.size() == (std::string(setting) == "0" ? thread_count * lookups_per_thread : 3),
            "concurrent lookups did not honor the cache switch");
}

constexpr std::size_t degree = 8, p_count = 2, field_padding = 3;

GpuEvaluationKeyData make_key(std::size_t storage_q_count)
{
    GpuEvaluationKeyData key;
    key.meta.degree = degree;
    key.meta.q_count = storage_q_count;
    key.meta.p_count = p_count;
    key.meta.key_count = 2; // Slot 1 is intentionally absent from the pointer tables.
    key.meta.decomposition_count = 2;
    key.meta.component_count = 2;
    key.meta.galois_format = GpuGaloisKeyFormat::InversePreRotated;
    key.galois_elts_by_key_index = {3, 0};
    const std::size_t poly_words = (storage_q_count + p_count) * degree;
    std::vector<GpuWord> words(field_padding + 4 * poly_words, 0);
    key.fields_.emplace_back(0, words.size());
    std::vector<const GpuWord *> q0(4, nullptr), p0(4, nullptr), q1(4, nullptr), p1(4, nullptr);
    for (std::size_t poly_id = 0; poly_id < 4; ++poly_id)
    {
        const auto offset = field_padding + poly_id * poly_words;
        GpuRNSPoly poly;
        poly.poly_id = poly_id;
        poly.degree = degree;
        poly.q_count = storage_q_count;
        poly.p_count = p_count;
        poly.shards.push_back({0, offset, 0, storage_q_count + p_count, 0, degree});
        key.polys_.push_back(std::move(poly));
        key.poly_metadata_.push_back({poly_id, 0, poly_id / 2, poly_id % 2});
        for (std::size_t limb = 0; limb < storage_q_count + p_count; ++limb)
            for (std::size_t coefficient = 0; coefficient < degree; ++coefficient)
                words[offset + limb * degree + coefficient] = static_cast<GpuWord>(
                    10000 * (poly_id + 1) + 100 * limb + coefficient);
        auto &q = poly_id % 2 == 0 ? q0 : q1;
        auto &p = poly_id % 2 == 0 ? p0 : p1;
        q[poly_id / 2] = key.fields_.front().data() + offset;
        p[poly_id / 2] = q[poly_id / 2] + storage_q_count * degree;
    }
    key.fields_.front().buffer.copy_from_host(words.data(), words.size());
    auto upload_table = [](DeviceVector<const GpuWord *> &table, const std::vector<const GpuWord *> &pointers) {
        table.allocate(pointers.size(), 0);
        table.copy_from_host(pointers.data(), pointers.size());
    };
    upload_table(key.galois_key_q0_ptrs, q0);
    upload_table(key.galois_key_p0_ptrs, p0);
    upload_table(key.galois_key_q1_ptrs, q1);
    upload_table(key.galois_key_p1_ptrs, p1);
    return key;
}

void check_storage_offsets(const GpuEvaluationKeyData &key, std::size_t active_q_count)
{
    const auto &view = key.const_view(active_q_count);
    const auto uncached = key.make_const_view(active_q_count);
    require(view.meta.q_count == active_q_count && view.storage_q_count == key.meta.q_count &&
                view.meta.p_count == p_count && view.polys.size() == key.polys_.size(),
            "cached view lost active/storage Q metadata");
    require(key.fields_.front().size() == field_padding + 4 * (key.meta.q_count + p_count) * degree,
            "test key does not model physically trimmed Q storage");
    for (std::size_t i = 0; i < view.polys.size(); ++i)
    {
        const auto &shard = view.polys[i].shards.front();
        require(shard.ptr == uncached.polys[i].shards.front().ptr &&
                    shard.ptr == key.fields_.front().data() + key.polys_[i].shards.front().field_offset &&
                    shard.limb_count == key.meta.q_count + p_count,
                "cached view changed authoritative storage/shard layout");
        std::vector<GpuWord> p_words(p_count * degree);
        gpu_check_cuda(cudaMemcpyAsync(p_words.data(), shard.ptr + view.storage_q_count * degree,
                                      p_words.size() * sizeof(GpuWord), cudaMemcpyDeviceToHost,
                                      gpu_execution_stream()), "test download P limbs");
        gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "test synchronize P limbs");
        for (std::size_t j = 0; j < p_words.size(); ++j)
            require(p_words[j] == 10000 * (i + 1) + 100 * (key.meta.q_count + j / degree) + j % degree,
                    "low-Q cached view addressed P at active Q instead of storage Q");
    }
}

void check_pointer_tables(const GpuEvaluationKeyData &key)
{
    const std::array<const DeviceVector<const GpuWord *> *, 4> tables{
        &key.galois_key_q0_ptrs, &key.galois_key_p0_ptrs,
        &key.galois_key_q1_ptrs, &key.galois_key_p1_ptrs};
    for (std::size_t t = 0; t < tables.size(); ++t)
    {
        std::vector<const GpuWord *> pointers(tables[t]->size());
        tables[t]->copy_to_host(pointers.data(), pointers.size());
        require(pointers.size() == 4 && pointers[2] == nullptr && pointers[3] == nullptr,
                "copy changed sparse Galois pointer table slots");
        for (std::size_t digit = 0; digit < 2; ++digit)
        {
            const auto poly_id = 2 * digit + t / 2;
            const auto offset = key.polys_[poly_id].shards.front().field_offset +
                                (t % 2 == 0 ? 0 : key.meta.q_count * degree);
            require(pointers[digit] == key.fields_.front().data() + offset,
                    "deep copy retained a Galois pointer into the source key");
        }
    }
}

void check_deep_copies()
{
    CacheSetting enabled("1");
    std::unique_ptr<GpuEvaluationKeyData> copied;
    auto assigned = make_key(2);
    (void)assigned.const_view(1); // Assignment must discard this older cache.
    std::vector<GpuWord> expected;
    {
        auto source = make_key(5);
        const auto &source_view = source.const_view(1);
        (void)source.const_view(3);
        copied = std::make_unique<GpuEvaluationKeyData>(source);
        assigned = source;
        require(copied->const_view(1).polys.front().shards.front().ptr !=
                    source_view.polys.front().shards.front().ptr &&
                    assigned.const_view(1).polys.front().shards.front().ptr !=
                    source_view.polys.front().shards.front().ptr,
                "copy reused source storage/cache pointers");
        expected.resize(source.fields_.front().size());
        source.fields_.front().buffer.copy_to_host(expected.data(), expected.size());
        source.invalidate_const_view_cache();
        source.fields_.front().buffer.fill_zero();
        gpu_check_cuda(cudaStreamSynchronize(gpu_execution_stream()), "test source mutation");
    } // The original allocations and cached view objects are now destroyed.
    for (const auto *copy : {copied.get(), &assigned})
    {
        std::vector<GpuWord> actual(copy->fields_.front().size());
        copy->fields_.front().buffer.copy_to_host(actual.data(), actual.size());
        require(actual == expected, "copy did not retain independent key residues");
        check_storage_offsets(*copy, 1);
        check_storage_offsets(*copy, 3);
        check_pointer_tables(*copy);
    }
}

void check_pointer_rebase_edges()
{
    CacheSetting enabled("1");
    GpuEvaluationKeyData source;
    source.meta.degree = 4;
    source.meta.q_count = 1;
    source.meta.key_count = source.meta.decomposition_count = 1;
    source.meta.component_count = 2;
    source.meta.galois_format = GpuGaloisKeyFormat::InversePreRotated;
    for (std::size_t component = 0; component < 2; ++component)
    {
        const std::size_t offset = component + 1;
        std::vector<GpuWord> words(offset + source.meta.degree, static_cast<GpuWord>(component + 11));
        source.fields_.emplace_back(0, words.size());
        source.fields_.back().buffer.copy_from_host(words.data(), words.size());
        GpuRNSPoly poly;
        poly.poly_id = component;
        poly.degree = source.meta.degree;
        poly.q_count = 1; // No P: the P-table entry is exactly one-past its field.
        poly.shards.push_back({component, offset, 0, 1, 0, poly.degree});
        source.polys_.push_back(std::move(poly));
        source.poly_metadata_.push_back({component, 0, 0, component});
    }
    std::array<DeviceVector<const GpuWord *> *, 4> tables{
        &source.galois_key_q0_ptrs, &source.galois_key_p0_ptrs,
        &source.galois_key_q1_ptrs, &source.galois_key_p1_ptrs};
    for (std::size_t i = 0; i < tables.size(); ++i)
    {
        const auto field = i / 2;
        const GpuWord *pointer = source.fields_[field].data() + field + 1 +
                                 (i % 2 == 0 ? 0 : source.meta.degree);
        tables[i]->allocate(1, 0);
        tables[i]->copy_from_host(&pointer, 1);
    }
    (void)source.const_view(1);
    GpuEvaluationKeyData copy(source);
    const std::array<const DeviceVector<const GpuWord *> *, 4> copied_tables{
        &copy.galois_key_q0_ptrs, &copy.galois_key_p0_ptrs,
        &copy.galois_key_q1_ptrs, &copy.galois_key_p1_ptrs};
    for (std::size_t i = 0; i < copied_tables.size(); ++i)
    {
        const auto field = i / 2;
        const GpuWord *pointer = nullptr;
        copied_tables[i]->copy_to_host(&pointer, 1);
        require(pointer == copy.fields_[field].data() + field + 1 +
                               (i % 2 == 0 ? 0 : copy.meta.degree),
                "multi-field/zero-P copy did not rebase pointer tables");
        require(copy.const_view(1).polys[field].shards.front().ptr ==
                    copy.fields_[field].data() + field + 1,
                "multi-field cache points into the wrong copied field");
    }

    DeviceVector<GpuWord> external(1, 0);
    const GpuWord *outside = external.data();
    source.invalidate_const_view_cache();
    source.galois_key_q0_ptrs.copy_from_host(&outside, 1);
    bool rejected = false;
    try { GpuEvaluationKeyData invalid_copy(source); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "copy silently retained an out-of-owner pointer table entry");
}

void check_storage_replacement()
{
    CacheSetting enabled("1");
    auto key = make_key(2);
    const auto old_address = reinterpret_cast<std::uintptr_t>(
        key.const_view(1).polys.front().shards.front().ptr);
    auto replacement = make_key(2); // Allocate before releasing the old storage.
    key.invalidate_const_view_cache();
    key.fields_ = std::move(replacement.fields_);
    // Layout is unchanged, but setup-time tables must follow the new storage.
    key.galois_key_q0_ptrs = std::move(replacement.galois_key_q0_ptrs);
    key.galois_key_p0_ptrs = std::move(replacement.galois_key_p0_ptrs);
    key.galois_key_q1_ptrs = std::move(replacement.galois_key_q1_ptrs);
    key.galois_key_p1_ptrs = std::move(replacement.galois_key_p1_ptrs);
    require(reinterpret_cast<std::uintptr_t>(key.const_view(1).polys.front().shards.front().ptr) !=
                old_address,
            "explicit invalidation retained a replaced allocation");
    check_storage_offsets(key, 1);
    check_pointer_tables(key);
}

void check_moves(const char *setting)
{
    CacheSetting cache_setting(setting);
    auto source = make_key(5);
    const auto *borrowed = &source.const_view(1);
    const auto *device_pointer = borrowed->polys.front().shards.front().ptr;
    auto *self = &source;
    source = *self;
    source = std::move(*self);
    require(!source.empty() && borrowed->polys.front().shards.front().ptr ==
                source.fields_.front().data() + field_padding,
            "self copy/move assignment invalidated key storage");
    GpuEvaluationKeyData moved(std::move(source));
    source = make_key(2); // Replacing the moved-from owner must not retire our view.
    check_storage_offsets(source, 1);
    require(borrowed->polys.front().shards.front().ptr == device_pointer,
            "move construction invalidated a borrowed view");
    if (std::string(setting) != "0")
        require(&moved.const_view(1) == borrowed, "move construction discarded cached views");
    auto destination = make_key(2);
    (void)destination.const_view(1);
    destination = std::move(moved);
    moved = GpuEvaluationKeyData{};
    require(borrowed->polys.front().shards.front().ptr == device_pointer &&
                device_pointer == destination.fields_.front().data() + field_padding,
            "move assignment did not transfer view/storage ownership together");
    if (std::string(setting) != "0")
        require(&destination.const_view(1) == borrowed, "move assignment discarded cached views");
    check_storage_offsets(destination, 1);
    check_pointer_tables(destination);
}

struct TestMemoryResource
{
    rmm::mr::cuda_memory_resource resource;
    rmm::mr::device_memory_resource *previous = rmm::mr::set_current_device_resource(&resource);
    ~TestMemoryResource() { rmm::mr::set_current_device_resource(previous); }
};
} // namespace

int main()
{
    try
    {
        check_cache_switch_and_invalidation();
        check_concurrent_lookups("1");
        check_concurrent_lookups("0");
        std::cout << "[PASS] cache reuse/switch/invalidation/concurrent host lookups\n";
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        {
            std::cout << "[SKIP] device copy/move/P-offset checks require a CUDA device\n";
            return 77;
        }
        gpu_check_cuda(cudaSetDevice(0), "test select CUDA device");
        TestMemoryResource memory;
        {
            CacheSetting enabled("1");
            auto full = make_key(5);
            const auto *low = &full.const_view(1);
            (void)full.const_view(3);
            require(low == &full.const_view(1), "multi-Q device view was rebuilt");
            check_storage_offsets(full, 1);
            auto trimmed = make_key(2);
            check_storage_offsets(trimmed, 1);
            check_storage_offsets(trimmed, 2);
        }
        check_deep_copies();
        check_pointer_rebase_edges();
        check_storage_replacement();
        check_moves("1");
        check_moves("0");
        std::cout << "[PASS] independent deep copies, preserved moves, low-Q/trimmed P offsets\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] GPU key view cache: " << error.what() << '\n';
        return 1;
    }
}
