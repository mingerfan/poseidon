// CPU-only isolated getter benchmark; not a full-model performance result.
#include "poseidon/poseidon_context.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
thread_local bool count_allocations = false;
thread_local std::uint64_t allocation_calls = 0, allocation_bytes = 0;
}

void* operator new(std::size_t bytes)
{
    void* result = std::malloc(bytes ? bytes : 1);
    if (!result) throw std::bad_alloc();
    if (count_allocations) {
        ++allocation_calls;
        allocation_bytes += bytes;
    }
    return result;
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

using Map = std::unordered_map<std::uint32_t, poseidon::parms_id_type>;
using Clock = std::chrono::steady_clock;

// The private getters differ only in their return type. Both have the same
// barrier and cannot be inlined/cloned or hoisted out of the dynamic lookup loop.
__attribute__((noinline, noclone)) const Map copy_getter(const Map& map)
{
    asm volatile("" : : "g"(&map) : "memory");
    return map;
}
__attribute__((noinline, noclone)) const Map& reference_getter(const Map& map)
{
    asm volatile("" : : "g"(&map) : "memory");
    return map;
}

template<bool Copy>
__attribute__((noinline, noclone)) std::uint64_t lookups(
    const Map& map, const std::vector<std::uint32_t>& levels)
{
    std::uint64_t checksum = 0;
    for (auto level : levels) {
        const auto id = [&] {
            if constexpr (Copy) return copy_getter(map).at(level);
            else return reference_getter(map).at(level);
        }();
        for (auto word : id) checksum = (checksum * 1099511628211ULL) ^ word;
    }
    return checksum;
}

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template<bool Copy>
std::uint64_t timed(const Map& map, const std::vector<std::uint32_t>& levels, int repeat)
{
    require(!count_allocations, "allocation instrumentation must be disabled during timing");
    const auto begin = Clock::now();
    const auto checksum = lookups<Copy>(map, levels);
    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    std::cout << "{\"event\":\"timing\",\"mode\":\"" << (Copy ? "copy" : "reference")
              << "\",\"repeat\":" << repeat << ",\"lookups\":" << levels.size()
              << ",\"milliseconds\":" << elapsed << ",\"checksum\":" << checksum
              << ",\"allocation_counting_enabled\":false}\n";
    return checksum;
}

template<bool Copy>
std::uint64_t counted(const Map& map, const std::vector<std::uint32_t>& levels)
{
    allocation_calls = allocation_bytes = 0;
    count_allocations = true;
    const auto checksum = lookups<Copy>(map, levels);
    count_allocations = false;
    const auto calls = allocation_calls, bytes = allocation_bytes;
    if constexpr (Copy) require(calls > 0, "copy getter unexpectedly performed no heap allocations");
    else require(calls == 0 && bytes == 0, "reference getter unexpectedly allocated heap storage");
    std::cout << "{\"event\":\"allocation_count\",\"mode\":\"" << (Copy ? "copy" : "reference")
              << "\",\"lookups\":" << levels.size() << ",\"new_calls\":" << calls
              << ",\"requested_allocation_bytes\":" << bytes << ",\"checksum\":" << checksum
              << ",\"scope\":\"separate counted pass; C++ new requested totals, not live/RSS peak\"}\n";
    return checksum;
}

int main(int argc, char** argv)
{
    try {
        const std::size_t iterations = argc > 1 ? std::stoull(argv[1]) : 1000000;
        std::uint32_t random = argc > 2 ? std::stoul(argv[2]) : 1729;
        require(argc <= 3 && iterations >= 1000 && random != 0, "usage: benchmark [lookups>=1000] [nonzero_u32_seed]");
        const auto seed = random;
        const auto context_begin = Clock::now();
        const auto primes = poseidon::CoeffModulus::Create(16384, std::vector<int>(58, 30));
        const std::vector<poseidon::Modulus> q(primes.begin(), primes.begin() + 50);
        const std::vector<poseidon::Modulus> p(primes.begin() + 50, primes.end());
        poseidon::ParametersLiteral parameters(CKKS, 14, 13, 40, 0, 0, poseidon::Modulus(0), q, p,
                                              poseidon::sec_level_type::none);
        const poseidon::PoseidonContext context(parameters);
        const auto owner = context.crt_context();
        const Map map = owner->parms_id_map();  // One untimed snapshot of the real context's map.
        require(map.size() == 51 && map.count(57), "expected 50 data levels and one Q+P key level");
        std::uint64_t map_checksum = 0;
        for (std::uint32_t level = 0; level < 50; ++level)
            for (auto word : map.at(level)) map_checksum = map_checksum * 1099511628211ULL ^ word;
        for (auto word : map.at(57)) map_checksum = map_checksum * 1099511628211ULL ^ word;
        const auto context_ms = std::chrono::duration<double, std::milli>(Clock::now() - context_begin).count();
        std::vector<std::uint32_t> levels(iterations);
        std::array<bool, 50> seen{};
        for (auto& level : levels) {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            level = random % 50;
            seen[level] = true;
        }
        require(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }), "not every data level was queried");
        std::cout << std::setprecision(17)
                  << "{\"event\":\"context\",\"poly_degree\":16384,\"q_count\":50,\"p_count\":8,\"scale_log2\":40"
                  << ",\"map_entries\":" << map.size() << ",\"bucket_count\":" << map.bucket_count()
                  << ",\"queried_levels\":50,\"seed\":" << seed << ",\"context_and_snapshot_ms\":" << context_ms
                  << ",\"map_checksum\":" << map_checksum << ",\"model_performance_claim\":false}\n";
        const std::vector<std::uint32_t> warmup(levels.begin(), levels.begin() + 1000);
        require(lookups<true>(map, warmup) == lookups<false>(map, warmup), "warmup checksum mismatch");
        std::uint64_t expected = 0;
        for (int repeat = 0; repeat < 3; ++repeat) {
            std::uint64_t copy, reference;
            if (repeat % 2 == 0) {
                copy = timed<true>(map, levels, repeat);
                reference = timed<false>(map, levels, repeat);
            } else {
                reference = timed<false>(map, levels, repeat);
                copy = timed<true>(map, levels, repeat);
            }
            require(copy == reference && (repeat == 0 || copy == expected), "timed checksum mismatch");
            expected = copy;
        }
        require(counted<true>(map, levels) == expected && counted<false>(map, levels) == expected,
                "counted checksum mismatch");
        std::cout << "{\"event\":\"complete\",\"passed\":true,\"full_model_measured\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        count_allocations = false;
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
