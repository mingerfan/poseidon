#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace poseidon
{
namespace gpu
{
namespace kernel
{

namespace
{

enum class NttAlgorithm
{
    Stage,
    Fused,
    FourStep
};

__device__ __forceinline__ GpuWord barrett_reduce_u64_u32(
    GpuWide value,
    GpuWord modulus,
    GpuWide barrett_ratio)
{
    const GpuWide quotient = __umul64hi(value, barrett_ratio);
    GpuWide reduced =
        value - quotient * static_cast<GpuWide>(modulus);

    if (reduced >= modulus)
    {
        reduced -= modulus;
    }
    if (reduced >= modulus)
    {
        reduced -= modulus;
    }

    return static_cast<GpuWord>(reduced);
}

__device__ __forceinline__ GpuWord add_mod(
    GpuWord left,
    GpuWord right,
    GpuWord modulus)
{
    const GpuWide sum =
        static_cast<GpuWide>(left) + static_cast<GpuWide>(right);
    return static_cast<GpuWord>(
        sum >= modulus ? sum - modulus : sum);
}

__device__ __forceinline__ GpuWord sub_mod(
    GpuWord left,
    GpuWord right,
    GpuWord modulus)
{
    return left >= right
        ? static_cast<GpuWord>(left - right)
        : static_cast<GpuWord>(
              static_cast<GpuWide>(left) + modulus - right);
}

__device__ __forceinline__ GpuWord mul_mod(
    GpuWord left,
    GpuWord right,
    GpuWord modulus,
    GpuWide barrett_ratio)
{
    return barrett_reduce_u64_u32(
        static_cast<GpuWide>(left) * static_cast<GpuWide>(right),
        modulus,
        barrett_ratio);
}

__device__ __forceinline__ void forward_butterfly(
    GpuWord &left,
    GpuWord &right,
    GpuWord root,
    GpuWord modulus,
    GpuWide barrett_ratio)
{
    const GpuWord u = left;
    const GpuWord v = mul_mod(right, root, modulus, barrett_ratio);

    left = add_mod(u, v, modulus);
    right = sub_mod(u, v, modulus);
}

__device__ __forceinline__ void inverse_butterfly(
    GpuWord &left,
    GpuWord &right,
    GpuWord root,
    GpuWord modulus,
    GpuWide barrett_ratio)
{
    const GpuWord u = left;
    const GpuWord v = right;

    left = add_mod(u, v, modulus);
    right = mul_mod(sub_mod(u, v, modulus), root, modulus, barrett_ratio);
}

bool is_power_of_two(std::size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

std::size_t log2_power_of_two(std::size_t value)
{
    std::size_t result = 0;
    while (value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

NttAlgorithm requested_ntt_algorithm()
{
    const char *raw = std::getenv("POSEIDON_NTT_ALGO");
    if (raw == nullptr || *raw == '\0')
    {
        return NttAlgorithm::Fused;
    }

    const std::string value(raw);
    if (value == "stage" || value == "baseline" || value == "single")
    {
        return NttAlgorithm::Stage;
    }
    if (value == "fused" || value == "fusion")
    {
        return NttAlgorithm::Fused;
    }
    if (value == "fourstep" || value == "four_step" ||
        value == "cheddar")
    {
        return NttAlgorithm::FourStep;
    }

    throw std::invalid_argument(
        "POSEIDON_NTT_ALGO must be stage, fused, or fourstep");
}

int requested_ntt_fusion_stages()
{
    constexpr int kDefaultFusionStages = 1;
    constexpr int kMaxFusionStages = 4;

    const char *raw = std::getenv("POSEIDON_NTT_FUSION_STAGES");
    if (raw == nullptr || *raw == '\0')
    {
        return kDefaultFusionStages;
    }

    char *end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || value <= 1)
    {
        return kDefaultFusionStages;
    }
    if (value > kMaxFusionStages)
    {
        return kMaxFusionStages;
    }
    return static_cast<int>(value);
}

void validate_ntt_launch_shape(
    const char *name,
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    bool inverse)
{
    if (destination_shard.ptr == nullptr ||
        source_shard.ptr == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null data pointer");
    }
    if (parameter_shard.rns_primes.data() == nullptr ||
        parameter_shard.rns_modulus_constants.data() == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null modulus table pointer");
    }
    if (parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null NTT table pointer");
    }
    if (inverse &&
        (parameter_shard.intt_tables.data() == nullptr ||
         parameter_shard.inv_degree_modulo.data() == nullptr))
    {
        throw std::invalid_argument(std::string(name) + ": null inverse NTT table pointer");
    }
    if (!inverse && parameter_shard.ntt_tables.data() == nullptr)
    {
        throw std::invalid_argument(std::string(name) + ": null forward NTT table pointer");
    }
    if (!is_power_of_two(degree) || degree < 2)
    {
        throw std::invalid_argument(std::string(name) + ": degree must be a power of two");
    }
    if (destination_shard.device_id != source_shard.device_id ||
        destination_shard.device_id != parameter_shard.device_id)
    {
        throw std::invalid_argument(std::string(name) + ": device mismatch");
    }
    if (destination_shard.limb_begin != source_shard.limb_begin ||
        destination_shard.limb_count != source_shard.limb_count ||
        destination_shard.coeff_begin != source_shard.coeff_begin ||
        destination_shard.coeff_count != source_shard.coeff_count)
    {
        throw std::invalid_argument(std::string(name) + ": shard shape mismatch");
    }
    if (destination_shard.coeff_begin != 0 ||
        destination_shard.coeff_count != degree)
    {
        throw std::invalid_argument(
            std::string(name) + ": first implementation requires full coefficient shards");
    }
    if (destination_shard.limb_count == 0)
    {
        throw std::invalid_argument(std::string(name) + ": empty limb range");
    }
    if (destination_shard.limb_begin < parameter_shard.limb_begin)
    {
        throw std::invalid_argument(std::string(name) + ": parameter shard does not cover limb range");
    }

    const std::size_t modulus_offset =
        destination_shard.limb_begin - parameter_shard.limb_begin;
    if (modulus_offset + destination_shard.limb_count >
            parameter_shard.rns_primes.size() ||
        modulus_offset + destination_shard.limb_count >
            parameter_shard.rns_modulus_constants.size())
    {
        throw std::invalid_argument(std::string(name) + ": RNS modulus tables do not cover limb range");
    }

    const std::size_t roots_size =
        (modulus_offset + destination_shard.limb_count) * degree;
    if (parameter_shard.ntt_tables.size() < roots_size)
    {
        throw std::invalid_argument(std::string(name) + ": NTT roots do not cover limb range");
    }
    if (inverse &&
        (parameter_shard.intt_tables.size() < roots_size ||
         parameter_shard.inv_degree_modulo.size() <
             modulus_offset + destination_shard.limb_count))
    {
        throw std::invalid_argument(std::string(name) + ": inverse NTT tables do not cover limb range");
    }
}

// NTT的一次迭代计算，forward NTT循环发射kernel，NTT的计算需要完整的coeff。
__global__ void forward_ntt_stage_kernel(
    GpuWord *values,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    std::size_t modulus_offset,/*？*/
    std::size_t limb_count,
    std::size_t degree, /*NTT需要完整的多项式，所以直接输入多项式阶数而非片段的coeff*/
    std::size_t m,
    std::size_t gap)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 每个线程独立计算一个BU，通过degree >> 1得到所需的蝶形算子
    const std::size_t butterflies_per_limb = degree >> 1;
    // RNS分量*对应的BU个数=所需启动的总线程
    const std::size_t total = limb_count * butterflies_per_limb;
    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / butterflies_per_limb;
    const std::size_t local_butterfly = tid % butterflies_per_limb;
    const std::size_t group = local_butterfly / gap; /*group代表在当前迭代轮次下的每组不同twiddle的个数*/
    const std::size_t j = local_butterfly % gap;/*比如NTT-8三次迭代，gap=4->2->1。第一次j=0,1,2,3*/
    const std::size_t table_limb = modulus_offset + local_limb;
    /*butterfly的两个输入位置,当前的degree+2*local_butterfly*/
    const std::size_t x_index = local_limb * degree + group * (gap << 1) + j;
    const std::size_t y_index = x_index + gap;

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    /*当前 stage、当前 group 要用的 twiddle factor，其中m为group个数；cpu侧root存储逻辑就是：stage1:root1；stage2:root2 root3...*/
    const GpuWord root = roots[table_limb * degree + m + group];

    const GpuWord u = values[x_index];
    const GpuWord v = mul_mod(values[y_index], root, modulus, barrett_ratio);

    values[x_index] = add_mod(u, v, modulus);
    values[y_index] = sub_mod(u, v, modulus);
}

template <int FusionStages>
__global__ void forward_ntt_fused_stage_kernel(
    GpuWord *values,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    std::size_t modulus_offset,
    std::size_t limb_count,
    std::size_t degree,
    std::size_t m,
    std::size_t gap)
{
    static_assert(
        FusionStages >= 2 && FusionStages <= 4,
        "NTT fusion currently supports 2-4 stages");

    constexpr std::size_t kLocalSize =
        static_cast<std::size_t>(1) << FusionStages;

    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t tiles_per_limb = degree >> FusionStages;
    const std::size_t total = limb_count * tiles_per_limb;
    if (tid >= total)
    {
        return;
    }

    /*确定属于第几个limb,以及在当前limb下属于第几个线程组组(线程组指的是由于融合，一个线程需要处理2^k个元素，这样的一个线程叫做线程组)*/
    const std::size_t local_limb = tid / tiles_per_limb;
    const std::size_t local_tile = tid % tiles_per_limb;
    const std::size_t table_limb = modulus_offset + local_limb;
    /*gap为当次迭代的蝶形输入的位置间隔，final_gap表示融合后的间距*/
    const std::size_t final_gap = gap >> (FusionStages - 1);
    /**/
    const std::size_t outer_group = local_tile / final_gap;
    const std::size_t j = local_tile % final_gap;
    const std::size_t base_index =
        local_limb * degree + outer_group * (gap << 1) + j;

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord *limb_roots = roots + table_limb * degree;

    GpuWord local[kLocalSize];
#pragma unroll
    for (std::size_t i = 0; i < kLocalSize; ++i)
    {
        local[i] = values[base_index + i * final_gap];
    }

#pragma unroll
    for (int stage = 0; stage < FusionStages; ++stage)
    {
        const std::size_t local_stride =
            static_cast<std::size_t>(1) << (FusionStages - 1 - stage);
        const std::size_t stage_m = m << stage;
        const std::size_t stage_group_base = outer_group << stage;

#pragma unroll
        for (std::size_t block = 0; block < kLocalSize;
             block += (local_stride << 1))
        {
            const std::size_t block_group =
                block / (local_stride << 1);
            const GpuWord root =
                limb_roots[stage_m + stage_group_base + block_group];

#pragma unroll
            for (std::size_t offset = 0; offset < local_stride; ++offset)
            {
                forward_butterfly(
                    local[block + offset],
                    local[block + offset + local_stride],
                    root,
                    modulus,
                    barrett_ratio);
            }
        }
    }

#pragma unroll
    for (std::size_t i = 0; i < kLocalSize; ++i)
    {
        values[base_index + i * final_gap] = local[i];
    }
}

// INTT,NTT的逆操作，最后还需要multiply_inv_degree_kernel整体除以N
__global__ void inverse_ntt_stage_kernel(
    GpuWord *values,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    std::size_t modulus_offset,
    std::size_t limb_count,
    std::size_t degree,
    std::size_t m,
    std::size_t gap)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t butterflies_per_limb = degree >> 1;
    const std::size_t total = limb_count * butterflies_per_limb;
    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / butterflies_per_limb;
    const std::size_t local_butterfly = tid % butterflies_per_limb;
    const std::size_t group = local_butterfly / gap;
    const std::size_t j = local_butterfly % gap;
    const std::size_t table_limb = modulus_offset + local_limb;
    const std::size_t root_base = degree - (m << 1) + 1;
    const std::size_t x_index =
        local_limb * degree + group * (gap << 1) + j;
    const std::size_t y_index = x_index + gap;

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord root = roots[table_limb * degree + root_base + group];

    const GpuWord u = values[x_index];
    const GpuWord v = values[y_index];

    values[x_index] = add_mod(u, v, modulus);
    values[y_index] = mul_mod(
        sub_mod(u, v, modulus),
        root,
        modulus,
        barrett_ratio);
}

template <int FusionStages>
__global__ void inverse_ntt_fused_stage_kernel(
    GpuWord *values,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    std::size_t modulus_offset,
    std::size_t limb_count,
    std::size_t degree,
    std::size_t m,
    std::size_t gap)
{
    static_assert(
        FusionStages >= 2 && FusionStages <= 4,
        "INTT fusion currently supports 2-4 stages");

    constexpr std::size_t kLocalSize =
        static_cast<std::size_t>(1) << FusionStages;

    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t tiles_per_limb = degree >> FusionStages;
    const std::size_t total = limb_count * tiles_per_limb;
    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / tiles_per_limb;
    const std::size_t local_tile = tid % tiles_per_limb;
    const std::size_t table_limb = modulus_offset + local_limb;
    const std::size_t outer_group = local_tile / gap;
    const std::size_t j = local_tile % gap;
    const std::size_t base_index =
        local_limb * degree + outer_group * (gap << FusionStages) + j;

    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord *limb_roots = roots + table_limb * degree;

    GpuWord local[kLocalSize];
#pragma unroll
    for (std::size_t i = 0; i < kLocalSize; ++i)
    {
        local[i] = values[base_index + i * gap];
    }

#pragma unroll
    for (int stage = 0; stage < FusionStages; ++stage)
    {
        const std::size_t local_stride =
            static_cast<std::size_t>(1) << stage;
        const std::size_t stage_m = m >> stage;
        const std::size_t stage_root_base =
            degree - (stage_m << 1) + 1;
        const std::size_t stage_group_base =
            outer_group << (FusionStages - 1 - stage);

#pragma unroll
        for (std::size_t block = 0; block < kLocalSize;
             block += (local_stride << 1))
        {
            const std::size_t block_group =
                block / (local_stride << 1);
            const GpuWord root =
                limb_roots[stage_root_base + stage_group_base + block_group];

#pragma unroll
            for (std::size_t offset = 0; offset < local_stride; ++offset)
            {
                inverse_butterfly(
                    local[block + offset],
                    local[block + offset + local_stride],
                    root,
                    modulus,
                    barrett_ratio);
            }
        }
    }

#pragma unroll
    for (std::size_t i = 0; i < kLocalSize; ++i)
    {
        values[base_index + i * gap] = local[i];
    }
}

template <int StageCount>
__global__ void forward_ntt_fourstep_phase_kernel(
    GpuWord *values,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    std::size_t modulus_offset,
    std::size_t limb_count,
    std::size_t degree,
    std::size_t m,
    std::size_t gap,
    std::size_t sets_per_block,
    std::size_t set_blocks_per_limb)
{
    static_assert(
        StageCount >= 1 && StageCount <= 9,
        "four-step NTT phase supports 1-9 stages");

    constexpr std::size_t kLocalSize =
        static_cast<std::size_t>(1) << StageCount;
    extern __shared__ GpuWord shared_values[];

    const std::size_t block_linear = blockIdx.x;
    const std::size_t local_limb = block_linear / set_blocks_per_limb;
    if (local_limb >= limb_count)
    {
        return;
    }

    const std::size_t set_block = block_linear % set_blocks_per_limb;
    const std::size_t sets_per_limb = degree >> StageCount;
    const std::size_t table_limb = modulus_offset + local_limb;
    const std::size_t final_gap = gap >> (StageCount - 1);
    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord *limb_roots = roots + table_limb * degree;
    const std::size_t shared_count = sets_per_block * kLocalSize;

    for (std::size_t shared_index = threadIdx.x;
         shared_index < shared_count;
         shared_index += blockDim.x)
    {
        const std::size_t local_set = shared_index / kLocalSize;
        const std::size_t local_index = shared_index % kLocalSize;
        const std::size_t global_set =
            set_block * sets_per_block + local_set;
        if (global_set >= sets_per_limb)
        {
            continue;
        }

        const std::size_t outer_group = global_set / final_gap;
        const std::size_t j = global_set % final_gap;
        const std::size_t base_index =
            local_limb * degree + outer_group * (gap << 1) + j;
        shared_values[shared_index] =
            values[base_index + local_index * final_gap];
    }
    __syncthreads();

    for (int chunk_start = 0; chunk_start < StageCount;)
    {
        const int remaining_stages = StageCount - chunk_start;
        int chunk_stages = remaining_stages > 3 ? 3 : remaining_stages;
        if constexpr (StageCount == 7)
        {
            chunk_stages = chunk_start == 0 ? 3 : 4;
        }
        const std::size_t chunk_local_size =
            static_cast<std::size_t>(1) << chunk_stages;
        const std::size_t chunk_final_stride =
            static_cast<std::size_t>(1)
            << (StageCount - chunk_start - chunk_stages);
        const std::size_t chunk_tiles_per_set =
            static_cast<std::size_t>(1) << (StageCount - chunk_stages);
        const std::size_t chunk_tile_count =
            sets_per_block * chunk_tiles_per_set;
        const bool is_last_chunk =
            chunk_start + chunk_stages == StageCount;

        for (std::size_t tile_index = threadIdx.x;
             tile_index < chunk_tile_count;
             tile_index += blockDim.x)
        {
            const std::size_t local_set =
                tile_index / chunk_tiles_per_set;
            const std::size_t global_set =
                set_block * sets_per_block + local_set;
            if (global_set >= sets_per_limb)
            {
                continue;
            }

            const std::size_t chunk_tile =
                tile_index % chunk_tiles_per_set;
            const std::size_t chunk_outer_group =
                chunk_tile / chunk_final_stride;
            const std::size_t chunk_offset =
                chunk_tile % chunk_final_stride;
            const std::size_t chunk_initial_stride =
                chunk_final_stride << (chunk_stages - 1);
            const std::size_t base_local_index =
                chunk_outer_group * (chunk_initial_stride << 1) +
                chunk_offset;
            const std::size_t outer_group = global_set / final_gap;
            GpuWord *local_values =
                shared_values + local_set * kLocalSize;

            GpuWord local[16];
#pragma unroll
            for (std::size_t i = 0; i < 16; ++i)
            {
                if (i < chunk_local_size)
                {
                    local[i] =
                        local_values[base_local_index + i * chunk_final_stride];
                }
            }

            for (int stage = 0; stage < chunk_stages; ++stage)
            {
                const int global_stage = chunk_start + stage;
                const std::size_t local_stride =
                    static_cast<std::size_t>(1)
                    << (chunk_stages - 1 - stage);
                const std::size_t stage_m = m << global_stage;
                const std::size_t root_group_base =
                    (outer_group << global_stage) +
                    (chunk_outer_group << stage);

                for (std::size_t block = 0; block < chunk_local_size;
                     block += (local_stride << 1))
                {
                    const std::size_t block_group =
                        block / (local_stride << 1);
                    const GpuWord root =
                        limb_roots[
                            stage_m + root_group_base + block_group];

#pragma unroll
                    for (std::size_t offset = 0; offset < 8; ++offset)
                    {
                        if (offset < local_stride)
                        {
                            forward_butterfly(
                                local[block + offset],
                                local[block + offset + local_stride],
                                root,
                                modulus,
                                barrett_ratio);
                        }
                    }
                }
            }

            if (is_last_chunk)
            {
                const std::size_t j = global_set % final_gap;
                const std::size_t base_index =
                    local_limb * degree + outer_group * (gap << 1) + j;
#pragma unroll
                for (std::size_t i = 0; i < 16; ++i)
                {
                    if (i < chunk_local_size)
                    {
                        const std::size_t local_index =
                            base_local_index + i * chunk_final_stride;
                        values[base_index + local_index * final_gap] =
                            local[i];
                    }
                }
            }
            else
            {
#pragma unroll
                for (std::size_t i = 0; i < 16; ++i)
                {
                    if (i < chunk_local_size)
                    {
                        local_values[
                            base_local_index + i * chunk_final_stride] =
                            local[i];
                    }
                }
            }
        }
        if (!is_last_chunk)
        {
            __syncthreads();
        }
        chunk_start += chunk_stages;
    }
}

template <int StageCount>
__global__ void inverse_ntt_fourstep_phase_kernel(
    GpuWord *values,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *roots,
    std::size_t modulus_offset,
    std::size_t limb_count,
    std::size_t degree,
    std::size_t m,
    std::size_t gap,
    std::size_t sets_per_block,
    std::size_t set_blocks_per_limb)
{
    static_assert(
        StageCount >= 1 && StageCount <= 9,
        "four-step INTT phase supports 1-9 stages");

    constexpr std::size_t kLocalSize =
        static_cast<std::size_t>(1) << StageCount;
    extern __shared__ GpuWord shared_values[];

    const std::size_t block_linear = blockIdx.x;
    const std::size_t local_limb = block_linear / set_blocks_per_limb;
    if (local_limb >= limb_count)
    {
        return;
    }

    const std::size_t set_block = block_linear % set_blocks_per_limb;
    const std::size_t sets_per_limb = degree >> StageCount;
    const std::size_t table_limb = modulus_offset + local_limb;
    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];
    const GpuWord *limb_roots = roots + table_limb * degree;
    const std::size_t shared_count = sets_per_block * kLocalSize;

    for (std::size_t shared_index = threadIdx.x;
         shared_index < shared_count;
         shared_index += blockDim.x)
    {
        const std::size_t local_set = shared_index / kLocalSize;
        const std::size_t local_index = shared_index % kLocalSize;
        const std::size_t global_set =
            set_block * sets_per_block + local_set;
        if (global_set >= sets_per_limb)
        {
            continue;
        }

        const std::size_t outer_group = global_set / gap;
        const std::size_t j = global_set % gap;
        const std::size_t base_index =
            local_limb * degree + outer_group * (gap << StageCount) + j;
        shared_values[shared_index] =
            values[base_index + local_index * gap];
    }
    __syncthreads();

    for (int chunk_start = 0; chunk_start < StageCount;)
    {
        const int remaining_stages = StageCount - chunk_start;
        int chunk_stages = remaining_stages > 3 ? 3 : remaining_stages;
        if constexpr (StageCount == 7)
        {
            chunk_stages = chunk_start == 0 ? 3 : 4;
        }
        const std::size_t chunk_local_size =
            static_cast<std::size_t>(1) << chunk_stages;
        const std::size_t chunk_initial_stride =
            static_cast<std::size_t>(1) << chunk_start;
        const std::size_t chunk_tiles_per_set =
            static_cast<std::size_t>(1) << (StageCount - chunk_stages);
        const std::size_t chunk_tile_count =
            sets_per_block * chunk_tiles_per_set;
        const bool is_last_chunk =
            chunk_start + chunk_stages == StageCount;

        for (std::size_t tile_index = threadIdx.x;
             tile_index < chunk_tile_count;
             tile_index += blockDim.x)
        {
            const std::size_t local_set =
                tile_index / chunk_tiles_per_set;
            const std::size_t global_set =
                set_block * sets_per_block + local_set;
            if (global_set >= sets_per_limb)
            {
                continue;
            }

            const std::size_t chunk_tile =
                tile_index % chunk_tiles_per_set;
            const std::size_t chunk_outer_group =
                chunk_tile / chunk_initial_stride;
            const std::size_t chunk_offset =
                chunk_tile % chunk_initial_stride;
            const std::size_t base_local_index =
                chunk_outer_group *
                    (chunk_initial_stride << chunk_stages) +
                chunk_offset;
            const std::size_t outer_group = global_set / gap;
            GpuWord *local_values =
                shared_values + local_set * kLocalSize;

            GpuWord local[16];
#pragma unroll
            for (std::size_t i = 0; i < 16; ++i)
            {
                if (i < chunk_local_size)
                {
                    local[i] =
                        local_values[
                            base_local_index + i * chunk_initial_stride];
                }
            }

            for (int stage = 0; stage < chunk_stages; ++stage)
            {
                const int global_stage = chunk_start + stage;
                const std::size_t local_stride =
                    static_cast<std::size_t>(1) << stage;
                const std::size_t stage_m = m >> global_stage;
                const std::size_t stage_root_base =
                    degree - (stage_m << 1) + 1;
                const std::size_t root_group_base =
                    (outer_group << (StageCount - 1 - global_stage)) +
                    (chunk_outer_group << (chunk_stages - 1 - stage));

                for (std::size_t block = 0; block < chunk_local_size;
                     block += (local_stride << 1))
                {
                    const std::size_t block_group =
                        block / (local_stride << 1);
                    const GpuWord root =
                        limb_roots[
                            stage_root_base +
                            root_group_base +
                            block_group];

#pragma unroll
                    for (std::size_t offset = 0; offset < 8; ++offset)
                    {
                        if (offset < local_stride)
                        {
                            inverse_butterfly(
                                local[block + offset],
                                local[block + offset + local_stride],
                                root,
                                modulus,
                                barrett_ratio);
                        }
                    }
                }
            }

            if (is_last_chunk)
            {
                const std::size_t j = global_set % gap;
                const std::size_t base_index =
                    local_limb * degree +
                    outer_group * (gap << StageCount) + j;
#pragma unroll
                for (std::size_t i = 0; i < 16; ++i)
                {
                    if (i < chunk_local_size)
                    {
                        const std::size_t local_index =
                            base_local_index + i * chunk_initial_stride;
                        values[base_index + local_index * gap] = local[i];
                    }
                }
            }
            else
            {
#pragma unroll
                for (std::size_t i = 0; i < 16; ++i)
                {
                    if (i < chunk_local_size)
                    {
                        local_values[
                            base_local_index + i * chunk_initial_stride] =
                            local[i];
                    }
                }
            }
        }
        if (!is_last_chunk)
        {
            __syncthreads();
        }
        chunk_start += chunk_stages;
    }
}

/* INTT最后的归一化 */
__global__ void multiply_inv_degree_kernel(
    GpuWord *values,
    const GpuWord *rns_primes,
    const GpuWide *rns_modulus_constants,
    const GpuWord *inv_degree_modulo,
    std::size_t modulus_offset,
    std::size_t limb_count,
    std::size_t degree)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = limb_count * degree;
    if (tid >= total)
    {
        return;
    }

    const std::size_t local_limb = tid / degree;
    const std::size_t table_limb = modulus_offset + local_limb;
    const GpuWord modulus = rns_primes[table_limb];
    const GpuWide barrett_ratio = rns_modulus_constants[table_limb];

    values[tid] = mul_mod(
        values[tid],
        inv_degree_modulo[table_limb],
        modulus,
        barrett_ratio);
}

// 计算逻辑是原位置计算的，也就是说需要输出结果有输入的初始值，然后迭代为中间结果，最终计算完成//？有优势吗
void copy_source_to_destination(
    const char *name,
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard)
{
    if (destination_shard.ptr == source_shard.ptr)
    {
        return;
    }

    const std::size_t count =
        destination_shard.limb_count * destination_shard.coeff_count;
    gpu_check_cuda(
        cudaMemcpy(
            destination_shard.ptr,
            source_shard.ptr,
            count * sizeof(GpuWord),
            cudaMemcpyDeviceToDevice),
        name);
}

template <int FusionStages>
void launch_forward_fused_stage(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t modulus_offset,
    std::size_t m,
    std::size_t gap)
{
    const std::size_t total_tiles =
        shard.limb_count * (degree >> FusionStages);

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total_tiles + block_size - 1) / block_size);

    forward_ntt_fused_stage_kernel<FusionStages><<<grid_size, block_size>>>(
        shard.ptr,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.ntt_tables.data(),
        modulus_offset,
        shard.limb_count,
        degree,
        m,
        gap);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_forward_ntt_poly_shard fused stage kernel launch");
}

void launch_forward_stage_group(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t modulus_offset,
    std::size_t m,
    std::size_t gap,
    std::size_t stage_count)
{
    if (stage_count == 1)
    {
        const std::size_t total_butterflies =
            shard.limb_count * (degree >> 1);

        constexpr int block_size = 256;
        const int grid_size = static_cast<int>(
            (total_butterflies + block_size - 1) / block_size);

        forward_ntt_stage_kernel<<<grid_size, block_size>>>(
            shard.ptr,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.ntt_tables.data(),
            modulus_offset,
            shard.limb_count,
            degree,
            m,
            gap);
        gpu_check_cuda(
            cudaGetLastError(),
            "launch_forward_ntt_poly_shard stage kernel launch");
        return;
    }

    switch (stage_count)
    {
    case 2:
        launch_forward_fused_stage<2>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 3:
        launch_forward_fused_stage<3>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 4:
        launch_forward_fused_stage<4>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    default:
        throw std::invalid_argument(
            "launch_forward_ntt_poly_shard: unsupported fusion stage count");
    }
}

template <int FusionStages>
void launch_inverse_fused_stage(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t modulus_offset,
    std::size_t m,
    std::size_t gap)
{
    const std::size_t total_tiles =
        shard.limb_count * (degree >> FusionStages);

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total_tiles + block_size - 1) / block_size);

    inverse_ntt_fused_stage_kernel<FusionStages><<<grid_size, block_size>>>(
        shard.ptr,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.intt_tables.data(),
        modulus_offset,
        shard.limb_count,
        degree,
        m,
        gap);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_inverse_ntt_poly_shard fused stage kernel launch");
}

void launch_inverse_stage_group(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t modulus_offset,
    std::size_t m,
    std::size_t gap,
    std::size_t stage_count)
{
    if (stage_count == 1)
    {
        const std::size_t total_butterflies =
            shard.limb_count * (degree >> 1);

        constexpr int block_size = 256;
        const int grid_size = static_cast<int>(
            (total_butterflies + block_size - 1) / block_size);

        inverse_ntt_stage_kernel<<<grid_size, block_size>>>(
            shard.ptr,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.intt_tables.data(),
            modulus_offset,
            shard.limb_count,
            degree,
            m,
            gap);
        gpu_check_cuda(
            cudaGetLastError(),
            "launch_inverse_ntt_poly_shard stage kernel launch");
        return;
    }

    switch (stage_count)
    {
    case 2:
        launch_inverse_fused_stage<2>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 3:
        launch_inverse_fused_stage<3>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 4:
        launch_inverse_fused_stage<4>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    default:
        throw std::invalid_argument(
            "launch_inverse_ntt_poly_shard: unsupported fusion stage count");
    }
}

std::size_t fourstep_sets_per_block(std::size_t stage_count)
{
    if (stage_count >= 9)
    {
        return 1;
    }
    if (stage_count == 8)
    {
        return 4;
    }
    return 16;
}

template <int StageCount>
constexpr int fourstep_block_size()
{
    if constexpr (StageCount >= 9)
    {
        return 64;
    }
    if constexpr (StageCount == 7)
    {
        return 128;
    }
    return 256;
}

template <int StageCount>
void launch_forward_fourstep_phase(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t modulus_offset,
    std::size_t m,
    std::size_t gap)
{
    const std::size_t sets_per_limb = degree >> StageCount;
    const std::size_t sets_per_block =
        fourstep_sets_per_block(StageCount);
    const std::size_t set_blocks_per_limb =
        (sets_per_limb + sets_per_block - 1) / sets_per_block;
    const std::size_t total_blocks =
        shard.limb_count * set_blocks_per_limb;

    constexpr int block_size = fourstep_block_size<StageCount>();
    const int grid_size = static_cast<int>(total_blocks);
    const std::size_t shared_bytes =
        sets_per_block *
        (static_cast<std::size_t>(1) << StageCount) *
        sizeof(GpuWord);

    forward_ntt_fourstep_phase_kernel<StageCount>
        <<<grid_size, block_size, shared_bytes>>>(
            shard.ptr,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.ntt_tables.data(),
            modulus_offset,
            shard.limb_count,
            degree,
            m,
            gap,
            sets_per_block,
            set_blocks_per_limb);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_forward_ntt_poly_shard fourstep phase kernel launch");
}

template <int StageCount>
void launch_inverse_fourstep_phase(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t modulus_offset,
    std::size_t m,
    std::size_t gap)
{
    const std::size_t sets_per_limb = degree >> StageCount;
    const std::size_t sets_per_block =
        fourstep_sets_per_block(StageCount);
    const std::size_t set_blocks_per_limb =
        (sets_per_limb + sets_per_block - 1) / sets_per_block;
    const std::size_t total_blocks =
        shard.limb_count * set_blocks_per_limb;

    constexpr int block_size = fourstep_block_size<StageCount>();
    const int grid_size = static_cast<int>(total_blocks);
    const std::size_t shared_bytes =
        sets_per_block *
        (static_cast<std::size_t>(1) << StageCount) *
        sizeof(GpuWord);

    inverse_ntt_fourstep_phase_kernel<StageCount>
        <<<grid_size, block_size, shared_bytes>>>(
            shard.ptr,
            parameter_shard.rns_primes.data(),
            parameter_shard.rns_modulus_constants.data(),
            parameter_shard.intt_tables.data(),
            modulus_offset,
            shard.limb_count,
            degree,
            m,
            gap,
            sets_per_block,
            set_blocks_per_limb);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_inverse_ntt_poly_shard fourstep phase kernel launch");
}

void launch_forward_fourstep_phase_by_count(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t modulus_offset,
    std::size_t m,
    std::size_t gap,
    std::size_t stage_count)
{
    switch (stage_count)
    {
    case 0:
        break;
    case 1:
        launch_forward_fourstep_phase<1>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 2:
        launch_forward_fourstep_phase<2>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 3:
        launch_forward_fourstep_phase<3>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 4:
        launch_forward_fourstep_phase<4>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 5:
        launch_forward_fourstep_phase<5>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 6:
        launch_forward_fourstep_phase<6>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 7:
        launch_forward_fourstep_phase<7>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 8:
        launch_forward_fourstep_phase<8>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 9:
        launch_forward_fourstep_phase<9>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    default:
        throw std::invalid_argument(
            "launch_forward_ntt_poly_shard: fourstep phase supports at most 9 stages");
    }
}

void launch_inverse_fourstep_phase_by_count(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    std::size_t modulus_offset,
    std::size_t m,
    std::size_t gap,
    std::size_t stage_count)
{
    switch (stage_count)
    {
    case 0:
        break;
    case 1:
        launch_inverse_fourstep_phase<1>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 2:
        launch_inverse_fourstep_phase<2>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 3:
        launch_inverse_fourstep_phase<3>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 4:
        launch_inverse_fourstep_phase<4>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 5:
        launch_inverse_fourstep_phase<5>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 6:
        launch_inverse_fourstep_phase<6>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 7:
        launch_inverse_fourstep_phase<7>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 8:
        launch_inverse_fourstep_phase<8>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    case 9:
        launch_inverse_fourstep_phase<9>(
            shard, parameter_shard, degree, modulus_offset, m, gap);
        break;
    default:
        throw std::invalid_argument(
            "launch_inverse_ntt_poly_shard: fourstep phase supports at most 9 stages");
    }
}

void launch_forward_stage_schedule(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    int fusion_stages)
{
    const std::size_t modulus_offset =
        shard.limb_begin - parameter_shard.limb_begin;
    std::size_t remaining_stages = log2_power_of_two(degree);

    for (std::size_t m = 1, gap = degree >> 1;
         remaining_stages > 0;)
    {
        std::size_t stage_count = 1;
        if (fusion_stages > 1)
        {
            stage_count = remaining_stages % fusion_stages;
            if (stage_count == 0)
            {
                stage_count = static_cast<std::size_t>(fusion_stages);
            }
        }

        launch_forward_stage_group(
            shard,
            parameter_shard,
            degree,
            modulus_offset,
            m,
            gap,
            stage_count);

        m <<= stage_count;
        gap >>= stage_count;
        remaining_stages -= stage_count;
    }
}

void launch_inverse_stage_schedule(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree,
    int fusion_stages)
{
    const std::size_t modulus_offset =
        shard.limb_begin - parameter_shard.limb_begin;
    std::size_t remaining_stages = log2_power_of_two(degree);

    for (std::size_t m = degree >> 1, gap = 1;
         remaining_stages > 0;)
    {
        std::size_t stage_count = 1;
        if (fusion_stages > 1)
        {
            stage_count =
                remaining_stages > static_cast<std::size_t>(fusion_stages)
                    ? static_cast<std::size_t>(fusion_stages)
                    : remaining_stages;
        }

        launch_inverse_stage_group(
            shard,
            parameter_shard,
            degree,
            modulus_offset,
            m,
            gap,
            stage_count);

        m >>= stage_count;
        gap <<= stage_count;
        remaining_stages -= stage_count;
    }
}

void launch_forward_fourstep_stages(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    const std::size_t log_degree = log2_power_of_two(degree);
    const std::size_t phase1_stages =
        log_degree == 16 ? 7 : log_degree / 2;
    const std::size_t phase2_stages = log_degree - phase1_stages;

    const std::size_t modulus_offset =
        shard.limb_begin - parameter_shard.limb_begin;
    std::size_t m = 1;
    std::size_t gap = degree >> 1;

    launch_forward_fourstep_phase_by_count(
        shard,
        parameter_shard,
        degree,
        modulus_offset,
        m,
        gap,
        phase1_stages);

    m <<= phase1_stages;
    gap >>= phase1_stages;

    launch_forward_fourstep_phase_by_count(
        shard,
        parameter_shard,
        degree,
        modulus_offset,
        m,
        gap,
        phase2_stages);
}

void launch_inverse_fourstep_stages(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    const std::size_t log_degree = log2_power_of_two(degree);
    const std::size_t forward_phase1_stages =
        log_degree == 16 ? 7 : log_degree / 2;
    const std::size_t phase1_stages =
        log_degree - forward_phase1_stages;
    const std::size_t phase2_stages = forward_phase1_stages;

    const std::size_t modulus_offset =
        shard.limb_begin - parameter_shard.limb_begin;
    std::size_t m = degree >> 1;
    std::size_t gap = 1;

    launch_inverse_fourstep_phase_by_count(
        shard,
        parameter_shard,
        degree,
        modulus_offset,
        m,
        gap,
        phase1_stages);

    m >>= phase1_stages;
    gap <<= phase1_stages;

    launch_inverse_fourstep_phase_by_count(
        shard,
        parameter_shard,
        degree,
        modulus_offset,
        m,
        gap,
        phase2_stages);
}

void launch_forward_stages(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    switch (requested_ntt_algorithm())
    {
    case NttAlgorithm::Stage:
        launch_forward_stage_schedule(shard, parameter_shard, degree, 1);
        break;
    case NttAlgorithm::Fused:
        launch_forward_stage_schedule(
            shard,
            parameter_shard,
            degree,
            requested_ntt_fusion_stages());
        break;
    case NttAlgorithm::FourStep:
        launch_forward_fourstep_stages(shard, parameter_shard, degree);
        break;
    }
}

void launch_inverse_stages(
    const GpuPolyShardView &shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    switch (requested_ntt_algorithm())
    {
    case NttAlgorithm::Stage:
        launch_inverse_stage_schedule(shard, parameter_shard, degree, 1);
        break;
    case NttAlgorithm::Fused:
        launch_inverse_stage_schedule(
            shard,
            parameter_shard,
            degree,
            requested_ntt_fusion_stages());
        break;
    case NttAlgorithm::FourStep:
        launch_inverse_fourstep_stages(shard, parameter_shard, degree);
        break;
    }

    const std::size_t modulus_offset =
        shard.limb_begin - parameter_shard.limb_begin;
    const std::size_t total_values = shard.limb_count * degree;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (total_values + block_size - 1) / block_size);
    multiply_inv_degree_kernel<<<grid_size, block_size>>>(
        shard.ptr,
        parameter_shard.rns_primes.data(),
        parameter_shard.rns_modulus_constants.data(),
        parameter_shard.inv_degree_modulo.data(),
        modulus_offset,
        shard.limb_count,
        degree);
    gpu_check_cuda(
        cudaGetLastError(),
        "launch_inverse_ntt_poly_shard inv degree kernel launch");
}

}  // anonymous namespace

void launch_forward_ntt_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_ntt_launch_shape(
        "launch_forward_ntt_poly_shard",
        destination_shard,
        source_shard,
        parameter_shard,
        degree,
        false);

    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_forward_ntt_poly_shard cudaSetDevice");
    copy_source_to_destination(
        "launch_forward_ntt_poly_shard cudaMemcpyDeviceToDevice",
        destination_shard,
        source_shard);

    launch_forward_stages(destination_shard, parameter_shard, degree);
}

void launch_inverse_ntt_poly_shard(
    const GpuPolyShardView &destination_shard,
    const GpuConstPolyShardView &source_shard,
    const GpuParameterShard &parameter_shard,
    std::size_t degree)
{
    validate_ntt_launch_shape(
        "launch_inverse_ntt_poly_shard",
        destination_shard,
        source_shard,
        parameter_shard,
        degree,
        true);

    gpu_check_cuda(
        cudaSetDevice(destination_shard.device_id),
        "launch_inverse_ntt_poly_shard cudaSetDevice");
    copy_source_to_destination(
        "launch_inverse_ntt_poly_shard cudaMemcpyDeviceToDevice",
        destination_shard,
        source_shard);

    launch_inverse_stages(destination_shard, parameter_shard, degree);
}

}  // namespace kernel
}  // namespace gpu
}  // namespace poseidon
