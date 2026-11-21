#include "poseidon/gpu/gpu_wide_ntt.h"
#include <cstdlib>

namespace poseidon::gpu
{
namespace
{
using Word = std::uint64_t;
using Operand = util::MultiplyUIntModOperand;

__device__ Word multiply(Word value, Operand constant, Word modulus)
{
    // Shoup multiplication. The low word contains the full remainder (<2q)
    // even when both products individually exceed 64 bits.
    const Word quotient = __umul64hi(value, constant.quotient);
    const Word remainder = value * constant.operand - quotient * modulus;
    return remainder >= modulus ? remainder - modulus : remainder;
}

template <bool Inverse>
__global__ void ntt_stage(Word *values, const Word *moduli, const Operand *roots,
                          const Operand *inverse_degrees, std::size_t limbs,
                          std::size_t degree, std::size_t groups, std::size_t gap,
                          std::size_t first_limb)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t half = degree / 2;
    if (tid >= limbs * half)
        return;

    const std::size_t limb = tid / half;
    const std::size_t butterfly = tid % half;
    const std::size_t group = butterfly / gap;
    const std::size_t first = limb * degree + group * (2 * gap) + butterfly % gap;
    const std::size_t second = first + gap;
    const std::size_t root_base = Inverse ? degree - 2 * groups + 1 : groups;
    const Operand root = roots[(first_limb + limb) * degree + root_base + group];
    const Word modulus = moduli[first_limb + limb];

    const Word left = values[first];
    const Word right = Inverse ? values[second] : multiply(values[second], root, modulus);
    Word sum = left + right;
    if (sum >= modulus)
        sum -= modulus;
    Word difference = left >= right ? left - right : left + modulus - right;

    if constexpr (Inverse)
    {
        difference = multiply(difference, root, modulus);
        if (groups == 1)
        {
            sum = multiply(sum, inverse_degrees[first_limb + limb], modulus);
            difference = multiply(difference, inverse_degrees[first_limb + limb], modulus);
        }
    }
    values[first] = sum;
    values[second] = difference;
}

// Two forward stages on four independent values; the operation order is unchanged.
__global__ void ntt_forward_pair(Word *values, const Word *moduli, const Operand *roots,
                                 std::size_t limbs, std::size_t degree, std::size_t groups,
                                 std::size_t gap, std::size_t first_limb)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t quarter = degree / 4;
    if (tid >= limbs * quarter) return;
    const std::size_t limb = tid / quarter, butterfly = tid % quarter;
    const std::size_t k = gap / 2, group = butterfly / k;
    const std::size_t base = limb * degree + group * (2 * gap) + butterfly % k;
    const Word q = moduli[first_limb + limb];
    const Operand r0 = roots[(first_limb + limb) * degree + groups + group];
    const Operand r1 = roots[(first_limb + limb) * degree + 2 * groups + 2 * group];
    const Operand r2 = roots[(first_limb + limb) * degree + 2 * groups + 2 * group + 1];
    Word a = values[base], b = values[base + k];
    const Word c = multiply(values[base + gap], r0, q);
    const Word d = multiply(values[base + gap + k], r0, q);
    Word x0 = a + c, x1 = b + d;
    if (x0 >= q) x0 -= q;
    if (x1 >= q) x1 -= q;
    const Word x2 = a >= c ? a - c : a + q - c;
    const Word x3 = b >= d ? b - d : b + q - d;
    const Word y1 = multiply(x1, r1, q), y3 = multiply(x3, r2, q);
    Word s0 = x0 + y1, s1 = x2 + y3;
    if (s0 >= q) s0 -= q;
    if (s1 >= q) s1 -= q;
    values[base] = s0;
    values[base + k] = x0 >= y1 ? x0 - y1 : x0 + q - y1;
    values[base + gap] = s1;
    values[base + gap + k] = x2 >= y3 ? x2 - y3 : x2 + q - y3;
}

// Three forward stages, preserving the original butterfly order.
__global__ void ntt_forward_triple(Word *values, const Word *moduli, const Operand *roots,
                                   std::size_t limbs, std::size_t degree, std::size_t groups,
                                   std::size_t gap, std::size_t first_limb)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t eighth = degree / 8;
    if (tid >= limbs * eighth) return;
    const std::size_t limb = tid / eighth, butterfly = tid % eighth;
    const std::size_t k = gap / 4, group = butterfly / k;
    const std::size_t base = limb * degree + group * (2 * gap) + butterfly % k;
    const Word q = moduli[first_limb + limb];
    Word x[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = values[base + j * k];
    #pragma unroll
    for (int stage = 0; stage < 3; ++stage)
    {
        const int width = 8 >> stage, half = width / 2;
        #pragma unroll
        for (int block = 0; block < (1 << stage); ++block)
        {
            const Operand root = roots[(first_limb + limb) * degree +
                (groups << stage) + (group << stage) + block];
            #pragma unroll
            for (int j = 0; j < half; ++j)
            {
                const int left = block * width + j, right = left + half;
                const Word a = x[left], b = multiply(x[right], root, q);
                Word sum = a + b;
                x[left] = sum >= q ? sum - q : sum;
                x[right] = a >= b ? a - b : a + q - b;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) values[base + j * k] = x[j];
}

__global__ void ntt_inverse_pair(Word *values, const Word *moduli, const Operand *roots,
                                 const Operand *inverse_degrees, std::size_t limbs,
                                 std::size_t degree, std::size_t groups, std::size_t gap,
                                 std::size_t first_limb)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t quarter = degree / 4;
    if (tid >= limbs * quarter) return;
    const std::size_t limb = tid / quarter, butterfly = tid % quarter;
    const std::size_t group = butterfly / gap;
    const std::size_t base = limb * degree + group * 4 * gap + butterfly % gap;
    const Word q = moduli[first_limb + limb];
    const std::size_t roots_base = (first_limb + limb) * degree;
    const Operand r0 = roots[roots_base + degree - 2 * groups + 1 + 2 * group];
    const Operand r1 = roots[roots_base + degree - 2 * groups + 2 + 2 * group];
    const Operand r2 = roots[roots_base + degree - groups + 1 + group];
    const Word a = values[base], b = values[base + gap];
    const Word c = values[base + 2 * gap], d = values[base + 3 * gap];
    Word x0 = a + b, x2 = c + d;
    if (x0 >= q) x0 -= q;
    if (x2 >= q) x2 -= q;
    const Word x1 = multiply(a >= b ? a - b : a + q - b, r0, q);
    const Word x3 = multiply(c >= d ? c - d : c + q - d, r1, q);
    Word s0 = x0 + x2, s1 = x1 + x3;
    if (s0 >= q) s0 -= q;
    if (s1 >= q) s1 -= q;
    Word d0 = multiply(x0 >= x2 ? x0 - x2 : x0 + q - x2, r2, q);
    Word d1 = multiply(x1 >= x3 ? x1 - x3 : x1 + q - x3, r2, q);
    if (groups == 2)
    {
        const Operand inv = inverse_degrees[first_limb + limb];
        s0 = multiply(s0, inv, q); s1 = multiply(s1, inv, q);
        d0 = multiply(d0, inv, q); d1 = multiply(d1, inv, q);
    }
    values[base] = s0; values[base + gap] = s1;
    values[base + 2 * gap] = d0; values[base + 3 * gap] = d1;
}

// Fuse butterfly stages that stay inside one aligned tile.
template <bool Inverse, int Tile>
__global__ void ntt_tile(Word *values, const Word *moduli, const Operand *roots,
                             std::size_t degree, std::size_t first_limb)
{
    __shared__ Word tile[Tile];
    const std::size_t tiles_per_limb = degree / Tile;
    const std::size_t limb = blockIdx.x / tiles_per_limb;
    const std::size_t offset = (blockIdx.x % tiles_per_limb) * Tile;
    const std::size_t base = limb * degree + offset;
    const std::size_t t = threadIdx.x;
    const Word modulus = moduli[first_limb + limb];
    tile[t] = values[base + t];
    tile[t + Tile / 2] = values[base + t + Tile / 2];
    __syncthreads();
    for (std::size_t stage = 0; (std::size_t{1} << stage) < Tile; ++stage)
    {
        const std::size_t gap = Inverse ? (std::size_t{1} << stage) : (std::size_t{Tile / 2} >> stage);
        const std::size_t groups = degree / (2 * gap);
        const std::size_t first = (t / gap) * (2 * gap) + t % gap;
        const std::size_t group = (offset + first) / (2 * gap);
        const std::size_t root_base = Inverse ? degree - 2 * groups + 1 : groups;
        const Operand root = roots[(first_limb + limb) * degree + root_base + group];
        const Word left = tile[first];
        const Word right = Inverse ? tile[first + gap] : multiply(tile[first + gap], root, modulus);
        Word sum = left + right;
        if (sum >= modulus) sum -= modulus;
        Word difference = left >= right ? left - right : left + modulus - right;
        if constexpr (Inverse) difference = multiply(difference, root, modulus);
        __syncthreads();
        tile[first] = sum;
        tile[first + gap] = difference;
        __syncthreads();
    }
    values[base + t] = tile[t];
    values[base + t + Tile / 2] = tile[t + Tile / 2];
}

__global__ void extend_q0(Word *out, const Word *in, const Word *moduli,
                          std::size_t degree, std::size_t limbs)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= degree * limbs)
        return;

    const Word q0 = moduli[0];
    const Word value = in[tid % degree];
    const Word modulus = moduli[tid / degree];
    const bool negative = value > (q0 >> 1);
    const Word residue = (negative ? q0 - value : value) % modulus;
    out[tid] = negative && residue != 0 ? modulus - residue : residue;
}
} // namespace

GpuWideNtt::GpuWideNtt(int log_degree, const std::vector<Modulus> &moduli, int device_id)
    : device_id_(device_id), degree_(0), limb_count_(moduli.size())
{
    if (log_degree < 1 || log_degree >= std::numeric_limits<std::size_t>::digits ||
        moduli.empty())
        throw std::invalid_argument("GpuWideNtt: invalid ring degree or empty RNS base");
    degree_ = std::size_t{1} << log_degree;
    if (degree_ > POSEIDON_POLY_MOD_DEGREE_MAX)
        throw std::invalid_argument("GpuWideNtt: degree exceeds Poseidon limit");
    if (limb_count_ > POSEIDON_COEFF_MOD_COUNT_MAX)
        throw std::invalid_argument("GpuWideNtt: RNS base exceeds supported modulus count");

    std::vector<Word> primes;
    std::vector<Operand> forward_roots, inverse_roots, inverse_degrees;
    for (const auto &modulus : moduli)
    {
        if (modulus.bit_count() > POSEIDON_USER_MOD_BIT_COUNT_MAX)
            throw std::invalid_argument("GpuWideNtt: modulus exceeds 60 bits");
        const util::NTTTables tables(log_degree, modulus);
        primes.push_back(modulus.value());
        const auto *forward = tables.get_from_root_powers();
        const auto *inverse = tables.get_from_inv_root_powers();
        forward_roots.insert(forward_roots.end(), forward, forward + degree_);
        inverse_roots.insert(inverse_roots.end(), inverse, inverse + degree_);
        inverse_degrees.push_back(tables.inv_degree_modulo());
    }
    moduli_.allocate(primes.size(), device_id_);
    moduli_.copy_from_host(primes.data(), primes.size());
    forward_roots_.allocate(forward_roots.size(), device_id_);
    forward_roots_.copy_from_host(forward_roots.data(), forward_roots.size());
    inverse_roots_.allocate(inverse_roots.size(), device_id_);
    inverse_roots_.copy_from_host(inverse_roots.data(), inverse_roots.size());
    inverse_degrees_.allocate(inverse_degrees.size(), device_id_);
    inverse_degrees_.copy_from_host(inverse_degrees.data(), inverse_degrees.size());
}

void GpuWideNtt::forward(DeviceVector<Word> &values) const
{
    if (values.size() != degree_ * limb_count_)
        throw std::invalid_argument("GpuWideNtt: full RNS buffer required");
    transform(values, false, 0);
}

void GpuWideNtt::inverse(DeviceVector<Word> &values) const
{
    if (values.size() != degree_ * limb_count_)
        throw std::invalid_argument("GpuWideNtt: full RNS buffer required");
    transform(values, true, 0);
}

void GpuWideNtt::forward_prefix(DeviceVector<Word> &values) const
{
    transform(values, false, 0);
}

void GpuWideNtt::inverse_limb(DeviceVector<Word> &values, std::size_t limb) const
{
    if (values.size() != degree_)
        throw std::invalid_argument("GpuWideNtt: a single limb is required");
    transform(values, true, limb);
}

void GpuWideNtt::extend_centered_q0(const DeviceVector<Word> &source,
                                    DeviceVector<Word> &destination) const
{
    if (source.size() != degree_ || destination.size() != degree_ * limb_count_ ||
        source.device_id() != device_id_ || destination.device_id() != device_id_ ||
        source.data() == destination.data())
        throw std::invalid_argument("GpuWideNtt: q0 extension requires distinct buffers on the plan device");

    gpu_check_cuda(cudaSetDevice(device_id_), "GpuWideNtt q0 extension cudaSetDevice");
    constexpr int threads = 256;
    const int blocks = static_cast<int>((destination.size() + threads - 1) / threads);
    extend_q0<<<blocks, threads, 0, gpu_execution_stream()>>>(
        destination.data(), source.data(), moduli_.data(), degree_, limb_count_);
    gpu_check_cuda(cudaGetLastError(), "GpuWideNtt q0 extension");
}

void GpuWideNtt::transform(DeviceVector<Word> &values, bool inverse,
                           std::size_t first_limb) const
{
    const std::size_t limbs = values.size() / degree_;
    if (first_limb >= limb_count_ || limbs > limb_count_ - first_limb ||
        limbs == 0 || values.size() % degree_ != 0 || values.device_id() != device_id_)
        throw std::invalid_argument("GpuWideNtt: RNS buffer shape or device mismatch");
    gpu_check_cuda(cudaSetDevice(device_id_), "GpuWideNtt cudaSetDevice");

    constexpr int threads = 256;
    const int blocks = static_cast<int>((limbs * (degree_ / 2) + threads - 1) / threads);
    const auto stream = gpu_execution_stream();
    const char *tile_env = std::getenv("POSEIDON_WIDE_NTT_TILE");
    const std::string tile_text = tile_env ? tile_env : "512";
    if (tile_text != "512" && tile_text != "1024" && tile_text != "2048")
        throw std::invalid_argument("POSEIDON_WIDE_NTT_TILE must be 512, 1024, or 2048");
    const std::size_t tile_size = std::stoul(tile_text);
    const bool fused = degree_ > tile_size && std::getenv("POSEIDON_WIDE_NTT_FUSED") != nullptr;
    if (inverse)
    {
        if (fused)
        {
            if (tile_size == 512) ntt_tile<true, 512><<<limbs * (degree_ / 512), 256, 0, stream>>>(values.data(), moduli_.data(), inverse_roots_.data(), degree_, first_limb);
            else if (tile_size == 1024) ntt_tile<true, 1024><<<limbs * (degree_ / 1024), 512, 0, stream>>>(values.data(), moduli_.data(), inverse_roots_.data(), degree_, first_limb);
            else ntt_tile<true, 2048><<<limbs * (degree_ / 2048), 1024, 0, stream>>>(values.data(), moduli_.data(), inverse_roots_.data(), degree_, first_limb);
            gpu_check_cuda(cudaGetLastError(), "GpuWideNtt inverse tile");
        }
        for (std::size_t groups = degree_ / 2, gap = 1; groups; groups /= 2, gap *= 2)
        {
            if (fused && gap <= tile_size / 2) continue;
            if (groups >= 2 && std::getenv("POSEIDON_WIDE_INTT_PAIR"))
            {
                const int pair_blocks = static_cast<int>((limbs * (degree_ / 4) + threads - 1) / threads);
                ntt_inverse_pair<<<pair_blocks, threads, 0, stream>>>(values.data(), moduli_.data(),
                    inverse_roots_.data(), inverse_degrees_.data(), limbs, degree_, groups, gap, first_limb);
                gpu_check_cuda(cudaGetLastError(), "GpuWideNtt inverse paired stages");
                groups /= 2; gap *= 2;
                continue;
            }
            ntt_stage<true><<<blocks, threads, 0, stream>>>(
                values.data(), moduli_.data(), inverse_roots_.data(),
                inverse_degrees_.data(), limbs, degree_, groups, gap, first_limb);
            gpu_check_cuda(cudaGetLastError(), "GpuWideNtt inverse stage");
        }
    }
    else
    {
        for (std::size_t groups = 1, gap = degree_ / 2; gap; groups *= 2, gap /= 2)
        {
            if (fused && gap <= tile_size / 2) break;
            if (fused && gap > 2 * tile_size && std::getenv("POSEIDON_WIDE_NTT_TRIPLE"))
            {
                const int triple_blocks = static_cast<int>((limbs * (degree_ / 8) + threads - 1) / threads);
                ntt_forward_triple<<<triple_blocks, threads, 0, stream>>>(values.data(), moduli_.data(),
                    forward_roots_.data(), limbs, degree_, groups, gap, first_limb);
                gpu_check_cuda(cudaGetLastError(), "GpuWideNtt forward triple stages");
                groups *= 4; gap /= 4;
                continue;
            }
            if (fused && gap > tile_size && std::getenv("POSEIDON_WIDE_NTT_PAIR"))
            {
                const int pair_blocks = static_cast<int>((limbs * (degree_ / 4) + threads - 1) / threads);
                ntt_forward_pair<<<pair_blocks, threads, 0, stream>>>(values.data(), moduli_.data(),
                    forward_roots_.data(), limbs, degree_, groups, gap, first_limb);
                gpu_check_cuda(cudaGetLastError(), "GpuWideNtt forward paired stages");
                groups *= 2; gap /= 2;
                continue;
            }
            ntt_stage<false><<<blocks, threads, 0, stream>>>(
                values.data(), moduli_.data(), forward_roots_.data(),
                nullptr, limbs, degree_, groups, gap, first_limb);
            gpu_check_cuda(cudaGetLastError(), "GpuWideNtt forward stage");
        }
        if (fused)
        {
            if (tile_size == 512) ntt_tile<false, 512><<<limbs * (degree_ / 512), 256, 0, stream>>>(values.data(), moduli_.data(), forward_roots_.data(), degree_, first_limb);
            else if (tile_size == 1024) ntt_tile<false, 1024><<<limbs * (degree_ / 1024), 512, 0, stream>>>(values.data(), moduli_.data(), forward_roots_.data(), degree_, first_limb);
            else ntt_tile<false, 2048><<<limbs * (degree_ / 2048), 1024, 0, stream>>>(values.data(), moduli_.data(), forward_roots_.data(), degree_, first_limb);
            gpu_check_cuda(cudaGetLastError(), "GpuWideNtt forward tile");
        }
    }
}

} // namespace poseidon::gpu
