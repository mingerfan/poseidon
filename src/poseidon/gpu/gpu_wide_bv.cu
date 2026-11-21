#include "poseidon/gpu/gpu_wide_bv.h"
#include <cstdlib>

namespace poseidon::gpu
{
namespace
{
using Word = std::uint64_t;
using Limb = detail::WideBvLimb;
constexpr int threads = 256;

template<bool Pack>
__global__ void key_56(std::uint64_t *wide, std::uint8_t *packed, std::size_t count)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if constexpr (Pack)
    {
        const auto value = wide[i];
        for (int j = 0; j < 7; ++j) packed[i * 7 + j] = value >> (8 * j);
    }
    else
    {
        std::uint64_t value = 0;
        for (int j = 0; j < 7; ++j) value |= std::uint64_t(packed[i * 7 + j]) << (8 * j);
        wide[i] = value;
    }
}

int blocks(std::size_t count)
{
    return static_cast<int>((count + threads - 1) / threads);
}

// Same two-word Barrett reduction as the CPU util::barrett_reduce_128.
__device__ Word multiply(Word a, Word b, Word modulus, Word ratio_low, Word ratio_high)
{
    const Word low = a * b;
    const Word high = __umul64hi(a, b);
    const Word carry = __umul64hi(low, ratio_low);
    const Word cross_low = low * ratio_high;
    const Word cross_high = high * ratio_low;
    const Word middle = cross_low + carry;
    const Word quotient = high * ratio_high + __umul64hi(low, ratio_high) +
                          __umul64hi(high, ratio_low) + (middle < cross_low) +
                          (middle + cross_high < middle);
    const Word remainder = low - quotient * modulus;
    return remainder >= modulus ? remainder - modulus : remainder;
}

// ratio_high = floor(2^64 / modulus). For a 64-bit value the estimated
// quotient is at most one too small; one correction yields the exact residue.
__device__ Word reduce_word(Word value, const Limb &limb, bool fast)
{
    if (!fast) return value % limb.modulus;
    const Word estimate = __umul64hi(value, limb.ratio_high);
    const Word remainder = value - estimate * limb.modulus;
    return remainder >= limb.modulus ? remainder - limb.modulus : remainder;
}

void copy_device(Word *destination, const Word *source, std::size_t words, const char *label)
{
    gpu_check_cuda(cudaMemcpyAsync(destination, source, words * sizeof(Word),
                                 cudaMemcpyDeviceToDevice, gpu_execution_stream()), label);
}
__global__ void lift_digit(Word *destination, const Word *source, const Limb *limbs,
                                     std::size_t degree, std::size_t q_count, bool fast)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < degree * (q_count + 1))
        destination[i] = reduce_word(source[i % degree], limbs[i / degree], fast);
}

__global__ void permute_coefficients(Word *out, const Word *in, const Limb *limbs,
                                     std::size_t degree, std::size_t words, std::uint32_t element)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words) return;
    const std::size_t limb = i / degree;
    const std::size_t exponent = ((i % degree) * element) % (2 * degree);
    const Word value = in[i];
    out[limb * degree + exponent % degree] =
        exponent >= degree && value ? limbs[limb].modulus - value : value;
}

__global__ void accumulate(Word *sum, const Word *digit, const Word *key,
                                     const Limb *limbs, std::size_t degree, std::size_t qp_words)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= qp_words)
        return;

    const Limb limb = limbs[i / degree];
    for (std::size_t component = 0; component < 2; ++component)
    {
        const std::size_t offset = i + component * qp_words;
        const Word product = multiply(digit[i], key[offset], limb.modulus,
                                      limb.ratio_low, limb.ratio_high);
        const Word value = sum[offset] + product;
        sum[offset] = value >= limb.modulus ? value - limb.modulus : value;
    }
}

// Accumulate cached hoisted digits in the same modular order as accumulate().
__global__ void accumulate_cached_digits(Word *sum, const Word *digits, const Word *keys,
                                          const Limb *limbs, std::size_t degree,
                                          std::size_t qp_words, std::size_t q_count)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= qp_words) return;
    const Limb limb = limbs[i / degree];
    Word total0 = 0, total1 = 0;
    for (std::size_t digit = 0; digit < q_count; ++digit)
    {
        const Word value = digits[digit * qp_words + i];
        const Word *key = keys + digit * 2 * qp_words;
        const Word p0 = multiply(value, key[i], limb.modulus, limb.ratio_low, limb.ratio_high);
        const Word p1 = multiply(value, key[qp_words + i], limb.modulus, limb.ratio_low, limb.ratio_high);
        total0 += p0; total1 += p1;
        if (total0 >= limb.modulus) total0 -= limb.modulus;
        if (total1 >= limb.modulus) total1 -= limb.modulus;
    }
    sum[i] = total0;
    sum[qp_words + i] = total1;
}

__global__ void lift_centered(Word *destination, const Word *source, const Limb *limbs,
                                 std::size_t degree, std::size_t q_count, bool fast)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= degree * q_count)
        return;

    const Word p = limbs[q_count].modulus;
    const Limb limb = limbs[i / degree];
    const Word q = limb.modulus;
    const Word value = source[i % degree];
    const bool negative = value > (p >> 1);
    const Word magnitude = reduce_word(negative ? p - value : value, limb, fast);
    destination[i] = negative && magnitude != 0 ? q - magnitude : magnitude;
}

template <bool Rescale>
__global__ void finish(Word *destination, const Word *sum, const Word *p_in_q,
                                 const Limb *limbs, std::size_t degree, std::size_t q_words,
                       const Word *addend)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= q_words)
        return;

    const Limb limb = limbs[i / degree];
    const Word value = sum[i];
    const Word p_residue = p_in_q[i];
    const Word difference = value >= p_residue ? value - p_residue
                                               : value + limb.modulus - p_residue;
    const Word inverse = Rescale ? limb.inverse_last_q : limb.inverse_p;
    Word result = multiply(difference, inverse, limb.modulus, limb.ratio_low, limb.ratio_high);
    if (addend)
    {
        result += addend[i];
        if (result >= limb.modulus)
            result -= limb.modulus;
    }
    destination[i] = result;
}

__global__ void permute_galois(Word *output, const Word *input, std::size_t words,
                               std::uint32_t degree, int log_degree, std::uint32_t element)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words)
        return;

    const std::uint32_t coefficient = i & (degree - 1);
    const std::uint32_t reversed = __brev(degree + coefficient) >> (31 - log_degree);
    const auto index = ((static_cast<Word>(element) * reversed) >> 1) & (degree - 1);
    const auto source = __brev(static_cast<std::uint32_t>(index)) >> (32 - log_degree);
    output[i] = input[i - coefficient + source];
}

__global__ void square(Word *output, const Word *input, const Limb *limbs,
                       std::size_t degree, std::size_t q_words)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= q_words)
        return;

    const Limb limb = limbs[i / degree];
    const Word c0 = input[i];
    const Word c1 = input[q_words + i];
    output[i] = multiply(c0, c0, limb.modulus, limb.ratio_low, limb.ratio_high);
    const Word cross = 2 * multiply(c0, c1, limb.modulus, limb.ratio_low, limb.ratio_high);
    output[q_words + i] = cross >= limb.modulus ? cross - limb.modulus : cross;
    output[2 * q_words + i] = multiply(c1, c1, limb.modulus, limb.ratio_low, limb.ratio_high);
}

__global__ void multiply_ciphertexts(Word *output, const Word *left, const Word *right,
                                     const Limb *limbs, std::size_t degree, std::size_t words)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words)
        return;

    const Limb limb = limbs[i / degree];
    const Word l0 = left[i];
    const Word l1 = left[words + i];
    const Word r0 = right[i];
    const Word r1 = right[words + i];
    const Word q = limb.modulus;
    const Word lo = limb.ratio_low;
    const Word hi = limb.ratio_high;
    output[i] = multiply(l0, r0, q, lo, hi);
    const Word cross = multiply(l0, r1, q, lo, hi) + multiply(l1, r0, q, lo, hi);
    output[words + i] = cross >= q ? cross - q : cross;
    output[2 * words + i] = multiply(l1, r1, q, lo, hi);
}

__global__ void add_plain_c0(Word *sum, const Word *plain, const Limb *limbs,
                             std::size_t degree, std::size_t words)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words)
        return;
    const Word q = limbs[i / degree].modulus;
    const Word value = sum[i] + plain[i];
    sum[i] = value >= q ? value - q : value;
}

template <bool MultiplyPlain>
__global__ void add_terms(Word *sum, const Word *input, const Word *plain,
                          const Limb *limbs, std::size_t degree, std::size_t words)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words)
        return;

    const Limb limb = limbs[i / degree];
    for (std::size_t component = 0; component < 2; ++component)
    {
        const std::size_t offset = i + component * words;
        Word value = input[offset];
        if constexpr (MultiplyPlain)
            value = multiply(value, plain[i], limb.modulus, limb.ratio_low, limb.ratio_high);
        value += sum[offset];
        sum[offset] = value >= limb.modulus ? value - limb.modulus : value;
    }
}

} // namespace

void wide_pack_56(const DeviceVector<std::uint64_t> &input, DeviceVector<std::uint8_t> &output)
{
    if (output.size() != input.size() * 7 || output.device_id() != input.device_id())
        throw std::invalid_argument("wide_pack_56 buffer mismatch");
    key_56<true><<<blocks(input.size()), threads, 0, gpu_execution_stream()>>>(
        const_cast<std::uint64_t *>(input.data()), output.data(), input.size());
    gpu_check_cuda(cudaGetLastError(), "wide pack 56");
}

void wide_unpack_56(const DeviceVector<std::uint8_t> &input, DeviceVector<std::uint64_t> &output)
{
    if (input.size() != output.size() * 7 || output.device_id() != input.device_id())
        throw std::invalid_argument("wide_unpack_56 buffer mismatch");
    key_56<false><<<blocks(output.size()), threads, 0, gpu_execution_stream()>>>(
        output.data(), const_cast<std::uint8_t *>(input.data()), output.size());
    gpu_check_cuda(cudaGetLastError(), "wide unpack 56");
}

std::vector<Modulus> GpuWideBv::append_p(const std::vector<Modulus> &q, const Modulus &p)
{
    auto qp = q;
    qp.push_back(p);
    return qp;
}

GpuWideBv::GpuWideBv(int log_degree, const std::vector<Modulus> &q, const Modulus &p,
                     int device_id)
    : device_id_(device_id), log_degree_(log_degree), degree_(0), q_count_(q.size()),
      q_ntt_(log_degree, q, device_id), qp_ntt_(log_degree, append_p(q, p), device_id),
      p_ntt_(log_degree, {p}, device_id)
{
    degree_ = std::size_t{1} << log_degree;
    std::vector<Limb> tables;
    for (const auto &prime : q)
    {
        Word inverse = 0;
        if (!util::try_invert_uint_mod(p.value() % prime.value(), prime, inverse))
            throw std::invalid_argument("GpuWideBv: P is not coprime with Q");
        const auto &ratio = prime.const_ratio();
        Word inverse_last = 0;
        if (prime != q.back() &&
            !util::try_invert_uint_mod(q.back().value() % prime.value(), prime, inverse_last))
            throw std::invalid_argument("GpuWideBv: Q moduli must be pairwise coprime");
        tables.push_back({prime.value(), ratio[0], ratio[1], inverse, inverse_last});
    }
    const auto &ratio = p.const_ratio();
    tables.push_back({p.value(), ratio[0], ratio[1], 0, 0});
    limb_tables_.allocate(tables.size(), device_id_);
    limb_tables_.copy_from_host(tables.data(), tables.size());

    const std::size_t q_words = degree_ * q_count_;
    const std::size_t qp_words = q_words + degree_;
    prepared_.allocate(3 * q_words, device_id_);
    source_coefficients_.allocate(q_words, device_id_);
    hoisted_coefficients_.allocate(q_words, device_id_);
    digit_.allocate(qp_words, device_id_);
    sum_.allocate(2 * qp_words, device_id_);
    p_coefficients_.allocate(degree_, device_id_);
    p_in_q_.allocate(q_words, device_id_);
    last_q_in_prefix_.allocate(q_words - degree_, device_id_);
}

void GpuWideBv::validate_buffers(const DeviceVector<Word> &input,
                                 const DeviceVector<Word> &key,
                                 const DeviceVector<Word> &output) const
{
    const std::size_t q_words = degree_ * q_count_;
    const std::size_t qp_words = q_words + degree_;
    if (input.size() != 2 * q_words || key.size() != 2 * qp_words * q_count_ ||
        output.size() != 2 * q_words || input.device_id() != device_id_ ||
        key.device_id() != device_id_ || output.device_id() != device_id_)
        throw std::invalid_argument("GpuWideBv: buffer shape or device mismatch");
}

void GpuWideBv::apply_galois(const DeviceVector<Word> &input,
                             const DeviceVector<Word> &key, std::uint32_t element,
                             DeviceVector<Word> &output)
{
    validate_buffers(input, key, output);
    if ((element & 1) == 0 || element >= 2 * degree_)
        throw std::invalid_argument("GpuWideBv: invalid Galois element");

    gpu_check_cuda(cudaSetDevice(device_id_), "GpuWideBv cudaSetDevice");
    permute_galois<<<blocks(input.size()), threads, 0, gpu_execution_stream()>>>(
        prepared_.data(), input.data(), input.size(), degree_, log_degree_, element);
    gpu_check_cuda(cudaGetLastError(), "GpuWideBv Galois permutation");
    const std::size_t q_words = degree_ * q_count_;
    switch_and_add(prepared_.data() + q_words, prepared_.data(), nullptr,
                   key.data(), output.data());
}

void GpuWideBv::prepare_galois(const DeviceVector<Word> &input)
{
    const std::size_t words = degree_ * q_count_;
    if (input.size() != 2 * words || input.device_id() != device_id_)
        throw std::invalid_argument("GpuWideBv: invalid hoisted input");
    hoisted_input_ = input.data();
    hoisted_digits_ready_ = false;
    copy_device(hoisted_coefficients_.data(), input.data() + words, words, "hoist copy c1");
    q_ntt_.inverse(hoisted_coefficients_);
}

void GpuWideBv::apply_galois_prepared(const DeviceVector<Word> &input,
    const DeviceVector<Word> &key, std::uint32_t element, DeviceVector<Word> &output)
{
    validate_buffers(input, key, output);
    if ((element & 1) == 0 || element >= 2 * degree_ || hoisted_input_ != input.data())
        throw std::invalid_argument("GpuWideBv: invalid Galois element");
    const std::size_t words = degree_ * q_count_;
    permute_galois<<<blocks(input.size()), threads, 0, gpu_execution_stream()>>>(
        prepared_.data(), input.data(), input.size(), degree_, log_degree_, element);
    permute_coefficients<<<blocks(words), threads, 0, gpu_execution_stream()>>>(
        source_coefficients_.data(), hoisted_coefficients_.data(), limb_tables_.data(),
        degree_, words, element);
    gpu_check_cuda(cudaGetLastError(), "GpuWideBv hoisted Galois permutation");
    switch_and_add(nullptr, prepared_.data(), nullptr, key.data(), output.data(), true);
}

void GpuWideBv::prepare_galois_digits(const DeviceVector<Word> &input)
{
    prepare_galois(input);
    const std::size_t qp_words = degree_ * (q_count_ + 1);
    if (hoisted_digits_.size() == 0)
    {
        hoisted_digits_.allocate(q_count_ * qp_words, device_id_);
        transformed_key_.allocate(2 * qp_words, device_id_);
        unrotated_output_.allocate(2 * degree_ * q_count_, device_id_);
    }
    for (std::size_t digit = 0; digit < q_count_; ++digit)
    {
        lift_digit<<<blocks(qp_words), threads, 0, gpu_execution_stream()>>>(
            digit_.data(), hoisted_coefficients_.data() + digit * degree_,
            limb_tables_.data(), degree_, q_count_, std::getenv("POSEIDON_WIDE_FAST_LIFT") != nullptr);
        gpu_check_cuda(cudaGetLastError(), "hoisted digit lift");
        qp_ntt_.forward(digit_);
        copy_device(hoisted_digits_.data() + digit * qp_words, digit_.data(), qp_words,
                    "store hoisted digit");
    }
    hoisted_digits_ready_ = true;
}

void GpuWideBv::inverse_rotate_key(const DeviceVector<Word> &key, std::uint32_t element,
                                  DeviceVector<Word> &output) const
{
    const std::size_t words = 2 * degree_ * (q_count_ + 1) * q_count_;
    if ((element & 1) == 0 || element >= 2 * degree_ || key.size() != words ||
        output.size() != words || key.device_id() != device_id_ || output.device_id() != device_id_ ||
        key.data() == output.data())
        throw std::invalid_argument("GpuWideBv: invalid inverse key rotation");
    std::uint32_t inverse = element;
    for (int i = 0; i < 5; ++i) inverse *= 2u - element * inverse;
    inverse &= static_cast<std::uint32_t>(2 * degree_ - 1);
    permute_galois<<<blocks(words), threads, 0, gpu_execution_stream()>>>(
        output.data(), key.data(), words, degree_, log_degree_, inverse);
    gpu_check_cuda(cudaGetLastError(), "inverse key rotation");
}

void GpuWideBv::apply_galois_digits(const DeviceVector<Word> &input,
    const DeviceVector<Word> &key, std::uint32_t element, DeviceVector<Word> &output,
    bool key_pretransformed)
{
    validate_buffers(input, key, output);
    if ((element & 1) == 0 || element >= 2 * degree_ || !hoisted_digits_ready_ ||
        hoisted_input_ != input.data())
        throw std::invalid_argument("GpuWideBv: invalid digit-hoisted rotation");
    const std::size_t q_words = degree_ * q_count_;
    const std::size_t qp_words = q_words + degree_;
    const auto stream = gpu_execution_stream();
    std::uint32_t inverse = element;
    for (int i = 0; i < 5; ++i) inverse *= 2u - element * inverse;
    inverse &= static_cast<std::uint32_t>(2 * degree_ - 1);
    const bool fused = key_pretransformed && std::getenv("POSEIDON_WIDE_FUSED_ACCUMULATE");
    if (fused)
    {
        accumulate_cached_digits<<<blocks(qp_words), threads, 0, stream>>>(
            sum_.data(), hoisted_digits_.data(), key.data(), limb_tables_.data(), degree_, qp_words, q_count_);
        gpu_check_cuda(cudaGetLastError(), "fused hoisted key multiply");
    }
    else
    {
    gpu_check_cuda(cudaMemsetAsync(sum_.data(), 0, sum_.size() * sizeof(Word), stream),
                   "clear hoisted accumulator");
    for (std::size_t digit = 0; digit < q_count_; ++digit)
    {
        const Word *active_key = key.data() + digit * 2 * qp_words;
        if (!key_pretransformed)
        {
            permute_galois<<<blocks(2 * qp_words), threads, 0, stream>>>(
                transformed_key_.data(), active_key, 2 * qp_words, degree_, log_degree_, inverse);
            active_key = transformed_key_.data();
        }
        accumulate<<<blocks(qp_words), threads, 0, stream>>>(
            sum_.data(), hoisted_digits_.data() + digit * qp_words,
            active_key, limb_tables_.data(), degree_, qp_words);
        gpu_check_cuda(cudaGetLastError(), "hoisted rotated key multiply");
    }
    }
    for (std::size_t component = 0; component < 2; ++component)
    {
        const Word *sum = sum_.data() + component * qp_words;
        copy_device(p_coefficients_.data(), sum + q_words, degree_, "hoisted P limb");
        p_ntt_.inverse(p_coefficients_);
        lift_centered<<<blocks(q_words), threads, 0, stream>>>(
            p_in_q_.data(), p_coefficients_.data(), limb_tables_.data(), degree_, q_count_,
            std::getenv("POSEIDON_WIDE_FAST_LIFT") != nullptr);
        q_ntt_.forward(p_in_q_);
        finish<false><<<blocks(q_words), threads, 0, stream>>>(
            unrotated_output_.data() + component * q_words, sum, p_in_q_.data(),
            limb_tables_.data(), degree_, q_words, component == 0 ? input.data() : nullptr);
        gpu_check_cuda(cudaGetLastError(), "hoisted moddown");
    }
    permute_galois<<<blocks(output.size()), threads, 0, stream>>>(
        output.data(), unrotated_output_.data(), output.size(), degree_, log_degree_, element);
    gpu_check_cuda(cudaGetLastError(), "hoisted output rotation");
}

void GpuWideBv::square_relinearize(const DeviceVector<Word> &input,
                                   const DeviceVector<Word> &key, DeviceVector<Word> &output)
{
    validate_buffers(input, key, output);
    gpu_check_cuda(cudaSetDevice(device_id_), "GpuWideBv cudaSetDevice");
    const std::size_t q_words = degree_ * q_count_;
    square<<<blocks(q_words), threads, 0, gpu_execution_stream()>>>(
        prepared_.data(), input.data(), limb_tables_.data(), degree_, q_words);
    gpu_check_cuda(cudaGetLastError(), "GpuWideBv square");
    switch_and_add(prepared_.data() + 2 * q_words, prepared_.data(),
                   prepared_.data() + q_words, key.data(), output.data());
}

void GpuWideBv::multiply_relinearize(const DeviceVector<Word> &left,
                                     const DeviceVector<Word> &right,
                                     const DeviceVector<Word> &key, DeviceVector<Word> &output)
{
    validate_buffers(left, key, output);
    if (right.size() != left.size() || right.device_id() != device_id_)
        throw std::invalid_argument("GpuWideBv: multiplication buffer mismatch");
    gpu_check_cuda(cudaSetDevice(device_id_), "GpuWideBv cudaSetDevice");
    const std::size_t words = degree_ * q_count_;
    multiply_ciphertexts<<<blocks(words), threads, 0, gpu_execution_stream()>>>(
        prepared_.data(), left.data(), right.data(), limb_tables_.data(), degree_, words);
    gpu_check_cuda(cudaGetLastError(), "GpuWideBv multiply");
    switch_and_add(prepared_.data() + 2 * words, prepared_.data(),
                   prepared_.data() + words, key.data(), output.data());
}

void GpuWideBv::add_plain(const DeviceVector<Word> &plain, DeviceVector<Word> &sum) const
{
    const std::size_t words = degree_ * q_count_;
    if (sum.size() != 2 * words || plain.size() != words ||
        sum.device_id() != device_id_ || plain.device_id() != device_id_)
        throw std::invalid_argument("GpuWideBv: add plaintext buffer mismatch");
    gpu_check_cuda(cudaSetDevice(device_id_), "GpuWideBv cudaSetDevice");
    add_plain_c0<<<blocks(words), threads, 0, gpu_execution_stream()>>>(
        sum.data(), plain.data(), limb_tables_.data(), degree_, words);
    gpu_check_cuda(cudaGetLastError(), "GpuWideBv add plaintext");
}

void GpuWideBv::multiply_plain_add(const DeviceVector<Word> &input,
                                   const DeviceVector<Word> &plain,
                                   DeviceVector<Word> &sum) const
{
    const std::size_t words = degree_ * q_count_;
    if (input.size() != 2 * words || sum.size() != 2 * words || plain.size() != words ||
        input.device_id() != device_id_ || sum.device_id() != device_id_ ||
        plain.device_id() != device_id_)
        throw std::invalid_argument("GpuWideBv: multiply-accumulate buffer mismatch");
    gpu_check_cuda(cudaSetDevice(device_id_), "GpuWideBv cudaSetDevice");
    add_terms<true><<<blocks(words), threads, 0, gpu_execution_stream()>>>(
        sum.data(), input.data(), plain.data(), limb_tables_.data(), degree_, words);
    gpu_check_cuda(cudaGetLastError(), "GpuWideBv plaintext multiply-accumulate");
}

void GpuWideBv::add(const DeviceVector<Word> &input, DeviceVector<Word> &sum) const
{
    const std::size_t words = degree_ * q_count_;
    if (input.size() != 2 * words || sum.size() != 2 * words ||
        input.device_id() != device_id_ || sum.device_id() != device_id_)
        throw std::invalid_argument("GpuWideBv: addition buffer mismatch");
    gpu_check_cuda(cudaSetDevice(device_id_), "GpuWideBv cudaSetDevice");
    add_terms<false><<<blocks(words), threads, 0, gpu_execution_stream()>>>(
        sum.data(), input.data(), nullptr, limb_tables_.data(), degree_, words);
    gpu_check_cuda(cudaGetLastError(), "GpuWideBv addition");
}

void GpuWideBv::rescale_last(const DeviceVector<Word> &input, DeviceVector<Word> &output)
{
    const std::size_t q_words = degree_ * q_count_;
    const std::size_t prefix_words = q_words - degree_;
    if (q_count_ < 2 || input.size() != 2 * q_words || output.size() != 2 * prefix_words ||
        input.device_id() != device_id_ || output.device_id() != device_id_)
        throw std::invalid_argument("GpuWideBv: rescale buffer mismatch");
    gpu_check_cuda(cudaSetDevice(device_id_), "GpuWideBv cudaSetDevice");
    const auto stream = gpu_execution_stream();
    for (std::size_t component = 0; component < 2; ++component)
    {
        const Word *poly = input.data() + component * q_words;
        copy_device(p_coefficients_.data(), poly + prefix_words, degree_, "GpuWideBv copy last Q");
        q_ntt_.inverse_limb(p_coefficients_, q_count_ - 1);
        lift_centered<<<blocks(prefix_words), threads, 0, stream>>>(
            last_q_in_prefix_.data(), p_coefficients_.data(), limb_tables_.data(),
            degree_, q_count_ - 1, std::getenv("POSEIDON_WIDE_FAST_LIFT") != nullptr);
        gpu_check_cuda(cudaGetLastError(), "GpuWideBv centered Q lift");
        q_ntt_.forward_prefix(last_q_in_prefix_);
        finish<true><<<blocks(prefix_words), threads, 0, stream>>>(
            output.data() + component * prefix_words, poly, last_q_in_prefix_.data(),
            limb_tables_.data(), degree_, prefix_words, nullptr);
        gpu_check_cuda(cudaGetLastError(), "GpuWideBv rescale");
    }
}

void GpuWideBv::switch_and_add(const Word *source, const Word *add_c0, const Word *add_c1,
                               const Word *key, Word *output, bool coefficients_ready)
{
    const std::size_t q_words = degree_ * q_count_;
    const std::size_t qp_words = q_words + degree_;
    const auto stream = gpu_execution_stream();
    if (!coefficients_ready)
    {
        copy_device(source_coefficients_.data(), source, q_words, "GpuWideBv copy input");
        q_ntt_.inverse(source_coefficients_);
    }
    gpu_check_cuda(cudaMemsetAsync(sum_.data(), 0, sum_.size() * sizeof(Word), stream),
                   "GpuWideBv clear accumulator");

    for (std::size_t digit = 0; digit < q_count_; ++digit)
    {
        lift_digit<<<blocks(qp_words), threads, 0, stream>>>(
            digit_.data(), source_coefficients_.data() + digit * degree_,
            limb_tables_.data(), degree_, q_count_, std::getenv("POSEIDON_WIDE_FAST_LIFT") != nullptr);
        gpu_check_cuda(cudaGetLastError(), "GpuWideBv ModUp digit");
        qp_ntt_.forward(digit_);
        accumulate<<<blocks(qp_words), threads, 0, stream>>>(
            sum_.data(), digit_.data(), key + digit * 2 * qp_words,
            limb_tables_.data(), degree_, qp_words);
        gpu_check_cuda(cudaGetLastError(), "GpuWideBv key multiply-accumulate");
    }

    for (std::size_t component = 0; component < 2; ++component)
    {
        const Word *sum = sum_.data() + component * qp_words;
        copy_device(p_coefficients_.data(), sum + q_words, degree_, "GpuWideBv copy P limb");
        p_ntt_.inverse(p_coefficients_);
        lift_centered<<<blocks(q_words), threads, 0, stream>>>(
            p_in_q_.data(), p_coefficients_.data(), limb_tables_.data(), degree_, q_count_,
            std::getenv("POSEIDON_WIDE_FAST_LIFT") != nullptr);
        gpu_check_cuda(cudaGetLastError(), "GpuWideBv centered P lift");
        q_ntt_.forward(p_in_q_);
        finish<false><<<blocks(q_words), threads, 0, stream>>>(
            output + component * q_words, sum, p_in_q_.data(),
            limb_tables_.data(), degree_, q_words, component == 0 ? add_c0 : add_c1);
        gpu_check_cuda(cudaGetLastError(), "GpuWideBv ModDown");
    }
}

} // namespace poseidon::gpu
