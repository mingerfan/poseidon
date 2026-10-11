#include "gpu_encode_experiment.h"
#include "poseidon/gpu/gpu_tensor_core_gemm.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"
#include <cmath>
#include <stdexcept>
#include <random>
#ifdef POSEIDON_ENCODE_TILELANG
#include "tilelang_ntt.h"
#endif

using namespace poseidon::gpu;
namespace {
void check_fft(cufftResult status, const char *name)
{
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(name) + ": cuFFT error " + std::to_string(status));
}

__global__ void permute_slots(const double *input, double2 *fft,
                              const std::uint32_t *indices, std::size_t n, int *error)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n / 2) return;
    const std::size_t batch = blockIdx.y;
    const double x = input[batch * (n / 2) + i];
    if (!isfinite(x)) { atomicExch(error, 1); }
    fft[batch * n + indices[i]] = make_double2(x, 0);
    fft[batch * n + indices[n / 2 + i]] = make_double2(x, 0);
}

__device__ void write_rns(double coefficient, GpuWord *rns, const GpuWord *primes,
                         std::size_t n, std::size_t limbs, std::size_t batch,
                         std::size_t i, int *error)
{
    const double magnitude = fabs(coefficient);
    if (!isfinite(coefficient) || magnitude >= 0x1p64) {
        atomicExch(error, 2);
        return;
    }
    const auto integer = static_cast<unsigned long long>(magnitude);
    for (std::size_t limb = 0; limb < limbs; ++limb) {
        const GpuWord q = primes[limb];
        const GpuWord residue = integer % q;
        rns[(batch * limbs + limb) * n + i] =
            coefficient < 0 && residue ? q - residue : residue;
    }
}

__global__ void expand_rns(const double2 *fft, const double2 *twists,
                           GpuWord *rns, const GpuWord *primes, std::size_t n,
                           std::size_t limbs, double fix, int *error)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::size_t batch = blockIdx.y;
    const double2 x = fft[batch * n + i], w = twists[i];
    write_rns(round((x.x * w.x - x.y * w.y) * fix), rns, primes, n, limbs, batch, i, error);
}

__global__ void expand_coefficients_rns(const double *input, GpuWord *rns,
                                       const GpuWord *primes, std::size_t n,
                                       std::size_t limbs, double scale, int *error)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::size_t batch = blockIdx.y;
    write_rns(round(input[batch * n + i] * scale), rns, primes, n, limbs, batch, i, error);
}
} // namespace

GpuEncodeExperiment::GpuEncodeExperiment(
    std::size_t degree, std::size_t limbs, std::size_t batch,
    const std::vector<std::uint32_t> &indices, const std::vector<double2> &twists)
    : output(degree * limbs * batch, 0), degree_(degree), limbs_(limbs), batch_(batch),
      indices_(degree, 0), twists_(degree, 0), fft_(degree * batch, 0),
      rns_(degree * limbs * batch, 0), error_(1, 0)
{
    if (degree != 65536 || !limbs || !batch || batch > 65535 ||
        indices.size() != degree || twists.size() != degree)
        throw std::invalid_argument("encode experiment requires N=65536 and complete tables");
    indices_.copy_from_host(indices.data(), degree);
    twists_.copy_from_host(twists.data(), degree);
    check_fft(cufftPlan1d(&single_plan_, degree, CUFFT_Z2Z, 1), "single FFT plan");
    try {
        check_fft(cufftPlan1d(&batch_plan_, degree, CUFFT_Z2Z, batch), "batch FFT plan");
        check_fft(cufftSetStream(single_plan_, gpu_execution_stream()), "single FFT stream");
        check_fft(cufftSetStream(batch_plan_, gpu_execution_stream()), "batch FFT stream");
    } catch (...) {
        if (batch_plan_) cufftDestroy(batch_plan_);
        cufftDestroy(single_plan_);
        throw;
    }
}

GpuEncodeExperiment::~GpuEncodeExperiment()
{
    if (batch_plan_) cufftDestroy(batch_plan_);
    if (single_plan_) cufftDestroy(single_plan_);
}

void GpuEncodeExperiment::prepare_one(const double *input, double scale,
    const GpuParameterShard &parameters, std::size_t first, std::size_t count, cufftHandle plan)
{
    auto *work = fft_.data() + first * degree_;
    permute_slots<<<dim3((degree_ / 2 + 255) / 256, count), 256, 0, gpu_execution_stream()>>>(
        input + first * (degree_ / 2), work, indices_.data(), degree_, error_.data());
    gpu_check_cuda(cudaGetLastError(), "permute slots");
    check_fft(cufftExecZ2Z(plan, work, work, CUFFT_FORWARD), "encode FFT");
    expand_rns<<<dim3((degree_ + 255) / 256, count), 256, 0, gpu_execution_stream()>>>(
        work, twists_.data(), rns_.data() + first * limbs_ * degree_,
        parameters.rns_primes.data(), degree_, limbs_, scale / degree_, error_.data());
    gpu_check_cuda(cudaGetLastError(), "expand RNS");
}

void GpuEncodeExperiment::prepare(const double *input, double scale,
    const GpuParameterShard &parameters, bool batched)
{
    if (!input || !std::isfinite(scale) || scale <= 0 || parameters.device_id != 0 ||
        parameters.limb_begin != 0 || parameters.limb_count < limbs_)
        throw std::invalid_argument("invalid GPU encode arguments");
    error_.fill_zero();
    if (batched) prepare_one(input, scale, parameters, 0, batch_, batch_plan_);
    else for (std::size_t b = 0; b < batch_; ++b)
        prepare_one(input, scale, parameters, b, 1, single_plan_);
}

void GpuEncodeExperiment::prepare_coefficients(const double *input, double scale,
    const GpuParameterShard &parameters, bool batched)
{
    if (!input || !std::isfinite(scale) || scale <= 0 || parameters.device_id != 0 ||
        parameters.limb_begin != 0 || parameters.limb_count < limbs_)
        throw std::invalid_argument("invalid precomputed-coefficient arguments");
    error_.fill_zero();
    const auto count = batched ? batch_ : 1;
    for (std::size_t b = 0; b < batch_; b += count) {
        expand_coefficients_rns<<<dim3((degree_ + 255) / 256, count), 256, 0, gpu_execution_stream()>>>(
            input + b * degree_, rns_.data() + b * limbs_ * degree_, parameters.rns_primes.data(),
            degree_, limbs_, scale, error_.data());
        gpu_check_cuda(cudaGetLastError(), "expand precomputed coefficients RNS");
    }
}

void GpuEncodeExperiment::initialize_tilelang(const GpuParameterShard &parameters)
{
#ifdef POSEIDON_ENCODE_TILELANG
    if (parameters.limb_count < limbs_ || parameters.ntt_fused_matrix_fusion_stages != 4 ||
        parameters.ntt_fused_matrices.size() < parameters.limb_count * 69888 ||
        parameters.rns_modulus_constants.size() < limbs_)
        throw std::invalid_argument("TileLang experiment requires full fusion=4 matrices and Barrett ratios");
    std::vector<GpuWord> primes(limbs_), weights(limbs_ * 7);
    parameters.rns_primes.copy_to_host(primes.data(), limbs_);
    for (std::size_t l = 0; l < limbs_; ++l) {
        if (primes[l] >= (1U << 31)) throw std::invalid_argument("TileLang NTT requires q<2^31");
        weights[l * 7] = 1;
        for (std::size_t k = 1; k < 7; ++k)
            weights[l * 7 + k] = static_cast<std::uint64_t>(weights[l * 7 + k - 1]) * 256 % primes[l];
    }
    tilelang_weights_ = DeviceVector<GpuWord>(weights.size(), 0);
    tilelang_weights_.copy_from_host(weights.data(), weights.size());
#else
    throw std::runtime_error("TileLang backend was not built");
#endif
}

void GpuEncodeExperiment::validate_tilelang_ntt(const GpuParameterShard &parameters)
{
#ifdef POSEIDON_ENCODE_TILELANG
    // Cover 0, 1, q-1, q-2 and unrelated random residues for every limb and
    // plaintext, independently of the FFT-generated encoding coefficients.
    std::vector<GpuWord> primes(limbs_), input(rns_.size()), expected(output.size()), actual(output.size());
    parameters.rns_primes.copy_to_host(primes.data(), limbs_);
    std::mt19937_64 rng(20261012);
    for (std::size_t b = 0; b < batch_; ++b)
        for (std::size_t l = 0; l < limbs_; ++l)
            for (std::size_t i = 0; i < degree_; ++i)
                input[(b * limbs_ + l) * degree_ + i] = i % 7 == 0 ? 0 : i % 7 == 1 ? 1 :
                    i % 7 == 2 ? primes[l] - 1 : i % 7 == 3 ? primes[l] - 2 : rng() % primes[l];
    rns_.copy_from_host(input.data(), input.size());
    ntt(parameters, Backend::cuda, true);
    output.copy_to_host(expected.data(), expected.size());
    for (auto backend : {Backend::tilelang_cuda, Backend::tilelang_tensor}) {
        ntt(parameters, backend, true);
        output.copy_to_host(actual.data(), actual.size());
        if (actual != expected) throw std::runtime_error("TileLang boundary/random NTT mismatch");
    }
#else
    throw std::runtime_error("TileLang backend was not built");
#endif
}

void GpuEncodeExperiment::ntt(const GpuParameterShard &parameters, Backend backend, bool batched)
{
    const bool tensor = backend == Backend::tensor || backend == Backend::tilelang_tensor;
    if (tensor && !supports_tensor_core_integer_gemm())
        throw std::runtime_error("tensor encoder requires integer Tensor Core support (SM 7.5+)");
    const auto stride = degree_ * limbs_;
    const auto count = batched ? batch_ : 1;
    for (std::size_t b = 0; b < batch_; b += count) {
        GpuPolyShardView destination{0, output.data() + b * stride, 0, limbs_, 0, degree_};
        GpuConstPolyShardView source{0, rns_.data() + b * stride, 0, limbs_, 0, degree_};
        if (backend == Backend::tilelang_cuda || backend == Backend::tilelang_tensor) {
#ifdef POSEIDON_ENCODE_TILELANG
            if (tilelang_weights_.size() != limbs_ * 7)
                throw std::runtime_error("TileLang weights were not initialized");
            gpu_check_cuda(launch_encode_tilelang_ntt(source.ptr, destination.ptr,
                parameters.ntt_tables.data(), parameters.ntt_fused_matrices.data(),
                parameters.rns_primes.data(), parameters.rns_modulus_constants.data(),
                tilelang_weights_.data(), limbs_, parameters.limb_count, count,
                tensor, gpu_execution_stream()), "TileLang NTT");
#else
            throw std::runtime_error("TileLang backend was not built");
#endif
        } else if (tensor)
            kernel::launch_forward_ntt_components_shard_tensor(
                destination, source, parameters, degree_, count, stride);
        else
            kernel::launch_forward_ntt_poly_shard_batch_fourstep_65536(
                destination, source, parameters, degree_, count);
    }
}

void GpuEncodeExperiment::check_result() const
{
    int error = 0;
    error_.copy_to_host(&error, 1);
    if (error) throw std::runtime_error("GPU Encode: non-finite input or coefficient magnitude >= 2^64");
}
