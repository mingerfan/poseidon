#include "gpu_encode_experiment.h"
#include "poseidon/gpu/gpu_tensor_core_gemm.h"
#include "poseidon/gpu/kernels/gpu_ntt_kernels.h"
#include <cmath>
#include <stdexcept>

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

__global__ void expand_rns(const double2 *fft, const double2 *twists,
                           GpuWord *rns, const GpuWord *primes, std::size_t n,
                           std::size_t limbs, double fix, int *error)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::size_t batch = blockIdx.y;
    const double2 x = fft[batch * n + i], w = twists[i];
    const double coefficient = round((x.x * w.x - x.y * w.y) * fix);
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

void GpuEncodeExperiment::ntt(const GpuParameterShard &parameters, bool tensor, bool batched)
{
    if (tensor && !supports_tensor_core_integer_gemm())
        throw std::runtime_error("tensor encoder requires integer Tensor Core support (SM 7.5+)");
    const auto stride = degree_ * limbs_;
    const auto count = batched ? batch_ : 1;
    for (std::size_t b = 0; b < batch_; b += count) {
        GpuPolyShardView destination{0, output.data() + b * stride, 0, limbs_, 0, degree_};
        GpuConstPolyShardView source{0, rns_.data() + b * stride, 0, limbs_, 0, degree_};
        if (tensor)
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
