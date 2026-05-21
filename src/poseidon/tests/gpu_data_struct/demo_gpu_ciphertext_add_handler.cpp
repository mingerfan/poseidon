#include "poseidon/ckks_encoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/gpu/gpu_elementwise_handler.h"
#include "poseidon/gpu/gpu_parameter.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "poseidon/plaintext.h"
#include "poseidon/poseidon_context.h"

#include <cuda_runtime_api.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int kSkip = 77;

class RmmPoolScope
{
public:
    explicit RmmPoolScope(int device_id)
        : device_id_(device_id), pool_(&upstream_, 1 << 20, std::nullopt)
    {
        auto status = cudaSetDevice(device_id_);
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("cudaSetDevice failed: ") + cudaGetErrorString(status));
        }

        previous_ = rmm::mr::get_current_device_resource();
        rmm::mr::set_current_device_resource(&pool_);
    }

    RmmPoolScope(const RmmPoolScope &) = delete;
    RmmPoolScope &operator=(const RmmPoolScope &) = delete;

    ~RmmPoolScope()
    {
        try
        {
            cudaSetDevice(device_id_);
            rmm::mr::set_current_device_resource(previous_);
        }
        catch (...)
        {}
    }

private:
    int device_id_ = 0;
    rmm::mr::cuda_memory_resource upstream_;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_;
    rmm::mr::device_memory_resource *previous_ = nullptr;
};

int check_cuda_runtime()
{
    int device_count = 0;
    auto status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess)
    {
        std::cerr << "[SKIP] CUDA runtime is unavailable: "
                  << cudaGetErrorString(status) << "\n";
        return kSkip;
    }
    if (device_count == 0)
    {
        std::cerr << "[SKIP] No CUDA device is visible\n";
        return kSkip;
    }
    return EXIT_SUCCESS;
}

poseidon::ParametersLiteral make_demo_parameters()
{
    return poseidon::ParametersLiteral(
        CKKS,
        /*log_n=*/12,
        /*log_slots=*/11,
        /*log_scale=*/10,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        poseidon::Modulus(0),
        std::vector<poseidon::Modulus>{
            poseidon::Modulus(786433),
            poseidon::Modulus(1032193),
        },
        std::vector<poseidon::Modulus>{},
        poseidon::sec_level_type::none);
}

void print_cipher_meta(const std::string &name, const poseidon::Ciphertext &cipher)
{
    std::cout << "\n[" << name << " metadata]\n";
    std::cout << "component_count = " << cipher.size() << "\n";
    std::cout << "degree          = " << cipher.poly_modulus_degree() << "\n";
    std::cout << "q_count         = " << cipher.coeff_modulus_size() << "\n";
    std::cout << "is_ntt_form     = " << cipher.is_ntt_form() << "\n";
    std::cout << "scale           = " << cipher.scale() << "\n";
}

void print_gpu_cipher_layout(
    const std::string &name,
    const poseidon::gpu::GpuCiphertextData &cipher)
{
    std::cout << "\n[" << name << " GPU layout]\n";
    std::cout << "component_count = " << cipher.meta.component_count << "\n";
    std::cout << "degree          = " << cipher.meta.degree << "\n";
    std::cout << "q_count         = " << cipher.meta.q_count << "\n";
    std::cout << "p_count         = " << cipher.meta.p_count << "\n";
    std::cout << "fields          = " << cipher.fields_.size() << "\n";
    std::cout << "polys           = " << cipher.polys_.size() << "\n";

    for (std::size_t i = 0; i < cipher.polys_.size(); ++i)
    {
        const auto &poly = cipher.polys_[i];
        std::cout << "  poly[" << i << "] shards = " << poly.shards.size() << "\n";
        for (std::size_t j = 0; j < poly.shards.size(); ++j)
        {
            const auto &shard = poly.shards[j];
            const auto &field = cipher.fields_[shard.field_index];
            std::cout << "    shard[" << j << "] field=" << shard.field_index
                      << " device=" << field.device_id
                      << " limb=[" << shard.limb_begin << ", "
                      << shard.limb_begin + shard.limb_count << ")"
                      << " coeff=[" << shard.coeff_begin << ", "
                      << shard.coeff_begin + shard.coeff_count << ")"
                      << " field_size=" << field.size() << "\n";
        }
    }
}

void print_first_words(
    const std::string &name,
    const poseidon::Ciphertext &cipher,
    std::size_t count)
{
    const std::size_t total =
        cipher.size() * cipher.poly_modulus_degree() * cipher.coeff_modulus_size();
    count = std::min(count, total);

    std::cout << "\n[" << name << " first " << count << " raw words]\n";
    for (std::size_t i = 0; i < count; ++i)
    {
        std::cout << name << ".data()[" << i << "] = " << cipher.data()[i] << "\n";
    }
}

void print_raw_comparison(
    const poseidon::Ciphertext &cpu_result,
    const poseidon::Ciphertext &gpu_result,
    std::size_t max_mismatches)
{
    const std::size_t cpu_total =
        cpu_result.size() * cpu_result.poly_modulus_degree() * cpu_result.coeff_modulus_size();
    const std::size_t gpu_total =
        gpu_result.size() * gpu_result.poly_modulus_degree() * gpu_result.coeff_modulus_size();
    const std::size_t total = std::min(cpu_total, gpu_total);

    std::size_t mismatch_count = 0;
    std::cout << "\n[CPU/GPU raw residue comparison]\n";
    std::cout << "cpu raw words = " << cpu_total << "\n";
    std::cout << "gpu raw words = " << gpu_total << "\n";

    for (std::size_t i = 0; i < total; ++i)
    {
        if (cpu_result.data()[i] != gpu_result.data()[i])
        {
            if (mismatch_count < max_mismatches)
            {
                std::cout << "mismatch[" << mismatch_count << "] index=" << i
                          << " cpu=" << cpu_result.data()[i]
                          << " gpu=" << gpu_result.data()[i] << "\n";
            }
            ++mismatch_count;
        }
    }

    std::cout << "mismatch_count = " << mismatch_count << "\n";
    std::cout << "raw_equal      = " << (mismatch_count == 0 && cpu_total == gpu_total ? "YES" : "NO")
              << "\n";
}

void print_decoded_slots(
    const std::string &name,
    const std::vector<double> &slots,
    std::size_t count)
{
    count = std::min(count, slots.size());
    std::cout << "\n[" << name << " decoded first " << count << " slots]\n";
    std::cout << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < count; ++i)
    {
        std::cout << name << "[" << i << "] = " << slots[i] << "\n";
    }
}

int run_demo()
{
    using namespace poseidon;
    using namespace poseidon::gpu;

    const int device_id = 0;
    RmmPoolScope rmm_scope(device_id);

    const auto parms = make_demo_parameters();
    PoseidonContext context(parms);

    std::cout << "===== GPU Ciphertext Add Handler Demo =====\n";
    std::cout << "scheme        = CKKS\n";
    std::cout << "degree        = " << parms.degree() << "\n";
    std::cout << "slots         = " << parms.slot() << "\n";
    std::cout << "q_count       = " << parms.q().size() << "\n";
    std::cout << "scale         = " << parms.scale() << "\n";
    std::cout << "device_id     = " << device_id << "\n";

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);

    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key, keygen.secret_key());
    Decryptor decryptor(context, keygen.secret_key());

    const std::vector<double> input0{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> input1{5.0, 6.0, 7.0, 8.0};

    std::cout << "\n[input slots]\n";
    for (std::size_t i = 0; i < input0.size(); ++i)
    {
        std::cout << "slot[" << i << "] left=" << input0[i]
                  << " right=" << input1[i]
                  << " expected_sum=" << input0[i] + input1[i] << "\n";
    }

    Plaintext plain0;
    Plaintext plain1;
    encoder.encode(input0, parms.scale(), plain0);
    encoder.encode(input1, parms.scale(), plain1);

    Ciphertext ct0;
    Ciphertext ct1;
    encryptor.encrypt(plain0, ct0);
    encryptor.encrypt(plain1, ct1);

    auto cpu_evaluator = PoseidonFactory::get_instance()->create_ckks_evaluator(context);
    Ciphertext cpu_result;
    cpu_evaluator->add(ct0, ct1, cpu_result);

    print_cipher_meta("ct0", ct0);
    print_cipher_meta("ct1", ct1);
    print_cipher_meta("cpu_result", cpu_result);
    print_first_words("ct0", ct0, 8);
    print_first_words("ct1", ct1, 8);
    print_first_words("cpu_result", cpu_result, 8);

    std::cout << "\n[GPU upload]\n";
    GpuParameterData gpu_params(context, device_id);
    auto gpu_ct0 = GpuUploader::upload_ciphertext(ct0, device_id);
    auto gpu_ct1 = GpuUploader::upload_ciphertext(ct1, device_id);

    print_gpu_cipher_layout("gpu_ct0", gpu_ct0);
    print_gpu_cipher_layout("gpu_ct1", gpu_ct1);

    const std::size_t output_components =
        std::max(gpu_ct0.meta.component_count, gpu_ct1.meta.component_count);

    auto gpu_output = GpuCiphertextData::allocate_single_device(
        gpu_ct0.meta.degree,
        gpu_ct0.meta.q_count,
        output_components,
        device_id,
        gpu_ct0.meta.p_count);

    gpu_output.meta = gpu_ct0.meta;
    gpu_output.meta.component_count = output_components;
    gpu_output.meta.parms_id = cpu_result.parms_id();
    gpu_output.meta.scale = cpu_result.scale();
    gpu_output.meta.correction_factor = cpu_result.correction_factor();
    gpu_output.meta.is_ntt_form = cpu_result.is_ntt_form();

    print_gpu_cipher_layout("gpu_output_before_add", gpu_output);

    std::cout << "\n[GPU handler add]\n";
    auto left_view = gpu_ct0.make_const_view();
    auto right_view = gpu_ct1.make_const_view();
    auto output_view = gpu_output.make_view();

    GpuElementwiseHandler handler(gpu_params);
    handler.add_ciphertext(
        output_view,
        left_view,
        right_view,
        gpu_params.get_level(gpu_ct0.meta.parms_id));
    std::cout << "GpuElementwiseHandler::add_ciphertext finished\n";

    Ciphertext gpu_result;
    GpuUploader::download_ciphertext(gpu_output, gpu_result, context);

    print_cipher_meta("gpu_result", gpu_result);
    print_first_words("gpu_result", gpu_result, 8);
    print_raw_comparison(cpu_result, gpu_result, 8);

    Plaintext cpu_plain_result;
    Plaintext gpu_plain_result;
    decryptor.decrypt(cpu_result, cpu_plain_result);
    decryptor.decrypt(gpu_result, gpu_plain_result);

    std::vector<double> cpu_slots;
    std::vector<double> gpu_slots;
    encoder.decode(cpu_plain_result, cpu_slots);
    encoder.decode(gpu_plain_result, gpu_slots);

    print_decoded_slots("cpu_result", cpu_slots, 8);
    print_decoded_slots("gpu_result", gpu_slots, 8);

    constexpr int timing_iterations = 200;
    std::cout << "\n[add operation timing]\n";
    std::cout << "iterations              = " << timing_iterations << "\n";
    std::cout << "included in timing       = add operation on existing ciphertext buffers\n";
    std::cout << "excluded from timing     = encode/encrypt/upload/download/decrypt/decode/allocation\n";

    Ciphertext cpu_timing_result;
    cpu_evaluator->add(ct0, ct1, cpu_timing_result);

    const auto cpu_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < timing_iterations; ++i)
    {
        cpu_evaluator->add(ct0, ct1, cpu_timing_result);
    }
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_total_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();

    gpu_check_cuda(cudaSetDevice(device_id), "timing cudaSetDevice");
    handler.add_ciphertext(
        output_view,
        left_view,
        right_view,
        gpu_params.get_level(gpu_ct0.meta.parms_id));
    gpu_check_cuda(cudaDeviceSynchronize(), "timing warmup sync");

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    gpu_check_cuda(cudaEventCreate(&gpu_start), "timing cudaEventCreate start");
    gpu_check_cuda(cudaEventCreate(&gpu_stop), "timing cudaEventCreate stop");

    const auto gpu_wall_begin = std::chrono::steady_clock::now();
    gpu_check_cuda(cudaEventRecord(gpu_start), "timing cudaEventRecord start");
    for (int i = 0; i < timing_iterations; ++i)
    {
        handler.add_ciphertext(
            output_view,
            left_view,
            right_view,
            gpu_params.get_level(gpu_ct0.meta.parms_id));
    }
    gpu_check_cuda(cudaEventRecord(gpu_stop), "timing cudaEventRecord stop");
    gpu_check_cuda(cudaEventSynchronize(gpu_stop), "timing cudaEventSynchronize stop");
    const auto gpu_wall_end = std::chrono::steady_clock::now();

    float gpu_event_total_ms = 0.0F;
    gpu_check_cuda(
        cudaEventElapsedTime(&gpu_event_total_ms, gpu_start, gpu_stop),
        "timing cudaEventElapsedTime");

    gpu_check_cuda(cudaEventDestroy(gpu_start), "timing cudaEventDestroy start");
    gpu_check_cuda(cudaEventDestroy(gpu_stop), "timing cudaEventDestroy stop");

    const double gpu_wall_total_ms =
        std::chrono::duration<double, std::milli>(gpu_wall_end - gpu_wall_begin).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "cpu evaluator.add total ms          = " << cpu_total_ms << "\n";
    std::cout << "cpu evaluator.add avg ms            = "
              << cpu_total_ms / timing_iterations << "\n";
    std::cout << "gpu handler.add wall total ms       = " << gpu_wall_total_ms << "\n";
    std::cout << "gpu handler.add wall avg ms         = "
              << gpu_wall_total_ms / timing_iterations << "\n";
    std::cout << "gpu handler.add cuda-event total ms = " << gpu_event_total_ms << "\n";
    std::cout << "gpu handler.add cuda-event avg ms   = "
              << gpu_event_total_ms / timing_iterations << "\n";

    std::cout << "\n===== DEMO FINISHED =====\n";
    return EXIT_SUCCESS;
}

}  // namespace

int main()
{
    const int cuda_status = check_cuda_runtime();
    if (cuda_status != EXIT_SUCCESS)
    {
        return cuda_status;
    }

    try
    {
        return run_demo();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[EXCEPTION] " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
