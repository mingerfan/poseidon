#include "poseidon/ckks_encoder.h"
#include "poseidon/ciphertext.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/factory/poseidon_factory.h"
#include "poseidon/gpu/gpu_evaluator.h"
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
    poseidon::ParametersLiteral parms(
        CKKS,
        /*log_n=*/14,
        /*log_slots=*/13,
        /*log_scale=*/25,
        /*hamming_weight=*/0,
        /*q0_level=*/0,
        poseidon::Modulus(0),
        std::vector<poseidon::Modulus>{},
        std::vector<poseidon::Modulus>{},
        poseidon::sec_level_type::none);

    parms.set_log_modulus(
        std::vector<std::uint32_t>(16, 30),
        std::vector<std::uint32_t>{});
    return parms;
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
                      << " offset=" << shard.field_offset
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

    std::cout << "===== GPU Ciphertext Elementwise Evaluator Demo =====\n";
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
                  << " expected_sum=" << input0[i] + input1[i]
                  << " expected_sub=" << input0[i] - input1[i]
                  << " expected_negate_left=" << -input0[i] << "\n";
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
    auto gpu_plain1 = GpuUploader::upload_plaintext(plain1, device_id);

    print_gpu_cipher_layout("gpu_ct0", gpu_ct0);
    print_gpu_cipher_layout("gpu_ct1", gpu_ct1);

    GpuEvaluator gpu_evaluator(gpu_params);
    GpuCiphertextData gpu_output;

    std::cout << "\n[GPU evaluator add]\n";
    gpu_evaluator.add(gpu_ct0, gpu_ct1, gpu_output);
    std::cout << "GpuEvaluator::add finished\n";
    print_gpu_cipher_layout("gpu_output_after_add", gpu_output);

    Ciphertext gpu_result;
    GpuUploader::download_ciphertext(gpu_output, gpu_result, context);

    print_cipher_meta("gpu_result", gpu_result);
    print_first_words("gpu_result", gpu_result, 8);
    print_raw_comparison(cpu_result, gpu_result, 8);

    std::cout << "\n[CPU/GPU evaluator sub]\n";
    Ciphertext cpu_sub_result;
    cpu_evaluator->sub(ct0, ct1, cpu_sub_result);

    GpuCiphertextData gpu_sub_output;
    gpu_evaluator.sub(gpu_ct0, gpu_ct1, gpu_sub_output);

    Ciphertext gpu_sub_result;
    GpuUploader::download_ciphertext(gpu_sub_output, gpu_sub_result, context);

    print_cipher_meta("cpu_sub_result", cpu_sub_result);
    print_cipher_meta("gpu_sub_result", gpu_sub_result);
    print_first_words("cpu_sub_result", cpu_sub_result, 8);
    print_first_words("gpu_sub_result", gpu_sub_result, 8);
    print_raw_comparison(cpu_sub_result, gpu_sub_result, 8);

    std::cout << "\n[CPU/GPU evaluator negate]\n";
    Ciphertext cpu_negate_result = ct0;
    for (std::size_t i = 0; i < cpu_negate_result.size(); ++i)
    {
        cpu_negate_result[i].negate();
    }

    GpuCiphertextData gpu_negate_output;
    gpu_evaluator.negate(gpu_ct0, gpu_negate_output);

    Ciphertext gpu_negate_result;
    GpuUploader::download_ciphertext(gpu_negate_output, gpu_negate_result, context);

    print_cipher_meta("cpu_negate_result", cpu_negate_result);
    print_cipher_meta("gpu_negate_result", gpu_negate_result);
    print_first_words("cpu_negate_result", cpu_negate_result, 8);
    print_first_words("gpu_negate_result", gpu_negate_result, 8);
    print_raw_comparison(cpu_negate_result, gpu_negate_result, 8);

    std::cout << "\n[CPU/GPU evaluator multiply_plain]\n";

    Ciphertext cpu_multiply_plain_result;
    cpu_evaluator->multiply_plain(ct0, plain1, cpu_multiply_plain_result);

    GpuCiphertextData gpu_multiply_plain_output;
    gpu_evaluator.multiply_plain(gpu_ct0, gpu_plain1, gpu_multiply_plain_output);

    Ciphertext gpu_multiply_plain_result;
    GpuUploader::download_ciphertext(
        gpu_multiply_plain_output,
        gpu_multiply_plain_result,
        context);

    print_cipher_meta("cpu_multiply_plain_result", cpu_multiply_plain_result);
    print_cipher_meta("gpu_multiply_plain_result", gpu_multiply_plain_result);
    print_first_words("cpu_multiply_plain_result", cpu_multiply_plain_result, 8);
    print_first_words("gpu_multiply_plain_result", gpu_multiply_plain_result, 8);
    print_raw_comparison(cpu_multiply_plain_result, gpu_multiply_plain_result, 8);

    std::cout << "\n[CPU/GPU evaluator rescale]\n";

    Ciphertext cpu_rescale_result;
    cpu_evaluator->rescale(cpu_multiply_plain_result, cpu_rescale_result);

    GpuCiphertextData gpu_rescale_output;
    gpu_evaluator.rescale(gpu_multiply_plain_output, gpu_rescale_output);
    gpu_check_cuda(cudaDeviceSynchronize(), "GpuEvaluator::rescale sync");

    Ciphertext gpu_rescale_result;
    GpuUploader::download_ciphertext(
        gpu_rescale_output,
        gpu_rescale_result,
        context);

    print_cipher_meta("cpu_rescale_result", cpu_rescale_result);
    print_cipher_meta("gpu_rescale_result", gpu_rescale_result);
    print_first_words("cpu_rescale_result", cpu_rescale_result, 8);
    print_first_words("gpu_rescale_result", gpu_rescale_result, 8);
    print_raw_comparison(cpu_rescale_result, gpu_rescale_result, 8);

    std::cout << "\n[CPU/GPU evaluator add_plain]\n";

    Ciphertext cpu_add_plain_result;
    cpu_evaluator->add_plain(ct0, plain1, cpu_add_plain_result);

    GpuCiphertextData gpu_add_plain_output;
    gpu_evaluator.add_plain(gpu_ct0, gpu_plain1, gpu_add_plain_output);

    Ciphertext gpu_add_plain_result;
    GpuUploader::download_ciphertext(gpu_add_plain_output, gpu_add_plain_result, context);

    print_cipher_meta("cpu_add_plain_result", cpu_add_plain_result);
    print_cipher_meta("gpu_add_plain_result", gpu_add_plain_result);
    print_first_words("cpu_add_plain_result", cpu_add_plain_result, 8);
    print_first_words("gpu_add_plain_result", gpu_add_plain_result, 8);
    print_raw_comparison(cpu_add_plain_result, gpu_add_plain_result, 8);

    std::cout << "\n[CPU/GPU evaluator sub_plain]\n";

    Ciphertext cpu_sub_plain_result = ct0;
    cpu_sub_plain_result[0].sub(plain1.poly(), cpu_sub_plain_result[0]);

    GpuCiphertextData gpu_sub_plain_output;
    gpu_evaluator.sub_plain(gpu_ct0, gpu_plain1, gpu_sub_plain_output);

    Ciphertext gpu_sub_plain_result;
    GpuUploader::download_ciphertext(gpu_sub_plain_output, gpu_sub_plain_result, context);

    print_cipher_meta("cpu_sub_plain_result", cpu_sub_plain_result);
    print_cipher_meta("gpu_sub_plain_result", gpu_sub_plain_result);
    print_first_words("cpu_sub_plain_result", cpu_sub_plain_result, 8);
    print_first_words("gpu_sub_plain_result", gpu_sub_plain_result, 8);
    print_raw_comparison(cpu_sub_plain_result, gpu_sub_plain_result, 8);

    std::cout << "\n[CPU/GPU evaluator ntt_inv]\n";

    Ciphertext cpu_ntt_inv_result;
    cpu_evaluator->ntt_inv(ct0, cpu_ntt_inv_result);

    GpuCiphertextData gpu_ntt_inv_output;
    gpu_evaluator.ntt_inv(gpu_ct0, gpu_ntt_inv_output);
    gpu_check_cuda(cudaDeviceSynchronize(), "GpuEvaluator::ntt_inv sync");

    Ciphertext gpu_ntt_inv_result;
    GpuUploader::download_ciphertext(gpu_ntt_inv_output, gpu_ntt_inv_result, context);

    print_cipher_meta("cpu_ntt_inv_result", cpu_ntt_inv_result);
    print_cipher_meta("gpu_ntt_inv_result", gpu_ntt_inv_result);
    print_first_words("cpu_ntt_inv_result", cpu_ntt_inv_result, 8);
    print_first_words("gpu_ntt_inv_result", gpu_ntt_inv_result, 8);
    print_raw_comparison(cpu_ntt_inv_result, gpu_ntt_inv_result, 8);

    std::cout << "\n[CPU/GPU evaluator ntt_fwd roundtrip]\n";

    Ciphertext cpu_ntt_fwd_result;
    cpu_evaluator->ntt_fwd(cpu_ntt_inv_result, cpu_ntt_fwd_result);

    GpuCiphertextData gpu_ntt_fwd_output;
    gpu_evaluator.ntt_fwd(gpu_ntt_inv_output, gpu_ntt_fwd_output);
    gpu_check_cuda(cudaDeviceSynchronize(), "GpuEvaluator::ntt_fwd sync");

    Ciphertext gpu_ntt_fwd_result;
    GpuUploader::download_ciphertext(gpu_ntt_fwd_output, gpu_ntt_fwd_result, context);

    print_cipher_meta("cpu_ntt_fwd_result", cpu_ntt_fwd_result);
    print_cipher_meta("gpu_ntt_fwd_result", gpu_ntt_fwd_result);
    print_first_words("cpu_ntt_fwd_result", cpu_ntt_fwd_result, 8);
    print_first_words("gpu_ntt_fwd_result", gpu_ntt_fwd_result, 8);
    print_raw_comparison(cpu_ntt_fwd_result, gpu_ntt_fwd_result, 8);

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
    Ciphertext cpu_timing_result;
    GpuCiphertextData gpu_timing_output;

    auto benchmark_operation =
        [&](const std::string &name, auto cpu_once, auto gpu_once)
    {
        std::cout << "\n[" << name << " operation timing]\n";
        std::cout << "iterations              = " << timing_iterations << "\n";
        std::cout << "included in timing       = top-level operation call\n";
        std::cout << "excluded from timing     = encode/encrypt/upload/download/decrypt/decode\n";

        cpu_once();

        const auto cpu_begin = std::chrono::steady_clock::now();
        for (int i = 0; i < timing_iterations; ++i)
        {
            cpu_once();
        }
        const auto cpu_end = std::chrono::steady_clock::now();
        const double cpu_total_ms =
            std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();

        gpu_check_cuda(cudaSetDevice(device_id), "timing cudaSetDevice");
        gpu_once();
        gpu_check_cuda(cudaDeviceSynchronize(), "timing warmup sync");

        cudaEvent_t gpu_start = nullptr;
        cudaEvent_t gpu_stop = nullptr;
        gpu_check_cuda(cudaEventCreate(&gpu_start), "timing cudaEventCreate start");
        gpu_check_cuda(cudaEventCreate(&gpu_stop), "timing cudaEventCreate stop");

        const auto gpu_wall_begin = std::chrono::steady_clock::now();
        gpu_check_cuda(cudaEventRecord(gpu_start), "timing cudaEventRecord start");
        for (int i = 0; i < timing_iterations; ++i)
        {
            gpu_once();
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
        const double cpu_avg_ms = cpu_total_ms / timing_iterations;
        const double gpu_wall_avg_ms = gpu_wall_total_ms / timing_iterations;
        const double gpu_event_avg_ms = gpu_event_total_ms / timing_iterations;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "cpu total ms        = " << cpu_total_ms << "\n";
        std::cout << "cpu avg ms          = " << cpu_avg_ms << "\n";
        std::cout << "gpu wall total ms   = " << gpu_wall_total_ms << "\n";
        std::cout << "gpu wall avg ms     = " << gpu_wall_avg_ms << "\n";
        std::cout << "gpu event total ms  = " << gpu_event_total_ms << "\n";
        std::cout << "gpu event avg ms    = " << gpu_event_avg_ms << "\n";
        std::cout << "speedup wall        = " << cpu_avg_ms / gpu_wall_avg_ms << "x\n";
        std::cout << "speedup cuda-event  = " << cpu_avg_ms / gpu_event_avg_ms << "x\n";
    };

    benchmark_operation(
        "add",
        [&]() { cpu_evaluator->add(ct0, ct1, cpu_timing_result); },
        [&]() { gpu_evaluator.add(gpu_ct0, gpu_ct1, gpu_timing_output); });

    benchmark_operation(
        "sub",
        [&]() { cpu_evaluator->sub(ct0, ct1, cpu_timing_result); },
        [&]() { gpu_evaluator.sub(gpu_ct0, gpu_ct1, gpu_timing_output); });

    benchmark_operation(
        "negate",
        [&]()
        {
            cpu_timing_result = ct0;
            for (std::size_t i = 0; i < cpu_timing_result.size(); ++i)
            {
                cpu_timing_result[i].negate();
            }
        },
        [&]() { gpu_evaluator.negate(gpu_ct0, gpu_timing_output); });

    benchmark_operation(
        "add_plain",
        [&]() { cpu_evaluator->add_plain(ct0, plain1, cpu_timing_result); },
        [&]() { gpu_evaluator.add_plain(gpu_ct0, gpu_plain1, gpu_timing_output); });

    benchmark_operation(
        "sub_plain",
        [&]()
        {
            cpu_timing_result = ct0;
            cpu_timing_result[0].sub(plain1.poly(), cpu_timing_result[0]);
        },
        [&]() { gpu_evaluator.sub_plain(gpu_ct0, gpu_plain1, gpu_timing_output); });

    benchmark_operation(
        "multiply_plain",
        [&]() { cpu_evaluator->multiply_plain(ct0, plain1, cpu_timing_result); },
        [&]() { gpu_evaluator.multiply_plain(gpu_ct0, gpu_plain1, gpu_timing_output); });

    benchmark_operation(
        "rescale",
        [&]() { cpu_evaluator->rescale(cpu_multiply_plain_result, cpu_timing_result); },
        [&]() { gpu_evaluator.rescale(gpu_multiply_plain_output, gpu_timing_output); });

    benchmark_operation(
        "ntt_inv",
        [&]() { cpu_evaluator->ntt_inv(ct0, cpu_timing_result); },
        [&]() { gpu_evaluator.ntt_inv(gpu_ct0, gpu_timing_output); });

    benchmark_operation(
        "ntt_fwd",
        [&]() { cpu_evaluator->ntt_fwd(cpu_ntt_inv_result, cpu_timing_result); },
        [&]() { gpu_evaluator.ntt_fwd(gpu_ntt_inv_output, gpu_timing_output); });

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
