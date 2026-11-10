#include "poseidon/advance/homomorphic_linear_transform.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/decryptor.h"
#include "poseidon/encryptor.h"
#include "poseidon/evaluator/software/evaluator_ckks_software.h"
#include "poseidon/gpu/gpu_evaluator.h"
#include "poseidon/gpu/gpu_uploader.h"
#include "poseidon/keygenerator.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace poseidon;
using namespace poseidon::gpu;
namespace
{
using Complex = std::complex<double>;

void require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

struct TestStream
{
    cudaStream_t value{};
    TestStream()
    {
        require(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking) == cudaSuccess,
                "cannot create nonblocking test stream");
    }
    ~TestStream()
    {
        cudaStreamSynchronize(value);
        cudaStreamDestroy(value);
    }
};

void switches(bool optimized, bool qp_c0 = false)
{
    const char *value = optimized ? "1" : "0";
    for (const char *name : {"POSEIDON_GPU_KEY_VIEW_CACHE", "POSEIDON_GPU_LAZY_BASECONV",
                             "POSEIDON_GPU_MODDOWN_FOURSTEP", "POSEIDON_GPU_QP_FOURSTEP_ANY_P"})
        setenv(name, value, 1);
    setenv("POSEIDON_DOUBLE_HOIST_QP_C0", qp_c0 ? "1" : "0", 1);
    setenv("POSEIDON_DOUBLE_HOIST_DIRECT_GIANT_ACCUMULATE", "1", 1);
    setenv("POSEIDON_NTT_ALGO", "fourstep", 1);
}

ParametersLiteral parameters(unsigned log_n, std::size_t q, std::size_t p)
{
    ParametersLiteral result{CKKS, log_n, log_n - 1, 44, 64, 0, 0, {}, {}, sec_level_type::none};
    std::vector<std::uint32_t> q_bits(q), p_bits(p);
    for (std::size_t i = 0; i < q; ++i) q_bits[i] = 29 + i % 4;
    for (std::size_t i = 0; i < p; ++i) p_bits[i] = 29 + (i + 1) % 4;
    result.set_log_modulus(q_bits, p_bits);
    return result;
}

std::vector<Complex> message(std::size_t slots)
{
    std::vector<Complex> result(slots);
    for (std::size_t i = 0; i < slots; ++i)
        result[i] = {0.2 * std::sin(0.017 * i), 0.1 * std::cos(0.013 * i)};
    return result;
}

std::vector<Complex> decode(const GpuCiphertextData &gpu, const PoseidonContext &context,
                            CKKSEncoder &encoder, Decryptor &decryptor)
{
    Ciphertext host;
    GpuUploader::download_ciphertext(gpu, host, context);
    Plaintext plain;
    decryptor.decrypt(host, plain);
    std::vector<Complex> result;
    encoder.decode(plain, result);
    return result;
}

void close(const std::vector<Complex> &actual, const std::vector<Complex> &expected,
           const std::string &label, double tolerance = 1e-4)
{
    require(actual.size() == expected.size(), label + ": slot count changed");
    double worst = 0;
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const double error = std::abs(actual[i] - expected[i]);
        require(std::isfinite(error) && error <= tolerance,
                label + ": error=" + std::to_string(error) + " slot=" + std::to_string(i));
        worst = std::max(worst, error);
    }
    std::cout << "ORACLE " << label << " max_error=" << worst << '\n';
}

void exact(const GpuCiphertextData &left, const GpuCiphertextData &right,
           const PoseidonContext &context, const std::string &label)
{
    Ciphertext a, b;
    GpuUploader::download_ciphertext(left, a, context);
    GpuUploader::download_ciphertext(right, b, context);
    require(a.parms_id() == b.parms_id() && a.scale() == b.scale() && a.size() == b.size() &&
                a.is_ntt_form() == b.is_ntt_form(), label + ": metadata changed");
    const auto words = a.size() * a.poly_modulus_degree() * a.coeff_modulus_size();
    require(std::equal(a.data(), a.data() + words, b.data()), label + ": RNS residues changed");
}

void uploader_stream_regression(const PoseidonContext &context, const Ciphertext &source)
{
    auto gpu = GpuUploader::upload_ciphertext(source, 0);
    struct Gate
    {
        std::mutex mutex;
        std::condition_variable cv;
        bool entered = false;
        bool readback_entered = false;
        bool released = false;
    } gate;
    Ciphertext result;
    std::thread release([&] {
        std::unique_lock<std::mutex> lock(gate.mutex);
        const bool ready = gate.cv.wait_for(lock, std::chrono::seconds(2), [&] {
            return gate.entered && gate.readback_entered;
        });
        if (ready)
            gate.cv.wait_for(lock, std::chrono::milliseconds(50), [] { return false; });
        gate.released = true;
        gate.cv.notify_all();
    });
    try
    {
        require(cudaLaunchHostFunc(gpu_execution_stream(), [](void *opaque) {
            auto &gate = *static_cast<Gate *>(opaque);
            std::unique_lock<std::mutex> lock(gate.mutex);
            gate.entered = true;
            gate.cv.notify_all();
            gate.cv.wait(lock, [&] { return gate.released; });
        }, &gate) == cudaSuccess, "cannot gate producer stream");
        // A default-stream copy would capture the old ciphertext, not zeros.
        for (auto &field : gpu.fields_)
            require(cudaMemsetAsync(field.data(), 0, field.size() * sizeof(GpuWord),
                                    gpu_execution_stream()) == cudaSuccess,
                    "cannot queue producer write");
        {
            std::lock_guard<std::mutex> lock(gate.mutex);
            gate.readback_entered = true;
            gate.cv.notify_all();
        }
        GpuUploader::download_ciphertext(gpu, result, context);
    }
    catch (...)
    {
        release.join();
        cudaStreamSynchronize(gpu_execution_stream());
        throw;
    }
    release.join();
    require(cudaStreamSynchronize(gpu_execution_stream()) == cudaSuccess, "producer did not complete");
    const auto words = result.size() * result.poly_modulus_degree() * result.coeff_modulus_size();
    require(std::all_of(result.data(), result.data() + words, [](auto value) { return value == 0; }),
            "uploader readback did not wait for the nonblocking producer");
    std::cout << "PASS stream-ordered uploader readback\n";
}

void key_switch_case(unsigned log_n, std::size_t q, std::size_t p)
{
    PoseidonContext context(parameters(log_n, q, p));
    KeyGenerator generator(context);
    require(!generator.secret_key().data().is_zero(), "test requires a nonzero secret");
    PublicKey public_key;
    RelinKeys relin;
    GaloisKeys rotations;
    generator.create_public_key(public_key);
    generator.create_relin_keys(relin);
    generator.create_galois_keys(std::vector<int>{1}, rotations);
    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, generator.secret_key());
    EvaluatorCkksSoftware cpu(context);
    const auto values = message(encoder.slot_count());
    Plaintext plain;
    encoder.encode(values, std::ldexp(1., 44), plain);
    Ciphertext source;
    encryptor.encrypt(plain, source);
    if (log_n == 12 && q == 4 && p == 2)
        uploader_stream_regression(context, source);
    GpuParameterData parameters(context, 0, {q, 3});
    GpuEvaluator evaluator(parameters);
    for (const auto active_q : {q, std::size_t{3}})
    {
        Ciphertext input = source;
        cpu.drop_modulus(input, input, context.crt_context()->parms_id_map().at(active_q - 1));
        auto gpu = GpuUploader::upload_ciphertext(input, 0);
        auto key = GpuUploader::upload_galois_keys(rotations, 0, active_q);
        auto relin_key = GpuUploader::upload_relin_keys(relin, 0, active_q);
        GpuCiphertextData reference_rotation, optimized_rotation, product, reference_relin, optimized_relin;
        switches(false);
        evaluator.rotate(gpu, 1, key, reference_rotation);
        evaluator.multiply(gpu, gpu, product);
        evaluator.relinearize(product, relin_key, reference_relin);
        switches(true);
        evaluator.rotate(gpu, 1, key, optimized_rotation);
        evaluator.relinearize(product, relin_key, optimized_relin);
        const auto label = "N=" + std::to_string(std::size_t{1} << log_n) + " Q=" +
                           std::to_string(active_q) + " P=" + std::to_string(p);
        exact(reference_rotation, optimized_rotation, context, label + " rotation A/B");
        exact(reference_relin, optimized_relin, context, label + " relin A/B");
        auto rotated = values;
        std::rotate(rotated.begin(), rotated.begin() + 1, rotated.end());
        close(decode(optimized_rotation, context, encoder, decryptor), rotated, label + " rotation");
        auto squared = values;
        for (auto &v : squared) v *= v;
        close(decode(optimized_relin, context, encoder, decryptor), squared, label + " relin");

        // The inverse-pre-rotated/double-hoist path uses the same arithmetic
        // with an independently staged decomposition, including short Q < P.
        auto pre_rotated = GpuUploader::upload_double_hoist_galois_keys(rotations, 0);
        GpuCiphertextData staged_reference, staged_optimized;
        switches(false);
        evaluator.rotate(gpu, 1, pre_rotated, staged_reference);
        switches(true);
        evaluator.rotate(gpu, 1, pre_rotated, staged_optimized);
        exact(staged_reference, staged_optimized, context, label + " pre-rotated A/B");
        close(decode(staged_optimized, context, encoder, decryptor), rotated, label + " pre-rotated");
        std::cout << "PASS key-switch " << label << " bit_exact=true\n";
    }
}

void qp_c0_case(unsigned log_n, std::size_t q, std::size_t p)
{
    PoseidonContext context(parameters(log_n, q, p));
    KeyGenerator generator(context);
    PublicKey public_key;
    generator.create_public_key(public_key);
    CKKSEncoder encoder(context);
    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, generator.secret_key());
    const auto values = message(encoder.slot_count());
    Plaintext plain;
    encoder.encode(values, std::ldexp(1., 44), plain);
    Ciphertext source;
    encryptor.encrypt(plain, source);
    auto gpu = GpuUploader::upload_ciphertext(source, 0);
    GpuParameterData parameters(context, 0, {q, 3});
    GpuEvaluator evaluator(parameters);
    // Reuse large-to-small retained buffers across Q levels and giant counts.
    GpuDoubleHoistWorkspace a, b;
    for (const auto active_q : {q, std::size_t{3}})
    {
      if (active_q != q)
          evaluator.drop_modulus_inplace(gpu, context.crt_context()->parms_id_map().at(active_q - 1));
      for (const int shape : {0, 4, 3, 1, 2})
      {
        std::vector<int> offsets = shape == 0 ? std::vector<int>{0, 1, 2, 4, 5, 6, 8, 9, 10} :
                                   shape == 1 ? std::vector<int>{4, 5, 6, 8, 9, 10} :
                                   shape == 2 ? std::vector<int>{0, 1, 2} :
                                   shape == 3 ? std::vector<int>{0, 1, 2, 4, 5, 6} :
                                                std::vector<int>{0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14};
        std::map<int, std::vector<Complex>> diagonals;
        std::vector<Complex> oracle(values.size());
        for (int offset : offsets)
        {
            auto &diagonal = diagonals[offset];
            diagonal.resize(values.size());
            for (std::size_t k = 0; k < values.size(); ++k)
            {
                diagonal[k] = {0.03 + 0.002 * (k % 4 + offset), 0.001 * ((k + offset) % 4)};
                oracle[k] += diagonal[k] * values[(k + offset) % values.size()];
            }
        }
        MatrixPlain matrix;
        gen_linear_transform_bsgs(matrix, matrix.rot_index, encoder, diagonals,
                                  static_cast<std::uint32_t>(active_q - 1), std::ldexp(1., 30),
                                  1, log_n - 1, 4);
        LinearMatrixGroup group;
        group.data().push_back(matrix);
        group.rot_index() = matrix.rot_index;
        group.set_step(1);
        GaloisKeys rotations;
        generator.create_galois_keys(matrix.rot_index, rotations);
        auto key = GpuUploader::upload_double_hoist_galois_keys(rotations, 0);
        for (const bool compressed : {false, true})
        {
            auto uploaded = GpuUploader::upload_linear_matrix_group_qp(group, context, 0, 1, compressed);
            const auto &matrix_gpu = uploaded.data().front();
            GpuCiphertextData reference, optimized;
            switches(true, false);
            evaluator.multiply_by_diag_matrix_bsgs_double_hoist(gpu, matrix_gpu, key, 0, a, reference);
            switches(true, true);
            evaluator.multiply_by_diag_matrix_bsgs_double_hoist(gpu, matrix_gpu, key, 0, b, optimized);
            const auto label = "QP-c0 N=" + std::to_string(std::size_t{1} << log_n) +
                               " Q=" + std::to_string(active_q) + " P=" + std::to_string(p) +
                               " shape=" + std::to_string(shape) +
                               " compressed=" + std::to_string(compressed);
            close(decode(reference, context, encoder, decryptor), oracle, label + " original");
            close(decode(optimized, context, encoder, decryptor), oracle, label + " optimized");
            close(decode(optimized, context, encoder, decryptor),
                  decode(reference, context, encoder, decryptor), label + " A/B", 1e-6);
            if (shape == 0 || shape == 3 || shape == 4)
                require(b.last_counts.inner_moddown_component_count +
                            matrix_gpu.plan.giant_steps.size() + 1 ==
                            a.last_counts.inner_moddown_component_count,
                        "QP-c0 selected-component path was not used");
            else
                exact(reference, optimized, context, label + " unchanged fallback");
            // Retained scratch is reused repeatedly, with all toggles live.
            evaluator.multiply_by_diag_matrix_bsgs_double_hoist(gpu, matrix_gpu, key, 0, b, optimized);
            close(decode(optimized, context, encoder, decryptor), oracle, label + " repeat");
            std::cout << "PASS " << label << " groups=" << matrix_gpu.plan.giant_steps.size()
                      << " inner_moddown_off=" << a.last_counts.inner_moddown_count
                      << " inner_moddown_on=" << b.last_counts.inner_moddown_count
                      << " components_off=" << a.last_counts.inner_moddown_component_count
                      << " components_on=" << b.last_counts.inner_moddown_component_count << '\n';
        }
      }
    }
}
} // namespace

int main()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try
    {
        TestStream stream;
        ScopedGpuExecutionStream execution_stream(stream.value);
        for (const auto log_n : {12U, 16U})
            for (const auto [q, p] : std::vector<std::pair<std::size_t, std::size_t>>{
                     {4, 2}, {10, 9}, {13, 10}, {28, 25}})
                key_switch_case(log_n, q, p);
        qp_c0_case(12, 7, 2);
        qp_c0_case(16, 13, 10);
        qp_c0_case(16, 28, 25);
        std::cout << "GPU Boot optimization regressions passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
