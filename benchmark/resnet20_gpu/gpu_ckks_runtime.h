#pragma once

#include "gpu_config.h"

#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/gpu/gpu_plaintext.h"

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace poseidon::benchmark::resnet20_gpu::core
{

// Owns a single-device Poseidon CKKS environment. Encoding, key generation,
// encryption and final decryption are setup/I/O work; every homomorphic
// operation exposed by this class is dispatched to GpuEvaluator.
class GpuCkksRuntime
{
public:
    using DeviceCiphertext = gpu::GpuCiphertextData;
    using DevicePlaintext = gpu::GpuPlaintextData;

    explicit GpuCkksRuntime(const GpuConfig &config, int device_id = 0);
    ~GpuCkksRuntime();

    GpuCkksRuntime(const GpuCkksRuntime &) = delete;
    GpuCkksRuntime &operator=(const GpuCkksRuntime &) = delete;
    GpuCkksRuntime(GpuCkksRuntime &&) = delete;
    GpuCkksRuntime &operator=(GpuCkksRuntime &&) = delete;

    DeviceCiphertext encrypt(const std::vector<double> &slots) const;
    DeviceCiphertext encrypt_constant(double value) const;
    std::vector<std::complex<double>> decrypt(const DeviceCiphertext &ciphertext) const;

    // Multiplies by a SIMD plaintext on GPU and consumes exactly one physical
    // q prime, restoring the ciphertext scale to approximately its input value.
    DeviceCiphertext multiply_plain_rescale(
        const DeviceCiphertext &source,
        const std::vector<double> &plain_slots) const;
    // Reuses a device-resident encoded plaintext for repeated structured
    // masks at the same modulus level (for example convolution selectors).
    DeviceCiphertext multiply_plain_rescale_cached(
        const DeviceCiphertext &source,
        const std::vector<double> &plain_slots,
        const std::string &cache_key) const;
    DeviceCiphertext multiply_plain(
        const DeviceCiphertext &source,
        const std::vector<double> &plain_slots,
        double plain_scale) const;
    // Fused GPU kernel for a lazy-rescale dot product. The destination must
    // already contain a product at source.scale * plain_scale.
    void multiply_plain_accumulate(
        const DeviceCiphertext &source,
        const std::vector<double> &plain_slots,
        double plain_scale,
        DeviceCiphertext &destination) const;
    // Split plaintext preparation from evaluation so a fixed encoded
    // diagonal can be reused by several ciphertexts without repeating CPU
    // CKKS encoding and Host-to-Device transfer.
    DevicePlaintext encode_and_upload_plain(
        const DeviceCiphertext &source,
        const std::vector<double> &plain_slots,
        double plain_scale) const;
    std::vector<DevicePlaintext> encode_and_upload_plain_batch(
        const DeviceCiphertext &source,
        const std::vector<std::vector<double>> &plain_slots,
        double plain_scale) const;
    DeviceCiphertext multiply_plain_preencoded(
        const DeviceCiphertext &source,
        const DevicePlaintext &plaintext) const;
    DeviceCiphertext multiply_plain_scalar_rescale(
        const DeviceCiphertext &source,
        double value) const;
    // Exact value of the last active q prime. Useful for lazy plaintext
    // products that are accumulated before a single rescale.
    double last_modulus_value(const DeviceCiphertext &source) const;

    // Evaluation keys are generated and uploaded only when nonlinear or
    // rotation operations are needed, keeping linear-only startup lightweight.
    void initialize_evaluation_keys(const std::vector<int> &rotation_steps = {});
    bool evaluation_keys_ready() const noexcept;

    DeviceCiphertext add(
        const DeviceCiphertext &left,
        const DeviceCiphertext &right) const;
    DeviceCiphertext add_aligned(
        const DeviceCiphertext &left,
        const DeviceCiphertext &right) const;
    DeviceCiphertext sub(
        const DeviceCiphertext &left,
        const DeviceCiphertext &right) const;
    DeviceCiphertext sub_aligned(
        const DeviceCiphertext &left,
        const DeviceCiphertext &right) const;
    DeviceCiphertext add_plain(
        const DeviceCiphertext &source,
        const std::vector<double> &plain_slots) const;
    DeviceCiphertext add_plain_scalar(
        const DeviceCiphertext &source,
        double value) const;

    // Benchmark mode: retain every encoded plaintext and encrypted input on
    // the device. A second identical execution then performs no CPU encoding,
    // encryption, or host-to-device model/input transfer.
    void enable_full_device_cache(bool enable = true);
    void synchronize() const;
    DeviceCiphertext drop_to_q_count(
        const DeviceCiphertext &source,
        std::size_t target_q_count) const;
    DeviceCiphertext rotate(const DeviceCiphertext &source, int step) const;
    DeviceCiphertext rotate_composed(
        const DeviceCiphertext &source,
        long long step) const;
    // Uses one shared HYBRID decomposition for several direct rotations when
    // direct keys are active. Falls back to the normal composed path in
    // validation modes that only generated power-of-two keys.
    std::vector<DeviceCiphertext> rotate_many_composed(
        const DeviceCiphertext &source,
        const std::vector<long long> &steps) const;
    // Generate and upload one direct Galois key for every supplied logical
    // rotation. Subsequent rotate_composed calls use one key switch each.
    void initialize_direct_rotation_keys(
        const std::vector<int> &rotation_steps);
    void initialize_inference_evaluation_keys();
    // Backward-compatible model-specific spelling.
    void initialize_all_evaluation_keys();
    void initialize_bootstrap();
    bool bootstrap_ready() const noexcept;
    DeviceCiphertext square_relinearize_rescale(
        const DeviceCiphertext &source) const;
    DeviceCiphertext multiply_relinearize_rescale(
        const DeviceCiphertext &left,
        const DeviceCiphertext &right) const;
    DeviceCiphertext rescale(
        const DeviceCiphertext &source,
        std::uint32_t physical_prime_count = 1) const;
    DeviceCiphertext bootstrap_modraise(
        const DeviceCiphertext &source) const;
    DeviceCiphertext bootstrap(const DeviceCiphertext &source) const;

    int device_id() const noexcept;
    std::size_t slot_count() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace poseidon::benchmark::resnet20_gpu::core
