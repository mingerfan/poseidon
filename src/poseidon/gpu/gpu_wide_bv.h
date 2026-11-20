#pragma once

#include "poseidon/gpu/gpu_wide_ntt.h"

namespace poseidon::gpu
{

// Lossless 56-bit storage for resident key material. The caller verifies bounds.
void wide_pack_56(const DeviceVector<std::uint64_t> &input, DeviceVector<std::uint8_t> &output);
void wide_unpack_56(const DeviceVector<std::uint8_t> &input, DeviceVector<std::uint64_t> &output);

namespace detail
{
struct WideBvLimb
{
    std::uint64_t modulus;
    std::uint64_t ratio_low;
    std::uint64_t ratio_high;
    std::uint64_t inverse_p;
    std::uint64_t inverse_last_q;
};
} // namespace detail

// One active Q prefix and one special prime P, in Poseidon NTT order.
// Calls on the same plan share scratch and must use a single CUDA stream.
class GpuWideBv
{
public:
    GpuWideBv(int log_degree, const std::vector<Modulus> &q, const Modulus &p,
              int device_id);

    // Ciphertext layout is [component][Q limb][coefficient]; each key is
    // packed [Q digit][component][Q prefix | P][coefficient]. An element of
    // 2N-1 performs conjugation. Squaring returns residues at scale^2.
    void apply_galois(const DeviceVector<std::uint64_t> &input,
                      const DeviceVector<std::uint64_t> &key,
                      std::uint32_t element, DeviceVector<std::uint64_t> &output);
    void prepare_galois(const DeviceVector<std::uint64_t> &input);
    void apply_galois_prepared(const DeviceVector<std::uint64_t> &input,
                               const DeviceVector<std::uint64_t> &key, std::uint32_t element,
                               DeviceVector<std::uint64_t> &output);
    // Experimental: digits are shared before automorphism. Results require
    // decoded-error validation because decomposition representatives differ.
    void prepare_galois_digits(const DeviceVector<std::uint64_t> &input);
    void apply_galois_digits(const DeviceVector<std::uint64_t> &input,
                             const DeviceVector<std::uint64_t> &key, std::uint32_t element,
                             DeviceVector<std::uint64_t> &output, bool key_pretransformed = false);
    void inverse_rotate_key(const DeviceVector<std::uint64_t> &key, std::uint32_t element,
                            DeviceVector<std::uint64_t> &output) const;
    void square_relinearize(const DeviceVector<std::uint64_t> &input,
                            const DeviceVector<std::uint64_t> &key,
                            DeviceVector<std::uint64_t> &output);

    void multiply_relinearize(const DeviceVector<std::uint64_t> &left,
                               const DeviceVector<std::uint64_t> &right,
                               const DeviceVector<std::uint64_t> &key,
                               DeviceVector<std::uint64_t> &output);
    void add_plain(const DeviceVector<std::uint64_t> &plain,
                   DeviceVector<std::uint64_t> &sum) const;

    // Operate on two-component NTT ciphertexts. The plaintext has one Q
    // prefix and may be shared between components. Rescale drops one Q limb.
    void multiply_plain_add(const DeviceVector<std::uint64_t> &input,
                            const DeviceVector<std::uint64_t> &plain,
                            DeviceVector<std::uint64_t> &sum) const;
    void add(const DeviceVector<std::uint64_t> &input,
             DeviceVector<std::uint64_t> &sum) const;
    void rescale_last(const DeviceVector<std::uint64_t> &input,
                      DeviceVector<std::uint64_t> &output);

private:
    void validate_buffers(const DeviceVector<std::uint64_t> &input,
                          const DeviceVector<std::uint64_t> &key,
                          const DeviceVector<std::uint64_t> &output) const;
    void switch_and_add(const std::uint64_t *source, const std::uint64_t *add_c0,
                        const std::uint64_t *add_c1, const std::uint64_t *key,
                        std::uint64_t *output, bool coefficients_ready = false);

    static std::vector<Modulus> append_p(const std::vector<Modulus> &q, const Modulus &p);

    int device_id_;
    int log_degree_;
    std::size_t degree_;
    std::size_t q_count_;
    GpuWideNtt q_ntt_;
    GpuWideNtt qp_ntt_;
    GpuWideNtt p_ntt_;
    DeviceVector<detail::WideBvLimb> limb_tables_;
    DeviceVector<std::uint64_t> prepared_;
    DeviceVector<std::uint64_t> source_coefficients_;
    DeviceVector<std::uint64_t> hoisted_coefficients_;
    DeviceVector<std::uint64_t> hoisted_digits_, transformed_key_, unrotated_output_;
    const std::uint64_t *hoisted_input_ = nullptr;
    bool hoisted_digits_ready_ = false;
    DeviceVector<std::uint64_t> digit_;
    DeviceVector<std::uint64_t> sum_;
    DeviceVector<std::uint64_t> p_coefficients_;
    DeviceVector<std::uint64_t> p_in_q_;
    DeviceVector<std::uint64_t> last_q_in_prefix_;
};

} // namespace poseidon::gpu
