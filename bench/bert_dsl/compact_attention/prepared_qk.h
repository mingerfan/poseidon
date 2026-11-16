#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// No CUDA headers: including the callback interface never initializes CUDA.
namespace poseidon {
class PoseidonContext;
class RelinKeys;
class GaloisKeys;
namespace gpu { class GpuCiphertextData; }
}

namespace bert::attention {

struct QkSpec {
    std::string id;
    std::size_t tokens = 128;
    std::size_t tile = 256;
    std::size_t head_dim = 64;
    std::size_t baby = 8;
    // Physical Poseidon level L = q_count-1, not Cerium q_count.
    std::size_t input_level = 8;
    // Defaults are the real q52-p11 canonical scales, NOT nominal 2^56.
    double input_scale_log2 = 55.35255783895382;
    // Must equal 2*input_scale_log2-log2(q[L])-log2(q[L-1]).
    double intermediate_scale_log2 = 55.3727119134628;
    std::size_t output_level = 4;
    double output_scale_log2 = 55.4442212582195;
    // Opt-in only: share the D64 reduction for two baby products packed into
    // the ideal-message zero gaps. CKKS encoding/noise gaps are NOT bitwise 0.
    // Does not reduce the 128 true CC/relin/Rescale2 or mask CP products.
    bool pair_products = false;
    // Separate opt-in: pack Q/K BEFORE CC, using all four D64 blocks. Computes
    // every logical key/head score with64 true CC terms (rather than128).
    // Same ideal-message zero-gap domain; mutually exclusive with pair_products.
    bool pack_inputs = false;
};

struct QkStats {
    std::string id;
    std::uint64_t mask_bytes = 0;
    double mask_scale_log2 = 0;
    double setup_seconds = 0;
    bool pair_products = false;
    bool pack_inputs = false;
    std::string schedule = "baseline_per_product_headreduce";
    std::string encoding = "full_ckks_head_start_mask";
};

struct EngineStats {
    std::uint64_t parameter_bytes = 0;
    std::uint64_t galois_key_bytes = 0;
    std::uint64_t relin_key_bytes = 0;
    std::uint64_t mask_bytes = 0;
    std::uint64_t allocated_bytes = 0; // persistent keys + parameters + masks; excludes transient scratch
    std::size_t key_storage_q_count = 0;
    std::size_t key_storage_p_count = 0; // always ALL context P primes
    bool key_storage_is_full_q = true;
    double setup_seconds = 0;
    // Cumulative actual evaluator calls, including completed work in a failed run.
    std::uint64_t run_calls = 0;
    std::uint64_t baseline_run_calls = 0;
    std::uint64_t paired_run_calls = 0;
    std::uint64_t packed_input_run_calls = 0;
    std::uint64_t key_uploads = 0;
    std::size_t uploaded_rotation_steps = 0; // actual unique Galois elements (binary parts included)
    std::uint64_t cc_multiply_calls = 0;
    std::uint64_t relinearize_calls = 0;
    std::uint64_t cp_multiply_calls = 0;
    std::uint64_t rescale2_calls = 0;
    std::uint64_t physical_rescale_primes = 0;
    std::uint64_t rotation_calls = 0; // includes binary fallback parts
    std::uint64_t hoisted_rotation_batches = 0;
    std::uint64_t add_calls = 0;
};

class PreparedQkEngine {
public:
    // Constructor does no CUDA work. prepare() and run() are GPU operations.
    PreparedQkEngine(const poseidon::PoseidonContext &context,
                     std::shared_ptr<const poseidon::RelinKeys> relin_keys,
                     std::shared_ptr<const poseidon::GaloisKeys> galois_keys,
                     int device = 0);
    ~PreparedQkEngine();
    PreparedQkEngine(const PreparedQkEngine &) = delete;
    PreparedQkEngine &operator=(const PreparedQkEngine &) = delete;

    // Untimed real full CKKS mask encoding and owning key/parameter upload.
    // Prepare ALL specs before the first run; further prepare() is rejected.
    // Highest input level first avoids growing key owners. Mixed baseline and
    // paired/packed-input specs upload their union at the highest Q count.
    void prepare(const QkSpec &spec);

    // Both arguments are synchronized, immutable size2 Q-only NTT ciphertexts.
    // One complete pair is enqueued on one private stream and synchronized once
    // before return (cold allocator growth may additionally synchronize).
    // Not concurrently callable. No Runtime.compute calls or secret keys.
    poseidon::gpu::GpuCiphertextData run(
        const std::string &id, const poseidon::gpu::GpuCiphertextData &query,
        const poseidon::gpu::GpuCiphertextData &key);

    // Positive LEFT rotation: output[i] = input[(i+step)%slots]. Fixed T128
    // production layout. Direct keys enable shared Q giants / K babies hoists.
    static std::vector<int> required_rotation_steps();
    static std::vector<int> required_binary_rotation_steps();
    // Superset of the unchanged baseline direct keys, plus -64 and +63.
    static std::vector<int> required_paired_rotation_steps();
    static std::vector<int> required_paired_binary_rotation_steps();
    // Unchanged baseline superset plus -64, +63, +192 (53 direct keys).
    static std::vector<int> required_packed_input_rotation_steps();
    static std::vector<int> required_packed_input_binary_rotation_steps();
    const QkStats &qk_stats(const std::string &id) const;
    EngineStats engine_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bert::attention
