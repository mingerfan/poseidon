#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// CUDA-header-free; constructor does not initialize CUDA.
namespace poseidon {
class PoseidonContext;
class RelinKeys;
class GaloisKeys;
namespace gpu { class GpuCiphertextData; }
}

namespace bert::attention {
// PackedInput2Partial is NOT the baseline full-slot AV contract. Its live
// features are complete AV; its gaps are defined cross-head/row partial sums.
enum class AvAlgorithm { Baseline, PackedInput2Partial };
const char *av_algorithm_name(AvAlgorithm algorithm) noexcept;
const char *av_output_semantics(AvAlgorithm algorithm) noexcept;
struct AvSpec {
    std::string id;
    std::size_t tokens = 128;
    std::size_t tile = 256;
    std::size_t head_dim = 64;
    std::size_t baby = 8;
    // Physical level L means q_count = L+1. Current engine accepts this
    // fixed physical interval only, with actual prime-derived scale checks.
    std::size_t score_input_level = 10;
    double score_input_scale_log2 = 55.360150528219684;
    std::size_t value_input_level = 8;
    double value_input_scale_log2 = 55.35255783895382;
    std::size_t common_level = 6;
    double common_scale_log2 = 55.3727119134628;
    std::size_t output_level = 4;
    double output_scale_log2 = 55.4442212582195;
    AvAlgorithm algorithm = AvAlgorithm::Baseline; // append-only; baseline remains default
};
struct AvStats {
    std::string id;
    std::uint64_t mask_bytes = 0; // shared bundle size, NOT exclusive per-id bytes
    double diagonal_mask_scale_log2 = 0;
    double score_mask_scale_log2 = 0;
    bool reused_mask_bundle = false;
    double setup_seconds = 0;
    AvAlgorithm algorithm = AvAlgorithm::Baseline;
    std::string schedule;
    std::string output_semantics;
    std::uint64_t mask_plaintext_count = 0; // actual encoded/uploaded bundle members
    bool partial_output = false;
    bool client_value_gap_zero_required = false;
};
struct AvEngineStats {
    std::uint64_t parameter_bytes = 0;
    std::uint64_t galois_key_bytes = 0;
    std::uint64_t relin_key_bytes = 0;
    std::uint64_t mask_bytes = 0; // unique physical bundles, counted once
    std::uint64_t mask_bundle_count = 0;
    std::uint64_t allocated_bytes = 0; // persistent only; excludes scratch/pool reserve
    std::size_t key_storage_q_count = 0;
    std::size_t key_storage_p_count = 0; // ALL context P primes, never dynamic P
    bool key_storage_is_full_q = true;
    double setup_seconds = 0;
    // Cumulative real evaluator calls, including completed calls in failed runs.
    std::uint64_t run_calls = 0;
    std::uint64_t cc_multiply_calls = 0;
    std::uint64_t relinearize_calls = 0;
    std::uint64_t cp_multiply_calls = 0;
    std::uint64_t rescale_calls = 0;
    std::uint64_t rescale2_calls = 0;
    std::uint64_t rescale4_calls = 0;
    std::uint64_t physical_rescale_primes = 0;
    std::uint64_t rotation_calls = 0; // includes binary fallback parts
    std::uint64_t hoisted_rotation_batches = 0;
    std::uint64_t add_calls = 0;
    std::uint64_t baseline_run_calls = 0;
    std::uint64_t packed_input2_partial_run_calls = 0;
    std::uint64_t relin_key_upload_calls = 0;
    std::uint64_t galois_key_upload_calls = 0;
    bool preparation_locked = false; // first run (even if failed) forbids later prepare
    std::uint64_t mask_plaintext_count = 0; // unique bundles, NOT per-ID sum
};
class PreparedAvEngine {
public:
    PreparedAvEngine(const poseidon::PoseidonContext &context,
                     std::shared_ptr<const poseidon::RelinKeys> relin_keys,
                     std::shared_ptr<const poseidon::GaloisKeys> galois_keys,
                     int device = 0);
    ~PreparedAvEngine();
    PreparedAvEngine(const PreparedAvEngine &) = delete;
    PreparedAvEngine &operator=(const PreparedAvEngine &) = delete;

    // GPU setup; prepare ALL IDs/algorithms before the first run. Later prepare
    // is rejected. Identical physical+algorithm specs share one mask bundle;
    // baseline uses383 masks, PackedInput2Partial uses319. Mixed preparation
    // extends/reuploads owning keys (Q never shrinks), re-preparing key views.
    void prepare(const AvSpec &spec);
    // Immutable synchronized size2 Q-only NTT inputs, kept alive by caller.
    // Full cyclic scores [t,head,d128], gapped values [t,head,feature128].
    // Baseline retains128 CC/relin/Rescale2. PackedInput2Partial pairs inputs
    // into64 genuine CCs and REQUIRES client V gap0. Its output gaps are NOT
    // zero: gap[t,h,D+j]=B[t,h,j]+A[next(t,h),j], where A/B split d by parity
    // of d-j, next h0=(t,h1), next h1=(t+1 modT,h0). Do not feed this output
    // to the old unmasked reciprocal/merge consumer. No secret key/host refresh.
    // Enqueues one pair on a private stream and synchronizes once at return.
    // Non-concurrent. Cold allocator growth may additionally synchronize.
    poseidon::gpu::GpuCiphertextData run(
        const std::string &id, const poseidon::gpu::GpuCiphertextData &scores,
        const poseidon::gpu::GpuCiphertextData &value);
    // LEFT convention: rotate(k)[i] = input[(i+k)%slots].
    static std::vector<int> required_rotation_steps();
    static std::vector<int> required_binary_rotation_steps();
    // Superset of the baseline52 direct steps, plus -64,+64,+192 (55 total).
    static std::vector<int> required_packed_input_rotation_steps();
    static std::vector<int> required_packed_input_binary_rotation_steps();
    const AvStats &av_stats(const std::string &id) const;
    AvEngineStats engine_stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace bert::attention
