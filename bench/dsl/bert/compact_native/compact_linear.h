#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Deliberately does not include CUDA/RMM headers: plan integration can include
// this header without exposing the helper's implementation dependencies.
namespace poseidon {
class PoseidonContext;
class GaloisKeys;
namespace gpu { class GpuCiphertextData; }
}

namespace bert::compact {

struct MatrixSpec {
    std::string id;
    std::size_t input_width = 0;
    std::size_t output_width = 0;
    std::size_t tile = 256;
    std::size_t baby = 16;
    std::size_t tokens = 128;
    // Physical Poseidon level L means q_count=L+1, NOT q_count=L.
    std::size_t input_level = 0;
    double input_scale_log2 = 30;
    std::size_t output_level = 0;
    double output_scale_log2 = 30;
    std::uint32_t rescale_count = 1;
    // Explicit opt-in: valid CKKS small-ring512 encoding + sparse full-ring
    // lift. Original float32 weights are unchanged; FFT rounding is not
    // universally claimed byte-identical to native fullEncode. Strict full
    // encoder remains the default and fallback for unsupported fast cases.
    bool periodic_encoding = false;
};

struct MatrixStats {
    std::string id;
    std::uint64_t weight_bytes = 0;
    std::uint64_t compact_bytes = 0; // exact Q residues, plus full bias
    std::uint64_t full_equivalent_bytes = 0;
    std::uint64_t cp_terms = 0;
    std::uint64_t baby_rotations = 0; // per complete output sequence
    std::uint64_t giant_rotations = 0;
    std::uint64_t compression_fallbacks = 0;
    std::uint64_t max_period = 0;
    double plaintext_scale_log2 = 0;
    double setup_seconds = 0;
    std::string encoding_mode = "strict_full";
    std::uint64_t strict_encoded_terms = 0;
    std::uint64_t periodic_encoded_terms = 0;
    std::uint64_t periodic_encoding_fallbacks = 0;
    std::string periodic_fallback_reason;
    bool periodic_full_ntt_mapping_verified = false;
    bool native_encoder_bitwise_guaranteed = true;
    // Fast mode audits edge/central diagonals of each matrix against actual
    // fullEncode and reports differences honestly, without changing weights.
    std::uint64_t native_encoder_audit_terms = 0;
    std::uint64_t native_encoder_mismatch_terms = 0;
    std::uint64_t native_encoder_mismatch_words = 0;
};

struct EngineStats {
    std::uint64_t parameter_bytes = 0;
    std::uint64_t key_bytes = 0;
    // The owning key allocation is cropped to the highest prepared input Q,
    // and keeps all P. True only when that highest Q is the full context Q.
    // Later q-prefix views do not further crop the owning allocation.
    bool key_storage_is_full_q = true;
    std::uint64_t run_calls = 0;
    std::uint64_t actual_rotation_calls = 0; // includes binary fallback
};

class CompactLinearEngine {
public:
    // Constructor performs no CUDA work. prepare()/run() are explicitly GPU
    // operations; CPU-only tests use compact_layout.h and setup_check.cpp.
    CompactLinearEngine(const poseidon::PoseidonContext &context,
                        std::shared_ptr<const poseidon::GaloisKeys> cpu_keys,
                        int device = 0);
    ~CompactLinearEngine();
    CompactLinearEngine(const CompactLinearEngine &) = delete;
    CompactLinearEngine &operator=(const CompactLinearEngine &) = delete;

    // Untimed: validates, encodes each genuine dense diagonal independently,
    // verifies full-N exact RNS reconstruction, compresses and uploads it.
    // Copies all original float32 weights and bias (no sparse substitute).
    // Prefer preparing matrices in descending input_level order to upload
    // the owning cropped rotation keys only once. Never prepare mid-sequence.
    void prepare(const MatrixSpec &spec, const float *rowmajor_out_in,
                 const float *bias);
    void prepare(const MatrixSpec &spec, const std::vector<float> &weights,
                 const std::vector<float> &bias);

    // Positive LEFT rotations, out[i]=in[(i+step)%slots]. Required direct
    // steps for the fixed 256/16 layout; callers may alternatively generate
    // required_binary_rotation_steps() and use the exact binary fallback.
    static std::vector<int> required_rotation_steps(std::size_t tile = 256,
                                                    std::size_t baby = 16);
    static std::vector<int> required_binary_rotation_steps(std::size_t tile = 256,
                                                           std::size_t baby = 16);

    // Inputs must be synchronized by caller and live through this sequence.
    // Output0 ALWAYS rebuilds baby rotations, even for identical addresses.
    // Thereafter require output1,...,last, same matrix and input objects.
    // A different/new sequence must start at output0 (which can interrupt a
    // previous sequence). No pointer-identity cache survives output0.
    // All real GPU operations execute on a private stream. One explicit
    // synchronization before return makes the result ready for BaseApi use
    // (allocator/scratch growth can synchronize additionally on cold runs).
    // One engine is not concurrently callable from multiple threads.
    poseidon::gpu::GpuCiphertextData run(
        const std::string &matrix_id, std::size_t output_tile,
        const std::vector<const poseidon::gpu::GpuCiphertextData *> &inputs);
    void reset_sequence();

    const MatrixStats &matrix_stats(const std::string &matrix_id) const;
    EngineStats engine_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bert::compact
