#pragma once
// CPU-only opt-in SUBRING ENCODER. Never used by the strict default path.
// This produces an exact RNS representation of a small integer polynomial's
// sparse full-ring lift. It does NOT assume native full/small FFT rounding is
// bitwise identical. periodic_encoder_check compares against the native oracle.
#include "compact_layout.h"
#include "poseidon/ckks_encoder.h"
#include "poseidon/basics/util/ntt.h"

namespace bert::compact {
class PeriodicEncoderExperiment {
public:
    PeriodicEncoderExperiment(const poseidon::PoseidonContext &full,
            poseidon::parms_id_type full_id, std::size_t slot_period = 256)
        : full_(full), full_id_(full_id), degree_(full.parameters_literal()->degree()),
          small_degree_(2 * slot_period), q_(q_prefix(full, full_id)),
          small_(small_context(q_, small_degree_)), encoder_(small_) {
        if (!power_of_two(small_degree_) || small_degree_ >= degree_ || degree_ % small_degree_)
            throw std::invalid_argument("periodic encoder requires a power-of-two subring");
        full_to_compact_.resize(degree_);
        const auto &reversed = bit_reversed_indices(degree_);
        for (std::size_t j = 0; j < degree_; ++j) full_to_compact_[j] = reversed[j] & (small_degree_ - 1);
        const auto *full_tables = full_.crt_context()->small_ntt_tables();
        const auto *small_tables = small_.crt_context()->small_ntt_tables();
        maps_.resize(q_.size());
        const auto small_log = log2_exact(small_degree_);
        for (std::size_t l = 0; l < q_.size(); ++l) {
            if (q_[l].value() > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("experimental compact Q residue must fit uint32");
            const auto root = small_tables[l].get_root();
            const auto lifted_root = poseidon::util::exponentiate_uint_mod(
                full_tables[l].get_root(), degree_ / small_degree_, q_[l]);
            const auto root_squared = poseidon::util::multiply_uint_mod(root, root, q_[l]);
            std::uint64_t power = root;
            std::size_t exponent = 0;
            for (std::size_t odd = 1; odd < 2 * small_degree_; odd += 2) {
                if (power == lifted_root) { exponent = odd; break; }
                power = poseidon::util::multiply_uint_mod(power, root_squared, q_[l]);
            }
            if (!exponent) throw std::logic_error("full/small primitive roots are not related");
            auto &map = maps_[l];
            map.resize(small_degree_);
            std::vector<bool> visited(small_degree_);
            for (std::size_t k = 0; k < small_degree_; ++k) {
                const auto odd = (exponent * (2 * k + 1)) & (2 * small_degree_ - 1);
                map[k] = reverse_bits(static_cast<std::uint32_t>((odd - 1) / 2), small_log);
                if (visited[map[k]]) throw std::logic_error("root map is not a permutation");
                visited[map[k]] = true;
                if (poseidon::util::exponentiate_uint_mod(root, odd, q_[l]) !=
                    poseidon::util::exponentiate_uint_mod(lifted_root, 2 * k + 1, q_[l]))
                    throw std::logic_error("root-map equality failed");
            }
        }
    }

    ExactPeriodicRns encode(const std::vector<double> &period, double scale) const {
        if (period.size() != small_degree_ / 2 || !(scale > 0) || !std::isfinite(scale))
            throw std::invalid_argument("invalid periodic encoder input/scale");
        for (double v : period)
            if (!std::isfinite(v)) throw std::invalid_argument("nonfinite period value");
        poseidon::Plaintext small;
        // Same real doubles and physical scale; no quantization of weights.
        encoder_.encode(period, scale, small);
        ExactPeriodicRns result{degree_, q_.size(), small_degree_, {}};
        result.values.resize(q_.size() * small_degree_);
        for (std::size_t l = 0; l < q_.size(); ++l)
            for (std::size_t k = 0; k < small_degree_; ++k) {
                const auto value = small.data()[l * small_degree_ + maps_[l][k]];
                if (value >= q_[l].value()) throw std::logic_error("noncanonical small NTT residue");
                result.values[l * small_degree_ + k] = static_cast<std::uint32_t>(value);
            }
        return result;
    }

    // Materialize the genuine full-ring sparse polynomial's NTT residues.
    // The caller then uses the SAME full-N lossless compressor/verifier as
    // strict fullEncode. This deliberately keeps the full residue validation
    // contract, without paying for a full complex FFT or full NTT per term.
    void materialize(const ExactPeriodicRns &compact, double scale,
            poseidon::Plaintext &full) const {
        if (compact.degree != degree_ || compact.limbs != q_.size() ||
            compact.period != small_degree_ || compact.values.size() != q_.size()*small_degree_)
            throw std::invalid_argument("subring materialization shape mismatch");
        full.resize(full_, full_id_, degree_ * q_.size());
        for (std::size_t l = 0; l < q_.size(); ++l) {
            auto *dst = full.data() + l*degree_;
            const auto *src = compact.values.data() + l*small_degree_;
            for (std::size_t j = 0; j < degree_; ++j) dst[j] = src[full_to_compact_[j]];
        }
        full.parms_id() = full_id_;
        full.scale() = scale;
    }

    // Independent actual NTT oracle: g(Y)=Y must lift to f(X)=X^(N/n).
    // Compare EVERY full-ring residue under the actual backend, in EVERY Q
    // limb. This proves the modular root remap, not FFT rounding equivalence.
    void check_monomial_mapping() const {
        const auto logn = log2_exact(degree_);
        std::vector<std::uint64_t> big(degree_), small(small_degree_);
        for (std::size_t l = 0; l < q_.size(); ++l) {
            std::fill(big.begin(), big.end(), 0);
            std::fill(small.begin(), small.end(), 0);
            big[degree_ / small_degree_] = 1;
            small[1] = 1;
            poseidon::util::ntt_negacyclic_harvey(big.data(), full_.crt_context()->small_ntt_tables()[l]);
            poseidon::util::ntt_negacyclic_harvey(small.data(), small_.crt_context()->small_ntt_tables()[l]);
            for (std::size_t j = 0; j < degree_; ++j) {
                const auto k = reverse_bits(static_cast<std::uint32_t>(j), logn) & (small_degree_ - 1);
                if (big[j] != small[maps_[l][k]])
                    throw std::logic_error("full NTT monomial reconstruction differs");
            }
        }
    }

private:
    static std::vector<poseidon::Modulus> q_prefix(const poseidon::PoseidonContext &ctx,
            const poseidon::parms_id_type &id) {
        auto level = ctx.crt_context()->get_context_data(id);
        if (!level || !level->parms().p().empty())
            throw std::invalid_argument("periodic encoder needs a full-context Q-only target");
        return level->parms().q();
    }
    static poseidon::PoseidonContext small_context(const std::vector<poseidon::Modulus> &q,
            std::size_t degree) {
        const auto logn = log2_exact(degree);
        // q0_level0 supports even a one-prime encoding-only context. P is
        // deliberately empty; this path neither encrypts nor switches keys.
        poseidon::ParametersLiteral params(CKKS, logn, logn - 1, 30, 0, 0,
            poseidon::Modulus(0), q, {}, poseidon::sec_level_type::none);
        return poseidon::PoseidonContext(params);
    }
    const poseidon::PoseidonContext &full_;
    poseidon::parms_id_type full_id_;
    std::size_t degree_, small_degree_;
    std::vector<poseidon::Modulus> q_;
    poseidon::PoseidonContext small_;
    poseidon::CKKSEncoder encoder_;
    std::vector<std::vector<std::size_t>> maps_;
    std::vector<std::uint32_t> full_to_compact_;
};
} // namespace bert::compact
