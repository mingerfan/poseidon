#pragma once
// Explicit handwritten execution contract, not a DSL compiler or proof of
// numerical accuracy. Both plans use N65536 and require independent GPU gates.
#include "source_cerium_layout.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace cerium_source::manual {
enum class Mode {HighPrecisionReal,HighPrecisionPairedSiluCandidate,Fast};
inline Mode parse_mode(std::string_view name) {
    if(name=="manual-highprecision-real-v1")return Mode::HighPrecisionReal;
    if(name=="manual-highprecision-paired-silu-candidate-v1")return Mode::HighPrecisionPairedSiluCandidate;
    // Optimization plan: paired SiLU plus optional paired Softmax/RMS refreshes
    // and fast linear; Q/scale/tile/seed boundaries are still enforced, Boot
    // counts are measured rather than fixed.
    if(name=="manual-fast-v1")return Mode::Fast;
    throw std::invalid_argument("explicit supported manual Llama plan required");
}
struct Plan {
    Mode mode;
    Plan()=delete;
    explicit constexpr Plan(Mode selected):mode(selected){}
    constexpr const char *name() const {
        return mode==Mode::HighPrecisionReal?"manual-highprecision-real-v1":mode==Mode::Fast?"manual-fast-v1":"manual-highprecision-paired-silu-candidate-v1";
    }
    constexpr bool paired_silu() const {return mode!=Mode::HighPrecisionReal;}
    constexpr bool fixed_boot_counts() const {return mode!=Mode::Fast;}
    constexpr std::uint64_t secondary_boots() const {return paired_silu()?112:224;}
    constexpr std::uint64_t total_boots() const {return 252+secondary_boots();}
    constexpr std::uint64_t projection_pairs() const {return paired_silu()?112:0;}
    void validate() const {
        if(mode!=Mode::HighPrecisionReal&&mode!=Mode::HighPrecisionPairedSiluCandidate&&mode!=Mode::Fast)
            throw std::invalid_argument("invalid manual plan enum");
        layout::require_ring(16,15);
    }
};
struct Boundary {const char *name;std::size_t tiles,q;std::uint64_t primary_boots;};
inline constexpr std::array<Boundary,15> boundaries{{
    {"attention_rms",16,16,22},{"q_projection",16,14,22},{"k_projection",4,14,22},
    {"v_projection",4,14,22},{"qk",16,7,22},{"softmax",16,11,230},{"av",16,8,230},
    {"o_projection",16,6,230},{"attention_residual",16,6,230},{"ffn_rms",16,16,252},
    {"gate_projection",56,14,252},{"up_projection",56,14,252},{"swiglu",56,10,252},
    {"down_projection",16,8,252},{"complete",16,6,252}}};
// Consumer evidence, not a requested/hardcoded flag. The caller reads actual
// ciphertext metadata and successful API/exp-branch counters after each stage.
struct Observation {
    std::string_view name;
    std::size_t tiles,q;
    double scale;
    std::uint64_t primary_boots,secondary_boots,projection_pairs,integer_exp_seeds,standard_exp_seeds;
};
class BoundaryTrace {
    Plan plan_;std::size_t cursor_=0;
public:
    explicit BoundaryTrace(Plan plan):plan_(plan){plan_.validate();}
    void observe(const Observation &o) {
        if(cursor_>=boundaries.size())throw std::runtime_error("extra manual model boundary");
        const auto &b=boundaries[cursor_];
        const auto secondary=cursor_>=12?plan_.secondary_boots():0;
        const auto pairs=cursor_>=12?plan_.projection_pairs():0;
        const auto seeds=cursor_>=5?16U:0U;
        if(o.name!=b.name||o.tiles!=b.tiles||(plan_.fixed_boot_counts()?o.q!=b.q:o.q<6)||!std::isfinite(o.scale)||o.scale<=0||
           std::abs(o.scale-std::exp2(40.))>1e-10*std::max(o.scale,std::exp2(40.))||
           (plan_.fixed_boot_counts()&&(o.primary_boots!=b.primary_boots||o.secondary_boots!=secondary||o.projection_pairs!=pairs))||
           o.integer_exp_seeds!=seeds||o.standard_exp_seeds!=0)
            throw std::runtime_error("manual model observed scale/Q/Boot/seed schedule mismatch");
        ++cursor_;
    }
    void finish() const {if(cursor_!=boundaries.size())throw std::runtime_error("incomplete manual model schedule");}
    std::size_t observed_boundaries() const {return cursor_;}
};
} // namespace cerium_source::manual
