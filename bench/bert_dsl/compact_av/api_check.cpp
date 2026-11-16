// CPU-only public API/lifecycle check. No GPU headers, device calls, keys or
// valid production-ring preparation. Can run under no-device-cuda-guard.so.
#include "prepared_av.h"
#include "diagnostic_profile.h"
#include "poseidon/poseidon_context.h"
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>
#include <type_traits>

static_assert(!std::is_copy_constructible<bert::attention::PreparedAvEngine>::value);
static_assert(!std::is_copy_assignable<bert::attention::PreparedAvEngine>::value);
int main() {
    try {
        using namespace poseidon;
        using namespace bert::attention;
        std::vector<Modulus> q, p;
        auto ordered_q = av_diagnostic::q52(); ordered_q.resize(11);
        for (auto prime : ordered_q) q.emplace_back(prime);
        for (auto prime : av_diagnostic::p11()) p.emplace_back(prime);
        ParametersLiteral literal(CKKS, 10, 9, 28, 0, 1, Modulus(0), q, p, sec_level_type::none);
        PoseidonContext context(literal);
        PreparedAvEngine engine(context, nullptr, nullptr); // deliberately no CUDA
        const auto before = engine.engine_stats();
        if (before.allocated_bytes || before.run_calls || before.mask_bundle_count || before.mask_plaintext_count ||
            before.relin_key_upload_calls || before.galois_key_upload_calls || before.preparation_locked)
            throw std::runtime_error("constructor allocated GPU owners or performed work");
        AvSpec spec; spec.id = "invalid-small-ring";
        if (spec.algorithm != AvAlgorithm::Baseline || std::string(av_algorithm_name(spec.algorithm)) != "Baseline" ||
            std::string(av_algorithm_name(AvAlgorithm::PackedInput2Partial)) != "PackedInput2Partial" ||
            std::string(av_output_semantics(AvAlgorithm::PackedInput2Partial)).find("gap_B_current") == std::string::npos)
            throw std::runtime_error("algorithm default / explicit partial output contract mismatch");
        bool rejected = false;
        try { engine.prepare(spec); } catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected || engine.engine_stats().allocated_bytes)
            throw std::runtime_error("invalid ring was not rejected before GPU setup");
        auto unknown = spec; unknown.algorithm = static_cast<AvAlgorithm>(999);
        bool unknown_rejected = false;
        try { engine.prepare(unknown); } catch (const std::invalid_argument &e) {
            unknown_rejected = std::string(e.what()).find("unknown prepared AV algorithm") != std::string::npos;
        }
        if (!unknown_rejected || engine.engine_stats().preparation_locked)
            throw std::runtime_error("unknown algorithm must fail CPU validation without locking prepare");
        const auto direct = PreparedAvEngine::required_rotation_steps();
        const auto binary = PreparedAvEngine::required_binary_rotation_steps();
        const std::set<int> steps(direct.begin(), direct.end()), bits(binary.begin(), binary.end());
        if (steps.size() != 52 || steps.count(0) || direct.size() != steps.size())
            throw std::runtime_error("direct rotation inventory mismatch");
        auto need = [&](int step) {
            if (!steps.count(step)) throw std::runtime_error("missing native schedule rotation");
            const int sign = step < 0 ? -1 : 1;
            unsigned value = static_cast<unsigned>(step < 0 ? -step : step);
            for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
                if ((value & 1) && !bits.count(sign * int(bit)))
                    throw std::runtime_error("binary inventory cannot realize native rotation");
        };
        for (int b = 1; b < 8; ++b) { need(256*b); need(b); }
        for (int b = 0; b < 8; ++b) need(b-128);
        for (int g = 1; g < 16; ++g) { need(2048*g); need(-2040*g); }
        const auto packed = PreparedAvEngine::required_packed_input_rotation_steps();
        const auto packed_binary = PreparedAvEngine::required_packed_input_binary_rotation_steps();
        const std::set<int> packed_steps(packed.begin(), packed.end()), packed_bits(packed_binary.begin(), packed_binary.end());
        if (packed_steps.size() != 55 || packed.size() != packed_steps.size() || packed_steps.count(0) ||
            !std::includes(packed_steps.begin(), packed_steps.end(), steps.begin(), steps.end()) ||
            !packed_steps.count(-64) || !packed_steps.count(64) || !packed_steps.count(192))
            throw std::runtime_error("packed55 inventory is not baseline52 plus3 packing rotations");
        for (int step : packed_steps) {
            const int sign = step < 0 ? -1 : 1;
            unsigned value = static_cast<unsigned>(step < 0 ? -step : step);
            for (unsigned bit = 1; value; bit <<= 1, value >>= 1)
                if ((value & 1) && !packed_bits.count(sign * int(bit)))
                    throw std::runtime_error("packed binary inventory cannot realize packed schedule");
        }
        std::cout << "{\"cpu_only\":true,\"constructor_no_device_calls\":true,"
                     "\"invalid_prepare_rejected_before_device_calls\":true,\"direct_rotations\":"
                  << direct.size() << ",\"binary_rotations\":" << binary.size()
                  << ",\"packed_input2_partial_direct_rotations\":" << packed.size()
                  << ",\"packed_input2_partial_binary_rotations\":" << packed_binary.size() << "}\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << "AV API check: " << e.what() << '\n'; return 1; }
}
