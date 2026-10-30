// Public evaluation-key coefficient counts only. Never export key contents.
// These counts exclude allocation slack, keygen caches and object overhead.
#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>

template<class Cipher>
std::uint64_t public_key_cipher_coeff_bytes(const Cipher &ct) {
    return std::uint64_t(ct.size()) * ct.poly_modulus_degree() *
           ct.coeff_modulus_size() * sizeof(std::uint64_t);
}

template<class SwitchingKeys>
nlohmann::json switching_key_payload_stats(const SwitchingKeys &keys) {
    std::uint64_t bytes=0, parts=0, groups=0;
    for (const auto &group : keys.data()) {
        if (!group.empty()) ++groups;
        for (const auto &public_part : group) {
            bytes += public_key_cipher_coeff_bytes(public_part.data());
            ++parts;
        }
    }
    return {{"nonempty_groups",groups},{"ciphertext_parts",parts},
            {"coefficient_bytes",bytes}};
}

inline std::uint64_t current_process_rss_bytes() {
    std::ifstream f("/proc/self/status");
    if (!f) throw std::runtime_error("Cannot read current process RSS");
    std::string line;
    while (std::getline(f,line)) {
        if (line.compare(0,6,"VmRSS:")==0)
            return std::stoull(line.substr(6))*1024;
    }
    throw std::runtime_error("VmRSS absent in process status");
}
