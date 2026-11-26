#pragma once
// Layer-to-layer encrypted hidden-state checkpoint using Poseidon's native
// Ciphertext::save / Ciphertext::load. Validates parameter + secret-key
// fingerprints so checkpoints can never be misapplied across keys or fixtures.
// Format is unified across 1-GPU and 2-GPU runners (always 16 global tiles).
#include "source_cache.hpp"
#include "poseidon/ciphertext.h"
#include "poseidon/gpu/gpu_uploader.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cerium_source::checkpoint {

namespace fs = std::filesystem;

inline constexpr char kCkptMagic[16] = {'L','L','A','M','A','_','C','K','P','T','_','V','1',0,0,0};

struct Metadata {
    std::uint64_t params_fp = 0;
    std::uint64_t secret_key_fp = 0;
    std::uint64_t completed_layers = 0;
    std::uint64_t total_tiles = 0;
    std::uint64_t valid_tokens = 0;
};

inline fs::path layer_checkpoint_path(const std::string &dir, std::size_t completed_layers) {
    return fs::path(dir) / ("layer_" + std::to_string(completed_layers) + ".ckpt");
}

// Save CPU ciphertexts (already downloaded from 1 or 2 GPUs) atomically.
inline void save_cpu_tiles(const fs::path &path,
                           const poseidon::PoseidonContext &context,
                           const poseidon::SecretKey &sk,
                           const std::vector<poseidon::Ciphertext> &tiles,
                           std::size_t completed_layers,
                           std::size_t valid_tokens) {
    if (tiles.empty()) throw std::invalid_argument("checkpoint: cannot save zero tiles");
    const std::uint64_t p_fp = cache::parameters_fingerprint(*context.parameters_literal());
    const std::uint64_t sk_fp = cache::secret_key_fingerprint(sk);
    cache::AtomicWriter writer(path);
    auto &os = writer.stream();
    os.write(kCkptMagic, 16);
    cache::detail::write_pod(os, p_fp);
    cache::detail::write_pod(os, sk_fp);
    cache::detail::write_pod(os, static_cast<std::uint64_t>(completed_layers));
    cache::detail::write_pod(os, static_cast<std::uint64_t>(tiles.size()));
    cache::detail::write_pod(os, static_cast<std::uint64_t>(valid_tokens));
    for (const auto &ct : tiles) {
        ct.save(os, poseidon::compr_mode_type::none);
    }
    writer.commit();
}

// Single-GPU convenience wrapper: downloads GPU ciphertexts and writes checkpoint.
inline void save_gpu_tiles(const fs::path &path,
                           const poseidon::PoseidonContext &context,
                           const poseidon::SecretKey &sk,
                           const std::vector<poseidon::gpu::GpuCiphertextData> &gpu_tiles,
                           std::size_t completed_layers,
                           std::size_t valid_tokens) {
    std::vector<poseidon::Ciphertext> cpu_tiles(gpu_tiles.size());
    for (std::size_t i = 0; i < gpu_tiles.size(); ++i) {
        poseidon::gpu::GpuUploader::download_ciphertext(gpu_tiles[i], cpu_tiles[i], context);
    }
    save_cpu_tiles(path, context, sk, cpu_tiles, completed_layers, valid_tokens);
}

// Load and validate CPU ciphertexts from a checkpoint file.
inline std::vector<poseidon::Ciphertext> load_cpu_tiles(const fs::path &path,
                                                        const poseidon::PoseidonContext &context,
                                                        const poseidon::SecretKey &sk,
                                                        Metadata &meta_out) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("checkpoint: cannot open " + path.string());
    char magic[16]{};
    is.read(magic, 16);
    if (!is || std::memcmp(magic, kCkptMagic, 16) != 0)
        throw std::runtime_error("checkpoint: invalid magic in " + path.string());
    meta_out.params_fp = cache::detail::read_pod<std::uint64_t>(is);
    meta_out.secret_key_fp = cache::detail::read_pod<std::uint64_t>(is);
    meta_out.completed_layers = cache::detail::read_pod<std::uint64_t>(is);
    meta_out.total_tiles = cache::detail::read_pod<std::uint64_t>(is);
    meta_out.valid_tokens = cache::detail::read_pod<std::uint64_t>(is);

    const std::uint64_t expected_p_fp = cache::parameters_fingerprint(*context.parameters_literal());
    const std::uint64_t expected_sk_fp = cache::secret_key_fingerprint(sk);
    if (meta_out.params_fp != expected_p_fp)
        throw std::runtime_error("checkpoint: parameter fingerprint mismatch in " + path.string());
    if (meta_out.secret_key_fp != expected_sk_fp)
        throw std::runtime_error(
            "checkpoint: secret key fingerprint mismatch (load the matching --dev-key-cache / LLAMA_DEV_KEY_CACHE_DIR)");
    if (meta_out.total_tiles == 0)
        throw std::runtime_error("checkpoint: empty tile count in " + path.string());

    std::vector<poseidon::Ciphertext> tiles;
    tiles.reserve(static_cast<std::size_t>(meta_out.total_tiles));
    for (std::size_t i = 0; i < meta_out.total_tiles; ++i) {
        poseidon::Ciphertext ct;
        ct.load(context, is);
        tiles.push_back(std::move(ct));
    }
    return tiles;
}

// Single-GPU convenience wrapper: reads checkpoint and uploads all tiles to device_id.
inline std::vector<poseidon::gpu::GpuCiphertextData> load_gpu_tiles(const fs::path &path,
                                                                    const poseidon::PoseidonContext &context,
                                                                    const poseidon::SecretKey &sk,
                                                                    int device_id,
                                                                    Metadata &meta_out) {
    auto cpu_tiles = load_cpu_tiles(path, context, sk, meta_out);
    std::vector<poseidon::gpu::GpuCiphertextData> gpu_tiles;
    gpu_tiles.reserve(cpu_tiles.size());
    for (const auto &ct : cpu_tiles) {
        gpu_tiles.push_back(poseidon::gpu::GpuUploader::upload_ciphertext(ct, device_id));
    }
    return gpu_tiles;
}

} // namespace cerium_source::checkpoint
