// Read-only ciphertext metadata for tests. No encryption/evaluation/decryption.
// Built against the same pinned SEAL headers/ABI as stock libSEAL_HEVM.so.
#include <seal/seal.h>
#include <cstdint>
#include <fstream>
#include <seal/util/galois.h>

// Inspect actual public evaluation-key material before any native VM execution.
// Return 2 for missing required keys, 1 for invalid files/parameters/step bounds.
extern "C" int verify_galois_file(const char *parameters, const char *galois,
                                  const std::int64_t *steps, std::uint64_t count) {
  try {
    if (!parameters || !galois || (count && !steps) || count > 6) return 1;
    std::ifstream param_stream(parameters, std::ios::binary);
    seal::EncryptionParameters parms;
    parms.load(param_stream);
    seal::SEALContext context(parms, true, seal::sec_level_type::tc128);
    if (!context.parameters_set() || parms.poly_modulus_degree() != 32768) return 1;
    std::ifstream key_stream(galois, std::ios::binary);
    seal::GaloisKeys keys;
    keys.load(context, key_stream);
    for (std::uint64_t i = 0; i < count; ++i) {
      if (!steps[i] || steps[i] < -3 || steps[i] > 3) return 1;
      const auto element = context.key_context_data()->galois_tool()->get_elt_from_step(static_cast<int>(steps[i]));
      if (!keys.has_key(element)) return 2;
    }
    return 0;
  } catch (...) {
    return 1;
  }
}

extern "C" int describe_cipher(const void *pointer, std::uint64_t *level,
                                double *scale, std::uint64_t *polynomials) {
  if (!pointer || !level || !scale || !polynomials) return 1;
  const auto &cipher = *static_cast<const seal::Ciphertext *>(pointer);
  *level = cipher.coeff_modulus_size();
  *scale = cipher.scale();
  *polynomials = cipher.size();
  return 0;
}

// Public ciphertext-only observation. No plaintext/key access or encryption.
extern "C" int inspect_cipher_integrity(const void *pointer, std::uint64_t *fingerprint) {
  if (!pointer || !fingerprint) return 1;
  const auto &cipher = *static_cast<const seal::Ciphertext *>(pointer);
  if (cipher.is_transparent() || cipher.size() != 2 || !cipher.is_ntt_form()) return 2;
  std::uint64_t hash = 14695981039346656037ULL;
  const auto count = cipher.size() * cipher.poly_modulus_degree() * cipher.coeff_modulus_size();
  for (std::size_t i = 0; i < count; ++i) {
    hash ^= cipher.data()[i];
    hash *= 1099511628211ULL;
  }
  *fingerprint = hash;
  return 0;
}
