// Key setup only. Evaluation remains in the unmodified upstream SEAL_HEVM.
// Match SEAL_HEVM::create_context parameters; provision only required rotations.
#include <seal/seal.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <string>

template <typename T>
void save(const std::filesystem::path &path, const T &object) {
  if (std::filesystem::exists(path))
    throw std::runtime_error("Refusing to overwrite existing key material");
  std::ofstream stream(path, std::ios::binary);
  stream.exceptions(std::ios::badbit | std::ios::failbit);
  object.save(stream);
}

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 8)
      throw std::runtime_error("Usage: seal_golden_keys NEW_PRIVATE_DIRECTORY [ROTATION_STEPS...]");
    std::vector<int> rotations{1, 2};
    if (argc > 2) {
      rotations.clear();
      for (int i = 2; i < argc; ++i) {
        const std::string text(argv[i]);
        std::size_t consumed = 0;
        const int step = std::stoi(text, &consumed);
        if (consumed != text.size() || step == 0 || step < -3 || step > 3 ||
            std::find(rotations.begin(), rotations.end(), step) != rotations.end())
          throw std::runtime_error("Invalid or duplicate rotation step");
        rotations.push_back(step);
      }
      std::sort(rotations.begin(), rotations.end());
    }
    const std::filesystem::path dir(argv[1]);
    if (!std::filesystem::is_directory(dir) || !std::filesystem::is_empty(dir))
      throw std::runtime_error("Requires a fresh, empty private directory");
    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    parms.set_poly_modulus_degree(32768);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(32768, std::vector<int>(14, 60)));
    // Explicitly require the same default tc128 validation as stock SEAL_HEVM.
    seal::SEALContext context(parms, true, seal::sec_level_type::tc128);
    if (!context.parameters_set())
      throw std::runtime_error(context.parameter_error_message());
    if (context.first_context_data()->parms().coeff_modulus().size() != 13)
      throw std::runtime_error("Unexpected SEAL data chain");
    save(dir / "parm.seal", parms);
    seal::KeyGenerator generator(context);
    seal::PublicKey pub;
    generator.create_public_key(pub);
    save(dir / "pub.seal", pub);
    save(dir / "sec.seal", generator.secret_key());
    seal::RelinKeys relin;
    generator.create_relin_keys(relin);
    save(dir / "relin.seal", relin);
    seal::GaloisKeys gal;
    generator.create_galois_keys(rotations, gal);
    save(dir / "gal.seal", gal);
    std::cout << "{\"seal_version\":\"" << SEAL_VERSION
              << "\",\"polynomial_degree\":32768,\"slots\":16384,"
                 "\"security_check\":\"tc128\",\"parameters_set\":true,"
                 "\"data_modulus_count\":13,\"rotation_steps\":[";
    for (std::size_t i = 0; i < rotations.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << rotations[i];
    }
    std::cout << "],\"modulus_bits\":[";
    for (std::size_t i = 0; i < parms.coeff_modulus().size(); ++i) {
      if (i) std::cout << ',';
      std::cout << parms.coeff_modulus()[i].bit_count();
    }
    std::cout << "],\"modulus_values\":[";
    for (std::size_t i = 0; i < parms.coeff_modulus().size(); ++i) {
      if (i) std::cout << ',';
      std::cout << '\"' << parms.coeff_modulus()[i].value() << '\"';
    }
    std::cout << "]}\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
