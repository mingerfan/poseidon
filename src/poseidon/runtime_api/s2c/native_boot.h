#pragma once
#include "poseidon/gpu/gpu_ciphertext.h"
#include "poseidon/keygenerator.h"
#include <memory>
#include <string_view>

namespace poseidon::runtime_api::s2c
{
class NativeBoot
{
  public:
    NativeBoot(const PoseidonContext &context, KeyGenerator &generator, int device, std::string_view id);
    ~NativeBoot();
    int output_scale_log2() const;
    int output_level() const;
    void execute(const gpu::GpuCiphertextData &input, bool real_only, gpu::GpuCiphertextData &output);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace poseidon::runtime_api::s2c
