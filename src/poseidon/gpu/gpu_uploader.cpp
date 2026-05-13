#include "poseidon/gpu/gpu_uploader.h"

#include <stdexcept>

namespace poseidon
{
namespace gpu
{

GpuCiphertextData GpuUploader::upload_ciphertext(
    const Ciphertext &src,
    int device_id)
{
    // TODO:
    // Implement CPU Ciphertext -> GPU Ciphertext upload.
    //
    // Important:
    // CPU Poseidon stores residues as uint64_t.
    // GPU backend stores residues as uint32_t.
    //
    // This conversion is valid only when all active RNS primes are below 32 bits
    // and residues are in canonical range.

    (void)src;
    (void)device_id;

    throw std::runtime_error("GpuUploader::upload_ciphertext is not implemented yet");
}

void GpuUploader::download_ciphertext(
    const GpuCiphertextData &src,
    Ciphertext &dst)
{
    // TODO:
    // Implement GPU Ciphertext -> CPU Ciphertext download.

    (void)src;
    (void)dst;

    throw std::runtime_error("GpuUploader::download_ciphertext is not implemented yet");
}

GpuPlaintextData GpuUploader::upload_plaintext(
    const Plaintext &src,
    int device_id)
{
    // TODO:
    // Implement CPU Plaintext -> GPU Plaintext upload.

    (void)src;
    (void)device_id;

    throw std::runtime_error("GpuUploader::upload_plaintext is not implemented yet");
}

void GpuUploader::download_plaintext(
    const GpuPlaintextData &src,
    Plaintext &dst)
{
    // TODO:
    // Implement GPU Plaintext -> CPU Plaintext download.

    (void)src;
    (void)dst;

    throw std::runtime_error("GpuUploader::download_plaintext is not implemented yet");
}

GpuRelinKeysData GpuUploader::upload_relin_keys(
    const RelinKeys &src,
    int device_id)
{
    // TODO:
    // Implement CPU RelinKeys -> GPU RelinKeys upload.

    (void)src;
    (void)device_id;

    throw std::runtime_error("GpuUploader::upload_relin_keys is not implemented yet");
}

GpuGaloisKeysData GpuUploader::upload_galois_keys(
    const GaloisKeys &src,
    int device_id)
{
    // TODO:
    // Implement CPU GaloisKeys -> GPU GaloisKeys upload.

    (void)src;
    (void)device_id;

    throw std::runtime_error("GpuUploader::upload_galois_keys is not implemented yet");
}

}  // namespace gpu
}  // namespace poseidon