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
    // 1. Extract metadata from src:
    //    src.size()
    //    src.poly_modulus_degree()
    //    src.coeff_modulus_size()
    //    src.parms_id()
    //    src.scale()
    //    src.is_ntt_form()
    //
    // 2. Create GpuCiphertextData::create_single_device(...)
    //
    // 3. For each component i:
    //    cudaMemcpy GPU field from src.data(i)
    //
    // 4. Return GPU ciphertext.

    (void)src;
    (void)device_id;

    throw std::runtime_error("GpuUploader::upload_ciphertext is not implemented yet");
}

void GpuUploader::download_ciphertext(
    const GpuCiphertextData &src,
    Ciphertext &dst)
{
    // TODO:
    // 1. Resize dst according to src.meta.
    // 2. Copy each GPU field back into dst.data(component_id).
    // 3. Set dst.parms_id(), dst.scale(), dst.is_ntt_form().
    //
    // This function should be called explicitly. GPU evaluators should not
    // hide CPU/GPU transfer inside add/multiply/ntt operations.

    (void)src;
    (void)dst;

    throw std::runtime_error("GpuUploader::download_ciphertext is not implemented yet");
}

}  // namespace gpu
}  // namespace poseidon