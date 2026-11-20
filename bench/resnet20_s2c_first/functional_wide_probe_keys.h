#pragma once

#include "poseidon/gpu/gpu_wide_bv.h"
#include "poseidon/key/publickey.h"

#include <algorithm>

namespace poseidon::gpu::probe
{

inline void upload_bv_key(const std::vector<PublicKey> &digits, std::size_t degree,
                          std::size_t q_count, std::vector<std::uint64_t> &scratch,
                          DeviceVector<std::uint64_t> &destination)
{
    const std::size_t q_words = q_count * degree;
    const std::size_t qp_words = q_words + degree;
    for (std::size_t digit = 0; digit < q_count; ++digit)
    {
        const Ciphertext &key = digits.at(digit).data();
        const std::size_t p_offset = (key.coeff_modulus_size() - 1) * degree;
        for (std::size_t component = 0; component < 2; ++component)
        {
            auto *row = scratch.data() + (2 * digit + component) * qp_words;
            std::copy_n(key.data(component), q_words, row);
            std::copy_n(key.data(component) + p_offset, degree, row + q_words);
        }
    }
    destination.copy_from_host(scratch.data(), scratch.size());
}

} // namespace poseidon::gpu::probe
