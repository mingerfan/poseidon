#include "poseidon/basics/modulus.h"
#include <gmpxx.h>
#include <array>
#include <cstring>
#include <iostream>
#include <numeric>

// Read-only admission diagnostic. No key generation or encryption.
int main()
{
    if (std::strcmp(gmp_version, "6.3.0") != 0)
        return 1;
    poseidon::Modulus modulus(17);
    if (modulus.value() != 17)
        return 1;
    const std::array<int, 34> q_bits{
        32,32,32,32,32,32,32,32,32,32,32,32,
        28,28,31,31,32,32,30,31,32,31,32,
        32,31,31,31,32,32,31,32,32,32,30};
    // Every b-bit prime is >= 2^(b-1); this lower bound is enough to reject.
    const int lower = std::accumulate(q_bits.begin(), q_bits.end(), 0)
                      - static_cast<int>(q_bits.size()) + 9 * (32 - 1) + 1;
    for (std::size_t degree : {16384, 32768, 65536})
    {
        const int bound = poseidon::CoeffModulus::MaxBitCount(
            degree, poseidon::sec_level_type::tc128);
        std::cout << "degree=" << degree << " tc128_max_bits=" << bound
                  << " existing_test_qp_min_bits=" << lower << "\n";
        if (bound >= lower)
            return 1; // Investigate if the library's security table changes.
    }
    std::cout << "GMP 6.3.0 and Poseidon loaded; legacy bootstrap parameters "
                 "rejected by tc128 capacity; 65536 requires separate security review\n";
    return 0;
}
