#include "poseidon/basics/util/uintarith.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

int main()
{
    const std::array<std::uint64_t, 6> values{
        0, 1, 2, (std::uint64_t{1} << 63), UINT64_MAX - 1, UINT64_MAX};
    for (auto a : values)
        for (auto b : values)
            for (unsigned char borrow = 0; borrow < 2; ++borrow)
            {
                unsigned long long result;
                const auto actual = poseidon::util::sub_uint64(a, b, borrow, &result);
                const unsigned __int128 rhs = static_cast<unsigned __int128>(b) + borrow;
                const auto expected_borrow = static_cast<unsigned __int128>(a) < rhs;
                const auto expected = static_cast<std::uint64_t>(
                    static_cast<unsigned __int128>(a) - rhs);
                if (actual != expected_borrow || result != expected)
                {
                    std::cerr << "borrow/value mismatch\n";
                    return 1;
                }
            }
    std::cout << "72 unsigned subtraction boundary cases passed\n";
    return 0;
}
