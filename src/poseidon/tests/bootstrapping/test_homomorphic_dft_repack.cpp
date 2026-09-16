#include "poseidon/advance/homomorphic_dft.h"

#include <complex>
#include <cstdlib>
#include <iostream>

int main()
{
    constexpr std::uint32_t log_slots = 4;
    constexpr std::size_t slots = std::size_t{1} << log_slots;
    const auto matrix = poseidon::gen_repack_matrix(log_slots, false);
    if (matrix.size() != 2 || !matrix.contains(0) ||
        !matrix.contains(static_cast<int>(slots)))
    {
        std::cerr << "unexpected sparse repack diagonals\n";
        return EXIT_FAILURE;
    }
    const auto &diagonal0 = matrix.at(0);
    const auto &diagonal_n = matrix.at(static_cast<int>(slots));
    if (diagonal0.size() != 2 * slots || diagonal_n.size() != 2 * slots)
    {
        std::cerr << "unexpected sparse repack diagonal length\n";
        return EXIT_FAILURE;
    }
    for (std::size_t i = 0; i < slots; ++i)
    {
        if (diagonal0[i] != std::complex<double>(1.0, 0.0) ||
            diagonal0[i + slots] != std::complex<double>(0.0, 1.0) ||
            diagonal_n[i] != std::complex<double>(0.0, 1.0) ||
            diagonal_n[i + slots] != std::complex<double>(1.0, 0.0))
        {
            std::cerr << "incorrect sparse repack coefficient\n";
            return EXIT_FAILURE;
        }
    }

    const auto assert_sparse_matrix_shape = [](const auto &matrices) {
        if (matrices.empty())
        {
            std::cerr << "sparse DFT generated no matrices\n";
            return false;
        }
        for (const auto &stage : matrices)
        {
            if (stage.empty())
            {
                std::cerr << "sparse DFT generated an empty stage\n";
                return false;
            }
            for (const auto &[rotation, diagonal] : stage)
            {
                (void)rotation;
                if (diagonal.size() != 2 * slots)
                {
                    std::cerr << "sparse DFT diagonal has the wrong length\n";
                    return false;
                }
            }
        }
        return true;
    };

    poseidon::HomomorphicDFTMatrixLiteral slots_to_coeffs(
        poseidon::decode, 6, log_slots, 4, {1, 1}, true, 1.0, false, 1);
    const auto stc_matrices = slots_to_coeffs.gen_matrices();
    if (!assert_sparse_matrix_shape(stc_matrices))
        return EXIT_FAILURE;

    poseidon::HomomorphicDFTMatrixLiteral coeffs_to_slots(
        poseidon::encode, 6, log_slots, 4, {1, 1}, true, 1.0, false, 1);
    const auto cts_matrices = coeffs_to_slots.gen_matrices();
    if (!assert_sparse_matrix_shape(cts_matrices))
        return EXIT_FAILURE;
    for (const auto &[rotation, diagonal] : cts_matrices.back())
    {
        (void)rotation;
        for (std::size_t i = slots; i < 2 * slots; ++i)
        {
            if (diagonal[i] != std::complex<double>(0.0, 0.0))
            {
                std::cerr << "sparse CoeffsToSlots did not mask the second half\n";
                return EXIT_FAILURE;
            }
        }
    }
    std::cout << "sparse DFT repack matrix passed\n";
    return EXIT_SUCCESS;
}
