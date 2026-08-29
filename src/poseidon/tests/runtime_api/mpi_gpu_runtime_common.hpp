#pragma once

#include "poseidon/basics/modulus.h"
#include "poseidon/keygenerator.h"
#include "poseidon/parameters_literal.h"
#include "runtime/plan.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace poseidon::runtime_api::test
{

inline void check_mpi(int status, const char *what)
{
    if (status == MPI_SUCCESS)
    {
        return;
    }
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    (void)MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(what) + ": " +
                             std::string(message,
                                         static_cast<std::size_t>(length)));
}

inline std::vector<int> parse_rank_to_node(const std::string &encoded)
{
    std::vector<int> result;
    std::size_t begin = 0;
    while (begin <= encoded.size())
    {
        const std::size_t end = encoded.find('x', begin);
        const std::string token = encoded.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (token.empty())
        {
            throw std::invalid_argument("rank-to-node contains an empty entry");
        }
        std::size_t consumed = 0;
        const int value = std::stoi(token, &consumed);
        if (consumed != token.size() || value < 0)
        {
            throw std::invalid_argument(
                "rank-to-node entries must be nonnegative integers");
        }
        result.push_back(value);
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return result;
}

inline std::string encode_x_list(const std::vector<int> &values)
{
    std::ostringstream encoded;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0)
        {
            encoded << 'x';
        }
        encoded << values[index];
    }
    return encoded.str();
}

inline std::uint32_t exact_log2(std::uint64_t value)
{
    if (value < 2 || (value & (value - 1)) != 0)
    {
        throw std::runtime_error("poly_degree must be a power of two");
    }
    std::uint32_t result = 0;
    while (value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

inline poseidon::PoseidonContext make_context(const fhegpu::OperatorSpec &spec)
{
    std::vector<int> modulus_bits = spec.rns_moduli_log2;
    modulus_bits.insert(modulus_bits.end(), 2, spec.max_modulus_log2);
    const auto moduli =
        poseidon::CoeffModulus::Create(spec.poly_degree, modulus_bits);
    const std::size_t q_count = spec.rns_moduli_log2.size();
    std::vector<poseidon::Modulus> q(moduli.begin(), moduli.begin() + q_count);
    std::vector<poseidon::Modulus> p(moduli.begin() + q_count, moduli.end());
    const std::uint32_t log_n = exact_log2(spec.poly_degree);
    poseidon::ParametersLiteral parameters(
        CKKS, log_n, log_n - 1,
        static_cast<std::uint32_t>(spec.default_scale_log2), 0, 0,
        poseidon::Modulus(0), q, p, poseidon::sec_level_type::none);
    return poseidon::PoseidonContext(parameters);
}

inline const fhegpu::ValueDesc &find_value(const fhegpu::RuntimePlan &plan,
                                           fhegpu::ValueId id)
{
    const auto found = std::find_if(
        plan.values.begin(), plan.values.end(),
        [id](const fhegpu::ValueDesc &value) { return value.id == id; });
    if (found == plan.values.end())
    {
        throw std::runtime_error("missing ValueDesc " + std::to_string(id));
    }
    return *found;
}

inline std::vector<int> local_cuda_devices(MPI_Comm world_comm,
                                           int local_device_count)
{
    MPI_Comm local_comm = MPI_COMM_NULL;
    check_mpi(MPI_Comm_split_type(world_comm, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &local_comm),
              "MPI_Comm_split_type");
    int local_rank = 0;
    check_mpi(MPI_Comm_rank(local_comm, &local_rank), "MPI_Comm_rank local");
    int device_offset = 0;
    check_mpi(MPI_Exscan(&local_device_count, &device_offset, 1, MPI_INT,
                         MPI_SUM, local_comm),
              "MPI_Exscan local device counts");
    if (local_rank == 0)
    {
        device_offset = 0;
    }
    check_mpi(MPI_Comm_free(&local_comm), "MPI_Comm_free local");

    std::vector<int> result(static_cast<std::size_t>(local_device_count));
    for (int index = 0; index < local_device_count; ++index)
    {
        result[static_cast<std::size_t>(index)] = device_offset + index;
    }
    return result;
}

inline void require_same_topology(const std::vector<int> &rank_to_node,
                                  int world_size)
{
    std::vector<int> gathered(static_cast<std::size_t>(world_size) *
                              rank_to_node.size());
    check_mpi(MPI_Allgather(rank_to_node.data(), world_size, MPI_INT,
                            gathered.data(), world_size, MPI_INT,
                            MPI_COMM_WORLD),
              "MPI_Allgather rank-to-node");
    for (int rank = 0; rank < world_size; ++rank)
    {
        const auto begin = gathered.begin() +
                           static_cast<std::ptrdiff_t>(rank) * world_size;
        if (!std::equal(rank_to_node.begin(), rank_to_node.end(), begin))
        {
            throw std::runtime_error("rank-to-node differs across MPI ranks");
        }
    }
}

inline void broadcast_secret_key(const poseidon::PoseidonContext &context,
                                 poseidon::SecretKey &secret_key, int rank)
{
    const auto mode = poseidon::compr_mode_type::none;
    std::uint64_t byte_count = 0;
    std::vector<poseidon::poseidon_byte> bytes;
    if (rank == 0)
    {
        const std::streamoff size = secret_key.save_size(mode);
        if (size <= 0 || static_cast<std::uint64_t>(size) >
                             static_cast<std::uint64_t>(
                                 std::numeric_limits<int>::max()))
        {
            throw std::runtime_error(
                "serialized secret key exceeds MPI count range");
        }
        byte_count = static_cast<std::uint64_t>(size);
        bytes.resize(static_cast<std::size_t>(byte_count));
        if (secret_key.save(bytes.data(), bytes.size(), mode) != size)
        {
            throw std::runtime_error("secret key serialization size mismatch");
        }
    }
    check_mpi(MPI_Bcast(&byte_count, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD),
              "MPI_Bcast secret key size");
    if (byte_count == 0 || byte_count > static_cast<std::uint64_t>(
                                   std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("invalid broadcast secret key size");
    }
    if (rank != 0)
    {
        bytes.resize(static_cast<std::size_t>(byte_count));
    }
    check_mpi(MPI_Bcast(bytes.data(), static_cast<int>(bytes.size()), MPI_BYTE,
                        0, MPI_COMM_WORLD),
              "MPI_Bcast secret key");
    if (rank != 0)
    {
        if (secret_key.load(context, bytes.data(), bytes.size()) !=
            static_cast<std::streamoff>(bytes.size()))
        {
            throw std::runtime_error("secret key load size mismatch");
        }
        secret_key.data().resize(context, secret_key.parms_id(),
                                 secret_key.data().coeff_count());
    }
}

} // namespace poseidon::runtime_api::test
